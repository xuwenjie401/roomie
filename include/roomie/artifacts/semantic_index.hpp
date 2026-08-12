#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"
#include "roomie/scene/scene_state.hpp"

namespace roomie {

using SemanticIndexGeneration = std::uint64_t;

// Shared lowercase SHA-256 for content/provenance identities. Callers must
// canonicalize structured input before hashing.
std::string stableSha256Hex(std::string_view value);

// Structured DAM fields that are useful for retrieval. Map iteration and
// value ordering are canonicalized by makeSemanticDocument(), so semantically
// identical worker output produces the same document hash.
struct SemanticRetrievalFields {
  std::string canonical_name;
  std::string short_description;
  std::string retrieval_text;
  std::map<std::string, std::vector<std::string>> attributes;
};

// Query filters deliberately live outside the embedded document. Ordinary
// motion, room reassignment, or lifecycle changes therefore do not invalidate
// an otherwise current embedding.
struct SemanticMetadata {
  std::string room_id;
  std::array<float, 3> position_world = {0.0f, 0.0f, 0.0f};
  bool has_position = false;
  bool active = true;
  std::map<std::string, std::string> filters;
};

struct SemanticDocumentInput {
  SceneObjectId object_id = -1;
  std::string name;
  std::string label;
  SemanticRetrievalFields retrieval;
  std::vector<std::string> human_tags;
  SemanticMetadata metadata;
  std::uint64_t semantic_revision = 0;
  SceneRevision created_scene_revision = 0;
};

struct SemanticDocument {
  SceneObjectId object_id = -1;
  // Stable, field-delimited representation used by the encoder.
  std::string text;
  std::string document_hash;
  SemanticMetadata metadata;
  std::uint64_t semantic_revision = 0;
  SceneRevision created_scene_revision = 0;
};

// Produces a normalized semantic-document.v1 payload and a lowercase SHA-256
// hash. Empty/duplicate tags and attribute values are removed deterministically.
SemanticDocument makeSemanticDocument(const SemanticDocumentInput& input);

// Builds the canonical semantic document for a reducer-owned scene object.
// Artifact producers and query freshness checks must use this same path so a
// current embedding can never be reported stale because of formatter drift.
enum class SemanticDocumentView : std::uint8_t {
  // Query/index projection may expose only reducer-current semantic state.
  kCurrent = 0,
  // Durable embedding intents are created in the same transaction as a DAM
  // result while that result is still pending reducer durability promotion.
  kPendingDescriptionIfPresent = 1,
};

SemanticDocument makeSemanticDocumentForObject(
    const SceneObject& object,
    SceneObjectId object_id,
    SceneRevision scene_revision,
    SemanticDocumentView view = SemanticDocumentView::kCurrent);

struct EmbeddingNamespace {
  std::string model_id;
  std::size_t dimension = 0;
};

bool operator==(const EmbeddingNamespace& lhs,
                const EmbeddingNamespace& rhs);
bool operator!=(const EmbeddingNamespace& lhs,
                const EmbeddingNamespace& rhs);
bool operator<(const EmbeddingNamespace& lhs,
               const EmbeddingNamespace& rhs);

struct EmbeddingJob {
  SceneObjectId object_id = -1;
  std::string document;
  std::string document_hash;
  EmbeddingNamespace name_space;
  std::uint64_t semantic_revision = 0;
  SceneRevision created_scene_revision = 0;
};

struct EmbeddingRecord {
  SceneObjectId object_id = -1;
  std::string document_hash;
  std::string model_id;
  std::vector<float> vector;
  SceneRevision created_scene_revision = 0;
};

// Encoders are owned by a long-lived worker. Neither VersionedSemanticIndex
// nor search() has an encoder reference, which makes cold-start work in a tool
// call structurally impossible.
class EmbeddingEncoder {
 public:
  virtual ~EmbeddingEncoder() = default;
  virtual std::string modelId() const = 0;
  virtual std::size_t dimension() const = 0;
  virtual void prewarm() = 0;
  virtual std::vector<std::vector<float>> encodeBatch(
      const std::vector<std::string>& documents) = 0;
};

struct EmbeddingWorkerConfig {
  std::size_t pending_capacity = 128;
  std::size_t maximum_batch_size = 16;
  std::chrono::milliseconds batch_window{20};
};

struct EmbeddingWorkerStats {
  bool warmed = false;
  bool warmup_failed = false;
  std::uint64_t submitted = 0;
  std::uint64_t replaced = 0;
  std::uint64_t encoded_documents = 0;
  std::uint64_t superseded_before_encode = 0;
  std::uint64_t failed_documents = 0;
  std::uint64_t batches = 0;
  std::string last_error;
  ChannelStats channel;
};

using EmbeddingRecordSink = std::function<void(EmbeddingRecord)>;

// Prewarms at thread start and compacts a short collection window into one
// encodeBatch() call. Pending jobs use LatestByObject; an already executing
// stale job is harmless because VersionedSemanticIndex fences its result with
// (object_id, document_hash, model_id, dimension).
class EmbeddingWorker final : public WorkerThread {
 public:
  EmbeddingWorker(std::shared_ptr<EmbeddingEncoder> encoder,
                  EmbeddingRecordSink sink,
                  EmbeddingWorkerConfig config = {});
  ~EmbeddingWorker() override;

  PushResult<EmbeddingJob> submit(EmbeddingJob job);
  bool waitUntilWarmed(std::chrono::milliseconds timeout);
  bool waitUntilIdle(std::chrono::milliseconds timeout);
  EmbeddingWorkerStats stats() const;

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  void finishConsumed(std::size_t count);

  std::shared_ptr<EmbeddingEncoder> encoder_;
  EmbeddingRecordSink sink_;
  EmbeddingWorkerConfig config_;
  BoundedChannel<EmbeddingJob> pending_;

  mutable std::mutex state_mutex_;
  std::condition_variable state_cv_;
  std::size_t outstanding_ = 0;
  EmbeddingWorkerStats stats_;
};

struct SemanticIndexConfig {
  EmbeddingNamespace initial_namespace{"semantic.default.v1", 384};
  std::int64_t default_read_ttl_ms = 30'000;
  // Diagnostic/recovery history is recent-only. Current records for every
  // live object and namespace remain independently retained in records.
  std::size_t record_history_capacity = 1024;
  // Diagnostic/test hook called on the builder thread after a new immutable
  // matrix is assembled but before it is atomically published.
  std::function<void()> before_generation_publish;
};

struct SemanticGenerationInfo {
  SemanticIndexGeneration generation = 0;
  std::uint64_t lexical_delta_revision = 0;
  std::string model_id;
  std::size_t dimension = 0;
  std::size_t rows = 0;
  SceneRevision created_through_scene_revision = 0;
};

// Opaque immutable read view retained by SemanticReadToken.
struct SemanticIndexReadView;

class SemanticReadToken {
 public:
  SemanticReadToken() = default;

  SemanticGenerationInfo info() const { return info_; }
  SceneRevision sceneRevision() const { return scene_revision_; }
  std::int64_t expiresAtUnixMs() const { return expires_at_unix_ms_; }
  bool expired(std::int64_t now_unix_ms) const {
    return !view_ || now_unix_ms >= expires_at_unix_ms_;
  }

 private:
  friend class VersionedSemanticIndex;

  SemanticGenerationInfo info_;
  SceneRevision scene_revision_ = 0;
  std::int64_t expires_at_unix_ms_ = 0;
  std::shared_ptr<const SemanticIndexReadView> view_;
};

enum class EmbeddingAcceptStatus : std::uint8_t {
  kAccepted = 0,
  kRejectedObjectMissing = 1,
  kRejectedDocumentStale = 2,
  kRejectedNamespace = 3,
  kRejectedDimension = 4,
  kRejectedInvalidVector = 5,
};

struct SemanticUpsertResult {
  bool document_changed = false;
  bool metadata_changed = false;
  // Normally one job. During an explicit model upgrade the changed object is
  // encoded once in each live namespace so neither namespace becomes mixed.
  std::vector<EmbeddingJob> jobs;
};

enum class SemanticFreshness : std::uint8_t {
  kCurrentEmbedding = 0,
  kEmbeddingPendingLexical = 1,
};

struct SemanticSearchFilter {
  std::optional<std::string> room_id;
  std::optional<bool> active;
  std::map<std::string, std::string> exact_metadata;
};

struct SemanticSearchRequest {
  std::string query_text;
  // Optional already-encoded query. Search never constructs or invokes an
  // encoder. When present its dimension must match the token namespace.
  std::vector<float> query_vector;
  std::size_t top_k = 10;
  SemanticSearchFilter filter;
};

enum class SemanticSearchStatus : std::uint8_t {
  kOk = 0,
  kExpiredToken = 1,
  kInvalidQueryDimension = 2,
  kInvalidQueryVector = 3,
};

struct SemanticSearchHit {
  SceneObjectId object_id = -1;
  std::string document_hash;
  std::string document;
  SemanticMetadata metadata;
  SceneRevision created_scene_revision = 0;
  float score = 0.0f;
  float vector_score = 0.0f;
  float lexical_score = 0.0f;
  SemanticFreshness freshness = SemanticFreshness::kCurrentEmbedding;
};

struct SemanticSearchResponse {
  SemanticSearchStatus status = SemanticSearchStatus::kOk;
  SemanticGenerationInfo index;
  SceneRevision scene_revision = 0;
  std::size_t pending_embeddings = 0;
  std::vector<SemanticSearchHit> hits;
};

// In-memory versioned semantic index core. EmbeddingRecord is also the exact
// persistence envelope expected by a future SceneStore adapter. Matrix builds
// run on a private background thread, use normalized contiguous row-major
// storage, and publish one immutable read view with atomic shared_ptr swap.
class VersionedSemanticIndex {
 public:
  explicit VersionedSemanticIndex(SemanticIndexConfig config = {});
  ~VersionedSemanticIndex();

  VersionedSemanticIndex(const VersionedSemanticIndex&) = delete;
  VersionedSemanticIndex& operator=(const VersionedSemanticIndex&) = delete;

  SemanticUpsertResult upsertDocument(SemanticDocument document);
  bool eraseObject(SceneObjectId object_id);
  bool mergeObjects(SceneObjectId retired_object_id,
                    SceneObjectId canonical_object_id);

  // Begins a parallel namespace build. The active namespace remains readable
  // until every current document has a matching record in the new model and a
  // complete generation is atomically published.
  std::vector<EmbeddingJob> beginModelUpgrade(EmbeddingNamespace name_space);

  EmbeddingAcceptStatus acceptEmbedding(EmbeddingRecord record);
  std::vector<EmbeddingRecord> embeddingRecords() const;

  SemanticReadToken pinRead(SceneRevision scene_revision,
                            std::int64_t now_unix_ms,
                            std::int64_t ttl_ms = 0) const;
  SemanticSearchResponse search(const SemanticReadToken& token,
                                const SemanticSearchRequest& request,
                                std::int64_t now_unix_ms) const;

  // Returns the immutable row hash from the generation pinned by token. This
  // deliberately does not consult the latest document map: query gateways use
  // it to detect a stale row without breaking an old generation lease.
  std::optional<std::string> indexedDocumentHash(
      const SemanticReadToken& token,
      SceneObjectId object_id) const;

  SemanticGenerationInfo activeGeneration() const;
  std::size_t pendingEmbeddingCount() const;
  bool waitForBuildIdle(std::chrono::milliseconds timeout) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace roomie
