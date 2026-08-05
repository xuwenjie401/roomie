#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "roomie/artifacts/semantic_index.hpp"
#include "roomie/scene/scene_snapshot.hpp"
#include "roomie/scene/scene_store.hpp"

namespace roomie {

struct EmbeddingTaskRequest {
  SceneRevision owning_scene_revision = 0;
  SceneObjectId object_id = -1;
  std::string document;
  std::string document_hash;
  EmbeddingNamespace name_space;
  std::uint64_t semantic_revision = 0;
  SceneRevision created_scene_revision = 0;
  std::uint64_t identity_revision = 0;
  std::uint64_t artifact_revision = 0;
  std::optional<std::uint64_t> annotation_revision;
  std::string description_input_hash;
  // Optional for legacy/manual embeddings. When tracked, this is copied from
  // the appearance-derived DAM request and is never part of document_hash.
  ArtifactSloContext artifact_slo;
};

struct EmbeddingTaskSchedulerConfig {
  std::int64_t lease_duration_ms = 30'000;
  std::int64_t retry_initial_backoff_ms = 1'000;
  std::int64_t retry_max_backoff_ms = 60'000;
};

// A malformed durable payload still returns a lease. This lets the runtime
// send the poison task to its fenced terminal-failure seam instead of leaving
// it in an expiry/re-lease loop.
struct EmbeddingTaskLease {
  DurableTaskRecord durable;
  std::optional<EmbeddingTaskRequest> request;
  std::string validation_error;

  bool valid() const { return request.has_value(); }
  std::uint64_t attempt() const { return durable.attempts; }
  const std::string& taskId() const { return durable.task.task_id; }
  const std::string& owner() const { return durable.lease_owner; }
};

struct EmbeddingTaskLeaseResult {
  SceneStoreStatus status;
  std::optional<EmbeddingTaskLease> lease;
};

struct EmbeddingTaskRetryResult {
  SceneStoreStatus status;
  std::int64_t retry_at_unix_ms = 0;
};

enum class EmbeddingDependencyFreshness : std::uint8_t {
  kCurrent = 0,
  kRetiredAlias = 1,
  kTombstoned = 2,
  kMissingObject = 3,
  kIdentityStale = 4,
  kDocumentStale = 5,
  kInvalid = 6,
};

// When current_document is non-null and the dependency is current, it receives
// the exact shared semantic document used for both payload verification and
// VersionedSemanticIndex upsert.
EmbeddingDependencyFreshness inspectEmbeddingDependency(
    const SceneSnapshot& snapshot,
    const EmbeddingTaskRequest& request,
    SemanticDocument* current_document = nullptr);

class EmbeddingTaskScheduler {
 public:
  static constexpr const char* kTaskType = "embedding.document.v1";

  explicit EmbeddingTaskScheduler(
      SceneStore* store,
      EmbeddingTaskSchedulerConfig config = {});

  DurableTaskSpec makeTaskSpec(
      const EmbeddingTaskRequest& request,
      std::int64_t not_before_unix_ms,
      std::string task_type = kTaskType,
      SceneStoreStatus* status = nullptr) const;

  // Converts a projector-reported missing job into the same strict durable
  // envelope used by reducer DescriptionCommitted intents. The owning scene
  // revision must already be durable; SceneStore remains the authoritative
  // idempotency fence across restart.
  SceneStoreStatus ensureTask(const EmbeddingJob& job,
                              const SceneSnapshot& snapshot,
                              std::int64_t not_before_unix_ms) const;

  EmbeddingTaskLeaseResult leaseNext(
      const std::string& lease_owner,
      std::int64_t now_unix_ms) const;
  SceneStoreStatus heartbeat(const EmbeddingTaskLease& lease,
                             std::int64_t now_unix_ms) const;
  SceneStoreStatus complete(const EmbeddingTaskLease& lease,
                            std::int64_t completed_at_unix_ms) const;
  EmbeddingTaskRetryResult retry(const EmbeddingTaskLease& lease,
                                 std::int64_t now_unix_ms,
                                 std::string error) const;

 private:
  SceneStore* store_ = nullptr;
  EmbeddingTaskSchedulerConfig config_;
};

}  // namespace roomie
