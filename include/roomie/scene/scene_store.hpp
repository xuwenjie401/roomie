#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "roomie/pipeline/map_checkpoint.hpp"
#include "roomie/scene/scene_snapshot.hpp"

namespace roomie {

struct SceneStoreStatus {
  bool ok = false;
  std::string error;

  explicit operator bool() const { return ok; }
  static SceneStoreStatus success() { return {true, {}}; }
  static SceneStoreStatus failure(std::string message) {
    return {false, std::move(message)};
  }
};

struct SceneStoreWatermarks {
  SceneRevision latest_scene_revision = 0;
  SceneRevision durable_scene_revision = 0;
  std::size_t pending_commits = 0;
  std::size_t max_pending_commits = 0;
  bool backpressure_required = false;
};

struct DurableTaskSpec {
  // task_id is the stable execution identity; dedupe_key is the stable intent
  // identity across retries/restarts and must also be unique.
  std::string task_id;
  std::string dedupe_key;
  std::string task_type;
  std::string payload;
  SceneRevision scene_revision = 0;
  std::int64_t not_before_unix_ms = 0;
};

// Cross-process SLO origin for the first appearance-derived artifact of a
// stable object.  The Unix origin and admission class are durable; a steady
// timestamp is intentionally process-local and is never restored.
struct DurableArtifactOrigin {
  SceneObjectId object_id = -1;
  SceneRevision scene_revision = 0;
  std::int64_t created_unix_ms = 0;
  ArtifactPriority priority = ArtifactPriority::kInteractive;
};

inline bool operator==(const DurableArtifactOrigin& lhs,
                       const DurableArtifactOrigin& rhs) {
  return lhs.object_id == rhs.object_id &&
         lhs.scene_revision == rhs.scene_revision &&
         lhs.created_unix_ms == rhs.created_unix_ms &&
         lhs.priority == rhs.priority;
}

enum class DurableTaskState {
  kPending,
  kLeased,
  kCompleted,
  kFailed,
};

struct DurableTaskRecord {
  DurableTaskSpec task;
  DurableTaskState state = DurableTaskState::kPending;
  std::string lease_owner;
  std::int64_t lease_until_unix_ms = 0;
  std::uint64_t attempts = 0;
  std::string last_error;
  std::int64_t completed_at_unix_ms = 0;
  std::string completed_by;
  std::int64_t failed_at_unix_ms = 0;
  std::string failed_by;
};

struct SceneRestoreResult {
  SceneStoreStatus status;
  bool found = false;
  SceneSnapshot snapshot;
};

struct TaskLeaseResult {
  SceneStoreStatus status;
  std::optional<DurableTaskRecord> task;
};

struct TaskLookupResult {
  SceneStoreStatus status;
  std::optional<DurableTaskRecord> task;
};

// Immutable persistence envelope for one semantic embedding. The dimension is
// stored explicitly and must exactly match vector.size(); model_id + dimension
// form the namespace, while document_hash fences stale semantic documents.
struct DurableEmbeddingRecord {
  SceneObjectId object_id = -1;
  std::string document_hash;
  std::string model_id;
  std::size_t dimension = 0;
  std::vector<float> vector;
  SceneRevision created_scene_revision = 0;
};

struct EmbeddingRecordLookupResult {
  SceneStoreStatus status;
  std::optional<DurableEmbeddingRecord> record;
};

struct EmbeddingRecordListResult {
  SceneStoreStatus status;
  std::vector<DurableEmbeddingRecord> records;
};

struct MapCheckpointLookupResult {
  SceneStoreStatus status;
  std::optional<MapCheckpointManifest> manifest;
};

struct ArtifactOriginListResult {
  SceneStoreStatus status;
  std::vector<DurableArtifactOrigin> origins;
};

// SceneStore has no background thread. enqueueCommit() advances the live
// watermark in memory; flush() atomically persists all queued revisions and
// advances the durable watermark. Destroying without gracefulFlush() drops
// only the in-memory suffix, which makes crash semantics deterministic.
class SceneStore {
 public:
  static constexpr int kCurrentSchemaVersion = 6;
  static constexpr std::size_t kDefaultMaxPendingCommits = 64;
  // A durable revision log contains one complete recovery checkpoint at this
  // cadence. Revisions between checkpoints are compact, hash-checked deltas,
  // which bounds replay work without copying every object on every commit.
  static constexpr SceneRevision kRevisionCheckpointInterval = 64;

  explicit SceneStore(
      std::string database_path,
      std::size_t max_pending_commits = kDefaultMaxPendingCommits);
  ~SceneStore();

  SceneStore(const SceneStore&) = delete;
  SceneStore& operator=(const SceneStore&) = delete;
  SceneStore(SceneStore&&) = delete;
  SceneStore& operator=(SceneStore&&) = delete;

  SceneStoreStatus open();
  bool isOpen() const;
  int schemaVersion() const;

  SceneStoreStatus enqueueCommit(
      SceneSnapshot snapshot,
      std::vector<DurableTaskSpec> outbox_tasks = {},
      std::vector<DurableArtifactOrigin> artifact_origins = {});
  SceneStoreStatus flush();
  // Atomically persists the queued semantic revisions and records that the
  // latest immutable map checkpoint is still valid for the resulting scene
  // revision. Intended for offline, map-invariant human edits.
  SceneStoreStatus flushWithMapCheckpointAlignment();
  SceneStoreStatus gracefulFlush();
  SceneStoreStatus closeGracefully();

  SceneStoreWatermarks watermarks() const;
  SceneRestoreResult restoreLatest() const;
  // Materializes an exact durable prefix without changing current state or
  // watermarks. This is used to inspect/validate a map manifest's aligned
  // scene point before coordinated startup abandons any newer semantic suffix.
  SceneRestoreResult restoreAt(SceneRevision revision) const;
  // Startup-only coordinated recovery. Atomically abandons every durable
  // scene/task/embedding/origin row newer than revision and rebuilds the
  // materialized current view from that exact prefix. This creates a new
  // contiguous branch; it is not a read-only historical query. Seed recovery
  // passes retain_aligned_map_checkpoints=false to discard the previous map
  // lineage in the same transaction, including a revision-zero manifest.
  SceneStoreStatus rewindTo(
      SceneRevision revision,
      bool retain_aligned_map_checkpoints = true);

  // Publishes only a checkpoint aligned to the store's current durable scene
  // watermark. The nvblox file must already have been atomically saved and
  // verified before this append-only manifest row is inserted.
  SceneStoreStatus publishMapCheckpoint(
      const MapCheckpointManifest& manifest);
  MapCheckpointLookupResult latestMapCheckpoint() const;
  ArtifactOriginListResult listArtifactOrigins() const;

  // Adds an intent that depends on an already-durable scene revision. The
  // operation is idempotent when task_id/dedupe_key and all payload fields are
  // bit-identical; conflicting reuse is rejected.
  SceneStoreStatus ensureTask(const DurableTaskSpec& task);
  // Atomically selects and leases one eligible task across all requested
  // types. An empty type list preserves the legacy "any task type" behavior.
  // Scheduling metadata is read compatibly from either the DAM payload's
  // top-level priority/due_unix_ms fields or embedding's artifact_slo object.
  // Missing/invalid metadata is treated as non-overdue bulk work.
  TaskLeaseResult leaseNextTask(const std::vector<std::string>& task_types,
                                const std::string& lease_owner,
                                std::int64_t now_unix_ms,
                                std::int64_t lease_duration_ms);
  // Compatibility overload for callers interested in one type. An empty
  // string selects across all task types, as it did before fair scheduling.
  TaskLeaseResult leaseNextTask(const std::string& task_type,
                                const std::string& lease_owner,
                                std::int64_t now_unix_ms,
                                std::int64_t lease_duration_ms);
  SceneStoreStatus renewTaskLease(const std::string& task_id,
                                  const std::string& lease_owner,
                                  std::uint64_t lease_attempt,
                                  std::int64_t now_unix_ms,
                                  std::int64_t lease_duration_ms);
  SceneStoreStatus completeTask(const std::string& task_id,
                                const std::string& lease_owner,
                                std::uint64_t lease_attempt,
                                std::int64_t completed_at_unix_ms);
  // Permanently fails exactly one live owner/attempt lease. Repeating the
  // acknowledgement from that same fenced attempt is idempotent; every other
  // owner/attempt and an expired lease are rejected.
  SceneStoreStatus failTask(const std::string& task_id,
                            const std::string& lease_owner,
                            std::uint64_t lease_attempt,
                            std::int64_t failed_at_unix_ms,
                            std::string error);
  SceneStoreStatus retryTask(const std::string& task_id,
                             const std::string& lease_owner,
                             std::uint64_t lease_attempt,
                             std::int64_t retry_at_unix_ms,
                             std::string error);
  TaskLookupResult lookupTask(const std::string& task_id) const;

  // Embeddings are immutable by their complete identity. Repeating an
  // identical upsert is successful; reusing that identity with different
  // canonical numeric vector content or provenance is rejected.
  SceneStoreStatus upsertEmbeddingRecord(DurableEmbeddingRecord record);
  EmbeddingRecordLookupResult getEmbeddingRecord(
      SceneObjectId object_id,
      const std::string& document_hash,
      const std::string& model_id,
      std::size_t dimension) const;
  EmbeddingRecordListResult listEmbeddingRecords() const;
  EmbeddingRecordListResult listEmbeddingRecords(
      const std::string& model_id,
      std::size_t dimension) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace roomie
