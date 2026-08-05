#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "roomie/scene/scene_command.hpp"
#include "roomie/scene/scene_snapshot.hpp"
#include "roomie/scene/scene_store.hpp"

namespace roomie {

const char* artifactPriorityName(ArtifactPriority priority);

// This is the complete stable identity of a DAM invocation. In particular,
// raw OBB/geometry revisions are deliberately absent: an unchanged visual
// evidence set must not be invalidated by ordinary geometry fusion.
struct DamTaskKey {
  SceneObjectId object_id = -1;
  std::uint64_t identity_revision = 0;
  std::uint64_t appearance_revision = 0;
  std::string snapshot_set_hash;
  std::string model_id;
  std::string prompt_hash;
  std::string output_schema_version;
};

bool operator==(const DamTaskKey& lhs, const DamTaskKey& rhs);
bool operator!=(const DamTaskKey& lhs, const DamTaskKey& rhs);

// Collision-free length-prefixed encoding used as both the durable task id
// and dedupe key. Every field in DamTaskKey is present in clear form.
std::string canonicalDamTaskKey(const DamTaskKey& key);

struct DamTaskIntent {
  DamTaskKey key;
  // DAM execution is appearance-derived. The durable request normalizes
  // semantic_revision to zero (wildcard) and ignores obb_revision so ordinary
  // observation fusion cannot supersede unchanged visual evidence.
  ObjectDependency dependency;
  SceneRevision scene_revision = 0;
  ArtifactPriority priority = ArtifactPriority::kBulk;
  std::int64_t created_unix_ms = 0;
  // Zero selects the configured 10 s interactive / 120 s bulk target.
  std::int64_t due_unix_ms = 0;
  // Process-local monotonic counterpart. It is persisted with an epoch fence
  // so a restarted process can safely ignore it and use the Unix deadline.
  RunId steady_clock_epoch;
  std::int64_t origin_steady_ns = 0;
  // Zero selects the same configured duration as due_unix_ms.
  std::int64_t due_steady_ns = 0;
  // Opaque, durable worker input (normally asset ids/crop/mask references).
  std::string input_payload;
};

struct DamTaskRequest {
  DamTaskKey key;
  ObjectDependency dependency;
  SceneRevision scene_revision = 0;
  ArtifactPriority priority = ArtifactPriority::kBulk;
  std::int64_t created_unix_ms = 0;
  std::int64_t due_unix_ms = 0;
  RunId steady_clock_epoch;
  std::int64_t origin_steady_ns = 0;
  std::int64_t due_steady_ns = 0;
  std::string input_payload;
};

struct ArtifactSchedulerConfig {
  std::int64_t interactive_due_ms = 10'000;
  std::int64_t bulk_due_ms = 120'000;
  std::int64_t lease_duration_ms = 30'000;
  std::int64_t retry_initial_backoff_ms = 1'000;
  std::int64_t retry_max_backoff_ms = 60'000;
};

struct ArtifactLease {
  DurableTaskRecord durable;
  DamTaskRequest request;

  std::uint64_t attempt() const { return durable.attempts; }
  const std::string& taskId() const { return durable.task.task_id; }
  const std::string& owner() const { return durable.lease_owner; }
};

struct ArtifactLeaseResult {
  SceneStoreStatus status;
  std::optional<ArtifactLease> lease;
};

struct ArtifactRetryResult {
  SceneStoreStatus status;
  std::int64_t retry_at_unix_ms = 0;
};

enum class DamArtifactParsePath : std::uint8_t {
  kSchemaValid = 0,
  kSchemaRepaired = 1,
  kUnstructuredFallback = 2,
};

struct DamVisualAttributes {
  std::vector<std::string> colors;
  std::vector<std::string> materials;
  std::vector<std::string> shape;
  std::vector<std::string> visible_parts;
  std::vector<std::string> state_or_pose;
  std::vector<std::string> distinctive_marks;
  std::vector<std::string> visible_text;
};

// The raw model output is always retained. normalized_json is populated only
// for a schema-valid value. durable_envelope_json records the parse path,
// schema error, raw text, and (when valid) the normalized structured value.
struct DamArtifact {
  DamArtifactParsePath parse_path =
      DamArtifactParsePath::kUnstructuredFallback;
  std::string canonical_name;
  std::string short_description;
  std::string retrieval_text;
  DamVisualAttributes visual_attributes;
  std::vector<std::string> uncertain_or_not_visible;
  double confidence = 0.0;
  std::vector<std::string> evidence_snapshot_ids;
  std::string mask_source;
  std::string raw_text;
  std::string normalized_json;
  std::string durable_envelope_json;
  std::string schema_error;

  bool structured() const {
    return parse_path != DamArtifactParsePath::kUnstructuredFallback;
  }
};

using DamRepairFunction =
    std::function<std::optional<std::string>(const std::string& raw_text,
                                             const std::string& schema_error)>;

// Pure parsing/validation. A deterministic fenced-JSON repair is attempted
// first, followed by at most one caller-supplied repair. Failure produces an
// explicitly tagged unstructured fallback; it never claims schema validity.
DamArtifact parseDamArtifact(const std::string& raw_text,
                             const std::string& output_schema_version,
                             DamRepairFunction repair = {});

using DamLeaseHeartbeat = std::function<bool(std::int64_t now_unix_ms)>;

struct DamWorkerResponse {
  bool success = false;
  bool retryable = true;
  std::string raw_output;
  std::string error;
};

class DamWorker {
 public:
  virtual ~DamWorker() = default;
  virtual DamWorkerResponse describe(const DamTaskRequest& request,
                                     const DamLeaseHeartbeat& heartbeat) = 0;
};

enum class ArtifactExecutionStatus : std::uint8_t {
  kReadyToApply = 0,
  kRetryableFailure = 1,
  kPermanentFailure = 2,
  kInvalidLease = 3,
};

struct ArtifactExecutionResult {
  ArtifactExecutionStatus status = ArtifactExecutionStatus::kInvalidLease;
  std::optional<ApplyDescriptionArtifactCommand> command;
  std::optional<DamArtifact> artifact;
  bool slo_violation = false;
  std::string error;
};

enum class DamDependencyFreshness : std::uint8_t {
  kCurrent = 0,
  kRetiredAlias = 1,
  kTombstoned = 2,
  kMissingObject = 3,
  kIdentityStale = 4,
  kAppearanceStale = 5,
  kSnapshotSetStale = 6,
  kSemanticStale = 7,
  kInvalid = 8,
};

DamDependencyFreshness inspectDamDependency(const SceneSnapshot& snapshot,
                                             const DamTaskRequest& request);

// Thin orchestration layer over SceneStore's durable outbox. It has no private
// in-memory task queue, so process restart preserves lease/retry state.
class ArtifactScheduler {
 public:
  static constexpr const char* kInteractiveTaskType = "dam.interactive.v1";
  static constexpr const char* kBulkTaskType = "dam.bulk.v1";

  explicit ArtifactScheduler(SceneStore* store,
                             ArtifactSchedulerConfig config = {});

  DurableTaskSpec makeTaskSpec(const DamTaskIntent& intent,
                               SceneStoreStatus* status = nullptr) const;
  SceneStoreStatus ensureTask(const DamTaskIntent& intent) const;

  // SceneStore atomically chooses across interactive and bulk work. Overdue
  // work is promoted first, then interactive beats bulk, followed by due time
  // and stable revision/task-id ordering. Expired leases remain eligible.
  ArtifactLeaseResult leaseNext(const std::string& lease_owner,
                                std::int64_t now_unix_ms) const;

  SceneStoreStatus heartbeat(const ArtifactLease& lease,
                             std::int64_t now_unix_ms) const;
  SceneStoreStatus complete(const ArtifactLease& lease,
                            std::int64_t completed_at_unix_ms) const;
  SceneStoreStatus fail(const ArtifactLease& lease,
                        std::int64_t failed_at_unix_ms,
                        std::string error) const;
  ArtifactRetryResult retry(const ArtifactLease& lease,
                            std::int64_t now_unix_ms,
                            std::string error) const;

  // execute() does not complete the durable task. The caller must reliably
  // enqueue/apply result.command and then acknowledge with complete(). If the
  // process crashes between those operations, the task runs again; reducer CAS
  // makes the repeated effect idempotent.
  ArtifactExecutionResult execute(const ArtifactLease& lease,
                                  DamWorker* worker,
                                  std::int64_t started_unix_ms,
                                  DamRepairFunction repair = {}) const;

 private:
  SceneStore* store_ = nullptr;
  ArtifactSchedulerConfig config_;
};

}  // namespace roomie
