#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "roomie/artifacts/artifact_scheduler.hpp"
#include "roomie/pipeline/worker_thread.hpp"
#include "roomie/scene/scene_reducer.hpp"

namespace roomie {

struct ArtifactReducerSubmitResult {
  bool delivered = false;
  SceneApplyResult apply_result;
  std::string error;

  explicit operator bool() const { return delivered; }

  static ArtifactReducerSubmitResult success(SceneApplyResult result) {
    ArtifactReducerSubmitResult submitted;
    submitted.delivered = true;
    submitted.apply_result = std::move(result);
    return submitted;
  }

  static ArtifactReducerSubmitResult failure(std::string message) {
    ArtifactReducerSubmitResult submitted;
    submitted.error = std::move(message);
    return submitted;
  }
};

struct ArtifactRuntimeActorConfig {
  std::string lease_owner = "dam-runtime";
  std::chrono::milliseconds idle_poll_period{20};
  std::chrono::milliseconds heartbeat_period{5000};
  std::chrono::milliseconds reducer_submit_timeout{2000};
  std::chrono::milliseconds durability_wait_slice{50};
  // requestStop() is cooperative. After this bound, an uncooperative worker
  // invocation is detached with an inert heartbeat and its result discarded;
  // the durable lease is released or left recoverable by expiry.
  std::chrono::milliseconds worker_stop_timeout{250};
};

struct ArtifactRuntimeActorStats {
  std::uint64_t lease_polls = 0;
  std::uint64_t leases_acquired = 0;
  std::uint64_t worker_invocations = 0;
  std::uint64_t reducer_submissions = 0;
  std::uint64_t durability_waits = 0;
  std::uint64_t completed = 0;
  std::uint64_t retried = 0;
  std::uint64_t terminal_failures = 0;
  std::uint64_t superseded = 0;
  std::uint64_t lease_lost = 0;
  std::uint64_t scheduler_errors = 0;
  std::uint64_t reducer_errors = 0;
  std::uint64_t durability_errors = 0;
  std::uint64_t interactive_slo_completions = 0;
  std::uint64_t bulk_slo_completions = 0;
  std::uint64_t untracked_slo_completions = 0;
  std::uint64_t interactive_slo_violations = 0;
  std::uint64_t bulk_slo_violations = 0;
  std::uint64_t slo_violations = 0;
  std::uint64_t admission_deferrals = 0;
  std::uint64_t stop_requeues = 0;
  std::uint64_t detached_worker_invocations = 0;
};

// Single-in-flight DAM actor over ArtifactScheduler's durable outbox.
//
// ReliableReducerSink must return only after the reducer has produced a
// SceneApplyResult (or the supplied timeout elapsed). DurabilityWaiter must
// honor its timeout. TerminalFailureSink is an optional override for the
// durable terminal disposition; when omitted, ArtifactScheduler::fail() is
// used. Permanent failures are never converted into retry or completion.
class ArtifactRuntimeActor final : public WorkerThread {
 public:
  using SceneSnapshotSupplier = std::function<SceneSnapshot()>;
  using ReliableReducerSink = std::function<ArtifactReducerSubmitResult(
      const ApplyDescriptionArtifactCommand&,
      std::chrono::milliseconds timeout)>;
  using DurabilityWaiter = std::function<bool(
      SceneRevision revision, std::chrono::milliseconds timeout)>;
  using TerminalFailureSink = std::function<SceneStoreStatus(
      const ArtifactLease& lease,
      std::int64_t failed_at_unix_ms,
      std::string error)>;
  using UnixMillisClock = std::function<std::int64_t()>;
  // Evaluated before every durable lease acquisition. Production binds this
  // to the perception backlog/GPU quiet-window so a DAM job never starts
  // while real-time inference is queued or in flight.
  using ExecutionAdmissionGate = std::function<bool()>;

  ArtifactRuntimeActor(
      ArtifactRuntimeActorConfig config,
      ArtifactScheduler* scheduler,
      std::shared_ptr<DamWorker> worker,
      SceneSnapshotSupplier snapshot_supplier,
      ReliableReducerSink reducer_sink,
      DurabilityWaiter durability_waiter,
      TerminalFailureSink terminal_failure_sink = {},
      UnixMillisClock unix_millis_clock = {},
      ExecutionAdmissionGate execution_admission_gate = {});
  ~ArtifactRuntimeActor() override;

  bool idle() const;
  bool waitUntilIdle(std::chrono::milliseconds timeout) const;
  ArtifactRuntimeActorStats stats() const;
  std::string lastError() const;
  std::optional<std::string> activeTaskId() const;

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  struct LeaseHeartbeat;
  struct WorkerInvocation;

  static ArtifactRuntimeActorConfig validateConfig(
      ArtifactRuntimeActorConfig config);

  bool processNextLease();
  ArtifactExecutionResult executeWorker(
      const ArtifactLease& lease,
      const std::shared_ptr<LeaseHeartbeat>& heartbeat);
  bool waitUntilRevisionDurable(
      const ArtifactLease& lease,
      SceneRevision revision,
      const std::shared_ptr<LeaseHeartbeat>& heartbeat);
  bool completeLease(
      const ArtifactLease& lease,
      const std::shared_ptr<LeaseHeartbeat>& heartbeat,
      bool account_slo = false);
  bool retryLease(
      const ArtifactLease& lease,
      std::string error,
      const std::shared_ptr<LeaseHeartbeat>& heartbeat,
      bool stopping = false);
  bool failLease(
      const ArtifactLease& lease,
      std::string error,
      const std::shared_ptr<LeaseHeartbeat>& heartbeat);
  bool completeSuperseded(
      const ArtifactLease& lease,
      const SceneSnapshot& snapshot,
      DamDependencyFreshness freshness,
      const std::shared_ptr<LeaseHeartbeat>& heartbeat);

  std::int64_t nowUnixMillis() const;
  void setActiveLease(
      const ArtifactLease& lease,
      std::shared_ptr<LeaseHeartbeat> heartbeat);
  void clearActiveLease();
  void recordError(std::string error);
  void accountSloCompletion(const ArtifactLease& lease,
                            std::int64_t completed_at_unix_ms);
  void interruptiblePollWait();

  ArtifactRuntimeActorConfig config_;
  ArtifactScheduler* scheduler_ = nullptr;
  std::shared_ptr<DamWorker> worker_;
  SceneSnapshotSupplier snapshot_supplier_;
  ReliableReducerSink reducer_sink_;
  DurabilityWaiter durability_waiter_;
  TerminalFailureSink terminal_failure_sink_;
  UnixMillisClock unix_millis_clock_;
  ExecutionAdmissionGate execution_admission_gate_;

  mutable std::mutex state_mutex_;
  mutable std::condition_variable state_cv_;
  std::string last_error_;
  std::optional<std::string> active_task_id_;
  std::shared_ptr<LeaseHeartbeat> active_heartbeat_;
  bool idle_ = true;

  std::atomic_uint64_t lease_polls_{0};
  std::atomic_uint64_t leases_acquired_{0};
  std::atomic_uint64_t worker_invocations_{0};
  std::atomic_uint64_t reducer_submissions_{0};
  std::atomic_uint64_t durability_waits_{0};
  std::atomic_uint64_t completed_{0};
  std::atomic_uint64_t retried_{0};
  std::atomic_uint64_t terminal_failures_{0};
  std::atomic_uint64_t superseded_{0};
  std::atomic_uint64_t lease_lost_{0};
  std::atomic_uint64_t scheduler_errors_{0};
  std::atomic_uint64_t reducer_errors_{0};
  std::atomic_uint64_t durability_errors_{0};
  std::atomic_uint64_t interactive_slo_completions_{0};
  std::atomic_uint64_t bulk_slo_completions_{0};
  std::atomic_uint64_t untracked_slo_completions_{0};
  std::atomic_uint64_t interactive_slo_violations_{0};
  std::atomic_uint64_t bulk_slo_violations_{0};
  std::atomic_uint64_t slo_violations_{0};
  std::atomic_uint64_t admission_deferrals_{0};
  std::atomic_uint64_t stop_requeues_{0};
  std::atomic_uint64_t detached_worker_invocations_{0};
};

}  // namespace roomie
