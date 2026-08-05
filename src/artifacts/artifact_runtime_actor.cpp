#include "roomie/artifacts/artifact_runtime_actor.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

#include "roomie/artifacts/artifact_slo_clock.hpp"
#include "roomie/utils/run_logger.hpp"

namespace roomie {
namespace {

using Milliseconds = std::chrono::milliseconds;

std::int64_t systemUnixMillis() {
  return std::chrono::duration_cast<Milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

const char* freshnessName(DamDependencyFreshness freshness) {
  switch (freshness) {
    case DamDependencyFreshness::kCurrent:
      return "current";
    case DamDependencyFreshness::kRetiredAlias:
      return "retired_alias";
    case DamDependencyFreshness::kTombstoned:
      return "tombstoned";
    case DamDependencyFreshness::kMissingObject:
      return "missing_object";
    case DamDependencyFreshness::kIdentityStale:
      return "identity_stale";
    case DamDependencyFreshness::kAppearanceStale:
      return "appearance_stale";
    case DamDependencyFreshness::kSnapshotSetStale:
      return "snapshot_set_stale";
    case DamDependencyFreshness::kSemanticStale:
      return "semantic_stale";
    case DamDependencyFreshness::kInvalid:
      return "invalid";
  }
  return "unknown";
}

class ReplayDamWorker final : public DamWorker {
 public:
  explicit ReplayDamWorker(DamWorkerResponse response)
      : response_(std::move(response)) {}

  DamWorkerResponse describe(const DamTaskRequest&,
                             const DamLeaseHeartbeat&) override {
    return std::move(response_);
  }

 private:
  DamWorkerResponse response_;
};

}  // namespace

struct ArtifactRuntimeActor::LeaseHeartbeat {
  LeaseHeartbeat(ArtifactScheduler* scheduler,
                 ArtifactLease lease,
                 std::int64_t leased_at_unix_ms,
                 Milliseconds configured_period,
                 UnixMillisClock clock)
      : scheduler(scheduler),
        lease(std::move(lease)),
        clock(std::move(clock)),
        lease_until_unix_ms(this->lease.durable.lease_until_unix_ms) {
    lease_duration_ms = lease_until_unix_ms - leased_at_unix_ms;
    if (lease_duration_ms <= 0) {
      lost.store(true, std::memory_order_release);
      active.store(false, std::memory_order_release);
      error = "leased task was already expired";
      return;
    }
    const std::int64_t third = std::max<std::int64_t>(1, lease_duration_ms / 3);
    period = std::min(configured_period, Milliseconds(third));
  }

  ~LeaseHeartbeat() { stop(); }

  void start() {
    std::lock_guard<std::mutex> stop_lock(stop_mutex);
    if (!active.load(std::memory_order_acquire) || watchdog.joinable()) {
      return;
    }
    watchdog = std::thread([this]() {
      std::unique_lock<std::mutex> wait_lock(wait_mutex);
      while (active.load(std::memory_order_acquire)) {
        if (wait_cv.wait_for(wait_lock, period, [this]() {
              return !active.load(std::memory_order_acquire);
            })) {
          break;
        }
        wait_lock.unlock();
        if (!beatNow()) {
          wait_lock.lock();
          break;
        }
        wait_lock.lock();
      }
    });
  }

  bool beatNow() {
    std::lock_guard<std::mutex> beat_lock(beat_mutex);
    if (!active.load(std::memory_order_acquire) ||
        lost.load(std::memory_order_acquire)) {
      return false;
    }

    std::int64_t now = 0;
    try {
      now = clock();
    } catch (const std::exception& exception) {
      markLost(std::string("artifact clock failed during heartbeat: ") +
               exception.what());
      return false;
    } catch (...) {
      markLost("artifact clock failed during heartbeat");
      return false;
    }
    if (now >= lease_until_unix_ms) {
      markLost("artifact lease expired before heartbeat");
      return false;
    }

    const SceneStoreStatus renewed = scheduler->heartbeat(lease, now);
    if (!renewed) {
      markLost("artifact lease heartbeat failed: " + renewed.error);
      return false;
    }
    if (now > std::numeric_limits<std::int64_t>::max() - lease_duration_ms) {
      markLost("artifact lease heartbeat time overflowed");
      return false;
    }
    lease_until_unix_ms = now + lease_duration_ms;
    return true;
  }

  bool healthy() const {
    return active.load(std::memory_order_acquire) &&
           !lost.load(std::memory_order_acquire);
  }

  bool leaseLost() const { return lost.load(std::memory_order_acquire); }

  std::string lastError() const {
    std::lock_guard<std::mutex> lock(error_mutex);
    return error;
  }

  void stop() {
    std::lock_guard<std::mutex> stop_lock(stop_mutex);
    active.store(false, std::memory_order_release);
    wait_cv.notify_all();
    if (watchdog.joinable()) {
      if (watchdog.get_id() == std::this_thread::get_id()) {
        watchdog.detach();
      } else {
        watchdog.join();
      }
    }
    // Synchronize with a worker-originated beatNow() before the scheduler and
    // its SceneStore owner are allowed to disappear.
    std::lock_guard<std::mutex> beat_lock(beat_mutex);
  }

 private:
  void markLost(std::string message) {
    {
      std::lock_guard<std::mutex> lock(error_mutex);
      error = std::move(message);
    }
    lost.store(true, std::memory_order_release);
    active.store(false, std::memory_order_release);
    wait_cv.notify_all();
  }

  ArtifactScheduler* scheduler = nullptr;
  ArtifactLease lease;
  UnixMillisClock clock;
  std::int64_t lease_duration_ms = 0;
  std::int64_t lease_until_unix_ms = 0;
  Milliseconds period{1};

  std::atomic_bool active{true};
  std::atomic_bool lost{false};
  mutable std::mutex error_mutex;
  std::string error;
  std::mutex beat_mutex;
  std::mutex stop_mutex;
  std::mutex wait_mutex;
  std::condition_variable wait_cv;
  std::thread watchdog;
};

struct ArtifactRuntimeActor::WorkerInvocation {
  std::mutex mutex;
  std::condition_variable cv;
  bool done = false;
  DamWorkerResponse response;
};

ArtifactRuntimeActorConfig ArtifactRuntimeActor::validateConfig(
    ArtifactRuntimeActorConfig config) {
  if (config.lease_owner.empty()) {
    throw std::invalid_argument(
        "ArtifactRuntimeActor lease owner must be non-empty");
  }
  if (config.idle_poll_period <= Milliseconds::zero() ||
      config.heartbeat_period <= Milliseconds::zero() ||
      config.reducer_submit_timeout <= Milliseconds::zero() ||
      config.durability_wait_slice <= Milliseconds::zero() ||
      config.worker_stop_timeout < Milliseconds::zero()) {
    throw std::invalid_argument(
        "ArtifactRuntimeActor durations must be positive");
  }
  return config;
}

ArtifactRuntimeActor::ArtifactRuntimeActor(
    ArtifactRuntimeActorConfig config,
    ArtifactScheduler* scheduler,
    std::shared_ptr<DamWorker> worker,
    SceneSnapshotSupplier snapshot_supplier,
    ReliableReducerSink reducer_sink,
    DurabilityWaiter durability_waiter,
    TerminalFailureSink terminal_failure_sink,
    UnixMillisClock unix_millis_clock,
    ExecutionAdmissionGate execution_admission_gate)
    : WorkerThread("artifact_runtime_actor"),
      config_(validateConfig(std::move(config))),
      scheduler_(scheduler),
      worker_(std::move(worker)),
      snapshot_supplier_(std::move(snapshot_supplier)),
      reducer_sink_(std::move(reducer_sink)),
      durability_waiter_(std::move(durability_waiter)),
      terminal_failure_sink_(std::move(terminal_failure_sink)),
      unix_millis_clock_(std::move(unix_millis_clock)),
      execution_admission_gate_(std::move(execution_admission_gate)) {
  if (!scheduler_ || !worker_ || !snapshot_supplier_ || !reducer_sink_ ||
      !durability_waiter_) {
    throw std::invalid_argument(
        "ArtifactRuntimeActor requires scheduler, worker, scene supplier, "
        "reducer sink, and durability waiter");
  }
  if (!unix_millis_clock_) {
    unix_millis_clock_ = systemUnixMillis;
  }
}

ArtifactRuntimeActor::~ArtifactRuntimeActor() { stop(); }

bool ArtifactRuntimeActor::idle() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return idle_;
}

bool ArtifactRuntimeActor::waitUntilIdle(Milliseconds timeout) const {
  if (timeout < Milliseconds::zero()) {
    timeout = Milliseconds::zero();
  }
  std::unique_lock<std::mutex> lock(state_mutex_);
  return state_cv_.wait_for(lock, timeout, [this]() { return idle_; });
}

ArtifactRuntimeActorStats ArtifactRuntimeActor::stats() const {
  ArtifactRuntimeActorStats result;
  result.lease_polls = lease_polls_.load();
  result.leases_acquired = leases_acquired_.load();
  result.worker_invocations = worker_invocations_.load();
  result.reducer_submissions = reducer_submissions_.load();
  result.durability_waits = durability_waits_.load();
  result.completed = completed_.load();
  result.retried = retried_.load();
  result.terminal_failures = terminal_failures_.load();
  result.superseded = superseded_.load();
  result.lease_lost = lease_lost_.load();
  result.scheduler_errors = scheduler_errors_.load();
  result.reducer_errors = reducer_errors_.load();
  result.durability_errors = durability_errors_.load();
  result.interactive_slo_completions =
      interactive_slo_completions_.load();
  result.bulk_slo_completions = bulk_slo_completions_.load();
  result.untracked_slo_completions = untracked_slo_completions_.load();
  result.interactive_slo_violations =
      interactive_slo_violations_.load();
  result.bulk_slo_violations = bulk_slo_violations_.load();
  result.slo_violations = slo_violations_.load();
  result.admission_deferrals = admission_deferrals_.load();
  result.stop_requeues = stop_requeues_.load();
  result.detached_worker_invocations =
      detached_worker_invocations_.load();
  return result;
}

std::string ArtifactRuntimeActor::lastError() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return last_error_;
}

std::optional<std::string> ArtifactRuntimeActor::activeTaskId() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return active_task_id_;
}

std::int64_t ArtifactRuntimeActor::nowUnixMillis() const {
  return unix_millis_clock_();
}

void ArtifactRuntimeActor::recordError(std::string error) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  last_error_ = std::move(error);
}

void ArtifactRuntimeActor::accountSloCompletion(
    const ArtifactLease& lease,
    std::int64_t completed_at_unix_ms) {
  const ArtifactSloContext slo{lease.request.created_unix_ms,
                               lease.request.due_unix_ms,
                               lease.request.priority,
                               lease.request.steady_clock_epoch,
                               lease.request.origin_steady_ns,
                               lease.request.due_steady_ns};
  if (!slo.tracked() || !slo.valid()) {
    ++untracked_slo_completions_;
    return;
  }
  if (slo.priority == ArtifactPriority::kInteractive) {
    ++interactive_slo_completions_;
  } else {
    ++bulk_slo_completions_;
  }
  const bool monotonic = hasCurrentArtifactSteadyClock(slo);
  const std::int64_t completed_at_steady_ns =
      monotonic ? artifactSteadyNowNanoseconds() : 0;
  const bool violated = monotonic
                            ? completed_at_steady_ns > slo.due_steady_ns
                            : completed_at_unix_ms > slo.due_unix_ms;
  if (violated) {
    ++slo_violations_;
    if (slo.priority == ArtifactPriority::kInteractive) {
      ++interactive_slo_violations_;
    } else {
      ++bulk_slo_violations_;
    }
  }
  RunLogger::logGlobal(
      "artifact_slo",
      "stage=dam_durable task_id=" + lease.taskId() +
          " object_id=" + std::to_string(lease.request.key.object_id) +
          " priority=" + artifactPriorityName(slo.priority) +
          " origin_created_unix_ms=" +
          std::to_string(slo.origin_created_unix_ms) +
          " due_unix_ms=" + std::to_string(slo.due_unix_ms) +
          " completed_at_unix_ms=" +
          std::to_string(completed_at_unix_ms) +
          " measurement_clock=" +
          std::string(monotonic ? "steady" : "unix_fallback") +
          " violation=" + std::string(violated ? "1" : "0") +
          " elapsed_ms=" +
          std::to_string(
              monotonic
                  ? (completed_at_steady_ns - slo.origin_steady_ns) /
                        1'000'000
                  : completed_at_unix_ms - slo.origin_created_unix_ms) +
          " lateness_ms=" +
          std::to_string(
              monotonic
                  ? (completed_at_steady_ns - slo.due_steady_ns) / 1'000'000
                  : completed_at_unix_ms - slo.due_unix_ms));
}

void ArtifactRuntimeActor::setActiveLease(
    const ArtifactLease& lease,
    std::shared_ptr<LeaseHeartbeat> heartbeat) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  idle_ = false;
  active_task_id_ = lease.taskId();
  active_heartbeat_ = std::move(heartbeat);
  state_cv_.notify_all();
}

void ArtifactRuntimeActor::clearActiveLease() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  active_heartbeat_.reset();
  active_task_id_.reset();
  idle_ = true;
  state_cv_.notify_all();
}

ArtifactExecutionResult ArtifactRuntimeActor::executeWorker(
    const ArtifactLease& lease,
    const std::shared_ptr<LeaseHeartbeat>& heartbeat) {
  ++worker_invocations_;
  std::int64_t started_at_unix_ms = 0;
  const std::int64_t started_at_steady_ns =
      artifactSteadyNowNanoseconds();
  try {
    started_at_unix_ms = nowUnixMillis();
  } catch (const std::exception& exception) {
    ArtifactExecutionResult result;
    result.status = ArtifactExecutionStatus::kRetryableFailure;
    result.error = std::string("artifact clock failed before DAM execution: ") +
                   exception.what();
    return result;
  } catch (...) {
    ArtifactExecutionResult result;
    result.status = ArtifactExecutionStatus::kRetryableFailure;
    result.error = "artifact clock failed before DAM execution";
    return result;
  }
  auto invocation = std::make_shared<WorkerInvocation>();
  const std::shared_ptr<DamWorker> worker = worker_;
  const DamTaskRequest request = lease.request;
  std::thread([invocation, worker, request, heartbeat]() {
    DamWorkerResponse response;
    try {
      response = worker->describe(
          request, [heartbeat](std::int64_t) {
            return heartbeat->beatNow();
          });
    } catch (const std::exception& exception) {
      response.success = false;
      response.retryable = true;
      response.error = std::string("DAM worker threw: ") + exception.what();
    } catch (...) {
      response.success = false;
      response.retryable = true;
      response.error = "DAM worker threw a non-standard exception";
    }
    {
      std::lock_guard<std::mutex> lock(invocation->mutex);
      invocation->response = std::move(response);
      invocation->done = true;
    }
    invocation->cv.notify_all();
  }).detach();

  std::unique_lock<std::mutex> lock(invocation->mutex);
  while (!invocation->done && !stopRequested() && heartbeat->healthy()) {
    invocation->cv.wait_for(lock, config_.durability_wait_slice);
  }
  if (!invocation->done) {
    const bool stopped = stopRequested();
    lock.unlock();
    heartbeat->stop();
    lock.lock();
    invocation->cv.wait_for(
        lock, config_.worker_stop_timeout,
        [&invocation]() { return invocation->done; });
    if (!invocation->done) {
      ++detached_worker_invocations_;
    }
    ArtifactExecutionResult result;
    result.status = stopped ? ArtifactExecutionStatus::kRetryableFailure
                            : ArtifactExecutionStatus::kInvalidLease;
    result.error = stopped
                       ? "artifact actor stopped during DAM execution"
                       : "DAM execution lost its lease heartbeat";
    return result;
  }

  DamWorkerResponse response = std::move(invocation->response);
  lock.unlock();
  if (stopRequested()) {
    ArtifactExecutionResult result;
    result.status = ArtifactExecutionStatus::kRetryableFailure;
    result.error = "artifact actor stopped after DAM execution";
    return result;
  }
  if (!heartbeat->healthy()) {
    ArtifactExecutionResult result;
    result.status = ArtifactExecutionStatus::kInvalidLease;
    result.error = heartbeat->lastError();
    return result;
  }

  ReplayDamWorker replay(std::move(response));
  ArtifactExecutionResult result;
  try {
    result = scheduler_->execute(lease, &replay, nowUnixMillis());
  } catch (const std::exception& exception) {
    result.status = ArtifactExecutionStatus::kRetryableFailure;
    result.error = std::string("artifact scheduler execution threw: ") +
                   exception.what();
  } catch (...) {
    result.status = ArtifactExecutionStatus::kRetryableFailure;
    result.error = "artifact scheduler execution threw";
  }
  const ArtifactSloContext execution_slo{
      lease.request.created_unix_ms,
      lease.request.due_unix_ms,
      lease.request.priority,
      lease.request.steady_clock_epoch,
      lease.request.origin_steady_ns,
      lease.request.due_steady_ns};
  result.slo_violation =
      hasCurrentArtifactSteadyClock(execution_slo)
          ? started_at_steady_ns > execution_slo.due_steady_ns
          : started_at_unix_ms > execution_slo.due_unix_ms;
  return result;
}

bool ArtifactRuntimeActor::waitUntilRevisionDurable(
    const ArtifactLease&,
    SceneRevision revision,
    const std::shared_ptr<LeaseHeartbeat>& heartbeat) {
  if (revision == 0) {
    ++durability_errors_;
    recordError("artifact reducer result has no scene revision");
    return false;
  }
  ++durability_waits_;
  while (!stopRequested() && heartbeat->healthy()) {
    try {
      if (durability_waiter_(revision, config_.durability_wait_slice)) {
        return heartbeat->healthy();
      }
    } catch (const std::exception& exception) {
      ++durability_errors_;
      recordError(std::string("artifact durability waiter failed: ") +
                  exception.what());
      return false;
    } catch (...) {
      ++durability_errors_;
      recordError("artifact durability waiter failed");
      return false;
    }
  }
  if (heartbeat->leaseLost()) {
    ++lease_lost_;
    recordError(heartbeat->lastError());
  }
  return false;
}

bool ArtifactRuntimeActor::completeLease(
    const ArtifactLease& lease,
    const std::shared_ptr<LeaseHeartbeat>& heartbeat,
    bool account_slo) {
  heartbeat->stop();
  SceneStoreStatus status;
  std::int64_t completed_at_unix_ms = 0;
  try {
    completed_at_unix_ms = nowUnixMillis();
    status = scheduler_->complete(lease, completed_at_unix_ms);
  } catch (const std::exception& exception) {
    status = SceneStoreStatus::failure(exception.what());
  } catch (...) {
    status = SceneStoreStatus::failure("artifact completion threw");
  }
  if (!status) {
    ++scheduler_errors_;
    ++lease_lost_;
    recordError("artifact completion failed: " + status.error);
    return false;
  }
  ++completed_;
  if (account_slo) {
    accountSloCompletion(lease, completed_at_unix_ms);
  }
  return true;
}

bool ArtifactRuntimeActor::retryLease(
    const ArtifactLease& lease,
    std::string error,
    const std::shared_ptr<LeaseHeartbeat>& heartbeat,
    bool stopping) {
  if (heartbeat) {
    heartbeat->stop();
  }
  const std::string retry_error = error.empty()
                                      ? "artifact execution requested retry"
                                      : std::move(error);
  ArtifactRetryResult retried;
  try {
    retried = scheduler_->retry(lease, nowUnixMillis(), retry_error);
  } catch (const std::exception& exception) {
    retried.status = SceneStoreStatus::failure(exception.what());
  } catch (...) {
    retried.status = SceneStoreStatus::failure("artifact retry threw");
  }
  if (!retried.status) {
    ++scheduler_errors_;
    ++lease_lost_;
    recordError("artifact retry failed: " + retried.status.error);
    return false;
  }
  ++retried_;
  if (stopping) {
    ++stop_requeues_;
  }
  recordError(retry_error);
  return true;
}

bool ArtifactRuntimeActor::failLease(
    const ArtifactLease& lease,
    std::string error,
    const std::shared_ptr<LeaseHeartbeat>& heartbeat) {
  ++terminal_failures_;
  heartbeat->stop();

  SceneStoreStatus failed;
  try {
    const std::int64_t failed_at = nowUnixMillis();
    failed = terminal_failure_sink_
                 ? terminal_failure_sink_(lease, failed_at, error)
                 : scheduler_->fail(lease, failed_at, error);
  } catch (const std::exception& exception) {
    failed = SceneStoreStatus::failure(exception.what());
  } catch (...) {
    failed = SceneStoreStatus::failure("terminal failure sink threw");
  }
  if (failed) {
    recordError(std::move(error));
    return true;
  }

  ++scheduler_errors_;
  ++lease_lost_;
  recordError("terminal failure disposition failed: " + failed.error);
  return false;
}

bool ArtifactRuntimeActor::completeSuperseded(
    const ArtifactLease& lease,
    const SceneSnapshot& snapshot,
    DamDependencyFreshness freshness,
    const std::shared_ptr<LeaseHeartbeat>& heartbeat) {
  ++superseded_;
  if (!waitUntilRevisionDurable(lease, snapshot.revision(), heartbeat)) {
    if (heartbeat->leaseLost()) {
      return false;
    }
    return retryLease(
        lease,
        std::string("superseding scene revision is not durable: ") +
            freshnessName(freshness),
        heartbeat, stopRequested());
  }
  return completeLease(lease, heartbeat);
}

bool ArtifactRuntimeActor::processNextLease() {
  if (execution_admission_gate_) {
    try {
      if (!execution_admission_gate_()) {
        ++admission_deferrals_;
        return false;
      }
    } catch (const std::exception& exception) {
      ++admission_deferrals_;
      recordError(std::string("artifact admission gate threw: ") +
                  exception.what());
      return false;
    } catch (...) {
      ++admission_deferrals_;
      recordError("artifact admission gate threw");
      return false;
    }
  }
  ++lease_polls_;
  ArtifactLeaseResult leased;
  std::int64_t leased_at = 0;
  try {
    leased_at = nowUnixMillis();
    leased = scheduler_->leaseNext(config_.lease_owner, leased_at);
  } catch (const std::exception& exception) {
    ++scheduler_errors_;
    recordError(std::string("artifact lease poll threw: ") +
                exception.what());
    return false;
  } catch (...) {
    ++scheduler_errors_;
    recordError("artifact lease poll threw");
    return false;
  }
  if (!leased.status) {
    ++scheduler_errors_;
    recordError("artifact lease poll failed: " + leased.status.error);
    return false;
  }
  if (!leased.lease) {
    return false;
  }

  ++leases_acquired_;
  const ArtifactLease lease = std::move(*leased.lease);
  auto heartbeat = std::make_shared<LeaseHeartbeat>(
      scheduler_, lease, leased_at, config_.heartbeat_period,
      unix_millis_clock_);
  setActiveLease(lease, heartbeat);
  struct ActiveLeaseGuard {
    ArtifactRuntimeActor* actor;
    std::shared_ptr<LeaseHeartbeat> heartbeat;
    ~ActiveLeaseGuard() {
      heartbeat->stop();
      actor->clearActiveLease();
    }
  } guard{this, heartbeat};
  heartbeat->start();

  if (!heartbeat->healthy()) {
    ++lease_lost_;
    recordError(heartbeat->lastError());
    return true;
  }
  if (stopRequested()) {
    retryLease(lease, "artifact actor stopped after leasing", heartbeat,
               true);
    return true;
  }

  SceneSnapshot snapshot;
  try {
    snapshot = snapshot_supplier_();
  } catch (const std::exception& exception) {
    retryLease(lease,
               std::string("scene snapshot supplier failed: ") +
                   exception.what(),
               heartbeat);
    return true;
  } catch (...) {
    retryLease(lease, "scene snapshot supplier failed", heartbeat);
    return true;
  }
  if (snapshot.revision() < lease.request.scene_revision) {
    retryLease(lease,
               "live scene is older than the durable DAM task dependency",
               heartbeat);
    return true;
  }

  const DamDependencyFreshness freshness =
      inspectDamDependency(snapshot, lease.request);
  if (freshness == DamDependencyFreshness::kInvalid) {
    failLease(lease, "invalid DAM dependency contract", heartbeat);
    return true;
  }
  if (freshness != DamDependencyFreshness::kCurrent) {
    completeSuperseded(lease, snapshot, freshness, heartbeat);
    return true;
  }

  const ArtifactExecutionResult execution = executeWorker(lease, heartbeat);
  switch (execution.status) {
    case ArtifactExecutionStatus::kRetryableFailure:
      retryLease(lease, execution.error, heartbeat, stopRequested());
      return true;
    case ArtifactExecutionStatus::kPermanentFailure:
      failLease(lease, execution.error, heartbeat);
      return true;
    case ArtifactExecutionStatus::kInvalidLease:
      if (stopRequested() && heartbeat->healthy()) {
        retryLease(lease, execution.error, heartbeat, true);
      } else {
        heartbeat->stop();
        ++lease_lost_;
        recordError(execution.error);
      }
      return true;
    case ArtifactExecutionStatus::kReadyToApply:
      break;
  }

  if (!execution.command) {
    failLease(lease, "DAM execution produced no reducer command", heartbeat);
    return true;
  }
  if (stopRequested()) {
    retryLease(lease, "artifact actor stopped before reducer submission",
               heartbeat, true);
    return true;
  }
  if (!heartbeat->beatNow()) {
    ++lease_lost_;
    recordError(heartbeat->lastError());
    return true;
  }

  ++reducer_submissions_;
  ArtifactReducerSubmitResult submitted;
  try {
    submitted = reducer_sink_(*execution.command,
                              config_.reducer_submit_timeout);
  } catch (const std::exception& exception) {
    ++reducer_errors_;
    retryLease(lease,
               std::string("artifact reducer sink threw: ") +
                   exception.what(),
               heartbeat);
    return true;
  } catch (...) {
    ++reducer_errors_;
    retryLease(lease, "artifact reducer sink threw", heartbeat);
    return true;
  }
  if (stopRequested()) {
    retryLease(lease, "artifact actor stopped after reducer submission",
               heartbeat, true);
    return true;
  }
  if (!submitted) {
    ++reducer_errors_;
    retryLease(lease,
               submitted.error.empty()
                   ? "artifact reducer delivery timed out or failed"
                   : "artifact reducer delivery failed: " + submitted.error,
               heartbeat);
    return true;
  }

  const SceneApplyResult& applied = submitted.apply_result;
  if (applied.status == SceneApplyStatus::kMetadataUpdated) {
    ++reducer_errors_;
    retryLease(lease,
               "description command unexpectedly produced metadata-only update",
               heartbeat);
    return true;
  }
  if (applied.status == SceneApplyStatus::kRejected) {
    const DamDependencyFreshness after_rejection =
        inspectDamDependency(applied.snapshot, lease.request);
    if (after_rejection == DamDependencyFreshness::kInvalid) {
      failLease(lease,
                applied.reason.empty()
                    ? "description command was rejected with invalid dependency"
                    : applied.reason,
                heartbeat);
      return true;
    }
    if (after_rejection == DamDependencyFreshness::kCurrent) {
      ++reducer_errors_;
      retryLease(lease,
                 applied.reason.empty()
                     ? "description command was rejected"
                     : "description command was rejected: " + applied.reason,
                 heartbeat);
      return true;
    }
    completeSuperseded(lease, applied.snapshot, after_rejection, heartbeat);
    return true;
  }

  SceneRevision durability_target = applied.revision;
  if (applied.status == SceneApplyStatus::kNoOp) {
    const SceneObjectPtr object =
        applied.snapshot.findObject(lease.request.key.object_id);
    const ArtifactComponent* artifact =
        object && object->artifact ? object->artifact.get() : nullptr;
    const std::string input_hash = canonicalDamTaskKey(lease.request.key);
    if (artifact && artifact->pending_description_input_hash == input_hash &&
        artifact->pending_description_scene_revision != 0) {
      durability_target = artifact->pending_description_scene_revision;
    } else if (artifact && artifact->description_input_hash == input_hash &&
               !artifact->description_stale) {
      // Reducer promotes pending -> current only from PersistedThroughCommand,
      // so a current-ready duplicate is already behind the durable watermark.
      return completeLease(lease, heartbeat, true);
    } else {
      ++reducer_errors_;
      return retryLease(
          lease,
          "description no-op did not identify the current or pending artifact",
          heartbeat);
    }
  }

  if (!waitUntilRevisionDurable(lease, durability_target, heartbeat)) {
    if (!heartbeat->leaseLost()) {
      retryLease(lease,
                 "description scene revision did not become durable",
                 heartbeat, stopRequested());
    }
    return true;
  }
  completeLease(lease, heartbeat, true);
  return true;
}

void ArtifactRuntimeActor::interruptiblePollWait() {
  std::unique_lock<std::mutex> lock(state_mutex_);
  state_cv_.wait_for(lock, config_.idle_poll_period,
                     [this]() { return stopRequested(); });
}

void ArtifactRuntimeActor::run() {
  while (!stopRequested()) {
    if (!processNextLease()) {
      interruptiblePollWait();
    }
  }
}

void ArtifactRuntimeActor::onStopRequested() {
  std::shared_ptr<LeaseHeartbeat> heartbeat;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    heartbeat = active_heartbeat_;
  }
  if (heartbeat) {
    heartbeat->stop();
  }
  state_cv_.notify_all();
}

}  // namespace roomie
