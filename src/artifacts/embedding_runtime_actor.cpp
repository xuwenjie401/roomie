#include "roomie/artifacts/embedding_runtime_actor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

#include "roomie/artifacts/artifact_slo_clock.hpp"
#include "roomie/utils/run_logger.hpp"

namespace roomie {
namespace {

using Milliseconds = std::chrono::milliseconds;

constexpr float kVectorNormEpsilon = 1.0e-12f;

std::int64_t systemUnixMillis() {
  return std::chrono::duration_cast<Milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

const char* freshnessName(EmbeddingDependencyFreshness freshness) {
  switch (freshness) {
    case EmbeddingDependencyFreshness::kCurrent:
      return "current";
    case EmbeddingDependencyFreshness::kRetiredAlias:
      return "retired_alias";
    case EmbeddingDependencyFreshness::kTombstoned:
      return "tombstoned";
    case EmbeddingDependencyFreshness::kMissingObject:
      return "missing_object";
    case EmbeddingDependencyFreshness::kIdentityStale:
      return "identity_stale";
    case EmbeddingDependencyFreshness::kDocumentStale:
      return "document_stale";
    case EmbeddingDependencyFreshness::kInvalid:
      return "invalid";
  }
  return "unknown";
}

bool normalizeVector(const std::vector<float>& input,
                     std::size_t dimension,
                     std::vector<float>* output) {
  if (input.size() != dimension) {
    return false;
  }
  double squared_norm = 0.0;
  for (float value : input) {
    if (!std::isfinite(value)) {
      return false;
    }
    squared_norm += static_cast<double>(value) * value;
  }
  if (!std::isfinite(squared_norm) ||
      squared_norm <= static_cast<double>(kVectorNormEpsilon)) {
    return false;
  }
  const float inverse_norm =
      static_cast<float>(1.0 / std::sqrt(squared_norm));
  output->clear();
  output->reserve(input.size());
  for (float value : input) {
    output->push_back(value * inverse_norm);
  }
  return true;
}

}  // namespace

struct EmbeddingRuntimeActor::LeaseHeartbeat {
  LeaseHeartbeat(EmbeddingTaskScheduler* scheduler,
                 EmbeddingTaskLease lease,
                 std::int64_t leased_at_unix_ms,
                 Milliseconds configured_period,
                 UnixMillisClock clock)
      : scheduler(scheduler),
        lease(std::move(lease)),
        clock(std::move(clock)),
        lease_until_unix_ms(this->lease.durable.lease_until_unix_ms) {
    lease_duration_ms = lease_until_unix_ms - leased_at_unix_ms;
    if (lease_duration_ms <= 0) {
      markLost("leased embedding task was already expired");
      return;
    }
    const std::int64_t third =
        std::max<std::int64_t>(1, lease_duration_ms / 3);
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
    } catch (const std::exception& error) {
      markLost(std::string("embedding clock failed during heartbeat: ") +
               error.what());
      return false;
    } catch (...) {
      markLost("embedding clock failed during heartbeat");
      return false;
    }
    if (now >= lease_until_unix_ms) {
      markLost("embedding lease expired before heartbeat");
      return false;
    }
    const SceneStoreStatus renewed = scheduler->heartbeat(lease, now);
    if (!renewed) {
      markLost("embedding lease heartbeat failed: " + renewed.error);
      return false;
    }
    if (now > std::numeric_limits<std::int64_t>::max() - lease_duration_ms) {
      markLost("embedding lease heartbeat time overflowed");
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

  EmbeddingTaskScheduler* scheduler = nullptr;
  EmbeddingTaskLease lease;
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

struct EmbeddingRuntimeActor::ActiveLease {
  EmbeddingTaskLease lease;
  std::shared_ptr<LeaseHeartbeat> heartbeat;
  std::optional<SemanticDocument> document;
};

struct EmbeddingRuntimeActor::EncoderInvocation {
  std::mutex mutex;
  std::condition_variable cv;
  bool done = false;
  std::string error;
  std::vector<std::vector<float>> vectors;
};

EmbeddingRuntimeActorConfig EmbeddingRuntimeActor::validateConfig(
    EmbeddingRuntimeActorConfig config) {
  if (config.lease_owner.empty() || config.maximum_batch_size == 0 ||
      config.slo_accounting_history_capacity == 0 ||
      config.collection_window < Milliseconds::zero() ||
      config.idle_poll_period <= Milliseconds::zero() ||
      config.heartbeat_period <= Milliseconds::zero() ||
      config.invocation_wait_slice <= Milliseconds::zero() ||
      config.index_publish_wait_slice <= Milliseconds::zero() ||
      config.index_publish_timeout <= Milliseconds::zero() ||
      config.encoder_stop_timeout < Milliseconds::zero()) {
    throw std::invalid_argument("embedding runtime configuration is invalid");
  }
  return config;
}

EmbeddingRuntimeActor::EmbeddingRuntimeActor(
    EmbeddingRuntimeActorConfig config,
    EmbeddingTaskScheduler* scheduler,
    std::shared_ptr<EmbeddingEncoder> encoder,
    VersionedSemanticIndex* index,
    SceneSnapshotSupplier snapshot_supplier,
    DurableEmbeddingSink durable_embedding_sink,
    TerminalFailureSink terminal_failure_sink,
    UnixMillisClock unix_millis_clock)
    : WorkerThread("embedding_runtime_actor"),
      config_(validateConfig(std::move(config))),
      scheduler_(scheduler),
      encoder_(std::move(encoder)),
      index_(index),
      snapshot_supplier_(std::move(snapshot_supplier)),
      durable_embedding_sink_(std::move(durable_embedding_sink)),
      terminal_failure_sink_(std::move(terminal_failure_sink)),
      unix_millis_clock_(std::move(unix_millis_clock)) {
  if (!scheduler_ || !encoder_ || !index_ || !snapshot_supplier_ ||
      !durable_embedding_sink_) {
    throw std::invalid_argument(
        "embedding runtime requires scheduler, encoder, index, scene "
        "supplier, and durable sink");
  }
  if (encoder_->modelId().empty() || encoder_->dimension() == 0) {
    throw std::invalid_argument("embedding runtime encoder namespace is invalid");
  }
  if (!unix_millis_clock_) {
    unix_millis_clock_ = systemUnixMillis;
  }
}

EmbeddingRuntimeActor::~EmbeddingRuntimeActor() { stop(); }

bool EmbeddingRuntimeActor::waitUntilWarmed(Milliseconds timeout) const {
  if (timeout < Milliseconds::zero()) {
    timeout = Milliseconds::zero();
  }
  std::unique_lock<std::mutex> lock(state_mutex_);
  return state_cv_.wait_for(lock, timeout, [this]() { return warmed_; });
}

bool EmbeddingRuntimeActor::waitUntilIdle(Milliseconds timeout) const {
  if (timeout < Milliseconds::zero()) {
    timeout = Milliseconds::zero();
  }
  std::unique_lock<std::mutex> lock(state_mutex_);
  return state_cv_.wait_for(lock, timeout, [this]() { return idle_; });
}

bool EmbeddingRuntimeActor::idle() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return idle_;
}

EmbeddingRuntimeActorStats EmbeddingRuntimeActor::stats() const {
  EmbeddingRuntimeActorStats result;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    result.warmed = warmed_;
  }
  result.warmup_attempts = warmup_attempts_.load();
  result.warmup_failures = warmup_failures_.load();
  result.lease_polls = lease_polls_.load();
  result.leases_acquired = leases_acquired_.load();
  result.batches = batches_.load();
  result.encoded_documents = encoded_documents_.load();
  result.persisted_records = persisted_records_.load();
  result.accepted_records = accepted_records_.load();
  result.completed = completed_.load();
  result.superseded = superseded_.load();
  result.retried = retried_.load();
  result.terminal_failures = terminal_failures_.load();
  result.lease_lost = lease_lost_.load();
  result.scheduler_errors = scheduler_errors_.load();
  result.persistence_errors = persistence_errors_.load();
  result.index_errors = index_errors_.load();
  result.interactive_slo_completions =
      interactive_slo_completions_.load();
  result.bulk_slo_completions = bulk_slo_completions_.load();
  result.untracked_slo_completions = untracked_slo_completions_.load();
  result.interactive_slo_violations =
      interactive_slo_violations_.load();
  result.bulk_slo_violations = bulk_slo_violations_.load();
  result.slo_violations = slo_violations_.load();
  result.stop_requeues = stop_requeues_.load();
  result.detached_encoder_invocations =
      detached_encoder_invocations_.load();
  {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    result.slo_accounting_history_size = slo_accounted_task_ids_.size();
  }
  return result;
}

std::string EmbeddingRuntimeActor::lastError() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return last_error_;
}

std::vector<std::string> EmbeddingRuntimeActor::activeTaskIds() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return active_task_ids_;
}

std::int64_t EmbeddingRuntimeActor::nowUnixMillis() const {
  return unix_millis_clock_();
}

void EmbeddingRuntimeActor::recordError(std::string error) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  last_error_ = std::move(error);
}

void EmbeddingRuntimeActor::accountSloQueryableAccept(
    const EmbeddingTaskLease& lease,
    std::int64_t accepted_at_unix_ms) {
  if (!lease.request) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    if (!slo_accounted_task_ids_.insert(lease.taskId()).second) {
      return;
    }
    slo_accounted_task_order_.push_back(lease.taskId());
    while (slo_accounted_task_ids_.size() >
           config_.slo_accounting_history_capacity) {
      slo_accounted_task_ids_.erase(slo_accounted_task_order_.front());
      slo_accounted_task_order_.pop_front();
    }
  }
  const ArtifactSloContext& slo = lease.request->artifact_slo;
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
  const std::int64_t accepted_at_steady_ns =
      monotonic ? artifactSteadyNowNanoseconds() : 0;
  const bool violated = monotonic
                            ? accepted_at_steady_ns > slo.due_steady_ns
                            : accepted_at_unix_ms > slo.due_unix_ms;
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
      "stage=semantic_index_accept task_id=" + lease.taskId() +
          " object_id=" + std::to_string(lease.request->object_id) +
          " priority=" +
          std::string(slo.priority == ArtifactPriority::kInteractive
                          ? "interactive"
                          : "bulk") +
          " origin_created_unix_ms=" +
          std::to_string(slo.origin_created_unix_ms) +
          " due_unix_ms=" + std::to_string(slo.due_unix_ms) +
          " completed_at_unix_ms=" +
          std::to_string(accepted_at_unix_ms) +
          " measurement_clock=" +
          std::string(monotonic ? "steady" : "unix_fallback") +
          " violation=" + std::string(violated ? "1" : "0") +
          " elapsed_ms=" +
          std::to_string(
              monotonic
                  ? (accepted_at_steady_ns - slo.origin_steady_ns) /
                        1'000'000
                  : accepted_at_unix_ms - slo.origin_created_unix_ms) +
          " lateness_ms=" +
          std::to_string(
              monotonic
                  ? (accepted_at_steady_ns - slo.due_steady_ns) / 1'000'000
                  : accepted_at_unix_ms - slo.due_unix_ms));
}

void EmbeddingRuntimeActor::setActive(
    const std::vector<std::shared_ptr<ActiveLease>>& active) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  idle_ = false;
  active_task_ids_.clear();
  active_task_ids_.reserve(active.size());
  for (const auto& entry : active) {
    active_task_ids_.push_back(entry->lease.taskId());
  }
  state_cv_.notify_all();
}

void EmbeddingRuntimeActor::clearActive() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  active_task_ids_.clear();
  idle_ = true;
  state_cv_.notify_all();
}

void EmbeddingRuntimeActor::interruptibleWait(Milliseconds duration) {
  std::unique_lock<std::mutex> lock(state_mutex_);
  state_cv_.wait_for(lock, duration,
                     [this]() { return stopRequested(); });
}

bool EmbeddingRuntimeActor::prewarmEncoder() {
  ++warmup_attempts_;
  auto invocation = std::make_shared<EncoderInvocation>();
  const std::shared_ptr<EmbeddingEncoder> encoder = encoder_;
  try {
    std::thread([invocation, encoder]() {
      try {
        encoder->prewarm();
      } catch (const std::exception& error) {
        invocation->error = error.what();
      } catch (...) {
        invocation->error = "encoder prewarm threw a non-standard exception";
      }
      {
        std::lock_guard<std::mutex> lock(invocation->mutex);
        invocation->done = true;
      }
      invocation->cv.notify_all();
    }).detach();
  } catch (const std::exception& error) {
    ++warmup_failures_;
    recordError(std::string("cannot start encoder prewarm: ") + error.what());
    return false;
  }

  std::unique_lock<std::mutex> lock(invocation->mutex);
  while (!invocation->done && !stopRequested()) {
    invocation->cv.wait_for(lock, config_.invocation_wait_slice);
  }
  if (!invocation->done && stopRequested()) {
    invocation->cv.wait_for(
        lock, config_.encoder_stop_timeout,
        [&invocation]() { return invocation->done; });
  }
  if (!invocation->done) {
    ++detached_encoder_invocations_;
    return false;
  }
  if (!invocation->error.empty()) {
    ++warmup_failures_;
    recordError("embedding encoder prewarm failed: " + invocation->error);
    return false;
  }
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    warmed_ = true;
  }
  state_cv_.notify_all();
  return true;
}

bool EmbeddingRuntimeActor::collectLease(
    std::vector<std::shared_ptr<ActiveLease>>* batch) {
  ++lease_polls_;
  EmbeddingTaskLeaseResult leased;
  std::int64_t leased_at = 0;
  try {
    leased_at = nowUnixMillis();
    leased = scheduler_->leaseNext(config_.lease_owner, leased_at);
  } catch (const std::exception& error) {
    ++scheduler_errors_;
    recordError(std::string("embedding lease poll threw: ") + error.what());
    return false;
  } catch (...) {
    ++scheduler_errors_;
    recordError("embedding lease poll threw");
    return false;
  }
  if (!leased.status) {
    ++scheduler_errors_;
    recordError("embedding lease poll failed: " + leased.status.error);
    return false;
  }
  if (!leased.lease) {
    return false;
  }
  ++leases_acquired_;
  auto active = std::make_shared<ActiveLease>();
  active->lease = std::move(*leased.lease);
  active->heartbeat = std::make_shared<LeaseHeartbeat>(
      scheduler_, active->lease, leased_at, config_.heartbeat_period,
      unix_millis_clock_);
  active->heartbeat->start();
  batch->push_back(std::move(active));
  return true;
}

bool EmbeddingRuntimeActor::completeLease(
    const std::shared_ptr<ActiveLease>& active) {
  active->heartbeat->stop();
  SceneStoreStatus status;
  try {
    status = scheduler_->complete(active->lease, nowUnixMillis());
  } catch (const std::exception& error) {
    status = SceneStoreStatus::failure(error.what());
  } catch (...) {
    status = SceneStoreStatus::failure("embedding completion threw");
  }
  if (!status) {
    ++scheduler_errors_;
    ++lease_lost_;
    recordError("embedding completion failed: " + status.error);
    return false;
  }
  ++completed_;
  return true;
}

bool EmbeddingRuntimeActor::retryLease(
    const std::shared_ptr<ActiveLease>& active,
    std::string error,
    bool stopping) {
  active->heartbeat->stop();
  EmbeddingTaskRetryResult retried;
  const std::string retry_error =
      error.empty() ? "embedding execution requested retry" : std::move(error);
  try {
    retried = scheduler_->retry(active->lease, nowUnixMillis(), retry_error);
  } catch (const std::exception& exception) {
    retried.status = SceneStoreStatus::failure(exception.what());
  } catch (...) {
    retried.status = SceneStoreStatus::failure("embedding retry threw");
  }
  if (!retried.status) {
    ++scheduler_errors_;
    ++lease_lost_;
    recordError("embedding retry failed: " + retried.status.error);
    return false;
  }
  ++retried_;
  if (stopping) {
    ++stop_requeues_;
  }
  recordError(retry_error);
  return true;
}

bool EmbeddingRuntimeActor::failLease(
    const std::shared_ptr<ActiveLease>& active,
    std::string error) {
  ++terminal_failures_;
  active->heartbeat->stop();
  if (!terminal_failure_sink_) {
    return retryLease(
        active, "terminal failure sink unavailable; " + std::move(error));
  }
  SceneStoreStatus failed;
  try {
    failed = terminal_failure_sink_(active->lease, nowUnixMillis(), error);
  } catch (const std::exception& exception) {
    failed = SceneStoreStatus::failure(exception.what());
  } catch (...) {
    failed = SceneStoreStatus::failure(
        "embedding terminal failure sink threw");
  }
  if (failed) {
    recordError(std::move(error));
    return true;
  }
  ++scheduler_errors_;
  const std::string sink_error =
      "embedding terminal disposition failed: " + failed.error;
  recordError(sink_error);
  return retryLease(active, sink_error);
}

bool EmbeddingRuntimeActor::disposeSuperseded(
    const std::shared_ptr<ActiveLease>& active,
    const SceneSnapshot& snapshot,
    EmbeddingDependencyFreshness freshness) {
  if (snapshot.revision() > snapshot.durableRevision()) {
    return retryLease(
        active,
        std::string("superseding embedding scene revision is not durable: ") +
            freshnessName(freshness),
        stopRequested());
  }
  ++superseded_;
  return completeLease(active);
}

bool EmbeddingRuntimeActor::waitUntilQueryable(
    const std::shared_ptr<ActiveLease>& active) {
  if (!active || !active->lease.request) {
    return false;
  }
  const EmbeddingTaskRequest& request = *active->lease.request;
  const auto deadline =
      std::chrono::steady_clock::now() + config_.index_publish_timeout;
  while (!stopRequested() && active->heartbeat->healthy() &&
         std::chrono::steady_clock::now() < deadline) {
    (void)index_->waitForBuildIdle(config_.index_publish_wait_slice);
    std::int64_t now = 0;
    try {
      now = nowUnixMillis();
      const SemanticReadToken token = index_->pinRead(
          request.created_scene_revision, now, 1'000);
      const std::optional<std::string> indexed =
          index_->indexedDocumentHash(token, request.object_id);
      if (indexed && *indexed == request.document_hash) {
        accountSloQueryableAccept(active->lease, now);
        return true;
      }
    } catch (const std::exception& error) {
      recordError(std::string("semantic queryable fence failed: ") +
                  error.what());
    } catch (...) {
      recordError("semantic queryable fence failed");
    }
  }
  if (!stopRequested() && active->heartbeat->healthy()) {
    ++index_errors_;
    recordError("semantic index generation publication timed out");
  }
  return false;
}

void EmbeddingRuntimeActor::retryBatch(
    const std::vector<std::shared_ptr<ActiveLease>>& batch,
    const std::string& error,
    bool stopping) {
  for (const auto& active : batch) {
    if (!active->heartbeat->healthy()) {
      ++lease_lost_;
      recordError(active->heartbeat->lastError());
      continue;
    }
    retryLease(active, error, stopping);
  }
}

bool EmbeddingRuntimeActor::validateCurrent(
    const EmbeddingTaskLease& lease,
    const SceneSnapshot& snapshot,
    SemanticDocument* document,
    EmbeddingDependencyFreshness* freshness) {
  if (!lease.request) {
    *freshness = EmbeddingDependencyFreshness::kInvalid;
    return false;
  }
  if (snapshot.revision() < lease.request->owning_scene_revision) {
    return false;
  }
  *freshness =
      inspectEmbeddingDependency(snapshot, *lease.request, document);
  return *freshness == EmbeddingDependencyFreshness::kCurrent;
}

bool EmbeddingRuntimeActor::processNextBatch() {
  std::vector<std::shared_ptr<ActiveLease>> batch;
  if (!collectLease(&batch)) {
    return false;
  }
  setActive(batch);
  struct BatchGuard {
    EmbeddingRuntimeActor* actor;
    std::vector<std::shared_ptr<ActiveLease>>* batch;
    ~BatchGuard() {
      for (const auto& active : *batch) {
        active->heartbeat->stop();
      }
      actor->clearActive();
    }
  } guard{this, &batch};

  const auto collection_deadline =
      std::chrono::steady_clock::now() + config_.collection_window;
  while (!stopRequested() && batch.size() < config_.maximum_batch_size &&
         std::chrono::steady_clock::now() < collection_deadline) {
    if (collectLease(&batch)) {
      setActive(batch);
      continue;
    }
    const auto remaining = std::chrono::duration_cast<Milliseconds>(
        collection_deadline - std::chrono::steady_clock::now());
    if (remaining > Milliseconds::zero()) {
      interruptibleWait(std::min(remaining, Milliseconds(1)));
    }
  }
  if (stopRequested()) {
    retryBatch(batch, "embedding actor stopped during batch collection", true);
    return true;
  }

  SceneSnapshot snapshot;
  try {
    snapshot = snapshot_supplier_();
  } catch (const std::exception& error) {
    retryBatch(batch,
               std::string("embedding scene supplier failed: ") +
                   error.what());
    return true;
  } catch (...) {
    retryBatch(batch, "embedding scene supplier failed");
    return true;
  }

  std::vector<std::shared_ptr<ActiveLease>> candidates;
  for (const auto& active : batch) {
    if (!active->heartbeat->healthy()) {
      ++lease_lost_;
      recordError(active->heartbeat->lastError());
      continue;
    }
    if (!active->lease.valid()) {
      failLease(active, active->lease.validation_error);
      continue;
    }
    const EmbeddingTaskRequest& request = *active->lease.request;
    const SemanticGenerationInfo generation = index_->activeGeneration();
    if (request.name_space.model_id != encoder_->modelId() ||
        request.name_space.dimension != encoder_->dimension() ||
        request.name_space.model_id != generation.model_id ||
        request.name_space.dimension != generation.dimension) {
      failLease(active,
                "embedding task namespace does not match the resident "
                "encoder/index namespace");
      continue;
    }
    if (snapshot.revision() < request.owning_scene_revision) {
      retryLease(active,
                 "live scene is older than the durable embedding task");
      continue;
    }
    SemanticDocument document;
    EmbeddingDependencyFreshness freshness =
        EmbeddingDependencyFreshness::kInvalid;
    if (!validateCurrent(active->lease, snapshot, &document, &freshness)) {
      if (freshness == EmbeddingDependencyFreshness::kInvalid) {
        failLease(active, "invalid embedding dependency contract");
      } else {
        disposeSuperseded(active, snapshot, freshness);
      }
      continue;
    }
    try {
      (void)index_->upsertDocument(document);
    } catch (const std::exception& error) {
      ++index_errors_;
      retryLease(active,
                 std::string("semantic document upsert failed: ") +
                     error.what());
      continue;
    } catch (...) {
      ++index_errors_;
      retryLease(active, "semantic document upsert failed");
      continue;
    }
    active->document = std::move(document);
    candidates.push_back(active);
  }
  if (candidates.empty()) {
    return true;
  }

  std::vector<std::string> documents;
  documents.reserve(candidates.size());
  for (const auto& active : candidates) {
    documents.push_back(active->lease.request->document);
  }
  auto invocation = std::make_shared<EncoderInvocation>();
  const std::shared_ptr<EmbeddingEncoder> encoder = encoder_;
  try {
    std::thread([invocation, encoder, documents = std::move(documents)]() {
      try {
        invocation->vectors = encoder->encodeBatch(documents);
      } catch (const std::exception& error) {
        invocation->error = error.what();
      } catch (...) {
        invocation->error =
            "embedding encoder threw a non-standard exception";
      }
      {
        std::lock_guard<std::mutex> lock(invocation->mutex);
        invocation->done = true;
      }
      invocation->cv.notify_all();
    }).detach();
  } catch (const std::exception& error) {
    retryBatch(candidates,
               std::string("cannot start embedding encoder: ") +
                   error.what());
    return true;
  }
  ++batches_;

  std::unique_lock<std::mutex> invocation_lock(invocation->mutex);
  bool all_leases_lost = false;
  while (!invocation->done && !stopRequested()) {
    invocation->cv.wait_for(invocation_lock, config_.invocation_wait_slice);
    all_leases_lost = std::none_of(
        candidates.begin(), candidates.end(), [](const auto& active) {
          return active->heartbeat->healthy();
        });
    if (all_leases_lost) {
      break;
    }
  }
  if (!invocation->done && stopRequested()) {
    invocation->cv.wait_for(
        invocation_lock, config_.encoder_stop_timeout,
        [&invocation]() { return invocation->done; });
  }
  if (!invocation->done) {
    ++detached_encoder_invocations_;
    invocation_lock.unlock();
    if (stopRequested()) {
      retryBatch(candidates,
                 "embedding actor stopped during encoder invocation", true);
    } else if (all_leases_lost) {
      for (const auto& active : candidates) {
        ++lease_lost_;
        recordError(active->heartbeat->lastError());
      }
    }
    return true;
  }
  std::string invocation_error = std::move(invocation->error);
  std::vector<std::vector<float>> vectors = std::move(invocation->vectors);
  invocation_lock.unlock();

  if (stopRequested()) {
    retryBatch(candidates,
               "embedding actor stopped after encoder invocation", true);
    return true;
  }
  if (!invocation_error.empty()) {
    retryBatch(candidates,
               "embedding encoder failed: " + invocation_error);
    return true;
  }
  if (vectors.size() != candidates.size()) {
    retryBatch(candidates,
               "embedding encoder returned the wrong batch size");
    return true;
  }
  encoded_documents_.fetch_add(vectors.size());

  SceneSnapshot latest;
  try {
    latest = snapshot_supplier_();
  } catch (const std::exception& error) {
    retryBatch(candidates,
               std::string("embedding post-encode scene supplier failed: ") +
                   error.what());
    return true;
  } catch (...) {
    retryBatch(candidates, "embedding post-encode scene supplier failed");
    return true;
  }

  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const auto& active = candidates[index];
    if (!active->heartbeat->healthy()) {
      ++lease_lost_;
      recordError(active->heartbeat->lastError());
      continue;
    }
    if (stopRequested()) {
      retryLease(active, "embedding actor stopped before persistence", true);
      continue;
    }
    SemanticDocument current_document;
    EmbeddingDependencyFreshness freshness =
        EmbeddingDependencyFreshness::kInvalid;
    if (!validateCurrent(active->lease, latest, &current_document,
                         &freshness)) {
      if (latest.revision() <
          active->lease.request->owning_scene_revision) {
        retryLease(active,
                   "live scene regressed before embedding persistence");
      } else if (freshness == EmbeddingDependencyFreshness::kInvalid) {
        failLease(active, "invalid post-encode embedding dependency");
      } else {
        disposeSuperseded(active, latest, freshness);
      }
      continue;
    }
    try {
      (void)index_->upsertDocument(current_document);
    } catch (const std::exception& error) {
      ++index_errors_;
      retryLease(active,
                 std::string("post-encode semantic upsert failed: ") +
                     error.what());
      continue;
    } catch (...) {
      ++index_errors_;
      retryLease(active, "post-encode semantic upsert failed");
      continue;
    }

    std::vector<float> normalized;
    if (!normalizeVector(vectors[index], encoder_->dimension(),
                         &normalized)) {
      failLease(active,
                "embedding encoder returned an invalid vector");
      continue;
    }
    EmbeddingRecord record;
    record.object_id = active->lease.request->object_id;
    record.document_hash = active->lease.request->document_hash;
    record.model_id = active->lease.request->name_space.model_id;
    record.vector = std::move(normalized);
    record.created_scene_revision =
        active->lease.request->created_scene_revision;

    SceneStoreStatus persisted;
    try {
      persisted = durable_embedding_sink_(record);
    } catch (const std::exception& error) {
      persisted = SceneStoreStatus::failure(error.what());
    } catch (...) {
      persisted = SceneStoreStatus::failure(
          "durable embedding sink threw");
    }
    if (!persisted) {
      ++persistence_errors_;
      retryLease(active,
                 "embedding persistence failed: " + persisted.error);
      continue;
    }
    ++persisted_records_;
    if (stopRequested()) {
      retryLease(active,
                 "embedding actor stopped after persistence", true);
      continue;
    }

    EmbeddingAcceptStatus accepted =
        EmbeddingAcceptStatus::kRejectedInvalidVector;
    try {
      accepted = index_->acceptEmbedding(record);
    } catch (const std::exception& error) {
      ++index_errors_;
      retryLease(active,
                 std::string("semantic index accept threw: ") +
                     error.what());
      continue;
    } catch (...) {
      ++index_errors_;
      retryLease(active, "semantic index accept threw");
      continue;
    }
    switch (accepted) {
      case EmbeddingAcceptStatus::kAccepted:
        ++accepted_records_;
        if (waitUntilQueryable(active)) {
          completeLease(active);
        } else if (active->heartbeat->healthy()) {
          retryLease(active,
                     "embedding index generation did not become queryable",
                     stopRequested());
        } else {
          ++lease_lost_;
          recordError(active->heartbeat->lastError());
        }
        break;
      case EmbeddingAcceptStatus::kRejectedObjectMissing:
      case EmbeddingAcceptStatus::kRejectedDocumentStale: {
        SceneSnapshot after_accept;
        try {
          after_accept = snapshot_supplier_();
        } catch (...) {
          retryLease(active,
                     "cannot verify semantic index rejection against scene");
          break;
        }
        SemanticDocument ignored;
        const EmbeddingDependencyFreshness after_freshness =
            inspectEmbeddingDependency(after_accept,
                                       *active->lease.request, &ignored);
        if (after_freshness != EmbeddingDependencyFreshness::kCurrent &&
            after_freshness != EmbeddingDependencyFreshness::kInvalid) {
          disposeSuperseded(active, after_accept, after_freshness);
        } else {
          ++index_errors_;
          retryLease(active,
                     "semantic index rejected a current embedding record");
        }
        break;
      }
      case EmbeddingAcceptStatus::kRejectedNamespace:
      case EmbeddingAcceptStatus::kRejectedDimension:
      case EmbeddingAcceptStatus::kRejectedInvalidVector:
        ++index_errors_;
        failLease(active,
                  "semantic index permanently rejected embedding record");
        break;
    }
  }
  return true;
}

void EmbeddingRuntimeActor::run() {
  while (!stopRequested()) {
    if (!stats().warmed) {
      if (!prewarmEncoder() && !stopRequested()) {
        interruptibleWait(config_.idle_poll_period);
      }
      continue;
    }
    try {
      if (!processNextBatch()) {
        interruptibleWait(config_.idle_poll_period);
      }
    } catch (const std::exception& error) {
      recordError(std::string("embedding runtime loop failed: ") +
                  error.what());
      interruptibleWait(config_.idle_poll_period);
    } catch (...) {
      recordError("embedding runtime loop failed");
      interruptibleWait(config_.idle_poll_period);
    }
  }
}

void EmbeddingRuntimeActor::onStopRequested() {
  // Do not stop healthy heartbeats here. The actor thread uses the remaining
  // fenced lease time to requeue every in-flight task before returning. An
  // uncooperative detached encoder cannot publish or acknowledge its result.
  state_cv_.notify_all();
}

}  // namespace roomie
