#include "roomie/scene/persistence_actor.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

namespace roomie {
namespace {

constexpr std::chrono::milliseconds kFlushFailureRetryDelay{10};

}  // namespace

PersistenceActor::PersistenceActor(std::string database_path,
                                   PersistenceActorConfig config)
    : PersistenceActor(
          std::make_unique<SceneStore>(
              std::move(database_path), storePendingLimit(config)),
          config) {}

PersistenceActor::PersistenceActor(std::unique_ptr<SceneStore> store,
                                   PersistenceActorConfig config)
    : WorkerThread("persistence_actor"),
      config_(validateConfig(std::move(config))),
      store_(std::move(store)),
      commit_queue_(config_.queue_capacity,
                    ChannelPolicy::kReliableBlocking) {
  if (!store_) {
    throw std::invalid_argument("PersistenceActor requires a SceneStore");
  }
}

PersistenceActor::~PersistenceActor() { stop(); }

PersistenceActorConfig PersistenceActor::validateConfig(
    PersistenceActorConfig config) {
  if (config.flush_period <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument(
        "persistence flush period must be positive");
  }
  if (config.flush_batch_size == 0) {
    throw std::invalid_argument(
        "persistence flush batch size must be positive");
  }
  if (config.queue_capacity == 0) {
    throw std::invalid_argument(
        "persistence commit queue capacity must be positive");
  }
  if (config.max_undurable_revisions == 0 ||
      config.hard_max_undurable_revisions == 0) {
    throw std::invalid_argument(
        "persistence lag limits must be positive");
  }
  if (config.max_undurable_revisions >
      config.hard_max_undurable_revisions) {
    throw std::invalid_argument(
        "persistence soft lag limit exceeds hard lag limit");
  }
  if (config.terminal_failure_timeout <=
      std::chrono::milliseconds::zero()) {
    throw std::invalid_argument(
        "persistence terminal failure timeout must be positive");
  }
  if (config.hard_max_undurable_revisions >
      static_cast<SceneRevision>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument(
        "persistence hard lag limit does not fit in size_t");
  }
  return config;
}

std::size_t PersistenceActor::storePendingLimit(
    const PersistenceActorConfig& config) {
  if (config.hard_max_undurable_revisions == 0 ||
      config.hard_max_undurable_revisions >
          static_cast<SceneRevision>(std::numeric_limits<std::size_t>::max())) {
    return SceneStore::kDefaultMaxPendingCommits;
  }
  return static_cast<std::size_t>(config.hard_max_undurable_revisions);
}

PushResult<SceneSnapshot> PersistenceActor::rejectedResult(
    SceneSnapshot snapshot) {
  PushResult<SceneSnapshot> result;
  result.outcome = PushOutcome::kRejected;
  result.unconsumed_item.emplace(std::move(snapshot));
  return result;
}

SceneStoreStatus PersistenceActor::open() {
  std::lock_guard<std::mutex> admission_lock(admission_mutex_);
  if (opened_.load(std::memory_order_acquire)) {
    return SceneStoreStatus::success();
  }
  const SceneStoreStatus opened = store_->isOpen()
                                      ? SceneStoreStatus::success()
                                      : store_->open();
  if (!opened) {
    recordFailure(opened.error, false);
    return opened;
  }

  const SceneStoreWatermarks watermarks = store_->watermarks();
  if (watermarks.latest_scene_revision !=
      watermarks.durable_scene_revision) {
    const SceneStoreStatus invalid = SceneStoreStatus::failure(
        "newly opened SceneStore has an undurable in-memory suffix");
    recordFailure(invalid.error, false);
    return invalid;
  }

  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    undurable_since_.clear();
    last_error_.clear();
    latest_scene_revision_.store(watermarks.latest_scene_revision,
                                 std::memory_order_release);
    durable_scene_revision_.store(watermarks.durable_scene_revision,
                                  std::memory_order_release);
  }
  last_acknowledged_revision_.store(watermarks.durable_scene_revision,
                                    std::memory_order_release);
  healthy_.store(true, std::memory_order_release);
  admission_fault_.store(false, std::memory_order_release);
  graceful_shutdown_complete_.store(false, std::memory_order_release);
  opened_.store(true, std::memory_order_release);
  durable_cv_.notify_all();
  return SceneStoreStatus::success();
}

bool PersistenceActor::isOpen() const {
  return opened_.load(std::memory_order_acquire) && store_->isOpen();
}

SceneRestoreResult PersistenceActor::restoreLatest() const {
  return store_->restoreLatest();
}

SceneRestoreResult PersistenceActor::restoreAt(SceneRevision revision) const {
  return store_->restoreAt(revision);
}

SceneStoreStatus PersistenceActor::rewindTo(
    SceneRevision revision,
    bool retain_aligned_map_checkpoints) {
  std::lock_guard<std::mutex> admission_lock(admission_mutex_);
  if (!opened_.load(std::memory_order_acquire)) {
    return SceneStoreStatus::failure("persistence actor is not open");
  }
  if (running() || commit_queue_.stats().depth != 0) {
    return SceneStoreStatus::failure(
        "persistence actor rewind is only allowed before start");
  }
  const SceneStoreStatus rewound = store_->rewindTo(
      revision, retain_aligned_map_checkpoints);
  if (!rewound) {
    recordFailure(rewound.error, false);
    return rewound;
  }
  const SceneStoreWatermarks watermarks = store_->watermarks();
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    undurable_since_.clear();
    last_error_.clear();
    latest_scene_revision_.store(watermarks.latest_scene_revision,
                                 std::memory_order_release);
    durable_scene_revision_.store(watermarks.durable_scene_revision,
                                  std::memory_order_release);
  }
  last_acknowledged_revision_.store(watermarks.durable_scene_revision,
                                    std::memory_order_release);
  healthy_.store(true, std::memory_order_release);
  admission_fault_.store(false, std::memory_order_release);
  durable_cv_.notify_all();
  return SceneStoreStatus::success();
}

SceneStore* PersistenceActor::durableArtifactStore() const {
  return opened_.load(std::memory_order_acquire) ? store_.get() : nullptr;
}

PushResult<SceneSnapshot> PersistenceActor::enqueueCommit(
    SceneSnapshot snapshot,
    std::vector<DurableTaskSpec> outbox_tasks,
    std::vector<DurableArtifactOrigin> artifact_origins) {
  std::lock_guard<std::mutex> admission_lock(admission_mutex_);
  if (!opened_.load(std::memory_order_acquire)) {
    ++commits_rejected_;
    return rejectedResult(std::move(snapshot));
  }

  const SceneRevision latest =
      latest_scene_revision_.load(std::memory_order_acquire);
  if (latest == std::numeric_limits<SceneRevision>::max() ||
      snapshot.revision() != latest + 1) {
    ++commits_rejected_;
    const std::string expected =
        latest == std::numeric_limits<SceneRevision>::max()
            ? "no further revision (counter exhausted)"
            : std::to_string(latest + 1);
    enterTerminalFault(
        "non-contiguous persistence revision: expected " + expected +
        " but received " + std::to_string(snapshot.revision()));
    return rejectedResult(std::move(snapshot));
  }

  const SceneRevision revision = snapshot.revision();
  PendingCommit pending{std::move(snapshot), std::move(outbox_tasks),
                        std::move(artifact_origins)};
  PushResult<PendingCommit> queued = commit_queue_.push(std::move(pending));
  if (!queued.accepted()) {
    ++commits_rejected_;
    admission_fault_.store(true, std::memory_order_release);
    recordFailure("persistence commit queue rejected scene revision " +
                      std::to_string(revision),
                  false);
    if (queued.unconsumed_item) {
      return rejectedResult(std::move(queued.unconsumed_item->snapshot));
    }
    return rejectedResult(SceneSnapshot{});
  }

  const Clock::time_point now = Clock::now();
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    latest_scene_revision_.store(revision, std::memory_order_release);
    if (durable_scene_revision_.load(std::memory_order_acquire) < revision) {
      undurable_since_.emplace_back(revision, now);
    }
  }
  ++commits_enqueued_;
  durable_cv_.notify_all();
  PushResult<SceneSnapshot> result;
  result.outcome = queued.outcome;
  return result;
}

void PersistenceActor::setDurabilityAckCallback(
    DurabilityAckCallback callback) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  durability_ack_callback_ = std::move(callback);
}

SceneRevision PersistenceActor::latestRevision() const {
  return latest_scene_revision_.load(std::memory_order_acquire);
}

SceneRevision PersistenceActor::durableRevision() const {
  return durable_scene_revision_.load(std::memory_order_acquire);
}

SceneRevision PersistenceActor::undurableLag() const {
  const SceneRevision latest = latestRevision();
  const SceneRevision durable = durableRevision();
  return latest >= durable ? latest - durable : 0;
}

bool PersistenceActor::hardLagReached() const {
  return undurableLag() >= config_.hard_max_undurable_revisions;
}

bool PersistenceActor::admissionAllowed() const {
  return opened_.load(std::memory_order_acquire) &&
         healthy_.load(std::memory_order_acquire) &&
         !admission_fault_.load(std::memory_order_acquire) &&
         !hardLagReached();
}

void PersistenceActor::failAdmission(std::string error) {
  enterTerminalFault(std::move(error));
}

PersistenceActorStatus PersistenceActor::status() const {
  PersistenceActorStatus result;
  result.latest_scene_revision = latestRevision();
  result.durable_scene_revision = durableRevision();
  result.undurable_revisions =
      result.latest_scene_revision >= result.durable_scene_revision
          ? result.latest_scene_revision - result.durable_scene_revision
          : 0;
  result.commit_queue = commit_queue_.stats();
  result.store_open = isOpen();
  result.healthy = healthy_.load(std::memory_order_acquire);
  result.graceful_shutdown_complete =
      graceful_shutdown_complete_.load(std::memory_order_acquire);
  result.last_acknowledged_revision =
      last_acknowledged_revision_.load(std::memory_order_acquire);
  result.commits_enqueued = commits_enqueued_.load(std::memory_order_relaxed);
  result.commits_rejected = commits_rejected_.load(std::memory_order_relaxed);
  result.successful_flushes =
      successful_flushes_.load(std::memory_order_relaxed);
  result.failed_flushes = failed_flushes_.load(std::memory_order_relaxed);

  if (result.store_open) {
    result.store_pending_commits = store_->watermarks().pending_commits;
  }
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    result.last_error = last_error_;
    if (!undurable_since_.empty()) {
      const Clock::time_point now = Clock::now();
      if (now > undurable_since_.front().second) {
        result.oldest_undurable_age =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now - undurable_since_.front().second);
      }
    }
  }
  {
    std::lock_guard<std::mutex> callback_lock(callback_mutex_);
    result.durability_ack_pending =
        static_cast<bool>(durability_ack_callback_) &&
        result.last_acknowledged_revision <
            result.durable_scene_revision;
  }

  result.soft_lag_reached =
      result.undurable_revisions >= config_.max_undurable_revisions;
  result.hard_lag_reached =
      result.undurable_revisions >=
      config_.hard_max_undurable_revisions;
  result.admission_allowed = result.store_open && result.healthy &&
                             !admission_fault_.load(std::memory_order_acquire) &&
                             !result.hard_lag_reached;
  return result;
}

bool PersistenceActor::waitUntilDurable(
    SceneRevision revision, std::chrono::milliseconds timeout) const {
  if (timeout < std::chrono::milliseconds::zero()) {
    timeout = std::chrono::milliseconds::zero();
  }
  std::unique_lock<std::mutex> lock(state_mutex_);
  return durable_cv_.wait_for(lock, timeout, [this, revision]() {
    return durable_scene_revision_.load(std::memory_order_acquire) >= revision;
  });
}

PersistenceActor::StoreEnqueueResult PersistenceActor::enqueueIntoStore(
    const PendingCommit& commit) {
  // BoundedChannel wakes its consumer before push() returns. Wait for the
  // producer's live-watermark publication so observers can never see the
  // durable watermark briefly overtake latest_scene_revision.
  {
    std::unique_lock<std::mutex> state_lock(state_mutex_);
    durable_cv_.wait(state_lock, [this, &commit]() {
      return latest_scene_revision_.load(std::memory_order_acquire) >=
             commit.snapshot.revision();
    });
  }

  SceneStoreStatus status =
      store_->enqueueCommit(commit.snapshot, commit.outbox_tasks,
                            commit.artifact_origins);
  if (status) {
    recordSuccess();
    return StoreEnqueueResult::kAccepted;
  }

  const SceneStoreWatermarks watermarks = store_->watermarks();
  if (watermarks.pending_commits > 0 &&
      watermarks.backpressure_required) {
    if (!flushStore(false)) {
      return StoreEnqueueResult::kRetryableFailure;
    }
    status = store_->enqueueCommit(commit.snapshot, commit.outbox_tasks,
                                   commit.artifact_origins);
    if (status) {
      recordSuccess();
      return StoreEnqueueResult::kAccepted;
    }
  }

  recordFailure("could not enqueue scene revision " +
                    std::to_string(commit.snapshot.revision()) + ": " + status.error,
                false);
  return StoreEnqueueResult::kFatalFailure;
}

bool PersistenceActor::flushStore(bool graceful) {
  const SceneStoreStatus status =
      graceful ? store_->gracefulFlush() : store_->flush();
  const SceneStoreWatermarks watermarks = store_->watermarks();

  // A gracefulFlush can fail in its WAL checkpoint after the transaction has
  // committed. Always publish the actual SceneStore watermark.
  if (watermarks.durable_scene_revision > durableRevision()) {
    advanceDurableWatermark(watermarks.durable_scene_revision);
  }
  const bool ack_delivered =
      deliverDurabilityAck(watermarks.durable_scene_revision);

  if (!status) {
    recordFailure("scene store flush failed: " + status.error, true);
    return false;
  }

  ++successful_flushes_;
  if (ack_delivered) {
    recordSuccess();
  }
  return true;
}

bool PersistenceActor::deliverDurabilityAck(SceneRevision revision) {
  if (revision == 0 ||
      revision <=
          last_acknowledged_revision_.load(std::memory_order_acquire)) {
    return true;
  }

  DurabilityAckCallback callback;
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callback = durability_ack_callback_;
  }
  if (!callback) {
    return true;
  }

  try {
    if (!callback(revision)) {
      recordFailure("durability acknowledgement was rejected for revision " +
                        std::to_string(revision),
                    false);
      return false;
    }
  } catch (const std::exception& error) {
    recordFailure("durability acknowledgement threw for revision " +
                      std::to_string(revision) + ": " + error.what(),
                  false);
    return false;
  } catch (...) {
    recordFailure("durability acknowledgement threw for revision " +
                      std::to_string(revision),
                  false);
    return false;
  }

  last_acknowledged_revision_.store(revision, std::memory_order_release);
  return true;
}

void PersistenceActor::advanceDurableWatermark(SceneRevision revision) {
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    const SceneRevision current =
        durable_scene_revision_.load(std::memory_order_acquire);
    if (revision <= current) {
      return;
    }
    durable_scene_revision_.store(revision, std::memory_order_release);
    while (!undurable_since_.empty() &&
           undurable_since_.front().first <= revision) {
      undurable_since_.pop_front();
    }
  }
  durable_cv_.notify_all();
}

void PersistenceActor::recordFailure(std::string error,
                                     bool flush_failure) {
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    last_error_ = std::move(error);
  }
  healthy_.store(false, std::memory_order_release);
  if (flush_failure) {
    ++failed_flushes_;
  }
  durable_cv_.notify_all();
}

void PersistenceActor::recordSuccess() {
  if (!admission_fault_.load(std::memory_order_acquire)) {
    healthy_.store(true, std::memory_order_release);
  }
}

void PersistenceActor::enterTerminalFault(std::string error) {
  admission_fault_.store(true, std::memory_order_release);
  recordFailure(std::move(error), false);
  // Reliable producers may currently be asleep on a full queue. A terminal
  // store failure cannot make space, so wake them with an explicit Stopped
  // result instead of participating in a shutdown cycle.
  commit_queue_.stop();
}

void PersistenceActor::run() {
  if (!opened_.load(std::memory_order_acquire)) {
    const SceneStoreStatus opened = open();
    if (!opened) {
      commit_queue_.stop();
      return;
    }
  }

  Clock::time_point next_flush = Clock::now() + config_.flush_period;
  std::optional<PendingCommit> in_flight;
  std::optional<Clock::time_point> continuous_failure_since;
  bool fatal_failure = false;

  const auto failureBecameTerminal = [this, &continuous_failure_since]() {
    const Clock::time_point now = Clock::now();
    if (!continuous_failure_since) {
      continuous_failure_since = now;
      return false;
    }
    return now - *continuous_failure_since >=
           config_.terminal_failure_timeout;
  };

  while (true) {
    if (!in_flight && commit_queue_.stopped() && commit_queue_.empty()) {
      break;
    }
    if (!in_flight) {
      PendingCommit commit;
      if (stopRequested()) {
        if (!commit_queue_.tryPop(&commit)) {
          break;
        }
        in_flight.emplace(std::move(commit));
      } else {
        const Clock::time_point now = Clock::now();
        const Clock::duration wait_duration =
            now < next_flush ? next_flush - now : Clock::duration::zero();
        if (commit_queue_.waitPopFor(&commit, wait_duration)) {
          in_flight.emplace(std::move(commit));
        }
      }
    }

    if (in_flight) {
      const StoreEnqueueResult enqueue_result =
          enqueueIntoStore(*in_flight);
      if (enqueue_result == StoreEnqueueResult::kAccepted) {
        in_flight.reset();
        continuous_failure_since.reset();
      } else if (enqueue_result == StoreEnqueueResult::kFatalFailure) {
        enterTerminalFault("terminal persistence enqueue failure: " +
                           status().last_error);
        fatal_failure = true;
        break;
      } else {
        if (stopRequested()) {
          break;
        }
        if (failureBecameTerminal()) {
          enterTerminalFault(
              "persistence store remained unavailable past terminal timeout: " +
              status().last_error);
          fatal_failure = true;
          break;
        }
        std::this_thread::sleep_for(kFlushFailureRetryDelay);
        continue;
      }
    }

    const SceneStoreWatermarks watermarks = store_->watermarks();
    const Clock::time_point now = Clock::now();
    const bool flush_due =
        stopRequested() || now >= next_flush ||
        watermarks.pending_commits >= config_.flush_batch_size ||
        undurableLag() >= config_.max_undurable_revisions ||
        watermarks.backpressure_required;
    if (flush_due) {
      if (flushStore(false)) {
        next_flush = Clock::now() + config_.flush_period;
        continuous_failure_since.reset();
      } else {
        if (stopRequested()) {
          break;
        }
        if (failureBecameTerminal()) {
          enterTerminalFault(
              "persistence flush remained unavailable past terminal timeout: " +
              status().last_error);
          fatal_failure = true;
          break;
        }
        std::this_thread::sleep_for(kFlushFailureRetryDelay);
        next_flush = Clock::now();
      }
    }
  }

  commit_queue_.stop();
  if (fatal_failure || in_flight || !commit_queue_.empty()) {
    // Preserve any valid contiguous prefix already admitted by SceneStore,
    // even though the complete live suffix cannot be declared graceful.
    std::string terminal_error;
    {
      std::lock_guard<std::mutex> state_lock(state_mutex_);
      terminal_error = last_error_;
    }
    (void)flushStore(true);
    // The best-effort final flush can add useful detail, but it must not erase
    // the terminal cause that released blocked reliable producers.
    {
      std::lock_guard<std::mutex> state_lock(state_mutex_);
      if (!terminal_error.empty() && last_error_ != terminal_error) {
        last_error_ = terminal_error + "; final flush: " + last_error_;
      }
    }
    graceful_shutdown_complete_.store(false, std::memory_order_release);
    durable_cv_.notify_all();
    return;
  }

  const bool flushed = flushStore(true);
  const bool complete =
      flushed && !admission_fault_.load(std::memory_order_acquire) &&
      commit_queue_.empty() && durableRevision() == latestRevision();
  graceful_shutdown_complete_.store(complete, std::memory_order_release);
  durable_cv_.notify_all();
}

void PersistenceActor::onStopRequested() {
  commit_queue_.stop();
  durable_cv_.notify_all();
}

}  // namespace roomie
