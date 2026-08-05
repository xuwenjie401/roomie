#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"
#include "roomie/scene/scene_snapshot.hpp"
#include "roomie/scene/scene_store.hpp"

namespace roomie {

struct PersistenceActorConfig {
  std::chrono::milliseconds flush_period{1000};
  std::size_t flush_batch_size = 16;
  std::size_t queue_capacity = 64;
  SceneRevision max_undurable_revisions = 32;
  SceneRevision hard_max_undurable_revisions = 64;
  // A continuously failing store becomes a terminal fault after this bound.
  // This releases reliable producers and makes shutdown finite while keeping
  // the already-durable prefix explicit in status().
  std::chrono::milliseconds terminal_failure_timeout{5000};
};

struct PersistenceActorStatus {
  SceneRevision latest_scene_revision = 0;
  SceneRevision durable_scene_revision = 0;
  SceneRevision undurable_revisions = 0;
  std::chrono::nanoseconds oldest_undurable_age{0};

  std::size_t store_pending_commits = 0;
  ChannelStats commit_queue;
  bool soft_lag_reached = false;
  bool hard_lag_reached = false;
  bool admission_allowed = true;
  bool store_open = false;
  bool healthy = true;
  bool graceful_shutdown_complete = false;

  SceneRevision last_acknowledged_revision = 0;
  bool durability_ack_pending = false;
  std::uint64_t commits_enqueued = 0;
  std::uint64_t commits_rejected = 0;
  std::uint64_t successful_flushes = 0;
  std::uint64_t failed_flushes = 0;
  std::string last_error;
};

// Asynchronous durable sidecar for the reducer's ordered immutable snapshots.
//
// enqueueCommit() is deliberately reliable and may block when the bounded
// channel is full. A hard-lag indication is meant to stop new perception
// admission before that happens; already-committed reducer output is never
// discarded. SceneStore remains the only SQLite owner and performs one WAL
// transaction for each actor flush batch.
class PersistenceActor : public WorkerThread {
 public:
  using DurabilityAckCallback = std::function<bool(SceneRevision)>;

  PersistenceActor(std::string database_path,
                   PersistenceActorConfig config = {});
  PersistenceActor(std::unique_ptr<SceneStore> store,
                   PersistenceActorConfig config = {});
  ~PersistenceActor() override;

  // Open is synchronous so startup can restore the last durable snapshot
  // before the reducer starts accepting live commands. It is idempotent at
  // the actor boundary.
  SceneStoreStatus open();
  bool isOpen() const;
  SceneRestoreResult restoreLatest() const;
  SceneRestoreResult restoreAt(SceneRevision revision) const;
  // May only be called after open() and before start(). Synchronizes the
  // actor's admission/durability watermarks with SceneStore's new prefix.
  SceneStoreStatus rewindTo(
      SceneRevision revision,
      bool retain_aligned_map_checkpoints = true);

  // Co-located durable artifact actors share the same internally serialized
  // SceneStore for fenced outbox and embedding-record operations. The actor
  // retains ownership; callers must not close it and must stop before this
  // PersistenceActor is destroyed.
  SceneStore* durableArtifactStore() const;

  PushResult<SceneSnapshot> enqueueCommit(
      SceneSnapshot snapshot,
      std::vector<DurableTaskSpec> outbox_tasks = {},
      std::vector<DurableArtifactOrigin> artifact_origins = {});

  void setDurabilityAckCallback(DurabilityAckCallback callback);

  SceneRevision latestRevision() const;
  SceneRevision durableRevision() const;
  bool hardLagReached() const;
  bool admissionAllowed() const;
  void failAdmission(std::string error);
  PersistenceActorStatus status() const;

  bool waitUntilDurable(SceneRevision revision,
                        std::chrono::milliseconds timeout) const;

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  using Clock = std::chrono::steady_clock;
  struct PendingCommit {
    SceneSnapshot snapshot;
    std::vector<DurableTaskSpec> outbox_tasks;
    std::vector<DurableArtifactOrigin> artifact_origins;
  };
  enum class StoreEnqueueResult {
    kAccepted,
    kRetryableFailure,
    kFatalFailure,
  };

  static PersistenceActorConfig validateConfig(
      PersistenceActorConfig config);
  static std::size_t storePendingLimit(
      const PersistenceActorConfig& config);
  static PushResult<SceneSnapshot> rejectedResult(SceneSnapshot snapshot);

  StoreEnqueueResult enqueueIntoStore(const PendingCommit& commit);
  bool flushStore(bool graceful);
  bool deliverDurabilityAck(SceneRevision revision);
  void advanceDurableWatermark(SceneRevision revision);
  void recordFailure(std::string error, bool flush_failure);
  void recordSuccess();
  void enterTerminalFault(std::string error);
  SceneRevision undurableLag() const;

  PersistenceActorConfig config_;
  std::unique_ptr<SceneStore> store_;
  ThreadSafeQueue<PendingCommit> commit_queue_;

  mutable std::mutex admission_mutex_;
  mutable std::mutex state_mutex_;
  mutable std::condition_variable durable_cv_;
  std::deque<std::pair<SceneRevision, Clock::time_point>> undurable_since_;
  std::string last_error_;

  mutable std::mutex callback_mutex_;
  DurabilityAckCallback durability_ack_callback_;

  std::atomic_bool opened_{false};
  std::atomic_bool healthy_{true};
  std::atomic_bool admission_fault_{false};
  std::atomic_bool graceful_shutdown_complete_{false};
  std::atomic<SceneRevision> latest_scene_revision_{0};
  std::atomic<SceneRevision> durable_scene_revision_{0};
  std::atomic<SceneRevision> last_acknowledged_revision_{0};
  std::atomic_uint64_t commits_enqueued_{0};
  std::atomic_uint64_t commits_rejected_{0};
  std::atomic_uint64_t successful_flushes_{0};
  std::atomic_uint64_t failed_flushes_{0};
};

}  // namespace roomie
