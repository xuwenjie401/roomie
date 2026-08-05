#pragma once

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <optional>
#include <filesystem>
#include <functional>
#include <chrono>
#include <string>
#include <vector>
#include <map>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/artifacts/online_snapshot_worker.hpp"
#include "roomie/dsg/object_graph.hpp"
#include "roomie/dsg/object_snapshot_remaker.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/snapshot_control_queue.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"
#include "roomie/scene/scene_reducer.hpp"

namespace roomie {

class InstanceMapThread : public WorkerThread, public InstanceStore {
 public:
  InstanceMapThread(ThreadSafeQueue<InferenceResponse>& response_queue,
                    const MapProjector& map_projector,
                    PipelineConfig config);

  bool enqueueDetections(InferenceResponse response) override;
  std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
  snapshotInstances() const override;
  std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
  snapshotTrackedInstances() const override;
  ObjectGraphSnapshot snapshotObjectGraph() const override;
  bool prepareSceneGraphForSave(ObjectGraphSnapshot* snapshot,
                                const std::filesystem::path& snapshot_image_dir,
                                const std::string& snapshot_uri_prefix,
                                std::string* error) const override;
  bool loadObjectGraphSnapshot(const ObjectGraphSnapshot& snapshot, std::string* error = nullptr);
  bool loadSceneSnapshot(const SceneSnapshot& snapshot,
                         std::string* error = nullptr);
  SceneSnapshot sceneSnapshot() const;
  using SceneCommandCompletion =
      std::function<void(const SceneApplyResult&)>;
  bool enqueueSceneCommand(SceneCommand command,
                           SceneCommandCompletion completion = {});
  std::optional<SceneApplyResult> applySceneCommandAndWait(
      SceneCommand command,
      std::chrono::milliseconds timeout = std::chrono::seconds(5));
  void setOnlineSnapshotWorker(OnlineSnapshotWorker* worker);
  std::size_t closeSnapshotControlsForShutdown();
  bool snapshotControlFaulted() const;
  bool requestDurabilityAck(SceneRevision revision);
  void setSceneCommitSink(
      std::function<bool(const SceneApplyResult&)> sink);
  // Evaluated on the reducer actor immediately before any command that would
  // create a durable content revision. A false result rejects the command
  // before ReducerCore mutates current state, closing admission TOCTOU races.
  void setSceneContentAdmission(std::function<bool()> admission);
  void setSceneCommitObserver(
      std::function<void(const SceneApplyResult&)> observer);
  bool waitUntilIdle(std::chrono::milliseconds timeout);
  SnapshotControlQueueStats snapshotControlStats() const;

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  void applyDetections(const InferenceResponse& response);
  bool commitReducerState(
      const InferenceResponse& response,
      const std::vector<InstanceObservation,
                        Eigen::aligned_allocator<InstanceObservation>>& observations);
  SceneApplyResult applyQueuedSceneCommand(SceneCommand command);
  SceneApplyResult rejectForPersistenceAdmission(
      const std::string& reason) const;
  bool sceneContentAdmissionAllowed() const;
  std::string sceneContentAdmissionRejectionReason() const;
  bool applyPendingDurabilityAck();
  SnapshotControlPushOutcome enqueueSnapshotControl(
      SnapshotControl control);
  bool flushPendingSnapshotControls();
  void publishReducerResult(const SceneApplyResult& result,
                            bool persist_content_commit);
  bool loadInitialSnapshot(const ObjectGraphSnapshot& graph,
                           const SceneSnapshot* restored_scene,
                           std::string* error);
  void refreshAssociationWorkingSet(const SceneSnapshot& snapshot);
  void applyFrozenInstanceSnapshotRemake(const InferenceResponse& response);
  std::optional<InstanceObservation> makeObservation(const InferenceResponse& response,
                                                     const RawDetection& detection) const;
  std::optional<std::size_t> findBestFrozenTrack(
      const InstanceObservation& observation) const;
  std::optional<std::size_t> findBestTrack(const InstanceObservation& observation,
                                           const std::vector<bool>& track_reserved) const;
  bool shouldRejectAsDuplicateOfConfirmed(const InstanceObservation& observation) const;
  bool createTrack(const InstanceObservation& observation);
  bool updateTrack(InstanceTrack* track,
                   const InstanceObservation& observation,
                   bool* geometry_update_suppressed = nullptr,
                   std::string* geometry_update_suppression_reason = nullptr);
  void ageUnmatchedTracks(const std::vector<bool>& track_matched,
                          TimeNanoseconds response_time_ns);
  std::size_t removeExpiredTentativeTracks(
      std::vector<int>* removed_track_ids = nullptr);
  std::size_t mergeDuplicateStableTracks(
      std::map<std::string, std::size_t>* small_duplicates_by_label = nullptr);
  bool maybePromoteOrUpdateObject(InstanceTrack* track);
  bool isPromotable(const InstanceTrack& track) const;

  ThreadSafeQueue<InferenceResponse>& response_queue_;
  struct QueuedSceneCommand {
    SceneCommand command;
    SceneCommandCompletion completion;
  };

  ThreadSafeQueue<QueuedSceneCommand> scene_command_queue_;
  const MapProjector& map_projector_;
  PipelineConfig config_;
  mutable std::mutex mutex_;
  int next_track_id_ = 0;
  std::uint64_t frame_index_ = 0;
  ObjectGraph object_graph_;
  ObjectSnapshotRemaker snapshot_remaker_;
  std::vector<InstanceTrack, Eigen::aligned_allocator<InstanceTrack>> tracks_;
  std::vector<std::pair<int, int>> pending_reducer_merges_;
  ReducerCore reducer_;
  std::shared_ptr<const SceneState> published_scene_state_;
  std::function<bool(const SceneApplyResult&)> scene_commit_sink_;
  std::function<bool()> scene_content_admission_;
  // Actor-owned terminal fence. Once one committed revision cannot enter the
  // ordered persistence stream, no later content revision can ever be made
  // contiguous, so all subsequent content commands are rejected pre-reducer.
  bool persistence_commit_failed_ = false;
  std::function<void(const SceneApplyResult&)> scene_commit_observer_;
  OnlineSnapshotWorker* online_snapshot_worker_ = nullptr;
  SnapshotControlQueue pending_snapshot_controls_;
  // Set only by the reducer actor when a distinct ownership transition cannot
  // enter the bounded control queue. Atomicity is only for diagnostics reads.
  std::atomic_bool snapshot_control_faulted_{false};
  std::atomic_bool processing_work_{false};
  std::atomic<SceneRevision> pending_durable_ack_{0};
  mutable std::mutex drain_mutex_;
  std::condition_variable drain_cv_;
};

}  // namespace roomie
