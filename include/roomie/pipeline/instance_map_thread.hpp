#pragma once

#include <mutex>
#include <optional>
#include <vector>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/dsg/object_graph.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"

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
  bool loadObjectGraphSnapshot(const ObjectGraphSnapshot& snapshot, std::string* error = nullptr);

 protected:
  void run() override;

 private:
  void applyDetections(const InferenceResponse& response);
  std::optional<InstanceObservation> makeObservation(const InferenceResponse& response,
                                                     const RawDetection& detection) const;
  std::optional<std::size_t> findBestTrack(const InstanceObservation& observation,
                                           const std::vector<bool>& track_reserved) const;
  bool shouldRejectAsDuplicateOfConfirmed(const InstanceObservation& observation) const;
  void createTrack(const InstanceObservation& observation);
  void updateTrack(InstanceTrack* track, const InstanceObservation& observation);
  void ageUnmatchedTracks(const std::vector<bool>& track_matched,
                          TimeNanoseconds response_time_ns);
  std::size_t removeExpiredTentativeTracks();
  std::size_t mergeDuplicateStableTracks();
  std::size_t applyGeometryMaintenance(const MapBackendSnapshot& map_snapshot,
                                       TimeNanoseconds now_ns,
                                       std::size_t* suppressed,
                                       std::size_t* recovered,
                                       std::size_t* deleted_empty);
  std::size_t applyGeometryMaintenance(const GeometrySurfaceCache& surface_cache,
                                       TimeNanoseconds now_ns,
                                       std::size_t* suppressed,
                                       std::size_t* recovered,
                                       std::size_t* deleted_empty);
  void removeTrackAndObjectByObjectId(int object_id);
  void maybePromoteOrUpdateObject(InstanceTrack* track);
  bool isPromotable(const InstanceTrack& track) const;
  bool shouldRunGeometryMaintenance(TimeNanoseconds now_ns) const;

  ThreadSafeQueue<InferenceResponse>& response_queue_;
  const MapProjector& map_projector_;
  PipelineConfig config_;
  mutable std::mutex mutex_;
  int next_track_id_ = 0;
  std::uint64_t frame_index_ = 0;
  TimeNanoseconds last_geometry_maintenance_ns_ = 0;
  ObjectGraph object_graph_;
  std::vector<InstanceTrack, Eigen::aligned_allocator<InstanceTrack>> tracks_;
};

}  // namespace roomie
