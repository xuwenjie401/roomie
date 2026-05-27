#include "roomie/pipeline/instance_map_thread.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>

#include "roomie/utils/run_logger.hpp"

namespace roomie {

InstanceMapThread::InstanceMapThread(ThreadSafeQueue<InferenceResponse>& response_queue,
                                     const MapProjector& map_projector,
                                     PipelineConfig config)
    : WorkerThread("instance_map_thread"),
      response_queue_(response_queue),
      map_projector_(map_projector),
      config_(std::move(config)) {}

bool InstanceMapThread::enqueueDetections(InferenceResponse response) {
  return response_queue_.pushDropOldest(std::move(response));
}

std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
InstanceMapThread::snapshotInstances() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return instances_;
}

void InstanceMapThread::run() {
  while (!stopRequested()) {
    InferenceResponse response;
    if (!response_queue_.waitPopFor(&response, std::chrono::milliseconds(50))) {
      continue;
    }
    if (!response.ok) {
      continue;
    }
    applyDetections(response);
  }
}

void InstanceMapThread::applyDetections(const InferenceResponse& response) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::size_t before = instances_.size();
  for (const RawDetection& detection : response.detections) {
    InstanceRecord instance;
    instance.track_id = next_track_id_++;
    instance.semantic_id = detection.semantic_id;
    instance.label = detection.label;
    instance.center_world = detection.center_world;
    instance.size_m = detection.size_m;
    instance.yaw_rad = detection.yaw_rad;
    instance.confidence = 0.5f * (detection.score_2d + detection.score_3d);
    instance.support_count = 1;
    instance.first_seen_ns = response.time_ns;
    instance.last_seen_ns = response.time_ns;
    instance.source_cameras.push_back(response.camera_id);
    instance.near_surface_voxels = map_projector_.collectNearSurfaceVoxels(detection);
    instances_.push_back(std::move(instance));
  }

  if (!response.detections.empty()) {
    std::ostringstream stream;
    stream << "applied camera=" << response.camera_id
           << " t=" << response.time_ns
           << " added=" << response.detections.size()
           << " total=" << instances_.size()
           << " before=" << before;
    RunLogger::logGlobal("instance_map", stream.str());
  }

  // TODO: replace append-only behavior with OBB IoU matching, track aging, and
  // duplicate merge once the raw detection schema is stable.
}

}  // namespace roomie
