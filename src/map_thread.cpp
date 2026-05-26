#include "roomie/pipeline/map_thread.hpp"

#include <chrono>

namespace roomie {

MapThread::MapThread(ThreadSafeQueue<MappingFrame>& mapping_queue, PipelineConfig config)
    : WorkerThread("map_thread"),
      mapping_queue_(mapping_queue),
      config_(std::move(config)) {}

bool MapThread::enqueueMappingFrame(MappingFrame frame) {
  return mapping_queue_.pushDropOldest(std::move(frame));
}

std::optional<PatchDepth> MapThread::projectPatchDepth(const DetectionFrame& frame) {
  (void)frame;
  std::lock_guard<std::mutex> lock(map_mutex_);
  if (map_version_.load() == 0 && !config_.load_map) {
    return std::nullopt;
  }

  PatchDepth result;
  result.map_version = map_version_.load();

  // TODO: replace with nvblox near-surface voxel projection. This placeholder
  // intentionally returns zero coverage so the detection bridge skips frames
  // until a real projector is wired in.
  return result;
}

std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> MapThread::collectNearSurfaceVoxels(
    const RawDetection& detection) const {
  (void)detection;
  std::lock_guard<std::mutex> lock(map_mutex_);
  // TODO: collect near-surface voxel refs whose centers fall inside the OBB.
  return {};
}

std::uint64_t MapThread::mapVersion() const { return map_version_.load(); }

void MapThread::run() {
  while (!stopRequested()) {
    MappingFrame frame;
    if (!mapping_queue_.waitPopFor(&frame, std::chrono::milliseconds(50))) {
      continue;
    }
    integrateFrame(frame);
  }
}

void MapThread::integrateFrame(const MappingFrame& frame) {
  (void)frame;
  std::lock_guard<std::mutex> lock(map_mutex_);
  // TODO: integrate RGB-D into nvblox color TSDF. Keep all mutation in this
  // thread so projectPatchDepth can use a short read/lock section later.
  ++integrated_frames_;
  ++map_version_;
}

}  // namespace roomie
