#pragma once

#include <atomic>
#include <mutex>
#include <optional>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

class MapThread : public WorkerThread, public MapProjector {
 public:
  MapThread(ThreadSafeQueue<MappingFrame>& mapping_queue, PipelineConfig config);

  bool enqueueMappingFrame(MappingFrame frame) override;
  std::optional<PatchDepth> projectPatchDepth(const DetectionFrame& frame) override;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> collectNearSurfaceVoxels(
      const RawDetection& detection) const override;

  std::uint64_t mapVersion() const;

 protected:
  void run() override;

 private:
  void integrateFrame(const MappingFrame& frame);

  ThreadSafeQueue<MappingFrame>& mapping_queue_;
  PipelineConfig config_;
  mutable std::mutex map_mutex_;
  std::atomic_uint64_t map_version_{0};
  std::atomic_uint64_t integrated_frames_{0};
};

}  // namespace roomie
