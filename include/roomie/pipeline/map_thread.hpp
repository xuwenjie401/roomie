#pragma once

#include <memory>
#include <optional>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/pipeline/map_backend.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

class MapThread : public WorkerThread, public MapProjector {
 public:
  MapThread(ThreadSafeQueue<MappingFrame>& mapping_queue, PipelineConfig config);
  ~MapThread() override;

  bool enqueueMappingFrame(MappingFrame frame) override;
  std::optional<PatchDepth> projectPatchDepth(const DetectionFrame& frame) override;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> collectNearSurfaceVoxels(
      const RawDetection& detection) const override;

  std::uint64_t mapVersion() const;
  MapBackendSnapshot debugSnapshot() const;

 protected:
  void run() override;

 private:
  PatchDepth projectWorldPointsToPatchDepth(const DetectionFrame& frame,
                                            const WorldPointVector& world_points,
                                            std::uint64_t map_version) const;

  ThreadSafeQueue<MappingFrame>& mapping_queue_;
  PipelineConfig config_;
  std::unique_ptr<MapBackend> map_backend_;
};

}  // namespace roomie
