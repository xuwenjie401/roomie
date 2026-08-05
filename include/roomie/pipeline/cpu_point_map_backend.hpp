#pragma once

#include <atomic>
#include <mutex>

#include "roomie/pipeline/map_backend.hpp"

namespace roomie {

class CpuPointMapBackend : public MapBackend {
 public:
  explicit CpuPointMapBackend(PipelineConfig config);

  MapIntegrationResult integrateFrame(const FrameBundle& frame) override;
  SurfaceRefreshResult refreshSurface(const MapIntegrationResult& integration,
                                      bool force_full_rebuild) override;
  MapBackendSnapshot snapshot() const override;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> collectNearSurfaceVoxels(
      const RawDetection& detection) const override;

 private:
  void appendDepthFramePoints(const FrameBundle& frame);

  PipelineConfig config_;
  mutable std::mutex mutex_;
  WorldPointVector map_points_world_;
  std::atomic_uint64_t map_version_{0};
  std::atomic_uint64_t integrated_frames_{0};
};

}  // namespace roomie
