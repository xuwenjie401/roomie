#pragma once

#include <chrono>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
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
  MapBackendSnapshot snapshotSurfacePoints() const override;
  std::shared_ptr<const GeometrySurfaceCache> geometrySurfaceCache() const override;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> collectNearSurfaceVoxels(
      const RawDetection& detection) const override;

  std::uint64_t mapVersion() const;
  MapBackendSnapshot debugSnapshot() const;

 protected:
  void run() override;

 private:
  MapBackendSnapshot timedBackendSnapshot(const MapBackendView* view) const;
  void maybeLogStatus();
  PatchDepth projectWorldPointsToPatchDepth(const DetectionFrame& frame,
                                            const WorldPointVector& world_points,
                                            std::uint64_t map_version,
                                            float min_depth_m,
                                            float max_depth_m) const;

  ThreadSafeQueue<MappingFrame>& mapping_queue_;
  PipelineConfig config_;
  std::unique_ptr<MapBackend> map_backend_;
  std::mutex status_mutex_;
  std::chrono::steady_clock::time_point last_status_log_time_ =
      std::chrono::steady_clock::now();
  std::atomic_uint64_t integrated_frames_{0};
  std::atomic_uint64_t projection_requests_{0};
  std::atomic_uint64_t projection_no_map_{0};
  std::atomic_int last_valid_patches_{0};
  std::atomic_int last_projected_points_{0};
  std::atomic_uint64_t last_projection_map_version_{0};
  std::atomic_uint64_t last_source_surface_points_{0};
  std::atomic_uint64_t last_source_tsdf_blocks_{0};
  std::atomic_uint64_t last_source_voxels_scanned_{0};
  std::atomic_uint64_t last_source_selected_blocks_{0};
  std::atomic_uint64_t last_source_cached_surface_points_{0};
  std::atomic_uint64_t last_surface_cache_rebuilds_{0};
  std::atomic<double> last_project_total_ms_{0.0};
  std::atomic<double> last_snapshot_ms_{0.0};
  std::atomic<double> last_surface_extract_ms_{0.0};
  std::atomic<double> last_cache_build_ms_{0.0};
  std::atomic<double> last_frustum_filter_ms_{0.0};
  std::atomic<double> last_projection_loop_ms_{0.0};
  std::atomic<double> last_projection_zbuffer_ms_{0.0};
};

}  // namespace roomie
