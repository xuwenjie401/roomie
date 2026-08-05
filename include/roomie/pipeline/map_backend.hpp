#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/map_checkpoint.hpp"
#include "roomie/pipeline/surface_snapshot.hpp"
#include "roomie/pipeline/types.hpp"

namespace roomie {

struct MapBackendSnapshot {
  WorldPointVector surface_points_world;
  std::vector<MapSurfacePoint, Eigen::aligned_allocator<MapSurfacePoint>> debug_surface_points;
  std::uint64_t map_version = 0;
  std::uint64_t latest_map_version = 0;
  std::uint64_t tsdf_blocks = 0;
  std::uint64_t surface_voxels_scanned = 0;
  std::uint64_t selected_blocks = 0;
  std::uint64_t cached_surface_points = 0;
  std::uint64_t cache_rebuilds = 0;
  double snapshot_ms = 0.0;
  double surface_extract_ms = 0.0;
  double cache_build_ms = 0.0;
  double frustum_filter_ms = 0.0;
  bool has_map = false;
  bool surface_cache_ready = false;
  bool cache_dirty = false;
  bool view_filtered = false;
};

struct MapIntegrationResult {
  bool success = false;
  bool map_changed = false;
  std::uint64_t map_revision = 0;
  TimeNanoseconds integrated_through_ns = 0;
  std::vector<BlockIndex> updated_blocks;
  bool updated_blocks_complete = false;
  std::string error;
};

struct SurfaceRefreshResult {
  bool success = false;
  bool full_rebuild = true;
  std::uint64_t source_map_revision = 0;
  std::vector<SurfaceBlockPtr> blocks;
  std::vector<BlockIndex> removed_blocks;
  MapBackendSnapshot diagnostics;
  std::string error;
};

struct MapBackendView {
  CameraIntrinsics intrinsics;
  Eigen::Isometry3f T_world_camera = Eigen::Isometry3f::Identity();
  float min_depth_m = 0.1f;
  float max_depth_m = 10.0f;
};

class MapBackend {
 public:
  virtual ~MapBackend() = default;

  virtual MapIntegrationResult integrateFrame(const FrameBundle& frame) = 0;
  // Only the MapActor may call refreshSurface(). Reader-facing methods below
  // return the last published cache and must not trigger a rebuild.
  virtual SurfaceRefreshResult refreshSurface(
      const MapIntegrationResult& integration,
      bool force_full_rebuild) = 0;
  virtual MapBackendSnapshot snapshot() const = 0;
  virtual MapBackendSnapshot snapshotForView(const MapBackendView& view) const {
    (void)view;
    return snapshot();
  }
  virtual std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> collectNearSurfaceVoxels(
      const RawDetection& detection) const = 0;
  virtual MapCheckpointOperationResult loadCheckpoint(
      const MapCheckpointManifest* expected_manifest) {
    (void)expected_manifest;
    MapCheckpointOperationResult result;
    result.disposition = MapCheckpointDisposition::kUnsupported;
    result.error = "map backend does not support persistent checkpoints";
    return result;
  }
  virtual MapCheckpointOperationResult saveCheckpoint(
      const MapStamp& map_stamp) {
    (void)map_stamp;
    MapCheckpointOperationResult result;
    result.disposition = MapCheckpointDisposition::kUnsupported;
    result.error = "map backend does not support persistent checkpoints";
    return result;
  }
};

std::unique_ptr<MapBackend> createMapBackend(const PipelineConfig& config);

}  // namespace roomie
