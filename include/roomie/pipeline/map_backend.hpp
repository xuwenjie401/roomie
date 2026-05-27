#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/types.hpp"

namespace roomie {

using WorldPointVector =
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;

struct MapSurfacePoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3f position_world = Eigen::Vector3f::Zero();
  std::uint8_t r = 160;
  std::uint8_t g = 160;
  std::uint8_t b = 160;
  float intensity = 0.0f;
  float weight = 0.0f;
  VoxelRef voxel_ref;
  bool has_voxel_ref = false;
};

struct GeometrySurfaceCache {
  std::vector<MapSurfacePoint, Eigen::aligned_allocator<MapSurfacePoint>> surface_points;
  std::uint64_t map_version = 0;
  std::uint64_t cache_rebuilds = 0;
  std::uint64_t tsdf_blocks = 0;
  std::uint64_t surface_voxels_scanned = 0;
  bool has_map = false;
};

struct MapBackendSnapshot {
  WorldPointVector surface_points_world;
  std::vector<MapSurfacePoint, Eigen::aligned_allocator<MapSurfacePoint>> debug_surface_points;
  std::uint64_t map_version = 0;
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
  bool view_filtered = false;
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

  virtual void integrateFrame(const MappingFrame& frame) = 0;
  virtual MapBackendSnapshot snapshot() const = 0;
  virtual MapBackendSnapshot snapshotForView(const MapBackendView& view) const {
    (void)view;
    return snapshot();
  }
  virtual std::shared_ptr<const GeometrySurfaceCache> geometrySurfaceCache() const = 0;
  virtual std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> collectNearSurfaceVoxels(
      const RawDetection& detection) const = 0;
  virtual void saveIfRequested() {}
};

std::unique_ptr<MapBackend> createMapBackend(const PipelineConfig& config);

}  // namespace roomie
