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
};

struct MapBackendSnapshot {
  WorldPointVector surface_points_world;
  std::vector<MapSurfacePoint, Eigen::aligned_allocator<MapSurfacePoint>> debug_surface_points;
  std::uint64_t map_version = 0;
  bool has_map = false;
};

class MapBackend {
 public:
  virtual ~MapBackend() = default;

  virtual void integrateFrame(const MappingFrame& frame) = 0;
  virtual MapBackendSnapshot snapshot() const = 0;
  virtual std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> collectNearSurfaceVoxels(
      const RawDetection& detection) const = 0;
  virtual void saveIfRequested() {}
};

std::unique_ptr<MapBackend> createMapBackend(const PipelineConfig& config);

}  // namespace roomie
