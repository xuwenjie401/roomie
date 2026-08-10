#include "roomie/pipeline/cpu_point_map_backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <utility>

namespace roomie {
namespace {

constexpr std::size_t kMaxPlaceholderMapPoints = 200000;

bool hasUsableIntrinsics(const CameraIntrinsics& intrinsics) {
  return intrinsics.fx > 0.0f && intrinsics.fy > 0.0f;
}

bool isRobotMaskedPixel(const ImageBuffer& mask,
                        int x,
                        int y,
                        int image_width,
                        int image_height) {
  if (mask.empty() || mask.width <= 0 || mask.height <= 0 || mask.channels <= 0) {
    return false;
  }
  const int mx = std::clamp(
      static_cast<int>((static_cast<float>(x) + 0.5f) *
                       static_cast<float>(mask.width) / static_cast<float>(image_width)),
      0,
      mask.width - 1);
  const int my = std::clamp(
      static_cast<int>((static_cast<float>(y) + 0.5f) *
                       static_cast<float>(mask.height) / static_cast<float>(image_height)),
      0,
      mask.height - 1);
  const std::size_t offset =
      (static_cast<std::size_t>(my) * static_cast<std::size_t>(mask.width) +
       static_cast<std::size_t>(mx)) *
      static_cast<std::size_t>(mask.channels);
  if (offset >= mask.data.size()) {
    return false;
  }
  return mask.data[offset] > 0U;
}

}  // namespace

CpuPointMapBackend::CpuPointMapBackend(PipelineConfig config)
    : config_(std::move(config)) {}

MapIntegrationResult CpuPointMapBackend::integrateFrame(const FrameBundle& frame) {
  MapIntegrationResult result;
  if (config_.freeze_tsdf_map) {
    result.success = true;
    result.map_revision = map_version_.load();
    result.integrated_through_ns = frame.provenance.sensor_time_ns;
    return result;
  }
  if (!frame.depth || frame.depth->empty() ||
      !hasUsableIntrinsics(frame.intrinsics)) {
    result.error = "CPU map integration requires depth and valid intrinsics";
    return result;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  appendDepthFramePoints(frame);
  ++integrated_frames_;
  ++map_version_;
  result.success = true;
  result.map_changed = true;
  result.map_revision = map_version_.load();
  result.integrated_through_ns = frame.provenance.sensor_time_ns;
  result.updated_blocks_complete = false;
  return result;
}

SurfaceRefreshResult CpuPointMapBackend::refreshSurface(
    const MapIntegrationResult& integration,
    bool force_full_rebuild) {
  (void)force_full_rebuild;
  std::lock_guard<std::mutex> lock(mutex_);
  SurfaceRefreshResult result;
  result.success = integration.success || config_.freeze_tsdf_map;
  result.full_rebuild = true;
  result.source_map_revision = map_version_.load();
  result.diagnostics.map_version = result.source_map_revision;
  result.diagnostics.latest_map_version = result.source_map_revision;
  result.diagnostics.has_map = !map_points_world_.empty();
  result.diagnostics.surface_cache_ready = true;
  result.diagnostics.cached_surface_points = map_points_world_.size();
  result.diagnostics.selected_blocks = map_points_world_.empty() ? 0U : 1U;
  result.diagnostics.cache_rebuilds = integrated_frames_.load();
  result.diagnostics.surface_points_world = map_points_world_;
  result.diagnostics.debug_surface_points.reserve(map_points_world_.size());

  if (map_points_world_.empty()) {
    return result;
  }

  Eigen::Vector3f min_point =
      Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
  Eigen::Vector3f max_point =
      Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
  MapSurfacePointVector points;
  points.reserve(map_points_world_.size());
  for (const Eigen::Vector3f& point_world : map_points_world_) {
    MapSurfacePoint point;
    point.position_world = point_world;
    point.weight = 1.0f;
    points.push_back(point);
    result.diagnostics.debug_surface_points.push_back(point);
    min_point = min_point.cwiseMin(point_world);
    max_point = max_point.cwiseMax(point_world);
  }
  constexpr float kBoundsPadding = 1.0e-4f;
  min_point.array() -= kBoundsPadding;
  max_point.array() += kBoundsPadding;
  result.blocks.push_back(makeSurfaceBlock(
      BlockIndex(0, 0, 0), SurfaceAabb(min_point, max_point), std::move(points)));
  return result;
}

MapBackendSnapshot CpuPointMapBackend::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  MapBackendSnapshot snapshot;
  snapshot.map_version = map_version_.load();
  snapshot.latest_map_version = snapshot.map_version;
  snapshot.has_map = !map_points_world_.empty();
  snapshot.surface_points_world = map_points_world_;
  snapshot.debug_surface_points.reserve(map_points_world_.size());
  for (const Eigen::Vector3f& point : map_points_world_) {
    MapSurfacePoint surface_point;
    surface_point.position_world = point;
    surface_point.r = 160;
    surface_point.g = 160;
    surface_point.b = 160;
    surface_point.intensity = 0.0f;
    surface_point.weight = 1.0f;
    snapshot.debug_surface_points.push_back(surface_point);
  }
  return snapshot;
}

std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>>
CpuPointMapBackend::collectNearSurfaceVoxels(const RawDetection& detection) const {
  (void)detection;
  std::lock_guard<std::mutex> lock(mutex_);
  // The CPU placeholder does not have stable voxel ids. The nvblox backend
  // returns real block/voxel references for this method.
  return {};
}

void CpuPointMapBackend::appendDepthFramePoints(const FrameBundle& frame) {
  if (!frame.depth || frame.depth->empty() ||
      !hasUsableIntrinsics(frame.intrinsics) ||
      map_points_world_.size() >= kMaxPlaceholderMapPoints) {
    return;
  }

  const int width = frame.depth->width;
  const int height = frame.depth->height;
  const std::size_t depth_count = frame.depth->depth_m.size();
  const std::size_t pixel_count =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (width <= 0 || height <= 0 || depth_count < pixel_count) {
    return;
  }

  const std::size_t remaining_capacity = kMaxPlaceholderMapPoints - map_points_world_.size();
  const std::size_t stride =
      std::max<std::size_t>(1, (pixel_count + remaining_capacity - 1) / remaining_capacity);
  const float max_integration_depth =
      config_.max_integration_distance_m > 0.0f
          ? std::min(config_.depth_max_m, config_.max_integration_distance_m)
          : config_.depth_max_m;

  for (std::size_t linear = 0;
       linear < pixel_count && map_points_world_.size() < kMaxPlaceholderMapPoints;
       linear += stride) {
    const float z = frame.depth->depth_m[linear];
    if (!std::isfinite(z) || z < config_.depth_min_m || z > max_integration_depth) {
      continue;
    }

    const int y = static_cast<int>(linear / static_cast<std::size_t>(width));
    const int x = static_cast<int>(linear % static_cast<std::size_t>(width));
    if (frame.robot_mask &&
        isRobotMaskedPixel(*frame.robot_mask, x, y, width, height)) {
      continue;
    }
    const Eigen::Vector3f point_camera(
        (static_cast<float>(x) - frame.intrinsics.cx) * z / frame.intrinsics.fx,
        (static_cast<float>(y) - frame.intrinsics.cy) * z / frame.intrinsics.fy, z);
    map_points_world_.push_back(frame.T_world_camera * point_camera);
  }
}

}  // namespace roomie
