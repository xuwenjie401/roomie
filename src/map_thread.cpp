#include "roomie/pipeline/map_thread.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "roomie/utils/run_logger.hpp"

namespace roomie {
namespace {

constexpr int kMaxZBufferScale = 16;
constexpr int kMaxZBufferSplatRadiusCells = 2;
constexpr int kMaxZBufferCellsPerPatch = kMaxZBufferScale * kMaxZBufferScale;

bool hasUsableIntrinsics(const CameraIntrinsics& intrinsics) {
  return intrinsics.fx > 0.0f && intrinsics.fy > 0.0f;
}

CameraIntrinsics scaledIntrinsicsForBoxerInput(const CameraIntrinsics& intrinsics,
                                               const ImageBuffer& rgb,
                                               int target_size) {
  CameraIntrinsics source = intrinsics;
  if ((source.width <= 0 || source.height <= 0) && rgb.width > 0 && rgb.height > 0) {
    source.width = rgb.width;
    source.height = rgb.height;
  }
  return source.scaledTo(target_size, target_size);
}

double elapsedMs(std::chrono::steady_clock::time_point start,
                 std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

float frontQuantileDepth(std::array<float, kMaxZBufferCellsPerPatch>* values,
                         int count,
                         float quantile) {
  if (count <= 0) {
    return -1.0f;
  }
  std::sort(values->begin(), values->begin() + count);
  const int index = std::clamp(
      static_cast<int>(std::floor(std::clamp(quantile, 0.0f, 1.0f) *
                                  static_cast<float>(count - 1))),
      0,
      count - 1);
  return (*values)[static_cast<std::size_t>(index)];
}

}  // namespace

MapThread::MapThread(ThreadSafeQueue<MappingFrame>& mapping_queue, PipelineConfig config)
    : WorkerThread("map_thread"),
      mapping_queue_(mapping_queue),
      config_(std::move(config)),
      map_backend_(createMapBackend(config_)) {}

MapThread::~MapThread() {
  stop();
  if (map_backend_) {
    map_backend_->saveIfRequested();
  }
}

bool MapThread::enqueueMappingFrame(MappingFrame frame) {
  return mapping_queue_.pushDropOldest(std::move(frame));
}

std::optional<PatchDepth> MapThread::projectPatchDepth(const DetectionFrame& frame) {
  ++projection_requests_;
  const auto project_total_start = std::chrono::steady_clock::now();
  const CameraIntrinsics intrinsics_960 =
      scaledIntrinsicsForBoxerInput(frame.intrinsics, frame.rgb, config_.boxer_input_size);
  MapBackendView view;
  view.intrinsics = intrinsics_960;
  view.T_world_camera = frame.T_world_camera;
  view.min_depth_m = config_.depth_min_m;
  view.max_depth_m =
      config_.patch_depth_max_m > 0.0f ? config_.patch_depth_max_m : config_.depth_max_m;
  MapBackendSnapshot snapshot = timedBackendSnapshot(&view);
  if (!snapshot.has_map) {
    ++projection_no_map_;
    maybeLogStatus();
    return std::nullopt;
  }
  PatchDepth patch_depth = projectWorldPointsToPatchDepth(
      frame,
      snapshot.surface_points_world,
      snapshot.map_version,
      view.min_depth_m,
      view.max_depth_m);
  patch_depth.source_surface_points =
      static_cast<std::uint64_t>(snapshot.surface_points_world.size());
  patch_depth.source_tsdf_blocks = snapshot.tsdf_blocks;
  patch_depth.source_voxels_scanned = snapshot.surface_voxels_scanned;
  patch_depth.source_selected_blocks = snapshot.selected_blocks;
  patch_depth.source_cached_surface_points = snapshot.cached_surface_points;
  patch_depth.surface_cache_rebuilds = snapshot.cache_rebuilds;
  patch_depth.snapshot_ms = snapshot.snapshot_ms;
  patch_depth.surface_extract_ms = snapshot.surface_extract_ms;
  patch_depth.cache_build_ms = snapshot.cache_build_ms;
  patch_depth.frustum_filter_ms = snapshot.frustum_filter_ms;
  patch_depth.surface_cache_ready = snapshot.surface_cache_ready;
  patch_depth.view_filtered = snapshot.view_filtered;
  patch_depth.project_total_ms =
      elapsedMs(project_total_start, std::chrono::steady_clock::now());
  last_valid_patches_ = patch_depth.valid_patches;
  last_projected_points_ = patch_depth.projected_points;
  last_projection_map_version_ = patch_depth.map_version;
  last_source_surface_points_ = patch_depth.source_surface_points;
  last_source_tsdf_blocks_ = patch_depth.source_tsdf_blocks;
  last_source_voxels_scanned_ = patch_depth.source_voxels_scanned;
  last_source_selected_blocks_ = patch_depth.source_selected_blocks;
  last_source_cached_surface_points_ = patch_depth.source_cached_surface_points;
  last_surface_cache_rebuilds_ = patch_depth.surface_cache_rebuilds;
  last_project_total_ms_.store(patch_depth.project_total_ms);
  last_snapshot_ms_.store(patch_depth.snapshot_ms);
  last_surface_extract_ms_.store(patch_depth.surface_extract_ms);
  last_cache_build_ms_.store(patch_depth.cache_build_ms);
  last_frustum_filter_ms_.store(patch_depth.frustum_filter_ms);
  last_projection_loop_ms_.store(patch_depth.projection_loop_ms);
  last_projection_zbuffer_ms_.store(patch_depth.projection_zbuffer_ms);
  maybeLogStatus();
  return patch_depth;
}

PatchDepth MapThread::projectWorldPointsToPatchDepth(const DetectionFrame& frame,
                                                     const WorldPointVector& world_points,
                                                     std::uint64_t map_version,
                                                     float min_depth_m,
                                                     float max_depth_m) const {
  PatchDepth result;
  result.map_version = map_version;

  const CameraIntrinsics intrinsics_960 =
      scaledIntrinsicsForBoxerInput(frame.intrinsics, frame.rgb, config_.boxer_input_size);
  if (!hasUsableIntrinsics(intrinsics_960) || world_points.empty()) {
    return result;
  }
  if (max_depth_m <= min_depth_m) {
    max_depth_m = config_.depth_max_m;
  }

  const Eigen::Isometry3f T_camera_world = frame.T_world_camera.inverse();
  const int zbuffer_scale =
      std::clamp(config_.patch_depth_zbuffer_scale, 1, kMaxZBufferScale);
  const int splat_radius =
      std::clamp(config_.patch_depth_zbuffer_splat_radius_cells,
                 0,
                 kMaxZBufferSplatRadiusCells);
  const int max_cells_per_patch = zbuffer_scale * zbuffer_scale;
  const int min_cells_per_patch =
      std::clamp(config_.patch_depth_zbuffer_min_cells_per_patch,
                 1,
                 max_cells_per_patch);
  const int zbuffer_cols = PatchDepth::kCols * zbuffer_scale;
  const int zbuffer_rows = PatchDepth::kRows * zbuffer_scale;
  const float zbuffer_cell_width_px =
      static_cast<float>(config_.boxer_input_size) / static_cast<float>(zbuffer_cols);
  const float zbuffer_cell_height_px =
      static_cast<float>(config_.boxer_input_size) / static_cast<float>(zbuffer_rows);
  constexpr float kInfinity = std::numeric_limits<float>::infinity();
  std::vector<float> z_buffer(static_cast<std::size_t>(zbuffer_rows * zbuffer_cols),
                              kInfinity);

  const auto projection_loop_start = std::chrono::steady_clock::now();
  for (const Eigen::Vector3f& point_world : world_points) {
    if (!point_world.allFinite()) {
      continue;
    }

    const Eigen::Vector3f point_camera = T_camera_world * point_world;
    const float z = point_camera.z();
    if (!std::isfinite(z) || z < min_depth_m || z > max_depth_m) {
      continue;
    }

    const float u = intrinsics_960.fx * point_camera.x() / z + intrinsics_960.cx;
    const float v = intrinsics_960.fy * point_camera.y() / z + intrinsics_960.cy;
    if (!std::isfinite(u) || !std::isfinite(v) || u < 0.0f || v < 0.0f ||
        u >= static_cast<float>(config_.boxer_input_size) ||
        v >= static_cast<float>(config_.boxer_input_size)) {
      continue;
    }

    const int center_col =
        std::clamp(static_cast<int>(u / zbuffer_cell_width_px), 0, zbuffer_cols - 1);
    const int center_row =
        std::clamp(static_cast<int>(v / zbuffer_cell_height_px), 0, zbuffer_rows - 1);
    const int row_begin = std::max(0, center_row - splat_radius);
    const int row_end = std::min(zbuffer_rows - 1, center_row + splat_radius);
    const int col_begin = std::max(0, center_col - splat_radius);
    const int col_end = std::min(zbuffer_cols - 1, center_col + splat_radius);
    for (int row = row_begin; row <= row_end; ++row) {
      for (int col = col_begin; col <= col_end; ++col) {
        float& current =
            z_buffer[static_cast<std::size_t>(row * zbuffer_cols + col)];
        if (z < current) {
          current = z;
        }
      }
    }
    ++result.projected_points;
  }
  result.projection_loop_ms =
      elapsedMs(projection_loop_start, std::chrono::steady_clock::now());

  const auto zbuffer_start = std::chrono::steady_clock::now();
  for (int patch_row = 0; patch_row < PatchDepth::kRows; ++patch_row) {
    for (int patch_col = 0; patch_col < PatchDepth::kCols; ++patch_col) {
      std::array<float, kMaxZBufferCellsPerPatch> cell_depths{};
      int cell_count = 0;
      const int row_begin = patch_row * zbuffer_scale;
      const int row_end = row_begin + zbuffer_scale;
      const int col_begin = patch_col * zbuffer_scale;
      const int col_end = col_begin + zbuffer_scale;
      for (int row = row_begin; row < row_end; ++row) {
        for (int col = col_begin; col < col_end; ++col) {
          const float depth =
              z_buffer[static_cast<std::size_t>(row * zbuffer_cols + col)];
          if (!std::isfinite(depth)) {
            continue;
          }
          cell_depths[static_cast<std::size_t>(cell_count++)] = depth;
        }
      }
      if (cell_count < min_cells_per_patch) {
        continue;
      }
      const int patch_index = patch_row * PatchDepth::kCols + patch_col;
      result.values[static_cast<std::size_t>(patch_index)] =
          frontQuantileDepth(&cell_depths,
                             cell_count,
                             config_.patch_depth_zbuffer_front_quantile);
      ++result.valid_patches;
    }
  }
  result.projection_zbuffer_ms =
      elapsedMs(zbuffer_start, std::chrono::steady_clock::now());

  return result;
}

std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> MapThread::collectNearSurfaceVoxels(
    const RawDetection& detection) const {
  return map_backend_->collectNearSurfaceVoxels(detection);
}

MapBackendSnapshot MapThread::snapshotSurfacePoints() const {
  MapBackendView view;
  view.max_depth_m = 0.0f;
  return timedBackendSnapshot(&view);
}

std::shared_ptr<const GeometrySurfaceCache> MapThread::geometrySurfaceCache() const {
  return map_backend_->geometrySurfaceCache();
}

std::uint64_t MapThread::mapVersion() const {
  return map_backend_->snapshot().map_version;
}

MapBackendSnapshot MapThread::debugSnapshot() const {
  return timedBackendSnapshot(nullptr);
}

MapBackendSnapshot MapThread::timedBackendSnapshot(const MapBackendView* view) const {
  const auto snapshot_start = std::chrono::steady_clock::now();
  MapBackendSnapshot snapshot = view != nullptr ? map_backend_->snapshotForView(*view)
                                                : map_backend_->snapshot();
  snapshot.snapshot_ms = elapsedMs(snapshot_start, std::chrono::steady_clock::now());
  return snapshot;
}

void MapThread::run() {
  while (!stopRequested()) {
    MappingFrame frame;
    if (!mapping_queue_.waitPopFor(&frame, std::chrono::milliseconds(50))) {
      continue;
    }
    map_backend_->integrateFrame(frame);
    ++integrated_frames_;
    maybeLogStatus();
  }
}

void MapThread::maybeLogStatus() {
  std::lock_guard<std::mutex> lock(status_mutex_);
  const auto now = std::chrono::steady_clock::now();
  if (now - last_status_log_time_ <
      std::chrono::duration<double>(config_.file_logging_period_sec)) {
    return;
  }
  last_status_log_time_ = now;

  std::ostringstream stream;
  stream << std::fixed << std::setprecision(2)
         << "status backend=" << config_.map_backend
         << " integrated_frames=" << integrated_frames_.load()
         << " projection_requests=" << projection_requests_.load()
         << " projection_no_map=" << projection_no_map_.load()
         << " last_valid_patches=" << last_valid_patches_.load()
         << " last_projected_points=" << last_projected_points_.load()
         << " last_map_version=" << last_projection_map_version_.load()
         << " last_source_surface_points=" << last_source_surface_points_.load()
         << " last_tsdf_blocks=" << last_source_tsdf_blocks_.load()
         << " last_voxels_scanned=" << last_source_voxels_scanned_.load()
         << " last_selected_blocks=" << last_source_selected_blocks_.load()
         << " last_cached_surface_points=" << last_source_cached_surface_points_.load()
         << " cache_rebuilds=" << last_surface_cache_rebuilds_.load()
         << " last_project_total_ms=" << last_project_total_ms_.load()
         << " last_snapshot_ms=" << last_snapshot_ms_.load()
         << " last_surface_extract_ms=" << last_surface_extract_ms_.load()
         << " last_cache_build_ms=" << last_cache_build_ms_.load()
         << " last_frustum_filter_ms=" << last_frustum_filter_ms_.load()
         << " last_projection_loop_ms=" << last_projection_loop_ms_.load()
         << " last_projection_zbuffer_ms=" << last_projection_zbuffer_ms_.load();
  RunLogger::logGlobal("map_thread", stream.str());
}

}  // namespace roomie
