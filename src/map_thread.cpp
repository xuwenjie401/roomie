#include "roomie/pipeline/map_thread.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

#include "roomie/utils/run_logger.hpp"

namespace roomie {
namespace {

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

float medianInPlace(std::vector<float>* values) {
  const std::size_t middle = values->size() / 2;
  std::nth_element(values->begin(), values->begin() + static_cast<std::ptrdiff_t>(middle),
                   values->end());
  if (values->size() % 2 == 1) {
    return (*values)[middle];
  }

  const float upper = (*values)[middle];
  const auto lower_it = std::max_element(values->begin(),
                                         values->begin() + static_cast<std::ptrdiff_t>(middle));
  return 0.5f * (*lower_it + upper);
}

double elapsedMs(std::chrono::steady_clock::time_point start,
                 std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
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
      config_.max_integration_distance_m > 0.0f
          ? std::min(config_.depth_max_m, config_.max_integration_distance_m)
          : config_.depth_max_m;
  MapBackendSnapshot snapshot = timedBackendSnapshot(&view);
  if (!snapshot.has_map) {
    ++projection_no_map_;
    maybeLogStatus();
    return std::nullopt;
  }
  PatchDepth patch_depth = projectWorldPointsToPatchDepth(
      frame, snapshot.surface_points_world, snapshot.map_version);
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
  last_projection_median_ms_.store(patch_depth.projection_median_ms);
  maybeLogStatus();
  return patch_depth;
}

PatchDepth MapThread::projectWorldPointsToPatchDepth(const DetectionFrame& frame,
                                                     const WorldPointVector& world_points,
                                                     std::uint64_t map_version) const {
  PatchDepth result;
  result.map_version = map_version;

  const CameraIntrinsics intrinsics_960 =
      scaledIntrinsicsForBoxerInput(frame.intrinsics, frame.rgb, config_.boxer_input_size);
  if (!hasUsableIntrinsics(intrinsics_960) || world_points.empty()) {
    return result;
  }

  const Eigen::Isometry3f T_camera_world = frame.T_world_camera.inverse();
  std::array<std::vector<float>, PatchDepth::kSize> patch_depths;
  const float patch_width_px =
      static_cast<float>(config_.boxer_input_size) / static_cast<float>(PatchDepth::kCols);
  const float patch_height_px =
      static_cast<float>(config_.boxer_input_size) / static_cast<float>(PatchDepth::kRows);

  const auto projection_loop_start = std::chrono::steady_clock::now();
  for (const Eigen::Vector3f& point_world : world_points) {
    if (!point_world.allFinite()) {
      continue;
    }

    const Eigen::Vector3f point_camera = T_camera_world * point_world;
    const float z = point_camera.z();
    if (!std::isfinite(z) || z <= 0.0f) {
      continue;
    }

    const float u = intrinsics_960.fx * point_camera.x() / z + intrinsics_960.cx;
    const float v = intrinsics_960.fy * point_camera.y() / z + intrinsics_960.cy;
    if (!std::isfinite(u) || !std::isfinite(v) || u < 0.0f || v < 0.0f ||
        u >= static_cast<float>(config_.boxer_input_size) ||
        v >= static_cast<float>(config_.boxer_input_size)) {
      continue;
    }

    const int col = std::clamp(static_cast<int>(u / patch_width_px), 0, PatchDepth::kCols - 1);
    const int row = std::clamp(static_cast<int>(v / patch_height_px), 0, PatchDepth::kRows - 1);
    patch_depths[static_cast<std::size_t>(row * PatchDepth::kCols + col)].push_back(z);
    ++result.projected_points;
  }
  result.projection_loop_ms =
      elapsedMs(projection_loop_start, std::chrono::steady_clock::now());

  const auto median_start = std::chrono::steady_clock::now();
  for (int index = 0; index < PatchDepth::kSize; ++index) {
    std::vector<float>& depths = patch_depths[static_cast<std::size_t>(index)];
    if (depths.empty()) {
      continue;
    }
    result.values[static_cast<std::size_t>(index)] = medianInPlace(&depths);
    ++result.valid_patches;
  }
  result.projection_median_ms = elapsedMs(median_start, std::chrono::steady_clock::now());

  return result;
}

std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> MapThread::collectNearSurfaceVoxels(
    const RawDetection& detection) const {
  return map_backend_->collectNearSurfaceVoxels(detection);
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
         << " last_projection_median_ms=" << last_projection_median_ms_.load();
  RunLogger::logGlobal("map_thread", stream.str());
}

}  // namespace roomie
