#include "roomie/pipeline/map_thread.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>
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
                                               const ImageBuffer* rgb,
                                               int target_size) {
  CameraIntrinsics source = intrinsics;
  if ((source.width <= 0 || source.height <= 0) && rgb != nullptr &&
      rgb->width > 0 && rgb->height > 0) {
    source.width = rgb->width;
    source.height = rgb->height;
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

bool pointInsideDetection(const Eigen::Vector3f& point_world,
                          const RawDetection& detection) {
  if ((detection.size_m.array() <= 0.0f).any()) {
    return false;
  }
  const Eigen::Vector3f delta = point_world - detection.center_world;
  const float c = std::cos(-detection.yaw_rad);
  const float s = std::sin(-detection.yaw_rad);
  const Eigen::Vector3f local(c * delta.x() - s * delta.y(),
                              s * delta.x() + c * delta.y(),
                              delta.z());
  return (local.cwiseAbs().array() <= (0.5f * detection.size_m).array()).all();
}

bool aabbIntersectsCameraFrustum(const SurfaceAabb& aabb,
                                 const Eigen::Isometry3f& T_camera_world,
                                 const CameraIntrinsics& intrinsics,
                                 int image_size,
                                 float min_depth_m,
                                 float max_depth_m) {
  if (!aabb.valid() || !hasUsableIntrinsics(intrinsics) || image_size <= 0 ||
      max_depth_m <= min_depth_m) {
    return false;
  }
  const float left = -intrinsics.cx / intrinsics.fx;
  const float right =
      (static_cast<float>(image_size) - intrinsics.cx) / intrinsics.fx;
  const float top = -intrinsics.cy / intrinsics.fy;
  const float bottom =
      (static_cast<float>(image_size) - intrinsics.cy) / intrinsics.fy;
  std::array<float, 6> maximum_signed;
  maximum_signed.fill(-std::numeric_limits<float>::infinity());
  for (const float x : {aabb.min.x(), aabb.max.x()}) {
    for (const float y : {aabb.min.y(), aabb.max.y()}) {
      for (const float z : {aabb.min.z(), aabb.max.z()}) {
        const Eigen::Vector3f point =
            T_camera_world * Eigen::Vector3f(x, y, z);
        maximum_signed[0] =
            std::max(maximum_signed[0], point.x() - left * point.z());
        maximum_signed[1] =
            std::max(maximum_signed[1], right * point.z() - point.x());
        maximum_signed[2] =
            std::max(maximum_signed[2], point.y() - top * point.z());
        maximum_signed[3] =
            std::max(maximum_signed[3], bottom * point.z() - point.y());
        maximum_signed[4] =
            std::max(maximum_signed[4], point.z() - min_depth_m);
        maximum_signed[5] =
            std::max(maximum_signed[5], max_depth_m - point.z());
      }
    }
  }
  return std::all_of(maximum_signed.begin(),
                     maximum_signed.end(),
                     [](float value) { return value >= 0.0f; });
}

}  // namespace

MapThread::MapThread(ThreadSafeQueue<FrameBundlePtr>& mapping_queue,
                     PipelineConfig config)
    : MapThread(mapping_queue, config, createMapBackend(config)) {}

MapThread::MapThread(ThreadSafeQueue<FrameBundlePtr>& mapping_queue,
                     PipelineConfig config,
                     std::unique_ptr<MapBackend> map_backend)
    : WorkerThread("map_thread"),
      mapping_queue_(mapping_queue),
      config_(std::move(config)),
      map_backend_(std::move(map_backend)) {
  if (!map_backend_) {
    throw std::invalid_argument("MapThread requires a map backend");
  }
}

MapThread::~MapThread() {
  stop();
}

bool MapThread::enqueueFrameBundle(FrameBundlePtr frame) {
  if (!frame) {
    return false;
  }
  const FrameId frame_id = frame->provenance.frame_id;
  PushResult<FrameBundlePtr> result;
  if (frame->perception_candidate &&
      frame->provenance.map_mode == MapMode::kOnline) {
    const auto remaining = frame->due_time - std::chrono::steady_clock::now();
    result = mapping_queue_.pushBarrierFor(
        std::move(frame), /*max_replaceable_ahead=*/1U, remaining);
  } else {
    result = mapping_queue_.push(std::move(frame));
  }
  if (result.outcome == PushOutcome::kReplaced && result.replaced_item) {
    const FrameBundlePtr& replaced = *result.replaced_item;
    RunLogger::logGlobal(
        "map_thread",
        "mapping_replaced incoming_frame_id=" + std::to_string(frame_id) +
            " replaced_frame_id=" +
            std::to_string(replaced ? replaced->provenance.frame_id : 0) +
            " reason=capacity");
  }
  return result.accepted();
}

bool MapThread::cancelPerceptionCandidate(FrameBundlePtr frame,
                                          const std::string& reason) {
  if (!frame || !frame->perception_candidate) {
    return false;
  }
  const FrameKey key{frame->provenance.run_id, frame->provenance.frame_id};
  bool removed_commit = false;
  {
    std::lock_guard<std::mutex> lock(commit_mutex_);
    auto existing = commits_by_frame_.find(key);
    if (existing != commits_by_frame_.end()) {
      commits_by_frame_.erase(existing);
      commit_order_.erase(std::remove(commit_order_.begin(),
                                      commit_order_.end(), key),
                          commit_order_.end());
      removed_commit = true;
    }
    if (cancelled_candidates_.count(key) == 0) {
      cancelled_candidate_order_.push_back(key);
    }
    cancelled_candidates_[key] = CancelledCandidate{
        reason.empty() ? "cancelled" : reason, frame->due_time};
  }

  const std::size_t removed_queued = mapping_queue_.cancelIf(
      [&frame, &key](const FrameBundlePtr& queued) {
        return queued && queued->perception_candidate &&
               (queued.get() == frame.get() ||
                (queued->provenance.run_id == key.run_id &&
                 queued->provenance.frame_id == key.frame_id));
      });

  // If the work was still queued, or its commit was already retained, no
  // actor-owned operation can still publish this key. Otherwise leave the
  // tombstone for the in-progress integration to consume at publishCommit().
  if (removed_queued > 0 || removed_commit) {
    std::lock_guard<std::mutex> lock(commit_mutex_);
    cancelled_candidates_.erase(key);
    cancelled_candidate_order_.erase(
        std::remove(cancelled_candidate_order_.begin(),
                    cancelled_candidate_order_.end(), key),
        cancelled_candidate_order_.end());
  }
  RunLogger::logGlobal(
      "map_thread",
      "perception_barrier_cancelled run_id=" + runIdString(key.run_id) +
          " frame_id=" + std::to_string(key.frame_id) + " reason=" +
          (reason.empty() ? "cancelled" : reason) + " queued_removed=" +
          std::to_string(removed_queued) + " commit_removed=" +
          std::string(removed_commit ? "true" : "false"));
  commit_cv_.notify_all();
  return true;
}

std::optional<PatchDepth> MapThread::projectPatchDepth(const FrameBundle& frame) {
  ++projection_requests_;
  const auto project_total_start = std::chrono::steady_clock::now();
  SurfaceSnapshotPtr pinned_surface;
  MapBackendSnapshot diagnostics;
  FrameProvenance provenance = frame.provenance;
  if (config_.freeze_tsdf_map) {
    {
      std::unique_lock<std::mutex> lock(commit_mutex_);
      const bool ready = commit_cv_.wait_until(
          lock, frame.due_time, [this]() {
            return stopRequested() || frozen_initialization_complete_;
          });
      if (!ready || stopRequested() || !frozen_initialization_error_.empty()) {
        RunLogger::logGlobal(
            "map_thread",
            "perception_drop frame_id=" +
                std::to_string(frame.provenance.frame_id) +
                " reason=frozen_map_not_ready error=" +
                frozen_initialization_error_);
        ++projection_no_map_;
        return std::nullopt;
      }
    }
    const std::shared_ptr<const PublishedSurface> published =
        std::atomic_load(&latest_published_surface_);
    if (published) {
      pinned_surface = published->snapshot;
      diagnostics = published->diagnostics;
    }
    if (pinned_surface) {
      provenance.map = pinned_surface->mapStamp();
      provenance.surface = pinned_surface->surfaceStamp();
      provenance.map_mode = MapMode::kFrozen;
      provenance.includes_current_frame = false;
      provenance.causality_verified = true;
    }
  } else {
    std::optional<CommitRecord> record = waitForCommit(frame);
    if (record && record->success) {
      pinned_surface = record->commit.snapshot;
      diagnostics = record->diagnostics;
      provenance.map = record->commit.map;
      provenance.surface = record->commit.surface;
      provenance.map_mode = MapMode::kOnline;
      provenance.includes_current_frame = record->commit.includes_current_frame;
      provenance.causality_verified = true;
    } else if (record) {
      RunLogger::logGlobal(
          "map_thread",
          "perception_drop frame_id=" +
              std::to_string(frame.provenance.frame_id) +
              " reason=map_commit_failed error=" + record->error);
    }
  }

  return projectPinnedSurface(frame,
                              std::move(pinned_surface),
                              std::move(diagnostics),
                              std::move(provenance),
                              project_total_start);
}

std::optional<PatchDepth> MapThread::projectPatchDepth(
    const FrameBundle& frame, const MapCommit& commit) {
  ++projection_requests_;
  const auto project_total_start = std::chrono::steady_clock::now();
  const FrameKey key{frame.provenance.run_id, frame.provenance.frame_id};
  const bool exact_identity =
      commit.run_id == key.run_id && commit.frame_id == key.frame_id &&
      commit.frame_bundle && commit.frame_bundle.get() == &frame;
  const bool exact_causality =
      commit.success && commit.snapshot && commit.includes_current_frame &&
      commit.map.map_epoch == commit.surface.map_epoch &&
      commit.surface.source_map_revision == commit.map.map_revision &&
      commit.map.integrated_through_ns >= frame.provenance.sensor_time_ns;
  if (!exact_identity || !exact_causality ||
      std::chrono::steady_clock::now() >= frame.due_time) {
    ++projection_no_map_;
    RunLogger::logGlobal(
        "map_thread",
        "perception_drop run_id=" + runIdString(key.run_id) +
            " frame_id=" + std::to_string(key.frame_id) + " reason=" +
            (!exact_identity
                 ? "map_commit_bundle_identity"
                 : (!exact_causality ? "map_commit_causality"
                                     : "map_commit_deadline")) +
            (commit.error.empty() ? std::string{} :
                                    " error=" + commit.error));
    maybeLogStatus();
    return std::nullopt;
  }

  // The observer carries all data required for correctness. Opportunistically
  // consume the legacy one-shot record only to retain its exact diagnostics;
  // this lookup never waits and is therefore safe on the scheduler thread.
  MapBackendSnapshot diagnostics;
  {
    std::lock_guard<std::mutex> lock(commit_mutex_);
    auto it = commits_by_frame_.find(key);
    if (it != commits_by_frame_.end() &&
        it->second.commit.frame_bundle.get() == &frame &&
        it->second.commit.snapshot == commit.snapshot) {
      diagnostics = it->second.diagnostics;
      commits_by_frame_.erase(it);
      commit_order_.erase(std::remove(commit_order_.begin(),
                                      commit_order_.end(), key),
                          commit_order_.end());
    }
  }

  FrameProvenance provenance = frame.provenance;
  provenance.map_mode = MapMode::kOnline;
  provenance.map = commit.map;
  provenance.surface = commit.surface;
  provenance.includes_current_frame = true;
  provenance.causality_verified = true;
  return projectPinnedSurface(frame,
                              commit.snapshot,
                              std::move(diagnostics),
                              std::move(provenance),
                              project_total_start);
}

std::optional<PatchDepth> MapThread::projectPinnedSurface(
    const FrameBundle& frame,
    SurfaceSnapshotPtr pinned_surface,
    MapBackendSnapshot diagnostics,
    FrameProvenance provenance,
    std::chrono::steady_clock::time_point project_total_start) {
  if (!pinned_surface || pinned_surface->empty()) {
    ++projection_no_map_;
    maybeLogStatus();
    return std::nullopt;
  }

  const float min_depth_m = config_.depth_min_m;
  const float max_depth_m =
      config_.patch_depth_max_m > 0.0f ? config_.patch_depth_max_m
                                      : config_.depth_max_m;
  const CameraIntrinsics intrinsics_960 = scaledIntrinsicsForBoxerInput(
      frame.intrinsics, frame.rgb.get(), config_.boxer_input_size);
  const Eigen::Isometry3f T_camera_world = frame.T_world_camera.inverse();
  const auto filter_start = std::chrono::steady_clock::now();
  std::vector<SurfaceBlockPtr> selected_blocks;
  std::uint64_t selected_points = 0;
  for (const SurfaceBlockPtr& block : pinned_surface->blockView()) {
    if (!block || block->empty() ||
        !aabbIntersectsCameraFrustum(block->aabb(),
                                    T_camera_world,
                                    intrinsics_960,
                                    config_.boxer_input_size,
                                    min_depth_m,
                                    max_depth_m)) {
      continue;
    }
    selected_points += static_cast<std::uint64_t>(block->pointCount());
    selected_blocks.push_back(block);
  }
  const double filter_ms =
      elapsedMs(filter_start, std::chrono::steady_clock::now());
  PatchDepth patch_depth = projectSurfaceBlocksToPatchDepth(
      frame,
      selected_blocks,
      pinned_surface->surfaceStamp().source_map_revision,
      min_depth_m,
      max_depth_m);
  patch_depth.provenance = std::move(provenance);
  patch_depth.source_surface_points = selected_points;
  patch_depth.source_tsdf_blocks = diagnostics.tsdf_blocks;
  patch_depth.source_voxels_scanned = diagnostics.surface_voxels_scanned;
  patch_depth.source_selected_blocks = selected_blocks.size();
  patch_depth.source_cached_surface_points = pinned_surface->pointCount();
  patch_depth.surface_cache_rebuilds = diagnostics.cache_rebuilds;
  patch_depth.snapshot_ms = 0.0;
  patch_depth.surface_extract_ms = diagnostics.surface_extract_ms;
  patch_depth.cache_build_ms = diagnostics.cache_build_ms;
  patch_depth.frustum_filter_ms = filter_ms;
  patch_depth.projection_compute_ms =
      patch_depth.frustum_filter_ms + patch_depth.projection_loop_ms +
      patch_depth.projection_zbuffer_ms;
  patch_depth.surface_cache_ready = true;
  patch_depth.view_filtered = true;
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
  last_projection_compute_ms_.store(patch_depth.projection_compute_ms);
  maybeLogStatus();
  return patch_depth;
}

PatchDepth MapThread::projectSurfaceBlocksToPatchDepth(
    const FrameBundle& frame,
    const std::vector<SurfaceBlockPtr>& surface_blocks,
    std::uint64_t map_version,
    float min_depth_m,
    float max_depth_m) const {
  PatchDepth result;
  result.map_version = map_version;

  const CameraIntrinsics intrinsics_960 =
      scaledIntrinsicsForBoxerInput(
          frame.intrinsics, frame.rgb.get(), config_.boxer_input_size);
  if (!hasUsableIntrinsics(intrinsics_960) || surface_blocks.empty()) {
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
  for (const SurfaceBlockPtr& block : surface_blocks) {
    if (!block) {
      continue;
    }
    for (const MapSurfacePoint& surface_point : block->points()) {
      const Eigen::Vector3f& point_world = surface_point.position_world;
      if (!point_world.allFinite()) {
        continue;
      }

      const Eigen::Vector3f point_camera = T_camera_world * point_world;
      const float z = point_camera.z();
      if (!std::isfinite(z) || z < min_depth_m || z > max_depth_m) {
        continue;
      }

      const float u =
          intrinsics_960.fx * point_camera.x() / z + intrinsics_960.cx;
      const float v =
          intrinsics_960.fy * point_camera.y() / z + intrinsics_960.cy;
      if (!std::isfinite(u) || !std::isfinite(v) || u < 0.0f || v < 0.0f ||
          u >= static_cast<float>(config_.boxer_input_size) ||
          v >= static_cast<float>(config_.boxer_input_size)) {
        continue;
      }

      const int center_col = std::clamp(
          static_cast<int>(u / zbuffer_cell_width_px), 0, zbuffer_cols - 1);
      const int center_row = std::clamp(
          static_cast<int>(v / zbuffer_cell_height_px), 0, zbuffer_rows - 1);
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
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> refs;
  const std::shared_ptr<const PublishedSurface> published =
      std::atomic_load(&latest_published_surface_);
  const SurfaceSnapshotPtr snapshot = published ? published->snapshot : nullptr;
  if (!snapshot) {
    return refs;
  }
  for (const SurfaceBlockPtr& block : snapshot->blockView()) {
    for (const MapSurfacePoint& point : block->points()) {
      if (point.has_voxel_ref &&
          pointInsideDetection(point.position_world, detection)) {
        refs.push_back(point.voxel_ref);
      }
    }
  }
  return refs;
}

MapBackendSnapshot MapThread::snapshotSurfacePoints() const {
  return snapshotFromPinnedSurface(/*include_debug=*/true);
}

std::uint64_t MapThread::mapVersion() const {
  const SurfaceSnapshotPtr snapshot = latestSurfaceSnapshot();
  return snapshot ? snapshot->mapStamp().map_revision : 0;
}

MapBackendSnapshot MapThread::debugSnapshot() const {
  return snapshotFromPinnedSurface(/*include_debug=*/true);
}

SurfaceSnapshotPtr MapThread::latestSurfaceSnapshot() const {
  const std::shared_ptr<const PublishedSurface> published =
      std::atomic_load(&latest_published_surface_);
  return published ? published->snapshot : nullptr;
}

MapDeltaQueryResult MapThread::queryMapDeltas(
    const SurfaceStamp& from,
    const SurfaceStamp& to,
    const std::vector<BlockIndex>& dependency_blocks) const {
  return delta_journal_.query(from, to, dependency_blocks);
}

void MapThread::setCommitObserver(
    std::function<void(const MapCommit&)> observer) {
  commit_observer_ = std::move(observer);
}

MapCheckpointOperationResult MapThread::loadConfiguredCheckpoint(
    const MapCheckpointManifest* expected_manifest) {
  MapCheckpointOperationResult result;
  // A durable manifest is authoritative coordinated-recovery state.  It must
  // be loadable after restart even when the fresh-run convenience flag
  // tsdf.load_map remains false.
  if (!config_.load_map && expected_manifest == nullptr) {
    result.disposition = MapCheckpointDisposition::kNotRequested;
    return result;
  }
  if (running()) {
    result.disposition = MapCheckpointDisposition::kFailed;
    result.error = "cannot load a map checkpoint while MapThread is running";
    return result;
  }
  if (checkpoint_load_attempted_) {
    result.disposition = MapCheckpointDisposition::kFailed;
    result.error = "map checkpoint load was already attempted";
    return result;
  }
  checkpoint_load_attempted_ = true;
  try {
    result = map_backend_->loadCheckpoint(expected_manifest);
    checkpoint_loaded_ = result.succeeded();
    return result;
  } catch (const std::exception& error) {
    result.disposition = MapCheckpointDisposition::kFailed;
    result.error = std::string("map checkpoint load threw: ") + error.what();
    return result;
  } catch (...) {
    result.disposition = MapCheckpointDisposition::kFailed;
    result.error = "map checkpoint load threw a non-standard exception";
    return result;
  }
}

MapCheckpointOperationResult MapThread::saveConfiguredCheckpoint() {
  MapCheckpointOperationResult result;
  if (!config_.save_map) {
    result.disposition = MapCheckpointDisposition::kNotRequested;
    return result;
  }
  if (running()) {
    result.disposition = MapCheckpointDisposition::kFailed;
    result.error = "cannot save a map checkpoint while MapThread is running";
    return result;
  }
  if (checkpoint_save_attempted_) {
    result.disposition = MapCheckpointDisposition::kFailed;
    result.error = "map checkpoint save was already attempted";
    return result;
  }
  checkpoint_save_attempted_ = true;

  MapStamp stamp;
  stamp.map_epoch = map_epoch_;
  const SurfaceSnapshotPtr surface = latestSurfaceSnapshot();
  if (surface) {
    stamp = surface->mapStamp();
  } else {
    const MapBackendSnapshot backend_snapshot = map_backend_->snapshot();
    stamp.map_revision = backend_snapshot.latest_map_version;
    stamp.integrated_through_ns = integrated_through_ns_.load();
  }
  try {
    return map_backend_->saveCheckpoint(stamp);
  } catch (const std::exception& error) {
    result.disposition = MapCheckpointDisposition::kFailed;
    result.error = std::string("map checkpoint save threw: ") + error.what();
    return result;
  } catch (...) {
    result.disposition = MapCheckpointDisposition::kFailed;
    result.error = "map checkpoint save threw a non-standard exception";
    return result;
  }
}

MapBackendSnapshot MapThread::snapshotFromPinnedSurface(bool include_debug) const {
  const auto snapshot_start = std::chrono::steady_clock::now();
  MapBackendSnapshot result;
  const std::shared_ptr<const PublishedSurface> published =
      std::atomic_load(&latest_published_surface_);
  if (published) {
    result = published->diagnostics;
  }
  const SurfaceSnapshotPtr snapshot = published ? published->snapshot : nullptr;
  if (!snapshot) {
    result.snapshot_ms = elapsedMs(snapshot_start, std::chrono::steady_clock::now());
    return result;
  }
  result.map_version = snapshot->surfaceStamp().source_map_revision;
  result.latest_map_version = snapshot->mapStamp().map_revision;
  result.surface_cache_ready = true;
  result.cache_dirty = false;
  result.has_map = !snapshot->empty();
  result.cached_surface_points = snapshot->pointCount();
  result.selected_blocks = snapshot->blockCount();
  result.surface_points_world = snapshot->flattenWorldPoints();
  if (include_debug) {
    result.debug_surface_points = snapshot->flatten();
  } else {
    result.debug_surface_points.clear();
  }
  result.snapshot_ms = elapsedMs(snapshot_start, std::chrono::steady_clock::now());
  return result;
}

std::optional<MapThread::CommitRecord> MapThread::waitForCommit(
    const FrameBundle& frame) {
  const auto deadline = frame.due_time;
  const FrameKey key{frame.provenance.run_id, frame.provenance.frame_id};
  std::unique_lock<std::mutex> lock(commit_mutex_);
  pruneExpiredCommitsLocked(std::chrono::steady_clock::now());
  const bool ready = commit_cv_.wait_until(lock, deadline, [this, &key]() {
    return commits_by_frame_.count(key) > 0 ||
           (stopRequested() && mapping_queue_.empty());
  });
  if (!ready || (stopRequested() && commits_by_frame_.count(key) == 0)) {
    pruneExpiredCommitsLocked(std::chrono::steady_clock::now());
    RunLogger::logGlobal(
        "map_thread",
        "perception_drop frame_id=" +
            std::to_string(frame.provenance.frame_id) +
            " reason=map_commit_deadline");
    return std::nullopt;
  }
  auto it = commits_by_frame_.find(key);
  if (it == commits_by_frame_.end()) {
    return std::nullopt;
  }
  // FrameKey finds the causal candidate, while pointer identity proves that
  // the caller owns the exact admitted FrameBundle. A copied/spoofed bundle
  // must not consume the real candidate's one-shot join record.
  if (!it->second.commit.frame_bundle ||
      it->second.commit.frame_bundle.get() != &frame) {
    CommitRecord mismatch = it->second;
    mismatch.success = false;
    mismatch.error = "map commit bundle identity mismatch";
    return mismatch;
  }
  CommitRecord record = std::move(it->second);
  commits_by_frame_.erase(it);
  commit_order_.erase(std::remove(commit_order_.begin(),
                                  commit_order_.end(),
                                  key),
                      commit_order_.end());
  return record;
}

void MapThread::pruneExpiredCommitsLocked(
    std::chrono::steady_clock::time_point now) {
  for (auto it = commit_order_.begin(); it != commit_order_.end();) {
    auto commit_it = commits_by_frame_.find(*it);
    if (commit_it == commits_by_frame_.end()) {
      it = commit_order_.erase(it);
      continue;
    }
    if (commit_it->second.commit.due_time <= now) {
      RunLogger::logGlobal(
          "map_thread",
          "map_commit_evicted run_id=" + runIdString(it->run_id) +
              " frame_id=" + std::to_string(it->frame_id) +
              " reason=deadline");
      commits_by_frame_.erase(commit_it);
      it = commit_order_.erase(it);
      continue;
    }
    ++it;
  }
  for (auto it = cancelled_candidate_order_.begin();
       it != cancelled_candidate_order_.end();) {
    auto cancelled = cancelled_candidates_.find(*it);
    if (cancelled == cancelled_candidates_.end()) {
      it = cancelled_candidate_order_.erase(it);
      continue;
    }
    if (cancelled->second.due_time <= now) {
      cancelled_candidates_.erase(cancelled);
      it = cancelled_candidate_order_.erase(it);
      continue;
    }
    ++it;
  }
  const std::size_t cancellation_limit =
      std::max<std::size_t>(8, config_.pending_frame_limit * 2U);
  while (cancelled_candidate_order_.size() > cancellation_limit) {
    cancelled_candidates_.erase(cancelled_candidate_order_.front());
    cancelled_candidate_order_.pop_front();
  }
}

MapThread::CommitRecord MapThread::buildCommit(
    const FrameBundle& frame,
    const MapIntegrationResult& integration,
    SurfaceRefreshResult refresh) {
  CommitRecord record;
  if (!integration.success || !refresh.success) {
    record.error = !integration.success ? integration.error : refresh.error;
    return record;
  }

  MapStamp map_stamp;
  map_stamp.map_epoch = map_epoch_;
  map_stamp.map_revision =
      integration.map_revision > 0 ? integration.map_revision
                                   : refresh.source_map_revision;
  map_stamp.integrated_through_ns =
      std::max(integration.integrated_through_ns,
               integrated_through_ns_.load());
  SurfaceStamp surface_stamp;
  surface_stamp.map_epoch = map_epoch_;
  surface_stamp.surface_revision = next_surface_revision_++;
  surface_stamp.source_map_revision = map_stamp.map_revision;

  const std::shared_ptr<const PublishedSurface> published =
      std::atomic_load(&latest_published_surface_);
  SurfaceSnapshotPtr base = published ? published->snapshot : nullptr;
  if (refresh.full_rebuild ||
      (base && base->surfaceStamp().map_epoch != surface_stamp.map_epoch)) {
    base.reset();
  }
  SurfaceSnapshotBuilder builder(base, map_stamp, surface_stamp);
  std::unordered_set<BlockIndex, BlockIndexHash> current_indices;
  current_indices.reserve(refresh.blocks.size());
  for (const SurfaceBlockPtr& block : refresh.blocks) {
    if (!block) {
      continue;
    }
    current_indices.insert(block->index());
    if (!base || base->block(block->index()) != block) {
      builder.setBlock(block);
    }
  }
  if (base && refresh.full_rebuild) {
    for (const auto& entry : base->blocks()) {
      if (current_indices.count(entry.first) == 0) {
        builder.removeBlock(entry.first);
      }
    }
  }
  for (const BlockIndex& removed : refresh.removed_blocks) {
    builder.removeBlock(removed);
  }
  SurfaceSnapshotBuildResult built = builder.build();
  built.delta.full_rebuild = built.delta.full_rebuild || refresh.full_rebuild;

  record.success = true;
  record.commit.run_id = frame.provenance.run_id;
  record.commit.frame_id = frame.provenance.frame_id;
  record.commit.map = map_stamp;
  record.commit.surface = surface_stamp;
  record.commit.snapshot = built.snapshot;
  record.commit.delta = std::move(built.delta);
  record.commit.includes_current_frame = !config_.freeze_tsdf_map;
  record.commit.perception_candidate = frame.perception_candidate;
  record.commit.due_time = frame.due_time;
  record.diagnostics = std::move(refresh.diagnostics);
  record.diagnostics.map_version = map_stamp.map_revision;
  record.diagnostics.latest_map_version = map_stamp.map_revision;
  record.diagnostics.surface_cache_ready = true;
  record.diagnostics.cache_dirty = false;
  record.diagnostics.has_map = !built.snapshot->empty();
  record.diagnostics.cached_surface_points = built.snapshot->pointCount();
  record.diagnostics.selected_blocks = built.snapshot->blockCount();
  return record;
}

void MapThread::publishCommit(CommitRecord record) {
  record.commit.published_at = std::chrono::steady_clock::now();
  record.commit.success = record.success && record.commit.snapshot;
  record.commit.error = record.error;
  const bool successful_publish = record.commit.success;
  std::optional<MapCommit> observer_commit;
  if (successful_publish) {
    auto published = std::make_shared<PublishedSurface>();
    published->snapshot = record.commit.snapshot;
    published->diagnostics = record.diagnostics;
    std::shared_ptr<const PublishedSurface> immutable_published =
        std::move(published);
    std::atomic_store(&latest_published_surface_, immutable_published);
    delta_journal_.append(record.commit.delta);
    RunLogger::logGlobal(
        "map_thread",
        "map_commit_published run_id=" + runIdString(record.commit.run_id) +
            " frame_id=" + std::to_string(record.commit.frame_id) +
            " map_epoch=" + runIdString(record.commit.map.map_epoch) +
            " map_revision=" +
            std::to_string(record.commit.map.map_revision) +
            " surface_revision=" +
            std::to_string(record.commit.surface.surface_revision) +
            " source_map_revision=" +
            std::to_string(record.commit.surface.source_map_revision) +
            " changed_blocks=" +
            std::to_string(record.commit.delta.changed_blocks.size()) +
            " removed_blocks=" +
            std::to_string(record.commit.delta.removed_blocks.size()) +
            " full_rebuild=" +
            std::string(record.commit.delta.full_rebuild ? "true" : "false") +
            " includes_current_frame=" +
            std::string(record.commit.includes_current_frame ? "true" :
                                                               "false"));
  }

  if (record.commit.frame_id != 0 && record.commit.perception_candidate) {
    std::lock_guard<std::mutex> lock(commit_mutex_);
    const auto now = std::chrono::steady_clock::now();
    pruneExpiredCommitsLocked(now);
    const FrameKey key{record.commit.run_id, record.commit.frame_id};
    const auto cancelled = cancelled_candidates_.find(key);
    if (cancelled != cancelled_candidates_.end()) {
      const std::string reason = cancelled->second.reason;
      cancelled_candidates_.erase(cancelled);
      cancelled_candidate_order_.erase(
          std::remove(cancelled_candidate_order_.begin(),
                      cancelled_candidate_order_.end(), key),
          cancelled_candidate_order_.end());
      record.commit.perception_candidate = false;
      RunLogger::logGlobal(
          "map_thread",
          "map_commit_not_joined run_id=" + runIdString(key.run_id) +
              " frame_id=" + std::to_string(key.frame_id) +
              " reason=cancelled_" + reason);
    } else if (record.commit.due_time <= now) {
      RunLogger::logGlobal(
          "map_thread",
          "map_commit_not_joined run_id=" +
              runIdString(record.commit.run_id) +
              " frame_id=" + std::to_string(record.commit.frame_id) +
              " reason=deadline");
    } else if (stopRequested()) {
      // The asynchronous observer still receives this accepted candidate;
      // only the legacy blocking-wait table is skipped during actor shutdown.
      RunLogger::logGlobal(
          "map_thread",
          "map_commit_not_retained run_id=" +
              runIdString(record.commit.run_id) +
              " frame_id=" + std::to_string(record.commit.frame_id) +
              " reason=shutdown_async_delivery");
    } else {
      observer_commit = record.commit;
      auto existing = commits_by_frame_.find(key);
      if (existing == commits_by_frame_.end()) {
        commit_order_.push_back(key);
      } else {
        RunLogger::logGlobal(
            "map_thread",
            "map_commit_replaced run_id=" + runIdString(key.run_id) +
                " frame_id=" + std::to_string(key.frame_id) +
                " reason=duplicate_key");
      }
      commits_by_frame_[key] = std::move(record);
      const std::size_t limit =
          std::max<std::size_t>(1, config_.pending_frame_limit);
      while (commit_order_.size() > limit) {
        const FrameKey expired = commit_order_.front();
        commit_order_.pop_front();
        commits_by_frame_.erase(expired);
        RunLogger::logGlobal(
            "map_thread",
            "map_commit_evicted run_id=" + runIdString(expired.run_id) +
                " frame_id=" + std::to_string(expired.frame_id) +
                " reason=pending_limit");
      }
    }
    if (!observer_commit &&
        (successful_publish || record.commit.perception_candidate)) {
      observer_commit = record.commit;
    }
  } else if (successful_publish) {
    observer_commit = record.commit;
  }

  // Downstream callbacks may synchronously query the exact frame commit. Make
  // both the published surface and the candidate join visible, and wake all
  // waiters, before invoking any observer-controlled code.
  commit_cv_.notify_all();
  if (observer_commit && commit_observer_) {
    try {
      commit_observer_(*observer_commit);
    } catch (const std::exception& error) {
      RunLogger::logGlobal(
          "map_thread",
          "map commit observer failed: " + std::string(error.what()));
    } catch (...) {
      RunLogger::logGlobal(
          "map_thread",
          "map commit observer failed with non-standard exception");
    }
  }
}

void MapThread::initializeStartupSurface() {
  FrameBundle frame;
  frame.provenance.frame_id = 0;
  CommitRecord record;
  try {
    MapIntegrationResult integration;
    integration.success = true;
    integration.map_changed = false;
    SurfaceRefreshResult refresh =
        map_backend_->refreshSurface(integration, /*force_full_rebuild=*/true);
    integration.map_revision = refresh.source_map_revision;
    record = buildCommit(frame, integration, std::move(refresh));
    // Frame zero is a synthetic publication of an already-loaded map. It
    // never contains a current camera frame, including in online mode.
    record.commit.includes_current_frame = false;
  } catch (const std::exception& error) {
    record.error = std::string("startup surface exception: ") + error.what();
  } catch (...) {
    record.error = "startup surface unknown exception";
  }
  const bool initialized = record.success;
  const std::string initialization_error = record.error;
  if (initialized) {
    publishCommit(std::move(record));
  }
  {
    std::lock_guard<std::mutex> lock(commit_mutex_);
    frozen_initialization_complete_ = true;
    if (!initialized) {
      frozen_initialization_error_ = initialization_error.empty()
                                         ? "startup surface refresh failed"
                                         : initialization_error;
    }
  }
  if (checkpoint_loaded_) {
    RunLogger::logGlobal(
        "map_checkpoint",
        initialized
            ? "published loaded checkpoint surface"
            : "failed to publish loaded checkpoint surface error=" +
                  frozen_initialization_error_);
  }
  commit_cv_.notify_all();
}

void MapThread::run() {
  // A restored online map must be published before the first new RGB-D frame;
  // otherwise RViz and read-only consumers see an empty map indefinitely when
  // the camera is idle. Frozen mode needs the same startup publication.
  if (config_.freeze_tsdf_map || checkpoint_loaded_) {
    initializeStartupSurface();
  }
  while (!stopRequested() || !mapping_queue_.empty()) {
    FrameBundlePtr frame;
    if (!mapping_queue_.waitPopFor(&frame, std::chrono::milliseconds(50))) {
      {
        std::lock_guard<std::mutex> lock(commit_mutex_);
        pruneExpiredCommitsLocked(std::chrono::steady_clock::now());
      }
      if (stopRequested()) {
        break;
      }
      continue;
    }
    if (!frame) {
      continue;
    }
    CommitRecord record;
    record.commit.run_id = frame->provenance.run_id;
    record.commit.frame_id = frame->provenance.frame_id;
    record.commit.perception_candidate = frame->perception_candidate;
    record.commit.due_time = frame->due_time;
    record.commit.frame_bundle = frame;
    bool cancelled_before_compute = false;
    {
      std::lock_guard<std::mutex> lock(commit_mutex_);
      cancelled_before_compute =
          cancelled_candidates_.count(
              FrameKey{frame->provenance.run_id,
                       frame->provenance.frame_id}) > 0;
    }
    const auto compute_start = std::chrono::steady_clock::now();
    if (cancelled_before_compute) {
      record.error = "perception candidate cancelled before map compute";
      publishCommit(std::move(record));
      maybeLogStatus();
      continue;
    }
    if (frame->perception_candidate && compute_start >= frame->due_time) {
      record.error = "perception deadline expired before map compute";
      RunLogger::logGlobal(
          "map_thread",
          "perception_drop run_id=" +
              runIdString(frame->provenance.run_id) + " frame_id=" +
              std::to_string(frame->provenance.frame_id) +
              " reason=deadline_before_map_compute");
      publishCommit(std::move(record));
      maybeLogStatus();
      continue;
    }
    last_integrate_ms_.store(0.0);
    last_refresh_ms_.store(0.0);
    try {
      const auto integrate_start = std::chrono::steady_clock::now();
      MapIntegrationResult integration = map_backend_->integrateFrame(*frame);
      last_integrate_ms_.store(
          elapsedMs(integrate_start, std::chrono::steady_clock::now()));
      if (integration.success) {
        const auto refresh_start = std::chrono::steady_clock::now();
        SurfaceRefreshResult refresh =
            map_backend_->refreshSurface(integration, /*force_full_rebuild=*/false);
        last_refresh_ms_.store(
            elapsedMs(refresh_start, std::chrono::steady_clock::now()));
        if (refresh.success &&
            refresh.source_map_revision != integration.map_revision) {
          refresh.success = false;
          refresh.error = "surface source revision does not match integration";
        }
        if (refresh.success && !integration.updated_blocks_complete &&
            !refresh.full_rebuild) {
          refresh.success = false;
          refresh.error =
              "incomplete dirty-block set requires a full surface rebuild";
        }
        record = buildCommit(*frame, integration, std::move(refresh));
        if (record.success) {
          const TimeNanoseconds integrated_time =
              std::max(integrated_through_ns_.load(),
                       frame->provenance.sensor_time_ns);
          integrated_through_ns_.store(integrated_time);
          ++integrated_frames_;
        }
      } else {
        record.error = integration.error;
      }
    } catch (const std::exception& error) {
      record.success = false;
      record.error = std::string("map actor exception: ") + error.what();
    } catch (...) {
      record.success = false;
      record.error = "map actor unknown exception";
    }
    record.commit.run_id = frame->provenance.run_id;
    record.commit.frame_id = frame->provenance.frame_id;
    record.commit.perception_candidate = frame->perception_candidate;
    record.commit.due_time = frame->due_time;
    record.commit.frame_bundle = frame;
    const ChannelStats queue_stats = mapping_queue_.stats();
    RunLogger::logGlobal(
        "map_thread",
        "map_compute run_id=" + runIdString(frame->provenance.run_id) +
            " frame_id=" + std::to_string(frame->provenance.frame_id) +
            " queue_wait_ms=" +
            std::to_string(
                std::chrono::duration<double, std::milli>(
                    queue_stats.last_dequeue_age)
                    .count()) +
            " integrate_ms=" + std::to_string(last_integrate_ms_.load()) +
            " refresh_ms=" + std::to_string(last_refresh_ms_.load()) +
            " compute_total_ms=" +
            std::to_string(elapsedMs(compute_start,
                                     std::chrono::steady_clock::now())));
    publishCommit(std::move(record));
    maybeLogStatus();
  }
  {
    std::lock_guard<std::mutex> lock(commit_mutex_);
    for (const FrameKey& key : commit_order_) {
      RunLogger::logGlobal(
          "map_thread",
          "map_commit_evicted run_id=" + runIdString(key.run_id) +
              " frame_id=" + std::to_string(key.frame_id) +
              " reason=shutdown");
    }
    commits_by_frame_.clear();
    commit_order_.clear();
    for (const FrameKey& key : cancelled_candidate_order_) {
      RunLogger::logGlobal(
          "map_thread",
          "perception_barrier_cancelled run_id=" + runIdString(key.run_id) +
              " frame_id=" + std::to_string(key.frame_id) +
              " reason=shutdown_tombstone_cleanup");
    }
    cancelled_candidates_.clear();
    cancelled_candidate_order_.clear();
  }
  commit_cv_.notify_all();
}

void MapThread::onStopRequested() { commit_cv_.notify_all(); }

void MapThread::maybeLogStatus() {
  std::lock_guard<std::mutex> lock(status_mutex_);
  const auto now = std::chrono::steady_clock::now();
  if (now - last_status_log_time_ <
      std::chrono::duration<double>(config_.file_logging_period_sec)) {
    return;
  }
  last_status_log_time_ = now;
  const ChannelStats mapping = mapping_queue_.stats();

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
         << " last_projection_zbuffer_ms=" << last_projection_zbuffer_ms_.load()
         << " last_projection_compute_ms=" << last_projection_compute_ms_.load()
         << " last_integrate_ms=" << last_integrate_ms_.load()
         << " last_refresh_ms=" << last_refresh_ms_.load()
         << " mapping_queue_depth=" << mapping.depth
         << " mapping_queue_oldest_ms="
         << std::chrono::duration<double, std::milli>(mapping.oldest_age).count()
         << " mapping_queue_last_dequeue_ms="
         << std::chrono::duration<double, std::milli>(mapping.last_dequeue_age).count()
         << " mapping_queue_max_dequeue_ms="
         << std::chrono::duration<double, std::milli>(mapping.max_dequeue_age).count()
         << " mapping_queue_cancelled=" << mapping.cancelled
         << " mapping_queue_barrier_shed=" << mapping.barrier_shed
         << " mapping_queue_producer_wait_ms="
         << std::chrono::duration<double, std::milli>(mapping.producer_wait).count();
  RunLogger::logGlobal("map_thread", stream.str());
}

}  // namespace roomie
