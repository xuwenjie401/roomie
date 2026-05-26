#include "roomie/pipeline/map_thread.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <utility>

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
  const MapBackendSnapshot snapshot = map_backend_->snapshot();
  if (!snapshot.has_map) {
    return std::nullopt;
  }
  return projectWorldPointsToPatchDepth(
      frame, snapshot.surface_points_world, snapshot.map_version);
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

  for (int index = 0; index < PatchDepth::kSize; ++index) {
    std::vector<float>& depths = patch_depths[static_cast<std::size_t>(index)];
    if (depths.empty()) {
      continue;
    }
    result.values[static_cast<std::size_t>(index)] = medianInPlace(&depths);
    ++result.valid_patches;
  }

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
  return map_backend_->snapshot();
}

void MapThread::run() {
  while (!stopRequested()) {
    MappingFrame frame;
    if (!mapping_queue_.waitPopFor(&frame, std::chrono::milliseconds(50))) {
      continue;
    }
    map_backend_->integrateFrame(frame);
  }
}

}  // namespace roomie
