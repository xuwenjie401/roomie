#include "roomie/pipeline/nvblox_map_backend.hpp"

#ifdef ROOMIE_ENABLE_NVBLOX

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include <nvblox/core/cuda_stream.h>
#include <nvblox/core/indexing.h>
#include <nvblox/geometry/bounding_boxes.h>
#include <nvblox/map/blox.h>
#include <nvblox/map/layer.h>
#include <nvblox/map/voxels.h>
#include <nvblox/mapper/mapper.h>
#include <nvblox/sensors/camera.h>
#include <nvblox/sensors/image.h>
#include <rclcpp/rclcpp.hpp>

#include "roomie/utils/run_logger.hpp"

namespace roomie {
namespace {

bool hasUsableIntrinsics(const CameraIntrinsics& intrinsics) {
  return intrinsics.fx > 0.0f && intrinsics.fy > 0.0f;
}

nvblox::Transform toNvbloxTransform(const Eigen::Isometry3f& transform) {
  nvblox::Transform output = nvblox::Transform::Identity();
  output.linear() = transform.linear();
  output.translation() = transform.translation();
  return output;
}

std::filesystem::path resolveMapPath(const std::string& path_string,
                                     const std::string& default_filename,
                                     bool create_parent_dirs) {
  namespace fs = std::filesystem;
  fs::path path(path_string);
  if (path_string.empty()) {
    return path;
  }
  const bool ends_with_separator =
      path_string.back() == '/' || path_string.back() == '\\';
  std::error_code error;
  if (ends_with_separator || fs::is_directory(path, error)) {
    path /= default_filename;
  }
  if (create_parent_dirs) {
    fs::create_directories(path.parent_path(), error);
  }
  return path;
}

bool pointInsideYawObb(const Eigen::Vector3f& point_world, const RawDetection& detection) {
  if (detection.size_m.x() <= 0.0f || detection.size_m.y() <= 0.0f ||
      detection.size_m.z() <= 0.0f) {
    return false;
  }
  const Eigen::Vector3f delta = point_world - detection.center_world;
  const float cos_yaw = std::cos(-detection.yaw_rad);
  const float sin_yaw = std::sin(-detection.yaw_rad);
  const Eigen::Vector3f local(cos_yaw * delta.x() - sin_yaw * delta.y(),
                              sin_yaw * delta.x() + cos_yaw * delta.y(),
                              delta.z());
  const Eigen::Vector3f half_size = 0.5f * detection.size_m;
  return std::abs(local.x()) <= half_size.x() &&
         std::abs(local.y()) <= half_size.y() &&
         std::abs(local.z()) <= half_size.z();
}

double elapsedMs(std::chrono::steady_clock::time_point start,
                 std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::uint8_t maskValueAt(const ImageBuffer& mask, int x, int y, int width, int height) {
  if (mask.empty() || mask.width <= 0 || mask.height <= 0 || mask.channels <= 0) {
    return 0;
  }
  const int mx = std::clamp(
      static_cast<int>((static_cast<float>(x) + 0.5f) *
                       static_cast<float>(mask.width) / static_cast<float>(width)),
      0,
      mask.width - 1);
  const int my = std::clamp(
      static_cast<int>((static_cast<float>(y) + 0.5f) *
                       static_cast<float>(mask.height) / static_cast<float>(height)),
      0,
      mask.height - 1);
  const std::size_t offset =
      (static_cast<std::size_t>(my) * static_cast<std::size_t>(mask.width) +
       static_cast<std::size_t>(mx)) *
      static_cast<std::size_t>(mask.channels);
  if (offset >= mask.data.size()) {
    return 0;
  }
  return mask.data[offset];
}

nvblox::Color colorAt(const ImageBuffer& rgb, int x, int y, int width, int height) {
  if (rgb.empty() || rgb.channels < 3 || rgb.width <= 0 || rgb.height <= 0) {
    return nvblox::Color::Gray();
  }
  const int rx = std::clamp(
      static_cast<int>((static_cast<float>(x) + 0.5f) *
                       static_cast<float>(rgb.width) / static_cast<float>(width)),
      0,
      rgb.width - 1);
  const int ry = std::clamp(
      static_cast<int>((static_cast<float>(y) + 0.5f) *
                       static_cast<float>(rgb.height) / static_cast<float>(height)),
      0,
      rgb.height - 1);
  const std::size_t offset =
      (static_cast<std::size_t>(ry) * static_cast<std::size_t>(rgb.width) +
       static_cast<std::size_t>(rx)) *
      static_cast<std::size_t>(rgb.channels);
  if (offset + 2 >= rgb.data.size()) {
    return nvblox::Color::Gray();
  }
  return nvblox::Color(rgb.data[offset], rgb.data[offset + 1], rgb.data[offset + 2]);
}

class NvbloxMapBackend : public MapBackend {
 public:
  explicit NvbloxMapBackend(PipelineConfig config)
      : config_(std::move(config)),
        mapper_(createMapper()),
        camera_(config_.camera_fx,
                config_.camera_fy,
                config_.camera_cx,
                config_.camera_cy,
                config_.camera_width,
                config_.camera_height),
        depth_image_(nvblox::MemoryType::kUnified),
        color_depth_image_(nvblox::MemoryType::kUnified),
        color_image_(nvblox::MemoryType::kUnified),
        mask_image_(nvblox::MemoryType::kUnified),
        logger_(rclcpp::get_logger("roomie.nvblox_map_backend")) {
    mapper_->tsdf_integrator().max_integration_distance_m(
        config_.max_integration_distance_m);
    mapper_->tsdf_integrator().truncation_distance_vox(
        config_.truncation_distance_vox);
    mapper_->tsdf_integrator().max_weight(config_.max_weight);
  }

  void integrateFrame(const MappingFrame& frame) override {
    if (config_.freeze_tsdf_map) {
      return;
    }
    if (frame.depth.empty() || !hasUsableIntrinsics(frame.intrinsics)) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    fillNvbloxImages(frame);
    const nvblox::MonoImageConstView mask_view(mask_image_);
    const nvblox::MaskedDepthImageConstView depth_view(
        depth_image_,
        std::optional<nvblox::ImageView<const std::uint8_t>>(mask_view),
        nvblox::MaskMode::kInverted);
    const nvblox::Transform T_world_camera = toNvbloxTransform(frame.T_world_camera);
    mapper_->integrateDepth(depth_view, T_world_camera, camera_);
    mapper_->integrateColor(color_image_, color_depth_image_, T_world_camera, camera_);
    cuda_stream_.synchronize();

    ++integrated_frames_;
    ++map_version_;
  }

  MapBackendSnapshot snapshot() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    MapBackendSnapshot snapshot;
    const std::uint64_t cache_rebuilds_before = cache_rebuilds_;
    ensureSurfaceCacheLocked();
    fillSnapshotHeaderLocked(&snapshot);
    if (snapshot.cache_rebuilds > cache_rebuilds_before) {
      snapshot.cache_build_ms = last_cache_build_ms_;
    }
    populateSnapshotFromCacheLocked(/*include_debug=*/true, &snapshot);
    return snapshot;
  }

  MapBackendSnapshot snapshotForView(const MapBackendView& view) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    MapBackendSnapshot snapshot;
    const std::uint64_t cache_rebuilds_before = cache_rebuilds_;
    ensureSurfaceCacheLocked();
    fillSnapshotHeaderLocked(&snapshot);
    if (snapshot.cache_rebuilds > cache_rebuilds_before) {
      snapshot.cache_build_ms = last_cache_build_ms_;
    }
    if (!snapshot.has_map) {
      return snapshot;
    }
    if (!hasUsableIntrinsics(view.intrinsics) || view.max_depth_m <= view.min_depth_m) {
      populateSnapshotFromCacheLocked(/*include_debug=*/false, &snapshot);
      return snapshot;
    }

    const nvblox::Camera camera(view.intrinsics.fx,
                                view.intrinsics.fy,
                                view.intrinsics.cx,
                                view.intrinsics.cy,
                                view.intrinsics.width,
                                view.intrinsics.height);
    const nvblox::Transform T_world_camera = toNvbloxTransform(view.T_world_camera);
    const nvblox::Frustum frustum(
        camera, T_world_camera, view.min_depth_m, view.max_depth_m);
    const auto filter_start = std::chrono::steady_clock::now();
    snapshot.view_filtered = true;
    std::vector<const SurfaceBlockCache*> selected_blocks;
    selected_blocks.reserve(surface_cache_blocks_.size());
    std::size_t selected_points = 0;
    for (const SurfaceBlockCache& block_cache : surface_cache_blocks_) {
      if (block_cache.surface_points_world.empty() ||
          !frustum.isAABBInView(block_cache.aabb_world)) {
        continue;
      }
      selected_blocks.push_back(&block_cache);
      selected_points += block_cache.surface_points_world.size();
    }
    snapshot.surface_points_world.reserve(selected_points);
    for (const SurfaceBlockCache* block_cache : selected_blocks) {
      snapshot.surface_points_world.insert(snapshot.surface_points_world.end(),
                                           block_cache->surface_points_world.begin(),
                                           block_cache->surface_points_world.end());
      ++snapshot.selected_blocks;
    }
    snapshot.frustum_filter_ms =
        elapsedMs(filter_start, std::chrono::steady_clock::now());
    return snapshot;
  }

  std::shared_ptr<const GeometrySurfaceCache> geometrySurfaceCache() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureSurfaceCacheLocked();
    return geometry_surface_cache_;
  }

  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> collectNearSurfaceVoxels(
      const RawDetection& detection) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureSurfaceCacheLocked();
    std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> refs;
    for (const SurfaceBlockCache& block_cache : surface_cache_blocks_) {
      for (std::size_t i = 0; i < block_cache.surface_points_world.size(); ++i) {
        if (!pointInsideYawObb(block_cache.surface_points_world[i], detection)) {
          continue;
        }
        VoxelRef ref;
        ref.block_index = Eigen::Vector3i(block_cache.block_index.x(),
                                          block_cache.block_index.y(),
                                          block_cache.block_index.z());
        const nvblox::Index3D& voxel_index = block_cache.voxel_indices[i];
        ref.voxel_index = Eigen::Vector3i(voxel_index.x(), voxel_index.y(), voxel_index.z());
        refs.push_back(ref);
      }
    }
    return refs;
  }

  void saveIfRequested() override {
    if (!config_.save_map) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (mapper_->tsdf_layer().numBlocks() == 0 || config_.map_save_path.empty()) {
      return;
    }
    const std::filesystem::path path =
        resolveMapPath(config_.map_save_path, "color_tsdf_map_roomie.nvblox", true);
    if (!mapper_->saveLayerCake(path.string())) {
      RCLCPP_ERROR(logger_, "failed to save nvblox map to %s", path.string().c_str());
    }
  }

 private:
  struct SurfaceBlockCache {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    nvblox::Index3D block_index = nvblox::Index3D::Zero();
    nvblox::AxisAlignedBoundingBox aabb_world;
    WorldPointVector surface_points_world;
    std::vector<MapSurfacePoint, Eigen::aligned_allocator<MapSurfacePoint>> debug_surface_points;
    std::vector<nvblox::Index3D, Eigen::aligned_allocator<nvblox::Index3D>> voxel_indices;
  };

  std::unique_ptr<nvblox::Mapper> createMapper() const {
    auto mapper = std::make_unique<nvblox::Mapper>(
        config_.voxel_size_m,
        nvblox::BlockMemoryPoolParams(nvblox::MemoryType::kUnified),
        nvblox::ProjectiveLayerType::kTsdf);
    if (config_.load_map && !config_.map_load_path.empty()) {
      const std::filesystem::path path =
          resolveMapPath(config_.map_load_path, "color_tsdf_map.nvblox", false);
      mapper->loadMap(path.string(),
                      nvblox::BlockMemoryPoolParams(nvblox::MemoryType::kUnified));
    }
    return mapper;
  }

  void fillNvbloxImages(const MappingFrame& frame) {
    const int width = frame.depth.width;
    const int height = frame.depth.height;
    if (depth_image_.rows() != height || depth_image_.cols() != width) {
      depth_image_.resizeAsync(height, width, cuda_stream_);
      color_depth_image_.resizeAsync(height, width, cuda_stream_);
      mask_image_.resizeAsync(height, width, cuda_stream_);
      color_image_.resizeAsync(height, width, cuda_stream_);
      cuda_stream_.synchronize();
    }

    const float max_integration_depth =
        config_.max_integration_distance_m > 0.0f
            ? std::min(config_.depth_max_m, config_.max_integration_distance_m)
            : config_.depth_max_m;

    for (int v = 0; v < height; ++v) {
      for (int u = 0; u < width; ++u) {
        const std::size_t linear =
            static_cast<std::size_t>(v) * static_cast<std::size_t>(width) +
            static_cast<std::size_t>(u);
        float depth_m = frame.depth.depth_m[linear];
        const bool invalid_depth =
            !std::isfinite(depth_m) || depth_m < config_.depth_min_m ||
            depth_m > max_integration_depth;
        if (invalid_depth) {
          depth_m = 0.0f;
        }

        const bool is_robot =
            maskValueAt(frame.robot_mask, u, v, width, height) >
            static_cast<std::uint8_t>(std::clamp(config_.mask_robot_threshold, 0, 255));
        mask_image_(v, u) = is_robot ? 1 : 0;
        depth_image_(v, u) = depth_m;
        color_depth_image_(v, u) = is_robot ? 0.0f : depth_m;
        color_image_(v, u) = colorAt(frame.rgb, u, v, width, height);
      }
    }
  }

  void ensureSurfaceCacheLocked() const {
    if (surface_cache_ready_ || mapper_->tsdf_layer().numBlocks() == 0) {
      return;
    }
    rebuildSurfaceCacheLocked();
  }

  void fillSnapshotHeaderLocked(MapBackendSnapshot* snapshot) const {
    snapshot->map_version = cached_map_version_;
    snapshot->has_map = surface_cache_ready_ && cached_surface_points_ > 0;
    snapshot->surface_cache_ready = surface_cache_ready_;
    snapshot->tsdf_blocks = cached_tsdf_blocks_;
    snapshot->surface_voxels_scanned = cached_voxels_scanned_;
    snapshot->cached_surface_points = cached_surface_points_;
    snapshot->cache_rebuilds = cache_rebuilds_;
  }

  void populateSnapshotFromCacheLocked(bool include_debug,
                                       MapBackendSnapshot* snapshot) const {
    const auto copy_start = std::chrono::steady_clock::now();
    snapshot->surface_points_world.reserve(
        static_cast<std::size_t>(cached_surface_points_));
    if (include_debug) {
      snapshot->debug_surface_points.reserve(
          static_cast<std::size_t>(cached_surface_points_));
    }
    for (const SurfaceBlockCache& block_cache : surface_cache_blocks_) {
      if (block_cache.surface_points_world.empty()) {
        continue;
      }
      snapshot->surface_points_world.insert(snapshot->surface_points_world.end(),
                                           block_cache.surface_points_world.begin(),
                                           block_cache.surface_points_world.end());
      if (include_debug) {
        snapshot->debug_surface_points.insert(snapshot->debug_surface_points.end(),
                                             block_cache.debug_surface_points.begin(),
                                             block_cache.debug_surface_points.end());
      }
      ++snapshot->selected_blocks;
    }
    snapshot->surface_extract_ms =
        elapsedMs(copy_start, std::chrono::steady_clock::now());
  }

  void rebuildSurfaceCacheLocked() const {
    const auto build_start = std::chrono::steady_clock::now();
    surface_cache_blocks_.clear();
    auto geometry_cache = std::make_shared<GeometrySurfaceCache>();
    cached_tsdf_blocks_ = 0;
    cached_voxels_scanned_ = 0;
    cached_surface_points_ = 0;

    const nvblox::TsdfLayer& layer = mapper_->tsdf_layer();
    const nvblox::ColorLayer& color_layer = mapper_->color_layer();
    const float block_size = layer.block_size();
    const float voxel_size = layer.voxel_size();
    const float max_surface_distance =
        config_.surface_visualization_distance_vox * voxel_size;
    constexpr int kVoxelsPerSide = nvblox::VoxelBlock<nvblox::TsdfVoxel>::kVoxelsPerSide;

    const std::vector<nvblox::Index3D> block_indices = layer.getAllBlockIndices();
    cached_tsdf_blocks_ = static_cast<std::uint64_t>(block_indices.size());
    surface_cache_blocks_.reserve(block_indices.size());
    for (const nvblox::Index3D& block_index : block_indices) {
      const nvblox::TsdfBlock::ConstPtr block = layer.getBlockAtIndex(block_index);
      if (block == nullptr) {
        continue;
      }
      cached_voxels_scanned_ +=
          static_cast<std::uint64_t>(kVoxelsPerSide) *
          static_cast<std::uint64_t>(kVoxelsPerSide) *
          static_cast<std::uint64_t>(kVoxelsPerSide);
      const nvblox::ColorBlock::ConstPtr color_block =
          color_layer.getBlockAtIndex(block_index);
      SurfaceBlockCache block_cache;
      block_cache.block_index = block_index;
      block_cache.aabb_world = nvblox::getAABBOfBlock(block_size, block_index);
      for (int x = 0; x < kVoxelsPerSide; ++x) {
        for (int y = 0; y < kVoxelsPerSide; ++y) {
          for (int z = 0; z < kVoxelsPerSide; ++z) {
            const nvblox::Index3D voxel_index(x, y, z);
            const nvblox::TsdfVoxel& voxel = (*block)(voxel_index);
            if (voxel.weight < config_.min_visualization_weight ||
                std::abs(voxel.distance) > max_surface_distance) {
              continue;
            }
            const nvblox::Vector3f point =
                nvblox::getCenterPositionFromBlockIndexAndVoxelIndex(
                    block_size, block_index, voxel_index);
            const Eigen::Vector3f point_world(point.x(), point.y(), point.z());
            block_cache.surface_points_world.push_back(point_world);
            block_cache.voxel_indices.push_back(voxel_index);

            MapSurfacePoint surface_point;
            surface_point.position_world = point_world;
            surface_point.intensity = voxel.distance;
            surface_point.weight = voxel.weight;
            surface_point.has_voxel_ref = true;
            surface_point.voxel_ref.block_index =
                Eigen::Vector3i(block_index.x(), block_index.y(), block_index.z());
            surface_point.voxel_ref.voxel_index =
                Eigen::Vector3i(voxel_index.x(), voxel_index.y(), voxel_index.z());
            if (color_block != nullptr) {
              const nvblox::ColorVoxel& color_voxel = (*color_block)(voxel_index);
              if (color_voxel.weight >= config_.min_color_weight) {
                surface_point.r = color_voxel.color.r();
                surface_point.g = color_voxel.color.g();
                surface_point.b = color_voxel.color.b();
              }
            }
            block_cache.debug_surface_points.push_back(surface_point);
          }
        }
      }
      cached_surface_points_ +=
          static_cast<std::uint64_t>(block_cache.surface_points_world.size());
      geometry_cache->surface_points.insert(geometry_cache->surface_points.end(),
                                            block_cache.debug_surface_points.begin(),
                                            block_cache.debug_surface_points.end());
      surface_cache_blocks_.push_back(std::move(block_cache));
    }
    surface_cache_ready_ = true;
    cached_map_version_ = map_version_.load();
    ++cache_rebuilds_;
    geometry_cache->map_version = cached_map_version_;
    geometry_cache->cache_rebuilds = cache_rebuilds_;
    geometry_cache->tsdf_blocks = cached_tsdf_blocks_;
    geometry_cache->surface_voxels_scanned = cached_voxels_scanned_;
    geometry_cache->has_map = cached_surface_points_ > 0;
    geometry_surface_cache_ = std::move(geometry_cache);
    last_cache_build_ms_ = elapsedMs(build_start, std::chrono::steady_clock::now());
    RCLCPP_INFO(logger_,
                "built nvblox surface cache blocks=%lu surface_points=%lu "
                "voxels_scanned=%lu build=%.1fms",
                static_cast<unsigned long>(cached_tsdf_blocks_),
                static_cast<unsigned long>(cached_surface_points_),
                static_cast<unsigned long>(cached_voxels_scanned_),
                last_cache_build_ms_);
    RunLogger::logGlobal("map",
                         "built_nvblox_surface_cache blocks=" +
                             std::to_string(cached_tsdf_blocks_) +
                             " surface_points=" +
                             std::to_string(cached_surface_points_) +
                             " voxels_scanned=" +
                             std::to_string(cached_voxels_scanned_) +
                             " map_version=" +
                             std::to_string(cached_map_version_) +
                             " build_ms=" +
                             std::to_string(last_cache_build_ms_));
  }

  PipelineConfig config_;
  mutable std::mutex mutex_;
  std::unique_ptr<nvblox::Mapper> mapper_;
  nvblox::Camera camera_;
  nvblox::CudaStreamOwning cuda_stream_;
  nvblox::DepthImage depth_image_;
  nvblox::DepthImage color_depth_image_;
  nvblox::ColorImage color_image_;
  nvblox::MonoImage mask_image_;
  std::atomic_uint64_t map_version_{0};
  std::atomic_uint64_t integrated_frames_{0};
  mutable bool surface_cache_ready_{false};
  mutable std::vector<SurfaceBlockCache, Eigen::aligned_allocator<SurfaceBlockCache>>
      surface_cache_blocks_;
  mutable std::shared_ptr<const GeometrySurfaceCache> geometry_surface_cache_;
  mutable std::uint64_t cached_tsdf_blocks_{0};
  mutable std::uint64_t cached_voxels_scanned_{0};
  mutable std::uint64_t cached_surface_points_{0};
  mutable std::uint64_t cached_map_version_{0};
  mutable std::uint64_t cache_rebuilds_{0};
  mutable double last_cache_build_ms_{0.0};
  rclcpp::Logger logger_;
};

}  // namespace

std::unique_ptr<MapBackend> createNvbloxMapBackend(const PipelineConfig& config) {
  return std::make_unique<NvbloxMapBackend>(config);
}

}  // namespace roomie

#endif  // ROOMIE_ENABLE_NVBLOX
