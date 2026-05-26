#include "roomie/pipeline/nvblox_map_backend.hpp"

#ifdef ROOMIE_ENABLE_NVBLOX

#include <algorithm>
#include <atomic>
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
#include <nvblox/map/blox.h>
#include <nvblox/map/layer.h>
#include <nvblox/map/voxels.h>
#include <nvblox/mapper/mapper.h>
#include <nvblox/sensors/camera.h>
#include <nvblox/sensors/image.h>
#include <rclcpp/rclcpp.hpp>

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
    snapshot.map_version = map_version_.load();
    snapshot.has_map = mapper_->tsdf_layer().numBlocks() > 0;
    collectSurfacePoints(&snapshot);
    return snapshot;
  }

  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> collectNearSurfaceVoxels(
      const RawDetection& detection) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> refs;

    const nvblox::TsdfLayer& layer = mapper_->tsdf_layer();
    const float block_size = layer.block_size();
    const float voxel_size = layer.voxel_size();
    const float max_surface_distance =
        config_.surface_visualization_distance_vox * voxel_size;
    constexpr int kVoxelsPerSide = nvblox::VoxelBlock<nvblox::TsdfVoxel>::kVoxelsPerSide;

    for (const nvblox::Index3D& block_index : layer.getAllBlockIndices()) {
      const nvblox::TsdfBlock::ConstPtr block = layer.getBlockAtIndex(block_index);
      if (block == nullptr) {
        continue;
      }
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
            if (!pointInsideYawObb(Eigen::Vector3f(point.x(), point.y(), point.z()),
                                   detection)) {
              continue;
            }
            VoxelRef ref;
            ref.block_index = Eigen::Vector3i(block_index.x(), block_index.y(), block_index.z());
            ref.voxel_index = Eigen::Vector3i(x, y, z);
            refs.push_back(ref);
          }
        }
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

  void collectSurfacePoints(MapBackendSnapshot* snapshot) const {
    const nvblox::TsdfLayer& layer = mapper_->tsdf_layer();
    const nvblox::ColorLayer& color_layer = mapper_->color_layer();
    const float block_size = layer.block_size();
    const float voxel_size = layer.voxel_size();
    const float max_surface_distance =
        config_.surface_visualization_distance_vox * voxel_size;
    constexpr int kVoxelsPerSide = nvblox::VoxelBlock<nvblox::TsdfVoxel>::kVoxelsPerSide;

    const std::vector<nvblox::Index3D> block_indices = layer.getAllBlockIndices();
    snapshot->surface_points_world.reserve(block_indices.size() * 32);
    snapshot->debug_surface_points.reserve(block_indices.size() * 32);
    for (const nvblox::Index3D& block_index : block_indices) {
      const nvblox::TsdfBlock::ConstPtr block = layer.getBlockAtIndex(block_index);
      if (block == nullptr) {
        continue;
      }
      const nvblox::ColorBlock::ConstPtr color_block =
          color_layer.getBlockAtIndex(block_index);
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
            snapshot->surface_points_world.push_back(point_world);

            MapSurfacePoint surface_point;
            surface_point.position_world = point_world;
            surface_point.intensity = voxel.distance;
            surface_point.weight = voxel.weight;
            if (color_block != nullptr) {
              const nvblox::ColorVoxel& color_voxel = (*color_block)(voxel_index);
              if (color_voxel.weight >= config_.min_color_weight) {
                surface_point.r = color_voxel.color.r();
                surface_point.g = color_voxel.color.g();
                surface_point.b = color_voxel.color.b();
              }
            }
            snapshot->debug_surface_points.push_back(surface_point);
          }
        }
      }
    }
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
  rclcpp::Logger logger_;
};

}  // namespace

std::unique_ptr<MapBackend> createNvbloxMapBackend(const PipelineConfig& config) {
  return std::make_unique<NvbloxMapBackend>(config);
}

}  // namespace roomie

#endif  // ROOMIE_ENABLE_NVBLOX
