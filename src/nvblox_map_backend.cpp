#include "roomie/pipeline/nvblox_map_backend.hpp"

#ifdef ROOMIE_ENABLE_NVBLOX

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

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

std::filesystem::path absoluteNormalizedPath(
    const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::path absolute = std::filesystem::absolute(path, error);
  return (error ? path : absolute).lexically_normal();
}

std::filesystem::path versionedCheckpointPath(
    const std::filesystem::path& base,
    const MapStamp& stamp) {
  std::ostringstream filename;
  filename << base.stem().string() << '.' << runIdString(stamp.map_epoch)
           << '.' << stamp.map_revision << base.extension().string();
  return base.parent_path() / filename.str();
}

std::string mapConfigFingerprint(const PipelineConfig& config) {
  std::ostringstream canonical;
  canonical << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "roomie.nvblox-checkpoint-config.v1"
            << "|voxel_size_m=" << config.voxel_size_m
            << "|truncation_distance_vox="
            << config.truncation_distance_vox
            << "|max_weight=" << config.max_weight
            << "|max_integration_distance_m="
            << config.max_integration_distance_m;
  std::uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char byte : canonical.str()) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  std::ostringstream encoded;
  encoded << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
          << hash;
  return encoded.str();
}

bool syncFile(const std::filesystem::path& path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    return false;
  }
  const bool ok = ::fsync(descriptor) == 0;
  ::close(descriptor);
  return ok;
}

bool syncDirectory(const std::filesystem::path& path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
  if (descriptor < 0) {
    return false;
  }
  const bool ok = ::fsync(descriptor) == 0;
  ::close(descriptor);
  return ok;
}

std::optional<std::string> fileContentHash(
    const std::filesystem::path& path,
    std::string* error) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    if (error != nullptr) {
      *error = "failed to open checkpoint for hashing: " + path.string();
    }
    return std::nullopt;
  }
  std::uint64_t hash = 1469598103934665603ULL;
  std::array<char, 1024U * 1024U> buffer{};
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = stream.gcount();
    for (std::streamsize index = 0; index < count; ++index) {
      hash ^= static_cast<unsigned char>(buffer[static_cast<std::size_t>(index)]);
      hash *= 1099511628211ULL;
    }
  }
  if (!stream.eof()) {
    if (error != nullptr) {
      *error = "failed while hashing checkpoint: " + path.string();
    }
    return std::nullopt;
  }
  std::ostringstream encoded;
  encoded << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
          << hash;
  return encoded.str();
}

std::optional<std::uint64_t> checkpointFileSize(
    const std::filesystem::path& path,
    std::string* error) {
  std::error_code size_error;
  const std::uintmax_t size = std::filesystem::file_size(path, size_error);
  if (size_error || size > std::numeric_limits<std::uint64_t>::max()) {
    if (error != nullptr) {
      *error = "failed to inspect checkpoint size: " + path.string() +
               (size_error ? " (" + size_error.message() + ')' : "");
    }
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(size);
}

std::int64_t unixTimeMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
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

nvblox::AxisAlignedBoundingBox detectionAabbWorld(const RawDetection& detection) {
  const Eigen::Vector3f half_size =
      0.5f * detection.size_m.cwiseMax(Eigen::Vector3f::Zero());
  const float cos_yaw = std::cos(detection.yaw_rad);
  const float sin_yaw = std::sin(detection.yaw_rad);
  nvblox::AxisAlignedBoundingBox aabb;
  bool initialized = false;
  for (const float x : {-half_size.x(), half_size.x()}) {
    for (const float y : {-half_size.y(), half_size.y()}) {
      for (const float z : {-half_size.z(), half_size.z()}) {
        const Eigen::Vector3f point_world(
            detection.center_world.x() + cos_yaw * x - sin_yaw * y,
            detection.center_world.y() + sin_yaw * x + cos_yaw * y,
            detection.center_world.z() + z);
        if (!initialized) {
          aabb = nvblox::AxisAlignedBoundingBox(point_world, point_world);
          initialized = true;
        } else {
          aabb.extend(point_world);
        }
      }
    }
  }
  return aabb;
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

BlockIndex toRoomieBlockIndex(const nvblox::Index3D& index) {
  return BlockIndex(index.x(), index.y(), index.z());
}

nvblox::Index3D toNvbloxBlockIndex(const BlockIndex& index) {
  return nvblox::Index3D(index.x, index.y, index.z);
}

class NvbloxMapBackend : public MapBackend {
 public:
  explicit NvbloxMapBackend(PipelineConfig config)
      : config_(std::move(config)),
        cuda_stream_(std::make_shared<nvblox::CudaStreamOwning>()),
        mapper_(createMapper(cuda_stream_)),
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
    if (mapper_->do_depth_preprocessing()) {
      RCLCPP_ERROR(logger_,
                   "Roomie direct nvblox integration does not support Mapper "
                   "depth preprocessing");
    }
  }

  MapCheckpointOperationResult loadCheckpoint(
      const MapCheckpointManifest* expected_manifest) override {
    MapCheckpointOperationResult result;
    // A durable SceneStore manifest is an automatic coordinated-recovery
    // request.  The load_map flag only controls an unaligned configured seed
    // path on a fresh store.
    if (!config_.load_map && expected_manifest == nullptr) {
      result.disposition = MapCheckpointDisposition::kNotRequested;
      return result;
    }

    std::filesystem::path path;
    if (expected_manifest != nullptr) {
      if (expected_manifest->backend != "nvblox") {
        result.disposition = MapCheckpointDisposition::kFailed;
        result.error = "checkpoint backend is not nvblox";
        return result;
      }
      if (expected_manifest->checkpoint_path.empty() ||
          expected_manifest->world_frame.empty() ||
          expected_manifest->config_fingerprint.empty() ||
          expected_manifest->file_size_bytes == 0 ||
          expected_manifest->content_hash.empty()) {
        result.disposition = MapCheckpointDisposition::kFailed;
        result.error = "checkpoint manifest is incomplete";
        return result;
      }
      if (expected_manifest->world_frame != config_.world_frame) {
        result.disposition = MapCheckpointDisposition::kFailed;
        result.error = "checkpoint world frame does not match configuration";
        return result;
      }
      if (expected_manifest->config_fingerprint !=
          mapConfigFingerprint(config_)) {
        result.disposition = MapCheckpointDisposition::kFailed;
        result.error =
            "checkpoint map configuration fingerprint does not match";
        return result;
      }
      path = expected_manifest->checkpoint_path;
      if (!config_.map_load_path.empty()) {
        const std::filesystem::path configured = absoluteNormalizedPath(
            resolveMapPath(config_.map_load_path,
                           "color_tsdf_map.nvblox", false));
        if (configured != absoluteNormalizedPath(path)) {
          result.disposition = MapCheckpointDisposition::kFailed;
          result.error =
              "configured map_load_path conflicts with durable manifest path";
          return result;
        }
      }
      result.manifest = *expected_manifest;
    } else {
      if (config_.map_load_path.empty()) {
        result.disposition = MapCheckpointDisposition::kFailed;
        result.error = "tsdf.load_map is enabled but map_load_path is empty";
        return result;
      }
      path = resolveMapPath(config_.map_load_path,
                            "color_tsdf_map.nvblox", false);
      result.manifest.backend = "nvblox";
      result.manifest.checkpoint_path =
          absoluteNormalizedPath(path).string();
      result.manifest.world_frame = config_.world_frame;
      result.manifest.config_fingerprint = mapConfigFingerprint(config_);
    }
    path = absoluteNormalizedPath(path);

    std::string metadata_error;
    const std::optional<std::uint64_t> size =
        checkpointFileSize(path, &metadata_error);
    const std::optional<std::string> hash =
        size ? fileContentHash(path, &metadata_error) : std::nullopt;
    if (!size || !hash) {
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = metadata_error;
      return result;
    }
    if (expected_manifest != nullptr &&
        (*size != expected_manifest->file_size_bytes ||
         *hash != expected_manifest->content_hash)) {
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = "checkpoint file does not match its durable manifest";
      return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!mapper_->loadMap(
            path.string(),
            nvblox::BlockMemoryPoolParams(nvblox::MemoryType::kUnified))) {
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = "failed to load nvblox map from " + path.string();
      return result;
    }
    surface_cache_ready_ = false;
    surface_cache_dirty_ = true;
    surface_cache_blocks_.clear();
    cached_tsdf_blocks_ = 0;
    cached_voxels_scanned_ = 0;
    cached_surface_points_ = 0;
    cached_map_version_ = 0;
    map_version_.store(0);
    integrated_through_ns_ = 0;
    result.manifest.backend = "nvblox";
    result.manifest.checkpoint_path = path.string();
    result.manifest.world_frame = config_.world_frame;
    result.manifest.config_fingerprint = mapConfigFingerprint(config_);
    result.manifest.file_size_bytes = *size;
    result.manifest.content_hash = *hash;
    result.disposition = MapCheckpointDisposition::kSucceeded;
    RCLCPP_INFO(logger_, "loaded nvblox checkpoint from %s",
                path.string().c_str());
    return result;
  }

  MapCheckpointOperationResult saveCheckpoint(
      const MapStamp& map_stamp) override {
    MapCheckpointOperationResult result;
    if (!config_.save_map) {
      result.disposition = MapCheckpointDisposition::kNotRequested;
      return result;
    }
    if (!map_stamp.map_epoch.valid()) {
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = "cannot save a checkpoint without a valid map epoch";
      return result;
    }
    if (config_.map_save_path.empty()) {
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = "tsdf.save_map is enabled but map_save_path is empty";
      return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (mapper_->tsdf_layer().numBlocks() == 0) {
      result.disposition = MapCheckpointDisposition::kSkipped;
      result.error = "nvblox map has no TSDF blocks";
      return result;
    }

    const std::filesystem::path configured = resolveMapPath(
        config_.map_save_path, "color_tsdf_map_roomie.nvblox", false);
    const std::filesystem::path final_path = absoluteNormalizedPath(
        versionedCheckpointPath(configured, map_stamp));
    std::error_code filesystem_error;
    std::filesystem::create_directories(final_path.parent_path(),
                                        filesystem_error);
    if (filesystem_error) {
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = "failed to create checkpoint directory: " +
                     filesystem_error.message();
      return result;
    }
    if (std::filesystem::exists(final_path, filesystem_error) ||
        filesystem_error) {
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = filesystem_error
                         ? "failed to inspect checkpoint destination: " +
                               filesystem_error.message()
                         : "refusing to overwrite immutable checkpoint " +
                               final_path.string();
      return result;
    }

    static std::atomic<std::uint64_t> temporary_sequence{1};
    std::ostringstream temporary_name;
    temporary_name << final_path.stem().string() << ".tmp."
                   << static_cast<long long>(::getpid()) << '.'
                   << temporary_sequence.fetch_add(1,
                                                    std::memory_order_relaxed)
                   << final_path.extension().string();
    const std::filesystem::path temporary_path =
        final_path.parent_path() / temporary_name.str();
    if (!mapper_->saveLayerCake(temporary_path.string())) {
      std::filesystem::remove(temporary_path, filesystem_error);
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = "failed to save nvblox checkpoint to " +
                     temporary_path.string();
      return result;
    }
    if (!syncFile(temporary_path)) {
      std::filesystem::remove(temporary_path, filesystem_error);
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = "failed to fsync nvblox checkpoint " +
                     temporary_path.string();
      return result;
    }
    std::filesystem::rename(temporary_path, final_path, filesystem_error);
    if (filesystem_error) {
      std::filesystem::remove(temporary_path, filesystem_error);
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = "failed to publish nvblox checkpoint: " +
                     filesystem_error.message();
      return result;
    }
    if (!syncDirectory(final_path.parent_path())) {
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = "checkpoint was renamed but its directory fsync failed";
      return result;
    }

    std::string metadata_error;
    const std::optional<std::uint64_t> size =
        checkpointFileSize(final_path, &metadata_error);
    const std::optional<std::string> hash =
        size ? fileContentHash(final_path, &metadata_error) : std::nullopt;
    if (!size || !hash || *size == 0) {
      result.disposition = MapCheckpointDisposition::kFailed;
      result.error = metadata_error.empty()
                         ? "saved checkpoint is empty"
                         : metadata_error;
      return result;
    }

    result.manifest.backend = "nvblox";
    result.manifest.checkpoint_path = final_path.string();
    result.manifest.world_frame = config_.world_frame;
    result.manifest.config_fingerprint = mapConfigFingerprint(config_);
    result.manifest.map_epoch = map_stamp.map_epoch;
    result.manifest.map_revision = map_stamp.map_revision;
    result.manifest.integrated_through_ns =
        map_stamp.integrated_through_ns;
    result.manifest.file_size_bytes = *size;
    result.manifest.content_hash = *hash;
    result.manifest.created_at_unix_ms = unixTimeMilliseconds();
    result.disposition = MapCheckpointDisposition::kSucceeded;
    RCLCPP_INFO(logger_, "saved immutable nvblox checkpoint to %s",
                final_path.string().c_str());
    return result;
  }

  MapIntegrationResult integrateFrame(const FrameBundle& frame) override {
    MapIntegrationResult result;
    std::lock_guard<std::mutex> lock(mutex_);
    result.map_revision = map_version_.load();
    result.integrated_through_ns = integrated_through_ns_;
    if (config_.freeze_tsdf_map) {
      result.success = true;
      result.updated_blocks_complete = true;
      return result;
    }
    if (!frame.depth || frame.depth->empty() || !frame.rgb ||
        !hasUsableIntrinsics(frame.intrinsics)) {
      result.error = "nvblox integration requires depth and valid intrinsics";
      return result;
    }
    const std::size_t expected_depth_size =
        static_cast<std::size_t>(frame.depth->width) *
        static_cast<std::size_t>(frame.depth->height);
    if (frame.depth->width <= 0 || frame.depth->height <= 0 ||
        frame.depth->depth_m.size() < expected_depth_size) {
      result.error = "nvblox integration received a truncated depth image";
      return result;
    }
    if (mapper_->do_depth_preprocessing()) {
      result.error =
          "direct nvblox integration requires depth preprocessing disabled";
      return result;
    }

    fillNvbloxImages(frame);
    const nvblox::MonoImageConstView mask_view(mask_image_);
    const nvblox::MaskedDepthImageConstView depth_view(
        depth_image_,
        std::optional<nvblox::ImageView<const std::uint8_t>>(mask_view),
        nvblox::MaskMode::kInverted);
    const nvblox::Transform T_world_camera = toNvbloxTransform(frame.T_world_camera);
    const nvblox::Camera frame_camera(frame.intrinsics.fx,
                                      frame.intrinsics.fy,
                                      frame.intrinsics.cx,
                                      frame.intrinsics.cy,
                                      frame.intrinsics.width,
                                      frame.intrinsics.height);

    std::vector<nvblox::Index3D> tsdf_updated_blocks;
    mapper_->tsdf_integrator().integrateFrame(depth_view,
                                              T_world_camera,
                                              frame_camera,
                                              &mapper_->tsdf_layer(),
                                              &tsdf_updated_blocks);
    mapper_->tsdf_layer().updateGpuHash(*cuda_stream_);

    const nvblox::MaskedColorImageConstView color_view(
        color_image_, nvblox::kMaskActiveEverywhere);
    const nvblox::MaskedDepthImageConstView color_depth_view(
        color_depth_image_, nvblox::kMaskActiveEverywhere);
    std::vector<nvblox::Index3D> color_updated_blocks;
    mapper_->color_integrator().integrateFrame(
        color_view,
        std::optional<nvblox::MaskedDepthImageConstView>(color_depth_view),
        T_world_camera,
        frame_camera,
        mapper_->tsdf_layer(),
        &mapper_->color_layer(),
        &color_updated_blocks);
    mapper_->color_layer().updateGpuHash(*cuda_stream_);
    cuda_stream_->synchronize();

    ++integrated_frames_;
    ++map_version_;
    integrated_through_ns_ = std::max(
        integrated_through_ns_, frame.provenance.sensor_time_ns);
    surface_cache_dirty_ = true;

    result.updated_blocks.reserve(tsdf_updated_blocks.size() +
                                  color_updated_blocks.size());
    for (const nvblox::Index3D& index : tsdf_updated_blocks) {
      result.updated_blocks.push_back(toRoomieBlockIndex(index));
    }
    for (const nvblox::Index3D& index : color_updated_blocks) {
      result.updated_blocks.push_back(toRoomieBlockIndex(index));
    }
    std::sort(result.updated_blocks.begin(), result.updated_blocks.end());
    result.updated_blocks.erase(
        std::unique(result.updated_blocks.begin(), result.updated_blocks.end()),
        result.updated_blocks.end());
    result.success = true;
    result.map_changed = true;
    result.map_revision = map_version_.load();
    result.integrated_through_ns = integrated_through_ns_;
    result.updated_blocks_complete = true;
    return result;
  }

  SurfaceRefreshResult refreshSurface(
      const MapIntegrationResult& integration,
      bool force_full_rebuild) override {
    SurfaceRefreshResult result;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t current_map_revision = map_version_.load();
    result.source_map_revision = current_map_revision;

    if (!integration.success && !force_full_rebuild) {
      result.error = integration.error.empty()
                         ? "cannot refresh surface after failed integration"
                         : integration.error;
      return result;
    }
    if (integration.map_revision != current_map_revision) {
      result.error = "surface refresh received a stale map revision";
      return result;
    }

    const bool revision_gap =
        surface_cache_ready_ && integration.map_changed &&
        current_map_revision != cached_map_version_ + 1U;
    const bool full_rebuild = force_full_rebuild || !surface_cache_ready_ ||
                              !integration.updated_blocks_complete ||
                              revision_gap;
    result.full_rebuild = full_rebuild;

    if (!integration.map_changed && !full_rebuild) {
      result.success = true;
      fillSnapshotHeaderLocked(&result.diagnostics);
      return result;
    }

    const auto build_start = std::chrono::steady_clock::now();
    std::uint64_t voxels_scanned = 0;
    if (full_rebuild) {
      SurfaceBlockMap rebuilt_blocks;
      std::vector<BlockIndex> block_indices;
      for (const nvblox::Index3D& index :
           mapper_->tsdf_layer().getAllBlockIndices()) {
        block_indices.push_back(toRoomieBlockIndex(index));
      }
      std::sort(block_indices.begin(), block_indices.end());
      block_indices.erase(
          std::unique(block_indices.begin(), block_indices.end()),
          block_indices.end());
      for (const BlockIndex& index : block_indices) {
        SurfaceBlockPtr block = extractSurfaceBlockLocked(index, &voxels_scanned);
        if (block) {
          rebuilt_blocks.emplace(index, block);
          result.blocks.push_back(std::move(block));
        }
      }
      for (const auto& old_entry : surface_cache_blocks_) {
        if (rebuilt_blocks.find(old_entry.first) == rebuilt_blocks.end()) {
          result.removed_blocks.push_back(old_entry.first);
        }
      }
      surface_cache_blocks_ = std::move(rebuilt_blocks);
    } else {
      std::vector<BlockIndex> dirty_blocks = integration.updated_blocks;
      std::sort(dirty_blocks.begin(), dirty_blocks.end());
      dirty_blocks.erase(
          std::unique(dirty_blocks.begin(), dirty_blocks.end()),
          dirty_blocks.end());
      for (const BlockIndex& index : dirty_blocks) {
        SurfaceBlockPtr block = extractSurfaceBlockLocked(index, &voxels_scanned);
        if (block) {
          surface_cache_blocks_[index] = block;
          result.blocks.push_back(std::move(block));
        } else {
          // A dirty TSDF block is not necessarily part of the extracted
          // surface. Report a removal only when this refresh actually removes
          // a block from the previous surface cache; otherwise the downstream
          // delta has no surface-space change at this index.
          if (surface_cache_blocks_.erase(index) != 0U) {
            result.removed_blocks.push_back(index);
          }
        }
      }
    }

    std::sort(result.blocks.begin(),
              result.blocks.end(),
              [](const SurfaceBlockPtr& lhs, const SurfaceBlockPtr& rhs) {
                return lhs->index() < rhs->index();
              });
    std::sort(result.removed_blocks.begin(), result.removed_blocks.end());
    result.removed_blocks.erase(
        std::unique(result.removed_blocks.begin(), result.removed_blocks.end()),
        result.removed_blocks.end());

    cached_tsdf_blocks_ = mapper_->tsdf_layer().numBlocks();
    cached_voxels_scanned_ = voxels_scanned;
    cached_surface_points_ = 0;
    for (const auto& entry : surface_cache_blocks_) {
      cached_surface_points_ += entry.second->pointCount();
    }
    cached_map_version_ = current_map_revision;
    surface_cache_ready_ = true;
    surface_cache_dirty_ = false;
    ++cache_rebuilds_;
    last_cache_rebuild_time_ = std::chrono::steady_clock::now();
    last_cache_build_ms_ = elapsedMs(build_start, last_cache_rebuild_time_);
    result.success = true;
    fillSnapshotHeaderLocked(&result.diagnostics);
    result.diagnostics.cache_build_ms = last_cache_build_ms_;
    result.diagnostics.surface_extract_ms = last_cache_build_ms_;

    RCLCPP_INFO(logger_,
                "refreshed nvblox surface cache mode=%s dirty_blocks=%lu "
                "removed_blocks=%lu cached_blocks=%lu surface_points=%lu "
                "voxels_scanned=%lu map_version=%lu build=%.1fms",
                full_rebuild ? "full" : "incremental",
                static_cast<unsigned long>(integration.updated_blocks.size()),
                static_cast<unsigned long>(result.removed_blocks.size()),
                static_cast<unsigned long>(surface_cache_blocks_.size()),
                static_cast<unsigned long>(cached_surface_points_),
                static_cast<unsigned long>(cached_voxels_scanned_),
                static_cast<unsigned long>(cached_map_version_),
                last_cache_build_ms_);
    RunLogger::logGlobal(
        "map",
        "refreshed_nvblox_surface_cache mode=" +
            std::string(full_rebuild ? "full" : "incremental") +
            " dirty_blocks=" +
            std::to_string(integration.updated_blocks.size()) +
            " removed_blocks=" +
            std::to_string(result.removed_blocks.size()) +
            " cached_blocks=" +
            std::to_string(surface_cache_blocks_.size()) +
            " surface_points=" + std::to_string(cached_surface_points_) +
            " voxels_scanned=" + std::to_string(cached_voxels_scanned_) +
            " map_version=" + std::to_string(cached_map_version_) +
            " build_ms=" + std::to_string(last_cache_build_ms_));
    return result;
  }

  MapBackendSnapshot snapshot() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    MapBackendSnapshot snapshot;
    fillSnapshotHeaderLocked(&snapshot);
    populateSnapshotFromCacheLocked(/*include_debug=*/true, &snapshot);
    return snapshot;
  }

  MapBackendSnapshot snapshotForView(const MapBackendView& view) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    MapBackendSnapshot snapshot;
    fillSnapshotHeaderLocked(&snapshot);
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
    std::vector<SurfaceBlockPtr> selected_blocks;
    selected_blocks.reserve(surface_cache_blocks_.size());
    std::size_t selected_points = 0;
    for (const SurfaceBlockPtr& block : orderedSurfaceBlocksLocked()) {
      const nvblox::AxisAlignedBoundingBox block_aabb(block->aabb().min,
                                                       block->aabb().max);
      if (block->empty() || !frustum.isAABBInView(block_aabb)) {
        continue;
      }
      selected_blocks.push_back(block);
      selected_points += block->pointCount();
    }
    snapshot.surface_points_world.reserve(selected_points);
    for (const SurfaceBlockPtr& block : selected_blocks) {
      for (const MapSurfacePoint& point : block->points()) {
        snapshot.surface_points_world.push_back(point.position_world);
      }
      ++snapshot.selected_blocks;
    }
    snapshot.frustum_filter_ms =
        elapsedMs(filter_start, std::chrono::steady_clock::now());
    return snapshot;
  }

  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> collectNearSurfaceVoxels(
      const RawDetection& detection) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    const nvblox::AxisAlignedBoundingBox detection_aabb = detectionAabbWorld(detection);
    std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> refs;
    for (const SurfaceBlockPtr& block : orderedSurfaceBlocksLocked()) {
      const nvblox::AxisAlignedBoundingBox block_aabb(block->aabb().min,
                                                       block->aabb().max);
      if (block->empty() || !block_aabb.intersects(detection_aabb)) {
        continue;
      }
      for (const MapSurfacePoint& point : block->points()) {
        if (!point.has_voxel_ref ||
            !pointInsideYawObb(point.position_world, detection)) {
          continue;
        }
        refs.push_back(point.voxel_ref);
      }
    }
    return refs;
  }

 private:
  using SurfaceBlockMap =
      std::unordered_map<BlockIndex, SurfaceBlockPtr, BlockIndexHash>;

  std::unique_ptr<nvblox::Mapper> createMapper(
      const std::shared_ptr<nvblox::CudaStream>& cuda_stream) const {
    auto mapper = std::make_unique<nvblox::Mapper>(
        config_.voxel_size_m,
        nvblox::BlockMemoryPoolParams(nvblox::MemoryType::kUnified),
        nvblox::ProjectiveLayerType::kTsdf,
        cuda_stream);
    return mapper;
  }

  void fillNvbloxImages(const FrameBundle& frame) {
    const int width = frame.depth->width;
    const int height = frame.depth->height;
    if (depth_image_.rows() != height || depth_image_.cols() != width) {
      depth_image_.resizeAsync(height, width, *cuda_stream_);
      color_depth_image_.resizeAsync(height, width, *cuda_stream_);
      mask_image_.resizeAsync(height, width, *cuda_stream_);
      color_image_.resizeAsync(height, width, *cuda_stream_);
      cuda_stream_->synchronize();
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
        float depth_m = frame.depth->depth_m[linear];
        const bool invalid_depth =
            !std::isfinite(depth_m) || depth_m < config_.depth_min_m ||
            depth_m > max_integration_depth;
        if (invalid_depth) {
          depth_m = 0.0f;
        }

        const bool is_robot =
            frame.robot_mask &&
            maskValueAt(*frame.robot_mask, u, v, width, height) >
            static_cast<std::uint8_t>(std::clamp(config_.mask_robot_threshold, 0, 255));
        mask_image_(v, u) = is_robot ? 1 : 0;
        depth_image_(v, u) = depth_m;
        color_depth_image_(v, u) = is_robot ? 0.0f : depth_m;
        color_image_(v, u) = colorAt(*frame.rgb, u, v, width, height);
      }
    }
  }

  void fillSnapshotHeaderLocked(MapBackendSnapshot* snapshot) const {
    const std::uint64_t latest_map_version = map_version_.load();
    snapshot->map_version = cached_map_version_;
    snapshot->latest_map_version = latest_map_version;
    snapshot->has_map = surface_cache_ready_ && cached_surface_points_ > 0;
    snapshot->surface_cache_ready = surface_cache_ready_;
    snapshot->cache_dirty =
        surface_cache_ready_ && cached_map_version_ != latest_map_version;
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
    for (const SurfaceBlockPtr& block : orderedSurfaceBlocksLocked()) {
      if (block->empty()) {
        continue;
      }
      for (const MapSurfacePoint& point : block->points()) {
        snapshot->surface_points_world.push_back(point.position_world);
        if (include_debug) {
          snapshot->debug_surface_points.push_back(point);
        }
      }
      ++snapshot->selected_blocks;
    }
    snapshot->surface_extract_ms =
        elapsedMs(copy_start, std::chrono::steady_clock::now());
  }

  SurfaceBlockPtr extractSurfaceBlockLocked(
      const BlockIndex& roomie_index,
      std::uint64_t* voxels_scanned) const {
    const nvblox::TsdfLayer& layer = mapper_->tsdf_layer();
    const nvblox::ColorLayer& color_layer = mapper_->color_layer();
    const float block_size = layer.block_size();
    const float voxel_size = layer.voxel_size();
    const float max_surface_distance =
        config_.surface_visualization_distance_vox * voxel_size;
    constexpr int kVoxelsPerSide = nvblox::VoxelBlock<nvblox::TsdfVoxel>::kVoxelsPerSide;

    const nvblox::Index3D block_index = toNvbloxBlockIndex(roomie_index);
    const nvblox::TsdfBlock::ConstPtr block =
        layer.getBlockAtIndex(block_index);
    if (block == nullptr) {
      return nullptr;
    }
    if (voxels_scanned != nullptr) {
      *voxels_scanned +=
          static_cast<std::uint64_t>(kVoxelsPerSide) *
          static_cast<std::uint64_t>(kVoxelsPerSide) *
          static_cast<std::uint64_t>(kVoxelsPerSide);
    }
    const nvblox::ColorBlock::ConstPtr color_block =
        color_layer.getBlockAtIndex(block_index);
    MapSurfacePointVector surface_points;
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
          MapSurfacePoint surface_point;
          surface_point.position_world =
              Eigen::Vector3f(point.x(), point.y(), point.z());
          surface_point.intensity = voxel.distance;
          surface_point.weight = voxel.weight;
          surface_point.has_voxel_ref = true;
          surface_point.voxel_ref.block_index = roomie_index.eigen();
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
          surface_points.push_back(std::move(surface_point));
        }
      }
    }
    if (surface_points.empty()) {
      return nullptr;
    }
    const nvblox::AxisAlignedBoundingBox block_aabb =
        nvblox::getAABBOfBlock(block_size, block_index);
    return makeSurfaceBlock(roomie_index,
                            SurfaceAabb(block_aabb.min(), block_aabb.max()),
                            std::move(surface_points));
  }

  std::vector<SurfaceBlockPtr> orderedSurfaceBlocksLocked() const {
    std::vector<SurfaceBlockPtr> result;
    result.reserve(surface_cache_blocks_.size());
    for (const auto& entry : surface_cache_blocks_) {
      result.push_back(entry.second);
    }
    std::sort(result.begin(),
              result.end(),
              [](const SurfaceBlockPtr& lhs, const SurfaceBlockPtr& rhs) {
                return lhs->index() < rhs->index();
              });
    return result;
  }

  PipelineConfig config_;
  mutable std::mutex mutex_;
  std::shared_ptr<nvblox::CudaStream> cuda_stream_;
  std::unique_ptr<nvblox::Mapper> mapper_;
  nvblox::Camera camera_;
  nvblox::DepthImage depth_image_;
  nvblox::DepthImage color_depth_image_;
  nvblox::ColorImage color_image_;
  nvblox::MonoImage mask_image_;
  std::atomic_uint64_t map_version_{0};
  std::atomic_uint64_t integrated_frames_{0};
  TimeNanoseconds integrated_through_ns_{0};
  bool surface_cache_ready_{false};
  bool surface_cache_dirty_{false};
  SurfaceBlockMap surface_cache_blocks_;
  std::uint64_t cached_tsdf_blocks_{0};
  std::uint64_t cached_voxels_scanned_{0};
  std::uint64_t cached_surface_points_{0};
  std::uint64_t cached_map_version_{0};
  std::uint64_t cache_rebuilds_{0};
  double last_cache_build_ms_{0.0};
  std::chrono::steady_clock::time_point last_cache_rebuild_time_ =
      std::chrono::steady_clock::time_point::min();
  rclcpp::Logger logger_;
};

}  // namespace

std::unique_ptr<MapBackend> createNvbloxMapBackend(const PipelineConfig& config) {
  return std::make_unique<NvbloxMapBackend>(config);
}

}  // namespace roomie

#endif  // ROOMIE_ENABLE_NVBLOX
