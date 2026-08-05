#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "roomie/pipeline/nvblox_map_backend.hpp"
#include "roomie/pipeline/surface_snapshot.hpp"

#ifndef ROOMIE_ENABLE_NVBLOX
#error "This conformance test must only be built with ROOMIE_ENABLE_NVBLOX=ON"
#endif

namespace roomie {
namespace {

constexpr int kWidth = 64;
constexpr int kHeight = 48;

class TemporaryCheckpointDirectory {
 public:
  TemporaryCheckpointDirectory() {
    char pattern[] = "/tmp/roomie_nvblox_checkpoint_XXXXXX";
    char* created = mkdtemp(pattern);
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }
  ~TemporaryCheckpointDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

PipelineConfig testConfig() {
  PipelineConfig config;
  config.map_backend = "nvblox";
  config.load_map = false;
  config.save_map = false;
  config.freeze_tsdf_map = false;
  config.camera_width = kWidth;
  config.camera_height = kHeight;
  config.camera_fx = 60.0f;
  config.camera_fy = 60.0f;
  config.camera_cx = (static_cast<float>(kWidth) - 1.0f) * 0.5f;
  config.camera_cy = (static_cast<float>(kHeight) - 1.0f) * 0.5f;
  config.depth_min_m = 0.1f;
  config.depth_max_m = 4.0f;
  config.max_integration_distance_m = 4.0f;
  config.voxel_size_m = 0.05f;
  config.truncation_distance_vox = 4.0f;
  config.max_weight = 20.0f;
  config.min_visualization_weight = 0.1f;
  config.min_color_weight = 0.1f;
  config.surface_visualization_distance_vox = 1.5f;
  config.mask_robot_threshold = 0;
  config.file_logging_enabled = false;
  return config;
}

ImageBuffer rgbImage() {
  ImageBuffer rgb;
  rgb.width = kWidth;
  rgb.height = kHeight;
  rgb.channels = 3;
  rgb.encoding = "rgb8";
  rgb.data.resize(static_cast<std::size_t>(kWidth * kHeight * 3));
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const std::size_t offset =
          static_cast<std::size_t>((y * kWidth + x) * 3);
      rgb.data[offset] = static_cast<std::uint8_t>(64 + x);
      rgb.data[offset + 1] = static_cast<std::uint8_t>(96 + y);
      rgb.data[offset + 2] = 160;
    }
  }
  return rgb;
}

DepthBuffer planeDepth(float depth_m) {
  DepthBuffer depth;
  depth.width = kWidth;
  depth.height = kHeight;
  depth.depth_m.assign(static_cast<std::size_t>(kWidth * kHeight), depth_m);
  return depth;
}

ImageBuffer maskImage(std::uint8_t value) {
  ImageBuffer mask;
  mask.width = kWidth;
  mask.height = kHeight;
  mask.channels = 1;
  mask.encoding = "mono8";
  mask.data.assign(static_cast<std::size_t>(kWidth * kHeight), value);
  return mask;
}

FrameBundle frame(FrameId frame_id,
                  TimeNanoseconds sensor_time_ns,
                  DepthBuffer depth,
                  std::uint8_t robot_mask_value = 0) {
  FrameBundle bundle;
  bundle.provenance.run_id = RunId{0x4e56424c4f58ULL, 0x434f4e464f524dULL};
  bundle.provenance.frame_id = frame_id;
  bundle.provenance.sensor_time_ns = sensor_time_ns;
  bundle.camera_id = "synthetic_nvblox_conformance";
  bundle.rgb = std::make_shared<const ImageBuffer>(rgbImage());
  bundle.depth =
      std::make_shared<const DepthBuffer>(std::move(depth));
  bundle.robot_mask = std::make_shared<const ImageBuffer>(
      maskImage(robot_mask_value));
  bundle.intrinsics.width = kWidth;
  bundle.intrinsics.height = kHeight;
  bundle.intrinsics.fx = 60.0f;
  bundle.intrinsics.fy = 60.0f;
  bundle.intrinsics.cx = (static_cast<float>(kWidth) - 1.0f) * 0.5f;
  bundle.intrinsics.cy = (static_cast<float>(kHeight) - 1.0f) * 0.5f;
  bundle.T_world_camera = Eigen::Isometry3f::Identity();
  return bundle;
}

MapStamp mapStamp(const RunId& epoch,
                  const MapIntegrationResult& integration) {
  return MapStamp{epoch, integration.map_revision,
                  integration.integrated_through_ns};
}

SurfaceStamp surfaceStamp(const RunId& epoch,
                          std::uint64_t surface_revision,
                          const MapIntegrationResult& integration) {
  return SurfaceStamp{epoch, surface_revision, integration.map_revision};
}

SurfaceSnapshotPtr buildFullSnapshot(
    const RunId& epoch,
    std::uint64_t surface_revision,
    const MapIntegrationResult& integration,
    const SurfaceRefreshResult& refresh) {
  return SurfaceSnapshotBuilder::buildFrom(
             nullptr, mapStamp(epoch, integration),
             surfaceStamp(epoch, surface_revision, integration),
             refresh.blocks, {})
      .snapshot;
}

void expectEquivalentPoints(const MapSurfacePoint& incremental,
                            const MapSurfacePoint& full) {
  EXPECT_TRUE(incremental.position_world.isApprox(full.position_world, 1e-6f));
  EXPECT_FLOAT_EQ(incremental.intensity, full.intensity);
  EXPECT_FLOAT_EQ(incremental.weight, full.weight);
  EXPECT_EQ(incremental.r, full.r);
  EXPECT_EQ(incremental.g, full.g);
  EXPECT_EQ(incremental.b, full.b);
  EXPECT_EQ(incremental.has_voxel_ref, full.has_voxel_ref);
  if (incremental.has_voxel_ref && full.has_voxel_ref) {
    EXPECT_TRUE(incremental.voxel_ref.block_index.isApprox(
        full.voxel_ref.block_index));
    EXPECT_TRUE(incremental.voxel_ref.voxel_index.isApprox(
        full.voxel_ref.voxel_index));
  }
}

void expectEquivalentSnapshots(const SurfaceSnapshot& incremental,
                               const SurfaceSnapshot& full) {
  ASSERT_EQ(incremental.blockCount(), full.blockCount());
  ASSERT_EQ(incremental.pointCount(), full.pointCount());
  const std::vector<SurfaceBlockPtr> incremental_blocks =
      incremental.blockView();
  const std::vector<SurfaceBlockPtr> full_blocks = full.blockView();
  ASSERT_EQ(incremental_blocks.size(), full_blocks.size());
  for (std::size_t block_index = 0;
       block_index < incremental_blocks.size(); ++block_index) {
    const SurfaceBlock& incremental_block = *incremental_blocks[block_index];
    const SurfaceBlock& full_block = *full_blocks[block_index];
    EXPECT_EQ(incremental_block.index(), full_block.index());
    EXPECT_TRUE(incremental_block.aabb().min.isApprox(full_block.aabb().min,
                                                      1e-6f));
    EXPECT_TRUE(incremental_block.aabb().max.isApprox(full_block.aabb().max,
                                                      1e-6f));
    ASSERT_EQ(incremental_block.pointCount(), full_block.pointCount());
    for (std::size_t point_index = 0;
         point_index < incremental_block.pointCount(); ++point_index) {
      expectEquivalentPoints(incremental_block.points()[point_index],
                             full_block.points()[point_index]);
    }
  }
}

class NvbloxBackendConformance : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
      GTEST_SKIP() << "A CUDA-capable GPU is required: "
                   << cudaGetErrorString(count_status);
    }
    if (device_count == 0) {
      GTEST_SKIP() << "A visible CUDA-capable GPU is required";
    }
    cudaDeviceProp device{};
    ASSERT_EQ(cudaGetDeviceProperties(&device, 0), cudaSuccess);
    RecordProperty("cuda_device", std::string(device.name));
    backend_ = createNvbloxMapBackend(testConfig());
    ASSERT_TRUE(backend_);
  }

  std::unique_ptr<MapBackend> backend_;
};

TEST_F(NvbloxBackendConformance,
       IncrementalDeltaMatchesSameRevisionForcedFullRebuild) {
  const RunId epoch{0x525458ULL, 0x4e56424c4f58ULL};

  const MapIntegrationResult first_integration =
      backend_->integrateFrame(frame(1, 1'000'000, planeDepth(1.5f)));
  ASSERT_TRUE(first_integration.success) << first_integration.error;
  ASSERT_TRUE(first_integration.map_changed);
  ASSERT_TRUE(first_integration.updated_blocks_complete);
  ASSERT_FALSE(first_integration.updated_blocks.empty());

  const SurfaceRefreshResult first_refresh =
      backend_->refreshSurface(first_integration, true);
  ASSERT_TRUE(first_refresh.success) << first_refresh.error;
  ASSERT_TRUE(first_refresh.full_rebuild);
  ASSERT_FALSE(first_refresh.blocks.empty());
  ASSERT_TRUE(first_refresh.removed_blocks.empty());
  const SurfaceSnapshotPtr first_snapshot =
      buildFullSnapshot(epoch, 1, first_integration, first_refresh);
  ASSERT_TRUE(first_snapshot);
  ASSERT_FALSE(first_snapshot->empty());

  DepthBuffer second_depth = planeDepth(2.0f);
  const MapIntegrationResult second_integration =
      backend_->integrateFrame(frame(2, 2'000'000, std::move(second_depth)));
  ASSERT_TRUE(second_integration.success) << second_integration.error;
  ASSERT_TRUE(second_integration.map_changed);
  ASSERT_TRUE(second_integration.updated_blocks_complete);
  ASSERT_EQ(second_integration.map_revision,
            first_integration.map_revision + 1U);

  const SurfaceRefreshResult incremental_refresh =
      backend_->refreshSurface(second_integration, false);
  ASSERT_TRUE(incremental_refresh.success) << incremental_refresh.error;
  ASSERT_FALSE(incremental_refresh.full_rebuild);
  ASSERT_FALSE(incremental_refresh.blocks.empty());
  ASSERT_FALSE(incremental_refresh.removed_blocks.empty());

  std::set<BlockIndex> changed_indices;
  for (const SurfaceBlockPtr& block : incremental_refresh.blocks) {
    ASSERT_TRUE(block);
    ASSERT_TRUE(changed_indices.insert(block->index()).second);
  }
  for (const BlockIndex& removed : incremental_refresh.removed_blocks) {
    EXPECT_TRUE(first_snapshot->block(removed))
        << "incremental removal was never in the previous surface cache";
    EXPECT_EQ(changed_indices.count(removed), 0U)
        << "a block cannot be both changed and removed";
  }

  const SurfaceSnapshotBuildResult incremental_build =
      SurfaceSnapshotBuilder::buildFrom(
          first_snapshot, mapStamp(epoch, second_integration),
          surfaceStamp(epoch, 2, second_integration),
          incremental_refresh.blocks, incremental_refresh.removed_blocks);
  ASSERT_TRUE(incremental_build.snapshot);

  const SurfaceRefreshResult forced_full_refresh =
      backend_->refreshSurface(second_integration, true);
  ASSERT_TRUE(forced_full_refresh.success) << forced_full_refresh.error;
  ASSERT_TRUE(forced_full_refresh.full_rebuild);
  ASSERT_FALSE(forced_full_refresh.blocks.empty());
  const SurfaceSnapshotPtr forced_full_snapshot =
      buildFullSnapshot(epoch, 2, second_integration, forced_full_refresh);
  ASSERT_TRUE(forced_full_snapshot);

  expectEquivalentSnapshots(*incremental_build.snapshot,
                            *forced_full_snapshot);
}

TEST_F(NvbloxBackendConformance,
       InvalidAndRobotMaskedDepthDoNotInventSurfaceRemovals) {
  const MapIntegrationResult seed_integration =
      backend_->integrateFrame(frame(1, 1'000'000, planeDepth(1.5f)));
  ASSERT_TRUE(seed_integration.success) << seed_integration.error;
  const SurfaceRefreshResult seed_refresh =
      backend_->refreshSurface(seed_integration, true);
  ASSERT_TRUE(seed_refresh.success) << seed_refresh.error;
  ASSERT_FALSE(seed_refresh.blocks.empty());
  const std::uint64_t seed_points =
      seed_refresh.diagnostics.cached_surface_points;

  DepthBuffer invalid_depth = planeDepth(0.0f);
  for (std::size_t index = 1; index < invalid_depth.depth_m.size(); index += 2) {
    invalid_depth.depth_m[index] =
        std::numeric_limits<float>::quiet_NaN();
  }
  const MapIntegrationResult invalid_integration =
      backend_->integrateFrame(frame(2, 2'000'000, std::move(invalid_depth)));
  ASSERT_TRUE(invalid_integration.success) << invalid_integration.error;
  ASSERT_TRUE(invalid_integration.updated_blocks_complete);
  const SurfaceRefreshResult invalid_refresh =
      backend_->refreshSurface(invalid_integration, false);
  ASSERT_TRUE(invalid_refresh.success) << invalid_refresh.error;
  ASSERT_FALSE(invalid_refresh.full_rebuild);
  EXPECT_TRUE(invalid_refresh.removed_blocks.empty());
  EXPECT_EQ(invalid_refresh.diagnostics.cached_surface_points, seed_points);

  const MapIntegrationResult masked_integration = backend_->integrateFrame(
      frame(3, 3'000'000, planeDepth(1.5f), 255));
  ASSERT_TRUE(masked_integration.success) << masked_integration.error;
  ASSERT_TRUE(masked_integration.updated_blocks_complete);
  const SurfaceRefreshResult masked_refresh =
      backend_->refreshSurface(masked_integration, false);
  ASSERT_TRUE(masked_refresh.success) << masked_refresh.error;
  ASSERT_FALSE(masked_refresh.full_rebuild);
  EXPECT_TRUE(masked_refresh.removed_blocks.empty());
  EXPECT_EQ(masked_refresh.diagnostics.cached_surface_points, seed_points);
}

TEST_F(NvbloxBackendConformance,
       CheckpointLoadFencesPathWorldFrameAndMapConfiguration) {
  TemporaryCheckpointDirectory directory;
  PipelineConfig save_config = testConfig();
  save_config.save_map = true;
  save_config.map_save_path =
      (directory.path() / "checkpoint.nvblox").string();
  std::unique_ptr<MapBackend> saving = createNvbloxMapBackend(save_config);
  ASSERT_TRUE(saving);
  const MapIntegrationResult integrated =
      saving->integrateFrame(frame(1, 1'000'000, planeDepth(1.5f)));
  ASSERT_TRUE(integrated.success) << integrated.error;
  const RunId epoch{0x1234U, 0x5678U};
  const MapCheckpointOperationResult saved =
      saving->saveCheckpoint(mapStamp(epoch, integrated));
  ASSERT_TRUE(saved.succeeded()) << saved.error;
  EXPECT_EQ(saved.manifest.world_frame, save_config.world_frame);
  EXPECT_FALSE(saved.manifest.config_fingerprint.empty());

  PipelineConfig load_config = testConfig();
  // Durable coordinated recovery is authoritative even when the fresh-run
  // convenience flag remains false and no seed path is configured.
  load_config.load_map = false;
  load_config.map_load_path.clear();
  std::unique_ptr<MapBackend> loading = createNvbloxMapBackend(load_config);
  ASSERT_TRUE(loading);
  EXPECT_TRUE(loading->loadCheckpoint(&saved.manifest).succeeded());
  MapIntegrationResult loaded_map;
  loaded_map.success = true;
  loaded_map.map_revision = 0;
  loaded_map.updated_blocks_complete = true;
  const SurfaceRefreshResult loaded_surface =
      loading->refreshSurface(loaded_map, /*force_full_rebuild=*/true);
  ASSERT_TRUE(loaded_surface.success) << loaded_surface.error;
  EXPECT_GT(loaded_surface.diagnostics.tsdf_blocks, 0U);
  EXPECT_GT(loaded_surface.diagnostics.cached_surface_points, 0U);

  PipelineConfig wrong_frame_config = load_config;
  wrong_frame_config.world_frame = "different_world";
  std::unique_ptr<MapBackend> wrong_frame =
      createNvbloxMapBackend(wrong_frame_config);
  const MapCheckpointOperationResult frame_result =
      wrong_frame->loadCheckpoint(&saved.manifest);
  EXPECT_EQ(frame_result.disposition, MapCheckpointDisposition::kFailed);
  EXPECT_NE(frame_result.error.find("world frame"), std::string::npos);

  PipelineConfig wrong_map_config = load_config;
  wrong_map_config.voxel_size_m *= 2.0f;
  std::unique_ptr<MapBackend> wrong_map =
      createNvbloxMapBackend(wrong_map_config);
  const MapCheckpointOperationResult config_result =
      wrong_map->loadCheckpoint(&saved.manifest);
  EXPECT_EQ(config_result.disposition, MapCheckpointDisposition::kFailed);
  EXPECT_NE(config_result.error.find("fingerprint"), std::string::npos);

  PipelineConfig wrong_path_config = load_config;
  wrong_path_config.map_load_path =
      (directory.path() / "other.nvblox").string();
  std::unique_ptr<MapBackend> wrong_path =
      createNvbloxMapBackend(wrong_path_config);
  const MapCheckpointOperationResult path_result =
      wrong_path->loadCheckpoint(&saved.manifest);
  EXPECT_EQ(path_result.disposition, MapCheckpointDisposition::kFailed);
  EXPECT_NE(path_result.error.find("conflicts"), std::string::npos);
}

}  // namespace
}  // namespace roomie
