#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "roomie/artifacts/snapshot_bank.hpp"
#include "roomie/pipeline/ephemeral_run_workspace.hpp"
#include "roomie/scene/scene_reducer.hpp"
#include "roomie/scene/scene_store.hpp"

namespace roomie {
namespace {

namespace fs = std::filesystem;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    std::string pattern =
        (fs::temp_directory_path() / "roomie_ephemeral_test_XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* const created = ::mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    fs::remove_all(path_, ignored);
  }

  const fs::path& path() const { return path_; }

 private:
  fs::path path_;
};

std::string readFile(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to read " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

ImageBuffer makeImage(std::uint8_t seed) {
  ImageBuffer image;
  image.width = 12;
  image.height = 8;
  image.channels = 3;
  image.encoding = "rgb8";
  image.data.resize(static_cast<std::size_t>(
      image.width * image.height * image.channels));
  for (std::size_t index = 0; index < image.data.size(); ++index) {
    image.data[index] =
        static_cast<std::uint8_t>((seed + index * 13U) % 251U);
  }
  return image;
}

std::vector<SceneSnapshot> makeSnapshots() {
  ReducerCore reducer;
  LoadSceneCommand load;
  ObjectNode node;
  node.object_id = 1;
  node.semantic_id = 101;
  node.label = "chair";
  node.center_world = Eigen::Vector3f(1.0f, 2.0f, 0.5f);
  node.size_m = Eigen::Vector3f(0.5f, 0.6f, 0.9f);
  node.active = true;
  node.publishable = true;
  load.graph.objects.push_back(node);
  load.graph.next_object_id = 2;
  SceneApplyResult result = reducer.apply(SceneCommand{load});
  if (!result.committedRevision()) {
    throw std::runtime_error(result.reason);
  }
  std::vector<SceneSnapshot> snapshots{result.snapshot};
  for (const std::string label : {"office chair", "blue office chair"}) {
    ApplyHumanAnnotationCommand annotation;
    annotation.object_id = 1;
    annotation.patch.label = label;
    result = reducer.apply(SceneCommand{annotation});
    if (!result.committedRevision()) {
      throw std::runtime_error(result.reason);
    }
    snapshots.push_back(result.snapshot);
  }
  return snapshots;
}

PipelineConfig seedBaseline(const fs::path& root,
                            const std::vector<SceneSnapshot>& snapshots,
                            AssetId* asset_id,
                            fs::path* asset_path) {
  const fs::path database_path = root / "baseline" / "roomie_scene.sqlite3";
  fs::create_directories(database_path.parent_path());
  SceneStore store(database_path.string());
  if (!store.open()) {
    throw std::runtime_error("failed to open baseline SceneStore");
  }
  if (!store.enqueueCommit(snapshots[0]) ||
      !store.enqueueCommit(snapshots[1]) || !store.gracefulFlush()) {
    throw std::runtime_error("failed to seed baseline SceneStore");
  }
  const fs::path checkpoint = root / "baseline" / "map.nvblox";
  {
    std::ofstream stream(checkpoint, std::ios::binary);
    stream << "map";
  }
  MapCheckpointManifest manifest;
  manifest.backend = "nvblox";
  manifest.checkpoint_path = checkpoint.string();
  manifest.world_frame = "map";
  manifest.config_fingerprint = "fnv1a64:test";
  manifest.map_epoch = RunId{11, 22};
  manifest.map_revision = 7;
  manifest.integrated_through_ns = 100;
  manifest.aligned_scene_revision = snapshots[1].revision();
  manifest.file_size_bytes = 3;
  manifest.content_hash = "test-map-hash";
  manifest.created_at_unix_ms = 1;
  if (!store.publishMapCheckpoint(manifest) || !store.closeGracefully()) {
    throw std::runtime_error("failed to publish baseline map manifest");
  }

  const fs::path asset_root = root / "baseline" / "asset_store";
  AssetStoreConfig asset_config;
  asset_config.root = asset_root;
  asset_config.grace_period_ns = 1000;
  AssetStore assets(asset_config);
  if (!assets.healthy()) {
    throw std::runtime_error(assets.initializationError());
  }
  const AssetWriteResult written = assets.materializeFrame(makeImage(7), 10);
  if (!written.success) {
    throw std::runtime_error(written.error);
  }
  std::string error;
  if (!assets.retain(written.asset.id, &error)) {
    throw std::runtime_error(error);
  }
  *asset_id = written.asset.id;
  *asset_path = written.asset.path;

  PipelineConfig config;
  config.scene_store_enabled = true;
  config.scene_store_path = database_path.string();
  config.ephemeral_run = true;
  config.ephemeral_workspace_root = (root / "workspaces").string();
  config.map_load_mode = "coordinated";
  config.save_map = true;
  config.map_save_path = (root / "baseline" / "nvblox").string();
  config.online_snapshot_enabled = true;
  config.asset_store_root = asset_root.string();
  config.scene_graph_save_path =
      (root / "baseline" / "instances").string();
  config.snapshot_staging_dir =
      (root / "baseline" / "snapshots").string();
  config.file_logging_root_dir =
      (root / "diagnostic_logs").string();
  return config;
}

TEST(EphemeralRunWorkspace, RedirectsWritesAndPreservesBaseline) {
  TemporaryDirectory temporary;
  const std::vector<SceneSnapshot> snapshots = makeSnapshots();
  AssetId baseline_asset_id;
  fs::path baseline_asset_path;
  PipelineConfig config = seedBaseline(
      temporary.path(), snapshots, &baseline_asset_id, &baseline_asset_path);

  const fs::path baseline_database = config.scene_store_path;
  const fs::path baseline_manifest =
      fs::path(config.asset_store_root) / "manifest.json";
  const std::string database_before = readFile(baseline_database);
  const std::string manifest_before = readFile(baseline_manifest);
  const std::string asset_before = readFile(baseline_asset_path);
  const std::string logging_root = config.file_logging_root_dir;
  fs::path workspace_path;

  {
    PreparedPipelineRuntime prepared =
        preparePipelineRuntime(std::move(config));
    ASSERT_TRUE(prepared.ephemeral_workspace);
    workspace_path = prepared.ephemeral_workspace->path();
    EXPECT_TRUE(fs::exists(workspace_path));
    EXPECT_FALSE(prepared.config.save_map);
    EXPECT_EQ(prepared.config.file_logging_root_dir, logging_root);
    EXPECT_NE(prepared.config.scene_store_path,
              baseline_database.string());
    EXPECT_NE(prepared.config.asset_store_root,
              baseline_manifest.parent_path().string());
    EXPECT_EQ(fs::path(prepared.config.scene_store_path).parent_path(),
              workspace_path / "state");
    EXPECT_EQ(prepared.config.map_save_path,
              (workspace_path / "nvblox").string());
    EXPECT_EQ(prepared.config.scene_graph_save_path,
              (workspace_path / "instances").string());
    EXPECT_EQ(prepared.config.snapshot_staging_dir,
              (workspace_path / "snapshots").string());

    SceneStore ephemeral_store(prepared.config.scene_store_path);
    ASSERT_TRUE(ephemeral_store.open());
    const SceneRestoreResult restored = ephemeral_store.restoreLatest();
    ASSERT_TRUE(restored.status) << restored.status.error;
    ASSERT_TRUE(restored.found);
    EXPECT_EQ(restored.snapshot.revision(), snapshots[1].revision());
    ASSERT_TRUE(ephemeral_store.enqueueCommit(snapshots[2]));
    ASSERT_TRUE(ephemeral_store.closeGracefully());

    AssetStoreConfig ephemeral_asset_config;
    ephemeral_asset_config.root = prepared.config.asset_store_root;
    ephemeral_asset_config.grace_period_ns = 1000;
    AssetStore ephemeral_assets(ephemeral_asset_config);
    ASSERT_TRUE(ephemeral_assets.healthy())
        << ephemeral_assets.initializationError();
    const std::optional<AssetRecord> staged =
        ephemeral_assets.record(baseline_asset_id);
    ASSERT_TRUE(staged);
    EXPECT_TRUE(fs::is_symlink(staged->path));
    std::string error;
    ASSERT_TRUE(ephemeral_assets.release(baseline_asset_id, 20, &error))
        << error;
    const AssetWriteResult new_asset =
        ephemeral_assets.materializeFrame(makeImage(19), 30);
    ASSERT_TRUE(new_asset.success) << new_asset.error;
    EXPECT_TRUE(new_asset.created);

    EXPECT_EQ(readFile(baseline_database), database_before);
    EXPECT_EQ(readFile(baseline_manifest), manifest_before);
    EXPECT_EQ(readFile(baseline_asset_path), asset_before);
  }

  EXPECT_FALSE(fs::exists(workspace_path));
  SceneStore baseline_store(baseline_database.string());
  ASSERT_TRUE(baseline_store.open());
  const SceneRestoreResult baseline = baseline_store.restoreLatest();
  ASSERT_TRUE(baseline.status) << baseline.status.error;
  ASSERT_TRUE(baseline.found);
  EXPECT_EQ(baseline.snapshot.revision(), snapshots[1].revision());
  ASSERT_TRUE(baseline_store.closeGracefully());
}

TEST(EphemeralRunWorkspace, RejectsUnsupportedOrIncompleteSources) {
  PipelineConfig config;
  config.ephemeral_run = true;
  config.scene_store_enabled = false;
  EXPECT_THROW(preparePipelineRuntime(config), std::invalid_argument);

  config.scene_store_enabled = true;
  config.map_load_mode = "seed";
  EXPECT_THROW(preparePipelineRuntime(config), std::invalid_argument);

  config.map_load_mode = "coordinated";
  config.scene_store_path = "/definitely/missing/roomie_scene.sqlite3";
  EXPECT_THROW(preparePipelineRuntime(config), std::invalid_argument);

  TemporaryDirectory temporary;
  const fs::path database_path = temporary.path() / "no_manifest.sqlite3";
  SceneStore store(database_path.string());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.closeGracefully());
  config.scene_store_path = database_path.string();
  config.ephemeral_workspace_root =
      (temporary.path() / "workspaces").string();
  config.online_snapshot_enabled = false;
  EXPECT_THROW(preparePipelineRuntime(config), std::invalid_argument);
}

TEST(EphemeralRunWorkspace, OrdinaryRunLeavesConfigurationUntouched) {
  PipelineConfig config;
  config.ephemeral_run = false;
  config.scene_store_path = "/tmp/original.sqlite3";
  config.asset_store_root = "/tmp/original_assets";
  config.save_map = true;
  PreparedPipelineRuntime prepared = preparePipelineRuntime(config);
  EXPECT_FALSE(prepared.ephemeral_workspace);
  EXPECT_EQ(prepared.config.scene_store_path, config.scene_store_path);
  EXPECT_EQ(prepared.config.asset_store_root, config.asset_store_root);
  EXPECT_TRUE(prepared.config.save_map);
}

}  // namespace
}  // namespace roomie
