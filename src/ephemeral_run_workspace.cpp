#include "roomie/pipeline/ephemeral_run_workspace.hpp"

#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sqlite3.h>

namespace roomie {
namespace {

namespace fs = std::filesystem;

constexpr const char* kWorkspaceMarkerName = ".roomie_ephemeral_session";
constexpr const char* kWorkspaceMarkerContents =
    "roomie.ephemeral-session.v1\n";

std::string sqliteError(sqlite3* database, const std::string& prefix) {
  return prefix + ": " +
         (database != nullptr ? sqlite3_errmsg(database)
                              : std::string("no SQLite handle"));
}

struct SqliteHandle {
  sqlite3* value = nullptr;

  ~SqliteHandle() {
    if (value != nullptr) {
      sqlite3_close_v2(value);
    }
  }
};

fs::path absoluteNormalized(const fs::path& path) {
  std::error_code error;
  const fs::path absolute = fs::absolute(path, error);
  return (error ? path : absolute).lexically_normal();
}

void copySceneStoreConsistently(const fs::path& source,
                                const fs::path& destination) {
  SqliteHandle source_database;
  const int source_open = sqlite3_open_v2(
      source.string().c_str(), &source_database.value,
      SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, nullptr);
  if (source_open != SQLITE_OK) {
    throw std::runtime_error(
        sqliteError(source_database.value,
                    "failed to open baseline SceneStore read-only"));
  }
  sqlite3_busy_timeout(source_database.value, 100);

  SqliteHandle destination_database;
  const int destination_open = sqlite3_open_v2(
      destination.string().c_str(), &destination_database.value,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
      nullptr);
  if (destination_open != SQLITE_OK) {
    throw std::runtime_error(
        sqliteError(destination_database.value,
                    "failed to create ephemeral SceneStore"));
  }
  sqlite3_busy_timeout(destination_database.value, 100);

  sqlite3_backup* backup = sqlite3_backup_init(
      destination_database.value, "main", source_database.value, "main");
  if (backup == nullptr) {
    throw std::runtime_error(
        sqliteError(destination_database.value,
                    "failed to initialize SceneStore backup"));
  }

  int result = SQLITE_OK;
  int busy_attempts = 0;
  for (;;) {
    result = sqlite3_backup_step(backup, 2048);
    if (result == SQLITE_DONE) {
      break;
    }
    if (result == SQLITE_OK) {
      busy_attempts = 0;
      continue;
    }
    if ((result == SQLITE_BUSY || result == SQLITE_LOCKED) &&
        busy_attempts++ < 100) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    const std::string error = sqliteError(
        destination_database.value, "ephemeral SceneStore backup failed");
    sqlite3_backup_finish(backup);
    throw std::runtime_error(error);
  }

  const int finish_result = sqlite3_backup_finish(backup);
  if (finish_result != SQLITE_OK) {
    throw std::runtime_error(
        sqliteError(destination_database.value,
                    "failed to finish ephemeral SceneStore backup"));
  }
}

void requireCoordinatedMapManifest(const fs::path& scene_store) {
  SqliteHandle database;
  const int open_result = sqlite3_open_v2(
      scene_store.string().c_str(), &database.value,
      SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, nullptr);
  if (open_result != SQLITE_OK) {
    throw std::runtime_error(
        sqliteError(database.value,
                    "failed to inspect ephemeral SceneStore manifest"));
  }
  sqlite3_stmt* statement = nullptr;
  const int prepare_result = sqlite3_prepare_v2(
      database.value,
      "SELECT 1 FROM map_checkpoint_manifest ORDER BY checkpoint_id DESC "
      "LIMIT 1;",
      -1, &statement, nullptr);
  if (prepare_result != SQLITE_OK) {
    throw std::runtime_error(
        sqliteError(database.value,
                    "ephemeral baseline has no readable map manifest table"));
  }
  const int step_result = sqlite3_step(statement);
  sqlite3_finalize(statement);
  if (step_result != SQLITE_ROW) {
    if (step_result == SQLITE_DONE) {
      throw std::invalid_argument(
          "persistence.ephemeral_run requires a coordinated map manifest");
    }
    throw std::runtime_error(
        sqliteError(database.value,
                    "failed to read coordinated map manifest"));
  }
}

void stageAssetStore(const fs::path& source_root,
                     const fs::path& destination_root) {
  std::error_code error;
  fs::create_directories(destination_root / "assets", error);
  if (error) {
    throw std::runtime_error("failed to create ephemeral AssetStore: " +
                             error.message());
  }

  if (source_root.empty() || !fs::exists(source_root, error)) {
    if (error) {
      throw std::runtime_error("failed to inspect baseline AssetStore: " +
                               error.message());
    }
    return;
  }
  if (!fs::is_directory(source_root, error) || error) {
    throw std::runtime_error(
        "baseline AssetStore root is not a directory: " +
        source_root.string());
  }

  const fs::path source_manifest = source_root / "manifest.json";
  if (fs::exists(source_manifest, error)) {
    if (error || !fs::is_regular_file(source_manifest, error)) {
      throw std::runtime_error(
          "baseline AssetStore manifest is not a regular file");
    }
    fs::copy_file(source_manifest, destination_root / "manifest.json",
                  fs::copy_options::none, error);
    if (error) {
      throw std::runtime_error(
          "failed to copy baseline AssetStore manifest: " +
          error.message());
    }
  } else if (error) {
    throw std::runtime_error("failed to inspect baseline AssetStore manifest: " +
                             error.message());
  }

  const fs::path source_assets = source_root / "assets";
  if (!fs::exists(source_assets, error)) {
    if (error) {
      throw std::runtime_error("failed to inspect baseline asset directory: " +
                               error.message());
    }
    return;
  }
  if (!fs::is_directory(source_assets, error) || error) {
    throw std::runtime_error(
        "baseline AssetStore assets path is not a directory");
  }

  for (const fs::directory_entry& entry :
       fs::directory_iterator(source_assets)) {
    if (!entry.is_regular_file(error)) {
      if (error) {
        throw std::runtime_error("failed to inspect baseline asset: " +
                                 error.message());
      }
      continue;
    }
    if (entry.path().extension() != ".png") {
      continue;
    }
    const fs::path target = absoluteNormalized(entry.path());
    const fs::path link = destination_root / "assets" /
                          entry.path().filename();
    fs::create_symlink(target, link, error);
    if (error) {
      throw std::runtime_error("failed to link baseline asset " +
                               target.string() + ": " + error.message());
    }
  }
}

std::shared_ptr<EphemeralRunWorkspace> createWorkspace(
    const PipelineConfig& config) {
  if (config.ephemeral_workspace_root.empty()) {
    throw std::invalid_argument(
        "persistence.ephemeral_workspace_root must not be empty");
  }
  const fs::path root = absoluteNormalized(config.ephemeral_workspace_root);
  std::error_code error;
  fs::create_directories(root, error);
  if (error) {
    throw std::runtime_error("failed to create ephemeral workspace root: " +
                             error.message());
  }

  std::string pattern = (root / "roomie_ephemeral_XXXXXX").string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  char* const created = ::mkdtemp(writable.data());
  if (created == nullptr) {
    throw std::runtime_error(
        "failed to create unique ephemeral workspace under " +
        root.string() + ": " + std::strerror(errno));
  }
  const fs::path workspace = absoluteNormalized(created);
  const fs::path marker = workspace / kWorkspaceMarkerName;
  {
    std::ofstream stream(marker, std::ios::out | std::ios::trunc);
    stream << kWorkspaceMarkerContents;
    stream.flush();
    if (!stream) {
      fs::remove_all(workspace, error);
      throw std::runtime_error(
          "failed to create ephemeral workspace safety marker");
    }
  }
  return std::shared_ptr<EphemeralRunWorkspace>(
      new EphemeralRunWorkspace(
          root, workspace, absoluteNormalized(config.scene_store_path)));
}

}  // namespace

EphemeralRunWorkspace::EphemeralRunWorkspace(
    fs::path root,
    fs::path path,
    fs::path baseline_scene_store_path)
    : root_(std::move(root)),
      path_(std::move(path)),
      baseline_scene_store_path_(std::move(baseline_scene_store_path)) {}

EphemeralRunWorkspace::~EphemeralRunWorkspace() {
  if (path_.empty() || path_.parent_path() != root_ ||
      path_.filename().string().rfind("roomie_ephemeral_", 0) != 0) {
    std::cerr << "Roomie refused unsafe ephemeral workspace cleanup path="
              << path_ << '\n';
    return;
  }
  const fs::path marker = path_ / kWorkspaceMarkerName;
  std::ifstream stream(marker);
  std::string contents((std::istreambuf_iterator<char>(stream)),
                       std::istreambuf_iterator<char>());
  if (!stream && !stream.eof()) {
    std::cerr << "Roomie could not read ephemeral cleanup marker path="
              << marker << '\n';
    return;
  }
  if (contents != kWorkspaceMarkerContents) {
    std::cerr << "Roomie refused ephemeral cleanup because the marker is "
                 "missing or invalid path="
              << marker << '\n';
    return;
  }
  std::error_code error;
  fs::remove_all(path_, error);
  if (error) {
    std::cerr << "Roomie failed to remove ephemeral workspace path=" << path_
              << " error=" << error.message() << '\n';
  }
}

PreparedPipelineRuntime preparePipelineRuntime(PipelineConfig config) {
  PreparedPipelineRuntime prepared;
  if (!config.ephemeral_run) {
    prepared.config = std::move(config);
    return prepared;
  }
  if (!config.scene_store_enabled) {
    throw std::invalid_argument(
        "persistence.ephemeral_run requires scene_store_enabled=true");
  }
  if (config.map_load_mode != "coordinated") {
    throw std::invalid_argument(
        "persistence.ephemeral_run is supported only with "
        "tsdf.map_load_mode=coordinated");
  }
  if (config.scene_store_path.empty()) {
    throw std::invalid_argument(
        "persistence.ephemeral_run requires a baseline scene_store_path");
  }
  const fs::path baseline_scene_store =
      absoluteNormalized(config.scene_store_path);
  std::error_code error;
  if (!fs::is_regular_file(baseline_scene_store, error) || error) {
    throw std::invalid_argument(
        "ephemeral baseline SceneStore is not a regular file: " +
        baseline_scene_store.string());
  }

  std::shared_ptr<EphemeralRunWorkspace> workspace =
      createWorkspace(config);
  const fs::path ephemeral_scene_store =
      workspace->path() / "state" / "roomie_scene.sqlite3";
  fs::create_directories(ephemeral_scene_store.parent_path(), error);
  if (error) {
    throw std::runtime_error(
        "failed to create ephemeral SceneStore directory: " +
        error.message());
  }
  copySceneStoreConsistently(baseline_scene_store,
                             ephemeral_scene_store);
  requireCoordinatedMapManifest(ephemeral_scene_store);

  if (config.online_snapshot_enabled && config.asset_store_root.empty()) {
    throw std::invalid_argument(
        "persistence.ephemeral_run with snapshots enabled requires "
        "artifacts.asset_store_root");
  }
  const fs::path ephemeral_asset_store =
      workspace->path() / "asset_store";
  const fs::path baseline_asset_store =
      config.asset_store_root.empty()
          ? fs::path{}
          : absoluteNormalized(config.asset_store_root);
  stageAssetStore(baseline_asset_store, ephemeral_asset_store);

  config.scene_store_path = ephemeral_scene_store.string();
  config.asset_store_root = ephemeral_asset_store.string();
  config.map_save_path = (workspace->path() / "nvblox").string();
  config.scene_graph_save_path =
      (workspace->path() / "instances").string();
  config.snapshot_staging_dir =
      (workspace->path() / "snapshots").string();
  // Runtime mapping remains online. Only the shutdown checkpoint is disabled
  // because it has no consumer after this disposable session is torn down.
  config.save_map = false;

  prepared.config = std::move(config);
  prepared.ephemeral_workspace = std::move(workspace);
  return prepared;
}

}  // namespace roomie
