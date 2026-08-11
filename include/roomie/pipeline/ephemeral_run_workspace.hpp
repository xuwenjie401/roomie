#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "roomie/pipeline/pipeline_config.hpp"

namespace roomie {

struct PreparedPipelineRuntime;

// Owns the disposable persistence overlay used by one coordinated-load run.
// The workspace is deliberately declared before all persistence consumers in
// RoomiePipeline so its destructor runs only after those consumers are gone.
class EphemeralRunWorkspace {
 public:
  EphemeralRunWorkspace(std::filesystem::path root,
                        std::filesystem::path path,
                        std::filesystem::path baseline_scene_store_path);
  ~EphemeralRunWorkspace();

  EphemeralRunWorkspace(const EphemeralRunWorkspace&) = delete;
  EphemeralRunWorkspace& operator=(const EphemeralRunWorkspace&) = delete;

  const std::filesystem::path& path() const { return path_; }
  const std::filesystem::path& baselineSceneStorePath() const {
    return baseline_scene_store_path_;
  }

 private:
  friend struct PreparedPipelineRuntime;
  friend PreparedPipelineRuntime preparePipelineRuntime(PipelineConfig config);

  std::filesystem::path root_;
  std::filesystem::path path_;
  std::filesystem::path baseline_scene_store_path_;
};

struct PreparedPipelineRuntime {
  PipelineConfig config;
  std::shared_ptr<EphemeralRunWorkspace> ephemeral_workspace;
};

// Returns the input unchanged for ordinary runs. For an ephemeral run it
// creates a consistent writable overlay and rewrites every persistence output
// before any pipeline actor is constructed. Failures are fail-closed: the
// original configuration is never returned as a fallback.
PreparedPipelineRuntime preparePipelineRuntime(PipelineConfig config);

}  // namespace roomie
