#pragma once

#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "roomie/pipeline/pipeline_config.hpp"

namespace roomie {

class RunLogger {
 public:
  explicit RunLogger(const PipelineConfig& config);
  ~RunLogger();

  RunLogger(const RunLogger&) = delete;
  RunLogger& operator=(const RunLogger&) = delete;

  static void setGlobal(std::shared_ptr<RunLogger> logger);
  static std::shared_ptr<RunLogger> global();
  static void logGlobal(const std::string& module, const std::string& message);

  bool enabled() const;
  const std::string& runDirectory() const;
  void log(const std::string& module, const std::string& message);

 private:
  std::ofstream& streamForModuleLocked(const std::string& module);

  bool enabled_ = false;
  std::string run_directory_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::ofstream> streams_;
};

}  // namespace roomie
