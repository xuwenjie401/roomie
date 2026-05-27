#include "roomie/utils/run_logger.hpp"

#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <utility>

namespace roomie {
namespace {

std::mutex g_logger_mutex;
std::weak_ptr<RunLogger> g_logger;

std::string timestampForPath() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&now_time, &tm);

  std::ostringstream stream;
  stream << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return stream.str();
}

std::string timestampForLine() {
  const auto now = std::chrono::system_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&now_time, &tm);

  std::ostringstream stream;
  stream << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "."
         << std::setw(3) << std::setfill('0') << ms.count();
  return stream.str();
}

std::string sanitizeModuleName(const std::string& module) {
  std::string out;
  out.reserve(module.size());
  for (const char c : module) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_' || c == '-') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out.empty() ? "roomie" : out;
}

}  // namespace

RunLogger::RunLogger(const PipelineConfig& config) : enabled_(config.file_logging_enabled) {
  if (!enabled_) {
    return;
  }

  namespace fs = std::filesystem;
  const fs::path root(config.file_logging_root_dir);
  std::error_code error;
  fs::create_directories(root, error);
  if (error) {
    enabled_ = false;
    return;
  }

  const std::string run_name =
      "run_" + timestampForPath() + "_pid" + std::to_string(static_cast<long>(::getpid()));
  const fs::path run_dir = root / run_name;
  fs::create_directories(run_dir, error);
  if (error) {
    enabled_ = false;
    return;
  }

  run_directory_ = run_dir.string();

  const fs::path latest = root / "latest";
  fs::remove(latest, error);
  error.clear();
  fs::create_directory_symlink(run_dir, latest, error);

  log("pipeline", "run_dir=" + run_directory_);
}

RunLogger::~RunLogger() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [unused, stream] : streams_) {
    (void)unused;
    stream.flush();
  }
}

void RunLogger::setGlobal(std::shared_ptr<RunLogger> logger) {
  std::lock_guard<std::mutex> lock(g_logger_mutex);
  g_logger = std::move(logger);
}

std::shared_ptr<RunLogger> RunLogger::global() {
  std::lock_guard<std::mutex> lock(g_logger_mutex);
  return g_logger.lock();
}

void RunLogger::logGlobal(const std::string& module, const std::string& message) {
  std::shared_ptr<RunLogger> logger = global();
  if (logger) {
    logger->log(module, message);
  }
}

bool RunLogger::enabled() const { return enabled_; }

const std::string& RunLogger::runDirectory() const { return run_directory_; }

void RunLogger::log(const std::string& module, const std::string& message) {
  if (!enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  std::ofstream& stream = streamForModuleLocked(module);
  if (!stream) {
    return;
  }
  stream << timestampForLine() << " " << message << "\n";
  stream.flush();
}

std::ofstream& RunLogger::streamForModuleLocked(const std::string& module) {
  const std::string key = sanitizeModuleName(module);
  auto it = streams_.find(key);
  if (it != streams_.end()) {
    return it->second;
  }

  const std::filesystem::path path =
      std::filesystem::path(run_directory_) / (key + ".log");
  auto [inserted_it, unused] = streams_.emplace(key, std::ofstream(path, std::ios::app));
  (void)unused;
  return inserted_it->second;
}

}  // namespace roomie
