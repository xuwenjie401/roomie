#include "roomie/artifacts/python_dam_worker.hpp"

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "roomie/artifacts/semantic_index.hpp"

extern char** environ;

namespace roomie {
namespace {

using Json = nlohmann::json;
using SteadyClock = std::chrono::steady_clock;
using Milliseconds = std::chrono::milliseconds;

constexpr const char* kProtocol = "roomie.dam-worker.v1";
constexpr Milliseconds kPollSlice{20};

std::string ioError(const std::string& operation, int error_number) {
  if (error_number == EPIPE) {
    return operation + ": EPIPE (broken pipe)";
  }
  return operation + ": " + std::strerror(error_number) + " (errno=" +
         std::to_string(error_number) + ")";
}

void closeFd(int fd) {
  if (fd >= 0) {
    ::close(fd);
  }
}

bool setNonBlocking(int fd, std::string* error) {
  const int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    if (error) {
      *error = ioError("configure DAM IPC pipe", errno);
    }
    return false;
  }
  return true;
}

bool processExited(pid_t pid) {
  int status = 0;
  while (true) {
    const pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == pid || (result < 0 && errno == ECHILD)) {
      return true;
    }
    if (result == 0) {
      return false;
    }
    if (errno != EINTR) {
      return false;
    }
  }
}

void terminateProcessBounded(pid_t pid, Milliseconds timeout) {
  if (pid <= 0 || processExited(pid)) {
    return;
  }
  const auto start = SteadyClock::now();
  const auto deadline = start + timeout;
  const auto term_deadline = start + timeout / 2;
  (void)::kill(pid, SIGTERM);
  while (SteadyClock::now() < term_deadline) {
    if (processExited(pid)) {
      return;
    }
    std::this_thread::sleep_for(Milliseconds(2));
  }
  (void)::kill(pid, SIGKILL);
  while (SteadyClock::now() < deadline) {
    if (processExited(pid)) {
      return;
    }
    std::this_thread::sleep_for(Milliseconds(2));
  }
  // SIGKILL has been issued. Avoid an unbounded wait even if the scheduler has
  // not run the child far enough to make it reapable yet.
  (void)processExited(pid);
}

class ScopedSigpipeBlock {
 public:
  ScopedSigpipeBlock() {
    sigset_t pending{};
    ::sigpending(&pending);
    was_pending_ = ::sigismember(&pending, SIGPIPE) == 1;
    ::sigemptyset(&set_);
    ::sigaddset(&set_, SIGPIPE);
    error_ = ::pthread_sigmask(SIG_BLOCK, &set_, &previous_);
  }

  ~ScopedSigpipeBlock() {
    if (error_ != 0) {
      return;
    }
    if (!was_pending_) {
      sigset_t pending{};
      ::sigpending(&pending);
      if (::sigismember(&pending, SIGPIPE) == 1) {
        timespec zero{};
        while (::sigtimedwait(&set_, nullptr, &zero) < 0 && errno == EINTR) {
        }
      }
    }
    (void)::pthread_sigmask(SIG_SETMASK, &previous_, nullptr);
  }

  bool ok() const { return error_ == 0; }
  int error() const { return error_; }

 private:
  sigset_t set_{};
  sigset_t previous_{};
  bool was_pending_ = false;
  int error_ = 0;
};

class AssetPins {
 public:
  explicit AssetPins(std::shared_ptr<AssetStore> store)
      : store_(std::move(store)) {}

  ~AssetPins() {
    for (const std::string& id : ids_) {
      std::string ignored;
      (void)store_->release(id, 0, &ignored);
    }
  }

  bool retain(const std::string& id, std::string* error) {
    if (!store_->retain(id, error)) {
      return false;
    }
    ids_.push_back(id);
    return true;
  }

 private:
  std::shared_ptr<AssetStore> store_;
  std::vector<std::string> ids_;
};

DamWorkerResponse failure(bool retryable, std::string error) {
  DamWorkerResponse response;
  response.success = false;
  response.retryable = retryable;
  response.error = std::move(error);
  return response;
}

bool validBbox(const Json& value) {
  if (!value.is_array() || value.size() != 4U) {
    return false;
  }
  try {
    for (const Json& coordinate : value) {
      if (!coordinate.is_number() ||
          !std::isfinite(coordinate.get<double>())) {
        return false;
      }
    }
    const double x0 = value.at(0).get<double>();
    const double y0 = value.at(1).get<double>();
    const double x1 = value.at(2).get<double>();
    const double y1 = value.at(3).get<double>();
    return std::abs(x1 - x0) > std::numeric_limits<double>::epsilon() &&
           std::abs(y1 - y0) > std::numeric_limits<double>::epsilon();
  } catch (...) {
    return false;
  }
}

void copyIfPresent(const Json& source, const char* key, Json* target) {
  const auto found = source.find(key);
  if (found != source.end()) {
    (*target)[key] = *found;
  }
}

}  // namespace

std::string canonicalDamExecutionPromptHash(
    const PythonDamWorkerConfig& config) {
  auto canonical_double = [](double value) {
    if (value == 0.0) {
      value = 0.0;
    }
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::hexfloat << value;
    return stream.str();
  };
  auto append = [](std::ostringstream* stream,
                   const char* name,
                   const std::string& value) {
    *stream << std::char_traits<char>::length(name) << ':' << name << '='
            << value.size() << ':' << value << '\n';
  };
  if (config.conversation_mode.empty() || config.prompt_mode.empty() ||
      config.query.empty() || config.max_new_tokens <= 0 ||
      !std::isfinite(config.temperature) || !std::isfinite(config.top_p) ||
      !std::isfinite(config.bbox_pad_px)) {
    throw std::invalid_argument(
        "cannot hash invalid DAM execution parameters");
  }
  std::ostringstream canonical;
  canonical.imbue(std::locale::classic());
  canonical << "roomie.dam-execution-provenance.v1\n";
  append(&canonical, "conversation_mode", config.conversation_mode);
  append(&canonical, "prompt_mode", config.prompt_mode);
  append(&canonical, "query", config.query);
  append(&canonical, "max_new_tokens",
         std::to_string(config.max_new_tokens));
  append(&canonical, "temperature",
         canonical_double(config.temperature));
  append(&canonical, "top_p", canonical_double(config.top_p));
  append(&canonical, "bbox_pad_px", canonical_double(config.bbox_pad_px));
  return stableSha256Hex(canonical.str());
}

struct PythonDamWorker::StartupResult {
  bool success = false;
  bool retryable = true;
  std::string error;
};

struct PythonDamWorker::PreparedRequest {
  std::uint64_t request_id = 0;
  std::string body;
};

PythonDamWorker::PythonDamWorker(PythonDamWorkerConfig config,
                                 std::shared_ptr<AssetStore> asset_store)
    : config_(std::move(config)), asset_store_(std::move(asset_store)) {
  if (!asset_store_) {
    throw std::invalid_argument("PythonDamWorker requires an AssetStore");
  }
  if (config_.python_executable.empty() || config_.worker_script.empty() ||
      config_.model_path.empty() || config_.model_id.empty() ||
      config_.conversation_mode.empty() ||
      config_.prompt_mode.empty() || config_.query.empty()) {
    throw std::invalid_argument("PythonDamWorker configuration is incomplete");
  }
  if (config_.max_new_tokens <= 0 || !std::isfinite(config_.temperature) ||
      !std::isfinite(config_.top_p) || !std::isfinite(config_.bbox_pad_px) ||
      config_.temperature < 0.0 || config_.top_p <= 0.0 ||
      config_.top_p > 1.0 || config_.bbox_pad_px < 0.0) {
    throw std::invalid_argument("PythonDamWorker model options are invalid");
  }
  const std::string computed_prompt_hash =
      canonicalDamExecutionPromptHash(config_);
  if (!config_.prompt_hash.empty() &&
      config_.prompt_hash != computed_prompt_hash) {
    throw std::invalid_argument(
        "PythonDamWorker prompt_hash does not match canonical execution parameters");
  }
  config_.prompt_hash = computed_prompt_hash;
  if (config_.startup_timeout <= Milliseconds::zero() ||
      config_.request_timeout <= Milliseconds::zero() ||
      config_.shutdown_timeout < Milliseconds(20) ||
      config_.max_message_bytes < 1024U ||
      config_.max_message_bytes >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::invalid_argument("PythonDamWorker IPC limits are invalid");
  }
}

PythonDamWorker::~PythonDamWorker() { shutdown(); }

DamWorkerResponse PythonDamWorker::describe(
    const DamTaskRequest& request,
    const DamLeaseHeartbeat& heartbeat) {
  // ArtifactRuntimeActor runs a watchdog heartbeat independently of this
  // synchronous invocation. Calling the callback here would couple lease
  // health to Python's response cadence.
  (void)heartbeat;
  std::lock_guard<std::mutex> transaction_lock(transaction_mutex_);
  if (shutdown_requested_.load(std::memory_order_acquire)) {
    return failure(true, "Python DAM worker is shut down");
  }

  ScopedSigpipeBlock sigpipe;
  if (!sigpipe.ok()) {
    return failure(true,
                   "cannot block SIGPIPE for DAM IPC: " +
                       std::string(std::strerror(sigpipe.error())));
  }
  if (!asset_store_->healthy()) {
    return failure(true,
                   "DAM AssetStore is unavailable: " +
                       asset_store_->initializationError());
  }
  if (request.key.model_id != config_.model_id) {
    return failure(false,
                   "DAM task model_id does not match the resident model: " +
                       request.key.model_id);
  }
  if (request.key.prompt_hash != config_.prompt_hash) {
    return failure(false,
                   "DAM task prompt_hash does not match the resident prompt: " +
                       request.key.prompt_hash);
  }

  Json input;
  try {
    input = Json::parse(request.input_payload);
  } catch (const std::exception& error) {
    return failure(false,
                   std::string("DAM input_payload is not valid JSON: ") +
                       error.what());
  }
  try {
    if (!input.is_object() ||
        input.at("payload_version").get<std::string>() !=
            "roomie.dam-input.v1" ||
        input.at("object_id").get<SceneObjectId>() != request.key.object_id ||
        input.at("snapshot_set_hash").get<std::string>() !=
            request.key.snapshot_set_hash ||
        !input.at("snapshots").is_array() ||
        input.at("snapshots").empty()) {
      return failure(false,
                     "DAM input_payload violates roomie.dam-input.v1");
    }
  } catch (const std::exception& error) {
    return failure(false,
                   std::string("DAM input_payload is incomplete: ") +
                       error.what());
  }

  const Json& snapshots = input.at("snapshots");
  std::vector<std::string> asset_ids;
  std::set<std::string> unique_asset_ids;
  for (std::size_t index = 0; index < snapshots.size(); ++index) {
    const Json& snapshot = snapshots.at(index);
    try {
      if (!snapshot.is_object() ||
          !snapshot.at("source_frame_asset_id").is_string() ||
          snapshot.at("source_frame_asset_id").get<std::string>().empty() ||
          !validBbox(snapshot.at("bbox_xyxy"))) {
        return failure(false,
                       "DAM snapshot " + std::to_string(index) +
                           " has no durable asset id or valid bbox");
      }
      const std::string id =
          snapshot.at("source_frame_asset_id").get<std::string>();
      if (unique_asset_ids.insert(id).second) {
        asset_ids.push_back(id);
      }
    } catch (const std::exception& error) {
      return failure(false,
                     "DAM snapshot " + std::to_string(index) +
                         " is malformed: " + error.what());
    }
  }

  // An id missing from the durable manifest cannot be recovered by retrying
  // this immutable task. A manifest entry whose file is temporarily missing
  // or corrupt remains retryable because rematerialization can heal it.
  for (const std::string& id : asset_ids) {
    if (!asset_store_->record(id)) {
      return failure(false, "DAM task references unknown asset: " + id);
    }
  }
  AssetPins pins(asset_store_);
  for (const std::string& id : asset_ids) {
    std::string error;
    if (!pins.retain(id, &error)) {
      return failure(true,
                     "could not pin DAM asset " + id + ": " + error);
    }
  }

  Json resolved_snapshots = Json::array();
  for (std::size_t index = 0; index < snapshots.size(); ++index) {
    const Json& snapshot = snapshots.at(index);
    const std::string id =
        snapshot.at("source_frame_asset_id").get<std::string>();
    const std::optional<AssetRecord> record = asset_store_->record(id);
    if (!record) {
      return failure(true,
                     "pinned DAM asset disappeared from manifest: " + id);
    }
    std::string validation_error;
    if (!asset_store_->validate(id, &validation_error)) {
      return failure(true,
                     "pinned DAM asset is not readable: " + id + ": " +
                         validation_error);
    }
    std::error_code path_error;
    const std::filesystem::path absolute_path =
        std::filesystem::absolute(record->path, path_error).lexically_normal();
    if (path_error) {
      return failure(true,
                     "cannot resolve DAM asset path " + id + ": " +
                         path_error.message());
    }

    Json resolved;
    copyIfPresent(snapshot, "image_index", &resolved);
    copyIfPresent(snapshot, "evidence_hash", &resolved);
    copyIfPresent(snapshot, "bbox_xyxy", &resolved);
    copyIfPresent(snapshot, "crop_xywh", &resolved);
    copyIfPresent(snapshot, "crop_output_scale", &resolved);
    copyIfPresent(snapshot, "mask_source", &resolved);
    copyIfPresent(snapshot, "mask_ref", &resolved);
    copyIfPresent(snapshot, "quality", &resolved);
    copyIfPresent(snapshot, "quality_components", &resolved);
    copyIfPresent(snapshot, "viewpoint", &resolved);
    copyIfPresent(snapshot, "time_ns", &resolved);
    copyIfPresent(snapshot, "camera_id", &resolved);
    copyIfPresent(snapshot, "provenance", &resolved);
    resolved["source_frame_asset_id"] = id;
    resolved["durable_asset"] =
        Json{{"asset_id", id},
             {"uri", "asset://" + id},
             {"path", absolute_path.string()},
             {"access_mode", "read_only"},
             {"immutable", true},
             {"width", record->width},
             {"height", record->height},
             {"channels", record->channels},
             {"encoding", record->source_encoding},
             {"encoded_bytes", record->encoded_bytes}};
    resolved_snapshots.push_back(std::move(resolved));
  }

  Json sanitized_input;
  copyIfPresent(input, "payload_version", &sanitized_input);
  copyIfPresent(input, "owning_scene_revision", &sanitized_input);
  copyIfPresent(input, "object_id", &sanitized_input);
  copyIfPresent(input, "label", &sanitized_input);
  copyIfPresent(input, "dependency", &sanitized_input);
  copyIfPresent(input, "snapshot_set_hash", &sanitized_input);
  sanitized_input["snapshots"] = std::move(resolved_snapshots);

  PreparedRequest prepared;
  prepared.request_id = next_request_id_++;
  Json message =
      {{"protocol", kProtocol},
       {"type", "describe"},
       {"request_id", prepared.request_id},
       {"task_id", canonicalDamTaskKey(request.key)},
       {"task",
        Json{{"object_id", request.key.object_id},
             {"identity_revision", request.key.identity_revision},
             {"appearance_revision", request.key.appearance_revision},
             {"snapshot_set_hash", request.key.snapshot_set_hash},
             {"model_id", request.key.model_id},
             {"prompt_hash", request.key.prompt_hash},
             {"output_schema_version",
              request.key.output_schema_version}}},
       {"input", std::move(sanitized_input)}};
  prepared.body = message.dump();
  if (prepared.body.size() > config_.max_message_bytes) {
    return failure(false, "resolved DAM request exceeds the IPC size limit");
  }
  return transact(prepared);
}

PythonDamWorker::StartupResult PythonDamWorker::ensureWorkerProcess() {
  StartupResult result;
  if (shutdown_requested_.load(std::memory_order_acquire)) {
    result.error = "DAM worker start cancelled during shutdown";
    return result;
  }

  pid_t spawned_pid = -1;
  {
    std::lock_guard<std::mutex> process_lock(process_mutex_);
    const int existing_pid = worker_pid_.load(std::memory_order_acquire);
    if (existing_pid > 0) {
      if (!processExited(static_cast<pid_t>(existing_pid))) {
        if (worker_ready_.load(std::memory_order_acquire)) {
          result.success = true;
          result.error.clear();
          return result;
        }
        result.error = "DAM worker exists but did not finish startup";
        return result;
      }
      closeFd(worker_stdin_fd_.exchange(-1));
      closeFd(worker_stdout_fd_.exchange(-1));
      worker_pid_.store(-1);
      worker_ready_.store(false);
    }

    if (::access(config_.worker_script.c_str(), R_OK) != 0) {
      result.retryable = false;
      result.error = "cannot read Python DAM worker script: " +
                     config_.worker_script.string();
      return result;
    }
    if (!config_.dam_source.empty() &&
        !std::filesystem::is_directory(config_.dam_source)) {
      result.retryable = false;
      result.error = "configured official DAM source directory does not exist: " +
                     config_.dam_source.string();
      return result;
    }

    int to_child[2] = {-1, -1};
    int from_child[2] = {-1, -1};
    if (::pipe(to_child) != 0) {
      result.error = ioError("create DAM request pipe", errno);
      return result;
    }
    if (::pipe(from_child) != 0) {
      const int pipe_error = errno;
      closeFd(to_child[0]);
      closeFd(to_child[1]);
      result.error = ioError("create DAM response pipe", pipe_error);
      return result;
    }

    std::vector<std::string> args = {
        config_.python_executable,
        config_.worker_script.string(),
        "--model-path",
        config_.model_path,
        "--model-id",
        config_.model_id,
        "--conv-mode",
        config_.conversation_mode,
        "--prompt-mode",
        config_.prompt_mode,
        "--query",
        config_.query,
        "--max-new-tokens",
        std::to_string(config_.max_new_tokens),
        "--temperature",
        std::to_string(config_.temperature),
        "--top-p",
        std::to_string(config_.top_p),
        "--bbox-pad-px",
        std::to_string(config_.bbox_pad_px),
        "--max-message-bytes",
        std::to_string(config_.max_message_bytes),
    };
    if (!config_.dam_source.empty()) {
      args.push_back("--dam-src");
      args.push_back(config_.dam_source.string());
    }
    std::vector<char*> argv;
    argv.reserve(args.size() + 1U);
    for (std::string& argument : args) {
      argv.push_back(argument.data());
    }
    argv.push_back(nullptr);

    posix_spawn_file_actions_t actions;
    const int actions_result = ::posix_spawn_file_actions_init(&actions);
    if (actions_result != 0) {
      closeFd(to_child[0]);
      closeFd(to_child[1]);
      closeFd(from_child[0]);
      closeFd(from_child[1]);
      result.error = "posix_spawn_file_actions_init: " +
                     std::string(std::strerror(actions_result));
      return result;
    }
    (void)::posix_spawn_file_actions_adddup2(&actions, to_child[0], STDIN_FILENO);
    (void)::posix_spawn_file_actions_adddup2(&actions, from_child[1], STDOUT_FILENO);
    (void)::posix_spawn_file_actions_addclose(&actions, to_child[0]);
    (void)::posix_spawn_file_actions_addclose(&actions, to_child[1]);
    (void)::posix_spawn_file_actions_addclose(&actions, from_child[0]);
    (void)::posix_spawn_file_actions_addclose(&actions, from_child[1]);

    pid_t pid = 0;
    const int spawn_result = ::posix_spawnp(
        &pid, argv.front(), &actions, nullptr, argv.data(), environ);
    (void)::posix_spawn_file_actions_destroy(&actions);
    if (spawn_result != 0) {
      closeFd(to_child[0]);
      closeFd(to_child[1]);
      closeFd(from_child[0]);
      closeFd(from_child[1]);
      result.retryable = spawn_result != ENOENT && spawn_result != EACCES;
      result.error = "posix_spawnp DAM worker: " +
                     std::string(std::strerror(spawn_result));
      return result;
    }

    closeFd(to_child[0]);
    closeFd(from_child[1]);
    std::string nonblocking_error;
    if (!setNonBlocking(to_child[1], &nonblocking_error) ||
        !setNonBlocking(from_child[0], &nonblocking_error)) {
      closeFd(to_child[1]);
      closeFd(from_child[0]);
      terminateProcessBounded(pid, config_.shutdown_timeout);
      result.error = std::move(nonblocking_error);
      return result;
    }
    if (shutdown_requested_.load(std::memory_order_acquire)) {
      closeFd(to_child[1]);
      closeFd(from_child[0]);
      terminateProcessBounded(pid, config_.shutdown_timeout);
      result.error = "DAM worker start cancelled during shutdown";
      return result;
    }
    worker_pid_.store(static_cast<int>(pid), std::memory_order_release);
    worker_stdin_fd_.store(to_child[1], std::memory_order_release);
    worker_stdout_fd_.store(from_child[0], std::memory_order_release);
    worker_ready_.store(false, std::memory_order_release);
    spawned_pid = pid;
  }

  std::string startup_body;
  std::string startup_error;
  const auto startup_deadline = SteadyClock::now() + config_.startup_timeout;
  if (!readJsonMessage(&startup_body, startup_deadline, &startup_error)) {
    stopWorkerProcess();
    result.error = "DAM worker startup failed: " + startup_error;
    return result;
  }

  try {
    const Json startup = Json::parse(startup_body);
    if (!startup.is_object() ||
        startup.at("protocol").get<std::string>() != kProtocol) {
      throw std::runtime_error("invalid startup protocol envelope");
    }
    const std::string type = startup.at("type").get<std::string>();
    if (type == "startup_error") {
      result.retryable = startup.at("retryable").get<bool>();
      result.error = startup.at("error").get<std::string>();
      if (result.error.empty()) {
        throw std::runtime_error("empty startup error");
      }
      stopWorkerProcess();
      return result;
    }
    if (type != "ready" ||
        startup.at("model_id").get<std::string>() != config_.model_id) {
      throw std::runtime_error("unexpected DAM startup response");
    }
  } catch (const std::exception& error) {
    stopWorkerProcess();
    result.error = std::string("malformed DAM startup response: ") +
                   error.what();
    return result;
  }

  if (shutdown_requested_.load(std::memory_order_acquire) ||
      worker_pid_.load(std::memory_order_acquire) != spawned_pid) {
    stopWorkerProcess();
    result.error = "DAM worker startup was interrupted";
    return result;
  }
  worker_ready_.store(true, std::memory_order_release);
  result.success = true;
  result.error.clear();
  return result;
}

DamWorkerResponse PythonDamWorker::transact(const PreparedRequest& request) {
  const StartupResult startup = ensureWorkerProcess();
  if (!startup.success) {
    return failure(startup.retryable,
                   startup.error.empty() ? "DAM worker failed to start"
                                         : startup.error);
  }
  if (shutdown_requested_.load(std::memory_order_acquire)) {
    return failure(true, "DAM request cancelled during shutdown");
  }

  const auto deadline = SteadyClock::now() + config_.request_timeout;
  std::string error;
  if (!writeJsonMessage(request.body, deadline, &error)) {
    stopWorkerProcess();
    return failure(true, "DAM request IPC failed: " + error);
  }
  std::string response_body;
  if (!readJsonMessage(&response_body, deadline, &error)) {
    stopWorkerProcess();
    return failure(true, "DAM response IPC failed: " + error);
  }

  try {
    const Json response = Json::parse(response_body);
    if (!response.is_object() ||
        response.at("protocol").get<std::string>() != kProtocol ||
        response.at("type").get<std::string>() != "result" ||
        response.at("request_id").get<std::uint64_t>() != request.request_id) {
      throw std::runtime_error("response envelope or request_id mismatch");
    }
    DamWorkerResponse result;
    result.success = response.at("success").get<bool>();
    if (result.success) {
      result.retryable = false;
      result.raw_output = response.at("raw_output").get<std::string>();
      if (result.raw_output.empty()) {
        throw std::runtime_error("successful response has empty raw_output");
      }
    } else {
      result.retryable = response.at("retryable").get<bool>();
      result.error = response.at("error").get<std::string>();
      if (result.error.empty()) {
        throw std::runtime_error("failed response has empty error");
      }
    }
    return result;
  } catch (const std::exception& parse_error) {
    stopWorkerProcess();
    return failure(true,
                   std::string("malformed DAM worker response: ") +
                       parse_error.what());
  }
}

bool PythonDamWorker::writeJsonMessage(
    const std::string& body,
    SteadyClock::time_point deadline,
    std::string* error) {
  if (body.size() > config_.max_message_bytes ||
      body.size() > std::numeric_limits<std::uint32_t>::max()) {
    *error = "DAM IPC message exceeds the configured size limit";
    return false;
  }
  const std::uint32_t size = static_cast<std::uint32_t>(body.size());
  const std::uint8_t header[4] = {
      static_cast<std::uint8_t>(size & 0xffU),
      static_cast<std::uint8_t>((size >> 8U) & 0xffU),
      static_cast<std::uint8_t>((size >> 16U) & 0xffU),
      static_cast<std::uint8_t>((size >> 24U) & 0xffU),
  };
  const int fd = worker_stdin_fd_.load(std::memory_order_acquire);
  if (fd < 0) {
    *error = "DAM request pipe is closed";
    return false;
  }
  return writeExact(fd, header, sizeof(header), deadline, error) &&
         writeExact(fd, body.data(), body.size(), deadline, error);
}

bool PythonDamWorker::readJsonMessage(
    std::string* body,
    SteadyClock::time_point deadline,
    std::string* error) {
  const int fd = worker_stdout_fd_.load(std::memory_order_acquire);
  if (fd < 0) {
    *error = "DAM response pipe is closed";
    return false;
  }
  std::uint8_t header[4] = {};
  if (!readExact(fd, header, sizeof(header), deadline, error)) {
    return false;
  }
  const std::uint32_t size =
      static_cast<std::uint32_t>(header[0]) |
      (static_cast<std::uint32_t>(header[1]) << 8U) |
      (static_cast<std::uint32_t>(header[2]) << 16U) |
      (static_cast<std::uint32_t>(header[3]) << 24U);
  if (size > config_.max_message_bytes) {
    *error = "DAM response exceeds the configured size limit";
    return false;
  }
  body->assign(size, '\0');
  return readExact(fd, body->data(), body->size(), deadline, error);
}

bool PythonDamWorker::writeExact(
    int fd,
    const void* data,
    std::size_t size,
    SteadyClock::time_point deadline,
    std::string* error) const {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    if (shutdown_requested_.load(std::memory_order_acquire)) {
      *error = "DAM IPC write cancelled during shutdown";
      return false;
    }
    const auto now = SteadyClock::now();
    if (now >= deadline) {
      *error = "DAM IPC write timed out";
      return false;
    }
    const auto remaining =
        std::chrono::duration_cast<Milliseconds>(deadline - now);
    pollfd descriptor{};
    descriptor.fd = fd;
    descriptor.events = POLLOUT;
    const int timeout_ms = static_cast<int>(
        std::max<Milliseconds>(Milliseconds(1),
                               std::min(kPollSlice, remaining))
            .count());
    const int polled = ::poll(&descriptor, 1, timeout_ms);
    if (polled == 0) {
      continue;
    }
    if (polled < 0) {
      if (errno == EINTR) {
        continue;
      }
      *error = ioError("poll DAM request pipe", errno);
      return false;
    }
    if ((descriptor.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0 &&
        (descriptor.revents & POLLOUT) == 0) {
      *error = "DAM request pipe closed before write (EPIPE)";
      return false;
    }
    const ssize_t written = ::write(fd, bytes + offset, size - offset);
    if (written > 0) {
      offset += static_cast<std::size_t>(written);
      continue;
    }
    if (written < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    *error = ioError("write DAM request", written < 0 ? errno : EPIPE);
    return false;
  }
  error->clear();
  return true;
}

bool PythonDamWorker::readExact(
    int fd,
    void* data,
    std::size_t size,
    SteadyClock::time_point deadline,
    std::string* error) const {
  auto* bytes = static_cast<std::uint8_t*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    if (shutdown_requested_.load(std::memory_order_acquire)) {
      *error = "DAM IPC read cancelled during shutdown";
      return false;
    }
    const auto now = SteadyClock::now();
    if (now >= deadline) {
      *error = "DAM IPC read timed out";
      return false;
    }
    const auto remaining =
        std::chrono::duration_cast<Milliseconds>(deadline - now);
    pollfd descriptor{};
    descriptor.fd = fd;
    descriptor.events = POLLIN;
    const int timeout_ms = static_cast<int>(
        std::max<Milliseconds>(Milliseconds(1),
                               std::min(kPollSlice, remaining))
            .count());
    const int polled = ::poll(&descriptor, 1, timeout_ms);
    if (polled == 0) {
      continue;
    }
    if (polled < 0) {
      if (errno == EINTR) {
        continue;
      }
      *error = ioError("poll DAM response pipe", errno);
      return false;
    }
    if ((descriptor.revents & POLLIN) == 0 &&
        (descriptor.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
      *error = "DAM response pipe closed before a complete frame";
      return false;
    }
    const ssize_t received = ::read(fd, bytes + offset, size - offset);
    if (received > 0) {
      offset += static_cast<std::size_t>(received);
      continue;
    }
    if (received < 0 &&
        (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    *error = received == 0
                 ? "DAM response pipe reached EOF before a complete frame"
                 : ioError("read DAM response", errno);
    return false;
  }
  error->clear();
  return true;
}

void PythonDamWorker::stopWorkerProcess() {
  std::lock_guard<std::mutex> process_lock(process_mutex_);
  worker_ready_.store(false, std::memory_order_release);
  const int pid = worker_pid_.exchange(-1, std::memory_order_acq_rel);
  const int stdin_fd = worker_stdin_fd_.exchange(-1, std::memory_order_acq_rel);
  const int stdout_fd =
      worker_stdout_fd_.exchange(-1, std::memory_order_acq_rel);
  if (pid > 0) {
    // Terminating before closing makes a concurrent poll wake on peer closure
    // without exposing it to an fd-number reuse race.
    terminateProcessBounded(static_cast<pid_t>(pid), config_.shutdown_timeout);
  }
  closeFd(stdin_fd);
  closeFd(stdout_fd);
}

void PythonDamWorker::shutdown() {
  shutdown_requested_.store(true, std::memory_order_release);
  stopWorkerProcess();
}

}  // namespace roomie
