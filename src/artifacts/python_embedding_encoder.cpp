#include "roomie/artifacts/python_embedding_encoder.hpp"

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

extern char** environ;

namespace roomie {
namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

constexpr char kProtocol[] = "roomie.embedding.v1";
constexpr std::uint64_t kReadyMessageId = 0;

void closeFd(int* fd) {
  if (*fd >= 0) {
    ::close(*fd);
    *fd = -1;
  }
}

std::string errnoMessage(const std::string& operation, int error_number) {
  std::string result = operation + ": " + std::strerror(error_number);
  if (error_number == EPIPE) {
    result += " (EPIPE)";
  }
  return result;
}

bool containsTimeout(const std::string& error) {
  return error.find("timed out") != std::string::npos ||
         error.find("timeout") != std::string::npos;
}

bool setNonBlocking(int fd, std::string* error) {
  const int current = ::fcntl(fd, F_GETFL, 0);
  if (current < 0 || ::fcntl(fd, F_SETFL, current | O_NONBLOCK) < 0) {
    *error = errnoMessage("fcntl(O_NONBLOCK)", errno);
    return false;
  }
  return true;
}

bool setCloseOnExec(int fd, std::string* error) {
  const int current = ::fcntl(fd, F_GETFD, 0);
  if (current < 0 || ::fcntl(fd, F_SETFD, current | FD_CLOEXEC) < 0) {
    *error = errnoMessage("fcntl(FD_CLOEXEC)", errno);
    return false;
  }
  return true;
}

bool waitForProcessExit(pid_t pid, std::chrono::milliseconds timeout) {
  const auto deadline = Clock::now() + timeout;
  while (true) {
    int status = 0;
    const pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == pid || (result < 0 && errno == ECHILD)) {
      return true;
    }
    if (result < 0 && errno != EINTR) {
      return false;
    }
    if (Clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

void terminateProcessBounded(pid_t pid,
                             std::chrono::milliseconds shutdown_timeout) {
  if (pid <= 0) {
    return;
  }
  if (::kill(pid, SIGTERM) != 0 && errno != ESRCH) {
    // Continue to wait/kill. A stale pid is handled by waitpid below.
  }
  const auto terminate_grace = std::max(
      std::chrono::milliseconds(1), shutdown_timeout / 2);
  if (waitForProcessExit(pid, terminate_grace)) {
    return;
  }
  if (::kill(pid, SIGKILL) != 0 && errno != ESRCH) {
    // The final bounded wait still attempts to reap the process.
  }
  (void)waitForProcessExit(pid, shutdown_timeout - terminate_grace);
}

bool hasOnlyKeys(const Json& value,
                 const std::unordered_set<std::string>& keys) {
  if (!value.is_object()) {
    return false;
  }
  for (auto it = value.begin(); it != value.end(); ++it) {
    if (keys.find(it.key()) == keys.end()) {
      return false;
    }
  }
  return true;
}

class ScopedSigpipeBlock {
 public:
  ScopedSigpipeBlock() {
    ::sigemptyset(&set_);
    ::sigaddset(&set_, SIGPIPE);
    sigset_t pending{};
    if (::sigpending(&pending) == 0) {
      was_pending_ = ::sigismember(&pending, SIGPIPE) == 1;
    }
    error_ = ::pthread_sigmask(SIG_BLOCK, &set_, &previous_);
    active_ = error_ == 0;
  }

  ~ScopedSigpipeBlock() {
    if (!active_) {
      return;
    }
    if (!was_pending_) {
      sigset_t pending{};
      if (::sigpending(&pending) == 0 &&
          ::sigismember(&pending, SIGPIPE) == 1) {
        timespec no_wait{};
        while (::sigtimedwait(&set_, nullptr, &no_wait) < 0 && errno == EINTR) {
        }
      }
    }
    (void)::pthread_sigmask(SIG_SETMASK, &previous_, nullptr);
  }

  bool active() const { return active_; }
  std::string error() const {
    return error_ == 0 ? std::string()
                       : std::string("pthread_sigmask(SIGPIPE): ") +
                             std::strerror(error_);
  }

 private:
  sigset_t set_{};
  sigset_t previous_{};
  int error_ = 0;
  bool active_ = false;
  bool was_pending_ = false;
};

}  // namespace

PythonEmbeddingEncoderConfig PythonEmbeddingEncoder::validateConfig(
    PythonEmbeddingEncoderConfig config) {
  if (config.python_executable.empty() || config.worker_script.empty() ||
      config.model_path.empty() || config.expected_dimension == 0 ||
      config.maximum_batch_size == 0 || config.maximum_document_bytes == 0 ||
      config.maximum_message_bytes == 0 ||
      config.maximum_message_bytes >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      config.maximum_document_bytes > config.maximum_message_bytes ||
      config.startup_timeout <= std::chrono::milliseconds::zero() ||
      config.request_timeout <= std::chrono::milliseconds::zero() ||
      config.shutdown_timeout <= std::chrono::milliseconds::zero() ||
      config.poll_period <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument(
        "Python embedding encoder configuration is invalid");
  }
  if (config.model_id.empty()) {
    config.model_id = config.model_path;
  }
  return config;
}

PythonEmbeddingEncoder::PythonEmbeddingEncoder(
    PythonEmbeddingEncoderConfig config)
    : config_(validateConfig(std::move(config))),
      model_id_(config_.model_id),
      dimension_(config_.expected_dimension) {}

PythonEmbeddingEncoder::~PythonEmbeddingEncoder() { shutdown(); }

std::string PythonEmbeddingEncoder::modelId() const { return model_id_; }

std::size_t PythonEmbeddingEncoder::dimension() const { return dimension_; }

void PythonEmbeddingEncoder::prewarm() {
  if (shutdown_requested_.load(std::memory_order_acquire)) {
    throw std::runtime_error("Python embedding encoder is shut down");
  }
  const auto deadline = Clock::now() + config_.startup_timeout;
  std::unique_lock<std::timed_mutex> operation_lock(operation_mutex_,
                                                     std::defer_lock);
  if (!operation_lock.try_lock_until(deadline)) {
    const std::string error =
        "embedding startup timed out waiting for the serialized encoder";
    ++timeouts_;
    ++request_failures_;
    recordError(error);
    throw std::runtime_error(error);
  }

  ScopedSigpipeBlock sigpipe;
  if (!sigpipe.active()) {
    ++request_failures_;
    recordError(sigpipe.error());
    throw std::runtime_error(sigpipe.error());
  }

  std::string error;
  if (!ensureWorker(deadline, &error)) {
    ++request_failures_;
    if (containsTimeout(error)) {
      ++timeouts_;
    }
    recordError(error);
    throw std::runtime_error(error);
  }
}

std::vector<std::vector<float>> PythonEmbeddingEncoder::encodeBatch(
    const std::vector<std::string>& documents) {
  if (documents.empty()) {
    return {};
  }
  if (documents.size() > config_.maximum_batch_size) {
    throw std::invalid_argument("embedding batch exceeds maximum_batch_size");
  }
  for (const std::string& document : documents) {
    if (document.size() > config_.maximum_document_bytes) {
      throw std::invalid_argument(
          "embedding document exceeds maximum_document_bytes");
    }
  }
  if (shutdown_requested_.load(std::memory_order_acquire)) {
    throw std::runtime_error("Python embedding encoder is shut down");
  }

  ++requests_;
  const auto deadline = Clock::now() + config_.request_timeout;
  std::unique_lock<std::timed_mutex> operation_lock(operation_mutex_,
                                                     std::defer_lock);
  if (!operation_lock.try_lock_until(deadline)) {
    const std::string error =
        "embedding request timed out waiting for the serialized encoder";
    ++request_failures_;
    ++timeouts_;
    recordError(error);
    throw std::runtime_error(error);
  }

  ScopedSigpipeBlock sigpipe;
  if (!sigpipe.active()) {
    ++request_failures_;
    recordError(sigpipe.error());
    throw std::runtime_error(sigpipe.error());
  }

  std::string final_error;
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (shutdown_requested_.load(std::memory_order_acquire)) {
      final_error = "embedding request cancelled because encoder is shut down";
      break;
    }

    Reply reply = transact(documents, deadline);
    if (reply.kind == ReplyKind::kSuccess) {
      encoded_documents_.fetch_add(documents.size(), std::memory_order_relaxed);
      return std::move(reply.embeddings);
    }
    final_error = std::move(reply.error);
    if (reply.kind == ReplyKind::kWorkerError) {
      break;
    }

    stopWorker(true);
    if (attempt == 1 || Clock::now() >= deadline ||
        shutdown_requested_.load(std::memory_order_acquire)) {
      break;
    }
  }

  if (final_error.empty()) {
    final_error = "embedding request failed without an error response";
  }
  ++request_failures_;
  if (containsTimeout(final_error)) {
    ++timeouts_;
  }
  recordError(final_error);
  throw std::runtime_error(final_error);
}

std::vector<float> PythonEmbeddingEncoder::encodeOne(
    const std::string& document) {
  std::vector<std::vector<float>> batch = encodeBatch({document});
  if (batch.size() != 1) {
    throw std::runtime_error("embedding worker returned an invalid single result");
  }
  return std::move(batch.front());
}

void PythonEmbeddingEncoder::shutdown() {
  std::lock_guard<std::mutex> shutdown_lock(shutdown_mutex_);
  if (shutdown_requested_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  // Kill first so an active poll/read becomes ready immediately. Descriptors
  // remain open until the serialized operation has observed that termination,
  // avoiding an fd-number reuse race in the normal shutdown path.
  {
    std::lock_guard<std::mutex> process_lock(process_mutex_);
    worker_ready_ = false;
    if (worker_pid_ > 0) {
      terminateProcessBounded(static_cast<pid_t>(worker_pid_),
                              config_.shutdown_timeout);
      worker_pid_ = -1;
    }
  }

  std::unique_lock<std::timed_mutex> operation_lock(operation_mutex_,
                                                     std::defer_lock);
  const auto lock_budget = config_.shutdown_timeout + config_.poll_period * 2;
  (void)operation_lock.try_lock_for(lock_budget);
  std::lock_guard<std::mutex> process_lock(process_mutex_);
  closeFd(&worker_stdin_fd_);
  closeFd(&worker_stdout_fd_);
}

PythonEmbeddingEncoderStats PythonEmbeddingEncoder::stats() const {
  PythonEmbeddingEncoderStats result;
  {
    std::lock_guard<std::mutex> process_lock(process_mutex_);
    result.worker_running = worker_pid_ > 0 && worker_ready_;
  }
  result.shut_down = shutdown_requested_.load(std::memory_order_acquire);
  result.worker_starts = worker_starts_.load(std::memory_order_relaxed);
  result.worker_restarts = worker_restarts_.load(std::memory_order_relaxed);
  result.requests = requests_.load(std::memory_order_relaxed);
  result.encoded_documents =
      encoded_documents_.load(std::memory_order_relaxed);
  result.request_failures = request_failures_.load(std::memory_order_relaxed);
  result.transport_failures =
      transport_failures_.load(std::memory_order_relaxed);
  result.timeouts = timeouts_.load(std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> error_lock(error_mutex_);
    result.last_error = last_error_;
  }
  return result;
}

bool PythonEmbeddingEncoder::ensureWorker(Clock::time_point deadline,
                                          std::string* error) {
  if (shutdown_requested_.load(std::memory_order_acquire)) {
    *error = "embedding worker startup cancelled because encoder is shut down";
    return false;
  }
  {
    std::lock_guard<std::mutex> process_lock(process_mutex_);
    if (worker_pid_ > 0) {
      int status = 0;
      pid_t wait_result = -1;
      do {
        wait_result =
            ::waitpid(static_cast<pid_t>(worker_pid_), &status, WNOHANG);
      } while (wait_result < 0 && errno == EINTR);
      if (wait_result == 0 && worker_ready_) {
        error->clear();
        return true;
      }
      if (wait_result == 0 && !worker_ready_) {
        *error = "embedding worker exists without a completed startup handshake";
        return false;
      }
      if (wait_result < 0 && errno != ECHILD) {
        *error = errnoMessage("waitpid embedding worker", errno);
        return false;
      }
      worker_pid_ = -1;
      worker_ready_ = false;
      closeFd(&worker_stdin_fd_);
      closeFd(&worker_stdout_fd_);
    }
  }
  return spawnWorker(deadline, error);
}

bool PythonEmbeddingEncoder::spawnWorker(Clock::time_point deadline,
                                         std::string* error) {
  int to_child[2] = {-1, -1};
  int from_child[2] = {-1, -1};
  if (::pipe(to_child) != 0 || ::pipe(from_child) != 0) {
    const int pipe_error = errno;
    closeFd(&to_child[0]);
    closeFd(&to_child[1]);
    closeFd(&from_child[0]);
    closeFd(&from_child[1]);
    *error = errnoMessage("pipe embedding worker", pipe_error);
    return false;
  }
  if (!setCloseOnExec(to_child[0], error) ||
      !setCloseOnExec(to_child[1], error) ||
      !setCloseOnExec(from_child[0], error) ||
      !setCloseOnExec(from_child[1], error) ||
      !setNonBlocking(to_child[1], error) ||
      !setNonBlocking(from_child[0], error)) {
    closeFd(&to_child[0]);
    closeFd(&to_child[1]);
    closeFd(&from_child[0]);
    closeFd(&from_child[1]);
    return false;
  }

  std::vector<std::string> arguments = {
      config_.python_executable,
      "-u",
      config_.worker_script,
      "--model",
      config_.model_path,
      "--model-id",
      model_id_,
      "--device",
      config_.device,
      "--expected-dimension",
      std::to_string(dimension_),
      "--maximum-batch-size",
      std::to_string(config_.maximum_batch_size),
      "--maximum-message-bytes",
      std::to_string(config_.maximum_message_bytes),
  };
  std::vector<char*> argv;
  argv.reserve(arguments.size() + 1);
  for (std::string& argument : arguments) {
    argv.push_back(argument.data());
  }
  argv.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  int spawn_error = ::posix_spawn_file_actions_init(&actions);
  if (spawn_error != 0) {
    closeFd(&to_child[0]);
    closeFd(&to_child[1]);
    closeFd(&from_child[0]);
    closeFd(&from_child[1]);
    *error = errnoMessage("posix_spawn_file_actions_init", spawn_error);
    return false;
  }
  auto add_action = [&](int result) {
    if (spawn_error == 0 && result != 0) {
      spawn_error = result;
    }
  };
  add_action(::posix_spawn_file_actions_adddup2(&actions, to_child[0],
                                                 STDIN_FILENO));
  add_action(::posix_spawn_file_actions_adddup2(&actions, from_child[1],
                                                 STDOUT_FILENO));
  add_action(::posix_spawn_file_actions_addclose(&actions, to_child[0]));
  add_action(::posix_spawn_file_actions_addclose(&actions, to_child[1]));
  add_action(::posix_spawn_file_actions_addclose(&actions, from_child[0]));
  add_action(::posix_spawn_file_actions_addclose(&actions, from_child[1]));

  posix_spawnattr_t attributes;
  if (spawn_error == 0) {
    spawn_error = ::posix_spawnattr_init(&attributes);
  }
  bool attributes_initialized = spawn_error == 0;
  if (attributes_initialized) {
    sigset_t child_mask{};
    ::sigemptyset(&child_mask);
    add_action(::posix_spawnattr_setsigmask(&attributes, &child_mask));
    add_action(::posix_spawnattr_setflags(&attributes,
                                          POSIX_SPAWN_SETSIGMASK));
  }

  pid_t child_pid = -1;
  if (spawn_error == 0) {
    spawn_error = ::posix_spawnp(&child_pid, argv[0], &actions, &attributes,
                                 argv.data(), environ);
  }
  if (attributes_initialized) {
    ::posix_spawnattr_destroy(&attributes);
  }
  ::posix_spawn_file_actions_destroy(&actions);
  closeFd(&to_child[0]);
  closeFd(&from_child[1]);
  if (spawn_error != 0) {
    closeFd(&to_child[1]);
    closeFd(&from_child[0]);
    *error = errnoMessage("posix_spawn embedding worker", spawn_error);
    return false;
  }

  {
    std::lock_guard<std::mutex> process_lock(process_mutex_);
    if (shutdown_requested_.load(std::memory_order_acquire)) {
      closeFd(&to_child[1]);
      closeFd(&from_child[0]);
      terminateProcessBounded(child_pid, config_.shutdown_timeout);
      *error = "embedding worker startup cancelled because encoder is shut down";
      return false;
    }
    worker_pid_ = static_cast<int>(child_pid);
    worker_stdin_fd_ = to_child[1];
    worker_stdout_fd_ = from_child[0];
    worker_ready_ = false;
    if (ever_started_) {
      ++worker_restarts_;
    }
    ever_started_ = true;
    ++worker_starts_;
  }

  std::string payload;
  if (!readFrame(&payload, deadline, error)) {
    stopWorker(false);
    return false;
  }

  Json ready;
  try {
    ready = Json::parse(payload);
  } catch (const std::exception& exception) {
    *error = std::string("invalid embedding startup JSON: ") + exception.what();
    stopWorker(false);
    return false;
  }

  const std::unordered_set<std::string> success_keys = {
      "protocol", "id", "op", "ok", "model_id", "dimension"};
  const std::unordered_set<std::string> error_keys = {
      "protocol", "id", "op", "ok", "error"};
  if (!ready.is_object() || !ready.contains("protocol") ||
      !ready["protocol"].is_string() ||
      ready["protocol"].get<std::string>() != kProtocol ||
      !ready.contains("id") || !ready["id"].is_number_unsigned() ||
      ready["id"].get<std::uint64_t>() != kReadyMessageId ||
      !ready.contains("op") || !ready["op"].is_string() ||
      ready["op"].get<std::string>() != "ready" ||
      !ready.contains("ok") || !ready["ok"].is_boolean()) {
    *error = "embedding worker returned an invalid startup envelope";
    stopWorker(false);
    return false;
  }
  if (!ready["ok"].get<bool>()) {
    if (!hasOnlyKeys(ready, error_keys) || !ready.contains("error") ||
        !ready["error"].is_string() || ready["error"].get<std::string>().empty()) {
      *error = "embedding worker returned an invalid startup error";
    } else {
      *error = "embedding worker startup failed: " +
               ready["error"].get<std::string>();
    }
    stopWorker(false);
    return false;
  }
  if (!hasOnlyKeys(ready, success_keys) || !ready.contains("model_id") ||
      !ready["model_id"].is_string() ||
      ready["model_id"].get<std::string>() != model_id_ ||
      !ready.contains("dimension") ||
      !ready["dimension"].is_number_unsigned() ||
      ready["dimension"].get<std::size_t>() != dimension_) {
    *error = "embedding worker startup namespace does not match configuration";
    stopWorker(false);
    return false;
  }
  {
    std::lock_guard<std::mutex> process_lock(process_mutex_);
    if (worker_pid_ != static_cast<int>(child_pid) ||
        shutdown_requested_.load(std::memory_order_acquire)) {
      *error = "embedding worker startup was superseded or cancelled";
      return false;
    }
    worker_ready_ = true;
  }
  error->clear();
  return true;
}

void PythonEmbeddingEncoder::stopWorker(bool count_transport_failure) {
  std::lock_guard<std::mutex> process_lock(process_mutex_);
  worker_ready_ = false;
  if (worker_pid_ > 0) {
    terminateProcessBounded(static_cast<pid_t>(worker_pid_),
                            config_.shutdown_timeout);
    worker_pid_ = -1;
  }
  closeFd(&worker_stdin_fd_);
  closeFd(&worker_stdout_fd_);
  if (count_transport_failure) {
    ++transport_failures_;
  }
}

PythonEmbeddingEncoder::Reply PythonEmbeddingEncoder::transact(
    const std::vector<std::string>& documents,
    Clock::time_point deadline) {
  Reply reply;
  std::string error;
  const auto startup_deadline =
      std::min(deadline, Clock::now() + config_.startup_timeout);
  if (!ensureWorker(startup_deadline, &error)) {
    reply.error = std::move(error);
    return reply;
  }

  const std::uint64_t request_id =
      next_request_id_.fetch_add(1, std::memory_order_relaxed);
  Json request = {{"protocol", kProtocol},
                  {"id", request_id},
                  {"op", "encode"},
                  {"documents", documents}};
  std::string request_payload;
  try {
    request_payload = request.dump(-1, ' ', false,
                                   Json::error_handler_t::strict);
  } catch (const std::exception& exception) {
    reply.kind = ReplyKind::kWorkerError;
    reply.error = std::string("could not serialize embedding request: ") +
                  exception.what();
    return reply;
  }
  if (request_payload.size() > config_.maximum_message_bytes) {
    reply.kind = ReplyKind::kWorkerError;
    reply.error = "embedding request exceeds maximum_message_bytes";
    return reply;
  }
  if (!writeFrame(request_payload, deadline, &reply.error)) {
    return reply;
  }

  std::string response_payload;
  if (!readFrame(&response_payload, deadline, &reply.error)) {
    return reply;
  }
  Json response;
  try {
    response = Json::parse(response_payload);
  } catch (const std::exception& exception) {
    reply.error = std::string("invalid embedding response JSON: ") +
                  exception.what();
    return reply;
  }

  const std::unordered_set<std::string> success_keys = {
      "protocol", "id", "op", "ok", "model_id", "dimension",
      "embeddings"};
  const std::unordered_set<std::string> error_keys = {
      "protocol", "id", "op", "ok", "error"};
  if (!response.is_object() || !response.contains("protocol") ||
      !response["protocol"].is_string() ||
      response["protocol"].get<std::string>() != kProtocol ||
      !response.contains("id") || !response["id"].is_number_unsigned() ||
      response["id"].get<std::uint64_t>() != request_id ||
      !response.contains("op") || !response["op"].is_string() ||
      response["op"].get<std::string>() != "encode_result" ||
      !response.contains("ok") || !response["ok"].is_boolean()) {
    reply.error = "embedding worker returned an invalid response envelope";
    return reply;
  }
  if (!response["ok"].get<bool>()) {
    if (!hasOnlyKeys(response, error_keys) || !response.contains("error") ||
        !response["error"].is_string() ||
        response["error"].get<std::string>().empty()) {
      reply.error = "embedding worker returned an invalid error response";
      return reply;
    }
    reply.kind = ReplyKind::kWorkerError;
    reply.error = "embedding worker rejected request: " +
                  response["error"].get<std::string>();
    return reply;
  }
  if (!hasOnlyKeys(response, success_keys) ||
      !response.contains("model_id") || !response["model_id"].is_string() ||
      response["model_id"].get<std::string>() != model_id_ ||
      !response.contains("dimension") ||
      !response["dimension"].is_number_unsigned() ||
      response["dimension"].get<std::size_t>() != dimension_ ||
      !response.contains("embeddings") ||
      !response["embeddings"].is_array() ||
      response["embeddings"].size() != documents.size()) {
    reply.error = "embedding worker returned an invalid result schema";
    return reply;
  }

  reply.embeddings.reserve(documents.size());
  for (const Json& encoded : response["embeddings"]) {
    if (!encoded.is_array() || encoded.size() != dimension_) {
      reply.embeddings.clear();
      reply.error = "embedding worker returned a vector with wrong dimension";
      return reply;
    }
    std::vector<float> vector;
    vector.reserve(dimension_);
    double squared_norm = 0.0;
    for (const Json& component : encoded) {
      if (!component.is_number()) {
        reply.embeddings.clear();
        reply.error = "embedding worker returned a non-numeric vector component";
        return reply;
      }
      const double value = component.get<double>();
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<float>::max()) {
        reply.embeddings.clear();
        reply.error = "embedding worker returned a non-finite vector component";
        return reply;
      }
      squared_norm += value * value;
      vector.push_back(static_cast<float>(value));
    }
    const double norm = std::sqrt(squared_norm);
    if (!std::isfinite(norm) || norm <= std::numeric_limits<double>::epsilon()) {
      reply.embeddings.clear();
      reply.error = "embedding worker returned a zero or invalid vector";
      return reply;
    }
    for (float& component : vector) {
      component = static_cast<float>(static_cast<double>(component) / norm);
    }
    reply.embeddings.push_back(std::move(vector));
  }
  reply.kind = ReplyKind::kSuccess;
  reply.error.clear();
  return reply;
}

bool PythonEmbeddingEncoder::writeFrame(const std::string& payload,
                                        Clock::time_point deadline,
                                        std::string* error) {
  if (payload.size() > config_.maximum_message_bytes) {
    *error = "embedding request exceeds maximum_message_bytes";
    return false;
  }
  int fd = -1;
  {
    std::lock_guard<std::mutex> process_lock(process_mutex_);
    fd = worker_stdin_fd_;
  }
  if (fd < 0) {
    *error = "embedding request pipe is closed";
    return false;
  }
  const std::uint32_t size = static_cast<std::uint32_t>(payload.size());
  const std::uint8_t header[4] = {
      static_cast<std::uint8_t>(size & 0xffU),
      static_cast<std::uint8_t>((size >> 8U) & 0xffU),
      static_cast<std::uint8_t>((size >> 16U) & 0xffU),
      static_cast<std::uint8_t>((size >> 24U) & 0xffU),
  };
  return writeExact(fd, header, sizeof(header), deadline, error) &&
         writeExact(fd, payload.data(), payload.size(), deadline, error);
}

bool PythonEmbeddingEncoder::readFrame(std::string* payload,
                                       Clock::time_point deadline,
                                       std::string* error) {
  int fd = -1;
  {
    std::lock_guard<std::mutex> process_lock(process_mutex_);
    fd = worker_stdout_fd_;
  }
  if (fd < 0) {
    *error = "embedding response pipe is closed";
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
  if (size > config_.maximum_message_bytes) {
    *error = "embedding response exceeds maximum_message_bytes";
    return false;
  }
  payload->assign(size, '\0');
  return readExact(fd, payload->data(), payload->size(), deadline, error);
}

bool PythonEmbeddingEncoder::writeExact(int fd,
                                        const void* data,
                                        std::size_t size,
                                        Clock::time_point deadline,
                                        std::string* error) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    if (!waitForFd(fd, POLLOUT, deadline, error)) {
      return false;
    }
    const ssize_t written = ::write(fd, bytes + offset, size - offset);
    if (written > 0) {
      offset += static_cast<std::size_t>(written);
      continue;
    }
    if (written < 0 && (errno == EINTR || errno == EAGAIN ||
                        errno == EWOULDBLOCK)) {
      continue;
    }
    const int io_error = written < 0 ? errno : EIO;
    *error = errnoMessage("write embedding request", io_error);
    return false;
  }
  error->clear();
  return true;
}

bool PythonEmbeddingEncoder::readExact(int fd,
                                       void* data,
                                       std::size_t size,
                                       Clock::time_point deadline,
                                       std::string* error) {
  auto* bytes = static_cast<std::uint8_t*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    if (!waitForFd(fd, POLLIN, deadline, error)) {
      return false;
    }
    const ssize_t count = ::read(fd, bytes + offset, size - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
      continue;
    }
    if (count < 0 &&
        (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    if (count == 0) {
      *error = "embedding response pipe closed before a complete frame";
    } else {
      *error = errnoMessage("read embedding response", errno);
    }
    return false;
  }
  error->clear();
  return true;
}

bool PythonEmbeddingEncoder::waitForFd(int fd,
                                       short events,
                                       Clock::time_point deadline,
                                       std::string* error) {
  while (true) {
    if (shutdown_requested_.load(std::memory_order_acquire)) {
      *error = "embedding IPC cancelled because encoder is shut down";
      return false;
    }
    const auto now = Clock::now();
    if (now >= deadline) {
      *error = "embedding IPC timed out";
      return false;
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    const auto wait = std::max(
        std::chrono::milliseconds(1), std::min(config_.poll_period, remaining));
    pollfd descriptor{};
    descriptor.fd = fd;
    descriptor.events = events;
    const int result =
        ::poll(&descriptor, 1, static_cast<int>(wait.count()));
    if (result > 0) {
      if ((descriptor.revents & events) != 0) {
        return true;
      }
      if ((descriptor.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
        *error = events == POLLIN
                     ? "embedding response pipe closed before data arrived"
                     : "embedding request pipe closed before data was written";
        return false;
      }
      continue;
    }
    if (result == 0 || (result < 0 && errno == EINTR)) {
      continue;
    }
    *error = errnoMessage("poll embedding worker", errno);
    return false;
  }
}

void PythonEmbeddingEncoder::recordError(const std::string& error) {
  std::lock_guard<std::mutex> error_lock(error_mutex_);
  last_error_ = error;
}

}  // namespace roomie
