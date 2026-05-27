#include "roomie/pipeline/python_inference_backend.hpp"

#include <sys/wait.h>
#include <spawn.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>

#include "roomie/utils/run_logger.hpp"

extern char** environ;

namespace roomie {
namespace {

constexpr char kRequestMagic[] = {'R', 'I', 'E', 'Q', '1'};
constexpr char kResponseMagic[] = {'R', 'I', 'R', 'S', '1'};
constexpr std::uint32_t kMaxMessageBytes = 64U * 1024U * 1024U;

void closeFd(int* fd) {
  if (*fd >= 0) {
    ::close(*fd);
    *fd = -1;
  }
}

template <typename T>
void appendPrimitive(std::vector<std::uint8_t>* out, T value) {
  static_assert(std::is_trivially_copyable<T>::value, "primitive must be copyable");
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
  out->insert(out->end(), bytes, bytes + sizeof(T));
}

void appendBytes(std::vector<std::uint8_t>* out, const void* data, std::size_t size) {
  if (size == 0) {
    return;
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(data);
  out->insert(out->end(), bytes, bytes + size);
}

void appendString(std::vector<std::uint8_t>* out, const std::string& value) {
  appendPrimitive<std::uint32_t>(out, static_cast<std::uint32_t>(value.size()));
  appendBytes(out, value.data(), value.size());
}

void appendImage(std::vector<std::uint8_t>* out, const ImageBuffer& image) {
  appendPrimitive<std::int32_t>(out, image.width);
  appendPrimitive<std::int32_t>(out, image.height);
  appendPrimitive<std::int32_t>(out, image.channels);
  appendString(out, image.encoding);
  appendPrimitive<std::uint32_t>(out, static_cast<std::uint32_t>(image.data.size()));
  appendBytes(out, image.data.data(), image.data.size());
}

class ResponseReader {
 public:
  explicit ResponseReader(const std::vector<std::uint8_t>& data) : data_(data) {}

  bool readMagic(const char* magic, std::size_t size) {
    if (!canRead(size)) {
      return false;
    }
    const bool ok = std::memcmp(data_.data() + offset_, magic, size) == 0;
    offset_ += size;
    return ok;
  }

  template <typename T>
  bool read(T* value) {
    static_assert(std::is_trivially_copyable<T>::value, "primitive must be copyable");
    if (!canRead(sizeof(T))) {
      return false;
    }
    std::memcpy(value, data_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return true;
  }

  bool readString(std::string* value) {
    std::uint32_t size = 0;
    if (!read(&size) || !canRead(size)) {
      return false;
    }
    value->assign(reinterpret_cast<const char*>(data_.data() + offset_), size);
    offset_ += size;
    return true;
  }

 private:
  bool canRead(std::size_t size) const { return size <= data_.size() - offset_; }

  const std::vector<std::uint8_t>& data_;
  std::size_t offset_ = 0;
};

std::string formatErrno(const char* operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

std::string dirnameOf(const std::string& path) {
  const std::size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return ".";
  }
  if (slash == 0) {
    return "/";
  }
  return path.substr(0, slash);
}

}  // namespace

PythonInferenceBackend::PythonInferenceBackend(PipelineConfig config)
    : WorkerThread("python_inference_backend"),
      request_queue_(config.inference_request_queue_size),
      response_queue_(config.inference_response_queue_size),
      config_(std::move(config)) {}

bool PythonInferenceBackend::enqueueRequest(InferenceRequest request) {
  return request_queue_.pushDropOldest(std::move(request));
}

bool PythonInferenceBackend::tryPopResponse(InferenceResponse* response) {
  return response_queue_.tryPop(response);
}

void PythonInferenceBackend::run() {
  RunLogger::logGlobal("inference",
                       "thread_start enabled=" +
                           std::string(config_.python_backend_enabled ? "true" : "false") +
                           " executable=" + config_.python_executable +
                           " worker=" + config_.python_worker_script +
                           " device=" + config_.inference_device);
  while (!stopRequested()) {
    InferenceRequest request;
    if (!request_queue_.waitPopFor(&request, std::chrono::milliseconds(50))) {
      continue;
    }

    InferenceResponse response;
    response.time_ns = request.time_ns;
    response.camera_id = request.camera_id;

    if (!config_.python_backend_enabled) {
      response.ok = false;
      response.error = "Python inference backend is disabled by config";
      RunLogger::logGlobal("inference", response.error);
      response_queue_.pushDropOldest(std::move(response));
      continue;
    }
    if (backend_disabled_after_failure_) {
      response.ok = false;
      response.error = "Python inference backend disabled after repeated IPC failures";
      RunLogger::logGlobal("inference", response.error);
      response_queue_.pushDropOldest(std::move(response));
      continue;
    }

    const auto transact_start = std::chrono::steady_clock::now();
    if (!transactWithWorker(request, &response)) {
      ++consecutive_ipc_failures_;
      response.time_ns = request.time_ns;
      response.camera_id = request.camera_id;
      response.ok = false;
      if (response.error.empty()) {
        response.error = "Python inference worker IPC failed";
      }
      RunLogger::logGlobal("inference",
                           "ipc_failure count=" +
                               std::to_string(consecutive_ipc_failures_) +
                               " error=" + response.error);
      stopWorkerProcess();
      if (consecutive_ipc_failures_ >= 3) {
        backend_disabled_after_failure_ = true;
        RunLogger::logGlobal("inference",
                             "disabled_after_repeated_ipc_failures");
      }
    } else {
      consecutive_ipc_failures_ = 0;
    }
    response.backend_ipc_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - transact_start)
            .count());
    RunLogger::logGlobal("inference",
                         "response camera=" + response.camera_id +
                             " t=" + std::to_string(response.time_ns) +
                             " ok=" + std::string(response.ok ? "true" : "false") +
                             " backend_ipc_ms=" +
                             std::to_string(response.backend_ipc_ms) +
                             " worker_ms=" +
                             std::to_string(response.python_worker_ms) +
                             " owl_ms=" + std::to_string(response.owl_ms) +
                             " boxernet_ms=" +
                             std::to_string(response.boxernet_ms));
    response_queue_.pushDropOldest(std::move(response));
  }
}

bool PythonInferenceBackend::ensureWorkerProcess(std::string* error) {
  if (worker_pid_.load() > 0) {
    int status = 0;
    const int existing_pid = worker_pid_.load();
    const pid_t wait_result =
        ::waitpid(static_cast<pid_t>(existing_pid), &status, WNOHANG);
    if (wait_result == 0) {
      return true;
    }
    const int stdin_fd = worker_stdin_fd_.exchange(-1);
    const int stdout_fd = worker_stdout_fd_.exchange(-1);
    if (stdin_fd >= 0) {
      ::close(stdin_fd);
    }
    if (stdout_fd >= 0) {
      ::close(stdout_fd);
    }
    worker_pid_.store(-1);
  }

  if (config_.python_executable.empty()) {
    *error = "detection.python_executable is empty";
    return false;
  }
  if (config_.python_worker_script.empty()) {
    *error = "detection.python_worker_script is empty";
    return false;
  }
  if (::access(config_.python_worker_script.c_str(), R_OK) != 0) {
    *error = "cannot read Python worker script: " + config_.python_worker_script;
    return false;
  }

  int to_child[2] = {-1, -1};
  int from_child[2] = {-1, -1};
  if (::pipe(to_child) != 0) {
    *error = formatErrno("pipe(to_child)");
    return false;
  }
  if (::pipe(from_child) != 0) {
    closeFd(&to_child[0]);
    closeFd(&to_child[1]);
    *error = formatErrno("pipe(from_child)");
    return false;
  }

  std::vector<std::string> args;
  args.push_back(config_.python_executable);
  args.push_back(config_.python_worker_script);
  args.push_back("--boxer-repo");
  args.push_back(config_.boxer_repo_path);
  args.push_back("--ckpt");
  args.push_back(config_.boxernet_ckpt_path);
  args.push_back("--device");
  args.push_back(config_.inference_device);
  args.push_back("--precision");
  args.push_back(config_.inference_precision);
  args.push_back("--owl-min-confidence");
  args.push_back(std::to_string(config_.owl_min_confidence));
  args.push_back("--owl-nms-iou-threshold");
  args.push_back(std::to_string(config_.owl_nms_iou_threshold));
  args.push_back("--boxernet-min-confidence");
  args.push_back(std::to_string(config_.boxernet_min_confidence));
  args.push_back("--robot-bbox-mask-overlap");
  args.push_back(std::to_string(config_.robot_bbox_mask_overlap));
  args.push_back("--robot-bbox-center-overlap");
  args.push_back(std::to_string(config_.robot_bbox_center_overlap));
  args.push_back("--robot-mask-dilate-px");
  args.push_back(std::to_string(config_.robot_mask_dilate_px));
  for (const std::string& prompt : config_.text_prompts) {
    args.push_back("--text-prompt");
    args.push_back(prompt);
  }

  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  if (::posix_spawn_file_actions_init(&actions) != 0) {
    closeFd(&to_child[0]);
    closeFd(&to_child[1]);
    closeFd(&from_child[0]);
    closeFd(&from_child[1]);
    *error = formatErrno("posix_spawn_file_actions_init");
    return false;
  }
  ::posix_spawn_file_actions_adddup2(&actions, to_child[0], STDIN_FILENO);
  ::posix_spawn_file_actions_adddup2(&actions, from_child[1], STDOUT_FILENO);
  ::posix_spawn_file_actions_addclose(&actions, to_child[0]);
  ::posix_spawn_file_actions_addclose(&actions, to_child[1]);
  ::posix_spawn_file_actions_addclose(&actions, from_child[0]);
  ::posix_spawn_file_actions_addclose(&actions, from_child[1]);

  ::setenv("PYTHONUNBUFFERED", "1", 1);
  ::setenv("BOXER_CKPT_DIR", dirnameOf(config_.boxernet_ckpt_path).c_str(), 1);
  pid_t pid = 0;
  const int spawn_result =
      ::posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
  ::posix_spawn_file_actions_destroy(&actions);
  if (spawn_result != 0) {
    closeFd(&to_child[0]);
    closeFd(&to_child[1]);
    closeFd(&from_child[0]);
    closeFd(&from_child[1]);
    *error = std::string("posix_spawnp: ") + std::strerror(spawn_result);
    return false;
  }

  closeFd(&to_child[0]);
  closeFd(&from_child[1]);
  worker_pid_.store(static_cast<int>(pid));
  worker_stdin_fd_.store(to_child[1]);
  worker_stdout_fd_.store(from_child[0]);
  RunLogger::logGlobal("inference",
                       "spawned_worker pid=" + std::to_string(static_cast<long>(pid)));
  return true;
}

void PythonInferenceBackend::stopWorkerProcess() {
  const int stdin_fd = worker_stdin_fd_.exchange(-1);
  const int stdout_fd = worker_stdout_fd_.exchange(-1);
  if (stdin_fd >= 0) {
    ::close(stdin_fd);
  }
  if (stdout_fd >= 0) {
    ::close(stdout_fd);
  }

  const int pid = worker_pid_.exchange(-1);
  if (pid > 0) {
    RunLogger::logGlobal("inference",
                         "stopping_worker pid=" + std::to_string(pid));
    ::kill(static_cast<pid_t>(pid), SIGTERM);
    int status = 0;
    while (::waitpid(static_cast<pid_t>(pid), &status, 0) < 0 && errno == EINTR) {
    }
  }
}

bool PythonInferenceBackend::transactWithWorker(const InferenceRequest& request,
                                                InferenceResponse* response) {
  std::string error;
  if (!ensureWorkerProcess(&error)) {
    response->error = error;
    return false;
  }

  if (!writeMessage(serializeRequest(request))) {
    response->error = "failed to write inference request to Python worker";
    return false;
  }

  std::vector<std::uint8_t> response_body;
  if (!readMessage(&response_body)) {
    response->error = "failed to read inference response from Python worker";
    return false;
  }

  if (!parseResponse(response_body, response, &error)) {
    response->ok = false;
    response->error = error;
    return false;
  }
  return true;
}

std::vector<std::uint8_t> PythonInferenceBackend::serializeRequest(
    const InferenceRequest& request) const {
  std::vector<std::uint8_t> body;
  body.reserve(3U * 960U * 960U + 64U * 1024U);
  appendBytes(&body, kRequestMagic, sizeof(kRequestMagic));
  appendPrimitive<std::int64_t>(&body, request.time_ns);
  appendString(&body, request.camera_id);
  appendImage(&body, request.rgb_960);
  appendImage(&body, request.mask_960);

  appendPrimitive<std::int32_t>(&body, request.intrinsics_960.width);
  appendPrimitive<std::int32_t>(&body, request.intrinsics_960.height);
  appendPrimitive<float>(&body, request.intrinsics_960.fx);
  appendPrimitive<float>(&body, request.intrinsics_960.fy);
  appendPrimitive<float>(&body, request.intrinsics_960.cx);
  appendPrimitive<float>(&body, request.intrinsics_960.cy);

  for (float value : request.patch_depth.values) {
    appendPrimitive<float>(&body, value);
  }
  appendPrimitive<std::int32_t>(&body, request.patch_depth.valid_patches);
  appendPrimitive<std::int32_t>(&body, request.patch_depth.projected_points);
  appendPrimitive<std::uint64_t>(&body, request.patch_depth.map_version);

  const Eigen::Matrix4f T_world_camera = request.T_world_camera.matrix();
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      appendPrimitive<float>(&body, T_world_camera(row, col));
    }
  }

  return body;
}

bool PythonInferenceBackend::parseResponse(const std::vector<std::uint8_t>& body,
                                           InferenceResponse* response,
                                           std::string* error) const {
  ResponseReader reader(body);
  if (!reader.readMagic(kResponseMagic, sizeof(kResponseMagic))) {
    *error = "Python worker response has invalid magic";
    return false;
  }

  std::uint8_t ok = 0;
  std::uint32_t filtered_2d_count = 0;
  std::uint32_t detection_count = 0;
  if (!reader.read(&response->time_ns) ||
      !reader.readString(&response->camera_id) ||
      !reader.read(&ok) ||
      !reader.readString(&response->error) ||
      !reader.read(&response->python_worker_ms) ||
      !reader.read(&response->python_preprocess_ms) ||
      !reader.read(&response->owl_ms) ||
      !reader.read(&response->robot_filter_ms) ||
      !reader.read(&response->boxernet_ms) ||
      !reader.read(&response->python_postprocess_ms) ||
      !reader.read(&filtered_2d_count)) {
    *error = "Python worker response header is truncated";
    return false;
  }

  response->ok = ok != 0;
  response->filtered_2d_detections.clear();
  response->filtered_2d_detections.reserve(filtered_2d_count);
  for (std::uint32_t i = 0; i < filtered_2d_count; ++i) {
    Raw2dDetection detection;
    if (!reader.read(&detection.score_2d)) {
      *error = "Python worker response 2D score is truncated";
      return false;
    }
    for (float& box_value : detection.box_xyxy) {
      if (!reader.read(&box_value)) {
        *error = "Python worker response filtered 2D box is truncated";
        return false;
      }
    }
    if (!reader.read(&detection.semantic_id) ||
        !reader.readString(&detection.label)) {
      *error = "Python worker response 2D label is truncated";
      return false;
    }
    response->filtered_2d_detections.push_back(std::move(detection));
  }

  if (!reader.read(&detection_count)) {
    *error = "Python worker response 3D detection count is truncated";
    return false;
  }
  response->detections.clear();
  response->detections.reserve(detection_count);
  for (std::uint32_t i = 0; i < detection_count; ++i) {
    RawDetection detection;
    if (!reader.read(&detection.center_world.x()) ||
        !reader.read(&detection.center_world.y()) ||
        !reader.read(&detection.center_world.z()) ||
        !reader.read(&detection.size_m.x()) ||
        !reader.read(&detection.size_m.y()) ||
        !reader.read(&detection.size_m.z()) ||
        !reader.read(&detection.yaw_rad) ||
        !reader.read(&detection.score_2d) ||
        !reader.read(&detection.score_3d)) {
      *error = "Python worker response detection is truncated";
      return false;
    }
    for (float& box_value : detection.box_xyxy) {
      if (!reader.read(&box_value)) {
        *error = "Python worker response 2D box is truncated";
        return false;
      }
    }
    if (!reader.read(&detection.semantic_id) ||
        !reader.readString(&detection.label)) {
      *error = "Python worker response label is truncated";
      return false;
    }
    response->detections.push_back(std::move(detection));
  }
  return true;
}

bool PythonInferenceBackend::writeMessage(const std::vector<std::uint8_t>& body) {
  if (body.size() > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const std::uint32_t size = static_cast<std::uint32_t>(body.size());
  const int fd = worker_stdin_fd_.load();
  return fd >= 0 &&
         writeExact(fd, &size, sizeof(size)) &&
         writeExact(fd, body.data(), body.size());
}

bool PythonInferenceBackend::readMessage(std::vector<std::uint8_t>* body) {
  std::uint32_t size = 0;
  const int fd = worker_stdout_fd_.load();
  if (fd < 0 || !readExact(fd, &size, sizeof(size)) || size > kMaxMessageBytes) {
    return false;
  }
  body->assign(size, 0);
  return readExact(fd, body->data(), body->size());
}

bool PythonInferenceBackend::writeExact(int fd, const void* data, std::size_t size) {
  const auto* ptr = static_cast<const std::uint8_t*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    const ssize_t written = ::write(fd, ptr + offset, size - offset);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (written == 0) {
      return false;
    }
    offset += static_cast<std::size_t>(written);
  }
  return true;
}

bool PythonInferenceBackend::readExact(int fd, void* data, std::size_t size) {
  auto* ptr = static_cast<std::uint8_t*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    const ssize_t n_read = ::read(fd, ptr + offset, size - offset);
    if (n_read < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (n_read == 0) {
      return false;
    }
    offset += static_cast<std::size_t>(n_read);
  }
  return true;
}

void PythonInferenceBackend::onStopRequested() {
  request_queue_.stop();
  response_queue_.stop();
  stopWorkerProcess();
}

}  // namespace roomie
