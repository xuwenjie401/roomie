#include "roomie/pipeline/python_inference_backend.hpp"

#include <sys/wait.h>
#include <poll.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>

#include "roomie/utils/run_logger.hpp"

extern char** environ;

namespace roomie {
namespace {

constexpr char kRequestMagic[] = {'R', 'I', 'E', 'Q', '2'};
constexpr char kResponseMagic[] = {'R', 'I', 'R', 'S', '2'};
constexpr std::size_t kMinimum2dDetectionBytes = 28U;
constexpr std::size_t kMinimum3dDetectionBytes = 56U;
constexpr std::chrono::milliseconds kWorkerPollPeriod{50};
constexpr std::chrono::milliseconds kWorkerTerminateGrace{100};
constexpr std::chrono::milliseconds kWorkerKillGrace{500};

void closeFd(int* fd) {
  if (*fd >= 0) {
    ::close(*fd);
    *fd = -1;
  }
}

void appendBytes(std::vector<std::uint8_t>* out, const void* data, std::size_t size) {
  if (size == 0) {
    return;
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(data);
  out->insert(out->end(), bytes, bytes + size);
}

void appendU8(std::vector<std::uint8_t>* out, std::uint8_t value) {
  out->push_back(value);
}

template <typename UInt>
void appendUnsigned(std::vector<std::uint8_t>* out, UInt value) {
  static_assert(std::is_unsigned<UInt>::value, "wire integer must be unsigned");
  for (std::size_t i = 0; i < sizeof(UInt); ++i) {
    out->push_back(static_cast<std::uint8_t>(value & static_cast<UInt>(0xffU)));
    value >>= 8U;
  }
}

void appendU32(std::vector<std::uint8_t>* out, std::uint32_t value) {
  appendUnsigned(out, value);
}

void appendU64(std::vector<std::uint8_t>* out, std::uint64_t value) {
  appendUnsigned(out, value);
}

void appendI32(std::vector<std::uint8_t>* out, std::int32_t value) {
  appendU32(out, static_cast<std::uint32_t>(value));
}

void appendI64(std::vector<std::uint8_t>* out, std::int64_t value) {
  appendU64(out, static_cast<std::uint64_t>(value));
}

void appendFloat(std::vector<std::uint8_t>* out, float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected float width");
  std::memcpy(&bits, &value, sizeof(bits));
  appendU32(out, bits);
}

void appendDouble(std::vector<std::uint8_t>* out, double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
  std::memcpy(&bits, &value, sizeof(bits));
  appendU64(out, bits);
}

bool appendString(std::vector<std::uint8_t>* out,
                  const std::string& value,
                  std::string* error) {
  if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
    if (error != nullptr) {
      *error = "wire string is too large";
    }
    return false;
  }
  appendU32(out, static_cast<std::uint32_t>(value.size()));
  appendBytes(out, value.data(), value.size());
  return true;
}

bool appendImage(std::vector<std::uint8_t>* out,
                 const ImageBuffer& image,
                 std::string* error) {
  if (image.data.size() > std::numeric_limits<std::uint32_t>::max()) {
    if (error != nullptr) {
      *error = "wire image is too large";
    }
    return false;
  }
  appendI32(out, image.width);
  appendI32(out, image.height);
  appendI32(out, image.channels);
  if (!appendString(out, image.encoding, error)) {
    return false;
  }
  appendU32(out, static_cast<std::uint32_t>(image.data.size()));
  appendBytes(out, image.data.data(), image.data.size());
  return true;
}

void appendRunId(std::vector<std::uint8_t>* out, const RunId& id) {
  appendU64(out, id.high);
  appendU64(out, id.low);
}

void appendProvenance(std::vector<std::uint8_t>* out,
                      const FrameProvenance& provenance) {
  appendRunId(out, provenance.run_id);
  appendU64(out, provenance.frame_id);
  appendU64(out, provenance.request_id);
  appendI64(out, provenance.sensor_time_ns);
  appendU8(out, static_cast<std::uint8_t>(provenance.map_mode));
  appendU8(out, provenance.includes_current_frame ? 1U : 0U);
  appendU8(out, provenance.causality_verified ? 1U : 0U);
  appendRunId(out, provenance.map.map_epoch);
  appendU64(out, provenance.map.map_revision);
  appendI64(out, provenance.map.integrated_through_ns);
  appendRunId(out, provenance.surface.map_epoch);
  appendU64(out, provenance.surface.surface_revision);
  appendU64(out, provenance.surface.source_map_revision);
}

void appendTiming(std::vector<std::uint8_t>* out, const PipelineTiming& timing) {
  appendDouble(out, timing.serialize_ms);
  appendDouble(out, timing.pipe_write_ms);
  appendDouble(out, timing.pipe_read_ms);
  appendDouble(out, timing.worker_queue_ms);
  appendDouble(out, timing.response_forward_ms);
}

class WireReader {
 public:
  explicit WireReader(const std::vector<std::uint8_t>& data) : data_(data) {}

  bool readMagic(const char* magic, std::size_t size) {
    if (!canRead(size)) {
      return false;
    }
    const bool ok = std::memcmp(data_.data() + offset_, magic, size) == 0;
    offset_ += size;
    return ok;
  }

  bool readU8(std::uint8_t* value) {
    if (!canRead(1U)) {
      return false;
    }
    *value = data_[offset_++];
    return true;
  }

  template <typename UInt>
  bool readUnsigned(UInt* value) {
    static_assert(std::is_unsigned<UInt>::value, "wire integer must be unsigned");
    if (!canRead(sizeof(UInt))) {
      return false;
    }
    UInt decoded = 0;
    for (std::size_t i = 0; i < sizeof(UInt); ++i) {
      decoded |= static_cast<UInt>(data_[offset_ + i]) << (8U * i);
    }
    offset_ += sizeof(UInt);
    *value = decoded;
    return true;
  }

  bool readU32(std::uint32_t* value) { return readUnsigned(value); }
  bool readU64(std::uint64_t* value) { return readUnsigned(value); }

  bool readI32(std::int32_t* value) {
    std::uint32_t bits = 0;
    if (!readU32(&bits)) {
      return false;
    }
    std::memcpy(value, &bits, sizeof(bits));
    return true;
  }

  bool readI64(std::int64_t* value) {
    std::uint64_t bits = 0;
    if (!readU64(&bits)) {
      return false;
    }
    std::memcpy(value, &bits, sizeof(bits));
    return true;
  }

  bool readFloat(float* value) {
    std::uint32_t bits = 0;
    if (!readU32(&bits)) {
      return false;
    }
    std::memcpy(value, &bits, sizeof(bits));
    return true;
  }

  bool readDouble(double* value) {
    std::uint64_t bits = 0;
    if (!readU64(&bits)) {
      return false;
    }
    std::memcpy(value, &bits, sizeof(bits));
    return true;
  }

  bool readString(std::string* value) {
    std::uint32_t size = 0;
    if (!readU32(&size) || !canRead(size)) {
      return false;
    }
    value->assign(reinterpret_cast<const char*>(data_.data() + offset_), size);
    offset_ += size;
    return true;
  }

  bool readImage(ImageBuffer* image) {
    std::int32_t width = 0;
    std::int32_t height = 0;
    std::int32_t channels = 0;
    std::uint32_t data_size = 0;
    if (!readI32(&width) || !readI32(&height) || !readI32(&channels) ||
        !readString(&image->encoding) ||
        !readU32(&data_size) || !canRead(data_size)) {
      return false;
    }
    image->width = width;
    image->height = height;
    image->channels = channels;
    image->data.assign(data_.begin() + static_cast<std::ptrdiff_t>(offset_),
                       data_.begin() + static_cast<std::ptrdiff_t>(offset_ + data_size));
    offset_ += data_size;
    return true;
  }

  std::size_t remaining() const { return data_.size() - offset_; }
  bool atEnd() const { return offset_ == data_.size(); }

 private:
  bool canRead(std::size_t size) const { return size <= data_.size() - offset_; }

  const std::vector<std::uint8_t>& data_;
  std::size_t offset_ = 0;
};

bool readRunId(WireReader* reader, RunId* id) {
  return reader->readU64(&id->high) && reader->readU64(&id->low);
}

bool readProvenance(WireReader* reader,
                    FrameProvenance* provenance,
                    std::string* error) {
  std::uint8_t map_mode = 0;
  std::uint8_t includes_current_frame = 0;
  std::uint8_t causality_verified = 0;
  if (!readRunId(reader, &provenance->run_id) ||
      !reader->readU64(&provenance->frame_id) ||
      !reader->readU64(&provenance->request_id) ||
      !reader->readI64(&provenance->sensor_time_ns) ||
      !reader->readU8(&map_mode) ||
      !reader->readU8(&includes_current_frame) ||
      !reader->readU8(&causality_verified) ||
      !readRunId(reader, &provenance->map.map_epoch) ||
      !reader->readU64(&provenance->map.map_revision) ||
      !reader->readI64(&provenance->map.integrated_through_ns) ||
      !readRunId(reader, &provenance->surface.map_epoch) ||
      !reader->readU64(&provenance->surface.surface_revision) ||
      !reader->readU64(&provenance->surface.source_map_revision)) {
    if (error != nullptr) {
      *error = "wire provenance is truncated";
    }
    return false;
  }
  if (map_mode > static_cast<std::uint8_t>(MapMode::kFrozen) ||
      includes_current_frame > 1U || causality_verified > 1U) {
    if (error != nullptr) {
      *error = "wire provenance contains an invalid enum or boolean";
    }
    return false;
  }
  provenance->map_mode = static_cast<MapMode>(map_mode);
  provenance->includes_current_frame = includes_current_frame != 0U;
  provenance->causality_verified = causality_verified != 0U;
  return true;
}

bool readTiming(WireReader* reader, PipelineTiming* timing) {
  return reader->readDouble(&timing->serialize_ms) &&
         reader->readDouble(&timing->pipe_write_ms) &&
         reader->readDouble(&timing->pipe_read_ms) &&
         reader->readDouble(&timing->worker_queue_ms) &&
         reader->readDouble(&timing->response_forward_ms);
}

std::string formatErrno(const char* operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

std::string formatIoError(const char* operation, int error_number) {
  if (error_number == 0) {
    return std::string(operation) + ": unexpected end of file";
  }
  std::string result =
      std::string(operation) + ": " + std::strerror(error_number);
  if (error_number == EPIPE) {
    result += " (EPIPE)";
  }
  return result;
}

bool waitForProcessExit(pid_t pid, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    int status = 0;
    const pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == pid || (result < 0 && errno == ECHILD)) {
      return true;
    }
    if (result < 0 && errno != EINTR) {
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void terminateProcessBounded(pid_t pid) {
  if (pid <= 0) {
    return;
  }
  if (::kill(pid, SIGTERM) != 0 && errno != ESRCH) {
    RunLogger::logGlobal(
        "inference",
        "failed to SIGTERM worker pid=" + std::to_string(pid) +
            " error=" + std::strerror(errno));
  }
  if (waitForProcessExit(pid, kWorkerTerminateGrace)) {
    return;
  }

  RunLogger::logGlobal(
      "inference",
      "worker did not exit after SIGTERM; sending SIGKILL pid=" +
          std::to_string(pid));
  if (::kill(pid, SIGKILL) != 0 && errno != ESRCH) {
    RunLogger::logGlobal(
        "inference",
        "failed to SIGKILL worker pid=" + std::to_string(pid) +
            " error=" + std::strerror(errno));
  }
  if (!waitForProcessExit(pid, kWorkerKillGrace)) {
    // Never turn process reaping into an unbounded pipeline shutdown. A child
    // stuck in uninterruptible kernel sleep remains visible until it can exit
    // or until the Roomie parent terminates.
    RunLogger::logGlobal(
        "inference",
        "worker could not be reaped within shutdown bound pid=" +
            std::to_string(pid));
  }
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

namespace inference_wire {

bool encodeRequest(const InferenceRequest& request,
                   std::vector<std::uint8_t>* body,
                   std::string* error) {
  if (body == nullptr) {
    if (error != nullptr) {
      *error = "request output body is null";
    }
    return false;
  }
  if (static_cast<std::uint8_t>(request.provenance.map_mode) >
      static_cast<std::uint8_t>(MapMode::kFrozen)) {
    if (error != nullptr) {
      *error = "request provenance has invalid map mode";
    }
    return false;
  }

  std::vector<std::uint8_t> encoded;
  encoded.reserve(3U * 960U * 960U + 64U * 1024U);
  appendBytes(&encoded, kRequestMagic, sizeof(kRequestMagic));
  appendI64(&encoded, request.time_ns);
  appendProvenance(&encoded, request.provenance);
  appendTiming(&encoded, request.timing);
  if (!appendString(&encoded, request.camera_id, error) ||
      !appendImage(&encoded, request.rgb_960, error) ||
      !appendImage(&encoded, request.mask_960, error)) {
    return false;
  }

  appendI32(&encoded, request.intrinsics_960.width);
  appendI32(&encoded, request.intrinsics_960.height);
  appendFloat(&encoded, request.intrinsics_960.fx);
  appendFloat(&encoded, request.intrinsics_960.fy);
  appendFloat(&encoded, request.intrinsics_960.cx);
  appendFloat(&encoded, request.intrinsics_960.cy);

  for (float value : request.patch_depth.values) {
    appendFloat(&encoded, value);
  }
  appendI32(&encoded, request.patch_depth.valid_patches);
  appendI32(&encoded, request.patch_depth.projected_points);
  appendU64(&encoded, request.patch_depth.map_version);

  const Eigen::Matrix4f T_world_camera = request.T_world_camera.matrix();
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      appendFloat(&encoded, T_world_camera(row, col));
    }
  }

  if (encoded.size() > kMaxMessageBytes) {
    if (error != nullptr) {
      *error = "request body exceeds the IPC size limit";
    }
    return false;
  }
  *body = std::move(encoded);
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

bool decodeRequest(const std::vector<std::uint8_t>& body,
                   InferenceRequest* request,
                   std::string* error) {
  if (request == nullptr) {
    if (error != nullptr) {
      *error = "request output is null";
    }
    return false;
  }
  if (body.size() > kMaxMessageBytes) {
    if (error != nullptr) {
      *error = "request body exceeds the IPC size limit";
    }
    return false;
  }

  WireReader reader(body);
  if (!reader.readMagic(kRequestMagic, sizeof(kRequestMagic))) {
    if (error != nullptr) {
      *error = "Python worker request has invalid magic";
    }
    return false;
  }

  InferenceRequest decoded;
  if (!reader.readI64(&decoded.time_ns) ||
      !readProvenance(&reader, &decoded.provenance, error) ||
      !readTiming(&reader, &decoded.timing) ||
      !reader.readString(&decoded.camera_id) ||
      !reader.readImage(&decoded.rgb_960) ||
      !reader.readImage(&decoded.mask_960)) {
    if (error != nullptr && error->empty()) {
      *error = "Python worker request header is truncated";
    }
    return false;
  }

  std::int32_t intrinsics_width = 0;
  std::int32_t intrinsics_height = 0;
  if (!reader.readI32(&intrinsics_width) ||
      !reader.readI32(&intrinsics_height) ||
      !reader.readFloat(&decoded.intrinsics_960.fx) ||
      !reader.readFloat(&decoded.intrinsics_960.fy) ||
      !reader.readFloat(&decoded.intrinsics_960.cx) ||
      !reader.readFloat(&decoded.intrinsics_960.cy)) {
    if (error != nullptr) {
      *error = "Python worker request intrinsics are truncated";
    }
    return false;
  }
  decoded.intrinsics_960.width = intrinsics_width;
  decoded.intrinsics_960.height = intrinsics_height;

  for (float& value : decoded.patch_depth.values) {
    if (!reader.readFloat(&value)) {
      if (error != nullptr) {
        *error = "Python worker request patch depth is truncated";
      }
      return false;
    }
  }
  std::int32_t valid_patches = 0;
  std::int32_t projected_points = 0;
  if (!reader.readI32(&valid_patches) ||
      !reader.readI32(&projected_points) ||
      !reader.readU64(&decoded.patch_depth.map_version)) {
    if (error != nullptr) {
      *error = "Python worker request patch metadata is truncated";
    }
    return false;
  }
  decoded.patch_depth.valid_patches = valid_patches;
  decoded.patch_depth.projected_points = projected_points;
  decoded.patch_depth.provenance = decoded.provenance;

  Eigen::Matrix4f T_world_camera;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      if (!reader.readFloat(&T_world_camera(row, col))) {
        if (error != nullptr) {
          *error = "Python worker request camera transform is truncated";
        }
        return false;
      }
    }
  }
  decoded.T_world_camera.matrix() = T_world_camera;

  if (!reader.atEnd()) {
    if (error != nullptr) {
      *error = "Python worker request has trailing bytes";
    }
    return false;
  }
  *request = std::move(decoded);
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

bool encodeResponse(const InferenceResponse& response,
                    std::vector<std::uint8_t>* body,
                    std::string* error) {
  if (body == nullptr) {
    if (error != nullptr) {
      *error = "response output body is null";
    }
    return false;
  }
  if (static_cast<std::uint8_t>(response.provenance.map_mode) >
      static_cast<std::uint8_t>(MapMode::kFrozen) ||
      response.filtered_2d_detections.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      response.detections.size() > std::numeric_limits<std::uint32_t>::max()) {
    if (error != nullptr) {
      *error = "response contains an out-of-range wire value";
    }
    return false;
  }

  std::vector<std::uint8_t> encoded;
  encoded.reserve(1024U + response.filtered_2d_detections.size() * 64U +
                  response.detections.size() * 96U);
  appendBytes(&encoded, kResponseMagic, sizeof(kResponseMagic));
  appendI64(&encoded, response.time_ns);
  appendProvenance(&encoded, response.provenance);
  appendTiming(&encoded, response.timing);
  if (!appendString(&encoded, response.camera_id, error)) {
    return false;
  }
  appendU8(&encoded, response.ok ? 1U : 0U);
  if (!appendString(&encoded, response.error, error)) {
    return false;
  }
  appendFloat(&encoded, response.python_worker_ms);
  appendFloat(&encoded, response.python_preprocess_ms);
  appendFloat(&encoded, response.owl_ms);
  appendFloat(&encoded, response.robot_filter_ms);
  appendFloat(&encoded, response.boxernet_ms);
  appendFloat(&encoded, response.python_postprocess_ms);

  appendU32(&encoded,
            static_cast<std::uint32_t>(response.filtered_2d_detections.size()));
  for (const Raw2dDetection& detection : response.filtered_2d_detections) {
    appendFloat(&encoded, detection.score_2d);
    for (float value : detection.box_xyxy) {
      appendFloat(&encoded, value);
    }
    appendI32(&encoded, detection.semantic_id);
    if (!appendString(&encoded, detection.label, error)) {
      return false;
    }
  }

  appendU32(&encoded, static_cast<std::uint32_t>(response.detections.size()));
  for (const RawDetection& detection : response.detections) {
    appendFloat(&encoded, detection.center_world.x());
    appendFloat(&encoded, detection.center_world.y());
    appendFloat(&encoded, detection.center_world.z());
    appendFloat(&encoded, detection.size_m.x());
    appendFloat(&encoded, detection.size_m.y());
    appendFloat(&encoded, detection.size_m.z());
    appendFloat(&encoded, detection.yaw_rad);
    appendFloat(&encoded, detection.score_2d);
    appendFloat(&encoded, detection.score_3d);
    for (float value : detection.box_xyxy) {
      appendFloat(&encoded, value);
    }
    appendI32(&encoded, detection.semantic_id);
    if (!appendString(&encoded, detection.label, error)) {
      return false;
    }
  }

  if (encoded.size() > kMaxMessageBytes) {
    if (error != nullptr) {
      *error = "response body exceeds the IPC size limit";
    }
    return false;
  }
  *body = std::move(encoded);
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

bool decodeResponse(const std::vector<std::uint8_t>& body,
                    InferenceResponse* response,
                    std::string* error) {
  if (response == nullptr) {
    if (error != nullptr) {
      *error = "response output is null";
    }
    return false;
  }
  if (body.size() > kMaxMessageBytes) {
    if (error != nullptr) {
      *error = "response body exceeds the IPC size limit";
    }
    return false;
  }

  WireReader reader(body);
  if (!reader.readMagic(kResponseMagic, sizeof(kResponseMagic))) {
    if (error != nullptr) {
      *error = "Python worker response has invalid magic";
    }
    return false;
  }

  InferenceResponse decoded;
  std::uint8_t ok = 0;
  std::uint32_t filtered_2d_count = 0;
  if (!reader.readI64(&decoded.time_ns) ||
      !readProvenance(&reader, &decoded.provenance, error) ||
      !readTiming(&reader, &decoded.timing) ||
      !reader.readString(&decoded.camera_id) ||
      !reader.readU8(&ok) || ok > 1U ||
      !reader.readString(&decoded.error) ||
      !reader.readFloat(&decoded.python_worker_ms) ||
      !reader.readFloat(&decoded.python_preprocess_ms) ||
      !reader.readFloat(&decoded.owl_ms) ||
      !reader.readFloat(&decoded.robot_filter_ms) ||
      !reader.readFloat(&decoded.boxernet_ms) ||
      !reader.readFloat(&decoded.python_postprocess_ms) ||
      !reader.readU32(&filtered_2d_count)) {
    if (error != nullptr && error->empty()) {
      *error = "Python worker response header is truncated or invalid";
    }
    return false;
  }
  if (filtered_2d_count > reader.remaining() / kMinimum2dDetectionBytes) {
    if (error != nullptr) {
      *error = "Python worker response has an invalid 2D detection count";
    }
    return false;
  }
  decoded.ok = ok != 0U;
  decoded.filtered_2d_detections.reserve(filtered_2d_count);
  for (std::uint32_t i = 0; i < filtered_2d_count; ++i) {
    Raw2dDetection detection;
    std::int32_t semantic_id = -1;
    if (!reader.readFloat(&detection.score_2d)) {
      if (error != nullptr) {
        *error = "Python worker response 2D score is truncated";
      }
      return false;
    }
    for (float& value : detection.box_xyxy) {
      if (!reader.readFloat(&value)) {
        if (error != nullptr) {
          *error = "Python worker response filtered 2D box is truncated";
        }
        return false;
      }
    }
    if (!reader.readI32(&semantic_id) || !reader.readString(&detection.label)) {
      if (error != nullptr) {
        *error = "Python worker response 2D label is truncated";
      }
      return false;
    }
    detection.semantic_id = semantic_id;
    decoded.filtered_2d_detections.push_back(std::move(detection));
  }

  std::uint32_t detection_count = 0;
  if (!reader.readU32(&detection_count) ||
      detection_count > reader.remaining() / kMinimum3dDetectionBytes) {
    if (error != nullptr) {
      *error = "Python worker response has an invalid 3D detection count";
    }
    return false;
  }
  decoded.detections.reserve(detection_count);
  for (std::uint32_t i = 0; i < detection_count; ++i) {
    RawDetection detection;
    std::int32_t semantic_id = -1;
    if (!reader.readFloat(&detection.center_world.x()) ||
        !reader.readFloat(&detection.center_world.y()) ||
        !reader.readFloat(&detection.center_world.z()) ||
        !reader.readFloat(&detection.size_m.x()) ||
        !reader.readFloat(&detection.size_m.y()) ||
        !reader.readFloat(&detection.size_m.z()) ||
        !reader.readFloat(&detection.yaw_rad) ||
        !reader.readFloat(&detection.score_2d) ||
        !reader.readFloat(&detection.score_3d)) {
      if (error != nullptr) {
        *error = "Python worker response detection is truncated";
      }
      return false;
    }
    for (float& value : detection.box_xyxy) {
      if (!reader.readFloat(&value)) {
        if (error != nullptr) {
          *error = "Python worker response 2D box is truncated";
        }
        return false;
      }
    }
    if (!reader.readI32(&semantic_id) || !reader.readString(&detection.label)) {
      if (error != nullptr) {
        *error = "Python worker response label is truncated";
      }
      return false;
    }
    detection.semantic_id = semantic_id;
    decoded.detections.push_back(std::move(detection));
  }
  if (!reader.atEnd()) {
    if (error != nullptr) {
      *error = "Python worker response has trailing bytes";
    }
    return false;
  }
  *response = std::move(decoded);
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

}  // namespace inference_wire

PythonInferenceBackend::PythonInferenceBackend(PipelineConfig config)
    : WorkerThread("python_inference_backend"),
      request_queue_(
          config.inference_request_queue_size,
          ChannelPolicy::kLatestByKey,
          [](const InferenceRequest& lhs, const InferenceRequest& rhs) {
            return lhs.camera_id == rhs.camera_id;
          }),
      response_queue_(config.inference_response_queue_size,
                      ChannelPolicy::kReliableBlocking),
      config_(std::move(config)) {}

ChannelStats PythonInferenceBackend::requestChannelStats() const {
  return request_queue_.stats();
}

ChannelStats PythonInferenceBackend::responseChannelStats() const {
  return response_queue_.stats();
}

PushResult<InferenceRequest> PythonInferenceBackend::enqueueRequest(
    InferenceRequest request) {
  const RequestId request_id = request.provenance.request_id;
  if (shutdown_requested_.load(std::memory_order_acquire) || stopRequested()) {
    RunLogger::logGlobal(
        "inference",
        "request_rejected request_id=" + std::to_string(request_id) +
            " reason=backend_shutting_down");
    PushResult<InferenceRequest> result;
    result.outcome = PushOutcome::kStopped;
    result.unconsumed_item.emplace(std::move(request));
    return result;
  }

  const auto now = std::chrono::steady_clock::now();
  if (request.due_time != std::chrono::steady_clock::time_point::max() &&
      now >= request.due_time) {
    RunLogger::logGlobal(
        "inference",
        "request_rejected request_id=" + std::to_string(request_id) +
            " reason=deadline_before_enqueue");
    PushResult<InferenceRequest> result;
    result.outcome = PushOutcome::kTimedOut;
    result.unconsumed_item.emplace(std::move(request));
    return result;
  }

  // Serialize publication with completion accounting so an extremely fast
  // worker cannot decrement an accepted request before its increment lands.
  std::lock_guard<std::mutex> accounting_lock(request_accounting_mutex_);
  if (shutdown_requested_.load(std::memory_order_acquire) || stopRequested()) {
    RunLogger::logGlobal(
        "inference",
        "request_rejected request_id=" + std::to_string(request_id) +
            " reason=backend_shutting_down");
    PushResult<InferenceRequest> result;
    result.outcome = PushOutcome::kStopped;
    result.unconsumed_item.emplace(std::move(request));
    return result;
  }
  request.backend_enqueued_at = std::chrono::steady_clock::now();
  PushResult<InferenceRequest> result = request_queue_.push(std::move(request));
  if (result.replaced_item) {
    RunLogger::logGlobal(
        "inference",
        "request_superseded request_id=" +
            std::to_string(result.replaced_item->provenance.request_id) +
            " by_request_id=" + std::to_string(request_id) +
            " reason=" +
            std::to_string(static_cast<int>(result.replacement_reason)));
  } else if (!result.accepted()) {
    RunLogger::logGlobal(
        "inference",
        "request_rejected request_id=" + std::to_string(request_id) +
            " reason=channel_outcome_" +
            std::to_string(static_cast<int>(result.outcome)));
  }
  if (result.outcome == PushOutcome::kAccepted) {
    outstanding_requests_.fetch_add(1, std::memory_order_release);
  }
  return result;
}

bool PythonInferenceBackend::tryPopResponse(InferenceResponse* response) {
  return response_queue_.tryPop(response);
}

bool PythonInferenceBackend::idle() const {
  return outstanding_requests_.load(std::memory_order_acquire) == 0;
}

void PythonInferenceBackend::beginShutdown() {
  bool first_request = false;
  {
    // This is the admission barrier: once shutdown_requested_ becomes visible,
    // no accepted request can still be waiting for its accounting increment.
    std::lock_guard<std::mutex> accounting_lock(request_accounting_mutex_);
    first_request =
        !shutdown_requested_.exchange(true, std::memory_order_acq_rel);
    request_queue_.stop();
  }
  if (first_request) {
    RunLogger::logGlobal("inference", "coordinated_shutdown_begin");
  }

  // Pending ingress remains drainable so the backend thread can turn every
  // accepted request into a terminal response.
  stopWorkerProcess();
}

void PythonInferenceBackend::run() {
  sigset_t sigpipe_set;
  ::sigemptyset(&sigpipe_set);
  ::sigaddset(&sigpipe_set, SIGPIPE);
  const int mask_error = ::pthread_sigmask(SIG_BLOCK, &sigpipe_set, nullptr);
  const bool sigpipe_protected = mask_error == 0;
  if (mask_error != 0) {
    RunLogger::logGlobal(
        "inference",
        "failed_to_block_sigpipe error=" +
            std::string(std::strerror(mask_error)));
  }

  RunLogger::logGlobal("inference",
                       "thread_start enabled=" +
                           std::string(config_.python_backend_enabled ? "true" : "false") +
                           " executable=" + config_.python_executable +
                           " worker=" + config_.python_worker_script +
                           " device=" + config_.inference_device);
  while (true) {
    const bool shutting_down =
        shutdown_requested_.load(std::memory_order_acquire) || stopRequested();
    if (shutting_down && request_queue_.empty()) {
      break;
    }

    InferenceRequest request;
    if (!request_queue_.waitPopFor(&request, std::chrono::milliseconds(50))) {
      if ((shutdown_requested_.load(std::memory_order_acquire) ||
           stopRequested()) &&
          request_queue_.empty()) {
        break;
      }
      continue;
    }
    processing_request_.store(true, std::memory_order_release);

    InferenceResponse response;
    response.time_ns = request.time_ns;
    response.provenance = request.provenance;
    response.timing = request.timing;
    response.camera_id = request.camera_id;

    const auto transact_start = std::chrono::steady_clock::now();
    const auto backend_enqueued_at =
        request.backend_enqueued_at ==
                std::chrono::steady_clock::time_point::min()
            ? transact_start
            : request.backend_enqueued_at;
    request.timing.worker_queue_ms =
        std::chrono::duration<double, std::milli>(
            transact_start - backend_enqueued_at)
            .count();
    response.timing.worker_queue_ms = request.timing.worker_queue_ms;
    const bool request_cancelled =
        shutdown_requested_.load(std::memory_order_acquire) || stopRequested();
    if (request_cancelled) {
      response.ok = false;
      response.error = "Python inference request cancelled during shutdown";
    } else if (request.due_time !=
                   std::chrono::steady_clock::time_point::max() &&
               transact_start >= request.due_time) {
      // Admission deadlines are checked again after dequeue. A request may
      // have been timely when accepted but become obsolete behind the single
      // in-flight model transaction; it still receives a reliable terminal
      // response so completion accounting and debug-frame ownership close.
      response.ok = false;
      response.error =
          "Python inference request superseded before worker start: deadline expired";
      RunLogger::logGlobal(
          "inference",
          "request_superseded request_id=" +
              std::to_string(request.provenance.request_id) +
              " reason=deadline_before_worker_start");
    } else if (!sigpipe_protected) {
      response.ok = false;
      response.error =
          "Python inference unavailable because SIGPIPE could not be blocked";
    } else if (!config_.python_backend_enabled) {
      response.ok = false;
      response.error = "Python inference backend is disabled by config";
      RunLogger::logGlobal("inference", response.error);
    } else if (backend_disabled_after_failure_) {
      response.ok = false;
      response.error = "Python inference backend disabled after repeated IPC failures";
      RunLogger::logGlobal("inference", response.error);
    } else {
      if (!transactWithWorker(request, &response)) {
        response.time_ns = request.time_ns;
        response.provenance = request.provenance;
        response.camera_id = request.camera_id;
        response.ok = false;
        if (shutdown_requested_.load(std::memory_order_acquire) ||
            stopRequested()) {
          if (response.error.empty()) {
            response.error =
                "Python inference request cancelled during shutdown";
          }
        } else {
          ++consecutive_ipc_failures_;
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
        }
      } else {
        consecutive_ipc_failures_ = 0;
      }
    }

    response.backend_ipc_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - transact_start)
            .count());
    const auto response_forward_start = std::chrono::steady_clock::now();
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
                             std::to_string(response.boxernet_ms) +
                             " serialize_ms=" +
                             std::to_string(response.timing.serialize_ms) +
                             " pipe_write_ms=" +
                             std::to_string(response.timing.pipe_write_ms) +
                             " pipe_read_ms=" +
                             std::to_string(response.timing.pipe_read_ms) +
                             " worker_queue_ms=" +
                             std::to_string(response.timing.worker_queue_ms));
    response.timing.response_forward_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - response_forward_start)
            .count();
    enqueueResponse(std::move(response));
    processing_request_.store(false, std::memory_order_release);
    {
      std::lock_guard<std::mutex> accounting_lock(request_accounting_mutex_);
      const std::uint64_t previous =
          outstanding_requests_.fetch_sub(1, std::memory_order_acq_rel);
      if (previous == 0) {
        outstanding_requests_.store(0, std::memory_order_release);
        RunLogger::logGlobal(
            "inference", "request_accounting_underflow_prevented");
      }
    }
  }
  processing_request_.store(false, std::memory_order_release);
}

bool PythonInferenceBackend::enqueueResponse(InferenceResponse response) {
  const RequestId request_id = response.provenance.request_id;
  PushResult<InferenceResponse> result = response_queue_.push(std::move(response));
  if (!result.accepted()) {
    RunLogger::logGlobal(
        "inference",
        "reliable_response_not_enqueued request_id=" +
            std::to_string(request_id) + " outcome=" +
            std::to_string(static_cast<int>(result.outcome)));
  }
  return result.accepted();
}

bool PythonInferenceBackend::ensureWorkerProcess(std::string* error) {
  if (shutdown_requested_.load(std::memory_order_acquire) || stopRequested()) {
    *error = "inference worker start cancelled during shutdown";
    return false;
  }

  std::lock_guard<std::mutex> process_lock(worker_process_mutex_);
  if (shutdown_requested_.load(std::memory_order_acquire) || stopRequested()) {
    *error = "inference worker start cancelled during shutdown";
    return false;
  }

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
  if (shutdown_requested_.load(std::memory_order_acquire) || stopRequested()) {
    closeFd(&to_child[1]);
    closeFd(&from_child[0]);
    terminateProcessBounded(pid);
    *error = "inference worker start cancelled during shutdown";
    return false;
  }
  worker_pid_.store(static_cast<int>(pid));
  worker_stdin_fd_.store(to_child[1]);
  worker_stdout_fd_.store(from_child[0]);
  RunLogger::logGlobal("inference",
                       "spawned_worker pid=" + std::to_string(static_cast<long>(pid)));
  return true;
}

void PythonInferenceBackend::stopWorkerProcess() {
  std::lock_guard<std::mutex> process_lock(worker_process_mutex_);
  const int stdin_fd = worker_stdin_fd_.exchange(-1);
  const int stdout_fd = worker_stdout_fd_.exchange(-1);
  const int pid = worker_pid_.exchange(-1);
  if (pid > 0) {
    RunLogger::logGlobal("inference",
                         "stopping_worker pid=" + std::to_string(pid));
    // Let peer closure unblock a concurrent pipe read/write before closing the
    // local descriptors. This avoids an fd-number reuse race across threads.
    terminateProcessBounded(static_cast<pid_t>(pid));
  }
  if (stdin_fd >= 0) {
    ::close(stdin_fd);
  }
  if (stdout_fd >= 0) {
    ::close(stdout_fd);
  }
}

bool PythonInferenceBackend::transactWithWorker(const InferenceRequest& request,
                                                InferenceResponse* response) {
  std::string error;
  if (shutdown_requested_.load(std::memory_order_acquire) || stopRequested()) {
    response->error = "inference transaction cancelled during shutdown";
    return false;
  }
  if (!ensureWorkerProcess(&error)) {
    response->error = error;
    return false;
  }
  if (shutdown_requested_.load(std::memory_order_acquire) || stopRequested()) {
    response->error = "inference transaction cancelled during shutdown";
    return false;
  }

  std::vector<std::uint8_t> request_body;
  const auto serialize_start = std::chrono::steady_clock::now();
  if (!inference_wire::encodeRequest(request, &request_body, &error)) {
    response->error = error;
    return false;
  }
  response->timing.serialize_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - serialize_start)
          .count();

  const auto write_start = std::chrono::steady_clock::now();
  if (!writeMessage(request_body, &error)) {
    response->timing.pipe_write_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - write_start)
            .count();
    response->error = error;
    return false;
  }
  response->timing.pipe_write_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - write_start)
          .count();

  std::vector<std::uint8_t> response_body;
  const int stdout_fd = worker_stdout_fd_.load();
  if (stdout_fd < 0) {
    response->error = "inference response pipe is closed";
    return false;
  }
  if (!waitForWorkerReadable(stdout_fd, &error)) {
    response->error = error;
    return false;
  }
  const auto read_start = std::chrono::steady_clock::now();
  if (!readMessage(&response_body, &error)) {
    response->timing.pipe_read_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - read_start)
            .count();
    response->error = error;
    return false;
  }
  response->timing.pipe_read_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - read_start)
          .count();

  InferenceResponse decoded;
  if (!inference_wire::decodeResponse(response_body, &decoded, &error)) {
    response->ok = false;
    response->error = error;
    return false;
  }
  if (!(decoded.provenance == request.provenance)) {
    response->ok = false;
    response->error = "Python worker response provenance does not match request";
    return false;
  }
  decoded.timing.serialize_ms = response->timing.serialize_ms;
  decoded.timing.pipe_write_ms = response->timing.pipe_write_ms;
  decoded.timing.pipe_read_ms = response->timing.pipe_read_ms;
  *response = std::move(decoded);
  return true;
}

bool PythonInferenceBackend::waitForWorkerReadable(int fd,
                                                   std::string* error) const {
  while (true) {
    if (shutdown_requested_.load(std::memory_order_acquire) ||
        stopRequested()) {
      *error = "waiting for inference response cancelled during shutdown";
      return false;
    }

    pollfd descriptor{};
    descriptor.fd = fd;
    descriptor.events = POLLIN;
    const int poll_result =
        ::poll(&descriptor, 1, static_cast<int>(kWorkerPollPeriod.count()));
    if (poll_result > 0) {
      // A pipe can report POLLIN and POLLHUP together when its final buffered
      // bytes are readable. Drain those bytes before treating HUP as failure.
      if ((descriptor.revents & POLLIN) != 0) {
        return true;
      }
      if ((descriptor.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
        *error = "inference response pipe closed before a response arrived";
        return false;
      }
      continue;
    }
    if (poll_result == 0) {
      continue;
    }
    const int poll_error = errno;
    if (poll_error == EINTR) {
      continue;
    }
    *error = formatIoError("poll inference response", poll_error);
    return false;
  }
}

bool PythonInferenceBackend::writeMessage(
    const std::vector<std::uint8_t>& body,
    std::string* error) {
  if (body.size() > inference_wire::kMaxMessageBytes) {
    *error = "inference request body exceeds the IPC size limit";
    return false;
  }
  const std::uint32_t size = static_cast<std::uint32_t>(body.size());
  const std::uint8_t size_bytes[4] = {
      static_cast<std::uint8_t>(size & 0xffU),
      static_cast<std::uint8_t>((size >> 8U) & 0xffU),
      static_cast<std::uint8_t>((size >> 16U) & 0xffU),
      static_cast<std::uint8_t>((size >> 24U) & 0xffU),
  };
  const int fd = worker_stdin_fd_.load();
  if (fd < 0) {
    *error = "inference request pipe is closed";
    return false;
  }
  int error_number = 0;
  if (!writeExact(fd, size_bytes, sizeof(size_bytes), &error_number) ||
      !writeExact(fd, body.data(), body.size(), &error_number)) {
    *error = formatIoError("write inference request", error_number);
    return false;
  }
  error->clear();
  return true;
}

bool PythonInferenceBackend::readMessage(std::vector<std::uint8_t>* body,
                                         std::string* error) {
  std::uint8_t size_bytes[4] = {};
  const int fd = worker_stdout_fd_.load();
  if (fd < 0) {
    *error = "inference response pipe is closed";
    return false;
  }
  int error_number = 0;
  if (!readExact(fd, size_bytes, sizeof(size_bytes), &error_number)) {
    *error = formatIoError("read inference response header", error_number);
    return false;
  }
  const std::uint32_t size =
      static_cast<std::uint32_t>(size_bytes[0]) |
      (static_cast<std::uint32_t>(size_bytes[1]) << 8U) |
      (static_cast<std::uint32_t>(size_bytes[2]) << 16U) |
      (static_cast<std::uint32_t>(size_bytes[3]) << 24U);
  if (size > inference_wire::kMaxMessageBytes) {
    *error = "inference response body exceeds the IPC size limit";
    return false;
  }
  body->assign(size, 0);
  if (!readExact(fd, body->data(), body->size(), &error_number)) {
    *error = formatIoError("read inference response body", error_number);
    return false;
  }
  error->clear();
  return true;
}

bool PythonInferenceBackend::writeExact(int fd,
                                        const void* data,
                                        std::size_t size,
                                        int* error_number) {
  *error_number = 0;
  const auto* ptr = static_cast<const std::uint8_t*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    const ssize_t written = ::write(fd, ptr + offset, size - offset);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      *error_number = errno;
      return false;
    }
    if (written == 0) {
      return false;
    }
    offset += static_cast<std::size_t>(written);
  }
  return true;
}

bool PythonInferenceBackend::readExact(int fd,
                                       void* data,
                                       std::size_t size,
                                       int* error_number) {
  *error_number = 0;
  auto* ptr = static_cast<std::uint8_t*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    const ssize_t n_read = ::read(fd, ptr + offset, size - offset);
    if (n_read < 0) {
      if (errno == EINTR) {
        continue;
      }
      *error_number = errno;
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
  beginShutdown();
  response_queue_.stop();
}

}  // namespace roomie
