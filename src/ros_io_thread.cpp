#include "roomie/pipeline/ros_io_thread.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <utility>

#include <geometry_msgs/msg/transform_stamped.hpp>

#include "roomie/utils/run_logger.hpp"

namespace roomie {
namespace {

TimeNanoseconds toNanoseconds(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<TimeNanoseconds>(stamp.sec) * 1000000000LL +
         static_cast<TimeNanoseconds>(stamp.nanosec);
}

TimeNanoseconds secondsToNanoseconds(double seconds) {
  return static_cast<TimeNanoseconds>(seconds * 1000000000.0);
}

TimeNanoseconds absoluteDelta(TimeNanoseconds lhs, TimeNanoseconds rhs) {
  return lhs > rhs ? lhs - rhs : rhs - lhs;
}

bool hasUsableIntrinsics(const CameraIntrinsics& intrinsics) {
  return intrinsics.width > 0 && intrinsics.height > 0 &&
         intrinsics.fx > 0.0f && intrinsics.fy > 0.0f;
}

std::string normalizeFrameId(std::string frame_id) {
  while (!frame_id.empty() && frame_id.front() == '/') {
    frame_id.erase(frame_id.begin());
  }
  return frame_id;
}

int channelsForEncoding(const std::string& encoding) {
  if (encoding == "rgb8" || encoding == "bgr8" || encoding == "rgba8" || encoding == "bgra8") {
    return 3;
  }
  if (encoding == "mono8" || encoding == "8UC1") {
    return 1;
  }
  return 0;
}

int sourceChannelsForEncoding(const std::string& encoding) {
  if (encoding == "rgb8" || encoding == "bgr8") {
    return 3;
  }
  if (encoding == "rgba8" || encoding == "bgra8") {
    return 4;
  }
  if (encoding == "mono8" || encoding == "8UC1") {
    return 1;
  }
  return 0;
}

void copyPixelAsCanonical(const std::uint8_t* src,
                          const std::string& encoding,
                          std::uint8_t* dst) {
  if (encoding == "rgb8") {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    return;
  }
  if (encoding == "bgr8") {
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = src[0];
    return;
  }
  if (encoding == "rgba8") {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    return;
  }
  if (encoding == "bgra8") {
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = src[0];
    return;
  }
  dst[0] = src[0];
}

std::optional<ImageBuffer> imageBufferFromRos(const sensor_msgs::msg::Image& msg) {
  const int channels = channelsForEncoding(msg.encoding);
  const int source_channels = sourceChannelsForEncoding(msg.encoding);
  if (channels <= 0 || source_channels <= 0 || msg.height == 0 || msg.width == 0) {
    return std::nullopt;
  }

  const std::size_t compact_size =
      static_cast<std::size_t>(msg.height) * static_cast<std::size_t>(msg.width) *
      static_cast<std::size_t>(channels);
  ImageBuffer buffer;
  buffer.width = static_cast<int>(msg.width);
  buffer.height = static_cast<int>(msg.height);
  buffer.channels = channels;
  buffer.encoding = channels == 3 ? "rgb8" : "mono8";
  buffer.data.resize(compact_size);

  const std::size_t source_row_bytes =
      static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(source_channels);
  if (msg.step < source_row_bytes ||
      msg.data.size() < msg.step * static_cast<std::size_t>(msg.height)) {
    return std::nullopt;
  }

  for (std::uint32_t y = 0; y < msg.height; ++y) {
    const std::uint8_t* src_row = msg.data.data() + static_cast<std::size_t>(y) * msg.step;
    std::uint8_t* dst_row =
        buffer.data.data() +
        static_cast<std::size_t>(y) * static_cast<std::size_t>(msg.width) *
            static_cast<std::size_t>(channels);
    for (std::uint32_t x = 0; x < msg.width; ++x) {
      copyPixelAsCanonical(src_row + static_cast<std::size_t>(x) *
                                         static_cast<std::size_t>(source_channels),
                           msg.encoding,
                           dst_row + static_cast<std::size_t>(x) *
                                         static_cast<std::size_t>(channels));
    }
  }
  return buffer;
}

float filteredDepth(float depth_m, float depth_min_m, float depth_max_m) {
  if (!std::isfinite(depth_m) || depth_m < depth_min_m || depth_m > depth_max_m) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return depth_m;
}

std::optional<DepthBuffer> depthBufferFromRos(const sensor_msgs::msg::Image& msg,
                                              float depth_scale,
                                              float depth_min_m,
                                              float depth_max_m) {
  if (msg.height == 0 || msg.width == 0) {
    return std::nullopt;
  }

  const std::size_t pixel_count =
      static_cast<std::size_t>(msg.height) * static_cast<std::size_t>(msg.width);
  DepthBuffer buffer;
  buffer.width = static_cast<int>(msg.width);
  buffer.height = static_cast<int>(msg.height);
  buffer.depth_m.resize(pixel_count);

  if (msg.encoding == "32FC1") {
    const std::size_t row_bytes = static_cast<std::size_t>(msg.width) * sizeof(float);
    if (msg.step < row_bytes || msg.data.size() < msg.step * static_cast<std::size_t>(msg.height)) {
      return std::nullopt;
    }
    for (std::uint32_t y = 0; y < msg.height; ++y) {
      std::memcpy(buffer.depth_m.data() + static_cast<std::size_t>(y) * msg.width,
                  msg.data.data() + static_cast<std::size_t>(y) * msg.step,
                  row_bytes);
      for (std::uint32_t x = 0; x < msg.width; ++x) {
        float& value = buffer.depth_m[static_cast<std::size_t>(y) * msg.width + x];
        value = filteredDepth(value, depth_min_m, depth_max_m);
      }
    }
    return buffer;
  }

  if (msg.encoding == "16UC1") {
    const std::size_t row_bytes = static_cast<std::size_t>(msg.width) * sizeof(std::uint16_t);
    if (msg.step < row_bytes || msg.data.size() < msg.step * static_cast<std::size_t>(msg.height)) {
      return std::nullopt;
    }
    for (std::uint32_t y = 0; y < msg.height; ++y) {
      const auto* row = reinterpret_cast<const std::uint16_t*>(
          msg.data.data() + static_cast<std::size_t>(y) * msg.step);
      for (std::uint32_t x = 0; x < msg.width; ++x) {
        const float depth_m = static_cast<float>(row[x]) * depth_scale;
        buffer.depth_m[static_cast<std::size_t>(y) * msg.width + x] =
            filteredDepth(depth_m, depth_min_m, depth_max_m);
      }
    }
    return buffer;
  }

  return std::nullopt;
}

CameraIntrinsics intrinsicsFromRos(const sensor_msgs::msg::CameraInfo& msg) {
  CameraIntrinsics intrinsics;
  intrinsics.width = static_cast<int>(msg.width);
  intrinsics.height = static_cast<int>(msg.height);
  intrinsics.fx = static_cast<float>(msg.k[0]);
  intrinsics.fy = static_cast<float>(msg.k[4]);
  intrinsics.cx = static_cast<float>(msg.k[2]);
  intrinsics.cy = static_cast<float>(msg.k[5]);
  return intrinsics;
}

std::optional<Eigen::Isometry3f> transformFromRos(
    const geometry_msgs::msg::TransformStamped& msg) {
  const auto& rotation = msg.transform.rotation;
  Eigen::Quaternionf quaternion(static_cast<float>(rotation.w),
                                static_cast<float>(rotation.x),
                                static_cast<float>(rotation.y),
                                static_cast<float>(rotation.z));
  const float norm = quaternion.norm();
  if (!std::isfinite(norm) || norm <= 0.0f) {
    return std::nullopt;
  }
  quaternion.normalize();

  Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();
  transform.linear() = quaternion.toRotationMatrix();
  transform.translation() = Eigen::Vector3f(
      static_cast<float>(msg.transform.translation.x),
      static_cast<float>(msg.transform.translation.y),
      static_cast<float>(msg.transform.translation.z));
  return transform;
}

bool stampCloseEnough(TimeNanoseconds lhs, TimeNanoseconds rhs, TimeNanoseconds max_delta_ns) {
  if (lhs == 0 || rhs == 0 || max_delta_ns <= 0) {
    return true;
  }
  return absoluteDelta(lhs, rhs) <= max_delta_ns;
}

}  // namespace

RosIoThread::RosIoThread(ThreadSafeQueue<MappingFrame>& mapping_queue,
                         ThreadSafeQueue<DetectionFrame>& detection_queue)
    : WorkerThread("ros_io_thread"),
      mapping_queue_(mapping_queue),
      detection_queue_(detection_queue),
      logger_(rclcpp::get_logger("roomie.ros_io_thread")) {}

void RosIoThread::configure(RosIoSubscriptionConfig config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = std::move(config);
  camera_states_.clear();
  for (const auto& camera : config_.cameras) {
    if (!camera.camera_id.empty()) {
      CameraState state;
      state.config = camera;
      if (hasUsableIntrinsics(camera.fallback_intrinsics)) {
        state.latest_intrinsics = camera.fallback_intrinsics;
      }
      camera_states_.emplace(camera.camera_id, std::move(state));
    }
  }
}

void RosIoThread::attachNode(rclcpp::Node& node) {
  std::lock_guard<std::mutex> lock(mutex_);
  subscriptions_.clear();
  tf_subscription_.reset();
  tf_static_subscription_.reset();
  const auto input_qos =
      rclcpp::QoS(static_cast<std::size_t>(std::max<std::size_t>(1, config_.input_queue_size)))
          .reliable();

  if (!config_.tf_topic.empty()) {
    tf_subscription_ = node.create_subscription<tf2_msgs::msg::TFMessage>(
        config_.tf_topic,
        input_qos,
        [this](tf2_msgs::msg::TFMessage::SharedPtr msg) {
          handleTf(std::move(msg), false);
        });
  }
  if (!config_.tf_static_topic.empty()) {
    tf_static_subscription_ = node.create_subscription<tf2_msgs::msg::TFMessage>(
        config_.tf_static_topic,
        rclcpp::QoS(100).transient_local().reliable(),
        [this](tf2_msgs::msg::TFMessage::SharedPtr msg) {
          handleTf(std::move(msg), true);
        });
  }

  for (const auto& camera : config_.cameras) {
    if (camera.camera_id.empty()) {
      RCLCPP_WARN(logger_, "skipping camera subscription with empty camera_id");
      continue;
    }

    auto& state = camera_states_[camera.camera_id];
    state.config = camera;
    if (hasUsableIntrinsics(camera.fallback_intrinsics)) {
      state.latest_intrinsics = camera.fallback_intrinsics;
    }

    CameraSubscriptions subscriptions;
    if (!camera.rgb_topic.empty()) {
      subscriptions.rgb = node.create_subscription<sensor_msgs::msg::Image>(
          camera.rgb_topic,
          input_qos,
          [this, camera_id = camera.camera_id](sensor_msgs::msg::Image::SharedPtr msg) {
            handleRgb(camera_id, std::move(msg));
          });
    }
    if (!camera.robot_mask_topic.empty()) {
      subscriptions.robot_mask = node.create_subscription<sensor_msgs::msg::Image>(
          camera.robot_mask_topic,
          input_qos,
          [this, camera_id = camera.camera_id](sensor_msgs::msg::Image::SharedPtr msg) {
            handleRobotMask(camera_id, std::move(msg));
          });
    }
    if (!camera.depth_topic.empty()) {
      subscriptions.depth = node.create_subscription<sensor_msgs::msg::Image>(
          camera.depth_topic,
          input_qos,
          [this, camera_id = camera.camera_id](sensor_msgs::msg::Image::SharedPtr msg) {
            handleDepth(camera_id, std::move(msg));
          });
    }
    if (!camera.camera_info_topic.empty()) {
      subscriptions.camera_info = node.create_subscription<sensor_msgs::msg::CameraInfo>(
          camera.camera_info_topic,
          input_qos,
          [this, camera_id = camera.camera_id](sensor_msgs::msg::CameraInfo::SharedPtr msg) {
            handleCameraInfo(camera_id, std::move(msg));
          });
    }
    subscriptions_.push_back(std::move(subscriptions));
  }

  RCLCPP_INFO(logger_, "attached ROS IO subscriptions for %zu camera(s)", subscriptions_.size());
}

void RosIoThread::setCameraPose(std::string camera_id,
                                TimeNanoseconds time_ns,
                                Eigen::Isometry3f T_world_camera) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& state = camera_states_[camera_id];
  state.config.camera_id = std::move(camera_id);
  state.latest_pose_time_ns = time_ns;
  state.latest_T_world_camera = std::move(T_world_camera);
}

bool RosIoThread::enqueueMappingFrame(MappingFrame frame) {
  return mapping_queue_.pushDropOldest(std::move(frame));
}

bool RosIoThread::enqueueDetectionFrame(DetectionFrame frame) {
  return detection_queue_.pushDropOldest(std::move(frame));
}

void RosIoThread::handleRgb(const std::string& camera_id,
                            const sensor_msgs::msg::Image::SharedPtr msg) {
  auto rgb = imageBufferFromRos(*msg);
  if (!rgb) {
    RCLCPP_WARN(logger_,
                "dropping RGB frame for %s with unsupported encoding '%s'",
                camera_id.c_str(), msg->encoding.c_str());
    return;
  }

  std::optional<MappingFrame> mapping_frame;
  std::optional<DetectionFrame> detection_frame;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = camera_states_[camera_id];
    state.config.camera_id = camera_id;
    ++rgb_messages_;
    state.latest_rgb_time_ns = toNanoseconds(msg->header.stamp);
    state.latest_rgb = std::move(rgb);
    auto frames = makeFramesLocked(camera_id, state.latest_rgb_time_ns);
    mapping_frame = std::move(frames.first);
    detection_frame = std::move(frames.second);
    maybeLogStatusLocked();
  }

  if (mapping_frame) {
    enqueueMappingFrame(std::move(*mapping_frame));
  }
  if (detection_frame) {
    enqueueDetectionFrame(std::move(*detection_frame));
  }
}

void RosIoThread::handleRobotMask(const std::string& camera_id,
                                  const sensor_msgs::msg::Image::SharedPtr msg) {
  auto mask = imageBufferFromRos(*msg);
  if (!mask) {
    RCLCPP_WARN(logger_,
                "dropping robot mask for %s with unsupported encoding '%s'",
                camera_id.c_str(), msg->encoding.c_str());
    return;
  }

  std::optional<MappingFrame> mapping_frame;
  std::optional<DetectionFrame> detection_frame;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = camera_states_[camera_id];
    state.config.camera_id = camera_id;
    ++mask_messages_;
    state.latest_robot_mask_time_ns = toNanoseconds(msg->header.stamp);
    state.latest_robot_mask = std::move(mask);
    auto frames = makeFramesLocked(camera_id, state.latest_rgb_time_ns);
    mapping_frame = std::move(frames.first);
    detection_frame = std::move(frames.second);
    maybeLogStatusLocked();
  }

  if (mapping_frame) {
    enqueueMappingFrame(std::move(*mapping_frame));
  }
  if (detection_frame) {
    enqueueDetectionFrame(std::move(*detection_frame));
  }
}

void RosIoThread::handleDepth(const std::string& camera_id,
                              const sensor_msgs::msg::Image::SharedPtr msg) {
  std::optional<MappingFrame> mapping_frame;
  std::optional<DetectionFrame> detection_frame;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = camera_states_[camera_id];
    state.config.camera_id = camera_id;
    ++depth_messages_;
    auto depth = depthBufferFromRos(*msg,
                                    state.config.depth_scale,
                                    state.config.depth_min_m,
                                    state.config.depth_max_m);
    if (!depth) {
      RCLCPP_WARN(logger_,
                  "dropping depth frame for %s with unsupported encoding '%s'",
                  camera_id.c_str(), msg->encoding.c_str());
      return;
    }
    state.latest_depth_time_ns = toNanoseconds(msg->header.stamp);
    state.latest_depth = std::move(depth);
    auto frames = makeFramesLocked(camera_id, state.latest_rgb_time_ns);
    mapping_frame = std::move(frames.first);
    detection_frame = std::move(frames.second);
    maybeLogStatusLocked();
  }

  if (mapping_frame) {
    enqueueMappingFrame(std::move(*mapping_frame));
  }
  if (detection_frame) {
    enqueueDetectionFrame(std::move(*detection_frame));
  }
}

void RosIoThread::handleCameraInfo(const std::string& camera_id,
                                   const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& state = camera_states_[camera_id];
  state.config.camera_id = camera_id;
  const CameraIntrinsics intrinsics = intrinsicsFromRos(*msg);
  if (hasUsableIntrinsics(intrinsics)) {
    ++camera_info_messages_;
    state.latest_intrinsics = intrinsics;
  } else if (!state.latest_intrinsics && hasUsableIntrinsics(state.config.fallback_intrinsics)) {
    state.latest_intrinsics = state.config.fallback_intrinsics;
  }
  maybeLogStatusLocked();
}

void RosIoThread::handleTf(const tf2_msgs::msg::TFMessage::SharedPtr msg, bool is_static) {
  std::vector<MappingFrame> mapping_frames;
  std::vector<DetectionFrame> detection_frames;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_static) {
      ++tf_static_messages_;
    } else {
      ++tf_messages_;
    }
    for (const auto& transform_msg : msg->transforms) {
      const std::string parent_frame = normalizeFrameId(transform_msg.header.frame_id);
      const std::string child_frame = normalizeFrameId(transform_msg.child_frame_id);
      if (parent_frame.empty() || child_frame.empty()) {
        continue;
      }

      auto transform = transformFromRos(transform_msg);
      if (!transform) {
        RCLCPP_WARN(logger_,
                    "skipping TF %s -> %s with invalid rotation",
                    parent_frame.c_str(),
                    child_frame.c_str());
        continue;
      }

      StoredTransform stored;
      stored.parent_frame = parent_frame;
      stored.T_parent_child = std::move(*transform);
      stored.time_ns = toNanoseconds(transform_msg.header.stamp);
      stored.is_static = is_static;
      transforms_by_child_frame_[child_frame] = std::move(stored);
    }

    for (auto& [camera_id, state] : camera_states_) {
      if (state.latest_rgb_time_ns == 0) {
        continue;
      }
      auto frames = makeFramesLocked(camera_id, state.latest_rgb_time_ns);
      if (frames.first) {
        mapping_frames.push_back(std::move(*frames.first));
      }
      if (frames.second) {
        detection_frames.push_back(std::move(*frames.second));
      }
    }
    maybeLogStatusLocked();
  }

  for (MappingFrame& frame : mapping_frames) {
    enqueueMappingFrame(std::move(frame));
  }
  for (DetectionFrame& frame : detection_frames) {
    enqueueDetectionFrame(std::move(frame));
  }
}

std::optional<Eigen::Isometry3f> RosIoThread::lookupTWorldFrameLocked(
    const std::string& target_frame,
    TimeNanoseconds time_ns) const {
  const std::string world_frame = normalizeFrameId(config_.world_frame);
  std::string current_frame = normalizeFrameId(target_frame);
  if (world_frame.empty() || current_frame.empty()) {
    return std::nullopt;
  }
  if (current_frame == world_frame) {
    return Eigen::Isometry3f::Identity();
  }

  const TimeNanoseconds max_gap_ns = secondsToNanoseconds(config_.max_tf_gap_sec);
  Eigen::Isometry3f T_world_current = Eigen::Isometry3f::Identity();
  std::unordered_set<std::string> visited;

  while (current_frame != world_frame) {
    if (!visited.insert(current_frame).second) {
      return std::nullopt;
    }

    const auto it = transforms_by_child_frame_.find(current_frame);
    if (it == transforms_by_child_frame_.end()) {
      return std::nullopt;
    }
    const StoredTransform& stored = it->second;
    if (!stored.is_static && max_gap_ns > 0 && time_ns > 0 && stored.time_ns > 0 &&
        absoluteDelta(time_ns, stored.time_ns) > max_gap_ns) {
      return std::nullopt;
    }

    T_world_current = stored.T_parent_child * T_world_current;
    current_frame = stored.parent_frame;
  }

  return T_world_current;
}

std::pair<std::optional<MappingFrame>, std::optional<DetectionFrame>>
RosIoThread::makeFramesLocked(const std::string& camera_id, TimeNanoseconds time_ns) {
  auto it = camera_states_.find(camera_id);
  if (it == camera_states_.end()) {
    return {};
  }

  CameraState& state = it->second;
  if (time_ns == 0) {
    return {};
  }
  if (!state.latest_rgb || !state.latest_robot_mask || !state.latest_intrinsics ||
      state.config.camera_frame.empty()) {
    return {};
  }

  auto pose = lookupTWorldFrameLocked(state.config.camera_frame, time_ns);
  if (!pose) {
    return {};
  }
  state.latest_T_world_camera = std::move(*pose);
  state.latest_pose_time_ns = time_ns;

  const TimeNanoseconds max_image_delta_ns =
      secondsToNanoseconds(config_.max_image_stamp_delta_sec);
  if (!stampCloseEnough(time_ns, state.latest_robot_mask_time_ns, max_image_delta_ns)) {
    return {};
  }

  std::optional<DetectionFrame> detection_frame;
  if (state.config.enable_detection &&
      state.last_emitted_detection_time_ns != time_ns) {
    DetectionFrame frame;
    frame.time_ns = time_ns;
    frame.camera_id = camera_id;
    frame.rgb = *state.latest_rgb;
    frame.robot_mask = *state.latest_robot_mask;
    frame.intrinsics = *state.latest_intrinsics;
    frame.T_world_camera = *state.latest_T_world_camera;
    detection_frame = std::move(frame);
    state.last_emitted_detection_time_ns = time_ns;
    ++detection_frames_emitted_;
  }

  std::optional<MappingFrame> mapping_frame;
  if (state.config.enable_mapping && state.latest_depth &&
      state.last_emitted_mapping_time_ns != time_ns) {
    if (!stampCloseEnough(time_ns, state.latest_depth_time_ns, max_image_delta_ns)) {
      return {std::move(mapping_frame), std::move(detection_frame)};
    }
    MappingFrame frame;
    frame.time_ns = time_ns;
    frame.camera_id = camera_id;
    frame.rgb = *state.latest_rgb;
    frame.robot_mask = *state.latest_robot_mask;
    frame.depth = *state.latest_depth;
    frame.intrinsics = *state.latest_intrinsics;
    frame.T_world_camera = *state.latest_T_world_camera;
    mapping_frame = std::move(frame);
    state.last_emitted_mapping_time_ns = time_ns;
    ++mapping_frames_emitted_;
  }

  return {std::move(mapping_frame), std::move(detection_frame)};
}

void RosIoThread::maybeLogStatusLocked() {
  const auto now = std::chrono::steady_clock::now();
  if (now - last_status_log_time_ <
      std::chrono::duration<double>(config_.log_period_sec)) {
    return;
  }
  last_status_log_time_ = now;

  std::ostringstream stream;
  stream << "status rgb=" << rgb_messages_
         << " mask=" << mask_messages_
         << " depth=" << depth_messages_
         << " camera_info=" << camera_info_messages_
         << " tf=" << tf_messages_
         << " tf_static=" << tf_static_messages_
         << " mapping_frames=" << mapping_frames_emitted_
         << " detection_frames=" << detection_frames_emitted_
         << " cameras=" << camera_states_.size()
         << " tf_edges=" << transforms_by_child_frame_.size();
  RunLogger::logGlobal("ros_io", stream.str());
}

void RosIoThread::run() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stopRequested()) {
    cv_.wait_for(lock, std::chrono::seconds(1), [this]() { return stopRequested(); });
  }
}

void RosIoThread::onStopRequested() { cv_.notify_all(); }

}  // namespace roomie
