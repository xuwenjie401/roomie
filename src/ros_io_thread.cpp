#include "roomie/pipeline/ros_io_thread.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>

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

const char* mapModeName(MapMode mode) {
  return mode == MapMode::kFrozen ? "frozen" : "online";
}

}  // namespace

RosIoThread::RosIoThread(ThreadSafeQueue<FrameBundlePtr>& mapping_queue,
                         ThreadSafeQueue<FrameBundlePtr>& detection_queue)
    : WorkerThread("ros_io_thread"),
      mapping_queue_(mapping_queue),
      detection_queue_(detection_queue),
      logger_(rclcpp::get_logger("roomie.ros_io_thread")) {}

void RosIoThread::configure(RosIoSubscriptionConfig config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = std::move(config);
  run_id_ = makeRunId();
  next_frame_id_ = 1;
  camera_states_.clear();
  for (const auto& camera : config_.cameras) {
    if (!camera.camera_id.empty()) {
      CameraState state;
      state.config = camera;
      if (hasUsableIntrinsics(camera.fallback_intrinsics)) {
        state.latest_intrinsics = camera.fallback_intrinsics;
        state.calibration_revision = 1;
      }
      camera_states_.emplace(camera.camera_id, std::move(state));
    }
  }
  RCLCPP_INFO(logger_,
              "configured ROS IO run_id=%s map_mode=%s cameras=%zu",
              runIdString(run_id_).c_str(),
              mapModeName(config_.map_mode),
              camera_states_.size());
  RunLogger::logGlobal("ros_io",
                       "configured run_id=" + runIdString(run_id_) +
                           " map_mode=" + mapModeName(config_.map_mode) +
                           " cameras=" + std::to_string(camera_states_.size()));
}

void RosIoThread::attachNode(rclcpp::Node& node) {
  std::lock_guard<std::mutex> lock(mutex_);
  subscriptions_.clear();
  tf_subscription_.reset();
  tf_static_subscription_.reset();
  tf_buffer_ = std::make_shared<tf2::BufferCore>(
      tf2::durationFromSec(config_.tf_buffer_duration_sec));
  const auto sensor_qos =
      rclcpp::QoS(static_cast<std::size_t>(std::max<std::size_t>(1, config_.input_queue_size)))
          .best_effort()
          .durability_volatile();
  const auto tf_qos =
      rclcpp::QoS(static_cast<std::size_t>(std::max<std::size_t>(1, config_.input_queue_size)))
          .reliable()
          .durability_volatile();

  if (!config_.tf_topic.empty()) {
    tf_subscription_ = node.create_subscription<tf2_msgs::msg::TFMessage>(
        config_.tf_topic,
        tf_qos,
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
          sensor_qos,
          [this, camera_id = camera.camera_id](sensor_msgs::msg::Image::SharedPtr msg) {
            handleRgb(camera_id, std::move(msg));
          });
    }
    if (!camera.robot_mask_topic.empty()) {
      subscriptions.robot_mask = node.create_subscription<sensor_msgs::msg::Image>(
          camera.robot_mask_topic,
          sensor_qos,
          [this, camera_id = camera.camera_id](sensor_msgs::msg::Image::SharedPtr msg) {
            handleRobotMask(camera_id, std::move(msg));
          });
    }
    if (!camera.depth_topic.empty()) {
      subscriptions.depth = node.create_subscription<sensor_msgs::msg::Image>(
          camera.depth_topic,
          sensor_qos,
          [this, camera_id = camera.camera_id](sensor_msgs::msg::Image::SharedPtr msg) {
            handleDepth(camera_id, std::move(msg));
          });
    }
    if (!camera.camera_info_topic.empty()) {
      subscriptions.camera_info = node.create_subscription<sensor_msgs::msg::CameraInfo>(
          camera.camera_info_topic,
          sensor_qos,
          [this, camera_id = camera.camera_id](sensor_msgs::msg::CameraInfo::SharedPtr msg) {
            handleCameraInfo(camera_id, std::move(msg));
          });
    }
    subscriptions_.push_back(std::move(subscriptions));
  }

  RCLCPP_INFO(logger_,
              "attached ROS IO subscriptions for %zu camera(s), run_id=%s",
              subscriptions_.size(),
              runIdString(run_id_).c_str());
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

bool RosIoThread::enqueueMappingBundle(FrameBundlePtr frame) {
  if (!frame) {
    return false;
  }
  const FrameId frame_id = frame->provenance.frame_id;
  const bool reliable_candidate =
      frame->perception_candidate && frame->provenance.map_mode == MapMode::kOnline;
  PushResult<FrameBundlePtr> result;
  if (reliable_candidate) {
    const auto remaining = frame->due_time - std::chrono::steady_clock::now();
    result = mapping_queue_.pushBarrierFor(
        std::move(frame), /*max_replaceable_ahead=*/1U, remaining);
  } else {
    result = mapping_queue_.push(std::move(frame));
  }
  if (result.replaced_item) {
    const FrameBundlePtr& replaced = *result.replaced_item;
    RunLogger::logGlobal(
        "ros_io",
        "mapping_replaced frame_id=" +
            std::to_string(replaced ? replaced->provenance.frame_id : 0) +
            " by_frame_id=" + std::to_string(frame_id) + " reason=capacity");
  } else if (!result.accepted()) {
    RunLogger::logGlobal("ros_io",
                         "mapping_rejected frame_id=" +
                             std::to_string(frame_id));
  }
  return result.accepted();
}

bool RosIoThread::enqueueDetectionBundle(FrameBundlePtr frame) {
  if (!frame) {
    return false;
  }
  const FrameBundlePtr incoming = frame;
  const FrameId frame_id = frame->provenance.frame_id;
  PushResult<FrameBundlePtr> result = detection_queue_.push(std::move(frame));
  if (result.replaced_item) {
    const FrameBundlePtr& replaced = *result.replaced_item;
    RunLogger::logGlobal(
        "ros_io",
        "perception_superseded frame_id=" +
            std::to_string(replaced ? replaced->provenance.frame_id : 0) +
            " by_frame_id=" + std::to_string(frame_id) +
            " camera=" + (replaced ? replaced->camera_id : std::string{}));
    if (replaced && config_.perception_candidate_cancelled) {
      try {
        config_.perception_candidate_cancelled(replaced, "superseded");
      } catch (const std::exception& error) {
        RunLogger::logGlobal(
            "ros_io",
            "perception_cancel_callback_failed frame_id=" +
                std::to_string(replaced->provenance.frame_id) +
                " reason=superseded error=" + error.what());
      } catch (...) {
        RunLogger::logGlobal(
            "ros_io",
            "perception_cancel_callback_failed frame_id=" +
                std::to_string(replaced->provenance.frame_id) +
                " reason=superseded error=unknown");
      }
    }
  } else if (!result.accepted()) {
    RunLogger::logGlobal("ros_io",
                         "perception_rejected frame_id=" +
                             std::to_string(frame_id));
    if (incoming && config_.perception_candidate_cancelled) {
      try {
        config_.perception_candidate_cancelled(incoming,
                                               "detection_admission_rejected");
      } catch (...) {
        RunLogger::logGlobal(
            "ros_io",
            "perception_cancel_callback_failed frame_id=" +
                std::to_string(frame_id) +
                " reason=detection_admission_rejected");
      }
    }
  }
  return result.accepted();
}

void RosIoThread::enqueueBundles(FrameBundlePtr mapping_bundle,
                                 FrameBundlePtr detection_bundle) {
  const bool online_candidate =
      detection_bundle && detection_bundle->perception_candidate &&
      detection_bundle->provenance.map_mode == MapMode::kOnline;
  if (online_candidate) {
    // LatestByCamera replacement must cancel the displaced map barrier before
    // the new protected barrier is admitted. This keeps at most the active
    // candidate plus the latest pending candidate in the mapping lifecycle.
    const FrameBundlePtr admitted_detection = detection_bundle;
    if (!enqueueDetectionBundle(detection_bundle)) {
      return;
    }
    const bool map_admitted =
        mapping_bundle && enqueueMappingBundle(std::move(mapping_bundle));
    if (map_admitted) {
      return;
    }

    detection_queue_.cancelIf(
        [&admitted_detection](const FrameBundlePtr& queued) {
          return queued && queued.get() == admitted_detection.get();
        });
    if (config_.perception_candidate_cancelled) {
      try {
        config_.perception_candidate_cancelled(admitted_detection,
                                               "map_admission_failed");
      } catch (...) {
        RunLogger::logGlobal(
            "ros_io",
            "perception_cancel_callback_failed frame_id=" +
                std::to_string(admitted_detection->provenance.frame_id) +
                " reason=map_admission_failed");
      }
    }
    RunLogger::logGlobal(
        "ros_io",
        "perception_rejected frame_id=" +
            std::to_string(admitted_detection->provenance.frame_id) +
            " reason=map_admission_failed");
    return;
  }

  bool map_admitted = false;
  if (mapping_bundle) {
    map_admitted = enqueueMappingBundle(std::move(mapping_bundle));
  }
  if (!detection_bundle) {
    return;
  }
  const bool requires_current_map =
      detection_bundle->provenance.map_mode == MapMode::kOnline;
  if (requires_current_map && !map_admitted) {
    RunLogger::logGlobal(
        "ros_io",
        "perception_rejected frame_id=" +
            std::to_string(detection_bundle->provenance.frame_id) +
            " reason=map_admission_failed");
    return;
  }
  enqueueDetectionBundle(std::move(detection_bundle));
}

void RosIoThread::handleRgb(const std::string& camera_id,
                            const sensor_msgs::msg::Image::SharedPtr msg) {
  const auto ingest_time = std::chrono::steady_clock::now();
  auto rgb = imageBufferFromRos(*msg);
  if (!rgb) {
    RCLCPP_WARN(logger_,
                "dropping RGB frame for %s with unsupported encoding '%s'",
                camera_id.c_str(), msg->encoding.c_str());
    return;
  }

  FrameBundlePtr mapping_bundle;
  FrameBundlePtr detection_bundle;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = camera_states_[camera_id];
    state.config.camera_id = camera_id;
    ++rgb_messages_;
    const TimeNanoseconds sensor_time_ns = toNanoseconds(msg->header.stamp);
    if (state.latest_rgb_time_ns != sensor_time_ns ||
        state.latest_rgb_ingest_time == std::chrono::steady_clock::time_point::min()) {
      state.latest_rgb_ingest_time = ingest_time;
    }
    state.latest_rgb_time_ns = sensor_time_ns;
    state.latest_rgb =
        std::make_shared<const ImageBuffer>(std::move(*rgb));
    auto bundles = makeFrameBundlesLocked(camera_id, state.latest_rgb_time_ns);
    mapping_bundle = std::move(bundles.first);
    detection_bundle = std::move(bundles.second);
    maybeLogStatusLocked();
  }

  enqueueBundles(std::move(mapping_bundle), std::move(detection_bundle));
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

  FrameBundlePtr mapping_bundle;
  FrameBundlePtr detection_bundle;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = camera_states_[camera_id];
    state.config.camera_id = camera_id;
    ++mask_messages_;
    state.latest_robot_mask_time_ns = toNanoseconds(msg->header.stamp);
    state.latest_robot_mask =
        std::make_shared<const ImageBuffer>(std::move(*mask));
    auto bundles = makeFrameBundlesLocked(camera_id, state.latest_rgb_time_ns);
    mapping_bundle = std::move(bundles.first);
    detection_bundle = std::move(bundles.second);
    maybeLogStatusLocked();
  }

  enqueueBundles(std::move(mapping_bundle), std::move(detection_bundle));
}

void RosIoThread::handleDepth(const std::string& camera_id,
                              const sensor_msgs::msg::Image::SharedPtr msg) {
  FrameBundlePtr mapping_bundle;
  FrameBundlePtr detection_bundle;
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
    state.latest_depth =
        std::make_shared<const DepthBuffer>(std::move(*depth));
    auto bundles = makeFrameBundlesLocked(camera_id, state.latest_rgb_time_ns);
    mapping_bundle = std::move(bundles.first);
    detection_bundle = std::move(bundles.second);
    maybeLogStatusLocked();
  }

  enqueueBundles(std::move(mapping_bundle), std::move(detection_bundle));
}

void RosIoThread::handleCameraInfo(const std::string& camera_id,
                                   const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  FrameBundlePtr mapping_bundle;
  FrameBundlePtr detection_bundle;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = camera_states_[camera_id];
    state.config.camera_id = camera_id;
    const CameraIntrinsics intrinsics = intrinsicsFromRos(*msg);
    if (hasUsableIntrinsics(intrinsics)) {
      ++camera_info_messages_;
      state.latest_intrinsics = intrinsics;
      ++state.calibration_revision;
    } else if (!state.latest_intrinsics &&
               hasUsableIntrinsics(state.config.fallback_intrinsics)) {
      state.latest_intrinsics = state.config.fallback_intrinsics;
      ++state.calibration_revision;
    }
    auto bundles = makeFrameBundlesLocked(camera_id, state.latest_rgb_time_ns);
    mapping_bundle = std::move(bundles.first);
    detection_bundle = std::move(bundles.second);
    maybeLogStatusLocked();
  }
  enqueueBundles(std::move(mapping_bundle), std::move(detection_bundle));
}

void RosIoThread::handleTf(const tf2_msgs::msg::TFMessage::SharedPtr msg, bool is_static) {
  std::vector<std::pair<FrameBundlePtr, FrameBundlePtr>> bundles_to_enqueue;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_static) {
      ++tf_static_messages_;
    } else {
      ++tf_messages_;
    }
    for (const auto& transform_msg : msg->transforms) {
      geometry_msgs::msg::TransformStamped stored = transform_msg;
      stored.header.frame_id = normalizeFrameId(stored.header.frame_id);
      stored.child_frame_id = normalizeFrameId(stored.child_frame_id);
      if (stored.header.frame_id.empty() || stored.child_frame_id.empty()) {
        continue;
      }

      auto transform = transformFromRos(stored);
      if (!transform) {
        RCLCPP_WARN(logger_,
                    "skipping TF %s -> %s with invalid rotation",
                    stored.header.frame_id.c_str(),
                    stored.child_frame_id.c_str());
        continue;
      }

      if (!tf_buffer_) {
        ++tf_set_failures_;
        last_tf_error_ = "tf_buffer is not initialized";
        continue;
      }
      try {
        if (!tf_buffer_->setTransform(stored, "roomie_ros_io", is_static)) {
          ++tf_set_failures_;
          last_tf_error_ = "tf2 rejected transform " + stored.header.frame_id + " -> " +
                           stored.child_frame_id;
        }
      } catch (const tf2::TransformException& error) {
        ++tf_set_failures_;
        last_tf_error_ = error.what();
      } catch (const std::exception& error) {
        ++tf_set_failures_;
        last_tf_error_ = error.what();
      }
    }

    for (auto& [camera_id, state] : camera_states_) {
      if (state.latest_rgb_time_ns == 0) {
        continue;
      }
      auto bundles = makeFrameBundlesLocked(camera_id, state.latest_rgb_time_ns);
      if (bundles.first || bundles.second) {
        bundles_to_enqueue.push_back(std::move(bundles));
      }
    }
    maybeLogStatusLocked();
  }

  for (auto& bundles : bundles_to_enqueue) {
    enqueueBundles(std::move(bundles.first), std::move(bundles.second));
  }
}

std::optional<Eigen::Isometry3f> RosIoThread::lookupTWorldFrameLocked(
    const std::string& target_frame,
    TimeNanoseconds time_ns) {
  const std::string world_frame = normalizeFrameId(config_.world_frame);
  std::string current_frame = normalizeFrameId(target_frame);
  if (world_frame.empty() || current_frame.empty()) {
    ++tf_lookup_failures_;
    last_tf_error_ = "empty world or target frame";
    return std::nullopt;
  }
  if (current_frame == world_frame) {
    ++tf_lookup_successes_;
    return Eigen::Isometry3f::Identity();
  }
  if (!tf_buffer_) {
    ++tf_lookup_failures_;
    last_tf_error_ = "tf_buffer is not initialized";
    return std::nullopt;
  }

  try {
    const geometry_msgs::msg::TransformStamped transform_msg = tf_buffer_->lookupTransform(
        world_frame,
        current_frame,
        tf2::TimePoint(std::chrono::nanoseconds(time_ns)));
    auto transform = transformFromRos(transform_msg);
    if (!transform) {
      ++tf_lookup_failures_;
      last_tf_error_ = "tf2 returned invalid rotation for " + world_frame + " <- " +
                       current_frame;
      return std::nullopt;
    }
    ++tf_lookup_successes_;
    return transform;
  } catch (const tf2::TransformException& error) {
    last_tf_error_ = error.what();
  } catch (const std::exception& error) {
    last_tf_error_ = error.what();
  }

  try {
    const geometry_msgs::msg::TransformStamped latest_msg = tf_buffer_->lookupTransform(
        world_frame,
        current_frame,
        tf2::TimePointZero);
    const TimeNanoseconds latest_time_ns = toNanoseconds(latest_msg.header.stamp);
    const TimeNanoseconds max_gap_ns = secondsToNanoseconds(config_.max_tf_gap_sec);
    if (latest_time_ns > 0 && max_gap_ns > 0 &&
        !stampCloseEnough(time_ns, latest_time_ns, max_gap_ns)) {
      ++tf_lookup_stale_latest_;
      return std::nullopt;
    }
    auto transform = transformFromRos(latest_msg);
    if (!transform) {
      ++tf_lookup_failures_;
      last_tf_error_ = "tf2 returned invalid latest rotation for " + world_frame + " <- " +
                       current_frame;
      return std::nullopt;
    }
    ++tf_lookup_successes_;
    ++tf_lookup_latest_fallbacks_;
    return transform;
  } catch (const tf2::TransformException& error) {
    ++tf_lookup_failures_;
    last_tf_error_ = error.what();
  } catch (const std::exception& error) {
    ++tf_lookup_failures_;
    last_tf_error_ = error.what();
  }

  return std::nullopt;
}

std::pair<FrameBundlePtr, FrameBundlePtr> RosIoThread::makeFrameBundlesLocked(
    const std::string& camera_id, TimeNanoseconds time_ns) {
  auto it = camera_states_.find(camera_id);
  if (it == camera_states_.end()) {
    return {};
  }

  CameraState& state = it->second;
  if (time_ns == 0) {
    return {};
  }
  const bool mapping_pending =
      state.config.enable_mapping && state.last_emitted_mapping_time_ns != time_ns;
  bool detection_pending =
      state.config.enable_detection && state.last_emitted_detection_time_ns != time_ns;
  if (!mapping_pending && !detection_pending) {
    return {};
  }
  if (!state.latest_rgb || !state.latest_robot_mask || !state.latest_intrinsics ||
      state.config.camera_frame.empty()) {
    return {};
  }

  const TimeNanoseconds max_image_delta_ns =
      secondsToNanoseconds(config_.max_image_stamp_delta_sec);
  if (!stampCloseEnough(time_ns, state.latest_robot_mask_time_ns, max_image_delta_ns)) {
    return {};
  }
  const bool depth_ready =
      state.latest_depth &&
      stampCloseEnough(time_ns, state.latest_depth_time_ns, max_image_delta_ns);
  if (mapping_pending && !depth_ready) {
    // Online perception must be assembled from the exact same immutable
    // bundle that enters mapping. Wait for the matching depth instead of
    // emitting an RGB-only detection copy with a misleading shared frame id.
    if (detection_pending && config_.map_mode == MapMode::kOnline) {
      return {};
    }
    if (!detection_pending) {
      return {};
    }
  }

  bool perception_candidate = false;
  std::chrono::steady_clock::time_point candidate_admission_time =
      std::chrono::steady_clock::time_point::min();
  if (detection_pending) {
    bool admission_allowed = true;
    if (config_.perception_admission_allowed) {
      try {
        admission_allowed = config_.perception_admission_allowed();
      } catch (...) {
        admission_allowed = false;
      }
    }
    if (!admission_allowed) {
      state.last_emitted_detection_time_ns = time_ns;
      ++perception_backpressure_skips_;
      detection_pending = false;
    } else if (config_.map_mode == MapMode::kOnline && !mapping_pending) {
      // An online detection without corresponding mapping work cannot satisfy
      // the include-current contract.
      state.last_emitted_detection_time_ns = time_ns;
      ++perception_rate_skips_;
      detection_pending = false;
    } else {
      const auto now = std::chrono::steady_clock::now();
      const auto min_period =
          config_.max_perception_fps > 0.0
              ? std::chrono::duration<double>(1.0 / config_.max_perception_fps)
              : std::chrono::duration<double>::zero();
      if (state.last_perception_admission_time !=
              std::chrono::steady_clock::time_point::min() &&
          now - state.last_perception_admission_time < min_period) {
        state.last_emitted_detection_time_ns = time_ns;
        ++perception_rate_skips_;
        detection_pending = false;
      } else {
        candidate_admission_time = now;
        perception_candidate = true;
      }
    }
  }
  if (!mapping_pending && !detection_pending) {
    return {};
  }

  auto pose = lookupTWorldFrameLocked(state.config.camera_frame, time_ns);
  if (!pose) {
    return {};
  }
  state.latest_T_world_camera = std::move(*pose);
  state.latest_pose_time_ns = time_ns;

  std::chrono::steady_clock::time_point ingest_time;
  const FrameProvenance provenance =
      frameProvenanceLocked(&state, time_ns, &ingest_time);

  auto mutable_bundle = std::make_shared<FrameBundle>();
  mutable_bundle->provenance = provenance;
  mutable_bundle->ingest_time = ingest_time;
  mutable_bundle->due_time =
      ingest_time + std::chrono::milliseconds(config_.perception_deadline_ms);
  mutable_bundle->camera_id = camera_id;
  mutable_bundle->rgb = state.latest_rgb;
  mutable_bundle->robot_mask = state.latest_robot_mask;
  mutable_bundle->depth = depth_ready ? state.latest_depth : nullptr;
  mutable_bundle->intrinsics = *state.latest_intrinsics;
  mutable_bundle->T_world_camera = *state.latest_T_world_camera;
  mutable_bundle->calibration_revision = state.calibration_revision;
  mutable_bundle->sync.rgb_mask_delta_ns =
      absoluteDelta(time_ns, state.latest_robot_mask_time_ns);
  mutable_bundle->sync.rgb_depth_delta_ns =
      depth_ready ? absoluteDelta(time_ns, state.latest_depth_time_ns) : 0;
  mutable_bundle->sync.tf_delta_ns =
      absoluteDelta(time_ns, state.latest_pose_time_ns);
  mutable_bundle->perception_candidate = perception_candidate;
  FrameBundlePtr bundle = std::move(mutable_bundle);

  FrameBundlePtr detection_bundle;
  if (detection_pending) {
    detection_bundle = bundle;
    if (perception_candidate) {
      state.last_perception_admission_time = candidate_admission_time;
    }
    state.last_emitted_detection_time_ns = time_ns;
    ++detection_frames_emitted_;
  }

  FrameBundlePtr mapping_bundle;
  if (mapping_pending && depth_ready) {
    mapping_bundle = bundle;
    state.last_emitted_mapping_time_ns = time_ns;
    ++mapping_frames_emitted_;
  }

  const bool mapping_done = !state.config.enable_mapping ||
                            state.last_emitted_mapping_time_ns == time_ns;
  const bool detection_done = !state.config.enable_detection ||
                              state.last_emitted_detection_time_ns == time_ns;
  if (mapping_done && detection_done) {
    state.logical_frames.erase(time_ns);
  }
  return {std::move(mapping_bundle), std::move(detection_bundle)};
}

FrameProvenance RosIoThread::frameProvenanceLocked(
    CameraState* state,
    TimeNanoseconds time_ns,
    std::chrono::steady_clock::time_point* ingest_time) {
  auto [identity_it, inserted] = state->logical_frames.try_emplace(time_ns);
  LogicalFrameIdentity& identity = identity_it->second;
  if (inserted) {
    FrameId frame_id = next_frame_id_++;
    if (frame_id == 0) {
      frame_id = next_frame_id_++;
    }
    identity.frame_id = frame_id;
    identity.ingest_time =
        state->latest_rgb_time_ns == time_ns &&
                state->latest_rgb_ingest_time !=
                    std::chrono::steady_clock::time_point::min()
            ? state->latest_rgb_ingest_time
            : std::chrono::steady_clock::now();
  }

  if (ingest_time != nullptr) {
    *ingest_time = identity.ingest_time;
  }
  FrameProvenance provenance;
  provenance.run_id = run_id_;
  provenance.frame_id = identity.frame_id;
  provenance.sensor_time_ns = time_ns;
  provenance.map_mode = config_.map_mode;
  // Map integration and the exact surface barrier happen downstream. ROS IO
  // must not claim a causal relationship that it cannot verify.
  provenance.includes_current_frame = false;
  provenance.causality_verified = false;
  return provenance;
}

void RosIoThread::maybeLogStatusLocked() {
  const auto now = std::chrono::steady_clock::now();
  if (now - last_status_log_time_ <
      std::chrono::duration<double>(config_.log_period_sec)) {
    return;
  }
  last_status_log_time_ = now;

  std::ostringstream stream;
  stream << "status run_id=" << runIdString(run_id_)
         << " map_mode=" << mapModeName(config_.map_mode)
         << " last_frame_id=" << (next_frame_id_ > 1 ? next_frame_id_ - 1 : 0)
         << " rgb=" << rgb_messages_
         << " mask=" << mask_messages_
         << " depth=" << depth_messages_
         << " camera_info=" << camera_info_messages_
         << " tf=" << tf_messages_
         << " tf_static=" << tf_static_messages_
         << " tf_set_failures=" << tf_set_failures_
         << " tf_lookup_ok=" << tf_lookup_successes_
         << " tf_latest_fallback=" << tf_lookup_latest_fallbacks_
         << " tf_stale_latest=" << tf_lookup_stale_latest_
         << " tf_lookup_fail=" << tf_lookup_failures_
         << " mapping_frames=" << mapping_frames_emitted_
         << " detection_frames=" << detection_frames_emitted_
         << " perception_rate_skips=" << perception_rate_skips_
         << " perception_backpressure_skips="
         << perception_backpressure_skips_
         << " cameras=" << camera_states_.size();
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
