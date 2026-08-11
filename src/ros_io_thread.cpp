#include "roomie/pipeline/ros_io_thread.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
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

std::optional<Eigen::Isometry3d> transformFromRosDouble(
    const geometry_msgs::msg::TransformStamped& msg) {
  const auto& rotation = msg.transform.rotation;
  Eigen::Quaterniond quaternion(
      rotation.w, rotation.x, rotation.y, rotation.z);
  const double norm = quaternion.norm();
  if (!std::isfinite(norm) || norm <= 0.0) {
    return std::nullopt;
  }
  quaternion.normalize();

  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = quaternion.toRotationMatrix();
  transform.translation() = Eigen::Vector3d(msg.transform.translation.x,
                                             msg.transform.translation.y,
                                             msg.transform.translation.z);
  return transform;
}

bool intrinsicsMatch(const CameraIntrinsics& actual,
                     const RobotMaskCameraInfo& expected,
                     double epsilon = 1.0e-3) {
  return actual.width == expected.width && actual.height == expected.height &&
         std::abs(static_cast<double>(actual.fx) - expected.fx) <= epsilon &&
         std::abs(static_cast<double>(actual.fy) - expected.fy) <= epsilon &&
         std::abs(static_cast<double>(actual.cx) - expected.cx) <= epsilon &&
         std::abs(static_cast<double>(actual.cy) - expected.cy) <= epsilon;
}

bool hasZeroDistortion(const sensor_msgs::msg::CameraInfo& msg) {
  return std::all_of(msg.d.begin(), msg.d.end(), [](double coefficient) {
    return std::isfinite(coefficient) && std::abs(coefficient) <= 1.0e-12;
  });
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

const char* robotActivityName(RobotActivityValue value) {
  switch (value) {
    case RobotActivityValue::kUnknown:
      return "unknown";
    case RobotActivityValue::kInactive:
      return "inactive";
    case RobotActivityValue::kActive:
      return "active";
  }
  return "unknown";
}

const char* robotFrameAdmissionReasonName(RobotFrameAdmissionReason reason) {
  switch (reason) {
    case RobotFrameAdmissionReason::kNone:
      return "none";
    case RobotFrameAdmissionReason::kRotating:
      return "robot_rotating";
    case RobotFrameAdmissionReason::kStateUnknown:
      return "robot_state_unknown";
    case RobotFrameAdmissionReason::kPolicyError:
      return "robot_state_policy_error";
    case RobotFrameAdmissionReason::kNavigationPosture:
      return "navigation_posture_inactive";
  }
  return "robot_state_policy_error";
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
  if (!config.robot_mask_generator) {
    throw std::invalid_argument("ROS IO requires a robot mask generator");
  }
  for (const auto& camera : config.cameras) {
    if (camera.camera_id.empty()) {
      continue;
    }
    if (!config.robot_mask_generator->hasCamera(camera.camera_id)) {
      throw std::invalid_argument(
          "robot mask config has no camera named '" + camera.camera_id + "'");
    }
    const RobotMaskCameraInfo expected =
        config.robot_mask_generator->cameraInfo(camera.camera_id);
    if (!hasUsableIntrinsics(camera.fallback_intrinsics) ||
        !intrinsicsMatch(camera.fallback_intrinsics, expected)) {
      throw std::invalid_argument(
          "fallback intrinsics do not match rectified robot mask profile for camera '" +
          camera.camera_id + "'");
    }
  }
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
  odom_subscription_.reset();
  hand_camera_enable_publisher_.reset();
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

  if (!config_.hand_camera_enable_topic.empty()) {
    hand_camera_enable_publisher_ =
        node.create_publisher<std_msgs::msg::Bool>(
            config_.hand_camera_enable_topic,
            rclcpp::QoS(1).reliable().transient_local());
    std_msgs::msg::Bool disabled;
    disabled.data = false;
    hand_camera_enable_publisher_->publish(disabled);
  }

  if (!config_.odom_topic.empty() && config_.odometry_observer) {
    odom_subscription_ = node.create_subscription<nav_msgs::msg::Odometry>(
        config_.odom_topic,
        sensor_qos,
        [this](nav_msgs::msg::Odometry::SharedPtr msg) {
          handleOdom(std::move(msg));
        });
  }

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
      detection_bundle->perception_candidate &&
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
  if (!msg) {
    return;
  }
  const auto ingest_time = std::chrono::steady_clock::now();
  const TimeNanoseconds sensor_time_ns = toNanoseconds(msg->header.stamp);
  bool detection_only = false;
  std::function<RobotFrameAdmissionDecision(const std::string&,
                                            TimeNanoseconds)>
      robot_frame_admission;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++rgb_messages_;
    const auto state_it = camera_states_.find(camera_id);
    if (state_it != camera_states_.end()) {
      detection_only = !state_it->second.config.enable_mapping &&
                       state_it->second.config.enable_detection;
    }
    robot_frame_admission = config_.robot_frame_admission;
  }

  // Detection-only cameras are admitted before decoding/copying the ROS
  // image, looking up robot TF, or rendering the robot mask.  The later gate
  // in makeFrameBundlesLocked remains authoritative across a state transition.
  if (detection_only && robot_frame_admission) {
    RobotFrameAdmissionDecision admission;
    try {
      admission = robot_frame_admission(camera_id, sensor_time_ns);
    } catch (...) {
      admission.allow_detection = false;
      admission.reason = RobotFrameAdmissionReason::kPolicyError;
    }
    if (!admission.allow_detection) {
      if (admission.reason == RobotFrameAdmissionReason::kNone) {
        admission.reason = RobotFrameAdmissionReason::kPolicyError;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      const auto state_it = camera_states_.find(camera_id);
      if (state_it != camera_states_.end()) {
        ++state_it->second.preprocess_admission_drops;
      }
      if (admission.reason == RobotFrameAdmissionReason::kRotating) {
        ++robot_rotation_detection_drops_;
      } else if (admission.reason ==
                 RobotFrameAdmissionReason::kStateUnknown) {
        ++robot_unknown_detection_drops_;
      } else if (admission.reason ==
                 RobotFrameAdmissionReason::kNavigationPosture) {
        ++robot_navigation_posture_detection_drops_;
      } else if (admission.reason ==
                 RobotFrameAdmissionReason::kPolicyError) {
        ++robot_policy_errors_;
      }
      maybeLogStatusLocked();
      return;
    }
  }

  auto rgb = imageBufferFromRos(*msg);
  if (!rgb) {
    RCLCPP_WARN(logger_,
                "dropping RGB frame for %s with unsupported encoding '%s'",
                camera_id.c_str(), msg->encoding.c_str());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = camera_states_[camera_id];
    state.config.camera_id = camera_id;
    const RobotMaskCameraInfo expected =
        config_.robot_mask_generator->cameraInfo(camera_id);
    if (rgb->width != expected.width || rgb->height != expected.height) {
      ++state.mask_geometry_rejections;
      RCLCPP_WARN(logger_,
                  "dropping RGB frame for %s: got %dx%d, rectified mask profile is %dx%d",
                  camera_id.c_str(),
                  rgb->width,
                  rgb->height,
                  expected.width,
                  expected.height);
      maybeLogStatusLocked();
      return;
    }
    if (state.latest_rgb_time_ns != sensor_time_ns ||
        state.latest_rgb_ingest_time == std::chrono::steady_clock::time_point::min()) {
      state.latest_rgb_ingest_time = ingest_time;
    }
    state.latest_rgb_time_ns = sensor_time_ns;
    state.latest_rgb =
        std::make_shared<const ImageBuffer>(std::move(*rgb));
    ++state.rgb_revision;
    state.latest_robot_mask.reset();
    state.latest_robot_mask_time_ns = 0;
    maybeLogStatusLocked();
  }
  tryAssembleCamera(camera_id);
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
    const RobotMaskCameraInfo expected =
        config_.robot_mask_generator->cameraInfo(camera_id);
    if (depth->width != expected.width || depth->height != expected.height) {
      ++state.mask_geometry_rejections;
      RCLCPP_WARN(logger_,
                  "dropping registered depth for %s: got %dx%d, expected %dx%d",
                  camera_id.c_str(),
                  depth->width,
                  depth->height,
                  expected.width,
                  expected.height);
      maybeLogStatusLocked();
      return;
    }
    const TimeNanoseconds depth_time_ns = toNanoseconds(msg->header.stamp);
    auto immutable_depth =
        std::make_shared<const DepthBuffer>(std::move(*depth));
    state.latest_depth_time_ns = depth_time_ns;
    state.latest_depth = immutable_depth;
    state.buffered_depth_frames[depth_time_ns] = std::move(immutable_depth);
    pruneBufferedFramesLocked(&state);
    const std::optional<TimeNanoseconds> rgb_time_ns =
        closestBufferedRgbTimeLocked(state, depth_time_ns);
    if (rgb_time_ns) {
      auto bundles = makeFrameBundlesLocked(camera_id, *rgb_time_ns);
      mapping_bundle = std::move(bundles.first);
      detection_bundle = std::move(bundles.second);
    }
    maybeLogStatusLocked();
  }
  enqueueBundles(std::move(mapping_bundle), std::move(detection_bundle));
  tryAssembleCamera(camera_id);
}

void RosIoThread::handleCameraInfo(const std::string& camera_id,
                                   const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = camera_states_[camera_id];
    state.config.camera_id = camera_id;
    const CameraIntrinsics intrinsics = intrinsicsFromRos(*msg);
    const RobotMaskCameraInfo expected =
        config_.robot_mask_generator->cameraInfo(camera_id);
    if (hasUsableIntrinsics(intrinsics) &&
        intrinsicsMatch(intrinsics, expected) && hasZeroDistortion(*msg)) {
      ++camera_info_messages_;
      state.latest_intrinsics = intrinsics;
      ++state.calibration_revision;
    } else {
      ++state.mask_geometry_rejections;
      state.latest_intrinsics.reset();
      ++state.calibration_revision;
      RCLCPP_WARN(logger_,
                  "rejecting CameraInfo for %s: mask input must be rectified and match the configured geometry",
                  camera_id.c_str());
    }
    maybeLogStatusLocked();
  }
  tryAssembleCamera(camera_id);
}

void RosIoThread::handleOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
  if (!msg) {
    return;
  }
  std::function<RobotStateEstimatorUpdate(const OdometryObservation&)> observer;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++odom_messages_;
    observer = config_.odometry_observer;
  }
  if (!observer) {
    return;
  }

  OdometryObservation observation;
  observation.time_ns = toNanoseconds(msg->header.stamp);
  observation.position = Eigen::Vector3d(msg->pose.pose.position.x,
                                         msg->pose.pose.position.y,
                                         msg->pose.pose.position.z);
  observation.orientation = Eigen::Quaterniond(
      msg->pose.pose.orientation.w,
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z);

  RobotStateEstimatorUpdate update;
  try {
    update = observer(observation);
  } catch (const std::exception& error) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++odom_invalid_;
    }
    RunLogger::logGlobal("robot_state",
                         "odom_observer_error error='" +
                             std::string(error.what()) + "'");
    return;
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++odom_invalid_;
    }
    RunLogger::logGlobal("robot_state",
                         "odom_observer_error error=unknown");
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (update.out_of_order) {
      ++odom_out_of_order_;
    }
    if (update.outcome == OdometryObservationOutcome::kInvalid) {
      ++odom_invalid_;
    } else if (update.outcome == OdometryObservationOutcome::kReplaced) {
      ++odom_replaced_;
    } else if (update.outcome == OdometryObservationOutcome::kTooOld) {
      ++odom_too_old_;
    }
    maybeLogStatusLocked();
  }
  if (update.latest_state_changed) {
    std::ostringstream stream;
    stream << "state_transition activity=rotating from="
           << robotActivityName(update.previous_latest_state)
           << " to=" << robotActivityName(update.latest.rotating.value)
           << " sensor_time_ns=" << update.latest.source_time_ns
           << " since_ns=" << update.latest.rotating.since_ns;
    if (update.latest.yaw_rate_rad_s) {
      stream << " yaw_rate_rad_s=" << *update.latest.yaw_rate_rad_s;
    }
    RunLogger::logGlobal("robot_state", stream.str());
  }
}

void RosIoThread::handleTf(const tf2_msgs::msg::TFMessage::SharedPtr msg, bool is_static) {
  if (!msg) {
    return;
  }
  std::vector<std::string> cameras_to_retry;
  std::function<RobotPostureEstimatorUpdate(const PostureObservation&)>
      posture_observer;
  PostureObservation posture_observation;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_static) {
      ++tf_static_messages_;
    } else {
      ++tf_messages_;
      posture_observer = config_.posture_observer;
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

      if (!is_static && posture_observer) {
        LinkTransformObservation observation;
        observation.parent_frame_id = stored.header.frame_id;
        observation.child_frame_id = stored.child_frame_id;
        observation.position = Eigen::Vector3d(
            stored.transform.translation.x,
            stored.transform.translation.y,
            stored.transform.translation.z);
        observation.orientation = Eigen::Quaterniond(
            stored.transform.rotation.w,
            stored.transform.rotation.x,
            stored.transform.rotation.y,
            stored.transform.rotation.z)
                                      .normalized();
        posture_observation.time_ns = std::max(
            posture_observation.time_ns, toNanoseconds(stored.header.stamp));
        posture_observation.transforms.push_back(std::move(observation));
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

    for (const auto& [camera_id, state] : camera_states_) {
      if (state.latest_rgb_time_ns == 0) {
        continue;
      }
      cameras_to_retry.push_back(camera_id);
    }
    maybeLogStatusLocked();
  }

  if (posture_observer && posture_observation.time_ns > 0 &&
      !posture_observation.transforms.empty()) {
    try {
      const RobotPostureEstimatorUpdate update =
          posture_observer(posture_observation);
      if (update.latest_state_changed) {
        if (hand_camera_enable_publisher_) {
          std_msgs::msg::Bool enabled;
          enabled.data =
              update.latest.navigation_posture_deviated.value ==
              RobotActivityValue::kActive;
          hand_camera_enable_publisher_->publish(enabled);
        }
        std::ostringstream stream;
        stream << "state_transition activity=navigation_posture_deviated from="
               << robotActivityName(update.previous_latest_state)
               << " to="
               << robotActivityName(
                      update.latest.navigation_posture_deviated.value)
               << " sensor_time_ns=" << update.latest.source_time_ns
               << " since_ns="
               << update.latest.navigation_posture_deviated.since_ns;
        RunLogger::logGlobal("robot_state", stream.str());
      }
    } catch (const std::exception& error) {
      if (hand_camera_enable_publisher_) {
        std_msgs::msg::Bool disabled;
        disabled.data = false;
        hand_camera_enable_publisher_->publish(disabled);
      }
      RunLogger::logGlobal("robot_state",
                           "posture_observer_error error='" +
                               std::string(error.what()) + "'");
    } catch (...) {
      if (hand_camera_enable_publisher_) {
        std_msgs::msg::Bool disabled;
        disabled.data = false;
        hand_camera_enable_publisher_->publish(disabled);
      }
      RunLogger::logGlobal("robot_state",
                           "posture_observer_error error=unknown");
    }
  }

  for (const std::string& camera_id : cameras_to_retry) {
    tryAssembleCamera(camera_id);
  }
}

void RosIoThread::tryAssembleCamera(const std::string& camera_id) {
  while (true) {
    TimeNanoseconds image_time_ns = 0;
    std::uint64_t rgb_revision = 0;
    std::shared_ptr<RobotMaskGenerator> generator;
    std::shared_ptr<tf2::BufferCore> tf_buffer;
    std::shared_ptr<const ImageBuffer> rgb_for_mask;
    std::optional<CameraIntrinsics> intrinsics_for_mask;
    std::chrono::steady_clock::time_point rgb_ingest_time =
        std::chrono::steady_clock::time_point::min();
    std::uint64_t calibration_revision = 0;
    FrameBundlePtr ready_mapping_bundle;
    FrameBundlePtr ready_detection_bundle;
    bool mask_was_already_ready = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto found = camera_states_.find(camera_id);
      if (found == camera_states_.end()) {
        return;
      }
      CameraState& state = found->second;
      if (!state.latest_rgb || state.latest_rgb_time_ns == 0 ||
          !config_.robot_mask_generator || !tf_buffer_) {
        return;
      }
      if (state.latest_robot_mask &&
          state.latest_robot_mask_time_ns == state.latest_rgb_time_ns) {
        mask_was_already_ready = true;
        auto bundles =
            makeFrameBundlesLocked(camera_id, state.latest_rgb_time_ns);
        ready_mapping_bundle = std::move(bundles.first);
        ready_detection_bundle = std::move(bundles.second);
        maybeLogStatusLocked();
      } else {
        if (state.mask_generation_in_flight_time_ns != 0) {
          return;
        }
        image_time_ns = state.latest_rgb_time_ns;
        rgb_revision = state.rgb_revision;
        state.mask_generation_in_flight_time_ns = image_time_ns;
        state.mask_generation_in_flight_rgb_revision = rgb_revision;
        generator = config_.robot_mask_generator;
        tf_buffer = tf_buffer_;
        rgb_for_mask = state.latest_rgb;
        intrinsics_for_mask = state.latest_intrinsics;
        rgb_ingest_time = state.latest_rgb_ingest_time;
        calibration_revision = state.calibration_revision;
      }
    }
    if (mask_was_already_ready) {
      enqueueBundles(std::move(ready_mapping_bundle),
                     std::move(ready_detection_bundle));
      return;
    }

    RobotPoseSnapshot pose;
    pose.time_ns = image_time_ns;
    bool pose_ready = true;
    std::string pose_error;
    for (const std::string& frame : generator->requiredFrames()) {
      if (normalizeFrameId(frame) == normalizeFrameId(generator->rootFrame())) {
        pose.root_T_frame.emplace(frame, Eigen::Isometry3d::Identity());
        continue;
      }
      try {
        const geometry_msgs::msg::TransformStamped transform_msg =
            tf_buffer->lookupTransform(
                normalizeFrameId(generator->rootFrame()),
                normalizeFrameId(frame),
                tf2::TimePoint(std::chrono::nanoseconds(image_time_ns)));
        auto transform = transformFromRosDouble(transform_msg);
        if (!transform) {
          pose_ready = false;
          pose_error = "invalid rotation for internal TF " +
                       generator->rootFrame() + " <- " + frame;
          break;
        }
        pose.root_T_frame.emplace(frame, std::move(*transform));
      } catch (const tf2::TransformException& error) {
        pose_ready = false;
        pose_error = error.what();
        break;
      } catch (const std::exception& error) {
        pose_ready = false;
        pose_error = error.what();
        break;
      }
    }

    if (!pose_ready) {
      bool newer_rgb_waiting = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = camera_states_.find(camera_id);
        if (found == camera_states_.end()) {
          return;
        }
        CameraState& state = found->second;
        if (state.mask_generation_in_flight_time_ns == image_time_ns &&
            state.mask_generation_in_flight_rgb_revision == rgb_revision) {
          state.mask_generation_in_flight_time_ns = 0;
          state.mask_generation_in_flight_rgb_revision = 0;
        }
        ++state.mask_tf_waits;
        state.last_mask_error = pose_error;
        newer_rgb_waiting = state.latest_rgb_time_ns != image_time_ns ||
                            state.rgb_revision != rgb_revision;
        maybeLogStatusLocked();
      }
      if (newer_rgb_waiting) {
        continue;
      }
      return;
    }

    RobotMaskResult result;
    try {
      result = generator->generate(camera_id, pose);
    } catch (const std::exception& error) {
      bool newer_rgb_waiting = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = camera_states_.find(camera_id);
        if (found == camera_states_.end()) {
          return;
        }
        CameraState& state = found->second;
        if (state.mask_generation_in_flight_time_ns == image_time_ns &&
            state.mask_generation_in_flight_rgb_revision == rgb_revision) {
          state.mask_generation_in_flight_time_ns = 0;
          state.mask_generation_in_flight_rgb_revision = 0;
        }
        ++state.mask_generation_failures;
        state.last_mask_error = error.what();
        newer_rgb_waiting = state.latest_rgb_time_ns != image_time_ns ||
                            state.rgb_revision != rgb_revision;
        maybeLogStatusLocked();
      }
      RCLCPP_ERROR(logger_,
                   "robot mask generation failed for %s: %s",
                   camera_id.c_str(),
                   error.what());
      if (newer_rgb_waiting) {
        continue;
      }
      return;
    }

    bool newer_rgb_waiting = false;
    FrameBundlePtr mapping_bundle;
    FrameBundlePtr detection_bundle;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto found = camera_states_.find(camera_id);
      if (found == camera_states_.end()) {
        return;
      }
      CameraState& state = found->second;
      if (state.mask_generation_in_flight_time_ns == image_time_ns &&
          state.mask_generation_in_flight_rgb_revision == rgb_revision) {
        state.mask_generation_in_flight_time_ns = 0;
        state.mask_generation_in_flight_rgb_revision = 0;
      }
      newer_rgb_waiting = state.latest_rgb_time_ns != image_time_ns ||
                          state.rgb_revision != rgb_revision;
      state.last_mask_error.clear();
      const std::shared_ptr<const ImageBuffer> generated_mask = result.mask;
      if (!newer_rgb_waiting) {
        state.latest_robot_mask = generated_mask;
        state.latest_robot_mask_time_ns = image_time_ns;
      }
      state.last_mask_pixels = result.mask_pixels;
      if (result.reused) {
        ++state.masks_reused;
      } else {
        ++state.masks_rendered;
        state.last_mask_render_ms = result.render_ms;
      }
      if (result.full_mask) {
        ++state.full_masks;
      }
      if (rgb_for_mask && generated_mask && intrinsics_for_mask) {
        BufferedRgbFrame buffered;
        buffered.rgb = std::move(rgb_for_mask);
        buffered.robot_mask = generated_mask;
        buffered.intrinsics = *intrinsics_for_mask;
        buffered.ingest_time = rgb_ingest_time;
        buffered.calibration_revision = calibration_revision;
        auto existing = state.buffered_rgb_frames.find(image_time_ns);
        if (existing != state.buffered_rgb_frames.end()) {
          buffered.mapping_consumed = existing->second.mapping_consumed;
          buffered.detection_consumed = existing->second.detection_consumed;
        }
        state.buffered_rgb_frames[image_time_ns] = std::move(buffered);
        pruneBufferedFramesLocked(&state);
        auto bundles = makeFrameBundlesLocked(camera_id, image_time_ns);
        mapping_bundle = std::move(bundles.first);
        detection_bundle = std::move(bundles.second);
      }
      maybeLogStatusLocked();
    }
    enqueueBundles(std::move(mapping_bundle), std::move(detection_bundle));
    if (!newer_rgb_waiting) {
      return;
    }
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

std::optional<TimeNanoseconds> RosIoThread::closestBufferedRgbTimeLocked(
    const CameraState& state, TimeNanoseconds depth_time_ns) const {
  if (!state.config.enable_mapping || depth_time_ns == 0) {
    return std::nullopt;
  }
  const TimeNanoseconds max_delta_ns =
      secondsToNanoseconds(config_.max_image_stamp_delta_sec);
  std::optional<TimeNanoseconds> best_time_ns;
  TimeNanoseconds best_delta_ns = std::numeric_limits<TimeNanoseconds>::max();
  for (const auto& [rgb_time_ns, rgb_frame] : state.buffered_rgb_frames) {
    if (rgb_frame.mapping_consumed || rgb_time_ns == 0) {
      continue;
    }
    const TimeNanoseconds delta_ns = absoluteDelta(rgb_time_ns, depth_time_ns);
    const bool within_tolerance =
        delta_ns == 0 || (max_delta_ns > 0 && delta_ns <= max_delta_ns);
    if (within_tolerance && delta_ns < best_delta_ns) {
      best_time_ns = rgb_time_ns;
      best_delta_ns = delta_ns;
    }
  }
  return best_time_ns;
}

void RosIoThread::pruneBufferedFramesLocked(CameraState* state) {
  if (state == nullptr) {
    return;
  }
  const std::size_t limit =
      std::max<std::size_t>(2, config_.input_queue_size);
  while (state->buffered_rgb_frames.size() > limit) {
    const TimeNanoseconds expired_time_ns =
        state->buffered_rgb_frames.begin()->first;
    state->logical_frames.erase(expired_time_ns);
    state->buffered_rgb_frames.erase(state->buffered_rgb_frames.begin());
  }
  while (state->buffered_depth_frames.size() > limit) {
    state->buffered_depth_frames.erase(state->buffered_depth_frames.begin());
  }
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
  auto rgb_it = state.buffered_rgb_frames.find(time_ns);
  if (rgb_it == state.buffered_rgb_frames.end()) {
    return {};
  }
  BufferedRgbFrame& rgb_frame = rgb_it->second;
  bool mapping_pending =
      state.config.enable_mapping && !rgb_frame.mapping_consumed;
  bool detection_pending =
      state.config.enable_detection && !rgb_frame.detection_consumed;
  if (!mapping_pending && !detection_pending) {
    return {};
  }
  if (!rgb_frame.rgb || !rgb_frame.robot_mask ||
      state.config.camera_frame.empty()) {
    return {};
  }
  if (rgb_frame.robot_mask->width != rgb_frame.rgb->width ||
      rgb_frame.robot_mask->height != rgb_frame.rgb->height ||
      rgb_frame.robot_mask->channels != 1) {
    return {};
  }

  RobotFrameAdmissionDecision robot_admission;
  if (config_.robot_frame_admission) {
    try {
      robot_admission = config_.robot_frame_admission(camera_id, time_ns);
    } catch (...) {
      robot_admission.allow_mapping = false;
      robot_admission.allow_detection = false;
      robot_admission.reason = RobotFrameAdmissionReason::kPolicyError;
    }
  }
  const bool mapping_state_blocked =
      mapping_pending && !robot_admission.allow_mapping;
  const bool detection_state_blocked =
      detection_pending && !robot_admission.allow_detection;
  if (mapping_state_blocked || detection_state_blocked) {
    if (robot_admission.reason == RobotFrameAdmissionReason::kNone) {
      robot_admission.reason = RobotFrameAdmissionReason::kPolicyError;
    }
    if (mapping_state_blocked) {
      rgb_frame.mapping_consumed = true;
      mapping_pending = false;
      if (robot_admission.reason == RobotFrameAdmissionReason::kRotating) {
        ++robot_rotation_mapping_drops_;
      } else if (robot_admission.reason ==
                 RobotFrameAdmissionReason::kStateUnknown) {
        ++robot_unknown_mapping_drops_;
      }
    }
    if (detection_state_blocked) {
      rgb_frame.detection_consumed = true;
      detection_pending = false;
      if (robot_admission.reason == RobotFrameAdmissionReason::kRotating) {
        ++robot_rotation_detection_drops_;
      } else if (robot_admission.reason ==
                 RobotFrameAdmissionReason::kStateUnknown) {
        ++robot_unknown_detection_drops_;
      } else if (robot_admission.reason ==
                 RobotFrameAdmissionReason::kNavigationPosture) {
        ++robot_navigation_posture_detection_drops_;
      }
    }
    if (robot_admission.reason == RobotFrameAdmissionReason::kPolicyError) {
      ++robot_policy_errors_;
    }
    std::ostringstream stream;
    stream << "frame_drop camera=" << camera_id
           << " sensor_time_ns=" << time_ns
           << " reason="
           << robotFrameAdmissionReasonName(robot_admission.reason)
           << " mapping=" << (mapping_state_blocked ? "true" : "false")
           << " detection="
           << (detection_state_blocked ? "true" : "false")
           << " state_source_time_ns="
           << robot_admission.state.source_time_ns;
    if (robot_admission.state.yaw_rate_rad_s) {
      stream << " yaw_rate_rad_s="
             << *robot_admission.state.yaw_rate_rad_s;
    }
    RunLogger::logGlobal("ros_io", stream.str());

    if (!mapping_pending && !detection_pending) {
      state.logical_frames.erase(time_ns);
      state.buffered_rgb_frames.erase(rgb_it);
      return {};
    }
  }

  const TimeNanoseconds max_image_delta_ns =
      secondsToNanoseconds(config_.max_image_stamp_delta_sec);
  auto matched_depth_it = state.buffered_depth_frames.end();
  TimeNanoseconds matched_depth_delta_ns =
      std::numeric_limits<TimeNanoseconds>::max();
  if (mapping_pending) {
    for (auto candidate = state.buffered_depth_frames.begin();
         candidate != state.buffered_depth_frames.end(); ++candidate) {
      const auto& depth = candidate->second;
      if (!depth || depth->width != rgb_frame.rgb->width ||
          depth->height != rgb_frame.rgb->height) {
        continue;
      }
      const TimeNanoseconds delta_ns = absoluteDelta(time_ns, candidate->first);
      const bool within_tolerance =
          delta_ns == 0 ||
          (max_image_delta_ns > 0 && delta_ns <= max_image_delta_ns);
      if (within_tolerance && delta_ns < matched_depth_delta_ns) {
        matched_depth_it = candidate;
        matched_depth_delta_ns = delta_ns;
      }
    }
  }
  const bool depth_ready = matched_depth_it != state.buffered_depth_frames.end();

  bool detection_admitted = false;
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
      rgb_frame.detection_consumed = true;
      ++perception_backpressure_skips_;
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
        rgb_frame.detection_consumed = true;
        ++perception_rate_skips_;
        detection_pending = false;
      } else {
        candidate_admission_time = now;
        detection_admitted = true;
      }
    }
  }
  if ((!mapping_pending || !depth_ready) && !detection_admitted) {
    if ((!state.config.enable_mapping || rgb_frame.mapping_consumed) &&
        (!state.config.enable_detection || rgb_frame.detection_consumed)) {
      state.logical_frames.erase(time_ns);
      state.buffered_rgb_frames.erase(rgb_it);
    }
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
      frameProvenanceLocked(&state, time_ns, rgb_frame.ingest_time,
                            &ingest_time);
  const auto make_bundle = [&](std::shared_ptr<const DepthBuffer> depth,
                               TimeNanoseconds rgb_depth_delta_ns) {
    auto mutable_bundle = std::make_shared<FrameBundle>();
    mutable_bundle->provenance = provenance;
    mutable_bundle->ingest_time = ingest_time;
    mutable_bundle->due_time =
        ingest_time + std::chrono::milliseconds(config_.perception_deadline_ms);
    mutable_bundle->camera_id = camera_id;
    mutable_bundle->rgb = rgb_frame.rgb;
    mutable_bundle->robot_mask = rgb_frame.robot_mask;
    mutable_bundle->depth = std::move(depth);
    mutable_bundle->intrinsics = rgb_frame.intrinsics;
    mutable_bundle->T_world_camera = *state.latest_T_world_camera;
    mutable_bundle->calibration_revision = rgb_frame.calibration_revision;
    mutable_bundle->sync.rgb_depth_delta_ns = rgb_depth_delta_ns;
    mutable_bundle->sync.tf_delta_ns =
        absoluteDelta(time_ns, state.latest_pose_time_ns);
    mutable_bundle->robot_state = robot_admission.state;
    // Independent detections project the latest committed map and therefore do
    // not reserve an exact include-current map commit barrier.
    mutable_bundle->perception_candidate = false;
    return FrameBundlePtr(std::move(mutable_bundle));
  };

  FrameBundlePtr detection_bundle;
  if (detection_admitted) {
    detection_bundle = make_bundle(nullptr, 0);
    state.last_perception_admission_time = candidate_admission_time;
    state.last_emitted_detection_time_ns = time_ns;
    rgb_frame.detection_consumed = true;
    ++detection_frames_emitted_;
  }

  FrameBundlePtr mapping_bundle;
  if (mapping_pending && depth_ready) {
    mapping_bundle =
        make_bundle(matched_depth_it->second, matched_depth_delta_ns);
    state.last_emitted_mapping_time_ns = time_ns;
    rgb_frame.mapping_consumed = true;
    state.buffered_depth_frames.erase(matched_depth_it);
    ++mapping_frames_emitted_;
  }

  const bool mapping_done = !state.config.enable_mapping ||
                            rgb_frame.mapping_consumed;
  const bool detection_done = !state.config.enable_detection ||
                              rgb_frame.detection_consumed;
  if (mapping_done && detection_done) {
    state.logical_frames.erase(time_ns);
    state.buffered_rgb_frames.erase(rgb_it);
  }
  return {std::move(mapping_bundle), std::move(detection_bundle)};
}

FrameProvenance RosIoThread::frameProvenanceLocked(
    CameraState* state,
    TimeNanoseconds time_ns,
    std::chrono::steady_clock::time_point observed_ingest_time,
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
        observed_ingest_time != std::chrono::steady_clock::time_point::min()
            ? observed_ingest_time
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
         << " depth=" << depth_messages_
         << " camera_info=" << camera_info_messages_
         << " odom=" << odom_messages_
         << " odom_invalid=" << odom_invalid_
         << " odom_out_of_order=" << odom_out_of_order_
         << " odom_replaced=" << odom_replaced_
         << " odom_too_old=" << odom_too_old_
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
         << " robot_rotation_mapping_drops="
         << robot_rotation_mapping_drops_
         << " robot_rotation_detection_drops="
         << robot_rotation_detection_drops_
         << " robot_unknown_mapping_drops="
         << robot_unknown_mapping_drops_
         << " robot_unknown_detection_drops="
         << robot_unknown_detection_drops_
         << " robot_navigation_posture_detection_drops="
         << robot_navigation_posture_detection_drops_
         << " robot_policy_errors=" << robot_policy_errors_
         << " cameras=" << camera_states_.size();
  for (const auto& [camera_id, state] : camera_states_) {
    stream << " camera[" << camera_id << "]={rendered="
           << state.masks_rendered << ",reused=" << state.masks_reused
           << ",buffered_rgb=" << state.buffered_rgb_frames.size()
           << ",buffered_depth=" << state.buffered_depth_frames.size()
           << ",tf_waits=" << state.mask_tf_waits
           << ",failures=" << state.mask_generation_failures
           << ",geometry_rejections=" << state.mask_geometry_rejections
           << ",preprocess_drops=" << state.preprocess_admission_drops
           << ",full_masks=" << state.full_masks
           << ",last_render_ms=" << state.last_mask_render_ms
           << ",last_pixels=" << state.last_mask_pixels;
    if (!state.last_mask_error.empty()) {
      stream << ",last_error='" << state.last_mask_error << "'";
    }
    stream << "}";
  }
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
