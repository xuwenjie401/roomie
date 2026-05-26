#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/types.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

struct RosCameraSubscriptionConfig {
  std::string camera_id;
  std::string camera_frame;
  std::string rgb_topic;
  std::string robot_mask_topic;
  std::string depth_topic;
  std::string camera_info_topic;
  CameraIntrinsics fallback_intrinsics;
  float depth_scale = 0.001f;
  float depth_min_m = 0.1f;
  float depth_max_m = 10.0f;
  int mask_robot_threshold = 0;
  bool enable_mapping = true;
  bool enable_detection = true;
};

struct RosIoSubscriptionConfig {
  std::string world_frame = "world";
  std::string tf_topic = "/tf";
  std::string tf_static_topic = "/tf_static";
  double max_image_stamp_delta_sec = 0.002;
  double max_tf_gap_sec = 0.2;
  std::size_t input_queue_size = 30;
  std::vector<RosCameraSubscriptionConfig> cameras;
};

class RosIoThread : public WorkerThread {
 public:
  RosIoThread(ThreadSafeQueue<MappingFrame>& mapping_queue,
              ThreadSafeQueue<DetectionFrame>& detection_queue);

  void configure(RosIoSubscriptionConfig config);
  void attachNode(rclcpp::Node& node);
  void setCameraPose(std::string camera_id,
                     TimeNanoseconds time_ns,
                     Eigen::Isometry3f T_world_camera);

  bool enqueueMappingFrame(MappingFrame frame);
  bool enqueueDetectionFrame(DetectionFrame frame);

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  struct CameraState {
    RosCameraSubscriptionConfig config;
    std::optional<ImageBuffer> latest_rgb;
    std::optional<ImageBuffer> latest_robot_mask;
    std::optional<DepthBuffer> latest_depth;
    std::optional<CameraIntrinsics> latest_intrinsics;
    std::optional<Eigen::Isometry3f> latest_T_world_camera;
    TimeNanoseconds latest_rgb_time_ns = 0;
    TimeNanoseconds latest_robot_mask_time_ns = 0;
    TimeNanoseconds latest_depth_time_ns = 0;
    TimeNanoseconds latest_pose_time_ns = 0;
    TimeNanoseconds last_emitted_mapping_time_ns = 0;
    TimeNanoseconds last_emitted_detection_time_ns = 0;
  };

  struct CameraSubscriptions {
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr robot_mask;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info;
  };

  struct StoredTransform {
    std::string parent_frame;
    Eigen::Isometry3f T_parent_child = Eigen::Isometry3f::Identity();
    TimeNanoseconds time_ns = 0;
    bool is_static = false;
  };

  void handleRgb(const std::string& camera_id, const sensor_msgs::msg::Image::SharedPtr msg);
  void handleRobotMask(const std::string& camera_id,
                       const sensor_msgs::msg::Image::SharedPtr msg);
  void handleDepth(const std::string& camera_id, const sensor_msgs::msg::Image::SharedPtr msg);
  void handleCameraInfo(const std::string& camera_id,
                        const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void handleTf(const tf2_msgs::msg::TFMessage::SharedPtr msg, bool is_static);
  std::pair<std::optional<MappingFrame>, std::optional<DetectionFrame>>
  makeFramesLocked(const std::string& camera_id, TimeNanoseconds time_ns);
  std::optional<Eigen::Isometry3f> lookupTWorldFrameLocked(
      const std::string& target_frame,
      TimeNanoseconds time_ns) const;

  ThreadSafeQueue<MappingFrame>& mapping_queue_;
  ThreadSafeQueue<DetectionFrame>& detection_queue_;

  std::mutex mutex_;
  std::condition_variable cv_;
  RosIoSubscriptionConfig config_;
  std::unordered_map<std::string, CameraState> camera_states_;
  std::unordered_map<std::string, StoredTransform> transforms_by_child_frame_;
  std::vector<CameraSubscriptions> subscriptions_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_subscription_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_static_subscription_;
  rclcpp::Logger logger_;
};

}  // namespace roomie
