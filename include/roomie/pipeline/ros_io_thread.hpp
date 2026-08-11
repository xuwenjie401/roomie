#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
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
#include <tf2/buffer_core.h>
#include <tf2_msgs/msg/tf_message.hpp>

#include "roomie/pipeline/robot_mask_generator.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/types.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

struct RosCameraSubscriptionConfig {
  std::string camera_id;
  std::string camera_frame;
  std::string rgb_topic;
  std::string depth_topic;
  std::string camera_info_topic;
  CameraIntrinsics fallback_intrinsics;
  float depth_scale = 0.001f;
  float depth_min_m = 0.1f;
  float depth_max_m = 10.0f;
  bool enable_mapping = true;
  bool enable_detection = true;
};

struct RosIoSubscriptionConfig {
  std::string world_frame = "world";
  std::string tf_topic = "/tf";
  std::string tf_static_topic = "/tf_static";
  double max_image_stamp_delta_sec = 0.002;
  double tf_buffer_duration_sec = 5.0;
  double max_tf_gap_sec = 0.2;
  double log_period_sec = 2.0;
  std::size_t input_queue_size = 30;
  MapMode map_mode = MapMode::kOnline;
  double max_perception_fps = 10.0;
  int perception_deadline_ms = 10000;
  std::function<bool()> perception_admission_allowed;
  std::function<void(const FrameBundlePtr&, const std::string&)>
      perception_candidate_cancelled;
  std::shared_ptr<RobotMaskGenerator> robot_mask_generator;
  std::vector<RosCameraSubscriptionConfig> cameras;
};

class RosIoThread : public WorkerThread {
 public:
  RosIoThread(ThreadSafeQueue<FrameBundlePtr>& mapping_queue,
              ThreadSafeQueue<FrameBundlePtr>& detection_queue);

  void configure(RosIoSubscriptionConfig config);
  void attachNode(rclcpp::Node& node);
  void setCameraPose(std::string camera_id,
                     TimeNanoseconds time_ns,
                     Eigen::Isometry3f T_world_camera);

  bool enqueueMappingBundle(FrameBundlePtr frame);
  bool enqueueDetectionBundle(FrameBundlePtr frame);

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  struct LogicalFrameIdentity {
    FrameId frame_id = 0;
    std::chrono::steady_clock::time_point ingest_time =
        std::chrono::steady_clock::time_point::min();
  };

  struct BufferedRgbFrame {
    std::shared_ptr<const ImageBuffer> rgb;
    std::shared_ptr<const ImageBuffer> robot_mask;
    CameraIntrinsics intrinsics;
    std::chrono::steady_clock::time_point ingest_time =
        std::chrono::steady_clock::time_point::min();
    std::uint64_t calibration_revision = 0;
    bool mapping_consumed = false;
    bool detection_consumed = false;
  };

  struct CameraState {
    RosCameraSubscriptionConfig config;
    std::shared_ptr<const ImageBuffer> latest_rgb;
    std::shared_ptr<const ImageBuffer> latest_robot_mask;
    std::shared_ptr<const DepthBuffer> latest_depth;
    std::optional<CameraIntrinsics> latest_intrinsics;
    std::optional<Eigen::Isometry3f> latest_T_world_camera;
    TimeNanoseconds latest_rgb_time_ns = 0;
    TimeNanoseconds latest_robot_mask_time_ns = 0;
    TimeNanoseconds latest_depth_time_ns = 0;
    TimeNanoseconds latest_pose_time_ns = 0;
    std::chrono::steady_clock::time_point latest_rgb_ingest_time =
        std::chrono::steady_clock::time_point::min();
    // Detection consumes each ready RGB frame immediately. Mapping joins the
    // same bounded RGB history with independently arriving registered depth.
    std::map<TimeNanoseconds, BufferedRgbFrame> buffered_rgb_frames;
    std::map<TimeNanoseconds, std::shared_ptr<const DepthBuffer>>
        buffered_depth_frames;
    std::unordered_map<TimeNanoseconds, LogicalFrameIdentity> logical_frames;
    std::chrono::steady_clock::time_point last_perception_admission_time =
        std::chrono::steady_clock::time_point::min();
    std::uint64_t calibration_revision = 0;
    std::uint64_t rgb_revision = 0;
    TimeNanoseconds mask_generation_in_flight_time_ns = 0;
    std::uint64_t mask_generation_in_flight_rgb_revision = 0;
    TimeNanoseconds last_emitted_mapping_time_ns = 0;
    TimeNanoseconds last_emitted_detection_time_ns = 0;
    std::uint64_t masks_rendered = 0;
    std::uint64_t masks_reused = 0;
    std::uint64_t mask_tf_waits = 0;
    std::uint64_t mask_generation_failures = 0;
    std::uint64_t mask_geometry_rejections = 0;
    std::uint64_t full_masks = 0;
    double last_mask_render_ms = 0.0;
    std::size_t last_mask_pixels = 0;
    std::string last_mask_error;
  };

  struct CameraSubscriptions {
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info;
  };

  void handleRgb(const std::string& camera_id, const sensor_msgs::msg::Image::SharedPtr msg);
  void handleDepth(const std::string& camera_id, const sensor_msgs::msg::Image::SharedPtr msg);
  void handleCameraInfo(const std::string& camera_id,
                        const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void handleTf(const tf2_msgs::msg::TFMessage::SharedPtr msg, bool is_static);
  void tryAssembleCamera(const std::string& camera_id);
  void maybeLogStatusLocked();
  FrameProvenance frameProvenanceLocked(CameraState* state,
                                        TimeNanoseconds time_ns,
                                        std::chrono::steady_clock::time_point
                                            observed_ingest_time,
                                        std::chrono::steady_clock::time_point* ingest_time);
  std::optional<TimeNanoseconds> closestBufferedRgbTimeLocked(
      const CameraState& state, TimeNanoseconds depth_time_ns) const;
  void pruneBufferedFramesLocked(CameraState* state);
  std::pair<FrameBundlePtr, FrameBundlePtr> makeFrameBundlesLocked(
      const std::string& camera_id, TimeNanoseconds time_ns);
  void enqueueBundles(FrameBundlePtr mapping_bundle,
                      FrameBundlePtr detection_bundle);
  std::optional<Eigen::Isometry3f> lookupTWorldFrameLocked(
      const std::string& target_frame,
      TimeNanoseconds time_ns);

  ThreadSafeQueue<FrameBundlePtr>& mapping_queue_;
  ThreadSafeQueue<FrameBundlePtr>& detection_queue_;

  std::mutex mutex_;
  std::condition_variable cv_;
  RosIoSubscriptionConfig config_;
  RunId run_id_;
  FrameId next_frame_id_ = 1;
  std::unordered_map<std::string, CameraState> camera_states_;
  std::shared_ptr<tf2::BufferCore> tf_buffer_;
  std::vector<CameraSubscriptions> subscriptions_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_subscription_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_static_subscription_;
  rclcpp::Logger logger_;
  std::chrono::steady_clock::time_point last_status_log_time_ =
      std::chrono::steady_clock::now();
  std::uint64_t rgb_messages_ = 0;
  std::uint64_t depth_messages_ = 0;
  std::uint64_t camera_info_messages_ = 0;
  std::uint64_t tf_messages_ = 0;
  std::uint64_t tf_static_messages_ = 0;
  std::uint64_t tf_set_failures_ = 0;
  std::uint64_t tf_lookup_successes_ = 0;
  std::uint64_t tf_lookup_latest_fallbacks_ = 0;
  std::uint64_t tf_lookup_stale_latest_ = 0;
  std::uint64_t tf_lookup_failures_ = 0;
  std::uint64_t mapping_frames_emitted_ = 0;
  std::uint64_t detection_frames_emitted_ = 0;
  std::uint64_t perception_rate_skips_ = 0;
  std::uint64_t perception_backpressure_skips_ = 0;
  std::string last_tf_error_;
};

}  // namespace roomie
