#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

class DetectionBridgeThread : public WorkerThread {
 public:
  DetectionBridgeThread(rclcpp::Node& node,
                        ThreadSafeQueue<DetectionFrame>& detection_queue,
                        ThreadSafeQueue<InferenceResponse>& response_queue,
                        MapProjector& map_projector,
                        InferenceBackend& inference_backend,
                        PipelineConfig config);

 protected:
  void run() override;

 private:
  struct PendingDebugFrame {
    ImageBuffer image;
    std::chrono::steady_clock::time_point sent_time;
    Eigen::Isometry3f T_world_camera = Eigen::Isometry3f::Identity();
    double projection_ms = 0.0;
    double resize_ms = 0.0;
    float patch_coverage = 0.0f;
    int valid_patches = 0;
    int projected_points = 0;
    std::uint64_t map_version = 0;
    std::uint64_t source_surface_points = 0;
    std::uint64_t source_tsdf_blocks = 0;
    std::uint64_t source_voxels_scanned = 0;
    std::uint64_t source_selected_blocks = 0;
    std::uint64_t source_cached_surface_points = 0;
    std::uint64_t surface_cache_rebuilds = 0;
    double project_total_ms = 0.0;
    double snapshot_ms = 0.0;
    double surface_extract_ms = 0.0;
    double cache_build_ms = 0.0;
    double frustum_filter_ms = 0.0;
    double projection_loop_ms = 0.0;
    double projection_zbuffer_ms = 0.0;
    bool surface_cache_ready = false;
    bool view_filtered = false;
  };

  bool readyForNextRequest();
  InferenceRequest makeRequest(const DetectionFrame& frame, PatchDepth patch_depth) const;
  void forwardBackendResponses();
  void stashDebugFrame(const InferenceRequest& request,
                       double projection_ms,
                       double resize_ms);
  std::optional<PendingDebugFrame> takeDebugFrame(const InferenceResponse& response);
  void publishDetectionDebugImage(const InferenceResponse& response,
                                  const std::optional<PendingDebugFrame>& pending);
  void publishRawDetectionMarkers(const InferenceResponse& response);
  void maybeLogStatus();

  ThreadSafeQueue<DetectionFrame>& detection_queue_;
  ThreadSafeQueue<InferenceResponse>& response_queue_;
  MapProjector& map_projector_;
  InferenceBackend& inference_backend_;
  PipelineConfig config_;
  rclcpp::Logger logger_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detection_debug_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr raw_detection_pub_;
  std::unordered_map<std::string, PendingDebugFrame> pending_debug_frames_;
  std::deque<std::string> pending_debug_order_;
  std::chrono::steady_clock::time_point last_request_time_;
  std::chrono::steady_clock::time_point last_status_log_time_;
  std::uint64_t frames_seen_ = 0;
  std::uint64_t skipped_rate_ = 0;
  std::uint64_t skipped_no_patch_ = 0;
  std::uint64_t skipped_low_coverage_ = 0;
  std::uint64_t skipped_resize_ = 0;
  std::uint64_t requests_sent_ = 0;
  std::uint64_t responses_seen_ = 0;
  std::uint64_t response_errors_ = 0;
  std::uint64_t last_logged_frames_seen_ = 0;
  std::uint64_t last_logged_skipped_rate_ = 0;
  std::uint64_t last_logged_requests_sent_ = 0;
  std::uint64_t last_logged_responses_seen_ = 0;
  TimeNanoseconds last_rate_skip_time_ns_ = 0;
  double last_rate_skip_since_request_ms_ = 0.0;
  double total_projection_ms_ = 0.0;
  double total_project_total_ms_ = 0.0;
  double total_snapshot_ms_ = 0.0;
  double total_surface_extract_ms_ = 0.0;
  double total_cache_build_ms_ = 0.0;
  double total_frustum_filter_ms_ = 0.0;
  double total_projection_loop_ms_ = 0.0;
  double total_projection_zbuffer_ms_ = 0.0;
  double total_resize_ms_ = 0.0;
  double total_roundtrip_ms_ = 0.0;
  double total_backend_ipc_ms_ = 0.0;
  double total_worker_ms_ = 0.0;
  double total_owl_ms_ = 0.0;
  double total_boxernet_ms_ = 0.0;
  std::uint64_t timing_response_count_ = 0;
};

}  // namespace roomie
