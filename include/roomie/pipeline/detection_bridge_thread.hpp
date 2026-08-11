#pragma once

#include <chrono>
#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
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
                        ThreadSafeQueue<FrameBundlePtr>& detection_queue,
                        ThreadSafeQueue<InferenceResponse>& response_queue,
                        MapProjector& map_projector,
                        InferenceBackend& inference_backend,
                        PipelineConfig config);

  // MapThread invokes this observer after publishing an immutable exact
  // commit. The callback never performs projection or waits.
  void onMapCommit(const MapCommit& commit);
  void cancelPendingCandidate(const FrameBundlePtr& frame,
                              const std::string& reason);
  bool perceptionBusy() const;

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  struct RequestBuildTiming {
    double rgb_resize_ms = 0.0;
    double mask_resize_ms = 0.0;
  };

  struct PendingDebugFrame {
    ImageBuffer image;
    FrameProvenance provenance;
    std::chrono::steady_clock::time_point ingest_time;
    std::chrono::steady_clock::time_point due_time;
    std::chrono::steady_clock::time_point sent_time;
    Eigen::Isometry3f T_world_camera = Eigen::Isometry3f::Identity();
    std::shared_ptr<const VisibilityContext> visibility_context;
    double projection_ms = 0.0;
    double resize_ms = 0.0;
    double rgb_resize_ms = 0.0;
    double mask_resize_ms = 0.0;
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
    double map_commit_wait_ms = 0.0;
    double map_commit_latency_ms = 0.0;
    double projection_compute_ms = 0.0;
    bool surface_cache_ready = false;
    bool view_filtered = false;
  };

  struct PendingMapCommit {
    MapCommit commit;
    std::chrono::steady_clock::time_point received_at =
        std::chrono::steady_clock::time_point::min();
  };

  void admitNextCandidate();
  void pollActiveCandidate();
  void processReadyCandidate(FrameBundlePtr frame,
                             const MapCommit* commit,
                             double map_commit_wait_ms,
                             double map_commit_latency_ms);
  void finishActiveCandidate(const std::string& reason);
  std::optional<PendingMapCommit> takeMapCommit(const FrameBundle& frame);
  void pruneMapCommitsLocked(std::chrono::steady_clock::time_point now);
  void discardUnstartedCandidatesForShutdown();
  RequestId allocateRequestId();
  InferenceRequest makeRequest(const FrameBundle& frame,
                               PatchDepth patch_depth,
                               RequestBuildTiming* timing);
  void forwardBackendResponses();
  void stashDebugFrame(const InferenceRequest& request,
                       double projection_ms,
                       double resize_ms,
                       const RequestBuildTiming& timing);
  std::optional<PendingDebugFrame> takeDebugFrame(const InferenceResponse& response);
  void publishDetectionDebugImage(const InferenceResponse& response,
                                  const std::optional<PendingDebugFrame>& pending);
  void publishRawDetectionMarkers(const InferenceResponse& response);
  void maybeLogStatus();

  ThreadSafeQueue<FrameBundlePtr>& detection_queue_;
  ThreadSafeQueue<InferenceResponse>& response_queue_;
  MapProjector& map_projector_;
  InferenceBackend& inference_backend_;
  PipelineConfig config_;
  rclcpp::Logger logger_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detection_debug_pub_;
  std::unordered_map<
      std::string,
      rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr>
      additional_detection_debug_pubs_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr raw_detection_pub_;
  std::unordered_map<RequestId, PendingDebugFrame> pending_debug_frames_;
  std::deque<RequestId> pending_debug_order_;
  mutable std::mutex join_mutex_;
  std::unordered_map<FrameKey, PendingMapCommit, FrameKeyHash>
      pending_map_commits_;
  std::deque<FrameKey> pending_map_commit_order_;
  std::unordered_map<FrameKey, std::string, FrameKeyHash>
      cancelled_join_keys_;
  std::deque<FrameKey> cancelled_join_order_;
  FrameBundlePtr active_frame_;
  std::chrono::steady_clock::time_point active_join_started_ =
      std::chrono::steady_clock::time_point::min();
  RequestId active_request_id_ = 0;
  std::atomic_bool perception_busy_{false};
  bool shutdown_candidates_discarded_ = false;
  RequestId next_request_id_ = 1;
  FrameId last_frame_id_ = 0;
  RequestId last_request_id_ = 0;
  RequestId last_response_request_id_ = 0;
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
  std::uint64_t map_join_deadlines_ = 0;
  std::uint64_t map_join_failures_ = 0;
  std::uint64_t shutdown_candidates_cancelled_ = 0;
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
  double total_map_commit_wait_ms_ = 0.0;
  double total_projection_compute_ms_ = 0.0;
  double total_response_queue_dwell_ms_ = 0.0;
  double total_ingest_to_forward_ms_ = 0.0;
  double total_resize_ms_ = 0.0;
  double total_rgb_resize_ms_ = 0.0;
  double total_mask_resize_ms_ = 0.0;
  double total_roundtrip_ms_ = 0.0;
  double total_backend_ipc_ms_ = 0.0;
  double total_worker_ms_ = 0.0;
  double total_owl_ms_ = 0.0;
  double total_boxernet_ms_ = 0.0;
  std::uint64_t timing_response_count_ = 0;
};

}  // namespace roomie
