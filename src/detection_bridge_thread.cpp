#include "roomie/pipeline/detection_bridge_thread.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "roomie/pipeline/image_utils.hpp"
#include "roomie/utils/run_logger.hpp"
#include "roomie/utils/visualization_utils.hpp"

namespace roomie {
namespace {

std::string debugFrameKey(TimeNanoseconds time_ns, const std::string& camera_id) {
  return std::to_string(time_ns) + "\n" + camera_id;
}

double elapsedMs(std::chrono::steady_clock::time_point start,
                 std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

builtin_interfaces::msg::Time stampFromNanoseconds(TimeNanoseconds time_ns) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(time_ns / 1000000000LL);
  stamp.nanosec = static_cast<std::uint32_t>(time_ns % 1000000000LL);
  return stamp;
}

sensor_msgs::msg::Image imageMessageFromBuffer(const ImageBuffer& image,
                                               TimeNanoseconds time_ns,
                                               const std::string& frame_id) {
  sensor_msgs::msg::Image message;
  message.header.stamp = stampFromNanoseconds(time_ns);
  message.header.frame_id = frame_id;
  message.height = static_cast<std::uint32_t>(image.height);
  message.width = static_cast<std::uint32_t>(image.width);
  message.encoding = image.channels == 1 ? "mono8" : "rgb8";
  message.is_bigendian = false;
  message.step = static_cast<std::uint32_t>(image.width * image.channels);
  message.data = image.data;
  return message;
}

float overallScore(const RawDetection& detection) {
  return 0.5f * (detection.score_2d + detection.score_3d);
}

std::string rawDetectionText(const RawDetection& detection, bool show_score) {
  std::ostringstream stream;
  stream << (detection.label.empty() ? "object" : detection.label);
  if (show_score) {
    stream << " " << static_cast<int>(std::round(overallScore(detection) * 100.0f))
           << "%";
  }
  return stream.str();
}

int channelValue(float value) {
  return std::clamp(static_cast<int>(std::round(value * 255.0f)), 0, 255);
}

cv::Scalar colorScalar(const RgbColor& color, float scale = 1.0f) {
  return cv::Scalar(channelValue(color.r * scale),
                    channelValue(color.g * scale),
                    channelValue(color.b * scale));
}

std::string shortLabel(const std::string& label) {
  constexpr std::size_t kMaxChars = 22;
  if (label.size() <= kMaxChars) {
    return label;
  }
  return label.substr(0, kMaxChars - 1) + ".";
}

void drawStatusText(cv::Mat* image, const std::string& text, const cv::Scalar& color) {
  constexpr double kFontScale = 0.55;
  constexpr int kThickness = 1;
  int baseline = 0;
  const cv::Size text_size =
      cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, kFontScale, kThickness, &baseline);
  const cv::Rect background(8,
                            8,
                            std::min(text_size.width + 14, image->cols - 8),
                            text_size.height + baseline + 12);
  cv::rectangle(*image, background, cv::Scalar(20, 20, 20), cv::FILLED);
  cv::putText(*image,
              text,
              cv::Point(15, 8 + text_size.height + 5),
              cv::FONT_HERSHEY_SIMPLEX,
              kFontScale,
              color,
              kThickness,
              cv::LINE_AA);
}

void drawDetectionBoxes(cv::Mat* image,
                        const std::vector<Raw2dDetection>& detections,
                        bool ok,
                        const std::string& error) {
  if (!ok) {
    drawStatusText(image, "inference error", cv::Scalar(255, 80, 80));
    return;
  }
  if (detections.empty()) {
    drawStatusText(image, "2D detections: 0", cv::Scalar(220, 220, 220));
    return;
  }

  constexpr double kFontScale = 0.72;
  constexpr int kTextThickness = 2;
  constexpr int kBoxThickness = 3;
  for (const Raw2dDetection& detection : detections) {
    const int x0 = std::clamp(
        static_cast<int>(std::floor(std::min(detection.box_xyxy[0], detection.box_xyxy[2]))),
        0,
        image->cols - 1);
    const int x1 = std::clamp(
        static_cast<int>(std::ceil(std::max(detection.box_xyxy[0], detection.box_xyxy[2]))),
        0,
        image->cols - 1);
    const int y0 = std::clamp(
        static_cast<int>(std::floor(std::min(detection.box_xyxy[1], detection.box_xyxy[3]))),
        0,
        image->rows - 1);
    const int y1 = std::clamp(
        static_cast<int>(std::ceil(std::max(detection.box_xyxy[1], detection.box_xyxy[3]))),
        0,
        image->rows - 1);
    if (x1 <= x0 || y1 <= y0) {
      continue;
    }

    const RgbColor color = colorForLabel(detection.label, detection.semantic_id);
    const cv::Scalar line_color = colorScalar(color);
    cv::rectangle(*image, cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1)), line_color,
                  kBoxThickness, cv::LINE_AA);

    std::ostringstream text_stream;
    text_stream << shortLabel(detection.label.empty() ? "object" : detection.label)
                << " " << std::fixed << std::setprecision(2) << detection.score_2d;
    const std::string text = text_stream.str();
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, kFontScale, kTextThickness,
                        &baseline);
    const int bg_x0 = x0;
    const int bg_y0 = std::max(0, y0 - text_size.height - baseline - 10);
    const int bg_x1 = std::min(image->cols - 1, x0 + text_size.width + 10);
    const int bg_y1 = std::min(image->rows - 1, bg_y0 + text_size.height + baseline + 10);
    cv::rectangle(*image,
                  cv::Rect(cv::Point(bg_x0, bg_y0), cv::Point(bg_x1, bg_y1)),
                  colorScalar(color, 0.34f),
                  cv::FILLED);
    cv::putText(*image,
                text,
                cv::Point(bg_x0 + 5, bg_y1 - baseline - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                kFontScale,
                cv::Scalar(255, 255, 255),
                kTextThickness,
                cv::LINE_AA);
  }

  (void)error;
}

}  // namespace

DetectionBridgeThread::DetectionBridgeThread(
    rclcpp::Node& node,
    ThreadSafeQueue<DetectionFrame>& detection_queue,
    ThreadSafeQueue<InferenceResponse>& response_queue,
    MapProjector& map_projector,
    InferenceBackend& inference_backend,
    PipelineConfig config)
    : WorkerThread("detection_bridge_thread"),
      detection_queue_(detection_queue),
      response_queue_(response_queue),
      map_projector_(map_projector),
      inference_backend_(inference_backend),
      config_(std::move(config)),
      logger_(node.get_logger().get_child("detection_bridge")),
      last_request_time_(std::chrono::steady_clock::time_point::min()),
      last_status_log_time_(std::chrono::steady_clock::now()) {
  detection_debug_pub_ = node.create_publisher<sensor_msgs::msg::Image>(
      config_.detection_debug_image_topic,
      rclcpp::QoS(1).reliable());
  raw_detection_pub_ = node.create_publisher<visualization_msgs::msg::MarkerArray>(
      config_.raw_detections_topic,
      rclcpp::QoS(1).reliable());
  RCLCPP_INFO(logger_,
              "detection bridge enabled: debug_image=%s raw_3d=%s max_fps=%.2f "
              "min_patch_coverage=%.3f",
              config_.detection_debug_image_topic.c_str(),
              config_.raw_detections_topic.c_str(),
              config_.max_inference_fps,
              config_.min_patch_coverage_ratio);
}

void DetectionBridgeThread::run() {
  while (!stopRequested()) {
    forwardBackendResponses();

    DetectionFrame frame;
    if (!detection_queue_.waitPopFor(&frame, std::chrono::milliseconds(20))) {
      maybeLogStatus();
      continue;
    }
    ++frames_seen_;
    if (!readyForNextRequest()) {
      ++skipped_rate_;
      last_rate_skip_time_ns_ = frame.time_ns;
      if (requests_sent_ > 0) {
        last_rate_skip_since_request_ms_ =
            elapsedMs(last_request_time_, std::chrono::steady_clock::now());
      }
      maybeLogStatus();
      continue;
    }

    const auto projection_start = std::chrono::steady_clock::now();
    std::optional<PatchDepth> patch_depth = map_projector_.projectPatchDepth(frame);
    const double projection_ms =
        elapsedMs(projection_start, std::chrono::steady_clock::now());
    if (!patch_depth) {
      ++skipped_no_patch_;
      maybeLogStatus();
      continue;
    }
    if (!patch_depth->hasMinimumCoverage(config_.min_patch_coverage_ratio)) {
      ++skipped_low_coverage_;
      maybeLogStatus();
      continue;
    }

    const auto resize_start = std::chrono::steady_clock::now();
    InferenceRequest request = makeRequest(frame, std::move(*patch_depth));
    const double resize_ms = elapsedMs(resize_start, std::chrono::steady_clock::now());
    if (request.rgb_960.empty()) {
      ++skipped_resize_;
      maybeLogStatus();
      continue;
    }
    stashDebugFrame(request, projection_ms, resize_ms);
    const double project_total_ms = request.patch_depth.project_total_ms;
    const double snapshot_ms = request.patch_depth.snapshot_ms;
    const double surface_extract_ms = request.patch_depth.surface_extract_ms;
    const double cache_build_ms = request.patch_depth.cache_build_ms;
    const double frustum_filter_ms = request.patch_depth.frustum_filter_ms;
    const double projection_loop_ms = request.patch_depth.projection_loop_ms;
    const double projection_zbuffer_ms = request.patch_depth.projection_zbuffer_ms;
    if (inference_backend_.enqueueRequest(std::move(request))) {
      ++requests_sent_;
      total_projection_ms_ += projection_ms;
      total_project_total_ms_ += project_total_ms;
      total_snapshot_ms_ += snapshot_ms;
      total_surface_extract_ms_ += surface_extract_ms;
      total_cache_build_ms_ += cache_build_ms;
      total_frustum_filter_ms_ += frustum_filter_ms;
      total_projection_loop_ms_ += projection_loop_ms;
      total_projection_zbuffer_ms_ += projection_zbuffer_ms;
      total_resize_ms_ += resize_ms;
      last_request_time_ = std::chrono::steady_clock::now();
      maybeLogStatus();
    }
  }
}

bool DetectionBridgeThread::readyForNextRequest() {
  if (config_.max_inference_fps <= 0.0) {
    return true;
  }
  if (requests_sent_ == 0) {
    return true;
  }
  const auto now = std::chrono::steady_clock::now();
  const auto min_period = std::chrono::duration<double>(1.0 / config_.max_inference_fps);
  return now - last_request_time_ >= min_period;
}

InferenceRequest DetectionBridgeThread::makeRequest(const DetectionFrame& frame,
                                                    PatchDepth patch_depth) const {
  InferenceRequest request;
  request.time_ns = frame.time_ns;
  request.camera_id = frame.camera_id;
  request.rgb_960 =
      resizeBilinear(frame.rgb, config_.boxer_input_size, config_.boxer_input_size);
  request.mask_960 =
      resizeNearest(frame.robot_mask, config_.boxer_input_size, config_.boxer_input_size);
  request.patch_depth = std::move(patch_depth);
  request.intrinsics_960 =
      frame.intrinsics.scaledTo(config_.boxer_input_size, config_.boxer_input_size);
  request.T_world_camera = frame.T_world_camera;
  return request;
}

void DetectionBridgeThread::forwardBackendResponses() {
  InferenceResponse response;
  while (inference_backend_.tryPopResponse(&response)) {
    ++responses_seen_;
    std::optional<PendingDebugFrame> pending = takeDebugFrame(response);
    if (pending) {
      response.has_camera_pose = true;
      response.T_world_camera = pending->T_world_camera;
    }
    double roundtrip_ms = 0.0;
    if (pending) {
      roundtrip_ms = elapsedMs(pending->sent_time, std::chrono::steady_clock::now());
      total_roundtrip_ms_ += roundtrip_ms;
    }
    total_backend_ipc_ms_ += response.backend_ipc_ms;
    total_worker_ms_ += response.python_worker_ms;
    total_owl_ms_ += response.owl_ms;
    total_boxernet_ms_ += response.boxernet_ms;
    ++timing_response_count_;

    if (!response.ok) {
      ++response_errors_;
      RCLCPP_WARN(logger_,
                  "inference response error camera=%s t=%ld backend=%.1fms "
                  "worker=%.1fms owl=%.1fms boxernet=%.1fms project=%.1fms "
                  "snapshot=%.1fms surface=%.1fms frustum=%.1fms "
                  "proj_loop=%.1fms zbuffer=%.1fms "
                  "roundtrip=%.1fms "
                  "error=%s",
                  response.camera_id.c_str(),
                  static_cast<long>(response.time_ns),
                  response.backend_ipc_ms,
                  response.python_worker_ms,
                  response.owl_ms,
                  response.boxernet_ms,
                  pending ? pending->projection_ms : 0.0,
                  pending ? pending->snapshot_ms : 0.0,
                  pending ? pending->surface_extract_ms : 0.0,
                  pending ? pending->frustum_filter_ms : 0.0,
                  pending ? pending->projection_loop_ms : 0.0,
                  pending ? pending->projection_zbuffer_ms : 0.0,
                  roundtrip_ms,
                  response.error.c_str());
      RunLogger::logGlobal("detection",
                           "response_error camera=" + response.camera_id +
                               " t=" + std::to_string(response.time_ns) +
                               " backend_ipc_ms=" +
                               std::to_string(response.backend_ipc_ms) +
                               " worker_ms=" +
                               std::to_string(response.python_worker_ms) +
                               " owl_ms=" + std::to_string(response.owl_ms) +
                               " boxernet_ms=" +
                               std::to_string(response.boxernet_ms) +
                               " project_ms=" +
                               std::to_string(pending ? pending->projection_ms : 0.0) +
                               " snapshot_ms=" +
                               std::to_string(pending ? pending->snapshot_ms : 0.0) +
                               " surface_extract_ms=" +
                               std::to_string(pending ? pending->surface_extract_ms : 0.0) +
                               " frustum_filter_ms=" +
                               std::to_string(pending ? pending->frustum_filter_ms : 0.0) +
                               " projection_loop_ms=" +
                               std::to_string(pending ? pending->projection_loop_ms : 0.0) +
                               " projection_zbuffer_ms=" +
                               std::to_string(pending ? pending->projection_zbuffer_ms : 0.0) +
                               " roundtrip_ms=" + std::to_string(roundtrip_ms) +
                               " error=" + response.error);
    } else {
      RCLCPP_INFO(logger_,
                  "inference response camera=%s t=%ld filtered_2d=%zu raw_3d=%zu "
                  "project=%.1fms snapshot=%.1fms surface=%.1fms frustum=%.1fms "
                  "proj_loop=%.1fms zbuffer=%.1fms resize=%.1fms backend=%.1fms worker=%.1fms "
                  "owl=%.1fms boxernet=%.1fms roundtrip=%.1fms",
                  response.camera_id.c_str(),
                  static_cast<long>(response.time_ns),
                  response.filtered_2d_detections.size(),
                  response.detections.size(),
                  pending ? pending->projection_ms : 0.0,
                  pending ? pending->snapshot_ms : 0.0,
                  pending ? pending->surface_extract_ms : 0.0,
                  pending ? pending->frustum_filter_ms : 0.0,
                  pending ? pending->projection_loop_ms : 0.0,
                  pending ? pending->projection_zbuffer_ms : 0.0,
                  pending ? pending->resize_ms : 0.0,
                  response.backend_ipc_ms,
                  response.python_worker_ms,
                  response.owl_ms,
                  response.boxernet_ms,
                  roundtrip_ms);
      std::ostringstream stream;
      stream << std::fixed << std::setprecision(2)
             << "response camera=" << response.camera_id
             << " t=" << response.time_ns
             << " filtered_2d=" << response.filtered_2d_detections.size()
             << " raw_3d=" << response.detections.size()
             << " patch_coverage=" << (pending ? pending->patch_coverage : 0.0f)
             << " valid_patches=" << (pending ? pending->valid_patches : 0)
             << " projected_points=" << (pending ? pending->projected_points : 0)
             << " map_version=" << (pending ? pending->map_version : 0)
             << " source_surface_points=" << (pending ? pending->source_surface_points : 0)
             << " tsdf_blocks=" << (pending ? pending->source_tsdf_blocks : 0)
             << " voxels_scanned=" << (pending ? pending->source_voxels_scanned : 0)
             << " selected_blocks=" << (pending ? pending->source_selected_blocks : 0)
             << " cached_surface_points="
             << (pending ? pending->source_cached_surface_points : 0)
             << " cache_rebuilds=" << (pending ? pending->surface_cache_rebuilds : 0)
             << " cache_ready=" << ((pending && pending->surface_cache_ready) ? "true" : "false")
             << " view_filtered=" << ((pending && pending->view_filtered) ? "true" : "false")
             << " project_ms=" << (pending ? pending->projection_ms : 0.0)
             << " map_project_total_ms=" << (pending ? pending->project_total_ms : 0.0)
             << " snapshot_ms=" << (pending ? pending->snapshot_ms : 0.0)
             << " surface_extract_ms=" << (pending ? pending->surface_extract_ms : 0.0)
             << " cache_build_ms=" << (pending ? pending->cache_build_ms : 0.0)
             << " frustum_filter_ms=" << (pending ? pending->frustum_filter_ms : 0.0)
             << " projection_loop_ms=" << (pending ? pending->projection_loop_ms : 0.0)
             << " projection_zbuffer_ms=" << (pending ? pending->projection_zbuffer_ms : 0.0)
             << " resize_ms=" << (pending ? pending->resize_ms : 0.0)
             << " backend_ipc_ms=" << response.backend_ipc_ms
             << " worker_ms=" << response.python_worker_ms
             << " preprocess_ms=" << response.python_preprocess_ms
             << " owl_ms=" << response.owl_ms
             << " robot_filter_ms=" << response.robot_filter_ms
             << " boxernet_ms=" << response.boxernet_ms
             << " postprocess_ms=" << response.python_postprocess_ms
             << " roundtrip_ms=" << roundtrip_ms;
      RunLogger::logGlobal("detection", stream.str());
    }
    publishDetectionDebugImage(response, pending);
    publishRawDetectionMarkers(response);
    response_queue_.pushDropOldest(std::move(response));
  }
}

void DetectionBridgeThread::stashDebugFrame(const InferenceRequest& request,
                                            double projection_ms,
                                            double resize_ms) {
  if (request.rgb_960.empty()) {
    return;
  }
  const std::string key = debugFrameKey(request.time_ns, request.camera_id);
  PendingDebugFrame debug;
  debug.image = request.rgb_960;
  debug.sent_time = std::chrono::steady_clock::now();
  debug.T_world_camera = request.T_world_camera;
  debug.projection_ms = projection_ms;
  debug.resize_ms = resize_ms;
  debug.patch_coverage = request.patch_depth.coverageRatio();
  debug.valid_patches = request.patch_depth.valid_patches;
  debug.projected_points = request.patch_depth.projected_points;
  debug.map_version = request.patch_depth.map_version;
  debug.source_surface_points = request.patch_depth.source_surface_points;
  debug.source_tsdf_blocks = request.patch_depth.source_tsdf_blocks;
  debug.source_voxels_scanned = request.patch_depth.source_voxels_scanned;
  debug.source_selected_blocks = request.patch_depth.source_selected_blocks;
  debug.source_cached_surface_points = request.patch_depth.source_cached_surface_points;
  debug.surface_cache_rebuilds = request.patch_depth.surface_cache_rebuilds;
  debug.project_total_ms = request.patch_depth.project_total_ms;
  debug.snapshot_ms = request.patch_depth.snapshot_ms;
  debug.surface_extract_ms = request.patch_depth.surface_extract_ms;
  debug.cache_build_ms = request.patch_depth.cache_build_ms;
  debug.frustum_filter_ms = request.patch_depth.frustum_filter_ms;
  debug.projection_loop_ms = request.patch_depth.projection_loop_ms;
  debug.projection_zbuffer_ms = request.patch_depth.projection_zbuffer_ms;
  debug.surface_cache_ready = request.patch_depth.surface_cache_ready;
  debug.view_filtered = request.patch_depth.view_filtered;
  pending_debug_frames_[key] = std::move(debug);
  pending_debug_order_.push_back(key);

  const std::size_t max_debug_frames =
      std::max<std::size_t>(8, config_.inference_request_queue_size +
                                   config_.inference_response_queue_size + 4);
  while (pending_debug_order_.size() > max_debug_frames) {
    pending_debug_frames_.erase(pending_debug_order_.front());
    pending_debug_order_.pop_front();
  }
}

std::optional<DetectionBridgeThread::PendingDebugFrame> DetectionBridgeThread::takeDebugFrame(
    const InferenceResponse& response) {
  const std::string key = debugFrameKey(response.time_ns, response.camera_id);
  auto it = pending_debug_frames_.find(key);
  if (it == pending_debug_frames_.end()) {
    return std::nullopt;
  }
  PendingDebugFrame image = std::move(it->second);
  pending_debug_frames_.erase(it);
  return image;
}

void DetectionBridgeThread::publishDetectionDebugImage(
    const InferenceResponse& response,
    const std::optional<PendingDebugFrame>& pending) {
  if (!detection_debug_pub_) {
    return;
  }
  if (!pending || pending->image.empty()) {
    return;
  }
  ImageBuffer image = pending->image;
  if (image.channels != 3) {
    return;
  }

  cv::Mat rgb(image.height, image.width, CV_8UC3, image.data.data());
  drawDetectionBoxes(&rgb, response.filtered_2d_detections, response.ok, response.error);

  detection_debug_pub_->publish(
      imageMessageFromBuffer(image, response.time_ns, response.camera_id));
}

void DetectionBridgeThread::publishRawDetectionMarkers(const InferenceResponse& response) {
  if (!raw_detection_pub_) {
    return;
  }

  visualization_msgs::msg::MarkerArray markers;

  visualization_msgs::msg::Marker clear;
  clear.header.frame_id = config_.world_frame;
  clear.header.stamp = stampFromNanoseconds(response.time_ns);
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  markers.markers.push_back(clear);

  if (response.ok) {
    int marker_id = 1;
    for (const RawDetection& detection : response.detections) {
      const RgbColor color = colorForLabel(detection.label, detection.semantic_id);
      visualization_msgs::msg::Marker box;
      box.header.frame_id = config_.world_frame;
      box.header.stamp = clear.header.stamp;
      box.ns = "roomie_raw_detections";
      box.id = marker_id++;
      box.type = visualization_msgs::msg::Marker::LINE_LIST;
      box.action = visualization_msgs::msg::Marker::ADD;
      box.pose.orientation.w = 1.0;
      box.scale.x = 0.025;
      box.color.r = color.r;
      box.color.g = color.g;
      box.color.b = color.b;
      box.color.a = 0.95f;
      fillLineListFromCorners(
          yawObbCorners(detection.center_world, detection.size_m, detection.yaw_rad),
          &box);
      markers.markers.push_back(box);

      visualization_msgs::msg::Marker text;
      text.header.frame_id = config_.world_frame;
      text.header.stamp = clear.header.stamp;
      text.ns = "roomie_raw_detection_labels";
      text.id = marker_id++;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.position.x = detection.center_world.x();
      text.pose.position.y = detection.center_world.y();
      text.pose.position.z = detection.center_world.z();
      text.pose.position.z += std::max(0.05f, detection.size_m.z() * 0.65f);
      text.scale.z = 0.10;
      text.color.r = 1.0f;
      text.color.g = 0.95f;
      text.color.b = 0.75f;
      text.color.a = 1.0f;
      text.text = rawDetectionText(detection, config_.show_3d_label_score);
      markers.markers.push_back(text);
    }
  }

  raw_detection_pub_->publish(markers);
}

void DetectionBridgeThread::maybeLogStatus() {
  const auto now = std::chrono::steady_clock::now();
  if (now - last_status_log_time_ <
      std::chrono::duration<double>(config_.file_logging_period_sec)) {
    return;
  }
  last_status_log_time_ = now;
  std::ostringstream status;
  const std::uint64_t delta_frames = frames_seen_ - last_logged_frames_seen_;
  const std::uint64_t delta_skip_rate = skipped_rate_ - last_logged_skipped_rate_;
  const std::uint64_t delta_sent = requests_sent_ - last_logged_requests_sent_;
  const std::uint64_t delta_responses = responses_seen_ - last_logged_responses_seen_;
  const double request_count = std::max<std::uint64_t>(1, requests_sent_);
  const double response_count = std::max<std::uint64_t>(1, timing_response_count_);
  status << "status frames=" << frames_seen_
         << " delta_frames=" << delta_frames
         << " sent=" << requests_sent_
         << " delta_sent=" << delta_sent
         << " responses=" << responses_seen_
         << " delta_responses=" << delta_responses
         << " errors=" << response_errors_
         << " skip_rate=" << skipped_rate_
         << " delta_skip_rate=" << delta_skip_rate
         << " no_patch=" << skipped_no_patch_
         << " low_coverage=" << skipped_low_coverage_
         << " resize_fail=" << skipped_resize_
         << " last_rate_skip_t=" << last_rate_skip_time_ns_
         << " last_rate_skip_since_request_ms=" << std::fixed << std::setprecision(2)
         << last_rate_skip_since_request_ms_
         << " avg_project_ms=" << total_projection_ms_ / request_count
         << " avg_map_project_total_ms=" << total_project_total_ms_ / request_count
         << " avg_snapshot_ms=" << total_snapshot_ms_ / request_count
         << " avg_surface_extract_ms=" << total_surface_extract_ms_ / request_count
         << " avg_cache_build_ms=" << total_cache_build_ms_ / request_count
         << " avg_frustum_filter_ms=" << total_frustum_filter_ms_ / request_count
         << " avg_projection_loop_ms=" << total_projection_loop_ms_ / request_count
         << " avg_projection_zbuffer_ms=" << total_projection_zbuffer_ms_ / request_count
         << " avg_resize_ms=" << total_resize_ms_ / request_count
         << " avg_backend_ipc_ms=" << total_backend_ipc_ms_ / response_count
         << " avg_worker_ms=" << total_worker_ms_ / response_count
         << " avg_owl_ms=" << total_owl_ms_ / response_count
         << " avg_boxernet_ms=" << total_boxernet_ms_ / response_count
         << " avg_roundtrip_ms=" << total_roundtrip_ms_ / response_count;
  RCLCPP_INFO(logger_,
              "detection status frames=%lu sent=%lu responses=%lu errors=%lu "
              "skip_rate=%lu(+%lu) no_patch=%lu low_coverage=%lu resize_fail=%lu "
              "avg_project=%.1fms avg_snapshot=%.1fms avg_surface=%.1fms "
              "avg_frustum=%.1fms avg_proj_loop=%.1fms avg_zbuffer=%.1fms "
              "avg_owl=%.1fms avg_boxernet=%.1fms "
              "avg_roundtrip=%.1fms",
              static_cast<unsigned long>(frames_seen_),
              static_cast<unsigned long>(requests_sent_),
              static_cast<unsigned long>(responses_seen_),
              static_cast<unsigned long>(response_errors_),
              static_cast<unsigned long>(skipped_rate_),
              static_cast<unsigned long>(delta_skip_rate),
              static_cast<unsigned long>(skipped_no_patch_),
              static_cast<unsigned long>(skipped_low_coverage_),
              static_cast<unsigned long>(skipped_resize_),
              total_projection_ms_ / request_count,
              total_snapshot_ms_ / request_count,
              total_surface_extract_ms_ / request_count,
              total_frustum_filter_ms_ / request_count,
              total_projection_loop_ms_ / request_count,
              total_projection_zbuffer_ms_ / request_count,
              total_owl_ms_ / response_count,
              total_boxernet_ms_ / response_count,
              total_roundtrip_ms_ / response_count);
  RunLogger::logGlobal("detection", status.str());
  last_logged_frames_seen_ = frames_seen_;
  last_logged_skipped_rate_ = skipped_rate_;
  last_logged_requests_sent_ = requests_sent_;
  last_logged_responses_seen_ = responses_seen_;
}

}  // namespace roomie
