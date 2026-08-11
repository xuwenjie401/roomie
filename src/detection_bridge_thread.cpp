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

double elapsedMs(std::chrono::steady_clock::time_point start,
                 std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

const char* mapModeName(MapMode mode) {
  return mode == MapMode::kFrozen ? "frozen" : "online";
}

std::string provenanceLogFields(const FrameProvenance& provenance) {
  std::ostringstream stream;
  stream << "run_id=" << runIdString(provenance.run_id)
         << " frame_id=" << provenance.frame_id
         << " request_id=" << provenance.request_id
         << " sensor_time_ns=" << provenance.sensor_time_ns
         << " map_mode=" << mapModeName(provenance.map_mode)
         << " includes_current_frame="
         << (provenance.includes_current_frame ? "true" : "false")
         << " causality_verified="
         << (provenance.causality_verified ? "true" : "false")
         << " map_epoch=" << runIdString(provenance.map.map_epoch)
         << " map_revision=" << provenance.map.map_revision
         << " integrated_through_ns=" << provenance.map.integrated_through_ns
         << " surface_epoch=" << runIdString(provenance.surface.map_epoch)
         << " surface_revision=" << provenance.surface.surface_revision
         << " surface_source_map_revision="
         << provenance.surface.source_map_revision;
  return stream.str();
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
    ThreadSafeQueue<FrameBundlePtr>& detection_queue,
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
  if (config_.detection_enabled) {
    RCLCPP_INFO(logger_,
                "detection bridge enabled: debug_image=%s raw_3d=%s max_fps=%.2f "
                "min_patch_coverage=%.3f",
                config_.detection_debug_image_topic.c_str(),
                config_.raw_detections_topic.c_str(),
                config_.max_inference_fps,
                config_.min_patch_coverage_ratio);
  } else {
    RCLCPP_INFO(logger_, "detection bridge disabled by config");
  }
}

void DetectionBridgeThread::run() {
  while (true) {
    forwardBackendResponses();

    if (stopRequested()) {
      discardUnstartedCandidatesForShutdown();
      if (active_request_id_ != 0 && inference_backend_.idle()) {
        RunLogger::logGlobal(
            "detection",
            "shutdown_orphaned_pending request_id=" +
                std::to_string(active_request_id_) +
                " reason=backend_idle_without_terminal_response");
        pending_debug_frames_.erase(active_request_id_);
        pending_debug_order_.erase(
            std::remove(pending_debug_order_.begin(),
                        pending_debug_order_.end(), active_request_id_),
            pending_debug_order_.end());
        active_request_id_ = 0;
        perception_busy_.store(false);
      }
      if (active_request_id_ == 0 && inference_backend_.idle()) {
        forwardBackendResponses();
        if (!pending_debug_frames_.empty()) {
          RunLogger::logGlobal(
              "detection",
              "shutdown_orphaned_pending count=" +
                  std::to_string(pending_debug_frames_.size()) +
                  " reason=backend_idle_without_terminal_response");
          pending_debug_frames_.clear();
          pending_debug_order_.clear();
        }
        perception_busy_.store(false);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      maybeLogStatus();
      continue;
    }

    admitNextCandidate();
    pollActiveCandidate();
    maybeLogStatus();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  forwardBackendResponses();
}

void DetectionBridgeThread::onStopRequested() {
  // Start backend cancellation before waiting for its idle barrier. The
  // backend leaves terminal responses drainable while this bridge exits.
  inference_backend_.beginShutdown();
}

void DetectionBridgeThread::onMapCommit(const MapCommit& commit) {
  if (!commit.perception_candidate || commit.frame_id == 0 ||
      !commit.frame_bundle) {
    return;
  }
  const FrameKey key{commit.run_id, commit.frame_id};
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(join_mutex_);
  pruneMapCommitsLocked(now);
  if (cancelled_join_keys_.count(key) > 0 || commit.due_time <= now) {
    RunLogger::logGlobal(
        "detection",
        "map_commit_join_ignored run_id=" + runIdString(key.run_id) +
            " frame_id=" + std::to_string(key.frame_id) + " reason=" +
            (cancelled_join_keys_.count(key) > 0 ? "superseded" :
                                                   "deadline"));
    return;
  }
  if (pending_map_commits_.count(key) == 0) {
    pending_map_commit_order_.push_back(key);
  }
  pending_map_commits_[key] = PendingMapCommit{commit, now};
  const std::size_t limit =
      std::max<std::size_t>(1, config_.pending_frame_limit);
  while (pending_map_commit_order_.size() > limit) {
    const FrameKey evicted = pending_map_commit_order_.front();
    pending_map_commit_order_.pop_front();
    pending_map_commits_.erase(evicted);
    RunLogger::logGlobal(
        "detection",
        "map_commit_join_evicted run_id=" + runIdString(evicted.run_id) +
            " frame_id=" + std::to_string(evicted.frame_id) +
            " reason=pending_limit");
  }
}

void DetectionBridgeThread::cancelPendingCandidate(
    const FrameBundlePtr& frame, const std::string& reason) {
  if (!frame) {
    return;
  }
  const FrameKey key{frame->provenance.run_id, frame->provenance.frame_id};
  std::lock_guard<std::mutex> lock(join_mutex_);
  pending_map_commits_.erase(key);
  pending_map_commit_order_.erase(
      std::remove(pending_map_commit_order_.begin(),
                  pending_map_commit_order_.end(), key),
      pending_map_commit_order_.end());
  if (cancelled_join_keys_.count(key) == 0) {
    cancelled_join_order_.push_back(key);
  }
  cancelled_join_keys_[key] = reason.empty() ? "cancelled" : reason;
  const std::size_t limit =
      std::max<std::size_t>(8, config_.pending_frame_limit * 2U);
  while (cancelled_join_order_.size() > limit) {
    cancelled_join_keys_.erase(cancelled_join_order_.front());
    cancelled_join_order_.pop_front();
  }
}

bool DetectionBridgeThread::perceptionBusy() const {
  return perception_busy_.load();
}

void DetectionBridgeThread::admitNextCandidate() {
  if (active_frame_ || active_request_id_ != 0) {
    return;
  }
  FrameBundlePtr frame;
  if (!detection_queue_.tryPop(&frame) || !frame) {
    return;
  }
  ++frames_seen_;
  const auto now = std::chrono::steady_clock::now();
  const ChannelStats queue_stats = detection_queue_.stats();
  if (now >= frame->due_time) {
    ++map_join_deadlines_;
    map_projector_.cancelPerceptionCandidate(frame,
                                             "deadline_before_join");
    RunLogger::logGlobal(
        "detection",
        "perception_drop " + provenanceLogFields(frame->provenance) +
            " reason=deadline_before_join detection_queue_wait_ms=" +
            std::to_string(std::chrono::duration<double, std::milli>(
                               queue_stats.last_dequeue_age)
                               .count()));
    return;
  }
  active_frame_ = std::move(frame);
  active_join_started_ = now;
  perception_busy_.store(true);
  RunLogger::logGlobal(
      "detection",
      "map_join_started " + provenanceLogFields(active_frame_->provenance) +
          " detection_queue_wait_ms=" +
          std::to_string(std::chrono::duration<double, std::milli>(
                             queue_stats.last_dequeue_age)
                             .count()));
}

void DetectionBridgeThread::pollActiveCandidate() {
  if (!active_frame_ || active_request_id_ != 0) {
    return;
  }
  const FrameKey active_key{active_frame_->provenance.run_id,
                            active_frame_->provenance.frame_id};
  std::string cancellation_reason;
  {
    std::lock_guard<std::mutex> lock(join_mutex_);
    const auto cancelled = cancelled_join_keys_.find(active_key);
    if (cancelled != cancelled_join_keys_.end()) {
      cancellation_reason = cancelled->second;
      cancelled_join_keys_.erase(cancelled);
      cancelled_join_order_.erase(
          std::remove(cancelled_join_order_.begin(),
                      cancelled_join_order_.end(), active_key),
          cancelled_join_order_.end());
    }
  }
  if (!cancellation_reason.empty()) {
    finishActiveCandidate("cancelled_" + cancellation_reason);
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  if (now >= active_frame_->due_time) {
    ++map_join_deadlines_;
    map_projector_.cancelPerceptionCandidate(active_frame_,
                                             "map_commit_deadline");
    RunLogger::logGlobal(
        "detection",
        "perception_drop " + provenanceLogFields(active_frame_->provenance) +
            " reason=map_commit_deadline map_commit_wait_ms=" +
            std::to_string(elapsedMs(active_join_started_, now)));
    finishActiveCandidate("map_commit_deadline");
    return;
  }

  if (active_frame_->provenance.map_mode == MapMode::kFrozen ||
      !active_frame_->perception_candidate) {
    processReadyCandidate(active_frame_, nullptr, 0.0, 0.0);
    return;
  }

  std::optional<PendingMapCommit> ready = takeMapCommit(*active_frame_);
  if (!ready) {
    return;
  }
  if (!ready->commit.success) {
    ++map_join_failures_;
    map_projector_.cancelPerceptionCandidate(active_frame_,
                                             "map_commit_failed");
    RunLogger::logGlobal(
        "detection",
        "perception_drop " + provenanceLogFields(active_frame_->provenance) +
            " reason=map_commit_failed error=" + ready->commit.error);
    finishActiveCandidate("map_commit_failed");
    return;
  }
  const auto published_at =
      ready->commit.published_at ==
              std::chrono::steady_clock::time_point::min()
          ? ready->received_at
          : ready->commit.published_at;
  const double wait_ms =
      published_at > active_join_started_
          ? elapsedMs(active_join_started_, published_at)
          : 0.0;
  const double latency_ms =
      published_at > active_frame_->ingest_time
          ? elapsedMs(active_frame_->ingest_time, published_at)
          : 0.0;
  processReadyCandidate(active_frame_, &ready->commit, wait_ms, latency_ms);
}

void DetectionBridgeThread::processReadyCandidate(
    FrameBundlePtr frame,
    const MapCommit* commit,
    double map_commit_wait_ms,
    double map_commit_latency_ms) {
  const auto projection_start = std::chrono::steady_clock::now();
  std::optional<PatchDepth> patch_depth =
      commit ? map_projector_.projectPatchDepth(*frame, *commit)
             : (frame->provenance.map_mode == MapMode::kOnline
                    ? map_projector_.projectLatestPatchDepth(*frame)
                    : map_projector_.projectPatchDepth(*frame));
  const double projection_ms =
      elapsedMs(projection_start, std::chrono::steady_clock::now());
  if (!patch_depth) {
    ++skipped_no_patch_;
    finishActiveCandidate("no_patch_depth");
    return;
  }
  patch_depth->map_commit_wait_ms = map_commit_wait_ms;
  if (patch_depth->projection_compute_ms <= 0.0) {
    patch_depth->projection_compute_ms =
        patch_depth->frustum_filter_ms + patch_depth->projection_loop_ms +
        patch_depth->projection_zbuffer_ms;
  }
  if (!patch_depth->hasMinimumCoverage(config_.min_patch_coverage_ratio)) {
    ++skipped_low_coverage_;
    finishActiveCandidate("low_patch_coverage");
    return;
  }

  RequestBuildTiming request_build_timing;
  const auto resize_start = std::chrono::steady_clock::now();
  InferenceRequest request = makeRequest(*frame, std::move(*patch_depth),
                                         &request_build_timing);
  const double resize_ms =
      elapsedMs(resize_start, std::chrono::steady_clock::now());
  if (request.rgb_960.empty()) {
    ++skipped_resize_;
    finishActiveCandidate("resize_failed");
    return;
  }
  stashDebugFrame(request, projection_ms, resize_ms, request_build_timing);
  PendingDebugFrame& debug =
      pending_debug_frames_.at(request.provenance.request_id);
  debug.map_commit_wait_ms = map_commit_wait_ms;
  debug.map_commit_latency_ms = map_commit_latency_ms;
  debug.projection_compute_ms = request.patch_depth.projection_compute_ms;
  const FrameProvenance request_provenance = request.provenance;
  const auto request_due_time = request.due_time;
  const double project_total_ms = request.patch_depth.project_total_ms;
  const double snapshot_ms = request.patch_depth.snapshot_ms;
  const double surface_extract_ms = request.patch_depth.surface_extract_ms;
  const double cache_build_ms = request.patch_depth.cache_build_ms;
  const double frustum_filter_ms = request.patch_depth.frustum_filter_ms;
  const double projection_loop_ms = request.patch_depth.projection_loop_ms;
  const double projection_zbuffer_ms = request.patch_depth.projection_zbuffer_ms;
  const double projection_compute_ms = request.patch_depth.projection_compute_ms;
  PushResult<InferenceRequest> enqueue_result =
      inference_backend_.enqueueRequest(std::move(request));
  if (enqueue_result.replaced_item) {
    const RequestId superseded_request_id =
        enqueue_result.replaced_item->provenance.request_id;
    pending_debug_frames_.erase(superseded_request_id);
    pending_debug_order_.erase(
        std::remove(pending_debug_order_.begin(),
                    pending_debug_order_.end(), superseded_request_id),
        pending_debug_order_.end());
    RunLogger::logGlobal(
        "detection",
        "request_superseded request_id=" +
            std::to_string(superseded_request_id) + " by_request_id=" +
            std::to_string(request_provenance.request_id));
  }
  if (enqueue_result.accepted()) {
    ++requests_sent_;
    active_request_id_ = request_provenance.request_id;
    active_frame_.reset();
    last_frame_id_ = request_provenance.frame_id;
    last_request_id_ = request_provenance.request_id;
    total_projection_ms_ += projection_ms;
    total_project_total_ms_ += project_total_ms;
    total_snapshot_ms_ += snapshot_ms;
    total_surface_extract_ms_ += surface_extract_ms;
    total_cache_build_ms_ += cache_build_ms;
    total_frustum_filter_ms_ += frustum_filter_ms;
    total_projection_loop_ms_ += projection_loop_ms;
    total_projection_zbuffer_ms_ += projection_zbuffer_ms;
    total_map_commit_wait_ms_ += map_commit_wait_ms;
    total_projection_compute_ms_ += projection_compute_ms;
    total_resize_ms_ += resize_ms;
    total_rgb_resize_ms_ += request_build_timing.rgb_resize_ms;
    total_mask_resize_ms_ += request_build_timing.mask_resize_ms;
    last_request_time_ = std::chrono::steady_clock::now();
    const double due_in_ms = elapsedMs(last_request_time_, request_due_time);
    RunLogger::logGlobal(
        "detection",
        "request " + provenanceLogFields(request_provenance) +
            " due_in_ms=" + std::to_string(due_in_ms) +
            " map_commit_wait_ms=" + std::to_string(map_commit_wait_ms) +
            " map_commit_latency_ms=" +
            std::to_string(map_commit_latency_ms) +
            " projection_compute_ms=" +
            std::to_string(projection_compute_ms) + " projection_ms=" +
            std::to_string(projection_ms) + " resize_ms=" +
            std::to_string(resize_ms) + " rgb_resize_ms=" +
            std::to_string(request_build_timing.rgb_resize_ms) +
            " mask_resize_ms=" +
            std::to_string(request_build_timing.mask_resize_ms));
  } else {
    pending_debug_frames_.erase(request_provenance.request_id);
    pending_debug_order_.erase(
        std::remove(pending_debug_order_.begin(),
                    pending_debug_order_.end(),
                    request_provenance.request_id),
        pending_debug_order_.end());
    RunLogger::logGlobal(
        "detection",
        "request_rejected " + provenanceLogFields(request_provenance));
    finishActiveCandidate("backend_request_rejected");
  }
}

void DetectionBridgeThread::finishActiveCandidate(const std::string& reason) {
  if (active_frame_) {
    RunLogger::logGlobal(
        "detection",
        "perception_terminal " +
            provenanceLogFields(active_frame_->provenance) + " reason=" +
            reason);
  }
  active_frame_.reset();
  active_join_started_ = std::chrono::steady_clock::time_point::min();
  if (active_request_id_ == 0) {
    perception_busy_.store(false);
  }
}

std::optional<DetectionBridgeThread::PendingMapCommit>
DetectionBridgeThread::takeMapCommit(const FrameBundle& frame) {
  const FrameKey key{frame.provenance.run_id, frame.provenance.frame_id};
  std::lock_guard<std::mutex> lock(join_mutex_);
  pruneMapCommitsLocked(std::chrono::steady_clock::now());
  auto it = pending_map_commits_.find(key);
  if (it == pending_map_commits_.end()) {
    return std::nullopt;
  }
  PendingMapCommit ready = std::move(it->second);
  pending_map_commits_.erase(it);
  pending_map_commit_order_.erase(
      std::remove(pending_map_commit_order_.begin(),
                  pending_map_commit_order_.end(), key),
      pending_map_commit_order_.end());
  if (!ready.commit.frame_bundle ||
      ready.commit.frame_bundle.get() != &frame) {
    ready.commit.success = false;
    ready.commit.error = "map commit bundle identity mismatch";
  }
  return ready;
}

void DetectionBridgeThread::pruneMapCommitsLocked(
    std::chrono::steady_clock::time_point now) {
  for (auto it = pending_map_commit_order_.begin();
       it != pending_map_commit_order_.end();) {
    auto commit = pending_map_commits_.find(*it);
    if (commit == pending_map_commits_.end()) {
      it = pending_map_commit_order_.erase(it);
      continue;
    }
    if (commit->second.commit.due_time <= now) {
      pending_map_commits_.erase(commit);
      it = pending_map_commit_order_.erase(it);
      continue;
    }
    ++it;
  }
}

void DetectionBridgeThread::discardUnstartedCandidatesForShutdown() {
  if (shutdown_candidates_discarded_) {
    return;
  }
  shutdown_candidates_discarded_ = true;
  if (active_frame_) {
    map_projector_.cancelPerceptionCandidate(active_frame_,
                                             "shutdown_pending_join");
    ++shutdown_candidates_cancelled_;
    finishActiveCandidate("shutdown_pending_join");
  }
  FrameBundlePtr queued;
  while (detection_queue_.tryPop(&queued)) {
    if (!queued) {
      continue;
    }
    map_projector_.cancelPerceptionCandidate(queued,
                                             "shutdown_not_started");
    cancelPendingCandidate(queued, "shutdown_not_started");
    ++shutdown_candidates_cancelled_;
    RunLogger::logGlobal(
        "detection",
        "perception_terminal " + provenanceLogFields(queued->provenance) +
            " reason=shutdown_not_started");
  }
  {
    std::lock_guard<std::mutex> lock(join_mutex_);
    pending_map_commits_.clear();
    pending_map_commit_order_.clear();
  }
  perception_busy_.store(active_request_id_ != 0);
}

RequestId DetectionBridgeThread::allocateRequestId() {
  RequestId request_id = next_request_id_++;
  if (request_id == 0) {
    request_id = next_request_id_++;
  }
  return request_id;
}

InferenceRequest DetectionBridgeThread::makeRequest(
    const FrameBundle& frame,
    PatchDepth patch_depth,
    RequestBuildTiming* timing) {
  InferenceRequest request;
  if (timing != nullptr) {
    *timing = RequestBuildTiming{};
  }
  request.time_ns = frame.provenance.sensor_time_ns;
  request.provenance = frame.provenance;
  if (request.provenance.sensor_time_ns == 0) {
    request.provenance.sensor_time_ns = request.time_ns;
  }

  // A projector may already have attached a non-causal map/surface stamp. Keep
  // it only when it demonstrably belongs to this exact logical frame.
  const FrameProvenance& projected_provenance = patch_depth.provenance;
  if (projected_provenance.run_id == request.provenance.run_id &&
      projected_provenance.frame_id != 0 &&
      projected_provenance.frame_id == request.provenance.frame_id) {
    request.provenance.map_mode = projected_provenance.map_mode;
    request.provenance.includes_current_frame =
        projected_provenance.includes_current_frame;
    request.provenance.causality_verified =
        projected_provenance.causality_verified;
    request.provenance.map = projected_provenance.map;
    request.provenance.surface = projected_provenance.surface;
  }
  request.provenance.request_id = allocateRequestId();
  request.ingest_time = frame.ingest_time;
  request.due_time = frame.due_time;
  request.camera_id = frame.camera_id;
  if (frame.rgb) {
    const auto started = std::chrono::steady_clock::now();
    request.rgb_960 = resizeBilinear(
        *frame.rgb, config_.boxer_input_size, config_.boxer_input_size);
    if (timing != nullptr) {
      timing->rgb_resize_ms =
          elapsedMs(started, std::chrono::steady_clock::now());
    }
  }
  if (frame.robot_mask) {
    const auto started = std::chrono::steady_clock::now();
    request.mask_960 = resizeNearest(
        *frame.robot_mask, config_.boxer_input_size, config_.boxer_input_size);
    if (timing != nullptr) {
      timing->mask_resize_ms =
          elapsedMs(started, std::chrono::steady_clock::now());
    }
  }
  request.patch_depth = std::move(patch_depth);
  request.patch_depth.provenance = request.provenance;
  request.intrinsics_960 =
      frame.intrinsics.scaledTo(config_.boxer_input_size, config_.boxer_input_size);
  request.T_world_camera = frame.T_world_camera;
  auto visibility = std::make_shared<VisibilityContext>();
  visibility->depth = frame.depth;
  visibility->robot_mask = frame.robot_mask;
  visibility->intrinsics = frame.intrinsics;
  if (visibility->valid()) {
    request.visibility_context = std::move(visibility);
  }
  return request;
}

void DetectionBridgeThread::forwardBackendResponses() {
  InferenceResponse response;
  while (inference_backend_.tryPopResponse(&response)) {
    const auto forward_time = std::chrono::steady_clock::now();
    const ChannelStats backend_response_stats =
        inference_backend_.responseChannelStats();
    const double response_queue_dwell_ms =
        std::chrono::duration<double, std::milli>(
            backend_response_stats.last_dequeue_age)
            .count();
    ++responses_seen_;
    std::optional<PendingDebugFrame> pending = takeDebugFrame(response);
    if (pending) {
      if (response.provenance.request_id != 0 &&
          !(response.provenance == pending->provenance)) {
        RCLCPP_WARN(logger_,
                    "inference response provenance mismatch: response={%s} expected={%s}",
                    provenanceLogFields(response.provenance).c_str(),
                    provenanceLogFields(pending->provenance).c_str());
        RunLogger::logGlobal(
            "detection",
            "response_provenance_mismatch response={" +
                provenanceLogFields(response.provenance) + "} expected={" +
                provenanceLogFields(pending->provenance) + "}");
      }
      response.provenance = pending->provenance;
      response.has_camera_pose = true;
      response.T_world_camera = pending->T_world_camera;
      response.visibility_context = pending->visibility_context;
      if (config_.online_snapshot_enabled ||
          config_.snapshot_remake_enabled) {
        response.source_rgb_960 = pending->image;
      }
    }
    last_response_request_id_ = response.provenance.request_id;
    const std::string response_provenance_fields =
        provenanceLogFields(response.provenance);
    double roundtrip_ms = 0.0;
    double ingest_to_forward_ms = 0.0;
    if (pending) {
      roundtrip_ms = elapsedMs(pending->sent_time, forward_time);
      ingest_to_forward_ms = elapsedMs(pending->ingest_time, forward_time);
      total_roundtrip_ms_ += roundtrip_ms;
      total_ingest_to_forward_ms_ += ingest_to_forward_ms;
    }
    total_response_queue_dwell_ms_ += response_queue_dwell_ms;
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
                  "trace={%s} error=%s",
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
                  response_provenance_fields.c_str(),
                  response.error.c_str());
      RunLogger::logGlobal("detection",
                           "response_error " + response_provenance_fields +
                               " camera=" + response.camera_id +
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
                               " map_commit_wait_ms=" +
                               std::to_string(pending ? pending->map_commit_wait_ms : 0.0) +
                               " projection_compute_ms=" +
                               std::to_string(pending ? pending->projection_compute_ms : 0.0) +
                               " resize_ms=" +
                               std::to_string(pending ? pending->resize_ms : 0.0) +
                               " rgb_resize_ms=" +
                               std::to_string(pending ? pending->rgb_resize_ms : 0.0) +
                               " mask_resize_ms=" +
                               std::to_string(pending ? pending->mask_resize_ms : 0.0) +
                               " backend_response_queue_dwell_ms=" +
                               std::to_string(response_queue_dwell_ms) +
                               " ingest_to_response_forward_ms=" +
                               std::to_string(ingest_to_forward_ms) +
                               " roundtrip_ms=" + std::to_string(roundtrip_ms) +
                               " error=" + response.error);
    } else {
      RCLCPP_INFO(logger_,
                  "inference response camera=%s t=%ld filtered_2d=%zu raw_3d=%zu "
                  "project=%.1fms snapshot=%.1fms surface=%.1fms frustum=%.1fms "
                  "proj_loop=%.1fms zbuffer=%.1fms resize=%.1fms backend=%.1fms worker=%.1fms "
                  "owl=%.1fms boxernet=%.1fms roundtrip=%.1fms trace={%s}",
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
                  roundtrip_ms,
                  response_provenance_fields.c_str());
      std::ostringstream stream;
      stream << std::fixed << std::setprecision(2)
             << "response " << response_provenance_fields
             << " camera=" << response.camera_id
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
             << " map_commit_wait_ms=" << (pending ? pending->map_commit_wait_ms : 0.0)
             << " map_commit_latency_ms=" << (pending ? pending->map_commit_latency_ms : 0.0)
             << " projection_compute_ms=" << (pending ? pending->projection_compute_ms : 0.0)
             << " resize_ms=" << (pending ? pending->resize_ms : 0.0)
             << " rgb_resize_ms=" << (pending ? pending->rgb_resize_ms : 0.0)
             << " mask_resize_ms=" << (pending ? pending->mask_resize_ms : 0.0)
             << " backend_ipc_ms=" << response.backend_ipc_ms
             << " worker_ms=" << response.python_worker_ms
             << " preprocess_ms=" << response.python_preprocess_ms
             << " owl_ms=" << response.owl_ms
             << " robot_filter_ms=" << response.robot_filter_ms
             << " boxernet_ms=" << response.boxernet_ms
             << " postprocess_ms=" << response.python_postprocess_ms
             << " backend_response_queue_dwell_ms=" << response_queue_dwell_ms
             << " ingest_to_response_forward_ms=" << ingest_to_forward_ms
             << " deadline_lateness_ms="
             << (pending && forward_time > pending->due_time
                     ? elapsedMs(pending->due_time, forward_time)
                     : 0.0)
             << " roundtrip_ms=" << roundtrip_ms;
      RunLogger::logGlobal("detection", stream.str());
    }
    publishDetectionDebugImage(response, pending);
    publishRawDetectionMarkers(response);
    const RequestId response_request_id = response.provenance.request_id;
    PushResult<InferenceResponse> push_result =
        response_queue_.push(std::move(response));
    if (!push_result.accepted()) {
      RunLogger::logGlobal(
          "detection",
          "reliable_response_not_enqueued request_id=" +
              std::to_string(response_request_id) + " outcome=" +
              std::to_string(static_cast<int>(push_result.outcome)));
    }
    if (response_request_id != 0 &&
        response_request_id == active_request_id_) {
      RunLogger::logGlobal(
          "detection",
          "perception_terminal request_id=" +
              std::to_string(response_request_id) + " reason=" +
              (push_result.accepted()
                   ? (stopRequested() ? "shutdown_terminal_forwarded" :
                                        "response_forwarded")
                   : "response_delivery_failed") +
              " backend_response_queue_dwell_ms=" +
              std::to_string(response_queue_dwell_ms) +
              " ingest_to_response_forward_ms=" +
              std::to_string(ingest_to_forward_ms));
      active_request_id_ = 0;
      perception_busy_.store(active_frame_ != nullptr);
    }
  }
}

void DetectionBridgeThread::stashDebugFrame(const InferenceRequest& request,
                                            double projection_ms,
                                            double resize_ms,
                                            const RequestBuildTiming& timing) {
  if (request.rgb_960.empty()) {
    return;
  }
  const RequestId request_id = request.provenance.request_id;
  if (request_id == 0) {
    return;
  }
  PendingDebugFrame debug;
  debug.image = request.rgb_960;
  debug.provenance = request.provenance;
  debug.ingest_time = request.ingest_time;
  debug.due_time = request.due_time;
  debug.sent_time = std::chrono::steady_clock::now();
  debug.T_world_camera = request.T_world_camera;
  debug.visibility_context = request.visibility_context;
  debug.projection_ms = projection_ms;
  debug.resize_ms = resize_ms;
  debug.rgb_resize_ms = timing.rgb_resize_ms;
  debug.mask_resize_ms = timing.mask_resize_ms;
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
  pending_debug_frames_[request_id] = std::move(debug);
  pending_debug_order_.push_back(request_id);

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
  const RequestId request_id = response.provenance.request_id;
  if (request_id == 0) {
    return std::nullopt;
  }
  auto it = pending_debug_frames_.find(request_id);
  if (it == pending_debug_frames_.end()) {
    return std::nullopt;
  }
  PendingDebugFrame image = std::move(it->second);
  pending_debug_frames_.erase(it);
  pending_debug_order_.erase(
      std::remove(pending_debug_order_.begin(),
                  pending_debug_order_.end(),
                  request_id),
      pending_debug_order_.end());
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
  const ChannelStats inference_requests =
      inference_backend_.requestChannelStats();
  const ChannelStats inference_responses =
      inference_backend_.responseChannelStats();
  const ChannelStats detection = detection_queue_.stats();
  const ChannelStats reducer_responses = response_queue_.stats();
  std::size_t pending_map_commits = 0;
  {
    std::lock_guard<std::mutex> lock(join_mutex_);
    pending_map_commits = pending_map_commits_.size();
  }
  std::ostringstream status;
  const std::uint64_t delta_frames = frames_seen_ - last_logged_frames_seen_;
  const std::uint64_t delta_skip_rate = skipped_rate_ - last_logged_skipped_rate_;
  const std::uint64_t delta_sent = requests_sent_ - last_logged_requests_sent_;
  const std::uint64_t delta_responses = responses_seen_ - last_logged_responses_seen_;
  const double request_count = std::max<std::uint64_t>(1, requests_sent_);
  const double response_count = std::max<std::uint64_t>(1, timing_response_count_);
  status << "status frames=" << frames_seen_
         << " last_frame_id=" << last_frame_id_
         << " last_request_id=" << last_request_id_
         << " last_response_request_id=" << last_response_request_id_
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
         << " perception_busy=" << (perception_busy_.load() ? "true" : "false")
         << " active_request_id=" << active_request_id_
         << " pending_map_commits=" << pending_map_commits
         << " map_join_deadlines=" << map_join_deadlines_
         << " map_join_failures=" << map_join_failures_
         << " shutdown_candidates_cancelled="
         << shutdown_candidates_cancelled_
         << " detection_queue_depth=" << detection.depth
         << " detection_queue_oldest_ms="
         << std::chrono::duration<double, std::milli>(detection.oldest_age).count()
         << " detection_queue_last_dequeue_ms="
         << std::chrono::duration<double, std::milli>(
                detection.last_dequeue_age)
                .count()
         << " detection_queue_max_dequeue_ms="
         << std::chrono::duration<double, std::milli>(
                detection.max_dequeue_age)
                .count()
         << " inference_request_queue_depth=" << inference_requests.depth
         << " inference_request_queue_high_watermark="
         << inference_requests.high_watermark
         << " inference_response_queue_depth=" << inference_responses.depth
         << " inference_response_queue_high_watermark="
         << inference_responses.high_watermark
         << " backend_response_queue_last_dequeue_ms="
         << std::chrono::duration<double, std::milli>(
                inference_responses.last_dequeue_age)
                .count()
         << " reducer_response_queue_depth=" << reducer_responses.depth
         << " reducer_response_queue_oldest_ms="
         << std::chrono::duration<double, std::milli>(
                reducer_responses.oldest_age)
                .count()
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
         << " avg_map_commit_wait_ms=" << total_map_commit_wait_ms_ / request_count
         << " avg_projection_compute_ms=" << total_projection_compute_ms_ / request_count
         << " avg_resize_ms=" << total_resize_ms_ / request_count
         << " avg_rgb_resize_ms=" << total_rgb_resize_ms_ / request_count
         << " avg_mask_resize_ms=" << total_mask_resize_ms_ / request_count
         << " avg_backend_ipc_ms=" << total_backend_ipc_ms_ / response_count
         << " avg_worker_ms=" << total_worker_ms_ / response_count
         << " avg_owl_ms=" << total_owl_ms_ / response_count
         << " avg_boxernet_ms=" << total_boxernet_ms_ / response_count
         << " avg_backend_response_queue_dwell_ms="
         << total_response_queue_dwell_ms_ / response_count
         << " avg_ingest_to_response_forward_ms="
         << total_ingest_to_forward_ms_ / response_count
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
