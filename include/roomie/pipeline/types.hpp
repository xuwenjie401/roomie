#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace roomie {

using TimeNanoseconds = std::int64_t;

enum class RobotActivityValue : std::uint8_t {
  kUnknown = 0,
  kInactive = 1,
  kActive = 2,
};

struct RobotActivityStatus {
  RobotActivityValue value = RobotActivityValue::kUnknown;
  TimeNanoseconds observed_at_ns = 0;
  TimeNanoseconds since_ns = 0;

  bool known() const { return value != RobotActivityValue::kUnknown; }
  bool active() const { return value == RobotActivityValue::kActive; }
};

// A timestamped, process-local view of robot activity. Arm activity means the
// corresponding arm is outside the configured navigation posture; it remains
// active while the arm is held still away from that posture.
struct RobotStateSnapshot {
  RobotActivityStatus rotating;
  RobotActivityStatus near_stationary;
  RobotActivityStatus body_bent;
  RobotActivityStatus left_arm_active;
  RobotActivityStatus right_arm_active;
  RobotActivityStatus navigation_posture_deviated;
  TimeNanoseconds source_time_ns = 0;
  std::optional<double> yaw_rate_rad_s;
};

struct RunId {
  std::uint64_t high = 0;
  std::uint64_t low = 0;

  bool valid() const { return high != 0 || low != 0; }
};

inline bool operator==(const RunId& lhs, const RunId& rhs) {
  return lhs.high == rhs.high && lhs.low == rhs.low;
}

inline bool operator!=(const RunId& lhs, const RunId& rhs) { return !(lhs == rhs); }

inline RunId makeRunId() {
  static std::atomic<std::uint64_t> sequence{1};
  static const std::uint64_t entropy = []() {
    std::random_device random;
    const std::uint64_t upper = static_cast<std::uint64_t>(random()) << 32U;
    return upper ^ static_cast<std::uint64_t>(random());
  }();
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  RunId id;
  id.high = entropy ^ static_cast<std::uint64_t>(now);
  id.low = sequence.fetch_add(1, std::memory_order_relaxed);
  if (!id.valid()) {
    id.low = 1;
  }
  return id;
}

inline std::string runIdString(const RunId& id) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << id.high
         << std::setw(16) << id.low;
  return stream.str();
}

using FrameId = std::uint64_t;
using RequestId = std::uint64_t;
using SceneRevision = std::uint64_t;

struct FrameKey {
  RunId run_id;
  FrameId frame_id = 0;
};

inline bool operator==(const FrameKey& lhs, const FrameKey& rhs) {
  return lhs.run_id == rhs.run_id && lhs.frame_id == rhs.frame_id;
}

inline bool operator!=(const FrameKey& lhs, const FrameKey& rhs) {
  return !(lhs == rhs);
}

struct FrameKeyHash {
  std::size_t operator()(const FrameKey& key) const noexcept {
    std::size_t seed = std::hash<std::uint64_t>{}(key.run_id.high);
    seed ^= std::hash<std::uint64_t>{}(key.run_id.low) + 0x9e3779b9U +
            (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<FrameId>{}(key.frame_id) + 0x9e3779b9U +
            (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

enum class MapMode : std::uint8_t {
  kOnline = 0,
  kFrozen = 1,
};

struct MapStamp {
  RunId map_epoch;
  std::uint64_t map_revision = 0;
  TimeNanoseconds integrated_through_ns = 0;
};

inline bool operator==(const MapStamp& lhs, const MapStamp& rhs) {
  return lhs.map_epoch == rhs.map_epoch &&
         lhs.map_revision == rhs.map_revision &&
         lhs.integrated_through_ns == rhs.integrated_through_ns;
}

inline bool operator!=(const MapStamp& lhs, const MapStamp& rhs) {
  return !(lhs == rhs);
}

struct SurfaceStamp {
  RunId map_epoch;
  std::uint64_t surface_revision = 0;
  std::uint64_t source_map_revision = 0;
};

inline bool operator==(const SurfaceStamp& lhs, const SurfaceStamp& rhs) {
  return lhs.map_epoch == rhs.map_epoch &&
         lhs.surface_revision == rhs.surface_revision &&
         lhs.source_map_revision == rhs.source_map_revision;
}


inline bool operator!=(const SurfaceStamp& lhs, const SurfaceStamp& rhs) {
  return !(lhs == rhs);
}

struct FrameProvenance {
  RunId run_id;
  FrameId frame_id = 0;
  RequestId request_id = 0;
  TimeNanoseconds sensor_time_ns = 0;
  MapMode map_mode = MapMode::kOnline;
  bool includes_current_frame = false;
  bool causality_verified = false;
  MapStamp map;
  SurfaceStamp surface;
};

inline bool operator==(const FrameProvenance& lhs, const FrameProvenance& rhs) {
  return lhs.run_id == rhs.run_id && lhs.frame_id == rhs.frame_id &&
         lhs.request_id == rhs.request_id &&
         lhs.sensor_time_ns == rhs.sensor_time_ns &&
         lhs.map_mode == rhs.map_mode &&
         lhs.includes_current_frame == rhs.includes_current_frame &&
         lhs.causality_verified == rhs.causality_verified && lhs.map == rhs.map &&
         lhs.surface == rhs.surface;
}

inline bool operator!=(const FrameProvenance& lhs, const FrameProvenance& rhs) {
  return !(lhs == rhs);
}

struct PipelineTiming {
  double serialize_ms = 0.0;
  double pipe_write_ms = 0.0;
  double pipe_read_ms = 0.0;
  double worker_queue_ms = 0.0;
  double response_forward_ms = 0.0;
};

enum class InstanceGeometryStatus {
  kUnchecked,
  kGood,
  kBad,
  kEmpty,
};

struct CameraIntrinsics {
  int width = 0;
  int height = 0;
  float fx = 0.0f;
  float fy = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;

  CameraIntrinsics scaledTo(int target_width, int target_height) const {
    CameraIntrinsics scaled = *this;
    if (width > 0 && height > 0) {
      const float sx = static_cast<float>(target_width) / static_cast<float>(width);
      const float sy = static_cast<float>(target_height) / static_cast<float>(height);
      scaled.fx *= sx;
      scaled.fy *= sy;
      scaled.cx *= sx;
      scaled.cy *= sy;
    }
    scaled.width = target_width;
    scaled.height = target_height;
    return scaled;
  }
};

struct ImageBuffer {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::string encoding;
  std::vector<std::uint8_t> data;

  bool empty() const { return width <= 0 || height <= 0 || channels <= 0 || data.empty(); }
};

struct DepthBuffer {
  int width = 0;
  int height = 0;
  std::vector<float> depth_m;

  bool empty() const { return width <= 0 || height <= 0 || depth_m.empty(); }
};

// Process-local evidence carried from the admitted RGB-D frame to the
// association actor.  It is deliberately excluded from the Python wire
// protocol: the worker never owns lifecycle decisions or the raw depth frame.
struct VisibilityContext {
  std::shared_ptr<const DepthBuffer> depth;
  std::shared_ptr<const ImageBuffer> robot_mask;
  CameraIntrinsics intrinsics;

  bool valid() const {
    return depth && !depth->empty() && intrinsics.width > 0 &&
           intrinsics.height > 0 && intrinsics.fx > 0.0f &&
           intrinsics.fy > 0.0f &&
           depth->width == intrinsics.width && depth->height == intrinsics.height;
  }
};

struct SyncDiagnostics {
  TimeNanoseconds rgb_depth_delta_ns = 0;
  TimeNanoseconds tf_delta_ns = 0;
  bool used_latest_tf_fallback = false;
};

struct FrameBundle {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FrameProvenance provenance;
  std::chrono::steady_clock::time_point ingest_time =
      std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point due_time =
      std::chrono::steady_clock::time_point::max();
  std::string camera_id;
  std::shared_ptr<const ImageBuffer> rgb;
  std::shared_ptr<const DepthBuffer> depth;
  std::shared_ptr<const ImageBuffer> robot_mask;
  CameraIntrinsics intrinsics;
  Eigen::Isometry3f T_world_camera = Eigen::Isometry3f::Identity();
  std::uint64_t calibration_revision = 0;
  SyncDiagnostics sync;
  RobotStateSnapshot robot_state;
  // True only when detection requires this exact frame's include-current map
  // commit. RGB-independent detection leaves this false and uses the latest
  // already-published surface.
  bool perception_candidate = false;
};

using FrameBundlePtr = std::shared_ptr<const FrameBundle>;

struct DetectionFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TimeNanoseconds time_ns = 0;
  FrameProvenance provenance;
  std::chrono::steady_clock::time_point ingest_time =
      std::chrono::steady_clock::now();
  std::string camera_id;
  ImageBuffer rgb;
  ImageBuffer robot_mask;
  CameraIntrinsics intrinsics;
  Eigen::Isometry3f T_world_camera = Eigen::Isometry3f::Identity();
};

struct MappingFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TimeNanoseconds time_ns = 0;
  FrameProvenance provenance;
  std::chrono::steady_clock::time_point ingest_time =
      std::chrono::steady_clock::now();
  std::string camera_id;
  ImageBuffer rgb;
  DepthBuffer depth;
  ImageBuffer robot_mask;
  CameraIntrinsics intrinsics;
  Eigen::Isometry3f T_world_camera = Eigen::Isometry3f::Identity();
};

struct PatchDepth {
  static constexpr int kRows = 60;
  static constexpr int kCols = 60;
  static constexpr int kSize = kRows * kCols;

  std::array<float, kSize> values{};
  int valid_patches = 0;
  int projected_points = 0;
  std::uint64_t map_version = 0;
  FrameProvenance provenance;
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
  // Exact causal-barrier wait and actual projection compute are reported
  // separately. project_total_ms remains for wire/log compatibility.
  double map_commit_wait_ms = 0.0;
  double projection_compute_ms = 0.0;
  bool surface_cache_ready = false;
  bool view_filtered = false;

  PatchDepth() { values.fill(-1.0f); }

  float coverageRatio() const {
    return static_cast<float>(valid_patches) / static_cast<float>(kSize);
  }

  bool hasMinimumCoverage(float threshold) const { return coverageRatio() >= threshold; }
};

struct InferenceRequest {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TimeNanoseconds time_ns = 0;
  FrameProvenance provenance;
  std::chrono::steady_clock::time_point ingest_time =
      std::chrono::steady_clock::now();
  // Process-local admission timestamp. It is intentionally not serialized:
  // worker_queue_ms measures only the C++ backend channel, while ingest_time
  // remains the end-to-end frame clock.
  std::chrono::steady_clock::time_point backend_enqueued_at =
      std::chrono::steady_clock::time_point::min();
  std::chrono::steady_clock::time_point due_time =
      std::chrono::steady_clock::time_point::max();
  PipelineTiming timing;
  std::string camera_id;
  ImageBuffer rgb_960;
  ImageBuffer mask_960;
  PatchDepth patch_depth;
  CameraIntrinsics intrinsics_960;
  Eigen::Isometry3f T_world_camera = Eigen::Isometry3f::Identity();
  std::shared_ptr<const VisibilityContext> visibility_context;
};

struct Raw2dDetection {
  float score_2d = 0.0f;
  std::array<float, 4> box_xyxy = {0.0f, 0.0f, 0.0f, 0.0f};
  int semantic_id = -1;
  std::string label;
};

struct RawDetection {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3f center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f size_m = Eigen::Vector3f::Zero();
  float yaw_rad = 0.0f;
  float score_2d = 0.0f;
  float score_3d = 0.0f;
  std::array<float, 4> box_xyxy = {0.0f, 0.0f, 0.0f, 0.0f};
  int semantic_id = -1;
  std::string label;
  std::string appearance_model_id;
  std::vector<float> appearance_descriptor;
};

struct InferenceResponse {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TimeNanoseconds time_ns = 0;
  FrameProvenance provenance;
  PipelineTiming timing;
  std::string camera_id;
  bool has_camera_pose = false;
  Eigen::Isometry3f T_world_camera = Eigen::Isometry3f::Identity();
  bool ok = false;
  std::string error;
  float backend_ipc_ms = 0.0f;
  float python_worker_ms = 0.0f;
  float python_preprocess_ms = 0.0f;
  float owl_ms = 0.0f;
  float robot_filter_ms = 0.0f;
  float boxernet_ms = 0.0f;
  float python_postprocess_ms = 0.0f;
  std::vector<Raw2dDetection> filtered_2d_detections;
  std::vector<RawDetection, Eigen::aligned_allocator<RawDetection>> detections;
  ImageBuffer source_rgb_960;
  std::shared_ptr<const VisibilityContext> visibility_context;
};

struct VoxelRef {
  Eigen::Vector3i block_index = Eigen::Vector3i::Zero();
  Eigen::Vector3i voxel_index = Eigen::Vector3i::Zero();
};

struct InstanceRecord {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int object_id = -1;
  int track_id = -1;
  int semantic_id = -1;
  std::string label;
  std::string description;
  Eigen::Vector3f center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f size_m = Eigen::Vector3f::Zero();
  float yaw_rad = 0.0f;
  float confidence = 0.0f;
  float confidence_mass = 0.0f;
  float object_quality_score = 0.0f;
  float geometry_score = 0.0f;
  float geometry_shell_ratio = 0.0f;
  float geometry_extent_score = 0.0f;
  float geometry_leak_ratio = 1.0f;
  float geometry_cavity_ratio = 0.0f;
  int geometry_in_box_points = 0;
  int geometry_shell_points = 0;
  int geometry_unique_voxels = 0;
  int geometry_expanded_points = 0;
  int geometry_bad_count = 0;
  int support_count = 0;
  int high_quality_observation_count = 0;
  float high_quality_observation_mass = 0.0f;
  bool active = true;
  bool publishable = true;
  float existence_log_odds = 0.0f;
  float existence_probability = 0.5f;
  std::string presence_state = "tentative";
  TimeNanoseconds last_presence_evidence_ns = 0;
  std::string last_presence_evidence_reason;
  InstanceGeometryStatus geometry_status = InstanceGeometryStatus::kUnchecked;
  TimeNanoseconds last_geometry_check_ns = 0;
  TimeNanoseconds first_seen_ns = 0;
  TimeNanoseconds last_seen_ns = 0;
  std::vector<std::string> source_cameras;
  std::vector<int> source_track_ids;
  std::vector<TimeNanoseconds> observation_timestamps_ns;
  int snapshot_image_index = -1;
  std::array<float, 4> snapshot_bbox_xyxy = {0.0f, 0.0f, 0.0f, 0.0f};
  float snapshot_quality = 0.0f;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> near_surface_voxels;
};

}  // namespace roomie
