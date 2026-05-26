#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace roomie {

using TimeNanoseconds = std::int64_t;

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

struct DetectionFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TimeNanoseconds time_ns = 0;
  std::string camera_id;
  ImageBuffer rgb;
  ImageBuffer robot_mask;
  CameraIntrinsics intrinsics;
  Eigen::Isometry3f T_world_camera = Eigen::Isometry3f::Identity();
};

struct MappingFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TimeNanoseconds time_ns = 0;
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

  PatchDepth() { values.fill(-1.0f); }

  float coverageRatio() const {
    return static_cast<float>(valid_patches) / static_cast<float>(kSize);
  }

  bool hasMinimumCoverage(float threshold) const { return coverageRatio() >= threshold; }
};

struct InferenceRequest {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TimeNanoseconds time_ns = 0;
  std::string camera_id;
  ImageBuffer rgb_960;
  ImageBuffer mask_960;
  PatchDepth patch_depth;
  CameraIntrinsics intrinsics_960;
  Eigen::Isometry3f T_world_camera = Eigen::Isometry3f::Identity();
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
};

struct InferenceResponse {
  TimeNanoseconds time_ns = 0;
  std::string camera_id;
  bool ok = false;
  std::string error;
  std::vector<RawDetection, Eigen::aligned_allocator<RawDetection>> detections;
};

struct VoxelRef {
  Eigen::Vector3i block_index = Eigen::Vector3i::Zero();
  Eigen::Vector3i voxel_index = Eigen::Vector3i::Zero();
};

struct InstanceRecord {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int track_id = -1;
  int semantic_id = -1;
  std::string label;
  Eigen::Vector3f center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f size_m = Eigen::Vector3f::Zero();
  float yaw_rad = 0.0f;
  float confidence = 0.0f;
  int support_count = 0;
  TimeNanoseconds first_seen_ns = 0;
  TimeNanoseconds last_seen_ns = 0;
  std::vector<std::string> source_cameras;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> near_surface_voxels;
};

}  // namespace roomie
