#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

#include "roomie/pipeline/types.hpp"

namespace roomie {

struct RobotPoseSnapshot {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TimeNanoseconds time_ns = 0;
  std::unordered_map<std::string, Eigen::Isometry3d> root_T_frame;
};

struct RobotMaskGeneratorConfig {
  std::filesystem::path robot_config;
  std::filesystem::path camera_config;
  double reuse_translation_epsilon_m = 5.0e-6;
  double reuse_rotation_epsilon_rad = 5.0e-6;
};

struct RobotMaskCameraInfo {
  std::string name;
  std::string topic;
  std::string camera_info_topic;
  std::string frame;
  int width = 0;
  int height = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

struct RobotMaskResult {
  std::shared_ptr<const ImageBuffer> mask;
  bool reused = false;
  double render_ms = 0.0;
  std::size_t mask_pixels = 0;
  bool full_mask = false;
};

// Thread-safe adapter around ego_shade::EgoShade. Each configured camera owns
// an independent last-rendered pose and mask cache, so asynchronous camera
// streams can never reuse another camera's stale mask.
class RobotMaskGenerator final {
 public:
  explicit RobotMaskGenerator(RobotMaskGeneratorConfig config);
  ~RobotMaskGenerator();

  RobotMaskGenerator(const RobotMaskGenerator&) = delete;
  RobotMaskGenerator& operator=(const RobotMaskGenerator&) = delete;
  RobotMaskGenerator(RobotMaskGenerator&&) = delete;
  RobotMaskGenerator& operator=(RobotMaskGenerator&&) = delete;

  [[nodiscard]] const std::string& rootFrame() const;
  [[nodiscard]] const std::vector<std::string>& requiredFrames() const;
  [[nodiscard]] bool hasCamera(const std::string& camera_id) const;
  [[nodiscard]] RobotMaskCameraInfo cameraInfo(
      const std::string& camera_id) const;

  RobotMaskResult generate(const std::string& camera_id,
                           const RobotPoseSnapshot& pose);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace roomie
