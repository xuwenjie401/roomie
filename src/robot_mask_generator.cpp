#include "roomie/pipeline/robot_mask_generator.hpp"

#include <ego_shade/ego_shade.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace roomie {
namespace {

constexpr double kQuaternionNormEpsilon = 1.0e-12;

bool posesEquivalent(const RobotPoseSnapshot& rendered,
                     const RobotPoseSnapshot& current,
                     const std::vector<std::string>& required_frames,
                     double translation_epsilon_m,
                     double rotation_epsilon_rad) {
  for (const std::string& frame : required_frames) {
    const auto old_it = rendered.root_T_frame.find(frame);
    const auto new_it = current.root_T_frame.find(frame);
    if (old_it == rendered.root_T_frame.end() ||
        new_it == current.root_T_frame.end()) {
      return false;
    }

    const Eigen::Isometry3d& old_pose = old_it->second;
    const Eigen::Isometry3d& new_pose = new_it->second;
    if (!old_pose.matrix().allFinite() || !new_pose.matrix().allFinite()) {
      return false;
    }
    if ((old_pose.translation() - new_pose.translation()).norm() >
        translation_epsilon_m) {
      return false;
    }

    Eigen::Quaterniond old_rotation(old_pose.rotation());
    Eigen::Quaterniond new_rotation(new_pose.rotation());
    const double old_norm = old_rotation.norm();
    const double new_norm = new_rotation.norm();
    if (!std::isfinite(old_norm) || !std::isfinite(new_norm) ||
        old_norm < kQuaternionNormEpsilon || new_norm < kQuaternionNormEpsilon) {
      return false;
    }
    old_rotation.normalize();
    new_rotation.normalize();
    const double cosine = std::clamp(
        std::abs(old_rotation.dot(new_rotation)), 0.0, 1.0);
    const double angle = 2.0 * std::acos(cosine);
    if (angle > rotation_epsilon_rad) {
      return false;
    }
  }
  return true;
}

RobotMaskCameraInfo loadCameraInfo(const YAML::Node& cameras,
                                   const std::string& name) {
  const YAML::Node camera = cameras[name];
  if (!camera) {
    throw std::invalid_argument("robot mask camera config is missing camera '" +
                                name + "'");
  }
  const YAML::Node resolution = camera["resolution"];
  const YAML::Node intrinsics = camera["intrinsics"];
  if (!resolution || !resolution.IsSequence() || resolution.size() != 2 ||
      !intrinsics) {
    throw std::invalid_argument("robot mask camera '" + name +
                                "' has incomplete geometry");
  }
  RobotMaskCameraInfo info;
  info.name = name;
  info.topic = camera["topic"].as<std::string>("");
  info.camera_info_topic =
      camera["camera_info_topic"].as<std::string>("");
  info.frame = camera["frame"].as<std::string>("");
  info.width = resolution[0].as<int>();
  info.height = resolution[1].as<int>();
  info.fx = intrinsics["fx"].as<double>();
  info.fy = intrinsics["fy"].as<double>();
  info.cx = intrinsics["cx"].as<double>();
  info.cy = intrinsics["cy"].as<double>();
  return info;
}

}  // namespace

class RobotMaskGenerator::Impl {
 public:
  struct CameraCache {
    std::mutex mutex;
    std::optional<RobotPoseSnapshot> rendered_pose;
    std::shared_ptr<const ImageBuffer> mask;
    std::size_t mask_pixels = 0;
    bool full_mask = false;
  };

  explicit Impl(RobotMaskGeneratorConfig config)
      : translation_epsilon_m_(config.reuse_translation_epsilon_m),
        rotation_epsilon_rad_(config.reuse_rotation_epsilon_rad) {
    if (config.robot_config.empty() || config.camera_config.empty()) {
      throw std::invalid_argument("robot mask config paths must not be empty");
    }
    if (!std::isfinite(translation_epsilon_m_) ||
        !std::isfinite(rotation_epsilon_rad_) ||
        translation_epsilon_m_ < 0.0 || rotation_epsilon_rad_ < 0.0) {
      throw std::invalid_argument(
          "robot mask pose reuse thresholds must be finite and non-negative");
    }

    shade_ = std::make_unique<ego_shade::EgoShade>(
        config.robot_config, config.camera_config);
    root_frame_ = shade_->rootFrame();
    required_frames_ = shade_->requiredFrames();

    YAML::Node camera_document;
    try {
      camera_document = YAML::LoadFile(config.camera_config.string());
    } catch (const YAML::Exception& error) {
      throw std::invalid_argument("unable to inspect robot mask camera config '" +
                                  config.camera_config.string() + "': " +
                                  error.what());
    }
    const YAML::Node configured_cameras = camera_document["cameras"];
    for (const ego_shade::CameraInfo& camera : shade_->cameras()) {
      RobotMaskCameraInfo info = loadCameraInfo(configured_cameras, camera.name);
      if (info.width != camera.width || info.height != camera.height) {
        throw std::invalid_argument("robot mask camera metadata disagrees for '" +
                                    camera.name + "'");
      }
      camera_info_.emplace(camera.name, std::move(info));
      camera_cache_.emplace(camera.name, std::make_unique<CameraCache>());
    }
  }

  std::unique_ptr<ego_shade::EgoShade> shade_;
  std::string root_frame_;
  std::vector<std::string> required_frames_;
  std::unordered_map<std::string, RobotMaskCameraInfo> camera_info_;
  std::unordered_map<std::string, std::unique_ptr<CameraCache>> camera_cache_;
  double translation_epsilon_m_ = 0.0;
  double rotation_epsilon_rad_ = 0.0;
};

RobotMaskGenerator::RobotMaskGenerator(RobotMaskGeneratorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

RobotMaskGenerator::~RobotMaskGenerator() = default;

const std::string& RobotMaskGenerator::rootFrame() const {
  return impl_->root_frame_;
}

const std::vector<std::string>& RobotMaskGenerator::requiredFrames() const {
  return impl_->required_frames_;
}

bool RobotMaskGenerator::hasCamera(const std::string& camera_id) const {
  return impl_->camera_info_.find(camera_id) != impl_->camera_info_.end();
}

RobotMaskCameraInfo RobotMaskGenerator::cameraInfo(
    const std::string& camera_id) const {
  const auto found = impl_->camera_info_.find(camera_id);
  if (found == impl_->camera_info_.end()) {
    throw std::invalid_argument("unknown robot mask camera: " + camera_id);
  }
  return found->second;
}

RobotMaskResult RobotMaskGenerator::generate(
    const std::string& camera_id, const RobotPoseSnapshot& pose) {
  const auto cache_it = impl_->camera_cache_.find(camera_id);
  if (cache_it == impl_->camera_cache_.end()) {
    throw std::invalid_argument("unknown robot mask camera: " + camera_id);
  }
  Impl::CameraCache& cache = *cache_it->second;
  std::lock_guard<std::mutex> lock(cache.mutex);

  if (cache.mask && cache.rendered_pose &&
      posesEquivalent(*cache.rendered_pose,
                      pose,
                      impl_->required_frames_,
                      impl_->translation_epsilon_m_,
                      impl_->rotation_epsilon_rad_)) {
    return RobotMaskResult{
        cache.mask, true, 0.0, cache.mask_pixels, cache.full_mask};
  }

  ego_shade::RobotState state;
  state.timestamp_ns = pose.time_ns;
  state.root_T_frame = pose.root_T_frame;
  ego_shade::RenderStats stats;
  const auto started = std::chrono::steady_clock::now();
  ego_shade::Mask rendered =
      impl_->shade_->maskInCamera(camera_id, state, &stats);
  const double render_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - started)
                               .count();

  const RobotMaskCameraInfo info = cameraInfo(camera_id);
  const std::size_t expected_size =
      static_cast<std::size_t>(info.width) * info.height;
  if (rendered.width != info.width || rendered.height != info.height ||
      rendered.stride != rendered.width || rendered.data.size() != expected_size) {
    throw std::runtime_error("ego_shade returned an invalid mask for camera '" +
                             camera_id + "'");
  }

  auto image = std::make_shared<ImageBuffer>();
  image->width = rendered.width;
  image->height = rendered.height;
  image->channels = 1;
  image->encoding = "mono8";
  image->data = std::move(rendered.data);

  cache.mask = std::move(image);
  // Deliberately retain the pose which actually produced the mask. Advancing
  // this baseline on a reuse would allow sub-threshold motion to accumulate
  // indefinitely without a redraw.
  cache.rendered_pose = pose;
  cache.mask_pixels = stats.mask_pixels;
  cache.full_mask = stats.full_mask;
  return RobotMaskResult{
      cache.mask, false, render_ms, cache.mask_pixels, cache.full_mask};
}

}  // namespace roomie
