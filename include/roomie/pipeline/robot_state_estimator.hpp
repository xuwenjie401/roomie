#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "roomie/pipeline/types.hpp"

namespace roomie {

enum class NavigationPostureGroup : std::uint8_t {
  kBody = 0,
  kLeftArm = 1,
  kRightArm = 2,
};

struct NavigationLinkPose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::string frame_id;
  NavigationPostureGroup group = NavigationPostureGroup::kBody;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

using NavigationLinkPoseVector =
    std::vector<NavigationLinkPose,
                Eigen::aligned_allocator<NavigationLinkPose>>;

// The whole navigation posture is loaded from one robot-specific document so
// changing the default does not require touching estimator logic or thresholds.
struct NavigationPosture {
  std::string root_frame;
  NavigationLinkPoseVector links;
};

NavigationPosture loadNavigationPosture(
    const std::filesystem::path& config_path);

struct RobotStateEstimatorConfig {
  double odom_history_sec = 5.0;
  double odom_stale_timeout_sec = 0.25;
  double rotation_window_sec = 0.10;
  double rotation_enter_rad_s = 0.05;
  double rotation_exit_rad_s = 0.02;
  double rotation_exit_hold_sec = 0.30;
  double posture_stale_timeout_sec = 0.25;
  double posture_enter_hold_sec = 0.10;
  double posture_exit_hold_sec = 0.50;
  double body_translation_enter_m = 0.02;
  double body_translation_exit_m = 0.01;
  double body_rotation_enter_rad = 0.05235987755982989;
  double body_rotation_exit_rad = 0.02617993877991494;
  double arm_translation_enter_m = 0.03;
  double arm_translation_exit_m = 0.015;
  double arm_rotation_enter_rad = 0.08726646259971647;
  double arm_rotation_exit_rad = 0.04363323129985824;
  std::size_t max_odom_samples = 2048;
  std::size_t max_posture_samples = 2048;
  NavigationPosture navigation_posture;
};

struct OdometryObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TimeNanoseconds time_ns = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

struct LinkTransformObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::string parent_frame_id;
  std::string child_frame_id;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

using LinkTransformObservationVector =
    std::vector<LinkTransformObservation,
                Eigen::aligned_allocator<LinkTransformObservation>>;

struct PostureObservation {
  TimeNanoseconds time_ns = 0;
  LinkTransformObservationVector transforms;
};

enum class OdometryObservationOutcome : std::uint8_t {
  kInserted = 0,
  kReplaced = 1,
  kInvalid = 2,
  kTooOld = 3,
};

struct RobotStateEstimatorUpdate {
  OdometryObservationOutcome outcome = OdometryObservationOutcome::kInvalid;
  bool out_of_order = false;
  bool latest_state_changed = false;
  RobotActivityValue previous_latest_state = RobotActivityValue::kUnknown;
  RobotStateSnapshot latest;
};

struct RobotPostureEstimatorUpdate {
  OdometryObservationOutcome outcome = OdometryObservationOutcome::kInvalid;
  bool out_of_order = false;
  bool latest_state_changed = false;
  RobotActivityValue previous_latest_state = RobotActivityValue::kUnknown;
  RobotStateSnapshot latest;
};

struct RobotStateEstimatorStats {
  std::uint64_t observations = 0;
  std::uint64_t invalid = 0;
  std::uint64_t out_of_order = 0;
  std::uint64_t replaced = 0;
  std::uint64_t too_old = 0;
  std::size_t retained_samples = 0;
};

class RobotStateProvider {
 public:
  virtual ~RobotStateProvider() = default;
  virtual RobotStateSnapshot snapshotAt(TimeNanoseconds time_ns) const = 0;
};

// A bounded, timestamp-indexed estimator. It has no worker thread: odometry
// and TF callbacks insert samples and recompute small retained timelines, while
// frame admission performs binary-search lookups at the image timestamp.
class RobotStateEstimator final : public RobotStateProvider {
 public:
  explicit RobotStateEstimator(RobotStateEstimatorConfig config = {});

  RobotStateEstimatorUpdate observeOdometry(
      const OdometryObservation& observation);
  RobotPostureEstimatorUpdate observePosture(
      const PostureObservation& observation);
  RobotStateSnapshot snapshotAt(TimeNanoseconds time_ns) const override;
  RobotStateEstimatorStats stats() const;

 private:
  struct MachineState {
    RobotActivityStatus rotating;
    TimeNanoseconds below_exit_since_ns = 0;
  };

  struct Sample {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TimeNanoseconds time_ns = 0;
    double yaw_rad = 0.0;
    bool valid = false;
    std::optional<double> yaw_rate_rad_s;
    MachineState machine;
  };

  using SampleVector =
      std::vector<Sample, Eigen::aligned_allocator<Sample>>;

  struct PostureFacetMachine {
    RobotActivityStatus status;
    TimeNanoseconds above_enter_since_ns = 0;
    TimeNanoseconds inside_exit_since_ns = 0;
  };

  struct PostureMachineState {
    PostureFacetMachine body;
    PostureFacetMachine left_arm;
    PostureFacetMachine right_arm;
    RobotActivityStatus deviated;
  };

  struct PostureSample {
    TimeNanoseconds time_ns = 0;
    bool valid = false;
    bool body_beyond_enter = false;
    bool body_inside_exit = false;
    bool left_arm_beyond_enter = false;
    bool left_arm_inside_exit = false;
    bool right_arm_beyond_enter = false;
    bool right_arm_inside_exit = false;
    PostureMachineState machine;
  };

  using PostureSampleVector = std::vector<PostureSample>;

  static RobotStateSnapshot snapshotFromSample(const Sample& sample);
  static void overlayPostureSnapshot(const PostureSample& sample,
                                     RobotStateSnapshot* snapshot);
  static RobotStateSnapshot unknownSnapshot(TimeNanoseconds observed_at_ns);
  static bool validObservation(const OdometryObservation& observation,
                               double* normalized_yaw_rad);
  void pruneLocked();
  void recomputeLocked();
  void prunePostureLocked();
  void recomputePostureLocked();
  PostureSample makePostureSample(
      const PostureObservation& observation) const;
  RobotActivityValue latestValueLocked() const;
  RobotActivityValue latestPostureValueLocked() const;
  RobotStateSnapshot latestSnapshotLocked() const;

  RobotStateEstimatorConfig config_;
  TimeNanoseconds history_ns_ = 0;
  TimeNanoseconds stale_timeout_ns_ = 0;
  TimeNanoseconds rotation_window_ns_ = 0;
  TimeNanoseconds rotation_exit_hold_ns_ = 0;
  TimeNanoseconds posture_stale_timeout_ns_ = 0;
  TimeNanoseconds posture_enter_hold_ns_ = 0;
  TimeNanoseconds posture_exit_hold_ns_ = 0;

  mutable std::mutex mutex_;
  SampleVector samples_;
  TimeNanoseconds latest_seen_ns_ = 0;
  TimeNanoseconds seed_time_ns_ = 0;
  MachineState seed_machine_;
  PostureSampleVector posture_samples_;
  TimeNanoseconds latest_posture_seen_ns_ = 0;
  TimeNanoseconds posture_seed_time_ns_ = 0;
  PostureMachineState posture_seed_machine_;
  RobotStateEstimatorStats stats_;
};

}  // namespace roomie
