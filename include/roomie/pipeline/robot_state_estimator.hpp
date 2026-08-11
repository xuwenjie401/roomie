#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include <Eigen/Geometry>

#include "roomie/pipeline/types.hpp"

namespace roomie {

struct RobotStateEstimatorConfig {
  double odom_history_sec = 5.0;
  double odom_stale_timeout_sec = 0.25;
  double rotation_window_sec = 0.10;
  double rotation_enter_rad_s = 0.05;
  double rotation_exit_rad_s = 0.02;
  double rotation_exit_hold_sec = 0.30;
  std::size_t max_odom_samples = 2048;
};

struct OdometryObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TimeNanoseconds time_ns = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
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
// callbacks insert a sample and recompute the small retained timeline, while
// frame admission performs a binary-search lookup at the image timestamp.
class RobotStateEstimator final : public RobotStateProvider {
 public:
  explicit RobotStateEstimator(RobotStateEstimatorConfig config = {});

  RobotStateEstimatorUpdate observeOdometry(
      const OdometryObservation& observation);
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

  static RobotStateSnapshot snapshotFromSample(const Sample& sample);
  static RobotStateSnapshot unknownSnapshot(TimeNanoseconds observed_at_ns);
  static bool validObservation(const OdometryObservation& observation,
                               double* normalized_yaw_rad);
  void pruneLocked();
  void recomputeLocked();
  RobotActivityValue latestValueLocked() const;
  RobotStateSnapshot latestSnapshotLocked() const;

  RobotStateEstimatorConfig config_;
  TimeNanoseconds history_ns_ = 0;
  TimeNanoseconds stale_timeout_ns_ = 0;
  TimeNanoseconds rotation_window_ns_ = 0;
  TimeNanoseconds rotation_exit_hold_ns_ = 0;

  mutable std::mutex mutex_;
  SampleVector samples_;
  TimeNanoseconds latest_seen_ns_ = 0;
  TimeNanoseconds seed_time_ns_ = 0;
  MachineState seed_machine_;
  RobotStateEstimatorStats stats_;
};

}  // namespace roomie
