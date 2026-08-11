#include "roomie/pipeline/robot_state_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace roomie {
namespace {

constexpr double kNanosecondsPerSecond = 1.0e9;
constexpr double kQuaternionNormEpsilon = 1.0e-12;
constexpr double kPi = 3.14159265358979323846;

TimeNanoseconds secondsToNanosecondsChecked(double seconds) {
  if (!std::isfinite(seconds) || seconds < 0.0 ||
      seconds > static_cast<double>(std::numeric_limits<TimeNanoseconds>::max()) /
                    kNanosecondsPerSecond) {
    throw std::invalid_argument("robot state duration must be finite and non-negative");
  }
  return static_cast<TimeNanoseconds>(std::llround(seconds * kNanosecondsPerSecond));
}

double shortestAngularDistance(double from, double to) {
  return std::remainder(to - from, 2.0 * kPi);
}

bool finiteVector(const Eigen::Vector3d& value) {
  return value.array().isFinite().all();
}

}  // namespace

RobotStateEstimator::RobotStateEstimator(RobotStateEstimatorConfig config)
    : config_(std::move(config)) {
  if (!std::isfinite(config_.odom_history_sec) ||
      !std::isfinite(config_.odom_stale_timeout_sec) ||
      !std::isfinite(config_.rotation_window_sec) ||
      !std::isfinite(config_.rotation_enter_rad_s) ||
      !std::isfinite(config_.rotation_exit_rad_s) ||
      !std::isfinite(config_.rotation_exit_hold_sec) ||
      config_.odom_history_sec <= 0.0 ||
      config_.odom_stale_timeout_sec <= 0.0 ||
      config_.rotation_window_sec <= 0.0 ||
      config_.rotation_enter_rad_s <= 0.0 ||
      config_.rotation_exit_rad_s < 0.0 ||
      config_.rotation_exit_hold_sec < 0.0 ||
      config_.rotation_enter_rad_s < config_.rotation_exit_rad_s ||
      config_.max_odom_samples < 2) {
    throw std::invalid_argument("invalid robot state estimator configuration");
  }
  const double required_history = config_.rotation_window_sec +
                                  config_.rotation_exit_hold_sec +
                                  config_.odom_stale_timeout_sec;
  if (config_.odom_history_sec < required_history) {
    throw std::invalid_argument(
        "robot state odom history is too short for the configured windows");
  }

  history_ns_ = secondsToNanosecondsChecked(config_.odom_history_sec);
  stale_timeout_ns_ =
      secondsToNanosecondsChecked(config_.odom_stale_timeout_sec);
  rotation_window_ns_ =
      secondsToNanosecondsChecked(config_.rotation_window_sec);
  rotation_exit_hold_ns_ =
      secondsToNanosecondsChecked(config_.rotation_exit_hold_sec);
}

RobotStateEstimatorUpdate RobotStateEstimator::observeOdometry(
    const OdometryObservation& observation) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++stats_.observations;
  RobotStateEstimatorUpdate update;
  update.previous_latest_state = latestValueLocked();

  double yaw_rad = 0.0;
  const bool valid = validObservation(observation, &yaw_rad);
  if (observation.time_ns <= 0) {
    ++stats_.invalid;
    update.outcome = OdometryObservationOutcome::kInvalid;
    update.latest = latestSnapshotLocked();
    return update;
  }

  const TimeNanoseconds retained_horizon_ns =
      history_ns_ + rotation_window_ns_ + rotation_exit_hold_ns_;
  if (latest_seen_ns_ > 0 && observation.time_ns < latest_seen_ns_ &&
      latest_seen_ns_ - observation.time_ns > retained_horizon_ns) {
    ++stats_.too_old;
    update.out_of_order = true;
    update.outcome = OdometryObservationOutcome::kTooOld;
    update.latest = latestSnapshotLocked();
    return update;
  }

  update.out_of_order =
      latest_seen_ns_ > 0 && observation.time_ns < latest_seen_ns_;
  if (update.out_of_order) {
    ++stats_.out_of_order;
  }
  latest_seen_ns_ = std::max(latest_seen_ns_, observation.time_ns);

  const auto found = std::lower_bound(
      samples_.begin(), samples_.end(), observation.time_ns,
      [](const Sample& sample, TimeNanoseconds time_ns) {
        return sample.time_ns < time_ns;
      });
  Sample sample;
  sample.time_ns = observation.time_ns;
  sample.yaw_rad = yaw_rad;
  sample.valid = valid;
  if (found != samples_.end() && found->time_ns == observation.time_ns) {
    *found = sample;
    ++stats_.replaced;
    update.outcome = valid ? OdometryObservationOutcome::kReplaced
                           : OdometryObservationOutcome::kInvalid;
  } else {
    samples_.insert(found, sample);
    update.outcome = valid ? OdometryObservationOutcome::kInserted
                           : OdometryObservationOutcome::kInvalid;
  }
  if (!valid) {
    ++stats_.invalid;
  }

  // Recompute before pruning so the state retained at the new history boundary
  // includes the newly inserted (possibly out-of-order) observation.
  recomputeLocked();
  pruneLocked();
  recomputeLocked();
  stats_.retained_samples = samples_.size();
  update.latest = latestSnapshotLocked();
  update.latest_state_changed =
      update.previous_latest_state != update.latest.rotating.value;
  return update;
}

RobotStateSnapshot RobotStateEstimator::snapshotAt(
    TimeNanoseconds time_ns) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (time_ns <= 0 || samples_.empty() ||
      (latest_seen_ns_ > history_ns_ &&
       time_ns < latest_seen_ns_ - history_ns_)) {
    return unknownSnapshot(0);
  }

  const auto upper = std::upper_bound(
      samples_.begin(), samples_.end(), time_ns,
      [](TimeNanoseconds query_ns, const Sample& sample) {
        return query_ns < sample.time_ns;
      });
  if (upper == samples_.begin()) {
    return unknownSnapshot(0);
  }
  const Sample& sample = *std::prev(upper);
  if (time_ns - sample.time_ns > stale_timeout_ns_) {
    return unknownSnapshot(sample.time_ns);
  }
  return snapshotFromSample(sample);
}

RobotStateEstimatorStats RobotStateEstimator::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  RobotStateEstimatorStats result = stats_;
  result.retained_samples = samples_.size();
  return result;
}

RobotStateSnapshot RobotStateEstimator::snapshotFromSample(
    const Sample& sample) {
  RobotStateSnapshot snapshot;
  snapshot.rotating = sample.machine.rotating;
  snapshot.source_time_ns = sample.time_ns;
  snapshot.yaw_rate_rad_s = sample.yaw_rate_rad_s;
  return snapshot;
}

RobotStateSnapshot RobotStateEstimator::unknownSnapshot(
    TimeNanoseconds observed_at_ns) {
  RobotStateSnapshot snapshot;
  snapshot.rotating.observed_at_ns = observed_at_ns;
  snapshot.rotating.since_ns = observed_at_ns;
  snapshot.source_time_ns = observed_at_ns;
  return snapshot;
}

bool RobotStateEstimator::validObservation(
    const OdometryObservation& observation,
    double* normalized_yaw_rad) {
  if (normalized_yaw_rad == nullptr || !finiteVector(observation.position) ||
      !std::isfinite(observation.orientation.x()) ||
      !std::isfinite(observation.orientation.y()) ||
      !std::isfinite(observation.orientation.z()) ||
      !std::isfinite(observation.orientation.w())) {
    return false;
  }
  const double norm = observation.orientation.norm();
  if (!std::isfinite(norm) || norm <= kQuaternionNormEpsilon) {
    return false;
  }
  const Eigen::Quaterniond orientation = observation.orientation.normalized();
  const double sin_yaw =
      2.0 * (orientation.w() * orientation.z() +
             orientation.x() * orientation.y());
  const double cos_yaw =
      1.0 - 2.0 * (orientation.y() * orientation.y() +
                   orientation.z() * orientation.z());
  *normalized_yaw_rad = std::atan2(sin_yaw, cos_yaw);
  return std::isfinite(*normalized_yaw_rad);
}

void RobotStateEstimator::pruneLocked() {
  if (samples_.empty()) {
    return;
  }
  const TimeNanoseconds retained_horizon_ns =
      history_ns_ + rotation_window_ns_ + rotation_exit_hold_ns_;
  const TimeNanoseconds cutoff_ns =
      latest_seen_ns_ > retained_horizon_ns
          ? latest_seen_ns_ - retained_horizon_ns
          : 0;
  std::size_t remove_count = 0;
  while (remove_count < samples_.size() &&
         samples_[remove_count].time_ns < cutoff_ns) {
    ++remove_count;
  }
  if (samples_.size() - remove_count > config_.max_odom_samples) {
    remove_count = samples_.size() - config_.max_odom_samples;
  }
  if (remove_count == 0) {
    return;
  }
  const Sample& boundary = samples_[remove_count - 1];
  seed_time_ns_ = boundary.time_ns;
  seed_machine_ = boundary.machine;
  samples_.erase(samples_.begin(), samples_.begin() + remove_count);
}

void RobotStateEstimator::recomputeLocked() {
  MachineState machine = seed_machine_;
  TimeNanoseconds previous_time_ns = seed_time_ns_;
  std::size_t segment_start = 0;
  std::size_t reference = 0;

  for (std::size_t index = 0; index < samples_.size(); ++index) {
    Sample& sample = samples_[index];
    sample.yaw_rate_rad_s.reset();
    const bool gap = !sample.valid ||
                     (previous_time_ns > 0 &&
                      sample.time_ns - previous_time_ns > stale_timeout_ns_);
    if (gap) {
      machine = {};
      machine.rotating.observed_at_ns = sample.time_ns;
      machine.rotating.since_ns = sample.time_ns;
      segment_start = sample.valid ? index : index + 1;
      reference = segment_start;
      sample.machine = machine;
      previous_time_ns = sample.time_ns;
      continue;
    }

    const TimeNanoseconds target_ns = sample.time_ns - rotation_window_ns_;
    reference = std::max(reference, segment_start);
    while (reference + 1 < index &&
           samples_[reference + 1].time_ns <= target_ns) {
      ++reference;
    }
    const bool has_window =
        reference < index && samples_[reference].valid &&
        samples_[reference].time_ns <= target_ns;
    if (has_window) {
      const double delta_seconds =
          static_cast<double>(sample.time_ns - samples_[reference].time_ns) /
          kNanosecondsPerSecond;
      const double yaw_rate = std::abs(shortestAngularDistance(
                                   samples_[reference].yaw_rad,
                                   sample.yaw_rad)) /
                              delta_seconds;
      if (std::isfinite(yaw_rate)) {
        sample.yaw_rate_rad_s = yaw_rate;
        const RobotActivityValue current = machine.rotating.value;
        if (current == RobotActivityValue::kUnknown) {
          machine.rotating.value =
              yaw_rate >= config_.rotation_enter_rad_s
                  ? RobotActivityValue::kActive
                  : RobotActivityValue::kInactive;
          machine.rotating.since_ns = sample.time_ns;
          machine.below_exit_since_ns = 0;
        } else if (current == RobotActivityValue::kInactive) {
          if (yaw_rate >= config_.rotation_enter_rad_s) {
            machine.rotating.value = RobotActivityValue::kActive;
            machine.rotating.since_ns = sample.time_ns;
            machine.below_exit_since_ns = 0;
          }
        } else if (yaw_rate <= config_.rotation_exit_rad_s) {
          if (machine.below_exit_since_ns == 0) {
            machine.below_exit_since_ns = sample.time_ns;
          }
          if (sample.time_ns - machine.below_exit_since_ns >=
              rotation_exit_hold_ns_) {
            machine.rotating.value = RobotActivityValue::kInactive;
            machine.rotating.since_ns =
                machine.below_exit_since_ns + rotation_exit_hold_ns_;
            machine.below_exit_since_ns = 0;
          }
        } else {
          machine.below_exit_since_ns = 0;
        }
      }
    }
    machine.rotating.observed_at_ns = sample.time_ns;
    sample.machine = machine;
    previous_time_ns = sample.time_ns;
  }
}

RobotActivityValue RobotStateEstimator::latestValueLocked() const {
  return samples_.empty() ? seed_machine_.rotating.value
                          : samples_.back().machine.rotating.value;
}

RobotStateSnapshot RobotStateEstimator::latestSnapshotLocked() const {
  if (samples_.empty()) {
    RobotStateSnapshot snapshot = unknownSnapshot(seed_time_ns_);
    snapshot.rotating = seed_machine_.rotating;
    return snapshot;
  }
  return snapshotFromSample(samples_.back());
}

}  // namespace roomie
