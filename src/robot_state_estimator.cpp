#include "roomie/pipeline/robot_state_estimator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <yaml-cpp/yaml.h>

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

std::string normalizeFrameId(std::string frame_id) {
  while (!frame_id.empty() && frame_id.front() == '/') {
    frame_id.erase(frame_id.begin());
  }
  return frame_id;
}

NavigationPostureGroup postureGroupFromName(const std::string& name) {
  if (name == "body") {
    return NavigationPostureGroup::kBody;
  }
  if (name == "left_arm") {
    return NavigationPostureGroup::kLeftArm;
  }
  if (name == "right_arm") {
    return NavigationPostureGroup::kRightArm;
  }
  throw std::invalid_argument("unknown navigation posture group '" + name + "'");
}

std::vector<double> fixedVector(const YAML::Node& node,
                                std::size_t expected_size,
                                const std::string& field_name) {
  if (!node || !node.IsSequence() || node.size() != expected_size) {
    throw std::invalid_argument("navigation posture " + field_name +
                                " must contain " +
                                std::to_string(expected_size) + " values");
  }
  std::vector<double> values;
  values.reserve(expected_size);
  for (const YAML::Node& value : node) {
    const double parsed = value.as<double>();
    if (!std::isfinite(parsed)) {
      throw std::invalid_argument("navigation posture " + field_name +
                                  " must be finite");
    }
    values.push_back(parsed);
  }
  return values;
}

double quaternionDistance(const Eigen::Quaterniond& lhs,
                          const Eigen::Quaterniond& rhs) {
  const double dot = std::clamp(std::abs(lhs.dot(rhs)), 0.0, 1.0);
  return 2.0 * std::acos(dot);
}

template <typename Facet>
void resetPostureFacet(Facet* facet, TimeNanoseconds time_ns) {
  *facet = {};
  facet->status.observed_at_ns = time_ns;
  facet->status.since_ns = time_ns;
}

template <typename Facet>
void updatePostureFacet(bool valid,
                        bool beyond_enter,
                        bool inside_exit,
                        TimeNanoseconds time_ns,
                        TimeNanoseconds enter_hold_ns,
                        TimeNanoseconds exit_hold_ns,
                        Facet* facet) {
  if (!valid) {
    resetPostureFacet(facet, time_ns);
    return;
  }

  if (facet->status.value == RobotActivityValue::kUnknown) {
    facet->inside_exit_since_ns = 0;
    if (!beyond_enter) {
      facet->status.value = RobotActivityValue::kInactive;
      facet->status.since_ns = time_ns;
      facet->above_enter_since_ns = 0;
    } else {
      if (facet->above_enter_since_ns == 0) {
        facet->above_enter_since_ns = time_ns;
      }
      if (time_ns - facet->above_enter_since_ns >= enter_hold_ns) {
        facet->status.value = RobotActivityValue::kActive;
        facet->status.since_ns =
            facet->above_enter_since_ns + enter_hold_ns;
        facet->above_enter_since_ns = 0;
      }
    }
  } else if (facet->status.value == RobotActivityValue::kInactive) {
    facet->inside_exit_since_ns = 0;
    if (beyond_enter) {
      if (facet->above_enter_since_ns == 0) {
        facet->above_enter_since_ns = time_ns;
      }
      if (time_ns - facet->above_enter_since_ns >= enter_hold_ns) {
        facet->status.value = RobotActivityValue::kActive;
        facet->status.since_ns =
            facet->above_enter_since_ns + enter_hold_ns;
        facet->above_enter_since_ns = 0;
      }
    } else {
      facet->above_enter_since_ns = 0;
    }
  } else {
    facet->above_enter_since_ns = 0;
    if (inside_exit) {
      if (facet->inside_exit_since_ns == 0) {
        facet->inside_exit_since_ns = time_ns;
      }
      if (time_ns - facet->inside_exit_since_ns >= exit_hold_ns) {
        facet->status.value = RobotActivityValue::kInactive;
        facet->status.since_ns =
            facet->inside_exit_since_ns + exit_hold_ns;
        facet->inside_exit_since_ns = 0;
      }
    } else {
      facet->inside_exit_since_ns = 0;
    }
  }
  facet->status.observed_at_ns = time_ns;
}

}  // namespace

NavigationPosture loadNavigationPosture(
    const std::filesystem::path& config_path) {
  YAML::Node document;
  try {
    document = YAML::LoadFile(config_path.string());
  } catch (const YAML::Exception& error) {
    throw std::invalid_argument("failed to load navigation posture '" +
                                config_path.string() + "': " + error.what());
  }

  NavigationPosture posture;
  posture.root_frame =
      normalizeFrameId(document["root_frame"].as<std::string>(""));
  const YAML::Node links = document["links"];
  if (posture.root_frame.empty() || !links || !links.IsMap()) {
    throw std::invalid_argument(
        "navigation posture requires root_frame and a links map");
  }

  std::unordered_set<std::string> frame_ids;
  std::array<std::size_t, 3> group_counts{};
  for (const auto& entry : links) {
    NavigationLinkPose pose;
    pose.frame_id = normalizeFrameId(entry.first.as<std::string>());
    const YAML::Node configured = entry.second;
    pose.group = postureGroupFromName(configured["group"].as<std::string>(""));
    const std::vector<double> translation =
        fixedVector(configured["translation"], 3,
                    pose.frame_id + ".translation");
    const std::vector<double> rotation =
        fixedVector(configured["rotation_xyzw"], 4,
                    pose.frame_id + ".rotation_xyzw");
    pose.position =
        Eigen::Vector3d(translation[0], translation[1], translation[2]);
    pose.orientation = Eigen::Quaterniond(
        rotation[3], rotation[0], rotation[1], rotation[2]);
    const double norm = pose.orientation.norm();
    if (pose.frame_id.empty() || pose.frame_id == posture.root_frame ||
        !frame_ids.insert(pose.frame_id).second ||
        !std::isfinite(norm) || norm <= kQuaternionNormEpsilon) {
      throw std::invalid_argument(
          "navigation posture contains an invalid or duplicate link");
    }
    pose.orientation.normalize();
    ++group_counts[static_cast<std::size_t>(pose.group)];
    posture.links.push_back(std::move(pose));
  }
  if (std::any_of(group_counts.begin(), group_counts.end(),
                  [](std::size_t count) { return count == 0; })) {
    throw std::invalid_argument(
        "navigation posture requires body, left_arm, and right_arm links");
  }
  return posture;
}

RobotStateEstimator::RobotStateEstimator(RobotStateEstimatorConfig config)
    : config_(std::move(config)) {
  if (!std::isfinite(config_.odom_history_sec) ||
      !std::isfinite(config_.odom_stale_timeout_sec) ||
      !std::isfinite(config_.rotation_window_sec) ||
      !std::isfinite(config_.rotation_enter_rad_s) ||
      !std::isfinite(config_.rotation_exit_rad_s) ||
      !std::isfinite(config_.rotation_exit_hold_sec) ||
      !std::isfinite(config_.posture_stale_timeout_sec) ||
      !std::isfinite(config_.posture_enter_hold_sec) ||
      !std::isfinite(config_.posture_exit_hold_sec) ||
      !std::isfinite(config_.body_translation_enter_m) ||
      !std::isfinite(config_.body_translation_exit_m) ||
      !std::isfinite(config_.body_rotation_enter_rad) ||
      !std::isfinite(config_.body_rotation_exit_rad) ||
      !std::isfinite(config_.arm_translation_enter_m) ||
      !std::isfinite(config_.arm_translation_exit_m) ||
      !std::isfinite(config_.arm_rotation_enter_rad) ||
      !std::isfinite(config_.arm_rotation_exit_rad) ||
      config_.odom_history_sec <= 0.0 ||
      config_.odom_stale_timeout_sec <= 0.0 ||
      config_.rotation_window_sec <= 0.0 ||
      config_.rotation_enter_rad_s <= 0.0 ||
      config_.rotation_exit_rad_s < 0.0 ||
      config_.rotation_exit_hold_sec < 0.0 ||
      config_.rotation_enter_rad_s < config_.rotation_exit_rad_s ||
      config_.posture_stale_timeout_sec <= 0.0 ||
      config_.posture_enter_hold_sec < 0.0 ||
      config_.posture_exit_hold_sec < 0.0 ||
      config_.body_translation_enter_m <= 0.0 ||
      config_.body_translation_exit_m < 0.0 ||
      config_.body_translation_enter_m < config_.body_translation_exit_m ||
      config_.body_rotation_enter_rad <= 0.0 ||
      config_.body_rotation_exit_rad < 0.0 ||
      config_.body_rotation_enter_rad < config_.body_rotation_exit_rad ||
      config_.arm_translation_enter_m <= 0.0 ||
      config_.arm_translation_exit_m < 0.0 ||
      config_.arm_translation_enter_m < config_.arm_translation_exit_m ||
      config_.arm_rotation_enter_rad <= 0.0 ||
      config_.arm_rotation_exit_rad < 0.0 ||
      config_.arm_rotation_enter_rad < config_.arm_rotation_exit_rad ||
      config_.max_odom_samples < 2 || config_.max_posture_samples < 2) {
    throw std::invalid_argument("invalid robot state estimator configuration");
  }
  const double required_history = config_.rotation_window_sec +
                                  config_.rotation_exit_hold_sec +
                                  config_.odom_stale_timeout_sec;
  const double required_posture_history =
      config_.posture_stale_timeout_sec + config_.posture_enter_hold_sec +
      config_.posture_exit_hold_sec;
  if (config_.odom_history_sec < required_history ||
      config_.odom_history_sec < required_posture_history) {
    throw std::invalid_argument(
        "robot state history is too short for the configured windows");
  }

  history_ns_ = secondsToNanosecondsChecked(config_.odom_history_sec);
  stale_timeout_ns_ =
      secondsToNanosecondsChecked(config_.odom_stale_timeout_sec);
  rotation_window_ns_ =
      secondsToNanosecondsChecked(config_.rotation_window_sec);
  rotation_exit_hold_ns_ =
      secondsToNanosecondsChecked(config_.rotation_exit_hold_sec);
  posture_stale_timeout_ns_ =
      secondsToNanosecondsChecked(config_.posture_stale_timeout_sec);
  posture_enter_hold_ns_ =
      secondsToNanosecondsChecked(config_.posture_enter_hold_sec);
  posture_exit_hold_ns_ =
      secondsToNanosecondsChecked(config_.posture_exit_hold_sec);
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

RobotPostureEstimatorUpdate RobotStateEstimator::observePosture(
    const PostureObservation& observation) {
  std::lock_guard<std::mutex> lock(mutex_);
  RobotPostureEstimatorUpdate update;
  update.previous_latest_state = latestPostureValueLocked();

  if (observation.time_ns <= 0 ||
      config_.navigation_posture.root_frame.empty() ||
      config_.navigation_posture.links.empty()) {
    update.latest = latestSnapshotLocked();
    return update;
  }

  PostureSample sample = makePostureSample(observation);
  if (!sample.valid) {
    update.latest = latestSnapshotLocked();
    return update;
  }

  const TimeNanoseconds retained_horizon_ns =
      history_ns_ + posture_enter_hold_ns_ + posture_exit_hold_ns_;
  if (latest_posture_seen_ns_ > 0 &&
      observation.time_ns < latest_posture_seen_ns_ &&
      latest_posture_seen_ns_ - observation.time_ns > retained_horizon_ns) {
    update.out_of_order = true;
    update.outcome = OdometryObservationOutcome::kTooOld;
    update.latest = latestSnapshotLocked();
    return update;
  }

  update.out_of_order = latest_posture_seen_ns_ > 0 &&
                        observation.time_ns < latest_posture_seen_ns_;
  latest_posture_seen_ns_ =
      std::max(latest_posture_seen_ns_, observation.time_ns);

  const auto found = std::lower_bound(
      posture_samples_.begin(), posture_samples_.end(), observation.time_ns,
      [](const PostureSample& candidate, TimeNanoseconds time_ns) {
        return candidate.time_ns < time_ns;
      });
  if (found != posture_samples_.end() &&
      found->time_ns == observation.time_ns) {
    *found = sample;
    update.outcome = OdometryObservationOutcome::kReplaced;
  } else {
    posture_samples_.insert(found, std::move(sample));
    update.outcome = OdometryObservationOutcome::kInserted;
  }

  recomputePostureLocked();
  prunePostureLocked();
  recomputePostureLocked();
  update.latest = latestSnapshotLocked();
  update.latest_state_changed =
      update.previous_latest_state != latestPostureValueLocked();
  return update;
}

RobotStateSnapshot RobotStateEstimator::snapshotAt(
    TimeNanoseconds time_ns) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (time_ns <= 0) {
    return unknownSnapshot(0);
  }

  RobotStateSnapshot snapshot = unknownSnapshot(0);
  if (!samples_.empty() &&
      !(latest_seen_ns_ > history_ns_ &&
        time_ns < latest_seen_ns_ - history_ns_)) {
    const auto upper = std::upper_bound(
        samples_.begin(), samples_.end(), time_ns,
        [](TimeNanoseconds query_ns, const Sample& sample) {
          return query_ns < sample.time_ns;
        });
    if (upper != samples_.begin()) {
      const Sample& sample = *std::prev(upper);
      if (time_ns - sample.time_ns <= stale_timeout_ns_) {
        snapshot = snapshotFromSample(sample);
      } else {
        snapshot.rotating.observed_at_ns = sample.time_ns;
        snapshot.rotating.since_ns = sample.time_ns;
      }
    }
  }

  if (!posture_samples_.empty() &&
      !(latest_posture_seen_ns_ > history_ns_ &&
        time_ns < latest_posture_seen_ns_ - history_ns_)) {
    const auto upper = std::upper_bound(
        posture_samples_.begin(), posture_samples_.end(), time_ns,
        [](TimeNanoseconds query_ns, const PostureSample& sample) {
          return query_ns < sample.time_ns;
        });
    if (upper != posture_samples_.begin()) {
      const PostureSample& sample = *std::prev(upper);
      if (time_ns - sample.time_ns <= posture_stale_timeout_ns_) {
        overlayPostureSnapshot(sample, &snapshot);
      } else {
        snapshot.body_bent.observed_at_ns = sample.time_ns;
        snapshot.body_bent.since_ns = sample.time_ns;
        snapshot.left_arm_active.observed_at_ns = sample.time_ns;
        snapshot.left_arm_active.since_ns = sample.time_ns;
        snapshot.right_arm_active.observed_at_ns = sample.time_ns;
        snapshot.right_arm_active.since_ns = sample.time_ns;
        snapshot.navigation_posture_deviated.observed_at_ns = sample.time_ns;
        snapshot.navigation_posture_deviated.since_ns = sample.time_ns;
        snapshot.source_time_ns =
            std::max(snapshot.source_time_ns, sample.time_ns);
      }
    }
  }
  return snapshot;
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

void RobotStateEstimator::overlayPostureSnapshot(
    const PostureSample& sample, RobotStateSnapshot* snapshot) {
  if (snapshot == nullptr) {
    return;
  }
  snapshot->body_bent = sample.machine.body.status;
  snapshot->left_arm_active = sample.machine.left_arm.status;
  snapshot->right_arm_active = sample.machine.right_arm.status;
  snapshot->navigation_posture_deviated = sample.machine.deviated;
  snapshot->source_time_ns =
      std::max(snapshot->source_time_ns, sample.time_ns);
}

RobotStateSnapshot RobotStateEstimator::unknownSnapshot(
    TimeNanoseconds observed_at_ns) {
  RobotStateSnapshot snapshot;
  snapshot.rotating.observed_at_ns = observed_at_ns;
  snapshot.rotating.since_ns = observed_at_ns;
  snapshot.near_stationary.observed_at_ns = observed_at_ns;
  snapshot.near_stationary.since_ns = observed_at_ns;
  snapshot.body_bent.observed_at_ns = observed_at_ns;
  snapshot.body_bent.since_ns = observed_at_ns;
  snapshot.left_arm_active.observed_at_ns = observed_at_ns;
  snapshot.left_arm_active.since_ns = observed_at_ns;
  snapshot.right_arm_active.observed_at_ns = observed_at_ns;
  snapshot.right_arm_active.since_ns = observed_at_ns;
  snapshot.navigation_posture_deviated.observed_at_ns = observed_at_ns;
  snapshot.navigation_posture_deviated.since_ns = observed_at_ns;
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

RobotStateEstimator::PostureSample RobotStateEstimator::makePostureSample(
    const PostureObservation& observation) const {
  PostureSample sample;
  sample.time_ns = observation.time_ns;
  sample.body_inside_exit = true;
  sample.left_arm_inside_exit = true;
  sample.right_arm_inside_exit = true;

  std::unordered_map<std::string, const LinkTransformObservation*> transforms;
  transforms.reserve(observation.transforms.size());
  for (const LinkTransformObservation& transform : observation.transforms) {
    if (normalizeFrameId(transform.parent_frame_id) !=
        config_.navigation_posture.root_frame) {
      continue;
    }
    transforms[normalizeFrameId(transform.child_frame_id)] = &transform;
  }

  std::array<bool, 3> groups_seen{};
  for (const NavigationLinkPose& reference :
       config_.navigation_posture.links) {
    const auto found = transforms.find(reference.frame_id);
    if (found == transforms.end() || found->second == nullptr) {
      return sample;
    }
    const LinkTransformObservation& actual = *found->second;
    if (!finiteVector(actual.position) ||
        !std::isfinite(actual.orientation.x()) ||
        !std::isfinite(actual.orientation.y()) ||
        !std::isfinite(actual.orientation.z()) ||
        !std::isfinite(actual.orientation.w()) ||
        !finiteVector(reference.position) ||
        !std::isfinite(reference.orientation.norm()) ||
        actual.orientation.norm() <= kQuaternionNormEpsilon ||
        reference.orientation.norm() <= kQuaternionNormEpsilon) {
      return sample;
    }

    const double translation_delta =
        (actual.position - reference.position).norm();
    const double rotation_delta = quaternionDistance(
        actual.orientation.normalized(), reference.orientation.normalized());
    const bool is_body = reference.group == NavigationPostureGroup::kBody;
    const bool beyond_enter =
        translation_delta >=
            (is_body ? config_.body_translation_enter_m
                     : config_.arm_translation_enter_m) ||
        rotation_delta >=
            (is_body ? config_.body_rotation_enter_rad
                     : config_.arm_rotation_enter_rad);
    const bool inside_exit =
        translation_delta <=
            (is_body ? config_.body_translation_exit_m
                     : config_.arm_translation_exit_m) &&
        rotation_delta <=
            (is_body ? config_.body_rotation_exit_rad
                     : config_.arm_rotation_exit_rad);
    groups_seen[static_cast<std::size_t>(reference.group)] = true;
    switch (reference.group) {
      case NavigationPostureGroup::kBody:
        sample.body_beyond_enter |= beyond_enter;
        sample.body_inside_exit &= inside_exit;
        break;
      case NavigationPostureGroup::kLeftArm:
        sample.left_arm_beyond_enter |= beyond_enter;
        sample.left_arm_inside_exit &= inside_exit;
        break;
      case NavigationPostureGroup::kRightArm:
        sample.right_arm_beyond_enter |= beyond_enter;
        sample.right_arm_inside_exit &= inside_exit;
        break;
    }
  }
  sample.valid =
      std::all_of(groups_seen.begin(), groups_seen.end(), [](bool seen) {
        return seen;
      });
  return sample;
}

void RobotStateEstimator::prunePostureLocked() {
  if (posture_samples_.empty()) {
    return;
  }
  const TimeNanoseconds retained_horizon_ns =
      history_ns_ + posture_enter_hold_ns_ + posture_exit_hold_ns_;
  const TimeNanoseconds cutoff_ns =
      latest_posture_seen_ns_ > retained_horizon_ns
          ? latest_posture_seen_ns_ - retained_horizon_ns
          : 0;
  std::size_t remove_count = 0;
  while (remove_count < posture_samples_.size() &&
         posture_samples_[remove_count].time_ns < cutoff_ns) {
    ++remove_count;
  }
  if (posture_samples_.size() - remove_count > config_.max_posture_samples) {
    remove_count = posture_samples_.size() - config_.max_posture_samples;
  }
  if (remove_count == 0) {
    return;
  }
  const PostureSample& boundary = posture_samples_[remove_count - 1];
  posture_seed_time_ns_ = boundary.time_ns;
  posture_seed_machine_ = boundary.machine;
  posture_samples_.erase(posture_samples_.begin(),
                         posture_samples_.begin() + remove_count);
}

void RobotStateEstimator::recomputePostureLocked() {
  PostureMachineState machine = posture_seed_machine_;
  TimeNanoseconds previous_time_ns = posture_seed_time_ns_;
  for (PostureSample& sample : posture_samples_) {
    const bool gap = !sample.valid ||
                     (previous_time_ns > 0 &&
                      sample.time_ns - previous_time_ns >
                          posture_stale_timeout_ns_);
    if (gap) {
      machine = {};
    }

    updatePostureFacet(sample.valid,
                       sample.body_beyond_enter,
                       sample.body_inside_exit,
                       sample.time_ns,
                       posture_enter_hold_ns_,
                       posture_exit_hold_ns_,
                       &machine.body);
    updatePostureFacet(sample.valid,
                       sample.left_arm_beyond_enter,
                       sample.left_arm_inside_exit,
                       sample.time_ns,
                       posture_enter_hold_ns_,
                       posture_exit_hold_ns_,
                       &machine.left_arm);
    updatePostureFacet(sample.valid,
                       sample.right_arm_beyond_enter,
                       sample.right_arm_inside_exit,
                       sample.time_ns,
                       posture_enter_hold_ns_,
                       posture_exit_hold_ns_,
                       &machine.right_arm);

    RobotActivityValue aggregate = RobotActivityValue::kUnknown;
    if (machine.body.status.active() || machine.left_arm.status.active() ||
        machine.right_arm.status.active()) {
      aggregate = RobotActivityValue::kActive;
    } else if (
        machine.body.status.value == RobotActivityValue::kInactive &&
        machine.left_arm.status.value == RobotActivityValue::kInactive &&
        machine.right_arm.status.value == RobotActivityValue::kInactive) {
      aggregate = RobotActivityValue::kInactive;
    }
    if (machine.deviated.value != aggregate) {
      machine.deviated.value = aggregate;
      machine.deviated.since_ns = sample.time_ns;
    }
    machine.deviated.observed_at_ns = sample.time_ns;
    sample.machine = machine;
    previous_time_ns = sample.time_ns;
  }
}

RobotActivityValue RobotStateEstimator::latestValueLocked() const {
  return samples_.empty() ? seed_machine_.rotating.value
                          : samples_.back().machine.rotating.value;
}

RobotActivityValue RobotStateEstimator::latestPostureValueLocked() const {
  return posture_samples_.empty()
             ? posture_seed_machine_.deviated.value
             : posture_samples_.back().machine.deviated.value;
}

RobotStateSnapshot RobotStateEstimator::latestSnapshotLocked() const {
  RobotStateSnapshot snapshot = unknownSnapshot(0);
  if (!samples_.empty()) {
    snapshot = snapshotFromSample(samples_.back());
  } else if (seed_time_ns_ > 0) {
    snapshot.rotating = seed_machine_.rotating;
    snapshot.source_time_ns = seed_time_ns_;
  }
  if (!posture_samples_.empty()) {
    overlayPostureSnapshot(posture_samples_.back(), &snapshot);
  } else if (posture_seed_time_ns_ > 0) {
    snapshot.body_bent = posture_seed_machine_.body.status;
    snapshot.left_arm_active = posture_seed_machine_.left_arm.status;
    snapshot.right_arm_active = posture_seed_machine_.right_arm.status;
    snapshot.navigation_posture_deviated =
        posture_seed_machine_.deviated;
    snapshot.source_time_ns =
        std::max(snapshot.source_time_ns, posture_seed_time_ns_);
  }
  return snapshot;
}

}  // namespace roomie
