#include "roomie/pipeline/robot_state_estimator.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace roomie {
namespace {

constexpr TimeNanoseconds kSecond = 1000000000LL;

OdometryObservation observation(double time_sec, double yaw_rad) {
  OdometryObservation result;
  result.time_ns =
      static_cast<TimeNanoseconds>(std::llround(time_sec * kSecond));
  result.orientation =
      Eigen::Quaterniond(Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()));
  return result;
}

void observeRange(RobotStateEstimator* estimator,
                  double first_sec,
                  double last_sec,
                  double step_sec,
                  const std::function<double(double)>& yaw_at) {
  ASSERT_NE(estimator, nullptr);
  for (double time = first_sec; time <= last_sec + 1.0e-9;
       time += step_sec) {
    estimator->observeOdometry(observation(time, yaw_at(time)));
  }
}

TEST(RobotStateEstimatorTest, EntersRotationAndUsesStrictExitDwell) {
  RobotStateEstimator estimator;
  observeRange(&estimator, 1.0, 1.20, 0.01,
               [](double) { return 0.0; });

  EXPECT_EQ(estimator.snapshotAt(1050000000LL).rotating.value,
            RobotActivityValue::kUnknown);
  const RobotStateSnapshot stationary = estimator.snapshotAt(1200000000LL);
  EXPECT_EQ(stationary.rotating.value, RobotActivityValue::kInactive);
  ASSERT_TRUE(stationary.yaw_rate_rad_s);
  EXPECT_LT(*stationary.yaw_rate_rad_s, 0.02);

  observeRange(&estimator, 1.21, 1.60, 0.01, [](double time) {
    return 0.20 * (time - 1.20);
  });
  const RobotStateSnapshot rotating = estimator.snapshotAt(1600000000LL);
  EXPECT_EQ(rotating.rotating.value, RobotActivityValue::kActive);
  ASSERT_TRUE(rotating.yaw_rate_rad_s);
  EXPECT_NEAR(*rotating.yaw_rate_rad_s, 0.20, 0.01);

  const double stopped_yaw = 0.20 * (1.60 - 1.20);
  observeRange(&estimator, 1.61, 1.85, 0.01,
               [stopped_yaw](double) { return stopped_yaw; });
  EXPECT_EQ(estimator.snapshotAt(1850000000LL).rotating.value,
            RobotActivityValue::kActive);

  observeRange(&estimator, 1.86, 2.10, 0.01,
               [stopped_yaw](double) { return stopped_yaw; });
  const RobotStateSnapshot recovered = estimator.snapshotAt(2100000000LL);
  EXPECT_EQ(recovered.rotating.value, RobotActivityValue::kInactive);
  EXPECT_GT(recovered.rotating.since_ns, 1850000000LL);
}

TEST(RobotStateEstimatorTest, HandlesRotationAcrossYawWrap) {
  RobotStateEstimator estimator;
  observeRange(&estimator, 1.0, 1.20, 0.01,
               [](double) { return 3.12; });
  observeRange(&estimator, 1.21, 1.60, 0.01, [](double time) {
    return 3.12 + 0.30 * (time - 1.20);
  });

  const RobotStateSnapshot state = estimator.snapshotAt(1600000000LL);
  EXPECT_EQ(state.rotating.value, RobotActivityValue::kActive);
  ASSERT_TRUE(state.yaw_rate_rad_s);
  EXPECT_NEAR(*state.yaw_rate_rad_s, 0.30, 0.02);
}

TEST(RobotStateEstimatorTest, OutOfOrderInputMatchesChronologicalInput) {
  RobotStateEstimator chronological;
  RobotStateEstimator shuffled;
  std::vector<OdometryObservation,
              Eigen::aligned_allocator<OdometryObservation>> observations;
  for (int index = 0; index <= 100; ++index) {
    const double time = 10.0 + 0.01 * index;
    const double yaw = index < 30 ? 0.0 : 0.25 * (time - 10.30);
    observations.push_back(observation(time, yaw));
    chronological.observeOdometry(observations.back());
  }
  for (std::size_t index = 0; index < observations.size(); index += 2) {
    if (index + 1 < observations.size()) {
      shuffled.observeOdometry(observations[index + 1]);
    }
    shuffled.observeOdometry(observations[index]);
  }

  for (const double query_sec : {10.2, 10.5, 10.8, 11.0}) {
    const TimeNanoseconds query_ns =
        static_cast<TimeNanoseconds>(std::llround(query_sec * kSecond));
    const RobotStateSnapshot expected = chronological.snapshotAt(query_ns);
    const RobotStateSnapshot actual = shuffled.snapshotAt(query_ns);
    EXPECT_EQ(actual.rotating.value, expected.rotating.value);
    EXPECT_EQ(actual.rotating.since_ns, expected.rotating.since_ns);
    EXPECT_EQ(actual.yaw_rate_rad_s.has_value(),
              expected.yaw_rate_rad_s.has_value());
    if (actual.yaw_rate_rad_s && expected.yaw_rate_rad_s) {
      EXPECT_NEAR(*actual.yaw_rate_rad_s, *expected.yaw_rate_rad_s, 1.0e-9);
    }
  }
  EXPECT_GT(shuffled.stats().out_of_order, 0U);
}

TEST(RobotStateEstimatorTest, InvalidAndStaleOdometryAreUnknown) {
  RobotStateEstimator estimator;
  observeRange(&estimator, 1.0, 1.20, 0.01,
               [](double) { return 0.0; });
  EXPECT_EQ(estimator.snapshotAt(1300000000LL).rotating.value,
            RobotActivityValue::kInactive);
  EXPECT_EQ(estimator.snapshotAt(1500000000LL).rotating.value,
            RobotActivityValue::kUnknown);

  OdometryObservation invalid = observation(2.0, 0.0);
  invalid.orientation = Eigen::Quaterniond(
      std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0);
  const RobotStateEstimatorUpdate update = estimator.observeOdometry(invalid);
  EXPECT_EQ(update.outcome, OdometryObservationOutcome::kInvalid);
  EXPECT_EQ(estimator.snapshotAt(2000000000LL).rotating.value,
            RobotActivityValue::kUnknown);

  observeRange(&estimator, 2.01, 2.15, 0.01,
               [](double) { return 0.0; });
  const RobotStateSnapshot recovered = estimator.snapshotAt(2150000000LL);
  EXPECT_EQ(recovered.rotating.value, RobotActivityValue::kInactive);
  EXPECT_EQ(recovered.near_stationary.value, RobotActivityValue::kUnknown);
  EXPECT_EQ(recovered.body_bent.value, RobotActivityValue::kUnknown);
  EXPECT_EQ(recovered.left_arm_active.value, RobotActivityValue::kUnknown);
  EXPECT_EQ(recovered.right_arm_active.value, RobotActivityValue::kUnknown);
}

TEST(RobotStateEstimatorTest, RetainedHistoryHasAHardSampleBound) {
  RobotStateEstimatorConfig config;
  config.max_odom_samples = 32;
  RobotStateEstimator estimator(config);
  observeRange(&estimator, 1.0, 11.0, 0.01,
               [](double) { return 0.0; });
  EXPECT_LE(estimator.stats().retained_samples, 32U);

  const RobotStateEstimatorUpdate too_old =
      estimator.observeOdometry(observation(1.0, 0.0));
  EXPECT_EQ(too_old.outcome, OdometryObservationOutcome::kTooOld);
}

TEST(RobotStateEstimatorTest, RejectsInconsistentConfiguration) {
  RobotStateEstimatorConfig config;
  config.rotation_enter_rad_s = 0.01;
  config.rotation_exit_rad_s = 0.02;
  EXPECT_THROW((void)RobotStateEstimator{config}, std::invalid_argument);

  config = {};
  config.odom_history_sec = 0.2;
  EXPECT_THROW((void)RobotStateEstimator{config}, std::invalid_argument);
}

}  // namespace
}  // namespace roomie
