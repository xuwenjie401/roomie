#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "roomie/pipeline/robot_mask_generator.hpp"
#include "roomie/pipeline/ros_io_thread.hpp"

namespace roomie {
namespace {

RobotMaskGeneratorConfig generatorConfig() {
  const std::filesystem::path config_root =
      std::filesystem::path(ROOMIE_SOURCE_DIR) / "config" / "robots" / "G2";
  RobotMaskGeneratorConfig config;
  config.robot_config = config_root / "robot.yaml";
  config.camera_config = config_root / "cameras.yaml";
  config.reuse_translation_epsilon_m = 5.0e-6;
  config.reuse_rotation_epsilon_rad = 5.0e-6;
  return config;
}

RobotMaskGenerator makeGenerator() {
  return RobotMaskGenerator(generatorConfig());
}

RobotPoseSnapshot identityPose(const RobotMaskGenerator& generator,
                               TimeNanoseconds time_ns) {
  RobotPoseSnapshot pose;
  pose.time_ns = time_ns;
  for (const std::string& frame : generator.requiredFrames()) {
    pose.root_T_frame.emplace(frame, Eigen::Isometry3d::Identity());
  }
  return pose;
}

geometry_msgs::msg::TransformStamped identityTransform(
    const std::string& parent,
    const std::string& child) {
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = parent;
  transform.child_frame_id = child;
  transform.transform.rotation.w = 1.0;
  return transform;
}

sensor_msgs::msg::Image rgbImage(std::int32_t stamp_seconds) {
  sensor_msgs::msg::Image image;
  image.header.stamp.sec = stamp_seconds;
  image.header.frame_id = "head_color";
  image.width = 640;
  image.height = 400;
  image.encoding = "bgr8";
  image.step = image.width * 3;
  image.data.resize(static_cast<std::size_t>(image.step) * image.height);
  return image;
}

sensor_msgs::msg::Image depthImage(std::int32_t stamp_seconds) {
  sensor_msgs::msg::Image image;
  image.header.stamp.sec = stamp_seconds;
  image.header.frame_id = "head_color";
  image.width = 640;
  image.height = 400;
  image.encoding = "16UC1";
  image.step = image.width * 2;
  image.data.resize(static_cast<std::size_t>(image.step) * image.height);
  return image;
}

template <typename Message>
void setStamp(Message* message, double stamp_seconds) {
  const auto whole_seconds = static_cast<std::int32_t>(std::floor(stamp_seconds));
  message->header.stamp.sec = whole_seconds;
  message->header.stamp.nanosec = static_cast<std::uint32_t>(std::llround(
      (stamp_seconds - static_cast<double>(whole_seconds)) * 1.0e9));
}

sensor_msgs::msg::Image rgbImageAt(double stamp_seconds) {
  sensor_msgs::msg::Image image =
      rgbImage(static_cast<std::int32_t>(std::floor(stamp_seconds)));
  setStamp(&image, stamp_seconds);
  return image;
}

sensor_msgs::msg::Image depthImageAt(double stamp_seconds) {
  sensor_msgs::msg::Image image =
      depthImage(static_cast<std::int32_t>(std::floor(stamp_seconds)));
  setStamp(&image, stamp_seconds);
  return image;
}

nav_msgs::msg::Odometry odometryAt(double stamp_seconds, double yaw_rad) {
  nav_msgs::msg::Odometry odometry;
  setStamp(&odometry, stamp_seconds);
  odometry.header.frame_id = "base_link";
  odometry.child_frame_id = "map";
  const Eigen::Quaterniond orientation(
      Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()));
  odometry.pose.pose.orientation.x = orientation.x();
  odometry.pose.pose.orientation.y = orientation.y();
  odometry.pose.pose.orientation.z = orientation.z();
  odometry.pose.pose.orientation.w = orientation.w();
  return odometry;
}

template <typename Predicate>
bool spinUntil(rclcpp::executors::SingleThreadedExecutor* executor,
               Predicate predicate,
               std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    executor->spin_some();
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  executor->spin_some();
  return predicate();
}

class ScopedRclcppInit {
 public:
  ScopedRclcppInit() {
    if (!rclcpp::ok()) {
      int argc = 0;
      char** argv = nullptr;
      rclcpp::init(argc, argv);
      owns_context_ = true;
    }
  }

  ~ScopedRclcppInit() {
    if (owns_context_ && rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }

 private:
  bool owns_context_ = false;
};

TEST(RobotMaskGeneratorTest, LoadsCanonicalG2CameraProfiles) {
  RobotMaskGenerator generator = makeGenerator();

  EXPECT_EQ(generator.rootFrame(), "base_link");
  EXPECT_FALSE(generator.requiredFrames().empty());
  EXPECT_TRUE(generator.hasCamera("head_color"));
  EXPECT_TRUE(generator.hasCamera("hand_left_color"));
  EXPECT_TRUE(generator.hasCamera("hand_right_color"));

  const RobotMaskCameraInfo head = generator.cameraInfo("head_color");
  EXPECT_EQ(head.width, 640);
  EXPECT_EQ(head.height, 400);
  EXPECT_NEAR(head.fx, 305.2087402344, 1.0e-9);
  EXPECT_THROW(generator.cameraInfo("unknown"), std::invalid_argument);
}

TEST(RobotMaskGeneratorTest, ReusesAgainstLastRenderedPoseWithoutCumulativeDrift) {
  RobotMaskGenerator generator = makeGenerator();
  RobotPoseSnapshot pose = identityPose(generator, 1);
  ASSERT_FALSE(generator.requiredFrames().empty());
  const std::string moving_frame = generator.requiredFrames().front();

  const RobotMaskResult first = generator.generate("head_color", pose);
  ASSERT_TRUE(first.mask);
  EXPECT_FALSE(first.reused);
  EXPECT_EQ(first.mask->width, 640);
  EXPECT_EQ(first.mask->height, 400);
  EXPECT_EQ(first.mask->channels, 1);
  EXPECT_EQ(first.mask->encoding, "mono8");

  pose.time_ns = 2;
  pose.root_T_frame.at(moving_frame).translation().x() = 4.0e-6;
  const RobotMaskResult below_threshold =
      generator.generate("head_color", pose);
  EXPECT_TRUE(below_threshold.reused);
  EXPECT_EQ(below_threshold.mask.get(), first.mask.get());

  // This is only 2 um beyond the preceding input, but 6 um beyond the pose
  // which actually rendered the cached mask. It must trigger a redraw.
  pose.time_ns = 3;
  pose.root_T_frame.at(moving_frame).translation().x() = 6.0e-6;
  const RobotMaskResult cumulative_motion =
      generator.generate("head_color", pose);
  EXPECT_FALSE(cumulative_motion.reused);
  EXPECT_NE(cumulative_motion.mask.get(), first.mask.get());

  pose.time_ns = 4;
  pose.root_T_frame.at(moving_frame).linear() =
      Eigen::AngleAxisd(6.0e-6, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const RobotMaskResult rotation = generator.generate("head_color", pose);
  EXPECT_FALSE(rotation.reused);
}

TEST(RobotMaskGeneratorTest, KeepsIndependentCachesForEveryCamera) {
  RobotMaskGenerator generator = makeGenerator();
  RobotPoseSnapshot pose = identityPose(generator, 10);

  const RobotMaskResult head = generator.generate("head_color", pose);
  const RobotMaskResult left = generator.generate("hand_left_color", pose);
  EXPECT_FALSE(head.reused);
  EXPECT_FALSE(left.reused);
  ASSERT_TRUE(head.mask);
  ASSERT_TRUE(left.mask);
  EXPECT_EQ(head.mask->width, 640);
  EXPECT_EQ(left.mask->width, 1280);

  pose.time_ns = 11;
  const RobotMaskResult head_reused = generator.generate("head_color", pose);
  const RobotMaskResult left_reused =
      generator.generate("hand_left_color", pose);
  EXPECT_TRUE(head_reused.reused);
  EXPECT_TRUE(left_reused.reused);
  EXPECT_EQ(head_reused.mask.get(), head.mask.get());
  EXPECT_EQ(left_reused.mask.get(), left.mask.get());
}

TEST(RobotMaskGeneratorTest, RosIoRejectsCameraGeometryOutsideMaskProfile) {
  ThreadSafeQueue<FrameBundlePtr> mapping_queue(2);
  ThreadSafeQueue<FrameBundlePtr> detection_queue(2);
  RosIoThread ros_io(mapping_queue, detection_queue);

  RosIoSubscriptionConfig config;
  config.robot_mask_generator =
      std::make_shared<RobotMaskGenerator>(generatorConfig());
  RosCameraSubscriptionConfig camera;
  camera.camera_id = "head_color";
  camera.camera_frame = "head_color";
  camera.fallback_intrinsics =
      CameraIntrinsics{640, 400, 305.2087402344f, 305.0057678223f,
                       318.5672912598f, 204.0587768555f};
  config.cameras.push_back(camera);
  EXPECT_NO_THROW(ros_io.configure(config));

  config.cameras.front().fallback_intrinsics.width = 641;
  EXPECT_THROW(ros_io.configure(config), std::invalid_argument);
}

TEST(RobotMaskGeneratorTest,
     RosIoWaitsForExactTfAndDepthThenReusesPerCameraMask) {
  ScopedRclcppInit rclcpp_init;
  auto io_node = std::make_shared<rclcpp::Node>("roomie_mask_io_test");
  auto publisher_node =
      std::make_shared<rclcpp::Node>("roomie_mask_publisher_test");

  ThreadSafeQueue<FrameBundlePtr> mapping_queue(2);
  ThreadSafeQueue<FrameBundlePtr> detection_queue(4);
  RosIoThread ros_io(mapping_queue, detection_queue);
  auto generator = std::make_shared<RobotMaskGenerator>(generatorConfig());

  RosIoSubscriptionConfig config;
  config.world_frame = "map";
  config.tf_topic.clear();
  config.tf_static_topic = "/roomie/test/mask/tf_static";
  config.map_mode = MapMode::kFrozen;
  config.max_perception_fps = 0.0;
  config.robot_mask_generator = generator;
  RosCameraSubscriptionConfig camera;
  camera.camera_id = "head_color";
  camera.camera_frame = "head_color";
  camera.rgb_topic = "/roomie/test/mask/head_color";
  camera.depth_topic = "/roomie/test/mask/head_depth";
  camera.fallback_intrinsics =
      CameraIntrinsics{640, 400, 305.2087402344f, 305.0057678223f,
                       318.5672912598f, 204.0587768555f};
  camera.enable_mapping = true;
  camera.enable_detection = false;
  config.cameras.push_back(camera);
  ros_io.configure(config);
  ros_io.attachNode(*io_node);

  const auto image_publisher =
      publisher_node->create_publisher<sensor_msgs::msg::Image>(
          camera.rgb_topic, rclcpp::QoS(4).best_effort().durability_volatile());
  const auto depth_publisher =
      publisher_node->create_publisher<sensor_msgs::msg::Image>(
          camera.depth_topic,
          rclcpp::QoS(4).best_effort().durability_volatile());
  const auto tf_publisher =
      publisher_node->create_publisher<tf2_msgs::msg::TFMessage>(
          config.tf_static_topic,
          rclcpp::QoS(100).transient_local().reliable());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(io_node);
  executor.add_node(publisher_node);
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() {
        return image_publisher->get_subscription_count() == 1 &&
               depth_publisher->get_subscription_count() == 1 &&
               tf_publisher->get_subscription_count() == 1;
      },
      std::chrono::seconds(2)));

  ASSERT_FALSE(generator->requiredFrames().empty());
  const std::string omitted_frame = generator->requiredFrames().front();
  tf2_msgs::msg::TFMessage partial_tf;
  partial_tf.transforms.push_back(identityTransform("map", "head_color"));
  for (const std::string& frame : generator->requiredFrames()) {
    if (frame != omitted_frame) {
      partial_tf.transforms.push_back(
          identityTransform(generator->rootFrame(), frame));
    }
  }
  tf_publisher->publish(partial_tf);
  image_publisher->publish(rgbImage(1));
  EXPECT_FALSE(spinUntil(&executor, []() { return false; },
                         std::chrono::milliseconds(100)));
  FrameBundlePtr frame;
  EXPECT_FALSE(mapping_queue.tryPop(&frame));

  tf2_msgs::msg::TFMessage missing_tf;
  missing_tf.transforms.push_back(
      identityTransform(generator->rootFrame(), omitted_frame));
  tf_publisher->publish(missing_tf);
  EXPECT_FALSE(spinUntil(&executor,
                         [&]() { return mapping_queue.tryPop(&frame); },
                         std::chrono::milliseconds(100)));

  // Exercise the state where the exact mask exists but matching depth does
  // not. A later TF callback must only retry frame assembly; it must not enter
  // mask generation with an empty generator job.
  tf_publisher->publish(missing_tf);
  EXPECT_FALSE(spinUntil(&executor,
                         [&]() { return mapping_queue.tryPop(&frame); },
                         std::chrono::milliseconds(100)));
  depth_publisher->publish(depthImage(1));
  ASSERT_TRUE(spinUntil(&executor,
                        [&]() { return mapping_queue.tryPop(&frame); },
                        std::chrono::seconds(2)));
  ASSERT_TRUE(frame);
  ASSERT_TRUE(frame->robot_mask);
  EXPECT_EQ(frame->provenance.sensor_time_ns, 1000000000LL);
  const ImageBuffer* first_mask = frame->robot_mask.get();

  FrameBundlePtr second_frame;
  image_publisher->publish(rgbImage(2));
  depth_publisher->publish(depthImage(2));
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() { return mapping_queue.tryPop(&second_frame); },
      std::chrono::seconds(2)));
  ASSERT_TRUE(second_frame);
  EXPECT_EQ(second_frame->provenance.sensor_time_ns, 2000000000LL);
  EXPECT_EQ(second_frame->robot_mask.get(), first_mask);

  executor.remove_node(publisher_node);
  executor.remove_node(io_node);
}

TEST(RobotMaskGeneratorTest,
     RosIoEmitsOnlineDetectionBeforeDepthAndBuffersRgbForMappingJoin) {
  ScopedRclcppInit rclcpp_init;
  auto io_node =
      std::make_shared<rclcpp::Node>("roomie_independent_rgb_io_test");
  auto publisher_node =
      std::make_shared<rclcpp::Node>("roomie_independent_rgb_publisher_test");

  ThreadSafeQueue<FrameBundlePtr> mapping_queue(4);
  ThreadSafeQueue<FrameBundlePtr> detection_queue(4);
  RosIoThread ros_io(mapping_queue, detection_queue);
  auto generator = std::make_shared<RobotMaskGenerator>(generatorConfig());

  RosIoSubscriptionConfig config;
  config.world_frame = "map";
  config.tf_topic.clear();
  config.tf_static_topic = "/roomie/test/independent_rgb/tf_static";
  config.map_mode = MapMode::kOnline;
  config.max_perception_fps = 0.0;
  config.robot_mask_generator = generator;
  RosCameraSubscriptionConfig camera;
  camera.camera_id = "head_color";
  camera.camera_frame = "head_color";
  camera.rgb_topic = "/roomie/test/independent_rgb/head_color";
  camera.depth_topic = "/roomie/test/independent_rgb/head_depth";
  camera.fallback_intrinsics =
      CameraIntrinsics{640, 400, 305.2087402344f, 305.0057678223f,
                       318.5672912598f, 204.0587768555f};
  camera.enable_mapping = true;
  camera.enable_detection = true;
  config.cameras.push_back(camera);
  ros_io.configure(config);
  ros_io.attachNode(*io_node);

  const auto image_publisher =
      publisher_node->create_publisher<sensor_msgs::msg::Image>(
          camera.rgb_topic, rclcpp::QoS(4).best_effort().durability_volatile());
  const auto depth_publisher =
      publisher_node->create_publisher<sensor_msgs::msg::Image>(
          camera.depth_topic,
          rclcpp::QoS(4).best_effort().durability_volatile());
  const auto tf_publisher =
      publisher_node->create_publisher<tf2_msgs::msg::TFMessage>(
          config.tf_static_topic,
          rclcpp::QoS(100).transient_local().reliable());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(io_node);
  executor.add_node(publisher_node);
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() {
        return image_publisher->get_subscription_count() == 1 &&
               depth_publisher->get_subscription_count() == 1 &&
               tf_publisher->get_subscription_count() == 1;
      },
      std::chrono::seconds(2)));

  tf2_msgs::msg::TFMessage transforms;
  transforms.transforms.push_back(identityTransform("map", "head_color"));
  for (const std::string& frame : generator->requiredFrames()) {
    transforms.transforms.push_back(
        identityTransform(generator->rootFrame(), frame));
  }
  tf_publisher->publish(transforms);

  FrameBundlePtr first_detection;
  image_publisher->publish(rgbImage(10));
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() { return detection_queue.tryPop(&first_detection); },
      std::chrono::seconds(2)));
  ASSERT_TRUE(first_detection);
  EXPECT_EQ(first_detection->provenance.sensor_time_ns, 10000000000LL);
  EXPECT_FALSE(first_detection->depth);
  EXPECT_FALSE(first_detection->perception_candidate);
  FrameBundlePtr mapping_frame;
  EXPECT_FALSE(mapping_queue.tryPop(&mapping_frame));

  // A newer RGB must not overwrite the older image retained for the delayed
  // depth join.
  FrameBundlePtr second_detection;
  image_publisher->publish(rgbImage(11));
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() { return detection_queue.tryPop(&second_detection); },
      std::chrono::seconds(2)));
  ASSERT_TRUE(second_detection);
  EXPECT_EQ(second_detection->provenance.sensor_time_ns, 11000000000LL);
  EXPECT_FALSE(mapping_queue.tryPop(&mapping_frame));

  depth_publisher->publish(depthImage(10));
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() { return mapping_queue.tryPop(&mapping_frame); },
      std::chrono::seconds(2)));
  ASSERT_TRUE(mapping_frame);
  ASSERT_TRUE(mapping_frame->depth);
  EXPECT_EQ(mapping_frame->provenance.sensor_time_ns, 10000000000LL);
  EXPECT_EQ(mapping_frame->provenance.frame_id,
            first_detection->provenance.frame_id);
  EXPECT_EQ(mapping_frame->sync.rgb_depth_delta_ns, 0);

  executor.remove_node(publisher_node);
  executor.remove_node(io_node);
}

TEST(RobotMaskGeneratorTest,
     RosIoRobotStateGateDropsUnknownAndRotatingFramesWithoutPurgingQueues) {
  ScopedRclcppInit rclcpp_init;
  auto io_node =
      std::make_shared<rclcpp::Node>("roomie_robot_state_io_test");
  auto publisher_node =
      std::make_shared<rclcpp::Node>("roomie_robot_state_publisher_test");

  ThreadSafeQueue<FrameBundlePtr> mapping_queue(4);
  ThreadSafeQueue<FrameBundlePtr> detection_queue(4);
  RosIoThread ros_io(mapping_queue, detection_queue);
  auto generator = std::make_shared<RobotMaskGenerator>(generatorConfig());
  auto estimator = std::make_shared<RobotStateEstimator>();

  RosIoSubscriptionConfig config;
  config.world_frame = "map";
  config.odom_topic = "/roomie/test/robot_state/odom";
  config.tf_topic.clear();
  config.tf_static_topic = "/roomie/test/robot_state/tf_static";
  config.map_mode = MapMode::kOnline;
  config.max_perception_fps = 0.0;
  config.robot_mask_generator = generator;
  config.odometry_observer =
      [estimator](const OdometryObservation& observation) {
        return estimator->observeOdometry(observation);
      };
  config.robot_frame_admission =
      [estimator](const std::string&, TimeNanoseconds time_ns) {
        RobotFrameAdmissionDecision decision;
        decision.state = estimator->snapshotAt(time_ns);
        if (decision.state.rotating.value == RobotActivityValue::kActive) {
          decision.allow_mapping = false;
          decision.allow_detection = false;
          decision.reason = RobotFrameAdmissionReason::kRotating;
        } else if (decision.state.rotating.value ==
                   RobotActivityValue::kUnknown) {
          decision.allow_mapping = false;
          decision.allow_detection = false;
          decision.reason = RobotFrameAdmissionReason::kStateUnknown;
        }
        return decision;
      };
  RosCameraSubscriptionConfig camera;
  camera.camera_id = "head_color";
  camera.camera_frame = "head_color";
  camera.rgb_topic = "/roomie/test/robot_state/head_color";
  camera.depth_topic = "/roomie/test/robot_state/head_depth";
  camera.fallback_intrinsics =
      CameraIntrinsics{640, 400, 305.2087402344f, 305.0057678223f,
                       318.5672912598f, 204.0587768555f};
  camera.enable_mapping = true;
  camera.enable_detection = true;
  config.cameras.push_back(camera);
  ros_io.configure(config);
  ros_io.attachNode(*io_node);

  const auto image_publisher =
      publisher_node->create_publisher<sensor_msgs::msg::Image>(
          camera.rgb_topic, rclcpp::QoS(4).best_effort().durability_volatile());
  const auto depth_publisher =
      publisher_node->create_publisher<sensor_msgs::msg::Image>(
          camera.depth_topic,
          rclcpp::QoS(4).best_effort().durability_volatile());
  const auto odom_publisher =
      publisher_node->create_publisher<nav_msgs::msg::Odometry>(
          config.odom_topic,
          rclcpp::QoS(30).best_effort().durability_volatile());
  const auto tf_publisher =
      publisher_node->create_publisher<tf2_msgs::msg::TFMessage>(
          config.tf_static_topic,
          rclcpp::QoS(100).transient_local().reliable());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(io_node);
  executor.add_node(publisher_node);
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() {
        return image_publisher->get_subscription_count() == 1 &&
               depth_publisher->get_subscription_count() == 1 &&
               odom_publisher->get_subscription_count() == 1 &&
               tf_publisher->get_subscription_count() == 1;
      },
      std::chrono::seconds(2)));

  tf2_msgs::msg::TFMessage transforms;
  transforms.transforms.push_back(identityTransform("map", "head_color"));
  for (const std::string& frame : generator->requiredFrames()) {
    transforms.transforms.push_back(
        identityTransform(generator->rootFrame(), frame));
  }
  tf_publisher->publish(transforms);

  // With no odometry yet, conservative admission rejects both consumers.
  image_publisher->publish(rgbImageAt(10.0));
  depth_publisher->publish(depthImageAt(10.0));
  for (int spin = 0; spin < 30; ++spin) {
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  FrameBundlePtr frame;
  EXPECT_FALSE(mapping_queue.tryPop(&frame));
  EXPECT_FALSE(detection_queue.tryPop(&frame));

  const auto publish_odom_range =
      [&](double first, double last, const std::function<double(double)>& yaw) {
        for (double time = first; time <= last + 1.0e-9; time += 0.01) {
          odom_publisher->publish(odometryAt(time, yaw(time)));
          executor.spin_some();
        }
      };

  publish_odom_range(10.0, 10.20, [](double) { return 0.0; });
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() {
        return estimator->snapshotAt(10200000000LL).rotating.value ==
               RobotActivityValue::kInactive;
      },
      std::chrono::seconds(2)));
  image_publisher->publish(rgbImageAt(10.20));
  depth_publisher->publish(depthImageAt(10.20));
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() { return !mapping_queue.empty() && !detection_queue.empty(); },
      std::chrono::seconds(2)));

  // The stable frame is deliberately left queued. Entering rotation must only
  // reject the matching rotating frame, never purge earlier admitted work.
  publish_odom_range(10.21, 10.60, [](double time) {
    return 0.20 * (time - 10.20);
  });
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() {
        return estimator->snapshotAt(10600000000LL).rotating.value ==
               RobotActivityValue::kActive;
      },
      std::chrono::seconds(2)));
  image_publisher->publish(rgbImageAt(10.60));
  depth_publisher->publish(depthImageAt(10.60));
  for (int spin = 0; spin < 30; ++spin) {
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

  FrameBundlePtr stable_mapping;
  FrameBundlePtr stable_detection;
  ASSERT_TRUE(mapping_queue.tryPop(&stable_mapping));
  ASSERT_TRUE(detection_queue.tryPop(&stable_detection));
  ASSERT_TRUE(stable_mapping);
  ASSERT_TRUE(stable_detection);
  EXPECT_EQ(stable_mapping->provenance.sensor_time_ns, 10200000000LL);
  EXPECT_EQ(stable_detection->provenance.sensor_time_ns, 10200000000LL);
  EXPECT_EQ(stable_mapping->robot_state.rotating.value,
            RobotActivityValue::kInactive);
  EXPECT_FALSE(mapping_queue.tryPop(&frame));
  EXPECT_FALSE(detection_queue.tryPop(&frame));

  const double stopped_yaw = 0.20 * (10.60 - 10.20);
  publish_odom_range(10.61, 11.10,
                     [stopped_yaw](double) { return stopped_yaw; });
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() {
        return estimator->snapshotAt(11100000000LL).rotating.value ==
               RobotActivityValue::kInactive;
      },
      std::chrono::seconds(2)));
  image_publisher->publish(rgbImageAt(11.10));
  depth_publisher->publish(depthImageAt(11.10));
  ASSERT_TRUE(spinUntil(
      &executor,
      [&]() { return !mapping_queue.empty() && !detection_queue.empty(); },
      std::chrono::seconds(2)));
  ASSERT_TRUE(mapping_queue.tryPop(&frame));
  ASSERT_TRUE(frame);
  EXPECT_EQ(frame->robot_state.rotating.value,
            RobotActivityValue::kInactive);

  executor.remove_node(publisher_node);
  executor.remove_node(io_node);
}

}  // namespace
}  // namespace roomie
