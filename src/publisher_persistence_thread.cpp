#include "roomie/pipeline/publisher_persistence_thread.hpp"

#include <chrono>
#include <cmath>
#include <thread>

namespace roomie {

PublisherPersistenceThread::PublisherPersistenceThread(rclcpp::Node& node,
                                                       const InstanceStore& instance_store,
                                                       PipelineConfig config)
    : WorkerThread("publisher_persistence_thread"),
      node_(node),
      instance_store_(instance_store),
      config_(std::move(config)) {
  marker_pub_ = node_.create_publisher<visualization_msgs::msg::MarkerArray>(
      "/roomie/instances",
      rclcpp::QoS(1).reliable());
}

void PublisherPersistenceThread::run() {
  const auto period = std::chrono::duration<double>(config_.publish_period_sec);
  while (!stopRequested()) {
    marker_pub_->publish(buildInstanceMarkers());
    std::this_thread::sleep_for(period);
  }
}

visualization_msgs::msg::MarkerArray PublisherPersistenceThread::buildInstanceMarkers() const {
  visualization_msgs::msg::MarkerArray markers;

  visualization_msgs::msg::Marker clear;
  clear.header.frame_id = config_.world_frame;
  clear.header.stamp = node_.now();
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  markers.markers.push_back(clear);

  const auto instances = instance_store_.snapshotInstances();
  int marker_id = 1;
  for (const InstanceRecord& instance : instances) {
    visualization_msgs::msg::Marker box;
    box.header.frame_id = config_.world_frame;
    box.header.stamp = node_.now();
    box.ns = "roomie_instances";
    box.id = marker_id++;
    box.type = visualization_msgs::msg::Marker::CUBE;
    box.action = visualization_msgs::msg::Marker::ADD;
    box.pose.position.x = instance.center_world.x();
    box.pose.position.y = instance.center_world.y();
    box.pose.position.z = instance.center_world.z();
    box.pose.orientation.z = std::sin(0.5 * instance.yaw_rad);
    box.pose.orientation.w = std::cos(0.5 * instance.yaw_rad);
    box.scale.x = std::max(0.01f, instance.size_m.x());
    box.scale.y = std::max(0.01f, instance.size_m.y());
    box.scale.z = std::max(0.01f, instance.size_m.z());
    box.color.r = 0.1f;
    box.color.g = 0.8f;
    box.color.b = 0.9f;
    box.color.a = 0.35f;
    markers.markers.push_back(box);

    visualization_msgs::msg::Marker text;
    text.header.frame_id = config_.world_frame;
    text.header.stamp = box.header.stamp;
    text.ns = "roomie_instance_labels";
    text.id = marker_id++;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position = box.pose.position;
    text.pose.position.z += std::max(0.05f, instance.size_m.z() * 0.6f);
    text.scale.z = 0.12;
    text.color.r = 1.0f;
    text.color.g = 1.0f;
    text.color.b = 1.0f;
    text.color.a = 1.0f;
    text.text = instance.label.empty() ? std::to_string(instance.track_id) : instance.label;
    markers.markers.push_back(text);
  }

  return markers;
}

}  // namespace roomie
