#include "roomie/pipeline/publisher_persistence_thread.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <thread>

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "roomie/utils/run_logger.hpp"
#include "roomie/utils/visualization_utils.hpp"

namespace roomie {
namespace {

float packRgbAsFloat(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  const std::uint32_t packed =
      (static_cast<std::uint32_t>(r) << 16U) |
      (static_cast<std::uint32_t>(g) << 8U) |
      static_cast<std::uint32_t>(b);
  float value = 0.0f;
  std::memcpy(&value, &packed, sizeof(value));
  return value;
}

std::string instanceText(const InstanceRecord& instance, bool show_score) {
  std::ostringstream stream;
  stream << (instance.label.empty() ? std::to_string(instance.track_id) : instance.label);
  if (show_score) {
    stream << " " << static_cast<int>(std::round(instance.confidence * 100.0f)) << "%";
  }
  return stream.str();
}

}  // namespace

PublisherPersistenceThread::PublisherPersistenceThread(rclcpp::Node& node,
                                                       const InstanceStore& instance_store,
                                                       const MapThread& map_thread,
                                                       PipelineConfig config)
    : WorkerThread("publisher_persistence_thread"),
      node_(node),
      instance_store_(instance_store),
      map_thread_(map_thread),
      config_(std::move(config)) {
  marker_pub_ = node_.create_publisher<visualization_msgs::msg::MarkerArray>(
      "/roomie/instances",
      rclcpp::QoS(1).reliable());
  map_surface_pub_ = node_.create_publisher<sensor_msgs::msg::PointCloud2>(
      config_.tsdf_output_topic,
      rclcpp::QoS(1).reliable());
  map_stats_pub_ = node_.create_publisher<std_msgs::msg::String>(
      "/roomie/map_stats",
      rclcpp::QoS(1).reliable());
}

void PublisherPersistenceThread::run() {
  const auto period = std::chrono::duration<double>(config_.publish_period_sec);
  while (!stopRequested()) {
    const MapBackendSnapshot map_snapshot = map_thread_.debugSnapshot();
    map_surface_pub_->publish(buildMapSurfaceCloud(map_snapshot));
    const std_msgs::msg::String map_stats = buildMapStats(map_snapshot);
    map_stats_pub_->publish(map_stats);
    marker_pub_->publish(buildInstanceMarkers());
    const auto now = std::chrono::steady_clock::now();
    if (now - last_log_time_ >=
        std::chrono::duration<double>(config_.file_logging_period_sec)) {
      last_log_time_ = now;
      RunLogger::logGlobal("map", map_stats.data);
      RunLogger::logGlobal("publisher",
                           "published surface_points=" +
                               std::to_string(map_snapshot.debug_surface_points.size()) +
                               " instances=" +
                               std::to_string(instance_store_.snapshotInstances().size()));
    }
    std::this_thread::sleep_for(period);
  }
}

sensor_msgs::msg::PointCloud2 PublisherPersistenceThread::buildMapSurfaceCloud(
    const MapBackendSnapshot& snapshot) const {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id = config_.world_frame;
  cloud.header.stamp = node_.now();
  cloud.height = 1;
  cloud.is_bigendian = false;
  cloud.is_dense = false;

  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2Fields(
      6,
      "x",
      1,
      sensor_msgs::msg::PointField::FLOAT32,
      "y",
      1,
      sensor_msgs::msg::PointField::FLOAT32,
      "z",
      1,
      sensor_msgs::msg::PointField::FLOAT32,
      "rgb",
      1,
      sensor_msgs::msg::PointField::FLOAT32,
      "intensity",
      1,
      sensor_msgs::msg::PointField::FLOAT32,
      "weight",
      1,
      sensor_msgs::msg::PointField::FLOAT32);
  modifier.resize(snapshot.debug_surface_points.size());

  sensor_msgs::PointCloud2Iterator<float> x_it(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> y_it(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> z_it(cloud, "z");
  sensor_msgs::PointCloud2Iterator<float> rgb_it(cloud, "rgb");
  sensor_msgs::PointCloud2Iterator<float> intensity_it(cloud, "intensity");
  sensor_msgs::PointCloud2Iterator<float> weight_it(cloud, "weight");
  for (const MapSurfacePoint& point : snapshot.debug_surface_points) {
    *x_it = point.position_world.x();
    *y_it = point.position_world.y();
    *z_it = point.position_world.z();
    *rgb_it = packRgbAsFloat(point.r, point.g, point.b);
    *intensity_it = point.intensity;
    *weight_it = point.weight;
    ++x_it;
    ++y_it;
    ++z_it;
    ++rgb_it;
    ++intensity_it;
    ++weight_it;
  }

  return cloud;
}

std_msgs::msg::String PublisherPersistenceThread::buildMapStats(
    const MapBackendSnapshot& snapshot) const {
  std_msgs::msg::String message;
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(2)
         << "map_backend=" << config_.map_backend
         << " map_version=" << snapshot.map_version
         << " has_map=" << (snapshot.has_map ? "true" : "false")
         << " cache_ready=" << (snapshot.surface_cache_ready ? "true" : "false")
         << " surface_points=" << snapshot.surface_points_world.size()
         << " cached_surface_points=" << snapshot.cached_surface_points
         << " tsdf_blocks=" << snapshot.tsdf_blocks
         << " selected_blocks=" << snapshot.selected_blocks
         << " cache_rebuilds=" << snapshot.cache_rebuilds
         << " voxels_scanned=" << snapshot.surface_voxels_scanned
         << " snapshot_ms=" << snapshot.snapshot_ms
         << " surface_extract_ms=" << snapshot.surface_extract_ms
         << " cache_build_ms=" << snapshot.cache_build_ms
         << " frustum_filter_ms=" << snapshot.frustum_filter_ms
         << " publish_period_sec=" << config_.publish_period_sec;
  message.data = stream.str();
  return message;
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
    const RgbColor color = colorForLabel(instance.label, instance.semantic_id);
    visualization_msgs::msg::Marker box;
    box.header.frame_id = config_.world_frame;
    box.header.stamp = node_.now();
    box.ns = "roomie_instances";
    box.id = marker_id++;
    box.type = visualization_msgs::msg::Marker::LINE_LIST;
    box.action = visualization_msgs::msg::Marker::ADD;
    box.pose.orientation.w = 1.0;
    box.scale.x = 0.03;
    box.color.r = color.r;
    box.color.g = color.g;
    box.color.b = color.b;
    box.color.a = 0.95f;
    fillLineListFromCorners(
        yawObbCorners(instance.center_world, instance.size_m, instance.yaw_rad),
        &box);
    markers.markers.push_back(box);

    visualization_msgs::msg::Marker text;
    text.header.frame_id = config_.world_frame;
    text.header.stamp = box.header.stamp;
    text.ns = "roomie_instance_labels";
    text.id = marker_id++;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position.x = instance.center_world.x();
    text.pose.position.y = instance.center_world.y();
    text.pose.position.z = instance.center_world.z();
    text.pose.position.z += std::max(0.05f, instance.size_m.z() * 0.6f);
    text.scale.z = 0.12;
    text.color.r = 1.0f;
    text.color.g = 1.0f;
    text.color.b = 1.0f;
    text.color.a = 1.0f;
    text.text = instanceText(instance, config_.show_3d_label_score);
    markers.markers.push_back(text);
  }

  return markers;
}

}  // namespace roomie
