#include "roomie/pipeline/publisher_persistence_thread.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <thread>

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "roomie/dsg/object_graph_io.hpp"
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

std::string instanceText(const InstanceRecord& instance,
                         bool show_score,
                         bool show_object_id,
                         bool show_track_id) {
  std::ostringstream stream;
  if (!instance.label.empty()) {
    stream << instance.label;
  } else if (show_track_id && instance.track_id >= 0) {
    stream << "instance";
  } else if (instance.object_id >= 0) {
    stream << "object " << instance.object_id;
  } else {
    stream << "object";
  }
  if (show_object_id && instance.object_id >= 0) {
    stream << " o" << instance.object_id;
  }
  if (show_track_id && instance.track_id >= 0) {
    stream << " t" << instance.track_id;
  }
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
  object_marker_pub_ = node_.create_publisher<visualization_msgs::msg::MarkerArray>(
      config_.object_markers_topic,
      rclcpp::QoS(1).reliable());
  instance_marker_pub_ = node_.create_publisher<visualization_msgs::msg::MarkerArray>(
      config_.instance_markers_topic,
      rclcpp::QoS(1).reliable());
  map_surface_pub_ = node_.create_publisher<sensor_msgs::msg::PointCloud2>(
      config_.tsdf_output_topic,
      rclcpp::QoS(1).reliable());
  map_stats_pub_ = node_.create_publisher<std_msgs::msg::String>(
      "/roomie/map_stats",
      rclcpp::QoS(1).reliable());
  if (config_.save_scene_graph && !config_.save_dsg_service.empty()) {
    save_dsg_service_ = node_.create_service<std_srvs::srv::Trigger>(
        config_.save_dsg_service,
        [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
               std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
          handleSaveDsg(request, response);
        });
  }
}

void PublisherPersistenceThread::run() {
  const auto period = std::chrono::duration<double>(config_.publish_period_sec);
  while (!stopRequested()) {
    const MapBackendSnapshot map_snapshot = map_thread_.debugSnapshot();
    map_surface_pub_->publish(buildMapSurfaceCloud(map_snapshot));
    const std_msgs::msg::String map_stats = buildMapStats(map_snapshot);
    map_stats_pub_->publish(map_stats);
    object_marker_pub_->publish(buildObjectMarkers());
    instance_marker_pub_->publish(buildTrackedInstanceMarkers());
    const auto now = std::chrono::steady_clock::now();
    if (now - last_log_time_ >=
        std::chrono::duration<double>(config_.file_logging_period_sec)) {
      last_log_time_ = now;
      RunLogger::logGlobal("map", map_stats.data);
      const std::size_t object_count = instance_store_.snapshotInstances().size();
      const std::size_t tracked_instance_count =
          instance_store_.snapshotTrackedInstances().size();
      RunLogger::logGlobal("publisher",
                           "published surface_points=" +
                               std::to_string(map_snapshot.debug_surface_points.size()) +
                               " objects=" + std::to_string(object_count) +
                               " tracked_instances=" +
                               std::to_string(tracked_instance_count));
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
         << " latest_map_version=" << snapshot.latest_map_version
         << " has_map=" << (snapshot.has_map ? "true" : "false")
         << " cache_ready=" << (snapshot.surface_cache_ready ? "true" : "false")
         << " cache_dirty=" << (snapshot.cache_dirty ? "true" : "false")
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
         << " surface_cache_rebuild_period_sec="
         << config_.surface_cache_rebuild_period_sec
         << " publish_period_sec=" << config_.publish_period_sec;
  message.data = stream.str();
  return message;
}

void PublisherPersistenceThread::handleSaveDsg(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) const {
  (void)request;
  const TimeNanoseconds saved_time_ns = node_.now().nanoseconds();
  ObjectGraphSavePaths paths;
  std::string error;
  if (!resolveObjectGraphSavePaths(config_.scene_graph_save_path,
                                   saved_time_ns,
                                   config_.snapshot_image_subdir,
                                   &paths,
                                   &error)) {
    response->success = false;
    response->message = "failed to resolve DSG save path: " + error;
    RCLCPP_WARN(node_.get_logger(), "%s", response->message.c_str());
    RunLogger::logGlobal("persistence", response->message);
    return;
  }

  ObjectGraphSnapshot snapshot;
  if (!instance_store_.prepareSceneGraphForSave(&snapshot,
                                                paths.snapshot_image_dir,
                                                paths.snapshot_uri_prefix,
                                                &error)) {
    response->success = false;
    response->message = "failed to prepare DSG: " + error;
    RCLCPP_WARN(node_.get_logger(), "%s", response->message.c_str());
    RunLogger::logGlobal("persistence", response->message);
    return;
  }

  if (!saveObjectGraphSnapshotJsonAtomic(snapshot,
                                         config_.world_frame,
                                         saved_time_ns,
                                         paths.primary_path,
                                         &error) ||
      (!paths.latest_path.empty() &&
       !saveObjectGraphSnapshotJsonAtomic(snapshot,
                                          config_.world_frame,
                                          saved_time_ns,
                                          paths.latest_path,
                                          &error))) {
    response->success = false;
    response->message = "failed to save DSG: " + error;
    RCLCPP_WARN(node_.get_logger(), "%s", response->message.c_str());
    RunLogger::logGlobal("persistence", response->message);
    return;
  }

  std::ostringstream stream;
  stream << "saved DSG objects=" << snapshot.objects.size()
         << " relations=" << snapshot.relations.size()
         << " snapshot_images=" << snapshot.snapshot_images.size()
         << " path=" << paths.primary_path.string();
  if (!paths.latest_path.empty()) {
    stream << " latest=" << paths.latest_path.string();
  }
  if (!snapshot.snapshot_images.empty() && !paths.snapshot_image_dir.empty()) {
    stream << " snapshots=" << paths.snapshot_image_dir.string();
  }
  response->success = true;
  response->message = stream.str();
  RCLCPP_INFO(node_.get_logger(), "%s", response->message.c_str());
  RunLogger::logGlobal("persistence", response->message);
}

visualization_msgs::msg::MarkerArray PublisherPersistenceThread::buildObjectMarkers() const {
  return buildMarkers(instance_store_.snapshotInstances(),
                      "roomie_objects",
                      "roomie_object_labels",
                      false,
                      false,
                      0.04f,
                      0.95f,
                      0.45f);
}

visualization_msgs::msg::MarkerArray
PublisherPersistenceThread::buildTrackedInstanceMarkers() const {
  return buildMarkers(instance_store_.snapshotTrackedInstances(),
                      "roomie_instance_tracks",
                      "roomie_instance_track_labels",
                      false,
                      true,
                      0.015f,
                      0.35f,
                      0.12f);
}

visualization_msgs::msg::MarkerArray PublisherPersistenceThread::buildMarkers(
    const std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>& records,
    const std::string& box_namespace,
    const std::string& label_namespace,
    bool show_object_id,
    bool show_track_id,
    float line_width,
    float active_alpha,
    float inactive_alpha) const {
  visualization_msgs::msg::MarkerArray markers;

  visualization_msgs::msg::Marker clear;
  clear.header.frame_id = config_.world_frame;
  clear.header.stamp = node_.now();
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  markers.markers.push_back(clear);

  int marker_id = 1;
  for (const InstanceRecord& instance : records) {
    const RgbColor color = colorForLabel(instance.label, instance.semantic_id);
    visualization_msgs::msg::Marker box;
    box.header.frame_id = config_.world_frame;
    box.header.stamp = node_.now();
    box.ns = box_namespace;
    box.id = marker_id++;
    box.type = visualization_msgs::msg::Marker::LINE_LIST;
    box.action = visualization_msgs::msg::Marker::ADD;
    box.pose.orientation.w = 1.0;
    box.scale.x = line_width;
    box.color.r = color.r;
    box.color.g = color.g;
    box.color.b = color.b;
    box.color.a = instance.active ? active_alpha : inactive_alpha;
    fillLineListFromCorners(
        yawObbCorners(instance.center_world, instance.size_m, instance.yaw_rad),
        &box);
    markers.markers.push_back(box);

    visualization_msgs::msg::Marker text;
    text.header.frame_id = config_.world_frame;
    text.header.stamp = box.header.stamp;
    text.ns = label_namespace;
    text.id = marker_id++;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position.x = instance.center_world.x();
    text.pose.position.y = instance.center_world.y();
    text.pose.position.z = instance.center_world.z();
    text.pose.position.z += std::max(0.05f, instance.size_m.z() * 0.6f);
    text.scale.z = line_width < 0.02f ? 0.08 : 0.12;
    text.color.r = 1.0f;
    text.color.g = 1.0f;
    text.color.b = 1.0f;
    text.color.a = instance.active ? std::min(1.0f, active_alpha + 0.25f)
                                   : std::min(0.75f, inactive_alpha + 0.20f);
    text.text = instanceText(
        instance, config_.show_3d_label_score, show_object_id, show_track_id);
    markers.markers.push_back(text);
  }

  return markers;
}

}  // namespace roomie
