#pragma once

#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/pipeline/map_backend.hpp"
#include "roomie/pipeline/map_thread.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

class PublisherPersistenceThread : public WorkerThread {
 public:
  PublisherPersistenceThread(rclcpp::Node& node,
                             const InstanceStore& instance_store,
                             const MapThread& map_thread,
                             PipelineConfig config);

 protected:
  void run() override;

 private:
  visualization_msgs::msg::MarkerArray buildInstanceMarkers() const;
  sensor_msgs::msg::PointCloud2 buildMapSurfaceCloud(
      const MapBackendSnapshot& snapshot) const;
  std_msgs::msg::String buildMapStats(const MapBackendSnapshot& snapshot) const;

  rclcpp::Node& node_;
  const InstanceStore& instance_store_;
  const MapThread& map_thread_;
  PipelineConfig config_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_surface_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr map_stats_pub_;
  std::chrono::steady_clock::time_point last_log_time_ =
      std::chrono::steady_clock::now();
};

}  // namespace roomie
