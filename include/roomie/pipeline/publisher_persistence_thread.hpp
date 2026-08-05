#pragma once

#include <chrono>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/dsg/object_graph_io.hpp"
#include "roomie/pipeline/map_backend.hpp"
#include "roomie/pipeline/map_thread.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
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
  void onStopRequested() override;

 private:
  struct SaveDsgJob {
    std::uint64_t job_id = 0;
    TimeNanoseconds saved_time_ns = 0;
    ObjectGraphSavePaths paths;
    ObjectGraphSnapshot snapshot;
  };

  visualization_msgs::msg::MarkerArray buildObjectMarkers() const;
  visualization_msgs::msg::MarkerArray buildTrackedInstanceMarkers() const;
  visualization_msgs::msg::MarkerArray buildMarkers(
      const std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>& records,
      const std::string& box_namespace,
      const std::string& label_namespace,
      bool show_object_id,
      bool show_track_id,
      float line_width,
      float active_alpha,
      float inactive_alpha) const;
  sensor_msgs::msg::PointCloud2 buildMapSurfaceCloud(
      const MapBackendSnapshot& snapshot) const;
  std_msgs::msg::String buildMapStats(const MapBackendSnapshot& snapshot) const;
  void handleSaveDsg(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void persistSaveDsgJob(SaveDsgJob job) const;

  rclcpp::Node& node_;
  const InstanceStore& instance_store_;
  const MapThread& map_thread_;
  PipelineConfig config_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr object_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr instance_marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_surface_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr map_stats_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_dsg_service_;
  ThreadSafeQueue<SaveDsgJob> save_dsg_jobs_;
  std::atomic<std::uint64_t> next_save_job_id_{1};
  std::chrono::steady_clock::time_point last_log_time_ =
      std::chrono::steady_clock::now();
};

}  // namespace roomie
