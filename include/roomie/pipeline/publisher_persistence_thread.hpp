#pragma once

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

class PublisherPersistenceThread : public WorkerThread {
 public:
  PublisherPersistenceThread(rclcpp::Node& node,
                             const InstanceStore& instance_store,
                             PipelineConfig config);

 protected:
  void run() override;

 private:
  visualization_msgs::msg::MarkerArray buildInstanceMarkers() const;

  rclcpp::Node& node_;
  const InstanceStore& instance_store_;
  PipelineConfig config_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

}  // namespace roomie
