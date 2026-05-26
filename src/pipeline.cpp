#include "roomie/pipeline/pipeline.hpp"

namespace roomie {

RoomiePipeline::RoomiePipeline(rclcpp::Node& node, PipelineConfig config)
    : config_(std::move(config)),
      mapping_queue_(config_.mapping_queue_size),
      detection_queue_(config_.detection_queue_size),
      inference_response_queue_(config_.inference_response_queue_size),
      ros_io_thread_(mapping_queue_, detection_queue_),
      map_thread_(mapping_queue_, config_),
      python_backend_(config_),
      instance_map_thread_(inference_response_queue_, map_thread_, config_),
      detection_bridge_thread_(detection_queue_,
                               inference_response_queue_,
                               map_thread_,
                               python_backend_,
                               config_),
      publisher_persistence_thread_(node, instance_map_thread_, config_) {}

RoomiePipeline::~RoomiePipeline() { stop(); }

void RoomiePipeline::start() {
  if (started_) {
    return;
  }
  python_backend_.start();
  map_thread_.start();
  instance_map_thread_.start();
  detection_bridge_thread_.start();
  publisher_persistence_thread_.start();
  ros_io_thread_.start();
  started_ = true;
}

void RoomiePipeline::stop() {
  if (!started_) {
    return;
  }

  ros_io_thread_.stop();
  detection_queue_.stop();
  mapping_queue_.stop();
  inference_response_queue_.stop();
  detection_bridge_thread_.stop();
  publisher_persistence_thread_.stop();
  instance_map_thread_.stop();
  map_thread_.stop();
  python_backend_.stop();
  started_ = false;
}

}  // namespace roomie
