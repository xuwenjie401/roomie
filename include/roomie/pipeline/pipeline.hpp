#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "roomie/pipeline/detection_bridge_thread.hpp"
#include "roomie/pipeline/instance_map_thread.hpp"
#include "roomie/pipeline/map_thread.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/publisher_persistence_thread.hpp"
#include "roomie/pipeline/python_inference_backend.hpp"
#include "roomie/pipeline/ros_io_thread.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"

namespace roomie {

class RoomiePipeline {
 public:
  RoomiePipeline(rclcpp::Node& node, PipelineConfig config);
  ~RoomiePipeline();

  void start();
  void stop();

 private:
  PipelineConfig config_;
  ThreadSafeQueue<MappingFrame> mapping_queue_;
  ThreadSafeQueue<DetectionFrame> detection_queue_;
  ThreadSafeQueue<InferenceResponse> inference_response_queue_;

  RosIoThread ros_io_thread_;
  MapThread map_thread_;
  PythonInferenceBackend python_backend_;
  InstanceMapThread instance_map_thread_;
  DetectionBridgeThread detection_bridge_thread_;
  PublisherPersistenceThread publisher_persistence_thread_;

  bool started_ = false;
};

}  // namespace roomie
