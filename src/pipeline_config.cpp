#include "roomie/pipeline/pipeline_config.hpp"

#include <algorithm>

namespace roomie {

namespace {

std::size_t positiveSizeOrDefault(int value, std::size_t fallback) {
  return value > 0 ? static_cast<std::size_t>(value) : fallback;
}

int positiveIntOrDefault(long value, int fallback) {
  return value > 0 ? static_cast<int>(value) : fallback;
}

}  // namespace

PipelineConfig PipelineConfig::declareAndLoad(rclcpp::Node& node) {
  PipelineConfig config;

  config.world_frame = node.declare_parameter<std::string>("world_frame", config.world_frame);
  config.mapping_camera_id =
      node.declare_parameter<std::string>("mapping_camera_id", config.mapping_camera_id);

  config.boxer_input_size = positiveIntOrDefault(
      node.declare_parameter<int>("boxer_input_size", config.boxer_input_size),
      config.boxer_input_size);
  config.patch_rows = positiveIntOrDefault(
      node.declare_parameter<int>("patch_rows", config.patch_rows),
      config.patch_rows);
  config.patch_cols = positiveIntOrDefault(
      node.declare_parameter<int>("patch_cols", config.patch_cols),
      config.patch_cols);
  config.min_patch_coverage_ratio = static_cast<float>(std::clamp(
      node.declare_parameter<double>("min_patch_coverage_ratio",
                                     config.min_patch_coverage_ratio),
      0.0,
      1.0));
  config.max_inference_fps =
      std::max(0.0, node.declare_parameter<double>("max_inference_fps",
                                                   config.max_inference_fps));

  config.mapping_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("mapping_queue_size",
                                  static_cast<int>(config.mapping_queue_size)),
      config.mapping_queue_size);
  config.detection_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("detection_queue_size",
                                  static_cast<int>(config.detection_queue_size)),
      config.detection_queue_size);
  config.inference_request_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("inference_request_queue_size",
                                  static_cast<int>(config.inference_request_queue_size)),
      config.inference_request_queue_size);
  config.inference_response_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("inference_response_queue_size",
                                  static_cast<int>(config.inference_response_queue_size)),
      config.inference_response_queue_size);

  config.publish_period_sec =
      std::max(0.05, node.declare_parameter<double>("publish_period_sec",
                                                    config.publish_period_sec));

  config.load_map = node.declare_parameter<bool>("load_map", config.load_map);
  config.map_load_path =
      node.declare_parameter<std::string>("map_load_path", config.map_load_path);
  config.save_map = node.declare_parameter<bool>("save_map", config.save_map);
  config.map_save_path =
      node.declare_parameter<std::string>("map_save_path", config.map_save_path);
  config.save_instance_map =
      node.declare_parameter<bool>("save_instance_map", config.save_instance_map);
  config.instance_map_save_path = node.declare_parameter<std::string>(
      "instance_map_save_path",
      config.instance_map_save_path);

  return config;
}

}  // namespace roomie
