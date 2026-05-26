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

float positiveFloatOrDefault(double value, float fallback) {
  return value > 0.0 ? static_cast<float>(value) : fallback;
}

double positiveDoubleOrDefault(double value, double fallback) {
  return value > 0.0 ? value : fallback;
}

}  // namespace

PipelineConfig PipelineConfig::declareAndLoad(rclcpp::Node& node) {
  PipelineConfig config;

  config.world_frame =
      node.declare_parameter<std::string>("basic.world_frame", config.world_frame);
  config.mapping_camera_id =
      node.declare_parameter<std::string>("basic.mapping_camera_id", config.mapping_camera_id);
  config.mapping_camera_frame = node.declare_parameter<std::string>(
      "basic.mapping_camera_frame", config.mapping_camera_frame);

  config.rgb_topic =
      node.declare_parameter<std::string>("topics.color_topic", config.rgb_topic);
  config.depth_topic =
      node.declare_parameter<std::string>("topics.depth_topic", config.depth_topic);
  config.mask_topic =
      node.declare_parameter<std::string>("topics.mask_topic", config.mask_topic);
  config.camera_info_topic = node.declare_parameter<std::string>(
      "topics.camera_info_topic", config.camera_info_topic);
  config.tf_topic =
      node.declare_parameter<std::string>("topics.tf_topic", config.tf_topic);
  config.tf_static_topic =
      node.declare_parameter<std::string>("topics.tf_static_topic", config.tf_static_topic);
  config.tsdf_output_topic = node.declare_parameter<std::string>(
      "topics.tsdf_output_topic", config.tsdf_output_topic);

  config.camera_width = positiveIntOrDefault(
      node.declare_parameter<int>("camera.width", config.camera_width),
      config.camera_width);
  config.camera_height = positiveIntOrDefault(
      node.declare_parameter<int>("camera.height", config.camera_height),
      config.camera_height);
  config.camera_fx = positiveFloatOrDefault(
      node.declare_parameter<double>("camera.fx", config.camera_fx),
      config.camera_fx);
  config.camera_fy = positiveFloatOrDefault(
      node.declare_parameter<double>("camera.fy", config.camera_fy),
      config.camera_fy);
  config.camera_cx =
      static_cast<float>(node.declare_parameter<double>("camera.cx", config.camera_cx));
  config.camera_cy =
      static_cast<float>(node.declare_parameter<double>("camera.cy", config.camera_cy));

  config.depth_min_m = positiveFloatOrDefault(
      node.declare_parameter<double>("input_filter.depth_min_m", config.depth_min_m),
      config.depth_min_m);
  config.depth_max_m = positiveFloatOrDefault(
      node.declare_parameter<double>("input_filter.depth_max_m", config.depth_max_m),
      config.depth_max_m);
  if (config.depth_max_m < config.depth_min_m) {
    std::swap(config.depth_min_m, config.depth_max_m);
  }
  config.depth_scale = positiveFloatOrDefault(
      node.declare_parameter<double>("input_filter.depth_scale", config.depth_scale),
      config.depth_scale);
  config.mask_robot_threshold = node.declare_parameter<int>(
      "input_filter.mask_robot_threshold", config.mask_robot_threshold);

  config.map_backend =
      node.declare_parameter<std::string>("tsdf.map_backend", config.map_backend);
  config.voxel_size_m = positiveFloatOrDefault(
      node.declare_parameter<double>("tsdf.voxel_size_m", config.voxel_size_m),
      config.voxel_size_m);
  config.truncation_distance_vox = positiveFloatOrDefault(
      node.declare_parameter<double>("tsdf.truncation_distance_vox",
                                     config.truncation_distance_vox),
      config.truncation_distance_vox);
  config.max_weight = positiveFloatOrDefault(
      node.declare_parameter<double>("tsdf.max_weight", config.max_weight),
      config.max_weight);
  config.max_integration_distance_m = positiveFloatOrDefault(
      node.declare_parameter<double>("tsdf.max_integration_distance_m",
                                     config.max_integration_distance_m),
      config.max_integration_distance_m);
  config.min_visualization_weight = positiveFloatOrDefault(
      node.declare_parameter<double>("tsdf.min_visualization_weight",
                                     config.min_visualization_weight),
      config.min_visualization_weight);
  config.min_color_weight = positiveFloatOrDefault(
      node.declare_parameter<double>("tsdf.min_color_weight", config.min_color_weight),
      config.min_color_weight);
  config.surface_visualization_distance_vox = positiveFloatOrDefault(
      node.declare_parameter<double>("tsdf.surface_visualization_distance_vox",
                                     config.surface_visualization_distance_vox),
      config.surface_visualization_distance_vox);
  config.load_map =
      node.declare_parameter<bool>("tsdf.load_map", config.load_map);
  config.map_load_path =
      node.declare_parameter<std::string>("tsdf.map_load_path", config.map_load_path);
  config.freeze_tsdf_map =
      node.declare_parameter<bool>("tsdf.freeze_tsdf_map", config.freeze_tsdf_map);
  config.save_map =
      node.declare_parameter<bool>("tsdf.save_map", config.save_map);
  config.map_save_path =
      node.declare_parameter<std::string>("tsdf.map_save_path", config.map_save_path);

  config.boxer_input_size = positiveIntOrDefault(
      node.declare_parameter<int>("detection.boxer_input_size", config.boxer_input_size),
      config.boxer_input_size);
  config.patch_rows = positiveIntOrDefault(
      node.declare_parameter<int>("detection.patch_rows", config.patch_rows),
      config.patch_rows);
  config.patch_cols = positiveIntOrDefault(
      node.declare_parameter<int>("detection.patch_cols", config.patch_cols),
      config.patch_cols);
  config.min_patch_coverage_ratio = static_cast<float>(std::clamp(
      node.declare_parameter<double>("detection.min_patch_coverage_ratio",
                                     config.min_patch_coverage_ratio),
      0.0,
      1.0));
  config.max_inference_fps = std::max(
      0.0,
      node.declare_parameter<double>("detection.max_inference_fps",
                                     config.max_inference_fps));

  config.input_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("queues.input_queue_size",
                                  static_cast<int>(config.input_queue_size)),
      config.input_queue_size);
  config.output_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("queues.output_queue_size",
                                  static_cast<int>(config.output_queue_size)),
      config.output_queue_size);
  config.sync_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("queues.sync_queue_size",
                                  static_cast<int>(config.sync_queue_size)),
      config.sync_queue_size);
  config.pending_frame_limit = positiveSizeOrDefault(
      node.declare_parameter<int>("queues.pending_frame_limit",
                                  static_cast<int>(config.pending_frame_limit)),
      config.pending_frame_limit);
  config.mapping_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("queues.mapping_queue_size",
                                  static_cast<int>(config.input_queue_size)),
      config.input_queue_size);
  config.detection_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("queues.detection_queue_size",
                                  static_cast<int>(config.detection_queue_size)),
      config.detection_queue_size);
  config.inference_request_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("queues.inference_request_queue_size",
                                  static_cast<int>(config.inference_request_queue_size)),
      config.inference_request_queue_size);
  config.inference_response_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("queues.inference_response_queue_size",
                                  static_cast<int>(config.inference_response_queue_size)),
      config.inference_response_queue_size);

  config.max_image_stamp_delta_sec = positiveDoubleOrDefault(
      node.declare_parameter<double>("sync.max_image_stamp_delta_sec",
                                     config.max_image_stamp_delta_sec),
      config.max_image_stamp_delta_sec);
  config.tf_buffer_duration_sec = positiveDoubleOrDefault(
      node.declare_parameter<double>("sync.tf_buffer_duration_sec",
                                     config.tf_buffer_duration_sec),
      config.tf_buffer_duration_sec);
  config.max_tf_gap_sec = positiveDoubleOrDefault(
      node.declare_parameter<double>("sync.max_tf_gap_sec", config.max_tf_gap_sec),
      config.max_tf_gap_sec);
  config.max_tf_translation_step_m = positiveDoubleOrDefault(
      node.declare_parameter<double>("sync.max_tf_translation_step_m",
                                     config.max_tf_translation_step_m),
      config.max_tf_translation_step_m);
  config.max_tf_rotation_step_deg = positiveDoubleOrDefault(
      node.declare_parameter<double>("sync.max_tf_rotation_step_deg",
                                     config.max_tf_rotation_step_deg),
      config.max_tf_rotation_step_deg);

  config.publish_period_sec = std::max(
      0.05,
      node.declare_parameter<double>("publishing.publish_period_sec",
                                     config.publish_period_sec));

  config.save_instance_map = node.declare_parameter<bool>(
      "persistence.save_instance_map", config.save_instance_map);
  config.instance_map_save_path = node.declare_parameter<std::string>(
      "persistence.instance_map_save_path", config.instance_map_save_path);

  return config;
}

}  // namespace roomie
