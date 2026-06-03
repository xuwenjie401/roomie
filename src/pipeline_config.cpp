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
  config.detection_debug_image_topic = node.declare_parameter<std::string>(
      "topics.detection_debug_image_topic", config.detection_debug_image_topic);
  config.raw_detections_topic = node.declare_parameter<std::string>(
      "topics.raw_detections_topic", config.raw_detections_topic);
  config.object_markers_topic = node.declare_parameter<std::string>(
      "topics.object_markers_topic", config.object_markers_topic);
  config.instance_markers_topic = node.declare_parameter<std::string>(
      "topics.instance_markers_topic", config.instance_markers_topic);

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
  config.surface_cache_rebuild_period_sec = std::max(
      0.0,
      node.declare_parameter<double>("tsdf.surface_cache_rebuild_period_sec",
                                     config.surface_cache_rebuild_period_sec));
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
  config.min_patch_coverage_ratio = static_cast<float>(std::clamp(
      node.declare_parameter<double>("detection.min_patch_coverage_ratio",
                                     config.min_patch_coverage_ratio),
      0.0,
      1.0));
  config.patch_depth_max_m = static_cast<float>(std::max(
      0.0,
      node.declare_parameter<double>("detection.patch_depth_max_m",
                                     config.patch_depth_max_m)));
  config.patch_depth_zbuffer_scale = positiveIntOrDefault(
      node.declare_parameter<int>("detection.patch_depth_zbuffer_scale",
                                  config.patch_depth_zbuffer_scale),
      config.patch_depth_zbuffer_scale);
  config.patch_depth_zbuffer_splat_radius_cells = std::max(
      0,
      static_cast<int>(
          node.declare_parameter<int>("detection.patch_depth_zbuffer_splat_radius_cells",
                                      config.patch_depth_zbuffer_splat_radius_cells)));
  config.patch_depth_zbuffer_front_quantile = static_cast<float>(std::clamp(
      node.declare_parameter<double>("detection.patch_depth_zbuffer_front_quantile",
                                     config.patch_depth_zbuffer_front_quantile),
      0.0,
      1.0));
  config.patch_depth_zbuffer_min_cells_per_patch = positiveIntOrDefault(
      node.declare_parameter<int>("detection.patch_depth_zbuffer_min_cells_per_patch",
                                  config.patch_depth_zbuffer_min_cells_per_patch),
      config.patch_depth_zbuffer_min_cells_per_patch);
  config.max_inference_fps = std::max(
      0.0,
      node.declare_parameter<double>("detection.max_inference_fps",
                                     config.max_inference_fps));
  config.python_backend_enabled = node.declare_parameter<bool>(
      "detection.python_backend_enabled", config.python_backend_enabled);
  config.python_executable = node.declare_parameter<std::string>(
      "detection.python_executable", config.python_executable);
  config.python_worker_script = node.declare_parameter<std::string>(
      "detection.python_worker_script", config.python_worker_script);
  config.boxer_repo_path = node.declare_parameter<std::string>(
      "detection.boxer_repo_path", config.boxer_repo_path);
  config.boxernet_ckpt_path = node.declare_parameter<std::string>(
      "detection.boxernet_ckpt_path", config.boxernet_ckpt_path);
  config.inference_device = node.declare_parameter<std::string>(
      "detection.inference_device", config.inference_device);
  config.inference_precision = node.declare_parameter<std::string>(
      "detection.inference_precision", config.inference_precision);
  config.text_prompts = node.declare_parameter<std::vector<std::string>>(
      "detection.text_prompts", config.text_prompts);
  config.owl_min_confidence = static_cast<float>(std::clamp(
      node.declare_parameter<double>("detection.owl_min_confidence",
                                     config.owl_min_confidence),
      0.0,
      1.0));
  config.owl_nms_iou_threshold = static_cast<float>(std::clamp(
      node.declare_parameter<double>("detection.owl_nms_iou_threshold",
                                     config.owl_nms_iou_threshold),
      0.0,
      1.0));
  config.boxernet_min_confidence = static_cast<float>(std::clamp(
      node.declare_parameter<double>("detection.boxernet_min_confidence",
                                     config.boxernet_min_confidence),
      0.0,
      1.0));
  config.robot_bbox_mask_overlap = static_cast<float>(std::clamp(
      node.declare_parameter<double>("detection.robot_bbox_mask_overlap",
                                     config.robot_bbox_mask_overlap),
      0.0,
      1.0));
  config.robot_bbox_center_overlap = static_cast<float>(std::clamp(
      node.declare_parameter<double>("detection.robot_bbox_center_overlap",
                                     config.robot_bbox_center_overlap),
      0.0,
      1.0));
  config.robot_mask_dilate_px = std::max(
      0,
      static_cast<int>(node.declare_parameter<int>("detection.robot_mask_dilate_px",
                                                   config.robot_mask_dilate_px)));
  config.show_3d_label_score = node.declare_parameter<bool>(
      "detection.show_3d_label_score", config.show_3d_label_score);

  config.instance_min_confidence = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.min_confidence",
                                     config.instance_min_confidence),
      0.0,
      1.0));
  config.instance_object_min_confidence = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.object_min_confidence",
                                     config.instance_object_min_confidence),
      0.0,
      1.0));
  config.instance_min_bbox_size_m = positiveFloatOrDefault(
      node.declare_parameter<double>("instance.min_bbox_size_m",
                                     config.instance_min_bbox_size_m),
      config.instance_min_bbox_size_m);
  config.instance_max_bbox_size_m = positiveFloatOrDefault(
      node.declare_parameter<double>("instance.max_bbox_size_m",
                                     config.instance_max_bbox_size_m),
      config.instance_max_bbox_size_m);
  if (config.instance_max_bbox_size_m < config.instance_min_bbox_size_m) {
    std::swap(config.instance_max_bbox_size_m, config.instance_min_bbox_size_m);
  }
  config.instance_match_iou_threshold = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.match_iou_threshold",
                                     config.instance_match_iou_threshold),
      0.0,
      1.0));
  config.instance_match_center_distance_m = static_cast<float>(std::max(
      0.0,
      node.declare_parameter<double>("instance.match_center_distance_m",
                                     config.instance_match_center_distance_m)));
  config.instance_min_support_count = positiveIntOrDefault(
      node.declare_parameter<int>("instance.min_support_count",
                                  config.instance_min_support_count),
      config.instance_min_support_count);
  config.instance_min_confidence_mass = static_cast<float>(std::max(
      0.0,
      node.declare_parameter<double>("instance.min_confidence_mass",
                                     config.instance_min_confidence_mass)));
  config.instance_tentative_max_missed = positiveIntOrDefault(
      node.declare_parameter<int>("instance.tentative_max_missed",
                                  config.instance_tentative_max_missed),
      config.instance_tentative_max_missed);
  config.instance_inactive_after_missed = positiveIntOrDefault(
      node.declare_parameter<int>("instance.inactive_after_missed",
                                  config.instance_inactive_after_missed),
      config.instance_inactive_after_missed);
  config.instance_duplicate_iou_threshold = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.duplicate_iou_threshold",
                                     config.instance_duplicate_iou_threshold),
      0.0,
      1.0));
  config.instance_duplicate_size_ratio_min = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.duplicate_size_ratio_min",
                                     config.instance_duplicate_size_ratio_min),
      0.0,
      1.0));
  config.instance_confirmed_duplicate_iou_threshold = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.confirmed_duplicate_iou_threshold",
                                     config.instance_confirmed_duplicate_iou_threshold),
      0.0,
      1.0));
  config.instance_duplicate_containment_threshold = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.duplicate_containment_threshold",
                                     config.instance_duplicate_containment_threshold),
      0.0,
      1.0));
  config.instance_duplicate_center_distance_m = static_cast<float>(std::max(
      0.0,
      node.declare_parameter<double>("instance.duplicate_center_distance_m",
                                     config.instance_duplicate_center_distance_m)));
  config.instance_duplicate_small_object_volume_ratio = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.duplicate_small_object_volume_ratio",
                                     config.instance_duplicate_small_object_volume_ratio),
      0.0,
      1.0));
  config.instance_small_duplicate_max_volume_m3 = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>(
                   "instance.small_duplicate_max_volume_m3",
                   config.instance_small_duplicate_max_volume_m3)));
  config.instance_small_duplicate_max_extent_m = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>(
                   "instance.small_duplicate_max_extent_m",
                   config.instance_small_duplicate_max_extent_m)));
  config.instance_small_duplicate_iou_threshold = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.small_duplicate_iou_threshold",
                                     config.instance_small_duplicate_iou_threshold),
      0.0,
      1.0));
  config.instance_small_duplicate_center_ratio = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>(
                   "instance.small_duplicate_center_ratio",
                   config.instance_small_duplicate_center_ratio)));
  config.instance_small_duplicate_size_ratio_min = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.small_duplicate_size_ratio_min",
                                     config.instance_small_duplicate_size_ratio_min),
      0.0,
      1.0));
  config.instance_quality_observation_min_quality = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.quality_observation_min_quality",
                                     config.instance_quality_observation_min_quality),
      0.0,
      1.0));
  config.instance_close_observation_distance_m = static_cast<float>(std::max(
      0.0,
      node.declare_parameter<double>("instance.close_observation_distance_m",
                                     config.instance_close_observation_distance_m)));
  config.instance_far_observation_distance_m = static_cast<float>(std::max(
      0.0,
      node.declare_parameter<double>("instance.far_observation_distance_m",
                                     config.instance_far_observation_distance_m)));
  if (config.instance_far_observation_distance_m <
      config.instance_close_observation_distance_m) {
    std::swap(config.instance_far_observation_distance_m,
              config.instance_close_observation_distance_m);
  }
  config.instance_far_promotion_weight = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.far_promotion_weight",
                                     config.instance_far_promotion_weight),
      0.0,
      1.0));
  config.instance_fusion_prior_mass_cap = positiveFloatOrDefault(
      node.declare_parameter<double>("instance.fusion_prior_mass_cap",
                                     config.instance_fusion_prior_mass_cap),
      config.instance_fusion_prior_mass_cap);
  config.instance_high_quality_min_count = positiveIntOrDefault(
      node.declare_parameter<int>("instance.high_quality_min_count",
                                  config.instance_high_quality_min_count),
      config.instance_high_quality_min_count);
  config.instance_high_quality_min_mass = static_cast<float>(std::max(
      0.0,
      node.declare_parameter<double>("instance.high_quality_min_mass",
                                     config.instance_high_quality_min_mass)));
  config.instance_geometry_check_period_sec = std::max(
      0.0,
      node.declare_parameter<double>("instance.geometry_check_period_sec",
                                     config.instance_geometry_check_period_sec));
  config.instance_geometry_recent_window_sec = std::max(
      0.0,
      node.declare_parameter<double>("instance.geometry_recent_window_sec",
                                     config.instance_geometry_recent_window_sec));
  config.instance_geometry_shell_thickness_m = positiveFloatOrDefault(
      node.declare_parameter<double>("instance.geometry_shell_thickness_m",
                                     config.instance_geometry_shell_thickness_m),
      config.instance_geometry_shell_thickness_m);
  config.instance_geometry_empty_inside_points = positiveIntOrDefault(
      node.declare_parameter<int>("instance.geometry_empty_inside_points",
                                  config.instance_geometry_empty_inside_points),
      config.instance_geometry_empty_inside_points);
  config.instance_geometry_min_unique_voxels = positiveIntOrDefault(
      node.declare_parameter<int>("instance.geometry_min_unique_voxels",
                                  config.instance_geometry_min_unique_voxels),
      config.instance_geometry_min_unique_voxels);
  config.instance_geometry_confirm_score = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.geometry_confirm_score",
                                     config.instance_geometry_confirm_score),
      0.0,
      1.0));
  config.instance_geometry_suppress_score = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.geometry_suppress_score",
                                     config.instance_geometry_suppress_score),
      0.0,
      1.0));
  config.instance_geometry_recover_score = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.geometry_recover_score",
                                     config.instance_geometry_recover_score),
      0.0,
      1.0));
  if (config.instance_geometry_recover_score < config.instance_geometry_suppress_score) {
    std::swap(config.instance_geometry_recover_score,
              config.instance_geometry_suppress_score);
  }
  config.instance_geometry_failures_before_suppress = positiveIntOrDefault(
      node.declare_parameter<int>("instance.geometry_failures_before_suppress",
                                  config.instance_geometry_failures_before_suppress),
      config.instance_geometry_failures_before_suppress);
  config.instance_geometry_inactive_delete_bad_count = positiveIntOrDefault(
      node.declare_parameter<int>("instance.geometry_inactive_delete_bad_count",
                                  config.instance_geometry_inactive_delete_bad_count),
      config.instance_geometry_inactive_delete_bad_count);
  config.instance_geometry_empty_small_object_max_volume_m3 = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>(
                   "instance.geometry_empty_small_object_max_volume_m3",
                   config.instance_geometry_empty_small_object_max_volume_m3)));
  config.instance_geometry_empty_small_object_max_extent_m = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>(
                   "instance.geometry_empty_small_object_max_extent_m",
                   config.instance_geometry_empty_small_object_max_extent_m)));
  config.instance_geometry_reevaluate_center_delta_m = positiveFloatOrDefault(
      node.declare_parameter<double>("instance.geometry_reevaluate_center_delta_m",
                                     config.instance_geometry_reevaluate_center_delta_m),
      config.instance_geometry_reevaluate_center_delta_m);
  config.instance_geometry_reevaluate_size_ratio = static_cast<float>(std::clamp(
      node.declare_parameter<double>("instance.geometry_reevaluate_size_ratio",
                                     config.instance_geometry_reevaluate_size_ratio),
      0.0,
      1.0));
  config.instance_geometry_reevaluate_yaw_delta_deg = positiveFloatOrDefault(
      node.declare_parameter<double>("instance.geometry_reevaluate_yaw_delta_deg",
                                     config.instance_geometry_reevaluate_yaw_delta_deg),
      config.instance_geometry_reevaluate_yaw_delta_deg);
  config.instance_confirmed_geometry_edge_freeze_weight = static_cast<float>(std::clamp(
      node.declare_parameter<double>(
          "instance.confirmed_geometry_edge_freeze_weight",
          config.instance_confirmed_geometry_edge_freeze_weight),
      0.0,
      1.0));
  config.instance_confirmed_geometry_large_min_volume_m3 = static_cast<float>(std::max(
      0.0,
      node.declare_parameter<double>(
          "instance.confirmed_geometry_large_min_volume_m3",
          config.instance_confirmed_geometry_large_min_volume_m3)));
  config.instance_confirmed_geometry_far_center_shift_ratio = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>(
                   "instance.confirmed_geometry_far_center_shift_ratio",
                   config.instance_confirmed_geometry_far_center_shift_ratio)));
  config.instance_confirmed_geometry_far_center_shift_min_m = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>(
                   "instance.confirmed_geometry_far_center_shift_min_m",
                   config.instance_confirmed_geometry_far_center_shift_min_m)));
  config.instance_confirmed_geometry_center_shift_min_extent_m = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>(
                   "instance.confirmed_geometry_center_shift_min_extent_m",
                   config.instance_confirmed_geometry_center_shift_min_extent_m)));
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

  config.load_scene_graph = node.declare_parameter<bool>(
      "persistence.load_scene_graph", config.load_scene_graph);
  config.scene_graph_load_path = node.declare_parameter<std::string>(
      "persistence.scene_graph_load_path", config.scene_graph_load_path);
  config.freeze_instances = node.declare_parameter<bool>(
      "persistence.freeze_instances", config.freeze_instances);
  config.save_scene_graph = node.declare_parameter<bool>(
      "persistence.save_scene_graph", config.save_scene_graph);
  config.scene_graph_save_path = node.declare_parameter<std::string>(
      "persistence.scene_graph_save_path", config.scene_graph_save_path);
  config.save_dsg_service = node.declare_parameter<std::string>(
      "persistence.save_dsg_service", config.save_dsg_service);
  config.snapshot_remake_enabled = node.declare_parameter<bool>(
      "persistence.snapshot_remake_enabled", config.snapshot_remake_enabled);
  config.snapshot_staging_dir = node.declare_parameter<std::string>(
      "persistence.snapshot_staging_dir", config.snapshot_staging_dir);
  config.snapshot_image_subdir = node.declare_parameter<std::string>(
      "persistence.snapshot_image_subdir", config.snapshot_image_subdir);
  config.instance_snapshot_first_min_quality = static_cast<float>(std::clamp(
      node.declare_parameter<double>("persistence.snapshot_first_min_quality",
                                     config.instance_snapshot_first_min_quality),
      0.0,
      1.0));
  config.instance_snapshot_min_quality = static_cast<float>(std::clamp(
      node.declare_parameter<double>("persistence.snapshot_min_quality",
                                     config.instance_snapshot_min_quality),
      0.0,
      1.0));
  config.instance_snapshot_min_box_area_px = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>("persistence.snapshot_min_box_area_px",
                                              config.instance_snapshot_min_box_area_px)));
  config.instance_snapshot_position_weight = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>("persistence.snapshot_position_weight",
                                              config.instance_snapshot_position_weight)));
  config.instance_snapshot_size_weight = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>("persistence.snapshot_size_weight",
                                              config.instance_snapshot_size_weight)));
  config.instance_snapshot_replace_min_quality_delta = static_cast<float>(
      std::max(0.0,
               node.declare_parameter<double>(
                   "persistence.snapshot_replace_min_quality_delta",
                   config.instance_snapshot_replace_min_quality_delta)));
  config.instance_snapshot_replace_min_quality_ratio = static_cast<float>(
      std::max(1.0,
               node.declare_parameter<double>(
                   "persistence.snapshot_replace_min_quality_ratio",
                   config.instance_snapshot_replace_min_quality_ratio)));

  config.file_logging_enabled = node.declare_parameter<bool>(
      "logging.enabled", config.file_logging_enabled);
  config.file_logging_root_dir = node.declare_parameter<std::string>(
      "logging.root_dir", config.file_logging_root_dir);
  config.file_logging_period_sec = std::max(
      0.25,
      node.declare_parameter<double>("logging.period_sec",
                                     config.file_logging_period_sec));

  return config;
}

}  // namespace roomie
