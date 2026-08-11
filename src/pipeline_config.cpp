#include "roomie/pipeline/pipeline_config.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include <ament_index_cpp/get_package_share_directory.hpp>

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
  config.camera_info_topic = node.declare_parameter<std::string>(
      "topics.camera_info_topic", config.camera_info_topic);
  config.odom_topic =
      node.declare_parameter<std::string>("topics.odom_topic", config.odom_topic);
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

  config.robot_mask_robot_config = node.declare_parameter<std::string>(
      "robot_mask.robot_config", config.robot_mask_robot_config);
  config.robot_mask_camera_config = node.declare_parameter<std::string>(
      "robot_mask.camera_config", config.robot_mask_camera_config);
  config.robot_mask_reuse_translation_epsilon_m =
      node.declare_parameter<double>(
          "robot_mask.reuse_translation_epsilon_m",
          config.robot_mask_reuse_translation_epsilon_m);
  config.robot_mask_reuse_rotation_epsilon_rad =
      node.declare_parameter<double>(
          "robot_mask.reuse_rotation_epsilon_rad",
          config.robot_mask_reuse_rotation_epsilon_rad);
  if (config.robot_mask_robot_config.empty() ||
      config.robot_mask_camera_config.empty()) {
    throw std::invalid_argument(
        "robot_mask robot_config and camera_config must not be empty");
  }
  if (!std::isfinite(config.robot_mask_reuse_translation_epsilon_m) ||
      !std::isfinite(config.robot_mask_reuse_rotation_epsilon_rad) ||
      config.robot_mask_reuse_translation_epsilon_m < 0.0 ||
      config.robot_mask_reuse_rotation_epsilon_rad < 0.0) {
    throw std::invalid_argument(
        "robot_mask reuse thresholds must be finite and non-negative");
  }

  config.robot_state_enabled = node.declare_parameter<bool>(
      "robot_state.enabled", config.robot_state_enabled);
  config.robot_state_odom_history_sec = node.declare_parameter<double>(
      "robot_state.odom_history_sec", config.robot_state_odom_history_sec);
  config.robot_state_odom_stale_timeout_sec = node.declare_parameter<double>(
      "robot_state.odom_stale_timeout_sec",
      config.robot_state_odom_stale_timeout_sec);
  config.robot_state_rotation_window_sec = node.declare_parameter<double>(
      "robot_state.rotation_window_sec",
      config.robot_state_rotation_window_sec);
  config.robot_state_rotation_enter_rad_s = node.declare_parameter<double>(
      "robot_state.rotation_enter_rad_s",
      config.robot_state_rotation_enter_rad_s);
  config.robot_state_rotation_exit_rad_s = node.declare_parameter<double>(
      "robot_state.rotation_exit_rad_s",
      config.robot_state_rotation_exit_rad_s);
  config.robot_state_rotation_exit_hold_sec = node.declare_parameter<double>(
      "robot_state.rotation_exit_hold_sec",
      config.robot_state_rotation_exit_hold_sec);
  config.robot_state_navigation_posture_config =
      node.declare_parameter<std::string>(
          "robot_state.navigation_posture_config",
          config.robot_state_navigation_posture_config);
  config.robot_state_posture_stale_timeout_sec =
      node.declare_parameter<double>(
          "robot_state.posture_stale_timeout_sec",
          config.robot_state_posture_stale_timeout_sec);
  config.robot_state_posture_enter_hold_sec = node.declare_parameter<double>(
      "robot_state.posture_enter_hold_sec",
      config.robot_state_posture_enter_hold_sec);
  config.robot_state_posture_exit_hold_sec = node.declare_parameter<double>(
      "robot_state.posture_exit_hold_sec",
      config.robot_state_posture_exit_hold_sec);
  config.robot_state_body_translation_enter_m =
      node.declare_parameter<double>(
          "robot_state.body_translation_enter_m",
          config.robot_state_body_translation_enter_m);
  config.robot_state_body_translation_exit_m =
      node.declare_parameter<double>(
          "robot_state.body_translation_exit_m",
          config.robot_state_body_translation_exit_m);
  config.robot_state_body_rotation_enter_rad = node.declare_parameter<double>(
      "robot_state.body_rotation_enter_rad",
      config.robot_state_body_rotation_enter_rad);
  config.robot_state_body_rotation_exit_rad = node.declare_parameter<double>(
      "robot_state.body_rotation_exit_rad",
      config.robot_state_body_rotation_exit_rad);
  config.robot_state_arm_translation_enter_m = node.declare_parameter<double>(
      "robot_state.arm_translation_enter_m",
      config.robot_state_arm_translation_enter_m);
  config.robot_state_arm_translation_exit_m = node.declare_parameter<double>(
      "robot_state.arm_translation_exit_m",
      config.robot_state_arm_translation_exit_m);
  config.robot_state_arm_rotation_enter_rad = node.declare_parameter<double>(
      "robot_state.arm_rotation_enter_rad",
      config.robot_state_arm_rotation_enter_rad);
  config.robot_state_arm_rotation_exit_rad = node.declare_parameter<double>(
      "robot_state.arm_rotation_exit_rad",
      config.robot_state_arm_rotation_exit_rad);
  config.robot_state_drop_frames_when_unknown = node.declare_parameter<bool>(
      "robot_state.drop_frames_when_unknown",
      config.robot_state_drop_frames_when_unknown);
  const bool valid_robot_state_config =
      std::isfinite(config.robot_state_odom_history_sec) &&
      std::isfinite(config.robot_state_odom_stale_timeout_sec) &&
      std::isfinite(config.robot_state_rotation_window_sec) &&
      std::isfinite(config.robot_state_rotation_enter_rad_s) &&
      std::isfinite(config.robot_state_rotation_exit_rad_s) &&
      std::isfinite(config.robot_state_rotation_exit_hold_sec) &&
      std::isfinite(config.robot_state_posture_stale_timeout_sec) &&
      std::isfinite(config.robot_state_posture_enter_hold_sec) &&
      std::isfinite(config.robot_state_posture_exit_hold_sec) &&
      std::isfinite(config.robot_state_body_translation_enter_m) &&
      std::isfinite(config.robot_state_body_translation_exit_m) &&
      std::isfinite(config.robot_state_body_rotation_enter_rad) &&
      std::isfinite(config.robot_state_body_rotation_exit_rad) &&
      std::isfinite(config.robot_state_arm_translation_enter_m) &&
      std::isfinite(config.robot_state_arm_translation_exit_m) &&
      std::isfinite(config.robot_state_arm_rotation_enter_rad) &&
      std::isfinite(config.robot_state_arm_rotation_exit_rad) &&
      config.robot_state_odom_history_sec > 0.0 &&
      config.robot_state_odom_stale_timeout_sec > 0.0 &&
      config.robot_state_rotation_window_sec > 0.0 &&
      config.robot_state_rotation_enter_rad_s > 0.0 &&
      config.robot_state_rotation_exit_rad_s >= 0.0 &&
      config.robot_state_rotation_exit_hold_sec >= 0.0 &&
      config.robot_state_posture_stale_timeout_sec > 0.0 &&
      config.robot_state_posture_enter_hold_sec >= 0.0 &&
      config.robot_state_posture_exit_hold_sec >= 0.0 &&
      config.robot_state_body_translation_enter_m > 0.0 &&
      config.robot_state_body_translation_exit_m >= 0.0 &&
      config.robot_state_body_translation_enter_m >=
          config.robot_state_body_translation_exit_m &&
      config.robot_state_body_rotation_enter_rad > 0.0 &&
      config.robot_state_body_rotation_exit_rad >= 0.0 &&
      config.robot_state_body_rotation_enter_rad >=
          config.robot_state_body_rotation_exit_rad &&
      config.robot_state_arm_translation_enter_m > 0.0 &&
      config.robot_state_arm_translation_exit_m >= 0.0 &&
      config.robot_state_arm_translation_enter_m >=
          config.robot_state_arm_translation_exit_m &&
      config.robot_state_arm_rotation_enter_rad > 0.0 &&
      config.robot_state_arm_rotation_exit_rad >= 0.0 &&
      config.robot_state_arm_rotation_enter_rad >=
          config.robot_state_arm_rotation_exit_rad &&
      config.robot_state_rotation_enter_rad_s >=
          config.robot_state_rotation_exit_rad_s &&
      config.robot_state_odom_history_sec >=
          config.robot_state_odom_stale_timeout_sec +
              config.robot_state_rotation_window_sec +
              config.robot_state_rotation_exit_hold_sec &&
      config.robot_state_odom_history_sec >=
          config.robot_state_posture_stale_timeout_sec +
              config.robot_state_posture_enter_hold_sec +
              config.robot_state_posture_exit_hold_sec;
  if (config.robot_state_enabled &&
      (config.odom_topic.empty() || config.tf_topic.empty() ||
       config.robot_state_navigation_posture_config.empty() ||
       !valid_robot_state_config)) {
    throw std::invalid_argument(
        "enabled robot_state requires odom/TF topics, a navigation posture, and valid windows");
  }

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
  config.map_load_mode = node.declare_parameter<std::string>(
      "tsdf.map_load_mode", config.map_load_mode);
  if (config.map_load_mode != "coordinated" &&
      config.map_load_mode != "seed") {
    throw std::invalid_argument(
        "tsdf.map_load_mode must be 'coordinated' or 'seed'");
  }
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
  config.detection_enabled = node.declare_parameter<bool>(
      "detection.enabled", config.detection_enabled);
  config.additional_detection_camera_ids =
      node.declare_parameter<std::vector<std::string>>(
          "detection.additional_camera_ids",
          config.additional_detection_camera_ids);
  std::vector<std::string> unique_detection_camera_ids;
  unique_detection_camera_ids.reserve(
      config.additional_detection_camera_ids.size());
  for (const std::string& camera_id :
       config.additional_detection_camera_ids) {
    if (camera_id.empty()) {
      throw std::invalid_argument(
          "detection.additional_camera_ids must not contain empty ids");
    }
    if (std::find(unique_detection_camera_ids.begin(),
                  unique_detection_camera_ids.end(),
                  camera_id) == unique_detection_camera_ids.end()) {
      unique_detection_camera_ids.push_back(camera_id);
    }
  }
  config.additional_detection_camera_ids =
      std::move(unique_detection_camera_ids);
  config.max_inference_fps = std::max(
      0.0,
      node.declare_parameter<double>("detection.max_inference_fps",
                                     config.max_inference_fps));
  config.perception_deadline_ms = positiveIntOrDefault(
      node.declare_parameter<int>("perception.deadline_ms",
                                  config.perception_deadline_ms),
      config.perception_deadline_ms);
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
  config.text_prompt_file = node.declare_parameter<std::string>(
      "detection.text_prompt_file", config.text_prompt_file);
  config.text_prompts = node.declare_parameter<std::vector<std::string>>(
      "detection.text_prompts", config.text_prompts);
  config.label_thresholds_file = node.declare_parameter<std::string>(
      "detection.label_thresholds_file", config.label_thresholds_file);
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
  config.instance_label_thresholds = loadLabelConfidenceThresholds(
      config.label_thresholds_file, "instance");
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
  config.instance_association_mode = node.declare_parameter<std::string>(
      "instance.association_mode", config.instance_association_mode);
  if (config.instance_association_mode != "legacy" &&
      config.instance_association_mode != "shadow" &&
      config.instance_association_mode != "evidence") {
    config.instance_association_mode = "legacy";
  }
  const auto unit_interval = [&node](const char* name, float fallback) {
    return static_cast<float>(std::clamp(
        node.declare_parameter<double>(name, fallback), 0.0, 1.0));
  };
  config.instance_physical_min_2d_iou = unit_interval(
      "instance.physical_min_2d_iou", config.instance_physical_min_2d_iou);
  config.instance_physical_min_volume_ratio = unit_interval(
      "instance.physical_min_volume_ratio",
      config.instance_physical_min_volume_ratio);
  config.instance_physical_max_normalized_center_distance =
      static_cast<float>(std::max(
          0.0, node.declare_parameter<double>(
                   "instance.physical_max_normalized_center_distance",
                   config.instance_physical_max_normalized_center_distance)));
  config.instance_physical_min_3d_iou = unit_interval(
      "instance.physical_min_3d_iou", config.instance_physical_min_3d_iou);
  config.instance_physical_min_containment = unit_interval(
      "instance.physical_min_containment",
      config.instance_physical_min_containment);
  config.instance_association_overlap_weight = static_cast<float>(std::max(
      0.0, node.declare_parameter<double>(
               "instance.association_overlap_weight",
               config.instance_association_overlap_weight)));
  config.instance_association_center_weight = static_cast<float>(std::max(
      0.0, node.declare_parameter<double>(
               "instance.association_center_weight",
               config.instance_association_center_weight)));
  config.instance_association_size_weight = static_cast<float>(std::max(
      0.0, node.declare_parameter<double>(
               "instance.association_size_weight",
               config.instance_association_size_weight)));
  config.instance_association_projected_2d_weight = static_cast<float>(std::max(
      0.0, node.declare_parameter<double>(
               "instance.association_projected_2d_weight",
               config.instance_association_projected_2d_weight)));
  config.instance_association_semantic_weight = static_cast<float>(std::max(
      0.0, node.declare_parameter<double>(
               "instance.association_semantic_weight",
               config.instance_association_semantic_weight)));
  config.instance_association_recency_weight = static_cast<float>(std::max(
      0.0, node.declare_parameter<double>(
               "instance.association_recency_weight",
               config.instance_association_recency_weight)));
  config.instance_association_min_size_ratio = unit_interval(
      "instance.association_min_size_ratio",
      config.instance_association_min_size_ratio);
  config.instance_association_active_threshold = unit_interval(
      "instance.association_active_threshold",
      config.instance_association_active_threshold);
  config.instance_association_archived_threshold = unit_interval(
      "instance.association_archived_threshold",
      config.instance_association_archived_threshold);
  config.instance_association_merge_threshold = unit_interval(
      "instance.association_merge_threshold",
      config.instance_association_merge_threshold);
  config.instance_presence_log_odds_cap = positiveFloatOrDefault(
      node.declare_parameter<double>("instance.presence_log_odds_cap",
                                     config.instance_presence_log_odds_cap),
      config.instance_presence_log_odds_cap);
  config.instance_presence_active_threshold = static_cast<float>(
      node.declare_parameter<double>("instance.presence_active_threshold",
                                     config.instance_presence_active_threshold));
  config.instance_presence_archive_threshold = static_cast<float>(
      node.declare_parameter<double>("instance.presence_archive_threshold",
                                     config.instance_presence_archive_threshold));
  config.instance_presence_window_sec = std::max(
      0.1, node.declare_parameter<double>("instance.presence_window_sec",
                                          config.instance_presence_window_sec));
  config.instance_presence_positive_window_eligible_frames =
      positiveIntOrDefault(
          node.declare_parameter<int>(
              "instance.presence_positive_window_eligible_frames",
              config.instance_presence_positive_window_eligible_frames),
          config.instance_presence_positive_window_eligible_frames);
  config.instance_presence_min_positive_frames = positiveIntOrDefault(
      node.declare_parameter<int>("instance.presence_min_positive_frames",
                                  config.instance_presence_min_positive_frames),
      config.instance_presence_min_positive_frames);
  config.instance_presence_max_positive_interruptions = std::max(
      0, static_cast<int>(node.declare_parameter<int>(
             "instance.presence_max_positive_interruptions",
             config.instance_presence_max_positive_interruptions)));
  config.instance_presence_viewpoint_baseline_ratio = positiveFloatOrDefault(
      node.declare_parameter<double>(
          "instance.presence_viewpoint_baseline_ratio",
          config.instance_presence_viewpoint_baseline_ratio),
      config.instance_presence_viewpoint_baseline_ratio);
  config.instance_presence_viewpoint_baseline_min_m = positiveFloatOrDefault(
      node.declare_parameter<double>(
          "instance.presence_viewpoint_baseline_min_m",
          config.instance_presence_viewpoint_baseline_min_m),
      config.instance_presence_viewpoint_baseline_min_m);
  config.instance_presence_viewpoint_baseline_max_m = std::max(
      config.instance_presence_viewpoint_baseline_min_m,
      positiveFloatOrDefault(
          node.declare_parameter<double>(
              "instance.presence_viewpoint_baseline_max_m",
              config.instance_presence_viewpoint_baseline_max_m),
          config.instance_presence_viewpoint_baseline_max_m));
  config.instance_presence_viewpoint_min_angle_deg = positiveFloatOrDefault(
      node.declare_parameter<double>(
          "instance.presence_viewpoint_min_angle_deg",
          config.instance_presence_viewpoint_min_angle_deg),
      config.instance_presence_viewpoint_min_angle_deg);
  config.instance_presence_same_viewpoint_max_distance_m =
      positiveFloatOrDefault(
          node.declare_parameter<double>(
              "instance.presence_same_viewpoint_max_distance_m",
              config.instance_presence_same_viewpoint_max_distance_m),
          config.instance_presence_same_viewpoint_max_distance_m);
  config.instance_presence_same_viewpoint_min_confidence = unit_interval(
      "instance.presence_same_viewpoint_min_confidence",
      config.instance_presence_same_viewpoint_min_confidence);
  config.instance_presence_same_viewpoint_min_bbox_quality = unit_interval(
      "instance.presence_same_viewpoint_min_bbox_quality",
      config.instance_presence_same_viewpoint_min_bbox_quality);
  config.instance_presence_min_negative_frames = positiveIntOrDefault(
      node.declare_parameter<int>("instance.presence_min_negative_frames",
                                  config.instance_presence_min_negative_frames),
      config.instance_presence_min_negative_frames);
  config.instance_presence_min_depth_samples = positiveIntOrDefault(
      node.declare_parameter<int>("instance.presence_min_depth_samples",
                                  config.instance_presence_min_depth_samples),
      config.instance_presence_min_depth_samples);
  config.instance_presence_min_valid_depth_coverage = unit_interval(
      "instance.presence_min_valid_depth_coverage",
      config.instance_presence_min_valid_depth_coverage);
  config.instance_presence_min_free_space_ratio = unit_interval(
      "instance.presence_min_free_space_ratio",
      config.instance_presence_min_free_space_ratio);
  config.instance_presence_max_occlusion_ratio = unit_interval(
      "instance.presence_max_occlusion_ratio",
      config.instance_presence_max_occlusion_ratio);
  config.instance_presence_depth_margin_m = positiveFloatOrDefault(
      node.declare_parameter<double>("instance.presence_depth_margin_m",
                                     config.instance_presence_depth_margin_m),
      config.instance_presence_depth_margin_m);
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
  config.instance_observation_history_capacity = positiveSizeOrDefault(
      node.declare_parameter<int>(
          "instance.observation_history_capacity",
          static_cast<int>(config.instance_observation_history_capacity)),
      config.instance_observation_history_capacity);
  config.input_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("queues.input_queue_size",
                                  static_cast<int>(config.input_queue_size)),
      config.input_queue_size);
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
  config.shutdown_drain_timeout_ms = positiveIntOrDefault(
      node.declare_parameter<int>("queues.shutdown_drain_timeout_ms",
                                  config.shutdown_drain_timeout_ms),
      config.shutdown_drain_timeout_ms);

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
  config.scene_store_enabled = node.declare_parameter<bool>(
      "persistence.scene_store_enabled", config.scene_store_enabled);
  config.scene_store_path = node.declare_parameter<std::string>(
      "persistence.scene_store_path", config.scene_store_path);
  config.ephemeral_run = node.declare_parameter<bool>(
      "persistence.ephemeral_run", config.ephemeral_run);
  config.ephemeral_workspace_root = node.declare_parameter<std::string>(
      "persistence.ephemeral_workspace_root",
      config.ephemeral_workspace_root);
  config.scene_store_flush_period_ms = positiveIntOrDefault(
      node.declare_parameter<int>("persistence.scene_store_flush_period_ms",
                                  config.scene_store_flush_period_ms),
      config.scene_store_flush_period_ms);
  config.scene_store_flush_batch_size = positiveSizeOrDefault(
      node.declare_parameter<int>("persistence.scene_store_flush_batch_size",
                                  static_cast<int>(
                                      config.scene_store_flush_batch_size)),
      config.scene_store_flush_batch_size);
  config.scene_store_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("persistence.scene_store_queue_size",
                                  static_cast<int>(
                                      config.scene_store_queue_size)),
      config.scene_store_queue_size);
  config.scene_store_soft_lag_revisions = positiveSizeOrDefault(
      node.declare_parameter<int>(
          "persistence.scene_store_soft_lag_revisions",
          static_cast<int>(config.scene_store_soft_lag_revisions)),
      static_cast<std::size_t>(config.scene_store_soft_lag_revisions));
  config.scene_store_hard_lag_revisions = positiveSizeOrDefault(
      node.declare_parameter<int>(
          "persistence.scene_store_hard_lag_revisions",
          static_cast<int>(config.scene_store_hard_lag_revisions)),
      static_cast<std::size_t>(config.scene_store_hard_lag_revisions));
  config.scene_store_hard_lag_revisions = std::max(
      config.scene_store_hard_lag_revisions,
      config.scene_store_soft_lag_revisions);
  config.scene_store_terminal_failure_timeout_ms = positiveIntOrDefault(
      node.declare_parameter<int>(
          "persistence.scene_store_terminal_failure_timeout_ms",
          config.scene_store_terminal_failure_timeout_ms),
      config.scene_store_terminal_failure_timeout_ms);
  config.furniture_config_file = node.declare_parameter<std::string>(
      "scene_graph.furniture_config_file", config.furniture_config_file);
  config.furniture_rebuild_timeout_ms = positiveIntOrDefault(
      node.declare_parameter<int>("scene_graph.furniture_rebuild_timeout_ms",
                                  config.furniture_rebuild_timeout_ms),
      config.furniture_rebuild_timeout_ms);
  std::filesystem::path furniture_config_path(
      config.furniture_config_file);
  if (furniture_config_path.is_relative()) {
    furniture_config_path = std::filesystem::path(
                                ament_index_cpp::get_package_share_directory(
                                    "roomie")) /
                            "config" / furniture_config_path;
  }
  config.furniture_config_file = furniture_config_path.string();
  config.furniture_graph_config =
      loadFurnitureGraphConfig(furniture_config_path);
  config.online_snapshot_enabled = node.declare_parameter<bool>(
      "artifacts.snapshot_enabled", config.online_snapshot_enabled);
  config.asset_store_root = node.declare_parameter<std::string>(
      "artifacts.asset_store_root", config.asset_store_root);
  config.snapshot_top_k = positiveSizeOrDefault(
      node.declare_parameter<int>("artifacts.snapshot_top_k",
                                  static_cast<int>(config.snapshot_top_k)),
      config.snapshot_top_k);
  config.snapshot_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>("artifacts.snapshot_queue_size",
                                  static_cast<int>(config.snapshot_queue_size)),
      config.snapshot_queue_size);
  config.snapshot_control_queue_size = positiveSizeOrDefault(
      node.declare_parameter<int>(
          "artifacts.snapshot_control_queue_size",
          static_cast<int>(config.snapshot_control_queue_size)),
      config.snapshot_control_queue_size);
  config.snapshot_minimum_quality = static_cast<float>(std::clamp(
      node.declare_parameter<double>("artifacts.snapshot_minimum_quality",
                                     config.snapshot_minimum_quality),
      0.0,
      1.0));
  config.snapshot_azimuth_bucket_degrees = positiveFloatOrDefault(
      node.declare_parameter<double>("artifacts.snapshot_azimuth_bucket_degrees",
                                     config.snapshot_azimuth_bucket_degrees),
      config.snapshot_azimuth_bucket_degrees);
  config.snapshot_elevation_bucket_degrees = positiveFloatOrDefault(
      node.declare_parameter<double>("artifacts.snapshot_elevation_bucket_degrees",
                                     config.snapshot_elevation_bucket_degrees),
      config.snapshot_elevation_bucket_degrees);
  config.snapshot_scale_bucket_ratio = static_cast<float>(std::max(
      1.0001,
      node.declare_parameter<double>("artifacts.snapshot_scale_bucket_ratio",
                                     config.snapshot_scale_bucket_ratio)));
  config.snapshot_diversity_min_quality_ratio = static_cast<float>(std::clamp(
      node.declare_parameter<double>(
          "artifacts.snapshot_diversity_min_quality_ratio",
          config.snapshot_diversity_min_quality_ratio),
      0.0,
      1.0));
  config.asset_gc_grace_period_sec = std::max(
      0.0,
      node.declare_parameter<double>("artifacts.asset_gc_grace_period_sec",
                                     config.asset_gc_grace_period_sec));
  config.asset_gc_period_ms = positiveIntOrDefault(
      node.declare_parameter<int>("artifacts.asset_gc_period_ms",
                                  config.asset_gc_period_ms),
      config.asset_gc_period_ms);
  config.dam_artifacts_enabled = node.declare_parameter<bool>(
      "artifacts.dam_enabled", config.dam_artifacts_enabled);
  config.dam_model_id = node.declare_parameter<std::string>(
      "artifacts.dam_model_id", config.dam_model_id);
  config.dam_prompt_hash = node.declare_parameter<std::string>(
      "artifacts.dam_prompt_hash", config.dam_prompt_hash);
  config.dam_output_schema_version = node.declare_parameter<std::string>(
      "artifacts.dam_output_schema_version",
      config.dam_output_schema_version);
  config.dam_python_executable = node.declare_parameter<std::string>(
      "artifacts.dam_python_executable", config.dam_python_executable);
  config.dam_worker_script = node.declare_parameter<std::string>(
      "artifacts.dam_worker_script", config.dam_worker_script);
  config.dam_source = node.declare_parameter<std::string>(
      "artifacts.dam_source", config.dam_source);
  config.dam_model_path = node.declare_parameter<std::string>(
      "artifacts.dam_model_path", config.dam_model_path);
  config.dam_conversation_mode = node.declare_parameter<std::string>(
      "artifacts.dam_conversation_mode", config.dam_conversation_mode);
  config.dam_prompt_mode = node.declare_parameter<std::string>(
      "artifacts.dam_prompt_mode", config.dam_prompt_mode);
  config.dam_query = node.declare_parameter<std::string>(
      "artifacts.dam_query", config.dam_query);
  config.dam_max_new_tokens = positiveIntOrDefault(
      node.declare_parameter<int>("artifacts.dam_max_new_tokens",
                                  config.dam_max_new_tokens),
      config.dam_max_new_tokens);
  config.dam_temperature = std::max(
      0.0, node.declare_parameter<double>("artifacts.dam_temperature",
                                          config.dam_temperature));
  config.dam_top_p = std::clamp(
      node.declare_parameter<double>("artifacts.dam_top_p",
                                     config.dam_top_p),
      0.0, 1.0);
  config.dam_bbox_pad_px = std::max(
      0.0, node.declare_parameter<double>("artifacts.dam_bbox_pad_px",
                                          config.dam_bbox_pad_px));
  config.dam_startup_timeout_ms = positiveIntOrDefault(
      node.declare_parameter<int>("artifacts.dam_startup_timeout_ms",
                                  config.dam_startup_timeout_ms),
      config.dam_startup_timeout_ms);
  config.dam_request_timeout_ms = positiveIntOrDefault(
      node.declare_parameter<int>("artifacts.dam_request_timeout_ms",
                                  config.dam_request_timeout_ms),
      config.dam_request_timeout_ms);
  config.dam_shutdown_timeout_ms = positiveIntOrDefault(
      node.declare_parameter<int>("artifacts.dam_shutdown_timeout_ms",
                                  config.dam_shutdown_timeout_ms),
      config.dam_shutdown_timeout_ms);
  config.artifact_new_object_window_ms = positiveIntOrDefault(
      node.declare_parameter<int>("artifacts.new_object_window_ms",
                                  config.artifact_new_object_window_ms),
      config.artifact_new_object_window_ms);
  config.artifact_new_object_interactive_limit = positiveSizeOrDefault(
      node.declare_parameter<int>(
          "artifacts.new_object_interactive_limit",
          static_cast<int>(config.artifact_new_object_interactive_limit)),
      config.artifact_new_object_interactive_limit);
  config.embedding_artifacts_enabled = node.declare_parameter<bool>(
      "artifacts.embedding_enabled", config.embedding_artifacts_enabled);
  config.embedding_model_id = node.declare_parameter<std::string>(
      "artifacts.embedding_model_id", config.embedding_model_id);
  config.embedding_dimension = positiveSizeOrDefault(
      node.declare_parameter<int>("artifacts.embedding_dimension",
                                  static_cast<int>(config.embedding_dimension)),
      config.embedding_dimension);
  config.embedding_python_executable = node.declare_parameter<std::string>(
      "artifacts.embedding_python_executable",
      config.embedding_python_executable);
  config.embedding_worker_script = node.declare_parameter<std::string>(
      "artifacts.embedding_worker_script", config.embedding_worker_script);
  config.embedding_model_path = node.declare_parameter<std::string>(
      "artifacts.embedding_model_path", config.embedding_model_path);
  config.embedding_device = node.declare_parameter<std::string>(
      "artifacts.embedding_device", config.embedding_device);
  config.embedding_batch_size = positiveSizeOrDefault(
      node.declare_parameter<int>(
          "artifacts.embedding_batch_size",
          static_cast<int>(config.embedding_batch_size)),
      config.embedding_batch_size);
  config.embedding_record_history_capacity = positiveSizeOrDefault(
      node.declare_parameter<int>(
          "artifacts.embedding_record_history_capacity",
          static_cast<int>(config.embedding_record_history_capacity)),
      config.embedding_record_history_capacity);
  config.embedding_startup_timeout_ms = positiveIntOrDefault(
      node.declare_parameter<int>("artifacts.embedding_startup_timeout_ms",
                                  config.embedding_startup_timeout_ms),
      config.embedding_startup_timeout_ms);
  config.embedding_request_timeout_ms = positiveIntOrDefault(
      node.declare_parameter<int>("artifacts.embedding_request_timeout_ms",
                                  config.embedding_request_timeout_ms),
      config.embedding_request_timeout_ms);
  config.embedding_shutdown_timeout_ms = positiveIntOrDefault(
      node.declare_parameter<int>("artifacts.embedding_shutdown_timeout_ms",
                                  config.embedding_shutdown_timeout_ms),
      config.embedding_shutdown_timeout_ms);
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
