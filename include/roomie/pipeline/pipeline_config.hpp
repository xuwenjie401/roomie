#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace roomie {

struct PipelineConfig {
  std::string world_frame = "world";
  std::string mapping_camera_id = "head_front_left_color";
  std::string mapping_camera_frame = "head_front_left_color";
  std::string rgb_topic = "/head_front_left_color_rgb";
  std::string depth_topic = "/head_front_left_color_depth";
  std::string mask_topic = "/head_front_left_color_robot_mask";
  std::string camera_info_topic = "/head_front_left_color_camera_info";
  std::string tf_topic = "/tf";
  std::string tf_static_topic = "/tf_static";
  std::string tsdf_output_topic = "/roomie/map_surface";
  std::string detection_debug_image_topic = "/roomie/detections_2d_image";
  std::string raw_detections_topic = "/roomie/raw_detections";
  std::string object_markers_topic = "/roomie/objects";
  std::string instance_markers_topic = "/roomie/instances";

  int camera_width = 640;
  int camera_height = 480;
  float camera_fx = 211.2f;
  float camera_fy = 211.2f;
  float camera_cx = 291.19999872f;
  float camera_cy = 240.0f;

  float depth_min_m = 0.1f;
  float depth_max_m = 10.0f;
  float depth_scale = 0.001f;
  int mask_robot_threshold = 0;

  float voxel_size_m = 0.03f;
  float truncation_distance_vox = 8.0f;
  float max_weight = 20.0f;
  float max_integration_distance_m = 6.0f;
  float min_visualization_weight = 1.0f;
  float min_color_weight = 0.1f;
  float surface_visualization_distance_vox = 1.0f;
  std::string map_backend = "cpu";

  int boxer_input_size = 960;
  float min_patch_coverage_ratio = 0.05f;
  float patch_depth_max_m = 0.0f;
  int patch_depth_zbuffer_scale = 4;
  int patch_depth_zbuffer_splat_radius_cells = 1;
  float patch_depth_zbuffer_front_quantile = 0.25f;
  int patch_depth_zbuffer_min_cells_per_patch = 1;
  double max_inference_fps = 10.0;
  bool python_backend_enabled = true;
  std::string python_executable = "/home/agxi/miniconda3/envs/jarvis/bin/python";
  std::string python_worker_script =
      "/home/agxi/RealityLab/jarvis/src/roomie/scripts/roomie_python_inference_worker.py";
  std::string boxer_repo_path = "/home/agxi/RealityLab/boxer";
  std::string boxernet_ckpt_path =
      "/home/agxi/huggingface/boxer/boxernet_hw960in4x6d768-3e37cfc4.ckpt";
  std::string inference_device = "cuda";
  std::string inference_precision = "auto";
  std::vector<std::string> text_prompts = {"lvisplus"};
  float owl_min_confidence = 0.25f;
  float owl_nms_iou_threshold = 0.5f;
  float boxernet_min_confidence = 0.35f;
  float robot_bbox_mask_overlap = 0.25f;
  float robot_bbox_center_overlap = 0.5f;
  int robot_mask_dilate_px = 3;
  bool show_3d_label_score = true;

  float instance_min_confidence = 0.35f;
  float instance_object_min_confidence = 0.5f;
  float instance_min_bbox_size_m = 0.02f;
  float instance_max_bbox_size_m = 8.0f;
  float instance_match_iou_threshold = 0.10f;
  float instance_match_center_distance_m = 0.75f;
  int instance_min_support_count = 3;
  float instance_min_confidence_mass = 1.5f;
  int instance_tentative_max_missed = 4;
  int instance_inactive_after_missed = 4;
  float instance_duplicate_iou_threshold = 0.70f;
  float instance_duplicate_size_ratio_min = 0.70f;
  float instance_confirmed_duplicate_iou_threshold = 0.20f;
  float instance_duplicate_containment_threshold = 0.48f;
  float instance_duplicate_center_distance_m = 0.85f;
  float instance_duplicate_small_object_volume_ratio = 0.15f;
  float instance_small_duplicate_max_volume_m3 = 0.025f;
  float instance_small_duplicate_max_extent_m = 0.65f;
  float instance_small_duplicate_iou_threshold = 0.25f;
  float instance_small_duplicate_center_ratio = 0.35f;
  float instance_small_duplicate_size_ratio_min = 0.55f;
  float instance_quality_observation_min_quality = 0.45f;
  float instance_close_observation_distance_m = 3.5f;
  float instance_far_observation_distance_m = 4.5f;
  float instance_far_promotion_weight = 0.25f;
  float instance_fusion_prior_mass_cap = 6.0f;
  int instance_high_quality_min_count = 3;
  float instance_high_quality_min_mass = 1.8f;
  double instance_geometry_check_period_sec = 2.0;
  double instance_geometry_recent_window_sec = 5.0;
  float instance_geometry_shell_thickness_m = 0.08f;
  int instance_geometry_empty_inside_points = 6;
  int instance_geometry_min_unique_voxels = 12;
  float instance_geometry_confirm_score = 0.62f;
  float instance_geometry_suppress_score = 0.35f;
  float instance_geometry_recover_score = 0.55f;
  int instance_geometry_failures_before_suppress = 2;
  int instance_geometry_inactive_delete_bad_count = 3;
  float instance_geometry_empty_small_object_max_volume_m3 = 0.025f;
  float instance_geometry_empty_small_object_max_extent_m = 0.65f;
  float instance_geometry_reevaluate_center_delta_m = 0.12f;
  float instance_geometry_reevaluate_size_ratio = 0.20f;
  float instance_geometry_reevaluate_yaw_delta_deg = 15.0f;
  float instance_confirmed_geometry_edge_freeze_weight = 0.999f;
  float instance_confirmed_geometry_large_min_volume_m3 = 0.35f;
  float instance_confirmed_geometry_far_center_shift_ratio = 0.08f;
  float instance_confirmed_geometry_far_center_shift_min_m = 0.04f;
  float instance_confirmed_geometry_center_shift_min_extent_m = 0.25f;

  std::size_t input_queue_size = 30;
  std::size_t output_queue_size = 2;
  std::size_t sync_queue_size = 30;
  std::size_t pending_frame_limit = 120;
  std::size_t mapping_queue_size = 30;
  std::size_t detection_queue_size = 8;
  std::size_t inference_request_queue_size = 2;
  std::size_t inference_response_queue_size = 8;

  double max_image_stamp_delta_sec = 0.002;
  double tf_buffer_duration_sec = 5.0;
  double max_tf_gap_sec = 0.2;
  double max_tf_translation_step_m = 0.5;
  double max_tf_rotation_step_deg = 45.0;

  double publish_period_sec = 5.0;

  bool load_map = true;
  std::string map_load_path;
  bool freeze_tsdf_map = false;
  bool save_map = false;
  std::string map_save_path;
  bool load_instance_map = false;
  std::string instance_map_load_path;
  bool save_instance_map = true;
  std::string instance_map_save_path;
  std::string save_dsg_service = "/roomie/save_dsg";

  bool file_logging_enabled = true;
  std::string file_logging_root_dir =
      "/home/agxi/RealityLab/jarvis/src/roomie/logs/roomie_runs";
  double file_logging_period_sec = 2.0;

  static PipelineConfig declareAndLoad(rclcpp::Node& node);
};

}  // namespace roomie
