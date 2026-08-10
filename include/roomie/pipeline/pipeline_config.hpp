#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "roomie/pipeline/label_thresholds.hpp"
#include "roomie/pipeline/types.hpp"
#include "roomie/scene/furniture_graph.hpp"

namespace roomie {

struct PipelineConfig {
  std::string world_frame = "map";
  std::string mapping_camera_id = "head_color";
  std::string mapping_camera_frame = "head_color";
  std::string rgb_topic = "/roomie/input/head_color/image_rect";
  std::string depth_topic = "/roomie/input/head_color/depth_registered";
  std::string camera_info_topic = "/roomie/input/head_color/camera_info";
  std::string tf_topic = "/tf";
  std::string tf_static_topic = "/tf_static";
  std::string tsdf_output_topic = "/roomie/map_surface";
  std::string detection_debug_image_topic = "/roomie/detections_2d_image";
  std::string raw_detections_topic = "/roomie/raw_detections";
  std::string object_markers_topic = "/roomie/objects";
  std::string instance_markers_topic = "/roomie/instances";

  int camera_width = 640;
  int camera_height = 400;
  float camera_fx = 305.2087402344f;
  float camera_fy = 305.0057678223f;
  float camera_cx = 318.5672912598f;
  float camera_cy = 204.0587768555f;

  std::string robot_mask_robot_config = "G2/robot.yaml";
  std::string robot_mask_camera_config = "G2/cameras.yaml";
  double robot_mask_reuse_translation_epsilon_m = 5.0e-6;
  double robot_mask_reuse_rotation_epsilon_rad = 5.0e-6;

  float depth_min_m = 0.1f;
  float depth_max_m = 10.0f;
  float depth_scale = 0.001f;

  float voxel_size_m = 0.03f;
  float truncation_distance_vox = 8.0f;
  float max_weight = 20.0f;
  float max_integration_distance_m = 6.0f;
  float min_visualization_weight = 1.0f;
  float min_color_weight = 0.1f;
  float surface_visualization_distance_vox = 1.0f;
  double surface_cache_rebuild_period_sec = 5.0;
  std::string map_backend = "cpu";

  int boxer_input_size = 960;
  float min_patch_coverage_ratio = 0.05f;
  float patch_depth_max_m = 0.0f;
  int patch_depth_zbuffer_scale = 4;
  int patch_depth_zbuffer_splat_radius_cells = 1;
  float patch_depth_zbuffer_front_quantile = 0.25f;
  int patch_depth_zbuffer_min_cells_per_patch = 1;
  bool detection_enabled = true;
  double max_inference_fps = 10.0;
  int perception_deadline_ms = 10000;
  bool python_backend_enabled = true;
  std::string python_executable = "/home/lindenbot/miniconda3/envs/jarvis/bin/python";
  std::string python_worker_script =
      "/home/lindenbot/RealityLab/jarvis/src/roomie/scripts/roomie_python_inference_worker.py";
  std::string boxer_repo_path = "/home/lindenbot/RealityLab/boxer";
  std::string boxernet_ckpt_path =
      "/home/lindenbot/hugging_face/boxer/boxernet_hw960in4x6d768-3e37cfc4.ckpt";
  std::string inference_device = "cuda";
  std::string inference_precision = "auto";
  std::string text_prompt_file;
  std::vector<std::string> text_prompts = {"lvisplus"};
  std::string label_thresholds_file;
  float owl_min_confidence = 0.25f;
  float owl_nms_iou_threshold = 0.5f;
  float boxernet_min_confidence = 0.35f;
  float robot_bbox_mask_overlap = 0.25f;
  float robot_bbox_center_overlap = 0.5f;
  int robot_mask_dilate_px = 3;
  bool show_3d_label_score = true;

  float instance_min_confidence = 0.35f;
  LabelConfidenceThresholds instance_label_thresholds;
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
  float instance_far_observation_distance_m = 3.5f;
  float instance_far_promotion_weight = 0.25f;
  float instance_fusion_prior_mass_cap = 6.0f;
  int instance_high_quality_min_count = 3;
  float instance_high_quality_min_mass = 1.8f;
  float instance_geometry_shell_thickness_m = 0.08f;
  int instance_geometry_empty_inside_points = 6;
  int instance_geometry_min_unique_voxels = 12;
  float instance_geometry_confirm_score = 0.62f;
  float instance_geometry_suppress_score = 0.35f;
  float instance_confirmed_geometry_edge_freeze_weight = 0.999f;
  float instance_confirmed_geometry_large_min_volume_m3 = 0.35f;
  float instance_confirmed_geometry_far_center_shift_ratio = 0.08f;
  float instance_confirmed_geometry_far_center_shift_min_m = 0.04f;
  float instance_confirmed_geometry_center_shift_min_extent_m = 0.25f;
  // Retained diagnostic evidence is recent-only; all-time promotion/support
  // aggregates remain independent scalar counters.
  std::size_t instance_observation_history_capacity = 256;
  std::size_t input_queue_size = 30;
  std::size_t pending_frame_limit = 120;
  std::size_t mapping_queue_size = 30;
  std::size_t detection_queue_size = 8;
  std::size_t inference_request_queue_size = 2;
  std::size_t inference_response_queue_size = 8;
  int shutdown_drain_timeout_ms = 5000;

  double max_image_stamp_delta_sec = 0.002;
  double tf_buffer_duration_sec = 5.0;
  double max_tf_gap_sec = 0.2;
  double max_tf_translation_step_m = 0.5;
  double max_tf_rotation_step_deg = 45.0;

  double publish_period_sec = 5.0;

  bool load_map = true;
  std::string map_load_path;
  // coordinated: a durable scene may only be restored with its published map
  // manifest. seed: the configured map is an explicit new base and all
  // durable scene-derived state is discarded before startup.
  std::string map_load_mode = "coordinated";
  bool freeze_tsdf_map = false;
  bool save_map = false;
  std::string map_save_path;
  bool load_scene_graph = false;
  std::string scene_graph_load_path;
  bool scene_store_enabled = true;
  std::string scene_store_path = "/tmp/roomie_scene.sqlite3";
  int scene_store_flush_period_ms = 1000;
  std::size_t scene_store_flush_batch_size = 16;
  std::size_t scene_store_queue_size = 64;
  SceneRevision scene_store_soft_lag_revisions = 32;
  SceneRevision scene_store_hard_lag_revisions = 64;
  int scene_store_terminal_failure_timeout_ms = 5000;
  bool online_snapshot_enabled = true;
  std::string asset_store_root = "/tmp/roomie_assets";
  std::size_t snapshot_top_k = 3;
  std::size_t snapshot_queue_size = 32;
  std::size_t snapshot_control_queue_size = 64;
  float snapshot_minimum_quality = 0.02f;
  float snapshot_azimuth_bucket_degrees = 45.0f;
  float snapshot_elevation_bucket_degrees = 30.0f;
  float snapshot_scale_bucket_ratio = 1.41421356f;
  float snapshot_diversity_min_quality_ratio = 0.65f;
  double asset_gc_grace_period_sec = 300.0;
  int asset_gc_period_ms = 60000;
  bool dam_artifacts_enabled = true;
  std::string dam_model_id = "nvidia-dam-3b-official-v1";
  // Optional expected canonical hash. Empty computes it from all resident DAM
  // execution parameters; a non-empty mismatch is a startup error.
  std::string dam_prompt_hash;
  std::string dam_output_schema_version = "roomie.dam.v1";
  std::string dam_python_executable =
      "/home/lindenbot/RealityLab/.venvs/roomie-dam/bin/python";
  std::string dam_worker_script =
      "/home/lindenbot/RealityLab/jarvis/src/roomie/scripts/roomie_python_dam_worker.py";
  std::string dam_source = "/home/lindenbot/RealityLab/describe-anything";
  std::string dam_model_path = "/home/lindenbot/hugging_face/DAM-3B";
  std::string dam_conversation_mode = "v1";
  std::string dam_prompt_mode = "full+focal_crop";
  std::string dam_query =
      "<image>\nDescribe only the visible object inside the masked region in detail. "
      "Include its color, material, shape, parts, pose, and distinctive visual "
      "details. Do not describe unrelated background.";
  int dam_max_new_tokens = 256;
  double dam_temperature = 0.2;
  double dam_top_p = 0.9;
  double dam_bbox_pad_px = 2.0;
  int dam_startup_timeout_ms = 120000;
  int dam_request_timeout_ms = 120000;
  int dam_shutdown_timeout_ms = 500;
  int artifact_new_object_window_ms = 10000;
  std::size_t artifact_new_object_interactive_limit = 5;
  bool embedding_artifacts_enabled = true;
  std::string embedding_model_id = "sentence-t5-large.local.v1";
  std::size_t embedding_dimension = 768;
  std::string embedding_python_executable =
      "/home/lindenbot/miniconda3/envs/jarvis/bin/python";
  std::string embedding_worker_script =
      "/home/lindenbot/RealityLab/jarvis/src/roomie/scripts/roomie_sentence_transformer_worker.py";
  std::string embedding_model_path =
      "/home/lindenbot/hugging_face/sentence_t5_large";
  std::string embedding_device = "cpu";
  std::size_t embedding_batch_size = 16;
  std::size_t embedding_record_history_capacity = 1024;
  int embedding_startup_timeout_ms = 120000;
  int embedding_request_timeout_ms = 120000;
  int embedding_shutdown_timeout_ms = 500;
  bool freeze_instances = false;
  std::string furniture_config_file = "scene_qa/furniture.json";
  FurnitureGraphConfig furniture_graph_config =
      defaultFurnitureGraphConfig();
  int furniture_rebuild_timeout_ms = 5000;
  bool save_scene_graph = true;
  std::string scene_graph_save_path;
  std::string save_dsg_service = "/roomie/save_dsg";
  bool snapshot_remake_enabled = false;
  std::string snapshot_staging_dir = "/tmp/roomie_object_snapshots";
  std::string snapshot_image_subdir = "snapshots";
  float instance_snapshot_first_min_quality = 0.05f;
  float instance_snapshot_min_quality = 0.18f;
  float instance_snapshot_min_box_area_px = 300.0f;
  float instance_snapshot_position_weight = 0.5f;
  float instance_snapshot_size_weight = 0.5f;
  float instance_snapshot_replace_min_quality_delta = 0.12f;
  float instance_snapshot_replace_min_quality_ratio = 1.20f;

  bool file_logging_enabled = true;
  std::string file_logging_root_dir =
      "/home/lindenbot/RealityLab/jarvis/src/roomie/logs/roomie_runs";
  double file_logging_period_sec = 2.0;

  static PipelineConfig declareAndLoad(rclcpp::Node& node);
};

}  // namespace roomie
