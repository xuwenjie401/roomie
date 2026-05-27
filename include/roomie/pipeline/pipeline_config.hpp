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
  int patch_rows = 60;
  int patch_cols = 60;
  float min_patch_coverage_ratio = 0.05f;
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
  float boxernet_min_confidence = 0.5f;
  float robot_bbox_mask_overlap = 0.25f;
  float robot_bbox_center_overlap = 0.5f;
  int robot_mask_dilate_px = 3;
  bool show_3d_label_score = true;

  std::size_t input_queue_size = 30;
  std::size_t output_queue_size = 1;
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
  bool save_instance_map = true;
  std::string instance_map_save_path;

  bool file_logging_enabled = true;
  std::string file_logging_root_dir =
      "/home/agxi/RealityLab/jarvis/src/roomie/logs/roomie_runs";
  double file_logging_period_sec = 2.0;

  static PipelineConfig declareAndLoad(rclcpp::Node& node);
};

}  // namespace roomie
