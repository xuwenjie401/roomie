#pragma once

#include <cstddef>
#include <string>

#include <rclcpp/rclcpp.hpp>

namespace roomie {

struct PipelineConfig {
  std::string world_frame = "world";
  std::string mapping_camera_id = "head_front_left_color";

  int boxer_input_size = 960;
  int patch_rows = 60;
  int patch_cols = 60;
  float min_patch_coverage_ratio = 0.05f;
  double max_inference_fps = 10.0;

  std::size_t mapping_queue_size = 30;
  std::size_t detection_queue_size = 8;
  std::size_t inference_request_queue_size = 2;
  std::size_t inference_response_queue_size = 8;

  double publish_period_sec = 1.0;

  bool load_map = true;
  std::string map_load_path;
  bool save_map = false;
  std::string map_save_path;
  bool save_instance_map = true;
  std::string instance_map_save_path;

  static PipelineConfig declareAndLoad(rclcpp::Node& node);
};

}  // namespace roomie
