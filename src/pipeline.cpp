#include "roomie/pipeline/pipeline.hpp"

#include "roomie/dsg/object_graph_io.hpp"

namespace roomie {

RoomiePipeline::RoomiePipeline(rclcpp::Node& node, PipelineConfig config)
    : config_(std::move(config)),
      run_logger_(std::make_shared<RunLogger>(config_)),
      mapping_queue_(config_.mapping_queue_size),
      detection_queue_(config_.detection_queue_size),
      inference_response_queue_(config_.inference_response_queue_size),
      ros_io_thread_(mapping_queue_, detection_queue_),
      map_thread_(mapping_queue_, config_),
      python_backend_(config_),
      instance_map_thread_(inference_response_queue_, map_thread_, config_),
      detection_bridge_thread_(node,
                               detection_queue_,
                               inference_response_queue_,
                               map_thread_,
                               python_backend_,
                               config_),
      publisher_persistence_thread_(node, instance_map_thread_, map_thread_, config_) {
  RunLogger::setGlobal(run_logger_);
  if (run_logger_ && run_logger_->enabled()) {
    RCLCPP_INFO(node.get_logger(),
                "roomie file logs: %s",
                run_logger_->runDirectory().c_str());
    run_logger_->log("pipeline",
                     "config map_backend=" + config_.map_backend +
                         " world_frame=" + config_.world_frame +
                         " camera_id=" + config_.mapping_camera_id +
                         " debug_image=" + config_.detection_debug_image_topic +
                         " raw_detections=" + config_.raw_detections_topic);
  }

  if (config_.load_scene_graph) {
    if (config_.scene_graph_load_path.empty()) {
      const std::string message =
          "persistence.load_scene_graph is true but scene_graph_load_path is empty";
      RCLCPP_WARN(node.get_logger(), "%s", message.c_str());
      RunLogger::logGlobal("persistence", message);
    } else {
      ObjectGraphSnapshot snapshot;
      std::string loaded_world_frame;
      std::string error;
      if (loadObjectGraphSnapshotJson(config_.scene_graph_load_path,
                                      &snapshot,
                                      &loaded_world_frame,
                                      &error)) {
        if (!loaded_world_frame.empty() && loaded_world_frame != config_.world_frame) {
          RCLCPP_WARN(node.get_logger(),
                      "loaded DSG world_frame=%s differs from configured world_frame=%s",
                      loaded_world_frame.c_str(),
                      config_.world_frame.c_str());
        }
        if (instance_map_thread_.loadObjectGraphSnapshot(snapshot, &error)) {
          RCLCPP_INFO(node.get_logger(),
                      "loaded roomie scene graph objects=%zu relations=%zu from %s",
                      snapshot.objects.size(),
                      snapshot.relations.size(),
                      config_.scene_graph_load_path.c_str());
          RunLogger::logGlobal(
              "persistence",
              "loaded scene_graph objects=" + std::to_string(snapshot.objects.size()) +
                  " relations=" + std::to_string(snapshot.relations.size()) +
                  " path=" + config_.scene_graph_load_path);
        } else {
          RCLCPP_WARN(node.get_logger(),
                      "failed to restore scene graph tracks: %s",
                      error.c_str());
          RunLogger::logGlobal("persistence",
                               "failed to restore scene graph tracks: " + error);
        }
      } else {
        RCLCPP_WARN(node.get_logger(),
                    "failed to load scene graph from %s: %s",
                    config_.scene_graph_load_path.c_str(),
                    error.c_str());
        RunLogger::logGlobal("persistence",
                             "failed to load scene_graph path=" +
                                 config_.scene_graph_load_path +
                                 " error=" + error);
      }
    }
  }

  RosCameraSubscriptionConfig mapping_camera;
  mapping_camera.camera_id = config_.mapping_camera_id;
  mapping_camera.camera_frame = config_.mapping_camera_frame;
  mapping_camera.rgb_topic = config_.rgb_topic;
  mapping_camera.robot_mask_topic = config_.mask_topic;
  mapping_camera.depth_topic = config_.depth_topic;
  mapping_camera.camera_info_topic = config_.camera_info_topic;
  mapping_camera.fallback_intrinsics.width = config_.camera_width;
  mapping_camera.fallback_intrinsics.height = config_.camera_height;
  mapping_camera.fallback_intrinsics.fx = config_.camera_fx;
  mapping_camera.fallback_intrinsics.fy = config_.camera_fy;
  mapping_camera.fallback_intrinsics.cx = config_.camera_cx;
  mapping_camera.fallback_intrinsics.cy = config_.camera_cy;
  mapping_camera.depth_scale = config_.depth_scale;
  mapping_camera.depth_min_m = config_.depth_min_m;
  mapping_camera.depth_max_m = config_.depth_max_m;
  mapping_camera.mask_robot_threshold = config_.mask_robot_threshold;
  mapping_camera.enable_mapping = true;
  mapping_camera.enable_detection = config_.detection_enabled;

  RosIoSubscriptionConfig ros_io_config;
  ros_io_config.world_frame = config_.world_frame;
  ros_io_config.tf_topic = config_.tf_topic;
  ros_io_config.tf_static_topic = config_.tf_static_topic;
  ros_io_config.max_image_stamp_delta_sec = config_.max_image_stamp_delta_sec;
  ros_io_config.tf_buffer_duration_sec = config_.tf_buffer_duration_sec;
  ros_io_config.max_tf_gap_sec = config_.max_tf_gap_sec;
  ros_io_config.log_period_sec = config_.file_logging_period_sec;
  ros_io_config.input_queue_size = config_.input_queue_size;
  ros_io_config.cameras.push_back(std::move(mapping_camera));
  ros_io_thread_.configure(std::move(ros_io_config));
  ros_io_thread_.attachNode(node);
}

RoomiePipeline::~RoomiePipeline() { stop(); }

void RoomiePipeline::start() {
  if (started_) {
    return;
  }
  RunLogger::logGlobal("pipeline", "start");
  if (config_.detection_enabled) {
    python_backend_.start();
    detection_bridge_thread_.start();
  }
  map_thread_.start();
  instance_map_thread_.start();
  publisher_persistence_thread_.start();
  ros_io_thread_.start();
  started_ = true;
}

void RoomiePipeline::stop() {
  if (!started_) {
    return;
  }

  RunLogger::logGlobal("pipeline", "stop requested");
  ros_io_thread_.stop();
  detection_queue_.stop();
  mapping_queue_.stop();
  inference_response_queue_.stop();
  if (config_.detection_enabled) {
    detection_bridge_thread_.stop();
  }
  publisher_persistence_thread_.stop();
  instance_map_thread_.stop();
  map_thread_.stop();
  if (config_.detection_enabled) {
    python_backend_.stop();
  }
  RunLogger::logGlobal("pipeline", "stopped");
  started_ = false;
}

}  // namespace roomie
