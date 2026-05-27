#pragma once

#include <map>
#include <string>
#include <vector>

#include "roomie/pipeline/types.hpp"

namespace roomie {

enum class InstanceTrackState {
  kTentative,
  kStable,
  kInactive,
};

struct InstanceObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TimeNanoseconds time_ns = 0;
  std::string camera_id;
  RawDetection detection;
  float confidence = 0.0f;
  float bbox_quality = 0.0f;
  float camera_distance_m = 0.0f;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> near_surface_voxels;
  bool rejected = false;
  std::string reject_reason;
};

struct ObservationQualitySample {
  TimeNanoseconds time_ns = 0;
  float quality = 0.0f;
  float camera_distance_m = 0.0f;
  bool high_quality = false;
};

struct InstanceTrack {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int track_id = -1;
  int object_id = -1;
  InstanceTrackState state = InstanceTrackState::kTentative;
  int semantic_id = -1;
  std::string label;
  Eigen::Vector3f center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f size_m = Eigen::Vector3f::Zero();
  float yaw_rad = 0.0f;
  float confidence = 0.0f;
  float confidence_mass = 0.0f;
  float object_quality_score = 0.0f;
  float bbox_quality_mass = 0.0f;
  float geometry_score = 0.0f;
  float geometry_shell_ratio = 0.0f;
  float geometry_extent_score = 0.0f;
  float geometry_leak_ratio = 1.0f;
  float geometry_cavity_ratio = 0.0f;
  int geometry_in_box_points = 0;
  int geometry_shell_points = 0;
  int geometry_unique_voxels = 0;
  int geometry_expanded_points = 0;
  int geometry_bad_count = 0;
  int support_count = 0;
  int high_quality_observation_count = 0;
  float high_quality_observation_mass = 0.0f;
  int missed_count = 0;
  bool publishable = true;
  InstanceGeometryStatus geometry_status = InstanceGeometryStatus::kUnchecked;
  std::uint64_t first_seen_frame_index = 0;
  std::uint64_t last_seen_frame_index = 0;
  std::uint64_t obb_revision = 0;
  std::uint64_t geometry_evaluation_obb_revision = 0;
  std::uint64_t geometry_evaluation_map_version = 0;
  TimeNanoseconds last_geometry_check_ns = 0;
  TimeNanoseconds first_seen_ns = 0;
  TimeNanoseconds last_seen_ns = 0;
  Eigen::Vector3f geometry_evaluated_center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f geometry_evaluated_size_m = Eigen::Vector3f::Zero();
  float geometry_evaluated_yaw_rad = 0.0f;
  std::string geometry_evaluation_reason;
  std::vector<std::string> source_cameras;
  std::vector<TimeNanoseconds> observation_timestamps_ns;
  std::vector<ObservationQualitySample> observation_quality_history;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> near_surface_voxels;
  std::map<std::string, float> label_weights;
  std::map<int, float> semantic_weights;
};

struct ObjectRelation {
  int source_object_id = -1;
  int target_object_id = -1;
  std::string relation_type;
  float confidence = 0.0f;
  std::string description;
};

struct ObjectNode {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int object_id = -1;
  int semantic_id = -1;
  std::string label;
  std::string description;
  Eigen::Vector3f center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f size_m = Eigen::Vector3f::Zero();
  float yaw_rad = 0.0f;
  float confidence = 0.0f;
  float confidence_mass = 0.0f;
  float object_quality_score = 0.0f;
  float geometry_score = 0.0f;
  float geometry_shell_ratio = 0.0f;
  float geometry_extent_score = 0.0f;
  float geometry_leak_ratio = 1.0f;
  float geometry_cavity_ratio = 0.0f;
  int geometry_in_box_points = 0;
  int geometry_shell_points = 0;
  int geometry_unique_voxels = 0;
  int geometry_expanded_points = 0;
  int geometry_bad_count = 0;
  int support_count = 0;
  int high_quality_observation_count = 0;
  float high_quality_observation_mass = 0.0f;
  bool active = true;
  bool publishable = true;
  InstanceGeometryStatus geometry_status = InstanceGeometryStatus::kUnchecked;
  std::uint64_t geometry_evaluation_obb_revision = 0;
  std::uint64_t geometry_evaluation_map_version = 0;
  TimeNanoseconds first_seen_ns = 0;
  TimeNanoseconds last_seen_ns = 0;
  TimeNanoseconds last_geometry_check_ns = 0;
  Eigen::Vector3f geometry_evaluated_center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f geometry_evaluated_size_m = Eigen::Vector3f::Zero();
  float geometry_evaluated_yaw_rad = 0.0f;
  std::string geometry_evaluation_reason;
  std::vector<int> source_track_ids;
  std::vector<std::string> source_cameras;
  std::vector<TimeNanoseconds> observation_timestamps_ns;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> near_surface_voxels;
  std::map<std::string, float> label_weights;
  std::map<int, float> semantic_weights;
};

struct ObjectGraphSnapshot {
  int next_object_id = 0;
  std::vector<ObjectNode, Eigen::aligned_allocator<ObjectNode>> objects;
  std::vector<ObjectRelation> relations;
};

class ObjectGraph {
 public:
  int createNodeFromTrack(const InstanceTrack& track);
  void updateNodeFromTrack(const InstanceTrack& track);
  bool removeNode(int object_id);
  void loadSnapshot(const ObjectGraphSnapshot& snapshot);
  ObjectGraphSnapshot snapshot() const;
  std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
  snapshotInstanceRecords(bool publishable_only = true) const;
  std::size_t objectCount() const;
  std::size_t relationCount() const;

 private:
  ObjectNode* findNode(int object_id);
  const ObjectNode* findNode(int object_id) const;
  static ObjectNode nodeFromTrack(const InstanceTrack& track, int object_id);
  static InstanceRecord recordFromNode(const ObjectNode& node);

  int next_object_id_ = 0;
  std::vector<ObjectNode, Eigen::aligned_allocator<ObjectNode>> objects_;
  std::vector<ObjectRelation> relations_;
};

}  // namespace roomie
