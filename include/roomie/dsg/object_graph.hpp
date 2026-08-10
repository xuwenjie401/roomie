#pragma once

#include <array>
#include <cstdint>
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

struct ObjectSnapshotRef {
  int image_index = -1;
  // Online snapshots address the durable full-frame PNG directly. A legacy
  // image_index may coexist for v1/v2 JSON viewers.
  std::string source_frame_asset_id;
  std::string evidence_hash;
  std::array<float, 4> bbox_xyxy = {0.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 4> crop_xywh = {0.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 2> crop_output_scale = {1.0f, 1.0f};
  std::string mask_source = "bbox_fallback";
  std::string mask_ref;
  float quality = 0.0f;
  std::map<std::string, float> quality_components;
  float viewpoint_azimuth_rad = 0.0f;
  float viewpoint_elevation_rad = 0.0f;
  float viewpoint_scale = 1.0f;
  TimeNanoseconds time_ns = 0;
  std::string camera_id;
  FrameProvenance provenance;

  bool valid() const {
    return image_index >= 0 || !source_frame_asset_id.empty();
  }
};

struct ObjectSnapshotImage {
  int image_index = -1;
  std::string uri;
  int width = 0;
  int height = 0;
  std::string encoding;
  TimeNanoseconds time_ns = 0;
  std::string camera_id;
  std::string source_path;
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
  ObjectSnapshotRef snapshot;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> near_surface_voxels;
  std::map<std::string, float> label_weights;
  std::map<int, float> semantic_weights;
};

enum class SceneEntityType {
  kObject,
  kRoom,
  // Furniture is a typed projection of the canonical object with the same id;
  // it is not a second copy of object geometry or semantics.
  kFurniture,
};

struct SceneEntityRef {
  SceneEntityType type = SceneEntityType::kObject;
  int id = -1;

  bool valid() const { return id >= 0; }
  bool operator==(const SceneEntityRef& other) const {
    return type == other.type && id == other.id;
  }
  bool operator!=(const SceneEntityRef& other) const {
    return !(*this == other);
  }
};

struct SceneRelation {
  // Legacy object-object endpoints. They remain populated for v1/v2 readers,
  // while source/target below are authoritative for schema v3+ and can also
  // address rooms and furniture.
  int source_object_id = -1;
  int target_object_id = -1;
  std::string relation_type;
  float confidence = 0.0f;
  std::string description;
  SceneEntityRef source;
  SceneEntityRef target;
  std::uint64_t revision = 0;
  bool derived = false;
};

// Compatibility name retained for the existing C++ call sites. New code uses
// SceneRelation because endpoints can address rooms and furniture as well.
using ObjectRelation = SceneRelation;

inline SceneEntityRef relationSource(const SceneRelation& relation) {
  if (relation.source.valid()) {
    return relation.source;
  }
  return SceneEntityRef{SceneEntityType::kObject,
                        relation.source_object_id};
}

inline SceneEntityRef relationTarget(const SceneRelation& relation) {
  if (relation.target.valid()) {
    return relation.target;
  }
  return SceneEntityRef{SceneEntityType::kObject,
                        relation.target_object_id};
}

inline void setRelationEndpoints(SceneRelation* relation,
                                 SceneEntityRef source,
                                 SceneEntityRef target) {
  if (relation == nullptr) {
    return;
  }
  relation->source = source;
  relation->target = target;
  // Furniture ids are canonical object ids, so legacy object-only readers can
  // still identify the physical endpoints even though they cannot represent
  // the furniture role itself.
  relation->source_object_id = source.type == SceneEntityType::kRoom
                                   ? -1
                                   : source.id;
  relation->target_object_id = target.type == SceneEntityType::kRoom
                                   ? -1
                                   : target.id;
}

inline bool entityBackedByObject(SceneEntityRef entity) {
  return entity.type == SceneEntityType::kObject ||
         entity.type == SceneEntityType::kFurniture;
}

struct FurnitureRole {
  // object_id is also the id used by kFurniture relation endpoints.
  int object_id = -1;
  std::uint64_t revision = 0;
  // The normalized class that selected this role. It remains stable until the
  // next explicit furniture rebuild, even if the object's label later changes.
  std::string classification_label;
};

struct RoomNode {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int room_id = -1;
  std::uint64_t revision = 0;
  std::string label;
  std::string color;
  Eigen::Vector3f center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f size_m = Eigen::Vector3f::Zero();
  std::array<float, 2> min_xy = {0.0f, 0.0f};
  std::array<float, 2> max_xy = {0.0f, 0.0f};
  bool has_xy_bounds = false;
  float height_m = 0.0f;
  std::map<std::string, std::string> attributes;
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
  ObjectSnapshotRef snapshot;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> near_surface_voxels;
  std::map<std::string, float> label_weights;
  std::map<int, float> semantic_weights;
};

struct ObjectGraphSnapshot {
  int schema_version = 4;
  int next_object_id = 0;
  std::vector<ObjectNode, Eigen::aligned_allocator<ObjectNode>> objects;
  std::vector<RoomNode, Eigen::aligned_allocator<RoomNode>> rooms;
  std::vector<FurnitureRole> furniture;
  std::vector<SceneRelation> relations;
  std::vector<ObjectSnapshotImage> snapshot_images;
  std::vector<std::string> import_warnings;
  bool has_scene_graph_envelope = false;
  std::string scene_graph_json;
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
  std::vector<RoomNode, Eigen::aligned_allocator<RoomNode>> rooms_;
  std::vector<FurnitureRole> furniture_;
  std::vector<SceneRelation> relations_;
  std::vector<ObjectSnapshotImage> snapshot_images_;
  std::vector<std::string> import_warnings_;
  bool has_scene_graph_envelope_ = false;
  std::string scene_graph_json_;
};

}  // namespace roomie
