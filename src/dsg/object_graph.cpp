#include "roomie/dsg/object_graph.hpp"

#include <algorithm>

namespace roomie {
namespace {

template <typename T>
void appendUnique(std::vector<T>* values, const T& value) {
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(value);
  }
}

template <typename T>
void appendUniqueVector(std::vector<T>* values, const std::vector<T>& incoming) {
  for (const T& value : incoming) {
    appendUnique(values, value);
  }
}

}  // namespace

int ObjectGraph::createNodeFromTrack(const InstanceTrack& track) {
  const int object_id = next_object_id_++;
  objects_.push_back(nodeFromTrack(track, object_id));
  return object_id;
}

void ObjectGraph::updateNodeFromTrack(const InstanceTrack& track) {
  if (track.object_id < 0) {
    return;
  }

  ObjectNode* node = findNode(track.object_id);
  if (node == nullptr) {
    objects_.push_back(nodeFromTrack(track, track.object_id));
    next_object_id_ = std::max(next_object_id_, track.object_id + 1);
    return;
  }

  node->semantic_id = track.semantic_id;
  node->label = track.label;
  node->center_world = track.center_world;
  node->size_m = track.size_m;
  node->yaw_rad = track.yaw_rad;
  node->confidence = track.confidence;
  node->confidence_mass = track.confidence_mass;
  node->object_quality_score = track.object_quality_score;
  node->geometry_score = track.geometry_score;
  node->geometry_shell_ratio = track.geometry_shell_ratio;
  node->geometry_extent_score = track.geometry_extent_score;
  node->geometry_leak_ratio = track.geometry_leak_ratio;
  node->geometry_cavity_ratio = track.geometry_cavity_ratio;
  node->geometry_in_box_points = track.geometry_in_box_points;
  node->geometry_shell_points = track.geometry_shell_points;
  node->geometry_unique_voxels = track.geometry_unique_voxels;
  node->geometry_expanded_points = track.geometry_expanded_points;
  node->geometry_bad_count = track.geometry_bad_count;
  node->support_count = track.support_count;
  node->high_quality_observation_count = track.high_quality_observation_count;
  node->high_quality_observation_mass = track.high_quality_observation_mass;
  node->active = track.state != InstanceTrackState::kInactive;
  node->publishable = track.publishable;
  node->geometry_status = track.geometry_status;
  node->geometry_evaluation_obb_revision = track.geometry_evaluation_obb_revision;
  node->geometry_evaluation_map_version = track.geometry_evaluation_map_version;
  node->first_seen_ns = track.first_seen_ns;
  node->last_seen_ns = track.last_seen_ns;
  node->last_geometry_check_ns = track.last_geometry_check_ns;
  node->geometry_evaluated_center_world = track.geometry_evaluated_center_world;
  node->geometry_evaluated_size_m = track.geometry_evaluated_size_m;
  node->geometry_evaluated_yaw_rad = track.geometry_evaluated_yaw_rad;
  node->geometry_evaluation_reason = track.geometry_evaluation_reason;
  appendUnique(&node->source_track_ids, track.track_id);
  appendUniqueVector(&node->source_cameras, track.source_cameras);
  // The track carries the configured recent-history window. Assigning it is
  // important: unioning successive sliding windows would regrow the object
  // history without bound.
  node->observation_timestamps_ns = track.observation_timestamps_ns;
  if (track.snapshot.valid() &&
      (!node->snapshot.valid() || track.snapshot.quality >= node->snapshot.quality)) {
    node->snapshot = track.snapshot;
  }
  node->near_surface_voxels = track.near_surface_voxels;
  node->label_weights = track.label_weights;
  node->semantic_weights = track.semantic_weights;
}

bool ObjectGraph::removeNode(int object_id) {
  const auto before = objects_.size();
  objects_.erase(std::remove_if(objects_.begin(),
                                objects_.end(),
                                [object_id](const ObjectNode& node) {
                                  return node.object_id == object_id;
                                }),
                 objects_.end());
  relations_.erase(std::remove_if(relations_.begin(),
                                  relations_.end(),
                                  [object_id](const ObjectRelation& relation) {
                                    const SceneEntityRef endpoint{
                                        SceneEntityType::kObject, object_id};
                                    return relationSource(relation) == endpoint ||
                                           relationTarget(relation) == endpoint;
                                  }),
                   relations_.end());
  return objects_.size() != before;
}

void ObjectGraph::loadSnapshot(const ObjectGraphSnapshot& snapshot) {
  objects_ = snapshot.objects;
  rooms_ = snapshot.rooms;
  relations_ = snapshot.relations;
  snapshot_images_ = snapshot.snapshot_images;
  import_warnings_ = snapshot.import_warnings;
  has_scene_graph_envelope_ = snapshot.has_scene_graph_envelope;
  scene_graph_json_ = snapshot.scene_graph_json;
  next_object_id_ = snapshot.next_object_id;
  for (const ObjectNode& object : objects_) {
    next_object_id_ = std::max(next_object_id_, object.object_id + 1);
  }
}

ObjectGraphSnapshot ObjectGraph::snapshot() const {
  ObjectGraphSnapshot snapshot;
  snapshot.next_object_id = next_object_id_;
  snapshot.objects = objects_;
  snapshot.rooms = rooms_;
  snapshot.relations = relations_;
  snapshot.snapshot_images = snapshot_images_;
  snapshot.import_warnings = import_warnings_;
  snapshot.has_scene_graph_envelope = has_scene_graph_envelope_;
  snapshot.scene_graph_json = scene_graph_json_;
  return snapshot;
}

std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
ObjectGraph::snapshotInstanceRecords(bool publishable_only) const {
  std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>> records;
  records.reserve(objects_.size());
  for (const ObjectNode& node : objects_) {
    if (publishable_only && (!node.publishable || !node.active)) {
      continue;
    }
    records.push_back(recordFromNode(node));
  }
  return records;
}

std::size_t ObjectGraph::objectCount() const { return objects_.size(); }

std::size_t ObjectGraph::relationCount() const { return relations_.size(); }

ObjectNode* ObjectGraph::findNode(int object_id) {
  auto it = std::find_if(objects_.begin(), objects_.end(), [object_id](const ObjectNode& node) {
    return node.object_id == object_id;
  });
  return it == objects_.end() ? nullptr : &(*it);
}

const ObjectNode* ObjectGraph::findNode(int object_id) const {
  auto it = std::find_if(objects_.begin(), objects_.end(), [object_id](const ObjectNode& node) {
    return node.object_id == object_id;
  });
  return it == objects_.end() ? nullptr : &(*it);
}

ObjectNode ObjectGraph::nodeFromTrack(const InstanceTrack& track, int object_id) {
  ObjectNode node;
  node.object_id = object_id;
  node.semantic_id = track.semantic_id;
  node.label = track.label;
  node.center_world = track.center_world;
  node.size_m = track.size_m;
  node.yaw_rad = track.yaw_rad;
  node.confidence = track.confidence;
  node.confidence_mass = track.confidence_mass;
  node.object_quality_score = track.object_quality_score;
  node.geometry_score = track.geometry_score;
  node.geometry_shell_ratio = track.geometry_shell_ratio;
  node.geometry_extent_score = track.geometry_extent_score;
  node.geometry_leak_ratio = track.geometry_leak_ratio;
  node.geometry_cavity_ratio = track.geometry_cavity_ratio;
  node.geometry_in_box_points = track.geometry_in_box_points;
  node.geometry_shell_points = track.geometry_shell_points;
  node.geometry_unique_voxels = track.geometry_unique_voxels;
  node.geometry_expanded_points = track.geometry_expanded_points;
  node.geometry_bad_count = track.geometry_bad_count;
  node.support_count = track.support_count;
  node.high_quality_observation_count = track.high_quality_observation_count;
  node.high_quality_observation_mass = track.high_quality_observation_mass;
  node.active = track.state != InstanceTrackState::kInactive;
  node.publishable = track.publishable;
  node.geometry_status = track.geometry_status;
  node.geometry_evaluation_obb_revision = track.geometry_evaluation_obb_revision;
  node.geometry_evaluation_map_version = track.geometry_evaluation_map_version;
  node.first_seen_ns = track.first_seen_ns;
  node.last_seen_ns = track.last_seen_ns;
  node.last_geometry_check_ns = track.last_geometry_check_ns;
  node.geometry_evaluated_center_world = track.geometry_evaluated_center_world;
  node.geometry_evaluated_size_m = track.geometry_evaluated_size_m;
  node.geometry_evaluated_yaw_rad = track.geometry_evaluated_yaw_rad;
  node.geometry_evaluation_reason = track.geometry_evaluation_reason;
  node.source_track_ids.push_back(track.track_id);
  node.source_cameras = track.source_cameras;
  node.observation_timestamps_ns = track.observation_timestamps_ns;
  node.snapshot = track.snapshot;
  node.near_surface_voxels = track.near_surface_voxels;
  node.label_weights = track.label_weights;
  node.semantic_weights = track.semantic_weights;
  return node;
}

InstanceRecord ObjectGraph::recordFromNode(const ObjectNode& node) {
  InstanceRecord record;
  record.object_id = node.object_id;
  if (!node.source_track_ids.empty()) {
    record.track_id = node.source_track_ids.back();
  }
  record.semantic_id = node.semantic_id;
  record.label = node.label;
  record.description = node.description;
  record.center_world = node.center_world;
  record.size_m = node.size_m;
  record.yaw_rad = node.yaw_rad;
  record.confidence = node.confidence;
  record.confidence_mass = node.confidence_mass;
  record.object_quality_score = node.object_quality_score;
  record.geometry_score = node.geometry_score;
  record.geometry_shell_ratio = node.geometry_shell_ratio;
  record.geometry_extent_score = node.geometry_extent_score;
  record.geometry_leak_ratio = node.geometry_leak_ratio;
  record.geometry_cavity_ratio = node.geometry_cavity_ratio;
  record.geometry_in_box_points = node.geometry_in_box_points;
  record.geometry_shell_points = node.geometry_shell_points;
  record.geometry_unique_voxels = node.geometry_unique_voxels;
  record.geometry_expanded_points = node.geometry_expanded_points;
  record.geometry_bad_count = node.geometry_bad_count;
  record.support_count = node.support_count;
  record.high_quality_observation_count = node.high_quality_observation_count;
  record.high_quality_observation_mass = node.high_quality_observation_mass;
  record.active = node.active;
  record.publishable = node.publishable;
  record.geometry_status = node.geometry_status;
  record.last_geometry_check_ns = node.last_geometry_check_ns;
  record.first_seen_ns = node.first_seen_ns;
  record.last_seen_ns = node.last_seen_ns;
  record.source_cameras = node.source_cameras;
  record.source_track_ids = node.source_track_ids;
  record.observation_timestamps_ns = node.observation_timestamps_ns;
  record.snapshot_image_index = node.snapshot.image_index;
  record.snapshot_bbox_xyxy = node.snapshot.bbox_xyxy;
  record.snapshot_quality = node.snapshot.quality;
  record.near_surface_voxels = node.near_surface_voxels;
  return record;
}

}  // namespace roomie
