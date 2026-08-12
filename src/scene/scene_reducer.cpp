#include "roomie/scene/scene_reducer.hpp"

#include "roomie/artifacts/semantic_index.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace roomie {
namespace {

std::uint64_t nextComponentRevision(std::uint64_t current) {
  if (current == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("component revision exhausted");
  }
  return current + 1;
}

SceneRevision nextSceneRevision(SceneRevision current) {
  if (current == std::numeric_limits<SceneRevision>::max()) {
    throw std::overflow_error("scene revision exhausted");
  }
  return current + 1;
}

SceneObjectId checkedNextObjectId(SceneObjectId object_id) {
  if (object_id < 0 || object_id == std::numeric_limits<SceneObjectId>::max()) {
    throw std::overflow_error("object id space exhausted");
  }
  return object_id + 1;
}

std::optional<SceneObjectId> resolveCanonical(const SceneAliasTable& aliases,
                                              SceneObjectId object_id) {
  std::set<SceneObjectId> visited;
  SceneObjectId current = object_id;
  while (true) {
    if (!visited.insert(current).second) {
      return std::nullopt;
    }
    const auto it = aliases.find(current);
    if (it == aliases.end()) {
      return current;
    }
    current = it->second.canonical_object_id;
  }
}

bool vectorsApprox(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
  return lhs.isApprox(rhs, 1.0e-6f);
}

bool snapshotRefEqual(const ObjectSnapshotRef& lhs,
                      const ObjectSnapshotRef& rhs) {
  return lhs.image_index == rhs.image_index && lhs.bbox_xyxy == rhs.bbox_xyxy &&
         lhs.source_frame_asset_id == rhs.source_frame_asset_id &&
         lhs.evidence_hash == rhs.evidence_hash &&
         lhs.crop_xywh == rhs.crop_xywh &&
         lhs.crop_output_scale == rhs.crop_output_scale &&
         lhs.mask_source == rhs.mask_source && lhs.mask_ref == rhs.mask_ref &&
         lhs.quality == rhs.quality && lhs.time_ns == rhs.time_ns &&
         lhs.quality_components == rhs.quality_components &&
         lhs.viewpoint_azimuth_rad == rhs.viewpoint_azimuth_rad &&
         lhs.viewpoint_elevation_rad == rhs.viewpoint_elevation_rad &&
         lhs.viewpoint_scale == rhs.viewpoint_scale &&
         lhs.camera_id == rhs.camera_id && lhs.provenance == rhs.provenance;
}

bool snapshotSetsEqual(const std::vector<ObjectSnapshotRef>& lhs,
                       const std::vector<ObjectSnapshotRef>& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), snapshotRefEqual);
}

void clearPendingDescription(ArtifactComponent* artifact) {
  artifact->pending_description_scene_revision = 0;
  artifact->pending_description.clear();
  artifact->pending_description_input_hash.clear();
  artifact->pending_description_model_id.clear();
  artifact->pending_description_schema_version.clear();
  artifact->pending_description_raw_text.clear();
  artifact->pending_description_normalized_json.clear();
  artifact->pending_description_durable_envelope_json.clear();
  artifact->pending_description_parse_path.clear();
  artifact->pending_description_slo = {};
}

bool invalidateDescription(ArtifactComponent* artifact) {
  const bool had_description = !artifact->description.empty();
  const bool had_pending = artifact->pending_description_scene_revision != 0;
  artifact->description_stale = had_description;
  clearPendingDescription(artifact);
  return had_description || had_pending;
}

bool promoteDurableDescription(ArtifactComponent* artifact,
                               SceneRevision durable_revision) {
  if (artifact->pending_description_scene_revision == 0 ||
      artifact->pending_description_scene_revision > durable_revision) {
    return false;
  }
  artifact->revision = nextComponentRevision(artifact->revision);
  artifact->description = std::move(artifact->pending_description);
  artifact->description_input_hash =
      std::move(artifact->pending_description_input_hash);
  artifact->description_model_id =
      std::move(artifact->pending_description_model_id);
  artifact->description_schema_version =
      std::move(artifact->pending_description_schema_version);
  artifact->description_raw_text =
      std::move(artifact->pending_description_raw_text);
  artifact->description_normalized_json =
      std::move(artifact->pending_description_normalized_json);
  artifact->description_durable_envelope_json =
      std::move(artifact->pending_description_durable_envelope_json);
  artifact->description_parse_path =
      std::move(artifact->pending_description_parse_path);
  artifact->description_slo = artifact->pending_description_slo;
  artifact->description_stale = false;
  clearPendingDescription(artifact);
  return true;
}

std::shared_ptr<const IdentityComponent> identityFromNode(
    const ObjectNode& node) {
  auto component = std::make_shared<IdentityComponent>();
  component->revision = 1;
  component->object_id = node.object_id;
  component->source_track_ids = node.source_track_ids;
  return component;
}

std::shared_ptr<const LifecycleComponent> lifecycleFromNode(
    const ObjectNode& node) {
  auto component = std::make_shared<LifecycleComponent>();
  component->revision = 1;
  component->track_state =
      node.active ? InstanceTrackState::kStable : InstanceTrackState::kInactive;
  component->active = node.active;
  component->publishable = node.publishable;
  component->existence_log_odds = node.existence_log_odds;
  component->last_presence_evidence_ns = node.last_presence_evidence_ns;
  component->last_presence_evidence_reason =
      node.last_presence_evidence_reason;
  component->first_seen_ns = node.first_seen_ns;
  component->last_seen_ns = node.last_seen_ns;
  return component;
}

std::shared_ptr<const GeometryComponent> geometryFromNode(
    const ObjectNode& node) {
  auto component = std::make_shared<GeometryComponent>();
  component->revision = 1;
  component->obb_revision = std::max<std::uint64_t>(
      1, node.geometry_evaluation_obb_revision);
  component->evaluated_obb_revision =
      node.geometry_evaluation_obb_revision;
  component->center_world = node.center_world;
  component->size_m = node.size_m;
  component->yaw_rad = node.yaw_rad;
  component->status = node.geometry_status;
  component->score = node.geometry_score;
  component->shell_ratio = node.geometry_shell_ratio;
  component->extent_score = node.geometry_extent_score;
  component->leak_ratio = node.geometry_leak_ratio;
  component->cavity_ratio = node.geometry_cavity_ratio;
  component->in_box_points = node.geometry_in_box_points;
  component->shell_points = node.geometry_shell_points;
  component->unique_voxels = node.geometry_unique_voxels;
  component->expanded_points = node.geometry_expanded_points;
  component->bad_count = node.geometry_bad_count;
  component->last_check_ns = node.last_geometry_check_ns;
  component->evaluated_center_world = node.geometry_evaluated_center_world;
  component->evaluated_size_m = node.geometry_evaluated_size_m;
  component->evaluated_yaw_rad = node.geometry_evaluated_yaw_rad;
  component->evaluation_reason = node.geometry_evaluation_reason;
  return component;
}

std::shared_ptr<const SemanticComponent> semanticFromNode(
    const ObjectNode& node) {
  auto component = std::make_shared<SemanticComponent>();
  component->revision = 1;
  component->semantic_id = node.semantic_id;
  component->label = node.label;
  component->confidence = node.confidence;
  component->confidence_mass = node.confidence_mass;
  component->object_quality_score = node.object_quality_score;
  component->support_count = node.support_count;
  component->high_quality_observation_count =
      node.high_quality_observation_count;
  component->high_quality_observation_mass =
      node.high_quality_observation_mass;
  component->source_cameras = node.source_cameras;
  component->observation_timestamps_ns = node.observation_timestamps_ns;
  component->label_weights = node.label_weights;
  component->semantic_weights = node.semantic_weights;
  return component;
}

SceneObjectPtr objectFromNode(const ObjectNode& node) {
  auto object = std::make_shared<SceneObject>();
  object->identity = identityFromNode(node);
  object->lifecycle = lifecycleFromNode(node);
  object->geometry = geometryFromNode(node);
  object->semantic = semanticFromNode(node);
  auto annotation = std::make_shared<AnnotationComponent>();
  annotation->revision = 1;
  annotation->name = node.name;
  object->annotation = std::move(annotation);
  auto artifact = std::make_shared<ArtifactComponent>();
  artifact->revision = 1;
  if (node.snapshot.valid()) {
    artifact->appearance_revision = 1;
    artifact->snapshots.push_back(node.snapshot);
  }
  artifact->description = node.description;
  object->artifact = std::move(artifact);
  return object;
}

SceneObjectPtr objectFromTrack(const InstanceTrack& track,
                               SceneObjectId object_id) {
  auto object = std::make_shared<SceneObject>();

  auto identity = std::make_shared<IdentityComponent>();
  identity->revision = 1;
  identity->object_id = object_id;
  if (track.track_id >= 0) {
    identity->source_track_ids.push_back(track.track_id);
  }
  object->identity = std::move(identity);

  auto lifecycle = std::make_shared<LifecycleComponent>();
  lifecycle->revision = 1;
  lifecycle->track_state = track.state;
  lifecycle->active = track.state != InstanceTrackState::kInactive;
  lifecycle->publishable = track.publishable;
  lifecycle->existence_log_odds = track.existence_log_odds;
  lifecycle->last_presence_evidence_ns = track.last_presence_evidence_ns;
  lifecycle->last_presence_evidence_reliability =
      track.last_presence_evidence_reliability;
  lifecycle->last_presence_evidence_reason =
      track.last_presence_evidence_reason;
  lifecycle->positive_evidence_timestamps_ns =
      track.positive_evidence_timestamps_ns;
  lifecycle->positive_presence_evidence_history =
      track.positive_presence_evidence_history;
  lifecycle->negative_evidence_timestamps_ns =
      track.negative_evidence_timestamps_ns;
  lifecycle->positive_window_interruptions =
      track.positive_window_interruptions;
  lifecycle->first_seen_ns = track.first_seen_ns;
  lifecycle->last_seen_ns = track.last_seen_ns;
  lifecycle->first_seen_frame_index = track.first_seen_frame_index;
  lifecycle->last_seen_frame_index = track.last_seen_frame_index;
  object->lifecycle = std::move(lifecycle);

  auto geometry = std::make_shared<GeometryComponent>();
  geometry->revision = 1;
  geometry->obb_revision = std::max<std::uint64_t>(1, track.obb_revision);
  geometry->evaluated_obb_revision =
      track.geometry_evaluation_obb_revision;
  geometry->center_world = track.center_world;
  geometry->size_m = track.size_m;
  geometry->yaw_rad = track.yaw_rad;
  geometry->status = track.geometry_status;
  geometry->score = track.geometry_score;
  geometry->shell_ratio = track.geometry_shell_ratio;
  geometry->extent_score = track.geometry_extent_score;
  geometry->leak_ratio = track.geometry_leak_ratio;
  geometry->cavity_ratio = track.geometry_cavity_ratio;
  geometry->in_box_points = track.geometry_in_box_points;
  geometry->shell_points = track.geometry_shell_points;
  geometry->unique_voxels = track.geometry_unique_voxels;
  geometry->expanded_points = track.geometry_expanded_points;
  geometry->bad_count = track.geometry_bad_count;
  geometry->last_check_ns = track.last_geometry_check_ns;
  geometry->evaluated_center_world = track.geometry_evaluated_center_world;
  geometry->evaluated_size_m = track.geometry_evaluated_size_m;
  geometry->evaluated_yaw_rad = track.geometry_evaluated_yaw_rad;
  geometry->evaluation_reason = track.geometry_evaluation_reason;
  object->geometry = std::move(geometry);

  auto semantic = std::make_shared<SemanticComponent>();
  semantic->revision = 1;
  semantic->semantic_id = track.semantic_id;
  semantic->label = track.label;
  semantic->confidence = track.confidence;
  semantic->confidence_mass = track.confidence_mass;
  semantic->object_quality_score = track.object_quality_score;
  semantic->support_count = track.support_count;
  semantic->high_quality_observation_count =
      track.high_quality_observation_count;
  semantic->high_quality_observation_mass =
      track.high_quality_observation_mass;
  semantic->source_cameras = track.source_cameras;
  semantic->observation_timestamps_ns = track.observation_timestamps_ns;
  semantic->label_weights = track.label_weights;
  semantic->semantic_weights = track.semantic_weights;
  object->semantic = std::move(semantic);

  auto annotation = std::make_shared<AnnotationComponent>();
  annotation->revision = 1;
  object->annotation = std::move(annotation);

  auto artifact = std::make_shared<ArtifactComponent>();
  artifact->revision = 1;
  if (track.snapshot.valid()) {
    artifact->appearance_revision = 1;
    artifact->snapshots.push_back(track.snapshot);
  }
  object->artifact = std::move(artifact);
  return object;
}

bool addTrackToIdentity(SceneObject* object, int track_id) {
  if (track_id < 0 || !object->identity) {
    return false;
  }
  auto tracks = object->identity->source_track_ids;
  if (std::find(tracks.begin(), tracks.end(), track_id) != tracks.end()) {
    return false;
  }
  tracks.push_back(track_id);
  std::sort(tracks.begin(), tracks.end());
  tracks.erase(std::unique(tracks.begin(), tracks.end()), tracks.end());
  auto identity = std::make_shared<IdentityComponent>(*object->identity);
  identity->revision = nextComponentRevision(identity->revision);
  identity->source_track_ids = std::move(tracks);
  object->identity = std::move(identity);
  return true;
}

bool updateObjectFromTrack(const InstanceTrack& track,
                           SceneObjectId object_id,
                           const SceneObjectPtr& current,
                           SceneObjectPtr* updated,
                           bool* obb_changed) {
  auto next = std::make_shared<SceneObject>(*current);
  bool changed = false;
  *obb_changed = false;

  if (track.track_id >= 0 && next->identity) {
    changed |= addTrackToIdentity(next.get(), track.track_id);
  }

  if (next->lifecycle) {
    const bool active = track.state != InstanceTrackState::kInactive;
    const LifecycleComponent& old = *next->lifecycle;
    if (old.track_state != track.state || old.active != active ||
        old.publishable != track.publishable ||
        old.existence_log_odds != track.existence_log_odds ||
        old.last_presence_evidence_ns != track.last_presence_evidence_ns ||
        old.last_presence_evidence_reliability !=
            track.last_presence_evidence_reliability ||
        old.last_presence_evidence_reason !=
            track.last_presence_evidence_reason ||
        old.positive_evidence_timestamps_ns !=
            track.positive_evidence_timestamps_ns ||
        old.positive_presence_evidence_history !=
            track.positive_presence_evidence_history ||
        old.negative_evidence_timestamps_ns !=
            track.negative_evidence_timestamps_ns ||
        old.positive_window_interruptions !=
            track.positive_window_interruptions ||
        old.first_seen_ns != track.first_seen_ns ||
        old.last_seen_ns != track.last_seen_ns ||
        old.first_seen_frame_index != track.first_seen_frame_index ||
        old.last_seen_frame_index != track.last_seen_frame_index) {
      auto component = std::make_shared<LifecycleComponent>(old);
      component->revision = nextComponentRevision(old.revision);
      component->track_state = track.state;
      component->active = active;
      component->publishable = track.publishable;
      component->existence_log_odds = track.existence_log_odds;
      component->last_presence_evidence_ns =
          track.last_presence_evidence_ns;
      component->last_presence_evidence_reliability =
          track.last_presence_evidence_reliability;
      component->last_presence_evidence_reason =
          track.last_presence_evidence_reason;
      component->positive_evidence_timestamps_ns =
          track.positive_evidence_timestamps_ns;
      component->positive_presence_evidence_history =
          track.positive_presence_evidence_history;
      component->negative_evidence_timestamps_ns =
          track.negative_evidence_timestamps_ns;
      component->positive_window_interruptions =
          track.positive_window_interruptions;
      component->first_seen_ns = track.first_seen_ns;
      component->last_seen_ns = track.last_seen_ns;
      component->first_seen_frame_index = track.first_seen_frame_index;
      component->last_seen_frame_index = track.last_seen_frame_index;
      next->lifecycle = std::move(component);
      changed = true;
    }
  }

  if (next->geometry) {
    const GeometryComponent& old = *next->geometry;
    *obb_changed = !vectorsApprox(old.center_world, track.center_world) ||
                   !vectorsApprox(old.size_m, track.size_m) ||
                   std::fabs(old.yaw_rad - track.yaw_rad) > 1.0e-6f;
    // Validation is reducer-owned derived state.  Association may update the
    // OBB, but must never replay a stale validation copy from InstanceTrack.
    if (*obb_changed) {
      auto component = std::make_shared<GeometryComponent>(old);
      component->revision = nextComponentRevision(old.revision);
      component->obb_revision = std::max(
          nextComponentRevision(old.obb_revision), track.obb_revision);
      component->center_world = track.center_world;
      component->size_m = track.size_m;
      component->yaw_rad = track.yaw_rad;
      next->geometry = std::move(component);
      changed = true;
    }
  }

  if (next->semantic) {
    const SemanticComponent& old = *next->semantic;
    if (old.semantic_id != track.semantic_id || old.label != track.label ||
        old.confidence != track.confidence ||
        old.confidence_mass != track.confidence_mass ||
        old.object_quality_score != track.object_quality_score ||
        old.support_count != track.support_count ||
        old.high_quality_observation_count !=
            track.high_quality_observation_count ||
        old.high_quality_observation_mass !=
            track.high_quality_observation_mass ||
        old.source_cameras != track.source_cameras ||
        old.observation_timestamps_ns != track.observation_timestamps_ns ||
        old.label_weights != track.label_weights ||
        old.semantic_weights != track.semantic_weights) {
      auto component = std::make_shared<SemanticComponent>(old);
      component->revision = nextComponentRevision(old.revision);
      component->semantic_id = track.semantic_id;
      component->label = track.label;
      component->confidence = track.confidence;
      component->confidence_mass = track.confidence_mass;
      component->object_quality_score = track.object_quality_score;
      component->support_count = track.support_count;
      component->high_quality_observation_count =
          track.high_quality_observation_count;
      component->high_quality_observation_mass =
          track.high_quality_observation_mass;
      component->source_cameras = track.source_cameras;
      component->observation_timestamps_ns = track.observation_timestamps_ns;
      component->label_weights = track.label_weights;
      component->semantic_weights = track.semantic_weights;
      next->semantic = std::move(component);
      changed = true;
    }
  }

  if (next->identity && next->identity->object_id != object_id) {
    throw std::logic_error("object identity id does not match table key");
  }
  *updated = changed ? SceneObjectPtr(std::move(next)) : current;
  return changed;
}

SceneObjectId allocateObjectId(SceneState* state,
                               const SceneObjectTable& objects,
                               const SceneAliasTable& aliases,
                               const SceneTombstoneTable& tombstones) {
  SceneObjectId candidate = state->next_object_id;
  while (objects.count(candidate) != 0 || aliases.count(candidate) != 0 ||
         tombstones.count(candidate) != 0) {
    candidate = checkedNextObjectId(candidate);
  }
  state->next_object_id = checkedNextObjectId(candidate);
  return candidate;
}

void validateDependencyObject(const SceneState& state,
                              const ObjectDependency& dependency,
                              SceneObjectId* canonical_id,
                              SceneObjectPtr* object) {
  const auto canonical = resolveCanonical(*state.aliases, dependency.object_id);
  if (!canonical) {
    throw std::runtime_error("object alias cycle");
  }
  if (*canonical != dependency.object_id) {
    throw std::runtime_error("dependency targets a retired alias");
  }
  if (state.tombstones->count(*canonical) != 0) {
    throw std::runtime_error("dependency targets a tombstoned object");
  }
  const auto object_it = state.objects->find(*canonical);
  if (object_it == state.objects->end()) {
    throw std::runtime_error("dependency object does not exist");
  }
  if (!object_it->second->identity ||
      object_it->second->identity->revision !=
          dependency.identity_revision) {
    throw std::runtime_error("identity dependency is stale");
  }
  *canonical_id = *canonical;
  *object = object_it->second;
}

std::string exceptionReason(const char* prefix, const std::exception& error) {
  return std::string(prefix) + error.what();
}

bool relationKeyEqual(const ObjectRelation& lhs,
                      const ObjectRelation& rhs) {
  return relationSource(lhs) == relationSource(rhs) &&
         relationTarget(lhs) == relationTarget(rhs) &&
         lhs.relation_type == rhs.relation_type;
}

bool relationValueEqual(const SceneRelation& lhs,
                        const SceneRelation& rhs) {
  return relationKeyEqual(lhs, rhs) &&
         std::abs(lhs.confidence - rhs.confidence) <= 1.0e-6f &&
         lhs.description == rhs.description && lhs.derived == rhs.derived;
}

bool furnitureRolesEqual(const std::vector<FurnitureRole>& lhs,
                         const std::vector<FurnitureRole>& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                    [](const FurnitureRole& a, const FurnitureRole& b) {
                      return a.object_id == b.object_id &&
                             a.revision == b.revision &&
                             a.classification_label ==
                                 b.classification_label;
                    });
}

bool refreshFurnitureRelations(const SceneObjectTable& objects,
                               SceneGraphMetadata* graph,
                               const FurnitureGraphConfig& config,
                               SceneRevision revision,
                               std::vector<SceneEvent>* events) {
  std::vector<SceneRelation> existing;
  std::vector<SceneRelation> retained;
  retained.reserve(graph->relations.size());
  for (const SceneRelation& relation : graph->relations) {
    if (isDerivedFurnitureRelation(relation)) {
      existing.push_back(relation);
    } else {
      retained.push_back(relation);
    }
  }

  std::vector<SceneRelation> desired = deriveFurnitureRelations(
      objects, graph->rooms, graph->furniture, config, revision);
  bool changed = existing.size() != desired.size();
  for (SceneRelation& relation : desired) {
    const auto same_key = std::find_if(
        existing.begin(), existing.end(), [&](const SceneRelation& current) {
          return relationKeyEqual(current, relation);
        });
    if (same_key != existing.end() && relationValueEqual(*same_key, relation)) {
      relation = *same_key;
      continue;
    }
    changed = true;
    if (events != nullptr) {
      events->push_back(RelationCommitted{revision, relation});
    }
  }
  for (const SceneRelation& relation : existing) {
    const auto same_value = std::find_if(
        desired.begin(), desired.end(), [&](const SceneRelation& candidate) {
          return relationValueEqual(relation, candidate);
        });
    if (same_value != desired.end()) {
      continue;
    }
    changed = true;
    if (events != nullptr) {
      const SceneEntityRef source = relationSource(relation);
      events->push_back(RelationInvalidated{
          revision, entityBackedByObject(source) ? source.id : -1, source});
    }
  }
  if (!changed) {
    return false;
  }
  retained.insert(retained.end(), desired.begin(), desired.end());
  graph->relations = std::move(retained);
  return true;
}

FurnitureGraphRebuilt furnitureGraphStats(const SceneGraphMetadata& graph,
                                          SceneRevision revision) {
  FurnitureGraphRebuilt result;
  result.revision = revision;
  result.furniture_count = graph.furniture.size();
  for (const SceneRelation& relation : graph.relations) {
    if (!isDerivedFurnitureRelation(relation)) {
      continue;
    }
    if (relation.relation_type == "in") {
      ++result.in_relation_count;
    } else if (relation.relation_type == "on") {
      ++result.on_relation_count;
    } else if (relation.relation_type == "room_contains_furniture") {
      ++result.room_relation_count;
    }
  }
  return result;
}

bool relationTouches(const ObjectRelation& relation,
                     SceneEntityRef endpoint) {
  return relationSource(relation) == endpoint ||
         relationTarget(relation) == endpoint;
}

bool isRoomContainment(const ObjectRelation& relation) {
  const SceneEntityRef source = relationSource(relation);
  const SceneEntityRef target = relationTarget(relation);
  return relation.relation_type == "room_contains_object" &&
         source.type == SceneEntityType::kRoom &&
         target.type == SceneEntityType::kObject;
}

bool roomBounds(const RoomNode& room,
                std::array<float, 2>* min_xy,
                std::array<float, 2>* max_xy) {
  if (room.has_xy_bounds) {
    *min_xy = room.min_xy;
    *max_xy = room.max_xy;
  } else if (room.size_m.x() > 0.0f && room.size_m.y() > 0.0f) {
    *min_xy = {room.center_world.x() - 0.5f * room.size_m.x(),
               room.center_world.y() - 0.5f * room.size_m.y()};
    *max_xy = {room.center_world.x() + 0.5f * room.size_m.x(),
               room.center_world.y() + 0.5f * room.size_m.y()};
  } else {
    return false;
  }
  return std::isfinite((*min_xy)[0]) && std::isfinite((*min_xy)[1]) &&
         std::isfinite((*max_xy)[0]) && std::isfinite((*max_xy)[1]) &&
         (*min_xy)[0] <= (*max_xy)[0] &&
         (*min_xy)[1] <= (*max_xy)[1];
}

bool normalizeRoomGeometry(RoomNode* room,
                           bool bounds_were_patched,
                           std::string* reason) {
  if (!room->center_world.allFinite() || !room->size_m.allFinite() ||
      !std::isfinite(room->height_m) || room->size_m.z() < 0.0f ||
      room->height_m < 0.0f) {
    if (reason != nullptr) {
      *reason = "room height/Z extent is invalid";
    }
    return false;
  }
  if (bounds_were_patched) {
    if (!room->has_xy_bounds ||
        !std::isfinite(room->min_xy[0]) ||
        !std::isfinite(room->min_xy[1]) ||
        !std::isfinite(room->max_xy[0]) ||
        !std::isfinite(room->max_xy[1]) ||
        room->min_xy[0] >= room->max_xy[0] ||
        room->min_xy[1] >= room->max_xy[1]) {
      if (reason != nullptr) {
        *reason = "room XY bounds are invalid";
      }
      return false;
    }
    room->center_world.x() = 0.5f * (room->min_xy[0] + room->max_xy[0]);
    room->center_world.y() = 0.5f * (room->min_xy[1] + room->max_xy[1]);
    room->size_m.x() = room->max_xy[0] - room->min_xy[0];
    room->size_m.y() = room->max_xy[1] - room->min_xy[1];
  } else if (room->size_m.x() > 0.0f && room->size_m.y() > 0.0f &&
             std::isfinite(room->center_world.x()) &&
             std::isfinite(room->center_world.y()) &&
             std::isfinite(room->size_m.x()) &&
             std::isfinite(room->size_m.y())) {
    room->min_xy = {room->center_world.x() - 0.5f * room->size_m.x(),
                    room->center_world.y() - 0.5f * room->size_m.y()};
    room->max_xy = {room->center_world.x() + 0.5f * room->size_m.x(),
                    room->center_world.y() + 0.5f * room->size_m.y()};
    room->has_xy_bounds = true;
  } else if (!room->has_xy_bounds && room->size_m.x() == 0.0f &&
             room->size_m.y() == 0.0f) {
    // Legacy manual files may name a room before assigning geometry. Keep it
    // canonical but do not derive containment from a synthetic point room.
    return true;
  } else {
    if (reason != nullptr) {
      *reason = "room center/size are invalid";
    }
    return false;
  }
  if (room->height_m <= 0.0f && room->size_m.z() > 0.0f) {
    room->height_m = room->size_m.z();
  } else if (room->height_m > 0.0f) {
    room->size_m.z() = room->height_m;
  }
  return true;
}

void normalizeRelation(ObjectRelation* relation,
                       SceneRevision default_revision) {
  setRelationEndpoints(relation, relationSource(*relation),
                       relationTarget(*relation));
  if (relation->revision == 0) {
    relation->revision = default_revision;
  }
  if (isRoomContainment(*relation)) {
    relation->derived = true;
  }
}

bool recomputeContainmentForObject(
    SceneObjectId object_id,
    const SceneObjectPtr& object,
    SceneGraphMetadata* graph,
    SceneRevision revision,
    std::vector<SceneEvent>* events) {
  std::vector<ObjectRelation> desired;
  if (object && object->geometry) {
    for (const RoomNode& room : graph->rooms) {
      std::optional<ObjectRelation> relation = deriveRoomContainmentRelation(
          room, object_id, *object->geometry, revision);
      if (relation) {
        desired.push_back(std::move(*relation));
      }
    }
  }

  std::vector<ObjectRelation> existing;
  for (const ObjectRelation& relation : graph->relations) {
    const SceneEntityRef target = relationTarget(relation);
    if (isRoomContainment(relation) && target.id == object_id) {
      existing.push_back(relation);
    }
  }
  const auto has_key = [](const std::vector<ObjectRelation>& values,
                          const ObjectRelation& candidate) {
    return std::any_of(values.begin(), values.end(),
                       [&](const ObjectRelation& value) {
                         return relationKeyEqual(value, candidate);
                       });
  };
  bool changed = existing.size() != desired.size();
  if (!changed) {
    for (const ObjectRelation& relation : existing) {
      if (!has_key(desired, relation)) {
        changed = true;
        break;
      }
    }
  }
  if (!changed) {
    return false;
  }

  graph->relations.erase(
      std::remove_if(graph->relations.begin(), graph->relations.end(),
                     [&](const ObjectRelation& relation) {
                       return isRoomContainment(relation) &&
                              relationTarget(relation).id == object_id;
                     }),
      graph->relations.end());
  graph->relations.insert(graph->relations.end(), desired.begin(),
                          desired.end());
  if (events != nullptr) {
    events->push_back(RelationInvalidated{
        revision, object_id,
        SceneEntityRef{SceneEntityType::kObject, object_id}});
    for (const ObjectRelation& relation : desired) {
      events->push_back(RelationCommitted{revision, relation});
    }
  }
  return true;
}

bool recomputeContainmentForRoom(
    int room_id,
    const SceneObjectTable& objects,
    SceneGraphMetadata* graph,
    SceneRevision revision,
    std::vector<SceneEvent>* events) {
  auto room_it = std::find_if(
      graph->rooms.begin(), graph->rooms.end(),
      [&](const RoomNode& room) { return room.room_id == room_id; });
  if (room_it == graph->rooms.end()) {
    return false;
  }
  std::vector<ObjectRelation> desired;
  for (const auto& [object_id, object] : objects) {
    if (!object || !object->geometry) {
      continue;
    }
    std::optional<ObjectRelation> relation = deriveRoomContainmentRelation(
        *room_it, object_id, *object->geometry, revision);
    if (relation) {
      desired.push_back(std::move(*relation));
    }
  }
  std::vector<ObjectRelation> existing;
  for (const ObjectRelation& relation : graph->relations) {
    if (isRoomContainment(relation) &&
        relationSource(relation).id == room_id) {
      existing.push_back(relation);
    }
  }
  const auto has_key = [](const std::vector<ObjectRelation>& values,
                          const ObjectRelation& candidate) {
    return std::any_of(values.begin(), values.end(),
                       [&](const ObjectRelation& value) {
                         return relationKeyEqual(value, candidate);
                       });
  };
  bool changed = existing.size() != desired.size();
  if (!changed) {
    changed = std::any_of(existing.begin(), existing.end(),
                          [&](const ObjectRelation& relation) {
                            return !has_key(desired, relation);
                          });
  }
  if (!changed) {
    return false;
  }
  graph->relations.erase(
      std::remove_if(graph->relations.begin(), graph->relations.end(),
                     [&](const ObjectRelation& relation) {
                       return isRoomContainment(relation) &&
                              relationSource(relation).id == room_id;
                     }),
      graph->relations.end());
  graph->relations.insert(graph->relations.end(), desired.begin(),
                          desired.end());
  if (events != nullptr) {
    events->push_back(RelationInvalidated{
        revision, -1, SceneEntityRef{SceneEntityType::kRoom, room_id}});
    for (const ObjectRelation& relation : desired) {
      events->push_back(RelationCommitted{revision, relation});
    }
  }
  return true;
}

std::vector<int> derivedRoomMemberships(const SceneGraphMetadata& graph,
                                        SceneEntityRef target) {
  std::vector<int> memberships;
  for (const ObjectRelation& relation : graph.relations) {
    if (!isRoomContainment(relation)) {
      continue;
    }
    const SceneEntityRef source = relationSource(relation);
    const SceneEntityRef destination = relationTarget(relation);
    if (target.type == SceneEntityType::kObject && destination == target) {
      memberships.push_back(source.id);
    } else if (target.type == SceneEntityType::kRoom && source == target) {
      memberships.push_back(destination.id);
    }
  }
  std::sort(memberships.begin(), memberships.end());
  memberships.erase(std::unique(memberships.begin(), memberships.end()),
                    memberships.end());
  return memberships;
}

bool derivedRoomMembershipsMatch(const SceneGraphMetadata& graph,
                                 SceneEntityRef target,
                                 const std::vector<int>& expected) {
  std::vector<int> normalized = expected;
  std::sort(normalized.begin(), normalized.end());
  normalized.erase(std::unique(normalized.begin(), normalized.end()),
                   normalized.end());
  return derivedRoomMemberships(graph, target) == normalized;
}

constexpr std::size_t kObservationReplayWindow = 256;

const ObservationRunWatermark* findObservationWatermark(
    const SceneState& state,
    const RunId& run_id) {
  const auto it = std::find_if(
      state.observation_watermarks.begin(),
      state.observation_watermarks.end(),
      [&](const ObservationRunWatermark& watermark) {
        return watermark.run_id == run_id;
      });
  return it == state.observation_watermarks.end() ? nullptr : &*it;
}

bool observationAlreadyApplied(const SceneState& state,
                               const FrameKey& key) {
  return std::find(state.recent_observation_frames.begin(),
                   state.recent_observation_frames.end(),
                   key) != state.recent_observation_frames.end();
}

void recordObservationFrame(SceneState* state, const FrameKey& key) {
  state->recent_observation_frames.push_back(key);
  if (state->recent_observation_frames.size() > kObservationReplayWindow) {
    state->recent_observation_frames.erase(
        state->recent_observation_frames.begin(),
        state->recent_observation_frames.begin() +
            static_cast<std::ptrdiff_t>(
                state->recent_observation_frames.size() -
                kObservationReplayWindow));
  }
  auto it = std::find_if(
      state->observation_watermarks.begin(),
      state->observation_watermarks.end(),
      [&](const ObservationRunWatermark& watermark) {
        return watermark.run_id == key.run_id;
      });
  if (it == state->observation_watermarks.end()) {
    state->observation_watermarks.push_back(
        ObservationRunWatermark{key.run_id, key.frame_id});
  } else {
    it->highest_frame_id = std::max(it->highest_frame_id, key.frame_id);
  }
}

}  // namespace

std::optional<ObjectRelation> deriveRoomContainmentRelation(
    const RoomNode& room,
    SceneObjectId object_id,
    const GeometryComponent& geometry,
    SceneRevision revision) {
  if (room.room_id < 0 || object_id < 0 ||
      !std::isfinite(geometry.center_world.x()) ||
      !std::isfinite(geometry.center_world.y())) {
    return std::nullopt;
  }
  std::array<float, 2> min_xy;
  std::array<float, 2> max_xy;
  if (!roomBounds(room, &min_xy, &max_xy) ||
      geometry.center_world.x() < min_xy[0] ||
      geometry.center_world.x() > max_xy[0] ||
      geometry.center_world.y() < min_xy[1] ||
      geometry.center_world.y() > max_xy[1]) {
    return std::nullopt;
  }
  ObjectRelation relation;
  setRelationEndpoints(
      &relation, SceneEntityRef{SceneEntityType::kRoom, room.room_id},
      SceneEntityRef{SceneEntityType::kObject, object_id});
  relation.relation_type = "room_contains_object";
  relation.confidence = 1.0f;
  relation.revision = revision;
  relation.derived = true;
  return relation;
}

ObjectGraphSnapshot SceneSnapshot::materializeObjectGraph() const {
  ObjectGraphSnapshot graph;
  graph.schema_version = 6;
  graph.next_object_id = nextObjectId();
  graph.rooms = graphMetadata().rooms;
  graph.furniture = graphMetadata().furniture;
  graph.relations = graphMetadata().relations;
  graph.snapshot_images = graphMetadata().snapshot_images;
  graph.import_warnings = graphMetadata().import_warnings;
  graph.has_scene_graph_envelope = graphMetadata().has_scene_graph_envelope;
  graph.scene_graph_json = graphMetadata().scene_graph_json;

  graph.objects.reserve(objects().size());
  for (const auto& [object_id, object] : objects()) {
    if (!object || !object->identity || !object->lifecycle ||
        !object->geometry || !object->semantic || !object->annotation ||
        !object->artifact) {
      continue;
    }
    ObjectNode node;
    node.object_id = object_id;
    node.semantic_id = object->annotation->semantic_id_override.value_or(
        object->semantic->semantic_id);
    node.name = object->annotation->name;
    node.label = object->annotation->label_override.value_or(
        object->semantic->label);
    node.description = object->annotation->description_override.value_or(
        object->artifact->description);
    node.center_world = object->geometry->center_world;
    node.size_m = object->geometry->size_m;
    node.yaw_rad = object->geometry->yaw_rad;
    node.confidence = object->semantic->confidence;
    node.confidence_mass = object->semantic->confidence_mass;
    node.object_quality_score = object->semantic->object_quality_score;
    node.geometry_score = object->geometry->score;
    node.geometry_shell_ratio = object->geometry->shell_ratio;
    node.geometry_extent_score = object->geometry->extent_score;
    node.geometry_leak_ratio = object->geometry->leak_ratio;
    node.geometry_cavity_ratio = object->geometry->cavity_ratio;
    node.geometry_in_box_points = object->geometry->in_box_points;
    node.geometry_shell_points = object->geometry->shell_points;
    node.geometry_unique_voxels = object->geometry->unique_voxels;
    node.geometry_expanded_points = object->geometry->expanded_points;
    node.geometry_bad_count = object->geometry->bad_count;
    node.support_count = object->semantic->support_count;
    node.high_quality_observation_count =
        object->semantic->high_quality_observation_count;
    node.high_quality_observation_mass =
        object->semantic->high_quality_observation_mass;
    node.active = object->lifecycle->active;
    node.publishable = object->lifecycle->publishable;
    node.existence_log_odds = object->lifecycle->existence_log_odds;
    node.last_presence_evidence_ns =
        object->lifecycle->last_presence_evidence_ns;
    node.last_presence_evidence_reason =
        object->lifecycle->last_presence_evidence_reason;
    node.geometry_status = object->geometry->status;
    node.geometry_evaluation_obb_revision =
        object->geometry->evaluated_obb_revision;
    node.geometry_evaluation_map_version =
        object->geometry->evaluated_surface.source_map_revision;
    node.first_seen_ns = object->lifecycle->first_seen_ns;
    node.last_seen_ns = object->lifecycle->last_seen_ns;
    node.last_geometry_check_ns = object->geometry->last_check_ns;
    node.geometry_evaluated_center_world =
        object->geometry->evaluated_center_world;
    node.geometry_evaluated_size_m = object->geometry->evaluated_size_m;
    node.geometry_evaluated_yaw_rad = object->geometry->evaluated_yaw_rad;
    node.geometry_evaluation_reason = object->geometry->evaluation_reason;
    node.source_track_ids = object->identity->source_track_ids;
    node.source_cameras = object->semantic->source_cameras;
    node.observation_timestamps_ns =
        object->semantic->observation_timestamps_ns;
    if (!object->artifact->snapshots.empty()) {
      node.snapshot = object->artifact->snapshots.front();
    }
    node.label_weights = object->semantic->label_weights;
    node.semantic_weights = object->semantic->semantic_weights;
    graph.objects.push_back(std::move(node));
  }
  return graph;
}

ReducerCore::ReducerCore()
    : state_(std::make_shared<const SceneState>()),
      furniture_config_(defaultFurnitureGraphConfig()) {}

ReducerCore::ReducerCore(FurnitureGraphConfig furniture_config)
    : state_(std::make_shared<const SceneState>()),
      furniture_config_(std::move(furniture_config)) {}

ReducerCore::ReducerCore(ObservationAssociator associator)
    : state_(std::make_shared<const SceneState>()),
      furniture_config_(defaultFurnitureGraphConfig()),
      observation_associator_(std::move(associator)) {}

ReducerCore::ReducerCore(ObservationAssociator associator,
                         FurnitureGraphConfig furniture_config)
    : state_(std::make_shared<const SceneState>()),
      furniture_config_(std::move(furniture_config)),
      observation_associator_(std::move(associator)) {}

SceneSnapshot ReducerCore::snapshot() const { return SceneSnapshot(state_); }

void ReducerCore::setObservationAssociator(ObservationAssociator associator) {
  if (applying_) {
    throw std::logic_error("cannot replace observation associator during apply");
  }
  observation_associator_ = std::move(associator);
}

void ReducerCore::setCommitObserver(CommitObserver observer) {
  if (applying_) {
    throw std::logic_error("cannot replace commit observer during apply");
  }
  commit_observer_ = std::move(observer);
}

SceneApplyResult ReducerCore::apply(const SceneCommand& command) {
  if (applying_) {
    return reject("nested apply is forbidden");
  }
  const std::thread::id caller = std::this_thread::get_id();
  const bool bootstrap_load =
      !owner_thread_ &&
      std::holds_alternative<LoadSceneCommand>(command) &&
      state_->latest_scene_revision == 0;
  if (!bootstrap_load) {
    if (!owner_thread_) {
      owner_thread_ = caller;
    } else if (*owner_thread_ != caller) {
      return reject("ReducerCore apply called from a non-owner thread");
    }
  }

  if (state_->shutdown_requested &&
      !std::holds_alternative<PersistedThroughCommand>(command) &&
      !std::holds_alternative<ShutdownCommand>(command)) {
    return reject("reducer is shutting down");
  }

  struct ApplyingGuard {
    explicit ApplyingGuard(bool* applying) : applying_(applying) {
      *applying_ = true;
    }
    ~ApplyingGuard() { *applying_ = false; }
    bool* applying_;
  } guard(&applying_);

  try {
    SceneApplyResult result = std::visit(
        [this](const auto& typed_command) {
          return applyCommand(typed_command);
        },
        command);
    notify(&result);
    return result;
  } catch (const std::exception& error) {
    return reject(exceptionReason("scene command failed: ", error));
  }
}

SceneApplyResult ReducerCore::commit(SceneState next,
                                     SceneRevision revision,
                                     std::vector<SceneEvent> events) {
  const SceneRevision previous_revision = state_->latest_scene_revision;
  if (revision <= previous_revision) {
    throw std::logic_error("content commit must advance scene revision");
  }
  next.latest_scene_revision = revision;
  if (next.durable_scene_revision > revision) {
    throw std::logic_error("durable revision cannot exceed live revision");
  }
  events.push_back(SceneRevisionCommitted{previous_revision, revision});
  state_ = std::make_shared<const SceneState>(std::move(next));

  SceneApplyResult result;
  result.status = SceneApplyStatus::kCommitted;
  result.revision = revision;
  result.snapshot = snapshot();
  result.events = std::move(events);
  return result;
}

SceneApplyResult ReducerCore::metadataUpdate(SceneState next,
                                             std::vector<SceneEvent> events,
                                             std::string reason) {
  next.latest_scene_revision = state_->latest_scene_revision;
  state_ = std::make_shared<const SceneState>(std::move(next));
  SceneApplyResult result;
  result.status = SceneApplyStatus::kMetadataUpdated;
  result.revision = state_->latest_scene_revision;
  result.reason = std::move(reason);
  result.snapshot = snapshot();
  result.events = std::move(events);
  return result;
}

SceneApplyResult ReducerCore::noOp(std::string reason) const {
  SceneApplyResult result;
  result.status = SceneApplyStatus::kNoOp;
  result.revision = state_->latest_scene_revision;
  result.reason = std::move(reason);
  result.snapshot = snapshot();
  return result;
}

SceneApplyResult ReducerCore::reject(std::string reason) const {
  SceneApplyResult result;
  result.status = SceneApplyStatus::kRejected;
  result.revision = state_->latest_scene_revision;
  result.reason = std::move(reason);
  result.snapshot = snapshot();
  return result;
}

void ReducerCore::notify(SceneApplyResult* result) {
  if (!commit_observer_ || result->events.empty()) {
    return;
  }
  try {
    commit_observer_(*result);
  } catch (const std::exception& error) {
    if (!result->reason.empty()) {
      result->reason += "; ";
    }
    result->reason += "commit observer threw: ";
    result->reason += error.what();
  } catch (...) {
    if (!result->reason.empty()) {
      result->reason += "; ";
    }
    result->reason += "commit observer threw a non-standard exception";
  }
}

SceneApplyResult ReducerCore::applyCommand(const LoadSceneCommand& command) {
  if (state_->latest_scene_revision != 0 || !state_->objects->empty() ||
      !state_->aliases->empty() || !state_->tombstones->empty()) {
    return reject("LoadSceneCommand is only valid as the first scene command");
  }
  if (command.restored_state) {
    if (command.restored_state->latest_scene_revision == 0 ||
        command.restored_state->latest_scene_revision !=
            command.restored_revision ||
        command.restored_state->durable_scene_revision !=
            command.durable_revision ||
        command.durable_revision > command.restored_revision ||
        !command.restored_state->objects ||
        !command.restored_state->aliases ||
        !command.restored_state->tombstones ||
        !command.restored_state->tracks || !command.restored_state->graph) {
      return reject("restored SceneState watermarks or tables are invalid");
    }
    SceneState next = *command.restored_state;
    next.latest_surface = command.latest_surface;
    next.shutdown_requested = false;
    std::vector<SceneEvent> events;
    events.push_back(
        SceneLoaded{command.restored_revision, next.objects->size()});
    for (const auto& [object_id, object] : *next.objects) {
      (void)object;
      events.push_back(ObjectCreated{command.restored_revision, object_id});
    }
    return commit(std::move(next), command.restored_revision,
                  std::move(events));
  }
  SceneState next;
  SceneObjectTable objects;
  SceneAliasTable aliases;
  SceneTombstoneTable tombstones;
  SceneTrackTable tracks;
  SceneObjectId high_watermark = std::max(0, state_->next_object_id);
  const SceneRevision revision = std::max(
      nextSceneRevision(state_->latest_scene_revision),
      command.restored_revision);

  for (const ObjectNode& node : command.graph.objects) {
    if (node.object_id < 0) {
      return reject("load contains a negative object id");
    }
    if (!objects.emplace(node.object_id, objectFromNode(node)).second) {
      return reject("load contains duplicate object ids");
    }
    high_watermark = std::max(high_watermark, checkedNextObjectId(node.object_id));
  }

  for (const ObjectTombstone& tombstone : command.tombstones) {
    if (tombstone.object_id < 0 || objects.count(tombstone.object_id) != 0) {
      return reject("load tombstone collides with a live or invalid object id");
    }
    if (!tombstones.emplace(tombstone.object_id, tombstone).second) {
      return reject("load contains duplicate tombstones");
    }
    high_watermark =
        std::max(high_watermark, checkedNextObjectId(tombstone.object_id));
  }

  for (const ObjectAlias& alias : command.aliases) {
    if (alias.retired_object_id < 0 || alias.canonical_object_id < 0 ||
        alias.retired_object_id == alias.canonical_object_id ||
        objects.count(alias.retired_object_id) != 0 ||
        tombstones.count(alias.retired_object_id) != 0) {
      return reject("load contains an invalid alias");
    }
    if (!aliases.emplace(alias.retired_object_id, alias).second) {
      return reject("load contains duplicate aliases");
    }
    high_watermark = std::max(
        high_watermark, checkedNextObjectId(alias.retired_object_id));
    high_watermark = std::max(
        high_watermark, checkedNextObjectId(alias.canonical_object_id));
  }

  for (const auto& [retired_id, alias] : aliases) {
    (void)retired_id;
    const auto canonical = resolveCanonical(aliases, alias.canonical_object_id);
    if (!canonical || (objects.count(*canonical) == 0 &&
                       tombstones.count(*canonical) == 0)) {
      return reject("load alias does not resolve to a live object or tombstone");
    }
  }

  for (const InstanceTrack& input_track : command.tracks) {
    if (input_track.track_id < 0) {
      return reject("load contains a negative track id");
    }
    if (!tracks.emplace(input_track.track_id,
                        std::make_shared<const InstanceTrack>(input_track))
             .second) {
      return reject("load contains duplicate track ids");
    }
    if (input_track.object_id < 0) {
      continue;
    }
    if (aliases.count(input_track.object_id) != 0 ||
        tombstones.count(input_track.object_id) != 0) {
      return reject("load track refers to a retired object id");
    }
    auto object_it = objects.find(input_track.object_id);
    if (object_it == objects.end()) {
      objects.emplace(input_track.object_id,
                      objectFromTrack(input_track, input_track.object_id));
      high_watermark = std::max(
          high_watermark, checkedNextObjectId(input_track.object_id));
    } else {
      auto object = std::make_shared<SceneObject>(*object_it->second);
      if (addTrackToIdentity(object.get(), input_track.track_id)) {
        object_it->second = std::move(object);
      }
    }
  }

  high_watermark = std::max(high_watermark, command.graph.next_object_id);
  next.next_object_id = high_watermark;
  next.latest_surface = command.latest_surface;
  next.recent_observation_frames = command.recent_observation_frames;
  next.observation_watermarks = command.observation_watermarks;
  next.objects = std::make_shared<const SceneObjectTable>(std::move(objects));
  next.aliases = std::make_shared<const SceneAliasTable>(std::move(aliases));
  next.tombstones =
      std::make_shared<const SceneTombstoneTable>(std::move(tombstones));
  next.tracks = std::make_shared<const SceneTrackTable>(std::move(tracks));
  auto graph_metadata = std::make_shared<SceneGraphMetadata>();
  std::set<int> room_ids;
  graph_metadata->rooms = command.graph.rooms;
  for (RoomNode& room : graph_metadata->rooms) {
    if (room.room_id < 0 || !room_ids.insert(room.room_id).second) {
      return reject("load contains an invalid or duplicate room id");
    }
    if (room.revision == 0) {
      room.revision = 1;
    }
    std::string room_error;
    if (!normalizeRoomGeometry(&room, room.has_xy_bounds, &room_error)) {
      return reject("load contains invalid room geometry: " + room_error);
    }
  }
  std::set<int> furniture_ids;
  graph_metadata->furniture = command.graph.furniture;
  for (FurnitureRole& role : graph_metadata->furniture) {
    role.classification_label =
        normalizeFurnitureLabel(role.classification_label);
    if (role.object_id < 0 ||
        next.objects->count(role.object_id) == 0U ||
        !furniture_ids.insert(role.object_id).second ||
        role.classification_label.empty() ||
        furniture_config_.classes.count(role.classification_label) == 0U) {
      return reject("load contains an invalid furniture role");
    }
    if (role.revision == 0) {
      role.revision = 1;
    }
  }
  std::sort(graph_metadata->furniture.begin(),
            graph_metadata->furniture.end(),
            [](const FurnitureRole& lhs, const FurnitureRole& rhs) {
              return lhs.object_id < rhs.object_id;
            });
  std::vector<ObjectRelation> relations;
  for (ObjectRelation relation : command.graph.relations) {
    normalizeRelation(&relation, revision);
    const SceneEntityRef source = relationSource(relation);
    const SceneEntityRef target = relationTarget(relation);
    const auto endpoint_exists = [&](SceneEntityRef endpoint) {
      if (endpoint.type == SceneEntityType::kRoom) {
        return room_ids.count(endpoint.id) != 0U;
      }
      if (endpoint.type == SceneEntityType::kFurniture) {
        return furniture_ids.count(endpoint.id) != 0U;
      }
      return next.objects->count(endpoint.id) != 0U;
    };
    if (!source.valid() || !target.valid() || relation.relation_type.empty() ||
        !endpoint_exists(source) || !endpoint_exists(target)) {
      return reject("load contains an invalid canonical relation");
    }
    if (std::none_of(relations.begin(), relations.end(),
                     [&](const ObjectRelation& existing) {
                       return relationKeyEqual(existing, relation);
                     })) {
      relations.push_back(std::move(relation));
    }
  }
  // Containment is derived from the canonical room and geometry components,
  // never from duplicated parent_room_ids in a manual JSON envelope.
  relations.erase(std::remove_if(
                      relations.begin(), relations.end(),
                      [](const SceneRelation& relation) {
                        return isRoomContainment(relation) ||
                               isDerivedFurnitureRelation(relation);
                      }),
                  relations.end());
  graph_metadata->relations = std::move(relations);
  for (const auto& [object_id, object] : *next.objects) {
    recomputeContainmentForObject(object_id, object, graph_metadata.get(),
                                  revision, nullptr);
  }
  refreshFurnitureRelations(*next.objects, graph_metadata.get(),
                            furniture_config_, revision, nullptr);
  graph_metadata->snapshot_images = command.graph.snapshot_images;
  graph_metadata->import_warnings = command.graph.import_warnings;
  graph_metadata->has_scene_graph_envelope =
      command.graph.has_scene_graph_envelope;
  graph_metadata->scene_graph_json = command.graph.scene_graph_json;
  next.graph = std::move(graph_metadata);

  if (command.durable_revision > revision) {
    return reject("restored durable revision exceeds restored live revision");
  }
  next.durable_scene_revision = command.durable_revision;

  std::vector<SceneEvent> events;
  events.push_back(SceneLoaded{revision, next.objects->size()});
  for (const auto& [object_id, object] : *next.objects) {
    (void)object;
    events.push_back(ObjectCreated{revision, object_id});
  }
  return commit(std::move(next), revision, std::move(events));
}

SceneApplyResult ReducerCore::applyCommand(
    const ApplyObservationBatchCommand& command) {
  const bool has_replay_key = command.provenance.run_id.valid();
  const FrameKey replay_key{command.provenance.run_id,
                            command.provenance.frame_id};
  if (has_replay_key && observationAlreadyApplied(*state_, replay_key)) {
    return noOp("observation batch retry is already committed");
  }
  if (has_replay_key) {
    const ObservationRunWatermark* watermark =
        findObservationWatermark(*state_, replay_key.run_id);
    if (watermark != nullptr &&
        replay_key.frame_id < watermark->highest_frame_id &&
        watermark->highest_frame_id - replay_key.frame_id >=
            kObservationReplayWindow) {
      return reject("observation batch is outside the bounded late window");
    }
  }
  std::vector<ObservationMutation> mutations = command.associated_mutations;
  if (mutations.empty() && observation_associator_) {
    mutations = observation_associator_(snapshot(), command);
  }

  SceneState next = *state_;
  SceneObjectTable objects = *state_->objects;
  SceneAliasTable aliases = *state_->aliases;
  SceneTombstoneTable tombstones = *state_->tombstones;
  SceneTrackTable tracks = *state_->tracks;
  SceneGraphMetadata graph = *state_->graph;
  bool graph_changed = false;
  bool furniture_relations_dirty = false;
  std::set<SceneObjectId> containment_dirty_objects;
  const SceneRevision revision = nextSceneRevision(state_->latest_scene_revision);
  std::vector<SceneEvent> events;

  for (const ObservationMutation& mutation : mutations) {
    std::visit(
        [&](const auto& typed_mutation) {
          using Mutation = std::decay_t<decltype(typed_mutation)>;
          if constexpr (std::is_same_v<Mutation, UpsertTrackMutation>) {
            InstanceTrack track = typed_mutation.track;
            SceneObjectId requested_id = typed_mutation.object_id >= 0
                                             ? typed_mutation.object_id
                                             : track.object_id;
            bool is_new = false;
            if (requested_id < 0) {
              requested_id = allocateObjectId(
                  &next, objects, aliases, tombstones);
              is_new = true;
            } else {
              const auto canonical = resolveCanonical(aliases, requested_id);
              if (!canonical) {
                throw std::runtime_error("upsert encountered an alias cycle");
              }
              if (*canonical != requested_id) {
                throw std::runtime_error(
                    "upsert targets a retired alias; association must use canonical id");
              }
              if (tombstones.count(requested_id) != 0) {
                throw std::runtime_error("upsert cannot revive a tombstoned id");
              }
              is_new = objects.count(requested_id) == 0;
              if (is_new && requested_id < next.next_object_id) {
                throw std::runtime_error("upsert would reuse a retired object id");
              }
              if (is_new) {
                next.next_object_id = checkedNextObjectId(requested_id);
              }
            }

            track.object_id = requested_id;
            if (is_new) {
              objects[requested_id] = objectFromTrack(track, requested_id);
              events.push_back(ObjectCreated{revision, requested_id});
              events.push_back(ObbChanged{
                  revision, requested_id,
                  objects[requested_id]->geometry->obb_revision});
              containment_dirty_objects.insert(requested_id);
              furniture_relations_dirty = !graph.furniture.empty();
            } else {
              SceneObjectPtr updated;
              bool obb_changed = false;
              const bool changed = updateObjectFromTrack(
                  track, requested_id, objects.at(requested_id), &updated,
                  &obb_changed);
              objects[requested_id] = std::move(updated);
              if (changed) {
                events.push_back(ObjectUpdated{
                    revision, requested_id,
                    objects[requested_id]->revisions()});
                furniture_relations_dirty = !graph.furniture.empty();
              }
              if (obb_changed) {
                const std::uint64_t obb_revision =
                    objects[requested_id]->geometry->obb_revision;
                events.push_back(
                    ObbChanged{revision, requested_id, obb_revision});
                events.push_back(GeometryInvalidated{
                    revision, requested_id, obb_revision});
                containment_dirty_objects.insert(requested_id);
                furniture_relations_dirty = !graph.furniture.empty();
              }
            }
            if (track.track_id >= 0) {
              tracks[track.track_id] =
                  std::make_shared<const InstanceTrack>(std::move(track));
            }
          } else if constexpr (std::is_same_v<
                                   Mutation,
                                   UpsertTentativeTrackMutation>) {
            if (typed_mutation.track.track_id < 0 ||
                typed_mutation.track.object_id >= 0) {
              throw std::runtime_error(
                  "tentative track mutation requires a non-negative track id "
                  "and no object id");
            }
            tracks[typed_mutation.track.track_id] =
                std::make_shared<const InstanceTrack>(typed_mutation.track);
          } else if constexpr (std::is_same_v<Mutation,
                                               RemoveTrackMutation>) {
            if (typed_mutation.track_id < 0) {
              throw std::runtime_error(
                  "remove track mutation requires a non-negative id");
            }
            tracks.erase(typed_mutation.track_id);
          } else if constexpr (std::is_same_v<Mutation,
                                               MergeObjectsMutation>) {
            const auto source =
                resolveCanonical(aliases, typed_mutation.retired_object_id);
            const auto target =
                resolveCanonical(aliases, typed_mutation.canonical_object_id);
            if (!source || !target) {
              throw std::runtime_error("merge encountered an alias cycle");
            }
            if (*source == *target) {
              throw std::runtime_error("merge source and target are identical");
            }
            if (tombstones.count(*source) != 0 ||
                tombstones.count(*target) != 0) {
              throw std::runtime_error("merge cannot use a tombstoned object");
            }
            const auto source_it = objects.find(*source);
            const auto target_it = objects.find(*target);
            if (source_it == objects.end() || target_it == objects.end()) {
              throw std::runtime_error("merge requires two live objects");
            }

            auto target_object =
                std::make_shared<SceneObject>(*target_it->second);
            auto identity = std::make_shared<IdentityComponent>(
                *target_object->identity);
            identity->revision = nextComponentRevision(identity->revision);
            identity->source_track_ids.insert(
                identity->source_track_ids.end(),
                source_it->second->identity->source_track_ids.begin(),
                source_it->second->identity->source_track_ids.end());
            std::sort(identity->source_track_ids.begin(),
                      identity->source_track_ids.end());
            identity->source_track_ids.erase(
                std::unique(identity->source_track_ids.begin(),
                            identity->source_track_ids.end()),
                identity->source_track_ids.end());
            target_object->identity = std::move(identity);
            bool appearance_changed = false;
            bool description_invalidated = false;
            std::uint64_t merged_appearance_revision = 0;
            if (typed_mutation.merged_snapshots &&
                target_object->artifact &&
                (!snapshotSetsEqual(target_object->artifact->snapshots,
                                    *typed_mutation.merged_snapshots) ||
                 target_object->artifact->snapshot_set_hash !=
                     typed_mutation.merged_snapshot_set_hash)) {
              auto artifact = std::make_shared<ArtifactComponent>(
                  *target_object->artifact);
              artifact->revision =
                  nextComponentRevision(artifact->revision);
              artifact->appearance_revision =
                  nextComponentRevision(artifact->appearance_revision);
              artifact->snapshots = *typed_mutation.merged_snapshots;
              artifact->snapshot_set_hash =
                  typed_mutation.merged_snapshot_set_hash;
              description_invalidated = invalidateDescription(artifact.get());
              merged_appearance_revision = artifact->appearance_revision;
              target_object->artifact = std::move(artifact);
              appearance_changed = true;
            }
            target_it->second = std::move(target_object);
            objects.erase(source_it);

            for (auto& [alias_id, alias] : aliases) {
              (void)alias_id;
              if (alias.canonical_object_id == *source) {
                alias.canonical_object_id = *target;
              }
            }
            aliases[*source] = ObjectAlias{*source, *target, revision};
            auto source_role = std::find_if(
                graph.furniture.begin(), graph.furniture.end(),
                [&](const FurnitureRole& role) {
                  return role.object_id == *source;
                });
            auto target_role = std::find_if(
                graph.furniture.begin(), graph.furniture.end(),
                [&](const FurnitureRole& role) {
                  return role.object_id == *target;
                });
            if (source_role != graph.furniture.end()) {
              if (target_role == graph.furniture.end()) {
                source_role->object_id = *target;
                source_role->revision =
                    nextComponentRevision(source_role->revision);
              } else {
                graph.furniture.erase(source_role);
              }
              std::sort(graph.furniture.begin(), graph.furniture.end(),
                        [](const FurnitureRole& lhs,
                           const FurnitureRole& rhs) {
                          return lhs.object_id < rhs.object_id;
                        });
              graph_changed = true;
              furniture_relations_dirty = true;
            }
            for (auto& [track_id, track_ptr] : tracks) {
              (void)track_id;
              if (track_ptr && track_ptr->object_id == *source) {
                auto updated_track =
                    std::make_shared<InstanceTrack>(*track_ptr);
                updated_track->object_id = *target;
                track_ptr = std::move(updated_track);
              }
            }
            for (ObjectRelation& relation : graph.relations) {
              SceneEntityRef relation_source = relationSource(relation);
              SceneEntityRef relation_target = relationTarget(relation);
              if (entityBackedByObject(relation_source) &&
                  relation_source.id == *source) {
                relation_source.id = *target;
                graph_changed = true;
              }
              if (entityBackedByObject(relation_target) &&
                  relation_target.id == *source) {
                relation_target.id = *target;
                graph_changed = true;
              }
              setRelationEndpoints(&relation, relation_source,
                                   relation_target);
            }
            const auto relation_end = std::remove_if(
                graph.relations.begin(), graph.relations.end(),
                [](const ObjectRelation& relation) {
                  return relationSource(relation) == relationTarget(relation);
                });
            if (relation_end != graph.relations.end()) {
              graph.relations.erase(relation_end, graph.relations.end());
              graph_changed = true;
            }
            events.push_back(ObjectMerged{revision, *source, *target});
            if (appearance_changed) {
              events.push_back(AppearanceInvalidated{
                  revision, *target, merged_appearance_revision});
              events.push_back(SnapshotSetCommitted{
                  revision, *target, merged_appearance_revision,
                  typed_mutation.merged_snapshot_set_hash});
              if (description_invalidated) {
                events.push_back(DescriptionInvalidated{
                    revision, *target, merged_appearance_revision});
              }
            }
            events.push_back(ObjectUpdated{
                revision, *target, target_it->second->revisions()});
            events.push_back(RelationInvalidated{
                revision, *target,
                SceneEntityRef{SceneEntityType::kObject, *target}});
            containment_dirty_objects.insert(*target);
            furniture_relations_dirty = furniture_relations_dirty ||
                                        !graph.furniture.empty();
          } else if constexpr (std::is_same_v<Mutation,
                                               TombstoneObjectMutation>) {
            const auto canonical =
                resolveCanonical(aliases, typed_mutation.object_id);
            if (!canonical) {
              throw std::runtime_error("tombstone encountered an alias cycle");
            }
            if (tombstones.count(*canonical) != 0) {
              throw std::runtime_error("object is already tombstoned");
            }
            if (objects.erase(*canonical) == 0) {
              throw std::runtime_error("tombstone requires a live object");
            }
            tombstones[*canonical] = ObjectTombstone{
                *canonical, revision, typed_mutation.reason};
            const auto role_end = std::remove_if(
                graph.furniture.begin(), graph.furniture.end(),
                [&](const FurnitureRole& role) {
                  return role.object_id == *canonical;
                });
            if (role_end != graph.furniture.end()) {
              graph.furniture.erase(role_end, graph.furniture.end());
              graph_changed = true;
              furniture_relations_dirty = true;
            }
            for (auto track_it = tracks.begin(); track_it != tracks.end();) {
              if (track_it->second &&
                  track_it->second->object_id == *canonical) {
                track_it = tracks.erase(track_it);
              } else {
                ++track_it;
              }
            }
            const auto relation_end = std::remove_if(
                graph.relations.begin(), graph.relations.end(),
                [&](const ObjectRelation& relation) {
                  const SceneEntityRef source = relationSource(relation);
                  const SceneEntityRef target = relationTarget(relation);
                  return (entityBackedByObject(source) &&
                          source.id == *canonical) ||
                         (entityBackedByObject(target) &&
                          target.id == *canonical);
                });
            if (relation_end != graph.relations.end()) {
              graph.relations.erase(relation_end, graph.relations.end());
              graph_changed = true;
              events.push_back(RelationInvalidated{
                  revision, *canonical,
                  SceneEntityRef{SceneEntityType::kObject, *canonical}});
            }
            events.push_back(ObjectTombstoned{
                revision, *canonical, typed_mutation.reason});
          }
        },
        mutation);
  }

  for (SceneObjectId object_id : containment_dirty_objects) {
    const auto object_it = objects.find(object_id);
    if (object_it != objects.end()) {
      graph_changed |= recomputeContainmentForObject(
          object_id, object_it->second, &graph, revision, &events);
    }
  }
  if (furniture_relations_dirty) {
    graph_changed |= refreshFurnitureRelations(
        objects, &graph, furniture_config_, revision, &events);
  }

  next.objects = std::make_shared<const SceneObjectTable>(std::move(objects));
  next.aliases = std::make_shared<const SceneAliasTable>(std::move(aliases));
  next.tombstones =
      std::make_shared<const SceneTombstoneTable>(std::move(tombstones));
  next.tracks = std::make_shared<const SceneTrackTable>(std::move(tracks));
  if (graph_changed) {
    next.graph =
        std::make_shared<const SceneGraphMetadata>(std::move(graph));
  }
  if (has_replay_key) {
    recordObservationFrame(&next, replay_key);
  }
  events.push_back(ObservationBatchCommitted{
      revision, command.provenance, command.observations.size(),
      mutations.size()});
  return commit(std::move(next), revision, std::move(events));
}

SceneApplyResult ReducerCore::applyCommand(
    const AdvanceSurfaceCommand& command) {
  if (!command.surface.map_epoch.valid() ||
      command.surface.surface_revision == 0) {
    return reject("surface advance contains an invalid stamp");
  }
  const SurfaceStamp& current = state_->latest_surface;
  if (current == command.surface) {
    return noOp("surface watermark is already current");
  }
  if (current.map_epoch == command.surface.map_epoch &&
      (command.surface.surface_revision < current.surface_revision ||
       command.surface.source_map_revision < current.source_map_revision ||
       (command.surface.surface_revision == current.surface_revision &&
        !(command.surface == current)))) {
    return reject("surface watermark regressed or conflicted");
  }

  SceneState next = *state_;
  next.latest_surface = command.surface;
  std::vector<SceneEvent> events;
  events.push_back(
      SurfaceAdvanced{state_->latest_scene_revision, command.surface});
  return metadataUpdate(std::move(next), std::move(events));
}

SceneApplyResult ReducerCore::applyCommand(
    const ApplyGeometryResultCommand& command) {
  SceneObjectId object_id = -1;
  SceneObjectPtr object;
  try {
    validateDependencyObject(*state_, command.dependency.object, &object_id,
                             &object);
  } catch (const std::exception& error) {
    return reject(exceptionReason("geometry result rejected: ", error));
  }
  if (!object->geometry ||
      object->geometry->obb_revision !=
          command.dependency.object.obb_revision) {
    return reject("geometry result rejected: OBB dependency is stale");
  }
  if (!command.dependency.surface.map_epoch.valid() ||
      !command.current_surface.map_epoch.valid()) {
    return reject("geometry result rejected: map epoch is invalid");
  }
  if (command.dependency.surface.map_epoch !=
      command.current_surface.map_epoch) {
    return reject("geometry result rejected: map epoch changed");
  }
  if (state_->latest_surface.map_epoch.valid() &&
      state_->latest_surface.map_epoch != command.current_surface.map_epoch) {
    return reject("geometry result rejected: reducer map epoch differs");
  }
  if (state_->latest_surface.map_epoch.valid() &&
      (command.current_surface.surface_revision <
           state_->latest_surface.surface_revision ||
       command.current_surface.source_map_revision <
           state_->latest_surface.source_map_revision ||
       (command.current_surface.surface_revision ==
            state_->latest_surface.surface_revision &&
        !(command.current_surface == state_->latest_surface)))) {
    return reject("geometry result rejected: supplied current surface regressed");
  }

  const bool exact_surface =
      command.dependency.surface == command.current_surface;
  if (!exact_surface) {
    if (command.current_surface.surface_revision <=
            command.dependency.surface.surface_revision ||
        command.current_surface.source_map_revision <
            command.dependency.surface.source_map_revision) {
      return reject("geometry result rejected: surface stamp is not a monotonic successor");
    }
    if (command.delta_overlap != DeltaOverlapVerdict::kNoOverlap) {
      switch (command.delta_overlap) {
        case DeltaOverlapVerdict::kOverlap:
          return reject("geometry result rejected: map delta overlaps evaluated blocks");
        case DeltaOverlapVerdict::kJournalGap:
          return reject("geometry result rejected: map delta journal has a gap");
        case DeltaOverlapVerdict::kExactSurface:
        case DeltaOverlapVerdict::kUnknown:
        case DeltaOverlapVerdict::kNoOverlap:
          return reject("geometry result rejected: no authoritative no-overlap verdict");
      }
    }
  }

  SceneState next = *state_;
  SceneObjectTable objects = *state_->objects;
  auto updated_object = std::make_shared<SceneObject>(*object);
  auto geometry = std::make_shared<GeometryComponent>(*object->geometry);
  geometry->revision = nextComponentRevision(geometry->revision);
  geometry->status = command.result.status;
  geometry->score = command.result.score;
  geometry->shell_ratio = command.result.shell_ratio;
  geometry->extent_score = command.result.extent_score;
  geometry->leak_ratio = command.result.leak_ratio;
  geometry->cavity_ratio = command.result.cavity_ratio;
  geometry->in_box_points = command.result.in_box_points;
  geometry->shell_points = command.result.shell_points;
  geometry->unique_voxels = command.result.unique_voxels;
  geometry->expanded_points = command.result.expanded_points;
  geometry->bad_count = command.result.bad_count;
  geometry->last_check_ns = command.result.checked_at_ns;
  geometry->evaluated_center_world = command.result.evaluated_center_world;
  geometry->evaluated_size_m = command.result.evaluated_size_m;
  geometry->evaluated_yaw_rad = command.result.evaluated_yaw_rad;
  geometry->evaluation_reason = command.result.reason;
  geometry->evaluated_obb_revision = command.dependency.object.obb_revision;
  geometry->evaluated_surface = command.dependency.surface;
  geometry->evaluated_blocks = command.dependency.evaluated_blocks;
  const std::uint64_t geometry_revision = geometry->revision;
  updated_object->geometry = std::move(geometry);
  objects[object_id] = std::move(updated_object);
  next.objects = std::make_shared<const SceneObjectTable>(std::move(objects));
  next.latest_surface = command.current_surface;

  const SceneRevision revision = nextSceneRevision(state_->latest_scene_revision);
  std::vector<SceneEvent> events;
  events.push_back(GeometryCommitted{
      revision, object_id, geometry_revision, command.dependency.surface});
  events.push_back(ObjectUpdated{
      revision, object_id, next.objects->at(object_id)->revisions()});
  return commit(std::move(next), revision, std::move(events));
}

SceneApplyResult ReducerCore::applyCommand(
    const ApplySnapshotSetCommand& command) {
  SceneObjectId object_id = -1;
  SceneObjectPtr object;
  try {
    validateDependencyObject(*state_, command.dependency, &object_id, &object);
  } catch (const std::exception& error) {
    return reject(exceptionReason("snapshot set rejected: ", error));
  }
  if (!object->artifact ||
      object->artifact->appearance_revision !=
          command.dependency.appearance_revision) {
    return reject("snapshot set rejected: appearance dependency is stale");
  }
  if (command.dependency.semantic_revision != 0 &&
      (!object->semantic || object->semantic->revision !=
                                command.dependency.semantic_revision)) {
    return reject("snapshot set rejected: semantic dependency is stale");
  }
  if (object->artifact->snapshot_set_hash == command.snapshot_set_hash &&
      snapshotSetsEqual(object->artifact->snapshots, command.snapshots)) {
    return noOp("snapshot set is already current");
  }

  SceneState next = *state_;
  SceneObjectTable objects = *state_->objects;
  auto updated_object = std::make_shared<SceneObject>(*object);
  auto artifact = std::make_shared<ArtifactComponent>(*object->artifact);
  artifact->revision = nextComponentRevision(artifact->revision);
  artifact->appearance_revision =
      nextComponentRevision(artifact->appearance_revision);
  artifact->snapshots = command.snapshots;
  artifact->snapshot_set_hash = command.snapshot_set_hash;
  const bool invalidated_description = invalidateDescription(artifact.get());
  const std::uint64_t appearance_revision = artifact->appearance_revision;
  updated_object->artifact = std::move(artifact);
  objects[object_id] = std::move(updated_object);
  next.objects = std::make_shared<const SceneObjectTable>(std::move(objects));

  const SceneRevision revision = nextSceneRevision(state_->latest_scene_revision);
  std::vector<SceneEvent> events;
  events.push_back(AppearanceInvalidated{
      revision, object_id, appearance_revision});
  events.push_back(SnapshotSetCommitted{
      revision, object_id, appearance_revision, command.snapshot_set_hash});
  if (invalidated_description) {
    events.push_back(DescriptionInvalidated{
        revision, object_id, appearance_revision});
  }
  events.push_back(ObjectUpdated{
      revision, object_id, next.objects->at(object_id)->revisions()});
  return commit(std::move(next), revision, std::move(events));
}

SceneApplyResult ReducerCore::applyCommand(
    const ApplyDescriptionArtifactCommand& command) {
  if (!command.artifact_slo.valid()) {
    return reject("description artifact rejected: SLO context is invalid");
  }
  SceneObjectId object_id = -1;
  SceneObjectPtr object;
  try {
    validateDependencyObject(*state_, command.dependency, &object_id, &object);
  } catch (const std::exception& error) {
    return reject(exceptionReason("description artifact rejected: ", error));
  }
  if (!object->artifact ||
      object->artifact->appearance_revision !=
          command.dependency.appearance_revision) {
    return reject("description artifact rejected: appearance dependency is stale");
  }
  if (command.dependency.semantic_revision != 0 &&
      (!object->semantic ||
       object->semantic->revision != command.dependency.semantic_revision)) {
    return reject("description artifact rejected: semantic dependency is stale");
  }
  const ArtifactComponent& old = *object->artifact;
  if ((old.description == command.description &&
       old.description_input_hash == command.input_hash &&
       old.description_model_id == command.model_id &&
       old.description_schema_version == command.schema_version &&
       old.description_slo == command.artifact_slo &&
       !old.description_stale &&
       old.pending_description_scene_revision == 0) ||
      (old.pending_description == command.description &&
       old.pending_description_input_hash == command.input_hash &&
       old.pending_description_model_id == command.model_id &&
       old.pending_description_schema_version == command.schema_version &&
       old.pending_description_slo == command.artifact_slo)) {
    return noOp("description artifact is already current or pending");
  }

  const SceneRevision revision =
      nextSceneRevision(state_->latest_scene_revision);
  SceneState next = *state_;
  SceneObjectTable objects = *state_->objects;
  auto updated_object = std::make_shared<SceneObject>(*object);
  auto artifact = std::make_shared<ArtifactComponent>(old);
  artifact->revision = nextComponentRevision(artifact->revision);
  artifact->pending_description_scene_revision = revision;
  artifact->pending_description = command.description;
  artifact->pending_description_input_hash = command.input_hash;
  artifact->pending_description_model_id = command.model_id;
  artifact->pending_description_schema_version = command.schema_version;
  artifact->pending_description_raw_text = command.raw_text;
  artifact->pending_description_normalized_json = command.normalized_json;
  artifact->pending_description_durable_envelope_json =
      command.durable_envelope_json;
  artifact->pending_description_parse_path = command.parse_path;
  artifact->pending_description_slo = command.artifact_slo;
  artifact->description_stale = !artifact->description.empty();
  const std::uint64_t artifact_revision = artifact->revision;
  updated_object->artifact = std::move(artifact);
  objects[object_id] = std::move(updated_object);
  next.objects = std::make_shared<const SceneObjectTable>(std::move(objects));

  std::vector<SceneEvent> events;
  events.push_back(DescriptionCommitted{
      revision, object_id, artifact_revision, command.input_hash});
  events.push_back(ObjectUpdated{
      revision, object_id, next.objects->at(object_id)->revisions()});
  return commit(std::move(next), revision, std::move(events));
}

SceneApplyResult ReducerCore::applyCommand(
    const ApplyHumanAnnotationCommand& command) {
  SceneEntityRef target = command.target;
  if (!target.valid()) {
    target = SceneEntityRef{SceneEntityType::kObject, command.object_id};
  } else if (command.object_id >= 0 &&
             (target.type != SceneEntityType::kObject ||
              target.id != command.object_id)) {
    return reject("human annotation rejected: target conflicts with object_id");
  }
  if (!target.valid()) {
    return reject("human annotation rejected: target is invalid");
  }
  if (command.expected_scene_revision &&
      *command.expected_scene_revision != state_->latest_scene_revision) {
    return reject("human annotation rejected: scene dependency is stale; "
                  "expected=" +
                  std::to_string(*command.expected_scene_revision) +
                  " current=" +
                  std::to_string(state_->latest_scene_revision));
  }

  if (target.type == SceneEntityType::kRoom) {
    if (!command.patch.empty()) {
      return reject("room annotation rejected: object patch is not applicable");
    }
    if (!command.room_patch || command.room_patch->empty()) {
      if (command.expected_room_memberships) {
        const auto room_it = std::find_if(
            state_->graph->rooms.begin(), state_->graph->rooms.end(),
            [&](const RoomNode& room) { return room.room_id == target.id; });
        if (room_it == state_->graph->rooms.end()) {
          return reject(
              "room annotation rejected: room membership assertion targets "
              "a missing room");
        }
        if (!derivedRoomMembershipsMatch(
                *state_->graph, target,
                *command.expected_room_memberships)) {
          return reject(
              "room annotation rejected: room membership assertion is stale");
        }
      }
      return noOp("room annotation patch is empty");
    }
    const RoomAnnotationPatch& patch = *command.room_patch;
    SceneState next = *state_;
    SceneGraphMetadata graph = *state_->graph;
    auto room_it = std::find_if(
        graph.rooms.begin(), graph.rooms.end(),
        [&](const RoomNode& room) { return room.room_id == target.id; });
    const bool exists = room_it != graph.rooms.end();
    if (command.expected_room_revision != 0 &&
        (!exists || room_it->revision != command.expected_room_revision)) {
      return reject("room annotation rejected: room dependency is stale");
    }
    const SceneRevision revision =
        nextSceneRevision(state_->latest_scene_revision);
    std::vector<SceneEvent> events;
    if (patch.remove) {
      if (!exists) {
        if (command.expected_room_memberships &&
            !command.expected_room_memberships->empty()) {
          return reject(
              "room annotation rejected: room membership assertion is stale");
        }
        return noOp("room is already absent");
      }
      graph.rooms.erase(room_it);
      graph.relations.erase(
          std::remove_if(
              graph.relations.begin(), graph.relations.end(),
              [&](const ObjectRelation& relation) {
                return relationTouches(
                    relation,
                    SceneEntityRef{SceneEntityType::kRoom, target.id});
              }),
          graph.relations.end());
      if (command.expected_room_memberships &&
          !derivedRoomMembershipsMatch(
              graph, target, *command.expected_room_memberships)) {
        return reject(
            "room annotation rejected: room membership assertion is stale");
      }
      events.push_back(RoomRemoved{revision, target.id});
      events.push_back(RelationInvalidated{revision, -1, target});
      events.push_back(HumanAnnotationCommitted{
          revision, -1, 0, target});
      next.graph =
          std::make_shared<const SceneGraphMetadata>(std::move(graph));
      return commit(std::move(next), revision, std::move(events));
    }

    RoomNode room;
    if (exists) {
      room = *room_it;
    } else {
      room.room_id = target.id;
      if (static_cast<bool>(patch.min_xy) !=
          static_cast<bool>(patch.max_xy)) {
        return reject(
            "room annotation rejected: a new room requires both min_xy and "
            "max_xy");
      }
    }
    bool changed = !exists;
    const auto assign_if_changed = [&](auto* field, const auto& value) {
      if (*field != value) {
        *field = value;
        changed = true;
      }
    };
    if (patch.label) {
      assign_if_changed(&room.label, *patch.label);
    }
    if (patch.color) {
      assign_if_changed(&room.color, *patch.color);
    }
    if (patch.center_world &&
        !vectorsApprox(room.center_world, *patch.center_world)) {
      room.center_world = *patch.center_world;
      changed = true;
    }
    if (patch.size_m && !vectorsApprox(room.size_m, *patch.size_m)) {
      room.size_m = *patch.size_m;
      changed = true;
    }
    if (patch.min_xy) {
      if (!room.has_xy_bounds) {
        changed = true;
      }
      assign_if_changed(&room.min_xy, *patch.min_xy);
      room.has_xy_bounds = true;
    }
    if (patch.max_xy) {
      if (!room.has_xy_bounds) {
        changed = true;
      }
      assign_if_changed(&room.max_xy, *patch.max_xy);
      room.has_xy_bounds = true;
    }
    if (patch.height_m) {
      assign_if_changed(&room.height_m, *patch.height_m);
    }
    for (const auto& [key, value] : patch.attributes) {
      auto attribute_it = room.attributes.find(key);
      if (attribute_it == room.attributes.end() ||
          attribute_it->second != value) {
        room.attributes[key] = value;
        changed = true;
      }
    }
    if (!changed) {
      if (command.expected_room_memberships &&
          !derivedRoomMembershipsMatch(
              graph, target, *command.expected_room_memberships)) {
        return reject(
            "room annotation rejected: room membership assertion is stale");
      }
      return noOp("room annotation is already current");
    }
    std::string room_error;
    const bool bounds_were_patched = patch.min_xy || patch.max_xy;
    if (!normalizeRoomGeometry(&room, bounds_were_patched, &room_error)) {
      return reject("room annotation rejected: " + room_error);
    }
    room.revision = exists ? nextComponentRevision(room.revision) : 1;
    const std::uint64_t room_revision = room.revision;
    if (exists) {
      *room_it = room;
    } else {
      graph.rooms.push_back(room);
      std::sort(graph.rooms.begin(), graph.rooms.end(),
                [](const RoomNode& lhs, const RoomNode& rhs) {
                  return lhs.room_id < rhs.room_id;
                });
    }
    recomputeContainmentForRoom(target.id, *state_->objects, &graph,
                                revision, &events);
    if (!graph.furniture.empty()) {
      refreshFurnitureRelations(*state_->objects, &graph,
                                furniture_config_, revision, &events);
    }
    if (command.expected_room_memberships &&
        !derivedRoomMembershipsMatch(
            graph, target, *command.expected_room_memberships)) {
      return reject(
          "room annotation rejected: room membership assertion is stale");
    }
    events.push_back(
        RoomUpdated{revision, target.id, room_revision, !exists});
    events.push_back(HumanAnnotationCommitted{
        revision, -1, room_revision, target});
    next.graph =
        std::make_shared<const SceneGraphMetadata>(std::move(graph));
    return commit(std::move(next), revision, std::move(events));
  }

  if (command.room_patch) {
    return reject("object annotation rejected: room patch is not applicable");
  }
  const auto canonical = resolveCanonical(*state_->aliases, target.id);
  if (!canonical) {
    return reject("human annotation rejected: object alias cycle");
  }
  if (state_->tombstones->count(*canonical) != 0) {
    return reject("human annotation rejected: object is tombstoned");
  }
  const auto object_it = state_->objects->find(*canonical);
  if (object_it == state_->objects->end() || !object_it->second->identity ||
      !object_it->second->annotation) {
    return reject("human annotation rejected: object does not exist");
  }
  const SceneObjectPtr& object = object_it->second;
  if (command.expected_identity_revision != 0 &&
      object->identity->revision != command.expected_identity_revision) {
    return reject("human annotation rejected: identity dependency is stale");
  }
  if (command.expected_annotation_revision != 0 &&
      object->annotation->revision != command.expected_annotation_revision) {
    return reject("human annotation rejected: annotation dependency is stale");
  }
  if (command.expected_room_memberships &&
      !derivedRoomMembershipsMatch(
          *state_->graph,
          SceneEntityRef{SceneEntityType::kObject, *canonical},
          *command.expected_room_memberships)) {
    return reject(
        "human annotation rejected: room membership assertion is stale");
  }
  if (command.patch.empty()) {
    return noOp("human annotation patch is empty");
  }

  auto annotation =
      std::make_shared<AnnotationComponent>(*object->annotation);
  bool changed = false;
  if (command.patch.name && annotation->name != *command.patch.name) {
    annotation->name = *command.patch.name;
    changed = true;
  }
  if (command.patch.semantic_id &&
      annotation->semantic_id_override != command.patch.semantic_id) {
    annotation->semantic_id_override = command.patch.semantic_id;
    changed = true;
  }
  if (command.patch.label &&
      annotation->label_override != command.patch.label) {
    annotation->label_override = command.patch.label;
    changed = true;
  }
  if (command.patch.description &&
      annotation->description_override != command.patch.description) {
    annotation->description_override = command.patch.description;
    changed = true;
  }
  for (const auto& [key, value] : command.patch.attributes) {
    auto attribute_it = annotation->attributes.find(key);
    if (attribute_it == annotation->attributes.end() ||
        attribute_it->second != value) {
      annotation->attributes[key] = value;
      changed = true;
    }
  }
  if (!changed) {
    return noOp("human annotation is already current");
  }

  annotation->revision = nextComponentRevision(annotation->revision);
  const std::uint64_t annotation_revision = annotation->revision;
  SceneState next = *state_;
  SceneObjectTable objects = *state_->objects;
  auto updated_object = std::make_shared<SceneObject>(*object);
  updated_object->annotation = std::move(annotation);
  bool semantic_document_changed = false;
  if (object->identity && object->semantic && object->artifact &&
      object->annotation && updated_object->identity &&
      updated_object->semantic && updated_object->artifact &&
      updated_object->annotation) {
    semantic_document_changed =
        makeSemanticDocumentForObject(*object, *canonical, 0)
            .document_hash !=
        makeSemanticDocumentForObject(*updated_object, *canonical, 0)
            .document_hash;
  }
  objects[*canonical] = std::move(updated_object);
  next.objects = std::make_shared<const SceneObjectTable>(std::move(objects));

  const SceneRevision revision = nextSceneRevision(state_->latest_scene_revision);
  std::vector<SceneEvent> events;
  events.push_back(HumanAnnotationCommitted{
      revision, *canonical, annotation_revision,
      SceneEntityRef{SceneEntityType::kObject, *canonical},
      semantic_document_changed});
  events.push_back(ObjectUpdated{
      revision, *canonical, next.objects->at(*canonical)->revisions()});
  return commit(std::move(next), revision, std::move(events));
}

SceneApplyResult ReducerCore::applyCommand(
    const DeleteObjectCommand& command) {
  if (command.object_id < 0) {
    return reject("object delete rejected: object id is invalid");
  }
  if (command.expected_scene_revision &&
      *command.expected_scene_revision != state_->latest_scene_revision) {
    return reject("object delete rejected: scene dependency is stale; expected=" +
                  std::to_string(*command.expected_scene_revision) +
                  " current=" +
                  std::to_string(state_->latest_scene_revision));
  }
  const auto canonical = resolveCanonical(*state_->aliases, command.object_id);
  if (!canonical) {
    return reject("object delete rejected: object alias cycle");
  }
  if (state_->tombstones->count(*canonical) != 0) {
    return noOp("object is already tombstoned");
  }
  const auto object_it = state_->objects->find(*canonical);
  if (object_it == state_->objects->end() || !object_it->second ||
      !object_it->second->identity) {
    return reject("object delete rejected: object does not exist");
  }
  if (command.expected_identity_revision != 0 &&
      object_it->second->identity->revision !=
          command.expected_identity_revision) {
    return reject("object delete rejected: identity dependency is stale");
  }

  const SceneRevision revision =
      nextSceneRevision(state_->latest_scene_revision);
  SceneState next = *state_;
  SceneObjectTable objects = *state_->objects;
  SceneTombstoneTable tombstones = *state_->tombstones;
  SceneTrackTable tracks = *state_->tracks;
  SceneGraphMetadata graph = *state_->graph;

  objects.erase(*canonical);
  tombstones[*canonical] = ObjectTombstone{
      *canonical, revision,
      command.reason.empty() ? std::string("offline human delete")
                             : command.reason};
  for (auto track_it = tracks.begin(); track_it != tracks.end();) {
    if (track_it->second && track_it->second->object_id == *canonical) {
      track_it = tracks.erase(track_it);
    } else {
      ++track_it;
    }
  }
  graph.furniture.erase(
      std::remove_if(graph.furniture.begin(), graph.furniture.end(),
                     [&](const FurnitureRole& role) {
                       return role.object_id == *canonical;
                     }),
      graph.furniture.end());
  const std::size_t relation_count_before = graph.relations.size();
  graph.relations.erase(
      std::remove_if(
          graph.relations.begin(), graph.relations.end(),
          [&](const SceneRelation& relation) {
            const SceneEntityRef source = relationSource(relation);
            const SceneEntityRef target = relationTarget(relation);
            return (entityBackedByObject(source) &&
                    source.id == *canonical) ||
                   (entityBackedByObject(target) &&
                    target.id == *canonical);
          }),
      graph.relations.end());

  next.objects =
      std::make_shared<const SceneObjectTable>(std::move(objects));
  next.tombstones =
      std::make_shared<const SceneTombstoneTable>(std::move(tombstones));
  next.tracks =
      std::make_shared<const SceneTrackTable>(std::move(tracks));
  next.graph =
      std::make_shared<const SceneGraphMetadata>(std::move(graph));
  std::vector<SceneEvent> events;
  if (relation_count_before != next.graph->relations.size()) {
    events.push_back(RelationInvalidated{
        revision, *canonical,
        SceneEntityRef{SceneEntityType::kObject, *canonical}});
  }
  events.push_back(ObjectTombstoned{
      revision, *canonical,
      command.reason.empty() ? std::string("offline human delete")
                             : command.reason});
  return commit(std::move(next), revision, std::move(events));
}

SceneApplyResult ReducerCore::applyCommand(
    const RebuildFurnitureGraphCommand& command) {
  (void)command;
  SceneState next = *state_;
  SceneGraphMetadata graph = *state_->graph;
  const SceneRevision revision =
      nextSceneRevision(state_->latest_scene_revision);
  const std::vector<FurnitureRole> roles = classifyFurnitureRoles(
      *state_->objects, graph.furniture, furniture_config_);
  const bool roles_changed = !furnitureRolesEqual(graph.furniture, roles);
  if (roles_changed) {
    graph.furniture = roles;
  }
  std::vector<SceneEvent> events;
  const bool relations_changed = refreshFurnitureRelations(
      *state_->objects, &graph, furniture_config_, revision, &events);
  if (!roles_changed && !relations_changed) {
    return noOp("furniture graph is already current");
  }
  events.push_back(furnitureGraphStats(graph, revision));
  next.graph =
      std::make_shared<const SceneGraphMetadata>(std::move(graph));
  return commit(std::move(next), revision, std::move(events));
}

SceneApplyResult ReducerCore::applyCommand(
    const PersistedThroughCommand& command) {
  if (command.revision > state_->latest_scene_revision) {
    return reject("persistence watermark cannot exceed live scene revision");
  }
  if (command.revision <= state_->durable_scene_revision) {
    return noOp("persistence watermark did not advance");
  }
  SceneState next = *state_;
  next.durable_scene_revision = command.revision;
  std::vector<SceneEvent> events;
  SceneObjectTable objects = *state_->objects;
  bool promoted_description = false;
  for (auto& [object_id, object] : objects) {
    if (!object || !object->artifact ||
        object->artifact->pending_description_scene_revision == 0 ||
        object->artifact->pending_description_scene_revision >
            command.revision) {
      continue;
    }
    auto updated_object = std::make_shared<SceneObject>(*object);
    auto artifact =
        std::make_shared<ArtifactComponent>(*object->artifact);
    if (!promoteDurableDescription(artifact.get(), command.revision)) {
      continue;
    }
    updated_object->artifact = std::move(artifact);
    object = std::move(updated_object);
    events.push_back(ObjectUpdated{
        state_->latest_scene_revision, object_id, object->revisions()});
    promoted_description = true;
  }
  if (promoted_description) {
    next.objects =
        std::make_shared<const SceneObjectTable>(std::move(objects));
  }
  events.push_back(DurabilityWatermarkAdvanced{
      state_->latest_scene_revision, command.revision});
  return metadataUpdate(std::move(next), std::move(events));
}

SceneApplyResult ReducerCore::applyCommand(const ShutdownCommand& command) {
  if (state_->shutdown_requested) {
    return noOp("shutdown was already requested");
  }
  SceneState next = *state_;
  next.shutdown_requested = true;
  std::vector<SceneEvent> events;
  events.push_back(
      ShutdownAccepted{state_->latest_scene_revision, command.reason});
  return metadataUpdate(std::move(next), std::move(events));
}

}  // namespace roomie
