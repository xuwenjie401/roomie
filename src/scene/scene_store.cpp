#include "roomie/scene/scene_store.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <nlohmann/json.hpp>

namespace roomie {
namespace {

using Json = nlohmann::json;

class StoreError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::uint64_t jsonUnsigned(const Json& object, const char* field) {
  if (!object.is_object()) {
    throw StoreError("stored JSON value is not an object");
  }
  const auto value = object.find(field);
  if (value == object.end() ||
      (!value->is_number_unsigned() && !value->is_number_integer())) {
    throw StoreError(std::string("stored integer field is invalid: ") + field);
  }
  if (value->is_number_unsigned()) {
    return value->get<std::uint64_t>();
  }
  const std::int64_t signed_value = value->get<std::int64_t>();
  if (signed_value < 0) {
    throw StoreError(std::string("stored unsigned field is negative: ") +
                     field);
  }
  return static_cast<std::uint64_t>(signed_value);
}

std::int64_t jsonInt64(const Json& object,
                       const char* field,
                       bool nonnegative = false) {
  if (!object.is_object()) {
    throw StoreError("stored JSON value is not an object");
  }
  const auto value = object.find(field);
  if (value == object.end() ||
      (!value->is_number_unsigned() && !value->is_number_integer())) {
    throw StoreError(std::string("stored integer field is invalid: ") + field);
  }
  std::int64_t result = 0;
  if (value->is_number_unsigned()) {
    const std::uint64_t raw = value->get<std::uint64_t>();
    if (raw > static_cast<std::uint64_t>(
                  std::numeric_limits<std::int64_t>::max())) {
      throw StoreError(std::string("stored integer field overflows int64: ") +
                       field);
    }
    result = static_cast<std::int64_t>(raw);
  } else {
    result = value->get<std::int64_t>();
  }
  if (nonnegative && result < 0) {
    throw StoreError(std::string("stored integer field is negative: ") +
                     field);
  }
  return result;
}

template <typename Integer>
Integer jsonUnsignedAs(const Json& object, const char* field) {
  const std::uint64_t value = jsonUnsigned(object, field);
  if (value > static_cast<std::uint64_t>(
                  std::numeric_limits<Integer>::max())) {
    throw StoreError(std::string("stored unsigned field overflows: ") + field);
  }
  return static_cast<Integer>(value);
}

int jsonInt(const Json& object, const char* field) {
  const std::int64_t value = jsonInt64(object, field);
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw StoreError(std::string("stored integer field overflows int: ") +
                     field);
  }
  return static_cast<int>(value);
}

std::uint64_t jsonUnsignedOr(const Json& object,
                             const char* field,
                             std::uint64_t fallback) {
  if (!object.is_object()) {
    throw StoreError("stored JSON value is not an object");
  }
  return object.contains(field) ? jsonUnsigned(object, field) : fallback;
}

std::int64_t jsonInt64Or(const Json& object,
                         const char* field,
                         std::int64_t fallback,
                         bool nonnegative = false) {
  if (!object.is_object()) {
    throw StoreError("stored JSON value is not an object");
  }
  if (!object.contains(field)) {
    return fallback;
  }
  return jsonInt64(object, field, nonnegative);
}

int jsonIntOr(const Json& object, const char* field, int fallback) {
  if (!object.is_object()) {
    throw StoreError("stored JSON value is not an object");
  }
  return object.contains(field) ? jsonInt(object, field) : fallback;
}

std::int64_t jsonInt64Value(const Json& value,
                            const char* description,
                            bool nonnegative = false) {
  if (!value.is_number_unsigned() && !value.is_number_integer()) {
    throw StoreError(std::string("stored integer value is invalid: ") +
                     description);
  }
  std::int64_t result = 0;
  if (value.is_number_unsigned()) {
    const std::uint64_t raw = value.get<std::uint64_t>();
    if (raw > static_cast<std::uint64_t>(
                  std::numeric_limits<std::int64_t>::max())) {
      throw StoreError(std::string("stored integer value overflows int64: ") +
                       description);
    }
    result = static_cast<std::int64_t>(raw);
  } else {
    result = value.get<std::int64_t>();
  }
  if (nonnegative && result < 0) {
    throw StoreError(std::string("stored integer value is negative: ") +
                     description);
  }
  return result;
}

int jsonIntValue(const Json& value, const char* description) {
  const std::int64_t parsed = jsonInt64Value(value, description);
  if (parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    throw StoreError(std::string("stored integer value overflows int: ") +
                     description);
  }
  return static_cast<int>(parsed);
}

SceneObjectId jsonObjectId(const Json& object,
                           const char* field,
                           bool allow_unassigned = false) {
  const int value = jsonInt(object, field);
  if (value < (allow_unassigned ? -1 : 0)) {
    throw StoreError(std::string("stored object id is invalid: ") + field);
  }
  return value;
}

Json vector3ToJson(const Eigen::Vector3f& value) {
  return Json::array({value.x(), value.y(), value.z()});
}

Eigen::Vector3f vector3FromJson(const Json& value) {
  if (!value.is_array() || value.size() != 3) {
    throw StoreError("expected a three-element vector");
  }
  return Eigen::Vector3f(value.at(0).get<float>(), value.at(1).get<float>(),
                         value.at(2).get<float>());
}

Json vector3iToJson(const Eigen::Vector3i& value) {
  return Json::array({value.x(), value.y(), value.z()});
}

Eigen::Vector3i vector3iFromJson(const Json& value) {
  if (!value.is_array() || value.size() != 3) {
    throw StoreError("expected a three-element integer vector");
  }
  return Eigen::Vector3i(
      jsonIntValue(value.at(0), "integer vector element"),
      jsonIntValue(value.at(1), "integer vector element"),
      jsonIntValue(value.at(2), "integer vector element"));
}

Json runIdToJson(const RunId& run_id) {
  return Json{{"high", run_id.high}, {"low", run_id.low}};
}

RunId runIdFromJson(const Json& value) {
  RunId run_id;
  run_id.high = jsonUnsigned(value, "high");
  run_id.low = jsonUnsigned(value, "low");
  return run_id;
}

Json surfaceStampToJson(const SurfaceStamp& stamp) {
  return Json{{"map_epoch", runIdToJson(stamp.map_epoch)},
              {"surface_revision", stamp.surface_revision},
              {"source_map_revision", stamp.source_map_revision}};
}

SurfaceStamp surfaceStampFromJson(const Json& value) {
  SurfaceStamp stamp;
  stamp.map_epoch = runIdFromJson(value.at("map_epoch"));
  stamp.surface_revision =
      jsonUnsigned(value, "surface_revision");
  stamp.source_map_revision =
      jsonUnsigned(value, "source_map_revision");
  return stamp;
}

Json artifactSloToJson(const ArtifactSloContext& slo) {
  Json value{{"origin_created_unix_ms", slo.origin_created_unix_ms},
             {"due_unix_ms", slo.due_unix_ms},
             {"priority",
              slo.priority == ArtifactPriority::kInteractive
                  ? "interactive"
                  : "bulk"}};
  if (slo.monotonicTracked()) {
    value["steady_clock_epoch"] = runIdToJson(slo.steady_clock_epoch);
    value["origin_steady_ns"] = slo.origin_steady_ns;
    value["due_steady_ns"] = slo.due_steady_ns;
  }
  return value;
}

ArtifactSloContext artifactSloFromJson(const Json& value) {
  if (!value.is_object()) {
    throw StoreError("stored artifact SLO context is not an object");
  }
  ArtifactSloContext slo;
  slo.origin_created_unix_ms = jsonInt64(
      value, "origin_created_unix_ms");
  slo.due_unix_ms = jsonInt64(value, "due_unix_ms");
  const std::string priority = value.value("priority", std::string("bulk"));
  if (priority == "interactive") {
    slo.priority = ArtifactPriority::kInteractive;
  } else if (priority == "bulk") {
    slo.priority = ArtifactPriority::kBulk;
  } else {
    throw StoreError("stored artifact SLO priority is invalid");
  }
  const bool has_epoch = value.contains("steady_clock_epoch");
  const bool has_origin_steady = value.contains("origin_steady_ns");
  const bool has_due_steady = value.contains("due_steady_ns");
  if (has_epoch || has_origin_steady || has_due_steady) {
    if (!has_epoch || !has_origin_steady || !has_due_steady) {
      throw StoreError("stored artifact monotonic SLO tuple is incomplete");
    }
    slo.steady_clock_epoch = runIdFromJson(value.at("steady_clock_epoch"));
    slo.origin_steady_ns = jsonInt64(value, "origin_steady_ns", true);
    slo.due_steady_ns = jsonInt64(value, "due_steady_ns", true);
  }
  if (!slo.valid()) {
    throw StoreError("stored artifact SLO context is invalid");
  }
  return slo;
}

Json provenanceToJson(const FrameProvenance& provenance) {
  return Json{{"run_id", runIdToJson(provenance.run_id)},
              {"frame_id", provenance.frame_id},
              {"request_id", provenance.request_id},
              {"sensor_time_ns", provenance.sensor_time_ns},
              {"map_mode", static_cast<int>(provenance.map_mode)},
              {"includes_current_frame", provenance.includes_current_frame},
              {"causality_verified", provenance.causality_verified},
              {"map",
               Json{{"map_epoch", runIdToJson(provenance.map.map_epoch)},
                    {"map_revision", provenance.map.map_revision},
                    {"integrated_through_ns",
                     provenance.map.integrated_through_ns}}},
              {"surface", surfaceStampToJson(provenance.surface)}};
}

FrameProvenance provenanceFromJson(const Json& value) {
  FrameProvenance provenance;
  if (!value.is_object()) {
    return provenance;
  }
  provenance.run_id = runIdFromJson(
      value.value("run_id", Json{{"high", 0}, {"low", 0}}));
  provenance.frame_id = static_cast<FrameId>(
      jsonUnsignedOr(value, "frame_id", 0));
  provenance.request_id = static_cast<RequestId>(
      jsonUnsignedOr(value, "request_id", 0));
  provenance.sensor_time_ns = static_cast<TimeNanoseconds>(
      jsonInt64Or(value, "sensor_time_ns", 0, true));
  const int map_mode = jsonIntOr(value, "map_mode", 0);
  if (map_mode < static_cast<int>(MapMode::kOnline) ||
      map_mode > static_cast<int>(MapMode::kFrozen)) {
    throw StoreError("stored provenance map mode is invalid");
  }
  provenance.map_mode = static_cast<MapMode>(map_mode);
  provenance.includes_current_frame =
      value.value("includes_current_frame", false);
  provenance.causality_verified =
      value.value("causality_verified", false);
  const Json map = value.value("map", Json::object());
  provenance.map.map_epoch = runIdFromJson(
      map.value("map_epoch", Json{{"high", 0}, {"low", 0}}));
  provenance.map.map_revision =
      jsonUnsignedOr(map, "map_revision", 0);
  provenance.map.integrated_through_ns = static_cast<TimeNanoseconds>(
      jsonInt64Or(map, "integrated_through_ns", 0, true));
  provenance.surface = surfaceStampFromJson(
      value.value("surface",
                  Json{{"map_epoch", Json{{"high", 0}, {"low", 0}}},
                       {"surface_revision", 0},
                       {"source_map_revision", 0}}));
  return provenance;
}

Json snapshotRefToJson(const ObjectSnapshotRef& snapshot) {
  return Json{{"image_index", snapshot.image_index},
              {"source_frame_asset_id", snapshot.source_frame_asset_id},
              {"evidence_hash", snapshot.evidence_hash},
              {"bbox_xyxy", snapshot.bbox_xyxy},
              {"crop_xywh", snapshot.crop_xywh},
              {"crop_output_scale", snapshot.crop_output_scale},
              {"mask_source", snapshot.mask_source},
              {"mask_ref", snapshot.mask_ref},
              {"quality", snapshot.quality},
              {"quality_components", snapshot.quality_components},
              {"viewpoint_azimuth_rad", snapshot.viewpoint_azimuth_rad},
              {"viewpoint_elevation_rad", snapshot.viewpoint_elevation_rad},
              {"viewpoint_scale", snapshot.viewpoint_scale},
              {"time_ns", snapshot.time_ns},
              {"camera_id", snapshot.camera_id},
              {"provenance", provenanceToJson(snapshot.provenance)}};
}

ObjectSnapshotRef snapshotRefFromJson(const Json& value) {
  ObjectSnapshotRef snapshot;
  snapshot.image_index = jsonIntOr(value, "image_index", -1);
  if (snapshot.image_index < -1) {
    throw StoreError("stored snapshot image index is invalid");
  }
  snapshot.source_frame_asset_id =
      value.value("source_frame_asset_id", std::string());
  snapshot.evidence_hash = value.value("evidence_hash", std::string());
  snapshot.bbox_xyxy = value.value(
      "bbox_xyxy", std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f});
  snapshot.crop_xywh = value.value(
      "crop_xywh", std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f});
  snapshot.crop_output_scale = value.value(
      "crop_output_scale", std::array<float, 2>{1.0f, 1.0f});
  snapshot.mask_source =
      value.value("mask_source", std::string("bbox_fallback"));
  snapshot.mask_ref = value.value("mask_ref", std::string());
  snapshot.quality = value.value("quality", 0.0f);
  snapshot.quality_components = value.value(
      "quality_components", std::map<std::string, float>{});
  snapshot.viewpoint_azimuth_rad =
      value.value("viewpoint_azimuth_rad", 0.0f);
  snapshot.viewpoint_elevation_rad =
      value.value("viewpoint_elevation_rad", 0.0f);
  snapshot.viewpoint_scale = value.value("viewpoint_scale", 1.0f);
  snapshot.time_ns = static_cast<TimeNanoseconds>(
      jsonInt64Or(value, "time_ns", 0, true));
  snapshot.camera_id = value.value("camera_id", std::string());
  snapshot.provenance =
      provenanceFromJson(value.value("provenance", Json::object()));
  return snapshot;
}

Json voxelRefToJson(const VoxelRef& voxel) {
  return Json{{"block", vector3iToJson(voxel.block_index)},
              {"voxel", vector3iToJson(voxel.voxel_index)}};
}

VoxelRef voxelRefFromJson(const Json& value) {
  VoxelRef voxel;
  voxel.block_index = vector3iFromJson(value.at("block"));
  voxel.voxel_index = vector3iFromJson(value.at("voxel"));
  return voxel;
}

Json semanticWeightsToJson(const std::map<int, float>& weights) {
  Json result = Json::array();
  for (const auto& [semantic_id, weight] : weights) {
    result.push_back(Json{{"semantic_id", semantic_id}, {"weight", weight}});
  }
  return result;
}

std::map<int, float> semanticWeightsFromJson(const Json& value) {
  std::map<int, float> result;
  for (const Json& item : value) {
    result.emplace(jsonInt(item, "semantic_id"),
                   item.at("weight").get<float>());
  }
  return result;
}

Json objectToJson(const SceneObject& object) {
  if (!object.identity || !object.lifecycle || !object.geometry ||
      !object.semantic || !object.annotation || !object.artifact) {
    throw StoreError("scene object has a missing component");
  }

  const IdentityComponent& identity = *object.identity;
  const LifecycleComponent& lifecycle = *object.lifecycle;
  const GeometryComponent& geometry = *object.geometry;
  const SemanticComponent& semantic = *object.semantic;
  const AnnotationComponent& annotation = *object.annotation;
  const ArtifactComponent& artifact = *object.artifact;

  Json evaluated_blocks = Json::array();
  for (const Eigen::Vector3i& block : geometry.evaluated_blocks) {
    evaluated_blocks.push_back(vector3iToJson(block));
  }
  Json snapshots = Json::array();
  for (const ObjectSnapshotRef& snapshot : artifact.snapshots) {
    snapshots.push_back(snapshotRefToJson(snapshot));
  }

  Json annotation_json{
      {"revision", annotation.revision},
      {"attributes", annotation.attributes},
  };
  annotation_json["semantic_id_override"] =
      annotation.semantic_id_override
          ? Json(*annotation.semantic_id_override)
          : Json(nullptr);
  annotation_json["label_override"] =
      annotation.label_override ? Json(*annotation.label_override)
                                : Json(nullptr);
  annotation_json["description_override"] =
      annotation.description_override ? Json(*annotation.description_override)
                                      : Json(nullptr);

  return Json{
      {"identity",
       Json{{"revision", identity.revision},
            {"object_id", identity.object_id},
            {"source_track_ids", identity.source_track_ids}}},
      {"lifecycle",
       Json{{"revision", lifecycle.revision},
            {"track_state", static_cast<int>(lifecycle.track_state)},
            {"active", lifecycle.active},
            {"publishable", lifecycle.publishable},
            {"existence_log_odds", lifecycle.existence_log_odds},
            {"last_presence_evidence_ns",
             lifecycle.last_presence_evidence_ns},
            {"last_presence_evidence_reliability",
             lifecycle.last_presence_evidence_reliability},
            {"last_presence_evidence_reason",
             lifecycle.last_presence_evidence_reason},
            {"positive_evidence_timestamps_ns",
             lifecycle.positive_evidence_timestamps_ns},
            {"negative_evidence_timestamps_ns",
             lifecycle.negative_evidence_timestamps_ns},
            {"positive_window_interruptions",
             lifecycle.positive_window_interruptions},
            {"first_seen_ns", lifecycle.first_seen_ns},
            {"last_seen_ns", lifecycle.last_seen_ns},
            {"first_seen_frame_index", lifecycle.first_seen_frame_index},
            {"last_seen_frame_index", lifecycle.last_seen_frame_index}}},
      {"geometry",
       Json{{"revision", geometry.revision},
            {"obb_revision", geometry.obb_revision},
            {"evaluated_obb_revision", geometry.evaluated_obb_revision},
            {"center_world", vector3ToJson(geometry.center_world)},
            {"size_m", vector3ToJson(geometry.size_m)},
            {"yaw_rad", geometry.yaw_rad},
            {"status", static_cast<int>(geometry.status)},
            {"score", geometry.score},
            {"shell_ratio", geometry.shell_ratio},
            {"extent_score", geometry.extent_score},
            {"leak_ratio", geometry.leak_ratio},
            {"cavity_ratio", geometry.cavity_ratio},
            {"in_box_points", geometry.in_box_points},
            {"shell_points", geometry.shell_points},
            {"unique_voxels", geometry.unique_voxels},
            {"expanded_points", geometry.expanded_points},
            {"bad_count", geometry.bad_count},
            {"last_check_ns", geometry.last_check_ns},
            {"evaluated_center_world",
             vector3ToJson(geometry.evaluated_center_world)},
            {"evaluated_size_m", vector3ToJson(geometry.evaluated_size_m)},
            {"evaluated_yaw_rad", geometry.evaluated_yaw_rad},
            {"evaluation_reason", geometry.evaluation_reason},
            {"evaluated_surface", surfaceStampToJson(geometry.evaluated_surface)},
            {"evaluated_blocks", std::move(evaluated_blocks)}}},
      {"semantic",
       Json{{"revision", semantic.revision},
            {"semantic_id", semantic.semantic_id},
            {"label", semantic.label},
            {"confidence", semantic.confidence},
            {"confidence_mass", semantic.confidence_mass},
            {"object_quality_score", semantic.object_quality_score},
            {"support_count", semantic.support_count},
            {"high_quality_observation_count",
             semantic.high_quality_observation_count},
            {"high_quality_observation_mass",
             semantic.high_quality_observation_mass},
            {"source_cameras", semantic.source_cameras},
            {"observation_timestamps_ns",
             semantic.observation_timestamps_ns},
            {"label_weights", semantic.label_weights},
            {"semantic_weights",
             semanticWeightsToJson(semantic.semantic_weights)}}},
      {"annotation", std::move(annotation_json)},
      {"artifact",
       Json{{"revision", artifact.revision},
            {"appearance_revision", artifact.appearance_revision},
            {"snapshots", std::move(snapshots)},
            {"snapshot_set_hash", artifact.snapshot_set_hash},
            {"description", artifact.description},
            {"description_input_hash", artifact.description_input_hash},
            {"description_model_id", artifact.description_model_id},
            {"description_schema_version",
             artifact.description_schema_version},
            {"description_raw_text", artifact.description_raw_text},
            {"description_normalized_json",
             artifact.description_normalized_json},
            {"description_durable_envelope_json",
             artifact.description_durable_envelope_json},
            {"description_parse_path", artifact.description_parse_path},
            {"description_slo", artifactSloToJson(artifact.description_slo)},
            {"description_stale", artifact.description_stale},
            {"pending_description_scene_revision",
             artifact.pending_description_scene_revision},
            {"pending_description", artifact.pending_description},
            {"pending_description_input_hash",
             artifact.pending_description_input_hash},
            {"pending_description_model_id",
             artifact.pending_description_model_id},
            {"pending_description_schema_version",
             artifact.pending_description_schema_version},
            {"pending_description_raw_text",
             artifact.pending_description_raw_text},
            {"pending_description_normalized_json",
             artifact.pending_description_normalized_json},
            {"pending_description_durable_envelope_json",
             artifact.pending_description_durable_envelope_json},
            {"pending_description_parse_path",
             artifact.pending_description_parse_path},
            {"pending_description_slo",
             artifactSloToJson(artifact.pending_description_slo)}}},
  };
}

SceneObjectPtr objectFromJson(const Json& value) {
  auto object = std::make_shared<SceneObject>();

  const Json& identity_json = value.at("identity");
  auto identity = std::make_shared<IdentityComponent>();
  identity->revision = jsonUnsigned(identity_json, "revision");
  identity->object_id = jsonObjectId(identity_json, "object_id");
  for (const Json& track_id : identity_json.at("source_track_ids")) {
    Json wrapper{{"value", track_id}};
    identity->source_track_ids.push_back(jsonInt(wrapper, "value"));
  }
  object->identity = std::move(identity);

  const Json& lifecycle_json = value.at("lifecycle");
  auto lifecycle = std::make_shared<LifecycleComponent>();
  lifecycle->revision = jsonUnsigned(lifecycle_json, "revision");
  const int track_state = jsonInt(lifecycle_json, "track_state");
  if (track_state < static_cast<int>(InstanceTrackState::kTentative) ||
      track_state > static_cast<int>(InstanceTrackState::kInactive)) {
    throw StoreError("stored track state is invalid");
  }
  lifecycle->track_state = static_cast<InstanceTrackState>(track_state);
  lifecycle->active = lifecycle_json.at("active").get<bool>();
  lifecycle->publishable = lifecycle_json.at("publishable").get<bool>();
  lifecycle->existence_log_odds = lifecycle_json.value(
      "existence_log_odds",
      lifecycle->active ? 1.0986123f : -1.0986123f);
  lifecycle->last_presence_evidence_ns = lifecycle_json.value(
      "last_presence_evidence_ns", TimeNanoseconds{0});
  lifecycle->last_presence_evidence_reliability = lifecycle_json.value(
      "last_presence_evidence_reliability", 0.0f);
  lifecycle->last_presence_evidence_reason = lifecycle_json.value(
      "last_presence_evidence_reason", std::string("legacy_restore"));
  lifecycle->positive_evidence_timestamps_ns = lifecycle_json.value(
      "positive_evidence_timestamps_ns", std::vector<TimeNanoseconds>());
  lifecycle->negative_evidence_timestamps_ns = lifecycle_json.value(
      "negative_evidence_timestamps_ns", std::vector<TimeNanoseconds>());
  lifecycle->positive_window_interruptions = lifecycle_json.value(
      "positive_window_interruptions", 0);
  lifecycle->first_seen_ns =
      jsonInt64(lifecycle_json, "first_seen_ns", true);
  lifecycle->last_seen_ns =
      jsonInt64(lifecycle_json, "last_seen_ns", true);
  lifecycle->first_seen_frame_index =
      jsonUnsigned(lifecycle_json, "first_seen_frame_index");
  lifecycle->last_seen_frame_index =
      jsonUnsigned(lifecycle_json, "last_seen_frame_index");
  object->lifecycle = std::move(lifecycle);

  const Json& geometry_json = value.at("geometry");
  auto geometry = std::make_shared<GeometryComponent>();
  geometry->revision = jsonUnsigned(geometry_json, "revision");
  geometry->obb_revision =
      jsonUnsigned(geometry_json, "obb_revision");
  geometry->evaluated_obb_revision =
      jsonUnsigned(geometry_json, "evaluated_obb_revision");
  geometry->center_world = vector3FromJson(geometry_json.at("center_world"));
  geometry->size_m = vector3FromJson(geometry_json.at("size_m"));
  geometry->yaw_rad = geometry_json.at("yaw_rad").get<float>();
  const int geometry_status = jsonInt(geometry_json, "status");
  if (geometry_status < static_cast<int>(InstanceGeometryStatus::kUnchecked) ||
      geometry_status > static_cast<int>(InstanceGeometryStatus::kEmpty)) {
    throw StoreError("stored geometry status is invalid");
  }
  geometry->status = static_cast<InstanceGeometryStatus>(geometry_status);
  geometry->score = geometry_json.at("score").get<float>();
  geometry->shell_ratio = geometry_json.at("shell_ratio").get<float>();
  geometry->extent_score = geometry_json.at("extent_score").get<float>();
  geometry->leak_ratio = geometry_json.at("leak_ratio").get<float>();
  geometry->cavity_ratio = geometry_json.at("cavity_ratio").get<float>();
  geometry->in_box_points = jsonInt(geometry_json, "in_box_points");
  geometry->shell_points = jsonInt(geometry_json, "shell_points");
  geometry->unique_voxels = jsonInt(geometry_json, "unique_voxels");
  geometry->expanded_points = jsonInt(geometry_json, "expanded_points");
  geometry->bad_count = jsonInt(geometry_json, "bad_count");
  geometry->last_check_ns =
      jsonInt64(geometry_json, "last_check_ns", true);
  geometry->evaluated_center_world =
      vector3FromJson(geometry_json.at("evaluated_center_world"));
  geometry->evaluated_size_m =
      vector3FromJson(geometry_json.at("evaluated_size_m"));
  geometry->evaluated_yaw_rad =
      geometry_json.at("evaluated_yaw_rad").get<float>();
  geometry->evaluation_reason =
      geometry_json.at("evaluation_reason").get<std::string>();
  geometry->evaluated_surface =
      surfaceStampFromJson(geometry_json.at("evaluated_surface"));
  for (const Json& block : geometry_json.at("evaluated_blocks")) {
    geometry->evaluated_blocks.push_back(vector3iFromJson(block));
  }
  object->geometry = std::move(geometry);

  const Json& semantic_json = value.at("semantic");
  auto semantic = std::make_shared<SemanticComponent>();
  semantic->revision = jsonUnsigned(semantic_json, "revision");
  semantic->semantic_id = jsonInt(semantic_json, "semantic_id");
  semantic->label = semantic_json.at("label").get<std::string>();
  semantic->confidence = semantic_json.at("confidence").get<float>();
  semantic->confidence_mass =
      semantic_json.at("confidence_mass").get<float>();
  semantic->object_quality_score =
      semantic_json.at("object_quality_score").get<float>();
  semantic->support_count = jsonInt(semantic_json, "support_count");
  semantic->high_quality_observation_count =
      jsonInt(semantic_json, "high_quality_observation_count");
  semantic->high_quality_observation_mass =
      semantic_json.at("high_quality_observation_mass").get<float>();
  semantic->source_cameras =
      semantic_json.at("source_cameras").get<std::vector<std::string>>();
  semantic->observation_timestamps_ns =
      semantic_json.at("observation_timestamps_ns")
          .get<std::vector<TimeNanoseconds>>();
  semantic->label_weights =
      semantic_json.at("label_weights").get<std::map<std::string, float>>();
  semantic->semantic_weights =
      semanticWeightsFromJson(semantic_json.at("semantic_weights"));
  object->semantic = std::move(semantic);

  const Json& annotation_json = value.at("annotation");
  auto annotation = std::make_shared<AnnotationComponent>();
  annotation->revision = jsonUnsigned(annotation_json, "revision");
  if (!annotation_json.at("semantic_id_override").is_null()) {
    annotation->semantic_id_override =
        jsonInt(annotation_json, "semantic_id_override");
  }
  if (!annotation_json.at("label_override").is_null()) {
    annotation->label_override =
        annotation_json.at("label_override").get<std::string>();
  }
  if (!annotation_json.at("description_override").is_null()) {
    annotation->description_override =
        annotation_json.at("description_override").get<std::string>();
  }
  annotation->attributes = annotation_json.at("attributes")
                               .get<std::map<std::string, std::string>>();
  object->annotation = std::move(annotation);

  const Json& artifact_json = value.at("artifact");
  auto artifact = std::make_shared<ArtifactComponent>();
  artifact->revision = jsonUnsigned(artifact_json, "revision");
  artifact->appearance_revision =
      jsonUnsigned(artifact_json, "appearance_revision");
  for (const Json& snapshot : artifact_json.at("snapshots")) {
    artifact->snapshots.push_back(snapshotRefFromJson(snapshot));
  }
  artifact->snapshot_set_hash =
      artifact_json.at("snapshot_set_hash").get<std::string>();
  artifact->description =
      artifact_json.at("description").get<std::string>();
  artifact->description_input_hash =
      artifact_json.at("description_input_hash").get<std::string>();
  artifact->description_model_id =
      artifact_json.at("description_model_id").get<std::string>();
  artifact->description_schema_version =
      artifact_json.at("description_schema_version").get<std::string>();
  artifact->description_raw_text =
      artifact_json.value("description_raw_text", std::string());
  artifact->description_normalized_json =
      artifact_json.value("description_normalized_json", std::string());
  artifact->description_durable_envelope_json = artifact_json.value(
      "description_durable_envelope_json", std::string());
  artifact->description_parse_path =
      artifact_json.value("description_parse_path", std::string());
  if (artifact_json.contains("description_slo")) {
    artifact->description_slo =
        artifactSloFromJson(artifact_json.at("description_slo"));
  }
  artifact->description_stale =
      artifact_json.value("description_stale", false);
  artifact->pending_description_scene_revision =
      artifact_json.contains("pending_description_scene_revision")
          ? jsonUnsignedAs<SceneRevision>(
                artifact_json, "pending_description_scene_revision")
          : SceneRevision{0};
  artifact->pending_description =
      artifact_json.value("pending_description", std::string());
  artifact->pending_description_input_hash = artifact_json.value(
      "pending_description_input_hash", std::string());
  artifact->pending_description_model_id = artifact_json.value(
      "pending_description_model_id", std::string());
  artifact->pending_description_schema_version = artifact_json.value(
      "pending_description_schema_version", std::string());
  artifact->pending_description_raw_text = artifact_json.value(
      "pending_description_raw_text", std::string());
  artifact->pending_description_normalized_json = artifact_json.value(
      "pending_description_normalized_json", std::string());
  artifact->pending_description_durable_envelope_json = artifact_json.value(
      "pending_description_durable_envelope_json", std::string());
  artifact->pending_description_parse_path = artifact_json.value(
      "pending_description_parse_path", std::string());
  if (artifact_json.contains("pending_description_slo")) {
    artifact->pending_description_slo =
        artifactSloFromJson(artifact_json.at("pending_description_slo"));
  }
  object->artifact = std::move(artifact);
  return object;
}

SceneObjectPtr promoteRestoredDurableDescription(
    SceneObjectPtr object,
    SceneRevision durable_revision) {
  if (!object || !object->artifact ||
      object->artifact->pending_description_scene_revision == 0 ||
      object->artifact->pending_description_scene_revision >
          durable_revision) {
    return object;
  }
  if (object->artifact->revision ==
      std::numeric_limits<std::uint64_t>::max()) {
    throw StoreError("restored artifact revision is exhausted");
  }
  auto updated = std::make_shared<SceneObject>(*object);
  auto artifact =
      std::make_shared<ArtifactComponent>(*object->artifact);
  ++artifact->revision;
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
  updated->artifact = std::move(artifact);
  return updated;
}

Json trackToJson(const InstanceTrack& track) {
  Json quality_history = Json::array();
  for (const ObservationQualitySample& sample : track.observation_quality_history) {
    quality_history.push_back(
        Json{{"time_ns", sample.time_ns},
             {"quality", sample.quality},
             {"camera_distance_m", sample.camera_distance_m},
             {"high_quality", sample.high_quality}});
  }
  Json voxels = Json::array();
  for (const VoxelRef& voxel : track.near_surface_voxels) {
    voxels.push_back(voxelRefToJson(voxel));
  }
  return Json{
      {"track_id", track.track_id},
      {"object_id", track.object_id},
      {"state", static_cast<int>(track.state)},
      {"semantic_id", track.semantic_id},
      {"label", track.label},
      {"center_world", vector3ToJson(track.center_world)},
      {"size_m", vector3ToJson(track.size_m)},
      {"yaw_rad", track.yaw_rad},
      {"confidence", track.confidence},
      {"confidence_mass", track.confidence_mass},
      {"object_quality_score", track.object_quality_score},
      {"bbox_quality_mass", track.bbox_quality_mass},
      {"geometry_score", track.geometry_score},
      {"geometry_shell_ratio", track.geometry_shell_ratio},
      {"geometry_extent_score", track.geometry_extent_score},
      {"geometry_leak_ratio", track.geometry_leak_ratio},
      {"geometry_cavity_ratio", track.geometry_cavity_ratio},
      {"geometry_in_box_points", track.geometry_in_box_points},
      {"geometry_shell_points", track.geometry_shell_points},
      {"geometry_unique_voxels", track.geometry_unique_voxels},
      {"geometry_expanded_points", track.geometry_expanded_points},
      {"geometry_bad_count", track.geometry_bad_count},
      {"support_count", track.support_count},
      {"high_quality_observation_count", track.high_quality_observation_count},
      {"high_quality_observation_mass", track.high_quality_observation_mass},
      {"missed_count", track.missed_count},
      {"publishable", track.publishable},
      {"existence_log_odds", track.existence_log_odds},
      {"positive_evidence_timestamps_ns",
       track.positive_evidence_timestamps_ns},
      {"negative_evidence_timestamps_ns",
       track.negative_evidence_timestamps_ns},
      {"positive_window_interruptions",
       track.positive_window_interruptions},
      {"last_presence_evidence_ns", track.last_presence_evidence_ns},
      {"last_presence_evidence_reliability",
       track.last_presence_evidence_reliability},
      {"last_presence_evidence_reason",
       track.last_presence_evidence_reason},
      {"geometry_status", static_cast<int>(track.geometry_status)},
      {"first_seen_frame_index", track.first_seen_frame_index},
      {"last_seen_frame_index", track.last_seen_frame_index},
      {"obb_revision", track.obb_revision},
      {"geometry_evaluation_obb_revision",
       track.geometry_evaluation_obb_revision},
      {"geometry_evaluation_map_version",
       track.geometry_evaluation_map_version},
      {"last_geometry_check_ns", track.last_geometry_check_ns},
      {"first_seen_ns", track.first_seen_ns},
      {"last_seen_ns", track.last_seen_ns},
      {"geometry_evaluated_center_world",
       vector3ToJson(track.geometry_evaluated_center_world)},
      {"geometry_evaluated_size_m",
       vector3ToJson(track.geometry_evaluated_size_m)},
      {"geometry_evaluated_yaw_rad", track.geometry_evaluated_yaw_rad},
      {"geometry_evaluation_reason", track.geometry_evaluation_reason},
      {"source_cameras", track.source_cameras},
      {"observation_timestamps_ns", track.observation_timestamps_ns},
      {"observation_quality_history", std::move(quality_history)},
      {"snapshot", snapshotRefToJson(track.snapshot)},
      {"near_surface_voxels", std::move(voxels)},
      {"label_weights", track.label_weights},
      {"semantic_weights", semanticWeightsToJson(track.semantic_weights)},
      {"appearance_model_id", track.appearance_model_id},
      {"appearance_descriptor_shadow",
       track.appearance_descriptor_shadow},
  };
}

InstanceTrack trackFromJson(const Json& value) {
  InstanceTrack track;
  track.track_id = jsonInt(value, "track_id");
  track.object_id = jsonObjectId(value, "object_id", true);
  const int state = jsonInt(value, "state");
  if (state < static_cast<int>(InstanceTrackState::kTentative) ||
      state > static_cast<int>(InstanceTrackState::kInactive)) {
    throw StoreError("stored InstanceTrack state is invalid");
  }
  track.state = static_cast<InstanceTrackState>(state);
  track.semantic_id = jsonInt(value, "semantic_id");
  track.label = value.at("label").get<std::string>();
  track.center_world = vector3FromJson(value.at("center_world"));
  track.size_m = vector3FromJson(value.at("size_m"));
  track.yaw_rad = value.at("yaw_rad").get<float>();
  track.confidence = value.at("confidence").get<float>();
  track.confidence_mass = value.at("confidence_mass").get<float>();
  track.object_quality_score = value.at("object_quality_score").get<float>();
  track.bbox_quality_mass = value.at("bbox_quality_mass").get<float>();
  track.geometry_score = value.at("geometry_score").get<float>();
  track.geometry_shell_ratio = value.at("geometry_shell_ratio").get<float>();
  track.geometry_extent_score = value.at("geometry_extent_score").get<float>();
  track.geometry_leak_ratio = value.at("geometry_leak_ratio").get<float>();
  track.geometry_cavity_ratio = value.at("geometry_cavity_ratio").get<float>();
  track.geometry_in_box_points = jsonInt(value, "geometry_in_box_points");
  track.geometry_shell_points = jsonInt(value, "geometry_shell_points");
  track.geometry_unique_voxels =
      jsonInt(value, "geometry_unique_voxels");
  track.geometry_expanded_points =
      jsonInt(value, "geometry_expanded_points");
  track.geometry_bad_count = jsonInt(value, "geometry_bad_count");
  track.support_count = jsonInt(value, "support_count");
  track.high_quality_observation_count =
      jsonInt(value, "high_quality_observation_count");
  track.high_quality_observation_mass =
      value.at("high_quality_observation_mass").get<float>();
  track.missed_count = jsonInt(value, "missed_count");
  track.publishable = value.at("publishable").get<bool>();
  track.existence_log_odds = value.value(
      "existence_log_odds",
      track.state == InstanceTrackState::kStable
          ? 1.0986123f
          : (track.state == InstanceTrackState::kInactive ? -1.0986123f
                                                          : 0.0f));
  track.positive_evidence_timestamps_ns = value.value(
      "positive_evidence_timestamps_ns", std::vector<TimeNanoseconds>());
  track.negative_evidence_timestamps_ns = value.value(
      "negative_evidence_timestamps_ns", std::vector<TimeNanoseconds>());
  track.positive_window_interruptions =
      value.value("positive_window_interruptions", 0);
  track.last_presence_evidence_ns =
      value.value("last_presence_evidence_ns", TimeNanoseconds{0});
  track.last_presence_evidence_reliability =
      value.value("last_presence_evidence_reliability", 0.0f);
  track.last_presence_evidence_reason = value.value(
      "last_presence_evidence_reason", std::string("legacy_restore"));
  const int geometry_status = jsonInt(value, "geometry_status");
  if (geometry_status < static_cast<int>(InstanceGeometryStatus::kUnchecked) ||
      geometry_status > static_cast<int>(InstanceGeometryStatus::kEmpty)) {
    throw StoreError("stored InstanceTrack geometry status is invalid");
  }
  track.geometry_status = static_cast<InstanceGeometryStatus>(geometry_status);
  track.first_seen_frame_index =
      jsonUnsigned(value, "first_seen_frame_index");
  track.last_seen_frame_index =
      jsonUnsigned(value, "last_seen_frame_index");
  track.obb_revision = jsonUnsigned(value, "obb_revision");
  track.geometry_evaluation_obb_revision =
      jsonUnsigned(value, "geometry_evaluation_obb_revision");
  track.geometry_evaluation_map_version =
      jsonUnsigned(value, "geometry_evaluation_map_version");
  track.last_geometry_check_ns =
      jsonInt64(value, "last_geometry_check_ns", true);
  track.first_seen_ns = jsonInt64(value, "first_seen_ns", true);
  track.last_seen_ns = jsonInt64(value, "last_seen_ns", true);
  track.geometry_evaluated_center_world =
      vector3FromJson(value.at("geometry_evaluated_center_world"));
  track.geometry_evaluated_size_m =
      vector3FromJson(value.at("geometry_evaluated_size_m"));
  track.geometry_evaluated_yaw_rad =
      value.at("geometry_evaluated_yaw_rad").get<float>();
  track.geometry_evaluation_reason =
      value.at("geometry_evaluation_reason").get<std::string>();
  track.source_cameras =
      value.at("source_cameras").get<std::vector<std::string>>();
  const Json& observation_timestamps = value.at("observation_timestamps_ns");
  if (!observation_timestamps.is_array()) {
    throw StoreError("stored observation timestamps are not an array");
  }
  track.observation_timestamps_ns.reserve(observation_timestamps.size());
  for (const Json& timestamp : observation_timestamps) {
    track.observation_timestamps_ns.push_back(
        static_cast<TimeNanoseconds>(jsonInt64Value(
            timestamp, "observation timestamp", true)));
  }
  for (const Json& sample_json : value.at("observation_quality_history")) {
    ObservationQualitySample sample;
    sample.time_ns = static_cast<TimeNanoseconds>(
        jsonInt64(sample_json, "time_ns", true));
    sample.quality = sample_json.at("quality").get<float>();
    sample.camera_distance_m =
        sample_json.at("camera_distance_m").get<float>();
    sample.high_quality = sample_json.at("high_quality").get<bool>();
    track.observation_quality_history.push_back(sample);
  }
  track.snapshot = snapshotRefFromJson(value.at("snapshot"));
  for (const Json& voxel_json : value.at("near_surface_voxels")) {
    track.near_surface_voxels.push_back(voxelRefFromJson(voxel_json));
  }
  track.label_weights =
      value.at("label_weights").get<std::map<std::string, float>>();
  track.semantic_weights =
      semanticWeightsFromJson(value.at("semantic_weights"));
  track.appearance_model_id =
      value.value("appearance_model_id", std::string());
  track.appearance_descriptor_shadow = value.value(
      "appearance_descriptor_shadow", std::vector<float>());
  return track;
}

Json graphMetadataToJson(const SceneGraphMetadata& graph) {
  const auto entity_to_json = [](SceneEntityRef entity) {
    const char* type = "object";
    if (entity.type == SceneEntityType::kRoom) {
      type = "room";
    } else if (entity.type == SceneEntityType::kFurniture) {
      type = "furniture";
    }
    return Json{{"type", type},
                {"id", entity.id}};
  };
  Json rooms = Json::array();
  for (const RoomNode& room : graph.rooms) {
    rooms.push_back(
        Json{{"room_id", room.room_id},
             {"revision", room.revision},
             {"label", room.label},
             {"color", room.color},
             {"center_world", vector3ToJson(room.center_world)},
             {"size_m", vector3ToJson(room.size_m)},
             {"min_xy", room.min_xy},
             {"max_xy", room.max_xy},
             {"has_xy_bounds", room.has_xy_bounds},
             {"height_m", room.height_m},
             {"attributes", room.attributes}});
  }
  Json relations = Json::array();
  for (const ObjectRelation& relation : graph.relations) {
    relations.push_back(
        Json{{"source_object_id", relation.source_object_id},
             {"target_object_id", relation.target_object_id},
             {"source", entity_to_json(relationSource(relation))},
             {"target", entity_to_json(relationTarget(relation))},
             {"relation_type", relation.relation_type},
             {"confidence", relation.confidence},
             {"description", relation.description},
             {"revision", relation.revision},
             {"derived", relation.derived}});
  }
  Json furniture = Json::array();
  for (const FurnitureRole& role : graph.furniture) {
    furniture.push_back(
        Json{{"object_id", role.object_id},
             {"revision", role.revision},
             {"classification_label", role.classification_label}});
  }
  Json images = Json::array();
  for (const ObjectSnapshotImage& image : graph.snapshot_images) {
    images.push_back(
        Json{{"image_index", image.image_index},
             {"uri", image.uri},
             {"width", image.width},
             {"height", image.height},
             {"encoding", image.encoding},
             {"time_ns", image.time_ns},
             {"camera_id", image.camera_id},
             {"source_path", image.source_path}});
  }
  return Json{{"rooms", std::move(rooms)},
              {"furniture", std::move(furniture)},
              {"relations", std::move(relations)},
              {"snapshot_images", std::move(images)},
              {"import_warnings", graph.import_warnings},
              {"has_scene_graph_envelope", graph.has_scene_graph_envelope},
              {"scene_graph_json", graph.scene_graph_json}};
}

std::shared_ptr<const SceneGraphMetadata> graphMetadataFromJson(
    const Json& value) {
  auto graph = std::make_shared<SceneGraphMetadata>();
  const auto entity_from_json = [](const Json& entity,
                                   int legacy_object_id) {
    if (!entity.is_object()) {
      return SceneEntityRef{SceneEntityType::kObject, legacy_object_id};
    }
    const std::string type = entity.value("type", std::string("object"));
    const int entity_id = jsonIntOr(entity, "id", legacy_object_id);
    if (entity_id < -1) {
      throw StoreError("stored scene entity id is invalid");
    }
    SceneEntityType entity_type = SceneEntityType::kObject;
    if (type == "room") {
      entity_type = SceneEntityType::kRoom;
    } else if (type == "furniture") {
      entity_type = SceneEntityType::kFurniture;
    } else if (type != "object") {
      throw StoreError("stored scene entity type is invalid");
    }
    return SceneEntityRef{entity_type, entity_id};
  };
  for (const Json& room_json : value.value("rooms", Json::array())) {
    RoomNode room;
    room.room_id = jsonIntOr(room_json, "room_id", -1);
    if (room.room_id < -1) {
      throw StoreError("stored room id is invalid");
    }
    room.revision = jsonUnsignedOr(room_json, "revision", 0);
    room.label = room_json.value("label", std::string());
    room.color = room_json.value("color", std::string());
    room.center_world = vector3FromJson(
        room_json.value("center_world", Json::array({0.0, 0.0, 0.0})));
    room.size_m = vector3FromJson(
        room_json.value("size_m", Json::array({0.0, 0.0, 0.0})));
    room.min_xy = room_json.value("min_xy", std::array<float, 2>{0.0f, 0.0f});
    room.max_xy = room_json.value("max_xy", std::array<float, 2>{0.0f, 0.0f});
    room.has_xy_bounds = room_json.value("has_xy_bounds", false);
    room.height_m = room_json.value("height_m", 0.0f);
    room.attributes = room_json.value(
        "attributes", std::map<std::string, std::string>{});
    graph->rooms.push_back(std::move(room));
  }
  for (const Json& role_json :
       value.value("furniture", Json::array())) {
    FurnitureRole role;
    role.object_id = jsonIntOr(role_json, "object_id", -1);
    if (role.object_id < 0) {
      throw StoreError("stored furniture object id is invalid");
    }
    role.revision = jsonUnsignedOr(role_json, "revision", 0);
    role.classification_label =
        role_json.value("classification_label", std::string());
    if (role.classification_label.empty()) {
      throw StoreError("stored furniture classification label is empty");
    }
    graph->furniture.push_back(std::move(role));
  }
  for (const Json& relation_json :
       value.value("relations", Json::array())) {
    ObjectRelation relation;
    relation.source_object_id =
        jsonIntOr(relation_json, "source_object_id", -1);
    relation.target_object_id =
        jsonIntOr(relation_json, "target_object_id", -1);
    if (relation.source_object_id < -1 || relation.target_object_id < -1) {
      throw StoreError("stored relation object id is invalid");
    }
    relation.source = entity_from_json(
        relation_json.value("source", Json()), relation.source_object_id);
    relation.target = entity_from_json(
        relation_json.value("target", Json()), relation.target_object_id);
    relation.relation_type =
        relation_json.value("relation_type", std::string());
    relation.confidence = relation_json.value("confidence", 0.0f);
    relation.description =
        relation_json.value("description", std::string());
    relation.revision = jsonUnsignedOr(relation_json, "revision", 0);
    relation.derived = relation_json.value("derived", false);
    graph->relations.push_back(std::move(relation));
  }
  for (const Json& image_json :
       value.value("snapshot_images", Json::array())) {
    ObjectSnapshotImage image;
    image.image_index = jsonInt(image_json, "image_index");
    if (image.image_index < 0) {
      throw StoreError("stored scene image index is invalid");
    }
    image.uri = image_json.at("uri").get<std::string>();
    image.width = jsonInt(image_json, "width");
    image.height = jsonInt(image_json, "height");
    if (image.width < 0 || image.height < 0) {
      throw StoreError("stored scene image dimensions are invalid");
    }
    image.encoding = image_json.at("encoding").get<std::string>();
    image.time_ns = static_cast<TimeNanoseconds>(
        jsonInt64(image_json, "time_ns", true));
    image.camera_id = image_json.at("camera_id").get<std::string>();
    image.source_path = image_json.at("source_path").get<std::string>();
    graph->snapshot_images.push_back(std::move(image));
  }
  graph->import_warnings = value.value(
      "import_warnings", std::vector<std::string>{});
  graph->has_scene_graph_envelope =
      value.value("has_scene_graph_envelope", false);
  graph->scene_graph_json =
      value.value("scene_graph_json", std::string());
  return graph;
}

std::string serializeSnapshot(const SceneSnapshot& snapshot) {
  Json objects = Json::array();
  for (const auto& [object_id, object] : snapshot.objects()) {
    if (!object || !object->identity || object->identity->object_id != object_id) {
      throw StoreError("scene object table contains an invalid identity");
    }
    objects.push_back(
        Json{{"object_id", object_id}, {"object", objectToJson(*object)}});
  }
  Json aliases = Json::array();
  for (const auto& [retired_id, alias] : snapshot.aliases()) {
    if (retired_id != alias.retired_object_id) {
      throw StoreError("scene alias table key does not match alias payload");
    }
    aliases.push_back(
        Json{{"retired_object_id", alias.retired_object_id},
             {"canonical_object_id", alias.canonical_object_id},
             {"created_revision", alias.created_revision}});
  }
  Json tombstones = Json::array();
  for (const auto& [object_id, tombstone] : snapshot.tombstones()) {
    if (object_id != tombstone.object_id) {
      throw StoreError("scene tombstone table key does not match payload");
    }
    tombstones.push_back(
        Json{{"object_id", tombstone.object_id},
             {"deleted_revision", tombstone.deleted_revision},
             {"reason", tombstone.reason}});
  }
  Json tracks = Json::array();
  for (const auto& [track_id, track] : snapshot.tracks()) {
    if (!track || track_id != track->track_id) {
      throw StoreError("scene track table key does not match payload");
    }
    tracks.push_back(trackToJson(*track));
  }
  Json recent_observations = Json::array();
  for (const FrameKey& key :
       snapshot.statePtr()->recent_observation_frames) {
    recent_observations.push_back(
        Json{{"run_id", runIdToJson(key.run_id)},
             {"frame_id", key.frame_id}});
  }
  Json observation_watermarks = Json::array();
  for (const ObservationRunWatermark& watermark :
       snapshot.statePtr()->observation_watermarks) {
    observation_watermarks.push_back(
        Json{{"run_id", runIdToJson(watermark.run_id)},
             {"highest_frame_id", watermark.highest_frame_id}});
  }
  Json payload{
      {"format_version", 1},
      {"latest_scene_revision", snapshot.revision()},
      {"durable_scene_revision", snapshot.durableRevision()},
      {"next_object_id", snapshot.nextObjectId()},
      {"latest_surface", surfaceStampToJson(snapshot.latestSurface())},
      {"shutdown_requested", snapshot.shutdownRequested()},
      {"recent_observation_frames", std::move(recent_observations)},
      {"observation_watermarks", std::move(observation_watermarks)},
      {"objects", std::move(objects)},
      {"aliases", std::move(aliases)},
      {"tombstones", std::move(tombstones)},
      {"tracks", std::move(tracks)},
      {"graph", graphMetadataToJson(snapshot.graphMetadata())},
  };
  return payload.dump();
}

SceneSnapshot deserializeSnapshot(const std::string& payload,
                                  SceneRevision durable_revision) {
  const Json value = Json::parse(payload);
  if (jsonInt(value, "format_version") != 1) {
    throw StoreError("unsupported scene snapshot payload version");
  }
  const SceneRevision stored_revision =
      jsonUnsignedAs<SceneRevision>(value, "latest_scene_revision");
  if (stored_revision != durable_revision) {
    throw StoreError("scene payload revision does not match durable watermark");
  }

  auto state = std::make_shared<SceneState>();
  state->latest_scene_revision = durable_revision;
  state->durable_scene_revision = durable_revision;
  state->next_object_id = jsonObjectId(value, "next_object_id");
  // Object-level evaluation stamps remain historical provenance, but a
  // restarted process has not restored/pinned the old process's live map.
  // The map actor must publish a fresh epoch before new geometry can commit.
  (void)surfaceStampFromJson(value.at("latest_surface"));
  state->latest_surface = SurfaceStamp{};
  // A restored process starts active even if the old process persisted a
  // shutdown metadata snapshot.
  state->shutdown_requested = false;
  for (const Json& item :
       value.value("recent_observation_frames", Json::array())) {
    state->recent_observation_frames.push_back(
        FrameKey{runIdFromJson(item.at("run_id")),
                 jsonUnsignedAs<FrameId>(item, "frame_id")});
  }
  for (const Json& item :
       value.value("observation_watermarks", Json::array())) {
    state->observation_watermarks.push_back(
        ObservationRunWatermark{
            runIdFromJson(item.at("run_id")),
            jsonUnsignedAs<FrameId>(item, "highest_frame_id")});
  }

  SceneObjectTable objects;
  for (const Json& item : value.at("objects")) {
    const SceneObjectId object_id = jsonObjectId(item, "object_id");
    SceneObjectPtr object = promoteRestoredDurableDescription(
        objectFromJson(item.at("object")), durable_revision);
    if (!object || !object->identity || object->identity->object_id != object_id ||
        !objects.emplace(object_id, std::move(object)).second) {
      throw StoreError("stored object table is inconsistent");
    }
  }
  state->objects = std::make_shared<const SceneObjectTable>(std::move(objects));

  SceneAliasTable aliases;
  for (const Json& item : value.at("aliases")) {
    ObjectAlias alias;
    alias.retired_object_id = jsonObjectId(item, "retired_object_id");
    alias.canonical_object_id = jsonObjectId(item, "canonical_object_id");
    alias.created_revision =
        jsonUnsignedAs<SceneRevision>(item, "created_revision");
    if (!aliases.emplace(alias.retired_object_id, alias).second) {
      throw StoreError("stored alias table contains duplicates");
    }
  }
  state->aliases = std::make_shared<const SceneAliasTable>(std::move(aliases));

  SceneTombstoneTable tombstones;
  for (const Json& item : value.at("tombstones")) {
    ObjectTombstone tombstone;
    tombstone.object_id = jsonObjectId(item, "object_id");
    tombstone.deleted_revision =
        jsonUnsignedAs<SceneRevision>(item, "deleted_revision");
    tombstone.reason = item.at("reason").get<std::string>();
    if (!tombstones.emplace(tombstone.object_id, tombstone).second) {
      throw StoreError("stored tombstone table contains duplicates");
    }
  }
  state->tombstones =
      std::make_shared<const SceneTombstoneTable>(std::move(tombstones));

  SceneTrackTable tracks;
  for (const Json& item : value.at("tracks")) {
    InstanceTrack track = trackFromJson(item);
    const int track_id = track.track_id;
    if (!tracks
             .emplace(track_id,
                      std::make_shared<const InstanceTrack>(std::move(track)))
             .second) {
      throw StoreError("stored track table contains duplicates");
    }
  }
  state->tracks = std::make_shared<const SceneTrackTable>(std::move(tracks));
  state->graph = graphMetadataFromJson(value.at("graph"));
  std::set<int> furniture_ids;
  for (const FurnitureRole& role : state->graph->furniture) {
    if (state->objects->count(role.object_id) == 0U ||
        !furniture_ids.insert(role.object_id).second) {
      throw StoreError(
          "stored furniture roles do not reference unique live objects");
    }
  }
  std::set<int> room_ids;
  for (const RoomNode& room : state->graph->rooms) {
    room_ids.insert(room.room_id);
  }
  const auto valid_endpoint = [&](SceneEntityRef endpoint) {
    if (endpoint.type == SceneEntityType::kRoom) {
      return room_ids.count(endpoint.id) != 0U;
    }
    if (endpoint.type == SceneEntityType::kFurniture) {
      return furniture_ids.count(endpoint.id) != 0U;
    }
    return state->objects->count(endpoint.id) != 0U;
  };
  for (const SceneRelation& relation : state->graph->relations) {
    if (!valid_endpoint(relationSource(relation)) ||
        !valid_endpoint(relationTarget(relation))) {
      throw StoreError("stored relation references a missing scene entity");
    }
  }
  return SceneSnapshot(std::move(state));
}

std::map<std::int64_t, Json> indexJsonTable(const Json& table,
                                            const char* key_name) {
  if (!table.is_array()) {
    throw StoreError(std::string("scene payload table is not an array: ") +
                     key_name);
  }
  std::map<std::int64_t, Json> indexed;
  for (const Json& item : table) {
    if (!item.is_object() || !item.contains(key_name)) {
      throw StoreError(std::string("scene payload table row has no key: ") +
                       key_name);
    }
    const std::int64_t key = jsonInt64(item, key_name, true);
    if (!indexed.emplace(key, item).second) {
      throw StoreError(std::string("scene payload table has duplicate key: ") +
                       key_name);
    }
  }
  return indexed;
}

Json buildJsonTableDelta(const Json& previous,
                         const Json& current,
                         const char* key_name) {
  const std::map<std::int64_t, Json> previous_rows =
      indexJsonTable(previous, key_name);
  const std::map<std::int64_t, Json> current_rows =
      indexJsonTable(current, key_name);
  Json upserts = Json::array();
  Json erases = Json::array();
  for (const auto& [key, row] : current_rows) {
    const auto old = previous_rows.find(key);
    if (old == previous_rows.end() || old->second != row) {
      upserts.push_back(row);
    }
  }
  for (const auto& [key, row] : previous_rows) {
    (void)row;
    if (current_rows.count(key) == 0) {
      erases.push_back(key);
    }
  }
  Json delta = Json::object();
  if (!upserts.empty()) {
    delta["upsert"] = std::move(upserts);
  }
  if (!erases.empty()) {
    delta["erase"] = std::move(erases);
  }
  return delta;
}

void applyJsonTableDelta(Json* checkpoint,
                         const Json& delta,
                         const char* table_name,
                         const char* key_name) {
  if (!checkpoint || !delta.is_object()) {
    throw StoreError("stored scene table delta is invalid");
  }
  std::map<std::int64_t, Json> rows =
      indexJsonTable(checkpoint->at(table_name), key_name);
  for (const Json& key_json : delta.value("erase", Json::array())) {
    rows.erase(jsonInt64Value(key_json, "scene table erase key", true));
  }
  for (const Json& row : delta.value("upsert", Json::array())) {
    if (!row.is_object() || !row.contains(key_name)) {
      throw StoreError("stored scene table upsert row has no key");
    }
    rows[jsonInt64(row, key_name, true)] = row;
  }
  Json materialized = Json::array();
  for (auto& [key, row] : rows) {
    (void)key;
    materialized.push_back(std::move(row));
  }
  (*checkpoint)[table_name] = std::move(materialized);
}

std::string serializeSnapshotDelta(const std::string& previous_payload,
                                   const std::string& current_payload) {
  // Payload format v1 remains the backward-compatible full checkpoint. The
  // self-describing v2 delta needs no SQLite schema migration, so databases
  // produced by older binaries restore as a chain containing only checkpoints.
  const Json previous = Json::parse(previous_payload);
  const Json current = Json::parse(current_payload);
  if (jsonInt(previous, "format_version") != 1 ||
      jsonInt(current, "format_version") != 1) {
    throw StoreError("scene delta requires full snapshot inputs");
  }
  const SceneRevision base_revision =
      jsonUnsignedAs<SceneRevision>(previous, "latest_scene_revision");
  const SceneRevision revision =
      jsonUnsignedAs<SceneRevision>(current, "latest_scene_revision");
  if (base_revision == std::numeric_limits<SceneRevision>::max() ||
      revision != base_revision + 1) {
    throw StoreError("scene delta inputs are not contiguous");
  }

  Json replacements = Json::object();
  constexpr std::array<const char*, 6> kReplaceFields{
      "durable_scene_revision", "next_object_id", "latest_surface",
      "shutdown_requested", "recent_observation_frames",
      "observation_watermarks"};
  for (const char* field : kReplaceFields) {
    if (previous.at(field) != current.at(field)) {
      replacements[field] = current.at(field);
    }
  }

  Json tables = Json::object();
  constexpr std::array<std::pair<const char*, const char*>, 4> kTables{{
      {"objects", "object_id"},
      {"aliases", "retired_object_id"},
      {"tombstones", "object_id"},
      {"tracks", "track_id"},
  }};
  for (const auto& [table_name, key_name] : kTables) {
    Json table_delta = buildJsonTableDelta(previous.at(table_name),
                                           current.at(table_name), key_name);
    if (!table_delta.empty()) {
      tables[table_name] = std::move(table_delta);
    }
  }

  Json delta{{"format_version", 2},
             {"payload_kind", "delta"},
             {"base_revision", base_revision},
             {"latest_scene_revision", revision}};
  if (!replacements.empty()) {
    delta["replace"] = std::move(replacements);
  }
  if (!tables.empty()) {
    delta["tables"] = std::move(tables);
  }
  const Json graph_patch =
      Json::diff(previous.at("graph"), current.at("graph"));
  if (!graph_patch.empty()) {
    delta["graph_patch"] = graph_patch;
  }
  return delta.dump();
}

Json applySnapshotDelta(Json checkpoint,
                        const Json& delta,
                        SceneRevision stored_revision) {
  if (jsonInt(checkpoint, "format_version") != 1 ||
      jsonInt(delta, "format_version") != 2 ||
      delta.value("payload_kind", std::string()) != "delta") {
    throw StoreError("stored scene delta has an unsupported format");
  }
  const SceneRevision base_revision = jsonUnsignedAs<SceneRevision>(
      checkpoint, "latest_scene_revision");
  if (jsonUnsignedAs<SceneRevision>(delta, "base_revision") != base_revision ||
      base_revision == std::numeric_limits<SceneRevision>::max() ||
      stored_revision != base_revision + 1 ||
      jsonUnsignedAs<SceneRevision>(delta, "latest_scene_revision") !=
          stored_revision) {
    throw StoreError("stored scene delta chain is not contiguous");
  }

  const Json replacements = delta.value("replace", Json::object());
  if (!replacements.is_object()) {
    throw StoreError("stored scene scalar delta is invalid");
  }
  const std::set<std::string> allowed_replacements{
      "durable_scene_revision", "next_object_id", "latest_surface",
      "shutdown_requested", "recent_observation_frames",
      "observation_watermarks"};
  for (const auto& [field, value] : replacements.items()) {
    if (allowed_replacements.count(field) == 0) {
      throw StoreError("stored scene delta replaces an unsupported field");
    }
    checkpoint[field] = value;
  }

  const Json tables = delta.value("tables", Json::object());
  if (!tables.is_object()) {
    throw StoreError("stored scene table deltas are invalid");
  }
  constexpr std::array<std::pair<const char*, const char*>, 4> kTables{{
      {"objects", "object_id"},
      {"aliases", "retired_object_id"},
      {"tombstones", "object_id"},
      {"tracks", "track_id"},
  }};
  std::set<std::string> known_tables;
  for (const auto& [table_name, key_name] : kTables) {
    known_tables.insert(table_name);
    if (tables.contains(table_name)) {
      applyJsonTableDelta(&checkpoint, tables.at(table_name), table_name,
                          key_name);
    }
  }
  for (const auto& [table_name, table_delta] : tables.items()) {
    (void)table_delta;
    if (known_tables.count(table_name) == 0) {
      throw StoreError("stored scene delta contains an unsupported table");
    }
  }

  if (delta.contains("graph_patch")) {
    if (!delta.at("graph_patch").is_array()) {
      throw StoreError("stored scene graph patch is invalid");
    }
    checkpoint["graph"] =
        checkpoint.at("graph").patch(delta.at("graph_patch"));
  }
  checkpoint["latest_scene_revision"] = stored_revision;
  return checkpoint;
}

std::string payloadHash(const std::string& payload) {
  // A corruption sentinel, not a security primitive. SQLite handles page
  // integrity; this detects mismatched/manual payload edits deterministically.
  std::uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char byte : payload) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << hash;
  return stream.str();
}

sqlite3_int64 sqliteRevision(SceneRevision revision) {
  if (revision >
      static_cast<SceneRevision>(std::numeric_limits<sqlite3_int64>::max())) {
    throw StoreError("scene revision exceeds SQLite INTEGER range");
  }
  return static_cast<sqlite3_int64>(revision);
}

SceneRevision sceneRevision(sqlite3_int64 revision) {
  if (revision < 0) {
    throw StoreError("stored scene revision is negative");
  }
  return static_cast<SceneRevision>(revision);
}

std::int64_t unixTimeMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

RunId storedRunId(const std::string& encoded) {
  if (encoded.size() != 32U ||
      !std::all_of(encoded.begin(), encoded.end(), [](unsigned char value) {
        return std::isxdigit(value) != 0;
      })) {
    throw StoreError("stored map checkpoint epoch is invalid");
  }
  try {
    RunId result;
    result.high = std::stoull(encoded.substr(0, 16), nullptr, 16);
    result.low = std::stoull(encoded.substr(16, 16), nullptr, 16);
    if (!result.valid()) {
      throw StoreError("stored map checkpoint epoch is zero");
    }
    return result;
  } catch (const StoreError&) {
    throw;
  } catch (const std::exception&) {
    throw StoreError("stored map checkpoint epoch is invalid");
  }
}

void validateMapCheckpointManifest(
    const MapCheckpointManifest& manifest,
    SceneRevision durable_scene_revision) {
  if (manifest.backend.empty() || manifest.checkpoint_path.empty() ||
      manifest.world_frame.empty() || manifest.config_fingerprint.empty() ||
      !manifest.map_epoch.valid() || manifest.file_size_bytes == 0 ||
      manifest.content_hash.empty() || manifest.created_at_unix_ms <= 0) {
    throw StoreError("map checkpoint manifest is incomplete");
  }
  if (manifest.integrated_through_ns < 0) {
    throw StoreError("map checkpoint integrated timestamp is negative");
  }
  if (manifest.aligned_scene_revision > durable_scene_revision) {
    throw StoreError(
        "map checkpoint references a non-durable scene revision");
  }
  (void)sqliteRevision(manifest.map_revision);
  (void)sqliteRevision(manifest.aligned_scene_revision);
  (void)sqliteRevision(manifest.file_size_bytes);
}

void validateArtifactOrigin(const DurableArtifactOrigin& origin,
                            SceneRevision durable_limit) {
  if (origin.object_id < 0 || origin.scene_revision == 0 ||
      origin.scene_revision > durable_limit || origin.created_unix_ms < 0 ||
      (origin.priority != ArtifactPriority::kInteractive &&
       origin.priority != ArtifactPriority::kBulk)) {
    throw StoreError("durable artifact origin is invalid");
  }
  (void)sqliteRevision(origin.scene_revision);
}

void validateSnapshotSqlRange(const SceneSnapshot& snapshot) {
  (void)sqliteRevision(snapshot.revision());
  for (const auto& [object_id, object] : snapshot.objects()) {
    (void)object_id;
    if (!object) {
      throw StoreError("scene object table contains a null object");
    }
    const ComponentRevisions revisions = object->revisions();
    (void)sqliteRevision(revisions.identity_revision);
    (void)sqliteRevision(revisions.lifecycle_revision);
    (void)sqliteRevision(revisions.geometry_revision);
    (void)sqliteRevision(revisions.semantic_revision);
    (void)sqliteRevision(revisions.annotation_revision);
    (void)sqliteRevision(revisions.artifact_revision);
  }
  for (const auto& [retired_id, alias] : snapshot.aliases()) {
    (void)retired_id;
    (void)sqliteRevision(alias.created_revision);
  }
  for (const auto& [object_id, tombstone] : snapshot.tombstones()) {
    (void)object_id;
    (void)sqliteRevision(tombstone.deleted_revision);
  }
}

bool tasksEqual(const DurableTaskSpec& lhs, const DurableTaskSpec& rhs) {
  return lhs.task_id == rhs.task_id && lhs.dedupe_key == rhs.dedupe_key &&
         lhs.task_type == rhs.task_type && lhs.payload == rhs.payload &&
         lhs.scene_revision == rhs.scene_revision &&
         lhs.not_before_unix_ms == rhs.not_before_unix_ms;
}

void validateTask(const DurableTaskSpec& task, SceneRevision durable_limit) {
  if (task.task_id.empty() || task.dedupe_key.empty() || task.task_type.empty()) {
    throw StoreError("durable task id, dedupe key, and type must be non-empty");
  }
  if (task.scene_revision == 0 || task.scene_revision > durable_limit) {
    throw StoreError("durable task depends on a non-durable scene revision");
  }
  (void)sqliteRevision(task.scene_revision);
}

struct DurableTaskSchedulingKey {
  // Lower values win for every field.
  int overdue_rank = 1;
  int priority_rank = 1;
  std::int64_t due_unix_ms = std::numeric_limits<std::int64_t>::max();
  SceneRevision scene_revision = 0;
  std::string task_id;
};

std::optional<int> schedulingPriority(const Json& value) {
  if (!value.is_object()) {
    return std::nullopt;
  }
  const auto priority = value.find("priority");
  if (priority == value.end() || !priority->is_string()) {
    return std::nullopt;
  }
  const std::string name = priority->get<std::string>();
  if (name == "interactive") {
    return 0;
  }
  if (name == "bulk") {
    return 1;
  }
  return std::nullopt;
}

std::optional<std::int64_t> schedulingDueTime(const Json& value) {
  if (!value.is_object()) {
    return std::nullopt;
  }
  const auto due = value.find("due_unix_ms");
  if (due == value.end()) {
    return std::nullopt;
  }
  try {
    std::int64_t parsed = 0;
    if (due->is_number_unsigned()) {
      const std::uint64_t raw = due->get<std::uint64_t>();
      if (raw > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
        return std::nullopt;
      }
      parsed = static_cast<std::int64_t>(raw);
    } else if (due->is_number_integer()) {
      parsed = due->get<std::int64_t>();
    } else {
      return std::nullopt;
    }
    return parsed > 0 ? std::optional<std::int64_t>(parsed) : std::nullopt;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

DurableTaskSchedulingKey schedulingKey(const DurableTaskRecord& task,
                                       std::int64_t now_unix_ms) {
  DurableTaskSchedulingKey key;
  key.scene_revision = task.task.scene_revision;
  key.task_id = task.task.task_id;

  const Json payload = Json::parse(task.task.payload, nullptr, false);
  if (payload.is_discarded() || !payload.is_object()) {
    return key;
  }
  const auto artifact_slo = payload.find("artifact_slo");
  const Json* nested = artifact_slo != payload.end() && artifact_slo->is_object()
                           ? &*artifact_slo
                           : nullptr;

  std::optional<int> priority = schedulingPriority(payload);
  if (!priority && nested != nullptr) {
    priority = schedulingPriority(*nested);
  }
  if (priority) {
    key.priority_rank = *priority;
  }

  std::optional<std::int64_t> due = schedulingDueTime(payload);
  if (!due && nested != nullptr) {
    due = schedulingDueTime(*nested);
  }
  if (due) {
    key.due_unix_ms = *due;
    // Match runtime SLO accounting: work started strictly after its due time
    // is overdue and is promoted ahead of every non-overdue task.
    key.overdue_rank = now_unix_ms > *due ? 0 : 1;
  }
  return key;
}

bool schedulingKeyLess(const DurableTaskSchedulingKey& lhs,
                       const DurableTaskSchedulingKey& rhs) {
  return std::tie(lhs.overdue_rank, lhs.priority_rank, lhs.due_unix_ms,
                  lhs.scene_revision, lhs.task_id) <
         std::tie(rhs.overdue_rank, rhs.priority_rank, rhs.due_unix_ms,
                  rhs.scene_revision, rhs.task_id);
}

}  // namespace

namespace {

std::string sqliteMessage(sqlite3* database, const std::string& operation) {
  return operation + ": " +
         (database ? sqlite3_errmsg(database) : "SQLite handle is null");
}

void execSql(sqlite3* database, const char* sql) {
  char* error_message = nullptr;
  const int rc = sqlite3_exec(database, sql, nullptr, nullptr, &error_message);
  if (rc == SQLITE_OK) {
    return;
  }
  std::string error = error_message ? error_message : sqlite3_errmsg(database);
  sqlite3_free(error_message);
  throw StoreError("SQLite exec failed: " + error);
}

class Statement {
 public:
  Statement(sqlite3* database, const char* sql) : database_(database) {
    const int rc = sqlite3_prepare_v2(database, sql, -1, &statement_, nullptr);
    if (rc != SQLITE_OK) {
      throw StoreError(sqliteMessage(database, "SQLite prepare failed"));
    }
  }

  ~Statement() { sqlite3_finalize(statement_); }
  Statement(const Statement&) = delete;
  Statement& operator=(const Statement&) = delete;

  void bindInt(int index, int value) {
    check(sqlite3_bind_int(statement_, index, value), "bind int");
  }

  void bindInt64(int index, sqlite3_int64 value) {
    check(sqlite3_bind_int64(statement_, index, value), "bind int64");
  }

  void bindText(int index, const std::string& value) {
    check(sqlite3_bind_text(statement_, index, value.data(),
                            static_cast<int>(value.size()), SQLITE_TRANSIENT),
          "bind text");
  }

  void bindBlob(int index, const std::vector<std::uint8_t>& value) {
    if (value.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw StoreError("SQLite blob exceeds the supported byte count");
    }
    check(sqlite3_bind_blob(statement_, index, value.data(),
                            static_cast<int>(value.size()), SQLITE_TRANSIENT),
          "bind blob");
  }

  void bindNull(int index) {
    check(sqlite3_bind_null(statement_, index), "bind null");
  }

  bool stepRow() {
    const int rc = sqlite3_step(statement_);
    if (rc == SQLITE_ROW) {
      return true;
    }
    if (rc == SQLITE_DONE) {
      return false;
    }
    throw StoreError(sqliteMessage(database_, "SQLite step failed"));
  }

  void stepDone() {
    const int rc = sqlite3_step(statement_);
    if (rc != SQLITE_DONE) {
      throw StoreError(sqliteMessage(database_, "SQLite statement failed"));
    }
  }

  void reset() {
    check(sqlite3_reset(statement_), "reset");
    check(sqlite3_clear_bindings(statement_), "clear bindings");
  }

  int columnInt(int index) const {
    return sqlite3_column_int(statement_, index);
  }

  sqlite3_int64 columnInt64(int index) const {
    return sqlite3_column_int64(statement_, index);
  }

  std::string columnText(int index) const {
    const unsigned char* text = sqlite3_column_text(statement_, index);
    if (!text) {
      return {};
    }
    return std::string(reinterpret_cast<const char*>(text),
                       static_cast<std::size_t>(
                           sqlite3_column_bytes(statement_, index)));
  }

  int columnType(int index) const {
    return sqlite3_column_type(statement_, index);
  }

  std::vector<std::uint8_t> columnBlob(int index) const {
    const int byte_count = sqlite3_column_bytes(statement_, index);
    if (byte_count < 0) {
      throw StoreError("SQLite returned a negative blob byte count");
    }
    const void* bytes = sqlite3_column_blob(statement_, index);
    if (byte_count != 0 && bytes == nullptr) {
      throw StoreError("SQLite returned a null non-empty blob");
    }
    const auto* begin = static_cast<const std::uint8_t*>(bytes);
    return byte_count == 0
               ? std::vector<std::uint8_t>{}
               : std::vector<std::uint8_t>(
                     begin, begin + static_cast<std::size_t>(byte_count));
  }

 private:
  void check(int rc, const char* operation) {
    if (rc != SQLITE_OK) {
      throw StoreError(sqliteMessage(database_,
                                     std::string("SQLite ") + operation +
                                         " failed"));
    }
  }

  sqlite3* database_ = nullptr;
  sqlite3_stmt* statement_ = nullptr;
};

std::string materializeDurableSnapshotPayload(
    sqlite3* database, SceneRevision durable_revision) {
  if (durable_revision == 0) {
    return {};
  }
  Statement revisions(
      database,
      "SELECT revision, payload, payload_hash FROM scene_revisions "
      "WHERE revision <= ?1 ORDER BY revision DESC LIMIT ?2;");
  revisions.bindInt64(1, sqliteRevision(durable_revision));
  revisions.bindInt64(
      2, sqliteRevision(SceneStore::kRevisionCheckpointInterval));

  Json checkpoint;
  SceneRevision checkpoint_revision = 0;
  std::vector<std::pair<SceneRevision, Json>> reverse_deltas;
  while (revisions.stepRow()) {
    if (revisions.columnType(0) != SQLITE_INTEGER ||
        revisions.columnType(1) != SQLITE_TEXT ||
        revisions.columnType(2) != SQLITE_TEXT) {
      throw StoreError("stored scene revision columns have invalid types");
    }
    const SceneRevision row_revision =
        sceneRevision(revisions.columnInt64(0));
    const std::string payload = revisions.columnText(1);
    if (payloadHash(payload) != revisions.columnText(2)) {
      throw StoreError("scene revision payload hash mismatch");
    }
    const Json value = Json::parse(payload);
    const int format_version = jsonInt(value, "format_version");
    if (format_version == 1) {
      if (jsonUnsignedAs<SceneRevision>(value, "latest_scene_revision") !=
          row_revision) {
        throw StoreError("scene checkpoint revision does not match its row");
      }
      checkpoint = value;
      checkpoint_revision = row_revision;
      break;
    }
    if (format_version != 2 ||
        jsonUnsignedAs<SceneRevision>(value, "latest_scene_revision") !=
            row_revision) {
      throw StoreError("stored scene revision has an unsupported payload");
    }
    reverse_deltas.emplace_back(row_revision, value);
  }
  if (checkpoint_revision == 0) {
    throw StoreError(
        "durable scene revision has no bounded recovery checkpoint");
  }

  for (auto it = reverse_deltas.rbegin(); it != reverse_deltas.rend(); ++it) {
    checkpoint = applySnapshotDelta(std::move(checkpoint), it->second,
                                    it->first);
  }
  if (jsonUnsignedAs<SceneRevision>(checkpoint, "latest_scene_revision") !=
      durable_revision) {
    throw StoreError("durable scene delta chain does not reach its watermark");
  }
  return checkpoint.dump();
}

constexpr const char* kMapCheckpointColumns =
    "backend, checkpoint_path, world_frame, config_fingerprint, "
    "map_epoch, map_revision, "
    "integrated_through_ns, aligned_scene_revision, file_size_bytes, "
    "content_hash, created_at_unix_ms";

MapCheckpointManifest mapCheckpointFromStatement(
    const Statement& statement,
    int first_column,
    SceneRevision durable_scene_revision) {
  for (int index = 0; index < 11; ++index) {
    const bool expected_text =
        index == 0 || index == 1 || index == 2 || index == 3 ||
        index == 4 || index == 9;
    const int expected_type = expected_text ? SQLITE_TEXT : SQLITE_INTEGER;
    if (statement.columnType(first_column + index) != expected_type) {
      throw StoreError(
          "stored map checkpoint manifest has invalid SQLite column types");
    }
  }
  MapCheckpointManifest manifest;
  manifest.backend = statement.columnText(first_column + 0);
  manifest.checkpoint_path = statement.columnText(first_column + 1);
  manifest.world_frame = statement.columnText(first_column + 2);
  manifest.config_fingerprint = statement.columnText(first_column + 3);
  manifest.map_epoch = storedRunId(statement.columnText(first_column + 4));
  manifest.map_revision =
      sceneRevision(statement.columnInt64(first_column + 5));
  manifest.integrated_through_ns =
      statement.columnInt64(first_column + 6);
  manifest.aligned_scene_revision =
      sceneRevision(statement.columnInt64(first_column + 7));
  manifest.file_size_bytes =
      sceneRevision(statement.columnInt64(first_column + 8));
  manifest.content_hash = statement.columnText(first_column + 9);
  manifest.created_at_unix_ms = statement.columnInt64(first_column + 10);
  validateMapCheckpointManifest(manifest, durable_scene_revision);
  return manifest;
}

bool mapCheckpointManifestsEqual(const MapCheckpointManifest& lhs,
                                 const MapCheckpointManifest& rhs) {
  return lhs.backend == rhs.backend &&
         lhs.checkpoint_path == rhs.checkpoint_path &&
         lhs.world_frame == rhs.world_frame &&
         lhs.config_fingerprint == rhs.config_fingerprint &&
         lhs.map_epoch == rhs.map_epoch &&
         lhs.map_revision == rhs.map_revision &&
         lhs.integrated_through_ns == rhs.integrated_through_ns &&
         lhs.aligned_scene_revision == rhs.aligned_scene_revision &&
         lhs.file_size_bytes == rhs.file_size_bytes &&
         lhs.content_hash == rhs.content_hash &&
         lhs.created_at_unix_ms == rhs.created_at_unix_ms;
}

void validateMapCheckpointTable(sqlite3* database,
                                SceneRevision durable_scene_revision) {
  const std::string sql = std::string("SELECT ") + kMapCheckpointColumns +
                          " FROM map_checkpoint_manifest "
                          "ORDER BY checkpoint_id;";
  Statement statement(database, sql.c_str());
  while (statement.stepRow()) {
    (void)mapCheckpointFromStatement(statement, 0,
                                     durable_scene_revision);
  }
}

void validateArtifactOriginTable(sqlite3* database,
                                 SceneRevision durable_scene_revision) {
  Statement statement(
      database,
      "SELECT object_id, scene_revision, created_unix_ms, priority "
      "FROM artifact_origins ORDER BY object_id;");
  while (statement.stepRow()) {
    for (int index = 0; index < 4; ++index) {
      if (statement.columnType(index) != SQLITE_INTEGER) {
        throw StoreError(
            "stored artifact origin has invalid SQLite column types");
      }
    }
    DurableArtifactOrigin origin;
    origin.object_id = statement.columnInt(0);
    origin.scene_revision = sceneRevision(statement.columnInt64(1));
    origin.created_unix_ms = statement.columnInt64(2);
    const int priority = statement.columnInt(3);
    if (priority == static_cast<int>(ArtifactPriority::kInteractive)) {
      origin.priority = ArtifactPriority::kInteractive;
    } else if (priority == static_cast<int>(ArtifactPriority::kBulk)) {
      origin.priority = ArtifactPriority::kBulk;
    } else {
      throw StoreError("stored artifact origin priority is invalid");
    }
    validateArtifactOrigin(origin, durable_scene_revision);
  }
}

class Transaction {
 public:
  explicit Transaction(sqlite3* database) : database_(database) {
    execSql(database_, "BEGIN IMMEDIATE;");
  }

  ~Transaction() {
    if (!committed_) {
      sqlite3_exec(database_, "ROLLBACK;", nullptr, nullptr, nullptr);
    }
  }

  void commit() {
    execSql(database_, "COMMIT;");
    committed_ = true;
  }

 private:
  sqlite3* database_ = nullptr;
  bool committed_ = false;
};

constexpr std::size_t kStoredFloatBytes = 4;

sqlite3_int64 sqliteEmbeddingDimension(std::size_t dimension) {
  if (dimension == 0 ||
      dimension > static_cast<std::size_t>(
                      std::numeric_limits<sqlite3_int64>::max()) ||
      dimension >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) /
              kStoredFloatBytes) {
    throw StoreError("embedding dimension is outside the supported range");
  }
  return static_cast<sqlite3_int64>(dimension);
}

void validateEmbeddingRecord(const DurableEmbeddingRecord& record,
                             SceneRevision durable_limit) {
  if (record.object_id < 0 || record.document_hash.empty() ||
      record.model_id.empty()) {
    throw StoreError(
        "embedding object id, document hash, and model id are required");
  }
  (void)sqliteEmbeddingDimension(record.dimension);
  if (record.vector.size() != record.dimension) {
    throw StoreError("embedding vector length does not match its dimension");
  }
  for (const float value : record.vector) {
    if (!std::isfinite(value)) {
      throw StoreError("embedding vector contains a non-finite value");
    }
  }
  if (record.created_scene_revision == 0 ||
      record.created_scene_revision > durable_limit) {
    throw StoreError(
        "embedding record depends on a non-durable scene revision");
  }
  (void)sqliteRevision(record.created_scene_revision);
}

std::vector<std::uint8_t> encodeEmbeddingVector(
    const std::vector<float>& vector) {
  static_assert(sizeof(float) == kStoredFloatBytes,
                "SceneStore requires 32-bit floats");
  static_assert(std::numeric_limits<float>::is_iec559,
                "SceneStore requires IEEE-754 floats");
  std::vector<std::uint8_t> bytes;
  bytes.reserve(vector.size() * kStoredFloatBytes);
  for (float value : vector) {
    if (value == 0.0f) {
      value = 0.0f;
    }
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (std::size_t index = 0; index < kStoredFloatBytes; ++index) {
      bytes.push_back(static_cast<std::uint8_t>(
          (bits >> static_cast<unsigned>(index * 8U)) & 0xffU));
    }
  }
  return bytes;
}

std::vector<float> decodeEmbeddingVector(
    const std::vector<std::uint8_t>& bytes,
    std::size_t dimension) {
  (void)sqliteEmbeddingDimension(dimension);
  if (bytes.size() != dimension * kStoredFloatBytes) {
    throw StoreError("stored embedding blob length does not match dimension");
  }
  std::vector<float> vector;
  vector.reserve(dimension);
  for (std::size_t offset = 0; offset < bytes.size();
       offset += kStoredFloatBytes) {
    std::uint32_t bits = 0;
    for (std::size_t index = 0; index < kStoredFloatBytes; ++index) {
      bits |= static_cast<std::uint32_t>(bytes[offset + index])
              << static_cast<unsigned>(index * 8U);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value)) {
      throw StoreError("stored embedding vector contains a non-finite value");
    }
    vector.push_back(value);
  }
  return vector;
}

DurableEmbeddingRecord embeddingRecordFromStatement(
    const Statement& statement,
    int first_column,
    SceneRevision durable_limit) {
  if (statement.columnType(first_column) != SQLITE_INTEGER ||
      statement.columnType(first_column + 1) != SQLITE_TEXT ||
      statement.columnType(first_column + 2) != SQLITE_TEXT ||
      statement.columnType(first_column + 3) != SQLITE_INTEGER ||
      statement.columnType(first_column + 4) != SQLITE_BLOB ||
      statement.columnType(first_column + 5) != SQLITE_INTEGER) {
    throw StoreError("stored embedding record has invalid SQLite column types");
  }
  const sqlite3_int64 stored_object_id =
      statement.columnInt64(first_column);
  if (stored_object_id < 0 ||
      stored_object_id >
          static_cast<sqlite3_int64>(
              std::numeric_limits<SceneObjectId>::max())) {
    throw StoreError("stored embedding object id is outside the valid range");
  }
  const sqlite3_int64 stored_dimension =
      statement.columnInt64(first_column + 3);
  if (stored_dimension <= 0 ||
      static_cast<std::uint64_t>(stored_dimension) >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max())) {
    throw StoreError("stored embedding dimension is invalid");
  }
  DurableEmbeddingRecord record;
  record.object_id = static_cast<SceneObjectId>(stored_object_id);
  record.document_hash = statement.columnText(first_column + 1);
  record.model_id = statement.columnText(first_column + 2);
  record.dimension = static_cast<std::size_t>(stored_dimension);
  record.vector = decodeEmbeddingVector(
      statement.columnBlob(first_column + 4), record.dimension);
  record.created_scene_revision =
      sceneRevision(statement.columnInt64(first_column + 5));
  validateEmbeddingRecord(record, durable_limit);
  return record;
}

bool embeddingRecordsEqual(const DurableEmbeddingRecord& lhs,
                           const DurableEmbeddingRecord& rhs) {
  return lhs.object_id == rhs.object_id &&
         lhs.document_hash == rhs.document_hash &&
         lhs.model_id == rhs.model_id && lhs.dimension == rhs.dimension &&
         lhs.vector == rhs.vector &&
         lhs.created_scene_revision == rhs.created_scene_revision;
}

constexpr const char* kEmbeddingColumns =
    "object_id, document_hash, model_id, dimension, vector, "
    "created_scene_revision";

std::optional<DurableEmbeddingRecord> queryEmbeddingRecord(
    sqlite3* database,
    SceneObjectId object_id,
    const std::string& document_hash,
    const std::string& model_id,
    std::size_t dimension,
    SceneRevision durable_limit) {
  const std::string sql = std::string("SELECT ") + kEmbeddingColumns +
                          " FROM embedding_records WHERE object_id = ?1 "
                          "AND document_hash = ?2 AND model_id = ?3 "
                          "AND dimension = ?4;";
  Statement statement(database, sql.c_str());
  statement.bindInt64(1, static_cast<sqlite3_int64>(object_id));
  statement.bindText(2, document_hash);
  statement.bindText(3, model_id);
  statement.bindInt64(4, sqliteEmbeddingDimension(dimension));
  if (!statement.stepRow()) {
    return std::nullopt;
  }
  return embeddingRecordFromStatement(statement, 0, durable_limit);
}

void validateEmbeddingTable(sqlite3* database,
                            SceneRevision durable_limit) {
  const std::string sql = std::string("SELECT ") + kEmbeddingColumns +
                          " FROM embedding_records ORDER BY object_id, "
                          "document_hash, model_id, dimension;";
  Statement statement(database, sql.c_str());
  while (statement.stepRow()) {
    (void)embeddingRecordFromStatement(statement, 0, durable_limit);
  }
}

std::vector<DurableEmbeddingRecord> queryEmbeddingRecords(
    sqlite3* database,
    SceneRevision durable_limit,
    const std::string* model_id,
    std::size_t dimension) {
  std::string sql = std::string("SELECT ") + kEmbeddingColumns +
                    " FROM embedding_records";
  if (model_id != nullptr) {
    sql += " WHERE model_id = ?1 AND dimension = ?2";
  }
  sql += " ORDER BY model_id, dimension, object_id, document_hash;";
  Statement statement(database, sql.c_str());
  if (model_id != nullptr) {
    statement.bindText(1, *model_id);
    statement.bindInt64(2, sqliteEmbeddingDimension(dimension));
  }
  std::vector<DurableEmbeddingRecord> records;
  while (statement.stepRow()) {
    records.push_back(
        embeddingRecordFromStatement(statement, 0, durable_limit));
  }
  return records;
}

int readUserVersion(sqlite3* database) {
  Statement statement(database, "PRAGMA user_version;");
  if (!statement.stepRow()) {
    throw StoreError("PRAGMA user_version returned no row");
  }
  return statement.columnInt(0);
}

void migrateSchema(sqlite3* database) {
  Transaction transaction(database);
  int version = readUserVersion(database);
  if (version > SceneStore::kCurrentSchemaVersion) {
    throw StoreError("scene store schema is newer than this binary");
  }
  if (version == SceneStore::kCurrentSchemaVersion) {
    transaction.commit();
    return;
  }
  if (version != 0 && version != 1 && version != 2 && version != 3 &&
      version != 4) {
    throw StoreError("scene store has no migration path from schema version " +
                     std::to_string(version));
  }

  if (version == 0) {
    execSql(
        database,
        R"sql(
CREATE TABLE schema_migrations(
  version INTEGER PRIMARY KEY,
  description TEXT NOT NULL
);
CREATE TABLE store_state(
  singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
  durable_scene_revision INTEGER NOT NULL CHECK(durable_scene_revision >= 0)
);
INSERT INTO store_state(singleton, durable_scene_revision) VALUES(1, 0);

CREATE TABLE scene_revisions(
  revision INTEGER PRIMARY KEY CHECK(revision > 0),
  payload TEXT NOT NULL,
  payload_hash TEXT NOT NULL
);

CREATE TABLE objects_current(
  object_id INTEGER PRIMARY KEY,
  scene_revision INTEGER NOT NULL,
  identity_revision INTEGER NOT NULL,
  lifecycle_revision INTEGER NOT NULL,
  geometry_revision INTEGER NOT NULL,
  semantic_revision INTEGER NOT NULL,
  annotation_revision INTEGER NOT NULL,
  artifact_revision INTEGER NOT NULL,
  payload TEXT NOT NULL,
  FOREIGN KEY(scene_revision) REFERENCES scene_revisions(revision)
);

CREATE TABLE object_history(
  scene_revision INTEGER NOT NULL,
  object_id INTEGER NOT NULL,
  is_deleted INTEGER NOT NULL CHECK(is_deleted IN (0, 1)),
  identity_revision INTEGER NOT NULL,
  lifecycle_revision INTEGER NOT NULL,
  geometry_revision INTEGER NOT NULL,
  semantic_revision INTEGER NOT NULL,
  annotation_revision INTEGER NOT NULL,
  artifact_revision INTEGER NOT NULL,
  payload TEXT,
  PRIMARY KEY(scene_revision, object_id),
  FOREIGN KEY(scene_revision) REFERENCES scene_revisions(revision)
);

CREATE TABLE object_aliases(
  retired_object_id INTEGER PRIMARY KEY,
  canonical_object_id INTEGER NOT NULL,
  created_revision INTEGER NOT NULL
);

CREATE TABLE object_tombstones(
  object_id INTEGER PRIMARY KEY,
  deleted_revision INTEGER NOT NULL,
  reason TEXT NOT NULL
);

CREATE TABLE outbox_tasks(
  task_id TEXT PRIMARY KEY,
  dedupe_key TEXT NOT NULL UNIQUE,
  task_type TEXT NOT NULL,
  payload TEXT NOT NULL,
  scene_revision INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('pending', 'leased', 'completed')),
  not_before_unix_ms INTEGER NOT NULL,
  lease_owner TEXT NOT NULL DEFAULT '',
  lease_until_unix_ms INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
  last_error TEXT NOT NULL DEFAULT '',
  completed_at_unix_ms INTEGER NOT NULL DEFAULT 0,
  completed_by TEXT NOT NULL DEFAULT '',
  FOREIGN KEY(scene_revision) REFERENCES scene_revisions(revision)
);
CREATE INDEX outbox_lease_order
  ON outbox_tasks(task_type, state, not_before_unix_ms,
                  lease_until_unix_ms, scene_revision, task_id);

INSERT INTO schema_migrations(version, description)
  VALUES(1, 'initial scene history and durable outbox');
PRAGMA user_version = 1;
)sql");
    version = 1;
  }

  if (version == 1) {
    execSql(
        database,
        R"sql(
CREATE TABLE embedding_records(
  object_id INTEGER NOT NULL CHECK(object_id >= 0),
  document_hash TEXT NOT NULL CHECK(length(document_hash) > 0),
  model_id TEXT NOT NULL CHECK(length(model_id) > 0),
  dimension INTEGER NOT NULL CHECK(dimension > 0),
  vector BLOB NOT NULL,
  created_scene_revision INTEGER NOT NULL CHECK(created_scene_revision > 0),
  PRIMARY KEY(object_id, document_hash, model_id, dimension),
  CHECK(length(vector) = dimension * 4),
  FOREIGN KEY(created_scene_revision) REFERENCES scene_revisions(revision)
);
CREATE INDEX embedding_records_namespace
  ON embedding_records(model_id, dimension, object_id, document_hash);

INSERT INTO schema_migrations(version, description)
  VALUES(2, 'durable semantic embedding records');
PRAGMA user_version = 2;
)sql");
    version = 2;
  }

  if (version == 2) {
    // SQLite cannot alter a CHECK constraint in place. Rebuild only the
    // outbox table and copy every v1/v2 column byte-for-byte; the v2
    // embedding_records table and all scene history remain untouched.
    execSql(
        database,
        R"sql(
DROP INDEX outbox_lease_order;
ALTER TABLE outbox_tasks RENAME TO outbox_tasks_pre_v3;

CREATE TABLE outbox_tasks(
  task_id TEXT PRIMARY KEY,
  dedupe_key TEXT NOT NULL UNIQUE,
  task_type TEXT NOT NULL,
  payload TEXT NOT NULL,
  scene_revision INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('pending', 'leased', 'completed', 'failed')),
  not_before_unix_ms INTEGER NOT NULL,
  lease_owner TEXT NOT NULL DEFAULT '',
  lease_until_unix_ms INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
  last_error TEXT NOT NULL DEFAULT '',
  completed_at_unix_ms INTEGER NOT NULL DEFAULT 0,
  completed_by TEXT NOT NULL DEFAULT '',
  failed_at_unix_ms INTEGER NOT NULL DEFAULT 0 CHECK(failed_at_unix_ms >= 0),
  failed_by TEXT NOT NULL DEFAULT '',
  CHECK((state = 'failed' AND failed_at_unix_ms > 0 AND
         length(failed_by) > 0 AND length(last_error) > 0) OR
        (state != 'failed' AND failed_at_unix_ms = 0 AND failed_by = '')),
  FOREIGN KEY(scene_revision) REFERENCES scene_revisions(revision)
);

INSERT INTO outbox_tasks(
  task_id, dedupe_key, task_type, payload, scene_revision, state,
  not_before_unix_ms, lease_owner, lease_until_unix_ms, attempts,
  last_error, completed_at_unix_ms, completed_by,
  failed_at_unix_ms, failed_by)
SELECT task_id, dedupe_key, task_type, payload, scene_revision, state,
       not_before_unix_ms, lease_owner, lease_until_unix_ms, attempts,
       last_error, completed_at_unix_ms, completed_by, 0, ''
FROM outbox_tasks_pre_v3;

DROP TABLE outbox_tasks_pre_v3;
CREATE INDEX outbox_lease_order
  ON outbox_tasks(task_type, state, not_before_unix_ms,
                  lease_until_unix_ms, scene_revision, task_id);

INSERT INTO schema_migrations(version, description)
  VALUES(3, 'explicit terminal failure for durable outbox tasks');
PRAGMA user_version = 3;
)sql");
    version = 3;
  }

  if (version == 3) {
    execSql(
        database,
        R"sql(
CREATE TABLE map_checkpoint_manifest(
  checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
  backend TEXT NOT NULL CHECK(length(backend) > 0),
  checkpoint_path TEXT NOT NULL UNIQUE CHECK(length(checkpoint_path) > 0),
  map_epoch TEXT NOT NULL CHECK(length(map_epoch) = 32),
  map_revision INTEGER NOT NULL CHECK(map_revision >= 0),
  integrated_through_ns INTEGER NOT NULL CHECK(integrated_through_ns >= 0),
  aligned_scene_revision INTEGER NOT NULL CHECK(aligned_scene_revision >= 0),
  file_size_bytes INTEGER NOT NULL CHECK(file_size_bytes > 0),
  content_hash TEXT NOT NULL CHECK(length(content_hash) > 0),
  created_at_unix_ms INTEGER NOT NULL CHECK(created_at_unix_ms > 0),
  published_at_unix_ms INTEGER NOT NULL CHECK(published_at_unix_ms > 0)
);
CREATE INDEX map_checkpoint_manifest_alignment
  ON map_checkpoint_manifest(aligned_scene_revision, checkpoint_id);

INSERT INTO schema_migrations(version, description)
  VALUES(4, 'durable nvblox checkpoint manifests aligned to scene revisions');
PRAGMA user_version = 4;
)sql");
    version = 4;
  }
  if (version == 4) {
    execSql(
        database,
        R"sql(
ALTER TABLE map_checkpoint_manifest
  ADD COLUMN world_frame TEXT NOT NULL DEFAULT 'legacy.unknown';
ALTER TABLE map_checkpoint_manifest
  ADD COLUMN config_fingerprint TEXT NOT NULL DEFAULT 'legacy.unknown';

CREATE TABLE artifact_origins(
  object_id INTEGER PRIMARY KEY CHECK(object_id >= 0),
  scene_revision INTEGER NOT NULL CHECK(scene_revision > 0),
  created_unix_ms INTEGER NOT NULL CHECK(created_unix_ms >= 0),
  priority INTEGER NOT NULL CHECK(priority IN (0, 1)),
  FOREIGN KEY(scene_revision) REFERENCES scene_revisions(revision)
);
CREATE INDEX artifact_origins_revision
  ON artifact_origins(scene_revision, object_id);

INSERT INTO schema_migrations(version, description)
  VALUES(5, 'checkpoint compatibility and durable artifact SLO origins');
PRAGMA user_version = 5;
)sql");
    version = 5;
  }
  if (version != SceneStore::kCurrentSchemaVersion) {
    throw StoreError("scene store migration did not reach the current schema");
  }
  transaction.commit();
}

DurableTaskState taskStateFromString(const std::string& state) {
  if (state == "pending") {
    return DurableTaskState::kPending;
  }
  if (state == "leased") {
    return DurableTaskState::kLeased;
  }
  if (state == "completed") {
    return DurableTaskState::kCompleted;
  }
  if (state == "failed") {
    return DurableTaskState::kFailed;
  }
  throw StoreError("stored outbox task has an invalid state");
}

DurableTaskRecord taskRecordFromStatement(const Statement& statement,
                                          int first_column = 0) {
  DurableTaskRecord record;
  record.task.task_id = statement.columnText(first_column + 0);
  record.task.dedupe_key = statement.columnText(first_column + 1);
  record.task.task_type = statement.columnText(first_column + 2);
  record.task.payload = statement.columnText(first_column + 3);
  record.task.scene_revision =
      sceneRevision(statement.columnInt64(first_column + 4));
  record.task.not_before_unix_ms =
      statement.columnInt64(first_column + 5);
  record.state = taskStateFromString(statement.columnText(first_column + 6));
  record.lease_owner = statement.columnText(first_column + 7);
  record.lease_until_unix_ms = statement.columnInt64(first_column + 8);
  record.attempts =
      static_cast<std::uint64_t>(statement.columnInt64(first_column + 9));
  record.last_error = statement.columnText(first_column + 10);
  record.completed_at_unix_ms = statement.columnInt64(first_column + 11);
  record.completed_by = statement.columnText(first_column + 12);
  record.failed_at_unix_ms = statement.columnInt64(first_column + 13);
  record.failed_by = statement.columnText(first_column + 14);
  return record;
}

constexpr const char* kTaskColumns =
    "task_id, dedupe_key, task_type, payload, scene_revision, "
    "not_before_unix_ms, state, lease_owner, lease_until_unix_ms, "
    "attempts, last_error, completed_at_unix_ms, completed_by, "
    "failed_at_unix_ms, failed_by";

std::optional<DurableTaskRecord> queryTask(sqlite3* database,
                                           const std::string& task_id) {
  const std::string sql =
      std::string("SELECT ") + kTaskColumns +
      " FROM outbox_tasks WHERE task_id = ?1;";
  Statement statement(database, sql.c_str());
  statement.bindText(1, task_id);
  if (!statement.stepRow()) {
    return std::nullopt;
  }
  return taskRecordFromStatement(statement);
}

void ensureTaskSql(sqlite3* database,
                   const DurableTaskSpec& task,
                   SceneRevision durable_limit) {
  validateTask(task, durable_limit);
  Statement existing(
      database,
      "SELECT task_id, dedupe_key, task_type, payload, scene_revision, "
      "not_before_unix_ms FROM outbox_tasks "
      "WHERE task_id = ?1 OR dedupe_key = ?2 ORDER BY task_id;");
  existing.bindText(1, task.task_id);
  existing.bindText(2, task.dedupe_key);
  std::optional<DurableTaskSpec> found;
  while (existing.stepRow()) {
    DurableTaskSpec candidate;
    candidate.task_id = existing.columnText(0);
    candidate.dedupe_key = existing.columnText(1);
    candidate.task_type = existing.columnText(2);
    candidate.payload = existing.columnText(3);
    candidate.scene_revision = sceneRevision(existing.columnInt64(4));
    candidate.not_before_unix_ms = existing.columnInt64(5);
    if (found) {
      throw StoreError(
          "task id and dedupe key resolve to different existing tasks");
    }
    found = std::move(candidate);
  }
  if (found) {
    if (!tasksEqual(*found, task)) {
      throw StoreError("task id or dedupe key was reused with different content");
    }
    return;
  }

  Statement insert(
      database,
      "INSERT INTO outbox_tasks("
      "task_id, dedupe_key, task_type, payload, scene_revision, state, "
      "not_before_unix_ms) VALUES(?1, ?2, ?3, ?4, ?5, 'pending', ?6);");
  insert.bindText(1, task.task_id);
  insert.bindText(2, task.dedupe_key);
  insert.bindText(3, task.task_type);
  insert.bindText(4, task.payload);
  insert.bindInt64(5, sqliteRevision(task.scene_revision));
  insert.bindInt64(6, task.not_before_unix_ms);
  insert.stepDone();
}

void bindRevisions(Statement* statement,
                   int first_index,
                   const ComponentRevisions& revisions) {
  statement->bindInt64(first_index + 0,
                       sqliteRevision(revisions.identity_revision));
  statement->bindInt64(first_index + 1,
                       sqliteRevision(revisions.lifecycle_revision));
  statement->bindInt64(first_index + 2,
                       sqliteRevision(revisions.geometry_revision));
  statement->bindInt64(first_index + 3,
                       sqliteRevision(revisions.semantic_revision));
  statement->bindInt64(first_index + 4,
                       sqliteRevision(revisions.annotation_revision));
  statement->bindInt64(first_index + 5,
                       sqliteRevision(revisions.artifact_revision));
}

void persistCurrentSceneView(sqlite3* database,
                             const SceneSnapshot& snapshot) {
  std::map<SceneObjectId, std::string> previous_objects;
  {
    Statement current(
        database, "SELECT object_id, payload FROM objects_current;");
    while (current.stepRow()) {
      previous_objects.emplace(current.columnInt(0), current.columnText(1));
    }
  }

  Statement history(
      database,
      "INSERT OR IGNORE INTO object_history("
      "scene_revision, object_id, is_deleted, identity_revision, "
      "lifecycle_revision, geometry_revision, semantic_revision, "
      "annotation_revision, artifact_revision, payload) "
      "VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10);");
  const auto insert_history = [&](SceneObjectId object_id,
                                  const SceneObjectPtr& object,
                                  const std::string* payload) {
    history.bindInt64(1, sqliteRevision(snapshot.revision()));
    history.bindInt(2, object_id);
    history.bindInt(3, object ? 0 : 1);
    if (object) {
      bindRevisions(&history, 4, object->revisions());
      history.bindText(10, payload ? *payload : objectToJson(*object).dump());
    } else {
      for (int index = 4; index <= 9; ++index) {
        history.bindInt64(index, 0);
      }
      history.bindNull(10);
    }
    history.stepDone();
    history.reset();
  };

  Statement upsert_current(
      database,
      "INSERT INTO objects_current("
      "object_id, scene_revision, identity_revision, lifecycle_revision, "
      "geometry_revision, semantic_revision, annotation_revision, "
      "artifact_revision, payload) "
      "VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) "
      "ON CONFLICT(object_id) DO UPDATE SET "
      "scene_revision=excluded.scene_revision, "
      "identity_revision=excluded.identity_revision, "
      "lifecycle_revision=excluded.lifecycle_revision, "
      "geometry_revision=excluded.geometry_revision, "
      "semantic_revision=excluded.semantic_revision, "
      "annotation_revision=excluded.annotation_revision, "
      "artifact_revision=excluded.artifact_revision, payload=excluded.payload;");
  for (const auto& [object_id, object] : snapshot.objects()) {
    const std::string payload = objectToJson(*object).dump();
    const auto previous = previous_objects.find(object_id);
    const bool changed = previous == previous_objects.end() ||
                         previous->second != payload;
    if (changed) {
      insert_history(object_id, object, &payload);
      upsert_current.bindInt(1, object_id);
      upsert_current.bindInt64(2, sqliteRevision(snapshot.revision()));
      bindRevisions(&upsert_current, 3, object->revisions());
      upsert_current.bindText(9, payload);
      upsert_current.stepDone();
      upsert_current.reset();
    }
    if (previous != previous_objects.end()) {
      previous_objects.erase(previous);
    }
  }

  Statement delete_current(
      database, "DELETE FROM objects_current WHERE object_id = ?1;");
  for (const auto& [object_id, payload] : previous_objects) {
    (void)payload;
    insert_history(object_id, nullptr, nullptr);
    delete_current.bindInt(1, object_id);
    delete_current.stepDone();
    delete_current.reset();
  }

  std::map<SceneObjectId, ObjectAlias> previous_aliases;
  {
    Statement aliases(
        database,
        "SELECT retired_object_id, canonical_object_id, created_revision "
        "FROM object_aliases;");
    while (aliases.stepRow()) {
      ObjectAlias alias;
      alias.retired_object_id = aliases.columnInt(0);
      alias.canonical_object_id = aliases.columnInt(1);
      alias.created_revision = sceneRevision(aliases.columnInt64(2));
      previous_aliases.emplace(alias.retired_object_id, alias);
    }
  }
  Statement upsert_alias(
      database,
      "INSERT INTO object_aliases(retired_object_id, canonical_object_id, "
      "created_revision) VALUES(?1, ?2, ?3) "
      "ON CONFLICT(retired_object_id) DO UPDATE SET "
      "canonical_object_id=excluded.canonical_object_id, "
      "created_revision=excluded.created_revision;");
  for (const auto& [retired_id, alias] : snapshot.aliases()) {
    const auto previous = previous_aliases.find(retired_id);
    const bool changed =
        previous == previous_aliases.end() ||
        previous->second.canonical_object_id != alias.canonical_object_id ||
        previous->second.created_revision != alias.created_revision;
    if (changed) {
      upsert_alias.bindInt(1, alias.retired_object_id);
      upsert_alias.bindInt(2, alias.canonical_object_id);
      upsert_alias.bindInt64(3, sqliteRevision(alias.created_revision));
      upsert_alias.stepDone();
      upsert_alias.reset();
      // Alias creation is historical state even if the retired row had
      // disappeared earlier. Record both endpoints, but never duplicate a row
      // already emitted by the object diff for this revision.
      insert_history(alias.retired_object_id, nullptr, nullptr);
      const SceneObjectPtr canonical =
          snapshot.findExactObject(alias.canonical_object_id);
      if (canonical) {
        insert_history(alias.canonical_object_id, canonical, nullptr);
      }
    }
    if (previous != previous_aliases.end()) {
      previous_aliases.erase(previous);
    }
  }
  Statement delete_alias(
      database,
      "DELETE FROM object_aliases WHERE retired_object_id = ?1;");
  for (const auto& [retired_id, alias] : previous_aliases) {
    (void)alias;
    delete_alias.bindInt(1, retired_id);
    delete_alias.stepDone();
    delete_alias.reset();
  }

  std::map<SceneObjectId, ObjectTombstone> previous_tombstones;
  {
    Statement tombstones(
        database,
        "SELECT object_id, deleted_revision, reason FROM object_tombstones;");
    while (tombstones.stepRow()) {
      ObjectTombstone tombstone;
      tombstone.object_id = tombstones.columnInt(0);
      tombstone.deleted_revision =
          sceneRevision(tombstones.columnInt64(1));
      tombstone.reason = tombstones.columnText(2);
      previous_tombstones.emplace(tombstone.object_id, std::move(tombstone));
    }
  }
  Statement upsert_tombstone(
      database,
      "INSERT INTO object_tombstones(object_id, deleted_revision, reason) "
      "VALUES(?1, ?2, ?3) ON CONFLICT(object_id) DO UPDATE SET "
      "deleted_revision=excluded.deleted_revision, reason=excluded.reason;");
  for (const auto& [object_id, tombstone] : snapshot.tombstones()) {
    const auto previous = previous_tombstones.find(object_id);
    const bool changed =
        previous == previous_tombstones.end() ||
        previous->second.deleted_revision != tombstone.deleted_revision ||
        previous->second.reason != tombstone.reason;
    if (changed) {
      upsert_tombstone.bindInt(1, tombstone.object_id);
      upsert_tombstone.bindInt64(
          2, sqliteRevision(tombstone.deleted_revision));
      upsert_tombstone.bindText(3, tombstone.reason);
      upsert_tombstone.stepDone();
      upsert_tombstone.reset();
      insert_history(tombstone.object_id, nullptr, nullptr);
    }
    if (previous != previous_tombstones.end()) {
      previous_tombstones.erase(previous);
    }
  }
  Statement delete_tombstone(
      database, "DELETE FROM object_tombstones WHERE object_id = ?1;");
  for (const auto& [object_id, tombstone] : previous_tombstones) {
    (void)tombstone;
    delete_tombstone.bindInt(1, object_id);
    delete_tombstone.stepDone();
    delete_tombstone.reset();
  }
}

}  // namespace

struct SceneStore::Impl {
  struct PendingCommit {
    SceneSnapshot snapshot;
    // full_payload is retained only in the bounded in-memory suffix so the
    // next commit can be diffed. journal_payload is what reaches SQLite.
    std::string full_payload;
    std::string journal_payload;
    std::vector<DurableTaskSpec> tasks;
    std::vector<DurableArtifactOrigin> artifact_origins;
  };

  Impl(std::string path, std::size_t pending_limit)
      : database_path(std::move(path)),
        max_pending_commits(pending_limit) {}

  SceneStoreStatus flushLocked();
  void closeWithoutFlushLocked();

  std::string database_path;
  sqlite3* database = nullptr;
  int schema_version = 0;
  SceneRevision latest_revision = 0;
  SceneRevision durable_revision = 0;
  std::string latest_full_payload;
  std::size_t max_pending_commits = 0;
  std::vector<PendingCommit> pending;
  mutable std::mutex mutex;
};

SceneStoreStatus SceneStore::Impl::flushLocked() {
  if (!database) {
    return SceneStoreStatus::failure("scene store is not open");
  }
  if (pending.empty()) {
    return SceneStoreStatus::success();
  }

  try {
    Transaction transaction(database);
    SceneRevision expected_revision = durable_revision;
    for (const PendingCommit& commit : pending) {
      if (expected_revision == std::numeric_limits<SceneRevision>::max() ||
          commit.snapshot.revision() != expected_revision + 1) {
        throw StoreError("pending scene commits are not contiguous");
      }
      Statement insert_revision(
          database,
          "INSERT INTO scene_revisions(revision, payload, payload_hash) "
          "VALUES(?1, ?2, ?3);");
      insert_revision.bindInt64(1, sqliteRevision(commit.snapshot.revision()));
      insert_revision.bindText(2, commit.journal_payload);
      insert_revision.bindText(3, payloadHash(commit.journal_payload));
      insert_revision.stepDone();

      persistCurrentSceneView(database, commit.snapshot);
      for (const DurableTaskSpec& task : commit.tasks) {
        ensureTaskSql(database, task, commit.snapshot.revision());
      }
      Statement insert_origin(
          database,
          "INSERT INTO artifact_origins(object_id, scene_revision, "
          "created_unix_ms, priority) VALUES(?1, ?2, ?3, ?4) "
          "ON CONFLICT(object_id) DO NOTHING;");
      for (const DurableArtifactOrigin& origin : commit.artifact_origins) {
        validateArtifactOrigin(origin, commit.snapshot.revision());
        insert_origin.bindInt(1, origin.object_id);
        insert_origin.bindInt64(2, sqliteRevision(origin.scene_revision));
        insert_origin.bindInt64(3, origin.created_unix_ms);
        insert_origin.bindInt(4, static_cast<int>(origin.priority));
        insert_origin.stepDone();
        insert_origin.reset();

        Statement verify(
            database,
            "SELECT scene_revision, created_unix_ms, priority "
            "FROM artifact_origins WHERE object_id = ?1;");
        verify.bindInt(1, origin.object_id);
        if (!verify.stepRow() ||
            sceneRevision(verify.columnInt64(0)) != origin.scene_revision ||
            verify.columnInt64(1) != origin.created_unix_ms ||
            verify.columnInt(2) != static_cast<int>(origin.priority)) {
          throw StoreError(
              "artifact origin object id was reused with different content");
        }
      }
      // Object ids are never reused. Once an object is retired/tombstoned its
      // origin can be removed without affecting any future first appearance.
      execSql(database,
              "DELETE FROM artifact_origins WHERE object_id NOT IN "
              "(SELECT object_id FROM objects_current);");
      expected_revision = commit.snapshot.revision();
    }

    Statement update_watermark(
        database,
        "UPDATE store_state SET durable_scene_revision = ?1 "
        "WHERE singleton = 1;");
    update_watermark.bindInt64(1, sqliteRevision(expected_revision));
    update_watermark.stepDone();
    if (sqlite3_changes(database) != 1) {
      throw StoreError("store_state singleton row is missing");
    }
    transaction.commit();

    durable_revision = expected_revision;
    latest_full_payload = pending.back().full_payload;
    pending.clear();
    return SceneStoreStatus::success();
  } catch (const std::exception& error) {
    return SceneStoreStatus::failure(error.what());
  }
}

void SceneStore::Impl::closeWithoutFlushLocked() {
  pending.clear();
  latest_revision = durable_revision;
  latest_full_payload.clear();
  if (database) {
    sqlite3_close_v2(database);
    database = nullptr;
  }
  schema_version = 0;
}

SceneStore::SceneStore(std::string database_path,
                       std::size_t max_pending_commits)
    : impl_(std::make_unique<Impl>(std::move(database_path),
                                   max_pending_commits)) {}

SceneStore::~SceneStore() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->closeWithoutFlushLocked();
}

SceneStoreStatus SceneStore::open() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->database) {
    return SceneStoreStatus::failure("scene store is already open");
  }
  if (impl_->database_path.empty()) {
    return SceneStoreStatus::failure("scene store path is empty");
  }
  if (impl_->max_pending_commits == 0) {
    return SceneStoreStatus::failure(
        "scene store pending commit limit must be positive");
  }

  sqlite3* database = nullptr;
  const int open_result = sqlite3_open_v2(
      impl_->database_path.c_str(), &database,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
      nullptr);
  if (open_result != SQLITE_OK) {
    const std::string error = sqliteMessage(database, "failed to open scene store");
    if (database) {
      sqlite3_close_v2(database);
    }
    return SceneStoreStatus::failure(error);
  }
  impl_->database = database;

  try {
    // PersistenceActor owns retry/backoff and its terminal-failure deadline.
    // Keep each SQLite call short enough for stop/fault coordination rather
    // than hiding five-second uninterruptible waits inside the store.
    sqlite3_busy_timeout(database, 100);
    execSql(database, "PRAGMA foreign_keys = ON;");
    execSql(database, "PRAGMA synchronous = FULL;");
    {
      Statement wal_mode(database, "PRAGMA journal_mode = WAL;");
      if (!wal_mode.stepRow() || wal_mode.columnText(0) != "wal") {
        throw StoreError("SQLite refused WAL journal mode");
      }
    }
    migrateSchema(database);
    impl_->schema_version = readUserVersion(database);

    Statement state(database,
                    "SELECT durable_scene_revision FROM store_state "
                    "WHERE singleton = 1;");
    if (!state.stepRow()) {
      throw StoreError("scene store watermark row is missing");
    }
    impl_->durable_revision = sceneRevision(state.columnInt64(0));

    Statement maximum(database,
                      "SELECT COALESCE(MAX(revision), 0) FROM scene_revisions;");
    if (!maximum.stepRow()) {
      throw StoreError("could not read latest stored scene revision");
    }
    const SceneRevision maximum_revision =
        sceneRevision(maximum.columnInt64(0));
    if (maximum_revision != impl_->durable_revision) {
      throw StoreError(
          "scene revision history does not match durable watermark");
    }
    Statement revision_count(
        database, "SELECT COUNT(*) FROM scene_revisions;");
    if (!revision_count.stepRow() ||
        sceneRevision(revision_count.columnInt64(0)) !=
            impl_->durable_revision) {
      throw StoreError("scene revision history is not contiguous from revision 1");
    }
    validateEmbeddingTable(database, impl_->durable_revision);
    validateMapCheckpointTable(database, impl_->durable_revision);
    validateArtifactOriginTable(database, impl_->durable_revision);
    impl_->latest_full_payload = materializeDurableSnapshotPayload(
        database, impl_->durable_revision);
    if (!impl_->latest_full_payload.empty()) {
      (void)deserializeSnapshot(impl_->latest_full_payload,
                                impl_->durable_revision);
    }
    impl_->latest_revision = impl_->durable_revision;
    impl_->pending.clear();
    return SceneStoreStatus::success();
  } catch (const std::exception& error) {
    impl_->closeWithoutFlushLocked();
    return SceneStoreStatus::failure(error.what());
  }
}

bool SceneStore::isOpen() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->database != nullptr;
}

int SceneStore::schemaVersion() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->schema_version;
}

SceneStoreStatus SceneStore::enqueueCommit(
    SceneSnapshot snapshot,
    std::vector<DurableTaskSpec> outbox_tasks,
    std::vector<DurableArtifactOrigin> artifact_origins) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->database) {
    return SceneStoreStatus::failure("scene store is not open");
  }
  try {
    if (impl_->latest_revision == std::numeric_limits<SceneRevision>::max() ||
        snapshot.revision() != impl_->latest_revision + 1) {
      return SceneStoreStatus::failure(
          "scene commits must be enqueued in contiguous revision order");
    }
    if (snapshot.durableRevision() > snapshot.revision()) {
      return SceneStoreStatus::failure(
          "snapshot durable watermark exceeds its live revision");
    }
    if (impl_->pending.size() >= impl_->max_pending_commits) {
      return SceneStoreStatus::failure(
          "scene store pending commit hard limit reached; backpressure required");
    }
    validateSnapshotSqlRange(snapshot);
    // Serialize once during admission so malformed component state cannot
    // poison an otherwise ordered pending suffix.
    const std::string full_payload = serializeSnapshot(snapshot);
    // Admission requires a full round trip so JSON cannot turn NaN/Inf into a
    // durable null that only fails after restart.
    (void)deserializeSnapshot(full_payload, snapshot.revision());

    std::string journal_payload = full_payload;
    const bool checkpoint =
        snapshot.revision() == 1 ||
        snapshot.revision() % SceneStore::kRevisionCheckpointInterval == 0;
    if (!checkpoint) {
      const std::string& previous_full_payload =
          impl_->pending.empty() ? impl_->latest_full_payload
                                 : impl_->pending.back().full_payload;
      if (previous_full_payload.empty()) {
        throw StoreError("scene delta has no predecessor checkpoint state");
      }
      journal_payload =
          serializeSnapshotDelta(previous_full_payload, full_payload);
    }

    std::map<std::string, DurableTaskSpec> task_ids;
    std::map<std::string, DurableTaskSpec> dedupe_keys;
    auto register_task = [&](const DurableTaskSpec& task) {
      validateTask(task, snapshot.revision());
      const auto by_id = task_ids.find(task.task_id);
      if (by_id != task_ids.end() && !tasksEqual(by_id->second, task)) {
        throw StoreError("pending task id has conflicting content");
      }
      const auto by_dedupe = dedupe_keys.find(task.dedupe_key);
      if (by_dedupe != dedupe_keys.end() &&
          !tasksEqual(by_dedupe->second, task)) {
        throw StoreError("pending task dedupe key has conflicting content");
      }
      task_ids[task.task_id] = task;
      dedupe_keys[task.dedupe_key] = task;
    };
    for (const Impl::PendingCommit& pending : impl_->pending) {
      for (const DurableTaskSpec& task : pending.tasks) {
        register_task(task);
      }
    }
    for (const DurableTaskSpec& task : outbox_tasks) {
      register_task(task);
    }
    std::map<SceneObjectId, DurableArtifactOrigin> pending_origins;
    for (const Impl::PendingCommit& pending : impl_->pending) {
      for (const DurableArtifactOrigin& origin : pending.artifact_origins) {
        pending_origins.emplace(origin.object_id, origin);
      }
    }
    for (const DurableArtifactOrigin& origin : artifact_origins) {
      validateArtifactOrigin(origin, snapshot.revision());
      if (origin.scene_revision != snapshot.revision()) {
        throw StoreError(
            "artifact origin must be admitted with its owning scene revision");
      }
      const auto existing = pending_origins.find(origin.object_id);
      if (existing != pending_origins.end() &&
          !(existing->second == origin)) {
        throw StoreError("pending artifact origin has conflicting content");
      }
      pending_origins[origin.object_id] = origin;
    }

    impl_->pending.push_back(
        Impl::PendingCommit{std::move(snapshot), full_payload,
                            std::move(journal_payload),
                            std::move(outbox_tasks),
                            std::move(artifact_origins)});
    impl_->latest_revision = impl_->pending.back().snapshot.revision();
    return SceneStoreStatus::success();
  } catch (const std::exception& error) {
    return SceneStoreStatus::failure(error.what());
  }
}

SceneStoreStatus SceneStore::flush() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->flushLocked();
}

SceneStoreStatus SceneStore::gracefulFlush() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  SceneStoreStatus status = impl_->flushLocked();
  if (!status) {
    return status;
  }
  int log_frames = 0;
  int checkpointed_frames = 0;
  const int rc = sqlite3_wal_checkpoint_v2(
      impl_->database, nullptr, SQLITE_CHECKPOINT_TRUNCATE, &log_frames,
      &checkpointed_frames);
  if (rc != SQLITE_OK) {
    return SceneStoreStatus::failure(
        sqliteMessage(impl_->database, "WAL checkpoint failed"));
  }
  return SceneStoreStatus::success();
}

SceneStoreStatus SceneStore::closeGracefully() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->database) {
    return SceneStoreStatus::failure("scene store is not open");
  }
  SceneStoreStatus status = impl_->flushLocked();
  if (!status) {
    return status;
  }
  int log_frames = 0;
  int checkpointed_frames = 0;
  const int checkpoint_result = sqlite3_wal_checkpoint_v2(
      impl_->database, nullptr, SQLITE_CHECKPOINT_TRUNCATE, &log_frames,
      &checkpointed_frames);
  if (checkpoint_result != SQLITE_OK) {
    return SceneStoreStatus::failure(
        sqliteMessage(impl_->database, "WAL checkpoint failed"));
  }
  const int close_result = sqlite3_close(impl_->database);
  if (close_result != SQLITE_OK) {
    return SceneStoreStatus::failure(
        sqliteMessage(impl_->database, "closing scene store failed"));
  }
  impl_->database = nullptr;
  impl_->schema_version = 0;
  impl_->latest_revision = impl_->durable_revision;
  return SceneStoreStatus::success();
}

SceneStoreWatermarks SceneStore::watermarks() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return SceneStoreWatermarks{impl_->latest_revision, impl_->durable_revision,
                              impl_->pending.size(),
                              impl_->max_pending_commits,
                              impl_->pending.size() >=
                                  impl_->max_pending_commits};
}

SceneRestoreResult SceneStore::restoreLatest() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  SceneRestoreResult result;
  if (!impl_->database) {
    result.status = SceneStoreStatus::failure("scene store is not open");
    return result;
  }
  if (impl_->durable_revision == 0) {
    result.status = SceneStoreStatus::success();
    return result;
  }
  try {
    const std::string payload = materializeDurableSnapshotPayload(
        impl_->database, impl_->durable_revision);
    result.snapshot = deserializeSnapshot(payload, impl_->durable_revision);
    result.found = true;
    result.status = SceneStoreStatus::success();
    return result;
  } catch (const std::exception& error) {
    result.status = SceneStoreStatus::failure(error.what());
    return result;
  }
}

SceneRestoreResult SceneStore::restoreAt(SceneRevision revision) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  SceneRestoreResult result;
  if (!impl_->database) {
    result.status = SceneStoreStatus::failure("scene store is not open");
    return result;
  }
  if (revision > impl_->durable_revision) {
    result.status = SceneStoreStatus::failure(
        "requested scene revision is newer than the durable watermark");
    return result;
  }
  if (revision == 0) {
    result.status = SceneStoreStatus::success();
    return result;
  }
  try {
    const std::string payload =
        materializeDurableSnapshotPayload(impl_->database, revision);
    result.snapshot = deserializeSnapshot(payload, revision);
    result.found = true;
    result.status = SceneStoreStatus::success();
    return result;
  } catch (const std::exception& error) {
    result.status = SceneStoreStatus::failure(error.what());
    return result;
  }
}

SceneStoreStatus SceneStore::rewindTo(
    SceneRevision revision,
    bool retain_aligned_map_checkpoints) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->database) {
    return SceneStoreStatus::failure("scene store is not open");
  }
  if (!impl_->pending.empty() || impl_->latest_revision !=
                                     impl_->durable_revision) {
    return SceneStoreStatus::failure(
        "scene store cannot rewind with an undurable in-memory suffix");
  }
  if (revision > impl_->durable_revision) {
    return SceneStoreStatus::failure(
        "rewind revision is newer than the durable watermark");
  }
  if (revision == impl_->durable_revision &&
      retain_aligned_map_checkpoints) {
    return SceneStoreStatus::success();
  }

  try {
    std::string full_payload;
    SceneSnapshot snapshot;
    if (revision != 0) {
      full_payload = materializeDurableSnapshotPayload(impl_->database,
                                                       revision);
      snapshot = deserializeSnapshot(full_payload, revision);
    }

    Transaction transaction(impl_->database);
    auto delete_after = [this, revision](const char* sql) {
      Statement statement(impl_->database, sql);
      statement.bindInt64(1, sqliteRevision(revision));
      statement.stepDone();
    };
    if (retain_aligned_map_checkpoints) {
      delete_after(
          "DELETE FROM map_checkpoint_manifest "
          "WHERE aligned_scene_revision > ?1;");
    } else {
      execSql(impl_->database, "DELETE FROM map_checkpoint_manifest;");
    }
    delete_after(
        "DELETE FROM embedding_records WHERE created_scene_revision > ?1;");
    delete_after("DELETE FROM outbox_tasks WHERE scene_revision > ?1;");
    delete_after("DELETE FROM artifact_origins WHERE scene_revision > ?1;");

    // Rebuild the materialized view at the branch point. Re-emitting its
    // object_history rows keeps history queries coherent even when the target
    // itself was represented by a compact delta.
    execSql(impl_->database, "DELETE FROM objects_current;");
    execSql(impl_->database, "DELETE FROM object_aliases;");
    execSql(impl_->database, "DELETE FROM object_tombstones;");
    delete_after("DELETE FROM object_history WHERE scene_revision >= ?1;");
    delete_after("DELETE FROM scene_revisions WHERE revision > ?1;");
    if (revision != 0) {
      persistCurrentSceneView(impl_->database, snapshot);
      execSql(impl_->database,
              "DELETE FROM artifact_origins WHERE object_id NOT IN "
              "(SELECT object_id FROM objects_current);");
    } else {
      execSql(impl_->database, "DELETE FROM artifact_origins;");
    }

    Statement update_watermark(
        impl_->database,
        "UPDATE store_state SET durable_scene_revision = ?1 "
        "WHERE singleton = 1;");
    update_watermark.bindInt64(1, sqliteRevision(revision));
    update_watermark.stepDone();
    if (sqlite3_changes(impl_->database) != 1) {
      throw StoreError("store_state singleton row is missing");
    }
    transaction.commit();

    impl_->durable_revision = revision;
    impl_->latest_revision = revision;
    impl_->latest_full_payload = std::move(full_payload);
    impl_->pending.clear();
    return SceneStoreStatus::success();
  } catch (const std::exception& error) {
    return SceneStoreStatus::failure(error.what());
  }
}

SceneStoreStatus SceneStore::publishMapCheckpoint(
    const MapCheckpointManifest& manifest) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->database) {
    return SceneStoreStatus::failure("scene store is not open");
  }
  try {
    validateMapCheckpointManifest(manifest, impl_->durable_revision);

    const std::string existing_sql =
        std::string("SELECT ") + kMapCheckpointColumns +
        " FROM map_checkpoint_manifest WHERE checkpoint_path = ?1;";
    Statement existing(impl_->database, existing_sql.c_str());
    existing.bindText(1, manifest.checkpoint_path);
    if (existing.stepRow()) {
      const MapCheckpointManifest stored = mapCheckpointFromStatement(
          existing, 0, impl_->durable_revision);
      if (!mapCheckpointManifestsEqual(stored, manifest)) {
        throw StoreError(
            "checkpoint path was reused with different manifest content");
      }
      return SceneStoreStatus::success();
    }
    if (manifest.aligned_scene_revision != impl_->durable_revision) {
      throw StoreError(
          "map checkpoint must align with the current durable scene revision");
    }

    Transaction transaction(impl_->database);
    Statement insert(
        impl_->database,
        "INSERT INTO map_checkpoint_manifest("
        "backend, checkpoint_path, world_frame, config_fingerprint, "
        "map_epoch, map_revision, "
        "integrated_through_ns, aligned_scene_revision, file_size_bytes, "
        "content_hash, created_at_unix_ms, published_at_unix_ms) "
        "VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12);");
    insert.bindText(1, manifest.backend);
    insert.bindText(2, manifest.checkpoint_path);
    insert.bindText(3, manifest.world_frame);
    insert.bindText(4, manifest.config_fingerprint);
    insert.bindText(5, runIdString(manifest.map_epoch));
    insert.bindInt64(6, sqliteRevision(manifest.map_revision));
    insert.bindInt64(7, manifest.integrated_through_ns);
    insert.bindInt64(8,
                     sqliteRevision(manifest.aligned_scene_revision));
    insert.bindInt64(9, sqliteRevision(manifest.file_size_bytes));
    insert.bindText(10, manifest.content_hash);
    insert.bindInt64(11, manifest.created_at_unix_ms);
    insert.bindInt64(12, unixTimeMilliseconds());
    insert.stepDone();
    transaction.commit();
    return SceneStoreStatus::success();
  } catch (const std::exception& error) {
    return SceneStoreStatus::failure(error.what());
  }
}

MapCheckpointLookupResult SceneStore::latestMapCheckpoint() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  MapCheckpointLookupResult result;
  if (!impl_->database) {
    result.status = SceneStoreStatus::failure("scene store is not open");
    return result;
  }
  try {
    const std::string sql =
        std::string("SELECT ") + kMapCheckpointColumns +
        " FROM map_checkpoint_manifest ORDER BY checkpoint_id DESC LIMIT 1;";
    Statement statement(impl_->database, sql.c_str());
    if (statement.stepRow()) {
      result.manifest = mapCheckpointFromStatement(
          statement, 0, impl_->durable_revision);
    }
    result.status = SceneStoreStatus::success();
    return result;
  } catch (const std::exception& error) {
    result.status = SceneStoreStatus::failure(error.what());
    return result;
  }
}

ArtifactOriginListResult SceneStore::listArtifactOrigins() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  ArtifactOriginListResult result;
  if (!impl_->database) {
    result.status = SceneStoreStatus::failure("scene store is not open");
    return result;
  }
  try {
    Statement statement(
        impl_->database,
        "SELECT object_id, scene_revision, created_unix_ms, priority "
        "FROM artifact_origins ORDER BY object_id;");
    while (statement.stepRow()) {
      DurableArtifactOrigin origin;
      origin.object_id = statement.columnInt(0);
      origin.scene_revision = sceneRevision(statement.columnInt64(1));
      origin.created_unix_ms = statement.columnInt64(2);
      const int priority = statement.columnInt(3);
      if (priority == static_cast<int>(ArtifactPriority::kInteractive)) {
        origin.priority = ArtifactPriority::kInteractive;
      } else if (priority == static_cast<int>(ArtifactPriority::kBulk)) {
        origin.priority = ArtifactPriority::kBulk;
      } else {
        throw StoreError("stored artifact origin priority is invalid");
      }
      validateArtifactOrigin(origin, impl_->durable_revision);
      result.origins.push_back(origin);
    }
    result.status = SceneStoreStatus::success();
  } catch (const std::exception& error) {
    result.status = SceneStoreStatus::failure(error.what());
  }
  return result;
}

SceneStoreStatus SceneStore::ensureTask(const DurableTaskSpec& task) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->database) {
    return SceneStoreStatus::failure("scene store is not open");
  }
  try {
    Transaction transaction(impl_->database);
    ensureTaskSql(impl_->database, task, impl_->durable_revision);
    transaction.commit();
    return SceneStoreStatus::success();
  } catch (const std::exception& error) {
    return SceneStoreStatus::failure(error.what());
  }
}

TaskLeaseResult SceneStore::leaseNextTask(
    const std::string& task_type,
    const std::string& lease_owner,
    std::int64_t now_unix_ms,
    std::int64_t lease_duration_ms) {
  return leaseNextTask(
      task_type.empty() ? std::vector<std::string>{}
                        : std::vector<std::string>{task_type},
      lease_owner, now_unix_ms, lease_duration_ms);
}

TaskLeaseResult SceneStore::leaseNextTask(
    const std::vector<std::string>& task_types,
    const std::string& lease_owner,
    std::int64_t now_unix_ms,
    std::int64_t lease_duration_ms) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  TaskLeaseResult result;
  if (!impl_->database) {
    result.status = SceneStoreStatus::failure("scene store is not open");
    return result;
  }
  if (lease_owner.empty() || lease_duration_ms <= 0 ||
      now_unix_ms >
          std::numeric_limits<std::int64_t>::max() - lease_duration_ms) {
    result.status = SceneStoreStatus::failure(
        "lease owner/duration/time is invalid");
    return result;
  }
  std::vector<std::string> normalized_types = task_types;
  if (std::any_of(normalized_types.begin(), normalized_types.end(),
                  [](const std::string& type) { return type.empty(); })) {
    result.status =
        SceneStoreStatus::failure("lease task types must be non-empty");
    return result;
  }
  std::sort(normalized_types.begin(), normalized_types.end());
  normalized_types.erase(
      std::unique(normalized_types.begin(), normalized_types.end()),
      normalized_types.end());
  try {
    Transaction transaction(impl_->database);
    std::string type_filter;
    if (!normalized_types.empty()) {
      type_filter = " AND task_type IN (";
      for (std::size_t index = 0; index < normalized_types.size(); ++index) {
        if (index != 0) {
          type_filter += ", ";
        }
        type_filter += "?" + std::to_string(index + 2);
      }
      type_filter += ")";
    }
    const std::string select_sql =
        std::string("SELECT ") + kTaskColumns +
        " FROM outbox_tasks WHERE not_before_unix_ms <= ?1 "
        "AND (state = 'pending' OR "
        "(state = 'leased' AND lease_until_unix_ms <= ?1))" +
        type_filter + ";";
    Statement select(impl_->database, select_sql.c_str());
    select.bindInt64(1, now_unix_ms);
    for (std::size_t index = 0; index < normalized_types.size(); ++index) {
      select.bindText(static_cast<int>(index + 2), normalized_types[index]);
    }

    std::optional<DurableTaskRecord> selected;
    std::optional<DurableTaskSchedulingKey> selected_key;
    while (select.stepRow()) {
      DurableTaskRecord candidate = taskRecordFromStatement(select);
      DurableTaskSchedulingKey candidate_key =
          schedulingKey(candidate, now_unix_ms);
      if (!selected_key || schedulingKeyLess(candidate_key, *selected_key)) {
        selected = std::move(candidate);
        selected_key = std::move(candidate_key);
      }
    }
    if (!selected) {
      transaction.commit();
      result.status = SceneStoreStatus::success();
      return result;
    }
    if (selected->attempts >= static_cast<std::uint64_t>(
                                 std::numeric_limits<sqlite3_int64>::max())) {
      throw StoreError("durable task attempt counter is exhausted");
    }
    const std::int64_t lease_until = now_unix_ms + lease_duration_ms;

    Statement update(
        impl_->database,
        "UPDATE outbox_tasks SET state = 'leased', lease_owner = ?1, "
        "lease_until_unix_ms = ?2, attempts = attempts + 1 "
        "WHERE task_id = ?3 AND "
        "(state = 'pending' OR "
        "(state = 'leased' AND lease_until_unix_ms <= ?4));");
    update.bindText(1, lease_owner);
    update.bindInt64(2, lease_until);
    update.bindText(3, selected->task.task_id);
    update.bindInt64(4, now_unix_ms);
    update.stepDone();
    if (sqlite3_changes(impl_->database) != 1) {
      throw StoreError("selected task could not be atomically leased");
    }
    std::optional<DurableTaskRecord> leased =
        queryTask(impl_->database, selected->task.task_id);
    if (!leased) {
      throw StoreError("leased task disappeared");
    }
    transaction.commit();
    result.task = std::move(leased);
    result.status = SceneStoreStatus::success();
    return result;
  } catch (const std::exception& error) {
    result.status = SceneStoreStatus::failure(error.what());
    return result;
  }
}

SceneStoreStatus SceneStore::renewTaskLease(
    const std::string& task_id,
    const std::string& lease_owner,
    std::uint64_t lease_attempt,
    std::int64_t now_unix_ms,
    std::int64_t lease_duration_ms) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->database) {
    return SceneStoreStatus::failure("scene store is not open");
  }
  if (task_id.empty() || lease_owner.empty() || lease_attempt == 0 ||
      lease_attempt > static_cast<std::uint64_t>(
                          std::numeric_limits<sqlite3_int64>::max()) ||
      lease_duration_ms <= 0 ||
      now_unix_ms >
          std::numeric_limits<std::int64_t>::max() - lease_duration_ms) {
    return SceneStoreStatus::failure("lease renewal arguments are invalid");
  }
  try {
    Statement update(
        impl_->database,
        "UPDATE outbox_tasks SET lease_until_unix_ms = ?1 "
        "WHERE task_id = ?2 AND state = 'leased' AND lease_owner = ?3 "
        "AND lease_until_unix_ms > ?4 AND attempts = ?5;");
    update.bindInt64(1, now_unix_ms + lease_duration_ms);
    update.bindText(2, task_id);
    update.bindText(3, lease_owner);
    update.bindInt64(4, now_unix_ms);
    update.bindInt64(5, static_cast<sqlite3_int64>(lease_attempt));
    update.stepDone();
    if (sqlite3_changes(impl_->database) != 1) {
      return SceneStoreStatus::failure(
          "task lease is missing, expired, or owned by another worker");
    }
    return SceneStoreStatus::success();
  } catch (const std::exception& error) {
    return SceneStoreStatus::failure(error.what());
  }
}

SceneStoreStatus SceneStore::completeTask(
    const std::string& task_id,
    const std::string& lease_owner,
    std::uint64_t lease_attempt,
    std::int64_t completed_at_unix_ms) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->database) {
    return SceneStoreStatus::failure("scene store is not open");
  }
  if (task_id.empty() || lease_owner.empty() || lease_attempt == 0 ||
      lease_attempt > static_cast<std::uint64_t>(
                          std::numeric_limits<sqlite3_int64>::max())) {
    return SceneStoreStatus::failure(
        "task id, lease owner, and lease attempt are required");
  }
  try {
    Transaction transaction(impl_->database);
    const std::optional<DurableTaskRecord> existing =
        queryTask(impl_->database, task_id);
    if (!existing) {
      return SceneStoreStatus::failure("task does not exist");
    }
    if (existing->state == DurableTaskState::kCompleted) {
      if (existing->attempts != lease_attempt ||
          existing->completed_by != lease_owner) {
        return SceneStoreStatus::failure(
            "task was completed by a different lease attempt");
      }
      transaction.commit();
      return SceneStoreStatus::success();
    }
    if (existing->state != DurableTaskState::kLeased ||
        existing->lease_owner != lease_owner ||
        existing->attempts != lease_attempt) {
      return SceneStoreStatus::failure(
          "task is not leased by the completing worker");
    }
    Statement update(
        impl_->database,
        "UPDATE outbox_tasks SET state = 'completed', completed_at_unix_ms = ?1, "
        "completed_by = ?2, lease_owner = '', lease_until_unix_ms = 0 "
        "WHERE task_id = ?3 AND state = 'leased' AND lease_owner = ?2 "
        "AND attempts = ?4;");
    update.bindInt64(1, completed_at_unix_ms);
    update.bindText(2, lease_owner);
    update.bindText(3, task_id);
    update.bindInt64(4, static_cast<sqlite3_int64>(lease_attempt));
    update.stepDone();
    if (sqlite3_changes(impl_->database) != 1) {
      throw StoreError("task completion lost its lease race");
    }
    transaction.commit();
    return SceneStoreStatus::success();
  } catch (const std::exception& error) {
    return SceneStoreStatus::failure(error.what());
  }
}

SceneStoreStatus SceneStore::failTask(
    const std::string& task_id,
    const std::string& lease_owner,
    std::uint64_t lease_attempt,
    std::int64_t failed_at_unix_ms,
    std::string error) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->database) {
    return SceneStoreStatus::failure("scene store is not open");
  }
  if (task_id.empty() || lease_owner.empty() || lease_attempt == 0 ||
      lease_attempt > static_cast<std::uint64_t>(
                          std::numeric_limits<sqlite3_int64>::max()) ||
      failed_at_unix_ms <= 0 || error.empty()) {
    return SceneStoreStatus::failure(
        "task id, lease owner/attempt, failure time, and error are required");
  }
  try {
    Transaction transaction(impl_->database);
    const std::optional<DurableTaskRecord> existing =
        queryTask(impl_->database, task_id);
    if (!existing) {
      return SceneStoreStatus::failure("task does not exist");
    }
    if (existing->state == DurableTaskState::kFailed) {
      if (existing->attempts != lease_attempt ||
          existing->failed_by != lease_owner) {
        return SceneStoreStatus::failure(
            "task was failed by a different lease attempt");
      }
      transaction.commit();
      return SceneStoreStatus::success();
    }
    if (existing->state != DurableTaskState::kLeased ||
        existing->lease_owner != lease_owner ||
        existing->attempts != lease_attempt ||
        existing->lease_until_unix_ms <= failed_at_unix_ms) {
      return SceneStoreStatus::failure(
          "task lease expired or is not owned by the failing worker");
    }

    Statement update(
        impl_->database,
        "UPDATE outbox_tasks SET state = 'failed', last_error = ?1, "
        "failed_at_unix_ms = ?2, failed_by = ?3, "
        "lease_owner = '', lease_until_unix_ms = 0, "
        "completed_at_unix_ms = 0, completed_by = '' "
        "WHERE task_id = ?4 AND state = 'leased' AND lease_owner = ?3 "
        "AND attempts = ?5 AND lease_until_unix_ms > ?2;");
    update.bindText(1, error);
    update.bindInt64(2, failed_at_unix_ms);
    update.bindText(3, lease_owner);
    update.bindText(4, task_id);
    update.bindInt64(5, static_cast<sqlite3_int64>(lease_attempt));
    update.stepDone();
    if (sqlite3_changes(impl_->database) != 1) {
      throw StoreError("task failure lost its lease race");
    }
    transaction.commit();
    return SceneStoreStatus::success();
  } catch (const std::exception& exception) {
    return SceneStoreStatus::failure(exception.what());
  }
}

SceneStoreStatus SceneStore::retryTask(const std::string& task_id,
                                       const std::string& lease_owner,
                                       std::uint64_t lease_attempt,
                                       std::int64_t retry_at_unix_ms,
                                       std::string error) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->database) {
    return SceneStoreStatus::failure("scene store is not open");
  }
  if (task_id.empty() || lease_owner.empty() || lease_attempt == 0 ||
      lease_attempt > static_cast<std::uint64_t>(
                          std::numeric_limits<sqlite3_int64>::max())) {
    return SceneStoreStatus::failure(
        "task id, lease owner, and lease attempt are required");
  }
  try {
    Statement update(
        impl_->database,
        "UPDATE outbox_tasks SET state = 'pending', not_before_unix_ms = ?1, "
        "last_error = ?2, lease_owner = '', lease_until_unix_ms = 0 "
        "WHERE task_id = ?3 AND state = 'leased' AND lease_owner = ?4 "
        "AND attempts = ?5;");
    update.bindInt64(1, retry_at_unix_ms);
    update.bindText(2, error);
    update.bindText(3, task_id);
    update.bindText(4, lease_owner);
    update.bindInt64(5, static_cast<sqlite3_int64>(lease_attempt));
    update.stepDone();
    if (sqlite3_changes(impl_->database) != 1) {
      return SceneStoreStatus::failure(
          "task is not leased by the retrying worker");
    }
    return SceneStoreStatus::success();
  } catch (const std::exception& exception) {
    return SceneStoreStatus::failure(exception.what());
  }
}

TaskLookupResult SceneStore::lookupTask(const std::string& task_id) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  TaskLookupResult result;
  if (!impl_->database) {
    result.status = SceneStoreStatus::failure("scene store is not open");
    return result;
  }
  try {
    result.task = queryTask(impl_->database, task_id);
    result.status = SceneStoreStatus::success();
    return result;
  } catch (const std::exception& error) {
    result.status = SceneStoreStatus::failure(error.what());
    return result;
  }
}

SceneStoreStatus SceneStore::upsertEmbeddingRecord(
    DurableEmbeddingRecord record) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->database) {
    return SceneStoreStatus::failure("scene store is not open");
  }
  try {
    validateEmbeddingRecord(record, impl_->durable_revision);
    Transaction transaction(impl_->database);
    const std::optional<DurableEmbeddingRecord> existing =
        queryEmbeddingRecord(impl_->database, record.object_id,
                             record.document_hash, record.model_id,
                             record.dimension, impl_->durable_revision);
    if (existing) {
      if (!embeddingRecordsEqual(*existing, record)) {
        return SceneStoreStatus::failure(
            "embedding identity was reused with different content");
      }
      transaction.commit();
      return SceneStoreStatus::success();
    }

    Statement insert(
        impl_->database,
        "INSERT INTO embedding_records("
        "object_id, document_hash, model_id, dimension, vector, "
        "created_scene_revision) VALUES(?1, ?2, ?3, ?4, ?5, ?6);");
    insert.bindInt64(1, static_cast<sqlite3_int64>(record.object_id));
    insert.bindText(2, record.document_hash);
    insert.bindText(3, record.model_id);
    insert.bindInt64(4, sqliteEmbeddingDimension(record.dimension));
    insert.bindBlob(5, encodeEmbeddingVector(record.vector));
    insert.bindInt64(6, sqliteRevision(record.created_scene_revision));
    insert.stepDone();
    transaction.commit();
    return SceneStoreStatus::success();
  } catch (const std::exception& error) {
    return SceneStoreStatus::failure(error.what());
  }
}

EmbeddingRecordLookupResult SceneStore::getEmbeddingRecord(
    SceneObjectId object_id,
    const std::string& document_hash,
    const std::string& model_id,
    std::size_t dimension) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  EmbeddingRecordLookupResult result;
  if (!impl_->database) {
    result.status = SceneStoreStatus::failure("scene store is not open");
    return result;
  }
  if (object_id < 0 || document_hash.empty() || model_id.empty()) {
    result.status = SceneStoreStatus::failure(
        "embedding lookup identity is incomplete");
    return result;
  }
  try {
    result.record = queryEmbeddingRecord(
        impl_->database, object_id, document_hash, model_id, dimension,
        impl_->durable_revision);
    result.status = SceneStoreStatus::success();
    return result;
  } catch (const std::exception& error) {
    result.status = SceneStoreStatus::failure(error.what());
    return result;
  }
}

EmbeddingRecordListResult SceneStore::listEmbeddingRecords() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  EmbeddingRecordListResult result;
  if (!impl_->database) {
    result.status = SceneStoreStatus::failure("scene store is not open");
    return result;
  }
  try {
    result.records = queryEmbeddingRecords(
        impl_->database, impl_->durable_revision, nullptr, 0);
    result.status = SceneStoreStatus::success();
    return result;
  } catch (const std::exception& error) {
    result.records.clear();
    result.status = SceneStoreStatus::failure(error.what());
    return result;
  }
}

EmbeddingRecordListResult SceneStore::listEmbeddingRecords(
    const std::string& model_id,
    std::size_t dimension) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  EmbeddingRecordListResult result;
  if (!impl_->database) {
    result.status = SceneStoreStatus::failure("scene store is not open");
    return result;
  }
  if (model_id.empty()) {
    result.status = SceneStoreStatus::failure(
        "embedding namespace model id is required");
    return result;
  }
  try {
    result.records = queryEmbeddingRecords(
        impl_->database, impl_->durable_revision, &model_id, dimension);
    result.status = SceneStoreStatus::success();
    return result;
  } catch (const std::exception& error) {
    result.records.clear();
    result.status = SceneStoreStatus::failure(error.what());
    return result;
  }
}

}  // namespace roomie
