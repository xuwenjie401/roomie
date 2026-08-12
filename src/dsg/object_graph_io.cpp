#include "roomie/dsg/object_graph_io.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace roomie {
namespace {

constexpr int kRoomieObjectGraphFormatVersion = 6;
constexpr const char* kRoomieObjectGraphFormat = "roomie_object_graph";
constexpr const char* kRoomieManualSceneGraphFormat = "roomie_manual_scene_graph";

using nlohmann::json;

void setError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool hasJsonExtension(const std::filesystem::path& path) {
  return lowercase(path.extension().string()) == ".json";
}

std::string saveTimeSuffix(TimeNanoseconds saved_time_ns) {
  std::time_t saved_time_sec = 0;
  if (saved_time_ns > 0) {
    saved_time_sec = static_cast<std::time_t>(saved_time_ns / 1000000000LL);
  } else {
    saved_time_sec = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
  }

  std::tm local_time{};
  if (localtime_r(&saved_time_sec, &local_time) == nullptr) {
    return "0000_0000";
  }

  std::ostringstream stream;
  stream << std::put_time(&local_time, "%m%d_%H%M");
  return stream.str();
}

std::string pathStem(const std::filesystem::path& path) {
  const std::string stem = path.stem().string();
  return stem.empty() ? "roomie_dsg" : stem;
}

std::string cleanSubdir(std::string value) {
  if (value.empty()) {
    return "snapshots";
  }
  for (char& c : value) {
    if (c == '\\') {
      c = '/';
    }
  }
  while (!value.empty() && value.front() == '/') {
    value.erase(value.begin());
  }
  while (!value.empty() && value.back() == '/') {
    value.pop_back();
  }
  return value.empty() ? "snapshots" : value;
}

json vector3fToJson(const Eigen::Vector3f& value) {
  return json::array({value.x(), value.y(), value.z()});
}

json vector3iToJson(const Eigen::Vector3i& value) {
  return json::array({value.x(), value.y(), value.z()});
}

json bboxToJson(const std::array<float, 4>& value) {
  return json::array({value[0], value[1], value[2], value[3]});
}

json vector2fToJson(const std::array<float, 2>& value) {
  return json::array({value[0], value[1]});
}

Eigen::Vector3f vector3fFromJson(const json& value) {
  Eigen::Vector3f result = Eigen::Vector3f::Zero();
  if (!value.is_array()) {
    return result;
  }
  for (int i = 0; i < 3 && i < static_cast<int>(value.size()); ++i) {
    result[i] = value.at(static_cast<std::size_t>(i)).get<float>();
  }
  return result;
}

Eigen::Vector3i vector3iFromJson(const json& value) {
  Eigen::Vector3i result = Eigen::Vector3i::Zero();
  if (!value.is_array()) {
    return result;
  }
  for (int i = 0; i < 3 && i < static_cast<int>(value.size()); ++i) {
    result[i] = value.at(static_cast<std::size_t>(i)).get<int>();
  }
  return result;
}

std::array<float, 4> bboxFromJson(const json& value) {
  std::array<float, 4> result = {0.0f, 0.0f, 0.0f, 0.0f};
  if (!value.is_array()) {
    return result;
  }
  for (int i = 0; i < 4 && i < static_cast<int>(value.size()); ++i) {
    result[static_cast<std::size_t>(i)] =
        value.at(static_cast<std::size_t>(i)).get<float>();
  }
  return result;
}

std::array<float, 2> vector2fFromJson(const json& value) {
  std::array<float, 2> result = {0.0f, 0.0f};
  if (!value.is_array()) {
    return result;
  }
  for (int i = 0; i < 2 && i < static_cast<int>(value.size()); ++i) {
    result[static_cast<std::size_t>(i)] =
        value.at(static_cast<std::size_t>(i)).get<float>();
  }
  return result;
}

std::string geometryStatusToString(InstanceGeometryStatus status) {
  switch (status) {
    case InstanceGeometryStatus::kUnchecked:
      return "unchecked";
    case InstanceGeometryStatus::kGood:
      return "good";
    case InstanceGeometryStatus::kBad:
      return "bad";
    case InstanceGeometryStatus::kEmpty:
      return "empty";
  }
  return "unchecked";
}

InstanceGeometryStatus geometryStatusFromString(const std::string& value) {
  const std::string lower = lowercase(value);
  if (lower == "good" || lower == "geometry_good") {
    return InstanceGeometryStatus::kGood;
  }
  if (lower == "bad" || lower == "geometry_bad") {
    return InstanceGeometryStatus::kBad;
  }
  if (lower == "empty" || lower == "geometry_empty") {
    return InstanceGeometryStatus::kEmpty;
  }
  return InstanceGeometryStatus::kUnchecked;
}

json voxelRefToJson(const VoxelRef& ref) {
  json value;
  value["block_index"] = vector3iToJson(ref.block_index);
  value["voxel_index"] = vector3iToJson(ref.voxel_index);
  return value;
}

VoxelRef voxelRefFromJson(const json& value) {
  VoxelRef ref;
  ref.block_index = vector3iFromJson(value.value("block_index", json::array()));
  ref.voxel_index = vector3iFromJson(value.value("voxel_index", json::array()));
  return ref;
}

template <typename T>
json vectorToJsonArray(const std::vector<T>& values) {
  json array = json::array();
  for (const T& value : values) {
    array.push_back(value);
  }
  return array;
}

template <typename T, typename Allocator>
json alignedVectorToJsonArray(const std::vector<T, Allocator>& values) {
  json array = json::array();
  for (const T& value : values) {
    array.push_back(value);
  }
  return array;
}

json labelWeightsToJson(const std::map<std::string, float>& weights) {
  json value = json::object();
  for (const auto& [label, weight] : weights) {
    value[label] = weight;
  }
  return value;
}

json semanticWeightsToJson(const std::map<int, float>& weights) {
  json array = json::array();
  for (const auto& [semantic_id, weight] : weights) {
    array.push_back({{"id", semantic_id}, {"weight", weight}});
  }
  return array;
}

std::map<std::string, float> labelWeightsFromJson(const json& value) {
  std::map<std::string, float> weights;
  if (!value.is_object()) {
    return weights;
  }
  for (auto it = value.begin(); it != value.end(); ++it) {
    if (it.value().is_number()) {
      weights[it.key()] = it.value().get<float>();
    }
  }
  return weights;
}

std::map<int, float> semanticWeightsFromJson(const json& value) {
  std::map<int, float> weights;
  if (value.is_array()) {
    for (const json& entry : value) {
      if (!entry.is_object()) {
        continue;
      }
      weights[entry.value("id", -1)] = entry.value("weight", 0.0f);
    }
  } else if (value.is_object()) {
    for (auto it = value.begin(); it != value.end(); ++it) {
      if (!it.value().is_number()) {
        continue;
      }
      try {
        weights[std::stoi(it.key())] = it.value().get<float>();
      } catch (const std::exception&) {
      }
    }
  }
  return weights;
}

json snapshotRefToJson(const ObjectSnapshotRef& snapshot) {
  json value;
  value["image_index"] = snapshot.image_index;
  value["source_frame_asset_id"] = snapshot.source_frame_asset_id;
  value["evidence_hash"] = snapshot.evidence_hash;
  value["bbox_xyxy"] = bboxToJson(snapshot.bbox_xyxy);
  value["crop_xywh"] = bboxToJson(snapshot.crop_xywh);
  value["crop_output_scale"] = snapshot.crop_output_scale;
  value["mask_source"] = snapshot.mask_source;
  value["mask_ref"] = snapshot.mask_ref;
  value["quality"] = snapshot.quality;
  value["quality_components"] = snapshot.quality_components;
  value["viewpoint"] = json{{"azimuth_rad", snapshot.viewpoint_azimuth_rad},
                             {"elevation_rad", snapshot.viewpoint_elevation_rad},
                             {"scale", snapshot.viewpoint_scale}};
  value["time_ns"] = snapshot.time_ns;
  value["camera_id"] = snapshot.camera_id;
  value["provenance"] =
      json{{"run_id", json{{"high", snapshot.provenance.run_id.high},
                             {"low", snapshot.provenance.run_id.low}}},
           {"frame_id", snapshot.provenance.frame_id},
           {"request_id", snapshot.provenance.request_id},
           {"sensor_time_ns", snapshot.provenance.sensor_time_ns},
           {"map_mode", static_cast<int>(snapshot.provenance.map_mode)},
           {"includes_current_frame",
            snapshot.provenance.includes_current_frame},
           {"causality_verified", snapshot.provenance.causality_verified},
           {"map",
            json{{"epoch_high", snapshot.provenance.map.map_epoch.high},
                 {"epoch_low", snapshot.provenance.map.map_epoch.low},
                 {"revision", snapshot.provenance.map.map_revision},
                 {"integrated_through_ns",
                  snapshot.provenance.map.integrated_through_ns}}},
           {"surface",
            json{{"epoch_high", snapshot.provenance.surface.map_epoch.high},
                 {"epoch_low", snapshot.provenance.surface.map_epoch.low},
                 {"revision", snapshot.provenance.surface.surface_revision},
                 {"source_map_revision",
                  snapshot.provenance.surface.source_map_revision}}}};
  return value;
}

ObjectSnapshotRef snapshotRefFromJson(const json& value) {
  ObjectSnapshotRef snapshot;
  if (!value.is_object()) {
    return snapshot;
  }
  snapshot.image_index = value.value("image_index", -1);
  snapshot.source_frame_asset_id =
      value.value("source_frame_asset_id", std::string());
  snapshot.evidence_hash = value.value("evidence_hash", std::string());
  snapshot.bbox_xyxy = bboxFromJson(value.value("bbox_xyxy", json::array()));
  snapshot.crop_xywh = bboxFromJson(value.value("crop_xywh", json::array()));
  snapshot.crop_output_scale = value.value(
      "crop_output_scale", std::array<float, 2>{1.0f, 1.0f});
  snapshot.mask_source =
      value.value("mask_source", std::string("bbox_fallback"));
  snapshot.mask_ref = value.value("mask_ref", std::string());
  snapshot.quality = value.value("quality", 0.0f);
  snapshot.quality_components = value.value(
      "quality_components", std::map<std::string, float>{});
  const json viewpoint = value.value("viewpoint", json::object());
  snapshot.viewpoint_azimuth_rad =
      viewpoint.value("azimuth_rad", 0.0f);
  snapshot.viewpoint_elevation_rad =
      viewpoint.value("elevation_rad", 0.0f);
  snapshot.viewpoint_scale = viewpoint.value("scale", 1.0f);
  snapshot.time_ns = value.value("time_ns", TimeNanoseconds{0});
  snapshot.camera_id = value.value("camera_id", std::string());
  const json provenance = value.value("provenance", json::object());
  const json run_id = provenance.value("run_id", json::object());
  snapshot.provenance.run_id.high =
      run_id.value("high", std::uint64_t{0});
  snapshot.provenance.run_id.low =
      run_id.value("low", std::uint64_t{0});
  snapshot.provenance.frame_id =
      provenance.value("frame_id", FrameId{0});
  snapshot.provenance.request_id =
      provenance.value("request_id", RequestId{0});
  snapshot.provenance.sensor_time_ns =
      provenance.value("sensor_time_ns", TimeNanoseconds{0});
  snapshot.provenance.map_mode =
      provenance.value("map_mode", 0) == 1 ? MapMode::kFrozen
                                            : MapMode::kOnline;
  snapshot.provenance.includes_current_frame =
      provenance.value("includes_current_frame", false);
  snapshot.provenance.causality_verified =
      provenance.value("causality_verified", false);
  const json map = provenance.value("map", json::object());
  snapshot.provenance.map.map_epoch.high =
      map.value("epoch_high", std::uint64_t{0});
  snapshot.provenance.map.map_epoch.low =
      map.value("epoch_low", std::uint64_t{0});
  snapshot.provenance.map.map_revision =
      map.value("revision", std::uint64_t{0});
  snapshot.provenance.map.integrated_through_ns =
      map.value("integrated_through_ns", TimeNanoseconds{0});
  const json surface = provenance.value("surface", json::object());
  snapshot.provenance.surface.map_epoch.high =
      surface.value("epoch_high", std::uint64_t{0});
  snapshot.provenance.surface.map_epoch.low =
      surface.value("epoch_low", std::uint64_t{0});
  snapshot.provenance.surface.surface_revision =
      surface.value("revision", std::uint64_t{0});
  snapshot.provenance.surface.source_map_revision =
      surface.value("source_map_revision", std::uint64_t{0});
  return snapshot;
}

json snapshotImageToJson(const ObjectSnapshotImage& image) {
  json value;
  value["image_index"] = image.image_index;
  value["uri"] = image.uri;
  value["width"] = image.width;
  value["height"] = image.height;
  value["encoding"] = image.encoding;
  value["time_ns"] = image.time_ns;
  value["camera_id"] = image.camera_id;
  if (!image.source_path.empty()) {
    value["source_path"] = image.source_path;
  }
  return value;
}

ObjectSnapshotImage snapshotImageFromJson(const json& value) {
  ObjectSnapshotImage image;
  if (!value.is_object()) {
    return image;
  }
  image.image_index = value.value("image_index", -1);
  image.uri = value.value("uri", std::string());
  image.width = value.value("width", 0);
  image.height = value.value("height", 0);
  image.encoding = value.value("encoding", std::string());
  image.time_ns = value.value("time_ns", TimeNanoseconds{0});
  image.camera_id = value.value("camera_id", std::string());
  image.source_path = value.value("source_path", std::string());
  return image;
}

json objectNodeToJson(const ObjectNode& object) {
  json value;
  value["object_id"] = object.object_id;
  value["semantic_id"] = object.semantic_id;
  value["name"] = object.name;
  value["label"] = object.label;
  value["description"] = object.description;
  value["center_world"] = vector3fToJson(object.center_world);
  value["size_m"] = vector3fToJson(object.size_m);
  value["yaw_rad"] = object.yaw_rad;
  value["confidence"] = object.confidence;
  value["confidence_mass"] = object.confidence_mass;
  value["object_quality_score"] = object.object_quality_score;
  value["geometry_score"] = object.geometry_score;
  value["geometry_shell_ratio"] = object.geometry_shell_ratio;
  value["geometry_extent_score"] = object.geometry_extent_score;
  value["geometry_leak_ratio"] = object.geometry_leak_ratio;
  value["geometry_cavity_ratio"] = object.geometry_cavity_ratio;
  value["geometry_in_box_points"] = object.geometry_in_box_points;
  value["geometry_shell_points"] = object.geometry_shell_points;
  value["geometry_unique_voxels"] = object.geometry_unique_voxels;
  value["geometry_expanded_points"] = object.geometry_expanded_points;
  value["geometry_bad_count"] = object.geometry_bad_count;
  value["support_count"] = object.support_count;
  value["high_quality_observation_count"] = object.high_quality_observation_count;
  value["high_quality_observation_mass"] = object.high_quality_observation_mass;
  value["active"] = object.active;
  value["publishable"] = object.publishable;
  value["existence_log_odds"] = object.existence_log_odds;
  value["last_presence_evidence_ns"] = object.last_presence_evidence_ns;
  value["last_presence_evidence_reason"] =
      object.last_presence_evidence_reason;
  value["geometry_status"] = geometryStatusToString(object.geometry_status);
  value["geometry_evaluation_obb_revision"] = object.geometry_evaluation_obb_revision;
  value["geometry_evaluation_map_version"] = object.geometry_evaluation_map_version;
  value["first_seen_ns"] = object.first_seen_ns;
  value["last_seen_ns"] = object.last_seen_ns;
  value["last_geometry_check_ns"] = object.last_geometry_check_ns;
  value["geometry_evaluated_center_world"] =
      vector3fToJson(object.geometry_evaluated_center_world);
  value["geometry_evaluated_size_m"] = vector3fToJson(object.geometry_evaluated_size_m);
  value["geometry_evaluated_yaw_rad"] = object.geometry_evaluated_yaw_rad;
  value["geometry_evaluation_reason"] = object.geometry_evaluation_reason;
  value["source_track_ids"] = vectorToJsonArray(object.source_track_ids);
  value["source_cameras"] = vectorToJsonArray(object.source_cameras);
  value["observation_timestamps_ns"] = vectorToJsonArray(object.observation_timestamps_ns);
  if (object.snapshot.valid()) {
    value["snapshot"] = snapshotRefToJson(object.snapshot);
  }
  value["near_surface_voxels"] = json::array();
  for (const VoxelRef& ref : object.near_surface_voxels) {
    value["near_surface_voxels"].push_back(voxelRefToJson(ref));
  }
  value["label_weights"] = labelWeightsToJson(object.label_weights);
  value["semantic_weights"] = semanticWeightsToJson(object.semantic_weights);
  return value;
}

ObjectNode objectNodeFromJson(const json& value) {
  ObjectNode object;
  object.object_id = value.value("object_id", -1);
  object.semantic_id = value.value("semantic_id", -1);
  object.name = value.value("name", std::string());
  object.label = value.value("label", std::string());
  object.description = value.value("description", std::string());
  object.center_world = vector3fFromJson(value.value("center_world", json::array()));
  object.size_m = vector3fFromJson(value.value("size_m", json::array()));
  object.yaw_rad = value.value("yaw_rad", 0.0f);
  object.confidence = value.value("confidence", 0.0f);
  object.confidence_mass = value.value("confidence_mass", 0.0f);
  object.object_quality_score = value.value("object_quality_score", 0.0f);
  object.geometry_score = value.value("geometry_score", 0.0f);
  object.geometry_shell_ratio = value.value("geometry_shell_ratio", 0.0f);
  object.geometry_extent_score = value.value("geometry_extent_score", 0.0f);
  object.geometry_leak_ratio = value.value("geometry_leak_ratio", 1.0f);
  object.geometry_cavity_ratio = value.value("geometry_cavity_ratio", 0.0f);
  object.geometry_in_box_points = value.value("geometry_in_box_points", 0);
  object.geometry_shell_points = value.value("geometry_shell_points", 0);
  object.geometry_unique_voxels = value.value("geometry_unique_voxels", 0);
  object.geometry_expanded_points = value.value("geometry_expanded_points", 0);
  object.geometry_bad_count = value.value("geometry_bad_count", 0);
  object.support_count = value.value("support_count", 0);
  object.high_quality_observation_count =
      value.value("high_quality_observation_count", 0);
  object.high_quality_observation_mass =
      value.value("high_quality_observation_mass", 0.0f);
  object.active = value.value("active", true);
  object.publishable = value.value("publishable", true);
  object.existence_log_odds = value.value(
      "existence_log_odds", object.active ? 1.0986123f : -1.0986123f);
  object.last_presence_evidence_ns =
      value.value("last_presence_evidence_ns", TimeNanoseconds{0});
  object.last_presence_evidence_reason =
      value.value("last_presence_evidence_reason", std::string("legacy_restore"));
  object.geometry_status =
      geometryStatusFromString(value.value("geometry_status", std::string()));
  object.geometry_evaluation_obb_revision =
      value.value("geometry_evaluation_obb_revision", std::uint64_t{0});
  object.geometry_evaluation_map_version =
      value.value("geometry_evaluation_map_version", std::uint64_t{0});
  object.first_seen_ns = value.value("first_seen_ns", TimeNanoseconds{0});
  object.last_seen_ns = value.value("last_seen_ns", TimeNanoseconds{0});
  object.last_geometry_check_ns =
      value.value("last_geometry_check_ns", TimeNanoseconds{0});
  object.geometry_evaluated_center_world =
      vector3fFromJson(value.value("geometry_evaluated_center_world", json::array()));
  object.geometry_evaluated_size_m =
      vector3fFromJson(value.value("geometry_evaluated_size_m", json::array()));
  object.geometry_evaluated_yaw_rad = value.value("geometry_evaluated_yaw_rad", 0.0f);
  object.geometry_evaluation_reason =
      value.value("geometry_evaluation_reason", std::string());
  object.source_track_ids =
      value.value("source_track_ids", std::vector<int>());
  object.source_cameras =
      value.value("source_cameras", std::vector<std::string>());
  object.observation_timestamps_ns =
      value.value("observation_timestamps_ns", std::vector<TimeNanoseconds>());
  object.snapshot = snapshotRefFromJson(value.value("snapshot", json::object()));

  const json near_surface_voxels = value.value("near_surface_voxels", json::array());
  if (near_surface_voxels.is_array()) {
    object.near_surface_voxels.reserve(near_surface_voxels.size());
    for (const json& ref_json : near_surface_voxels) {
      object.near_surface_voxels.push_back(voxelRefFromJson(ref_json));
    }
  }
  object.label_weights = labelWeightsFromJson(value.value("label_weights", json::object()));
  object.semantic_weights =
      semanticWeightsFromJson(value.value("semantic_weights", json::array()));
  return object;
}

json roomNodeToJson(const RoomNode& room) {
  json value;
  value["room_id"] = room.room_id;
  value["revision"] = room.revision;
  value["label"] = room.label;
  value["color"] = room.color;
  value["center_world"] = vector3fToJson(room.center_world);
  value["size_m"] = vector3fToJson(room.size_m);
  if (room.has_xy_bounds) {
    value["min_xy"] = vector2fToJson(room.min_xy);
    value["max_xy"] = vector2fToJson(room.max_xy);
  }
  value["height_m"] = room.height_m;
  if (!room.attributes.empty()) {
    value["attributes"] = room.attributes;
  }
  return value;
}

RoomNode roomNodeFromJson(const json& value) {
  RoomNode room;
  room.room_id = value.value("room_id", value.value("id", -1));
  room.revision = value.value("revision", std::uint64_t{0});
  room.label = value.value("label", value.value("name", std::string()));
  room.color = value.value("color", std::string());
  room.center_world =
      vector3fFromJson(value.value("center_world", json::array()));
  room.size_m = vector3fFromJson(value.value("size_m", json::array()));
  if (value.contains("min_xy") && value.contains("max_xy") &&
      value.at("min_xy").is_array() && value.at("max_xy").is_array() &&
      value.at("min_xy").size() >= 2 && value.at("max_xy").size() >= 2) {
    room.min_xy = vector2fFromJson(value.at("min_xy"));
    room.max_xy = vector2fFromJson(value.at("max_xy"));
    room.has_xy_bounds = true;
  } else if (room.size_m.x() > 0.0f && room.size_m.y() > 0.0f) {
    room.min_xy = {room.center_world.x() - 0.5f * room.size_m.x(),
                   room.center_world.y() - 0.5f * room.size_m.y()};
    room.max_xy = {room.center_world.x() + 0.5f * room.size_m.x(),
                   room.center_world.y() + 0.5f * room.size_m.y()};
    room.has_xy_bounds = true;
  }
  room.height_m = value.value("height_m", room.size_m.z());
  if (value.contains("attributes") && value.at("attributes").is_object()) {
    for (auto it = value.at("attributes").begin();
         it != value.at("attributes").end(); ++it) {
      if (it.value().is_string()) {
        room.attributes[it.key()] = it.value().get<std::string>();
      }
    }
  }
  return room;
}

const char* entityTypeToString(SceneEntityType type) {
  switch (type) {
    case SceneEntityType::kObject:
      return "object";
    case SceneEntityType::kRoom:
      return "room";
    case SceneEntityType::kFurniture:
      return "furniture";
  }
  throw std::invalid_argument("unknown scene entity type");
}

SceneEntityType entityTypeFromString(const std::string& value) {
  const std::string normalized = lowercase(value);
  if (normalized == "object") {
    return SceneEntityType::kObject;
  }
  if (normalized == "room") {
    return SceneEntityType::kRoom;
  }
  if (normalized == "furniture") {
    return SceneEntityType::kFurniture;
  }
  throw std::invalid_argument("unknown scene entity type: " + value);
}

json entityRefToJson(SceneEntityRef ref) {
  return json{{"type", entityTypeToString(ref.type)}, {"id", ref.id}};
}

SceneEntityRef entityRefFromJson(const json& value,
                                 SceneEntityType default_type,
                                 int default_id) {
  if (!value.is_object()) {
    return SceneEntityRef{default_type, default_id};
  }
  const std::string type = value.value("type", std::string());
  return SceneEntityRef{
      type.empty() ? default_type : entityTypeFromString(type),
      value.value("id", default_id)};
}

json relationToJson(const ObjectRelation& relation) {
  json value;
  const SceneEntityRef source = relationSource(relation);
  const SceneEntityRef target = relationTarget(relation);
  value["source"] = entityRefToJson(source);
  value["target"] = entityRefToJson(target);
  // Furniture uses the same canonical object id, so legacy readers can still
  // identify that physical endpoint even though they lose the role type.
  if (entityBackedByObject(source)) {
    value["source_object_id"] = source.id;
  }
  if (entityBackedByObject(target)) {
    value["target_object_id"] = target.id;
  }
  value["relation_type"] = relation.relation_type;
  value["confidence"] = relation.confidence;
  value["description"] = relation.description;
  value["revision"] = relation.revision;
  value["derived"] = relation.derived;
  return value;
}

ObjectRelation relationFromJson(const json& value) {
  ObjectRelation relation;
  relation.source_object_id = value.value("source_object_id", -1);
  relation.target_object_id = value.value("target_object_id", -1);
  const SceneEntityRef source = entityRefFromJson(
      value.value("source", json::object()), SceneEntityType::kObject,
      relation.source_object_id);
  const SceneEntityRef target = entityRefFromJson(
      value.value("target", json::object()), SceneEntityType::kObject,
      relation.target_object_id);
  setRelationEndpoints(&relation, source, target);
  relation.relation_type = value.value("relation_type", std::string());
  relation.confidence = value.value("confidence", 0.0f);
  relation.description = value.value("description", std::string());
  relation.revision = value.value("revision", std::uint64_t{0});
  relation.derived = value.value(
      "derived", relation.relation_type == "room_contains_object" &&
                     source.type == SceneEntityType::kRoom &&
                     target.type == SceneEntityType::kObject);
  return relation;
}

json furnitureRoleToJson(const FurnitureRole& role) {
  return json{{"object_id", role.object_id},
              {"revision", role.revision},
              {"classification_label", role.classification_label}};
}

FurnitureRole furnitureRoleFromJson(const json& value) {
  FurnitureRole role;
  role.object_id = value.value("object_id", -1);
  role.revision = value.value("revision", std::uint64_t{0});
  role.classification_label =
      value.value("classification_label", std::string());
  return role;
}

json objectGraphToJson(const ObjectGraphSnapshot& snapshot,
                       const std::string& world_frame,
                       TimeNanoseconds saved_time_ns) {
  json root;
  root["format"] = kRoomieObjectGraphFormat;
  root["format_version"] = kRoomieObjectGraphFormatVersion;
  root["world_frame"] = world_frame;
  root["saved_time_ns"] = saved_time_ns;
  root["next_object_id"] = snapshot.next_object_id;
  root["objects"] = json::array();
  for (const ObjectNode& object : snapshot.objects) {
    root["objects"].push_back(objectNodeToJson(object));
  }
  root["rooms"] = json::array();
  for (const RoomNode& room : snapshot.rooms) {
    root["rooms"].push_back(roomNodeToJson(room));
  }
  root["furniture"] = json::array();
  for (const FurnitureRole& role : snapshot.furniture) {
    root["furniture"].push_back(furnitureRoleToJson(role));
  }
  root["relations"] = json::array();
  for (const ObjectRelation& relation : snapshot.relations) {
    root["relations"].push_back(relationToJson(relation));
  }
  root["snapshot_images"] = json::array();
  for (const ObjectSnapshotImage& image : snapshot.snapshot_images) {
    root["snapshot_images"].push_back(snapshotImageToJson(image));
  }
  if (!snapshot.import_warnings.empty()) {
    root["migration_warnings"] = snapshot.import_warnings;
  }
  root["notes"] =
      "near_surface_voxels are cached map-local references for debugging; "
      "runtime geometry checks rebuild object-map coupling from the current map.";
  return root;
}

json snapshotToJson(const ObjectGraphSnapshot& snapshot,
                    const std::string& world_frame,
                    TimeNanoseconds saved_time_ns) {
  const json object_graph = objectGraphToJson(snapshot, world_frame, saved_time_ns);
  if (!snapshot.has_scene_graph_envelope || snapshot.scene_graph_json.empty()) {
    return object_graph;
  }
  try {
    json root = json::parse(snapshot.scene_graph_json);
    // Schema v4 has exactly one canonical object list. Preserve envelope-only
    // metadata (map, root, source paths, etc.) but remove the legacy nested
    // object graph before overwriting every canonical scene field.
    root.erase("object_graph");
    root["format"] = kRoomieManualSceneGraphFormat;
    root["format_version"] = kRoomieObjectGraphFormatVersion;
    root["world_frame"] = world_frame;
    root["saved_time_ns"] = saved_time_ns;
    root["next_object_id"] = snapshot.next_object_id;
    root["objects"] = object_graph.at("objects");
    root["rooms"] = object_graph.at("rooms");
    root["furniture"] = object_graph.at("furniture");
    root["relations"] = object_graph.at("relations");
    root["snapshot_images"] = object_graph.at("snapshot_images");
    if (object_graph.contains("migration_warnings")) {
      root["migration_warnings"] = object_graph.at("migration_warnings");
    } else {
      root.erase("migration_warnings");
    }
    if (root.contains("root") && root.at("root").is_object()) {
      json object_ids = json::array();
      for (const ObjectNode& object : snapshot.objects) {
        object_ids.push_back(object.object_id);
      }
      json room_ids = json::array();
      for (const RoomNode& room : snapshot.rooms) {
        room_ids.push_back(room.room_id);
      }
      root["root"]["object_ids"] = std::move(object_ids);
      root["root"]["room_ids"] = std::move(room_ids);
      json furniture_ids = json::array();
      for (const FurnitureRole& role : snapshot.furniture) {
        furniture_ids.push_back(role.object_id);
      }
      root["root"]["furniture_ids"] = std::move(furniture_ids);
    }
    return root;
  } catch (const std::exception&) {
    return object_graph;
  }
}

ObjectGraphSnapshot snapshotFromJson(const json& root) {
  ObjectGraphSnapshot snapshot;
  snapshot.schema_version = root.value("format_version", 1);
  snapshot.next_object_id = root.value("next_object_id", 0);

  const json objects = root.value("objects", json::array());
  if (objects.is_array()) {
    snapshot.objects.reserve(objects.size());
    for (const json& object_json : objects) {
      snapshot.objects.push_back(objectNodeFromJson(object_json));
    }
  }

  const json rooms = root.value("rooms", json::array());
  if (rooms.is_array()) {
    snapshot.rooms.reserve(rooms.size());
    for (const json& room_json : rooms) {
      snapshot.rooms.push_back(roomNodeFromJson(room_json));
    }
  }

  const json furniture = root.value("furniture", json::array());
  if (furniture.is_array()) {
    snapshot.furniture.reserve(furniture.size());
    for (const json& role_json : furniture) {
      snapshot.furniture.push_back(furnitureRoleFromJson(role_json));
    }
  }

  const json relations = root.value("relations", json::array());
  if (relations.is_array()) {
    snapshot.relations.reserve(relations.size());
    for (const json& relation_json : relations) {
      snapshot.relations.push_back(relationFromJson(relation_json));
    }
  }
  const json snapshot_images = root.value("snapshot_images", json::array());
  if (snapshot_images.is_array()) {
    snapshot.snapshot_images.reserve(snapshot_images.size());
    for (const json& image_json : snapshot_images) {
      snapshot.snapshot_images.push_back(snapshotImageFromJson(image_json));
    }
  }
  snapshot.import_warnings =
      root.value("migration_warnings", std::vector<std::string>());
  return snapshot;
}

bool indexObjectsById(const json& objects,
                      std::map<int, const json*>* indexed,
                      std::string* conflict) {
  if (!objects.is_array()) {
    if (conflict != nullptr) {
      *conflict = "objects must be an array";
    }
    return false;
  }
  for (const json& object : objects) {
    if (!object.is_object()) {
      if (conflict != nullptr) {
        *conflict = "objects contains a non-object entry";
      }
      return false;
    }
    const int object_id = object.value("object_id", -1);
    if (object_id < 0 || !indexed->emplace(object_id, &object).second) {
      if (conflict != nullptr) {
        *conflict = "objects contains an invalid or duplicate object_id " +
                    std::to_string(object_id);
      }
      return false;
    }
  }
  return true;
}

bool compatibleDuplicateObjects(const json& top_objects,
                                const json& nested_objects,
                                std::string* conflict) {
  std::map<int, const json*> top;
  std::map<int, const json*> nested;
  if (!indexObjectsById(top_objects, &top, conflict) ||
      !indexObjectsById(nested_objects, &nested, conflict)) {
    return false;
  }
  if (top.size() != nested.size()) {
    if (conflict != nullptr) {
      *conflict = "top-level and nested objects contain different id sets";
    }
    return false;
  }
  const auto is_legacy_relationship_field = [](const std::string& field) {
    static const std::set<std::string> fields = {
        "parent_room_ids", "parent_object_ids", "child_object_ids",
        "room_id", "rooms"};
    return fields.count(field) != 0;
  };
  for (const auto& [object_id, top_object] : top) {
    const auto nested_it = nested.find(object_id);
    if (nested_it == nested.end()) {
      if (conflict != nullptr) {
        *conflict = "top-level object " + std::to_string(object_id) +
                    " is absent from nested object_graph";
      }
      return false;
    }
    for (auto field_it = top_object->begin(); field_it != top_object->end();
         ++field_it) {
      const std::string& field = field_it.key();
      if (!is_legacy_relationship_field(field) &&
          nested_it->second->contains(field) &&
          field_it.value() != nested_it->second->at(field)) {
        if (conflict != nullptr) {
          *conflict = "object " + std::to_string(object_id) + " field '" +
                      field + "' differs between top-level and nested lists";
        }
        return false;
      }
    }
  }
  return true;
}

json mergeCompatibleDuplicateObjects(const json& top_objects,
                                     const json& nested_objects) {
  std::map<int, const json*> top;
  std::string ignored;
  indexObjectsById(top_objects, &top, &ignored);
  json merged = nested_objects;
  static const std::set<std::string> kLegacyRelationshipFields = {
      "parent_room_ids", "parent_object_ids", "child_object_ids",
      "room_id", "rooms"};
  for (json& object : merged) {
    const auto top_it = top.find(object.value("object_id", -1));
    if (top_it == top.end()) {
      continue;
    }
    for (auto field_it = top_it->second->begin();
         field_it != top_it->second->end(); ++field_it) {
      if (!object.contains(field_it.key()) &&
          kLegacyRelationshipFields.count(field_it.key()) == 0) {
        object[field_it.key()] = field_it.value();
      }
    }
  }
  return merged;
}

bool validateCanonicalSnapshot(const ObjectGraphSnapshot& snapshot,
                               std::string* error) {
  std::set<int> object_ids;
  for (const ObjectNode& object : snapshot.objects) {
    if (object.object_id < 0 || !object_ids.insert(object.object_id).second) {
      setError(error, "canonical objects contains an invalid or duplicate id");
      return false;
    }
  }
  std::set<int> room_ids;
  for (const RoomNode& room : snapshot.rooms) {
    if (room.room_id < 0 || !room_ids.insert(room.room_id).second) {
      setError(error, "canonical rooms contains an invalid or duplicate id");
      return false;
    }
  }
  std::set<int> furniture_ids;
  for (const FurnitureRole& role : snapshot.furniture) {
    if (role.object_id < 0 || object_ids.count(role.object_id) == 0U ||
        role.classification_label.empty() ||
        !furniture_ids.insert(role.object_id).second) {
      setError(error,
               "canonical furniture contains an invalid or duplicate role");
      return false;
    }
  }
  for (const ObjectRelation& relation : snapshot.relations) {
    const SceneEntityRef source = relationSource(relation);
    const SceneEntityRef target = relationTarget(relation);
    if (!source.valid() || !target.valid() || relation.relation_type.empty()) {
      setError(error, "canonical relations contains an invalid endpoint or type");
      return false;
    }
    const auto exists = [&](SceneEntityRef endpoint) {
      if (endpoint.type == SceneEntityType::kRoom) {
        return room_ids.count(endpoint.id) != 0U;
      }
      if (endpoint.type == SceneEntityType::kFurniture) {
        return furniture_ids.count(endpoint.id) != 0U;
      }
      return object_ids.count(endpoint.id) != 0U;
    };
    if (!exists(source) || !exists(target)) {
      setError(error, "canonical relation endpoint does not exist");
      return false;
    }
  }
  return true;
}

bool writeJsonAtomic(const json& value, const std::filesystem::path& path, std::string* error) {
  try {
    if (path.empty()) {
      setError(error, "empty output path");
      return false;
    }
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }

    const std::filesystem::path tmp_path = path.string() + ".tmp";
    {
      std::ofstream stream(tmp_path, std::ios::out | std::ios::trunc);
      if (!stream) {
        setError(error, "failed to open temporary file: " + tmp_path.string());
        return false;
      }
      stream << value.dump(2) << '\n';
      if (!stream) {
        setError(error, "failed to write temporary file: " + tmp_path.string());
        return false;
      }
    }
    std::filesystem::rename(tmp_path, path);
    return true;
  } catch (const std::exception& ex) {
    setError(error, ex.what());
    return false;
  }
}

}  // namespace

bool resolveObjectGraphSavePaths(const std::string& configured_path,
                                 TimeNanoseconds saved_time_ns,
                                 ObjectGraphSavePaths* paths,
                                 std::string* error) {
  return resolveObjectGraphSavePaths(configured_path,
                                     saved_time_ns,
                                     "snapshots",
                                     paths,
                                     error);
}

bool resolveObjectGraphSavePaths(const std::string& configured_path,
                                 TimeNanoseconds saved_time_ns,
                                 const std::string& snapshot_image_subdir,
                                 ObjectGraphSavePaths* paths,
                                 std::string* error) {
  if (paths == nullptr) {
    setError(error, "null ObjectGraphSavePaths output");
    return false;
  }
  if (configured_path.empty()) {
    setError(error, "persistence.scene_graph_save_path is empty");
    return false;
  }

  const std::filesystem::path base_path(configured_path);
  const std::string snapshot_subdir = cleanSubdir(snapshot_image_subdir);
  paths->primary_path.clear();
  paths->latest_path.clear();
  paths->snapshot_image_dir.clear();
  paths->snapshot_uri_prefix.clear();
  if (hasJsonExtension(base_path)) {
    paths->primary_path = base_path;
    const std::string dir_name = pathStem(base_path) + "_snapshots";
    paths->snapshot_image_dir =
        base_path.parent_path().empty() ? std::filesystem::path(dir_name)
                                        : base_path.parent_path() / dir_name;
    paths->snapshot_uri_prefix = dir_name;
    return true;
  }

  std::ostringstream filename;
  filename << "roomie_dsg_" << saveTimeSuffix(saved_time_ns) << ".json";
  paths->primary_path = base_path / filename.str();
  paths->latest_path = base_path / "latest.json";
  const std::string snapshot_run_dir = pathStem(paths->primary_path);
  paths->snapshot_image_dir = base_path / snapshot_subdir / snapshot_run_dir;
  paths->snapshot_uri_prefix = snapshot_subdir + "/" + snapshot_run_dir;
  return true;
}

bool saveObjectGraphSnapshotJsonAtomic(const ObjectGraphSnapshot& snapshot,
                                       const std::string& world_frame,
                                       TimeNanoseconds saved_time_ns,
                                       const std::filesystem::path& path,
                                       std::string* error) {
  if (!validateCanonicalSnapshot(snapshot, error)) {
    return false;
  }
  return writeJsonAtomic(snapshotToJson(snapshot, world_frame, saved_time_ns), path, error);
}

bool saveObjectGraphSnapshotJson(const ObjectGraphSnapshot& snapshot,
                                 const std::string& world_frame,
                                 TimeNanoseconds saved_time_ns,
                                 const std::string& configured_path,
                                 ObjectGraphSavePaths* paths,
                                 std::string* error) {
  if (!validateCanonicalSnapshot(snapshot, error)) {
    return false;
  }
  ObjectGraphSavePaths resolved;
  if (!resolveObjectGraphSavePaths(configured_path, saved_time_ns, &resolved, error)) {
    return false;
  }

  const json root = snapshotToJson(snapshot, world_frame, saved_time_ns);
  if (!writeJsonAtomic(root, resolved.primary_path, error)) {
    return false;
  }
  if (!resolved.latest_path.empty() &&
      !writeJsonAtomic(root, resolved.latest_path, error)) {
    return false;
  }
  if (paths != nullptr) {
    *paths = resolved;
  }
  return true;
}

bool loadObjectGraphSnapshotJson(const std::filesystem::path& path,
                                 ObjectGraphSnapshot* snapshot,
                                 std::string* world_frame,
                                 std::string* error) {
  if (snapshot == nullptr) {
    setError(error, "null ObjectGraphSnapshot output");
    return false;
  }
  try {
    std::ifstream stream(path);
    if (!stream) {
      setError(error, "failed to open DSG JSON: " + path.string());
      return false;
    }
    const json root = json::parse(stream);
    const std::string format = root.value("format", std::string());
    json canonical_root = root;
    bool manual_envelope = false;
    if (format == kRoomieManualSceneGraphFormat) {
      manual_envelope = true;
      const int envelope_version = root.value("format_version", 1);
      if (envelope_version > kRoomieObjectGraphFormatVersion) {
        setError(error, "unsupported manual scene graph version: " +
                            std::to_string(envelope_version));
        return false;
      }
      const bool has_nested =
          root.contains("object_graph") && root.at("object_graph").is_object();
      const bool has_top_objects =
          root.contains("objects") && root.at("objects").is_array();
      if (!has_nested && !has_top_objects) {
        setError(error,
                 "manual scene graph contains neither canonical objects nor "
                 "a legacy object_graph");
        return false;
      }
      if (envelope_version >= 3 && has_nested) {
        setError(error,
                 "schema v3+ manual scene graph must not contain a nested "
                 "object_graph; objects have a single canonical top-level list");
        return false;
      }
      if (has_nested) {
        canonical_root = root.at("object_graph");
        const std::string nested_format =
            canonical_root.value("format", std::string());
        if (!nested_format.empty() &&
            nested_format != kRoomieObjectGraphFormat) {
          setError(error, "unsupported embedded object graph format: " +
                              nested_format);
          return false;
        }
        const int nested_version = canonical_root.value("format_version", 1);
        if (nested_version > kRoomieObjectGraphFormatVersion) {
          setError(error, "unsupported embedded DSG JSON version: " +
                              std::to_string(nested_version));
          return false;
        }
        if (has_top_objects) {
          std::string conflict;
          if (!compatibleDuplicateObjects(root.at("objects"),
                                          canonical_root.value(
                                              "objects", json::array()),
                                          &conflict)) {
            setError(error,
                     "manual scene graph object conflict: " + conflict);
            return false;
          }
          canonical_root["objects"] = mergeCompatibleDuplicateObjects(
              root.at("objects"), canonical_root.at("objects"));
          canonical_root["migration_warnings"].push_back(
              "legacy manual envelope contained duplicate compatible object "
              "lists; nested object_graph objects were selected");
        }
        // Legacy manual envelopes own rooms and cross-layer relations at the
        // top level. Overlay them onto the selected object graph once.
        for (const char* field : {"rooms", "furniture", "relations",
                                  "snapshot_images"}) {
          if (root.contains(field)) {
            canonical_root[field] = root.at(field);
          }
        }
        if (root.contains("next_object_id")) {
          canonical_root["next_object_id"] = root.at("next_object_id");
        }
      }
    } else if (!format.empty() && format != kRoomieObjectGraphFormat) {
      setError(error, "unsupported DSG JSON format: " + format);
      return false;
    }

    const std::string object_graph_format =
        canonical_root.value("format", std::string());
    if (!object_graph_format.empty() && object_graph_format != kRoomieObjectGraphFormat) {
      // A schema-v3+ manual envelope is itself the canonical graph root.
      if (!(manual_envelope &&
            object_graph_format == kRoomieManualSceneGraphFormat)) {
        setError(error, "unsupported embedded object graph format: " + object_graph_format);
        return false;
      }
    }
    const int version = canonical_root.value("format_version", 1);
    if (version > kRoomieObjectGraphFormatVersion) {
      setError(error, "unsupported DSG JSON version: " + std::to_string(version));
      return false;
    }
    *snapshot = snapshotFromJson(canonical_root);
    if (!validateCanonicalSnapshot(*snapshot, error)) {
      return false;
    }
    if (manual_envelope) {
      snapshot->has_scene_graph_envelope = true;
      snapshot->scene_graph_json = root.dump();
    }
    const std::filesystem::path base_dir = path.parent_path();
    for (ObjectSnapshotImage& image : snapshot->snapshot_images) {
      if (!image.source_path.empty() && std::filesystem::exists(image.source_path)) {
        continue;
      }
      if (image.uri.empty()) {
        continue;
      }
      const std::filesystem::path uri_path(image.uri);
      image.source_path =
          (uri_path.is_absolute() ? uri_path : base_dir / uri_path).string();
    }
    if (world_frame != nullptr) {
      *world_frame = canonical_root.value(
          "world_frame",
          root.value("world_frame", std::string()));
    }
    return true;
  } catch (const std::exception& ex) {
    setError(error, ex.what());
    return false;
  }
}

}  // namespace roomie
