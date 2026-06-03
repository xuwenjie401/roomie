#include "roomie/dsg/object_graph_io.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

#include <nlohmann/json.hpp>

namespace roomie {
namespace {

constexpr int kRoomieObjectGraphFormatVersion = 1;
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

json vector3fToJson(const Eigen::Vector3f& value) {
  return json::array({value.x(), value.y(), value.z()});
}

json vector3iToJson(const Eigen::Vector3i& value) {
  return json::array({value.x(), value.y(), value.z()});
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

json objectNodeToJson(const ObjectNode& object) {
  json value;
  value["object_id"] = object.object_id;
  value["semantic_id"] = object.semantic_id;
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

json relationToJson(const ObjectRelation& relation) {
  json value;
  value["source_object_id"] = relation.source_object_id;
  value["target_object_id"] = relation.target_object_id;
  value["relation_type"] = relation.relation_type;
  value["confidence"] = relation.confidence;
  value["description"] = relation.description;
  return value;
}

ObjectRelation relationFromJson(const json& value) {
  ObjectRelation relation;
  relation.source_object_id = value.value("source_object_id", -1);
  relation.target_object_id = value.value("target_object_id", -1);
  relation.relation_type = value.value("relation_type", std::string());
  relation.confidence = value.value("confidence", 0.0f);
  relation.description = value.value("description", std::string());
  return relation;
}

json snapshotToJson(const ObjectGraphSnapshot& snapshot,
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
  root["relations"] = json::array();
  for (const ObjectRelation& relation : snapshot.relations) {
    root["relations"].push_back(relationToJson(relation));
  }
  root["notes"] =
      "near_surface_voxels are cached map-local references for debugging; "
      "runtime geometry checks rebuild object-map coupling from the current map.";
  return root;
}

ObjectGraphSnapshot snapshotFromJson(const json& root) {
  ObjectGraphSnapshot snapshot;
  snapshot.next_object_id = root.value("next_object_id", 0);

  const json objects = root.value("objects", json::array());
  if (objects.is_array()) {
    snapshot.objects.reserve(objects.size());
    for (const json& object_json : objects) {
      snapshot.objects.push_back(objectNodeFromJson(object_json));
    }
  }

  const json relations = root.value("relations", json::array());
  if (relations.is_array()) {
    snapshot.relations.reserve(relations.size());
    for (const json& relation_json : relations) {
      snapshot.relations.push_back(relationFromJson(relation_json));
    }
  }
  return snapshot;
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
  if (paths == nullptr) {
    setError(error, "null ObjectGraphSavePaths output");
    return false;
  }
  if (configured_path.empty()) {
    setError(error, "persistence.instance_map_save_path is empty");
    return false;
  }

  const std::filesystem::path base_path(configured_path);
  paths->primary_path.clear();
  paths->latest_path.clear();
  if (hasJsonExtension(base_path)) {
    paths->primary_path = base_path;
    return true;
  }

  std::ostringstream filename;
  filename << "roomie_dsg_" << saveTimeSuffix(saved_time_ns) << ".json";
  paths->primary_path = base_path / filename.str();
  paths->latest_path = base_path / "latest.json";
  return true;
}

bool saveObjectGraphSnapshotJsonAtomic(const ObjectGraphSnapshot& snapshot,
                                       const std::string& world_frame,
                                       TimeNanoseconds saved_time_ns,
                                       const std::filesystem::path& path,
                                       std::string* error) {
  return writeJsonAtomic(snapshotToJson(snapshot, world_frame, saved_time_ns), path, error);
}

bool saveObjectGraphSnapshotJson(const ObjectGraphSnapshot& snapshot,
                                 const std::string& world_frame,
                                 TimeNanoseconds saved_time_ns,
                                 const std::string& configured_path,
                                 ObjectGraphSavePaths* paths,
                                 std::string* error) {
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
    const json* object_graph_root = &root;
    if (format == kRoomieManualSceneGraphFormat) {
      if (!root.contains("object_graph") || !root.at("object_graph").is_object()) {
        setError(error, "manual scene graph does not contain object_graph");
        return false;
      }
      object_graph_root = &root.at("object_graph");
    } else if (!format.empty() && format != kRoomieObjectGraphFormat) {
      setError(error, "unsupported DSG JSON format: " + format);
      return false;
    }

    const std::string object_graph_format =
        object_graph_root->value("format", std::string());
    if (!object_graph_format.empty() && object_graph_format != kRoomieObjectGraphFormat) {
      setError(error, "unsupported embedded object graph format: " + object_graph_format);
      return false;
    }
    const int version = object_graph_root->value("format_version", 0);
    if (version > kRoomieObjectGraphFormatVersion) {
      setError(error, "unsupported DSG JSON version: " + std::to_string(version));
      return false;
    }
    *snapshot = snapshotFromJson(*object_graph_root);
    if (world_frame != nullptr) {
      *world_frame = object_graph_root->value(
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
