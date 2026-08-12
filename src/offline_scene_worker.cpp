#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "roomie/scene/furniture_graph.hpp"
#include "roomie/scene/scene_reducer.hpp"
#include "roomie/scene/scene_store.hpp"

namespace roomie {
namespace {

using Json = nlohmann::json;

Json vector3Json(const Eigen::Vector3f& value) {
  return Json::array({value.x(), value.y(), value.z()});
}

Json entityJson(const SceneEntityRef& entity) {
  const char* type = "object";
  if (entity.type == SceneEntityType::kRoom) {
    type = "room";
  } else if (entity.type == SceneEntityType::kFurniture) {
    type = "furniture";
  }
  return Json{{"type", type}, {"id", entity.id}};
}

Json snapshotRefJson(const ObjectSnapshotRef& ref) {
  return Json{{"image_index", ref.image_index},
              {"source_frame_asset_id", ref.source_frame_asset_id},
              {"evidence_hash", ref.evidence_hash},
              {"bbox_xyxy", ref.bbox_xyxy},
              {"crop_xywh", ref.crop_xywh},
              {"crop_output_scale", ref.crop_output_scale},
              {"mask_source", ref.mask_source},
              {"mask_ref", ref.mask_ref},
              {"quality", ref.quality},
              {"time_ns", ref.time_ns},
              {"camera_id", ref.camera_id}};
}

std::string effectiveLabel(const SceneObject& object) {
  if (object.annotation && object.annotation->label_override &&
      !object.annotation->label_override->empty()) {
    return *object.annotation->label_override;
  }
  return object.semantic ? object.semantic->label : std::string{};
}

std::string effectiveDescription(const SceneObject& object) {
  if (object.annotation && object.annotation->description_override) {
    return *object.annotation->description_override;
  }
  return object.artifact ? object.artifact->description : std::string{};
}

Json graphJson(const SceneSnapshot& snapshot,
               const std::optional<MapCheckpointManifest>& checkpoint) {
  Json root{{"format", "roomie_offline_scene"},
            {"format_version", 1},
            {"scene_revision", snapshot.revision()},
            {"durable_scene_revision", snapshot.durableRevision()},
            {"next_object_id", snapshot.nextObjectId()},
            {"objects", Json::array()},
            {"rooms", Json::array()},
            {"furniture", Json::array()},
            {"relations", Json::array()}};
  if (checkpoint) {
    root["map"] = Json{{"backend", checkpoint->backend},
                       {"checkpoint_path", checkpoint->checkpoint_path},
                       {"world_frame", checkpoint->world_frame},
                       {"map_revision", checkpoint->map_revision},
                       {"aligned_scene_revision",
                        checkpoint->aligned_scene_revision}};
  } else {
    root["map"] = nullptr;
  }

  std::set<int> furniture_ids;
  for (const FurnitureRole& role : snapshot.graphMetadata().furniture) {
    furniture_ids.insert(role.object_id);
    root["furniture"].push_back(
        Json{{"object_id", role.object_id},
             {"classification_label", role.classification_label},
             {"revision", role.revision}});
  }
  for (const auto& [object_id, object] : snapshot.objects()) {
    if (!object || !object->identity || !object->lifecycle ||
        !object->geometry || !object->semantic || !object->annotation ||
        !object->artifact) {
      continue;
    }
    Json snapshots = Json::array();
    for (const ObjectSnapshotRef& ref : object->artifact->snapshots) {
      snapshots.push_back(snapshotRefJson(ref));
    }
    Json value{
        {"object_id", object_id},
        {"name", object->annotation->name},
        {"label", effectiveLabel(*object)},
        {"detector_label", object->semantic->label},
        {"label_override",
         object->annotation->label_override
             ? Json(*object->annotation->label_override)
             : Json(nullptr)},
        {"description", effectiveDescription(*object)},
        {"semantic_id",
         object->annotation->semantic_id_override.value_or(
             object->semantic->semantic_id)},
        {"center_world", vector3Json(object->geometry->center_world)},
        {"size_m", vector3Json(object->geometry->size_m)},
        {"yaw_rad", object->geometry->yaw_rad},
        {"confidence", object->semantic->confidence},
        {"object_quality_score", object->semantic->object_quality_score},
        {"active", object->lifecycle->active},
        {"publishable", object->lifecycle->publishable},
        {"first_seen_ns", object->lifecycle->first_seen_ns},
        {"last_seen_ns", object->lifecycle->last_seen_ns},
        {"source_track_ids", object->identity->source_track_ids},
        {"snapshots", std::move(snapshots)},
        {"snapshot_set_hash", object->artifact->snapshot_set_hash},
        {"is_furniture", furniture_ids.count(object_id) != 0U},
        {"revisions",
         Json{{"identity", object->identity->revision},
              {"lifecycle", object->lifecycle->revision},
              {"geometry", object->geometry->revision},
              {"semantic", object->semantic->revision},
              {"annotation", object->annotation->revision},
              {"artifact", object->artifact->revision}}}};
    root["objects"].push_back(std::move(value));
  }
  for (const RoomNode& room : snapshot.graphMetadata().rooms) {
    root["rooms"].push_back(
        Json{{"room_id", room.room_id},
             {"revision", room.revision},
             {"label", room.label},
             {"color", room.color},
             {"center_world", vector3Json(room.center_world)},
             {"size_m", vector3Json(room.size_m)},
             {"min_xy", room.min_xy},
             {"max_xy", room.max_xy},
             {"has_xy_bounds", room.has_xy_bounds},
             {"height_m", room.height_m}});
  }
  for (const SceneRelation& relation : snapshot.graphMetadata().relations) {
    root["relations"].push_back(
        Json{{"source", entityJson(relationSource(relation))},
             {"target", entityJson(relationTarget(relation))},
             {"relation_type", relation.relation_type},
             {"confidence", relation.confidence},
             {"description", relation.description},
             {"revision", relation.revision},
             {"derived", relation.derived}});
  }
  return root;
}

LoadSceneCommand loadCommand(const SceneSnapshot& snapshot) {
  LoadSceneCommand load;
  load.restored_state = snapshot.statePtr();
  load.restored_revision = snapshot.revision();
  load.durable_revision = snapshot.durableRevision();
  load.latest_surface = snapshot.latestSurface();
  return load;
}

class OfflineSceneWorker {
 public:
  OfflineSceneWorker(std::string database_path,
                     FurnitureGraphConfig furniture_config)
      : store_(std::move(database_path)),
        furniture_config_(std::move(furniture_config)) {
    const SceneStoreStatus opened = store_.open();
    if (!opened) {
      throw std::runtime_error("failed to open SceneStore: " + opened.error);
    }
    const SceneRestoreResult restored = store_.restoreLatest();
    if (!restored.status) {
      throw std::runtime_error("failed to restore SceneStore: " +
                               restored.status.error);
    }
    if (!restored.found) {
      throw std::runtime_error("SceneStore contains no durable scene");
    }
    reducer_ = std::make_unique<ReducerCore>(furniture_config_);
    const SceneApplyResult loaded =
        reducer_->apply(SceneCommand{loadCommand(restored.snapshot)});
    if (!loaded.accepted()) {
      throw std::runtime_error("failed to initialize reducer: " +
                               loaded.reason);
    }
    const MapCheckpointLookupResult lookup = store_.latestMapCheckpoint();
    if (!lookup.status) {
      throw std::runtime_error("failed to read map checkpoint: " +
                               lookup.status.error);
    }
    checkpoint_ = lookup.manifest;
  }

  Json dispatch(const Json& request) {
    if (!request.is_object()) {
      throw std::invalid_argument("request must be an object");
    }
    const std::string operation = request.value("op", std::string());
    if (operation == "snapshot") {
      return response();
    }
    if (operation == "apply_batch") {
      return applyBatch(request);
    }
    if (operation == "shutdown") {
      shutdown_requested_ = true;
      return Json{{"ok", true}};
    }
    throw std::invalid_argument("unsupported operation: " + operation);
  }

  bool shutdownRequested() const { return shutdown_requested_; }

 private:
  Json response() const {
    return Json{{"ok", true},
                {"graph", graphJson(reducer_->snapshot(), checkpoint_)}};
  }

  Json applyBatch(const Json& request) {
    if (fatal_persistence_error_) {
      throw std::runtime_error(
          "worker is read-only after a previous persistence failure");
    }
    const SceneSnapshot before = reducer_->snapshot();
    const SceneRevision base = request.at("base_scene_revision")
                                   .get<SceneRevision>();
    if (base != before.revision()) {
      throw std::runtime_error("scene dependency is stale");
    }
    const Json edits_json = request.value("edits", Json::array());
    if (!edits_json.is_array()) {
      throw std::invalid_argument("edits must be an array");
    }
    std::vector<Json> edits(edits_json.begin(), edits_json.end());
    std::sort(edits.begin(), edits.end(), [](const Json& lhs, const Json& rhs) {
      return lhs.at("object_id").get<int>() <
             rhs.at("object_id").get<int>();
    });
    std::set<int> object_ids;
    auto candidate = std::make_unique<ReducerCore>(furniture_config_);
    const SceneApplyResult loaded =
        candidate->apply(SceneCommand{loadCommand(before)});
    if (!loaded.accepted()) {
      throw std::runtime_error("failed to clone current scene: " +
                               loaded.reason);
    }
    std::vector<SceneSnapshot> commits;
    for (const Json& edit : edits) {
      if (!edit.is_object()) {
        throw std::invalid_argument("each edit must be an object");
      }
      const int object_id = edit.at("object_id").get<int>();
      if (object_id < 0 || !object_ids.insert(object_id).second) {
        throw std::invalid_argument("edit object ids must be unique and valid");
      }
      const bool erase = edit.value("delete", false);
      if (erase && (edit.contains("name") || edit.contains("label"))) {
        throw std::invalid_argument(
            "delete cannot be combined with name or label edits");
      }
      SceneApplyResult applied;
      if (erase) {
        DeleteObjectCommand command;
        command.expected_scene_revision = candidate->snapshot().revision();
        command.object_id = object_id;
        command.expected_identity_revision =
            edit.value("identity_revision", std::uint64_t{0});
        command.reason = "offline scene editor";
        applied = candidate->apply(SceneCommand{std::move(command)});
      } else {
        ApplyHumanAnnotationCommand command;
        command.expected_scene_revision = candidate->snapshot().revision();
        command.target = SceneEntityRef{SceneEntityType::kObject, object_id};
        command.object_id = object_id;
        command.expected_identity_revision =
            edit.value("identity_revision", std::uint64_t{0});
        command.expected_annotation_revision =
            edit.value("annotation_revision", std::uint64_t{0});
        if (edit.contains("name")) {
          command.patch.name = edit.at("name").get<std::string>();
        }
        if (edit.contains("label")) {
          const std::string label = edit.at("label").get<std::string>();
          if (label.empty()) {
            throw std::invalid_argument("label must not be empty");
          }
          command.patch.label = label;
        }
        applied = candidate->apply(SceneCommand{std::move(command)});
      }
      if (!applied.accepted()) {
        throw std::runtime_error(applied.reason);
      }
      if (applied.committedRevision()) {
        commits.push_back(applied.snapshot);
      }
    }
    if (request.value("rebuild_furniture", false)) {
      const SceneApplyResult rebuilt = candidate->apply(
          SceneCommand{RebuildFurnitureGraphCommand{}});
      if (!rebuilt.accepted()) {
        throw std::runtime_error(rebuilt.reason);
      }
      if (rebuilt.committedRevision()) {
        commits.push_back(rebuilt.snapshot);
      }
    }
    if (commits.empty()) {
      reducer_ = std::move(candidate);
      return response();
    }
    for (const SceneSnapshot& snapshot : commits) {
      const SceneStoreStatus queued = store_.enqueueCommit(snapshot);
      if (!queued) {
        fatal_persistence_error_ = true;
        throw std::runtime_error("failed to queue scene edit: " +
                                 queued.error);
      }
    }
    const SceneStoreStatus flushed = checkpoint_
                                         ? store_.flushWithMapCheckpointAlignment()
                                         : store_.flush();
    if (!flushed) {
      fatal_persistence_error_ = true;
      throw std::runtime_error("failed to persist scene edit: " +
                               flushed.error);
    }
    const SceneRevision durable = commits.back().revision();
    const SceneApplyResult acknowledged = candidate->apply(
        SceneCommand{PersistedThroughCommand{durable}});
    if (!acknowledged.accepted()) {
      fatal_persistence_error_ = true;
      throw std::runtime_error("failed to acknowledge durable edit: " +
                               acknowledged.reason);
    }
    reducer_ = std::move(candidate);
    const MapCheckpointLookupResult lookup = store_.latestMapCheckpoint();
    if (!lookup.status || (checkpoint_ && !lookup.manifest)) {
      fatal_persistence_error_ = true;
      throw std::runtime_error(
          "failed to refresh map checkpoint state after save");
    }
    checkpoint_ = lookup.manifest;
    return response();
  }

  SceneStore store_;
  FurnitureGraphConfig furniture_config_;
  std::unique_ptr<ReducerCore> reducer_;
  std::optional<MapCheckpointManifest> checkpoint_;
  bool shutdown_requested_ = false;
  bool fatal_persistence_error_ = false;
};

struct Arguments {
  std::string database_path;
  std::string furniture_config_path;
};

Arguments parseArguments(int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--database" && index + 1 < argc) {
      result.database_path = argv[++index];
    } else if (argument == "--furniture-config" && index + 1 < argc) {
      result.furniture_config_path = argv[++index];
    } else {
      throw std::invalid_argument("unsupported worker argument: " + argument);
    }
  }
  if (result.database_path.empty()) {
    throw std::invalid_argument("--database is required");
  }
  return result;
}

}  // namespace
}  // namespace roomie

int main(int argc, char** argv) {
  using namespace roomie;
  try {
    const Arguments arguments = parseArguments(argc, argv);
    FurnitureGraphConfig furniture_config = defaultFurnitureGraphConfig();
    if (!arguments.furniture_config_path.empty()) {
      furniture_config =
          loadFurnitureGraphConfig(arguments.furniture_config_path);
    }
    OfflineSceneWorker worker(arguments.database_path,
                              std::move(furniture_config));
    std::string line;
    while (!worker.shutdownRequested() && std::getline(std::cin, line)) {
      if (line.empty()) {
        continue;
      }
      Json response;
      try {
        response = worker.dispatch(Json::parse(line));
      } catch (const std::exception& error) {
        response = Json{{"ok", false}, {"error", error.what()}};
      }
      std::cout << response.dump() << '\n' << std::flush;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "roomie_offline_scene_worker: " << error.what() << '\n';
    return 2;
  }
}
