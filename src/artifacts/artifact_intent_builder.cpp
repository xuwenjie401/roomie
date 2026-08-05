#include "roomie/artifacts/artifact_intent_builder.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "roomie/artifacts/artifact_slo_clock.hpp"

namespace roomie {
namespace {

using Json = nlohmann::json;

void appendKeyField(std::ostringstream* output,
                    const std::string& name,
                    const std::string& value) {
  *output << '|' << name.size() << ':' << name << '=' << value.size() << ':'
          << value;
}

void appendKeyField(std::ostringstream* output,
                    const std::string& name,
                    std::uint64_t value) {
  appendKeyField(output, name, std::to_string(value));
}

std::string damMemoKey(const SnapshotSetCommitted& event) {
  std::ostringstream output;
  output << "dam-event.v1";
  appendKeyField(&output, "revision", event.revision);
  appendKeyField(&output,
                 "object_id",
                 static_cast<std::uint64_t>(event.object_id));
  appendKeyField(&output, "appearance_revision", event.appearance_revision);
  appendKeyField(&output, "snapshot_set_hash", event.snapshot_set_hash);
  return output.str();
}

std::string embeddingMemoKey(const DescriptionCommitted& event) {
  std::ostringstream output;
  output << "embedding-event.v1";
  appendKeyField(&output, "revision", event.revision);
  appendKeyField(&output,
                 "object_id",
                 static_cast<std::uint64_t>(event.object_id));
  appendKeyField(&output, "artifact_revision", event.artifact_revision);
  appendKeyField(&output, "input_hash", event.input_hash);
  return output.str();
}

std::string embeddingMemoKey(const HumanAnnotationCommitted& event) {
  std::ostringstream output;
  output << "annotation-embedding-event.v1";
  appendKeyField(&output, "revision", event.revision);
  appendKeyField(&output,
                 "object_id",
                 static_cast<std::uint64_t>(event.object_id));
  appendKeyField(&output, "annotation_revision", event.annotation_revision);
  return output.str();
}

Json dependencyJson(const ObjectDependency& dependency) {
  return Json{{"object_id", dependency.object_id},
              {"identity_revision", dependency.identity_revision},
              {"obb_revision", dependency.obb_revision},
              {"appearance_revision", dependency.appearance_revision},
              {"semantic_revision", dependency.semantic_revision}};
}

Json imageMetadataJson(const ObjectSnapshotImage& image) {
  return Json{{"image_index", image.image_index},
              {"uri", image.uri},
              {"width", image.width},
              {"height", image.height},
              {"encoding", image.encoding},
              {"time_ns", image.time_ns},
              {"camera_id", image.camera_id},
              {"source_path", image.source_path}};
}

Json snapshotJson(const ObjectSnapshotRef& snapshot,
                  const SceneGraphMetadata& graph) {
  Json value = {{"image_index", snapshot.image_index},
                {"source_frame_asset_id", snapshot.source_frame_asset_id},
                {"evidence_hash", snapshot.evidence_hash},
                {"bbox_xyxy",
                 Json::array({snapshot.bbox_xyxy[0], snapshot.bbox_xyxy[1],
                              snapshot.bbox_xyxy[2], snapshot.bbox_xyxy[3]})},
                {"crop_xywh", snapshot.crop_xywh},
                {"crop_output_scale", snapshot.crop_output_scale},
                {"mask_source", snapshot.mask_source},
                {"mask_ref", snapshot.mask_ref},
                {"quality", snapshot.quality},
                {"quality_components", snapshot.quality_components},
                {"viewpoint",
                 Json{{"azimuth_rad", snapshot.viewpoint_azimuth_rad},
                      {"elevation_rad", snapshot.viewpoint_elevation_rad},
                      {"scale", snapshot.viewpoint_scale}}},
                {"time_ns", snapshot.time_ns},
                {"camera_id", snapshot.camera_id},
                {"provenance",
                 Json{{"image_index", snapshot.image_index},
                      {"run_id",
                       Json{{"high", snapshot.provenance.run_id.high},
                            {"low", snapshot.provenance.run_id.low}}},
                      {"frame_id", snapshot.provenance.frame_id},
                      {"request_id", snapshot.provenance.request_id},
                      {"sensor_time_ns", snapshot.provenance.sensor_time_ns},
                      {"map_revision",
                       snapshot.provenance.map.map_revision},
                      {"surface_revision",
                       snapshot.provenance.surface.surface_revision},
                      {"source_map_revision",
                       snapshot.provenance.surface.source_map_revision},
                      {"includes_current_frame",
                       snapshot.provenance.includes_current_frame},
                      {"causality_verified",
                       snapshot.provenance.causality_verified}}}};
  const auto image = std::find_if(
      graph.snapshot_images.begin(),
      graph.snapshot_images.end(),
      [&snapshot](const ObjectSnapshotImage& candidate) {
        return candidate.image_index == snapshot.image_index;
      });
  if (image != graph.snapshot_images.end()) {
    value["image"] = imageMetadataJson(*image);
  }
  return value;
}

std::string effectiveLabel(const SceneObject& object) {
  if (object.annotation && object.annotation->label_override &&
      !object.annotation->label_override->empty()) {
    return *object.annotation->label_override;
  }
  return object.semantic ? object.semantic->label : std::string{};
}

void validateConfig(const ArtifactIntentBuilderConfig& config) {
  if (config.dam_model_id.empty() || config.dam_prompt_hash.empty() ||
      config.dam_output_schema_version.empty()) {
    throw std::invalid_argument("DAM intent configuration is incomplete");
  }
  if (config.embedding_namespace.model_id.empty() ||
      config.embedding_namespace.dimension == 0 ||
      config.embedding_task_type.empty()) {
    throw std::invalid_argument(
        "embedding intent configuration is incomplete");
  }
  if (config.new_object_window_ms <= 0 ||
      config.new_object_interactive_limit == 0 ||
      config.replay_memo_capacity == 0 ||
      config.new_object_window_ms >
          std::numeric_limits<std::int64_t>::max() / 1'000'000) {
    throw std::invalid_argument("new-object burst policy is invalid");
  }
}

void validateCommit(const SceneApplyResult& commit) {
  if (!commit.committedRevision()) {
    return;
  }
  if (commit.revision == 0 || commit.snapshot.revision() != commit.revision) {
    throw std::invalid_argument(
        "artifact intents require one coherent committed scene revision");
  }
}

}  // namespace

std::string canonicalEmbeddingTaskKey(
    SceneObjectId object_id,
    const std::string& document_hash,
    const EmbeddingNamespace& name_space) {
  if (object_id < 0 || document_hash.empty() || name_space.model_id.empty() ||
      name_space.dimension == 0) {
    throw std::invalid_argument("embedding task key fields are invalid");
  }
  std::ostringstream output;
  output << "embedding-task.v1";
  appendKeyField(&output,
                 "object_id",
                 static_cast<std::uint64_t>(object_id));
  appendKeyField(&output, "document_hash", document_hash);
  appendKeyField(&output, "model_id", name_space.model_id);
  appendKeyField(&output,
                 "dimension",
                 static_cast<std::uint64_t>(name_space.dimension));
  return output.str();
}

ArtifactIntentBuilder::ArtifactIntentBuilder(ArtifactIntentBuilderConfig config)
    : config_(std::move(config)),
      artifact_scheduler_(nullptr, config_.dam_scheduler),
      embedding_scheduler_(nullptr) {
  validateConfig(config_);
}

ArtifactPriority ArtifactIntentBuilder::classifySnapshotSet(
    const SnapshotSetCommitted& event,
    std::int64_t created_steady_ns) {
  if (event.appearance_revision != 1) {
    return ArtifactPriority::kInteractive;
  }
  const std::int64_t window_ns =
      config_.new_object_window_ms * 1'000'000;
  const std::int64_t lower_bound =
      created_steady_ns <=
              std::numeric_limits<std::int64_t>::min() + window_ns
          ? std::numeric_limits<std::int64_t>::min()
          : created_steady_ns - window_ns;
  new_object_arrivals_steady_ns_.erase(
      std::remove_if(new_object_arrivals_steady_ns_.begin(),
                     new_object_arrivals_steady_ns_.end(),
                     [lower_bound](std::int64_t arrival) {
                       return arrival < lower_bound;
                     }),
      new_object_arrivals_steady_ns_.end());
  const std::size_t recent = static_cast<std::size_t>(std::count_if(
      new_object_arrivals_steady_ns_.begin(),
      new_object_arrivals_steady_ns_.end(),
      [lower_bound, created_steady_ns](std::int64_t arrival) {
        return arrival >= lower_bound && arrival <= created_steady_ns;
      }));
  new_object_arrivals_steady_ns_.push_back(created_steady_ns);
  return recent + 1 > config_.new_object_interactive_limit
             ? ArtifactPriority::kBulk
             : ArtifactPriority::kInteractive;
}

void ArtifactIntentBuilder::memoizeTask(std::string key,
                                        DurableTaskSpec task) {
  const auto inserted = memoized_tasks_.emplace(key, std::move(task));
  if (!inserted.second) {
    return;
  }
  memoized_task_order_.push_back(std::move(key));
  while (memoized_tasks_.size() > config_.replay_memo_capacity) {
    memoized_tasks_.erase(memoized_task_order_.front());
    memoized_task_order_.pop_front();
  }
}

std::size_t ArtifactIntentBuilder::memoizedTaskCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return memoized_tasks_.size();
}

std::size_t ArtifactIntentBuilder::objectOriginCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return object_origins_.size();
}

void ArtifactIntentBuilder::restoreObjectOrigins(
    const std::vector<DurableArtifactOrigin>& origins) {
  std::lock_guard<std::mutex> lock(mutex_);
  object_origins_.clear();
  for (const DurableArtifactOrigin& origin : origins) {
    if (origin.object_id < 0 || origin.scene_revision == 0 ||
        origin.created_unix_ms < 0 ||
        (origin.priority != ArtifactPriority::kInteractive &&
         origin.priority != ArtifactPriority::kBulk)) {
      throw std::invalid_argument("durable artifact origin is invalid");
    }
    const auto inserted = object_origins_.emplace(
        origin.object_id,
        ArtifactOrigin{origin.created_unix_ms, 0, origin.priority});
    if (!inserted.second &&
        (inserted.first->second.created_unix_ms != origin.created_unix_ms ||
         inserted.first->second.priority != origin.priority)) {
      throw std::invalid_argument("durable artifact origin conflicts");
    }
  }
}

void ArtifactIntentBuilder::setDurableTaskLookup(DurableTaskLookup lookup) {
  std::lock_guard<std::mutex> lock(mutex_);
  durable_task_lookup_ = std::move(lookup);
}

std::optional<DurableTaskSpec> ArtifactIntentBuilder::lookupDurableTask(
    const std::string& task_id) const {
  if (!durable_task_lookup_) {
    return std::nullopt;
  }
  const TaskLookupResult lookup = durable_task_lookup_(task_id);
  if (!lookup.status) {
    throw std::runtime_error("durable artifact task lookup failed: " +
                             lookup.status.error);
  }
  if (!lookup.task) {
    return std::nullopt;
  }
  return lookup.task->task;
}

void ArtifactIntentBuilder::rememberObjectOrigin(
    SceneObjectId object_id,
    ArtifactOrigin origin) {
  if (object_id < 0 || object_origins_.count(object_id) != 0) {
    return;
  }
  object_origins_.emplace(object_id, std::move(origin));
}

void ArtifactIntentBuilder::forgetObjectOrigin(SceneObjectId object_id) {
  object_origins_.erase(object_id);
}

DurableTaskSpec ArtifactIntentBuilder::buildDamTask(
    const SceneApplyResult& commit,
    const SnapshotSetCommitted& event,
    const ArtifactOrigin& origin) const {
  if (event.revision != commit.revision || event.object_id < 0 ||
      event.snapshot_set_hash.empty()) {
    throw std::runtime_error("snapshot event provenance is inconsistent");
  }
  const SceneObjectPtr object = commit.snapshot.findExactObject(event.object_id);
  if (!object || !object->identity || !object->artifact || !object->semantic ||
      object->artifact->appearance_revision != event.appearance_revision ||
      object->artifact->snapshot_set_hash != event.snapshot_set_hash) {
    throw std::runtime_error(
        "snapshot event does not match its owning scene snapshot");
  }

  DamTaskIntent intent;
  intent.key.object_id = event.object_id;
  intent.key.identity_revision = object->identity->revision;
  intent.key.appearance_revision = event.appearance_revision;
  intent.key.snapshot_set_hash = event.snapshot_set_hash;
  intent.key.model_id = config_.dam_model_id;
  intent.key.prompt_hash = config_.dam_prompt_hash;
  intent.key.output_schema_version = config_.dam_output_schema_version;
  intent.dependency = dependencyFor(*object);
  intent.scene_revision = commit.revision;
  intent.priority = origin.priority;
  intent.created_unix_ms = origin.created_unix_ms;
  if (origin.created_steady_ns > 0) {
    intent.steady_clock_epoch = artifactSteadyClockEpoch();
    intent.origin_steady_ns = origin.created_steady_ns;
  }

  Json snapshots = Json::array();
  for (const ObjectSnapshotRef& snapshot : object->artifact->snapshots) {
    if (!snapshot.valid()) {
      throw std::runtime_error("DAM intent contains an invalid snapshot ref");
    }
    snapshots.push_back(snapshotJson(snapshot, commit.snapshot.graphMetadata()));
  }
  const Json input = {
      {"payload_version", "roomie.dam-input.v1"},
      {"owning_scene_revision", commit.revision},
      {"object_id", event.object_id},
      {"label", effectiveLabel(*object)},
      {"dependency", dependencyJson(intent.dependency)},
      {"snapshot_set_hash", event.snapshot_set_hash},
      {"snapshots", std::move(snapshots)}};
  intent.input_payload = input.dump();

  SceneStoreStatus status;
  DurableTaskSpec spec = artifact_scheduler_.makeTaskSpec(intent, &status);
  if (!status) {
    throw std::runtime_error("cannot build DAM outbox intent: " + status.error);
  }
  return spec;
}

DurableTaskSpec ArtifactIntentBuilder::buildEmbeddingTask(
    const SceneApplyResult& commit,
    const DescriptionCommitted& event,
    std::int64_t created_unix_ms) const {
  if (event.revision != commit.revision || event.object_id < 0 ||
      event.input_hash.empty()) {
    throw std::runtime_error("description event provenance is inconsistent");
  }
  const SceneObjectPtr object = commit.snapshot.findExactObject(event.object_id);
  if (!object || !object->identity || !object->semantic || !object->artifact ||
      object->artifact->revision != event.artifact_revision ||
      object->artifact->pending_description_input_hash != event.input_hash ||
      object->artifact->pending_description_scene_revision !=
          commit.revision) {
    throw std::runtime_error(
        "description event does not match its owning scene snapshot");
  }

  const SemanticDocument document =
      makeSemanticDocumentForObject(
          *object, event.object_id, commit.revision,
          SemanticDocumentView::kPendingDescriptionIfPresent);
  EmbeddingTaskRequest request;
  request.owning_scene_revision = commit.revision;
  request.object_id = event.object_id;
  request.document = document.text;
  request.document_hash = document.document_hash;
  request.name_space = config_.embedding_namespace;
  request.semantic_revision = document.semantic_revision;
  request.created_scene_revision = document.created_scene_revision;
  request.identity_revision = object->identity->revision;
  request.artifact_revision = event.artifact_revision;
  request.description_input_hash = event.input_hash;
  request.artifact_slo = object->artifact->pending_description_slo;
  SceneStoreStatus status;
  DurableTaskSpec spec = embedding_scheduler_.makeTaskSpec(
      request, created_unix_ms, config_.embedding_task_type, &status);
  if (!status) {
    throw std::runtime_error("cannot build embedding outbox intent: " +
                             status.error);
  }
  return spec;
}

DurableTaskSpec ArtifactIntentBuilder::buildEmbeddingTask(
    const SceneApplyResult& commit,
    const HumanAnnotationCommitted& event,
    std::int64_t created_unix_ms) const {
  if (event.revision != commit.revision || event.object_id < 0 ||
      event.target.type != SceneEntityType::kObject ||
      event.target.id != event.object_id || !event.semantic_document_changed) {
    throw std::runtime_error(
        "annotation event provenance is inconsistent");
  }
  const SceneObjectPtr object = commit.snapshot.findExactObject(event.object_id);
  if (!object || !object->identity || !object->semantic || !object->artifact ||
      !object->annotation ||
      object->annotation->revision != event.annotation_revision) {
    throw std::runtime_error(
        "annotation event does not match its owning scene snapshot");
  }

  const SemanticDocument document = makeSemanticDocumentForObject(
      *object, event.object_id, commit.revision);
  const std::string description_input_hash =
      object->artifact->pending_description_scene_revision != 0
          ? object->artifact->pending_description_input_hash
          : object->artifact->description_input_hash;
  EmbeddingTaskRequest request;
  request.owning_scene_revision = commit.revision;
  request.object_id = event.object_id;
  request.document = document.text;
  request.document_hash = document.document_hash;
  request.name_space = config_.embedding_namespace;
  request.semantic_revision = document.semantic_revision;
  request.created_scene_revision = document.created_scene_revision;
  request.identity_revision = object->identity->revision;
  request.artifact_revision = object->artifact->revision;
  request.annotation_revision = event.annotation_revision;
  request.description_input_hash = description_input_hash;
  request.artifact_slo =
      object->artifact->pending_description_scene_revision != 0
          ? object->artifact->pending_description_slo
          : object->artifact->description_slo;
  SceneStoreStatus status;
  DurableTaskSpec spec = embedding_scheduler_.makeTaskSpec(
      request, created_unix_ms, config_.embedding_task_type, &status);
  if (!status) {
    throw std::runtime_error("cannot build annotation embedding intent: " +
                             status.error);
  }
  return spec;
}

ArtifactCommitIntents ArtifactIntentBuilder::build(
    const SceneApplyResult& commit,
    std::int64_t created_unix_ms,
    std::int64_t created_steady_ns) {
  validateCommit(commit);
  if (!commit.committedRevision()) {
    return {};
  }
  if (created_unix_ms < 0) {
    throw std::invalid_argument("artifact intent time cannot be negative");
  }
  if (created_steady_ns <= 0) {
    created_steady_ns = artifactSteadyNowNanoseconds();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  ArtifactCommitIntents intents;
  // New-object burst latency starts at stable object commit, not when the
  // asynchronous SnapshotBank happens to publish its first Top-K set.
  for (const SceneEvent& event : commit.events) {
    if (const auto* created = std::get_if<ObjectCreated>(&event)) {
      if (object_origins_.count(created->object_id) == 0) {
        SnapshotSetCommitted first_appearance;
        first_appearance.appearance_revision = 1;
        rememberObjectOrigin(
            created->object_id,
            ArtifactOrigin{created_unix_ms, created_steady_ns,
                           classifySnapshotSet(first_appearance,
                                               created_steady_ns)});
        const ArtifactOrigin& origin = object_origins_.at(created->object_id);
        intents.origins.push_back(DurableArtifactOrigin{
            created->object_id, commit.revision, origin.created_unix_ms,
            origin.priority});
      }
    } else if (const auto* merged = std::get_if<ObjectMerged>(&event)) {
      forgetObjectOrigin(merged->retired_object_id);
    } else if (const auto* tombstoned =
                   std::get_if<ObjectTombstoned>(&event)) {
      forgetObjectOrigin(tombstoned->object_id);
    }
  }
  std::set<std::string> emitted_task_ids;
  auto append_once = [&intents, &emitted_task_ids](
                         const DurableTaskSpec& task) {
    if (emitted_task_ids.insert(task.task_id).second) {
      intents.tasks.push_back(task);
    }
  };
  for (const SceneEvent& event : commit.events) {
    if (const auto* snapshot = std::get_if<SnapshotSetCommitted>(&event)) {
      const std::string memo_key = damMemoKey(*snapshot);
      const auto memoized = memoized_tasks_.find(memo_key);
      if (memoized != memoized_tasks_.end()) {
        append_once(memoized->second);
        continue;
      }
      ArtifactOrigin origin{created_unix_ms, created_steady_ns,
                            ArtifactPriority::kInteractive};
      bool consumed_stable_origin = false;
      if (snapshot->appearance_revision == 1) {
        const auto remembered = object_origins_.find(snapshot->object_id);
        if (remembered != object_origins_.end()) {
          origin = remembered->second;
          consumed_stable_origin = true;
        } else {
          origin.priority =
              classifySnapshotSet(*snapshot, created_steady_ns);
        }
      }
      DurableTaskSpec task;
      try {
        task = buildDamTask(commit, *snapshot, origin);
      } catch (...) {
        if (snapshot->appearance_revision == 1 &&
            !consumed_stable_origin &&
            !new_object_arrivals_steady_ns_.empty()) {
          new_object_arrivals_steady_ns_.pop_back();
        }
        throw;
      }
      if (const auto durable = lookupDurableTask(task.task_id)) {
        task = *durable;
      }
      memoizeTask(memo_key, task);
      append_once(task);
    } else if (const auto* description =
                   std::get_if<DescriptionCommitted>(&event)) {
      const std::string memo_key = embeddingMemoKey(*description);
      const auto memoized = memoized_tasks_.find(memo_key);
      if (memoized != memoized_tasks_.end()) {
        append_once(memoized->second);
        continue;
      }
      DurableTaskSpec task =
          buildEmbeddingTask(commit, *description, created_unix_ms);
      if (const auto durable = lookupDurableTask(task.task_id)) {
        task = *durable;
      }
      memoizeTask(memo_key, task);
      append_once(task);
    } else if (const auto* annotation =
                   std::get_if<HumanAnnotationCommitted>(&event)) {
      if (!annotation->semantic_document_changed ||
          annotation->target.type != SceneEntityType::kObject) {
        continue;
      }
      const std::string memo_key = embeddingMemoKey(*annotation);
      const auto memoized = memoized_tasks_.find(memo_key);
      if (memoized != memoized_tasks_.end()) {
        append_once(memoized->second);
        continue;
      }
      DurableTaskSpec task =
          buildEmbeddingTask(commit, *annotation, created_unix_ms);
      if (const auto durable = lookupDurableTask(task.task_id)) {
        task = *durable;
      }
      memoizeTask(memo_key, task);
      append_once(task);
    }
  }
  return intents;
}

}  // namespace roomie
