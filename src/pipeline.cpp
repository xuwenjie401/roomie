#include "roomie/pipeline/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "roomie/dsg/object_graph_io.hpp"
#include "roomie/artifacts/python_dam_worker.hpp"
#include "roomie/artifacts/python_embedding_encoder.hpp"
#include "roomie/artifacts/semantic_index_store_adapter.hpp"
#include "roomie/artifacts/semantic_scene_projector.hpp"
#include "roomie/artifacts/artifact_slo_clock.hpp"

namespace roomie {
namespace {

std::filesystem::path resolveRobotMaskConfigPath(
    const std::string& configured_path) {
  std::filesystem::path path(configured_path);
  if (path.is_absolute()) {
    return path;
  }
  return std::filesystem::path(
             ament_index_cpp::get_package_share_directory("roomie")) /
         "config" / "robots" / path;
}

GeometrySchedulerConfig geometrySchedulerConfig(
    const PipelineConfig& config) {
  GeometrySchedulerConfig result;
  result.pending_capacity =
      std::max<std::size_t>(64U, config.pending_frame_limit);
  result.thresholds.shell_thickness_m =
      config.instance_geometry_shell_thickness_m;
  result.thresholds.empty_inside_points =
      config.instance_geometry_empty_inside_points;
  result.thresholds.min_unique_voxels =
      config.instance_geometry_min_unique_voxels;
  result.thresholds.confirm_score = config.instance_geometry_confirm_score;
  result.thresholds.suppress_score = config.instance_geometry_suppress_score;
  return result;
}

PersistenceActorConfig persistenceActorConfig(
    const PipelineConfig& config) {
  PersistenceActorConfig result;
  result.flush_period =
      std::chrono::milliseconds(config.scene_store_flush_period_ms);
  result.flush_batch_size = config.scene_store_flush_batch_size;
  result.queue_capacity = config.scene_store_queue_size;
  result.max_undurable_revisions =
      config.scene_store_soft_lag_revisions;
  result.hard_max_undurable_revisions =
      config.scene_store_hard_lag_revisions;
  result.terminal_failure_timeout = std::chrono::milliseconds(
      config.scene_store_terminal_failure_timeout_ms);
  return result;
}

ArtifactIntentBuilderConfig artifactIntentBuilderConfig(
    const PipelineConfig& config) {
  ArtifactIntentBuilderConfig result;
  result.dam_model_id = config.dam_model_id;
  result.dam_prompt_hash = config.dam_prompt_hash;
  result.dam_output_schema_version = config.dam_output_schema_version;
  result.embedding_namespace =
      EmbeddingNamespace{config.embedding_model_id,
                         config.embedding_dimension};
  result.new_object_window_ms = config.artifact_new_object_window_ms;
  result.new_object_interactive_limit =
      config.artifact_new_object_interactive_limit;
  return result;
}

PythonDamWorkerConfig pythonDamWorkerConfig(const PipelineConfig& config) {
  PythonDamWorkerConfig result;
  result.python_executable = config.dam_python_executable;
  result.worker_script = config.dam_worker_script;
  result.dam_source = config.dam_source;
  result.model_path = config.dam_model_path;
  result.model_id = config.dam_model_id;
  result.prompt_hash = config.dam_prompt_hash;
  result.conversation_mode = config.dam_conversation_mode;
  result.prompt_mode = config.dam_prompt_mode;
  result.query = config.dam_query;
  result.max_new_tokens = config.dam_max_new_tokens;
  result.temperature = config.dam_temperature;
  result.top_p = config.dam_top_p;
  result.bbox_pad_px = config.dam_bbox_pad_px;
  result.startup_timeout =
      std::chrono::milliseconds(config.dam_startup_timeout_ms);
  result.request_timeout =
      std::chrono::milliseconds(config.dam_request_timeout_ms);
  result.shutdown_timeout =
      std::chrono::milliseconds(config.dam_shutdown_timeout_ms);
  return result;
}

PythonEmbeddingEncoderConfig pythonEmbeddingEncoderConfig(
    const PipelineConfig& config) {
  PythonEmbeddingEncoderConfig result;
  result.python_executable = config.embedding_python_executable;
  result.worker_script = config.embedding_worker_script;
  result.model_path = config.embedding_model_path;
  result.model_id = config.embedding_model_id;
  result.device = config.embedding_device;
  result.expected_dimension = config.embedding_dimension;
  result.maximum_batch_size = config.embedding_batch_size;
  result.startup_timeout =
      std::chrono::milliseconds(config.embedding_startup_timeout_ms);
  result.request_timeout =
      std::chrono::milliseconds(config.embedding_request_timeout_ms);
  result.shutdown_timeout =
      std::chrono::milliseconds(config.embedding_shutdown_timeout_ms);
  return result;
}

std::int64_t unixTimeMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

class PinnedAssetStoreResolver final : public PinnedSnapshotAssetResolver {
 public:
  PinnedAssetStoreResolver(
      std::shared_ptr<AssetStore> store,
      std::map<std::string, ResolvedSnapshotAsset> assets,
      std::vector<std::string> retained_ids)
      : store_(std::move(store)),
        assets_(std::move(assets)),
        retained_ids_(std::move(retained_ids)) {}

  ~PinnedAssetStoreResolver() override {
    if (!store_ || retained_ids_.empty()) {
      return;
    }
    try {
      std::map<AssetId, std::int64_t> releases;
      for (const std::string& id : retained_ids_) {
        releases[id] = -1;
      }
      std::string error;
      if (!store_->applyReferenceDelta(releases, 0, &error)) {
        RunLogger::logGlobal(
            "query_assets", "failed to release pinned assets: " + error);
      }
    } catch (const std::exception& error) {
      RunLogger::logGlobal(
          "query_assets",
          "exception while releasing pinned assets: " +
              std::string(error.what()));
    } catch (...) {
      RunLogger::logGlobal(
          "query_assets", "unknown exception while releasing pinned assets");
    }
  }

  std::optional<ResolvedSnapshotAsset> resolve(
      std::string_view source_frame_asset_id) const override {
    const auto asset = assets_.find(std::string(source_frame_asset_id));
    return asset == assets_.end()
               ? std::nullopt
               : std::optional<ResolvedSnapshotAsset>(asset->second);
  }

 private:
  std::shared_ptr<AssetStore> store_;
  const std::map<std::string, ResolvedSnapshotAsset> assets_;
  const std::vector<std::string> retained_ids_;
};

class AssetStoreQueryProvider final : public SnapshotAssetProvider {
 public:
  explicit AssetStoreQueryProvider(std::shared_ptr<AssetStore> store)
      : store_(std::move(store)) {}

  std::shared_ptr<const PinnedSnapshotAssetResolver> pin(
      const std::vector<std::string>& source_frame_asset_ids) const override {
    if (!store_) {
      return nullptr;
    }

    std::map<std::string, ResolvedSnapshotAsset> resolved_assets;
    std::map<AssetId, std::int64_t> retains;
    for (const std::string& id : source_frame_asset_ids) {
      if (id.empty() || resolved_assets.count(id) != 0U) {
        continue;
      }
      const std::optional<AssetRecord> record = store_->record(id);
      std::error_code physical_error;
      const bool physical_file_available =
          record && std::filesystem::is_regular_file(record->path,
                                                      physical_error);
      if (!physical_file_available || physical_error) {
        RunLogger::logGlobal(
            "query_assets",
            "cannot pin source_frame_asset_id=" + id + " error=" +
                (physical_error ? physical_error.message()
                                : "asset is absent"));
        continue;
      }

      std::error_code absolute_error;
      std::filesystem::path physical_path =
          std::filesystem::absolute(record->path, absolute_error);
      if (absolute_error) {
        physical_path = record->path;
      }
      ResolvedSnapshotAsset resolved;
      resolved.source_frame_asset_id = id;
      resolved.source_path = physical_path.string();
      resolved.uri = "file://" + physical_path.generic_string();
      resolved.width = record->width;
      resolved.height = record->height;
      resolved.channels = record->channels;
      resolved.encoding = "png";
      resolved.source_encoding = record->source_encoding;
      resolved.encoded_bytes = record->encoded_bytes;
      if (!resolved.valid()) {
        continue;
      }
      resolved_assets.emplace(id, std::move(resolved));
      retains[id] = 1;
    }

    if (!retains.empty()) {
      std::string error;
      if (!store_->applyReferenceDelta(retains, 0, &error)) {
        RunLogger::logGlobal(
            "query_assets", "failed to retain pinned assets: " + error);
        return nullptr;
      }
    }
    std::vector<std::string> retained_ids;
    retained_ids.reserve(retains.size());
    for (const auto& [id, count] : retains) {
      (void)count;
      retained_ids.push_back(id);
    }
    return std::make_shared<const PinnedAssetStoreResolver>(
        store_, std::move(resolved_assets), std::move(retained_ids));
  }

 private:
  std::shared_ptr<AssetStore> store_;
};

int stableSnapshotImageIndex(const SnapshotRecord& record,
                             std::size_t ordinal) {
  std::uint32_t hash = 2166136261U;
  const std::string& key = record.evidence_hash.empty()
                               ? record.source_frame_asset_id
                               : record.evidence_hash;
  for (const unsigned char byte : key) {
    hash ^= static_cast<std::uint32_t>(byte);
    hash *= 16777619U;
  }
  hash ^= static_cast<std::uint32_t>(ordinal);
  hash &= 0x7fffffffU;
  return static_cast<int>(hash == 0 ? 1U : hash);
}

ObjectSnapshotRef onlineSnapshotReference(const SnapshotRecord& record,
                                          std::size_t ordinal) {
  ObjectSnapshotRef reference;
  reference.image_index = stableSnapshotImageIndex(record, ordinal);
  reference.source_frame_asset_id = record.source_frame_asset_id;
  reference.evidence_hash = record.evidence_hash;
  reference.bbox_xyxy = record.bbox_xyxy;
  reference.crop_xywh = {record.crop.source_x, record.crop.source_y,
                         record.crop.source_width,
                         record.crop.source_height};
  reference.crop_output_scale = {record.crop.output_scale_x,
                                 record.crop.output_scale_y};
  reference.mask_source = snapshotMaskSourceName(record.mask_source);
  reference.mask_ref = record.mask_ref;
  reference.quality = record.quality_score;
  reference.quality_components = {
      {"confidence", record.quality.confidence},
      {"edge_completeness", record.quality.edge_completeness},
      {"distance", record.quality.distance},
      {"position", record.quality.position},
      {"size", record.quality.size},
      {"blur", record.quality.blur},
      {"exposure", record.quality.exposure},
      {"truncation", record.quality.truncation}};
  reference.viewpoint_azimuth_rad = record.viewpoint.azimuth_rad;
  reference.viewpoint_elevation_rad = record.viewpoint.elevation_rad;
  reference.viewpoint_scale = record.viewpoint.scale;
  reference.time_ns = record.time_ns;
  reference.camera_id = record.camera_id;
  reference.provenance = record.provenance;
  return reference;
}

float snapshotQualityComponent(const ObjectSnapshotRef& reference,
                               const char* name) {
  const auto component = reference.quality_components.find(name);
  return component == reference.quality_components.end() ? 1.0f
                                                          : component->second;
}

bool restoredSnapshotRecord(const ObjectSnapshotRef& reference,
                            SnapshotRecord* record,
                            std::string* error) {
  if (record == nullptr || reference.source_frame_asset_id.empty()) {
    if (error != nullptr) {
      *error = "online snapshot reference has no source frame asset id";
    }
    return false;
  }
  record->source_frame_asset_id = reference.source_frame_asset_id;
  record->evidence_hash = reference.evidence_hash;
  record->bbox_xyxy = reference.bbox_xyxy;
  record->crop.source_x = reference.crop_xywh[0];
  record->crop.source_y = reference.crop_xywh[1];
  record->crop.source_width = reference.crop_xywh[2];
  record->crop.source_height = reference.crop_xywh[3];
  record->crop.output_scale_x = reference.crop_output_scale[0];
  record->crop.output_scale_y = reference.crop_output_scale[1];
  if (reference.mask_source == "instance_mask") {
    record->mask_source = SnapshotMaskSource::kInstanceMask;
  } else if (reference.mask_source == "bbox_fallback" ||
             reference.mask_source.empty()) {
    record->mask_source = SnapshotMaskSource::kBboxFallback;
  } else {
    if (error != nullptr) {
      *error = "online snapshot reference has an unknown mask source: " +
               reference.mask_source;
    }
    return false;
  }
  record->mask_ref = reference.mask_ref;
  record->quality.confidence =
      snapshotQualityComponent(reference, "confidence");
  record->quality.edge_completeness =
      snapshotQualityComponent(reference, "edge_completeness");
  record->quality.distance = snapshotQualityComponent(reference, "distance");
  record->quality.position = snapshotQualityComponent(reference, "position");
  record->quality.size = snapshotQualityComponent(reference, "size");
  record->quality.blur = snapshotQualityComponent(reference, "blur");
  record->quality.exposure = snapshotQualityComponent(reference, "exposure");
  record->quality.truncation =
      snapshotQualityComponent(reference, "truncation");
  record->quality_score = reference.quality;
  record->viewpoint.azimuth_rad = reference.viewpoint_azimuth_rad;
  record->viewpoint.elevation_rad = reference.viewpoint_elevation_rad;
  record->viewpoint.scale = reference.viewpoint_scale;
  record->time_ns = reference.time_ns;
  record->camera_id = reference.camera_id;
  record->provenance = reference.provenance;
  return true;
}

bool restoreOnlineSnapshotState(
    const SceneSnapshot& scene,
    const std::shared_ptr<SnapshotBank>& bank,
    const std::shared_ptr<AssetStore>& store,
    std::string* error) {
  if (!bank || !store) {
    if (error != nullptr) {
      *error = "online snapshot restore requires bank and asset store";
    }
    return false;
  }
  std::map<AssetId, std::uint64_t> desired_references;
  for (const auto& [object_id, object] : scene.objects()) {
    if (!object || !object->artifact || object->artifact->snapshots.empty()) {
      continue;
    }
    const auto& references = object->artifact->snapshots;
    const bool has_online_reference = std::any_of(
        references.begin(), references.end(),
        [](const ObjectSnapshotRef& reference) {
          return !reference.source_frame_asset_id.empty();
        });
    if (!has_online_reference) {
      continue;
    }
    if (std::any_of(references.begin(), references.end(),
                    [](const ObjectSnapshotRef& reference) {
                      return reference.source_frame_asset_id.empty();
                    })) {
      if (error != nullptr) {
        *error = "object " + std::to_string(object_id) +
                 " mixes durable online and legacy snapshot references";
      }
      return false;
    }

    SnapshotSet snapshots;
    snapshots.snapshot_set_hash = object->artifact->snapshot_set_hash;
    snapshots.appearance_revision = object->artifact->appearance_revision;
    snapshots.records.reserve(references.size());
    for (const ObjectSnapshotRef& reference : references) {
      SnapshotRecord record;
      if (!restoredSnapshotRecord(reference, &record, error)) {
        return false;
      }
      ++desired_references[record.source_frame_asset_id];
      snapshots.records.push_back(std::move(record));
    }
    if (!bank->restoreObjectSnapshots(object_id, std::move(snapshots), error)) {
      return false;
    }
  }
  for (const auto& [retired_id, alias] : scene.aliases()) {
    if (!bank->restoreObjectAlias(retired_id, alias.canonical_object_id,
                                  error)) {
      return false;
    }
  }
  return store->reconcileReferenceCounts(desired_references, 0, error);
}

}  // namespace

void RoomiePipeline::scheduleMissingEmbeddingJobs(
    std::vector<EmbeddingJob> jobs,
    const SceneSnapshot& snapshot) {
  auto job_key = [](const EmbeddingJob& job) {
    return std::to_string(job.object_id) + "|" + job.name_space.model_id +
           "|" + std::to_string(job.name_space.dimension);
  };
  {
    std::lock_guard<std::mutex> lock(pending_embedding_jobs_mutex_);
    for (EmbeddingJob& job : jobs) {
      const std::string key = job_key(job);
      const auto existing = pending_embedding_jobs_.find(key);
      if (existing == pending_embedding_jobs_.end() ||
          existing->second.created_scene_revision <=
              job.created_scene_revision) {
        pending_embedding_jobs_[key] = std::move(job);
      }
    }
  }
  if (!embedding_task_scheduler_ || !snapshot.statePtr()) {
    return;
  }

  std::vector<std::pair<std::string, EmbeddingJob>> eligible;
  {
    std::lock_guard<std::mutex> lock(pending_embedding_jobs_mutex_);
    for (const auto& entry : pending_embedding_jobs_) {
      if (entry.second.created_scene_revision <=
          snapshot.durableRevision()) {
        eligible.push_back(entry);
      }
    }
  }
  for (const auto& [key, job] : eligible) {
    bool obsolete = false;
    const SceneObjectPtr object = snapshot.findExactObject(job.object_id);
    if (!object || snapshot.isTombstoned(job.object_id)) {
      obsolete = true;
    } else {
      try {
        obsolete = makeSemanticDocumentForObject(
                       *object, job.object_id,
                       job.created_scene_revision)
                       .document_hash != job.document_hash;
      } catch (...) {
        obsolete = true;
      }
    }
    SceneStoreStatus status = SceneStoreStatus::success();
    if (!obsolete) {
      status = embedding_task_scheduler_->ensureTask(
          job, snapshot, unixTimeMilliseconds());
    }
    if (!status) {
      RunLogger::logGlobal(
          "embedding_outbox",
          "missing embedding admission deferred object_id=" +
              std::to_string(job.object_id) + " scene_revision=" +
              std::to_string(job.created_scene_revision) + " error=" +
              status.error);
      continue;
    }
    std::lock_guard<std::mutex> lock(pending_embedding_jobs_mutex_);
    const auto current = pending_embedding_jobs_.find(key);
    if (current != pending_embedding_jobs_.end() &&
        current->second.document_hash == job.document_hash &&
        current->second.created_scene_revision ==
            job.created_scene_revision) {
      pending_embedding_jobs_.erase(current);
    }
  }
}

RoomiePipeline::RoomiePipeline(
    rclcpp::Node& node,
    PipelineConfig config,
    RoomiePipelineRuntimeDependencies runtime_dependencies)
    : config_(std::move(config)),
      run_logger_(std::make_shared<RunLogger>(config_)),
      mapping_queue_(config_.mapping_queue_size, ChannelPolicy::kDropOldest),
      detection_queue_(
          config_.detection_queue_size,
          ChannelPolicy::kLatestByKey,
          [](const FrameBundlePtr& lhs, const FrameBundlePtr& rhs) {
            return lhs && rhs && lhs->camera_id == rhs->camera_id;
          }),
      inference_response_queue_(config_.inference_response_queue_size,
                                ChannelPolicy::kReliableBlocking),
      ros_io_thread_(mapping_queue_, detection_queue_),
      map_thread_(mapping_queue_, config_),
      python_backend_(config_),
      instance_map_thread_(inference_response_queue_, map_thread_, config_),
      geometry_worker_thread_(
          geometrySchedulerConfig(config_),
          [this]() { return instance_map_thread_.sceneSnapshot(); },
          [this](SceneCommand command) {
            return instance_map_thread_.enqueueSceneCommand(
                std::move(command));
          }),
      detection_bridge_thread_(node,
                               detection_queue_,
                               inference_response_queue_,
                               map_thread_,
                               python_backend_,
                               config_),
      publisher_persistence_thread_(node, instance_map_thread_, map_thread_, config_) {
  runtime_dependencies_ = std::move(runtime_dependencies);
  if (config_.dam_artifacts_enabled) {
    const std::string expected_prompt_hash = config_.dam_prompt_hash;
    const std::string computed_prompt_hash =
        canonicalDamExecutionPromptHash(pythonDamWorkerConfig(config_));
    if (!expected_prompt_hash.empty() &&
        expected_prompt_hash != computed_prompt_hash) {
      throw std::invalid_argument(
          "artifacts.dam_prompt_hash does not match canonical DAM execution parameters");
    }
    config_.dam_prompt_hash = computed_prompt_hash;
    RunLogger::logGlobal(
        "artifact_runtime",
        "canonical DAM prompt_hash=" + computed_prompt_hash);
  }
  if (config_.embedding_artifacts_enabled) {
    SemanticIndexConfig semantic_config;
    semantic_config.initial_namespace =
        EmbeddingNamespace{config_.embedding_model_id,
                           config_.embedding_dimension};
    semantic_config.record_history_capacity =
        config_.embedding_record_history_capacity;
    semantic_index_ =
        std::make_shared<VersionedSemanticIndex>(std::move(semantic_config));
  }
  map_thread_.setCommitObserver(
      [this](const MapCommit& commit) {
        if (commit.success && commit.snapshot) {
          geometry_worker_thread_.onMapCommit(commit);
        }
        if (commit.perception_candidate) {
          detection_bridge_thread_.onMapCommit(commit);
        }
      });
  instance_map_thread_.setSceneCommitObserver(
      [this](const SceneApplyResult& result) {
        geometry_worker_thread_.onSceneCommit(result);
        if (semantic_index_) {
          SemanticSceneProjectionResult projected =
              projectSceneCommitToSemanticIndex(result,
                                                semantic_index_.get());
          if (!projected.ok) {
            RunLogger::logGlobal(
                "semantic_index",
                "scene projection failed revision=" +
                    std::to_string(result.revision) + " error=" +
                    projected.error);
          } else {
            scheduleMissingEmbeddingJobs(
                std::move(projected.missing_embeddings), result.snapshot);
          }
        }
      });
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

  bool restored_from_scene_store = false;
  bool snapshot_assets_reconciled = false;
  bool map_checkpoint_load_initialized = false;
  if (config_.dam_artifacts_enabled ||
      config_.embedding_artifacts_enabled) {
    if (!config_.scene_store_enabled) {
      throw std::invalid_argument(
          "durable DAM/embedding artifacts require persistence.scene_store_enabled");
    }
    artifact_intent_builder_ = std::make_unique<ArtifactIntentBuilder>(
        artifactIntentBuilderConfig(config_));
  }
  if (config_.online_snapshot_enabled) {
    if (config_.asset_store_root.empty()) {
      throw std::invalid_argument(
          "artifacts.asset_store_root must not be empty when snapshots are enabled");
    }
    AssetStoreConfig asset_config;
    asset_config.root = config_.asset_store_root;
    asset_config.grace_period_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(
                config_.asset_gc_grace_period_sec))
            .count();
    asset_store_ = std::make_shared<AssetStore>(std::move(asset_config));
    if (!asset_store_->healthy()) {
      throw std::runtime_error("failed to initialize AssetStore: " +
                               asset_store_->initializationError());
    }
    SnapshotBankConfig bank_config;
    bank_config.top_k = config_.snapshot_top_k;
    bank_config.minimum_quality = config_.snapshot_minimum_quality;
    bank_config.azimuth_bucket_degrees =
        config_.snapshot_azimuth_bucket_degrees;
    bank_config.elevation_bucket_degrees =
        config_.snapshot_elevation_bucket_degrees;
    bank_config.scale_bucket_ratio = config_.snapshot_scale_bucket_ratio;
    bank_config.replacement_min_quality_delta =
        config_.instance_snapshot_replace_min_quality_delta;
    bank_config.replacement_min_quality_ratio =
        config_.instance_snapshot_replace_min_quality_ratio;
    bank_config.diversity_min_quality_ratio =
        config_.snapshot_diversity_min_quality_ratio;
    snapshot_bank_ = std::make_shared<SnapshotBank>(
        std::move(bank_config), asset_store_);
    OnlineSnapshotWorkerConfig worker_config;
    worker_config.event_queue_capacity = config_.snapshot_queue_size;
    worker_config.asset_gc_period =
        std::chrono::milliseconds(config_.asset_gc_period_ms);
    online_snapshot_worker_ = std::make_unique<OnlineSnapshotWorker>(
        std::move(worker_config), snapshot_bank_,
        [this]() { return instance_map_thread_.sceneSnapshot(); },
        [this](const ApplySnapshotSetCommand& command) {
          const std::optional<SceneApplyResult> applied =
              instance_map_thread_.applySceneCommandAndWait(
                  SceneCommand{command});
          SnapshotCommandSubmitResult result;
          if (!applied) {
            result.reason = "scene reducer command timed out or stopped";
            return result;
          }
          result.reason = applied->reason;
          if (applied->accepted()) {
            result.disposition = SnapshotCommandDisposition::kAccepted;
          } else if (applied->reason.find("stale") != std::string::npos ||
                     applied->reason.find("alias") != std::string::npos) {
            result.disposition = SnapshotCommandDisposition::kStale;
          }
          return result;
        },
        onlineSnapshotReference);
    instance_map_thread_.setOnlineSnapshotWorker(
        online_snapshot_worker_.get());
  }
  if (config_.dam_artifacts_enabled) {
    if (!asset_store_) {
      throw std::invalid_argument(
          "artifacts.dam_enabled requires artifacts.snapshot_enabled and an AssetStore");
    }
    if (!runtime_dependencies_.dam_worker) {
      runtime_dependencies_.dam_worker =
          std::make_shared<PythonDamWorker>(
              pythonDamWorkerConfig(config_), asset_store_);
    }
  }
  if (config_.embedding_artifacts_enabled &&
      !runtime_dependencies_.embedding_encoder) {
    runtime_dependencies_.embedding_encoder =
        std::make_shared<PythonEmbeddingEncoder>(
            pythonEmbeddingEncoderConfig(config_));
  }
  if (config_.scene_store_enabled) {
    if (config_.scene_store_path.empty()) {
      throw std::invalid_argument(
          "persistence.scene_store_path must not be empty when enabled");
    }
    persistence_actor_ = std::make_unique<PersistenceActor>(
        config_.scene_store_path, persistenceActorConfig(config_));
    const SceneStoreStatus opened = persistence_actor_->open();
    if (!opened) {
      throw std::runtime_error(
          "failed to open Roomie SceneStore: " + opened.error);
    }

    SceneStore* const durable_store =
        persistence_actor_->durableArtifactStore();
    if (durable_store == nullptr) {
      throw std::runtime_error("opened persistence actor has no SceneStore");
    }
    SceneStoreWatermarks store_watermarks = durable_store->watermarks();
    const MapCheckpointLookupResult checkpoint_lookup =
        durable_store->latestMapCheckpoint();
    if (!checkpoint_lookup.status) {
      throw std::runtime_error(
          "failed to query map checkpoint manifest: " +
          checkpoint_lookup.status.error);
    }
    const bool coordinated_manifest_recovery =
        config_.map_load_mode == "coordinated" &&
        checkpoint_lookup.manifest.has_value();
    const bool coordinated_manifest_missing =
        config_.map_load_mode == "coordinated" &&
        store_watermarks.durable_scene_revision != 0 &&
        !checkpoint_lookup.manifest.has_value();
    const bool seed_reset_required =
        config_.map_load_mode == "seed" &&
        (store_watermarks.durable_scene_revision != 0 ||
         checkpoint_lookup.manifest.has_value());
    if (config_.load_map || coordinated_manifest_recovery ||
        coordinated_manifest_missing || seed_reset_required) {
      MapCheckpointOperationResult loaded_checkpoint;
      if (config_.map_load_mode == "seed") {
        // Seed mode is intentionally explicit and destructive with respect to
        // durable scene-derived state. It never mixes an arbitrary configured
        // map with objects/artifacts from a different map lineage.
        if (config_.load_map) {
          loaded_checkpoint = map_thread_.loadConfiguredCheckpoint();
          if (loaded_checkpoint.disposition ==
              MapCheckpointDisposition::kFailed) {
            throw std::runtime_error("failed to load configured seed map: " +
                                     loaded_checkpoint.error);
          }
          if (loaded_checkpoint.disposition ==
              MapCheckpointDisposition::kUnsupported) {
            RunLogger::logGlobal(
                "map_checkpoint",
                "configured seed map was not loaded: " +
                    loaded_checkpoint.error);
          }
        }
        if (store_watermarks.durable_scene_revision != 0 ||
            checkpoint_lookup.manifest) {
          const SceneStoreStatus rewound = persistence_actor_->rewindTo(
              0, /*retain_aligned_map_checkpoints=*/false);
          if (!rewound) {
            throw std::runtime_error(
                "failed to discard scene state for explicit seed mode: " +
                rewound.error);
          }
          store_watermarks = durable_store->watermarks();
        }
        if (loaded_checkpoint.succeeded()) {
          RunLogger::logGlobal(
              "map_checkpoint",
              "loaded explicit seed map path=" +
                  loaded_checkpoint.manifest.checkpoint_path +
                  " discarded_scene_suffix=true");
        } else if (seed_reset_required) {
          RunLogger::logGlobal(
              "map_checkpoint",
              "started explicit empty seed and discarded durable map/scene "
              "lineage");
        }
      } else if (checkpoint_lookup.manifest) {
        const SceneRestoreResult aligned_scene =
            persistence_actor_->restoreAt(
                checkpoint_lookup.manifest->aligned_scene_revision);
        if (!aligned_scene.status ||
            (checkpoint_lookup.manifest->aligned_scene_revision != 0 &&
             (!aligned_scene.found ||
              aligned_scene.snapshot.revision() !=
                  checkpoint_lookup.manifest->aligned_scene_revision))) {
          throw std::runtime_error(
              "map checkpoint's aligned scene prefix cannot be restored: " +
              aligned_scene.status.error);
        }
        loaded_checkpoint = map_thread_.loadConfiguredCheckpoint(
            &*checkpoint_lookup.manifest);
        if (!loaded_checkpoint.succeeded()) {
          throw std::runtime_error(
              "failed to load the durable map checkpoint: " +
              loaded_checkpoint.error);
        }
        const SceneRevision abandoned_suffix =
            store_watermarks.durable_scene_revision -
            checkpoint_lookup.manifest->aligned_scene_revision;
        if (abandoned_suffix != 0) {
          const SceneStoreStatus rewound = persistence_actor_->rewindTo(
              checkpoint_lookup.manifest->aligned_scene_revision);
          if (!rewound) {
            throw std::runtime_error(
                "failed to coordinate scene recovery with map checkpoint: " +
                rewound.error);
          }
          store_watermarks = durable_store->watermarks();
        }
        RunLogger::logGlobal(
            "map_checkpoint",
            "loaded durable checkpoint path=" +
                loaded_checkpoint.manifest.checkpoint_path +
                " aligned_scene_revision=" +
                std::to_string(
                    checkpoint_lookup.manifest->aligned_scene_revision) +
                " durable_scene_revision=" +
                std::to_string(store_watermarks.durable_scene_revision) +
                " abandoned_semantic_suffix_revisions=" +
                std::to_string(abandoned_suffix));
      } else {
        if (store_watermarks.durable_scene_revision != 0) {
          throw std::runtime_error(
              "coordinated map recovery requires a durable map checkpoint "
              "manifest when SceneStore is non-empty; set tsdf.map_load_mode="
              "seed to explicitly discard the durable scene");
        }
        loaded_checkpoint = map_thread_.loadConfiguredCheckpoint();
        if (loaded_checkpoint.disposition ==
            MapCheckpointDisposition::kFailed) {
          throw std::runtime_error(
              "failed to load configured seed map: " +
              loaded_checkpoint.error);
        }
        if (loaded_checkpoint.disposition ==
            MapCheckpointDisposition::kUnsupported) {
          RunLogger::logGlobal(
              "map_checkpoint",
              "configured seed map was not loaded: " +
                  loaded_checkpoint.error);
        } else if (loaded_checkpoint.succeeded()) {
          RunLogger::logGlobal(
              "map_checkpoint",
              "loaded initial unaligned map into empty SceneStore path=" +
                  loaded_checkpoint.manifest.checkpoint_path);
        }
      }
      map_checkpoint_load_initialized = true;
    }
    if (artifact_intent_builder_) {
      const ArtifactOriginListResult origins =
          durable_store->listArtifactOrigins();
      if (!origins.status) {
        throw std::runtime_error(
            "failed to restore durable artifact origins: " +
            origins.status.error);
      }
      artifact_intent_builder_->restoreObjectOrigins(origins.origins);
      artifact_intent_builder_->setDurableTaskLookup(
          [durable_store](const std::string& task_id) {
            return durable_store->lookupTask(task_id);
          });
    }
    instance_map_thread_.setSceneCommitSink(
        [this](const SceneApplyResult& result) {
          if (!persistence_actor_) {
            return true;
          }
          SceneSnapshot snapshot = result.snapshot;
          const SceneRevision stored = persistence_actor_->latestRevision();
          if (snapshot.revision() <= stored) {
            return snapshot.revision() == stored;
          }
          ArtifactCommitIntents artifact_intents;
          if (artifact_intent_builder_) {
            try {
              artifact_intents = artifact_intent_builder_->build(
                  result, unixTimeMilliseconds(),
                  artifactSteadyNowNanoseconds());
              artifact_intents.tasks.erase(
                  std::remove_if(
                      artifact_intents.tasks.begin(),
                      artifact_intents.tasks.end(),
                      [this](const DurableTaskSpec& task) {
                        const bool dam =
                            task.task_type ==
                                ArtifactScheduler::kInteractiveTaskType ||
                            task.task_type ==
                                ArtifactScheduler::kBulkTaskType;
                        if (dam) {
                          return !config_.dam_artifacts_enabled;
                        }
                        return !config_.embedding_artifacts_enabled;
                      }),
                  artifact_intents.tasks.end());
            } catch (const std::exception& error) {
              const std::string message =
                  "failed to build scene_revision=" +
                  std::to_string(snapshot.revision()) +
                  " outbox: " + error.what();
              RunLogger::logGlobal(
                  "artifact_outbox",
                  message);
              persistence_actor_->failAdmission(message);
              return false;
            }
          }
          return persistence_actor_->enqueueCommit(
                     std::move(snapshot),
                     std::move(artifact_intents.tasks),
                     std::move(artifact_intents.origins))
              .accepted();
        });
    instance_map_thread_.setSceneContentAdmission([this]() {
      return persistence_actor_ && persistence_actor_->admissionAllowed();
    });
    persistence_actor_->setDurabilityAckCallback(
        [this](SceneRevision revision) {
          return instance_map_thread_.requestDurabilityAck(revision);
        });

    const SceneRestoreResult restored = persistence_actor_->restoreLatest();
    if (!restored.status) {
      throw std::runtime_error(
          "failed to restore Roomie SceneStore: " + restored.status.error);
    }
    if (restored.found) {
      std::string error;
      if (!instance_map_thread_.loadSceneSnapshot(restored.snapshot, &error)) {
        throw std::runtime_error(
            "failed to bootstrap reducer from Roomie SceneStore: " + error);
      }
      if (snapshot_bank_ &&
          !restoreOnlineSnapshotState(restored.snapshot, snapshot_bank_,
                                      asset_store_, &error)) {
        throw std::runtime_error(
            "failed to restore online snapshot state: " + error);
      }
      snapshot_assets_reconciled = snapshot_bank_ != nullptr;
      restored_from_scene_store = true;
      RunLogger::logGlobal(
          "scene_persistence",
          "restored scene_revision=" +
              std::to_string(restored.snapshot.revision()) +
              " durable_scene_revision=" +
              std::to_string(restored.snapshot.durableRevision()) +
              " path=" + config_.scene_store_path);
    }
  }

  if (config_.load_map && !map_checkpoint_load_initialized) {
    const MapCheckpointOperationResult loaded_checkpoint =
        map_thread_.loadConfiguredCheckpoint();
    if (loaded_checkpoint.disposition ==
        MapCheckpointDisposition::kFailed) {
      throw std::runtime_error(
          "failed to load configured map checkpoint: " +
          loaded_checkpoint.error);
    }
    if (loaded_checkpoint.disposition ==
        MapCheckpointDisposition::kUnsupported) {
      RunLogger::logGlobal(
          "map_checkpoint",
          "configured map load is unsupported by the active backend: " +
              loaded_checkpoint.error);
    } else if (loaded_checkpoint.succeeded()) {
      RunLogger::logGlobal(
          "map_checkpoint",
          "loaded configured map without SceneStore alignment path=" +
              loaded_checkpoint.manifest.checkpoint_path);
    }
  }

  if (config_.load_scene_graph && !restored_from_scene_store) {
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
          if (snapshot_bank_ &&
              !restoreOnlineSnapshotState(
                  instance_map_thread_.sceneSnapshot(), snapshot_bank_,
                  asset_store_, &error)) {
            throw std::runtime_error(
                "failed to restore online snapshot state from imported scene graph: " +
                error);
          }
          snapshot_assets_reconciled = snapshot_bank_ != nullptr;
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

  // Reconcile only after both SQLite and compatibility JSON restore paths
  // have had a chance to reconstruct SnapshotBank. Reconciling an empty set
  // before importing a v3 graph can drop the only durable references to its
  // content-addressed frame assets and make them eligible for GC.
  if (snapshot_bank_ && !snapshot_assets_reconciled) {
    std::string error;
    if (!asset_store_->reconcileReferenceCounts({}, 0, &error)) {
      throw std::runtime_error(
          "failed to reconcile empty online snapshot state: " + error);
    }
  }

  RosCameraSubscriptionConfig mapping_camera;
  mapping_camera.camera_id = config_.mapping_camera_id;
  mapping_camera.camera_frame = config_.mapping_camera_frame;
  mapping_camera.rgb_topic = config_.rgb_topic;
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
  mapping_camera.enable_mapping = !config_.freeze_tsdf_map;
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
  ros_io_config.map_mode =
      config_.freeze_tsdf_map ? MapMode::kFrozen : MapMode::kOnline;
  ros_io_config.max_perception_fps = config_.max_inference_fps;
  ros_io_config.perception_deadline_ms = config_.perception_deadline_ms;
  ros_io_config.perception_admission_allowed = [this]() {
    return !persistence_actor_ || persistence_actor_->admissionAllowed();
  };
  ros_io_config.perception_candidate_cancelled =
      [this](const FrameBundlePtr& frame, const std::string& reason) {
        // Erase any commit that won the publication race before cancelling
        // queued/in-progress map work. Both operations are idempotent.
        detection_bridge_thread_.cancelPendingCandidate(frame, reason);
        map_thread_.cancelPerceptionCandidate(frame, reason);
      };
  RobotMaskGeneratorConfig robot_mask_config;
  robot_mask_config.robot_config =
      resolveRobotMaskConfigPath(config_.robot_mask_robot_config);
  robot_mask_config.camera_config =
      resolveRobotMaskConfigPath(config_.robot_mask_camera_config);
  robot_mask_config.reuse_translation_epsilon_m =
      config_.robot_mask_reuse_translation_epsilon_m;
  robot_mask_config.reuse_rotation_epsilon_rad =
      config_.robot_mask_reuse_rotation_epsilon_rad;
  ros_io_config.robot_mask_generator =
      std::make_shared<RobotMaskGenerator>(std::move(robot_mask_config));
  ros_io_config.cameras.push_back(std::move(mapping_camera));
  ros_io_thread_.configure(std::move(ros_io_config));
  ros_io_thread_.attachNode(node);

  if (asset_store_) {
    snapshot_asset_provider_ =
        std::make_shared<AssetStoreQueryProvider>(asset_store_);
  }
  if (semantic_index_) {
    const SceneSnapshot scene = instance_map_thread_.sceneSnapshot();
    const SemanticSceneProjectionResult initialized =
        initializeSemanticIndexFromScene(scene, semantic_index_.get());
    if (!initialized.ok) {
      throw std::runtime_error(
          "failed to initialize semantic index documents: " +
          initialized.error);
    }
    if (persistence_actor_) {
      SceneStore* store = persistence_actor_->durableArtifactStore();
      const SemanticEmbeddingRestoreResult restored =
          restoreSemanticIndexEmbeddingRecords(store,
                                               semantic_index_.get());
      if (!restored.status) {
        throw std::runtime_error(
            "failed to restore semantic embedding records: " +
            restored.status.error);
      }
      RunLogger::logGlobal(
          "semantic_index",
          "restored embeddings scanned=" +
              std::to_string(restored.records_scanned) + " accepted=" +
              std::to_string(restored.accepted));
      SemanticSceneProjectionResult missing =
          initializeSemanticIndexFromScene(scene, semantic_index_.get());
      if (!missing.ok) {
        throw std::runtime_error(
            "failed to audit missing semantic embeddings after restore: " +
            missing.error);
      }
      scheduleMissingEmbeddingJobs(
          std::move(missing.missing_embeddings), scene);
    }
    if (runtime_dependencies_.embedding_encoder) {
      if (runtime_dependencies_.embedding_encoder->modelId() !=
              config_.embedding_model_id ||
          runtime_dependencies_.embedding_encoder->dimension() !=
              config_.embedding_dimension) {
        throw std::invalid_argument(
            "embedding encoder namespace does not match artifacts.embedding_model_id/dimension");
      }
      if (!runtime_dependencies_.query_encoder) {
        const std::shared_ptr<EmbeddingEncoder> encoder =
            runtime_dependencies_.embedding_encoder;
        runtime_dependencies_.query_encoder =
            [encoder](std::string_view query) {
              std::vector<std::vector<float>> vectors =
                  encoder->encodeBatch({std::string(query)});
              if (vectors.size() != 1U) {
                throw std::runtime_error(
                    "embedding query encoder returned the wrong batch size");
              }
              return std::move(vectors.front());
            };
      }
    }
    if (runtime_dependencies_.query_encoder) {
      semantic_search_provider_ =
          std::make_shared<VersionedSemanticSearchProvider>(
              semantic_index_, runtime_dependencies_.query_encoder);
    } else {
      RunLogger::logGlobal(
          "semantic_index",
          "no prewarmed query encoder; live search uses pinned lexical fallback");
    }
  }

  if (persistence_actor_ && config_.dam_artifacts_enabled) {
    if (runtime_dependencies_.dam_worker) {
      SceneStore* const store = persistence_actor_->durableArtifactStore();
      artifact_scheduler_ = std::make_unique<ArtifactScheduler>(store);
      artifact_runtime_actor_ = std::make_unique<ArtifactRuntimeActor>(
          ArtifactRuntimeActorConfig{}, artifact_scheduler_.get(),
          runtime_dependencies_.dam_worker,
          [this]() { return instance_map_thread_.sceneSnapshot(); },
          [this](const ApplyDescriptionArtifactCommand& command,
                 std::chrono::milliseconds timeout) {
            const std::optional<SceneApplyResult> applied =
                instance_map_thread_.applySceneCommandAndWait(
                    SceneCommand{command}, timeout);
            if (!applied) {
              return ArtifactReducerSubmitResult::failure(
                  "scene reducer command timed out or stopped");
            }
            return ArtifactReducerSubmitResult::success(*applied);
          },
          [this](SceneRevision revision,
                 std::chrono::milliseconds timeout) {
            return persistence_actor_ &&
                   persistence_actor_->waitUntilDurable(revision, timeout);
          },
          ArtifactRuntimeActor::TerminalFailureSink{},
          ArtifactRuntimeActor::UnixMillisClock{},
          [this]() {
            return !config_.detection_enabled ||
                   (detection_queue_.empty() && python_backend_.idle() &&
                    inference_response_queue_.empty());
          });
    } else {
      RunLogger::logGlobal(
          "artifact_runtime",
          "DAM enabled without a resident worker; durable tasks remain pending");
    }
  }

  if (persistence_actor_ && semantic_index_ &&
      config_.embedding_artifacts_enabled) {
    if (runtime_dependencies_.embedding_encoder) {
      SceneStore* const store = persistence_actor_->durableArtifactStore();
      embedding_task_scheduler_ =
          std::make_unique<EmbeddingTaskScheduler>(store);
      scheduleMissingEmbeddingJobs(
          {}, instance_map_thread_.sceneSnapshot());
      EmbeddingRuntimeActorConfig runtime_config;
      runtime_config.maximum_batch_size = config_.embedding_batch_size;
      embedding_runtime_actor_ = std::make_unique<EmbeddingRuntimeActor>(
          std::move(runtime_config), embedding_task_scheduler_.get(),
          runtime_dependencies_.embedding_encoder, semantic_index_.get(),
          [this]() { return instance_map_thread_.sceneSnapshot(); },
          [store](const EmbeddingRecord& record) {
            return persistSemanticEmbeddingRecord(store, record);
          },
          [store](const EmbeddingTaskLease& lease,
                  std::int64_t failed_at_unix_ms,
                  std::string error) {
            return store->failTask(lease.taskId(), lease.owner(),
                                   lease.attempt(), failed_at_unix_ms,
                                   std::move(error));
          });
    } else {
      RunLogger::logGlobal(
          "embedding_runtime",
          "embedding enabled without a resident encoder; durable tasks remain pending");
    }
  }
  scene_query_gateway_ = std::make_unique<SceneQueryGateway>(
      [this]() { return instance_map_thread_.sceneSnapshot(); },
      semantic_search_provider_, snapshot_asset_provider_);
}

RoomiePipeline::~RoomiePipeline() { stop(); }

const SceneQueryGateway& RoomiePipeline::sceneQueryGateway() const {
  if (!scene_query_gateway_) {
    throw std::logic_error("RoomiePipeline query gateway is not initialized");
  }
  return *scene_query_gateway_;
}

SceneMutationSubmitResult RoomiePipeline::mutateScene(
    ApplyHumanAnnotationCommand command,
    std::chrono::milliseconds timeout) {
  std::lock_guard<std::mutex> mutation_lock(scene_mutation_mutex_);
  SceneMutationSubmitResult response;
  if (timeout < std::chrono::milliseconds::zero()) {
    timeout = std::chrono::milliseconds::zero();
  }
  if (persistence_actor_ && !persistence_actor_->admissionAllowed()) {
    const SceneSnapshot current = instance_map_thread_.sceneSnapshot();
    const PersistenceActorStatus persistence = persistence_actor_->status();
    response.status = SceneMutationStatus::kUnavailable;
    response.latest_scene_revision = current.revision();
    response.durable_scene_revision =
        persistence.durable_scene_revision;
    response.message =
        "scene mutation admission is closed because persistence cannot "
        "accept another authoritative revision";
    if (persistence.hard_lag_reached) {
      response.message +=
          "; hard undurable revision limit reached (live=" +
          std::to_string(current.revision()) + " durable=" +
          std::to_string(persistence.durable_scene_revision) + ")";
    }
    if (!persistence.last_error.empty()) {
      response.message += ": " + persistence.last_error;
    }
    RunLogger::logGlobal(
        "scene_mutation",
        "admission_rejected latest_scene_revision=" +
            std::to_string(response.latest_scene_revision) +
            " durable_scene_revision=" +
            std::to_string(response.durable_scene_revision) +
            " hard_lag=" +
            std::string(persistence.hard_lag_reached ? "true" : "false") +
            " healthy=" +
            std::string(persistence.healthy ? "true" : "false"));
    return response;
  }
  const auto started = std::chrono::steady_clock::now();
  const std::optional<SceneApplyResult> applied =
      instance_map_thread_.applySceneCommandAndWait(
          SceneCommand{std::move(command)}, timeout);
  if (!applied) {
    const SceneSnapshot current = instance_map_thread_.sceneSnapshot();
    response.status = SceneMutationStatus::kTimeout;
    response.message =
        "scene reducer did not acknowledge the command before the deadline; "
        "the outcome is unknown and a CAS-safe retry is required";
    response.latest_scene_revision = current.revision();
    response.durable_scene_revision =
        persistence_actor_ ? persistence_actor_->durableRevision()
                           : current.durableRevision();
    return response;
  }

  response.latest_scene_revision = applied->revision;
  response.durable_scene_revision =
      persistence_actor_ ? persistence_actor_->durableRevision()
                         : applied->snapshot.durableRevision();
  response.message = applied->reason;
  for (const SceneEvent& event : applied->events) {
    if (const auto* annotation =
            std::get_if<HumanAnnotationCommitted>(&event)) {
      response.component_revision = annotation->annotation_revision;
      break;
    }
  }

  if (!applied->accepted()) {
    if (applied->reason.find("persistence") != std::string::npos) {
      response.status = SceneMutationStatus::kUnavailable;
    } else {
      response.status = applied->reason.find("stale") != std::string::npos
                            ? SceneMutationStatus::kStale
                            : SceneMutationStatus::kRejected;
    }
    return response;
  }
  if (!applied->committedRevision()) {
    response.status = SceneMutationStatus::kNoOp;
    return response;
  }

  response.status = SceneMutationStatus::kCommitted;
  response.committed_scene_revision = applied->revision;
  if (!persistence_actor_) {
    return response;
  }

  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  const std::chrono::milliseconds remaining =
      elapsed < timeout ? timeout - elapsed : std::chrono::milliseconds::zero();
  if (persistence_actor_->waitUntilDurable(applied->revision, remaining)) {
    response.durable_scene_revision = persistence_actor_->durableRevision();
    return response;
  }

  const PersistenceActorStatus persistence = persistence_actor_->status();
  response.status = SceneMutationStatus::kCommittedNotDurable;
  response.durable_scene_revision = persistence.durable_scene_revision;
  response.message =
      "scene mutation committed in memory at revision " +
      std::to_string(applied->revision) +
      " but durability did not reach it before the deadline";
  if (!persistence.last_error.empty()) {
    response.message += ": " + persistence.last_error;
  }
  return response;
}

void RoomiePipeline::start() {
  if (started_) {
    return;
  }
  RunLogger::logGlobal("pipeline", "start");
  started_ = true;
  try {
    if (persistence_actor_) {
      persistence_actor_->start();
    }
    if (online_snapshot_worker_) {
      online_snapshot_worker_->start();
    }
    instance_map_thread_.start();
    if (embedding_runtime_actor_) {
      embedding_runtime_actor_->start();
      if (!embedding_runtime_actor_->waitUntilWarmed(
              std::chrono::milliseconds(
                  static_cast<std::int64_t>(
                      config_.embedding_startup_timeout_ms) +
                  1000))) {
        throw std::runtime_error(
            "resident embedding encoder did not become ready before its startup deadline: " +
            embedding_runtime_actor_->lastError());
      }
    }
    geometry_worker_thread_.start();
    map_thread_.start();
    if (config_.detection_enabled) {
      python_backend_.start();
      detection_bridge_thread_.start();
    }
    publisher_persistence_thread_.start();
    ros_io_thread_.start();
    // DAM is the lowest-priority GPU consumer. Start it only after perception
    // admission is live; its per-job gate then defers every lease boundary
    // while a detection is queued, executing, or awaiting reducer delivery.
    if (artifact_runtime_actor_) {
      artifact_runtime_actor_->start();
    }
  } catch (...) {
    stop();
    throw;
  }
}

void RoomiePipeline::stop() {
  if (!started_) {
    return;
  }

  RunLogger::logGlobal("pipeline", "stop requested");
  const auto instance_drain_timeout =
      std::chrono::milliseconds(config_.shutdown_drain_timeout_ms);
  const auto drain_instance = [this, instance_drain_timeout](
                                  const std::string& stage) {
    if (instance_map_thread_.waitUntilIdle(instance_drain_timeout)) {
      return true;
    }
    const SnapshotControlQueueStats controls =
        instance_map_thread_.snapshotControlStats();
    RunLogger::logGlobal(
        "pipeline",
        "instance drain deadline expired stage=" + stage +
            " timeout_ms=" +
            std::to_string(instance_drain_timeout.count()) +
            " snapshot_control_depth=" +
            std::to_string(controls.depth) +
            " snapshot_control_rejected=" +
            std::to_string(controls.rejected_capacity) +
            " snapshot_control_rejected_closed=" +
            std::to_string(controls.rejected_closed) +
            " snapshot_control_abandoned=" +
            std::to_string(controls.abandoned_on_stop));
    return false;
  };
  // Stop admission first. Candidate map entries already in the mapping
  // channel are replacement-protected and are drained before perception.
  ros_io_thread_.stop();
  detection_queue_.stop();
  mapping_queue_.stop();
  map_thread_.stop();
  if (config_.detection_enabled) {
    detection_bridge_thread_.stop();
    python_backend_.stop();
  }
  inference_response_queue_.stop();
  // Let all observation commits schedule geometry before closing its input.
  (void)drain_instance("post_perception");
  if (online_snapshot_worker_) {
    // Ownership controls must be handed off while the worker still admits
    // them. Any residual backlog after the deadline is explicitly closed and
    // counted instead of keeping shutdown alive forever.
    online_snapshot_worker_->stop();
    (void)instance_map_thread_.closeSnapshotControlsForShutdown();
    (void)drain_instance("post_snapshot_worker");
  }
  geometry_worker_thread_.stop();
  // Geometry's final CAS commands are part of the same reducer stream.
  (void)drain_instance("post_geometry");
  if (artifact_runtime_actor_) {
    artifact_runtime_actor_->stop();
    if (const auto python_worker =
            std::dynamic_pointer_cast<PythonDamWorker>(
                runtime_dependencies_.dam_worker)) {
      python_worker->shutdown();
    }
    // A final accepted DAM command must enter the same durability stream
    // before the store is flushed and closed.
    (void)drain_instance("post_artifact_runtime");
  }
  if (embedding_runtime_actor_) {
    embedding_runtime_actor_->stop();
    if (const auto python_encoder =
            std::dynamic_pointer_cast<PythonEmbeddingEncoder>(
                runtime_dependencies_.embedding_encoder)) {
      python_encoder->shutdown();
    }
  }
  if (persistence_actor_) {
    // Final durability acknowledgement is still serialized by the live
    // reducer before its command channel is closed.
    persistence_actor_->stop();
    (void)drain_instance("post_persistence");
  }
  if (config_.save_map) {
    if (!persistence_actor_) {
      RunLogger::logGlobal(
          "map_checkpoint",
          "checkpoint skipped: SceneStore is disabled, so no durable scene "
          "alignment can be published");
    } else {
      const PersistenceActorStatus persistence = persistence_actor_->status();
      if (!persistence.graceful_shutdown_complete ||
          persistence.latest_scene_revision !=
              persistence.durable_scene_revision) {
        RunLogger::logGlobal(
            "map_checkpoint",
            "checkpoint skipped: scene persistence did not reach a graceful "
            "durable boundary latest=" +
                std::to_string(persistence.latest_scene_revision) +
                " durable=" +
                std::to_string(persistence.durable_scene_revision));
      } else {
        MapCheckpointOperationResult checkpoint =
            map_thread_.saveConfiguredCheckpoint();
        if (checkpoint.succeeded()) {
          checkpoint.manifest.aligned_scene_revision =
              persistence.durable_scene_revision;
          SceneStore* const store =
              persistence_actor_->durableArtifactStore();
          const SceneStoreStatus published =
              store != nullptr
                  ? store->publishMapCheckpoint(checkpoint.manifest)
                  : SceneStoreStatus::failure(
                        "SceneStore closed before checkpoint publication");
          if (published) {
            RunLogger::logGlobal(
                "map_checkpoint",
                "published path=" +
                    checkpoint.manifest.checkpoint_path +
                    " map_epoch=" +
                    runIdString(checkpoint.manifest.map_epoch) +
                    " map_revision=" +
                    std::to_string(checkpoint.manifest.map_revision) +
                    " aligned_scene_revision=" +
                    std::to_string(
                        checkpoint.manifest.aligned_scene_revision));
          } else {
            RunLogger::logGlobal(
                "map_checkpoint",
                "checkpoint file saved but manifest publication failed path=" +
                    checkpoint.manifest.checkpoint_path + " error=" +
                    published.error);
          }
        } else if (checkpoint.disposition !=
                   MapCheckpointDisposition::kNotRequested) {
          RunLogger::logGlobal(
              "map_checkpoint",
              "checkpoint not published: " + checkpoint.error);
        }
      }
    }
  }
  instance_map_thread_.stop();
  publisher_persistence_thread_.stop();
  RunLogger::logGlobal("pipeline", "stopped");
  started_ = false;
}

}  // namespace roomie
