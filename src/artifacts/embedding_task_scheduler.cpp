#include "roomie/artifacts/embedding_task_scheduler.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "roomie/artifacts/artifact_intent_builder.hpp"

namespace roomie {
namespace {

using Json = nlohmann::json;

SceneStoreStatus invalid(std::string error) {
  return SceneStoreStatus::failure(std::move(error));
}

void validateConfig(const EmbeddingTaskSchedulerConfig& config) {
  if (config.lease_duration_ms <= 0 ||
      config.retry_initial_backoff_ms <= 0 ||
      config.retry_max_backoff_ms <= 0 ||
      config.retry_initial_backoff_ms > config.retry_max_backoff_ms) {
    throw std::invalid_argument(
        "embedding task scheduler durations are invalid");
  }
}

bool lowercaseSha256(const std::string& value) {
  if (value.size() != 64U) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](unsigned char byte) {
    return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
  });
}

const Json& requiredField(const Json& payload,
                          const char* name,
                          Json::value_t type) {
  const auto it = payload.find(name);
  if (it == payload.end() || it->type() != type) {
    throw std::runtime_error(std::string("embedding payload field '") +
                             name + "' has an invalid type");
  }
  return *it;
}

std::uint64_t requiredUnsigned(const Json& payload, const char* name) {
  const auto it = payload.find(name);
  if (it == payload.end() || !it->is_number_integer()) {
    throw std::runtime_error(std::string("embedding payload field '") +
                             name + "' must be an integer");
  }
  if (it->is_number_unsigned()) {
    return it->get<std::uint64_t>();
  }
  const std::int64_t value = it->get<std::int64_t>();
  if (value < 0) {
    throw std::runtime_error(std::string("embedding payload field '") +
                             name + "' cannot be negative");
  }
  return static_cast<std::uint64_t>(value);
}

std::int64_t requiredNonnegativeInt64(const Json& payload,
                                      const char* name) {
  const std::uint64_t value = requiredUnsigned(payload, name);
  if (value > static_cast<std::uint64_t>(
                  std::numeric_limits<std::int64_t>::max())) {
    throw std::runtime_error(std::string("embedding payload field '") +
                             name + "' exceeds int64 range");
  }
  return static_cast<std::int64_t>(value);
}

std::size_t checkedDimension(std::uint64_t value) {
  if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("embedding payload dimension is invalid");
  }
  return static_cast<std::size_t>(value);
}

ArtifactPriority priorityFromName(const std::string& name) {
  if (name == "interactive") {
    return ArtifactPriority::kInteractive;
  }
  if (name == "bulk") {
    return ArtifactPriority::kBulk;
  }
  throw std::runtime_error("embedding artifact SLO priority is invalid");
}

ArtifactSloContext artifactSloFromJson(const Json& value) {
  if (!value.is_object()) {
    throw std::runtime_error("embedding artifact_slo must be an object");
  }
  static const std::set<std::string> kFields = {
      "origin_created_unix_ms", "due_unix_ms", "priority",
      "steady_clock_epoch", "origin_steady_ns", "due_steady_ns"};
  for (const auto& [name, ignored] : value.items()) {
    (void)ignored;
    if (kFields.count(name) == 0) {
      throw std::runtime_error(
          "embedding artifact_slo has unknown field '" + name + "'");
    }
  }
  ArtifactSloContext slo;
  const auto origin = value.find("origin_created_unix_ms");
  const auto due = value.find("due_unix_ms");
  if (origin == value.end() || !origin->is_number_integer() ||
      due == value.end() || !due->is_number_integer()) {
    throw std::runtime_error(
        "embedding artifact_slo times must be integers");
  }
  slo.origin_created_unix_ms = origin->get<std::int64_t>();
  slo.due_unix_ms = due->get<std::int64_t>();
  slo.priority = priorityFromName(
      requiredField(value, "priority", Json::value_t::string)
          .get<std::string>());
  const bool has_epoch = value.contains("steady_clock_epoch");
  const bool has_origin_steady = value.contains("origin_steady_ns");
  const bool has_due_steady = value.contains("due_steady_ns");
  if (has_epoch || has_origin_steady || has_due_steady) {
    if (!has_epoch || !has_origin_steady || !has_due_steady) {
      throw std::runtime_error(
          "embedding artifact_slo monotonic tuple is incomplete");
    }
    const Json& epoch = requiredField(
        value, "steady_clock_epoch", Json::value_t::object);
    slo.steady_clock_epoch.high =
        requiredUnsigned(epoch, "high");
    slo.steady_clock_epoch.low =
        requiredUnsigned(epoch, "low");
    slo.origin_steady_ns =
        requiredNonnegativeInt64(value, "origin_steady_ns");
    slo.due_steady_ns =
        requiredNonnegativeInt64(value, "due_steady_ns");
  }
  if (!slo.tracked() || !slo.valid()) {
    throw std::runtime_error("embedding artifact_slo is invalid");
  }
  return slo;
}

Json artifactSloToJson(const ArtifactSloContext& slo) {
  Json value{{"origin_created_unix_ms", slo.origin_created_unix_ms},
             {"due_unix_ms", slo.due_unix_ms},
             {"priority",
              slo.priority == ArtifactPriority::kInteractive
                  ? "interactive"
                  : "bulk"}};
  if (slo.monotonicTracked()) {
    value["steady_clock_epoch"] =
        Json{{"high", slo.steady_clock_epoch.high},
             {"low", slo.steady_clock_epoch.low}};
    value["origin_steady_ns"] = slo.origin_steady_ns;
    value["due_steady_ns"] = slo.due_steady_ns;
  }
  return value;
}

SceneRevision checkedRevision(std::uint64_t value, const char* field) {
  if (value == 0 || value > std::numeric_limits<SceneRevision>::max()) {
    throw std::runtime_error(std::string("embedding payload field '") +
                             field + "' is not a valid revision");
  }
  return static_cast<SceneRevision>(value);
}

EmbeddingTaskRequest requestFromRecord(const DurableTaskRecord& record) {
  const Json payload = Json::parse(record.task.payload);
  if (!payload.is_object()) {
    throw std::runtime_error("embedding payload must be a JSON object");
  }
  static const std::set<std::string> kRequiredFields = {
      "payload_version",       "owning_scene_revision",
      "object_id",             "document",
      "document_hash",         "model_id",
      "dimension",             "semantic_revision",
      "created_scene_revision", "identity_revision",
      "artifact_revision",     "description_input_hash"};
  static const std::set<std::string> kOptionalFields = {
      "annotation_revision", "artifact_slo"};
  for (const auto& [name, value] : payload.items()) {
    (void)value;
    if (kRequiredFields.count(name) == 0 &&
        kOptionalFields.count(name) == 0) {
      throw std::runtime_error("embedding payload has unknown field '" +
                               name + "'");
    }
  }
  for (const std::string& name : kRequiredFields) {
    if (!payload.contains(name)) {
      throw std::runtime_error("embedding payload is missing field '" + name +
                               "'");
    }
  }

  const std::string payload_version =
      requiredField(payload, "payload_version", Json::value_t::string)
          .get<std::string>();
  if (payload_version != "roomie.embedding-input.v1") {
    throw std::runtime_error("embedding payload version is unsupported");
  }

  EmbeddingTaskRequest request;
  request.owning_scene_revision = checkedRevision(
      requiredUnsigned(payload, "owning_scene_revision"),
      "owning_scene_revision");
  const std::uint64_t object_id = requiredUnsigned(payload, "object_id");
  if (object_id >
      static_cast<std::uint64_t>(std::numeric_limits<SceneObjectId>::max())) {
    throw std::runtime_error("embedding payload object id is invalid");
  }
  request.object_id = static_cast<SceneObjectId>(object_id);
  request.document =
      requiredField(payload, "document", Json::value_t::string)
          .get<std::string>();
  request.document_hash =
      requiredField(payload, "document_hash", Json::value_t::string)
          .get<std::string>();
  request.name_space.model_id =
      requiredField(payload, "model_id", Json::value_t::string)
          .get<std::string>();
  request.name_space.dimension =
      checkedDimension(requiredUnsigned(payload, "dimension"));
  request.semantic_revision =
      requiredUnsigned(payload, "semantic_revision");
  request.created_scene_revision = checkedRevision(
      requiredUnsigned(payload, "created_scene_revision"),
      "created_scene_revision");
  request.identity_revision =
      requiredUnsigned(payload, "identity_revision");
  request.artifact_revision =
      requiredUnsigned(payload, "artifact_revision");
  request.description_input_hash =
      requiredField(payload, "description_input_hash", Json::value_t::string)
          .get<std::string>();
  if (payload.contains("annotation_revision")) {
    request.annotation_revision =
        requiredUnsigned(payload, "annotation_revision");
  }
  if (payload.contains("artifact_slo")) {
    request.artifact_slo = artifactSloFromJson(payload.at("artifact_slo"));
  }

  if (request.document.empty() || !lowercaseSha256(request.document_hash) ||
      request.name_space.model_id.empty() || request.semantic_revision == 0 ||
      request.identity_revision == 0 || request.artifact_revision == 0 ||
      (request.annotation_revision && *request.annotation_revision == 0) ||
      !request.artifact_slo.valid()) {
    throw std::runtime_error("embedding payload has an empty or zero field");
  }
  if (request.created_scene_revision != request.owning_scene_revision ||
      record.task.scene_revision != request.owning_scene_revision ||
      record.task.task_type != EmbeddingTaskScheduler::kTaskType ||
      record.task.task_id != record.task.dedupe_key ||
      record.task.task_id != canonicalEmbeddingTaskKey(
                                 request.object_id,
                                 request.document_hash,
                                 request.name_space)) {
    throw std::runtime_error(
        "embedding durable record does not match its payload identity");
  }
  return request;
}

SceneStoreStatus validateRequest(const EmbeddingTaskRequest& request,
                                 std::int64_t not_before_unix_ms,
                                 const std::string& task_type) {
  if (request.owning_scene_revision == 0 || request.object_id < 0 ||
      request.document.empty() || !lowercaseSha256(request.document_hash) ||
      request.name_space.model_id.empty() ||
      request.name_space.dimension == 0 || request.semantic_revision == 0 ||
      request.created_scene_revision == 0 ||
      request.created_scene_revision != request.owning_scene_revision ||
      request.identity_revision == 0 || request.artifact_revision == 0 ||
      (request.annotation_revision && *request.annotation_revision == 0) ||
      !request.artifact_slo.valid() || not_before_unix_ms < 0 ||
      task_type.empty()) {
    return invalid("embedding task request is invalid");
  }
  return SceneStoreStatus::success();
}

Json requestToJson(const EmbeddingTaskRequest& request) {
  Json payload{{"payload_version", "roomie.embedding-input.v1"},
               {"owning_scene_revision", request.owning_scene_revision},
               {"object_id", request.object_id},
               {"document", request.document},
               {"document_hash", request.document_hash},
               {"model_id", request.name_space.model_id},
               {"dimension", request.name_space.dimension},
               {"semantic_revision", request.semantic_revision},
               {"created_scene_revision", request.created_scene_revision},
               {"identity_revision", request.identity_revision},
               {"artifact_revision", request.artifact_revision},
               {"description_input_hash", request.description_input_hash}};
  if (request.annotation_revision) {
    payload["annotation_revision"] = *request.annotation_revision;
  }
  if (request.artifact_slo.tracked()) {
    payload["artifact_slo"] = artifactSloToJson(request.artifact_slo);
  }
  return payload;
}

std::int64_t retryDelay(const EmbeddingTaskSchedulerConfig& config,
                        std::uint64_t attempt) {
  std::int64_t delay = config.retry_initial_backoff_ms;
  for (std::uint64_t index = 1;
       index < attempt && delay < config.retry_max_backoff_ms; ++index) {
    if (delay > config.retry_max_backoff_ms / 2) {
      return config.retry_max_backoff_ms;
    }
    delay *= 2;
  }
  return std::min(delay, config.retry_max_backoff_ms);
}

std::optional<std::int64_t> checkedAdd(std::int64_t lhs, std::int64_t rhs) {
  if (rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

}  // namespace

EmbeddingDependencyFreshness inspectEmbeddingDependency(
    const SceneSnapshot& snapshot,
    const EmbeddingTaskRequest& request,
    SemanticDocument* current_document) {
  const std::optional<SceneObjectId> canonical =
      snapshot.resolveCanonicalId(request.object_id);
  if (!canonical) {
    return EmbeddingDependencyFreshness::kInvalid;
  }
  if (*canonical != request.object_id) {
    return EmbeddingDependencyFreshness::kRetiredAlias;
  }
  if (snapshot.isTombstoned(request.object_id)) {
    return EmbeddingDependencyFreshness::kTombstoned;
  }
  const SceneObjectPtr object = snapshot.findExactObject(request.object_id);
  if (!object) {
    return EmbeddingDependencyFreshness::kMissingObject;
  }
  if (!object->identity || object->identity->object_id != request.object_id ||
      object->identity->revision != request.identity_revision) {
    return EmbeddingDependencyFreshness::kIdentityStale;
  }
  if (!object->semantic || !object->artifact || !object->annotation) {
    return EmbeddingDependencyFreshness::kInvalid;
  }
  SemanticDocument document;
  try {
    document = makeSemanticDocumentForObject(
        *object, request.object_id, request.created_scene_revision,
        SemanticDocumentView::kPendingDescriptionIfPresent);
  } catch (...) {
    return EmbeddingDependencyFreshness::kInvalid;
  }
  if (document.document_hash != request.document_hash ||
      document.text != request.document) {
    return EmbeddingDependencyFreshness::kDocumentStale;
  }
  if (current_document) {
    *current_document = std::move(document);
  }
  return EmbeddingDependencyFreshness::kCurrent;
}

EmbeddingTaskScheduler::EmbeddingTaskScheduler(
    SceneStore* store,
    EmbeddingTaskSchedulerConfig config)
    : store_(store), config_(config) {
  validateConfig(config_);
}

DurableTaskSpec EmbeddingTaskScheduler::makeTaskSpec(
    const EmbeddingTaskRequest& request,
    std::int64_t not_before_unix_ms,
    std::string task_type,
    SceneStoreStatus* status) const {
  auto set_status = [status](SceneStoreStatus value) {
    if (status != nullptr) {
      *status = std::move(value);
    }
  };
  const SceneStoreStatus validated =
      validateRequest(request, not_before_unix_ms, task_type);
  if (!validated) {
    set_status(validated);
    return {};
  }
  DurableTaskSpec task;
  task.task_id = canonicalEmbeddingTaskKey(
      request.object_id, request.document_hash, request.name_space);
  task.dedupe_key = task.task_id;
  task.task_type = std::move(task_type);
  task.payload = requestToJson(request).dump();
  task.scene_revision = request.owning_scene_revision;
  task.not_before_unix_ms = not_before_unix_ms;
  set_status(SceneStoreStatus::success());
  return task;
}

SceneStoreStatus EmbeddingTaskScheduler::ensureTask(
    const EmbeddingJob& job,
    const SceneSnapshot& snapshot,
    std::int64_t not_before_unix_ms) const {
  if (store_ == nullptr) {
    return invalid("embedding scheduler has no SceneStore");
  }
  if (!snapshot.statePtr() || job.object_id < 0 ||
      job.created_scene_revision == 0 ||
      job.created_scene_revision > snapshot.durableRevision()) {
    return invalid(
        "missing embedding task depends on a non-durable scene revision");
  }
  const SceneObjectPtr object = snapshot.findExactObject(job.object_id);
  if (!object || snapshot.isTombstoned(job.object_id) || !object->identity ||
      !object->semantic || !object->annotation || !object->artifact) {
    return invalid("missing embedding task object is not current");
  }
  SemanticDocument current;
  try {
    current = makeSemanticDocumentForObject(
        *object, job.object_id, job.created_scene_revision);
  } catch (const std::exception& error) {
    return invalid(std::string("cannot build missing embedding document: ") +
                   error.what());
  }
  // The durable identity of an embedding is the canonical document hash, not
  // the component revision that happened to expose it. Geometry/metadata-only
  // commits may advance semantic_revision without changing the document. Such
  // a commit must not turn the same missing embedding into a permanently stale
  // recovery job.
  if (current.text != job.document ||
      current.document_hash != job.document_hash) {
    return invalid("missing embedding job is stale for the durable scene");
  }

  EmbeddingTaskRequest request;
  request.owning_scene_revision = job.created_scene_revision;
  request.object_id = job.object_id;
  request.document = job.document;
  request.document_hash = job.document_hash;
  request.name_space = job.name_space;
  request.semantic_revision = job.semantic_revision;
  request.created_scene_revision = job.created_scene_revision;
  request.identity_revision = object->identity->revision;
  request.artifact_revision = object->artifact->revision;
  if (object->annotation->revision != 0) {
    request.annotation_revision = object->annotation->revision;
  }
  request.description_input_hash = object->artifact->description_input_hash;
  request.artifact_slo = object->artifact->description_slo;
  SceneStoreStatus status;
  const DurableTaskSpec task = makeTaskSpec(
      request, not_before_unix_ms, kTaskType, &status);
  if (!status) {
    return status;
  }

  // A same-identity task atomically emitted with its reducer commit is already
  // authoritative. It may carry more precise event provenance than this
  // projector recovery path, so never replace it with a conflicting envelope.
  const TaskLookupResult existing = store_->lookupTask(task.task_id);
  if (!existing.status) {
    return existing.status;
  }
  // task_id is derived from object/document/namespace. Any existing record,
  // including a completed one, therefore satisfies this recovery admission.
  // Retrying ensureTask() with newer provenance or not_before metadata would
  // otherwise conflict with the immutable envelope and hot-loop forever.
  if (existing.task) {
    return SceneStoreStatus::success();
  }

  const SceneStoreStatus ensured = store_->ensureTask(task);
  if (ensured) {
    return ensured;
  }
  // retryTask() mutates not_before while preserving immutable payload.
  const TaskLookupResult after = store_->lookupTask(task.task_id);
  if (after.status && after.task && after.task->task.task_id == task.task_id &&
      after.task->task.dedupe_key == task.dedupe_key &&
      after.task->task.task_type == task.task_type &&
      after.task->task.payload == task.payload &&
      after.task->task.scene_revision == task.scene_revision &&
      after.task->task.not_before_unix_ms >= task.not_before_unix_ms) {
    return SceneStoreStatus::success();
  }
  return ensured;
}

EmbeddingTaskLeaseResult EmbeddingTaskScheduler::leaseNext(
    const std::string& lease_owner,
    std::int64_t now_unix_ms) const {
  EmbeddingTaskLeaseResult result;
  if (!store_) {
    result.status = invalid("embedding scheduler has no SceneStore");
    return result;
  }
  TaskLeaseResult leased = store_->leaseNextTask(
      std::vector<std::string>{kTaskType}, lease_owner, now_unix_ms,
      config_.lease_duration_ms);
  result.status = leased.status;
  if (!leased.status || !leased.task) {
    return result;
  }
  EmbeddingTaskLease embedding_lease;
  embedding_lease.durable = std::move(*leased.task);
  try {
    embedding_lease.request = requestFromRecord(embedding_lease.durable);
  } catch (const std::exception& error) {
    embedding_lease.validation_error =
        std::string("invalid durable embedding task: ") + error.what();
  }
  result.lease = std::move(embedding_lease);
  return result;
}

SceneStoreStatus EmbeddingTaskScheduler::heartbeat(
    const EmbeddingTaskLease& lease,
    std::int64_t now_unix_ms) const {
  if (!store_) {
    return invalid("embedding scheduler has no SceneStore");
  }
  return store_->renewTaskLease(lease.taskId(), lease.owner(), lease.attempt(),
                                now_unix_ms, config_.lease_duration_ms);
}

SceneStoreStatus EmbeddingTaskScheduler::complete(
    const EmbeddingTaskLease& lease,
    std::int64_t completed_at_unix_ms) const {
  if (!store_) {
    return invalid("embedding scheduler has no SceneStore");
  }
  const TaskLookupResult current = store_->lookupTask(lease.taskId());
  if (!current.status) {
    return current.status;
  }
  if (!current.task) {
    return invalid("embedding task disappeared before completion");
  }
  if (current.task->state != DurableTaskState::kCompleted &&
      (current.task->state != DurableTaskState::kLeased ||
       current.task->lease_owner != lease.owner() ||
       current.task->attempts != lease.attempt() ||
       current.task->lease_until_unix_ms <= completed_at_unix_ms)) {
    return invalid("embedding task lease expired or lost completion fence");
  }
  return store_->completeTask(lease.taskId(), lease.owner(), lease.attempt(),
                              completed_at_unix_ms);
}

EmbeddingTaskRetryResult EmbeddingTaskScheduler::retry(
    const EmbeddingTaskLease& lease,
    std::int64_t now_unix_ms,
    std::string error) const {
  EmbeddingTaskRetryResult result;
  if (!store_) {
    result.status = invalid("embedding scheduler has no SceneStore");
    return result;
  }
  const TaskLookupResult current = store_->lookupTask(lease.taskId());
  if (!current.status) {
    result.status = current.status;
    return result;
  }
  if (!current.task || current.task->state != DurableTaskState::kLeased ||
      current.task->lease_owner != lease.owner() ||
      current.task->attempts != lease.attempt() ||
      current.task->lease_until_unix_ms <= now_unix_ms) {
    result.status =
        invalid("embedding task lease expired or lost retry fence");
    return result;
  }
  const std::optional<std::int64_t> retry_at =
      checkedAdd(now_unix_ms, retryDelay(config_, lease.attempt()));
  if (!retry_at) {
    result.status = invalid("embedding retry time overflows");
    return result;
  }
  result.retry_at_unix_ms = *retry_at;
  result.status = store_->retryTask(lease.taskId(), lease.owner(),
                                    lease.attempt(), *retry_at,
                                    std::move(error));
  return result;
}

}  // namespace roomie
