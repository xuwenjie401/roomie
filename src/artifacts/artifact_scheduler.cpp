#include "roomie/artifacts/artifact_scheduler.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "roomie/artifacts/artifact_slo_clock.hpp"

namespace roomie {
namespace {

using Json = nlohmann::json;

constexpr int kTaskPayloadVersion = 1;

SceneStoreStatus invalid(std::string message) {
  return SceneStoreStatus::failure(std::move(message));
}

bool positiveDuration(std::int64_t value) { return value > 0; }

SceneStoreStatus validateConfig(const ArtifactSchedulerConfig& config) {
  if (!positiveDuration(config.interactive_due_ms) ||
      !positiveDuration(config.bulk_due_ms) ||
      !positiveDuration(config.lease_duration_ms) ||
      !positiveDuration(config.retry_initial_backoff_ms) ||
      !positiveDuration(config.retry_max_backoff_ms) ||
      config.retry_initial_backoff_ms > config.retry_max_backoff_ms) {
    return invalid("artifact scheduler durations are invalid");
  }
  return SceneStoreStatus::success();
}

SceneStoreStatus validateKey(const DamTaskKey& key) {
  if (key.object_id < 0 || key.identity_revision == 0 ||
      key.appearance_revision == 0 || key.snapshot_set_hash.empty() ||
      key.model_id.empty() || key.prompt_hash.empty() ||
      key.output_schema_version.empty()) {
    return invalid("DAM task key has an invalid or empty field");
  }
  return SceneStoreStatus::success();
}

std::optional<std::int64_t> checkedAdd(std::int64_t lhs, std::int64_t rhs) {
  if (rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() - rhs) {
    return std::nullopt;
  }
  if (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::min() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

std::string taskType(ArtifactPriority priority) {
  switch (priority) {
    case ArtifactPriority::kInteractive:
      return ArtifactScheduler::kInteractiveTaskType;
    case ArtifactPriority::kBulk:
      return ArtifactScheduler::kBulkTaskType;
  }
  return {};
}

ArtifactPriority priorityFromName(const std::string& name) {
  if (name == "interactive") {
    return ArtifactPriority::kInteractive;
  }
  if (name == "bulk") {
    return ArtifactPriority::kBulk;
  }
  throw std::runtime_error("DAM task payload has invalid priority");
}

const Json& requiredField(const Json& value,
                          const char* name,
                          Json::value_t type) {
  if (!value.is_object()) {
    throw std::runtime_error("DAM task payload value must be an object");
  }
  const auto field = value.find(name);
  if (field == value.end() || field->type() != type) {
    throw std::runtime_error(std::string("DAM task payload field '") + name +
                             "' has an invalid type");
  }
  return *field;
}

std::uint64_t requiredUnsigned(const Json& value, const char* name) {
  const auto field = value.find(name);
  if (!value.is_object() || field == value.end() ||
      (!field->is_number_unsigned() && !field->is_number_integer())) {
    throw std::runtime_error(std::string("DAM task payload field '") + name +
                             "' must be an unsigned integer");
  }
  if (field->is_number_unsigned()) {
    return field->get<std::uint64_t>();
  }
  const std::int64_t signed_value = field->get<std::int64_t>();
  if (signed_value < 0) {
    throw std::runtime_error(std::string("DAM task payload field '") + name +
                             "' is negative");
  }
  return static_cast<std::uint64_t>(signed_value);
}

std::int64_t requiredInt64(const Json& value,
                           const char* name,
                           bool nonnegative) {
  const auto field = value.find(name);
  if (!value.is_object() || field == value.end() ||
      (!field->is_number_unsigned() && !field->is_number_integer())) {
    throw std::runtime_error(std::string("DAM task payload field '") + name +
                             "' must be an integer");
  }
  std::int64_t result = 0;
  if (field->is_number_unsigned()) {
    const std::uint64_t raw = field->get<std::uint64_t>();
    if (raw > static_cast<std::uint64_t>(
                  std::numeric_limits<std::int64_t>::max())) {
      throw std::runtime_error(std::string("DAM task payload field '") +
                               name + "' exceeds int64 range");
    }
    result = static_cast<std::int64_t>(raw);
  } else {
    result = field->get<std::int64_t>();
  }
  if (nonnegative && result < 0) {
    throw std::runtime_error(std::string("DAM task payload field '") + name +
                             "' is negative");
  }
  return result;
}

SceneObjectId requiredObjectId(const Json& value, const char* name) {
  const std::uint64_t raw = requiredUnsigned(value, name);
  if (raw > static_cast<std::uint64_t>(
                std::numeric_limits<SceneObjectId>::max())) {
    throw std::runtime_error(std::string("DAM task payload field '") + name +
                             "' exceeds object id range");
  }
  return static_cast<SceneObjectId>(raw);
}

Json keyToJson(const DamTaskKey& key) {
  return Json{{"object_id", key.object_id},
              {"identity_revision", key.identity_revision},
              {"appearance_revision", key.appearance_revision},
              {"snapshot_set_hash", key.snapshot_set_hash},
              {"model_id", key.model_id},
              {"prompt_hash", key.prompt_hash},
              {"output_schema_version", key.output_schema_version}};
}

DamTaskKey keyFromJson(const Json& value) {
  DamTaskKey key;
  key.object_id = requiredObjectId(value, "object_id");
  key.identity_revision = requiredUnsigned(value, "identity_revision");
  key.appearance_revision = requiredUnsigned(value, "appearance_revision");
  key.snapshot_set_hash = requiredField(
      value, "snapshot_set_hash", Json::value_t::string).get<std::string>();
  key.model_id = requiredField(
      value, "model_id", Json::value_t::string).get<std::string>();
  key.prompt_hash = requiredField(
      value, "prompt_hash", Json::value_t::string).get<std::string>();
  key.output_schema_version = requiredField(
      value, "output_schema_version",
      Json::value_t::string).get<std::string>();
  return key;
}

Json dependencyToJson(const ObjectDependency& dependency) {
  return Json{{"object_id", dependency.object_id},
              {"identity_revision", dependency.identity_revision},
              {"obb_revision", dependency.obb_revision},
              {"appearance_revision", dependency.appearance_revision},
              {"semantic_revision", dependency.semantic_revision}};
}

ObjectDependency dependencyFromJson(const Json& value) {
  ObjectDependency dependency;
  dependency.object_id = requiredObjectId(value, "object_id");
  dependency.identity_revision = requiredUnsigned(value, "identity_revision");
  dependency.obb_revision = requiredUnsigned(value, "obb_revision");
  dependency.appearance_revision =
      requiredUnsigned(value, "appearance_revision");
  dependency.semantic_revision = requiredUnsigned(value, "semantic_revision");
  return dependency;
}

SceneStoreStatus validateIntent(const DamTaskIntent& intent) {
  const SceneStoreStatus key_status = validateKey(intent.key);
  if (!key_status) {
    return key_status;
  }
  if (intent.scene_revision == 0 || intent.created_unix_ms < 0) {
    return invalid("DAM task scene revision/time is invalid");
  }
  const bool no_monotonic = !intent.steady_clock_epoch.valid() &&
                            intent.origin_steady_ns == 0 &&
                            intent.due_steady_ns == 0;
  const bool valid_monotonic = intent.steady_clock_epoch.valid() &&
                               intent.origin_steady_ns > 0 &&
                               (intent.due_steady_ns == 0 ||
                                intent.due_steady_ns >=
                                    intent.origin_steady_ns);
  if (!no_monotonic && !valid_monotonic) {
    return invalid("DAM task monotonic SLO tuple is invalid");
  }
  switch (intent.priority) {
    case ArtifactPriority::kInteractive:
    case ArtifactPriority::kBulk:
      break;
    default:
      return invalid("DAM task priority is invalid");
  }
  if (intent.dependency.object_id != intent.key.object_id ||
      intent.dependency.identity_revision != intent.key.identity_revision ||
      intent.dependency.appearance_revision !=
          intent.key.appearance_revision) {
    return invalid("DAM dependency does not match its stable task key");
  }
  return SceneStoreStatus::success();
}

Json requestToJson(const DamTaskRequest& request) {
  Json value{{"payload_version", kTaskPayloadVersion},
             {"key", keyToJson(request.key)},
             {"dependency", dependencyToJson(request.dependency)},
             {"scene_revision", request.scene_revision},
             {"priority", artifactPriorityName(request.priority)},
             {"created_unix_ms", request.created_unix_ms},
             {"due_unix_ms", request.due_unix_ms},
             {"input_payload", request.input_payload}};
  if (request.steady_clock_epoch.valid()) {
    value["steady_clock_epoch"] =
        Json{{"high", request.steady_clock_epoch.high},
             {"low", request.steady_clock_epoch.low}};
    value["origin_steady_ns"] = request.origin_steady_ns;
    value["due_steady_ns"] = request.due_steady_ns;
  }
  return value;
}

DamTaskRequest requestFromRecord(const DurableTaskRecord& record) {
  const Json payload = Json::parse(record.task.payload);
  if (requiredInt64(payload, "payload_version", true) !=
      kTaskPayloadVersion) {
    throw std::runtime_error("DAM task payload version is unsupported");
  }
  DamTaskRequest request;
  request.key = keyFromJson(payload.at("key"));
  request.dependency = dependencyFromJson(payload.at("dependency"));
  const std::uint64_t scene_revision =
      requiredUnsigned(payload, "scene_revision");
  if (scene_revision == 0 ||
      scene_revision > std::numeric_limits<SceneRevision>::max()) {
    throw std::runtime_error("DAM task scene revision is invalid");
  }
  request.scene_revision = static_cast<SceneRevision>(scene_revision);
  request.priority =
      priorityFromName(requiredField(
          payload, "priority", Json::value_t::string).get<std::string>());
  request.created_unix_ms =
      requiredInt64(payload, "created_unix_ms", true);
  request.due_unix_ms = requiredInt64(payload, "due_unix_ms", true);
  const bool has_epoch = payload.contains("steady_clock_epoch");
  const bool has_origin_steady = payload.contains("origin_steady_ns");
  const bool has_due_steady = payload.contains("due_steady_ns");
  if (has_epoch || has_origin_steady || has_due_steady) {
    if (!has_epoch || !has_origin_steady || !has_due_steady) {
      throw std::runtime_error("DAM task monotonic SLO tuple is incomplete");
    }
    const Json& epoch = requiredField(
        payload, "steady_clock_epoch", Json::value_t::object);
    request.steady_clock_epoch.high = requiredUnsigned(epoch, "high");
    request.steady_clock_epoch.low = requiredUnsigned(epoch, "low");
    request.origin_steady_ns =
        requiredInt64(payload, "origin_steady_ns", true);
    request.due_steady_ns =
        requiredInt64(payload, "due_steady_ns", true);
  }
  request.input_payload = requiredField(
      payload, "input_payload", Json::value_t::string).get<std::string>();

  const DamTaskIntent intent{request.key,
                             request.dependency,
                             request.scene_revision,
                             request.priority,
                             request.created_unix_ms,
                             request.due_unix_ms,
                             request.steady_clock_epoch,
                             request.origin_steady_ns,
                             request.due_steady_ns,
                             request.input_payload};
  const SceneStoreStatus intent_status = validateIntent(intent);
  if (!intent_status) {
    throw std::runtime_error(intent_status.error);
  }
  const std::string stable_key = canonicalDamTaskKey(request.key);
  if (record.task.task_id != stable_key ||
      record.task.dedupe_key != stable_key ||
      record.task.task_type != taskType(request.priority) ||
      record.task.scene_revision != request.scene_revision ||
      // retryTask() advances the durable not-before column while the original
      // creation time remains part of the immutable payload.
      record.task.not_before_unix_ms < request.created_unix_ms ||
      request.due_unix_ms < request.created_unix_ms ||
      (request.steady_clock_epoch.valid() &&
       request.due_steady_ns < request.origin_steady_ns)) {
    throw std::runtime_error("DAM durable record does not match its payload");
  }
  return request;
}

std::string trim(std::string value) {
  const auto first = std::find_if_not(
      value.begin(), value.end(),
      [](unsigned char character) { return std::isspace(character) != 0; });
  const auto last = std::find_if_not(
      value.rbegin(), value.rend(),
      [](unsigned char character) { return std::isspace(character) != 0; })
                        .base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

bool stringArray(const Json& value,
                 const char* field,
                 std::vector<std::string>* output,
                 std::string* error) {
  const auto it = value.find(field);
  if (it == value.end() || !it->is_array()) {
    *error = std::string("field '") + field + "' must be an array";
    return false;
  }
  output->clear();
  for (const Json& item : *it) {
    if (!item.is_string()) {
      *error = std::string("field '") + field +
               "' must contain only strings";
      return false;
    }
    output->push_back(item.get<std::string>());
  }
  return true;
}

bool requiredString(const Json& value,
                    const char* field,
                    std::string* output,
                    std::string* error) {
  const auto it = value.find(field);
  if (it == value.end() || !it->is_string()) {
    *error = std::string("field '") + field + "' must be a string";
    return false;
  }
  *output = trim(it->get<std::string>());
  if (output->empty()) {
    *error = std::string("field '") + field + "' must not be empty";
    return false;
  }
  return true;
}

bool decodeStructured(const Json& value,
                      DamArtifact* artifact,
                      std::string* error) {
  if (!value.is_object()) {
    *error = "DAM output must be a JSON object";
    return false;
  }
  if (!requiredString(value, "canonical_name", &artifact->canonical_name,
                      error) ||
      !requiredString(value, "short_description",
                      &artifact->short_description, error) ||
      !requiredString(value, "retrieval_text", &artifact->retrieval_text,
                      error)) {
    return false;
  }

  const auto attributes_it = value.find("visual_attributes");
  if (attributes_it == value.end() || !attributes_it->is_object()) {
    *error = "field 'visual_attributes' must be an object";
    return false;
  }
  const Json& attributes = *attributes_it;
  if (!stringArray(attributes, "colors", &artifact->visual_attributes.colors,
                   error) ||
      !stringArray(attributes, "materials",
                   &artifact->visual_attributes.materials, error) ||
      !stringArray(attributes, "shape", &artifact->visual_attributes.shape,
                   error) ||
      !stringArray(attributes, "visible_parts",
                   &artifact->visual_attributes.visible_parts, error) ||
      !stringArray(attributes, "state_or_pose",
                   &artifact->visual_attributes.state_or_pose, error) ||
      !stringArray(attributes, "distinctive_marks",
                   &artifact->visual_attributes.distinctive_marks, error) ||
      !stringArray(attributes, "visible_text",
                   &artifact->visual_attributes.visible_text, error) ||
      !stringArray(value, "uncertain_or_not_visible",
                   &artifact->uncertain_or_not_visible, error) ||
      !stringArray(value, "evidence_snapshot_ids",
                   &artifact->evidence_snapshot_ids, error)) {
    return false;
  }

  const auto confidence_it = value.find("confidence");
  if (confidence_it == value.end() || !confidence_it->is_number()) {
    *error = "field 'confidence' must be a number";
    return false;
  }
  artifact->confidence = confidence_it->get<double>();
  if (!std::isfinite(artifact->confidence) || artifact->confidence < 0.0 ||
      artifact->confidence > 1.0) {
    *error = "field 'confidence' must be finite and within [0, 1]";
    return false;
  }

  if (!requiredString(value, "mask_source", &artifact->mask_source, error)) {
    return false;
  }
  if (artifact->mask_source != "instance_mask" &&
      artifact->mask_source != "bbox_fallback") {
    *error = "field 'mask_source' has an unsupported value";
    return false;
  }
  return true;
}

Json structuredToJson(const DamArtifact& artifact) {
  return Json{
      {"canonical_name", artifact.canonical_name},
      {"short_description", artifact.short_description},
      {"retrieval_text", artifact.retrieval_text},
      {"visual_attributes",
       {{"colors", artifact.visual_attributes.colors},
        {"materials", artifact.visual_attributes.materials},
        {"shape", artifact.visual_attributes.shape},
        {"visible_parts", artifact.visual_attributes.visible_parts},
        {"state_or_pose", artifact.visual_attributes.state_or_pose},
        {"distinctive_marks", artifact.visual_attributes.distinctive_marks},
        {"visible_text", artifact.visual_attributes.visible_text}}},
      {"uncertain_or_not_visible", artifact.uncertain_or_not_visible},
      {"confidence", artifact.confidence},
      {"evidence_snapshot_ids", artifact.evidence_snapshot_ids},
      {"mask_source", artifact.mask_source}};
}

std::string parsePathName(DamArtifactParsePath path) {
  switch (path) {
    case DamArtifactParsePath::kSchemaValid:
      return "schema_valid";
    case DamArtifactParsePath::kSchemaRepaired:
      return "schema_repaired";
    case DamArtifactParsePath::kUnstructuredFallback:
      return "unstructured_fallback";
  }
  return "unstructured_fallback";
}

void finalizeArtifact(DamArtifact* artifact,
                      const std::string& output_schema_version) {
  Json envelope{{"artifact_status", parsePathName(artifact->parse_path)},
                {"output_schema_version", output_schema_version},
                {"raw_text", artifact->raw_text}};
  if (!artifact->schema_error.empty()) {
    envelope["schema_error"] = artifact->schema_error;
  }
  if (artifact->structured()) {
    const Json normalized = structuredToJson(*artifact);
    artifact->normalized_json = normalized.dump();
    envelope["structured"] = std::move(normalized);
  } else {
    envelope["structured"] = nullptr;
  }
  artifact->durable_envelope_json = envelope.dump();
}

bool tryParseStructured(const std::string& candidate,
                        DamArtifact* artifact,
                        std::string* error,
                        std::string* worker_repair_note = nullptr) {
  try {
    const Json value = Json::parse(candidate);
    if (!decodeStructured(value, artifact, error)) {
      return false;
    }
    if (worker_repair_note != nullptr &&
        value.value("_roomie_parse_path", std::string{}) ==
            "schema_repaired") {
      *worker_repair_note = value.value(
          "_roomie_schema_error",
          std::string("DAM worker conservatively repaired model output"));
    }
    return true;
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

std::optional<std::string> fencedJsonCandidate(const std::string& raw_text) {
  const std::string cleaned = trim(raw_text);
  const std::size_t first = cleaned.find('{');
  const std::size_t last = cleaned.rfind('}');
  if (first == std::string::npos || last == std::string::npos || first >= last) {
    return std::nullopt;
  }
  const std::string candidate = cleaned.substr(first, last - first + 1);
  if (candidate == cleaned) {
    return std::nullopt;
  }
  return candidate;
}

std::int64_t retryDelay(const ArtifactSchedulerConfig& config,
                        std::uint64_t attempt) {
  std::int64_t delay = config.retry_initial_backoff_ms;
  for (std::uint64_t index = 1;
       index < attempt && delay < config.retry_max_backoff_ms; ++index) {
    if (delay > config.retry_max_backoff_ms / 2) {
      delay = config.retry_max_backoff_ms;
    } else {
      delay *= 2;
    }
  }
  return std::min(delay, config.retry_max_backoff_ms);
}

}  // namespace

const char* artifactPriorityName(ArtifactPriority priority) {
  switch (priority) {
    case ArtifactPriority::kInteractive:
      return "interactive";
    case ArtifactPriority::kBulk:
      return "bulk";
  }
  return "invalid";
}

bool operator==(const DamTaskKey& lhs, const DamTaskKey& rhs) {
  return lhs.object_id == rhs.object_id &&
         lhs.identity_revision == rhs.identity_revision &&
         lhs.appearance_revision == rhs.appearance_revision &&
         lhs.snapshot_set_hash == rhs.snapshot_set_hash &&
         lhs.model_id == rhs.model_id && lhs.prompt_hash == rhs.prompt_hash &&
         lhs.output_schema_version == rhs.output_schema_version;
}

bool operator!=(const DamTaskKey& lhs, const DamTaskKey& rhs) {
  return !(lhs == rhs);
}

std::string canonicalDamTaskKey(const DamTaskKey& key) {
  auto append_string = [](std::ostringstream* stream,
                          const char* name,
                          const std::string& value) {
    *stream << '|' << name << '=' << value.size() << ':' << value;
  };
  std::ostringstream stream;
  stream << "dam:v1|object_id=" << key.object_id
         << "|identity_revision=" << key.identity_revision
         << "|appearance_revision=" << key.appearance_revision;
  append_string(&stream, "snapshot_set_hash", key.snapshot_set_hash);
  append_string(&stream, "model_id", key.model_id);
  append_string(&stream, "prompt_hash", key.prompt_hash);
  append_string(&stream, "output_schema_version", key.output_schema_version);
  return stream.str();
}

DamArtifact parseDamArtifact(const std::string& raw_text,
                             const std::string& output_schema_version,
                             DamRepairFunction repair) {
  DamArtifact artifact;
  artifact.raw_text = raw_text;

  std::string direct_error;
  std::string worker_repair_note;
  if (tryParseStructured(raw_text, &artifact, &direct_error,
                         &worker_repair_note)) {
    artifact.parse_path = worker_repair_note.empty()
                              ? DamArtifactParsePath::kSchemaValid
                              : DamArtifactParsePath::kSchemaRepaired;
    artifact.schema_error = std::move(worker_repair_note);
    finalizeArtifact(&artifact, output_schema_version);
    return artifact;
  }

  if (const std::optional<std::string> candidate =
          fencedJsonCandidate(raw_text)) {
    DamArtifact repaired;
    repaired.raw_text = raw_text;
    std::string repair_error;
    if (tryParseStructured(*candidate, &repaired, &repair_error)) {
      repaired.parse_path = DamArtifactParsePath::kSchemaRepaired;
      repaired.schema_error = direct_error;
      finalizeArtifact(&repaired, output_schema_version);
      return repaired;
    }
    direct_error += "; fenced JSON repair failed: " + repair_error;
  }

  if (repair) {
    try {
      const std::optional<std::string> repaired_text =
          repair(raw_text, direct_error);
      if (repaired_text) {
        DamArtifact repaired;
        repaired.raw_text = raw_text;
        std::string repair_error;
        if (tryParseStructured(*repaired_text, &repaired, &repair_error)) {
          repaired.parse_path = DamArtifactParsePath::kSchemaRepaired;
          repaired.schema_error = direct_error;
          finalizeArtifact(&repaired, output_schema_version);
          return repaired;
        }
        direct_error += "; supplied repair failed: " + repair_error;
      }
    } catch (const std::exception& exception) {
      direct_error += "; repair callback threw: ";
      direct_error += exception.what();
    } catch (...) {
      direct_error += "; repair callback threw a non-standard exception";
    }
  }

  DamArtifact fallback;
  fallback.parse_path = DamArtifactParsePath::kUnstructuredFallback;
  fallback.raw_text = raw_text;
  fallback.short_description = trim(raw_text);
  fallback.retrieval_text = fallback.short_description;
  fallback.schema_error = std::move(direct_error);
  finalizeArtifact(&fallback, output_schema_version);
  return fallback;
}

DamDependencyFreshness inspectDamDependency(const SceneSnapshot& snapshot,
                                             const DamTaskRequest& request) {
  if (!validateKey(request.key) ||
      request.dependency.object_id != request.key.object_id ||
      request.dependency.identity_revision != request.key.identity_revision ||
      request.dependency.appearance_revision !=
          request.key.appearance_revision) {
    return DamDependencyFreshness::kInvalid;
  }
  const std::optional<SceneObjectId> canonical =
      snapshot.resolveCanonicalId(request.key.object_id);
  if (!canonical) {
    return DamDependencyFreshness::kInvalid;
  }
  if (*canonical != request.key.object_id) {
    return DamDependencyFreshness::kRetiredAlias;
  }
  if (snapshot.isTombstoned(request.key.object_id)) {
    return DamDependencyFreshness::kTombstoned;
  }
  const SceneObjectPtr object = snapshot.findExactObject(request.key.object_id);
  if (!object) {
    return DamDependencyFreshness::kMissingObject;
  }
  if (!object->identity ||
      object->identity->revision != request.key.identity_revision) {
    return DamDependencyFreshness::kIdentityStale;
  }
  if (!object->artifact || object->artifact->appearance_revision !=
                               request.key.appearance_revision) {
    return DamDependencyFreshness::kAppearanceStale;
  }
  if (object->artifact->snapshot_set_hash != request.key.snapshot_set_hash) {
    return DamDependencyFreshness::kSnapshotSetStale;
  }
  if (request.dependency.semantic_revision != 0 &&
      (!object->semantic || object->semantic->revision !=
                                request.dependency.semantic_revision)) {
    return DamDependencyFreshness::kSemanticStale;
  }
  return DamDependencyFreshness::kCurrent;
}

ArtifactScheduler::ArtifactScheduler(SceneStore* store,
                                     ArtifactSchedulerConfig config)
    : store_(store), config_(config) {}

DurableTaskSpec ArtifactScheduler::makeTaskSpec(
    const DamTaskIntent& intent,
    SceneStoreStatus* status) const {
  auto set_status = [status](SceneStoreStatus value) {
    if (status) {
      *status = std::move(value);
    }
  };
  const SceneStoreStatus config_status = validateConfig(config_);
  if (!config_status) {
    set_status(config_status);
    return {};
  }
  const SceneStoreStatus intent_status = validateIntent(intent);
  if (!intent_status) {
    set_status(intent_status);
    return {};
  }

  DamTaskRequest request;
  request.key = intent.key;
  request.dependency = intent.dependency;
  // DAM pixels and output identity do not depend on the continuously fused
  // semantic component. Zero is the reducer's explicit wildcard and keeps
  // ordinary observation updates from starving a long-running DAM task.
  request.dependency.semantic_revision = 0;
  request.scene_revision = intent.scene_revision;
  request.priority = intent.priority;
  request.created_unix_ms = intent.created_unix_ms;
  request.steady_clock_epoch = intent.steady_clock_epoch;
  request.origin_steady_ns = intent.origin_steady_ns;
  request.input_payload = intent.input_payload;
  const std::int64_t duration =
      intent.priority == ArtifactPriority::kInteractive
          ? config_.interactive_due_ms
          : config_.bulk_due_ms;
  if (intent.due_unix_ms != 0) {
    request.due_unix_ms = intent.due_unix_ms;
  } else {
    const std::optional<std::int64_t> due =
        checkedAdd(intent.created_unix_ms, duration);
    if (!due) {
      set_status(invalid("DAM due time overflows"));
      return {};
    }
    request.due_unix_ms = *due;
  }
  if (request.due_unix_ms < request.created_unix_ms) {
    set_status(invalid("DAM due time precedes creation time"));
    return {};
  }
  if (request.steady_clock_epoch.valid()) {
    if (intent.due_steady_ns != 0) {
      request.due_steady_ns = intent.due_steady_ns;
    } else {
      if (duration > std::numeric_limits<std::int64_t>::max() / 1'000'000) {
        set_status(invalid("DAM monotonic due duration overflows"));
        return {};
      }
      const std::optional<std::int64_t> due = checkedAdd(
          intent.origin_steady_ns, duration * 1'000'000);
      if (!due) {
        set_status(invalid("DAM monotonic due time overflows"));
        return {};
      }
      request.due_steady_ns = *due;
    }
    if (request.due_steady_ns < request.origin_steady_ns) {
      set_status(invalid("DAM monotonic due time precedes creation time"));
      return {};
    }
  }

  DurableTaskSpec spec;
  spec.task_id = canonicalDamTaskKey(intent.key);
  spec.dedupe_key = spec.task_id;
  spec.task_type = taskType(intent.priority);
  spec.payload = requestToJson(request).dump();
  spec.scene_revision = intent.scene_revision;
  spec.not_before_unix_ms = intent.created_unix_ms;
  set_status(SceneStoreStatus::success());
  return spec;
}

SceneStoreStatus ArtifactScheduler::ensureTask(
    const DamTaskIntent& intent) const {
  if (!store_) {
    return invalid("artifact scheduler has no SceneStore");
  }
  SceneStoreStatus status;
  const DurableTaskSpec spec = makeTaskSpec(intent, &status);
  if (!status) {
    return status;
  }
  const SceneStoreStatus ensured = store_->ensureTask(spec);
  if (ensured) {
    return ensured;
  }

  // SceneStore uses one not_before column for both immutable admission and
  // mutable retry scheduling. After retryTask() advances that column, a
  // byte-identical ensureTask() currently reports a content conflict. Treat an
  // otherwise exact record as idempotent here; do not mask any real payload,
  // type, revision, id, or dedupe-key conflict.
  const TaskLookupResult lookup = store_->lookupTask(spec.task_id);
  if (!lookup.status || !lookup.task ||
      lookup.task->task.task_id != spec.task_id ||
      lookup.task->task.dedupe_key != spec.dedupe_key ||
      lookup.task->task.task_type != spec.task_type ||
      lookup.task->task.payload != spec.payload ||
      lookup.task->task.scene_revision != spec.scene_revision ||
      lookup.task->task.not_before_unix_ms < spec.not_before_unix_ms) {
    return ensured;
  }
  return SceneStoreStatus::success();
}

ArtifactLeaseResult ArtifactScheduler::leaseNext(
    const std::string& lease_owner,
    std::int64_t now_unix_ms) const {
  ArtifactLeaseResult result;
  if (!store_) {
    result.status = invalid("artifact scheduler has no SceneStore");
    return result;
  }
  const SceneStoreStatus config_status = validateConfig(config_);
  if (!config_status) {
    result.status = config_status;
    return result;
  }
  TaskLeaseResult leased = store_->leaseNextTask(
      std::vector<std::string>{kInteractiveTaskType, kBulkTaskType},
      lease_owner, now_unix_ms, config_.lease_duration_ms);
  result.status = leased.status;
  if (!leased.status || !leased.task) {
    return result;
  }
  try {
    ArtifactLease artifact_lease;
    artifact_lease.request = requestFromRecord(*leased.task);
    artifact_lease.durable = std::move(*leased.task);
    result.lease = std::move(artifact_lease);
    return result;
  } catch (const std::exception& error) {
    result.status = invalid(std::string("invalid durable DAM task: ") +
                            error.what());
    return result;
  }
}

SceneStoreStatus ArtifactScheduler::heartbeat(
    const ArtifactLease& lease,
    std::int64_t now_unix_ms) const {
  if (!store_) {
    return invalid("artifact scheduler has no SceneStore");
  }
  return store_->renewTaskLease(lease.taskId(), lease.owner(), lease.attempt(),
                                now_unix_ms, config_.lease_duration_ms);
}

SceneStoreStatus ArtifactScheduler::complete(
    const ArtifactLease& lease,
    std::int64_t completed_at_unix_ms) const {
  if (!store_) {
    return invalid("artifact scheduler has no SceneStore");
  }
  const TaskLookupResult current = store_->lookupTask(lease.taskId());
  if (!current.status) {
    return current.status;
  }
  if (!current.task) {
    return invalid("DAM task disappeared before completion");
  }
  if (current.task->state != DurableTaskState::kCompleted &&
      (current.task->state != DurableTaskState::kLeased ||
       current.task->lease_owner != lease.owner() ||
       current.task->attempts != lease.attempt() ||
       current.task->lease_until_unix_ms <= completed_at_unix_ms)) {
    return invalid("DAM task lease expired or lost its completion fence");
  }
  return store_->completeTask(lease.taskId(), lease.owner(), lease.attempt(),
                              completed_at_unix_ms);
}

SceneStoreStatus ArtifactScheduler::fail(
    const ArtifactLease& lease,
    std::int64_t failed_at_unix_ms,
    std::string error) const {
  if (!store_) {
    return invalid("artifact scheduler has no SceneStore");
  }
  if (error.empty()) {
    return invalid("DAM terminal failure requires an error");
  }
  const TaskLookupResult current = store_->lookupTask(lease.taskId());
  if (!current.status) {
    return current.status;
  }
  if (!current.task) {
    return invalid("DAM task disappeared before terminal failure");
  }
  if (current.task->state == DurableTaskState::kFailed) {
    if (current.task->failed_by != lease.owner() ||
        current.task->attempts != lease.attempt()) {
      return invalid("DAM task was failed by a different lease attempt");
    }
  } else if (current.task->state != DurableTaskState::kLeased ||
             current.task->lease_owner != lease.owner() ||
             current.task->attempts != lease.attempt() ||
             current.task->lease_until_unix_ms <= failed_at_unix_ms) {
    return invalid("DAM task lease expired or lost its failure fence");
  }
  return store_->failTask(lease.taskId(), lease.owner(), lease.attempt(),
                          failed_at_unix_ms, std::move(error));
}

ArtifactRetryResult ArtifactScheduler::retry(const ArtifactLease& lease,
                                             std::int64_t now_unix_ms,
                                             std::string error) const {
  ArtifactRetryResult result;
  if (!store_) {
    result.status = invalid("artifact scheduler has no SceneStore");
    return result;
  }
  const SceneStoreStatus config_status = validateConfig(config_);
  if (!config_status) {
    result.status = config_status;
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
    result.status = invalid("DAM task lease expired or lost its retry fence");
    return result;
  }
  const std::int64_t delay = retryDelay(config_, lease.attempt());
  const std::optional<std::int64_t> retry_at = checkedAdd(now_unix_ms, delay);
  if (!retry_at) {
    result.status = invalid("artifact retry time overflows");
    return result;
  }
  result.retry_at_unix_ms = *retry_at;
  result.status = store_->retryTask(lease.taskId(), lease.owner(),
                                    lease.attempt(), *retry_at,
                                    std::move(error));
  return result;
}

ArtifactExecutionResult ArtifactScheduler::execute(
    const ArtifactLease& lease,
    DamWorker* worker,
    std::int64_t started_unix_ms,
    DamRepairFunction repair) const {
  ArtifactExecutionResult result;
  const ArtifactSloContext execution_slo{
      lease.request.created_unix_ms,
      lease.request.due_unix_ms,
      lease.request.priority,
      lease.request.steady_clock_epoch,
      lease.request.origin_steady_ns,
      lease.request.due_steady_ns};
  result.slo_violation =
      hasCurrentArtifactSteadyClock(execution_slo)
          ? artifactSteadyNowNanoseconds() > execution_slo.due_steady_ns
          : started_unix_ms > execution_slo.due_unix_ms;
  if (!worker || !store_ ||
      lease.durable.state != DurableTaskState::kLeased ||
      lease.owner().empty() || lease.attempt() == 0) {
    result.status = ArtifactExecutionStatus::kInvalidLease;
    result.error = "DAM execution requires a live fenced lease and worker";
    return result;
  }
  const TaskLookupResult current = store_->lookupTask(lease.taskId());
  if (!current.status || !current.task ||
      current.task->state != DurableTaskState::kLeased ||
      current.task->lease_owner != lease.owner() ||
      current.task->attempts != lease.attempt() ||
      current.task->lease_until_unix_ms <= started_unix_ms) {
    result.status = ArtifactExecutionStatus::kInvalidLease;
    result.error = current.status
                       ? "DAM execution lease expired or lost its attempt fence"
                       : current.status.error;
    return result;
  }

  bool heartbeat_failed = false;
  const DamLeaseHeartbeat heartbeat_callback =
      [this, &lease, &heartbeat_failed](std::int64_t now_unix_ms) {
        const SceneStoreStatus status = heartbeat(lease, now_unix_ms);
        heartbeat_failed = heartbeat_failed || !status;
        return static_cast<bool>(status);
      };

  DamWorkerResponse response;
  try {
    response = worker->describe(lease.request, heartbeat_callback);
  } catch (const std::exception& error) {
    result.status = ArtifactExecutionStatus::kRetryableFailure;
    result.error = std::string("DAM worker threw: ") + error.what();
    return result;
  } catch (...) {
    result.status = ArtifactExecutionStatus::kRetryableFailure;
    result.error = "DAM worker threw a non-standard exception";
    return result;
  }
  if (heartbeat_failed) {
    result.status = ArtifactExecutionStatus::kInvalidLease;
    result.error = "DAM worker lost its fenced lease during execution";
    return result;
  }
  if (!response.success) {
    result.status = response.retryable
                        ? ArtifactExecutionStatus::kRetryableFailure
                        : ArtifactExecutionStatus::kPermanentFailure;
    result.error = response.error.empty() ? "DAM worker failed"
                                          : std::move(response.error);
    return result;
  }

  DamArtifact artifact = parseDamArtifact(
      response.raw_output, lease.request.key.output_schema_version,
      std::move(repair));
  ApplyDescriptionArtifactCommand command;
  command.dependency = lease.request.dependency;
  command.description = artifact.short_description;
  command.input_hash = canonicalDamTaskKey(lease.request.key);
  command.model_id = lease.request.key.model_id;
  command.schema_version = lease.request.key.output_schema_version;
  command.raw_text = artifact.raw_text;
  command.normalized_json = artifact.normalized_json;
  command.durable_envelope_json = artifact.durable_envelope_json;
  command.parse_path = parsePathName(artifact.parse_path);
  command.artifact_slo.origin_created_unix_ms =
      lease.request.created_unix_ms;
  command.artifact_slo.due_unix_ms = lease.request.due_unix_ms;
  command.artifact_slo.priority = lease.request.priority;
  command.artifact_slo.steady_clock_epoch =
      lease.request.steady_clock_epoch;
  command.artifact_slo.origin_steady_ns = lease.request.origin_steady_ns;
  command.artifact_slo.due_steady_ns = lease.request.due_steady_ns;
  if (!artifact.structured()) {
    command.schema_version += "/unstructured_fallback";
  }

  result.status = ArtifactExecutionStatus::kReadyToApply;
  result.command = std::move(command);
  result.artifact = std::move(artifact);
  return result;
}

}  // namespace roomie
