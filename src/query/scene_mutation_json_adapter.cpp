#include "roomie/query/scene_mutation_json_adapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace roomie {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kMaximumRequestBytes = 256U * 1024U;
constexpr std::size_t kMaximumTextBytes = 64U * 1024U;
constexpr std::size_t kMaximumAttributeCount = 256U;
constexpr std::size_t kMaximumAttributeKeyBytes = 256U;
constexpr std::size_t kMaximumAttributeValueBytes = 4096U;

struct ParsedMutation {
  ApplyHumanAnnotationCommand command;
  std::chrono::milliseconds timeout =
      SceneMutationJsonAdapter::kDefaultTimeout;
  SceneRevision base_scene_revision = 0;
  SceneEntityRef target;
  std::string request_id;
};

[[noreturn]] void invalid(std::string message) {
  throw std::invalid_argument(std::move(message));
}

void requireObject(const Json& value, std::string_view path) {
  if (!value.is_object()) {
    invalid(std::string(path) + " must be an object");
  }
}

void requireOnlyFields(const Json& value,
                       std::initializer_list<const char*> allowed,
                       std::string_view path) {
  requireObject(value, path);
  std::set<std::string> names;
  for (const char* field : allowed) {
    names.emplace(field);
  }
  for (const auto& [field, ignored] : value.items()) {
    (void)ignored;
    if (names.count(field) == 0) {
      invalid(std::string(path) + " contains unsupported field '" + field +
              "'");
    }
  }
}

std::uint64_t unsignedInteger(const Json& value,
                              std::string_view path,
                              std::uint64_t maximum =
                                  std::numeric_limits<std::uint64_t>::max()) {
  if (!value.is_number_integer()) {
    invalid(std::string(path) + " must be a non-negative integer");
  }
  std::uint64_t result = 0;
  if (value.is_number_unsigned()) {
    result = value.get<std::uint64_t>();
  } else {
    const std::int64_t signed_value = value.get<std::int64_t>();
    if (signed_value < 0) {
      invalid(std::string(path) + " must be a non-negative integer");
    }
    result = static_cast<std::uint64_t>(signed_value);
  }
  if (result > maximum) {
    invalid(std::string(path) + " exceeds the supported range");
  }
  return result;
}

int entityId(const Json& value, std::string_view path) {
  return static_cast<int>(unsignedInteger(
      value, path, static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
}

std::string boundedString(const Json& value,
                          std::string_view path,
                          std::size_t maximum = kMaximumTextBytes) {
  if (!value.is_string()) {
    invalid(std::string(path) + " must be a string");
  }
  std::string result = value.get<std::string>();
  if (result.size() > maximum) {
    invalid(std::string(path) + " is too large");
  }
  return result;
}

float finiteFloat(const Json& value, std::string_view path) {
  if (!value.is_number()) {
    invalid(std::string(path) + " must be a finite number");
  }
  const double result = value.get<double>();
  if (!std::isfinite(result) ||
      std::abs(result) > std::numeric_limits<float>::max()) {
    invalid(std::string(path) + " must be a finite float");
  }
  return static_cast<float>(result);
}

template <std::size_t Size>
std::array<float, Size> floatArray(const Json& value,
                                   std::string_view path) {
  if (!value.is_array() || value.size() != Size) {
    invalid(std::string(path) + " must contain exactly " +
            std::to_string(Size) + " numbers");
  }
  std::array<float, Size> result{};
  for (std::size_t index = 0; index < Size; ++index) {
    result[index] = finiteFloat(
        value.at(index), std::string(path) + "[" + std::to_string(index) + "]");
  }
  return result;
}

Eigen::Vector3f vector3(const Json& value, std::string_view path) {
  const std::array<float, 3> values = floatArray<3>(value, path);
  return Eigen::Vector3f(values[0], values[1], values[2]);
}

std::map<std::string, std::string> attributes(const Json& value,
                                               std::string_view path) {
  requireObject(value, path);
  if (value.size() > kMaximumAttributeCount) {
    invalid(std::string(path) + " has too many entries");
  }
  std::map<std::string, std::string> result;
  for (const auto& [key, item] : value.items()) {
    if (key.empty() || key.size() > kMaximumAttributeKeyBytes) {
      invalid(std::string(path) + " contains an invalid attribute key");
    }
    result.emplace(
        key, boundedString(item, std::string(path) + "." + key,
                           kMaximumAttributeValueBytes));
  }
  return result;
}

std::vector<int> membershipAssertion(const Json& value,
                                     std::string_view path) {
  if (!value.is_array()) {
    invalid(std::string(path) + " must be an array of entity ids");
  }
  std::set<int> unique;
  std::vector<int> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const int id = entityId(
        value.at(index), std::string(path) + "[" + std::to_string(index) + "]");
    if (!unique.insert(id).second) {
      invalid(std::string(path) + " must not contain duplicate ids");
    }
    result.push_back(id);
  }
  std::sort(result.begin(), result.end());
  return result;
}

SceneEntityRef parseTarget(const Json& root) {
  std::optional<SceneEntityRef> target;
  if (root.contains("target")) {
    const Json& value = root.at("target");
    requireOnlyFields(value, {"type", "id"}, "target");
    if (!value.contains("type") || !value.contains("id")) {
      invalid("target requires type and id");
    }
    const std::string type = boundedString(value.at("type"), "target.type", 16);
    SceneEntityType entity_type;
    if (type == "object") {
      entity_type = SceneEntityType::kObject;
    } else if (type == "room") {
      entity_type = SceneEntityType::kRoom;
    } else {
      invalid("target.type must be 'object' or 'room'");
    }
    target = SceneEntityRef{entity_type, entityId(value.at("id"), "target.id")};
  }
  if (root.contains("object_id")) {
    const SceneEntityRef shorthand{
        SceneEntityType::kObject,
        entityId(root.at("object_id"), "object_id")};
    if (target && *target != shorthand) {
      invalid("object_id conflicts with target");
    }
    target = shorthand;
  }
  if (!target) {
    invalid("request requires target or object_id");
  }
  return *target;
}

void parseDependencies(const Json& value,
                       SceneEntityType target_type,
                       ApplyHumanAnnotationCommand* command) {
  requireOnlyFields(value,
                    {"identity_revision", "annotation_revision",
                     "room_revision"},
                    "dependencies");
  if (target_type == SceneEntityType::kRoom) {
    if (value.contains("identity_revision") ||
        value.contains("annotation_revision")) {
      invalid("room target cannot use object component dependencies");
    }
    if (value.contains("room_revision")) {
      command->expected_room_revision = unsignedInteger(
          value.at("room_revision"), "dependencies.room_revision");
    }
    return;
  }
  if (value.contains("room_revision")) {
    invalid("object target cannot use room_revision");
  }
  if (value.contains("identity_revision")) {
    command->expected_identity_revision = unsignedInteger(
        value.at("identity_revision"), "dependencies.identity_revision");
  }
  if (value.contains("annotation_revision")) {
    command->expected_annotation_revision = unsignedInteger(
        value.at("annotation_revision"), "dependencies.annotation_revision");
  }
}

void parseObjectPatch(const Json& value,
                      ApplyHumanAnnotationCommand* command) {
  requireOnlyFields(value,
                    {"semantic_id", "label", "description", "attributes",
                     "room_memberships"},
                    "patch");
  if (value.contains("semantic_id")) {
    command->patch.semantic_id = entityId(value.at("semantic_id"),
                                           "patch.semantic_id");
  }
  if (value.contains("label")) {
    command->patch.label = boundedString(value.at("label"), "patch.label");
  }
  if (value.contains("description")) {
    command->patch.description =
        boundedString(value.at("description"), "patch.description");
  }
  if (value.contains("attributes")) {
    command->patch.attributes =
        attributes(value.at("attributes"), "patch.attributes");
  }
  if (value.contains("room_memberships")) {
    command->expected_room_memberships = membershipAssertion(
        value.at("room_memberships"), "patch.room_memberships");
  }
  if (command->patch.empty() && !command->expected_room_memberships) {
    invalid("object patch is empty");
  }
}

void parseRoomPatch(const Json& value,
                    ApplyHumanAnnotationCommand* command) {
  requireOnlyFields(value,
                    {"label", "color", "center_world", "size_m", "min_xy",
                     "max_xy", "height_m", "attributes", "remove",
                     "room_memberships"},
                    "patch");
  RoomAnnotationPatch patch;
  if (value.contains("label")) {
    patch.label = boundedString(value.at("label"), "patch.label");
  }
  if (value.contains("color")) {
    patch.color = boundedString(value.at("color"), "patch.color", 256);
  }
  if (value.contains("center_world")) {
    patch.center_world = vector3(value.at("center_world"), "patch.center_world");
  }
  if (value.contains("size_m")) {
    patch.size_m = vector3(value.at("size_m"), "patch.size_m");
  }
  if (value.contains("min_xy")) {
    patch.min_xy = floatArray<2>(value.at("min_xy"), "patch.min_xy");
  }
  if (value.contains("max_xy")) {
    patch.max_xy = floatArray<2>(value.at("max_xy"), "patch.max_xy");
  }
  if (value.contains("height_m")) {
    patch.height_m = finiteFloat(value.at("height_m"), "patch.height_m");
  }
  if (value.contains("attributes")) {
    patch.attributes = attributes(value.at("attributes"), "patch.attributes");
  }
  if (value.contains("remove")) {
    if (!value.at("remove").is_boolean()) {
      invalid("patch.remove must be a boolean");
    }
    patch.remove = value.at("remove").get<bool>();
  }
  if (value.contains("room_memberships")) {
    command->expected_room_memberships = membershipAssertion(
        value.at("room_memberships"), "patch.room_memberships");
  }
  if (patch.remove) {
    std::size_t mutation_fields = value.size();
    if (value.contains("room_memberships")) {
      --mutation_fields;
    }
    if (mutation_fields != 1U) {
      invalid("patch.remove cannot be combined with room field updates");
    }
  }
  if (patch.empty() && !command->expected_room_memberships) {
    invalid("room patch is empty");
  }
  command->room_patch = std::move(patch);
}

ParsedMutation parseRequest(std::string_view request_json) {
  if (request_json.empty()) {
    invalid("request_json is empty");
  }
  if (request_json.size() > kMaximumRequestBytes) {
    invalid("request_json exceeds the maximum size");
  }
  const Json root = Json::parse(request_json.begin(), request_json.end());
  requireOnlyFields(root,
                    {"schema_version", "operation", "request_id",
                     "base_scene_revision", "target", "object_id",
                     "dependencies", "patch", "timeout_ms"},
                    "request");
  if (!root.contains("schema_version") || !root.contains("operation") ||
      !root.contains("base_scene_revision") || !root.contains("patch")) {
    invalid("request requires schema_version, operation, base_scene_revision, and patch");
  }
  if (boundedString(root.at("schema_version"), "schema_version", 64) !=
      "roomie.mutate_scene.v1") {
    invalid("unsupported schema_version");
  }
  if (boundedString(root.at("operation"), "operation", 64) !=
      "apply_human_annotation") {
    invalid("unsupported mutation operation");
  }

  ParsedMutation parsed;
  parsed.base_scene_revision = unsignedInteger(
      root.at("base_scene_revision"), "base_scene_revision");
  parsed.target = parseTarget(root);
  parsed.command.expected_scene_revision = parsed.base_scene_revision;
  parsed.command.target = parsed.target;
  if (parsed.target.type == SceneEntityType::kObject) {
    parsed.command.object_id = parsed.target.id;
  }
  if (root.contains("request_id")) {
    parsed.request_id = boundedString(root.at("request_id"), "request_id", 256);
  }
  if (root.contains("dependencies")) {
    parseDependencies(root.at("dependencies"), parsed.target.type,
                      &parsed.command);
  }
  requireObject(root.at("patch"), "patch");
  if (parsed.target.type == SceneEntityType::kRoom) {
    parseRoomPatch(root.at("patch"), &parsed.command);
  } else {
    parseObjectPatch(root.at("patch"), &parsed.command);
  }
  if (root.contains("timeout_ms")) {
    parsed.timeout = std::chrono::milliseconds(unsignedInteger(
        root.at("timeout_ms"), "timeout_ms",
        static_cast<std::uint64_t>(
            SceneMutationJsonAdapter::kMaximumTimeout.count())));
  }
  return parsed;
}

const char* statusName(SceneMutationStatus status) {
  switch (status) {
    case SceneMutationStatus::kCommitted:
      return "committed";
    case SceneMutationStatus::kNoOp:
      return "no_op";
    case SceneMutationStatus::kStale:
      return "stale";
    case SceneMutationStatus::kRejected:
      return "rejected";
    case SceneMutationStatus::kTimeout:
      return "timeout";
    case SceneMutationStatus::kCommittedNotDurable:
      return "committed_not_durable";
    case SceneMutationStatus::kUnavailable:
      return "unavailable";
  }
  return "unavailable";
}

Json targetJson(SceneEntityRef target) {
  return Json{{"type", target.type == SceneEntityType::kRoom ? "room" : "object"},
              {"id", target.id}};
}

SceneMutationJsonResponse renderResponse(
    const ParsedMutation& parsed,
    const SceneMutationSubmitResult& submitted) {
  SceneMutationJsonResponse response;
  response.success = submitted.accepted();
  response.latest_scene_revision = submitted.latest_scene_revision;
  response.durable_scene_revision = submitted.durable_scene_revision;
  if (!response.success) {
    response.error = submitted.message.empty() ? statusName(submitted.status)
                                                : submitted.message;
  }
  Json body{{"schema_version", "roomie.mutate_scene.response.v1"},
            {"request_id", parsed.request_id},
            {"success", response.success},
            {"status", statusName(submitted.status)},
            {"message", submitted.message},
            {"base_scene_revision", parsed.base_scene_revision},
            {"target", targetJson(parsed.target)},
            {"latest_scene_revision", submitted.latest_scene_revision},
            {"durable_scene_revision", submitted.durable_scene_revision},
            {"durable", submitted.durable()},
            {"room_memberships_are_derived", true}};
  body["committed_scene_revision"] =
      submitted.committed_scene_revision
          ? Json(*submitted.committed_scene_revision)
          : Json(nullptr);
  body["component_revision"] =
      submitted.component_revision ? Json(*submitted.component_revision)
                                   : Json(nullptr);
  response.response_json = body.dump();
  return response;
}

}  // namespace

SceneMutationJsonResponse SceneMutationJsonAdapter::dispatch(
    std::string_view request_json) const {
  ParsedMutation parsed;
  try {
    parsed = parseRequest(request_json);
  } catch (const std::exception& error) {
    SceneMutationJsonResponse response;
    response.error = error.what();
    return response;
  }

  SceneMutationSubmitResult submitted;
  if (!submitter_) {
    submitted.status = SceneMutationStatus::kUnavailable;
    submitted.message = "scene mutation submitter is unavailable";
  } else {
    try {
      submitted = submitter_(std::move(parsed.command), parsed.timeout);
    } catch (const std::exception& error) {
      submitted.status = SceneMutationStatus::kUnavailable;
      submitted.message =
          std::string("scene mutation submission failed: ") + error.what();
    } catch (...) {
      submitted.status = SceneMutationStatus::kUnavailable;
      submitted.message = "scene mutation submission failed";
    }
  }
  return renderResponse(parsed, submitted);
}

}  // namespace roomie
