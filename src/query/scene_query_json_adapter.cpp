#include "roomie/query/scene_query_json_adapter.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace roomie {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kRequestSchema = "roomie.query_scene.v1";
constexpr std::string_view kResponseSchema =
    "roomie.query_scene.response.v1";
constexpr std::size_t kMaximumCalls = 128;
constexpr std::int64_t kMaximumTtlMs = 300000;

const char* queryStatusName(QueryStatus status) {
  switch (status) {
    case QueryStatus::kOk:
      return "ok";
    case QueryStatus::kExpiredToken:
      return "expired_token";
    case QueryStatus::kForeignToken:
      return "foreign_token";
    case QueryStatus::kInvalidArgument:
      return "invalid_argument";
    case QueryStatus::kNotFound:
      return "not_found";
    case QueryStatus::kIndexUnavailable:
      return "index_unavailable";
  }
  return "unknown";
}

const char* freshnessName(Freshness freshness) {
  switch (freshness) {
    case Freshness::kCurrent:
      return "current";
    case Freshness::kStale:
      return "stale";
    case Freshness::kPending:
      return "pending";
    case Freshness::kMissing:
      return "missing";
  }
  return "missing";
}

Json vector3Json(const Eigen::Vector3f& value) {
  return Json::array({value.x(), value.y(), value.z()});
}

Json componentRevisionsJson(const ComponentRevisions& revisions) {
  return Json{{"identity", revisions.identity_revision},
              {"lifecycle", revisions.lifecycle_revision},
              {"geometry", revisions.geometry_revision},
              {"semantic", revisions.semantic_revision},
              {"annotation", revisions.annotation_revision},
              {"artifact", revisions.artifact_revision}};
}

Json objectFreshnessJson(const ObjectFreshness& freshness) {
  return Json{{"geometry", freshnessName(freshness.geometry)},
              {"semantic", freshnessName(freshness.semantic)},
              {"description", freshnessName(freshness.description)},
              {"artifact_pending", freshness.artifactPending()},
              {"pending_artifacts", freshness.pending_artifacts}};
}

Json queryMetadataJson(const QueryMetadata& metadata) {
  return Json{
      {"scene_revision", metadata.scene_revision},
      {"durable_scene_revision", metadata.durable_scene_revision},
      {"index_generation", metadata.index_generation},
      {"as_of_ns", metadata.as_of_ns},
      {"index_as_of_ns", metadata.index_as_of_ns},
      {"freshness",
       {{"geometry", freshnessName(metadata.geometry_freshness)},
        {"semantic", freshnessName(metadata.semantic_freshness)},
        {"description", freshnessName(metadata.description_freshness)}}},
      {"artifact_pending", metadata.artifact_pending},
      {"index_available", metadata.index_available},
      {"lexical_delta_used", metadata.lexical_delta_used},
      {"pending_object_ids", metadata.pending_object_ids}};
}

QueryMetadata metadataFromToken(const SceneReadToken& token) {
  QueryMetadata metadata;
  metadata.scene_revision = token.sceneRevision();
  metadata.durable_scene_revision = token.durableSceneRevision();
  metadata.index_generation = token.indexGeneration();
  metadata.as_of_ns = token.asOfNanoseconds();
  return metadata;
}

Json readTokenJson(const SceneReadToken& token) {
  return Json{{"scene_revision", token.sceneRevision()},
              {"durable_scene_revision", token.durableSceneRevision()},
              {"index_generation", token.indexGeneration()},
              {"as_of_ns", token.asOfNanoseconds()}};
}

Json nullableString(const std::optional<std::string>& value) {
  return value ? Json(*value) : Json(nullptr);
}

Json queryObjectJson(const QueryObjectView& object) {
  return Json{
      {"requested_object_id", object.requested_object_id},
      {"object_id", object.object_id},
      {"resolved_alias", object.resolved_alias},
      {"semantic_id", object.semantic_id},
      {"label", object.label},
      {"canonical_description", nullableString(object.canonical_description)},
      {"display_description", object.display_description},
      {"description_is_label_fallback",
       object.description_is_label_fallback},
      {"active", object.active},
      {"publishable", object.publishable},
      {"geometry",
       {{"center_world", vector3Json(object.center_world)},
        {"size_m", vector3Json(object.size_m)},
        {"yaw_rad", object.yaw_rad}}},
      {"revisions", componentRevisionsJson(object.revisions)},
      {"semantic_document_hash", object.semantic_document_hash},
      {"indexed_document_hash", nullableString(object.indexed_document_hash)},
      {"snapshot_set_hash", object.snapshot_set_hash},
      {"freshness", objectFreshnessJson(object.freshness)}};
}

Json entityRefJson(const SceneEntityRef& entity) {
  return Json{{"type", entity.type == SceneEntityType::kRoom ? "room" : "object"},
              {"id", entity.id}};
}

Json relationJson(const CanonicalRelation& relation) {
  return Json{{"source", entityRefJson(relation.source)},
              {"target", entityRefJson(relation.target)},
              {"source_object_id", relation.source_object_id},
              {"target_object_id", relation.target_object_id},
              {"relation_type", relation.relation_type},
              {"confidence", relation.confidence},
              {"description", relation.description},
              {"revision", relation.revision},
              {"derived", relation.derived},
              {"source_resolved_alias", relation.source_resolved_alias},
              {"target_resolved_alias", relation.target_resolved_alias}};
}

Json roomJson(const CanonicalRoom& room) {
  return Json{{"room_id", room.room_id},
              {"name", room.name},
              {"room_object_id",
               room.room_object_id ? Json(*room.room_object_id) : Json(nullptr)},
              {"attributes", room.attributes},
              {"object_ids", room.object_ids}};
}

Json mapStampJson(const MapStamp& stamp) {
  return Json{{"map_epoch",
               {{"high", stamp.map_epoch.high},
                {"low", stamp.map_epoch.low},
                {"id", runIdString(stamp.map_epoch)}}},
              {"map_revision", stamp.map_revision},
              {"integrated_through_ns", stamp.integrated_through_ns}};
}

Json surfaceStampJson(const SurfaceStamp& stamp) {
  return Json{{"map_epoch",
               {{"high", stamp.map_epoch.high},
                {"low", stamp.map_epoch.low},
                {"id", runIdString(stamp.map_epoch)}}},
              {"surface_revision", stamp.surface_revision},
              {"source_map_revision", stamp.source_map_revision}};
}

Json provenanceJson(const FrameProvenance& provenance) {
  return Json{
      {"run_id",
       {{"high", provenance.run_id.high},
        {"low", provenance.run_id.low},
        {"id", runIdString(provenance.run_id)}}},
      {"frame_id", provenance.frame_id},
      {"request_id", provenance.request_id},
      {"sensor_time_ns", provenance.sensor_time_ns},
      {"map_mode", provenance.map_mode == MapMode::kFrozen ? "frozen" : "online"},
      {"includes_current_frame", provenance.includes_current_frame},
      {"causality_verified", provenance.causality_verified},
      {"map", mapStampJson(provenance.map)},
      {"surface", surfaceStampJson(provenance.surface)}};
}

Json snapshotReferenceJson(const ObjectSnapshotRef& reference) {
  return Json{
      {"image_index", reference.image_index},
      {"source_frame_asset_id", reference.source_frame_asset_id},
      {"evidence_hash", reference.evidence_hash},
      {"bbox_xyxy", reference.bbox_xyxy},
      {"crop_xywh", reference.crop_xywh},
      {"crop_output_scale", reference.crop_output_scale},
      {"mask_source", reference.mask_source},
      {"mask_ref", reference.mask_ref},
      {"quality", reference.quality},
      {"quality_components", reference.quality_components},
      {"viewpoint_azimuth_rad", reference.viewpoint_azimuth_rad},
      {"viewpoint_elevation_rad", reference.viewpoint_elevation_rad},
      {"viewpoint_scale", reference.viewpoint_scale},
      {"time_ns", reference.time_ns},
      {"camera_id", reference.camera_id},
      {"provenance", provenanceJson(reference.provenance)}};
}

Json resolvedAssetJson(const ResolvedSnapshotAsset& asset) {
  return Json{{"source_frame_asset_id", asset.source_frame_asset_id},
              {"uri", asset.uri},
              {"source_path", asset.source_path},
              {"width", asset.width},
              {"height", asset.height},
              {"channels", asset.channels},
              {"encoding", asset.encoding},
              {"source_encoding", asset.source_encoding},
              {"encoded_bytes", asset.encoded_bytes}};
}

Json legacyAssetJson(const ObjectSnapshotImage& asset) {
  return Json{{"image_index", asset.image_index},
              {"uri", asset.uri},
              {"width", asset.width},
              {"height", asset.height},
              {"encoding", asset.encoding},
              {"time_ns", asset.time_ns},
              {"camera_id", asset.camera_id},
              {"source_path", asset.source_path}};
}

Json snapshotInspectionJson(const SnapshotInspection& inspection) {
  Json snapshots = Json::array();
  for (const SnapshotAssetInspection& snapshot : inspection.snapshots) {
    snapshots.push_back(
        Json{{"reference", snapshotReferenceJson(snapshot.reference)},
             {"available", snapshot.available},
             {"physical_asset",
              snapshot.physical_asset ? resolvedAssetJson(*snapshot.physical_asset)
                                      : Json(nullptr)},
             {"asset", snapshot.asset ? legacyAssetJson(*snapshot.asset)
                                       : Json(nullptr)}});
  }
  return Json{{"object", queryObjectJson(inspection.object)},
              {"snapshots", std::move(snapshots)},
              {"snapshot_set_hash", inspection.snapshot_set_hash}};
}

Json searchMatchesJson(const SearchObjectMatches& matches) {
  Json result = Json::array();
  for (const SearchObjectMatch& match : matches) {
    result.push_back(
        Json{{"object", queryObjectJson(match.object)},
             {"score", match.score},
             {"source", match.source == SearchMatchSource::kVector
                            ? "vector"
                            : "pinned_lexical_delta"},
             {"vector_document_hash", match.vector_document_hash}});
  }
  return result;
}

Json makeCallResult(const Json& call, std::string method, QueryStatus status,
                    std::string message, const QueryMetadata& metadata,
                    Json value = nullptr) {
  Json response{{"method", std::move(method)},
                {"success", status == QueryStatus::kOk},
                {"status", queryStatusName(status)},
                {"message", std::move(message)},
                {"metadata", queryMetadataJson(metadata)},
                {"result", std::move(value)}};
  if (call.is_object() && call.contains("id")) {
    response["id"] = call.at("id");
  }
  return response;
}

template <typename T>
Json makeCallResult(const Json& call, std::string method,
                    const QueryResult<T>& result, Json value) {
  return makeCallResult(call, std::move(method), result.status, result.message,
                        result.metadata,
                        result.ok() ? std::move(value) : Json(nullptr));
}

Json invalidCall(const Json& call, std::string method, std::string message,
                 const SceneReadToken& token) {
  return makeCallResult(call, std::move(method), QueryStatus::kInvalidArgument,
                        std::move(message), metadataFromToken(token));
}

bool readRequiredInteger(const Json& params, std::string_view key, int* value,
                         std::string* error) {
  const std::string name(key);
  const auto iterator = params.find(name);
  if (iterator == params.end() || !iterator->is_number_integer()) {
    *error = "params." + name + " must be an integer";
    return false;
  }
  try {
    const std::int64_t parsed = iterator->get<std::int64_t>();
    if (parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
      *error = "params." + name + " is out of range";
      return false;
    }
    *value = static_cast<int>(parsed);
    return true;
  } catch (const std::exception&) {
    *error = "params." + name + " is out of range";
    return false;
  }
}

bool readOptionalBoolean(const Json& params, std::string_view key, bool* value,
                         std::string* error) {
  const std::string name(key);
  const auto iterator = params.find(name);
  if (iterator == params.end()) {
    return true;
  }
  if (!iterator->is_boolean()) {
    *error = "params." + name + " must be a boolean";
    return false;
  }
  *value = iterator->get<bool>();
  return true;
}

bool readOptionalLimit(const Json& params, std::size_t default_value,
                       std::size_t* value, std::string* error) {
  *value = default_value;
  const auto iterator = params.find("limit");
  if (iterator == params.end()) {
    return true;
  }
  if (!iterator->is_number_unsigned() && !iterator->is_number_integer()) {
    *error = "params.limit must be a positive integer";
    return false;
  }
  try {
    const std::int64_t parsed = iterator->get<std::int64_t>();
    if (parsed <= 0 || parsed > 10000) {
      *error = "params.limit must be between 1 and 10000";
      return false;
    }
    *value = static_cast<std::size_t>(parsed);
    return true;
  } catch (const std::exception&) {
    *error = "params.limit is out of range";
    return false;
  }
}

Json dispatchCall(const Json& call, const LocalSceneQueryHandlers& handlers) {
  if (!call.is_object()) {
    return invalidCall(call, "", "call must be an object", handlers.token());
  }
  const auto method_iterator = call.find("method");
  if (method_iterator == call.end() || !method_iterator->is_string() ||
      method_iterator->get_ref<const std::string&>().empty()) {
    return invalidCall(call, "", "call.method must be a non-empty string",
                       handlers.token());
  }
  const std::string method = method_iterator->get<std::string>();
  const auto params_iterator = call.find("params");
  if (params_iterator != call.end() && !params_iterator->is_object()) {
    return invalidCall(call, method, "call.params must be an object",
                       handlers.token());
  }
  const Json empty_params = Json::object();
  const Json& params = params_iterator == call.end() ? empty_params
                                                      : *params_iterator;
  std::string error;

  if (method == "get_object") {
    int object_id = -1;
    if (!readRequiredInteger(params, "object_id", &object_id, &error)) {
      return invalidCall(call, method, std::move(error), handlers.token());
    }
    const auto result = handlers.getObject(object_id);
    return makeCallResult(call, method, result, queryObjectJson(result.value));
  }

  if (method == "get_objects_near") {
    ObjectsNearRequest request;
    const auto center = params.find("center_world");
    if (center == params.end() || !center->is_array() || center->size() != 3U ||
        !(*center)[0].is_number() || !(*center)[1].is_number() ||
        !(*center)[2].is_number()) {
      return invalidCall(call, method,
                         "params.center_world must be an array of 3 numbers",
                         handlers.token());
    }
    try {
      request.center_world = Eigen::Vector3f(
          (*center)[0].get<float>(), (*center)[1].get<float>(),
          (*center)[2].get<float>());
    } catch (const std::exception&) {
      return invalidCall(call, method,
                         "params.center_world contains an out-of-range number",
                         handlers.token());
    }
    const auto radius = params.find("radius_m");
    if (radius != params.end()) {
      if (!radius->is_number()) {
        return invalidCall(call, method, "params.radius_m must be a number",
                           handlers.token());
      }
      request.radius_m = radius->get<float>();
    }
    if (!readOptionalLimit(params, request.limit, &request.limit, &error) ||
        !readOptionalBoolean(params, "include_inactive",
                             &request.include_inactive, &error) ||
        !readOptionalBoolean(params, "include_unpublishable",
                             &request.include_unpublishable, &error)) {
      return invalidCall(call, method, std::move(error), handlers.token());
    }
    const auto result = handlers.getObjectsNear(request);
    Json objects = Json::array();
    for (const QueryObjectView& object : result.value) {
      objects.push_back(queryObjectJson(object));
    }
    return makeCallResult(call, method, result, std::move(objects));
  }

  if (method == "rooms") {
    const auto result = handlers.rooms();
    Json rooms = Json::array();
    for (const CanonicalRoom& room : result.value) {
      rooms.push_back(roomJson(room));
    }
    return makeCallResult(call, method, result, std::move(rooms));
  }

  if (method == "relations") {
    RelationRequest request;
    const auto object = params.find("object_id");
    if (object != params.end()) {
      int object_id = -1;
      if (!readRequiredInteger(params, "object_id", &object_id, &error)) {
        return invalidCall(call, method, std::move(error), handlers.token());
      }
      request.object_id = object_id;
    }
    const auto type = params.find("relation_type");
    if (type != params.end()) {
      if (!type->is_string() || type->get_ref<const std::string&>().empty()) {
        return invalidCall(
            call, method,
            "params.relation_type must be a non-empty string",
            handlers.token());
      }
      request.relation_type = type->get<std::string>();
    }
    const auto direction = params.find("direction");
    if (direction != params.end()) {
      if (!direction->is_string()) {
        return invalidCall(call, method,
                           "params.direction must be either, outgoing, or incoming",
                           handlers.token());
      }
      const std::string value = direction->get<std::string>();
      if (value == "either") {
        request.direction = RelationDirection::kEither;
      } else if (value == "outgoing") {
        request.direction = RelationDirection::kOutgoing;
      } else if (value == "incoming") {
        request.direction = RelationDirection::kIncoming;
      } else {
        return invalidCall(call, method,
                           "params.direction must be either, outgoing, or incoming",
                           handlers.token());
      }
    }
    const auto result = handlers.relations(request);
    Json relations = Json::array();
    for (const CanonicalRelation& relation : result.value) {
      relations.push_back(relationJson(relation));
    }
    return makeCallResult(call, method, result, std::move(relations));
  }

  if (method == "inspect_snapshot") {
    int object_id = -1;
    if (!readRequiredInteger(params, "object_id", &object_id, &error)) {
      return invalidCall(call, method, std::move(error), handlers.token());
    }
    std::optional<int> image_index;
    if (params.contains("image_index")) {
      int parsed = -1;
      if (!readRequiredInteger(params, "image_index", &parsed, &error)) {
        return invalidCall(call, method, std::move(error), handlers.token());
      }
      image_index = parsed;
    }
    const auto result = handlers.inspectSnapshot(object_id, image_index);
    return makeCallResult(call, method, result,
                          snapshotInspectionJson(result.value));
  }

  if (method == "search_objects") {
    SearchRequest request;
    const auto query = params.find("query");
    if (query == params.end() || !query->is_string() ||
        query->get_ref<const std::string&>().empty()) {
      return invalidCall(call, method,
                         "params.query must be a non-empty string",
                         handlers.token());
    }
    request.query = query->get<std::string>();
    if (!readOptionalLimit(params, request.limit, &request.limit, &error) ||
        !readOptionalBoolean(params, "include_inactive",
                             &request.include_inactive, &error) ||
        !readOptionalBoolean(params, "include_unpublishable",
                             &request.include_unpublishable, &error)) {
      return invalidCall(call, method, std::move(error), handlers.token());
    }
    const auto room = params.find("room_id");
    if (room != params.end()) {
      if (!room->is_string() || room->get_ref<const std::string&>().empty()) {
        return invalidCall(call, method,
                           "params.room_id must be a non-empty string",
                           handlers.token());
      }
      request.room_id = room->get<std::string>();
    }
    const auto result = handlers.searchObjects(request);
    return makeCallResult(call, method, result,
                          searchMatchesJson(result.value));
  }

  return invalidCall(call, method, "unsupported method: " + method,
                     handlers.token());
}

SceneQueryJsonResponse envelopeFailure(std::string error) {
  SceneQueryJsonResponse response;
  response.error = std::move(error);
  response.error_code = "invalid_request";
  return response;
}

const char* readSessionStatusName(SceneReadSessionStatus status) {
  switch (status) {
    case SceneReadSessionStatus::kOk:
      return "ok";
    case SceneReadSessionStatus::kInvalidArgument:
      return "invalid_session";
    case SceneReadSessionStatus::kNotFound:
      return "unknown_session";
    case SceneReadSessionStatus::kExpired:
      return "expired_session";
    case SceneReadSessionStatus::kRevisionMismatch:
      return "stale_session";
    case SceneReadSessionStatus::kCapacityExceeded:
      return "session_capacity_exhausted";
  }
  return "session_error";
}

SceneQueryJsonResponse sessionFailure(
    const SceneReadSessionResult& session) {
  SceneQueryJsonResponse response;
  response.error = session.message;
  response.error_code = readSessionStatusName(session.status);
  response.response_json =
      Json{{"schema_version", kResponseSchema},
           {"success", false},
           {"status", response.error_code},
           {"error", response.error},
           {"session_id", session.session_id.empty()
                              ? Json(nullptr)
                              : Json(session.session_id)}}
          .dump();
  return response;
}

std::optional<SceneRevision> readExpectedRevision(
    const Json& request, std::string* error) {
  const auto expected = request.find("expected_scene_revision");
  if (expected == request.end()) {
    return std::nullopt;
  }
  try {
    if (expected->is_number_unsigned()) {
      return expected->get<SceneRevision>();
    }
    if (expected->is_number_integer()) {
      const std::int64_t value = expected->get<std::int64_t>();
      if (value >= 0) {
        return static_cast<SceneRevision>(value);
      }
    }
  } catch (const std::exception&) {
  }
  if (error) {
    *error = "expected_scene_revision must be a non-negative integer";
  }
  return std::nullopt;
}

}  // namespace

SceneQueryJsonResponse SceneQueryJsonAdapter::dispatch(
    std::string_view request_json) const {
  if (!gateway_) {
    return envelopeFailure("query gateway is unavailable");
  }

  Json request;
  try {
    request = Json::parse(request_json);
  } catch (const std::exception& exception) {
    return envelopeFailure(std::string("invalid request JSON: ") +
                           exception.what());
  }
  if (!request.is_object()) {
    return envelopeFailure("request must be a JSON object");
  }
  const auto schema = request.find("schema_version");
  if (schema != request.end() &&
      (!schema->is_string() || schema->get<std::string>() != kRequestSchema)) {
    return envelopeFailure("schema_version must be roomie.query_scene.v1");
  }

  std::string operation;
  const auto operation_value = request.find("operation");
  const auto method_value = request.find("method");
  if (operation_value != request.end() && method_value != request.end()) {
    return envelopeFailure("provide only one of operation or method");
  }
  const auto selected_operation =
      operation_value != request.end() ? operation_value : method_value;
  if (selected_operation != request.end()) {
    if (!selected_operation->is_string() ||
        selected_operation->get_ref<const std::string&>().empty()) {
      return envelopeFailure("operation must be a non-empty string");
    }
    operation = selected_operation->get<std::string>();
    if (operation != "begin_session" && operation != "create_session") {
      return envelopeFailure("unsupported operation: " + operation);
    }
  }

  std::chrono::milliseconds ttl = std::chrono::seconds(30);
  const auto ttl_value = request.find("ttl_ms");
  if (ttl_value != request.end()) {
    if (!ttl_value->is_number_integer()) {
      return envelopeFailure("ttl_ms must be an integer");
    }
    try {
      const std::int64_t parsed = ttl_value->get<std::int64_t>();
      if (parsed <= 0 || parsed > kMaximumTtlMs) {
        return envelopeFailure("ttl_ms must be between 1 and 300000");
      }
      ttl = std::chrono::milliseconds(parsed);
    } catch (const std::exception&) {
      return envelopeFailure("ttl_ms is out of range");
    }
  }

  const bool begin_session = operation == "begin_session" ||
                             operation == "create_session";
  if (begin_session) {
    if (request.find("session_id") != request.end()) {
      return envelopeFailure("begin_session must not include session_id");
    }
    const auto begin_calls = request.find("calls");
    if (begin_calls != request.end() &&
        (!begin_calls->is_array() || !begin_calls->empty())) {
      return envelopeFailure(
          "begin_session calls, when present, must be an empty array");
    }
    try {
      const SceneReadSessionResult session =
          gateway_->beginReadSession(ttl);
      if (!session.ok()) {
        return sessionFailure(session);
      }
      SceneQueryJsonResponse response;
      response.success = true;
      response.scene_revision = session.token.sceneRevision();
      response.durable_scene_revision =
          session.token.durableSceneRevision();
      response.index_generation = session.token.indexGeneration();
      response.response_json =
          Json{{"schema_version", kResponseSchema},
               {"success", true},
               {"operation", "begin_session"},
               {"session_id", session.session_id},
               {"session",
                {{"session_id", session.session_id},
                 {"ttl_ms", ttl.count()},
                 {"scene_revision", session.token.sceneRevision()},
                 {"durable_scene_revision",
                  session.token.durableSceneRevision()},
                 {"index_generation", session.token.indexGeneration()}}},
               {"read_token", readTokenJson(session.token)},
               {"calls", Json::array()}}
              .dump();
      return response;
    } catch (const std::exception& exception) {
      return envelopeFailure(std::string("query dispatch failed: ") +
                             exception.what());
    }
  }

  const auto calls = request.find("calls");
  if (calls == request.end() || !calls->is_array() || calls->empty()) {
    return envelopeFailure("calls must be a non-empty array");
  }
  if (calls->size() > kMaximumCalls) {
    return envelopeFailure("calls exceeds the maximum batch size of 128");
  }

  std::optional<std::string> session_id;
  const auto session_value = request.find("session_id");
  if (session_value != request.end()) {
    if (!session_value->is_string() ||
        session_value->get_ref<const std::string&>().empty()) {
      return envelopeFailure("session_id must be a non-empty string");
    }
    session_id = session_value->get<std::string>();
    if (ttl_value != request.end()) {
      return envelopeFailure(
          "ttl_ms cannot change an existing read session");
    }
  }
  std::string expected_revision_error;
  const bool has_expected_revision =
      request.find("expected_scene_revision") != request.end();
  const std::optional<SceneRevision> expected_scene_revision =
      readExpectedRevision(request, &expected_revision_error);
  if (has_expected_revision && !expected_scene_revision) {
    return envelopeFailure(std::move(expected_revision_error));
  }
  if (expected_scene_revision && !session_id) {
    return envelopeFailure(
        "expected_scene_revision requires session_id");
  }

  try {
    SceneReadToken token;
    if (session_id) {
      const SceneReadSessionResult session = gateway_->resumeReadSession(
          *session_id, expected_scene_revision);
      if (!session.ok()) {
        return sessionFailure(session);
      }
      token = session.token;
    } else {
      // Compatibility mode: one request remains one coherent pinned batch.
      token = gateway_->pin(ttl);
    }
    LocalSceneQueryHandlers handlers(*gateway_, std::move(token));
    Json call_results = Json::array();
    for (const Json& call : *calls) {
      try {
        call_results.push_back(dispatchCall(call, handlers));
      } catch (const std::exception& exception) {
        std::string method;
        if (call.is_object()) {
          const auto candidate = call.find("method");
          if (candidate != call.end() && candidate->is_string()) {
            method = candidate->get<std::string>();
          }
        }
        call_results.push_back(invalidCall(
            call, std::move(method),
            std::string("invalid call parameters: ") + exception.what(),
            handlers.token()));
      }
    }

    SceneQueryJsonResponse response;
    response.success = true;
    response.scene_revision = handlers.token().sceneRevision();
    response.durable_scene_revision =
        handlers.token().durableSceneRevision();
    response.index_generation = handlers.token().indexGeneration();
    response.call_count = calls->size();
    Json response_root =
        Json{{"schema_version", kResponseSchema},
             {"success", true},
             {"read_token", readTokenJson(handlers.token())},
             {"calls", std::move(call_results)}};
    if (session_id) {
      response_root["session_id"] = *session_id;
    }
    response.response_json = response_root.dump();
    return response;
  } catch (const std::exception& exception) {
    return envelopeFailure(std::string("query dispatch failed: ") +
                           exception.what());
  }
}

}  // namespace roomie
