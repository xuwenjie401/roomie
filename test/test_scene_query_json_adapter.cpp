#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "roomie/query/scene_query_json_adapter.hpp"

namespace roomie {
namespace {

using Json = nlohmann::json;

SceneSnapshot makeQuerySnapshot(SceneRevision revision) {
  auto object = std::make_shared<SceneObject>();

  auto identity = std::make_shared<IdentityComponent>();
  identity->revision = 2;
  identity->object_id = 7;
  object->identity = identity;

  auto lifecycle = std::make_shared<LifecycleComponent>();
  lifecycle->revision = 3;
  lifecycle->track_state = InstanceTrackState::kStable;
  lifecycle->active = true;
  lifecycle->publishable = true;
  lifecycle->existence_log_odds = 2.0f;
  lifecycle->last_presence_evidence_ns = 1234;
  lifecycle->last_presence_evidence_reason = "matched_detection";
  object->lifecycle = lifecycle;

  SurfaceStamp surface;
  surface.map_epoch = RunId{11, 22};
  surface.surface_revision = 9;
  surface.source_map_revision = 8;

  auto geometry = std::make_shared<GeometryComponent>();
  geometry->revision = 4;
  geometry->obb_revision = 5;
  geometry->evaluated_obb_revision = 5;
  geometry->center_world = Eigen::Vector3f(1.0f, 2.0f, 0.5f);
  geometry->size_m = Eigen::Vector3f(0.6f, 0.7f, 0.8f);
  geometry->yaw_rad = 0.25f;
  geometry->status = InstanceGeometryStatus::kGood;
  geometry->evaluated_surface = surface;
  object->geometry = geometry;

  auto semantic = std::make_shared<SemanticComponent>();
  semantic->revision = 6;
  semantic->semantic_id = 41;
  semantic->label = "walnut chair";
  semantic->confidence = 0.94f;
  object->semantic = semantic;

  auto annotation = std::make_shared<AnnotationComponent>();
  annotation->revision = 1;
  annotation->attributes["room_id"] = "reading-room";
  object->annotation = annotation;

  ObjectSnapshotRef reference;
  reference.image_index = 3;
  reference.evidence_hash = "evidence-3";
  reference.camera_id = "head_rgb";
  reference.time_ns = 123456;
  reference.provenance.run_id = RunId{101, 202};
  reference.provenance.frame_id = 303;
  reference.provenance.request_id = 404;
  reference.provenance.sensor_time_ns = 123456;
  reference.provenance.includes_current_frame = true;
  reference.provenance.causality_verified = true;
  reference.provenance.map.map_epoch = RunId{101, 202};
  reference.provenance.map.map_revision = 12;
  reference.provenance.map.integrated_through_ns = 123456;
  reference.provenance.surface.map_epoch = RunId{101, 202};
  reference.provenance.surface.surface_revision = 13;
  reference.provenance.surface.source_map_revision = 12;

  auto artifact = std::make_shared<ArtifactComponent>();
  artifact->revision = 8;
  artifact->appearance_revision = 1;
  artifact->description = "an old carved walnut chair";
  artifact->description_input_hash = "dam-input-7";
  artifact->snapshots.push_back(reference);
  artifact->snapshot_set_hash = "snapshot-set-7";
  object->artifact = artifact;

  auto state = std::make_shared<SceneState>();
  state->latest_scene_revision = revision;
  state->durable_scene_revision = revision - 1;
  state->next_object_id = 8;
  state->latest_surface = surface;
  auto objects = std::make_shared<SceneObjectTable>();
  objects->emplace(7, object);
  state->objects = objects;

  RoomNode room;
  room.room_id = 10;
  room.revision = 2;
  room.label = "Reading room";
  room.attributes["room_id"] = "reading-room";

  ObjectRelation relation;
  relation.relation_type = "room_contains_object";
  relation.confidence = 0.91f;
  relation.revision = 5;
  setRelationEndpoints(&relation,
                       SceneEntityRef{SceneEntityType::kRoom, 10},
                       SceneEntityRef{SceneEntityType::kObject, 7});
  ObjectRelation furniture_relation;
  furniture_relation.relation_type = "room_contains_furniture";
  furniture_relation.confidence = 1.0f;
  furniture_relation.revision = 6;
  furniture_relation.derived = true;
  setRelationEndpoints(&furniture_relation,
                       SceneEntityRef{SceneEntityType::kRoom, 10},
                       SceneEntityRef{SceneEntityType::kFurniture, 7});

  ObjectSnapshotImage image;
  image.image_index = 3;
  image.uri = "asset://snapshot/3";
  image.width = 640;
  image.height = 480;
  image.encoding = "png";
  image.camera_id = "head_rgb";
  image.time_ns = 123456;

  auto graph = std::make_shared<SceneGraphMetadata>();
  graph->rooms.push_back(room);
  graph->furniture.push_back(FurnitureRole{7, 2, "chair"});
  graph->relations.push_back(relation);
  graph->relations.push_back(furniture_relation);
  graph->snapshot_images.push_back(image);
  state->graph = graph;
  return SceneSnapshot{state};
}

TEST(SceneQueryJsonAdapter, BatchPinsOnceAndEveryCallUsesOneRevision) {
  int supplier_calls = 0;
  SceneQueryGateway gateway([&supplier_calls]() {
    ++supplier_calls;
    return makeQuerySnapshot(40 + supplier_calls);
  });
  SceneQueryJsonAdapter adapter(gateway);

  const Json request = {
      {"schema_version", "roomie.query_scene.v1"},
      {"calls",
       Json::array(
           {{{"id", "one"},
             {"method", "get_object"},
             {"params", {{"object_id", 7}}}},
            {{"id", "near"},
             {"method", "get_objects_near"},
             {"params",
              {{"center_world", Json::array({1.0, 2.0, 0.5})},
               {"radius_m", 2.0}}}},
            {{"id", "room"}, {"method", "rooms"}},
            {{"id", "furniture"}, {"method", "furniture"}},
            {{"id", "edges"},
             {"method", "relations"},
             {"params", {{"object_id", 7}, {"direction", "incoming"}}}},
            {{"id", "image"},
             {"method", "inspect_snapshot"},
             {"params", {{"object_id", 7}, {"image_index", 3}}}},
            {{"id", "search"},
             {"method", "search_objects"},
             {"params", {{"query", "walnut chair"}, {"limit", 3}}}}})}};

  const SceneQueryJsonResponse response = adapter.dispatch(request.dump());
  ASSERT_TRUE(response.success) << response.error;
  EXPECT_EQ(response.scene_revision, 41U);
  EXPECT_EQ(response.durable_scene_revision, 40U);
  EXPECT_EQ(response.call_count, 7U);
  EXPECT_EQ(supplier_calls, 1);
  const Json result = Json::parse(response.response_json);
  EXPECT_EQ(result.at("schema_version"),
            "roomie.query_scene.response.v1");
  EXPECT_EQ(result.at("read_token").at("scene_revision"), 41);
  ASSERT_EQ(result.at("calls").size(), 7U);
  for (const Json& call : result.at("calls")) {
    EXPECT_TRUE(call.at("success")) << call.dump();
    EXPECT_EQ(call.at("status"), "ok");
    EXPECT_EQ(call.at("metadata").at("scene_revision"), 41);
    EXPECT_EQ(call.at("metadata").at("durable_scene_revision"), 40);
  }
  const Json& object_result = result.at("calls").at(0).at("result");
  EXPECT_EQ(object_result.at("presence_state"), "active");
  EXPECT_GT(object_result.at("existence_probability").get<float>(), 0.8f);
  EXPECT_EQ(object_result.at("last_presence_evidence_reason"),
            "matched_detection");
  ASSERT_TRUE(object_result.at("furniture_role").is_object());
  EXPECT_EQ(object_result.at("furniture_role").at("object_id"), 7);
  const Json& furniture_result = result.at("calls").at(3).at("result");
  ASSERT_EQ(furniture_result.size(), 1U);
  EXPECT_EQ(furniture_result.at(0).at("role").at("object_id"), 7);
  EXPECT_EQ(furniture_result.at(0).at("object").at("object_id"), 7);
}

TEST(SceneQueryJsonAdapter, ParameterErrorsArePerCallAndEnvelopeErrorsDoNotPin) {
  int supplier_calls = 0;
  SceneSnapshot snapshot = makeQuerySnapshot(17);
  SceneQueryGateway gateway([&]() {
    ++supplier_calls;
    return snapshot;
  });
  SceneQueryJsonAdapter adapter(gateway);

  const Json request = {
      {"calls",
       Json::array(
           {{{"id", 1},
             {"method", "get_object"},
             {"params", {{"object_id", "seven"}}}},
            {{"id", 2},
             {"method", "relations"},
             {"params", {{"direction", "sideways"}}}},
            {{"id", 3}, {"method", "does_not_exist"}}})}};
  const SceneQueryJsonResponse response = adapter.dispatch(request.dump());
  ASSERT_TRUE(response.success) << response.error;
  EXPECT_EQ(supplier_calls, 1);
  const Json calls = Json::parse(response.response_json).at("calls");
  ASSERT_EQ(calls.size(), 3U);
  for (const Json& call : calls) {
    EXPECT_FALSE(call.at("success"));
    EXPECT_EQ(call.at("status"), "invalid_argument");
    EXPECT_TRUE(call.at("result").is_null());
    EXPECT_EQ(call.at("metadata").at("scene_revision"), 17);
  }

  const SceneQueryJsonResponse malformed =
      adapter.dispatch(R"({"calls":{"method":"rooms"}})");
  EXPECT_FALSE(malformed.success);
  EXPECT_TRUE(malformed.response_json.empty());
  EXPECT_NE(malformed.error.find("calls"), std::string::npos);
  EXPECT_EQ(supplier_calls, 1) << "invalid envelopes must not acquire a token";
}

TEST(SceneQueryJsonAdapter, ResultSchemaIncludesRevisionsFreshnessAndProvenance) {
  SceneSnapshot snapshot = makeQuerySnapshot(23);
  SceneQueryGateway gateway([&snapshot]() { return snapshot; });
  SceneQueryJsonAdapter adapter(gateway);

  const SceneQueryJsonResponse response = adapter.dispatch(
      R"({"calls":[{"method":"inspect_snapshot","params":{"object_id":7}}]})");
  ASSERT_TRUE(response.success) << response.error;
  const Json root = Json::parse(response.response_json);
  const Json& call = root.at("calls").at(0);
  ASSERT_TRUE(call.at("success"));
  const Json& metadata = call.at("metadata");
  EXPECT_TRUE(metadata.contains("scene_revision"));
  EXPECT_TRUE(metadata.contains("durable_scene_revision"));
  EXPECT_TRUE(metadata.contains("index_generation"));
  EXPECT_TRUE(metadata.contains("freshness"));
  EXPECT_TRUE(metadata.at("freshness").contains("geometry"));
  EXPECT_TRUE(metadata.contains("artifact_pending"));

  const Json& object = call.at("result").at("object");
  EXPECT_EQ(object.at("object_id"), 7);
  EXPECT_EQ(object.at("revisions").at("identity"), 2);
  EXPECT_EQ(object.at("revisions").at("geometry"), 4);
  EXPECT_EQ(object.at("revisions").at("semantic"), 6);
  EXPECT_TRUE(object.at("freshness").contains("description"));

  const Json& snapshot_json = call.at("result").at("snapshots").at(0);
  EXPECT_TRUE(snapshot_json.at("available"));
  const Json& provenance = snapshot_json.at("reference").at("provenance");
  EXPECT_EQ(provenance.at("frame_id"), 303);
  EXPECT_EQ(provenance.at("request_id"), 404);
  EXPECT_TRUE(provenance.at("includes_current_frame"));
  EXPECT_TRUE(provenance.at("causality_verified"));
  EXPECT_EQ(provenance.at("map").at("map_revision"), 12);
  EXPECT_EQ(provenance.at("surface").at("surface_revision"), 13);
}

TEST(SceneQueryJsonAdapter,
     OpaqueSessionReusesRevisionAndReportsStaleAndExpiredErrors) {
  int supplier_calls = 0;
  auto steady = SceneReadToken::Clock::time_point(std::chrono::seconds(1));
  SceneQueryGateway gateway(
      [&supplier_calls]() {
        ++supplier_calls;
        return makeQuerySnapshot(100 + supplier_calls);
      },
      std::shared_ptr<const SearchProvider>{}, std::chrono::seconds(30),
      [&steady]() { return steady; }, []() { return TimeNanoseconds{777}; });
  SceneQueryJsonAdapter adapter(gateway);

  const SceneQueryJsonResponse begin = adapter.dispatch(
      R"({"schema_version":"roomie.query_scene.v1","operation":"begin_session","ttl_ms":20})");
  ASSERT_TRUE(begin.success) << begin.error;
  EXPECT_EQ(begin.call_count, 0U);
  EXPECT_EQ(begin.scene_revision, 101U);
  EXPECT_EQ(supplier_calls, 1);
  const Json begin_json = Json::parse(begin.response_json);
  ASSERT_TRUE(begin_json.at("success"));
  const std::string session_id = begin_json.at("session_id");
  EXPECT_FALSE(session_id.empty());
  EXPECT_EQ(begin_json.at("session").at("scene_revision"), 101);
  EXPECT_EQ(begin_json.at("session").at("ttl_ms"), 20);

  const Json call_request = {
      {"schema_version", "roomie.query_scene.v1"},
      {"session_id", session_id},
      {"expected_scene_revision", 101},
      {"calls",
       Json::array({{{"method", "get_object"},
                     {"params", {{"object_id", 7}}}}})}};
  const SceneQueryJsonResponse first_call =
      adapter.dispatch(call_request.dump());
  ASSERT_TRUE(first_call.success) << first_call.error;
  EXPECT_EQ(supplier_calls, 1)
      << "resuming a session must not repin the supplier";
  const Json first_json = Json::parse(first_call.response_json);
  EXPECT_EQ(first_json.at("session_id"), session_id);
  EXPECT_EQ(first_json.at("read_token").at("scene_revision"), 101);
  EXPECT_EQ(first_json.at("calls").at(0).at("metadata").at("scene_revision"),
            101);

  Json stale_request = call_request;
  stale_request["expected_scene_revision"] = 999;
  const SceneQueryJsonResponse stale =
      adapter.dispatch(stale_request.dump());
  EXPECT_FALSE(stale.success);
  EXPECT_EQ(stale.error_code, "stale_session");
  const Json stale_json = Json::parse(stale.response_json);
  EXPECT_EQ(stale_json.at("status"), "stale_session");
  EXPECT_EQ(stale_json.at("session_id"), session_id);
  EXPECT_EQ(supplier_calls, 1);

  steady += std::chrono::milliseconds(20);
  const SceneQueryJsonResponse expired =
      adapter.dispatch(call_request.dump());
  EXPECT_FALSE(expired.success);
  EXPECT_EQ(expired.error_code, "expired_session");
  const Json expired_json = Json::parse(expired.response_json);
  EXPECT_EQ(expired_json.at("status"), "expired_session");
  EXPECT_NE(expired.error.find("expired"), std::string::npos);
}

TEST(SceneQueryJsonAdapter, UnknownSessionIsAnExplicitStructuredError) {
  SceneSnapshot snapshot = makeQuerySnapshot(8);
  SceneQueryGateway gateway([&snapshot]() { return snapshot; });
  SceneQueryJsonAdapter adapter(gateway);
  const SceneQueryJsonResponse response = adapter.dispatch(
      R"({"session_id":"not-issued","calls":[{"method":"rooms"}]})");
  EXPECT_FALSE(response.success);
  EXPECT_EQ(response.error_code, "unknown_session");
  ASSERT_FALSE(response.response_json.empty());
  EXPECT_EQ(Json::parse(response.response_json).at("status"),
            "unknown_session");
}

}  // namespace
}  // namespace roomie
