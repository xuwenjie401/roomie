#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <optional>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "roomie/query/scene_mutation_json_adapter.hpp"
#include "roomie/scene/scene_reducer.hpp"

namespace roomie {
namespace {

using Json = nlohmann::json;

ObjectNode mutationNode(SceneObjectId object_id, std::string label) {
  ObjectNode node;
  node.object_id = object_id;
  node.semantic_id = object_id + 100;
  node.label = std::move(label);
  node.center_world = Eigen::Vector3f(1.0f, 1.0f, 0.5f);
  node.size_m = Eigen::Vector3f(0.5f, 0.6f, 0.7f);
  node.confidence = 0.9f;
  node.active = true;
  node.publishable = true;
  return node;
}

SceneMutationSubmitResult submittedResult(const SceneApplyResult& applied) {
  SceneMutationSubmitResult result;
  result.latest_scene_revision = applied.revision;
  result.durable_scene_revision = applied.snapshot.durableRevision();
  result.message = applied.reason;
  if (!applied.accepted()) {
    result.status = applied.reason.find("stale") != std::string::npos
                        ? SceneMutationStatus::kStale
                        : SceneMutationStatus::kRejected;
    return result;
  }
  if (!applied.committedRevision()) {
    result.status = SceneMutationStatus::kNoOp;
    return result;
  }
  result.status = SceneMutationStatus::kCommitted;
  result.committed_scene_revision = applied.revision;
  for (const SceneEvent& event : applied.events) {
    if (const auto* annotation =
            std::get_if<HumanAnnotationCommitted>(&event)) {
      result.component_revision = annotation->annotation_revision;
    }
  }
  return result;
}

TEST(SceneMutationJsonAdapter,
     ObjectPatchParsesDependenciesAndDerivedMembershipAssertion) {
  std::optional<ApplyHumanAnnotationCommand> captured;
  std::chrono::milliseconds captured_timeout{0};
  SceneMutationJsonAdapter adapter(
      [&](ApplyHumanAnnotationCommand command,
          std::chrono::milliseconds timeout) {
        captured = command;
        captured_timeout = timeout;
        SceneMutationSubmitResult result;
        result.status = SceneMutationStatus::kCommitted;
        result.latest_scene_revision = 43;
        result.durable_scene_revision = 43;
        result.committed_scene_revision = 43;
        result.component_revision = 8;
        return result;
      });

  const Json request = {
      {"schema_version", "roomie.mutate_scene.v1"},
      {"operation", "apply_human_annotation"},
      {"request_id", "edit-42"},
      {"base_scene_revision", 42},
      {"object_id", 7},
      {"dependencies",
       {{"identity_revision", 3}, {"annotation_revision", 6}}},
      {"patch",
       {{"name", "Ada's chair"},
        {"semantic_id", 91},
        {"label", "reading chair"},
        {"description", "human verified walnut chair"},
        {"attributes", {{"owner", "library"}, {"tag", "favorite"}}},
        {"room_memberships", Json::array({10, 12})}}},
      {"timeout_ms", 1200}};

  const SceneMutationJsonResponse response = adapter.dispatch(request.dump());
  ASSERT_TRUE(response.success) << response.error;
  ASSERT_TRUE(captured.has_value());
  ASSERT_TRUE(captured->expected_scene_revision.has_value());
  EXPECT_EQ(*captured->expected_scene_revision, 42U);
  EXPECT_EQ(captured->target,
            (SceneEntityRef{SceneEntityType::kObject, 7}));
  EXPECT_EQ(captured->object_id, 7);
  EXPECT_EQ(captured->expected_identity_revision, 3U);
  EXPECT_EQ(captured->expected_annotation_revision, 6U);
  ASSERT_TRUE(captured->patch.name.has_value());
  EXPECT_EQ(*captured->patch.name, "Ada's chair");
  ASSERT_TRUE(captured->patch.semantic_id.has_value());
  EXPECT_EQ(*captured->patch.semantic_id, 91);
  EXPECT_EQ(*captured->patch.label, "reading chair");
  EXPECT_EQ(*captured->patch.description,
            "human verified walnut chair");
  EXPECT_EQ(captured->patch.attributes.at("owner"), "library");
  ASSERT_TRUE(captured->expected_room_memberships.has_value());
  EXPECT_EQ(*captured->expected_room_memberships,
            (std::vector<int>{10, 12}));
  EXPECT_EQ(captured_timeout, std::chrono::milliseconds(1200));

  const Json body = Json::parse(response.response_json);
  EXPECT_TRUE(body.at("success"));
  EXPECT_EQ(body.at("status"), "committed");
  EXPECT_EQ(body.at("request_id"), "edit-42");
  EXPECT_EQ(body.at("base_scene_revision"), 42);
  EXPECT_EQ(body.at("committed_scene_revision"), 43);
  EXPECT_EQ(body.at("durable_scene_revision"), 43);
  EXPECT_EQ(body.at("component_revision"), 8);
  EXPECT_TRUE(body.at("room_memberships_are_derived"));
}

TEST(SceneMutationJsonAdapter,
     RoomGeometryCreatesOnlyReducerDerivedContainment) {
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(mutationNode(1, "chair"));
  load.graph.next_object_id = 2;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  SceneMutationJsonAdapter adapter(
      [&](ApplyHumanAnnotationCommand command, std::chrono::milliseconds) {
        return submittedResult(reducer.apply(SceneCommand{command}));
      });
  const Json request = {
      {"schema_version", "roomie.mutate_scene.v1"},
      {"operation", "apply_human_annotation"},
      {"base_scene_revision", loaded.revision},
      {"target", {{"type", "room"}, {"id", 10}}},
      {"patch",
       {{"label", "study"},
        {"color", "#445566"},
        {"min_xy", Json::array({0.0, 0.0})},
        {"max_xy", Json::array({2.0, 2.0})},
        {"height_m", 3.0},
        {"attributes", {{"floor", "ground"}}},
        // For a room target these are the expected contained object ids after
        // the reducer derives containment from the proposed bounds.
        {"room_memberships", Json::array({1})}}}};

  const SceneMutationJsonResponse response = adapter.dispatch(request.dump());
  ASSERT_TRUE(response.success) << response.error;
  const Json body = Json::parse(response.response_json);
  EXPECT_EQ(body.at("status"), "committed");
  const SceneSnapshot snapshot = reducer.snapshot();
  ASSERT_EQ(snapshot.graphMetadata().rooms.size(), 1U);
  ASSERT_EQ(snapshot.graphMetadata().relations.size(), 1U);
  const ObjectRelation& relation = snapshot.graphMetadata().relations.front();
  EXPECT_TRUE(relation.derived);
  EXPECT_EQ(relation.relation_type, "room_contains_object");
  EXPECT_EQ(relationSource(relation),
            (SceneEntityRef{SceneEntityType::kRoom, 10}));
  EXPECT_EQ(relationTarget(relation),
            (SceneEntityRef{SceneEntityType::kObject, 1}));
}

TEST(SceneMutationJsonAdapter,
     SceneCasIsCheckedByReducerAfterAConcurrentCommit) {
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(mutationNode(7, "chair"));
  load.graph.next_object_id = 8;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision());

  bool advanced = false;
  SceneMutationJsonAdapter adapter(
      [&](ApplyHumanAnnotationCommand command, std::chrono::milliseconds) {
        if (!advanced) {
          ApplyHumanAnnotationCommand concurrent;
          concurrent.object_id = 7;
          concurrent.patch.attributes["writer"] = "other-client";
          const SceneApplyResult committed =
              reducer.apply(SceneCommand{concurrent});
          EXPECT_TRUE(committed.committedRevision()) << committed.reason;
          advanced = true;
        }
        return submittedResult(reducer.apply(SceneCommand{command}));
      });

  const Json request = {
      {"schema_version", "roomie.mutate_scene.v1"},
      {"operation", "apply_human_annotation"},
      {"base_scene_revision", loaded.revision},
      {"object_id", 7},
      {"patch", {{"label", "my edit"}}}};
  const SceneMutationJsonResponse response = adapter.dispatch(request.dump());
  EXPECT_FALSE(response.success);
  const Json body = Json::parse(response.response_json);
  EXPECT_EQ(body.at("status"), "stale");
  EXPECT_EQ(body.at("latest_scene_revision"), loaded.revision + 1);
  EXPECT_EQ(response.error.find("expected=1 current=2") != std::string::npos,
            true);
  const SceneObjectPtr object = reducer.snapshot().findExactObject(7);
  ASSERT_TRUE(object);
  EXPECT_FALSE(object->annotation->label_override.has_value());
  EXPECT_EQ(object->annotation->attributes.at("writer"), "other-client");
}

TEST(SceneMutationJsonAdapter,
     InvalidOrDirectRelationWritesFailBeforeSubmission) {
  int submissions = 0;
  SceneMutationJsonAdapter adapter(
      [&](ApplyHumanAnnotationCommand, std::chrono::milliseconds) {
        ++submissions;
        return SceneMutationSubmitResult{};
      });

  const SceneMutationJsonResponse missing_base = adapter.dispatch(
      R"({"schema_version":"roomie.mutate_scene.v1","operation":"apply_human_annotation","object_id":1,"patch":{"label":"chair"}})");
  EXPECT_FALSE(missing_base.success);
  EXPECT_TRUE(missing_base.response_json.empty());
  EXPECT_NE(missing_base.error.find("base_scene_revision"), std::string::npos);

  const SceneMutationJsonResponse relation_write = adapter.dispatch(
      R"({"schema_version":"roomie.mutate_scene.v1","operation":"apply_human_annotation","base_scene_revision":1,"object_id":1,"patch":{"relations":[]}})");
  EXPECT_FALSE(relation_write.success);
  EXPECT_TRUE(relation_write.response_json.empty());
  EXPECT_NE(relation_write.error.find("unsupported field 'relations'"),
            std::string::npos);
  EXPECT_EQ(submissions, 0);
}

TEST(SceneMutationJsonAdapter,
     CommittedButUndurableIsExplicitAndNeverReportedAsARejection) {
  SceneMutationJsonAdapter adapter(
      [](ApplyHumanAnnotationCommand, std::chrono::milliseconds) {
        SceneMutationSubmitResult result;
        result.status = SceneMutationStatus::kCommittedNotDurable;
        result.message = "commit 9 is not durable";
        result.latest_scene_revision = 9;
        result.durable_scene_revision = 8;
        result.committed_scene_revision = 9;
        return result;
      });
  const SceneMutationJsonResponse response = adapter.dispatch(
      R"({"schema_version":"roomie.mutate_scene.v1","operation":"apply_human_annotation","base_scene_revision":8,"object_id":1,"patch":{"label":"chair"}})");
  EXPECT_FALSE(response.success);
  const Json body = Json::parse(response.response_json);
  EXPECT_EQ(body.at("status"), "committed_not_durable");
  EXPECT_EQ(body.at("committed_scene_revision"), 9);
  EXPECT_EQ(body.at("durable_scene_revision"), 8);
  EXPECT_FALSE(body.at("durable"));
  EXPECT_NE(response.error.find("not durable"), std::string::npos);
}

}  // namespace
}  // namespace roomie
