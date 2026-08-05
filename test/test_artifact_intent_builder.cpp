#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "roomie/artifacts/artifact_intent_builder.hpp"
#include "roomie/query/scene_query_gateway.hpp"

namespace roomie {
namespace {

using Json = nlohmann::json;

class TemporaryIntentDatabase {
 public:
  TemporaryIntentDatabase() {
    char pattern[] = "/tmp/roomie_artifact_intents_XXXXXX";
    const int descriptor = mkstemp(pattern);
    if (descriptor < 0) {
      throw std::runtime_error("mkstemp failed");
    }
    close(descriptor);
    path_ = pattern;
  }
  ~TemporaryIntentDatabase() {
    std::remove(path_.c_str());
    std::remove((path_ + "-wal").c_str());
    std::remove((path_ + "-shm").c_str());
  }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

ObjectNode intentNode(SceneObjectId object_id,
                      Eigen::Vector3f center = Eigen::Vector3f::Zero(),
                      bool active = true) {
  ObjectNode node;
  node.object_id = object_id;
  node.semantic_id = object_id + 100;
  node.label = "reading lamp";
  node.center_world = center;
  node.size_m = Eigen::Vector3f(0.3f, 0.4f, 0.5f);
  node.active = active;
  node.publishable = true;
  return node;
}

ObjectSnapshotImage snapshotImage(int image_index) {
  ObjectSnapshotImage image;
  image.image_index = image_index;
  image.uri = "asset://frame-" + std::to_string(image_index);
  image.width = 1280;
  image.height = 720;
  image.encoding = "rgb8";
  image.time_ns = 1'000'000 + image_index;
  image.camera_id = "head_rgbd";
  image.source_path = "/recording/frame-" + std::to_string(image_index) +
                      ".png";
  return image;
}

ObjectSnapshotRef snapshotRef(int image_index, float x = 10.0f) {
  ObjectSnapshotRef snapshot;
  snapshot.image_index = image_index;
  snapshot.bbox_xyxy = {x, 20.0f, x + 100.0f, 180.0f};
  snapshot.quality = 0.91f;
  snapshot.time_ns = 1'000'000 + image_index;
  snapshot.camera_id = "head_rgbd";
  return snapshot;
}

SceneApplyResult applySnapshot(ReducerCore* reducer,
                               SceneObjectId object_id,
                               int image_index,
                               std::string hash) {
  ApplySnapshotSetCommand command;
  command.dependency =
      dependencyFor(*reducer->snapshot().findObject(object_id));
  command.snapshots.push_back(snapshotRef(image_index));
  command.snapshot_set_hash = std::move(hash);
  return reducer->apply(SceneCommand{command});
}

void expectSameTask(const DurableTaskSpec& lhs, const DurableTaskSpec& rhs) {
  EXPECT_EQ(lhs.task_id, rhs.task_id);
  EXPECT_EQ(lhs.dedupe_key, rhs.dedupe_key);
  EXPECT_EQ(lhs.task_type, rhs.task_type);
  EXPECT_EQ(lhs.payload, rhs.payload);
  EXPECT_EQ(lhs.scene_revision, rhs.scene_revision);
  EXPECT_EQ(lhs.not_before_unix_ms, rhs.not_before_unix_ms);
}

ArtifactIntentBuilderConfig builderConfig() {
  ArtifactIntentBuilderConfig config;
  config.dam_model_id = "fake-dam-v2";
  config.dam_prompt_hash = "sha256:prompt";
  config.dam_output_schema_version = "dam.schema.v7";
  config.embedding_namespace = {"fake-embedding-v3", 3};
  config.embedding_task_type = "embedding.test.v1";
  config.new_object_window_ms = 10'000;
  config.new_object_interactive_limit = 5;
  return config;
}

TEST(ArtifactIntentBuilder,
     SnapshotEventsUseStableDamKeysBurstPriorityAndFullProvenance) {
  ReducerCore reducer;
  LoadSceneCommand load;
  for (SceneObjectId object_id = 1; object_id <= 6; ++object_id) {
    load.graph.objects.push_back(intentNode(object_id));
    load.graph.snapshot_images.push_back(snapshotImage(object_id));
  }
  load.graph.next_object_id = 7;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  ArtifactIntentBuilder builder(builderConfig());
  const ArtifactCommitIntents loaded_intents = builder.build(loaded, 1000);
  EXPECT_TRUE(loaded_intents.tasks.empty());
  EXPECT_EQ(loaded_intents.origins.size(), 6U);

  std::vector<SceneApplyResult> commits;
  std::vector<DurableTaskSpec> tasks;
  for (SceneObjectId object_id = 1; object_id <= 6; ++object_id) {
    SceneApplyResult commit = applySnapshot(
        &reducer,
        object_id,
        object_id,
        "snapshot-set-" + std::to_string(object_id));
    ASSERT_TRUE(commit.committedRevision()) << commit.reason;
    std::vector<DurableTaskSpec> built =
        builder.build(commit, 1100 + object_id * 100);
    ASSERT_EQ(built.size(), 1U);
    EXPECT_EQ(built.front().scene_revision, commit.revision);
    commits.push_back(std::move(commit));
    tasks.push_back(std::move(built.front()));
  }

  for (std::size_t index = 0; index < 5; ++index) {
    EXPECT_EQ(tasks[index].task_type,
              ArtifactScheduler::kInteractiveTaskType);
  }
  EXPECT_EQ(tasks[5].task_type, ArtifactScheduler::kBulkTaskType)
      << "the sixth new object in one rolling 10 s window is bulk";

  const Json outer = Json::parse(tasks.front().payload);
  EXPECT_EQ(outer.at("payload_version"), 1);
  EXPECT_EQ(outer.at("priority"), "interactive");
  EXPECT_EQ(outer.at("scene_revision"), commits.front().revision);
  EXPECT_EQ(tasks.front().not_before_unix_ms, 1000)
      << "first appearance must retain the stable object commit origin";
  EXPECT_EQ(outer.at("due_unix_ms"), tasks.front().not_before_unix_ms + 10'000);
  EXPECT_TRUE(outer.contains("steady_clock_epoch"));
  EXPECT_GT(outer.at("origin_steady_ns").get<std::int64_t>(), 0);
  EXPECT_EQ(outer.at("due_steady_ns").get<std::int64_t>(),
            outer.at("origin_steady_ns").get<std::int64_t>() +
                10'000'000'000LL);
  const Json input =
      Json::parse(outer.at("input_payload").get<std::string>());
  EXPECT_EQ(input.at("owning_scene_revision"), commits.front().revision);
  EXPECT_EQ(input.at("snapshot_set_hash"), "snapshot-set-1");
  ASSERT_EQ(input.at("snapshots").size(), 1U);
  const Json& snapshot = input.at("snapshots").front();
  EXPECT_EQ(snapshot.at("image_index"), 1);
  EXPECT_EQ(snapshot.at("bbox_xyxy"),
            Json::array({10.0f, 20.0f, 110.0f, 180.0f}));
  EXPECT_EQ(snapshot.at("time_ns"), 1'000'001);
  EXPECT_EQ(snapshot.at("camera_id"), "head_rgbd");
  EXPECT_EQ(snapshot.at("provenance").at("image_index"), 1);
  EXPECT_EQ(snapshot.at("image").at("uri"), "asset://frame-1");
  EXPECT_EQ(snapshot.at("image").at("width"), 1280);
  EXPECT_EQ(snapshot.at("image").at("source_path"),
            "/recording/frame-1.png");

  // Replaying the first owning commit after the burst must not change its
  // admission class, timestamp, payload, or stable key.
  const std::vector<DurableTaskSpec> replay =
      builder.build(commits.front(), 99'999);
  ASSERT_EQ(replay.size(), 1U);
  expectSameTask(replay.front(), tasks.front());

  // A later appearance update is interactive even while the new-object burst
  // remains above the threshold.
  const SceneApplyResult update =
      applySnapshot(&reducer, 1, 101, "snapshot-set-1-revised");
  ASSERT_TRUE(update.committedRevision()) << update.reason;
  const std::vector<DurableTaskSpec> updated = builder.build(update, 2000);
  ASSERT_EQ(updated.size(), 1U);
  EXPECT_EQ(updated.front().task_type,
            ArtifactScheduler::kInteractiveTaskType);
  EXPECT_NE(updated.front().task_id, tasks.front().task_id);
  EXPECT_EQ(updated.front().scene_revision, update.revision);
}

SceneApplyResult descriptionCommit(Eigen::Vector3f center,
                                   bool active,
                                   std::string human_tag,
                                   std::string room) {
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(intentNode(8, center, active));
  load.graph.snapshot_images.push_back(snapshotImage(8));
  load.graph.next_object_id = 9;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  if (!loaded.committedRevision()) {
    throw std::runtime_error(loaded.reason);
  }

  ApplyHumanAnnotationCommand annotation;
  annotation.object_id = 8;
  annotation.patch.attributes["owner_tag"] = std::move(human_tag);
  annotation.patch.attributes["room"] = std::move(room);
  const SceneApplyResult annotated = reducer.apply(SceneCommand{annotation});
  if (!annotated.committedRevision()) {
    throw std::runtime_error(annotated.reason);
  }

  const SceneApplyResult snapshotted =
      applySnapshot(&reducer, 8, 8, "snapshot-set-description");
  if (!snapshotted.committedRevision()) {
    throw std::runtime_error(snapshotted.reason);
  }

  ApplyDescriptionArtifactCommand description;
  description.dependency = dependencyFor(*reducer.snapshot().findObject(8));
  description.description =
      R"({"canonical_name":"task lamp","short_description":"a red desk lamp","retrieval_text":"red metal adjustable task lamp","visual_attributes":{"colors":["red"],"materials":["metal"],"shape":["conical shade"],"visible_parts":["shade","arm"],"state_or_pose":["upright"],"distinctive_marks":[],"visible_text":[]}})";
  description.input_hash = "dam-input-hash-v1";
  description.model_id = "fake-dam-v2";
  description.schema_version = "dam.schema.v7";
  description.artifact_slo.origin_created_unix_ms = 1'000;
  description.artifact_slo.due_unix_ms = 11'000;
  description.artifact_slo.priority = ArtifactPriority::kInteractive;
  SceneApplyResult described = reducer.apply(SceneCommand{description});
  if (!described.committedRevision()) {
    throw std::runtime_error(described.reason);
  }
  return described;
}

TEST(ArtifactIntentBuilder,
     DescriptionCommitCreatesDeterministicEmbeddingIntentWithoutMetadataHash) {
  const SceneApplyResult near_active =
      descriptionCommit(Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                        true,
                        "alice",
                        "study");
  const SceneApplyResult moved_inactive =
      descriptionCommit(Eigen::Vector3f(90.0f, -4.0f, 0.5f),
                        false,
                        "alice",
                        "garage");
  ASSERT_EQ(near_active.revision, moved_inactive.revision);

  ArtifactIntentBuilder first_builder(builderConfig());
  ArtifactIntentBuilder second_builder(builderConfig());
  const std::vector<DurableTaskSpec> first =
      first_builder.build(near_active, 5000);
  const std::vector<DurableTaskSpec> second =
      second_builder.build(moved_inactive, 5000);
  ASSERT_EQ(first.size(), 1U);
  ASSERT_EQ(second.size(), 1U);
  EXPECT_EQ(first.front().task_type, "embedding.test.v1");
  EXPECT_EQ(first.front().scene_revision, near_active.revision);
  EXPECT_EQ(second.front().scene_revision, moved_inactive.revision);

  const Json first_payload = Json::parse(first.front().payload);
  const Json second_payload = Json::parse(second.front().payload);
  EXPECT_EQ(first_payload.at("owning_scene_revision"), near_active.revision);
  EXPECT_EQ(first_payload.at("document_hash"),
            second_payload.at("document_hash"));
  EXPECT_EQ(first_payload.at("document"), second_payload.at("document"));
  EXPECT_EQ(first.front().task_id, second.front().task_id);
  EXPECT_EQ(first_payload.at("model_id"), "fake-embedding-v3");
  EXPECT_EQ(first_payload.at("dimension"), 3);
  EXPECT_EQ(first_payload.at("created_scene_revision"), near_active.revision);
  EXPECT_EQ(first_payload.at("semantic_revision"), 1);
  EXPECT_EQ(first_payload.at("description_input_hash"), "dam-input-hash-v1");
  EXPECT_EQ(first_payload.at("artifact_slo").at("origin_created_unix_ms"),
            1'000);
  EXPECT_EQ(first_payload.at("artifact_slo").at("due_unix_ms"), 11'000);
  EXPECT_EQ(first_payload.at("artifact_slo").at("priority"), "interactive");
  EXPECT_EQ(first_payload.at("document_hash").get<std::string>().size(), 64U);
  ASSERT_TRUE(near_active.snapshot.findExactObject(8));
  EXPECT_EQ(first_payload.at("document_hash").get<std::string>(),
            makeSemanticDocumentForObject(
                *near_active.snapshot.findExactObject(8), 8,
                near_active.revision,
                SemanticDocumentView::kPendingDescriptionIfPresent)
                .document_hash);
  EXPECT_NE(first_payload.at("document_hash").get<std::string>(),
            SceneQueryGateway::semanticDocumentHash(
                *near_active.snapshot.findExactObject(8)))
      << "query projection must hide the pending description until durable";
  EXPECT_EQ(first.front().task_id,
            canonicalEmbeddingTaskKey(
                8,
                first_payload.at("document_hash").get<std::string>(),
                EmbeddingNamespace{"fake-embedding-v3", 3}));

  SceneObject changed_slo = *near_active.snapshot.findExactObject(8);
  auto changed_artifact =
      std::make_shared<ArtifactComponent>(*changed_slo.artifact);
  changed_artifact->pending_description_slo.due_unix_ms += 5'000;
  changed_slo.artifact = std::move(changed_artifact);
  EXPECT_EQ(makeSemanticDocumentForObject(
                *near_active.snapshot.findExactObject(8), 8,
                near_active.revision,
                SemanticDocumentView::kPendingDescriptionIfPresent)
                .document_hash,
            makeSemanticDocumentForObject(
                changed_slo, 8, near_active.revision,
                SemanticDocumentView::kPendingDescriptionIfPresent)
                .document_hash)
      << "SLO accounting metadata must not pollute semantic document hash";

  // A true human semantic tag is part of the document, unlike position and
  // lifecycle metadata.
  const SceneApplyResult different_tag =
      descriptionCommit(Eigen::Vector3f(1.0f, 2.0f, 3.0f),
                        true,
                        "bob",
                        "study");
  ArtifactIntentBuilder third_builder(builderConfig());
  const std::vector<DurableTaskSpec> third =
      third_builder.build(different_tag, 5000);
  ASSERT_EQ(third.size(), 1U);
  const Json third_payload = Json::parse(third.front().payload);
  EXPECT_NE(first_payload.at("document_hash"),
            third_payload.at("document_hash"));
  EXPECT_NE(first.front().task_id, third.front().task_id);

  // Full commit replay is byte-idempotent even if the caller supplies a later
  // wall-clock value.
  const std::vector<DurableTaskSpec> replay =
      first_builder.build(near_active, 6000);
  ASSERT_EQ(replay.size(), 1U);
  expectSameTask(first.front(), replay.front());
}

TEST(ArtifactIntentBuilder,
     HumanSemanticAnnotationCreatesSameRevisionEmbeddingIntent) {
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(intentNode(8));
  load.graph.snapshot_images.push_back(snapshotImage(8));
  load.graph.next_object_id = 9;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;
  const SceneApplyResult snapshotted =
      applySnapshot(&reducer, 8, 8, "snapshot-set-annotation");
  ASSERT_TRUE(snapshotted.committedRevision()) << snapshotted.reason;

  ApplyDescriptionArtifactCommand description;
  description.dependency = dependencyFor(*reducer.snapshot().findObject(8));
  description.description = "a brass reading lamp";
  description.normalized_json =
      R"({"canonical_name":"reading lamp","short_description":"a brass reading lamp","retrieval_text":"brass adjustable reading lamp"})";
  description.input_hash = "dam-input-annotation-v1";
  description.model_id = "fake-dam-v2";
  description.schema_version = "dam.schema.v7";
  const SceneApplyResult described =
      reducer.apply(SceneCommand{description});
  ASSERT_TRUE(described.committedRevision()) << described.reason;

  ArtifactIntentBuilder builder(builderConfig());
  const std::vector<DurableTaskSpec> description_tasks =
      builder.build(described, 7000);
  ASSERT_EQ(description_tasks.size(), 1u);
  const Json description_payload = Json::parse(description_tasks.front().payload);

  ApplyHumanAnnotationCommand annotation;
  annotation.object_id = 8;
  annotation.patch.label = "banker's lamp";
  annotation.patch.attributes["owner_tag"] = "alice";
  const SceneApplyResult annotated =
      reducer.apply(SceneCommand{annotation});
  ASSERT_TRUE(annotated.committedRevision()) << annotated.reason;
  const auto committed_event = std::find_if(
      annotated.events.begin(), annotated.events.end(),
      [](const SceneEvent& event) {
        return std::holds_alternative<HumanAnnotationCommitted>(event);
      });
  ASSERT_NE(committed_event, annotated.events.end());
  EXPECT_TRUE(std::get<HumanAnnotationCommitted>(*committed_event)
                  .semantic_document_changed);

  const std::vector<DurableTaskSpec> annotation_tasks =
      builder.build(annotated, 7100);
  ASSERT_EQ(annotation_tasks.size(), 1u);
  EXPECT_EQ(annotation_tasks.front().scene_revision, annotated.revision);
  EXPECT_NE(annotation_tasks.front().task_id,
            description_tasks.front().task_id);
  const Json annotation_payload = Json::parse(annotation_tasks.front().payload);
  EXPECT_EQ(annotation_payload.at("owning_scene_revision"), annotated.revision);
  EXPECT_EQ(annotation_payload.at("created_scene_revision"), annotated.revision);
  EXPECT_EQ(annotation_payload.at("annotation_revision"),
            annotated.snapshot.findExactObject(8)->annotation->revision);
  EXPECT_NE(annotation_payload.at("document_hash"),
            description_payload.at("document_hash"));
  EXPECT_EQ(annotation_payload.at("document_hash").get<std::string>(),
            SceneQueryGateway::semanticDocumentHash(
                *annotated.snapshot.findExactObject(8)));

  ApplyHumanAnnotationCommand metadata_only;
  metadata_only.object_id = 8;
  metadata_only.patch.attributes["room"] = "study";
  const SceneApplyResult moved =
      reducer.apply(SceneCommand{metadata_only});
  ASSERT_TRUE(moved.committedRevision()) << moved.reason;
  EXPECT_TRUE(builder.build(moved, 7200).empty())
      << "metadata-only edits keep the content-addressed embedding current";
}

TEST(ArtifactIntentBuilder, EmbeddingKeySeparatesModelAndDimensionNamespaces) {
  const std::string hash(64, 'a');
  const std::string base =
      canonicalEmbeddingTaskKey(3, hash, {"model-a", 3});
  EXPECT_EQ(base, canonicalEmbeddingTaskKey(3, hash, {"model-a", 3}));
  EXPECT_NE(base, canonicalEmbeddingTaskKey(3, hash, {"model-b", 3}));
  EXPECT_NE(base, canonicalEmbeddingTaskKey(3, hash, {"model-a", 4}));
  EXPECT_NE(base, canonicalEmbeddingTaskKey(4, hash, {"model-a", 3}));

  ArtifactIntentBuilder builder(builderConfig());
  SceneApplyResult noop;
  noop.status = SceneApplyStatus::kNoOp;
  EXPECT_TRUE(builder.build(noop, 1).empty());
}

TEST(ArtifactIntentBuilder, ReplayMemoIsBounded) {
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(intentNode(1));
  load.graph.objects.push_back(intentNode(2));
  load.graph.snapshot_images.push_back(snapshotImage(1));
  load.graph.snapshot_images.push_back(snapshotImage(2));
  load.graph.next_object_id = 3;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  ArtifactIntentBuilderConfig config = builderConfig();
  config.replay_memo_capacity = 1;
  ArtifactIntentBuilder builder(std::move(config));

  const SceneApplyResult first =
      applySnapshot(&reducer, 1, 1, "snapshot-set-memo-1");
  ASSERT_TRUE(first.committedRevision()) << first.reason;
  ASSERT_EQ(builder.build(first, 1000).size(), 1U);
  EXPECT_EQ(builder.memoizedTaskCount(), 1U);

  const SceneApplyResult second =
      applySnapshot(&reducer, 2, 2, "snapshot-set-memo-2");
  ASSERT_TRUE(second.committedRevision()) << second.reason;
  ASSERT_EQ(builder.build(second, 1100).size(), 1U);
  EXPECT_EQ(builder.memoizedTaskCount(), 1U);
}

TEST(ArtifactIntentBuilder,
     DurableOriginSurvivesRestartBeforeFirstSnapshotWithoutSteadyReuse) {
  TemporaryIntentDatabase database;
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(intentNode(1));
  load.graph.snapshot_images.push_back(snapshotImage(1));
  load.graph.next_object_id = 2;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  ArtifactIntentBuilder first(builderConfig());
  const ArtifactCommitIntents created =
      first.build(loaded, 1'000, 5'000'000'000LL);
  ASSERT_EQ(created.origins.size(), 1U);
  EXPECT_EQ(created.origins.front().created_unix_ms, 1'000);

  {
    SceneStore store(database.path());
    ASSERT_TRUE(store.open());
    ASSERT_TRUE(store.enqueueCommit(loaded.snapshot, created.tasks,
                                    created.origins));
    ASSERT_TRUE(store.closeGracefully());
  }

  SceneStore reopened(database.path());
  ASSERT_TRUE(reopened.open());
  const ArtifactOriginListResult origins = reopened.listArtifactOrigins();
  ASSERT_TRUE(origins.status) << origins.status.error;
  ASSERT_EQ(origins.origins.size(), 1U);

  ArtifactIntentBuilder restarted(builderConfig());
  restarted.restoreObjectOrigins(origins.origins);
  restarted.setDurableTaskLookup(
      [&reopened](const std::string& task_id) {
        return reopened.lookupTask(task_id);
      });
  const SceneApplyResult snapshotted =
      applySnapshot(&reducer, 1, 1, "snapshot-after-restart");
  ASSERT_TRUE(snapshotted.committedRevision()) << snapshotted.reason;
  const ArtifactCommitIntents tasks =
      restarted.build(snapshotted, 99'000, 99'000'000'000LL);
  ASSERT_EQ(tasks.tasks.size(), 1U);
  EXPECT_EQ(tasks.tasks.front().not_before_unix_ms, 1'000);
  const Json payload = Json::parse(tasks.tasks.front().payload);
  EXPECT_EQ(payload.at("created_unix_ms"), 1'000);
  EXPECT_EQ(payload.at("due_unix_ms"), 11'000);
  EXPECT_FALSE(payload.contains("steady_clock_epoch"))
      << "a steady timestamp from the previous process must never be reused";
}

TEST(ArtifactIntentBuilder,
     LiveObjectOriginsAreNotEvictedWithTheReplayMemo) {
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(intentNode(1));
  load.graph.objects.push_back(intentNode(2));
  load.graph.snapshot_images.push_back(snapshotImage(1));
  load.graph.snapshot_images.push_back(snapshotImage(2));
  load.graph.next_object_id = 3;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  ArtifactIntentBuilderConfig config = builderConfig();
  config.replay_memo_capacity = 1;
  ArtifactIntentBuilder builder(std::move(config));
  const ArtifactCommitIntents origins =
      builder.build(loaded, 1'000, 1'000'000'000LL);
  ASSERT_EQ(origins.origins.size(), 2U);
  EXPECT_EQ(builder.objectOriginCount(), 2U);

  const SceneApplyResult first =
      applySnapshot(&reducer, 1, 1, "long-lived-origin");
  const ArtifactCommitIntents task =
      builder.build(first, 50'000, 50'000'000'000LL);
  ASSERT_EQ(task.tasks.size(), 1U);
  EXPECT_EQ(task.tasks.front().not_before_unix_ms, 1'000);
}

TEST(ArtifactIntentBuilder, NewObjectBurstUsesSteadyClockAcrossWallRollback) {
  ArtifactIntentBuilder builder(builderConfig());
  std::vector<DurableArtifactOrigin> origins;
  for (SceneObjectId object_id = 1; object_id <= 6; ++object_id) {
    ReducerCore reducer;
    LoadSceneCommand load;
    load.graph.objects.push_back(intentNode(object_id));
    load.graph.next_object_id = object_id + 1;
    const SceneApplyResult created = reducer.apply(SceneCommand{load});
    ASSERT_TRUE(created.committedRevision()) << created.reason;
    const ArtifactCommitIntents intents = builder.build(
        created, 10'000 - object_id * 100,
        1'000'000'000LL + object_id * 100'000'000LL);
    ASSERT_EQ(intents.origins.size(), 1U);
    origins.push_back(intents.origins.front());
  }
  for (std::size_t index = 0; index < 5; ++index) {
    EXPECT_EQ(origins[index].priority, ArtifactPriority::kInteractive);
  }
  EXPECT_EQ(origins.back().priority, ArtifactPriority::kBulk);
}

TEST(ArtifactIntentBuilder,
     DurableLookupKeepsReplayByteIdenticalAfterMemoEviction) {
  TemporaryIntentDatabase database;
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());

  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(intentNode(1));
  load.graph.objects.push_back(intentNode(2));
  load.graph.snapshot_images.push_back(snapshotImage(1));
  load.graph.snapshot_images.push_back(snapshotImage(2));
  load.graph.next_object_id = 3;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});

  ArtifactIntentBuilderConfig config = builderConfig();
  config.replay_memo_capacity = 1;
  ArtifactIntentBuilder builder(std::move(config));
  builder.setDurableTaskLookup(
      [&store](const std::string& task_id) {
        return store.lookupTask(task_id);
      });
  ArtifactCommitIntents intents = builder.build(loaded, 1'000);
  ASSERT_TRUE(store.enqueueCommit(loaded.snapshot, intents.tasks,
                                  intents.origins));
  ASSERT_TRUE(store.flush());

  const SceneApplyResult first =
      applySnapshot(&reducer, 1, 1, "durable-replay-1");
  intents = builder.build(first, 2'000);
  ASSERT_EQ(intents.tasks.size(), 1U);
  const DurableTaskSpec original = intents.tasks.front();
  ASSERT_TRUE(store.enqueueCommit(first.snapshot, intents.tasks,
                                  intents.origins));
  ASSERT_TRUE(store.flush());

  const SceneApplyResult second =
      applySnapshot(&reducer, 2, 2, "durable-replay-2");
  intents = builder.build(second, 3'000);
  ASSERT_TRUE(store.enqueueCommit(second.snapshot, intents.tasks,
                                  intents.origins));
  ASSERT_TRUE(store.flush());
  EXPECT_EQ(builder.memoizedTaskCount(), 1U);

  const ArtifactCommitIntents replay = builder.build(first, 999'000);
  ASSERT_EQ(replay.tasks.size(), 1U);
  expectSameTask(replay.tasks.front(), original);
}

}  // namespace
}  // namespace roomie
