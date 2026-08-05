#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdio>
#include <stdexcept>

#include "roomie/artifacts/artifact_intent_builder.hpp"
#include "roomie/artifacts/embedding_task_scheduler.hpp"
#include "roomie/artifacts/semantic_index_store_adapter.hpp"
#include "roomie/artifacts/semantic_scene_projector.hpp"

namespace roomie {
namespace {

ObjectNode node(int object_id, std::string label,
                const Eigen::Vector3f& center) {
  ObjectNode value;
  value.object_id = object_id;
  value.semantic_id = object_id + 100;
  value.label = std::move(label);
  value.center_world = center;
  value.size_m = Eigen::Vector3f(0.4f, 0.5f, 0.6f);
  value.active = true;
  value.publishable = true;
  return value;
}

SemanticIndexConfig indexConfig() {
  SemanticIndexConfig config;
  config.initial_namespace = {"projector-test", 3};
  return config;
}

class TemporaryProjectorDatabase {
 public:
  TemporaryProjectorDatabase() {
    char pattern[] = "/tmp/roomie_projector_XXXXXX";
    const int descriptor = mkstemp(pattern);
    if (descriptor < 0) {
      throw std::runtime_error("mkstemp failed");
    }
    close(descriptor);
    path_ = pattern;
  }

  ~TemporaryProjectorDatabase() {
    std::remove(path_.c_str());
    std::remove((path_ + "-wal").c_str());
    std::remove((path_ + "-shm").c_str());
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

TEST(SemanticSceneProjector,
     InitializesEveryCanonicalObjectAndUpdatesOnlyTouchedDocuments) {
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(node(1, "chair", {0.0f, 0.0f, 0.5f}));
  load.graph.objects.push_back(node(2, "lamp", {2.0f, 0.0f, 0.8f}));
  load.graph.next_object_id = 3;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  VersionedSemanticIndex index(indexConfig());
  const SemanticSceneProjectionResult initialized =
      initializeSemanticIndexFromScene(loaded.snapshot, &index);
  ASSERT_TRUE(initialized.ok) << initialized.error;
  EXPECT_EQ(initialized.documents_upserted, 2U);
  EXPECT_EQ(initialized.documents_changed, 2U);
  EXPECT_EQ(initialized.missing_embeddings.size(), 2U);

  ApplyHumanAnnotationCommand annotation;
  annotation.object_id = 1;
  annotation.patch.label = "reading chair";
  const SceneApplyResult annotated =
      reducer.apply(SceneCommand{annotation});
  ASSERT_TRUE(annotated.committedRevision()) << annotated.reason;
  const SemanticSceneProjectionResult projected =
      projectSceneCommitToSemanticIndex(annotated, &index);
  ASSERT_TRUE(projected.ok) << projected.error;
  EXPECT_EQ(projected.documents_upserted, 1U);
  EXPECT_EQ(projected.documents_changed, 1U);
  ASSERT_EQ(projected.missing_embeddings.size(), 1U);
  EXPECT_EQ(projected.missing_embeddings.front().object_id, 1);
  EXPECT_EQ(index.pendingEmbeddingCount(), 2U);
}

TEST(SemanticSceneProjector,
     MetadataOnlyUpdateRefreshesFiltersWithoutReencodingDocument) {
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(node(4, "table", {0.0f, 0.0f, 0.4f}));
  load.graph.next_object_id = 5;
  SceneApplyResult result = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(result.committedRevision()) << result.reason;

  VersionedSemanticIndex index(indexConfig());
  ASSERT_TRUE(initializeSemanticIndexFromScene(result.snapshot, &index).ok);

  // Room membership is query metadata and must not invalidate the document.
  ApplyHumanAnnotationCommand metadata;
  metadata.object_id = 4;
  metadata.patch.attributes["room_id"] = "kitchen";
  result = reducer.apply(SceneCommand{metadata});
  ASSERT_TRUE(result.committedRevision()) << result.reason;

  const SemanticSceneProjectionResult projected =
      projectSceneCommitToSemanticIndex(result, &index);
  ASSERT_TRUE(projected.ok) << projected.error;
  EXPECT_EQ(projected.documents_upserted, 1U);
  EXPECT_EQ(projected.documents_changed, 0U);
  EXPECT_EQ(projected.metadata_changed, 1U);
  EXPECT_TRUE(projected.missing_embeddings.empty());
}

TEST(SemanticSceneProjector,
     PendingDescriptionIsHiddenUntilDurabilityWatermarkPromotesIt) {
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(node(7, "lamp", {0.0f, 0.0f, 0.8f}));
  load.graph.next_object_id = 8;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  VersionedSemanticIndex index(indexConfig());
  ASSERT_TRUE(initializeSemanticIndexFromScene(loaded.snapshot, &index).ok);
  const std::string current_hash = makeSemanticDocumentForObject(
      *loaded.snapshot.findObject(7), 7, loaded.revision)
                                       .document_hash;
  ASSERT_FALSE(current_hash.empty());

  ApplyDescriptionArtifactCommand description;
  description.dependency = dependencyFor(*loaded.snapshot.findObject(7));
  description.description = "a red adjustable task lamp";
  description.input_hash = "dam-input-red-lamp";
  description.model_id = "dam-test";
  description.schema_version = "roomie.dam.v1";
  const SceneApplyResult pending =
      reducer.apply(SceneCommand{description});
  ASSERT_TRUE(pending.committedRevision()) << pending.reason;
  const SemanticSceneProjectionResult pending_projection =
      projectSceneCommitToSemanticIndex(pending, &index);
  ASSERT_TRUE(pending_projection.ok) << pending_projection.error;
  EXPECT_EQ(pending_projection.documents_changed, 0U);
  EXPECT_EQ(makeSemanticDocumentForObject(
                *pending.snapshot.findObject(7), 7, pending.revision)
                .document_hash,
            current_hash);

  const SceneApplyResult durable = reducer.apply(
      SceneCommand{PersistedThroughCommand{pending.revision}});
  ASSERT_EQ(durable.status, SceneApplyStatus::kMetadataUpdated)
      << durable.reason;
  const SemanticSceneProjectionResult durable_projection =
      projectSceneCommitToSemanticIndex(durable, &index);
  ASSERT_TRUE(durable_projection.ok) << durable_projection.error;
  EXPECT_EQ(durable_projection.documents_changed, 1U);
  EXPECT_NE(makeSemanticDocumentForObject(
                *durable.snapshot.findObject(7), 7, durable.revision)
                .document_hash,
            current_hash);
}

TEST(SemanticSceneProjector,
     RestartWithNewNamespaceDurablyEnqueuesEveryStillMissingJob) {
  TemporaryProjectorDatabase database;
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(node(21, "cabinet", {1.0f, 2.0f, 0.8f}));
  load.graph.next_object_id = 22;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;
  const SemanticDocument document = makeSemanticDocumentForObject(
      *loaded.snapshot.findExactObject(21), 21, loaded.revision);

  {
    SceneStore old_store(database.path());
    ASSERT_TRUE(old_store.open());
    ASSERT_TRUE(old_store.enqueueCommit(loaded.snapshot));
    ASSERT_TRUE(old_store.flush());
    DurableEmbeddingRecord old_record;
    old_record.object_id = 21;
    old_record.document_hash = document.document_hash;
    old_record.model_id = "old-model";
    old_record.dimension = 3;
    old_record.vector = {1.0f, 0.0f, 0.0f};
    old_record.created_scene_revision = loaded.revision;
    ASSERT_TRUE(old_store.upsertEmbeddingRecord(old_record));
    ASSERT_TRUE(old_store.closeGracefully());
  }

  std::string expected_task_id;
  {
    SceneStore migrated(database.path());
    ASSERT_TRUE(migrated.open());
    const SceneRestoreResult restored = migrated.restoreLatest();
    ASSERT_TRUE(restored.status) << restored.status.error;
    ASSERT_TRUE(restored.found);
    SemanticIndexConfig config;
    config.initial_namespace = {"new-model", 3};
    VersionedSemanticIndex index(config);
    ASSERT_TRUE(
        initializeSemanticIndexFromScene(restored.snapshot, &index).ok);
    const SemanticEmbeddingRestoreResult old_namespace_ignored =
        restoreSemanticIndexEmbeddingRecords(&migrated, &index);
    ASSERT_TRUE(old_namespace_ignored.status)
        << old_namespace_ignored.status.error;
    EXPECT_EQ(old_namespace_ignored.records_scanned, 0U);

    const SemanticSceneProjectionResult audited =
        initializeSemanticIndexFromScene(restored.snapshot, &index);
    ASSERT_TRUE(audited.ok) << audited.error;
    ASSERT_EQ(audited.missing_embeddings.size(), 1U);
    EmbeddingTaskScheduler scheduler(&migrated);
    ASSERT_TRUE(scheduler.ensureTask(audited.missing_embeddings.front(),
                                     restored.snapshot, 10));
    expected_task_id = canonicalEmbeddingTaskKey(
        21, document.document_hash, {"new-model", 3});
    ASSERT_TRUE(migrated.closeGracefully());
  }

  SceneStore restarted(database.path());
  ASSERT_TRUE(restarted.open());
  EmbeddingTaskScheduler scheduler(&restarted);
  const EmbeddingTaskLeaseResult leased =
      scheduler.leaseNext("restart-new-namespace", 10);
  ASSERT_TRUE(leased.status) << leased.status.error;
  ASSERT_TRUE(leased.lease);
  ASSERT_TRUE(leased.lease->request) << leased.lease->validation_error;
  EXPECT_EQ(leased.lease->taskId(), expected_task_id);
  EXPECT_EQ(leased.lease->request->name_space.model_id, "new-model");
  EXPECT_EQ(leased.lease->request->name_space.dimension, 3U);
}

TEST(SemanticSceneProjector,
     MissingEmbeddingRecoveryUsesDocumentIdentityAndCompletedTaskIsIdempotent) {
  TemporaryProjectorDatabase database;
  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(node(31, "stool", {0.0f, 0.0f, 0.4f}));
  load.graph.next_object_id = 32;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(loaded.snapshot));
  ASSERT_TRUE(store.flush());
  SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;

  VersionedSemanticIndex index(indexConfig());
  const SemanticSceneProjectionResult projected =
      initializeSemanticIndexFromScene(restored.snapshot, &index);
  ASSERT_TRUE(projected.ok) << projected.error;
  ASSERT_EQ(projected.missing_embeddings.size(), 1U);
  const EmbeddingJob job = projected.missing_embeddings.front();

  // Simulate a later metadata/geometry commit that advances the semantic
  // component revision while leaving the canonical document unchanged.
  SceneState next = *restored.snapshot.statePtr();
  SceneObjectTable objects = restored.snapshot.objects();
  const SceneObjectPtr original = objects.at(31);
  auto updated = std::make_shared<SceneObject>(*original);
  auto semantic = std::make_shared<SemanticComponent>(*original->semantic);
  ++semantic->revision;
  semantic->confidence = 0.95f;
  updated->semantic = std::move(semantic);
  objects[31] = std::move(updated);
  next.objects =
      std::make_shared<const SceneObjectTable>(std::move(objects));
  ++next.latest_scene_revision;
  const SceneSnapshot revision_only{
      std::make_shared<const SceneState>(std::move(next))};
  ASSERT_EQ(job.document_hash,
            makeSemanticDocumentForObject(
                *revision_only.findExactObject(31), 31,
                revision_only.revision())
                .document_hash);
  ASSERT_TRUE(store.enqueueCommit(revision_only));
  ASSERT_TRUE(store.flush());
  restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;

  EmbeddingTaskScheduler scheduler(&store);
  ASSERT_TRUE(scheduler.ensureTask(job, restored.snapshot, 10));
  const EmbeddingTaskLeaseResult leased =
      scheduler.leaseNext("document-identity-test", 10);
  ASSERT_TRUE(leased.status) << leased.status.error;
  ASSERT_TRUE(leased.lease);
  ASSERT_TRUE(scheduler.complete(*leased.lease, 11));

  // A recovery audit may rediscover the same missing document after the task
  // has completed. New timing/provenance must not conflict with its immutable
  // canonical task envelope.
  EXPECT_TRUE(scheduler.ensureTask(job, restored.snapshot, 999));
}

}  // namespace
}  // namespace roomie
