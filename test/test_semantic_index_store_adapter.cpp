#include <gtest/gtest.h>

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "roomie/artifacts/semantic_index_store_adapter.hpp"
#include "roomie/scene/scene_reducer.hpp"

namespace roomie {
namespace {

using namespace std::chrono_literals;

class TemporaryDatabase {
 public:
  TemporaryDatabase() {
    char pattern[] = "/tmp/roomie_semantic_restore_XXXXXX";
    const int descriptor = mkstemp(pattern);
    if (descriptor < 0) {
      throw std::runtime_error("mkstemp failed");
    }
    close(descriptor);
    path_ = pattern;
  }

  ~TemporaryDatabase() {
    std::remove(path_.c_str());
    std::remove((path_ + "-wal").c_str());
    std::remove((path_ + "-shm").c_str());
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

SceneApplyResult durableScene() {
  LoadSceneCommand command;
  ObjectNode node;
  node.object_id = 1;
  node.semantic_id = 101;
  node.label = "lamp";
  node.size_m = Eigen::Vector3f::Ones();
  node.active = true;
  node.publishable = true;
  command.graph.objects.push_back(std::move(node));
  command.graph.next_object_id = 2;
  ReducerCore reducer;
  return reducer.apply(SceneCommand{command});
}

SemanticDocument semanticDocument(SceneObjectId object_id,
                                  std::string text,
                                  SceneRevision revision = 1) {
  SemanticDocumentInput input;
  input.object_id = object_id;
  input.label = text;
  input.retrieval.canonical_name = text;
  input.retrieval.short_description = text;
  input.retrieval.retrieval_text = std::move(text);
  input.semantic_revision = revision;
  input.created_scene_revision = revision;
  return makeSemanticDocument(input);
}

EmbeddingRecord embeddingRecord(const SemanticDocument& document,
                                std::string model_id,
                                std::vector<float> vector) {
  EmbeddingRecord record;
  record.object_id = document.object_id;
  record.document_hash = document.document_hash;
  record.model_id = std::move(model_id);
  record.vector = std::move(vector);
  record.created_scene_revision = document.created_scene_revision;
  return record;
}

TEST(SemanticIndexStoreAdapter,
     RestartRestoreRejectsStaleDocumentsAndIsolatesNamespace) {
  TemporaryDatabase database;
  const SceneApplyResult scene = durableScene();
  ASSERT_TRUE(scene.committedRevision()) << scene.reason;

  const SemanticDocument current_one =
      semanticDocument(1, "red reading lamp");
  const SemanticDocument stale_two = semanticDocument(2, "old blue chair");
  const SemanticDocument current_two =
      semanticDocument(2, "new green chair");

  {
    SceneStore store(database.path());
    ASSERT_TRUE(store.open());
    ASSERT_TRUE(store.enqueueCommit(scene.snapshot));
    ASSERT_TRUE(store.flush());

    const EmbeddingRecord active =
        embeddingRecord(current_one, "model-a", {1.0f, 0.0f, 0.0f});
    ASSERT_TRUE(persistSemanticEmbeddingRecord(&store, active));
    EXPECT_TRUE(persistSemanticEmbeddingRecord(&store, active))
        << "the worker may redeliver its successful durable write";
    ASSERT_TRUE(persistSemanticEmbeddingRecord(
        &store, embeddingRecord(stale_two, "model-a", {0.0f, 1.0f, 0.0f})));
    ASSERT_TRUE(persistSemanticEmbeddingRecord(
        &store, embeddingRecord(current_one, "model-b", {0.0f, 0.0f, 1.0f})));
    ASSERT_TRUE(persistSemanticEmbeddingRecord(
        &store, embeddingRecord(current_one, "model-a", {1.0f, 0.0f})));
    ASSERT_TRUE(store.closeGracefully());
  }

  SceneStore reopened(database.path());
  ASSERT_TRUE(reopened.open());
  SemanticIndexConfig config;
  config.initial_namespace = {"model-a", 3};
  VersionedSemanticIndex index(config);
  ASSERT_EQ(index.upsertDocument(current_one).jobs.size(), 1u);
  ASSERT_EQ(index.upsertDocument(current_two).jobs.size(), 1u);

  const SemanticEmbeddingRestoreResult restored =
      restoreSemanticIndexEmbeddingRecords(&reopened, &index);
  ASSERT_TRUE(restored.status) << restored.status.error;
  EXPECT_EQ(restored.name_space,
            (EmbeddingNamespace{"model-a", 3}));
  EXPECT_EQ(restored.records_scanned, 2u)
      << "other models and dimensions must not enter the active generation";
  EXPECT_EQ(restored.accepted, 1u);
  EXPECT_EQ(restored.rejected_document_stale, 1u);
  EXPECT_EQ(restored.rejected_namespace, 0u);
  EXPECT_EQ(restored.rejected_dimension, 0u);
  ASSERT_TRUE(index.waitForBuildIdle(1s));

  const SemanticGenerationInfo generation = index.activeGeneration();
  EXPECT_EQ(generation.model_id, "model-a");
  EXPECT_EQ(generation.dimension, 3u);
  EXPECT_EQ(generation.rows, 1u);
  EXPECT_EQ(index.pendingEmbeddingCount(), 1u)
      << "the stale row stays lexical/pending until freshly encoded";
  const std::vector<EmbeddingRecord> records = index.embeddingRecords();
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(records.front().object_id, current_one.object_id);
  EXPECT_EQ(records.front().document_hash, current_one.document_hash);
}

}  // namespace
}  // namespace roomie
