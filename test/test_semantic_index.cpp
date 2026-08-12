#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "roomie/artifacts/semantic_index.hpp"

namespace roomie {
namespace {

using namespace std::chrono_literals;

SemanticDocument document(SceneObjectId object_id,
                          std::string label,
                          std::string retrieval_text,
                          SceneRevision scene_revision,
                          SemanticMetadata metadata = {}) {
  SemanticDocumentInput input;
  input.object_id = object_id;
  input.label = std::move(label);
  input.retrieval.canonical_name = input.label;
  input.retrieval.short_description = retrieval_text;
  input.retrieval.retrieval_text = std::move(retrieval_text);
  input.retrieval.attributes["color"] = {"red", "matte"};
  input.human_tags = {"home", "favorite"};
  input.metadata = std::move(metadata);
  input.semantic_revision = scene_revision;
  input.created_scene_revision = scene_revision;
  return makeSemanticDocument(input);
}

EmbeddingRecord recordFor(const EmbeddingJob& job,
                          std::vector<float> vector) {
  EmbeddingRecord record;
  record.object_id = job.object_id;
  record.document_hash = job.document_hash;
  record.model_id = job.name_space.model_id;
  record.vector = std::move(vector);
  record.created_scene_revision = job.created_scene_revision;
  return record;
}

SemanticIndexConfig indexConfig(std::string model_id = "fake-v1",
                                std::size_t dimension = 3) {
  SemanticIndexConfig config;
  config.initial_namespace = {std::move(model_id), dimension};
  config.default_read_ttl_ms = 1000;
  return config;
}

class FakeEncoder final : public EmbeddingEncoder {
 public:
  FakeEncoder(std::string model_id, std::size_t dimension)
      : model_id_(std::move(model_id)), dimension_(dimension) {}

  std::string modelId() const override { return model_id_; }
  std::size_t dimension() const override { return dimension_; }

  void prewarm() override { ++prewarm_calls; }

  std::vector<std::vector<float>> encodeBatch(
      const std::vector<std::string>& documents) override {
    ++batch_calls;
    encoded_documents.fetch_add(documents.size());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      batches.push_back(documents);
    }
    std::vector<std::vector<float>> result;
    result.reserve(documents.size());
    for (const std::string& value : documents) {
      std::vector<float> vector(dimension_, 0.0f);
      if (value.find("lamp") != std::string::npos) {
        vector[0] = 4.0f;
      } else if (value.find("chair") != std::string::npos) {
        vector[dimension_ > 1 ? 1 : 0] = 3.0f;
      } else {
        vector[dimension_ - 1] = 2.0f;
      }
      result.push_back(std::move(vector));
    }
    return result;
  }

  std::atomic<int> prewarm_calls{0};
  std::atomic<int> batch_calls{0};
  std::atomic<std::size_t> encoded_documents{0};
  std::mutex mutex_;
  std::vector<std::vector<std::string>> batches;

 private:
  std::string model_id_;
  std::size_t dimension_ = 0;
};

class BuildGate {
 public:
  ~BuildGate() { release(); }

  void hook() {
    if (!armed_.load()) {
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    entered_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this]() { return released_; });
  }

  void arm() {
    std::lock_guard<std::mutex> lock(mutex_);
    entered_ = false;
    released_ = false;
    armed_ = true;
  }

  bool waitUntilEntered(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [this]() { return entered_; });
  }

  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
      armed_ = false;
    }
    cv_.notify_all();
  }

 private:
  std::atomic_bool armed_{false};
  std::mutex mutex_;
  std::condition_variable cv_;
  bool entered_ = false;
  bool released_ = false;
};

TEST(SemanticDocument, IsCanonicalAndExcludesVolatileMetadata) {
  SemanticDocumentInput first;
  first.object_id = 7;
  first.label = "  Reading   LAMP ";
  first.retrieval.canonical_name = "Lamp";
  first.retrieval.short_description = " A red lamp ";
  first.retrieval.retrieval_text = "red metal task lamp";
  first.retrieval.attributes[" Material "] = {"Metal", " metal ", "Steel"};
  first.retrieval.attributes["COLOR"] = {" Red ", "black"};
  first.human_tags = {" Favorite", "desk", "favorite ", ""};
  first.metadata.room_id = "office";
  first.metadata.position_world = {1.0f, 2.0f, 3.0f};
  first.metadata.has_position = true;
  first.metadata.active = true;
  first.semantic_revision = 3;
  first.created_scene_revision = 11;

  SemanticDocumentInput reordered = first;
  reordered.object_id = 99;
  reordered.retrieval.attributes.clear();
  reordered.retrieval.attributes["color"] = {"black", "red"};
  reordered.retrieval.attributes["material"] = {"steel", "metal"};
  reordered.human_tags = {"desk", "favorite"};
  reordered.metadata.room_id = "garage";
  reordered.metadata.position_world = {-10.0f, 5.0f, 0.0f};
  reordered.metadata.active = false;
  reordered.semantic_revision = 500;
  reordered.created_scene_revision = 900;

  const SemanticDocument lhs = makeSemanticDocument(first);
  const SemanticDocument rhs = makeSemanticDocument(reordered);
  EXPECT_EQ(lhs.text, rhs.text);
  EXPECT_EQ(lhs.document_hash, rhs.document_hash);
  EXPECT_EQ(lhs.document_hash.size(), 64U);
  EXPECT_NE(lhs.metadata.room_id, rhs.metadata.room_id);

  reordered.label = "floor lamp";
  EXPECT_NE(makeSemanticDocument(reordered).document_hash, lhs.document_hash);

  SemanticDocumentInput named = first;
  named.name = "Ada's lamp";
  const SemanticDocument named_document = makeSemanticDocument(named);
  EXPECT_NE(named_document.document_hash, lhs.document_hash);
  EXPECT_NE(named_document.text.find("ada's lamp"), std::string::npos);
}

TEST(EmbeddingWorker, FirstSearchIsLexicalAndDoesNotLoadOrEncodeModel) {
  VersionedSemanticIndex index(indexConfig());
  const SemanticUpsertResult upsert =
      index.upsertDocument(document(1, "lamp", "scarlet reading lamp", 10));
  ASSERT_EQ(upsert.jobs.size(), 1U);

  auto encoder = std::make_shared<FakeEncoder>("fake-v1", 3);
  EXPECT_EQ(encoder->prewarm_calls.load(), 0);
  EXPECT_EQ(encoder->encoded_documents.load(), 0U);

  SemanticSearchRequest lexical;
  lexical.query_text = "scarlet lamp";
  const SemanticSearchResponse cold =
      index.search(index.pinRead(10, 100), lexical, 101);
  ASSERT_EQ(cold.status, SemanticSearchStatus::kOk);
  ASSERT_EQ(cold.hits.size(), 1U);
  EXPECT_EQ(cold.hits.front().object_id, 1);
  EXPECT_EQ(cold.hits.front().freshness,
            SemanticFreshness::kEmbeddingPendingLexical);
  EXPECT_EQ(cold.pending_embeddings, 1U);
  EXPECT_EQ(encoder->prewarm_calls.load(), 0);
  EXPECT_EQ(encoder->encoded_documents.load(), 0U);

  std::atomic<EmbeddingAcceptStatus> accepted{
      EmbeddingAcceptStatus::kRejectedNamespace};
  EmbeddingWorkerConfig worker_config;
  worker_config.batch_window = 2ms;
  EmbeddingWorker worker(
      encoder,
      [&](EmbeddingRecord record) {
        accepted = index.acceptEmbedding(std::move(record));
      },
      worker_config);
  worker.start();
  ASSERT_TRUE(worker.waitUntilWarmed(1s));
  ASSERT_TRUE(worker.submit(upsert.jobs.front()).accepted());
  ASSERT_TRUE(worker.waitUntilIdle(1s));
  worker.stop();
  EXPECT_EQ(accepted.load(), EmbeddingAcceptStatus::kAccepted);
  EXPECT_EQ(encoder->prewarm_calls.load(), 1);
  EXPECT_EQ(encoder->encoded_documents.load(), 1U);
  ASSERT_TRUE(index.waitForBuildIdle(1s));

  SemanticSearchRequest vector_search;
  vector_search.query_vector = {5.0f, 0.0f, 0.0f};
  const SemanticSearchResponse warm =
      index.search(index.pinRead(10, 200), vector_search, 201);
  ASSERT_EQ(warm.hits.size(), 1U);
  EXPECT_EQ(warm.hits.front().freshness,
            SemanticFreshness::kCurrentEmbedding);
  EXPECT_NEAR(warm.hits.front().vector_score, 1.0f, 1.0e-6f);
  EXPECT_EQ(warm.pending_embeddings, 0U);
}

TEST(EmbeddingWorker, LatestByObjectAndSingleDocumentIncrementalEncoding) {
  VersionedSemanticIndex index(indexConfig());
  auto encoder = std::make_shared<FakeEncoder>("fake-v1", 3);
  EmbeddingWorkerConfig worker_config;
  worker_config.batch_window = 10ms;
  EmbeddingWorker worker(
      encoder,
      [&](EmbeddingRecord record) { index.acceptEmbedding(std::move(record)); },
      worker_config);

  const SemanticUpsertResult first =
      index.upsertDocument(document(1, "lamp", "old lamp", 1));
  const SemanticUpsertResult superseding =
      index.upsertDocument(document(1, "lamp", "new lamp", 2));
  ASSERT_EQ(first.jobs.size(), 1U);
  ASSERT_EQ(superseding.jobs.size(), 1U);
  ASSERT_TRUE(worker.submit(first.jobs.front()).accepted());
  const PushResult<EmbeddingJob> replacement =
      worker.submit(superseding.jobs.front());
  ASSERT_EQ(replacement.outcome, PushOutcome::kReplaced);
  ASSERT_TRUE(replacement.replaced_item);
  EXPECT_EQ(replacement.replaced_item->document_hash,
            first.jobs.front().document_hash);

  worker.start();
  ASSERT_TRUE(worker.waitUntilWarmed(1s));
  ASSERT_TRUE(worker.waitUntilIdle(1s));
  ASSERT_EQ(encoder->encoded_documents.load(), 1U);
  EXPECT_EQ(worker.stats().replaced, 1U);

  const SemanticUpsertResult second_object =
      index.upsertDocument(document(2, "chair", "blue desk chair", 3));
  ASSERT_EQ(second_object.jobs.size(), 1U);
  ASSERT_TRUE(worker.submit(second_object.jobs.front()).accepted());
  ASSERT_TRUE(worker.waitUntilIdle(1s));
  EXPECT_EQ(encoder->encoded_documents.load(), 2U);

  const SemanticUpsertResult one_changed =
      index.upsertDocument(document(1, "lamp", "new brass lamp", 4));
  ASSERT_EQ(one_changed.jobs.size(), 1U)
      << "one semantic update must enqueue only its own document";
  ASSERT_TRUE(worker.submit(one_changed.jobs.front()).accepted());
  ASSERT_TRUE(worker.waitUntilIdle(1s));
  EXPECT_EQ(encoder->encoded_documents.load(), 3U);

  SemanticMetadata moved;
  moved.room_id = "bedroom";
  moved.position_world = {8.0f, 9.0f, 10.0f};
  moved.has_position = true;
  moved.active = false;
  const SemanticUpsertResult metadata_only = index.upsertDocument(
      document(1, "lamp", "new brass lamp", 5, moved));
  EXPECT_FALSE(metadata_only.document_changed);
  EXPECT_TRUE(metadata_only.metadata_changed);
  EXPECT_TRUE(metadata_only.jobs.empty());
  EXPECT_EQ(encoder->encoded_documents.load(), 3U);
  worker.stop();
  ASSERT_TRUE(index.waitForBuildIdle(1s));

  const std::vector<EmbeddingRecord> persisted = index.embeddingRecords();
  ASSERT_EQ(persisted.size(), 3U);
  EXPECT_EQ(persisted.back().object_id, 1);
  EXPECT_EQ(persisted.back().model_id, "fake-v1");
  EXPECT_EQ(persisted.back().created_scene_revision, 4U);
  float squared_norm = 0.0f;
  for (const float value : persisted.back().vector) {
    squared_norm += value * value;
  }
  EXPECT_NEAR(squared_norm, 1.0f, 1.0e-6f);
}

TEST(VersionedSemanticIndex,
     RecordHistoryRetainsNewestWithoutEvictingCurrentRecords) {
  SemanticIndexConfig config = indexConfig();
  config.record_history_capacity = 2;
  VersionedSemanticIndex index(config);

  for (SceneObjectId object_id = 1; object_id <= 3; ++object_id) {
    const SemanticUpsertResult upsert = index.upsertDocument(document(
        object_id, "object " + std::to_string(object_id),
        "semantic object " + std::to_string(object_id),
        static_cast<SceneRevision>(object_id)));
    ASSERT_EQ(upsert.jobs.size(), 1U);
    EXPECT_EQ(index.acceptEmbedding(
                  recordFor(upsert.jobs.front(), {1.0f, 0.0f, 0.0f})),
              EmbeddingAcceptStatus::kAccepted);
  }
  ASSERT_TRUE(index.waitForBuildIdle(1s));

  const std::vector<EmbeddingRecord> history = index.embeddingRecords();
  ASSERT_EQ(history.size(), 2U);
  EXPECT_EQ(history[0].object_id, 2);
  EXPECT_EQ(history[1].object_id, 3);
  EXPECT_EQ(index.activeGeneration().rows, 3U)
      << "history eviction must not remove current namespace records";
}

TEST(VersionedSemanticIndex, GenerationSwapPinsOldViewAndExpiresExplicitly) {
  auto gate = std::make_shared<BuildGate>();
  SemanticIndexConfig config = indexConfig("pin-model", 2);
  config.before_generation_publish = [gate]() { gate->hook(); };
  VersionedSemanticIndex index(config);

  const SemanticUpsertResult old_upsert =
      index.upsertDocument(document(4, "chair", "old wooden chair", 1));
  ASSERT_EQ(old_upsert.jobs.size(), 1U);
  EXPECT_EQ(index.acceptEmbedding(recordFor(old_upsert.jobs.front(), {0, 5})),
            EmbeddingAcceptStatus::kAccepted);
  ASSERT_TRUE(index.waitForBuildIdle(1s));
  const SemanticReadToken old_token = index.pinRead(1, 100, 500);
  ASSERT_GT(old_token.info().generation, 0U);

  const SemanticUpsertResult changed =
      index.upsertDocument(document(4, "chair", "new striped chair", 2));
  ASSERT_EQ(changed.jobs.size(), 1U);
  ASSERT_TRUE(index.waitForBuildIdle(1s));
  gate->arm();
  EXPECT_EQ(index.acceptEmbedding(recordFor(changed.jobs.front(), {5, 0})),
            EmbeddingAcceptStatus::kAccepted);
  ASSERT_TRUE(gate->waitUntilEntered(1s));

  SemanticSearchRequest old_query;
  old_query.query_text = "wooden";
  const SemanticSearchResponse pinned_old =
      index.search(old_token, old_query, 200);
  ASSERT_EQ(pinned_old.hits.size(), 1U);
  EXPECT_EQ(pinned_old.hits.front().document_hash,
            old_upsert.jobs.front().document_hash);
  EXPECT_EQ(pinned_old.hits.front().freshness,
            SemanticFreshness::kCurrentEmbedding);

  SemanticSearchRequest pending_query;
  pending_query.query_text = "striped";
  const SemanticReadToken during_token = index.pinRead(2, 200, 500);
  const SemanticSearchResponse during =
      index.search(during_token, pending_query, 201);
  ASSERT_EQ(during.hits.size(), 1U);
  EXPECT_EQ(during.hits.front().freshness,
            SemanticFreshness::kEmbeddingPendingLexical);

  gate->release();
  ASSERT_TRUE(index.waitForBuildIdle(1s));
  const SemanticReadToken new_token = index.pinRead(2, 300, 500);
  EXPECT_GT(new_token.info().generation, old_token.info().generation);
  SemanticSearchRequest new_vector;
  new_vector.query_vector = {1, 0};
  const SemanticSearchResponse current =
      index.search(new_token, new_vector, 301);
  ASSERT_EQ(current.hits.size(), 1U);
  EXPECT_EQ(current.hits.front().document_hash,
            changed.jobs.front().document_hash);

  const SemanticSearchResponse old_still_pinned =
      index.search(old_token, old_query, 599);
  EXPECT_EQ(old_still_pinned.status, SemanticSearchStatus::kOk);
  ASSERT_EQ(old_still_pinned.hits.size(), 1U);
  const SemanticSearchResponse expired =
      index.search(old_token, old_query, 600);
  EXPECT_EQ(expired.status, SemanticSearchStatus::kExpiredToken);
  EXPECT_TRUE(expired.hits.empty());
}

TEST(VersionedSemanticIndex, ModelUpgradeNeverMixesModelOrDimension) {
  VersionedSemanticIndex index(indexConfig("model-v1", 2));
  const SemanticUpsertResult lamp =
      index.upsertDocument(document(1, "lamp", "red lamp", 10));
  const SemanticUpsertResult chair =
      index.upsertDocument(document(2, "chair", "blue chair", 11));
  ASSERT_EQ(lamp.jobs.size(), 1U);
  ASSERT_EQ(chair.jobs.size(), 1U);
  ASSERT_EQ(index.acceptEmbedding(recordFor(lamp.jobs.front(), {4, 0})),
            EmbeddingAcceptStatus::kAccepted);
  ASSERT_EQ(index.acceptEmbedding(recordFor(chair.jobs.front(), {0, 3})),
            EmbeddingAcceptStatus::kAccepted);
  ASSERT_TRUE(index.waitForBuildIdle(1s));
  const SemanticReadToken old_token = index.pinRead(11, 100);
  EXPECT_EQ(old_token.info().model_id, "model-v1");
  EXPECT_EQ(old_token.info().dimension, 2U);
  EXPECT_EQ(old_token.info().rows, 2U);

  const std::vector<EmbeddingJob> upgrade =
      index.beginModelUpgrade({"model-v2", 3});
  ASSERT_EQ(upgrade.size(), 2U);
  EXPECT_EQ(index.activeGeneration().model_id, "model-v1");
  ASSERT_EQ(index.acceptEmbedding(recordFor(upgrade[0], {1, 1, 0})),
            EmbeddingAcceptStatus::kAccepted);
  ASSERT_TRUE(index.waitForBuildIdle(100ms));
  EXPECT_EQ(index.activeGeneration().model_id, "model-v1")
      << "an incomplete upgrade must not partially mix namespaces";

  ASSERT_EQ(index.acceptEmbedding(recordFor(upgrade[1], {0, 1, 1})),
            EmbeddingAcceptStatus::kAccepted);
  ASSERT_TRUE(index.waitForBuildIdle(1s));
  const SemanticGenerationInfo upgraded = index.activeGeneration();
  EXPECT_EQ(upgraded.model_id, "model-v2");
  EXPECT_EQ(upgraded.dimension, 3U);
  EXPECT_EQ(upgraded.rows, 2U);
  EXPECT_GT(upgraded.generation, old_token.info().generation);

  SemanticSearchRequest wrong_dimension;
  wrong_dimension.query_vector = {1, 0};
  const SemanticReadToken new_token = index.pinRead(11, 200);
  EXPECT_EQ(index.search(new_token, wrong_dimension, 201).status,
            SemanticSearchStatus::kInvalidQueryDimension);

  SemanticSearchRequest new_dimension;
  new_dimension.query_vector = {1, 1, 0};
  EXPECT_EQ(index.search(new_token, new_dimension, 201).hits.size(), 2U);

  SemanticSearchRequest old_dimension;
  old_dimension.query_vector = {1, 0};
  const SemanticSearchResponse pinned =
      index.search(old_token, old_dimension, 201);
  EXPECT_EQ(pinned.status, SemanticSearchStatus::kOk);
  EXPECT_EQ(pinned.index.model_id, "model-v1");
  EXPECT_EQ(pinned.index.dimension, 2U);
  EXPECT_EQ(pinned.hits.size(), 2U);
}

TEST(VersionedSemanticIndex, PendingLexicalMergeDeleteAndStaleJobFence) {
  VersionedSemanticIndex index(indexConfig("merge-model", 2));
  const SemanticUpsertResult canonical =
      index.upsertDocument(document(10, "table", "oak dining table", 1));
  const SemanticUpsertResult retired =
      index.upsertDocument(document(20, "stool", "plain stool", 2));
  ASSERT_EQ(canonical.jobs.size(), 1U);
  ASSERT_EQ(retired.jobs.size(), 1U);
  ASSERT_EQ(index.acceptEmbedding(recordFor(canonical.jobs.front(), {1, 0})),
            EmbeddingAcceptStatus::kAccepted);
  ASSERT_EQ(index.acceptEmbedding(recordFor(retired.jobs.front(), {0, 1})),
            EmbeddingAcceptStatus::kAccepted);
  ASSERT_TRUE(index.waitForBuildIdle(1s));

  const SemanticUpsertResult changed =
      index.upsertDocument(document(20, "stool", "striped velvet ottoman", 3));
  ASSERT_EQ(changed.jobs.size(), 1U);
  ASSERT_TRUE(index.waitForBuildIdle(1s));
  SemanticSearchRequest lexical;
  lexical.query_text = "striped ottoman";
  const SemanticReadToken pending_token = index.pinRead(3, 100);
  const SemanticSearchResponse pending =
      index.search(pending_token, lexical, 101);
  ASSERT_EQ(pending.hits.size(), 1U);
  EXPECT_EQ(pending.hits.front().object_id, 20);
  EXPECT_EQ(pending.hits.front().freshness,
            SemanticFreshness::kEmbeddingPendingLexical);
  EXPECT_EQ(pending.pending_embeddings, 1U);

  ASSERT_TRUE(index.mergeObjects(20, 10));
  ASSERT_TRUE(index.waitForBuildIdle(1s));
  EXPECT_EQ(index.acceptEmbedding(recordFor(changed.jobs.front(), {1, 1})),
            EmbeddingAcceptStatus::kRejectedObjectMissing);
  const SemanticSearchResponse after_merge =
      index.search(index.pinRead(3, 200), lexical, 201);
  EXPECT_TRUE(after_merge.hits.empty());
  // A previously pinned answer remains coherent even after the merge.
  EXPECT_EQ(index.search(pending_token, lexical, 202).hits.size(), 1U);

  ASSERT_TRUE(index.eraseObject(10));
  ASSERT_TRUE(index.waitForBuildIdle(1s));
  SemanticSearchRequest everything;
  everything.query_text = "table";
  EXPECT_TRUE(index.search(index.pinRead(4, 300), everything, 301).hits.empty());
  EXPECT_EQ(index.activeGeneration().rows, 0U);
}

}  // namespace
}  // namespace roomie
