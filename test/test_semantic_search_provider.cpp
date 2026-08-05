#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "roomie/query/semantic_search_provider.hpp"

namespace roomie {
namespace {

using namespace std::chrono_literals;

SemanticDocument document(SceneObjectId object_id,
                          std::string text,
                          SceneRevision revision) {
  SemanticDocumentInput input;
  input.object_id = object_id;
  input.label = std::move(text);
  input.semantic_revision = revision;
  input.created_scene_revision = revision;
  return makeSemanticDocument(input);
}

EmbeddingRecord record(const EmbeddingJob& job,
                       std::vector<float> vector) {
  EmbeddingRecord result;
  result.object_id = job.object_id;
  result.document_hash = job.document_hash;
  result.model_id = job.name_space.model_id;
  result.vector = std::move(vector);
  result.created_scene_revision = job.created_scene_revision;
  return result;
}

TEST(VersionedSemanticSearchProvider,
     PinsGenerationHashesAndUsesOnlyCurrentVectorRows) {
  SemanticIndexConfig config;
  config.initial_namespace = {"provider-test", 2};
  auto index = std::make_shared<VersionedSemanticIndex>(config);
  const SemanticUpsertResult chair =
      index->upsertDocument(document(1, "red chair", 1));
  const SemanticUpsertResult lamp =
      index->upsertDocument(document(2, "blue lamp", 2));
  ASSERT_EQ(chair.jobs.size(), 1U);
  ASSERT_EQ(lamp.jobs.size(), 1U);
  ASSERT_EQ(index->acceptEmbedding(record(chair.jobs.front(), {1.0f, 0.0f})),
            EmbeddingAcceptStatus::kAccepted);
  ASSERT_EQ(index->acceptEmbedding(record(lamp.jobs.front(), {0.0f, 1.0f})),
            EmbeddingAcceptStatus::kAccepted);
  ASSERT_TRUE(index->waitForBuildIdle(1s));

  auto provider = std::make_shared<VersionedSemanticSearchProvider>(
      index, [](std::string_view query) {
        return query.find("chair") != std::string_view::npos
                   ? std::vector<float>{1.0f, 0.0f}
                   : std::vector<float>{0.0f, 1.0f};
      });
  const std::shared_ptr<const PinnedSearchIndex> pinned =
      provider->pinCurrent();
  ASSERT_TRUE(pinned);
  EXPECT_GT(pinned->generation(), 0U);
  EXPECT_EQ(pinned->documentHash(1), chair.jobs.front().document_hash);
  EXPECT_EQ(pinned->documentHash(2), lamp.jobs.front().document_hash);

  const std::vector<IndexedSearchHit> hits =
      pinned->search("chair", 2);
  ASSERT_EQ(hits.size(), 2U);
  EXPECT_EQ(hits.front().object_id, 1);
  EXPECT_EQ(hits.front().document_hash, chair.jobs.front().document_hash);

  const SemanticUpsertResult changed =
      index->upsertDocument(document(1, "striped chair", 3));
  ASSERT_TRUE(changed.document_changed);
  ASSERT_EQ(changed.jobs.size(), 1U);
  ASSERT_TRUE(index->waitForBuildIdle(1s));

  // An already-pinned generation remains coherent. A newly pinned provider
  // still exposes the old row hash until the replacement vector is accepted.
  EXPECT_EQ(pinned->documentHash(1), chair.jobs.front().document_hash);
  const auto pending = provider->pinCurrent();
  EXPECT_FALSE(pending->documentHash(1).has_value());
  const auto pending_hits = pending->search("striped chair", 4);
  EXPECT_TRUE(std::none_of(
      pending_hits.begin(), pending_hits.end(), [](const IndexedSearchHit& hit) {
        return hit.object_id == 1;
      }));
}

}  // namespace
}  // namespace roomie
