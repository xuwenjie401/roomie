#include "roomie/query/semantic_search_provider.hpp"

#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace roomie {
namespace {

std::int64_t unixMillis() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

class PinnedVersionedSemanticIndex final : public PinnedSearchIndex {
 public:
  PinnedVersionedSemanticIndex(
      std::shared_ptr<VersionedSemanticIndex> index,
      SemanticReadToken token,
      SemanticQueryVectorEncoder query_encoder)
      : index_(std::move(index)),
        token_(std::move(token)),
        query_encoder_(std::move(query_encoder)) {}

  std::uint64_t generation() const override {
    return token_.info().generation;
  }

  std::optional<std::string> documentHash(
      SceneObjectId object_id) const override {
    return index_->indexedDocumentHash(token_, object_id);
  }

  std::vector<IndexedSearchHit> search(
      std::string_view query,
      std::size_t limit) const override {
    if (query.empty() || limit == 0) {
      return {};
    }
    SemanticSearchRequest request;
    request.query_text = std::string(query);
    request.query_vector = query_encoder_(query);
    request.top_k = limit;
    const SemanticSearchResponse searched =
        index_->search(token_, request, unixMillis());
    if (searched.status != SemanticSearchStatus::kOk) {
      throw std::runtime_error("pinned semantic index search failed");
    }

    std::vector<IndexedSearchHit> hits;
    hits.reserve(searched.hits.size());
    for (const SemanticSearchHit& hit : searched.hits) {
      // Pending documents are intentionally omitted here. SceneQueryGateway
      // owns the pinned lexical-delta fallback and reports that freshness.
      if (hit.freshness != SemanticFreshness::kCurrentEmbedding) {
        continue;
      }
      hits.push_back(
          IndexedSearchHit{hit.object_id, hit.score, hit.document_hash});
    }
    return hits;
  }

 private:
  std::shared_ptr<VersionedSemanticIndex> index_;
  SemanticReadToken token_;
  SemanticQueryVectorEncoder query_encoder_;
};

}  // namespace

VersionedSemanticSearchProvider::VersionedSemanticSearchProvider(
    std::shared_ptr<VersionedSemanticIndex> index,
    SemanticQueryVectorEncoder query_encoder)
    : index_(std::move(index)),
      query_encoder_(std::move(query_encoder)) {
  if (!index_ || !query_encoder_) {
    throw std::invalid_argument(
        "semantic search provider requires an index and prewarmed encoder");
  }
}

std::shared_ptr<const PinnedSearchIndex>
VersionedSemanticSearchProvider::pinCurrent() const {
  // The gateway owns user-visible TTL. A semantic token is bounded by this
  // immutable provider object, so give it a saturating wall-clock horizon.
  SemanticReadToken token = index_->pinRead(
      0, 0, std::numeric_limits<std::int64_t>::max());
  return std::make_shared<const PinnedVersionedSemanticIndex>(
      index_, std::move(token), query_encoder_);
}

}  // namespace roomie
