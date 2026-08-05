#pragma once

#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "roomie/artifacts/semantic_index.hpp"
#include "roomie/query/scene_query_gateway.hpp"

namespace roomie {

// Must be backed by an already-prewarmed query encoder in the same namespace
// as VersionedSemanticIndex. Loading a model or encoding every object from a
// tool call is outside this contract.
using SemanticQueryVectorEncoder =
    std::function<std::vector<float>(std::string_view query)>;

class VersionedSemanticSearchProvider final : public SearchProvider {
 public:
  VersionedSemanticSearchProvider(
      std::shared_ptr<VersionedSemanticIndex> index,
      SemanticQueryVectorEncoder query_encoder);

  std::shared_ptr<const PinnedSearchIndex> pinCurrent() const override;

 private:
  std::shared_ptr<VersionedSemanticIndex> index_;
  SemanticQueryVectorEncoder query_encoder_;
};

}  // namespace roomie
