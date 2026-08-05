#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "roomie/artifacts/semantic_index.hpp"
#include "roomie/scene/scene_reducer.hpp"

namespace roomie {

struct SemanticSceneProjectionResult {
  bool ok = true;
  std::size_t documents_upserted = 0;
  std::size_t documents_changed = 0;
  std::size_t metadata_changed = 0;
  std::size_t objects_erased = 0;
  std::size_t aliases_applied = 0;
  std::vector<EmbeddingJob> missing_embeddings;
  std::string error;
};

// Seeds a newly-created index from one immutable scene view. Durable embedding
// records must be restored only after this call, so acceptEmbedding() can
// reject rows whose object or document hash is no longer current.
SemanticSceneProjectionResult initializeSemanticIndexFromScene(
    const SceneSnapshot& snapshot,
    VersionedSemanticIndex* index);

// Applies the object-local effects of one reducer commit. This keeps metadata
// filters and the lexical delta live without scanning the complete scene on
// every observation. Returned jobs are diagnostic: the durable outbox remains
// the sole execution source and callers must not create a private task queue.
SemanticSceneProjectionResult projectSceneCommitToSemanticIndex(
    const SceneApplyResult& commit,
    VersionedSemanticIndex* index);

}  // namespace roomie
