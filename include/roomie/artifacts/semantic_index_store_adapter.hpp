#pragma once

#include <cstddef>

#include "roomie/artifacts/semantic_index.hpp"
#include "roomie/scene/scene_store.hpp"

namespace roomie {

struct SemanticEmbeddingRestoreResult {
  SceneStoreStatus status;
  EmbeddingNamespace name_space;
  std::size_t records_scanned = 0;
  std::size_t accepted = 0;
  std::size_t rejected_object_missing = 0;
  std::size_t rejected_document_stale = 0;
  std::size_t rejected_namespace = 0;
  std::size_t rejected_dimension = 0;
  std::size_t rejected_invalid_vector = 0;
};

// Converts the in-memory worker envelope to SceneStore's explicit durable
// envelope. SceneStore provides the idempotency and durable-revision fence.
SceneStoreStatus persistSemanticEmbeddingRecord(
    SceneStore* store,
    const EmbeddingRecord& record);

// Restores only the index's active (model_id, dimension) namespace. Documents
// must be upserted before this call; acceptEmbedding() then rejects stale hashes
// and missing/retired objects before any generation can contain them.
SemanticEmbeddingRestoreResult restoreSemanticIndexEmbeddingRecords(
    SceneStore* store,
    VersionedSemanticIndex* index);

}  // namespace roomie
