#include "roomie/artifacts/semantic_index_store_adapter.hpp"

#include <utility>

namespace roomie {

SceneStoreStatus persistSemanticEmbeddingRecord(
    SceneStore* store,
    const EmbeddingRecord& record) {
  if (store == nullptr) {
    return SceneStoreStatus::failure(
        "semantic embedding persistence requires a SceneStore");
  }
  DurableEmbeddingRecord durable;
  durable.object_id = record.object_id;
  durable.document_hash = record.document_hash;
  durable.model_id = record.model_id;
  durable.dimension = record.vector.size();
  durable.vector = record.vector;
  durable.created_scene_revision = record.created_scene_revision;
  return store->upsertEmbeddingRecord(std::move(durable));
}

SemanticEmbeddingRestoreResult restoreSemanticIndexEmbeddingRecords(
    SceneStore* store,
    VersionedSemanticIndex* index) {
  SemanticEmbeddingRestoreResult result;
  if (store == nullptr || index == nullptr) {
    result.status = SceneStoreStatus::failure(
        "semantic embedding restore requires a store and index");
    return result;
  }

  const SemanticGenerationInfo active = index->activeGeneration();
  result.name_space = {active.model_id, active.dimension};
  if (result.name_space.model_id.empty() || result.name_space.dimension == 0) {
    result.status = SceneStoreStatus::failure(
        "semantic index has an invalid active namespace");
    return result;
  }

  EmbeddingRecordListResult listed = store->listEmbeddingRecords(
      result.name_space.model_id, result.name_space.dimension);
  result.status = listed.status;
  if (!listed.status) {
    return result;
  }
  result.records_scanned = listed.records.size();
  for (DurableEmbeddingRecord& durable : listed.records) {
    EmbeddingRecord record;
    record.object_id = durable.object_id;
    record.document_hash = std::move(durable.document_hash);
    record.model_id = std::move(durable.model_id);
    record.vector = std::move(durable.vector);
    record.created_scene_revision = durable.created_scene_revision;
    switch (index->acceptEmbedding(std::move(record))) {
      case EmbeddingAcceptStatus::kAccepted:
        ++result.accepted;
        break;
      case EmbeddingAcceptStatus::kRejectedObjectMissing:
        ++result.rejected_object_missing;
        break;
      case EmbeddingAcceptStatus::kRejectedDocumentStale:
        ++result.rejected_document_stale;
        break;
      case EmbeddingAcceptStatus::kRejectedNamespace:
        ++result.rejected_namespace;
        break;
      case EmbeddingAcceptStatus::kRejectedDimension:
        ++result.rejected_dimension;
        break;
      case EmbeddingAcceptStatus::kRejectedInvalidVector:
        ++result.rejected_invalid_vector;
        break;
    }
  }
  return result;
}

}  // namespace roomie
