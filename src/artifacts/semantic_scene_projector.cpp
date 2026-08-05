#include "roomie/artifacts/semantic_scene_projector.hpp"

#include <iterator>
#include <set>
#include <stdexcept>
#include <utility>

namespace roomie {
namespace {

void appendJobs(std::vector<EmbeddingJob>* destination,
                std::vector<EmbeddingJob> jobs) {
  destination->insert(destination->end(),
                      std::make_move_iterator(jobs.begin()),
                      std::make_move_iterator(jobs.end()));
}

void upsertObject(const SceneSnapshot& snapshot,
                  SceneObjectId object_id,
                  VersionedSemanticIndex* index,
                  SemanticSceneProjectionResult* result,
                  bool include_existing_missing_embeddings) {
  const SceneObjectPtr object = snapshot.findExactObject(object_id);
  if (!object || snapshot.isTombstoned(object_id)) {
    return;
  }
  SemanticUpsertResult upsert = index->upsertDocument(
      makeSemanticDocumentForObject(*object, object_id,
                                    snapshot.revision()));
  ++result->documents_upserted;
  result->documents_changed += upsert.document_changed ? 1U : 0U;
  result->metadata_changed += upsert.metadata_changed ? 1U : 0U;
  if (upsert.document_changed || include_existing_missing_embeddings) {
    appendJobs(&result->missing_embeddings, std::move(upsert.jobs));
  }
}

}  // namespace

SemanticSceneProjectionResult initializeSemanticIndexFromScene(
    const SceneSnapshot& snapshot,
    VersionedSemanticIndex* index) {
  SemanticSceneProjectionResult result;
  if (index == nullptr || !snapshot.statePtr()) {
    result.ok = false;
    result.error = "semantic index initialization requires an index and scene";
    return result;
  }
  try {
    for (const auto& [object_id, object] : snapshot.objects()) {
      if (object && !snapshot.isTombstoned(object_id)) {
        upsertObject(snapshot, object_id, index, &result, true);
      }
    }
  } catch (const std::exception& error) {
    result.ok = false;
    result.error = error.what();
  } catch (...) {
    result.ok = false;
    result.error = "semantic scene initialization failed";
  }
  return result;
}

SemanticSceneProjectionResult projectSceneCommitToSemanticIndex(
    const SceneApplyResult& commit,
    VersionedSemanticIndex* index) {
  SemanticSceneProjectionResult result;
  if (index == nullptr || !commit.snapshot.statePtr()) {
    result.ok = false;
    result.error = "semantic scene projection requires an index and snapshot";
    return result;
  }
  if (!commit.accepted() ||
      commit.status == SceneApplyStatus::kNoOp) {
    return result;
  }

  try {
    std::set<SceneObjectId> upserts;
    for (const SceneEvent& event : commit.events) {
      if (const auto* tombstone = std::get_if<ObjectTombstoned>(&event)) {
        if (index->eraseObject(tombstone->object_id)) {
          ++result.objects_erased;
        }
        upserts.erase(tombstone->object_id);
      } else if (const auto* merge = std::get_if<ObjectMerged>(&event)) {
        if (index->mergeObjects(merge->retired_object_id,
                                merge->canonical_object_id)) {
          ++result.aliases_applied;
        }
        upserts.erase(merge->retired_object_id);
        upserts.insert(merge->canonical_object_id);
      } else if (const auto* created = std::get_if<ObjectCreated>(&event)) {
        upserts.insert(created->object_id);
      } else if (const auto* updated = std::get_if<ObjectUpdated>(&event)) {
        upserts.insert(updated->object_id);
      }
    }
    for (const SceneObjectId object_id : upserts) {
      upsertObject(commit.snapshot, object_id, index, &result, false);
    }
  } catch (const std::exception& error) {
    result.ok = false;
    result.error = error.what();
  } catch (...) {
    result.ok = false;
    result.error = "semantic scene projection failed";
  }
  return result;
}

}  // namespace roomie
