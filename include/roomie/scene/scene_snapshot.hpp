#pragma once

#include <memory>
#include <optional>
#include <set>
#include <utility>

#include "roomie/scene/scene_state.hpp"

namespace roomie {

class SceneSnapshot {
 public:
  SceneSnapshot() : state_(std::make_shared<const SceneState>()) {}
  explicit SceneSnapshot(std::shared_ptr<const SceneState> state)
      : state_(state ? std::move(state) : std::make_shared<const SceneState>()) {}

  SceneRevision revision() const { return state_->latest_scene_revision; }
  SceneRevision durableRevision() const { return state_->durable_scene_revision; }
  SceneObjectId nextObjectId() const { return state_->next_object_id; }
  const SurfaceStamp& latestSurface() const { return state_->latest_surface; }
  bool shutdownRequested() const { return state_->shutdown_requested; }

  const SceneObjectTable& objects() const { return *state_->objects; }
  const SceneAliasTable& aliases() const { return *state_->aliases; }
  const SceneTombstoneTable& tombstones() const { return *state_->tombstones; }
  const SceneTrackTable& tracks() const { return *state_->tracks; }
  const SceneGraphMetadata& graphMetadata() const { return *state_->graph; }

  std::optional<SceneObjectId> resolveCanonicalId(SceneObjectId object_id) const {
    std::set<SceneObjectId> visited;
    SceneObjectId current = object_id;
    while (true) {
      if (!visited.insert(current).second) {
        return std::nullopt;
      }
      auto alias_it = state_->aliases->find(current);
      if (alias_it == state_->aliases->end()) {
        return current;
      }
      current = alias_it->second.canonical_object_id;
    }
  }

  bool isTombstoned(SceneObjectId object_id) const {
    const auto canonical = resolveCanonicalId(object_id);
    return canonical && state_->tombstones->count(*canonical) != 0;
  }

  SceneObjectPtr findExactObject(SceneObjectId object_id) const {
    const auto it = state_->objects->find(object_id);
    return it == state_->objects->end() ? nullptr : it->second;
  }

  SceneObjectPtr findObject(SceneObjectId object_id) const {
    const auto canonical = resolveCanonicalId(object_id);
    if (!canonical || state_->tombstones->count(*canonical) != 0) {
      return nullptr;
    }
    return findExactObject(*canonical);
  }

  const std::shared_ptr<const SceneState>& statePtr() const { return state_; }

  // Compatibility materialization for legacy ObjectGraph readers. The
  // returned value is detached; mutating it cannot mutate this snapshot.
  ObjectGraphSnapshot materializeObjectGraph() const;

 private:
  std::shared_ptr<const SceneState> state_;
};

}  // namespace roomie
