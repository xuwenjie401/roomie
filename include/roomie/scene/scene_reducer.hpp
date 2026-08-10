#pragma once

#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "roomie/scene/scene_command.hpp"
#include "roomie/scene/scene_event.hpp"
#include "roomie/scene/furniture_graph.hpp"
#include "roomie/scene/scene_snapshot.hpp"

namespace roomie {

// Pure schema-v3+ containment helper. A room contains an object when the
// object's current OBB center lies inside the room's XY bounds. The relation
// carries typed room/object endpoints and no duplicated parent-room field.
std::optional<ObjectRelation> deriveRoomContainmentRelation(
    const RoomNode& room,
    SceneObjectId object_id,
    const GeometryComponent& geometry,
    SceneRevision revision = 0);

enum class SceneApplyStatus {
  kCommitted,
  kMetadataUpdated,
  kNoOp,
  kRejected,
};

struct SceneApplyResult {
  SceneApplyStatus status = SceneApplyStatus::kRejected;
  SceneRevision revision = 0;
  std::string reason;
  SceneSnapshot snapshot;
  std::vector<SceneEvent> events;

  bool accepted() const { return status != SceneApplyStatus::kRejected; }
  bool committedRevision() const { return status == SceneApplyStatus::kCommitted; }
};

// ReducerCore is intentionally synchronous and single-writer. A future
// SceneReducerThread owns it and serializes channel input; this core keeps the
// state transition and stale-result rules independently testable.
class ReducerCore {
 public:
  using ObservationAssociator =
      std::function<std::vector<ObservationMutation>(
          const SceneSnapshot&, const ApplyObservationBatchCommand&)>;
  using CommitObserver = std::function<void(const SceneApplyResult&)>;

  ReducerCore();
  explicit ReducerCore(FurnitureGraphConfig furniture_config);
  explicit ReducerCore(ObservationAssociator associator);
  ReducerCore(ObservationAssociator associator,
              FurnitureGraphConfig furniture_config);

  SceneApplyResult apply(const SceneCommand& command);
  SceneSnapshot snapshot() const;

  void setObservationAssociator(ObservationAssociator associator);
  void setCommitObserver(CommitObserver observer);

 private:
  SceneApplyResult applyCommand(const LoadSceneCommand& command);
  SceneApplyResult applyCommand(const ApplyObservationBatchCommand& command);
  SceneApplyResult applyCommand(const AdvanceSurfaceCommand& command);
  SceneApplyResult applyCommand(const ApplyGeometryResultCommand& command);
  SceneApplyResult applyCommand(const ApplySnapshotSetCommand& command);
  SceneApplyResult applyCommand(const ApplyDescriptionArtifactCommand& command);
  SceneApplyResult applyCommand(const ApplyHumanAnnotationCommand& command);
  SceneApplyResult applyCommand(const RebuildFurnitureGraphCommand& command);
  SceneApplyResult applyCommand(const PersistedThroughCommand& command);
  SceneApplyResult applyCommand(const ShutdownCommand& command);

  SceneApplyResult commit(SceneState next,
                          SceneRevision revision,
                          std::vector<SceneEvent> events);
  SceneApplyResult metadataUpdate(SceneState next,
                                  std::vector<SceneEvent> events,
                                  std::string reason = {});
  SceneApplyResult noOp(std::string reason) const;
  SceneApplyResult reject(std::string reason) const;
  void notify(SceneApplyResult* result);

  std::shared_ptr<const SceneState> state_;
  FurnitureGraphConfig furniture_config_;
  ObservationAssociator observation_associator_;
  CommitObserver commit_observer_;
  std::optional<std::thread::id> owner_thread_;
  bool applying_ = false;
};

}  // namespace roomie
