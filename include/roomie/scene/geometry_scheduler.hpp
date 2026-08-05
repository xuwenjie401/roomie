#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "roomie/pipeline/surface_snapshot.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/scene/scene_command.hpp"
#include "roomie/scene/scene_event.hpp"
#include "roomie/scene/scene_snapshot.hpp"

namespace roomie {

// Thresholds consumed by the pure evaluator. They are copied into each task so
// a queued task never observes a later configuration mutation.
struct GeometryThresholds {
  float shell_thickness_m = 0.08f;
  int empty_inside_points = 6;
  int min_unique_voxels = 12;
  float confirm_score = 0.62f;
  float suppress_score = 0.35f;
};

struct GeometryObjectState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ObjectDependency dependency;
  Eigen::Vector3f center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f size_m = Eigen::Vector3f::Zero();
  float yaw_rad = 0.0f;
};

enum class GeometryTrigger : std::uint8_t {
  kObjectCreated,
  kObbChanged,
  kMapDelta,
  kExplicit,
};

// GeometryInput is a self-contained worker payload. In particular, surface is
// pinned and immutable, so evaluation never enters the reducer or map writer.
struct GeometryInput {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ObjectDependency object;
  Eigen::Vector3f center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f size_m = Eigen::Vector3f::Zero();
  float yaw_rad = 0.0f;
  GeometryThresholds thresholds;
  SurfaceSnapshotPtr surface;
  TimeNanoseconds checked_at_ns = 0;
  GeometryTrigger trigger = GeometryTrigger::kExplicit;
};

struct GeometryResult {
  GeometryDependency dependency;
  GeometryEvaluationResult evaluation;
  GeometryTrigger trigger = GeometryTrigger::kExplicit;
};

// Returns the world-axis-aligned bounds of the exact expanded OBB used by the
// evaluator. An invalid object size produces an invalid SurfaceAabb.
SurfaceAabb geometryExpandedAabb(const GeometryObjectState& object,
                                 const GeometryThresholds& thresholds);
SurfaceAabb geometryExpandedAabb(const GeometryInput& input);

// Local production evaluator: only intersecting immutable surface blocks are
// scanned, and those block indices become the dependency set in the result.
GeometryResult evaluateGeometry(const GeometryInput& input);

// Reference/full fallback used for equivalence testing. It evaluates the same
// formula over every block in the pinned snapshot.
GeometryResult evaluateGeometryFullSurface(const GeometryInput& input);

// A deliberately small spatial-index abstraction. The initial implementation
// uses a deterministic ordered table because Roomie object counts are small;
// callers are isolated from that choice and can replace it with an R-tree
// without changing scheduler/evaluator contracts.
class GeometryObjectSpatialIndex {
 public:
  void upsert(GeometryObjectState object,
              const GeometryThresholds& thresholds);
  bool erase(SceneObjectId object_id);
  void clear();

  std::optional<GeometryObjectState> find(SceneObjectId object_id) const;
  std::vector<SceneObjectId> query(const SurfaceAabb& bounds) const;
  std::vector<SceneObjectId> allObjectIds() const;
  std::size_t size() const;

 private:
  struct Entry {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GeometryObjectState object;
    SurfaceAabb expanded_aabb;
  };

  std::map<SceneObjectId, Entry> entries_;
};

struct GeometrySchedulerConfig {
  GeometryThresholds thresholds;
  // Zero is unbounded. A bounded scheduler supersedes work for the same
  // object, but applies reliable backpressure instead of evicting a distinct
  // object's convergence work.
  std::size_t pending_capacity = 0;
};

struct GeometryScheduleBatch {
  std::size_t requested = 0;
  std::size_t accepted = 0;
  std::size_t replaced = 0;
  bool conservative_full_scan = false;
  std::vector<SceneObjectId> scheduled_object_ids;
};

// Scheduler core only; worker thread ownership is intentionally left to the
// integration layer. All event handlers are safe to call from one actor or
// concurrently, and the pending channel implements LatestByObject semantics.
// Cross-object capacity pressure is reliable: schedule() may wait until a
// worker consumes an item, and stop() wakes every blocked producer.
class GeometryScheduler {
 public:
  explicit GeometryScheduler(GeometrySchedulerConfig config = {});

  PushResult<GeometryInput> schedule(GeometryInput input);
  bool tryPop(GeometryInput* input);
  bool waitPop(GeometryInput* input);
  void stop();
  ChannelStats channelStats() const;

  // Materializes current immutable work without entering the bounded channel.
  // The sole geometry consumer uses this for CAS retries so it can never
  // deadlock itself by waiting for capacity that only it can release.
  std::optional<GeometryInput> currentInput(
      SceneObjectId object_id,
      const SceneSnapshot& scene,
      SurfaceSnapshotPtr surface,
      GeometryTrigger trigger,
      std::uint64_t minimum_obb_revision = 0);

  bool onObjectCreated(const ObjectCreated& event,
                       const SceneSnapshot& scene,
                       SurfaceSnapshotPtr surface);
  bool onObbChanged(const ObbChanged& event,
                    const SceneSnapshot& scene,
                    SurfaceSnapshotPtr surface);
  GeometryScheduleBatch onMapDelta(const MapDelta& delta,
                                   SurfaceSnapshotPtr surface);
  bool onObjectMerged(const ObjectMerged& event);
  bool onObjectTombstoned(const ObjectTombstoned& event);

  // Seeds bounds for removed-block lookup without scheduling work.
  void observeSurface(const SurfaceSnapshotPtr& surface);
  std::size_t indexedObjectCount() const;

 private:
  std::optional<GeometryObjectState> objectState(
      SceneObjectId object_id, const SceneSnapshot& scene) const;
  GeometryInput makeInput(const GeometryObjectState& object,
                          SurfaceSnapshotPtr surface,
                          GeometryTrigger trigger) const;
  bool upsertAndSchedule(SceneObjectId object_id,
                         const SceneSnapshot& scene,
                         SurfaceSnapshotPtr surface,
                         GeometryTrigger trigger,
                         std::uint64_t minimum_obb_revision);

  GeometrySchedulerConfig config_;
  mutable std::mutex mutex_;
  GeometryObjectSpatialIndex object_index_;
  std::map<BlockIndex, SurfaceAabb> block_bounds_;
  BoundedChannel<GeometryInput> pending_;
};

enum class GeometryCasDecision : std::uint8_t {
  kAcceptExactSurface,
  kAcceptUnrelatedSurfaceAdvance,
  kRejectAliasCycle,
  kRejectRetiredObject,
  kRejectObjectMissing,
  kRejectIdentityRevision,
  kRejectObbRevision,
  kRejectMapEpoch,
  kRejectSurfaceRegression,
  kRejectOverlappingDelta,
  kRejectJournalGap,
};

struct GeometryCasCheck {
  GeometryCasDecision decision = GeometryCasDecision::kRejectObjectMissing;
  SceneObjectId canonical_object_id = -1;
  DeltaOverlapVerdict delta_overlap = DeltaOverlapVerdict::kUnknown;
  MapDeltaQueryResult delta_query;

  bool accepted() const {
    return decision == GeometryCasDecision::kAcceptExactSurface ||
           decision ==
               GeometryCasDecision::kAcceptUnrelatedSurfaceAdvance;
  }
};

// Performs the worker-result CAS preflight against immutable scene state and
// the authoritative bounded map-delta journal. Unknown coverage is rejected
// conservatively.
GeometryCasCheck checkGeometryCas(const GeometryResult& result,
                                  const SceneSnapshot& current_scene,
                                  const SurfaceStamp& current_surface,
                                  const MapDeltaJournal& delta_journal);

// Builds the typed reducer command only after a successful preflight.
std::optional<ApplyGeometryResultCommand> makeGeometryApplyCommand(
    const GeometryResult& result,
    const SurfaceStamp& current_surface,
    const GeometryCasCheck& check);

}  // namespace roomie
