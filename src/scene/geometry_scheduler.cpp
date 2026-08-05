#include "roomie/scene/geometry_scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

namespace roomie {
namespace {

constexpr float kEpsilon = 1.0e-6f;

float clamp01(float value) {
  return std::max(0.0f, std::min(1.0f, value));
}

float finiteNonnegative(float value) {
  return std::isfinite(value) ? std::max(0.0f, value) : 0.0f;
}

GeometryObjectState objectStateFor(const GeometryInput& input) {
  GeometryObjectState object;
  object.dependency = input.object;
  object.center_world = input.center_world;
  object.size_m = input.size_m;
  object.yaw_rad = input.yaw_rad;
  return object;
}

std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>
evaluatedBlockIndices(const std::vector<SurfaceBlockPtr>& blocks) {
  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> result;
  result.reserve(blocks.size());
  for (const SurfaceBlockPtr& block : blocks) {
    if (block) {
      result.push_back(block->index().eigen());
    }
  }
  return result;
}

GeometryResult evaluateBlocks(const GeometryInput& input,
                              const std::vector<SurfaceBlockPtr>& blocks) {
  GeometryResult result;
  result.dependency.object = input.object;
  result.trigger = input.trigger;
  if (input.surface) {
    result.dependency.surface = input.surface->surfaceStamp();
  }
  result.dependency.evaluated_blocks = evaluatedBlockIndices(blocks);

  GeometryEvaluationResult& evaluation = result.evaluation;
  evaluation.checked_at_ns = input.checked_at_ns;
  evaluation.evaluated_center_world = input.center_world;
  evaluation.evaluated_size_m = input.size_m;
  evaluation.evaluated_yaw_rad = input.yaw_rad;

  const bool valid_obb = input.center_world.allFinite() &&
                         input.size_m.allFinite() &&
                         (input.size_m.array() > 0.0f).all() &&
                         std::isfinite(input.yaw_rad);
  if (!valid_obb || !input.surface || input.surface->empty()) {
    evaluation.status = InstanceGeometryStatus::kEmpty;
    evaluation.reason = "no_surface_points";
    return result;
  }

  const Eigen::Vector3f half = 0.5f * input.size_m;
  const float min_half = std::max(kEpsilon, half.minCoeff());
  const float shell = std::min(
      finiteNonnegative(input.thresholds.shell_thickness_m),
      0.85f * min_half);
  const Eigen::Vector3f inner_half =
      (half.array() - shell).max(0.0f).matrix();
  const Eigen::Vector3f expanded_half =
      (half.array() + std::max(shell, 0.02f)).matrix();
  const SurfaceAabb expanded_world_aabb = geometryExpandedAabb(input);
  const float c = std::cos(-input.yaw_rad);
  const float s = std::sin(-input.yaw_rad);

  Eigen::Vector3f local_min =
      Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
  Eigen::Vector3f local_max =
      Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
  std::set<std::tuple<int, int, int, int, int, int>> unique_voxels;
  int cavity_points = 0;

  for (const SurfaceBlockPtr& block : blocks) {
    if (!block) {
      continue;
    }
    for (const MapSurfacePoint& surface_point : block->points()) {
      const Eigen::Vector3f& point_world = surface_point.position_world;
      if (!point_world.allFinite() ||
          !expanded_world_aabb.contains(point_world)) {
        continue;
      }
      const Eigen::Vector3f delta = point_world - input.center_world;
      const Eigen::Vector3f local(c * delta.x() - s * delta.y(),
                                  s * delta.x() + c * delta.y(),
                                  delta.z());
      const Eigen::Vector3f abs_local = local.cwiseAbs();
      if (!(abs_local.array() <= expanded_half.array()).all()) {
        continue;
      }
      ++evaluation.expanded_points;
      if (!(abs_local.array() <= half.array()).all()) {
        continue;
      }
      ++evaluation.in_box_points;
      if (surface_point.has_voxel_ref) {
        unique_voxels.emplace(
            surface_point.voxel_ref.block_index.x(),
            surface_point.voxel_ref.block_index.y(),
            surface_point.voxel_ref.block_index.z(),
            surface_point.voxel_ref.voxel_index.x(),
            surface_point.voxel_ref.voxel_index.y(),
            surface_point.voxel_ref.voxel_index.z());
      }
      local_min = local_min.cwiseMin(local);
      local_max = local_max.cwiseMax(local);
      if ((abs_local.array() < inner_half.array()).all()) {
        ++cavity_points;
      } else {
        ++evaluation.shell_points;
      }
    }
  }

  evaluation.unique_voxels =
      unique_voxels.empty() ? evaluation.in_box_points
                            : static_cast<int>(unique_voxels.size());
  if (evaluation.in_box_points <= 0) {
    evaluation.status = InstanceGeometryStatus::kEmpty;
    evaluation.reason = "empty_box";
    return result;
  }

  evaluation.shell_ratio =
      static_cast<float>(evaluation.shell_points) /
      static_cast<float>(evaluation.in_box_points);
  evaluation.cavity_ratio =
      static_cast<float>(cavity_points) /
      static_cast<float>(evaluation.in_box_points);
  const Eigen::Vector3f occupied_extent =
      (local_max - local_min).cwiseMax(Eigen::Vector3f::Zero());
  const Eigen::Vector3f extent_ratio = occupied_extent.cwiseQuotient(
      input.size_m.cwiseMax(Eigen::Vector3f::Constant(kEpsilon)));
  const float mean_extent_ratio = clamp01(
      (extent_ratio.x() + extent_ratio.y() + extent_ratio.z()) / 3.0f);
  evaluation.extent_score =
      clamp01((mean_extent_ratio - 0.35f) / 0.55f);
  evaluation.leak_ratio =
      evaluation.expanded_points > 0
          ? static_cast<float>(evaluation.expanded_points -
                               evaluation.in_box_points) /
                static_cast<float>(evaluation.expanded_points)
          : 1.0f;
  const int min_unique_voxels =
      std::max(1, input.thresholds.min_unique_voxels);
  const float density_score =
      clamp01(static_cast<float>(evaluation.unique_voxels) /
              static_cast<float>(min_unique_voxels * 3));
  evaluation.score = clamp01(
      0.35f * evaluation.shell_ratio +
      0.30f * evaluation.extent_score + 0.25f * density_score +
      0.10f * (1.0f - evaluation.leak_ratio) -
      0.10f * evaluation.cavity_ratio);
  evaluation.reason = "evaluated";

  if (evaluation.in_box_points <
          std::max(0, input.thresholds.empty_inside_points) ||
      evaluation.unique_voxels <
          std::max(0, input.thresholds.min_unique_voxels)) {
    evaluation.status = InstanceGeometryStatus::kEmpty;
  } else if (evaluation.score >= clamp01(input.thresholds.confirm_score)) {
    evaluation.status = InstanceGeometryStatus::kGood;
  } else if (evaluation.score < clamp01(input.thresholds.suppress_score)) {
    evaluation.status = InstanceGeometryStatus::kBad;
  } else {
    // The legacy path used mutable lifecycle state to resolve this middle
    // band. A pure evaluator leaves it explicitly undecided.
    evaluation.status = InstanceGeometryStatus::kUnchecked;
  }
  return result;
}

std::vector<BlockIndex> dependencyBlocks(const GeometryDependency& dependency) {
  std::vector<BlockIndex> result;
  result.reserve(dependency.evaluated_blocks.size());
  for (const Eigen::Vector3i& index : dependency.evaluated_blocks) {
    result.emplace_back(index);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

bool surfaceRegressed(const SurfaceStamp& source,
                      const SurfaceStamp& current) {
  return current.surface_revision <= source.surface_revision ||
         current.source_map_revision < source.source_map_revision;
}

}  // namespace

SurfaceAabb geometryExpandedAabb(const GeometryObjectState& object,
                                 const GeometryThresholds& thresholds) {
  if (!object.center_world.allFinite() || !object.size_m.allFinite() ||
      (object.size_m.array() <= 0.0f).any() ||
      !std::isfinite(object.yaw_rad)) {
    return SurfaceAabb();
  }
  const Eigen::Vector3f half = 0.5f * object.size_m;
  const float min_half = std::max(kEpsilon, half.minCoeff());
  const float shell = std::min(
      finiteNonnegative(thresholds.shell_thickness_m), 0.85f * min_half);
  const Eigen::Vector3f expanded_half =
      (half.array() + std::max(shell, 0.02f)).matrix();
  const float c = std::abs(std::cos(object.yaw_rad));
  const float s = std::abs(std::sin(object.yaw_rad));
  const Eigen::Vector3f world_half(
      c * expanded_half.x() + s * expanded_half.y(),
      s * expanded_half.x() + c * expanded_half.y(),
      expanded_half.z());
  return SurfaceAabb(object.center_world - world_half,
                     object.center_world + world_half);
}

SurfaceAabb geometryExpandedAabb(const GeometryInput& input) {
  return geometryExpandedAabb(objectStateFor(input), input.thresholds);
}

GeometryResult evaluateGeometry(const GeometryInput& input) {
  if (!input.surface) {
    return evaluateBlocks(input, {});
  }
  const SurfaceAabb query = geometryExpandedAabb(input);
  const std::vector<SurfaceBlockPtr> blocks =
      query.valid() ? input.surface->blocksIntersecting(query)
                    : std::vector<SurfaceBlockPtr>{};
  return evaluateBlocks(input, blocks);
}

GeometryResult evaluateGeometryFullSurface(const GeometryInput& input) {
  return evaluateBlocks(input,
                        input.surface ? input.surface->blockView()
                                      : std::vector<SurfaceBlockPtr>{});
}

void GeometryObjectSpatialIndex::upsert(
    GeometryObjectState object, const GeometryThresholds& thresholds) {
  if (object.dependency.object_id < 0) {
    return;
  }
  Entry entry;
  entry.expanded_aabb = geometryExpandedAabb(object, thresholds);
  entry.object = std::move(object);
  entries_[entry.object.dependency.object_id] = std::move(entry);
}

bool GeometryObjectSpatialIndex::erase(SceneObjectId object_id) {
  return entries_.erase(object_id) > 0;
}

void GeometryObjectSpatialIndex::clear() { entries_.clear(); }

std::optional<GeometryObjectState> GeometryObjectSpatialIndex::find(
    SceneObjectId object_id) const {
  const auto it = entries_.find(object_id);
  return it == entries_.end()
             ? std::optional<GeometryObjectState>{}
             : std::optional<GeometryObjectState>{it->second.object};
}

std::vector<SceneObjectId> GeometryObjectSpatialIndex::query(
    const SurfaceAabb& bounds) const {
  std::vector<SceneObjectId> result;
  if (!bounds.valid()) {
    return result;
  }
  for (const auto& entry : entries_) {
    if (entry.second.expanded_aabb.intersects(bounds)) {
      result.push_back(entry.first);
    }
  }
  return result;
}

std::vector<SceneObjectId> GeometryObjectSpatialIndex::allObjectIds() const {
  std::vector<SceneObjectId> result;
  result.reserve(entries_.size());
  for (const auto& entry : entries_) {
    result.push_back(entry.first);
  }
  return result;
}

std::size_t GeometryObjectSpatialIndex::size() const {
  return entries_.size();
}

GeometryScheduler::GeometryScheduler(GeometrySchedulerConfig config)
    : config_(std::move(config)),
      pending_(config_.pending_capacity,
               ChannelPolicy::kLatestByKey,
               [](const GeometryInput& lhs, const GeometryInput& rhs) {
                 return lhs.object.object_id == rhs.object.object_id;
               }) {}

PushResult<GeometryInput> GeometryScheduler::schedule(GeometryInput input) {
  return pending_.pushLatestByKeyReliably(std::move(input));
}

bool GeometryScheduler::tryPop(GeometryInput* input) {
  return input != nullptr && pending_.tryPop(input);
}

bool GeometryScheduler::waitPop(GeometryInput* input) {
  return input != nullptr && pending_.waitPop(input);
}

void GeometryScheduler::stop() { pending_.stop(); }

ChannelStats GeometryScheduler::channelStats() const {
  return pending_.stats();
}

std::optional<GeometryInput> GeometryScheduler::currentInput(
    SceneObjectId object_id,
    const SceneSnapshot& scene,
    SurfaceSnapshotPtr surface,
    GeometryTrigger trigger,
    std::uint64_t minimum_obb_revision) {
  if (!surface) {
    return std::nullopt;
  }
  const std::optional<GeometryObjectState> object =
      objectState(object_id, scene);
  if (!object || object->dependency.obb_revision < minimum_obb_revision) {
    return std::nullopt;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    object_index_.upsert(*object, config_.thresholds);
  }
  return makeInput(*object, std::move(surface), trigger);
}

std::optional<GeometryObjectState> GeometryScheduler::objectState(
    SceneObjectId object_id, const SceneSnapshot& scene) const {
  const SceneObjectPtr object = scene.findObject(object_id);
  if (!object || !object->identity || !object->geometry) {
    return std::nullopt;
  }
  GeometryObjectState state;
  state.dependency = dependencyFor(*object);
  state.center_world = object->geometry->center_world;
  state.size_m = object->geometry->size_m;
  state.yaw_rad = object->geometry->yaw_rad;
  return state;
}

GeometryInput GeometryScheduler::makeInput(
    const GeometryObjectState& object,
    SurfaceSnapshotPtr surface,
    GeometryTrigger trigger) const {
  GeometryInput input;
  input.object = object.dependency;
  input.center_world = object.center_world;
  input.size_m = object.size_m;
  input.yaw_rad = object.yaw_rad;
  input.thresholds = config_.thresholds;
  input.checked_at_ns =
      surface ? surface->mapStamp().integrated_through_ns : 0;
  input.surface = std::move(surface);
  input.trigger = trigger;
  return input;
}

bool GeometryScheduler::upsertAndSchedule(
    SceneObjectId object_id,
    const SceneSnapshot& scene,
    SurfaceSnapshotPtr surface,
    GeometryTrigger trigger,
    std::uint64_t minimum_obb_revision) {
  std::optional<GeometryInput> input = currentInput(
      object_id, scene, std::move(surface), trigger, minimum_obb_revision);
  return input && schedule(std::move(*input)).accepted();
}

bool GeometryScheduler::onObjectCreated(
    const ObjectCreated& event,
    const SceneSnapshot& scene,
    SurfaceSnapshotPtr surface) {
  return upsertAndSchedule(event.object_id,
                           scene,
                           std::move(surface),
                           GeometryTrigger::kObjectCreated,
                           0);
}

bool GeometryScheduler::onObbChanged(const ObbChanged& event,
                                     const SceneSnapshot& scene,
                                     SurfaceSnapshotPtr surface) {
  return upsertAndSchedule(event.object_id,
                           scene,
                           std::move(surface),
                           GeometryTrigger::kObbChanged,
                           event.obb_revision);
}

GeometryScheduleBatch GeometryScheduler::onMapDelta(
    const MapDelta& delta, SurfaceSnapshotPtr surface) {
  GeometryScheduleBatch batch;
  if (!surface) {
    return batch;
  }

  std::vector<GeometryObjectState> objects;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool invalid_delta = !delta.valid() || !delta.complete ||
                               delta.full_rebuild ||
                               delta.to_surface != surface->surfaceStamp();
    std::set<SceneObjectId> object_ids;
    bool unknown_block_bounds = false;

    if (!invalid_delta) {
      for (const BlockIndex& index : delta.touchedBlocks()) {
        bool found_bounds = false;
        const auto old_bounds = block_bounds_.find(index);
        if (old_bounds != block_bounds_.end()) {
          found_bounds = true;
          const std::vector<SceneObjectId> matches =
              object_index_.query(old_bounds->second);
          object_ids.insert(matches.begin(), matches.end());
        }
        const SurfaceBlockPtr current_block = surface->block(index);
        if (current_block) {
          found_bounds = true;
          const std::vector<SceneObjectId> matches =
              object_index_.query(current_block->aabb());
          object_ids.insert(matches.begin(), matches.end());
        }
        unknown_block_bounds = unknown_block_bounds || !found_bounds;
      }
    }

    if (invalid_delta || unknown_block_bounds) {
      batch.conservative_full_scan = true;
      const std::vector<SceneObjectId> all = object_index_.allObjectIds();
      object_ids.insert(all.begin(), all.end());
    }

    if (invalid_delta || delta.to_surface.map_epoch !=
                             surface->surfaceStamp().map_epoch) {
      block_bounds_.clear();
      for (const SurfaceBlockPtr& block : surface->blockView()) {
        block_bounds_[block->index()] = block->aabb();
      }
    } else {
      for (const BlockIndex& index : delta.changed_blocks) {
        const SurfaceBlockPtr block = surface->block(index);
        if (block) {
          block_bounds_[index] = block->aabb();
        } else {
          block_bounds_.erase(index);
        }
      }
      for (const BlockIndex& index : delta.removed_blocks) {
        block_bounds_.erase(index);
      }
    }

    objects.reserve(object_ids.size());
    for (const SceneObjectId object_id : object_ids) {
      const std::optional<GeometryObjectState> object =
          object_index_.find(object_id);
      if (object) {
        objects.push_back(*object);
      }
    }
  }

  batch.requested = objects.size();
  batch.scheduled_object_ids.reserve(objects.size());
  for (const GeometryObjectState& object : objects) {
    PushResult<GeometryInput> pushed = schedule(
        makeInput(object, surface, GeometryTrigger::kMapDelta));
    if (!pushed.accepted()) {
      continue;
    }
    ++batch.accepted;
    if (pushed.outcome == PushOutcome::kReplaced) {
      ++batch.replaced;
    }
    batch.scheduled_object_ids.push_back(object.dependency.object_id);
  }
  return batch;
}

bool GeometryScheduler::onObjectMerged(const ObjectMerged& event) {
  std::lock_guard<std::mutex> lock(mutex_);
  return object_index_.erase(event.retired_object_id);
}

bool GeometryScheduler::onObjectTombstoned(
    const ObjectTombstoned& event) {
  std::lock_guard<std::mutex> lock(mutex_);
  return object_index_.erase(event.object_id);
}

void GeometryScheduler::observeSurface(const SurfaceSnapshotPtr& surface) {
  if (!surface) {
    return;
  }
  std::map<BlockIndex, SurfaceAabb> bounds;
  for (const SurfaceBlockPtr& block : surface->blockView()) {
    bounds[block->index()] = block->aabb();
  }
  std::lock_guard<std::mutex> lock(mutex_);
  block_bounds_ = std::move(bounds);
}

std::size_t GeometryScheduler::indexedObjectCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return object_index_.size();
}

GeometryCasCheck checkGeometryCas(
    const GeometryResult& result,
    const SceneSnapshot& current_scene,
    const SurfaceStamp& current_surface,
    const MapDeltaJournal& delta_journal) {
  GeometryCasCheck check;
  const ObjectDependency& dependency = result.dependency.object;
  const std::optional<SceneObjectId> canonical =
      current_scene.resolveCanonicalId(dependency.object_id);
  if (!canonical) {
    check.decision = GeometryCasDecision::kRejectAliasCycle;
    return check;
  }
  check.canonical_object_id = *canonical;
  if (*canonical != dependency.object_id) {
    check.decision = GeometryCasDecision::kRejectRetiredObject;
    return check;
  }
  const SceneObjectPtr object = current_scene.findExactObject(*canonical);
  if (!object || current_scene.isTombstoned(*canonical) ||
      !object->identity || !object->geometry) {
    check.decision = GeometryCasDecision::kRejectObjectMissing;
    return check;
  }
  if (object->identity->revision != dependency.identity_revision) {
    check.decision = GeometryCasDecision::kRejectIdentityRevision;
    return check;
  }
  if (object->geometry->obb_revision != dependency.obb_revision) {
    check.decision = GeometryCasDecision::kRejectObbRevision;
    return check;
  }

  const SurfaceStamp& source_surface = result.dependency.surface;
  if (!source_surface.map_epoch.valid() ||
      !current_surface.map_epoch.valid() ||
      source_surface.map_epoch != current_surface.map_epoch ||
      (current_scene.latestSurface().map_epoch.valid() &&
       current_scene.latestSurface().map_epoch != current_surface.map_epoch)) {
    check.decision = GeometryCasDecision::kRejectMapEpoch;
    return check;
  }
  const SurfaceStamp& scene_surface = current_scene.latestSurface();
  if (source_surface.surface_revision == 0 ||
      current_surface.surface_revision == 0 ||
      (scene_surface.map_epoch.valid() &&
       (current_surface.surface_revision <
            scene_surface.surface_revision ||
        current_surface.source_map_revision <
            scene_surface.source_map_revision ||
        (current_surface.surface_revision ==
             scene_surface.surface_revision &&
         current_surface != scene_surface)))) {
    check.decision = GeometryCasDecision::kRejectSurfaceRegression;
    return check;
  }

  if (source_surface == current_surface) {
    check.decision = GeometryCasDecision::kAcceptExactSurface;
    check.delta_overlap = DeltaOverlapVerdict::kExactSurface;
    check.delta_query.status = MapDeltaQueryStatus::kNoIntersection;
    check.delta_query.unknown_reason = MapDeltaUnknownReason::kNone;
    check.delta_query.covered_through_revision =
        current_surface.surface_revision;
    return check;
  }
  if (surfaceRegressed(source_surface, current_surface)) {
    check.decision = GeometryCasDecision::kRejectSurfaceRegression;
    return check;
  }

  check.delta_query = delta_journal.query(
      source_surface,
      current_surface,
      dependencyBlocks(result.dependency));
  if (check.delta_query.status == MapDeltaQueryStatus::kNoIntersection) {
    check.decision =
        GeometryCasDecision::kAcceptUnrelatedSurfaceAdvance;
    check.delta_overlap = DeltaOverlapVerdict::kNoOverlap;
  } else if (check.delta_query.status ==
             MapDeltaQueryStatus::kIntersection) {
    check.decision = GeometryCasDecision::kRejectOverlappingDelta;
    check.delta_overlap = DeltaOverlapVerdict::kOverlap;
  } else {
    check.decision = GeometryCasDecision::kRejectJournalGap;
    check.delta_overlap = DeltaOverlapVerdict::kJournalGap;
  }
  return check;
}

std::optional<ApplyGeometryResultCommand> makeGeometryApplyCommand(
    const GeometryResult& result,
    const SurfaceStamp& current_surface,
    const GeometryCasCheck& check) {
  if (!check.accepted()) {
    return std::nullopt;
  }
  ApplyGeometryResultCommand command;
  command.dependency = result.dependency;
  command.current_surface = current_surface;
  command.delta_overlap = check.delta_overlap;
  command.result = result.evaluation;
  return command;
}

}  // namespace roomie
