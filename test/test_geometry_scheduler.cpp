#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <initializer_list>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "roomie/scene/geometry_scheduler.hpp"

namespace roomie {
namespace {

RunId epoch(std::uint64_t low = 1) { return RunId{41, low}; }

MapStamp mapStamp(const RunId& map_epoch, std::uint64_t revision) {
  MapStamp stamp;
  stamp.map_epoch = map_epoch;
  stamp.map_revision = revision;
  stamp.integrated_through_ns = static_cast<TimeNanoseconds>(revision * 1000);
  return stamp;
}

SurfaceStamp surfaceStamp(const RunId& map_epoch, std::uint64_t revision) {
  SurfaceStamp stamp;
  stamp.map_epoch = map_epoch;
  stamp.surface_revision = revision;
  stamp.source_map_revision = revision;
  return stamp;
}

MapSurfacePoint point(const BlockIndex& block,
                      int voxel,
                      float x,
                      float y,
                      float z) {
  MapSurfacePoint result;
  result.position_world = Eigen::Vector3f(x, y, z);
  result.weight = 1.0f;
  result.has_voxel_ref = true;
  result.voxel_ref.block_index = block.eigen();
  result.voxel_ref.voxel_index = Eigen::Vector3i(voxel, 0, 0);
  return result;
}

SurfaceBlockPtr block(
    const BlockIndex& index,
    const Eigen::Vector3f& min,
    const Eigen::Vector3f& max,
    std::initializer_list<MapSurfacePoint> points) {
  MapSurfacePointVector values(points.begin(), points.end());
  return makeSurfaceBlock(index,
                          SurfaceAabb(min, max),
                          std::move(values));
}

SurfaceSnapshotBuildResult initialSurface(
    const RunId& map_epoch,
    std::uint64_t revision,
    const std::vector<SurfaceBlockPtr>& blocks) {
  SurfaceSnapshotBuilder builder(
      nullptr, mapStamp(map_epoch, revision), surfaceStamp(map_epoch, revision));
  for (const SurfaceBlockPtr& surface_block : blocks) {
    builder.setBlock(surface_block);
  }
  return builder.build();
}

SurfaceSnapshotBuildResult advanceSurface(
    SurfaceSnapshotPtr base,
    std::uint64_t revision,
    const std::vector<SurfaceBlockPtr>& changed,
    const std::vector<BlockIndex>& removed = {}) {
  return SurfaceSnapshotBuilder::buildFrom(
      base,
      mapStamp(base->mapStamp().map_epoch, revision),
      surfaceStamp(base->surfaceStamp().map_epoch, revision),
      changed,
      removed);
}

SceneObjectPtr sceneObject(SceneObjectId object_id,
                           std::uint64_t identity_revision,
                           std::uint64_t obb_revision,
                           const Eigen::Vector3f& center,
                           const Eigen::Vector3f& size) {
  auto identity = std::make_shared<IdentityComponent>();
  identity->object_id = object_id;
  identity->revision = identity_revision;
  auto geometry = std::make_shared<GeometryComponent>();
  geometry->revision = 1;
  geometry->obb_revision = obb_revision;
  geometry->center_world = center;
  geometry->size_m = size;
  auto object = std::make_shared<SceneObject>();
  object->identity = std::move(identity);
  object->geometry = std::move(geometry);
  return object;
}

SceneSnapshot sceneSnapshot(
    const SurfaceStamp& latest_surface,
    std::initializer_list<SceneObjectPtr> objects) {
  auto table = std::make_shared<SceneObjectTable>();
  for (const SceneObjectPtr& object : objects) {
    (*table)[object->identity->object_id] = object;
  }
  auto state = std::make_shared<SceneState>();
  state->latest_surface = latest_surface;
  state->objects = std::move(table);
  return SceneSnapshot(std::move(state));
}

MapDelta delta(const RunId& map_epoch,
               std::uint64_t from,
               std::uint64_t to,
               std::initializer_list<BlockIndex> changed) {
  MapDelta result;
  result.from_map = mapStamp(map_epoch, from);
  result.to_map = mapStamp(map_epoch, to);
  result.from_surface = surfaceStamp(map_epoch, from);
  result.to_surface = surfaceStamp(map_epoch, to);
  result.changed_blocks.assign(changed.begin(), changed.end());
  return result;
}

void expectEvaluationEqual(const GeometryEvaluationResult& lhs,
                           const GeometryEvaluationResult& rhs) {
  EXPECT_EQ(lhs.status, rhs.status);
  EXPECT_FLOAT_EQ(lhs.score, rhs.score);
  EXPECT_FLOAT_EQ(lhs.shell_ratio, rhs.shell_ratio);
  EXPECT_FLOAT_EQ(lhs.extent_score, rhs.extent_score);
  EXPECT_FLOAT_EQ(lhs.leak_ratio, rhs.leak_ratio);
  EXPECT_FLOAT_EQ(lhs.cavity_ratio, rhs.cavity_ratio);
  EXPECT_EQ(lhs.in_box_points, rhs.in_box_points);
  EXPECT_EQ(lhs.shell_points, rhs.shell_points);
  EXPECT_EQ(lhs.unique_voxels, rhs.unique_voxels);
  EXPECT_EQ(lhs.expanded_points, rhs.expanded_points);
  EXPECT_EQ(lhs.reason, rhs.reason);
}

TEST(GeometryEvaluator, LocalBlocksAreEquivalentToFullSurfaceScan) {
  const RunId map_epoch = epoch();
  const BlockIndex local_index(0, 0, 0);
  const BlockIndex far_index(10, 0, 0);
  const SurfaceBlockPtr local = block(
      local_index,
      Eigen::Vector3f(-1.0f, -1.0f, -1.0f),
      Eigen::Vector3f(1.0f, 1.0f, 1.0f),
      {point(local_index, 0, -0.5f, -0.5f, -0.5f),
       point(local_index, 1, 0.5f, -0.5f, -0.5f),
       point(local_index, 2, -0.5f, 0.5f, -0.5f),
       point(local_index, 3, 0.5f, 0.5f, 0.5f),
       point(local_index, 4, 0.0f, 0.0f, 0.0f)});
  const SurfaceBlockPtr far = block(
      far_index,
      Eigen::Vector3f(10.0f, 0.0f, 0.0f),
      Eigen::Vector3f(11.0f, 1.0f, 1.0f),
      {point(far_index, 0, 10.5f, 0.5f, 0.5f)});
  const SurfaceSnapshotPtr surface =
      initialSurface(map_epoch, 1, {local, far}).snapshot;

  GeometryInput input;
  input.object.object_id = 7;
  input.object.identity_revision = 3;
  input.object.obb_revision = 4;
  input.center_world = Eigen::Vector3f::Zero();
  input.size_m = Eigen::Vector3f::Ones();
  input.thresholds.empty_inside_points = 1;
  input.thresholds.min_unique_voxels = 1;
  input.thresholds.confirm_score = 0.0f;
  input.surface = surface;
  input.checked_at_ns = 123;

  const GeometryResult local_result = evaluateGeometry(input);
  const GeometryResult full_result = evaluateGeometryFullSurface(input);
  expectEvaluationEqual(local_result.evaluation, full_result.evaluation);
  ASSERT_EQ(local_result.dependency.evaluated_blocks.size(), 1U);
  EXPECT_EQ(local_result.dependency.evaluated_blocks.front(),
            local_index.eigen());
  ASSERT_EQ(full_result.dependency.evaluated_blocks.size(), 2U);
  EXPECT_EQ(local_result.dependency.object.object_id, 7);
  EXPECT_EQ(local_result.dependency.surface, surface->surfaceStamp());
}

TEST(GeometryScheduler, MapDeltaSchedulesOnlyIntersectingObject) {
  const RunId map_epoch = epoch(2);
  const BlockIndex near_index(0, 0, 0);
  const BlockIndex far_index(10, 0, 0);
  const SurfaceBlockPtr near = block(
      near_index,
      Eigen::Vector3f(0.0f, 0.0f, 0.0f),
      Eigen::Vector3f(1.0f, 1.0f, 1.0f),
      {point(near_index, 0, 0.25f, 0.25f, 0.25f)});
  const SurfaceBlockPtr far = block(
      far_index,
      Eigen::Vector3f(10.0f, 0.0f, 0.0f),
      Eigen::Vector3f(11.0f, 1.0f, 1.0f),
      {point(far_index, 0, 10.25f, 0.25f, 0.25f)});
  const SurfaceSnapshotPtr first =
      initialSurface(map_epoch, 1, {near, far}).snapshot;

  const SceneObjectPtr near_object = sceneObject(
      1, 1, 1, Eigen::Vector3f(0.25f, 0.25f, 0.25f),
      Eigen::Vector3f::Constant(0.4f));
  const SceneObjectPtr far_object = sceneObject(
      2, 1, 1, Eigen::Vector3f(10.25f, 0.25f, 0.25f),
      Eigen::Vector3f::Constant(0.4f));
  const SceneSnapshot scene =
      sceneSnapshot(first->surfaceStamp(), {near_object, far_object});

  GeometryScheduler scheduler;
  ASSERT_TRUE(scheduler.onObjectCreated(ObjectCreated{1, 1}, scene, first));
  ASSERT_TRUE(scheduler.onObjectCreated(ObjectCreated{1, 2}, scene, first));
  GeometryInput discarded;
  ASSERT_TRUE(scheduler.tryPop(&discarded));
  ASSERT_TRUE(scheduler.tryPop(&discarded));
  EXPECT_FALSE(scheduler.tryPop(&discarded));
  scheduler.observeSurface(first);

  const SurfaceBlockPtr updated_near = block(
      near_index,
      near->aabb().min,
      near->aabb().max,
      {point(near_index, 1, 0.3f, 0.3f, 0.3f)});
  SurfaceSnapshotBuildResult second =
      advanceSurface(first, 2, {updated_near});
  const GeometryScheduleBatch batch =
      scheduler.onMapDelta(second.delta, second.snapshot);
  ASSERT_EQ(batch.scheduled_object_ids.size(), 1U);
  EXPECT_EQ(batch.scheduled_object_ids.front(), 1);
  EXPECT_FALSE(batch.conservative_full_scan);

  GeometryInput scheduled;
  ASSERT_TRUE(scheduler.tryPop(&scheduled));
  EXPECT_EQ(scheduled.object.object_id, 1);
  EXPECT_EQ(scheduled.trigger, GeometryTrigger::kMapDelta);
  EXPECT_EQ(scheduled.surface.get(), second.snapshot.get());
  EXPECT_FALSE(scheduler.tryPop(&scheduled));
}

TEST(GeometryScheduler, PendingQueueKeepsLatestInputPerObject) {
  GeometrySchedulerConfig config;
  config.pending_capacity = 1;
  GeometryScheduler scheduler(config);
  GeometryInput first;
  first.object.object_id = 9;
  first.object.obb_revision = 1;
  GeometryInput second = first;
  second.object.obb_revision = 2;

  EXPECT_EQ(scheduler.schedule(first).outcome, PushOutcome::kAccepted);
  const PushResult<GeometryInput> replaced = scheduler.schedule(second);
  EXPECT_EQ(replaced.outcome, PushOutcome::kReplaced);
  EXPECT_EQ(replaced.replacement_reason,
            ChannelReplacementReason::kMatchingKey);
  ASSERT_TRUE(replaced.replaced_item.has_value());
  EXPECT_EQ(replaced.replaced_item->object.obb_revision, 1U);

  GeometryInput latest;
  ASSERT_TRUE(scheduler.tryPop(&latest));
  EXPECT_EQ(latest.object.obb_revision, 2U);
  EXPECT_FALSE(scheduler.tryPop(&latest));
  EXPECT_EQ(scheduler.channelStats().producer_wait_count, 0U);
}

TEST(GeometryScheduler,
     CapacityBackpressureEventuallyDeliversEveryDistinctObject) {
  GeometrySchedulerConfig config;
  config.pending_capacity = 2;
  GeometryScheduler scheduler(config);

  constexpr SceneObjectId kObjectCount = 10;
  for (SceneObjectId object_id = 1; object_id <= 2; ++object_id) {
    GeometryInput input;
    input.object.object_id = object_id;
    ASSERT_EQ(scheduler.schedule(std::move(input)).outcome,
              PushOutcome::kAccepted);
  }

  std::atomic_bool producer_done{false};
  std::vector<PushOutcome> producer_outcomes;
  std::thread producer([&]() {
    for (SceneObjectId object_id = 3; object_id <= kObjectCount;
         ++object_id) {
      GeometryInput input;
      input.object.object_id = object_id;
      producer_outcomes.push_back(
          scheduler.schedule(std::move(input)).outcome);
    }
    producer_done = true;
  });

  const auto backpressure_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (scheduler.channelStats().producer_wait_count == 0 &&
         std::chrono::steady_clock::now() < backpressure_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_GT(scheduler.channelStats().producer_wait_count, 0U);

  std::vector<SceneObjectId> consumed;
  const auto convergence_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (consumed.size() < static_cast<std::size_t>(kObjectCount) &&
         std::chrono::steady_clock::now() < convergence_deadline) {
    GeometryInput input;
    if (scheduler.tryPop(&input)) {
      consumed.push_back(input.object.object_id);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // stop() is also the bounded failure path if the convergence assertion
  // above ever regresses: it releases a producer still waiting for capacity.
  scheduler.stop();
  producer.join();

  ASSERT_TRUE(producer_done.load());
  ASSERT_EQ(producer_outcomes.size(),
            static_cast<std::size_t>(kObjectCount - 2));
  EXPECT_TRUE(std::all_of(
      producer_outcomes.begin(), producer_outcomes.end(),
      [](PushOutcome outcome) { return outcome == PushOutcome::kAccepted; }));
  std::sort(consumed.begin(), consumed.end());
  ASSERT_EQ(consumed.size(), static_cast<std::size_t>(kObjectCount));
  for (SceneObjectId object_id = 1; object_id <= kObjectCount; ++object_id) {
    EXPECT_EQ(consumed[static_cast<std::size_t>(object_id - 1)], object_id);
  }
  const ChannelStats stats = scheduler.channelStats();
  EXPECT_LE(stats.high_watermark, config.pending_capacity);
  EXPECT_EQ(stats.replaced, 0U);
  EXPECT_GT(stats.producer_wait_count, 0U);
}

TEST(GeometryScheduler, StopReleasesCapacityBlockedProducer) {
  GeometrySchedulerConfig config;
  config.pending_capacity = 1;
  GeometryScheduler scheduler(config);

  GeometryInput first;
  first.object.object_id = 1;
  ASSERT_TRUE(scheduler.schedule(std::move(first)).accepted());

  std::atomic<PushOutcome> outcome{PushOutcome::kAccepted};
  std::thread producer([&]() {
    GeometryInput second;
    second.object.object_id = 2;
    outcome = scheduler.schedule(std::move(second)).outcome;
  });
  const auto blocked_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (scheduler.channelStats().producer_wait_count == 0 &&
         std::chrono::steady_clock::now() < blocked_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_GT(scheduler.channelStats().producer_wait_count, 0U);

  const auto stop_begin = std::chrono::steady_clock::now();
  scheduler.stop();
  producer.join();
  const auto stop_elapsed = std::chrono::steady_clock::now() - stop_begin;

  EXPECT_EQ(outcome.load(), PushOutcome::kStopped);
  EXPECT_LT(stop_elapsed, std::chrono::milliseconds(250));
  EXPECT_EQ(scheduler.channelStats().stopped_pushes, 1U);

  // stop remains drain-first for work that was already admitted.
  GeometryInput drained;
  ASSERT_TRUE(scheduler.waitPop(&drained));
  EXPECT_EQ(drained.object.object_id, 1);
  EXPECT_FALSE(scheduler.waitPop(&drained));
}

TEST(GeometryScheduler, ObbChangedReplacesPendingObjectCreation) {
  const RunId map_epoch = epoch(6);
  const BlockIndex index(0, 0, 0);
  const SurfaceSnapshotPtr surface =
      initialSurface(
          map_epoch,
          1,
          {block(index,
                 Eigen::Vector3f(-1.0f, -1.0f, -1.0f),
                 Eigen::Vector3f(1.0f, 1.0f, 1.0f),
                 {point(index, 0, 0.0f, 0.0f, 0.0f)})})
          .snapshot;
  const SceneSnapshot initial = sceneSnapshot(
      surface->surfaceStamp(),
      {sceneObject(11,
                   1,
                   1,
                   Eigen::Vector3f::Zero(),
                   Eigen::Vector3f::Ones())});
  const SceneSnapshot moved = sceneSnapshot(
      surface->surfaceStamp(),
      {sceneObject(11,
                   1,
                   2,
                   Eigen::Vector3f(0.2f, 0.0f, 0.0f),
                   Eigen::Vector3f::Ones())});

  GeometryScheduler scheduler;
  ASSERT_TRUE(
      scheduler.onObjectCreated(ObjectCreated{1, 11}, initial, surface));
  ASSERT_TRUE(scheduler.onObbChanged(ObbChanged{2, 11, 2}, moved, surface));
  EXPECT_EQ(scheduler.channelStats().replaced, 1U);

  GeometryInput latest;
  ASSERT_TRUE(scheduler.tryPop(&latest));
  EXPECT_EQ(latest.object.obb_revision, 2U);
  EXPECT_EQ(latest.trigger, GeometryTrigger::kObbChanged);
  EXPECT_FLOAT_EQ(latest.center_world.x(), 0.2f);
  EXPECT_FALSE(scheduler.tryPop(&latest));
}

TEST(GeometryCas, RejectsStaleObbRevision) {
  const RunId map_epoch = epoch(3);
  const SurfaceStamp current = surfaceStamp(map_epoch, 4);
  const SceneSnapshot scene = sceneSnapshot(
      current,
      {sceneObject(5,
                   2,
                   8,
                   Eigen::Vector3f::Zero(),
                   Eigen::Vector3f::Ones())});
  GeometryResult result;
  result.dependency.object.object_id = 5;
  result.dependency.object.identity_revision = 2;
  result.dependency.object.obb_revision = 7;
  result.dependency.surface = current;
  MapDeltaJournal journal(4);

  const GeometryCasCheck check =
      checkGeometryCas(result, scene, current, journal);
  EXPECT_FALSE(check.accepted());
  EXPECT_EQ(check.decision, GeometryCasDecision::kRejectObbRevision);
  EXPECT_FALSE(makeGeometryApplyCommand(result, current, check).has_value());
}

TEST(GeometryCas, AcceptsAdvanceWhenJournalProvesBlocksUnrelated) {
  const RunId map_epoch = epoch(4);
  const SurfaceStamp source = surfaceStamp(map_epoch, 1);
  const SurfaceStamp current = surfaceStamp(map_epoch, 2);
  const SceneSnapshot scene = sceneSnapshot(
      current,
      {sceneObject(6,
                   3,
                   4,
                   Eigen::Vector3f::Zero(),
                   Eigen::Vector3f::Ones())});
  GeometryResult result;
  result.dependency.object.object_id = 6;
  result.dependency.object.identity_revision = 3;
  result.dependency.object.obb_revision = 4;
  result.dependency.surface = source;
  result.dependency.evaluated_blocks.push_back(Eigen::Vector3i(0, 0, 0));
  result.evaluation.score = 0.75f;

  MapDeltaJournal journal(4);
  ASSERT_TRUE(journal.append(delta(
      map_epoch, 1, 2, {BlockIndex(20, 0, 0)})));
  const GeometryCasCheck check =
      checkGeometryCas(result, scene, current, journal);
  ASSERT_TRUE(check.accepted());
  EXPECT_EQ(check.decision,
            GeometryCasDecision::kAcceptUnrelatedSurfaceAdvance);
  EXPECT_EQ(check.delta_overlap, DeltaOverlapVerdict::kNoOverlap);
  const std::optional<ApplyGeometryResultCommand> command =
      makeGeometryApplyCommand(result, current, check);
  ASSERT_TRUE(command.has_value());
  EXPECT_EQ(command->current_surface, current);
  EXPECT_EQ(command->delta_overlap, DeltaOverlapVerdict::kNoOverlap);
  EXPECT_FLOAT_EQ(command->result.score, 0.75f);
}

TEST(GeometryCas, JournalGapIsRejectedConservatively) {
  const RunId map_epoch = epoch(5);
  const SurfaceStamp source = surfaceStamp(map_epoch, 1);
  const SurfaceStamp current = surfaceStamp(map_epoch, 3);
  const SceneSnapshot scene = sceneSnapshot(
      current,
      {sceneObject(7,
                   1,
                   1,
                   Eigen::Vector3f::Zero(),
                   Eigen::Vector3f::Ones())});
  GeometryResult result;
  result.dependency.object.object_id = 7;
  result.dependency.object.identity_revision = 1;
  result.dependency.object.obb_revision = 1;
  result.dependency.surface = source;
  result.dependency.evaluated_blocks.push_back(Eigen::Vector3i(0, 0, 0));

  MapDeltaJournal journal(4);
  ASSERT_TRUE(journal.append(delta(
      map_epoch, 2, 3, {BlockIndex(20, 0, 0)})));
  const GeometryCasCheck check =
      checkGeometryCas(result, scene, current, journal);
  EXPECT_FALSE(check.accepted());
  EXPECT_EQ(check.decision, GeometryCasDecision::kRejectJournalGap);
  EXPECT_EQ(check.delta_overlap, DeltaOverlapVerdict::kJournalGap);
  EXPECT_EQ(check.delta_query.unknown_reason,
            MapDeltaUnknownReason::kRevisionGap);
}

}  // namespace
}  // namespace roomie
