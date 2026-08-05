#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "roomie/pipeline/surface_snapshot.hpp"

namespace roomie {
namespace {

RunId epoch(std::uint64_t value) {
  RunId result;
  result.high = 0x1234U;
  result.low = value;
  return result;
}

MapStamp mapStamp(const RunId& map_epoch, std::uint64_t revision) {
  MapStamp stamp;
  stamp.map_epoch = map_epoch;
  stamp.map_revision = revision;
  stamp.integrated_through_ns = static_cast<TimeNanoseconds>(revision * 100U);
  return stamp;
}

SurfaceStamp surfaceStamp(const RunId& map_epoch,
                          std::uint64_t surface_revision,
                          std::uint64_t map_revision) {
  SurfaceStamp stamp;
  stamp.map_epoch = map_epoch;
  stamp.surface_revision = surface_revision;
  stamp.source_map_revision = map_revision;
  return stamp;
}

SurfaceAabb aabb(float min_x, float max_x) {
  return SurfaceAabb(Eigen::Vector3f(min_x, 0.0f, 0.0f),
                     Eigen::Vector3f(max_x, 1.0f, 1.0f));
}

MapSurfacePoint point(float x, float y = 0.5f, float z = 0.5f) {
  MapSurfacePoint result;
  result.position_world = Eigen::Vector3f(x, y, z);
  result.weight = 1.0f;
  return result;
}

SurfaceBlockPtr block(BlockIndex index,
                      SurfaceAabb bounds,
                      std::initializer_list<MapSurfacePoint> points) {
  MapSurfacePointVector storage(points.begin(), points.end());
  return makeSurfaceBlock(index, std::move(bounds), std::move(storage));
}

SurfaceBlockPtr emptyBlock(BlockIndex index, SurfaceAabb bounds) {
  return makeSurfaceBlock(index, std::move(bounds), MapSurfacePointVector{});
}

MapDelta delta(const RunId& map_epoch,
               std::uint64_t from_revision,
               std::uint64_t to_revision,
               std::vector<BlockIndex> changed,
               std::vector<BlockIndex> removed = {},
               bool full_rebuild = false) {
  MapDelta result;
  result.from_map = mapStamp(map_epoch, from_revision);
  result.to_map = mapStamp(map_epoch, to_revision);
  result.from_surface =
      surfaceStamp(map_epoch, from_revision, from_revision);
  result.to_surface = surfaceStamp(map_epoch, to_revision, to_revision);
  result.changed_blocks = std::move(changed);
  result.removed_blocks = std::move(removed);
  result.full_rebuild = full_rebuild;
  return result;
}

TEST(SurfaceSnapshot, BuilderReusesUnchangedBlocksAndKeepsOldSnapshotImmutable) {
  const RunId map_epoch = epoch(1);
  const BlockIndex changed_index(0, 0, 0);
  const BlockIndex reused_index(1, 0, 0);
  const SurfaceBlockPtr old_changed =
      block(changed_index, aabb(0.0f, 1.0f), {point(0.25f)});
  const SurfaceBlockPtr reused =
      block(reused_index, aabb(1.0f, 2.0f), {point(1.25f)});

  SurfaceSnapshotBuilder initial_builder(
      nullptr, mapStamp(map_epoch, 1), surfaceStamp(map_epoch, 1, 1));
  initial_builder.setBlock(old_changed);
  initial_builder.setBlock(reused);
  const SurfaceSnapshotBuildResult initial = initial_builder.build();
  ASSERT_TRUE(initial.snapshot);
  EXPECT_TRUE(initial.delta.full_rebuild);

  const SurfaceBlockPtr new_changed =
      block(changed_index, aabb(0.0f, 1.0f), {point(0.75f)});
  SurfaceSnapshotBuilder next_builder(
      initial.snapshot, mapStamp(map_epoch, 2), surfaceStamp(map_epoch, 2, 2));
  next_builder.setBlock(new_changed);
  const SurfaceSnapshotBuildResult next = next_builder.build();

  ASSERT_TRUE(next.snapshot);
  EXPECT_EQ(next.snapshot->block(changed_index), new_changed);
  EXPECT_EQ(next.snapshot->block(reused_index), reused);
  EXPECT_EQ(next.snapshot->block(reused_index),
            initial.snapshot->block(reused_index));
  EXPECT_NE(next.snapshot->block(changed_index),
            initial.snapshot->block(changed_index));

  // The old snapshot still owns and exposes the old block contents.
  ASSERT_EQ(initial.snapshot->block(changed_index)->pointCount(), 1U);
  EXPECT_FLOAT_EQ(initial.snapshot->block(changed_index)
                      ->points()
                      .front()
                      .position_world.x(),
                  0.25f);
  EXPECT_FLOAT_EQ(next.snapshot->block(changed_index)
                      ->points()
                      .front()
                      .position_world.x(),
                  0.75f);

  const SurfaceSnapshotStats& stats = next.snapshot->stats();
  EXPECT_EQ(stats.block_count, 2U);
  EXPECT_EQ(stats.point_count, 2U);
  EXPECT_EQ(stats.dirty_block_count, 1U);
  EXPECT_EQ(stats.reused_block_count, 1U);
  EXPECT_EQ(stats.replaced_block_count, 1U);
  EXPECT_EQ(stats.added_block_count, 0U);
  EXPECT_EQ(stats.removed_block_count, 0U);

  ASSERT_EQ(next.delta.changed_blocks.size(), 1U);
  EXPECT_EQ(next.delta.changed_blocks.front(), changed_index);
  EXPECT_TRUE(next.delta.removed_blocks.empty());
  EXPECT_FALSE(next.delta.full_rebuild);
}

TEST(SurfaceSnapshot, EmptyDirtyBlockDeletesPreviousBlock) {
  const RunId map_epoch = epoch(2);
  const BlockIndex index(0, 0, 0);
  SurfaceSnapshotBuilder initial_builder(
      nullptr, mapStamp(map_epoch, 1), surfaceStamp(map_epoch, 1, 1));
  initial_builder.setBlock(
      block(index, aabb(0.0f, 1.0f), {point(0.5f)}));
  const SurfaceSnapshotBuildResult initial = initial_builder.build();
  ASSERT_TRUE(initial.snapshot->block(index));

  SurfaceSnapshotBuilder removal_builder(
      initial.snapshot, mapStamp(map_epoch, 2), surfaceStamp(map_epoch, 2, 2));
  removal_builder.setBlock(emptyBlock(index, aabb(0.0f, 1.0f)));
  const SurfaceSnapshotBuildResult removed = removal_builder.build();

  EXPECT_FALSE(removed.snapshot->block(index));
  EXPECT_TRUE(removed.snapshot->empty());
  EXPECT_EQ(removed.snapshot->stats().removed_block_count, 1U);
  EXPECT_EQ(removed.snapshot->stats().point_count, 0U);
  ASSERT_EQ(removed.delta.removed_blocks.size(), 1U);
  EXPECT_EQ(removed.delta.removed_blocks.front(), index);

  // Removing from the new revision never mutates the pinned old revision.
  ASSERT_TRUE(initial.snapshot->block(index));
  EXPECT_EQ(initial.snapshot->pointCount(), 1U);
}

TEST(SurfaceSnapshot, RemovingNeverCachedDirtyBlockIsNotASurfaceDelta) {
  const RunId map_epoch = epoch(22);
  const BlockIndex retained_index(0, 0, 0);
  const BlockIndex never_cached_index(9, 9, 9);
  SurfaceSnapshotBuilder initial_builder(
      nullptr, mapStamp(map_epoch, 1), surfaceStamp(map_epoch, 1, 1));
  initial_builder.setBlock(
      block(retained_index, aabb(0.0f, 1.0f), {point(0.5f)}));
  const SurfaceSnapshotBuildResult initial = initial_builder.build();

  SurfaceSnapshotBuilder next_builder(
      initial.snapshot, mapStamp(map_epoch, 2), surfaceStamp(map_epoch, 2, 2));
  next_builder.removeBlock(never_cached_index);
  const SurfaceSnapshotBuildResult next = next_builder.build();

  EXPECT_EQ(next.snapshot->block(retained_index),
            initial.snapshot->block(retained_index));
  EXPECT_EQ(next.snapshot->stats().dirty_block_count, 0U);
  EXPECT_EQ(next.snapshot->stats().removed_block_count, 0U);
  EXPECT_TRUE(next.delta.changed_blocks.empty());
  EXPECT_TRUE(next.delta.removed_blocks.empty());
}

TEST(SurfaceSnapshot, AddThenRemoveOfNewBlockCollapsesToNoOp) {
  const RunId map_epoch = epoch(23);
  const BlockIndex transient_index(3, 2, 1);
  SurfaceSnapshotBuilder builder(
      nullptr, mapStamp(map_epoch, 1), surfaceStamp(map_epoch, 1, 1));
  builder.setBlock(
      block(transient_index, aabb(0.0f, 1.0f), {point(0.5f)}));
  builder.removeBlock(transient_index);
  const SurfaceSnapshotBuildResult result = builder.build();

  EXPECT_TRUE(result.snapshot->empty());
  EXPECT_EQ(result.snapshot->stats().dirty_block_count, 0U);
  EXPECT_TRUE(result.delta.changed_blocks.empty());
  EXPECT_TRUE(result.delta.removed_blocks.empty());
}

TEST(SurfaceSnapshot, FlattenAndAabbQueriesAreDeterministicAndLocal) {
  const RunId map_epoch = epoch(3);
  SurfaceSnapshotBuilder builder(
      nullptr, mapStamp(map_epoch, 1), surfaceStamp(map_epoch, 1, 1));
  // Insert in reverse index order; flatten/view are still index ordered.
  builder.setBlock(block(BlockIndex(1, 0, 0),
                         aabb(1.0f, 2.0f),
                         {point(1.1f), point(1.8f)}));
  builder.setBlock(block(BlockIndex(0, 0, 0),
                         aabb(0.0f, 1.0f),
                         {point(0.2f), point(0.9f)}));
  const SurfaceSnapshotPtr snapshot = builder.build().snapshot;

  const std::vector<SurfaceBlockPtr> view = snapshot->blockView();
  ASSERT_EQ(view.size(), 2U);
  EXPECT_EQ(view[0]->index(), BlockIndex(0, 0, 0));
  EXPECT_EQ(view[1]->index(), BlockIndex(1, 0, 0));

  const MapSurfacePointVector flattened = snapshot->flatten();
  ASSERT_EQ(flattened.size(), 4U);
  EXPECT_FLOAT_EQ(flattened[0].position_world.x(), 0.2f);
  EXPECT_FLOAT_EQ(flattened[1].position_world.x(), 0.9f);
  EXPECT_FLOAT_EQ(flattened[2].position_world.x(), 1.1f);
  EXPECT_FLOAT_EQ(flattened[3].position_world.x(), 1.8f);

  const SurfaceAabb local(Eigen::Vector3f(0.8f, 0.0f, 0.0f),
                          Eigen::Vector3f(1.2f, 1.0f, 1.0f));
  const std::vector<SurfaceBlockPtr> intersecting =
      snapshot->blocksIntersecting(local);
  EXPECT_EQ(intersecting.size(), 2U);

  const MapSurfacePointVector local_points = snapshot->pointsInAabb(local);
  ASSERT_EQ(local_points.size(), 2U);
  EXPECT_FLOAT_EQ(local_points[0].position_world.x(), 0.9f);
  EXPECT_FLOAT_EQ(local_points[1].position_world.x(), 1.1f);

  const WorldPointVector world_points = snapshot->flattenWorldPoints();
  ASSERT_EQ(world_points.size(), 4U);
  EXPECT_FLOAT_EQ(world_points.front().x(), 0.2f);
  EXPECT_TRUE(snapshot->stats().has_bounds);
  EXPECT_FLOAT_EQ(snapshot->stats().bounds.min.x(), 0.0f);
  EXPECT_FLOAT_EQ(snapshot->stats().bounds.max.x(), 2.0f);
}

TEST(MapDeltaJournal, ReportsLocalIntersectionAndProvesUnrelatedUpdatesSafe) {
  const RunId map_epoch = epoch(4);
  const BlockIndex block_a(0, 0, 0);
  const BlockIndex block_b(1, 0, 0);
  const BlockIndex block_c(2, 0, 0);
  MapDeltaJournal journal(4);
  ASSERT_TRUE(journal.append(delta(map_epoch, 1, 2, {block_a})));
  ASSERT_TRUE(journal.append(delta(map_epoch, 2, 3, {}, {block_b})));

  const MapDeltaQueryResult unrelated = journal.query(
      surfaceStamp(map_epoch, 1, 1),
      surfaceStamp(map_epoch, 3, 3),
      {block_c});
  EXPECT_EQ(unrelated.status, MapDeltaQueryStatus::kNoIntersection);
  EXPECT_EQ(unrelated.unknown_reason, MapDeltaUnknownReason::kNone);
  EXPECT_EQ(unrelated.checked_deltas, 2U);
  EXPECT_FALSE(unrelated.conservativelyAffected());

  const MapDeltaQueryResult changed = journal.query(
      surfaceStamp(map_epoch, 1, 1),
      surfaceStamp(map_epoch, 3, 3),
      {block_b});
  EXPECT_EQ(changed.status, MapDeltaQueryStatus::kIntersection);
  EXPECT_TRUE(changed.conservativelyAffected());
}

TEST(MapDeltaJournal, EvictionAndForwardGapReturnConservativeUnknown) {
  const RunId map_epoch = epoch(5);
  const BlockIndex index(0, 0, 0);

  MapDeltaJournal bounded(1);
  ASSERT_TRUE(bounded.append(delta(map_epoch, 1, 2, {index})));
  ASSERT_TRUE(bounded.append(delta(map_epoch, 2, 3, {index})));
  const MapDeltaQueryResult evicted = bounded.query(
      surfaceStamp(map_epoch, 1, 1),
      surfaceStamp(map_epoch, 3, 3),
      {BlockIndex(9, 9, 9)});
  EXPECT_EQ(evicted.status, MapDeltaQueryStatus::kUnknown);
  EXPECT_EQ(evicted.unknown_reason, MapDeltaUnknownReason::kRevisionGap);
  EXPECT_TRUE(evicted.conservativelyAffected());

  MapDeltaJournal gap(4);
  ASSERT_TRUE(gap.append(delta(map_epoch, 1, 2, {index})));
  ASSERT_TRUE(gap.append(delta(map_epoch, 3, 4, {index})));
  const MapDeltaQueryResult missing_revision = gap.query(
      surfaceStamp(map_epoch, 1, 1),
      surfaceStamp(map_epoch, 4, 4),
      {BlockIndex(9, 9, 9)});
  EXPECT_EQ(missing_revision.status, MapDeltaQueryStatus::kUnknown);
  EXPECT_EQ(missing_revision.unknown_reason,
            MapDeltaUnknownReason::kRevisionGap);
}

TEST(MapDeltaJournal, EpochMismatchAndFullRebuildReturnUnknown) {
  const RunId old_epoch = epoch(6);
  const RunId new_epoch = epoch(7);
  const BlockIndex index(0, 0, 0);
  MapDeltaJournal journal(4);
  ASSERT_TRUE(journal.append(delta(old_epoch, 1, 2, {index})));

  const MapDeltaQueryResult epoch_mismatch = journal.query(
      surfaceStamp(old_epoch, 1, 1),
      surfaceStamp(new_epoch, 2, 2),
      {index});
  EXPECT_EQ(epoch_mismatch.status, MapDeltaQueryStatus::kUnknown);
  EXPECT_EQ(epoch_mismatch.unknown_reason,
            MapDeltaUnknownReason::kEpochMismatch);

  MapDeltaJournal rebuild_journal(4);
  ASSERT_TRUE(rebuild_journal.append(
      delta(old_epoch, 1, 2, {index}, {}, true)));
  const MapDeltaQueryResult rebuild = rebuild_journal.query(
      surfaceStamp(old_epoch, 1, 1),
      surfaceStamp(old_epoch, 2, 2),
      {BlockIndex(9, 9, 9)});
  EXPECT_EQ(rebuild.status, MapDeltaQueryStatus::kUnknown);
  EXPECT_EQ(rebuild.unknown_reason, MapDeltaUnknownReason::kFullRebuild);
}

}  // namespace
}  // namespace roomie
