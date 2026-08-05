#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "roomie/scene/geometry_worker.hpp"

namespace roomie {
namespace {

SurfaceSnapshotBuildResult makeSurface(const RunId& epoch) {
  MapStamp map;
  map.map_epoch = epoch;
  map.map_revision = 1;
  map.integrated_through_ns = 100;
  SurfaceStamp surface;
  surface.map_epoch = epoch;
  surface.surface_revision = 1;
  surface.source_map_revision = 1;

  const BlockIndex index(0, 0, 0);
  MapSurfacePointVector points;
  for (int i = 0; i < 16; ++i) {
    MapSurfacePoint point;
    point.position_world = Eigen::Vector3f(
        (i & 1) ? 0.45f : -0.45f,
        (i & 2) ? 0.45f : -0.45f,
        (i & 4) ? 0.45f : -0.45f);
    point.has_voxel_ref = true;
    point.voxel_ref.block_index = index.eigen();
    point.voxel_ref.voxel_index = Eigen::Vector3i(i, 0, 0);
    points.push_back(point);
  }
  SurfaceSnapshotBuilder builder(nullptr, map, surface);
  builder.setBlock(makeSurfaceBlock(
      index,
      SurfaceAabb(Eigen::Vector3f::Constant(-1.0f),
                  Eigen::Vector3f::Constant(1.0f)),
      std::move(points)));
  return builder.build();
}

SceneSnapshot makeScene(const SurfaceStamp& surface,
                        std::uint64_t identity_revision = 2) {
  auto identity = std::make_shared<IdentityComponent>();
  identity->object_id = 7;
  identity->revision = identity_revision;
  auto geometry = std::make_shared<GeometryComponent>();
  geometry->obb_revision = 3;
  geometry->center_world = Eigen::Vector3f::Zero();
  geometry->size_m = Eigen::Vector3f::Ones();
  auto object = std::make_shared<SceneObject>();
  object->identity = std::move(identity);
  object->geometry = std::move(geometry);
  auto objects = std::make_shared<SceneObjectTable>();
  (*objects)[7] = std::move(object);
  auto state = std::make_shared<SceneState>();
  state->latest_scene_revision = 4;
  state->latest_surface = surface;
  state->objects = std::move(objects);
  return SceneSnapshot(std::move(state));
}

TEST(GeometryWorker, SurfaceWatermarkPrecedesAsynchronousCasResult) {
  const RunId epoch{91, 2};
  SurfaceSnapshotBuildResult built = makeSurface(epoch);
  const SceneSnapshot scene = makeScene(built.snapshot->surfaceStamp());

  std::mutex mutex;
  std::condition_variable cv;
  std::vector<SceneCommand> commands;
  GeometrySchedulerConfig config;
  config.pending_capacity = 8;
  config.thresholds.empty_inside_points = 1;
  config.thresholds.min_unique_voxels = 1;
  config.thresholds.confirm_score = 0.0f;
  GeometryWorkerThread worker(
      config,
      [scene]() { return scene; },
      [&](SceneCommand command) {
        {
          std::lock_guard<std::mutex> lock(mutex);
          commands.push_back(std::move(command));
        }
        cv.notify_all();
        return true;
      });
  worker.start();

  MapCommit commit;
  commit.surface = built.snapshot->surfaceStamp();
  commit.map = built.snapshot->mapStamp();
  commit.snapshot = built.snapshot;
  commit.delta = built.delta;
  worker.onMapCommit(commit);

  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&]() {
      return commands.size() >= 2;
    }));
  }
  worker.stop();

  ASSERT_TRUE(std::holds_alternative<AdvanceSurfaceCommand>(commands.front()));
  const auto geometry_it = std::find_if(
      commands.begin(), commands.end(), [](const SceneCommand& command) {
        return std::holds_alternative<ApplyGeometryResultCommand>(command);
      });
  ASSERT_NE(geometry_it, commands.end());
  const auto& geometry =
      std::get<ApplyGeometryResultCommand>(*geometry_it);
  EXPECT_EQ(geometry.dependency.object.object_id, 7);
  EXPECT_EQ(geometry.dependency.object.identity_revision, 2U);
  EXPECT_EQ(geometry.dependency.object.obb_revision, 3U);
  EXPECT_EQ(geometry.current_surface, commit.surface);
  EXPECT_EQ(geometry.delta_overlap, DeltaOverlapVerdict::kExactSurface);
  EXPECT_GE(worker.stats().evaluated, 1U);
}

TEST(GeometryWorker, StopDrainsAlreadyScheduledEvaluation) {
  const RunId epoch{91, 3};
  SurfaceSnapshotBuildResult built = makeSurface(epoch);
  const SceneSnapshot scene = makeScene(built.snapshot->surfaceStamp());
  std::atomic_uint64_t geometry_commands{0};
  GeometrySchedulerConfig config;
  config.pending_capacity = 8;
  GeometryWorkerThread worker(
      config,
      [scene]() { return scene; },
      [&](SceneCommand command) {
        if (std::holds_alternative<ApplyGeometryResultCommand>(command)) {
          ++geometry_commands;
        }
        return true;
      });
  worker.start();
  MapCommit commit;
  commit.surface = built.snapshot->surfaceStamp();
  commit.map = built.snapshot->mapStamp();
  commit.snapshot = built.snapshot;
  commit.delta = built.delta;
  worker.onMapCommit(commit);
  worker.stop();
  EXPECT_GE(worker.stats().evaluated, 1U);
  EXPECT_GE(geometry_commands.load(), 1U);
}

TEST(GeometryWorker, CasRetryIsLocalAndReportedSeparately) {
  const RunId epoch{91, 4};
  SurfaceSnapshotBuildResult built = makeSurface(epoch);
  const SceneSnapshot initial =
      makeScene(built.snapshot->surfaceStamp(), 2);
  const SceneSnapshot advanced =
      makeScene(built.snapshot->surfaceStamp(), 3);

  std::atomic_uint64_t scene_reads{0};
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<SceneCommand> commands;
  GeometrySchedulerConfig config;
  config.pending_capacity = 1;
  GeometryWorkerThread worker(
      config,
      [&]() {
        return scene_reads.fetch_add(1) == 0 ? initial : advanced;
      },
      [&](SceneCommand command) {
        {
          std::lock_guard<std::mutex> lock(mutex);
          commands.push_back(std::move(command));
        }
        cv.notify_all();
        return true;
      });
  worker.start();

  MapCommit commit;
  commit.surface = built.snapshot->surfaceStamp();
  commit.map = built.snapshot->mapStamp();
  commit.snapshot = built.snapshot;
  commit.delta = built.delta;
  worker.onMapCommit(commit);

  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&]() {
      return std::any_of(
          commands.begin(), commands.end(), [](const SceneCommand& command) {
            return std::holds_alternative<ApplyGeometryResultCommand>(
                command);
          });
    }));
  }
  worker.stop();

  const auto geometry_it = std::find_if(
      commands.begin(), commands.end(), [](const SceneCommand& command) {
        return std::holds_alternative<ApplyGeometryResultCommand>(command);
      });
  ASSERT_NE(geometry_it, commands.end());
  EXPECT_EQ(std::get<ApplyGeometryResultCommand>(*geometry_it)
                .dependency.object.identity_revision,
            3U);
  const GeometryWorkerStats stats = worker.stats();
  EXPECT_GE(stats.superseded, 1U);
  EXPECT_GE(stats.retry_attempts, 1U);
  EXPECT_GE(stats.retries_scheduled, 1U);
  EXPECT_EQ(stats.backpressure_waits, 0U);
}

}  // namespace
}  // namespace roomie
