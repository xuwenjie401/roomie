#include <chrono>
#include <cstdint>
#include <filesystem>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "roomie/dsg/observation_history.hpp"
#include "roomie/pipeline/instance_map_thread.hpp"
#include "roomie/pipeline/snapshot_control_queue.hpp"

namespace roomie {
namespace {

using namespace std::chrono_literals;

class EmptyMapProjector final : public MapProjector {
 public:
  bool enqueueFrameBundle(FrameBundlePtr) override { return true; }

  std::optional<PatchDepth> projectPatchDepth(const FrameBundle&) override {
    return std::nullopt;
  }

  MapBackendSnapshot snapshotSurfacePoints() const override {
    return MapBackendSnapshot{};
  }

  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>>
  collectNearSurfaceVoxels(const RawDetection&) const override {
    return {};
  }
};

TEST(ObservationHistoryRetention, KeepsNewestEvidenceAndAllTimeAggregates) {
  InstanceTrack track;
  track.support_count = 100;
  track.high_quality_observation_count = 73;
  track.high_quality_observation_mass = 51.5f;
  track.observation_timestamps_ns = {7, 2, 9, 1, 8, 8};
  for (const TimeNanoseconds time_ns : {7, 2, 9, 1, 8}) {
    ObservationQualitySample sample;
    sample.time_ns = time_ns;
    sample.quality = static_cast<float>(time_ns) / 10.0f;
    track.observation_quality_history.push_back(sample);
  }

  retainRecentObservationHistory(&track, 3);

  EXPECT_EQ(track.observation_timestamps_ns,
            (std::vector<TimeNanoseconds>{7, 8, 9}));
  ASSERT_EQ(track.observation_quality_history.size(), 3U);
  EXPECT_EQ(track.observation_quality_history[0].time_ns, 7);
  EXPECT_EQ(track.observation_quality_history[1].time_ns, 8);
  EXPECT_EQ(track.observation_quality_history[2].time_ns, 9);
  EXPECT_EQ(track.support_count, 100);
  EXPECT_EQ(track.high_quality_observation_count, 73);
  EXPECT_FLOAT_EQ(track.high_quality_observation_mass, 51.5f);
}

TEST(SnapshotControlQueue, CoalescesByOwnerAndRejectsDistinctOverflow) {
  SnapshotControlQueue queue(2);
  EXPECT_EQ(queue.push(SnapshotControl{SnapshotControlKind::kMerge,
                                       10, 20, 100, 5}),
            SnapshotControlPushOutcome::kQueued);
  EXPECT_EQ(queue.push(SnapshotControl{SnapshotControlKind::kMerge,
                                       10, 99, 90, 4}),
            SnapshotControlPushOutcome::kCoalesced);
  std::optional<SnapshotControl> head = queue.front();
  ASSERT_TRUE(head);
  EXPECT_EQ(head->second_id, 20)
      << "an older revision must not overwrite a newer transition";

  EXPECT_EQ(queue.push(SnapshotControl{SnapshotControlKind::kMerge,
                                       10, 30, 110, 6}),
            SnapshotControlPushOutcome::kCoalesced);
  head = queue.front();
  ASSERT_TRUE(head);
  EXPECT_EQ(head->second_id, 30);
  EXPECT_EQ(head->scene_revision, 6U);

  EXPECT_EQ(queue.push(SnapshotControl{SnapshotControlKind::kEraseObject,
                                       40, -1, 120, 7}),
            SnapshotControlPushOutcome::kQueued);
  EXPECT_EQ(queue.push(SnapshotControl{SnapshotControlKind::kEraseObject,
                                       41, -1, 130, 8}),
            SnapshotControlPushOutcome::kRejectedCapacity);

  SnapshotControlQueueStats stats = queue.stats();
  EXPECT_EQ(stats.depth, 2U);
  EXPECT_EQ(stats.high_watermark, 2U);
  EXPECT_EQ(stats.queued, 2U);
  EXPECT_EQ(stats.coalesced, 2U);
  EXPECT_EQ(stats.rejected_capacity, 1U);
  EXPECT_EQ(queue.closeAndAbandon(), 2U);
  EXPECT_EQ(queue.push(SnapshotControl{SnapshotControlKind::kEraseObject,
                                       42, -1, 140, 9}),
            SnapshotControlPushOutcome::kRejectedClosed);
  stats = queue.stats();
  EXPECT_EQ(stats.depth, 0U);
  EXPECT_EQ(stats.abandoned_on_stop, 2U);
  EXPECT_EQ(stats.rejected_closed, 1U);
}

TEST(InstanceMapThreadBounds, NormalizesOversizedImportedHistory) {
  PipelineConfig config;
  config.instance_observation_history_capacity = 3;
  ThreadSafeQueue<InferenceResponse> responses(4);
  EmptyMapProjector projector;
  InstanceMapThread instance_map(responses, projector, config);

  ObjectGraphSnapshot graph;
  ObjectNode object;
  object.object_id = 7;
  object.label = "cup";
  object.observation_timestamps_ns = {10, 40, 20, 30, 40};
  graph.objects.push_back(object);
  std::string error;
  ASSERT_TRUE(instance_map.loadObjectGraphSnapshot(graph, &error)) << error;

  const SceneSnapshot loaded = instance_map.sceneSnapshot();
  const SceneObjectPtr loaded_object = loaded.findExactObject(7);
  ASSERT_TRUE(loaded_object);
  ASSERT_TRUE(loaded_object->semantic);
  EXPECT_EQ(loaded_object->semantic->observation_timestamps_ns,
            (std::vector<TimeNanoseconds>{20, 30, 40}));
  ASSERT_EQ(loaded.tracks().size(), 1U);
  ASSERT_TRUE(loaded.tracks().begin()->second);
  EXPECT_EQ(loaded.tracks().begin()->second->observation_timestamps_ns,
            (std::vector<TimeNanoseconds>{20, 30, 40}));
}

TEST(InstanceMapThreadBounds, IdleWaitHasDeadlineAndThenDrains) {
  PipelineConfig config;
  ThreadSafeQueue<InferenceResponse> responses(4);
  EmptyMapProjector projector;
  InstanceMapThread instance_map(responses, projector, config);

  ApplyHumanAnnotationCommand command;
  command.object_id = -1;
  ASSERT_TRUE(instance_map.enqueueSceneCommand(SceneCommand{command}));
  const auto started = std::chrono::steady_clock::now();
  EXPECT_FALSE(instance_map.waitUntilIdle(25ms));
  EXPECT_LT(std::chrono::steady_clock::now() - started, 500ms);

  instance_map.start();
  EXPECT_TRUE(instance_map.waitUntilIdle(1s));
  instance_map.stop();
}

TEST(InstanceMapThreadBounds,
     SnapshotControlSaturationTripsTerminalContentFuseButCloseDoesNot) {
  const std::filesystem::path asset_root =
      std::filesystem::temp_directory_path() /
      ("roomie_runtime_bounds_assets_" +
       std::to_string(std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count()));
  std::filesystem::remove_all(asset_root);
  {
  AssetStoreConfig asset_config;
  asset_config.root = asset_root;
  auto asset_store = std::make_shared<AssetStore>(asset_config);
  ASSERT_TRUE(asset_store->healthy()) << asset_store->initializationError();
  auto snapshot_bank = std::make_shared<SnapshotBank>(
      SnapshotBankConfig{}, asset_store);
  OnlineSnapshotWorkerConfig worker_config;
  worker_config.event_queue_capacity = 1;
  OnlineSnapshotWorker worker(
      worker_config, snapshot_bank, []() { return SceneSnapshot{}; },
      [](const ApplySnapshotSetCommand&) {
        return SnapshotCommandSubmitResult{
            SnapshotCommandDisposition::kAccepted, {}};
      });

  const auto graphWithObjects = [](std::initializer_list<int> object_ids) {
    ObjectGraphSnapshot graph;
    for (int object_id : object_ids) {
      ObjectNode object;
      object.object_id = object_id;
      object.label = "object";
      graph.objects.push_back(object);
    }
    return graph;
  };
  const auto tombstoneCommand = [](int object_id) {
    ApplyObservationBatchCommand command;
    command.association_source = "runtime_bounds_test";
    command.associated_mutations.emplace_back(
        TombstoneObjectMutation{object_id, "test delete"});
    return SceneCommand{std::move(command)};
  };

  {
    PipelineConfig config;
    config.snapshot_control_queue_size = 1;
    ThreadSafeQueue<InferenceResponse> responses(4);
    EmptyMapProjector projector;
    InstanceMapThread instance_map(responses, projector, config);
    std::string error;
    ASSERT_TRUE(instance_map.loadObjectGraphSnapshot(
        graphWithObjects({1, 2, 3, 4}), &error))
        << error;
    instance_map.setOnlineSnapshotWorker(&worker);
    instance_map.start();

    // The stopped/not-started worker accepts one event into its own bounded
    // channel. One more remains in the instance backlog; the third distinct
    // object then saturates that backlog and activates the terminal fuse.
    for (int object_id : {1, 2, 3}) {
      const auto result = instance_map.applySceneCommandAndWait(
          tombstoneCommand(object_id), 1s);
      ASSERT_TRUE(result);
      ASSERT_TRUE(result->committedRevision()) << result->reason;
    }
    ASSERT_TRUE(instance_map.snapshotControlFaulted());
    EXPECT_EQ(instance_map.snapshotControlStats().rejected_capacity, 1U);
    const SceneRevision fault_revision = instance_map.sceneSnapshot().revision();

    const auto rejected = instance_map.applySceneCommandAndWait(
        tombstoneCommand(4), 1s);
    ASSERT_TRUE(rejected);
    EXPECT_EQ(rejected->status, SceneApplyStatus::kRejected);
    EXPECT_NE(rejected->reason.find("snapshot control admission fault"),
              std::string::npos);
    EXPECT_EQ(rejected->revision, fault_revision);
    instance_map.stop();
  }

  {
    PipelineConfig config;
    config.snapshot_control_queue_size = 1;
    ThreadSafeQueue<InferenceResponse> responses(4);
    EmptyMapProjector projector;
    InstanceMapThread instance_map(responses, projector, config);
    std::string error;
    ASSERT_TRUE(instance_map.loadObjectGraphSnapshot(
        graphWithObjects({10, 11}), &error))
        << error;
    instance_map.setOnlineSnapshotWorker(&worker);
    (void)instance_map.closeSnapshotControlsForShutdown();
    instance_map.start();

    const auto closed_control = instance_map.applySceneCommandAndWait(
        tombstoneCommand(10), 1s);
    ASSERT_TRUE(closed_control);
    EXPECT_TRUE(closed_control->committedRevision()) << closed_control->reason;
    EXPECT_FALSE(instance_map.snapshotControlFaulted());
    EXPECT_EQ(instance_map.snapshotControlStats().rejected_closed, 1U);

    ApplyHumanAnnotationCommand annotation;
    annotation.object_id = 11;
    annotation.patch.label = "still admitted";
    const auto later_content = instance_map.applySceneCommandAndWait(
        SceneCommand{annotation}, 1s);
    ASSERT_TRUE(later_content);
    EXPECT_TRUE(later_content->committedRevision()) << later_content->reason;
    instance_map.stop();
  }
  }

  std::filesystem::remove_all(asset_root);
}

}  // namespace
}  // namespace roomie
