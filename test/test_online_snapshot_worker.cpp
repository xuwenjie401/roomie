#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include "roomie/artifacts/online_snapshot_worker.hpp"

namespace roomie {
namespace {

class TempDirectory {
 public:
  explicit TempDirectory(const std::string& label) {
    static std::atomic<std::uint64_t> sequence{1};
    path_ = std::filesystem::temp_directory_path() /
            ("roomie_online_snapshot_" + label + "_" +
             std::to_string(::getpid()) + "_" +
             std::to_string(sequence.fetch_add(1)));
    std::filesystem::create_directories(path_);
  }

  ~TempDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

std::shared_ptr<const ImageBuffer> makeFrame(std::uint8_t seed) {
  auto image = std::make_shared<ImageBuffer>();
  image->width = 24;
  image->height = 16;
  image->channels = 3;
  image->encoding = "rgb8";
  image->data.resize(
      static_cast<std::size_t>(image->width * image->height * image->channels));
  for (std::size_t index = 0; index < image->data.size(); ++index) {
    image->data[index] =
        static_cast<std::uint8_t>((seed + index * 13U) % 251U);
  }
  return image;
}

SnapshotCandidate makeCandidate(std::shared_ptr<const ImageBuffer> full_frame,
                                FrameId frame_id, float quality,
                                float azimuth, float x_offset = 0.0f) {
  SnapshotCandidate candidate;
  candidate.full_frame = std::move(full_frame);
  candidate.bbox_xyxy = {2.0f + x_offset, 2.0f, 13.0f + x_offset, 12.0f};
  candidate.quality.confidence = quality;
  candidate.viewpoint.azimuth_rad = azimuth;
  candidate.viewpoint.scale = 0.1f;
  candidate.time_ns = 1000 + static_cast<TimeNanoseconds>(frame_id);
  candidate.camera_id = "head_rgbd";
  candidate.provenance.run_id = RunId{44, 55};
  candidate.provenance.frame_id = frame_id;
  return candidate;
}

SceneObjectPtr makeSceneObject(SceneObjectId object_id) {
  auto object = std::make_shared<SceneObject>();
  auto identity = std::make_shared<IdentityComponent>();
  identity->revision = 1;
  identity->object_id = object_id;
  object->identity = std::move(identity);
  auto lifecycle = std::make_shared<LifecycleComponent>();
  lifecycle->revision = 1;
  lifecycle->track_state = InstanceTrackState::kStable;
  object->lifecycle = std::move(lifecycle);
  auto geometry = std::make_shared<GeometryComponent>();
  geometry->revision = 1;
  geometry->obb_revision = 1;
  object->geometry = std::move(geometry);
  auto semantic = std::make_shared<SemanticComponent>();
  semantic->revision = 1;
  semantic->label = "object-" + std::to_string(object_id);
  object->semantic = std::move(semantic);
  auto annotation = std::make_shared<AnnotationComponent>();
  annotation->revision = 1;
  object->annotation = std::move(annotation);
  auto artifact = std::make_shared<ArtifactComponent>();
  artifact->revision = 1;
  object->artifact = std::move(artifact);
  return object;
}

class FakeLiveScene {
 public:
  explicit FakeLiveScene(std::vector<SceneObjectId> object_ids) {
    auto state = std::make_shared<SceneState>();
    state->latest_scene_revision = 1;
    auto objects = std::make_shared<SceneObjectTable>();
    for (SceneObjectId object_id : object_ids) {
      objects->emplace(object_id, makeSceneObject(object_id));
      state->next_object_id = std::max(state->next_object_id, object_id + 1);
    }
    state->objects = objects;
    snapshot_ = SceneSnapshot{state};
  }

  SceneSnapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
  }

  void rejectNextAsStale() {
    std::lock_guard<std::mutex> lock(mutex_);
    stale_once_ = true;
  }

  SnapshotCommandSubmitResult submit(const ApplySnapshotSetCommand& command) {
    std::lock_guard<std::mutex> lock(mutex_);
    attempts_.push_back(command);
    if (stale_once_) {
      stale_once_ = false;
      bumpIdentityLocked(command.dependency.object_id);
      return {SnapshotCommandDisposition::kStale,
              "simulated concurrent object update"};
    }

    const std::optional<SceneObjectId> canonical =
        snapshot_.resolveCanonicalId(command.dependency.object_id);
    const SceneObjectPtr object =
        canonical ? snapshot_.findObject(*canonical) : nullptr;
    if (!canonical || !object || !object->identity || !object->artifact ||
        object->identity->revision !=
            command.dependency.identity_revision ||
        object->artifact->appearance_revision !=
            command.dependency.appearance_revision ||
        (command.dependency.obb_revision != 0 &&
         (!object->geometry || object->geometry->obb_revision !=
                                  command.dependency.obb_revision)) ||
        (command.dependency.semantic_revision != 0 &&
         (!object->semantic || object->semantic->revision !=
                                  command.dependency.semantic_revision))) {
      return {SnapshotCommandDisposition::kStale, "dependency mismatch"};
    }

    SceneState next = *snapshot_.statePtr();
    SceneObjectTable objects = snapshot_.objects();
    auto updated = std::make_shared<SceneObject>(*object);
    auto artifact = std::make_shared<ArtifactComponent>(*object->artifact);
    ++artifact->revision;
    ++artifact->appearance_revision;
    artifact->snapshots = command.snapshots;
    artifact->snapshot_set_hash = command.snapshot_set_hash;
    updated->artifact = std::move(artifact);
    objects[*canonical] = std::move(updated);
    next.objects =
        std::make_shared<const SceneObjectTable>(std::move(objects));
    ++next.latest_scene_revision;
    snapshot_ = SceneSnapshot{std::make_shared<const SceneState>(std::move(next))};
    accepted_.push_back(command);
    return {SnapshotCommandDisposition::kAccepted, {}};
  }

  void merge(SceneObjectId retired, SceneObjectId canonical) {
    std::lock_guard<std::mutex> lock(mutex_);
    SceneState next = *snapshot_.statePtr();
    SceneObjectTable objects = snapshot_.objects();
    objects.erase(retired);
    SceneAliasTable aliases = snapshot_.aliases();
    aliases[retired] = ObjectAlias{retired, canonical,
                                   next.latest_scene_revision + 1};
    next.objects =
        std::make_shared<const SceneObjectTable>(std::move(objects));
    next.aliases =
        std::make_shared<const SceneAliasTable>(std::move(aliases));
    ++next.latest_scene_revision;
    snapshot_ = SceneSnapshot{std::make_shared<const SceneState>(std::move(next))};
  }

  void erase(SceneObjectId object_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    SceneState next = *snapshot_.statePtr();
    SceneObjectTable objects = snapshot_.objects();
    objects.erase(object_id);
    next.objects =
        std::make_shared<const SceneObjectTable>(std::move(objects));
    ++next.latest_scene_revision;
    snapshot_ = SceneSnapshot{std::make_shared<const SceneState>(std::move(next))};
  }

  std::vector<ApplySnapshotSetCommand> attempts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return attempts_;
  }

  std::vector<ApplySnapshotSetCommand> accepted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return accepted_;
  }

 private:
  void bumpIdentityLocked(SceneObjectId object_id) {
    const std::optional<SceneObjectId> canonical =
        snapshot_.resolveCanonicalId(object_id);
    const SceneObjectPtr object =
        canonical ? snapshot_.findObject(*canonical) : nullptr;
    if (!canonical || !object || !object->identity) {
      return;
    }
    SceneState next = *snapshot_.statePtr();
    SceneObjectTable objects = snapshot_.objects();
    auto updated = std::make_shared<SceneObject>(*object);
    auto identity = std::make_shared<IdentityComponent>(*object->identity);
    ++identity->revision;
    updated->identity = std::move(identity);
    objects[*canonical] = std::move(updated);
    next.objects =
        std::make_shared<const SceneObjectTable>(std::move(objects));
    ++next.latest_scene_revision;
    snapshot_ = SceneSnapshot{std::make_shared<const SceneState>(std::move(next))};
  }

  mutable std::mutex mutex_;
  SceneSnapshot snapshot_;
  bool stale_once_ = false;
  std::vector<ApplySnapshotSetCommand> attempts_;
  std::vector<ApplySnapshotSetCommand> accepted_;
};

std::shared_ptr<AssetStore> makeStore(const std::filesystem::path& root) {
  AssetStoreConfig config;
  config.root = root;
  config.grace_period_ns = 100;
  auto store = std::make_shared<AssetStore>(config);
  EXPECT_TRUE(store->healthy()) << store->initializationError();
  return store;
}

std::shared_ptr<SnapshotBank> makeBank(
    const std::shared_ptr<AssetStore>& store) {
  SnapshotBankConfig config;
  config.top_k = 4;
  config.replacement_min_quality_delta = 0.01f;
  config.replacement_min_quality_ratio = 1.01f;
  return std::make_shared<SnapshotBank>(config, store);
}

OnlineSnapshotCandidate forTrack(int track_id, SnapshotCandidate candidate) {
  return OnlineSnapshotCandidate{
      SnapshotOwner{SnapshotOwnerKind::kTentativeTrack, track_id},
      std::move(candidate)};
}

OnlineSnapshotCandidate forObject(SceneObjectId object_id,
                                  SnapshotCandidate candidate) {
  return OnlineSnapshotCandidate{
      SnapshotOwner{SnapshotOwnerKind::kObject, object_id},
      std::move(candidate)};
}

TEST(OnlineSnapshotWorker,
     PromotionAbsorbsFencedCandidateSharesFrameAndRetriesStaleCas) {
  TempDirectory temporary("promotion");
  const auto store = makeStore(temporary.path());
  const auto bank = makeBank(store);
  FakeLiveScene scene({10, 20});
  OnlineSnapshotWorkerConfig config;
  config.event_queue_capacity = 3;
  OnlineSnapshotWorker worker(
      config, bank, [&scene]() { return scene.snapshot(); },
      [&scene](const ApplySnapshotSetCommand& command) {
        return scene.submit(command);
      });

  const auto shared_frame = makeFrame(7);
  ASSERT_TRUE(worker
                  .enqueueCandidate(forTrack(
                      5, makeCandidate(shared_frame, 1, 0.92f, -0.8f)))
                  .accepted());
  ASSERT_TRUE(worker
                  .enqueueCandidate(forObject(
                      20, makeCandidate(shared_frame, 1, 0.85f, 0.5f, 5.0f)))
                  .accepted());
  ASSERT_TRUE(worker.enqueuePromotion(5, 10).accepted());

  // The queue is full and promotion is protected. This candidate displaces
  // the original track event, but promotion's absorbed shared candidate must
  // still be submitted before ownership transfer.
  const auto displaced = worker.enqueueCandidate(forTrack(
      99, makeCandidate(shared_frame, 1, 0.80f, 1.4f, 8.0f)));
  ASSERT_EQ(displaced.outcome, PushOutcome::kReplaced);
  EXPECT_EQ(displaced.replacement_reason,
            ChannelReplacementReason::kCapacity);
  ASSERT_TRUE(displaced.replaced_candidate);
  EXPECT_EQ(*displaced.replaced_candidate,
            (SnapshotOwner{SnapshotOwnerKind::kTentativeTrack, 5}));

  worker.start();
  ASSERT_TRUE(worker.waitUntilIdle(std::chrono::seconds(5)));
  ASSERT_TRUE(bank->objectSnapshots(10));
  ASSERT_TRUE(bank->objectSnapshots(20));
  EXPECT_FALSE(bank->tentativeSnapshots(5));
  EXPECT_EQ(store->assetCount(), 1u)
      << "all ROIs from one full frame must share one physical asset";

  const auto first_commands = scene.accepted();
  ASSERT_EQ(first_commands.size(), 2u);
  EXPECT_EQ(first_commands[0].dependency.object_id, 20);
  EXPECT_EQ(first_commands[1].dependency.object_id, 10);
  ASSERT_FALSE(first_commands[1].snapshots.empty());
  EXPECT_TRUE(first_commands[1].snapshots.front().valid());

  ASSERT_TRUE(worker.enqueueDropTentative(99).accepted());
  ASSERT_TRUE(worker.waitUntilIdle(std::chrono::seconds(5)));
  EXPECT_FALSE(bank->tentativeSnapshots(99));

  // A later stable-object update races a scene identity update. The first CAS
  // is stale; the worker must repin and submit exactly once more.
  scene.rejectNextAsStale();
  ASSERT_TRUE(worker
                  .enqueueCandidate(forObject(
                      10, makeCandidate(makeFrame(9), 2, 0.97f, 2.2f)))
                  .accepted());
  ASSERT_TRUE(worker.waitUntilIdle(std::chrono::seconds(5)));
  const auto after_update = scene.accepted();
  ASSERT_EQ(after_update.size(), 3u);
  EXPECT_EQ(after_update.back().dependency.object_id, 10);
  EXPECT_EQ(after_update.back().dependency.identity_revision, 2u);
  EXPECT_EQ(worker.stats().command_stale_retries, 1u);

  // Scene merge is authoritative first; the reliable control event then
  // reranks the bank union and publishes one command for the current id.
  scene.merge(20, 10);
  ASSERT_TRUE(worker.enqueueMerge(20, 10).accepted());
  ASSERT_TRUE(worker.waitUntilIdle(std::chrono::seconds(5)));
  ASSERT_EQ(bank->resolveCanonicalObjectId(20), std::optional<int>(10));
  ASSERT_TRUE(bank->objectSnapshots(20));
  EXPECT_EQ(bank->objectSnapshots(20)->snapshot_set_hash,
            bank->objectSnapshots(10)->snapshot_set_hash);
  ASSERT_FALSE(scene.accepted().empty());
  EXPECT_EQ(scene.accepted().back().dependency.object_id, 10);

  scene.erase(10);
  ASSERT_TRUE(worker.enqueueEraseObject(10).accepted());
  worker.stop();  // stop must drain the erase control before joining.
  EXPECT_TRUE(worker.idle());
  EXPECT_FALSE(bank->objectSnapshots(10));
  EXPECT_TRUE(worker.queueStats().stopped);
}

TEST(OnlineSnapshotWorker,
     LatestByTrackReplacementCapacityDropAndShutdownDrainAreObservable) {
  TempDirectory temporary("replacement_shutdown");
  const auto store = makeStore(temporary.path());
  const auto bank = makeBank(store);
  FakeLiveScene scene({});
  OnlineSnapshotWorkerConfig config;
  config.event_queue_capacity = 4;
  OnlineSnapshotWorker worker(
      config, bank, [&scene]() { return scene.snapshot(); },
      [&scene](const ApplySnapshotSetCommand& command) {
        return scene.submit(command);
      });

  ASSERT_EQ(worker
                .enqueueCandidate(forTrack(
                    1, makeCandidate(makeFrame(1), 1, 0.50f, 0.0f)))
                .outcome,
            PushOutcome::kAccepted);
  const auto same_owner = worker.enqueueCandidate(forTrack(
      1, makeCandidate(makeFrame(2), 2, 0.90f, 0.2f)));
  ASSERT_EQ(same_owner.outcome, PushOutcome::kReplaced);
  EXPECT_EQ(same_owner.replacement_reason,
            ChannelReplacementReason::kMatchingKey);
  ASSERT_TRUE(same_owner.replaced_candidate);
  EXPECT_EQ(same_owner.replaced_candidate->id, 1);

  ASSERT_TRUE(worker
                  .enqueueCandidate(forTrack(
                      2, makeCandidate(makeFrame(3), 3, 0.70f, 0.4f)))
                  .accepted());
  ASSERT_TRUE(worker
                  .enqueueCandidate(forTrack(
                      3, makeCandidate(makeFrame(4), 4, 0.70f, 0.6f)))
                  .accepted());
  ASSERT_TRUE(worker
                  .enqueueCandidate(forTrack(
                      4, makeCandidate(makeFrame(5), 5, 0.70f, 0.8f)))
                  .accepted());
  const auto capacity_drop = worker.enqueueCandidate(forTrack(
      5, makeCandidate(makeFrame(6), 6, 0.70f, 1.0f)));
  ASSERT_EQ(capacity_drop.outcome, PushOutcome::kReplaced);
  EXPECT_EQ(capacity_drop.replacement_reason,
            ChannelReplacementReason::kCapacity);
  ASSERT_TRUE(capacity_drop.replaced_candidate);
  EXPECT_EQ(capacity_drop.replaced_candidate->id, 1);

  worker.start();
  for (int track_id = 1; track_id <= 5; ++track_id) {
    ASSERT_TRUE(worker.enqueueDropTentative(track_id).accepted());
  }
  worker.stop();  // no explicit wait: every accepted event must drain.

  EXPECT_TRUE(worker.idle());
  EXPECT_EQ(bank->tentativeTrackCount(), 0u);
  const OnlineSnapshotWorkerStats stats = worker.stats();
  EXPECT_EQ(stats.candidate_replaced_same_owner, 1u);
  EXPECT_EQ(stats.candidate_dropped_for_capacity, 1u);
  EXPECT_EQ(stats.candidate_processed, 4u);
  EXPECT_EQ(stats.control_enqueued, 5u);
  EXPECT_EQ(stats.control_processed, 5u);
  EXPECT_EQ(stats.control_failed, 0u) << worker.lastError();
  EXPECT_EQ(stats.events_processed, 9u);

  const auto after_stop = worker.enqueueCandidate(forTrack(
      9, makeCandidate(makeFrame(9), 9, 0.8f, 0.0f)));
  EXPECT_EQ(after_stop.outcome, PushOutcome::kStopped);
  EXPECT_FALSE(after_stop.accepted());
}

TEST(OnlineSnapshotWorker, TryControlNeverWaitsForSaturatedReliableQueue) {
  TempDirectory temporary("try_control_saturation");
  const auto store = makeStore(temporary.path());
  const auto bank = makeBank(store);
  FakeLiveScene scene({});
  OnlineSnapshotWorkerConfig config;
  config.event_queue_capacity = 1;
  OnlineSnapshotWorker worker(
      config, bank, [&scene]() { return scene.snapshot(); },
      [&scene](const ApplySnapshotSetCommand& command) {
        return scene.submit(command);
      });

  ASSERT_TRUE(worker.tryEnqueueDropTentative(1).accepted());
  const auto started = std::chrono::steady_clock::now();
  const SnapshotWorkerEnqueueResult saturated =
      worker.tryEnqueueMerge(2, 1);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_FALSE(saturated.accepted());
  EXPECT_EQ(saturated.outcome, PushOutcome::kTimedOut);
  EXPECT_LT(elapsed, std::chrono::milliseconds(50));

  worker.start();
  worker.stop();
  EXPECT_TRUE(worker.idle());
  EXPECT_EQ(worker.stats().control_enqueued, 1U);
}

TEST(OnlineSnapshotWorker,
     ReducerPublicationRetriesAndIdenticalCandidateRepairsAfterExhaustion) {
  TempDirectory temporary("publication_retry");
  const auto store = makeStore(temporary.path());
  const auto bank = makeBank(store);
  FakeLiveScene scene({10});
  std::atomic<bool> reject{true};
  std::atomic<std::uint64_t> sink_calls{0};
  OnlineSnapshotWorkerConfig config;
  config.poll_period = std::chrono::milliseconds(1);
  config.max_retry_attempts = 1;
  config.retry_initial_backoff = std::chrono::milliseconds(1);
  config.retry_max_backoff = std::chrono::milliseconds(2);
  OnlineSnapshotWorker worker(
      config, bank, [&scene]() { return scene.snapshot(); },
      [&scene, &reject, &sink_calls](const ApplySnapshotSetCommand& command) {
        sink_calls.fetch_add(1);
        if (reject.load()) {
          return SnapshotCommandSubmitResult{
              SnapshotCommandDisposition::kRejected,
              "simulated unavailable reducer"};
        }
        return scene.submit(command);
      });

  const auto frame = makeFrame(71);
  const SnapshotCandidate candidate =
      makeCandidate(frame, 1, 0.93f, 0.2f);
  worker.start();
  ASSERT_TRUE(worker.enqueueCandidate(forObject(10, candidate)).accepted());
  ASSERT_TRUE(worker.waitUntilIdle(std::chrono::seconds(5)));
  EXPECT_EQ(sink_calls.load(), 2U);
  EXPECT_EQ(worker.stats().retry_attempts, 1U);
  EXPECT_EQ(worker.stats().retry_exhausted, 1U);
  ASSERT_TRUE(bank->objectSnapshots(10));
  ASSERT_TRUE(scene.snapshot().findObject(10));
  EXPECT_NE(scene.snapshot().findObject(10)->artifact->snapshot_set_hash,
            bank->objectSnapshots(10)->snapshot_set_hash);

  // The bank now rejects the evidence as a duplicate. Reconciliation must
  // still publish its already-committed set instead of keying only off
  // effective_evidence_changed.
  reject.store(false);
  ASSERT_TRUE(worker.enqueueCandidate(forObject(10, candidate)).accepted());
  ASSERT_TRUE(worker.waitUntilIdle(std::chrono::seconds(5)));
  ASSERT_TRUE(scene.snapshot().findObject(10));
  EXPECT_EQ(scene.snapshot().findObject(10)->artifact->snapshot_set_hash,
            bank->objectSnapshots(10)->snapshot_set_hash);
  EXPECT_EQ(sink_calls.load(), 3U);
  worker.stop();
}

TEST(OnlineSnapshotWorker,
     PromotionAndMergeRetryIdempotentlyAfterCommittedBankMutation) {
  TempDirectory temporary("control_retry");
  const auto store = makeStore(temporary.path());
  const auto bank = makeBank(store);
  FakeLiveScene scene({10, 20});
  std::atomic<int> reject_remaining{1};
  std::atomic<std::uint64_t> sink_calls{0};
  OnlineSnapshotWorkerConfig config;
  config.poll_period = std::chrono::milliseconds(1);
  config.retry_initial_backoff = std::chrono::milliseconds(1);
  config.retry_max_backoff = std::chrono::milliseconds(4);
  OnlineSnapshotWorker worker(
      config, bank, [&scene]() { return scene.snapshot(); },
      [&scene, &reject_remaining,
       &sink_calls](const ApplySnapshotSetCommand& command) {
        sink_calls.fetch_add(1);
        int remaining = reject_remaining.load();
        while (remaining > 0 &&
               !reject_remaining.compare_exchange_weak(remaining,
                                                        remaining - 1)) {
        }
        if (remaining > 0) {
          return SnapshotCommandSubmitResult{
              SnapshotCommandDisposition::kRejected,
              "simulated transient reducer failure"};
        }
        return scene.submit(command);
      });

  worker.start();
  ASSERT_TRUE(worker
                  .enqueueCandidate(forTrack(
                      5, makeCandidate(makeFrame(81), 1, 0.91f, -0.4f)))
                  .accepted());
  ASSERT_TRUE(worker.enqueuePromotion(5, 10).accepted());
  ASSERT_TRUE(worker.waitUntilIdle(std::chrono::seconds(5)));
  ASSERT_TRUE(bank->objectSnapshots(10));
  EXPECT_EQ(scene.snapshot().findObject(10)->artifact->snapshot_set_hash,
            bank->objectSnapshots(10)->snapshot_set_hash);

  ASSERT_TRUE(worker
                  .enqueueCandidate(forObject(
                      20, makeCandidate(makeFrame(82), 2, 0.94f, 0.8f)))
                  .accepted());
  ASSERT_TRUE(worker.waitUntilIdle(std::chrono::seconds(5)));
  reject_remaining.store(1);
  scene.merge(20, 10);
  ASSERT_TRUE(worker.enqueueMerge(20, 10).accepted());
  ASSERT_TRUE(worker.waitUntilIdle(std::chrono::seconds(5)));
  ASSERT_TRUE(bank->objectSnapshots(10));
  EXPECT_EQ(scene.snapshot().findObject(10)->artifact->snapshot_set_hash,
            bank->objectSnapshots(10)->snapshot_set_hash);
  EXPECT_EQ(worker.stats().retry_attempts, 2U);
  EXPECT_EQ(worker.stats().retry_exhausted, 0U);
  EXPECT_EQ(worker.stats().control_failed, 0U) << worker.lastError();
  EXPECT_GE(sink_calls.load(), 5U);
  worker.stop();
}

TEST(OnlineSnapshotWorker, StopInterruptsLongRetryBackoff) {
  TempDirectory temporary("bounded_retry_stop");
  const auto store = makeStore(temporary.path());
  const auto bank = makeBank(store);
  FakeLiveScene scene({10});
  std::atomic<std::uint64_t> sink_calls{0};
  OnlineSnapshotWorkerConfig config;
  config.poll_period = std::chrono::milliseconds(1);
  config.max_retry_attempts = 100;
  config.retry_initial_backoff = std::chrono::seconds(5);
  config.retry_max_backoff = std::chrono::seconds(5);
  OnlineSnapshotWorker worker(
      config, bank, [&scene]() { return scene.snapshot(); },
      [&sink_calls](const ApplySnapshotSetCommand&) {
        sink_calls.fetch_add(1);
        return SnapshotCommandSubmitResult{
            SnapshotCommandDisposition::kRejected, "persistent failure"};
      });
  worker.start();
  ASSERT_TRUE(worker
                  .enqueueCandidate(forObject(
                      10, makeCandidate(makeFrame(91), 1, 0.93f, 0.1f)))
                  .accepted());
  const auto first_call_deadline = std::chrono::steady_clock::now() +
                                   std::chrono::seconds(2);
  while (sink_calls.load() == 0U &&
         std::chrono::steady_clock::now() < first_call_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_EQ(sink_calls.load(), 1U);

  const auto started = std::chrono::steady_clock::now();
  worker.stop();
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_LT(elapsed, std::chrono::milliseconds(500));
  EXPECT_TRUE(worker.idle());
  EXPECT_EQ(worker.stats().retry_abandoned_on_stop, 1U);
}

TEST(OnlineSnapshotWorker, PeriodicallyCollectsUnreferencedDurableAssets) {
  TempDirectory temporary("periodic_gc");
  const auto store = makeStore(temporary.path());
  const auto frame = makeFrame(33);
  const AssetWriteResult materialized = store->materializeFrame(*frame, 1);
  ASSERT_TRUE(materialized.success) << materialized.error;
  ASSERT_EQ(store->assetCount(), 1U);

  const auto bank = makeBank(store);
  FakeLiveScene scene({});
  OnlineSnapshotWorkerConfig config;
  config.poll_period = std::chrono::milliseconds(1);
  config.asset_gc_period = std::chrono::milliseconds(5);
  OnlineSnapshotWorker worker(
      config, bank, [&scene]() { return scene.snapshot(); },
      [&scene](const ApplySnapshotSetCommand& command) {
        return scene.submit(command);
      });
  worker.start();
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(2);
  while (worker.stats().asset_gc_runs == 0U &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  worker.stop();

  const OnlineSnapshotWorkerStats stats = worker.stats();
  EXPECT_GE(stats.asset_gc_runs, 1U);
  EXPECT_EQ(stats.asset_gc_failures, 0U) << worker.lastError();
  EXPECT_EQ(stats.assets_collected, 1U);
  EXPECT_EQ(store->assetCount(), 0U);
}

}  // namespace
}  // namespace roomie
