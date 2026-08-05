#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "roomie/pipeline/cpu_point_map_backend.hpp"
#include "roomie/pipeline/detection_bridge_thread.hpp"
#include "roomie/pipeline/map_thread.hpp"

namespace roomie {
namespace {

class FakeRevisionMapBackend final : public MapBackend {
 public:
  MapIntegrationResult integrateFrame(const FrameBundle& frame) override {
    if (frame.provenance.frame_id == throwing_frame_id) {
      throw std::runtime_error("injected integration exception");
    }
    if (integration_delay > std::chrono::milliseconds::zero()) {
      std::this_thread::sleep_for(integration_delay);
    }
    MapIntegrationResult result;
    result.integrated_through_ns = frame.provenance.sensor_time_ns;
    if (frame.provenance.frame_id == failing_frame_id) {
      result.error = "injected map failure";
      return result;
    }
    result.success = true;
    result.map_changed = true;
    result.map_revision = ++revision_;
    return result;
  }

  SurfaceRefreshResult refreshSurface(const MapIntegrationResult& integration,
                                      bool force_full_rebuild) override {
    ++surface_refreshes;
    last_force_full_rebuild = force_full_rebuild;
    SurfaceRefreshResult result;
    result.success = integration.success || force_full_rebuild;
    result.full_rebuild = true;
    result.source_map_revision = integration.success ? integration.map_revision
                                                     : revision_;
    MapSurfacePoint point;
    point.position_world = Eigen::Vector3f(
        0.0f, 0.0f, static_cast<float>(result.source_map_revision + 1U));
    point.weight = 1.0f;
    const float point_z = point.position_world.z();
    MapSurfacePointVector points{point};
    result.blocks.push_back(makeSurfaceBlock(
        BlockIndex(0, 0, 0),
        SurfaceAabb(Eigen::Vector3f(-0.1f, -0.1f, point_z - 0.1f),
                    Eigen::Vector3f(0.1f, 0.1f, point_z + 0.1f)),
        std::move(points)));
    MapSurfacePoint outside;
    outside.position_world = Eigen::Vector3f(100.0f, 0.0f, 2.0f);
    outside.weight = 1.0f;
    result.blocks.push_back(makeSurfaceBlock(
        BlockIndex(99, 0, 0),
        SurfaceAabb(Eigen::Vector3f(99.9f, -0.1f, 1.9f),
                    Eigen::Vector3f(100.1f, 0.1f, 2.1f)),
        MapSurfacePointVector{outside}));
    result.diagnostics.has_map = true;
    result.diagnostics.tsdf_blocks = 2;
    return result;
  }

  MapBackendSnapshot snapshot() const override { return {}; }

  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>>
  collectNearSurfaceVoxels(const RawDetection&) const override {
    return {};
  }

  MapCheckpointOperationResult loadCheckpoint(
      const MapCheckpointManifest* expected_manifest) override {
    ++checkpoint_loads;
    if (expected_manifest != nullptr) {
      loaded_manifest = *expected_manifest;
    }
    MapCheckpointOperationResult result;
    result.disposition = MapCheckpointDisposition::kSucceeded;
    if (expected_manifest != nullptr) {
      result.manifest = *expected_manifest;
    }
    return result;
  }

  MapCheckpointOperationResult saveCheckpoint(
      const MapStamp& map_stamp) override {
    ++checkpoint_saves;
    saved_stamp = map_stamp;
    MapCheckpointOperationResult result;
    result.disposition = MapCheckpointDisposition::kSucceeded;
    result.manifest.backend = "fake";
    result.manifest.checkpoint_path = "/tmp/fake-checkpoint";
    result.manifest.map_epoch = map_stamp.map_epoch;
    result.manifest.map_revision = map_stamp.map_revision;
    result.manifest.integrated_through_ns =
        map_stamp.integrated_through_ns;
    result.manifest.file_size_bytes = 1;
    result.manifest.content_hash = "fake:1";
    result.manifest.created_at_unix_ms = 1;
    return result;
  }

  FrameId failing_frame_id = 0;
  FrameId throwing_frame_id = 0;
  std::chrono::milliseconds integration_delay{0};
  int checkpoint_loads = 0;
  int checkpoint_saves = 0;
  int surface_refreshes = 0;
  bool last_force_full_rebuild = false;
  std::optional<MapCheckpointManifest> loaded_manifest;
  std::optional<MapStamp> saved_stamp;

 private:
  std::uint64_t revision_ = 0;
};

FrameBundlePtr frameBundle(
    FrameId frame_id,
    TimeNanoseconds time_ns,
    std::chrono::milliseconds deadline = std::chrono::seconds(1)) {
  auto frame = std::make_shared<FrameBundle>();
  frame->provenance.run_id = RunId{1, 2};
  frame->provenance.frame_id = frame_id;
  frame->provenance.sensor_time_ns = time_ns;
  frame->ingest_time = std::chrono::steady_clock::now();
  frame->due_time = frame->ingest_time + deadline;
  frame->perception_candidate = true;
  auto rgb = std::make_shared<ImageBuffer>();
  rgb->width = 8;
  rgb->height = 8;
  rgb->channels = 3;
  rgb->data.assign(8U * 8U * 3U, 0U);
  frame->rgb = std::move(rgb);
  frame->intrinsics.width = 8;
  frame->intrinsics.height = 8;
  frame->intrinsics.fx = 4.0f;
  frame->intrinsics.fy = 4.0f;
  frame->intrinsics.cx = 4.0f;
  frame->intrinsics.cy = 4.0f;
  return frame;
}

bool waitForMapRevision(const MapThread& actor, std::uint64_t revision) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (actor.mapVersion() >= revision) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

bool waitForExpiration(const std::weak_ptr<const FrameBundle>& frame,
                       std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (frame.expired()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return frame.expired();
}

class FakeAsyncMapProjector final : public MapProjector {
 public:
  bool enqueueFrameBundle(FrameBundlePtr) override { return true; }

  std::optional<PatchDepth> projectPatchDepth(const FrameBundle&) override {
    ++blocking_projection_calls;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return std::nullopt;
  }

  std::optional<PatchDepth> projectPatchDepth(
      const FrameBundle& frame, const MapCommit& commit) override {
    ++exact_projection_calls;
    PatchDepth patch;
    patch.valid_patches = PatchDepth::kSize;
    patch.projected_points = PatchDepth::kSize;
    patch.provenance = frame.provenance;
    patch.provenance.map = commit.map;
    patch.provenance.surface = commit.surface;
    patch.provenance.includes_current_frame = true;
    patch.provenance.causality_verified = true;
    patch.frustum_filter_ms = 1.0;
    patch.projection_loop_ms = 2.0;
    patch.projection_zbuffer_ms = 3.0;
    patch.projection_compute_ms = 6.0;
    return patch;
  }

  bool cancelPerceptionCandidate(FrameBundlePtr frame,
                                 const std::string& reason) override {
    std::lock_guard<std::mutex> lock(mutex_);
    cancellations.emplace_back(frame ? frame->provenance.frame_id : 0,
                               reason);
    return true;
  }

  MapBackendSnapshot snapshotSurfacePoints() const override { return {}; }

  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>>
  collectNearSurfaceVoxels(const RawDetection&) const override {
    return {};
  }

  std::atomic_int blocking_projection_calls{0};
  std::atomic_int exact_projection_calls{0};
  mutable std::mutex mutex_;
  std::vector<std::pair<FrameId, std::string>> cancellations;
};

class FakeAsyncInferenceBackend final : public InferenceBackend {
 public:
  PushResult<InferenceRequest> enqueueRequest(
      InferenceRequest request) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        PushResult<InferenceRequest> result;
        result.outcome = PushOutcome::kStopped;
        result.unconsumed_item = std::move(request);
        return result;
      }
      requests_.push_back(std::move(request));
      outstanding_ = true;
    }
    cv_.notify_all();
    PushResult<InferenceRequest> result;
    result.outcome = PushOutcome::kAccepted;
    return result;
  }

  bool tryPopResponse(InferenceResponse* response) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (responses_.empty()) {
      return false;
    }
    *response = std::move(responses_.front());
    responses_.pop_front();
    outstanding_ = false;
    return true;
  }

  bool idle() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return !outstanding_ && responses_.empty();
  }

  void beginShutdown() override {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
    if (outstanding_ && responses_.empty() && !requests_.empty()) {
      InferenceResponse terminal;
      const InferenceRequest& request = requests_.back();
      terminal.time_ns = request.time_ns;
      terminal.provenance = request.provenance;
      terminal.camera_id = request.camera_id;
      terminal.ok = false;
      terminal.error = "shutdown terminal";
      responses_.push_back(std::move(terminal));
    }
  }

  bool waitForRequests(std::size_t count,
                       std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout,
                        [&]() { return requests_.size() >= count; });
  }

  InferenceRequest request(std::size_t index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_.at(index);
  }

  void complete(const InferenceRequest& request, bool ok = true) {
    std::lock_guard<std::mutex> lock(mutex_);
    InferenceResponse response;
    response.time_ns = request.time_ns;
    response.provenance = request.provenance;
    response.camera_id = request.camera_id;
    response.ok = ok;
    if (!ok) {
      response.error = "injected failure";
    }
    responses_.push_back(std::move(response));
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<InferenceRequest> requests_;
  std::deque<InferenceResponse> responses_;
  bool outstanding_ = false;
  bool shutdown_ = false;
};

MapCommit exactCommit(const FrameBundlePtr& frame, std::uint64_t revision) {
  MapCommit commit;
  commit.run_id = frame->provenance.run_id;
  commit.frame_id = frame->provenance.frame_id;
  commit.success = true;
  commit.map.map_epoch = RunId{11, 12};
  commit.map.map_revision = revision;
  commit.map.integrated_through_ns = frame->provenance.sensor_time_ns;
  commit.surface.map_epoch = commit.map.map_epoch;
  commit.surface.surface_revision = revision;
  commit.surface.source_map_revision = revision;
  commit.includes_current_frame = true;
  commit.perception_candidate = true;
  commit.due_time = frame->due_time;
  commit.published_at = std::chrono::steady_clock::now();
  commit.frame_bundle = frame;
  // FakeAsyncMapProjector owns projection semantics; a non-null marker is
  // enough for the scheduler event contract in this deterministic test.
  SurfaceSnapshotBuilder builder(nullptr, commit.map, commit.surface);
  MapSurfacePoint point;
  point.position_world = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  builder.setBlock(makeSurfaceBlock(
      BlockIndex(0, 0, 0),
      SurfaceAabb(Eigen::Vector3f(-0.1f, -0.1f, 0.9f),
                  Eigen::Vector3f(0.1f, 0.1f, 1.1f)),
      MapSurfacePointVector{point}));
  commit.snapshot = builder.build().snapshot;
  return commit;
}

bool waitForInferenceResponse(ThreadSafeQueue<InferenceResponse>* queue,
                              InferenceResponse* response,
                              std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (queue->tryPop(response)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return queue->tryPop(response);
}

class ScopedRclcppInit {
 public:
  ScopedRclcppInit() {
    if (!rclcpp::ok()) {
      int argc = 0;
      char** argv = nullptr;
      rclcpp::init(argc, argv);
      owns_context_ = true;
    }
  }

  ~ScopedRclcppInit() {
    if (owns_context_ && rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }

 private:
  bool owns_context_ = false;
};

TEST(MapCausalBarrier, PinsExactCommitAfterActorAdvancesToNewerFrame) {
  ThreadSafeQueue<FrameBundlePtr> queue(4, ChannelPolicy::kReliableBlocking);
  PipelineConfig config;
  config.freeze_tsdf_map = false;
  config.perception_deadline_ms = 1000;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  MapThread actor(queue, config, std::move(backend));

  const FrameBundlePtr first = frameBundle(1, 100);
  const FrameBundlePtr second = frameBundle(2, 200);
  ASSERT_TRUE(queue.push(first).accepted());
  ASSERT_TRUE(queue.push(second).accepted());
  actor.start();
  ASSERT_TRUE(waitForMapRevision(actor, 2));

  const auto first_patch = actor.projectPatchDepth(*first);
  const auto second_patch = actor.projectPatchDepth(*second);
  ASSERT_TRUE(first_patch);
  ASSERT_TRUE(second_patch);
  EXPECT_EQ(first_patch->provenance.map.map_revision, 1U);
  EXPECT_EQ(first_patch->provenance.surface.source_map_revision, 1U);
  EXPECT_EQ(second_patch->provenance.map.map_revision, 2U);
  EXPECT_EQ(second_patch->provenance.surface.source_map_revision, 2U);
  EXPECT_TRUE(first_patch->provenance.includes_current_frame);
  EXPECT_TRUE(second_patch->provenance.includes_current_frame);
  EXPECT_TRUE(first_patch->provenance.causality_verified);
  EXPECT_TRUE(second_patch->provenance.causality_verified);
  EXPECT_TRUE(first_patch->view_filtered);
  EXPECT_EQ(first_patch->source_selected_blocks, 1U);
  EXPECT_EQ(first_patch->source_surface_points, 1U);
  EXPECT_EQ(first_patch->source_cached_surface_points, 2U);
  EXPECT_LT(first_patch->provenance.surface.surface_revision,
            second_patch->provenance.surface.surface_revision);

  actor.stop();
  queue.stop();
}

TEST(MapCausalBarrier, FailedMapCommitNeverEntersPerception) {
  ThreadSafeQueue<FrameBundlePtr> queue(2, ChannelPolicy::kReliableBlocking);
  PipelineConfig config;
  config.perception_deadline_ms = 100;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  backend->failing_frame_id = 7;
  MapThread actor(queue, config, std::move(backend));
  const FrameBundlePtr failed = frameBundle(7, 700);
  ASSERT_TRUE(queue.push(failed).accepted());
  actor.start();
  EXPECT_FALSE(actor.projectPatchDepth(*failed));
  actor.stop();
  queue.stop();
}

TEST(MapCausalBarrier, DetectionMayArriveBeforeItsCommit) {
  ThreadSafeQueue<FrameBundlePtr> queue(2, ChannelPolicy::kDropOldest);
  PipelineConfig config;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  MapThread actor(queue, config, std::move(backend));
  actor.start();

  const FrameBundlePtr frame = frameBundle(11, 1100);
  auto projected = std::async(std::launch::async, [&actor, frame]() {
    return actor.projectPatchDepth(*frame);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ASSERT_TRUE(actor.enqueueFrameBundle(frame));
  const auto patch = projected.get();
  ASSERT_TRUE(patch);
  EXPECT_EQ(patch->provenance.frame_id, 11U);
  EXPECT_EQ(patch->provenance.surface.source_map_revision, 1U);

  actor.stop();
  queue.stop();
}

TEST(MapCausalBarrier, JoinIsPublishedAndNotifiedBeforeObserverRuns) {
  ThreadSafeQueue<FrameBundlePtr> queue(2, ChannelPolicy::kDropOldest);
  PipelineConfig config;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  MapThread actor(queue, config, std::move(backend));
  const FrameBundlePtr frame = frameBundle(16, 1600);

  std::promise<void> waiter_started_signal;
  std::future<void> waiter_started = waiter_started_signal.get_future();
  auto projected =
      std::async(std::launch::async, [&actor, frame, &waiter_started_signal]() {
        waiter_started_signal.set_value();
        return actor.projectPatchDepth(*frame);
      }).share();
  std::promise<bool> waiter_ready_in_observer;
  std::future<bool> observer_result = waiter_ready_in_observer.get_future();
  actor.setCommitObserver(
      [&projected, &waiter_ready_in_observer](const MapCommit& commit) {
        if (commit.frame_id != 16) {
          return;
        }
        waiter_ready_in_observer.set_value(
            projected.wait_for(std::chrono::milliseconds(250)) ==
            std::future_status::ready);
      });

  actor.start();
  ASSERT_EQ(waiter_started.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  ASSERT_TRUE(actor.enqueueFrameBundle(frame));
  ASSERT_EQ(observer_result.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  EXPECT_TRUE(observer_result.get());
  const auto patch = projected.get();
  ASSERT_TRUE(patch);
  EXPECT_EQ(patch->provenance.frame_id, 16U);
  EXPECT_TRUE(patch->provenance.causality_verified);

  actor.stop();
  queue.stop();
}

TEST(MapCausalBarrier, SameIdFromDifferentRunCannotConsumeCommit) {
  ThreadSafeQueue<FrameBundlePtr> queue(2, ChannelPolicy::kReliableBlocking);
  PipelineConfig config;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  MapThread actor(queue, config, std::move(backend));
  const FrameBundlePtr original = frameBundle(12, 1200);
  ASSERT_TRUE(queue.push(original));
  actor.start();
  ASSERT_TRUE(waitForMapRevision(actor, 1));

  auto wrong_run = std::make_shared<FrameBundle>(*original);
  wrong_run->provenance.run_id = RunId{9, 9};
  wrong_run->due_time =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
  EXPECT_FALSE(actor.projectPatchDepth(*wrong_run));

  actor.stop();
  queue.stop();
}

TEST(MapCausalBarrier, SameRunAndIdStillRequiresExactSharedBundle) {
  ThreadSafeQueue<FrameBundlePtr> queue(2, ChannelPolicy::kReliableBlocking);
  PipelineConfig config;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  MapThread actor(queue, config, std::move(backend));
  const FrameBundlePtr original = frameBundle(13, 1300);
  ASSERT_TRUE(queue.push(original));
  actor.start();
  ASSERT_TRUE(waitForMapRevision(actor, 1));

  const auto copied_bundle = std::make_shared<FrameBundle>(*original);
  EXPECT_FALSE(actor.projectPatchDepth(*copied_bundle));
  const auto original_patch = actor.projectPatchDepth(*original);
  ASSERT_TRUE(original_patch);
  EXPECT_TRUE(original_patch->provenance.causality_verified);

  actor.stop();
  queue.stop();
}

TEST(MapCausalBarrier, CommitFinishingAfterDeadlineIsNotRetainedForJoin) {
  ThreadSafeQueue<FrameBundlePtr> queue(2, ChannelPolicy::kReliableBlocking);
  PipelineConfig config;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  backend->integration_delay = std::chrono::milliseconds(80);
  MapThread actor(queue, config, std::move(backend));

  std::promise<void> observer_called;
  std::future<void> observer_result = observer_called.get_future();
  actor.setCommitObserver([&observer_called](const MapCommit& commit) {
    if (commit.frame_id == 17) {
      observer_called.set_value();
    }
  });
  FrameBundlePtr expired =
      frameBundle(17, 1700, std::chrono::milliseconds(20));
  std::weak_ptr<const FrameBundle> weak_expired = expired;
  ASSERT_TRUE(queue.push(expired).accepted());
  actor.start();

  ASSERT_EQ(observer_result.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  EXPECT_FALSE(actor.projectPatchDepth(*expired));
  expired.reset();
  EXPECT_TRUE(waitForExpiration(weak_expired, std::chrono::milliseconds(500)));

  actor.stop();
  queue.stop();
}

TEST(MapCausalBarrier, UnconsumedCandidateCommitExpiresWhileActorIsIdle) {
  ThreadSafeQueue<FrameBundlePtr> queue(2, ChannelPolicy::kReliableBlocking);
  PipelineConfig config;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  MapThread actor(queue, config, std::move(backend));

  FrameBundlePtr orphan =
      frameBundle(18, 1800, std::chrono::milliseconds(150));
  std::weak_ptr<const FrameBundle> weak_orphan = orphan;
  ASSERT_TRUE(queue.push(orphan).accepted());
  orphan.reset();
  actor.start();
  ASSERT_TRUE(waitForMapRevision(actor, 1));
  EXPECT_TRUE(waitForExpiration(weak_orphan, std::chrono::seconds(1)));

  actor.stop();
  queue.stop();
}

TEST(MapCausalBarrier, BackendExceptionIsTerminalForFrameAndActorSurvives) {
  ThreadSafeQueue<FrameBundlePtr> queue(3, ChannelPolicy::kReliableBlocking);
  PipelineConfig config;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  backend->throwing_frame_id = 14;
  MapThread actor(queue, config, std::move(backend));
  const FrameBundlePtr throwing = frameBundle(14, 1400);
  const FrameBundlePtr healthy = frameBundle(15, 1500);
  ASSERT_TRUE(queue.push(throwing));
  ASSERT_TRUE(queue.push(healthy));
  actor.start();

  EXPECT_FALSE(actor.projectPatchDepth(*throwing));
  const auto patch = actor.projectPatchDepth(*healthy);
  ASSERT_TRUE(patch);
  EXPECT_TRUE(patch->provenance.causality_verified);

  actor.stop();
  queue.stop();
}

TEST(MapCausalBarrier, ObserverCommitProjectsWithoutWaitingAndPinsExactBundle) {
  ThreadSafeQueue<FrameBundlePtr> queue(2, ChannelPolicy::kDropOldest);
  PipelineConfig config;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  MapThread actor(queue, config, std::move(backend));
  const FrameBundlePtr frame = frameBundle(21, 2100);

  std::promise<MapCommit> published;
  std::future<MapCommit> published_future = published.get_future();
  actor.setCommitObserver([&published](const MapCommit& commit) {
    if (commit.frame_id == 21) {
      published.set_value(commit);
    }
  });
  actor.start();
  ASSERT_TRUE(actor.enqueueFrameBundle(frame));
  ASSERT_EQ(published_future.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  const MapCommit commit = published_future.get();
  ASSERT_TRUE(commit.success) << commit.error;
  ASSERT_EQ(commit.frame_bundle.get(), frame.get());

  const auto copied = std::make_shared<FrameBundle>(*frame);
  EXPECT_FALSE(actor.projectPatchDepth(*copied, commit));
  const auto projection_start = std::chrono::steady_clock::now();
  const auto patch = actor.projectPatchDepth(*frame, commit);
  const auto projection_elapsed =
      std::chrono::steady_clock::now() - projection_start;
  ASSERT_TRUE(patch);
  EXPECT_LT(projection_elapsed, std::chrono::milliseconds(100));
  EXPECT_TRUE(patch->provenance.causality_verified);
  EXPECT_TRUE(patch->provenance.includes_current_frame);
  EXPECT_EQ(patch->provenance.surface, commit.surface);
  EXPECT_GT(patch->projection_compute_ms, 0.0);

  actor.stop();
  queue.stop();
}

TEST(MapCausalBarrier, SupersededQueuedCandidateCancelsProtectedMapWork) {
  ThreadSafeQueue<FrameBundlePtr> queue(3, ChannelPolicy::kDropOldest);
  PipelineConfig config;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  backend->integration_delay = std::chrono::milliseconds(100);
  MapThread actor(queue, config, std::move(backend));
  const FrameBundlePtr active = frameBundle(22, 2200);
  const FrameBundlePtr stale = frameBundle(23, 2300);
  ASSERT_TRUE(actor.enqueueFrameBundle(active));
  ASSERT_TRUE(actor.enqueueFrameBundle(stale));
  actor.start();

  const auto pop_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (queue.stats().dequeued == 0 &&
         std::chrono::steady_clock::now() < pop_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_EQ(queue.stats().dequeued, 1U);
  ASSERT_TRUE(actor.cancelPerceptionCandidate(stale, "superseded"));
  EXPECT_EQ(queue.stats().cancelled, 1U);
  EXPECT_TRUE(waitForMapRevision(actor, 1));
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  EXPECT_EQ(actor.mapVersion(), 1U);
  EXPECT_FALSE(actor.projectPatchDepth(*stale));

  actor.stop();
  queue.stop();
}

TEST(DetectionBridgeAsyncJoin,
     CompletedResponseIsForwardedWhileNextExactCommitIsDelayed) {
  ScopedRclcppInit rclcpp_init;
  auto node = std::make_shared<rclcpp::Node>("roomie_async_join_hol_test");
  ThreadSafeQueue<FrameBundlePtr> detections(
      2,
      ChannelPolicy::kLatestByKey,
      [](const FrameBundlePtr& lhs, const FrameBundlePtr& rhs) {
        return lhs && rhs && lhs->camera_id == rhs->camera_id;
      });
  ThreadSafeQueue<InferenceResponse> reducer_responses(
      2, ChannelPolicy::kReliableBlocking);
  FakeAsyncMapProjector projector;
  FakeAsyncInferenceBackend backend;
  PipelineConfig config;
  config.boxer_input_size = 8;
  config.min_patch_coverage_ratio = 0.01f;
  config.file_logging_period_sec = 60.0;
  DetectionBridgeThread bridge(*node,
                               detections,
                               reducer_responses,
                               projector,
                               backend,
                               config);

  FrameBundlePtr first = frameBundle(31, 3100, std::chrono::seconds(2));
  FrameBundlePtr second = frameBundle(32, 3200, std::chrono::seconds(2));
  std::const_pointer_cast<FrameBundle>(first)->camera_id = "camera";
  std::const_pointer_cast<FrameBundle>(second)->camera_id = "camera";
  bridge.onMapCommit(exactCommit(first, 1));  // commit-before-frame order
  ASSERT_TRUE(detections.push(first));
  bridge.start();
  ASSERT_TRUE(backend.waitForRequests(1, std::chrono::seconds(1)));
  const InferenceRequest first_request = backend.request(0);

  // No commit is supplied for the second candidate. The legacy synchronous
  // bridge would enter its deliberately slow blocking projector here.
  ASSERT_TRUE(detections.push(second));
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  const auto completed_at = std::chrono::steady_clock::now();
  backend.complete(first_request);

  InferenceResponse forwarded;
  ASSERT_TRUE(waitForInferenceResponse(
      &reducer_responses, &forwarded, std::chrono::milliseconds(150)));
  EXPECT_LT(std::chrono::steady_clock::now() - completed_at,
            std::chrono::milliseconds(150));
  EXPECT_EQ(forwarded.provenance.request_id,
            first_request.provenance.request_id);
  EXPECT_EQ(projector.blocking_projection_calls.load(), 0);
  EXPECT_EQ(projector.exact_projection_calls.load(), 1);

  bridge.stop();
  detections.stop();
  reducer_responses.stop();
  node.reset();
}

TEST(DetectionBridgeAsyncJoin,
     ShutdownDrainsTerminalResponseForAcceptedBackendRequest) {
  ScopedRclcppInit rclcpp_init;
  auto node = std::make_shared<rclcpp::Node>("roomie_async_join_stop_test");
  ThreadSafeQueue<FrameBundlePtr> detections(
      1,
      ChannelPolicy::kLatestByKey,
      [](const FrameBundlePtr& lhs, const FrameBundlePtr& rhs) {
        return lhs && rhs && lhs->camera_id == rhs->camera_id;
      });
  ThreadSafeQueue<InferenceResponse> reducer_responses(
      2, ChannelPolicy::kReliableBlocking);
  FakeAsyncMapProjector projector;
  FakeAsyncInferenceBackend backend;
  PipelineConfig config;
  config.boxer_input_size = 8;
  config.min_patch_coverage_ratio = 0.01f;
  config.file_logging_period_sec = 60.0;
  DetectionBridgeThread bridge(*node,
                               detections,
                               reducer_responses,
                               projector,
                               backend,
                               config);
  FrameBundlePtr frame = frameBundle(33, 3300, std::chrono::seconds(2));
  std::const_pointer_cast<FrameBundle>(frame)->camera_id = "camera";
  bridge.onMapCommit(exactCommit(frame, 1));
  ASSERT_TRUE(detections.push(frame));
  bridge.start();
  ASSERT_TRUE(backend.waitForRequests(1, std::chrono::seconds(1)));
  const RequestId accepted_id = backend.request(0).provenance.request_id;

  bridge.stop();
  InferenceResponse terminal;
  ASSERT_TRUE(waitForInferenceResponse(
      &reducer_responses, &terminal, std::chrono::milliseconds(100)));
  EXPECT_EQ(terminal.provenance.request_id, accepted_id);
  EXPECT_FALSE(terminal.ok);
  EXPECT_NE(terminal.error.find("shutdown"), std::string::npos);
  EXPECT_FALSE(bridge.perceptionBusy());

  detections.stop();
  reducer_responses.stop();
  node.reset();
}

TEST(MapCausalBarrier, FrozenSnapshotIsVerifiedButExcludesCurrentFrame) {
  ThreadSafeQueue<FrameBundlePtr> queue(1, ChannelPolicy::kReliableBlocking);
  PipelineConfig config;
  config.freeze_tsdf_map = true;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  MapThread actor(queue, config, std::move(backend));
  actor.start();

  const auto ready_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!actor.snapshotSurfacePoints().has_map &&
         std::chrono::steady_clock::now() < ready_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(actor.snapshotSurfacePoints().has_map);

  const FrameBundlePtr frozen_frame = frameBundle(9, 900);
  const auto patch = actor.projectPatchDepth(*frozen_frame);
  ASSERT_TRUE(patch);
  EXPECT_EQ(patch->provenance.map_mode, MapMode::kFrozen);
  EXPECT_FALSE(patch->provenance.includes_current_frame);
  EXPECT_TRUE(patch->provenance.causality_verified);

  actor.stop();
  queue.stop();
}

TEST(MapCheckpoint, ExplicitBoundaryLoadsOnceAndSavesStoppedMapStamp) {
  ThreadSafeQueue<FrameBundlePtr> queue(2, ChannelPolicy::kReliableBlocking);
  PipelineConfig config;
  config.load_map = true;
  config.save_map = true;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  FakeRevisionMapBackend* const backend_view = backend.get();
  MapThread actor(queue, config, std::move(backend));

  MapCheckpointManifest expected;
  expected.backend = "fake";
  expected.checkpoint_path = "/tmp/fake-loaded-checkpoint";
  expected.world_frame = "world";
  expected.config_fingerprint = "fake.v1";
  expected.map_epoch = RunId{8, 9};
  expected.map_revision = 6;
  expected.file_size_bytes = 1;
  expected.content_hash = "fake:loaded";
  expected.created_at_unix_ms = 1;
  ASSERT_TRUE(actor.loadConfiguredCheckpoint(&expected).succeeded());
  EXPECT_EQ(backend_view->checkpoint_loads, 1);
  ASSERT_TRUE(backend_view->loaded_manifest.has_value());
  EXPECT_EQ(backend_view->loaded_manifest->checkpoint_path,
            expected.checkpoint_path);
  EXPECT_EQ(actor.loadConfiguredCheckpoint(&expected).disposition,
            MapCheckpointDisposition::kFailed);

  const FrameBundlePtr frame = frameBundle(25, 2500);
  ASSERT_TRUE(queue.push(frame).accepted());
  actor.start();
  ASSERT_TRUE(waitForMapRevision(actor, 1));
  EXPECT_EQ(actor.saveConfiguredCheckpoint().disposition,
            MapCheckpointDisposition::kFailed);
  actor.stop();
  queue.stop();

  const MapCheckpointOperationResult saved =
      actor.saveConfiguredCheckpoint();
  ASSERT_TRUE(saved.succeeded()) << saved.error;
  ASSERT_TRUE(backend_view->saved_stamp.has_value());
  EXPECT_TRUE(backend_view->saved_stamp->map_epoch.valid());
  EXPECT_EQ(backend_view->saved_stamp->map_revision, 1U);
  EXPECT_EQ(backend_view->saved_stamp->integrated_through_ns, 2500);
  EXPECT_EQ(backend_view->checkpoint_saves, 1);
  EXPECT_EQ(actor.saveConfiguredCheckpoint().disposition,
            MapCheckpointDisposition::kFailed);
}

TEST(MapCheckpoint, LoadedOnlineMapPublishesSurfaceBeforeFirstFrame) {
  ThreadSafeQueue<FrameBundlePtr> queue(1, ChannelPolicy::kReliableBlocking);
  PipelineConfig config;
  config.load_map = true;
  config.freeze_tsdf_map = false;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  FakeRevisionMapBackend* const backend_view = backend.get();
  MapThread actor(queue, config, std::move(backend));

  MapCheckpointManifest expected;
  expected.backend = "fake";
  expected.checkpoint_path = "/tmp/fake-loaded-checkpoint";
  expected.world_frame = "world";
  expected.config_fingerprint = "fake.v1";
  expected.map_epoch = RunId{8, 9};
  expected.map_revision = 6;
  expected.file_size_bytes = 1;
  expected.content_hash = "fake:loaded";
  expected.created_at_unix_ms = 1;
  ASSERT_TRUE(actor.loadConfiguredCheckpoint(&expected).succeeded());

  actor.start();
  const auto ready_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!actor.snapshotSurfacePoints().has_map &&
         std::chrono::steady_clock::now() < ready_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const MapBackendSnapshot restored = actor.snapshotSurfacePoints();
  EXPECT_TRUE(restored.has_map);
  EXPECT_EQ(restored.cached_surface_points, 2U);
  EXPECT_EQ(backend_view->surface_refreshes, 1);
  EXPECT_TRUE(backend_view->last_force_full_rebuild);

  actor.stop();
  queue.stop();
}

TEST(MapCheckpoint, CpuBackendReportsCheckpointPersistenceUnsupported) {
  PipelineConfig config;
  config.load_map = true;
  config.save_map = true;
  CpuPointMapBackend backend(config);
  MapStamp stamp;
  stamp.map_epoch = RunId{1, 1};
  EXPECT_EQ(backend.loadCheckpoint(nullptr).disposition,
            MapCheckpointDisposition::kUnsupported);
  EXPECT_EQ(backend.saveCheckpoint(stamp).disposition,
            MapCheckpointDisposition::kUnsupported);
}

TEST(MapCheckpoint, DurableManifestLoadsWhenFreshRunLoadFlagIsFalse) {
  ThreadSafeQueue<FrameBundlePtr> queue(1, ChannelPolicy::kReliableBlocking);
  PipelineConfig config;
  config.load_map = false;
  auto backend = std::make_unique<FakeRevisionMapBackend>();
  FakeRevisionMapBackend* const backend_view = backend.get();
  MapThread actor(queue, config, std::move(backend));

  MapCheckpointManifest durable;
  durable.backend = "fake";
  durable.checkpoint_path = "/tmp/fake-durable-checkpoint";
  durable.world_frame = "world";
  durable.config_fingerprint = "fake.v1";
  durable.map_epoch = RunId{10, 11};
  durable.map_revision = 9;
  durable.file_size_bytes = 1;
  durable.content_hash = "fake:durable";
  durable.created_at_unix_ms = 1;

  EXPECT_EQ(actor.loadConfiguredCheckpoint().disposition,
            MapCheckpointDisposition::kNotRequested);
  ASSERT_TRUE(actor.loadConfiguredCheckpoint(&durable).succeeded());
  ASSERT_TRUE(backend_view->loaded_manifest.has_value());
  EXPECT_EQ(backend_view->loaded_manifest->checkpoint_path,
            durable.checkpoint_path);
}

}  // namespace
}  // namespace roomie
