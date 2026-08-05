#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "roomie/artifacts/artifact_runtime_actor.hpp"

namespace roomie {
namespace {

using namespace std::chrono_literals;

class TemporaryRuntimeDatabase {
 public:
  TemporaryRuntimeDatabase() {
    char pattern[] = "/tmp/roomie_artifact_runtime_XXXXXX";
    const int descriptor = mkstemp(pattern);
    if (descriptor < 0) {
      throw std::runtime_error("mkstemp failed");
    }
    close(descriptor);
    path_ = pattern;
  }

  ~TemporaryRuntimeDatabase() {
    std::remove(path_.c_str());
    std::remove((path_ + "-wal").c_str());
    std::remove((path_ + "-shm").c_str());
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

std::int64_t unixMillis() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template <typename Predicate>
bool waitFor(Predicate predicate, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(1ms);
  }
  return predicate();
}

ObjectNode runtimeNode() {
  ObjectNode node;
  node.object_id = 7;
  node.semantic_id = 107;
  node.label = "chair";
  node.center_world = Eigen::Vector3f(1.0f, 2.0f, 0.5f);
  node.size_m = Eigen::Vector3f(0.5f, 0.6f, 0.9f);
  node.active = true;
  node.publishable = true;
  return node;
}

struct PreparedRuntimeScene {
  SceneSnapshot snapshot;
  ObjectDependency dependency;
  std::string snapshot_set_hash;
};

PreparedRuntimeScene prepareRuntimeScene(SceneStore* store) {
  ReducerCore reducer;
  std::vector<SceneSnapshot> history;

  LoadSceneCommand load;
  load.graph.objects.push_back(runtimeNode());
  load.graph.next_object_id = 8;
  SceneApplyResult result = reducer.apply(SceneCommand{load});
  if (!result.committedRevision()) {
    throw std::runtime_error(result.reason);
  }
  history.push_back(result.snapshot);

  ApplySnapshotSetCommand snapshots;
  snapshots.dependency = dependencyFor(*result.snapshot.findObject(7));
  ObjectSnapshotRef reference;
  reference.image_index = 3;
  reference.camera_id = "head_rgbd";
  reference.quality = 0.95f;
  snapshots.snapshots.push_back(reference);
  snapshots.snapshot_set_hash = "runtime-snapshot-set";
  result = reducer.apply(SceneCommand{snapshots});
  if (!result.committedRevision()) {
    throw std::runtime_error(result.reason);
  }
  history.push_back(result.snapshot);

  for (const SceneSnapshot& snapshot : history) {
    const SceneStoreStatus enqueued = store->enqueueCommit(snapshot);
    if (!enqueued) {
      throw std::runtime_error(enqueued.error);
    }
  }
  const SceneStoreStatus flushed = store->flush();
  if (!flushed) {
    throw std::runtime_error(flushed.error);
  }
  const SceneRestoreResult restored = store->restoreLatest();
  if (!restored.status || !restored.found) {
    throw std::runtime_error(restored.status.error);
  }

  PreparedRuntimeScene prepared;
  prepared.snapshot = restored.snapshot;
  prepared.dependency =
      dependencyFor(*prepared.snapshot.findExactObject(7));
  prepared.snapshot_set_hash = snapshots.snapshot_set_hash;
  return prepared;
}

DamTaskIntent runtimeIntent(const PreparedRuntimeScene& scene,
                            std::int64_t created_unix_ms) {
  DamTaskIntent intent;
  intent.key.object_id = scene.dependency.object_id;
  intent.key.identity_revision = scene.dependency.identity_revision;
  intent.key.appearance_revision = scene.dependency.appearance_revision;
  intent.key.snapshot_set_hash = scene.snapshot_set_hash;
  intent.key.model_id = "fake-runtime-dam";
  intent.key.prompt_hash = "fake-runtime-prompt";
  intent.key.output_schema_version = "dam.schema.v1";
  intent.dependency = scene.dependency;
  intent.scene_revision = scene.snapshot.revision();
  intent.priority = ArtifactPriority::kInteractive;
  intent.created_unix_ms = created_unix_ms;
  intent.input_payload = "asset:runtime-frame";
  return intent;
}

std::string artifactJson(std::string description) {
  return nlohmann::json{
      {"canonical_name", "chair"},
      {"short_description", std::move(description)},
      {"retrieval_text", "red upholstered chair with four legs"},
      {"visual_attributes",
       {{"colors", nlohmann::json::array({"red"})},
        {"materials", nlohmann::json::array({"fabric"})},
        {"shape", nlohmann::json::array({"rectangular"})},
        {"visible_parts", nlohmann::json::array({"seat", "legs"})},
        {"state_or_pose", nlohmann::json::array({"upright"})},
        {"distinctive_marks", nlohmann::json::array()},
        {"visible_text", nlohmann::json::array()}}},
      {"uncertain_or_not_visible", nlohmann::json::array()},
      {"confidence", 0.94},
      {"evidence_snapshot_ids", nlohmann::json::array({"runtime-frame"})},
      {"mask_source", "instance_mask"}}
      .dump();
}

DamWorkerResponse successfulResponse(std::string description) {
  DamWorkerResponse response;
  response.success = true;
  response.raw_output = artifactJson(std::move(description));
  return response;
}

class SequenceDamWorker final : public DamWorker {
 public:
  explicit SequenceDamWorker(std::vector<DamWorkerResponse> responses)
      : responses_(std::move(responses)) {
    if (responses_.empty()) {
      throw std::invalid_argument("fake worker needs a response");
    }
  }

  DamWorkerResponse describe(const DamTaskRequest& request,
                             const DamLeaseHeartbeat&) override {
    std::lock_guard<std::mutex> lock(mutex_);
    requests_.push_back(request);
    const std::size_t index =
        std::min(calls_, responses_.size() - 1);
    ++calls_;
    return responses_[index];
  }

  std::size_t calls() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return calls_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<DamWorkerResponse> responses_;
  std::vector<DamTaskRequest> requests_;
  std::size_t calls_ = 0;
};

class BlockingDamWorker final : public DamWorker {
 public:
  DamWorkerResponse describe(const DamTaskRequest&,
                             const DamLeaseHeartbeat&) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      started_ = true;
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return released_; });
    finished_ = true;
    lock.unlock();
    cv_.notify_all();
    return successfulResponse("late detached result");
  }

  bool waitUntilStarted(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [this]() { return started_; });
  }

  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
    }
    cv_.notify_all();
  }

  bool waitUntilFinished(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [this]() { return finished_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool started_ = false;
  bool released_ = false;
  bool finished_ = false;
};

class FakeReducerSink {
 public:
  explicit FakeReducerSink(SceneSnapshot snapshot)
      : snapshot_(std::move(snapshot)) {}

  SceneSnapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
  }

  ArtifactReducerSubmitResult submit(
      const ApplyDescriptionArtifactCommand& command,
      std::chrono::milliseconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    commands_.push_back(command);
    const SceneObjectPtr object =
        snapshot_.findExactObject(command.dependency.object_id);
    if (!object || !object->artifact) {
      SceneApplyResult rejected;
      rejected.status = SceneApplyStatus::kRejected;
      rejected.revision = snapshot_.revision();
      rejected.snapshot = snapshot_;
      rejected.reason = "fake reducer object is absent";
      return ArtifactReducerSubmitResult::success(std::move(rejected));
    }

    SceneState next = *snapshot_.statePtr();
    SceneObjectTable objects = snapshot_.objects();
    auto updated = std::make_shared<SceneObject>(*object);
    auto artifact = std::make_shared<ArtifactComponent>(*object->artifact);
    const SceneRevision revision = next.latest_scene_revision + 1;
    ++artifact->revision;
    artifact->pending_description_scene_revision = revision;
    artifact->pending_description = command.description;
    artifact->pending_description_input_hash = command.input_hash;
    artifact->pending_description_model_id = command.model_id;
    artifact->pending_description_schema_version = command.schema_version;
    artifact->pending_description_raw_text = command.raw_text;
    artifact->pending_description_normalized_json = command.normalized_json;
    artifact->pending_description_durable_envelope_json =
        command.durable_envelope_json;
    artifact->pending_description_parse_path = command.parse_path;
    artifact->pending_description_slo = command.artifact_slo;
    updated->artifact = std::move(artifact);
    objects[command.dependency.object_id] = std::move(updated);
    next.objects =
        std::make_shared<const SceneObjectTable>(std::move(objects));
    next.latest_scene_revision = revision;
    snapshot_ =
        SceneSnapshot{std::make_shared<const SceneState>(std::move(next))};

    SceneApplyResult committed;
    committed.status = SceneApplyStatus::kCommitted;
    committed.revision = revision;
    committed.snapshot = snapshot_;
    return ArtifactReducerSubmitResult::success(std::move(committed));
  }

  std::vector<ApplyDescriptionArtifactCommand> commands() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return commands_;
  }

 private:
  mutable std::mutex mutex_;
  SceneSnapshot snapshot_;
  std::vector<ApplyDescriptionArtifactCommand> commands_;
};

class DurabilityGate {
 public:
  explicit DurabilityGate(SceneRevision durable_revision)
      : durable_revision_(durable_revision) {}

  bool wait(SceneRevision revision, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    requested_revisions_.push_back(revision);
    return cv_.wait_for(lock, timeout, [this, revision]() {
      return durable_revision_ >= revision;
    });
  }

  void advance(SceneRevision revision) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      durable_revision_ = std::max(durable_revision_, revision);
    }
    cv_.notify_all();
  }

  bool requested(SceneRevision revision) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::find(requested_revisions_.begin(), requested_revisions_.end(),
                     revision) != requested_revisions_.end();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  SceneRevision durable_revision_ = 0;
  std::vector<SceneRevision> requested_revisions_;
};

ArtifactSchedulerConfig schedulerConfig() {
  ArtifactSchedulerConfig config;
  config.lease_duration_ms = 500;
  config.retry_initial_backoff_ms = 5;
  config.retry_max_backoff_ms = 5;
  return config;
}

ArtifactRuntimeActorConfig actorConfig(std::string owner) {
  ArtifactRuntimeActorConfig config;
  config.lease_owner = std::move(owner);
  config.idle_poll_period = 1ms;
  config.heartbeat_period = 10ms;
  config.reducer_submit_timeout = 50ms;
  config.durability_wait_slice = 5ms;
  config.worker_stop_timeout = 25ms;
  return config;
}

TEST(ArtifactRuntimeActor,
     SuccessfulWorkerAppliesAndCompletesOnlyItsFencedLease) {
  TemporaryRuntimeDatabase database;
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  const PreparedRuntimeScene scene = prepareRuntimeScene(&store);
  ArtifactScheduler scheduler(&store, schedulerConfig());
  const DamTaskIntent intent = runtimeIntent(scene, unixMillis());
  ASSERT_TRUE(scheduler.ensureTask(intent));

  auto worker = std::make_shared<SequenceDamWorker>(
      std::vector<DamWorkerResponse>{successfulResponse("a red chair")});
  FakeReducerSink reducer(scene.snapshot);
  DurabilityGate durability(scene.snapshot.revision() + 1);
  ArtifactRuntimeActor actor(
      actorConfig("runtime-success"), &scheduler, worker,
      [&reducer]() { return reducer.snapshot(); },
      [&reducer](const ApplyDescriptionArtifactCommand& command,
                 std::chrono::milliseconds timeout) {
        return reducer.submit(command, timeout);
      },
      [&durability](SceneRevision revision,
                    std::chrono::milliseconds timeout) {
        return durability.wait(revision, timeout);
      });

  actor.start();
  ASSERT_TRUE(waitFor([&actor]() { return actor.stats().completed == 1; }, 1s));
  actor.stop();

  const TaskLookupResult task =
      store.lookupTask(canonicalDamTaskKey(intent.key));
  ASSERT_TRUE(task.status) << task.status.error;
  ASSERT_TRUE(task.task.has_value());
  EXPECT_EQ(task.task->state, DurableTaskState::kCompleted);
  EXPECT_EQ(task.task->attempts, 1u);
  ASSERT_EQ(reducer.commands().size(), 1u);
  EXPECT_EQ(reducer.commands().front().description, "a red chair");
  EXPECT_EQ(worker->calls(), 1u);
  EXPECT_EQ(actor.stats().retried, 0u);
}

TEST(ArtifactRuntimeActor,
     CompletionAfterDueIsCountedEvenWhenDamStartedBeforeDue) {
  TemporaryRuntimeDatabase database;
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  const PreparedRuntimeScene scene = prepareRuntimeScene(&store);
  ArtifactSchedulerConfig scheduler_config = schedulerConfig();
  scheduler_config.lease_duration_ms = 30'000;
  ArtifactScheduler scheduler(&store, scheduler_config);
  const DamTaskIntent intent = runtimeIntent(scene, 1'000);
  ASSERT_TRUE(scheduler.ensureTask(intent));

  auto worker = std::make_shared<SequenceDamWorker>(
      std::vector<DamWorkerResponse>{successfulResponse("late red chair")});
  FakeReducerSink reducer(scene.snapshot);
  std::atomic<std::int64_t> logical_time{1'001};
  ArtifactRuntimeActor actor(
      actorConfig("runtime-slo"), &scheduler, worker,
      [&reducer]() { return reducer.snapshot(); },
      [&reducer](const ApplyDescriptionArtifactCommand& command,
                 std::chrono::milliseconds timeout) {
        return reducer.submit(command, timeout);
      },
      [&logical_time](SceneRevision, std::chrono::milliseconds) {
        logical_time.store(11'001);
        return true;
      },
      ArtifactRuntimeActor::TerminalFailureSink{},
      [&logical_time]() { return logical_time.load(); });

  actor.start();
  ASSERT_TRUE(waitFor([&actor]() { return actor.stats().completed == 1; }, 1s));
  actor.stop();

  const ArtifactRuntimeActorStats stats = actor.stats();
  EXPECT_EQ(stats.interactive_slo_completions, 1U);
  EXPECT_EQ(stats.interactive_slo_violations, 1U);
  EXPECT_EQ(stats.bulk_slo_violations, 0U);
  EXPECT_EQ(stats.slo_violations, 1U);
  ASSERT_EQ(reducer.commands().size(), 1U);
  EXPECT_EQ(reducer.commands().front().artifact_slo.origin_created_unix_ms,
            1'000);
  EXPECT_EQ(reducer.commands().front().artifact_slo.due_unix_ms, 11'000);
}

TEST(ArtifactRuntimeActor,
     PerceptionAdmissionGateDefersBeforeLeasingDurableTask) {
  TemporaryRuntimeDatabase database;
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  const PreparedRuntimeScene scene = prepareRuntimeScene(&store);
  ArtifactScheduler scheduler(&store, schedulerConfig());
  const DamTaskIntent intent = runtimeIntent(scene, unixMillis());
  ASSERT_TRUE(scheduler.ensureTask(intent));

  auto worker = std::make_shared<SequenceDamWorker>(
      std::vector<DamWorkerResponse>{successfulResponse("a quiet chair")});
  FakeReducerSink reducer(scene.snapshot);
  DurabilityGate durability(scene.snapshot.revision() + 1);
  std::atomic_bool perception_idle{false};
  ArtifactRuntimeActor actor(
      actorConfig("runtime-admission"), &scheduler, worker,
      [&reducer]() { return reducer.snapshot(); },
      [&reducer](const ApplyDescriptionArtifactCommand& command,
                 std::chrono::milliseconds timeout) {
        return reducer.submit(command, timeout);
      },
      [&durability](SceneRevision revision,
                    std::chrono::milliseconds timeout) {
        return durability.wait(revision, timeout);
      },
      ArtifactRuntimeActor::TerminalFailureSink{},
      ArtifactRuntimeActor::UnixMillisClock{},
      [&perception_idle]() { return perception_idle.load(); });

  actor.start();
  ASSERT_TRUE(waitFor(
      [&actor]() { return actor.stats().admission_deferrals > 2; }, 1s));
  EXPECT_EQ(worker->calls(), 0U);
  EXPECT_EQ(actor.stats().leases_acquired, 0U);
  const TaskLookupResult pending =
      store.lookupTask(canonicalDamTaskKey(intent.key));
  ASSERT_TRUE(pending.status) << pending.status.error;
  ASSERT_TRUE(pending.task.has_value());
  EXPECT_EQ(pending.task->state, DurableTaskState::kPending);

  perception_idle.store(true);
  ASSERT_TRUE(waitFor([&actor]() { return actor.stats().completed == 1; }, 1s));
  actor.stop();
  EXPECT_EQ(worker->calls(), 1U);
}

TEST(ArtifactRuntimeActor, RetryableWorkerFailureUsesDurableBackoffThenRecovers) {
  TemporaryRuntimeDatabase database;
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  const PreparedRuntimeScene scene = prepareRuntimeScene(&store);
  ArtifactScheduler scheduler(&store, schedulerConfig());
  const DamTaskIntent intent = runtimeIntent(scene, unixMillis());
  ASSERT_TRUE(scheduler.ensureTask(intent));

  DamWorkerResponse temporary_failure;
  temporary_failure.success = false;
  temporary_failure.retryable = true;
  temporary_failure.error = "temporary accelerator failure";
  auto worker = std::make_shared<SequenceDamWorker>(
      std::vector<DamWorkerResponse>{temporary_failure,
                                     successfulResponse("recovered chair")});
  FakeReducerSink reducer(scene.snapshot);
  DurabilityGate durability(scene.snapshot.revision() + 1);
  ArtifactRuntimeActor actor(
      actorConfig("runtime-retry"), &scheduler, worker,
      [&reducer]() { return reducer.snapshot(); },
      [&reducer](const ApplyDescriptionArtifactCommand& command,
                 std::chrono::milliseconds timeout) {
        return reducer.submit(command, timeout);
      },
      [&durability](SceneRevision revision,
                    std::chrono::milliseconds timeout) {
        return durability.wait(revision, timeout);
      });

  actor.start();
  ASSERT_TRUE(waitFor([&actor]() { return actor.stats().completed == 1; }, 2s));
  actor.stop();

  const TaskLookupResult task =
      store.lookupTask(canonicalDamTaskKey(intent.key));
  ASSERT_TRUE(task.status) << task.status.error;
  ASSERT_TRUE(task.task.has_value());
  EXPECT_EQ(task.task->state, DurableTaskState::kCompleted);
  EXPECT_EQ(task.task->attempts, 2u);
  EXPECT_EQ(worker->calls(), 2u);
  EXPECT_EQ(actor.stats().retried, 1u);
  EXPECT_EQ(reducer.commands().size(), 1u);
}

TEST(ArtifactRuntimeActor, PermanentWorkerFailureBecomesDurableFailedState) {
  TemporaryRuntimeDatabase database;
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  const PreparedRuntimeScene scene = prepareRuntimeScene(&store);
  ArtifactScheduler scheduler(&store, schedulerConfig());
  const DamTaskIntent intent = runtimeIntent(scene, unixMillis());
  ASSERT_TRUE(scheduler.ensureTask(intent));

  DamWorkerResponse permanent_failure;
  permanent_failure.success = false;
  permanent_failure.retryable = false;
  permanent_failure.error = "unsupported DAM model schema";
  auto worker = std::make_shared<SequenceDamWorker>(
      std::vector<DamWorkerResponse>{permanent_failure});
  FakeReducerSink reducer(scene.snapshot);
  DurabilityGate durability(scene.snapshot.durableRevision());
  ArtifactRuntimeActor actor(
      actorConfig("runtime-terminal"), &scheduler, worker,
      [&reducer]() { return reducer.snapshot(); },
      [&reducer](const ApplyDescriptionArtifactCommand& command,
                 std::chrono::milliseconds timeout) {
        return reducer.submit(command, timeout);
      },
      [&durability](SceneRevision revision,
                    std::chrono::milliseconds timeout) {
        return durability.wait(revision, timeout);
      });

  actor.start();
  ASSERT_TRUE(waitFor(
      [&actor]() { return actor.stats().terminal_failures == 1; }, 1s));
  actor.stop();

  const TaskLookupResult task =
      store.lookupTask(canonicalDamTaskKey(intent.key));
  ASSERT_TRUE(task.status) << task.status.error;
  ASSERT_TRUE(task.task.has_value());
  EXPECT_EQ(task.task->state, DurableTaskState::kFailed);
  EXPECT_EQ(task.task->last_error, "unsupported DAM model schema");
  EXPECT_EQ(task.task->failed_by, "runtime-terminal");
  EXPECT_GT(task.task->failed_at_unix_ms, 0);
  EXPECT_EQ(task.task->completed_at_unix_ms, 0);
  EXPECT_TRUE(task.task->completed_by.empty());
  EXPECT_EQ(actor.stats().retried, 0u);
  EXPECT_EQ(actor.stats().completed, 0u);
  EXPECT_TRUE(reducer.commands().empty());
}

TEST(ArtifactRuntimeActor, RestartReclaimsAnExpiredUnfinishedLease) {
  TemporaryRuntimeDatabase database;
  PreparedRuntimeScene scene;
  DamTaskIntent intent;
  const std::int64_t now = unixMillis();
  {
    SceneStore first_store(database.path());
    ASSERT_TRUE(first_store.open());
    scene = prepareRuntimeScene(&first_store);
    ArtifactScheduler first_scheduler(&first_store, schedulerConfig());
    intent = runtimeIntent(scene, now - 2000);
    ASSERT_TRUE(first_scheduler.ensureTask(intent));
    const ArtifactLeaseResult abandoned =
        first_scheduler.leaseNext("crashed-runtime", now - 1000);
    ASSERT_TRUE(abandoned.status) << abandoned.status.error;
    ASSERT_TRUE(abandoned.lease.has_value());
    EXPECT_EQ(abandoned.lease->attempt(), 1u);
  }

  SceneStore recovered_store(database.path());
  ASSERT_TRUE(recovered_store.open());
  ArtifactScheduler recovered_scheduler(&recovered_store, schedulerConfig());
  auto worker = std::make_shared<SequenceDamWorker>(
      std::vector<DamWorkerResponse>{successfulResponse("restart recovery")});
  FakeReducerSink reducer(scene.snapshot);
  DurabilityGate durability(scene.snapshot.revision() + 1);
  ArtifactRuntimeActor actor(
      actorConfig("recovered-runtime"), &recovered_scheduler, worker,
      [&reducer]() { return reducer.snapshot(); },
      [&reducer](const ApplyDescriptionArtifactCommand& command,
                 std::chrono::milliseconds timeout) {
        return reducer.submit(command, timeout);
      },
      [&durability](SceneRevision revision,
                    std::chrono::milliseconds timeout) {
        return durability.wait(revision, timeout);
      });

  actor.start();
  ASSERT_TRUE(waitFor([&actor]() { return actor.stats().completed == 1; }, 1s));
  actor.stop();

  const TaskLookupResult task =
      recovered_store.lookupTask(canonicalDamTaskKey(intent.key));
  ASSERT_TRUE(task.status) << task.status.error;
  ASSERT_TRUE(task.task.has_value());
  EXPECT_EQ(task.task->state, DurableTaskState::kCompleted);
  EXPECT_EQ(task.task->attempts, 2u);
  EXPECT_EQ(task.task->completed_by, "recovered-runtime");
}

TEST(ArtifactRuntimeActor, DoesNotCompleteBeforeReducerRevisionIsDurable) {
  TemporaryRuntimeDatabase database;
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  const PreparedRuntimeScene scene = prepareRuntimeScene(&store);
  ArtifactScheduler scheduler(&store, schedulerConfig());
  const DamTaskIntent intent = runtimeIntent(scene, unixMillis());
  ASSERT_TRUE(scheduler.ensureTask(intent));

  auto worker = std::make_shared<SequenceDamWorker>(
      std::vector<DamWorkerResponse>{successfulResponse("durability gated")});
  FakeReducerSink reducer(scene.snapshot);
  DurabilityGate durability(scene.snapshot.durableRevision());
  ArtifactRuntimeActor actor(
      actorConfig("runtime-durability"), &scheduler, worker,
      [&reducer]() { return reducer.snapshot(); },
      [&reducer](const ApplyDescriptionArtifactCommand& command,
                 std::chrono::milliseconds timeout) {
        return reducer.submit(command, timeout);
      },
      [&durability](SceneRevision revision,
                    std::chrono::milliseconds timeout) {
        return durability.wait(revision, timeout);
      });

  actor.start();
  ASSERT_TRUE(waitFor(
      [&actor]() {
        return actor.stats().reducer_submissions == 1 &&
               actor.stats().durability_waits == 1;
      },
      1s));
  const SceneRevision artifact_revision = scene.snapshot.revision() + 1;
  ASSERT_TRUE(durability.requested(artifact_revision));
  std::this_thread::sleep_for(20ms);

  TaskLookupResult before_durable =
      store.lookupTask(canonicalDamTaskKey(intent.key));
  ASSERT_TRUE(before_durable.status) << before_durable.status.error;
  ASSERT_TRUE(before_durable.task.has_value());
  EXPECT_EQ(before_durable.task->state, DurableTaskState::kLeased);
  EXPECT_EQ(actor.stats().completed, 0u);

  durability.advance(artifact_revision);
  ASSERT_TRUE(waitFor([&actor]() { return actor.stats().completed == 1; }, 1s));
  actor.stop();

  const TaskLookupResult after_durable =
      store.lookupTask(canonicalDamTaskKey(intent.key));
  ASSERT_TRUE(after_durable.status) << after_durable.status.error;
  ASSERT_TRUE(after_durable.task.has_value());
  EXPECT_EQ(after_durable.task->state, DurableTaskState::kCompleted);
}

TEST(ArtifactRuntimeActor, StopIsBoundedAndRequeuesAnInFlightLease) {
  TemporaryRuntimeDatabase database;
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  const PreparedRuntimeScene scene = prepareRuntimeScene(&store);
  ArtifactScheduler scheduler(&store, schedulerConfig());
  const DamTaskIntent intent = runtimeIntent(scene, unixMillis());
  ASSERT_TRUE(scheduler.ensureTask(intent));

  auto worker = std::make_shared<BlockingDamWorker>();
  FakeReducerSink reducer(scene.snapshot);
  DurabilityGate durability(scene.snapshot.durableRevision());
  ArtifactRuntimeActor actor(
      actorConfig("runtime-stop"), &scheduler, worker,
      [&reducer]() { return reducer.snapshot(); },
      [&reducer](const ApplyDescriptionArtifactCommand& command,
                 std::chrono::milliseconds timeout) {
        return reducer.submit(command, timeout);
      },
      [&durability](SceneRevision revision,
                    std::chrono::milliseconds timeout) {
        return durability.wait(revision, timeout);
      });

  actor.start();
  ASSERT_TRUE(worker->waitUntilStarted(1s));
  const auto stop_started = std::chrono::steady_clock::now();
  actor.stop();
  const auto stop_elapsed = std::chrono::steady_clock::now() - stop_started;
  EXPECT_LT(stop_elapsed, 500ms);

  const TaskLookupResult task =
      store.lookupTask(canonicalDamTaskKey(intent.key));
  ASSERT_TRUE(task.status) << task.status.error;
  ASSERT_TRUE(task.task.has_value());
  EXPECT_EQ(task.task->state, DurableTaskState::kPending);
  EXPECT_EQ(actor.stats().stop_requeues, 1u);
  EXPECT_EQ(actor.stats().detached_worker_invocations, 1u);

  worker->release();
  EXPECT_TRUE(worker->waitUntilFinished(1s));
  EXPECT_TRUE(reducer.commands().empty())
      << "a detached late worker result must never reach the reducer";
}

}  // namespace
}  // namespace roomie
