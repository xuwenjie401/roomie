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

#include "roomie/artifacts/artifact_slo_clock.hpp"
#include "roomie/artifacts/artifact_intent_builder.hpp"
#include "roomie/artifacts/embedding_runtime_actor.hpp"
#include "roomie/artifacts/semantic_index_store_adapter.hpp"
#include "roomie/scene/scene_reducer.hpp"

namespace roomie {
namespace {

using namespace std::chrono_literals;
using Json = nlohmann::json;
using Milliseconds = std::chrono::milliseconds;

class TemporaryEmbeddingDatabase {
 public:
  TemporaryEmbeddingDatabase() {
    char pattern[] = "/tmp/roomie_embedding_runtime_XXXXXX";
    const int descriptor = mkstemp(pattern);
    if (descriptor < 0) {
      throw std::runtime_error("mkstemp failed");
    }
    close(descriptor);
    path_ = pattern;
  }

  ~TemporaryEmbeddingDatabase() {
    std::remove(path_.c_str());
    std::remove((path_ + "-wal").c_str());
    std::remove((path_ + "-shm").c_str());
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

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

ObjectNode embeddingNode(SceneObjectId object_id, std::string label) {
  ObjectNode node;
  node.object_id = object_id;
  node.semantic_id = object_id + 100;
  node.label = std::move(label);
  node.description = "durable description for " + node.label;
  node.center_world = Eigen::Vector3f(static_cast<float>(object_id), 0.0f,
                                     0.5f);
  node.size_m = Eigen::Vector3f(0.5f, 0.5f, 1.0f);
  node.confidence = 0.8f;
  node.active = true;
  node.publishable = true;
  return node;
}

struct LoadedEmbeddingScene {
  std::unique_ptr<ReducerCore> reducer;
  SceneApplyResult loaded;
};

LoadedEmbeddingScene loadEmbeddingScene(std::size_t object_count) {
  auto reducer = std::make_unique<ReducerCore>();
  LoadSceneCommand command;
  for (std::size_t index = 0; index < object_count; ++index) {
    const SceneObjectId object_id = static_cast<SceneObjectId>(index + 1);
    command.graph.objects.push_back(
        embeddingNode(object_id, "object " + std::to_string(object_id)));
  }
  command.graph.next_object_id =
      static_cast<SceneObjectId>(object_count + 1);
  SceneApplyResult loaded = reducer->apply(SceneCommand{command});
  if (!loaded.committedRevision()) {
    throw std::runtime_error(loaded.reason);
  }
  return {std::move(reducer), std::move(loaded)};
}

DurableTaskSpec embeddingTask(const SceneSnapshot& snapshot,
                              SceneObjectId object_id,
                              EmbeddingNamespace name_space = {"model-a", 3},
                              ArtifactSloContext artifact_slo = {}) {
  const SceneObjectPtr object = snapshot.findExactObject(object_id);
  if (!object || !object->identity || !object->semantic || !object->artifact) {
    throw std::runtime_error("embedding test object is incomplete");
  }
  const SemanticDocument document = makeSemanticDocumentForObject(
      *object, object_id, snapshot.revision());
  DurableTaskSpec task;
  task.task_id =
      canonicalEmbeddingTaskKey(object_id, document.document_hash, name_space);
  task.dedupe_key = task.task_id;
  task.task_type = EmbeddingTaskScheduler::kTaskType;
  task.scene_revision = snapshot.revision();
  task.not_before_unix_ms = 0;
  Json payload{{"payload_version", "roomie.embedding-input.v1"},
               {"owning_scene_revision", snapshot.revision()},
               {"object_id", object_id},
               {"document", document.text},
               {"document_hash", document.document_hash},
               {"model_id", name_space.model_id},
               {"dimension", name_space.dimension},
               {"semantic_revision", document.semantic_revision},
               {"created_scene_revision", snapshot.revision()},
               {"identity_revision", object->identity->revision},
               {"artifact_revision", object->artifact->revision},
               {"description_input_hash",
                object->artifact->description_input_hash}};
  if (artifact_slo.tracked()) {
    Json slo{{"origin_created_unix_ms",
              artifact_slo.origin_created_unix_ms},
             {"due_unix_ms", artifact_slo.due_unix_ms},
             {"priority",
              artifact_slo.priority == ArtifactPriority::kInteractive
                  ? "interactive"
                  : "bulk"}};
    if (artifact_slo.monotonicTracked()) {
      slo["steady_clock_epoch"] =
          Json{{"high", artifact_slo.steady_clock_epoch.high},
               {"low", artifact_slo.steady_clock_epoch.low}};
      slo["origin_steady_ns"] = artifact_slo.origin_steady_ns;
      slo["due_steady_ns"] = artifact_slo.due_steady_ns;
    }
    payload["artifact_slo"] = std::move(slo);
  }
  task.payload = payload.dump();
  return task;
}

EmbeddingTaskSchedulerConfig schedulerConfig(
    std::int64_t lease_duration_ms = 300) {
  EmbeddingTaskSchedulerConfig config;
  config.lease_duration_ms = lease_duration_ms;
  config.retry_initial_backoff_ms = 5;
  config.retry_max_backoff_ms = 20;
  return config;
}

EmbeddingRuntimeActorConfig actorConfig() {
  EmbeddingRuntimeActorConfig config;
  config.lease_owner = "embedding-test-runtime";
  config.maximum_batch_size = 8;
  config.collection_window = 20ms;
  config.idle_poll_period = 2ms;
  config.heartbeat_period = 10ms;
  config.invocation_wait_slice = 2ms;
  config.encoder_stop_timeout = 50ms;
  return config;
}

SemanticIndexConfig semanticConfig() {
  SemanticIndexConfig config;
  config.initial_namespace = {"model-a", 3};
  config.default_read_ttl_ms = 1000;
  return config;
}

bool taskCompleted(SceneStore* store, const std::string& task_id) {
  const TaskLookupResult lookup = store->lookupTask(task_id);
  return lookup.status && lookup.task &&
         lookup.task->state == DurableTaskState::kCompleted;
}

class ControlledEncoder final : public EmbeddingEncoder {
 public:
  explicit ControlledEncoder(std::chrono::milliseconds delay = 0ms,
                             int failures = 0,
                             bool blocking = false)
      : delay_(delay), failures_remaining_(failures), blocking_(blocking) {}

  std::string modelId() const override { return "model-a"; }
  std::size_t dimension() const override { return 3; }

  void prewarm() override { ++prewarm_calls_; }

  std::vector<std::vector<float>> encodeBatch(
      const std::vector<std::string>& documents) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      batch_sizes_.push_back(documents.size());
      started_ = true;
    }
    cv_.notify_all();
    if (blocking_) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return released_; });
      finished_ = true;
      lock.unlock();
      cv_.notify_all();
    }
    if (delay_ > 0ms) {
      std::this_thread::sleep_for(delay_);
    }
    int remaining = failures_remaining_.load();
    while (remaining > 0 &&
           !failures_remaining_.compare_exchange_weak(remaining,
                                                       remaining - 1)) {
    }
    if (remaining > 0) {
      throw std::runtime_error("transient fake encoder failure");
    }
    std::vector<std::vector<float>> result;
    result.reserve(documents.size());
    for (const std::string& document : documents) {
      (void)document;
      result.push_back({3.0f, 4.0f, 0.0f});
    }
    return result;
  }

  int prewarmCalls() const { return prewarm_calls_.load(); }

  std::vector<std::size_t> batchSizes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return batch_sizes_;
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
  std::chrono::milliseconds delay_{0};
  std::atomic_int failures_remaining_{0};
  bool blocking_ = false;
  std::atomic_int prewarm_calls_{0};
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<std::size_t> batch_sizes_;
  bool started_ = false;
  bool released_ = false;
  bool finished_ = false;
};

TEST(EmbeddingTaskScheduler,
     StrictPayloadReturnsPoisonLeaseForTerminalDisposition) {
  TemporaryEmbeddingDatabase database;
  LoadedEmbeddingScene scene = loadEmbeddingScene(1);
  DurableTaskSpec malformed = embeddingTask(scene.loaded.snapshot, 1);
  Json payload = Json::parse(malformed.payload);
  payload["unexpected_field"] = true;
  malformed.payload = payload.dump();

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(scene.loaded.snapshot, {malformed}));
  ASSERT_TRUE(store.flush());
  EmbeddingTaskScheduler scheduler(&store, schedulerConfig());
  const EmbeddingTaskLeaseResult leased =
      scheduler.leaseNext("strict-parser", 100);
  ASSERT_TRUE(leased.status) << leased.status.error;
  ASSERT_TRUE(leased.lease);
  EXPECT_FALSE(leased.lease->valid());
  EXPECT_NE(leased.lease->validation_error.find("unknown field"),
            std::string::npos);
  const EmbeddingTaskRetryResult requeued =
      scheduler.retry(*leased.lease, 100, "test poison task cleanup");
  EXPECT_TRUE(requeued.status) << requeued.status.error;
}

TEST(EmbeddingTaskScheduler, InteractiveLeaseOvertakesBulkBacklog) {
  TemporaryEmbeddingDatabase database;
  LoadedEmbeddingScene scene = loadEmbeddingScene(3);
  ArtifactSloContext bulk;
  bulk.origin_created_unix_ms = 100;
  bulk.due_unix_ms = 1'000;
  bulk.priority = ArtifactPriority::kBulk;
  ArtifactSloContext interactive = bulk;
  interactive.priority = ArtifactPriority::kInteractive;

  const DurableTaskSpec first_bulk =
      embeddingTask(scene.loaded.snapshot, 1, {"model-a", 3}, bulk);
  const DurableTaskSpec second_bulk =
      embeddingTask(scene.loaded.snapshot, 2, {"model-a", 3}, bulk);
  const DurableTaskSpec later_interactive = embeddingTask(
      scene.loaded.snapshot, 3, {"model-a", 3}, interactive);

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(
      scene.loaded.snapshot,
      {first_bulk, second_bulk, later_interactive}));
  ASSERT_TRUE(store.flush());

  EmbeddingTaskScheduler scheduler(&store, schedulerConfig());
  const EmbeddingTaskLeaseResult leased =
      scheduler.leaseNext("priority-worker", 100);
  ASSERT_TRUE(leased.status) << leased.status.error;
  ASSERT_TRUE(leased.lease);
  ASSERT_TRUE(leased.lease->request) << leased.lease->validation_error;
  EXPECT_EQ(leased.lease->request->object_id, 3);
  EXPECT_EQ(leased.lease->request->artifact_slo.priority,
            ArtifactPriority::kInteractive);
}

TEST(EmbeddingRuntimeActor,
     BatchesHeartbeatsAndPersistsBeforeFencedCompletion) {
  TemporaryEmbeddingDatabase database;
  LoadedEmbeddingScene scene = loadEmbeddingScene(2);
  const DurableTaskSpec first = embeddingTask(scene.loaded.snapshot, 1);
  const DurableTaskSpec second = embeddingTask(scene.loaded.snapshot, 2);

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(scene.loaded.snapshot, {first, second}));
  ASSERT_TRUE(store.flush());
  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);

  EmbeddingTaskScheduler scheduler(&store, schedulerConfig(80));
  VersionedSemanticIndex index(semanticConfig());
  auto encoder = std::make_shared<ControlledEncoder>(140ms);
  std::atomic_bool persisted_while_leased{true};
  EmbeddingRuntimeActor actor(
      actorConfig(), &scheduler, encoder, &index,
      [snapshot = restored.snapshot]() { return snapshot; },
      [&](const EmbeddingRecord& record) {
        const SceneStoreStatus status =
            persistSemanticEmbeddingRecord(&store, record);
        const std::string task_id = canonicalEmbeddingTaskKey(
            record.object_id, record.document_hash,
            EmbeddingNamespace{record.model_id, record.vector.size()});
        const TaskLookupResult lookup = store.lookupTask(task_id);
        if (!status || !lookup.status || !lookup.task ||
            lookup.task->state != DurableTaskState::kLeased) {
          persisted_while_leased = false;
        }
        return status;
      });
  actor.start();
  ASSERT_TRUE(actor.waitUntilWarmed(1s));
  ASSERT_TRUE(waitFor(
      [&]() {
        return taskCompleted(&store, first.task_id) &&
               taskCompleted(&store, second.task_id);
      },
      3s));
  actor.stop();
  ASSERT_TRUE(index.waitForBuildIdle(1s));

  EXPECT_TRUE(persisted_while_leased.load());
  EXPECT_EQ(encoder->prewarmCalls(), 1);
  const std::vector<std::size_t> batches = encoder->batchSizes();
  ASSERT_EQ(batches.size(), 1u);
  EXPECT_EQ(batches.front(), 2u);
  const EmbeddingRuntimeActorStats stats = actor.stats();
  EXPECT_EQ(stats.encoded_documents, 2u);
  EXPECT_EQ(stats.persisted_records, 2u);
  EXPECT_EQ(stats.accepted_records, 2u);
  EXPECT_EQ(stats.completed, 2u);
  EXPECT_EQ(stats.lease_lost, 0u)
      << "the 140 ms batch crossed the original 80 ms lease";
  EXPECT_EQ(index.activeGeneration().rows, 2u);
}

TEST(EmbeddingRuntimeActor,
     QueryableAcceptAccountsInteractiveAndBulkCompletionDeadlines) {
  TemporaryEmbeddingDatabase database;
  LoadedEmbeddingScene scene = loadEmbeddingScene(2);
  ArtifactSloContext interactive;
  interactive.origin_created_unix_ms = 100;
  interactive.due_unix_ms = 150;
  interactive.priority = ArtifactPriority::kInteractive;
  ArtifactSloContext bulk = interactive;
  bulk.priority = ArtifactPriority::kBulk;
  const DurableTaskSpec first = embeddingTask(
      scene.loaded.snapshot, 1, {"model-a", 3}, interactive);
  const DurableTaskSpec second = embeddingTask(
      scene.loaded.snapshot, 2, {"model-a", 3}, bulk);

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(scene.loaded.snapshot, {first, second}));
  ASSERT_TRUE(store.flush());
  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;

  EmbeddingTaskScheduler scheduler(&store, schedulerConfig(500));
  VersionedSemanticIndex index(semanticConfig());
  auto encoder = std::make_shared<ControlledEncoder>(80ms);
  const auto clock_origin = std::chrono::steady_clock::now();
  EmbeddingRuntimeActorConfig runtime_config = actorConfig();
  runtime_config.slo_accounting_history_capacity = 1;
  EmbeddingRuntimeActor actor(
      runtime_config, &scheduler, encoder, &index,
      [snapshot = restored.snapshot]() { return snapshot; },
      [&](const EmbeddingRecord& record) {
        return persistSemanticEmbeddingRecord(&store, record);
      },
      {},
      [clock_origin]() {
        return std::int64_t{100} +
               std::chrono::duration_cast<Milliseconds>(
                   std::chrono::steady_clock::now() - clock_origin)
                   .count();
      });
  actor.start();
  ASSERT_TRUE(waitFor(
      [&]() {
        return taskCompleted(&store, first.task_id) &&
               taskCompleted(&store, second.task_id);
      },
      2s));
  actor.stop();

  const EmbeddingRuntimeActorStats stats = actor.stats();
  EXPECT_EQ(stats.interactive_slo_completions, 1U);
  EXPECT_EQ(stats.bulk_slo_completions, 1U);
  EXPECT_EQ(stats.interactive_slo_violations, 1U);
  EXPECT_EQ(stats.bulk_slo_violations, 1U);
  EXPECT_EQ(stats.slo_accounting_history_size, 1U);
  EXPECT_EQ(stats.slo_violations, 2U);
  EXPECT_EQ(index.activeGeneration().rows, 2U)
      << "durable completion/SLO accounting must wait for a queryable generation";
}

TEST(EmbeddingRuntimeActor,
     CurrentProcessSteadyDeadlineIgnoresWallClockJump) {
  TemporaryEmbeddingDatabase database;
  LoadedEmbeddingScene scene = loadEmbeddingScene(1);
  ArtifactSloContext slo;
  slo.origin_created_unix_ms = 100;
  slo.due_unix_ms = 150;
  slo.priority = ArtifactPriority::kInteractive;
  slo.steady_clock_epoch = artifactSteadyClockEpoch();
  slo.origin_steady_ns = artifactSteadyNowNanoseconds();
  slo.due_steady_ns = slo.origin_steady_ns + 5'000'000'000LL;
  const DurableTaskSpec task =
      embeddingTask(scene.loaded.snapshot, 1, {"model-a", 3}, slo);

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(scene.loaded.snapshot, {task}));
  ASSERT_TRUE(store.flush());
  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;

  EmbeddingTaskScheduler scheduler(&store, schedulerConfig(500));
  VersionedSemanticIndex index(semanticConfig());
  auto encoder = std::make_shared<ControlledEncoder>();
  EmbeddingRuntimeActor actor(
      actorConfig(), &scheduler, encoder, &index,
      [snapshot = restored.snapshot]() { return snapshot; },
      [&](const EmbeddingRecord& record) {
        return persistSemanticEmbeddingRecord(&store, record);
      },
      {}, []() { return std::int64_t{10'000}; });
  actor.start();
  const bool completed = waitFor(
      [&]() { return taskCompleted(&store, task.task_id); }, 2s);
  const TaskLookupResult lookup = store.lookupTask(task.task_id);
  EXPECT_TRUE(completed)
      << "actor_error=" << actor.lastError()
      << " task_error="
      << (lookup.task ? lookup.task->last_error : lookup.status.error);
  actor.stop();

  const EmbeddingRuntimeActorStats stats = actor.stats();
  EXPECT_EQ(stats.interactive_slo_completions, 1U);
  EXPECT_EQ(stats.interactive_slo_violations, 0U)
      << "fake wall time is far past due, but the steady deadline is not";
  EXPECT_EQ(stats.slo_violations, 0U);
}

TEST(EmbeddingRuntimeActor,
     StaleMergedAndDeletedTasksCompleteWithoutEncoding) {
  TemporaryEmbeddingDatabase database;
  LoadedEmbeddingScene scene = loadEmbeddingScene(4);
  const DurableTaskSpec stale = embeddingTask(scene.loaded.snapshot, 1);
  const DurableTaskSpec merged = embeddingTask(scene.loaded.snapshot, 2);
  const DurableTaskSpec deleted = embeddingTask(scene.loaded.snapshot, 3);

  ApplyHumanAnnotationCommand annotation;
  annotation.object_id = 1;
  annotation.patch.label = "new semantic label";
  const SceneApplyResult annotated =
      scene.reducer->apply(SceneCommand{annotation});
  ASSERT_TRUE(annotated.committedRevision()) << annotated.reason;

  ApplyObservationBatchCommand merge_command;
  merge_command.association_source = "embedding-runtime-test";
  MergeObjectsMutation merge;
  merge.retired_object_id = 2;
  merge.canonical_object_id = 4;
  merge_command.associated_mutations.push_back(merge);
  const SceneApplyResult merged_scene =
      scene.reducer->apply(SceneCommand{merge_command});
  ASSERT_TRUE(merged_scene.committedRevision()) << merged_scene.reason;

  ApplyObservationBatchCommand delete_command;
  delete_command.association_source = "embedding-runtime-test";
  TombstoneObjectMutation tombstone;
  tombstone.object_id = 3;
  tombstone.reason = "deleted for embedding test";
  delete_command.associated_mutations.push_back(tombstone);
  const SceneApplyResult deleted_scene =
      scene.reducer->apply(SceneCommand{delete_command});
  ASSERT_TRUE(deleted_scene.committedRevision()) << deleted_scene.reason;

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(scene.loaded.snapshot,
                                  {stale, merged, deleted}));
  ASSERT_TRUE(store.enqueueCommit(annotated.snapshot));
  ASSERT_TRUE(store.enqueueCommit(merged_scene.snapshot));
  ASSERT_TRUE(store.enqueueCommit(deleted_scene.snapshot));
  ASSERT_TRUE(store.flush());
  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;

  EmbeddingTaskScheduler scheduler(&store, schedulerConfig());
  VersionedSemanticIndex index(semanticConfig());
  auto encoder = std::make_shared<ControlledEncoder>();
  EmbeddingRuntimeActor actor(
      actorConfig(), &scheduler, encoder, &index,
      [snapshot = restored.snapshot]() { return snapshot; },
      [&](const EmbeddingRecord& record) {
        return persistSemanticEmbeddingRecord(&store, record);
      });
  actor.start();
  ASSERT_TRUE(waitFor(
      [&]() {
        return taskCompleted(&store, stale.task_id) &&
               taskCompleted(&store, merged.task_id) &&
               taskCompleted(&store, deleted.task_id);
      },
      2s));
  actor.stop();

  EXPECT_TRUE(encoder->batchSizes().empty());
  EXPECT_EQ(actor.stats().superseded, 3u);
  EXPECT_EQ(actor.stats().completed, 3u);
  EXPECT_EQ(store.listEmbeddingRecords().records.size(), 0u);
}

TEST(EmbeddingRuntimeActor,
     SemanticRevisionAdvanceWithoutDocumentChangeRemainsCurrent) {
  TemporaryEmbeddingDatabase database;
  LoadedEmbeddingScene scene = loadEmbeddingScene(1);
  const DurableTaskSpec task = embeddingTask(scene.loaded.snapshot, 1);

  SceneState next = *scene.loaded.snapshot.statePtr();
  SceneObjectTable objects = scene.loaded.snapshot.objects();
  const SceneObjectPtr original = objects.at(1);
  auto updated = std::make_shared<SceneObject>(*original);
  auto semantic = std::make_shared<SemanticComponent>(*original->semantic);
  ++semantic->revision;
  semantic->confidence = 0.95f;
  updated->semantic = std::move(semantic);
  objects[1] = std::move(updated);
  next.objects =
      std::make_shared<const SceneObjectTable>(std::move(objects));
  next.latest_scene_revision = scene.loaded.revision + 1;
  const SceneSnapshot revision_only{
      std::make_shared<const SceneState>(std::move(next))};
  ASSERT_EQ(makeSemanticDocumentForObject(
                *scene.loaded.snapshot.findExactObject(1), 1, 1)
                .document_hash,
            makeSemanticDocumentForObject(
                *revision_only.findExactObject(1), 1, 2)
                .document_hash);

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(scene.loaded.snapshot, {task}));
  ASSERT_TRUE(store.enqueueCommit(revision_only));
  ASSERT_TRUE(store.flush());
  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;

  EmbeddingTaskScheduler scheduler(&store, schedulerConfig());
  VersionedSemanticIndex index(semanticConfig());
  auto encoder = std::make_shared<ControlledEncoder>();
  EmbeddingRuntimeActor actor(
      actorConfig(), &scheduler, encoder, &index,
      [snapshot = restored.snapshot]() { return snapshot; },
      [&](const EmbeddingRecord& record) {
        return persistSemanticEmbeddingRecord(&store, record);
      });
  actor.start();
  ASSERT_TRUE(waitFor([&]() { return taskCompleted(&store, task.task_id); },
                      2s));
  actor.stop();

  ASSERT_EQ(encoder->batchSizes().size(), 1u);
  EXPECT_EQ(actor.stats().superseded, 0u);
  EXPECT_EQ(actor.stats().accepted_records, 1u);
}

TEST(EmbeddingRuntimeActor, RetryBackoffEventuallyCompletes) {
  TemporaryEmbeddingDatabase database;
  LoadedEmbeddingScene scene = loadEmbeddingScene(1);
  const DurableTaskSpec task = embeddingTask(scene.loaded.snapshot, 1);
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(scene.loaded.snapshot, {task}));
  ASSERT_TRUE(store.flush());
  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;

  EmbeddingTaskScheduler scheduler(&store, schedulerConfig());
  VersionedSemanticIndex index(semanticConfig());
  auto encoder = std::make_shared<ControlledEncoder>(0ms, 1);
  EmbeddingRuntimeActor actor(
      actorConfig(), &scheduler, encoder, &index,
      [snapshot = restored.snapshot]() { return snapshot; },
      [&](const EmbeddingRecord& record) {
        return persistSemanticEmbeddingRecord(&store, record);
      });
  actor.start();
  ASSERT_TRUE(waitFor([&]() { return taskCompleted(&store, task.task_id); },
                      2s));
  actor.stop();

  const TaskLookupResult lookup = store.lookupTask(task.task_id);
  ASSERT_TRUE(lookup.status);
  ASSERT_TRUE(lookup.task);
  EXPECT_GE(lookup.task->attempts, 2u);
  EXPECT_EQ(encoder->batchSizes().size(), 2u);
  EXPECT_GE(actor.stats().retried, 1u);
  EXPECT_EQ(actor.stats().completed, 1u);
}

TEST(EmbeddingRuntimeActor,
     PersistedButUnackedLeaseRestartsWithAtLeastOnceDelivery) {
  TemporaryEmbeddingDatabase database;
  LoadedEmbeddingScene scene = loadEmbeddingScene(1);
  const DurableTaskSpec task = embeddingTask(scene.loaded.snapshot, 1);

  {
    SceneStore crashed(database.path());
    ASSERT_TRUE(crashed.open());
    ASSERT_TRUE(crashed.enqueueCommit(scene.loaded.snapshot, {task}));
    ASSERT_TRUE(crashed.flush());
    EmbeddingTaskScheduler first_scheduler(&crashed, schedulerConfig(50));
    const EmbeddingTaskLeaseResult leased =
        first_scheduler.leaseNext("crashed-worker", 1000);
    ASSERT_TRUE(leased.status) << leased.status.error;
    ASSERT_TRUE(leased.lease);
    ASSERT_TRUE(leased.lease->request);
    EmbeddingRecord record;
    record.object_id = leased.lease->request->object_id;
    record.document_hash = leased.lease->request->document_hash;
    record.model_id = leased.lease->request->name_space.model_id;
    record.vector = {0.6f, 0.8f, 0.0f};
    record.created_scene_revision =
        leased.lease->request->created_scene_revision;
    ASSERT_TRUE(persistSemanticEmbeddingRecord(&crashed, record));
    // Simulated crash: no acceptEmbedding() and no complete().
  }

  SceneStore reopened(database.path());
  ASSERT_TRUE(reopened.open());
  const SceneRestoreResult restored = reopened.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  EmbeddingTaskScheduler scheduler(&reopened, schedulerConfig(50));
  VersionedSemanticIndex index(semanticConfig());
  auto encoder = std::make_shared<ControlledEncoder>();
  EmbeddingRuntimeActor actor(
      actorConfig(), &scheduler, encoder, &index,
      [snapshot = restored.snapshot]() { return snapshot; },
      [&](const EmbeddingRecord& record) {
        return persistSemanticEmbeddingRecord(&reopened, record);
      },
      {}, []() { return 2000; });
  actor.start();
  ASSERT_TRUE(waitFor(
      [&]() { return taskCompleted(&reopened, task.task_id); }, 2s));
  actor.stop();

  const TaskLookupResult lookup = reopened.lookupTask(task.task_id);
  ASSERT_TRUE(lookup.status);
  ASSERT_TRUE(lookup.task);
  EXPECT_EQ(lookup.task->attempts, 2u);
  const EmbeddingRecordListResult records =
      reopened.listEmbeddingRecords();
  ASSERT_TRUE(records.status) << records.status.error;
  EXPECT_EQ(records.records.size(), 1u);
  EXPECT_EQ(actor.stats().accepted_records, 1u);
  EXPECT_EQ(actor.stats().completed, 1u);
}

TEST(EmbeddingRuntimeActor,
     ShutdownIsBoundedRequeuesLeaseAndNeverCompletesDetachedResult) {
  TemporaryEmbeddingDatabase database;
  LoadedEmbeddingScene scene = loadEmbeddingScene(1);
  const DurableTaskSpec task = embeddingTask(scene.loaded.snapshot, 1);
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(scene.loaded.snapshot, {task}));
  ASSERT_TRUE(store.flush());
  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;

  EmbeddingTaskScheduler scheduler(&store, schedulerConfig(500));
  VersionedSemanticIndex index(semanticConfig());
  auto encoder = std::make_shared<ControlledEncoder>(0ms, 0, true);
  EmbeddingRuntimeActorConfig config = actorConfig();
  config.encoder_stop_timeout = 40ms;
  EmbeddingRuntimeActor actor(
      config, &scheduler, encoder, &index,
      [snapshot = restored.snapshot]() { return snapshot; },
      [&](const EmbeddingRecord& record) {
        return persistSemanticEmbeddingRecord(&store, record);
      });
  actor.start();
  ASSERT_TRUE(encoder->waitUntilStarted(1s));
  const auto started = std::chrono::steady_clock::now();
  actor.stop();
  const auto elapsed = std::chrono::duration_cast<Milliseconds>(
      std::chrono::steady_clock::now() - started);
  EXPECT_LT(elapsed, 300ms);

  const TaskLookupResult lookup = store.lookupTask(task.task_id);
  ASSERT_TRUE(lookup.status);
  ASSERT_TRUE(lookup.task);
  EXPECT_EQ(lookup.task->state, DurableTaskState::kPending);
  EXPECT_EQ(actor.stats().completed, 0u);
  EXPECT_EQ(actor.stats().stop_requeues, 1u);
  EXPECT_EQ(actor.stats().detached_encoder_invocations, 1u);
  EXPECT_TRUE(store.listEmbeddingRecords().records.empty());

  encoder->release();
  ASSERT_TRUE(encoder->waitUntilFinished(1s));
  std::this_thread::sleep_for(10ms);
  const TaskLookupResult after_release = store.lookupTask(task.task_id);
  ASSERT_TRUE(after_release.status);
  ASSERT_TRUE(after_release.task);
  EXPECT_EQ(after_release.task->state, DurableTaskState::kPending);
  EXPECT_TRUE(store.listEmbeddingRecords().records.empty());
}

}  // namespace
}  // namespace roomie
