#include <gtest/gtest.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sqlite3.h>

#include "roomie/scene/persistence_actor.hpp"
#include "roomie/scene/scene_reducer.hpp"

namespace roomie {
namespace {

using namespace std::chrono_literals;

class TemporaryPersistenceDatabase {
 public:
  TemporaryPersistenceDatabase() {
    char pattern[] = "/tmp/roomie_persistence_actor_XXXXXX";
    const int descriptor = mkstemp(pattern);
    if (descriptor < 0) {
      throw std::runtime_error("mkstemp failed");
    }
    close(descriptor);
    path_ = pattern;
  }

  ~TemporaryPersistenceDatabase() {
    std::remove(path_.c_str());
    std::remove((path_ + "-wal").c_str());
    std::remove((path_ + "-shm").c_str());
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

ObjectNode persistenceNode() {
  ObjectNode node;
  node.object_id = 1;
  node.semantic_id = 101;
  node.label = "chair";
  node.center_world = Eigen::Vector3f(1.0f, 2.0f, 0.5f);
  node.size_m = Eigen::Vector3f(0.5f, 0.6f, 0.9f);
  node.active = true;
  node.publishable = true;
  return node;
}

std::vector<SceneSnapshot> makeSnapshots(std::size_t count) {
  std::vector<SceneSnapshot> snapshots;
  if (count == 0) {
    return snapshots;
  }

  ReducerCore reducer;
  LoadSceneCommand load;
  load.graph.objects.push_back(persistenceNode());
  load.graph.next_object_id = 2;
  SceneApplyResult result = reducer.apply(SceneCommand{load});
  if (!result.committedRevision()) {
    throw std::runtime_error(result.reason);
  }
  snapshots.push_back(result.snapshot);

  while (snapshots.size() < count) {
    ApplyHumanAnnotationCommand annotation;
    annotation.object_id = 1;
    annotation.patch.attributes["revision"] =
        std::to_string(snapshots.size() + 1);
    result = reducer.apply(SceneCommand{annotation});
    if (!result.committedRevision()) {
      throw std::runtime_error(result.reason);
    }
    snapshots.push_back(result.snapshot);
  }
  return snapshots;
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

PersistenceActorConfig slowConfig() {
  PersistenceActorConfig config;
  config.flush_period = std::chrono::hours(1);
  config.flush_batch_size = 64;
  config.queue_capacity = 8;
  config.max_undurable_revisions = 64;
  config.hard_max_undurable_revisions = 64;
  return config;
}

TEST(PersistenceActor, ExposesSoftAndHardUndurableLagForAdmission) {
  TemporaryPersistenceDatabase database;
  PersistenceActorConfig config = slowConfig();
  config.queue_capacity = 4;
  config.max_undurable_revisions = 2;
  config.hard_max_undurable_revisions = 3;
  PersistenceActor actor(database.path(), config);
  ASSERT_TRUE(actor.open());

  const std::vector<SceneSnapshot> snapshots = makeSnapshots(3);
  ASSERT_TRUE(actor.enqueueCommit(snapshots[0]));
  ASSERT_TRUE(actor.enqueueCommit(snapshots[1]));
  ASSERT_TRUE(actor.enqueueCommit(snapshots[2]));
  std::this_thread::sleep_for(1ms);

  const PersistenceActorStatus status = actor.status();
  EXPECT_EQ(status.latest_scene_revision, 3u);
  EXPECT_EQ(status.durable_scene_revision, 0u);
  EXPECT_EQ(status.undurable_revisions, 3u);
  EXPECT_TRUE(status.soft_lag_reached);
  EXPECT_TRUE(status.hard_lag_reached);
  EXPECT_FALSE(status.admission_allowed);
  EXPECT_FALSE(actor.admissionAllowed());
  EXPECT_GT(status.oldest_undurable_age, 0ns);

  actor.start();
  actor.stop();
  EXPECT_EQ(actor.durableRevision(), 3u);
}

TEST(PersistenceActor, ReliableBoundedQueueBackpressuresWithoutDropping) {
  TemporaryPersistenceDatabase database;
  PersistenceActorConfig config = slowConfig();
  config.queue_capacity = 1;
  PersistenceActor actor(database.path(), config);
  ASSERT_TRUE(actor.open());
  const std::vector<SceneSnapshot> snapshots = makeSnapshots(2);

  ASSERT_TRUE(actor.enqueueCommit(snapshots[0]));
  std::future<PushResult<SceneSnapshot>> second = std::async(
      std::launch::async,
      [&actor, &snapshots]() { return actor.enqueueCommit(snapshots[1]); });

  EXPECT_EQ(second.wait_for(20ms), std::future_status::timeout);
  actor.start();
  const std::future_status unblocked = second.wait_for(1s);
  EXPECT_EQ(unblocked, std::future_status::ready);
  if (unblocked == std::future_status::ready) {
    EXPECT_TRUE(second.get().accepted());
  }
  actor.stop();

  const PersistenceActorStatus status = actor.status();
  EXPECT_EQ(status.commits_enqueued, 2u);
  EXPECT_EQ(status.commit_queue.replaced, 0u);
  EXPECT_EQ(status.commit_queue.rejected, 0u);
  EXPECT_GE(status.commit_queue.producer_wait_count, 1u);
  EXPECT_EQ(status.durable_scene_revision, 2u);
}

TEST(PersistenceActor, BatchAndPeriodicFlushAdvanceAckWatermark) {
  TemporaryPersistenceDatabase database;
  PersistenceActorConfig config;
  config.flush_period = 20ms;
  config.flush_batch_size = 2;
  config.queue_capacity = 8;
  config.max_undurable_revisions = 8;
  config.hard_max_undurable_revisions = 16;
  PersistenceActor actor(database.path(), config);
  ASSERT_TRUE(actor.open());

  std::atomic<SceneRevision> acknowledged{0};
  actor.setDurabilityAckCallback([&acknowledged](SceneRevision revision) {
    acknowledged.store(revision, std::memory_order_release);
    return true;
  });
  actor.start();

  const std::vector<SceneSnapshot> snapshots = makeSnapshots(3);
  ASSERT_TRUE(actor.enqueueCommit(snapshots[0]));
  ASSERT_TRUE(actor.enqueueCommit(snapshots[1]));
  ASSERT_TRUE(actor.waitUntilDurable(2, 1s));
  ASSERT_TRUE(waitFor(
      [&acknowledged]() {
        return acknowledged.load(std::memory_order_acquire) >= 2;
      },
      1s));

  ASSERT_TRUE(actor.enqueueCommit(snapshots[2]));
  ASSERT_TRUE(actor.waitUntilDurable(3, 1s));
  ASSERT_TRUE(waitFor(
      [&acknowledged]() {
        return acknowledged.load(std::memory_order_acquire) >= 3;
      },
      1s));
  actor.stop();

  const PersistenceActorStatus status = actor.status();
  EXPECT_EQ(status.last_acknowledged_revision, 3u);
  EXPECT_FALSE(status.durability_ack_pending);
  EXPECT_GE(status.successful_flushes, 2u);
  EXPECT_TRUE(status.healthy) << status.last_error;
}

TEST(PersistenceActor, StopDrainsQueueAndGracefullyFlushesLatestRevision) {
  TemporaryPersistenceDatabase database;
  PersistenceActor actor(database.path(), slowConfig());
  ASSERT_TRUE(actor.open());
  actor.start();

  const std::vector<SceneSnapshot> snapshots = makeSnapshots(3);
  ASSERT_TRUE(actor.enqueueCommit(snapshots[0]));
  ASSERT_TRUE(actor.enqueueCommit(snapshots[1]));
  ASSERT_TRUE(actor.enqueueCommit(snapshots[2]));
  actor.stop();

  const PersistenceActorStatus status = actor.status();
  EXPECT_TRUE(status.graceful_shutdown_complete) << status.last_error;
  EXPECT_EQ(status.latest_scene_revision, 3u);
  EXPECT_EQ(status.durable_scene_revision, 3u);
  EXPECT_EQ(status.undurable_revisions, 0u);
  EXPECT_EQ(status.commit_queue.depth, 0u);

  const SceneRestoreResult restored = actor.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);
  EXPECT_EQ(restored.snapshot.revision(), 3u);
  EXPECT_EQ(restored.snapshot.durableRevision(), 3u);
}

TEST(PersistenceActor, StartupRewindSynchronizesAdmissionWatermarks) {
  TemporaryPersistenceDatabase database;
  const std::vector<SceneSnapshot> snapshots = makeSnapshots(3);
  {
    SceneStore seed(database.path());
    ASSERT_TRUE(seed.open());
    for (const SceneSnapshot& snapshot : snapshots) {
      ASSERT_TRUE(seed.enqueueCommit(snapshot));
    }
    ASSERT_TRUE(seed.closeGracefully());
  }

  PersistenceActor actor(database.path(), slowConfig());
  ASSERT_TRUE(actor.open());
  ASSERT_TRUE(actor.rewindTo(2));
  EXPECT_EQ(actor.latestRevision(), 2U);
  EXPECT_EQ(actor.durableRevision(), 2U);
  EXPECT_EQ(actor.restoreLatest().snapshot.revision(), 2U);
  EXPECT_TRUE(actor.enqueueCommit(snapshots[2]).accepted());
  actor.start();
  actor.stop();
  EXPECT_EQ(actor.restoreLatest().snapshot.revision(), 3U);
}

TEST(PersistenceActor, RejectsNonContiguousRevisionBeforeQueueAdmission) {
  TemporaryPersistenceDatabase database;
  PersistenceActor actor(database.path(), slowConfig());
  ASSERT_TRUE(actor.open());
  const std::vector<SceneSnapshot> snapshots = makeSnapshots(2);

  const PushResult<SceneSnapshot> skipped =
      actor.enqueueCommit(snapshots[1]);
  EXPECT_EQ(skipped.outcome, PushOutcome::kRejected);
  ASSERT_TRUE(skipped.unconsumed_item.has_value());
  EXPECT_EQ(skipped.unconsumed_item->revision(), 2u);
  EXPECT_EQ(actor.latestRevision(), 0u);
  const PersistenceActorStatus status = actor.status();
  EXPECT_EQ(status.commits_rejected, 1u);
  EXPECT_FALSE(status.healthy);
  EXPECT_FALSE(status.admission_allowed);
  EXPECT_FALSE(actor.admissionAllowed());
  EXPECT_NE(status.last_error.find("non-contiguous"), std::string::npos);
}

TEST(PersistenceActor,
     PermanentWriterFailureReleasesBlockedProducerAndStopsBoundedly) {
  TemporaryPersistenceDatabase database;
  PersistenceActorConfig config = slowConfig();
  config.flush_period = 1ms;
  config.flush_batch_size = 1;
  config.queue_capacity = 1;
  config.max_undurable_revisions = 3;
  config.hard_max_undurable_revisions = 3;
  config.terminal_failure_timeout = 50ms;
  PersistenceActor actor(database.path(), config);
  ASSERT_TRUE(actor.open());

  sqlite3* blocker = nullptr;
  ASSERT_EQ(sqlite3_open(database.path().c_str(), &blocker), SQLITE_OK);
  ASSERT_NE(blocker, nullptr);
  ASSERT_EQ(sqlite3_exec(blocker, "BEGIN IMMEDIATE", nullptr, nullptr,
                         nullptr),
            SQLITE_OK);

  actor.start();
  const std::vector<SceneSnapshot> snapshots = makeSnapshots(32);
  ASSERT_TRUE(actor.enqueueCommit(snapshots[0]));
  std::future<PushResult<SceneSnapshot>> blocked = std::async(
      std::launch::async,
      [&actor, &snapshots]() {
        for (std::size_t index = 1; index < snapshots.size(); ++index) {
          PushResult<SceneSnapshot> result =
              actor.enqueueCommit(snapshots[index]);
          if (!result.accepted()) {
            return result;
          }
        }
        PushResult<SceneSnapshot> exhausted;
        exhausted.outcome = PushOutcome::kAccepted;
        return exhausted;
      });

  ASSERT_EQ(blocked.wait_for(2s), std::future_status::ready)
      << "terminal persistence fault must wake a reliable producer";
  const PushResult<SceneSnapshot> rejected = blocked.get();
  EXPECT_FALSE(rejected.accepted());
  EXPECT_TRUE(rejected.outcome == PushOutcome::kStopped ||
              rejected.outcome == PushOutcome::kRejected);

  const auto stop_started = std::chrono::steady_clock::now();
  actor.stop();
  const auto stop_elapsed = std::chrono::steady_clock::now() - stop_started;
  EXPECT_LT(stop_elapsed, 2s);
  const PersistenceActorStatus status = actor.status();
  EXPECT_FALSE(status.healthy);
  EXPECT_FALSE(status.admission_allowed);
  EXPECT_FALSE(status.graceful_shutdown_complete);
  EXPECT_LT(status.durable_scene_revision, status.latest_scene_revision);
  EXPECT_NE(status.last_error.find("terminal"), std::string::npos);

  EXPECT_EQ(sqlite3_exec(blocker, "ROLLBACK", nullptr, nullptr, nullptr),
            SQLITE_OK);
  sqlite3_close(blocker);
}

TEST(PersistenceActor, CommitsDurableOutboxWithOwningSceneRevision) {
  TemporaryPersistenceDatabase database;
  const std::vector<SceneSnapshot> snapshots = makeSnapshots(1);
  DurableTaskSpec task;
  task.task_id = "dam-object-1-appearance-1";
  task.dedupe_key = task.task_id;
  task.task_type = "dam.interactive.v1";
  task.payload = R"({"object_id":1,"appearance_revision":1})";
  task.scene_revision = snapshots.front().revision();

  {
    PersistenceActor actor(database.path(), slowConfig());
    ASSERT_TRUE(actor.open());
    actor.start();
    ASSERT_TRUE(actor.enqueueCommit(snapshots.front(), {task}));
    actor.stop();
    ASSERT_TRUE(actor.status().graceful_shutdown_complete);
  }

  SceneStore restored(database.path());
  ASSERT_TRUE(restored.open());
  const SceneRestoreResult scene = restored.restoreLatest();
  ASSERT_TRUE(scene.status) << scene.status.error;
  ASSERT_TRUE(scene.found);
  EXPECT_EQ(scene.snapshot.revision(), task.scene_revision);
  const TaskLookupResult outbox = restored.lookupTask(task.task_id);
  ASSERT_TRUE(outbox.status) << outbox.status.error;
  ASSERT_TRUE(outbox.task.has_value());
  EXPECT_EQ(outbox.task->task.payload, task.payload);
  EXPECT_EQ(outbox.task->state, DurableTaskState::kPending);
}

}  // namespace
}  // namespace roomie
