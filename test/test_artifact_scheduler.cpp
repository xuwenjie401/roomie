#include <gtest/gtest.h>

#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <unistd.h>

#include "roomie/artifacts/artifact_scheduler.hpp"
#include "roomie/scene/scene_reducer.hpp"

namespace roomie {
namespace {

class TemporaryDatabase {
 public:
  TemporaryDatabase() {
    char pattern[] = "/tmp/roomie_artifact_scheduler_XXXXXX";
    const int descriptor = mkstemp(pattern);
    if (descriptor < 0) {
      throw std::runtime_error("mkstemp failed");
    }
    close(descriptor);
    path_ = pattern;
  }

  ~TemporaryDatabase() {
    std::remove(path_.c_str());
    std::remove((path_ + "-wal").c_str());
    std::remove((path_ + "-shm").c_str());
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

ObjectNode artifactNode(SceneObjectId object_id, std::string label) {
  ObjectNode node;
  node.object_id = object_id;
  node.semantic_id = object_id + 100;
  node.label = std::move(label);
  node.center_world = Eigen::Vector3f(static_cast<float>(object_id), 0.0f, 1.0f);
  node.size_m = Eigen::Vector3f(0.4f, 0.5f, 0.6f);
  node.active = true;
  node.publishable = true;
  return node;
}

struct PreparedScene {
  std::unique_ptr<ReducerCore> reducer = std::make_unique<ReducerCore>();
  std::vector<SceneSnapshot> history;
  ObjectDependency dependency;
  std::string snapshot_set_hash;
};

PreparedScene prepareScene(SceneObjectId object_id = 4,
                           bool add_merge_target = false) {
  PreparedScene scene;
  LoadSceneCommand load;
  load.graph.objects.push_back(artifactNode(object_id, "chair"));
  if (add_merge_target) {
    load.graph.objects.push_back(artifactNode(object_id + 1, "chair"));
  }
  load.graph.next_object_id = object_id + (add_merge_target ? 2 : 1);
  SceneApplyResult loaded = scene.reducer->apply(SceneCommand{load});
  if (!loaded.committedRevision()) {
    throw std::runtime_error(loaded.reason);
  }
  scene.history.push_back(loaded.snapshot);

  ApplySnapshotSetCommand snapshots;
  snapshots.dependency =
      dependencyFor(*scene.reducer->snapshot().findObject(object_id));
  ObjectSnapshotRef snapshot;
  snapshot.image_index = 7;
  snapshot.camera_id = "front";
  snapshot.quality = 0.9f;
  snapshots.snapshots.push_back(snapshot);
  snapshots.snapshot_set_hash = "snapshot-set-v1";
  SceneApplyResult snapshotted =
      scene.reducer->apply(SceneCommand{snapshots});
  if (!snapshotted.committedRevision()) {
    throw std::runtime_error(snapshotted.reason);
  }
  scene.history.push_back(snapshotted.snapshot);
  scene.dependency = dependencyFor(*snapshotted.snapshot.findObject(object_id));
  scene.snapshot_set_hash = snapshots.snapshot_set_hash;
  return scene;
}

void persistPreparedScene(SceneStore* store, const PreparedScene& scene) {
  for (const SceneSnapshot& snapshot : scene.history) {
    const SceneStoreStatus status = store->enqueueCommit(snapshot);
    if (!status) {
      throw std::runtime_error(status.error);
    }
  }
  const SceneStoreStatus status = store->flush();
  if (!status) {
    throw std::runtime_error(status.error);
  }
}

DamTaskIntent taskIntent(const PreparedScene& scene,
                         ArtifactPriority priority,
                         std::string suffix,
                         std::int64_t created_unix_ms) {
  DamTaskIntent intent;
  intent.key.object_id = scene.dependency.object_id;
  intent.key.identity_revision = scene.dependency.identity_revision;
  intent.key.appearance_revision = scene.dependency.appearance_revision;
  intent.key.snapshot_set_hash = scene.snapshot_set_hash;
  intent.key.model_id = "fake-dam-" + suffix;
  intent.key.prompt_hash = "prompt-hash-" + suffix;
  intent.key.output_schema_version = "dam.schema.v1";
  intent.dependency = scene.dependency;
  intent.scene_revision = scene.history.back().revision();
  intent.priority = priority;
  intent.created_unix_ms = created_unix_ms;
  intent.input_payload = "asset:frame-" + suffix;
  return intent;
}

std::string validArtifactJson(std::string description = "a red chair") {
  nlohmann::json value = {
      {"canonical_name", "chair"},
      {"short_description", description},
      {"retrieval_text", "red upholstered chair with four legs"},
      {"visual_attributes",
       {{"colors", nlohmann::json::array({"red"})},
        {"materials", nlohmann::json::array({"fabric"})},
        {"shape", nlohmann::json::array({"rectangular back"})},
        {"visible_parts", nlohmann::json::array({"seat", "legs"})},
        {"state_or_pose", nlohmann::json::array({"upright"})},
        {"distinctive_marks", nlohmann::json::array()},
        {"visible_text", nlohmann::json::array()}}},
      {"uncertain_or_not_visible", nlohmann::json::array({"rear"})},
      {"confidence", 0.91},
      {"evidence_snapshot_ids", nlohmann::json::array({"frame-a"})},
      {"mask_source", "instance_mask"}};
  return value.dump();
}

class FakeDamWorker final : public DamWorker {
 public:
  explicit FakeDamWorker(DamWorkerResponse response,
                         std::optional<std::int64_t> heartbeat_at = std::nullopt)
      : response_(std::move(response)), heartbeat_at_(heartbeat_at) {}

  DamWorkerResponse describe(const DamTaskRequest& request,
                             const DamLeaseHeartbeat& heartbeat) override {
    ++calls;
    requests.push_back(request);
    if (heartbeat_at_) {
      heartbeat_results.push_back(heartbeat(*heartbeat_at_));
    }
    return response_;
  }

  int calls = 0;
  std::vector<DamTaskRequest> requests;
  std::vector<bool> heartbeat_results;

 private:
  DamWorkerResponse response_;
  std::optional<std::int64_t> heartbeat_at_;
};

DamWorkerResponse successfulResponse(std::string description = "a red chair") {
  DamWorkerResponse response;
  response.success = true;
  response.raw_output = validArtifactJson(std::move(description));
  return response;
}

TEST(ArtifactScheduler, StableKeyContainsEveryDependencyButNotRawObbRevision) {
  const PreparedScene scene = prepareScene();
  DamTaskIntent intent =
      taskIntent(scene, ArtifactPriority::kInteractive, "stable", 500);
  ArtifactScheduler scheduler(nullptr);
  SceneStoreStatus status;
  const DurableTaskSpec first = scheduler.makeTaskSpec(intent, &status);
  ASSERT_TRUE(status) << status.error;
  EXPECT_EQ(first.task_id, first.dedupe_key);
  EXPECT_NE(first.task_id.find("object_id=4"), std::string::npos);
  EXPECT_NE(first.task_id.find("identity_revision=1"), std::string::npos);
  EXPECT_NE(first.task_id.find("appearance_revision=1"), std::string::npos);
  EXPECT_NE(first.task_id.find("snapshot-set-v1"), std::string::npos);
  EXPECT_NE(first.task_id.find("fake-dam-stable"), std::string::npos);
  EXPECT_NE(first.task_id.find("prompt-hash-stable"), std::string::npos);
  EXPECT_NE(first.task_id.find("dam.schema.v1"), std::string::npos);
  EXPECT_EQ(first.task_type, ArtifactScheduler::kInteractiveTaskType);

  intent.dependency.obb_revision += 1000;
  const DurableTaskSpec geometry_changed =
      scheduler.makeTaskSpec(intent, &status);
  ASSERT_TRUE(status) << status.error;
  EXPECT_EQ(geometry_changed.task_id, first.task_id);
  EXPECT_EQ(geometry_changed.dedupe_key, first.dedupe_key);
}

TEST(ArtifactScheduler, NonOverdueInteractivePrecedesBulk) {
  TemporaryDatabase database;
  PreparedScene scene = prepareScene();
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  persistPreparedScene(&store, scene);

  ArtifactScheduler scheduler(&store);
  const DamTaskIntent bulk =
      taskIntent(scene, ArtifactPriority::kBulk, "bulk", 100);
  const DamTaskIntent interactive =
      taskIntent(scene, ArtifactPriority::kInteractive, "interactive", 100);
  ASSERT_TRUE(scheduler.ensureTask(bulk));
  ASSERT_TRUE(scheduler.ensureTask(interactive));

  ArtifactLeaseResult first = scheduler.leaseNext("worker", 100);
  ASSERT_TRUE(first.status) << first.status.error;
  ASSERT_TRUE(first.lease.has_value());
  EXPECT_EQ(first.lease->request.priority, ArtifactPriority::kInteractive);
  EXPECT_EQ(first.lease->request.due_unix_ms, 10'100);
  ASSERT_TRUE(scheduler.complete(*first.lease, 101));

  // Due time ranks/observes the SLO; it is not a destructive TTL. The bulk
  // job remains durable and leasable after its 120 s target.
  ArtifactLeaseResult second = scheduler.leaseNext("worker", 120'101);
  ASSERT_TRUE(second.status) << second.status.error;
  ASSERT_TRUE(second.lease.has_value());
  EXPECT_EQ(second.lease->request.priority, ArtifactPriority::kBulk);
  EXPECT_EQ(second.lease->request.due_unix_ms, 120'100);
  FakeDamWorker late_worker(successfulResponse());
  const ArtifactExecutionResult late =
      scheduler.execute(*second.lease, &late_worker, 120'102);
  EXPECT_EQ(late.status, ArtifactExecutionStatus::kReadyToApply);
  EXPECT_TRUE(late.slo_violation);
  ASSERT_TRUE(late.command);
  EXPECT_EQ(late.command->artifact_slo.origin_created_unix_ms, 100);
  EXPECT_EQ(late.command->artifact_slo.due_unix_ms, 120'100);
  EXPECT_EQ(late.command->artifact_slo.priority, ArtifactPriority::kBulk);
}

TEST(ArtifactScheduler, OverdueBulkPrecedesNewInteractive) {
  TemporaryDatabase database;
  PreparedScene scene = prepareScene();
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  persistPreparedScene(&store, scene);

  ArtifactScheduler scheduler(&store);
  DamTaskIntent overdue_bulk =
      taskIntent(scene, ArtifactPriority::kBulk, "overdue-bulk", 100);
  overdue_bulk.due_unix_ms = 200;
  DamTaskIntent new_interactive = taskIntent(
      scene, ArtifactPriority::kInteractive, "new-interactive", 201);
  new_interactive.due_unix_ms = 1'000;
  ASSERT_TRUE(scheduler.ensureTask(overdue_bulk));
  ASSERT_TRUE(scheduler.ensureTask(new_interactive));

  const ArtifactLeaseResult leased = scheduler.leaseNext("worker", 201);
  ASSERT_TRUE(leased.status) << leased.status.error;
  ASSERT_TRUE(leased.lease);
  EXPECT_EQ(leased.lease->request.priority, ArtifactPriority::kBulk);
  EXPECT_EQ(leased.lease->request.due_unix_ms, 200);
  EXPECT_EQ(leased.lease->request.key.model_id, "fake-dam-overdue-bulk");
}

TEST(ArtifactScheduler, DurableJsonRejectsNegativeAndOverflowingIntegers) {
  auto expect_invalid = [](const std::string& field,
                           const nlohmann::json& invalid_value,
                           const std::string& expected_error) {
    TemporaryDatabase database;
    PreparedScene scene = prepareScene();
    SceneStore store(database.path());
    ASSERT_TRUE(store.open());
    persistPreparedScene(&store, scene);
    ArtifactScheduler scheduler(&store);
    SceneStoreStatus status;
    DurableTaskSpec task = scheduler.makeTaskSpec(
        taskIntent(scene, ArtifactPriority::kInteractive, field, 100),
        &status);
    ASSERT_TRUE(status) << status.error;
    nlohmann::json payload = nlohmann::json::parse(task.payload);
    if (field == "identity_revision") {
      payload["key"][field] = invalid_value;
    } else {
      payload[field] = invalid_value;
    }
    if (field == "origin_steady_ns") {
      payload["steady_clock_epoch"] = {{"high", 1}, {"low", 2}};
      payload["due_steady_ns"] = 10;
    }
    task.payload = payload.dump();
    ASSERT_TRUE(store.ensureTask(task));
    const ArtifactLeaseResult lease = scheduler.leaseNext("worker", 100);
    EXPECT_FALSE(lease.status);
    EXPECT_NE(lease.status.error.find(expected_error), std::string::npos)
        << lease.status.error;
  };

  expect_invalid("identity_revision", -1, "negative");
  expect_invalid("due_unix_ms",
                 std::numeric_limits<std::uint64_t>::max(),
                 "exceeds int64 range");
  expect_invalid("origin_steady_ns", -1, "negative");
}

TEST(ArtifactScheduler, LeaseExpiryAndAttemptFenceMakeDuplicateExecutionSafe) {
  TemporaryDatabase database;
  PreparedScene scene = prepareScene();
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  persistPreparedScene(&store, scene);

  ArtifactSchedulerConfig config;
  config.lease_duration_ms = 100;
  ArtifactScheduler scheduler(&store, config);
  ASSERT_TRUE(scheduler.ensureTask(
      taskIntent(scene, ArtifactPriority::kInteractive, "aba", 1000)));

  ArtifactLeaseResult first_result = scheduler.leaseNext("same-owner", 1000);
  ASSERT_TRUE(first_result.status);
  ASSERT_TRUE(first_result.lease.has_value());
  const ArtifactLease first = *first_result.lease;
  EXPECT_EQ(first.attempt(), 1u);
  FakeDamWorker first_worker(successfulResponse());
  const ArtifactExecutionResult first_execution =
      scheduler.execute(first, &first_worker, 1001);
  ASSERT_EQ(first_execution.status, ArtifactExecutionStatus::kReadyToApply);
  ASSERT_TRUE(first_execution.command.has_value());
  const SceneApplyResult first_apply =
      scene.reducer->apply(SceneCommand{*first_execution.command});
  ASSERT_TRUE(first_apply.committedRevision()) << first_apply.reason;

  ArtifactLeaseResult before_expiry =
      scheduler.leaseNext("other-owner", 1099);
  ASSERT_TRUE(before_expiry.status);
  EXPECT_FALSE(before_expiry.lease.has_value());
  // SceneStore's raw completion API does not itself compare completion time
  // with lease expiry; the scheduler adapter closes that expiry-fencing gap.
  EXPECT_FALSE(scheduler.complete(first, 1100));
  ArtifactLeaseResult second_result =
      scheduler.leaseNext("same-owner", 1100);
  ASSERT_TRUE(second_result.status) << second_result.status.error;
  ASSERT_TRUE(second_result.lease.has_value());
  const ArtifactLease second = *second_result.lease;
  EXPECT_EQ(second.attempt(), 2u);

  EXPECT_FALSE(scheduler.heartbeat(first, 1101));
  EXPECT_FALSE(scheduler.complete(first, 1101));

  FakeDamWorker second_worker(successfulResponse());
  const ArtifactExecutionResult duplicate =
      scheduler.execute(second, &second_worker, 1101);
  ASSERT_EQ(duplicate.status, ArtifactExecutionStatus::kReadyToApply);
  ASSERT_TRUE(duplicate.command.has_value());
  const SceneApplyResult duplicate_apply =
      scene.reducer->apply(SceneCommand{*duplicate.command});
  EXPECT_EQ(duplicate_apply.status, SceneApplyStatus::kNoOp);
  EXPECT_TRUE(scheduler.complete(second, 1102));
  EXPECT_TRUE(scheduler.complete(second, 1103));
}

TEST(ArtifactScheduler, RetryUsesFencedAttemptAndDeterministicBackoff) {
  TemporaryDatabase database;
  PreparedScene scene = prepareScene();
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  persistPreparedScene(&store, scene);

  ArtifactSchedulerConfig config;
  config.lease_duration_ms = 100;
  config.retry_initial_backoff_ms = 50;
  config.retry_max_backoff_ms = 1000;
  ArtifactScheduler scheduler(&store, config);
  const DamTaskIntent intent =
      taskIntent(scene, ArtifactPriority::kInteractive, "retry", 100);
  ASSERT_TRUE(scheduler.ensureTask(intent));

  ArtifactLeaseResult leased_result = scheduler.leaseNext("worker-a", 100);
  ASSERT_TRUE(leased_result.status);
  ASSERT_TRUE(leased_result.lease.has_value());
  const ArtifactLease attempt_one = *leased_result.lease;
  DamWorkerResponse failure;
  failure.success = false;
  failure.retryable = true;
  failure.error = "temporary GPU failure";
  FakeDamWorker failing_worker(failure);
  const ArtifactExecutionResult execution =
      scheduler.execute(attempt_one, &failing_worker, 101);
  EXPECT_EQ(execution.status, ArtifactExecutionStatus::kRetryableFailure);

  const ArtifactRetryResult retry =
      scheduler.retry(attempt_one, 102, execution.error);
  ASSERT_TRUE(retry.status) << retry.status.error;
  EXPECT_EQ(retry.retry_at_unix_ms, 152);
  EXPECT_FALSE(scheduler.retry(attempt_one, 103, "duplicate retry").status);
  // SceneStore reuses not_before for retry state. The scheduler adapter keeps
  // re-observing the same durable intent idempotent after that mutation.
  EXPECT_TRUE(scheduler.ensureTask(intent));

  ArtifactLeaseResult too_early = scheduler.leaseNext("worker-b", 151);
  ASSERT_TRUE(too_early.status);
  EXPECT_FALSE(too_early.lease.has_value());
  ArtifactLeaseResult attempt_two_result =
      scheduler.leaseNext("worker-b", 152);
  ASSERT_TRUE(attempt_two_result.status);
  ASSERT_TRUE(attempt_two_result.lease.has_value());
  EXPECT_EQ(attempt_two_result.lease->attempt(), 2u);
  EXPECT_EQ(attempt_two_result.lease->durable.last_error,
            "temporary GPU failure");

  FakeDamWorker recovered(successfulResponse(), 160);
  const ArtifactExecutionResult recovered_execution =
      scheduler.execute(*attempt_two_result.lease, &recovered, 153);
  EXPECT_EQ(recovered_execution.status,
            ArtifactExecutionStatus::kReadyToApply);
  ASSERT_EQ(recovered.heartbeat_results.size(), 1u);
  EXPECT_TRUE(recovered.heartbeat_results.front());
  EXPECT_TRUE(scheduler.complete(*attempt_two_result.lease, 170));
}

TEST(ArtifactScheduler, PermanentFailureIsFencedTerminalAndNotReLeased) {
  TemporaryDatabase database;
  PreparedScene scene = prepareScene();
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  persistPreparedScene(&store, scene);

  ArtifactSchedulerConfig config;
  config.lease_duration_ms = 100;
  ArtifactScheduler scheduler(&store, config);
  const DamTaskIntent intent =
      taskIntent(scene, ArtifactPriority::kInteractive, "terminal", 100);
  ASSERT_TRUE(scheduler.ensureTask(intent));
  ArtifactLeaseResult leased = scheduler.leaseNext("worker-a", 100);
  ASSERT_TRUE(leased.status) << leased.status.error;
  ASSERT_TRUE(leased.lease.has_value());

  DamWorkerResponse permanent;
  permanent.success = false;
  permanent.retryable = false;
  permanent.error = "unsupported model schema";
  FakeDamWorker worker(permanent);
  const ArtifactExecutionResult execution =
      scheduler.execute(*leased.lease, &worker, 101);
  ASSERT_EQ(execution.status, ArtifactExecutionStatus::kPermanentFailure);

  ArtifactLease wrong_owner = *leased.lease;
  wrong_owner.durable.lease_owner = "worker-b";
  EXPECT_FALSE(scheduler.fail(wrong_owner, 102, execution.error));
  EXPECT_FALSE(scheduler.fail(*leased.lease, 200, execution.error))
      << "expiry is part of the terminal failure fence";
  ASSERT_TRUE(scheduler.fail(*leased.lease, 103, execution.error));
  EXPECT_TRUE(scheduler.fail(*leased.lease, 1000, "replayed ack"));
  EXPECT_FALSE(scheduler.complete(*leased.lease, 104));
  EXPECT_FALSE(scheduler.retry(*leased.lease, 104, "must not retry").status);

  const TaskLookupResult failed = store.lookupTask(leased.lease->taskId());
  ASSERT_TRUE(failed.status) << failed.status.error;
  ASSERT_TRUE(failed.task.has_value());
  EXPECT_EQ(failed.task->state, DurableTaskState::kFailed);
  EXPECT_EQ(failed.task->last_error, "unsupported model schema");
  EXPECT_EQ(failed.task->failed_at_unix_ms, 103);
  EXPECT_EQ(failed.task->failed_by, "worker-a");
  EXPECT_TRUE(scheduler.ensureTask(intent));
  const ArtifactLeaseResult none = scheduler.leaseNext("worker-c", 5000);
  ASSERT_TRUE(none.status) << none.status.error;
  EXPECT_FALSE(none.lease.has_value());
}

TEST(ArtifactScheduler,
     SemanticFusionDoesNotSupersedeUnchangedAppearanceTask) {
  TemporaryDatabase database;
  PreparedScene scene = prepareScene();
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  persistPreparedScene(&store, scene);

  ArtifactScheduler scheduler(&store);
  const DamTaskIntent intent =
      taskIntent(scene, ArtifactPriority::kInteractive, "semantic-wildcard", 10);
  ASSERT_NE(intent.dependency.semantic_revision, 0u);
  ASSERT_TRUE(scheduler.ensureTask(intent));
  ArtifactLeaseResult leased = scheduler.leaseNext("worker", 10);
  ASSERT_TRUE(leased.status) << leased.status.error;
  ASSERT_TRUE(leased.lease.has_value());
  EXPECT_EQ(leased.lease->request.dependency.semantic_revision, 0u);

  auto advanced_state =
      std::make_shared<SceneState>(*scene.reducer->snapshot().statePtr());
  SceneObjectTable advanced_objects = scene.reducer->snapshot().objects();
  const SceneObjectPtr original =
      advanced_objects.at(intent.key.object_id);
  auto advanced_object = std::make_shared<SceneObject>(*original);
  auto advanced_semantic =
      std::make_shared<SemanticComponent>(*original->semantic);
  ++advanced_semantic->revision;
  ++advanced_semantic->support_count;
  advanced_object->semantic = std::move(advanced_semantic);
  advanced_objects[intent.key.object_id] = std::move(advanced_object);
  advanced_state->objects =
      std::make_shared<const SceneObjectTable>(std::move(advanced_objects));
  ++advanced_state->latest_scene_revision;

  ReducerCore advanced_reducer;
  LoadSceneCommand restore;
  restore.restored_state =
      std::shared_ptr<const SceneState>(std::move(advanced_state));
  restore.restored_revision = restore.restored_state->latest_scene_revision;
  restore.durable_revision = restore.restored_state->durable_scene_revision;
  const SceneApplyResult restored =
      advanced_reducer.apply(SceneCommand{restore});
  ASSERT_TRUE(restored.committedRevision()) << restored.reason;
  const SceneObjectPtr fused =
      restored.snapshot.findExactObject(intent.key.object_id);
  ASSERT_TRUE(fused && fused->semantic && fused->artifact);
  EXPECT_NE(fused->semantic->revision, intent.dependency.semantic_revision);
  EXPECT_EQ(fused->artifact->appearance_revision,
            intent.key.appearance_revision);
  EXPECT_EQ(fused->artifact->snapshot_set_hash, intent.key.snapshot_set_hash);
  EXPECT_EQ(inspectDamDependency(restored.snapshot, leased.lease->request),
            DamDependencyFreshness::kCurrent);

  FakeDamWorker worker(successfulResponse("semantic fusion safe"));
  const ArtifactExecutionResult execution =
      scheduler.execute(*leased.lease, &worker, 11);
  ASSERT_EQ(execution.status, ArtifactExecutionStatus::kReadyToApply);
  ASSERT_TRUE(execution.command.has_value());
  EXPECT_EQ(execution.command->dependency.semantic_revision, 0u);

  ApplyDescriptionArtifactCommand strict = *execution.command;
  strict.dependency.semantic_revision = intent.dependency.semantic_revision;
  EXPECT_EQ(advanced_reducer.apply(SceneCommand{strict}).status,
            SceneApplyStatus::kRejected)
      << "nonzero semantic dependencies retain the legacy exact-CAS contract";
  const SceneApplyResult applied =
      advanced_reducer.apply(SceneCommand{*execution.command});
  EXPECT_TRUE(applied.committedRevision()) << applied.reason;
}

TEST(ArtifactScheduler, StaleAppearanceResultStillBecomesReducerCasCommand) {
  TemporaryDatabase database;
  PreparedScene scene = prepareScene();
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  persistPreparedScene(&store, scene);

  ArtifactScheduler scheduler(&store);
  const DamTaskIntent intent =
      taskIntent(scene, ArtifactPriority::kInteractive, "stale", 10);
  ASSERT_TRUE(scheduler.ensureTask(intent));
  ArtifactLeaseResult leased = scheduler.leaseNext("worker", 10);
  ASSERT_TRUE(leased.status);
  ASSERT_TRUE(leased.lease.has_value());

  ApplySnapshotSetCommand newer_snapshots;
  newer_snapshots.dependency = scene.dependency;
  ObjectSnapshotRef newer;
  newer.image_index = 9;
  newer.camera_id = "rear";
  newer_snapshots.snapshots.push_back(newer);
  newer_snapshots.snapshot_set_hash = "snapshot-set-v2";
  const SceneApplyResult appearance_changed =
      scene.reducer->apply(SceneCommand{newer_snapshots});
  ASSERT_TRUE(appearance_changed.committedRevision())
      << appearance_changed.reason;
  EXPECT_EQ(inspectDamDependency(scene.reducer->snapshot(),
                                 leased.lease->request),
            DamDependencyFreshness::kAppearanceStale);

  FakeDamWorker worker(successfulResponse("description from old pixels"));
  const ArtifactExecutionResult execution =
      scheduler.execute(*leased.lease, &worker, 11);
  ASSERT_EQ(execution.status, ArtifactExecutionStatus::kReadyToApply);
  ASSERT_TRUE(execution.command.has_value());
  EXPECT_EQ(execution.command->dependency.appearance_revision,
            intent.key.appearance_revision);
  EXPECT_EQ(execution.command->input_hash, canonicalDamTaskKey(intent.key));
  const SceneApplyResult stale_apply =
      scene.reducer->apply(SceneCommand{*execution.command});
  EXPECT_EQ(stale_apply.status, SceneApplyStatus::kRejected);
  EXPECT_NE(stale_apply.reason.find("appearance dependency is stale"),
            std::string::npos);
  // A dependency rejection is terminal supersession, so the leased intent can
  // be acknowledged instead of retried forever.
  EXPECT_TRUE(scheduler.complete(*leased.lease, 12));
}

TEST(ArtifactScheduler, MergeAliasIsReportedAndCannotReceiveLateArtifact) {
  PreparedScene scene = prepareScene(20, true);
  DamTaskRequest request;
  request.key.object_id = scene.dependency.object_id;
  request.key.identity_revision = scene.dependency.identity_revision;
  request.key.appearance_revision = scene.dependency.appearance_revision;
  request.key.snapshot_set_hash = scene.snapshot_set_hash;
  request.key.model_id = "fake";
  request.key.prompt_hash = "prompt";
  request.key.output_schema_version = "v1";
  request.dependency = scene.dependency;

  MergeObjectsMutation merge;
  merge.retired_object_id = 20;
  merge.canonical_object_id = 21;
  ApplyObservationBatchCommand batch;
  batch.associated_mutations.push_back(merge);
  const SceneApplyResult merged = scene.reducer->apply(SceneCommand{batch});
  ASSERT_TRUE(merged.committedRevision()) << merged.reason;
  EXPECT_EQ(inspectDamDependency(merged.snapshot, request),
            DamDependencyFreshness::kRetiredAlias);

  ApplyDescriptionArtifactCommand late;
  late.dependency = request.dependency;
  late.description = "late";
  late.input_hash = canonicalDamTaskKey(request.key);
  late.model_id = request.key.model_id;
  late.schema_version = request.key.output_schema_version;
  EXPECT_EQ(scene.reducer->apply(SceneCommand{late}).status,
            SceneApplyStatus::kRejected);
}

TEST(ArtifactScheduler, ParsesValidRepairAndExplicitFallbackPaths) {
  const std::string valid_raw = validArtifactJson();
  const DamArtifact valid = parseDamArtifact(valid_raw, "v1");
  EXPECT_EQ(valid.parse_path, DamArtifactParsePath::kSchemaValid);
  EXPECT_TRUE(valid.structured());
  EXPECT_EQ(valid.short_description, "a red chair");
  EXPECT_EQ(valid.raw_text, valid_raw);
  EXPECT_FALSE(valid.normalized_json.empty());

  const std::string fenced = "model preface\n```json\n" + valid_raw +
                             "\n```\nmodel suffix";
  const DamArtifact repaired = parseDamArtifact(fenced, "v1");
  EXPECT_EQ(repaired.parse_path, DamArtifactParsePath::kSchemaRepaired);
  EXPECT_TRUE(repaired.structured());
  EXPECT_EQ(repaired.raw_text, fenced);
  EXPECT_FALSE(repaired.schema_error.empty());

  int repair_calls = 0;
  const DamArtifact callback_repaired = parseDamArtifact(
      R"({"short_description":"incomplete"})", "v1",
      [&repair_calls, &valid_raw](const std::string&, const std::string&) {
        ++repair_calls;
        return std::optional<std::string>(valid_raw);
      });
  EXPECT_EQ(repair_calls, 1);
  EXPECT_EQ(callback_repaired.parse_path,
            DamArtifactParsePath::kSchemaRepaired);

  nlohmann::json worker_repaired_json =
      nlohmann::json::parse(valid_raw);
  worker_repaired_json["_roomie_parse_path"] = "schema_repaired";
  worker_repaired_json["_roomie_schema_error"] =
      "official DAM text was conservatively wrapped";
  const DamArtifact worker_repaired =
      parseDamArtifact(worker_repaired_json.dump(), "v1");
  EXPECT_EQ(worker_repaired.parse_path,
            DamArtifactParsePath::kSchemaRepaired);
  EXPECT_EQ(worker_repaired.schema_error,
            "official DAM text was conservatively wrapped");
  EXPECT_TRUE(worker_repaired.structured());

  const std::string unstructured = "probably a red chair, details uncertain";
  const DamArtifact fallback = parseDamArtifact(
      unstructured, "v1",
      [](const std::string&, const std::string&) {
        return std::optional<std::string>("still not JSON");
      });
  EXPECT_EQ(fallback.parse_path,
            DamArtifactParsePath::kUnstructuredFallback);
  EXPECT_FALSE(fallback.structured());
  EXPECT_EQ(fallback.short_description, unstructured);
  EXPECT_EQ(fallback.raw_text, unstructured);
  EXPECT_TRUE(fallback.normalized_json.empty());
  const nlohmann::json envelope =
      nlohmann::json::parse(fallback.durable_envelope_json);
  EXPECT_EQ(envelope.at("artifact_status"), "unstructured_fallback");
  EXPECT_EQ(envelope.at("raw_text"), unstructured);
  EXPECT_TRUE(envelope.at("structured").is_null());
}

}  // namespace
}  // namespace roomie
