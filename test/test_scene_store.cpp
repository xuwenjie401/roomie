#include <gtest/gtest.h>

#include <sqlite3.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "roomie/scene/scene_reducer.hpp"
#include "roomie/scene/scene_store.hpp"

namespace roomie {
namespace {

class TemporaryDatabase {
 public:
  TemporaryDatabase() {
    char pattern[] = "/tmp/roomie_scene_store_XXXXXX";
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

ObjectNode makeStoreNode(SceneObjectId object_id, std::string label) {
  ObjectNode node;
  node.object_id = object_id;
  node.semantic_id = object_id + 200;
  node.label = std::move(label);
  node.center_world =
      Eigen::Vector3f(static_cast<float>(object_id), 1.0f, 2.0f);
  node.size_m = Eigen::Vector3f(0.4f, 0.5f, 0.6f);
  node.active = true;
  node.publishable = true;
  return node;
}

LoadSceneCommand storeLoadCommand(std::initializer_list<ObjectNode> nodes) {
  LoadSceneCommand command;
  for (const ObjectNode& node : nodes) {
    command.graph.objects.push_back(node);
    command.graph.next_object_id =
        std::max(command.graph.next_object_id, node.object_id + 1);
  }
  return command;
}

SceneSnapshot annotate(ReducerCore* reducer,
                       SceneObjectId object_id,
                       std::string label) {
  ApplyHumanAnnotationCommand command;
  command.object_id = object_id;
  command.patch.label = std::move(label);
  const SceneApplyResult result = reducer->apply(SceneCommand{command});
  if (!result.committedRevision()) {
    throw std::runtime_error(result.reason);
  }
  return result.snapshot;
}

SceneSnapshot geometryOnlyStoreRevision(const SceneSnapshot& previous,
                                        SceneObjectId object_id,
                                        float score) {
  SceneState next = *previous.statePtr();
  next.latest_scene_revision = previous.revision() + 1;
  SceneObjectTable objects = previous.objects();
  const auto existing = objects.find(object_id);
  if (existing == objects.end() || !existing->second ||
      !existing->second->geometry) {
    throw std::runtime_error("geometry-only store test object is missing");
  }
  auto object = std::make_shared<SceneObject>(*existing->second);
  auto geometry =
      std::make_shared<GeometryComponent>(*existing->second->geometry);
  ++geometry->revision;
  geometry->score = score;
  object->geometry = std::move(geometry);
  objects[object_id] = std::move(object);
  next.objects = std::make_shared<const SceneObjectTable>(std::move(objects));
  return SceneSnapshot(
      std::make_shared<const SceneState>(std::move(next)));
}

ApplyObservationBatchCommand storeMutation(ObservationMutation mutation) {
  ApplyObservationBatchCommand command;
  command.association_source = "scene_store_test";
  command.associated_mutations.push_back(std::move(mutation));
  return command;
}

int sqliteScalar(const std::string& path, const char* sql) {
  sqlite3* database = nullptr;
  if (sqlite3_open_v2(path.c_str(), &database, SQLITE_OPEN_READONLY, nullptr) !=
      SQLITE_OK) {
    const std::string error = database ? sqlite3_errmsg(database) : "open failed";
    if (database) {
      sqlite3_close(database);
    }
    throw std::runtime_error(error);
  }
  sqlite3_stmt* statement = nullptr;
  if (sqlite3_prepare_v2(database, sql, -1, &statement, nullptr) != SQLITE_OK ||
      sqlite3_step(statement) != SQLITE_ROW) {
    const std::string error = sqlite3_errmsg(database);
    sqlite3_finalize(statement);
    sqlite3_close(database);
    throw std::runtime_error(error);
  }
  const int value = sqlite3_column_int(statement, 0);
  sqlite3_finalize(statement);
  sqlite3_close(database);
  return value;
}

void sqliteExecute(const std::string& path, const char* sql) {
  sqlite3* database = nullptr;
  if (sqlite3_open_v2(path.c_str(), &database,
                      SQLITE_OPEN_READWRITE, nullptr) != SQLITE_OK) {
    const std::string error = database ? sqlite3_errmsg(database) : "open failed";
    if (database) {
      sqlite3_close(database);
    }
    throw std::runtime_error(error);
  }
  char* error_message = nullptr;
  if (sqlite3_exec(database, sql, nullptr, nullptr, &error_message) !=
      SQLITE_OK) {
    const std::string error =
        error_message ? error_message : sqlite3_errmsg(database);
    sqlite3_free(error_message);
    sqlite3_close(database);
    throw std::runtime_error(error);
  }
  sqlite3_close(database);
}

std::string testPayloadHash(const std::string& payload) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char byte : payload) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << hash;
  return stream.str();
}

void rewriteLatestScenePayload(
    const std::string& path,
    const std::function<void(nlohmann::json&)>& rewrite) {
  sqlite3* database = nullptr;
  if (sqlite3_open_v2(path.c_str(), &database,
                      SQLITE_OPEN_READWRITE, nullptr) != SQLITE_OK) {
    throw std::runtime_error(database ? sqlite3_errmsg(database)
                                      : "open failed");
  }
  sqlite3_stmt* select = nullptr;
  if (sqlite3_prepare_v2(
          database,
          "SELECT revision, payload FROM scene_revisions "
          "ORDER BY revision DESC LIMIT 1;",
          -1, &select, nullptr) != SQLITE_OK ||
      sqlite3_step(select) != SQLITE_ROW) {
    const std::string error = sqlite3_errmsg(database);
    sqlite3_finalize(select);
    sqlite3_close(database);
    throw std::runtime_error(error);
  }
  const sqlite3_int64 revision = sqlite3_column_int64(select, 0);
  const auto* payload_text = sqlite3_column_text(select, 1);
  if (payload_text == nullptr) {
    sqlite3_finalize(select);
    sqlite3_close(database);
    throw std::runtime_error("stored test payload is null");
  }
  nlohmann::json payload = nlohmann::json::parse(
      reinterpret_cast<const char*>(payload_text));
  sqlite3_finalize(select);

  rewrite(payload);
  const std::string encoded = payload.dump();
  const std::string hash = testPayloadHash(encoded);
  sqlite3_stmt* update = nullptr;
  if (sqlite3_prepare_v2(
          database,
          "UPDATE scene_revisions SET payload = ?1, payload_hash = ?2 "
          "WHERE revision = ?3;",
          -1, &update, nullptr) != SQLITE_OK ||
      sqlite3_bind_text(update, 1, encoded.c_str(), -1, SQLITE_TRANSIENT) !=
          SQLITE_OK ||
      sqlite3_bind_text(update, 2, hash.c_str(), -1, SQLITE_TRANSIENT) !=
          SQLITE_OK ||
      sqlite3_bind_int64(update, 3, revision) != SQLITE_OK ||
      sqlite3_step(update) != SQLITE_DONE) {
    const std::string error = sqlite3_errmsg(database);
    sqlite3_finalize(update);
    sqlite3_close(database);
    throw std::runtime_error(error);
  }
  sqlite3_finalize(update);
  sqlite3_close(database);
}

void downgradeOutboxSchemaForMigrationTest(const std::string& path,
                                            int legacy_version) {
  if (legacy_version != 1 && legacy_version != 2) {
    throw std::invalid_argument("legacy schema must be v1 or v2");
  }
  std::string sql = R"sql(
PRAGMA foreign_keys = OFF;
BEGIN IMMEDIATE;
DROP INDEX IF EXISTS map_checkpoint_manifest_alignment;
DROP TABLE IF EXISTS map_checkpoint_manifest;
DROP INDEX IF EXISTS artifact_origins_revision;
DROP TABLE IF EXISTS artifact_origins;
DROP INDEX IF EXISTS outbox_lease_order;
ALTER TABLE outbox_tasks RENAME TO outbox_tasks_v3_seed;
CREATE TABLE outbox_tasks(
  task_id TEXT PRIMARY KEY,
  dedupe_key TEXT NOT NULL UNIQUE,
  task_type TEXT NOT NULL,
  payload TEXT NOT NULL,
  scene_revision INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('pending', 'leased', 'completed')),
  not_before_unix_ms INTEGER NOT NULL,
  lease_owner TEXT NOT NULL DEFAULT '',
  lease_until_unix_ms INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
  last_error TEXT NOT NULL DEFAULT '',
  completed_at_unix_ms INTEGER NOT NULL DEFAULT 0,
  completed_by TEXT NOT NULL DEFAULT '',
  FOREIGN KEY(scene_revision) REFERENCES scene_revisions(revision)
);
INSERT INTO outbox_tasks(
  task_id, dedupe_key, task_type, payload, scene_revision, state,
  not_before_unix_ms, lease_owner, lease_until_unix_ms, attempts,
  last_error, completed_at_unix_ms, completed_by)
SELECT task_id, dedupe_key, task_type, payload, scene_revision, state,
       not_before_unix_ms, lease_owner, lease_until_unix_ms, attempts,
       last_error, completed_at_unix_ms, completed_by
FROM outbox_tasks_v3_seed;
DROP TABLE outbox_tasks_v3_seed;
CREATE INDEX outbox_lease_order
  ON outbox_tasks(task_type, state, not_before_unix_ms,
                  lease_until_unix_ms, scene_revision, task_id);
)sql";
  if (legacy_version == 1) {
    sql += R"sql(
DROP INDEX IF EXISTS embedding_records_namespace;
DROP TABLE IF EXISTS embedding_records;
DELETE FROM schema_migrations WHERE version >= 2;
PRAGMA user_version = 1;
)sql";
  } else {
    sql += R"sql(
DELETE FROM schema_migrations WHERE version >= 3;
PRAGMA user_version = 2;
)sql";
  }
  sql += "COMMIT;";
  sqliteExecute(path, sql.c_str());
}

void expectLegacySchemaMigratesWithoutTaskLoss(int legacy_version) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(29, "storage bin")})});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  auto task = [&loaded](std::string id) {
    DurableTaskSpec value;
    value.task_id = std::move(id);
    value.dedupe_key = value.task_id;
    value.task_type = "migration-task";
    value.payload = "payload:" + value.task_id;
    value.scene_revision = loaded.revision;
    value.not_before_unix_ms = 0;
    return value;
  };
  const DurableTaskSpec completed = task("a-completed");
  const DurableTaskSpec leased = task("b-leased");
  const DurableTaskSpec pending = task("c-pending");

  DurableEmbeddingRecord embedding;
  embedding.object_id = 29;
  embedding.document_hash = std::string(64, 'e');
  embedding.model_id = "migration-embedding";
  embedding.dimension = 3;
  embedding.vector = {0.25f, -0.5f, 1.0f};
  embedding.created_scene_revision = loaded.revision;

  {
    SceneStore seed(database.path());
    ASSERT_TRUE(seed.open());
    ASSERT_TRUE(seed.enqueueCommit(
        loaded.snapshot, {completed, leased, pending}));
    ASSERT_TRUE(seed.flush());
    TaskLeaseResult first =
        seed.leaseNextTask("migration-task", "worker-complete", 100, 1000);
    ASSERT_TRUE(first.status) << first.status.error;
    ASSERT_TRUE(first.task.has_value());
    ASSERT_EQ(first.task->task.task_id, completed.task_id);
    ASSERT_TRUE(seed.completeTask(completed.task_id, "worker-complete",
                                  first.task->attempts, 110));
    TaskLeaseResult second =
        seed.leaseNextTask("migration-task", "worker-leased", 120, 1000);
    ASSERT_TRUE(second.status) << second.status.error;
    ASSERT_TRUE(second.task.has_value());
    ASSERT_EQ(second.task->task.task_id, leased.task_id);
    if (legacy_version == 2) {
      ASSERT_TRUE(seed.upsertEmbeddingRecord(embedding));
    }
    ASSERT_TRUE(seed.closeGracefully());
  }

  downgradeOutboxSchemaForMigrationTest(database.path(), legacy_version);
  ASSERT_EQ(sqliteScalar(database.path(), "PRAGMA user_version;"),
            legacy_version);

  SceneStore migrated(database.path());
  ASSERT_TRUE(migrated.open());
  EXPECT_EQ(migrated.schemaVersion(), SceneStore::kCurrentSchemaVersion);
  EXPECT_EQ(sqliteScalar(database.path(), "PRAGMA user_version;"),
            SceneStore::kCurrentSchemaVersion);
  EXPECT_EQ(sqliteScalar(database.path(),
                         "SELECT COUNT(*) FROM schema_migrations;"),
            SceneStore::kCurrentSchemaVersion);

  const TaskLookupResult completed_after =
      migrated.lookupTask(completed.task_id);
  ASSERT_TRUE(completed_after.status) << completed_after.status.error;
  ASSERT_TRUE(completed_after.task.has_value());
  EXPECT_EQ(completed_after.task->state, DurableTaskState::kCompleted);
  EXPECT_EQ(completed_after.task->attempts, 1u);
  EXPECT_EQ(completed_after.task->completed_at_unix_ms, 110);
  EXPECT_EQ(completed_after.task->completed_by, "worker-complete");
  EXPECT_EQ(completed_after.task->failed_at_unix_ms, 0);
  EXPECT_TRUE(completed_after.task->failed_by.empty());

  const TaskLookupResult leased_after = migrated.lookupTask(leased.task_id);
  ASSERT_TRUE(leased_after.status) << leased_after.status.error;
  ASSERT_TRUE(leased_after.task.has_value());
  EXPECT_EQ(leased_after.task->state, DurableTaskState::kLeased);
  EXPECT_EQ(leased_after.task->attempts, 1u);
  EXPECT_EQ(leased_after.task->lease_owner, "worker-leased");
  EXPECT_EQ(leased_after.task->lease_until_unix_ms, 1120);
  EXPECT_EQ(leased_after.task->failed_at_unix_ms, 0);
  EXPECT_TRUE(leased_after.task->failed_by.empty());

  const TaskLookupResult pending_after = migrated.lookupTask(pending.task_id);
  ASSERT_TRUE(pending_after.status) << pending_after.status.error;
  ASSERT_TRUE(pending_after.task.has_value());
  EXPECT_EQ(pending_after.task->state, DurableTaskState::kPending);
  EXPECT_EQ(pending_after.task->attempts, 0u);

  if (legacy_version == 2) {
    const EmbeddingRecordLookupResult restored = migrated.getEmbeddingRecord(
        embedding.object_id, embedding.document_hash, embedding.model_id,
        embedding.dimension);
    ASSERT_TRUE(restored.status) << restored.status.error;
    ASSERT_TRUE(restored.record.has_value());
    EXPECT_EQ(restored.record->vector, embedding.vector);
    EXPECT_EQ(restored.record->created_scene_revision,
              embedding.created_scene_revision);
  } else {
    const EmbeddingRecordListResult empty = migrated.listEmbeddingRecords();
    ASSERT_TRUE(empty.status) << empty.status.error;
    EXPECT_TRUE(empty.records.empty());
  }

  ASSERT_TRUE(migrated.failTask(leased.task_id, "worker-leased", 1, 200,
                                "terminal after migration"));
  const TaskLookupResult failed_after = migrated.lookupTask(leased.task_id);
  ASSERT_TRUE(failed_after.status) << failed_after.status.error;
  ASSERT_TRUE(failed_after.task.has_value());
  EXPECT_EQ(failed_after.task->state, DurableTaskState::kFailed);
  EXPECT_EQ(failed_after.task->failed_by, "worker-leased");
}

TEST(SceneStore, OrderedCommitsAdvanceLatestThenDurableWatermarks) {
  TemporaryDatabase database;
  SceneStore store(database.path());
  const SceneStoreStatus opened = store.open();
  ASSERT_TRUE(opened) << opened.error;
  EXPECT_EQ(store.schemaVersion(), SceneStore::kCurrentSchemaVersion);

  ReducerCore reducer;
  const SceneApplyResult load = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(1, "chair")})});
  ASSERT_TRUE(load.committedRevision()) << load.reason;
  const SceneSnapshot revision_one = load.snapshot;
  const SceneSnapshot revision_two = annotate(&reducer, 1, "reading chair");
  const SceneSnapshot revision_three = annotate(&reducer, 1, "desk chair");

  ASSERT_TRUE(store.enqueueCommit(revision_one));
  SceneStoreWatermarks watermarks = store.watermarks();
  EXPECT_EQ(watermarks.latest_scene_revision, 1u);
  EXPECT_EQ(watermarks.durable_scene_revision, 0u);
  EXPECT_EQ(watermarks.pending_commits, 1u);

  const SceneStoreStatus out_of_order = store.enqueueCommit(revision_three);
  EXPECT_FALSE(out_of_order);
  EXPECT_NE(out_of_order.error.find("contiguous"), std::string::npos);
  EXPECT_EQ(store.watermarks().latest_scene_revision, 1u);

  ASSERT_TRUE(store.enqueueCommit(revision_two));
  ASSERT_TRUE(store.enqueueCommit(revision_three));
  ASSERT_TRUE(store.flush());
  watermarks = store.watermarks();
  EXPECT_EQ(watermarks.latest_scene_revision, 3u);
  EXPECT_EQ(watermarks.durable_scene_revision, 3u);
  EXPECT_EQ(watermarks.pending_commits, 0u);

  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);
  EXPECT_EQ(restored.snapshot.revision(), 3u);
  EXPECT_EQ(restored.snapshot.durableRevision(), 3u);
  ASSERT_TRUE(restored.snapshot.findObject(1));
  EXPECT_EQ(restored.snapshot.findObject(1)->annotation->label_override,
            "desk chair");

  const SceneRestoreResult aligned_prefix = store.restoreAt(2);
  ASSERT_TRUE(aligned_prefix.status) << aligned_prefix.status.error;
  ASSERT_TRUE(aligned_prefix.found);
  EXPECT_EQ(aligned_prefix.snapshot.revision(), 2u);
  EXPECT_EQ(aligned_prefix.snapshot.findObject(1)->annotation->label_override,
            "reading chair");
  const SceneRestoreResult empty_prefix = store.restoreAt(0);
  EXPECT_TRUE(empty_prefix.status) << empty_prefix.status.error;
  EXPECT_FALSE(empty_prefix.found);
  const SceneRestoreResult future_prefix = store.restoreAt(4);
  EXPECT_FALSE(future_prefix.status);
  EXPECT_FALSE(future_prefix.found);

  ASSERT_TRUE(store.closeGracefully());
  EXPECT_EQ(sqliteScalar(database.path(), "PRAGMA user_version;"),
            SceneStore::kCurrentSchemaVersion);
  EXPECT_EQ(sqliteScalar(database.path(),
                         "SELECT COUNT(*) FROM schema_migrations;"),
            SceneStore::kCurrentSchemaVersion);
}

TEST(SceneStore, PositivePresenceViewpointsSurviveRestore) {
  TemporaryDatabase database;
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());

  ReducerCore reducer;
  UpsertTrackMutation create;
  create.track.track_id = 42;
  create.track.state = InstanceTrackState::kStable;
  create.track.label = "backpack";
  create.track.positive_evidence_timestamps_ns = {100, 200};
  PositivePresenceEvidenceSample first;
  first.time_ns = 100;
  first.camera_id = "head";
  first.camera_position_world = {0.0f, 0.0f, 0.0f};
  PositivePresenceEvidenceSample second = first;
  second.time_ns = 200;
  second.camera_position_world = {0.12f, 0.0f, 0.0f};
  create.track.positive_presence_evidence_history = {first, second};
  const SceneApplyResult applied =
      reducer.apply(SceneCommand{storeMutation(create)});
  ASSERT_TRUE(applied.committedRevision()) << applied.reason;
  ASSERT_TRUE(store.enqueueCommit(applied.snapshot));
  ASSERT_TRUE(store.flush());

  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);
  const auto track = restored.snapshot.tracks().find(42);
  ASSERT_NE(track, restored.snapshot.tracks().end());
  ASSERT_TRUE(track->second);
  EXPECT_EQ(track->second->positive_presence_evidence_history,
            create.track.positive_presence_evidence_history);
  const SceneObjectPtr object =
      restored.snapshot.findObject(track->second->object_id);
  ASSERT_TRUE(object);
  ASSERT_TRUE(object->lifecycle);
  EXPECT_EQ(object->lifecycle->positive_presence_evidence_history,
            create.track.positive_presence_evidence_history);
}

TEST(SceneStore, MapCheckpointManifestIsAlignedIdempotentAndRestorable) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(31, "workbench")})});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;
  const SceneSnapshot revision_one = loaded.snapshot;
  const SceneSnapshot revision_two =
      annotate(&reducer, 31, "durable workbench");

  MapCheckpointManifest manifest;
  manifest.backend = "nvblox";
  manifest.checkpoint_path = "/tmp/roomie-map-checkpoint-1.nvblox";
  manifest.world_frame = "world";
  manifest.config_fingerprint = "fnv1a64:test-config";
  manifest.map_epoch = RunId{0x1234U, 0x5678U};
  manifest.map_revision = 77;
  manifest.integrated_through_ns = 123456789;
  manifest.aligned_scene_revision = 1;
  manifest.file_size_bytes = 4096;
  manifest.content_hash = "fnv1a64:0123456789abcdef";
  manifest.created_at_unix_ms = 1000;

  {
    SceneStore store(database.path());
    ASSERT_TRUE(store.open());
    ASSERT_TRUE(store.enqueueCommit(revision_one));

    const SceneStoreStatus premature =
        store.publishMapCheckpoint(manifest);
    EXPECT_FALSE(premature);
    EXPECT_NE(premature.error.find("non-durable"), std::string::npos);

    ASSERT_TRUE(store.flush());
    ASSERT_TRUE(store.publishMapCheckpoint(manifest));
    ASSERT_TRUE(store.publishMapCheckpoint(manifest));

    MapCheckpointLookupResult lookup = store.latestMapCheckpoint();
    ASSERT_TRUE(lookup.status) << lookup.status.error;
    ASSERT_TRUE(lookup.manifest.has_value());
    EXPECT_EQ(lookup.manifest->backend, manifest.backend);
    EXPECT_EQ(lookup.manifest->checkpoint_path,
              manifest.checkpoint_path);
    EXPECT_EQ(lookup.manifest->world_frame, manifest.world_frame);
    EXPECT_EQ(lookup.manifest->config_fingerprint,
              manifest.config_fingerprint);
    EXPECT_EQ(lookup.manifest->map_epoch, manifest.map_epoch);
    EXPECT_EQ(lookup.manifest->map_revision, manifest.map_revision);
    EXPECT_EQ(lookup.manifest->integrated_through_ns,
              manifest.integrated_through_ns);
    EXPECT_EQ(lookup.manifest->aligned_scene_revision, 1U);
    EXPECT_EQ(lookup.manifest->file_size_bytes,
              manifest.file_size_bytes);
    EXPECT_EQ(lookup.manifest->content_hash, manifest.content_hash);

    ASSERT_TRUE(store.enqueueCommit(revision_two));
    ASSERT_TRUE(store.flush());
    // Replaying an already-published immutable row stays idempotent even
    // after the durable scene watermark advances.
    ASSERT_TRUE(store.publishMapCheckpoint(manifest));

    MapCheckpointManifest stale_new_file = manifest;
    stale_new_file.checkpoint_path =
        "/tmp/roomie-map-checkpoint-stale.nvblox";
    const SceneStoreStatus stale =
        store.publishMapCheckpoint(stale_new_file);
    EXPECT_FALSE(stale);
    EXPECT_NE(stale.error.find("current durable"), std::string::npos);

    MapCheckpointManifest conflicting = manifest;
    conflicting.content_hash = "fnv1a64:ffffffffffffffff";
    const SceneStoreStatus conflict =
        store.publishMapCheckpoint(conflicting);
    EXPECT_FALSE(conflict);
    EXPECT_NE(conflict.error.find("reused"), std::string::npos);
    ASSERT_TRUE(store.closeGracefully());
  }

  SceneStore reopened(database.path());
  ASSERT_TRUE(reopened.open());
  const MapCheckpointLookupResult restored =
      reopened.latestMapCheckpoint();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.manifest.has_value());
  EXPECT_EQ(restored.manifest->checkpoint_path,
            manifest.checkpoint_path);
  EXPECT_EQ(restored.manifest->aligned_scene_revision, 1U);
  EXPECT_EQ(reopened.watermarks().durable_scene_revision, 2U);

  // Explicit seed recovery may keep the scene branch point unchanged while
  // still invalidating every checkpoint from the previous map lineage.
  ASSERT_TRUE(reopened.rewindTo(
      2, /*retain_aligned_map_checkpoints=*/false));
  const MapCheckpointLookupResult discarded =
      reopened.latestMapCheckpoint();
  ASSERT_TRUE(discarded.status) << discarded.status.error;
  EXPECT_FALSE(discarded.manifest.has_value());
  EXPECT_EQ(reopened.restoreLatest().snapshot.revision(), 2U);
}

TEST(SceneStore,
     CoordinatedRewindBranchesAtCheckpointAndDropsEveryDurableSuffix) {
  TemporaryDatabase database;
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());

  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(41, "tool chest")})});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;
  std::vector<SceneSnapshot> revisions{loaded.snapshot};
  while (revisions.size() < 65U) {
    revisions.push_back(geometryOnlyStoreRevision(
        revisions.back(), 41, static_cast<float>(revisions.size())));
  }

  for (std::size_t index = 0; index < 63U; ++index) {
    ASSERT_TRUE(store.enqueueCommit(revisions[index]));
  }
  ASSERT_TRUE(store.flush());
  for (SceneRevision revision : {SceneRevision{63}, SceneRevision{64},
                                 SceneRevision{65}}) {
    if (revision > store.watermarks().durable_scene_revision) {
      ASSERT_TRUE(store.enqueueCommit(revisions[revision - 1]));
      ASSERT_TRUE(store.flush());
    }
    const SceneRestoreResult restored = store.restoreAt(revision);
    ASSERT_TRUE(restored.status) << restored.status.error;
    ASSERT_TRUE(restored.found);
    EXPECT_EQ(restored.snapshot.revision(), revision);
  }

  DurableTaskSpec task64{"task-64", "task-64", "test", "payload-64",
                         64, 0};
  DurableTaskSpec task65{"task-65", "task-65", "test", "payload-65",
                         65, 0};
  // Rewind the setup to 63 so the boundary tasks are admitted atomically with
  // their owning revisions.
  ASSERT_TRUE(store.rewindTo(63));
  ASSERT_TRUE(store.enqueueCommit(revisions[63], {task64}));
  ASSERT_TRUE(store.flush());

  MapCheckpointManifest manifest;
  manifest.backend = "nvblox";
  manifest.checkpoint_path = "/tmp/roomie-rewind-64.nvblox";
  manifest.world_frame = "world";
  manifest.config_fingerprint = "fnv1a64:rewind-test";
  manifest.map_epoch = RunId{0x1111U, 0x2222U};
  manifest.map_revision = 64;
  manifest.integrated_through_ns = 64000;
  manifest.aligned_scene_revision = 64;
  manifest.file_size_bytes = 1024;
  manifest.content_hash = "fnv1a64:checkpoint";
  manifest.created_at_unix_ms = 1000;
  ASSERT_TRUE(store.publishMapCheckpoint(manifest));

  DurableArtifactOrigin suffix_origin{41, 65, 1234,
                                       ArtifactPriority::kBulk};
  ASSERT_TRUE(store.enqueueCommit(revisions[64], {task65}, {suffix_origin}));
  ASSERT_TRUE(store.flush());
  DurableEmbeddingRecord suffix_embedding;
  suffix_embedding.object_id = 41;
  suffix_embedding.document_hash = std::string(64, 'f');
  suffix_embedding.model_id = "rewind-model";
  suffix_embedding.dimension = 2;
  suffix_embedding.vector = {0.25f, 0.75f};
  suffix_embedding.created_scene_revision = 65;
  ASSERT_TRUE(store.upsertEmbeddingRecord(suffix_embedding));

  ASSERT_TRUE(store.rewindTo(64));
  EXPECT_EQ(store.watermarks().durable_scene_revision, 64U);
  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);
  EXPECT_EQ(restored.snapshot.revision(), 64U);
  EXPECT_TRUE(store.lookupTask(task64.task_id).task.has_value());
  EXPECT_FALSE(store.lookupTask(task65.task_id).task.has_value());
  EXPECT_TRUE(store.listArtifactOrigins().origins.empty());
  EXPECT_FALSE(store.getEmbeddingRecord(
      suffix_embedding.object_id, suffix_embedding.document_hash,
      suffix_embedding.model_id, suffix_embedding.dimension).record.has_value());
  const MapCheckpointLookupResult checkpoint = store.latestMapCheckpoint();
  ASSERT_TRUE(checkpoint.status) << checkpoint.status.error;
  ASSERT_TRUE(checkpoint.manifest);
  EXPECT_EQ(checkpoint.manifest->aligned_scene_revision, 64U);

  // A new revision 65 can now form a different contiguous branch.
  ASSERT_TRUE(store.enqueueCommit(revisions[64]));
  ASSERT_TRUE(store.closeGracefully());
  SceneStore reopened(database.path());
  ASSERT_TRUE(reopened.open());
  EXPECT_EQ(reopened.restoreLatest().snapshot.revision(), 65U);
}

TEST(SceneStore, MigratesV1OutboxThroughCurrentWithoutTaskLoss) {
  expectLegacySchemaMigratesWithoutTaskLoss(1);
}

TEST(SceneStore, MigratesV2OutboxToCurrentAndPreservesEmbeddingRecords) {
  expectLegacySchemaMigratesWithoutTaskLoss(2);
}

TEST(SceneStore, MigratesV3ByAddingCheckpointManifestWithoutSceneLoss) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(32, "cabinet")})});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;
  {
    SceneStore seed(database.path());
    ASSERT_TRUE(seed.open());
    ASSERT_TRUE(seed.enqueueCommit(loaded.snapshot));
    ASSERT_TRUE(seed.closeGracefully());
  }

  sqliteExecute(
      database.path(),
      "PRAGMA foreign_keys = OFF; BEGIN IMMEDIATE; "
      "DROP INDEX map_checkpoint_manifest_alignment; "
      "DROP TABLE map_checkpoint_manifest; "
      "DROP INDEX artifact_origins_revision; "
      "DROP TABLE artifact_origins; "
      "DELETE FROM schema_migrations WHERE version >= 4; "
      "PRAGMA user_version = 3; COMMIT;");

  SceneStore migrated(database.path());
  ASSERT_TRUE(migrated.open());
  EXPECT_EQ(migrated.schemaVersion(), SceneStore::kCurrentSchemaVersion);
  const SceneRestoreResult restored = migrated.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);
  EXPECT_EQ(restored.snapshot.revision(), 1U);
  ASSERT_TRUE(restored.snapshot.findObject(32));
  const MapCheckpointLookupResult no_checkpoint =
      migrated.latestMapCheckpoint();
  ASSERT_TRUE(no_checkpoint.status) << no_checkpoint.status.error;
  EXPECT_FALSE(no_checkpoint.manifest.has_value());
}

TEST(SceneStore, RestoreIgnoresUnflushedSuffixAfterCrashStyleClose) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult load = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(4, "lamp")})});
  ASSERT_TRUE(load.committedRevision());
  const SceneSnapshot revision_one = load.snapshot;
  const SceneSnapshot revision_two = annotate(&reducer, 4, "floor lamp");

  {
    SceneStore store(database.path());
    ASSERT_TRUE(store.open());
    ASSERT_TRUE(store.enqueueCommit(revision_one));
    ASSERT_TRUE(store.flush());
    ASSERT_TRUE(store.enqueueCommit(revision_two));
    EXPECT_EQ(store.watermarks().latest_scene_revision, 2u);
    EXPECT_EQ(store.watermarks().durable_scene_revision, 1u);
    // Destructor intentionally does not flush pending commits.
  }

  SceneStore reopened(database.path());
  ASSERT_TRUE(reopened.open());
  const SceneStoreWatermarks watermarks = reopened.watermarks();
  EXPECT_EQ(watermarks.latest_scene_revision, 1u);
  EXPECT_EQ(watermarks.durable_scene_revision, 1u);
  const SceneRestoreResult restored = reopened.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);
  EXPECT_EQ(restored.snapshot.revision(), 1u);
  EXPECT_FALSE(
      restored.snapshot.findObject(4)->annotation->label_override.has_value());
}

TEST(SceneStore, DurablePendingDescriptionBecomesCurrentOnRestore) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(12, "mug")})});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  ApplySnapshotSetCommand snapshots;
  snapshots.dependency = dependencyFor(*loaded.snapshot.findObject(12));
  ObjectSnapshotRef snapshot;
  snapshot.image_index = 4;
  snapshot.bbox_xyxy = {1.0f, 2.0f, 8.0f, 9.0f};
  snapshots.snapshots.push_back(snapshot);
  snapshots.snapshot_set_hash = "snapshot-set-v1";
  const SceneApplyResult snapshot_committed =
      reducer.apply(SceneCommand{snapshots});
  ASSERT_TRUE(snapshot_committed.committedRevision())
      << snapshot_committed.reason;

  ApplyDescriptionArtifactCommand description;
  description.dependency =
      dependencyFor(*snapshot_committed.snapshot.findObject(12));
  description.description = "a blue ceramic mug";
  description.input_hash = "dam-input-v1";
  description.model_id = "dam-model-v1";
  description.schema_version = "roomie.dam.v1";
  description.raw_text = "```json\n{...}\n```";
  description.normalized_json =
      R"({"canonical_name":"mug","short_description":"a blue ceramic mug"})";
  description.durable_envelope_json =
      R"({"parse_path":"schema_valid"})";
  description.parse_path = "schema_valid";
  description.artifact_slo.origin_created_unix_ms = 1'000;
  description.artifact_slo.due_unix_ms = 11'000;
  description.artifact_slo.priority = ArtifactPriority::kInteractive;
  description.artifact_slo.steady_clock_epoch = RunId{17, 23};
  description.artifact_slo.origin_steady_ns = 5'000'000'000LL;
  description.artifact_slo.due_steady_ns = 15'000'000'000LL;
  const SceneApplyResult described =
      reducer.apply(SceneCommand{description});
  ASSERT_TRUE(described.committedRevision()) << described.reason;
  ASSERT_TRUE(described.snapshot.findObject(12)->artifact->description.empty());
  ASSERT_EQ(described.snapshot.findObject(12)
                ->artifact->pending_description_scene_revision,
            described.revision);

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(loaded.snapshot));
  ASSERT_TRUE(store.enqueueCommit(snapshot_committed.snapshot));
  ASSERT_TRUE(store.enqueueCommit(described.snapshot));
  ASSERT_TRUE(store.gracefulFlush());

  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);
  const SceneObjectPtr object = restored.snapshot.findObject(12);
  ASSERT_TRUE(object);
  ASSERT_TRUE(object->artifact);
  EXPECT_EQ(object->artifact->description, "a blue ceramic mug");
  EXPECT_EQ(object->artifact->description_input_hash, "dam-input-v1");
  EXPECT_EQ(object->artifact->description_model_id, "dam-model-v1");
  EXPECT_EQ(object->artifact->description_schema_version, "roomie.dam.v1");
  EXPECT_EQ(object->artifact->description_raw_text,
            "```json\n{...}\n```");
  EXPECT_EQ(object->artifact->description_normalized_json,
            description.normalized_json);
  EXPECT_EQ(object->artifact->description_durable_envelope_json,
            description.durable_envelope_json);
  EXPECT_EQ(object->artifact->description_parse_path, "schema_valid");
  EXPECT_EQ(object->artifact->description_slo, description.artifact_slo);
  EXPECT_FALSE(object->artifact->description_stale);
  EXPECT_EQ(object->artifact->pending_description_scene_revision, 0U);
  EXPECT_TRUE(object->artifact->pending_description.empty());
}

TEST(SceneStore, CanonicalRoomsTypedRelationsAndWarningsSurviveRestore) {
  TemporaryDatabase database;
  LoadSceneCommand command =
      storeLoadCommand({makeStoreNode(1, "chair")});
  RoomNode room;
  room.room_id = 9;
  room.revision = 4;
  room.label = "office";
  room.color = "blue";
  room.center_world = Eigen::Vector3f(1.0f, 1.0f, 1.5f);
  room.size_m = Eigen::Vector3f(4.0f, 5.0f, 3.0f);
  room.min_xy = {-1.0f, -1.5f};
  room.max_xy = {3.0f, 3.5f};
  room.has_xy_bounds = true;
  room.height_m = 3.0f;
  room.attributes["floor"] = "2";
  command.graph.rooms.push_back(room);
  command.graph.furniture.push_back(FurnitureRole{1, 5, "chair"});
  ObjectRelation containment;
  setRelationEndpoints(
      &containment,
      SceneEntityRef{SceneEntityType::kObject, 1},
      SceneEntityRef{SceneEntityType::kRoom, 9});
  containment.relation_type = "inside";
  containment.confidence = 0.95f;
  containment.description = "derived containment";
  containment.revision = 7;
  containment.derived = true;
  command.graph.relations.push_back(containment);
  command.graph.import_warnings.push_back("legacy envelope conflict");
  command.recent_observation_frames.push_back(
      FrameKey{RunId{100, 200}, 17});
  command.observation_watermarks.push_back(
      ObservationRunWatermark{RunId{100, 200}, 23});

  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{command});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;
  const SceneGraphMetadata expected_graph = loaded.snapshot.graphMetadata();
  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(loaded.snapshot));
  ASSERT_TRUE(store.gracefulFlush());

  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);
  const SceneGraphMetadata& graph = restored.snapshot.graphMetadata();
  ASSERT_EQ(graph.rooms.size(), 1u);
  EXPECT_EQ(graph.rooms.front().room_id, 9);
  EXPECT_EQ(graph.rooms.front().revision, 4u);
  EXPECT_EQ(graph.rooms.front().attributes.at("floor"), "2");
  ASSERT_EQ(graph.furniture.size(), 1U);
  EXPECT_EQ(graph.furniture.front().object_id, 1);
  EXPECT_EQ(graph.furniture.front().revision, 5U);
  EXPECT_EQ(graph.furniture.front().classification_label, "chair");
  ASSERT_EQ(graph.relations.size(), expected_graph.relations.size());
  const auto saved_relation = std::find_if(
      graph.relations.begin(), graph.relations.end(),
      [](const ObjectRelation& relation) {
        return relation.description == "derived containment";
      });
  ASSERT_NE(saved_relation, graph.relations.end());
  EXPECT_EQ(relationSource(*saved_relation).type,
            SceneEntityType::kObject);
  EXPECT_EQ(relationSource(*saved_relation).id, 1);
  EXPECT_EQ(relationTarget(*saved_relation).type,
            SceneEntityType::kRoom);
  EXPECT_EQ(relationTarget(*saved_relation).id, 9);
  EXPECT_EQ(saved_relation->revision, 7u);
  EXPECT_TRUE(saved_relation->derived);
  ASSERT_EQ(graph.import_warnings.size(), 1u);
  EXPECT_EQ(graph.import_warnings.front(), "legacy envelope conflict");
  ASSERT_EQ(restored.snapshot.statePtr()->recent_observation_frames.size(),
            1u);
  EXPECT_EQ(restored.snapshot.statePtr()->recent_observation_frames.front(),
            (FrameKey{RunId{100, 200}, 17}));
  ASSERT_EQ(restored.snapshot.statePtr()->observation_watermarks.size(), 1u);
  EXPECT_EQ(restored.snapshot.statePtr()
                ->observation_watermarks.front()
                .highest_frame_id,
            23u);
}

TEST(SceneStore, AliasesTombstonesCurrentViewAndHistoryRestoreTogether) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{storeLoadCommand(
      {makeStoreNode(1, "cup"), makeStoreNode(2, "cup")})});
  ASSERT_TRUE(loaded.committedRevision());

  MergeObjectsMutation merge;
  merge.retired_object_id = 1;
  merge.canonical_object_id = 2;
  const SceneApplyResult merged =
      reducer.apply(SceneCommand{storeMutation(merge)});
  ASSERT_TRUE(merged.committedRevision()) << merged.reason;

  TombstoneObjectMutation tombstone;
  tombstone.object_id = 1;
  tombstone.reason = "removed";
  const SceneApplyResult deleted =
      reducer.apply(SceneCommand{storeMutation(tombstone)});
  ASSERT_TRUE(deleted.committedRevision()) << deleted.reason;

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(loaded.snapshot));
  ASSERT_TRUE(store.enqueueCommit(merged.snapshot));
  ASSERT_TRUE(store.enqueueCommit(deleted.snapshot));
  ASSERT_TRUE(store.gracefulFlush());

  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);
  EXPECT_EQ(restored.snapshot.resolveCanonicalId(1),
            std::optional<SceneObjectId>(2));
  EXPECT_TRUE(restored.snapshot.isTombstoned(1));
  EXPECT_TRUE(restored.snapshot.isTombstoned(2));
  EXPECT_TRUE(restored.snapshot.objects().empty());

  ASSERT_TRUE(store.closeGracefully());
  EXPECT_EQ(sqliteScalar(database.path(),
                         "SELECT COUNT(*) FROM object_aliases;"),
            1);
  EXPECT_EQ(sqliteScalar(database.path(),
                         "SELECT COUNT(*) FROM object_tombstones;"),
            1);
  EXPECT_EQ(sqliteScalar(database.path(),
                         "SELECT COUNT(*) FROM objects_current;"),
            0);
  EXPECT_EQ(sqliteScalar(database.path(),
                         "SELECT COUNT(*) FROM object_history;"),
            5);
}

TEST(SceneStore, GeometryOnlyCommitsUseBoundedDeltasAndSparseHistoryRows) {
  TemporaryDatabase database;
  ReducerCore reducer;
  LoadSceneCommand load;
  constexpr int kObjectCount = 40;
  constexpr int kGeometryCommits = 80;
  for (int object_id = 0; object_id < kObjectCount; ++object_id) {
    ObjectNode node =
        makeStoreNode(object_id, "fixture-object-" + std::to_string(object_id));
    load.graph.objects.push_back(std::move(node));
    load.graph.next_object_id = object_id + 1;
  }
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;
  SceneSnapshot latest = loaded.snapshot;
  const std::uint64_t initial_geometry_revision =
      latest.findExactObject(0)->geometry->revision;

  {
    SceneStore store(database.path());
    ASSERT_TRUE(store.open());
    ASSERT_TRUE(store.enqueueCommit(latest));
    for (int update = 1; update <= kGeometryCommits; ++update) {
      latest = geometryOnlyStoreRevision(latest, 0,
                                         static_cast<float>(update));
      ASSERT_TRUE(store.enqueueCommit(latest));
      if (update % 20 == 0) {
        ASSERT_TRUE(store.flush());
      }
    }
    ASSERT_TRUE(store.closeGracefully());
  }

  const int revision_count = kGeometryCommits + 1;
  EXPECT_EQ(sqliteScalar(database.path(),
                         "SELECT COUNT(*) FROM scene_revisions;"),
            revision_count);
  EXPECT_EQ(sqliteScalar(
                database.path(),
                "SELECT COUNT(*) FROM scene_revisions "
                "WHERE instr(payload, '\"format_version\":1') > 0;"),
            2);
  EXPECT_EQ(sqliteScalar(
                database.path(),
                "SELECT COUNT(*) FROM scene_revisions "
                "WHERE instr(payload, '\"format_version\":2') > 0;"),
            revision_count - 2);

  // Initial materialization writes each object once. Every later revision
  // changes exactly one object, so history growth is O(objects + revisions),
  // not O(objects * revisions).
  EXPECT_EQ(sqliteScalar(database.path(),
                         "SELECT COUNT(*) FROM object_history;"),
            kObjectCount + kGeometryCommits);
  EXPECT_EQ(sqliteScalar(database.path(),
                         "SELECT COUNT(*) FROM object_history "
                         "WHERE object_id = 0;"),
            kGeometryCommits + 1);
  EXPECT_EQ(sqliteScalar(database.path(),
                         "SELECT COUNT(*) FROM object_history "
                         "WHERE object_id = 1;"),
            1);
  EXPECT_EQ(sqliteScalar(database.path(),
                         "SELECT COUNT(*) FROM objects_current;"),
            kObjectCount);

  const int checkpoint_bytes = sqliteScalar(
      database.path(),
      "SELECT length(payload) FROM scene_revisions WHERE revision = 1;");
  const int journal_bytes = sqliteScalar(
      database.path(), "SELECT SUM(length(payload)) FROM scene_revisions;");
  EXPECT_LT(journal_bytes, checkpoint_bytes * 6);

  SceneStore reopened(database.path());
  ASSERT_TRUE(reopened.open());
  const SceneRestoreResult restored = reopened.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);
  EXPECT_EQ(restored.snapshot.revision(),
            static_cast<SceneRevision>(revision_count));
  const SceneObjectPtr restored_object = restored.snapshot.findExactObject(0);
  ASSERT_TRUE(restored_object);
  EXPECT_FLOAT_EQ(restored_object->geometry->score,
                  static_cast<float>(kGeometryCommits));
  EXPECT_EQ(restored_object->geometry->revision,
            initial_geometry_revision + kGeometryCommits);
}

TEST(SceneStore, AdmissionRejectsInvalidPayloadAndBoundsUndurableSuffix) {
  TemporaryDatabase database;
  ReducerCore reducer;
  LoadSceneCommand load = storeLoadCommand({makeStoreNode(6, "shelf")});
  load.latest_surface.map_epoch = RunId{9, 10};
  load.latest_surface.surface_revision = 4;
  load.latest_surface.source_map_revision = 4;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision());
  const SceneSnapshot revision_one = loaded.snapshot;
  const SceneSnapshot revision_two = annotate(&reducer, 6, "book shelf");
  const SceneSnapshot revision_three = annotate(&reducer, 6, "bookshelf");

  SceneStore store(database.path(), 2);
  ASSERT_TRUE(store.open());

  SceneState null_state = *revision_one.statePtr();
  SceneObjectTable null_objects = *null_state.objects;
  null_objects[6].reset();
  null_state.objects =
      std::make_shared<const SceneObjectTable>(std::move(null_objects));
  const SceneStoreStatus null_status = store.enqueueCommit(
      SceneSnapshot(std::make_shared<const SceneState>(std::move(null_state))));
  EXPECT_FALSE(null_status);

  SceneState nan_state = *revision_one.statePtr();
  SceneObjectTable nan_objects = *nan_state.objects;
  auto nan_object = std::make_shared<SceneObject>(*nan_objects.at(6));
  auto nan_geometry =
      std::make_shared<GeometryComponent>(*nan_object->geometry);
  nan_geometry->score = std::numeric_limits<float>::quiet_NaN();
  nan_object->geometry = std::move(nan_geometry);
  nan_objects[6] = std::move(nan_object);
  nan_state.objects =
      std::make_shared<const SceneObjectTable>(std::move(nan_objects));
  const SceneStoreStatus nan_status = store.enqueueCommit(
      SceneSnapshot(std::make_shared<const SceneState>(std::move(nan_state))));
  EXPECT_FALSE(nan_status);
  EXPECT_EQ(store.watermarks().latest_scene_revision, 0u);

  SceneState invalid_slo_state = *revision_one.statePtr();
  SceneObjectTable invalid_slo_objects = *invalid_slo_state.objects;
  auto invalid_slo_object =
      std::make_shared<SceneObject>(*invalid_slo_objects.at(6));
  auto invalid_artifact =
      std::make_shared<ArtifactComponent>(*invalid_slo_object->artifact);
  invalid_artifact->description_slo.due_unix_ms = -1;
  invalid_slo_object->artifact = std::move(invalid_artifact);
  invalid_slo_objects[6] = std::move(invalid_slo_object);
  invalid_slo_state.objects = std::make_shared<const SceneObjectTable>(
      std::move(invalid_slo_objects));
  const SceneStoreStatus invalid_slo_status = store.enqueueCommit(
      SceneSnapshot(std::make_shared<const SceneState>(
          std::move(invalid_slo_state))));
  EXPECT_FALSE(invalid_slo_status);
  EXPECT_NE(invalid_slo_status.error.find("SLO"), std::string::npos);

  ASSERT_TRUE(store.enqueueCommit(revision_one));
  ASSERT_TRUE(store.enqueueCommit(revision_two));
  const SceneStoreWatermarks saturated = store.watermarks();
  EXPECT_EQ(saturated.pending_commits, 2u);
  EXPECT_EQ(saturated.max_pending_commits, 2u);
  EXPECT_TRUE(saturated.backpressure_required);
  const SceneStoreStatus over_limit = store.enqueueCommit(revision_three);
  EXPECT_FALSE(over_limit);
  EXPECT_NE(over_limit.error.find("backpressure"), std::string::npos);

  ASSERT_TRUE(store.flush());
  ASSERT_TRUE(store.enqueueCommit(revision_three));
  ASSERT_TRUE(store.flush());
  const SceneRestoreResult restored = store.restoreLatest();
  ASSERT_TRUE(restored.status) << restored.status.error;
  ASSERT_TRUE(restored.found);
  // The persisted object provenance survives, while the old process's live
  // map epoch is deliberately not advertised as current after restart.
  EXPECT_FALSE(restored.snapshot.latestSurface().map_epoch.valid());
}

TEST(SceneStore, DurableTaskLeaseExpiryAndCompletionAreIdempotent) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(7, "sofa")})});
  ASSERT_TRUE(loaded.committedRevision());

  DurableTaskSpec task;
  task.task_id = "description:7:v1";
  task.dedupe_key = "description-input-hash:v1";
  task.task_type = "description";
  task.payload = R"({"object_id":7})";
  task.scene_revision = loaded.snapshot.revision();
  task.not_before_unix_ms = 1000;

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(loaded.snapshot, {task}));
  ASSERT_TRUE(store.flush());
  EXPECT_TRUE(store.ensureTask(task));

  DurableTaskSpec conflicting = task;
  conflicting.payload = "different";
  EXPECT_FALSE(store.ensureTask(conflicting));
  conflicting = task;
  conflicting.task_id = "another-id";
  EXPECT_FALSE(store.ensureTask(conflicting));

  const TaskLeaseResult first =
      store.leaseNextTask("description", "worker-a", 1000, 100);
  ASSERT_TRUE(first.status) << first.status.error;
  ASSERT_TRUE(first.task.has_value());
  EXPECT_EQ(first.task->task.task_id, task.task_id);
  EXPECT_EQ(first.task->state, DurableTaskState::kLeased);
  EXPECT_EQ(first.task->lease_owner, "worker-a");
  EXPECT_EQ(first.task->lease_until_unix_ms, 1100);
  EXPECT_EQ(first.task->attempts, 1u);

  const TaskLeaseResult before_expiry =
      store.leaseNextTask("description", "worker-b", 1099, 100);
  ASSERT_TRUE(before_expiry.status);
  EXPECT_FALSE(before_expiry.task.has_value());

  const TaskLeaseResult after_expiry =
      store.leaseNextTask("description", "worker-a", 1100, 100);
  ASSERT_TRUE(after_expiry.status) << after_expiry.status.error;
  ASSERT_TRUE(after_expiry.task.has_value());
  EXPECT_EQ(after_expiry.task->lease_owner, "worker-a");
  EXPECT_EQ(after_expiry.task->attempts, 2u);
  // Same-owner re-lease is the ABA case: the delayed attempt-1 callback must
  // not be allowed to mutate attempt 2.
  EXPECT_FALSE(store.completeTask(task.task_id, "worker-a", 1, 1110));
  EXPECT_TRUE(
      store.renewTaskLease(task.task_id, "worker-a", 2, 1110, 100));
  EXPECT_TRUE(store.completeTask(task.task_id, "worker-a", 2, 1120));
  // A completion acknowledgement may be retried after the worker loses its
  // response; completed is terminal and therefore returns success.
  EXPECT_TRUE(store.completeTask(task.task_id, "worker-a", 2, 1130));

  const TaskLookupResult lookup = store.lookupTask(task.task_id);
  ASSERT_TRUE(lookup.status);
  ASSERT_TRUE(lookup.task.has_value());
  EXPECT_EQ(lookup.task->state, DurableTaskState::kCompleted);
  EXPECT_EQ(lookup.task->attempts, 2u);
  const TaskLeaseResult none =
      store.leaseNextTask("description", "worker-d", 2000, 100);
  ASSERT_TRUE(none.status);
  EXPECT_FALSE(none.task.has_value());

  ASSERT_TRUE(store.closeGracefully());
  SceneStore reopened(database.path());
  ASSERT_TRUE(reopened.open());
  const TaskLookupResult durable = reopened.lookupTask(task.task_id);
  ASSERT_TRUE(durable.status);
  ASSERT_TRUE(durable.task.has_value());
  EXPECT_EQ(durable.task->state, DurableTaskState::kCompleted);
  EXPECT_EQ(durable.task->attempts, 2u);
}

TEST(SceneStore, DurableTaskFailureIsFencedIdempotentAndRestartTerminal) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(17, "cabinet")})});
  ASSERT_TRUE(loaded.committedRevision());

  DurableTaskSpec task;
  task.task_id = "description:17:terminal";
  task.dedupe_key = task.task_id;
  task.task_type = "description";
  task.payload = R"({"object_id":17})";
  task.scene_revision = loaded.revision;
  task.not_before_unix_ms = 1000;

  {
    SceneStore store(database.path());
    ASSERT_TRUE(store.open());
    ASSERT_TRUE(store.enqueueCommit(loaded.snapshot, {task}));
    ASSERT_TRUE(store.flush());
    const TaskLeaseResult leased =
        store.leaseNextTask("description", "worker-a", 1000, 100);
    ASSERT_TRUE(leased.status) << leased.status.error;
    ASSERT_TRUE(leased.task.has_value());
    ASSERT_EQ(leased.task->attempts, 1u);

    EXPECT_FALSE(store.failTask(task.task_id, "worker-b", 1, 1050,
                                "wrong owner"));
    EXPECT_FALSE(store.failTask(task.task_id, "worker-a", 2, 1050,
                                "wrong attempt"));
    EXPECT_FALSE(store.failTask(task.task_id, "worker-a", 1, 1100,
                                "expired"));
    ASSERT_TRUE(store.failTask(task.task_id, "worker-a", 1, 1050,
                               "unsupported model schema"));
    // Lost acknowledgement replay from the exact same attempt is idempotent;
    // it cannot rewrite the original terminal provenance.
    EXPECT_TRUE(store.failTask(task.task_id, "worker-a", 1, 5000,
                               "different replay text"));
    EXPECT_FALSE(store.failTask(task.task_id, "worker-b", 1, 1050,
                                "different owner"));
    EXPECT_FALSE(store.completeTask(task.task_id, "worker-a", 1, 1060));
    EXPECT_FALSE(store.retryTask(task.task_id, "worker-a", 1, 1060,
                                 "must remain terminal"));
    EXPECT_FALSE(
        store.renewTaskLease(task.task_id, "worker-a", 1, 1060, 100));

    const TaskLookupResult failed = store.lookupTask(task.task_id);
    ASSERT_TRUE(failed.status) << failed.status.error;
    ASSERT_TRUE(failed.task.has_value());
    EXPECT_EQ(failed.task->state, DurableTaskState::kFailed);
    EXPECT_EQ(failed.task->attempts, 1u);
    EXPECT_EQ(failed.task->last_error, "unsupported model schema");
    EXPECT_EQ(failed.task->failed_at_unix_ms, 1050);
    EXPECT_EQ(failed.task->failed_by, "worker-a");
    EXPECT_EQ(failed.task->completed_at_unix_ms, 0);
    EXPECT_TRUE(failed.task->completed_by.empty());
    EXPECT_TRUE(failed.task->lease_owner.empty());
    EXPECT_EQ(failed.task->lease_until_unix_ms, 0);

    const TaskLeaseResult none =
        store.leaseNextTask("description", "worker-c", 10'000, 100);
    ASSERT_TRUE(none.status) << none.status.error;
    EXPECT_FALSE(none.task.has_value());
    ASSERT_TRUE(store.closeGracefully());
  }

  SceneStore reopened(database.path());
  ASSERT_TRUE(reopened.open());
  const TaskLookupResult durable = reopened.lookupTask(task.task_id);
  ASSERT_TRUE(durable.status) << durable.status.error;
  ASSERT_TRUE(durable.task.has_value());
  EXPECT_EQ(durable.task->state, DurableTaskState::kFailed);
  EXPECT_EQ(durable.task->failed_at_unix_ms, 1050);
  EXPECT_EQ(durable.task->failed_by, "worker-a");
  EXPECT_TRUE(reopened.failTask(task.task_id, "worker-a", 1, 20'000,
                                "replayed after restart"));
  EXPECT_FALSE(reopened.failTask(task.task_id, "worker-a", 2, 20'000,
                                 "wrong attempt after restart"));
}

TEST(SceneStore, RetryRespectsNotBeforeAndTasksCannotLeadDurability) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(8, "desk")})});
  ASSERT_TRUE(loaded.committedRevision());
  const SceneSnapshot revision_two = annotate(&reducer, 8, "work desk");

  SceneStore store(database.path());
  ASSERT_TRUE(store.open());
  ASSERT_TRUE(store.enqueueCommit(loaded.snapshot));
  ASSERT_TRUE(store.flush());
  ASSERT_TRUE(store.enqueueCommit(revision_two));

  DurableTaskSpec future;
  future.task_id = "snapshot:8:v2";
  future.dedupe_key = "snapshot-input:v2";
  future.task_type = "snapshot";
  future.payload = "payload";
  future.scene_revision = 2;
  EXPECT_FALSE(store.ensureTask(future));

  ASSERT_TRUE(store.flush());
  ASSERT_TRUE(store.ensureTask(future));
  TaskLeaseResult leased =
      store.leaseNextTask("snapshot", "worker-a", 10, 50);
  ASSERT_TRUE(leased.status);
  ASSERT_TRUE(leased.task.has_value());
  ASSERT_TRUE(store.retryTask(future.task_id, "worker-a",
                              leased.task->attempts, 100, "transient"));

  TaskLeaseResult too_early =
      store.leaseNextTask("snapshot", "worker-b", 99, 50);
  ASSERT_TRUE(too_early.status);
  EXPECT_FALSE(too_early.task.has_value());
  leased = store.leaseNextTask("snapshot", "worker-b", 100, 50);
  ASSERT_TRUE(leased.status);
  ASSERT_TRUE(leased.task.has_value());
  EXPECT_EQ(leased.task->attempts, 2u);
  EXPECT_EQ(leased.task->last_error, "transient");
}

TEST(SceneStore,
     EmbeddingRecordsAreRestartDurableIdempotentAndNamespaceIsolated) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(15, "book")})});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  DurableEmbeddingRecord active;
  active.object_id = 15;
  active.document_hash = std::string(64, 'a');
  active.model_id = "embedding-a";
  active.dimension = 3;
  active.vector = {1.0f, -0.0f, 2.5f};
  active.created_scene_revision = loaded.revision;

  DurableEmbeddingRecord other_model = active;
  other_model.model_id = "embedding-b";
  other_model.dimension = 2;
  other_model.vector = {4.0f, 5.0f};

  {
    SceneStore store(database.path());
    ASSERT_TRUE(store.open());
    ASSERT_TRUE(store.enqueueCommit(loaded.snapshot));
    ASSERT_TRUE(store.flush());

    EXPECT_TRUE(store.upsertEmbeddingRecord(active));
    EXPECT_TRUE(store.upsertEmbeddingRecord(active));
    EXPECT_TRUE(store.upsertEmbeddingRecord(other_model));

    DurableEmbeddingRecord conflicting = active;
    conflicting.vector[0] = 9.0f;
    EXPECT_FALSE(store.upsertEmbeddingRecord(conflicting));
    DurableEmbeddingRecord wrong_length = active;
    wrong_length.vector.pop_back();
    EXPECT_FALSE(store.upsertEmbeddingRecord(wrong_length));
    DurableEmbeddingRecord non_finite = active;
    non_finite.document_hash = std::string(64, 'n');
    non_finite.vector[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(store.upsertEmbeddingRecord(non_finite));
    DurableEmbeddingRecord future = active;
    future.document_hash = std::string(64, 'f');
    future.created_scene_revision = loaded.revision + 1;
    EXPECT_FALSE(store.upsertEmbeddingRecord(future));
  }

  SceneStore reopened(database.path());
  ASSERT_TRUE(reopened.open());
  const EmbeddingRecordLookupResult lookup = reopened.getEmbeddingRecord(
      active.object_id, active.document_hash, active.model_id,
      active.dimension);
  ASSERT_TRUE(lookup.status) << lookup.status.error;
  ASSERT_TRUE(lookup.record.has_value());
  EXPECT_EQ(lookup.record->object_id, active.object_id);
  EXPECT_EQ(lookup.record->document_hash, active.document_hash);
  EXPECT_EQ(lookup.record->model_id, active.model_id);
  EXPECT_EQ(lookup.record->dimension, active.dimension);
  EXPECT_EQ(lookup.record->vector,
            (std::vector<float>{1.0f, 0.0f, 2.5f}));
  EXPECT_EQ(lookup.record->created_scene_revision,
            active.created_scene_revision);

  const EmbeddingRecordListResult all = reopened.listEmbeddingRecords();
  ASSERT_TRUE(all.status) << all.status.error;
  EXPECT_EQ(all.records.size(), 2u);
  const EmbeddingRecordListResult namespace_a =
      reopened.listEmbeddingRecords("embedding-a", 3);
  ASSERT_TRUE(namespace_a.status) << namespace_a.status.error;
  ASSERT_EQ(namespace_a.records.size(), 1u);
  EXPECT_EQ(namespace_a.records.front().document_hash, active.document_hash);
  const EmbeddingRecordListResult wrong_namespace =
      reopened.listEmbeddingRecords("embedding-a", 2);
  ASSERT_TRUE(wrong_namespace.status) << wrong_namespace.status.error;
  EXPECT_TRUE(wrong_namespace.records.empty());
}

TEST(SceneStore, CorruptEmbeddingDimensionTypeRejectsReopen) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(16, "plant")})});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  {
    SceneStore store(database.path());
    ASSERT_TRUE(store.open());
    ASSERT_TRUE(store.enqueueCommit(loaded.snapshot));
    ASSERT_TRUE(store.flush());
    DurableEmbeddingRecord record;
    record.object_id = 16;
    record.document_hash = std::string(64, 'c');
    record.model_id = "embedding-a";
    record.dimension = 2;
    record.vector = {1.0f, 2.0f};
    record.created_scene_revision = loaded.revision;
    ASSERT_TRUE(store.upsertEmbeddingRecord(std::move(record)));
    ASSERT_TRUE(store.closeGracefully());
  }

  // SQLite's dynamic typing allows this value through the numeric CHECK and
  // length expression. SceneStore must reject it instead of silently coercing
  // the dimension prefix to integer 2.
  sqliteExecute(database.path(),
                "UPDATE embedding_records SET dimension = '2junk';");
  SceneStore corrupted(database.path());
  const SceneStoreStatus opened = corrupted.open();
  EXPECT_FALSE(opened);
  EXPECT_NE(opened.error.find("SQLite column types"), std::string::npos);
}

TEST(SceneStore, CorruptLegacyGraphUnsignedIntegerRejectsReopen) {
  TemporaryDatabase database;
  ReducerCore reducer;
  const SceneApplyResult loaded = reducer.apply(
      SceneCommand{storeLoadCommand({makeStoreNode(18, "shelf")})});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;

  {
    SceneStore store(database.path());
    ASSERT_TRUE(store.open());
    ASSERT_TRUE(store.enqueueCommit(loaded.snapshot));
    ASSERT_TRUE(store.gracefulFlush());
  }

  rewriteLatestScenePayload(database.path(), [](nlohmann::json& payload) {
    payload.at("graph").at("rooms").push_back(
        nlohmann::json{{"room_id", 4}, {"revision", -1}});
  });

  SceneStore corrupted(database.path());
  const SceneStoreStatus opened = corrupted.open();
  EXPECT_FALSE(opened);
  EXPECT_NE(opened.error.find("negative"), std::string::npos);
}

}  // namespace
}  // namespace roomie
