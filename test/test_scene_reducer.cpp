#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <string>
#include <variant>

#include "roomie/scene/scene_reducer.hpp"

namespace roomie {
namespace {

ObjectNode makeNode(SceneObjectId object_id, std::string label) {
  ObjectNode node;
  node.object_id = object_id;
  node.semantic_id = object_id + 100;
  node.label = std::move(label);
  node.center_world = Eigen::Vector3f(static_cast<float>(object_id), 0.0f, 1.0f);
  node.size_m = Eigen::Vector3f(0.5f, 0.6f, 0.7f);
  node.yaw_rad = 0.1f;
  node.confidence = 0.8f;
  node.active = true;
  node.publishable = true;
  node.source_track_ids.push_back(object_id + 1000);
  return node;
}

LoadSceneCommand loadCommand(std::initializer_list<ObjectNode> nodes) {
  LoadSceneCommand command;
  SceneObjectId next_id = 0;
  for (const ObjectNode& node : nodes) {
    command.graph.objects.push_back(node);
    next_id = std::max(next_id, node.object_id + 1);
  }
  command.graph.next_object_id = next_id;
  return command;
}

SurfaceStamp surfaceStamp(std::uint64_t epoch_low,
                          std::uint64_t surface_revision,
                          std::uint64_t map_revision) {
  SurfaceStamp stamp;
  stamp.map_epoch = RunId{17, epoch_low};
  stamp.surface_revision = surface_revision;
  stamp.source_map_revision = map_revision;
  return stamp;
}

ApplyObservationBatchCommand mutationBatch(ObservationMutation mutation) {
  ApplyObservationBatchCommand command;
  command.association_source = "unit_test";
  command.associated_mutations.push_back(std::move(mutation));
  return command;
}

TEST(SceneReducer, LoadAndHumanAnnotationUseComponentStructuralSharing) {
  ReducerCore reducer;
  LoadSceneCommand load = loadCommand({makeNode(1, "chair"), makeNode(2, "table")});
  load.restored_revision = 5;
  load.durable_revision = 3;
  const SceneApplyResult loaded = reducer.apply(SceneCommand{load});
  ASSERT_TRUE(loaded.committedRevision()) << loaded.reason;
  EXPECT_EQ(loaded.revision, 5u);
  EXPECT_EQ(loaded.snapshot.durableRevision(), 3u);
  ASSERT_EQ(loaded.snapshot.objects().size(), 2u);

  const SceneSnapshot before = loaded.snapshot;
  const SceneObjectPtr before_one = before.findObject(1);
  const SceneObjectPtr before_two = before.findObject(2);
  ASSERT_TRUE(before_one);
  ASSERT_TRUE(before_two);

  ApplyHumanAnnotationCommand annotation;
  annotation.object_id = 1;
  annotation.expected_identity_revision = before_one->identity->revision;
  annotation.expected_annotation_revision = before_one->annotation->revision;
  annotation.patch.label = "reading chair";
  annotation.patch.description = "human verified";
  annotation.patch.attributes["room"] = "study";
  const SceneApplyResult annotated = reducer.apply(SceneCommand{annotation});
  ASSERT_TRUE(annotated.committedRevision()) << annotated.reason;
  EXPECT_EQ(annotated.revision, 6u);

  const SceneObjectPtr after_one = annotated.snapshot.findObject(1);
  const SceneObjectPtr after_two = annotated.snapshot.findObject(2);
  ASSERT_TRUE(after_one);
  ASSERT_TRUE(after_two);
  EXPECT_NE(after_one.get(), before_one.get());
  EXPECT_NE(after_one->annotation.get(), before_one->annotation.get());
  EXPECT_EQ(after_one->geometry.get(), before_one->geometry.get());
  EXPECT_EQ(after_one->semantic.get(), before_one->semantic.get());
  EXPECT_EQ(after_two.get(), before_two.get());
  EXPECT_FALSE(before_one->annotation->label_override.has_value());
  EXPECT_EQ(after_one->annotation->label_override, "reading chair");

  const ObjectGraphSnapshot materialized =
      annotated.snapshot.materializeObjectGraph();
  ASSERT_EQ(materialized.objects.size(), 2u);
  EXPECT_EQ(materialized.objects.front().label, "reading chair");
  EXPECT_EQ(materialized.objects.front().description, "human verified");
}

TEST(SceneReducer, MergeAndTombstoneRetireIdsAndRejectLateResults) {
  ReducerCore reducer;
  ASSERT_TRUE(reducer
                  .apply(SceneCommand{loadCommand(
                      {makeNode(1, "cup"), makeNode(2, "cup")})})
                  .committedRevision());
  const SceneSnapshot initial = reducer.snapshot();
  const ObjectDependency source_dependency =
      dependencyFor(*initial.findObject(1));
  const ObjectDependency target_dependency =
      dependencyFor(*initial.findObject(2));

  MergeObjectsMutation merge;
  merge.retired_object_id = 1;
  merge.canonical_object_id = 2;
  const SceneApplyResult merged =
      reducer.apply(SceneCommand{mutationBatch(merge)});
  ASSERT_TRUE(merged.committedRevision()) << merged.reason;
  ASSERT_EQ(merged.snapshot.resolveCanonicalId(1), std::optional<SceneObjectId>(2));
  EXPECT_FALSE(merged.snapshot.findExactObject(1));
  EXPECT_TRUE(merged.snapshot.findObject(1));

  ApplyGeometryResultCommand late_source;
  late_source.dependency.object = source_dependency;
  late_source.dependency.surface = surfaceStamp(1, 4, 4);
  late_source.current_surface = late_source.dependency.surface;
  late_source.delta_overlap = DeltaOverlapVerdict::kExactSurface;
  const SceneApplyResult source_rejected =
      reducer.apply(SceneCommand{late_source});
  EXPECT_EQ(source_rejected.status, SceneApplyStatus::kRejected);
  EXPECT_NE(source_rejected.reason.find("retired alias"), std::string::npos);

  TombstoneObjectMutation tombstone;
  tombstone.object_id = 1;  // Resolves through the alias to canonical id 2.
  tombstone.reason = "human delete";
  const SceneApplyResult deleted =
      reducer.apply(SceneCommand{mutationBatch(tombstone)});
  ASSERT_TRUE(deleted.committedRevision()) << deleted.reason;
  EXPECT_TRUE(deleted.snapshot.isTombstoned(1));
  EXPECT_TRUE(deleted.snapshot.isTombstoned(2));
  EXPECT_FALSE(deleted.snapshot.findObject(2));

  ApplyGeometryResultCommand late_target;
  late_target.dependency.object = target_dependency;
  late_target.dependency.surface = surfaceStamp(1, 4, 4);
  late_target.current_surface = late_target.dependency.surface;
  late_target.delta_overlap = DeltaOverlapVerdict::kExactSurface;
  const SceneApplyResult target_rejected =
      reducer.apply(SceneCommand{late_target});
  EXPECT_EQ(target_rejected.status, SceneApplyStatus::kRejected);
  EXPECT_NE(target_rejected.reason.find("tombstoned"), std::string::npos);

  UpsertTrackMutation create;
  create.track.track_id = 99;
  create.track.object_id = -1;
  create.track.state = InstanceTrackState::kStable;
  create.track.label = "new object";
  const SceneApplyResult created =
      reducer.apply(SceneCommand{mutationBatch(create)});
  ASSERT_TRUE(created.committedRevision()) << created.reason;
  EXPECT_TRUE(created.snapshot.findObject(3));
  EXPECT_FALSE(created.snapshot.findExactObject(1));
  EXPECT_FALSE(created.snapshot.findExactObject(2));
  EXPECT_EQ(created.snapshot.nextObjectId(), 4);
}

TEST(SceneReducer, MergeCommitsDurableSnapshotUnionInSameRevision) {
  ReducerCore reducer;
  ASSERT_TRUE(reducer
                  .apply(SceneCommand{loadCommand(
                      {makeNode(1, "cup"), makeNode(2, "cup")})})
                  .committedRevision());

  ObjectSnapshotRef source_ref;
  source_ref.image_index = 11;
  source_ref.source_frame_asset_id = std::string(64, 'a');
  source_ref.evidence_hash = std::string(64, '1');
  source_ref.quality = 0.8f;
  ApplySnapshotSetCommand source_set;
  source_set.dependency = dependencyFor(*reducer.snapshot().findObject(1));
  source_set.snapshots = {source_ref};
  source_set.snapshot_set_hash = "source-set";
  ASSERT_TRUE(reducer.apply(SceneCommand{source_set}).committedRevision());

  ObjectSnapshotRef target_ref = source_ref;
  target_ref.image_index = 12;
  target_ref.source_frame_asset_id = std::string(64, 'b');
  target_ref.evidence_hash = std::string(64, '2');
  target_ref.quality = 0.9f;
  ApplySnapshotSetCommand target_set;
  target_set.dependency = dependencyFor(*reducer.snapshot().findObject(2));
  target_set.snapshots = {target_ref};
  target_set.snapshot_set_hash = "target-set";
  ASSERT_TRUE(reducer.apply(SceneCommand{target_set}).committedRevision());

  MergeObjectsMutation merge;
  merge.retired_object_id = 1;
  merge.canonical_object_id = 2;
  merge.merged_snapshots =
      std::vector<ObjectSnapshotRef>{target_ref, source_ref};
  merge.merged_snapshot_set_hash = "durable-union";
  const SceneApplyResult merged =
      reducer.apply(SceneCommand{mutationBatch(merge)});
  ASSERT_TRUE(merged.committedRevision()) << merged.reason;
  ASSERT_FALSE(merged.snapshot.findExactObject(1));
  const SceneObjectPtr canonical = merged.snapshot.findObject(2);
  ASSERT_TRUE(canonical);
  ASSERT_TRUE(canonical->artifact);
  ASSERT_EQ(canonical->artifact->snapshots.size(), 2U);
  EXPECT_EQ(canonical->artifact->snapshots[0].evidence_hash,
            target_ref.evidence_hash);
  EXPECT_EQ(canonical->artifact->snapshots[1].evidence_hash,
            source_ref.evidence_hash);
  EXPECT_EQ(canonical->artifact->snapshot_set_hash, "durable-union");
  EXPECT_TRUE(std::any_of(
      merged.events.begin(), merged.events.end(),
      [&merged](const SceneEvent& event) {
        const auto* committed = std::get_if<SnapshotSetCommitted>(&event);
        return committed && committed->revision == merged.revision &&
               committed->object_id == 2 &&
               committed->snapshot_set_hash == "durable-union";
      }));
}

TEST(SceneReducer, GeometryCasRequiresIdentityObbEpochAndSafeSurfaceAdvance) {
  ReducerCore reducer;
  LoadSceneCommand load = loadCommand({makeNode(4, "cabinet")});
  load.latest_surface = surfaceStamp(8, 10, 10);
  ASSERT_TRUE(reducer.apply(SceneCommand{load}).committedRevision());

  const SceneObjectPtr object = reducer.snapshot().findObject(4);
  ASSERT_TRUE(object);
  GeometryDependency dependency;
  dependency.object = dependencyFor(*object);
  dependency.surface = surfaceStamp(8, 10, 10);
  dependency.evaluated_blocks.push_back(Eigen::Vector3i(1, 2, 3));

  ApplyGeometryResultCommand exact;
  exact.dependency = dependency;
  exact.current_surface = dependency.surface;
  exact.delta_overlap = DeltaOverlapVerdict::kExactSurface;
  exact.result.status = InstanceGeometryStatus::kGood;
  exact.result.score = 0.9f;
  exact.result.reason = "exact";
  const SceneApplyResult exact_result = reducer.apply(SceneCommand{exact});
  ASSERT_TRUE(exact_result.committedRevision()) << exact_result.reason;
  EXPECT_FLOAT_EQ(exact_result.snapshot.findObject(4)->geometry->score, 0.9f);

  ApplyGeometryResultCommand advanced = exact;
  advanced.current_surface = surfaceStamp(8, 11, 11);
  advanced.delta_overlap = DeltaOverlapVerdict::kNoOverlap;
  advanced.result.score = 0.8f;
  advanced.result.reason = "safe delta";
  const SceneApplyResult advanced_result =
      reducer.apply(SceneCommand{advanced});
  ASSERT_TRUE(advanced_result.committedRevision()) << advanced_result.reason;
  EXPECT_FLOAT_EQ(advanced_result.snapshot.findObject(4)->geometry->score, 0.8f);

  UpsertTrackMutation stale_validation;
  stale_validation.object_id = 4;
  stale_validation.track.track_id = 1004;
  stale_validation.track.object_id = 4;
  stale_validation.track.state = InstanceTrackState::kStable;
  stale_validation.track.publishable = true;
  stale_validation.track.label = "cabinet";
  stale_validation.track.center_world =
      advanced_result.snapshot.findObject(4)->geometry->center_world;
  stale_validation.track.size_m =
      advanced_result.snapshot.findObject(4)->geometry->size_m;
  stale_validation.track.yaw_rad =
      advanced_result.snapshot.findObject(4)->geometry->yaw_rad;
  stale_validation.track.geometry_status = InstanceGeometryStatus::kUnchecked;
  stale_validation.track.geometry_score = 0.0f;
  const SceneApplyResult validation_preserved =
      reducer.apply(SceneCommand{mutationBatch(stale_validation)});
  ASSERT_TRUE(validation_preserved.accepted()) << validation_preserved.reason;
  EXPECT_FLOAT_EQ(
      validation_preserved.snapshot.findObject(4)->geometry->score, 0.8f);
  EXPECT_EQ(validation_preserved.snapshot.findObject(4)->geometry->status,
            InstanceGeometryStatus::kGood);

  const SceneApplyResult regressed = reducer.apply(SceneCommand{exact});
  EXPECT_EQ(regressed.status, SceneApplyStatus::kRejected);
  EXPECT_NE(regressed.reason.find("regressed"), std::string::npos);

  ApplyGeometryResultCommand overlap = exact;
  overlap.current_surface = surfaceStamp(8, 12, 12);
  overlap.delta_overlap = DeltaOverlapVerdict::kOverlap;
  EXPECT_EQ(reducer.apply(SceneCommand{overlap}).status,
            SceneApplyStatus::kRejected);

  ApplyGeometryResultCommand gap = overlap;
  gap.delta_overlap = DeltaOverlapVerdict::kJournalGap;
  EXPECT_EQ(reducer.apply(SceneCommand{gap}).status,
            SceneApplyStatus::kRejected);

  ApplyGeometryResultCommand reset = exact;
  reset.current_surface = surfaceStamp(9, 1, 1);
  EXPECT_EQ(reducer.apply(SceneCommand{reset}).status,
            SceneApplyStatus::kRejected);

  UpsertTrackMutation move;
  move.object_id = 4;
  move.track.track_id = 1004;
  move.track.object_id = 4;
  move.track.state = InstanceTrackState::kStable;
  move.track.publishable = true;
  move.track.label = "cabinet";
  move.track.semantic_id = 104;
  move.track.center_world = Eigen::Vector3f(9.0f, 0.0f, 1.0f);
  move.track.size_m = Eigen::Vector3f(0.5f, 0.6f, 0.7f);
  move.track.yaw_rad = 0.1f;
  ASSERT_TRUE(
      reducer.apply(SceneCommand{mutationBatch(move)}).committedRevision());
  EXPECT_FLOAT_EQ(reducer.snapshot().findObject(4)->geometry->score, 0.8f);
  const SceneApplyResult stale_obb = reducer.apply(SceneCommand{exact});
  EXPECT_EQ(stale_obb.status, SceneApplyStatus::kRejected);
  EXPECT_NE(stale_obb.reason.find("OBB dependency"), std::string::npos);
}

TEST(SceneReducer, ArtifactDependenciesAndDurabilityWatermarkAreIndependent) {
  ReducerCore reducer;
  ASSERT_TRUE(reducer
                  .apply(SceneCommand{loadCommand({makeNode(5, "lamp")})})
                  .committedRevision());
  const ObjectDependency initial =
      dependencyFor(*reducer.snapshot().findObject(5));

  ApplySnapshotSetCommand snapshots;
  snapshots.dependency = initial;
  ObjectSnapshotRef image;
  image.image_index = 3;
  image.camera_id = "front";
  snapshots.snapshots.push_back(image);
  snapshots.snapshot_set_hash = "snap-v1";
  const SceneApplyResult snapshot_result =
      reducer.apply(SceneCommand{snapshots});
  ASSERT_TRUE(snapshot_result.committedRevision()) << snapshot_result.reason;

  ApplyDescriptionArtifactCommand stale_description;
  stale_description.dependency = initial;
  stale_description.description = "a lamp";
  stale_description.input_hash = "input-v1";
  stale_description.model_id = "model";
  stale_description.schema_version = "v1";
  EXPECT_EQ(reducer.apply(SceneCommand{stale_description}).status,
            SceneApplyStatus::kRejected);

  ApplyDescriptionArtifactCommand description = stale_description;
  description.dependency =
      dependencyFor(*reducer.snapshot().findObject(5));
  const SceneApplyResult described = reducer.apply(SceneCommand{description});
  ASSERT_TRUE(described.committedRevision()) << described.reason;
  const SceneRevision latest = described.snapshot.revision();
  ASSERT_TRUE(described.snapshot.findObject(5)->artifact);
  EXPECT_TRUE(described.snapshot.findObject(5)->artifact->description.empty());
  EXPECT_EQ(
      described.snapshot.findObject(5)->artifact->pending_description,
      "a lamp");
  EXPECT_EQ(described.snapshot.findObject(5)
                ->artifact->pending_description_scene_revision,
            latest);

  const SceneApplyResult persisted_one =
      reducer.apply(SceneCommand{PersistedThroughCommand{1}});
  EXPECT_EQ(persisted_one.status, SceneApplyStatus::kMetadataUpdated);
  EXPECT_EQ(persisted_one.snapshot.revision(), latest);
  EXPECT_EQ(persisted_one.snapshot.durableRevision(), 1u);
  EXPECT_TRUE(
      persisted_one.snapshot.findObject(5)->artifact->description.empty());

  const SceneApplyResult impossible =
      reducer.apply(SceneCommand{PersistedThroughCommand{latest + 1}});
  EXPECT_EQ(impossible.status, SceneApplyStatus::kRejected);
  EXPECT_EQ(impossible.snapshot.durableRevision(), 1u);

  const SceneApplyResult fully_persisted =
      reducer.apply(SceneCommand{PersistedThroughCommand{latest}});
  EXPECT_EQ(fully_persisted.status, SceneApplyStatus::kMetadataUpdated);
  EXPECT_EQ(fully_persisted.snapshot.revision(), latest);
  EXPECT_EQ(fully_persisted.snapshot.durableRevision(), latest);
  EXPECT_EQ(fully_persisted.snapshot.findObject(5)->artifact->description,
            "a lamp");
  EXPECT_EQ(fully_persisted.snapshot.findObject(5)
                ->artifact->pending_description_scene_revision,
            0U);
}

TEST(SceneReducer, ExactRestoredStatePreservesComponentAndArtifactHistory) {
  ReducerCore original;
  ASSERT_TRUE(original
                  .apply(SceneCommand{loadCommand({makeNode(5, "lamp")})})
                  .committedRevision());

  ApplySnapshotSetCommand snapshots;
  snapshots.dependency = dependencyFor(*original.snapshot().findObject(5));
  ObjectSnapshotRef first;
  first.image_index = 21;
  first.source_frame_asset_id = std::string(64, 'a');
  first.evidence_hash = std::string(64, '1');
  first.quality = 0.91f;
  ObjectSnapshotRef second = first;
  second.image_index = 22;
  second.source_frame_asset_id = std::string(64, 'b');
  second.evidence_hash = std::string(64, '2');
  second.quality = 0.82f;
  snapshots.snapshots = {first, second};
  snapshots.snapshot_set_hash = "top-k-v2";
  ASSERT_TRUE(original.apply(SceneCommand{snapshots}).committedRevision());

  ApplyDescriptionArtifactCommand description;
  description.dependency = dependencyFor(*original.snapshot().findObject(5));
  description.description = "pending durable description";
  description.input_hash = "description-input";
  description.model_id = "dam-v1";
  description.schema_version = "schema-v1";
  description.raw_text = "raw output";
  description.parse_path = "unstructured_fallback";
  const SceneApplyResult described =
      original.apply(SceneCommand{description});
  ASSERT_TRUE(described.committedRevision()) << described.reason;

  LoadSceneCommand restore;
  restore.restored_state = described.snapshot.statePtr();
  restore.restored_revision = described.snapshot.revision();
  restore.durable_revision = described.snapshot.durableRevision();
  ReducerCore restarted;
  const SceneApplyResult restored =
      restarted.apply(SceneCommand{std::move(restore)});
  ASSERT_TRUE(restored.committedRevision()) << restored.reason;
  const SceneObjectPtr before = described.snapshot.findObject(5);
  const SceneObjectPtr after = restored.snapshot.findObject(5);
  ASSERT_TRUE(before);
  ASSERT_TRUE(after);
  EXPECT_EQ(after.get(), before.get());
  EXPECT_EQ(after->revisions().artifact_revision,
            before->revisions().artifact_revision);
  ASSERT_EQ(after->artifact->snapshots.size(), 2U);
  EXPECT_EQ(after->artifact->snapshot_set_hash, "top-k-v2");
  EXPECT_EQ(after->artifact->pending_description,
            "pending durable description");
  EXPECT_EQ(after->artifact->pending_description_raw_text, "raw output");
  EXPECT_EQ(after->artifact->pending_description_scene_revision,
            described.revision);
}

TEST(SceneReducer, AssociationCallbackAndCommitObserverCannotNestApply) {
  ReducerCore reducer([](const SceneSnapshot&,
                         const ApplyObservationBatchCommand&) {
    UpsertTrackMutation create;
    create.track.track_id = 42;
    create.track.state = InstanceTrackState::kStable;
    create.track.label = "callback object";
    return std::vector<ObservationMutation>{create};
  });

  ApplyObservationBatchCommand batch;
  batch.observations.emplace_back();
  const SceneApplyResult associated = reducer.apply(SceneCommand{batch});
  ASSERT_TRUE(associated.committedRevision()) << associated.reason;
  ASSERT_EQ(associated.snapshot.objects().size(), 1u);

  std::optional<SceneApplyResult> nested;
  reducer.setCommitObserver([&](const SceneApplyResult&) {
    nested = reducer.apply(SceneCommand{PersistedThroughCommand{1}});
  });
  ApplyHumanAnnotationCommand annotation;
  annotation.object_id = associated.snapshot.objects().begin()->first;
  annotation.patch.label = "annotated";
  const SceneApplyResult outer = reducer.apply(SceneCommand{annotation});
  ASSERT_TRUE(outer.committedRevision()) << outer.reason;
  ASSERT_TRUE(nested.has_value());
  EXPECT_EQ(nested->status, SceneApplyStatus::kRejected);
  EXPECT_NE(nested->reason.find("nested apply"), std::string::npos);
  EXPECT_EQ(outer.snapshot.durableRevision(), 0u);

  const SceneApplyResult shutdown =
      reducer.apply(SceneCommand{ShutdownCommand{"test complete"}});
  EXPECT_EQ(shutdown.status, SceneApplyStatus::kMetadataUpdated);
  EXPECT_TRUE(shutdown.snapshot.shutdownRequested());
  EXPECT_EQ(reducer.apply(SceneCommand{batch}).status,
            SceneApplyStatus::kRejected);
}

TEST(SceneReducer, RoomAnnotationsOwnCanonicalContainmentAndRecomputeLocally) {
  ReducerCore reducer;
  ObjectNode inside = makeNode(1, "chair");
  inside.center_world = Eigen::Vector3f(1.0f, 1.0f, 0.5f);
  ObjectNode elsewhere = makeNode(2, "lamp");
  elsewhere.center_world = Eigen::Vector3f(5.0f, 1.0f, 0.5f);
  ASSERT_TRUE(reducer
                  .apply(SceneCommand{loadCommand({inside, elsewhere})})
                  .committedRevision());

  RoomNode pure_room;
  pure_room.room_id = 10;
  pure_room.min_xy = {0.0f, 0.0f};
  pure_room.max_xy = {2.0f, 2.0f};
  pure_room.has_xy_bounds = true;
  const auto pure_relation = deriveRoomContainmentRelation(
      pure_room, 1, *reducer.snapshot().findObject(1)->geometry, 7);
  ASSERT_TRUE(pure_relation.has_value());
  EXPECT_EQ(relationSource(*pure_relation),
            (SceneEntityRef{SceneEntityType::kRoom, 10}));
  EXPECT_EQ(relationTarget(*pure_relation),
            (SceneEntityRef{SceneEntityType::kObject, 1}));

  ApplyHumanAnnotationCommand add_room;
  add_room.target = SceneEntityRef{SceneEntityType::kRoom, 10};
  add_room.room_patch.emplace();
  add_room.room_patch->label = "study";
  add_room.room_patch->min_xy = std::array<float, 2>{0.0f, 0.0f};
  add_room.room_patch->max_xy = std::array<float, 2>{2.0f, 2.0f};
  add_room.room_patch->height_m = 3.0f;
  const SceneApplyResult room_added =
      reducer.apply(SceneCommand{add_room});
  ASSERT_TRUE(room_added.committedRevision()) << room_added.reason;
  ASSERT_EQ(room_added.snapshot.graphMetadata().rooms.size(), 1U);
  ASSERT_EQ(room_added.snapshot.graphMetadata().relations.size(), 1U);
  const ObjectRelation first_relation =
      room_added.snapshot.graphMetadata().relations.front();
  EXPECT_EQ(relationSource(first_relation).type, SceneEntityType::kRoom);
  EXPECT_EQ(relationSource(first_relation).id, 10);
  EXPECT_EQ(relationTarget(first_relation).id, 1);
  EXPECT_TRUE(first_relation.derived);

  ApplyHumanAnnotationCommand add_other_room;
  add_other_room.target = SceneEntityRef{SceneEntityType::kRoom, 11};
  add_other_room.room_patch.emplace();
  add_other_room.room_patch->label = "hall";
  add_other_room.room_patch->min_xy = std::array<float, 2>{4.0f, 0.0f};
  add_other_room.room_patch->max_xy = std::array<float, 2>{6.0f, 2.0f};
  ASSERT_TRUE(reducer.apply(SceneCommand{add_other_room}).committedRevision());
  ASSERT_EQ(reducer.snapshot().graphMetadata().relations.size(), 2U);
  const auto relation_two_it = std::find_if(
      reducer.snapshot().graphMetadata().relations.begin(),
      reducer.snapshot().graphMetadata().relations.end(),
      [](const ObjectRelation& relation) {
        return relationTarget(relation) ==
               SceneEntityRef{SceneEntityType::kObject, 2};
      });
  ASSERT_NE(relation_two_it,
            reducer.snapshot().graphMetadata().relations.end());
  const ObjectRelation relation_for_object_two = *relation_two_it;

  UpsertTrackMutation move;
  move.object_id = 1;
  move.track.track_id = 1001;
  move.track.object_id = 1;
  move.track.state = InstanceTrackState::kStable;
  move.track.publishable = true;
  move.track.label = "chair";
  move.track.center_world = Eigen::Vector3f(9.0f, 9.0f, 0.5f);
  move.track.size_m = inside.size_m;
  move.track.yaw_rad = inside.yaw_rad;
  const SceneApplyResult moved =
      reducer.apply(SceneCommand{mutationBatch(move)});
  ASSERT_TRUE(moved.committedRevision()) << moved.reason;
  ASSERT_EQ(moved.snapshot.graphMetadata().relations.size(), 1U);
  EXPECT_EQ(relationTarget(moved.snapshot.graphMetadata().relations.front()).id,
            2);
  EXPECT_EQ(moved.snapshot.graphMetadata().relations.front().revision,
            relation_for_object_two.revision);

  const ObjectGraphSnapshot materialized =
      moved.snapshot.materializeObjectGraph();
  ASSERT_EQ(materialized.rooms.size(), 2U);
  ASSERT_EQ(materialized.relations.size(), 1U);
  EXPECT_EQ(materialized.objects.size(), 2U);

  ApplyHumanAnnotationCommand remove_room = add_other_room;
  remove_room.room_patch = RoomAnnotationPatch{};
  remove_room.room_patch->remove = true;
  const SceneApplyResult removed =
      reducer.apply(SceneCommand{remove_room});
  ASSERT_TRUE(removed.committedRevision()) << removed.reason;
  EXPECT_TRUE(removed.snapshot.graphMetadata().relations.empty());
  ASSERT_EQ(removed.snapshot.graphMetadata().rooms.size(), 1U);
}

TEST(SceneReducer, FurnitureRolesRebuildExplicitlyAndRelationsTrackGeometry) {
  ReducerCore reducer;
  ObjectNode table = makeNode(1, "table");
  table.center_world = Eigen::Vector3f(0.0f, 0.0f, 0.5f);
  table.size_m = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
  table.yaw_rad = 0.0f;
  ObjectNode cup = makeNode(2, "cup");
  cup.center_world = Eigen::Vector3f(0.0f, 0.0f, 1.1f);
  cup.size_m = Eigen::Vector3f(0.2f, 0.2f, 0.2f);
  cup.yaw_rad = 0.0f;
  ObjectNode lamp = makeNode(3, "lamp");
  lamp.center_world = Eigen::Vector3f(4.0f, 0.0f, 0.5f);
  ASSERT_TRUE(reducer
                  .apply(SceneCommand{loadCommand({table, cup, lamp})})
                  .committedRevision());

  const SceneApplyResult rebuilt =
      reducer.apply(SceneCommand{RebuildFurnitureGraphCommand{}});
  ASSERT_TRUE(rebuilt.committedRevision()) << rebuilt.reason;
  ASSERT_EQ(rebuilt.snapshot.graphMetadata().furniture.size(), 1U);
  EXPECT_EQ(rebuilt.snapshot.graphMetadata().furniture.front().object_id, 1);
  ASSERT_EQ(std::count_if(
                rebuilt.snapshot.graphMetadata().relations.begin(),
                rebuilt.snapshot.graphMetadata().relations.end(),
                [](const SceneRelation& relation) {
                  return relation.relation_type == "on";
                }),
            1);
  EXPECT_EQ(
      reducer.apply(SceneCommand{RebuildFurnitureGraphCommand{}}).status,
      SceneApplyStatus::kNoOp);

  ApplyHumanAnnotationCommand relabel_lamp;
  relabel_lamp.object_id = 3;
  relabel_lamp.expected_identity_revision =
      reducer.snapshot().findObject(3)->identity->revision;
  relabel_lamp.expected_annotation_revision =
      reducer.snapshot().findObject(3)->annotation->revision;
  relabel_lamp.patch.label = "chair";
  ASSERT_TRUE(reducer.apply(SceneCommand{relabel_lamp}).committedRevision());
  EXPECT_EQ(reducer.snapshot().graphMetadata().furniture.size(), 1U);

  ASSERT_TRUE(reducer
                  .apply(SceneCommand{RebuildFurnitureGraphCommand{}})
                  .committedRevision());
  EXPECT_EQ(reducer.snapshot().graphMetadata().furniture.size(), 2U);

  UpsertTrackMutation move_cup;
  move_cup.object_id = 2;
  move_cup.track.track_id = 1002;
  move_cup.track.object_id = 2;
  move_cup.track.state = InstanceTrackState::kStable;
  move_cup.track.publishable = true;
  move_cup.track.label = "cup";
  move_cup.track.center_world = Eigen::Vector3f(8.0f, 0.0f, 1.1f);
  move_cup.track.size_m = cup.size_m;
  move_cup.track.yaw_rad = cup.yaw_rad;
  const SceneApplyResult moved =
      reducer.apply(SceneCommand{mutationBatch(move_cup)});
  ASSERT_TRUE(moved.committedRevision()) << moved.reason;
  EXPECT_EQ(std::count_if(
                moved.snapshot.graphMetadata().relations.begin(),
                moved.snapshot.graphMetadata().relations.end(),
                [](const SceneRelation& relation) {
                  return relation.relation_type == "on" &&
                         relationSource(relation).id == 2;
                }),
            0);
  EXPECT_EQ(moved.snapshot.graphMetadata().furniture.size(), 2U);
}

TEST(SceneReducer, ObservationRetriesAreIdempotentAndLateWindowIsBounded) {
  ReducerCore reducer;
  UpsertTrackMutation create;
  create.track.track_id = 1;
  create.track.state = InstanceTrackState::kStable;
  create.track.label = "cup";
  ApplyObservationBatchCommand first = mutationBatch(create);
  first.provenance.run_id = RunId{42, 9};
  first.provenance.frame_id = 300;
  first.provenance.request_id = 1;
  const SceneApplyResult committed = reducer.apply(SceneCommand{first});
  ASSERT_TRUE(committed.committedRevision()) << committed.reason;
  const SceneRevision revision = committed.revision;
  ASSERT_EQ(committed.snapshot.objects().size(), 1U);

  ApplyObservationBatchCommand retry = first;
  retry.provenance.request_id = 2;
  const SceneApplyResult duplicate = reducer.apply(SceneCommand{retry});
  EXPECT_EQ(duplicate.status, SceneApplyStatus::kNoOp);
  EXPECT_EQ(duplicate.revision, revision);
  EXPECT_EQ(duplicate.snapshot.objects().size(), 1U);

  LoadSceneCommand restored;
  restored.graph = duplicate.snapshot.materializeObjectGraph();
  restored.restored_revision = duplicate.snapshot.revision();
  restored.durable_revision = duplicate.snapshot.revision();
  restored.recent_observation_frames =
      duplicate.snapshot.statePtr()->recent_observation_frames;
  restored.observation_watermarks =
      duplicate.snapshot.statePtr()->observation_watermarks;
  ReducerCore restarted;
  const SceneApplyResult loaded =
      restarted.apply(SceneCommand{std::move(restored)});
  ASSERT_TRUE(loaded.accepted()) << loaded.reason;
  const SceneApplyResult retry_after_restore =
      restarted.apply(SceneCommand{retry});
  EXPECT_EQ(retry_after_restore.status, SceneApplyStatus::kNoOp);
  EXPECT_EQ(retry_after_restore.revision, revision);

  ApplyObservationBatchCommand far_ahead;
  far_ahead.provenance.run_id = first.provenance.run_id;
  far_ahead.provenance.frame_id = 600;
  far_ahead.provenance.request_id = 3;
  ASSERT_TRUE(reducer.apply(SceneCommand{far_ahead}).committedRevision());

  ApplyObservationBatchCommand bounded_late;
  bounded_late.provenance.run_id = first.provenance.run_id;
  bounded_late.provenance.frame_id = 500;
  bounded_late.provenance.request_id = 4;
  EXPECT_TRUE(
      reducer.apply(SceneCommand{bounded_late}).committedRevision());

  ApplyObservationBatchCommand too_old;
  too_old.provenance.run_id = first.provenance.run_id;
  too_old.provenance.frame_id = 1;
  too_old.provenance.request_id = 5;
  const SceneApplyResult rejected = reducer.apply(SceneCommand{too_old});
  EXPECT_EQ(rejected.status, SceneApplyStatus::kRejected);
  EXPECT_NE(rejected.reason.find("bounded late window"), std::string::npos);
}

TEST(SceneReducer, AdvanceSurfaceRemainsMetadataOnlyAndMonotonic) {
  ReducerCore reducer;
  ASSERT_TRUE(reducer
                  .apply(SceneCommand{loadCommand({makeNode(1, "chair")})})
                  .committedRevision());
  const SceneRevision content_revision = reducer.snapshot().revision();
  AdvanceSurfaceCommand advance{surfaceStamp(3, 7, 8)};
  const SceneApplyResult advanced = reducer.apply(SceneCommand{advance});
  EXPECT_EQ(advanced.status, SceneApplyStatus::kMetadataUpdated);
  EXPECT_EQ(advanced.revision, content_revision);
  EXPECT_EQ(advanced.snapshot.latestSurface(), advance.surface);
  EXPECT_TRUE(std::any_of(
      advanced.events.begin(), advanced.events.end(),
      [](const SceneEvent& event) {
        return std::holds_alternative<SurfaceAdvanced>(event);
      }));
  EXPECT_EQ(reducer.apply(SceneCommand{advance}).status,
            SceneApplyStatus::kNoOp);
  AdvanceSurfaceCommand regressed{surfaceStamp(3, 6, 7)};
  EXPECT_EQ(reducer.apply(SceneCommand{regressed}).status,
            SceneApplyStatus::kRejected);
}

TEST(SceneReducer, RoomMutationRejectsPartialOrNegativeGeometry) {
  ReducerCore reducer;
  ASSERT_TRUE(reducer.apply(SceneCommand{loadCommand({makeNode(1, "chair")})})
                  .committedRevision());

  ApplyHumanAnnotationCommand named_room;
  named_room.target = SceneEntityRef{SceneEntityType::kRoom, 9};
  RoomAnnotationPatch named_patch;
  named_patch.label = "study";
  named_room.room_patch = named_patch;
  const SceneApplyResult named =
      reducer.apply(SceneCommand{named_room});
  ASSERT_TRUE(named.committedRevision()) << named.reason;
  const SceneRevision room_revision = named.revision;

  ApplyHumanAnnotationCommand partial_bounds;
  partial_bounds.target = named_room.target;
  RoomAnnotationPatch partial_patch;
  partial_patch.min_xy = std::array<float, 2>{0.0f, 0.0f};
  partial_bounds.room_patch = partial_patch;
  const SceneApplyResult partial =
      reducer.apply(SceneCommand{partial_bounds});
  EXPECT_EQ(partial.status, SceneApplyStatus::kRejected);
  EXPECT_NE(partial.reason.find("XY bounds"), std::string::npos);
  EXPECT_EQ(reducer.snapshot().revision(), room_revision);

  ApplyHumanAnnotationCommand negative_height;
  negative_height.target = named_room.target;
  RoomAnnotationPatch height_patch;
  height_patch.min_xy = std::array<float, 2>{0.0f, 0.0f};
  height_patch.max_xy = std::array<float, 2>{2.0f, 2.0f};
  height_patch.height_m = -1.0f;
  negative_height.room_patch = height_patch;
  const SceneApplyResult height =
      reducer.apply(SceneCommand{negative_height});
  EXPECT_EQ(height.status, SceneApplyStatus::kRejected);
  EXPECT_NE(height.reason.find("height/Z extent"), std::string::npos);
  EXPECT_EQ(reducer.snapshot().revision(), room_revision);

  ApplyHumanAnnotationCommand new_partial;
  new_partial.target = SceneEntityRef{SceneEntityType::kRoom, 10};
  RoomAnnotationPatch new_partial_patch;
  new_partial_patch.min_xy = std::array<float, 2>{-1.0f, -1.0f};
  new_partial.room_patch = new_partial_patch;
  const SceneApplyResult new_room =
      reducer.apply(SceneCommand{new_partial});
  EXPECT_EQ(new_room.status, SceneApplyStatus::kRejected);
  EXPECT_NE(new_room.reason.find("requires both"), std::string::npos);
  EXPECT_EQ(reducer.snapshot().revision(), room_revision);
}

}  // namespace
}  // namespace roomie
