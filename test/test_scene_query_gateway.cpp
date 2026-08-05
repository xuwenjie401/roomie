#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "roomie/query/scene_query_gateway.hpp"

namespace roomie {
namespace {

SurfaceStamp testSurface(std::uint64_t revision = 4) {
  SurfaceStamp stamp;
  stamp.map_epoch = RunId{11, 29};
  stamp.surface_revision = revision;
  stamp.source_map_revision = revision;
  return stamp;
}

SceneObjectPtr makeObject(
    SceneObjectId object_id, std::string label,
    std::optional<std::string> description, const Eigen::Vector3f& center,
    std::map<std::string, std::string> attributes = {},
    InstanceGeometryStatus geometry_status = InstanceGeometryStatus::kGood,
    std::uint64_t obb_revision = 1,
    std::uint64_t evaluated_obb_revision = 1,
    std::string description_input_hash = "dam-input") {
  auto object = std::make_shared<SceneObject>();

  auto identity = std::make_shared<IdentityComponent>();
  identity->revision = 2;
  identity->object_id = object_id;
  object->identity = std::move(identity);

  auto lifecycle = std::make_shared<LifecycleComponent>();
  lifecycle->revision = 3;
  lifecycle->track_state = InstanceTrackState::kStable;
  lifecycle->active = true;
  lifecycle->publishable = true;
  object->lifecycle = std::move(lifecycle);

  auto geometry = std::make_shared<GeometryComponent>();
  geometry->revision = 4;
  geometry->obb_revision = obb_revision;
  geometry->evaluated_obb_revision = evaluated_obb_revision;
  geometry->center_world = center;
  geometry->size_m = Eigen::Vector3f(0.5f, 0.5f, 0.8f);
  geometry->status = geometry_status;
  geometry->evaluated_surface = testSurface();
  object->geometry = std::move(geometry);

  auto semantic = std::make_shared<SemanticComponent>();
  semantic->revision = 5;
  semantic->semantic_id = object_id + 100;
  semantic->label = std::move(label);
  semantic->confidence = 0.9f;
  object->semantic = std::move(semantic);

  auto annotation = std::make_shared<AnnotationComponent>();
  annotation->revision = 2;
  annotation->attributes = std::move(attributes);
  object->annotation = std::move(annotation);

  auto artifact = std::make_shared<ArtifactComponent>();
  artifact->revision = 8;
  artifact->appearance_revision = 2;
  if (description) {
    artifact->description = *description;
  }
  artifact->description_input_hash = std::move(description_input_hash);
  object->artifact = std::move(artifact);
  return object;
}

SceneSnapshot makeSnapshot(
    SceneRevision revision,
    const std::vector<std::pair<SceneObjectId, SceneObjectPtr>>& objects,
    std::vector<ObjectRelation> relations = {},
    std::vector<ObjectAlias> aliases = {},
    std::vector<ObjectSnapshotImage> images = {},
    std::vector<RoomNode, Eigen::aligned_allocator<RoomNode>> rooms = {}) {
  auto state = std::make_shared<SceneState>();
  state->latest_scene_revision = revision;
  state->durable_scene_revision = revision > 0 ? revision - 1 : 0;
  state->latest_surface = testSurface();

  auto object_table = std::make_shared<SceneObjectTable>();
  for (const auto& entry : objects) {
    object_table->emplace(entry.first, entry.second);
    state->next_object_id = std::max(state->next_object_id, entry.first + 1);
  }
  state->objects = object_table;

  auto alias_table = std::make_shared<SceneAliasTable>();
  for (const ObjectAlias& alias : aliases) {
    alias_table->emplace(alias.retired_object_id, alias);
  }
  state->aliases = alias_table;

  auto graph = std::make_shared<SceneGraphMetadata>();
  graph->rooms = std::move(rooms);
  graph->relations = std::move(relations);
  graph->snapshot_images = std::move(images);
  state->graph = graph;
  return SceneSnapshot{state};
}

class FakePinnedIndex final : public PinnedSearchIndex {
 public:
  std::uint64_t generation_value = 0;
  TimeNanoseconds as_of_ns = 0;
  std::map<SceneObjectId, std::string> hashes;
  std::vector<IndexedSearchHit> hits;
  mutable int search_calls = 0;

  std::uint64_t generation() const override { return generation_value; }
  TimeNanoseconds asOfNanoseconds() const override { return as_of_ns; }
  std::optional<std::string> documentHash(
      SceneObjectId object_id) const override {
    const auto it = hashes.find(object_id);
    if (it == hashes.end()) {
      return std::nullopt;
    }
    return it->second;
  }
  std::vector<IndexedSearchHit> search(std::string_view,
                                       std::size_t limit) const override {
    ++search_calls;
    std::vector<IndexedSearchHit> result = hits;
    if (result.size() > limit) {
      result.resize(limit);
    }
    return result;
  }
};

class FakeSearchProvider final : public SearchProvider {
 public:
  std::shared_ptr<const PinnedSearchIndex> current;

  std::shared_ptr<const PinnedSearchIndex> pinCurrent() const override {
    return current;
  }
};

class FakePinnedAssetResolver final : public PinnedSnapshotAssetResolver {
 public:
  std::map<std::string, ResolvedSnapshotAsset> assets;

  std::optional<ResolvedSnapshotAsset> resolve(
      std::string_view source_frame_asset_id) const override {
    const auto asset = assets.find(std::string(source_frame_asset_id));
    return asset == assets.end()
               ? std::nullopt
               : std::optional<ResolvedSnapshotAsset>(asset->second);
  }
};

class FakeSnapshotAssetProvider final : public SnapshotAssetProvider {
 public:
  std::shared_ptr<const PinnedSnapshotAssetResolver> current;
  mutable std::vector<std::vector<std::string>> pin_requests;

  std::shared_ptr<const PinnedSnapshotAssetResolver> pin(
      const std::vector<std::string>& source_frame_asset_ids) const override {
    pin_requests.push_back(source_frame_asset_ids);
    return current;
  }
};

struct FakeClocks {
  SceneReadToken::Clock::time_point steady =
      SceneReadToken::Clock::time_point(std::chrono::seconds(1));
  TimeNanoseconds as_of_ns = 123456789;
};

TEST(SceneQueryGateway, ReadSessionPinsLiveObjectsRoomsRelationsAndAssets) {
  SceneObjectPtr chair = makeObject(
      1, "walnut chair", "old carved walnut chair",
      Eigen::Vector3f(0.0f, 0.0f, 0.0f), {{"room_id", "kitchen"}});
  auto chair_artifact = std::make_shared<ArtifactComponent>(*chair->artifact);
  ObjectSnapshotRef snapshot_ref;
  snapshot_ref.image_index = 7;
  snapshot_ref.camera_id = "front";
  chair_artifact->snapshots.push_back(snapshot_ref);
  chair_artifact->snapshot_set_hash = "snapshot-set-v1";
  auto chair_with_snapshot = std::make_shared<SceneObject>(*chair);
  chair_with_snapshot->artifact = chair_artifact;
  chair = chair_with_snapshot;

  const SceneObjectPtr room = makeObject(
      10, "Kitchen", std::nullopt, Eigen::Vector3f(8.0f, 0.0f, 0.0f),
      {{"entity_type", "room"}, {"room_id", "kitchen"},
       {"room_name", "Kitchen"}});

  ObjectRelation alias_relation;
  alias_relation.source_object_id = 2;
  alias_relation.target_object_id = 10;
  alias_relation.relation_type = "in-room";
  alias_relation.confidence = 0.7f;
  ObjectRelation duplicate_relation = alias_relation;
  duplicate_relation.source_object_id = 1;
  duplicate_relation.relation_type = "in_room";
  duplicate_relation.confidence = 0.9f;
  ObjectAlias alias;
  alias.retired_object_id = 2;
  alias.canonical_object_id = 1;
  alias.created_revision = 4;
  ObjectSnapshotImage image;
  image.image_index = 7;
  image.uri = "asset://snapshot/7";

  SceneSnapshot current = makeSnapshot(
      5, {{1, chair}, {10, room}}, {alias_relation, duplicate_relation},
      {alias}, {image});

  auto index = std::make_shared<FakePinnedIndex>();
  index->generation_value = 7;
  index->as_of_ns = 777;
  index->hashes[1] = SceneQueryGateway::semanticDocumentHash(*chair);
  index->hashes[10] = SceneQueryGateway::semanticDocumentHash(*room);
  index->hits.push_back(
      IndexedSearchHit{1, 0.95f, index->hashes.at(1)});
  auto provider = std::make_shared<FakeSearchProvider>();
  provider->current = index;
  FakeClocks clocks;
  SceneQueryGateway gateway(
      [&current]() { return current; }, provider, std::chrono::seconds(30),
      [&clocks]() { return clocks.steady; },
      [&clocks]() { return clocks.as_of_ns; });

  LocalSceneQueryHandlers answer(gateway, gateway.pin());
  ASSERT_EQ(answer.token().sceneRevision(), 5u);
  ASSERT_EQ(answer.token().indexGeneration(), 7u);

  const SceneObjectPtr sofa = makeObject(
      1, "blue sofa", "new live state", Eigen::Vector3f(20.0f, 0.0f, 0.0f),
      {{"room_id", "lounge"}});
  current = makeSnapshot(6, {{1, sofa}});

  const auto object_result = answer.getObject(2);
  ASSERT_TRUE(object_result.ok()) << object_result.message;
  EXPECT_EQ(object_result.metadata.scene_revision, 5u);
  EXPECT_EQ(object_result.metadata.index_generation, 7u);
  EXPECT_EQ(object_result.metadata.as_of_ns, clocks.as_of_ns);
  EXPECT_EQ(object_result.value.object_id, 1);
  EXPECT_TRUE(object_result.value.resolved_alias);
  EXPECT_EQ(object_result.value.label, "walnut chair");

  ObjectsNearRequest near_request;
  near_request.radius_m = 1.0f;
  const auto near_result = answer.getObjectsNear(near_request);
  ASSERT_TRUE(near_result.ok());
  ASSERT_EQ(near_result.value.size(), 1u);
  EXPECT_EQ(near_result.value.front().object_id, 1);

  const auto room_result = answer.rooms();
  ASSERT_TRUE(room_result.ok());
  ASSERT_EQ(room_result.value.size(), 1u);
  EXPECT_EQ(room_result.value.front().room_id, "kitchen");
  ASSERT_EQ(room_result.value.front().object_ids.size(), 1u);
  EXPECT_EQ(room_result.value.front().object_ids.front(), 1);

  const auto relation_result = answer.relations();
  ASSERT_TRUE(relation_result.ok());
  ASSERT_EQ(relation_result.value.size(), 1u);
  EXPECT_EQ(relation_result.value.front().source_object_id, 1);
  EXPECT_EQ(relation_result.value.front().target_object_id, 10);
  EXPECT_FLOAT_EQ(relation_result.value.front().confidence, 0.9f);

  const auto inspection = answer.inspectSnapshot(1);
  ASSERT_TRUE(inspection.ok());
  ASSERT_EQ(inspection.value.snapshots.size(), 1u);
  EXPECT_TRUE(inspection.value.snapshots.front().available);
  ASSERT_TRUE(inspection.value.snapshots.front().asset);
  EXPECT_EQ(inspection.value.snapshots.front().asset->uri,
            "asset://snapshot/7");

  SearchRequest search_request;
  search_request.query = "walnut";
  const auto search_result = answer.searchObjects(search_request);
  ASSERT_TRUE(search_result.ok());
  ASSERT_EQ(search_result.value.size(), 1u);
  EXPECT_EQ(search_result.value.front().object.object_id, 1);
  EXPECT_EQ(search_result.value.front().source, SearchMatchSource::kVector);

  LocalSceneQueryHandlers next_answer(gateway, gateway.pin());
  const auto next_object = next_answer.getObject(1);
  ASSERT_TRUE(next_object.ok());
  EXPECT_EQ(next_object.metadata.scene_revision, 6u);
  EXPECT_EQ(next_object.value.label, "blue sofa");
  EXPECT_TRUE(next_answer.relations().value.empty());
}

TEST(SceneQueryGateway,
     GenerationSwitchCannotPolluteOldTokenAndStaleRowsUseLexicalDelta) {
  const SceneObjectPtr object = makeObject(
      3, "armchair", "blue velvet reading chair",
      Eigen::Vector3f(1.0f, 0.0f, 0.0f));
  SceneSnapshot current = makeSnapshot(20, {{3, object}});
  const std::string current_hash =
      SceneQueryGateway::semanticDocumentHash(*object);

  auto old_index = std::make_shared<FakePinnedIndex>();
  old_index->generation_value = 41;
  old_index->hashes[3] = "stale-document-hash";
  old_index->hits.push_back(
      IndexedSearchHit{3, 0.99f, "stale-document-hash"});
  auto new_index = std::make_shared<FakePinnedIndex>();
  new_index->generation_value = 42;
  new_index->hashes[3] = current_hash;
  new_index->hits.push_back(IndexedSearchHit{3, 0.93f, current_hash});

  auto provider = std::make_shared<FakeSearchProvider>();
  provider->current = old_index;
  FakeClocks clocks;
  SceneQueryGateway gateway(
      [&current]() { return current; }, provider, std::chrono::seconds(30),
      [&clocks]() { return clocks.steady; },
      [&clocks]() { return clocks.as_of_ns; });

  LocalSceneQueryHandlers old_answer(gateway, gateway.pin());
  provider->current = new_index;
  LocalSceneQueryHandlers new_answer(gateway, gateway.pin());

  SearchRequest request;
  request.query = "blue velvet";
  const auto old_result = old_answer.searchObjects(request);
  ASSERT_TRUE(old_result.ok()) << old_result.message;
  EXPECT_EQ(old_result.metadata.index_generation, 41u);
  EXPECT_TRUE(old_result.metadata.lexical_delta_used);
  ASSERT_EQ(old_result.value.size(), 1u);
  EXPECT_EQ(old_result.value.front().source,
            SearchMatchSource::kPinnedLexicalDelta);
  EXPECT_EQ(old_result.value.front().vector_document_hash,
            "stale-document-hash");
  EXPECT_EQ(old_result.value.front().object.freshness.semantic,
            Freshness::kStale);

  const auto new_result = new_answer.searchObjects(request);
  ASSERT_TRUE(new_result.ok());
  EXPECT_EQ(new_result.metadata.index_generation, 42u);
  EXPECT_FALSE(new_result.metadata.lexical_delta_used);
  ASSERT_EQ(new_result.value.size(), 1u);
  EXPECT_EQ(new_result.value.front().source, SearchMatchSource::kVector);
  EXPECT_EQ(new_result.value.front().object.freshness.semantic,
            Freshness::kCurrent);

  // The provider has switched, but the first answer still owns generation 41.
  const auto repeated_old_result = old_answer.searchObjects(request);
  EXPECT_EQ(repeated_old_result.metadata.index_generation, 41u);
  EXPECT_EQ(old_index->search_calls, 2);
  EXPECT_EQ(new_index->search_calls, 1);
}

TEST(SceneQueryGateway, ExpiryAndPendingOrStaleArtifactsAreExplicit) {
  SceneObjectPtr object = makeObject(
      8, "storage bin", "legacy description",
      Eigen::Vector3f(0.0f, 0.0f, 0.0f), {},
      InstanceGeometryStatus::kUnchecked, 4, 3, "");
  auto artifact = std::make_shared<ArtifactComponent>(*object->artifact);
  artifact->appearance_revision = 9;
  artifact->snapshot_set_hash.clear();
  ObjectSnapshotRef reference;
  reference.image_index = 99;
  artifact->snapshots.push_back(reference);
  auto pending_object = std::make_shared<SceneObject>(*object);
  pending_object->artifact = artifact;
  object = pending_object;

  SceneSnapshot current = makeSnapshot(30, {{8, object}});
  auto index = std::make_shared<FakePinnedIndex>();
  index->generation_value = 12;
  auto provider = std::make_shared<FakeSearchProvider>();
  provider->current = index;
  FakeClocks clocks;
  SceneQueryGateway gateway(
      [&current]() { return current; }, provider,
      std::chrono::milliseconds(10),
      [&clocks]() { return clocks.steady; },
      [&clocks]() { return clocks.as_of_ns; });
  LocalSceneQueryHandlers answer(gateway, gateway.pin());

  const auto before_expiry = answer.getObject(8);
  ASSERT_TRUE(before_expiry.ok());
  EXPECT_EQ(before_expiry.value.freshness.geometry, Freshness::kPending);
  EXPECT_EQ(before_expiry.value.freshness.semantic, Freshness::kPending);
  EXPECT_EQ(before_expiry.value.freshness.description, Freshness::kStale);
  EXPECT_TRUE(before_expiry.metadata.artifact_pending);
  EXPECT_FALSE(before_expiry.value.freshness.pending_artifacts.empty());

  const auto inspection = answer.inspectSnapshot(8);
  ASSERT_TRUE(inspection.ok());
  ASSERT_EQ(inspection.value.snapshots.size(), 1u);
  EXPECT_FALSE(inspection.value.snapshots.front().available);
  EXPECT_TRUE(inspection.metadata.artifact_pending);

  clocks.steady += std::chrono::milliseconds(10);
  const auto expired = answer.getObject(8);
  EXPECT_EQ(expired.status, QueryStatus::kExpiredToken);
  EXPECT_NE(expired.message.find("expired"), std::string::npos);
  EXPECT_EQ(expired.metadata.scene_revision, 30u);
  EXPECT_EQ(expired.metadata.index_generation, 12u);
  EXPECT_EQ(answer.rooms().status, QueryStatus::kExpiredToken);

  FakeClocks other_clocks;
  SceneQueryGateway other_gateway(
      [&current]() { return current; }, provider, std::chrono::seconds(1),
      [&other_clocks]() { return other_clocks.steady; },
      [&other_clocks]() { return other_clocks.as_of_ns; });
  EXPECT_EQ(other_gateway.getObject(answer.token(), 8).status,
            QueryStatus::kForeignToken);
}

TEST(SceneQueryGateway, DescriptionFallbackLivesOnlyInQueryCompatibilityView) {
  const SceneObjectPtr object = makeObject(
      4, "plain label", std::nullopt, Eigen::Vector3f::Zero(), {},
      InstanceGeometryStatus::kGood, 1, 1, "");
  SceneSnapshot current = makeSnapshot(2, {{4, object}});
  auto provider = std::make_shared<FakeSearchProvider>();
  FakeClocks clocks;
  SceneQueryGateway gateway(
      [&current]() { return current; }, provider, std::chrono::seconds(1),
      [&clocks]() { return clocks.steady; },
      [&clocks]() { return clocks.as_of_ns; });

  const auto result = gateway.getObject(gateway.pin(), 4);
  ASSERT_TRUE(result.ok());
  EXPECT_FALSE(result.value.canonical_description.has_value());
  EXPECT_EQ(result.value.display_description, "plain label");
  EXPECT_TRUE(result.value.description_is_label_fallback);
  EXPECT_TRUE(object->artifact->description.empty());
}

TEST(SceneQueryGateway, CanonicalRoomsAndTypedRelationEndpointsStayTyped) {
  const SceneObjectPtr chair = makeObject(
      1, "chair", std::nullopt, Eigen::Vector3f::Zero());
  RoomNode room;
  room.room_id = 42;
  room.revision = 3;
  room.label = "Reading room";
  room.attributes["room_id"] = "reading-room";

  ObjectRelation containment;
  containment.relation_type = "room_contains_object";
  containment.confidence = 0.95f;
  containment.revision = 9;
  containment.derived = true;
  setRelationEndpoints(
      &containment, SceneEntityRef{SceneEntityType::kRoom, 42},
      SceneEntityRef{SceneEntityType::kObject, 1});

  SceneSnapshot current = makeSnapshot(
      10, {{1, chair}}, {containment}, {}, {}, {room});
  SceneQueryGateway gateway([&current]() { return current; });
  const SceneReadToken token = gateway.pin();

  const auto rooms = gateway.rooms(token);
  ASSERT_TRUE(rooms.ok()) << rooms.message;
  ASSERT_EQ(rooms.value.size(), 1U);
  EXPECT_EQ(rooms.value.front().room_id, "reading-room");
  EXPECT_EQ(rooms.value.front().name, "Reading room");
  EXPECT_EQ(rooms.value.front().object_ids,
            std::vector<SceneObjectId>{1});

  RelationRequest request;
  request.object_id = 1;
  request.direction = RelationDirection::kIncoming;
  const auto relations = gateway.relations(token, request);
  ASSERT_TRUE(relations.ok()) << relations.message;
  ASSERT_EQ(relations.value.size(), 1U);
  EXPECT_EQ(relations.value.front().source,
            (SceneEntityRef{SceneEntityType::kRoom, 42}));
  EXPECT_EQ(relations.value.front().target,
            (SceneEntityRef{SceneEntityType::kObject, 1}));
  EXPECT_EQ(relations.value.front().source_object_id, -1);
  EXPECT_EQ(relations.value.front().target_object_id, 1);
  EXPECT_EQ(relations.value.front().revision, 9U);
  EXPECT_TRUE(relations.value.front().derived);
}

TEST(SceneQueryGateway,
     OnlineContentAssetsAreResolvedFromTheTokenPinnedProviderView) {
  SceneObjectPtr object = makeObject(
      6, "lamp", std::nullopt, Eigen::Vector3f::Zero());
  auto artifact = std::make_shared<ArtifactComponent>(*object->artifact);
  ObjectSnapshotRef available_reference;
  available_reference.image_index = 17;
  available_reference.source_frame_asset_id = "frame-content-a";
  available_reference.time_ns = 500;
  available_reference.camera_id = "front";
  ObjectSnapshotRef missing_reference;
  missing_reference.image_index = 18;
  missing_reference.source_frame_asset_id = "frame-content-missing";
  artifact->snapshots = {available_reference, missing_reference};
  artifact->snapshot_set_hash = "online-set";
  auto updated_object = std::make_shared<SceneObject>(*object);
  updated_object->artifact = std::move(artifact);
  object = std::move(updated_object);

  // Matching legacy indices must not mask a missing content-addressed asset.
  ObjectSnapshotImage stale_legacy_available;
  stale_legacy_available.image_index = 17;
  stale_legacy_available.uri = "asset://legacy/wrong-a";
  ObjectSnapshotImage stale_legacy_missing;
  stale_legacy_missing.image_index = 18;
  stale_legacy_missing.uri = "asset://legacy/wrong-missing";
  SceneSnapshot current = makeSnapshot(
      12, {{6, object}}, {}, {},
      {stale_legacy_available, stale_legacy_missing});

  auto first_assets = std::make_shared<FakePinnedAssetResolver>();
  ResolvedSnapshotAsset first;
  first.source_frame_asset_id = "frame-content-a";
  first.uri = "file:///assets/frame-content-a.png";
  first.source_path = "/assets/frame-content-a.png";
  first.width = 960;
  first.height = 540;
  first.channels = 3;
  first.encoding = "png";
  first.source_encoding = "rgb8";
  first.encoded_bytes = 1234;
  first_assets->assets.emplace(first.source_frame_asset_id, first);

  auto second_assets = std::make_shared<FakePinnedAssetResolver>();
  ResolvedSnapshotAsset second = first;
  second.uri = "file:///relocated/frame-content-a.png";
  second.source_path = "/relocated/frame-content-a.png";
  second_assets->assets.emplace(second.source_frame_asset_id, second);

  auto asset_provider = std::make_shared<FakeSnapshotAssetProvider>();
  asset_provider->current = first_assets;
  SceneQueryGateway gateway(
      [&current]() { return current; },
      std::shared_ptr<const SearchProvider>{}, asset_provider,
      std::chrono::seconds(30));

  const SceneReadToken first_token = gateway.pin();
  ASSERT_EQ(asset_provider->pin_requests.size(), 1U);
  EXPECT_EQ(asset_provider->pin_requests.front(),
            (std::vector<std::string>{"frame-content-a",
                                      "frame-content-missing"}));
  asset_provider->current = second_assets;
  const SceneReadToken second_token = gateway.pin();

  const auto first_inspection = gateway.inspectSnapshot(first_token, 6);
  ASSERT_TRUE(first_inspection.ok()) << first_inspection.message;
  ASSERT_EQ(first_inspection.value.snapshots.size(), 2U);
  const SnapshotAssetInspection& first_available =
      first_inspection.value.snapshots[0];
  EXPECT_TRUE(first_available.available);
  ASSERT_TRUE(first_available.physical_asset);
  EXPECT_EQ(first_available.physical_asset->source_frame_asset_id,
            "frame-content-a");
  EXPECT_EQ(first_available.physical_asset->encoded_bytes, 1234U);
  EXPECT_EQ(first_available.physical_asset->source_encoding, "rgb8");
  ASSERT_TRUE(first_available.asset);
  EXPECT_EQ(first_available.asset->uri,
            "file:///assets/frame-content-a.png");
  EXPECT_EQ(first_available.asset->source_path,
            "/assets/frame-content-a.png");
  EXPECT_EQ(first_available.asset->width, 960);
  EXPECT_EQ(first_available.asset->height, 540);
  EXPECT_EQ(first_available.asset->encoding, "png");
  EXPECT_EQ(first_available.asset->time_ns, 500);
  EXPECT_EQ(first_available.asset->camera_id, "front");

  const SnapshotAssetInspection& first_missing =
      first_inspection.value.snapshots[1];
  EXPECT_FALSE(first_missing.available);
  EXPECT_FALSE(first_missing.physical_asset);
  EXPECT_FALSE(first_missing.asset)
      << "online refs must not fall back to a colliding legacy image index";
  EXPECT_TRUE(first_inspection.metadata.artifact_pending);

  // The provider switched, but the first token retains its original resolver.
  const auto repeated_first = gateway.inspectSnapshot(first_token, 6, 17);
  ASSERT_TRUE(repeated_first.ok());
  ASSERT_EQ(repeated_first.value.snapshots.size(), 1U);
  ASSERT_TRUE(repeated_first.value.snapshots.front().asset);
  EXPECT_EQ(repeated_first.value.snapshots.front().asset->uri,
            "file:///assets/frame-content-a.png");

  const auto second_inspection = gateway.inspectSnapshot(second_token, 6, 17);
  ASSERT_TRUE(second_inspection.ok());
  ASSERT_EQ(second_inspection.value.snapshots.size(), 1U);
  ASSERT_TRUE(second_inspection.value.snapshots.front().asset);
  EXPECT_EQ(second_inspection.value.snapshots.front().asset->uri,
            "file:///relocated/frame-content-a.png");
}

TEST(SceneQueryGateway,
     ServerReadSessionsPinRevisionExpireAndRespectCapacity) {
  SceneSnapshot current = makeSnapshot(
      50, {{1, makeObject(1, "chair", "wood chair",
                          Eigen::Vector3f::Zero())}});
  FakeClocks clocks;
  SceneQueryGateway gateway(
      [&current]() { return current; },
      std::shared_ptr<const SearchProvider>{}, std::chrono::seconds(30),
      [&clocks]() { return clocks.steady; },
      [&clocks]() { return clocks.as_of_ns; }, 2);

  const SceneReadSessionResult first =
      gateway.beginReadSession(std::chrono::milliseconds(20));
  ASSERT_TRUE(first.ok()) << first.message;
  EXPECT_FALSE(first.session_id.empty());
  EXPECT_EQ(first.token.sceneRevision(), 50U);
  EXPECT_EQ(gateway.activeReadSessionCount(), 1U);

  current = makeSnapshot(
      51, {{1, makeObject(1, "sofa", "blue sofa",
                          Eigen::Vector3f::Zero())}});
  const SceneReadSessionResult resumed =
      gateway.resumeReadSession(first.session_id, 50);
  ASSERT_TRUE(resumed.ok()) << resumed.message;
  EXPECT_EQ(resumed.token.sceneRevision(), 50U);
  const auto pinned_object = gateway.getObject(resumed.token, 1);
  ASSERT_TRUE(pinned_object.ok()) << pinned_object.message;
  EXPECT_EQ(pinned_object.value.label, "chair");

  const SceneReadSessionResult stale =
      gateway.resumeReadSession(first.session_id, 51);
  EXPECT_EQ(stale.status, SceneReadSessionStatus::kRevisionMismatch);
  EXPECT_NE(stale.message.find("revision 50"), std::string::npos);

  const SceneReadSessionResult second =
      gateway.beginReadSession(std::chrono::milliseconds(20));
  ASSERT_TRUE(second.ok()) << second.message;
  EXPECT_NE(second.session_id, first.session_id);
  const SceneReadSessionResult full =
      gateway.beginReadSession(std::chrono::milliseconds(20));
  EXPECT_EQ(full.status, SceneReadSessionStatus::kCapacityExceeded);
  EXPECT_EQ(gateway.activeReadSessionCount(), 2U);

  clocks.steady += std::chrono::milliseconds(20);
  const SceneReadSessionResult expired =
      gateway.resumeReadSession(first.session_id, 50);
  EXPECT_EQ(expired.status, SceneReadSessionStatus::kExpired);
  EXPECT_NE(expired.message.find("expired"), std::string::npos);
  EXPECT_EQ(gateway.activeReadSessionCount(), 0U);

  const SceneReadSessionResult replacement =
      gateway.beginReadSession(std::chrono::milliseconds(5));
  EXPECT_TRUE(replacement.ok()) << replacement.message;
  EXPECT_EQ(gateway.resumeReadSession("never-issued").status,
            SceneReadSessionStatus::kNotFound);
}

}  // namespace
}  // namespace roomie
