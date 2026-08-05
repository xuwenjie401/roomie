#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "roomie/dsg/object_graph_io.hpp"
#include "roomie/dsg/object_snapshot_remaker.hpp"
#include "roomie/pipeline/instance_map_thread.hpp"

namespace roomie {
namespace {

std::filesystem::path makeTempDir(const std::string& name) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("roomie_" + name);
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

ObjectSnapshotRef makeRef(int image_index, float quality) {
  ObjectSnapshotRef ref;
  ref.image_index = image_index;
  ref.source_frame_asset_id = "sha256-frame";
  ref.evidence_hash = "sha256-evidence";
  ref.bbox_xyxy = {1.0f, 1.0f, 5.0f, 5.0f};
  ref.crop_xywh = {1.0f, 1.0f, 4.0f, 4.0f};
  ref.crop_output_scale = {2.0f, 2.0f};
  ref.mask_source = "bbox_fallback";
  ref.quality = quality;
  ref.quality_components["blur"] = 0.9f;
  ref.viewpoint_azimuth_rad = 0.25f;
  ref.viewpoint_elevation_rad = -0.1f;
  ref.viewpoint_scale = 0.2f;
  ref.time_ns = 123;
  ref.camera_id = "head";
  ref.provenance.run_id = RunId{11, 12};
  ref.provenance.frame_id = 13;
  ref.provenance.request_id = 14;
  ref.provenance.sensor_time_ns = 123;
  ref.provenance.includes_current_frame = true;
  ref.provenance.causality_verified = true;
  return ref;
}

ObjectNode makeObject(int object_id) {
  ObjectNode object;
  object.object_id = object_id;
  object.label = "cup";
  object.active = true;
  object.publishable = true;
  return object;
}

ImageBuffer makeRgbImage(int width, int height) {
  ImageBuffer image;
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.encoding = "rgb8";
  image.data.resize(static_cast<std::size_t>(width * height * image.channels));
  for (std::size_t i = 0; i < image.data.size(); ++i) {
    image.data[i] = static_cast<std::uint8_t>(i % 251U);
  }
  return image;
}

class FakeMapProjector final : public MapProjector {
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

TEST(ObjectGraphIo, SnapshotJsonRoundTrip) {
  const std::filesystem::path dir = makeTempDir("snapshot_json");
  const std::filesystem::path image_dir = dir / "snapshots";
  std::filesystem::create_directories(image_dir);
  {
    std::ofstream image(image_dir / "snapshot_0000.bmp", std::ios::binary);
    image << "BM";
  }

  ObjectGraphSnapshot snapshot;
  snapshot.next_object_id = 2;
  ObjectNode object = makeObject(1);
  object.snapshot = makeRef(0, 0.75f);
  snapshot.objects.push_back(object);
  ObjectSnapshotImage image;
  image.image_index = 0;
  image.uri = "snapshots/snapshot_0000.bmp";
  image.width = 8;
  image.height = 8;
  image.encoding = "bmp";
  image.time_ns = 123;
  image.camera_id = "head";
  snapshot.snapshot_images.push_back(image);

  const std::filesystem::path path = dir / "graph.json";
  std::string error;
  ASSERT_TRUE(saveObjectGraphSnapshotJsonAtomic(snapshot,
                                               "world",
                                               456,
                                               path,
                                               &error))
      << error;

  ObjectGraphSnapshot loaded;
  std::string world_frame;
  ASSERT_TRUE(loadObjectGraphSnapshotJson(path, &loaded, &world_frame, &error))
      << error;
  ASSERT_EQ(world_frame, "world");
  ASSERT_EQ(loaded.objects.size(), 1U);
  ASSERT_TRUE(loaded.objects.front().snapshot.valid());
  EXPECT_EQ(loaded.objects.front().snapshot.image_index, 0);
  EXPECT_EQ(loaded.objects.front().snapshot.source_frame_asset_id,
            "sha256-frame");
  EXPECT_EQ(loaded.objects.front().snapshot.evidence_hash,
            "sha256-evidence");
  EXPECT_EQ(loaded.objects.front().snapshot.crop_xywh,
            (std::array<float, 4>{1.0f, 1.0f, 4.0f, 4.0f}));
  EXPECT_EQ(loaded.objects.front().snapshot.quality_components.at("blur"),
            0.9f);
  EXPECT_EQ(loaded.objects.front().snapshot.provenance.frame_id, 13u);
  ASSERT_EQ(loaded.snapshot_images.size(), 1U);
  EXPECT_EQ(loaded.snapshot_images.front().uri, "snapshots/snapshot_0000.bmp");
  EXPECT_TRUE(std::filesystem::exists(loaded.snapshot_images.front().source_path));
}

TEST(ObjectGraphIo, ManualSceneGraphLoadsWithoutRooms) {
  const std::filesystem::path dir = makeTempDir("manual_scene_graph_no_rooms");
  const std::filesystem::path path = dir / "manual_scene_graph.json";

  nlohmann::json root;
  root["format"] = "roomie_manual_scene_graph";
  root["world_frame"] = "world";
  root["object_graph"] = {
      {"format", "roomie_object_graph"},
      {"format_version", 2},
      {"world_frame", "world"},
      {"next_object_id", 8},
      {"objects",
       nlohmann::json::array(
           {{{"object_id", 7},
             {"semantic_id", -1},
             {"label", "chair"},
             {"center_world", nlohmann::json::array({1.0, 2.0, 0.5})},
             {"size_m", nlohmann::json::array({0.5, 0.6, 0.7})},
             {"yaw_rad", 0.0},
             {"active", true},
             {"publishable", true}}})},
      {"relations", nlohmann::json::array()},
  };
  {
    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    stream << root.dump(2) << '\n';
  }

  ObjectGraphSnapshot loaded;
  std::string world_frame;
  std::string error;
  ASSERT_TRUE(loadObjectGraphSnapshotJson(path, &loaded, &world_frame, &error))
      << error;
  ASSERT_EQ(world_frame, "world");
  ASSERT_EQ(loaded.objects.size(), 1U);
  EXPECT_EQ(loaded.objects.front().object_id, 7);
  EXPECT_EQ(loaded.objects.front().label, "chair");
  EXPECT_TRUE(loaded.relations.empty());
}

TEST(ObjectGraphIo, ManualSceneGraphPreservesEnvelopeWhenSavingSnapshots) {
  const std::filesystem::path dir = makeTempDir("manual_scene_graph_snapshot_save");
  const std::filesystem::path path = dir / "manual_scene_graph.json";

  nlohmann::json root;
  root["format"] = "roomie_manual_scene_graph";
  root["world_frame"] = "world";
  root["rooms"] = nlohmann::json::array(
      {{{"room_id", 3}, {"name", "kitchen"}, {"object_ids", nlohmann::json::array({7})}}});
  root["objects"] = nlohmann::json::array(
      {{{"object_id", 7},
        {"label", "chair"},
        {"description", "human-authored note"},
        {"room_id", 3}}});
  root["object_graph"] = {
      {"format", "roomie_object_graph"},
      {"format_version", 2},
      {"world_frame", "world"},
      {"next_object_id", 8},
      {"objects",
       nlohmann::json::array(
           {{{"object_id", 7},
             {"semantic_id", -1},
             {"label", "chair"},
             {"center_world", nlohmann::json::array({1.0, 2.0, 0.5})},
             {"size_m", nlohmann::json::array({0.5, 0.6, 0.7})},
             {"yaw_rad", 0.0},
             {"active", true},
             {"publishable", true}}})},
      {"relations", nlohmann::json::array()},
  };
  {
    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    stream << root.dump(2) << '\n';
  }

  ObjectGraphSnapshot loaded;
  std::string world_frame;
  std::string error;
  ASSERT_TRUE(loadObjectGraphSnapshotJson(path, &loaded, &world_frame, &error))
      << error;
  ASSERT_TRUE(loaded.has_scene_graph_envelope);
  ASSERT_EQ(loaded.objects.size(), 1U);
  EXPECT_EQ(loaded.objects.front().description, "human-authored note");
  loaded.objects.front().snapshot = makeRef(0, 0.82f);
  ObjectSnapshotImage image;
  image.image_index = 0;
  image.uri = "snapshots/run/snapshot_0000.bmp";
  image.width = 32;
  image.height = 32;
  image.encoding = "bmp";
  image.time_ns = 123;
  image.camera_id = "head";
  loaded.snapshot_images.push_back(image);

  const std::filesystem::path saved_path = dir / "saved.json";
  ASSERT_TRUE(saveObjectGraphSnapshotJsonAtomic(loaded,
                                               "world",
                                               456,
                                               saved_path,
                                               &error))
      << error;

  std::ifstream saved_stream(saved_path);
  ASSERT_TRUE(saved_stream);
  const nlohmann::json saved = nlohmann::json::parse(saved_stream);
  EXPECT_EQ(saved.value("format", std::string()), "roomie_manual_scene_graph");
  EXPECT_EQ(saved.value("format_version", 0), 3);
  EXPECT_FALSE(saved.contains("object_graph"));
  ASSERT_TRUE(saved.contains("rooms"));
  EXPECT_EQ(saved.at("rooms").at(0).value("room_id", -1), 3);
  ASSERT_EQ(saved.at("objects").size(), 1U);
  ASSERT_TRUE(saved.at("objects").at(0).contains("snapshot"));
  ASSERT_TRUE(saved.contains("snapshot_images"));
  EXPECT_EQ(saved.at("snapshot_images").size(), 1U);
  ASSERT_TRUE(saved.contains("migration_warnings"));
  EXPECT_FALSE(saved.at("migration_warnings").empty());

  ObjectGraphSnapshot reloaded;
  ASSERT_TRUE(loadObjectGraphSnapshotJson(
      saved_path, &reloaded, &world_frame, &error)) << error;
  ASSERT_EQ(reloaded.objects.size(), 1U);
  EXPECT_EQ(reloaded.objects.front().description, "human-authored note");
  ASSERT_EQ(reloaded.rooms.size(), 1U);
}

TEST(ObjectGraphIo, ManualEnvelopeRejectsConflictingDuplicateObjects) {
  const std::filesystem::path dir =
      makeTempDir("manual_scene_graph_object_conflict");
  const std::filesystem::path path = dir / "manual_scene_graph.json";
  nlohmann::json root;
  root["format"] = "roomie_manual_scene_graph";
  root["format_version"] = 1;
  root["objects"] = nlohmann::json::array(
      {{{"object_id", 7}, {"label", "table"}}});
  root["object_graph"] = {
      {"format", "roomie_object_graph"},
      {"format_version", 2},
      {"objects", nlohmann::json::array(
                      {{{"object_id", 7}, {"label", "chair"}}})},
      {"relations", nlohmann::json::array()},
  };
  {
    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    stream << root.dump(2) << '\n';
  }

  ObjectGraphSnapshot loaded;
  std::string world_frame;
  std::string error;
  EXPECT_FALSE(loadObjectGraphSnapshotJson(path, &loaded, &world_frame,
                                           &error));
  EXPECT_NE(error.find("object conflict"), std::string::npos) << error;
  EXPECT_NE(error.find("label"), std::string::npos) << error;
}

TEST(ObjectGraphIo, LegacyV1AndSchemaV3TypedRoomsRelationsAreCompatible) {
  const std::filesystem::path dir = makeTempDir("schema_v1_v3_compat");
  const std::filesystem::path legacy_path = dir / "legacy_v1.json";
  nlohmann::json legacy = {
      {"format", "roomie_object_graph"},
      {"format_version", 1},
      {"world_frame", "map"},
      {"next_object_id", 2},
      {"objects", nlohmann::json::array(
                      {{{"object_id", 1}, {"label", "cup"}}})},
      {"relations", nlohmann::json::array()},
  };
  {
    std::ofstream stream(legacy_path, std::ios::out | std::ios::trunc);
    stream << legacy.dump(2) << '\n';
  }
  ObjectGraphSnapshot loaded_v1;
  std::string world_frame;
  std::string error;
  ASSERT_TRUE(loadObjectGraphSnapshotJson(
      legacy_path, &loaded_v1, &world_frame, &error)) << error;
  ASSERT_EQ(loaded_v1.objects.size(), 1U);
  EXPECT_TRUE(loaded_v1.objects.front().description.empty());

  RoomNode room;
  room.room_id = 9;
  room.revision = 4;
  room.label = "kitchen";
  room.center_world = Eigen::Vector3f(1.0f, 2.0f, 1.5f);
  room.size_m = Eigen::Vector3f(4.0f, 5.0f, 3.0f);
  room.min_xy = {-1.0f, -0.5f};
  room.max_xy = {3.0f, 4.5f};
  room.has_xy_bounds = true;
  loaded_v1.rooms.push_back(room);
  ObjectRelation containment;
  setRelationEndpoints(
      &containment, SceneEntityRef{SceneEntityType::kRoom, 9},
      SceneEntityRef{SceneEntityType::kObject, 1});
  containment.relation_type = "room_contains_object";
  containment.confidence = 1.0f;
  containment.revision = 5;
  containment.derived = true;
  loaded_v1.relations.push_back(containment);

  const std::filesystem::path v3_path = dir / "canonical_v3.json";
  ASSERT_TRUE(saveObjectGraphSnapshotJsonAtomic(
      loaded_v1, "map", 456, v3_path, &error)) << error;
  std::ifstream stream(v3_path);
  const nlohmann::json v3 = nlohmann::json::parse(stream);
  EXPECT_EQ(v3.value("format_version", 0), 3);
  ASSERT_EQ(v3.at("objects").size(), 1U);
  ASSERT_EQ(v3.at("rooms").size(), 1U);
  ASSERT_EQ(v3.at("relations").size(), 1U);
  EXPECT_EQ(v3.at("relations").at(0).at("source").value(
                "type", std::string()), "room");
  EXPECT_TRUE(v3.at("objects").at(0).value(
                  "description", std::string("unexpected")).empty());

  ObjectGraphSnapshot loaded_v3;
  ASSERT_TRUE(loadObjectGraphSnapshotJson(
      v3_path, &loaded_v3, &world_frame, &error)) << error;
  ASSERT_EQ(loaded_v3.rooms.size(), 1U);
  ASSERT_EQ(loaded_v3.relations.size(), 1U);
  EXPECT_EQ(relationSource(loaded_v3.relations.front()).type,
            SceneEntityType::kRoom);
  EXPECT_EQ(relationTarget(loaded_v3.relations.front()).id, 1);
}

TEST(ObjectGraph, DescriptionIsNotSynthesizedFromDetectorLabel) {
  ObjectGraph graph;
  InstanceTrack track;
  track.track_id = 4;
  track.state = InstanceTrackState::kStable;
  track.label = "detector chair";
  const int object_id = graph.createNodeFromTrack(track);
  ObjectGraphSnapshot created = graph.snapshot();
  ASSERT_EQ(created.objects.size(), 1U);
  EXPECT_TRUE(created.objects.front().description.empty());

  track.object_id = object_id;
  track.label = "updated chair";
  graph.updateNodeFromTrack(track);
  const ObjectGraphSnapshot updated = graph.snapshot();
  ASSERT_EQ(updated.objects.size(), 1U);
  EXPECT_EQ(updated.objects.front().label, "updated chair");
  EXPECT_TRUE(updated.objects.front().description.empty());
}

TEST(ObjectSnapshotRemaker, KeepsOnlySignificantObjectSnapshotImprovements) {
  const std::filesystem::path dir = makeTempDir("snapshot_remaker");
  ObjectSnapshotRemakerConfig config;
  config.enabled = true;
  config.staging_dir = dir / "staging";
  config.min_quality = 0.0f;
  config.min_box_area_px = 1.0f;
  config.position_weight = 0.5f;
  config.size_weight = 0.5f;
  config.replace_min_quality_delta = 0.12f;
  config.replace_min_quality_ratio = 1.20f;

  ObjectGraphSnapshot snapshot;
  snapshot.objects.push_back(makeObject(4));
  ObjectSnapshotRemaker remaker(config);
  remaker.loadSnapshot(snapshot);

  const ImageBuffer image = makeRgbImage(32, 32);
  ObjectSnapshotRemakeCandidate candidate;
  candidate.object_id = 4;
  candidate.bbox_xyxy = {8.0f, 8.0f, 24.0f, 24.0f};
  candidate.bbox_quality = 0.50f;
  candidate.time_ns = 100;
  candidate.camera_id = "head";

  std::string error;
  ObjectSnapshotRemakeFrameResult first =
      remaker.submitFrame(image, 100, "head", {candidate}, &error);
  ASSERT_TRUE(error.empty()) << error;
  EXPECT_TRUE(first.saved_frame);
  EXPECT_EQ(first.improved_objects, 1U);

  candidate.bbox_quality = 0.54f;
  ObjectSnapshotRemakeFrameResult small_improvement =
      remaker.submitFrame(image, 200, "head", {candidate}, &error);
  ASSERT_TRUE(error.empty()) << error;
  EXPECT_FALSE(small_improvement.saved_frame);
  EXPECT_EQ(small_improvement.improved_objects, 0U);

  ASSERT_TRUE(remaker.populateSnapshot(&snapshot,
                                       dir / "exported",
                                       "snapshots/run",
                                       &error))
      << error;
  ASSERT_EQ(snapshot.snapshot_images.size(), 1U);
  ASSERT_TRUE(snapshot.objects.front().snapshot.valid());
  EXPECT_EQ(snapshot.objects.front().snapshot.image_index, 0);
  EXPECT_NEAR(snapshot.objects.front().snapshot.quality, 0.50f, 1.0e-4f);
  EXPECT_EQ(snapshot.snapshot_images.front().uri, "snapshots/run/snapshot_0000.bmp");
  EXPECT_TRUE(std::filesystem::exists(dir / "exported" / "snapshot_0000.bmp"));
}

TEST(ObjectSnapshotRemaker, AcceptsFirstSnapshotBelowReplacementThreshold) {
  const std::filesystem::path dir = makeTempDir("snapshot_remaker_first_snapshot");
  ObjectSnapshotRemakerConfig config;
  config.enabled = true;
  config.staging_dir = dir / "staging";
  config.first_min_quality = 0.05f;
  config.min_quality = 0.50f;
  config.min_box_area_px = 1.0f;
  config.position_weight = 0.5f;
  config.size_weight = 0.5f;
  config.replace_min_quality_delta = 0.12f;
  config.replace_min_quality_ratio = 1.20f;

  ObjectGraphSnapshot snapshot;
  snapshot.objects.push_back(makeObject(5));
  ObjectSnapshotRemaker remaker(config);
  remaker.loadSnapshot(snapshot);

  const ImageBuffer image = makeRgbImage(32, 32);
  ObjectSnapshotRemakeCandidate candidate;
  candidate.object_id = 5;
  candidate.bbox_xyxy = {8.0f, 8.0f, 24.0f, 24.0f};
  candidate.bbox_quality = 0.20f;
  candidate.time_ns = 100;
  candidate.camera_id = "head";

  std::string error;
  ObjectSnapshotRemakeFrameResult first =
      remaker.submitFrame(image, 100, "head", {candidate}, &error);
  ASSERT_TRUE(error.empty()) << error;
  EXPECT_TRUE(first.saved_frame);
  EXPECT_EQ(first.first_snapshot_objects, 1U);
  EXPECT_EQ(first.replaced_objects, 0U);

  candidate.bbox_quality = 0.30f;
  ObjectSnapshotRemakeFrameResult low_replacement =
      remaker.submitFrame(image, 200, "head", {candidate}, &error);
  ASSERT_TRUE(error.empty()) << error;
  EXPECT_FALSE(low_replacement.saved_frame);
  EXPECT_EQ(low_replacement.quality_rejected_candidates, 1U);

  ASSERT_TRUE(remaker.populateSnapshot(&snapshot,
                                       dir / "exported",
                                       "snapshots/run",
                                       &error))
      << error;
  ASSERT_TRUE(snapshot.objects.front().snapshot.valid());
  EXPECT_NEAR(snapshot.objects.front().snapshot.quality, 0.20f, 1.0e-4f);
}

TEST(InstanceMapThread, FrozenSnapshotRemakeMatchesInactivePublishableObjects) {
  const std::filesystem::path dir =
      makeTempDir("snapshot_remake_inactive_publishable");

  PipelineConfig config;
  config.load_scene_graph = true;
  config.freeze_instances = true;
  config.snapshot_remake_enabled = true;
  config.snapshot_staging_dir = (dir / "staging").string();
  config.boxer_input_size = 32;
  config.instance_min_confidence = 0.0f;
  config.instance_min_bbox_size_m = 0.0f;
  config.instance_snapshot_first_min_quality = 0.0f;
  config.instance_snapshot_min_quality = 0.0f;
  config.instance_snapshot_min_box_area_px = 1.0f;

  ThreadSafeQueue<InferenceResponse> queue(4);
  FakeMapProjector projector;
  InstanceMapThread instance_map(queue, projector, config);

  ObjectGraphSnapshot loaded;
  ObjectNode object = makeObject(42);
  object.label = "chair";
  object.semantic_id = 231;
  object.active = false;
  object.publishable = true;
  object.center_world = Eigen::Vector3f(1.0f, 2.0f, 0.5f);
  object.size_m = Eigen::Vector3f(0.6f, 0.7f, 0.8f);
  object.yaw_rad = 0.0f;
  loaded.objects.push_back(object);

  std::string error;
  ASSERT_TRUE(instance_map.loadObjectGraphSnapshot(loaded, &error)) << error;

  InferenceResponse response;
  response.ok = true;
  response.time_ns = 100;
  response.camera_id = "head";
  response.source_rgb_960 = makeRgbImage(32, 32);

  RawDetection detection;
  detection.label = "chair";
  detection.semantic_id = 231;
  detection.center_world = object.center_world;
  detection.size_m = object.size_m;
  detection.yaw_rad = object.yaw_rad;
  detection.score_2d = 1.0f;
  detection.score_3d = 1.0f;
  detection.box_xyxy = {8.0f, 8.0f, 24.0f, 24.0f};
  response.detections.push_back(detection);

  instance_map.start();
  ASSERT_TRUE(instance_map.enqueueDetections(std::move(response)));

  ObjectGraphSnapshot saved;
  bool found_snapshot = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    ASSERT_TRUE(instance_map.prepareSceneGraphForSave(&saved,
                                                      dir / "exported",
                                                      "snapshots/run",
                                                      &error))
        << error;
    if (!saved.objects.empty() && saved.objects.front().snapshot.valid()) {
      found_snapshot = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  instance_map.stop();

  ASSERT_TRUE(found_snapshot);
  ASSERT_EQ(saved.snapshot_images.size(), 1U);
  EXPECT_TRUE(std::filesystem::exists(saved.snapshot_images.front().source_path));
  EXPECT_EQ(saved.objects.front().snapshot.camera_id, "head");
}

TEST(InstanceMapThread,
     PersistenceAdmissionAndTerminalSinkFuseRejectBeforeReducer) {
  PipelineConfig config;
  ThreadSafeQueue<InferenceResponse> queue(4);
  FakeMapProjector projector;
  InstanceMapThread instance_map(queue, projector, config);

  ObjectGraphSnapshot loaded;
  loaded.objects.push_back(makeObject(7));
  std::string error;
  ASSERT_TRUE(instance_map.loadObjectGraphSnapshot(loaded, &error)) << error;
  const SceneRevision loaded_revision = instance_map.sceneSnapshot().revision();

  std::atomic_bool admission_allowed{false};
  std::atomic_int sink_calls{0};
  instance_map.setSceneContentAdmission(
      [&admission_allowed]() { return admission_allowed.load(); });
  instance_map.setSceneCommitSink(
      [&sink_calls](const SceneApplyResult&) {
        ++sink_calls;
        return false;
      });
  instance_map.start();

  ApplyHumanAnnotationCommand blocked;
  blocked.object_id = 7;
  blocked.patch.label = "blocked";
  const auto blocked_result = instance_map.applySceneCommandAndWait(
      SceneCommand{blocked}, std::chrono::seconds(1));
  ASSERT_TRUE(blocked_result.has_value());
  EXPECT_EQ(blocked_result->status, SceneApplyStatus::kRejected);
  EXPECT_NE(blocked_result->reason.find("persistence admission"),
            std::string::npos);
  EXPECT_EQ(blocked_result->revision, loaded_revision);
  EXPECT_EQ(sink_calls.load(), 0);

  admission_allowed.store(true);
  ApplyHumanAnnotationCommand first = blocked;
  first.patch.label = "committed once";
  const auto committed = instance_map.applySceneCommandAndWait(
      SceneCommand{first}, std::chrono::seconds(1));
  ASSERT_TRUE(committed.has_value());
  ASSERT_TRUE(committed->committedRevision()) << committed->reason;
  EXPECT_EQ(committed->revision, loaded_revision + 1);
  EXPECT_EQ(sink_calls.load(), 1);

  ApplyHumanAnnotationCommand after_fuse = blocked;
  after_fuse.patch.label = "must not commit";
  const auto rejected = instance_map.applySceneCommandAndWait(
      SceneCommand{after_fuse}, std::chrono::seconds(1));
  ASSERT_TRUE(rejected.has_value());
  EXPECT_EQ(rejected->status, SceneApplyStatus::kRejected);
  EXPECT_NE(rejected->reason.find("commit fuse"), std::string::npos);
  EXPECT_EQ(rejected->revision, committed->revision);
  EXPECT_EQ(sink_calls.load(), 1);

  AdvanceSurfaceCommand advance;
  advance.surface.map_epoch = RunId{4, 5};
  advance.surface.surface_revision = 1;
  advance.surface.source_map_revision = 1;
  const auto metadata = instance_map.applySceneCommandAndWait(
      SceneCommand{advance}, std::chrono::seconds(1));
  ASSERT_TRUE(metadata.has_value());
  EXPECT_EQ(metadata->status, SceneApplyStatus::kMetadataUpdated);
  EXPECT_EQ(metadata->revision, committed->revision);
  EXPECT_EQ(sink_calls.load(), 1);

  instance_map.stop();
}

}  // namespace
}  // namespace roomie
