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
  EXPECT_EQ(saved.value("format_version", 0), 6);
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
  EXPECT_TRUE(loaded_v1.objects.front().name.empty());
  EXPECT_TRUE(loaded_v1.objects.front().description.empty());
  loaded_v1.objects.front().name = "blue cup by the sink";

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
  loaded_v1.furniture.push_back(FurnitureRole{1, 2, "chair"});
  ObjectRelation containment;
  setRelationEndpoints(
      &containment, SceneEntityRef{SceneEntityType::kRoom, 9},
      SceneEntityRef{SceneEntityType::kObject, 1});
  containment.relation_type = "room_contains_object";
  containment.confidence = 1.0f;
  containment.revision = 5;
  containment.derived = true;
  loaded_v1.relations.push_back(containment);
  ObjectRelation room_furniture;
  setRelationEndpoints(
      &room_furniture, SceneEntityRef{SceneEntityType::kRoom, 9},
      SceneEntityRef{SceneEntityType::kFurniture, 1});
  room_furniture.relation_type = "room_contains_furniture";
  room_furniture.confidence = 1.0f;
  room_furniture.revision = 5;
  room_furniture.derived = true;
  loaded_v1.relations.push_back(room_furniture);

  const std::filesystem::path v3_path = dir / "canonical_v4.json";
  ASSERT_TRUE(saveObjectGraphSnapshotJsonAtomic(
      loaded_v1, "map", 456, v3_path, &error)) << error;
  std::ifstream stream(v3_path);
  const nlohmann::json v3 = nlohmann::json::parse(stream);
  EXPECT_EQ(v3.value("format_version", 0), 6);
  ASSERT_EQ(v3.at("objects").size(), 1U);
  EXPECT_EQ(v3.at("objects").at(0).at("name"),
            "blue cup by the sink");
  ASSERT_EQ(v3.at("rooms").size(), 1U);
  ASSERT_EQ(v3.at("furniture").size(), 1U);
  ASSERT_EQ(v3.at("relations").size(), 2U);
  EXPECT_EQ(v3.at("relations").at(0).at("source").value(
                "type", std::string()), "room");
  EXPECT_TRUE(v3.at("objects").at(0).value(
                  "description", std::string("unexpected")).empty());

  ObjectGraphSnapshot loaded_v3;
  ASSERT_TRUE(loadObjectGraphSnapshotJson(
      v3_path, &loaded_v3, &world_frame, &error)) << error;
  ASSERT_EQ(loaded_v3.objects.size(), 1U);
  EXPECT_EQ(loaded_v3.objects.front().name, "blue cup by the sink");
  ASSERT_EQ(loaded_v3.rooms.size(), 1U);
  ASSERT_EQ(loaded_v3.furniture.size(), 1U);
  EXPECT_EQ(loaded_v3.furniture.front().object_id, 1);
  EXPECT_EQ(loaded_v3.furniture.front().classification_label, "chair");
  ASSERT_EQ(loaded_v3.relations.size(), 2U);
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
     EvidenceModeConfirmsArchivesAndReactivatesWithTheSameObjectId) {
  PipelineConfig config;
  config.instance_association_mode = "evidence";
  config.instance_min_confidence = 0.0f;
  config.instance_min_bbox_size_m = 0.001f;
  config.instance_max_bbox_size_m = 10.0f;
  config.instance_presence_min_depth_samples = 8;
  config.instance_presence_window_sec = 2.0;

  ThreadSafeQueue<InferenceResponse> queue(16);
  FakeMapProjector projector;
  InstanceMapThread instance_map(queue, projector, config);
  instance_map.start();

  const RunId run{100, 200};
  const auto make_response = [&](FrameId frame_id, bool detected,
                                 bool with_depth, float camera_x = 0.0f) {
    InferenceResponse response;
    response.ok = true;
    response.time_ns = static_cast<TimeNanoseconds>(frame_id) * 100'000'000LL;
    response.camera_id = "head";
    response.provenance.run_id = run;
    response.provenance.frame_id = frame_id;
    response.provenance.request_id = frame_id;
    response.provenance.sensor_time_ns = response.time_ns;
    response.has_camera_pose = true;
    response.T_world_camera = Eigen::Isometry3f::Identity();
    response.T_world_camera.translation().x() = camera_x;
    if (detected) {
      RawDetection detection;
      detection.label = "backpack";
      detection.semantic_id = 7;
      detection.center_world = {0.0f, 0.0f, 2.0f};
      detection.size_m = {0.4f, 0.4f, 0.4f};
      detection.score_2d = 0.9f;
      detection.score_3d = 0.9f;
      detection.box_xyxy = {400.0f, 400.0f, 560.0f, 560.0f};
      response.detections.push_back(std::move(detection));
    }
    if (with_depth) {
      auto depth = std::make_shared<DepthBuffer>();
      depth->width = 100;
      depth->height = 100;
      depth->depth_m.assign(10'000, 4.0f);
      auto visibility = std::make_shared<VisibilityContext>();
      visibility->depth = std::move(depth);
      visibility->intrinsics = CameraIntrinsics{100, 100, 100.0f, 100.0f,
                                                50.0f, 50.0f};
      response.visibility_context = std::move(visibility);
    }
    return response;
  };

  ASSERT_TRUE(instance_map.enqueueDetections(make_response(1, true, false)));
  ASSERT_TRUE(instance_map.waitUntilIdle(std::chrono::seconds(1)));
  EXPECT_TRUE(instance_map.snapshotInstances().empty());
  ASSERT_EQ(instance_map.snapshotTrackedInstances().size(), 1U);
  EXPECT_EQ(instance_map.snapshotTrackedInstances().front().presence_state,
            "tentative");

  ASSERT_TRUE(instance_map.enqueueDetections(make_response(2, true, false)));
  ASSERT_TRUE(instance_map.waitUntilIdle(std::chrono::seconds(1)));
  ASSERT_EQ(instance_map.snapshotInstances().size(), 1U);
  const int original_object_id =
      instance_map.snapshotInstances().front().object_id;

  for (FrameId frame_id = 3; frame_id <= 5; ++frame_id) {
    ASSERT_TRUE(
        instance_map.enqueueDetections(make_response(frame_id, false, true)));
    ASSERT_TRUE(instance_map.waitUntilIdle(std::chrono::seconds(1)));
  }
  EXPECT_TRUE(instance_map.snapshotInstances().empty());
  ASSERT_EQ(instance_map.snapshotTrackedInstances().size(), 1U);
  EXPECT_EQ(instance_map.snapshotTrackedInstances().front().presence_state,
            "archived");
  EXPECT_TRUE(instance_map.sceneSnapshot().findObject(original_object_id));

  ASSERT_TRUE(instance_map.enqueueDetections(make_response(6, true, false)));
  ASSERT_TRUE(instance_map.waitUntilIdle(std::chrono::seconds(1)));
  EXPECT_TRUE(instance_map.snapshotInstances().empty());
  ASSERT_TRUE(instance_map.enqueueDetections(make_response(7, true, false)));
  ASSERT_TRUE(instance_map.waitUntilIdle(std::chrono::seconds(1)));
  ASSERT_EQ(instance_map.snapshotInstances().size(), 1U);
  EXPECT_EQ(instance_map.snapshotInstances().front().object_id,
            original_object_id);
  instance_map.stop();
}

TEST(InstanceMapThread, EvidenceModeDurablyMergesLegacyBackpackHandbagPair) {
  PipelineConfig config;
  config.instance_association_mode = "evidence";
  config.instance_min_confidence = 0.0f;
  config.instance_min_bbox_size_m = 0.001f;

  ThreadSafeQueue<InferenceResponse> queue(8);
  FakeMapProjector projector;
  InstanceMapThread instance_map(queue, projector, config);
  ObjectGraphSnapshot loaded;
  ObjectNode backpack = makeObject(76);
  backpack.label = "backpack";
  backpack.semantic_id = 1;
  backpack.center_world = {2.415f, -0.731f, 0.833f};
  backpack.size_m = {0.46f, 0.375f, 0.462f};
  backpack.yaw_rad = -1.492f;
  backpack.existence_log_odds = 2.0f;
  ObjectNode handbag = makeObject(99);
  handbag.label = "handbag";
  handbag.semantic_id = 2;
  handbag.center_world = {2.412f, -0.764f, 0.796f};
  handbag.size_m = {0.395f, 0.312f, 0.472f};
  handbag.yaw_rad = -1.477f;
  handbag.existence_log_odds = 2.0f;
  loaded.objects = {backpack, handbag};
  loaded.next_object_id = 100;
  std::string error;
  ASSERT_TRUE(instance_map.loadObjectGraphSnapshot(loaded, &error)) << error;
  instance_map.start();

  for (FrameId frame_id = 1; frame_id <= 2; ++frame_id) {
    InferenceResponse response;
    response.ok = true;
    response.time_ns = static_cast<TimeNanoseconds>(frame_id) * 100'000'000LL;
    response.camera_id = "head";
    response.provenance.run_id = RunId{300, 400};
    response.provenance.frame_id = frame_id;
    response.provenance.request_id = frame_id;
    RawDetection detection;
    detection.label = "backpack";
    detection.semantic_id = 1;
    detection.center_world = backpack.center_world;
    detection.size_m = backpack.size_m;
    detection.yaw_rad = backpack.yaw_rad;
    detection.score_2d = 0.9f;
    detection.score_3d = 0.9f;
    detection.box_xyxy = {100, 100, 300, 400};
    response.detections.push_back(std::move(detection));
    ASSERT_TRUE(instance_map.enqueueDetections(std::move(response)));
    ASSERT_TRUE(instance_map.waitUntilIdle(std::chrono::seconds(1)));
  }

  const SceneSnapshot snapshot = instance_map.sceneSnapshot();
  EXPECT_EQ(snapshot.objects().size(), 1U);
  EXPECT_EQ(snapshot.resolveCanonicalId(99),
            std::optional<SceneObjectId>(76));
  EXPECT_TRUE(snapshot.findExactObject(76));
  EXPECT_FALSE(snapshot.findExactObject(99));
  instance_map.stop();
}

TEST(InstanceMapThread,
     EvidenceModeMergesMeasuredBottleTripleButKeepsBottleCapSeparate) {
  PipelineConfig config;
  config.instance_association_mode = "evidence";
  config.instance_min_confidence = 0.0f;
  config.instance_min_bbox_size_m = 0.001f;
  config.instance_small_object_identity_groups = {"bottle|bottled_water"};

  ThreadSafeQueue<InferenceResponse> queue(8);
  FakeMapProjector projector;
  InstanceMapThread instance_map(queue, projector, config);
  const auto bottle = [](int object_id,
                         std::string label,
                         Eigen::Vector3f center,
                         Eigen::Vector3f size,
                         float quality) {
    ObjectNode object = makeObject(object_id);
    object.label = std::move(label);
    object.center_world = center;
    object.size_m = size;
    object.confidence = quality;
    object.confidence_mass = quality;
    object.object_quality_score = quality;
    object.support_count = 1;
    object.label_weights[object.label] = 1.0f;
    return object;
  };

  ObjectGraphSnapshot loaded;
  loaded.objects = {
      bottle(87, "bottle", {0.866043f, -2.538530f, 0.781970f},
             {0.057092f, 0.057383f, 0.130203f}, 0.90f),
      bottle(90, "bottle", {0.884047f, -2.594693f, 0.802578f},
             {0.064928f, 0.063798f, 0.174672f}, 0.80f),
      bottle(95, "bottled_water", {0.861847f, -2.563005f, 0.757086f},
             {0.061129f, 0.051063f, 0.092013f}, 0.70f),
      bottle(101, "bottle_cap", {0.866043f, -2.538530f, 0.781970f},
             {0.057092f, 0.057383f, 0.130203f}, 0.60f)};
  loaded.next_object_id = 102;
  std::string error;
  ASSERT_TRUE(instance_map.loadObjectGraphSnapshot(loaded, &error)) << error;
  instance_map.start();

  InferenceResponse response;
  response.ok = true;
  response.time_ns = 1'000'000'000;
  response.camera_id = "head";
  response.provenance.run_id = RunId{450, 550};
  response.provenance.frame_id = 1;
  response.provenance.request_id = 1;
  ASSERT_TRUE(instance_map.enqueueDetections(std::move(response)));
  ASSERT_TRUE(instance_map.waitUntilIdle(std::chrono::seconds(1)));

  const SceneSnapshot snapshot = instance_map.sceneSnapshot();
  ASSERT_EQ(snapshot.objects().size(), 2U);
  EXPECT_EQ(snapshot.resolveCanonicalId(90),
            std::optional<SceneObjectId>(87));
  EXPECT_EQ(snapshot.resolveCanonicalId(95),
            std::optional<SceneObjectId>(87));
  EXPECT_TRUE(snapshot.findExactObject(87));
  EXPECT_TRUE(snapshot.findExactObject(101));
  const SceneObjectPtr canonical = snapshot.findExactObject(87);
  ASSERT_TRUE(canonical && canonical->semantic);
  EXPECT_GT(canonical->semantic->label_weights.at("bottle"), 0.0f);
  EXPECT_GT(canonical->semantic->label_weights.at("bottled_water"), 0.0f);
  instance_map.stop();
}

TEST(InstanceMapThread,
     EvidenceModeMergesTransitivePackageFamilyAndPreservesLabelEvidence) {
  PipelineConfig config;
  config.instance_association_mode = "evidence";
  config.instance_min_confidence = 0.0f;
  config.instance_min_bbox_size_m = 0.001f;
  config.instance_small_object_identity_groups = {
      "box|labeled_package|printed_carton|medicine_carton"};

  ThreadSafeQueue<InferenceResponse> queue(8);
  FakeMapProjector projector;
  InstanceMapThread instance_map(queue, projector, config);
  const auto package = [](int object_id,
                          std::string label,
                          Eigen::Vector3f center,
                          Eigen::Vector3f size,
                          float yaw,
                          float quality) {
    ObjectNode object = makeObject(object_id);
    object.label = std::move(label);
    object.center_world = center;
    object.size_m = size;
    object.yaw_rad = yaw;
    object.confidence = quality;
    object.confidence_mass = quality;
    object.object_quality_score = quality;
    object.support_count = 1;
    object.label_weights[object.label] = 1.0f;
    return object;
  };

  ObjectGraphSnapshot loaded;
  loaded.objects = {
      package(42, "box", {2.162210f, 3.844470f, 0.828921f},
              {0.211200f, 0.185070f, 0.264856f}, -0.011774f, 0.50f),
      package(50, "labeled_package", {2.208026f, 3.920180f, 0.841667f},
              {0.219637f, 0.095061f, 0.265831f}, -0.606546f, 0.70f),
      package(55, "medicine_carton", {2.178208f, 3.908470f, 0.766520f},
              {0.253268f, 0.158681f, 0.084193f}, 0.008728f, 0.90f)};
  loaded.next_object_id = 56;
  std::string error;
  ASSERT_TRUE(instance_map.loadObjectGraphSnapshot(loaded, &error)) << error;
  instance_map.start();

  InferenceResponse response;
  response.ok = true;
  response.time_ns = 1'000'000'000;
  response.camera_id = "head";
  response.provenance.run_id = RunId{500, 600};
  response.provenance.frame_id = 1;
  response.provenance.request_id = 1;
  ASSERT_TRUE(instance_map.enqueueDetections(std::move(response)));
  ASSERT_TRUE(instance_map.waitUntilIdle(std::chrono::seconds(1)));

  const SceneSnapshot snapshot = instance_map.sceneSnapshot();
  ASSERT_EQ(snapshot.objects().size(), 1U);
  EXPECT_EQ(snapshot.resolveCanonicalId(50),
            std::optional<SceneObjectId>(42));
  EXPECT_EQ(snapshot.resolveCanonicalId(55),
            std::optional<SceneObjectId>(42));
  const SceneObjectPtr canonical = snapshot.findExactObject(42);
  ASSERT_TRUE(canonical);
  ASSERT_TRUE(canonical->semantic);
  EXPECT_GT(canonical->semantic->label_weights.at("box"), 0.0f);
  EXPECT_GT(canonical->semantic->label_weights.at("labeled_package"), 0.0f);
  EXPECT_GT(canonical->semantic->label_weights.at("medicine_carton"), 0.0f);
  ASSERT_TRUE(canonical->geometry);
  EXPECT_TRUE(canonical->geometry->center_world.isApprox(
      Eigen::Vector3f(2.178208f, 3.908470f, 0.766520f), 1.0e-5f));
  instance_map.stop();
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
