#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "roomie/scene/furniture_graph.hpp"

namespace roomie {
namespace {

class TemporaryFurnitureConfig {
 public:
  explicit TemporaryFurnitureConfig(const std::string& contents) {
    char pattern[] = "/tmp/roomie_furniture_config_XXXXXX";
    const int descriptor = mkstemp(pattern);
    if (descriptor < 0) {
      throw std::runtime_error("mkstemp failed");
    }
    close(descriptor);
    path_ = pattern;
    std::ofstream stream(path_);
    stream << contents;
    if (!stream) {
      throw std::runtime_error("failed to write temporary furniture config");
    }
  }

  ~TemporaryFurnitureConfig() { std::remove(path_.c_str()); }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

std::string minimalFurnitureConfig(const std::string& object_creation = {}) {
  return std::string(R"({
    "schema_version": "roomie.furniture.v1",
    "classes": [
      {"label": "desk", "support_mode": "top", "can_contain": false}
    ],
  )") + object_creation + R"(
    "on": {
      "min_horizontal_overlap_ratio": 0.5,
      "max_gap_m": 0.12,
      "max_penetration_m": 0.15
    },
    "in": {
      "min_child_containment_ratio": 0.8,
      "max_child_parent_volume_ratio": 0.8
    }
  })";
}

SceneObjectPtr makeFurnitureTestObject(
    int object_id, std::string label, const Eigen::Vector3f& center,
    const Eigen::Vector3f& size,
    std::optional<std::string> label_override = std::nullopt) {
  auto object = std::make_shared<SceneObject>();
  auto identity = std::make_shared<IdentityComponent>();
  identity->object_id = object_id;
  identity->revision = 1;
  object->identity = std::move(identity);
  auto lifecycle = std::make_shared<LifecycleComponent>();
  lifecycle->revision = 1;
  lifecycle->active = true;
  lifecycle->publishable = true;
  lifecycle->track_state = InstanceTrackState::kStable;
  object->lifecycle = std::move(lifecycle);
  auto geometry = std::make_shared<GeometryComponent>();
  geometry->revision = 1;
  geometry->obb_revision = 1;
  geometry->center_world = center;
  geometry->size_m = size;
  geometry->status = InstanceGeometryStatus::kGood;
  object->geometry = std::move(geometry);
  auto semantic = std::make_shared<SemanticComponent>();
  semantic->revision = 1;
  semantic->label = std::move(label);
  semantic->confidence = 0.9f;
  object->semantic = std::move(semantic);
  auto annotation = std::make_shared<AnnotationComponent>();
  annotation->revision = 1;
  annotation->label_override = std::move(label_override);
  object->annotation = std::move(annotation);
  return object;
}

TEST(FurnitureGraph, NormalizesAndClassifiesExactConfiguredLabels) {
  EXPECT_EQ(normalizeFurnitureLabel("  Night-Stand  "), "night_stand");
  EXPECT_EQ(normalizeFurnitureLabel("DINING__TABLE"), "dining_table");

  SceneObjectTable objects;
  objects.emplace(1, makeFurnitureTestObject(
                         1, "not furniture", Eigen::Vector3f::Zero(),
                         Eigen::Vector3f::Ones(), "Chair"));
  objects.emplace(2, makeFurnitureTestObject(
                         2, "office chair", Eigen::Vector3f::Zero(),
                         Eigen::Vector3f::Ones()));
  const std::vector<FurnitureRole> roles = classifyFurnitureRoles(
      objects, {FurnitureRole{1, 7, "chair"}},
      defaultFurnitureGraphConfig());
  ASSERT_EQ(roles.size(), 1U);
  EXPECT_EQ(roles.front().object_id, 1);
  EXPECT_EQ(roles.front().classification_label, "chair");
  EXPECT_EQ(roles.front().revision, 7U);
}

TEST(FurnitureGraph, LoadsObjectCreationThresholdsFromConfig) {
  const FurnitureGraphConfig config = loadFurnitureGraphConfig(
      std::filesystem::path(ROOMIE_SOURCE_DIR) / "config" / "scene_qa" /
      "furniture.json");
  EXPECT_EQ(config.object_creation_detection_window_frames, 30U);
  EXPECT_EQ(config.object_creation_min_same_class_detection_frames, 11U);
}

TEST(FurnitureGraph, DefaultsObjectCreationThresholdsForLegacyConfig) {
  const TemporaryFurnitureConfig file(minimalFurnitureConfig());
  const FurnitureGraphConfig config = loadFurnitureGraphConfig(file.path());
  EXPECT_EQ(config.object_creation_detection_window_frames, 30U);
  EXPECT_EQ(config.object_creation_min_same_class_detection_frames, 11U);
}

TEST(FurnitureGraph, RejectsObjectCreationMinimumAboveWindow) {
  const TemporaryFurnitureConfig file(minimalFurnitureConfig(R"(
    "object_creation": {
      "detection_window_frames": 30,
      "min_same_class_detection_frames": 31
    },
  )"));
  EXPECT_THROW(loadFurnitureGraphConfig(file.path()), std::invalid_argument);
}

TEST(FurnitureGraph, DerivesRoomAndSingleBestOnOrInParent) {
  SceneObjectTable objects;
  objects.emplace(1, makeFurnitureTestObject(
                         1, "table", Eigen::Vector3f(0.0f, 0.0f, 0.5f),
                         Eigen::Vector3f(1.0f, 1.0f, 1.0f)));
  objects.emplace(2, makeFurnitureTestObject(
                         2, "cabinet", Eigen::Vector3f(3.0f, 0.0f, 1.0f),
                         Eigen::Vector3f(2.0f, 2.0f, 2.0f)));
  objects.emplace(3, makeFurnitureTestObject(
                         3, "cup", Eigen::Vector3f(0.0f, 0.0f, 1.1f),
                         Eigen::Vector3f(0.2f, 0.2f, 0.2f)));
  objects.emplace(4, makeFurnitureTestObject(
                         4, "box", Eigen::Vector3f(3.0f, 0.0f, 1.0f),
                         Eigen::Vector3f(0.4f, 0.4f, 0.4f)));
  // A larger, otherwise equivalent support must lose the deterministic tie
  // to the tighter table parent.
  objects.emplace(5, makeFurnitureTestObject(
                         5, "table", Eigen::Vector3f(0.0f, 0.0f, 0.5f),
                         Eigen::Vector3f(2.0f, 2.0f, 1.0f)));

  std::vector<RoomNode, Eigen::aligned_allocator<RoomNode>> rooms(1);
  rooms.front().room_id = 10;
  rooms.front().has_xy_bounds = true;
  rooms.front().min_xy = {-5.0f, -5.0f};
  rooms.front().max_xy = {5.0f, 5.0f};
  const std::vector<FurnitureRole> roles = classifyFurnitureRoles(
      objects, {}, defaultFurnitureGraphConfig());
  ASSERT_EQ(roles.size(), 3U);

  const std::vector<SceneRelation> relations = deriveFurnitureRelations(
      objects, rooms, roles, defaultFurnitureGraphConfig(), 42);
  EXPECT_EQ(std::count_if(
                relations.begin(), relations.end(), [](const auto& relation) {
                  return relation.relation_type ==
                         "room_contains_furniture";
                }),
            3);

  const auto on = std::find_if(
      relations.begin(), relations.end(), [](const auto& relation) {
        return relation.relation_type == "on" &&
               relationSource(relation).id == 3;
      });
  ASSERT_NE(on, relations.end());
  EXPECT_EQ(relationSource(*on).type, SceneEntityType::kObject);
  EXPECT_EQ(relationTarget(*on),
            (SceneEntityRef{SceneEntityType::kFurniture, 1}));

  const auto inside = std::find_if(
      relations.begin(), relations.end(), [](const auto& relation) {
        return relation.relation_type == "in" &&
               relationSource(relation).id == 4;
      });
  ASSERT_NE(inside, relations.end());
  EXPECT_EQ(relationTarget(*inside),
            (SceneEntityRef{SceneEntityType::kFurniture, 2}));

  EXPECT_EQ(std::count_if(relations.begin(), relations.end(),
                          [](const auto& relation) {
                            return (relation.relation_type == "in" ||
                                    relation.relation_type == "on") &&
                                   relationSource(relation).id == 3;
                          }),
            1);
}

}  // namespace
}  // namespace roomie
