#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "roomie/scene/scene_state.hpp"

namespace roomie {

enum class FurnitureSupportMode {
  kTop,
  kUpperVolume,
};

struct FurnitureClassRule {
  FurnitureSupportMode support_mode = FurnitureSupportMode::kTop;
  bool can_contain = false;
};

struct FurnitureGraphConfig {
  std::map<std::string, FurnitureClassRule> classes;
  std::size_t object_creation_detection_window_frames = 30;
  std::size_t object_creation_min_same_class_detection_frames = 11;
  float on_min_horizontal_overlap_ratio = 0.50f;
  float on_max_gap_m = 0.12f;
  float on_max_penetration_m = 0.15f;
  float in_min_child_containment_ratio = 0.80f;
  float in_max_child_parent_volume_ratio = 0.80f;
};

// Normalizes detector/human labels for exact furniture-class lookup.
std::string normalizeFurnitureLabel(std::string label);

FurnitureGraphConfig defaultFurnitureGraphConfig();
FurnitureGraphConfig loadFurnitureGraphConfig(
    const std::filesystem::path& path);

// Rebuilds the role projection from effective object labels. Unchanged roles
// retain their role revision; classification changes advance it.
std::vector<FurnitureRole> classifyFurnitureRoles(
    const SceneObjectTable& objects,
    const std::vector<FurnitureRole>& previous,
    const FurnitureGraphConfig& config);

// Derives only furniture-owned relations: room_contains_furniture plus one
// best in/on furniture parent per eligible object. room_contains_object is
// maintained independently by the existing reducer path.
std::vector<SceneRelation> deriveFurnitureRelations(
    const SceneObjectTable& objects,
    const std::vector<RoomNode, Eigen::aligned_allocator<RoomNode>>& rooms,
    const std::vector<FurnitureRole>& roles,
    const FurnitureGraphConfig& config,
    SceneRevision revision);

bool isDerivedFurnitureRelation(const SceneRelation& relation);

}  // namespace roomie
