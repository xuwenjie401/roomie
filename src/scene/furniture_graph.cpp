#include "roomie/scene/furniture_graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>

#include <nlohmann/json.hpp>

namespace roomie {
namespace {

using Json = nlohmann::json;
using Polygon2 =
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>;

constexpr float kEpsilon = 1.0e-6f;

float clamp01(float value) {
  return std::max(0.0f, std::min(1.0f, value));
}

bool finiteGeometry(const GeometryComponent& geometry) {
  return geometry.center_world.allFinite() && geometry.size_m.allFinite() &&
         std::isfinite(geometry.yaw_rad) &&
         (geometry.size_m.array() > 0.0f).all();
}

float boxVolume(const GeometryComponent& geometry) {
  return geometry.size_m.x() * geometry.size_m.y() * geometry.size_m.z();
}

Polygon2 yawRectCorners2d(const GeometryComponent& geometry) {
  const Eigen::Vector2f half(0.5f * geometry.size_m.x(),
                             0.5f * geometry.size_m.y());
  const float c = std::cos(geometry.yaw_rad);
  const float s = std::sin(geometry.yaw_rad);
  const Eigen::Matrix2f rotation =
      (Eigen::Matrix2f() << c, -s, s, c).finished();
  const Eigen::Vector2f center(geometry.center_world.x(),
                               geometry.center_world.y());
  Polygon2 corners;
  corners.reserve(4);
  corners.push_back(center + rotation * Eigen::Vector2f(-half.x(), -half.y()));
  corners.push_back(center + rotation * Eigen::Vector2f(half.x(), -half.y()));
  corners.push_back(center + rotation * Eigen::Vector2f(half.x(), half.y()));
  corners.push_back(center + rotation * Eigen::Vector2f(-half.x(), half.y()));
  return corners;
}

float cross2d(const Eigen::Vector2f& lhs, const Eigen::Vector2f& rhs) {
  return lhs.x() * rhs.y() - lhs.y() * rhs.x();
}

bool insideClipEdge(const Eigen::Vector2f& point,
                    const Eigen::Vector2f& edge_start,
                    const Eigen::Vector2f& edge_end) {
  return cross2d(edge_end - edge_start, point - edge_start) >= -1.0e-5f;
}

Eigen::Vector2f lineIntersection(const Eigen::Vector2f& p0,
                                 const Eigen::Vector2f& p1,
                                 const Eigen::Vector2f& q0,
                                 const Eigen::Vector2f& q1) {
  const Eigen::Vector2f r = p1 - p0;
  const Eigen::Vector2f s = q1 - q0;
  const float denominator = cross2d(r, s);
  if (std::abs(denominator) < kEpsilon) {
    return p1;
  }
  return p0 + (cross2d(q0 - p0, s) / denominator) * r;
}

Polygon2 clipPolygon(const Polygon2& subject, const Polygon2& clipper) {
  Polygon2 output = subject;
  for (std::size_t edge = 0; edge < clipper.size(); ++edge) {
    if (output.empty()) {
      break;
    }
    const Eigen::Vector2f edge_start = clipper[edge];
    const Eigen::Vector2f edge_end = clipper[(edge + 1U) % clipper.size()];
    const Polygon2 input = output;
    output.clear();
    Eigen::Vector2f previous = input.back();
    bool previous_inside = insideClipEdge(previous, edge_start, edge_end);
    for (const Eigen::Vector2f& current : input) {
      const bool current_inside =
          insideClipEdge(current, edge_start, edge_end);
      if (current_inside != previous_inside) {
        output.push_back(
            lineIntersection(previous, current, edge_start, edge_end));
      }
      if (current_inside) {
        output.push_back(current);
      }
      previous = current;
      previous_inside = current_inside;
    }
  }
  return output;
}

float polygonArea(const Polygon2& polygon) {
  if (polygon.size() < 3U) {
    return 0.0f;
  }
  float twice_area = 0.0f;
  for (std::size_t index = 0; index < polygon.size(); ++index) {
    twice_area +=
        cross2d(polygon[index], polygon[(index + 1U) % polygon.size()]);
  }
  return 0.5f * std::abs(twice_area);
}

float horizontalIntersectionArea(const GeometryComponent& child,
                                 const GeometryComponent& parent) {
  return polygonArea(
      clipPolygon(yawRectCorners2d(child), yawRectCorners2d(parent)));
}

float horizontalChildCoverage(const GeometryComponent& child,
                              const GeometryComponent& parent) {
  const float child_area = child.size_m.x() * child.size_m.y();
  return child_area > kEpsilon
             ? clamp01(horizontalIntersectionArea(child, parent) / child_area)
             : 0.0f;
}

float childVolumeContainment(const GeometryComponent& child,
                             const GeometryComponent& parent) {
  const float child_volume = boxVolume(child);
  if (child_volume <= kEpsilon) {
    return 0.0f;
  }
  const float child_min_z = child.center_world.z() - 0.5f * child.size_m.z();
  const float child_max_z = child.center_world.z() + 0.5f * child.size_m.z();
  const float parent_min_z =
      parent.center_world.z() - 0.5f * parent.size_m.z();
  const float parent_max_z =
      parent.center_world.z() + 0.5f * parent.size_m.z();
  const float z_overlap =
      std::max(0.0f, std::min(child_max_z, parent_max_z) -
                         std::max(child_min_z, parent_min_z));
  return clamp01(horizontalIntersectionArea(child, parent) * z_overlap /
                 child_volume);
}

bool eligibleObject(const SceneObjectPtr& object) {
  return object && object->identity && object->lifecycle && object->geometry &&
         object->semantic && object->annotation &&
         object->identity->object_id >= 0 && object->lifecycle->publishable;
}

std::string effectiveLabel(const SceneObject& object) {
  if (object.annotation && object.annotation->label_override) {
    return *object.annotation->label_override;
  }
  return object.semantic ? object.semantic->label : std::string();
}

std::array<float, 2> roomMin(const RoomNode& room) {
  if (room.has_xy_bounds) {
    return room.min_xy;
  }
  return {room.center_world.x() - 0.5f * room.size_m.x(),
          room.center_world.y() - 0.5f * room.size_m.y()};
}

std::array<float, 2> roomMax(const RoomNode& room) {
  if (room.has_xy_bounds) {
    return room.max_xy;
  }
  return {room.center_world.x() + 0.5f * room.size_m.x(),
          room.center_world.y() + 0.5f * room.size_m.y()};
}

bool roomContainsCenter(const RoomNode& room,
                        const GeometryComponent& geometry) {
  if (room.room_id < 0 || !geometry.center_world.allFinite()) {
    return false;
  }
  if (!room.has_xy_bounds &&
      (room.size_m.x() <= 0.0f || room.size_m.y() <= 0.0f)) {
    return false;
  }
  const auto minimum = roomMin(room);
  const auto maximum = roomMax(room);
  return std::isfinite(minimum[0]) && std::isfinite(minimum[1]) &&
         std::isfinite(maximum[0]) && std::isfinite(maximum[1]) &&
         geometry.center_world.x() >= minimum[0] &&
         geometry.center_world.x() <= maximum[0] &&
         geometry.center_world.y() >= minimum[1] &&
         geometry.center_world.y() <= maximum[1];
}

struct ParentCandidate {
  int parent_id = -1;
  std::string relation_type;
  float confidence = 0.0f;
  float parent_volume = 0.0f;
  float vertical_error = 0.0f;
};

std::optional<ParentCandidate> onCandidate(
    int parent_id,
    const GeometryComponent& child,
    const GeometryComponent& parent,
    const FurnitureClassRule& rule,
    const FurnitureGraphConfig& config) {
  const float coverage = horizontalChildCoverage(child, parent);
  if (coverage < config.on_min_horizontal_overlap_ratio) {
    return std::nullopt;
  }
  const float child_bottom =
      child.center_world.z() - 0.5f * child.size_m.z();
  const float parent_bottom =
      parent.center_world.z() - 0.5f * parent.size_m.z();
  const float parent_top =
      parent.center_world.z() + 0.5f * parent.size_m.z();
  float vertical_error = 0.0f;
  float vertical_score = 0.0f;
  if (rule.support_mode == FurnitureSupportMode::kTop) {
    vertical_error = child_bottom - parent_top;
    if (vertical_error < -config.on_max_penetration_m ||
        vertical_error > config.on_max_gap_m) {
      return std::nullopt;
    }
    const float scale = vertical_error < 0.0f
                            ? config.on_max_penetration_m
                            : config.on_max_gap_m;
    vertical_score =
        1.0f - std::abs(vertical_error) / std::max(kEpsilon, scale);
  } else {
    if (child.center_world.z() < parent.center_world.z() ||
        child_bottom < parent_bottom ||
        child_bottom > parent_top + config.on_max_gap_m) {
      return std::nullopt;
    }
    vertical_error = child_bottom - parent.center_world.z();
    const float height_position =
        (child_bottom - parent_bottom) / parent.size_m.z();
    vertical_score = clamp01(height_position * 2.0f);
  }
  return ParentCandidate{parent_id, "on",
                         clamp01(0.70f * coverage +
                                 0.30f * vertical_score),
                         boxVolume(parent), std::abs(vertical_error)};
}

std::optional<ParentCandidate> inCandidate(
    int parent_id,
    const GeometryComponent& child,
    const GeometryComponent& parent,
    const FurnitureClassRule& rule,
    const FurnitureGraphConfig& config) {
  if (!rule.can_contain) {
    return std::nullopt;
  }
  const float parent_volume = boxVolume(parent);
  const float child_volume = boxVolume(child);
  if (parent_volume <= kEpsilon || child_volume <= kEpsilon ||
      child_volume / parent_volume >
          config.in_max_child_parent_volume_ratio) {
    return std::nullopt;
  }
  const float containment = childVolumeContainment(child, parent);
  if (containment < config.in_min_child_containment_ratio) {
    return std::nullopt;
  }
  return ParentCandidate{parent_id, "in", containment, parent_volume, 0.0f};
}

bool wouldCreateCycle(int child_id,
                      int parent_id,
                      const std::map<int, int>& selected_parents) {
  std::set<int> visited;
  int current = parent_id;
  while (visited.insert(current).second) {
    if (current == child_id) {
      return true;
    }
    const auto it = selected_parents.find(current);
    if (it == selected_parents.end()) {
      return false;
    }
    current = it->second;
  }
  return true;
}

float requireRatio(const Json& object,
                   const char* key,
                   bool allow_above_one = false) {
  if (!object.contains(key) || !object.at(key).is_number()) {
    throw std::invalid_argument(std::string("furniture config field '") +
                                key + "' must be numeric");
  }
  const float value = object.at(key).get<float>();
  if (!std::isfinite(value) || value < 0.0f ||
      (!allow_above_one && value > 1.0f)) {
    throw std::invalid_argument(std::string("furniture config field '") +
                                key + "' is out of range");
  }
  return value;
}

}  // namespace

std::string normalizeFurnitureLabel(std::string label) {
  std::string normalized;
  normalized.reserve(label.size());
  bool previous_separator = true;
  for (unsigned char character : label) {
    const bool separator = std::isspace(character) || character == '-' ||
                           character == '_';
    if (separator) {
      if (!previous_separator && !normalized.empty()) {
        normalized.push_back('_');
      }
      previous_separator = true;
      continue;
    }
    normalized.push_back(static_cast<char>(std::tolower(character)));
    previous_separator = false;
  }
  if (!normalized.empty() && normalized.back() == '_') {
    normalized.pop_back();
  }
  return normalized;
}

FurnitureGraphConfig defaultFurnitureGraphConfig() {
  FurnitureGraphConfig config;
  const auto top = FurnitureSupportMode::kTop;
  const auto upper = FurnitureSupportMode::kUpperVolume;
  config.classes = {
      {"desk", {top, false}},       {"table", {top, false}},
      {"sofa", {upper, false}},     {"chair", {upper, false}},
      {"shelf", {top, false}},      {"bed", {upper, false}},
      {"cabinet", {top, true}},     {"nightstand", {top, true}},
      {"drawer", {top, true}},
  };
  return config;
}

FurnitureGraphConfig loadFurnitureGraphConfig(
    const std::filesystem::path& path) {
  if (path.empty()) {
    throw std::invalid_argument("furniture config path is empty");
  }
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open furniture config: " +
                             path.string());
  }
  Json root;
  try {
    stream >> root;
  } catch (const std::exception& error) {
    throw std::invalid_argument("failed to parse furniture config: " +
                                std::string(error.what()));
  }
  if (!root.is_object() ||
      root.value("schema_version", std::string()) !=
          "roomie.furniture.v1") {
    throw std::invalid_argument("unsupported furniture config schema_version");
  }
  const Json classes = root.value("classes", Json::array());
  if (!classes.is_array() || classes.empty()) {
    throw std::invalid_argument("furniture config classes must be non-empty");
  }
  FurnitureGraphConfig config;
  for (const Json& value : classes) {
    if (!value.is_object() || !value.contains("label") ||
        !value.at("label").is_string()) {
      throw std::invalid_argument("furniture class requires a string label");
    }
    const std::string label =
        normalizeFurnitureLabel(value.at("label").get<std::string>());
    if (label.empty()) {
      throw std::invalid_argument("furniture class label is empty");
    }
    const std::string mode =
        value.value("support_mode", std::string("top"));
    FurnitureClassRule rule;
    if (mode == "top") {
      rule.support_mode = FurnitureSupportMode::kTop;
    } else if (mode == "upper_volume") {
      rule.support_mode = FurnitureSupportMode::kUpperVolume;
    } else {
      throw std::invalid_argument("unsupported furniture support_mode: " +
                                  mode);
    }
    rule.can_contain = value.value("can_contain", false);
    if (!config.classes.emplace(label, rule).second) {
      throw std::invalid_argument("duplicate normalized furniture label: " +
                                  label);
    }
  }
  const Json on = root.value("on", Json::object());
  const Json inside = root.value("in", Json::object());
  config.on_min_horizontal_overlap_ratio =
      requireRatio(on, "min_horizontal_overlap_ratio");
  config.on_max_gap_m = requireRatio(on, "max_gap_m", true);
  config.on_max_penetration_m =
      requireRatio(on, "max_penetration_m", true);
  config.in_min_child_containment_ratio =
      requireRatio(inside, "min_child_containment_ratio");
  config.in_max_child_parent_volume_ratio =
      requireRatio(inside, "max_child_parent_volume_ratio");
  return config;
}

std::vector<FurnitureRole> classifyFurnitureRoles(
    const SceneObjectTable& objects,
    const std::vector<FurnitureRole>& previous,
    const FurnitureGraphConfig& config) {
  std::map<int, FurnitureRole> old_roles;
  for (const FurnitureRole& role : previous) {
    if (role.object_id >= 0) {
      old_roles[role.object_id] = role;
    }
  }
  std::vector<FurnitureRole> result;
  for (const auto& [object_id, object] : objects) {
    if (!eligibleObject(object)) {
      continue;
    }
    const std::string label = normalizeFurnitureLabel(effectiveLabel(*object));
    if (config.classes.count(label) == 0U) {
      continue;
    }
    FurnitureRole role;
    role.object_id = object_id;
    role.classification_label = label;
    const auto old = old_roles.find(object_id);
    if (old == old_roles.end()) {
      role.revision = 1;
    } else if (old->second.classification_label == label) {
      role.revision = std::max<std::uint64_t>(1, old->second.revision);
    } else {
      if (old->second.revision == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("furniture role revision exhausted");
      }
      role.revision = old->second.revision + 1U;
    }
    result.push_back(std::move(role));
  }
  return result;
}

std::vector<SceneRelation> deriveFurnitureRelations(
    const SceneObjectTable& objects,
    const std::vector<RoomNode, Eigen::aligned_allocator<RoomNode>>& rooms,
    const std::vector<FurnitureRole>& roles,
    const FurnitureGraphConfig& config,
    SceneRevision revision) {
  std::map<int, const FurnitureRole*> role_by_id;
  for (const FurnitureRole& role : roles) {
    const auto object = objects.find(role.object_id);
    if (role.object_id >= 0 && object != objects.end() &&
        eligibleObject(object->second) &&
        config.classes.count(role.classification_label) != 0U) {
      role_by_id[role.object_id] = &role;
    }
  }

  std::vector<SceneRelation> result;
  for (const auto& [furniture_id, role] : role_by_id) {
    (void)role;
    const SceneObjectPtr& furniture = objects.at(furniture_id);
    for (const RoomNode& room : rooms) {
      if (!roomContainsCenter(room, *furniture->geometry)) {
        continue;
      }
      SceneRelation relation;
      setRelationEndpoints(
          &relation, SceneEntityRef{SceneEntityType::kRoom, room.room_id},
          SceneEntityRef{SceneEntityType::kFurniture, furniture_id});
      relation.relation_type = "room_contains_furniture";
      relation.confidence = 1.0f;
      relation.revision = revision;
      relation.derived = true;
      result.push_back(std::move(relation));
    }
  }

  std::map<int, int> selected_parents;
  for (const auto& [child_id, child] : objects) {
    if (!eligibleObject(child) || !finiteGeometry(*child->geometry)) {
      continue;
    }
    std::vector<ParentCandidate> on_candidates;
    std::vector<ParentCandidate> in_candidates;
    for (const auto& [parent_id, role] : role_by_id) {
      if (parent_id == child_id) {
        continue;
      }
      const SceneObjectPtr& parent = objects.at(parent_id);
      if (!finiteGeometry(*parent->geometry)) {
        continue;
      }
      const FurnitureClassRule& rule =
          config.classes.at(role->classification_label);
      if (auto candidate = onCandidate(parent_id, *child->geometry,
                                       *parent->geometry, rule, config)) {
        on_candidates.push_back(std::move(*candidate));
      }
      if (auto candidate = inCandidate(parent_id, *child->geometry,
                                       *parent->geometry, rule, config)) {
        in_candidates.push_back(std::move(*candidate));
      }
    }

    auto& candidates = on_candidates.empty() ? in_candidates : on_candidates;
    if (on_candidates.empty()) {
      std::sort(candidates.begin(), candidates.end(),
                [](const ParentCandidate& lhs, const ParentCandidate& rhs) {
                  if (lhs.parent_volume != rhs.parent_volume) {
                    return lhs.parent_volume < rhs.parent_volume;
                  }
                  if (lhs.confidence != rhs.confidence) {
                    return lhs.confidence > rhs.confidence;
                  }
                  return lhs.parent_id < rhs.parent_id;
                });
    } else {
      std::sort(candidates.begin(), candidates.end(),
                [](const ParentCandidate& lhs, const ParentCandidate& rhs) {
                  if (lhs.confidence != rhs.confidence) {
                    return lhs.confidence > rhs.confidence;
                  }
                  if (lhs.vertical_error != rhs.vertical_error) {
                    return lhs.vertical_error < rhs.vertical_error;
                  }
                  return std::tie(lhs.parent_volume, lhs.parent_id) <
                         std::tie(rhs.parent_volume, rhs.parent_id);
                });
    }
    const ParentCandidate* selected = nullptr;
    for (const ParentCandidate& candidate : candidates) {
      if (role_by_id.count(child_id) != 0U &&
          wouldCreateCycle(child_id, candidate.parent_id,
                           selected_parents)) {
        continue;
      }
      selected = &candidate;
      break;
    }
    if (selected == nullptr) {
      continue;
    }
    if (role_by_id.count(child_id) != 0U) {
      selected_parents[child_id] = selected->parent_id;
    }
    SceneRelation relation;
    setRelationEndpoints(
        &relation,
        SceneEntityRef{role_by_id.count(child_id) != 0U
                           ? SceneEntityType::kFurniture
                           : SceneEntityType::kObject,
                       child_id},
        SceneEntityRef{SceneEntityType::kFurniture, selected->parent_id});
    relation.relation_type = selected->relation_type;
    relation.confidence = selected->confidence;
    relation.description = "derived furniture spatial relation";
    relation.revision = revision;
    relation.derived = true;
    result.push_back(std::move(relation));
  }
  return result;
}

bool isDerivedFurnitureRelation(const SceneRelation& relation) {
  if (!relation.derived) {
    return false;
  }
  const SceneEntityRef source = relationSource(relation);
  const SceneEntityRef target = relationTarget(relation);
  return relation.relation_type == "room_contains_furniture" ||
         ((relation.relation_type == "in" || relation.relation_type == "on") &&
          (source.type == SceneEntityType::kFurniture ||
           target.type == SceneEntityType::kFurniture));
}

}  // namespace roomie
