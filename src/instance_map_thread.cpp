#include "roomie/pipeline/instance_map_thread.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

#include "roomie/utils/run_logger.hpp"

namespace roomie {
namespace {

constexpr float kEpsilon = 1.0e-6f;
constexpr float kPi = 3.14159265358979323846f;

struct Aabb {
  Eigen::Vector3f min = Eigen::Vector3f::Zero();
  Eigen::Vector3f max = Eigen::Vector3f::Zero();
};

struct GeometryEvaluation {
  float score = 0.0f;
  float shell_ratio = 0.0f;
  float extent_score = 0.0f;
  float leak_ratio = 1.0f;
  float cavity_ratio = 0.0f;
  int in_box_points = 0;
  int shell_points = 0;
  int cavity_points = 0;
  int expanded_points = 0;
  int unique_voxels = 0;
  std::string reason;
};

template <typename T>
void appendUnique(std::vector<T>* values, const T& value) {
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(value);
  }
}

bool isFiniteVector(const Eigen::Vector3f& value) {
  return std::isfinite(value.x()) && std::isfinite(value.y()) && std::isfinite(value.z());
}

float clamp01(float value) {
  return std::max(0.0f, std::min(1.0f, value));
}

float rawDetectionConfidence(const RawDetection& detection) {
  return clamp01(0.5f * (detection.score_2d + detection.score_3d));
}

TimeNanoseconds secondsToNanoseconds(double seconds) {
  return static_cast<TimeNanoseconds>(seconds * 1000000000.0);
}

float normalizeYaw(float yaw) {
  while (yaw >= kPi) {
    yaw -= 2.0f * kPi;
  }
  while (yaw < -kPi) {
    yaw += 2.0f * kPi;
  }
  return yaw;
}

float angularDistancePiSymmetric(float a, float b) {
  float diff = normalizeYaw(a - b);
  diff = std::abs(diff);
  if (diff > 0.5f * kPi) {
    diff = kPi - diff;
  }
  return diff;
}

float weightedYawMean(float yaw_a, float weight_a, float yaw_b, float weight_b) {
  const float x = weight_a * std::cos(2.0f * yaw_a) +
                  weight_b * std::cos(2.0f * yaw_b);
  const float y = weight_a * std::sin(2.0f * yaw_a) +
                  weight_b * std::sin(2.0f * yaw_b);
  if (std::abs(x) < kEpsilon && std::abs(y) < kEpsilon) {
    return 0.0f;
  }
  return normalizeYaw(0.5f * std::atan2(y, x));
}

void alignBoxToReference(Eigen::Vector3f* size, float* yaw, float reference_yaw) {
  const float diff_keep = angularDistancePiSymmetric(*yaw, reference_yaw);
  const float yaw_plus = *yaw + 0.5f * kPi;
  const float yaw_minus = *yaw - 0.5f * kPi;
  const float diff_plus = angularDistancePiSymmetric(yaw_plus, reference_yaw);
  const float diff_minus = angularDistancePiSymmetric(yaw_minus, reference_yaw);

  float rotated_yaw = yaw_minus;
  float diff_rotated = diff_minus;
  if (diff_plus < diff_minus) {
    rotated_yaw = yaw_plus;
    diff_rotated = diff_plus;
  }

  if (diff_rotated < diff_keep) {
    std::swap(size->x(), size->y());
    *yaw = normalizeYaw(rotated_yaw);
  }
}

Aabb yawAabb(const Eigen::Vector3f& center, const Eigen::Vector3f& size, float yaw) {
  const Eigen::Vector3f half = 0.5f * size.cwiseMax(Eigen::Vector3f::Constant(0.0f));
  const float c = std::cos(yaw);
  const float s = std::sin(yaw);
  const Eigen::Matrix3f R =
      (Eigen::Matrix3f() << c, -s, 0.0f, s, c, 0.0f, 0.0f, 0.0f, 1.0f).finished();

  Aabb box;
  box.min = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
  box.max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
  for (int ix = -1; ix <= 1; ix += 2) {
    for (int iy = -1; iy <= 1; iy += 2) {
      for (int iz = -1; iz <= 1; iz += 2) {
        const Eigen::Vector3f corner =
            center + R * Eigen::Vector3f(ix * half.x(), iy * half.y(), iz * half.z());
        box.min = box.min.cwiseMin(corner);
        box.max = box.max.cwiseMax(corner);
      }
    }
  }
  return box;
}

float volume(const Aabb& box) {
  const Eigen::Vector3f extent = (box.max - box.min).cwiseMax(Eigen::Vector3f::Zero());
  return extent.x() * extent.y() * extent.z();
}

float aabbIou(const Aabb& lhs, const Aabb& rhs) {
  const Eigen::Vector3f intersection_min = lhs.min.cwiseMax(rhs.min);
  const Eigen::Vector3f intersection_max = lhs.max.cwiseMin(rhs.max);
  const Eigen::Vector3f intersection_extent =
      (intersection_max - intersection_min).cwiseMax(Eigen::Vector3f::Zero());
  const float intersection =
      intersection_extent.x() * intersection_extent.y() * intersection_extent.z();
  const float union_volume = volume(lhs) + volume(rhs) - intersection;
  return union_volume > kEpsilon ? intersection / union_volume : 0.0f;
}

float sizeVolume(const Eigen::Vector3f& size) {
  if ((size.array() <= 0.0f).any()) {
    return 0.0f;
  }
  return size.x() * size.y() * size.z();
}

float centerDistance(const InstanceTrack& lhs, const InstanceTrack& rhs) {
  return (lhs.center_world - rhs.center_world).norm();
}

float trackDetectionCenterDistance(const InstanceTrack& track, const RawDetection& detection) {
  return (track.center_world - detection.center_world).norm();
}

float trackDetectionIou(const InstanceTrack& track, const RawDetection& detection) {
  return aabbIou(yawAabb(track.center_world, track.size_m, track.yaw_rad),
                 yawAabb(detection.center_world, detection.size_m, detection.yaw_rad));
}

float trackDetectionContainment(const InstanceTrack& track, const RawDetection& detection) {
  const Aabb lhs = yawAabb(track.center_world, track.size_m, track.yaw_rad);
  const Aabb rhs = yawAabb(detection.center_world, detection.size_m, detection.yaw_rad);
  const Eigen::Vector3f intersection_min = lhs.min.cwiseMax(rhs.min);
  const Eigen::Vector3f intersection_max = lhs.max.cwiseMin(rhs.max);
  const Eigen::Vector3f intersection_extent =
      (intersection_max - intersection_min).cwiseMax(Eigen::Vector3f::Zero());
  const float intersection =
      intersection_extent.x() * intersection_extent.y() * intersection_extent.z();
  const float smaller = std::min(sizeVolume(track.size_m), sizeVolume(detection.size_m));
  return smaller > kEpsilon ? intersection / smaller : 0.0f;
}

using Polygon2 = std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>;

Polygon2 yawRectCorners2d(const Eigen::Vector3f& center,
                          const Eigen::Vector3f& size,
                          float yaw) {
  const Eigen::Vector2f half(0.5f * std::max(0.0f, size.x()),
                             0.5f * std::max(0.0f, size.y()));
  const float c = std::cos(yaw);
  const float s = std::sin(yaw);
  const Eigen::Matrix2f R =
      (Eigen::Matrix2f() << c, -s, s, c).finished();
  const Eigen::Vector2f center_xy(center.x(), center.y());
  Polygon2 corners;
  corners.reserve(4);
  corners.push_back(center_xy + R * Eigen::Vector2f(-half.x(), -half.y()));
  corners.push_back(center_xy + R * Eigen::Vector2f(half.x(), -half.y()));
  corners.push_back(center_xy + R * Eigen::Vector2f(half.x(), half.y()));
  corners.push_back(center_xy + R * Eigen::Vector2f(-half.x(), half.y()));
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
  const float denom = cross2d(r, s);
  if (std::abs(denom) < kEpsilon) {
    return p1;
  }
  const float t = cross2d(q0 - p0, s) / denom;
  return p0 + t * r;
}

Polygon2 clipPolygonByConvexPolygon(const Polygon2& subject, const Polygon2& clipper) {
  Polygon2 output = subject;
  for (std::size_t i = 0; i < clipper.size(); ++i) {
    if (output.empty()) {
      break;
    }
    const Eigen::Vector2f edge_start = clipper[i];
    const Eigen::Vector2f edge_end = clipper[(i + 1) % clipper.size()];
    const Polygon2 input = output;
    output.clear();
    Eigen::Vector2f previous = input.back();
    bool previous_inside = insideClipEdge(previous, edge_start, edge_end);
    for (const Eigen::Vector2f& current : input) {
      const bool current_inside = insideClipEdge(current, edge_start, edge_end);
      if (current_inside != previous_inside) {
        output.push_back(lineIntersection(previous, current, edge_start, edge_end));
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
  if (polygon.size() < 3) {
    return 0.0f;
  }
  float area = 0.0f;
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    const Eigen::Vector2f& a = polygon[i];
    const Eigen::Vector2f& b = polygon[(i + 1) % polygon.size()];
    area += cross2d(a, b);
  }
  return 0.5f * std::abs(area);
}

float obbIou(const InstanceTrack& lhs, const InstanceTrack& rhs) {
  if ((lhs.size_m.array() <= 0.0f).any() || (rhs.size_m.array() <= 0.0f).any()) {
    return 0.0f;
  }
  const Polygon2 lhs_rect = yawRectCorners2d(lhs.center_world, lhs.size_m, lhs.yaw_rad);
  const Polygon2 rhs_rect = yawRectCorners2d(rhs.center_world, rhs.size_m, rhs.yaw_rad);
  const float area_xy = polygonArea(clipPolygonByConvexPolygon(lhs_rect, rhs_rect));
  if (area_xy <= kEpsilon) {
    return 0.0f;
  }
  const float lhs_min_z = lhs.center_world.z() - 0.5f * lhs.size_m.z();
  const float lhs_max_z = lhs.center_world.z() + 0.5f * lhs.size_m.z();
  const float rhs_min_z = rhs.center_world.z() - 0.5f * rhs.size_m.z();
  const float rhs_max_z = rhs.center_world.z() + 0.5f * rhs.size_m.z();
  const float z_overlap = std::max(0.0f, std::min(lhs_max_z, rhs_max_z) -
                                             std::max(lhs_min_z, rhs_min_z));
  const float intersection = area_xy * z_overlap;
  const float lhs_volume = lhs.size_m.x() * lhs.size_m.y() * lhs.size_m.z();
  const float rhs_volume = rhs.size_m.x() * rhs.size_m.y() * rhs.size_m.z();
  const float union_volume = lhs_volume + rhs_volume - intersection;
  return union_volume > kEpsilon ? intersection / union_volume : 0.0f;
}

float obbContainment(const InstanceTrack& lhs, const InstanceTrack& rhs) {
  if ((lhs.size_m.array() <= 0.0f).any() || (rhs.size_m.array() <= 0.0f).any()) {
    return 0.0f;
  }
  const Polygon2 lhs_rect = yawRectCorners2d(lhs.center_world, lhs.size_m, lhs.yaw_rad);
  const Polygon2 rhs_rect = yawRectCorners2d(rhs.center_world, rhs.size_m, rhs.yaw_rad);
  const float area_xy = polygonArea(clipPolygonByConvexPolygon(lhs_rect, rhs_rect));
  if (area_xy <= kEpsilon) {
    return 0.0f;
  }
  const float lhs_min_z = lhs.center_world.z() - 0.5f * lhs.size_m.z();
  const float lhs_max_z = lhs.center_world.z() + 0.5f * lhs.size_m.z();
  const float rhs_min_z = rhs.center_world.z() - 0.5f * rhs.size_m.z();
  const float rhs_max_z = rhs.center_world.z() + 0.5f * rhs.size_m.z();
  const float z_overlap = std::max(0.0f, std::min(lhs_max_z, rhs_max_z) -
                                             std::max(lhs_min_z, rhs_min_z));
  const float intersection = area_xy * z_overlap;
  const float smaller = std::min(sizeVolume(lhs.size_m), sizeVolume(rhs.size_m));
  return smaller > kEpsilon ? intersection / smaller : 0.0f;
}

float sizeRatioScore(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
  if ((lhs.array() <= 0.0f).any() || (rhs.array() <= 0.0f).any()) {
    return 0.0f;
  }
  const auto direct_ratio = [&]() {
    float ratio = 1.0f;
    for (int axis = 0; axis < 3; ++axis) {
      ratio = std::min(ratio, std::min(lhs[axis], rhs[axis]) /
                                  std::max(lhs[axis], rhs[axis]));
    }
    return ratio;
  };
  const Eigen::Vector3f rhs_swapped(rhs.y(), rhs.x(), rhs.z());
  float swapped = 1.0f;
  for (int axis = 0; axis < 3; ++axis) {
    swapped = std::min(swapped, std::min(lhs[axis], rhs_swapped[axis]) /
                                  std::max(lhs[axis], rhs_swapped[axis]));
  }
  return std::max(direct_ratio(), swapped);
}

float edgeCompletenessWeight(const RawDetection& detection, int image_size) {
  if (image_size <= 0) {
    return 1.0f;
  }
  const float x0 = std::min(detection.box_xyxy[0], detection.box_xyxy[2]);
  const float x1 = std::max(detection.box_xyxy[0], detection.box_xyxy[2]);
  const float y0 = std::min(detection.box_xyxy[1], detection.box_xyxy[3]);
  const float y1 = std::max(detection.box_xyxy[1], detection.box_xyxy[3]);
  if (!std::isfinite(x0) || !std::isfinite(x1) || !std::isfinite(y0) ||
      !std::isfinite(y1) || x1 <= x0 || y1 <= y0) {
    return 1.0f;
  }
  const float min_edge_distance =
      std::min(std::min(x0, y0),
               std::min(static_cast<float>(image_size) - x1,
                        static_cast<float>(image_size) - y1));
  const float margin_px = std::max(8.0f, 0.04f * static_cast<float>(image_size));
  if (min_edge_distance >= margin_px) {
    return 1.0f;
  }
  if (min_edge_distance <= 0.0f) {
    return 0.35f;
  }
  return 0.35f + 0.65f * min_edge_distance / margin_px;
}

float distanceQualityWeight(const InferenceResponse& response,
                            const RawDetection& detection,
                            float* camera_distance_m) {
  if (!response.has_camera_pose) {
    if (camera_distance_m != nullptr) {
      *camera_distance_m = 0.0f;
    }
    return 1.0f;
  }
  const float distance =
      (detection.center_world - response.T_world_camera.translation()).norm();
  if (camera_distance_m != nullptr) {
    *camera_distance_m = distance;
  }
  const float volume =
      std::max(kEpsilon, detection.size_m.x() * detection.size_m.y() * detection.size_m.z());
  const float object_scale = std::cbrt(volume);
  const float smallness = clamp01((0.50f - object_scale) / 0.45f);
  const float far = clamp01((distance - 2.5f) / 3.5f);
  return clamp01(1.0f - 0.55f * smallness * far);
}

float observationBboxQuality(const InferenceResponse& response,
                             const RawDetection& detection,
                             float confidence,
                             int image_size,
                             float* camera_distance_m) {
  const float edge_weight = edgeCompletenessWeight(detection, image_size);
  const float distance_weight =
      distanceQualityWeight(response, detection, camera_distance_m);
  return std::max(kEpsilon, confidence * edge_weight * distance_weight);
}

Eigen::Vector3f pointToLocalObb(const Eigen::Vector3f& point_world,
                                const Eigen::Vector3f& center_world,
                                float yaw_rad) {
  const Eigen::Vector3f delta = point_world - center_world;
  const float c = std::cos(-yaw_rad);
  const float s = std::sin(-yaw_rad);
  return Eigen::Vector3f(c * delta.x() - s * delta.y(),
                         s * delta.x() + c * delta.y(),
                         delta.z());
}

GeometryEvaluation evaluateGeometryAgainstSurface(
    const InstanceTrack& track,
    const GeometrySurfaceCache& surface_cache,
    float shell_thickness_m,
    int min_unique_voxels) {
  GeometryEvaluation evaluation;
  if ((track.size_m.array() <= 0.0f).any() || surface_cache.surface_points.empty()) {
    evaluation.reason = "no_surface_points";
    return evaluation;
  }

  const Eigen::Vector3f half = 0.5f * track.size_m;
  const float min_half = std::max(kEpsilon, half.minCoeff());
  const float shell = std::min(std::max(shell_thickness_m, 0.0f), 0.85f * min_half);
  const Eigen::Vector3f inner_half =
      (half.array() - shell).max(0.0f).matrix();
  const Eigen::Vector3f expanded_half =
      (half.array() + std::max(shell, 0.02f)).matrix();

  Eigen::Vector3f local_min =
      Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
  Eigen::Vector3f local_max =
      Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
  std::set<std::tuple<int, int, int, int, int, int>> unique_voxels;

  for (const MapSurfacePoint& surface_point : surface_cache.surface_points) {
    const Eigen::Vector3f& point_world = surface_point.position_world;
    if (!point_world.allFinite()) {
      continue;
    }
    const Eigen::Vector3f local =
        pointToLocalObb(point_world, track.center_world, track.yaw_rad);
    const Eigen::Vector3f abs_local = local.cwiseAbs();
    const bool inside_expanded = (abs_local.array() <= expanded_half.array()).all();
    if (!inside_expanded) {
      continue;
    }
    ++evaluation.expanded_points;
    const bool inside_box = (abs_local.array() <= half.array()).all();
    if (!inside_box) {
      continue;
    }
    ++evaluation.in_box_points;
    if (surface_point.has_voxel_ref) {
      unique_voxels.insert(std::make_tuple(surface_point.voxel_ref.block_index.x(),
                                           surface_point.voxel_ref.block_index.y(),
                                           surface_point.voxel_ref.block_index.z(),
                                           surface_point.voxel_ref.voxel_index.x(),
                                           surface_point.voxel_ref.voxel_index.y(),
                                           surface_point.voxel_ref.voxel_index.z()));
    }
    local_min = local_min.cwiseMin(local);
    local_max = local_max.cwiseMax(local);
    const bool inside_cavity = (abs_local.array() < inner_half.array()).all();
    if (!inside_cavity) {
      ++evaluation.shell_points;
    } else {
      ++evaluation.cavity_points;
    }
  }
  evaluation.unique_voxels =
      unique_voxels.empty() ? evaluation.in_box_points
                            : static_cast<int>(unique_voxels.size());

  if (evaluation.in_box_points <= 0) {
    evaluation.reason = "empty_box";
    return evaluation;
  }

  evaluation.shell_ratio =
      static_cast<float>(evaluation.shell_points) /
      static_cast<float>(evaluation.in_box_points);
  evaluation.cavity_ratio =
      static_cast<float>(evaluation.cavity_points) /
      static_cast<float>(evaluation.in_box_points);
  const Eigen::Vector3f occupied_extent =
      (local_max - local_min).cwiseMax(Eigen::Vector3f::Zero());
  const Eigen::Vector3f extent_ratio =
      occupied_extent.cwiseQuotient(track.size_m.cwiseMax(Eigen::Vector3f::Constant(kEpsilon)));
  const float mean_extent_ratio =
      clamp01((extent_ratio.x() + extent_ratio.y() + extent_ratio.z()) / 3.0f);
  evaluation.extent_score = clamp01((mean_extent_ratio - 0.35f) / 0.55f);
  evaluation.leak_ratio =
      evaluation.expanded_points > 0
          ? static_cast<float>(evaluation.expanded_points - evaluation.in_box_points) /
                static_cast<float>(evaluation.expanded_points)
          : 1.0f;
  const float density_score =
      clamp01(static_cast<float>(evaluation.unique_voxels) /
              static_cast<float>(std::max(1, min_unique_voxels) * 3));
  evaluation.score =
      clamp01(0.35f * evaluation.shell_ratio + 0.30f * evaluation.extent_score +
              0.25f * density_score + 0.10f * (1.0f - evaluation.leak_ratio) -
              0.10f * evaluation.cavity_ratio);
  evaluation.reason = "evaluated";
  return evaluation;
}

void appendUniqueVoxelRefs(
    std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>>* values,
    const std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>>& incoming) {
  for (const VoxelRef& ref : incoming) {
    const auto it = std::find_if(values->begin(),
                                 values->end(),
                                 [&ref](const VoxelRef& existing) {
                                   return existing.block_index == ref.block_index &&
                                          existing.voxel_index == ref.voxel_index;
                                 });
    if (it == values->end()) {
      values->push_back(ref);
    }
  }
}

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    if (c == '-' || c == '_') {
      return ' ';
    }
    return static_cast<char>(std::tolower(c));
  });
  value.erase(std::unique(value.begin(),
                          value.end(),
                          [](char a, char b) {
                            return a == ' ' && b == ' ';
                          }),
              value.end());
  return value;
}

bool isSemanticOverride(const std::string& lhs, const std::string& rhs) {
  const std::string a = lowercase(lhs);
  const std::string b = lowercase(rhs);
  const auto in_group = [](const std::string& label,
                           std::initializer_list<const char*> group) {
    return std::find(group.begin(), group.end(), label) != group.end();
  };
  const bool table_group =
      in_group(a, {"table", "desk", "coffee table"}) &&
      in_group(b, {"table", "desk", "coffee table"});
  const bool bed_group =
      in_group(a, {"bed", "sofa bed", "futon"}) &&
      in_group(b, {"bed", "sofa bed", "futon"});
  const bool sofa_group =
      in_group(a, {"sofa", "futon", "armchair", "arm chair"}) &&
      in_group(b, {"sofa", "futon", "armchair", "arm chair"});
  return table_group || bed_group || sofa_group;
}

bool semanticCompatible(const InstanceTrack& track, const InstanceObservation& observation) {
  if (track.semantic_id >= 0 && observation.detection.semantic_id >= 0) {
    return track.semantic_id == observation.detection.semantic_id ||
           isSemanticOverride(track.label, observation.detection.label);
  }
  if (!track.label.empty() && !observation.detection.label.empty()) {
    return track.label == observation.detection.label ||
           isSemanticOverride(track.label, observation.detection.label);
  }
  return true;
}

bool geometryConfirmed(const InstanceTrack& track, const PipelineConfig& config);

bool labelsCompatibleForDuplicate(const InstanceTrack& track, const RawDetection& detection) {
  if (track.semantic_id >= 0 && detection.semantic_id >= 0 &&
      track.semantic_id == detection.semantic_id) {
    return true;
  }
  if (!track.label.empty() && !detection.label.empty()) {
    return lowercase(track.label) == lowercase(detection.label) ||
           isSemanticOverride(track.label, detection.label);
  }
  return false;
}

bool labelsCompatibleForDuplicate(const InstanceTrack& lhs, const InstanceTrack& rhs) {
  if (lhs.semantic_id >= 0 && rhs.semantic_id >= 0 &&
      lhs.semantic_id == rhs.semantic_id) {
    return true;
  }
  if (!lhs.label.empty() && !rhs.label.empty()) {
    return lowercase(lhs.label) == lowercase(rhs.label) ||
           isSemanticOverride(lhs.label, rhs.label);
  }
  return false;
}

float dynamicDuplicateCenterGate(const Eigen::Vector3f& lhs_size,
                                 const Eigen::Vector3f& rhs_size,
                                 const PipelineConfig& config) {
  const float lhs_diag = lhs_size.cwiseMax(Eigen::Vector3f::Zero()).norm();
  const float rhs_diag = rhs_size.cwiseMax(Eigen::Vector3f::Zero()).norm();
  return std::max(config.instance_duplicate_center_distance_m,
                  0.45f * std::max(lhs_diag, rhs_diag));
}

bool isSmallContainedDifferentObject(float volume_a,
                                     float volume_b,
                                     bool label_compatible,
                                     const PipelineConfig& config) {
  if (label_compatible) {
    return false;
  }
  const float larger = std::max(volume_a, volume_b);
  const float smaller = std::min(volume_a, volume_b);
  return larger > kEpsilon &&
         smaller / larger < config.instance_duplicate_small_object_volume_ratio;
}

bool confirmedTrackDuplicatesDetection(const InstanceTrack& track,
                                       const RawDetection& detection,
                                       const PipelineConfig& config) {
  if (!geometryConfirmed(track, config) || !track.publishable) {
    return false;
  }
  const float distance = trackDetectionCenterDistance(track, detection);
  const float center_gate = dynamicDuplicateCenterGate(track.size_m,
                                                      detection.size_m,
                                                      config);
  if (distance > center_gate) {
    return false;
  }
  const float iou = trackDetectionIou(track, detection);
  const float containment = trackDetectionContainment(track, detection);
  if (iou < config.instance_confirmed_duplicate_iou_threshold &&
      containment < config.instance_duplicate_containment_threshold) {
    return false;
  }
  const bool label_compatible = labelsCompatibleForDuplicate(track, detection);
  if (isSmallContainedDifferentObject(sizeVolume(track.size_m),
                                      sizeVolume(detection.size_m),
                                      label_compatible,
                                      config)) {
    return false;
  }
  return true;
}

bool stableTracksAreDuplicates(const InstanceTrack& lhs,
                               const InstanceTrack& rhs,
                               const PipelineConfig& config) {
  const bool lhs_confirmed = geometryConfirmed(lhs, config);
  const bool rhs_confirmed = geometryConfirmed(rhs, config);
  const float iou = obbIou(lhs, rhs);
  if (!lhs_confirmed && !rhs_confirmed) {
    return iou >= config.instance_duplicate_iou_threshold &&
           sizeRatioScore(lhs.size_m, rhs.size_m) >=
               config.instance_duplicate_size_ratio_min;
  }
  const float distance = centerDistance(lhs, rhs);
  const float center_gate = dynamicDuplicateCenterGate(lhs.size_m, rhs.size_m, config);
  if (distance > center_gate) {
    return false;
  }
  const float containment = obbContainment(lhs, rhs);
  if (iou < config.instance_confirmed_duplicate_iou_threshold &&
      containment < config.instance_duplicate_containment_threshold) {
    return false;
  }
  const bool label_compatible = labelsCompatibleForDuplicate(lhs, rhs);
  if (isSmallContainedDifferentObject(sizeVolume(lhs.size_m),
                                      sizeVolume(rhs.size_m),
                                      label_compatible,
                                      config)) {
    return false;
  }
  return true;
}

template <typename Key>
Key bestWeightedKey(const std::map<Key, float>& weights, const Key& fallback) {
  if (weights.empty()) {
    return fallback;
  }
  return std::max_element(weights.begin(),
                          weights.end(),
                          [](const auto& lhs, const auto& rhs) {
                            return lhs.second < rhs.second;
                          })
      ->first;
}

void updateObjectQualityScore(InstanceTrack* track) {
  const float mean_bbox_quality =
      track->support_count > 0
          ? clamp01(track->bbox_quality_mass / static_cast<float>(track->support_count))
          : 0.0f;
  const float support_score =
      track->support_count > 0
          ? clamp01(1.0f - std::exp(-static_cast<float>(track->support_count) / 4.0f))
          : 0.0f;
  const float geometry_component =
      track->geometry_status == InstanceGeometryStatus::kUnchecked ? 0.5f
                                                                   : track->geometry_score;
  track->object_quality_score =
      clamp01(0.45f * track->confidence + 0.25f * mean_bbox_quality +
              0.20f * support_score + 0.10f * geometry_component);
}

float observationPromotionWeight(const PipelineConfig& config,
                                 const InstanceObservation& observation) {
  if (config.instance_far_observation_distance_m <= kEpsilon ||
      observation.camera_distance_m <= 0.0f ||
      observation.camera_distance_m <= config.instance_close_observation_distance_m) {
    return 1.0f;
  }
  if (observation.camera_distance_m >= config.instance_far_observation_distance_m) {
    return config.instance_far_promotion_weight;
  }
  const float t =
      (observation.camera_distance_m - config.instance_close_observation_distance_m) /
      std::max(kEpsilon,
               config.instance_far_observation_distance_m -
                   config.instance_close_observation_distance_m);
  return (1.0f - t) + t * config.instance_far_promotion_weight;
}

bool isHighQualityObservation(const PipelineConfig& config,
                              const InstanceObservation& observation) {
  const bool close_enough =
      config.instance_close_observation_distance_m <= 0.0f ||
      observation.camera_distance_m <= 0.0f ||
      observation.camera_distance_m <= config.instance_close_observation_distance_m;
  return close_enough &&
         observation.bbox_quality >= config.instance_quality_observation_min_quality;
}

void recordObservationQuality(InstanceTrack* track,
                              const InstanceObservation& observation,
                              const PipelineConfig& config) {
  const bool high_quality = isHighQualityObservation(config, observation);
  ObservationQualitySample sample;
  sample.time_ns = observation.time_ns;
  sample.quality = observation.bbox_quality;
  sample.camera_distance_m = observation.camera_distance_m;
  sample.high_quality = high_quality;
  track->observation_quality_history.push_back(sample);
  if (high_quality) {
    ++track->high_quality_observation_count;
    track->high_quality_observation_mass += observation.bbox_quality;
  }
}

std::pair<int, float> recentHighQualityStats(const InstanceTrack& track,
                                             TimeNanoseconds now_ns,
                                             double recent_window_sec) {
  const TimeNanoseconds recent_window_ns = secondsToNanoseconds(recent_window_sec);
  int count = 0;
  float mass = 0.0f;
  for (const ObservationQualitySample& sample : track.observation_quality_history) {
    if (!sample.high_quality) {
      continue;
    }
    if (recent_window_ns > 0 && now_ns > 0 && sample.time_ns > 0 &&
        now_ns - sample.time_ns > recent_window_ns) {
      continue;
    }
    ++count;
    mass += sample.quality;
  }
  return {count, mass};
}

bool hasPromotionQuality(const InstanceTrack& track, const PipelineConfig& config) {
  return track.high_quality_observation_count >= config.instance_high_quality_min_count ||
         track.high_quality_observation_mass >= config.instance_high_quality_min_mass;
}

bool geometryObbChangedEnough(const InstanceTrack& track, const PipelineConfig& config) {
  if (track.geometry_evaluation_obb_revision == 0) {
    return true;
  }
  if ((track.geometry_evaluated_size_m.array() <= 0.0f).any()) {
    return true;
  }
  if ((track.center_world - track.geometry_evaluated_center_world).norm() >=
      config.instance_geometry_reevaluate_center_delta_m) {
    return true;
  }
  for (int axis = 0; axis < 3; ++axis) {
    const float denom = std::max(kEpsilon, track.geometry_evaluated_size_m[axis]);
    if (std::abs(track.size_m[axis] - track.geometry_evaluated_size_m[axis]) / denom >=
        config.instance_geometry_reevaluate_size_ratio) {
      return true;
    }
  }
  const float yaw_delta_deg =
      angularDistancePiSymmetric(track.yaw_rad, track.geometry_evaluated_yaw_rad) *
      180.0f / kPi;
  return yaw_delta_deg >= config.instance_geometry_reevaluate_yaw_delta_deg;
}

bool shouldEvaluateTrackGeometry(const InstanceTrack& track,
                                 const GeometrySurfaceCache& surface_cache,
                                 TimeNanoseconds now_ns,
                                 const PipelineConfig& config) {
  if (track.object_id < 0 || !surface_cache.has_map) {
    return false;
  }
  const bool changed_enough = geometryObbChangedEnough(track, config);
  const bool already_confirmed =
      track.geometry_status == InstanceGeometryStatus::kGood && !changed_enough &&
      track.geometry_evaluation_map_version == surface_cache.map_version;
  if (already_confirmed) {
    return false;
  }
  if (track.state == InstanceTrackState::kInactive) {
    return changed_enough || track.geometry_status == InstanceGeometryStatus::kUnchecked;
  }
  const auto [recent_count, recent_mass] =
      recentHighQualityStats(track, now_ns, config.instance_geometry_recent_window_sec);
  return recent_count >= config.instance_high_quality_min_count ||
         recent_mass >= config.instance_high_quality_min_mass ||
         track.geometry_status == InstanceGeometryStatus::kUnchecked;
}

bool geometryConfirmed(const InstanceTrack& track, const PipelineConfig& config) {
  return track.geometry_status == InstanceGeometryStatus::kGood &&
         track.geometry_score >= config.instance_geometry_confirm_score &&
         track.geometry_unique_voxels >= config.instance_geometry_min_unique_voxels;
}

float trackAuthorityScore(const InstanceTrack& track, const PipelineConfig& config) {
  const float geometry_bonus = geometryConfirmed(track, config) ? 1.0f : track.geometry_score;
  return 1.50f * track.high_quality_observation_mass +
         0.75f * geometry_bonus +
         0.50f * track.object_quality_score +
         0.01f * static_cast<float>(track.support_count);
}

std::string geometryStatusName(InstanceGeometryStatus status) {
  switch (status) {
    case InstanceGeometryStatus::kUnchecked:
      return "geometry_unchecked";
    case InstanceGeometryStatus::kGood:
      return "geometry_good";
    case InstanceGeometryStatus::kBad:
      return "geometry_bad";
    case InstanceGeometryStatus::kEmpty:
      return "geometry_empty";
  }
  return "geometry_unknown";
}

InstanceRecord recordFromTrack(const InstanceTrack& track) {
  InstanceRecord record;
  record.object_id = track.object_id;
  record.track_id = track.track_id;
  record.semantic_id = track.semantic_id;
  record.label = track.label;
  record.description =
      track.object_id >= 0 ? geometryStatusName(track.geometry_status) : "tentative";
  record.center_world = track.center_world;
  record.size_m = track.size_m;
  record.yaw_rad = track.yaw_rad;
  record.confidence = track.confidence;
  record.confidence_mass = track.confidence_mass;
  record.object_quality_score = track.object_quality_score;
  record.geometry_score = track.geometry_score;
  record.geometry_shell_ratio = track.geometry_shell_ratio;
  record.geometry_extent_score = track.geometry_extent_score;
  record.geometry_leak_ratio = track.geometry_leak_ratio;
  record.geometry_cavity_ratio = track.geometry_cavity_ratio;
  record.geometry_in_box_points = track.geometry_in_box_points;
  record.geometry_shell_points = track.geometry_shell_points;
  record.geometry_unique_voxels = track.geometry_unique_voxels;
  record.geometry_expanded_points = track.geometry_expanded_points;
  record.geometry_bad_count = track.geometry_bad_count;
  record.support_count = track.support_count;
  record.high_quality_observation_count = track.high_quality_observation_count;
  record.high_quality_observation_mass = track.high_quality_observation_mass;
  record.active = track.state != InstanceTrackState::kInactive;
  record.publishable = track.publishable;
  record.geometry_status = track.geometry_status;
  record.last_geometry_check_ns = track.last_geometry_check_ns;
  record.first_seen_ns = track.first_seen_ns;
  record.last_seen_ns = track.last_seen_ns;
  record.source_cameras = track.source_cameras;
  record.source_track_ids.push_back(track.track_id);
  record.observation_timestamps_ns = track.observation_timestamps_ns;
  record.near_surface_voxels = track.near_surface_voxels;
  return record;
}

}  // namespace

InstanceMapThread::InstanceMapThread(ThreadSafeQueue<InferenceResponse>& response_queue,
                                     const MapProjector& map_projector,
                                     PipelineConfig config)
    : WorkerThread("instance_map_thread"),
      response_queue_(response_queue),
      map_projector_(map_projector),
      config_(std::move(config)) {}

bool InstanceMapThread::enqueueDetections(InferenceResponse response) {
  return response_queue_.pushDropOldest(std::move(response));
}

std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
InstanceMapThread::snapshotInstances() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return object_graph_.snapshotInstanceRecords(/*publishable_only=*/false);
}

std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
InstanceMapThread::snapshotTrackedInstances() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>> records;
  records.reserve(tracks_.size());
  for (const InstanceTrack& track : tracks_) {
    records.push_back(recordFromTrack(track));
  }
  return records;
}

ObjectGraphSnapshot InstanceMapThread::snapshotObjectGraph() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return object_graph_.snapshot();
}

bool InstanceMapThread::loadObjectGraphSnapshot(const ObjectGraphSnapshot& snapshot,
                                                std::string* error) {
  std::lock_guard<std::mutex> lock(mutex_);
  object_graph_.loadSnapshot(snapshot);
  tracks_.clear();

  std::set<int> used_track_ids;
  int next_candidate_track_id = 0;
  const auto allocateTrackId = [&]() {
    while (used_track_ids.count(next_candidate_track_id) > 0) {
      ++next_candidate_track_id;
    }
    const int track_id = next_candidate_track_id++;
    used_track_ids.insert(track_id);
    return track_id;
  };

  for (const ObjectNode& object : snapshot.objects) {
    if (object.object_id < 0) {
      continue;
    }

    int track_id = -1;
    for (auto it = object.source_track_ids.rbegin();
         it != object.source_track_ids.rend();
         ++it) {
      if (*it >= 0 && used_track_ids.count(*it) == 0) {
        track_id = *it;
        used_track_ids.insert(track_id);
        next_candidate_track_id = std::max(next_candidate_track_id, track_id + 1);
        break;
      }
    }
    if (track_id < 0) {
      track_id = allocateTrackId();
    }

    InstanceTrack track;
    track.track_id = track_id;
    track.object_id = object.object_id;
    track.state = object.active ? InstanceTrackState::kStable : InstanceTrackState::kInactive;
    track.semantic_id = object.semantic_id;
    track.label = object.label;
    track.center_world = object.center_world;
    track.size_m = object.size_m;
    track.yaw_rad = normalizeYaw(object.yaw_rad);
    track.confidence = object.confidence;
    track.confidence_mass = object.confidence_mass;
    track.object_quality_score = object.object_quality_score;
    track.geometry_score = object.geometry_score;
    track.geometry_shell_ratio = object.geometry_shell_ratio;
    track.geometry_extent_score = object.geometry_extent_score;
    track.geometry_leak_ratio = object.geometry_leak_ratio;
    track.geometry_cavity_ratio = object.geometry_cavity_ratio;
    track.geometry_in_box_points = object.geometry_in_box_points;
    track.geometry_shell_points = object.geometry_shell_points;
    track.geometry_unique_voxels = object.geometry_unique_voxels;
    track.geometry_expanded_points = object.geometry_expanded_points;
    track.geometry_bad_count = object.geometry_bad_count;
    track.support_count = object.support_count;
    track.high_quality_observation_count = object.high_quality_observation_count;
    track.high_quality_observation_mass = object.high_quality_observation_mass;
    track.missed_count = object.active ? 0 : config_.instance_inactive_after_missed;
    track.publishable = object.publishable;
    track.geometry_status = object.geometry_status;
    track.geometry_evaluation_obb_revision = object.geometry_evaluation_obb_revision;
    track.geometry_evaluation_map_version = object.geometry_evaluation_map_version;
    track.first_seen_ns = object.first_seen_ns;
    track.last_seen_ns = object.last_seen_ns;
    track.last_geometry_check_ns = object.last_geometry_check_ns;
    track.geometry_evaluated_center_world = object.geometry_evaluated_center_world;
    track.geometry_evaluated_size_m = object.geometry_evaluated_size_m;
    track.geometry_evaluated_yaw_rad = object.geometry_evaluated_yaw_rad;
    track.geometry_evaluation_reason = object.geometry_evaluation_reason;
    track.source_cameras = object.source_cameras;
    track.observation_timestamps_ns = object.observation_timestamps_ns;
    track.near_surface_voxels = object.near_surface_voxels;
    track.label_weights = object.label_weights;
    track.semantic_weights = object.semantic_weights;
    if (track.label_weights.empty() && !track.label.empty()) {
      track.label_weights[track.label] = std::max(0.0f, track.confidence_mass);
    }
    if (track.semantic_weights.empty() && track.semantic_id >= 0) {
      track.semantic_weights[track.semantic_id] = std::max(0.0f, track.confidence_mass);
    }
    track.bbox_quality_mass =
        std::max(track.confidence_mass,
                 track.object_quality_score *
                     static_cast<float>(std::max(1, track.support_count)));
    tracks_.push_back(std::move(track));
  }

  next_track_id_ = 0;
  for (int track_id : used_track_ids) {
    next_track_id_ = std::max(next_track_id_, track_id + 1);
  }
  frame_index_ = 0;
  last_geometry_maintenance_ns_ = 0;
  if (error != nullptr) {
    error->clear();
  }

  RunLogger::logGlobal("instance_map",
                       "loaded object_graph objects=" +
                           std::to_string(snapshot.objects.size()) +
                           " tracks=" + std::to_string(tracks_.size()) +
                           " relations=" + std::to_string(snapshot.relations.size()));
  return true;
}

void InstanceMapThread::run() {
  while (!stopRequested()) {
    InferenceResponse response;
    if (!response_queue_.waitPopFor(&response, std::chrono::milliseconds(50))) {
      continue;
    }
    if (!response.ok) {
      continue;
    }
    applyDetections(response);
  }
}

void InstanceMapThread::applyDetections(const InferenceResponse& response) {
  std::vector<InstanceObservation, Eigen::aligned_allocator<InstanceObservation>> observations;
  observations.reserve(response.detections.size());
  std::size_t rejected = 0;
  for (const RawDetection& detection : response.detections) {
    std::optional<InstanceObservation> observation = makeObservation(response, detection);
    if (observation) {
      observations.push_back(std::move(*observation));
    } else {
      ++rejected;
    }
  }

  std::size_t created = 0;
  std::size_t updated = 0;
  std::size_t objects_before = 0;
  std::size_t objects_after = 0;
  std::size_t tracks_before = 0;
  std::size_t tracks_after = 0;
  std::size_t removed_tentative = 0;
  std::size_t merged_duplicates = 0;
  std::size_t duplicate_rejected = 0;
  std::size_t geometry_checked = 0;
  std::size_t geometry_suppressed = 0;
  std::size_t geometry_recovered = 0;
  std::size_t geometry_deleted_empty = 0;
  std::size_t published_objects_after = 0;
  bool run_geometry = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++frame_index_;
    objects_before = object_graph_.objectCount();
    tracks_before = tracks_.size();

    std::vector<bool> track_matched(tracks_.size(), false);
    for (const InstanceObservation& observation : observations) {
      if (shouldRejectAsDuplicateOfConfirmed(observation)) {
        ++duplicate_rejected;
        continue;
      }
      const std::optional<std::size_t> track_index =
          findBestTrack(observation, track_matched);
      if (track_index) {
        updateTrack(&tracks_[*track_index], observation);
        track_matched[*track_index] = true;
        ++updated;
      } else {
        createTrack(observation);
        track_matched.push_back(true);
        ++created;
      }
    }

    ageUnmatchedTracks(track_matched, response.time_ns);
    removed_tentative = removeExpiredTentativeTracks();
    merged_duplicates += mergeDuplicateStableTracks();
    for (InstanceTrack& track : tracks_) {
      if (track.object_id >= 0) {
        object_graph_.updateNodeFromTrack(track);
      }
    }
    if (shouldRunGeometryMaintenance(response.time_ns)) {
      run_geometry = true;
      last_geometry_maintenance_ns_ = response.time_ns;
    }
    tracks_after = tracks_.size();
    objects_after = object_graph_.objectCount();
    published_objects_after =
        object_graph_.snapshotInstanceRecords(/*publishable_only=*/false).size();
  }

  if (run_geometry) {
    const std::shared_ptr<const GeometrySurfaceCache> surface_cache =
        map_projector_.geometrySurfaceCache();
    std::lock_guard<std::mutex> lock(mutex_);
    if (surface_cache) {
      geometry_checked = applyGeometryMaintenance(*surface_cache,
                                                  response.time_ns,
                                                  &geometry_suppressed,
                                                  &geometry_recovered,
                                                  &geometry_deleted_empty);
    }
    merged_duplicates += mergeDuplicateStableTracks();
    for (InstanceTrack& track : tracks_) {
      if (track.object_id >= 0) {
        object_graph_.updateNodeFromTrack(track);
      }
    }
    tracks_after = tracks_.size();
    objects_after = object_graph_.objectCount();
    published_objects_after =
        object_graph_.snapshotInstanceRecords(/*publishable_only=*/false).size();
  }

  if (!response.detections.empty() || !observations.empty() || run_geometry) {
    std::ostringstream stream;
    stream << "applied camera=" << response.camera_id
           << " t=" << response.time_ns
           << " raw=" << response.detections.size()
           << " accepted=" << (observations.size() - duplicate_rejected)
           << " rejected=" << (rejected + duplicate_rejected)
           << " duplicate_rejected=" << duplicate_rejected
           << " created_tracks=" << created
           << " updated_tracks=" << updated
           << " removed_tentative=" << removed_tentative
           << " merged_duplicates=" << merged_duplicates
           << " geometry_checked=" << geometry_checked
           << " geometry_suppressed=" << geometry_suppressed
           << " geometry_recovered=" << geometry_recovered
           << " geometry_deleted_empty=" << geometry_deleted_empty
           << " tracks=" << tracks_before << "->" << tracks_after
           << " objects=" << objects_before << "->" << objects_after
           << " published_objects=" << published_objects_after;
    RunLogger::logGlobal("instance_map", stream.str());
  }
}

std::optional<InstanceObservation> InstanceMapThread::makeObservation(
    const InferenceResponse& response,
    const RawDetection& detection) const {
  InstanceObservation observation;
  observation.time_ns = response.time_ns;
  observation.camera_id = response.camera_id;
  observation.detection = detection;
  observation.confidence = rawDetectionConfidence(detection);

  if (!isFiniteVector(detection.center_world) || !isFiniteVector(detection.size_m) ||
      !std::isfinite(detection.yaw_rad)) {
    return std::nullopt;
  }
  if (observation.confidence < config_.instance_min_confidence) {
    return std::nullopt;
  }
  if ((detection.size_m.array() < config_.instance_min_bbox_size_m).any() ||
      (detection.size_m.array() > config_.instance_max_bbox_size_m).any()) {
    return std::nullopt;
  }

  observation.bbox_quality = observationBboxQuality(response,
                                                    detection,
                                                    observation.confidence,
                                                    config_.boxer_input_size,
                                                    &observation.camera_distance_m);
  observation.near_surface_voxels = map_projector_.collectNearSurfaceVoxels(detection);
  return observation;
}

std::optional<std::size_t> InstanceMapThread::findBestTrack(
    const InstanceObservation& observation,
    const std::vector<bool>& track_reserved) const {
  float best_score = -1.0f;
  std::optional<std::size_t> best_index;
  for (std::size_t i = 0; i < tracks_.size(); ++i) {
    if (i < track_reserved.size() && track_reserved[i]) {
      continue;
    }
    const InstanceTrack& track = tracks_[i];
    if (!semanticCompatible(track, observation)) {
      continue;
    }

    const float iou = trackDetectionIou(track, observation.detection);
    const float distance =
        (track.center_world - observation.detection.center_world).norm();
    const bool passes_iou = iou >= config_.instance_match_iou_threshold;
    const bool passes_distance =
        config_.instance_match_center_distance_m > 0.0f &&
        distance <= config_.instance_match_center_distance_m;
    if (!passes_iou && !passes_distance) {
      continue;
    }

    const float distance_score =
        config_.instance_match_center_distance_m > kEpsilon
            ? std::max(0.0f, 1.0f - distance / config_.instance_match_center_distance_m)
            : 0.0f;
    const float score = iou + 0.25f * distance_score;
    if (score > best_score) {
      best_score = score;
      best_index = i;
    }
  }
  return best_index;
}

bool InstanceMapThread::shouldRejectAsDuplicateOfConfirmed(
    const InstanceObservation& observation) const {
  for (const InstanceTrack& track : tracks_) {
    if (track.object_id < 0) {
      continue;
    }
    if (!confirmedTrackDuplicatesDetection(track, observation.detection, config_)) {
      continue;
    }
    if (labelsCompatibleForDuplicate(track, observation.detection)) {
      return false;
    }
    return true;
  }
  return false;
}

void InstanceMapThread::createTrack(const InstanceObservation& observation) {
  const float promotion_weight = observationPromotionWeight(config_, observation);
  InstanceTrack track;
  track.track_id = next_track_id_++;
  track.semantic_id = observation.detection.semantic_id;
  track.label = observation.detection.label;
  track.center_world = observation.detection.center_world;
  track.size_m = observation.detection.size_m;
  track.yaw_rad = normalizeYaw(observation.detection.yaw_rad);
  track.confidence = observation.confidence;
  track.confidence_mass = observation.confidence;
  track.bbox_quality_mass = observation.bbox_quality * promotion_weight;
  track.support_count = 1;
  track.obb_revision = 1;
  track.first_seen_frame_index = frame_index_;
  track.last_seen_frame_index = frame_index_;
  track.first_seen_ns = observation.time_ns;
  track.last_seen_ns = observation.time_ns;
  track.source_cameras.push_back(observation.camera_id);
  track.observation_timestamps_ns.push_back(observation.time_ns);
  track.near_surface_voxels = observation.near_surface_voxels;
  recordObservationQuality(&track, observation, config_);
  if (!track.label.empty()) {
    track.label_weights[track.label] = observation.confidence * promotion_weight;
  }
  if (track.semantic_id >= 0) {
    track.semantic_weights[track.semantic_id] = observation.confidence * promotion_weight;
  }
  updateObjectQualityScore(&track);
  tracks_.push_back(std::move(track));
  maybePromoteOrUpdateObject(&tracks_.back());
}

void InstanceMapThread::updateTrack(InstanceTrack* track,
                                    const InstanceObservation& observation) {
  const float promotion_weight = observationPromotionWeight(config_, observation);
  const float old_mass =
      std::max(kEpsilon,
               std::min(track->bbox_quality_mass, config_.instance_fusion_prior_mass_cap));
  const float new_weight =
      std::max(kEpsilon, observation.bbox_quality * promotion_weight);
  const float total_weight = old_mass + new_weight;
  const float w_old = old_mass / total_weight;
  const float w_new = new_weight / total_weight;

  Eigen::Vector3f old_size = track->size_m;
  Eigen::Vector3f new_size = observation.detection.size_m;
  float old_yaw = track->yaw_rad;
  float new_yaw = normalizeYaw(observation.detection.yaw_rad);
  const float reference_yaw = weightedYawMean(old_yaw, w_old, new_yaw, w_new);
  alignBoxToReference(&old_size, &old_yaw, reference_yaw);
  alignBoxToReference(&new_size, &new_yaw, reference_yaw);

  track->center_world =
      w_old * track->center_world + w_new * observation.detection.center_world;
  track->size_m = w_old * old_size + w_new * new_size;
  track->yaw_rad = weightedYawMean(old_yaw, w_old, new_yaw, w_new);
  ++track->obb_revision;
  track->confidence_mass += observation.confidence;
  track->bbox_quality_mass += new_weight;
  ++track->support_count;
  track->confidence =
      clamp01(track->confidence_mass / static_cast<float>(track->support_count));
  track->missed_count = 0;
  track->last_seen_frame_index = frame_index_;
  track->last_seen_ns = observation.time_ns;
  appendUnique(&track->source_cameras, observation.camera_id);
  appendUnique(&track->observation_timestamps_ns, observation.time_ns);
  track->near_surface_voxels = observation.near_surface_voxels;
  recordObservationQuality(track, observation, config_);

  if (!observation.detection.label.empty()) {
    track->label_weights[observation.detection.label] +=
        observation.confidence * promotion_weight;
    track->label = bestWeightedKey(track->label_weights, track->label);
  }
  if (observation.detection.semantic_id >= 0) {
    track->semantic_weights[observation.detection.semantic_id] +=
        observation.confidence * promotion_weight;
    track->semantic_id = bestWeightedKey(track->semantic_weights, track->semantic_id);
  }

  if (track->object_id >= 0) {
    track->state = InstanceTrackState::kStable;
  }
  updateObjectQualityScore(track);
  maybePromoteOrUpdateObject(track);
}

void InstanceMapThread::ageUnmatchedTracks(const std::vector<bool>& track_matched,
                                           TimeNanoseconds response_time_ns) {
  for (std::size_t i = 0; i < tracks_.size(); ++i) {
    if (i < track_matched.size() && track_matched[i]) {
      continue;
    }
    InstanceTrack& track = tracks_[i];
    ++track.missed_count;
    if (track.object_id >= 0 &&
        track.missed_count >= config_.instance_inactive_after_missed) {
      track.state = InstanceTrackState::kInactive;
    }
    (void)response_time_ns;
  }
}

std::size_t InstanceMapThread::removeExpiredTentativeTracks() {
  const std::size_t before = tracks_.size();
  tracks_.erase(std::remove_if(tracks_.begin(),
                               tracks_.end(),
                               [this](const InstanceTrack& track) {
                                 return track.object_id < 0 &&
                                        track.state == InstanceTrackState::kTentative &&
                                        track.missed_count >=
                                            config_.instance_tentative_max_missed;
                               }),
                tracks_.end());
  return before - tracks_.size();
}

std::size_t InstanceMapThread::mergeDuplicateStableTracks() {
  std::size_t merged = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t i = 0; i < tracks_.size() && !changed; ++i) {
      const InstanceTrack& lhs = tracks_[i];
      if (lhs.object_id < 0 || !lhs.publishable) {
        continue;
      }
      for (std::size_t j = i + 1; j < tracks_.size(); ++j) {
        const InstanceTrack& rhs = tracks_[j];
        if (rhs.object_id < 0 || !rhs.publishable) {
          continue;
        }
        if (!stableTracksAreDuplicates(lhs, rhs, config_)) {
          continue;
        }

        std::size_t winner_index = i;
        std::size_t loser_index = j;
        const auto winner_key = [this](const InstanceTrack& track) {
          return std::make_tuple(trackAuthorityScore(track, config_),
                                 track.high_quality_observation_mass,
                                 track.object_quality_score,
                                 track.support_count,
                                 -track.object_id);
        };
        if (winner_key(rhs) > winner_key(lhs)) {
          winner_index = j;
          loser_index = i;
        }

        const int winner_track_id = tracks_[winner_index].track_id;
        const int loser_object_id = tracks_[loser_index].object_id;
        const InstanceTrack loser = tracks_[loser_index];
        InstanceTrack& winner = tracks_[winner_index];
        winner.confidence_mass += loser.confidence_mass;
        winner.support_count += loser.support_count;
        winner.confidence =
            clamp01(winner.confidence_mass / static_cast<float>(winner.support_count));
        winner.bbox_quality_mass += loser.bbox_quality_mass;
        winner.high_quality_observation_count += loser.high_quality_observation_count;
        winner.high_quality_observation_mass += loser.high_quality_observation_mass;
        winner.first_seen_frame_index =
            std::min(winner.first_seen_frame_index, loser.first_seen_frame_index);
        winner.last_seen_frame_index =
            std::max(winner.last_seen_frame_index, loser.last_seen_frame_index);
        winner.first_seen_ns = std::min(winner.first_seen_ns, loser.first_seen_ns);
        winner.last_seen_ns = std::max(winner.last_seen_ns, loser.last_seen_ns);
        winner.missed_count = std::min(winner.missed_count, loser.missed_count);
        winner.geometry_score = std::max(winner.geometry_score, loser.geometry_score);
        winner.geometry_shell_ratio =
            std::max(winner.geometry_shell_ratio, loser.geometry_shell_ratio);
        winner.geometry_extent_score =
            std::max(winner.geometry_extent_score, loser.geometry_extent_score);
        winner.geometry_leak_ratio =
            std::min(winner.geometry_leak_ratio, loser.geometry_leak_ratio);
        winner.geometry_cavity_ratio =
            std::min(winner.geometry_cavity_ratio, loser.geometry_cavity_ratio);
        winner.geometry_in_box_points =
            std::max(winner.geometry_in_box_points, loser.geometry_in_box_points);
        winner.geometry_shell_points =
            std::max(winner.geometry_shell_points, loser.geometry_shell_points);
        winner.geometry_unique_voxels =
            std::max(winner.geometry_unique_voxels, loser.geometry_unique_voxels);
        winner.geometry_expanded_points =
            std::max(winner.geometry_expanded_points, loser.geometry_expanded_points);
        winner.geometry_bad_count =
            std::min(winner.geometry_bad_count, loser.geometry_bad_count);
        winner.publishable = winner.publishable || loser.publishable;
        if (winner.geometry_status == InstanceGeometryStatus::kUnchecked &&
            loser.geometry_status != InstanceGeometryStatus::kUnchecked) {
          winner.geometry_status = loser.geometry_status;
        }
        if (geometryConfirmed(loser, config_) && !geometryConfirmed(winner, config_)) {
          winner.geometry_status = loser.geometry_status;
          winner.geometry_score = loser.geometry_score;
          winner.geometry_evaluation_obb_revision = loser.geometry_evaluation_obb_revision;
          winner.geometry_evaluation_map_version = loser.geometry_evaluation_map_version;
          winner.geometry_evaluated_center_world = loser.geometry_evaluated_center_world;
          winner.geometry_evaluated_size_m = loser.geometry_evaluated_size_m;
          winner.geometry_evaluated_yaw_rad = loser.geometry_evaluated_yaw_rad;
          winner.geometry_evaluation_reason = loser.geometry_evaluation_reason;
        }
        for (const std::string& camera : loser.source_cameras) {
          appendUnique(&winner.source_cameras, camera);
        }
        for (TimeNanoseconds timestamp : loser.observation_timestamps_ns) {
          appendUnique(&winner.observation_timestamps_ns, timestamp);
        }
        winner.observation_quality_history.insert(winner.observation_quality_history.end(),
                                                  loser.observation_quality_history.begin(),
                                                  loser.observation_quality_history.end());
        appendUniqueVoxelRefs(&winner.near_surface_voxels, loser.near_surface_voxels);
        for (const auto& [label, weight] : loser.label_weights) {
          winner.label_weights[label] += weight;
        }
        for (const auto& [semantic_id, weight] : loser.semantic_weights) {
          winner.semantic_weights[semantic_id] += weight;
        }
        winner.label = bestWeightedKey(winner.label_weights, winner.label);
        winner.semantic_id = bestWeightedKey(winner.semantic_weights, winner.semantic_id);
        updateObjectQualityScore(&winner);

        object_graph_.removeNode(loser_object_id);
        tracks_.erase(tracks_.begin() + static_cast<std::ptrdiff_t>(loser_index));
        auto winner_it = std::find_if(tracks_.begin(),
                                      tracks_.end(),
                                      [winner_track_id](const InstanceTrack& track) {
                                        return track.track_id == winner_track_id;
                                      });
        if (winner_it != tracks_.end()) {
          object_graph_.updateNodeFromTrack(*winner_it);
        }
        ++merged;
        changed = true;
        break;
      }
    }
  }
  return merged;
}

std::size_t InstanceMapThread::applyGeometryMaintenance(
    const MapBackendSnapshot& map_snapshot,
    TimeNanoseconds now_ns,
    std::size_t* suppressed,
    std::size_t* recovered,
    std::size_t* deleted_empty) {
  GeometrySurfaceCache surface_cache;
  surface_cache.map_version = map_snapshot.map_version;
  surface_cache.cache_rebuilds = map_snapshot.cache_rebuilds;
  surface_cache.tsdf_blocks = map_snapshot.tsdf_blocks;
  surface_cache.surface_voxels_scanned = map_snapshot.surface_voxels_scanned;
  surface_cache.has_map = map_snapshot.has_map;
  if (!map_snapshot.debug_surface_points.empty()) {
    surface_cache.surface_points = map_snapshot.debug_surface_points;
  } else {
    surface_cache.surface_points.reserve(map_snapshot.surface_points_world.size());
    for (const Eigen::Vector3f& point : map_snapshot.surface_points_world) {
      MapSurfacePoint surface_point;
      surface_point.position_world = point;
      surface_point.weight = 1.0f;
      surface_cache.surface_points.push_back(surface_point);
    }
  }
  return applyGeometryMaintenance(surface_cache,
                                  now_ns,
                                  suppressed,
                                  recovered,
                                  deleted_empty);
}

std::size_t InstanceMapThread::applyGeometryMaintenance(
    const GeometrySurfaceCache& surface_cache,
    TimeNanoseconds now_ns,
    std::size_t* suppressed,
    std::size_t* recovered,
    std::size_t* deleted_empty) {
  if (suppressed != nullptr) {
    *suppressed = 0;
  }
  if (recovered != nullptr) {
    *recovered = 0;
  }
  if (deleted_empty != nullptr) {
    *deleted_empty = 0;
  }
  if (!surface_cache.has_map || surface_cache.surface_points.empty()) {
    return 0;
  }

  std::vector<int> object_ids_to_delete;
  std::size_t checked = 0;
  for (InstanceTrack& track : tracks_) {
    if (track.object_id < 0) {
      continue;
    }
    if (track.state == InstanceTrackState::kInactive && !track.publishable &&
        track.geometry_bad_count >=
            config_.instance_geometry_inactive_delete_bad_count) {
      object_ids_to_delete.push_back(track.object_id);
      continue;
    }
    if (!shouldEvaluateTrackGeometry(track, surface_cache, now_ns, config_)) {
      continue;
    }

    const bool was_publishable = track.publishable;
    const GeometryEvaluation evaluation =
        evaluateGeometryAgainstSurface(track,
                                       surface_cache,
                                       config_.instance_geometry_shell_thickness_m,
                                       config_.instance_geometry_min_unique_voxels);
    ++checked;
    track.geometry_score = evaluation.score;
    track.geometry_shell_ratio = evaluation.shell_ratio;
    track.geometry_extent_score = evaluation.extent_score;
    track.geometry_leak_ratio = evaluation.leak_ratio;
    track.geometry_cavity_ratio = evaluation.cavity_ratio;
    track.geometry_in_box_points = evaluation.in_box_points;
    track.geometry_shell_points = evaluation.shell_points;
    track.geometry_unique_voxels = evaluation.unique_voxels;
    track.geometry_expanded_points = evaluation.expanded_points;
    track.last_geometry_check_ns = now_ns;
    track.geometry_evaluation_obb_revision = track.obb_revision;
    track.geometry_evaluation_map_version = surface_cache.map_version;
    track.geometry_evaluated_center_world = track.center_world;
    track.geometry_evaluated_size_m = track.size_m;
    track.geometry_evaluated_yaw_rad = track.yaw_rad;
    track.geometry_evaluation_reason = evaluation.reason;

    if (evaluation.in_box_points < config_.instance_geometry_empty_inside_points ||
        evaluation.unique_voxels < config_.instance_geometry_min_unique_voxels) {
      track.geometry_status = InstanceGeometryStatus::kEmpty;
      track.publishable = false;
      object_ids_to_delete.push_back(track.object_id);
      if (deleted_empty != nullptr) {
        ++(*deleted_empty);
      }
      continue;
    }

    if (evaluation.score >= config_.instance_geometry_confirm_score) {
      track.geometry_status = InstanceGeometryStatus::kGood;
      track.geometry_bad_count = 0;
      track.publishable = true;
      if (!was_publishable && recovered != nullptr) {
        ++(*recovered);
      }
    } else if (evaluation.score < config_.instance_geometry_suppress_score) {
      track.geometry_status = InstanceGeometryStatus::kBad;
      ++track.geometry_bad_count;
      if (track.geometry_bad_count >=
          config_.instance_geometry_failures_before_suppress) {
        track.publishable = false;
        if (was_publishable && suppressed != nullptr) {
          ++(*suppressed);
        }
      }
    } else {
      track.geometry_status =
          track.publishable ? InstanceGeometryStatus::kGood : InstanceGeometryStatus::kBad;
    }
    updateObjectQualityScore(&track);
    object_graph_.updateNodeFromTrack(track);
  }

  for (int object_id : object_ids_to_delete) {
    removeTrackAndObjectByObjectId(object_id);
  }
  return checked;
}

void InstanceMapThread::removeTrackAndObjectByObjectId(int object_id) {
  object_graph_.removeNode(object_id);
  tracks_.erase(std::remove_if(tracks_.begin(),
                               tracks_.end(),
                               [object_id](const InstanceTrack& track) {
                                 return track.object_id == object_id;
                               }),
                tracks_.end());
}

bool InstanceMapThread::shouldRunGeometryMaintenance(TimeNanoseconds now_ns) const {
  if (config_.instance_geometry_check_period_sec <= 0.0) {
    return false;
  }
  if (last_geometry_maintenance_ns_ == 0) {
    return true;
  }
  return now_ns - last_geometry_maintenance_ns_ >=
         secondsToNanoseconds(config_.instance_geometry_check_period_sec);
}

void InstanceMapThread::maybePromoteOrUpdateObject(InstanceTrack* track) {
  if (track->object_id < 0 && isPromotable(*track)) {
    track->state = InstanceTrackState::kStable;
    track->object_id = object_graph_.createNodeFromTrack(*track);
  }
  if (track->object_id >= 0) {
    object_graph_.updateNodeFromTrack(*track);
  }
}

bool InstanceMapThread::isPromotable(const InstanceTrack& track) const {
  return track.support_count >= config_.instance_min_support_count &&
         track.confidence >= config_.instance_object_min_confidence &&
         track.confidence_mass >= config_.instance_min_confidence_mass &&
         hasPromotionQuality(track, config_) &&
         isFiniteVector(track.center_world) && isFiniteVector(track.size_m) &&
         (track.size_m.array() >= config_.instance_min_bbox_size_m).all() &&
         (track.size_m.array() <= config_.instance_max_bbox_size_m).all();
}

}  // namespace roomie
