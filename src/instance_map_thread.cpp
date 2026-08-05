#include "roomie/pipeline/instance_map_thread.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <future>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

#include "roomie/dsg/observation_history.hpp"
#include "roomie/utils/run_logger.hpp"

namespace roomie {
namespace {

constexpr float kEpsilon = 1.0e-6f;
constexpr float kPi = 3.14159265358979323846f;

double elapsedMs(std::chrono::steady_clock::time_point start,
                 std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct Aabb {
  Eigen::Vector3f min = Eigen::Vector3f::Zero();
  Eigen::Vector3f max = Eigen::Vector3f::Zero();
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

MergeObjectsMutation durableMergeMutation(
    const SceneSnapshot& before,
    SceneObjectId retired_object_id,
    SceneObjectId canonical_object_id,
    std::size_t top_k) {
  MergeObjectsMutation mutation;
  mutation.retired_object_id = retired_object_id;
  mutation.canonical_object_id = canonical_object_id;
  const SceneObjectPtr retired = before.findExactObject(retired_object_id);
  const SceneObjectPtr canonical = before.findExactObject(canonical_object_id);
  if (!retired || !canonical || !retired->artifact || !canonical->artifact ||
      top_k == 0U) {
    return mutation;
  }

  std::map<std::string, ObjectSnapshotRef> unique;
  const auto collect = [&unique](const std::vector<ObjectSnapshotRef>& refs) {
    for (const ObjectSnapshotRef& reference : refs) {
      if (reference.evidence_hash.empty() ||
          reference.source_frame_asset_id.empty()) {
        return false;
      }
      auto [it, inserted] =
          unique.emplace(reference.evidence_hash, reference);
      if (!inserted && reference.quality > it->second.quality) {
        it->second = reference;
      }
    }
    return true;
  };
  if (!collect(canonical->artifact->snapshots) ||
      !collect(retired->artifact->snapshots) || unique.empty()) {
    return mutation;
  }

  std::vector<ObjectSnapshotRef> merged;
  merged.reserve(unique.size());
  for (auto& [hash, reference] : unique) {
    (void)hash;
    merged.push_back(std::move(reference));
  }
  std::sort(merged.begin(), merged.end(),
            [](const ObjectSnapshotRef& lhs,
               const ObjectSnapshotRef& rhs) {
              const float lhs_quality =
                  std::isfinite(lhs.quality) ? lhs.quality : -1.0f;
              const float rhs_quality =
                  std::isfinite(rhs.quality) ? rhs.quality : -1.0f;
              if (std::abs(lhs_quality - rhs_quality) > kEpsilon) {
                return lhs_quality > rhs_quality;
              }
              return lhs.evidence_hash < rhs.evidence_hash;
            });
  if (merged.size() > top_k) {
    merged.resize(top_k);
  }
  mutation.merged_snapshot_set_hash =
      snapshotSetHashForReferences(merged);
  if (!mutation.merged_snapshot_set_hash.empty()) {
    mutation.merged_snapshots = std::move(merged);
  }
  return mutation;
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

Eigen::Vector3f yawLocalDelta(const Eigen::Vector3f& delta_world, float yaw_rad) {
  const float c = std::cos(-yaw_rad);
  const float s = std::sin(-yaw_rad);
  return Eigen::Vector3f(c * delta_world.x() - s * delta_world.y(),
                         s * delta_world.x() + c * delta_world.y(),
                         delta_world.z());
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

std::pair<float, float> snapshotPatchBlurAndExposure(
    const ImageBuffer& image, const std::array<float, 4>& bbox) {
  if (image.empty() || image.channels <= 0) {
    return {1.0f, 1.0f};
  }
  const int x0 = std::clamp(
      static_cast<int>(std::floor(std::min(bbox[0], bbox[2]))),
      0, image.width - 1);
  const int y0 = std::clamp(
      static_cast<int>(std::floor(std::min(bbox[1], bbox[3]))),
      0, image.height - 1);
  const int x1 = std::clamp(
      static_cast<int>(std::ceil(std::max(bbox[0], bbox[2]))),
      x0 + 1, image.width);
  const int y1 = std::clamp(
      static_cast<int>(std::ceil(std::max(bbox[1], bbox[3]))),
      y0 + 1, image.height);
  const std::size_t pixel_stride = static_cast<std::size_t>(image.channels);
  const std::size_t row_stride =
      static_cast<std::size_t>(image.width) * pixel_stride;
  if (image.data.size() <
      static_cast<std::size_t>(image.height) * row_stride) {
    return {1.0f, 1.0f};
  }
  const int sample_step = std::max(
      1, static_cast<int>(std::sqrt(
             static_cast<double>((x1 - x0) * (y1 - y0)) / 4096.0)));
  double luminance_sum = 0.0;
  double gradient_sum = 0.0;
  std::size_t samples = 0;
  std::size_t gradient_samples = 0;
  const auto luminance = [&image, pixel_stride, row_stride](int x, int y) {
    const std::size_t offset = static_cast<std::size_t>(y) * row_stride +
                               static_cast<std::size_t>(x) * pixel_stride;
    if (image.channels == 1) {
      return static_cast<float>(image.data[offset]);
    }
    // rgb8 and bgr8 use the same coefficients after swapping red/blue only
    // up to a small quality heuristic error; no color fact is derived here.
    return 0.299f * static_cast<float>(image.data[offset]) +
           0.587f * static_cast<float>(image.data[offset + 1]) +
           0.114f * static_cast<float>(image.data[offset + 2]);
  };
  for (int y = y0; y < y1; y += sample_step) {
    for (int x = x0; x < x1; x += sample_step) {
      const float value = luminance(x, y);
      luminance_sum += value;
      ++samples;
      if (x + sample_step < x1) {
        gradient_sum += std::abs(value - luminance(x + sample_step, y));
        ++gradient_samples;
      }
      if (y + sample_step < y1) {
        gradient_sum += std::abs(value - luminance(x, y + sample_step));
        ++gradient_samples;
      }
    }
  }
  if (samples == 0) {
    return {1.0f, 1.0f};
  }
  const float mean = static_cast<float>(luminance_sum / samples);
  const float exposure = clamp01(
      1.0f - std::abs(mean - 127.5f) / 127.5f);
  const float blur =
      gradient_samples == 0
          ? 0.5f
          : clamp01(static_cast<float>(gradient_sum / gradient_samples) /
                    24.0f);
  return {blur, exposure};
}

SnapshotCandidate makeOnlineSnapshotCandidate(
    const InferenceResponse& response,
    const InstanceObservation& observation,
    std::shared_ptr<const ImageBuffer> full_frame) {
  SnapshotCandidate candidate;
  candidate.full_frame = std::move(full_frame);
  candidate.bbox_xyxy = observation.detection.box_xyxy;
  candidate.mask_source = SnapshotMaskSource::kBboxFallback;
  candidate.time_ns = observation.time_ns;
  candidate.camera_id = observation.camera_id;
  candidate.provenance = response.provenance;
  candidate.quality.confidence = observation.confidence;
  candidate.quality.edge_completeness = edgeCompletenessWeight(
      observation.detection,
      candidate.full_frame ? candidate.full_frame->width : 0);
  float unused_distance = 0.0f;
  candidate.quality.distance = distanceQualityWeight(
      response, observation.detection, &unused_distance);

  if (candidate.full_frame && !candidate.full_frame->empty()) {
    const float width = static_cast<float>(candidate.full_frame->width);
    const float height = static_cast<float>(candidate.full_frame->height);
    const float x0 = std::clamp(
        std::min(candidate.bbox_xyxy[0], candidate.bbox_xyxy[2]), 0.0f,
        width);
    const float x1 = std::clamp(
        std::max(candidate.bbox_xyxy[0], candidate.bbox_xyxy[2]), 0.0f,
        width);
    const float y0 = std::clamp(
        std::min(candidate.bbox_xyxy[1], candidate.bbox_xyxy[3]), 0.0f,
        height);
    const float y1 = std::clamp(
        std::max(candidate.bbox_xyxy[1], candidate.bbox_xyxy[3]), 0.0f,
        height);
    const float area_ratio = std::max(
        kEpsilon, (x1 - x0) * (y1 - y0) / std::max(1.0f, width * height));
    const float dx = (0.5f * (x0 + x1) - 0.5f * width) /
                     std::max(1.0f, 0.5f * width);
    const float dy = (0.5f * (y0 + y1) - 0.5f * height) /
                     std::max(1.0f, 0.5f * height);
    candidate.quality.position =
        clamp01(1.0f - 0.5f * std::sqrt(dx * dx + dy * dy));
    candidate.quality.size = clamp01(std::sqrt(area_ratio) / 0.35f);
    const float unclipped_width = std::max(
        kEpsilon, std::abs(candidate.bbox_xyxy[2] - candidate.bbox_xyxy[0]));
    const float unclipped_height = std::max(
        kEpsilon, std::abs(candidate.bbox_xyxy[3] - candidate.bbox_xyxy[1]));
    candidate.quality.truncation = clamp01(
        (x1 - x0) * (y1 - y0) /
        (unclipped_width * unclipped_height));
    const auto [blur, exposure] = snapshotPatchBlurAndExposure(
        *candidate.full_frame, candidate.bbox_xyxy);
    candidate.quality.blur = blur;
    candidate.quality.exposure = exposure;
    candidate.viewpoint.scale = area_ratio;
  }

  if (response.has_camera_pose) {
    const Eigen::Vector3f camera_object =
        response.T_world_camera.inverse() * observation.detection.center_world;
    candidate.viewpoint.azimuth_rad =
        std::atan2(camera_object.x(), camera_object.z());
    candidate.viewpoint.elevation_rad =
        std::atan2(-camera_object.y(),
                   std::hypot(camera_object.x(), camera_object.z()));
  }
  candidate.viewpoint.scale =
      std::max(candidate.viewpoint.scale, kEpsilon);
  return candidate;
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

std::string diagnosticsLabel(std::string label) {
  if (label.empty()) {
    return "object";
  }
  for (char& c : label) {
    if (std::isspace(static_cast<unsigned char>(c)) || c == '=') {
      c = '_';
    }
  }
  return label;
}

bool shouldIgnoreDetectionLabel(const std::string& label) {
  const std::string normalized = lowercase(label);
  return normalized == "bathhub" || normalized == "bath hub";
}

std::string formatLabelCounts(const std::map<std::string, std::size_t>& counts) {
  if (counts.empty()) {
    return "{}";
  }
  std::ostringstream stream;
  stream << "{";
  bool first = true;
  for (const auto& [label, count] : counts) {
    if (!first) {
      stream << ",";
    }
    first = false;
    stream << label << ":" << count;
  }
  stream << "}";
  return stream.str();
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

bool isSmallDuplicateTrack(const InstanceTrack& track, const PipelineConfig& config) {
  if (config.instance_small_duplicate_max_volume_m3 <= 0.0f ||
      config.instance_small_duplicate_max_extent_m <= 0.0f ||
      (track.size_m.array() <= 0.0f).any()) {
    return false;
  }
  const float track_volume = sizeVolume(track.size_m);
  const float max_extent = track.size_m.cwiseMax(Eigen::Vector3f::Zero()).maxCoeff();
  return track_volume > 0.0f &&
         track_volume <= config.instance_small_duplicate_max_volume_m3 &&
         max_extent <= config.instance_small_duplicate_max_extent_m;
}

bool smallStableTracksAreDuplicates(const InstanceTrack& lhs,
                                    const InstanceTrack& rhs,
                                    const PipelineConfig& config) {
  if (!isSmallDuplicateTrack(lhs, config) ||
      !isSmallDuplicateTrack(rhs, config)) {
    return false;
  }
  const float size_ratio = sizeRatioScore(lhs.size_m, rhs.size_m);
  if (size_ratio < config.instance_small_duplicate_size_ratio_min) {
    return false;
  }
  const float iou = obbIou(lhs, rhs);
  if (iou >= config.instance_small_duplicate_iou_threshold) {
    return true;
  }
  const float lhs_diag = lhs.size_m.cwiseMax(Eigen::Vector3f::Zero()).norm();
  const float rhs_diag = rhs.size_m.cwiseMax(Eigen::Vector3f::Zero()).norm();
  const float center_gate =
      config.instance_small_duplicate_center_ratio * std::max(lhs_diag, rhs_diag);
  return center_gate > 0.0f && centerDistance(lhs, rhs) <= center_gate;
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

bool isLargeFurnitureLabel(const std::string& label) {
  const std::string normalized = lowercase(label);
  const auto in_group = [&normalized](std::initializer_list<const char*> group) {
    return std::find(group.begin(), group.end(), normalized) != group.end();
  };
  return in_group({"bed",
                   "bunk bed",
                   "sofa bed",
                   "futon",
                   "sofa",
                   "couch",
                   "table",
                   "coffee table",
                   "desk",
                   "cabinet",
                   "cupboard",
                   "wardrobe",
                   "dresser",
                   "storage",
                   "shelf",
                   "bookcase",
                   "refrigerator",
                   "dishwasher",
                   "stove",
                   "oven",
                   "tv",
                   "television set"});
}

bool isLargeFurnitureTrack(const InstanceTrack& track,
                           const InstanceObservation& observation,
                           const PipelineConfig& config) {
  const bool label_matches = isLargeFurnitureLabel(track.label) ||
                             isLargeFurnitureLabel(observation.detection.label);
  if (!label_matches) {
    return false;
  }
  return sizeVolume(track.size_m) >=
         config.instance_confirmed_geometry_large_min_volume_m3;
}

bool isFarObservationForConfirmedGeometry(const InstanceObservation& observation,
                                          const PipelineConfig& config) {
  return config.instance_far_observation_distance_m > 0.0f &&
         observation.camera_distance_m > 0.0f &&
         observation.camera_distance_m >= config.instance_far_observation_distance_m;
}

bool hasCloseObservationGeometryLock(const InstanceTrack& track,
                                     const PipelineConfig& config) {
  if (track.object_id < 0 || track.state == InstanceTrackState::kTentative) {
    return false;
  }
  return track.high_quality_observation_count >= config.instance_high_quality_min_count ||
         track.high_quality_observation_mass >= config.instance_high_quality_min_mass;
}

bool shouldFreezeFarObservationForCloseStableObject(
    const InstanceTrack& track,
    const InstanceObservation& observation,
    const PipelineConfig& config) {
  return hasCloseObservationGeometryLock(track, config) &&
         isFarObservationForConfirmedGeometry(observation, config);
}

bool wouldFarObservationShiftLastGoodObb(const InstanceTrack& track,
                                         const InstanceObservation& observation,
                                         const PipelineConfig& config,
                                         float old_mass,
                                         float new_weight) {
  if (!isFarObservationForConfirmedGeometry(observation, config) ||
      config.instance_confirmed_geometry_far_center_shift_ratio <= 0.0f ||
      track.geometry_evaluation_obb_revision == 0 ||
      !isFiniteVector(track.geometry_evaluated_center_world) ||
      !isFiniteVector(track.geometry_evaluated_size_m) ||
      (track.geometry_evaluated_size_m.array() <= 0.0f).any()) {
    return false;
  }
  const float total_weight = old_mass + new_weight;
  if (total_weight <= kEpsilon) {
    return false;
  }
  const Eigen::Vector3f candidate_center =
      (old_mass * track.center_world +
       new_weight * observation.detection.center_world) /
      total_weight;
  const Eigen::Vector3f delta_world =
      candidate_center - track.geometry_evaluated_center_world;
  if (delta_world.norm() <
      config.instance_confirmed_geometry_far_center_shift_min_m) {
    return false;
  }
  const Eigen::Vector3f local_delta =
      yawLocalDelta(delta_world, track.geometry_evaluated_yaw_rad);
  float max_axis_ratio = 0.0f;
  for (int axis = 0; axis < 3; ++axis) {
    const float denom =
        std::max(kEpsilon,
                 std::max(std::abs(track.geometry_evaluated_size_m[axis]),
                          config.instance_confirmed_geometry_center_shift_min_extent_m));
    max_axis_ratio = std::max(max_axis_ratio,
                              std::abs(local_delta[axis]) / denom);
  }
  return max_axis_ratio >=
         config.instance_confirmed_geometry_far_center_shift_ratio;
}

std::string confirmedGeometryUpdateSuppressionReason(
    const InstanceTrack& track,
    const InstanceObservation& observation,
    const PipelineConfig& config,
    float old_mass,
    float new_weight) {
  if (shouldFreezeFarObservationForCloseStableObject(track, observation, config)) {
    return "far_close_stable_lock";
  }
  if (track.object_id < 0 || !track.publishable ||
      !geometryConfirmed(track, config)) {
    return "";
  }
  const float edge_weight =
      edgeCompletenessWeight(observation.detection, config.boxer_input_size);
  if (isLargeFurnitureTrack(track, observation, config) &&
      edge_weight < config.instance_confirmed_geometry_edge_freeze_weight) {
    return "edge";
  }
  if (wouldFarObservationShiftLastGoodObb(track,
                                          observation,
                                          config,
                                          old_mass,
                                          new_weight)) {
    return "far_center_shift";
  }
  return "";
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
  retainRecentObservationQuality(
      &track->observation_quality_history,
      config.instance_observation_history_capacity);
  if (high_quality) {
    ++track->high_quality_observation_count;
    track->high_quality_observation_mass += observation.bbox_quality;
  }
}

bool hasPromotionQuality(const InstanceTrack& track, const PipelineConfig& config) {
  return track.high_quality_observation_count >= config.instance_high_quality_min_count ||
         track.high_quality_observation_mass >= config.instance_high_quality_min_mass;
}

bool hasBasePromotionEvidence(const InstanceTrack& track, const PipelineConfig& config) {
  return track.support_count >= config.instance_min_support_count &&
         track.confidence >= config.instance_object_min_confidence &&
         track.confidence_mass >= config.instance_min_confidence_mass &&
         isFiniteVector(track.center_world) && isFiniteVector(track.size_m) &&
         (track.size_m.array() >= config.instance_min_bbox_size_m).all() &&
         (track.size_m.array() <= config.instance_max_bbox_size_m).all();
}

bool hasMatureDuplicateAuthority(const InstanceTrack& track,
                                 const PipelineConfig& config) {
  if (!geometryConfirmed(track, config)) {
    return false;
  }
  const bool mature_quality =
      track.high_quality_observation_count >= config.instance_high_quality_min_count + 1 ||
      track.high_quality_observation_mass >=
          config.instance_high_quality_min_mass +
              config.instance_quality_observation_min_quality;
  const bool mature_support =
      track.support_count >= std::max(config.instance_min_support_count + 3,
                                      config.instance_min_support_count * 2);
  return mature_quality || mature_support;
}

std::size_t countPromotionQualityBlocked(
    const std::vector<InstanceTrack, Eigen::aligned_allocator<InstanceTrack>>& tracks,
    const PipelineConfig& config,
    std::map<std::string, std::size_t>* by_label = nullptr) {
  if (by_label != nullptr) {
    by_label->clear();
  }
  std::size_t count = 0;
  for (const InstanceTrack& track : tracks) {
    if (track.object_id < 0 && hasBasePromotionEvidence(track, config) &&
        !hasPromotionQuality(track, config)) {
      ++count;
      if (by_label != nullptr) {
        ++(*by_label)[diagnosticsLabel(track.label)];
      }
    }
  }
  return count;
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
  record.snapshot_image_index = track.snapshot.image_index;
  record.snapshot_bbox_xyxy = track.snapshot.bbox_xyxy;
  record.snapshot_quality = track.snapshot.quality;
  record.near_surface_voxels = track.near_surface_voxels;
  return record;
}

ObjectSnapshotRemakerConfig snapshotRemakerConfigFromPipeline(
    const PipelineConfig& config) {
  ObjectSnapshotRemakerConfig remaker_config;
  remaker_config.enabled =
      config.load_scene_graph && config.freeze_instances && config.snapshot_remake_enabled;
  remaker_config.staging_dir = config.snapshot_staging_dir;
  remaker_config.first_min_quality = config.instance_snapshot_first_min_quality;
  remaker_config.min_quality = config.instance_snapshot_min_quality;
  remaker_config.min_box_area_px = config.instance_snapshot_min_box_area_px;
  remaker_config.position_weight = config.instance_snapshot_position_weight;
  remaker_config.size_weight = config.instance_snapshot_size_weight;
  remaker_config.replace_min_quality_delta =
      config.instance_snapshot_replace_min_quality_delta;
  remaker_config.replace_min_quality_ratio =
      config.instance_snapshot_replace_min_quality_ratio;
  return remaker_config;
}

std::shared_ptr<const SceneState> boundedObservationHistoryState(
    const SceneState& source,
    std::size_t capacity) {
  auto bounded = std::make_shared<SceneState>(source);
  if (source.objects) {
    SceneObjectTable objects = *source.objects;
    bool objects_changed = false;
    for (auto& [object_id, object] : objects) {
      (void)object_id;
      if (!object || !object->semantic ||
          object->semantic->observation_timestamps_ns.size() <= capacity) {
        continue;
      }
      auto semantic =
          std::make_shared<SemanticComponent>(*object->semantic);
      retainRecentObservationTimestamps(
          &semantic->observation_timestamps_ns, capacity);
      auto replacement = std::make_shared<SceneObject>(*object);
      replacement->semantic = std::move(semantic);
      object = std::move(replacement);
      objects_changed = true;
    }
    if (objects_changed) {
      bounded->objects =
          std::make_shared<const SceneObjectTable>(std::move(objects));
    }
  }
  if (source.tracks) {
    SceneTrackTable tracks = *source.tracks;
    bool tracks_changed = false;
    for (auto& [track_id, track] : tracks) {
      (void)track_id;
      if (!track ||
          (track->observation_timestamps_ns.size() <= capacity &&
           track->observation_quality_history.size() <= capacity)) {
        continue;
      }
      auto replacement = std::make_shared<InstanceTrack>(*track);
      retainRecentObservationHistory(replacement.get(), capacity);
      track = std::move(replacement);
      tracks_changed = true;
    }
    if (tracks_changed) {
      bounded->tracks =
          std::make_shared<const SceneTrackTable>(std::move(tracks));
    }
  }
  return bounded;
}

}  // namespace

InstanceMapThread::InstanceMapThread(ThreadSafeQueue<InferenceResponse>& response_queue,
                                     const MapProjector& map_projector,
                                     PipelineConfig config)
    : WorkerThread("instance_map_thread"),
      response_queue_(response_queue),
      scene_command_queue_(
          std::max<std::size_t>(64U, config.pending_frame_limit),
          ChannelPolicy::kReliableBlocking),
      map_projector_(map_projector),
      config_(std::move(config)),
      snapshot_remaker_(snapshotRemakerConfigFromPipeline(config_)),
      published_scene_state_(reducer_.snapshot().statePtr()),
      pending_snapshot_controls_(config_.snapshot_control_queue_size) {}

bool InstanceMapThread::enqueueDetections(InferenceResponse response) {
  return response_queue_.push(std::move(response)).accepted();
}

std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
InstanceMapThread::snapshotInstances() const {
  ObjectGraph graph;
  graph.loadSnapshot(sceneSnapshot().materializeObjectGraph());
  return graph.snapshotInstanceRecords(/*publishable_only=*/false);
}

std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
InstanceMapThread::snapshotTrackedInstances() const {
  std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>> records;
  const SceneSnapshot snapshot = sceneSnapshot();
  records.reserve(snapshot.tracks().size());
  for (const auto& [track_id, track] : snapshot.tracks()) {
    (void)track_id;
    if (track) {
      records.push_back(recordFromTrack(*track));
    }
  }
  return records;
}

ObjectGraphSnapshot InstanceMapThread::snapshotObjectGraph() const {
  return sceneSnapshot().materializeObjectGraph();
}

SceneSnapshot InstanceMapThread::sceneSnapshot() const {
  return SceneSnapshot(std::atomic_load(&published_scene_state_));
}

bool InstanceMapThread::enqueueSceneCommand(
    SceneCommand command, SceneCommandCompletion completion) {
  const PushResult<QueuedSceneCommand> result =
      scene_command_queue_.push(
          QueuedSceneCommand{std::move(command), std::move(completion)});
  if (!result.accepted()) {
    RunLogger::logGlobal(
        "scene_reducer",
        "scene command rejected by queue outcome=" +
            std::to_string(static_cast<int>(result.outcome)));
  }
  return result.accepted();
}

std::optional<SceneApplyResult> InstanceMapThread::applySceneCommandAndWait(
    SceneCommand command, std::chrono::milliseconds timeout) {
  auto promise = std::make_shared<std::promise<SceneApplyResult>>();
  std::future<SceneApplyResult> future = promise->get_future();
  if (!enqueueSceneCommand(
          std::move(command),
          [promise](const SceneApplyResult& result) {
            try {
              promise->set_value(result);
            } catch (const std::future_error&) {
              // A timed-out caller may have already abandoned the future;
              // reducer completion is still valid and needs no cancellation.
            }
          })) {
    return std::nullopt;
  }
  if (timeout < std::chrono::milliseconds::zero()) {
    timeout = std::chrono::milliseconds::zero();
  }
  if (future.wait_for(timeout) != std::future_status::ready) {
    return std::nullopt;
  }
  return future.get();
}

void InstanceMapThread::setOnlineSnapshotWorker(
    OnlineSnapshotWorker* worker) {
  online_snapshot_worker_ = worker;
}

std::size_t InstanceMapThread::closeSnapshotControlsForShutdown() {
  const std::size_t abandoned =
      pending_snapshot_controls_.closeAndAbandon();
  if (abandoned != 0) {
    RunLogger::logGlobal(
        "snapshot_control",
        "abandoned controls after snapshot worker shutdown count=" +
            std::to_string(abandoned));
  }
  drain_cv_.notify_all();
  return abandoned;
}

bool InstanceMapThread::snapshotControlFaulted() const {
  return snapshot_control_faulted_.load(std::memory_order_acquire);
}

bool InstanceMapThread::requestDurabilityAck(SceneRevision revision) {
  SceneRevision current = pending_durable_ack_.load(std::memory_order_acquire);
  while (current < revision &&
         !pending_durable_ack_.compare_exchange_weak(
             current, revision, std::memory_order_acq_rel,
             std::memory_order_acquire)) {
  }
  drain_cv_.notify_all();
  return true;
}

SnapshotControlPushOutcome InstanceMapThread::enqueueSnapshotControl(
    SnapshotControl control) {
  if (online_snapshot_worker_ == nullptr) {
    return SnapshotControlPushOutcome::kRejectedNoWorker;
  }
  // Give the worker a chance to accept the current head before applying the
  // local capacity limit to a distinct ownership transition.
  (void)flushPendingSnapshotControls();
  const SnapshotControlKind kind = control.kind;
  const int first_id = control.first_id;
  const int second_id = control.second_id;
  const SceneRevision scene_revision = control.scene_revision;
  const SnapshotControlPushOutcome outcome =
      pending_snapshot_controls_.push(std::move(control));
  if (outcome == SnapshotControlPushOutcome::kRejectedCapacity ||
      outcome == SnapshotControlPushOutcome::kRejectedClosed) {
    const SnapshotControlQueueStats stats = pending_snapshot_controls_.stats();
    RunLogger::logGlobal(
        "snapshot_control",
        "control rejected outcome=" +
            std::to_string(static_cast<int>(outcome)) + " kind=" +
            std::to_string(static_cast<int>(kind)) + " first_id=" +
            std::to_string(first_id) + " second_id=" +
            std::to_string(second_id) + " scene_revision=" +
            std::to_string(scene_revision) + " depth=" +
            std::to_string(stats.depth) + " capacity=" +
            std::to_string(stats.capacity) + " rejected_capacity=" +
            std::to_string(stats.rejected_capacity) + " rejected_closed=" +
            std::to_string(stats.rejected_closed));
  }
  if (outcome == SnapshotControlPushOutcome::kRejectedCapacity) {
    const bool already_faulted =
        snapshot_control_faulted_.exchange(true, std::memory_order_acq_rel);
    if (!already_faulted) {
      RunLogger::logGlobal(
          "snapshot_control",
          "terminal control-admission fault activated at scene_revision=" +
              std::to_string(scene_revision) +
              "; later persistent content will be rejected pre-reducer");
    }
  }
  drain_cv_.notify_all();
  if (flushPendingSnapshotControls()) {
    drain_cv_.notify_all();
  }
  return outcome;
}

bool InstanceMapThread::flushPendingSnapshotControls() {
  bool progressed = false;
  while (online_snapshot_worker_ != nullptr) {
    const std::optional<SnapshotControl> control =
        pending_snapshot_controls_.front();
    if (!control) {
      break;
    }
    SnapshotWorkerEnqueueResult result;
    switch (control->kind) {
      case SnapshotControlKind::kPromote:
        result = online_snapshot_worker_->tryEnqueuePromotion(
            control->first_id, control->second_id, control->now_ns);
        break;
      case SnapshotControlKind::kMerge:
        result = online_snapshot_worker_->tryEnqueueMerge(
            control->first_id, control->second_id, control->now_ns);
        break;
      case SnapshotControlKind::kDropTentative:
        result = online_snapshot_worker_->tryEnqueueDropTentative(
            control->first_id, control->now_ns);
        break;
      case SnapshotControlKind::kEraseObject:
        result = online_snapshot_worker_->tryEnqueueEraseObject(
            control->first_id, control->now_ns);
        break;
    }
    if (!result.accepted()) {
      break;
    }
    (void)pending_snapshot_controls_.popFront();
    progressed = true;
  }
  return progressed;
}

void InstanceMapThread::setSceneCommitSink(
    std::function<bool(const SceneApplyResult&)> sink) {
  scene_commit_sink_ = std::move(sink);
}

void InstanceMapThread::setSceneContentAdmission(
    std::function<bool()> admission) {
  scene_content_admission_ = std::move(admission);
}

void InstanceMapThread::setSceneCommitObserver(
    std::function<void(const SceneApplyResult&)> observer) {
  scene_commit_observer_ = std::move(observer);
}

bool InstanceMapThread::waitUntilIdle(std::chrono::milliseconds timeout) {
  if (timeout < std::chrono::milliseconds::zero()) {
    timeout = std::chrono::milliseconds::zero();
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  const auto idle = [this]() {
    return response_queue_.empty() && scene_command_queue_.empty() &&
           pending_durable_ack_.load() == 0 &&
           pending_snapshot_controls_.empty() && !processing_work_.load();
  };
  std::unique_lock<std::mutex> lock(drain_mutex_);
  while (true) {
    if (!idle() && !drain_cv_.wait_until(lock, deadline, idle)) {
      return false;
    }
    // Require a stable idle window to close the dequeue-before-processing
    // observation gap without putting a mutex around the channel itself.
    const auto stable_until = std::min(
        deadline,
        std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
    if (!drain_cv_.wait_until(lock, stable_until, [&idle]() {
          return !idle();
        })) {
      return idle();
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
  }
}

SnapshotControlQueueStats InstanceMapThread::snapshotControlStats() const {
  return pending_snapshot_controls_.stats();
}

bool InstanceMapThread::prepareSceneGraphForSave(
    ObjectGraphSnapshot* snapshot,
    const std::filesystem::path& snapshot_image_dir,
    const std::string& snapshot_uri_prefix,
    std::string* error) const {
  if (snapshot == nullptr) {
    if (error != nullptr) {
      *error = "null ObjectGraphSnapshot output";
    }
    return false;
  }
  *snapshot = sceneSnapshot().materializeObjectGraph();
  if (!config_.snapshot_remake_enabled || online_snapshot_worker_ != nullptr) {
    if (error != nullptr) {
      error->clear();
    }
    return true;
  }
  return snapshot_remaker_.populateSnapshot(snapshot,
                                            snapshot_image_dir,
                                            snapshot_uri_prefix,
                                            error);
}

bool InstanceMapThread::loadObjectGraphSnapshot(const ObjectGraphSnapshot& snapshot,
                                                std::string* error) {
  return loadInitialSnapshot(snapshot, nullptr, error);
}

bool InstanceMapThread::loadSceneSnapshot(const SceneSnapshot& snapshot,
                                          std::string* error) {
  return loadInitialSnapshot(
      snapshot.materializeObjectGraph(), &snapshot, error);
}

bool InstanceMapThread::loadInitialSnapshot(
    const ObjectGraphSnapshot& snapshot,
    const SceneSnapshot* restored_scene,
    std::string* error) {
  std::lock_guard<std::mutex> lock(mutex_);
  ObjectGraphSnapshot bounded_graph = snapshot;
  for (ObjectNode& object : bounded_graph.objects) {
    retainRecentObservationHistory(
        &object, config_.instance_observation_history_capacity);
  }
  const std::shared_ptr<const SceneState> bounded_restored_state =
      restored_scene != nullptr
          ? boundedObservationHistoryState(
                *restored_scene->statePtr(),
                config_.instance_observation_history_capacity)
          : nullptr;
  const SceneSnapshot bounded_restored_scene(bounded_restored_state);
  const SceneSnapshot* effective_restored_scene =
      restored_scene != nullptr ? &bounded_restored_scene : nullptr;

  object_graph_.loadSnapshot(bounded_graph);
  snapshot_remaker_.loadSnapshot(bounded_graph);
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

  for (const ObjectNode& object : bounded_graph.objects) {
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
    track.snapshot = object.snapshot;
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

  if (effective_restored_scene != nullptr &&
      !effective_restored_scene->tracks().empty()) {
    tracks_.clear();
    used_track_ids.clear();
    for (const auto& [track_id, track] : effective_restored_scene->tracks()) {
      if (!track || track_id < 0) {
        continue;
      }
      tracks_.push_back(*track);
      used_track_ids.insert(track_id);
    }
  }

  next_track_id_ = 0;
  for (int track_id : used_track_ids) {
    next_track_id_ = std::max(next_track_id_, track_id + 1);
  }
  frame_index_ = 0;
  LoadSceneCommand load_command;
  if (effective_restored_scene != nullptr) {
    load_command.restored_state = effective_restored_scene->statePtr();
  }
  load_command.graph = bounded_graph;
  load_command.tracks = tracks_;
  if (effective_restored_scene != nullptr) {
    for (const auto& [retired_id, alias] :
         effective_restored_scene->aliases()) {
      (void)retired_id;
      load_command.aliases.push_back(alias);
    }
    for (const auto& [object_id, tombstone] :
         effective_restored_scene->tombstones()) {
      (void)object_id;
      load_command.tombstones.push_back(tombstone);
    }
    load_command.restored_revision = effective_restored_scene->revision();
    load_command.durable_revision =
        effective_restored_scene->durableRevision();
    load_command.recent_observation_frames =
        effective_restored_scene->statePtr()->recent_observation_frames;
    load_command.observation_watermarks =
        effective_restored_scene->statePtr()->observation_watermarks;
    // A restored scene keeps object/revision watermarks but never inherits a
    // previous process's map epoch. MapActor publishes the new epoch through
    // AdvanceSurfaceCommand after startup.
    load_command.latest_surface = SurfaceStamp{};
  }
  const SceneApplyResult load_result =
      reducer_.apply(SceneCommand{std::move(load_command)});
  if (!load_result.accepted()) {
    if (error != nullptr) {
      *error = load_result.reason;
    }
    return false;
  }
  publishReducerResult(load_result, /*persist_content_commit=*/true);
  if (error != nullptr) {
    error->clear();
  }

  RunLogger::logGlobal("instance_map",
                       "loaded object_graph objects=" +
                           std::to_string(bounded_graph.objects.size()) +
                           " tracks=" + std::to_string(tracks_.size()) +
                           " relations=" +
                           std::to_string(bounded_graph.relations.size()));
  return true;
}

void InstanceMapThread::refreshAssociationWorkingSet(
    const SceneSnapshot& snapshot) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Association remains an algorithmic working set, not a second current
  // scene. Recreate it from the reducer's pinned immutable snapshot before
  // every observation so a rejected command can never contaminate the next
  // frame.
  object_graph_.loadSnapshot(snapshot.materializeObjectGraph());
  tracks_.clear();
  tracks_.reserve(snapshot.tracks().size());
  std::uint64_t highest_frame_index = frame_index_;
  for (const auto& [track_id, track] : snapshot.tracks()) {
    if (!track || track_id < 0) {
      continue;
    }
    tracks_.push_back(*track);
    next_track_id_ = std::max(next_track_id_, track_id + 1);
    highest_frame_index =
        std::max(highest_frame_index, track->last_seen_frame_index);
  }
  frame_index_ = highest_frame_index;
  pending_reducer_merges_.clear();
}

void InstanceMapThread::run() {
  while (!stopRequested() || !response_queue_.empty() ||
         !scene_command_queue_.empty() ||
         pending_durable_ack_.load(std::memory_order_acquire) != 0 ||
         (!stopRequested() && !pending_snapshot_controls_.empty())) {
    if (flushPendingSnapshotControls()) {
      drain_cv_.notify_all();
    }
    if (applyPendingDurabilityAck()) {
      drain_cv_.notify_all();
    }
    QueuedSceneCommand queued;
    bool applied_command = false;
    // Bound each command burst so a busy artifact/geometry producer cannot
    // indefinitely starve the observation stream.
    for (std::size_t i = 0; i < 16 && scene_command_queue_.tryPop(&queued);
         ++i) {
      processing_work_.store(true);
      try {
        const SceneApplyResult result =
            applyQueuedSceneCommand(std::move(queued.command));
        if (queued.completion) {
          queued.completion(result);
        }
      } catch (const std::exception& error) {
        RunLogger::logGlobal(
            "scene_reducer",
            "queued command actor exception: " + std::string(error.what()));
      } catch (...) {
        RunLogger::logGlobal(
            "scene_reducer",
            "queued command actor non-standard exception");
      }
      processing_work_.store(false);
      drain_cv_.notify_all();
      applied_command = true;
    }

    InferenceResponse response;
    const bool have_response =
        applied_command
            ? response_queue_.tryPop(&response)
            : response_queue_.waitPopFor(&response,
                                         std::chrono::milliseconds(20));
    if (!have_response) {
      continue;
    }
    if (!response.ok) {
      continue;
    }
    if (!sceneContentAdmissionAllowed()) {
      RunLogger::logGlobal(
          "scene_persistence",
          "observation_rejected_pre_reducer frame_id=" +
              std::to_string(response.provenance.frame_id) +
              " reason=" +
              sceneContentAdmissionRejectionReason());
      continue;
    }
    processing_work_.store(true);
    try {
      applyDetections(response);
    } catch (const std::exception& error) {
      RunLogger::logGlobal(
          "instance_map",
          "observation actor exception: " + std::string(error.what()));
    } catch (...) {
      RunLogger::logGlobal(
          "instance_map",
          "observation actor non-standard exception");
    }
    processing_work_.store(false);
    drain_cv_.notify_all();
  }
  const std::size_t abandoned = pending_snapshot_controls_.abandonAll();
  if (abandoned != 0) {
    const SnapshotControlQueueStats stats = pending_snapshot_controls_.stats();
    RunLogger::logGlobal(
        "snapshot_control",
        "abandoned controls during bounded shutdown count=" +
            std::to_string(abandoned) + " total_abandoned=" +
            std::to_string(stats.abandoned_on_stop));
  }
  drain_cv_.notify_all();
}

bool InstanceMapThread::applyPendingDurabilityAck() {
  const SceneRevision revision =
      pending_durable_ack_.exchange(0, std::memory_order_acq_rel);
  if (revision == 0) {
    return false;
  }
  processing_work_.store(true);
  try {
    const SceneApplyResult result = applyQueuedSceneCommand(
        SceneCommand{PersistedThroughCommand{revision}});
    if (!result.accepted()) {
      RunLogger::logGlobal(
          "scene_persistence",
          "durability watermark rejected revision=" +
              std::to_string(revision) + " reason=" + result.reason);
    }
  } catch (const std::exception& error) {
    RunLogger::logGlobal(
        "scene_persistence",
        "durability watermark actor exception revision=" +
            std::to_string(revision) + " error=" + error.what());
  } catch (...) {
    RunLogger::logGlobal(
        "scene_persistence",
        "durability watermark actor non-standard exception revision=" +
            std::to_string(revision));
  }
  processing_work_.store(false);
  return true;
}

void InstanceMapThread::onStopRequested() {
  scene_command_queue_.stop();
}

SceneApplyResult InstanceMapThread::applyQueuedSceneCommand(
    SceneCommand command) {
  const auto* geometry_command =
      std::get_if<ApplyGeometryResultCommand>(&command);
  const std::optional<ApplyGeometryResultCommand> geometry_update =
      geometry_command
          ? std::optional<ApplyGeometryResultCommand>(*geometry_command)
          : std::nullopt;
  const bool persist_content_commit =
      !std::holds_alternative<PersistedThroughCommand>(command) &&
      !std::holds_alternative<AdvanceSurfaceCommand>(command);
  if (persist_content_commit && !sceneContentAdmissionAllowed()) {
    const std::string reason = sceneContentAdmissionRejectionReason();
    RunLogger::logGlobal("scene_persistence",
                         "command_rejected_pre_reducer reason=" + reason);
    return rejectForPersistenceAdmission(reason);
  }
  const SceneApplyResult result = reducer_.apply(command);
  if (!result.accepted()) {
    RunLogger::logGlobal("scene_reducer",
                         "queued command rejected reason=" + result.reason);
    return result;
  }

  // tracks_ is now only the association working set. Keep the geometry fields
  // required by the unchanged association formulas in sync with the reducer's
  // accepted CAS result; it is never published as current scene state.
  if (geometry_update) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (InstanceTrack& track : tracks_) {
      if (track.object_id != geometry_update->dependency.object.object_id ||
          track.obb_revision !=
              geometry_update->dependency.object.obb_revision) {
        continue;
      }
      const GeometryEvaluationResult& evaluation = geometry_update->result;
      track.geometry_status = evaluation.status;
      track.geometry_score = evaluation.score;
      track.geometry_shell_ratio = evaluation.shell_ratio;
      track.geometry_extent_score = evaluation.extent_score;
      track.geometry_leak_ratio = evaluation.leak_ratio;
      track.geometry_cavity_ratio = evaluation.cavity_ratio;
      track.geometry_in_box_points = evaluation.in_box_points;
      track.geometry_shell_points = evaluation.shell_points;
      track.geometry_unique_voxels = evaluation.unique_voxels;
      track.geometry_expanded_points = evaluation.expanded_points;
      track.geometry_bad_count = evaluation.bad_count;
      track.last_geometry_check_ns = evaluation.checked_at_ns;
      track.geometry_evaluation_obb_revision = track.obb_revision;
      track.geometry_evaluation_map_version =
          geometry_update->dependency.surface.source_map_revision;
      track.geometry_evaluated_center_world =
          evaluation.evaluated_center_world;
      track.geometry_evaluated_size_m = evaluation.evaluated_size_m;
      track.geometry_evaluated_yaw_rad = evaluation.evaluated_yaw_rad;
      track.geometry_evaluation_reason = evaluation.reason;
      updateObjectQualityScore(&track);
      object_graph_.updateNodeFromTrack(track);
      break;
    }
  }
  publishReducerResult(result, persist_content_commit);
  return result;
}

SceneApplyResult InstanceMapThread::rejectForPersistenceAdmission(
    const std::string& reason) const {
  SceneApplyResult result;
  result.status = SceneApplyStatus::kRejected;
  result.snapshot = reducer_.snapshot();
  result.revision = result.snapshot.revision();
  result.reason = reason;
  return result;
}

bool InstanceMapThread::sceneContentAdmissionAllowed() const {
  if (persistence_commit_failed_ || snapshotControlFaulted()) {
    return false;
  }
  if (!scene_content_admission_) {
    return true;
  }
  try {
    return scene_content_admission_();
  } catch (const std::exception& error) {
    RunLogger::logGlobal(
        "scene_persistence",
        "content admission callback failed: " + std::string(error.what()));
  } catch (...) {
    RunLogger::logGlobal(
        "scene_persistence",
        "content admission callback failed with non-standard exception");
  }
  return false;
}

std::string InstanceMapThread::sceneContentAdmissionRejectionReason() const {
  if (snapshotControlFaulted()) {
    return "terminal snapshot control admission fault is active";
  }
  if (persistence_commit_failed_) {
    return "scene persistence commit fuse is active";
  }
  return "scene persistence admission is closed";
}

void InstanceMapThread::publishReducerResult(
    const SceneApplyResult& result,
    bool persist_content_commit) {
  std::atomic_store(&published_scene_state_, result.snapshot.statePtr());
  if (scene_commit_observer_ && !result.events.empty()) {
    try {
      scene_commit_observer_(result);
    } catch (const std::exception& error) {
      RunLogger::logGlobal(
          "scene_reducer",
          "scene commit observer failed: " + std::string(error.what()));
    } catch (...) {
      RunLogger::logGlobal(
          "scene_reducer",
          "scene commit observer failed with non-standard exception");
    }
  }
  if (persist_content_commit && result.committedRevision() &&
      scene_commit_sink_) {
    bool persisted = false;
    std::string sink_error;
    try {
      persisted = scene_commit_sink_(result);
    } catch (const std::exception& error) {
      sink_error = error.what();
    } catch (...) {
      sink_error = "non-standard exception";
    }
    if (!persisted) {
      persistence_commit_failed_ = true;
      RunLogger::logGlobal(
          "scene_persistence",
          "failed to enqueue scene_revision=" +
              std::to_string(result.revision) +
              "; terminal content-commit fuse activated" +
              (sink_error.empty() ? std::string{}
                                  : "; error=" + sink_error));
    }
  }
  if (online_snapshot_worker_ != nullptr && result.committedRevision()) {
    for (const SceneEvent& event : result.events) {
      if (const auto* merged = std::get_if<ObjectMerged>(&event)) {
        enqueueSnapshotControl(SnapshotControl{
            SnapshotControlKind::kMerge,
            merged->retired_object_id,
            merged->canonical_object_id,
            0,
            merged->revision});
      } else if (const auto* tombstoned =
                     std::get_if<ObjectTombstoned>(&event)) {
        enqueueSnapshotControl(SnapshotControl{
            SnapshotControlKind::kEraseObject,
            tombstoned->object_id,
            -1,
            0,
            tombstoned->revision});
      }
    }
  }
  const ChannelStats scene_command_stats = scene_command_queue_.stats();
  const ChannelStats reducer_response_stats = response_queue_.stats();
  RunLogger::logGlobal(
      "scene_reducer",
      "published scene_revision=" + std::to_string(result.revision) +
          " durable_scene_revision=" +
          std::to_string(result.snapshot.durableRevision()) +
          " status=" + std::to_string(static_cast<int>(result.status)) +
          " scene_command_queue_depth=" +
          std::to_string(scene_command_stats.depth) +
          " scene_command_queue_high_watermark=" +
          std::to_string(scene_command_stats.high_watermark) +
          " reducer_response_queue_depth=" +
          std::to_string(reducer_response_stats.depth) +
          " reducer_response_queue_high_watermark=" +
          std::to_string(reducer_response_stats.high_watermark));
}

void InstanceMapThread::applyDetections(const InferenceResponse& response) {
  if (config_.freeze_instances) {
    applyFrozenInstanceSnapshotRemake(response);
    return;
  }

  const SceneSnapshot association_base = sceneSnapshot();
  if (response.provenance.run_id.valid() &&
      response.provenance.frame_id != 0) {
    const FrameKey key{response.provenance.run_id,
                       response.provenance.frame_id};
    if (std::find(
            association_base.statePtr()->recent_observation_frames.begin(),
            association_base.statePtr()->recent_observation_frames.end(),
            key) !=
        association_base.statePtr()->recent_observation_frames.end()) {
      RunLogger::logGlobal(
          "instance_map",
          "ignored duplicate observation frame_id=" +
              std::to_string(response.provenance.frame_id));
      return;
    }
    const auto watermark = std::find_if(
        association_base.statePtr()->observation_watermarks.begin(),
        association_base.statePtr()->observation_watermarks.end(),
        [&key](const ObservationRunWatermark& candidate) {
          return candidate.run_id == key.run_id;
        });
    constexpr FrameId kObservationReplayWindow = 256;
    if (watermark !=
            association_base.statePtr()->observation_watermarks.end() &&
        key.frame_id < watermark->highest_frame_id &&
        watermark->highest_frame_id - key.frame_id >=
            kObservationReplayWindow) {
      RunLogger::logGlobal(
          "instance_map",
          "ignored out-of-window observation frame_id=" +
              std::to_string(response.provenance.frame_id));
      return;
    }
  }
  refreshAssociationWorkingSet(association_base);

  const auto apply_start = std::chrono::steady_clock::now();
  std::vector<InstanceObservation, Eigen::aligned_allocator<InstanceObservation>> observations;
  observations.reserve(response.detections.size());
  std::map<std::string, std::size_t> raw_by_label;
  std::map<std::string, std::size_t> observed_by_label;
  std::map<std::string, std::size_t> make_rejected_by_label;
  std::map<std::string, std::size_t> accepted_by_label;
  std::map<std::string, std::size_t> duplicate_rejected_by_label;
  std::map<std::string, std::size_t> created_by_label;
  std::map<std::string, std::size_t> updated_by_label;
  std::map<std::string, std::size_t> promoted_by_label;
  std::map<std::string, std::size_t> promotion_quality_blocked_by_label;
  std::map<std::string, std::size_t> geometry_update_suppressed_by_label;
  std::map<std::string, std::size_t> geometry_update_suppressed_by_reason;
  std::map<std::string, std::size_t> merged_small_duplicates_by_label;
  std::size_t rejected = 0;
  for (const RawDetection& detection : response.detections) {
    const std::string label = diagnosticsLabel(detection.label);
    ++raw_by_label[label];
    std::optional<InstanceObservation> observation =
        makeObservation(response, detection);
    if (observation) {
      ++observed_by_label[label];
      observations.push_back(std::move(*observation));
    } else {
      ++make_rejected_by_label[label];
      ++rejected;
    }
  }

  std::shared_ptr<const ImageBuffer> snapshot_frame;
  std::vector<std::optional<SnapshotCandidate>> snapshot_candidates;
  if (online_snapshot_worker_ != nullptr && !response.source_rgb_960.empty()) {
    snapshot_frame =
        std::make_shared<const ImageBuffer>(response.source_rgb_960);
    snapshot_candidates.reserve(observations.size());
    for (std::size_t observation_index = 0;
         observation_index < observations.size(); ++observation_index) {
      const InstanceObservation& observation = observations[observation_index];
      snapshot_candidates.emplace_back(makeOnlineSnapshotCandidate(
          response, observation, snapshot_frame));
    }
  }
  struct SnapshotDispatch {
    OnlineSnapshotCandidate input;
    std::optional<std::pair<int, SceneObjectId>> promotion;
  };
  std::vector<SnapshotDispatch> snapshot_dispatches;
  std::vector<int> expired_snapshot_tracks;

  std::size_t created = 0;
  std::size_t updated = 0;
  std::size_t objects_before = 0;
  std::size_t objects_after = 0;
  std::size_t tracks_before = 0;
  std::size_t tracks_after = 0;
  std::size_t removed_tentative = 0;
  std::size_t merged_duplicates = 0;
  std::size_t duplicate_rejected = 0;
  std::size_t promotion_quality_blocked = 0;
  std::size_t geometry_update_suppressed = 0;
  std::size_t published_objects_after = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++frame_index_;
    objects_before = object_graph_.objectCount();
    tracks_before = tracks_.size();

    std::vector<bool> track_matched(tracks_.size(), false);
    for (std::size_t observation_index = 0;
         observation_index < observations.size(); ++observation_index) {
      const InstanceObservation& observation = observations[observation_index];
      const std::string observation_label = diagnosticsLabel(observation.detection.label);
      if (shouldRejectAsDuplicateOfConfirmed(observation)) {
        ++duplicate_rejected;
        ++duplicate_rejected_by_label[observation_label];
        continue;
      }
      ++accepted_by_label[observation_label];
      const std::optional<std::size_t> track_index =
          findBestTrack(observation, track_matched);
      if (track_index) {
        const bool was_tentative = tracks_[*track_index].object_id < 0;
        bool suppressed_geometry_update = false;
        std::string geometry_suppression_reason;
        const bool promoted =
            updateTrack(&tracks_[*track_index],
                        observation,
                        &suppressed_geometry_update,
                        &geometry_suppression_reason);
        ++updated_by_label[diagnosticsLabel(tracks_[*track_index].label)];
        if (suppressed_geometry_update) {
          const std::string updated_label =
              diagnosticsLabel(tracks_[*track_index].label);
          ++geometry_update_suppressed;
          ++geometry_update_suppressed_by_label[updated_label];
          if (!geometry_suppression_reason.empty()) {
            ++geometry_update_suppressed_by_reason[
                updated_label + ":" + geometry_suppression_reason];
          }
        }
        if (promoted) {
          ++promoted_by_label[diagnosticsLabel(tracks_[*track_index].label)];
        }
        if (observation_index < snapshot_candidates.size() &&
            snapshot_candidates[observation_index]) {
          const InstanceTrack& snapshot_track = tracks_[*track_index];
          SnapshotDispatch dispatch;
          dispatch.input.candidate =
              std::move(*snapshot_candidates[observation_index]);
          if (promoted && was_tentative) {
            dispatch.input.owner =
                SnapshotOwner{SnapshotOwnerKind::kTentativeTrack,
                              snapshot_track.track_id};
            dispatch.promotion =
                std::make_pair(snapshot_track.track_id,
                               snapshot_track.object_id);
          } else if (snapshot_track.object_id >= 0) {
            dispatch.input.owner =
                SnapshotOwner{SnapshotOwnerKind::kObject,
                              snapshot_track.object_id};
          } else {
            dispatch.input.owner =
                SnapshotOwner{SnapshotOwnerKind::kTentativeTrack,
                              snapshot_track.track_id};
          }
          snapshot_dispatches.push_back(std::move(dispatch));
        }
        track_matched[*track_index] = true;
        ++updated;
      } else {
        const bool promoted = createTrack(observation);
        ++created_by_label[observation_label];
        if (promoted) {
          ++promoted_by_label[observation_label];
        }
        if (observation_index < snapshot_candidates.size() &&
            snapshot_candidates[observation_index]) {
          const InstanceTrack& snapshot_track = tracks_.back();
          SnapshotDispatch dispatch;
          dispatch.input.candidate =
              std::move(*snapshot_candidates[observation_index]);
          dispatch.input.owner =
              SnapshotOwner{SnapshotOwnerKind::kTentativeTrack,
                            snapshot_track.track_id};
          if (promoted) {
            dispatch.promotion =
                std::make_pair(snapshot_track.track_id,
                               snapshot_track.object_id);
          }
          snapshot_dispatches.push_back(std::move(dispatch));
        }
        track_matched.push_back(true);
        ++created;
      }
    }

    ageUnmatchedTracks(track_matched, response.time_ns);
    removed_tentative =
        removeExpiredTentativeTracks(&expired_snapshot_tracks);
    merged_duplicates += mergeDuplicateStableTracks(&merged_small_duplicates_by_label);
    for (InstanceTrack& track : tracks_) {
      if (track.object_id >= 0) {
        object_graph_.updateNodeFromTrack(track);
      }
    }
    // Surface scanning is owned by GeometryWorker. The association actor only
    // emits OBB/object events through the reducer and never evaluates geometry
    // synchronously.
    tracks_after = tracks_.size();
    objects_after = object_graph_.objectCount();
    promotion_quality_blocked =
        countPromotionQualityBlocked(tracks_, config_, &promotion_quality_blocked_by_label);
    published_objects_after =
        object_graph_.snapshotInstanceRecords(/*publishable_only=*/false).size();
  }

  if (!response.detections.empty() || !observations.empty()) {
    const double apply_ms = elapsedMs(apply_start, std::chrono::steady_clock::now());
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2)
           << "applied camera=" << response.camera_id
           << " t=" << response.time_ns
           << " raw=" << response.detections.size()
           << " accepted=" << (observations.size() - duplicate_rejected)
           << " rejected=" << (rejected + duplicate_rejected)
           << " duplicate_rejected=" << duplicate_rejected
           << " geometry_update_suppressed=" << geometry_update_suppressed
           << " promotion_quality_blocked=" << promotion_quality_blocked
           << " created_tracks=" << created
           << " updated_tracks=" << updated
           << " removed_tentative=" << removed_tentative
           << " merged_duplicates=" << merged_duplicates
           << " tracks=" << tracks_before << "->" << tracks_after
           << " objects=" << objects_before << "->" << objects_after
           << " published_objects=" << published_objects_after
           << " apply_ms=" << apply_ms;
    RunLogger::logGlobal("instance_map", stream.str());

    std::ostringstream detail;
    detail << "labels camera=" << response.camera_id
           << " t=" << response.time_ns
           << " raw=" << formatLabelCounts(raw_by_label)
           << " observed=" << formatLabelCounts(observed_by_label)
           << " make_rejected=" << formatLabelCounts(make_rejected_by_label)
           << " accepted=" << formatLabelCounts(accepted_by_label)
           << " duplicate_rejected=" << formatLabelCounts(duplicate_rejected_by_label)
           << " created=" << formatLabelCounts(created_by_label)
           << " updated=" << formatLabelCounts(updated_by_label)
           << " promoted=" << formatLabelCounts(promoted_by_label)
           << " promotion_quality_blocked="
           << formatLabelCounts(promotion_quality_blocked_by_label)
           << " geometry_update_suppressed="
           << formatLabelCounts(geometry_update_suppressed_by_label)
           << " geometry_update_suppressed_reason="
           << formatLabelCounts(geometry_update_suppressed_by_reason)
           << " merged_small_duplicates="
           << formatLabelCounts(merged_small_duplicates_by_label);
    RunLogger::logGlobal("instance_map_detail", detail.str());
  }
  if (!commitReducerState(response, observations) ||
      online_snapshot_worker_ == nullptr) {
    return;
  }
  const SceneRevision control_revision = sceneSnapshot().revision();
  for (SnapshotDispatch& dispatch : snapshot_dispatches) {
    (void)online_snapshot_worker_->enqueueCandidate(
        std::move(dispatch.input), response.time_ns);
    if (dispatch.promotion) {
      enqueueSnapshotControl(SnapshotControl{
          SnapshotControlKind::kPromote,
          dispatch.promotion->first,
          dispatch.promotion->second,
          response.time_ns,
          control_revision});
    }
  }
  for (int track_id : expired_snapshot_tracks) {
    enqueueSnapshotControl(SnapshotControl{
        SnapshotControlKind::kDropTentative,
        track_id,
        -1,
        response.time_ns,
        control_revision});
  }
}

bool InstanceMapThread::commitReducerState(
    const InferenceResponse& response,
    const std::vector<InstanceObservation,
                      Eigen::aligned_allocator<InstanceObservation>>& observations) {
  std::vector<InstanceTrack, Eigen::aligned_allocator<InstanceTrack>> current_tracks;
  std::vector<std::pair<int, int>> merges;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_tracks.reserve(tracks_.size());
    for (const InstanceTrack& track : tracks_) {
      current_tracks.push_back(track);
    }
    merges = std::move(pending_reducer_merges_);
    pending_reducer_merges_.clear();
  }

  const SceneSnapshot before = reducer_.snapshot();
  std::map<int, int> current_object_by_track;
  std::set<int> current_object_ids;
  std::set<int> current_track_ids;
  ApplyObservationBatchCommand command;
  command.provenance = response.provenance;
  command.observations = observations;
  command.association_source = "scene_snapshot_association_v1";
  command.associated_mutations.reserve(current_tracks.size() +
                                       before.objects().size());
  for (const InstanceTrack& track : current_tracks) {
    if (track.track_id >= 0) {
      current_track_ids.insert(track.track_id);
    }
    if (track.object_id >= 0) {
      current_object_ids.insert(track.object_id);
      current_object_by_track[track.track_id] = track.object_id;
      UpsertTrackMutation upsert;
      upsert.track = track;
      upsert.object_id = track.object_id;
      command.associated_mutations.emplace_back(std::move(upsert));
    } else if (track.track_id >= 0) {
      command.associated_mutations.emplace_back(
          UpsertTentativeTrackMutation{track});
    }
  }
  for (const auto& [track_id, track] : before.tracks()) {
    (void)track;
    if (current_track_ids.count(track_id) == 0) {
      command.associated_mutations.emplace_back(
          RemoveTrackMutation{track_id});
    }
  }

  std::set<int> explicitly_merged_ids;
  for (const auto& [retired_id, canonical_id] : merges) {
    if (retired_id < 0 || canonical_id < 0 || retired_id == canonical_id) {
      continue;
    }
    command.associated_mutations.emplace_back(durableMergeMutation(
        before, retired_id, canonical_id, config_.snapshot_top_k));
    explicitly_merged_ids.insert(retired_id);
  }

  for (const auto& [old_object_id, old_object] : before.objects()) {
    if (current_object_ids.count(old_object_id) != 0 || !old_object ||
        !old_object->identity ||
        explicitly_merged_ids.count(old_object_id) != 0) {
      continue;
    }
    std::optional<int> merged_into;
    for (int source_track_id : old_object->identity->source_track_ids) {
      const auto target = current_object_by_track.find(source_track_id);
      if (target != current_object_by_track.end() &&
          target->second != old_object_id) {
        merged_into = target->second;
        break;
      }
    }
    if (merged_into) {
      command.associated_mutations.emplace_back(durableMergeMutation(
          before, old_object_id, *merged_into, config_.snapshot_top_k));
    } else {
      command.associated_mutations.emplace_back(
          TombstoneObjectMutation{old_object_id,
                                  "retired_by_association"});
    }
  }

  const SceneApplyResult result =
      reducer_.apply(SceneCommand{std::move(command)});
  if (!result.accepted()) {
    refreshAssociationWorkingSet(reducer_.snapshot());
    RunLogger::logGlobal(
        "scene_reducer",
        "command_rejected frame_id=" +
            std::to_string(response.provenance.frame_id) +
            " reason=" + result.reason);
    return false;
  }
  publishReducerResult(result, /*persist_content_commit=*/true);
  RunLogger::logGlobal(
      "scene_reducer",
      "observation_commit frame_id=" +
          std::to_string(response.provenance.frame_id) +
          " objects=" + std::to_string(result.snapshot.objects().size()));
  return true;
}

void InstanceMapThread::applyFrozenInstanceSnapshotRemake(
    const InferenceResponse& response) {
  if (!config_.load_scene_graph ||
      (!config_.snapshot_remake_enabled &&
       online_snapshot_worker_ == nullptr)) {
    return;
  }

  const auto apply_start = std::chrono::steady_clock::now();
  std::vector<InstanceObservation, Eigen::aligned_allocator<InstanceObservation>> observations;
  observations.reserve(response.detections.size());
  std::map<std::string, std::size_t> raw_by_label;
  std::map<std::string, std::size_t> observed_by_label;
  std::map<std::string, std::size_t> make_rejected_by_label;
  std::map<std::string, std::size_t> matched_by_label;
  std::size_t rejected = 0;

  for (const RawDetection& detection : response.detections) {
    const std::string label = diagnosticsLabel(detection.label);
    ++raw_by_label[label];
    std::optional<InstanceObservation> observation =
        makeObservation(response, detection);
    if (observation) {
      ++observed_by_label[label];
      observations.push_back(std::move(*observation));
    } else {
      ++make_rejected_by_label[label];
      ++rejected;
    }
  }

  std::vector<ObjectSnapshotRemakeCandidate> candidates;
  std::vector<OnlineSnapshotCandidate> online_candidates;
  std::shared_ptr<const ImageBuffer> online_frame;
  if (online_snapshot_worker_ != nullptr &&
      !response.source_rgb_960.empty()) {
    online_frame =
        std::make_shared<const ImageBuffer>(response.source_rgb_960);
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    candidates.reserve(observations.size());
    online_candidates.reserve(observations.size());
    for (const InstanceObservation& observation : observations) {
      const std::optional<std::size_t> track_index =
          findBestFrozenTrack(observation);
      if (!track_index) {
        continue;
      }
      const InstanceTrack& track = tracks_[*track_index];
      ObjectSnapshotRemakeCandidate candidate;
      candidate.object_id = track.object_id;
      candidate.bbox_xyxy = observation.detection.box_xyxy;
      candidate.bbox_quality = observation.bbox_quality;
      candidate.time_ns = observation.time_ns;
      candidate.camera_id = observation.camera_id;
      candidates.push_back(std::move(candidate));
      if (online_frame) {
        OnlineSnapshotCandidate online;
        online.owner =
            SnapshotOwner{SnapshotOwnerKind::kObject, track.object_id};
        online.candidate = makeOnlineSnapshotCandidate(
            response, observation, online_frame);
        online_candidates.push_back(std::move(online));
      }
      ++matched_by_label[diagnosticsLabel(track.label)];
    }
  }

  std::string error;
  ObjectSnapshotRemakeFrameResult result;
  std::size_t online_accepted = 0;
  if (online_snapshot_worker_ != nullptr) {
    result.candidate_count = online_candidates.size();
    for (OnlineSnapshotCandidate& candidate : online_candidates) {
      const SnapshotWorkerEnqueueResult enqueued =
          online_snapshot_worker_->enqueueCandidate(
              std::move(candidate), response.time_ns);
      if (enqueued.accepted()) {
        ++online_accepted;
      } else if (error.empty()) {
        error = enqueued.reason;
      }
    }
  } else {
    // Deprecated compatibility path for installations that explicitly
    // disable the online SnapshotBank. It only affects a later JSON export;
    // the normal pipeline always routes frozen evidence through reducer-owned
    // ApplySnapshotSetCommand above.
    result = snapshot_remaker_.submitFrame(response.source_rgb_960,
                                           response.time_ns,
                                           response.camera_id,
                                           candidates,
                                           &error);
  }
  const double apply_ms = elapsedMs(apply_start, std::chrono::steady_clock::now());
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(2)
         << "remake camera=" << response.camera_id
         << " t=" << response.time_ns
         << " raw=" << response.detections.size()
         << " observed=" << observations.size()
         << " rejected=" << rejected
         << " matched=" << candidates.size()
         << " candidate_count=" << result.candidate_count
         << " online_snapshot_bank="
         << (online_snapshot_worker_ != nullptr ? "true" : "false")
         << " online_accepted=" << online_accepted
         << " saved_frame=" << (result.saved_frame ? "true" : "false")
         << " improved_objects=" << result.improved_objects
         << " first_snapshot_objects=" << result.first_snapshot_objects
         << " replaced_objects=" << result.replaced_objects
         << " quality_rejected=" << result.quality_rejected_candidates
         << " replace_rejected=" << result.replace_rejected_candidates
         << " apply_ms=" << apply_ms;
  if (!error.empty()) {
    stream << " error=" << error;
  }
  RunLogger::logGlobal("instance_snapshot_remake", stream.str());

  std::ostringstream detail;
  detail << "labels camera=" << response.camera_id
         << " t=" << response.time_ns
         << " raw=" << formatLabelCounts(raw_by_label)
         << " observed=" << formatLabelCounts(observed_by_label)
         << " make_rejected=" << formatLabelCounts(make_rejected_by_label)
         << " matched=" << formatLabelCounts(matched_by_label);
  RunLogger::logGlobal("instance_snapshot_remake_detail", detail.str());
}

std::optional<InstanceObservation> InstanceMapThread::makeObservation(
    const InferenceResponse& response,
    const RawDetection& detection) const {
  InstanceObservation observation;
  observation.time_ns = response.time_ns;
  observation.camera_id = response.camera_id;
  observation.detection = detection;
  observation.confidence = rawDetectionConfidence(detection);

  if (shouldIgnoreDetectionLabel(detection.label)) {
    return std::nullopt;
  }
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
  return observation;
}

std::optional<std::size_t> InstanceMapThread::findBestFrozenTrack(
    const InstanceObservation& observation) const {
  float best_score = -1.0f;
  std::optional<std::size_t> best_index;
  for (std::size_t i = 0; i < tracks_.size(); ++i) {
    const InstanceTrack& track = tracks_[i];
    if (track.object_id < 0 || !track.publishable) {
      continue;
    }
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
    if (!hasMatureDuplicateAuthority(track, config_)) {
      continue;
    }
    return true;
  }
  return false;
}

bool InstanceMapThread::createTrack(const InstanceObservation& observation) {
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
  const bool promoted = maybePromoteOrUpdateObject(&tracks_.back());
  return promoted;
}

bool InstanceMapThread::updateTrack(InstanceTrack* track,
                                    const InstanceObservation& observation,
                                    bool* geometry_update_suppressed,
                                    std::string* geometry_update_suppression_reason) {
  if (geometry_update_suppressed != nullptr) {
    *geometry_update_suppressed = false;
  }
  if (geometry_update_suppression_reason != nullptr) {
    geometry_update_suppression_reason->clear();
  }
  const float promotion_weight = observationPromotionWeight(config_, observation);
  const float old_mass =
      std::max(kEpsilon,
               std::min(track->bbox_quality_mass, config_.instance_fusion_prior_mass_cap));
  const float new_weight =
      std::max(kEpsilon, observation.bbox_quality * promotion_weight);
  const std::string suppression_reason =
      confirmedGeometryUpdateSuppressionReason(*track,
                                               observation,
                                               config_,
                                               old_mass,
                                               new_weight);
  const bool suppress_geometry_update = !suppression_reason.empty();

  if (!suppress_geometry_update) {
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
  } else if (geometry_update_suppressed != nullptr) {
    *geometry_update_suppressed = true;
    if (geometry_update_suppression_reason != nullptr) {
      *geometry_update_suppression_reason = suppression_reason;
    }
  }
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
  retainRecentObservationTimestamps(
      &track->observation_timestamps_ns,
      config_.instance_observation_history_capacity);
  if (!observation.near_surface_voxels.empty()) {
    track->near_surface_voxels = observation.near_surface_voxels;
  }
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
  const bool promoted = maybePromoteOrUpdateObject(track);
  return promoted;
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

std::size_t InstanceMapThread::removeExpiredTentativeTracks(
    std::vector<int>* removed_track_ids) {
  const std::size_t before = tracks_.size();
  tracks_.erase(std::remove_if(tracks_.begin(),
                               tracks_.end(),
                               [this, removed_track_ids](const InstanceTrack& track) {
                                 const bool expired =
                                     track.object_id < 0 &&
                                     track.state == InstanceTrackState::kTentative &&
                                     track.missed_count >=
                                         config_.instance_tentative_max_missed;
                                 if (expired && removed_track_ids != nullptr) {
                                   removed_track_ids->push_back(track.track_id);
                                 }
                                 return expired;
                               }),
                tracks_.end());
  return before - tracks_.size();
}

std::size_t InstanceMapThread::mergeDuplicateStableTracks(
    std::map<std::string, std::size_t>* small_duplicates_by_label) {
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
        const bool small_duplicate =
            smallStableTracksAreDuplicates(lhs, rhs, config_);
        if (!small_duplicate && !stableTracksAreDuplicates(lhs, rhs, config_)) {
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
        retainRecentObservationHistory(
            &winner, config_.instance_observation_history_capacity);
        if (loser.snapshot.valid() &&
            (!winner.snapshot.valid() || loser.snapshot.quality >= winner.snapshot.quality)) {
          winner.snapshot = loser.snapshot;
        }
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
        if (small_duplicate && small_duplicates_by_label != nullptr) {
          ++(*small_duplicates_by_label)[diagnosticsLabel(winner.label)];
        }

        object_graph_.removeNode(loser_object_id);
        pending_reducer_merges_.emplace_back(loser_object_id,
                                             winner.object_id);
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

bool InstanceMapThread::maybePromoteOrUpdateObject(InstanceTrack* track) {
  bool promoted = false;
  if (track->object_id < 0 && isPromotable(*track)) {
    track->state = InstanceTrackState::kStable;
    track->object_id = object_graph_.createNodeFromTrack(*track);
    promoted = true;
  }
  if (track->object_id >= 0) {
    object_graph_.updateNodeFromTrack(*track);
  }
  return promoted;
}

bool InstanceMapThread::isPromotable(const InstanceTrack& track) const {
  return hasBasePromotionEvidence(track, config_) &&
         hasPromotionQuality(track, config_);
}

}  // namespace roomie
