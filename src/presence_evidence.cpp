#include "roomie/pipeline/presence_evidence.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace roomie {
namespace {

constexpr float kEpsilon = 1.0e-6f;

void retainWindow(std::vector<TimeNanoseconds>* values,
                  TimeNanoseconds now_ns,
                  TimeNanoseconds window_ns) {
  values->erase(std::remove_if(values->begin(), values->end(),
                               [now_ns, window_ns](TimeNanoseconds value) {
                                 return value > now_ns ||
                                        now_ns - value > window_ns;
                               }),
                values->end());
}

bool masked(const VisibilityContext& context, int x, int y) {
  if (!context.robot_mask || context.robot_mask->empty() ||
      context.robot_mask->width != context.depth->width ||
      context.robot_mask->height != context.depth->height) {
    return false;
  }
  const ImageBuffer& mask = *context.robot_mask;
  const std::size_t offset =
      (static_cast<std::size_t>(y) * mask.width + x) * mask.channels;
  return offset < mask.data.size() && mask.data[offset] != 0;
}

bool intersectRayObb(const InstanceTrack& track,
                     const Eigen::Vector3f& camera_origin_world,
                     const Eigen::Vector3f& ray_world,
                     float* entry,
                     float* exit) {
  const float c = std::cos(-track.yaw_rad);
  const float s = std::sin(-track.yaw_rad);
  const Eigen::Matrix3f world_to_box =
      (Eigen::Matrix3f() << c, -s, 0.0f, s, c, 0.0f, 0.0f, 0.0f, 1.0f)
          .finished();
  const Eigen::Vector3f origin =
      world_to_box * (camera_origin_world - track.center_world);
  const Eigen::Vector3f direction = world_to_box * ray_world;
  const Eigen::Vector3f half = 0.5f * track.size_m;
  float t_min = 0.0f;
  float t_max = std::numeric_limits<float>::infinity();
  for (int axis = 0; axis < 3; ++axis) {
    if (std::abs(direction[axis]) < kEpsilon) {
      if (origin[axis] < -half[axis] || origin[axis] > half[axis]) {
        return false;
      }
      continue;
    }
    float a = (-half[axis] - origin[axis]) / direction[axis];
    float b = (half[axis] - origin[axis]) / direction[axis];
    if (a > b) {
      std::swap(a, b);
    }
    t_min = std::max(t_min, a);
    t_max = std::min(t_max, b);
    if (t_max < t_min) {
      return false;
    }
  }
  if (t_max <= 0.0f) {
    return false;
  }
  *entry = t_min;
  *exit = t_max;
  return true;
}

}  // namespace

VisibilityEvidence evaluateVisibility(const InstanceTrack& track,
                                      const InferenceResponse& response,
                                      const PresenceEvidenceConfig& config) {
  VisibilityEvidence evidence;
  if (!response.ok || !response.has_camera_pose ||
      !response.visibility_context || !response.visibility_context->valid() ||
      (track.size_m.array() <= 0.0f).any()) {
    evidence.reason = "missing_visibility_context";
    return evidence;
  }
  const VisibilityContext& context = *response.visibility_context;
  const CameraIntrinsics& intrinsics = context.intrinsics;
  const Eigen::Isometry3f camera_from_world = response.T_world_camera.inverse();
  const Eigen::Vector3f half = 0.5f * track.size_m;
  const float c = std::cos(track.yaw_rad);
  const float s = std::sin(track.yaw_rad);
  const Eigen::Matrix3f box_to_world =
      (Eigen::Matrix3f() << c, -s, 0.0f, s, c, 0.0f, 0.0f, 0.0f, 1.0f)
          .finished();
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  for (int ix : {-1, 1}) {
    for (int iy : {-1, 1}) {
      for (int iz : {-1, 1}) {
        const Eigen::Vector3f world =
            track.center_world +
            box_to_world * Eigen::Vector3f(ix * half.x(), iy * half.y(),
                                           iz * half.z());
        const Eigen::Vector3f camera = camera_from_world * world;
        if (camera.z() <= kEpsilon) {
          evidence.reason = "behind_camera";
          return evidence;
        }
        const float x = intrinsics.fx * camera.x() / camera.z() + intrinsics.cx;
        const float y = intrinsics.fy * camera.y() / camera.z() + intrinsics.cy;
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
      }
    }
  }
  if (max_x < 0.0f || max_y < 0.0f || min_x >= intrinsics.width ||
      min_y >= intrinsics.height) {
    evidence.reason = "out_of_frustum";
    return evidence;
  }
  if (min_x < config.image_edge_margin_px ||
      min_y < config.image_edge_margin_px ||
      max_x >= intrinsics.width - config.image_edge_margin_px ||
      max_y >= intrinsics.height - config.image_edge_margin_px) {
    evidence.reason = "image_edge";
    return evidence;
  }
  const int x0 = std::clamp(static_cast<int>(std::floor(min_x)), 0,
                            intrinsics.width - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(min_y)), 0,
                            intrinsics.height - 1);
  const int x1 = std::clamp(static_cast<int>(std::ceil(max_x)), x0 + 1,
                            intrinsics.width);
  const int y1 = std::clamp(static_cast<int>(std::ceil(max_y)), y0 + 1,
                            intrinsics.height);
  const int step = std::max(
      1, static_cast<int>(std::sqrt(
             static_cast<double>((x1 - x0) * (y1 - y0)) / 1024.0)));
  const Eigen::Vector3f camera_origin_world = response.T_world_camera.translation();
  for (int y = y0; y < y1; y += step) {
    for (int x = x0; x < x1; x += step) {
      if (masked(context, x, y)) {
        continue;
      }
      const Eigen::Vector3f ray_camera(
          (static_cast<float>(x) - intrinsics.cx) / intrinsics.fx,
          (static_cast<float>(y) - intrinsics.cy) / intrinsics.fy, 1.0f);
      const Eigen::Vector3f ray_world =
          response.T_world_camera.linear() * ray_camera;
      float entry = 0.0f;
      float exit = 0.0f;
      if (!intersectRayObb(track, camera_origin_world, ray_world, &entry, &exit)) {
        continue;
      }
      ++evidence.projected_samples;
      const std::size_t offset =
          static_cast<std::size_t>(y) * context.depth->width + x;
      if (offset >= context.depth->depth_m.size()) {
        continue;
      }
      const float depth = context.depth->depth_m[offset];
      if (!std::isfinite(depth) || depth <= 0.0f) {
        continue;
      }
      ++evidence.valid_depth_samples;
      if (depth < entry - config.depth_margin_m) {
        ++evidence.occluded_samples;
      } else if (depth <= exit + config.depth_margin_m) {
        ++evidence.occupied_samples;
      } else {
        ++evidence.free_space_samples;
      }
    }
  }
  if (evidence.projected_samples > 0) {
    evidence.valid_coverage = static_cast<float>(evidence.valid_depth_samples) /
                              evidence.projected_samples;
  }
  if (evidence.valid_depth_samples > 0) {
    evidence.occlusion_ratio = static_cast<float>(evidence.occluded_samples) /
                               evidence.valid_depth_samples;
    evidence.free_space_ratio = static_cast<float>(evidence.free_space_samples) /
                                evidence.valid_depth_samples;
  }
  if (evidence.occupied_samples > 0 &&
      evidence.occupied_samples >= evidence.free_space_samples) {
    evidence.kind = VisibilityEvidenceKind::kOccupied;
    evidence.reliability = std::clamp(
        static_cast<float>(evidence.occupied_samples) /
            std::max(1, evidence.valid_depth_samples),
        0.0f, 1.0f);
    evidence.reason = "occupied_depth";
    return evidence;
  }
  if (evidence.projected_samples < config.min_depth_samples ||
      evidence.valid_coverage < config.min_valid_depth_coverage) {
    evidence.reason = "insufficient_depth";
    return evidence;
  }
  if (evidence.occlusion_ratio > config.max_occlusion_ratio) {
    evidence.reason = "occluded";
    return evidence;
  }
  if (evidence.free_space_ratio >= config.min_free_space_ratio) {
    evidence.kind = VisibilityEvidenceKind::kFreeSpace;
    evidence.reliability = std::clamp(
        evidence.free_space_ratio * evidence.valid_coverage, 0.55f, 0.95f);
    evidence.reason = "registered_depth_free_space";
    return evidence;
  }
  evidence.reason = "ambiguous_depth";
  return evidence;
}

void addPositivePresenceEvidence(InstanceTrack* track,
                                 const InstanceObservation& observation,
                                 const PresenceEvidenceConfig& config) {
  if (track == nullptr) {
    return;
  }
  const float quality = std::clamp(observation.bbox_quality, 0.01f, 0.95f);
  const float probability = 0.5f + 0.5f * quality;
  track->existence_log_odds = std::clamp(
      track->existence_log_odds + std::log(probability / (1.0f - probability)),
      -config.log_odds_cap, config.log_odds_cap);
  retainWindow(&track->positive_evidence_timestamps_ns, observation.time_ns,
               config.evidence_window_ns);
  if (track->positive_evidence_timestamps_ns.empty() ||
      track->positive_evidence_timestamps_ns.back() != observation.time_ns) {
    track->positive_evidence_timestamps_ns.push_back(observation.time_ns);
  }
  retainWindow(&track->negative_evidence_timestamps_ns, observation.time_ns,
               config.evidence_window_ns);
  track->positive_window_interruptions = 0;
  track->last_presence_evidence_ns = observation.time_ns;
  track->last_presence_evidence_reliability = quality;
  track->last_presence_evidence_reason = "matched_detection";
}

void addFreeSpacePresenceEvidence(InstanceTrack* track,
                                  TimeNanoseconds time_ns,
                                  const VisibilityEvidence& evidence,
                                  const PresenceEvidenceConfig& config) {
  if (track == nullptr || evidence.kind != VisibilityEvidenceKind::kFreeSpace) {
    return;
  }
  const float reliability = std::clamp(evidence.reliability, 0.55f, 0.95f);
  track->existence_log_odds = std::clamp(
      track->existence_log_odds +
          std::log((1.0f - reliability) / reliability),
      -config.log_odds_cap, config.log_odds_cap);
  retainWindow(&track->negative_evidence_timestamps_ns, time_ns,
               config.evidence_window_ns);
  if (track->negative_evidence_timestamps_ns.empty() ||
      track->negative_evidence_timestamps_ns.back() != time_ns) {
    track->negative_evidence_timestamps_ns.push_back(time_ns);
  }
  retainWindow(&track->positive_evidence_timestamps_ns, time_ns,
               config.evidence_window_ns);
  ++track->positive_window_interruptions;
  if (track->positive_window_interruptions > config.max_positive_interruptions) {
    track->positive_evidence_timestamps_ns.clear();
  }
  track->last_presence_evidence_ns = time_ns;
  track->last_presence_evidence_reliability = reliability;
  track->last_presence_evidence_reason = evidence.reason;
}

bool hasPositivePresenceConfirmation(const InstanceTrack& track,
                                     TimeNanoseconds now_ns,
                                     const PresenceEvidenceConfig& config) {
  const auto count = std::count_if(
      track.positive_evidence_timestamps_ns.begin(),
      track.positive_evidence_timestamps_ns.end(),
      [now_ns, &config](TimeNanoseconds time_ns) {
        return time_ns <= now_ns && now_ns - time_ns <= config.evidence_window_ns;
      });
  return count >= config.min_positive_frames &&
         track.positive_window_interruptions <= config.max_positive_interruptions &&
         track.existence_log_odds >= config.active_threshold;
}

bool hasNegativePresenceConfirmation(const InstanceTrack& track,
                                     TimeNanoseconds now_ns,
                                     const PresenceEvidenceConfig& config) {
  const auto count = std::count_if(
      track.negative_evidence_timestamps_ns.begin(),
      track.negative_evidence_timestamps_ns.end(),
      [now_ns, &config](TimeNanoseconds time_ns) {
        return time_ns <= now_ns && now_ns - time_ns <= config.evidence_window_ns;
      });
  return count >= config.min_negative_frames &&
         track.existence_log_odds <= config.archive_threshold;
}

float presenceProbability(float log_odds) {
  return 1.0f / (1.0f + std::exp(-log_odds));
}

const char* presenceStateName(InstanceTrackState state) {
  switch (state) {
    case InstanceTrackState::kTentative:
      return "tentative";
    case InstanceTrackState::kStable:
      return "active";
    case InstanceTrackState::kInactive:
      return "archived";
  }
  return "tentative";
}

}  // namespace roomie
