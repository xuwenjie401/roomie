#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/StdVector>

#include "roomie/dsg/object_graph.hpp"

namespace roomie {

struct SmallObjectIdentityConfig {
  float max_volume_m3 = 0.025f;
  float max_extent_m = 0.65f;
  float min_iou_3d = 0.01f;
  float max_normalized_center_distance = 0.35f;
  float min_volume_ratio = 0.50f;
  // Normalized label -> normalized family name. Exact normalized labels are
  // always compatible even when they are not present in this map.
  std::map<std::string, std::string> family_by_label;
};

struct SmallObjectIdentityFeatures {
  bool label_compatible = false;
  bool identity_conflict = false;
  bool eligible = false;
  std::string family;
  float iou_3d = 0.0f;
  float normalized_center_distance = 0.0f;
  float volume_ratio = 0.0f;
};

struct PhysicalObservationConfig {
  float min_2d_iou = 0.65f;
  float min_volume_ratio = 0.50f;
  float max_normalized_center_distance = 0.35f;
  float min_3d_iou = 0.20f;
  float min_containment = 0.60f;
  SmallObjectIdentityConfig small_object_identity;
};

struct AssociationScoringConfig {
  float overlap_weight = 0.30f;
  float center_weight = 0.25f;
  float size_weight = 0.15f;
  float projected_2d_weight = 0.15f;
  float semantic_weight = 0.10f;
  float recency_weight = 0.05f;
  float min_size_ratio = 0.35f;
  float active_match_threshold = 0.55f;
  float archived_match_threshold = 0.75f;
  float merge_support_threshold = 0.75f;
  float max_center_gate_m = 0.75f;
  SmallObjectIdentityConfig small_object_identity;
};

struct AssociationPairFeatures {
  float iou_3d = 0.0f;
  float containment = 0.0f;
  float center_score = 0.0f;
  float size_ratio = 0.0f;
  float semantic_similarity = 0.0f;
  float recency_score = 0.0f;
  float appearance_similarity_shadow = 0.0f;
  float identity_score = 0.0f;
  float score = 0.0f;
  bool candidate = false;
  bool strong_identity = false;
  bool identity_conflict = false;
};

struct AssociationResult {
  // Indexed by physical observation; nullopt means a new track.
  std::vector<std::optional<std::size_t>> track_by_observation;
  std::vector<std::vector<AssociationPairFeatures>> pair_features;
};

float box2dIou(const std::array<float, 4>& lhs,
               const std::array<float, 4>& rhs);
float orientedBoxIou(const Eigen::Vector3f& lhs_center,
                     const Eigen::Vector3f& lhs_size,
                     float lhs_yaw,
                     const Eigen::Vector3f& rhs_center,
                     const Eigen::Vector3f& rhs_size,
                     float rhs_yaw);
float orientedBoxContainment(const Eigen::Vector3f& lhs_center,
                             const Eigen::Vector3f& lhs_size,
                             float lhs_yaw,
                             const Eigen::Vector3f& rhs_center,
                             const Eigen::Vector3f& rhs_size,
                             float rhs_yaw);

SmallObjectIdentityConfig makeSmallObjectIdentityConfig(
    const std::vector<std::string>& group_specs,
    float max_volume_m3 = 0.025f,
    float max_extent_m = 0.65f,
    float min_iou_3d = 0.01f,
    float max_normalized_center_distance = 0.35f,
    float min_volume_ratio = 0.50f);

SmallObjectIdentityFeatures evaluateSmallObjectIdentity(
    const std::string& lhs_label,
    const Eigen::Vector3f& lhs_center,
    const Eigen::Vector3f& lhs_size,
    float lhs_yaw,
    const std::string& rhs_label,
    const Eigen::Vector3f& rhs_center,
    const Eigen::Vector3f& rhs_size,
    float rhs_yaw,
    const SmallObjectIdentityConfig& config);

std::vector<InstanceObservation, Eigen::aligned_allocator<InstanceObservation>>
clusterPhysicalObservations(
    const std::vector<InstanceObservation,
                      Eigen::aligned_allocator<InstanceObservation>>& input,
    const PhysicalObservationConfig& config);

AssociationResult associatePhysicalObservations(
    const std::vector<InstanceObservation,
                      Eigen::aligned_allocator<InstanceObservation>>& observations,
    const std::vector<InstanceTrack, Eigen::aligned_allocator<InstanceTrack>>& tracks,
    TimeNanoseconds now_ns,
    const AssociationScoringConfig& config);

}  // namespace roomie
