#pragma once

#include <string>

#include "roomie/dsg/object_graph.hpp"

namespace roomie {

struct PresenceEvidenceConfig {
  float log_odds_cap = 4.0f;
  float active_threshold = 1.0986123f;
  float archive_threshold = -1.0986123f;
  TimeNanoseconds evidence_window_ns = 2'000'000'000LL;
  int min_positive_frames = 2;
  int max_positive_interruptions = 1;
  int min_negative_frames = 3;
  int min_depth_samples = 32;
  float min_valid_depth_coverage = 0.50f;
  float min_free_space_ratio = 0.70f;
  float max_occlusion_ratio = 0.20f;
  float depth_margin_m = 0.04f;
  int image_edge_margin_px = 4;
};

enum class VisibilityEvidenceKind {
  kUnknown,
  kOccupied,
  kFreeSpace,
};

struct VisibilityEvidence {
  VisibilityEvidenceKind kind = VisibilityEvidenceKind::kUnknown;
  int projected_samples = 0;
  int valid_depth_samples = 0;
  int occluded_samples = 0;
  int occupied_samples = 0;
  int free_space_samples = 0;
  float valid_coverage = 0.0f;
  float occlusion_ratio = 0.0f;
  float free_space_ratio = 0.0f;
  float reliability = 0.0f;
  std::string reason = "unknown";
};

VisibilityEvidence evaluateVisibility(const InstanceTrack& track,
                                      const InferenceResponse& response,
                                      const PresenceEvidenceConfig& config);

void addPositivePresenceEvidence(InstanceTrack* track,
                                 const InstanceObservation& observation,
                                 const PresenceEvidenceConfig& config);
void addFreeSpacePresenceEvidence(InstanceTrack* track,
                                  TimeNanoseconds time_ns,
                                  const VisibilityEvidence& evidence,
                                  const PresenceEvidenceConfig& config);
bool hasPositivePresenceConfirmation(const InstanceTrack& track,
                                     TimeNanoseconds now_ns,
                                     const PresenceEvidenceConfig& config);
bool hasNegativePresenceConfirmation(const InstanceTrack& track,
                                     TimeNanoseconds now_ns,
                                     const PresenceEvidenceConfig& config);
float presenceProbability(float log_odds);
const char* presenceStateName(InstanceTrackState state);

}  // namespace roomie
