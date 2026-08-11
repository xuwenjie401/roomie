#include <gtest/gtest.h>

#include "roomie/pipeline/presence_evidence.hpp"

namespace roomie {
namespace {

InferenceResponse responseWithDepth(float depth_m) {
  InferenceResponse response;
  response.ok = true;
  response.has_camera_pose = true;
  response.T_world_camera = Eigen::Isometry3f::Identity();
  auto depth = std::make_shared<DepthBuffer>();
  depth->width = 100;
  depth->height = 100;
  depth->depth_m.assign(10'000, depth_m);
  auto context = std::make_shared<VisibilityContext>();
  context->depth = std::move(depth);
  context->intrinsics = CameraIntrinsics{100, 100, 100.0f, 100.0f,
                                         50.0f, 50.0f};
  response.visibility_context = std::move(context);
  return response;
}

InstanceTrack visibleTrack() {
  InstanceTrack track;
  track.track_id = 1;
  track.object_id = 1;
  track.state = InstanceTrackState::kStable;
  track.publishable = true;
  track.center_world = {0.0f, 0.0f, 2.0f};
  track.size_m = {0.4f, 0.4f, 0.4f};
  return track;
}

TEST(PresenceEvidence, DistinguishesFreeSpaceOccupancyAndOcclusion) {
  const PresenceEvidenceConfig config;
  const InstanceTrack track = visibleTrack();
  EXPECT_EQ(evaluateVisibility(track, responseWithDepth(4.0f), config).kind,
            VisibilityEvidenceKind::kFreeSpace);
  EXPECT_EQ(evaluateVisibility(track, responseWithDepth(2.0f), config).kind,
            VisibilityEvidenceKind::kOccupied);
  const VisibilityEvidence occluded =
      evaluateVisibility(track, responseWithDepth(1.0f), config);
  EXPECT_EQ(occluded.kind, VisibilityEvidenceKind::kUnknown);
  EXPECT_EQ(occluded.reason, "occluded");
}

TEST(PresenceEvidence, RequiresDistinctPositiveFramesAndThreeNegativeFrames) {
  PresenceEvidenceConfig config;
  InstanceTrack track;
  InstanceObservation observation;
  observation.bbox_quality = 0.8f;
  observation.time_ns = 1'000'000'000LL;
  addPositivePresenceEvidence(&track, observation, config);
  addPositivePresenceEvidence(&track, observation, config);
  EXPECT_FALSE(hasPositivePresenceConfirmation(track, observation.time_ns,
                                               config));
  observation.time_ns += 100'000'000LL;
  addPositivePresenceEvidence(&track, observation, config);
  EXPECT_TRUE(hasPositivePresenceConfirmation(track, observation.time_ns,
                                              config));

  track.existence_log_odds = 0.0f;
  track.negative_evidence_timestamps_ns.clear();
  VisibilityEvidence free;
  free.kind = VisibilityEvidenceKind::kFreeSpace;
  free.reliability = 0.9f;
  free.reason = "registered_depth_free_space";
  for (int i = 0; i < 2; ++i) {
    addFreeSpacePresenceEvidence(&track, 2'000'000'000LL + i, free, config);
  }
  EXPECT_FALSE(hasNegativePresenceConfirmation(track, 2'000'000'001LL,
                                               config));
  addFreeSpacePresenceEvidence(&track, 2'000'000'002LL, free, config);
  EXPECT_TRUE(hasNegativePresenceConfirmation(track, 2'000'000'002LL,
                                              config));
}

TEST(PresenceEvidence, UnknownVisibilityDoesNotChangeBelief) {
  const PresenceEvidenceConfig config;
  InstanceTrack track = visibleTrack();
  track.existence_log_odds = 2.0f;
  InferenceResponse response;
  response.ok = true;
  EXPECT_EQ(evaluateVisibility(track, response, config).kind,
            VisibilityEvidenceKind::kUnknown);
  EXPECT_FLOAT_EQ(track.existence_log_odds, 2.0f);
}

}  // namespace
}  // namespace roomie
