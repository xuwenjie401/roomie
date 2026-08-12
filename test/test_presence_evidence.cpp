#include <gtest/gtest.h>

#include <cmath>

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

TEST(PresenceEvidence,
     RequiresDistinctPositiveViewpointsAndThreeNegativeFrames) {
  PresenceEvidenceConfig config;
  InstanceTrack track;
  track.center_world = {0.0f, 0.0f, 2.0f};
  track.size_m = {0.4f, 0.4f, 0.4f};
  InstanceObservation observation;
  observation.bbox_quality = 0.8f;
  observation.time_ns = 1'000'000'000LL;
  observation.camera_id = "head";
  observation.eligible_frame_index = 1;
  observation.has_camera_pose = true;
  observation.camera_position_world = Eigen::Vector3f::Zero();
  addPositivePresenceEvidence(&track, observation, config);
  addPositivePresenceEvidence(&track, observation, config);
  EXPECT_FALSE(hasPositivePresenceConfirmation(track, 1, config));
  observation.time_ns += 100'000'000LL;
  observation.eligible_frame_index = 2;
  addPositivePresenceEvidence(&track, observation, config);
  EXPECT_FALSE(hasPositivePresenceConfirmation(track, 2, config));
  EXPECT_EQ(track.last_presence_evidence_reason,
            "matched_detection_same_viewpoint");

  observation.time_ns += 100'000'000LL;
  observation.eligible_frame_index = 3;
  observation.camera_position_world.x() = 0.12f;
  addPositivePresenceEvidence(&track, observation, config);
  EXPECT_TRUE(hasPositivePresenceConfirmation(track, 3, config));

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

TEST(PresenceEvidence, ObjectRelativeViewAngleCanQualifyBelowBaseline) {
  const PresenceEvidenceConfig config;
  InstanceTrack track;
  track.center_world = Eigen::Vector3f::Zero();
  track.size_m = {2.0f, 2.0f, 2.0f};

  InstanceObservation observation;
  observation.camera_id = "head";
  observation.has_camera_pose = true;
  observation.bbox_quality = 0.8f;
  observation.time_ns = 1'000'000'000LL;
  observation.eligible_frame_index = 1;
  observation.camera_position_world = {0.0f, 0.0f, -1.0f};
  addPositivePresenceEvidence(&track, observation, config);

  constexpr float angle_rad = 7.0f * 3.14159265358979323846f / 180.0f;
  observation.time_ns += 100'000'000LL;
  observation.eligible_frame_index = 2;
  observation.camera_position_world =
      {std::sin(angle_rad), 0.0f, -std::cos(angle_rad)};
  ASSERT_LT((observation.camera_position_world -
             track.positive_presence_evidence_history.front()
                 .camera_position_world)
                .norm(),
            config.viewpoint_baseline_max_m);
  addPositivePresenceEvidence(&track, observation, config);
  EXPECT_TRUE(hasPositivePresenceConfirmation(track, 2, config));
}

TEST(PresenceEvidence, PositiveWindowAdvancesOnlyOnEligibleFrames) {
  PresenceEvidenceConfig config;
  config.positive_window_eligible_frames = 3;
  InstanceTrack track;
  track.center_world = {0.0f, 0.0f, 2.0f};
  track.size_m = {0.4f, 0.4f, 0.4f};

  InstanceObservation observation;
  observation.camera_id = "head";
  observation.has_camera_pose = true;
  observation.bbox_quality = 0.8f;
  observation.time_ns = 1'000'000'000LL;
  observation.eligible_frame_index = 10;
  observation.camera_position_world = Eigen::Vector3f::Zero();
  addPositivePresenceEvidence(&track, observation, config);

  observation.time_ns += 60'000'000'000LL;
  observation.eligible_frame_index = 11;
  observation.camera_position_world.x() = 0.12f;
  addPositivePresenceEvidence(&track, observation, config);
  EXPECT_TRUE(hasPositivePresenceConfirmation(track, 11, config));

  advancePositivePresenceWindow(&track, 13, config);
  EXPECT_FALSE(hasPositivePresenceConfirmation(track, 13, config));
  ASSERT_EQ(track.positive_presence_evidence_history.size(), 1U);
  EXPECT_EQ(track.positive_presence_evidence_history.front()
                .eligible_frame_index,
            11U);
}

TEST(PresenceEvidence, SameViewpointRequiresTwoStrictHighQualityObservations) {
  const PresenceEvidenceConfig config;
  InstanceTrack track;
  track.center_world = {0.0f, 0.0f, 1.5f};
  track.size_m = {0.4f, 0.4f, 0.4f};

  InstanceObservation observation;
  observation.camera_id = "head";
  observation.has_camera_pose = true;
  observation.camera_position_world = Eigen::Vector3f::Zero();
  observation.confidence = 0.90f;
  observation.bbox_quality = 0.90f;
  observation.camera_distance_m = 1.5f;
  observation.time_ns = 1'000'000'000LL;
  observation.eligible_frame_index = 1;
  addPositivePresenceEvidence(&track, observation, config);
  EXPECT_FALSE(hasPositivePresenceConfirmation(track, 1, config));

  observation.time_ns += 100'000'000LL;
  observation.eligible_frame_index = 2;
  addPositivePresenceEvidence(&track, observation, config);
  EXPECT_TRUE(hasPositivePresenceConfirmation(track, 2, config));
  EXPECT_EQ(track.last_presence_evidence_reason,
            "matched_detection_same_viewpoint_high_quality");
}

TEST(PresenceEvidence, SameViewpointFallbackRejectsWeakQualitySignals) {
  const PresenceEvidenceConfig config;
  const auto confirms = [&config](float distance,
                                  float confidence,
                                  float bbox_quality) {
    InstanceTrack track;
    track.center_world = {0.0f, 0.0f, 1.5f};
    track.size_m = {0.4f, 0.4f, 0.4f};
    InstanceObservation observation;
    observation.camera_id = "head";
    observation.has_camera_pose = true;
    observation.camera_position_world = Eigen::Vector3f::Zero();
    observation.camera_distance_m = distance;
    observation.confidence = confidence;
    observation.bbox_quality = bbox_quality;
    for (std::uint64_t frame = 1; frame <= 2; ++frame) {
      observation.eligible_frame_index = frame;
      observation.time_ns = static_cast<TimeNanoseconds>(frame) * 100'000'000LL;
      addPositivePresenceEvidence(&track, observation, config);
    }
    return hasPositivePresenceConfirmation(track, 2, config);
  };

  EXPECT_FALSE(confirms(2.6f, 0.90f, 0.90f));
  EXPECT_FALSE(confirms(1.5f, 0.84f, 0.90f));
  EXPECT_FALSE(confirms(1.5f, 0.90f, 0.79f));
}

TEST(PresenceEvidence,
     ConfirmationBypassWhitelistSkipsOnlyTheViewpointQualityClause) {
  PresenceEvidenceConfig config;
  config.confirmation_bypass_labels = {"medicine_carton"};
  InstanceTrack track;
  track.label = "Medicine Carton";
  track.center_world = {0.0f, 0.0f, 2.0f};
  track.size_m = {0.2f, 0.1f, 0.1f};

  InstanceObservation observation;
  observation.camera_id = "hand_left_color";
  observation.has_camera_pose = true;
  observation.camera_position_world = Eigen::Vector3f::Zero();
  observation.camera_distance_m = 2.0f;
  observation.confidence = 0.30f;
  observation.bbox_quality = 0.80f;
  observation.time_ns = 1'000'000'000LL;
  observation.eligible_frame_index = 1;
  addPositivePresenceEvidence(&track, observation, config);
  EXPECT_FALSE(hasPositivePresenceConfirmation(track, 1, config));

  observation.time_ns += 100'000'000LL;
  observation.eligible_frame_index = 2;
  addPositivePresenceEvidence(&track, observation, config);
  ASSERT_TRUE(hasPositivePresenceConfirmation(track, 2, config));

  track.existence_log_odds = config.active_threshold - 0.01f;
  EXPECT_FALSE(hasPositivePresenceConfirmation(track, 2, config));
  track.existence_log_odds = config.active_threshold;
  track.positive_window_interruptions = config.max_positive_interruptions + 1;
  EXPECT_FALSE(hasPositivePresenceConfirmation(track, 2, config));

  track.positive_window_interruptions = 0;
  track.label = "book";
  EXPECT_FALSE(hasPositivePresenceConfirmation(track, 2, config));
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
