#include <gtest/gtest.h>

#include "roomie/pipeline/association_engine.hpp"

namespace roomie {
namespace {

InstanceObservation observation(std::string label,
                                int semantic_id,
                                Eigen::Vector3f center,
                                Eigen::Vector3f size,
                                std::array<float, 4> box) {
  InstanceObservation value;
  value.time_ns = 1'000'000'000;
  value.confidence = 0.8f;
  value.bbox_quality = 0.8f;
  value.detection.label = std::move(label);
  value.detection.semantic_id = semantic_id;
  value.detection.center_world = center;
  value.detection.size_m = size;
  value.detection.box_xyxy = box;
  return value;
}

TEST(AssociationEngine, ClustersCrossLabelDetectionsIntoOnePhysicalObservation) {
  std::vector<InstanceObservation,
              Eigen::aligned_allocator<InstanceObservation>> input;
  input.push_back(observation("backpack", 1, {2.415f, -0.731f, 0.833f},
                              {0.46f, 0.375f, 0.462f}, {100, 100, 300, 400}));
  input.push_back(observation("handbag", 2, {2.412f, -0.764f, 0.796f},
                              {0.395f, 0.312f, 0.472f}, {104, 102, 296, 397}));
  const auto clustered =
      clusterPhysicalObservations(input, PhysicalObservationConfig{});
  ASSERT_EQ(clustered.size(), 1U);
  EXPECT_EQ(clustered.front().member_count, 2U);
  EXPECT_GT(clustered.front().label_votes.at("backpack"), 0.0f);
  EXPECT_GT(clustered.front().label_votes.at("handbag"), 0.0f);
}

TEST(AssociationEngine, CompleteLinkKeepsAdjacentPhysicalItemsSeparate) {
  std::vector<InstanceObservation,
              Eigen::aligned_allocator<InstanceObservation>> input;
  input.push_back(observation("bag", 1, {0, 0, 1}, {0.4f, 0.3f, 0.5f},
                              {10, 10, 110, 210}));
  input.push_back(observation("bag", 1, {0.7f, 0, 1}, {0.4f, 0.3f, 0.5f},
                              {120, 10, 220, 210}));
  EXPECT_EQ(clusterPhysicalObservations(input, PhysicalObservationConfig{}).size(),
            2U);
}

TEST(AssociationEngine, ExistingBackpackHandbagPairHasDurableMergeSupport) {
  InstanceTrack backpack;
  backpack.track_id = 76;
  backpack.object_id = 76;
  backpack.state = InstanceTrackState::kStable;
  backpack.publishable = true;
  backpack.center_world = {2.415f, -0.731f, 0.833f};
  backpack.size_m = {0.46f, 0.375f, 0.462f};
  backpack.yaw_rad = -1.492f;
  backpack.label = "backpack";
  InstanceTrack handbag = backpack;
  handbag.track_id = 99;
  handbag.object_id = 99;
  handbag.center_world = {2.412f, -0.764f, 0.796f};
  handbag.size_m = {0.395f, 0.312f, 0.472f};
  handbag.yaw_rad = -1.477f;
  handbag.label = "handbag";
  std::vector<InstanceTrack, Eigen::aligned_allocator<InstanceTrack>> tracks{
      backpack, handbag};
  std::vector<InstanceObservation,
              Eigen::aligned_allocator<InstanceObservation>> observations{
      observation("backpack", 1, backpack.center_world, backpack.size_m,
                  {100, 100, 300, 400})};
  const AssociationResult result = associatePhysicalObservations(
      observations, tracks, 1'000'000'000, AssociationScoringConfig{});
  ASSERT_EQ(result.pair_features.size(), 1U);
  EXPECT_GE(result.pair_features[0][0].identity_score, 0.75f);
  EXPECT_GE(result.pair_features[0][1].identity_score, 0.75f);
}

TEST(AssociationEngine, HungarianIsOrderInvariantAndSemanticsAreSoft) {
  InstanceTrack left;
  left.track_id = 10;
  left.state = InstanceTrackState::kStable;
  left.publishable = true;
  left.center_world = {0, 0, 1};
  left.size_m = {0.4f, 0.4f, 0.4f};
  left.label = "handbag";
  left.label_weights[left.label] = 2.0f;
  InstanceTrack right = left;
  right.track_id = 20;
  right.center_world.x() = 1.0f;
  right.label = "backpack";
  right.label_weights.clear();
  right.label_weights[right.label] = 2.0f;
  std::vector<InstanceTrack, Eigen::aligned_allocator<InstanceTrack>> tracks{
      left, right};

  std::vector<InstanceObservation,
              Eigen::aligned_allocator<InstanceObservation>> observations;
  observations.push_back(observation("backpack", 2, {0.02f, 0, 1},
                                     {0.4f, 0.4f, 0.4f}, {0, 0, 10, 10}));
  observations.push_back(observation("handbag", 1, {1.02f, 0, 1},
                                     {0.4f, 0.4f, 0.4f}, {20, 0, 30, 10}));
  const AssociationResult result = associatePhysicalObservations(
      observations, tracks, 1'000'000'000, AssociationScoringConfig{});
  ASSERT_TRUE(result.track_by_observation[0]);
  ASSERT_TRUE(result.track_by_observation[1]);
  EXPECT_EQ(tracks[*result.track_by_observation[0]].track_id, 10);
  EXPECT_EQ(tracks[*result.track_by_observation[1]].track_id, 20);

  std::reverse(observations.begin(), observations.end());
  const AssociationResult reversed = associatePhysicalObservations(
      observations, tracks, 1'000'000'000, AssociationScoringConfig{});
  ASSERT_TRUE(reversed.track_by_observation[0]);
  ASSERT_TRUE(reversed.track_by_observation[1]);
  EXPECT_EQ(tracks[*reversed.track_by_observation[0]].track_id, 20);
  EXPECT_EQ(tracks[*reversed.track_by_observation[1]].track_id, 10);
}

}  // namespace
}  // namespace roomie
