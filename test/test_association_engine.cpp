#include <gtest/gtest.h>

#include <stdexcept>

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

TEST(AssociationEngine, ParsesExplicitSmallObjectIdentityFamiliesStrictly) {
  const SmallObjectIdentityConfig config = makeSmallObjectIdentityConfig(
      {"Bottle | bottled-water",
       "box|labeled_package|printed carton|medicine_carton"});
  EXPECT_EQ(config.family_by_label.at("bottled_water"), "bottle");
  EXPECT_EQ(config.family_by_label.at("printed_carton"), "box");
  EXPECT_THROW(makeSmallObjectIdentityConfig({"box"}), std::invalid_argument);
  EXPECT_THROW(makeSmallObjectIdentityConfig({"box||carton"}),
               std::invalid_argument);
  EXPECT_THROW(makeSmallObjectIdentityConfig({"box|carton", "box|package"}),
               std::invalid_argument);
}

TEST(AssociationEngine, StrictBottleIdentityClustersAndBypassesSoftThreshold) {
  const Eigen::Vector3f bottle_center(0.866043f, -2.538530f, 0.781970f);
  const Eigen::Vector3f bottle_size(0.057092f, 0.057383f, 0.130203f);
  const Eigen::Vector3f water_center(0.884047f, -2.594693f, 0.802578f);
  const Eigen::Vector3f water_size(0.064928f, 0.063798f, 0.174672f);

  PhysicalObservationConfig physical_config;
  physical_config.small_object_identity =
      makeSmallObjectIdentityConfig({"bottle|bottled_water"});
  std::vector<InstanceObservation,
              Eigen::aligned_allocator<InstanceObservation>> input;
  input.push_back(observation("bottle", 45, bottle_center, bottle_size,
                              {0, 0, 20, 20}));
  input.push_back(observation("bottled_water", 93, water_center, water_size,
                              {100, 100, 120, 120}));
  const auto clustered = clusterPhysicalObservations(input, physical_config);
  ASSERT_EQ(clustered.size(), 1U);
  EXPECT_EQ(clustered.front().strong_identity_family, "bottle");
  EXPECT_GT(clustered.front().label_votes.at("bottle"), 0.0f);
  EXPECT_GT(clustered.front().label_votes.at("bottled_water"), 0.0f);

  InstanceTrack track;
  track.track_id = 292;
  track.object_id = 87;
  track.state = InstanceTrackState::kStable;
  track.publishable = true;
  track.center_world = bottle_center;
  track.size_m = bottle_size;
  track.label = "bottle";
  track.label_weights[track.label] = 1.0f;
  AssociationScoringConfig scoring;
  scoring.active_match_threshold = 0.99f;
  scoring.small_object_identity = physical_config.small_object_identity;
  const AssociationResult result = associatePhysicalObservations(
      {observation("bottled_water", 93, water_center, water_size,
                   {100, 100, 120, 120})},
      {track}, 1'000'000'000, scoring);
  ASSERT_TRUE(result.track_by_observation.front());
  EXPECT_TRUE(result.pair_features.front().front().strong_identity);
}

TEST(AssociationEngine, ConfiguredFamilyRejectsBottleCapEvenAtSameGeometry) {
  const SmallObjectIdentityConfig identity =
      makeSmallObjectIdentityConfig({"bottle|bottled_water"});
  InstanceTrack track;
  track.track_id = 1;
  track.object_id = 1;
  track.state = InstanceTrackState::kStable;
  track.publishable = true;
  track.center_world = {0, 0, 1};
  track.size_m = {0.06f, 0.06f, 0.15f};
  track.label = "bottle";
  track.label_weights[track.label] = 1.0f;
  AssociationScoringConfig scoring;
  scoring.small_object_identity = identity;
  const AssociationResult result = associatePhysicalObservations(
      {observation("bottle_cap", 101, track.center_world, track.size_m,
                   {0, 0, 20, 20})},
      {track}, 1'000'000'000, scoring);
  EXPECT_FALSE(result.track_by_observation.front());
  EXPECT_TRUE(result.pair_features.front().front().identity_conflict);

  PhysicalObservationConfig physical;
  physical.small_object_identity = identity;
  const auto clustered = clusterPhysicalObservations(
      {observation("bottle", 45, track.center_world, track.size_m,
                   {0, 0, 20, 20}),
       observation("bottle_cap", 101, track.center_world, track.size_m,
                   {0, 0, 20, 20})},
      physical);
  EXPECT_EQ(clustered.size(), 2U);
}

TEST(AssociationEngine, PackageFamilySupportsStrictTransitiveGeometry) {
  const SmallObjectIdentityConfig identity = makeSmallObjectIdentityConfig(
      {"box|labeled_package|printed_carton|medicine_carton"});
  const auto evaluate = [&identity](const std::string& lhs_label,
                                    const Eigen::Vector3f& lhs_center,
                                    const Eigen::Vector3f& lhs_size,
                                    float lhs_yaw,
                                    const std::string& rhs_label,
                                    const Eigen::Vector3f& rhs_center,
                                    const Eigen::Vector3f& rhs_size,
                                    float rhs_yaw) {
    return evaluateSmallObjectIdentity(
        lhs_label, lhs_center, lhs_size, lhs_yaw,
        rhs_label, rhs_center, rhs_size, rhs_yaw, identity);
  };
  const Eigen::Vector3f center42(2.162210f, 3.844470f, 0.828921f);
  const Eigen::Vector3f size42(0.211200f, 0.185070f, 0.264856f);
  const Eigen::Vector3f center50(2.208026f, 3.920180f, 0.841667f);
  const Eigen::Vector3f size50(0.219637f, 0.095061f, 0.265831f);
  const Eigen::Vector3f center55(2.178208f, 3.908470f, 0.766520f);
  const Eigen::Vector3f size55(0.253268f, 0.158681f, 0.084193f);
  const Eigen::Vector3f center44(2.177099f, 3.826206f, 1.166305f);
  const Eigen::Vector3f size44(0.219421f, 0.126531f, 0.250656f);

  EXPECT_TRUE(evaluate("box", center42, size42, -0.011774f,
                       "labeled_package", center50, size50, -0.606546f)
                  .eligible);
  EXPECT_TRUE(evaluate("labeled_package", center50, size50, -0.606546f,
                       "medicine_carton", center55, size55, 0.008728f)
                  .eligible);
  EXPECT_FALSE(evaluate("box", center42, size42, -0.011774f,
                        "medicine_carton", center55, size55, 0.008728f)
                   .eligible);
  EXPECT_FALSE(evaluate("box", center42, size42, -0.011774f,
                        "labeled_package", center44, size44, -0.196161f)
                   .eligible);
  EXPECT_TRUE(evaluate("printed_carton", center50, size50, -0.606546f,
                       "medicine_carton", center50, size50, -0.606546f)
                  .eligible);
  EXPECT_TRUE(evaluate("toy", center50, size50, -0.606546f,
                       "toy", center50, size50, -0.606546f)
                  .eligible);
  EXPECT_FALSE(evaluate("toy", center50, size50, -0.606546f,
                        "toy", center50 + Eigen::Vector3f(1.0f, 0.0f, 0.0f),
                        size50, -0.606546f)
                   .eligible);
}

}  // namespace
}  // namespace roomie
