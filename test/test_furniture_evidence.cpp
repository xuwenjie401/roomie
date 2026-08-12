#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "roomie/pipeline/furniture_evidence.hpp"

namespace roomie {
namespace {

InstanceObservation furnitureObservation(std::uint64_t frame_index,
                                         const std::string& label) {
  InstanceObservation observation;
  observation.eligible_frame_index = frame_index;
  observation.detection.label = label;
  observation.label_votes[label] = 1.0f;
  // A stationary/no-pose observation must remain valid promotion evidence.
  observation.has_camera_pose = false;
  return observation;
}

void addLabelFrames(InstanceTrack* track,
                    const std::string& label,
                    std::uint64_t first_frame,
                    std::size_t count,
                    const FurnitureGraphConfig& config) {
  for (std::size_t offset = 0; offset < count; ++offset) {
    recordFurnitureDetectionEvidence(
        track, furnitureObservation(first_frame + offset, label), config);
  }
}

TEST(FurnitureEvidence, RequiresElevenSameClassFramesWithoutViewpointChange) {
  const FurnitureGraphConfig config = defaultFurnitureGraphConfig();
  InstanceTrack track;
  track.label = "sofa";

  addLabelFrames(&track, "sofa", 1, 10, config);
  EXPECT_EQ(furnitureDetectionEvidenceCount(track, "sofa", 10, 30), 10U);
  EXPECT_FALSE(hasFurnitureCreationEvidence(track, 10, config));

  recordFurnitureDetectionEvidence(
      &track, furnitureObservation(11, "sofa"), config);
  EXPECT_EQ(furnitureDetectionEvidenceCount(track, "sofa", 11, 30), 11U);
  EXPECT_TRUE(hasFurnitureCreationEvidence(track, 11, config));
}

TEST(FurnitureEvidence, CountsEachClassAtMostOncePerAssociatedFrame) {
  const FurnitureGraphConfig config = defaultFurnitureGraphConfig();
  InstanceTrack track;
  track.label = "chair";
  const InstanceObservation observation = furnitureObservation(7, "chair");

  recordFurnitureDetectionEvidence(&track, observation, config);
  recordFurnitureDetectionEvidence(&track, observation, config);

  EXPECT_EQ(furnitureDetectionEvidenceCount(track, "chair", 7, 30), 1U);
}

TEST(FurnitureEvidence, DoesNotUseOtherClassSupportForCurrentFurnitureLabel) {
  const FurnitureGraphConfig config = defaultFurnitureGraphConfig();
  InstanceTrack track;
  track.label = "sofa";

  addLabelFrames(&track, "sofa", 1, 7, config);
  addLabelFrames(&track, "bathtub", 8, 6, config);

  EXPECT_EQ(furnitureDetectionEvidenceCount(track, "sofa", 13, 30), 7U);
  EXPECT_FALSE(hasFurnitureCreationEvidence(track, 13, config));
  EXPECT_EQ(track.furniture_detection_frames_by_label.count("bathtub"), 0U);
}

TEST(FurnitureEvidence, ExpiresFramesOutsideThirtyFrameWindow) {
  const FurnitureGraphConfig config = defaultFurnitureGraphConfig();
  InstanceTrack track;
  track.label = "desk";
  addLabelFrames(&track, "desk", 1, 11, config);

  EXPECT_TRUE(hasFurnitureCreationEvidence(track, 30, config));
  EXPECT_EQ(furnitureDetectionEvidenceCount(track, "desk", 31, 30), 10U);
  EXPECT_FALSE(hasFurnitureCreationEvidence(track, 31, config));
}

TEST(FurnitureEvidence, MatchesReviewedReplayAcceptanceMatrix) {
  const FurnitureGraphConfig config = defaultFurnitureGraphConfig();
  struct ReplayCase {
    int object_id;
    const char* label;
    std::size_t same_class_frames_in_window;
    bool should_create;
  };
  const std::vector<ReplayCase> cases = {
      {12, "cabinet", 21, true}, {64, "sofa", 17, true},
      {53, "shelf", 17, true},   {32, "desk", 7, false},
      {40, "desk", 3, false},   {66, "desk", 9, false},
      {50, "sofa", 7, false},   {34, "chair", 4, false},
      {23, "desk", 10, false},  {52, "nightstand", 6, false},
  };

  for (const ReplayCase& replay : cases) {
    InstanceTrack track;
    track.track_id = replay.object_id;
    track.label = replay.label;
    addLabelFrames(&track, replay.label, 1,
                   replay.same_class_frames_in_window, config);
    EXPECT_EQ(hasFurnitureCreationEvidence(track, 30, config),
              replay.should_create)
        << "reviewed object #" << replay.object_id;
  }
}

}  // namespace
}  // namespace roomie
