#include "roomie/pipeline/furniture_evidence.hpp"

#include <algorithm>
#include <iterator>
#include <set>

namespace roomie {
namespace {

std::uint64_t firstRetainedFrame(std::uint64_t current_frame_index,
                                 std::size_t detection_window_frames) {
  if (detection_window_frames == 0U) {
    return current_frame_index;
  }
  const std::uint64_t window =
      static_cast<std::uint64_t>(detection_window_frames);
  return current_frame_index >= window
             ? current_frame_index - window + 1U
             : 0U;
}

void pruneFurnitureDetectionEvidence(
    InstanceTrack* track,
    std::uint64_t current_frame_index,
    std::size_t detection_window_frames) {
  const std::uint64_t first_frame =
      firstRetainedFrame(current_frame_index, detection_window_frames);
  for (auto it = track->furniture_detection_frames_by_label.begin();
       it != track->furniture_detection_frames_by_label.end();) {
    auto& frames = it->second;
    const auto first_retained =
        std::lower_bound(frames.begin(), frames.end(), first_frame);
    frames.erase(frames.begin(), first_retained);
    if (frames.empty()) {
      it = track->furniture_detection_frames_by_label.erase(it);
    } else {
      ++it;
    }
  }
}

void appendFrameOnce(std::vector<std::uint64_t>* frames,
                     std::uint64_t frame_index) {
  const auto position =
      std::lower_bound(frames->begin(), frames->end(), frame_index);
  if (position == frames->end() || *position != frame_index) {
    frames->insert(position, frame_index);
  }
}

}  // namespace

bool isConfiguredFurnitureLabel(const std::string& label,
                                const FurnitureGraphConfig& config) {
  return config.classes.count(normalizeFurnitureLabel(label)) != 0U;
}

void recordFurnitureDetectionEvidence(
    InstanceTrack* track,
    const InstanceObservation& observation,
    const FurnitureGraphConfig& config) {
  if (track == nullptr) {
    return;
  }
  pruneFurnitureDetectionEvidence(
      track, observation.eligible_frame_index,
      config.object_creation_detection_window_frames);

  std::set<std::string> labels_in_frame;
  for (const auto& [raw_label, vote] : observation.label_votes) {
    if (vote <= 0.0f) {
      continue;
    }
    const std::string label = normalizeFurnitureLabel(raw_label);
    if (config.classes.count(label) != 0U) {
      labels_in_frame.insert(label);
    }
  }
  if (observation.label_votes.empty()) {
    const std::string label =
        normalizeFurnitureLabel(observation.detection.label);
    if (config.classes.count(label) != 0U) {
      labels_in_frame.insert(label);
    }
  }

  for (const std::string& label : labels_in_frame) {
    appendFrameOnce(&track->furniture_detection_frames_by_label[label],
                    observation.eligible_frame_index);
  }
}

std::size_t furnitureDetectionEvidenceCount(
    const InstanceTrack& track,
    const std::string& label,
    std::uint64_t current_frame_index,
    std::size_t detection_window_frames) {
  if (detection_window_frames == 0U) {
    return 0U;
  }
  const auto evidence = track.furniture_detection_frames_by_label.find(
      normalizeFurnitureLabel(label));
  if (evidence == track.furniture_detection_frames_by_label.end()) {
    return 0U;
  }
  const std::uint64_t first_frame =
      firstRetainedFrame(current_frame_index, detection_window_frames);
  const auto first = std::lower_bound(evidence->second.begin(),
                                      evidence->second.end(), first_frame);
  const auto after_last =
      std::upper_bound(first, evidence->second.end(), current_frame_index);
  return static_cast<std::size_t>(std::distance(first, after_last));
}

bool hasFurnitureCreationEvidence(const InstanceTrack& track,
                                  std::uint64_t current_frame_index,
                                  const FurnitureGraphConfig& config) {
  if (!isConfiguredFurnitureLabel(track.label, config)) {
    return false;
  }
  return furnitureDetectionEvidenceCount(
             track, track.label, current_frame_index,
             config.object_creation_detection_window_frames) >=
         config.object_creation_min_same_class_detection_frames;
}

}  // namespace roomie
