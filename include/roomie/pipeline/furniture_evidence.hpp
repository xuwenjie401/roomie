#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "roomie/dsg/object_graph.hpp"
#include "roomie/scene/furniture_graph.hpp"

namespace roomie {

bool isConfiguredFurnitureLabel(const std::string& label,
                                const FurnitureGraphConfig& config);

// Records each configured furniture class represented by this associated
// physical observation once for its detector frame. Camera pose is
// intentionally irrelevant.
void recordFurnitureDetectionEvidence(
    InstanceTrack* track,
    const InstanceObservation& observation,
    const FurnitureGraphConfig& config);

std::size_t furnitureDetectionEvidenceCount(
    const InstanceTrack& track,
    const std::string& label,
    std::uint64_t current_frame_index,
    std::size_t detection_window_frames);

bool hasFurnitureCreationEvidence(const InstanceTrack& track,
                                  std::uint64_t current_frame_index,
                                  const FurnitureGraphConfig& config);

}  // namespace roomie
