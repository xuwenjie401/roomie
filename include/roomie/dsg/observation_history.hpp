#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "roomie/dsg/object_graph.hpp"

namespace roomie {

inline void retainRecentObservationTimestamps(
    std::vector<TimeNanoseconds>* timestamps,
    std::size_t capacity) {
  if (timestamps == nullptr || timestamps->size() <= capacity) {
    return;
  }
  std::sort(timestamps->begin(), timestamps->end());
  timestamps->erase(std::unique(timestamps->begin(), timestamps->end()),
                    timestamps->end());
  if (timestamps->size() > capacity) {
    timestamps->erase(
        timestamps->begin(),
        timestamps->end() - static_cast<std::ptrdiff_t>(capacity));
  }
}

inline void retainRecentObservationQuality(
    std::vector<ObservationQualitySample>* history,
    std::size_t capacity) {
  if (history == nullptr || history->size() <= capacity) {
    return;
  }
  std::stable_sort(
      history->begin(), history->end(),
      [](const ObservationQualitySample& lhs,
         const ObservationQualitySample& rhs) {
        return lhs.time_ns < rhs.time_ns;
      });
  history->erase(
      history->begin(),
      history->end() - static_cast<std::ptrdiff_t>(capacity));
}

inline void retainRecentObservationHistory(InstanceTrack* track,
                                           std::size_t capacity) {
  if (track == nullptr) {
    return;
  }
  retainRecentObservationTimestamps(&track->observation_timestamps_ns,
                                    capacity);
  retainRecentObservationQuality(&track->observation_quality_history,
                                 capacity);
}

inline void retainRecentObservationHistory(ObjectNode* object,
                                           std::size_t capacity) {
  if (object == nullptr) {
    return;
  }
  retainRecentObservationTimestamps(&object->observation_timestamps_ns,
                                    capacity);
}

}  // namespace roomie
