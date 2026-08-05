#pragma once

#include <chrono>
#include <cstdint>

#include "roomie/scene/scene_state.hpp"

namespace roomie {

// steady_clock has no portable cross-process epoch. Tag every monotonic SLO
// timestamp with a process-unique id so restored durable tasks explicitly
// fall back to their Unix deadline instead of comparing unrelated counters.
inline const RunId& artifactSteadyClockEpoch() {
  static const RunId epoch = makeRunId();
  return epoch;
}

inline std::int64_t artifactSteadyNowNanoseconds() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline bool hasCurrentArtifactSteadyClock(const ArtifactSloContext& slo) {
  return slo.monotonicTracked() &&
         slo.steady_clock_epoch == artifactSteadyClockEpoch();
}

}  // namespace roomie
