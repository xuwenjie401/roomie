#pragma once

#include <optional>
#include <string>
#include <unordered_map>

namespace roomie {

struct LabelConfidenceThresholds {
  std::optional<float> configured_default;
  std::unordered_map<std::string, float> by_label;
};

LabelConfidenceThresholds loadLabelConfidenceThresholds(
    const std::string& path,
    const std::string& stage);

float labelConfidenceThreshold(
    const LabelConfidenceThresholds& thresholds,
    const std::string& label,
    float fallback);

}  // namespace roomie
