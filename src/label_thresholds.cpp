#include "roomie/pipeline/label_thresholds.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace roomie {

namespace {

float parseThreshold(const nlohmann::json& value,
                     const std::string& description,
                     const std::string& path) {
  if (!value.is_number()) {
    throw std::runtime_error(description + " must be a number: " + path);
  }
  const double threshold = value.get<double>();
  if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
    throw std::runtime_error(description + " must be in [0, 1]: " + path);
  }
  return static_cast<float>(threshold);
}

}  // namespace

LabelConfidenceThresholds loadLabelConfidenceThresholds(
    const std::string& path,
    const std::string& stage) {
  if (path.empty()) {
    return {};
  }

  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot read label confidence thresholds: " + path);
  }

  nlohmann::json document;
  try {
    stream >> document;
  } catch (const nlohmann::json::exception& exception) {
    throw std::runtime_error("invalid label confidence thresholds in " + path +
                             ": " + exception.what());
  }
  if (!document.is_object() || !document.contains(stage) ||
      !document.at(stage).is_object()) {
    throw std::runtime_error(
        "label confidence thresholds must contain an object for stage '" +
        stage + "': " + path);
  }

  const nlohmann::json& stage_config = document.at(stage);
  if (!stage_config.contains("default")) {
    throw std::runtime_error("label confidence threshold stage '" + stage +
                             "' is missing default: " + path);
  }
  if (!stage_config.contains("labels") ||
      !stage_config.at("labels").is_object()) {
    throw std::runtime_error("label confidence threshold stage '" + stage +
                             "' must contain a labels object: " + path);
  }

  LabelConfidenceThresholds thresholds;
  thresholds.configured_default = parseThreshold(
      stage_config.at("default"),
      "default threshold for stage '" + stage + "'",
      path);
  const nlohmann::json& labels = stage_config.at("labels");
  thresholds.by_label.reserve(labels.size());
  for (const auto& [label, value] : labels.items()) {
    if (label.empty()) {
      throw std::runtime_error(
          "label confidence threshold labels must be non-empty: " + path);
    }
    thresholds.by_label.emplace(
        label,
        parseThreshold(value,
                       "label confidence threshold for '" + label + "'",
                       path));
  }
  return thresholds;
}

float labelConfidenceThreshold(
    const LabelConfidenceThresholds& thresholds,
    const std::string& label,
    float fallback) {
  const auto found = thresholds.by_label.find(label);
  if (found != thresholds.by_label.end()) {
    return found->second;
  }
  return thresholds.configured_default.value_or(fallback);
}

}  // namespace roomie
