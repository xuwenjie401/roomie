#include "roomie/utils/visualization_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace roomie {
namespace {

std::uint32_t fnv1a(const std::string& value) {
  std::uint32_t hash = 2166136261u;
  for (const unsigned char c : value) {
    hash ^= c;
    hash *= 16777619u;
  }
  return hash;
}

RgbColor hsvToRgb(float h, float s, float v) {
  h = h - std::floor(h);
  const float c = v * s;
  const float x = c * (1.0f - std::abs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
  const float m = v - c;

  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  if (h < 1.0f / 6.0f) {
    r = c;
    g = x;
  } else if (h < 2.0f / 6.0f) {
    r = x;
    g = c;
  } else if (h < 3.0f / 6.0f) {
    g = c;
    b = x;
  } else if (h < 4.0f / 6.0f) {
    g = x;
    b = c;
  } else if (h < 5.0f / 6.0f) {
    r = x;
    b = c;
  } else {
    r = c;
    b = x;
  }

  return {r + m, g + m, b + m};
}

}  // namespace

RgbColor colorForLabel(const std::string& label, int semantic_id) {
  const std::string key =
      label.empty() ? ("semantic:" + std::to_string(semantic_id)) : label;
  const std::uint32_t hash = fnv1a(key);
  const float hue = static_cast<float>(hash & 0xffu) / 255.0f;
  float saturation = 0.62f + 0.30f * static_cast<float>((hash >> 8u) & 0xffu) / 255.0f;
  float value = 0.68f + 0.22f * static_cast<float>((hash >> 16u) & 0xffu) / 255.0f;
  if ((hue >= 0.11f && hue <= 0.20f) || hue >= 0.86f || hue <= 0.02f) {
    saturation = std::max(saturation, 0.82f);
    value = std::min(value, 0.74f);
  }
  return hsvToRgb(hue, saturation, value);
}

std::array<Eigen::Vector3f, 8> yawObbCorners(const Eigen::Vector3f& center_world,
                                             const Eigen::Vector3f& size_m,
                                             float yaw_rad) {
  const Eigen::Vector3f half = 0.5f * size_m.cwiseMax(Eigen::Vector3f::Constant(0.01f));
  const float c = std::cos(yaw_rad);
  const float s = std::sin(yaw_rad);
  const Eigen::Matrix3f R =
      (Eigen::Matrix3f() << c, -s, 0.0f, s, c, 0.0f, 0.0f, 0.0f, 1.0f).finished();

  const std::array<Eigen::Vector3f, 8> local = {
      Eigen::Vector3f(-half.x(), -half.y(), -half.z()),
      Eigen::Vector3f(half.x(), -half.y(), -half.z()),
      Eigen::Vector3f(half.x(), half.y(), -half.z()),
      Eigen::Vector3f(-half.x(), half.y(), -half.z()),
      Eigen::Vector3f(-half.x(), -half.y(), half.z()),
      Eigen::Vector3f(half.x(), -half.y(), half.z()),
      Eigen::Vector3f(half.x(), half.y(), half.z()),
      Eigen::Vector3f(-half.x(), half.y(), half.z()),
  };

  std::array<Eigen::Vector3f, 8> corners;
  for (std::size_t i = 0; i < local.size(); ++i) {
    corners[i] = center_world + R * local[i];
  }
  return corners;
}

void fillLineListFromCorners(const std::array<Eigen::Vector3f, 8>& corners,
                             visualization_msgs::msg::Marker* marker) {
  static constexpr std::array<std::array<int, 2>, 12> kEdges = {{
      {{0, 1}},
      {{1, 2}},
      {{2, 3}},
      {{3, 0}},
      {{4, 5}},
      {{5, 6}},
      {{6, 7}},
      {{7, 4}},
      {{0, 4}},
      {{1, 5}},
      {{2, 6}},
      {{3, 7}},
  }};

  marker->points.clear();
  marker->points.reserve(kEdges.size() * 2);
  for (const auto& edge : kEdges) {
    for (const int index : edge) {
      geometry_msgs::msg::Point point;
      point.x = corners[static_cast<std::size_t>(index)].x();
      point.y = corners[static_cast<std::size_t>(index)].y();
      point.z = corners[static_cast<std::size_t>(index)].z();
      marker->points.push_back(point);
    }
  }
}

}  // namespace roomie
