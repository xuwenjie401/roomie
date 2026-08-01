#pragma once

#include <array>
#include <string>

#include <Eigen/Geometry>
#include <visualization_msgs/msg/marker.hpp>

namespace roomie {

struct RgbColor {
  float r = 1.0f;
  float g = 1.0f;
  float b = 1.0f;
};

RgbColor colorForLabel(const std::string& label, int semantic_id);

std::array<Eigen::Vector3f, 8> yawObbCorners(const Eigen::Vector3f& center_world,
                                             const Eigen::Vector3f& size_m,
                                             float yaw_rad);

void fillLineListFromCorners(const std::array<Eigen::Vector3f, 8>& corners,
                             visualization_msgs::msg::Marker* marker);

}  // namespace roomie
