#include "roomie/pipeline/association_engine.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace roomie {
namespace {

constexpr float kEpsilon = 1.0e-6f;
using Polygon =
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>;

float clamp01(float value) {
  return std::clamp(value, 0.0f, 1.0f);
}

std::string normalizeIdentityLabel(const std::string& label) {
  std::string normalized;
  normalized.reserve(label.size());
  bool previous_separator = true;
  for (unsigned char character : label) {
    const bool separator = std::isspace(character) || character == '-' ||
                           character == '_';
    if (separator) {
      if (!previous_separator && !normalized.empty()) {
        normalized.push_back('_');
      }
      previous_separator = true;
      continue;
    }
    normalized.push_back(static_cast<char>(std::tolower(character)));
    previous_separator = false;
  }
  if (!normalized.empty() && normalized.back() == '_') {
    normalized.pop_back();
  }
  return normalized;
}

float volume(const Eigen::Vector3f& size) {
  if ((size.array() <= 0.0f).any()) {
    return 0.0f;
  }
  return size.x() * size.y() * size.z();
}

float volumeRatio(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
  const float a = volume(lhs);
  const float b = volume(rhs);
  return std::max(a, b) > kEpsilon ? std::min(a, b) / std::max(a, b)
                                   : 0.0f;
}

bool isSmallObjectSize(const Eigen::Vector3f& size,
                       const SmallObjectIdentityConfig& config) {
  return config.max_volume_m3 > 0.0f && config.max_extent_m > 0.0f &&
         (size.array() > 0.0f).all() &&
         volume(size) <= config.max_volume_m3 &&
         size.maxCoeff() <= config.max_extent_m;
}

float cross(const Eigen::Vector2f& lhs, const Eigen::Vector2f& rhs) {
  return lhs.x() * rhs.y() - lhs.y() * rhs.x();
}

Polygon corners(const Eigen::Vector3f& center,
                const Eigen::Vector3f& size,
                float yaw) {
  const Eigen::Vector2f half(0.5f * std::max(0.0f, size.x()),
                             0.5f * std::max(0.0f, size.y()));
  const float c = std::cos(yaw);
  const float s = std::sin(yaw);
  const Eigen::Matrix2f rotation =
      (Eigen::Matrix2f() << c, -s, s, c).finished();
  const Eigen::Vector2f origin(center.x(), center.y());
  return {origin + rotation * Eigen::Vector2f(-half.x(), -half.y()),
          origin + rotation * Eigen::Vector2f(half.x(), -half.y()),
          origin + rotation * Eigen::Vector2f(half.x(), half.y()),
          origin + rotation * Eigen::Vector2f(-half.x(), half.y())};
}

bool inside(const Eigen::Vector2f& point,
            const Eigen::Vector2f& start,
            const Eigen::Vector2f& end) {
  return cross(end - start, point - start) >= -1.0e-5f;
}

Eigen::Vector2f intersection(const Eigen::Vector2f& p0,
                             const Eigen::Vector2f& p1,
                             const Eigen::Vector2f& q0,
                             const Eigen::Vector2f& q1) {
  const Eigen::Vector2f r = p1 - p0;
  const Eigen::Vector2f s = q1 - q0;
  const float denominator = cross(r, s);
  if (std::abs(denominator) < kEpsilon) {
    return p1;
  }
  return p0 + cross(q0 - p0, s) / denominator * r;
}

Polygon clip(const Polygon& subject, const Polygon& clipper) {
  Polygon output = subject;
  for (std::size_t i = 0; i < clipper.size() && !output.empty(); ++i) {
    const Polygon input = output;
    output.clear();
    const Eigen::Vector2f start = clipper[i];
    const Eigen::Vector2f end = clipper[(i + 1) % clipper.size()];
    Eigen::Vector2f previous = input.back();
    bool previous_inside = inside(previous, start, end);
    for (const Eigen::Vector2f& current : input) {
      const bool current_inside = inside(current, start, end);
      if (current_inside != previous_inside) {
        output.push_back(intersection(previous, current, start, end));
      }
      if (current_inside) {
        output.push_back(current);
      }
      previous = current;
      previous_inside = current_inside;
    }
  }
  return output;
}

float area(const Polygon& polygon) {
  float twice_area = 0.0f;
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    twice_area += cross(polygon[i], polygon[(i + 1) % polygon.size()]);
  }
  return polygon.size() >= 3 ? 0.5f * std::abs(twice_area) : 0.0f;
}

float intersectionVolume(const Eigen::Vector3f& lhs_center,
                         const Eigen::Vector3f& lhs_size,
                         float lhs_yaw,
                         const Eigen::Vector3f& rhs_center,
                         const Eigen::Vector3f& rhs_size,
                         float rhs_yaw) {
  if (volume(lhs_size) <= kEpsilon || volume(rhs_size) <= kEpsilon) {
    return 0.0f;
  }
  const float overlap_area =
      area(clip(corners(lhs_center, lhs_size, lhs_yaw),
                corners(rhs_center, rhs_size, rhs_yaw)));
  const float lhs_min_z = lhs_center.z() - 0.5f * lhs_size.z();
  const float lhs_max_z = lhs_center.z() + 0.5f * lhs_size.z();
  const float rhs_min_z = rhs_center.z() - 0.5f * rhs_size.z();
  const float rhs_max_z = rhs_center.z() + 0.5f * rhs_size.z();
  return overlap_area *
         std::max(0.0f, std::min(lhs_max_z, rhs_max_z) -
                            std::max(lhs_min_z, rhs_min_z));
}

float sizeRatio(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
  if ((lhs.array() <= 0.0f).any() || (rhs.array() <= 0.0f).any()) {
    return 0.0f;
  }
  const auto score = [&lhs](const Eigen::Vector3f& candidate) {
    float value = 1.0f;
    for (int axis = 0; axis < 3; ++axis) {
      value = std::min(value, std::min(lhs[axis], candidate[axis]) /
                                  std::max(lhs[axis], candidate[axis]));
    }
    return value;
  };
  return std::max(score(rhs), score(Eigen::Vector3f(rhs.y(), rhs.x(), rhs.z())));
}

float semanticCosine(const InstanceTrack& track,
                     const InstanceObservation& observation) {
  std::map<std::string, float> observation_votes = observation.label_votes;
  if (observation_votes.empty() && !observation.detection.label.empty()) {
    observation_votes[observation.detection.label] =
        std::max(kEpsilon, observation.confidence);
  }
  std::map<std::string, float> track_votes = track.label_weights;
  if (track_votes.empty() && !track.label.empty()) {
    track_votes[track.label] = std::max(kEpsilon, track.confidence_mass);
  }
  float dot = 0.0f;
  float lhs_norm = 0.0f;
  float rhs_norm = 0.0f;
  for (const auto& [label, weight] : track_votes) {
    lhs_norm += weight * weight;
    const auto found = observation_votes.find(label);
    if (found != observation_votes.end()) {
      dot += weight * found->second;
    }
  }
  for (const auto& [label, weight] : observation_votes) {
    (void)label;
    rhs_norm += weight * weight;
  }
  return lhs_norm > kEpsilon && rhs_norm > kEpsilon
             ? clamp01(dot / std::sqrt(lhs_norm * rhs_norm))
             : 0.0f;
}

float vectorCosine(const std::vector<float>& lhs,
                   const std::vector<float>& rhs) {
  if (lhs.empty() || lhs.size() != rhs.size()) {
    return 0.0f;
  }
  double dot = 0.0;
  double lhs_norm = 0.0;
  double rhs_norm = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    dot += static_cast<double>(lhs[i]) * rhs[i];
    lhs_norm += static_cast<double>(lhs[i]) * lhs[i];
    rhs_norm += static_cast<double>(rhs[i]) * rhs[i];
  }
  return lhs_norm > kEpsilon && rhs_norm > kEpsilon
             ? clamp01(static_cast<float>(dot /
                                          std::sqrt(lhs_norm * rhs_norm)))
             : 0.0f;
}

bool samePhysicalItem(const InstanceObservation& lhs,
                      const InstanceObservation& rhs,
                      const PhysicalObservationConfig& config) {
  const RawDetection& a = lhs.detection;
  const RawDetection& b = rhs.detection;
  const SmallObjectIdentityFeatures identity = evaluateSmallObjectIdentity(
      a.label, a.center_world, a.size_m, a.yaw_rad,
      b.label, b.center_world, b.size_m, b.yaw_rad,
      config.small_object_identity);
  if (identity.identity_conflict) {
    return false;
  }
  const float diagonal = std::max(a.size_m.norm(), b.size_m.norm());
  const float normalized_distance =
      diagonal > kEpsilon
          ? (a.center_world - b.center_world).norm() / diagonal
          : std::numeric_limits<float>::infinity();
  const bool ordinary_physical_match =
      box2dIou(a.box_xyxy, b.box_xyxy) >= config.min_2d_iou &&
      volumeRatio(a.size_m, b.size_m) >= config.min_volume_ratio &&
      normalized_distance <= config.max_normalized_center_distance &&
      (orientedBoxIou(a.center_world, a.size_m, a.yaw_rad,
                      b.center_world, b.size_m, b.yaw_rad) >=
           config.min_3d_iou ||
       orientedBoxContainment(a.center_world, a.size_m, a.yaw_rad,
                              b.center_world, b.size_m, b.yaw_rad) >=
           config.min_containment);
  if (ordinary_physical_match) {
    return true;
  }
  return identity.eligible;
}

std::vector<int> hungarianMaximum(const std::vector<std::vector<float>>& score) {
  const std::size_t rows = score.size();
  const std::size_t columns = rows == 0 ? 0 : score.front().size();
  if (rows == 0 || columns == 0) {
    return {};
  }
  const auto solve_rows_not_greater_than_columns =
      [](const std::vector<std::vector<float>>& matrix) {
        const std::size_t n = matrix.size();
        const std::size_t m = matrix.front().size();
        std::vector<double> u(n + 1), v(m + 1);
        std::vector<std::size_t> p(m + 1), way(m + 1);
        for (std::size_t i = 1; i <= n; ++i) {
          p[0] = i;
          std::size_t j0 = 0;
          std::vector<double> min_value(
              m + 1, std::numeric_limits<double>::infinity());
          std::vector<bool> used(m + 1, false);
          do {
            used[j0] = true;
            const std::size_t i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            std::size_t j1 = 0;
            for (std::size_t j = 1; j <= m; ++j) {
              if (used[j]) {
                continue;
              }
              const double current =
                  -static_cast<double>(matrix[i0 - 1][j - 1]) - u[i0] - v[j];
              if (current < min_value[j]) {
                min_value[j] = current;
                way[j] = j0;
              }
              if (min_value[j] < delta) {
                delta = min_value[j];
                j1 = j;
              }
            }
            for (std::size_t j = 0; j <= m; ++j) {
              if (used[j]) {
                u[p[j]] += delta;
                v[j] -= delta;
              } else {
                min_value[j] -= delta;
              }
            }
            j0 = j1;
          } while (p[j0] != 0);
          do {
            const std::size_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
          } while (j0 != 0);
        }
        std::vector<int> result(n, -1);
        for (std::size_t j = 1; j <= m; ++j) {
          if (p[j] != 0) {
            result[p[j] - 1] = static_cast<int>(j - 1);
          }
        }
        return result;
      };

  if (rows <= columns) {
    return solve_rows_not_greater_than_columns(score);
  }
  std::vector<std::vector<float>> transposed(
      columns, std::vector<float>(rows, 0.0f));
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      transposed[column][row] = score[row][column];
    }
  }
  const std::vector<int> row_by_column =
      solve_rows_not_greater_than_columns(transposed);
  std::vector<int> column_by_row(rows, -1);
  for (std::size_t column = 0; column < row_by_column.size(); ++column) {
    if (row_by_column[column] >= 0) {
      column_by_row[static_cast<std::size_t>(row_by_column[column])] =
          static_cast<int>(column);
    }
  }
  return column_by_row;
}

}  // namespace

float box2dIou(const std::array<float, 4>& lhs,
               const std::array<float, 4>& rhs) {
  const float lhs_x0 = std::min(lhs[0], lhs[2]);
  const float lhs_y0 = std::min(lhs[1], lhs[3]);
  const float lhs_x1 = std::max(lhs[0], lhs[2]);
  const float lhs_y1 = std::max(lhs[1], lhs[3]);
  const float rhs_x0 = std::min(rhs[0], rhs[2]);
  const float rhs_y0 = std::min(rhs[1], rhs[3]);
  const float rhs_x1 = std::max(rhs[0], rhs[2]);
  const float rhs_y1 = std::max(rhs[1], rhs[3]);
  const float intersection = std::max(0.0f, std::min(lhs_x1, rhs_x1) -
                                                 std::max(lhs_x0, rhs_x0)) *
                             std::max(0.0f, std::min(lhs_y1, rhs_y1) -
                                                 std::max(lhs_y0, rhs_y0));
  const float lhs_area = std::max(0.0f, lhs_x1 - lhs_x0) *
                         std::max(0.0f, lhs_y1 - lhs_y0);
  const float rhs_area = std::max(0.0f, rhs_x1 - rhs_x0) *
                         std::max(0.0f, rhs_y1 - rhs_y0);
  const float union_area = lhs_area + rhs_area - intersection;
  return union_area > kEpsilon ? intersection / union_area : 0.0f;
}

float orientedBoxIou(const Eigen::Vector3f& lhs_center,
                     const Eigen::Vector3f& lhs_size,
                     float lhs_yaw,
                     const Eigen::Vector3f& rhs_center,
                     const Eigen::Vector3f& rhs_size,
                     float rhs_yaw) {
  const float overlap = intersectionVolume(lhs_center, lhs_size, lhs_yaw,
                                           rhs_center, rhs_size, rhs_yaw);
  const float union_volume = volume(lhs_size) + volume(rhs_size) - overlap;
  return union_volume > kEpsilon ? overlap / union_volume : 0.0f;
}

float orientedBoxContainment(const Eigen::Vector3f& lhs_center,
                             const Eigen::Vector3f& lhs_size,
                             float lhs_yaw,
                             const Eigen::Vector3f& rhs_center,
                             const Eigen::Vector3f& rhs_size,
                             float rhs_yaw) {
  const float overlap = intersectionVolume(lhs_center, lhs_size, lhs_yaw,
                                           rhs_center, rhs_size, rhs_yaw);
  const float smaller = std::min(volume(lhs_size), volume(rhs_size));
  return smaller > kEpsilon ? overlap / smaller : 0.0f;
}

SmallObjectIdentityConfig makeSmallObjectIdentityConfig(
    const std::vector<std::string>& group_specs,
    float max_volume_m3,
    float max_extent_m,
    float min_iou_3d,
    float max_normalized_center_distance,
    float min_volume_ratio) {
  SmallObjectIdentityConfig config;
  config.max_volume_m3 = std::max(0.0f, max_volume_m3);
  config.max_extent_m = std::max(0.0f, max_extent_m);
  config.min_iou_3d = clamp01(min_iou_3d);
  config.max_normalized_center_distance =
      std::max(0.0f, max_normalized_center_distance);
  config.min_volume_ratio = clamp01(min_volume_ratio);

  std::set<std::string> assigned_labels;
  for (const std::string& spec : group_specs) {
    std::vector<std::string> labels;
    std::set<std::string> unique_labels;
    std::size_t begin = 0;
    while (begin <= spec.size()) {
      const std::size_t separator = spec.find('|', begin);
      const std::string label = normalizeIdentityLabel(
          spec.substr(begin, separator == std::string::npos
                                 ? std::string::npos
                                 : separator - begin));
      if (label.empty()) {
        throw std::invalid_argument(
            "small-object identity group contains an empty label: '" + spec +
            "'");
      }
      if (!unique_labels.insert(label).second) {
        throw std::invalid_argument(
            "small-object identity group contains a duplicate normalized label '" +
            label + "': '" + spec + "'");
      }
      labels.push_back(label);
      if (separator == std::string::npos) {
        break;
      }
      begin = separator + 1;
    }
    if (labels.size() < 2U) {
      throw std::invalid_argument(
          "small-object identity group must contain at least two labels: '" +
          spec + "'");
    }
    const std::string& family = labels.front();
    for (const std::string& label : labels) {
      if (!assigned_labels.insert(label).second) {
        throw std::invalid_argument(
            "small-object identity label belongs to multiple groups: '" + label +
            "'");
      }
      config.family_by_label.emplace(label, family);
    }
  }
  return config;
}

SmallObjectIdentityFeatures evaluateSmallObjectIdentity(
    const std::string& lhs_label,
    const Eigen::Vector3f& lhs_center,
    const Eigen::Vector3f& lhs_size,
    float lhs_yaw,
    const std::string& rhs_label,
    const Eigen::Vector3f& rhs_center,
    const Eigen::Vector3f& rhs_size,
    float rhs_yaw,
    const SmallObjectIdentityConfig& config) {
  SmallObjectIdentityFeatures features;
  features.normalized_center_distance =
      std::numeric_limits<float>::infinity();
  if (!isSmallObjectSize(lhs_size, config) ||
      !isSmallObjectSize(rhs_size, config)) {
    return features;
  }
  const float lhs_volume = volume(lhs_size);
  const float rhs_volume = volume(rhs_size);

  const std::string lhs = normalizeIdentityLabel(lhs_label);
  const std::string rhs = normalizeIdentityLabel(rhs_label);
  if (lhs.empty() || rhs.empty()) {
    return features;
  }

  const auto lhs_family = config.family_by_label.find(lhs);
  const auto rhs_family = config.family_by_label.find(rhs);
  if (lhs == rhs) {
    features.label_compatible = true;
    features.family = lhs_family == config.family_by_label.end()
                          ? lhs
                          : lhs_family->second;
  } else if (lhs_family != config.family_by_label.end() &&
             rhs_family != config.family_by_label.end() &&
             lhs_family->second == rhs_family->second) {
    features.label_compatible = true;
    features.family = lhs_family->second;
  }
  features.identity_conflict =
      !features.label_compatible &&
      (lhs_family != config.family_by_label.end() ||
       rhs_family != config.family_by_label.end());
  if (!features.label_compatible) {
    return features;
  }
  features.volume_ratio =
      std::max(lhs_volume, rhs_volume) > kEpsilon
          ? std::min(lhs_volume, rhs_volume) /
                std::max(lhs_volume, rhs_volume)
          : 0.0f;
  const float diagonal = std::max(lhs_size.norm(), rhs_size.norm());
  features.normalized_center_distance =
      diagonal > kEpsilon ? (lhs_center - rhs_center).norm() / diagonal
                          : std::numeric_limits<float>::infinity();
  features.iou_3d = orientedBoxIou(lhs_center, lhs_size, lhs_yaw,
                                   rhs_center, rhs_size, rhs_yaw);
  features.eligible =
      features.iou_3d >= config.min_iou_3d &&
      features.normalized_center_distance <=
          config.max_normalized_center_distance &&
      features.volume_ratio >= config.min_volume_ratio;
  return features;
}

std::vector<InstanceObservation, Eigen::aligned_allocator<InstanceObservation>>
clusterPhysicalObservations(
    const std::vector<InstanceObservation,
                      Eigen::aligned_allocator<InstanceObservation>>& input,
    const PhysicalObservationConfig& config) {
  using Observations =
      std::vector<InstanceObservation,
                  Eigen::aligned_allocator<InstanceObservation>>;
  if (input.empty()) {
    return {};
  }
  std::vector<std::size_t> order(input.size());
  std::iota(order.begin(), order.end(), 0U);
  std::sort(order.begin(), order.end(), [&input](std::size_t lhs, std::size_t rhs) {
    const RawDetection& a = input[lhs].detection;
    const RawDetection& b = input[rhs].detection;
    return std::tie(a.center_world.x(), a.center_world.y(), a.center_world.z(),
                    a.label, a.semantic_id) <
           std::tie(b.center_world.x(), b.center_world.y(), b.center_world.z(),
                    b.label, b.semantic_id);
  });

  std::vector<std::vector<std::size_t>> clusters;
  for (std::size_t index : order) {
    auto found = std::find_if(
        clusters.begin(), clusters.end(), [&input, &config, index](const auto& cluster) {
          return std::all_of(cluster.begin(), cluster.end(),
                             [&input, &config, index](std::size_t member) {
                               return samePhysicalItem(input[index], input[member], config);
                             });
        });
    if (found == clusters.end()) {
      clusters.push_back({index});
    } else {
      found->push_back(index);
    }
  }

  Observations output;
  output.reserve(clusters.size());
  for (const auto& cluster : clusters) {
    const std::size_t representative = *std::max_element(
        cluster.begin(), cluster.end(), [&input](std::size_t lhs, std::size_t rhs) {
          return std::make_tuple(input[lhs].bbox_quality,
                                 input[lhs].confidence,
                                 input[lhs].detection.label) <
                 std::make_tuple(input[rhs].bbox_quality,
                                 input[rhs].confidence,
                                 input[rhs].detection.label);
        });
    InstanceObservation physical = input[representative];
    physical.member_count = cluster.size();
    physical.label_votes.clear();
    physical.semantic_votes.clear();
    physical.strong_identity_family.clear();
    for (std::size_t member : cluster) {
      const InstanceObservation& observation = input[member];
      const float vote = std::max(kEpsilon,
                                  observation.confidence * observation.bbox_quality);
      if (!observation.detection.label.empty()) {
        physical.label_votes[observation.detection.label] += vote;
      }
      if (observation.detection.semantic_id >= 0) {
        physical.semantic_votes[observation.detection.semantic_id] += vote;
      }
    }
    for (std::size_t i = 0; i < cluster.size() &&
                            physical.strong_identity_family.empty(); ++i) {
      for (std::size_t j = i + 1; j < cluster.size(); ++j) {
        const RawDetection& lhs = input[cluster[i]].detection;
        const RawDetection& rhs = input[cluster[j]].detection;
        const SmallObjectIdentityFeatures identity =
            evaluateSmallObjectIdentity(
                lhs.label, lhs.center_world, lhs.size_m, lhs.yaw_rad,
                rhs.label, rhs.center_world, rhs.size_m, rhs.yaw_rad,
                config.small_object_identity);
        if (identity.eligible) {
          physical.strong_identity_family = identity.family;
          break;
        }
      }
    }
    output.push_back(std::move(physical));
  }
  return output;
}

AssociationResult associatePhysicalObservations(
    const std::vector<InstanceObservation,
                      Eigen::aligned_allocator<InstanceObservation>>& observations,
    const std::vector<InstanceTrack, Eigen::aligned_allocator<InstanceTrack>>& tracks,
    TimeNanoseconds now_ns,
    const AssociationScoringConfig& config) {
  AssociationResult result;
  result.track_by_observation.resize(observations.size());
  result.pair_features.resize(
      observations.size(), std::vector<AssociationPairFeatures>(tracks.size()));
  std::vector<std::vector<float>> scores(
      observations.size(), std::vector<float>(tracks.size(), 0.0f));
  std::vector<unsigned char> small_observations(observations.size(), 0U);
  for (std::size_t i = 0; i < observations.size(); ++i) {
    small_observations[i] = isSmallObjectSize(
        observations[i].detection.size_m, config.small_object_identity);
  }
  std::vector<unsigned char> small_tracks(tracks.size(), 0U);
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    small_tracks[i] =
        isSmallObjectSize(tracks[i].size_m, config.small_object_identity);
  }
  for (std::size_t observation_index = 0;
       observation_index < observations.size(); ++observation_index) {
    const InstanceObservation& observation = observations[observation_index];
    const bool observation_is_small = small_observations[observation_index] != 0U;
    for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index) {
      const InstanceTrack& track = tracks[track_index];
      AssociationPairFeatures& features =
          result.pair_features[observation_index][track_index];
      bool identity_iou_available = false;
      float identity_iou = 0.0f;
      if (observation_is_small && small_tracks[track_index] != 0U) {
        const SmallObjectIdentityFeatures strong_identity =
            evaluateSmallObjectIdentity(
                track.label, track.center_world, track.size_m, track.yaw_rad,
                observation.detection.label,
                observation.detection.center_world,
                observation.detection.size_m,
                observation.detection.yaw_rad,
                config.small_object_identity);
        features.strong_identity = strong_identity.eligible;
        features.identity_conflict = strong_identity.identity_conflict;
        identity_iou_available = strong_identity.label_compatible;
        identity_iou = strong_identity.iou_3d;
      }
      features.size_ratio = sizeRatio(track.size_m, observation.detection.size_m);
      const float track_diagonal = track.size_m.norm();
      const float observation_diagonal = observation.detection.size_m.norm();
      const float diagonal = std::max(track_diagonal, observation_diagonal);
      const float center_gate = std::min(
          config.max_center_gate_m, std::max(0.15f, 0.75f * diagonal));
      const float distance =
          (track.center_world - observation.detection.center_world).norm();
      features.center_score =
          center_gate > kEpsilon ? clamp01(1.0f - distance / center_gate) : 0.0f;
      const float conservative_overlap_reach =
          0.5f * (track_diagonal + observation_diagonal);
      if (!track.publishable || features.identity_conflict ||
          (!features.strong_identity &&
           (features.size_ratio < config.min_size_ratio ||
            (distance > center_gate && distance > conservative_overlap_reach)))) {
        continue;
      }
      features.iou_3d = identity_iou_available
                            ? identity_iou
                            : orientedBoxIou(
                                  track.center_world, track.size_m,
                                  track.yaw_rad,
                                  observation.detection.center_world,
                                  observation.detection.size_m,
                                  observation.detection.yaw_rad);
      features.containment = orientedBoxContainment(
          track.center_world, track.size_m, track.yaw_rad,
          observation.detection.center_world, observation.detection.size_m,
          observation.detection.yaw_rad);
      features.semantic_similarity = semanticCosine(track, observation);
      if (track.appearance_model_id ==
          observation.detection.appearance_model_id) {
        features.appearance_similarity_shadow = vectorCosine(
            track.appearance_descriptor_shadow,
            observation.detection.appearance_descriptor);
      }
      const float age_seconds =
          now_ns > track.last_seen_ns
              ? static_cast<float>(now_ns - track.last_seen_ns) * 1.0e-9f
              : 0.0f;
      features.recency_score = clamp01(1.0f - age_seconds / 2.0f);
      features.candidate = features.strong_identity ||
                           distance <= center_gate || features.iou_3d >= 0.05f;
      if (!features.candidate) {
        continue;
      }
      // Tracks do not retain a camera-specific 2D box.  Renormalize over the
      // five factors that are available for every persisted object.
      const float weight_sum = config.overlap_weight + config.center_weight +
                               config.size_weight + config.semantic_weight +
                               config.recency_weight;
      features.score =
          (config.overlap_weight *
               std::max(features.iou_3d, 0.7f * features.containment) +
           config.center_weight * features.center_score +
           config.size_weight * features.size_ratio +
           config.semantic_weight * features.semantic_similarity +
           config.recency_weight * features.recency_score) /
          std::max(kEpsilon, weight_sum);
      const float identity_weight_sum = config.overlap_weight +
                                        config.center_weight +
                                        config.size_weight;
      features.identity_score =
          (config.overlap_weight *
               std::max(features.iou_3d, features.containment) +
           config.center_weight * features.center_score +
           config.size_weight * features.size_ratio) /
          std::max(kEpsilon, identity_weight_sum);
      const float ordinary_score =
          track.state == InstanceTrackState::kInactive
              ? features.identity_score
              : features.score;
      // Strong identity candidates must win assignment over ordinary semantic
      // candidates; geometry still orders multiple strong candidates.
      scores[observation_index][track_index] =
          features.strong_identity ? 2.0f + features.identity_score
                                   : ordinary_score;
    }
  }

  const std::vector<int> assignment = hungarianMaximum(scores);
  for (std::size_t observation_index = 0;
       observation_index < assignment.size(); ++observation_index) {
    if (assignment[observation_index] < 0) {
      continue;
    }
    const std::size_t track_index =
        static_cast<std::size_t>(assignment[observation_index]);
    const InstanceTrack& track = tracks[track_index];
    const float threshold = track.state == InstanceTrackState::kInactive
                                ? config.archived_match_threshold
                                : config.active_match_threshold;
    const AssociationPairFeatures& features =
        result.pair_features[observation_index][track_index];
    const float decision_score = track.state == InstanceTrackState::kInactive
                                     ? features.identity_score
                                     : features.score;
    if (features.candidate &&
        (features.strong_identity || decision_score >= threshold)) {
      result.track_by_observation[observation_index] = track_index;
    }
  }
  return result;
}

}  // namespace roomie
