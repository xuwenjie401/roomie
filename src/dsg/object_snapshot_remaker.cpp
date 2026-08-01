#include "roomie/dsg/object_snapshot_remaker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace roomie {
namespace {

constexpr float kEpsilon = 1.0e-6f;

void setError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

float clamp01(float value) {
  return std::clamp(value, 0.0f, 1.0f);
}

float boxArea(const std::array<float, 4>& box) {
  const float x0 = std::min(box[0], box[2]);
  const float x1 = std::max(box[0], box[2]);
  const float y0 = std::min(box[1], box[3]);
  const float y1 = std::max(box[1], box[3]);
  if (!std::isfinite(x0) || !std::isfinite(x1) ||
      !std::isfinite(y0) || !std::isfinite(y1) || x1 <= x0 || y1 <= y0) {
    return 0.0f;
  }
  return (x1 - x0) * (y1 - y0);
}

float positionScore(const std::array<float, 4>& box, int image_size) {
  if (image_size <= 0) {
    return 0.0f;
  }
  const float x0 = std::min(box[0], box[2]);
  const float x1 = std::max(box[0], box[2]);
  const float y0 = std::min(box[1], box[3]);
  const float y1 = std::max(box[1], box[3]);
  const float px =
      std::clamp(0.5f * (x0 + x1) / static_cast<float>(image_size), 1.0e-4f, 0.9999f);
  const float py =
      std::clamp(0.5f * (y0 + y1) / static_cast<float>(image_size), 1.0e-4f, 0.9999f);
  const float entropy_x = -px * std::log(px) - (1.0f - px) * std::log(1.0f - px);
  const float entropy_y = -py * std::log(py) - (1.0f - py) * std::log(1.0f - py);
  return clamp01((entropy_x + entropy_y) / std::log(4.0f));
}

bool writeBmpRgb(const ImageBuffer& image,
                 const std::filesystem::path& path,
                 std::string* error) {
  if (image.empty() || image.channels != 3) {
    setError(error, "snapshot image must be non-empty rgb8");
    return false;
  }
  try {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    const int width = image.width;
    const int height = image.height;
    const int row_stride = ((width * 3 + 3) / 4) * 4;
    const std::uint32_t pixel_bytes =
        static_cast<std::uint32_t>(row_stride * height);
    const std::uint32_t file_bytes = 54U + pixel_bytes;

    std::ofstream stream(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!stream) {
      setError(error, "failed to open snapshot image: " + path.string());
      return false;
    }

    auto write_u16 = [&stream](std::uint16_t value) {
      stream.put(static_cast<char>(value & 0xffU));
      stream.put(static_cast<char>((value >> 8U) & 0xffU));
    };
    auto write_u32 = [&stream](std::uint32_t value) {
      stream.put(static_cast<char>(value & 0xffU));
      stream.put(static_cast<char>((value >> 8U) & 0xffU));
      stream.put(static_cast<char>((value >> 16U) & 0xffU));
      stream.put(static_cast<char>((value >> 24U) & 0xffU));
    };

    stream.put('B');
    stream.put('M');
    write_u32(file_bytes);
    write_u16(0);
    write_u16(0);
    write_u32(54);
    write_u32(40);
    write_u32(static_cast<std::uint32_t>(width));
    write_u32(static_cast<std::uint32_t>(height));
    write_u16(1);
    write_u16(24);
    write_u32(0);
    write_u32(pixel_bytes);
    write_u32(0);
    write_u32(0);
    write_u32(0);
    write_u32(0);

    std::vector<std::uint8_t> padding(static_cast<std::size_t>(row_stride - width * 3), 0);
    for (int y = height - 1; y >= 0; --y) {
      const std::size_t row_offset =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(width) * 3U;
      for (int x = 0; x < width; ++x) {
        const std::size_t offset = row_offset + static_cast<std::size_t>(x) * 3U;
        stream.put(static_cast<char>(image.data[offset + 2]));
        stream.put(static_cast<char>(image.data[offset + 1]));
        stream.put(static_cast<char>(image.data[offset]));
      }
      if (!padding.empty()) {
        stream.write(reinterpret_cast<const char*>(padding.data()),
                     static_cast<std::streamsize>(padding.size()));
      }
    }
    return static_cast<bool>(stream);
  } catch (const std::exception& ex) {
    setError(error, ex.what());
    return false;
  }
}

std::string snapshotFilename(int index) {
  std::ostringstream stream;
  stream << "snapshot_" << std::setw(4) << std::setfill('0') << index << ".bmp";
  return stream.str();
}

std::string joinUri(const std::string& prefix, const std::string& filename) {
  if (prefix.empty()) {
    return filename;
  }
  if (prefix.back() == '/') {
    return prefix + filename;
  }
  return prefix + "/" + filename;
}

}  // namespace

ObjectSnapshotRemaker::ObjectSnapshotRemaker(ObjectSnapshotRemakerConfig config)
    : config_(std::move(config)) {}

void ObjectSnapshotRemaker::loadSnapshot(const ObjectGraphSnapshot& snapshot) {
  std::lock_guard<std::mutex> lock(mutex_);
  frames_by_temp_index_.clear();
  best_by_object_id_.clear();
  next_temp_image_index_ = 100000000;
  for (const ObjectNode& object : snapshot.objects) {
    if (object.object_id < 0) {
      continue;
    }
    best_by_object_id_[object.object_id] = StoredBest{};
  }
}

ObjectSnapshotRemakeFrameResult ObjectSnapshotRemaker::submitFrame(
    const ImageBuffer& image,
    TimeNanoseconds time_ns,
    const std::string& camera_id,
    const std::vector<ObjectSnapshotRemakeCandidate>& candidates,
    std::string* error) {
  ObjectSnapshotRemakeFrameResult result;
  result.candidate_count = candidates.size();
  if (!config_.enabled || candidates.empty()) {
    return result;
  }
  if (image.empty()) {
    setError(error, "snapshot remake frame has no source image");
    return result;
  }

  std::vector<std::pair<ObjectSnapshotRemakeCandidate, float>> improvements;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<int, std::size_t> improvement_by_object;
    for (const ObjectSnapshotRemakeCandidate& candidate : candidates) {
      if (candidate.object_id < 0) {
        continue;
      }
      const float quality = candidateQuality(candidate, image);
      const auto best_it = best_by_object_id_.find(candidate.object_id);
      const bool has_previous =
          best_it != best_by_object_id_.end() && best_it->second.has_snapshot;
      const float min_quality =
          has_previous ? config_.min_quality : config_.first_min_quality;
      if (quality < min_quality) {
        ++result.quality_rejected_candidates;
        continue;
      }
      if (has_previous && !shouldReplaceLocked(candidate.object_id, quality)) {
        ++result.replace_rejected_candidates;
        continue;
      }
      const auto existing = improvement_by_object.find(candidate.object_id);
      if (existing == improvement_by_object.end()) {
        improvement_by_object[candidate.object_id] = improvements.size();
        improvements.emplace_back(candidate, quality);
      } else if (quality > improvements[existing->second].second) {
        improvements[existing->second] = std::make_pair(candidate, quality);
      }
    }
  }

  if (improvements.empty()) {
    return result;
  }

  const int temp_index = [&]() {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_temp_image_index_++;
  }();
  const std::filesystem::path staging_path = stagingPathForIndex(temp_index);
  if (!writeBmpRgb(image, staging_path, error)) {
    return result;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  StoredFrame frame;
  frame.temp_image_index = temp_index;
  frame.path = staging_path;
  frame.width = image.width;
  frame.height = image.height;
  frame.time_ns = time_ns;
  frame.camera_id = camera_id;
  frames_by_temp_index_[temp_index] = frame;

  for (const auto& [candidate, quality] : improvements) {
    StoredBest& best = best_by_object_id_[candidate.object_id];
    const bool has_previous = best.has_snapshot;
    const float min_quality =
        has_previous ? config_.min_quality : config_.first_min_quality;
    if (quality < min_quality) {
      ++result.quality_rejected_candidates;
      continue;
    }
    if (has_previous && !shouldReplaceLocked(candidate.object_id, quality)) {
      ++result.replace_rejected_candidates;
      continue;
    }
    best.has_snapshot = true;
    best.remade = true;
    best.temp_image_index = temp_index;
    best.bbox_xyxy = candidate.bbox_xyxy;
    best.quality = quality;
    best.time_ns = candidate.time_ns;
    best.camera_id = candidate.camera_id;
    ++result.improved_objects;
    if (has_previous) {
      ++result.replaced_objects;
    } else {
      ++result.first_snapshot_objects;
    }
  }
  result.saved_frame = result.improved_objects > 0;
  return result;
}

bool ObjectSnapshotRemaker::populateSnapshot(ObjectGraphSnapshot* snapshot,
                                             const std::filesystem::path& image_dir,
                                             const std::string& image_uri_prefix,
                                             std::string* error) const {
  if (snapshot == nullptr) {
    setError(error, "null ObjectGraphSnapshot output");
    return false;
  }
  if (!config_.enabled) {
    return true;
  }

  std::unordered_map<int, StoredBest> best_by_object;
  std::unordered_map<int, StoredFrame> frames;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    best_by_object = best_by_object_id_;
    frames = frames_by_temp_index_;
  }

  std::unordered_set<int> referenced_temp_indices;
  for (const ObjectNode& object : snapshot->objects) {
    const auto best_it = best_by_object.find(object.object_id);
    if (best_it != best_by_object.end() && best_it->second.remade) {
      referenced_temp_indices.insert(best_it->second.temp_image_index);
    }
  }

  try {
    if (!referenced_temp_indices.empty()) {
      std::filesystem::create_directories(image_dir);
    }
  } catch (const std::exception& ex) {
    setError(error, ex.what());
    return false;
  }

  snapshot->snapshot_images.clear();
  std::unordered_map<int, int> final_index_by_temp_index;
  int next_final_image_index = 0;
  for (int temp_index : referenced_temp_indices) {
    const auto frame_it = frames.find(temp_index);
    if (frame_it == frames.end()) {
      continue;
    }
    const StoredFrame& frame = frame_it->second;
    const int final_index = next_final_image_index++;
    const std::string filename = snapshotFilename(final_index);
    const std::filesystem::path output_path = image_dir / filename;
    try {
      std::filesystem::copy_file(frame.path,
                                 output_path,
                                 std::filesystem::copy_options::overwrite_existing);
    } catch (const std::exception& ex) {
      setError(error, ex.what());
      return false;
    }

    ObjectSnapshotImage image;
    image.image_index = final_index;
    image.uri = joinUri(image_uri_prefix, filename);
    image.width = frame.width;
    image.height = frame.height;
    image.encoding = "bmp";
    image.time_ns = frame.time_ns;
    image.camera_id = frame.camera_id;
    image.source_path = output_path.string();
    snapshot->snapshot_images.push_back(std::move(image));
    final_index_by_temp_index[temp_index] = final_index;
  }

  for (ObjectNode& object : snapshot->objects) {
    object.snapshot = ObjectSnapshotRef{};
    const auto best_it = best_by_object.find(object.object_id);
    if (best_it == best_by_object.end() || !best_it->second.remade) {
      continue;
    }
    const auto final_it = final_index_by_temp_index.find(best_it->second.temp_image_index);
    if (final_it == final_index_by_temp_index.end()) {
      continue;
    }
    object.snapshot.image_index = final_it->second;
    object.snapshot.bbox_xyxy = best_it->second.bbox_xyxy;
    object.snapshot.quality = best_it->second.quality;
    object.snapshot.time_ns = best_it->second.time_ns;
    object.snapshot.camera_id = best_it->second.camera_id;
  }
  return true;
}

bool ObjectSnapshotRemaker::shouldReplaceLocked(int object_id, float quality) const {
  const auto best_it = best_by_object_id_.find(object_id);
  if (best_it == best_by_object_id_.end() || !best_it->second.has_snapshot) {
    return true;
  }
  const float previous = best_it->second.quality;
  return quality >= previous + config_.replace_min_quality_delta &&
         quality >= previous * config_.replace_min_quality_ratio;
}

float ObjectSnapshotRemaker::candidateQuality(
    const ObjectSnapshotRemakeCandidate& candidate,
    const ImageBuffer& image) const {
  const float area = boxArea(candidate.bbox_xyxy);
  if (area < config_.min_box_area_px) {
    return 0.0f;
  }
  const float min_area = std::max(kEpsilon, config_.min_box_area_px);
  const float size_score = clamp01(std::tanh(area / (min_area * 10.0f)));
  const float position_score =
      positionScore(candidate.bbox_xyxy, std::min(image.width, image.height));
  const float weight_sum = config_.position_weight + config_.size_weight;
  const float image_reward =
      weight_sum > kEpsilon
          ? (config_.position_weight * position_score +
             config_.size_weight * size_score) /
                weight_sum
          : 1.0f;
  return clamp01(candidate.bbox_quality * image_reward);
}

std::filesystem::path ObjectSnapshotRemaker::stagingPathForIndex(
    int temp_image_index) const {
  std::ostringstream filename;
  filename << "snapshot_candidate_" << temp_image_index << ".bmp";
  return config_.staging_dir / filename.str();
}

}  // namespace roomie
