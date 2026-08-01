#pragma once

#include <array>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "roomie/dsg/object_graph.hpp"
#include "roomie/pipeline/types.hpp"

namespace roomie {

struct ObjectSnapshotRemakerConfig {
  bool enabled = false;
  std::filesystem::path staging_dir = "/tmp/roomie_object_snapshots";
  float first_min_quality = 0.05f;
  float min_quality = 0.18f;
  float min_box_area_px = 300.0f;
  float position_weight = 0.5f;
  float size_weight = 0.5f;
  float replace_min_quality_delta = 0.12f;
  float replace_min_quality_ratio = 1.20f;
};

struct ObjectSnapshotRemakeCandidate {
  int object_id = -1;
  std::array<float, 4> bbox_xyxy = {0.0f, 0.0f, 0.0f, 0.0f};
  float bbox_quality = 0.0f;
  TimeNanoseconds time_ns = 0;
  std::string camera_id;
};

struct ObjectSnapshotRemakeFrameResult {
  bool saved_frame = false;
  std::size_t candidate_count = 0;
  std::size_t improved_objects = 0;
  std::size_t first_snapshot_objects = 0;
  std::size_t replaced_objects = 0;
  std::size_t quality_rejected_candidates = 0;
  std::size_t replace_rejected_candidates = 0;
};

class ObjectSnapshotRemaker {
 public:
  explicit ObjectSnapshotRemaker(ObjectSnapshotRemakerConfig config = {});

  void loadSnapshot(const ObjectGraphSnapshot& snapshot);
  ObjectSnapshotRemakeFrameResult submitFrame(
      const ImageBuffer& image,
      TimeNanoseconds time_ns,
      const std::string& camera_id,
      const std::vector<ObjectSnapshotRemakeCandidate>& candidates,
      std::string* error = nullptr);
  bool populateSnapshot(ObjectGraphSnapshot* snapshot,
                        const std::filesystem::path& image_dir,
                        const std::string& image_uri_prefix,
                        std::string* error = nullptr) const;

 private:
  struct StoredFrame {
    int temp_image_index = -1;
    std::filesystem::path path;
    int width = 0;
    int height = 0;
    TimeNanoseconds time_ns = 0;
    std::string camera_id;
  };

  struct StoredBest {
    bool has_snapshot = false;
    bool remade = false;
    int temp_image_index = -1;
    std::array<float, 4> bbox_xyxy = {0.0f, 0.0f, 0.0f, 0.0f};
    float quality = 0.0f;
    TimeNanoseconds time_ns = 0;
    std::string camera_id;
  };

  bool shouldReplaceLocked(int object_id, float quality) const;
  float candidateQuality(const ObjectSnapshotRemakeCandidate& candidate,
                         const ImageBuffer& image) const;
  std::filesystem::path stagingPathForIndex(int temp_image_index) const;

  ObjectSnapshotRemakerConfig config_;
  mutable std::mutex mutex_;
  int next_temp_image_index_ = 100000000;
  std::unordered_map<int, StoredFrame> frames_by_temp_index_;
  std::unordered_map<int, StoredBest> best_by_object_id_;
};

}  // namespace roomie
