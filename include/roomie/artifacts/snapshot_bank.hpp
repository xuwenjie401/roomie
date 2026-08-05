#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "roomie/dsg/object_graph.hpp"
#include "roomie/pipeline/types.hpp"

namespace roomie {

using AssetId = std::string;

struct AssetStoreConfig {
  std::filesystem::path root;
  TimeNanoseconds grace_period_ns = 5LL * 60LL * 1000LL * 1000LL * 1000LL;
};

struct AssetRecord {
  AssetId id;
  std::filesystem::path path;
  int width = 0;
  int height = 0;
  int channels = 0;
  std::string source_encoding;
  std::uint64_t encoded_bytes = 0;
  std::uint64_t ref_count = 0;
  TimeNanoseconds unreferenced_since_ns = 0;
};

struct AssetWriteResult {
  bool success = false;
  bool created = false;
  AssetRecord asset;
  std::string error;
};

struct AssetGcResult {
  std::size_t removed_assets = 0;
  std::uint64_t removed_bytes = 0;
  std::vector<AssetId> removed_ids;
  std::string error;
};

// Stores lossless, content-addressed full frames. Reference counts and asset
// metadata are persisted in an atomically replaced manifest. A successfully
// returned materializeFrame() result is durable but initially unreferenced;
// callers retain it only after committing an owning snapshot record.
class AssetStore {
 public:
  explicit AssetStore(AssetStoreConfig config);
  ~AssetStore();

  AssetStore(const AssetStore&) = delete;
  AssetStore& operator=(const AssetStore&) = delete;

  bool healthy() const;
  std::string initializationError() const;

  AssetWriteResult materializeFrame(const ImageBuffer& frame,
                                    TimeNanoseconds now_ns = 0);

  // Applies all reference changes as one manifest transaction. Negative
  // counts are rejected before any state is changed.
  bool applyReferenceDelta(const std::map<AssetId, std::int64_t>& delta,
                           TimeNanoseconds now_ns,
                           std::string* error = nullptr);
  // Startup reconciliation makes the durable SceneStore snapshot the source
  // of truth after a crash. Assets omitted from desired_counts become
  // unreferenced (and remain protected by the configured GC grace period).
  bool reconcileReferenceCounts(
      const std::map<AssetId, std::uint64_t>& desired_counts,
      TimeNanoseconds now_ns = 0,
      std::string* error = nullptr);
  bool retain(const AssetId& id, std::string* error = nullptr);
  bool release(const AssetId& id,
               TimeNanoseconds now_ns = 0,
               std::string* error = nullptr);

  std::optional<AssetRecord> record(const AssetId& id) const;
  std::optional<std::filesystem::path> pathFor(const AssetId& id) const;
  std::optional<ImageBuffer> readFrame(const AssetId& id,
                                       std::string* error = nullptr) const;
  bool validate(const AssetId& id, std::string* error = nullptr) const;
  std::size_t assetCount() const;
  std::uint64_t totalReferencedCount() const;

  AssetGcResult collectGarbage(TimeNanoseconds now_ns = 0);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

enum class SnapshotMaskSource : std::uint8_t {
  kInstanceMask = 0,
  kBboxFallback = 1,
};

const char* snapshotMaskSourceName(SnapshotMaskSource source);

// Computes the same order-independent v1 set hash used by SnapshotBank from
// serialized scene references. Returns empty when any reference lacks an
// effective evidence hash.
std::string snapshotSetHashForReferences(
    const std::vector<ObjectSnapshotRef>& references);

// All fields are normalized quality scores in [0, 1]. A value of one is
// neutral, so producers may progressively populate newly available evidence.
struct SnapshotQualityComponents {
  float confidence = 1.0f;
  float edge_completeness = 1.0f;
  float distance = 1.0f;
  float position = 1.0f;
  float size = 1.0f;
  float blur = 1.0f;
  float exposure = 1.0f;
  float truncation = 1.0f;

  float score() const;
};

struct SnapshotViewpoint {
  float azimuth_rad = 0.0f;
  float elevation_rad = 0.0f;
  // Projected object area divided by full-frame area. Must be positive.
  float scale = 1.0f;
};

// Describes how a consumer reconstructs the effective crop from the retained
// full frame. Zero source width/height means "derive from bbox".
struct SnapshotCropTransform {
  float source_x = 0.0f;
  float source_y = 0.0f;
  float source_width = 0.0f;
  float source_height = 0.0f;
  float output_scale_x = 1.0f;
  float output_scale_y = 1.0f;
};

struct SnapshotCandidate {
  std::shared_ptr<const ImageBuffer> full_frame;
  std::array<float, 4> bbox_xyxy = {0.0f, 0.0f, 0.0f, 0.0f};
  SnapshotCropTransform crop;
  SnapshotMaskSource mask_source = SnapshotMaskSource::kBboxFallback;
  // Stable content hash or durable asset id for an instance mask. Empty is
  // valid only for bbox_fallback.
  std::string mask_ref;
  SnapshotQualityComponents quality;
  SnapshotViewpoint viewpoint;
  TimeNanoseconds time_ns = 0;
  std::string camera_id;
  FrameProvenance provenance;
  // Recorded as shadow evidence only; it does not affect admission or score.
  std::optional<float> patch_depth_occlusion_shadow;
};

struct SnapshotViewBucket {
  int azimuth = 0;
  int elevation = 0;
  int scale = 0;
};

inline bool operator==(const SnapshotViewBucket& lhs,
                       const SnapshotViewBucket& rhs) {
  return lhs.azimuth == rhs.azimuth && lhs.elevation == rhs.elevation &&
         lhs.scale == rhs.scale;
}

inline bool operator!=(const SnapshotViewBucket& lhs,
                       const SnapshotViewBucket& rhs) {
  return !(lhs == rhs);
}

inline bool operator<(const SnapshotViewBucket& lhs,
                      const SnapshotViewBucket& rhs) {
  if (lhs.azimuth != rhs.azimuth) {
    return lhs.azimuth < rhs.azimuth;
  }
  if (lhs.elevation != rhs.elevation) {
    return lhs.elevation < rhs.elevation;
  }
  return lhs.scale < rhs.scale;
}

struct SnapshotRecord {
  AssetId source_frame_asset_id;
  // Hash of the effective visual evidence: frame + bbox + crop + mask.
  std::string evidence_hash;
  std::array<float, 4> bbox_xyxy = {0.0f, 0.0f, 0.0f, 0.0f};
  SnapshotCropTransform crop;
  SnapshotMaskSource mask_source = SnapshotMaskSource::kBboxFallback;
  std::string mask_ref;
  SnapshotQualityComponents quality;
  float quality_score = 0.0f;
  SnapshotViewpoint viewpoint;
  SnapshotViewBucket view_bucket;
  TimeNanoseconds time_ns = 0;
  std::string camera_id;
  FrameProvenance provenance;
  std::optional<float> patch_depth_occlusion_shadow;
};

struct SnapshotSet {
  // Ordered by quality, then evidence hash. The first entry is primary.
  std::vector<SnapshotRecord> records;
  // Order-independent hash of effective pixel evidence.
  std::string snapshot_set_hash;
  std::uint64_t appearance_revision = 0;

  const SnapshotRecord* primary() const {
    return records.empty() ? nullptr : &records.front();
  }
};

struct SnapshotBankConfig {
  std::size_t top_k = 3;
  float minimum_quality = 0.0f;
  float azimuth_bucket_degrees = 45.0f;
  float elevation_bucket_degrees = 30.0f;
  // Multiplicative scale buckets; sqrt(2) gives two buckets per octave.
  float scale_bucket_ratio = 1.41421356f;
  float replacement_min_quality_delta = 0.05f;
  float replacement_min_quality_ratio = 1.05f;
  // A novel bucket may replace a duplicate-view incumbent at this fraction
  // of its quality. This is the explicit quality/diversity tradeoff.
  float diversity_min_quality_ratio = 0.65f;
};

struct SnapshotSubmitResult {
  bool accepted = false;
  bool effective_evidence_changed = false;
  std::uint64_t appearance_revision = 0;
  std::string snapshot_set_hash;
  std::string reason;
  std::string error;
};

struct SnapshotMergeResult {
  bool success = false;
  bool effective_evidence_changed = false;
  int canonical_object_id = -1;
  std::uint64_t appearance_revision = 0;
  std::string snapshot_set_hash;
  std::string error;
};

// Online Top-K evidence owner. Tentative tracks and stable objects use the
// same selection implementation; promotion transfers ownership, while merge
// resolves aliases and reranks the union without duplicating asset refs.
class SnapshotBank {
 public:
  SnapshotBank(SnapshotBankConfig config,
               std::shared_ptr<AssetStore> asset_store);
  ~SnapshotBank();

  SnapshotBank(const SnapshotBank&) = delete;
  SnapshotBank& operator=(const SnapshotBank&) = delete;

  SnapshotSubmitResult submitForTentativeTrack(
      int track_id,
      const SnapshotCandidate& candidate,
      TimeNanoseconds now_ns = 0);
  SnapshotSubmitResult submitForObject(int object_id,
                                       const SnapshotCandidate& candidate,
                                       TimeNanoseconds now_ns = 0);

  SnapshotMergeResult promoteTentativeTrack(int track_id,
                                            int object_id,
                                            TimeNanoseconds now_ns = 0);
  SnapshotMergeResult mergeObjects(int retired_object_id,
                                   int canonical_object_id,
                                   TimeNanoseconds now_ns = 0);

  bool dropTentativeTrack(int track_id,
                          TimeNanoseconds now_ns = 0,
                          std::string* error = nullptr);
  bool eraseObject(int object_id,
                   TimeNanoseconds now_ns = 0,
                   std::string* error = nullptr);

  // Hydrates the in-memory owner/alias tables from an already-durable scene.
  // These calls intentionally do not retain assets: follow them with one
  // AssetStore::reconcileReferenceCounts() transaction for the full scene.
  bool restoreObjectSnapshots(int object_id,
                              SnapshotSet snapshots,
                              std::string* error = nullptr);
  bool restoreObjectAlias(int retired_object_id,
                          int canonical_object_id,
                          std::string* error = nullptr);

  std::optional<SnapshotSet> tentativeSnapshots(int track_id) const;
  std::optional<SnapshotSet> objectSnapshots(int object_id) const;
  std::optional<int> resolveCanonicalObjectId(int object_id) const;
  std::size_t objectCount() const;
  std::size_t tentativeTrackCount() const;
  // Runtime GC stays behind the same owner so Pipeline code never needs a
  // second mutable handle to AssetStore.
  AssetGcResult collectGarbage(TimeNanoseconds now_ns = 0) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace roomie
