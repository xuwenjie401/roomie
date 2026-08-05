#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "roomie/pipeline/types.hpp"

namespace roomie {

using WorldPointVector =
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;

struct MapSurfacePoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3f position_world = Eigen::Vector3f::Zero();
  std::uint8_t r = 160;
  std::uint8_t g = 160;
  std::uint8_t b = 160;
  float intensity = 0.0f;
  float weight = 0.0f;
  VoxelRef voxel_ref;
  bool has_voxel_ref = false;
};

using MapSurfacePointVector =
    std::vector<MapSurfacePoint, Eigen::aligned_allocator<MapSurfacePoint>>;

struct BlockIndex {
  int x = 0;
  int y = 0;
  int z = 0;

  BlockIndex() = default;
  BlockIndex(int x_value, int y_value, int z_value)
      : x(x_value), y(y_value), z(z_value) {}
  explicit BlockIndex(const Eigen::Vector3i& index)
      : x(index.x()), y(index.y()), z(index.z()) {}

  Eigen::Vector3i eigen() const { return Eigen::Vector3i(x, y, z); }
};

inline bool operator==(const BlockIndex& lhs, const BlockIndex& rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

inline bool operator!=(const BlockIndex& lhs, const BlockIndex& rhs) {
  return !(lhs == rhs);
}

inline bool operator<(const BlockIndex& lhs, const BlockIndex& rhs) {
  if (lhs.x != rhs.x) {
    return lhs.x < rhs.x;
  }
  if (lhs.y != rhs.y) {
    return lhs.y < rhs.y;
  }
  return lhs.z < rhs.z;
}

struct BlockIndexHash {
  std::size_t operator()(const BlockIndex& index) const noexcept {
    std::size_t seed = std::hash<int>{}(index.x);
    seed ^= std::hash<int>{}(index.y) + 0x9e3779b9U + (seed << 6U) +
            (seed >> 2U);
    seed ^= std::hash<int>{}(index.z) + 0x9e3779b9U + (seed << 6U) +
            (seed >> 2U);
    return seed;
  }
};

struct SurfaceAabb {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3f min = Eigen::Vector3f::Constant(
      std::numeric_limits<float>::infinity());
  Eigen::Vector3f max = Eigen::Vector3f::Constant(
      -std::numeric_limits<float>::infinity());

  SurfaceAabb() = default;
  SurfaceAabb(Eigen::Vector3f min_value, Eigen::Vector3f max_value)
      : min(std::move(min_value)), max(std::move(max_value)) {}

  bool valid() const {
    return min.allFinite() && max.allFinite() &&
           (min.array() <= max.array()).all();
  }

  bool contains(const Eigen::Vector3f& point) const {
    return valid() && point.allFinite() &&
           (point.array() >= min.array()).all() &&
           (point.array() <= max.array()).all();
  }

  bool intersects(const SurfaceAabb& other) const {
    return valid() && other.valid() &&
           (min.array() <= other.max.array()).all() &&
           (max.array() >= other.min.array()).all();
  }

  void extend(const SurfaceAabb& other) {
    if (!other.valid()) {
      return;
    }
    if (!valid()) {
      *this = other;
      return;
    }
    min = min.cwiseMin(other.min);
    max = max.cwiseMax(other.max);
  }
};

class SurfaceBlock {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SurfaceBlock(BlockIndex index,
               SurfaceAabb aabb,
               MapSurfacePointVector points)
      : index_(index), aabb_(std::move(aabb)), points_(std::move(points)) {
    if (!aabb_.valid()) {
      throw std::invalid_argument("SurfaceBlock requires a valid AABB");
    }
  }

  const BlockIndex& index() const { return index_; }
  const SurfaceAabb& aabb() const { return aabb_; }
  const MapSurfacePointVector& points() const { return points_; }
  std::size_t pointCount() const { return points_.size(); }
  bool empty() const { return points_.empty(); }

 private:
  const BlockIndex index_;
  const SurfaceAabb aabb_;
  const MapSurfacePointVector points_;
};

using SurfaceBlockPtr = std::shared_ptr<const SurfaceBlock>;

inline SurfaceBlockPtr makeSurfaceBlock(BlockIndex index,
                                        SurfaceAabb aabb,
                                        MapSurfacePointVector points) {
  return std::make_shared<const SurfaceBlock>(
      index, std::move(aabb), std::move(points));
}

struct SurfaceSnapshotStats {
  std::size_t block_count = 0;
  std::size_t point_count = 0;
  std::size_t dirty_block_count = 0;
  std::size_t reused_block_count = 0;
  std::size_t added_block_count = 0;
  std::size_t replaced_block_count = 0;
  std::size_t removed_block_count = 0;
  bool has_bounds = false;
  SurfaceAabb bounds;
};

class SurfaceSnapshot;
using SurfaceSnapshotPtr = std::shared_ptr<const SurfaceSnapshot>;

class SurfaceSnapshotBuilder;

class SurfaceSnapshot {
 public:
  using BlockMap =
      std::unordered_map<BlockIndex, SurfaceBlockPtr, BlockIndexHash>;

  const MapStamp& mapStamp() const { return map_stamp_; }
  const SurfaceStamp& surfaceStamp() const { return surface_stamp_; }
  const SurfaceSnapshotStats& stats() const { return stats_; }
  const BlockMap& blocks() const { return blocks_; }

  bool empty() const { return blocks_.empty(); }
  std::size_t blockCount() const { return stats_.block_count; }
  std::size_t pointCount() const { return stats_.point_count; }

  SurfaceBlockPtr block(const BlockIndex& index) const {
    const auto it = blocks_.find(index);
    return it == blocks_.end() ? nullptr : it->second;
  }

  // Returns a deterministic, read-only block view ordered by BlockIndex.
  std::vector<SurfaceBlockPtr> blockView() const {
    std::vector<SurfaceBlockPtr> result;
    result.reserve(blocks_.size());
    for (const auto& entry : blocks_) {
      result.push_back(entry.second);
    }
    std::sort(result.begin(),
              result.end(),
              [](const SurfaceBlockPtr& lhs, const SurfaceBlockPtr& rhs) {
                return lhs->index() < rhs->index();
              });
    return result;
  }

  MapSurfacePointVector flatten() const {
    MapSurfacePointVector result;
    result.reserve(stats_.point_count);
    for (const SurfaceBlockPtr& block_ptr : blockView()) {
      result.insert(result.end(),
                    block_ptr->points().begin(),
                    block_ptr->points().end());
    }
    return result;
  }

  WorldPointVector flattenWorldPoints() const {
    WorldPointVector result;
    result.reserve(stats_.point_count);
    for (const SurfaceBlockPtr& block_ptr : blockView()) {
      for (const MapSurfacePoint& point : block_ptr->points()) {
        result.push_back(point.position_world);
      }
    }
    return result;
  }

  std::vector<SurfaceBlockPtr> blocksIntersecting(
      const SurfaceAabb& query) const {
    std::vector<SurfaceBlockPtr> result;
    if (!query.valid()) {
      return result;
    }
    for (const auto& entry : blocks_) {
      if (entry.second->aabb().intersects(query)) {
        result.push_back(entry.second);
      }
    }
    std::sort(result.begin(),
              result.end(),
              [](const SurfaceBlockPtr& lhs, const SurfaceBlockPtr& rhs) {
                return lhs->index() < rhs->index();
              });
    return result;
  }

  // The block AABB is used for coarse selection and the point itself is then
  // tested against the query, so callers do not receive neighboring points
  // merely because their blocks touch the requested region.
  MapSurfacePointVector pointsInAabb(const SurfaceAabb& query) const {
    MapSurfacePointVector result;
    for (const SurfaceBlockPtr& block_ptr : blocksIntersecting(query)) {
      for (const MapSurfacePoint& point : block_ptr->points()) {
        if (query.contains(point.position_world)) {
          result.push_back(point);
        }
      }
    }
    return result;
  }

 private:
  friend class SurfaceSnapshotBuilder;

  SurfaceSnapshot(MapStamp map_stamp,
                  SurfaceStamp surface_stamp,
                  BlockMap blocks,
                  SurfaceSnapshotStats stats)
      : map_stamp_(std::move(map_stamp)),
        surface_stamp_(std::move(surface_stamp)),
        blocks_(std::move(blocks)),
        stats_(std::move(stats)) {}

  const MapStamp map_stamp_;
  const SurfaceStamp surface_stamp_;
  const BlockMap blocks_;
  const SurfaceSnapshotStats stats_;
};

struct MapDelta {
  MapStamp from_map;
  MapStamp to_map;
  SurfaceStamp from_surface;
  SurfaceStamp to_surface;
  std::vector<BlockIndex> changed_blocks;
  std::vector<BlockIndex> removed_blocks;
  bool complete = true;
  bool full_rebuild = false;

  bool valid() const {
    const RunId& epoch = to_surface.map_epoch;
    return from_map.map_epoch == epoch && to_map.map_epoch == epoch &&
           from_surface.map_epoch == epoch &&
           from_surface.surface_revision < to_surface.surface_revision &&
           from_map.map_revision <= to_map.map_revision &&
           from_surface.source_map_revision == from_map.map_revision &&
           to_surface.source_map_revision == to_map.map_revision;
  }

  std::vector<BlockIndex> touchedBlocks() const {
    std::vector<BlockIndex> result = changed_blocks;
    result.insert(result.end(), removed_blocks.begin(), removed_blocks.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }
};

struct MapCommit {
  RunId run_id;
  FrameId frame_id = 0;
  bool success = true;
  std::string error;
  MapStamp map;
  SurfaceStamp surface;
  SurfaceSnapshotPtr snapshot;
  MapDelta delta;
  bool includes_current_frame = false;
  bool perception_candidate = false;
  std::chrono::steady_clock::time_point due_time =
      std::chrono::steady_clock::time_point::max();
  // Process-local publication timestamp used by the asynchronous perception
  // join. It is intentionally never serialized.
  std::chrono::steady_clock::time_point published_at =
      std::chrono::steady_clock::time_point::min();
  FrameBundlePtr frame_bundle;
};

struct SurfaceSnapshotBuildResult {
  SurfaceSnapshotPtr snapshot;
  MapDelta delta;
};

class SurfaceSnapshotBuilder {
 public:
  SurfaceSnapshotBuilder(SurfaceSnapshotPtr base,
                         MapStamp map_stamp,
                         SurfaceStamp surface_stamp)
      : base_(std::move(base)),
        map_stamp_(std::move(map_stamp)),
        surface_stamp_(std::move(surface_stamp)) {
    validateStamps();
  }

  void setBlock(SurfaceBlockPtr block) {
    if (!block) {
      throw std::invalid_argument("setBlock requires a non-null block");
    }
    // Capture metadata before moving the shared_ptr. In an assignment the
    // right-hand side may be evaluated before operator[]'s key expression.
    const BlockIndex index = block->index();
    const bool empty = block->empty();
    if (empty) {
      removeBlock(index);
      return;
    }
    updates_[index] = std::move(block);
  }

  void removeBlock(const BlockIndex& index) {
    // Delta removals describe transitions from an existing surface block to
    // absence. A backend may legitimately inspect a dirty TSDF block that has
    // never yielded surface points; retaining that no-op as a removal makes
    // geometry scheduling conservatively rescan every object.
    if (base_ && base_->block(index)) {
      updates_[index] = nullptr;
    } else {
      // Also collapse set-then-remove on a newly added block to its net no-op.
      updates_.erase(index);
    }
  }

  SurfaceSnapshotBuildResult build() const {
    SurfaceSnapshot::BlockMap blocks;
    if (base_) {
      blocks = base_->blocks();
    }

    for (const auto& update : updates_) {
      if (update.second && !update.second->empty()) {
        blocks[update.first] = update.second;
      } else {
        blocks.erase(update.first);
      }
    }

    SurfaceSnapshotStats stats;
    stats.block_count = blocks.size();
    stats.dirty_block_count = updates_.size();
    for (const auto& entry : blocks) {
      stats.point_count += entry.second->pointCount();
      if (!stats.has_bounds) {
        stats.bounds = entry.second->aabb();
        stats.has_bounds = true;
      } else {
        stats.bounds.extend(entry.second->aabb());
      }

      const SurfaceBlockPtr old_block =
          base_ ? base_->block(entry.first) : nullptr;
      if (!old_block) {
        ++stats.added_block_count;
      } else if (old_block == entry.second) {
        ++stats.reused_block_count;
      } else {
        ++stats.replaced_block_count;
      }
    }
    if (base_) {
      for (const auto& old_entry : base_->blocks()) {
        if (blocks.find(old_entry.first) == blocks.end()) {
          ++stats.removed_block_count;
        }
      }
    }

    SurfaceSnapshotPtr snapshot(new SurfaceSnapshot(
        map_stamp_, surface_stamp_, std::move(blocks), stats));

    MapDelta delta;
    if (base_) {
      delta.from_map = base_->mapStamp();
      delta.from_surface = base_->surfaceStamp();
    } else {
      delta.from_map.map_epoch = map_stamp_.map_epoch;
      delta.from_surface.map_epoch = surface_stamp_.map_epoch;
    }
    delta.to_map = map_stamp_;
    delta.to_surface = surface_stamp_;
    delta.full_rebuild = !base_;
    for (const auto& update : updates_) {
      if (update.second && !update.second->empty()) {
        delta.changed_blocks.push_back(update.first);
      } else {
        delta.removed_blocks.push_back(update.first);
      }
    }
    std::sort(delta.changed_blocks.begin(), delta.changed_blocks.end());
    std::sort(delta.removed_blocks.begin(), delta.removed_blocks.end());

    return SurfaceSnapshotBuildResult{std::move(snapshot), std::move(delta)};
  }

  static SurfaceSnapshotBuildResult buildFrom(
      SurfaceSnapshotPtr base,
      MapStamp map_stamp,
      SurfaceStamp surface_stamp,
      const std::vector<SurfaceBlockPtr>& changed_blocks,
      const std::vector<BlockIndex>& removed_blocks) {
    SurfaceSnapshotBuilder builder(
        std::move(base), std::move(map_stamp), std::move(surface_stamp));
    for (const SurfaceBlockPtr& block : changed_blocks) {
      builder.setBlock(block);
    }
    // Explicit removals win over entries also present in changed_blocks.
    for (const BlockIndex& index : removed_blocks) {
      builder.removeBlock(index);
    }
    return builder.build();
  }

 private:
  void validateStamps() const {
    if (map_stamp_.map_epoch != surface_stamp_.map_epoch) {
      throw std::invalid_argument(
          "map and surface stamps must use the same epoch");
    }
    if (surface_stamp_.source_map_revision != map_stamp_.map_revision) {
      throw std::invalid_argument(
          "surface source revision must equal the committed map revision");
    }
    if (surface_stamp_.surface_revision == 0) {
      throw std::invalid_argument("surface revision zero is not publishable");
    }
    if (!base_) {
      return;
    }
    if (base_->mapStamp().map_epoch != map_stamp_.map_epoch ||
        base_->surfaceStamp().map_epoch != surface_stamp_.map_epoch) {
      throw std::invalid_argument(
          "surface blocks cannot be reused across map epochs");
    }
    if (map_stamp_.map_revision < base_->mapStamp().map_revision ||
        surface_stamp_.surface_revision <=
            base_->surfaceStamp().surface_revision) {
      throw std::invalid_argument(
          "map/surface revisions must advance monotonically");
    }
  }

  SurfaceSnapshotPtr base_;
  MapStamp map_stamp_;
  SurfaceStamp surface_stamp_;
  std::unordered_map<BlockIndex, SurfaceBlockPtr, BlockIndexHash> updates_;
};

enum class MapDeltaQueryStatus {
  kNoIntersection,
  kIntersection,
  kUnknown,
};

enum class MapDeltaUnknownReason {
  kNone,
  kEpochMismatch,
  kInvalidRange,
  kJournalEmpty,
  kRevisionGap,
  kIncompleteDelta,
  kFullRebuild,
};

struct MapDeltaQueryResult {
  MapDeltaQueryStatus status = MapDeltaQueryStatus::kUnknown;
  MapDeltaUnknownReason unknown_reason =
      MapDeltaUnknownReason::kJournalEmpty;
  std::size_t checked_deltas = 0;
  std::uint64_t covered_through_revision = 0;

  // Geometry CAS should reject both a definite intersection and unknown
  // coverage. Only kNoIntersection proves that accepting an older result is
  // safe for the requested local dependency blocks.
  bool conservativelyAffected() const {
    return status != MapDeltaQueryStatus::kNoIntersection;
  }
};

class MapDeltaJournal {
 public:
  explicit MapDeltaJournal(std::size_t capacity) : capacity_(capacity) {}

  MapDeltaJournal(const MapDeltaJournal&) = delete;
  MapDeltaJournal& operator=(const MapDeltaJournal&) = delete;

  bool append(MapDelta delta) {
    if (!delta.valid()) {
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (capacity_ == 0) {
      return true;
    }

    if (!entries_.empty() &&
        entries_.back().to_surface.map_epoch !=
            delta.to_surface.map_epoch) {
      entries_.clear();
    }

    // An overlap or out-of-order append makes the old chain ambiguous. Drop
    // it so subsequent queries fail conservatively instead of guessing which
    // delta supersedes another. Forward gaps are retained and detected by
    // query().
    if (!entries_.empty() &&
        delta.from_surface.surface_revision <
            entries_.back().to_surface.surface_revision) {
      entries_.clear();
    }

    entries_.push_back(std::move(delta));
    while (entries_.size() > capacity_) {
      entries_.pop_front();
    }
    return true;
  }

  MapDeltaQueryResult query(
      const SurfaceStamp& from,
      const SurfaceStamp& to,
      const std::vector<BlockIndex>& dependency_blocks) const {
    if (from.map_epoch != to.map_epoch) {
      return unknownResult(MapDeltaUnknownReason::kEpochMismatch,
                           from.surface_revision);
    }
    if (from.surface_revision > to.surface_revision ||
        (from.surface_revision == to.surface_revision &&
         from.source_map_revision != to.source_map_revision)) {
      return unknownResult(MapDeltaUnknownReason::kInvalidRange,
                           from.surface_revision);
    }
    if (from.surface_revision == to.surface_revision) {
      MapDeltaQueryResult result;
      result.status = MapDeltaQueryStatus::kNoIntersection;
      result.unknown_reason = MapDeltaUnknownReason::kNone;
      result.covered_through_revision = to.surface_revision;
      return result;
    }

    std::unordered_set<BlockIndex, BlockIndexHash> dependencies(
        dependency_blocks.begin(), dependency_blocks.end());
    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.empty()) {
      return unknownResult(MapDeltaUnknownReason::kJournalEmpty,
                           from.surface_revision);
    }

    std::uint64_t cursor = from.surface_revision;
    bool intersects = false;
    std::size_t checked = 0;
    for (const MapDelta& delta : entries_) {
      if (delta.to_surface.map_epoch != to.map_epoch) {
        continue;
      }
      if (delta.to_surface.surface_revision <= cursor) {
        continue;
      }
      if (delta.from_surface.surface_revision > cursor) {
        MapDeltaQueryResult result = unknownResult(
            MapDeltaUnknownReason::kRevisionGap, cursor);
        result.checked_deltas = checked;
        return result;
      }

      ++checked;
      if (!delta.complete) {
        MapDeltaQueryResult result = unknownResult(
            MapDeltaUnknownReason::kIncompleteDelta, cursor);
        result.checked_deltas = checked;
        return result;
      }
      if (delta.full_rebuild) {
        MapDeltaQueryResult result = unknownResult(
            MapDeltaUnknownReason::kFullRebuild, cursor);
        result.checked_deltas = checked;
        return result;
      }

      for (const BlockIndex& index : delta.changed_blocks) {
        intersects = intersects || dependencies.count(index) != 0;
      }
      for (const BlockIndex& index : delta.removed_blocks) {
        intersects = intersects || dependencies.count(index) != 0;
      }

      cursor = std::max(cursor, delta.to_surface.surface_revision);
      if (cursor >= to.surface_revision) {
        break;
      }
    }

    if (cursor < to.surface_revision) {
      MapDeltaQueryResult result = unknownResult(
          MapDeltaUnknownReason::kRevisionGap, cursor);
      result.checked_deltas = checked;
      return result;
    }

    MapDeltaQueryResult result;
    result.status = intersects ? MapDeltaQueryStatus::kIntersection
                               : MapDeltaQueryStatus::kNoIntersection;
    result.unknown_reason = MapDeltaUnknownReason::kNone;
    result.checked_deltas = checked;
    result.covered_through_revision = cursor;
    return result;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
  }

  std::size_t capacity() const { return capacity_; }

 private:
  static MapDeltaQueryResult unknownResult(
      MapDeltaUnknownReason reason, std::uint64_t covered_through) {
    MapDeltaQueryResult result;
    result.status = MapDeltaQueryStatus::kUnknown;
    result.unknown_reason = reason;
    result.covered_through_revision = covered_through;
    return result;
  }

  const std::size_t capacity_;
  mutable std::mutex mutex_;
  std::deque<MapDelta> entries_;
};

}  // namespace roomie
