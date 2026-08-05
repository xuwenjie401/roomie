#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#include "roomie/pipeline/types.hpp"

namespace roomie {

enum class SnapshotControlKind : std::uint8_t {
  kPromote = 0,
  kMerge = 1,
  kDropTentative = 2,
  kEraseObject = 3,
};

// Ownership controls must preserve their relative order, but repeated updates
// for the same logical owner can safely collapse to the newest scene revision.
struct SnapshotControl {
  SnapshotControlKind kind = SnapshotControlKind::kDropTentative;
  int first_id = -1;
  int second_id = -1;
  TimeNanoseconds now_ns = 0;
  SceneRevision scene_revision = 0;
};

enum class SnapshotControlPushOutcome : std::uint8_t {
  kQueued = 0,
  kCoalesced = 1,
  kRejectedCapacity = 2,
  kRejectedNoWorker = 3,
  kRejectedClosed = 4,
};

struct SnapshotControlQueueStats {
  std::size_t capacity = 0;
  std::size_t depth = 0;
  std::size_t high_watermark = 0;
  std::uint64_t queued = 0;
  std::uint64_t coalesced = 0;
  std::uint64_t rejected_capacity = 0;
  std::uint64_t rejected_closed = 0;
  std::uint64_t dequeued = 0;
  std::uint64_t abandoned_on_stop = 0;
};

// Small actor hand-off queue used only for SnapshotBank ownership controls.
// It rejects a distinct control at capacity instead of silently evicting an
// older ownership transition. The rejection counter and caller-side log make
// overload explicit. Exact logical keys coalesce to the newest revision.
class SnapshotControlQueue {
 public:
  explicit SnapshotControlQueue(std::size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
      throw std::invalid_argument(
          "snapshot control queue capacity must be positive");
    }
    stats_.capacity = capacity_;
  }

  SnapshotControlPushOutcome push(SnapshotControl control) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      ++stats_.rejected_closed;
      return SnapshotControlPushOutcome::kRejectedClosed;
    }
    const auto matching = std::find_if(
        queue_.begin(), queue_.end(), [&control](const SnapshotControl& item) {
          return item.kind == control.kind &&
                 item.first_id == control.first_id;
        });
    if (matching != queue_.end()) {
      // A non-zero revision is authoritative. Do not let a delayed older
      // notification overwrite a newer ownership transition.
      if (matching->scene_revision == 0 ||
          (control.scene_revision != 0 &&
           control.scene_revision >= matching->scene_revision)) {
        *matching = std::move(control);
      }
      ++stats_.coalesced;
      return SnapshotControlPushOutcome::kCoalesced;
    }
    if (queue_.size() >= capacity_) {
      ++stats_.rejected_capacity;
      return SnapshotControlPushOutcome::kRejectedCapacity;
    }
    queue_.push_back(std::move(control));
    ++stats_.queued;
    stats_.high_watermark = std::max(stats_.high_watermark, queue_.size());
    return SnapshotControlPushOutcome::kQueued;
  }

  std::optional<SnapshotControl> front() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return std::nullopt;
    }
    return queue_.front();
  }

  bool popFront() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    queue_.pop_front();
    ++stats_.dequeued;
    return true;
  }

  std::size_t abandonAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t abandoned = queue_.size();
    queue_.clear();
    stats_.abandoned_on_stop += abandoned;
    return abandoned;
  }

  std::size_t closeAndAbandon() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    const std::size_t abandoned = queue_.size();
    queue_.clear();
    stats_.abandoned_on_stop += abandoned;
    return abandoned;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  SnapshotControlQueueStats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    SnapshotControlQueueStats result = stats_;
    result.depth = queue_.size();
    return result;
  }

 private:
  const std::size_t capacity_;
  mutable std::mutex mutex_;
  std::deque<SnapshotControl> queue_;
  SnapshotControlQueueStats stats_;
  bool closed_ = false;
};

}  // namespace roomie
