#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "roomie/artifacts/snapshot_bank.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"
#include "roomie/scene/scene_command.hpp"
#include "roomie/scene/scene_snapshot.hpp"

namespace roomie {

enum class SnapshotOwnerKind : std::uint8_t {
  kTentativeTrack = 0,
  kObject = 1,
};

struct SnapshotOwner {
  SnapshotOwnerKind kind = SnapshotOwnerKind::kTentativeTrack;
  int id = -1;

  bool valid() const { return id >= 0; }
};

inline bool operator==(const SnapshotOwner& lhs, const SnapshotOwner& rhs) {
  return lhs.kind == rhs.kind && lhs.id == rhs.id;
}

inline bool operator!=(const SnapshotOwner& lhs, const SnapshotOwner& rhs) {
  return !(lhs == rhs);
}

// SnapshotCandidate already owns its full frame through shared_ptr. Producers
// should copy that pointer into every ROI from the same FrameBundle; the
// worker never clones the full image.
struct OnlineSnapshotCandidate {
  SnapshotOwner owner;
  SnapshotCandidate candidate;
};

struct OnlineSnapshotWorkerConfig {
  // Candidate ingress is bounded and non-blocking. Control events share the
  // same ordered channel but use reliable delivery.
  std::size_t event_queue_capacity = 64;
  std::chrono::milliseconds poll_period{20};
  std::chrono::milliseconds asset_gc_period{60'000};
  // A failed bank mutation or reducer publication stays at the head of the
  // ordered stream and is retried with bounded exponential backoff. The
  // retry count is bounded so a permanently bad event cannot block all later
  // ownership controls forever.
  std::size_t max_retry_attempts = 8;
  std::chrono::milliseconds retry_initial_backoff{10};
  std::chrono::milliseconds retry_max_backoff{500};
};

enum class SnapshotCommandDisposition : std::uint8_t {
  kAccepted = 0,
  kStale = 1,
  kRejected = 2,
};

struct SnapshotCommandSubmitResult {
  SnapshotCommandDisposition disposition =
      SnapshotCommandDisposition::kRejected;
  std::string reason;

  bool accepted() const {
    return disposition == SnapshotCommandDisposition::kAccepted;
  }
};

struct SnapshotWorkerEnqueueResult {
  PushOutcome outcome = PushOutcome::kRejected;
  ChannelReplacementReason replacement_reason =
      ChannelReplacementReason::kNone;
  std::optional<SnapshotOwner> replaced_candidate;
  std::string reason;

  bool accepted() const {
    return outcome == PushOutcome::kAccepted ||
           outcome == PushOutcome::kReplaced;
  }
};

struct OnlineSnapshotWorkerStats {
  std::uint64_t candidate_enqueued = 0;
  std::uint64_t candidate_replaced_same_owner = 0;
  std::uint64_t candidate_dropped_for_capacity = 0;
  std::uint64_t candidate_rejected = 0;
  std::uint64_t candidate_stopped = 0;
  std::uint64_t candidate_processed = 0;
  std::uint64_t candidate_bank_accepted = 0;
  std::uint64_t candidate_bank_rejected = 0;
  std::uint64_t control_enqueued = 0;
  std::uint64_t control_processed = 0;
  std::uint64_t control_failed = 0;
  std::uint64_t commands_submitted = 0;
  std::uint64_t commands_accepted = 0;
  std::uint64_t command_stale_retries = 0;
  std::uint64_t commands_rejected = 0;
  std::uint64_t retry_attempts = 0;
  std::uint64_t retry_exhausted = 0;
  std::uint64_t retry_abandoned_on_stop = 0;
  std::uint64_t events_processed = 0;
  std::uint64_t asset_gc_runs = 0;
  std::uint64_t assets_collected = 0;
  std::uint64_t asset_bytes_collected = 0;
  std::uint64_t asset_gc_failures = 0;
};

class OnlineSnapshotWorker final : public WorkerThread {
 public:
  using SnapshotSupplier = std::function<SceneSnapshot()>;
  using CommandSink = std::function<SnapshotCommandSubmitResult(
      const ApplySnapshotSetCommand&)>;
  // This is the only compatibility seam for ObjectSnapshotRef. The default
  // factory fills the legacy fields; schema-v3 integration can additionally
  // attach durable asset/evidence identifiers without changing scheduling.
  using SnapshotReferenceFactory = std::function<ObjectSnapshotRef(
      const SnapshotRecord&, std::size_t ordinal)>;

  OnlineSnapshotWorker(OnlineSnapshotWorkerConfig config,
                       std::shared_ptr<SnapshotBank> snapshot_bank,
                       SnapshotSupplier snapshot_supplier,
                       CommandSink command_sink,
                       SnapshotReferenceFactory reference_factory = {});
  ~OnlineSnapshotWorker() override;

  SnapshotWorkerEnqueueResult enqueueCandidate(
      OnlineSnapshotCandidate candidate, TimeNanoseconds now_ns = 0);

  SnapshotWorkerEnqueueResult enqueuePromotion(
      int tentative_track_id, SceneObjectId object_id,
      TimeNanoseconds now_ns = 0);
  SnapshotWorkerEnqueueResult enqueueMerge(
      SceneObjectId retired_object_id, SceneObjectId canonical_object_id,
      TimeNanoseconds now_ns = 0);
  SnapshotWorkerEnqueueResult enqueueDropTentative(
      int tentative_track_id, TimeNanoseconds now_ns = 0);
  SnapshotWorkerEnqueueResult enqueueEraseObject(
      SceneObjectId object_id, TimeNanoseconds now_ns = 0);

  // Actor-to-actor integration uses these zero-wait variants and retains a
  // failed control locally for ordered retry. This avoids circular waits when
  // the snapshot worker is synchronously awaiting a reducer CAS result.
  SnapshotWorkerEnqueueResult tryEnqueuePromotion(
      int tentative_track_id, SceneObjectId object_id,
      TimeNanoseconds now_ns = 0);
  SnapshotWorkerEnqueueResult tryEnqueueMerge(
      SceneObjectId retired_object_id, SceneObjectId canonical_object_id,
      TimeNanoseconds now_ns = 0);
  SnapshotWorkerEnqueueResult tryEnqueueDropTentative(
      int tentative_track_id, TimeNanoseconds now_ns = 0);
  SnapshotWorkerEnqueueResult tryEnqueueEraseObject(
      SceneObjectId object_id, TimeNanoseconds now_ns = 0);

  bool idle() const;
  bool waitUntilIdle(std::chrono::milliseconds timeout) const;
  OnlineSnapshotWorkerStats stats() const;
  ChannelStats queueStats() const;
  std::string lastError() const;

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace roomie
