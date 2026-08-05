#include "roomie/artifacts/online_snapshot_worker.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace roomie {
namespace {

struct CandidateWork {
  OnlineSnapshotCandidate input;
  TimeNanoseconds now_ns = 0;
  std::uint64_t ingress_sequence = 0;
};

struct PromotionWork {
  int tentative_track_id = -1;
  SceneObjectId object_id = -1;
  TimeNanoseconds now_ns = 0;
  // A reliable promotion absorbs the most recent admitted tentative
  // candidate. Re-submitting identical evidence is idempotent, and this copy
  // prevents later capacity pressure from dropping the promotion keyframe.
  std::optional<SnapshotCandidate> absorbed_candidate;
  std::uint64_t absorbed_candidate_sequence = 0;
};

struct MergeWork {
  SceneObjectId retired_object_id = -1;
  SceneObjectId canonical_object_id = -1;
  TimeNanoseconds now_ns = 0;
};

struct DropTentativeWork {
  int tentative_track_id = -1;
  TimeNanoseconds now_ns = 0;
};

struct EraseObjectWork {
  SceneObjectId object_id = -1;
  TimeNanoseconds now_ns = 0;
};

using WorkPayload =
    std::variant<CandidateWork, PromotionWork, MergeWork, DropTentativeWork,
                 EraseObjectWork>;

struct WorkEvent {
  // Candidates may coalesce only inside one control-event epoch. This is the
  // fence that guarantees candidate(track) followed by promotion(track) is
  // processed in that order and cannot later be replaced across promotion.
  std::uint64_t coalescing_epoch = 0;
  WorkPayload payload;
};

const CandidateWork* candidateWork(const WorkEvent& event) {
  return std::get_if<CandidateWork>(&event.payload);
}

bool sameCandidateKey(const WorkEvent& lhs, const WorkEvent& rhs) {
  const CandidateWork* lhs_candidate = candidateWork(lhs);
  const CandidateWork* rhs_candidate = candidateWork(rhs);
  return lhs_candidate != nullptr && rhs_candidate != nullptr &&
         lhs.coalescing_epoch == rhs.coalescing_epoch &&
         lhs_candidate->input.owner == rhs_candidate->input.owner;
}

std::uint32_t stableReferenceIndex(const SnapshotRecord& record,
                                   std::size_t ordinal) {
  // FNV-1a is only a legacy image_index bridge. The content/evidence ids stay
  // authoritative in SnapshotBank and can be copied by a schema-v3 factory.
  std::uint32_t hash = 2166136261U;
  const std::string& key = record.source_frame_asset_id.empty()
                               ? record.evidence_hash
                               : record.source_frame_asset_id;
  for (const unsigned char character : key) {
    hash ^= static_cast<std::uint32_t>(character);
    hash *= 16777619U;
  }
  if (key.empty()) {
    hash ^= static_cast<std::uint32_t>(
        std::min<std::size_t>(ordinal,
                              std::numeric_limits<std::uint32_t>::max()));
  }
  hash &= 0x7fffffffU;
  return hash == 0 ? 1U : hash;
}

ObjectSnapshotRef defaultReference(const SnapshotRecord& record,
                                   std::size_t ordinal) {
  ObjectSnapshotRef reference;
  reference.image_index =
      static_cast<int>(stableReferenceIndex(record, ordinal));
  reference.bbox_xyxy = record.bbox_xyxy;
  reference.quality = record.quality_score;
  reference.time_ns = record.time_ns;
  reference.camera_id = record.camera_id;
  return reference;
}

std::string pushFailureReason(PushOutcome outcome) {
  switch (outcome) {
    case PushOutcome::kAccepted:
    case PushOutcome::kReplaced:
      return {};
    case PushOutcome::kRejected:
      return "snapshot event queue rejected ingress";
    case PushOutcome::kTimedOut:
      return "snapshot control enqueue timed out";
    case PushOutcome::kStopped:
      return "snapshot worker is stopping";
  }
  return "unknown snapshot enqueue outcome";
}

enum class WorkDisposition : std::uint8_t {
  kComplete = 0,
  kRetryableFailure = 1,
  kPermanentFailure = 2,
};

}  // namespace

struct OnlineSnapshotWorker::Impl {
  struct AtomicStats {
    std::atomic<std::uint64_t> candidate_enqueued{0};
    std::atomic<std::uint64_t> candidate_replaced_same_owner{0};
    std::atomic<std::uint64_t> candidate_dropped_for_capacity{0};
    std::atomic<std::uint64_t> candidate_rejected{0};
    std::atomic<std::uint64_t> candidate_stopped{0};
    std::atomic<std::uint64_t> candidate_processed{0};
    std::atomic<std::uint64_t> candidate_bank_accepted{0};
    std::atomic<std::uint64_t> candidate_bank_rejected{0};
    std::atomic<std::uint64_t> control_enqueued{0};
    std::atomic<std::uint64_t> control_processed{0};
    std::atomic<std::uint64_t> control_failed{0};
    std::atomic<std::uint64_t> commands_submitted{0};
    std::atomic<std::uint64_t> commands_accepted{0};
    std::atomic<std::uint64_t> command_stale_retries{0};
    std::atomic<std::uint64_t> commands_rejected{0};
    std::atomic<std::uint64_t> retry_attempts{0};
    std::atomic<std::uint64_t> retry_exhausted{0};
    std::atomic<std::uint64_t> retry_abandoned_on_stop{0};
    std::atomic<std::uint64_t> events_processed{0};
    std::atomic<std::uint64_t> asset_gc_runs{0};
    std::atomic<std::uint64_t> assets_collected{0};
    std::atomic<std::uint64_t> asset_bytes_collected{0};
    std::atomic<std::uint64_t> asset_gc_failures{0};

    OnlineSnapshotWorkerStats snapshot() const {
      OnlineSnapshotWorkerStats result;
      result.candidate_enqueued = candidate_enqueued.load();
      result.candidate_replaced_same_owner =
          candidate_replaced_same_owner.load();
      result.candidate_dropped_for_capacity =
          candidate_dropped_for_capacity.load();
      result.candidate_rejected = candidate_rejected.load();
      result.candidate_stopped = candidate_stopped.load();
      result.candidate_processed = candidate_processed.load();
      result.candidate_bank_accepted = candidate_bank_accepted.load();
      result.candidate_bank_rejected = candidate_bank_rejected.load();
      result.control_enqueued = control_enqueued.load();
      result.control_processed = control_processed.load();
      result.control_failed = control_failed.load();
      result.commands_submitted = commands_submitted.load();
      result.commands_accepted = commands_accepted.load();
      result.command_stale_retries = command_stale_retries.load();
      result.commands_rejected = commands_rejected.load();
      result.retry_attempts = retry_attempts.load();
      result.retry_exhausted = retry_exhausted.load();
      result.retry_abandoned_on_stop = retry_abandoned_on_stop.load();
      result.events_processed = events_processed.load();
      result.asset_gc_runs = asset_gc_runs.load();
      result.assets_collected = assets_collected.load();
      result.asset_bytes_collected = asset_bytes_collected.load();
      result.asset_gc_failures = asset_gc_failures.load();
      return result;
    }
  };

  Impl(OnlineSnapshotWorkerConfig input_config,
       std::shared_ptr<SnapshotBank> input_snapshot_bank,
       SnapshotSupplier input_snapshot_supplier,
       CommandSink input_command_sink,
       SnapshotReferenceFactory input_reference_factory)
      : config(std::move(input_config)),
        snapshot_bank(std::move(input_snapshot_bank)),
        snapshot_supplier(std::move(input_snapshot_supplier)),
        command_sink(std::move(input_command_sink)),
        reference_factory(input_reference_factory
                              ? std::move(input_reference_factory)
                              : SnapshotReferenceFactory(defaultReference)),
        events(config.event_queue_capacity, ChannelPolicy::kLatestByKey,
               sameCandidateKey) {}

  void reserveOutstanding() {
    outstanding.fetch_add(1, std::memory_order_acq_rel);
  }

  void releaseOutstanding() {
    const std::uint64_t previous =
        outstanding.fetch_sub(1, std::memory_order_acq_rel);
    if (previous == 1) {
      std::lock_guard<std::mutex> lock(idle_mutex);
      idle_cv.notify_all();
    }
  }

  void recordError(std::string error) {
    std::lock_guard<std::mutex> lock(error_mutex);
    last_error = std::move(error);
  }

  std::string error() const {
    std::lock_guard<std::mutex> lock(error_mutex);
    return last_error;
  }

  SnapshotWorkerEnqueueResult enqueueReliableControl(WorkPayload payload) {
    SnapshotWorkerEnqueueResult result;
    std::lock_guard<std::mutex> enqueue_lock(control_enqueue_mutex);
    WorkEvent event;
    event.coalescing_epoch =
        coalescing_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    event.payload = std::move(payload);
    reserveOutstanding();
    PushResult<WorkEvent> pushed = events.pushUsingPolicy(
        std::move(event), ChannelPolicy::kReliableBlocking);
    result.outcome = pushed.outcome;
    result.replacement_reason = pushed.replacement_reason;
    result.reason = pushFailureReason(pushed.outcome);
    if (pushed.accepted()) {
      stats.control_enqueued.fetch_add(1);
    } else {
      releaseOutstanding();
    }
    return result;
  }

  SnapshotWorkerEnqueueResult tryEnqueueReliableControl(WorkPayload payload) {
    SnapshotWorkerEnqueueResult result;
    std::unique_lock<std::mutex> enqueue_lock(control_enqueue_mutex,
                                               std::try_to_lock);
    if (!enqueue_lock.owns_lock()) {
      result.outcome = PushOutcome::kTimedOut;
      result.reason = "control enqueue is busy";
      return result;
    }
    WorkEvent event;
    event.coalescing_epoch =
        coalescing_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    event.payload = std::move(payload);
    reserveOutstanding();
    PushResult<WorkEvent> pushed = events.pushUsingPolicyFor(
        std::move(event), ChannelPolicy::kReliableBlocking,
        std::chrono::milliseconds::zero());
    result.outcome = pushed.outcome;
    result.replacement_reason = pushed.replacement_reason;
    result.reason = pushFailureReason(pushed.outcome);
    if (pushed.accepted()) {
      stats.control_enqueued.fetch_add(1);
    } else {
      releaseOutstanding();
    }
    return result;
  }

  WorkDisposition submitCurrentSet(SceneObjectId requested_object_id) {
    SceneObjectId retry_object_id = requested_object_id;
    for (int attempt = 0; attempt < 2; ++attempt) {
      const SceneSnapshot pinned = snapshot_supplier();
      const std::optional<SceneObjectId> canonical =
          pinned.resolveCanonicalId(retry_object_id);
      if (!canonical) {
        recordError("snapshot command skipped: invalid scene alias chain");
        stats.commands_rejected.fetch_add(1);
        return WorkDisposition::kPermanentFailure;
      }
      const SceneObjectPtr object = pinned.findObject(*canonical);
      if (!object) {
        recordError("snapshot command skipped: object is missing or tombstoned");
        stats.commands_rejected.fetch_add(1);
        return WorkDisposition::kPermanentFailure;
      }

      const std::optional<int> bank_canonical =
          snapshot_bank->resolveCanonicalObjectId(*canonical);
      const SceneObjectId bank_object_id =
          bank_canonical ? *bank_canonical : *canonical;
      const std::optional<SnapshotSet> snapshot_set =
          snapshot_bank->objectSnapshots(bank_object_id);
      if (!snapshot_set) {
        // An absent set carries no evidence that needs publication. This is
        // common for an idempotent promotion/merge with no retained images.
        return WorkDisposition::kComplete;
      }
      if (object->artifact &&
          object->artifact->snapshot_set_hash ==
              snapshot_set->snapshot_set_hash) {
        return WorkDisposition::kComplete;
      }

      ApplySnapshotSetCommand command;
      command.dependency = dependencyFor(*object);
      command.snapshot_set_hash = snapshot_set->snapshot_set_hash;
      command.snapshots.reserve(snapshot_set->records.size());
      try {
        for (std::size_t index = 0; index < snapshot_set->records.size();
             ++index) {
          command.snapshots.push_back(
              reference_factory(snapshot_set->records[index], index));
        }
      } catch (const std::exception& error) {
        recordError(std::string("snapshot reference conversion failed: ") +
                    error.what());
        stats.commands_rejected.fetch_add(1);
        return WorkDisposition::kPermanentFailure;
      }

      stats.commands_submitted.fetch_add(1);
      SnapshotCommandSubmitResult submitted;
      try {
        submitted = command_sink(command);
      } catch (const std::exception& error) {
        recordError(std::string("snapshot command sink failed: ") +
                    error.what());
        stats.commands_rejected.fetch_add(1);
        return WorkDisposition::kRetryableFailure;
      }
      if (submitted.disposition == SnapshotCommandDisposition::kAccepted) {
        stats.commands_accepted.fetch_add(1);
        return WorkDisposition::kComplete;
      }
      if (submitted.disposition == SnapshotCommandDisposition::kStale &&
          attempt == 0) {
        stats.command_stale_retries.fetch_add(1);
        retry_object_id = *canonical;
        continue;
      }

      recordError(submitted.reason.empty()
                      ? "snapshot command was rejected"
                      : "snapshot command was rejected: " + submitted.reason);
      stats.commands_rejected.fetch_add(1);
      return WorkDisposition::kRetryableFailure;
    }
    return WorkDisposition::kRetryableFailure;
  }

  WorkDisposition processCandidate(const CandidateWork& work) {
    stats.candidate_processed.fetch_add(1);
    SnapshotSubmitResult submitted;
    std::optional<SceneObjectId> changed_object;
    bool submitted_as_tentative = false;
    if (work.input.owner.kind == SnapshotOwnerKind::kTentativeTrack) {
      const auto promoted = promoted_tracks.find(work.input.owner.id);
      if (promoted == promoted_tracks.end()) {
        submitted_as_tentative = true;
        submitted = snapshot_bank->submitForTentativeTrack(
            work.input.owner.id, work.input.candidate, work.now_ns);
      } else {
        submitted = snapshot_bank->submitForObject(
            promoted->second, work.input.candidate, work.now_ns);
        changed_object = promoted->second;
      }
    } else {
      submitted = snapshot_bank->submitForObject(
          work.input.owner.id, work.input.candidate, work.now_ns);
      const std::optional<int> canonical =
          snapshot_bank->resolveCanonicalObjectId(work.input.owner.id);
      changed_object = canonical ? *canonical : work.input.owner.id;
    }

    if (!submitted.accepted) {
      stats.candidate_bank_rejected.fetch_add(1);
      if (!submitted.error.empty()) {
        recordError("snapshot candidate rejected by bank: " +
                    submitted.error);
        return WorkDisposition::kRetryableFailure;
      }
    } else {
      stats.candidate_bank_accepted.fetch_add(1);
    }

    if (work.input.owner.kind == SnapshotOwnerKind::kTentativeTrack) {
      if (submitted_as_tentative) {
        processed_tentative_candidates[work.input.owner.id] =
            std::max(processed_tentative_candidates[work.input.owner.id],
                     work.ingress_sequence);
      }
      std::lock_guard<std::mutex> lock(pending_candidate_mutex);
      const auto pending =
          pending_tentative_candidates.find(work.input.owner.id);
      if (pending != pending_tentative_candidates.end() &&
          pending->second.first == work.ingress_sequence) {
        pending_tentative_candidates.erase(pending);
      }
    }
    // Reconcile even when this candidate did not change the Top-K. A prior
    // reducer submission may have failed after the bank committed, and the
    // same evidence is then the event that must close that gap.
    if (changed_object) {
      return submitCurrentSet(*changed_object);
    }
    return WorkDisposition::kComplete;
  }

  WorkDisposition processPromotion(const PromotionWork& work) {
    const auto routed = promoted_tracks.find(work.tentative_track_id);
    if (routed != promoted_tracks.end() &&
        !snapshot_bank->tentativeSnapshots(work.tentative_track_id)) {
      return submitCurrentSet(routed->second);
    }
    const auto already_processed =
        processed_tentative_candidates.find(work.tentative_track_id);
    const bool absorbed_was_processed =
        already_processed != processed_tentative_candidates.end() &&
        already_processed->second >= work.absorbed_candidate_sequence;
    if (work.absorbed_candidate && !absorbed_was_processed) {
      const SnapshotSubmitResult absorbed =
          snapshot_bank->submitForTentativeTrack(
              work.tentative_track_id, *work.absorbed_candidate, work.now_ns);
      if (!absorbed.accepted && !absorbed.error.empty()) {
        recordError("snapshot promotion candidate failed: " + absorbed.error);
        return WorkDisposition::kRetryableFailure;
      }
    }
    // Promotion is still meaningful without retained evidence: remember the
    // routing fence so late candidate callbacks cannot recreate tentative
    // ownership after the track has become an object.
    if (!snapshot_bank->tentativeSnapshots(work.tentative_track_id)) {
      promoted_tracks[work.tentative_track_id] = work.object_id;
      processed_tentative_candidates.erase(work.tentative_track_id);
      return submitCurrentSet(work.object_id);
    }
    const SnapshotMergeResult promoted = snapshot_bank->promoteTentativeTrack(
        work.tentative_track_id, work.object_id, work.now_ns);
    if (!promoted.success) {
      recordError("snapshot promotion failed: " + promoted.error);
      return WorkDisposition::kRetryableFailure;
    }
    promoted_tracks[work.tentative_track_id] =
        promoted.canonical_object_id;
    processed_tentative_candidates.erase(work.tentative_track_id);
    return submitCurrentSet(promoted.canonical_object_id);
  }

  WorkDisposition processMerge(const MergeWork& work) {
    const SnapshotMergeResult merged = snapshot_bank->mergeObjects(
        work.retired_object_id, work.canonical_object_id, work.now_ns);
    if (!merged.success) {
      recordError("snapshot object merge failed: " + merged.error);
      return WorkDisposition::kRetryableFailure;
    }
    for (auto& routing : promoted_tracks) {
      if (routing.second == work.retired_object_id) {
        routing.second = merged.canonical_object_id;
      } else {
        const std::optional<int> canonical =
            snapshot_bank->resolveCanonicalObjectId(routing.second);
        if (canonical) {
          routing.second = *canonical;
        }
      }
    }
    return submitCurrentSet(merged.canonical_object_id);
  }

  WorkDisposition processDropTentative(const DropTentativeWork& work) {
    std::string error;
    const bool dropped = snapshot_bank->dropTentativeTrack(
        work.tentative_track_id, work.now_ns, &error);
    promoted_tracks.erase(work.tentative_track_id);
    processed_tentative_candidates.erase(work.tentative_track_id);
    {
      std::lock_guard<std::mutex> lock(pending_candidate_mutex);
      pending_tentative_candidates.erase(work.tentative_track_id);
    }
    if (!dropped) {
      recordError("snapshot tentative drop failed: " + error);
    }
    return dropped ? WorkDisposition::kComplete
                   : WorkDisposition::kRetryableFailure;
  }

  WorkDisposition processEraseObject(const EraseObjectWork& work) {
    const std::optional<int> canonical =
        snapshot_bank->resolveCanonicalObjectId(work.object_id);
    std::string error;
    const bool erased =
        snapshot_bank->eraseObject(work.object_id, work.now_ns, &error);
    if (erased) {
      for (auto it = promoted_tracks.begin(); it != promoted_tracks.end();) {
        if (it->second == work.object_id ||
            (canonical && it->second == *canonical)) {
          it = promoted_tracks.erase(it);
        } else {
          ++it;
        }
      }
    } else {
      recordError("snapshot object erase failed: " + error);
    }
    return erased ? WorkDisposition::kComplete
                  : WorkDisposition::kRetryableFailure;
  }

  WorkDisposition process(const WorkEvent& event) {
    WorkDisposition disposition = WorkDisposition::kComplete;
    std::visit(
        [&](const auto& work) {
          using Work = std::decay_t<decltype(work)>;
          if constexpr (std::is_same_v<Work, CandidateWork>) {
            disposition = processCandidate(work);
          } else if constexpr (std::is_same_v<Work, PromotionWork>) {
            disposition = processPromotion(work);
          } else if constexpr (std::is_same_v<Work, MergeWork>) {
            disposition = processMerge(work);
          } else if constexpr (std::is_same_v<Work, DropTentativeWork>) {
            disposition = processDropTentative(work);
          } else if constexpr (std::is_same_v<Work, EraseObjectWork>) {
            disposition = processEraseObject(work);
          }
        },
        event.payload);
    return disposition;
  }

  std::chrono::milliseconds retryBackoff(std::size_t retry_attempts) const {
    std::chrono::milliseconds delay = config.retry_initial_backoff;
    for (std::size_t index = 0;
         index < retry_attempts && delay < config.retry_max_backoff; ++index) {
      if (delay > config.retry_max_backoff / 2) {
        return config.retry_max_backoff;
      }
      delay *= 2;
    }
    return std::min(delay, config.retry_max_backoff);
  }

  void finalizeEvent(const WorkEvent& event, bool success) {
    if (!candidateWork(event)) {
      stats.control_processed.fetch_add(1);
      if (!success) {
        stats.control_failed.fetch_add(1);
      }
    }
    releaseOutstanding();
  }

  OnlineSnapshotWorkerConfig config;
  std::shared_ptr<SnapshotBank> snapshot_bank;
  SnapshotSupplier snapshot_supplier;
  CommandSink command_sink;
  SnapshotReferenceFactory reference_factory;
  BoundedChannel<WorkEvent> events;
  std::atomic<std::uint64_t> coalescing_epoch{0};
  std::atomic<std::uint64_t> candidate_sequence{1};
  std::mutex control_enqueue_mutex;
  mutable std::mutex pending_candidate_mutex;
  std::map<int, std::pair<std::uint64_t, SnapshotCandidate>>
      pending_tentative_candidates;
  std::map<int, SceneObjectId> promoted_tracks;
  std::map<int, std::uint64_t> processed_tentative_candidates;
  std::atomic<std::uint64_t> outstanding{0};
  mutable std::mutex idle_mutex;
  mutable std::condition_variable idle_cv;
  mutable std::mutex retry_wait_mutex;
  mutable std::condition_variable retry_wait_cv;
  mutable std::mutex error_mutex;
  std::string last_error;
  AtomicStats stats;
};

OnlineSnapshotWorker::OnlineSnapshotWorker(
    OnlineSnapshotWorkerConfig config,
    std::shared_ptr<SnapshotBank> snapshot_bank,
    SnapshotSupplier snapshot_supplier, CommandSink command_sink,
    SnapshotReferenceFactory reference_factory)
    : WorkerThread("online_snapshot_worker") {
  if (config.event_queue_capacity == 0) {
    throw std::invalid_argument(
        "online snapshot event queue capacity must be positive");
  }
  if (config.poll_period <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument(
        "online snapshot worker poll period must be positive");
  }
  if (config.asset_gc_period <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument(
        "online snapshot asset GC period must be positive");
  }
  if (config.max_retry_attempts == 0) {
    throw std::invalid_argument(
        "online snapshot worker max retry attempts must be positive");
  }
  if (config.retry_initial_backoff <= std::chrono::milliseconds::zero() ||
      config.retry_max_backoff < config.retry_initial_backoff) {
    throw std::invalid_argument(
        "online snapshot worker retry backoff range is invalid");
  }
  if (!snapshot_bank) {
    throw std::invalid_argument("online snapshot worker requires SnapshotBank");
  }
  if (!snapshot_supplier || !command_sink) {
    throw std::invalid_argument(
        "online snapshot worker requires snapshot and command callbacks");
  }
  impl_ = std::make_unique<Impl>(
      std::move(config), std::move(snapshot_bank),
      std::move(snapshot_supplier), std::move(command_sink),
      std::move(reference_factory));
}

OnlineSnapshotWorker::~OnlineSnapshotWorker() { stop(); }

SnapshotWorkerEnqueueResult OnlineSnapshotWorker::enqueueCandidate(
    OnlineSnapshotCandidate candidate, TimeNanoseconds now_ns) {
  SnapshotWorkerEnqueueResult result;
  if (!candidate.owner.valid()) {
    result.reason = "snapshot candidate owner id must be non-negative";
    impl_->stats.candidate_rejected.fetch_add(1);
    return result;
  }
  if (!candidate.candidate.full_frame) {
    result.reason = "snapshot candidate requires a shared full frame";
    impl_->stats.candidate_rejected.fetch_add(1);
    return result;
  }

  WorkEvent event;
  event.coalescing_epoch =
      impl_->coalescing_epoch.load(std::memory_order_acquire);
  const SnapshotOwner owner = candidate.owner;
  const std::uint64_t ingress_sequence =
      impl_->candidate_sequence.fetch_add(1, std::memory_order_relaxed);
  std::optional<SnapshotCandidate> tentative_copy;
  if (owner.kind == SnapshotOwnerKind::kTentativeTrack) {
    tentative_copy = candidate.candidate;
  }
  event.payload =
      CandidateWork{std::move(candidate), now_ns, ingress_sequence};
  impl_->reserveOutstanding();
  PushResult<WorkEvent> pushed = impl_->events.tryPush(std::move(event));
  result.outcome = pushed.outcome;
  result.replacement_reason = pushed.replacement_reason;
  result.reason = pushFailureReason(pushed.outcome);
  if (pushed.outcome == PushOutcome::kAccepted) {
    impl_->stats.candidate_enqueued.fetch_add(1);
  } else if (pushed.outcome == PushOutcome::kReplaced) {
    impl_->stats.candidate_enqueued.fetch_add(1);
    impl_->releaseOutstanding();
    if (pushed.replacement_reason ==
        ChannelReplacementReason::kMatchingKey) {
      impl_->stats.candidate_replaced_same_owner.fetch_add(1);
    } else if (pushed.replacement_reason ==
               ChannelReplacementReason::kCapacity) {
      impl_->stats.candidate_dropped_for_capacity.fetch_add(1);
    }
    if (pushed.replaced_item) {
      const CandidateWork* replaced = candidateWork(*pushed.replaced_item);
      if (replaced) {
        result.replaced_candidate = replaced->input.owner;
      }
    }
  } else {
    impl_->releaseOutstanding();
    if (pushed.outcome == PushOutcome::kStopped) {
      impl_->stats.candidate_stopped.fetch_add(1);
    } else {
      impl_->stats.candidate_rejected.fetch_add(1);
    }
  }
  if (result.accepted() && tentative_copy) {
    std::lock_guard<std::mutex> lock(impl_->pending_candidate_mutex);
    auto& pending = impl_->pending_tentative_candidates[owner.id];
    if (ingress_sequence >= pending.first) {
      pending = std::make_pair(ingress_sequence, std::move(*tentative_copy));
    }
  }
  return result;
}

SnapshotWorkerEnqueueResult OnlineSnapshotWorker::enqueuePromotion(
    int tentative_track_id, SceneObjectId object_id,
    TimeNanoseconds now_ns) {
  if (tentative_track_id < 0 || object_id < 0) {
    SnapshotWorkerEnqueueResult result;
    result.reason = "promotion ids must be non-negative";
    return result;
  }
  PromotionWork work;
  work.tentative_track_id = tentative_track_id;
  work.object_id = object_id;
  work.now_ns = now_ns;
  {
    std::lock_guard<std::mutex> lock(impl_->pending_candidate_mutex);
    const auto pending =
        impl_->pending_tentative_candidates.find(tentative_track_id);
    if (pending != impl_->pending_tentative_candidates.end()) {
      work.absorbed_candidate_sequence = pending->second.first;
      work.absorbed_candidate = pending->second.second;
    }
  }
  const std::uint64_t absorbed_sequence = work.absorbed_candidate_sequence;
  SnapshotWorkerEnqueueResult result =
      impl_->enqueueReliableControl(std::move(work));
  if (result.accepted() && absorbed_sequence != 0) {
    std::lock_guard<std::mutex> lock(impl_->pending_candidate_mutex);
    const auto pending =
        impl_->pending_tentative_candidates.find(tentative_track_id);
    if (pending != impl_->pending_tentative_candidates.end() &&
        pending->second.first == absorbed_sequence) {
      impl_->pending_tentative_candidates.erase(pending);
    }
  }
  return result;
}

SnapshotWorkerEnqueueResult OnlineSnapshotWorker::enqueueMerge(
    SceneObjectId retired_object_id, SceneObjectId canonical_object_id,
    TimeNanoseconds now_ns) {
  if (retired_object_id < 0 || canonical_object_id < 0) {
    SnapshotWorkerEnqueueResult result;
    result.reason = "merge ids must be non-negative";
    return result;
  }
  return impl_->enqueueReliableControl(
      MergeWork{retired_object_id, canonical_object_id, now_ns});
}

SnapshotWorkerEnqueueResult OnlineSnapshotWorker::enqueueDropTentative(
    int tentative_track_id, TimeNanoseconds now_ns) {
  if (tentative_track_id < 0) {
    SnapshotWorkerEnqueueResult result;
    result.reason = "tentative track id must be non-negative";
    return result;
  }
  return impl_->enqueueReliableControl(
      DropTentativeWork{tentative_track_id, now_ns});
}

SnapshotWorkerEnqueueResult OnlineSnapshotWorker::enqueueEraseObject(
    SceneObjectId object_id, TimeNanoseconds now_ns) {
  if (object_id < 0) {
    SnapshotWorkerEnqueueResult result;
    result.reason = "object id must be non-negative";
    return result;
  }
  return impl_->enqueueReliableControl(EraseObjectWork{object_id, now_ns});
}

SnapshotWorkerEnqueueResult OnlineSnapshotWorker::tryEnqueuePromotion(
    int tentative_track_id, SceneObjectId object_id,
    TimeNanoseconds now_ns) {
  if (tentative_track_id < 0 || object_id < 0) {
    SnapshotWorkerEnqueueResult result;
    result.reason = "promotion ids must be non-negative";
    return result;
  }
  PromotionWork work;
  work.tentative_track_id = tentative_track_id;
  work.object_id = object_id;
  work.now_ns = now_ns;
  {
    std::lock_guard<std::mutex> lock(impl_->pending_candidate_mutex);
    const auto pending =
        impl_->pending_tentative_candidates.find(tentative_track_id);
    if (pending != impl_->pending_tentative_candidates.end()) {
      work.absorbed_candidate_sequence = pending->second.first;
      work.absorbed_candidate = pending->second.second;
    }
  }
  const std::uint64_t absorbed_sequence = work.absorbed_candidate_sequence;
  SnapshotWorkerEnqueueResult result =
      impl_->tryEnqueueReliableControl(std::move(work));
  if (result.accepted() && absorbed_sequence != 0) {
    std::lock_guard<std::mutex> lock(impl_->pending_candidate_mutex);
    const auto pending =
        impl_->pending_tentative_candidates.find(tentative_track_id);
    if (pending != impl_->pending_tentative_candidates.end() &&
        pending->second.first == absorbed_sequence) {
      impl_->pending_tentative_candidates.erase(pending);
    }
  }
  return result;
}

SnapshotWorkerEnqueueResult OnlineSnapshotWorker::tryEnqueueMerge(
    SceneObjectId retired_object_id, SceneObjectId canonical_object_id,
    TimeNanoseconds now_ns) {
  if (retired_object_id < 0 || canonical_object_id < 0) {
    SnapshotWorkerEnqueueResult result;
    result.reason = "merge ids must be non-negative";
    return result;
  }
  return impl_->tryEnqueueReliableControl(
      MergeWork{retired_object_id, canonical_object_id, now_ns});
}

SnapshotWorkerEnqueueResult OnlineSnapshotWorker::tryEnqueueDropTentative(
    int tentative_track_id, TimeNanoseconds now_ns) {
  if (tentative_track_id < 0) {
    SnapshotWorkerEnqueueResult result;
    result.reason = "tentative track id must be non-negative";
    return result;
  }
  return impl_->tryEnqueueReliableControl(
      DropTentativeWork{tentative_track_id, now_ns});
}

SnapshotWorkerEnqueueResult OnlineSnapshotWorker::tryEnqueueEraseObject(
    SceneObjectId object_id, TimeNanoseconds now_ns) {
  if (object_id < 0) {
    SnapshotWorkerEnqueueResult result;
    result.reason = "object id must be non-negative";
    return result;
  }
  return impl_->tryEnqueueReliableControl(
      EraseObjectWork{object_id, now_ns});
}

bool OnlineSnapshotWorker::idle() const {
  return impl_->outstanding.load(std::memory_order_acquire) == 0;
}

bool OnlineSnapshotWorker::waitUntilIdle(
    std::chrono::milliseconds timeout) const {
  if (timeout < std::chrono::milliseconds::zero()) {
    timeout = std::chrono::milliseconds::zero();
  }
  std::unique_lock<std::mutex> lock(impl_->idle_mutex);
  return impl_->idle_cv.wait_for(lock, timeout, [this]() { return idle(); });
}

OnlineSnapshotWorkerStats OnlineSnapshotWorker::stats() const {
  return impl_->stats.snapshot();
}

ChannelStats OnlineSnapshotWorker::queueStats() const {
  return impl_->events.stats();
}

std::string OnlineSnapshotWorker::lastError() const {
  return impl_->error();
}

void OnlineSnapshotWorker::run() {
  struct RetryState {
    WorkEvent event;
    std::size_t attempts = 0;
    std::chrono::steady_clock::time_point due;
  };

  auto next_gc = std::chrono::steady_clock::now() +
                 impl_->config.asset_gc_period;
  std::optional<RetryState> retry;
  while (!stopRequested() || !impl_->events.empty() || retry) {
    if (retry && stopRequested()) {
      impl_->stats.retry_abandoned_on_stop.fetch_add(1);
      impl_->recordError(
          "online snapshot retry abandoned during bounded shutdown");
      impl_->finalizeEvent(retry->event, false);
      retry.reset();
      continue;
    }

    if (retry && std::chrono::steady_clock::now() < retry->due) {
      std::unique_lock<std::mutex> lock(impl_->retry_wait_mutex);
      impl_->retry_wait_cv.wait_until(
          lock, retry->due, [this]() { return stopRequested(); });
      continue;
    }

    WorkEvent event;
    const bool retrying = retry.has_value();
    if (retrying) {
      event = std::move(retry->event);
      ++retry->attempts;
      impl_->stats.retry_attempts.fetch_add(1);
    } else if (!impl_->events.waitPopFor(&event,
                                         impl_->config.poll_period)) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= next_gc) {
        const AssetGcResult collected = impl_->snapshot_bank->collectGarbage();
        impl_->stats.asset_gc_runs.fetch_add(1);
        impl_->stats.assets_collected.fetch_add(collected.removed_assets);
        impl_->stats.asset_bytes_collected.fetch_add(collected.removed_bytes);
        if (!collected.error.empty()) {
          impl_->stats.asset_gc_failures.fetch_add(1);
          impl_->recordError("snapshot asset GC failed: " + collected.error);
        }
        next_gc = now + impl_->config.asset_gc_period;
      }
      continue;
    }

    WorkDisposition disposition = WorkDisposition::kRetryableFailure;
    try {
      disposition = impl_->process(event);
    } catch (const std::exception& error) {
      impl_->recordError(std::string("online snapshot event failed: ") +
                         error.what());
    } catch (...) {
      impl_->recordError(
          "online snapshot event failed with unknown exception");
    }

    if (!retrying) {
      impl_->stats.events_processed.fetch_add(1);
    }
    if (disposition == WorkDisposition::kComplete) {
      impl_->finalizeEvent(event, true);
      retry.reset();
    } else if (disposition == WorkDisposition::kPermanentFailure) {
      impl_->finalizeEvent(event, false);
      retry.reset();
    } else if (stopRequested()) {
      impl_->stats.retry_abandoned_on_stop.fetch_add(1);
      impl_->finalizeEvent(event, false);
      retry.reset();
    } else {
      const std::size_t attempts = retrying ? retry->attempts : 0;
      if (attempts >= impl_->config.max_retry_attempts) {
        impl_->stats.retry_exhausted.fetch_add(1);
        impl_->recordError("online snapshot event exhausted retry budget");
        impl_->finalizeEvent(event, false);
        retry.reset();
      } else {
        RetryState next_retry;
        next_retry.event = std::move(event);
        next_retry.attempts = attempts;
        next_retry.due = std::chrono::steady_clock::now() +
                         impl_->retryBackoff(attempts);
        retry = std::move(next_retry);
      }
    }

    const auto now = std::chrono::steady_clock::now();
    if (now >= next_gc) {
      const AssetGcResult collected = impl_->snapshot_bank->collectGarbage();
      impl_->stats.asset_gc_runs.fetch_add(1);
      impl_->stats.assets_collected.fetch_add(collected.removed_assets);
      impl_->stats.asset_bytes_collected.fetch_add(collected.removed_bytes);
      if (!collected.error.empty()) {
        impl_->stats.asset_gc_failures.fetch_add(1);
        impl_->recordError("snapshot asset GC failed: " + collected.error);
      }
      next_gc = now + impl_->config.asset_gc_period;
    }
  }
}

void OnlineSnapshotWorker::onStopRequested() {
  impl_->events.stop();
  impl_->retry_wait_cv.notify_all();
}

}  // namespace roomie
