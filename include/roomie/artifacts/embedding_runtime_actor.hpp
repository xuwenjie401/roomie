#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "roomie/artifacts/embedding_task_scheduler.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

struct EmbeddingRuntimeActorConfig {
  std::string lease_owner = "embedding-runtime";
  std::size_t maximum_batch_size = 16;
  std::chrono::milliseconds collection_window{20};
  std::chrono::milliseconds idle_poll_period{20};
  std::chrono::milliseconds heartbeat_period{5000};
  std::chrono::milliseconds invocation_wait_slice{10};
  std::chrono::milliseconds index_publish_wait_slice{10};
  std::chrono::milliseconds index_publish_timeout{2000};
  // Bound used when stop races an encoder call. The invocation owns a shared
  // encoder reference and may finish detached; its result is never accepted or
  // acknowledged after the actor has stopped.
  std::chrono::milliseconds encoder_stop_timeout{250};
  std::size_t slo_accounting_history_capacity = 4096;
};

struct EmbeddingRuntimeActorStats {
  bool warmed = false;
  std::uint64_t warmup_attempts = 0;
  std::uint64_t warmup_failures = 0;
  std::uint64_t lease_polls = 0;
  std::uint64_t leases_acquired = 0;
  std::uint64_t batches = 0;
  std::uint64_t encoded_documents = 0;
  std::uint64_t persisted_records = 0;
  std::uint64_t accepted_records = 0;
  std::uint64_t completed = 0;
  std::uint64_t superseded = 0;
  std::uint64_t retried = 0;
  std::uint64_t terminal_failures = 0;
  std::uint64_t lease_lost = 0;
  std::uint64_t scheduler_errors = 0;
  std::uint64_t persistence_errors = 0;
  std::uint64_t index_errors = 0;
  std::uint64_t interactive_slo_completions = 0;
  std::uint64_t bulk_slo_completions = 0;
  std::uint64_t untracked_slo_completions = 0;
  std::uint64_t interactive_slo_violations = 0;
  std::uint64_t bulk_slo_violations = 0;
  std::uint64_t slo_violations = 0;
  std::uint64_t stop_requeues = 0;
  std::uint64_t detached_encoder_invocations = 0;
  std::size_t slo_accounting_history_size = 0;
};

// Durable embedding outbox consumer. DurableEmbeddingSink must not return
// success before persistSemanticEmbeddingRecord() (or an equivalent atomic
// durable write) has committed. TerminalFailureSink is the compatibility seam
// for a future fenced SceneStore kFailed transition; without it, permanent
// failures remain safely retryable and are never disguised as completion.
class EmbeddingRuntimeActor final : public WorkerThread {
 public:
  using SceneSnapshotSupplier = std::function<SceneSnapshot()>;
  using DurableEmbeddingSink =
      std::function<SceneStoreStatus(const EmbeddingRecord&)>;
  using TerminalFailureSink = std::function<SceneStoreStatus(
      const EmbeddingTaskLease& lease,
      std::int64_t failed_at_unix_ms,
      std::string error)>;
  using UnixMillisClock = std::function<std::int64_t()>;

  EmbeddingRuntimeActor(
      EmbeddingRuntimeActorConfig config,
      EmbeddingTaskScheduler* scheduler,
      std::shared_ptr<EmbeddingEncoder> encoder,
      VersionedSemanticIndex* index,
      SceneSnapshotSupplier snapshot_supplier,
      DurableEmbeddingSink durable_embedding_sink,
      TerminalFailureSink terminal_failure_sink = {},
      UnixMillisClock unix_millis_clock = {});
  ~EmbeddingRuntimeActor() override;

  bool waitUntilWarmed(std::chrono::milliseconds timeout) const;
  bool waitUntilIdle(std::chrono::milliseconds timeout) const;
  bool idle() const;
  EmbeddingRuntimeActorStats stats() const;
  std::string lastError() const;
  std::vector<std::string> activeTaskIds() const;

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  struct LeaseHeartbeat;
  struct ActiveLease;
  struct EncoderInvocation;

  static EmbeddingRuntimeActorConfig validateConfig(
      EmbeddingRuntimeActorConfig config);

  bool prewarmEncoder();
  bool processNextBatch();
  bool collectLease(std::vector<std::shared_ptr<ActiveLease>>* batch);
  bool validateCurrent(
      const EmbeddingTaskLease& lease,
      const SceneSnapshot& snapshot,
      SemanticDocument* document,
      EmbeddingDependencyFreshness* freshness);
  bool completeLease(const std::shared_ptr<ActiveLease>& active);
  bool retryLease(const std::shared_ptr<ActiveLease>& active,
                  std::string error,
                  bool stopping = false);
  bool failLease(const std::shared_ptr<ActiveLease>& active,
                 std::string error);
  bool disposeSuperseded(const std::shared_ptr<ActiveLease>& active,
                         const SceneSnapshot& snapshot,
                         EmbeddingDependencyFreshness freshness);
  bool waitUntilQueryable(const std::shared_ptr<ActiveLease>& active);
  void retryBatch(const std::vector<std::shared_ptr<ActiveLease>>& batch,
                  const std::string& error,
                  bool stopping = false);

  std::int64_t nowUnixMillis() const;
  void setActive(const std::vector<std::shared_ptr<ActiveLease>>& active);
  void clearActive();
  void recordError(std::string error);
  void accountSloQueryableAccept(const EmbeddingTaskLease& lease,
                                 std::int64_t accepted_at_unix_ms);
  void interruptibleWait(std::chrono::milliseconds duration);

  EmbeddingRuntimeActorConfig config_;
  EmbeddingTaskScheduler* scheduler_ = nullptr;
  std::shared_ptr<EmbeddingEncoder> encoder_;
  VersionedSemanticIndex* index_ = nullptr;
  SceneSnapshotSupplier snapshot_supplier_;
  DurableEmbeddingSink durable_embedding_sink_;
  TerminalFailureSink terminal_failure_sink_;
  UnixMillisClock unix_millis_clock_;

  mutable std::mutex state_mutex_;
  mutable std::condition_variable state_cv_;
  std::string last_error_;
  std::vector<std::string> active_task_ids_;
  bool idle_ = true;
  bool warmed_ = false;

  mutable std::mutex slo_mutex_;
  std::unordered_set<std::string> slo_accounted_task_ids_;
  std::deque<std::string> slo_accounted_task_order_;

  std::atomic_uint64_t warmup_attempts_{0};
  std::atomic_uint64_t warmup_failures_{0};
  std::atomic_uint64_t lease_polls_{0};
  std::atomic_uint64_t leases_acquired_{0};
  std::atomic_uint64_t batches_{0};
  std::atomic_uint64_t encoded_documents_{0};
  std::atomic_uint64_t persisted_records_{0};
  std::atomic_uint64_t accepted_records_{0};
  std::atomic_uint64_t completed_{0};
  std::atomic_uint64_t superseded_{0};
  std::atomic_uint64_t retried_{0};
  std::atomic_uint64_t terminal_failures_{0};
  std::atomic_uint64_t lease_lost_{0};
  std::atomic_uint64_t scheduler_errors_{0};
  std::atomic_uint64_t persistence_errors_{0};
  std::atomic_uint64_t index_errors_{0};
  std::atomic_uint64_t interactive_slo_completions_{0};
  std::atomic_uint64_t bulk_slo_completions_{0};
  std::atomic_uint64_t untracked_slo_completions_{0};
  std::atomic_uint64_t interactive_slo_violations_{0};
  std::atomic_uint64_t bulk_slo_violations_{0};
  std::atomic_uint64_t slo_violations_{0};
  std::atomic_uint64_t stop_requeues_{0};
  std::atomic_uint64_t detached_encoder_invocations_{0};
};

}  // namespace roomie
