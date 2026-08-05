#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <functional>
#include <vector>

#include "roomie/artifacts/artifact_scheduler.hpp"
#include "roomie/artifacts/embedding_task_scheduler.hpp"
#include "roomie/artifacts/semantic_index.hpp"
#include "roomie/scene/scene_reducer.hpp"

namespace roomie {

struct ArtifactIntentBuilderConfig {
  std::string dam_model_id = "roomie-dam.default.v1";
  std::string dam_prompt_hash = "roomie-dam-prompt.default.v1";
  std::string dam_output_schema_version = "dam.schema.v1";
  ArtifactSchedulerConfig dam_scheduler;

  EmbeddingNamespace embedding_namespace{"semantic.default.v1", 384};
  std::string embedding_task_type = "embedding.document.v1";

  // For first appearance only: the sixth distinct object entering this
  // rolling window is bulk. Later appearance revisions always stay
  // interactive, independent of burst size.
  std::int64_t new_object_window_ms = 10'000;
  std::size_t new_object_interactive_limit = 5;
  // Replay memoization is process-local and bounded; durable dedupe remains
  // authoritative across restarts.
  std::size_t replay_memo_capacity = 4096;
};

// Stable identity of an embedding invocation. Document hash is the semantic
// dependency fence; volatile metadata and scene revision are intentionally not
// part of the key.
std::string canonicalEmbeddingTaskKey(
    SceneObjectId object_id,
    const std::string& document_hash,
    const EmbeddingNamespace& name_space);

struct ArtifactCommitIntents {
  std::vector<DurableTaskSpec> tasks;
  std::vector<DurableArtifactOrigin> origins;

  bool empty() const { return tasks.empty() && origins.empty(); }
  std::size_t size() const { return tasks.size(); }
  const DurableTaskSpec& front() const { return tasks.front(); }
  operator std::vector<DurableTaskSpec>() const { return tasks; }
};

// Translates reducer events into outbox rows that can be committed atomically
// with SceneApplyResult::snapshot by SceneStore::enqueueCommit(). The builder
// is thread-safe and memoizes owning event identities: replaying a commit
// returns the byte-identical task instead of re-admitting it into the burst
// window with a different priority/time.
class ArtifactIntentBuilder {
 public:
  using DurableTaskLookup =
      std::function<TaskLookupResult(const std::string& task_id)>;

  explicit ArtifactIntentBuilder(ArtifactIntentBuilderConfig config = {});

  ArtifactCommitIntents build(const SceneApplyResult& commit,
                              std::int64_t created_unix_ms,
                              std::int64_t created_steady_ns = 0);
  // Restores only durable Unix origins. Their process-local steady clock is
  // deliberately cleared, so subsequent SLO accounting reports Unix fallback.
  void restoreObjectOrigins(
      const std::vector<DurableArtifactOrigin>& origins);
  void setDurableTaskLookup(DurableTaskLookup lookup);
  std::size_t memoizedTaskCount() const;
  std::size_t objectOriginCount() const;

 private:
  struct ArtifactOrigin {
    std::int64_t created_unix_ms = 0;
    std::int64_t created_steady_ns = 0;
    ArtifactPriority priority = ArtifactPriority::kInteractive;
  };

  DurableTaskSpec buildDamTask(const SceneApplyResult& commit,
                               const SnapshotSetCommitted& event,
                               const ArtifactOrigin& origin) const;
  DurableTaskSpec buildEmbeddingTask(const SceneApplyResult& commit,
                                     const DescriptionCommitted& event,
                                     std::int64_t created_unix_ms) const;
  DurableTaskSpec buildEmbeddingTask(
      const SceneApplyResult& commit,
      const HumanAnnotationCommitted& event,
      std::int64_t created_unix_ms) const;
  ArtifactPriority classifySnapshotSet(const SnapshotSetCommitted& event,
                                       std::int64_t created_steady_ns);
  std::optional<DurableTaskSpec> lookupDurableTask(
      const std::string& task_id) const;
  void memoizeTask(std::string key, DurableTaskSpec task);
  void rememberObjectOrigin(SceneObjectId object_id,
                            ArtifactOrigin origin);
  void forgetObjectOrigin(SceneObjectId object_id);

  ArtifactIntentBuilderConfig config_;
  ArtifactScheduler artifact_scheduler_;
  EmbeddingTaskScheduler embedding_scheduler_;
  mutable std::mutex mutex_;
  std::deque<std::int64_t> new_object_arrivals_steady_ns_;
  std::map<SceneObjectId, ArtifactOrigin> object_origins_;
  std::map<std::string, DurableTaskSpec> memoized_tasks_;
  std::deque<std::string> memoized_task_order_;
  DurableTaskLookup durable_task_lookup_;
};

}  // namespace roomie
