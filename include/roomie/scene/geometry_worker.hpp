#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>

#include "roomie/pipeline/surface_snapshot.hpp"
#include "roomie/pipeline/worker_thread.hpp"
#include "roomie/scene/geometry_scheduler.hpp"
#include "roomie/scene/scene_reducer.hpp"

namespace roomie {

struct GeometryWorkerStats {
  std::uint64_t map_commits = 0;
  std::uint64_t scheduled = 0;
  std::uint64_t evaluated = 0;
  std::uint64_t commands_enqueued = 0;
  // Evaluated results rejected by CAS because their dependencies advanced.
  std::uint64_t superseded = 0;
  // Pending work replaced before evaluation by newer work for the same
  // object. This is deliberately separate from CAS supersession.
  std::uint64_t pending_superseded = 0;
  // Distinct-object producers that waited for bounded queue capacity.
  std::uint64_t backpressure_waits = 0;
  // CAS-rejected objects explicitly resubmitted against current state.
  std::uint64_t retry_attempts = 0;
  std::uint64_t retries_scheduled = 0;
  std::uint64_t command_queue_rejected = 0;
};

// Owns the expensive geometry evaluator. Map/reducer actors only enqueue
// immutable inputs and remain free of surface scans.
class GeometryWorkerThread : public WorkerThread {
 public:
  using SceneProvider = std::function<SceneSnapshot()>;
  using CommandSink = std::function<bool(SceneCommand)>;

  GeometryWorkerThread(GeometrySchedulerConfig config,
                       SceneProvider scene_provider,
                       CommandSink command_sink,
                       std::size_t delta_journal_capacity = 128);

  void onMapCommit(const MapCommit& commit);
  void onSceneCommit(const SceneApplyResult& result);
  GeometryWorkerStats stats() const;

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  bool queueRetry(SceneObjectId object_id, GeometryTrigger trigger);

  GeometryScheduler scheduler_;
  MapDeltaJournal delta_journal_;
  SceneProvider scene_provider_;
  CommandSink command_sink_;
  SurfaceSnapshotPtr latest_surface_;
  // Worker-local, LatestByObject retry state. It is bounded by the number of
  // live scene objects and avoids a sole-consumer self-deadlock on a full
  // pending channel.
  std::map<SceneObjectId, GeometryInput> pending_retries_;
  std::atomic_uint64_t map_commits_{0};
  std::atomic_uint64_t scheduled_{0};
  std::atomic_uint64_t evaluated_{0};
  std::atomic_uint64_t commands_enqueued_{0};
  std::atomic_uint64_t superseded_{0};
  std::atomic_uint64_t retry_attempts_{0};
  std::atomic_uint64_t retries_scheduled_{0};
  std::atomic_uint64_t command_queue_rejected_{0};
};

}  // namespace roomie
