#pragma once

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/pipeline/map_backend.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

class MapThread : public WorkerThread, public MapProjector {
 public:
  MapThread(ThreadSafeQueue<FrameBundlePtr>& mapping_queue, PipelineConfig config);
  MapThread(ThreadSafeQueue<FrameBundlePtr>& mapping_queue,
            PipelineConfig config,
            std::unique_ptr<MapBackend> map_backend);
  ~MapThread() override;

  bool enqueueFrameBundle(FrameBundlePtr frame) override;
  std::optional<PatchDepth> projectPatchDepth(const FrameBundle& frame) override;
  std::optional<PatchDepth> projectPatchDepth(
      const FrameBundle& frame, const MapCommit& commit) override;
  bool cancelPerceptionCandidate(FrameBundlePtr frame,
                                 const std::string& reason) override;
  MapBackendSnapshot snapshotSurfacePoints() const override;
  std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>> collectNearSurfaceVoxels(
      const RawDetection& detection) const override;

  std::uint64_t mapVersion() const;
  MapBackendSnapshot debugSnapshot() const;
  SurfaceSnapshotPtr latestSurfaceSnapshot() const;
  MapDeltaQueryResult queryMapDeltas(
      const SurfaceStamp& from,
      const SurfaceStamp& to,
      const std::vector<BlockIndex>& dependency_blocks) const;
  void setCommitObserver(
      std::function<void(const MapCommit&)> observer);
  // Startup-only load and post-stop save boundaries. Neither operation is
  // allowed while the actor is running, preserving MapThread as the sole map
  // writer during normal execution.
  MapCheckpointOperationResult loadConfiguredCheckpoint(
      const MapCheckpointManifest* expected_manifest = nullptr);
  MapCheckpointOperationResult saveConfiguredCheckpoint();

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  struct PublishedSurface {
    SurfaceSnapshotPtr snapshot;
    MapBackendSnapshot diagnostics;
  };

  struct CommitRecord {
    bool success = false;
    MapCommit commit;
    MapBackendSnapshot diagnostics;
    std::string error;
  };

  struct CancelledCandidate {
    std::string reason;
    std::chrono::steady_clock::time_point due_time =
        std::chrono::steady_clock::time_point::max();
  };

  std::optional<CommitRecord> waitForCommit(const FrameBundle& frame);
  std::optional<PatchDepth> projectPinnedSurface(
      const FrameBundle& frame,
      SurfaceSnapshotPtr pinned_surface,
      MapBackendSnapshot diagnostics,
      FrameProvenance provenance,
      std::chrono::steady_clock::time_point project_total_start);
  CommitRecord buildCommit(const FrameBundle& frame,
                           const MapIntegrationResult& integration,
                           SurfaceRefreshResult refresh);
  void pruneExpiredCommitsLocked(
      std::chrono::steady_clock::time_point now);
  void publishCommit(CommitRecord record);
  void initializeStartupSurface();
  MapBackendSnapshot snapshotFromPinnedSurface(bool include_debug) const;
  void maybeLogStatus();
  PatchDepth projectSurfaceBlocksToPatchDepth(
      const FrameBundle& frame,
      const std::vector<SurfaceBlockPtr>& surface_blocks,
      std::uint64_t map_version,
      float min_depth_m,
      float max_depth_m) const;

  ThreadSafeQueue<FrameBundlePtr>& mapping_queue_;
  PipelineConfig config_;
  std::unique_ptr<MapBackend> map_backend_;
  std::shared_ptr<const PublishedSurface> latest_published_surface_;
  mutable std::mutex commit_mutex_;
  std::condition_variable commit_cv_;
  std::unordered_map<FrameKey, CommitRecord, FrameKeyHash> commits_by_frame_;
  std::deque<FrameKey> commit_order_;
  std::unordered_map<FrameKey, CancelledCandidate, FrameKeyHash>
      cancelled_candidates_;
  std::deque<FrameKey> cancelled_candidate_order_;
  bool frozen_initialization_complete_ = false;
  std::string frozen_initialization_error_;
  std::uint64_t next_surface_revision_ = 1;
  MapDeltaJournal delta_journal_{128};
  std::function<void(const MapCommit&)> commit_observer_;
  bool checkpoint_load_attempted_ = false;
  bool checkpoint_loaded_ = false;
  bool checkpoint_save_attempted_ = false;
  std::mutex status_mutex_;
  std::chrono::steady_clock::time_point last_status_log_time_ =
      std::chrono::steady_clock::now();
  std::atomic_uint64_t integrated_frames_{0};
  RunId map_epoch_ = makeRunId();
  std::atomic<TimeNanoseconds> integrated_through_ns_{0};
  std::atomic_uint64_t projection_requests_{0};
  std::atomic_uint64_t projection_no_map_{0};
  std::atomic_int last_valid_patches_{0};
  std::atomic_int last_projected_points_{0};
  std::atomic_uint64_t last_projection_map_version_{0};
  std::atomic_uint64_t last_source_surface_points_{0};
  std::atomic_uint64_t last_source_tsdf_blocks_{0};
  std::atomic_uint64_t last_source_voxels_scanned_{0};
  std::atomic_uint64_t last_source_selected_blocks_{0};
  std::atomic_uint64_t last_source_cached_surface_points_{0};
  std::atomic_uint64_t last_surface_cache_rebuilds_{0};
  std::atomic<double> last_project_total_ms_{0.0};
  std::atomic<double> last_snapshot_ms_{0.0};
  std::atomic<double> last_surface_extract_ms_{0.0};
  std::atomic<double> last_cache_build_ms_{0.0};
  std::atomic<double> last_frustum_filter_ms_{0.0};
  std::atomic<double> last_projection_loop_ms_{0.0};
  std::atomic<double> last_projection_zbuffer_ms_{0.0};
  std::atomic<double> last_projection_compute_ms_{0.0};
  std::atomic<double> last_integrate_ms_{0.0};
  std::atomic<double> last_refresh_ms_{0.0};
};

}  // namespace roomie
