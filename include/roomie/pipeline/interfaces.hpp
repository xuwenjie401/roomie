#pragma once

#include <memory>
#include <optional>
#include <filesystem>
#include <vector>

#include "roomie/dsg/object_graph.hpp"
#include "roomie/pipeline/map_backend.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/types.hpp"

namespace roomie {

class MapProjector {
 public:
  virtual ~MapProjector() = default;

  virtual bool enqueueFrameBundle(FrameBundlePtr frame) = 0;
  virtual std::optional<PatchDepth> projectPatchDepth(const FrameBundle& frame) = 0;
  // Non-blocking projection against the most recently published immutable
  // surface. Independent RGB detections use this path and never wait for an
  // exact commit of their own sensor timestamp.
  virtual std::optional<PatchDepth> projectLatestPatchDepth(
      const FrameBundle& frame) {
    return projectPatchDepth(frame);
  }
  // Non-blocking exact-commit projection seam used by PerceptionScheduler.
  // Legacy/test projectors may keep implementing only the original method;
  // production MapThread validates and projects the supplied pinned commit.
  virtual std::optional<PatchDepth> projectPatchDepth(
      const FrameBundle& frame, const MapCommit& commit) {
    (void)commit;
    return projectPatchDepth(frame);
  }
  virtual bool cancelPerceptionCandidate(FrameBundlePtr frame,
                                         const std::string& reason) {
    (void)frame;
    (void)reason;
    return false;
  }
  virtual MapBackendSnapshot snapshotSurfacePoints() const = 0;
  virtual std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>>
  collectNearSurfaceVoxels(const RawDetection& detection) const = 0;
};

class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;

  virtual PushResult<InferenceRequest> enqueueRequest(InferenceRequest request) = 0;
  virtual bool tryPopResponse(InferenceResponse* response) = 0;
  virtual bool idle() const = 0;
  virtual ChannelStats requestChannelStats() const { return {}; }
  virtual ChannelStats responseChannelStats() const { return {}; }

  // Starts a coordinated shutdown without closing the result side of the
  // backend. Producers must be rejected and in-flight work must be interrupted,
  // while already accepted requests are accounted for with terminal responses
  // that remain drainable by DetectionBridgeThread.
  //
  // The default is intentionally a no-op for synchronous/test backends. Async
  // backends that can block outside Roomie's channels must override it.
  virtual void beginShutdown() {}
};

class InstanceStore {
 public:
  virtual ~InstanceStore() = default;

  virtual bool enqueueDetections(InferenceResponse response) = 0;
  virtual std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
  snapshotInstances() const = 0;
  virtual std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
  snapshotTrackedInstances() const = 0;
  virtual ObjectGraphSnapshot snapshotObjectGraph() const = 0;
  virtual bool prepareSceneGraphForSave(ObjectGraphSnapshot* snapshot,
                                        const std::filesystem::path& snapshot_image_dir,
                                        const std::string& snapshot_uri_prefix,
                                        std::string* error) const = 0;
};

}  // namespace roomie
