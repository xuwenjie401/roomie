#pragma once

#include <optional>
#include <vector>

#include "roomie/pipeline/types.hpp"

namespace roomie {

class MapProjector {
 public:
  virtual ~MapProjector() = default;

  virtual bool enqueueMappingFrame(MappingFrame frame) = 0;
  virtual std::optional<PatchDepth> projectPatchDepth(const DetectionFrame& frame) = 0;
  virtual std::vector<VoxelRef, Eigen::aligned_allocator<VoxelRef>>
  collectNearSurfaceVoxels(const RawDetection& detection) const = 0;
};

class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;

  virtual bool enqueueRequest(InferenceRequest request) = 0;
  virtual bool tryPopResponse(InferenceResponse* response) = 0;
};

class InstanceStore {
 public:
  virtual ~InstanceStore() = default;

  virtual bool enqueueDetections(InferenceResponse response) = 0;
  virtual std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
  snapshotInstances() const = 0;
};

}  // namespace roomie
