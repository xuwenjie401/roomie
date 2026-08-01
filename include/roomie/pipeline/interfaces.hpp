#pragma once

#include <memory>
#include <optional>
#include <filesystem>
#include <vector>

#include "roomie/dsg/object_graph.hpp"
#include "roomie/pipeline/map_backend.hpp"
#include "roomie/pipeline/types.hpp"

namespace roomie {

class MapProjector {
 public:
  virtual ~MapProjector() = default;

  virtual bool enqueueMappingFrame(MappingFrame frame) = 0;
  virtual std::optional<PatchDepth> projectPatchDepth(const DetectionFrame& frame) = 0;
  virtual MapBackendSnapshot snapshotSurfacePoints() const = 0;
  virtual std::shared_ptr<const GeometrySurfaceCache> geometrySurfaceCache() const = 0;
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
  virtual std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
  snapshotTrackedInstances() const = 0;
  virtual ObjectGraphSnapshot snapshotObjectGraph() const = 0;
  virtual bool prepareSceneGraphForSave(ObjectGraphSnapshot* snapshot,
                                        const std::filesystem::path& snapshot_image_dir,
                                        const std::string& snapshot_uri_prefix,
                                        std::string* error) const = 0;
};

}  // namespace roomie
