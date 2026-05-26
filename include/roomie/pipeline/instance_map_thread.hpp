#pragma once

#include <mutex>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

class InstanceMapThread : public WorkerThread, public InstanceStore {
 public:
  InstanceMapThread(ThreadSafeQueue<InferenceResponse>& response_queue,
                    const MapProjector& map_projector,
                    PipelineConfig config);

  bool enqueueDetections(InferenceResponse response) override;
  std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>>
  snapshotInstances() const override;

 protected:
  void run() override;

 private:
  void applyDetections(const InferenceResponse& response);

  ThreadSafeQueue<InferenceResponse>& response_queue_;
  const MapProjector& map_projector_;
  PipelineConfig config_;
  mutable std::mutex mutex_;
  int next_track_id_ = 0;
  std::vector<InstanceRecord, Eigen::aligned_allocator<InstanceRecord>> instances_;
};

}  // namespace roomie
