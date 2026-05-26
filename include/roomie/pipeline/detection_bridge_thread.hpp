#pragma once

#include <chrono>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

class DetectionBridgeThread : public WorkerThread {
 public:
  DetectionBridgeThread(ThreadSafeQueue<DetectionFrame>& detection_queue,
                        ThreadSafeQueue<InferenceResponse>& response_queue,
                        MapProjector& map_projector,
                        InferenceBackend& inference_backend,
                        PipelineConfig config);

 protected:
  void run() override;

 private:
  bool readyForNextRequest();
  InferenceRequest makeRequest(const DetectionFrame& frame, PatchDepth patch_depth) const;
  void forwardBackendResponses();

  ThreadSafeQueue<DetectionFrame>& detection_queue_;
  ThreadSafeQueue<InferenceResponse>& response_queue_;
  MapProjector& map_projector_;
  InferenceBackend& inference_backend_;
  PipelineConfig config_;
  std::chrono::steady_clock::time_point last_request_time_;
};

}  // namespace roomie
