#pragma once

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

class PythonInferenceBackend : public WorkerThread, public InferenceBackend {
 public:
  explicit PythonInferenceBackend(PipelineConfig config);

  bool enqueueRequest(InferenceRequest request) override;
  bool tryPopResponse(InferenceResponse* response) override;

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  ThreadSafeQueue<InferenceRequest> request_queue_;
  ThreadSafeQueue<InferenceResponse> response_queue_;
  PipelineConfig config_;
};

}  // namespace roomie
