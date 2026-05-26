#include "roomie/pipeline/python_inference_backend.hpp"

#include <chrono>

namespace roomie {

PythonInferenceBackend::PythonInferenceBackend(PipelineConfig config)
    : WorkerThread("python_inference_backend"),
      request_queue_(config.inference_request_queue_size),
      response_queue_(config.inference_response_queue_size),
      config_(std::move(config)) {}

bool PythonInferenceBackend::enqueueRequest(InferenceRequest request) {
  return request_queue_.pushDropOldest(std::move(request));
}

bool PythonInferenceBackend::tryPopResponse(InferenceResponse* response) {
  return response_queue_.tryPop(response);
}

void PythonInferenceBackend::run() {
  while (!stopRequested()) {
    InferenceRequest request;
    if (!request_queue_.waitPopFor(&request, std::chrono::milliseconds(50))) {
      continue;
    }

    InferenceResponse response;
    response.time_ns = request.time_ns;
    response.camera_id = request.camera_id;
    response.ok = false;
    response.error = "Python shared-memory backend is not connected yet";
    response_queue_.pushDropOldest(std::move(response));
  }
}

void PythonInferenceBackend::onStopRequested() {
  request_queue_.stop();
  response_queue_.stop();
}

}  // namespace roomie
