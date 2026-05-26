#include "roomie/pipeline/detection_bridge_thread.hpp"

#include <thread>

#include "roomie/pipeline/image_utils.hpp"

namespace roomie {

DetectionBridgeThread::DetectionBridgeThread(
    ThreadSafeQueue<DetectionFrame>& detection_queue,
    ThreadSafeQueue<InferenceResponse>& response_queue,
    MapProjector& map_projector,
    InferenceBackend& inference_backend,
    PipelineConfig config)
    : WorkerThread("detection_bridge_thread"),
      detection_queue_(detection_queue),
      response_queue_(response_queue),
      map_projector_(map_projector),
      inference_backend_(inference_backend),
      config_(std::move(config)),
      last_request_time_(std::chrono::steady_clock::time_point::min()) {}

void DetectionBridgeThread::run() {
  while (!stopRequested()) {
    forwardBackendResponses();

    DetectionFrame frame;
    if (!detection_queue_.waitPopFor(&frame, std::chrono::milliseconds(20))) {
      continue;
    }
    if (!readyForNextRequest()) {
      continue;
    }

    std::optional<PatchDepth> patch_depth = map_projector_.projectPatchDepth(frame);
    if (!patch_depth ||
        !patch_depth->hasMinimumCoverage(config_.min_patch_coverage_ratio)) {
      continue;
    }

    if (inference_backend_.enqueueRequest(makeRequest(frame, std::move(*patch_depth)))) {
      last_request_time_ = std::chrono::steady_clock::now();
    }
  }
}

bool DetectionBridgeThread::readyForNextRequest() {
  if (config_.max_inference_fps <= 0.0) {
    return true;
  }
  const auto now = std::chrono::steady_clock::now();
  const auto min_period = std::chrono::duration<double>(1.0 / config_.max_inference_fps);
  return now - last_request_time_ >= min_period;
}

InferenceRequest DetectionBridgeThread::makeRequest(const DetectionFrame& frame,
                                                    PatchDepth patch_depth) const {
  InferenceRequest request;
  request.time_ns = frame.time_ns;
  request.camera_id = frame.camera_id;
  request.rgb_960 =
      resizeBilinear(frame.rgb, config_.boxer_input_size, config_.boxer_input_size);
  request.mask_960 =
      resizeNearest(frame.robot_mask, config_.boxer_input_size, config_.boxer_input_size);
  request.patch_depth = std::move(patch_depth);
  request.intrinsics_960 =
      frame.intrinsics.scaledTo(config_.boxer_input_size, config_.boxer_input_size);
  request.T_world_camera = frame.T_world_camera;
  return request;
}

void DetectionBridgeThread::forwardBackendResponses() {
  InferenceResponse response;
  while (inference_backend_.tryPopResponse(&response)) {
    response_queue_.pushDropOldest(std::move(response));
  }
}

}  // namespace roomie
