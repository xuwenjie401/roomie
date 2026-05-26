#include "roomie/pipeline/ros_io_thread.hpp"

#include <chrono>
#include <thread>

namespace roomie {

RosIoThread::RosIoThread(ThreadSafeQueue<MappingFrame>& mapping_queue,
                         ThreadSafeQueue<DetectionFrame>& detection_queue)
    : WorkerThread("ros_io_thread"),
      mapping_queue_(mapping_queue),
      detection_queue_(detection_queue) {}

bool RosIoThread::enqueueMappingFrame(MappingFrame frame) {
  return mapping_queue_.pushDropOldest(std::move(frame));
}

bool RosIoThread::enqueueDetectionFrame(DetectionFrame frame) {
  return detection_queue_.pushDropOldest(std::move(frame));
}

void RosIoThread::run() {
  while (!stopRequested()) {
    // TODO: attach ROS subscriptions and convert callbacks into MappingFrame and
    // DetectionFrame packets. The thread exists now to make ownership explicit.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

}  // namespace roomie
