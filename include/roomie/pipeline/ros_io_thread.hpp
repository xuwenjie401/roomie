#pragma once

#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/types.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

class RosIoThread : public WorkerThread {
 public:
  RosIoThread(ThreadSafeQueue<MappingFrame>& mapping_queue,
              ThreadSafeQueue<DetectionFrame>& detection_queue);

  bool enqueueMappingFrame(MappingFrame frame);
  bool enqueueDetectionFrame(DetectionFrame frame);

 protected:
  void run() override;

 private:
  ThreadSafeQueue<MappingFrame>& mapping_queue_;
  ThreadSafeQueue<DetectionFrame>& detection_queue_;
};

}  // namespace roomie
