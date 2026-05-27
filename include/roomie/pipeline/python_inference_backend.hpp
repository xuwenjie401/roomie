#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <string>
#include <vector>

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
  bool ensureWorkerProcess(std::string* error);
  void stopWorkerProcess();
  bool transactWithWorker(const InferenceRequest& request, InferenceResponse* response);

  std::vector<std::uint8_t> serializeRequest(const InferenceRequest& request) const;
  bool parseResponse(const std::vector<std::uint8_t>& body,
                     InferenceResponse* response,
                     std::string* error) const;
  bool writeMessage(const std::vector<std::uint8_t>& body);
  bool readMessage(std::vector<std::uint8_t>* body);

  static bool writeExact(int fd, const void* data, std::size_t size);
  static bool readExact(int fd, void* data, std::size_t size);

  ThreadSafeQueue<InferenceRequest> request_queue_;
  ThreadSafeQueue<InferenceResponse> response_queue_;
  PipelineConfig config_;
  std::atomic_int worker_pid_{-1};
  std::atomic_int worker_stdin_fd_{-1};
  std::atomic_int worker_stdout_fd_{-1};
  int consecutive_ipc_failures_ = 0;
  bool backend_disabled_after_failure_ = false;
};

}  // namespace roomie
