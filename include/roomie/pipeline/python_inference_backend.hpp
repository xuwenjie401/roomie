#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "roomie/pipeline/interfaces.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/pipeline/worker_thread.hpp"

namespace roomie {

// Stateless IPC v2 codecs are public so protocol compatibility can be tested
// without spawning or loading the Python inference worker.
namespace inference_wire {

constexpr std::uint32_t kMaxMessageBytes = 64U * 1024U * 1024U;

bool encodeRequest(const InferenceRequest& request,
                   std::vector<std::uint8_t>* body,
                   std::string* error);
bool decodeRequest(const std::vector<std::uint8_t>& body,
                   InferenceRequest* request,
                   std::string* error);
bool encodeResponse(const InferenceResponse& response,
                    std::vector<std::uint8_t>* body,
                    std::string* error);
bool decodeResponse(const std::vector<std::uint8_t>& body,
                    InferenceResponse* response,
                    std::string* error);

}  // namespace inference_wire

class PythonInferenceBackend : public WorkerThread, public InferenceBackend {
 public:
  explicit PythonInferenceBackend(PipelineConfig config);

  PushResult<InferenceRequest> enqueueRequest(InferenceRequest request) override;
  bool tryPopResponse(InferenceResponse* response) override;
  bool idle() const override;
  ChannelStats requestChannelStats() const override;
  ChannelStats responseChannelStats() const override;
  void beginShutdown() override;

 protected:
  void run() override;
  void onStopRequested() override;

 private:
  bool ensureWorkerProcess(std::string* error);
  void stopWorkerProcess();
  bool enqueueResponse(InferenceResponse response);
  bool transactWithWorker(const InferenceRequest& request, InferenceResponse* response);

  bool writeMessage(const std::vector<std::uint8_t>& body, std::string* error);
  bool readMessage(std::vector<std::uint8_t>* body, std::string* error);
  bool waitForWorkerReadable(int fd, std::string* error) const;

  static bool writeExact(int fd,
                         const void* data,
                         std::size_t size,
                         int* error_number);
  static bool readExact(int fd,
                        void* data,
                        std::size_t size,
                        int* error_number);

  ThreadSafeQueue<InferenceRequest> request_queue_;
  ThreadSafeQueue<InferenceResponse> response_queue_;
  PipelineConfig config_;
  mutable std::mutex request_accounting_mutex_;
  mutable std::mutex worker_process_mutex_;
  std::atomic_int worker_pid_{-1};
  std::atomic_int worker_stdin_fd_{-1};
  std::atomic_int worker_stdout_fd_{-1};
  int consecutive_ipc_failures_ = 0;
  bool backend_disabled_after_failure_ = false;
  std::atomic_bool processing_request_{false};
  std::atomic_bool shutdown_requested_{false};
  std::atomic_uint64_t outstanding_requests_{0};
};

}  // namespace roomie
