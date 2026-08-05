#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "roomie/artifacts/semantic_index.hpp"

namespace roomie {

// Configuration for the persistent SentenceTransformer subprocess. The
// expected namespace is deliberately known before startup: durable embedding
// tasks and semantic-index generations must never depend on a model process
// being alive merely to identify their vector namespace.
struct PythonEmbeddingEncoderConfig {
  std::string python_executable = "python3";
  std::string worker_script = "roomie_sentence_transformer_worker.py";
  std::string model_path;
  std::string model_id;
  std::string device = "cuda";
  std::size_t expected_dimension = 0;
  std::size_t maximum_batch_size = 64;
  std::size_t maximum_document_bytes = 1024U * 1024U;
  std::size_t maximum_message_bytes = 16U * 1024U * 1024U;
  std::chrono::milliseconds startup_timeout{120000};
  std::chrono::milliseconds request_timeout{120000};
  std::chrono::milliseconds shutdown_timeout{500};
  std::chrono::milliseconds poll_period{20};
};

struct PythonEmbeddingEncoderStats {
  bool worker_running = false;
  bool shut_down = false;
  std::uint64_t worker_starts = 0;
  std::uint64_t worker_restarts = 0;
  std::uint64_t requests = 0;
  std::uint64_t encoded_documents = 0;
  std::uint64_t request_failures = 0;
  std::uint64_t transport_failures = 0;
  std::uint64_t timeouts = 0;
  std::string last_error;
};

// Thread-safe, serialized access to one long-lived Python model process.
// encodeBatch() is the EmbeddingEncoder seam used by the durable document
// worker; encodeOne() intentionally uses the same process for query vectors.
// Transport failures are retried once because encoding is side-effect free.
class PythonEmbeddingEncoder final : public EmbeddingEncoder {
 public:
  explicit PythonEmbeddingEncoder(PythonEmbeddingEncoderConfig config);
  ~PythonEmbeddingEncoder() override;

  PythonEmbeddingEncoder(const PythonEmbeddingEncoder&) = delete;
  PythonEmbeddingEncoder& operator=(const PythonEmbeddingEncoder&) = delete;

  std::string modelId() const override;
  std::size_t dimension() const override;
  void prewarm() override;
  std::vector<std::vector<float>> encodeBatch(
      const std::vector<std::string>& documents) override;
  std::vector<float> encodeOne(const std::string& document);

  // Idempotent. Active IPC is interrupted by terminating the child; callers
  // waiting for their serialized turn receive a terminal exception.
  void shutdown();
  PythonEmbeddingEncoderStats stats() const;

 private:
  enum class ReplyKind { kSuccess, kWorkerError, kTransportError };

  struct Reply {
    ReplyKind kind = ReplyKind::kTransportError;
    std::vector<std::vector<float>> embeddings;
    std::string error;
  };

  static PythonEmbeddingEncoderConfig validateConfig(
      PythonEmbeddingEncoderConfig config);

  bool ensureWorker(std::chrono::steady_clock::time_point deadline,
                    std::string* error);
  bool spawnWorker(std::chrono::steady_clock::time_point deadline,
                   std::string* error);
  void stopWorker(bool count_transport_failure);
  Reply transact(const std::vector<std::string>& documents,
                 std::chrono::steady_clock::time_point deadline);

  bool writeFrame(const std::string& payload,
                  std::chrono::steady_clock::time_point deadline,
                  std::string* error);
  bool readFrame(std::string* payload,
                 std::chrono::steady_clock::time_point deadline,
                 std::string* error);
  bool writeExact(int fd,
                  const void* data,
                  std::size_t size,
                  std::chrono::steady_clock::time_point deadline,
                  std::string* error);
  bool readExact(int fd,
                 void* data,
                 std::size_t size,
                 std::chrono::steady_clock::time_point deadline,
                 std::string* error);
  bool waitForFd(int fd,
                 short events,
                 std::chrono::steady_clock::time_point deadline,
                 std::string* error);
  void recordError(const std::string& error);

  const PythonEmbeddingEncoderConfig config_;
  const std::string model_id_;
  const std::size_t dimension_;

  mutable std::timed_mutex operation_mutex_;
  mutable std::mutex shutdown_mutex_;
  mutable std::mutex process_mutex_;
  int worker_pid_ = -1;
  int worker_stdin_fd_ = -1;
  int worker_stdout_fd_ = -1;
  bool worker_ready_ = false;
  bool ever_started_ = false;

  std::atomic_bool shutdown_requested_{false};
  std::atomic_uint64_t next_request_id_{1};
  std::atomic_uint64_t worker_starts_{0};
  std::atomic_uint64_t worker_restarts_{0};
  std::atomic_uint64_t requests_{0};
  std::atomic_uint64_t encoded_documents_{0};
  std::atomic_uint64_t request_failures_{0};
  std::atomic_uint64_t transport_failures_{0};
  std::atomic_uint64_t timeouts_{0};
  mutable std::mutex error_mutex_;
  std::string last_error_;
};

}  // namespace roomie
