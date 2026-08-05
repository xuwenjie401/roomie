#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

#include "roomie/artifacts/artifact_scheduler.hpp"
#include "roomie/artifacts/snapshot_bank.hpp"

namespace roomie {

// Configuration for the resident DAM subprocess. model_id and prompt_hash are
// execution fences: a durable task produced for a different model/prompt is
// rejected permanently instead of being run with unverifiable provenance.
struct PythonDamWorkerConfig {
  std::string python_executable = "python3";
  std::filesystem::path worker_script;
  std::filesystem::path dam_source;
  std::string model_path;
  std::string model_id = "roomie-dam-v1";
  // Optional expected canonicalDamExecutionPromptHash(); empty computes it.
  std::string prompt_hash;
  std::string conversation_mode = "v1";
  std::string prompt_mode = "full+focal_crop";
  std::string query =
      "<image>\nDescribe only the visible object inside the masked region in detail. "
      "Include its color, material, shape, parts, pose, and distinctive visual "
      "details. Do not describe unrelated background.";
  int max_new_tokens = 256;
  double temperature = 0.2;
  double top_p = 0.9;
  double bbox_pad_px = 2.0;

  std::chrono::milliseconds startup_timeout{120'000};
  std::chrono::milliseconds request_timeout{120'000};
  std::chrono::milliseconds shutdown_timeout{500};
  std::size_t max_message_bytes = 8U * 1024U * 1024U;
};

// Hashes every execution parameter that can change the resident DAM prompt or
// decoded output. prompt_hash itself and transport-only timeouts are excluded.
// The result is a lowercase 64-character SHA-256 digest.
std::string canonicalDamExecutionPromptHash(
    const PythonDamWorkerConfig& config);

// Synchronous, single-in-flight DamWorker backed by one persistent Python
// process. ArtifactRuntimeActor owns lease heartbeats independently while this
// adapter is waiting for the model. The adapter never trusts paths embedded in
// a durable task: every source_frame_asset_id is pinned and resolved through
// AssetStore, and only that immutable path is sent to Python.
class PythonDamWorker final : public DamWorker {
 public:
  PythonDamWorker(PythonDamWorkerConfig config,
                  std::shared_ptr<AssetStore> asset_store);
  ~PythonDamWorker() override;

  PythonDamWorker(const PythonDamWorker&) = delete;
  PythonDamWorker& operator=(const PythonDamWorker&) = delete;

  DamWorkerResponse describe(const DamTaskRequest& request,
                             const DamLeaseHeartbeat& heartbeat) override;

  // Interrupts an in-flight request and permanently closes this adapter.
  // Calls are idempotent and bounded by shutdown_timeout (plus scheduler
  // jitter); subsequent describe() calls return a retryable stopped error.
  void shutdown();

 private:
  struct StartupResult;
  struct PreparedRequest;

  StartupResult ensureWorkerProcess();
  void stopWorkerProcess();
  DamWorkerResponse transact(const PreparedRequest& request);

  bool writeJsonMessage(const std::string& body,
                        std::chrono::steady_clock::time_point deadline,
                        std::string* error);
  bool readJsonMessage(std::string* body,
                       std::chrono::steady_clock::time_point deadline,
                       std::string* error);
  bool writeExact(int fd,
                  const void* data,
                  std::size_t size,
                  std::chrono::steady_clock::time_point deadline,
                  std::string* error) const;
  bool readExact(int fd,
                 void* data,
                 std::size_t size,
                 std::chrono::steady_clock::time_point deadline,
                 std::string* error) const;

  PythonDamWorkerConfig config_;
  std::shared_ptr<AssetStore> asset_store_;
  std::mutex transaction_mutex_;
  std::mutex process_mutex_;
  std::atomic_bool shutdown_requested_{false};
  std::atomic_bool worker_ready_{false};
  std::atomic_int worker_pid_{-1};
  std::atomic_int worker_stdin_fd_{-1};
  std::atomic_int worker_stdout_fd_{-1};
  std::uint64_t next_request_id_ = 1;
};

}  // namespace roomie
