#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "roomie/pipeline/python_inference_backend.hpp"

#ifndef ROOMIE_FAKE_INFERENCE_WORKER_PATH
#error "ROOMIE_FAKE_INFERENCE_WORKER_PATH must identify the adversarial worker"
#endif

namespace roomie {
namespace {

using namespace std::chrono_literals;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    char path[] = "/tmp/roomie_python_backend_test_XXXXXX";
    char* created = ::mkdtemp(path);
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
    marker_path_ = path_ + "/worker-ready";
  }

  ~TemporaryDirectory() {
    ::unlink(marker_path_.c_str());
    ::rmdir(path_.c_str());
  }

  const std::string& markerPath() const { return marker_path_; }

 private:
  std::string path_;
  std::string marker_path_;
};

class ScopedEnvironment {
 public:
  ScopedEnvironment(std::string name, std::string value)
      : name_(std::move(name)) {
    const char* previous = ::getenv(name_.c_str());
    if (previous != nullptr) {
      previous_ = previous;
    }
    if (::setenv(name_.c_str(), value.c_str(), 1) != 0) {
      throw std::runtime_error("setenv failed for " + name_);
    }
  }

  ~ScopedEnvironment() {
    if (previous_) {
      ::setenv(name_.c_str(), previous_->c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

class BackendStopGuard {
 public:
  explicit BackendStopGuard(PythonInferenceBackend* backend)
      : backend_(backend) {}

  ~BackendStopGuard() { backend_->stop(); }

 private:
  PythonInferenceBackend* backend_;
};

class ScopedSignalDisposition {
 public:
  ScopedSignalDisposition(int signal_number, void (*handler)(int))
      : signal_number_(signal_number) {
    struct sigaction action {};
    action.sa_handler = handler;
    ::sigemptyset(&action.sa_mask);
    if (::sigaction(signal_number_, &action, &previous_) != 0) {
      throw std::runtime_error("sigaction failed");
    }
  }

  ~ScopedSignalDisposition() {
    ::sigaction(signal_number_, &previous_, nullptr);
  }

 private:
  int signal_number_;
  struct sigaction previous_ {};
};

template <typename Predicate>
bool waitUntil(Predicate predicate, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(5ms);
  } while (std::chrono::steady_clock::now() < deadline);
  return predicate();
}

PipelineConfig makeConfig() {
  PipelineConfig config;
  config.python_backend_enabled = true;
  config.python_executable = "python3";
  config.python_worker_script = ROOMIE_FAKE_INFERENCE_WORKER_PATH;
  config.inference_request_queue_size = 2;
  config.inference_response_queue_size = 2;
  return config;
}

InferenceRequest makeRequest(RequestId request_id, std::size_t rgb_bytes = 3) {
  InferenceRequest request;
  request.time_ns = 123456789;
  request.provenance.run_id = {0x1234U, 0x5678U};
  request.provenance.frame_id = 42;
  request.provenance.request_id = request_id;
  request.provenance.sensor_time_ns = request.time_ns;
  request.provenance.map.map_epoch = {0x9abcU, 0xdef0U};
  request.provenance.map.map_revision = 11;
  request.provenance.map.integrated_through_ns = request.time_ns;
  request.provenance.surface.map_epoch = request.provenance.map.map_epoch;
  request.provenance.surface.surface_revision = 12;
  request.provenance.surface.source_map_revision = 11;
  request.camera_id = "test-camera";
  request.rgb_960.width = static_cast<int>(rgb_bytes);
  request.rgb_960.height = 1;
  request.rgb_960.channels = 1;
  request.rgb_960.encoding = "mono8";
  request.rgb_960.data.assign(rgb_bytes, 0x5aU);
  request.mask_960.width = 1;
  request.mask_960.height = 1;
  request.mask_960.channels = 1;
  request.mask_960.encoding = "mono8";
  request.mask_960.data = {0U};
  request.patch_depth.provenance = request.provenance;
  return request;
}

bool waitForResponse(PythonInferenceBackend* backend,
                     InferenceResponse* response,
                     std::chrono::milliseconds timeout) {
  return waitUntil([&]() { return backend->tryPopResponse(response); }, timeout);
}

TEST(PythonInferenceBackendShutdown, HungWorkerIsCancelledWithinBound) {
  TemporaryDirectory temporary;
  ScopedEnvironment mode("ROOMIE_FAKE_WORKER_MODE", "hang_after_request");
  ScopedEnvironment marker("ROOMIE_FAKE_WORKER_MARKER",
                           temporary.markerPath());
  PythonInferenceBackend backend(makeConfig());
  BackendStopGuard stop_guard(&backend);
  backend.start();

  const InferenceRequest request = makeRequest(7001);
  PushResult<InferenceRequest> admitted = backend.enqueueRequest(request);
  ASSERT_TRUE(admitted.accepted());
  ASSERT_TRUE(waitUntil(
      [&]() { return ::access(temporary.markerPath().c_str(), F_OK) == 0; },
      5s));

  InferenceRequest queued_request = makeRequest(7002);
  queued_request.camera_id = "second-test-camera";
  PushResult<InferenceRequest> queued_admission =
      backend.enqueueRequest(queued_request);
  ASSERT_TRUE(queued_admission.accepted());

  const auto shutdown_start = std::chrono::steady_clock::now();
  backend.beginShutdown();

  bool saw_active_request = false;
  bool saw_queued_request = false;
  for (int i = 0; i < 2; ++i) {
    InferenceResponse response;
    const bool received = waitForResponse(&backend, &response, 2s);
    EXPECT_TRUE(received);
    if (!received) {
      continue;
    }
    EXPECT_FALSE(response.ok);
    EXPECT_FALSE(response.error.empty());
    if (response.provenance.request_id == request.provenance.request_id) {
      saw_active_request = true;
      EXPECT_TRUE(response.provenance == request.provenance);
      EXPECT_EQ(response.time_ns, request.time_ns);
      EXPECT_EQ(response.camera_id, request.camera_id);
    } else if (response.provenance.request_id ==
               queued_request.provenance.request_id) {
      saw_queued_request = true;
      EXPECT_TRUE(response.provenance == queued_request.provenance);
      EXPECT_EQ(response.time_ns, queued_request.time_ns);
      EXPECT_EQ(response.camera_id, queued_request.camera_id);
    } else {
      ADD_FAILURE() << "unexpected terminal response request_id="
                    << response.provenance.request_id;
    }
  }
  EXPECT_TRUE(saw_active_request);
  EXPECT_TRUE(saw_queued_request);
  EXPECT_TRUE(waitUntil([&]() { return backend.idle(); }, 2s));

  PushResult<InferenceRequest> rejected = backend.enqueueRequest(makeRequest(7003));
  EXPECT_EQ(rejected.outcome, PushOutcome::kStopped);
  EXPECT_TRUE(rejected.unconsumed_item.has_value());

  backend.stop();
  const auto shutdown_elapsed = std::chrono::steady_clock::now() - shutdown_start;
  EXPECT_LT(shutdown_elapsed, 2s);
}

TEST(PythonInferenceBackendConfig, LabelThresholdFilesAreForwarded) {
  TemporaryDirectory temporary;
  ScopedEnvironment mode("ROOMIE_FAKE_WORKER_MODE", "hang_after_request");
  ScopedEnvironment marker("ROOMIE_FAKE_WORKER_MARKER",
                           temporary.markerPath());
  ScopedEnvironment expected(
      "ROOMIE_FAKE_WORKER_EXPECT_LABEL_THRESHOLD_FILE", __FILE__);
  PipelineConfig config = makeConfig();
  config.label_thresholds_file = __FILE__;
  PythonInferenceBackend backend(std::move(config));
  BackendStopGuard stop_guard(&backend);
  backend.start();

  ASSERT_TRUE(backend.enqueueRequest(makeRequest(7101)).accepted());
  EXPECT_TRUE(waitUntil(
      [&]() { return ::access(temporary.markerPath().c_str(), F_OK) == 0; },
      5s));
}

TEST(PythonInferenceBackendAdmission,
     QueuedDeadlineIsRecheckedBeforeStartingWorker) {
  TemporaryDirectory temporary;
  ScopedEnvironment mode("ROOMIE_FAKE_WORKER_MODE", "hang_after_request");
  ScopedEnvironment marker("ROOMIE_FAKE_WORKER_MARKER",
                           temporary.markerPath());
  PythonInferenceBackend backend(makeConfig());
  BackendStopGuard stop_guard(&backend);

  InferenceRequest request = makeRequest(7501);
  request.ingest_time = std::chrono::steady_clock::now();
  request.due_time = request.ingest_time + 25ms;
  ASSERT_TRUE(backend.enqueueRequest(request).accepted());
  std::this_thread::sleep_for(50ms);
  backend.start();

  InferenceResponse response;
  ASSERT_TRUE(waitForResponse(&backend, &response, 2s));
  EXPECT_FALSE(response.ok);
  EXPECT_EQ(response.provenance, request.provenance);
  EXPECT_NE(response.error.find("deadline expired"), std::string::npos)
      << response.error;
  EXPECT_GT(response.timing.worker_queue_ms, 0.0);
  EXPECT_NE(::access(temporary.markerPath().c_str(), F_OK), 0)
      << "expired request unexpectedly started the Python worker";
  EXPECT_TRUE(waitUntil([&]() { return backend.idle(); }, 2s));

  backend.beginShutdown();
  backend.stop();
}

TEST(PythonInferenceBackendIpc, ClosedWorkerPipeReportsEpipeWithoutSigpipe) {
  TemporaryDirectory temporary;
  ScopedEnvironment mode("ROOMIE_FAKE_WORKER_MODE", "close_stdin");
  ScopedEnvironment marker("ROOMIE_FAKE_WORKER_MARKER",
                           temporary.markerPath());
  ScopedSignalDisposition sigpipe_disposition(SIGPIPE, SIG_DFL);
  PythonInferenceBackend backend(makeConfig());
  BackendStopGuard stop_guard(&backend);
  backend.start();

  // A body larger than the pipe capacity guarantees that a worker closing its
  // read end races into write(2), rather than letting the whole request flush.
  const InferenceRequest request = makeRequest(8001, 4U * 1024U * 1024U);
  PushResult<InferenceRequest> admitted = backend.enqueueRequest(request);
  ASSERT_TRUE(admitted.accepted());
  ASSERT_TRUE(waitUntil(
      [&]() { return ::access(temporary.markerPath().c_str(), F_OK) == 0; },
      5s));

  InferenceResponse response;
  const bool received = waitForResponse(&backend, &response, 5s);
  EXPECT_TRUE(received);
  if (received) {
    EXPECT_FALSE(response.ok);
    EXPECT_TRUE(response.provenance == request.provenance);
    EXPECT_EQ(response.time_ns, request.time_ns);
    EXPECT_EQ(response.camera_id, request.camera_id);
    EXPECT_NE(response.error.find("EPIPE"), std::string::npos)
        << response.error;
  }
  EXPECT_TRUE(waitUntil([&]() { return backend.idle(); }, 2s));

  backend.beginShutdown();
  backend.stop();
}

}  // namespace
}  // namespace roomie
