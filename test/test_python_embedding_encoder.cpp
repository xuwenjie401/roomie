#include <unistd.h>

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "roomie/artifacts/python_embedding_encoder.hpp"

#ifndef ROOMIE_FAKE_EMBEDDING_WORKER_PATH
#error "ROOMIE_FAKE_EMBEDDING_WORKER_PATH must identify the fake worker"
#endif

namespace roomie {
namespace {

using namespace std::chrono_literals;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    char path[] = "/tmp/roomie_embedding_encoder_test_XXXXXX";
    char* created = ::mkdtemp(path);
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    ::unlink(markerPath().c_str());
    ::unlink(startCountPath().c_str());
    ::rmdir(path_.c_str());
  }

  std::string markerPath() const { return path_ + "/marker"; }
  std::string startCountPath() const { return path_ + "/starts"; }

 private:
  std::string path_;
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
    (void)::sigaction(signal_number_, &previous_, nullptr);
  }

 private:
  int signal_number_;
  struct sigaction previous_ {};
};

template <typename Predicate>
bool waitUntil(Predicate predicate, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(5ms);
  }
  return predicate();
}

PythonEmbeddingEncoderConfig makeConfig() {
  PythonEmbeddingEncoderConfig config;
  config.python_executable = "python3";
  config.worker_script = ROOMIE_FAKE_EMBEDDING_WORKER_PATH;
  config.model_path = "unused-fake-model";
  config.model_id = "fake-model-v1";
  config.device = "cpu";
  config.expected_dimension = 3;
  config.maximum_batch_size = 8;
  config.maximum_document_bytes = 4U * 1024U * 1024U;
  config.maximum_message_bytes = 8U * 1024U * 1024U;
  config.startup_timeout = 2s;
  config.request_timeout = 3s;
  config.shutdown_timeout = 100ms;
  config.poll_period = 10ms;
  return config;
}

double norm(const std::vector<float>& vector) {
  double squared = 0.0;
  for (float component : vector) {
    squared += static_cast<double>(component) * component;
  }
  return std::sqrt(squared);
}

int readStartCount(const std::string& path) {
  std::ifstream stream(path);
  int count = 0;
  stream >> count;
  return count;
}

std::string captureEncodeError(PythonEmbeddingEncoder* encoder,
                               const std::string& document) {
  try {
    (void)encoder->encodeOne(document);
  } catch (const std::exception& exception) {
    return exception.what();
  }
  return {};
}

TEST(PythonEmbeddingEncoder, PrewarmsOnceAndSharesNormalizedBatchEncoder) {
  TemporaryDirectory temporary;
  ScopedEnvironment mode("ROOMIE_FAKE_EMBEDDING_MODE", "normal");
  ScopedEnvironment count("ROOMIE_FAKE_EMBEDDING_START_COUNT",
                          temporary.startCountPath());
  PythonEmbeddingEncoder encoder(makeConfig());

  EXPECT_EQ(encoder.modelId(), "fake-model-v1");
  EXPECT_EQ(encoder.dimension(), 3U);
  encoder.prewarm();
  encoder.prewarm();

  const std::vector<float> query = encoder.encodeOne("chair");
  const auto batch = encoder.encodeBatch({"table", "lamp"});
  ASSERT_EQ(query.size(), 3U);
  ASSERT_EQ(batch.size(), 2U);
  EXPECT_NEAR(norm(query), 1.0, 1e-6);
  EXPECT_NEAR(norm(batch[0]), 1.0, 1e-6);
  EXPECT_NEAR(norm(batch[1]), 1.0, 1e-6);
  EXPECT_EQ(readStartCount(temporary.startCountPath()), 1);

  const PythonEmbeddingEncoderStats stats = encoder.stats();
  EXPECT_TRUE(stats.worker_running);
  EXPECT_EQ(stats.worker_starts, 1U);
  EXPECT_EQ(stats.worker_restarts, 0U);
  EXPECT_EQ(stats.requests, 2U);
  EXPECT_EQ(stats.encoded_documents, 3U);
  EXPECT_EQ(stats.request_failures, 0U);
}

TEST(PythonEmbeddingEncoder, RetriesOneIdempotentBatchAfterChildCrash) {
  TemporaryDirectory temporary;
  ScopedEnvironment mode("ROOMIE_FAKE_EMBEDDING_MODE", "crash_once");
  ScopedEnvironment marker("ROOMIE_FAKE_EMBEDDING_MARKER",
                           temporary.markerPath());
  ScopedEnvironment count("ROOMIE_FAKE_EMBEDDING_START_COUNT",
                          temporary.startCountPath());
  PythonEmbeddingEncoder encoder(makeConfig());

  const auto vectors = encoder.encodeBatch({"sofa", "plant"});
  ASSERT_EQ(vectors.size(), 2U);
  EXPECT_NEAR(norm(vectors[0]), 1.0, 1e-6);
  EXPECT_EQ(readStartCount(temporary.startCountPath()), 2);

  const PythonEmbeddingEncoderStats stats = encoder.stats();
  EXPECT_EQ(stats.worker_starts, 2U);
  EXPECT_EQ(stats.worker_restarts, 1U);
  EXPECT_EQ(stats.transport_failures, 1U);
  EXPECT_EQ(stats.requests, 1U);
  EXPECT_EQ(stats.encoded_documents, 2U);
}

TEST(PythonEmbeddingEncoder, ExplicitWorkerErrorIsTerminalAndNotRetried) {
  TemporaryDirectory temporary;
  ScopedEnvironment mode("ROOMIE_FAKE_EMBEDDING_MODE", "worker_error");
  ScopedEnvironment count("ROOMIE_FAKE_EMBEDDING_START_COUNT",
                          temporary.startCountPath());
  PythonEmbeddingEncoder encoder(makeConfig());

  const std::string error = captureEncodeError(&encoder, "bad document");
  EXPECT_NE(error.find("synthetic model failure"), std::string::npos) << error;
  EXPECT_EQ(readStartCount(temporary.startCountPath()), 1);
  const PythonEmbeddingEncoderStats stats = encoder.stats();
  EXPECT_EQ(stats.worker_starts, 1U);
  EXPECT_EQ(stats.worker_restarts, 0U);
  EXPECT_EQ(stats.request_failures, 1U);
  EXPECT_EQ(stats.transport_failures, 0U);
}

TEST(PythonEmbeddingEncoder, CrashSafeShutdownTerminatesActiveAndQueuedCalls) {
  TemporaryDirectory temporary;
  ScopedEnvironment mode("ROOMIE_FAKE_EMBEDDING_MODE", "hang_after_request");
  ScopedEnvironment marker("ROOMIE_FAKE_EMBEDDING_MARKER",
                           temporary.markerPath());
  PythonEmbeddingEncoderConfig config = makeConfig();
  config.request_timeout = 10s;
  config.shutdown_timeout = 80ms;
  PythonEmbeddingEncoder encoder(config);

  auto active = std::async(std::launch::async, [&]() {
    return captureEncodeError(&encoder, "active");
  });
  ASSERT_TRUE(waitUntil(
      [&]() { return ::access(temporary.markerPath().c_str(), F_OK) == 0; },
      2s));
  auto queued = std::async(std::launch::async, [&]() {
    return captureEncodeError(&encoder, "queued");
  });

  const auto shutdown_start = std::chrono::steady_clock::now();
  encoder.shutdown();
  const auto elapsed = std::chrono::steady_clock::now() - shutdown_start;
  EXPECT_LT(elapsed, 1s);
  ASSERT_EQ(active.wait_for(2s), std::future_status::ready);
  ASSERT_EQ(queued.wait_for(2s), std::future_status::ready);
  EXPECT_FALSE(active.get().empty());
  EXPECT_FALSE(queued.get().empty());
  EXPECT_TRUE(encoder.stats().shut_down);

  const std::string after_shutdown = captureEncodeError(&encoder, "late");
  EXPECT_NE(after_shutdown.find("shut down"), std::string::npos)
      << after_shutdown;
}

TEST(PythonEmbeddingEncoder, ClosedPipeReportsEpipeWithoutSigpipeTermination) {
  TemporaryDirectory temporary;
  ScopedEnvironment mode("ROOMIE_FAKE_EMBEDDING_MODE", "close_stdin");
  ScopedEnvironment marker("ROOMIE_FAKE_EMBEDDING_MARKER",
                           temporary.markerPath());
  ScopedSignalDisposition signal_disposition(SIGPIPE, SIG_DFL);
  PythonEmbeddingEncoder encoder(makeConfig());
  encoder.prewarm();
  ASSERT_TRUE(waitUntil(
      [&]() { return ::access(temporary.markerPath().c_str(), F_OK) == 0; },
      2s));

  const std::string large_document(2U * 1024U * 1024U, 'x');
  const std::string error = captureEncodeError(&encoder, large_document);
  EXPECT_NE(error.find("EPIPE"), std::string::npos) << error;
  EXPECT_GE(encoder.stats().transport_failures, 1U);
}

TEST(PythonEmbeddingEncoder, StartupTimeoutAndForcedKillAreBounded) {
  TemporaryDirectory temporary;
  ScopedEnvironment mode("ROOMIE_FAKE_EMBEDDING_MODE", "hang_before_ready");
  ScopedEnvironment marker("ROOMIE_FAKE_EMBEDDING_MARKER",
                           temporary.markerPath());
  PythonEmbeddingEncoderConfig config = makeConfig();
  config.startup_timeout = 120ms;
  config.shutdown_timeout = 60ms;
  PythonEmbeddingEncoder encoder(config);

  const auto start = std::chrono::steady_clock::now();
  std::string error;
  try {
    encoder.prewarm();
  } catch (const std::exception& exception) {
    error = exception.what();
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_NE(error.find("timed out"), std::string::npos) << error;
  EXPECT_LT(elapsed, 1s);
  EXPECT_EQ(encoder.stats().timeouts, 1U);
}

}  // namespace
}  // namespace roomie
