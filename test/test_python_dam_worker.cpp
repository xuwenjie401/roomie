#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "roomie/artifacts/python_dam_worker.hpp"

#ifndef ROOMIE_FAKE_DAM_WORKER_PATH
#error "ROOMIE_FAKE_DAM_WORKER_PATH must identify the fake DAM worker"
#endif

namespace roomie {
namespace {

using Json = nlohmann::json;
using namespace std::chrono_literals;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    char path[] = "/tmp/roomie_python_dam_test_XXXXXX";
    char* created = ::mkdtemp(path);
    if (!created) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  std::filesystem::path path(const std::string& child) const {
    return path_ / child;
  }

 private:
  std::filesystem::path path_;
};

class ScopedEnvironment {
 public:
  ScopedEnvironment(std::string name, std::string value)
      : name_(std::move(name)) {
    if (const char* previous = ::getenv(name_.c_str())) {
      previous_ = previous;
    }
    if (::setenv(name_.c_str(), value.c_str(), 1) != 0) {
      throw std::runtime_error("setenv failed");
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
  do {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(5ms);
  } while (std::chrono::steady_clock::now() < deadline);
  return predicate();
}

std::size_t lineCount(const std::filesystem::path& path) {
  std::ifstream stream(path);
  std::size_t count = 0;
  std::string line;
  while (std::getline(stream, line)) {
    ++count;
  }
  return count;
}

std::shared_ptr<AssetStore> makeStore(const std::filesystem::path& root) {
  AssetStoreConfig config;
  config.root = root;
  config.grace_period_ns = 60'000'000'000LL;
  auto store = std::make_shared<AssetStore>(config);
  if (!store->healthy()) {
    throw std::runtime_error(store->initializationError());
  }
  return store;
}

AssetWriteResult makeAsset(const std::shared_ptr<AssetStore>& store) {
  ImageBuffer image;
  image.width = 4;
  image.height = 3;
  image.channels = 3;
  image.encoding = "rgb8";
  image.data.resize(4U * 3U * 3U, 0x4aU);
  return store->materializeFrame(image, 1);
}

PythonDamWorkerConfig workerConfig() {
  PythonDamWorkerConfig config;
  config.python_executable = "python3";
  config.worker_script = ROOMIE_FAKE_DAM_WORKER_PATH;
  config.model_path = "fake/model";
  config.model_id = "fake-dam-v1";
  config.prompt_hash.clear();
  config.startup_timeout = 2s;
  config.request_timeout = 2s;
  config.shutdown_timeout = 120ms;
  return config;
}

DamTaskRequest requestFor(const std::string& asset_id) {
  DamTaskRequest request;
  request.key.object_id = 17;
  request.key.identity_revision = 2;
  request.key.appearance_revision = 3;
  request.key.snapshot_set_hash = "snapshot-set-17";
  request.key.model_id = "fake-dam-v1";
  request.key.prompt_hash = canonicalDamExecutionPromptHash(workerConfig());
  request.key.output_schema_version = "roomie.dam.v1";
  request.dependency.object_id = request.key.object_id;
  request.dependency.identity_revision = request.key.identity_revision;
  request.dependency.appearance_revision = request.key.appearance_revision;
  request.scene_revision = 9;
  request.priority = ArtifactPriority::kInteractive;
  request.created_unix_ms = 100;
  request.due_unix_ms = 10'100;
  request.input_payload =
      Json{{"payload_version", "roomie.dam-input.v1"},
           {"owning_scene_revision", request.scene_revision},
           {"object_id", request.key.object_id},
           {"label", "blue test object"},
           {"dependency",
            Json{{"object_id", request.key.object_id},
                 {"identity_revision", request.key.identity_revision},
                 {"obb_revision", 99},
                 {"appearance_revision", request.key.appearance_revision},
                 {"semantic_revision", 4}}},
           {"snapshot_set_hash", request.key.snapshot_set_hash},
           {"snapshots",
            Json::array(
                {Json{{"image_index", 123},
                      {"source_frame_asset_id", asset_id},
                      {"evidence_hash", "evidence-17"},
                      {"bbox_xyxy", Json::array({0.0, 0.0, 3.0, 2.0})},
                      {"mask_source", "bbox_fallback"},
                      {"quality", 0.9},
                      {"image",
                       Json{{"source_path", "/tmp/untrusted-task-path.png"},
                            {"uri", "file:///tmp/untrusted-task-path.png"}}}}})}}
          .dump();
  return request;
}

TEST(PythonDamWorker, CanonicalPromptHashFencesEveryExecutionParameter) {
  const PythonDamWorkerConfig base = workerConfig();
  const std::string base_hash = canonicalDamExecutionPromptHash(base);
  ASSERT_EQ(base_hash.size(), 64U);

  std::vector<PythonDamWorkerConfig> variants(7, base);
  variants[0].query += " Include texture.";
  variants[1].conversation_mode = "v2";
  variants[2].prompt_mode = "alternate_prompt";
  variants[3].max_new_tokens += 1;
  variants[4].temperature += 0.01;
  variants[5].top_p -= 0.01;
  variants[6].bbox_pad_px += 1.0;
  DamTaskKey key;
  key.object_id = 1;
  key.identity_revision = 1;
  key.appearance_revision = 1;
  key.snapshot_set_hash = "snapshot";
  key.model_id = "model";
  key.prompt_hash = base_hash;
  key.output_schema_version = "schema";
  const std::string base_key = canonicalDamTaskKey(key);
  for (const PythonDamWorkerConfig& variant : variants) {
    const std::string variant_hash =
        canonicalDamExecutionPromptHash(variant);
    EXPECT_NE(variant_hash, base_hash);
    key.prompt_hash = variant_hash;
    EXPECT_NE(canonicalDamTaskKey(key), base_key);
  }

  TemporaryDirectory temporary;
  const auto store = makeStore(temporary.path("assets"));
  PythonDamWorkerConfig mismatched = base;
  mismatched.prompt_hash = std::string(64, '0');
  EXPECT_THROW(PythonDamWorker(std::move(mismatched), store),
               std::invalid_argument);
}

TEST(PythonDamWorker, ResolvesAndPinsImmutableAssetsAndReusesResidentModel) {
  TemporaryDirectory temporary;
  const auto store = makeStore(temporary.path("assets"));
  const AssetWriteResult asset = makeAsset(store);
  ASSERT_TRUE(asset.success) << asset.error;
  const auto starts = temporary.path("starts");
  ScopedEnvironment mode("ROOMIE_FAKE_DAM_MODE", "success");
  ScopedEnvironment expected("ROOMIE_FAKE_DAM_EXPECT_ASSET_ID", asset.asset.id);
  ScopedEnvironment start_count("ROOMIE_FAKE_DAM_START_COUNT", starts.string());

  PythonDamWorker worker(workerConfig(), store);
  const DamWorkerResponse first = worker.describe(requestFor(asset.asset.id), {});
  ASSERT_TRUE(first.success) << first.error;
  EXPECT_FALSE(first.raw_output.empty());
  const Json artifact = Json::parse(first.raw_output);
  EXPECT_EQ(artifact.at("evidence_snapshot_ids").front(), "evidence-17");
  EXPECT_EQ(store->record(asset.asset.id)->ref_count, 0U)
      << "the invocation pin must be released after the response";

  const DamWorkerResponse second = worker.describe(requestFor(asset.asset.id), {});
  ASSERT_TRUE(second.success) << second.error;
  EXPECT_EQ(lineCount(starts), 1U) << "the model process must stay resident";
}

TEST(PythonDamWorker, ChildCrashIsRetryableAndNextInvocationRestarts) {
  TemporaryDirectory temporary;
  const auto store = makeStore(temporary.path("assets"));
  const AssetWriteResult asset = makeAsset(store);
  ASSERT_TRUE(asset.success) << asset.error;
  const auto crash_marker = temporary.path("crashed-once");
  const auto starts = temporary.path("starts");
  ScopedEnvironment mode("ROOMIE_FAKE_DAM_MODE", "crash_once");
  ScopedEnvironment marker("ROOMIE_FAKE_DAM_MARKER", crash_marker.string());
  ScopedEnvironment start_count("ROOMIE_FAKE_DAM_START_COUNT", starts.string());

  PythonDamWorker worker(workerConfig(), store);
  const DamWorkerResponse crashed = worker.describe(requestFor(asset.asset.id), {});
  EXPECT_FALSE(crashed.success);
  EXPECT_TRUE(crashed.retryable);
  EXPECT_FALSE(crashed.error.empty());

  const DamWorkerResponse recovered =
      worker.describe(requestFor(asset.asset.id), {});
  EXPECT_TRUE(recovered.success) << recovered.error;
  EXPECT_EQ(lineCount(starts), 2U);
}

TEST(PythonDamWorker, RequestTimeoutKillsTermIgnoringChildWithinBound) {
  TemporaryDirectory temporary;
  const auto store = makeStore(temporary.path("assets"));
  const AssetWriteResult asset = makeAsset(store);
  ASSERT_TRUE(asset.success) << asset.error;
  const auto marker_path = temporary.path("request-seen");
  ScopedEnvironment mode("ROOMIE_FAKE_DAM_MODE", "hang_request");
  ScopedEnvironment marker("ROOMIE_FAKE_DAM_MARKER", marker_path.string());
  PythonDamWorkerConfig config = workerConfig();
  config.request_timeout = 100ms;
  config.shutdown_timeout = 100ms;

  PythonDamWorker worker(config, store);
  const auto start = std::chrono::steady_clock::now();
  const DamWorkerResponse response =
      worker.describe(requestFor(asset.asset.id), {});
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_FALSE(response.success);
  EXPECT_TRUE(response.retryable);
  EXPECT_NE(response.error.find("timed out"), std::string::npos)
      << response.error;
  EXPECT_LT(elapsed, 1s);
  EXPECT_TRUE(std::filesystem::exists(marker_path));
}

TEST(PythonDamWorker, ShutdownInterruptsHungRequestAndIsBounded) {
  TemporaryDirectory temporary;
  const auto store = makeStore(temporary.path("assets"));
  const AssetWriteResult asset = makeAsset(store);
  ASSERT_TRUE(asset.success) << asset.error;
  const auto marker_path = temporary.path("request-seen");
  ScopedEnvironment mode("ROOMIE_FAKE_DAM_MODE", "hang_request");
  ScopedEnvironment marker("ROOMIE_FAKE_DAM_MARKER", marker_path.string());
  PythonDamWorkerConfig config = workerConfig();
  config.request_timeout = 5s;
  config.shutdown_timeout = 100ms;
  PythonDamWorker worker(config, store);

  auto invocation = std::async(std::launch::async, [&]() {
    return worker.describe(requestFor(asset.asset.id), {});
  });
  ASSERT_TRUE(waitUntil(
      [&]() { return std::filesystem::exists(marker_path); }, 2s));
  const auto start = std::chrono::steady_clock::now();
  worker.shutdown();
  EXPECT_LT(std::chrono::steady_clock::now() - start, 1s);
  ASSERT_EQ(invocation.wait_for(1s), std::future_status::ready);
  const DamWorkerResponse response = invocation.get();
  EXPECT_FALSE(response.success);
  EXPECT_TRUE(response.retryable);
  EXPECT_FALSE(response.error.empty());

  const DamWorkerResponse after_stop =
      worker.describe(requestFor(asset.asset.id), {});
  EXPECT_FALSE(after_stop.success);
  EXPECT_TRUE(after_stop.retryable);
}

TEST(PythonDamWorker, MissingAssetClassificationIsExplicit) {
  TemporaryDirectory temporary;
  const auto store = makeStore(temporary.path("assets"));
  const AssetWriteResult asset = makeAsset(store);
  ASSERT_TRUE(asset.success) << asset.error;
  ScopedEnvironment mode("ROOMIE_FAKE_DAM_MODE", "success");
  PythonDamWorker worker(workerConfig(), store);

  DamTaskRequest malformed = requestFor(asset.asset.id);
  Json malformed_payload = Json::parse(malformed.input_payload);
  malformed_payload["snapshots"][0].erase("source_frame_asset_id");
  malformed.input_payload = malformed_payload.dump();
  const DamWorkerResponse malformed_response = worker.describe(malformed, {});
  EXPECT_FALSE(malformed_response.success);
  EXPECT_FALSE(malformed_response.retryable);

  const DamWorkerResponse unknown =
      worker.describe(requestFor(std::string(64, 'f')), {});
  EXPECT_FALSE(unknown.success);
  EXPECT_FALSE(unknown.retryable);
  EXPECT_NE(unknown.error.find("unknown asset"), std::string::npos);

  ASSERT_TRUE(std::filesystem::remove(asset.asset.path));
  const DamWorkerResponse disappeared =
      worker.describe(requestFor(asset.asset.id), {});
  EXPECT_FALSE(disappeared.success);
  EXPECT_TRUE(disappeared.retryable);
  EXPECT_NE(disappeared.error.find("not readable"), std::string::npos);
}

TEST(PythonDamWorker, StartupDependencyFailureIsPermanentAndExplicit) {
  TemporaryDirectory temporary;
  const auto store = makeStore(temporary.path("assets"));
  const AssetWriteResult asset = makeAsset(store);
  ASSERT_TRUE(asset.success) << asset.error;
  ScopedEnvironment mode("ROOMIE_FAKE_DAM_MODE", "startup_error");
  PythonDamWorker worker(workerConfig(), store);

  const DamWorkerResponse response =
      worker.describe(requestFor(asset.asset.id), {});
  EXPECT_FALSE(response.success);
  EXPECT_FALSE(response.retryable);
  EXPECT_NE(response.error.find("dependency is unavailable"), std::string::npos)
      << response.error;
}

TEST(PythonDamWorker, ClosedRequestPipeReportsEpipeWithoutKillingCaller) {
  TemporaryDirectory temporary;
  const auto store = makeStore(temporary.path("assets"));
  const AssetWriteResult asset = makeAsset(store);
  ASSERT_TRUE(asset.success) << asset.error;
  ScopedEnvironment mode("ROOMIE_FAKE_DAM_MODE", "close_stdin");
  ScopedSignalDisposition default_sigpipe(SIGPIPE, SIG_DFL);
  PythonDamWorker worker(workerConfig(), store);

  const DamWorkerResponse response =
      worker.describe(requestFor(asset.asset.id), {});
  EXPECT_FALSE(response.success);
  EXPECT_TRUE(response.retryable);
  EXPECT_NE(response.error.find("EPIPE"), std::string::npos) << response.error;
}

}  // namespace
}  // namespace roomie
