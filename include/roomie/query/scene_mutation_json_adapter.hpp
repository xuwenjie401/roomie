#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "roomie/scene/scene_command.hpp"

namespace roomie {

enum class SceneMutationStatus {
  kCommitted,
  kNoOp,
  kStale,
  kRejected,
  kTimeout,
  // The reducer commit is already authoritative in memory, but its durable
  // sidecar did not reach that revision within the request deadline.
  kCommittedNotDurable,
  kUnavailable,
};

struct SceneMutationSubmitResult {
  SceneMutationStatus status = SceneMutationStatus::kUnavailable;
  std::string message;
  SceneRevision latest_scene_revision = 0;
  SceneRevision durable_scene_revision = 0;
  std::optional<SceneRevision> committed_scene_revision;
  std::optional<std::uint64_t> component_revision;

  bool accepted() const {
    return status == SceneMutationStatus::kCommitted ||
           status == SceneMutationStatus::kNoOp;
  }
  bool durable() const {
    return latest_scene_revision == 0 ||
           durable_scene_revision >= latest_scene_revision;
  }
};

using SceneMutationSubmitter = std::function<SceneMutationSubmitResult(
    ApplyHumanAnnotationCommand, std::chrono::milliseconds)>;

// Transport-neutral result used by the ROS service. A valid request always
// receives response_json, including stale/rejected outcomes. error is reserved
// for a malformed envelope or a failed mutation and mirrors its explicit JSON
// status for simple ROS clients.
struct SceneMutationJsonResponse {
  bool success = false;
  std::string response_json;
  std::string error;
  SceneRevision latest_scene_revision = 0;
  SceneRevision durable_scene_revision = 0;
};

// Strict JSON-to-command boundary for live human annotations. It never edits
// a snapshot itself: every parsed request is handed to the supplied reducer
// submission function. room_memberships is an optimistic assertion over
// reducer-derived containment, not a relation mutation.
class SceneMutationJsonAdapter {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout{5000};
  static constexpr std::chrono::milliseconds kMaximumTimeout{30000};

  explicit SceneMutationJsonAdapter(SceneMutationSubmitter submitter)
      : submitter_(std::move(submitter)) {}

  SceneMutationJsonResponse dispatch(std::string_view request_json) const;

 private:
  SceneMutationSubmitter submitter_;
};

}  // namespace roomie
