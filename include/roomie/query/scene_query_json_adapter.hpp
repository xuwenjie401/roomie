#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "roomie/query/scene_query_gateway.hpp"

namespace roomie {

// Transport-neutral result used by the ROS service and by other JSON-RPC
// frontends. success only describes whether the request envelope was parsed
// and dispatched; individual calls carry their own QueryStatus.
struct SceneQueryJsonResponse {
  bool success = false;
  std::string response_json;
  std::string error;
  // Machine-readable envelope/session failure. Per-call failures remain in
  // response_json.calls[].status.
  std::string error_code;
  SceneRevision scene_revision = 0;
  SceneRevision durable_scene_revision = 0;
  std::uint64_t index_generation = 0;
  std::size_t call_count = 0;
};

// Legacy batches without session_id pin once per request. begin_session
// creates a bounded server-side TTL lease so calls arriving in later requests
// can reuse the same revision for one complete model answer.
class SceneQueryJsonAdapter {
 public:
  explicit SceneQueryJsonAdapter(const SceneQueryGateway& gateway)
      : gateway_(&gateway) {}

  SceneQueryJsonResponse dispatch(std::string_view request_json) const;

 private:
  const SceneQueryGateway* gateway_ = nullptr;
};

}  // namespace roomie
