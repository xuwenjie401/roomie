#pragma once

#include <cstdint>
#include <string>

#include "roomie/pipeline/types.hpp"

namespace roomie {

// Durable identity for one immutable map file and the exact scene prefix that
// was known durable before the file was published. The checkpoint file lives
// outside SQLite; SceneStore persists only this manifest.
struct MapCheckpointManifest {
  std::string backend;
  std::string checkpoint_path;
  // A checkpoint may only be restored into the same coordinate frame and a
  // map backend with the same persistence-relevant configuration.  The
  // fingerprint is backend-defined and versioned.
  std::string world_frame;
  std::string config_fingerprint;
  RunId map_epoch;
  std::uint64_t map_revision = 0;
  TimeNanoseconds integrated_through_ns = 0;
  SceneRevision aligned_scene_revision = 0;
  std::uint64_t file_size_bytes = 0;
  // Includes the algorithm prefix, for example "fnv1a64:...".
  std::string content_hash;
  std::int64_t created_at_unix_ms = 0;
};

enum class MapCheckpointDisposition {
  kNotRequested,
  kUnsupported,
  kSucceeded,
  kSkipped,
  kFailed,
};

struct MapCheckpointOperationResult {
  MapCheckpointDisposition disposition =
      MapCheckpointDisposition::kNotRequested;
  MapCheckpointManifest manifest;
  std::string error;

  bool succeeded() const {
    return disposition == MapCheckpointDisposition::kSucceeded;
  }
};

}  // namespace roomie
