#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "roomie/dsg/object_graph.hpp"
#include "roomie/pipeline/types.hpp"

namespace roomie {

// Object ids are process-independent scene identities. An id that appears in
// the alias or tombstone table is retired and must never be allocated again.
using SceneObjectId = int;

// Service-level context for one appearance-derived artifact chain. Unix time
// is the durable/cross-process scheduling representation. The optional steady
// clock tuple is valid only while steady_clock_epoch matches this process and
// is used for acceptance latency so wall-clock adjustments cannot make the
// SLO pass or fail spuriously. It deliberately lives beside reducer state
// rather than in SemanticDocument: scheduling/accounting metadata must never
// change semantic document content or its content-addressed hash. A zero due
// time is the backward-compatible representation for legacy/untracked
// artifacts.
enum class ArtifactPriority : std::uint8_t {
  kInteractive = 0,
  kBulk = 1,
};

struct ArtifactSloContext {
  std::int64_t origin_created_unix_ms = 0;
  std::int64_t due_unix_ms = 0;
  ArtifactPriority priority = ArtifactPriority::kBulk;
  RunId steady_clock_epoch;
  std::int64_t origin_steady_ns = 0;
  std::int64_t due_steady_ns = 0;

  bool tracked() const { return due_unix_ms > 0; }
  bool monotonicTracked() const {
    return steady_clock_epoch.valid() && origin_steady_ns > 0 &&
           due_steady_ns >= origin_steady_ns;
  }
  bool valid() const {
    const bool valid_priority = priority == ArtifactPriority::kInteractive ||
                                priority == ArtifactPriority::kBulk;
    // The all-zero tuple is the sole representation of an untracked legacy
    // artifact.  In particular a negative due value must not accidentally be
    // accepted merely because tracked() is defined in terms of due > 0.
    const bool unix_valid =
        (origin_created_unix_ms == 0 && due_unix_ms == 0) ||
        (origin_created_unix_ms >= 0 && due_unix_ms > 0 &&
         due_unix_ms >= origin_created_unix_ms);
    const bool no_monotonic = !steady_clock_epoch.valid() &&
                              origin_steady_ns == 0 && due_steady_ns == 0;
    return valid_priority && unix_valid &&
           (no_monotonic || monotonicTracked());
  }
};

inline bool operator==(const ArtifactSloContext& lhs,
                       const ArtifactSloContext& rhs) {
  return lhs.origin_created_unix_ms == rhs.origin_created_unix_ms &&
         lhs.due_unix_ms == rhs.due_unix_ms &&
         lhs.priority == rhs.priority &&
         lhs.steady_clock_epoch == rhs.steady_clock_epoch &&
         lhs.origin_steady_ns == rhs.origin_steady_ns &&
         lhs.due_steady_ns == rhs.due_steady_ns;
}

inline bool operator!=(const ArtifactSloContext& lhs,
                       const ArtifactSloContext& rhs) {
  return !(lhs == rhs);
}

struct ComponentRevisions {
  std::uint64_t identity_revision = 0;
  std::uint64_t lifecycle_revision = 0;
  std::uint64_t geometry_revision = 0;
  std::uint64_t semantic_revision = 0;
  std::uint64_t annotation_revision = 0;
  std::uint64_t artifact_revision = 0;
};

struct IdentityComponent {
  std::uint64_t revision = 0;
  SceneObjectId object_id = -1;
  std::vector<int> source_track_ids;
};

struct LifecycleComponent {
  std::uint64_t revision = 0;
  InstanceTrackState track_state = InstanceTrackState::kTentative;
  bool active = true;
  bool publishable = true;
  float existence_log_odds = 0.0f;
  TimeNanoseconds last_presence_evidence_ns = 0;
  float last_presence_evidence_reliability = 0.0f;
  std::string last_presence_evidence_reason;
  std::vector<TimeNanoseconds> positive_evidence_timestamps_ns;
  PositivePresenceEvidenceHistory positive_presence_evidence_history;
  std::vector<TimeNanoseconds> negative_evidence_timestamps_ns;
  int positive_window_interruptions = 0;
  TimeNanoseconds first_seen_ns = 0;
  TimeNanoseconds last_seen_ns = 0;
  std::uint64_t first_seen_frame_index = 0;
  std::uint64_t last_seen_frame_index = 0;
};

struct GeometryComponent {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::uint64_t revision = 0;
  std::uint64_t obb_revision = 0;
  std::uint64_t evaluated_obb_revision = 0;
  Eigen::Vector3f center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f size_m = Eigen::Vector3f::Zero();
  float yaw_rad = 0.0f;
  InstanceGeometryStatus status = InstanceGeometryStatus::kUnchecked;
  float score = 0.0f;
  float shell_ratio = 0.0f;
  float extent_score = 0.0f;
  float leak_ratio = 1.0f;
  float cavity_ratio = 0.0f;
  int in_box_points = 0;
  int shell_points = 0;
  int unique_voxels = 0;
  int expanded_points = 0;
  int bad_count = 0;
  TimeNanoseconds last_check_ns = 0;
  Eigen::Vector3f evaluated_center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f evaluated_size_m = Eigen::Vector3f::Zero();
  float evaluated_yaw_rad = 0.0f;
  std::string evaluation_reason;
  SurfaceStamp evaluated_surface;
  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>
      evaluated_blocks;
};

struct SemanticComponent {
  std::uint64_t revision = 0;
  int semantic_id = -1;
  std::string label;
  float confidence = 0.0f;
  float confidence_mass = 0.0f;
  float object_quality_score = 0.0f;
  int support_count = 0;
  int high_quality_observation_count = 0;
  float high_quality_observation_mass = 0.0f;
  std::vector<std::string> source_cameras;
  std::vector<TimeNanoseconds> observation_timestamps_ns;
  std::map<std::string, float> label_weights;
  std::map<int, float> semantic_weights;
};

struct AnnotationComponent {
  std::uint64_t revision = 0;
  std::optional<int> semantic_id_override;
  std::optional<std::string> label_override;
  std::optional<std::string> description_override;
  std::map<std::string, std::string> attributes;
};

struct ArtifactComponent {
  std::uint64_t revision = 0;
  std::uint64_t appearance_revision = 0;
  std::vector<ObjectSnapshotRef> snapshots;
  std::string snapshot_set_hash;
  std::string description;
  std::string description_input_hash;
  std::string description_model_id;
  std::string description_schema_version;
  std::string description_raw_text;
  std::string description_normalized_json;
  std::string description_durable_envelope_json;
  std::string description_parse_path;
  ArtifactSloContext description_slo;
  bool description_stale = false;

  // A DAM result first enters the owning content revision as pending. The
  // durable outbox for its embedding is committed in the same transaction.
  // PersistedThroughCommand atomically promotes it to the current-ready
  // fields above; until then the previous description remains a stale
  // fallback.
  SceneRevision pending_description_scene_revision = 0;
  std::string pending_description;
  std::string pending_description_input_hash;
  std::string pending_description_model_id;
  std::string pending_description_schema_version;
  std::string pending_description_raw_text;
  std::string pending_description_normalized_json;
  std::string pending_description_durable_envelope_json;
  std::string pending_description_parse_path;
  ArtifactSloContext pending_description_slo;
};

struct SceneObject {
  std::shared_ptr<const IdentityComponent> identity;
  std::shared_ptr<const LifecycleComponent> lifecycle;
  std::shared_ptr<const GeometryComponent> geometry;
  std::shared_ptr<const SemanticComponent> semantic;
  std::shared_ptr<const AnnotationComponent> annotation;
  std::shared_ptr<const ArtifactComponent> artifact;

  ComponentRevisions revisions() const {
    ComponentRevisions result;
    result.identity_revision = identity ? identity->revision : 0;
    result.lifecycle_revision = lifecycle ? lifecycle->revision : 0;
    result.geometry_revision = geometry ? geometry->revision : 0;
    result.semantic_revision = semantic ? semantic->revision : 0;
    result.annotation_revision = annotation ? annotation->revision : 0;
    result.artifact_revision = artifact ? artifact->revision : 0;
    return result;
  }
};

struct ObjectDependency {
  SceneObjectId object_id = -1;
  std::uint64_t identity_revision = 0;
  std::uint64_t obb_revision = 0;
  std::uint64_t appearance_revision = 0;
  std::uint64_t semantic_revision = 0;
};

struct GeometryDependency {
  ObjectDependency object;
  SurfaceStamp surface;
  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>
      evaluated_blocks;
};

enum class DeltaOverlapVerdict {
  // No journal lookup was needed because the exact surface stamp is current.
  kExactSurface,
  // The surface advanced and the complete journal proves no evaluated block changed.
  kNoOverlap,
  // At least one evaluated block changed.
  kOverlap,
  // The journal cannot cover the full source-to-current interval.
  kJournalGap,
  // No authoritative verdict was supplied.
  kUnknown,
};

struct ObjectAlias {
  SceneObjectId retired_object_id = -1;
  SceneObjectId canonical_object_id = -1;
  SceneRevision created_revision = 0;
};

struct ObjectTombstone {
  SceneObjectId object_id = -1;
  SceneRevision deleted_revision = 0;
  std::string reason;
};

struct SceneGraphMetadata {
  // Rooms and relations are canonical, typed scene state. The legacy
  // ObjectGraphSnapshot is materialized from this component; the preserved
  // envelope is metadata only and is never consulted for current facts.
  std::vector<RoomNode, Eigen::aligned_allocator<RoomNode>> rooms;
  std::vector<FurnitureRole> furniture;
  std::vector<SceneRelation> relations;
  std::vector<ObjectSnapshotImage> snapshot_images;
  std::vector<std::string> import_warnings;
  bool has_scene_graph_envelope = false;
  std::string scene_graph_json;
};

struct ObservationRunWatermark {
  RunId run_id;
  FrameId highest_frame_id = 0;
};

using SceneObjectPtr = std::shared_ptr<const SceneObject>;
using SceneObjectTable = std::map<SceneObjectId, SceneObjectPtr>;
using SceneAliasTable = std::map<SceneObjectId, ObjectAlias>;
using SceneTombstoneTable = std::map<SceneObjectId, ObjectTombstone>;
using SceneTrackPtr = std::shared_ptr<const InstanceTrack>;
using SceneTrackTable = std::map<int, SceneTrackPtr>;

// SceneState is copied only shallowly. Tables are replaced copy-on-write and
// SceneObject components are immutable shared_ptrs, so old snapshots retain a
// coherent revision without deep-copying observation or artifact payloads.
struct SceneState {
  SceneRevision latest_scene_revision = 0;
  SceneRevision durable_scene_revision = 0;
  SceneObjectId next_object_id = 0;
  SurfaceStamp latest_surface;
  bool shutdown_requested = false;
  // Bounded replay ledger for inference retries. It is authoritative only for
  // valid run ids; bootstrap/tests without provenance retain legacy behavior.
  std::vector<FrameKey> recent_observation_frames;
  std::vector<ObservationRunWatermark> observation_watermarks;
  std::shared_ptr<const SceneObjectTable> objects =
      std::make_shared<const SceneObjectTable>();
  std::shared_ptr<const SceneAliasTable> aliases =
      std::make_shared<const SceneAliasTable>();
  std::shared_ptr<const SceneTombstoneTable> tombstones =
      std::make_shared<const SceneTombstoneTable>();
  std::shared_ptr<const SceneTrackTable> tracks =
      std::make_shared<const SceneTrackTable>();
  std::shared_ptr<const SceneGraphMetadata> graph =
      std::make_shared<const SceneGraphMetadata>();
};

inline ObjectDependency dependencyFor(const SceneObject& object) {
  ObjectDependency dependency;
  if (object.identity) {
    dependency.object_id = object.identity->object_id;
    dependency.identity_revision = object.identity->revision;
  }
  if (object.geometry) {
    dependency.obb_revision = object.geometry->obb_revision;
  }
  if (object.artifact) {
    dependency.appearance_revision = object.artifact->appearance_revision;
  }
  if (object.semantic) {
    dependency.semantic_revision = object.semantic->revision;
  }
  return dependency;
}

}  // namespace roomie
