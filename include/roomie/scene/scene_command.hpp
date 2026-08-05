#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <Eigen/StdVector>

#include "roomie/scene/scene_state.hpp"

namespace roomie {

struct LoadSceneCommand {
  // SceneStore restore supplies the exact immutable component graph here so
  // bootstrap does not collapse Top-K artifacts/revisions through the legacy
  // ObjectGraph compatibility view.
  std::shared_ptr<const SceneState> restored_state;
  ObjectGraphSnapshot graph;
  std::vector<InstanceTrack, Eigen::aligned_allocator<InstanceTrack>> tracks;
  std::vector<ObjectAlias> aliases;
  std::vector<ObjectTombstone> tombstones;
  SceneRevision restored_revision = 0;
  SceneRevision durable_revision = 0;
  SurfaceStamp latest_surface;
  std::vector<FrameKey> recent_observation_frames;
  std::vector<ObservationRunWatermark> observation_watermarks;
};

// Association remains outside ReducerCore in this skeleton. An associator can
// translate raw observations into these deterministic, typed mutations before
// the reducer commits them.
struct UpsertTrackMutation {
  InstanceTrack track;
  // A negative value requests a never-before-used id from the reducer.
  SceneObjectId object_id = -1;
};

struct UpsertTentativeTrackMutation {
  InstanceTrack track;
};

struct RemoveTrackMutation {
  int track_id = -1;
};

struct MergeObjectsMutation {
  SceneObjectId retired_object_id = -1;
  SceneObjectId canonical_object_id = -1;
  // When present, this is the durable Top-K union selected from the pinned
  // pre-merge scene. Carrying it in the same reducer command closes the crash
  // window between retiring the source object and merging SnapshotBank.
  std::optional<std::vector<ObjectSnapshotRef>> merged_snapshots;
  std::string merged_snapshot_set_hash;
};

struct TombstoneObjectMutation {
  SceneObjectId object_id = -1;
  std::string reason;
};

using ObservationMutation =
    std::variant<UpsertTrackMutation, UpsertTentativeTrackMutation,
                 RemoveTrackMutation, MergeObjectsMutation,
                 TombstoneObjectMutation>;

struct ApplyObservationBatchCommand {
  FrameProvenance provenance;
  std::vector<InstanceObservation,
              Eigen::aligned_allocator<InstanceObservation>> observations;
  std::vector<ObservationMutation> associated_mutations;
  std::string association_source;
};

// Serializes the authoritative map surface watermark through the same actor
// as geometry results. This closes the check/enqueue race without turning map
// state into a second mutable scene writer.
struct AdvanceSurfaceCommand {
  SurfaceStamp surface;
};

struct GeometryEvaluationResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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
  TimeNanoseconds checked_at_ns = 0;
  Eigen::Vector3f evaluated_center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f evaluated_size_m = Eigen::Vector3f::Zero();
  float evaluated_yaw_rad = 0.0f;
  std::string reason;
};

struct ApplyGeometryResultCommand {
  GeometryDependency dependency;
  // The scheduler supplies the surface that is current at commit time.
  SurfaceStamp current_surface;
  DeltaOverlapVerdict delta_overlap = DeltaOverlapVerdict::kUnknown;
  GeometryEvaluationResult result;
};

struct ApplySnapshotSetCommand {
  ObjectDependency dependency;
  std::vector<ObjectSnapshotRef> snapshots;
  std::string snapshot_set_hash;
};

struct ApplyDescriptionArtifactCommand {
  ObjectDependency dependency;
  std::string description;
  std::string input_hash;
  std::string model_id;
  std::string schema_version;
  std::string raw_text;
  std::string normalized_json;
  std::string durable_envelope_json;
  std::string parse_path;
  ArtifactSloContext artifact_slo;
};

struct HumanAnnotationPatch {
  std::optional<int> semantic_id;
  std::optional<std::string> label;
  std::optional<std::string> description;
  std::map<std::string, std::string> attributes;

  bool empty() const {
    return !semantic_id && !label && !description && attributes.empty();
  }
};

struct RoomAnnotationPatch {
  std::optional<std::string> label;
  std::optional<std::string> color;
  std::optional<Eigen::Vector3f> center_world;
  std::optional<Eigen::Vector3f> size_m;
  std::optional<std::array<float, 2>> min_xy;
  std::optional<std::array<float, 2>> max_xy;
  std::optional<float> height_m;
  std::map<std::string, std::string> attributes;
  bool remove = false;

  bool empty() const {
    return !label && !color && !center_world && !size_m && !min_xy &&
           !max_xy && !height_m && attributes.empty() && !remove;
  }
};

struct ApplyHumanAnnotationCommand {
  // Present for external mutation gateways that need an exact scene-level
  // compare-and-swap. Keeping this optional preserves the component-level CAS
  // used by internal artifact producers while making a client supplied zero a
  // real revision rather than a wildcard.
  std::optional<SceneRevision> expected_scene_revision;
  // target is authoritative for schema-v3 callers. object_id remains a
  // compatibility shorthand for existing object annotation producers.
  SceneEntityRef target;
  SceneObjectId object_id = -1;
  // Zero is an explicit wildcard for interactive edits. Non-zero values make
  // the command idempotent/stale-safe across alias and concurrent updates.
  std::uint64_t expected_identity_revision = 0;
  std::uint64_t expected_annotation_revision = 0;
  std::uint64_t expected_room_revision = 0;
  // This is an assertion over reducer-derived room containment, never a
  // relation write. For an object target the ids are containing rooms; for a
  // room target they are contained objects. The reducer rejects a mismatch as
  // stale after resolving aliases/recomputing the proposed room geometry.
  std::optional<std::vector<int>> expected_room_memberships;
  HumanAnnotationPatch patch;
  std::optional<RoomAnnotationPatch> room_patch;
};

struct PersistedThroughCommand {
  SceneRevision revision = 0;
};

struct ShutdownCommand {
  std::string reason;
};

using SceneCommand =
    std::variant<LoadSceneCommand, ApplyObservationBatchCommand,
                 AdvanceSurfaceCommand,
                 ApplyGeometryResultCommand, ApplySnapshotSetCommand,
                 ApplyDescriptionArtifactCommand, ApplyHumanAnnotationCommand,
                 PersistedThroughCommand, ShutdownCommand>;

}  // namespace roomie
