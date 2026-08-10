#pragma once

#include <cstddef>
#include <string>
#include <variant>

#include "roomie/scene/scene_state.hpp"

namespace roomie {

struct SceneLoaded {
  SceneRevision revision = 0;
  std::size_t object_count = 0;
};

struct SceneRevisionCommitted {
  SceneRevision previous_revision = 0;
  SceneRevision revision = 0;
};

struct ObservationBatchCommitted {
  SceneRevision revision = 0;
  FrameProvenance provenance;
  std::size_t observation_count = 0;
  std::size_t mutation_count = 0;
};

struct SurfaceAdvanced {
  SceneRevision latest_scene_revision = 0;
  SurfaceStamp surface;
};

struct ObjectCreated {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
};

struct ObjectUpdated {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
  ComponentRevisions components;
};

struct ObjectMerged {
  SceneRevision revision = 0;
  SceneObjectId retired_object_id = -1;
  SceneObjectId canonical_object_id = -1;
};

struct ObjectTombstoned {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
  std::string reason;
};

struct ObbChanged {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
  std::uint64_t obb_revision = 0;
};

struct GeometryInvalidated {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
  std::uint64_t obb_revision = 0;
};

struct GeometryCommitted {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
  std::uint64_t geometry_revision = 0;
  SurfaceStamp surface;
};

struct AppearanceInvalidated {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
  std::uint64_t appearance_revision = 0;
};

struct SnapshotSetCommitted {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
  std::uint64_t appearance_revision = 0;
  std::string snapshot_set_hash;
};

struct DescriptionInvalidated {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
  std::uint64_t appearance_revision = 0;
};

struct DescriptionCommitted {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
  std::uint64_t artifact_revision = 0;
  std::string input_hash;
};

struct HumanAnnotationCommitted {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
  std::uint64_t annotation_revision = 0;
  SceneEntityRef target;
  // True only when the edit changes fields embedded by
  // makeSemanticDocumentForObject(). Metadata-only annotation edits do not
  // need a new content-addressed embedding task.
  bool semantic_document_changed = false;
};

struct RoomUpdated {
  SceneRevision revision = 0;
  int room_id = -1;
  std::uint64_t room_revision = 0;
  bool created = false;
};

struct RoomRemoved {
  SceneRevision revision = 0;
  int room_id = -1;
};

struct RelationInvalidated {
  SceneRevision revision = 0;
  SceneObjectId object_id = -1;
  SceneEntityRef target;
};

struct RelationCommitted {
  SceneRevision revision = 0;
  ObjectRelation relation;
};

struct FurnitureGraphRebuilt {
  SceneRevision revision = 0;
  std::size_t furniture_count = 0;
  std::size_t in_relation_count = 0;
  std::size_t on_relation_count = 0;
  std::size_t room_relation_count = 0;
};

struct DurabilityWatermarkAdvanced {
  SceneRevision latest_scene_revision = 0;
  SceneRevision durable_scene_revision = 0;
};

struct ShutdownAccepted {
  SceneRevision latest_scene_revision = 0;
  std::string reason;
};

using SceneEvent =
    std::variant<SceneLoaded, SceneRevisionCommitted,
                 ObservationBatchCommitted, SurfaceAdvanced,
                 ObjectCreated, ObjectUpdated,
                 ObjectMerged, ObjectTombstoned, ObbChanged,
                 GeometryInvalidated, GeometryCommitted,
                 AppearanceInvalidated, SnapshotSetCommitted,
                 DescriptionInvalidated, DescriptionCommitted,
                 HumanAnnotationCommitted, RoomUpdated, RoomRemoved,
                 RelationInvalidated,
                 RelationCommitted, FurnitureGraphRebuilt,
                 DurabilityWatermarkAdvanced,
                 ShutdownAccepted>;

}  // namespace roomie
