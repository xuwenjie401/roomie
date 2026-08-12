#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "roomie/scene/scene_snapshot.hpp"

namespace roomie {

enum class QueryStatus {
  kOk,
  kExpiredToken,
  kForeignToken,
  kInvalidArgument,
  kNotFound,
  kIndexUnavailable,
};

enum class Freshness {
  kCurrent,
  kStale,
  kPending,
  kMissing,
};

struct QueryMetadata {
  SceneRevision scene_revision = 0;
  SceneRevision durable_scene_revision = 0;
  std::uint64_t index_generation = 0;
  TimeNanoseconds as_of_ns = 0;
  TimeNanoseconds index_as_of_ns = 0;
  Freshness geometry_freshness = Freshness::kMissing;
  Freshness semantic_freshness = Freshness::kMissing;
  Freshness description_freshness = Freshness::kMissing;
  bool artifact_pending = false;
  bool index_available = false;
  bool lexical_delta_used = false;
  std::vector<SceneObjectId> pending_object_ids;
};

template <typename T>
struct QueryResult {
  QueryStatus status = QueryStatus::kOk;
  std::string message;
  QueryMetadata metadata;
  T value{};

  bool ok() const { return status == QueryStatus::kOk; }
};

struct IndexedSearchHit {
  SceneObjectId object_id = -1;
  float score = 0.0f;
  std::string document_hash;
};

// A generation is immutable after it is returned by SearchProvider. Keeping
// the shared_ptr alive is the generation lease used by SceneReadToken.
class PinnedSearchIndex {
 public:
  virtual ~PinnedSearchIndex() = default;

  virtual std::uint64_t generation() const = 0;
  virtual TimeNanoseconds asOfNanoseconds() const { return 0; }
  virtual std::optional<std::string> documentHash(
      SceneObjectId object_id) const = 0;
  virtual std::vector<IndexedSearchHit> search(
      std::string_view query, std::size_t limit) const = 0;
};

class SearchProvider {
 public:
  virtual ~SearchProvider() = default;
  virtual std::shared_ptr<const PinnedSearchIndex> pinCurrent() const = 0;
};

// Physical metadata for one content-addressed full-frame asset. The stable
// source_frame_asset_id remains authoritative; uri/source_path are resolver
// outputs and never become a second scene-state fact source.
struct ResolvedSnapshotAsset {
  std::string source_frame_asset_id;
  std::string uri;
  std::string source_path;
  int width = 0;
  int height = 0;
  int channels = 0;
  // Physical file encoding followed by the decoded/source pixel encoding.
  std::string encoding;
  std::string source_encoding;
  std::uint64_t encoded_bytes = 0;

  bool valid() const {
    return !source_frame_asset_id.empty() &&
           (!uri.empty() || !source_path.empty());
  }
};

// A resolver is immutable for its lifetime. Providers must also keep every
// returned physical asset readable while this lease is alive; this lets a
// SceneReadToken pin scene, index, and snapshot assets as one coherent view.
class PinnedSnapshotAssetResolver {
 public:
  virtual ~PinnedSnapshotAssetResolver() = default;
  virtual std::optional<ResolvedSnapshotAsset> resolve(
      std::string_view source_frame_asset_id) const = 0;
};

class SnapshotAssetProvider {
 public:
  virtual ~SnapshotAssetProvider() = default;
  virtual std::shared_ptr<const PinnedSnapshotAssetResolver> pin(
      const std::vector<std::string>& source_frame_asset_ids) const = 0;
};

struct ObjectFreshness {
  Freshness geometry = Freshness::kMissing;
  Freshness semantic = Freshness::kMissing;
  Freshness description = Freshness::kMissing;
  std::vector<std::string> pending_artifacts;

  bool artifactPending() const { return !pending_artifacts.empty(); }
};

struct QueryObjectView {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SceneObjectId requested_object_id = -1;
  SceneObjectId object_id = -1;
  bool resolved_alias = false;
  int semantic_id = -1;
  std::string name;
  std::string label;
  // Canonical description is empty when neither a human annotation nor a
  // description artifact exists. display_description is a query/UI fallback.
  std::optional<std::string> canonical_description;
  std::string display_description;
  bool description_is_label_fallback = false;
  bool active = false;
  bool publishable = false;
  float existence_log_odds = 0.0f;
  float existence_probability = 0.5f;
  std::string presence_state = "tentative";
  TimeNanoseconds last_presence_evidence_ns = 0;
  std::string last_presence_evidence_reason;
  Eigen::Vector3f center_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f size_m = Eigen::Vector3f::Zero();
  float yaw_rad = 0.0f;
  ComponentRevisions revisions;
  std::string semantic_document_hash;
  std::optional<std::string> indexed_document_hash;
  std::string snapshot_set_hash;
  // Present when this canonical object also has a furniture role. The role
  // projects the same object id; it is not a copied furniture object.
  std::optional<FurnitureRole> furniture_role;
  ObjectFreshness freshness;
};

using QueryObjectList =
    std::vector<QueryObjectView, Eigen::aligned_allocator<QueryObjectView>>;

struct QueryFurnitureView {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FurnitureRole role;
  QueryObjectView object;
};

using QueryFurnitureList =
    std::vector<QueryFurnitureView,
                Eigen::aligned_allocator<QueryFurnitureView>>;

struct ObjectsNearRequest {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3f center_world = Eigen::Vector3f::Zero();
  float radius_m = 1.0f;
  std::size_t limit = 20;
  bool include_inactive = false;
  bool include_unpublishable = false;
};

enum class RelationDirection {
  kEither,
  kOutgoing,
  kIncoming,
};

struct RelationRequest {
  std::optional<SceneObjectId> object_id;
  std::optional<std::string> relation_type;
  RelationDirection direction = RelationDirection::kEither;
};

struct CanonicalRelation {
  SceneEntityRef source;
  SceneEntityRef target;
  // Compatibility object-only endpoints. Room endpoints remain -1.
  SceneObjectId source_object_id = -1;
  SceneObjectId target_object_id = -1;
  std::string relation_type;
  float confidence = 0.0f;
  std::string description;
  std::uint64_t revision = 0;
  bool derived = false;
  bool source_resolved_alias = false;
  bool target_resolved_alias = false;
};

struct CanonicalRoom {
  std::string room_id;
  std::string name;
  std::optional<SceneObjectId> room_object_id;
  std::map<std::string, std::string> attributes;
  std::vector<SceneObjectId> object_ids;
  std::vector<SceneObjectId> furniture_ids;
};

struct SnapshotAssetInspection {
  ObjectSnapshotRef reference;
  // Populated for online content-addressed snapshots. `asset` below remains
  // the compatibility projection shared with legacy snapshot_images users.
  std::optional<ResolvedSnapshotAsset> physical_asset;
  std::optional<ObjectSnapshotImage> asset;
  bool available = false;
};

struct SnapshotInspection {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  QueryObjectView object;
  std::vector<SnapshotAssetInspection> snapshots;
  std::string snapshot_set_hash;
};

enum class SearchMatchSource {
  kVector,
  kPinnedLexicalDelta,
};

struct SearchRequest {
  std::string query;
  std::size_t limit = 10;
  bool include_inactive = false;
  bool include_unpublishable = false;
  std::optional<std::string> room_id;
};

struct SearchObjectMatch {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  QueryObjectView object;
  float score = 0.0f;
  SearchMatchSource source = SearchMatchSource::kVector;
  std::string vector_document_hash;
};

using SearchObjectMatches =
    std::vector<SearchObjectMatch,
                Eigen::aligned_allocator<SearchObjectMatch>>;

class SceneReadToken {
 public:
  using Clock = std::chrono::steady_clock;

  SceneReadToken() = default;

  bool valid() const { return gateway_id_ != 0 && snapshot_.statePtr(); }
  SceneRevision sceneRevision() const { return snapshot_.revision(); }
  SceneRevision durableSceneRevision() const {
    return snapshot_.durableRevision();
  }
  std::uint64_t indexGeneration() const { return index_generation_; }
  TimeNanoseconds asOfNanoseconds() const { return as_of_ns_; }
  Clock::time_point expiresAt() const { return expires_at_; }

 private:
  friend class SceneQueryGateway;

  std::uint64_t gateway_id_ = 0;
  SceneSnapshot snapshot_;
  std::shared_ptr<const PinnedSearchIndex> index_;
  std::shared_ptr<const PinnedSnapshotAssetResolver> snapshot_assets_;
  std::uint64_t index_generation_ = 0;
  TimeNanoseconds index_as_of_ns_ = 0;
  TimeNanoseconds as_of_ns_ = 0;
  Clock::time_point expires_at_ = Clock::time_point::min();
};

enum class SceneReadSessionStatus {
  kOk,
  kInvalidArgument,
  kNotFound,
  kExpired,
  kRevisionMismatch,
  kCapacityExceeded,
};

// Server-side lease for transports whose tool calls arrive as separate
// requests. The opaque id never encodes a revision; the pinned token is held
// only by the gateway and is released on expiry.
struct SceneReadSessionResult {
  SceneReadSessionStatus status = SceneReadSessionStatus::kOk;
  std::string message;
  std::string session_id;
  SceneReadToken token;

  bool ok() const { return status == SceneReadSessionStatus::kOk; }
};

class SceneQueryGateway {
 public:
  using SnapshotSupplier = std::function<SceneSnapshot()>;
  using SteadyNow = std::function<SceneReadToken::Clock::time_point()>;
  using AsOfNow = std::function<TimeNanoseconds()>;

  explicit SceneQueryGateway(
      SnapshotSupplier snapshot_supplier,
      std::shared_ptr<const SearchProvider> search_provider = nullptr,
      std::chrono::milliseconds default_token_ttl =
          std::chrono::seconds(30),
      std::size_t max_read_sessions = 128);
  SceneQueryGateway(
      SnapshotSupplier snapshot_supplier,
      std::shared_ptr<const SearchProvider> search_provider,
      std::shared_ptr<const SnapshotAssetProvider> snapshot_asset_provider,
      std::chrono::milliseconds default_token_ttl =
          std::chrono::seconds(30),
      std::size_t max_read_sessions = 128);
  SceneQueryGateway(SnapshotSupplier snapshot_supplier,
                    std::shared_ptr<const SearchProvider> search_provider,
                    std::chrono::milliseconds default_token_ttl,
                    SteadyNow steady_now, AsOfNow as_of_now,
                    std::size_t max_read_sessions = 128);
  SceneQueryGateway(
      SnapshotSupplier snapshot_supplier,
      std::shared_ptr<const SearchProvider> search_provider,
      std::shared_ptr<const SnapshotAssetProvider> snapshot_asset_provider,
      std::chrono::milliseconds default_token_ttl,
      SteadyNow steady_now, AsOfNow as_of_now,
      std::size_t max_read_sessions = 128);
  ~SceneQueryGateway() = default;

  SceneQueryGateway(const SceneQueryGateway&) = delete;
  SceneQueryGateway& operator=(const SceneQueryGateway&) = delete;

  SceneReadToken pin() const;
  SceneReadToken pin(std::chrono::milliseconds ttl) const;

  // beginReadSession pins exactly once. resumeReadSession returns that same
  // token until its fixed (non-sliding) TTL expires. expected_scene_revision
  // lets a client detect accidental reuse of a session from another answer.
  SceneReadSessionResult beginReadSession(
      std::chrono::milliseconds ttl) const;
  SceneReadSessionResult resumeReadSession(
      std::string_view session_id,
      std::optional<SceneRevision> expected_scene_revision =
          std::nullopt) const;
  std::size_t activeReadSessionCount() const;

  QueryResult<QueryObjectView> getObject(
      const SceneReadToken& token, SceneObjectId object_id) const;
  QueryResult<QueryObjectList> getObjectsNear(
      const SceneReadToken& token, const ObjectsNearRequest& request) const;
  QueryResult<QueryFurnitureList> furniture(
      const SceneReadToken& token) const;
  QueryResult<std::vector<CanonicalRoom>> rooms(
      const SceneReadToken& token) const;
  QueryResult<std::vector<CanonicalRelation>> relations(
      const SceneReadToken& token,
      const RelationRequest& request = RelationRequest{}) const;
  QueryResult<SnapshotInspection> inspectSnapshot(
      const SceneReadToken& token, SceneObjectId object_id,
      std::optional<int> image_index = std::nullopt) const;
  QueryResult<SearchObjectMatches> searchObjects(
      const SceneReadToken& token, const SearchRequest& request) const;

  // The exact deterministic document used to verify index rows. Providers
  // may reuse these helpers when constructing a compatible index.
  static std::string semanticDocument(const SceneObject& object);
  static std::string semanticDocumentHash(const SceneObject& object);

 private:
  QueryStatus validateToken(const SceneReadToken& token,
                            std::string* message) const;
  QueryMetadata metadataFor(const SceneReadToken& token) const;
  QueryObjectView objectView(const SceneReadToken& token,
                             SceneObjectId requested_object_id,
                             SceneObjectId canonical_object_id,
                             const SceneObject& object) const;
  std::vector<CanonicalRelation> canonicalRelations(
      const SceneReadToken& token) const;
  std::vector<CanonicalRoom> canonicalRooms(
      const SceneReadToken& token) const;
  void pruneExpiredReadSessionsLocked(
      SceneReadToken::Clock::time_point now) const;

  struct ReadSessionEntry {
    SceneReadToken token;
  };

  SnapshotSupplier snapshot_supplier_;
  std::shared_ptr<const SearchProvider> search_provider_;
  std::shared_ptr<const SnapshotAssetProvider> snapshot_asset_provider_;
  std::chrono::milliseconds default_token_ttl_;
  SteadyNow steady_now_;
  AsOfNow as_of_now_;
  std::uint64_t gateway_id_ = 0;
  std::size_t max_read_sessions_ = 128;
  mutable std::mutex read_sessions_mutex_;
  mutable std::unordered_map<std::string, ReadSessionEntry> read_sessions_;
};

// Tool transports bind one instance per answer/turn. Every method therefore
// reuses exactly the same read token instead of silently pinning latest state.
class LocalSceneQueryHandlers {
 public:
  LocalSceneQueryHandlers(const SceneQueryGateway& gateway,
                          SceneReadToken token)
      : gateway_(&gateway), token_(std::move(token)) {}

  const SceneReadToken& token() const { return token_; }
  QueryResult<QueryObjectView> getObject(SceneObjectId object_id) const;
  QueryResult<QueryObjectList> getObjectsNear(
      const ObjectsNearRequest& request) const;
  QueryResult<QueryFurnitureList> furniture() const;
  QueryResult<std::vector<CanonicalRoom>> rooms() const;
  QueryResult<std::vector<CanonicalRelation>> relations(
      const RelationRequest& request = RelationRequest{}) const;
  QueryResult<SnapshotInspection> inspectSnapshot(
      SceneObjectId object_id,
      std::optional<int> image_index = std::nullopt) const;
  QueryResult<SearchObjectMatches> searchObjects(
      const SearchRequest& request) const;

 private:
  const SceneQueryGateway* gateway_ = nullptr;
  SceneReadToken token_;
};

struct OfflineJsonImportResult {
  SceneSnapshot snapshot;
  std::vector<std::string> warnings;
};

// Legacy JSON is deliberately isolated behind an adapter. It may seed a live
// SceneSnapshot or export one, but it is never a second mutable query store.
class OfflineJsonSceneAdapter {
 public:
  virtual ~OfflineJsonSceneAdapter() = default;
  virtual OfflineJsonImportResult importJson(std::string_view json) const = 0;
  virtual std::string exportJson(const SceneSnapshot& snapshot) const = 0;
};

}  // namespace roomie
