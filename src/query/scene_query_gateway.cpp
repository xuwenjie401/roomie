#include "roomie/query/scene_query_gateway.hpp"

#include "roomie/artifacts/semantic_index.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace roomie {
namespace {

std::atomic<std::uint64_t> g_next_gateway_id{1};

TimeNanoseconds systemNowNanoseconds() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string lowerAscii(std::string_view text) {
  std::string result;
  result.reserve(text.size());
  for (const unsigned char character : text) {
    result.push_back(static_cast<char>(std::tolower(character)));
  }
  return result;
}

std::string normalizedKey(std::string_view text) {
  std::string result;
  result.reserve(text.size());
  bool previous_separator = false;
  for (const unsigned char character : text) {
    if (std::isalnum(character) || character >= 0x80U) {
      result.push_back(static_cast<char>(std::tolower(character)));
      previous_separator = false;
    } else if (!result.empty() && !previous_separator) {
      result.push_back('_');
      previous_separator = true;
    }
  }
  while (!result.empty() && result.back() == '_') {
    result.pop_back();
  }
  return result;
}

bool truthy(std::string_view value) {
  const std::string normalized = normalizedKey(value);
  return normalized == "true" || normalized == "yes" || normalized == "1";
}

bool validSurface(const SurfaceStamp& stamp) {
  return stamp.map_epoch.valid() || stamp.surface_revision != 0 ||
         stamp.source_map_revision != 0;
}

int freshnessRank(Freshness freshness) {
  switch (freshness) {
    case Freshness::kCurrent:
      return 0;
    case Freshness::kStale:
      return 1;
    case Freshness::kPending:
      return 2;
    case Freshness::kMissing:
      return 3;
  }
  return 3;
}

Freshness leastFresh(Freshness lhs, Freshness rhs) {
  return freshnessRank(lhs) >= freshnessRank(rhs) ? lhs : rhs;
}

void appendUnique(std::vector<std::string>* values, std::string value) {
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(std::move(value));
  }
}

void absorbFreshness(QueryMetadata* metadata,
                     const QueryObjectView& object) {
  metadata->geometry_freshness =
      leastFresh(metadata->geometry_freshness, object.freshness.geometry);
  metadata->semantic_freshness =
      leastFresh(metadata->semantic_freshness, object.freshness.semantic);
  metadata->description_freshness =
      leastFresh(metadata->description_freshness,
                 object.freshness.description);
  if (object.freshness.artifactPending()) {
    metadata->artifact_pending = true;
    metadata->pending_object_ids.push_back(object.object_id);
  }
}

void finishMetadata(QueryMetadata* metadata, bool saw_object) {
  if (!saw_object) {
    metadata->geometry_freshness = Freshness::kMissing;
    metadata->semantic_freshness = Freshness::kMissing;
    metadata->description_freshness = Freshness::kMissing;
  }
  std::sort(metadata->pending_object_ids.begin(),
            metadata->pending_object_ids.end());
  metadata->pending_object_ids.erase(
      std::unique(metadata->pending_object_ids.begin(),
                  metadata->pending_object_ids.end()),
      metadata->pending_object_ids.end());
}

std::optional<std::string> effectiveDescription(const SceneObject& object) {
  if (object.annotation && object.annotation->description_override) {
    return object.annotation->description_override;
  }
  if (object.artifact && !object.artifact->description.empty()) {
    return object.artifact->description;
  }
  return std::nullopt;
}

std::string effectiveLabel(const SceneObject& object) {
  if (object.annotation && object.annotation->label_override &&
      !object.annotation->label_override->empty()) {
    return *object.annotation->label_override;
  }
  return object.semantic ? object.semantic->label : std::string{};
}

int effectiveSemanticId(const SceneObject& object) {
  if (object.annotation && object.annotation->semantic_id_override) {
    return *object.annotation->semantic_id_override;
  }
  return object.semantic ? object.semantic->semantic_id : -1;
}

std::vector<std::string> lexicalTokens(std::string_view text) {
  std::vector<std::string> result;
  std::string token;
  for (const unsigned char character : text) {
    if (std::isalnum(character) || character >= 0x80U) {
      token.push_back(static_cast<char>(std::tolower(character)));
    } else if (!token.empty()) {
      result.push_back(std::move(token));
      token.clear();
    }
  }
  if (!token.empty()) {
    result.push_back(std::move(token));
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

float lexicalScore(std::string_view query, std::string_view document) {
  const std::string normalized_query = lowerAscii(query);
  const std::string normalized_document = lowerAscii(document);
  if (normalized_query.empty() || normalized_document.empty()) {
    return 0.0f;
  }

  float score = 0.0f;
  if (normalized_document.find(normalized_query) != std::string::npos) {
    score = 0.82f;
  }

  const std::vector<std::string> query_tokens = lexicalTokens(query);
  const std::vector<std::string> document_tokens = lexicalTokens(document);
  if (!query_tokens.empty()) {
    std::size_t overlap = 0;
    for (const std::string& token : query_tokens) {
      if (std::binary_search(document_tokens.begin(), document_tokens.end(),
                             token)) {
        ++overlap;
      }
    }
    const float token_score =
        0.75f * static_cast<float>(overlap) /
        static_cast<float>(query_tokens.size());
    score = std::max(score, token_score);
  }
  return score;
}

bool relationMeansMemberToRoom(std::string_view relation_type) {
  const std::string key = normalizedKey(relation_type);
  return key == "in_room" || key == "inside_room" ||
         key == "contained_in_room" || key == "belongs_to_room" ||
         key == "located_in_room";
}

bool relationMeansRoomContains(std::string_view relation_type) {
  const std::string key = normalizedKey(relation_type);
  return key == "contains" || key == "room_contains" ||
         key == "room_contains_object";
}

std::optional<std::string> attribute(
    const std::shared_ptr<const AnnotationComponent>& annotation,
    std::string_view name) {
  if (!annotation) {
    return std::nullopt;
  }
  const auto it = annotation->attributes.find(std::string(name));
  if (it == annotation->attributes.end() || it->second.empty()) {
    return std::nullopt;
  }
  return it->second;
}

bool isRoomObject(const SceneObject& object) {
  const auto entity_type = attribute(object.annotation, "entity_type");
  const auto object_type = attribute(object.annotation, "object_type");
  const auto is_room = attribute(object.annotation, "is_room");
  return (entity_type && normalizedKey(*entity_type) == "room") ||
         (object_type && normalizedKey(*object_type) == "room") ||
         (is_room && truthy(*is_room));
}

std::string objectRoomId(const SceneObject& object,
                         SceneObjectId object_id) {
  if (const auto room_id = attribute(object.annotation, "room_id")) {
    return *room_id;
  }
  return "object:" + std::to_string(object_id);
}

bool objectVisible(const SceneObject& object, bool include_inactive,
                   bool include_unpublishable) {
  if (!include_inactive && object.lifecycle && !object.lifecycle->active) {
    return false;
  }
  if (!include_unpublishable && object.lifecycle &&
      !object.lifecycle->publishable) {
    return false;
  }
  return true;
}

std::vector<std::string> snapshotAssetIds(const SceneSnapshot& snapshot) {
  std::set<std::string> unique_ids;
  for (const auto& [object_id, object] : snapshot.objects()) {
    (void)object_id;
    if (!object || !object->artifact) {
      continue;
    }
    for (const ObjectSnapshotRef& reference : object->artifact->snapshots) {
      if (!reference.source_frame_asset_id.empty()) {
        unique_ids.insert(reference.source_frame_asset_id);
      }
    }
  }
  return {unique_ids.begin(), unique_ids.end()};
}

template <typename T>
QueryResult<T> tokenFailure(QueryStatus status, std::string message,
                            QueryMetadata metadata) {
  QueryResult<T> result;
  result.status = status;
  result.message = std::move(message);
  result.metadata = std::move(metadata);
  return result;
}

}  // namespace

SceneQueryGateway::SceneQueryGateway(
    SnapshotSupplier snapshot_supplier,
    std::shared_ptr<const SearchProvider> search_provider,
    std::chrono::milliseconds default_token_ttl,
    std::size_t max_read_sessions)
    : SceneQueryGateway(std::move(snapshot_supplier),
                        std::move(search_provider), nullptr, default_token_ttl,
                        []() { return SceneReadToken::Clock::now(); },
                        []() { return systemNowNanoseconds(); },
                        max_read_sessions) {}

SceneQueryGateway::SceneQueryGateway(
    SnapshotSupplier snapshot_supplier,
    std::shared_ptr<const SearchProvider> search_provider,
    std::shared_ptr<const SnapshotAssetProvider> snapshot_asset_provider,
    std::chrono::milliseconds default_token_ttl,
    std::size_t max_read_sessions)
    : SceneQueryGateway(std::move(snapshot_supplier),
                        std::move(search_provider),
                        std::move(snapshot_asset_provider), default_token_ttl,
                        []() { return SceneReadToken::Clock::now(); },
                        []() { return systemNowNanoseconds(); },
                        max_read_sessions) {}

SceneQueryGateway::SceneQueryGateway(
    SnapshotSupplier snapshot_supplier,
    std::shared_ptr<const SearchProvider> search_provider,
    std::chrono::milliseconds default_token_ttl, SteadyNow steady_now,
    AsOfNow as_of_now, std::size_t max_read_sessions)
    : SceneQueryGateway(std::move(snapshot_supplier),
                        std::move(search_provider), nullptr,
                        default_token_ttl, std::move(steady_now),
                        std::move(as_of_now), max_read_sessions) {}

SceneQueryGateway::SceneQueryGateway(
    SnapshotSupplier snapshot_supplier,
    std::shared_ptr<const SearchProvider> search_provider,
    std::shared_ptr<const SnapshotAssetProvider> snapshot_asset_provider,
    std::chrono::milliseconds default_token_ttl, SteadyNow steady_now,
    AsOfNow as_of_now, std::size_t max_read_sessions)
    : snapshot_supplier_(std::move(snapshot_supplier)),
      search_provider_(std::move(search_provider)),
      snapshot_asset_provider_(std::move(snapshot_asset_provider)),
      default_token_ttl_(default_token_ttl),
      steady_now_(std::move(steady_now)),
      as_of_now_(std::move(as_of_now)),
      gateway_id_(g_next_gateway_id.fetch_add(1, std::memory_order_relaxed)),
      max_read_sessions_(max_read_sessions) {
  if (!snapshot_supplier_) {
    throw std::invalid_argument("SceneQueryGateway requires a snapshot supplier");
  }
  if (!steady_now_ || !as_of_now_) {
    throw std::invalid_argument("SceneQueryGateway requires clock functions");
  }
  if (default_token_ttl_ <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument("SceneQueryGateway token TTL must be positive");
  }
  if (max_read_sessions_ == 0) {
    throw std::invalid_argument(
        "SceneQueryGateway read-session capacity must be positive");
  }
  if (gateway_id_ == 0) {
    gateway_id_ = g_next_gateway_id.fetch_add(1, std::memory_order_relaxed);
  }
}

SceneReadToken SceneQueryGateway::pin() const {
  return pin(default_token_ttl_);
}

SceneReadToken SceneQueryGateway::pin(std::chrono::milliseconds ttl) const {
  SceneReadToken token;
  token.gateway_id_ = gateway_id_;
  token.snapshot_ = snapshot_supplier_();
  token.as_of_ns_ = as_of_now_();
  token.expires_at_ = steady_now_() + ttl;
  if (search_provider_) {
    token.index_ = search_provider_->pinCurrent();
  }
  if (snapshot_asset_provider_) {
    const std::vector<std::string> asset_ids =
        snapshotAssetIds(token.snapshot_);
    if (!asset_ids.empty()) {
      token.snapshot_assets_ = snapshot_asset_provider_->pin(asset_ids);
    }
  }
  if (token.index_) {
    token.index_generation_ = token.index_->generation();
    token.index_as_of_ns_ = token.index_->asOfNanoseconds();
  }
  return token;
}

void SceneQueryGateway::pruneExpiredReadSessionsLocked(
    SceneReadToken::Clock::time_point now) const {
  for (auto session = read_sessions_.begin();
       session != read_sessions_.end();) {
    if (now >= session->second.token.expiresAt()) {
      session = read_sessions_.erase(session);
    } else {
      ++session;
    }
  }
}

SceneReadSessionResult SceneQueryGateway::beginReadSession(
    std::chrono::milliseconds ttl) const {
  SceneReadSessionResult result;
  if (ttl <= std::chrono::milliseconds::zero()) {
    result.status = SceneReadSessionStatus::kInvalidArgument;
    result.message = "read-session TTL must be positive";
    return result;
  }

  std::lock_guard<std::mutex> lock(read_sessions_mutex_);
  pruneExpiredReadSessionsLocked(steady_now_());
  if (read_sessions_.size() >= max_read_sessions_) {
    result.status = SceneReadSessionStatus::kCapacityExceeded;
    result.message = "read-session capacity is exhausted";
    return result;
  }

  result.token = pin(ttl);
  do {
    result.session_id = runIdString(makeRunId());
  } while (read_sessions_.find(result.session_id) != read_sessions_.end());
  read_sessions_.emplace(result.session_id,
                         ReadSessionEntry{result.token});
  return result;
}

SceneReadSessionResult SceneQueryGateway::resumeReadSession(
    std::string_view session_id,
    std::optional<SceneRevision> expected_scene_revision) const {
  SceneReadSessionResult result;
  result.session_id = std::string(session_id);
  if (session_id.empty()) {
    result.status = SceneReadSessionStatus::kInvalidArgument;
    result.message = "session_id must not be empty";
    return result;
  }

  std::lock_guard<std::mutex> lock(read_sessions_mutex_);
  const auto session = read_sessions_.find(result.session_id);
  if (session == read_sessions_.end()) {
    pruneExpiredReadSessionsLocked(steady_now_());
    result.status = SceneReadSessionStatus::kNotFound;
    result.message = "read session is unknown";
    return result;
  }
  const auto now = steady_now_();
  if (now >= session->second.token.expiresAt()) {
    read_sessions_.erase(session);
    pruneExpiredReadSessionsLocked(now);
    result.status = SceneReadSessionStatus::kExpired;
    result.message = "read session expired";
    return result;
  }
  pruneExpiredReadSessionsLocked(now);
  const auto active = read_sessions_.find(result.session_id);
  if (active == read_sessions_.end()) {
    result.status = SceneReadSessionStatus::kExpired;
    result.message = "read session expired";
    return result;
  }
  if (expected_scene_revision &&
      *expected_scene_revision != active->second.token.sceneRevision()) {
    result.status = SceneReadSessionStatus::kRevisionMismatch;
    result.message = "read session is pinned to scene revision " +
                     std::to_string(active->second.token.sceneRevision()) +
                     ", not expected revision " +
                     std::to_string(*expected_scene_revision);
    return result;
  }
  result.token = active->second.token;
  return result;
}

std::size_t SceneQueryGateway::activeReadSessionCount() const {
  std::lock_guard<std::mutex> lock(read_sessions_mutex_);
  pruneExpiredReadSessionsLocked(steady_now_());
  return read_sessions_.size();
}

QueryStatus SceneQueryGateway::validateToken(const SceneReadToken& token,
                                             std::string* message) const {
  if (token.gateway_id_ == 0 || token.gateway_id_ != gateway_id_) {
    if (message) {
      *message = "read token was not issued by this gateway";
    }
    return QueryStatus::kForeignToken;
  }
  if (steady_now_() >= token.expires_at_) {
    if (message) {
      *message = "scene read token expired";
    }
    return QueryStatus::kExpiredToken;
  }
  if (message) {
    message->clear();
  }
  return QueryStatus::kOk;
}

QueryMetadata SceneQueryGateway::metadataFor(
    const SceneReadToken& token) const {
  QueryMetadata metadata;
  metadata.scene_revision = token.snapshot_.revision();
  metadata.durable_scene_revision = token.snapshot_.durableRevision();
  metadata.index_generation = token.index_generation_;
  metadata.as_of_ns = token.as_of_ns_;
  metadata.index_as_of_ns = token.index_as_of_ns_;
  metadata.index_available = static_cast<bool>(token.index_);
  metadata.geometry_freshness = Freshness::kCurrent;
  metadata.semantic_freshness = Freshness::kCurrent;
  metadata.description_freshness = Freshness::kCurrent;
  return metadata;
}

std::string SceneQueryGateway::semanticDocument(const SceneObject& object) {
  const SceneObjectId object_id =
      object.identity ? object.identity->object_id : -1;
  return makeSemanticDocumentForObject(object, object_id, 0).text;
}

std::string SceneQueryGateway::semanticDocumentHash(const SceneObject& object) {
  const SceneObjectId object_id =
      object.identity ? object.identity->object_id : -1;
  return makeSemanticDocumentForObject(object, object_id, 0).document_hash;
}

QueryObjectView SceneQueryGateway::objectView(
    const SceneReadToken& token, SceneObjectId requested_object_id,
    SceneObjectId canonical_object_id, const SceneObject& object) const {
  QueryObjectView view;
  view.requested_object_id = requested_object_id;
  view.object_id = canonical_object_id;
  view.resolved_alias = requested_object_id != canonical_object_id;
  view.semantic_id = effectiveSemanticId(object);
  view.label = effectiveLabel(object);
  view.canonical_description = effectiveDescription(object);
  if (view.canonical_description && !view.canonical_description->empty()) {
    view.display_description = *view.canonical_description;
  } else {
    view.display_description = view.label;
    view.description_is_label_fallback = !view.label.empty();
  }
  if (object.lifecycle) {
    view.active = object.lifecycle->active;
    view.publishable = object.lifecycle->publishable;
  }
  if (object.geometry) {
    view.center_world = object.geometry->center_world;
    view.size_m = object.geometry->size_m;
    view.yaw_rad = object.geometry->yaw_rad;
  }
  view.revisions = object.revisions();
  if (object.artifact) {
    view.snapshot_set_hash = object.artifact->snapshot_set_hash;
  }
  view.semantic_document_hash = semanticDocumentHash(object);

  if (!object.geometry) {
    view.freshness.geometry = Freshness::kMissing;
  } else if (object.geometry->status == InstanceGeometryStatus::kUnchecked ||
             object.geometry->evaluated_obb_revision <
                 object.geometry->obb_revision) {
    view.freshness.geometry = Freshness::kPending;
    appendUnique(&view.freshness.pending_artifacts, "geometry");
  } else if (validSurface(token.snapshot_.latestSurface()) &&
             object.geometry->evaluated_surface !=
                 token.snapshot_.latestSurface()) {
    view.freshness.geometry = Freshness::kStale;
    appendUnique(&view.freshness.pending_artifacts, "geometry");
  } else {
    view.freshness.geometry = Freshness::kCurrent;
  }

  const bool has_semantic_content =
      !view.label.empty() || !semanticDocument(object).empty();
  try {
    if (token.index_) {
      view.indexed_document_hash =
          token.index_->documentHash(canonical_object_id);
    }
  } catch (const std::exception&) {
    view.indexed_document_hash.reset();
  }
  if (!has_semantic_content) {
    view.freshness.semantic = Freshness::kMissing;
  } else if (view.indexed_document_hash &&
             *view.indexed_document_hash == view.semantic_document_hash) {
    view.freshness.semantic = Freshness::kCurrent;
  } else if (view.indexed_document_hash) {
    view.freshness.semantic = Freshness::kStale;
    appendUnique(&view.freshness.pending_artifacts, "embedding");
  } else {
    view.freshness.semantic = Freshness::kPending;
    appendUnique(&view.freshness.pending_artifacts, "embedding");
  }

  if (object.annotation && object.annotation->description_override) {
    view.freshness.description = Freshness::kCurrent;
  } else if (object.artifact &&
             object.artifact->pending_description_scene_revision != 0) {
    view.freshness.description = object.artifact->description.empty()
                                     ? Freshness::kPending
                                     : Freshness::kStale;
    appendUnique(&view.freshness.pending_artifacts, "description");
  } else if (object.artifact && !object.artifact->description.empty()) {
    if (object.artifact->description_stale ||
        object.artifact->description_input_hash.empty()) {
      view.freshness.description = Freshness::kStale;
      appendUnique(&view.freshness.pending_artifacts, "description");
    } else {
      view.freshness.description = Freshness::kCurrent;
    }
  } else if (object.artifact &&
             (object.artifact->appearance_revision != 0 ||
              !object.artifact->snapshots.empty())) {
    view.freshness.description = Freshness::kPending;
    appendUnique(&view.freshness.pending_artifacts, "description");
  } else {
    view.freshness.description = Freshness::kMissing;
  }

  if (object.artifact && object.artifact->appearance_revision != 0 &&
      (object.artifact->snapshots.empty() ||
       object.artifact->snapshot_set_hash.empty())) {
    appendUnique(&view.freshness.pending_artifacts, "snapshot_set");
  }
  std::sort(view.freshness.pending_artifacts.begin(),
            view.freshness.pending_artifacts.end());
  return view;
}

QueryResult<QueryObjectView> SceneQueryGateway::getObject(
    const SceneReadToken& token, SceneObjectId object_id) const {
  QueryMetadata metadata = metadataFor(token);
  std::string message;
  const QueryStatus validation = validateToken(token, &message);
  if (validation != QueryStatus::kOk) {
    return tokenFailure<QueryObjectView>(validation, std::move(message),
                                         std::move(metadata));
  }
  if (object_id < 0) {
    return tokenFailure<QueryObjectView>(
        QueryStatus::kInvalidArgument, "object id must be non-negative",
        std::move(metadata));
  }

  const auto canonical = token.snapshot_.resolveCanonicalId(object_id);
  if (!canonical) {
    return tokenFailure<QueryObjectView>(
        QueryStatus::kNotFound, "object alias chain is invalid",
        std::move(metadata));
  }
  const SceneObjectPtr object = token.snapshot_.findObject(object_id);
  if (!object) {
    return tokenFailure<QueryObjectView>(
        QueryStatus::kNotFound, "object is missing or tombstoned",
        std::move(metadata));
  }

  QueryResult<QueryObjectView> result;
  result.metadata = std::move(metadata);
  result.value = objectView(token, object_id, *canonical, *object);
  absorbFreshness(&result.metadata, result.value);
  finishMetadata(&result.metadata, true);
  return result;
}

QueryResult<QueryObjectList> SceneQueryGateway::getObjectsNear(
    const SceneReadToken& token, const ObjectsNearRequest& request) const {
  QueryMetadata metadata = metadataFor(token);
  std::string message;
  const QueryStatus validation = validateToken(token, &message);
  if (validation != QueryStatus::kOk) {
    return tokenFailure<QueryObjectList>(validation, std::move(message),
                                         std::move(metadata));
  }
  if (!request.center_world.allFinite() || !std::isfinite(request.radius_m) ||
      request.radius_m < 0.0f || request.limit == 0) {
    return tokenFailure<QueryObjectList>(
        QueryStatus::kInvalidArgument,
        "near query requires a finite center, non-negative radius, and limit",
        std::move(metadata));
  }

  struct Candidate {
    float distance = 0.0f;
    SceneObjectId object_id = -1;
    SceneObjectPtr object;
  };
  std::vector<Candidate> candidates;
  for (const auto& entry : token.snapshot_.objects()) {
    if (!entry.second || token.snapshot_.isTombstoned(entry.first) ||
        !entry.second->geometry ||
        !objectVisible(*entry.second, request.include_inactive,
                       request.include_unpublishable)) {
      continue;
    }
    const float distance =
        (entry.second->geometry->center_world - request.center_world).norm();
    if (distance <= request.radius_m) {
      candidates.push_back(Candidate{distance, entry.first, entry.second});
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs) {
              return std::tie(lhs.distance, lhs.object_id) <
                     std::tie(rhs.distance, rhs.object_id);
            });

  QueryResult<QueryObjectList> result;
  result.metadata = std::move(metadata);
  const std::size_t count = std::min(request.limit, candidates.size());
  result.value.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const Candidate& candidate = candidates[index];
    result.value.push_back(objectView(token, candidate.object_id,
                                      candidate.object_id,
                                      *candidate.object));
    absorbFreshness(&result.metadata, result.value.back());
  }
  finishMetadata(&result.metadata, !result.value.empty());
  return result;
}

std::vector<CanonicalRelation> SceneQueryGateway::canonicalRelations(
    const SceneReadToken& token) const {
  using Key = std::tuple<int, int, int, int, std::string>;
  std::map<Key, CanonicalRelation> deduplicated;
  std::set<int> room_ids;
  for (const RoomNode& room : token.snapshot_.graphMetadata().rooms) {
    if (room.room_id >= 0) {
      room_ids.insert(room.room_id);
    }
  }
  for (const ObjectRelation& relation :
       token.snapshot_.graphMetadata().relations) {
    SceneEntityRef source = relationSource(relation);
    SceneEntityRef target = relationTarget(relation);
    if (!source.valid() || !target.valid()) {
      continue;
    }
    bool source_resolved_alias = false;
    bool target_resolved_alias = false;
    const auto canonicalize = [&](SceneEntityRef* endpoint,
                                  bool* resolved_alias) {
      if (endpoint->type == SceneEntityType::kRoom) {
        return room_ids.count(endpoint->id) != 0U;
      }
      const auto canonical =
          token.snapshot_.resolveCanonicalId(endpoint->id);
      if (!canonical || token.snapshot_.isTombstoned(*canonical) ||
          !token.snapshot_.findExactObject(*canonical)) {
        return false;
      }
      *resolved_alias = *canonical != endpoint->id;
      endpoint->id = *canonical;
      return true;
    };
    if (!canonicalize(&source, &source_resolved_alias) ||
        !canonicalize(&target, &target_resolved_alias)) {
      continue;
    }
    CanonicalRelation current;
    current.source = source;
    current.target = target;
    current.source_object_id =
        source.type == SceneEntityType::kObject ? source.id : -1;
    current.target_object_id =
        target.type == SceneEntityType::kObject ? target.id : -1;
    current.relation_type = relation.relation_type;
    current.confidence = relation.confidence;
    current.description = relation.description;
    current.revision = relation.revision;
    current.derived = relation.derived;
    current.source_resolved_alias = source_resolved_alias;
    current.target_resolved_alias = target_resolved_alias;

    const Key key{static_cast<int>(source.type), source.id,
                  static_cast<int>(target.type), target.id,
                  normalizedKey(relation.relation_type)};
    auto inserted = deduplicated.emplace(key, current);
    if (!inserted.second &&
        current.confidence > inserted.first->second.confidence) {
      inserted.first->second = std::move(current);
    } else if (!inserted.second) {
      inserted.first->second.source_resolved_alias =
          inserted.first->second.source_resolved_alias ||
          current.source_resolved_alias;
      inserted.first->second.target_resolved_alias =
          inserted.first->second.target_resolved_alias ||
          current.target_resolved_alias;
    }
  }

  std::vector<CanonicalRelation> result;
  result.reserve(deduplicated.size());
  for (auto& entry : deduplicated) {
    result.push_back(std::move(entry.second));
  }
  return result;
}

QueryResult<std::vector<CanonicalRelation>> SceneQueryGateway::relations(
    const SceneReadToken& token, const RelationRequest& request) const {
  QueryMetadata metadata = metadataFor(token);
  std::string message;
  const QueryStatus validation = validateToken(token, &message);
  if (validation != QueryStatus::kOk) {
    return tokenFailure<std::vector<CanonicalRelation>>(
        validation, std::move(message), std::move(metadata));
  }

  std::optional<SceneObjectId> canonical_filter;
  if (request.object_id) {
    if (*request.object_id < 0) {
      return tokenFailure<std::vector<CanonicalRelation>>(
          QueryStatus::kInvalidArgument,
          "relation object id must be non-negative", std::move(metadata));
    }
    canonical_filter =
        token.snapshot_.resolveCanonicalId(*request.object_id);
    if (!canonical_filter || token.snapshot_.isTombstoned(*canonical_filter)) {
      return tokenFailure<std::vector<CanonicalRelation>>(
          QueryStatus::kNotFound, "relation object is missing or tombstoned",
          std::move(metadata));
    }
  }

  QueryResult<std::vector<CanonicalRelation>> result;
  result.metadata = std::move(metadata);
  const std::string type_filter =
      request.relation_type ? normalizedKey(*request.relation_type) : "";
  std::set<SceneObjectId> involved_objects;
  for (CanonicalRelation relation : canonicalRelations(token)) {
    if (!type_filter.empty() &&
        normalizedKey(relation.relation_type) != type_filter) {
      continue;
    }
    if (canonical_filter) {
      bool matches = false;
      switch (request.direction) {
        case RelationDirection::kEither:
          matches =
              (relation.source.type == SceneEntityType::kObject &&
               relation.source.id == *canonical_filter) ||
              (relation.target.type == SceneEntityType::kObject &&
               relation.target.id == *canonical_filter);
          break;
        case RelationDirection::kOutgoing:
          matches = relation.source.type == SceneEntityType::kObject &&
                    relation.source.id == *canonical_filter;
          break;
        case RelationDirection::kIncoming:
          matches = relation.target.type == SceneEntityType::kObject &&
                    relation.target.id == *canonical_filter;
          break;
      }
      if (!matches) {
        continue;
      }
    }
    if (relation.source.type == SceneEntityType::kObject) {
      involved_objects.insert(relation.source.id);
    }
    if (relation.target.type == SceneEntityType::kObject) {
      involved_objects.insert(relation.target.id);
    }
    result.value.push_back(std::move(relation));
  }

  bool saw_object = false;
  for (SceneObjectId object_id : involved_objects) {
    const SceneObjectPtr object = token.snapshot_.findObject(object_id);
    if (!object) {
      continue;
    }
    const QueryObjectView view =
        objectView(token, object_id, object_id, *object);
    absorbFreshness(&result.metadata, view);
    saw_object = true;
  }
  finishMetadata(&result.metadata, saw_object);
  return result;
}

std::vector<CanonicalRoom> SceneQueryGateway::canonicalRooms(
    const SceneReadToken& token) const {
  std::map<std::string, CanonicalRoom> rooms_by_id;
  std::map<SceneObjectId, std::string> room_object_to_id;
  std::map<int, std::string> room_node_to_id;
  std::map<std::string, std::string> room_alias_to_id;

  for (const RoomNode& node : token.snapshot_.graphMetadata().rooms) {
    if (node.room_id < 0) {
      continue;
    }
    CanonicalRoom room;
    const auto external_id = node.attributes.find("room_id");
    room.room_id = external_id != node.attributes.end() &&
                           !external_id->second.empty()
                       ? external_id->second
                       : std::to_string(node.room_id);
    room.name = node.label.empty() ? room.room_id : node.label;
    room.attributes = node.attributes;
    rooms_by_id[room.room_id] = room;
    room_node_to_id[node.room_id] = room.room_id;
    room_alias_to_id[normalizedKey(room.room_id)] = room.room_id;
    room_alias_to_id[normalizedKey(room.name)] = room.room_id;
    room_alias_to_id[std::to_string(node.room_id)] = room.room_id;
  }

  for (const auto& entry : token.snapshot_.objects()) {
    if (!entry.second || token.snapshot_.isTombstoned(entry.first) ||
        !isRoomObject(*entry.second)) {
      continue;
    }
    CanonicalRoom room;
    room.room_id = objectRoomId(*entry.second, entry.first);
    room.room_object_id = entry.first;
    if (const auto name = attribute(entry.second->annotation, "room_name")) {
      room.name = *name;
    } else {
      room.name = effectiveLabel(*entry.second);
    }
    if (room.name.empty()) {
      room.name = room.room_id;
    }
    if (entry.second->annotation) {
      room.attributes = entry.second->annotation->attributes;
    }
    rooms_by_id[room.room_id] = room;
    room_object_to_id[entry.first] = room.room_id;
    room_alias_to_id[normalizedKey(room.room_id)] = room.room_id;
    room_alias_to_id[normalizedKey(room.name)] = room.room_id;
    room_alias_to_id[std::to_string(entry.first)] = room.room_id;
    room_alias_to_id["object_" + std::to_string(entry.first)] = room.room_id;
  }

  const auto ensure_compatibility_room =
      [&rooms_by_id, &room_alias_to_id](const std::string& raw_id)
          -> std::string {
    const std::string alias = normalizedKey(raw_id);
    const auto alias_it = room_alias_to_id.find(alias);
    if (alias_it != room_alias_to_id.end()) {
      return alias_it->second;
    }
    CanonicalRoom room;
    room.room_id = raw_id;
    room.name = raw_id;
    rooms_by_id.emplace(room.room_id, room);
    room_alias_to_id[alias] = room.room_id;
    return room.room_id;
  };

  for (const auto& entry : token.snapshot_.objects()) {
    if (!entry.second || token.snapshot_.isTombstoned(entry.first) ||
        isRoomObject(*entry.second)) {
      continue;
    }
    std::optional<std::string> room_id =
        attribute(entry.second->annotation, "room_id");
    if (!room_id) {
      room_id = attribute(entry.second->annotation, "room");
    }
    if (room_id) {
      rooms_by_id[ensure_compatibility_room(*room_id)].object_ids.push_back(
          entry.first);
    }
  }

  for (const CanonicalRelation& relation : canonicalRelations(token)) {
    if (relationMeansMemberToRoom(relation.relation_type) &&
        relation.source.type == SceneEntityType::kObject) {
      if (relation.target.type == SceneEntityType::kRoom) {
        const auto target = room_node_to_id.find(relation.target.id);
        if (target != room_node_to_id.end()) {
          rooms_by_id[target->second].object_ids.push_back(
              relation.source.id);
        }
      } else {
        const auto target = room_object_to_id.find(relation.target.id);
        if (target != room_object_to_id.end()) {
          rooms_by_id[target->second].object_ids.push_back(
              relation.source.id);
        }
      }
    } else if (relationMeansRoomContains(relation.relation_type) &&
               relation.target.type == SceneEntityType::kObject) {
      if (relation.source.type == SceneEntityType::kRoom) {
        const auto source = room_node_to_id.find(relation.source.id);
        if (source != room_node_to_id.end()) {
          rooms_by_id[source->second].object_ids.push_back(
              relation.target.id);
        }
      } else {
        const auto source = room_object_to_id.find(relation.source.id);
        if (source != room_object_to_id.end()) {
          rooms_by_id[source->second].object_ids.push_back(
              relation.target.id);
        }
      }
    }
  }

  std::vector<CanonicalRoom> result;
  result.reserve(rooms_by_id.size());
  for (auto& entry : rooms_by_id) {
    CanonicalRoom& room = entry.second;
    std::sort(room.object_ids.begin(), room.object_ids.end());
    room.object_ids.erase(
        std::unique(room.object_ids.begin(), room.object_ids.end()),
        room.object_ids.end());
    result.push_back(std::move(room));
  }
  return result;
}

QueryResult<std::vector<CanonicalRoom>> SceneQueryGateway::rooms(
    const SceneReadToken& token) const {
  QueryMetadata metadata = metadataFor(token);
  std::string message;
  const QueryStatus validation = validateToken(token, &message);
  if (validation != QueryStatus::kOk) {
    return tokenFailure<std::vector<CanonicalRoom>>(
        validation, std::move(message), std::move(metadata));
  }

  QueryResult<std::vector<CanonicalRoom>> result;
  result.metadata = std::move(metadata);
  result.value = canonicalRooms(token);
  bool saw_object = false;
  std::set<SceneObjectId> object_ids;
  for (const CanonicalRoom& room : result.value) {
    if (room.room_object_id) {
      object_ids.insert(*room.room_object_id);
    }
    object_ids.insert(room.object_ids.begin(), room.object_ids.end());
  }
  for (SceneObjectId object_id : object_ids) {
    const SceneObjectPtr object = token.snapshot_.findObject(object_id);
    if (!object) {
      continue;
    }
    absorbFreshness(&result.metadata,
                    objectView(token, object_id, object_id, *object));
    saw_object = true;
  }
  finishMetadata(&result.metadata, saw_object);
  return result;
}

QueryResult<SnapshotInspection> SceneQueryGateway::inspectSnapshot(
    const SceneReadToken& token, SceneObjectId object_id,
    std::optional<int> image_index) const {
  const QueryResult<QueryObjectView> object_result =
      getObject(token, object_id);
  if (!object_result.ok()) {
    return tokenFailure<SnapshotInspection>(
        object_result.status, object_result.message, object_result.metadata);
  }

  const SceneObjectPtr object = token.snapshot_.findObject(object_id);
  QueryResult<SnapshotInspection> result;
  result.metadata = object_result.metadata;
  result.value.object = object_result.value;
  if (!object || !object->artifact) {
    result.metadata.artifact_pending = true;
    result.metadata.pending_object_ids.push_back(result.value.object.object_id);
    finishMetadata(&result.metadata, true);
    return result;
  }
  result.value.snapshot_set_hash = object->artifact->snapshot_set_hash;

  std::map<int, ObjectSnapshotImage> assets;
  for (const ObjectSnapshotImage& asset :
       token.snapshot_.graphMetadata().snapshot_images) {
    assets[asset.image_index] = asset;
  }

  for (const ObjectSnapshotRef& reference : object->artifact->snapshots) {
    if (image_index && reference.image_index != *image_index) {
      continue;
    }
    SnapshotAssetInspection inspection;
    inspection.reference = reference;
    if (!reference.source_frame_asset_id.empty()) {
      if (token.snapshot_assets_) {
        std::optional<ResolvedSnapshotAsset> resolved =
            token.snapshot_assets_->resolve(reference.source_frame_asset_id);
        if (resolved && resolved->valid() &&
            resolved->source_frame_asset_id ==
                reference.source_frame_asset_id) {
          ObjectSnapshotImage compatibility_asset;
          compatibility_asset.image_index = reference.image_index;
          compatibility_asset.uri = resolved->uri;
          compatibility_asset.width = resolved->width;
          compatibility_asset.height = resolved->height;
          compatibility_asset.encoding = resolved->encoding;
          compatibility_asset.time_ns = reference.time_ns;
          compatibility_asset.camera_id = reference.camera_id;
          compatibility_asset.source_path = resolved->source_path;
          inspection.physical_asset = std::move(resolved);
          inspection.asset = std::move(compatibility_asset);
          inspection.available = true;
        }
      }
    } else {
      const auto asset = assets.find(reference.image_index);
      if (asset != assets.end()) {
        inspection.asset = asset->second;
        inspection.available = !asset->second.uri.empty() ||
                               !asset->second.source_path.empty();
      }
    }
    if (!inspection.available) {
      result.metadata.artifact_pending = true;
      result.metadata.pending_object_ids.push_back(
          result.value.object.object_id);
    }
    result.value.snapshots.push_back(std::move(inspection));
  }

  if (image_index && result.value.snapshots.empty()) {
    result.status = QueryStatus::kNotFound;
    result.message = "snapshot image is not present in the pinned scene";
    return result;
  }
  if (result.value.snapshots.empty()) {
    result.metadata.artifact_pending = true;
    result.metadata.pending_object_ids.push_back(result.value.object.object_id);
  }
  finishMetadata(&result.metadata, true);
  return result;
}

QueryResult<SearchObjectMatches> SceneQueryGateway::searchObjects(
    const SceneReadToken& token, const SearchRequest& request) const {
  QueryMetadata metadata = metadataFor(token);
  std::string message;
  const QueryStatus validation = validateToken(token, &message);
  if (validation != QueryStatus::kOk) {
    return tokenFailure<SearchObjectMatches>(
        validation, std::move(message), std::move(metadata));
  }
  if (request.query.empty() || request.limit == 0) {
    return tokenFailure<SearchObjectMatches>(
        QueryStatus::kInvalidArgument,
        "search query must be non-empty and limit must be positive",
        std::move(metadata));
  }

  std::optional<std::set<SceneObjectId>> room_filter;
  if (request.room_id) {
    room_filter.emplace();
    const std::string requested_room = normalizedKey(*request.room_id);
    for (const CanonicalRoom& room : canonicalRooms(token)) {
      if (normalizedKey(room.room_id) == requested_room ||
          normalizedKey(room.name) == requested_room) {
        room_filter->insert(room.object_ids.begin(), room.object_ids.end());
        if (room.room_object_id) {
          room_filter->insert(*room.room_object_id);
        }
      }
    }
  }

  std::map<SceneObjectId, QueryObjectView> eligible;
  for (const auto& entry : token.snapshot_.objects()) {
    if (!entry.second || token.snapshot_.isTombstoned(entry.first) ||
        !objectVisible(*entry.second, request.include_inactive,
                       request.include_unpublishable) ||
        (room_filter && room_filter->count(entry.first) == 0)) {
      continue;
    }
    eligible.emplace(entry.first,
                     objectView(token, entry.first, entry.first, *entry.second));
  }

  std::map<SceneObjectId, SearchObjectMatch> matches;
  std::set<SceneObjectId> lexical_delta;
  std::map<SceneObjectId, std::string> stale_hit_hashes;

  if (!token.index_) {
    for (const auto& entry : eligible) {
      lexical_delta.insert(entry.first);
    }
  } else {
    for (const auto& entry : eligible) {
      if (!entry.second.indexed_document_hash ||
          *entry.second.indexed_document_hash !=
              entry.second.semantic_document_hash) {
        lexical_delta.insert(entry.first);
      }
    }

    try {
      const std::size_t candidate_limit =
          request.limit > std::numeric_limits<std::size_t>::max() / 4
              ? request.limit
              : request.limit * 4;
      for (const IndexedSearchHit& hit :
           token.index_->search(request.query, candidate_limit)) {
        if (!std::isfinite(hit.score)) {
          continue;
        }
        const auto canonical = token.snapshot_.resolveCanonicalId(hit.object_id);
        if (!canonical) {
          continue;
        }
        const auto object = eligible.find(*canonical);
        if (object == eligible.end()) {
          continue;
        }
        const bool valid_row = object->second.indexed_document_hash &&
                               *object->second.indexed_document_hash ==
                                   object->second.semantic_document_hash &&
                               hit.document_hash ==
                                   object->second.semantic_document_hash;
        if (!valid_row) {
          lexical_delta.insert(*canonical);
          stale_hit_hashes[*canonical] = hit.document_hash;
          continue;
        }

        SearchObjectMatch match;
        match.object = object->second;
        match.score = hit.score;
        match.source = SearchMatchSource::kVector;
        match.vector_document_hash = hit.document_hash;
        auto inserted = matches.emplace(*canonical, match);
        if (!inserted.second && hit.score > inserted.first->second.score) {
          inserted.first->second = std::move(match);
        }
      }
    } catch (const std::exception& error) {
      metadata.index_available = false;
      message = std::string("pinned index unavailable; lexical fallback: ") +
                error.what();
      for (const auto& entry : eligible) {
        lexical_delta.insert(entry.first);
      }
    }
  }

  if (!lexical_delta.empty()) {
    metadata.lexical_delta_used = true;
  }
  for (SceneObjectId object_id : lexical_delta) {
    const auto object = eligible.find(object_id);
    if (object == eligible.end() || matches.count(object_id) != 0) {
      continue;
    }
    const SceneObjectPtr scene_object = token.snapshot_.findObject(object_id);
    if (!scene_object) {
      continue;
    }
    const float score = lexicalScore(request.query,
                                     semanticDocument(*scene_object));
    if (score <= 0.0f) {
      continue;
    }
    SearchObjectMatch match;
    match.object = object->second;
    match.score = score;
    match.source = SearchMatchSource::kPinnedLexicalDelta;
    const auto stale_hash = stale_hit_hashes.find(object_id);
    if (stale_hash != stale_hit_hashes.end()) {
      match.vector_document_hash = stale_hash->second;
    } else if (object->second.indexed_document_hash) {
      match.vector_document_hash = *object->second.indexed_document_hash;
    }
    matches.emplace(object_id, std::move(match));
  }

  QueryResult<SearchObjectMatches> result;
  result.metadata = std::move(metadata);
  result.message = std::move(message);
  result.value.reserve(matches.size());
  for (auto& entry : matches) {
    result.value.push_back(std::move(entry.second));
  }
  std::sort(result.value.begin(), result.value.end(),
            [](const SearchObjectMatch& lhs, const SearchObjectMatch& rhs) {
              if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
              }
              if (lhs.source != rhs.source) {
                return lhs.source == SearchMatchSource::kVector;
              }
              return lhs.object.object_id < rhs.object.object_id;
            });
  if (result.value.size() > request.limit) {
    result.value.resize(request.limit);
  }
  for (const SearchObjectMatch& match : result.value) {
    absorbFreshness(&result.metadata, match.object);
  }
  finishMetadata(&result.metadata, !result.value.empty());
  return result;
}

QueryResult<QueryObjectView> LocalSceneQueryHandlers::getObject(
    SceneObjectId object_id) const {
  return gateway_->getObject(token_, object_id);
}

QueryResult<QueryObjectList> LocalSceneQueryHandlers::getObjectsNear(
    const ObjectsNearRequest& request) const {
  return gateway_->getObjectsNear(token_, request);
}

QueryResult<std::vector<CanonicalRoom>> LocalSceneQueryHandlers::rooms() const {
  return gateway_->rooms(token_);
}

QueryResult<std::vector<CanonicalRelation>>
LocalSceneQueryHandlers::relations(const RelationRequest& request) const {
  return gateway_->relations(token_, request);
}

QueryResult<SnapshotInspection> LocalSceneQueryHandlers::inspectSnapshot(
    SceneObjectId object_id, std::optional<int> image_index) const {
  return gateway_->inspectSnapshot(token_, object_id, image_index);
}

QueryResult<SearchObjectMatches> LocalSceneQueryHandlers::searchObjects(
    const SearchRequest& request) const {
  return gateway_->searchObjects(token_, request);
}

}  // namespace roomie
