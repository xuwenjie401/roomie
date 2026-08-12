#include "roomie/artifacts/semantic_index.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace roomie {
namespace {

using Json = nlohmann::json;

constexpr float kVectorNormEpsilon = 1.0e-12f;

std::uint32_t rotateRight(std::uint32_t value, std::uint32_t bits) {
  return (value >> bits) | (value << (32U - bits));
}

// Local SHA-256 keeps document ids deterministic across processes and standard
// library implementations. The same compact implementation is used by the
// content-addressed snapshot bank, but remains private to each subsystem.
class Sha256 {
 public:
  Sha256()
      : state_{0x6a09e667U,
               0xbb67ae85U,
               0x3c6ef372U,
               0xa54ff53aU,
               0x510e527fU,
               0x9b05688cU,
               0x1f83d9abU,
               0x5be0cd19U} {}

  void update(const void* bytes, std::size_t size) {
    const auto* data = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t index = 0; index < size; ++index) {
      buffer_[buffer_size_++] = data[index];
      if (buffer_size_ == buffer_.size()) {
        transform(buffer_.data());
        bit_count_ += 512U;
        buffer_size_ = 0;
      }
    }
  }

  void update(const std::string& value) { update(value.data(), value.size()); }

  std::string finishHex() {
    const std::uint64_t final_bit_count =
        bit_count_ + static_cast<std::uint64_t>(buffer_size_) * 8U;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56U) {
      while (buffer_size_ < 64U) {
        buffer_[buffer_size_++] = 0U;
      }
      transform(buffer_.data());
      buffer_size_ = 0;
    }
    while (buffer_size_ < 56U) {
      buffer_[buffer_size_++] = 0U;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
      buffer_[buffer_size_++] =
          static_cast<std::uint8_t>((final_bit_count >> shift) & 0xffU);
    }
    transform(buffer_.data());

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint32_t word : state_) {
      output << std::setw(8) << word;
    }
    return output.str();
  }

 private:
  void transform(const std::uint8_t* block) {
    static constexpr std::array<std::uint32_t, 64> kConstants = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t index = 0; index < 16U; ++index) {
      const std::size_t offset = index * 4U;
      schedule[index] =
          (static_cast<std::uint32_t>(block[offset]) << 24U) |
          (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
          (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
          static_cast<std::uint32_t>(block[offset + 3U]);
    }
    for (std::size_t index = 16U; index < schedule.size(); ++index) {
      const std::uint32_t s0 = rotateRight(schedule[index - 15U], 7U) ^
                               rotateRight(schedule[index - 15U], 18U) ^
                               (schedule[index - 15U] >> 3U);
      const std::uint32_t s1 = rotateRight(schedule[index - 2U], 17U) ^
                               rotateRight(schedule[index - 2U], 19U) ^
                               (schedule[index - 2U] >> 10U);
      schedule[index] = schedule[index - 16U] + s0 +
                        schedule[index - 7U] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < schedule.size(); ++index) {
      const std::uint32_t sigma1 =
          rotateRight(e, 6U) ^ rotateRight(e, 11U) ^ rotateRight(e, 25U);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 =
          h + sigma1 + choose + kConstants[index] + schedule[index];
      const std::uint32_t sigma0 =
          rotateRight(a, 2U) ^ rotateRight(a, 13U) ^ rotateRight(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sigma0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_;
  std::array<std::uint8_t, 64> buffer_{};
  std::uint64_t bit_count_ = 0;
  std::size_t buffer_size_ = 0;
};

std::string sha256(const std::string& value) {
  Sha256 digest;
  digest.update(value);
  return digest.finishHex();
}

std::string normalizeText(const std::string& value) {
  std::string result;
  result.reserve(value.size());
  bool pending_space = false;
  for (const unsigned char byte : value) {
    if (std::isspace(byte) != 0) {
      pending_space = !result.empty();
      continue;
    }
    if (pending_space) {
      result.push_back(' ');
      pending_space = false;
    }
    result.push_back(static_cast<char>(byte < 0x80U ? std::tolower(byte) : byte));
  }
  return result;
}

void appendCanonicalField(std::ostringstream* output,
                          const std::string& name,
                          const std::string& value) {
  *output << name.size() << ':' << name << '=' << value.size() << ':' << value
          << '\n';
}

std::string effectiveSemanticLabel(const SceneObject& object) {
  if (object.annotation && object.annotation->label_override &&
      !object.annotation->label_override->empty()) {
    return *object.annotation->label_override;
  }
  return object.semantic ? object.semantic->label : std::string{};
}

std::string effectiveSemanticDescription(const SceneObject& object,
                                         SemanticDocumentView view) {
  if (object.annotation && object.annotation->description_override &&
      !object.annotation->description_override->empty()) {
    return *object.annotation->description_override;
  }
  if (!object.artifact) {
    return {};
  }
  if (view == SemanticDocumentView::kPendingDescriptionIfPresent &&
      !object.artifact->pending_description.empty()) {
    return object.artifact->pending_description;
  }
  return object.artifact->description;
}

std::string normalizedMetadataKey(const std::string& value) {
  std::string result;
  result.reserve(value.size());
  for (const unsigned char byte : value) {
    if (byte < 0x80U && std::isalnum(byte) != 0) {
      result.push_back(static_cast<char>(std::tolower(byte)));
    } else if (byte == '_' || byte == '-' ||
               (byte < 0x80U && std::isspace(byte) != 0)) {
      if (!result.empty() && result.back() != '_') {
        result.push_back('_');
      }
    }
  }
  while (!result.empty() && result.back() == '_') {
    result.pop_back();
  }
  return result;
}

bool isVolatileMetadataAttribute(const std::string& name) {
  static const std::set<std::string> kMetadataNames = {
      "room",       "room_id",       "active",      "position",
      "position_world", "center",    "center_world", "x",
      "y",          "z"};
  return kMetadataNames.count(normalizedMetadataKey(name)) != 0;
}

void appendStringArray(const Json& object,
                       const std::string& name,
                       std::map<std::string, std::vector<std::string>>* fields) {
  const auto it = object.find(name);
  if (it == object.end() || !it->is_array()) {
    return;
  }
  for (const Json& entry : *it) {
    if (entry.is_string()) {
      (*fields)[name].push_back(entry.get<std::string>());
    }
  }
}

std::vector<float> normalizedVector(const std::vector<float>& input,
                                    bool* valid) {
  double squared_norm = 0.0;
  for (const float value : input) {
    if (!std::isfinite(value)) {
      *valid = false;
      return {};
    }
    squared_norm += static_cast<double>(value) * value;
  }
  if (!std::isfinite(squared_norm) ||
      squared_norm <= static_cast<double>(kVectorNormEpsilon)) {
    *valid = false;
    return {};
  }
  const float inverse_norm =
      static_cast<float>(1.0 / std::sqrt(squared_norm));
  std::vector<float> output;
  output.reserve(input.size());
  for (const float value : input) {
    output.push_back(value * inverse_norm);
  }
  *valid = true;
  return output;
}

std::set<std::string> lexicalTerms(const std::string& text) {
  std::set<std::string> result;
  std::string term;
  for (const unsigned char byte : text) {
    const bool ascii_word =
        byte < 0x80U && (std::isalnum(byte) != 0 || byte == '_');
    const bool utf8_byte = byte >= 0x80U;
    if (ascii_word || utf8_byte) {
      term.push_back(
          static_cast<char>(byte < 0x80U ? std::tolower(byte) : byte));
    } else if (!term.empty()) {
      result.insert(std::move(term));
      term.clear();
    }
  }
  if (!term.empty()) {
    result.insert(std::move(term));
  }
  return result;
}

float lexicalScore(const std::set<std::string>& query,
                   const std::string& document) {
  if (query.empty()) {
    return 0.0f;
  }
  const std::set<std::string> terms = lexicalTerms(document);
  std::size_t matched = 0;
  for (const std::string& term : query) {
    matched += terms.count(term);
  }
  return static_cast<float>(matched) / static_cast<float>(query.size());
}

bool sameMetadata(const SemanticMetadata& lhs, const SemanticMetadata& rhs) {
  return lhs.room_id == rhs.room_id &&
         lhs.position_world == rhs.position_world &&
         lhs.has_position == rhs.has_position && lhs.active == rhs.active &&
         lhs.filters == rhs.filters;
}

bool passesFilter(const SemanticMetadata& metadata,
                  const SemanticSearchFilter& filter) {
  if (filter.room_id && metadata.room_id != *filter.room_id) {
    return false;
  }
  if (filter.active && metadata.active != *filter.active) {
    return false;
  }
  for (const auto& [key, expected] : filter.exact_metadata) {
    const auto it = metadata.filters.find(key);
    if (it == metadata.filters.end() || it->second != expected) {
      return false;
    }
  }
  return true;
}

std::int64_t saturatingAdd(std::int64_t lhs, std::int64_t rhs) {
  if (rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() - rhs) {
    return std::numeric_limits<std::int64_t>::max();
  }
  if (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::min() - rhs) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return lhs + rhs;
}

}  // namespace

std::string stableSha256Hex(std::string_view value) {
  return sha256(std::string(value));
}

SemanticDocument makeSemanticDocument(const SemanticDocumentInput& input) {
  SemanticDocument result;
  result.object_id = input.object_id;
  result.metadata = input.metadata;
  result.semantic_revision = input.semantic_revision;
  result.created_scene_revision = input.created_scene_revision;

  std::ostringstream canonical;
  canonical << "semantic-document.v1\n";
  // Keep hashes for every pre-name object byte-identical. The optional field
  // is emitted only after a human assigns a non-empty instance name.
  if (!normalizeText(input.name).empty()) {
    appendCanonicalField(&canonical, "name", normalizeText(input.name));
  }
  appendCanonicalField(&canonical, "label", normalizeText(input.label));
  appendCanonicalField(&canonical,
                       "canonical_name",
                       normalizeText(input.retrieval.canonical_name));
  appendCanonicalField(&canonical,
                       "short_description",
                       normalizeText(input.retrieval.short_description));
  appendCanonicalField(&canonical,
                       "retrieval_text",
                       normalizeText(input.retrieval.retrieval_text));

  std::map<std::string, std::set<std::string>> normalized_attributes;
  for (const auto& [raw_name, raw_values] : input.retrieval.attributes) {
    const std::string name = normalizeText(raw_name);
    if (name.empty()) {
      continue;
    }
    for (const std::string& raw_value : raw_values) {
      const std::string value = normalizeText(raw_value);
      if (!value.empty()) {
        normalized_attributes[name].insert(value);
      }
    }
  }
  for (const auto& [name, values] : normalized_attributes) {
    for (const std::string& value : values) {
      appendCanonicalField(&canonical, "attribute." + name, value);
    }
  }

  std::set<std::string> tags;
  for (const std::string& raw_tag : input.human_tags) {
    const std::string tag = normalizeText(raw_tag);
    if (!tag.empty()) {
      tags.insert(tag);
    }
  }
  for (const std::string& tag : tags) {
    appendCanonicalField(&canonical, "human_tag", tag);
  }

  result.text = canonical.str();
  result.document_hash = sha256(result.text);
  return result;
}

SemanticDocument makeSemanticDocumentForObject(
    const SceneObject& object,
    SceneObjectId object_id,
    SceneRevision scene_revision,
    SemanticDocumentView view) {
  if (object_id < 0 || !object.identity || !object.semantic ||
      !object.artifact || !object.annotation) {
    throw std::invalid_argument(
        "scene object lacks canonical semantic-document components");
  }

  SemanticDocumentInput input;
  input.object_id = object_id;
  input.name = object.annotation->name;
  input.label = effectiveSemanticLabel(object);
  input.retrieval.canonical_name = input.label;
  const std::string description = effectiveSemanticDescription(object, view);
  input.retrieval.short_description = description;
  input.retrieval.retrieval_text = description;

  const std::string structured_payload =
      view == SemanticDocumentView::kPendingDescriptionIfPresent &&
              !object.artifact->pending_description_normalized_json.empty()
          ? object.artifact->pending_description_normalized_json
          : object.artifact->description_normalized_json;
  const Json structured = Json::parse(
      structured_payload.empty() ? description : structured_payload,
      nullptr, false);
  if (structured.is_object()) {
    if (structured.contains("canonical_name") &&
        structured["canonical_name"].is_string()) {
      input.retrieval.canonical_name =
          structured["canonical_name"].get<std::string>();
    }
    if (structured.contains("short_description") &&
        structured["short_description"].is_string()) {
      input.retrieval.short_description =
          structured["short_description"].get<std::string>();
    }
    if (structured.contains("retrieval_text") &&
        structured["retrieval_text"].is_string()) {
      input.retrieval.retrieval_text =
          structured["retrieval_text"].get<std::string>();
    }
    const auto attributes = structured.find("visual_attributes");
    if (attributes != structured.end() && attributes->is_object()) {
      static const std::vector<std::string> kRetrievalAttributeNames = {
          "colors",          "materials",     "shape",
          "visible_parts",   "state_or_pose", "distinctive_marks",
          "visible_text"};
      for (const std::string& name : kRetrievalAttributeNames) {
        appendStringArray(*attributes, name, &input.retrieval.attributes);
      }
    }
  }

  for (const auto& [name, value] : object.annotation->attributes) {
    if (isVolatileMetadataAttribute(name)) {
      const std::string metadata_name = normalizedMetadataKey(name);
      input.metadata.filters[metadata_name] = value;
      if (metadata_name == "room" || metadata_name == "room_id") {
        input.metadata.room_id = value;
      }
      continue;
    }
    input.retrieval.attributes["human." + name].push_back(value);
    input.human_tags.push_back(name + "=" + value);
  }
  input.semantic_revision = object.semantic->revision;
  input.created_scene_revision = scene_revision;
  if (object.lifecycle) {
    input.metadata.active = object.lifecycle->active;
  }
  if (object.geometry) {
    input.metadata.position_world = {object.geometry->center_world.x(),
                                     object.geometry->center_world.y(),
                                     object.geometry->center_world.z()};
    input.metadata.has_position = true;
  }
  return makeSemanticDocument(input);
}

bool operator==(const EmbeddingNamespace& lhs,
                const EmbeddingNamespace& rhs) {
  return lhs.model_id == rhs.model_id && lhs.dimension == rhs.dimension;
}

bool operator!=(const EmbeddingNamespace& lhs,
                const EmbeddingNamespace& rhs) {
  return !(lhs == rhs);
}

bool operator<(const EmbeddingNamespace& lhs,
               const EmbeddingNamespace& rhs) {
  if (lhs.model_id != rhs.model_id) {
    return lhs.model_id < rhs.model_id;
  }
  return lhs.dimension < rhs.dimension;
}

EmbeddingWorker::EmbeddingWorker(std::shared_ptr<EmbeddingEncoder> encoder,
                                 EmbeddingRecordSink sink,
                                 EmbeddingWorkerConfig config)
    : WorkerThread("embedding_worker"),
      encoder_(std::move(encoder)),
      sink_(std::move(sink)),
      config_(std::move(config)),
      pending_(
          config_.pending_capacity,
          ChannelPolicy::kLatestByKey,
          [](const EmbeddingJob& lhs, const EmbeddingJob& rhs) {
            return lhs.object_id == rhs.object_id;
          }) {
  if (!encoder_) {
    throw std::invalid_argument("EmbeddingWorker requires an encoder");
  }
  if (!sink_) {
    throw std::invalid_argument("EmbeddingWorker requires a record sink");
  }
  if (encoder_->modelId().empty() || encoder_->dimension() == 0) {
    throw std::invalid_argument("EmbeddingWorker encoder namespace is invalid");
  }
  if (config_.maximum_batch_size == 0) {
    config_.maximum_batch_size = 1;
  }
  if (config_.batch_window < std::chrono::milliseconds::zero()) {
    config_.batch_window = std::chrono::milliseconds::zero();
  }
}

EmbeddingWorker::~EmbeddingWorker() { stop(); }

PushResult<EmbeddingJob> EmbeddingWorker::submit(EmbeddingJob job) {
  if (job.object_id < 0 || job.document_hash.empty() ||
      job.name_space.model_id != encoder_->modelId() ||
      job.name_space.dimension != encoder_->dimension()) {
    PushResult<EmbeddingJob> rejected;
    rejected.outcome = PushOutcome::kRejected;
    rejected.unconsumed_item.emplace(std::move(job));
    return rejected;
  }

  PushResult<EmbeddingJob> result;
  {
    // Keep accounting ordered before a consumer can finish the newly visible
    // item. Without this fence, a very fast fake/local encoder could decrement
    // outstanding_ before the producer increments it.
    std::lock_guard<std::mutex> lock(state_mutex_);
    result = pending_.push(std::move(job));
    if (result.accepted()) {
      ++stats_.submitted;
      if (result.outcome == PushOutcome::kAccepted) {
        ++outstanding_;
      } else {
        ++stats_.replaced;
      }
    }
  }
  state_cv_.notify_all();
  return result;
}

bool EmbeddingWorker::waitUntilWarmed(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(state_mutex_);
  state_cv_.wait_for(lock, timeout, [this]() {
    return stats_.warmed || stats_.warmup_failed;
  });
  return stats_.warmed;
}

bool EmbeddingWorker::waitUntilIdle(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(state_mutex_);
  return state_cv_.wait_for(
      lock, timeout, [this]() { return outstanding_ == 0; });
}

EmbeddingWorkerStats EmbeddingWorker::stats() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  EmbeddingWorkerStats result = stats_;
  result.channel = pending_.stats();
  return result;
}

void EmbeddingWorker::finishConsumed(std::size_t count) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    outstanding_ = count >= outstanding_ ? 0 : outstanding_ - count;
  }
  state_cv_.notify_all();
}

void EmbeddingWorker::run() {
  try {
    encoder_->prewarm();
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      stats_.warmed = true;
    }
  } catch (const std::exception& error) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    stats_.warmup_failed = true;
    stats_.last_error = error.what();
  } catch (...) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    stats_.warmup_failed = true;
    stats_.last_error = "encoder prewarm failed with unknown exception";
  }
  state_cv_.notify_all();

  while (!stopRequested() || !pending_.empty()) {
    EmbeddingJob first;
    if (!pending_.waitPop(&first)) {
      if (stopRequested() || pending_.stopped()) {
        break;
      }
      continue;
    }

    std::map<SceneObjectId, EmbeddingJob> latest;
    latest.emplace(first.object_id, std::move(first));
    std::size_t consumed = 1;
    const auto deadline =
        std::chrono::steady_clock::now() + config_.batch_window;
    while (consumed < config_.maximum_batch_size) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        break;
      }
      EmbeddingJob next;
      if (!pending_.waitPopFor(&next, deadline - now)) {
        break;
      }
      latest[next.object_id] = std::move(next);
      ++consumed;
    }

    std::vector<EmbeddingJob> jobs;
    std::vector<std::string> documents;
    jobs.reserve(latest.size());
    documents.reserve(latest.size());
    for (auto& [object_id, job] : latest) {
      (void)object_id;
      documents.push_back(job.document);
      jobs.push_back(std::move(job));
    }

    bool warmup_failed = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      warmup_failed = stats_.warmup_failed;
      stats_.superseded_before_encode += consumed - jobs.size();
    }
    if (warmup_failed) {
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        stats_.failed_documents += jobs.size();
      }
      finishConsumed(consumed);
      continue;
    }

    try {
      std::vector<std::vector<float>> vectors =
          encoder_->encodeBatch(documents);
      if (vectors.size() != jobs.size()) {
        throw std::runtime_error(
            "encoder returned a different number of vectors than documents");
      }

      std::uint64_t encoded = 0;
      std::uint64_t failed = 0;
      for (std::size_t index = 0; index < jobs.size(); ++index) {
        bool valid = vectors[index].size() == encoder_->dimension();
        std::vector<float> vector;
        if (valid) {
          vector = normalizedVector(vectors[index], &valid);
        }
        if (!valid) {
          ++failed;
          continue;
        }
        EmbeddingRecord record;
        record.object_id = jobs[index].object_id;
        record.document_hash = jobs[index].document_hash;
        record.model_id = jobs[index].name_space.model_id;
        record.vector = std::move(vector);
        record.created_scene_revision = jobs[index].created_scene_revision;
        try {
          sink_(std::move(record));
          ++encoded;
        } catch (...) {
          ++failed;
        }
      }
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        ++stats_.batches;
        stats_.encoded_documents += encoded;
        stats_.failed_documents += failed;
        if (failed != 0) {
          stats_.last_error = "one or more embedding records were rejected";
        }
      }
    } catch (const std::exception& error) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      ++stats_.batches;
      stats_.failed_documents += jobs.size();
      stats_.last_error = error.what();
    } catch (...) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      ++stats_.batches;
      stats_.failed_documents += jobs.size();
      stats_.last_error = "encoder batch failed with unknown exception";
    }
    finishConsumed(consumed);
  }
}

void EmbeddingWorker::onStopRequested() { pending_.stop(); }

namespace {

struct SemanticGenerationData {
  SemanticGenerationInfo info;
  std::vector<SceneObjectId> object_ids;
  std::vector<std::string> document_hashes;
  std::vector<SceneRevision> created_scene_revisions;
  std::vector<float> matrix;
};

struct GenerationBuild {
  EmbeddingNamespace name_space;
  std::uint64_t source_mutation = 0;
  bool activate = false;
  bool complete = false;
  std::map<SceneObjectId, SemanticDocument> documents;
  std::map<SceneObjectId, std::shared_ptr<const EmbeddingRecord>> records;
  std::shared_ptr<SemanticGenerationData> generation;
};

void assembleGeneration(GenerationBuild* build) {
  build->complete = true;
  build->generation = std::make_shared<SemanticGenerationData>();
  build->generation->info.model_id = build->name_space.model_id;
  build->generation->info.dimension = build->name_space.dimension;
  for (const auto& [object_id, document] : build->documents) {
    const auto record_it = build->records.find(object_id);
    if (record_it == build->records.end() || !record_it->second ||
        record_it->second->document_hash != document.document_hash ||
        record_it->second->vector.size() != build->name_space.dimension) {
      build->complete = false;
      continue;
    }
    const EmbeddingRecord& record = *record_it->second;
    build->generation->object_ids.push_back(object_id);
    build->generation->document_hashes.push_back(document.document_hash);
    build->generation->created_scene_revisions.push_back(
        record.created_scene_revision);
    build->generation->matrix.insert(build->generation->matrix.end(),
                                     record.vector.begin(),
                                     record.vector.end());
    build->generation->info.created_through_scene_revision =
        std::max(build->generation->info.created_through_scene_revision,
                 record.created_scene_revision);
  }
  build->generation->info.rows = build->generation->object_ids.size();
  build->documents.clear();
  build->records.clear();
}

}  // namespace

struct SemanticIndexReadView {
  std::shared_ptr<const SemanticGenerationData> generation;
  std::map<SceneObjectId, SemanticDocument> documents;
  std::map<SceneObjectId, SemanticDocument> lexical_delta;
  std::map<std::string, std::vector<SceneObjectId>> lexical_delta_postings;
  std::uint64_t lexical_delta_revision = 0;
  SceneRevision scene_revision = 0;
};

struct VersionedSemanticIndex::Impl {
  explicit Impl(SemanticIndexConfig input_config)
      : config(std::move(input_config)),
        active_namespace(config.initial_namespace) {
    if (active_namespace.model_id.empty() || active_namespace.dimension == 0) {
      throw std::invalid_argument("semantic index namespace is invalid");
    }
    if (config.default_read_ttl_ms <= 0) {
      throw std::invalid_argument("semantic read TTL must be positive");
    }
    if (config.record_history_capacity == 0) {
      throw std::invalid_argument(
          "semantic record history capacity must be positive");
    }

    auto initial_generation = std::make_shared<SemanticGenerationData>();
    initial_generation->info.model_id = active_namespace.model_id;
    initial_generation->info.dimension = active_namespace.dimension;
    active_generation = std::move(initial_generation);
    publishViewLocked();
    builder = std::thread([this]() { builderLoop(); });
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    build_cv.notify_all();
    if (builder.joinable()) {
      builder.join();
    }
  }

  std::optional<SceneObjectId> resolveCanonicalLocked(
      SceneObjectId object_id) const {
    std::set<SceneObjectId> visited;
    SceneObjectId current = object_id;
    while (true) {
      if (!visited.insert(current).second) {
        return std::nullopt;
      }
      const auto it = aliases.find(current);
      if (it == aliases.end()) {
        return current;
      }
      current = it->second;
    }
  }

  bool hasRecordLocked(const EmbeddingNamespace& name_space,
                       const SemanticDocument& document) const {
    const auto namespace_it = records.find(name_space);
    if (namespace_it == records.end()) {
      return false;
    }
    const auto record_it = namespace_it->second.find(document.object_id);
    return record_it != namespace_it->second.end() &&
           record_it->second &&
           record_it->second->document_hash == document.document_hash &&
           record_it->second->vector.size() == name_space.dimension;
  }

  bool namespaceCompleteLocked(const EmbeddingNamespace& name_space) const {
    for (const auto& [object_id, document] : documents) {
      (void)object_id;
      if (!hasRecordLocked(name_space, document)) {
        return false;
      }
    }
    return true;
  }

  EmbeddingJob makeJobLocked(const SemanticDocument& document,
                             const EmbeddingNamespace& name_space) const {
    EmbeddingJob job;
    job.object_id = document.object_id;
    job.document = document.text;
    job.document_hash = document.document_hash;
    job.name_space = name_space;
    job.semantic_revision = document.semantic_revision;
    job.created_scene_revision = document.created_scene_revision;
    return job;
  }

  void requestBuildLocked(const EmbeddingNamespace& name_space,
                          bool activate) {
    auto [it, inserted] = build_requests.emplace(name_space, activate);
    if (!inserted) {
      it->second = it->second || activate;
    }
    build_cv.notify_all();
  }

  void publishViewLocked() {
    auto view = std::make_shared<SemanticIndexReadView>();
    view->generation = active_generation;
    view->documents = documents;
    view->lexical_delta_revision = ++lexical_delta_revision;
    for (const auto& [object_id, document] : documents) {
      bool represented = false;
      if (active_generation) {
        const auto row_it = std::lower_bound(active_generation->object_ids.begin(),
                                             active_generation->object_ids.end(),
                                             object_id);
        if (row_it != active_generation->object_ids.end() &&
            *row_it == object_id) {
          const std::size_t row = static_cast<std::size_t>(
              std::distance(active_generation->object_ids.begin(), row_it));
          represented =
              active_generation->document_hashes[row] == document.document_hash;
        }
      }
      if (!represented) {
        view->lexical_delta.emplace(object_id, document);
        for (const std::string& term : lexicalTerms(document.text)) {
          view->lexical_delta_postings[term].push_back(object_id);
        }
      }
      view->scene_revision =
          std::max(view->scene_revision, document.created_scene_revision);
    }
    std::atomic_store(
        &published_view,
        std::shared_ptr<const SemanticIndexReadView>(std::move(view)));
  }

  GenerationBuild captureBuildLocked(const EmbeddingNamespace& name_space,
                                     bool activate) const {
    GenerationBuild build;
    build.name_space = name_space;
    build.source_mutation = mutation_serial;
    build.activate = activate;
    build.documents = documents;
    const auto namespace_it = records.find(name_space);
    if (namespace_it != records.end()) {
      build.records = namespace_it->second;
    }
    return build;
  }

  void builderLoop() {
    while (true) {
      GenerationBuild build;
      {
        std::unique_lock<std::mutex> lock(mutex);
        build_cv.wait(lock,
                      [this]() { return stopping || !build_requests.empty(); });
        if (stopping && build_requests.empty()) {
          return;
        }
        const auto request = build_requests.begin();
        const EmbeddingNamespace name_space = request->first;
        const bool activate = request->second;
        build_requests.erase(request);
        building = true;
        build = captureBuildLocked(name_space, activate);
      }

      // Only a document snapshot plus shared immutable record handles are
      // captured under the writer lock. Matrix allocation/copy/validation
      // work happens here on the builder thread.
      assembleGeneration(&build);

      if (config.before_generation_publish) {
        try {
          config.before_generation_publish();
        } catch (...) {
          // A diagnostic hook cannot compromise the index builder.
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex);
        const bool source_changed = build.source_mutation != mutation_serial;
        const bool still_active = build.name_space == active_namespace;
        const bool still_target =
            upgrade_target && build.name_space == *upgrade_target;
        if (source_changed) {
          if (still_active || still_target) {
            requestBuildLocked(build.name_space,
                               build.activate && still_target);
          }
        } else if (build.activate) {
          if (still_target && build.complete) {
            build.generation->info.generation = next_generation++;
            active_namespace = build.name_space;
            active_generation = std::move(build.generation);
            upgrade_target.reset();
            publishViewLocked();
          }
        } else if (still_active) {
          build.generation->info.generation = next_generation++;
          active_generation = std::move(build.generation);
          publishViewLocked();
        }
        building = false;
      }
      idle_cv.notify_all();
    }
  }

  SemanticIndexConfig config;
  mutable std::mutex mutex;
  mutable std::condition_variable idle_cv;
  std::condition_variable build_cv;
  std::thread builder;
  bool stopping = false;
  bool building = false;

  EmbeddingNamespace active_namespace;
  std::optional<EmbeddingNamespace> upgrade_target;
  std::map<SceneObjectId, SemanticDocument> documents;
  std::map<SceneObjectId, SceneObjectId> aliases;
  std::set<SceneObjectId> tombstones;
  std::map<EmbeddingNamespace,
           std::map<SceneObjectId, std::shared_ptr<const EmbeddingRecord>>>
      records;
  std::vector<EmbeddingRecord> record_history;

  std::map<EmbeddingNamespace, bool> build_requests;
  std::uint64_t mutation_serial = 0;
  SemanticIndexGeneration next_generation = 1;
  std::uint64_t lexical_delta_revision = 0;
  std::shared_ptr<const SemanticGenerationData> active_generation;
  std::shared_ptr<const SemanticIndexReadView> published_view;
};

VersionedSemanticIndex::VersionedSemanticIndex(SemanticIndexConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

VersionedSemanticIndex::~VersionedSemanticIndex() = default;

SemanticUpsertResult VersionedSemanticIndex::upsertDocument(
    SemanticDocument document) {
  if (document.object_id < 0 || document.document_hash.empty() ||
      document.text.empty()) {
    throw std::invalid_argument("semantic document is incomplete");
  }

  std::lock_guard<std::mutex> lock(impl_->mutex);
  const std::optional<SceneObjectId> canonical =
      impl_->resolveCanonicalLocked(document.object_id);
  if (!canonical || *canonical != document.object_id ||
      impl_->tombstones.count(document.object_id) != 0) {
    throw std::invalid_argument("cannot upsert a retired semantic object id");
  }

  SemanticUpsertResult result;
  const auto existing = impl_->documents.find(document.object_id);
  result.document_changed =
      existing == impl_->documents.end() ||
      existing->second.document_hash != document.document_hash ||
      existing->second.text != document.text;
  result.metadata_changed =
      existing == impl_->documents.end() ||
      !sameMetadata(existing->second.metadata, document.metadata);

  auto append_missing_jobs = [this, &result](
                                 const SemanticDocument& current) {
    std::set<EmbeddingNamespace> required_namespaces;
    required_namespaces.insert(impl_->active_namespace);
    if (impl_->upgrade_target) {
      required_namespaces.insert(*impl_->upgrade_target);
    }
    for (const EmbeddingNamespace& name_space : required_namespaces) {
      if (!impl_->hasRecordLocked(name_space, current)) {
        result.jobs.push_back(impl_->makeJobLocked(current, name_space));
      }
    }
  };

  if (!result.document_changed && !result.metadata_changed &&
      existing->second.semantic_revision == document.semantic_revision &&
      existing->second.created_scene_revision ==
          document.created_scene_revision) {
    append_missing_jobs(existing->second);
    return result;
  }

  const SceneObjectId object_id = document.object_id;
  impl_->documents[object_id] = std::move(document);
  ++impl_->mutation_serial;
  const SemanticDocument& stored = impl_->documents.at(object_id);

  append_missing_jobs(stored);

  impl_->requestBuildLocked(impl_->active_namespace, false);
  if (impl_->upgrade_target &&
      impl_->namespaceCompleteLocked(*impl_->upgrade_target)) {
    impl_->requestBuildLocked(*impl_->upgrade_target, true);
  }
  impl_->publishViewLocked();
  return result;
}

bool VersionedSemanticIndex::eraseObject(SceneObjectId object_id) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const std::optional<SceneObjectId> canonical =
      impl_->resolveCanonicalLocked(object_id);
  if (!canonical) {
    return false;
  }
  const bool erased = impl_->documents.erase(*canonical) != 0;
  if (!erased) {
    return false;
  }
  impl_->tombstones.insert(*canonical);
  ++impl_->mutation_serial;
  impl_->requestBuildLocked(impl_->active_namespace, false);
  if (impl_->upgrade_target &&
      impl_->namespaceCompleteLocked(*impl_->upgrade_target)) {
    impl_->requestBuildLocked(*impl_->upgrade_target, true);
  }
  impl_->publishViewLocked();
  return true;
}

bool VersionedSemanticIndex::mergeObjects(SceneObjectId retired_object_id,
                                          SceneObjectId canonical_object_id) {
  if (retired_object_id < 0 || canonical_object_id < 0 ||
      retired_object_id == canonical_object_id) {
    return false;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const std::optional<SceneObjectId> retired =
      impl_->resolveCanonicalLocked(retired_object_id);
  const std::optional<SceneObjectId> canonical =
      impl_->resolveCanonicalLocked(canonical_object_id);
  if (!retired || !canonical || *retired == *canonical ||
      impl_->documents.count(*canonical) == 0 ||
      impl_->tombstones.count(*canonical) != 0) {
    return false;
  }

  impl_->aliases[retired_object_id] = *canonical;
  impl_->aliases[*retired] = *canonical;
  impl_->documents.erase(*retired);
  ++impl_->mutation_serial;
  impl_->requestBuildLocked(impl_->active_namespace, false);
  if (impl_->upgrade_target &&
      impl_->namespaceCompleteLocked(*impl_->upgrade_target)) {
    impl_->requestBuildLocked(*impl_->upgrade_target, true);
  }
  impl_->publishViewLocked();
  return true;
}

std::vector<EmbeddingJob> VersionedSemanticIndex::beginModelUpgrade(
    EmbeddingNamespace name_space) {
  if (name_space.model_id.empty() || name_space.dimension == 0) {
    throw std::invalid_argument("embedding upgrade namespace is invalid");
  }

  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (name_space == impl_->active_namespace) {
    return {};
  }
  impl_->upgrade_target = name_space;
  ++impl_->mutation_serial;
  std::vector<EmbeddingJob> jobs;
  for (const auto& [object_id, document] : impl_->documents) {
    (void)object_id;
    if (!impl_->hasRecordLocked(name_space, document)) {
      jobs.push_back(impl_->makeJobLocked(document, name_space));
    }
  }
  if (jobs.empty()) {
    impl_->requestBuildLocked(name_space, true);
  }
  return jobs;
}

EmbeddingAcceptStatus VersionedSemanticIndex::acceptEmbedding(
    EmbeddingRecord record) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto document_it = impl_->documents.find(record.object_id);
  if (document_it == impl_->documents.end() ||
      impl_->tombstones.count(record.object_id) != 0 ||
      impl_->aliases.count(record.object_id) != 0) {
    return EmbeddingAcceptStatus::kRejectedObjectMissing;
  }
  if (record.document_hash != document_it->second.document_hash) {
    return EmbeddingAcceptStatus::kRejectedDocumentStale;
  }

  std::optional<EmbeddingNamespace> name_space;
  bool known_model = false;
  if (record.model_id == impl_->active_namespace.model_id) {
    known_model = true;
    if (record.vector.size() == impl_->active_namespace.dimension) {
      name_space = impl_->active_namespace;
    }
  }
  if (impl_->upgrade_target &&
      record.model_id == impl_->upgrade_target->model_id) {
    known_model = true;
    if (record.vector.size() == impl_->upgrade_target->dimension) {
      name_space = impl_->upgrade_target;
    }
  }
  if (!name_space) {
    return known_model ? EmbeddingAcceptStatus::kRejectedDimension
                       : EmbeddingAcceptStatus::kRejectedNamespace;
  }
  bool valid = false;
  record.vector = normalizedVector(record.vector, &valid);
  if (!valid) {
    return EmbeddingAcceptStatus::kRejectedInvalidVector;
  }

  impl_->records[*name_space][record.object_id] =
      std::make_shared<const EmbeddingRecord>(record);
  impl_->record_history.push_back(std::move(record));
  if (impl_->record_history.size() > impl_->config.record_history_capacity) {
    const std::size_t excess =
        impl_->record_history.size() - impl_->config.record_history_capacity;
    impl_->record_history.erase(
        impl_->record_history.begin(),
        impl_->record_history.begin() + static_cast<std::ptrdiff_t>(excess));
  }
  ++impl_->mutation_serial;
  if (*name_space == impl_->active_namespace) {
    impl_->requestBuildLocked(*name_space, false);
  }
  if (impl_->upgrade_target && *name_space == *impl_->upgrade_target &&
      impl_->namespaceCompleteLocked(*impl_->upgrade_target)) {
    impl_->requestBuildLocked(*impl_->upgrade_target, true);
  }
  return EmbeddingAcceptStatus::kAccepted;
}

std::vector<EmbeddingRecord> VersionedSemanticIndex::embeddingRecords() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->record_history;
}

SemanticReadToken VersionedSemanticIndex::pinRead(
    SceneRevision scene_revision,
    std::int64_t now_unix_ms,
    std::int64_t ttl_ms) const {
  SemanticReadToken token;
  token.view_ = std::atomic_load(&impl_->published_view);
  if (!token.view_) {
    return token;
  }
  if (ttl_ms <= 0) {
    ttl_ms = impl_->config.default_read_ttl_ms;
  }
  token.expires_at_unix_ms_ = saturatingAdd(now_unix_ms, ttl_ms);
  token.scene_revision_ =
      scene_revision == 0 ? token.view_->scene_revision : scene_revision;
  token.info_ = token.view_->generation->info;
  token.info_.lexical_delta_revision = token.view_->lexical_delta_revision;
  return token;
}

SemanticSearchResponse VersionedSemanticIndex::search(
    const SemanticReadToken& token,
    const SemanticSearchRequest& request,
    std::int64_t now_unix_ms) const {
  SemanticSearchResponse response;
  response.index = token.info_;
  response.scene_revision = token.scene_revision_;
  if (token.expired(now_unix_ms)) {
    response.status = SemanticSearchStatus::kExpiredToken;
    return response;
  }

  std::vector<float> query_vector;
  if (!request.query_vector.empty()) {
    if (request.query_vector.size() != token.info_.dimension) {
      response.status = SemanticSearchStatus::kInvalidQueryDimension;
      return response;
    }
    bool valid = false;
    query_vector = normalizedVector(request.query_vector, &valid);
    if (!valid) {
      response.status = SemanticSearchStatus::kInvalidQueryVector;
      return response;
    }
  }

  const std::set<std::string> query_terms = lexicalTerms(request.query_text);
  const SemanticIndexReadView& view = *token.view_;
  response.pending_embeddings = view.lexical_delta.size();

  if (!query_vector.empty() || !query_terms.empty()) {
    const SemanticGenerationData& generation = *view.generation;
    for (std::size_t row = 0; row < generation.object_ids.size(); ++row) {
      const SceneObjectId object_id = generation.object_ids[row];
      const auto document_it = view.documents.find(object_id);
      if (document_it == view.documents.end() ||
          document_it->second.document_hash !=
              generation.document_hashes[row] ||
          !passesFilter(document_it->second.metadata, request.filter)) {
        continue;
      }

      SemanticSearchHit hit;
      hit.object_id = object_id;
      hit.document_hash = document_it->second.document_hash;
      hit.document = document_it->second.text;
      hit.metadata = document_it->second.metadata;
      hit.created_scene_revision =
          generation.created_scene_revisions[row];
      hit.freshness = SemanticFreshness::kCurrentEmbedding;
      if (!query_vector.empty()) {
        const std::size_t offset = row * generation.info.dimension;
        for (std::size_t column = 0; column < generation.info.dimension;
             ++column) {
          hit.vector_score +=
              query_vector[column] * generation.matrix[offset + column];
        }
        hit.score = hit.vector_score;
      } else {
        hit.lexical_score = lexicalScore(query_terms, hit.document);
        hit.score = hit.lexical_score;
        if (hit.lexical_score <= 0.0f) {
          continue;
        }
      }
      response.hits.push_back(std::move(hit));
    }

    std::map<SceneObjectId, std::size_t> lexical_matches;
    for (const std::string& term : query_terms) {
      const auto posting = view.lexical_delta_postings.find(term);
      if (posting == view.lexical_delta_postings.end()) {
        continue;
      }
      for (const SceneObjectId object_id : posting->second) {
        ++lexical_matches[object_id];
      }
    }
    for (const auto& [object_id, matched_terms] : lexical_matches) {
      const auto document_it = view.lexical_delta.find(object_id);
      if (document_it == view.lexical_delta.end()) {
        continue;
      }
      const SemanticDocument& document = document_it->second;
      if (!passesFilter(document.metadata, request.filter)) {
        continue;
      }
      const float score = static_cast<float>(matched_terms) /
                          static_cast<float>(query_terms.size());
      if (score <= 0.0f) {
        continue;
      }
      SemanticSearchHit hit;
      hit.object_id = object_id;
      hit.document_hash = document.document_hash;
      hit.document = document.text;
      hit.metadata = document.metadata;
      hit.created_scene_revision = document.created_scene_revision;
      hit.score = score;
      hit.lexical_score = score;
      hit.freshness = SemanticFreshness::kEmbeddingPendingLexical;
      response.hits.push_back(std::move(hit));
    }
  }

  std::sort(response.hits.begin(), response.hits.end(),
            [](const SemanticSearchHit& lhs, const SemanticSearchHit& rhs) {
              if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
              }
              return lhs.object_id < rhs.object_id;
            });
  if (response.hits.size() > request.top_k) {
    response.hits.resize(request.top_k);
  }
  return response;
}

std::optional<std::string> VersionedSemanticIndex::indexedDocumentHash(
    const SemanticReadToken& token,
    SceneObjectId object_id) const {
  if (!token.view_ || !token.view_->generation || object_id < 0) {
    return std::nullopt;
  }
  const SemanticGenerationData& generation = *token.view_->generation;
  const auto row = std::lower_bound(generation.object_ids.begin(),
                                    generation.object_ids.end(), object_id);
  if (row == generation.object_ids.end() || *row != object_id) {
    return std::nullopt;
  }
  const std::size_t index = static_cast<std::size_t>(
      std::distance(generation.object_ids.begin(), row));
  if (index >= generation.document_hashes.size()) {
    return std::nullopt;
  }
  return generation.document_hashes[index];
}

SemanticGenerationInfo VersionedSemanticIndex::activeGeneration() const {
  const std::shared_ptr<const SemanticIndexReadView> view =
      std::atomic_load(&impl_->published_view);
  if (!view) {
    return {};
  }
  SemanticGenerationInfo info = view->generation->info;
  info.lexical_delta_revision = view->lexical_delta_revision;
  return info;
}

std::size_t VersionedSemanticIndex::pendingEmbeddingCount() const {
  const std::shared_ptr<const SemanticIndexReadView> view =
      std::atomic_load(&impl_->published_view);
  return view ? view->lexical_delta.size() : 0;
}

bool VersionedSemanticIndex::waitForBuildIdle(
    std::chrono::milliseconds timeout) const {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  return impl_->idle_cv.wait_for(lock, timeout, [this]() {
    return !impl_->building && impl_->build_requests.empty();
  });
}

}  // namespace roomie
