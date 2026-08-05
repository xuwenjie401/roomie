#include "roomie/artifacts/snapshot_bank.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <system_error>
#include <unordered_set>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <png.h>

namespace roomie {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr float kScoreEpsilon = 1.0e-7f;

TimeNanoseconds wallNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

TimeNanoseconds resolvedNow(TimeNanoseconds now_ns) {
  return now_ns == 0 ? wallNowNs() : now_ns;
}

void setError(std::string* error, const std::string& value) {
  if (error != nullptr) {
    *error = value;
  }
}

float normalizedScore(float value) {
  if (!std::isfinite(value)) {
    return 0.0f;
  }
  return std::clamp(value, 0.0f, 1.0f);
}

int boundedFloorToInt(double value) {
  const double floored = std::floor(value);
  if (floored <= static_cast<double>(std::numeric_limits<int>::min())) {
    return std::numeric_limits<int>::min();
  }
  if (floored >= static_cast<double>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(floored);
}

std::uint32_t rotateRight(std::uint32_t value, std::uint32_t bits) {
  return (value >> bits) | (value << (32U - bits));
}

// Compact SHA-256 implementation used for both physical assets and logical
// evidence sets. Keeping it local avoids platform-dependent std::hash values.
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
      schedule[index] = schedule[index - 16U] + s0 + schedule[index - 7U] + s1;
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

std::string assetDescriptor(const AssetRecord& asset) {
  std::ostringstream descriptor;
  descriptor << "roomie.lossless-frame.v1\n" << asset.width << '\n'
             << asset.height << '\n' << asset.channels << '\n'
             << asset.source_encoding << '\n';
  return descriptor.str();
}

std::string hashAssetBytes(const AssetRecord& asset,
                           const std::vector<std::uint8_t>& png) {
  Sha256 digest;
  digest.update(assetDescriptor(asset));
  if (!png.empty()) {
    digest.update(png.data(), png.size());
  }
  return digest.finishHex();
}

bool validAssetId(const std::string& id) {
  return id.size() == 64U &&
         std::all_of(id.begin(), id.end(), [](char value) {
           return (value >= '0' && value <= '9') ||
                  (value >= 'a' && value <= 'f');
         });
}

bool readBytes(const std::filesystem::path& path,
               std::vector<std::uint8_t>* bytes,
               std::string* error) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    setError(error, "failed to open asset: " + path.string());
    return false;
  }
  const std::streamsize size = stream.tellg();
  if (size < 0) {
    setError(error, "failed to determine asset size: " + path.string());
    return false;
  }
  bytes->resize(static_cast<std::size_t>(size));
  stream.seekg(0, std::ios::beg);
  if (size > 0 &&
      !stream.read(reinterpret_cast<char*>(bytes->data()), size)) {
    setError(error, "failed to read asset: " + path.string());
    return false;
  }
  return true;
}

bool syncPath(const std::filesystem::path& path, bool directory) {
  const int flags = directory ? O_RDONLY | O_DIRECTORY : O_RDONLY;
  const int descriptor = ::open(path.c_str(), flags);
  if (descriptor < 0) {
    return false;
  }
  const bool success = ::fsync(descriptor) == 0;
  ::close(descriptor);
  return success;
}

bool atomicWrite(const std::filesystem::path& target,
                 const void* bytes,
                 std::size_t size,
                 std::string* error) {
  static std::atomic<std::uint64_t> sequence{1};
  try {
    if (!target.parent_path().empty()) {
      std::filesystem::create_directories(target.parent_path());
    }
    std::ostringstream suffix;
    suffix << ".tmp." << static_cast<long long>(::getpid()) << '.'
           << sequence.fetch_add(1, std::memory_order_relaxed);
    const std::filesystem::path temporary = target.string() + suffix.str();
    {
      std::ofstream stream(
          temporary, std::ios::binary | std::ios::out | std::ios::trunc);
      if (!stream) {
        setError(error, "failed to open temporary file: " + temporary.string());
        return false;
      }
      if (size > 0U) {
        stream.write(static_cast<const char*>(bytes),
                     static_cast<std::streamsize>(size));
      }
      stream.flush();
      if (!stream) {
        std::filesystem::remove(temporary);
        setError(error, "failed to write temporary file: " + temporary.string());
        return false;
      }
    }
    if (!syncPath(temporary, false)) {
      std::filesystem::remove(temporary);
      setError(error, "failed to fsync temporary file: " + temporary.string());
      return false;
    }
    std::filesystem::rename(temporary, target);
    if (!target.parent_path().empty()) {
      (void)syncPath(target.parent_path(), true);
    }
    return true;
  } catch (const std::exception& exception) {
    setError(error, exception.what());
    return false;
  }
}

bool atomicWrite(const std::filesystem::path& target,
                 const std::vector<std::uint8_t>& bytes,
                 std::string* error) {
  return atomicWrite(target, bytes.data(), bytes.size(), error);
}

bool atomicWrite(const std::filesystem::path& target,
                 const std::string& value,
                 std::string* error) {
  return atomicWrite(target, value.data(), value.size(), error);
}

std::string normalizedEncoding(const ImageBuffer& image) {
  if (!image.encoding.empty()) {
    return image.encoding;
  }
  if (image.channels == 1) {
    return "mono8";
  }
  if (image.channels == 3) {
    return "rgb8";
  }
  if (image.channels == 4) {
    return "rgba8";
  }
  return {};
}

bool encodePng(const ImageBuffer& image,
               std::string* source_encoding,
               std::vector<std::uint8_t>* bytes,
               std::string* error) {
  if (image.empty()) {
    setError(error, "full-frame asset is empty");
    return false;
  }
  if (image.channels != 1 && image.channels != 3 && image.channels != 4) {
    setError(error, "full-frame asset must have one, three, or four channels");
    return false;
  }
  const std::size_t expected = static_cast<std::size_t>(image.width) *
                               static_cast<std::size_t>(image.height) *
                               static_cast<std::size_t>(image.channels);
  if (image.data.size() != expected) {
    setError(error, "full-frame asset byte count does not match dimensions");
    return false;
  }
  *source_encoding = normalizedEncoding(image);
  const bool mono = *source_encoding == "mono8" || *source_encoding == "8UC1";
  const bool rgb = *source_encoding == "rgb8" || *source_encoding == "8UC3";
  const bool bgr = *source_encoding == "bgr8";
  const bool rgba = *source_encoding == "rgba8" || *source_encoding == "8UC4";
  const bool bgra = *source_encoding == "bgra8";
  if ((image.channels == 1 && !mono) ||
      (image.channels == 3 && !rgb && !bgr) ||
      (image.channels == 4 && !rgba && !bgra)) {
    setError(error, "unsupported full-frame encoding: " + *source_encoding);
    return false;
  }

  std::vector<std::uint8_t> converted;
  const std::uint8_t* pixels = image.data.data();
  if (bgr || bgra) {
    converted = image.data;
    for (std::size_t offset = 0; offset < converted.size();
         offset += static_cast<std::size_t>(image.channels)) {
      std::swap(converted[offset], converted[offset + 2]);
    }
    pixels = converted.data();
  }

  png_image png{};
  png.version = PNG_IMAGE_VERSION;
  png.width = static_cast<png_uint_32>(image.width);
  png.height = static_cast<png_uint_32>(image.height);
  png.format = image.channels == 1
                   ? PNG_FORMAT_GRAY
                   : (image.channels == 3 ? PNG_FORMAT_RGB
                                          : PNG_FORMAT_RGBA);
  png_alloc_size_t byte_count = 0;
  if (!png_image_write_to_memory(
          &png, nullptr, &byte_count, 0, pixels, 0, nullptr)) {
    setError(error, png.message);
    png_image_free(&png);
    return false;
  }
  bytes->resize(static_cast<std::size_t>(byte_count));
  if (!png_image_write_to_memory(
          &png, bytes->data(), &byte_count, 0, pixels, 0, nullptr)) {
    setError(error, png.message);
    png_image_free(&png);
    bytes->clear();
    return false;
  }
  bytes->resize(static_cast<std::size_t>(byte_count));
  png_image_free(&png);
  return true;
}

bool decodePng(const AssetRecord& asset,
               const std::vector<std::uint8_t>& bytes,
               ImageBuffer* frame,
               std::string* error) {
  png_image png{};
  png.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_memory(&png, bytes.data(), bytes.size())) {
    setError(error, png.message);
    png_image_free(&png);
    return false;
  }
  if (static_cast<int>(png.width) != asset.width ||
      static_cast<int>(png.height) != asset.height) {
    setError(error, "decoded PNG metadata mismatch: " + asset.id);
    png_image_free(&png);
    return false;
  }
  png.format = asset.channels == 1
                   ? PNG_FORMAT_GRAY
                   : (asset.channels == 3 ? PNG_FORMAT_RGB
                                          : PNG_FORMAT_RGBA);
  std::vector<std::uint8_t> pixels(PNG_IMAGE_SIZE(png));
  if (!png_image_finish_read(
          &png, nullptr, pixels.data(), 0, nullptr)) {
    setError(error, png.message);
    png_image_free(&png);
    return false;
  }
  png_image_free(&png);
  if (asset.source_encoding == "bgr8" ||
      asset.source_encoding == "bgra8") {
    for (std::size_t offset = 0; offset < pixels.size();
         offset += static_cast<std::size_t>(asset.channels)) {
      std::swap(pixels[offset], pixels[offset + 2]);
    }
  }
  frame->width = asset.width;
  frame->height = asset.height;
  frame->channels = asset.channels;
  frame->encoding = asset.source_encoding;
  frame->data = std::move(pixels);
  return true;
}

std::string floatToken(float value) {
  if (value == 0.0f) {
    value = 0.0f;  // Canonicalize negative zero.
  }
  std::ostringstream output;
  output << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return output.str();
}

std::string evidenceHash(const SnapshotRecord& record) {
  std::ostringstream evidence;
  evidence << "roomie.snapshot-evidence.v1\n" << record.source_frame_asset_id
           << '\n';
  for (const float value : record.bbox_xyxy) {
    evidence << floatToken(value) << '\n';
  }
  evidence << floatToken(record.crop.source_x) << '\n'
           << floatToken(record.crop.source_y) << '\n'
           << floatToken(record.crop.source_width) << '\n'
           << floatToken(record.crop.source_height) << '\n'
           << floatToken(record.crop.output_scale_x) << '\n'
           << floatToken(record.crop.output_scale_y) << '\n'
           << snapshotMaskSourceName(record.mask_source) << '\n'
           << record.mask_ref << '\n';
  return sha256(evidence.str());
}

std::vector<std::string> evidenceKeys(const SnapshotSet& snapshots) {
  std::vector<std::string> keys;
  keys.reserve(snapshots.records.size());
  for (const SnapshotRecord& record : snapshots.records) {
    keys.push_back(record.evidence_hash);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

std::string setHash(const std::vector<SnapshotRecord>& records) {
  std::vector<std::string> keys;
  keys.reserve(records.size());
  for (const SnapshotRecord& record : records) {
    keys.push_back(record.evidence_hash);
  }
  std::sort(keys.begin(), keys.end());
  std::ostringstream value;
  value << "roomie.snapshot-set.v1\n";
  for (const std::string& key : keys) {
    value << key << '\n';
  }
  return sha256(value.str());
}

bool sameEffectiveEvidence(const SnapshotSet& lhs, const SnapshotSet& rhs) {
  return evidenceKeys(lhs) == evidenceKeys(rhs);
}

void sortRecords(std::vector<SnapshotRecord>* records) {
  std::sort(records->begin(), records->end(),
            [](const SnapshotRecord& lhs, const SnapshotRecord& rhs) {
              if (std::abs(lhs.quality_score - rhs.quality_score) >
                  kScoreEpsilon) {
                return lhs.quality_score > rhs.quality_score;
              }
              return lhs.evidence_hash < rhs.evidence_hash;
            });
}

std::map<AssetId, std::int64_t> referenceDelta(
    const std::vector<SnapshotSet>& before,
    const std::vector<SnapshotSet>& after) {
  std::map<AssetId, std::int64_t> delta;
  for (const SnapshotSet& snapshots : before) {
    for (const SnapshotRecord& record : snapshots.records) {
      --delta[record.source_frame_asset_id];
    }
  }
  for (const SnapshotSet& snapshots : after) {
    for (const SnapshotRecord& record : snapshots.records) {
      ++delta[record.source_frame_asset_id];
    }
  }
  for (auto iterator = delta.begin(); iterator != delta.end();) {
    if (iterator->second == 0) {
      iterator = delta.erase(iterator);
    } else {
      ++iterator;
    }
  }
  return delta;
}

}  // namespace

struct AssetStore::Impl {
  explicit Impl(AssetStoreConfig input_config)
      : config(std::move(input_config)),
        assets_dir(config.root / "assets"),
        manifest_path(config.root / "manifest.json") {
    initialize();
  }

  void initialize() {
    try {
      if (config.root.empty()) {
        initialization_error = "asset store root must not be empty";
        return;
      }
      if (config.grace_period_ns < 0) {
        initialization_error = "asset store grace period must not be negative";
        return;
      }
      std::filesystem::create_directories(assets_dir);
      if (std::filesystem::exists(manifest_path)) {
        std::ifstream stream(manifest_path);
        if (!stream) {
          initialization_error = "failed to open asset manifest";
          return;
        }
        const nlohmann::json manifest = nlohmann::json::parse(stream);
        if (manifest.value("format", std::string()) !=
                "roomie_asset_store" ||
            manifest.value("version", 0) != 1 ||
            !manifest.contains("assets") ||
            !manifest.at("assets").is_object()) {
          initialization_error = "unsupported or malformed asset manifest";
          return;
        }
        for (auto iterator = manifest.at("assets").begin();
             iterator != manifest.at("assets").end(); ++iterator) {
          if (!validAssetId(iterator.key())) {
            initialization_error = "asset manifest contains an invalid id";
            return;
          }
          const nlohmann::json& value = iterator.value();
          AssetRecord record;
          record.id = iterator.key();
          record.path = assets_dir / (record.id + ".png");
          record.width = value.value("width", 0);
          record.height = value.value("height", 0);
          record.channels = value.value("channels", 0);
          record.source_encoding = value.value("source_encoding", std::string());
          record.encoded_bytes = value.value("encoded_bytes", std::uint64_t{0});
          record.ref_count = value.value("ref_count", std::uint64_t{0});
          record.unreferenced_since_ns =
              value.value("unreferenced_since_ns", TimeNanoseconds{0});
          if (record.width <= 0 || record.height <= 0 || record.channels <= 0 ||
              record.source_encoding.empty()) {
            initialization_error = "asset manifest contains invalid metadata";
            return;
          }
          assets.emplace(record.id, std::move(record));
        }
      }

      const TimeNanoseconds now_ns = wallNowNs();
      for (const std::filesystem::directory_entry& entry :
           std::filesystem::directory_iterator(assets_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".png") {
          continue;
        }
        const std::string id = entry.path().stem().string();
        if (validAssetId(id) && assets.count(id) == 0U) {
          orphan_since_ns.emplace(id, now_ns);
        }
      }
      initialized = true;
    } catch (const std::exception& exception) {
      initialization_error = exception.what();
    }
  }

  bool persistLocked(std::string* error) const {
    nlohmann::json manifest;
    manifest["format"] = "roomie_asset_store";
    manifest["version"] = 1;
    manifest["assets"] = nlohmann::json::object();
    for (const auto& [id, asset] : assets) {
      manifest["assets"][id] = {
          {"width", asset.width},
          {"height", asset.height},
          {"channels", asset.channels},
          {"source_encoding", asset.source_encoding},
          {"encoded_bytes", asset.encoded_bytes},
          {"ref_count", asset.ref_count},
          {"unreferenced_since_ns", asset.unreferenced_since_ns},
      };
    }
    return atomicWrite(manifest_path, manifest.dump(2) + "\n", error);
  }

  bool validateLocked(const AssetRecord& asset, std::string* error) const {
    std::vector<std::uint8_t> bytes;
    if (!readBytes(asset.path, &bytes, error)) {
      return false;
    }
    if (hashAssetBytes(asset, bytes) != asset.id) {
      setError(error, "asset content hash mismatch: " + asset.id);
      return false;
    }
    ImageBuffer decoded;
    return decodePng(asset, bytes, &decoded, error);
  }

  AssetStoreConfig config;
  std::filesystem::path assets_dir;
  std::filesystem::path manifest_path;
  mutable std::mutex mutex;
  std::map<AssetId, AssetRecord> assets;
  std::map<AssetId, TimeNanoseconds> orphan_since_ns;
  bool initialized = false;
  std::string initialization_error;
};

AssetStore::AssetStore(AssetStoreConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

AssetStore::~AssetStore() = default;

bool AssetStore::healthy() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->initialized;
}

std::string AssetStore::initializationError() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->initialization_error;
}

AssetWriteResult AssetStore::materializeFrame(const ImageBuffer& frame,
                                              TimeNanoseconds now_ns) {
  AssetWriteResult result;
  std::string encoding;
  std::vector<std::uint8_t> png;
  if (!encodePng(frame, &encoding, &png, &result.error)) {
    return result;
  }

  AssetRecord candidate;
  candidate.width = frame.width;
  candidate.height = frame.height;
  candidate.channels = frame.channels;
  candidate.source_encoding = encoding;
  candidate.encoded_bytes = png.size();
  candidate.unreferenced_since_ns = resolvedNow(now_ns);
  candidate.id = hashAssetBytes(candidate, png);

  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) {
    result.error = impl_->initialization_error;
    return result;
  }
  candidate.path = impl_->assets_dir / (candidate.id + ".png");
  auto existing = impl_->assets.find(candidate.id);
  if (existing != impl_->assets.end()) {
    const AssetRecord& stored = existing->second;
    if (stored.width != candidate.width || stored.height != candidate.height ||
        stored.channels != candidate.channels ||
        stored.source_encoding != candidate.source_encoding) {
      result.error = "content-address collision has incompatible metadata";
      return result;
    }
    std::string validation_error;
    if (!impl_->validateLocked(stored, &validation_error)) {
      if (!atomicWrite(stored.path, png, &result.error)) {
        return result;
      }
    }
    result.success = true;
    result.created = false;
    result.asset = stored;
    return result;
  }

  if (!atomicWrite(candidate.path, png, &result.error)) {
    return result;
  }
  impl_->assets.emplace(candidate.id, candidate);
  impl_->orphan_since_ns.erase(candidate.id);
  if (!impl_->persistLocked(&result.error)) {
    impl_->assets.erase(candidate.id);
    impl_->orphan_since_ns.emplace(candidate.id, candidate.unreferenced_since_ns);
    return result;
  }
  result.success = true;
  result.created = true;
  result.asset = candidate;
  return result;
}

bool AssetStore::applyReferenceDelta(
    const std::map<AssetId, std::int64_t>& delta,
    TimeNanoseconds now_ns,
    std::string* error) {
  const TimeNanoseconds now = resolvedNow(now_ns);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) {
    setError(error, impl_->initialization_error);
    return false;
  }

  std::map<AssetId, std::pair<std::uint64_t, TimeNanoseconds>> previous;
  for (const auto& [id, change] : delta) {
    if (change == 0) {
      continue;
    }
    if (change == std::numeric_limits<std::int64_t>::min()) {
      setError(error, "reference delta is outside the supported range: " + id);
      return false;
    }
    auto asset = impl_->assets.find(id);
    if (asset == impl_->assets.end()) {
      setError(error, "reference delta names an unknown asset: " + id);
      return false;
    }
    if (change < 0 &&
        static_cast<std::uint64_t>(-change) > asset->second.ref_count) {
      setError(error, "reference delta would underflow asset: " + id);
      return false;
    }
    if (change > 0 &&
        static_cast<std::uint64_t>(change) >
            std::numeric_limits<std::uint64_t>::max() -
                asset->second.ref_count) {
      setError(error, "reference delta would overflow asset: " + id);
      return false;
    }
  }

  for (const auto& [id, change] : delta) {
    if (change == 0) {
      continue;
    }
    AssetRecord& asset = impl_->assets.at(id);
    previous.emplace(id,
                     std::make_pair(asset.ref_count,
                                    asset.unreferenced_since_ns));
    if (change < 0) {
      asset.ref_count -= static_cast<std::uint64_t>(-change);
    } else {
      asset.ref_count += static_cast<std::uint64_t>(change);
    }
    if (asset.ref_count == 0U) {
      if (asset.unreferenced_since_ns == 0) {
        asset.unreferenced_since_ns = now;
      }
    } else {
      asset.unreferenced_since_ns = 0;
    }
  }
  if (!impl_->persistLocked(error)) {
    for (const auto& [id, values] : previous) {
      impl_->assets.at(id).ref_count = values.first;
      impl_->assets.at(id).unreferenced_since_ns = values.second;
    }
    return false;
  }
  return true;
}

bool AssetStore::reconcileReferenceCounts(
    const std::map<AssetId, std::uint64_t>& desired_counts,
    TimeNanoseconds now_ns,
    std::string* error) {
  const TimeNanoseconds now = resolvedNow(now_ns);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) {
    setError(error, impl_->initialization_error);
    return false;
  }
  for (const auto& [id, count] : desired_counts) {
    (void)count;
    if (impl_->assets.count(id) == 0U) {
      setError(error, "reference reconciliation names an unknown asset: " + id);
      return false;
    }
  }

  std::map<AssetId, std::pair<std::uint64_t, TimeNanoseconds>> previous;
  for (auto& [id, asset] : impl_->assets) {
    previous.emplace(id, std::make_pair(asset.ref_count,
                                        asset.unreferenced_since_ns));
    const auto desired = desired_counts.find(id);
    asset.ref_count = desired == desired_counts.end() ? 0U : desired->second;
    if (asset.ref_count == 0U) {
      if (asset.unreferenced_since_ns == 0) {
        asset.unreferenced_since_ns = now;
      }
    } else {
      asset.unreferenced_since_ns = 0;
    }
  }
  if (!impl_->persistLocked(error)) {
    for (const auto& [id, values] : previous) {
      impl_->assets.at(id).ref_count = values.first;
      impl_->assets.at(id).unreferenced_since_ns = values.second;
    }
    return false;
  }
  return true;
}

bool AssetStore::retain(const AssetId& id, std::string* error) {
  return applyReferenceDelta({{id, 1}}, wallNowNs(), error);
}

bool AssetStore::release(const AssetId& id,
                         TimeNanoseconds now_ns,
                         std::string* error) {
  return applyReferenceDelta({{id, -1}}, resolvedNow(now_ns), error);
}

std::optional<AssetRecord> AssetStore::record(const AssetId& id) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto iterator = impl_->assets.find(id);
  if (iterator == impl_->assets.end()) {
    return std::nullopt;
  }
  return iterator->second;
}

std::optional<std::filesystem::path> AssetStore::pathFor(
    const AssetId& id) const {
  const std::optional<AssetRecord> asset = record(id);
  return asset ? std::optional<std::filesystem::path>(asset->path) : std::nullopt;
}

std::optional<ImageBuffer> AssetStore::readFrame(const AssetId& id,
                                                 std::string* error) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto iterator = impl_->assets.find(id);
  if (iterator == impl_->assets.end()) {
    setError(error, "unknown asset: " + id);
    return std::nullopt;
  }
  std::vector<std::uint8_t> bytes;
  if (!readBytes(iterator->second.path, &bytes, error)) {
    return std::nullopt;
  }
  if (hashAssetBytes(iterator->second, bytes) != id) {
    setError(error, "asset content hash mismatch: " + id);
    return std::nullopt;
  }
  ImageBuffer frame;
  if (!decodePng(iterator->second, bytes, &frame, error)) {
    return std::nullopt;
  }
  return frame;
}

bool AssetStore::validate(const AssetId& id, std::string* error) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto iterator = impl_->assets.find(id);
  if (iterator == impl_->assets.end()) {
    setError(error, "unknown asset: " + id);
    return false;
  }
  return impl_->validateLocked(iterator->second, error);
}

std::size_t AssetStore::assetCount() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->assets.size();
}

std::uint64_t AssetStore::totalReferencedCount() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::uint64_t count = 0;
  for (const auto& [id, asset] : impl_->assets) {
    (void)id;
    count += asset.ref_count;
  }
  return count;
}

AssetGcResult AssetStore::collectGarbage(TimeNanoseconds now_ns) {
  AssetGcResult result;
  const TimeNanoseconds now = resolvedNow(now_ns);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) {
    result.error = impl_->initialization_error;
    return result;
  }

  std::vector<AssetRecord> expired;
  for (const auto& [id, asset] : impl_->assets) {
    (void)id;
    if (asset.ref_count == 0U && asset.unreferenced_since_ns != 0 &&
        now >= asset.unreferenced_since_ns &&
        now - asset.unreferenced_since_ns >= impl_->config.grace_period_ns) {
      expired.push_back(asset);
    }
  }
  const auto previous_assets = impl_->assets;
  for (const AssetRecord& asset : expired) {
    impl_->assets.erase(asset.id);
  }
  if (!expired.empty() && !impl_->persistLocked(&result.error)) {
    impl_->assets = previous_assets;
    return result;
  }

  for (const AssetRecord& asset : expired) {
    std::error_code remove_error;
    if (std::filesystem::remove(asset.path, remove_error)) {
      ++result.removed_assets;
      result.removed_bytes += asset.encoded_bytes;
      result.removed_ids.push_back(asset.id);
    } else if (remove_error) {
      impl_->orphan_since_ns.emplace(asset.id, now);
      if (result.error.empty()) {
        result.error = "failed to remove asset " + asset.id + ": " +
                       remove_error.message();
      }
    }
  }

  for (auto iterator = impl_->orphan_since_ns.begin();
       iterator != impl_->orphan_since_ns.end();) {
    if (now < iterator->second ||
        now - iterator->second < impl_->config.grace_period_ns) {
      ++iterator;
      continue;
    }
    const std::filesystem::path orphan_path =
        impl_->assets_dir / (iterator->first + ".png");
    std::error_code remove_error;
    const bool exists = std::filesystem::exists(orphan_path, remove_error);
    std::uint64_t byte_count = 0U;
    if (!remove_error && exists) {
      byte_count = std::filesystem::file_size(orphan_path, remove_error);
    }
    const bool removed =
        !remove_error &&
        (!exists || std::filesystem::remove(orphan_path, remove_error));
    if (removed && !remove_error) {
      ++result.removed_assets;
      result.removed_bytes += byte_count;
      result.removed_ids.push_back(iterator->first);
      iterator = impl_->orphan_since_ns.erase(iterator);
    } else {
      if (result.error.empty()) {
        result.error = "failed to remove orphan asset " + iterator->first;
      }
      ++iterator;
    }
  }
  return result;
}

const char* snapshotMaskSourceName(SnapshotMaskSource source) {
  switch (source) {
    case SnapshotMaskSource::kInstanceMask:
      return "instance_mask";
    case SnapshotMaskSource::kBboxFallback:
      return "bbox_fallback";
  }
  return "unknown";
}

std::string snapshotSetHashForReferences(
    const std::vector<ObjectSnapshotRef>& references) {
  std::vector<std::string> keys;
  keys.reserve(references.size());
  for (const ObjectSnapshotRef& reference : references) {
    if (reference.evidence_hash.empty()) {
      return {};
    }
    keys.push_back(reference.evidence_hash);
  }
  std::sort(keys.begin(), keys.end());
  std::ostringstream value;
  value << "roomie.snapshot-set.v1\n";
  for (const std::string& key : keys) {
    value << key << '\n';
  }
  return sha256(value.str());
}

float SnapshotQualityComponents::score() const {
  return normalizedScore(confidence) * normalizedScore(edge_completeness) *
         normalizedScore(distance) * normalizedScore(position) *
         normalizedScore(size) * normalizedScore(blur) *
         normalizedScore(exposure) * normalizedScore(truncation);
}

struct SnapshotBank::Impl {
  struct OwnerState {
    SnapshotSet snapshots;
  };

  Impl(SnapshotBankConfig input_config,
       std::shared_ptr<AssetStore> input_asset_store)
      : config(std::move(input_config)), asset_store(std::move(input_asset_store)) {
    if (!asset_store) {
      configuration_error = "SnapshotBank requires an AssetStore";
    } else if (!asset_store->healthy()) {
      configuration_error = asset_store->initializationError();
    } else if (config.top_k == 0U) {
      configuration_error = "snapshot top_k must be positive";
    } else if (!std::isfinite(config.azimuth_bucket_degrees) ||
               config.azimuth_bucket_degrees <= 0.0f ||
               !std::isfinite(config.elevation_bucket_degrees) ||
               config.elevation_bucket_degrees <= 0.0f ||
               !std::isfinite(config.scale_bucket_ratio) ||
               config.scale_bucket_ratio <= 1.0f) {
      configuration_error = "snapshot diversity bucket configuration is invalid";
    } else if (!std::isfinite(config.minimum_quality) ||
               config.minimum_quality < 0.0f ||
               !std::isfinite(config.replacement_min_quality_delta) ||
               config.replacement_min_quality_delta < 0.0f ||
               !std::isfinite(config.replacement_min_quality_ratio) ||
               config.replacement_min_quality_ratio < 1.0f ||
               !std::isfinite(config.diversity_min_quality_ratio) ||
               config.diversity_min_quality_ratio < 0.0f ||
               config.diversity_min_quality_ratio > 1.0f) {
      configuration_error = "snapshot quality/hysteresis configuration is invalid";
    }
  }

  SnapshotViewBucket bucketFor(const SnapshotViewpoint& viewpoint) const {
    const double azimuth_width =
        static_cast<double>(config.azimuth_bucket_degrees) * kPi / 180.0;
    double azimuth = std::fmod(static_cast<double>(viewpoint.azimuth_rad) + kPi,
                               2.0 * kPi);
    if (azimuth < 0.0) {
      azimuth += 2.0 * kPi;
    }
    const double elevation_width =
        static_cast<double>(config.elevation_bucket_degrees) * kPi / 180.0;
    const double elevation = std::clamp(
        static_cast<double>(viewpoint.elevation_rad), -0.5 * kPi, 0.5 * kPi);
    const double scale = std::max(static_cast<double>(viewpoint.scale), 1.0e-12);
    SnapshotViewBucket bucket;
    bucket.azimuth = boundedFloorToInt(azimuth / azimuth_width);
    bucket.elevation =
        boundedFloorToInt((elevation + 0.5 * kPi) / elevation_width);
    bucket.scale = boundedFloorToInt(
        std::log(scale) /
        std::log(static_cast<double>(config.scale_bucket_ratio)));
    return bucket;
  }

  bool makeRecord(const SnapshotCandidate& candidate,
                  const AssetRecord& asset,
                  SnapshotRecord* record,
                  std::string* error) const {
    if (!candidate.full_frame || candidate.full_frame->empty()) {
      setError(error, "snapshot candidate has no full frame");
      return false;
    }
    if (candidate.mask_source == SnapshotMaskSource::kInstanceMask &&
        candidate.mask_ref.empty()) {
      setError(error, "instance-mask snapshot candidate has no mask ref");
      return false;
    }
    float x0 = std::min(candidate.bbox_xyxy[0], candidate.bbox_xyxy[2]);
    float y0 = std::min(candidate.bbox_xyxy[1], candidate.bbox_xyxy[3]);
    float x1 = std::max(candidate.bbox_xyxy[0], candidate.bbox_xyxy[2]);
    float y1 = std::max(candidate.bbox_xyxy[1], candidate.bbox_xyxy[3]);
    if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) ||
        !std::isfinite(y1)) {
      setError(error, "snapshot bbox contains non-finite coordinates");
      return false;
    }
    x0 = std::clamp(x0, 0.0f, static_cast<float>(asset.width));
    x1 = std::clamp(x1, 0.0f, static_cast<float>(asset.width));
    y0 = std::clamp(y0, 0.0f, static_cast<float>(asset.height));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(asset.height));
    if (x1 <= x0 || y1 <= y0) {
      setError(error, "snapshot bbox is empty after clipping");
      return false;
    }

    record->source_frame_asset_id = asset.id;
    record->bbox_xyxy = {x0, y0, x1, y1};
    record->crop = candidate.crop;
    if (record->crop.source_width <= 0.0f ||
        record->crop.source_height <= 0.0f) {
      record->crop.source_x = x0;
      record->crop.source_y = y0;
      record->crop.source_width = x1 - x0;
      record->crop.source_height = y1 - y0;
    }
    if (!std::isfinite(record->crop.source_x) ||
        !std::isfinite(record->crop.source_y) ||
        !std::isfinite(record->crop.source_width) ||
        !std::isfinite(record->crop.source_height) ||
        !std::isfinite(record->crop.output_scale_x) ||
        !std::isfinite(record->crop.output_scale_y) ||
        record->crop.source_x < 0.0f || record->crop.source_y < 0.0f ||
        record->crop.source_width <= 0.0f ||
        record->crop.source_height <= 0.0f ||
        record->crop.source_x + record->crop.source_width >
            static_cast<float>(asset.width) + kScoreEpsilon ||
        record->crop.source_y + record->crop.source_height >
            static_cast<float>(asset.height) + kScoreEpsilon ||
        record->crop.output_scale_x <= 0.0f ||
        record->crop.output_scale_y <= 0.0f) {
      setError(error, "snapshot crop transform is invalid");
      return false;
    }
    record->mask_source = candidate.mask_source;
    record->mask_ref =
        candidate.mask_source == SnapshotMaskSource::kInstanceMask
            ? candidate.mask_ref
            : std::string();
    record->quality = candidate.quality;
    record->quality_score = candidate.quality.score();
    record->viewpoint = candidate.viewpoint;
    if (!std::isfinite(record->viewpoint.azimuth_rad) ||
        !std::isfinite(record->viewpoint.elevation_rad) ||
        !std::isfinite(record->viewpoint.scale) ||
        record->viewpoint.scale <= 0.0f) {
      setError(error, "snapshot viewpoint is invalid");
      return false;
    }
    record->view_bucket = bucketFor(candidate.viewpoint);
    record->time_ns = candidate.time_ns;
    record->camera_id = candidate.camera_id;
    record->provenance = candidate.provenance;
    record->patch_depth_occlusion_shadow =
        candidate.patch_depth_occlusion_shadow;
    record->evidence_hash = evidenceHash(*record);
    return true;
  }

  bool validateRestoredRecord(SnapshotRecord* record,
                              std::string* error) const {
    if (record == nullptr || record->source_frame_asset_id.empty()) {
      setError(error, "restored snapshot has no source frame asset id");
      return false;
    }
    const std::optional<AssetRecord> asset =
        asset_store->record(record->source_frame_asset_id);
    if (!asset) {
      setError(error, "restored snapshot names an unknown asset: " +
                          record->source_frame_asset_id);
      return false;
    }
    if (!asset_store->validate(record->source_frame_asset_id, error)) {
      return false;
    }
    if (record->mask_source == SnapshotMaskSource::kInstanceMask &&
        record->mask_ref.empty()) {
      setError(error, "restored instance-mask snapshot has no mask ref");
      return false;
    }
    for (float coordinate : record->bbox_xyxy) {
      if (!std::isfinite(coordinate)) {
        setError(error, "restored snapshot bbox is not finite");
        return false;
      }
    }
    const float x0 = record->bbox_xyxy[0];
    const float y0 = record->bbox_xyxy[1];
    const float x1 = record->bbox_xyxy[2];
    const float y1 = record->bbox_xyxy[3];
    if (x0 < 0.0f || y0 < 0.0f || x1 <= x0 || y1 <= y0 ||
        x1 > static_cast<float>(asset->width) + kScoreEpsilon ||
        y1 > static_cast<float>(asset->height) + kScoreEpsilon) {
      setError(error, "restored snapshot bbox is outside its source frame");
      return false;
    }
    if (!std::isfinite(record->crop.source_x) ||
        !std::isfinite(record->crop.source_y) ||
        !std::isfinite(record->crop.source_width) ||
        !std::isfinite(record->crop.source_height) ||
        !std::isfinite(record->crop.output_scale_x) ||
        !std::isfinite(record->crop.output_scale_y) ||
        record->crop.source_x < 0.0f || record->crop.source_y < 0.0f ||
        record->crop.source_width <= 0.0f ||
        record->crop.source_height <= 0.0f ||
        record->crop.output_scale_x <= 0.0f ||
        record->crop.output_scale_y <= 0.0f ||
        record->crop.source_x + record->crop.source_width >
            static_cast<float>(asset->width) + kScoreEpsilon ||
        record->crop.source_y + record->crop.source_height >
            static_cast<float>(asset->height) + kScoreEpsilon) {
      setError(error, "restored snapshot crop is outside its source frame");
      return false;
    }
    if (!std::isfinite(record->viewpoint.azimuth_rad) ||
        !std::isfinite(record->viewpoint.elevation_rad) ||
        !std::isfinite(record->viewpoint.scale) ||
        record->viewpoint.scale <= 0.0f) {
      setError(error, "restored snapshot viewpoint is invalid");
      return false;
    }
    record->quality_score = record->quality.score();
    record->view_bucket = bucketFor(record->viewpoint);
    const std::string expected_evidence_hash = evidenceHash(*record);
    if (record->evidence_hash != expected_evidence_hash) {
      setError(error, "restored snapshot evidence hash mismatch");
      return false;
    }
    return true;
  }

  bool significantImprovement(float candidate, float incumbent) const {
    return candidate + kScoreEpsilon >=
               incumbent + config.replacement_min_quality_delta &&
           candidate + kScoreEpsilon >=
               incumbent * config.replacement_min_quality_ratio;
  }

  std::vector<SnapshotRecord> rankUnion(
      std::vector<SnapshotRecord> candidates) const {
    std::map<std::string, SnapshotRecord> unique;
    for (SnapshotRecord& candidate : candidates) {
      auto existing = unique.find(candidate.evidence_hash);
      if (existing == unique.end() ||
          candidate.quality_score > existing->second.quality_score +
                                        kScoreEpsilon) {
        unique[candidate.evidence_hash] = std::move(candidate);
      }
    }
    candidates.clear();
    candidates.reserve(unique.size());
    for (auto& [hash, candidate] : unique) {
      (void)hash;
      candidates.push_back(std::move(candidate));
    }
    sortRecords(&candidates);

    std::vector<SnapshotRecord> selected;
    selected.reserve(std::min(config.top_k, candidates.size()));
    std::set<SnapshotViewBucket> buckets;
    std::unordered_set<std::string> selected_hashes;
    for (const SnapshotRecord& candidate : candidates) {
      if (selected.size() >= config.top_k) {
        break;
      }
      if (buckets.insert(candidate.view_bucket).second) {
        selected.push_back(candidate);
        selected_hashes.insert(candidate.evidence_hash);
      }
    }
    for (const SnapshotRecord& candidate : candidates) {
      if (selected.size() >= config.top_k) {
        break;
      }
      if (selected_hashes.insert(candidate.evidence_hash).second) {
        selected.push_back(candidate);
      }
    }
    sortRecords(&selected);
    return selected;
  }

  SnapshotSubmitResult submit(OwnerState* owner,
                              const SnapshotCandidate& candidate,
                              TimeNanoseconds now_ns) {
    SnapshotSubmitResult result;
    if (!configuration_error.empty()) {
      result.error = configuration_error;
      return result;
    }
    if (!candidate.full_frame) {
      result.error = "snapshot candidate has no full frame";
      return result;
    }
    AssetWriteResult materialized =
        asset_store->materializeFrame(*candidate.full_frame, now_ns);
    if (!materialized.success) {
      result.error = materialized.error;
      return result;
    }
    SnapshotRecord incoming;
    if (!makeRecord(candidate, materialized.asset, &incoming, &result.error)) {
      return result;
    }
    if (incoming.quality_score + kScoreEpsilon < config.minimum_quality) {
      result.reason = "below_minimum_quality";
      result.appearance_revision = owner->snapshots.appearance_revision;
      result.snapshot_set_hash = owner->snapshots.snapshot_set_hash;
      return result;
    }

    SnapshotSet proposed = owner->snapshots;
    auto duplicate = std::find_if(
        proposed.records.begin(),
        proposed.records.end(),
        [&incoming](const SnapshotRecord& record) {
          return record.evidence_hash == incoming.evidence_hash;
        });
    if (duplicate != proposed.records.end()) {
      if (incoming.quality_score <= duplicate->quality_score + kScoreEpsilon) {
        result.reason = "duplicate_evidence_without_quality_improvement";
        result.appearance_revision = owner->snapshots.appearance_revision;
        result.snapshot_set_hash = owner->snapshots.snapshot_set_hash;
        return result;
      }
      *duplicate = incoming;
      sortRecords(&proposed.records);
      result.reason = "updated_evidence_metadata";
    } else if (proposed.records.size() < config.top_k) {
      proposed.records.push_back(incoming);
      sortRecords(&proposed.records);
      result.reason = "filled_available_slot";
    } else {
      std::map<SnapshotViewBucket, std::size_t> bucket_counts;
      for (const SnapshotRecord& record : proposed.records) {
        ++bucket_counts[record.view_bucket];
      }
      std::optional<std::size_t> target;
      bool diversity_replacement = false;
      for (std::size_t index = 0; index < proposed.records.size(); ++index) {
        if (proposed.records[index].view_bucket != incoming.view_bucket) {
          continue;
        }
        if (!target || proposed.records[index].quality_score <
                           proposed.records[*target].quality_score) {
          target = index;
        }
      }
      if (!target) {
        for (std::size_t index = 0; index < proposed.records.size(); ++index) {
          if (bucket_counts[proposed.records[index].view_bucket] <= 1U) {
            continue;
          }
          if (!target || proposed.records[index].quality_score <
                             proposed.records[*target].quality_score) {
            target = index;
          }
        }
        diversity_replacement = target.has_value();
      }
      if (!target) {
        target = static_cast<std::size_t>(std::distance(
            proposed.records.begin(),
            std::min_element(
                proposed.records.begin(),
                proposed.records.end(),
                [](const SnapshotRecord& lhs, const SnapshotRecord& rhs) {
                  return lhs.quality_score < rhs.quality_score;
                })));
      }

      const float incumbent_quality = proposed.records[*target].quality_score;
      const bool should_replace =
          diversity_replacement
              ? incoming.quality_score + kScoreEpsilon >=
                    incumbent_quality * config.diversity_min_quality_ratio
              : significantImprovement(incoming.quality_score,
                                       incumbent_quality);
      if (!should_replace) {
        result.reason = diversity_replacement
                            ? "diversity_candidate_below_quality_floor"
                            : "quality_hysteresis_rejected";
        result.appearance_revision = owner->snapshots.appearance_revision;
        result.snapshot_set_hash = owner->snapshots.snapshot_set_hash;
        return result;
      }
      proposed.records[*target] = incoming;
      sortRecords(&proposed.records);
      result.reason = diversity_replacement ? "improved_view_diversity"
                                            : "significant_quality_improvement";
    }

    proposed.snapshot_set_hash = setHash(proposed.records);
    const bool evidence_changed =
        !sameEffectiveEvidence(owner->snapshots, proposed);
    proposed.appearance_revision = owner->snapshots.appearance_revision +
                                   (evidence_changed ? 1U : 0U);
    const auto delta = referenceDelta({owner->snapshots}, {proposed});
    if (!asset_store->applyReferenceDelta(delta, now_ns, &result.error)) {
      return result;
    }
    owner->snapshots = std::move(proposed);
    result.accepted = true;
    result.effective_evidence_changed = evidence_changed;
    result.appearance_revision = owner->snapshots.appearance_revision;
    result.snapshot_set_hash = owner->snapshots.snapshot_set_hash;
    return result;
  }

  std::optional<int> resolveObject(int object_id) const {
    if (object_id < 0) {
      return std::nullopt;
    }
    std::set<int> visited;
    int current = object_id;
    while (true) {
      if (!visited.insert(current).second) {
        return std::nullopt;
      }
      const auto alias = aliases.find(current);
      if (alias == aliases.end()) {
        return current;
      }
      current = alias->second;
    }
  }

  SnapshotBankConfig config;
  std::shared_ptr<AssetStore> asset_store;
  mutable std::mutex mutex;
  std::map<int, OwnerState> tentative_tracks;
  std::map<int, OwnerState> objects;
  std::map<int, int> aliases;
  std::string configuration_error;
};

SnapshotBank::SnapshotBank(SnapshotBankConfig config,
                           std::shared_ptr<AssetStore> asset_store)
    : impl_(std::make_unique<Impl>(std::move(config),
                                   std::move(asset_store))) {}

SnapshotBank::~SnapshotBank() = default;

SnapshotSubmitResult SnapshotBank::submitForTentativeTrack(
    int track_id,
    const SnapshotCandidate& candidate,
    TimeNanoseconds now_ns) {
  SnapshotSubmitResult result;
  if (track_id < 0) {
    result.error = "tentative track id must be non-negative";
    return result;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto existing = impl_->tentative_tracks.find(track_id);
  Impl::OwnerState staged = existing == impl_->tentative_tracks.end()
                                ? Impl::OwnerState{}
                                : existing->second;
  result = impl_->submit(&staged, candidate, now_ns);
  if (result.accepted) {
    impl_->tentative_tracks[track_id] = std::move(staged);
  }
  return result;
}

SnapshotSubmitResult SnapshotBank::submitForObject(
    int object_id,
    const SnapshotCandidate& candidate,
    TimeNanoseconds now_ns) {
  SnapshotSubmitResult result;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const std::optional<int> canonical = impl_->resolveObject(object_id);
  if (!canonical) {
    result.error = "object alias cycle or invalid object id";
    return result;
  }
  const auto existing = impl_->objects.find(*canonical);
  Impl::OwnerState staged = existing == impl_->objects.end()
                                ? Impl::OwnerState{}
                                : existing->second;
  result = impl_->submit(&staged, candidate, now_ns);
  if (result.accepted) {
    impl_->objects[*canonical] = std::move(staged);
  }
  return result;
}

SnapshotMergeResult SnapshotBank::promoteTentativeTrack(
    int track_id,
    int object_id,
    TimeNanoseconds now_ns) {
  SnapshotMergeResult result;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->configuration_error.empty()) {
    result.error = impl_->configuration_error;
    return result;
  }
  const auto tentative = impl_->tentative_tracks.find(track_id);
  if (tentative == impl_->tentative_tracks.end()) {
    result.error = "unknown tentative track";
    return result;
  }
  const std::optional<int> canonical = impl_->resolveObject(object_id);
  if (!canonical) {
    result.error = "object alias cycle or invalid object id";
    return result;
  }

  const auto object = impl_->objects.find(*canonical);
  const SnapshotSet before_object =
      object == impl_->objects.end() ? SnapshotSet{} : object->second.snapshots;
  std::vector<SnapshotRecord> candidates = before_object.records;
  candidates.insert(candidates.end(),
                    tentative->second.snapshots.records.begin(),
                    tentative->second.snapshots.records.end());
  SnapshotSet proposed;
  proposed.records = impl_->rankUnion(std::move(candidates));
  proposed.snapshot_set_hash = setHash(proposed.records);
  result.effective_evidence_changed =
      !sameEffectiveEvidence(before_object, proposed);
  proposed.appearance_revision =
      before_object.appearance_revision +
      (result.effective_evidence_changed ? 1U : 0U);

  const auto delta = referenceDelta(
      {before_object, tentative->second.snapshots}, {proposed});
  if (!impl_->asset_store->applyReferenceDelta(delta, now_ns, &result.error)) {
    return result;
  }
  impl_->tentative_tracks.erase(tentative);
  impl_->objects[*canonical].snapshots = proposed;
  result.success = true;
  result.canonical_object_id = *canonical;
  result.appearance_revision = proposed.appearance_revision;
  result.snapshot_set_hash = proposed.snapshot_set_hash;
  return result;
}

SnapshotMergeResult SnapshotBank::mergeObjects(int retired_object_id,
                                               int canonical_object_id,
                                               TimeNanoseconds now_ns) {
  SnapshotMergeResult result;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->configuration_error.empty()) {
    result.error = impl_->configuration_error;
    return result;
  }
  const std::optional<int> retired = impl_->resolveObject(retired_object_id);
  const std::optional<int> canonical = impl_->resolveObject(canonical_object_id);
  if (!retired || !canonical) {
    result.error = "object alias cycle or invalid object id";
    return result;
  }
  if (*retired == *canonical) {
    result.success = true;
    result.canonical_object_id = *canonical;
    const auto owner = impl_->objects.find(*canonical);
    if (owner != impl_->objects.end()) {
      result.appearance_revision = owner->second.snapshots.appearance_revision;
      result.snapshot_set_hash = owner->second.snapshots.snapshot_set_hash;
    }
    return result;
  }

  const auto retired_owner = impl_->objects.find(*retired);
  const auto canonical_owner = impl_->objects.find(*canonical);
  const SnapshotSet before_retired =
      retired_owner == impl_->objects.end() ? SnapshotSet{}
                                            : retired_owner->second.snapshots;
  const SnapshotSet before_canonical =
      canonical_owner == impl_->objects.end() ? SnapshotSet{}
                                              : canonical_owner->second.snapshots;
  std::vector<SnapshotRecord> candidates = before_canonical.records;
  candidates.insert(candidates.end(),
                    before_retired.records.begin(),
                    before_retired.records.end());
  SnapshotSet proposed;
  proposed.records = impl_->rankUnion(std::move(candidates));
  proposed.snapshot_set_hash = setHash(proposed.records);
  result.effective_evidence_changed =
      !sameEffectiveEvidence(before_canonical, proposed);
  proposed.appearance_revision =
      before_canonical.appearance_revision +
      (result.effective_evidence_changed ? 1U : 0U);

  const auto delta =
      referenceDelta({before_canonical, before_retired}, {proposed});
  if (!impl_->asset_store->applyReferenceDelta(delta, now_ns, &result.error)) {
    return result;
  }
  impl_->objects.erase(*retired);
  impl_->objects[*canonical].snapshots = proposed;
  impl_->aliases[*retired] = *canonical;
  impl_->aliases[retired_object_id] = *canonical;
  result.success = true;
  result.canonical_object_id = *canonical;
  result.appearance_revision = proposed.appearance_revision;
  result.snapshot_set_hash = proposed.snapshot_set_hash;
  return result;
}

bool SnapshotBank::dropTentativeTrack(int track_id,
                                      TimeNanoseconds now_ns,
                                      std::string* error) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto owner = impl_->tentative_tracks.find(track_id);
  if (owner == impl_->tentative_tracks.end()) {
    return true;
  }
  const auto delta = referenceDelta({owner->second.snapshots}, {});
  if (!impl_->asset_store->applyReferenceDelta(delta, now_ns, error)) {
    return false;
  }
  impl_->tentative_tracks.erase(owner);
  return true;
}

bool SnapshotBank::eraseObject(int object_id,
                               TimeNanoseconds now_ns,
                               std::string* error) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const std::optional<int> canonical = impl_->resolveObject(object_id);
  if (!canonical) {
    setError(error, "object alias cycle or invalid object id");
    return false;
  }
  const auto owner = impl_->objects.find(*canonical);
  if (owner == impl_->objects.end()) {
    return true;
  }
  const auto delta = referenceDelta({owner->second.snapshots}, {});
  if (!impl_->asset_store->applyReferenceDelta(delta, now_ns, error)) {
    return false;
  }
  impl_->objects.erase(owner);
  return true;
}

bool SnapshotBank::restoreObjectSnapshots(int object_id,
                                          SnapshotSet snapshots,
                                          std::string* error) {
  if (object_id < 0) {
    setError(error, "restored object id must be non-negative");
    return false;
  }
  if (!impl_->configuration_error.empty()) {
    setError(error, impl_->configuration_error);
    return false;
  }
  if (snapshots.records.size() > impl_->config.top_k) {
    setError(error, "restored snapshot set exceeds configured top_k");
    return false;
  }
  std::set<std::string> evidence_hashes;
  for (SnapshotRecord& record : snapshots.records) {
    if (!impl_->validateRestoredRecord(&record, error)) {
      return false;
    }
    if (!evidence_hashes.insert(record.evidence_hash).second) {
      setError(error, "restored snapshot set contains duplicate evidence");
      return false;
    }
  }
  sortRecords(&snapshots.records);
  const std::string expected_set_hash = setHash(snapshots.records);
  if (snapshots.snapshot_set_hash != expected_set_hash) {
    setError(error, "restored snapshot set hash mismatch");
    return false;
  }

  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->aliases.count(object_id) != 0U) {
    setError(error, "restored snapshot owner is already an alias");
    return false;
  }
  if (impl_->objects.count(object_id) != 0U) {
    setError(error, "restored snapshot owner is duplicated");
    return false;
  }
  impl_->objects[object_id].snapshots = std::move(snapshots);
  return true;
}

bool SnapshotBank::restoreObjectAlias(int retired_object_id,
                                      int canonical_object_id,
                                      std::string* error) {
  if (retired_object_id < 0 || canonical_object_id < 0 ||
      retired_object_id == canonical_object_id) {
    setError(error, "restored snapshot alias is invalid");
    return false;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->objects.count(retired_object_id) != 0U ||
      impl_->aliases.count(retired_object_id) != 0U) {
    setError(error, "restored snapshot alias source is duplicated");
    return false;
  }
  impl_->aliases.emplace(retired_object_id, canonical_object_id);
  for (const auto& [retired, canonical] : impl_->aliases) {
    (void)canonical;
    if (!impl_->resolveObject(retired)) {
      impl_->aliases.erase(retired_object_id);
      setError(error, "restored snapshot aliases contain a cycle");
      return false;
    }
  }
  return true;
}

std::optional<SnapshotSet> SnapshotBank::tentativeSnapshots(int track_id) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto owner = impl_->tentative_tracks.find(track_id);
  return owner == impl_->tentative_tracks.end()
             ? std::nullopt
             : std::optional<SnapshotSet>(owner->second.snapshots);
}

std::optional<SnapshotSet> SnapshotBank::objectSnapshots(int object_id) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const std::optional<int> canonical = impl_->resolveObject(object_id);
  if (!canonical) {
    return std::nullopt;
  }
  const auto owner = impl_->objects.find(*canonical);
  return owner == impl_->objects.end()
             ? std::nullopt
             : std::optional<SnapshotSet>(owner->second.snapshots);
}

std::optional<int> SnapshotBank::resolveCanonicalObjectId(int object_id) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->resolveObject(object_id);
}

std::size_t SnapshotBank::objectCount() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->objects.size();
}

std::size_t SnapshotBank::tentativeTrackCount() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->tentative_tracks.size();
}

AssetGcResult SnapshotBank::collectGarbage(TimeNanoseconds now_ns) const {
  if (!impl_->asset_store) {
    AssetGcResult result;
    result.error = "SnapshotBank has no AssetStore";
    return result;
  }
  return impl_->asset_store->collectGarbage(now_ns);
}

}  // namespace roomie
