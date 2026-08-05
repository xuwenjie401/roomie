#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>

#include <unistd.h>

#include <gtest/gtest.h>

#include "roomie/artifacts/snapshot_bank.hpp"

namespace roomie {
namespace {

class TempDirectory {
 public:
  explicit TempDirectory(const std::string& label) {
    static std::atomic<std::uint64_t> sequence{1};
    path_ = std::filesystem::temp_directory_path() /
            ("roomie_" + label + "_" + std::to_string(::getpid()) + "_" +
             std::to_string(sequence.fetch_add(1)));
    std::filesystem::create_directories(path_);
  }

  ~TempDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

ImageBuffer makeImage(std::uint8_t seed, int width = 24, int height = 16) {
  ImageBuffer image;
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.encoding = "rgb8";
  image.data.resize(static_cast<std::size_t>(width * height * 3));
  for (std::size_t index = 0; index < image.data.size(); ++index) {
    image.data[index] =
        static_cast<std::uint8_t>((seed + index * 17U) % 251U);
  }
  return image;
}

SnapshotCandidate makeCandidate(std::uint8_t image_seed,
                                float quality,
                                float azimuth,
                                float elevation,
                                float scale,
                                float x_offset = 0.0f) {
  SnapshotCandidate candidate;
  candidate.full_frame =
      std::make_shared<const ImageBuffer>(makeImage(image_seed));
  candidate.bbox_xyxy = {2.0f + x_offset, 2.0f, 14.0f + x_offset, 12.0f};
  candidate.quality.confidence = quality;
  candidate.viewpoint.azimuth_rad = azimuth;
  candidate.viewpoint.elevation_rad = elevation;
  candidate.viewpoint.scale = scale;
  candidate.time_ns = 1000 + image_seed;
  candidate.camera_id = "head_rgbd";
  candidate.provenance.run_id = RunId{11, 22};
  candidate.provenance.frame_id = image_seed;
  candidate.patch_depth_occlusion_shadow = 0.25f;
  return candidate;
}

std::shared_ptr<AssetStore> makeStore(const std::filesystem::path& root,
                                      TimeNanoseconds grace_ns = 100) {
  AssetStoreConfig config;
  config.root = root;
  config.grace_period_ns = grace_ns;
  auto store = std::make_shared<AssetStore>(config);
  EXPECT_TRUE(store->healthy()) << store->initializationError();
  return store;
}

TEST(AssetStore, LosslessContentAddressingRefcountsRestartAndGc) {
  TempDirectory temporary("asset_store_lifecycle");
  const ImageBuffer image = makeImage(7);
  AssetId id;
  {
    auto store = makeStore(temporary.path(), 50);
    const AssetWriteResult first = store->materializeFrame(image, 100);
    ASSERT_TRUE(first.success) << first.error;
    EXPECT_TRUE(first.created);
    EXPECT_EQ(first.asset.id.size(), 64U);
    id = first.asset.id;

    const AssetWriteResult duplicate = store->materializeFrame(image, 110);
    ASSERT_TRUE(duplicate.success) << duplicate.error;
    EXPECT_FALSE(duplicate.created);
    EXPECT_EQ(duplicate.asset.id, id);
    EXPECT_EQ(store->assetCount(), 1U);

    std::string error;
    ASSERT_TRUE(store->retain(id, &error)) << error;
    ASSERT_TRUE(store->retain(id, &error)) << error;
    ASSERT_TRUE(store->record(id));
    EXPECT_EQ(store->record(id)->ref_count, 2U);
    ASSERT_TRUE(store->release(id, 190, &error)) << error;
    ASSERT_TRUE(store->release(id, 200, &error)) << error;
    EXPECT_EQ(store->record(id)->ref_count, 0U);
    EXPECT_EQ(store->record(id)->unreferenced_since_ns, 200);
  }

  {
    auto restarted = makeStore(temporary.path(), 50);
    ASSERT_EQ(restarted->assetCount(), 1U);
    ASSERT_TRUE(restarted->record(id));
    EXPECT_EQ(restarted->record(id)->ref_count, 0U);
    std::string error;
    const std::optional<ImageBuffer> decoded = restarted->readFrame(id, &error);
    ASSERT_TRUE(decoded) << error;
    EXPECT_EQ(decoded->width, image.width);
    EXPECT_EQ(decoded->height, image.height);
    EXPECT_EQ(decoded->encoding, image.encoding);
    EXPECT_EQ(decoded->data, image.data);

    EXPECT_EQ(restarted->collectGarbage(249).removed_assets, 0U);
    const AssetGcResult collected = restarted->collectGarbage(250);
    EXPECT_TRUE(collected.error.empty()) << collected.error;
    EXPECT_EQ(collected.removed_assets, 1U);
    EXPECT_EQ(collected.removed_ids, std::vector<AssetId>{id});
    EXPECT_EQ(restarted->assetCount(), 0U);
    EXPECT_FALSE(std::filesystem::exists(temporary.path() / "assets" /
                                         (id + ".png")));
  }
}

TEST(AssetStore, DetectsCorruptionAndRematerializationHealsAtomically) {
  TempDirectory temporary("asset_store_corruption");
  auto store = makeStore(temporary.path(), 10);
  const ImageBuffer image = makeImage(19);
  const AssetWriteResult written = store->materializeFrame(image, 100);
  ASSERT_TRUE(written.success) << written.error;
  std::string error;
  ASSERT_TRUE(store->retain(written.asset.id, &error)) << error;
  const std::optional<std::filesystem::path> path =
      store->pathFor(written.asset.id);
  ASSERT_TRUE(path);
  {
    std::ofstream corrupt(*path,
                          std::ios::binary | std::ios::out | std::ios::trunc);
    corrupt << "corrupt-png";
  }

  EXPECT_FALSE(store->validate(written.asset.id, &error));
  EXPECT_NE(error.find("hash mismatch"), std::string::npos);
  error.clear();
  EXPECT_FALSE(store->readFrame(written.asset.id, &error));
  EXPECT_EQ(store->collectGarbage(1000).removed_assets, 0U)
      << "referenced corrupt assets must be diagnosed, not silently deleted";

  const AssetWriteResult healed = store->materializeFrame(image, 500);
  ASSERT_TRUE(healed.success) << healed.error;
  EXPECT_FALSE(healed.created);
  error.clear();
  EXPECT_TRUE(store->validate(written.asset.id, &error)) << error;
  const std::optional<ImageBuffer> decoded =
      store->readFrame(written.asset.id, &error);
  ASSERT_TRUE(decoded) << error;
  EXPECT_EQ(decoded->data, image.data);

  for (const auto& entry :
       std::filesystem::directory_iterator(temporary.path() / "assets")) {
    EXPECT_EQ(entry.path().filename().string().find(".tmp."),
              std::string::npos);
  }
}

TEST(SnapshotQuality, ExposesEveryComponentInTheScore) {
  SnapshotQualityComponents quality;
  quality.confidence = 0.9f;
  quality.edge_completeness = 0.8f;
  quality.distance = 0.7f;
  quality.position = 0.6f;
  quality.size = 0.5f;
  quality.blur = 0.4f;
  quality.exposure = 0.3f;
  quality.truncation = 0.2f;
  EXPECT_NEAR(quality.score(),
              0.9f * 0.8f * 0.7f * 0.6f * 0.5f * 0.4f * 0.3f * 0.2f,
              1.0e-7f);
}

TEST(SnapshotBank, TopKDiversityHysteresisAndMaskProvenance) {
  TempDirectory temporary("snapshot_topk");
  auto store = makeStore(temporary.path());
  SnapshotBankConfig config;
  config.top_k = 3;
  config.replacement_min_quality_delta = 0.05f;
  config.replacement_min_quality_ratio = 1.05f;
  SnapshotBank bank(config, store);

  const SnapshotSubmitResult first = bank.submitForObject(
      4, makeCandidate(1, 0.70f, -2.0f, -0.4f, 0.01f), 100);
  const SnapshotSubmitResult second = bank.submitForObject(
      4, makeCandidate(2, 0.80f, 0.0f, 0.0f, 0.03f), 110);
  const SnapshotSubmitResult third = bank.submitForObject(
      4, makeCandidate(3, 0.60f, 2.0f, 0.4f, 0.10f), 120);
  ASSERT_TRUE(first.accepted) << first.error;
  ASSERT_TRUE(second.accepted) << second.error;
  ASSERT_TRUE(third.accepted) << third.error;
  const std::optional<SnapshotSet> initial_snapshots = bank.objectSnapshots(4);
  ASSERT_TRUE(initial_snapshots);
  EXPECT_EQ(initial_snapshots->records.size(), 3U);
  EXPECT_EQ(initial_snapshots->appearance_revision, 3U);

  std::set<SnapshotViewBucket> buckets;
  for (const SnapshotRecord& record : initial_snapshots->records) {
    buckets.insert(record.view_bucket);
    EXPECT_EQ(record.mask_source, SnapshotMaskSource::kBboxFallback);
    EXPECT_STREQ(snapshotMaskSourceName(record.mask_source), "bbox_fallback");
    ASSERT_TRUE(record.patch_depth_occlusion_shadow);
  }
  EXPECT_EQ(buckets.size(), 3U);

  // A slightly better frame in the first bucket cannot churn the set.
  const SnapshotSubmitResult small = bank.submitForObject(
      4, makeCandidate(4, 0.72f, -2.0f, -0.4f, 0.01f), 130);
  EXPECT_FALSE(small.accepted);
  EXPECT_EQ(small.reason, "quality_hysteresis_rejected");
  EXPECT_EQ(bank.objectSnapshots(4)->appearance_revision, 3U);

  const SnapshotSubmitResult large = bank.submitForObject(
      4, makeCandidate(5, 0.86f, -2.0f, -0.4f, 0.01f), 140);
  ASSERT_TRUE(large.accepted) << large.error;
  EXPECT_TRUE(large.effective_evidence_changed);
  EXPECT_EQ(large.appearance_revision, 4U);
  ASSERT_TRUE(bank.objectSnapshots(4));
  EXPECT_EQ(bank.objectSnapshots(4)->records.size(), 3U);
  EXPECT_NEAR(bank.objectSnapshots(4)->primary()->quality_score, 0.86f, 1.0e-6f);

  // Updating quality/provenance for identical pixels does not invalidate
  // appearance-dependent artifacts.
  SnapshotCandidate same_evidence =
      makeCandidate(5, 0.90f, -2.0f, -0.4f, 0.01f);
  same_evidence.provenance.frame_id = 999;
  const SnapshotSubmitResult metadata =
      bank.submitForObject(4, same_evidence, 150);
  ASSERT_TRUE(metadata.accepted) << metadata.error;
  EXPECT_FALSE(metadata.effective_evidence_changed);
  EXPECT_EQ(metadata.appearance_revision, 4U);
  EXPECT_EQ(metadata.snapshot_set_hash, large.snapshot_set_hash);
  EXPECT_NEAR(bank.objectSnapshots(4)->primary()->quality_score, 0.90f, 1.0e-6f);
}

TEST(SnapshotBank, SameFrameAcrossObjectsIsOnePhysicalAsset) {
  TempDirectory temporary("snapshot_frame_dedupe");
  auto store = makeStore(temporary.path());
  SnapshotBank bank(SnapshotBankConfig{}, store);
  const SnapshotCandidate shared = makeCandidate(42, 0.8f, 0.0f, 0.0f, 0.1f);

  ASSERT_TRUE(bank.submitForObject(1, shared, 100).accepted);
  SnapshotCandidate other_roi = shared;
  other_roi.bbox_xyxy = {8.0f, 3.0f, 20.0f, 14.0f};
  ASSERT_TRUE(bank.submitForObject(2, other_roi, 110).accepted);

  ASSERT_EQ(store->assetCount(), 1U);
  EXPECT_EQ(store->totalReferencedCount(), 2U);
  const std::optional<SnapshotSet> first_set = bank.objectSnapshots(1);
  const std::optional<SnapshotSet> second_set = bank.objectSnapshots(2);
  ASSERT_TRUE(first_set);
  ASSERT_TRUE(second_set);
  const SnapshotRecord& first = first_set->records.front();
  const SnapshotRecord& second = second_set->records.front();
  EXPECT_EQ(first.source_frame_asset_id, second.source_frame_asset_id);
  EXPECT_NE(first.evidence_hash, second.evidence_hash)
      << "object-specific bbox/crop must remain distinct logical evidence";
}

TEST(SnapshotBank, TentativePromotionAndAliasMergeLeaveNoDanglingRefs) {
  TempDirectory temporary("snapshot_promotion_merge");
  auto store = makeStore(temporary.path(), 20);
  SnapshotBankConfig config;
  config.top_k = 3;
  SnapshotBank bank(config, store);

  const SnapshotCandidate shared =
      makeCandidate(10, 0.90f, -1.0f, 0.0f, 0.02f);
  ASSERT_TRUE(bank.submitForTentativeTrack(5, shared, 100).accepted);
  ASSERT_TRUE(bank.submitForObject(
      10, makeCandidate(11, 0.80f, 0.5f, 0.0f, 0.04f), 110).accepted);
  const SnapshotMergeResult promoted = bank.promoteTentativeTrack(5, 10, 120);
  ASSERT_TRUE(promoted.success) << promoted.error;
  EXPECT_EQ(bank.tentativeTrackCount(), 0U);
  ASSERT_TRUE(bank.objectSnapshots(10));
  EXPECT_EQ(bank.objectSnapshots(10)->records.size(), 2U);

  // Retired object deliberately shares one exact evidence ref and contributes
  // one new view. Merge must collapse the duplicate owner ref and rerank.
  ASSERT_TRUE(bank.submitForObject(20, shared, 130).accepted);
  ASSERT_TRUE(bank.submitForObject(
      20, makeCandidate(12, 0.85f, 2.2f, 0.3f, 0.09f), 140).accepted);
  EXPECT_EQ(store->totalReferencedCount(), 4U);

  const SnapshotMergeResult merged = bank.mergeObjects(20, 10, 150);
  ASSERT_TRUE(merged.success) << merged.error;
  EXPECT_EQ(merged.canonical_object_id, 10);
  EXPECT_TRUE(merged.effective_evidence_changed);
  EXPECT_EQ(bank.resolveCanonicalObjectId(20), 10);
  ASSERT_TRUE(bank.objectSnapshots(20));
  ASSERT_TRUE(bank.objectSnapshots(10));
  const std::optional<SnapshotSet> retired_set = bank.objectSnapshots(20);
  const std::optional<SnapshotSet> canonical_set = bank.objectSnapshots(10);
  ASSERT_TRUE(retired_set);
  ASSERT_TRUE(canonical_set);
  EXPECT_EQ(retired_set->snapshot_set_hash, canonical_set->snapshot_set_hash);
  EXPECT_EQ(canonical_set->records.size(), 3U);
  EXPECT_EQ(store->totalReferencedCount(), 3U);

  std::set<std::string> evidence;
  for (const SnapshotRecord& record : canonical_set->records) {
    EXPECT_TRUE(evidence.insert(record.evidence_hash).second);
    std::string error;
    ASSERT_TRUE(store->record(record.source_frame_asset_id));
    EXPECT_GT(store->record(record.source_frame_asset_id)->ref_count, 0U);
    EXPECT_TRUE(store->validate(record.source_frame_asset_id, &error)) << error;
  }

  std::string error;
  ASSERT_TRUE(bank.eraseObject(10, 200, &error)) << error;
  EXPECT_EQ(store->totalReferencedCount(), 0U);
  const AssetGcResult gc = store->collectGarbage(220);
  EXPECT_TRUE(gc.error.empty()) << gc.error;
  EXPECT_EQ(store->assetCount(), 0U);
}

TEST(SnapshotBank, AppearanceRevisionTracksEffectivePixelsNotQuality) {
  TempDirectory temporary("snapshot_appearance_revision");
  auto store = makeStore(temporary.path());
  SnapshotBankConfig config;
  config.top_k = 1;
  config.replacement_min_quality_delta = 0.05f;
  config.replacement_min_quality_ratio = 1.05f;
  SnapshotBank bank(config, store);

  SnapshotCandidate candidate =
      makeCandidate(30, 0.70f, 0.0f, 0.0f, 0.1f);
  const SnapshotSubmitResult first = bank.submitForObject(7, candidate, 100);
  ASSERT_TRUE(first.accepted);
  EXPECT_EQ(first.appearance_revision, 1U);

  candidate.quality.confidence = 0.90f;
  const SnapshotSubmitResult quality_only =
      bank.submitForObject(7, candidate, 110);
  ASSERT_TRUE(quality_only.accepted);
  EXPECT_FALSE(quality_only.effective_evidence_changed);
  EXPECT_EQ(quality_only.appearance_revision, 1U);

  candidate.bbox_xyxy = {4.0f, 2.0f, 18.0f, 12.0f};
  candidate.crop = SnapshotCropTransform{};
  candidate.quality.confidence = 1.0f;
  const SnapshotSubmitResult changed_pixels =
      bank.submitForObject(7, candidate, 120);
  ASSERT_TRUE(changed_pixels.accepted) << changed_pixels.error;
  EXPECT_TRUE(changed_pixels.effective_evidence_changed);
  EXPECT_EQ(changed_pixels.appearance_revision, 2U);
  EXPECT_EQ(store->assetCount(), 1U);
  EXPECT_EQ(store->totalReferencedCount(), 1U);

  SnapshotCandidate invalid_mask = candidate;
  invalid_mask.mask_source = SnapshotMaskSource::kInstanceMask;
  invalid_mask.mask_ref.clear();
  const SnapshotSubmitResult invalid =
      bank.submitForObject(8, invalid_mask, 130);
  EXPECT_FALSE(invalid.accepted);
  EXPECT_NE(invalid.error.find("mask ref"), std::string::npos);
}

TEST(SnapshotBank, DurableRestoreAndManifestReconciliationSurviveRestart) {
  TempDirectory temporary("snapshot_restore");
  auto store = makeStore(temporary.path(), 100);
  SnapshotBankConfig config;
  config.top_k = 3;
  SnapshotBank original(config, store);

  ASSERT_TRUE(original.submitForObject(
      10, makeCandidate(70, 0.8f, -0.5f, 0.1f, 0.05f), 100).accepted);
  const std::optional<SnapshotSet> durable = original.objectSnapshots(10);
  ASSERT_TRUE(durable);
  ASSERT_EQ(durable->records.size(), 1U);
  const AssetId durable_id = durable->records.front().source_frame_asset_id;

  const AssetWriteResult orphan =
      store->materializeFrame(makeImage(71), 110);
  ASSERT_TRUE(orphan.success) << orphan.error;
  std::string error;
  ASSERT_TRUE(store->retain(durable_id, &error)) << error;
  ASSERT_TRUE(store->retain(orphan.asset.id, &error)) << error;
  EXPECT_EQ(store->totalReferencedCount(), 3U)
      << "simulate refcounts left ahead of the durable scene by a crash";

  SnapshotBank restored(config, store);
  ASSERT_TRUE(restored.restoreObjectSnapshots(10, *durable, &error)) << error;
  ASSERT_TRUE(restored.restoreObjectAlias(20, 10, &error)) << error;
  ASSERT_TRUE(store->reconcileReferenceCounts({{durable_id, 1U}}, 500,
                                               &error))
      << error;
  EXPECT_EQ(store->record(durable_id)->ref_count, 1U);
  EXPECT_EQ(store->record(orphan.asset.id)->ref_count, 0U);
  EXPECT_EQ(store->record(orphan.asset.id)->unreferenced_since_ns, 500);
  ASSERT_EQ(restored.resolveCanonicalObjectId(20), 10);
  ASSERT_TRUE(restored.objectSnapshots(20));
  EXPECT_EQ(restored.objectSnapshots(20)->snapshot_set_hash,
            durable->snapshot_set_hash);

  const SnapshotSubmitResult appended = restored.submitForObject(
      20, makeCandidate(72, 0.9f, 1.5f, 0.2f, 0.08f), 510);
  ASSERT_TRUE(appended.accepted) << appended.error;
  EXPECT_EQ(appended.appearance_revision,
            durable->appearance_revision + 1U);
  EXPECT_EQ(store->totalReferencedCount(), 2U);

  SnapshotSet corrupted = *durable;
  corrupted.snapshot_set_hash = "not-the-durable-hash";
  SnapshotBank rejected(config, store);
  error.clear();
  EXPECT_FALSE(rejected.restoreObjectSnapshots(11, corrupted, &error));
  EXPECT_NE(error.find("set hash mismatch"), std::string::npos);
}

}  // namespace
}  // namespace roomie
