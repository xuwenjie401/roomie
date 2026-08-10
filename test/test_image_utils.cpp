#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "roomie/pipeline/image_utils.hpp"

namespace roomie {
namespace {

ImageBuffer image(int width,
                  int height,
                  int channels,
                  std::string encoding,
                  std::vector<std::uint8_t> data) {
  ImageBuffer result;
  result.width = width;
  result.height = height;
  result.channels = channels;
  result.encoding = std::move(encoding);
  result.data = std::move(data);
  return result;
}

TEST(ImageUtils, RejectsInvalidInput) {
  EXPECT_TRUE(resizeNearest(ImageBuffer{}, 4, 4).empty());
  EXPECT_TRUE(resizeBilinear(ImageBuffer{}, 4, 4).empty());

  const ImageBuffer valid = image(
      2, 2, 3, "bgr8", std::vector<std::uint8_t>(12, 7));
  EXPECT_TRUE(resizeNearest(valid, 0, 4).empty());
  EXPECT_TRUE(resizeBilinear(valid, 4, -1).empty());

  const ImageBuffer truncated = image(
      2, 2, 3, "bgr8", std::vector<std::uint8_t>(11, 7));
  EXPECT_TRUE(resizeNearest(truncated, 4, 4).empty());
  EXPECT_TRUE(resizeBilinear(truncated, 4, 4).empty());
}

TEST(ImageUtils, SameSizePreservesBytesAndMetadata) {
  const ImageBuffer input = image(
      2,
      2,
      3,
      "bgr8",
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

  const ImageBuffer nearest = resizeNearest(input, 2, 2);
  const ImageBuffer bilinear = resizeBilinear(input, 2, 2);
  for (const ImageBuffer* output : {&nearest, &bilinear}) {
    EXPECT_EQ(output->width, input.width);
    EXPECT_EQ(output->height, input.height);
    EXPECT_EQ(output->channels, input.channels);
    EXPECT_EQ(output->encoding, input.encoding);
    EXPECT_EQ(output->data, input.data);
  }
}

TEST(ImageUtils, NearestNeighborPreservesMaskLabels) {
  const ImageBuffer input = image(2, 2, 1, "mono8", {0, 64, 128, 255});
  const ImageBuffer output = resizeNearest(input, 4, 4);

  ASSERT_EQ(output.width, 4);
  ASSERT_EQ(output.height, 4);
  ASSERT_EQ(output.channels, 1);
  ASSERT_EQ(output.encoding, "mono8");
  ASSERT_EQ(output.data.size(), 16U);
  const std::set<std::uint8_t> labels(output.data.begin(), output.data.end());
  EXPECT_EQ(labels, (std::set<std::uint8_t>{0, 64, 128, 255}));
  EXPECT_EQ(output.data,
            (std::vector<std::uint8_t>{0, 0, 64, 64,
                                       0, 0, 64, 64,
                                       128, 128, 255, 255,
                                       128, 128, 255, 255}));
}

TEST(ImageUtils, BilinearResizePreservesConstantBgrImage) {
  std::vector<std::uint8_t> pixels;
  for (int index = 0; index < 4; ++index) {
    pixels.insert(pixels.end(), {17, 83, 211});
  }
  const ImageBuffer input = image(2, 2, 3, "bgr8", std::move(pixels));
  const ImageBuffer output = resizeBilinear(input, 7, 5);

  ASSERT_EQ(output.width, 7);
  ASSERT_EQ(output.height, 5);
  ASSERT_EQ(output.channels, 3);
  ASSERT_EQ(output.encoding, "bgr8");
  ASSERT_EQ(output.data.size(), 7U * 5U * 3U);
  for (std::size_t offset = 0; offset < output.data.size(); offset += 3U) {
    EXPECT_EQ(output.data[offset], 17);
    EXPECT_EQ(output.data[offset + 1U], 83);
    EXPECT_EQ(output.data[offset + 2U], 211);
  }
}

TEST(ImageUtils, ResizesProductionGeometry) {
  constexpr int kInputWidth = 640;
  constexpr int kInputHeight = 400;
  constexpr int kTargetSize = 960;
  std::vector<std::uint8_t> pixels(
      static_cast<std::size_t>(kInputWidth * kInputHeight * 3));
  for (std::size_t index = 0; index < pixels.size(); ++index) {
    pixels[index] = static_cast<std::uint8_t>(index % 251U);
  }
  const ImageBuffer input = image(
      kInputWidth, kInputHeight, 3, "bgr8", std::move(pixels));

  const ImageBuffer output = resizeBilinear(input, kTargetSize, kTargetSize);
  EXPECT_EQ(output.width, kTargetSize);
  EXPECT_EQ(output.height, kTargetSize);
  EXPECT_EQ(output.channels, 3);
  EXPECT_EQ(output.encoding, "bgr8");
  EXPECT_EQ(output.data.size(),
            static_cast<std::size_t>(kTargetSize * kTargetSize * 3));
}

TEST(ImageUtils, ReportsProductionGeometryBenchmark) {
  constexpr int kInputWidth = 640;
  constexpr int kInputHeight = 400;
  constexpr int kTargetSize = 960;
  constexpr int kWarmupIterations = 5;
  constexpr int kMeasuredIterations = 100;
  const ImageBuffer rgb = image(
      kInputWidth,
      kInputHeight,
      3,
      "bgr8",
      std::vector<std::uint8_t>(
          static_cast<std::size_t>(kInputWidth * kInputHeight * 3), 127));
  const ImageBuffer mask = image(
      kInputWidth,
      kInputHeight,
      1,
      "mono8",
      std::vector<std::uint8_t>(
          static_cast<std::size_t>(kInputWidth * kInputHeight), 0));

  ImageBuffer resized_rgb;
  ImageBuffer resized_mask;
  for (int index = 0; index < kWarmupIterations; ++index) {
    resized_rgb = resizeBilinear(rgb, kTargetSize, kTargetSize);
    resized_mask = resizeNearest(mask, kTargetSize, kTargetSize);
  }

  std::vector<double> elapsed_ms;
  elapsed_ms.reserve(kMeasuredIterations);
  for (int index = 0; index < kMeasuredIterations; ++index) {
    const auto started = std::chrono::steady_clock::now();
    resized_rgb = resizeBilinear(rgb, kTargetSize, kTargetSize);
    resized_mask = resizeNearest(mask, kTargetSize, kTargetSize);
    elapsed_ms.push_back(std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count());
  }
  std::sort(elapsed_ms.begin(), elapsed_ms.end());

  ASSERT_FALSE(resized_rgb.empty());
  ASSERT_FALSE(resized_mask.empty());
  const double p50 = elapsed_ms[elapsed_ms.size() / 2U];
  const double p90 = elapsed_ms[elapsed_ms.size() * 9U / 10U];
  std::cout << "production_resize_benchmark combined_rgb_mask_p50_ms="
            << p50 << " combined_rgb_mask_p90_ms=" << p90 << '\n';
  RecordProperty("combined_rgb_mask_p50_ms", std::to_string(p50));
  RecordProperty("combined_rgb_mask_p90_ms", std::to_string(p90));
}

}  // namespace
}  // namespace roomie
