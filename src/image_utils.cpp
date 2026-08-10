#include "roomie/pipeline/image_utils.hpp"

#include <cstddef>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace roomie {

namespace {

ImageBuffer makeOutputLike(const ImageBuffer& input, int target_width, int target_height) {
  ImageBuffer output;
  output.width = target_width;
  output.height = target_height;
  output.channels = input.channels;
  output.encoding = input.encoding;
  output.data.resize(static_cast<std::size_t>(target_width) *
                     static_cast<std::size_t>(target_height) *
                     static_cast<std::size_t>(input.channels));
  return output;
}

bool canResize(const ImageBuffer& input, int target_width, int target_height) {
  if (input.empty() || target_width <= 0 || target_height <= 0) {
    return false;
  }
  const std::size_t expected = static_cast<std::size_t>(input.width) *
                               static_cast<std::size_t>(input.height) *
                               static_cast<std::size_t>(input.channels);
  return input.data.size() >= expected;
}

ImageBuffer resizeWithOpenCv(const ImageBuffer& input,
                             int target_width,
                             int target_height,
                             int interpolation) {
  if (!canResize(input, target_width, target_height) ||
      input.channels > CV_CN_MAX) {
    return {};
  }
  if (input.width == target_width && input.height == target_height) {
    return input;
  }

  ImageBuffer output = makeOutputLike(input, target_width, target_height);
  const int type = CV_MAKETYPE(CV_8U, input.channels);
  const cv::Mat source(
      input.height, input.width, type,
      const_cast<std::uint8_t*>(input.data.data()));
  cv::Mat destination(
      target_height, target_width, type, output.data.data());
  cv::resize(source,
             destination,
             cv::Size(target_width, target_height),
             0.0,
             0.0,
             interpolation);
  return output;
}

}  // namespace

ImageBuffer resizeNearest(const ImageBuffer& input, int target_width, int target_height) {
  return resizeWithOpenCv(input, target_width, target_height,
                          cv::INTER_NEAREST);
}

ImageBuffer resizeBilinear(const ImageBuffer& input, int target_width, int target_height) {
  return resizeWithOpenCv(input, target_width, target_height,
                          cv::INTER_LINEAR);
}

}  // namespace roomie
