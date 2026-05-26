#include "roomie/pipeline/image_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

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

}  // namespace

ImageBuffer resizeNearest(const ImageBuffer& input, int target_width, int target_height) {
  if (!canResize(input, target_width, target_height)) {
    return {};
  }

  ImageBuffer output = makeOutputLike(input, target_width, target_height);
  const float scale_x = static_cast<float>(input.width) / static_cast<float>(target_width);
  const float scale_y = static_cast<float>(input.height) / static_cast<float>(target_height);

  for (int y = 0; y < target_height; ++y) {
    const int src_y =
        std::clamp(static_cast<int>(std::floor((static_cast<float>(y) + 0.5f) * scale_y)),
                   0,
                   input.height - 1);
    for (int x = 0; x < target_width; ++x) {
      const int src_x =
          std::clamp(static_cast<int>(std::floor((static_cast<float>(x) + 0.5f) * scale_x)),
                     0,
                     input.width - 1);
      const std::size_t src_offset =
          (static_cast<std::size_t>(src_y) * input.width + static_cast<std::size_t>(src_x)) *
          static_cast<std::size_t>(input.channels);
      const std::size_t dst_offset =
          (static_cast<std::size_t>(y) * target_width + static_cast<std::size_t>(x)) *
          static_cast<std::size_t>(input.channels);
      for (int c = 0; c < input.channels; ++c) {
        output.data[dst_offset + static_cast<std::size_t>(c)] =
            input.data[src_offset + static_cast<std::size_t>(c)];
      }
    }
  }

  return output;
}

ImageBuffer resizeBilinear(const ImageBuffer& input, int target_width, int target_height) {
  if (!canResize(input, target_width, target_height)) {
    return {};
  }

  if (input.width == target_width && input.height == target_height) {
    return input;
  }

  ImageBuffer output = makeOutputLike(input, target_width, target_height);
  const float scale_x = static_cast<float>(input.width) / static_cast<float>(target_width);
  const float scale_y = static_cast<float>(input.height) / static_cast<float>(target_height);

  for (int y = 0; y < target_height; ++y) {
    const float src_y_f =
        std::max(0.0f, (static_cast<float>(y) + 0.5f) * scale_y - 0.5f);
    const int y0 = std::clamp(static_cast<int>(std::floor(src_y_f)), 0, input.height - 1);
    const int y1 = std::min(y0 + 1, input.height - 1);
    const float wy = src_y_f - static_cast<float>(y0);

    for (int x = 0; x < target_width; ++x) {
      const float src_x_f =
          std::max(0.0f, (static_cast<float>(x) + 0.5f) * scale_x - 0.5f);
      const int x0 = std::clamp(static_cast<int>(std::floor(src_x_f)), 0, input.width - 1);
      const int x1 = std::min(x0 + 1, input.width - 1);
      const float wx = src_x_f - static_cast<float>(x0);

      const std::size_t dst_offset =
          (static_cast<std::size_t>(y) * target_width + static_cast<std::size_t>(x)) *
          static_cast<std::size_t>(input.channels);
      for (int c = 0; c < input.channels; ++c) {
        const std::size_t c_idx = static_cast<std::size_t>(c);
        const auto at = [&](int yy, int xx) -> float {
          const std::size_t offset =
              (static_cast<std::size_t>(yy) * input.width + static_cast<std::size_t>(xx)) *
                  static_cast<std::size_t>(input.channels) +
              c_idx;
          return static_cast<float>(input.data[offset]);
        };
        const float top = at(y0, x0) * (1.0f - wx) + at(y0, x1) * wx;
        const float bottom = at(y1, x0) * (1.0f - wx) + at(y1, x1) * wx;
        const float value = top * (1.0f - wy) + bottom * wy;
        output.data[dst_offset + c_idx] =
            static_cast<std::uint8_t>(std::clamp(std::lround(value), 0l, 255l));
      }
    }
  }

  return output;
}

}  // namespace roomie
