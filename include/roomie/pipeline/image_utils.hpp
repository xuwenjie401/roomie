#pragma once

#include "roomie/pipeline/types.hpp"

namespace roomie {

ImageBuffer resizeNearest(const ImageBuffer& input, int target_width, int target_height);
ImageBuffer resizeBilinear(const ImageBuffer& input, int target_width, int target_height);

}  // namespace roomie
