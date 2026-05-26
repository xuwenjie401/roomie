#pragma once

#include <memory>

#include "roomie/pipeline/map_backend.hpp"

namespace roomie {

std::unique_ptr<MapBackend> createNvbloxMapBackend(const PipelineConfig& config);

}  // namespace roomie
