#include "roomie/pipeline/map_backend.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "roomie/pipeline/cpu_point_map_backend.hpp"

#ifdef ROOMIE_ENABLE_NVBLOX
#include "roomie/pipeline/nvblox_map_backend.hpp"
#endif

namespace roomie {
namespace {

std::string normalizedBackendName(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return name;
}

}  // namespace

std::unique_ptr<MapBackend> createMapBackend(const PipelineConfig& config) {
  const std::string backend = normalizedBackendName(config.map_backend);
  if (backend == "nvblox") {
#ifdef ROOMIE_ENABLE_NVBLOX
    return createNvbloxMapBackend(config);
#else
    RCLCPP_WARN(rclcpp::get_logger("roomie.map_backend"),
                "map_backend is 'nvblox' but ROOMIE_ENABLE_NVBLOX is disabled; "
                "falling back to CPU point backend");
#endif
  } else if (backend != "cpu" && backend != "cpu_points") {
    RCLCPP_WARN(rclcpp::get_logger("roomie.map_backend"),
                "unknown map_backend '%s'; falling back to CPU point backend",
                config.map_backend.c_str());
  }
  return std::make_unique<CpuPointMapBackend>(config);
}

}  // namespace roomie
