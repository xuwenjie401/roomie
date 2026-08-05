#include <gtest/gtest.h>

#include <set>
#include <string>

#include "roomie/pipeline/types.hpp"
#include "roomie/scene/scene_state.hpp"

namespace roomie {
namespace {

TEST(VersionTypes, RunIdsAreValidAndUnique) {
  std::set<std::string> ids;
  for (int i = 0; i < 128; ++i) {
    const RunId id = makeRunId();
    EXPECT_TRUE(id.valid());
    EXPECT_EQ(runIdString(id).size(), 32U);
    EXPECT_TRUE(ids.insert(runIdString(id)).second);
  }
}

TEST(VersionTypes, ProvenanceEqualityIncludesEveryDependency) {
  FrameProvenance lhs;
  lhs.run_id = makeRunId();
  lhs.frame_id = 17;
  lhs.request_id = 23;
  lhs.sensor_time_ns = 42;
  lhs.map_mode = MapMode::kFrozen;
  lhs.includes_current_frame = false;
  lhs.causality_verified = true;
  lhs.map.map_epoch = makeRunId();
  lhs.map.map_revision = 9;
  lhs.map.integrated_through_ns = 40;
  lhs.surface.map_epoch = lhs.map.map_epoch;
  lhs.surface.surface_revision = 11;
  lhs.surface.source_map_revision = 9;

  FrameProvenance rhs = lhs;
  EXPECT_TRUE(lhs == rhs);
  ++rhs.request_id;
  EXPECT_FALSE(lhs == rhs);
  rhs = lhs;
  ++rhs.surface.surface_revision;
  EXPECT_FALSE(lhs == rhs);
}

TEST(VersionTypes, ArtifactSloRequiresCanonicalUnixTrackingTuple) {
  ArtifactSloContext slo;
  EXPECT_TRUE(slo.valid());
  slo.due_unix_ms = -1;
  EXPECT_FALSE(slo.valid());
  slo = {};
  slo.origin_created_unix_ms = 1;
  EXPECT_FALSE(slo.valid());
  slo.due_unix_ms = 10;
  EXPECT_TRUE(slo.valid());
  slo.origin_created_unix_ms = 11;
  EXPECT_FALSE(slo.valid());
}

}  // namespace
}  // namespace roomie
