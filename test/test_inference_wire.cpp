#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "roomie/pipeline/python_inference_backend.hpp"

namespace roomie {
namespace {

FrameProvenance makeProvenance(RequestId request_id) {
  FrameProvenance provenance;
  provenance.run_id = {0x0123456789abcdefULL, 0xfedcba9876543210ULL};
  provenance.frame_id = 0x1020304050607080ULL;
  provenance.request_id = request_id;
  provenance.sensor_time_ns = -123456789012345LL;
  provenance.map_mode = MapMode::kFrozen;
  provenance.includes_current_frame = false;
  provenance.causality_verified = true;
  provenance.map.map_epoch = {0x1112131415161718ULL, 0x2122232425262728ULL};
  provenance.map.map_revision = 0x3132333435363738ULL;
  provenance.map.integrated_through_ns = -998877665544LL;
  provenance.surface.map_epoch = {0x4142434445464748ULL,
                                  0x5152535455565758ULL};
  provenance.surface.surface_revision = 0x6162636465666768ULL;
  provenance.surface.source_map_revision = 0x7172737475767778ULL;
  return provenance;
}

PipelineTiming makeTiming() {
  PipelineTiming timing;
  timing.serialize_ms = 0.125;
  timing.pipe_write_ms = 1.25;
  timing.pipe_read_ms = 2.5;
  timing.worker_queue_ms = 5.0;
  timing.response_forward_ms = 10.0;
  return timing;
}

void expectTimingEqual(const PipelineTiming& expected,
                       const PipelineTiming& actual) {
  EXPECT_EQ(actual.serialize_ms, expected.serialize_ms);
  EXPECT_EQ(actual.pipe_write_ms, expected.pipe_write_ms);
  EXPECT_EQ(actual.pipe_read_ms, expected.pipe_read_ms);
  EXPECT_EQ(actual.worker_queue_ms, expected.worker_queue_ms);
  EXPECT_EQ(actual.response_forward_ms, expected.response_forward_ms);
}

InferenceRequest makeRequest(RequestId request_id) {
  InferenceRequest request;
  request.time_ns = 4242424242LL;
  request.provenance = makeProvenance(request_id);
  request.timing = makeTiming();
  request.camera_id = "head/color";
  request.rgb_960.width = 2;
  request.rgb_960.height = 1;
  request.rgb_960.channels = 3;
  request.rgb_960.encoding = "rgb8";
  request.rgb_960.data = {0, 1, 2, 253, 254, 255};
  request.mask_960.width = 2;
  request.mask_960.height = 1;
  request.mask_960.channels = 1;
  request.mask_960.encoding = "mono8";
  request.mask_960.data = {0, 255};
  request.intrinsics_960 = {960, 960, 501.25f, 502.5f, 479.75f, 480.125f};
  for (std::size_t i = 0; i < request.patch_depth.values.size(); ++i) {
    request.patch_depth.values[i] = static_cast<float>(i) * 0.03125f - 1.0f;
  }
  request.patch_depth.valid_patches = 3599;
  request.patch_depth.projected_points = 123456;
  request.patch_depth.map_version = 0xf0e0d0c0b0a09080ULL;
  request.patch_depth.provenance = request.provenance;
  request.T_world_camera = Eigen::Translation3f(1.25f, -2.5f, 3.75f) *
                           Eigen::AngleAxisf(0.25f, Eigen::Vector3f::UnitZ());
  return request;
}

TEST(InferenceWireV2, RequestRoundTripPreservesProvenanceAndPayload) {
  const InferenceRequest expected = makeRequest(0x8877665544332211ULL);
  std::vector<std::uint8_t> body;
  std::string error;
  ASSERT_TRUE(inference_wire::encodeRequest(expected, &body, &error)) << error;
  ASSERT_GE(body.size(), 5U);
  EXPECT_EQ(std::string(body.begin(), body.begin() + 5), "RIEQ2");

  InferenceRequest actual;
  ASSERT_TRUE(inference_wire::decodeRequest(body, &actual, &error)) << error;
  EXPECT_EQ(actual.time_ns, expected.time_ns);
  EXPECT_TRUE(actual.provenance == expected.provenance);
  EXPECT_TRUE(actual.patch_depth.provenance == expected.provenance);
  expectTimingEqual(expected.timing, actual.timing);
  EXPECT_EQ(actual.camera_id, expected.camera_id);
  EXPECT_EQ(actual.rgb_960.data, expected.rgb_960.data);
  EXPECT_EQ(actual.mask_960.data, expected.mask_960.data);
  EXPECT_EQ(actual.intrinsics_960.fx, expected.intrinsics_960.fx);
  EXPECT_EQ(actual.patch_depth.values, expected.patch_depth.values);
  EXPECT_EQ(actual.patch_depth.valid_patches, expected.patch_depth.valid_patches);
  EXPECT_EQ(actual.patch_depth.projected_points, expected.patch_depth.projected_points);
  EXPECT_EQ(actual.patch_depth.map_version, expected.patch_depth.map_version);
  EXPECT_TRUE(actual.T_world_camera.matrix().isApprox(
      expected.T_world_camera.matrix(), 0.0f));
}

TEST(InferenceWireV2, ResponseRoundTripPreservesProvenanceAndDetections) {
  InferenceResponse expected;
  expected.time_ns = 4242424242LL;
  expected.provenance = makeProvenance(9001);
  expected.timing = makeTiming();
  expected.camera_id = "head/color";
  expected.ok = true;
  expected.python_worker_ms = 11.0f;
  expected.python_preprocess_ms = 1.0f;
  expected.owl_ms = 2.0f;
  expected.robot_filter_ms = 3.0f;
  expected.boxernet_ms = 4.0f;
  expected.python_postprocess_ms = 5.0f;

  Raw2dDetection detection_2d;
  detection_2d.score_2d = 0.875f;
  detection_2d.box_xyxy = {1.0f, 2.0f, 30.0f, 40.0f};
  detection_2d.semantic_id = 17;
  detection_2d.label = "coffee mug";
  expected.filtered_2d_detections.push_back(detection_2d);

  RawDetection detection_3d;
  detection_3d.center_world = Eigen::Vector3f(1.0f, -2.0f, 3.0f);
  detection_3d.size_m = Eigen::Vector3f(0.1f, 0.2f, 0.3f);
  detection_3d.yaw_rad = -0.5f;
  detection_3d.score_2d = 0.75f;
  detection_3d.score_3d = 0.625f;
  detection_3d.box_xyxy = {4.0f, 5.0f, 60.0f, 70.0f};
  detection_3d.semantic_id = 23;
  detection_3d.label = "chair";
  expected.detections.push_back(detection_3d);

  std::vector<std::uint8_t> body;
  std::string error;
  ASSERT_TRUE(inference_wire::encodeResponse(expected, &body, &error)) << error;
  EXPECT_EQ(std::string(body.begin(), body.begin() + 5), "RIRS2");

  InferenceResponse actual;
  ASSERT_TRUE(inference_wire::decodeResponse(body, &actual, &error)) << error;
  EXPECT_EQ(actual.time_ns, expected.time_ns);
  EXPECT_TRUE(actual.provenance == expected.provenance);
  expectTimingEqual(expected.timing, actual.timing);
  ASSERT_EQ(actual.filtered_2d_detections.size(), 1U);
  EXPECT_EQ(actual.filtered_2d_detections.front().label, detection_2d.label);
  EXPECT_EQ(actual.filtered_2d_detections.front().box_xyxy,
            detection_2d.box_xyxy);
  ASSERT_EQ(actual.detections.size(), 1U);
  EXPECT_EQ(actual.detections.front().label, detection_3d.label);
  EXPECT_EQ(actual.detections.front().center_world, detection_3d.center_world);
  EXPECT_EQ(actual.detections.front().score_3d, detection_3d.score_3d);
}

TEST(InferenceWireV2, RejectsBadMagicAndTruncatedBodies) {
  std::vector<std::uint8_t> request_body;
  std::string error;
  ASSERT_TRUE(inference_wire::encodeRequest(makeRequest(1), &request_body, &error));
  request_body[4] = '1';
  InferenceRequest request;
  EXPECT_FALSE(inference_wire::decodeRequest(request_body, &request, &error));
  EXPECT_NE(error.find("magic"), std::string::npos);

  InferenceResponse response_source;
  response_source.provenance = makeProvenance(1);
  std::vector<std::uint8_t> response_body;
  ASSERT_TRUE(
      inference_wire::encodeResponse(response_source, &response_body, &error));
  response_body.pop_back();
  InferenceResponse response;
  EXPECT_FALSE(inference_wire::decodeResponse(response_body, &response, &error));
}

TEST(InferenceWireV2, RequestIdDistinguishesSameCameraAndSensorTime) {
  InferenceRequest first = makeRequest(1001);
  InferenceRequest retry = makeRequest(1002);
  ASSERT_EQ(first.camera_id, retry.camera_id);
  ASSERT_EQ(first.time_ns, retry.time_ns);
  ASSERT_EQ(first.provenance.sensor_time_ns, retry.provenance.sensor_time_ns);

  std::vector<std::uint8_t> first_body;
  std::vector<std::uint8_t> retry_body;
  std::string error;
  ASSERT_TRUE(inference_wire::encodeRequest(first, &first_body, &error)) << error;
  ASSERT_TRUE(inference_wire::encodeRequest(retry, &retry_body, &error)) << error;
  EXPECT_NE(first_body, retry_body);

  InferenceRequest decoded_first;
  InferenceRequest decoded_retry;
  ASSERT_TRUE(
      inference_wire::decodeRequest(first_body, &decoded_first, &error));
  ASSERT_TRUE(
      inference_wire::decodeRequest(retry_body, &decoded_retry, &error));
  EXPECT_EQ(decoded_first.provenance.request_id, 1001U);
  EXPECT_EQ(decoded_retry.provenance.request_id, 1002U);
}

}  // namespace
}  // namespace roomie
