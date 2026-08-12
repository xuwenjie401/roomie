#!/usr/bin/env python3
"""Tests for the AgiBot hand-color rectification adapter."""

from __future__ import annotations

import array
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import yaml


ROOMIE_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOMIE_ROOT / "scripts" / "roomie_agibot_hand_color_adapter.py"
SPEC = importlib.util.spec_from_file_location(
    "roomie_agibot_hand_color_adapter", ADAPTER_PATH
)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


CAMERA_CONFIG = ROOMIE_ROOT / "config" / "robots" / "G2" / "cameras.yaml"


def test_loads_both_hand_profiles_and_accepts_topic_overrides() -> None:
    calibrations = adapter.load_hand_camera_calibrations(
        CAMERA_CONFIG,
        ["hand_left_color", "hand_right_color"],
        ["/raw/left", "/raw/right"],
    )
    left, right = calibrations
    assert left.input_topic == "/raw/left"
    assert left.output_topic == "/roomie/input/hand_left_color/image_rect"
    assert left.camera_info_topic == "/roomie/input/hand_left_color/camera_info"
    assert left.frame == "hand_left_color"
    assert left.mount_frame == "arm_l_end_link"
    assert (left.width, left.height) == (1280, 1056)
    np.testing.assert_allclose(left.matrix[0], [491.1436278861935, 0.0, 644.7129435102285])
    assert np.linalg.norm(left.source_distortion) > 0.0
    assert right.input_topic == "/raw/right"
    assert right.frame == "hand_right_color"
    assert right.mount_frame == "arm_r_end_link"
    assert np.linalg.norm(right.source_distortion) > 0.0


def test_rectification_maps_raw_hand_geometry_into_same_pinhole_profile() -> None:
    calibration = adapter.load_hand_camera_calibrations(
        CAMERA_CONFIG, ["hand_left_color"]
    )[0]
    map_x, map_y = cv2.initUndistortRectifyMap(
        calibration.matrix,
        calibration.source_distortion,
        None,
        calibration.matrix,
        (calibration.width, calibration.height),
        cv2.CV_32FC1,
    )
    assert map_x.shape == (1056, 1280)
    assert map_y.shape == (1056, 1280)
    # Nonzero source distortion must produce a real remap away from the
    # principal point; otherwise map projection and image pixels disagree.
    assert abs(float(map_x[0, 0])) > 1.0
    assert abs(float(map_y[0, 0])) > 1.0
    center_x = int(round(calibration.matrix[0, 2]))
    center_y = int(round(calibration.matrix[1, 2]))
    assert abs(float(map_x[center_y, center_x]) - center_x) < 0.01
    assert abs(float(map_y[center_y, center_x]) - center_y) < 0.01


def test_output_messages_use_rectified_geometry_and_camera_frame() -> None:
    calibration = adapter.load_hand_camera_calibrations(
        CAMERA_CONFIG, ["hand_right_color"]
    )[0]
    stamp = adapter.Image().header.stamp
    stamp.sec = 123
    stamp.nanosec = 456
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    output = adapter._image_message(image, calibration.frame, stamp)
    assert output.header.frame_id == "hand_right_color"
    assert output.header.stamp == stamp
    assert output.encoding == "bgr8"
    assert isinstance(output.data, array.array)
    assert bytes(output.data) == image.tobytes()

    info = adapter._camera_info_message(calibration, stamp)
    assert info.header.frame_id == "hand_right_color"
    assert list(info.d) == [0.0] * 5
    assert info.k[0] == calibration.matrix[0, 0]
    assert info.k[4] == calibration.matrix[1, 1]
    transform = adapter._static_transform(calibration)
    assert transform.header.frame_id == "arm_r_end_link"
    assert transform.child_frame_id == "hand_right_color"


def test_pipeline_configs_enable_detection_only_hand_cameras() -> None:
    confirmation_bypass_labels = [
        "medicine_carton",
        "labeled_package",
        "printed_carton",
        "box",
        "bottled_water",
        "plastic_bag",
    ]
    for name in (
        "pipeline_agibot_head_mapping.yaml",
        "pipeline_genie_live.yaml",
        "pipeline_genie_load.yaml",
        "pipeline_genie_ephemeral.yaml",
    ):
        document = yaml.safe_load((ROOMIE_ROOT / "config" / name).read_text())
        detection = document["roomie_pipeline_node"]["ros__parameters"]["detection"]
        assert detection["additional_camera_ids"] == [
            "hand_left_color",
            "hand_right_color",
        ]
        queues = document["roomie_pipeline_node"]["ros__parameters"]["queues"]
        assert queues["detection_queue_size"] >= 3
        instance = document["roomie_pipeline_node"]["ros__parameters"]["instance"]
        assert (
            instance["presence_confirmation_bypass_labels"]
            == confirmation_bypass_labels
        )


def test_adapter_defaults_disabled_and_skips_before_image_processing() -> None:
    runtime = SimpleNamespace(received=0, disabled_skips=0)
    fake_node = SimpleNamespace(
        enabled=False,
        runtimes={"hand_left_color": runtime},
    )
    # No calibration, remap tables, publishers, or even valid pixel payload
    # are provided. Reaching any preprocessing code would fail this test.
    adapter.AgibotHandColorAdapter._handle_image(
        fake_node, "hand_left_color", adapter.Image()
    )
    assert runtime.received == 1
    assert runtime.disabled_skips == 1


def test_adapter_enable_topic_matches_pipeline_control_topic() -> None:
    document = yaml.safe_load(
        (ROOMIE_ROOT / "config" / "agibot_hand_color_adapter.yaml").read_text()
    )
    parameters = document["roomie_agibot_hand_color_adapter"]["ros__parameters"]
    assert parameters["enable_topic"] == "/roomie/hand_cameras/enabled"


def test_rviz_shows_per_camera_detection_images_without_raw_head_rgb() -> None:
    rviz_config = (ROOMIE_ROOT / "rviz" / "roomie_pipeline.rviz").read_text()

    assert "/roomie/input/head_color/image_rect" not in rviz_config
    assert "Value: /roomie/detections_2d_image\n" in rviz_config
    assert (
        "Value: /roomie/detections_2d_image/hand_left_color" in rviz_config
    )
    assert (
        "Value: /roomie/detections_2d_image/hand_right_color" in rviz_config
    )
