#!/usr/bin/env python3
"""Tests for the AgiBot head RGB-D registration adapter."""

from __future__ import annotations

import array
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml


ROOMIE_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOMIE_ROOT / "scripts" / "roomie_agibot_head_rgbd_adapter.py"
SPEC = importlib.util.spec_from_file_location("roomie_agibot_head_rgbd_adapter", ADAPTER_PATH)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


CALIBRATION_PATH = Path(
    "/home/lindenbot/sensor_base/agibot_genie/"
    "roomie_base/calib/head_camera_params.yaml"
)


def _roomie_parameters(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document["roomie_pipeline_node"]["ros__parameters"]


def _stamped_image(stamp_ns: int) -> object:
    message = adapter.Image()
    message.header.stamp.sec = stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = stamp_ns % 1_000_000_000
    return message


def test_image_message_uses_fast_byte_array_without_changing_payload() -> None:
    for source, encoding in (
        (np.arange(18, dtype=np.uint8).reshape(2, 3, 3), "bgr8"),
        (np.arange(6, dtype=np.float32).reshape(2, 3), "32FC1"),
    ):
        message = adapter._image_message(
            source,
            encoding,
            "camera",
            _stamped_image(1).header.stamp,
        )
        assert isinstance(message.data, array.array)
        assert message.data.typecode == "B"
        assert bytes(message.data) == source.tobytes()
        assert message.step == source.strides[0]


def test_scheduler_matches_depth_before_or_after_rgb_and_reuses_emitted_rgb() -> None:
    scheduler = adapter.RgbdPairScheduler(tolerance_ns=2_000_000, queue_size=4)
    color = _stamped_image(1_000_000_000)
    depth = _stamped_image(1_001_500_000)

    assert scheduler.add_color(color) is None
    selected = scheduler.take_latest_independent_color()
    assert selected == (1_000_000_000, color)
    scheduler.mark_color_emitted(selected[0])
    pair = scheduler.add_depth(depth)
    assert pair is not None
    assert pair.color is color
    assert pair.depth is depth
    assert pair.color_was_emitted
    assert pair.delta_ns == 1_500_000

    second_depth = _stamped_image(2_000_000_000)
    second_color = _stamped_image(2_001_000_000)
    assert scheduler.add_depth(second_depth) is None
    second_pair = scheduler.add_color(second_color)
    assert second_pair is not None
    assert not second_pair.color_was_emitted
    assert second_pair.delta_ns == 1_000_000


def test_mapping_pair_promotes_rgb_coalesced_by_latest_detection() -> None:
    scheduler = adapter.RgbdPairScheduler(tolerance_ns=2_000_000, queue_size=4)
    colors = [_stamped_image(stamp) for stamp in (10_000_000, 20_000_000, 30_000_000)]
    for color in colors:
        assert scheduler.add_color(color) is None

    selected = scheduler.take_latest_independent_color()
    assert selected == (30_000_000, colors[2])
    scheduler.mark_color_emitted(selected[0])
    assert scheduler.independent_coalesced == 2

    pair = scheduler.add_depth(_stamped_image(11_000_000))
    assert pair is not None
    assert pair.color is colors[0]
    assert not pair.color_was_emitted


def test_matched_depth_uses_rgb_stamp_without_republishing_emitted_rgb() -> None:
    color = _stamped_image(3_000_000_000)
    depth = _stamped_image(3_001_000_000)
    color_calls: list[object] = []
    depth_calls: list[tuple[object, object]] = []
    fake_node = SimpleNamespace(
        pair_matches=0,
        matched_pairs=0,
        paired_color_outputs=0,
        pair_processing_failures=0,
        last_sync_delta_ms=0.0,
        max_sync_delta_ms=0.0,
        _process_color=lambda message: color_calls.append(message) or True,
        _process_depth=lambda message, stamp: depth_calls.append((message, stamp)) or True,
    )
    pair = adapter.MatchedRgbdFrame(
        color=color,
        depth=depth,
        color_was_emitted=True,
        delta_ns=1_000_000,
    )

    adapter.AgibotHeadRgbdAdapter._process_matched_pair(fake_node, pair)

    assert color_calls == []
    assert depth_calls == [(depth, color.header.stamp)]
    assert fake_node.pair_matches == 1
    assert fake_node.matched_pairs == 1
    assert fake_node.last_sync_delta_ms == 1.0


def test_scheduler_bounds_unmatched_inputs_and_rejects_duplicate_depth() -> None:
    scheduler = adapter.RgbdPairScheduler(tolerance_ns=0, queue_size=2)
    scheduler.add_color(_stamped_image(1))
    scheduler.add_color(_stamped_image(2))
    scheduler.add_color(_stamped_image(3))
    assert list(scheduler.color_buffer) == [2, 3]
    assert scheduler.color_expirations == 1

    depth = _stamped_image(3)
    pair = scheduler.add_depth(depth)
    assert pair is not None
    assert scheduler.add_depth(depth) is None
    assert not scheduler.depth_buffer


def test_roomie_owns_robot_mask_generation_contract() -> None:
    adapter_params = yaml.safe_load(
        (ROOMIE_ROOT / "config" / "agibot_head_rgbd_adapter.yaml").read_text(
            encoding="utf-8"
        )
    )["roomie_agibot_head_rgbd_adapter"]["ros__parameters"]
    assert "output_mask_topic" not in adapter_params
    assert "publish_zero_robot_mask" not in adapter_params
    assert adapter_params["independent_rgb_max_fps"] == 5.0

    for name in (
        "pipeline_genie_live.yaml",
        "pipeline_genie_load.yaml",
        "pipeline_agibot_head_mapping.yaml",
    ):
        params = _roomie_parameters(ROOMIE_ROOT / "config" / name)
        assert "mask_topic" not in params["topics"]
        assert "mask_robot_threshold" not in params["input_filter"]
        assert params["robot_mask"] == {
            "robot_config": "G2/robot.yaml",
            "camera_config": "G2/cameras.yaml",
            "reuse_translation_epsilon_m": 5.0e-6,
            "reuse_rotation_epsilon_rad": 5.0e-6,
        }

    camera_document = yaml.safe_load(
        (ROOMIE_ROOT / "config" / "robots" / "G2" / "cameras.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert set(camera_document["cameras"]) == {
        "head_color",
        "hand_left_color",
        "hand_right_color",
    }
    for camera in camera_document["cameras"].values():
        assert camera["topic"].endswith("/image_rect")
        assert camera["distortion"]["coefficients"] == [0.0] * 5


def test_loads_head_color_depth_calibration() -> None:
    calibration = adapter.load_head_rgbd_calibration(
        CALIBRATION_PATH,
        "/gdk/camera/head_color",
        "/gdk/camera/head_depth",
    )
    assert calibration.color.frame == "head_color"
    assert calibration.depth.frame == "head_depth"
    assert (calibration.color.width, calibration.color.height) == (640, 400)
    assert (calibration.depth.width, calibration.depth.height) == (640, 400)
    np.testing.assert_allclose(
        calibration.translation_color_depth,
        [-0.0238200435638, -0.0002575260699, 0.0000125430729],
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        calibration.rotation_color_depth
        @ calibration.rotation_color_depth.T,
        np.eye(3),
        atol=1.0e-6,
    )


def test_identity_registration_preserves_depth_pixels() -> None:
    matrix = np.asarray(
        [
            [100.0, 0.0, 1.0],
            [0.0, 100.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera = adapter.CameraCalibration(
        camera_id="test",
        frame="camera",
        topic="/camera",
        width=3,
        height=3,
        projection_model="pinhole",
        distortion_model="plumb_bob",
        matrix=matrix,
        distortion=np.zeros(5, dtype=np.float64),
    )
    calibration = adapter.HeadRgbdCalibration(
        color=camera,
        depth=camera,
        rotation_color_depth=np.eye(3, dtype=np.float32),
        translation_color_depth=np.zeros(3, dtype=np.float32),
        static_transforms=(),
    )
    registrar = adapter.DepthToColorRegistrar(
        calibration,
        depth_min_m=0.1,
        depth_max_m=6.0,
    )
    source_depth = np.asarray(
        [
            [1.0, 1.1, 1.2],
            [1.3, 0.0, 1.5],
            [1.6, 1.7, 7.0],
        ],
        dtype=np.float32,
    )
    registered = registrar.register_depth(source_depth)
    expected = source_depth.copy()
    expected[expected > 6.0] = 0.0
    np.testing.assert_allclose(registered, expected, atol=1.0e-6)


def test_depth_registration_uses_nearest_z_buffer_value() -> None:
    color_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    depth_matrix = np.asarray(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    color = adapter.CameraCalibration(
        camera_id="color",
        frame="color",
        topic="/color",
        width=1,
        height=1,
        projection_model="pinhole",
        distortion_model="plumb_bob",
        matrix=color_matrix,
        distortion=np.zeros(5, dtype=np.float64),
    )
    depth = adapter.CameraCalibration(
        camera_id="depth",
        frame="depth",
        topic="/depth",
        width=2,
        height=1,
        projection_model="pinhole",
        distortion_model="plumb_bob",
        matrix=depth_matrix,
        distortion=np.zeros(5, dtype=np.float64),
    )
    calibration = adapter.HeadRgbdCalibration(
        color=color,
        depth=depth,
        rotation_color_depth=np.eye(3, dtype=np.float32),
        translation_color_depth=np.zeros(3, dtype=np.float32),
        static_transforms=(),
    )
    registrar = adapter.DepthToColorRegistrar(
        calibration,
        depth_min_m=0.1,
        depth_max_m=6.0,
    )
    registered = registrar.register_depth(
        np.asarray([[2.0, 1.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(registered, [[1.0]], atol=1.0e-6)
