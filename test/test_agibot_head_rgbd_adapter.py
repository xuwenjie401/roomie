#!/usr/bin/env python3
"""Tests for the AgiBot head RGB-D registration adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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


def test_roomie_owns_robot_mask_generation_contract() -> None:
    adapter_params = yaml.safe_load(
        (ROOMIE_ROOT / "config" / "agibot_head_rgbd_adapter.yaml").read_text(
            encoding="utf-8"
        )
    )["roomie_agibot_head_rgbd_adapter"]["ros__parameters"]
    assert "output_mask_topic" not in adapter_params
    assert "publish_zero_robot_mask" not in adapter_params

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
