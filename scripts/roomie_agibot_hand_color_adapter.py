#!/usr/bin/python3
"""Rectify AgiBot hand-color streams for Roomie's pinhole detector input."""

from __future__ import annotations

import array
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import yaml


@dataclass(frozen=True)
class HandCameraCalibration:
    camera_id: str
    input_topic: str
    output_topic: str
    camera_info_topic: str
    frame: str
    mount_frame: str
    width: int
    height: int
    matrix: np.ndarray
    source_distortion_model: str
    source_distortion: np.ndarray
    translation: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


@dataclass
class CameraRuntime:
    calibration: HandCameraCalibration
    map_x: np.ndarray
    map_y: np.ndarray
    image_publisher: Any
    camera_info_publisher: Any
    last_output_stamp_ns: int | None = None
    received: int = 0
    published: int = 0
    disabled_skips: int = 0
    rate_skips: int = 0
    failures: int = 0
    last_processing_ms: float = 0.0


def _finite_float(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _finite_vector(value: Any, size: int, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{context} must contain {size} values")
    return tuple(_finite_float(component, context) for component in value)


def load_hand_camera_calibrations(
    path: str | Path,
    camera_ids: list[str],
    input_topics: list[str] | None = None,
) -> list[HandCameraCalibration]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cameras = document.get("cameras", {}) if isinstance(document, dict) else {}
    if not isinstance(cameras, dict):
        raise ValueError("camera config must contain a cameras mapping")
    if input_topics and len(input_topics) != len(camera_ids):
        raise ValueError("input_topics must be empty or match camera_ids")

    result: list[HandCameraCalibration] = []
    seen: set[str] = set()
    for index, camera_id in enumerate(camera_ids):
        if not camera_id or camera_id in seen:
            raise ValueError("camera_ids must be non-empty and unique")
        seen.add(camera_id)
        entry = cameras.get(camera_id)
        if not isinstance(entry, dict):
            raise ValueError(f"camera config has no camera {camera_id!r}")

        resolution = entry.get("resolution")
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ValueError(f"camera {camera_id!r} has invalid resolution")
        width, height = (int(resolution[0]), int(resolution[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"camera {camera_id!r} has invalid resolution")

        intrinsics = entry.get("intrinsics", {})
        fx = _finite_float(intrinsics.get("fx"), f"{camera_id}.fx")
        fy = _finite_float(intrinsics.get("fy"), f"{camera_id}.fy")
        cx = _finite_float(intrinsics.get("cx"), f"{camera_id}.cx")
        cy = _finite_float(intrinsics.get("cy"), f"{camera_id}.cy")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError(f"camera {camera_id!r} has invalid focal lengths")
        matrix = np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        source = entry.get("source", {})
        source_distortion = source.get("distortion", {})
        distortion_model = str(source_distortion.get("model") or "")
        coefficients = _finite_vector(
            source_distortion.get("coefficients"),
            5,
            f"{camera_id}.source.distortion.coefficients",
        )
        if distortion_model != "plumb_bob":
            raise ValueError(
                f"camera {camera_id!r} source distortion must be plumb_bob"
            )

        transform = entry.get("T_mount_camera", {})
        translation = _finite_vector(
            transform.get("translation"), 3, f"{camera_id}.translation"
        )
        quaternion = _finite_vector(
            transform.get("quaternion_wxyz"), 4, f"{camera_id}.quaternion"
        )
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        if quaternion_norm <= 1.0e-12:
            raise ValueError(f"camera {camera_id!r} has invalid quaternion")
        quaternion = tuple(value / quaternion_norm for value in quaternion)

        configured_input_topic = str(source.get("topic") or "")
        input_topic = (
            input_topics[index]
            if input_topics
            else configured_input_topic
        )
        output_topic = str(entry.get("topic") or "")
        camera_info_topic = str(entry.get("camera_info_topic") or "")
        frame = str(entry.get("frame") or "")
        mount_frame = str(entry.get("mount_frame") or "")
        if not all(
            (input_topic, output_topic, camera_info_topic, frame, mount_frame)
        ):
            raise ValueError(f"camera {camera_id!r} has incomplete ROS metadata")

        result.append(
            HandCameraCalibration(
                camera_id=camera_id,
                input_topic=input_topic,
                output_topic=output_topic,
                camera_info_topic=camera_info_topic,
                frame=frame,
                mount_frame=mount_frame,
                width=width,
                height=height,
                matrix=matrix,
                source_distortion_model=distortion_model,
                source_distortion=np.asarray(coefficients, dtype=np.float64),
                translation=translation,  # type: ignore[arg-type]
                quaternion_wxyz=quaternion,  # type: ignore[arg-type]
            )
        )
    return result


def _color_message_to_bgr(message: Image) -> np.ndarray:
    channels_by_encoding = {
        "mono8": 1,
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
    }
    channels = channels_by_encoding.get(message.encoding)
    if channels is None:
        raise ValueError(f"unsupported color encoding {message.encoding!r}")
    compact_step = int(message.width) * channels
    if int(message.step) < compact_step:
        raise ValueError("color image step is smaller than one compact row")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = int(message.step) * int(message.height)
    if raw.size < required:
        raise ValueError("color image data is truncated")
    rows = raw[:required].reshape(int(message.height), int(message.step))
    image = rows[:, :compact_step].reshape(
        int(message.height), int(message.width), channels
    )
    if message.encoding == "mono8":
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if message.encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if message.encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if message.encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(image)


def _image_message(image: np.ndarray, frame: str, stamp: Any) -> Image:
    contiguous = np.ascontiguousarray(image, dtype=np.uint8)
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = frame
    message.height = int(contiguous.shape[0])
    message.width = int(contiguous.shape[1])
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = int(contiguous.strides[0])
    message.data = array.array("B", contiguous.tobytes())
    return message


def _camera_info_message(
    calibration: HandCameraCalibration, stamp: Any
) -> CameraInfo:
    message = CameraInfo()
    message.header.stamp = stamp
    message.header.frame_id = calibration.frame
    message.width = calibration.width
    message.height = calibration.height
    message.distortion_model = "plumb_bob"
    message.d = [0.0] * 5
    matrix = calibration.matrix
    message.k = [float(value) for value in matrix.reshape(-1)]
    message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    message.p = [
        float(matrix[0, 0]),
        0.0,
        float(matrix[0, 2]),
        0.0,
        0.0,
        float(matrix[1, 1]),
        float(matrix[1, 2]),
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    return message


def _static_transform(calibration: HandCameraCalibration) -> TransformStamped:
    message = TransformStamped()
    message.header.frame_id = calibration.mount_frame
    message.child_frame_id = calibration.frame
    tx, ty, tz = calibration.translation
    qw, qx, qy, qz = calibration.quaternion_wxyz
    message.transform.translation.x = tx
    message.transform.translation.y = ty
    message.transform.translation.z = tz
    message.transform.rotation.w = qw
    message.transform.rotation.x = qx
    message.transform.rotation.y = qy
    message.transform.rotation.z = qz
    return message


class AgibotHandColorAdapter(Node):
    def __init__(self) -> None:
        super().__init__("roomie_agibot_hand_color_adapter")
        camera_config = str(self.declare_parameter("camera_config", "").value)
        enable_topic = str(
            self.declare_parameter(
                "enable_topic", "/roomie/hand_cameras/enabled"
            ).value
        )
        if not enable_topic:
            raise ValueError("enable_topic must not be empty")
        self.enabled = False
        self.enable_updates = 0
        camera_ids = [
            str(value)
            for value in self.declare_parameter(
                "camera_ids", ["hand_left_color", "hand_right_color"]
            ).value
        ]
        topic_overrides = {
            "hand_left_color": str(
                self.declare_parameter("hand_left_input_topic", "").value
            ),
            "hand_right_color": str(
                self.declare_parameter("hand_right_input_topic", "").value
            ),
        }
        input_topics = [topic_overrides.get(camera_id, "") for camera_id in camera_ids]
        use_topic_overrides = any(input_topics)
        if use_topic_overrides and not all(input_topics):
            raise ValueError(
                "hand input topic overrides must be provided for every camera id"
            )
        self.max_fps = max(
            0.0, float(self.declare_parameter("max_fps", 5.0).value)
        )
        self.min_period_ns = (
            int(round(1.0e9 / self.max_fps)) if self.max_fps > 0.0 else 0
        )
        publish_static_tf = bool(
            self.declare_parameter("publish_static_tf", True).value
        )
        log_period_sec = max(
            0.1, float(self.declare_parameter("log_period_sec", 5.0).value)
        )
        if not camera_config:
            raise ValueError("camera_config must not be empty")
        calibrations = load_hand_camera_calibrations(
            camera_config,
            camera_ids,
            input_topics if use_topic_overrides else None,
        )

        qos = QoSProfile(depth=2)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE
        enable_qos = QoSProfile(depth=1)
        enable_qos.reliability = ReliabilityPolicy.RELIABLE
        enable_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.enable_subscription = self.create_subscription(
            Bool, enable_topic, self._handle_enable, enable_qos
        )
        self.runtimes: dict[str, CameraRuntime] = {}
        self.image_subscriptions: list[Any] = []
        for calibration in calibrations:
            map_x, map_y = cv2.initUndistortRectifyMap(
                calibration.matrix,
                calibration.source_distortion,
                None,
                calibration.matrix,
                (calibration.width, calibration.height),
                cv2.CV_32FC1,
            )
            runtime = CameraRuntime(
                calibration=calibration,
                map_x=map_x,
                map_y=map_y,
                image_publisher=self.create_publisher(
                    Image, calibration.output_topic, qos
                ),
                camera_info_publisher=self.create_publisher(
                    CameraInfo, calibration.camera_info_topic, qos
                ),
            )
            self.runtimes[calibration.camera_id] = runtime
            self.image_subscriptions.append(
                self.create_subscription(
                    Image,
                    calibration.input_topic,
                    lambda message, camera_id=calibration.camera_id: self._handle_image(
                        camera_id, message
                    ),
                    qos,
                )
            )

        self.static_tf_broadcaster: StaticTransformBroadcaster | None = None
        if publish_static_tf:
            self.static_tf_broadcaster = StaticTransformBroadcaster(self)
            transforms = [_static_transform(item) for item in calibrations]
            now = self.get_clock().now().to_msg()
            for transform in transforms:
                transform.header.stamp = now
            self.static_tf_broadcaster.sendTransform(transforms)

        self.status_timer = self.create_timer(log_period_sec, self._log_status)
        routes = ", ".join(
            f"{item.input_topic}->{item.output_topic}" for item in calibrations
        )
        self.get_logger().info(
            f"configured {len(calibrations)} hand camera(s): {routes}; "
            f"max_fps={self.max_fps:.2f} enable_topic={enable_topic} "
            f"publish_static_tf={publish_static_tf}"
        )

    def _handle_enable(self, message: Bool) -> None:
        previous = self.enabled
        self.enabled = bool(message.data)
        self.enable_updates += 1
        if self.enabled != previous:
            self.get_logger().info(
                f"hand camera processing {'enabled' if self.enabled else 'disabled'}"
            )

    def _handle_image(self, camera_id: str, message: Image) -> None:
        runtime = self.runtimes[camera_id]
        runtime.received += 1
        if not self.enabled:
            runtime.disabled_skips += 1
            return
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        admission_stamp_ns = stamp_ns if stamp_ns > 0 else time.monotonic_ns()
        if runtime.last_output_stamp_ns is not None and self.min_period_ns > 0:
            delta_ns = admission_stamp_ns - runtime.last_output_stamp_ns
            jitter_tolerance_ns = min(5_000_000, self.min_period_ns // 50)
            if 0 < delta_ns + jitter_tolerance_ns < self.min_period_ns:
                runtime.rate_skips += 1
                return

        started = time.perf_counter()
        try:
            calibration = runtime.calibration
            if (
                int(message.width) != calibration.width
                or int(message.height) != calibration.height
            ):
                raise ValueError(
                    f"image is {message.width}x{message.height}, expected "
                    f"{calibration.width}x{calibration.height}"
                )
            source = _color_message_to_bgr(message)
            rectified = cv2.remap(
                source,
                runtime.map_x,
                runtime.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            runtime.camera_info_publisher.publish(
                _camera_info_message(calibration, message.header.stamp)
            )
            runtime.image_publisher.publish(
                _image_message(rectified, calibration.frame, message.header.stamp)
            )
            runtime.last_output_stamp_ns = admission_stamp_ns
            runtime.published += 1
        except Exception as error:  # noqa: BLE001 - ROS callback boundary
            runtime.failures += 1
            self.get_logger().error(f"{camera_id} frame rejected: {error}")
        finally:
            runtime.last_processing_ms = (time.perf_counter() - started) * 1.0e3

    def _log_status(self) -> None:
        details = " ".join(
            f"{camera_id}={{recv={runtime.received} pub={runtime.published} "
            f"disabled_skip={runtime.disabled_skips} "
            f"rate_skip={runtime.rate_skips} fail={runtime.failures} "
            f"last_ms={runtime.last_processing_ms:.1f}}}"
            for camera_id, runtime in self.runtimes.items()
        )
        self.get_logger().info(
            f"status enabled={self.enabled} enable_updates={self.enable_updates} "
            f"{details}"
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: AgibotHandColorAdapter | None = None
    try:
        node = AgibotHandColorAdapter()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
