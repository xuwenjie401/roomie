#!/usr/bin/python3
"""Rectify and register AgiBot head RGB-D images for the Roomie pipeline."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import cv2
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster
import yaml


DEFAULT_CALIBRATION_FILE = (
    "/home/lindenbot/sensor_base/agibot_genie/"
    "roomie_base/calib/head_camera_params.yaml"
)


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    frame: str
    topic: str
    width: int
    height: int
    projection_model: str
    distortion_model: str
    matrix: np.ndarray
    distortion: np.ndarray


@dataclass(frozen=True)
class StaticTransform:
    parent_frame: str
    child_frame: str
    translation: np.ndarray
    quaternion_wxyz: np.ndarray


@dataclass(frozen=True)
class HeadRgbdCalibration:
    color: CameraCalibration
    depth: CameraCalibration
    rotation_color_depth: np.ndarray
    translation_color_depth: np.ndarray
    static_transforms: tuple[StaticTransform, ...]


def _finite_float(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _quaternion_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("rotation quaternion must have non-zero finite norm")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _camera_from_document(entry: dict[str, Any]) -> CameraCalibration:
    resolution = entry.get("resolution") or {}
    distortion = entry.get("distortion") or {}
    width = int(resolution.get("width", 0))
    height = int(resolution.get("height", 0))
    fx = _finite_float(entry.get("fx"), "camera fx")
    fy = _finite_float(entry.get("fy"), "camera fy")
    cx = _finite_float(entry.get("cx"), "camera cx")
    cy = _finite_float(entry.get("cy"), "camera cy")
    if width <= 0 or height <= 0 or fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"invalid camera calibration entry: {entry}")
    coefficients = np.asarray(
        [_finite_float(value, "distortion coefficient") for value in distortion.get("coefficients", [])],
        dtype=np.float64,
    )
    return CameraCalibration(
        camera_id=str(entry.get("camera_id") or ""),
        frame=str(entry.get("frame") or ""),
        topic=str(entry.get("topic") or ""),
        width=width,
        height=height,
        projection_model=str(entry.get("projection_model") or "pinhole"),
        distortion_model=str(distortion.get("model") or "plumb_bob"),
        matrix=np.asarray(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        distortion=coefficients,
    )


def _transform_from_document(entry: dict[str, Any]) -> StaticTransform:
    translation = entry.get("translation") or {}
    rotation = entry.get("rotation") or {}
    parent_frame = str(entry.get("parent_frame") or "")
    child_frame = str(entry.get("child_frame") or "")
    if not parent_frame or not child_frame:
        raise ValueError(f"static transform is missing frame names: {entry}")
    return StaticTransform(
        parent_frame=parent_frame,
        child_frame=child_frame,
        translation=np.asarray(
            [
                _finite_float(translation.get("x"), "translation x"),
                _finite_float(translation.get("y"), "translation y"),
                _finite_float(translation.get("z"), "translation z"),
            ],
            dtype=np.float64,
        ),
        quaternion_wxyz=np.asarray(
            [
                _finite_float(rotation.get("w"), "rotation w"),
                _finite_float(rotation.get("x"), "rotation x"),
                _finite_float(rotation.get("y"), "rotation y"),
                _finite_float(rotation.get("z"), "rotation z"),
            ],
            dtype=np.float64,
        ),
    )


def load_head_rgbd_calibration(
    path: str | Path,
    color_topic: str,
    depth_topic: str,
) -> HeadRgbdCalibration:
    calibration_path = Path(path).expanduser().resolve()
    with calibration_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("calibration YAML root must be a mapping")

    cameras = [
        _camera_from_document(entry)
        for entry in document.get("intrinsics", [])
        if isinstance(entry, dict)
    ]
    color = next((camera for camera in cameras if camera.topic == color_topic), None)
    depth = next((camera for camera in cameras if camera.topic == depth_topic), None)
    if color is None:
        raise ValueError(f"no intrinsic entry found for color topic {color_topic}")
    if depth is None:
        raise ValueError(f"no intrinsic entry found for depth topic {depth_topic}")
    if color.projection_model != "pinhole" or depth.projection_model != "pinhole":
        raise ValueError("only pinhole camera models are supported")

    static_transforms = tuple(
        _transform_from_document(entry)
        for entry in document.get("transforms", [])
        if isinstance(entry, dict)
    )
    direct = next(
        (
            transform
            for transform in static_transforms
            if transform.parent_frame == color.frame
            and transform.child_frame == depth.frame
        ),
        None,
    )
    inverse = next(
        (
            transform
            for transform in static_transforms
            if transform.parent_frame == depth.frame
            and transform.child_frame == color.frame
        ),
        None,
    )
    if direct is not None:
        rotation_color_depth = _quaternion_to_rotation(direct.quaternion_wxyz)
        translation_color_depth = direct.translation
    elif inverse is not None:
        rotation_depth_color = _quaternion_to_rotation(inverse.quaternion_wxyz)
        rotation_color_depth = rotation_depth_color.T
        translation_color_depth = -rotation_color_depth @ inverse.translation
    else:
        raise ValueError(
            f"calibration has no direct transform between {color.frame} and {depth.frame}"
        )

    return HeadRgbdCalibration(
        color=color,
        depth=depth,
        rotation_color_depth=rotation_color_depth.astype(np.float32),
        translation_color_depth=translation_color_depth.astype(np.float32),
        static_transforms=static_transforms,
    )


class DepthToColorRegistrar:
    """Register depth-camera Z values into a rectified color pinhole image."""

    def __init__(
        self,
        calibration: HeadRgbdCalibration,
        depth_min_m: float,
        depth_max_m: float,
    ) -> None:
        self.calibration = calibration
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.output_width = calibration.color.width
        self.output_height = calibration.color.height

        color_size = (calibration.color.width, calibration.color.height)
        self.color_map_x, self.color_map_y = cv2.initUndistortRectifyMap(
            calibration.color.matrix,
            calibration.color.distortion,
            np.eye(3, dtype=np.float64),
            calibration.color.matrix,
            color_size,
            cv2.CV_32FC1,
        )

        depth_u, depth_v = np.meshgrid(
            np.arange(calibration.depth.width, dtype=np.float32),
            np.arange(calibration.depth.height, dtype=np.float32),
        )
        depth_pixels = np.stack((depth_u, depth_v), axis=-1).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(
            depth_pixels,
            calibration.depth.matrix,
            calibration.depth.distortion,
        ).reshape(-1, 2)
        self.depth_ray_x = normalized[:, 0].astype(np.float32)
        self.depth_ray_y = normalized[:, 1].astype(np.float32)

        self.rotation = calibration.rotation_color_depth.astype(np.float32)
        self.translation = calibration.translation_color_depth.astype(np.float32)
        self.color_fx = float(calibration.color.matrix[0, 0])
        self.color_fy = float(calibration.color.matrix[1, 1])
        self.color_cx = float(calibration.color.matrix[0, 2])
        self.color_cy = float(calibration.color.matrix[1, 2])

    def rectify_color(self, color_bgr: np.ndarray) -> np.ndarray:
        expected_shape = (
            self.calibration.color.height,
            self.calibration.color.width,
        )
        if color_bgr.shape[:2] != expected_shape:
            raise ValueError(
                f"color image shape {color_bgr.shape[:2]} does not match "
                f"calibration {expected_shape}"
            )
        return cv2.remap(
            color_bgr,
            self.color_map_x,
            self.color_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def register_depth(self, depth_m: np.ndarray) -> np.ndarray:
        expected_shape = (
            self.calibration.depth.height,
            self.calibration.depth.width,
        )
        if depth_m.shape != expected_shape:
            raise ValueError(
                f"depth image shape {depth_m.shape} does not match "
                f"calibration {expected_shape}"
            )

        depth_flat = np.asarray(depth_m, dtype=np.float32).reshape(-1)
        valid = (
            np.isfinite(depth_flat)
            & (depth_flat >= self.depth_min_m)
            & (depth_flat <= self.depth_max_m)
        )
        valid_indices = np.flatnonzero(valid)
        registered = np.full(
            self.output_width * self.output_height,
            np.inf,
            dtype=np.float32,
        )
        if valid_indices.size == 0:
            registered.fill(0.0)
            return registered.reshape(self.output_height, self.output_width)

        depth_values = depth_flat[valid_indices]
        points_depth = np.column_stack(
            (
                self.depth_ray_x[valid_indices] * depth_values,
                self.depth_ray_y[valid_indices] * depth_values,
                depth_values,
            )
        )
        points_color = points_depth @ self.rotation.T + self.translation
        z_color = points_color[:, 2]
        projectable = np.isfinite(z_color) & (z_color > 0.0)
        points_color = points_color[projectable]
        z_color = z_color[projectable]
        if z_color.size == 0:
            registered.fill(0.0)
            return registered.reshape(self.output_height, self.output_width)

        pixel_u = np.rint(
            self.color_fx * points_color[:, 0] / z_color + self.color_cx
        ).astype(np.int32)
        pixel_v = np.rint(
            self.color_fy * points_color[:, 1] / z_color + self.color_cy
        ).astype(np.int32)
        inside = (
            (pixel_u >= 0)
            & (pixel_u < self.output_width)
            & (pixel_v >= 0)
            & (pixel_v < self.output_height)
        )
        linear = pixel_v[inside] * self.output_width + pixel_u[inside]
        np.minimum.at(registered, linear, z_color[inside])
        registered[~np.isfinite(registered)] = 0.0
        return registered.reshape(self.output_height, self.output_width)


def _stamp_nanoseconds(message: Image) -> int:
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def _color_message_to_bgr(message: Image) -> np.ndarray:
    channels_by_encoding = {
        "mono8": 1,
        "8UC1": 1,
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
    }
    channels = channels_by_encoding.get(message.encoding)
    if channels is None:
        raise ValueError(f"unsupported color encoding {message.encoding!r}")
    row_bytes = int(message.width) * channels
    if int(message.step) < row_bytes:
        raise ValueError("color image step is smaller than one compact row")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected = int(message.step) * int(message.height)
    if raw.size < expected:
        raise ValueError("color image data is truncated")
    compact = raw[:expected].reshape(int(message.height), int(message.step))[:, :row_bytes]
    image = compact.reshape(int(message.height), int(message.width), channels)
    if message.encoding in ("mono8", "8UC1"):
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if message.encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if message.encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if message.encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(image)


def _depth_message_to_meters(message: Image, depth_scale: float) -> np.ndarray:
    if message.encoding == "16UC1":
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
    elif message.encoding == "32FC1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    else:
        raise ValueError(f"unsupported depth encoding {message.encoding!r}")
    row_values = int(message.step) // dtype.itemsize
    if row_values < int(message.width):
        raise ValueError("depth image step is smaller than one compact row")
    raw = np.frombuffer(message.data, dtype=dtype)
    expected = row_values * int(message.height)
    if raw.size < expected:
        raise ValueError("depth image data is truncated")
    compact = raw[:expected].reshape(int(message.height), row_values)[:, : int(message.width)]
    depth_m = compact.astype(np.float32)
    if message.encoding == "16UC1":
        depth_m *= float(depth_scale)
    return depth_m


def _image_message(
    array: np.ndarray,
    encoding: str,
    frame_id: str,
    stamp: Any,
) -> Image:
    compact = np.ascontiguousarray(array)
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height = int(compact.shape[0])
    message.width = int(compact.shape[1])
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = int(compact.strides[0])
    message.data = compact.tobytes()
    return message


def _camera_info_message(
    calibration: CameraCalibration,
    frame_id: str,
    stamp: Any,
) -> CameraInfo:
    message = CameraInfo()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.width = calibration.width
    message.height = calibration.height
    message.distortion_model = "plumb_bob"
    message.d = [0.0] * 5
    matrix = calibration.matrix
    message.k = [
        float(matrix[0, 0]),
        0.0,
        float(matrix[0, 2]),
        0.0,
        float(matrix[1, 1]),
        float(matrix[1, 2]),
        0.0,
        0.0,
        1.0,
    ]
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


class AgibotHeadRgbdAdapter(Node):
    def __init__(self) -> None:
        super().__init__("roomie_agibot_head_rgbd_adapter")
        calibration_file = str(
            self.declare_parameter("calibration_file", DEFAULT_CALIBRATION_FILE).value
        )
        color_topic = str(
            self.declare_parameter("color_topic", "/gdk/camera/head_color").value
        )
        depth_topic = str(
            self.declare_parameter("depth_topic", "/gdk/camera/head_depth").value
        )
        self.output_color_topic = str(
            self.declare_parameter(
                "output_color_topic",
                "/roomie/input/head_color/image_rect",
            ).value
        )
        self.output_depth_topic = str(
            self.declare_parameter(
                "output_depth_topic",
                "/roomie/input/head_color/depth_registered",
            ).value
        )
        self.output_camera_info_topic = str(
            self.declare_parameter(
                "output_camera_info_topic",
                "/roomie/input/head_color/camera_info",
            ).value
        )
        self.depth_scale = float(
            self.declare_parameter("input_depth_scale", 0.001).value
        )
        depth_min_m = float(self.declare_parameter("depth_min_m", 0.1).value)
        depth_max_m = float(self.declare_parameter("depth_max_m", 6.0).value)
        self.sync_tolerance_ns = int(
            max(0.0, float(self.declare_parameter("sync_tolerance_sec", 0.002).value))
            * 1_000_000_000
        )
        self.sync_queue_size = max(
            2,
            int(self.declare_parameter("sync_queue_size", 60).value),
        )
        publish_static_tf = bool(
            self.declare_parameter("publish_static_tf", False).value
        )
        status_period_sec = max(
            0.5,
            float(self.declare_parameter("status_period_sec", 5.0).value),
        )

        self.calibration = load_head_rgbd_calibration(
            calibration_file,
            color_topic,
            depth_topic,
        )
        configured_output_frame = str(
            self.declare_parameter("output_frame", self.calibration.color.frame).value
        )
        self.output_frame = configured_output_frame or self.calibration.color.frame
        self.registrar = DepthToColorRegistrar(
            self.calibration,
            depth_min_m,
            depth_max_m,
        )

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.sync_queue_size,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.color_publisher = self.create_publisher(
            Image,
            self.output_color_topic,
            sensor_qos,
        )
        self.depth_publisher = self.create_publisher(
            Image,
            self.output_depth_topic,
            sensor_qos,
        )
        self.camera_info_publisher = self.create_publisher(
            CameraInfo,
            self.output_camera_info_topic,
            sensor_qos,
        )
        self.color_subscription = self.create_subscription(
            Image,
            color_topic,
            self._handle_color,
            sensor_qos,
        )
        self.depth_subscription = self.create_subscription(
            Image,
            depth_topic,
            self._handle_depth,
            sensor_qos,
        )

        self.color_buffer: OrderedDict[int, Image] = OrderedDict()
        self.depth_buffer: OrderedDict[int, Image] = OrderedDict()
        self.emitted_stamps: set[int] = set()
        self.emitted_stamp_order: deque[int] = deque()
        self.color_messages = 0
        self.depth_messages = 0
        self.matched_pairs = 0
        self.processing_errors = 0
        self.last_processing_ms = 0.0

        self.static_broadcaster = None
        if publish_static_tf:
            self.static_broadcaster = StaticTransformBroadcaster(self)
            self.static_broadcaster.sendTransform(self._static_transform_messages())

        self.status_timer = self.create_timer(status_period_sec, self._log_status)
        self.get_logger().info(
            "AgiBot head RGB-D adapter ready: "
            f"{color_topic} + {depth_topic} -> {self.output_color_topic} + "
            f"{self.output_depth_topic}; frame={self.output_frame}; "
            f"sync_tolerance={self.sync_tolerance_ns / 1e6:.3f}ms; "
            "sensor_qos=BEST_EFFORT"
        )

    def _static_transform_messages(self) -> list[TransformStamped]:
        stamp = self.get_clock().now().to_msg()
        messages = []
        for transform in self.calibration.static_transforms:
            message = TransformStamped()
            message.header.stamp = stamp
            message.header.frame_id = transform.parent_frame
            message.child_frame_id = transform.child_frame
            message.transform.translation.x = float(transform.translation[0])
            message.transform.translation.y = float(transform.translation[1])
            message.transform.translation.z = float(transform.translation[2])
            w, x, y, z = transform.quaternion_wxyz
            message.transform.rotation.w = float(w)
            message.transform.rotation.x = float(x)
            message.transform.rotation.y = float(y)
            message.transform.rotation.z = float(z)
            messages.append(message)
        return messages

    def _handle_color(self, message: Image) -> None:
        self.color_messages += 1
        self._insert_and_match(message, self.color_buffer, self.depth_buffer)

    def _handle_depth(self, message: Image) -> None:
        self.depth_messages += 1
        self._insert_and_match(message, self.depth_buffer, self.color_buffer)

    def _insert_and_match(
        self,
        message: Image,
        own_buffer: OrderedDict[int, Image],
        other_buffer: OrderedDict[int, Image],
    ) -> None:
        stamp_ns = _stamp_nanoseconds(message)
        if stamp_ns <= 0 or stamp_ns in self.emitted_stamps:
            return
        own_buffer[stamp_ns] = message
        own_buffer.move_to_end(stamp_ns)
        while len(own_buffer) > self.sync_queue_size:
            own_buffer.popitem(last=False)

        match_stamp = self._nearest_stamp(stamp_ns, other_buffer)
        if match_stamp is None:
            return
        own = own_buffer.pop(stamp_ns, None)
        other = other_buffer.pop(match_stamp, None)
        if own is None or other is None:
            return
        if own is message and own_buffer is self.color_buffer:
            color_message, depth_message = own, other
        elif own is message and own_buffer is self.depth_buffer:
            color_message, depth_message = other, own
        elif own_buffer is self.color_buffer:
            color_message, depth_message = own, other
        else:
            color_message, depth_message = other, own
        output_stamp_ns = _stamp_nanoseconds(depth_message)
        if output_stamp_ns in self.emitted_stamps:
            return
        self._remember_emitted_stamp(output_stamp_ns)
        self._process_pair(color_message, depth_message)

    def _nearest_stamp(
        self,
        stamp_ns: int,
        candidates: OrderedDict[int, Image],
    ) -> int | None:
        if stamp_ns in candidates:
            return stamp_ns
        if self.sync_tolerance_ns <= 0 or not candidates:
            return None
        nearest = min(candidates, key=lambda candidate: abs(candidate - stamp_ns))
        if abs(nearest - stamp_ns) <= self.sync_tolerance_ns:
            return nearest
        return None

    def _remember_emitted_stamp(self, stamp_ns: int) -> None:
        self.emitted_stamps.add(stamp_ns)
        self.emitted_stamp_order.append(stamp_ns)
        while len(self.emitted_stamp_order) > self.sync_queue_size * 4:
            expired = self.emitted_stamp_order.popleft()
            self.emitted_stamps.discard(expired)

    def _process_pair(self, color_message: Image, depth_message: Image) -> None:
        started = time.perf_counter()
        try:
            color_bgr = _color_message_to_bgr(color_message)
            depth_m = _depth_message_to_meters(depth_message, self.depth_scale)
            rectified_color = self.registrar.rectify_color(color_bgr)
            registered_depth = self.registrar.register_depth(depth_m)
            stamp = depth_message.header.stamp

            camera_info = _camera_info_message(
                self.calibration.color,
                self.output_frame,
                stamp,
            )
            color_output = _image_message(
                rectified_color,
                "bgr8",
                self.output_frame,
                stamp,
            )
            depth_output = _image_message(
                registered_depth.astype(np.float32, copy=False),
                "32FC1",
                self.output_frame,
                stamp,
            )
            self.camera_info_publisher.publish(camera_info)
            self.color_publisher.publish(color_output)
            self.depth_publisher.publish(depth_output)
            self.matched_pairs += 1
        except Exception as error:
            self.processing_errors += 1
            self.get_logger().error(f"failed to register RGB-D pair: {error}")
        self.last_processing_ms = (time.perf_counter() - started) * 1000.0

    def _log_status(self) -> None:
        self.get_logger().info(
            "status "
            f"color={self.color_messages} depth={self.depth_messages} "
            f"matched={self.matched_pairs} errors={self.processing_errors} "
            f"buffer_color={len(self.color_buffer)} "
            f"buffer_depth={len(self.depth_buffer)} "
            f"last_processing={self.last_processing_ms:.1f}ms"
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = AgibotHeadRgbdAdapter()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
