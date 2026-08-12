"""Current head-camera sampling and 3D OBB frustum projection.

The VLM task uses only camera geometry and the image header.  RGB pixels are
deliberately not returned: "in view" means that at least one 3D bounding-box
corner projects into the current head-camera frustum, not that the object is
strictly visible or unoccluded in the current image.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import threading
from typing import Any


class CameraViewError(RuntimeError):
    """Structured current-camera acquisition failure."""

    def __init__(self, status: str, message: str, payload: dict[str, Any] | None = None):
        self.status = status
        self.payload = payload or {}
        super().__init__(f"{status}: {message}" if status else message)


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vec(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    return tuple(_finite(item, name) for item in value)


@dataclass(frozen=True)
class CameraViewSample:
    header: dict[str, Any]
    world_frame: str
    camera_frame: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    translation_world_camera: tuple[float, float, float]
    rotation_world_camera_xyzw: tuple[float, float, float, float]
    tf_mode: str = "unknown"
    view_mode: str = "fresh"
    cached: bool = False
    cache_reason: str | None = None
    cache_age_sec: float = 0.0

    @classmethod
    def from_mapping(cls, value: Any) -> "CameraViewSample":
        if not isinstance(value, dict):
            raise ValueError("camera sample must be an object")
        header = value.get("header")
        camera = value.get("camera")
        transform = value.get("transform")
        if not isinstance(header, dict):
            raise ValueError("camera sample is missing header")
        if not isinstance(camera, dict):
            raise ValueError("camera sample is missing camera geometry")
        if not isinstance(transform, dict):
            raise ValueError("camera sample is missing transform")
        width = int(camera.get("width", 0))
        height = int(camera.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("camera width and height must be positive")
        fx = _finite(camera.get("fx"), "fx")
        fy = _finite(camera.get("fy"), "fy")
        cx = _finite(camera.get("cx"), "cx")
        cy = _finite(camera.get("cy"), "cy")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        translation = _vec(transform.get("translation"), 3, "translation")
        rotation = _vec(transform.get("rotation_xyzw"), 4, "rotation_xyzw")
        norm = math.sqrt(sum(component * component for component in rotation))
        if norm <= 1.0e-9:
            raise ValueError("camera rotation quaternion is invalid")
        normalized_rotation = tuple(component / norm for component in rotation)
        camera_frame = str(
            transform.get("camera_frame") or header.get("frame_id") or ""
        ).strip()
        world_frame = str(transform.get("world_frame") or "").strip()
        if not camera_frame or not world_frame:
            raise ValueError("camera and world frames must be non-empty")
        return cls(
            header=dict(header),
            world_frame=world_frame,
            camera_frame=camera_frame,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            translation_world_camera=translation,  # type: ignore[arg-type]
            rotation_world_camera_xyzw=normalized_rotation,  # type: ignore[arg-type]
            tf_mode=str(value.get("tf_mode") or "unknown"),
            view_mode=str(value.get("view_mode") or "fresh"),
            cached=bool(value.get("cached", False)),
            cache_reason=(
                str(value["cache_reason"])
                if value.get("cache_reason") is not None
                else None
            ),
            cache_age_sec=max(
                0.0,
                _finite(value.get("cache_age_sec", 0.0), "cache_age_sec"),
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "header": dict(self.header),
            "world_frame": self.world_frame,
            "camera_frame": self.camera_frame,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "tf_mode": self.tf_mode,
            "view_mode": self.view_mode,
            "cached": self.cached,
            "cache_reason": self.cache_reason,
            "cache_age_sec": self.cache_age_sec,
        }


@dataclass(frozen=True)
class ObjectProjection:
    in_view: bool
    bbox: dict[str, Any] | None
    visible_corner_count: int
    positive_depth_corner_count: int
    min_corner_depth_m: float | None
    max_corner_depth_m: float | None

    def to_dict(self, header: dict[str, Any]) -> dict[str, Any]:
        return {
            "in_view": self.in_view,
            "object_in_head_cam": (
                {"header": dict(header), "bbox": dict(self.bbox or {})}
                if self.in_view and self.bbox is not None
                else None
            ),
            "visible_corner_count": self.visible_corner_count,
            "positive_depth_corner_count": self.positive_depth_corner_count,
            "min_corner_depth_m": self.min_corner_depth_m,
            "max_corner_depth_m": self.max_corner_depth_m,
            "visibility_definition": "at_least_one_3d_bbox_corner_in_frustum",
        }


def _rotate_by_quaternion(
    point: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Rotate a vector by a normalized quaternion."""

    x, y, z, w = quaternion_xyzw
    px, py, pz = point
    # q * p * q^-1, expanded to avoid a numpy dependency in this module.
    tx = 2.0 * (y * pz - z * py)
    ty = 2.0 * (z * px - x * pz)
    tz = 2.0 * (x * py - y * px)
    return (
        px + w * tx + (y * tz - z * ty),
        py + w * ty + (z * tx - x * tz),
        pz + w * tz + (x * ty - y * tx),
    )


def _world_to_camera(
    point_world: tuple[float, float, float], sample: CameraViewSample
) -> tuple[float, float, float]:
    delta = tuple(
        point_world[index] - sample.translation_world_camera[index]
        for index in range(3)
    )
    x, y, z, w = sample.rotation_world_camera_xyzw
    return _rotate_by_quaternion(delta, (-x, -y, -z, w))


def project_yaw_obb_to_camera(
    center_world: Any,
    size_m: Any,
    yaw_rad: Any,
    sample: CameraViewSample,
    *,
    min_depth_m: float = 0.1,
    max_depth_m: float = 6.0,
) -> ObjectProjection:
    """Project a world-frame yaw OBB using the task's corner-in-frustum rule."""

    center = _vec(center_world, 3, "center_world")
    size = _vec(size_m, 3, "size_m")
    yaw = _finite(yaw_rad if yaw_rad is not None else 0.0, "yaw_rad")
    min_depth = _finite(min_depth_m, "min_depth_m")
    max_depth = _finite(max_depth_m, "max_depth_m")
    if min_depth <= 0.0 or max_depth <= min_depth:
        raise ValueError("depth range must satisfy 0 < min_depth_m < max_depth_m")
    if any(dimension <= 0.0 for dimension in size):
        return ObjectProjection(False, None, 0, 0, None, None)

    half = tuple(dimension * 0.5 for dimension in size)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    camera_corners: list[tuple[float, float, float]] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local_x = sx * half[0]
                local_y = sy * half[1]
                corner_world = (
                    center[0] + cosine * local_x - sine * local_y,
                    center[1] + sine * local_x + cosine * local_y,
                    center[2] + sz * half[2],
                )
                camera_corners.append(_world_to_camera(corner_world, sample))

    projected_positive: list[tuple[float, float]] = []
    visible = 0
    positive_depths: list[float] = []
    for x, y, z in camera_corners:
        if z <= 1.0e-6:
            continue
        positive_depths.append(z)
        u = sample.fx * x / z + sample.cx
        v = sample.fy * y / z + sample.cy
        if math.isfinite(u) and math.isfinite(v):
            projected_positive.append((u, v))
            if (
                min_depth <= z <= max_depth
                and 0.0 <= u < sample.width
                and 0.0 <= v < sample.height
            ):
                visible += 1

    if visible == 0 or not projected_positive:
        return ObjectProjection(
            False,
            None,
            visible,
            len(positive_depths),
            min(positive_depths) if positive_depths else None,
            max(positive_depths) if positive_depths else None,
        )

    min_u = max(0.0, min(point[0] for point in projected_positive))
    min_v = max(0.0, min(point[1] for point in projected_positive))
    max_u = min(float(sample.width), max(point[0] for point in projected_positive))
    max_v = min(float(sample.height), max(point[1] for point in projected_positive))
    left = min(sample.width - 1, max(0, int(math.floor(min_u))))
    top = min(sample.height - 1, max(0, int(math.floor(min_v))))
    right = min(sample.width, max(left + 1, int(math.ceil(max_u))))
    bottom = min(sample.height, max(top + 1, int(math.ceil(max_v))))
    bbox = {
        "x_offset": left,
        "y_offset": top,
        "height": bottom - top,
        "width": right - left,
        "do_rectify": False,
    }
    return ObjectProjection(
        True,
        bbox,
        visible,
        len(positive_depths),
        min(positive_depths),
        max(positive_depths),
    )


class RosCameraViewSampler:
    """Persistent system-Python bridge for one current head-camera sample."""

    def __init__(
        self,
        *,
        image_topic: str = "/roomie/input/head_color/image_rect",
        camera_info_topic: str = "/roomie/input/head_color/camera_info",
        world_frame: str = "map",
        camera_frame: str = "head_color",
        timeout_sec: float = 1.0,
        tf_tolerance_sec: float = 0.2,
    ):
        if not image_topic or not camera_info_topic:
            raise ValueError("head-camera topics must be non-empty")
        if not world_frame:
            raise ValueError("world_frame must be non-empty")
        if timeout_sec <= 0.0 or tf_tolerance_sec < 0.0:
            raise ValueError("camera timeout must be positive and TF tolerance non-negative")
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.world_frame = world_frame
        self.camera_frame = camera_frame
        self.timeout_sec = float(timeout_sec)
        self.tf_tolerance_sec = float(tf_tolerance_sec)
        self._lock = threading.Lock()
        worker_path = Path(__file__).with_name("_ros_camera_view_worker.py")
        ros_python = os.environ.get("ROOMIE_ROS_PYTHON", "/usr/bin/python3")
        try:
            self._worker = subprocess.Popen(
                [
                    ros_python,
                    str(worker_path),
                    image_topic,
                    camera_info_topic,
                    world_frame,
                    camera_frame,
                    str(self.timeout_sec),
                    str(self.tf_tolerance_sec),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise CameraViewError(
                "camera_view_worker_startup_error",
                f"could not start ROS camera-view worker: {exc}",
            ) from exc

    def sample(self) -> CameraViewSample:
        with self._lock:
            worker = self._worker
            if worker is None or worker.stdin is None or worker.stdout is None:
                raise CameraViewError("camera_view_unavailable", "camera worker is closed")
            if worker.poll() is not None:
                raise CameraViewError(
                    "camera_view_unavailable", self._worker_exit_message(worker)
                )
            try:
                worker.stdin.write("{}\n")
                worker.stdin.flush()
                line = worker.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                raise CameraViewError("camera_view_unavailable", str(exc)) from exc
            if not line:
                raise CameraViewError(
                    "camera_view_unavailable", self._worker_exit_message(worker)
                )
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CameraViewError(
                    "camera_view_invalid_response",
                    "ROS camera worker returned invalid JSON",
                ) from exc
            if not isinstance(response, dict) or not response.get("ok"):
                status = (
                    str(response.get("status") or "camera_view_unavailable")
                    if isinstance(response, dict)
                    else "camera_view_unavailable"
                )
                message = (
                    str(response.get("error") or "camera sampling failed")
                    if isinstance(response, dict)
                    else "camera sampling failed"
                )
                raise CameraViewError(
                    status,
                    message,
                    response if isinstance(response, dict) else None,
                )
            try:
                return CameraViewSample.from_mapping(response)
            except ValueError as exc:
                raise CameraViewError(
                    "camera_view_invalid_response", str(exc), response
                ) from exc

    @staticmethod
    def _worker_exit_message(worker: subprocess.Popen[str]) -> str:
        detail = ""
        if worker.poll() is not None and worker.stderr is not None:
            try:
                detail = worker.stderr.read().strip()
            except OSError:
                detail = ""
        message = f"ROS camera worker exited with code {worker.poll()}"
        return f"{message}: {detail}" if detail else message

    def close(self) -> None:
        with self._lock:
            worker = self._worker
            self._worker = None
        if worker is None:
            return
        if worker.stdin is not None:
            try:
                worker.stdin.close()
            except OSError:
                pass
        try:
            worker.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:
                worker.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=1.0)

    def __enter__(self) -> "RosCameraViewSampler":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
