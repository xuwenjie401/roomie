"""Latest retained 2D detector results for VLM inspection."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any


DETECTION_CAMERA_IDS = (
    "head_color",
    "hand_left_color",
    "hand_right_color",
)


class DetectionViewError(RuntimeError):
    """Structured failure of the ROS latest-detection bridge."""

    def __init__(self, status: str, message: str, payload: dict[str, Any] | None = None):
        self.status = status
        self.payload = payload or {}
        super().__init__(f"{status}: {message}" if status else message)


@dataclass(frozen=True)
class Latest2dDetectionFrame:
    camera_id: str
    available: bool
    image_available: bool
    result_available: bool
    synchronized: bool
    image_header: dict[str, Any] | None
    result: dict[str, Any] | None
    image_data: bytes | None
    image_mime_type: str
    image_error: str
    result_error: str

    @classmethod
    def from_mapping(cls, value: Any) -> "Latest2dDetectionFrame":
        if not isinstance(value, dict):
            raise ValueError("latest 2D detection camera entry must be an object")
        encoded = str(value.get("image_base64") or "")
        try:
            image_data = base64.b64decode(encoded, validate=True) if encoded else None
        except (binascii.Error, ValueError) as exc:
            raise ValueError("latest 2D detection image is not valid base64") from exc
        result = value.get("result")
        return cls(
            camera_id=str(value.get("camera_id") or ""),
            available=bool(value.get("available")),
            image_available=bool(value.get("image_available")) and image_data is not None,
            result_available=bool(value.get("result_available"))
            and isinstance(result, dict),
            synchronized=bool(value.get("synchronized")),
            image_header=(
                dict(value["image_header"])
                if isinstance(value.get("image_header"), dict)
                else None
            ),
            result=dict(result) if isinstance(result, dict) else None,
            image_data=image_data,
            image_mime_type=str(value.get("image_mime_type") or "image/png"),
            image_error=str(value.get("image_error") or ""),
            result_error=str(value.get("result_error") or ""),
        )

    def metadata(self) -> dict[str, Any]:
        result = self.result or {}
        detections = result.get("detections")
        if not isinstance(detections, list):
            detections = []
        return {
            "camera_id": self.camera_id,
            "available": self.available,
            "image_available": self.image_available,
            "result_available": self.result_available,
            "image_and_result_synchronized": self.synchronized,
            "header": result.get("header") or self.image_header,
            "image_header": self.image_header,
            "result_header": result.get("header"),
            "detector_ok": result.get("ok") if self.result_available else None,
            "detector_error": result.get("error") if self.result_available else None,
            "detection_count": len(detections),
            "detections": detections,
            "image_error": self.image_error or None,
            "result_error": self.result_error or None,
        }


def detection_topics(
    image_base_topic: str,
    result_base_topic: str,
) -> dict[str, dict[str, str]]:
    def topic(base: str, camera_id: str) -> str:
        if camera_id == "head_color":
            return base.rstrip("/")
        return f"{base.rstrip('/')}/{camera_id}"

    return {
        camera_id: {
            "image": topic(image_base_topic, camera_id),
            "result": topic(result_base_topic, camera_id),
        }
        for camera_id in DETECTION_CAMERA_IDS
    }


class RosLatest2dDetectionsSampler:
    """Read each camera's transient-local latest 2D detector result."""

    def __init__(
        self,
        *,
        image_base_topic: str = "/roomie/detections_2d_image",
        result_base_topic: str = "/roomie/detections_2d",
        timeout_sec: float = 1.0,
    ):
        if not image_base_topic or not result_base_topic:
            raise ValueError("2D detection base topics must be non-empty")
        if timeout_sec <= 0.0:
            raise ValueError("2D detection timeout must be positive")
        self.topics = detection_topics(image_base_topic, result_base_topic)
        self.timeout_sec = float(timeout_sec)
        self._lock = threading.Lock()
        worker_path = Path(__file__).with_name("_ros_detection_view_worker.py")
        ros_python = os.environ.get("ROOMIE_ROS_PYTHON", "/usr/bin/python3")
        try:
            self._worker = subprocess.Popen(
                [
                    ros_python,
                    str(worker_path),
                    json.dumps(self.topics, separators=(",", ":")),
                    str(self.timeout_sec),
                    json.dumps(DETECTION_CAMERA_IDS, separators=(",", ":")),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise DetectionViewError(
                "detection_view_worker_startup_error",
                f"could not start ROS 2D detection worker: {exc}",
            ) from exc

    def sample(self) -> list[Latest2dDetectionFrame]:
        with self._lock:
            worker = self._worker
            if worker is None or worker.stdin is None or worker.stdout is None:
                raise DetectionViewError(
                    "detection_view_unavailable", "2D detection worker is closed"
                )
            if worker.poll() is not None:
                raise DetectionViewError(
                    "detection_view_unavailable", self._worker_exit_message(worker)
                )
            try:
                worker.stdin.write("{}\n")
                worker.stdin.flush()
                line = worker.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                raise DetectionViewError("detection_view_unavailable", str(exc)) from exc
            if not line:
                raise DetectionViewError(
                    "detection_view_unavailable", self._worker_exit_message(worker)
                )
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DetectionViewError(
                    "detection_view_invalid_response",
                    "ROS 2D detection worker returned invalid JSON",
                ) from exc
            if not isinstance(response, dict) or not response.get("ok"):
                status = (
                    str(response.get("status") or "detection_view_unavailable")
                    if isinstance(response, dict)
                    else "detection_view_unavailable"
                )
                message = (
                    str(response.get("error") or "2D detection sampling failed")
                    if isinstance(response, dict)
                    else "2D detection sampling failed"
                )
                raise DetectionViewError(
                    status,
                    message,
                    response if isinstance(response, dict) else None,
                )
            cameras = response.get("cameras")
            if not isinstance(cameras, list):
                raise DetectionViewError(
                    "detection_view_invalid_response",
                    "ROS 2D detection worker omitted cameras",
                    response,
                )
            frames = [Latest2dDetectionFrame.from_mapping(value) for value in cameras]
            returned_ids = {frame.camera_id for frame in frames}
            if (
                len(frames) != len(DETECTION_CAMERA_IDS)
                or returned_ids != set(DETECTION_CAMERA_IDS)
            ):
                raise DetectionViewError(
                    "detection_view_invalid_response",
                    "ROS 2D detection worker returned an unexpected camera set",
                    response,
                )
            return frames

    @staticmethod
    def _worker_exit_message(worker: subprocess.Popen[str]) -> str:
        detail = ""
        if worker.poll() is not None and worker.stderr is not None:
            try:
                detail = worker.stderr.read().strip()
            except OSError:
                detail = ""
        message = f"ROS 2D detection worker exited with code {worker.poll()}"
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

    def __enter__(self) -> "RosLatest2dDetectionsSampler":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
