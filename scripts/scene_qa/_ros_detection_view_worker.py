#!/usr/bin/python3
"""Cache the latest retained Roomie 2D detection outputs over JSON lines."""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from typing import Any


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def header_payload(header: Any) -> dict[str, Any]:
    return {
        "stamp": {
            "sec": int(header.stamp.sec),
            "nanosec": int(header.stamp.nanosec),
        },
        "frame_id": str(header.frame_id),
    }


def stamp_ns(header: dict[str, Any] | None) -> int | None:
    if not isinstance(header, dict) or not isinstance(header.get("stamp"), dict):
        return None
    stamp = header["stamp"]
    try:
        return int(stamp.get("sec", 0)) * 1_000_000_000 + int(
            stamp.get("nanosec", 0)
        )
    except (TypeError, ValueError):
        return None


def encode_image(message: Any, cv2: Any, np: Any) -> tuple[str, str]:
    encodings = {
        "rgb8": (3, cv2.COLOR_RGB2BGR),
        "bgr8": (3, None),
        "rgba8": (4, cv2.COLOR_RGBA2BGRA),
        "bgra8": (4, None),
        "mono8": (1, None),
    }
    encoding = str(message.encoding).lower()
    if encoding not in encodings:
        raise ValueError(f"unsupported detection image encoding: {message.encoding}")
    channels, color_conversion = encodings[encoding]
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    if width <= 0 or height <= 0 or step < width * channels:
        raise ValueError("detection image has invalid width, height, or step")
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError("detection image payload is shorter than height * step")
    packed = raw[: height * step].reshape(height, step)[:, : width * channels]
    image = packed.reshape(height, width) if channels == 1 else packed.reshape(
        height, width, channels
    )
    if color_conversion is not None:
        image = cv2.cvtColor(image, color_conversion)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode detection image as PNG")
    return "image/png", base64.b64encode(encoded.tobytes()).decode("ascii")


def main() -> int:
    if len(sys.argv) != 4:
        emit(
            {
                "ok": False,
                "status": "detection_view_worker_startup_error",
                "error": "invalid arguments",
            }
        )
        return 2
    try:
        topics = json.loads(sys.argv[1])
        timeout_sec = float(sys.argv[2])
        camera_ids = json.loads(sys.argv[3])
        if not isinstance(topics, dict) or not isinstance(camera_ids, list):
            raise ValueError("topics must be an object and camera_ids must be an array")
        import cv2
        import numpy as np
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
    except Exception as exc:
        emit(
            {
                "ok": False,
                "status": "detection_view_worker_startup_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return 2

    rclpy.init(args=None)
    node = rclpy.create_node(f"roomie_detection_view_{os.getpid()}")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    latest: dict[str, dict[str, Any]] = {
        str(camera_id): {"image": None, "result": None, "result_error": ""}
        for camera_id in camera_ids
    }
    subscriptions = []

    def handle_image(camera_id: str, message: Any) -> None:
        latest[camera_id]["image"] = message

    def handle_result(camera_id: str, message: Any) -> None:
        try:
            value = json.loads(str(message.data))
            if not isinstance(value, dict):
                raise ValueError("2D detection result is not a JSON object")
            latest[camera_id]["result"] = value
            latest[camera_id]["result_error"] = ""
        except Exception as exc:
            latest[camera_id]["result_error"] = f"{type(exc).__name__}: {exc}"

    for raw_camera_id in camera_ids:
        camera_id = str(raw_camera_id)
        camera_topics = topics.get(camera_id)
        if not isinstance(camera_topics, dict):
            continue
        image_topic = str(camera_topics.get("image") or "")
        result_topic = str(camera_topics.get("result") or "")
        if image_topic:
            subscriptions.append(
                node.create_subscription(
                    Image,
                    image_topic,
                    lambda message, camera_id=camera_id: handle_image(camera_id, message),
                    qos,
                )
            )
        if result_topic:
            subscriptions.append(
                node.create_subscription(
                    String,
                    result_topic,
                    lambda message, camera_id=camera_id: handle_result(camera_id, message),
                    qos,
                )
            )

    try:
        for _line in sys.stdin:
            try:
                deadline = time.monotonic() + timeout_sec
                while time.monotonic() < deadline:
                    if all(
                        state["image"] is not None and state["result"] is not None
                        for state in latest.values()
                    ):
                        break
                    rclpy.spin_once(
                        node,
                        timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())),
                    )

                cameras = []
                for camera_id in camera_ids:
                    camera_id = str(camera_id)
                    state = latest[camera_id]
                    image = state["image"]
                    result = state["result"]
                    image_error = ""
                    mime_type = ""
                    image_base64 = ""
                    if image is not None:
                        try:
                            mime_type, image_base64 = encode_image(image, cv2, np)
                        except Exception as exc:
                            image_error = f"{type(exc).__name__}: {exc}"
                    image_header = header_payload(image.header) if image is not None else None
                    result_header = (
                        result.get("header") if isinstance(result, dict) else None
                    )
                    synchronized = (
                        stamp_ns(image_header) is not None
                        and stamp_ns(image_header) == stamp_ns(result_header)
                    )
                    cameras.append(
                        {
                            "camera_id": camera_id,
                            "available": image is not None or result is not None,
                            "image_available": bool(image_base64),
                            "result_available": isinstance(result, dict),
                            "synchronized": synchronized,
                            "image_header": image_header,
                            "result": result,
                            "image_mime_type": mime_type,
                            "image_base64": image_base64,
                            "image_error": image_error,
                            "result_error": state["result_error"],
                        }
                    )
                emit({"ok": True, "cameras": cameras})
            except Exception as exc:
                emit(
                    {
                        "ok": False,
                        "status": "detection_view_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
