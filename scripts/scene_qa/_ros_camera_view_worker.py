#!/usr/bin/python3
"""Sample the latest head image geometry and matching TF over JSON lines."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import threading
import time
from typing import Any


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def transform_payload(
    transform: Any,
    *,
    world_frame: str,
    camera_frame: str,
) -> dict[str, Any]:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    values = [
        translation.x,
        translation.y,
        translation.z,
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    ]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("TF contains a non-finite value")
    norm = math.sqrt(
        float(rotation.x) ** 2
        + float(rotation.y) ** 2
        + float(rotation.z) ** 2
        + float(rotation.w) ** 2
    )
    if norm <= 1.0e-9:
        raise ValueError("TF rotation quaternion is invalid")
    return {
        "world_frame": world_frame,
        "camera_frame": camera_frame,
        "translation": [
            float(translation.x),
            float(translation.y),
            float(translation.z),
        ],
        "rotation_xyzw": [
            float(rotation.x) / norm,
            float(rotation.y) / norm,
            float(rotation.z) / norm,
            float(rotation.w) / norm,
        ],
        "tf_stamp": {
            "sec": int(transform.header.stamp.sec),
            "nanosec": int(transform.header.stamp.nanosec),
        },
    }


def main() -> int:
    if len(sys.argv) != 7:
        emit(
            {
                "ok": False,
                "status": "camera_view_worker_startup_error",
                "error": "invalid arguments",
            }
        )
        return 2
    image_topic, camera_info_topic, world_frame, configured_camera_frame = sys.argv[1:5]
    try:
        timeout_sec = float(sys.argv[5])
        tf_tolerance_sec = float(sys.argv[6])
        import rclpy
        from rclpy.duration import Duration
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import CameraInfo, Image
        import tf2_ros
    except Exception as exc:
        emit(
            {
                "ok": False,
                "status": "camera_view_worker_startup_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return 2

    rclpy.init(args=None)
    node = rclpy.create_node(f"roomie_camera_view_{os.getpid()}")
    tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=max(2.0, timeout_sec + 1.0)))
    latest: dict[str, Any] = {
        "image": None,
        "camera_info": None,
        "image_sequence": 0,
        "image_received_monotonic": None,
    }
    last_valid: dict[str, Any] | None = None
    latest_condition = threading.Condition()

    def save_latest(name: str, message: Any) -> None:
        with latest_condition:
            latest[name] = message
            if name == "image":
                latest["image_sequence"] += 1
                latest["image_received_monotonic"] = time.monotonic()
            latest_condition.notify_all()

    def emit_last_valid(reason: str) -> bool:
        if last_valid is None:
            return False
        response = copy.deepcopy(last_valid["response"])
        response["view_mode"] = "cached_last_valid"
        response["cached"] = True
        response["cache_reason"] = reason
        response["cache_age_sec"] = round(
            max(
                0.0,
                time.monotonic() - last_valid["image_received_monotonic"],
            ),
            3,
        )
        emit(response)
        return True

    image_sub = node.create_subscription(
        Image,
        image_topic,
        lambda message: save_latest("image", message),
        qos_profile_sensor_data,
    )
    info_sub = node.create_subscription(
        CameraInfo,
        camera_info_topic,
        lambda message: save_latest("camera_info", message),
        qos_profile_sensor_data,
    )
    # This executor spins the whole node, including Image and CameraInfo.  TF
    # lookups below block the stdin/request thread, so relying on spin_once in
    # that thread would starve /tf callbacks while a lookup is waiting.
    tf_listener = tf2_ros.TransformListener(tf_buffer, node, spin_thread=True)

    try:
        for _line in sys.stdin:
            try:
                deadline = time.monotonic() + timeout_sec
                with latest_condition:
                    request_image_sequence = int(latest["image_sequence"])
                    while (
                        int(latest["image_sequence"]) <= request_image_sequence
                        or latest["camera_info"] is None
                    ):
                        remaining_sec = deadline - time.monotonic()
                        if remaining_sec <= 0.0:
                            break
                        latest_condition.wait(timeout=remaining_sec)
                    image = latest["image"]
                    camera_info = latest["camera_info"]
                    image_received_monotonic = latest["image_received_monotonic"]
                    fresh_image = (
                        int(latest["image_sequence"]) > request_image_sequence
                    )
                if image is None:
                    if emit_last_valid("no_image_received_since_worker_start"):
                        continue
                    emit(
                        {
                            "ok": False,
                            "status": "camera_image_timeout",
                            "error": f"no Image received from {image_topic} within {timeout_sec:g}s",
                        }
                    )
                    continue
                if camera_info is None:
                    if emit_last_valid("camera_info_unavailable"):
                        continue
                    emit(
                        {
                            "ok": False,
                            "status": "camera_info_timeout",
                            "error": (
                                f"no CameraInfo received from {camera_info_topic} "
                                f"within {timeout_sec:g}s"
                            ),
                        }
                    )
                    continue

                camera_frame = (
                    configured_camera_frame.strip()
                    or str(image.header.frame_id).strip()
                    or str(camera_info.header.frame_id).strip()
                )
                if not camera_frame:
                    raise ValueError("head image and configuration have no camera frame")
                image_time = Time.from_msg(image.header.stamp)
                tf_deadline = time.monotonic() + timeout_sec

                def remaining_tf_timeout() -> Duration:
                    return Duration(
                        seconds=max(0.01, tf_deadline - time.monotonic())
                    )

                try:
                    transform = tf_buffer.lookup_transform(
                        world_frame,
                        camera_frame,
                        image_time,
                        timeout=remaining_tf_timeout(),
                    )
                    tf_mode = "image_stamp"
                except Exception as exact_error:
                    try:
                        transform = tf_buffer.lookup_transform(
                            world_frame,
                            camera_frame,
                            Time(),
                            timeout=remaining_tf_timeout(),
                        )
                    except Exception as latest_error:
                        if emit_last_valid("camera_tf_unavailable_for_latest_image"):
                            continue
                        emit(
                            {
                                "ok": False,
                                "status": "camera_tf_unavailable",
                                "error": (
                                    f"could not resolve {world_frame} <- {camera_frame}: "
                                    f"exact={exact_error}; latest={latest_error}"
                                ),
                            }
                        )
                        continue
                    image_ns = stamp_ns(image.header.stamp)
                    tf_ns = stamp_ns(transform.header.stamp)
                    if (
                        image_ns > 0
                        and tf_ns > 0
                        and abs(image_ns - tf_ns)
                        > int(tf_tolerance_sec * 1_000_000_000)
                    ):
                        if emit_last_valid("camera_tf_stale_for_latest_image"):
                            continue
                        emit(
                            {
                                "ok": False,
                                "status": "camera_tf_stale",
                                "error": (
                                    f"latest TF differs from image stamp by "
                                    f"{abs(image_ns - tf_ns) / 1.0e9:.3f}s"
                                ),
                            }
                        )
                        continue
                    tf_mode = "latest_fallback"

                k = list(camera_info.k)
                width = int(image.width or camera_info.width)
                height = int(image.height or camera_info.height)
                if len(k) != 9 or width <= 0 or height <= 0:
                    raise ValueError("CameraInfo has invalid K, width, or height")
                view_mode = "fresh" if fresh_image else "cached_last_valid"
                result = {
                    "ok": True,
                    "header": {
                        "stamp": {
                            "sec": int(image.header.stamp.sec),
                            "nanosec": int(image.header.stamp.nanosec),
                        },
                        "frame_id": str(image.header.frame_id),
                    },
                    "camera": {
                        "width": width,
                        "height": height,
                        "fx": float(k[0]),
                        "fy": float(k[4]),
                        "cx": float(k[2]),
                        "cy": float(k[5]),
                    },
                    "transform": transform_payload(
                        transform,
                        world_frame=world_frame,
                        camera_frame=camera_frame,
                    ),
                    "tf_mode": tf_mode,
                    "view_mode": view_mode,
                    "cached": not fresh_image,
                    "cache_reason": (
                        None
                        if fresh_image
                        else "no_new_image_within_sensor_timeout"
                    ),
                    "cache_age_sec": round(
                        max(
                            0.0,
                            time.monotonic() - float(image_received_monotonic),
                        ),
                        3,
                    ),
                }
                last_valid = {
                    "response": copy.deepcopy(result),
                    "image_received_monotonic": float(image_received_monotonic),
                }
                emit(result)
            except Exception as exc:
                emit(
                    {
                        "ok": False,
                        "status": "camera_view_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        # Stop the listener executor before destroying subscriptions owned by
        # its node. TransformListener.__del__ repeats these operations safely.
        executor = getattr(tf_listener, "executor", None)
        listener_thread = getattr(tf_listener, "dedicated_listener_thread", None)
        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        if listener_thread is not None:
            listener_thread.join(timeout=1.0)
        node.destroy_subscription(image_sub)
        node.destroy_subscription(info_sub)
        tf_listener.unregister()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
