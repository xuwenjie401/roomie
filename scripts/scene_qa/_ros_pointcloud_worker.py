#!/usr/bin/python3
"""Return bounded snapshots from a ROS PointCloud2 topic over JSON lines."""

from __future__ import annotations

import base64
import json
import math
import os
import struct
import sys
import time
from typing import Any


FLOAT32 = 7


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def sample_cloud(message: Any, max_points: int) -> dict[str, Any]:
    fields = {field.name: field for field in message.fields}
    for name in ("x", "y", "z"):
        field = fields.get(name)
        if field is None or field.datatype != FLOAT32:
            raise ValueError(f"PointCloud2 requires FLOAT32 field {name}")
    point_count = int(message.width) * int(message.height)
    if point_count <= 0:
        return {
            "ok": True,
            "count": 0,
            "source_count": 0,
            "frame_id": message.header.frame_id,
            "positions_b64": "",
            "colors_b64": "",
            "bounds": None,
        }
    stride = max(1, math.ceil(point_count / max_points))
    endian = ">" if message.is_bigendian else "<"
    positions = bytearray()
    colors = bytearray()
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    rgb_field = fields.get("rgb") or fields.get("rgba")
    raw = message.data
    accepted = 0
    for index in range(0, point_count, stride):
        if accepted >= max_points:
            break
        offset = index * int(message.point_step)
        x = struct.unpack_from(endian + "f", raw, offset + fields["x"].offset)[0]
        y = struct.unpack_from(endian + "f", raw, offset + fields["y"].offset)[0]
        z = struct.unpack_from(endian + "f", raw, offset + fields["z"].offset)[0]
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        # Browser payload is always little-endian, independent of ROS storage.
        positions.extend(struct.pack("<fff", x, y, z))
        if rgb_field is not None and rgb_field.datatype == FLOAT32:
            packed = struct.unpack_from(
                endian + "I", raw, offset + rgb_field.offset
            )[0]
            colors.extend(((packed >> 16) & 255, (packed >> 8) & 255, packed & 255))
        elif all(channel in fields for channel in ("r", "g", "b")):
            colors.extend(
                int(raw[offset + fields[channel].offset])
                for channel in ("r", "g", "b")
            )
        else:
            colors.extend((180, 188, 184))
        for axis, value in enumerate((x, y, z)):
            minimum[axis] = min(minimum[axis], value)
            maximum[axis] = max(maximum[axis], value)
        accepted += 1
    return {
        "ok": True,
        "count": accepted,
        "source_count": point_count,
        "frame_id": message.header.frame_id,
        "positions_b64": base64.b64encode(positions).decode("ascii"),
        "colors_b64": base64.b64encode(colors).decode("ascii"),
        "bounds": {"min": minimum, "max": maximum} if accepted else None,
    }


def main() -> int:
    if len(sys.argv) != 3:
        emit({"ok": False, "status": "worker_startup_error", "error": "invalid arguments"})
        return 2
    topic = sys.argv[1]
    try:
        default_timeout_sec = float(sys.argv[2])
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2
    except Exception as exc:
        emit(
            {
                "ok": False,
                "status": "worker_startup_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return 2

    rclpy.init(args=None)
    node = rclpy.create_node(f"roomie_scene_qa_points_{os.getpid()}")
    latest = {"message": None}
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    subscription = node.create_subscription(
        PointCloud2, topic, lambda message: latest.__setitem__("message", message), qos
    )
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                max_points = int(request.get("max_points", 120_000))
                if max_points <= 0 or max_points > 500_000:
                    raise ValueError("max_points must be between 1 and 500000")
                timeout_sec = float(request.get("wait_timeout_sec", default_timeout_sec))
                deadline = time.monotonic() + max(0.05, timeout_sec)
                latest["message"] = None
                while latest["message"] is None and time.monotonic() < deadline:
                    rclpy.spin_once(
                        node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic()))
                    )
                message = latest["message"]
                if message is None:
                    emit(
                        {
                            "ok": False,
                            "status": "point_cloud_timeout",
                            "error": f"no PointCloud2 received from {topic} within {timeout_sec:g}s",
                        }
                    )
                    continue
                emit(sample_cloud(message, max_points))
            except Exception as exc:
                emit(
                    {
                        "ok": False,
                        "status": "point_cloud_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
