#!/usr/bin/python3
"""Python-3.10 ROS service bridge for the Gemini environment.

ROS Humble's rclpy extension is tied to the system Python ABI, while the
Gemini environment may use a newer Python. This worker keeps that boundary to
newline-delimited JSON on stdin/stdout.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    if len(sys.argv) != 3:
        emit({"ok": False, "status": "worker_startup_error", "error": "invalid arguments"})
        return 2
    service_name = sys.argv[1]
    try:
        timeout_sec = float(sys.argv[2])
        import rclpy
        from roomie.srv import QueryScene
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
    node = rclpy.create_node(f"roomie_scene_qa_bridge_{os.getpid()}")
    client = node.create_client(QueryScene, service_name)
    try:
        for line in sys.stdin:
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                if not client.wait_for_service(timeout_sec=timeout_sec):
                    emit(
                        {
                            "ok": False,
                            "status": "service_unavailable",
                            "error": (
                                f"service {service_name} was not available within "
                                f"{timeout_sec:g}s"
                            ),
                        }
                    )
                    continue
                request = QueryScene.Request()
                request.request_json = json.dumps(payload, ensure_ascii=False)
                future = client.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
                if not future.done():
                    emit(
                        {
                            "ok": False,
                            "status": "service_timeout",
                            "error": (
                                f"service {service_name} did not respond within "
                                f"{timeout_sec:g}s"
                            ),
                        }
                    )
                    continue
                response = future.result()
                if response is None:
                    raise RuntimeError(str(future.exception() or "service call failed"))
                emit(
                    {
                        "ok": True,
                        "success": bool(response.success),
                        "response_json": response.response_json,
                        "error": response.error,
                    }
                )
            except Exception as exc:
                emit(
                    {
                        "ok": False,
                        "status": "transport_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
