#!/usr/bin/env python3
"""ROS integration regression for retained three-camera 2D detections."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import select
import subprocess
import time


ROOMIE_ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOMIE_ROOT / "scripts" / "scene_qa" / "_ros_detection_view_worker.py"
CAMERA_IDS = ("head_color", "hand_left_color", "hand_right_color")


def test_worker_reads_results_published_before_late_subscription(monkeypatch) -> None:
    domain_id = 150 + os.getpid() % 70
    monkeypatch.setenv("ROS_DOMAIN_ID", str(domain_id))
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")

    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    rclpy.init(args=None)
    node = rclpy.create_node(f"detection_view_worker_test_{os.getpid()}")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    prefix = f"/roomie/test/p{os.getpid()}"
    topics = {}
    publishers = []
    stamp = node.get_clock().now().to_msg()
    for index, camera_id in enumerate(CAMERA_IDS):
        image_topic = f"{prefix}/image/{camera_id}"
        result_topic = f"{prefix}/result/{camera_id}"
        topics[camera_id] = {"image": image_topic, "result": result_topic}
        image_pub = node.create_publisher(Image, image_topic, qos)
        result_pub = node.create_publisher(String, result_topic, qos)
        publishers.extend([image_pub, result_pub])

        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = camera_id
        image.width = 2
        image.height = 1
        image.encoding = "rgb8"
        image.step = 6
        image.data = [255, 0, 0, 0, 255, index]
        image_pub.publish(image)
        result_pub.publish(
            String(
                data=json.dumps(
                    {
                        "schema_version": 1,
                        "header": {
                            "stamp": {
                                "sec": int(stamp.sec),
                                "nanosec": int(stamp.nanosec),
                            },
                            "frame_id": camera_id,
                        },
                        "camera_id": camera_id,
                        "ok": True,
                        "error": "",
                        "detections": [
                            {
                                "label": f"object-{index}",
                                "semantic_id": index,
                                "score_2d": 0.9,
                                "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                            }
                        ],
                    },
                    separators=(",", ":"),
                )
            )
        )

    process = subprocess.Popen(
        [
            "/usr/bin/python3",
            "-u",
            str(WORKER),
            json.dumps(topics, separators=(",", ":")),
            "2.0",
            json.dumps(CAMERA_IDS, separators=(",", ":")),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    assert process.stdin is not None
    assert process.stdout is not None

    try:
        process.stdin.write("{}\n")
        process.stdin.flush()
        readable, _, _ = select.select([process.stdout], [], [], 4.0)
        assert readable, "2D detection worker did not answer"
        response = json.loads(process.stdout.readline())
        assert response["ok"] is True, response
        assert [entry["camera_id"] for entry in response["cameras"]] == list(
            CAMERA_IDS
        )
        for index, entry in enumerate(response["cameras"]):
            assert entry["available"]
            assert entry["image_available"]
            assert entry["result_available"]
            assert entry["synchronized"]
            assert entry["result"]["detections"][0]["label"] == f"object-{index}"
            assert base64.b64decode(entry["image_base64"]).startswith(b"\x89PNG")

        process.stdin.close()
        assert process.wait(timeout=3.0) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3.0)
        for publisher in publishers:
            node.destroy_publisher(publisher)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
