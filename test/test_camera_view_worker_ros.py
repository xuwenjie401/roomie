#!/usr/bin/env python3
"""ROS integration regression for the current-camera worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import time


ROOMIE_ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOMIE_ROOT / "scripts" / "scene_qa" / "_ros_camera_view_worker.py"


def test_worker_processes_tf_while_lookup_is_waiting(monkeypatch) -> None:
    # Keep the synthetic map <- head_color transform away from a live robot ROS
    # graph, while still giving parent and worker the same DDS domain.
    domain_id = 150 + os.getpid() % 70
    monkeypatch.setenv("ROS_DOMAIN_ID", str(domain_id))
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")

    import rclpy
    from rclpy.qos import QoSProfile, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped

    rclpy.init(args=None)
    node = rclpy.create_node(f"camera_view_worker_test_{os.getpid()}")
    image_topic = f"/roomie/test/p{os.getpid()}/image"
    info_topic = f"/roomie/test/p{os.getpid()}/camera_info"
    image_pub = node.create_publisher(Image, image_topic, qos_profile_sensor_data)
    info_pub = node.create_publisher(CameraInfo, info_topic, qos_profile_sensor_data)
    tf_pub = node.create_publisher(TFMessage, "/tf", QoSProfile(depth=100))

    environment = os.environ.copy()
    process = subprocess.Popen(
        [
            "/usr/bin/python3",
            "-u",
            str(WORKER),
            image_topic,
            info_topic,
            "map",
            "head_color",
            "1.0",
            "0.2",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert process.stdin is not None
    assert process.stdout is not None

    try:
        discovery_deadline = time.monotonic() + 4.0
        while time.monotonic() < discovery_deadline:
            if node.get_subscriptions_info_by_topic(image_topic):
                break
            time.sleep(0.02)
        assert node.get_subscriptions_info_by_topic(image_topic)

        stamp = node.get_clock().now().to_msg()
        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = "head_color"
        image.width = 640
        image.height = 400
        camera_info = CameraInfo()
        camera_info.header = image.header
        camera_info.width = image.width
        camera_info.height = image.height
        camera_info.k = [305.0, 0.0, 318.0, 0.0, 305.0, 204.0, 0.0, 0.0, 1.0]

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "map"
        transform.child_frame_id = "head_color"
        transform.transform.translation.x = 1.0
        transform.transform.translation.y = 2.0
        transform.transform.translation.z = 3.0
        transform.transform.rotation.w = 1.0
        tf_message = TFMessage(transforms=[transform])

        # Prime the worker before any query, then simulate a bag pause longer
        # than the configured sensor timeout. This is the first-query-after-
        # pause case that requires the viewer to start the worker eagerly.
        camera_deadline = time.monotonic() + 0.5
        while time.monotonic() < camera_deadline:
            image_pub.publish(image)
            info_pub.publish(camera_info)
            tf_pub.publish(tf_message)
            time.sleep(0.02)
        time.sleep(1.1)

        process.stdin.write("{}\n")
        process.stdin.flush()
        readable, _, _ = select.select([process.stdout], [], [], 2.0)
        assert readable, "camera worker did not answer its first query from cache"
        primed = json.loads(process.stdout.readline())
        assert primed["ok"] is True, primed
        assert primed["view_mode"] == "cached_last_valid"
        assert primed["cached"] is True
        assert primed["cache_reason"] == "no_new_image_within_sensor_timeout"
        assert primed["cache_age_sec"] >= 1.0

        # Now request again and deliver a new image before its matching TF. The
        # executor must keep processing TF while lookup_transform is blocked.
        stamp = node.get_clock().now().to_msg()
        image.header.stamp = stamp
        camera_info.header.stamp = stamp
        transform.header.stamp = stamp
        process.stdin.write("{}\n")
        process.stdin.flush()
        camera_deadline = time.monotonic() + 0.3
        while time.monotonic() < camera_deadline:
            image_pub.publish(image)
            info_pub.publish(camera_info)
            time.sleep(0.02)
        time.sleep(0.1)

        tf_deadline = time.monotonic() + 0.3
        while time.monotonic() < tf_deadline:
            tf_pub.publish(tf_message)
            time.sleep(0.02)

        readable, _, _ = select.select([process.stdout], [], [], 2.0)
        assert readable, "camera worker did not answer after delayed TF arrived"
        response = json.loads(process.stdout.readline())
        assert response["ok"] is True, response
        assert response["tf_mode"] == "image_stamp"
        assert response["view_mode"] == "fresh"
        assert response["cached"] is False
        assert response["transform"]["world_frame"] == "map"
        assert response["transform"]["camera_frame"] == "head_color"
        assert response["transform"]["translation"] == [1.0, 2.0, 3.0]

        # With the publishers stopped, a later request represents a long bag
        # pause. It must return the last complete image/geometry/TF bundle.
        time.sleep(1.1)
        process.stdin.write("{}\n")
        process.stdin.flush()
        readable, _, _ = select.select([process.stdout], [], [], 2.0)
        assert readable, "camera worker did not answer from its last-valid cache"
        cached = json.loads(process.stdout.readline())
        assert cached["ok"] is True, cached
        assert cached["view_mode"] == "cached_last_valid"
        assert cached["cached"] is True
        assert cached["cache_reason"] == "no_new_image_within_sensor_timeout"
        assert cached["cache_age_sec"] >= 1.0
        assert cached["header"] == response["header"]
        assert cached["transform"] == response["transform"]

        process.stdin.close()
        assert process.wait(timeout=3.0) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
