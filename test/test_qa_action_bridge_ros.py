#!/usr/bin/env python3
"""ROS integration test for the Web-to-action-to-QA round trip."""

from __future__ import annotations

from pathlib import Path
import sys


ROOMIE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOMIE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scene_qa.action_bridge import RosQaActionBridge  # noqa: E402


def test_three_action_round_trips_preserve_web_envelopes(monkeypatch) -> None:
    monkeypatch.setenv("ROS_DOMAIN_ID", "181")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    seen = []

    def execute(message):
        seen.append(message)
        task = message["task"]
        if task == "scene_qa":
            answer = "桌上有一个黄色瓶子。"
        elif task == "navigation":
            answer = {
                "status": 0,
                "reason": "found desk",
                "target_furniture": {
                    "furniture_id": 3,
                    "name": "study desk",
                    "label": "desk",
                    "description": "wooden desk",
                    "position": {
                        "header": {
                            "stamp": {"sec": 0, "nanosec": 0},
                            "frame_id": "map",
                        },
                        "point": {"x": 1.0, "y": 2.0, "z": 0.4},
                    },
                    "size": {"x": 1.2, "y": 0.7, "z": 0.8},
                },
                "exploration_furniture": [],
                "follow_up_question": "",
                "vlm_description": "目标在书桌上。",
            }
        else:
            answer = {
                "status": 1,
                "reason": "not in view",
                "target_object": None,
                "target_object_in_head_cam": None,
                "suspected_objects": [],
                "suspected_objects_in_head_cam": [],
                "vlm_description": "当前没有可信候选。",
            }
        body = {
            "reasoning": "test reasoning",
            "answer": answer,
            "raw_text": "{}",
            "history": {"task": task, "request_transport": "ros_action"},
            "highlight_groups": [],
            "provider": message["provider"],
            "task": task,
            "model": "test-model",
        }
        return {
            "body": body,
            "structured_result": answer if isinstance(answer, dict) else {},
        }

    bridge = RosQaActionBridge(
        execute_callback=execute,
        cancel_callback=lambda _message: None,
        action_prefix="/roomie/test_qa_actions",
        server_timeout_sec=5.0,
    )
    try:
        scene = bridge.request(
            task="scene_qa",
            query="桌上有什么",
            provider="gemini",
            allow_follow_up_question=True,
        )
        navigation = bridge.request(
            task="navigation",
            query="去书桌",
            provider="doubao",
            allow_follow_up_question=False,
        )
        in_view = bridge.request(
            task="find_object_in_view",
            query="找黄色瓶子",
            provider="gemini",
            allow_follow_up_question=True,
        )
    finally:
        bridge.close()

    assert scene["answer"] == "桌上有一个黄色瓶子。"
    assert navigation["answer"]["target_furniture"]["furniture_id"] == 3
    assert navigation["provider"] == "doubao"
    assert not in_view["answer"]["suspected_objects"]
    assert [message["task"] for message in seen] == [
        "scene_qa",
        "navigation",
        "find_object_in_view",
    ]
    assert seen[1]["allow_follow_up_question"] is False
