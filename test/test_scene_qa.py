#!/usr/bin/env python3
"""Offline tests for the Roomie scene QA tool layer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOMIE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOMIE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scene_qa.config import SceneQaConfig  # noqa: E402
from scene_qa.embeddings import ObjectSearchIndex  # noqa: E402
from scene_qa.gemini_agent import GeminiSceneQaAgent, sanitize_proxy_environment  # noqa: E402
from scene_qa.graph_store import GraphStore  # noqa: E402
from scene_qa.tools import create_default_tool_registry  # noqa: E402
from roomie_scene_qa_viewer import highlight_groups_from_history  # noqa: E402


def write_fixture(root: Path) -> Path:
    snapshot_dir = root / "snapshots"
    snapshot_dir.mkdir()
    image_path = snapshot_dir / "snapshot_0000.bmp"
    Image.new("RGB", (64, 64), (220, 220, 220)).save(image_path)
    graph = {
        "format": "roomie_manual_scene_graph",
        "world_frame": "world",
        "objects": [
            {
                "object_id": 1,
                "label": "yellow_bottle",
                "description": "A small yellow bottle with a white cap.",
                "center_world": [1.0, 0.0, 0.5],
                "size_m": [0.1, 0.1, 0.2],
                "yaw_rad": 0.0,
                "parent_room_ids": [10],
                "confidence": 0.9,
                "object_quality_score": 0.8,
                "geometry_status": "good",
                "first_seen_ns": 1000000000,
                "last_seen_ns": 3000000000,
                "snapshot": {
                    "image_index": 0,
                    "bbox_xyxy": [10, 12, 32, 48],
                    "camera_id": "test_camera",
                    "quality": 0.7,
                    "time_ns": 1000000000,
                },
                "near_surface_voxels": [{"large": "omitted"}],
                "observation_timestamps_ns": [1, 2, 3],
            },
            {
                "object_id": 2,
                "label": "sink",
                "description": "A metal kitchen sink.",
                "center_world": [1.4, 0.0, 0.4],
                "size_m": [0.6, 0.5, 0.2],
                "yaw_rad": 0.0,
                "parent_room_ids": [10],
                "confidence": 0.8,
                "object_quality_score": 0.9,
                "geometry_status": "good",
                "first_seen_ns": 1000000000,
                "last_seen_ns": 4000000000,
            },
            {
                "object_id": 3,
                "label": "bed",
                "description": "A bed in the bedroom.",
                "center_world": [5.0, 0.0, 0.5],
                "size_m": [2.0, 1.4, 0.7],
                "yaw_rad": 0.0,
                "parent_room_ids": [11],
                "confidence": 0.8,
                "object_quality_score": 0.9,
                "geometry_status": "good",
                "first_seen_ns": 1000000000,
                "last_seen_ns": 4000000000,
            },
        ],
        "rooms": [
            {
                "room_id": 10,
                "label": "kitchen",
                "center_world": [1.0, 0.0, 1.0],
                "size_m": [3.0, 3.0, 2.0],
                "min_xy": [0.0, -1.0],
                "max_xy": [2.0, 1.0],
            },
            {
                "room_id": 11,
                "label": "bedroom",
                "center_world": [5.0, 0.0, 1.0],
                "size_m": [3.0, 3.0, 2.0],
                "min_xy": [4.0, -1.0],
                "max_xy": [6.0, 1.0],
            },
        ],
        "relations": [
            {
                "relation_type": "room_contains_object",
                "source": {"type": "room", "id": 10},
                "target": {"type": "object", "id": 1},
                "confidence": 1.0,
            }
        ],
        "snapshot_images": [
            {
                "image_index": 0,
                "camera_id": "test_camera",
                "encoding": "bmp",
                "width": 64,
                "height": 64,
                "uri": "snapshots/snapshot_0000.bmp",
                "source_path": str(image_path),
            }
        ],
    }
    path = root / "scene.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    return path


class SceneQaToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.json_path = write_fixture(Path(self.tmp.name))
        self.graph = GraphStore.load(self.json_path)
        self.config = SceneQaConfig(
            graph_json=self.json_path,
            embedding_backend="lexical",
            top_k=5,
        )
        self.search = ObjectSearchIndex(
            self.graph,
            model_path=self.config.embedding_model,
            backend="lexical",
        )
        self.registry = create_default_tool_registry(self.graph, self.search, self.config)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_graph_store_indexes_lightweight_scene(self) -> None:
        self.assertEqual(self.graph.object_count, 3)
        self.assertEqual(self.graph.room_count, 2)
        bottle = self.graph.get_object(1)
        self.assertEqual(bottle.room_ids, (10,))
        raw = self.graph.sanitized_raw_object(1)
        self.assertNotIn("near_surface_voxels", raw)
        self.assertNotIn("observation_timestamps_ns", raw)
        self.assertTrue(self.graph.snapshot_metadata(bottle)["image_available"])

    def test_scene_qa_config_loads_json(self) -> None:
        prompt_dir = Path(self.tmp.name) / "prompts"
        prompt_dir.mkdir()
        prompt_path = prompt_dir / "system.txt"
        prompt_path.write_text("test system prompt", encoding="utf-8")
        config_path = Path(self.tmp.name) / "scene_qa.json"
        config_path.write_text(
            json.dumps(
                {
                    "graph_json": str(self.json_path),
                    "gemini_model": "gemini-test-model",
                    "embedding_model": "/tmp/fake_embedding",
                    "embedding_backend": "lexical",
                    "device": "cpu",
                    "top_k": 3,
                    "max_iterations": 4,
                    "temperature": 0.1,
                    "max_output_tokens": 128,
                    "snapshot_max_side_px": 320,
                    "snapshot_bbox_pad_px": 5,
                    "system_prompt_path": "prompts/system.txt",
                }
            ),
            encoding="utf-8",
        )
        config = SceneQaConfig.from_json(config_path)
        self.assertEqual(config.resolved_graph_json(), self.json_path.resolve())
        self.assertEqual(config.gemini_model, "gemini-test-model")
        self.assertEqual(config.embedding_backend, "lexical")
        self.assertEqual(config.top_k, 3)
        self.assertEqual(config.snapshot_bbox_pad_px, 5)
        self.assertEqual(config.resolved_system_prompt_path(), prompt_path.resolve())
        self.assertEqual(config.load_system_prompt(), "test system prompt")

    def test_tool_registry_search_room_and_nearby(self) -> None:
        result = self.registry.call_tool(
            "search_objects",
            {"description": "yellow bottle", "top_k": 2},
        )
        self.assertEqual(result.response["objects"][0]["object_id"], 1)

        room_result = self.registry.call_tool(
            "get_objects_in_room",
            {"room_id": 10, "description": "sink", "top_k": 2},
        )
        ids = [obj["object_id"] for obj in room_result.response["objects"]]
        self.assertIn(2, ids)

        nearby = self.registry.call_tool(
            "get_objects_near",
            {"object_id": 1, "radius_m": 0.6, "top_k": 5},
        )
        nearby_ids = [obj["object_id"] for obj in nearby.response["objects"]]
        self.assertEqual(nearby_ids, [1, 2])

    def test_inspect_snapshot_returns_media_attachment(self) -> None:
        result = self.registry.call_tool(
            "inspect_snapshot",
            {"object_id": 1, "crop": True, "draw_bbox": True},
        )
        self.assertTrue(result.response["image_attached"])
        self.assertEqual(len(result.media), 1)
        self.assertEqual(result.media[0].mime_type, "image/jpeg")

    def test_gemini_loop_executes_local_tool_with_fake_client(self) -> None:
        class FakeCall:
            name = "search_objects"
            args = {"description": "yellow bottle", "top_k": 1}

        class FakeResponse:
            def __init__(self, text: str = "", function_calls: list | None = None):
                self.text = text
                self.function_calls = function_calls or []
                self.candidates = []
                self.parts = []

        class FakeModels:
            def __init__(self):
                self.calls = []
                self.responses = [
                    FakeResponse(function_calls=[FakeCall()]),
                    FakeResponse(
                        '{"reasoning": "searched objects", "answer": "object 1 is the yellow bottle"}'
                    ),
                ]

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return self.responses.pop(0)

        class FakeClient:
            def __init__(self):
                self.models = FakeModels()

        fake_client = FakeClient()
        progress_events = []
        agent = GeminiSceneQaAgent(
            self.graph,
            self.registry,
            self.config,
            client=fake_client,
            progress_callback=progress_events.append,
        )
        response = agent.answer_query("Where is the yellow bottle?")
        self.assertEqual(response.answer, "object 1 is the yellow bottle")
        self.assertEqual(
            response.history["iterations"][0]["function_calls"][0]["name"],
            "search_objects",
        )
        self.assertEqual(len(fake_client.models.calls), 2)
        phases = [event["phase"] for event in progress_events]
        self.assertIn("gemini_start", phases)
        self.assertIn("tool_start", phases)
        self.assertTrue(
            any(
                event["phase"] == "tool_start"
                and "sentence querying" in event["message"]
                for event in progress_events
            )
        )

    def test_viewer_highlight_groups_track_object_tool_results(self) -> None:
        history = {
            "iterations": [
                {
                    "iteration": 1,
                    "function_calls": [
                        {
                            "name": "search_objects",
                            "args": {"description": "book"},
                            "response": {
                                "objects": [
                                    {"object_id": 1, "label": "book"},
                                    {"object_id": 3, "label": "book"},
                                ]
                            },
                        },
                        {
                            "name": "list_rooms",
                            "args": {},
                            "response": {
                                "rooms": [
                                    {
                                        "room_id": 10,
                                        "representative_objects": [
                                            {"object_id": 2, "label": "sink"}
                                        ],
                                    }
                                ]
                            },
                        },
                    ],
                }
            ]
        }
        groups = highlight_groups_from_history(history)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["tool"], "search_objects")
        self.assertEqual(groups[0]["object_ids"], [1, 3])

    def test_proxy_sanitizer_handles_plain_socks_scheme(self) -> None:
        names = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
        old_env = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ.pop(name, None)
            os.environ["https_proxy"] = "socks://127.0.0.1:7890/"
            os.environ["ALL_PROXY"] = "socks://127.0.0.1:7890/"
            sanitize_proxy_environment()
            self.assertTrue(os.environ["https_proxy"].startswith(("http://", "socks5://")))
            self.assertNotEqual(os.environ.get("ALL_PROXY", ""), "socks://127.0.0.1:7890/")
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
