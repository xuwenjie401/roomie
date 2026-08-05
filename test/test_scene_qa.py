#!/usr/bin/env python3
"""Offline tests for the Roomie scene QA tool layer."""

from __future__ import annotations

import json
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

ROOMIE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOMIE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scene_qa.config import SceneQaConfig  # noqa: E402
from scene_qa.doubao_agent import DoubaoSceneQaAgent  # noqa: E402
from scene_qa.embeddings import ObjectSearchIndex  # noqa: E402
from scene_qa.gemini_agent import GeminiSceneQaAgent, sanitize_proxy_environment  # noqa: E402
from scene_qa.graph_store import GraphStore  # noqa: E402
from scene_qa.live_query import LiveSceneQueryClient, LiveSceneQueryError  # noqa: E402
from scene_qa.tools import create_default_tool_registry, create_live_tool_registry  # noqa: E402
import roomie_scene_qa  # noqa: E402
import roomie_scene_qa_viewer  # noqa: E402
from roomie_scene_qa_viewer import (  # noqa: E402
    highlight_groups_from_history,
    live_graph_payload,
)


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
                    "doubao_model": "doubao-test-model",
                    "doubao_base_url": "https://ark.example.test/api/v3/",
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
        self.assertEqual(config.doubao_model, "doubao-test-model")
        self.assertEqual(config.doubao_base_url, "https://ark.example.test/api/v3")
        self.assertEqual(config.embedding_backend, "lexical")
        self.assertEqual(config.top_k, 3)
        self.assertEqual(config.snapshot_bbox_pad_px, 5)
        self.assertEqual(config.resolved_system_prompt_path(), prompt_path.resolve())
        self.assertEqual(config.load_system_prompt(), "test system prompt")
        with patch.object(
            sys,
            "argv",
            ["roomie_scene_qa.py", "--qa-config", str(config_path)],
        ):
            args = roomie_scene_qa.parse_args()
        overridden = roomie_scene_qa.apply_cli_overrides(config, args)
        self.assertEqual(overridden.resolved_system_prompt_path(), prompt_path.resolve())
        self.assertEqual(overridden.load_system_prompt(), "test system prompt")

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

    def test_doubao_loop_executes_local_tool_over_responses_api(self) -> None:
        class FakeHttpResponse:
            status_code = 200

            def __init__(self, payload: dict):
                self.payload = payload

            def json(self) -> dict:
                return self.payload

        class FakeHttpClient:
            def __init__(self):
                self.calls: list[dict] = []
                self.responses = [
                    {
                        "id": "resp-tool",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call-search",
                                "name": "search_objects",
                                "arguments": json.dumps(
                                    {"description": "yellow bottle", "top_k": 1}
                                ),
                            },
                            {
                                "type": "function_call",
                                "call_id": "call-snapshot",
                                "name": "inspect_snapshot",
                                "arguments": json.dumps(
                                    {
                                        "object_id": 1,
                                        "crop": True,
                                        "draw_bbox": True,
                                    }
                                ),
                            },
                        ],
                    },
                    {
                        "id": "resp-answer",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": (
                                            '{"reasoning":"searched objects",'
                                            '"answer":"object 1 is the yellow bottle"}'
                                        ),
                                    }
                                ],
                            }
                        ],
                    },
                ]

            def post(self, url: str, **kwargs):
                self.calls.append({"url": url, **kwargs})
                return FakeHttpResponse(self.responses.pop(0))

        fake_client = FakeHttpClient()
        progress_events = []
        config = SceneQaConfig(
            graph_json=self.json_path,
            embedding_backend="lexical",
            doubao_model="doubao-test-model",
            doubao_base_url="https://ark.example.test/api/v3",
            top_k=5,
        )
        agent = DoubaoSceneQaAgent(
            self.graph,
            self.registry,
            config,
            api_key="test-key",
            client=fake_client,
            progress_callback=progress_events.append,
        )
        response = agent.answer_query("Where is the yellow bottle?")

        self.assertEqual(response.answer, "object 1 is the yellow bottle")
        self.assertEqual(response.history["provider"], "doubao")
        self.assertEqual(response.history["model"], "doubao-test-model")
        self.assertEqual(len(fake_client.calls), 2)
        self.assertEqual(
            fake_client.calls[0]["url"],
            "https://ark.example.test/api/v3/responses",
        )
        self.assertEqual(
            fake_client.calls[0]["headers"]["Authorization"],
            "Bearer test-key",
        )
        first_tool = fake_client.calls[0]["json"]["tools"][0]
        self.assertEqual(first_tool["type"], "function")
        self.assertIn("parameters", first_tool)
        continuation = fake_client.calls[1]["json"]
        self.assertEqual(continuation["previous_response_id"], "resp-tool")
        self.assertEqual(continuation["input"][0]["type"], "function_call_output")
        self.assertEqual(continuation["input"][0]["call_id"], "call-search")
        self.assertEqual(continuation["input"][1]["call_id"], "call-snapshot")
        self.assertEqual(continuation["input"][2]["type"], "message")
        image_part = continuation["input"][2]["content"][1]
        self.assertEqual(image_part["type"], "input_image")
        self.assertTrue(image_part["image_url"].startswith("data:image/jpeg;base64,"))
        phases = [event["phase"] for event in progress_events]
        self.assertIn("doubao_start", phases)
        self.assertIn("tool_start", phases)
        self.assertIn("final_answer", phases)

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


class FakeLiveQueryTransport:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.session_number = 0
        self.expire_calls = False

    def __call__(self, payload: dict) -> dict:
        self.requests.append(json.loads(json.dumps(payload)))
        if payload.get("operation") == "begin_session":
            self.session_number += 1
            session_id = f"opaque-{self.session_number}"
            return {
                "success": True,
                "session_id": session_id,
                "session": {
                    "session_id": session_id,
                    "scene_revision": 77,
                    "durable_scene_revision": 76,
                    "index_generation": 9,
                },
                "read_token": {
                    "scene_revision": 77,
                    "durable_scene_revision": 76,
                    "index_generation": 9,
                },
                "calls": [],
            }
        if self.expire_calls:
            return {
                "success": False,
                "status": "expired_session",
                "error": "read session expired",
                "session_id": payload.get("session_id"),
            }
        self.assert_session_request(payload)
        results = []
        for call in payload["calls"]:
            method = call["method"]
            params = call.get("params") or {}
            if method == "search_objects":
                result = [
                    {
                        "object": self.object_result(1, "yellow bottle", [1.0, 0.0, 0.5]),
                        "score": 0.96,
                        "source": "vector",
                    }
                ]
            elif method == "get_object":
                result = self.object_result(
                    int(params["object_id"]), "yellow bottle", [1.0, 0.0, 0.5]
                )
            elif method == "get_objects_near":
                result = [self.object_result(1, "yellow bottle", [1.0, 0.0, 0.5])]
            elif method == "rooms":
                result = [
                    {
                        "room_id": "kitchen",
                        "name": "Kitchen",
                        "object_ids": [1],
                    }
                ]
            elif method == "inspect_snapshot":
                result = {
                    "object": self.object_result(1, "yellow bottle", [1.0, 0.0, 0.5]),
                    "snapshots": [],
                }
            else:
                raise AssertionError(f"unexpected method: {method}")
            results.append(
                {
                    "method": method,
                    "success": True,
                    "status": "ok",
                    "message": "",
                    "metadata": {"scene_revision": 77},
                    "result": result,
                }
            )
        return {
            "success": True,
            "session_id": payload["session_id"],
            "read_token": {"scene_revision": 77},
            "calls": results,
        }

    def assert_session_request(self, payload: dict) -> None:
        if payload.get("session_id") != f"opaque-{self.session_number}":
            raise AssertionError("tool call did not reuse current session_id")
        if payload.get("expected_scene_revision") != 77:
            raise AssertionError("tool call did not carry pinned revision")

    @staticmethod
    def object_result(object_id: int, label: str, center: list[float]) -> dict:
        return {
            "object_id": object_id,
            "label": label,
            "display_description": label,
            "geometry": {
                "center_world": center,
                "size_m": [0.1, 0.1, 0.2],
                "yaw_rad": 0.0,
            },
        }


class LiveSceneQaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = FakeLiveQueryTransport()
        self.client = LiveSceneQueryClient(self.transport, session_ttl_ms=5_000)
        self.config = SceneQaConfig(embedding_backend="lexical", max_iterations=4)

    def test_client_reuses_one_opaque_session_and_detects_expiry(self) -> None:
        metadata = self.client.begin_answer()
        self.assertEqual(metadata["session_id"], "opaque-1")
        self.assertEqual(metadata["scene_revision"], 77)
        self.client.call("search_objects", {"query": "yellow bottle", "limit": 1})
        self.client.call("get_object", {"object_id": 1})
        self.assertEqual(len(self.transport.requests), 3)
        for request in self.transport.requests[1:]:
            self.assertEqual(request["session_id"], "opaque-1")
            self.assertEqual(request["expected_scene_revision"], 77)

        self.transport.expire_calls = True
        with self.assertRaises(LiveSceneQueryError) as context:
            self.client.call("rooms", {})
        self.assertEqual(context.exception.status, "expired_session")

    def test_live_registry_normalizes_gateway_results(self) -> None:
        registry = create_live_tool_registry(self.client, self.config)
        session = registry.begin_answer()
        self.assertEqual(session["scene_revision"], 77)
        search = registry.call_tool(
            "search_objects", {"description": "yellow bottle", "top_k": 1}
        )
        self.assertEqual(search.response["objects"][0]["object_id"], 1)
        self.assertEqual(search.response["objects"][0]["semantic_score"], 0.96)
        nearby = registry.call_tool(
            "get_objects_near", {"object_id": 1, "radius_m": 1.0}
        )
        self.assertEqual(nearby.response["objects"][0]["object_id"], 1)
        registry.end_answer()
        self.assertIsNone(self.client.session_id)

    def test_one_gemini_answer_uses_one_live_read_session(self) -> None:
        class FakeCall:
            def __init__(self, name: str, args: dict):
                self.name = name
                self.args = args

        class FakeResponse:
            def __init__(self, text: str = "", function_calls: list | None = None):
                self.text = text
                self.function_calls = function_calls or []
                self.candidates = []
                self.parts = []

        class FakeModels:
            def __init__(self):
                self.responses = [
                    FakeResponse(
                        function_calls=[
                            FakeCall("search_objects", {"description": "yellow bottle"})
                        ]
                    ),
                    FakeResponse(
                        function_calls=[
                            FakeCall(
                                "get_object",
                                {"object_id": 1, "include_neighbors": False},
                            )
                        ]
                    ),
                    FakeResponse('{"reasoning":"live tools","answer":"object 1"}'),
                ]

            def generate_content(self, **_kwargs):
                return self.responses.pop(0)

        class FakeGeminiClient:
            def __init__(self):
                self.models = FakeModels()

        registry = create_live_tool_registry(self.client, self.config)
        agent = GeminiSceneQaAgent(
            None,
            registry,
            self.config,
            client=FakeGeminiClient(),
        )
        response = agent.answer_query("黄色瓶子在哪里？")
        self.assertEqual(response.answer, "object 1")
        self.assertEqual(response.history["source"], "live_query_service")
        self.assertEqual(response.history["read_session"]["session_id"], "opaque-1")
        begin_requests = [
            request
            for request in self.transport.requests
            if request.get("operation") == "begin_session"
        ]
        self.assertEqual(len(begin_requests), 1)
        for request in self.transport.requests[1:]:
            self.assertEqual(request["session_id"], "opaque-1")
            self.assertEqual(request["expected_scene_revision"], 77)
        self.assertIsNone(self.client.session_id)

    def test_cli_defaults_to_live_and_offline_requires_explicit_flag(self) -> None:
        with patch.object(sys, "argv", ["roomie_scene_qa.py"]):
            prompt_args = roomie_scene_qa.parse_args()
        self.assertEqual(prompt_args.query, [])
        self.assertFalse(prompt_args.once)
        self.assertIsNone(prompt_args.offline_json)
        self.assertEqual(prompt_args.provider, "gemini")

        with patch.object(sys, "argv", ["roomie_scene_qa.py", "where is the chair"]):
            args = roomie_scene_qa.parse_args()
        self.assertIsNone(args.offline_json)
        self.assertEqual(args.service, "/roomie/query_scene")
        self.assertFalse(args.once)

        offline_path = Path("/tmp/explicit-scene.json")
        with patch.object(
            sys,
            "argv",
            [
                "roomie_scene_qa.py",
                "--offline-json",
                str(offline_path),
                "where",
            ],
        ):
            offline_args = roomie_scene_qa.parse_args()
        self.assertEqual(offline_args.offline_json, offline_path)

    def test_cli_default_loop_reuses_agent_until_exit(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.queries: list[str] = []

            def answer_query(self, query: str) -> SimpleNamespace:
                self.queries.append(query)
                return SimpleNamespace(
                    reasoning=f"trace for {query}",
                    answer=f"answer for {query}",
                    history={"query": query, "iterations": []},
                )

        inputs = iter(["second question", "", "/exit"])
        stdout = io.StringIO()
        stderr = io.StringIO()
        agent = FakeAgent()
        result = roomie_scene_qa.run_query_loop(
            agent,  # type: ignore[arg-type]
            initial_query="first question",
            input_fn=lambda _prompt: next(inputs),
            stdout=stdout,
            stderr=stderr,
        )
        self.assertEqual(result, 0)
        self.assertEqual(agent.queries, ["first question", "second question"])
        self.assertIn("interactive mode", stdout.getvalue())
        self.assertIn("answer for first question", stdout.getvalue())
        self.assertIn("answer for second question", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_cli_once_mode_requires_and_answers_one_query(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.queries: list[str] = []

            def answer_query(self, query: str) -> SimpleNamespace:
                self.queries.append(query)
                return SimpleNamespace(
                    reasoning="one trace",
                    answer="one answer",
                    history={"query": query},
                )

        agent = FakeAgent()
        stdout = io.StringIO()
        self.assertEqual(
            roomie_scene_qa.run_query_loop(
                agent,  # type: ignore[arg-type]
                initial_query="one question",
                once=True,
                stdout=stdout,
            ),
            0,
        )
        self.assertEqual(agent.queries, ["one question"])
        self.assertNotIn("interactive mode", stdout.getvalue())

        stderr = io.StringIO()
        self.assertEqual(
            roomie_scene_qa.run_query_loop(
                agent,  # type: ignore[arg-type]
                once=True,
                stderr=stderr,
            ),
            2,
        )
        self.assertIn("requires", stderr.getvalue())

    def test_viewer_defaults_to_live_service(self) -> None:
        with patch.object(sys, "argv", ["roomie_scene_qa_viewer.py"]):
            args = roomie_scene_qa_viewer.parse_args()
        self.assertIsNone(args.offline_json)
        self.assertEqual(args.service, "/roomie/query_scene")
        self.assertEqual(args.default_provider, "gemini")
        self.assertIn('data-provider="gemini"', roomie_scene_qa_viewer.HTML)
        self.assertIn('data-provider="doubao"', roomie_scene_qa_viewer.HTML)
        self.assertIn("JSON.stringify({query, provider})", roomie_scene_qa_viewer.HTML)
        payload = live_graph_payload(args.service)
        self.assertTrue(payload["live"])
        self.assertEqual(payload["objects"], [])

        with patch.object(
            sys,
            "argv",
            ["roomie_scene_qa_viewer.py", "--offline-json", "/tmp/scene.json"],
        ):
            offline_args = roomie_scene_qa_viewer.parse_args()
        self.assertEqual(offline_args.offline_json, Path("/tmp/scene.json"))


if __name__ == "__main__":
    unittest.main()
