#!/usr/bin/env python3
"""Task-profile, structured-result, and current-frustum tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOMIE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOMIE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scene_qa.camera_view import (  # noqa: E402
    CameraViewError,
    CameraViewSample,
    project_yaw_obb_to_camera,
)
from scene_qa.config import SceneQaConfig  # noqa: E402
from scene_qa.detection_view import (  # noqa: E402
    DETECTION_CAMERA_IDS,
    Latest2dDetectionFrame,
    detection_topics,
)
from scene_qa.embeddings import ObjectSearchIndex  # noqa: E402
from scene_qa.gemini_agent import GeminiSceneQaAgent, QaResponse  # noqa: E402
from scene_qa.graph_store import GraphStore  # noqa: E402
from scene_qa.tasks import (  # noqa: E402
    TASK_FIND_OBJECT_IN_VIEW,
    TASK_NAVIGATION,
    StructuredTaskAgent,
    TaskEvidence,
    build_find_object_in_view_result,
    build_navigation_result,
    create_find_object_in_view_task_registry,
    create_navigation_task_registry,
)
from scene_qa.tools import create_default_tool_registry  # noqa: E402
import roomie_scene_qa  # noqa: E402


def camera_sample() -> CameraViewSample:
    return CameraViewSample.from_mapping(
        {
            "header": {
                "stamp": {"sec": 12, "nanosec": 34},
                "frame_id": "head_color",
            },
            "camera": {
                "width": 100,
                "height": 80,
                "fx": 100.0,
                "fy": 100.0,
                "cx": 50.0,
                "cy": 40.0,
            },
            "transform": {
                "world_frame": "map",
                "camera_frame": "head_color",
                "translation": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
    )


def latest_detection_frames() -> list[Latest2dDetectionFrame]:
    frames = []
    for index, camera_id in enumerate(DETECTION_CAMERA_IDS):
        stamp = {"sec": 12, "nanosec": 34 + index}
        result = {
            "header": {"stamp": stamp, "frame_id": camera_id},
            "camera_id": camera_id,
            "ok": True,
            "error": "",
            "detections": [
                {
                    "label": "yellow_bottle",
                    "semantic_id": 7,
                    "score_2d": 0.91,
                    "bbox_xyxy": [10.0, 20.0, 30.0, 50.0],
                }
            ],
        }
        frames.append(
            Latest2dDetectionFrame(
                camera_id=camera_id,
                available=True,
                image_available=True,
                result_available=True,
                synchronized=True,
                image_header={"stamp": stamp, "frame_id": camera_id},
                result=result,
                image_data=f"png-{camera_id}".encode(),
                image_mime_type="image/png",
                image_error="",
                result_error="",
            )
        )
    return frames


def task_graph(root: Path) -> GraphStore:
    path = root / "task_scene.json"
    path.write_text(
        json.dumps(
            {
                "format": "roomie_test_scene",
                "world_frame": "map",
                "objects": [
                    {
                        "object_id": 1,
                        "label": "yellow_bottle",
                        "description": "small yellow bottle with white cap",
                        "center_world": [0.0, 0.0, 2.0],
                        "size_m": [0.2, 0.2, 0.4],
                        "yaw_rad": 0.0,
                        "publishable": True,
                        "active": False,
                    },
                    {
                        "object_id": 2,
                        "label": "medicine_carton",
                        "description": "small printed medicine carton",
                        "center_world": [0.35, 0.0, 2.0],
                        "size_m": [0.2, 0.2, 0.3],
                        "yaw_rad": 0.2,
                        "publishable": True,
                        "active": True,
                    },
                    {
                        "object_id": 3,
                        "label": "desk",
                        "name": "study desk",
                        "description": "wooden study desk",
                        "center_world": [0.0, 0.0, 2.5],
                        "size_m": [1.2, 0.7, 0.8],
                        "yaw_rad": 0.0,
                        "publishable": True,
                        "active": True,
                    },
                ],
                "furniture": [
                    {
                        "object_id": 3,
                        "classification_label": "desk",
                        "revision": 1,
                    }
                ],
                "relations": [
                    {
                        "source": {"type": "object", "id": 1},
                        "target": {"type": "furniture", "id": 3},
                        "relation_type": "on",
                        "confidence": 0.9,
                    }
                ],
                "rooms": [],
                "snapshot_images": [],
            }
        ),
        encoding="utf-8",
    )
    return GraphStore.load(path)


def registry_for(graph: GraphStore):
    config = SceneQaConfig(
        graph_json=graph.json_path,
        embedding_backend="lexical",
        top_k=12,
    )
    search = ObjectSearchIndex(
        graph,
        model_path=config.embedding_model,
        backend="lexical",
    )
    return config, create_default_tool_registry(graph, search, config)


def qa_response(answer) -> QaResponse:
    return QaResponse(
        reasoning="model reasoning",
        answer=answer,
        raw_text=json.dumps(answer, ensure_ascii=False),
        history={},
    )


def test_corner_frustum_projection_and_clipped_roi() -> None:
    sample = camera_sample()
    centered = project_yaw_obb_to_camera(
        [0.0, 0.0, 2.0], [0.2, 0.2, 0.4], 0.0, sample
    )
    assert centered.in_view
    assert centered.visible_corner_count == 8
    assert centered.bbox is not None
    assert centered.bbox["width"] > 0
    assert centered.bbox["height"] > 0

    # Only the left-hand corners enter the image, which is sufficient by the
    # product definition and produces an image-clipped ROI.
    partial = project_yaw_obb_to_camera(
        [1.7, 0.0, 2.0], [2.0, 0.4, 0.4], 0.0, sample
    )
    assert partial.in_view
    assert 0 < partial.visible_corner_count < 8
    assert partial.bbox["x_offset"] < sample.width
    assert partial.bbox["x_offset"] + partial.bbox["width"] <= sample.width

    behind = project_yaw_obb_to_camera(
        [0.0, 0.0, -2.0], [0.2, 0.2, 0.4], 0.0, sample
    )
    assert not behind.in_view
    assert behind.bbox is None


def test_navigation_primary_tool_and_explicit_furniture_override() -> None:
    with tempfile.TemporaryDirectory() as directory:
        graph = task_graph(Path(directory))
        config, base = registry_for(graph)
        evidence = TaskEvidence(TASK_NAVIGATION)
        registry = create_navigation_task_registry(base, config, evidence)

        primary = registry.call_tool(
            "locate_navigation_target",
            {
                "target_description": "vitamin C medicine carton",
                "requested_furniture_description": "desk",
            },
        )

        assert registry.list_tools() == [
            "locate_navigation_target",
            "inspect_navigation_candidates",
        ]
        assert primary.response["explicit_furniture_unique"]
        assert primary.response["explicit_furniture_matches"][0]["object_id"] == 3
        result = build_navigation_result(
            qa_response(
                {
                    "status": "NOT_FOUND",
                    "reason": "object not confirmed",
                    "exploration_furniture_ids": [3],
                }
            ),
            evidence,
            allow_follow_up_question=True,
            world_frame="map",
        )
        assert result["status"] == 0
        assert result["target_furniture"]["furniture_id"] == 3
        assert result["target_furniture"]["position"]["header"]["frame_id"] == "map"


def test_navigation_forbidden_follow_up_selects_best_witnessed_destination() -> None:
    with tempfile.TemporaryDirectory() as directory:
        graph = task_graph(Path(directory))
        config, base = registry_for(graph)
        evidence = TaskEvidence(TASK_NAVIGATION)
        registry = create_navigation_task_registry(base, config, evidence)
        registry.call_tool(
            "locate_navigation_target", {"target_description": "yellow bottle"}
        )
        result = build_navigation_result(
            qa_response(
                {
                    "status": "NEED_CONFIRM",
                    "reason": "ambiguous",
                    "follow_up_question": "which one?",
                }
            ),
            evidence,
            allow_follow_up_question=False,
            world_frame="map",
        )
        assert result["status"] == 0
        assert result["target_furniture"]["furniture_id"] == 3
        assert result["follow_up_question"] == ""


def test_in_view_primary_strong_match_and_structured_bbox() -> None:
    with tempfile.TemporaryDirectory() as directory:
        graph = task_graph(Path(directory))
        config, base = registry_for(graph)
        evidence = TaskEvidence(TASK_FIND_OBJECT_IN_VIEW)
        sampler = SimpleNamespace(sample=camera_sample)
        registry = create_find_object_in_view_task_registry(
            base, config, evidence, sampler
        )

        primary = registry.call_tool(
            "find_objects_in_view", {"target_description": "yellow bottle"}
        )

        assert primary.response["strong_match_object_id"] == 1
        assert not primary.response["current_rgb_inspected"]
        result = build_find_object_in_view_result(
            qa_response(
                {
                    "status": "FOUND",
                    "reason": "unique exact label",
                    "target_object_id": 1,
                    "suspected_object_ids": [],
                }
            ),
            evidence,
            world_frame="map",
        )
        assert result["status"] == 0
        assert result["target_object"]["object_id"] == 1
        camera_value = result["target_object_in_head_cam"]
        assert camera_value["header"]["stamp"] == {"sec": 12, "nanosec": 34}
        assert camera_value["bbox"]["width"] > 0
        assert result["suspected_objects"] == []


def test_in_view_vlm_tool_returns_latest_three_camera_2d_results() -> None:
    with tempfile.TemporaryDirectory() as directory:
        graph = task_graph(Path(directory))
        config, base = registry_for(graph)
        evidence = TaskEvidence(TASK_FIND_OBJECT_IN_VIEW)
        detection_sampler = SimpleNamespace(sample=latest_detection_frames)
        registry = create_find_object_in_view_task_registry(
            base,
            config,
            evidence,
            SimpleNamespace(sample=camera_sample),
            detection_sampler=detection_sampler,
        )

        before_primary = registry.call_tool("inspect_latest_2d_detections", {})
        assert "call find_objects_in_view" in before_primary.response["error"]
        registry.call_tool(
            "find_objects_in_view", {"target_description": "yellow bottle"}
        )
        result = registry.call_tool("inspect_latest_2d_detections", {})

        assert registry.list_tools() == [
            "find_objects_in_view",
            "inspect_latest_2d_detections",
            "inspect_in_view_candidates",
        ]
        assert result.response["available_camera_count"] == 3
        assert [camera["camera_id"] for camera in result.response["cameras"]] == list(
            DETECTION_CAMERA_IDS
        )
        assert result.response["cameras"][0]["detections"][0]["label"] == (
            "yellow_bottle"
        )
        assert result.response["cameras"][0]["head_geometry_sample_delta_ms"] == 0.0
        assert len(result.media) == 3
        assert [media.summary["camera_id"] for media in result.media] == list(
            DETECTION_CAMERA_IDS
        )
        repeated = registry.call_tool("inspect_latest_2d_detections", {})
        assert "only once" in repeated.response["error"]


def test_latest_detection_topics_use_unsuffixed_head_and_suffixed_hands() -> None:
    topics = detection_topics("/images/", "/results/")
    assert topics["head_color"] == {"image": "/images", "result": "/results"}
    assert topics["hand_left_color"] == {
        "image": "/images/hand_left_color",
        "result": "/results/hand_left_color",
    }
    assert topics["hand_right_color"] == {
        "image": "/images/hand_right_color",
        "result": "/results/hand_right_color",
    }


def test_invalid_in_view_model_id_downgrades_to_aligned_suspected_arrays() -> None:
    with tempfile.TemporaryDirectory() as directory:
        graph = task_graph(Path(directory))
        config, base = registry_for(graph)
        evidence = TaskEvidence(TASK_FIND_OBJECT_IN_VIEW)
        registry = create_find_object_in_view_task_registry(
            base, config, evidence, SimpleNamespace(sample=camera_sample)
        )
        registry.call_tool(
            "find_objects_in_view", {"target_description": "medicine package"}
        )

        result = build_find_object_in_view_result(
            qa_response(
                {
                    "status": "FOUND",
                    "reason": "invented",
                    "target_object_id": 999,
                }
            ),
            evidence,
            world_frame="map",
        )

        assert result["status"] == 1
        assert "unwitnessed" in result["reason"]
        assert len(result["suspected_objects"]) == len(
            result["suspected_objects_in_head_cam"]
        )
        assert result["suspected_objects"]


def test_in_view_sensor_failure_is_fatal_not_not_found() -> None:
    class FailedSampler:
        def sample(self):
            raise CameraViewError("camera_image_timeout", "no image")

    with tempfile.TemporaryDirectory() as directory:
        graph = task_graph(Path(directory))
        config, base = registry_for(graph)
        registry = create_find_object_in_view_task_registry(
            base,
            config,
            TaskEvidence(TASK_FIND_OBJECT_IN_VIEW),
            FailedSampler(),
        )
        with pytest.raises(CameraViewError) as context:
            registry.call_tool(
                "find_objects_in_view", {"target_description": "yellow bottle"}
            )
        assert context.value.status == "camera_image_timeout"


def test_in_view_rejects_invalid_depth_range_at_startup() -> None:
    with tempfile.TemporaryDirectory() as directory:
        graph = task_graph(Path(directory))
        config, base = registry_for(graph)
        config.in_view_min_depth_m = 2.0
        config.in_view_max_depth_m = 1.0
        with pytest.raises(ValueError, match="in-view depth range"):
            create_find_object_in_view_task_registry(
                base,
                config,
                TaskEvidence(TASK_FIND_OBJECT_IN_VIEW),
                SimpleNamespace(sample=camera_sample),
            )


def test_task_cli_selects_prompt_and_three_iteration_default() -> None:
    with patch.object(
        sys,
        "argv",
        ["roomie_scene_qa.py", "--task", "navigation", "find", "medicine"],
    ):
        args = roomie_scene_qa.parse_args()
    config = roomie_scene_qa.apply_cli_overrides(SceneQaConfig(), args)
    assert args.task == TASK_NAVIGATION
    assert config.max_iterations == 3
    assert config.system_prompt_path.name == "navigation.txt"

    with patch.object(
        sys,
        "argv",
        ["roomie_scene_qa.py", "--task", "find_object_in_view", "find", "bottle"],
    ):
        in_view_args = roomie_scene_qa.parse_args()
    in_view_config = roomie_scene_qa.apply_cli_overrides(
        SceneQaConfig(), in_view_args
    )
    assert in_view_config.max_iterations == 4
    assert in_view_config.system_prompt_path.name == "find_object_in_view.txt"

    with patch.object(sys, "argv", ["roomie_scene_qa.py"]):
        default_args = roomie_scene_qa.parse_args()
    default_config = roomie_scene_qa.apply_cli_overrides(
        SceneQaConfig(max_iterations=9), default_args
    )
    assert default_config.max_iterations == 9
    assert default_config.system_prompt_path.name == "system.txt"


def test_gemini_navigation_profile_exposes_only_bounded_task_tools() -> None:
    class FakeCall:
        name = "locate_navigation_target"
        args = {"target_description": "yellow bottle"}

    class FakeResponse:
        candidates = []
        parts = []

        def __init__(self, text="", function_calls=None):
            self.text = text
            self.function_calls = function_calls or []

    class FakeModels:
        def __init__(self):
            self.configs = []
            self.responses = [
                FakeResponse(function_calls=[FakeCall()]),
                FakeResponse(
                    json.dumps(
                        {
                            "reasoning": "exact scene match",
                            "answer": {
                                "status": "FOUND",
                                "reason": "found on desk",
                                "target_object_id": 1,
                                "target_furniture_id": 3,
                                "exploration_furniture_ids": [],
                                "follow_up_question": "",
                                "vlm_description": "目标在书桌上。",
                            },
                        },
                        ensure_ascii=False,
                    )
                ),
            ]

        def generate_content(self, **kwargs):
            self.configs.append(kwargs["config"])
            return self.responses.pop(0)

    with tempfile.TemporaryDirectory() as directory:
        graph = task_graph(Path(directory))
        config, base = registry_for(graph)
        config.system_prompt_path = config.navigation_system_prompt_path
        config.max_iterations = 3
        evidence = TaskEvidence(TASK_NAVIGATION)
        registry = create_navigation_task_registry(base, config, evidence)
        fake_models = FakeModels()
        inner = GeminiSceneQaAgent(
            graph,
            registry,
            config,
            client=SimpleNamespace(models=fake_models),
        )
        agent = StructuredTaskAgent(
            inner,
            evidence,
            task=TASK_NAVIGATION,
            world_frame="map",
        )

        response = agent.answer_query("帮我拿黄色瓶子")

        assert response.answer["status"] == 0
        assert response.answer["target_furniture"]["furniture_id"] == 3
        first_config = fake_models.configs[0]
        assert "navigation-target selector" in first_config.system_instruction
        declarations = first_config.tools[0].function_declarations
        assert [declaration.name for declaration in declarations] == [
            "locate_navigation_target",
            "inspect_navigation_candidates",
        ]
        assert len(response.history["iterations"]) == 2
