#!/usr/bin/env python3
"""Tests for the curated object-reference build and Scene QA tools."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

from PIL import Image


ROOMIE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOMIE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scene_qa.config import SceneQaConfig  # noqa: E402
from scene_qa.embeddings import ObjectSearchIndex  # noqa: E402
from scene_qa.graph_store import GraphStore  # noqa: E402
from scene_qa.live_query import LiveSceneQueryClient  # noqa: E402
from scene_qa.object_references import (  # noqa: E402
    ObjectReferenceCatalog,
    build_object_reference_catalog,
    load_or_scan_object_reference_manifest,
    save_object_reference_manifest,
    try_load_object_reference_catalog,
    validate_object_reference_manifest,
)
from scene_qa.tools import (  # noqa: E402
    create_default_tool_registry,
    create_live_tool_registry,
)


def create_reference_build(root: Path):
    raw = root / "raw" / "维生素C片"
    raw.mkdir(parents=True)
    for index in range(1, 6):
        Image.new(
            "RGB",
            (120, 90),
            (30 * index, 220 - 20 * index, 80 + 10 * index),
        ).save(raw / f"vc{index}.png")
    manifest = load_or_scan_object_reference_manifest(root)
    reference = manifest["references"][0]
    reference["aliases"] = ["Vitamin C tablets", "维C片"]
    for rank, image in enumerate(reference["images"][:3], start=1):
        image.update(
            {
                "crop_xyxy": [10, 5, 110, 85],
                "crop_confirmed": True,
                "selected": True,
                "rank": rank,
                "view_label": ["正面", "侧面", "背面"][rank - 1],
            }
        )
    save_object_reference_manifest(root, manifest)
    return manifest, build_object_reference_catalog(root, manifest)


def create_scene_fixture(root: Path) -> tuple[GraphStore, Path]:
    snapshot = root / "snapshot.png"
    Image.new("RGB", (100, 80), (230, 230, 230)).save(snapshot)
    graph_json = root / "scene.json"
    graph_json.write_text(
        json.dumps(
            {
                "format": "roomie_test_scene",
                "objects": [
                    {
                        "object_id": 7,
                        "label": "medicine_carton",
                        "description": "a small printed medicine carton",
                        "publishable": True,
                        "active": True,
                        "snapshot": {
                            "image_index": 0,
                            "bbox_xyxy": [15, 10, 75, 70],
                        },
                    }
                ],
                "rooms": [],
                "relations": [],
                "snapshot_images": [
                    {
                        "image_index": 0,
                        "encoding": "png",
                        "width": 100,
                        "height": 80,
                        "source_path": str(snapshot),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return GraphStore.load(graph_json), snapshot


def test_editor_draft_requires_human_crop_and_three_ranked_views() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "raw" / "维生素C片"
        raw.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (64, 48), (index * 50, 100, 200)).save(
                raw / f"view{index}.png"
            )

        manifest = load_or_scan_object_reference_manifest(root)

        assert manifest["references"][0]["reference_id"] == "维生素C片"
        assert all(
            not image["crop_confirmed"]
            for image in manifest["references"][0]["images"]
        )
        errors = validate_object_reference_manifest(
            root, manifest, require_publishable=True
        )
        assert errors
        assert "ranks 1, 2, and 3" in errors[0]


def test_build_publishes_three_processed_views_and_detects_corruption() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest, catalog = create_reference_build(root)

        assert ObjectReferenceCatalog.load(root).build_id == catalog.build_id
        reference = catalog.get("维生素C片")
        assert reference.aliases == ("Vitamin C tablets", "维C片")
        assert [image.rank for image in reference.images] == [1, 2, 3]
        assert all(max(image.width, image.height) <= 768 for image in reference.images)
        assert len(list((catalog.build_root / "images").glob("*.jpg"))) == 3
        assert build_object_reference_catalog(root, manifest).build_id == catalog.build_id

        reference.images[0].path.write_bytes(b"corrupt")
        failed = try_load_object_reference_catalog(root)
        assert not failed.available
        assert "size mismatch" in failed.status


def test_offline_reference_tools_attach_labeled_joint_evidence() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference_root = root / "references"
        _, catalog = create_reference_build(reference_root)
        load = try_load_object_reference_catalog(reference_root)
        graph, _ = create_scene_fixture(root)
        config = SceneQaConfig(
            graph_json=graph.json_path,
            embedding_backend="lexical",
            object_reference_root=reference_root,
        )
        search = ObjectSearchIndex(
            graph, model_path=config.embedding_model, backend="lexical"
        )
        registry = create_default_tool_registry(graph, search, config, load)

        assert "inspect_object_reference" in registry.list_tools()
        assert "compare_scene_objects_to_reference" in registry.list_tools()
        assert "reference_id='维生素C片'" in registry.system_prompt("base")
        inspected = registry.call_tool(
            "inspect_object_reference", {"reference_id": "维生素C片"}
        )
        assert len(inspected.media) == 3
        assert all(item.summary["role"] == "reference" for item in inspected.media)

        compared = registry.call_tool(
            "compare_scene_objects_to_reference",
            {"reference_id": "维生素C片", "object_ids": [7]},
        )
        assert compared.response["scene_object_count"] == 1
        assert compared.response["scene_snapshot_attached_object_ids"] == [7]
        assert len(compared.media) == 4
        assert compared.media[-1].summary["role"] == "scene_object"
        assert compared.media[-1].summary["object_id"] == 7
        assert catalog.build_id == compared.response["catalog_build_id"]


def test_missing_catalog_removes_reference_tools_and_changes_prompt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        graph, _ = create_scene_fixture(root)
        config = SceneQaConfig(embedding_backend="lexical")
        search = ObjectSearchIndex(
            graph, model_path=config.embedding_model, backend="lexical"
        )
        load = try_load_object_reference_catalog(root / "missing")
        registry = create_default_tool_registry(graph, search, config, load)

        assert "inspect_object_reference" not in registry.list_tools()
        assert "compare_scene_objects_to_reference" not in registry.list_tools()
        assert "unavailable for this run" in registry.system_prompt("base")


def test_live_joint_comparison_batches_all_scene_calls() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference_root = root / "references"
        create_reference_build(reference_root)
        load = try_load_object_reference_catalog(reference_root)
        snapshot = root / "live.png"
        Image.new("RGB", (80, 60), (180, 180, 180)).save(snapshot)
        requests: list[dict] = []

        def transport(request: dict) -> dict:
            requests.append(request)
            if request.get("operation") == "begin_session":
                return {
                    "success": True,
                    "session_id": "reference-session",
                    "read_token": {"scene_revision": 9},
                }
            results = []
            for call in request["calls"]:
                object_id = int(call["params"]["object_id"])
                if call["method"] == "get_object":
                    value = {
                        "object_id": object_id,
                        "label": "medicine_carton",
                        "publishable": True,
                    }
                else:
                    value = {
                        "snapshots": [
                            {
                                "available": True,
                                "asset": {"source_path": str(snapshot)},
                                "reference": {"bbox_xyxy": [5, 5, 70, 55]},
                            }
                        ]
                    }
                results.append(
                    {
                        "success": True,
                        "metadata": {"scene_revision": 9},
                        "result": value,
                    }
                )
            return {
                "success": True,
                "session_id": "reference-session",
                "read_token": {"scene_revision": 9},
                "calls": results,
            }

        client = LiveSceneQueryClient(transport)
        config = SceneQaConfig(
            embedding_backend="lexical",
            object_reference_root=reference_root,
        )
        registry = create_live_tool_registry(client, config, load)

        result = registry.call_tool(
            "compare_scene_objects_to_reference",
            {"reference_id": "维生素C片", "object_ids": [7, 8]},
        )

        assert len(requests) == 2
        assert [call["method"] for call in requests[1]["calls"]] == [
            "get_object",
            "inspect_snapshot",
            "get_object",
            "inspect_snapshot",
        ]
        assert result.response["scene_snapshot_attached_object_ids"] == [7, 8]
        assert len(result.media) == 5
        assert result.response["read_session"]["scene_revision"] == 9
