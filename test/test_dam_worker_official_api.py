#!/usr/bin/env python3
"""Model-free contract tests for the official NVlabs DAM adapter."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types

from PIL import Image
import pytest


ROOMIE_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOMIE_ROOT / "scripts" / "roomie_python_dam_worker.py"
SPEC = importlib.util.spec_from_file_location("roomie_python_dam_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


def _startup_args(source: Path, model: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dam_src=source,
        model_path=str(model),
        model_id="official-dam-test",
        conv_mode="v1",
        prompt_mode="focal_prompt",
        query=worker.DEFAULT_QUERY,
    )


def test_prepare_uses_official_describe_anything_api(monkeypatch, tmp_path):
    source = tmp_path / "describe-anything"
    model = tmp_path / "DAM-3B"
    source.mkdir()
    model.mkdir()

    calls: dict[str, object] = {}

    class FakeDescribeAnythingModel:
        def __init__(self, *, model_path, conv_mode, prompt_mode):
            calls["constructor"] = (model_path, conv_mode, prompt_mode)

    fake_dam = types.ModuleType("dam")
    fake_dam.DescribeAnythingModel = FakeDescribeAnythingModel
    fake_dam.disable_torch_init = lambda: calls.setdefault("disabled", True)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
    )
    fake_torch.version = types.SimpleNamespace(cuda="test-cuda")
    monkeypatch.setitem(sys.modules, "dam", fake_dam)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    source_text = str(source.resolve())
    try:
        agent = worker.prepare_dam_agent(_startup_args(source, model))
    finally:
        if source_text in sys.path:
            sys.path.remove(source_text)

    assert isinstance(agent, FakeDescribeAnythingModel)
    assert calls["disabled"] is True
    assert calls["constructor"] == (
        str(model.resolve()),
        "v1",
        "full+focal_crop",
    )


def test_prepare_rejects_query_without_official_image_token(tmp_path):
    source = tmp_path / "describe-anything"
    model = tmp_path / "DAM-3B"
    source.mkdir()
    model.mkdir()
    args = _startup_args(source, model)
    args.query = "Describe the masked region."
    with pytest.raises(worker.WorkerFailure, match="<image>") as failure:
        worker.prepare_dam_agent(args)
    assert failure.value.retryable is False


def test_describe_calls_get_description_with_non_streaming_contract(tmp_path):
    image_path = (tmp_path / "frame.png").resolve()
    Image.new("RGB", (8, 6), (20, 40, 60)).save(image_path)
    captured: dict[str, object] = {}

    class FakeAgent:
        def get_description(self, image, mask, query, **kwargs):
            captured["image"] = image.copy()
            captured["mask"] = mask.copy()
            captured["query"] = query
            captured["kwargs"] = kwargs
            return "a blue rectangular object"

    args = argparse.Namespace(
        bbox_pad_px=0.0,
        query=worker.DEFAULT_QUERY,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=256,
    )
    input_value = {
        "snapshots": [
            {
                "source_frame_asset_id": "asset-1",
                "bbox_xyxy": [1.0, 1.0, 5.0, 4.0],
                "durable_asset": {
                    "asset_id": "asset-1",
                    "access_mode": "read_only",
                    "immutable": True,
                    "path": str(image_path),
                },
            }
        ]
    }

    assert worker.describe(FakeAgent(), input_value, args) == (
        "a blue rectangular object"
    )
    assert captured["image"].mode == "RGB"
    assert captured["mask"].mode == "L"
    assert captured["query"].startswith("<image>\n")
    assert captured["kwargs"] == {
        "streaming": False,
        "temperature": 0.2,
        "top_p": 0.9,
        "num_beams": 1,
        "max_new_tokens": 256,
    }


def test_describe_normalizes_model_facts_and_injects_durable_provenance(tmp_path):
    image_path = (tmp_path / "frame.png").resolve()
    Image.new("RGB", (8, 6), (20, 40, 60)).save(image_path)
    model_output = {
        "canonical_name": "task lamp",
        "short_description": "a red metal task lamp",
        "visual_attributes": {
            "colors": ["red"],
            "materials": ["metal"],
            "shape": ["conical shade"],
            "visible_parts": ["shade", "arm"],
            "state_or_pose": ["upright"],
            "distinctive_marks": [],
            "visible_text": [],
        },
        "uncertain_or_not_visible": ["rear surface"],
        "confidence": 0.8,
        # These untrusted values must be replaced from the task envelope.
        "retrieval_text": "model-controlled retrieval",
        "evidence_snapshot_ids": ["invented"],
        "mask_source": "instance_mask",
    }

    class FakeAgent:
        def get_description(self, *_args, **_kwargs):
            return "```json\n" + worker.json.dumps(model_output) + "\n```"

    args = argparse.Namespace(
        bbox_pad_px=0.0,
        query=worker.DEFAULT_QUERY,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=256,
    )
    input_value = {
        "snapshots": [
            {
                "source_frame_asset_id": "asset-1",
                "evidence_hash": "evidence-real-1",
                "mask_source": "bbox_fallback",
                "bbox_xyxy": [1.0, 1.0, 5.0, 4.0],
                "durable_asset": {
                    "asset_id": "asset-1",
                    "access_mode": "read_only",
                    "immutable": True,
                    "path": str(image_path),
                },
            },
            {
                "evidence_hash": "evidence-real-2",
            },
        ]
    }

    normalized = worker.json.loads(worker.describe(FakeAgent(), input_value, args))
    assert normalized["canonical_name"] == "task lamp"
    assert normalized["retrieval_text"] == (
        "task lamp; a red metal task lamp; red; metal; conical shade; shade; "
        "arm; upright"
    )
    assert normalized["evidence_snapshot_ids"] == [
        "evidence-real-1",
        "evidence-real-2",
    ]
    assert normalized["mask_source"] == "bbox_fallback"
    assert normalized["raw_model_output"].startswith("```json")


def test_incomplete_model_json_gets_conservative_explicit_repair_marker():
    raw = '{"short_description":"missing required visual facts"}'
    normalized = worker.json.loads(worker.normalize_structured_output(
        raw,
        {
            "label": "test object",
            "snapshots": [
                {
                    "evidence_hash": "evidence",
                    "mask_source": "bbox_fallback",
                }
            ]
        },
    ))
    assert normalized["canonical_name"] == "test object"
    assert normalized["short_description"] == raw
    assert normalized["confidence"] == 0.0
    assert normalized["visual_attributes"] == {
        "colors": [],
        "materials": [],
        "shape": [],
        "visible_parts": [],
        "state_or_pose": [],
        "distinctive_marks": [],
        "visible_text": [],
    }
    assert normalized["_roomie_parse_path"] == "schema_repaired"
