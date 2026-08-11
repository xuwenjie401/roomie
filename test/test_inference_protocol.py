#!/usr/bin/env python3
"""Model-free compatibility tests for the Roomie inference IPC v2 worker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys

import pytest


ROOMIE_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOMIE_ROOT / "scripts" / "roomie_python_inference_worker.py"
SPEC = importlib.util.spec_from_file_location("roomie_python_inference_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


def provenance(request_id: int) -> dict:
    return {
        "run_id": {"high": 0x0123456789ABCDEF, "low": 0xFEDCBA9876543210},
        "frame_id": 0x1020304050607080,
        "request_id": request_id,
        "sensor_time_ns": -123456789012345,
        "map_mode": 1,
        "includes_current_frame": False,
        "causality_verified": True,
        "map": {
            "map_epoch": {
                "high": 0x1112131415161718,
                "low": 0x2122232425262728,
            },
            "map_revision": 0x3132333435363738,
            "integrated_through_ns": -998877665544,
        },
        "surface": {
            "map_epoch": {
                "high": 0x4142434445464748,
                "low": 0x5152535455565758,
            },
            "surface_revision": 0x6162636465666768,
            "source_map_revision": 0x7172737475767778,
        },
    }


def pipeline_timing() -> dict:
    return {
        "serialize_ms": 0.125,
        "pipe_write_ms": 1.25,
        "pipe_read_ms": 2.5,
        "worker_queue_ms": 5.0,
        "response_forward_ms": 10.0,
    }


def pack_run_id(value: dict) -> bytes:
    return struct.pack("<QQ", value["high"], value["low"])


def pack_provenance(value: dict) -> bytes:
    map_stamp = value["map"]
    surface_stamp = value["surface"]
    return b"".join(
        [
            pack_run_id(value["run_id"]),
            struct.pack(
                "<QQqBBB",
                value["frame_id"],
                value["request_id"],
                value["sensor_time_ns"],
                value["map_mode"],
                int(value["includes_current_frame"]),
                int(value["causality_verified"]),
            ),
            pack_run_id(map_stamp["map_epoch"]),
            struct.pack(
                "<Qq",
                map_stamp["map_revision"],
                map_stamp["integrated_through_ns"],
            ),
            pack_run_id(surface_stamp["map_epoch"]),
            struct.pack(
                "<QQ",
                surface_stamp["surface_revision"],
                surface_stamp["source_map_revision"],
            ),
        ]
    )


def pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def pack_image(width: int, height: int, channels: int, encoding: str, data: bytes) -> bytes:
    return (
        struct.pack("<iii", width, height, channels)
        + pack_string(encoding)
        + struct.pack("<I", len(data))
        + data
    )


def encode_request(request_id: int) -> bytes:
    timing = pipeline_timing()
    body = bytearray(worker.REQUEST_MAGIC)
    body.extend(struct.pack("<q", 4242424242))
    body.extend(pack_provenance(provenance(request_id)))
    body.extend(
        struct.pack(
            "<5d",
            timing["serialize_ms"],
            timing["pipe_write_ms"],
            timing["pipe_read_ms"],
            timing["worker_queue_ms"],
            timing["response_forward_ms"],
        )
    )
    body.extend(pack_string("head/color"))
    body.extend(pack_image(2, 1, 3, "rgb8", bytes([0, 1, 2, 253, 254, 255])))
    body.extend(pack_image(2, 1, 1, "mono8", bytes([0, 255])))
    body.extend(struct.pack("<ii4f", 960, 960, 501.25, 502.5, 479.75, 480.125))
    body.extend(struct.pack("<3600f", *[float(i) * 0.03125 - 1.0 for i in range(3600)]))
    body.extend(struct.pack("<iiQ", 3599, 123456, 0xF0E0D0C0B0A09080))
    body.extend(
        struct.pack(
            "<16f",
            1.0,
            0.0,
            0.0,
            1.25,
            0.0,
            1.0,
            0.0,
            -2.5,
            0.0,
            0.0,
            1.0,
            3.75,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    )
    return bytes(body)


def test_request_v2_round_trip_preserves_all_provenance() -> None:
    parsed = worker.parse_request(encode_request(9001))
    assert parsed["time_ns"] == 4242424242
    assert parsed["camera_id"] == "head/color"
    assert parsed["provenance"] == provenance(9001)
    assert parsed["pipeline_timing"] == pipeline_timing()
    assert parsed["map_version"] == 0xF0E0D0C0B0A09080

    response = worker.empty_response(parsed, ok=True)
    response_body = worker.ResponseWriter().encode_response(response)
    reader = worker.RequestReader(response_body)
    assert reader.read_bytes(5) == b"RIRS3"
    assert reader.read_struct("q") == parsed["time_ns"]
    assert reader.read_provenance() == parsed["provenance"]
    assert reader.read_pipeline_timing() == parsed["pipeline_timing"]


def test_request_v2_rejects_bad_magic_and_truncation() -> None:
    body = encode_request(1)
    with pytest.raises(worker.ProtocolError, match="magic"):
        worker.parse_request(b"RIEQ1" + body[5:])
    with pytest.raises(worker.ProtocolError, match="truncated"):
        worker.parse_request(body[:-1])


def test_request_id_distinguishes_same_camera_and_time() -> None:
    first = worker.parse_request(encode_request(1001))
    retry = worker.parse_request(encode_request(1002))
    assert first["time_ns"] == retry["time_ns"]
    assert first["camera_id"] == retry["camera_id"]
    assert first["provenance"]["sensor_time_ns"] == retry["provenance"]["sensor_time_ns"]
    assert first["provenance"]["request_id"] == 1001
    assert retry["provenance"]["request_id"] == 1002


def test_text_prompt_file_ignores_blank_lines(tmp_path: Path) -> None:
    prompt_file = tmp_path / "roomie_classes.csv"
    prompt_file.write_text("chair\n\n table \n", encoding="utf-8")

    assert worker.load_text_prompt_file(str(prompt_file)) == ["chair", "table"]


def test_text_prompt_file_rejects_empty_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "empty.csv"
    prompt_file.write_text("\n  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="text prompt file is empty"):
        worker.load_text_prompt_file(str(prompt_file))


def test_label_confidence_thresholds_load_overrides_and_fallback(
    tmp_path: Path,
) -> None:
    thresholds_file = tmp_path / "thresholds.json"
    thresholds_file.write_text(
        """{
          "owl": {
            "default": 0.25,
            "labels": {"chair": 0.2, "table": 0.75}
          }
        }""",
        encoding="utf-8",
    )

    thresholds = worker.load_label_confidence_thresholds(
        str(thresholds_file), {"owl": 0.5}
    )["owl"]
    assert worker.label_confidence_threshold(thresholds, "chair") == 0.2
    assert worker.label_confidence_threshold(thresholds, "table") == 0.75
    assert worker.label_confidence_threshold(thresholds, "lamp") == 0.25


def test_label_confidence_thresholds_keep_code_defaults_without_file() -> None:
    thresholds = worker.load_label_confidence_thresholds(
        "", {"owl": 0.25, "boxernet": 0.35}
    )
    assert worker.label_confidence_threshold(thresholds["owl"], "chair") == 0.25
    assert (
        worker.label_confidence_threshold(thresholds["boxernet"], "chair")
        == 0.35
    )


@pytest.mark.parametrize(
    "contents",
    [
        '[0.25, 0.35]',
        '{}',
        '{"owl": {"labels": {}}}',
        '{"owl": {"default": 0.25}}',
        '{"owl": {"default": 1.1, "labels": {}}}',
        '{"owl": {"default": 0.25, "labels": {"chair": -0.1}}}',
        '{"owl": {"default": 0.25, "labels": {"chair": "high"}}}',
        '{"owl": {"default": 0.25, "labels": {"chair": true}}}',
    ],
)
def test_label_confidence_thresholds_reject_invalid_values(
    tmp_path: Path, contents: str
) -> None:
    thresholds_file = tmp_path / "invalid_thresholds.json"
    thresholds_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        worker.load_label_confidence_thresholds(
            str(thresholds_file), {"owl": 0.25}
        )
