#!/usr/bin/env python3
"""Adversarial framed-JSON peer for PythonDamWorker tests."""

import argparse
import json
import os
from pathlib import Path
import signal
import struct
import time


PROTOCOL = "roomie.dam-worker.v1"


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--max-message-bytes", type=int, default=8 * 1024 * 1024)
    args, _ = parser.parse_known_args()
    return args


def read_exact(size):
    chunks = []
    while size:
        chunk = os.read(0, size)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def read_message():
    size = struct.unpack("<I", read_exact(4))[0]
    return json.loads(read_exact(size).decode("utf-8"))


def write_message(value):
    body = json.dumps(value, separators=(",", ":")).encode("utf-8")
    data = struct.pack("<I", len(body)) + body
    offset = 0
    while offset < len(data):
        offset += os.write(1, data[offset:])


def touch(path_value):
    if path_value:
        Path(path_value).write_text("ready\n", encoding="utf-8")


def count_start(path_value):
    if path_value:
        with Path(path_value).open("a", encoding="utf-8") as stream:
            stream.write("start\n")


args = parse_args()
mode = os.environ.get("ROOMIE_FAKE_DAM_MODE", "success")
marker = os.environ.get("ROOMIE_FAKE_DAM_MARKER", "")
count_start(os.environ.get("ROOMIE_FAKE_DAM_START_COUNT", ""))

if mode == "startup_error":
    write_message(
        {
            "protocol": PROTOCOL,
            "type": "startup_error",
            "retryable": False,
            "error": "DAM Python dependency is unavailable: No module named dam",
        }
    )
    raise SystemExit(1)

if mode == "close_stdin":
    os.close(0)
    touch(marker)
    write_message({"protocol": PROTOCOL, "type": "ready", "model_id": args.model_id})
    time.sleep(60)

write_message({"protocol": PROTOCOL, "type": "ready", "model_id": args.model_id})

while True:
    try:
        request = read_message()
    except EOFError:
        break

    request_id = request.get("request_id", 0)
    if mode == "crash_once" and marker and not Path(marker).exists():
        touch(marker)
        os._exit(23)
    if mode == "hang_request":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        touch(marker)
        while True:
            time.sleep(60)

    try:
        assert request["protocol"] == PROTOCOL
        assert request["type"] == "describe"
        snapshots = request["input"]["snapshots"]
        assert snapshots
        snapshot = snapshots[0]
        asset = snapshot["durable_asset"]
        expected_id = os.environ.get("ROOMIE_FAKE_DAM_EXPECT_ASSET_ID", "")
        if expected_id:
            assert snapshot["source_frame_asset_id"] == expected_id
            assert asset["asset_id"] == expected_id
        assert asset["access_mode"] == "read_only"
        assert asset["immutable"] is True
        assert Path(asset["path"]).is_absolute()
        with Path(asset["path"]).open("rb") as stream:
            assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
        # The adapter must discard any source_path supplied by the task.
        assert "image" not in snapshot
        assert "source_path" not in snapshot
        raw_output = json.dumps(
            {
                "canonical_name": "test object",
                "short_description": f"asset {asset['asset_id'][:12]}",
                "retrieval_text": "test object from immutable pixels",
                "visual_attributes": {
                    "colors": ["blue"],
                    "materials": [],
                    "shape": [],
                    "visible_parts": [],
                    "state_or_pose": [],
                    "distinctive_marks": [],
                    "visible_text": [],
                },
                "uncertain_or_not_visible": [],
                "confidence": 0.9,
                "evidence_snapshot_ids": [
                    snapshot.get("evidence_hash")
                    or snapshot["source_frame_asset_id"]
                ],
                "mask_source": snapshot.get("mask_source", "bbox_fallback"),
            },
            separators=(",", ":"),
        )
        response = {
            "protocol": PROTOCOL,
            "type": "result",
            "request_id": request_id,
            "success": True,
            "raw_output": raw_output,
        }
    except Exception as exc:
        response = {
            "protocol": PROTOCOL,
            "type": "result",
            "request_id": request_id,
            "success": False,
            "retryable": False,
            "error": f"fake validation failed: {type(exc).__name__}: {exc}",
        }
    write_message(response)
