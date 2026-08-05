#!/usr/bin/env python3
"""Adversarial framed worker for PythonEmbeddingEncoder tests."""

from __future__ import annotations

import json
import math
import os
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Any


PROTOCOL = "roomie.embedding.v1"
MAXIMUM_MESSAGE_BYTES = 32 * 1024 * 1024


def read_exact(size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = os.read(0, size)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def read_frame() -> dict[str, Any]:
    size = struct.unpack("<I", read_exact(4))[0]
    if size > MAXIMUM_MESSAGE_BYTES:
        raise RuntimeError("oversized request")
    return json.loads(read_exact(size).decode("utf-8"))


def write_frame(value: dict[str, Any]) -> None:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    os.write(1, struct.pack("<I", len(payload)) + payload)


def touch(path: str, text: str = "ready\n") -> None:
    if path:
        Path(path).write_text(text, encoding="utf-8")


def increment_start_count() -> None:
    path = os.environ.get("ROOMIE_FAKE_EMBEDDING_START_COUNT", "")
    if not path:
        return
    counter = Path(path)
    previous = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
    counter.write_text(str(previous + 1), encoding="utf-8")


def normalized_vector(document: str) -> list[float]:
    raw = [float(len(document) + 1), float(sum(document.encode()) % 17 + 1), 2.0]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


mode = os.environ.get("ROOMIE_FAKE_EMBEDDING_MODE", "normal")
marker = os.environ.get("ROOMIE_FAKE_EMBEDDING_MARKER", "")
increment_start_count()

if mode == "hang_before_ready":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    touch(marker)
    while True:
        time.sleep(60)

write_frame(
    {
        "protocol": PROTOCOL,
        "id": 0,
        "op": "ready",
        "ok": True,
        "model_id": "fake-model-v1",
        "dimension": 3,
    }
)

if mode == "close_stdin":
    os.close(0)
    touch(marker)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(60)

while True:
    try:
        request = read_frame()
    except EOFError:
        break

    if mode == "hang_after_request":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        touch(marker)
        while True:
            time.sleep(60)

    if mode == "crash_once" and marker and not Path(marker).exists():
        touch(marker, "crashed\n")
        os._exit(17)

    request_id = request["id"]
    if mode == "worker_error":
        write_frame(
            {
                "protocol": PROTOCOL,
                "id": request_id,
                "op": "encode_result",
                "ok": False,
                "error": "synthetic model failure",
            }
        )
        continue

    documents = request["documents"]
    write_frame(
        {
            "protocol": PROTOCOL,
            "id": request_id,
            "op": "encode_result",
            "ok": True,
            "model_id": "fake-model-v1",
            "dimension": 3,
            "embeddings": [normalized_vector(document) for document in documents],
        }
    )
