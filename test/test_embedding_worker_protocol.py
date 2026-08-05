#!/usr/bin/env python3
"""Protocol contract for the production SentenceTransformer worker."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys


ROOMIE_ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOMIE_ROOT / "scripts" / "roomie_sentence_transformer_worker.py"
FAKE_PACKAGE = ROOMIE_ROOT / "test" / "fixtures" / "fake_sentence_transformers"
PROTOCOL = "roomie.embedding.v1"


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise EOFError("worker closed its response pipe")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _read_frame(stream):
    size = struct.unpack("<I", _read_exact(stream, 4))[0]
    return json.loads(_read_exact(stream, size).decode("utf-8"))


def _write_frame(stream, value) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("<I", len(payload)) + payload)
    stream.flush()


def test_worker_loads_and_warms_once_and_survives_model_error(tmp_path):
    counter = tmp_path / "counter"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(FAKE_PACKAGE)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["ROOMIE_FAKE_SENTENCE_TRANSFORMER_COUNTER"] = str(counter)
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(WORKER),
            "--model",
            "fake-model",
            "--model-id",
            "fake-model-v1",
            "--device",
            "cpu",
            "--expected-dimension",
            "3",
            "--maximum-batch-size",
            "4",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        ready = _read_frame(process.stdout)
        assert ready == {
            "protocol": PROTOCOL,
            "id": 0,
            "op": "ready",
            "ok": True,
            "model_id": "fake-model-v1",
            "dimension": 3,
        }

        _write_frame(
            process.stdin,
            {
                "protocol": PROTOCOL,
                "id": 1,
                "op": "encode",
                "documents": ["chair", "lamp"],
            },
        )
        result = _read_frame(process.stdout)
        assert result["ok"] is True
        assert len(result["embeddings"]) == 2
        for vector in result["embeddings"]:
            assert math.isclose(
                math.sqrt(sum(component * component for component in vector)),
                1.0,
                abs_tol=1e-9,
            )

        _write_frame(
            process.stdin,
            {
                "protocol": PROTOCOL,
                "id": 2,
                "op": "encode",
                "documents": ["__raise__"],
            },
        )
        failure = _read_frame(process.stdout)
        assert failure == {
            "protocol": PROTOCOL,
            "id": 2,
            "op": "encode_result",
            "ok": False,
            "error": "synthetic SentenceTransformer error",
        }

        _write_frame(
            process.stdin,
            {
                "protocol": PROTOCOL,
                "id": 3,
                "op": "encode",
                "documents": ["still alive"],
            },
        )
        assert _read_frame(process.stdout)["ok"] is True
        assert counter.read_text(encoding="utf-8").splitlines() == [
            "load",
            "encode:1",
            "encode:2",
            "encode:1",
            "encode:1",
        ]
    finally:
        process.stdin.close()
        process.wait(timeout=2)
