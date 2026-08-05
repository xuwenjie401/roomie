#!/usr/bin/env python3
"""Persistent SentenceTransformer worker for Roomie's embedding IPC v1."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import struct
import sys
from typing import Any, BinaryIO


PROTOCOL = "roomie.embedding.v1"
READY_ID = 0


class ProtocolError(RuntimeError):
    """A malformed request that can be reported without killing the worker."""


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("embedding request pipe closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(stream: BinaryIO, maximum_message_bytes: int) -> bytes:
    size = struct.unpack("<I", _read_exact(stream, 4))[0]
    if size > maximum_message_bytes:
        raise ProtocolError("request exceeds maximum_message_bytes")
    return _read_exact(stream, size)


def _write_frame(stream: BinaryIO, value: dict[str, Any], maximum_message_bytes: int) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > maximum_message_bytes:
        raise ProtocolError("response exceeds maximum_message_bytes")
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)
    stream.flush()


def _error(message_id: int, operation: str, error: BaseException | str) -> dict[str, Any]:
    text = str(error).strip() or type(error).__name__
    return {
        "protocol": PROTOCOL,
        "id": message_id,
        "op": operation,
        "ok": False,
        "error": text,
    }


def _parse_request(payload: bytes, maximum_batch_size: int) -> tuple[int, list[str]]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise ProtocolError(f"invalid request JSON: {exception}") from exception
    if not isinstance(value, dict) or set(value) != {
        "protocol",
        "id",
        "op",
        "documents",
    }:
        raise ProtocolError("request does not match the embedding v1 schema")
    if value["protocol"] != PROTOCOL or value["op"] != "encode":
        raise ProtocolError("request protocol or operation is invalid")
    message_id = value["id"]
    # bool is an int subclass in Python, so exclude it explicitly.
    if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
        raise ProtocolError("request id must be a positive integer")
    documents = value["documents"]
    if (
        not isinstance(documents, list)
        or not documents
        or len(documents) > maximum_batch_size
        or any(not isinstance(document, str) for document in documents)
    ):
        raise ProtocolError("documents must be a non-empty bounded string array")
    return message_id, documents


def _normalize_rows(vectors: Any, expected_dimension: int) -> list[list[float]]:
    # Avoid relying on NumPy scalar JSON behavior at the IPC boundary.
    rows = vectors.tolist() if hasattr(vectors, "tolist") else vectors
    if not isinstance(rows, list):
        raise RuntimeError("model returned a non-array embedding result")
    normalized: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != expected_dimension:
            raise RuntimeError("model returned an embedding with wrong dimension")
        values = [float(component) for component in row]
        if any(not math.isfinite(component) for component in values):
            raise RuntimeError("model returned a non-finite embedding")
        norm = math.sqrt(sum(component * component for component in values))
        if not math.isfinite(norm) or norm <= sys.float_info.epsilon:
            raise RuntimeError("model returned a zero or invalid embedding")
        normalized.append([component / norm for component in values])
    return normalized


def _load_model(args: argparse.Namespace) -> tuple[Any, int]:
    # Import only inside startup so dependency/model errors become an explicit
    # ready response instead of arbitrary stdout or an unexplained pipe close.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device=args.device)
    dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        raise RuntimeError("model does not report a sentence embedding dimension")
    dimension = int(dimension)
    if dimension != args.expected_dimension:
        raise RuntimeError(
            f"model dimension {dimension} does not match expected "
            f"{args.expected_dimension}"
        )

    # Loading and one warmup encode happen exactly once for the process.
    warmup = model.encode(
        ["roomie embedding worker warmup"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _normalize_rows(warmup, dimension)
    return model, dimension


def _serve(args: argparse.Namespace) -> int:
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    try:
        # Third-party model code occasionally prints diagnostics. Redirect
        # Python-level stdout so it can never corrupt the framed channel.
        with contextlib.redirect_stdout(sys.stderr):
            model, dimension = _load_model(args)
    except BaseException as exception:  # startup must always be terminal and explicit
        _write_frame(
            output_stream,
            _error(READY_ID, "ready", exception),
            args.maximum_message_bytes,
        )
        return 2

    _write_frame(
        output_stream,
        {
            "protocol": PROTOCOL,
            "id": READY_ID,
            "op": "ready",
            "ok": True,
            "model_id": args.model_id,
            "dimension": dimension,
        },
        args.maximum_message_bytes,
    )

    while True:
        try:
            payload = _read_frame(input_stream, args.maximum_message_bytes)
        except EOFError:
            return 0
        except BaseException as exception:
            # No trustworthy request id exists for a broken frame, so exiting
            # is safer than inventing an acknowledgement.
            print(f"embedding worker framing error: {exception}", file=sys.stderr)
            return 3

        message_id = -1
        try:
            message_id, documents = _parse_request(payload, args.maximum_batch_size)
            with contextlib.redirect_stdout(sys.stderr):
                vectors = model.encode(
                    documents,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            normalized = _normalize_rows(vectors, dimension)
            response = {
                "protocol": PROTOCOL,
                "id": message_id,
                "op": "encode_result",
                "ok": True,
                "model_id": args.model_id,
                "dimension": dimension,
                "embeddings": normalized,
            }
        except BaseException as exception:
            # A well-framed request receives a terminal error and the worker
            # remains usable. The C++ side intentionally does not retry these.
            response = _error(message_id, "encode_result", exception)
        try:
            _write_frame(output_stream, response, args.maximum_message_bytes)
        except (BrokenPipeError, EOFError):
            return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-dimension", required=True, type=int)
    parser.add_argument("--maximum-batch-size", default=64, type=int)
    parser.add_argument("--maximum-message-bytes", default=16 * 1024 * 1024, type=int)
    args = parser.parse_args()
    if (
        args.expected_dimension <= 0
        or args.maximum_batch_size <= 0
        or args.maximum_message_bytes <= 0
    ):
        parser.error("dimension and protocol limits must be positive")
    return args


def main() -> int:
    # Keep stdout exclusively reserved for framed protocol messages.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return _serve(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
