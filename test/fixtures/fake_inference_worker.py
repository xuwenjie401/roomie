#!/usr/bin/env python3
"""Small adversarial worker used by PythonInferenceBackend shutdown tests."""

import os
import signal
import struct
import sys
import time


def mark_ready() -> None:
    marker = os.environ.get("ROOMIE_FAKE_WORKER_MARKER", "")
    if marker:
        with open(marker, "w", encoding="utf-8") as stream:
            stream.write("ready\n")


def read_exact(size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = os.read(0, remaining)
        if not chunk:
            raise EOFError("request pipe closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


expected_threshold_file = os.environ.get(
    "ROOMIE_FAKE_WORKER_EXPECT_LABEL_THRESHOLD_FILE", ""
)
if expected_threshold_file:
    flag = "--label-thresholds-file"
    if flag not in sys.argv:
        raise RuntimeError(f"missing expected worker argument: {flag}")
    value_index = sys.argv.index(flag) + 1
    if (
        value_index >= len(sys.argv)
        or sys.argv[value_index] != expected_threshold_file
    ):
        raise RuntimeError(f"unexpected value for worker argument: {flag}")


mode = os.environ.get("ROOMIE_FAKE_WORKER_MODE", "hang_after_request")
if mode == "close_stdin":
    os.close(0)
    mark_ready()
    time.sleep(60)
elif mode == "hang_after_request":
    size = struct.unpack("<I", read_exact(4))[0]
    read_exact(size)
    # Force the backend's bounded shutdown path through its SIGKILL fallback.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    mark_ready()
    while True:
        time.sleep(60)
else:
    raise RuntimeError(f"unsupported fake worker mode: {mode}")
