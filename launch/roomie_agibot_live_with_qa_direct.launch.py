"""Compatibility launch using the original in-process Web QA request path."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base_launch():
    path = Path(__file__).with_name("roomie_agibot_live_with_qa.launch.py")
    spec = importlib.util.spec_from_file_location("roomie_live_with_qa_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load base launch: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_launch_description():
    return _load_base_launch().generate_launch_description(
        qa_request_transport="direct"
    )
