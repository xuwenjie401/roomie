#!/usr/bin/env python3
"""Browser UI for live or explicit offline Roomie scene QA."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import threading
import time
from typing import Any
import urllib.parse
import webbrowser

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from roomie_dsg_viewer import (  # noqa: E402
    as_path,
    empty_point_cloud,
    find_auto_point_cloud,
    load_pipeline_params,
    load_point_cloud,
    points_meta,
    resolve_json_from_config,
    resolve_points_from_config,
)
from scene_qa.config import SceneQaConfig, default_qa_config_path  # noqa: E402
from scene_qa.doubao_agent import DoubaoSceneQaAgent  # noqa: E402
from scene_qa.embeddings import ObjectSearchIndex  # noqa: E402
from scene_qa.gemini_agent import (  # noqa: E402
    GeminiSceneQaAgent,
    SceneQaCancelledError,
)
from scene_qa.graph_store import GraphStore  # noqa: E402
from scene_qa.live_query import LiveSceneQueryClient, RosQuerySceneTransport  # noqa: E402
from scene_qa.session_log import (  # noqa: E402
    SceneQaSessionLog,
    default_scene_qa_log_dir,
)
from scene_qa.tools import create_default_tool_registry, create_live_tool_registry  # noqa: E402


PALETTE = [
    "#e45756",
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#b279a2",
    "#ff9da6",
    "#72b7b2",
    "#9d755d",
]

LIVE_READ_SESSION_TTL_MS = 300_000


def collect_object_ids(value: Any) -> list[int]:
    """Collect object ids from tool responses while preserving encounter order."""
    found: list[int] = []
    seen: set[int] = set()

    def add(candidate: Any) -> None:
        if isinstance(candidate, bool):
            return
        if isinstance(candidate, int):
            object_id = candidate
        elif isinstance(candidate, str) and candidate.isdigit():
            object_id = int(candidate)
        else:
            return
        if object_id not in seen:
            seen.add(object_id)
            found.append(object_id)

    def walk(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key in {"representative_objects", "rooms"}:
                    continue
                if key == "object_id":
                    add(item)
                elif key == "id" and parent_key in {"objects", "object"}:
                    add(item)
                else:
                    walk(item, key)
        elif isinstance(node, list):
            for item in node:
                walk(item, parent_key)

    walk(value)
    return found


def highlight_groups_from_history(history: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for iteration in history.get("iterations", []):
        for call in iteration.get("function_calls", []):
            object_ids = collect_object_ids(call.get("response"))
            if not object_ids:
                continue
            groups.append(
                {
                    "iteration": iteration.get("iteration"),
                    "tool": call.get("name"),
                    "args": call.get("args", {}),
                    "object_ids": object_ids,
                    "color": PALETTE[len(groups) % len(PALETTE)],
                }
            )
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-config", type=Path, default=None, help=f"scene QA JSON config (default: {default_qa_config_path()})")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        help="use the live QueryScene service (default)",
    )
    mode.add_argument(
        "--offline-json",
        "--json",
        dest="offline_json",
        type=Path,
        default=None,
        help="explicitly use a static Roomie DSG JSON",
    )
    parser.add_argument("--service", default="/roomie/query_scene", help="live QueryScene service")
    parser.add_argument("--service-timeout-sec", type=float, default=10.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="pipeline YAML used to resolve an offline point cloud",
    )
    parser.add_argument("--model", default=None, help="Gemini model; overrides QA config")
    parser.add_argument("--doubao-model", default=None, help="Doubao model; overrides QA config")
    parser.add_argument("--doubao-base-url", default=None, help="Ark API v3 base URL; overrides QA config")
    parser.add_argument(
        "--default-provider",
        choices=["gemini", "doubao"],
        default="gemini",
        help="initial provider selected in the browser",
    )
    parser.add_argument("--embedding-model", type=Path, default=None, help="local SentenceTransformer checkpoint")
    parser.add_argument("--embedding-backend", choices=["embedding", "lexical"], default=None)
    parser.add_argument("--device", default=None, help="embedding device: auto, cuda, or cpu")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--points", type=Path, default=None, help="RGB point cloud: .nvblox, .ply, .pcd, .npy, .npz, or JSON")
    parser.add_argument("--no-points", action="store_true", help="disable point cloud loading")
    parser.add_argument("--max-points", type=int, default=700000, help="point cloud downsample limit")
    parser.add_argument("--surface-threshold-m", type=float, default=0.0, help="nvblox surface distance threshold; 0 uses 1.5 voxels")
    parser.add_argument("--min-tsdf-weight", type=float, default=1.0e-4, help="nvblox TSDF weight gate")
    parser.add_argument("--point-size", type=float, default=2.5, help="initial WebGL point size")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8776)
    parser.add_argument(
        "--browser",
        choices=["auto", "off"],
        default="auto",
        help="open the system browser after binding the server (default: auto)",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--api-key", default=None, help="Gemini API key; defaults to GEMINI_API_KEY or GOOGLE_API_KEY")
    parser.add_argument(
        "--doubao-api-key",
        default=None,
        help="Doubao key; defaults to DOUBAO_API_KEY or ARK_API_KEY",
    )
    return parser.parse_args()


def load_qa_config(path_arg: Path | None) -> SceneQaConfig:
    path = as_path(path_arg) if path_arg is not None else default_qa_config_path()
    if path is not None and path.exists():
        return SceneQaConfig.from_json(path)
    return SceneQaConfig()


def resolve_graph_json(args: argparse.Namespace, config: SceneQaConfig) -> Path:
    explicit = as_path(args.offline_json)
    if explicit is not None:
        return explicit
    configured = config.graph_json.expanduser()
    if str(configured):
        return configured.resolve()
    params = load_pipeline_params(as_path(args.config))
    resolved = resolve_json_from_config(params)
    if resolved is not None and resolved.exists():
        described = resolved.with_name(f"{resolved.stem}.dam_described{resolved.suffix}")
        if described.exists():
            return described
        return resolved
    return config.resolved_graph_json()


def apply_cli_overrides(config: SceneQaConfig, args: argparse.Namespace) -> SceneQaConfig:
    return SceneQaConfig(
        graph_json=(
            resolve_graph_json(args, config)
            if args.offline_json is not None
            else config.graph_json
        ),
        gemini_model=args.model or config.gemini_model,
        doubao_model=args.doubao_model or config.doubao_model,
        doubao_base_url=(args.doubao_base_url or config.doubao_base_url).rstrip("/"),
        doubao_thinking_type=config.doubao_thinking_type,
        embedding_model=args.embedding_model or config.embedding_model,
        embedding_backend=args.embedding_backend or config.embedding_backend,
        device=args.device or config.device,
        top_k=args.top_k if args.top_k is not None else config.top_k,
        max_iterations=args.max_iterations if args.max_iterations is not None else config.max_iterations,
        temperature=args.temperature if args.temperature is not None else config.temperature,
        max_output_tokens=args.max_output_tokens if args.max_output_tokens is not None else config.max_output_tokens,
        snapshot_max_side_px=config.snapshot_max_side_px,
        snapshot_bbox_pad_px=config.snapshot_bbox_pad_px,
        system_prompt_path=config.system_prompt_path,
        _config_dir=config._config_dir,
    )


def graph_payload(graph: GraphStore) -> dict[str, Any]:
    return {
        "format": graph.graph.get("format"),
        "world_frame": graph.graph.get("world_frame"),
        "saved_time": graph.graph.get("saved_time"),
        "objects": [graph.object_to_dict(obj) for obj in graph.object_records()],
        "rooms": [graph.room_to_dict(room) for room in graph.room_records()],
        "snapshot_images": graph.graph.get("snapshot_images", []),
    }


def live_graph_payload(service_name: str) -> dict[str, Any]:
    """Return an empty graph shell that the browser enriches from live tool results."""

    return {
        "format": "roomie_live_query",
        "world_frame": "map",
        "saved_time": None,
        "objects": [],
        "rooms": [],
        "snapshot_images": [],
        "live": True,
        "service": service_name,
    }


class ProgressLog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._counter = 0

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._counter = 0

    def add(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._counter += 1
            item = {
                "id": self._counter,
                "time_s": round(time.time(), 3),
                **event,
            }
            self._events.append(item)
            if len(self._events) > 240:
                self._events = self._events[-240:]
            return item

    def snapshot(self, after: int = 0) -> dict[str, Any]:
        with self._lock:
            events = [event for event in self._events if int(event.get("id", 0)) > after]
            return {
                "events": events,
                "last_event_id": self._counter,
            }


class QaRuntime:
    """Resettable agents and read-session state for one QA generation."""

    def __init__(
        self,
        generation: int,
        scene_log: SceneQaSessionLog,
        progress_log: ProgressLog,
    ) -> None:
        self.generation = int(generation)
        self.scene_log = scene_log
        self.progress_log = progress_log
        self.qa_lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.agents: dict[str, Any] = {}
        self.provider_status: dict[str, dict[str, Any]] = {}
        self.retired = False
        self._state_lock = threading.Lock()
        self._active_turn: int | None = None
        self._active_started_time_s: float | None = None
        self._active_progress_after = 0

    def activate_turn(
        self,
        turn_index: int,
        *,
        started_time_s: float,
        progress_after: int,
    ) -> None:
        with self._state_lock:
            self._active_turn = int(turn_index)
            self._active_started_time_s = float(started_time_s)
            self._active_progress_after = int(progress_after)

    def checkpoint_history(self, history: dict[str, Any]) -> None:
        with self._state_lock:
            turn_index = self._active_turn
            progress_after = self._active_progress_after
        if turn_index is None:
            return
        progress = self.progress_log.snapshot(progress_after)
        self.scene_log.checkpoint_turn(
            turn_index,
            history=history,
            progress_events=progress["events"],
        )

    def add_progress(self, event: dict[str, Any]) -> dict[str, Any]:
        if self.retired:
            return event
        return self.progress_log.add({"generation": self.generation, **event})

    def finish_turn(
        self,
        *,
        response: dict[str, Any] | None = None,
        history: dict[str, Any] | None = None,
        error: BaseException | None = None,
        status: str | None = None,
    ) -> bool:
        with self._state_lock:
            turn_index = self._active_turn
            started = self._active_started_time_s
            progress_after = self._active_progress_after
            self._active_turn = None
            self._active_started_time_s = None
        if turn_index is None or started is None:
            return False
        progress = self.progress_log.snapshot(progress_after)
        return self.scene_log.finish_turn(
            turn_index,
            started_time_s=started,
            response=response,
            history=history,
            error=error,
            status=status,
            progress_events=progress["events"],
        )

    def cancel(self) -> None:
        self.cancel_event.set()
        self.finish_turn(
            error=SceneQaCancelledError("question was reset from the browser"),
            status="reset",
        )

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Roomie Scene QA</title>
  <style>
    :root {
      --bg: #f7f8f5;
      --panel: #ffffff;
      --canvas: #edf0e9;
      --text: #1d211f;
      --muted: #69736d;
      --line: #d7ddd4;
      --accent: #176b60;
      --accent-2: #2f6f9f;
      --warn: #d96c2c;
      --answer: #f0f7f4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100vh;
      color: var(--text);
      background: var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }
    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: minmax(320px, 390px) minmax(460px, 1fr) minmax(360px, 430px);
      min-width: 0;
      min-height: 0;
    }
    aside, main {
      min-width: 0;
      min-height: 0;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }
    main { position: relative; background: var(--canvas); }
    header {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    h1, h2, h3 {
      margin: 0;
      font-weight: 650;
      letter-spacing: 0;
    }
    h1 { font-size: 16px; }
    h2 { font-size: 15px; }
    h3 { font-size: 13px; color: var(--muted); text-transform: uppercase; }
    .meta {
      margin-top: 6px;
      font-size: 12px;
      line-height: 1.35;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .qa {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      height: 100%;
    }
    .ask-box {
      display: grid;
      gap: 9px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }
    textarea {
      width: 100%;
      min-height: 96px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 10px;
      font: inherit;
      line-height: 1.4;
      color: var(--text);
      background: #fff;
    }
    .provider-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      align-items: center;
      gap: 9px;
    }
    .provider-toggle {
      display: inline-grid;
      grid-template-columns: repeat(2, minmax(74px, 1fr));
      padding: 3px;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: #f3f5f1;
    }
    button.provider-button {
      height: 29px;
      border: 0;
      background: transparent;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    button.provider-button.active {
      background: #fff;
      color: var(--accent);
      box-shadow: 0 1px 4px rgba(24, 37, 31, .14);
    }
    .provider-status {
      min-width: 0;
      color: var(--muted);
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--text);
      font: inherit;
      cursor: pointer;
    }
    button.primary {
      border-color: #1d7569;
      background: #176b60;
      color: #fff;
      font-weight: 650;
    }
    .ask-actions { display: grid; grid-template-columns: 1fr auto; gap: 8px; }
    .reset-button { color: var(--warn); border-color: #dfb08f; min-width: 92px; }
    button:disabled { opacity: .55; cursor: default; }
    .answer {
      margin: 12px 14px;
      padding: 12px;
      border: 1px solid #b8d8ce;
      border-radius: 8px;
      background: var(--answer);
    }
    .answer .label {
      font-size: 12px;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 7px;
    }
    .answer .text {
      font-size: 18px;
      line-height: 1.42;
      font-weight: 650;
      white-space: pre-wrap;
    }
    .reasoning {
      margin-top: 10px;
      color: #3c514b;
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .progress-list {
      display: none;
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      background: #fbfcfa;
      max-height: 176px;
      overflow: auto;
    }
    .progress-list.active { display: block; }
    .progress-item {
      display: grid;
      grid-template-columns: 10px minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      padding: 5px 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .progress-dot {
      width: 7px;
      height: 7px;
      margin-top: 4px;
      border-radius: 999px;
      background: #a9b4ad;
    }
    .progress-item.running .progress-dot { background: var(--accent-2); }
    .progress-item.error .progress-dot { background: var(--warn); }
    .scroll {
      overflow: auto;
      min-height: 0;
      padding: 0 14px 16px;
    }
    .trace-panel {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      height: 100%;
    }
    .trace-list {
      overflow: auto;
      padding: 12px;
    }
    .tool-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 10px;
      background: #fff;
    }
    .tool-card.active {
      border-color: var(--accent-2);
      box-shadow: 0 0 0 2px rgba(47, 111, 159, .12);
    }
    .tool-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      font-size: 13px;
      font-weight: 650;
    }
    .swatch {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 2px;
      margin-right: 6px;
      vertical-align: -1px;
    }
    .chip-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      min-height: 24px;
      padding: 3px 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f8faf7;
      font-size: 12px;
      color: var(--text);
      cursor: pointer;
    }
    .chip:hover { border-color: var(--accent); }
    pre {
      margin: 0;
      max-height: 180px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #33413c;
      background: #f6f8f5;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px;
    }
    details { margin-top: 8px; }
    summary { color: var(--accent); cursor: pointer; font-size: 12px; }
    .evidence-title {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .description-list {
      display: grid;
      gap: 6px;
      margin-top: 6px;
      font-size: 12px;
      line-height: 1.35;
      color: #33413c;
    }
    .description-item {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 7px;
      background: #f8faf7;
    }
    .snapshot-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(116px, 1fr));
      gap: 8px;
      margin-top: 6px;
    }
    .snapshot-thumb {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: hidden;
      background: #f8faf7;
    }
    .snapshot-frame {
      position: relative;
      width: 100%;
      background: #e9ede7;
    }
    .snapshot-frame img {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .snapshot-box {
      position: absolute;
      border: 2px solid #e45756;
      box-shadow: 0 0 0 1px rgba(255,255,255,.8);
      pointer-events: none;
    }
    .snapshot-caption {
      padding: 5px 6px;
      font-size: 11px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
    }
    canvas.dragging { cursor: grabbing; }
    .hud {
      position: absolute;
      left: 12px;
      bottom: 12px;
      max-width: calc(100% - 24px);
      padding: 7px 9px;
      border: 1px solid rgba(216, 221, 213, .9);
      border-radius: 7px;
      background: rgba(255, 255, 255, .88);
      color: var(--muted);
      font-size: 12px;
      backdrop-filter: blur(6px);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .object-bar {
      position: absolute;
      top: 12px;
      left: 12px;
      right: 255px;
      display: flex;
      gap: 8px;
      align-items: center;
      pointer-events: none;
    }
    .scene-controls {
      position: absolute;
      top: 12px;
      right: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 8px;
      border: 1px solid rgba(216, 221, 213, .9);
      border-radius: 7px;
      background: rgba(255, 255, 255, .9);
      backdrop-filter: blur(6px);
      color: var(--muted);
      font-size: 12px;
    }
    .scene-controls label {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      white-space: nowrap;
    }
    .scene-controls input[type="range"] {
      width: 72px;
    }
    .object-pill {
      padding: 6px 9px;
      border-radius: 7px;
      border: 1px solid rgba(216, 221, 213, .9);
      background: rgba(255, 255, 255, .9);
      font-size: 12px;
      color: var(--muted);
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .object-list {
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 12px;
    }
    .object-row {
      width: 100%;
      display: grid;
      grid-template-columns: 46px minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      min-height: 38px;
      padding: 6px 7px;
      border: 1px solid transparent;
      border-radius: 7px;
      text-align: left;
      background: transparent;
    }
    .object-row.highlighted { background: #fff4e6; border-color: #eab676; }
    .object-row.selected { background: #e8f4f1; border-color: #8bc2b8; }
    .id { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
    .name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .room { color: var(--muted); font-size: 12px; }
    .empty {
      padding: 18px 0;
      color: var(--muted);
      font-size: 14px;
    }
    @media (max-width: 1180px) {
      .app { grid-template-columns: minmax(300px, 360px) 1fr; }
      .trace-panel {
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        width: min(430px, 45vw);
        z-index: 4;
        box-shadow: -10px 0 24px rgba(0,0,0,.08);
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="qa">
      <header>
        <h1>Roomie Scene QA</h1>
        <div class="meta" id="graphMeta">Loading scene...</div>
      </header>
      <div class="ask-box">
        <textarea id="question" placeholder="Ask about objects, rooms, spatial relations, or visual details."></textarea>
        <div class="provider-row">
          <div class="provider-toggle" role="group" aria-label="Model provider">
            <button class="provider-button active" type="button" data-provider="gemini" aria-pressed="true">Gemini</button>
            <button class="provider-button" type="button" data-provider="doubao" aria-pressed="false">Doubao</button>
          </div>
          <div class="provider-status" id="providerStatus">Loading providers...</div>
        </div>
        <div class="ask-actions">
          <button id="askBtn" class="primary">Ask</button>
          <button id="resetBtn" class="reset-button" type="button">Reset QA</button>
        </div>
      </div>
      <div class="scroll">
        <div class="answer" id="answerBox">
          <div class="label" id="answerLabel">Final answer</div>
          <div class="text" id="answerText">No question asked yet.</div>
          <div class="reasoning" id="reasoningText"></div>
        </div>
        <h3>Highlighted Objects</h3>
        <div id="highlightSummary" class="meta">Tool results will appear here.</div>
        <div class="object-list" id="objectList"></div>
      </div>
    </aside>
    <main>
      <canvas id="view"></canvas>
      <div class="object-bar"><div class="object-pill" id="selectedInfo">No object selected</div></div>
      <div class="scene-controls">
        <label><input type="checkbox" id="showPoints" checked> Points</label>
        <label><input type="checkbox" id="showRooms" checked> Rooms</label>
        <label>Size <input type="range" id="pointSize" min="1" max="7" step="0.5"></label>
      </div>
      <div class="hud" id="hud"></div>
    </main>
    <aside class="trace-panel">
      <header>
        <h2>Tool Trace</h2>
        <div class="meta">Visible model text, local tool calls, and object results. Hidden model chain-of-thought is not exposed.</div>
      </header>
      <div class="progress-list" id="progressList"></div>
      <div class="trace-list" id="traceList"></div>
    </aside>
  </div>
  <script>
    const DEFAULT_POINT_SIZE = __POINT_SIZE__;
    const state = {
      graph: null,
      objects: [],
      rooms: [],
      selectedId: null,
      highlightedIds: new Set(),
      highlightGroups: [],
      activeGroup: null,
      progressEvents: [],
      lastEventId: 0,
      pollTimer: null,
      asking: false,
      askController: null,
      provider: 'gemini',
      yaw: 2.45,
      pitch: 0.72,
      distance: 8,
      target: [0, 0, 0],
      radius: 2,
      dragging: false,
      moved: false,
      lastX: 0,
      lastY: 0,
      screenObjects: [],
      gl: null,
      pointCount: 0,
      pointSource: '',
      pointBounds: null,
      pointBuffersReady: false,
      pointProgram: null,
      lineProgram: null,
      pointPositionBuffer: null,
      pointColorBuffer: null,
      boxPositionBuffer: null,
      boxColorBuffer: null,
      boxVertexCount: 0,
      roomPositionBuffer: null,
      roomColorBuffer: null,
      roomVertexCount: 0,
      viewProj: null
    };

    const canvas = document.getElementById('view');
    const questionInput = document.getElementById('question');
    const askBtn = document.getElementById('askBtn');
    const resetBtn = document.getElementById('resetBtn');
    const objectList = document.getElementById('objectList');
    const traceList = document.getElementById('traceList');
    const progressList = document.getElementById('progressList');
    const providerStatus = document.getElementById('providerStatus');
    const providerButtons = Array.from(document.querySelectorAll('[data-provider]'));
    const showPoints = document.getElementById('showPoints');
    const showRooms = document.getElementById('showRooms');
    const pointSize = document.getElementById('pointSize');
    pointSize.value = String(DEFAULT_POINT_SIZE);

    function escapeHtml(text) {
      return String(text ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }
    function providerInfo(name) {
      return state.graph?.qa?.providers?.[name] || null;
    }
    function selectProvider(name) {
      const info = providerInfo(name);
      if (!info || !info.available || state.asking) return;
      state.provider = name;
      for (const button of providerButtons) {
        const active = button.dataset.provider === name;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', active ? 'true' : 'false');
      }
      providerStatus.textContent = `${info.label || name} · ${info.model || ''}`;
      document.getElementById('answerLabel').textContent = `Final answer · ${info.label || name}`;
    }
    function updateProviderControls() {
      for (const button of providerButtons) {
        const info = providerInfo(button.dataset.provider);
        button.disabled = state.asking || !info?.available;
        button.title = info?.available ? (info.model || '') : (info?.error || 'Provider unavailable');
      }
      const preferred = providerInfo(state.provider)?.available
        ? state.provider
        : (state.graph?.qa?.default_provider || state.provider);
      const available = providerInfo(preferred)?.available
        ? preferred
        : providerButtons.map(button => button.dataset.provider).find(name => providerInfo(name)?.available);
      if (available) selectProvider(available);
      else providerStatus.textContent = 'No model provider is available';
    }
    function fmt(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
      return Number(value).toFixed(digits).replace(/\.?0+$/, '');
    }
    function vec3(value) {
      if (!Array.isArray(value) || value.length < 3) return [0, 0, 0];
      return [Number(value[0]) || 0, Number(value[1]) || 0, Number(value[2]) || 0];
    }
    function hexToRgb(hex) {
      const value = String(hex || '#e45756').replace('#', '');
      return [parseInt(value.slice(0, 2), 16), parseInt(value.slice(2, 4), 16), parseInt(value.slice(4, 6), 16)];
    }
    function hslToRgb(h, s, l) {
      h /= 360; s /= 100; l /= 100;
      const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1; if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      return [
        Math.round(hue2rgb(p, q, h + 1 / 3) * 255),
        Math.round(hue2rgb(p, q, h) * 255),
        Math.round(hue2rgb(p, q, h - 1 / 3) * 255)
      ];
    }
    function colorForLabel(label, id) {
      const text = `${label || 'object'}:${id}`;
      let hash = 2166136261;
      for (let i = 0; i < text.length; ++i) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
      }
      return hslToRgb(Math.abs(hash) % 360, 48, 45);
    }
    function objectColor(object) {
      if (object.object_id === state.selectedId) return [18, 20, 22];
      const group = state.highlightGroups.find(g => g.object_ids.includes(object.object_id));
      if (group) return hexToRgb(group.color);
      return colorForLabel(object.label, object.object_id).map(v => Math.round(v * 0.72 + 255 * 0.28));
    }
    function mat4Multiply(a, b) {
      const out = new Array(16);
      for (let c = 0; c < 4; ++c) {
        for (let r = 0; r < 4; ++r) {
          out[c * 4 + r] =
            a[0 * 4 + r] * b[c * 4 + 0] +
            a[1 * 4 + r] * b[c * 4 + 1] +
            a[2 * 4 + r] * b[c * 4 + 2] +
            a[3 * 4 + r] * b[c * 4 + 3];
        }
      }
      return out;
    }
    function mat4Perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2);
      const nf = 1 / (near - far);
      return [f / aspect,0,0,0, 0,f,0,0, 0,0,(far + near) * nf,-1, 0,0,2 * far * near * nf,0];
    }
    function normalize(v) {
      const len = Math.hypot(v[0], v[1], v[2]) || 1;
      return [v[0] / len, v[1] / len, v[2] / len];
    }
    function cross(a, b) {
      return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
    }
    function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
    function mat4LookAt(eye, center, up) {
      const z = normalize([eye[0]-center[0], eye[1]-center[1], eye[2]-center[2]]);
      const x = normalize(cross(up, z));
      const y = cross(z, x);
      return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,eye),-dot(y,eye),-dot(z,eye),1];
    }
    function currentEye() {
      const cp = Math.cos(state.pitch);
      const sp = Math.sin(state.pitch);
      return [
        state.target[0] + state.distance * cp * Math.cos(state.yaw),
        state.target[1] + state.distance * cp * Math.sin(state.yaw),
        state.target[2] + state.distance * sp
      ];
    }
    function currentViewProj() {
      const rect = canvas.getBoundingClientRect();
      const view = mat4LookAt(currentEye(), state.target, [0, 0, 1]);
      const proj = mat4Perspective(45 * Math.PI / 180, Math.max(1e-6, rect.width / rect.height), 0.01, Math.max(100, state.radius * 80));
      return mat4Multiply(proj, view);
    }
    function projectPoint(point, viewProj) {
      const x = point[0], y = point[1], z = point[2];
      const cx = viewProj[0]*x + viewProj[4]*y + viewProj[8]*z + viewProj[12];
      const cy = viewProj[1]*x + viewProj[5]*y + viewProj[9]*z + viewProj[13];
      const cz = viewProj[2]*x + viewProj[6]*y + viewProj[10]*z + viewProj[14];
      const cw = viewProj[3]*x + viewProj[7]*y + viewProj[11]*z + viewProj[15];
      const rect = canvas.getBoundingClientRect();
      const invW = cw ? 1 / cw : 1;
      return {x: (cx * invW * .5 + .5) * rect.width, y: (1 - (cy * invW * .5 + .5)) * rect.height, z: cz * invW};
    }
    function createShader(gl, type, source) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    }
    function createProgram(gl, vertex, fragment) {
      const program = gl.createProgram();
      gl.attachShader(program, createShader(gl, gl.VERTEX_SHADER, vertex));
      gl.attachShader(program, createShader(gl, gl.FRAGMENT_SHADER, fragment));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
      return program;
    }
    function initGl() {
      const gl = canvas.getContext('webgl', {antialias: true});
      if (!gl) throw new Error('WebGL is not available');
      state.gl = gl;
      const vertex = `
        attribute vec3 a_position;
        attribute vec3 a_color;
        uniform mat4 u_viewProj;
        uniform float u_pointSize;
        varying vec3 v_color;
        void main() {
          gl_Position = u_viewProj * vec4(a_position, 1.0);
          gl_PointSize = u_pointSize;
          v_color = a_color / 255.0;
        }`;
      const pointFragment = `
        precision mediump float;
        varying vec3 v_color;
        void main() {
          vec2 uv = gl_PointCoord - vec2(0.5);
          if (dot(uv, uv) > 0.25) discard;
          gl_FragColor = vec4(v_color, 1.0);
        }`;
      const lineFragment = `precision mediump float; varying vec3 v_color; void main() { gl_FragColor = vec4(v_color, 1.0); }`;
      state.pointProgram = createProgram(gl, vertex, pointFragment);
      state.lineProgram = createProgram(gl, vertex, lineFragment);
      state.pointPositionBuffer = gl.createBuffer();
      state.pointColorBuffer = gl.createBuffer();
      state.boxPositionBuffer = gl.createBuffer();
      state.boxColorBuffer = gl.createBuffer();
      state.roomPositionBuffer = gl.createBuffer();
      state.roomColorBuffer = gl.createBuffer();
      gl.clearColor(0.929, 0.941, 0.925, 1);
    }
    function setProgramAttributes(program, positionBuffer, colorBuffer) {
      const gl = state.gl;
      gl.useProgram(program);
      const posLoc = gl.getAttribLocation(program, 'a_position');
      const colorLoc = gl.getAttribLocation(program, 'a_color');
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
      gl.enableVertexAttribArray(colorLoc);
      gl.vertexAttribPointer(colorLoc, 3, gl.UNSIGNED_BYTE, false, 0, 0);
      gl.uniformMatrix4fv(gl.getUniformLocation(program, 'u_viewProj'), false, new Float32Array(state.viewProj));
      const pointSizeLoc = gl.getUniformLocation(program, 'u_pointSize');
      if (pointSizeLoc) gl.uniform1f(pointSizeLoc, Number(pointSize.value) || DEFAULT_POINT_SIZE);
    }
    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const w = Math.max(1, Math.floor(rect.width * dpr));
      const h = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      state.gl.viewport(0, 0, w, h);
    }
    function objectCorners(object) {
      const c = vec3(object.center_world);
      const s = vec3(object.size_m);
      const yaw = Number(object.yaw_rad) || 0;
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const hx = Math.max(0.04, s[0] * .5), hy = Math.max(0.04, s[1] * .5), hz = Math.max(0.04, s[2] * .5);
      const corners = [];
      for (const ix of [-1,1]) for (const iy of [-1,1]) for (const iz of [-1,1]) {
        const x = ix * hx, y = iy * hy;
        corners.push([c[0] + cy * x - sy * y, c[1] + sy * x + cy * y, c[2] + iz * hz]);
      }
      return corners;
    }
    function roomCorners(room) {
      let c = vec3(room.center_world);
      let s = vec3(room.size_m);
      if ((!Array.isArray(room.center_world) || !Array.isArray(room.size_m)) && Array.isArray(room.min_xy) && Array.isArray(room.max_xy)) {
        const minX = Number(room.min_xy[0]) || 0;
        const minY = Number(room.min_xy[1]) || 0;
        const maxX = Number(room.max_xy[0]) || 0;
        const maxY = Number(room.max_xy[1]) || 0;
        c = [(minX + maxX) * .5, (minY + maxY) * .5, 1.25];
        s = [Math.abs(maxX - minX), Math.abs(maxY - minY), 2.5];
      }
      const hx = Math.max(0.04, s[0] * .5), hy = Math.max(0.04, s[1] * .5), hz = Math.max(0.04, s[2] * .5);
      const corners = [];
      for (const ix of [-1,1]) for (const iy of [-1,1]) for (const iz of [-1,1]) {
        corners.push([c[0] + ix * hx, c[1] + iy * hy, c[2] + iz * hz]);
      }
      return corners;
    }
    const edges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
    function appendBox(lines, colors, object) {
      const corners = objectCorners(object);
      const color = objectColor(object);
      const repeat = state.highlightedIds.has(object.object_id) || object.object_id === state.selectedId ? 2 : 1;
      for (let r = 0; r < repeat; ++r) {
        for (const [a, b] of edges) {
          lines.push(...corners[a], ...corners[b]);
          colors.push(...color, ...color);
        }
      }
    }
    function appendRoomBox(lines, colors, room) {
      const corners = roomCorners(room);
      const color = [68, 110, 150];
      for (let r = 0; r < 2; ++r) {
        for (const [a, b] of edges) {
          lines.push(...corners[a], ...corners[b]);
          colors.push(...color, ...color);
        }
      }
    }
    function uploadBoxes() {
      const lines = [], colors = [];
      for (const object of state.objects) appendBox(lines, colors, object);
      state.boxVertexCount = lines.length / 3;
      const gl = state.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, state.boxPositionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, state.boxColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(colors), gl.DYNAMIC_DRAW);
    }
    function uploadRoomBoxes() {
      const lines = [], colors = [];
      for (const room of state.rooms) {
        const hasBox = (Array.isArray(room.center_world) && Array.isArray(room.size_m)) ||
          (Array.isArray(room.min_xy) && Array.isArray(room.max_xy));
        if (hasBox) appendRoomBox(lines, colors, room);
      }
      state.roomVertexCount = lines.length / 3;
      const gl = state.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, state.roomPositionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, state.roomColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(colors), gl.STATIC_DRAW);
    }
    function updateScreenObjects() {
      const rect = canvas.getBoundingClientRect();
      state.screenObjects = [];
      for (const object of state.objects) {
        const projected = objectCorners(object).map(p => projectPoint(p, state.viewProj));
        const visible = projected.some(p => p.z > -1 && p.z < 1);
        if (!visible) continue;
        const xs = projected.map(p => p.x), ys = projected.map(p => p.y);
        const minX = Math.max(0, Math.min(...xs)), maxX = Math.min(rect.width, Math.max(...xs));
        const minY = Math.max(0, Math.min(...ys)), maxY = Math.min(rect.height, Math.max(...ys));
        if (maxX <= minX || maxY <= minY) continue;
        state.screenObjects.push({id: object.object_id, minX, maxX, minY, maxY, area: (maxX-minX)*(maxY-minY), depth: projected.reduce((a,p)=>a+p.z,0)/projected.length});
      }
    }
    function render() {
      if (!state.gl) return;
      resizeCanvas();
      state.viewProj = currentViewProj();
      uploadBoxes();
      updateScreenObjects();
      const gl = state.gl;
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      if (showPoints.checked && state.pointBuffersReady && state.pointCount > 0) {
        setProgramAttributes(state.pointProgram, state.pointPositionBuffer, state.pointColorBuffer);
        gl.drawArrays(gl.POINTS, 0, state.pointCount);
      }
      gl.disable(gl.DEPTH_TEST);
      if (showRooms.checked && state.roomVertexCount > 0) {
        setProgramAttributes(state.lineProgram, state.roomPositionBuffer, state.roomColorBuffer);
        gl.drawArrays(gl.LINES, 0, state.roomVertexCount);
      }
      setProgramAttributes(state.lineProgram, state.boxPositionBuffer, state.boxColorBuffer);
      gl.drawArrays(gl.LINES, 0, state.boxVertexCount);
      const pointText = state.pointCount > 0 ? `${state.pointCount.toLocaleString()} points` : 'no point cloud';
      document.getElementById('hud').textContent = `${state.objects.length} objects | ${state.rooms.length} rooms | ${state.highlightedIds.size} highlighted | ${pointText}`;
      updateSelectedInfo();
    }
    async function loadPointCloud() {
      const metaResponse = await fetch('/points-meta.json');
      const meta = await metaResponse.json();
      state.pointCount = meta.count || 0;
      state.pointSource = meta.source || '';
      state.pointBounds = meta.bounds || null;
      if (!state.pointCount) return;
      const [pointData, colorData] = await Promise.all([
        fetch('/points.bin').then(r => r.arrayBuffer()),
        fetch('/colors.bin').then(r => r.arrayBuffer())
      ]);
      const gl = state.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, state.pointPositionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pointData), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, state.pointColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(colorData), gl.STATIC_DRAW);
      state.pointBuffersReady = true;
    }
    function computeBounds() {
      let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
      const include = (lo, hi) => {
        for (let i = 0; i < 3; ++i) { min[i] = Math.min(min[i], lo[i]); max[i] = Math.max(max[i], hi[i]); }
      };
      if (state.pointBounds) {
        include(state.pointBounds.min, state.pointBounds.max);
      }
      for (const room of state.rooms) {
        const hasBox = (Array.isArray(room.center_world) && Array.isArray(room.size_m)) ||
          (Array.isArray(room.min_xy) && Array.isArray(room.max_xy));
        if (!hasBox) continue;
        const corners = roomCorners(room);
        include([
          Math.min(...corners.map(p => p[0])),
          Math.min(...corners.map(p => p[1])),
          Math.min(...corners.map(p => p[2]))
        ], [
          Math.max(...corners.map(p => p[0])),
          Math.max(...corners.map(p => p[1])),
          Math.max(...corners.map(p => p[2]))
        ]);
      }
      for (const object of state.objects) {
        const c = vec3(object.center_world), s = vec3(object.size_m);
        include([c[0]-s[0]*.5, c[1]-s[1]*.5, c[2]-s[2]*.5], [c[0]+s[0]*.5, c[1]+s[1]*.5, c[2]+s[2]*.5]);
      }
      if (!Number.isFinite(min[0])) { min = [-1,-1,-1]; max = [1,1,1]; }
      state.target = [(min[0]+max[0])*.5, (min[1]+max[1])*.5, (min[2]+max[2])*.5];
      state.radius = Math.max(1, Math.hypot(max[0]-min[0], max[1]-min[1], max[2]-min[2])*.5);
      state.distance = state.radius * 2.7;
    }
    function setHighlights(groups) {
      state.highlightGroups = groups || [];
      state.highlightedIds = new Set();
      for (const group of state.highlightGroups) for (const id of group.object_ids || []) state.highlightedIds.add(Number(id));
      renderObjectList();
      renderTraceGroups();
      render();
    }
    function selectObject(id) {
      state.selectedId = Number(id);
      renderObjectList();
      render();
    }
    function objectLabel(id) {
      const object = state.objects.find(o => o.object_id === Number(id));
      return object ? `${object.label || 'object'} #${object.object_id}` : `#${id}`;
    }
    function roomText(object) {
      return (object.rooms || []).map(r => r.label).join(', ');
    }
    function updateSelectedInfo() {
      const object = state.objects.find(o => o.object_id === state.selectedId);
      document.getElementById('selectedInfo').textContent = object
        ? `#${object.object_id} ${object.label || 'object'} | ${roomText(object)} | ${vec3(object.center_world).map(v => fmt(v)).join(', ')}`
        : 'No object selected';
    }
    function renderObjectList() {
      const highlighted = state.objects.filter(o => state.highlightedIds.has(o.object_id));
      const rows = (highlighted.length ? highlighted : state.objects.slice(0, 25)).map(object => {
        const cls = [
          'object-row',
          state.highlightedIds.has(object.object_id) ? 'highlighted' : '',
          object.object_id === state.selectedId ? 'selected' : ''
        ].filter(Boolean).join(' ');
        return `<button class="${cls}" data-id="${object.object_id}">
          <span class="id">#${object.object_id}</span>
          <span class="name">${escapeHtml(object.label || 'object')}</span>
          <span class="room">${escapeHtml(roomText(object))}</span>
        </button>`;
      }).join('');
      objectList.innerHTML = rows || '<div class="empty">No objects.</div>';
      objectList.querySelectorAll('button[data-id]').forEach(btn => btn.addEventListener('click', () => selectObject(btn.dataset.id)));
      document.getElementById('highlightSummary').textContent = highlighted.length
        ? `${highlighted.length} objects are highlighted from tool results.`
        : 'No highlighted tool objects yet.';
    }
    function responseObjectsFromCall(call) {
      const objects = [];
      function walk(node) {
        if (!node) return;
        if (Array.isArray(node)) { node.forEach(walk); return; }
        if (typeof node === 'object') {
          if (Number.isFinite(Number(node.object_id))) objects.push(node);
          for (const value of Object.values(node)) walk(value);
        }
      }
      walk(call.response);
      const seen = new Set();
      return objects.filter(obj => {
        const id = Number(obj.object_id);
        if (seen.has(id)) return false;
        seen.add(id);
        return true;
      });
    }
    function mergeSceneFromHistory(history) {
      if (!history || !state.graph?.live) return;
      const objects = new Map(state.objects.map(object => [Number(object.object_id), object]));
      const rooms = new Map(state.rooms.map(room => [String(room.room_id), room]));

      function mergeObject(candidate) {
        const id = Number(candidate?.object_id);
        if (!Number.isFinite(id)) return;
        const geometry = candidate.geometry && typeof candidate.geometry === 'object'
          ? candidate.geometry : {};
        const normalized = {
          ...candidate,
          object_id: id,
          label: candidate.label || candidate.display_description || 'object',
          description: candidate.description || candidate.canonical_description || candidate.display_description || '',
          center_world: candidate.center_world || geometry.center_world,
          size_m: candidate.size_m || geometry.size_m,
          yaw_rad: candidate.yaw_rad ?? geometry.yaw_rad
        };
        const useful = Object.fromEntries(Object.entries(normalized).filter(([, value]) => value !== null && value !== undefined && value !== ''));
        objects.set(id, {...(objects.get(id) || {}), ...useful});
      }
      function walkRooms(node) {
        if (!node) return;
        if (Array.isArray(node)) { node.forEach(walkRooms); return; }
        if (typeof node !== 'object') return;
        if (node.room_id !== undefined && Array.isArray(node.object_ids)) {
          const key = String(node.room_id);
          rooms.set(key, {
            ...(rooms.get(key) || {}),
            ...node,
            room_id: key,
            label: node.label || node.name || key
          });
        }
        for (const value of Object.values(node)) walkRooms(value);
      }
      for (const iteration of history.iterations || []) {
        for (const call of iteration.function_calls || []) {
          responseObjectsFromCall(call).forEach(mergeObject);
          walkRooms(call.response);
        }
      }
      for (const room of rooms.values()) {
        for (const objectId of room.object_ids || []) {
          const object = objects.get(Number(objectId));
          if (!object) continue;
          const membership = {room_id: room.room_id, label: room.label};
          object.rooms = [
            ...(object.rooms || []).filter(item => String(item.room_id) !== String(room.room_id)),
            membership
          ];
        }
      }
      state.objects = [...objects.values()].sort((a, b) => Number(a.object_id) - Number(b.object_id));
      state.rooms = [...rooms.values()].sort((a, b) => String(a.room_id).localeCompare(String(b.room_id)));
      state.graph.objects = state.objects;
      state.graph.rooms = state.rooms;
      if (state.selectedId === null && state.objects.length) state.selectedId = state.objects[0].object_id;
      computeBounds();
      uploadRoomBoxes();
      const session = history.read_session || {};
      const revision = session.scene_revision === undefined ? '?' : session.scene_revision;
      document.getElementById('graphMeta').textContent =
        `LIVE ${state.graph.service} | revision ${revision} | ${state.objects.length} traced objects | ${state.rooms.length} rooms`;
    }
    function snapshotImageMeta(imageIndex) {
      return (state.graph?.snapshot_images || []).find(img => Number(img.image_index) === Number(imageIndex)) || null;
    }
    function snapshotThumbHtml(object) {
      const ref = object.snapshot;
      if (!ref || !Number.isFinite(Number(ref.image_index))) return '';
      const image = snapshotImageMeta(ref.image_index);
      if (!image) return '';
      const w = Math.max(1, Number(image.width) || 1);
      const h = Math.max(1, Number(image.height) || 1);
      let box = '';
      if (Array.isArray(ref.bbox_xyxy) && ref.bbox_xyxy.length >= 4) {
        const x0 = Math.max(0, Math.min(w, Math.min(Number(ref.bbox_xyxy[0]) || 0, Number(ref.bbox_xyxy[2]) || 0)));
        const x1 = Math.max(0, Math.min(w, Math.max(Number(ref.bbox_xyxy[0]) || 0, Number(ref.bbox_xyxy[2]) || 0)));
        const y0 = Math.max(0, Math.min(h, Math.min(Number(ref.bbox_xyxy[1]) || 0, Number(ref.bbox_xyxy[3]) || 0)));
        const y1 = Math.max(0, Math.min(h, Math.max(Number(ref.bbox_xyxy[1]) || 0, Number(ref.bbox_xyxy[3]) || 0)));
        box = `<div class="snapshot-box" style="left:${100 * x0 / w}%;top:${100 * y0 / h}%;width:${100 * Math.max(0, x1 - x0) / w}%;height:${100 * Math.max(0, y1 - y0) / h}%;"></div>`;
      }
      const uri = `/snapshot-image/${encodeURIComponent(String(ref.image_index))}`;
      return `<div class="snapshot-thumb" data-id="${Number(object.object_id)}">
        <div class="snapshot-frame" style="aspect-ratio:${w} / ${h};">
          <img src="${uri}" alt="">
          ${box}
        </div>
        <div class="snapshot-caption">#${Number(object.object_id)} ${escapeHtml(object.label || 'object')} | image ${Number(ref.image_index)}</div>
      </div>`;
    }
    function descriptionEvidenceFromCall(call) {
      return responseObjectsFromCall(call)
        .filter(obj => typeof obj.description === 'string' && obj.description.trim())
        .slice(0, 8);
    }
    function snapshotEvidenceFromCall(call) {
      return responseObjectsFromCall(call)
        .filter(obj => obj.snapshot && Number.isFinite(Number(obj.snapshot.image_index)))
        .slice(0, 6);
    }
    function renderProgress() {
      const events = state.progressEvents.slice(-14);
      if (!events.length && !state.asking) {
        progressList.classList.remove('active');
        progressList.innerHTML = '';
        return;
      }
      progressList.classList.add('active');
      const rows = events.map(event => {
        const phase = String(event.phase || '');
        const cls = phase.endsWith('_start') || phase === 'tool_start' || phase === 'gemini_start' ? ' running' : phase.includes('error') ? ' error' : '';
        const tool = event.tool ? ` · ${escapeHtml(event.tool)}` : '';
        return `<div class="progress-item${cls}">
          <span class="progress-dot"></span>
          <span>${escapeHtml(event.message || phase)}${tool}</span>
        </div>`;
      }).join('');
      progressList.innerHTML = rows || `<div class="progress-item running"><span class="progress-dot"></span><span>Waiting for model/tool activity...</span></div>`;
      progressList.scrollTop = progressList.scrollHeight;
    }
    async function pollProgress() {
      try {
        const response = await fetch(`/events?after=${encodeURIComponent(String(state.lastEventId))}`);
        const data = await response.json();
        const events = data.events || [];
        if (events.length) {
          state.progressEvents.push(...events);
          state.lastEventId = Math.max(state.lastEventId, ...events.map(e => Number(e.id) || 0));
          renderProgress();
        }
      } catch (error) {
        state.progressEvents.push({id: ++state.lastEventId, phase: 'poll_error', message: String(error)});
        renderProgress();
      }
    }
    function startProgressPolling() {
      stopProgressPolling();
      pollProgress();
      state.pollTimer = window.setInterval(pollProgress, 450);
    }
    function stopProgressPolling() {
      if (state.pollTimer !== null) {
        window.clearInterval(state.pollTimer);
        state.pollTimer = null;
      }
    }
    function renderTraceGroups() {
      if (!state.lastHistory) {
        traceList.innerHTML = '<div class="empty">Ask a question to see model/tool activity.</div>';
        return;
      }
      let html = '';
      for (const iteration of state.lastHistory.iterations || []) {
        if (iteration.model_text) {
          html += `<div class="tool-card"><div class="tool-head">Visible model text <span>iter ${iteration.iteration}</span></div><pre>${escapeHtml(iteration.model_text)}</pre></div>`;
        }
        for (const call of iteration.function_calls || []) {
          const group = state.highlightGroups.find(g => g.iteration === iteration.iteration && g.tool === call.name);
          const color = group ? group.color : '#d7ddd4';
          const objects = responseObjectsFromCall(call);
          const chips = (group?.object_ids || []).map(id => `<button class="chip" data-id="${id}">${escapeHtml(objectLabel(id))}</button>`).join('');
          const objectSummary = objects.slice(0, 8).map(obj => {
            const score = Number.isFinite(Number(obj.semantic_score)) ? ` score ${fmt(obj.semantic_score, 3)}` : '';
            return `#${obj.object_id} ${obj.label || ''}${score}`;
          }).join('\n');
          const descriptions = descriptionEvidenceFromCall(call);
          const descriptionHtml = descriptions.map(obj => `<div class="description-item"><strong>#${Number(obj.object_id)} ${escapeHtml(obj.label || 'object')}</strong><br>${escapeHtml(obj.description)}</div>`).join('');
          const snapshots = snapshotEvidenceFromCall(call);
          const snapshotHtml = snapshots.map(snapshotThumbHtml).join('');
          html += `<div class="tool-card" data-tool="${escapeHtml(call.name)}">
            <div class="tool-head"><span><span class="swatch" style="background:${color}"></span>${escapeHtml(call.name)}</span><span>iter ${iteration.iteration}</span></div>
            <div class="meta">args</div>
            <pre>${escapeHtml(JSON.stringify(call.args || {}, null, 2))}</pre>
            <div class="chip-list">${chips}</div>
            ${objectSummary ? `<div class="meta">object results</div><pre>${escapeHtml(objectSummary)}</pre>` : ''}
            ${descriptionHtml ? `<div class="evidence-title">descriptions</div><div class="description-list">${descriptionHtml}</div>` : ''}
            ${snapshotHtml ? `<div class="evidence-title">snapshots</div><div class="snapshot-grid">${snapshotHtml}</div>` : ''}
            <details><summary>full tool response</summary><pre>${escapeHtml(JSON.stringify(call.response || {}, null, 2))}</pre></details>
          </div>`;
        }
      }
      traceList.innerHTML = html || '<div class="empty">No tool calls were made.</div>';
      traceList.querySelectorAll('[data-id]').forEach(node => node.addEventListener('click', () => selectObject(node.dataset.id)));
    }
    async function askQuestion() {
      const query = questionInput.value.trim();
      if (!query) return;
      const provider = state.provider;
      const controller = new AbortController();
      state.askController = controller;
      askBtn.disabled = true;
      askBtn.textContent = 'Asking...';
      document.getElementById('answerText').textContent = 'Working...';
      document.getElementById('reasoningText').textContent = '';
      state.asking = true;
      updateProviderControls();
      state.progressEvents = [];
      state.lastEventId = 0;
      renderProgress();
      try {
        await fetch('/events?clear=1');
        startProgressPolling();
        const response = await fetch('/ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({query, provider}),
          signal: controller.signal
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || response.statusText);
        state.lastHistory = data.history;
        state.provider = data.provider || provider;
        mergeSceneFromHistory(data.history);
        document.getElementById('answerText').textContent = typeof data.answer === 'string' ? data.answer : JSON.stringify(data.answer, null, 2);
        document.getElementById('reasoningText').textContent = data.reasoning || '';
        setHighlights(data.highlight_groups || []);
      } catch (error) {
        if (error?.name !== 'AbortError') {
          document.getElementById('answerText').textContent = 'Request failed';
          document.getElementById('reasoningText').textContent = String(error);
        }
      } finally {
        if (state.askController === controller) {
          await pollProgress();
          state.askController = null;
          state.asking = false;
          stopProgressPolling();
          renderProgress();
          askBtn.disabled = false;
          askBtn.textContent = 'Ask';
          updateProviderControls();
        }
      }
    }
    async function resetQuestion() {
      resetBtn.disabled = true;
      resetBtn.textContent = 'Resetting...';
      const controller = state.askController;
      state.askController = null;
      if (controller) controller.abort();
      state.asking = false;
      stopProgressPolling();
      try {
        const response = await fetch('/reset', {method: 'POST'});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || response.statusText);
        if (state.graph?.qa && data.providers) {
          state.graph.qa.providers = data.providers;
          state.graph.qa.default_provider = data.default_provider || state.provider;
        }
        state.progressEvents = [];
        state.lastEventId = 0;
        state.lastHistory = null;
        setHighlights([]);
        renderTraceGroups();
        document.getElementById('answerText').textContent = 'QA reset. Ready for a new question.';
        document.getElementById('reasoningText').textContent = '';
      } catch (error) {
        document.getElementById('answerText').textContent = 'Reset failed';
        document.getElementById('reasoningText').textContent = String(error);
      } finally {
        resetBtn.disabled = false;
        resetBtn.textContent = 'Reset QA';
        askBtn.disabled = false;
        askBtn.textContent = 'Ask';
        updateProviderControls();
        renderProgress();
      }
    }
    canvas.addEventListener('mousedown', event => {
      state.dragging = true; state.moved = false; state.lastX = event.clientX; state.lastY = event.clientY; canvas.classList.add('dragging');
    });
    window.addEventListener('mousemove', event => {
      if (!state.dragging) return;
      const dx = event.clientX - state.lastX, dy = event.clientY - state.lastY;
      if (Math.abs(dx) + Math.abs(dy) > 2) state.moved = true;
      state.lastX = event.clientX; state.lastY = event.clientY;
      state.yaw += dx * .006;
      state.pitch = Math.max(-1.35, Math.min(1.35, state.pitch + dy * .005));
      render();
    });
    window.addEventListener('mouseup', event => {
      if (!state.dragging) return;
      state.dragging = false; canvas.classList.remove('dragging');
      if (state.moved) return;
      const rect = canvas.getBoundingClientRect();
      const pos = {x: event.clientX - rect.left, y: event.clientY - rect.top};
      const hits = state.screenObjects.filter(o => pos.x >= o.minX && pos.x <= o.maxX && pos.y >= o.minY && pos.y <= o.maxY).sort((a,b) => a.area - b.area || a.depth - b.depth);
      if (hits.length) selectObject(hits[0].id);
    });
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      state.distance = Math.max(state.radius * .08, Math.min(state.radius * 30, state.distance * Math.exp(event.deltaY * .001)));
      render();
    }, {passive: false});
    askBtn.addEventListener('click', askQuestion);
    resetBtn.addEventListener('click', resetQuestion);
    for (const button of providerButtons) {
      button.addEventListener('click', () => selectProvider(button.dataset.provider));
    }
    questionInput.addEventListener('keydown', event => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') askQuestion();
    });
    showPoints.addEventListener('change', render);
    showRooms.addEventListener('change', render);
    pointSize.addEventListener('input', render);
    window.addEventListener('resize', render);
    async function loadGraph() {
      initGl();
      const [response] = await Promise.all([fetch('/graph.json'), loadPointCloud()]);
      state.graph = await response.json();
      state.provider = state.graph?.qa?.default_provider || state.provider;
      state.objects = (state.graph.objects || []).slice().sort((a, b) => Number(a.object_id) - Number(b.object_id));
      state.rooms = (state.graph.rooms || []).slice().sort((a, b) => Number(a.room_id) - Number(b.room_id));
      state.selectedId = state.objects.length ? state.objects[0].object_id : null;
      computeBounds();
      uploadRoomBoxes();
      document.getElementById('graphMeta').textContent = state.graph.live
        ? `LIVE ${state.graph.service} | waiting for the first question`
        : `${state.objects.length} objects | ${(state.graph.rooms || []).length} rooms | ${state.graph.world_frame || 'world'}`;
      updateProviderControls();
      renderObjectList();
      renderTraceGroups();
      render();
    }
    loadGraph().catch(error => {
      document.getElementById('graphMeta').textContent = String(error);
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = self.server  # type: ignore[assignment]
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("", "/"):
            body = HTML.replace("__POINT_SIZE__", self.server.point_size)  # type: ignore[attr-defined]
            self._send_bytes(body.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/graph.json":
            self._send_bytes(server.graph_json, "application/json; charset=utf-8")
            return
        if parsed.path == "/points-meta.json":
            self._send_bytes(server.points_meta_json, "application/json; charset=utf-8")  # type: ignore[attr-defined]
            return
        if parsed.path == "/points.bin":
            self._send_bytes(server.points_bytes, "application/octet-stream")  # type: ignore[attr-defined]
            return
        if parsed.path == "/colors.bin":
            self._send_bytes(server.colors_bytes, "application/octet-stream")  # type: ignore[attr-defined]
            return
        if parsed.path == "/events":
            query_params = urllib.parse.parse_qs(parsed.query)
            if query_params.get("clear"):
                server.progress_log.clear()  # type: ignore[attr-defined]
            try:
                after = int((query_params.get("after") or ["0"])[0])
            except ValueError:
                after = 0
            self._send_json(server.progress_log.snapshot(after))  # type: ignore[attr-defined]
            return
        if parsed.path.startswith("/snapshot-image/"):
            token = urllib.parse.unquote(parsed.path[len("/snapshot-image/"):])
            try:
                image_index = int(token)
            except ValueError:
                self.send_error(404, "not found")
                return
            image_path = server.snapshot_paths.get(image_index)  # type: ignore[attr-defined]
            if image_path is None or not image_path.exists() or not image_path.is_file():
                self.send_error(404, "not found")
                return
            content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            self._send_bytes(image_path.read_bytes(), content_type)
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/reset":
            self._reset_qa()
            return
        if parsed.path != "/ask":
            self.send_error(404, "not found")
            return
        acquired = False
        request_lock: threading.Lock | None = None
        runtime: QaRuntime | None = None
        query = ""
        provider = ""
        turn_finished = False
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0 or length > 1_048_576:
                raise ValueError("request body must be between 1 byte and 1 MiB")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            query = str(payload.get("query") or "").strip()
            if not query:
                raise ValueError("query must not be empty")
            provider = str(
                payload.get("provider") or self.server.default_provider  # type: ignore[attr-defined]
            ).strip().lower()
            runtime = self.server.current_runtime()  # type: ignore[attr-defined]
            if provider not in runtime.agents:
                provider_status = runtime.provider_status
                known = provider_status.get(provider)
                if known is not None:
                    self._send_json(
                        {"error": str(known.get("error") or f"{provider} is unavailable")},
                        status=503,
                    )
                else:
                    self._send_json(
                        {"error": f"unknown model provider: {provider}"},
                        status=400,
                    )
                return
            request_lock = runtime.qa_lock
            acquired = request_lock.acquire(blocking=False)
            if not acquired:
                self._send_json(
                    {"error": "another scene QA request is still running"},
                    status=409,
                )
                return
            if runtime.retired or not self.server.is_current_runtime(runtime):  # type: ignore[attr-defined]
                self._send_json({"error": "QA was reset; retry the question"}, status=409)
                return
            started_time_s = time.time()
            request_event = runtime.add_progress(
                {
                    "phase": "request_start",
                    "message": f"Question received by {provider}.",
                    "provider": provider,
                }
            )
            progress_after = int(request_event.get("id", 1)) - 1
            model = str(runtime.provider_status.get(provider, {}).get("model") or "")
            turn_index = self.server.scene_qa_log.begin_turn(  # type: ignore[attr-defined]
                query=query,
                provider=provider,
                model=model,
                started_time_s=started_time_s,
            )
            runtime.activate_turn(
                turn_index,
                started_time_s=started_time_s,
                progress_after=progress_after,
            )
            response = runtime.agents[provider].answer_query(query)
            highlight_groups = highlight_groups_from_history(response.history)
            body = {
                "reasoning": response.reasoning,
                "answer": response.answer,
                "raw_text": response.raw_text,
                "history": response.history,
                "highlight_groups": highlight_groups,
                "provider": provider,
                "model": response.history.get("model"),
            }
            runtime.add_progress(
                {
                    "phase": "request_done",
                    "message": f"{provider} answer ready.",
                    "provider": provider,
                }
            )
            turn_finished = runtime.finish_turn(
                response={
                    "reasoning": response.reasoning,
                    "answer": response.answer,
                    "raw_text": response.raw_text,
                },
                history=response.history,
            )
            if not self.server.is_current_runtime(runtime):  # type: ignore[attr-defined]
                self._send_json({"error": "question result discarded after reset"}, status=409)
                return
            self._send_json(body)
        except Exception as exc:
            if runtime is None:
                self.server.progress_log.add(  # type: ignore[attr-defined]
                    {"phase": "request_error", "message": f"{type(exc).__name__}: {exc}"}
                )
            else:
                runtime.add_progress(
                    {"phase": "request_error", "message": f"{type(exc).__name__}: {exc}"}
                )
            if runtime is not None and acquired and query and provider and not turn_finished:
                history = getattr(exc, "scene_qa_history", None)
                if not isinstance(history, dict):
                    history = {}
                turn_finished = runtime.finish_turn(
                    history=history,
                    error=exc,
                )
            if runtime is None or self.server.is_current_runtime(runtime):  # type: ignore[attr-defined]
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        finally:
            if acquired and request_lock is not None:
                request_lock.release()

    def _reset_qa(self) -> None:
        try:
            runtime = self.server.reset_runtime()  # type: ignore[attr-defined]
            self._send_json(
                {
                    "reset": True,
                    "generation": runtime.generation,
                    "providers": runtime.provider_status,
                    "default_provider": self.server.default_provider,  # type: ignore[attr-defined]
                }
            )
        except Exception as exc:
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=500,
            )

    def _send_json(self, body: dict[str, Any], status: int = 200) -> None:
        self._send_bytes(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("roomie_scene_qa_viewer: " + (fmt % args) + "\n")


def main() -> int:
    args = parse_args()
    qa_config = apply_cli_overrides(load_qa_config(args.qa_config), args)
    offline = args.offline_json is not None
    point_cloud = empty_point_cloud()
    if offline:
        json_path = qa_config.resolved_graph_json()
        if not json_path.exists():
            print(f"DSG JSON does not exist: {json_path}", file=sys.stderr)
            return 2
        pipeline_params = load_pipeline_params(as_path(args.config))
        graph = GraphStore.load(json_path)
        point_path: Path | None = None
        if not args.no_points:
            point_path = (
                as_path(args.points)
                if args.points is not None
                else resolve_points_from_config(pipeline_params)
                or find_auto_point_cloud(json_path)
            )
            if point_path is not None:
                print(f"Loading point cloud: {point_path}")
                point_cloud = load_point_cloud(
                    point_path,
                    max_points=args.max_points,
                    surface_threshold_m=args.surface_threshold_m,
                    min_weight=args.min_tsdf_weight,
                )
            else:
                print("No point cloud found; pass --points to show RGB points.")
        if point_cloud.count:
            print(f"Point cloud: {point_cloud.count} points from {point_cloud.source}")

        search_index = ObjectSearchIndex(
            graph,
            model_path=qa_config.resolved_embedding_model(),
            backend=qa_config.embedding_backend,
            device=qa_config.device,
        )
        graph_data = graph_payload(graph)
        snapshot_paths = graph.snapshot_paths
        source_label = f"Loaded DSG: {json_path}"
    else:
        graph = None
        graph_data = live_graph_payload(args.service)
        snapshot_paths = {}
        source_label = f"Live QueryScene service: {args.service}"

    progress_log = ProgressLog()
    configured_providers: dict[str, dict[str, Any]] = {
        "gemini": {
            "label": "Gemini",
            "model": qa_config.gemini_model,
            "available": False,
        },
        "doubao": {
            "label": "Doubao",
            "model": qa_config.doubao_model,
            "available": False,
        },
    }

    try:
        qa_config_path = (
            as_path(args.qa_config)
            if args.qa_config is not None
            else default_qa_config_path()
        )
        scene_qa_log = SceneQaSessionLog(
            default_scene_qa_log_dir(),
            metadata={
                "mode": "offline" if offline else "live",
                "source": source_label,
                "service": None if offline else args.service,
                "qa_config": str(qa_config_path) if qa_config_path is not None else None,
                "providers": configured_providers,
            },
        )
    except Exception as exc:
        print(
            f"Could not initialize Scene QA log: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    live_transport: RosQuerySceneTransport | None = None
    live_transport_lock = threading.Lock()
    if not offline:
        try:
            live_transport = RosQuerySceneTransport(
                service_name=args.service,
                timeout_sec=args.service_timeout_sec,
            )
        except Exception as exc:
            scene_qa_log.close()
            print(
                f"Could not initialize live QueryScene transport: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2

    def call_live_scene(payload: dict[str, Any]) -> dict[str, Any]:
        if live_transport is None:
            raise RuntimeError("live QueryScene transport is unavailable")
        # rclpy nodes/clients are not used concurrently. A retired agent can
        # finish a bounded service call while the new generation is already
        # accepting model requests.
        with live_transport_lock:
            return live_transport(payload)

    def build_runtime(generation: int) -> QaRuntime:
        runtime = QaRuntime(generation, scene_qa_log, progress_log)
        if offline:
            def registry_factory():
                return create_default_tool_registry(graph, search_index, qa_config)
        else:
            def registry_factory():
                live_client = LiveSceneQueryClient(
                    call_live_scene,
                    session_ttl_ms=LIVE_READ_SESSION_TTL_MS,
                )
                return create_live_tool_registry(live_client, qa_config)

        runtime.provider_status = {
            name: dict(status) for name, status in configured_providers.items()
        }
        provider_builders = {
            "gemini": lambda: GeminiSceneQaAgent(
                graph,
                registry_factory(),
                qa_config,
                api_key=args.api_key,
                progress_callback=runtime.add_progress,
                history_callback=runtime.checkpoint_history,
                cancel_event=runtime.cancel_event,
            ),
            "doubao": lambda: DoubaoSceneQaAgent(
                graph,
                registry_factory(),
                qa_config,
                api_key=args.doubao_api_key,
                progress_callback=runtime.add_progress,
                history_callback=runtime.checkpoint_history,
                cancel_event=runtime.cancel_event,
            ),
        }
        for provider, builder in provider_builders.items():
            try:
                runtime.agents[provider] = builder()
                runtime.provider_status[provider]["available"] = True
                runtime.provider_status[provider].pop("error", None)
            except Exception as exc:
                runtime.provider_status[provider]["available"] = False
                runtime.provider_status[provider]["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                print(
                    f"Scene QA provider {provider} unavailable: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
        return runtime

    initial_runtime = build_runtime(1)
    if not initial_runtime.agents:
        if live_transport is not None:
            live_transport.close()
        scene_qa_log.close()
        print("No Scene QA model provider is available.", file=sys.stderr)
        return 2
    default_provider = args.default_provider
    if default_provider not in initial_runtime.agents:
        default_provider = next(iter(initial_runtime.agents))
        print(
            f"Requested default provider is unavailable; using {default_provider}.",
            file=sys.stderr,
        )
    graph_data["qa"] = {
        "default_provider": default_provider,
        "providers": initial_runtime.provider_status,
    }

    server: ThreadingHTTPServer | None = None
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        server.daemon_threads = True
        runtime_lock = threading.Lock()
        reset_lock = threading.Lock()
        runtime_state = {"current": initial_runtime}

        def current_runtime() -> QaRuntime:
            with runtime_lock:
                return runtime_state["current"]

        def is_current_runtime(candidate: QaRuntime) -> bool:
            with runtime_lock:
                return runtime_state["current"] is candidate

        def reset_runtime() -> QaRuntime:
            nonlocal default_provider
            with reset_lock:
                old_runtime = current_runtime()
                new_runtime = build_runtime(old_runtime.generation + 1)
                if not new_runtime.agents:
                    raise RuntimeError("no Scene QA model provider is available after reset")
                with runtime_lock:
                    if runtime_state["current"] is not old_runtime:
                        return runtime_state["current"]
                    runtime_state["current"] = new_runtime
                old_runtime.retired = True
                old_runtime.cancel()
                if default_provider not in new_runtime.agents:
                    default_provider = next(iter(new_runtime.agents))
                server.default_provider = default_provider  # type: ignore[attr-defined]
                progress_log.clear()
                progress_log.add(
                    {
                        "phase": "qa_reset",
                        "message": "Scene QA runtime reset; ready for a new question.",
                        "generation": new_runtime.generation,
                    }
                )
                return new_runtime

        server.graph_json = json.dumps(graph_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")  # type: ignore[attr-defined]
        server.snapshot_paths = snapshot_paths  # type: ignore[attr-defined]
        server.points_meta_json = json.dumps(points_meta(point_cloud), separators=(",", ":")).encode("utf-8")  # type: ignore[attr-defined]
        server.points_bytes = point_cloud.points.astype("<f4", copy=False).tobytes()  # type: ignore[attr-defined]
        server.colors_bytes = point_cloud.colors.astype("u1", copy=False).tobytes()  # type: ignore[attr-defined]
        server.point_size = repr(float(args.point_size))  # type: ignore[attr-defined]
        server.progress_log = progress_log  # type: ignore[attr-defined]
        server.default_provider = default_provider  # type: ignore[attr-defined]
        server.scene_qa_log = scene_qa_log  # type: ignore[attr-defined]
        server.current_runtime = current_runtime  # type: ignore[attr-defined]
        server.is_current_runtime = is_current_runtime  # type: ignore[attr-defined]
        server.reset_runtime = reset_runtime  # type: ignore[attr-defined]

        browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        url = f"http://{browser_host}:{server.server_port}/"
        print(f"Roomie Scene QA viewer: {url}")
        print(source_label)
        print(f"Scene QA conversation log: {scene_qa_log.path}")
        print(
            "QA providers: "
            + ", ".join(
                f"{name}={initial_runtime.provider_status[name]['model']}"
                for name in initial_runtime.agents
            )
        )
        if args.browser == "auto" and not args.no_browser:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        if server is not None:
            server.server_close()
        current_runtime = initial_runtime if server is None else server.current_runtime()  # type: ignore[attr-defined]
        current_runtime.cancel()
        if live_transport is not None:
            live_transport.close()
        scene_qa_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
