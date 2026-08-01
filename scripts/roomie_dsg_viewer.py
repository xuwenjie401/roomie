#!/usr/bin/env python3
"""Offline WebGL viewer for Roomie DSG snapshots."""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sqlite3
import struct
import sys
from typing import Any
import urllib.parse
import webbrowser

import numpy as np
import yaml


DEFAULT_JSON = Path("/home/lindenbot/Datasets/output/jarvis_home/instances/latest.json")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "pipeline_nvblox.yaml"


@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray
    source: str

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


def as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def load_pipeline_params(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"config yaml does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        root = yaml.safe_load(stream) or {}
    if not isinstance(root, dict):
        return {}
    node = root.get("roomie_pipeline_node")
    if isinstance(node, dict):
        params = node.get("ros__parameters")
        if isinstance(params, dict):
            return params
    params = root.get("ros__parameters")
    return params if isinstance(params, dict) else root


def newest_file(root: Path, patterns: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(p for p in root.glob(pattern) if p.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def resolve_json_from_config(params: dict[str, Any]) -> Path | None:
    persistence = params.get("persistence")
    if not isinstance(persistence, dict):
        return None
    save_path = persistence.get("scene_graph_save_path")
    if isinstance(save_path, str) and save_path.strip():
        base = Path(save_path).expanduser()
        if base.suffix.lower() == ".json":
            return base.resolve()
        latest = base / "latest.json"
        if latest.exists():
            return latest.resolve()
        candidate = newest_file(base, ("*.json",))
        if candidate is not None:
            return candidate
    load_path = persistence.get("scene_graph_load_path")
    if isinstance(load_path, str) and load_path.strip():
        return Path(load_path).expanduser().resolve()
    return None


def resolve_points_from_config(params: dict[str, Any]) -> Path | None:
    tsdf = params.get("tsdf")
    if not isinstance(tsdf, dict):
        return None
    load_path = tsdf.get("map_load_path")
    if isinstance(load_path, str) and load_path.strip():
        path = Path(load_path).expanduser()
        if path.exists() and path.is_file():
            return path.resolve()
    save_path = tsdf.get("map_save_path")
    if isinstance(save_path, str) and save_path.strip():
        base = Path(save_path).expanduser()
        if base.suffix.lower() in (".nvblox", ".ply", ".pcd", ".npy", ".npz", ".json"):
            return base.resolve()
        candidate = newest_file(base, ("*.nvblox", "*.ply", "*.pcd", "*.npz", "*.npy", "*.json"))
        if candidate is not None:
            return candidate
    return None


def normalize_graph(raw: dict[str, Any]) -> dict[str, Any]:
    object_graph = raw.get("object_graph") if isinstance(raw.get("object_graph"), dict) else raw
    objects = raw.get("objects")
    if not isinstance(objects, list):
        objects = object_graph.get("objects", [])
    relations = raw.get("relations")
    if not isinstance(relations, list):
        relations = object_graph.get("relations", [])
    snapshot_images = raw.get("snapshot_images")
    if not isinstance(snapshot_images, list):
        snapshot_images = object_graph.get("snapshot_images", [])
    return {
        "format": raw.get("format", object_graph.get("format", "roomie_object_graph")),
        "world_frame": raw.get("world_frame", object_graph.get("world_frame", "world")),
        "saved_time": raw.get("saved_time", ""),
        "saved_time_ns": raw.get("saved_time_ns", object_graph.get("saved_time_ns", 0)),
        "objects": objects,
        "relations": relations,
        "rooms": raw.get("rooms", []),
        "map": raw.get("map", {}),
        "snapshot_images": snapshot_images,
    }


def snapshot_paths(graph: dict[str, Any], json_path: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    root = json_path.parent
    for image in graph.get("snapshot_images", []):
        if not isinstance(image, dict):
            continue
        image_index = image.get("image_index")
        if not isinstance(image_index, int):
            continue
        candidates = []
        source_path = image.get("source_path")
        if isinstance(source_path, str) and source_path:
            candidates.append(Path(source_path).expanduser())
        uri = image.get("uri")
        if isinstance(uri, str) and uri:
            candidates.append(root / uri)
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists() and resolved.is_file():
                paths[image_index] = resolved
                break
    return paths


def graph_bounds(graph: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for obj in graph.get("objects", []):
        if not isinstance(obj, dict):
            continue
        center = np.asarray(obj.get("center_world", [0, 0, 0]), dtype=np.float32)
        size = np.asarray(obj.get("size_m", [0, 0, 0]), dtype=np.float32)
        if center.shape[0] < 3 or size.shape[0] < 3:
            continue
        half = np.maximum(size[:3], 0.0) * 0.5
        mins.append(center[:3] - half)
        maxs.append(center[:3] + half)
    if not mins:
        return None
    return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0)


def downsample(points: np.ndarray,
               colors: np.ndarray,
               max_points: int,
               seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    indices.sort()
    return points[indices], colors[indices]


def finish_point_cloud(points: np.ndarray,
                       colors: np.ndarray | None,
                       source: str,
                       max_points: int) -> PointCloud:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("point cloud must be Nx3 or Nx6")
    points = points[:, :3]
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if colors is None:
        z = points[:, 2] if points.size else np.zeros((0,), dtype=np.float32)
        if z.size:
            zn = (z - float(z.min())) / max(1.0e-6, float(z.max() - z.min()))
            colors = np.stack(
                [80 + 110 * zn, 120 + 80 * (1.0 - zn), 150 + 80 * zn], axis=1
            )
        else:
            colors = np.zeros((0, 3), dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise ValueError("point colors must be Nx3")
    colors = colors[valid, :3] if colors.shape[0] == valid.shape[0] else colors[:, :3]
    if np.issubdtype(colors.dtype, np.floating):
        if colors.size and float(np.nanmax(colors)) <= 1.0:
            colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    points, colors = downsample(points, colors, max_points)
    return PointCloud(np.ascontiguousarray(points, dtype=np.float32),
                      np.ascontiguousarray(colors, dtype=np.uint8),
                      source)


def parse_packed_rgb(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.floating):
        packed = values.astype("<f4").view("<u4")
    else:
        packed = values.astype("<u4")
    return np.stack(
        [(packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF], axis=1
    ).astype(np.uint8)


def load_numpy_points(path: Path, max_points: int) -> PointCloud:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        keys = set(data.files)
        if "points" in keys:
            points = data["points"]
        elif "xyz" in keys:
            points = data["xyz"]
        elif "positions" in keys:
            points = data["positions"]
        else:
            raise ValueError("npz point cloud needs points, xyz, or positions")
        colors = None
        if "colors" in keys:
            colors = data["colors"]
        elif "rgb" in keys:
            colors = data["rgb"]
    else:
        arr = np.asarray(data)
        points = arr[:, :3]
        colors = arr[:, 3:6] if arr.ndim == 2 and arr.shape[1] >= 6 else None
    return finish_point_cloud(points, colors, str(path), max_points)


PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def load_ply(path: Path, max_points: int) -> PointCloud:
    with path.open("rb") as stream:
        header_lines: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("invalid ply header")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        header_bytes = stream.tell()
    fmt = ""
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[:1] == ["element"]:
            in_vertex = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and parts[0] == "property" and len(parts) == 3:
            properties.append((parts[2], parts[1]))
        elif parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
    names = [name for name, _ in properties]
    if "x" not in names or "y" not in names or "z" not in names:
        raise ValueError("ply is missing x/y/z vertex properties")
    if fmt == "ascii":
        rows = []
        with path.open("r", encoding="ascii", errors="replace") as stream:
            for _ in header_lines:
                next(stream)
            for _ in range(vertex_count):
                rows.append([float(v) for v in next(stream).split()[: len(properties)]])
        arr = np.asarray(rows, dtype=np.float32)
        columns = {name: arr[:, i] for i, (name, _) in enumerate(properties)}
    elif fmt == "binary_little_endian":
        dtype = np.dtype([(name, PLY_TYPES[typ]) for name, typ in properties])
        with path.open("rb") as stream:
            stream.seek(header_bytes)
            arr = np.frombuffer(stream.read(vertex_count * dtype.itemsize), dtype=dtype)
        columns = {name: arr[name] for name, _ in properties}
    else:
        raise ValueError(f"unsupported ply format: {fmt}")
    points = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    colors = None
    if all(name in columns for name in ("red", "green", "blue")):
        colors = np.stack([columns["red"], columns["green"], columns["blue"]], axis=1)
    elif all(name in columns for name in ("r", "g", "b")):
        colors = np.stack([columns["r"], columns["g"], columns["b"]], axis=1)
    return finish_point_cloud(points, colors, str(path), max_points)


def load_pcd(path: Path, max_points: int) -> PointCloud:
    with path.open("rb") as stream:
        header: dict[str, list[str]] = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("invalid pcd header")
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                data_start = stream.tell()
                break
        fields = header.get("FIELDS", [])
        sizes = [int(v) for v in header.get("SIZE", [])]
        types = header.get("TYPE", [])
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
        points_count = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
        data_type = header["DATA"][0].lower()
        raw = stream.read()
    if data_type == "ascii":
        arr = np.loadtxt(io.BytesIO(raw), dtype=np.float32, max_rows=points_count)
        columns = {name: arr[:, i] for i, name in enumerate(fields)}
    elif data_type == "binary":
        dtype_fields = []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            if typ == "F" and size == 4:
                dt = "<f4"
            elif typ == "F" and size == 8:
                dt = "<f8"
            elif typ == "U":
                dt = f"<u{size}"
            elif typ == "I":
                dt = f"<i{size}"
            else:
                raise ValueError(f"unsupported pcd field {name} type {typ}{size}")
            dtype_fields.append((name, dt, (count,)) if count > 1 else (name, dt))
        dtype = np.dtype(dtype_fields)
        arr = np.frombuffer(raw[: points_count * dtype.itemsize], dtype=dtype)
        columns = {name: arr[name] for name in fields}
    else:
        raise ValueError(f"unsupported pcd DATA {data_type}")
    points = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    colors = None
    if all(name in columns for name in ("r", "g", "b")):
        colors = np.stack([columns["r"], columns["g"], columns["b"]], axis=1)
    elif "rgb" in columns:
        colors = parse_packed_rgb(columns["rgb"])
    elif "rgba" in columns:
        colors = parse_packed_rgb(columns["rgba"])
    return finish_point_cloud(points, colors, str(path), max_points)


def load_json_points(path: Path, max_points: int) -> PointCloud:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, dict):
        values = data.get("points") or data.get("xyz")
    else:
        values = data
    arr = np.asarray(values, dtype=np.float32)
    colors = arr[:, 3:6] if arr.ndim == 2 and arr.shape[1] >= 6 else None
    return finish_point_cloud(arr[:, :3], colors, str(path), max_points)


def load_nvblox(path: Path,
                max_points: int,
                surface_threshold_m: float,
                min_weight: float) -> PointCloud:
    con = sqlite3.connect(path)
    try:
        block_size_row = con.execute(
            "SELECT value_float FROM tsdf_layer_metadata WHERE param_name='block_size'"
        ).fetchone()
        if not block_size_row or block_size_row[0] is None:
            raise ValueError("nvblox tsdf metadata missing block_size")
        block_size = float(block_size_row[0])
        voxels_per_side = 8
        voxel_size = block_size / voxels_per_side
        threshold = surface_threshold_m if surface_threshold_m > 0 else voxel_size * 1.5

        color_blocks = {
            (int(ix), int(iy), int(iz)): bytes(blob)
            for ix, iy, iz, blob in con.execute(
                "SELECT index_x,index_y,index_z,data FROM color_layer_data"
            )
        }

        grid = np.stack(
            np.meshgrid(
                np.arange(voxels_per_side, dtype=np.float32),
                np.arange(voxels_per_side, dtype=np.float32),
                np.arange(voxels_per_side, dtype=np.float32),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        voxel_offsets = (grid + 0.5) * voxel_size
        tsdf_dtype = np.dtype([("distance", "<f4"), ("weight", "<f4")])
        color_dtype = np.dtype([("rgba", "u1", (4,)), ("weight", "<f4")])

        point_chunks: list[np.ndarray] = []
        color_chunks: list[np.ndarray] = []
        for ix, iy, iz, blob in con.execute(
            "SELECT index_x,index_y,index_z,data FROM tsdf_layer_data"
        ):
            tsdf = np.frombuffer(blob, dtype=tsdf_dtype, count=voxels_per_side ** 3)
            mask = (tsdf["weight"] >= min_weight) & (np.abs(tsdf["distance"]) <= threshold)
            if not np.any(mask):
                continue
            origin = np.array([ix, iy, iz], dtype=np.float32) * block_size
            point_chunks.append(origin[None, :] + voxel_offsets[mask])

            color_blob = color_blocks.get((int(ix), int(iy), int(iz)))
            if color_blob is not None and len(color_blob) >= color_dtype.itemsize * voxels_per_side ** 3:
                color = np.frombuffer(color_blob, dtype=color_dtype, count=voxels_per_side ** 3)
                rgb = color["rgba"][:, :3].copy()
                invalid = color["weight"] <= 0
                if np.any(invalid):
                    rgb[invalid] = np.array([145, 145, 145], dtype=np.uint8)
                color_chunks.append(rgb[mask])
            else:
                color_chunks.append(
                    np.full((int(np.count_nonzero(mask)), 3), 145, dtype=np.uint8)
                )
        if not point_chunks:
            return finish_point_cloud(
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                str(path),
                max_points,
            )
        points = np.concatenate(point_chunks, axis=0)
        colors = np.concatenate(color_chunks, axis=0)
        return finish_point_cloud(points, colors, str(path), max_points)
    finally:
        con.close()


def load_point_cloud(path: Path,
                     max_points: int,
                     surface_threshold_m: float,
                     min_weight: float) -> PointCloud:
    suffix = path.suffix.lower()
    if suffix == ".nvblox":
        return load_nvblox(path, max_points, surface_threshold_m, min_weight)
    if suffix == ".ply":
        return load_ply(path, max_points)
    if suffix == ".pcd":
        return load_pcd(path, max_points)
    if suffix in (".npy", ".npz"):
        return load_numpy_points(path, max_points)
    if suffix == ".json":
        return load_json_points(path, max_points)
    raise ValueError(f"unsupported point cloud file: {path}")


def find_auto_point_cloud(json_path: Path) -> Path | None:
    candidates: list[Path] = []
    roots = [
        json_path.parent / "pointclouds",
        json_path.parent / "points",
        json_path.parent.parent / "nvblox",
        json_path.parent.parent / "pointclouds",
        json_path.parent.parent / "points",
    ]
    for root in roots:
        if root.exists():
            for pattern in ("*.ply", "*.pcd", "*.npz", "*.npy", "*.json", "*.nvblox"):
                candidates.extend(root.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Roomie DSG Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f4;
      --panel: #ffffff;
      --text: #202428;
      --muted: #697178;
      --line: #d7ddd4;
      --accent: #0f7569;
      --warn: #b44f31;
      --canvas: #edf0ec;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100vh;
      overflow: hidden;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    .app {
      display: grid;
      grid-template-columns: minmax(280px, 340px) 1fr minmax(320px, 400px);
      height: 100vh;
      min-width: 0;
    }
    aside {
      min-width: 0;
      overflow: hidden;
      border-right: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      flex-direction: column;
    }
    aside.details { border-right: 0; border-left: 1px solid var(--line); }
    header {
      padding: 14px 16px 12px;
      border-bottom: 1px solid var(--line);
      min-width: 0;
    }
    h1, h2 {
      margin: 0;
      font-size: 16px;
      line-height: 1.25;
      font-weight: 650;
      letter-spacing: 0;
    }
    .meta {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .controls {
      display: grid;
      gap: 9px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    input[type="search"] {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      font-size: 14px;
      color: var(--text);
      background: #fff;
    }
    label.toggle, label.slider {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      user-select: none;
    }
    label.slider { justify-content: space-between; }
    input[type="range"] { width: 118px; }
    .list {
      overflow: auto;
      padding: 8px;
    }
    .row {
      width: 100%;
      display: grid;
      grid-template-columns: 52px minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      min-height: 48px;
      padding: 7px 8px;
      border: 1px solid transparent;
      border-radius: 7px;
      background: transparent;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }
    .row:hover { background: #f2f5ef; }
    .row.selected {
      border-color: #8bc2b8;
      background: #e8f4f1;
    }
    .id {
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      color: var(--muted);
    }
    .name {
      min-width: 0;
      font-size: 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .score {
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      color: var(--accent);
    }
    main {
      min-width: 0;
      min-height: 0;
      position: relative;
      background: var(--canvas);
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
      display: flex;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid rgba(216, 221, 213, 0.9);
      border-radius: 7px;
      padding: 7px 9px;
      backdrop-filter: blur(6px);
      max-width: calc(100% - 24px);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .detail-body {
      overflow: auto;
      padding: 12px 14px 18px;
    }
    .kv {
      display: grid;
      grid-template-columns: 122px minmax(0, 1fr);
      gap: 7px 12px;
      align-items: start;
      font-size: 13px;
      line-height: 1.35;
    }
    .key { color: var(--muted); }
    .value {
      min-width: 0;
      overflow-wrap: anywhere;
      font-variant-numeric: tabular-nums;
    }
    .section {
      margin-top: 16px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }
    .snapshot {
      position: relative;
      width: 100%;
      margin-bottom: 12px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #eef1ea;
    }
    .snapshot img {
      width: 100%;
      height: 100%;
      display: block;
    }
    .snapshot-box {
      position: absolute;
      border: 2px solid var(--warn);
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.85);
      pointer-events: none;
    }
    .empty {
      padding: 20px 16px;
      color: var(--muted);
      font-size: 14px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      height: 20px;
      padding: 0 7px;
      border-radius: 999px;
      background: #eef4f2;
      color: var(--accent);
      font-size: 12px;
    }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 280px 1fr; }
      aside.details {
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        width: min(400px, 46vw);
        box-shadow: -10px 0 24px rgba(0, 0, 0, 0.08);
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>Roomie DSG</h1>
        <div class="meta" id="graphMeta">Loading...</div>
      </header>
      <div class="controls">
        <input id="filter" type="search" placeholder="Filter id or label">
        <label class="toggle"><input id="showInactive" type="checkbox" checked> inactive / suppressed</label>
        <label class="toggle"><input id="showPoints" type="checkbox" checked> RGB point cloud</label>
        <label class="slider">point size <input id="pointSize" type="range" min="1" max="8" step="0.5" value="2.5"></label>
      </div>
      <div class="list" id="objectList"></div>
    </aside>
    <main>
      <canvas id="view"></canvas>
      <div class="hud" id="hud"></div>
    </main>
    <aside class="details">
      <header>
        <h2 id="detailTitle">Object</h2>
        <div class="meta" id="detailSub">Select an object</div>
      </header>
      <div class="detail-body" id="detailBody"></div>
    </aside>
  </div>
  <script>
    const INITIAL_OBJECT_ID = __INITIAL_OBJECT_ID__;
    const DEFAULT_POINT_SIZE = __POINT_SIZE__;
    const state = {
      graph: null,
      objects: [],
      filtered: [],
      selectedId: Number.isFinite(INITIAL_OBJECT_ID) ? INITIAL_OBJECT_ID : null,
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
      selectedPositionBuffer: null,
      selectedColorBuffer: null,
      selectedVertexCount: 0,
      viewProj: null
    };

    const canvas = document.getElementById('view');
    const objectList = document.getElementById('objectList');
    const filterInput = document.getElementById('filter');
    const showInactive = document.getElementById('showInactive');
    const showPoints = document.getElementById('showPoints');
    const pointSize = document.getElementById('pointSize');
    pointSize.value = String(DEFAULT_POINT_SIZE);

    function fmt(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
      return Number(value).toFixed(digits).replace(/\.?0+$/, '');
    }

    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }

    function valueRow(key, value) {
      return `<div class="key">${escapeHtml(key)}</div><div class="value">${escapeHtml(value)}</div>`;
    }

    function objectScore(object) {
      if (Number.isFinite(object.object_quality_score)) return object.object_quality_score;
      if (Number.isFinite(object.confidence)) return object.confidence;
      return 0;
    }

    function vec3(value) {
      if (!Array.isArray(value) || value.length < 3) return [0, 0, 0];
      return [Number(value[0]) || 0, Number(value[1]) || 0, Number(value[2]) || 0];
    }

    function vecText(value) {
      const v = vec3(value);
      return `${fmt(v[0])}, ${fmt(v[1])}, ${fmt(v[2])}`;
    }

    function hslToRgb(h, s, l) {
      h /= 360; s /= 100; l /= 100;
      const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
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
      return hslToRgb(Math.abs(hash) % 360, 58, 42);
    }

    function mat4Identity() {
      return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
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
      return [
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0
      ];
    }

    function normalize(v) {
      const len = Math.hypot(v[0], v[1], v[2]) || 1;
      return [v[0] / len, v[1] / len, v[2] / len];
    }

    function cross(a, b) {
      return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
      ];
    }

    function dot(a, b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    function mat4LookAt(eye, center, up) {
      const z = normalize([eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]]);
      const x = normalize(cross(up, z));
      const y = cross(z, x);
      return [
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
      ];
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
      const eye = currentEye();
      const view = mat4LookAt(eye, state.target, [0, 0, 1]);
      const proj = mat4Perspective(45 * Math.PI / 180, Math.max(1e-6, rect.width / rect.height), 0.01, Math.max(100, state.radius * 80));
      return mat4Multiply(proj, view);
    }

    function projectPoint(point, viewProj) {
      const x = point[0], y = point[1], z = point[2];
      const cx = viewProj[0] * x + viewProj[4] * y + viewProj[8] * z + viewProj[12];
      const cy = viewProj[1] * x + viewProj[5] * y + viewProj[9] * z + viewProj[13];
      const cz = viewProj[2] * x + viewProj[6] * y + viewProj[10] * z + viewProj[14];
      const cw = viewProj[3] * x + viewProj[7] * y + viewProj[11] * z + viewProj[15];
      const rect = canvas.getBoundingClientRect();
      const invW = cw ? 1 / cw : 1;
      return {
        x: (cx * invW * 0.5 + 0.5) * rect.width,
        y: (1 - (cy * invW * 0.5 + 0.5)) * rect.height,
        z: cz * invW
      };
    }

    function createShader(gl, type, source) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader));
      }
      return shader;
    }

    function createProgram(gl, vertex, fragment) {
      const program = gl.createProgram();
      gl.attachShader(program, createShader(gl, gl.VERTEX_SHADER, vertex));
      gl.attachShader(program, createShader(gl, gl.FRAGMENT_SHADER, fragment));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program));
      }
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
      const lineFragment = `
        precision mediump float;
        varying vec3 v_color;
        void main() { gl_FragColor = vec4(v_color, 1.0); }`;
      state.pointProgram = createProgram(gl, vertex, pointFragment);
      state.lineProgram = createProgram(gl, vertex, lineFragment);
      state.pointPositionBuffer = gl.createBuffer();
      state.pointColorBuffer = gl.createBuffer();
      state.boxPositionBuffer = gl.createBuffer();
      state.boxColorBuffer = gl.createBuffer();
      state.selectedPositionBuffer = gl.createBuffer();
      state.selectedColorBuffer = gl.createBuffer();
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
      const matrixLoc = gl.getUniformLocation(program, 'u_viewProj');
      gl.uniformMatrix4fv(matrixLoc, false, new Float32Array(state.viewProj));
      const pointSizeLoc = gl.getUniformLocation(program, 'u_pointSize');
      gl.uniform1f(pointSizeLoc, Number(pointSize.value) || DEFAULT_POINT_SIZE);
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
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const hx = Math.max(0, s[0]) * 0.5;
      const hy = Math.max(0, s[1]) * 0.5;
      const hz = Math.max(0, s[2]) * 0.5;
      const corners = [];
      for (const ix of [-1, 1]) for (const iy of [-1, 1]) for (const iz of [-1, 1]) {
        const x = ix * hx;
        const y = iy * hy;
        corners.push([c[0] + cy * x - sy * y, c[1] + sy * x + cy * y, c[2] + iz * hz]);
      }
      return corners;
    }

    const edges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];

    function appendBox(lines, colors, object, selected) {
      const corners = objectCorners(object);
      const base = selected ? [18, 20, 22] : colorForLabel(object.label, object.object_id);
      const inactive = object.active === false || object.publishable === false;
      const color = inactive && !selected ? [150, 155, 152] : base;
      for (const [a, b] of edges) {
        lines.push(...corners[a], ...corners[b]);
        colors.push(...color, ...color);
      }
    }

    function uploadBoxes() {
      const gl = state.gl;
      const lines = [];
      const colors = [];
      const selectedLines = [];
      const selectedColors = [];
      for (const object of state.filtered) {
        if (object.object_id === state.selectedId) {
          appendBox(selectedLines, selectedColors, object, true);
        } else {
          appendBox(lines, colors, object, false);
        }
      }
      state.boxVertexCount = lines.length / 3;
      state.selectedVertexCount = selectedLines.length / 3;
      gl.bindBuffer(gl.ARRAY_BUFFER, state.boxPositionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, state.boxColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(colors), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, state.selectedPositionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(selectedLines), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, state.selectedColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(selectedColors), gl.DYNAMIC_DRAW);
    }

    function updateScreenObjects() {
      const rect = canvas.getBoundingClientRect();
      state.screenObjects = [];
      for (const object of state.filtered) {
        const projected = objectCorners(object).map(p => projectPoint(p, state.viewProj));
        const visible = projected.some(p => p.z > -1 && p.z < 1);
        if (!visible) continue;
        const xs = projected.map(p => p.x);
        const ys = projected.map(p => p.y);
        const minX = Math.max(0, Math.min(...xs));
        const maxX = Math.min(rect.width, Math.max(...xs));
        const minY = Math.max(0, Math.min(...ys));
        const maxY = Math.min(rect.height, Math.max(...ys));
        if (maxX <= minX || maxY <= minY) continue;
        state.screenObjects.push({
          id: object.object_id,
          minX, maxX, minY, maxY,
          area: (maxX - minX) * (maxY - minY),
          depth: projected.reduce((acc, p) => acc + p.z, 0) / projected.length
        });
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
      setProgramAttributes(state.lineProgram, state.boxPositionBuffer, state.boxColorBuffer);
      gl.drawArrays(gl.LINES, 0, state.boxVertexCount);
      setProgramAttributes(state.lineProgram, state.selectedPositionBuffer, state.selectedColorBuffer);
      gl.drawArrays(gl.LINES, 0, state.selectedVertexCount);

      const pointText = state.pointCount > 0 ? `${state.pointCount.toLocaleString()} points` : 'no point cloud';
      document.getElementById('hud').textContent =
        `${state.filtered.length}/${state.objects.length} objects | ${pointText}`;
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
      let min = [Infinity, Infinity, Infinity];
      let max = [-Infinity, -Infinity, -Infinity];
      const include = (lo, hi) => {
        for (let i = 0; i < 3; ++i) {
          min[i] = Math.min(min[i], lo[i]);
          max[i] = Math.max(max[i], hi[i]);
        }
      };
      if (state.pointBounds) {
        include(state.pointBounds.min, state.pointBounds.max);
      }
      for (const object of state.objects) {
        const c = vec3(object.center_world);
        const s = vec3(object.size_m);
        include([c[0] - s[0] * 0.5, c[1] - s[1] * 0.5, c[2] - s[2] * 0.5],
                [c[0] + s[0] * 0.5, c[1] + s[1] * 0.5, c[2] + s[2] * 0.5]);
      }
      if (!Number.isFinite(min[0])) {
        min = [-1, -1, -1];
        max = [1, 1, 1];
      }
      state.target = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
      const dx = max[0] - min[0], dy = max[1] - min[1], dz = max[2] - min[2];
      state.radius = Math.max(1, Math.hypot(dx, dy, dz) * 0.5);
      state.distance = state.radius * 2.7;
    }

    function passesFilter(object) {
      if (!showInactive.checked && (object.active === false || object.publishable === false)) {
        return false;
      }
      const q = filterInput.value.trim().toLowerCase();
      if (!q) return true;
      const text = `${object.object_id} ${object.label || ''} ${object.semantic_id ?? ''}`.toLowerCase();
      return text.includes(q);
    }

    function updateFilter() {
      state.filtered = state.objects.filter(passesFilter);
      if (state.selectedId === null && state.filtered.length) {
        state.selectedId = state.filtered[0].object_id;
      }
      renderList();
      renderDetails();
      render();
    }

    function renderList() {
      objectList.innerHTML = '';
      if (!state.filtered.length) {
        objectList.innerHTML = '<div class="empty">No objects match the filter.</div>';
        return;
      }
      const fragment = document.createDocumentFragment();
      for (const object of state.filtered) {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'row' + (object.object_id === state.selectedId ? ' selected' : '');
        const snap = object.snapshot && Number.isFinite(object.snapshot.image_index) ? '<span class="pill">snap</span>' : '';
        row.innerHTML = `
          <div class="id">#${object.object_id}</div>
          <div class="name">${escapeHtml(object.label || 'object')}</div>
          <div class="score">${snap || fmt(objectScore(object), 2)}</div>`;
        row.addEventListener('click', () => selectObject(object.object_id));
        fragment.appendChild(row);
      }
      objectList.appendChild(fragment);
    }

    function snapshotImageFor(object) {
      const ref = object.snapshot;
      if (!ref || !Number.isFinite(ref.image_index)) return null;
      return (state.graph.snapshot_images || []).find(img => img.image_index === ref.image_index) || null;
    }

    function snapshotHtml(object) {
      const image = snapshotImageFor(object);
      const ref = object.snapshot;
      if (!image || !ref || !Array.isArray(ref.bbox_xyxy) || ref.bbox_xyxy.length < 4) {
        return '<div class="empty">No snapshot for this object.</div>';
      }
      const w = Number(image.width) || 1;
      const h = Number(image.height) || 1;
      const x0 = Math.max(0, Math.min(w, Math.min(Number(ref.bbox_xyxy[0]) || 0, Number(ref.bbox_xyxy[2]) || 0)));
      const x1 = Math.max(0, Math.min(w, Math.max(Number(ref.bbox_xyxy[0]) || 0, Number(ref.bbox_xyxy[2]) || 0)));
      const y0 = Math.max(0, Math.min(h, Math.min(Number(ref.bbox_xyxy[1]) || 0, Number(ref.bbox_xyxy[3]) || 0)));
      const y1 = Math.max(0, Math.min(h, Math.max(Number(ref.bbox_xyxy[1]) || 0, Number(ref.bbox_xyxy[3]) || 0)));
      const uri = `/snapshot-image/${encodeURIComponent(String(image.image_index))}`;
      return `
        <div class="snapshot" style="aspect-ratio:${w} / ${h};">
          <img src="${uri}" alt="">
          <div class="snapshot-box" style="left:${100 * x0 / w}%;top:${100 * y0 / h}%;width:${100 * Math.max(0, x1 - x0) / w}%;height:${100 * Math.max(0, y1 - y0) / h}%;"></div>
        </div>`;
    }

    function renderDetails() {
      const object = state.objects.find(o => o.object_id === state.selectedId);
      const title = document.getElementById('detailTitle');
      const sub = document.getElementById('detailSub');
      const body = document.getElementById('detailBody');
      if (!object) {
        title.textContent = 'Object';
        sub.textContent = 'Select an object';
        body.innerHTML = '<div class="empty">No object selected.</div>';
        return;
      }
      title.textContent = `Object #${object.object_id}`;
      sub.textContent = object.label || 'unlabeled';
      const status = [
        object.active === false ? 'inactive' : 'active',
        object.publishable === false ? 'suppressed' : 'publishable',
        object.geometry_status || 'unchecked'
      ].join(' / ');
      body.innerHTML = `
        ${snapshotHtml(object)}
        <div class="kv">
          ${valueRow('id', object.object_id)}
          ${valueRow('label', object.label || '')}
          ${valueRow('description', object.description || '')}
          ${valueRow('semantic id', object.semantic_id ?? '')}
          ${valueRow('score', fmt(objectScore(object), 4))}
          ${valueRow('confidence', fmt(object.confidence, 4))}
          ${valueRow('confidence mass', fmt(object.confidence_mass, 4))}
          ${valueRow('support count', object.support_count ?? '')}
          ${valueRow('status', status)}
        </div>
        <div class="section kv">
          ${valueRow('center world', vecText(object.center_world))}
          ${valueRow('size m', vecText(object.size_m))}
          ${valueRow('yaw rad', fmt(object.yaw_rad, 5))}
        </div>
        <div class="section kv">
          ${valueRow('geometry score', fmt(object.geometry_score, 4))}
          ${valueRow('in box points', object.geometry_in_box_points ?? 0)}
          ${valueRow('shell points', object.geometry_shell_points ?? 0)}
          ${valueRow('unique voxels', object.geometry_unique_voxels ?? 0)}
          ${valueRow('bad count', object.geometry_bad_count ?? 0)}
        </div>
        <div class="section kv">
          ${valueRow('source tracks', (object.source_track_ids || []).join(', '))}
          ${valueRow('source cameras', (object.source_cameras || []).join(', '))}
          ${valueRow('observations', (object.observation_timestamps_ns || []).length)}
          ${valueRow('snapshot image', object.snapshot?.image_index ?? '')}
          ${valueRow('snapshot quality', fmt(object.snapshot?.quality, 4))}
          ${valueRow('2d bbox', Array.isArray(object.snapshot?.bbox_xyxy) ? object.snapshot.bbox_xyxy.map(v => fmt(v, 1)).join(', ') : '')}
        </div>`;
    }

    function selectObject(id) {
      state.selectedId = id;
      renderList();
      renderDetails();
      render();
    }

    function pointerPosition(event) {
      const rect = canvas.getBoundingClientRect();
      return {x: event.clientX - rect.left, y: event.clientY - rect.top};
    }

    canvas.addEventListener('mousedown', event => {
      state.dragging = true;
      state.moved = false;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      canvas.classList.add('dragging');
    });
    window.addEventListener('mousemove', event => {
      if (!state.dragging) return;
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      if (Math.abs(dx) + Math.abs(dy) > 2) state.moved = true;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      state.yaw += dx * 0.006;
      state.pitch = Math.max(-1.35, Math.min(1.35, state.pitch + dy * 0.005));
      render();
    });
    window.addEventListener('mouseup', event => {
      if (!state.dragging) return;
      state.dragging = false;
      canvas.classList.remove('dragging');
      if (state.moved) return;
      const pos = pointerPosition(event);
      const hits = state.screenObjects
        .filter(o => pos.x >= o.minX && pos.x <= o.maxX && pos.y >= o.minY && pos.y <= o.maxY)
        .sort((a, b) => a.area - b.area || a.depth - b.depth);
      if (hits.length) selectObject(hits[0].id);
    });
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      const factor = Math.exp(event.deltaY * 0.001);
      state.distance = Math.max(state.radius * 0.08, Math.min(state.radius * 30, state.distance * factor));
      render();
    }, {passive: false});
    window.addEventListener('resize', render);
    filterInput.addEventListener('input', updateFilter);
    showInactive.addEventListener('change', updateFilter);
    showPoints.addEventListener('change', render);
    pointSize.addEventListener('input', render);

    async function loadGraph() {
      initGl();
      const [graphResponse] = await Promise.all([fetch('/graph.json'), loadPointCloud()]);
      state.graph = await graphResponse.json();
      state.objects = (state.graph.objects || []).slice()
        .sort((a, b) => Number(a.object_id) - Number(b.object_id));
      if (!state.objects.some(o => o.object_id === state.selectedId)) {
        state.selectedId = state.objects.length ? state.objects[0].object_id : null;
      }
      computeBounds();
      const snapCount = (state.graph.snapshot_images || []).length;
      document.getElementById('graphMeta').textContent =
        `${state.objects.length} objects | ${snapCount} snapshots | ${state.graph.world_frame || 'world'}`;
      updateFilter();
    }

    loadGraph().catch(error => {
      objectList.innerHTML = `<div class="empty">${escapeHtml(String(error))}</div>`;
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
            body = HTML.replace("__INITIAL_OBJECT_ID__", server.initial_object_id)
            body = body.replace("__POINT_SIZE__", server.point_size)
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/graph.json":
            payload = server.graph_json
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/points-meta.json":
            payload = server.points_meta_json
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/points.bin":
            payload = server.points_bytes
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/colors.bin":
            payload = server.colors_bytes
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
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
            payload = image_path.read_bytes()
            content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404, "not found")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("roomie_dsg_viewer: " + (fmt % args) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="optional DSG JSON file or pipeline YAML",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"pipeline YAML used to resolve saved DSG/map paths (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="explicit Roomie DSG JSON file; overrides YAML persistence paths",
    )
    parser.add_argument(
        "--points",
        type=Path,
        default=None,
        help="RGB point cloud: .nvblox, .ply, .pcd, .npy, .npz, or JSON",
    )
    parser.add_argument("--no-points", action="store_true", help="disable point cloud loading")
    parser.add_argument("--max-points", type=int, default=700000, help="point cloud downsample limit")
    parser.add_argument(
        "--surface-threshold-m",
        type=float,
        default=0.0,
        help="nvblox surface distance threshold; 0 uses 1.5 voxels",
    )
    parser.add_argument("--min-tsdf-weight", type=float, default=1.0e-4, help="nvblox TSDF weight gate")
    parser.add_argument("--point-size", type=float, default=2.5, help="initial WebGL point size")
    parser.add_argument("--host", default="127.0.0.1", help="server host")
    parser.add_argument("--port", type=int, default=8765, help="server port")
    parser.add_argument("--object-id", type=int, default=None, help="initial object id")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    return parser.parse_args()


def empty_point_cloud() -> PointCloud:
    return PointCloud(
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.uint8),
        "",
    )


def points_meta(point_cloud: PointCloud) -> dict[str, Any]:
    if point_cloud.count == 0:
        return {"count": 0, "source": point_cloud.source, "bounds": None}
    lo = point_cloud.points.min(axis=0).astype(float).tolist()
    hi = point_cloud.points.max(axis=0).astype(float).tolist()
    return {
        "count": point_cloud.count,
        "source": point_cloud.source,
        "bounds": {"min": lo, "max": hi},
    }


def main() -> int:
    args = parse_args()

    positional_json: Path | None = None
    config_path: Path | None = as_path(args.config)
    if args.path is not None:
        positional = as_path(args.path)
        if positional is not None and positional.suffix.lower() in (".yaml", ".yml"):
            config_path = positional
        else:
            positional_json = positional

    params = load_pipeline_params(config_path)
    json_path = as_path(args.json) or positional_json or resolve_json_from_config(params) or DEFAULT_JSON.resolve()
    if not json_path.exists():
        print(f"DSG JSON does not exist: {json_path}", file=sys.stderr)
        return 2

    with json_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    graph = normalize_graph(raw)
    graph_json = json.dumps(graph, separators=(",", ":")).encode("utf-8")

    point_cloud = empty_point_cloud()
    point_path: Path | None = None
    if not args.no_points:
        point_path = (
            as_path(args.points)
            if args.points is not None
            else resolve_points_from_config(params) or find_auto_point_cloud(json_path)
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

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.graph_json = graph_json  # type: ignore[attr-defined]
    server.snapshot_paths = snapshot_paths(graph, json_path)  # type: ignore[attr-defined]
    server.points_meta_json = json.dumps(points_meta(point_cloud), separators=(",", ":")).encode("utf-8")  # type: ignore[attr-defined]
    server.points_bytes = point_cloud.points.astype("<f4", copy=False).tobytes()  # type: ignore[attr-defined]
    server.colors_bytes = point_cloud.colors.astype("u1", copy=False).tobytes()  # type: ignore[attr-defined]
    server.initial_object_id = str(args.object_id) if args.object_id is not None else "NaN"  # type: ignore[attr-defined]
    server.point_size = repr(float(args.point_size))  # type: ignore[attr-defined]

    url = f"http://{args.host}:{server.server_port}/"
    print(f"Roomie DSG viewer: {url}")
    if config_path is not None:
        print(f"Using config: {config_path}")
    print(f"Loaded DSG: {json_path}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
