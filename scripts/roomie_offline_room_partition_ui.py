#!/usr/bin/env python3
"""DEPRECATED compatibility-only room editor.

This tool exports legacy/manual scene-graph JSON; that file is not the live
authoritative Roomie scene. New integrations must submit versioned room patches
to /roomie/mutate_scene so SceneReducer derives containment and persists the
result through the single-writer path.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlparse
import webbrowser

import numpy as np
from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
import yaml


ROOM_COLORS = [
    "#e45756",
    "#4c78a8",
    "#54a24b",
    "#f58518",
    "#b279a2",
    "#72b7b2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
    "#59a14f",
]


@dataclass
class GridSettings:
  resolution_m: float = 0.10
  slice_height_above_floor_m: float = 0.80
  slice_half_thickness_m: float = 0.08
  floor_band_m: float = 0.20
  floor_percentile: float = 8.0
  padding_m: float = 0.50
  max_grid_cells: int = 900000


@dataclass
class GridMap:
  width: int
  height: int
  resolution_m: float
  origin_x: float
  origin_y: float
  floor_z: float
  slice_z: float
  frame_id: str
  stamp_ns: int
  point_count: int
  occupied_count: int
  free_count: int
  unknown_count: int
  cells_top: bytes


def default_pipeline_config_path() -> Path:
  script_path = Path(__file__).resolve()
  for parent in script_path.parents:
    for candidate in (
        parent / "src" / "roomie" / "config" / "pipeline_nvblox.yaml",
        parent / "roomie" / "config" / "pipeline_nvblox.yaml",
        parent / "share" / "roomie" / "config" / "pipeline_nvblox.yaml",
        parent / "config" / "pipeline_nvblox.yaml",
    ):
      if candidate.exists():
        return candidate
  return script_path.parents[1] / "config" / "pipeline_nvblox.yaml"


def load_pipeline_params(config_path: Path) -> dict[str, Any]:
  if not config_path.exists():
    return {}
  with config_path.open("r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
  if "roomie_pipeline_node" in data:
    return data.get("roomie_pipeline_node", {}).get("ros__parameters", {}) or {}
  for value in data.values():
    if isinstance(value, dict) and "ros__parameters" in value:
      return value.get("ros__parameters", {}) or {}
  return data if isinstance(data, dict) else {}


def resolve_object_graph_path(params: dict[str, Any]) -> Path | None:
  persistence = params.get("persistence", {}) if isinstance(params, dict) else {}
  if not bool(persistence.get("load_scene_graph", False)):
    return None
  candidates: list[Path] = []
  load_path = str(persistence.get("scene_graph_load_path", "") or "").strip()
  save_path = str(persistence.get("scene_graph_save_path", "") or "").strip()
  if load_path:
    candidates.append(Path(load_path).expanduser())
  if save_path:
    base = Path(save_path).expanduser()
    if base.suffix.lower() == ".json":
      candidates.append(base)
    else:
      candidates.append(base / "latest.json")
  for path in candidates:
    if path.exists():
      return path
  return candidates[0] if candidates else None


def resolve_scene_graph_load_path(params: dict[str, Any]) -> Path | None:
  persistence = params.get("persistence", {}) if isinstance(params, dict) else {}
  if not bool(persistence.get("load_scene_graph", False)):
    return None
  configured = str(persistence.get("scene_graph_load_path", "") or "").strip()
  if not configured:
    return None
  return Path(configured).expanduser()


def resolve_scene_graph_save_base(params: dict[str, Any], override: Path | None) -> Path:
  if override is not None:
    return override.expanduser()
  persistence = params.get("persistence", {}) if isinstance(params, dict) else {}
  configured = str(persistence.get("scene_graph_save_path", "") or "").strip()
  if configured:
    path = Path(configured).expanduser()
    if path.suffix.lower() == ".json":
      return path.with_name(path.stem + "_manual_scene_graph.json")
    return path
  return Path.cwd() / "manual_scene_graphs"


def latest_scene_graph_path(save_base: Path) -> Path:
  if save_base.suffix.lower() == ".json":
    return save_base
  return save_base / "latest_manual_scene_graph.json"


def rooms_from_scene_graph(value: dict[str, Any]) -> list[dict[str, Any]]:
  rooms = []
  for index, room in enumerate(value.get("rooms", [])):
    if not isinstance(room, dict):
      continue
    min_xy = room.get("min_xy") or [None, None]
    max_xy = room.get("max_xy") or [None, None]
    if len(min_xy) < 2 or len(max_xy) < 2:
      center = room.get("center_world") or [None, None, None]
      size = room.get("size_m") or [0.0, 0.0, 0.0]
      if len(center) < 2 or len(size) < 2:
        continue
      try:
        cx = float(center[0])
        cy = float(center[1])
        sx = float(size[0])
        sy = float(size[1])
        min_xy = [cx - 0.5 * sx, cy - 0.5 * sy]
        max_xy = [cx + 0.5 * sx, cy + 0.5 * sy]
      except (TypeError, ValueError):
        continue
    try:
      rooms.append(normalize_room({
          "id": room.get("room_id", room.get("id", index + 1)),
          "label": room.get("label", "unknown"),
          "color": room.get("color", ""),
          "min_x": float(min_xy[0]),
          "min_y": float(min_xy[1]),
          "max_x": float(max_xy[0]),
          "max_y": float(max_xy[1]),
          "parent_room_ids": room.get("parent_room_ids", []),
          "child_room_ids": room.get("child_room_ids", []),
          "parent_object_ids": room.get("parent_object_ids", []),
          "child_object_ids": room.get("child_object_ids", []),
      }, index + 1))
    except (TypeError, ValueError):
      continue
  return rooms


def point_cloud_xyz(message: PointCloud2) -> np.ndarray:
  try:
    values = point_cloud2.read_points_numpy(
        message, field_names=("x", "y", "z"), skip_nans=True)
    array = np.asarray(values)
    if array.dtype.fields:
      points = np.stack([array["x"], array["y"], array["z"]], axis=-1)
    else:
      points = array.reshape((-1, 3))
    return points.astype(np.float32, copy=False)
  except Exception:
    points = [
        (float(point[0]), float(point[1]), float(point[2]))
        for point in point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True)
    ]
    if not points:
      return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def build_grid(points: np.ndarray,
               frame_id: str,
               stamp_ns: int,
               settings: GridSettings) -> GridMap | None:
  if points.size == 0:
    return None
  finite_mask = np.isfinite(points).all(axis=1)
  points = points[finite_mask]
  if points.shape[0] == 0:
    return None

  floor_z = float(np.percentile(points[:, 2], settings.floor_percentile))
  min_x = float(np.min(points[:, 0]) - settings.padding_m)
  max_x = float(np.max(points[:, 0]) + settings.padding_m)
  min_y = float(np.min(points[:, 1]) - settings.padding_m)
  max_y = float(np.max(points[:, 1]) + settings.padding_m)

  resolution = max(0.01, float(settings.resolution_m))
  area = max(0.01, (max_x - min_x) * (max_y - min_y))
  width = int(math.ceil((max_x - min_x) / resolution)) + 1
  height = int(math.ceil((max_y - min_y) / resolution)) + 1
  if width * height > settings.max_grid_cells:
    resolution = math.sqrt(area / float(settings.max_grid_cells))
    width = int(math.ceil((max_x - min_x) / resolution)) + 1
    height = int(math.ceil((max_y - min_y) / resolution)) + 1

  cells = np.zeros((height, width), dtype=np.uint8)
  x_index = np.floor((points[:, 0] - min_x) / resolution).astype(np.int64)
  y_index = np.floor((points[:, 1] - min_y) / resolution).astype(np.int64)
  valid = (
      (x_index >= 0) & (x_index < width) &
      (y_index >= 0) & (y_index < height)
  )

  free_mask = (
      valid &
      (points[:, 2] >= floor_z - 0.05) &
      (points[:, 2] <= floor_z + settings.floor_band_m)
  )
  cells[y_index[free_mask], x_index[free_mask]] = 1

  slice_z = floor_z + settings.slice_height_above_floor_m
  occupied_mask = (
      valid &
      (points[:, 2] >= slice_z - settings.slice_half_thickness_m) &
      (points[:, 2] <= slice_z + settings.slice_half_thickness_m)
  )
  cells[y_index[occupied_mask], x_index[occupied_mask]] = 2

  occupied_count = int(np.count_nonzero(cells == 2))
  free_count = int(np.count_nonzero(cells == 1))
  unknown_count = int(width * height - occupied_count - free_count)
  return GridMap(
      width=width,
      height=height,
      resolution_m=resolution,
      origin_x=min_x,
      origin_y=min_y,
      floor_z=floor_z,
      slice_z=slice_z,
      frame_id=frame_id,
      stamp_ns=stamp_ns,
      point_count=int(points.shape[0]),
      occupied_count=occupied_count,
      free_count=free_count,
      unknown_count=unknown_count,
      cells_top=np.flipud(cells).tobytes(),
  )


def normalize_room(raw: dict[str, Any], fallback_id: int) -> dict[str, Any]:
  room_id = int(raw.get("id", raw.get("room_id", fallback_id)))
  min_x = float(raw.get("min_x", 0.0))
  min_y = float(raw.get("min_y", 0.0))
  max_x = float(raw.get("max_x", min_x))
  max_y = float(raw.get("max_y", min_y))
  if min_x > max_x:
    min_x, max_x = max_x, min_x
  if min_y > max_y:
    min_y, max_y = max_y, min_y
  return {
      "id": room_id,
      "label": str(raw.get("label") or "unknown"),
      "color": str(raw.get("color") or ROOM_COLORS[room_id % len(ROOM_COLORS)]),
      "min_x": min_x,
      "min_y": min_y,
      "max_x": max_x,
      "max_y": max_y,
      "parent_room_ids": list(raw.get("parent_room_ids") or []),
      "child_room_ids": list(raw.get("child_room_ids") or []),
      "parent_object_ids": list(raw.get("parent_object_ids") or []),
      "child_object_ids": list(raw.get("child_object_ids") or []),
  }


def color_to_rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
  value = color.strip().lstrip("#")
  if len(value) != 6:
    return (0.8, 0.2, 0.2, alpha)
  try:
    red = int(value[0:2], 16) / 255.0
    green = int(value[2:4], 16) / 255.0
    blue = int(value[4:6], 16) / 255.0
    return (red, green, blue, alpha)
  except ValueError:
    return (0.8, 0.2, 0.2, alpha)


def point(x: float, y: float, z: float) -> Point:
  value = Point()
  value.x = x
  value.y = y
  value.z = z
  return value


class PartitionState:
  def __init__(self,
               params: dict[str, Any],
               settings: GridSettings,
               room_height_m: float,
               object_graph_path: Path | None,
               scene_graph_load_path: Path | None,
               scene_graph_save_base: Path):
    self.params = params
    self.settings = settings
    self.room_height_m = room_height_m
    self.object_graph_path = object_graph_path
    self.scene_graph_load_path = scene_graph_load_path
    self.scene_graph_save_base = scene_graph_save_base
    self.lock = threading.Lock()
    self.grid: GridMap | None = None
    self.last_points: np.ndarray | None = None
    self.last_frame_id = "world"
    self.last_stamp_ns = 0
    self.saved_frame_id = str(params.get("basic", {}).get("world_frame", "world"))
    self.saved_floor_z = 0.0
    self.saved_map_metadata: dict[str, Any] = {}
    self.rooms: list[dict[str, Any]] = []
    self.next_room_id = 1
    self.saved_path = ""
    self.last_error = ""
    self.load_saved_scene_graph()

  def load_saved_scene_graph(self) -> None:
    if self.scene_graph_load_path is None:
      return
    path = self.scene_graph_load_path
    if not path.exists():
      return
    try:
      with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
      rooms = rooms_from_scene_graph(value)
      map_metadata = value.get("map", {}) if isinstance(value.get("map", {}), dict) else {}
      next_id = 1
      for room in rooms:
        next_id = max(next_id, int(room["id"]) + 1)
      self.rooms = rooms
      self.next_room_id = next_id
      self.saved_path = str(path)
      self.saved_map_metadata = copy.deepcopy(map_metadata)
      self.saved_floor_z = float(map_metadata.get("floor_z", 0.0) or 0.0)
      self.saved_frame_id = str(value.get("world_frame") or self.saved_frame_id)
    except Exception as exc:
      self.last_error = f"failed to load saved manual scene graph {path}: {exc}"

  def update_from_cloud(self, message: PointCloud2) -> None:
    points = point_cloud_xyz(message)
    stamp_ns = int(message.header.stamp.sec) * 1000000000 + int(message.header.stamp.nanosec)
    frame_id = message.header.frame_id or self.last_frame_id
    with self.lock:
      settings = copy.deepcopy(self.settings)
    grid = build_grid(points, frame_id, stamp_ns, settings)
    with self.lock:
      self.last_points = points
      self.last_frame_id = frame_id
      self.last_stamp_ns = stamp_ns
      self.grid = grid

  def update_settings(self, values: dict[str, Any]) -> None:
    with self.lock:
      for key, attr in (
          ("resolution_m", "resolution_m"),
          ("slice_height_above_floor_m", "slice_height_above_floor_m"),
          ("slice_half_thickness_m", "slice_half_thickness_m"),
          ("floor_band_m", "floor_band_m"),
      ):
        if key in values:
          setattr(self.settings, attr, max(0.001, float(values[key])))
      points = None if self.last_points is None else self.last_points.copy()
      frame_id = self.last_frame_id
      stamp_ns = self.last_stamp_ns
      settings = copy.deepcopy(self.settings)
    grid = build_grid(points, frame_id, stamp_ns, settings) if points is not None else None
    with self.lock:
      if grid is not None:
        self.grid = grid

  def update_rooms(self, rooms: list[dict[str, Any]]) -> None:
    normalized = [normalize_room(room, i + 1) for i, room in enumerate(rooms)]
    next_id = 1
    for room in normalized:
      next_id = max(next_id, int(room["id"]) + 1)
    with self.lock:
      self.rooms = normalized
      self.next_room_id = next_id

  def upsert_room(self, room: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_room(room, self.next_room_id)
    with self.lock:
      replaced = False
      for index, existing in enumerate(self.rooms):
        if int(existing["id"]) == int(normalized["id"]):
          self.rooms[index] = normalized
          replaced = True
          break
      if not replaced:
        self.rooms.append(normalized)
      self.next_room_id = max(self.next_room_id, int(normalized["id"]) + 1)
    return normalized

  def delete_room(self, room_id: int) -> None:
    with self.lock:
      self.rooms = [room for room in self.rooms if int(room["id"]) != int(room_id)]

  def state_json(self) -> dict[str, Any]:
    with self.lock:
      grid = self.grid
      settings = copy.deepcopy(self.settings)
      rooms = copy.deepcopy(self.rooms)
      next_room_id = self.next_room_id
      saved_path = self.saved_path
      last_error = self.last_error
      object_graph_path = str(self.object_graph_path) if self.object_graph_path else ""
      scene_graph_load_path = str(self.scene_graph_load_path) if self.scene_graph_load_path else ""
      scene_graph_save_base = str(self.scene_graph_save_base)
    grid_json = None
    if grid is not None:
      grid_json = {
          "width": grid.width,
          "height": grid.height,
          "resolution_m": grid.resolution_m,
          "origin_x": grid.origin_x,
          "origin_y": grid.origin_y,
          "floor_z": grid.floor_z,
          "slice_z": grid.slice_z,
          "frame_id": grid.frame_id,
          "stamp_ns": grid.stamp_ns,
          "point_count": grid.point_count,
          "occupied_count": grid.occupied_count,
          "free_count": grid.free_count,
          "unknown_count": grid.unknown_count,
          "cells_b64": base64.b64encode(grid.cells_top).decode("ascii"),
      }
    return {
        "map": grid_json,
        "rooms": rooms,
        "next_room_id": next_room_id,
        "settings": {
            "resolution_m": settings.resolution_m,
            "slice_height_above_floor_m": settings.slice_height_above_floor_m,
            "slice_half_thickness_m": settings.slice_half_thickness_m,
            "floor_band_m": settings.floor_band_m,
        },
        "object_graph_path": object_graph_path,
        "scene_graph_load_path": scene_graph_load_path,
        "scene_graph_save_base": scene_graph_save_base,
        "saved_path": saved_path,
        "last_error": last_error,
    }

  def room_ids(self) -> set[int]:
    with self.lock:
      return {int(room["id"]) for room in self.rooms}

  def build_markers(self,
                    stamp: Any,
                    fallback_frame: str,
                    stale_room_ids: set[int] | None = None) -> MarkerArray:
    with self.lock:
      rooms = copy.deepcopy(self.rooms)
      grid = self.grid
    frame_id = grid.frame_id if grid is not None else (self.saved_frame_id or fallback_frame)
    floor_z = grid.floor_z if grid is not None else self.saved_floor_z

    markers = MarkerArray()
    for room_id in sorted(stale_room_ids or set()):
      for namespace, marker_id in (
          ("manual_room_boxes", room_id),
          ("manual_room_labels", 10000 + room_id),
      ):
        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = stamp
        delete_marker.ns = namespace
        delete_marker.id = marker_id
        delete_marker.action = Marker.DELETE
        markers.markers.append(delete_marker)

    for index, room in enumerate(rooms):
      min_x = float(room["min_x"])
      min_y = float(room["min_y"])
      max_x = float(room["max_x"])
      max_y = float(room["max_y"])
      if max_x - min_x <= 0.01 or max_y - min_y <= 0.01:
        continue
      red, green, blue, _ = color_to_rgba(room.get("color", ""), 1.0)
      z_min = floor_z
      z_max = floor_z + self.room_height_m
      marker = Marker()
      marker.header.frame_id = frame_id
      marker.header.stamp = stamp
      marker.ns = "manual_room_boxes"
      marker.id = int(room.get("id", index + 1))
      marker.type = Marker.LINE_LIST
      marker.action = Marker.ADD
      marker.pose.orientation.w = 1.0
      marker.scale.x = 0.045
      marker.color.r = red
      marker.color.g = green
      marker.color.b = blue
      marker.color.a = 1.0
      corners = {
          "swb": point(min_x, min_y, z_min),
          "seb": point(max_x, min_y, z_min),
          "neb": point(max_x, max_y, z_min),
          "nwb": point(min_x, max_y, z_min),
          "swt": point(min_x, min_y, z_max),
          "set": point(max_x, min_y, z_max),
          "net": point(max_x, max_y, z_max),
          "nwt": point(min_x, max_y, z_max),
      }
      for start, end in (
          ("swb", "seb"), ("seb", "neb"), ("neb", "nwb"), ("nwb", "swb"),
          ("swt", "set"), ("set", "net"), ("net", "nwt"), ("nwt", "swt"),
          ("swb", "swt"), ("seb", "set"), ("neb", "net"), ("nwb", "nwt"),
      ):
        marker.points.append(corners[start])
        marker.points.append(corners[end])
      markers.markers.append(marker)

      label = Marker()
      label.header.frame_id = frame_id
      label.header.stamp = stamp
      label.ns = "manual_room_labels"
      label.id = 10000 + int(room.get("id", index + 1))
      label.type = Marker.TEXT_VIEW_FACING
      label.action = Marker.ADD
      label.pose.position.x = 0.5 * (min_x + max_x)
      label.pose.position.y = 0.5 * (min_y + max_y)
      label.pose.position.z = floor_z + self.room_height_m + 0.15
      label.pose.orientation.w = 1.0
      label.scale.z = 0.24
      label.color.r = red
      label.color.g = green
      label.color.b = blue
      label.color.a = 1.0
      label.text = str(room.get("label") or "unknown")
      markers.markers.append(label)
    return markers

  def save_scene_graph(self) -> Path:
    with self.lock:
      rooms = copy.deepcopy(self.rooms)
      grid = self.grid
      object_graph_path = self.object_graph_path
      save_base = self.scene_graph_save_base
    object_graph = {}
    if object_graph_path is not None and object_graph_path.exists():
      with object_graph_path.open("r", encoding="utf-8") as handle:
        object_graph = json.load(handle)

    objects = copy.deepcopy(object_graph.get("objects", []))
    object_relations = copy.deepcopy(object_graph.get("relations", []))
    scene_relations = []
    scene_rooms = []
    floor_z = grid.floor_z if grid is not None else self.saved_floor_z
    frame_id = grid.frame_id if grid is not None else self.saved_frame_id
    map_metadata = copy.deepcopy(self.saved_map_metadata)
    if grid is not None:
      map_metadata = {
          "grid_width": grid.width,
          "grid_height": grid.height,
          "grid_resolution_m": grid.resolution_m,
          "grid_origin_xy": [grid.origin_x, grid.origin_y],
          "floor_z": floor_z,
          "slice_z": grid.slice_z,
      }

    for room in rooms:
      room_id = int(room["id"])
      min_x = float(room["min_x"])
      min_y = float(room["min_y"])
      max_x = float(room["max_x"])
      max_y = float(room["max_y"])
      scene_rooms.append({
          "room_id": room_id,
          "label": room.get("label") or "unknown",
          "min_xy": [min_x, min_y],
          "max_xy": [max_x, max_y],
          "center_world": [0.5 * (min_x + max_x), 0.5 * (min_y + max_y),
                           floor_z + 0.5 * self.room_height_m],
          "size_m": [max_x - min_x, max_y - min_y, self.room_height_m],
          "height_m": self.room_height_m,
          "color": room.get("color", ""),
          "parent_room_ids": list(room.get("parent_room_ids") or []),
          "child_room_ids": list(room.get("child_room_ids") or []),
          "parent_object_ids": list(room.get("parent_object_ids") or []),
          "child_object_ids": list(room.get("child_object_ids") or []),
      })

    for obj in objects:
      center = obj.get("center_world") or [None, None, None]
      parent_room_ids = []
      if isinstance(center, list) and len(center) >= 2:
        try:
          x = float(center[0])
          y = float(center[1])
          for room in rooms:
            if (float(room["min_x"]) <= x <= float(room["max_x"]) and
                float(room["min_y"]) <= y <= float(room["max_y"])):
              parent_room_ids.append(int(room["id"]))
        except (TypeError, ValueError):
          parent_room_ids = []
      obj["parent_room_ids"] = parent_room_ids
      obj.setdefault("parent_object_ids", [])
      obj.setdefault("child_object_ids", [])
      object_id = int(obj.get("object_id", -1))
      for room_id in parent_room_ids:
        scene_relations.append({
            "source": {"type": "room", "id": room_id},
            "target": {"type": "object", "id": object_id},
            "relation_type": "room_contains_object",
            "confidence": 1.0,
        })

    scene = {
        "format": "roomie_manual_scene_graph",
        "format_version": 1,
        "saved_time": datetime.now().isoformat(timespec="seconds"),
        "world_frame": frame_id,
        "source_object_graph_path": str(object_graph_path) if object_graph_path else "",
        "object_graph": object_graph,
        "map": map_metadata,
        "root": {
            "room_ids": [room["room_id"] for room in scene_rooms],
            "object_ids": [int(obj.get("object_id", -1)) for obj in objects],
        },
        "rooms": scene_rooms,
        "objects": objects,
        "relations": object_relations + scene_relations,
    }

    saved_path = write_scene_graph(scene, save_base)
    with self.lock:
      self.saved_path = str(saved_path)
      self.last_error = ""
    return saved_path


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = path.with_suffix(path.suffix + ".tmp")
  with tmp_path.open("w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2)
    handle.write("\n")
  os.replace(tmp_path, path)


def write_scene_graph(value: dict[str, Any], save_base: Path) -> Path:
  if save_base.suffix.lower() == ".json":
    write_json_atomic(save_base, value)
    return save_base
  timestamp = datetime.now().strftime("%m%d_%H%M%S")
  primary = save_base / f"roomie_manual_scene_graph_{timestamp}.json"
  latest = save_base / "latest_manual_scene_graph.json"
  write_json_atomic(primary, value)
  write_json_atomic(latest, value)
  return primary


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roomie Manual Room Partition</title>
<style>
:root {
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #191b1f;
  color: #eceff4;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  display: grid;
  grid-template-columns: 1fr 320px;
  background: #191b1f;
}
main {
  min-width: 0;
  min-height: 100vh;
  display: grid;
  grid-template-rows: 44px 1fr;
}
.topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 14px;
  border-bottom: 1px solid #30343b;
  background: #202329;
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
}
.status { color: #aeb7c6; overflow: hidden; text-overflow: ellipsis; }
.stage {
  position: relative;
  overflow: hidden;
}
canvas {
  width: 100%;
  height: 100%;
  display: block;
  background: #24272d;
}
aside {
  min-height: 100vh;
  border-left: 1px solid #30343b;
  background: #202329;
  display: grid;
  grid-template-rows: auto auto 1fr auto;
}
.section {
  padding: 14px;
  border-bottom: 1px solid #30343b;
}
.title {
  font-size: 12px;
  font-weight: 700;
  color: #aeb7c6;
  text-transform: uppercase;
  letter-spacing: 0;
  margin-bottom: 10px;
}
label {
  display: grid;
  gap: 5px;
  font-size: 12px;
  color: #aeb7c6;
  margin: 0 0 10px;
}
input {
  width: 100%;
  height: 32px;
  border: 1px solid #454b55;
  background: #16191e;
  color: #eceff4;
  border-radius: 4px;
  padding: 0 8px;
}
.row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.row.actions {
  grid-template-columns: 1fr 1fr 1fr;
}
button {
  height: 32px;
  border: 1px solid #4b5665;
  border-radius: 4px;
  background: #2c323a;
  color: #eceff4;
  cursor: pointer;
  font-weight: 600;
}
button:hover { background: #38414d; }
button.danger { border-color: #8f4c4c; color: #ffb3b3; }
.rooms {
  overflow: auto;
  min-height: 0;
}
.room {
  width: 100%;
  display: grid;
  grid-template-columns: 16px 1fr auto;
  align-items: center;
  gap: 8px;
  padding: 9px 14px;
  border-bottom: 1px solid #30343b;
  cursor: pointer;
}
.room.selected { background: #303947; }
.room.pending .swatch {
  outline: 2px solid #e6b450;
  outline-offset: 2px;
}
.swatch {
  width: 14px;
  height: 14px;
  border-radius: 2px;
}
.room-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
}
.room-size {
  font-variant-numeric: tabular-nums;
  color: #aeb7c6;
  font-size: 11px;
}
.footer {
  padding: 12px 14px;
  color: #aeb7c6;
  font-size: 11px;
  line-height: 1.5;
  border-top: 1px solid #30343b;
}
@media (max-width: 900px) {
  body { grid-template-columns: 1fr; grid-template-rows: 58vh auto; }
  main { min-height: 58vh; }
  aside { min-height: 42vh; border-left: 0; border-top: 1px solid #30343b; }
}
</style>
</head>
<body>
<main>
  <div class="topbar">
    <strong>Manual Rooms — compatibility export only</strong>
    <span id="mapStatus" class="status">waiting for map</span>
  </div>
  <div class="stage"><canvas id="canvas"></canvas></div>
</main>
<aside>
  <div class="section">
    <div class="title">Selected</div>
    <label>Room label<input id="labelInput" value="unknown"></label>
    <div class="row actions">
      <button id="confirmBtn">Confirm</button>
      <button id="deleteBtn" class="danger">Delete</button>
      <button id="saveBtn">Export compatibility JSON</button>
    </div>
  </div>
  <div class="section">
    <div class="title">Slice</div>
    <div class="row">
      <label>Height m<input id="sliceHeight" type="number" step="0.05"></label>
      <label>Half thick<input id="sliceThickness" type="number" step="0.01"></label>
    </div>
    <div class="row">
      <label>Resolution<input id="resolution" type="number" step="0.01"></label>
      <label>Floor band<input id="floorBand" type="number" step="0.01"></label>
    </div>
    <button id="rebuildBtn">Rebuild Map</button>
  </div>
  <div id="rooms" class="rooms"></div>
  <div id="footer" class="footer"></div>
</aside>
<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const offscreen = document.createElement("canvas");
const offctx = offscreen.getContext("2d");
const mapStatus = document.getElementById("mapStatus");
const roomsEl = document.getElementById("rooms");
const footerEl = document.getElementById("footer");
const labelInput = document.getElementById("labelInput");
const confirmBtn = document.getElementById("confirmBtn");
const sliceHeight = document.getElementById("sliceHeight");
const sliceThickness = document.getElementById("sliceThickness");
const resolution = document.getElementById("resolution");
const floorBand = document.getElementById("floorBand");
let map = null;
let cells = null;
let rooms = [];
let nextRoomId = 1;
let selectedId = null;
let drag = null;
let view = {scale: 1, ox: 0, oy: 0};
let stateLoaded = false;
let localDirty = false;
const pendingRoomIds = new Set();
const colors = ["#e45756", "#4c78a8", "#54a24b", "#f58518", "#b279a2", "#72b7b2", "#ff9da6", "#9d755d"];

function decodeCells(encoded) {
  const binary = atob(encoded);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; ++i) out[i] = binary.charCodeAt(i);
  return out;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width));
  canvas.height = Math.max(1, Math.floor(rect.height));
  draw();
}

function updateView() {
  if (!map) return;
  const scale = Math.min(canvas.width / map.width, canvas.height / map.height);
  view.scale = Number.isFinite(scale) && scale > 0 ? scale : 1;
  view.ox = 0.5 * (canvas.width - map.width * view.scale);
  view.oy = 0.5 * (canvas.height - map.height * view.scale);
}

function cellToWorld(cx, cy) {
  return {
    x: map.origin_x + cx * map.resolution_m,
    y: map.origin_y + (map.height - cy) * map.resolution_m,
  };
}

function worldToCell(x, y) {
  return {
    x: (x - map.origin_x) / map.resolution_m,
    y: map.height - (y - map.origin_y) / map.resolution_m,
  };
}

function screenToCell(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return {
    x: (x - view.ox) / view.scale,
    y: (y - view.oy) / view.scale,
  };
}

function rectPixels(room) {
  const a = worldToCell(room.min_x, room.max_y);
  const b = worldToCell(room.max_x, room.min_y);
  return {x: a.x, y: a.y, w: b.x - a.x, h: b.y - a.y};
}

function normalizeRoom(room) {
  if (room.min_x > room.max_x) [room.min_x, room.max_x] = [room.max_x, room.min_x];
  if (room.min_y > room.max_y) [room.min_y, room.max_y] = [room.max_y, room.min_y];
  const minSize = map ? map.resolution_m : 0.05;
  if (room.max_x - room.min_x < minSize) room.max_x = room.min_x + minSize;
  if (room.max_y - room.min_y < minSize) room.max_y = room.min_y + minSize;
}

function selectedRoom() {
  return rooms.find(room => room.id === selectedId) || null;
}

function isEditingLabel() {
  return document.activeElement === labelInput;
}

function markPending(roomId) {
  if (roomId === null || roomId === undefined) return;
  pendingRoomIds.add(roomId);
  localDirty = true;
  footerEl.textContent = `pending room ${roomId}`;
}

function clearPending(roomId) {
  pendingRoomIds.delete(roomId);
  localDirty = pendingRoomIds.size > 0;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!map || !cells) {
    ctx.fillStyle = "#24272d";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#aeb7c6";
    ctx.font = "13px system-ui";
    ctx.fillText("waiting for /roomie/map_surface", 18, 28);
    return;
  }
  updateView();
  if (offscreen.width !== map.width || offscreen.height !== map.height) {
    offscreen.width = map.width;
    offscreen.height = map.height;
  }
  const image = offctx.createImageData(map.width, map.height);
  for (let i = 0; i < cells.length; ++i) {
    const v = cells[i];
    const p = i * 4;
    if (v === 2) {
      image.data[p] = 31; image.data[p + 1] = 35; image.data[p + 2] = 43;
    } else if (v === 1) {
      image.data[p] = 214; image.data[p + 1] = 219; image.data[p + 2] = 226;
    } else {
      image.data[p] = 99; image.data[p + 1] = 106; image.data[p + 2] = 116;
    }
    image.data[p + 3] = 255;
  }
  offctx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, view.ox, view.oy, map.width * view.scale, map.height * view.scale);

  for (const room of rooms) {
    const r = rectPixels(room);
    const x = view.ox + r.x * view.scale;
    const y = view.oy + r.y * view.scale;
    const w = r.w * view.scale;
    const h = r.h * view.scale;
    ctx.save();
    ctx.strokeStyle = room.color || "#e45756";
    ctx.fillStyle = (room.color || "#e45756") + "24";
    ctx.lineWidth = room.id === selectedId ? 3 : 2;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.font = "12px system-ui";
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(room.label || "unknown", x + 6, y + 16);
    if (room.id === selectedId) {
      ctx.fillStyle = room.color || "#e45756";
      for (const [hx, hy] of [[x,y],[x+w,y],[x,y+h],[x+w,y+h]]) {
        ctx.fillRect(hx - 4, hy - 4, 8, 8);
      }
    }
    ctx.restore();
  }
}

function hitTest(cell) {
  const px = cell.x;
  const py = cell.y;
  const tolerance = 8 / view.scale;
  for (let i = rooms.length - 1; i >= 0; --i) {
    const room = rooms[i];
    const r = rectPixels(room);
    const inside = px >= r.x - tolerance && px <= r.x + r.w + tolerance &&
                   py >= r.y - tolerance && py <= r.y + r.h + tolerance;
    if (!inside) continue;
    const west = Math.abs(px - r.x) <= tolerance;
    const east = Math.abs(px - (r.x + r.w)) <= tolerance;
    const north = Math.abs(py - r.y) <= tolerance;
    const south = Math.abs(py - (r.y + r.h)) <= tolerance;
    let mode = "";
    if (north) mode += "n";
    if (south) mode += "s";
    if (west) mode += "w";
    if (east) mode += "e";
    return {room, mode: mode || "move"};
  }
  return null;
}

function selectRoom(id) {
  selectedId = id;
  const room = selectedRoom();
  if (room) labelInput.value = room.label || "unknown";
  renderRooms();
  draw();
}

function newRoomAt(world) {
  const room = {
    id: nextRoomId++,
    label: labelInput.value.trim() || "unknown",
    color: colors[rooms.length % colors.length],
    min_x: world.x,
    min_y: world.y,
    max_x: world.x,
    max_y: world.y,
    parent_room_ids: [],
    child_room_ids: [],
    parent_object_ids: [],
    child_object_ids: [],
  };
  rooms.push(room);
  selectedId = room.id;
  markPending(room.id);
  return room;
}

function applyDrag(world) {
  if (!drag) return;
  const room = drag.room;
  if (drag.mode === "move") {
    const dx = world.x - drag.startWorld.x;
    const dy = world.y - drag.startWorld.y;
    room.min_x = drag.startRoom.min_x + dx;
    room.max_x = drag.startRoom.max_x + dx;
    room.min_y = drag.startRoom.min_y + dy;
    room.max_y = drag.startRoom.max_y + dy;
  } else if (drag.mode === "create") {
    room.min_x = Math.min(drag.anchor.x, world.x);
    room.max_x = Math.max(drag.anchor.x, world.x);
    room.min_y = Math.min(drag.anchor.y, world.y);
    room.max_y = Math.max(drag.anchor.y, world.y);
  } else {
    if (drag.mode.includes("w")) room.min_x = world.x;
    if (drag.mode.includes("e")) room.max_x = world.x;
    if (drag.mode.includes("s")) room.min_y = world.y;
    if (drag.mode.includes("n")) room.max_y = world.y;
  }
  normalizeRoom(room);
}

canvas.addEventListener("mousedown", event => {
  if (!map) return;
  const cell = screenToCell(event);
  const world = cellToWorld(cell.x, cell.y);
  const hit = hitTest(cell);
  if (hit) {
    selectRoom(hit.room.id);
    drag = {
      room: hit.room,
      mode: hit.mode,
      startWorld: world,
      startRoom: {...hit.room},
    };
  } else {
    const room = newRoomAt(world);
    drag = {room, mode: "create", anchor: world, startWorld: world, startRoom: {...room}};
    renderRooms();
  }
  draw();
});

window.addEventListener("mousemove", event => {
  if (!drag || !map) return;
  const cell = screenToCell(event);
  const world = cellToWorld(cell.x, cell.y);
  applyDrag(world);
  markPending(drag.room.id);
  renderRooms();
  draw();
});

window.addEventListener("mouseup", () => {
  if (!drag) return;
  drag = null;
});

window.addEventListener("keydown", event => {
  if (event.key === "Delete" || event.key === "Backspace") {
    if (document.activeElement && document.activeElement.tagName === "INPUT") return;
    deleteSelected();
  }
});

labelInput.addEventListener("input", () => {
  const room = selectedRoom();
  if (!room) return;
  room.label = labelInput.value.trim() || "unknown";
  markPending(room.id);
  renderRooms();
  draw();
});

confirmBtn.addEventListener("click", confirmSelectedRoom);
document.getElementById("deleteBtn").addEventListener("click", deleteSelected);
document.getElementById("saveBtn").addEventListener("click", saveGraph);
document.getElementById("rebuildBtn").addEventListener("click", sendSettings);

async function deleteSelected() {
  if (selectedId === null) return;
  const deletedId = selectedId;
  rooms = rooms.filter(room => room.id !== selectedId);
  selectedId = rooms.length ? rooms[rooms.length - 1].id : null;
  const room = selectedRoom();
  labelInput.value = room ? room.label : "unknown";
  clearPending(deletedId);
  renderRooms();
  draw();
  const response = await fetch("/delete_room", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id: deletedId}),
  });
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    footerEl.textContent = result.error || "failed to delete room";
    return;
  }
  footerEl.textContent = `deleted room ${deletedId}`;
}

function renderRooms() {
  roomsEl.innerHTML = "";
  for (const room of rooms) {
    const el = document.createElement("div");
    el.className = "room" + (room.id === selectedId ? " selected" : "") +
      (pendingRoomIds.has(room.id) ? " pending" : "");
    el.innerHTML = `<div class="swatch" style="background:${room.color}"></div>
      <div class="room-name"></div><div class="room-size"></div>`;
    el.querySelector(".room-name").textContent = room.label || "unknown";
    el.querySelector(".room-size").textContent =
      `${(room.max_x - room.min_x).toFixed(1)}x${(room.max_y - room.min_y).toFixed(1)}m`;
    el.addEventListener("click", () => selectRoom(room.id));
    roomsEl.appendChild(el);
  }
}

async function fetchState() {
  const response = await fetch("/state");
  const state = await response.json();
  if (state.map) {
    map = state.map;
    cells = decodeCells(state.map.cells_b64);
    mapStatus.textContent =
      `${map.frame_id} ${map.width}x${map.height} res=${map.resolution_m.toFixed(3)}m floor=${map.floor_z.toFixed(2)} slice=${map.slice_z.toFixed(2)}`;
  } else {
    mapStatus.textContent = "waiting for /roomie/map_surface";
  }
  const firstLoad = !stateLoaded;
  if (firstLoad || (!localDirty && !isEditingLabel())) {
    rooms = state.rooms || [];
    nextRoomId = state.next_room_id || 1;
    pendingRoomIds.clear();
    if (firstLoad && rooms.length) selectedId = rooms[0].id;
    if (selectedId !== null && !rooms.some(room => room.id === selectedId)) {
      selectedId = rooms.length ? rooms[0].id : null;
    }
    const room = selectedRoom();
    if (room && !isEditingLabel()) labelInput.value = room.label || "unknown";
    stateLoaded = true;
  }
  if (state.settings) {
    sliceHeight.value = state.settings.slice_height_above_floor_m;
    sliceThickness.value = state.settings.slice_half_thickness_m;
    resolution.value = state.settings.resolution_m;
    floorBand.value = state.settings.floor_band_m;
  }
  if (!localDirty && !isEditingLabel()) {
    footerEl.textContent = state.last_error ? `error ${state.last_error}` :
      (state.saved_path ? `saved ${state.saved_path}` :
      (state.object_graph_path ? `instance map ${state.object_graph_path}` : "instance map not found"));
  }
  renderRooms();
  draw();
}

async function sendRooms() {
  const response = await fetch("/rooms", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({rooms}),
  });
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    footerEl.textContent = result.error || "failed to sync rooms";
    return false;
  }
  pendingRoomIds.clear();
  localDirty = false;
  return true;
}

async function confirmSelectedRoom() {
  const room = selectedRoom();
  if (!room) return;
  const response = await fetch("/room", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({room}),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    footerEl.textContent = result.error || "failed to confirm room";
    return;
  }
  if (result.room) {
    const index = rooms.findIndex(value => value.id === result.room.id);
    if (index >= 0) rooms[index] = result.room;
  }
  clearPending(room.id);
  renderRooms();
  draw();
  footerEl.textContent = `confirmed room ${room.id}`;
}

async function sendSettings() {
  await fetch("/settings", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      resolution_m: Number(resolution.value),
      slice_height_above_floor_m: Number(sliceHeight.value),
      slice_half_thickness_m: Number(sliceThickness.value),
      floor_band_m: Number(floorBand.value),
    }),
  });
  await fetchState();
}

async function saveGraph() {
  if (!await sendRooms()) return;
  const response = await fetch("/save", {method: "POST"});
  const result = await response.json();
  footerEl.textContent = result.path ? `saved ${result.path}` : (result.error || "save failed");
}

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
fetchState().catch(() => {});
setInterval(() => fetchState().catch(() => {}), 2000);
</script>
</body>
</html>
"""


class PartitionRequestHandler(BaseHTTPRequestHandler):
  def log_message(self, fmt: str, *args: Any) -> None:
    return

  @property
  def partition_node(self) -> "OfflineRoomPartitionNode":
    return self.server.partition_node  # type: ignore[attr-defined]

  def send_json(self, value: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(value).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def read_json_body(self) -> dict[str, Any]:
    length = int(self.headers.get("Content-Length", "0"))
    if length <= 0:
      return {}
    data = self.rfile.read(length)
    return json.loads(data.decode("utf-8"))

  def do_GET(self) -> None:
    path = urlparse(self.path).path
    if path == "/":
      body = HTML.encode("utf-8")
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
      return
    if path == "/state":
      self.send_json(self.partition_node.state.state_json())
      return
    self.send_error(404)

  def do_POST(self) -> None:
    path = urlparse(self.path).path
    try:
      if path == "/rooms":
        payload = self.read_json_body()
        self.partition_node.state.update_rooms(list(payload.get("rooms") or []))
        self.partition_node.publish_room_markers()
        self.send_json({"ok": True})
        return
      if path == "/room":
        payload = self.read_json_body()
        room = self.partition_node.state.upsert_room(dict(payload.get("room") or {}))
        self.partition_node.publish_room_markers()
        self.send_json({"ok": True, "room": room})
        return
      if path == "/delete_room":
        payload = self.read_json_body()
        self.partition_node.state.delete_room(int(payload.get("id", -1)))
        self.partition_node.publish_room_markers()
        self.send_json({"ok": True})
        return
      if path == "/settings":
        payload = self.read_json_body()
        self.partition_node.state.update_settings(payload)
        self.partition_node.publish_room_markers()
        self.send_json({"ok": True})
        return
      if path == "/save":
        saved_path = self.partition_node.state.save_scene_graph()
        self.partition_node.publish_room_markers()
        self.send_json({"ok": True, "path": str(saved_path)})
        return
    except Exception as exc:
      with self.partition_node.state.lock:
        self.partition_node.state.last_error = str(exc)
      self.send_json({"ok": False, "error": str(exc)}, status=500)
      return
    self.send_error(404)


class OfflineRoomPartitionNode(Node):
  def __init__(self, args: argparse.Namespace, params: dict[str, Any]):
    super().__init__("roomie_offline_room_partition_ui")
    self.world_frame = str(params.get("basic", {}).get("world_frame", "world"))
    map_topic = args.map_topic or str(
        params.get("topics", {}).get("tsdf_output_topic", "/roomie/map_surface"))
    marker_topic = args.marker_topic
    settings = GridSettings(
        resolution_m=args.grid_resolution,
        slice_height_above_floor_m=args.slice_height,
        slice_half_thickness_m=args.slice_half_thickness,
        floor_band_m=args.floor_band,
        max_grid_cells=args.max_grid_cells,
    )
    object_graph_path = resolve_object_graph_path(params)
    scene_graph_load_path = resolve_scene_graph_load_path(params)
    save_base = resolve_scene_graph_save_base(params, args.scene_graph_save_path)
    self.state = PartitionState(
        params=params,
        settings=settings,
        room_height_m=args.room_height,
        object_graph_path=object_graph_path,
        scene_graph_load_path=scene_graph_load_path,
        scene_graph_save_base=save_base,
    )
    self.last_room_marker_ids: set[int] = set()
    self.marker_clear_published = False

    marker_qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    self.marker_pub = self.create_publisher(MarkerArray, marker_topic, marker_qos)
    self.map_sub = self.create_subscription(
        PointCloud2,
        map_topic,
        self.map_callback,
        QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
    )
    self.timer = self.create_timer(1.0, self.publish_room_markers)
    self.get_logger().info(f"subscribing map surface: {map_topic}")
    self.get_logger().info(f"publishing manual room boxes: {marker_topic}")
    if scene_graph_load_path is not None:
      self.get_logger().info(f"loading scene graph: {scene_graph_load_path}")
    self.get_logger().info(f"manual scene graph save base: {save_base}")
    if self.state.rooms:
      self.get_logger().info(f"loaded manual rooms: {len(self.state.rooms)}")

  def map_callback(self, message: PointCloud2) -> None:
    start = time.monotonic()
    self.state.update_from_cloud(message)
    elapsed_ms = (time.monotonic() - start) * 1000.0
    state = self.state.state_json()
    grid = state.get("map")
    if grid:
      self.get_logger().info(
          "rebuilt ternary map "
          f"{grid['width']}x{grid['height']} "
          f"points={grid['point_count']} "
          f"occupied={grid['occupied_count']} "
          f"free={grid['free_count']} "
          f"{elapsed_ms:.1f}ms")

  def publish_room_markers(self) -> None:
    stamp = self.get_clock().now().to_msg()
    if not self.marker_clear_published:
      self.marker_pub.publish(self.build_clear_markers(stamp))
      self.marker_clear_published = True
    current_ids = self.state.room_ids()
    stale_ids = self.last_room_marker_ids - current_ids
    self.marker_pub.publish(self.state.build_markers(stamp, self.world_frame, stale_ids))
    self.last_room_marker_ids = current_ids

  def build_clear_markers(self, stamp: Any) -> MarkerArray:
    markers = MarkerArray()
    for namespace in ("manual_room_boxes", "manual_room_labels"):
      marker = Marker()
      marker.header.frame_id = self.world_frame
      marker.header.stamp = stamp
      marker.ns = namespace
      marker.id = 0
      marker.action = Marker.DELETEALL
      markers.markers.append(marker)
    return markers


def start_http_server(node: OfflineRoomPartitionNode,
                      host: str,
                      port: int) -> ThreadingHTTPServer:
  server = ThreadingHTTPServer((host, port), PartitionRequestHandler)
  server.partition_node = node  # type: ignore[attr-defined]
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  return server


def parse_args() -> tuple[argparse.Namespace, list[str]]:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pipeline-config", type=Path,
                      default=default_pipeline_config_path())
  parser.add_argument("--map-topic", default="")
  parser.add_argument("--marker-topic", default="/roomie/manual_room_boxes")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--no-browser", action="store_true")
  parser.add_argument("--marker-only", action="store_true")
  parser.add_argument("--grid-resolution", type=float, default=0.10)
  parser.add_argument("--slice-height", type=float, default=0.80)
  parser.add_argument("--slice-half-thickness", type=float, default=0.08)
  parser.add_argument("--floor-band", type=float, default=0.20)
  parser.add_argument("--max-grid-cells", type=int, default=900000)
  parser.add_argument("--room-height", type=float, default=2.30)
  parser.add_argument("--scene-graph-save-path", type=Path, default=None)
  return parser.parse_known_args()


def main() -> None:
  args, ros_args = parse_args()
  params = load_pipeline_params(args.pipeline_config.expanduser())
  rclpy.init(args=ros_args)
  node = OfflineRoomPartitionNode(args, params)
  node.get_logger().warning(
      "DEPRECATED compatibility-only UI: exported JSON does not mutate the "
      "live authoritative scene; use /roomie/mutate_scene for live edits")
  server = None
  if not args.marker_only:
    server = start_http_server(node, args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    node.get_logger().info(f"manual room partition UI: {url}")
    if not args.no_browser:
      webbrowser.open(url)
  else:
    node.get_logger().info("running marker-only manual room publisher")
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    if server is not None:
      server.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
