"""Local tools exposed to Gemini for Roomie scene QA."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import math
from typing import Any, Callable

from PIL import Image, ImageDraw

from .config import SceneQaConfig
from .embeddings import ObjectSearchIndex
from .graph_store import GraphStore, _as_vec3, _round


@dataclass
class MediaAttachment:
    data: bytes
    mime_type: str
    summary: dict[str, Any]


@dataclass
class ToolResult:
    response: dict[str, Any]
    media: list[MediaAttachment] = field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters_json_schema: dict[str, Any]
    handler: Callable[..., ToolResult]


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ToolRegistry:
    """Registry for local JSON tools."""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def declarations(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters_json_schema": spec.parameters_json_schema,
            }
            for spec in self._tools.values()
        ]

    def call_tool(self, name: str, args: dict[str, Any] | None) -> ToolResult:
        if name not in self._tools:
            return ToolResult({"error": f"unknown tool: {name}"})
        try:
            return self._tools[name].handler(**(args or {}))
        except Exception as exc:
            return ToolResult({"error": f"{type(exc).__name__}: {exc}"})

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


def create_default_tool_registry(
    graph: GraphStore,
    search_index: ObjectSearchIndex,
    config: SceneQaConfig,
) -> ToolRegistry:
    registry = ToolRegistry()

    def search_objects(
        description: str,
        top_k: int | None = None,
        room_id: int | None = None,
    ) -> ToolResult:
        k = _integer(top_k, config.top_k)
        rid = None if room_id is None else _integer(room_id, room_id)
        results = search_index.search(description, top_k=k, room_id=rid)
        objects = []
        for result in results:
            item = graph.object_to_dict(result.record)
            item["semantic_score"] = round(float(result.score), 4)
            objects.append(item)
        return ToolResult(
            {
                "query": description,
                "room_id": rid,
                "count": len(objects),
                "objects": objects,
            }
        )

    registry.register(
        ToolSpec(
            name="search_objects",
            description=(
                "Find objects by semantic description using the local object embedding index. "
                "Use this first for object identity, color, material, or descriptive queries."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Detailed semantic description of the object(s) to find.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of candidates to return.",
                    },
                    "room_id": {
                        "type": "integer",
                        "description": "Optional room id to restrict the search.",
                    },
                },
                "required": ["description"],
            },
            handler=search_objects,
        )
    )

    def get_object(object_id: int, include_neighbors: bool = True) -> ToolResult:
        obj = graph.get_object(_integer(object_id, -1))
        data = graph.object_to_dict(obj)
        data["raw_without_large_fields"] = graph.sanitized_raw_object(obj.object_id)
        if include_neighbors and obj.center_world is not None:
            nearby = graph.objects_near(obj.center_world, 1.0, top_k=12)
            data["nearby_objects_1m"] = [
                {
                    **graph.object_to_dict(other, include_snapshot=False),
                    "distance_m": round(dist, 4),
                }
                for other, dist in nearby
                if other.object_id != obj.object_id
            ][:8]
        return ToolResult(data)

    registry.register(
        ToolSpec(
            name="get_object",
            description=(
                "Return detailed lightweight metadata for one object id, including room, "
                "geometry, snapshot metadata, and nearby objects."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "object_id": {"type": "integer", "description": "Roomie object id."},
                    "include_neighbors": {
                        "type": "boolean",
                        "description": "Whether to include nearby objects within 1 meter.",
                    },
                },
                "required": ["object_id"],
            },
            handler=get_object,
        )
    )

    def list_rooms() -> ToolResult:
        return ToolResult(
            {
                "count": graph.room_count,
                "rooms": [graph.room_to_dict(room) for room in graph.room_records()],
            }
        )

    registry.register(
        ToolSpec(
            name="list_rooms",
            description=(
                "List all manually annotated rooms/regions with bounds, object counts, "
                "and representative objects."
            ),
            parameters_json_schema={"type": "object", "properties": {}},
            handler=list_rooms,
        )
    )

    def get_objects_in_room(
        room_id: int,
        description: str | None = None,
        top_k: int | None = None,
    ) -> ToolResult:
        rid = _integer(room_id, -1)
        graph.get_room(rid)
        k = _integer(top_k, config.top_k)
        if description:
            matches = search_index.search(description, top_k=k, room_id=rid)
            objects = [
                {
                    **graph.object_to_dict(match.record),
                    "semantic_score": round(float(match.score), 4),
                }
                for match in matches
            ]
        else:
            records = graph.objects_in_room(rid)[: max(0, k)]
            objects = [graph.object_to_dict(record) for record in records]
        return ToolResult(
            {
                "room": graph.room_to_dict(graph.get_room(rid)),
                "description": description,
                "count": len(objects),
                "objects": objects,
            }
        )

    registry.register(
        ToolSpec(
            name="get_objects_in_room",
            description=(
                "Return objects in a specific room. If description is provided, rank room "
                "objects semantically against that description."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "integer", "description": "Room id to inspect."},
                    "description": {
                        "type": "string",
                        "description": "Optional object description to search for inside the room.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of objects to return.",
                    },
                },
                "required": ["room_id"],
            },
            handler=get_objects_in_room,
        )
    )

    def get_objects_near(
        radius_m: float,
        object_id: int | None = None,
        position: list[float] | None = None,
        top_k: int | None = None,
    ) -> ToolResult:
        if object_id is not None:
            center = graph.get_object(_integer(object_id, -1)).center_world
            if center is None:
                return ToolResult({"error": f"object {object_id} has no center_world"})
            query_position = center
        elif position is not None:
            parsed = _as_vec3(position)
            if parsed is None:
                return ToolResult({"error": "position must be [x, y, z]"})
            query_position = parsed
        else:
            return ToolResult({"error": "provide object_id or position"})
        radius = max(0.0, _number(radius_m, 1.0))
        k = _integer(top_k, config.top_k)
        nearby = graph.objects_near(query_position, radius, top_k=k)
        return ToolResult(
            {
                "query_position": _round(query_position),
                "radius_m": round(radius, 4),
                "count": len(nearby),
                "objects": [
                    {
                        **graph.object_to_dict(obj),
                        "distance_m": round(dist, 4),
                    }
                    for obj, dist in nearby
                ],
            }
        )

    registry.register(
        ToolSpec(
            name="get_objects_near",
            description=(
                "Return objects within a metric radius of either an object id or an explicit "
                "3D position. Use for spatial proximity questions."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "radius_m": {"type": "number", "description": "Search radius in meters."},
                    "object_id": {
                        "type": "integer",
                        "description": "Optional object id used as the search center.",
                    },
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Optional explicit [x, y, z] search center.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of objects to return.",
                    },
                },
                "required": ["radius_m"],
            },
            handler=get_objects_near,
        )
    )

    def compare_objects(object_ids: list[int]) -> ToolResult:
        objects = []
        for object_id in object_ids:
            obj = graph.get_object(_integer(object_id, -1))
            objects.append(graph.object_to_dict(obj))
        return ToolResult({"count": len(objects), "objects": objects})

    registry.register(
        ToolSpec(
            name="compare_objects",
            description=(
                "Return compact metadata for several candidate object ids side by side. "
                "Use after search_objects when candidates need disambiguation."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "object_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Object ids to compare.",
                    }
                },
                "required": ["object_ids"],
            },
            handler=compare_objects,
        )
    )

    def inspect_snapshot(
        object_id: int,
        crop: bool = True,
        draw_bbox: bool = True,
    ) -> ToolResult:
        obj = graph.get_object(_integer(object_id, -1))
        image_path = graph.snapshot_image_path(obj)
        if image_path is None:
            return ToolResult({"error": f"object {object_id} has no available snapshot image"})
        if obj.snapshot is None:
            return ToolResult({"error": f"object {object_id} has no snapshot metadata"})
        bbox = obj.snapshot.get("bbox_xyxy")
        with Image.open(image_path) as stream:
            image = stream.convert("RGB")
        original_size = image.size
        crop_box = None
        if isinstance(bbox, list) and len(bbox) >= 4:
            x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
            left = max(0, int(math.floor(min(x0, x1) - config.snapshot_bbox_pad_px)))
            top = max(0, int(math.floor(min(y0, y1) - config.snapshot_bbox_pad_px)))
            right = min(image.width, int(math.ceil(max(x0, x1) + config.snapshot_bbox_pad_px)))
            bottom = min(image.height, int(math.ceil(max(y0, y1) + config.snapshot_bbox_pad_px)))
            if right > left and bottom > top:
                crop_box = (left, top, right, bottom)
            if draw_bbox:
                draw = ImageDraw.Draw(image)
                draw.rectangle([x0, y0, x1, y1], outline=(255, 32, 32), width=4)
        if crop and crop_box is not None:
            image = image.crop(crop_box)
        max_side = max(64, int(config.snapshot_max_side_px))
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        metadata = {
            "object": graph.object_to_dict(obj),
            "image_attached": True,
            "image_path": str(image_path),
            "original_size": list(original_size),
            "returned_size": list(image.size),
            "cropped": bool(crop and crop_box is not None),
            "bbox_drawn": bool(draw_bbox and crop_box is not None),
        }
        media = MediaAttachment(
            data=buffer.getvalue(),
            mime_type="image/jpeg",
            summary={
                "mime_type": "image/jpeg",
                "bytes": buffer.tell(),
                "returned_size": list(image.size),
            },
        )
        return ToolResult(metadata, [media])

    registry.register(
        ToolSpec(
            name="inspect_snapshot",
            description=(
                "Attach the object's snapshot image for visual inspection by Gemini. "
                "Use this when labels/descriptions are ambiguous or the answer depends on visual details."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "object_id": {"type": "integer", "description": "Object id to inspect."},
                    "crop": {
                        "type": "boolean",
                        "description": "Whether to crop around the object's 2D bbox.",
                    },
                    "draw_bbox": {
                        "type": "boolean",
                        "description": "Whether to draw the bbox on the returned image.",
                    },
                },
                "required": ["object_id"],
            },
            handler=inspect_snapshot,
        )
    )

    return registry
