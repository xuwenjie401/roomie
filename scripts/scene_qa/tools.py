"""Local tools exposed to Gemini for Roomie scene QA."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import math
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from PIL import Image, ImageDraw

from .config import SceneQaConfig
from .embeddings import ObjectSearchIndex
from .graph_store import GraphStore, _as_vec3, _round
from .live_query import LiveSceneQueryClient


IMPLICIT_ROOM_ID = "room-0"
IMPLICIT_ROOM_MESSAGE = (
    "No explicit room is defined; all stable scene objects belong to room-0."
)

# These failures cannot be corrected by asking the model to choose another
# argument. Feeding them back as ordinary tool output makes a tool-calling VLM
# repeat the same request until max_iterations is exhausted.
FATAL_SCENE_QUERY_STATUSES = {
    "invalid_session",
    "unknown_session",
    "expired_session",
    "stale_session",
    "session_capacity_exhausted",
    "expired_token",
    "foreign_token",
    "transport_error",
    "service_unavailable",
    "service_timeout",
    "invalid_response",
}


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

    def __init__(
        self,
        *,
        begin_answer: Callable[[], dict[str, Any] | None] | None = None,
        end_answer: Callable[[], None] | None = None,
        answer_metadata: Callable[[], dict[str, Any] | None] | None = None,
    ):
        self._tools: dict[str, ToolSpec] = {}
        self._begin_answer = begin_answer
        self._end_answer = end_answer
        self._answer_metadata = answer_metadata

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
            status = getattr(exc, "status", None)
            if status in FATAL_SCENE_QUERY_STATUSES:
                raise
            response = {"error": f"{type(exc).__name__}: {exc}"}
            if status:
                response["status"] = str(status)
            return ToolResult(response)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def begin_answer(self) -> dict[str, Any] | None:
        return self._begin_answer() if self._begin_answer is not None else None

    def end_answer(self) -> None:
        if self._end_answer is not None:
            self._end_answer()

    def answer_metadata(self) -> dict[str, Any] | None:
        return self._answer_metadata() if self._answer_metadata is not None else None


def _implicit_room(
    object_ids: list[int] | None = None,
    furniture_ids: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "room_id": IMPLICIT_ROOM_ID,
        "name": IMPLICIT_ROOM_ID,
        "label": IMPLICIT_ROOM_ID,
        "implicit": True,
        "attributes": {"implicit": "true"},
        "object_ids": sorted(set(object_ids or [])),
        "furniture_ids": sorted(set(furniture_ids or [])),
    }


def _is_implicit_room(room: dict[str, Any]) -> bool:
    attributes = room.get("attributes")
    implicit_attribute = (
        attributes.get("implicit") if isinstance(attributes, dict) else None
    )
    return bool(room.get("implicit")) or (
        str(room.get("room_id")) == IMPLICIT_ROOM_ID
        and str(implicit_attribute).lower() == "true"
    )


def _list_rooms_response(
    rooms: list[dict[str, Any]],
    *,
    object_ids: list[int] | None = None,
    furniture_ids: list[int] | None = None,
) -> dict[str, Any]:
    normalized = [dict(room) for room in rooms if isinstance(room, dict)]
    if not normalized:
        normalized = [_implicit_room(object_ids, furniture_ids)]
    for room in normalized:
        if _is_implicit_room(room):
            room["implicit"] = True
            room.setdefault("label", room.get("name") or IMPLICIT_ROOM_ID)
    response: dict[str, Any] = {
        "count": len(normalized),
        "rooms": normalized,
    }
    if len(normalized) == 1 and _is_implicit_room(normalized[0]):
        response["room_hierarchy_available"] = False
        response["message"] = IMPLICIT_ROOM_MESSAGE
    return response


def _furniture_child_memberships(
    relations: list[dict[str, Any]], furniture_id: int
) -> list[tuple[int, str]]:
    memberships: set[tuple[int, str]] = set()
    for relation in relations:
        relation_type = str(relation.get("relation_type") or "").strip().lower()
        source = relation.get("source")
        target = relation.get("target")
        if (
            relation_type not in {"in", "on"}
            or not isinstance(source, dict)
            or source.get("type") not in {"object", "furniture"}
            or not isinstance(source.get("id"), int)
            or not isinstance(target, dict)
            or target.get("type") != "furniture"
            or target.get("id") != furniture_id
        ):
            continue
        source_id = int(source["id"])
        if source_id != furniture_id:
            memberships.add((source_id, relation_type))
    return sorted(memberships)


def create_default_tool_registry(
    graph: GraphStore,
    search_index: ObjectSearchIndex,
    config: SceneQaConfig,
) -> ToolRegistry:
    registry = ToolRegistry()

    def room_to_dict(room_id: int) -> dict[str, Any]:
        data = graph.room_to_dict(graph.get_room(room_id))
        data["room_id"] = str(data["room_id"])
        return data

    def resolve_room_id(
        room_id: str | int | None,
    ) -> tuple[int | None, str | None]:
        if room_id is None:
            return None, None
        if graph.room_count == 0 and str(room_id) == IMPLICIT_ROOM_ID:
            return None, IMPLICIT_ROOM_ID
        numeric_id = _integer(room_id, -1)
        graph.get_room(numeric_id)
        return numeric_id, str(numeric_id)

    def search_objects(
        description: str,
        top_k: int | None = None,
        room_id: str | int | None = None,
    ) -> ToolResult:
        k = _integer(top_k, config.top_k)
        rid, response_room_id = resolve_room_id(room_id)
        results = search_index.search(description, top_k=k, room_id=rid)
        objects = []
        for result in results:
            item = graph.object_to_dict(result.record)
            item["semantic_score"] = round(float(result.score), 4)
            objects.append(item)
        return ToolResult(
            {
                "query": description,
                "room_id": response_room_id,
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
                        "type": "string",
                        "description": "Optional room id returned by list_rooms.",
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
        rooms = [room_to_dict(room.room_id) for room in graph.room_records()]
        stable_objects = [
            record.object_id for record in graph.object_records() if record.publishable
        ]
        stable_furniture = [
            record.object_id
            for record in graph.furniture_records()
            if record.object.publishable
        ]
        return ToolResult(
            _list_rooms_response(
                rooms,
                object_ids=stable_objects,
                furniture_ids=stable_furniture,
            )
        )

    registry.register(
        ToolSpec(
            name="list_rooms",
            description=(
                "List all manually annotated rooms/regions with bounds, object counts, "
                "and representative objects. If no explicit room exists, all stable "
                "scene objects are assigned to the queryable room-0."
            ),
            parameters_json_schema={"type": "object", "properties": {}},
            handler=list_rooms,
        )
    )

    def get_objects_in_room(
        room_id: str | int,
        description: str | None = None,
        top_k: int | None = None,
    ) -> ToolResult:
        rid, response_room_id = resolve_room_id(room_id)
        implicit = response_room_id == IMPLICIT_ROOM_ID
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
            records = (
                [record for record in graph.object_records() if record.publishable]
                if implicit
                else graph.objects_in_room(rid if rid is not None else -1)
            )[: max(0, k)]
            objects = [graph.object_to_dict(record) for record in records]
        room = (
            _implicit_room(
                [
                    record.object_id
                    for record in graph.object_records()
                    if record.publishable
                ],
                [
                    record.object_id
                    for record in graph.furniture_records()
                    if record.object.publishable
                ],
            )
            if implicit
            else room_to_dict(rid if rid is not None else -1)
        )
        return ToolResult(
            {
                "room": room,
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
                    "room_id": {
                        "type": "string",
                        "description": "Room id returned by list_rooms.",
                    },
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

    def list_furniture(classification_label: str | None = None) -> ToolResult:
        records = [
            record for record in graph.furniture_records() if record.object.publishable
        ]
        if classification_label:
            records = [
                record
                for record in records
                if record.classification_label == classification_label
            ]
        furniture = [graph.furniture_to_dict(record) for record in records]
        return ToolResult(
            {
                "classification_label": classification_label,
                "count": len(furniture),
                "furniture": furniture,
            }
        )

    registry.register(
        ToolSpec(
            name="list_furniture",
            description=(
                "List canonical furniture roles and their underlying object metadata. "
                "Use for questions about desks, tables, sofas, chairs, shelves, beds, "
                "cabinets, nightstands, or drawers."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "classification_label": {
                        "type": "string",
                        "description": "Optional exact normalized furniture class.",
                    }
                },
            },
            handler=list_furniture,
        )
    )

    def get_objects_related(furniture_id: int) -> ToolResult:
        fid = _integer(furniture_id, -1)
        furniture = graph.get_furniture(fid)
        if not furniture.object.publishable:
            return ToolResult({"error": f"furniture {fid} is not query-visible"})
        relations = graph.relations_for_object(fid, direction="incoming")
        objects = []
        for object_id, relation_type in _furniture_child_memberships(relations, fid):
            obj = graph.get_object(object_id)
            if not obj.publishable:
                continue
            item = graph.object_to_dict(obj)
            item["relation"] = relation_type
            objects.append(item)
        return ToolResult(
            {
                "furniture_id": fid,
                "furniture": graph.furniture_to_dict(furniture),
                "count": len(objects),
                "objects": objects,
            }
        )

    registry.register(
        ToolSpec(
            name="get_objects_related",
            description=(
                "Return all stable objects canonically classified as in or on one "
                "furniture object. Each object includes relation='in' or relation='on'; "
                "this is a containment/support query, not a proximity query."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "furniture_id": {
                        "type": "integer",
                        "description": "Same-id furniture object from list_furniture.",
                    }
                },
                "required": ["furniture_id"],
            },
            handler=get_objects_related,
        )
    )

    def get_relations(
        object_id: int | None = None,
        relation_type: str | None = None,
        direction: str = "either",
    ) -> ToolResult:
        oid = None if object_id is None else _integer(object_id, -1)
        relations = graph.relations_for_object(
            oid, relation_type=relation_type, direction=direction
        )
        return ToolResult(
            {
                "object_id": oid,
                "relation_type": relation_type,
                "direction": direction,
                "count": len(relations),
                "relations": relations,
            }
        )

    registry.register(
        ToolSpec(
            name="get_relations",
            description=(
                "Return canonical scene relations, including object/furniture 'in' and "
                "'on' relations and room containment."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "object_id": {"type": "integer"},
                    "relation_type": {"type": "string"},
                    "direction": {
                        "type": "string",
                        "enum": ["either", "outgoing", "incoming"],
                    },
                },
            },
            handler=get_relations,
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


def create_live_tool_registry(
    client: LiveSceneQueryClient,
    config: SceneQaConfig,
) -> ToolRegistry:
    """Create Gemini tools backed exclusively by the live query gateway."""

    registry = ToolRegistry(
        begin_answer=client.begin_answer,
        end_answer=client.end_answer,
        answer_metadata=client.session_metadata,
    )

    def search_objects(
        description: str,
        top_k: int | None = None,
        room_id: str | int | None = None,
    ) -> ToolResult:
        k = max(1, _integer(top_k, config.top_k))
        params: dict[str, Any] = {
            "query": description,
            "limit": k,
            "include_inactive": True,
            "include_unpublishable": False,
        }
        if room_id is not None:
            params["room_id"] = str(room_id)
        matches = client.call("search_objects", params)
        objects = [_live_search_match(match) for match in matches or []]
        return ToolResult(
            {
                "query": description,
                "room_id": params.get("room_id"),
                "count": len(objects),
                "objects": objects,
            }
        )

    registry.register(
        ToolSpec(
            name="search_objects",
            description=(
                "Find objects by semantic description in the live Roomie scene. "
                "The QA-visible set contains stable publishable objects regardless "
                "of active state. Results are pinned to this answer's scene revision."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Detailed semantic object description.",
                    },
                    "top_k": {"type": "integer"},
                    "room_id": {
                        "type": "string",
                        "description": "Optional canonical room id from list_rooms.",
                    },
                },
                "required": ["description"],
            },
            handler=search_objects,
        )
    )

    def get_object(object_id: int, include_neighbors: bool = True) -> ToolResult:
        data = _live_object(client.call("get_object", {"object_id": _integer(object_id, -1)}))
        if include_neighbors:
            center = data.get("center_world")
            if isinstance(center, list) and len(center) == 3:
                neighbors = client.call(
                    "get_objects_near",
                    {
                        "center_world": center,
                        "radius_m": 1.0,
                        "limit": 12,
                        "include_inactive": True,
                        "include_unpublishable": False,
                    },
                )
                data["nearby_objects_1m"] = [
                    _live_object(item)
                    for item in (neighbors or [])
                    if item.get("object_id") != data.get("object_id")
                ][:8]
        return ToolResult(data)

    registry.register(
        ToolSpec(
            name="get_object",
            description=(
                "Return metadata for one live object and optionally nearby objects "
                "from the same pinned scene revision."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "object_id": {"type": "integer"},
                    "include_neighbors": {"type": "boolean"},
                },
                "required": ["object_id"],
            },
            handler=get_object,
        )
    )

    def list_rooms() -> ToolResult:
        rooms = client.call("rooms", {}) or []
        return ToolResult(_list_rooms_response(rooms))

    registry.register(
        ToolSpec(
            name="list_rooms",
            description=(
                "List canonical rooms and their live object ids. If no explicit room "
                "exists, all stable scene objects are assigned to the queryable room-0."
            ),
            parameters_json_schema={"type": "object", "properties": {}},
            handler=list_rooms,
        )
    )

    def get_objects_in_room(
        room_id: str,
        description: str | None = None,
        top_k: int | None = None,
    ) -> ToolResult:
        room_key = str(room_id)
        k = max(1, _integer(top_k, config.top_k))
        rooms = client.call("rooms", {}) or []
        room = next(
            (item for item in rooms if str(item.get("room_id")) == room_key),
            None,
        )
        if room is None and not rooms and room_key == IMPLICIT_ROOM_ID:
            room = _implicit_room()
        if room is None:
            return ToolResult({"error": f"unknown room_id: {room_key}"})
        if description:
            matches = client.call(
                "search_objects",
                {
                    "query": description,
                    "room_id": room_key,
                    "limit": k,
                    "include_inactive": True,
                    "include_unpublishable": False,
                },
            )
            objects = [_live_search_match(match) for match in matches or []]
        else:
            object_ids = list(room.get("object_ids") or [])[:k]
            values = client.call_many(
                [
                    {"method": "get_object", "params": {"object_id": object_id}}
                    for object_id in object_ids
                ]
            )
            objects = [_live_object(value) for value in values]
        return ToolResult(
            {
                "room": room,
                "description": description,
                "count": len(objects),
                "objects": objects,
            }
        )

    registry.register(
        ToolSpec(
            name="get_objects_in_room",
            description=(
                "Return live objects in a canonical room; optionally rank them by "
                "a semantic description."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "room_id": {
                        "type": "string",
                        "description": "Canonical room id returned by list_rooms.",
                    },
                    "description": {"type": "string"},
                    "top_k": {"type": "integer"},
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
            anchor = _live_object(
                client.call("get_object", {"object_id": _integer(object_id, -1)})
            )
            center = anchor.get("center_world")
        else:
            center = _as_vec3(position) if position is not None else None
        if not isinstance(center, (list, tuple)) or len(center) != 3:
            return ToolResult({"error": "provide an object with geometry or position [x,y,z]"})
        center_list = [float(value) for value in center]
        radius = max(0.0, _number(radius_m, 1.0))
        k = max(1, _integer(top_k, config.top_k))
        values = client.call(
            "get_objects_near",
            {
                "center_world": center_list,
                "radius_m": radius,
                "limit": k,
                "include_inactive": True,
                "include_unpublishable": False,
            },
        )
        objects = []
        for value in values or []:
            item = _live_object(value)
            item_center = item.get("center_world")
            if isinstance(item_center, list) and len(item_center) == 3:
                item["distance_m"] = round(
                    math.dist(center_list, [float(v) for v in item_center]), 4
                )
            objects.append(item)
        return ToolResult(
            {
                "query_position": _round(center_list),
                "radius_m": round(radius, 4),
                "count": len(objects),
                "objects": objects,
            }
        )

    registry.register(
        ToolSpec(
            name="get_objects_near",
            description=(
                "Return live objects within a metric radius of an object or world position."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "radius_m": {"type": "number"},
                    "object_id": {"type": "integer"},
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    "top_k": {"type": "integer"},
                },
                "required": ["radius_m"],
            },
            handler=get_objects_near,
        )
    )

    def list_furniture(classification_label: str | None = None) -> ToolResult:
        values = client.call("furniture", {}) or []
        furniture = []
        for value in values:
            if not isinstance(value, dict):
                continue
            role = (
                value.get("role")
                if isinstance(value.get("role"), dict)
                else {}
            )
            if (
                classification_label
                and role.get("classification_label") != classification_label
            ):
                continue
            object_value = _live_object(value.get("object"))
            if object_value.get("publishable") is False:
                continue
            furniture.append(
                {"role": role, "object": object_value}
            )
        return ToolResult(
            {
                "classification_label": classification_label,
                "count": len(furniture),
                "furniture": furniture,
            }
        )

    registry.register(
        ToolSpec(
            name="list_furniture",
            description=(
                "List live canonical furniture roles and their same-id object metadata."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "classification_label": {
                        "type": "string",
                        "description": "Optional exact normalized furniture class.",
                    }
                },
            },
            handler=list_furniture,
        )
    )

    def get_objects_related(furniture_id: int) -> ToolResult:
        fid = _integer(furniture_id, -1)
        anchor_value, relation_values = client.call_many(
            [
                {"method": "get_object", "params": {"object_id": fid}},
                {
                    "method": "relations",
                    "params": {"object_id": fid, "direction": "incoming"},
                },
            ]
        )
        anchor = _live_object(anchor_value)
        role = anchor.get("furniture_role")
        if not isinstance(role, dict) or role.get("object_id") != fid:
            return ToolResult({"error": f"object {fid} has no furniture role"})
        if anchor.get("publishable") is False:
            return ToolResult({"error": f"furniture {fid} is not query-visible"})
        relations = (
            [value for value in relation_values if isinstance(value, dict)]
            if isinstance(relation_values, list)
            else []
        )
        memberships = _furniture_child_memberships(relations, fid)
        object_ids = sorted({object_id for object_id, _ in memberships})
        values = client.call_many(
            [
                {"method": "get_object", "params": {"object_id": object_id}}
                for object_id in object_ids
            ]
        )
        objects_by_id = {
            int(value["object_id"]): _live_object(value)
            for value in values
            if isinstance(value, dict) and isinstance(value.get("object_id"), int)
        }
        objects = []
        for object_id, relation_type in memberships:
            obj = objects_by_id.get(object_id)
            if obj is None or obj.get("publishable") is False:
                continue
            item = dict(obj)
            item["relation"] = relation_type
            objects.append(item)
        return ToolResult(
            {
                "furniture_id": fid,
                "furniture": {"role": role, "object": anchor},
                "count": len(objects),
                "objects": objects,
            }
        )

    registry.register(
        ToolSpec(
            name="get_objects_related",
            description=(
                "Return all stable objects canonically classified as in or on one "
                "furniture object. Each object includes relation='in' or relation='on'; "
                "this is a containment/support query, not a proximity query."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "furniture_id": {
                        "type": "integer",
                        "description": "Same-id furniture object from list_furniture.",
                    }
                },
                "required": ["furniture_id"],
            },
            handler=get_objects_related,
        )
    )

    def get_relations(
        object_id: int | None = None,
        relation_type: str | None = None,
        direction: str = "either",
    ) -> ToolResult:
        params: dict[str, Any] = {"direction": direction}
        if object_id is not None:
            params["object_id"] = _integer(object_id, -1)
        if relation_type:
            params["relation_type"] = relation_type
        relations = client.call("relations", params) or []
        return ToolResult(
            {
                "object_id": params.get("object_id"),
                "relation_type": relation_type,
                "direction": direction,
                "count": len(relations),
                "relations": relations,
            }
        )

    registry.register(
        ToolSpec(
            name="get_relations",
            description=(
                "Return live canonical object, furniture, and room relations from the "
                "same pinned scene revision."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "object_id": {"type": "integer"},
                    "relation_type": {"type": "string"},
                    "direction": {
                        "type": "string",
                        "enum": ["either", "outgoing", "incoming"],
                    },
                },
            },
            handler=get_relations,
        )
    )

    def compare_objects(object_ids: list[int]) -> ToolResult:
        ids = [_integer(object_id, -1) for object_id in object_ids]
        values = client.call_many(
            [
                {"method": "get_object", "params": {"object_id": object_id}}
                for object_id in ids
            ]
        )
        objects = [_live_object(value) for value in values]
        return ToolResult({"count": len(objects), "objects": objects})

    registry.register(
        ToolSpec(
            name="compare_objects",
            description="Compare metadata for several live object candidates.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "object_ids": {"type": "array", "items": {"type": "integer"}}
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
        inspection = client.call(
            "inspect_snapshot", {"object_id": _integer(object_id, -1)}
        )
        media = _live_snapshot_media(inspection, config, crop, draw_bbox)
        response = dict(inspection or {})
        response["image_attached"] = media is not None
        if media is None:
            response.setdefault("image_error", "no readable snapshot asset")
            return ToolResult(response)
        response.update(media[0])
        return ToolResult(response, [media[1]])

    registry.register(
        ToolSpec(
            name="inspect_snapshot",
            description=(
                "Attach a snapshot asset resolved by the live pinned scene token."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "object_id": {"type": "integer"},
                    "crop": {"type": "boolean"},
                    "draw_bbox": {"type": "boolean"},
                },
                "required": ["object_id"],
            },
            handler=inspect_snapshot,
        )
    )

    return registry


def _live_object(value: Any) -> dict[str, Any]:
    data = dict(value) if isinstance(value, dict) else {}
    geometry = data.get("geometry")
    if isinstance(geometry, dict):
        for key in ("center_world", "size_m", "yaw_rad"):
            if key in geometry:
                data.setdefault(key, geometry[key])
    data.setdefault(
        "description",
        data.get("canonical_description") or data.get("display_description") or "",
    )
    return data


def _live_search_match(value: Any) -> dict[str, Any]:
    match = value if isinstance(value, dict) else {}
    data = _live_object(match.get("object"))
    data["semantic_score"] = round(_number(match.get("score"), 0.0), 4)
    data["search_source"] = match.get("source")
    return data


def _live_snapshot_media(
    inspection: Any,
    config: SceneQaConfig,
    crop: bool,
    draw_bbox: bool,
) -> tuple[dict[str, Any], MediaAttachment] | None:
    if not isinstance(inspection, dict):
        return None
    snapshots = inspection.get("snapshots")
    if not isinstance(snapshots, list):
        return None
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or not snapshot.get("available"):
            continue
        asset = snapshot.get("physical_asset") or snapshot.get("asset")
        if not isinstance(asset, dict):
            continue
        image_path = _asset_path(asset)
        if image_path is None or not image_path.is_file():
            continue
        reference = snapshot.get("reference")
        reference = reference if isinstance(reference, dict) else {}
        bbox = reference.get("bbox_xyxy")
        with Image.open(image_path) as stream:
            image = stream.convert("RGB")
        original_size = image.size
        crop_box = None
        if isinstance(bbox, list) and len(bbox) >= 4:
            x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
            left = max(0, int(math.floor(min(x0, x1) - config.snapshot_bbox_pad_px)))
            top = max(0, int(math.floor(min(y0, y1) - config.snapshot_bbox_pad_px)))
            right = min(
                image.width,
                int(math.ceil(max(x0, x1) + config.snapshot_bbox_pad_px)),
            )
            bottom = min(
                image.height,
                int(math.ceil(max(y0, y1) + config.snapshot_bbox_pad_px)),
            )
            if right > left and bottom > top:
                crop_box = (left, top, right, bottom)
            if draw_bbox:
                ImageDraw.Draw(image).rectangle(
                    [x0, y0, x1, y1], outline=(255, 32, 32), width=4
                )
        if crop and crop_box is not None:
            image = image.crop(crop_box)
        max_side = max(64, int(config.snapshot_max_side_px))
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        summary = {
            "image_path": str(image_path),
            "original_size": list(original_size),
            "returned_size": list(image.size),
            "cropped": bool(crop and crop_box is not None),
            "bbox_drawn": bool(draw_bbox and crop_box is not None),
        }
        attachment = MediaAttachment(
            data=buffer.getvalue(),
            mime_type="image/jpeg",
            summary={
                "mime_type": "image/jpeg",
                "bytes": buffer.tell(),
                "returned_size": list(image.size),
            },
        )
        return summary, attachment
    return None


def _asset_path(asset: dict[str, Any]) -> Path | None:
    source_path = asset.get("source_path")
    if isinstance(source_path, str) and source_path:
        return Path(source_path).expanduser()
    uri = asset.get("uri")
    if not isinstance(uri, str) or not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(uri).expanduser()
    return None
