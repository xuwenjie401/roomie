"""Lightweight indexes over Roomie DSG JSON files."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

from roomie_dsg_viewer import normalize_graph, snapshot_paths


OMITTED_OBJECT_FIELDS = {
    "near_surface_voxels",
    "observation_timestamps_ns",
    "semantic_weights",
    "label_weights",
}


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return None
    values = [_as_float(value[0]), _as_float(value[1]), _as_float(value[2])]
    if any(v is None for v in values):
        return None
    return values[0], values[1], values[2]  # type: ignore[return-value]


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, tuple):
        return [_round(v, digits) for v in value]
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    return value


def _ns_to_s(value: Any) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return number / 1.0e9


def _distance(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay, az = a
    bx, by, bz = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


@dataclass(frozen=True)
class RoomRecord:
    room_id: int
    label: str
    center_world: tuple[float, float, float] | None
    size_m: tuple[float, float, float] | None
    min_xy: tuple[float, float] | None
    max_xy: tuple[float, float] | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class ObjectRecord:
    object_id: int
    label: str
    description: str
    center_world: tuple[float, float, float] | None
    size_m: tuple[float, float, float] | None
    yaw_rad: float | None
    room_ids: tuple[int, ...]
    confidence: float | None
    quality: float | None
    geometry_status: str
    first_seen_s: float | None
    last_seen_s: float | None
    snapshot: dict[str, Any] | None
    active: bool
    publishable: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class FurnitureRecord:
    object_id: int
    classification_label: str
    revision: int
    object: ObjectRecord
    raw: dict[str, Any]


class GraphStore:
    """Normalized, query-friendly view of a Roomie scene graph JSON file."""

    def __init__(self, json_path: Path, raw_root: dict[str, Any]):
        self.json_path = json_path.expanduser().resolve()
        self.raw_root = raw_root
        self.graph = normalize_graph(raw_root)
        self.snapshot_paths = snapshot_paths(self.graph, self.json_path)

        self.rooms_by_id = self._build_rooms(self.graph.get("rooms", []))
        relation_room_map = self._room_map_from_relations(self.graph.get("relations", []))
        self.objects_by_id = self._build_objects(
            self.graph.get("objects", []), relation_room_map
        )
        self.furniture_by_id = self._build_furniture(
            self.graph.get("furniture", [])
        )
        self.relations = [
            relation
            for relation in self.graph.get("relations", [])
            if isinstance(relation, dict)
        ]
        self.room_objects = self._build_room_objects()

    @classmethod
    def load(cls, json_path: str | Path) -> "GraphStore":
        import json

        path = Path(json_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as stream:
            root = json.load(stream)
        if not isinstance(root, dict):
            raise ValueError(f"DSG JSON root must be an object: {path}")
        return cls(path, root)

    @property
    def object_count(self) -> int:
        return len(self.objects_by_id)

    @property
    def room_count(self) -> int:
        return len(self.rooms_by_id)

    @property
    def furniture_count(self) -> int:
        return len(self.furniture_by_id)

    @property
    def snapshot_count(self) -> int:
        return len(self.graph.get("snapshot_images", []))

    def object_records(self) -> list[ObjectRecord]:
        return [self.objects_by_id[k] for k in sorted(self.objects_by_id)]

    def room_records(self) -> list[RoomRecord]:
        return [self.rooms_by_id[k] for k in sorted(self.rooms_by_id)]

    def furniture_records(self) -> list[FurnitureRecord]:
        return [self.furniture_by_id[k] for k in sorted(self.furniture_by_id)]

    def get_object(self, object_id: int) -> ObjectRecord:
        try:
            return self.objects_by_id[int(object_id)]
        except KeyError as exc:
            raise KeyError(f"unknown object_id: {object_id}") from exc

    def get_room(self, room_id: int) -> RoomRecord:
        try:
            return self.rooms_by_id[int(room_id)]
        except KeyError as exc:
            raise KeyError(f"unknown room_id: {room_id}") from exc

    def get_furniture(self, object_id: int) -> FurnitureRecord:
        try:
            return self.furniture_by_id[int(object_id)]
        except KeyError as exc:
            raise KeyError(f"object_id is not furniture: {object_id}") from exc

    def object_text(self, record: ObjectRecord) -> str:
        room_labels = [
            self.rooms_by_id[rid].label
            for rid in record.room_ids
            if rid in self.rooms_by_id and self.rooms_by_id[rid].label
        ]
        parts = [record.label, record.description, " ".join(room_labels)]
        return " ".join(p for p in parts if p).strip()

    def object_to_dict(
        self,
        record: ObjectRecord,
        *,
        include_snapshot: bool = True,
        include_rooms: bool = True,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "object_id": record.object_id,
            "label": record.label,
            "description": record.description,
            "center_world": _round(record.center_world),
            "size_m": _round(record.size_m),
            "yaw_rad": _round(record.yaw_rad),
            "room_ids": list(record.room_ids),
            "confidence": _round(record.confidence),
            "object_quality_score": _round(record.quality),
            "geometry_status": record.geometry_status,
            "first_seen_s": _round(record.first_seen_s),
            "last_seen_s": _round(record.last_seen_s),
            "active": record.active,
            "publishable": record.publishable,
        }
        if include_rooms:
            data["rooms"] = [
                {
                    "room_id": rid,
                    "label": self.rooms_by_id[rid].label,
                }
                for rid in record.room_ids
                if rid in self.rooms_by_id
            ]
        if include_snapshot:
            data["snapshot"] = self.snapshot_metadata(record)
        role = self.furniture_by_id.get(record.object_id)
        data["furniture_role"] = (
            {
                "object_id": role.object_id,
                "classification_label": role.classification_label,
                "revision": role.revision,
            }
            if role is not None
            else None
        )
        return data

    def furniture_to_dict(self, record: FurnitureRecord) -> dict[str, Any]:
        return {
            "role": {
                "object_id": record.object_id,
                "classification_label": record.classification_label,
                "revision": record.revision,
            },
            "object": self.object_to_dict(record.object, include_snapshot=False),
        }

    def relations_for_object(
        self,
        object_id: int | None = None,
        relation_type: str | None = None,
        direction: str = "either",
    ) -> list[dict[str, Any]]:
        if direction not in {"either", "outgoing", "incoming"}:
            raise ValueError("direction must be either, outgoing, or incoming")
        if object_id is not None:
            self.get_object(object_id)

        def object_endpoint(value: Any) -> int | None:
            if not isinstance(value, dict) or value.get("type") not in {
                "object",
                "furniture",
            }:
                return None
            endpoint_id = value.get("id")
            return endpoint_id if isinstance(endpoint_id, int) else None

        result = []
        for relation in self.relations:
            if relation_type is not None and str(
                relation.get("relation_type") or ""
            ) != relation_type:
                continue
            source_id = object_endpoint(relation.get("source"))
            target_id = object_endpoint(relation.get("target"))
            if object_id is not None:
                matches = (
                    (direction in {"either", "outgoing"} and source_id == object_id)
                    or (direction in {"either", "incoming"} and target_id == object_id)
                )
                if not matches:
                    continue
            result.append(_round(dict(relation)))
        return result

    def room_to_dict(self, room: RoomRecord, *, include_objects: bool = False) -> dict[str, Any]:
        object_ids = sorted(self.room_objects.get(room.room_id, set()))
        data: dict[str, Any] = {
            "room_id": room.room_id,
            "label": room.label,
            "center_world": _round(room.center_world),
            "size_m": _round(room.size_m),
            "min_xy": _round(room.min_xy),
            "max_xy": _round(room.max_xy),
            "object_count": len(object_ids),
        }
        if include_objects:
            data["objects"] = [
                self.object_to_dict(self.objects_by_id[obj_id], include_snapshot=False)
                for obj_id in object_ids
            ]
        else:
            representatives = sorted(
                (self.objects_by_id[obj_id] for obj_id in object_ids),
                key=lambda obj: (obj.quality is not None, obj.quality or 0.0),
                reverse=True,
            )[:8]
            data["representative_objects"] = [
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "description": obj.description,
                }
                for obj in representatives
            ]
        return data

    def snapshot_metadata(self, record: ObjectRecord) -> dict[str, Any] | None:
        if not isinstance(record.snapshot, dict):
            return None
        image_index = record.snapshot.get("image_index")
        image_path = self.snapshot_paths.get(image_index) if isinstance(image_index, int) else None
        bbox = record.snapshot.get("bbox_xyxy")
        return {
            "image_index": image_index,
            "camera_id": record.snapshot.get("camera_id"),
            "bbox_xyxy": _round(bbox),
            "quality": _round(_as_float(record.snapshot.get("quality"))),
            "time_s": _round(_ns_to_s(record.snapshot.get("time_ns"))),
            "image_available": image_path is not None,
        }

    def snapshot_image_path(self, record: ObjectRecord) -> Path | None:
        if not isinstance(record.snapshot, dict):
            return None
        image_index = record.snapshot.get("image_index")
        return self.snapshot_paths.get(image_index) if isinstance(image_index, int) else None

    def objects_in_room(
        self,
        room_id: int,
        *,
        include_unpublishable: bool = False,
    ) -> list[ObjectRecord]:
        object_ids = self.room_objects.get(int(room_id), set())
        return [
            self.objects_by_id[obj_id]
            for obj_id in sorted(object_ids)
            if include_unpublishable or self.objects_by_id[obj_id].publishable
        ]

    def objects_near(
        self,
        position: tuple[float, float, float],
        radius_m: float,
        *,
        top_k: int,
        include_unpublishable: bool = False,
    ) -> list[tuple[ObjectRecord, float]]:
        results: list[tuple[ObjectRecord, float]] = []
        for obj in self.objects_by_id.values():
            if not include_unpublishable and not obj.publishable:
                continue
            if obj.center_world is None:
                continue
            dist = _distance(obj.center_world, position)
            if dist <= radius_m:
                results.append((obj, dist))
        results.sort(key=lambda item: item[1])
        return results[: max(0, top_k)]

    def sanitized_raw_object(self, object_id: int) -> dict[str, Any]:
        record = self.get_object(object_id)
        return {
            key: _round(value)
            for key, value in record.raw.items()
            if key not in OMITTED_OBJECT_FIELDS
        }

    def _build_rooms(self, rooms: Any) -> dict[int, RoomRecord]:
        records: dict[int, RoomRecord] = {}
        if not isinstance(rooms, list):
            return records
        for item in rooms:
            if not isinstance(item, dict):
                continue
            room_id = item.get("room_id")
            if not isinstance(room_id, int):
                continue
            min_xy = item.get("min_xy")
            max_xy = item.get("max_xy")
            records[room_id] = RoomRecord(
                room_id=room_id,
                label=str(item.get("label") or f"room {room_id}"),
                center_world=_as_vec3(item.get("center_world")),
                size_m=_as_vec3(item.get("size_m")),
                min_xy=(
                    float(min_xy[0]),
                    float(min_xy[1]),
                )
                if isinstance(min_xy, list) and len(min_xy) >= 2
                else None,
                max_xy=(
                    float(max_xy[0]),
                    float(max_xy[1]),
                )
                if isinstance(max_xy, list) and len(max_xy) >= 2
                else None,
                raw=item,
            )
        return records

    def _room_map_from_relations(self, relations: Any) -> dict[int, set[int]]:
        mapping: dict[int, set[int]] = {}
        if not isinstance(relations, list):
            return mapping
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            if rel.get("relation_type") != "room_contains_object":
                continue
            source = rel.get("source")
            target = rel.get("target")
            if not isinstance(source, dict) or not isinstance(target, dict):
                continue
            if source.get("type") != "room" or target.get("type") != "object":
                continue
            room_id = source.get("id")
            object_id = target.get("id")
            if isinstance(room_id, int) and isinstance(object_id, int):
                mapping.setdefault(object_id, set()).add(room_id)
        return mapping

    def _build_objects(
        self,
        objects: Any,
        relation_room_map: dict[int, set[int]],
    ) -> dict[int, ObjectRecord]:
        records: dict[int, ObjectRecord] = {}
        if not isinstance(objects, list):
            return records
        for item in objects:
            if not isinstance(item, dict):
                continue
            object_id = item.get("object_id")
            if not isinstance(object_id, int):
                continue
            room_ids = set()
            for room_id in item.get("parent_room_ids") or []:
                if isinstance(room_id, int):
                    room_ids.add(room_id)
            room_ids.update(relation_room_map.get(object_id, set()))
            snapshot = item.get("snapshot") if isinstance(item.get("snapshot"), dict) else None
            records[object_id] = ObjectRecord(
                object_id=object_id,
                label=str(item.get("label") or f"object {object_id}"),
                description=str(item.get("description") or ""),
                center_world=_as_vec3(item.get("center_world")),
                size_m=_as_vec3(item.get("size_m")),
                yaw_rad=_as_float(item.get("yaw_rad")),
                room_ids=tuple(sorted(room_ids)),
                confidence=_as_float(item.get("confidence")),
                quality=_as_float(item.get("object_quality_score")),
                geometry_status=str(item.get("geometry_status") or ""),
                first_seen_s=_ns_to_s(item.get("first_seen_ns")),
                last_seen_s=_ns_to_s(item.get("last_seen_ns")),
                snapshot=snapshot,
                active=bool(item.get("active", True)),
                publishable=bool(item.get("publishable", True)),
                raw=item,
            )
        return records

    def _build_furniture(self, furniture: Any) -> dict[int, FurnitureRecord]:
        records: dict[int, FurnitureRecord] = {}
        if not isinstance(furniture, list):
            return records
        for item in furniture:
            if not isinstance(item, dict):
                continue
            object_id = item.get("object_id")
            label = item.get("classification_label")
            if (
                not isinstance(object_id, int)
                or object_id not in self.objects_by_id
                or not isinstance(label, str)
                or not label
            ):
                continue
            revision = item.get("revision", 0)
            records[object_id] = FurnitureRecord(
                object_id=object_id,
                classification_label=label,
                revision=revision if isinstance(revision, int) else 0,
                object=self.objects_by_id[object_id],
                raw=item,
            )
        return records

    def _build_room_objects(self) -> dict[int, set[int]]:
        room_objects: dict[int, set[int]] = {room_id: set() for room_id in self.rooms_by_id}
        for obj in self.objects_by_id.values():
            for room_id in obj.room_ids:
                room_objects.setdefault(room_id, set()).add(obj.object_id)
        return room_objects
