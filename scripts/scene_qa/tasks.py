"""Task-specific Roomie tool registries and action-shaped result validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
import unicodedata
from typing import Any, Callable

from .camera_view import CameraViewSample, project_yaw_obb_to_camera
from .config import SceneQaConfig
from .detection_view import DETECTION_CAMERA_IDS, Latest2dDetectionFrame
from .gemini_agent import QaResponse
from .object_references import ObjectReferenceCatalogLoad
from .tools import MediaAttachment, ToolRegistry, ToolResult, ToolSpec


TASK_SCENE_QA = "scene_qa"
TASK_NAVIGATION = "navigation"
TASK_FIND_OBJECT_IN_VIEW = "find_object_in_view"
TASK_NAMES = (TASK_SCENE_QA, TASK_NAVIGATION, TASK_FIND_OBJECT_IN_VIEW)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _stamp_ns(header: Any) -> int | None:
    if not isinstance(header, dict) or not isinstance(header.get("stamp"), dict):
        return None
    stamp = header["stamp"]
    try:
        sec = int(stamp.get("sec", 0))
        nanosec = int(stamp.get("nanosec", 0))
    except (TypeError, ValueError):
        return None
    return sec * 1_000_000_000 + nanosec


def _normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _tokens(value: Any) -> set[str]:
    return set(_normalize_text(value).split())


def _object_id(value: Any) -> int | None:
    return _integer(value.get("object_id")) if isinstance(value, dict) else None


def _object_text(value: dict[str, Any]) -> str:
    return " ".join(
        str(value.get(key) or "") for key in ("name", "label", "description")
    ).strip()


def _match_strength(query: str, value: dict[str, Any]) -> tuple[int, str]:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return 0, "semantic"
    fields = [
        ("name", _normalize_text(value.get("name"))),
        ("label", _normalize_text(value.get("label"))),
    ]
    for key, text in fields:
        if text and text == normalized_query:
            return 4, f"exact_{key}"
    matching_fields = [
        (key, text) for key, text in fields if text and text in normalized_query
    ]
    if matching_fields:
        key, _ = max(matching_fields, key=lambda item: len(item[1]))
        return 3, f"{key}_in_query"
    description = _normalize_text(value.get("description"))
    if len(normalized_query) >= 4 and normalized_query in description:
        return 2, "query_in_description"
    return 1, "semantic"


def _furniture_item(value: Any) -> tuple[int, dict[str, Any]] | None:
    if not isinstance(value, dict):
        return None
    role = value.get("role") if isinstance(value.get("role"), dict) else {}
    obj = value.get("object") if isinstance(value.get("object"), dict) else {}
    object_id = _integer(role.get("object_id"))
    if object_id is None:
        object_id = _object_id(obj)
    if object_id is None:
        return None
    result = dict(obj)
    result["object_id"] = object_id
    result["furniture_role"] = dict(role)
    result["classification_label"] = role.get("classification_label")
    return object_id, result


def _relation_endpoint(value: Any) -> tuple[str, int] | None:
    if not isinstance(value, dict):
        return None
    entity_id = _integer(value.get("id"))
    entity_type = str(value.get("type") or "")
    if entity_id is None or entity_type not in {"object", "furniture", "room"}:
        return None
    return entity_type, entity_id


def _reference_prompt(
    load: ObjectReferenceCatalogLoad | None,
    evidence_tool: str,
) -> str:
    if load is None:
        return ""
    if load.catalog is None:
        return (
            "The curated object-reference catalog is unavailable in this run. "
            "Do not claim a reference-image comparison. "
            f"Runtime status: {load.status}"
        )
    lines = [
        "A curated local object-reference catalog is available. Known exact "
        "reference_id values are:"
    ]
    for reference in load.catalog.references():
        aliases = ", ".join(reference.aliases)
        suffix = f"; aliases: {aliases}" if aliases else ""
        lines.append(
            f"- reference_id={reference.reference_id!r}; name={reference.name!r}{suffix}"
        )
    lines.append(
        f"Pass an exact reference_id to {evidence_tool} only after the primary "
        "tool has returned concrete scene object ids. The tool supplies labeled "
        "historical scene snapshots and reference views; make the visual judgment "
        "yourself and report inconclusive evidence honestly."
    )
    return "\n".join(lines)


@dataclass
class TaskEvidence:
    task: str
    turn_closed: bool = True
    primary_called: bool = False
    inspection_called: bool = False
    detection_inspection_called: bool = False
    primary_response: dict[str, Any] = field(default_factory=dict)
    objects: dict[int, dict[str, Any]] = field(default_factory=dict)
    furniture: dict[int, dict[str, Any]] = field(default_factory=dict)
    object_to_furniture: dict[int, list[int]] = field(default_factory=dict)
    projections: dict[int, dict[str, Any]] = field(default_factory=dict)
    ranked_furniture_ids: list[int] = field(default_factory=list)
    explicit_furniture_ids: list[int] = field(default_factory=list)
    camera_sample: CameraViewSample | None = None

    def begin_primary(self) -> None:
        self.turn_closed = False
        self.primary_called = True
        self.inspection_called = False
        self.detection_inspection_called = False
        self.primary_response = {}
        self.objects.clear()
        self.furniture.clear()
        self.object_to_furniture.clear()
        self.projections.clear()
        self.ranked_furniture_ids.clear()
        self.explicit_furniture_ids.clear()
        self.camera_sample = None


def _task_registry(
    base: ToolRegistry,
    evidence: TaskEvidence,
    *,
    prompt_suffix: str,
) -> ToolRegistry:
    def end_answer() -> None:
        base.end_answer()
        evidence.turn_closed = True

    return ToolRegistry(
        begin_answer=base.begin_answer,
        end_answer=end_answer,
        answer_metadata=base.answer_metadata,
        system_prompt_suffix=prompt_suffix,
    )


def _candidate_ids(object_ids: Any, evidence: TaskEvidence) -> list[int]:
    if not isinstance(object_ids, list) or not object_ids:
        raise ValueError("object_ids must be a non-empty array")
    result: list[int] = []
    for raw_id in object_ids:
        object_id = _integer(raw_id)
        if object_id is None or object_id not in evidence.objects:
            raise ValueError(f"object_id {raw_id!r} was not returned by the primary tool")
        if object_id not in result:
            result.append(object_id)
    if len(result) > 4:
        raise ValueError("at most 4 unique object_ids may be inspected")
    return result


def _labeled_media(
    media: MediaAttachment, *, object_id: int, role: str = "scene_object"
) -> MediaAttachment:
    return MediaAttachment(
        data=media.data,
        mime_type=media.mime_type,
        summary={
            **media.summary,
            "role": role,
            "object_id": object_id,
            "label": f"SCENE OBJECT object_id={object_id}; historical object snapshot.",
        },
    )


def _inspect_task_candidates(
    base: ToolRegistry,
    evidence: TaskEvidence,
    object_ids: Any,
    reference_id: str | None,
) -> ToolResult:
    if not evidence.primary_called:
        raise ValueError("call the primary task tool before inspecting candidates")
    if evidence.inspection_called:
        raise ValueError("candidate evidence may be inspected only once per task")
    ids = _candidate_ids(object_ids, evidence)
    evidence.inspection_called = True
    if reference_id:
        if "compare_scene_objects_to_reference" not in base.list_tools():
            raise ValueError("the curated object-reference catalog is unavailable")
        result = base.call_tool(
            "compare_scene_objects_to_reference",
            {"reference_id": str(reference_id), "object_ids": ids},
        )
        response = dict(result.response)
        response["inspected_object_ids"] = ids
        response["evidence_scope"] = "historical_snapshots_and_curated_reference"
        return ToolResult(response, result.media)

    objects = []
    media: list[MediaAttachment] = []
    for object_id in ids:
        snapshot = base.call_tool(
            "inspect_snapshot",
            {"object_id": object_id, "crop": True, "draw_bbox": False},
        )
        item = dict(evidence.objects[object_id])
        item["historical_snapshot"] = dict(snapshot.response)
        item["snapshot_attached"] = bool(snapshot.media)
        objects.append(item)
        media.extend(
            _labeled_media(attachment, object_id=object_id)
            for attachment in snapshot.media
        )
    return ToolResult(
        {
            "inspected_object_ids": ids,
            "objects": objects,
            "evidence_scope": "historical_scene_snapshots",
            "message": (
                "These images are historical scene-graph evidence, not the current RGB frame."
            ),
        },
        media,
    )


def create_navigation_task_registry(
    base: ToolRegistry,
    config: SceneQaConfig,
    evidence: TaskEvidence,
    reference_catalog_load: ObjectReferenceCatalogLoad | None = None,
) -> ToolRegistry:
    """Expose only the two bounded tools needed by the navigation task."""

    del config
    registry = _task_registry(
        base,
        evidence,
        prompt_suffix=_reference_prompt(
            reference_catalog_load, "inspect_navigation_candidates"
        ),
    )

    def locate_navigation_target(
        target_description: str,
        requested_furniture_description: str | None = None,
        reference_id: str | None = None,
    ) -> ToolResult:
        if evidence.primary_called and not evidence.turn_closed:
            raise ValueError("locate_navigation_target may be called only once per task")
        if not str(target_description).strip():
            raise ValueError("target_description must be non-empty")
        evidence.begin_primary()

        search_description = str(target_description).strip()
        if reference_id and reference_catalog_load and reference_catalog_load.catalog:
            reference = reference_catalog_load.catalog.get(str(reference_id))
            search_description = " ".join(
                [search_description, reference.name, *reference.aliases]
            )
        search = base.call_tool(
            "search_objects", {"description": search_description, "top_k": 12}
        )
        furniture_result = base.call_tool("list_furniture", {})
        relations_result = base.call_tool("get_relations", {})
        furniture_values = furniture_result.response.get("furniture")
        if not isinstance(furniture_values, list):
            furniture_values = []
        for value in furniture_values:
            parsed = _furniture_item(value)
            if parsed is not None:
                evidence.furniture[parsed[0]] = parsed[1]

        relations = relations_result.response.get("relations")
        if not isinstance(relations, list):
            relations = []
        relation_map: dict[int, list[tuple[int, str, float]]] = {}
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            relation_type = str(relation.get("relation_type") or "").lower()
            source = _relation_endpoint(relation.get("source"))
            target = _relation_endpoint(relation.get("target"))
            if (
                relation_type not in {"in", "on"}
                or source is None
                or target is None
                or source[0] not in {"object", "furniture"}
                or target[0] != "furniture"
                or target[1] not in evidence.furniture
            ):
                continue
            relation_map.setdefault(source[1], []).append(
                (target[1], relation_type, _number(relation.get("confidence")))
            )

        search_objects = search.response.get("objects")
        if not isinstance(search_objects, list):
            search_objects = []
        candidates = []
        for value in search_objects:
            if not isinstance(value, dict):
                continue
            object_id = _object_id(value)
            if object_id is None:
                continue
            item = dict(value)
            evidence.objects[object_id] = item
            memberships = sorted(
                relation_map.get(object_id, []),
                key=lambda entry: (-entry[2], entry[0]),
            )
            role = item.get("furniture_role")
            if (
                isinstance(role, dict)
                and _integer(role.get("object_id")) in evidence.furniture
            ):
                own_id = int(role["object_id"])
                memberships.insert(0, (own_id, "self", 1.0))
            furniture_ids = []
            relations_for_candidate = []
            for furniture_id, relation_type, confidence in memberships:
                if furniture_id not in furniture_ids:
                    furniture_ids.append(furniture_id)
                    relations_for_candidate.append(
                        {
                            "furniture_id": furniture_id,
                            "relation": relation_type,
                            "confidence": round(confidence, 4),
                        }
                    )
            evidence.object_to_furniture[object_id] = furniture_ids
            strength, match_type = _match_strength(str(target_description), item)
            candidates.append(
                {
                    **item,
                    "match_type": match_type,
                    "match_strength": strength,
                    "furniture_relations": relations_for_candidate,
                }
            )

        explicit_matches: list[dict[str, Any]] = []
        if requested_furniture_description:
            query = str(requested_furniture_description)
            requested_tokens = _tokens(query)
            search_ids: set[int] = set()
            furniture_search = base.call_tool(
                "search_objects", {"description": query, "top_k": 8}
            )
            for item in furniture_search.response.get("objects") or []:
                object_id = _object_id(item)
                if object_id is not None:
                    search_ids.add(object_id)
            ranked_explicit = []
            for furniture_id, item in evidence.furniture.items():
                class_label = item.get("classification_label")
                normalized_fields = {
                    _normalize_text(class_label),
                    _normalize_text(item.get("label")),
                    _normalize_text(item.get("name")),
                }
                exact = _normalize_text(query) in normalized_fields
                overlap = len(requested_tokens & _tokens(_object_text(item)))
                score = (4.0 if exact else 0.0) + float(overlap)
                if score == 0.0 and furniture_id in search_ids:
                    score += 1.0
                if score > 0.0:
                    ranked_explicit.append((score, furniture_id, item))
            ranked_explicit.sort(key=lambda entry: (-entry[0], entry[1]))
            if ranked_explicit:
                best_score = ranked_explicit[0][0]
                explicit_matches = [
                    {**item, "furniture_match_score": score}
                    for score, _fid, item in ranked_explicit
                    if score == best_score
                ]
                evidence.explicit_furniture_ids = [
                    int(item["object_id"]) for item in explicit_matches
                ]

        ranked_furniture: list[int] = []
        for furniture_id in evidence.explicit_furniture_ids:
            if furniture_id not in ranked_furniture:
                ranked_furniture.append(furniture_id)
        for candidate in candidates:
            object_id = int(candidate["object_id"])
            for furniture_id in evidence.object_to_furniture.get(object_id, []):
                if furniture_id not in ranked_furniture:
                    ranked_furniture.append(furniture_id)
        for furniture_id in sorted(evidence.furniture):
            if furniture_id not in ranked_furniture:
                ranked_furniture.append(furniture_id)
        evidence.ranked_furniture_ids = ranked_furniture
        furniture_candidates = [
            evidence.furniture[furniture_id]
            for furniture_id in ranked_furniture[:12]
        ]
        response = {
            "target_description": str(target_description),
            "requested_furniture_description": requested_furniture_description,
            "reference_id": reference_id,
            "object_candidates": candidates,
            "explicit_furniture_matches": explicit_matches,
            "explicit_furniture_unique": len(explicit_matches) == 1,
            "furniture_candidates": furniture_candidates,
            "instructions": (
                "Choose only ids in this response. Objects without a furniture relation "
                "cannot directly provide a navigation target."
            ),
        }
        evidence.primary_response = response
        return ToolResult(response)

    registry.register(
        ToolSpec(
            name="locate_navigation_target",
            description=(
                "Primary one-shot navigation lookup. It searches the stable scene, "
                "joins target candidates to canonical in/on furniture relations, and "
                "returns explicit requested-furniture matches plus exploration options."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "target_description": {
                        "type": "string",
                        "description": "English semantic description of the object to retrieve.",
                    },
                    "requested_furniture_description": {
                        "type": "string",
                        "description": "English furniture/place phrase explicitly stated by the user.",
                    },
                    "reference_id": {
                        "type": "string",
                        "description": "Optional exact curated reference_id from the runtime prompt.",
                    },
                },
                "required": ["target_description"],
            },
            handler=locate_navigation_target,
        )
    )

    def inspect_navigation_candidates(
        object_ids: list[int], reference_id: str | None = None
    ) -> ToolResult:
        return _inspect_task_candidates(base, evidence, object_ids, reference_id)

    registry.register(
        ToolSpec(
            name="inspect_navigation_candidates",
            description=(
                "Optional single evidence check for up to four object ids already "
                "returned by locate_navigation_target. It attaches historical scene "
                "snapshots and, when requested, curated reference views."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "object_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 4,
                    },
                    "reference_id": {"type": "string"},
                },
                "required": ["object_ids"],
            },
            handler=inspect_navigation_candidates,
        )
    )
    return registry


def create_find_object_in_view_task_registry(
    base: ToolRegistry,
    config: SceneQaConfig,
    evidence: TaskEvidence,
    camera_sampler: Any,
    reference_catalog_load: ObjectReferenceCatalogLoad | None = None,
    *,
    detection_sampler: Any | None = None,
) -> ToolRegistry:
    """Expose bounded frustum, latest-2D, and historical evidence tools."""

    if (
        config.in_view_min_depth_m <= 0.0
        or config.in_view_max_depth_m <= config.in_view_min_depth_m
    ):
        raise ValueError(
            "in-view depth range must satisfy "
            "0 < in_view_min_depth_m < in_view_max_depth_m"
        )

    registry = _task_registry(
        base,
        evidence,
        prompt_suffix=_reference_prompt(
            reference_catalog_load, "inspect_in_view_candidates"
        ),
    )

    def find_objects_in_view(
        target_description: str,
        reference_id: str | None = None,
    ) -> ToolResult:
        if evidence.primary_called and not evidence.turn_closed:
            raise ValueError("find_objects_in_view may be called only once per task")
        if not str(target_description).strip():
            raise ValueError("target_description must be non-empty")
        evidence.begin_primary()
        sample_value = camera_sampler.sample()
        sample = (
            sample_value
            if isinstance(sample_value, CameraViewSample)
            else CameraViewSample.from_mapping(sample_value)
        )
        evidence.camera_sample = sample
        search_description = str(target_description).strip()
        if reference_id and reference_catalog_load and reference_catalog_load.catalog:
            reference = reference_catalog_load.catalog.get(str(reference_id))
            search_description = " ".join(
                [search_description, reference.name, *reference.aliases]
            )
        search = base.call_tool(
            "search_objects", {"description": search_description, "top_k": 100}
        )
        search_objects = search.response.get("objects")
        if not isinstance(search_objects, list):
            search_objects = []
        candidates = []
        for value in search_objects:
            if not isinstance(value, dict):
                continue
            object_id = _object_id(value)
            if object_id is None:
                continue
            center = value.get("center_world")
            size = value.get("size_m")
            if center is None or size is None:
                continue
            try:
                projection = project_yaw_obb_to_camera(
                    center,
                    size,
                    value.get("yaw_rad", 0.0),
                    sample,
                    min_depth_m=config.in_view_min_depth_m,
                    max_depth_m=config.in_view_max_depth_m,
                )
            except ValueError:
                continue
            if not projection.in_view:
                continue
            item = dict(value)
            strength, match_type = _match_strength(str(target_description), item)
            projection_value = projection.to_dict(sample.header)
            candidate = {
                **item,
                "match_type": match_type,
                "match_strength": strength,
                **projection_value,
            }
            evidence.objects[object_id] = item
            evidence.projections[object_id] = projection_value
            candidates.append(candidate)
            if len(candidates) >= 12:
                break

        strongest = max(
            (int(candidate["match_strength"]) for candidate in candidates),
            default=0,
        )
        strongest_ids = [
            int(candidate["object_id"])
            for candidate in candidates
            if int(candidate["match_strength"]) == strongest and strongest >= 2
        ]
        strong_match_id = strongest_ids[0] if len(strongest_ids) == 1 else None
        response = {
            "target_description": str(target_description),
            "reference_id": reference_id,
            "camera": sample.metadata(),
            "visibility_definition": "any_3d_bbox_corner_in_sampled_head_camera_frustum",
            "occlusion_checked": False,
            "current_rgb_inspected": False,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "strong_match_object_id": strong_match_id,
            "strong_match_is_unique": strong_match_id is not None,
            "instructions": (
                "FOUND may use the unique strong match directly. Otherwise inspect the "
                "latest retained three-camera 2D detections once. If identity remains "
                "uncertain and candidates exist, historical/reference evidence for at "
                "most four candidates may also be inspected once."
            ),
        }
        evidence.primary_response = response
        return ToolResult(response)

    registry.register(
        ToolSpec(
            name="find_objects_in_view",
            description=(
                "Primary one-shot in-view lookup. It semantically searches stable "
                "scene objects, samples the latest retained head-camera geometry once, "
                "applies the any-3D-bbox-corner frustum rule, and returns projected ROIs. "
                "Camera metadata says whether the sample is fresh or cached during a "
                "sensor/bag pause."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "target_description": {
                        "type": "string",
                        "description": "English semantic description of the target object.",
                    },
                    "reference_id": {
                        "type": "string",
                        "description": "Optional exact curated reference_id from the runtime prompt.",
                    },
                },
                "required": ["target_description"],
            },
            handler=find_objects_in_view,
        )
    )

    def inspect_latest_2d_detections() -> ToolResult:
        if not evidence.primary_called:
            raise ValueError("call find_objects_in_view before inspecting 2D detections")
        if evidence.detection_inspection_called:
            raise ValueError("latest 2D detections may be inspected only once per task")
        if detection_sampler is None:
            raise ValueError("latest 2D detection sampling is unavailable")
        evidence.detection_inspection_called = True
        sampled = detection_sampler.sample()
        if not isinstance(sampled, list):
            raise ValueError("latest 2D detection sampler must return an array")
        frames = [
            value
            if isinstance(value, Latest2dDetectionFrame)
            else Latest2dDetectionFrame.from_mapping(value)
            for value in sampled
        ]
        by_camera = {frame.camera_id: frame for frame in frames}
        if (
            len(frames) != len(DETECTION_CAMERA_IDS)
            or set(by_camera) != set(DETECTION_CAMERA_IDS)
        ):
            raise ValueError("latest 2D detection sampler returned an unexpected camera set")

        head_sample_ns = (
            _stamp_ns(evidence.camera_sample.header)
            if evidence.camera_sample is not None
            else None
        )
        cameras: list[dict[str, Any]] = []
        media: list[MediaAttachment] = []
        for camera_id in DETECTION_CAMERA_IDS:
            frame = by_camera[camera_id]
            metadata = frame.metadata()
            result_ns = _stamp_ns(metadata.get("header"))
            metadata["head_geometry_sample_delta_ms"] = (
                round((result_ns - head_sample_ns) / 1.0e6, 3)
                if result_ns is not None and head_sample_ns is not None
                else None
            )
            metadata["image_attached"] = bool(
                frame.image_available and frame.image_data
            )
            cameras.append(metadata)
            if frame.image_available and frame.image_data:
                header = metadata.get("image_header")
                stamp = header.get("stamp") if isinstance(header, dict) else None
                media.append(
                    MediaAttachment(
                        data=frame.image_data,
                        mime_type=frame.image_mime_type,
                        summary={
                            "role": "latest_2d_detection_result",
                            "camera_id": camera_id,
                            "header": header,
                            "label": (
                                "LATEST 2D DETECTION RESULT "
                                f"camera_id={camera_id}; stamp={stamp}; "
                                "boxes, labels, and scores are detector output."
                            ),
                        },
                    )
                )
        available_count = sum(bool(camera["available"]) for camera in cameras)
        return ToolResult(
            {
                "evidence_scope": "latest_retained_2d_detection_results",
                "requested_camera_ids": list(DETECTION_CAMERA_IDS),
                "available_camera_count": available_count,
                "cameras": cameras,
                "message": (
                    "Each entry is the last retained detector result for that camera, "
                    "not a newly synchronized three-camera capture. Use headers and "
                    "head_geometry_sample_delta_ms when judging temporal alignment."
                ),
            },
            media,
        )

    registry.register(
        ToolSpec(
            name="inspect_latest_2d_detections",
            description=(
                "Optional one-shot VLM inspection of the latest retained 2D detector "
                "results for head_color, hand_left_color, and hand_right_color. It "
                "returns exact label/score/bbox data and attaches each available "
                "detector-annotated RGB image. Call only after find_objects_in_view."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=inspect_latest_2d_detections,
        )
    )

    def inspect_in_view_candidates(
        object_ids: list[int], reference_id: str | None = None
    ) -> ToolResult:
        ids = _candidate_ids(object_ids, evidence)
        if any(object_id not in evidence.projections for object_id in ids):
            raise ValueError("every inspected object must be in the sampled camera frustum")
        result = _inspect_task_candidates(base, evidence, ids, reference_id)
        response = dict(result.response)
        response["current_camera_projections"] = {
            str(object_id): evidence.projections[object_id] for object_id in ids
        }
        response["current_rgb_inspected"] = False
        return ToolResult(response, result.media)

    registry.register(
        ToolSpec(
            name="inspect_in_view_candidates",
            description=(
                "Optional single VLM evidence check for up to four ids already proven "
                "inside the sampled frustum. It uses historical scene snapshots and "
                "optional curated references, never the current RGB pixels."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "object_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 4,
                    },
                    "reference_id": {"type": "string"},
                },
                "required": ["object_ids"],
            },
            handler=inspect_in_view_candidates,
        )
    )
    return registry


def _decision(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _world_header(frame_id: str) -> dict[str, Any]:
    return {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": frame_id}


def _vector3(value: Any) -> dict[str, float]:
    values = value if isinstance(value, (list, tuple)) and len(value) >= 3 else [0, 0, 0]
    return {"x": _number(values[0]), "y": _number(values[1]), "z": _number(values[2])}


def _furniture_info(value: dict[str, Any], world_frame: str) -> dict[str, Any]:
    object_id = _object_id(value)
    role = value.get("furniture_role")
    role = role if isinstance(role, dict) else {}
    return {
        "furniture_id": object_id if object_id is not None else -1,
        "name": str(value.get("name") or ""),
        "label": str(
            role.get("classification_label")
            or value.get("classification_label")
            or value.get("label")
            or ""
        ),
        "description": str(value.get("description") or ""),
        "position": {
            "header": _world_header(world_frame),
            "point": _vector3(value.get("center_world")),
        },
        "size": _vector3(value.get("size_m")),
    }


def _object_info(value: dict[str, Any], world_frame: str) -> dict[str, Any]:
    yaw = _number(value.get("yaw_rad"))
    return {
        "object_id": _object_id(value) if _object_id(value) is not None else -1,
        "label": str(value.get("label") or ""),
        "name": str(value.get("name") or ""),
        "description": str(value.get("description") or ""),
        "pose": {
            "header": _world_header(world_frame),
            "pose": {
                "position": _vector3(value.get("center_world")),
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": math.sin(yaw * 0.5),
                    "w": math.cos(yaw * 0.5),
                },
            },
        },
        "size": _vector3(value.get("size_m")),
    }


def _decision_status(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return str(value or "").strip().upper()


def build_navigation_result(
    response: QaResponse,
    evidence: TaskEvidence,
    *,
    allow_follow_up_question: bool,
    world_frame: str,
) -> dict[str, Any]:
    decision = _decision(response.answer)
    invalid_reason = ""
    if not evidence.primary_called:
        invalid_reason = "model did not call locate_navigation_target"
    if decision is None:
        invalid_reason = invalid_reason or "model returned invalid task JSON"
        decision = {}
    status = _decision_status(decision.get("status"))
    status_aliases = {
        "0": "FOUND",
        "1": "NOT_FOUND",
        "2": "NEED_CONFIRM",
    }
    status = status_aliases.get(status, status)
    if status not in {"FOUND", "NOT_FOUND", "NEED_CONFIRM"}:
        invalid_reason = invalid_reason or "model returned an invalid navigation status"
        status = "NOT_FOUND"

    requested_furniture = _integer(decision.get("target_furniture_id"))
    target_object_id = _integer(decision.get("target_object_id"))
    if requested_furniture is None and target_object_id is not None:
        related = evidence.object_to_furniture.get(target_object_id, [])
        requested_furniture = related[0] if related else None

    # A unique explicitly requested furniture destination is actionable even
    # when the stable SG has not yet confirmed the requested object there.
    if len(evidence.explicit_furniture_ids) == 1:
        status = "FOUND"
        requested_furniture = evidence.explicit_furniture_ids[0]

    if status == "NEED_CONFIRM" and not allow_follow_up_question:
        status = "FOUND" if evidence.ranked_furniture_ids else "NOT_FOUND"
        requested_furniture = (
            evidence.ranked_furniture_ids[0]
            if evidence.ranked_furniture_ids
            else None
        )

    if status == "FOUND" and requested_furniture not in evidence.furniture:
        invalid_reason = invalid_reason or "model selected an unwitnessed target_furniture_id"
        status = "NOT_FOUND"
        requested_furniture = None

    raw_exploration = decision.get("exploration_furniture_ids")
    exploration_ids = []
    if isinstance(raw_exploration, list):
        for value in raw_exploration:
            furniture_id = _integer(value)
            if (
                furniture_id is not None
                and furniture_id in evidence.furniture
                and furniture_id not in exploration_ids
            ):
                exploration_ids.append(furniture_id)
    if status == "NOT_FOUND" and (invalid_reason or not exploration_ids):
        exploration_ids = list(evidence.ranked_furniture_ids[:5])
    exploration_ids = exploration_ids[:5]

    reason = str(decision.get("reason") or response.reasoning or "")
    if invalid_reason:
        reason = f"{invalid_reason}. {reason}".strip()
    vlm_description = str(
        decision.get("vlm_description") or response.reasoning or reason
    )
    follow_up = str(decision.get("follow_up_question") or "")
    if status == "NEED_CONFIRM" and not follow_up:
        follow_up = "你希望我去哪个位置寻找这个物体？"

    return {
        "status": {"FOUND": 0, "NOT_FOUND": 1, "NEED_CONFIRM": 2}[status],
        "reason": reason,
        "target_furniture": (
            _furniture_info(evidence.furniture[requested_furniture], world_frame)
            if status == "FOUND" and requested_furniture is not None
            else None
        ),
        "exploration_furniture": (
            [
                _furniture_info(evidence.furniture[furniture_id], world_frame)
                for furniture_id in exploration_ids
            ]
            if status == "NOT_FOUND"
            else []
        ),
        "follow_up_question": follow_up if status == "NEED_CONFIRM" else "",
        "vlm_description": vlm_description,
    }


def build_find_object_in_view_result(
    response: QaResponse,
    evidence: TaskEvidence,
    *,
    world_frame: str,
) -> dict[str, Any]:
    decision = _decision(response.answer)
    invalid_reason = ""
    if not evidence.primary_called:
        invalid_reason = "model did not call find_objects_in_view"
    if decision is None:
        invalid_reason = invalid_reason or "model returned invalid task JSON"
        decision = {}
    status = _decision_status(decision.get("status"))
    status = {"0": "FOUND", "1": "NOT_FOUND"}.get(status, status)
    if status not in {"FOUND", "NOT_FOUND"}:
        invalid_reason = invalid_reason or "model returned an invalid in-view status"
        status = "NOT_FOUND"

    target_id = _integer(decision.get("target_object_id"))
    if status == "FOUND" and (
        target_id not in evidence.objects or target_id not in evidence.projections
    ):
        invalid_reason = invalid_reason or "model selected an unwitnessed target_object_id"
        status = "NOT_FOUND"
        target_id = None

    raw_suspected = decision.get("suspected_object_ids")
    suspected_ids = []
    invalid_suspected = False
    if isinstance(raw_suspected, list):
        for value in raw_suspected:
            object_id = _integer(value)
            if object_id not in evidence.projections:
                invalid_suspected = True
                continue
            if object_id is not None and object_id not in suspected_ids:
                suspected_ids.append(object_id)
    if invalid_suspected:
        invalid_reason = invalid_reason or "model selected an unwitnessed suspected object"
    if status == "NOT_FOUND" and invalid_reason:
        suspected_ids = list(evidence.projections)[:4]
    suspected_ids = suspected_ids[:4]

    reason = str(decision.get("reason") or response.reasoning or "")
    if invalid_reason:
        reason = f"{invalid_reason}. {reason}".strip()
    vlm_description = str(
        decision.get("vlm_description") or response.reasoning or reason
    )
    target_object = None
    target_in_camera = None
    if status == "FOUND" and target_id is not None:
        target_object = _object_info(evidence.objects[target_id], world_frame)
        target_in_camera = evidence.projections[target_id]["object_in_head_cam"]

    suspected_objects = [
        _object_info(evidence.objects[object_id], world_frame)
        for object_id in suspected_ids
    ]
    suspected_in_camera = [
        evidence.projections[object_id]["object_in_head_cam"]
        for object_id in suspected_ids
    ]
    return {
        "status": 0 if status == "FOUND" else 1,
        "reason": reason,
        "target_object": target_object,
        "target_object_in_head_cam": target_in_camera,
        "suspected_objects": suspected_objects if status == "NOT_FOUND" else [],
        "suspected_objects_in_head_cam": (
            suspected_in_camera if status == "NOT_FOUND" else []
        ),
        "vlm_description": vlm_description,
    }


class StructuredTaskAgent:
    """Validate a VLM decision against tool evidence and fill ROS-shaped fields."""

    def __init__(
        self,
        agent: Any,
        evidence: TaskEvidence,
        *,
        task: str,
        allow_follow_up_question: bool = True,
        world_frame: str = "map",
    ):
        if task not in {TASK_NAVIGATION, TASK_FIND_OBJECT_IN_VIEW}:
            raise ValueError(f"unsupported structured task: {task}")
        self.agent = agent
        self.evidence = evidence
        self.task = task
        self.allow_follow_up_question = bool(allow_follow_up_question)
        self.world_frame = world_frame

    def answer_query(
        self,
        query: str,
        *,
        allow_follow_up_question: bool | None = None,
    ) -> QaResponse:
        allow_follow_up = (
            self.allow_follow_up_question
            if allow_follow_up_question is None
            else bool(allow_follow_up_question)
        )
        policy = ""
        if self.task == TASK_NAVIGATION:
            policy = (
                "\nRuntime policy: allow_follow_up_question="
                + ("true" if allow_follow_up else "false")
                + "."
            )
        model_response = self.agent.answer_query(
            f"Original task_description:\n{query.strip()}{policy}"
        )
        if self.task == TASK_NAVIGATION:
            result = build_navigation_result(
                model_response,
                self.evidence,
                allow_follow_up_question=allow_follow_up,
                world_frame=self.world_frame,
            )
        else:
            result = build_find_object_in_view_result(
                model_response,
                self.evidence,
                world_frame=self.world_frame,
            )
        model_response.history["task"] = self.task
        model_response.history["structured_result"] = result
        return QaResponse(
            reasoning=model_response.reasoning,
            answer=result,
            raw_text=model_response.raw_text,
            history=model_response.history,
        )
