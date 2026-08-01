#!/usr/bin/env python3
"""Replace Roomie DSG object descriptions with DAM-generated descriptions."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw

from roomie_dsg_viewer import (
    DEFAULT_CONFIG,
    DEFAULT_JSON,
    as_path,
    load_pipeline_params,
    normalize_graph,
    resolve_json_from_config,
    snapshot_paths,
)


DEFAULT_DAAAM_SRC = Path("/home/agxi/RealityLab/memory_ws/src/daaam/src")
LOCAL_DAM_MODEL = Path("/home/lindenbot/hugging_face/DAM-3B")
DEFAULT_QUERY = (
    "Describe only the visible object inside the masked region in detail. "
    "Include its color, material, shape, parts, pose, and distinctive visual "
    "details. Do not describe unrelated background."
)


def default_model_path() -> str:
    if LOCAL_DAM_MODEL.exists():
        return str(LOCAL_DAM_MODEL)
    return "nvidia/DAM-3B"


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
        help=f"pipeline YAML used to resolve the DSG path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="explicit Roomie DSG JSON file; overrides YAML persistence paths",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSON copy; default creates a non-overwriting .dam_described copy",
    )
    parser.add_argument(
        "--force-output",
        action="store_true",
        help="allow replacing --output if it already exists",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        action="append",
        default=None,
        help="process only this object id; may be repeated",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of selected objects to touch, useful for smoke tests",
    )
    parser.add_argument(
        "--skip-existing-description",
        action="store_true",
        help="skip objects with a non-empty description; useful when resuming a copy",
    )
    parser.add_argument("--model-path", default=default_model_path(), help="DAM model path")
    parser.add_argument("--conv-mode", default="v1", help="DAM conversation mode")
    parser.add_argument("--prompt-mode", default="focal_prompt", help="DAM prompt mode")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="DAM prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--bbox-pad-px",
        type=float,
        default=2.0,
        help="padding added around the snapshot bbox when building the mask",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="checkpoint output after this many object updates",
    )
    parser.add_argument(
        "--daaam-src",
        type=Path,
        default=DEFAULT_DAAAM_SRC,
        help="source path added to PYTHONPATH before importing daaam",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve inputs and count work without loading DAM or writing output",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="abort on the first per-object DAM failure",
    )
    return parser.parse_args()


def resolve_input_path(args: argparse.Namespace) -> tuple[Path, Path | None]:
    positional_json: Path | None = None
    config_path: Path | None = as_path(args.config)
    if args.path is not None:
        positional = as_path(args.path)
        if positional is not None and positional.suffix.lower() in (".yaml", ".yml"):
            config_path = positional
        else:
            positional_json = positional

    params = load_pipeline_params(config_path)
    json_path = (
        as_path(args.json)
        or positional_json
        or resolve_json_from_config(params)
        or DEFAULT_JSON.resolve()
    )
    if not json_path.exists():
        raise FileNotFoundError(f"DSG JSON does not exist: {json_path}")
    return json_path, config_path


def choose_output_path(input_path: Path,
                       output_arg: Path | None,
                       force_output: bool) -> Path:
    if output_arg is not None:
        output_path = output_arg.expanduser().resolve()
        if output_path == input_path.resolve():
            raise ValueError("output path must not be the input DSG path")
        if output_path.exists() and not force_output:
            raise FileExistsError(f"output exists; pass --force-output: {output_path}")
        return output_path

    suffix = input_path.suffix or ".json"
    first = input_path.with_name(f"{input_path.stem}.dam_described{suffix}")
    if not first.exists():
        return first

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = input_path.with_name(f"{input_path.stem}.dam_described_{stamp}{suffix}")
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = input_path.with_name(
            f"{input_path.stem}.dam_described_{stamp}_{index}{suffix}"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not choose a unique output path")


def object_lists(root: dict[str, Any]) -> list[list[Any]]:
    lists: list[list[Any]] = []
    if isinstance(root.get("objects"), list):
        lists.append(root["objects"])
    object_graph = root.get("object_graph")
    if isinstance(object_graph, dict) and isinstance(object_graph.get("objects"), list):
        lists.append(object_graph["objects"])
    return lists


def canonical_objects(root: dict[str, Any]) -> list[Any]:
    lists = object_lists(root)
    if not lists:
        raise ValueError("DSG JSON does not contain an objects list")
    return lists[0]


def object_id_of(obj: Any) -> int | None:
    if not isinstance(obj, dict):
        return None
    object_id = obj.get("object_id")
    return object_id if isinstance(object_id, int) else None


def current_description(root: dict[str, Any], object_id: int) -> str:
    for objects in object_lists(root):
        for obj in objects:
            if not isinstance(obj, dict) or obj.get("object_id") != object_id:
                continue
            description = obj.get("description")
            if isinstance(description, str) and description.strip():
                return description.strip()
    return ""


def set_description(root: dict[str, Any], object_id: int, description: str) -> None:
    for objects in object_lists(root):
        for obj in objects:
            if isinstance(obj, dict) and obj.get("object_id") == object_id:
                obj["description"] = description


def snapshot_ref(obj: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = obj.get("snapshot")
    return snapshot if isinstance(snapshot, dict) else None


def snapshot_image_index(obj: dict[str, Any]) -> int | None:
    snapshot = snapshot_ref(obj)
    if snapshot is None:
        return None
    image_index = snapshot.get("image_index")
    return image_index if isinstance(image_index, int) else None


def snapshot_bbox(obj: dict[str, Any]) -> list[float] | None:
    snapshot = snapshot_ref(obj)
    if snapshot is None:
        return None
    bbox = snapshot.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    try:
        values = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def make_bbox_mask(size: tuple[int, int],
                   bbox_xyxy: list[float],
                   pad_px: float) -> Image.Image | None:
    width, height = size
    x0 = min(bbox_xyxy[0], bbox_xyxy[2]) - pad_px
    y0 = min(bbox_xyxy[1], bbox_xyxy[3]) - pad_px
    x1 = max(bbox_xyxy[0], bbox_xyxy[2]) + pad_px
    y1 = max(bbox_xyxy[1], bbox_xyxy[3]) + pad_px

    x0_i = max(0, min(width, math.floor(x0)))
    y0_i = max(0, min(height, math.floor(y0)))
    x1_i = max(0, min(width, math.ceil(x1)))
    y1_i = max(0, min(height, math.ceil(y1)))
    if x1_i <= x0_i or y1_i <= y0_i:
        return None

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x0_i, y0_i, x1_i, y1_i], fill=255)
    return mask


def atomic_write_json(path: Path, root: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as stream:
        json.dump(root, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp_path, path)


def prepare_dam_agent(args: argparse.Namespace):
    daaam_src = as_path(args.daaam_src)
    if daaam_src is not None and daaam_src.exists():
        sys.path.insert(0, str(daaam_src))

    model_path = Path(args.model_path).expanduser()
    if model_path.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        args.model_path = str(model_path.resolve())

    import torch
    from daaam.query_manager.dam import DAMAgentPanoptic

    print(
        "Torch CUDA: "
        f"available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()} "
        f"cuda={torch.version.cuda}"
    )
    print(f"DAM model path: {args.model_path}")
    return DAMAgentPanoptic(
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        prompt_mode=args.prompt_mode,
    )


def selected_objects(root: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    id_filter = set(args.object_id or [])
    selected: list[dict[str, Any]] = []
    for obj in canonical_objects(root):
        object_id = object_id_of(obj)
        if object_id is None:
            continue
        if id_filter and object_id not in id_filter:
            continue
        selected.append(obj)
    if args.limit is not None:
        selected = selected[:max(0, args.limit)]
    return selected


def needs_dam(root: dict[str, Any],
              objects: list[dict[str, Any]],
              image_paths: dict[int, Path],
              skip_existing: bool) -> bool:
    for obj in objects:
        object_id = object_id_of(obj)
        if object_id is None:
            continue
        if skip_existing and current_description(root, object_id):
            continue
        image_index = snapshot_image_index(obj)
        if image_index is None or image_index not in image_paths:
            continue
        if snapshot_bbox(obj) is None:
            continue
        return True
    return False


def describe_object(agent: Any,
                    image_path: Path,
                    bbox_xyxy: list[float],
                    args: argparse.Namespace) -> str:
    with Image.open(image_path) as stream:
        image = stream.convert("RGB")
    mask = make_bbox_mask(image.size, bbox_xyxy, args.bbox_pad_px)
    if mask is None:
        raise ValueError("invalid or empty bbox mask")
    description = agent.query_single(
        image=image,
        mask=mask,
        query=args.query,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    return str(description or "").strip()


def main() -> int:
    args = parse_args()
    json_path, config_path = resolve_input_path(args)
    output_path = choose_output_path(json_path, args.output, args.force_output)

    with json_path.open("r", encoding="utf-8") as stream:
        root = json.load(stream)
    if not isinstance(root, dict):
        raise ValueError("DSG JSON root must be an object")

    graph = normalize_graph(root)
    image_paths = snapshot_paths(graph, json_path)
    objects = selected_objects(root, args)
    if not objects:
        print("No matching objects selected.", file=sys.stderr)
        return 2

    dam_needed = needs_dam(
        root,
        objects,
        image_paths,
        skip_existing=args.skip_existing_description,
    )
    print(f"Input DSG: {json_path}")
    if config_path is not None:
        print(f"Using config: {config_path}")
    print(f"Output DSG copy: {output_path}")
    print(f"Selected objects: {len(objects)}")
    print(f"Snapshot images resolved: {len(image_paths)}")
    print(f"DAM needed: {dam_needed}")

    if args.dry_run:
        return 0

    agent = prepare_dam_agent(args) if dam_needed else None
    save_every = max(1, args.save_every)
    updated_since_save = 0
    stats = {
        "described": 0,
        "existing": 0,
        "no_snapshot": 0,
        "missing_image": 0,
        "bad_bbox": 0,
        "errors": 0,
    }

    for index, obj in enumerate(objects, start=1):
        object_id = object_id_of(obj)
        if object_id is None:
            continue

        label = str(obj.get("label") or obj.get("description") or "object")
        existing = current_description(root, object_id)
        if existing and args.skip_existing_description:
            set_description(root, object_id, existing)
            stats["existing"] += 1
            updated_since_save += 1
            print(f"[{index}/{len(objects)}] object {object_id} {label}: existing")
        else:
            image_index = snapshot_image_index(obj)
            bbox = snapshot_bbox(obj)
            if image_index is None:
                set_description(root, object_id, existing)
                stats["no_snapshot"] += 1
                updated_since_save += 1
                print(f"[{index}/{len(objects)}] object {object_id} {label}: no snapshot")
            elif image_index not in image_paths:
                set_description(root, object_id, existing)
                stats["missing_image"] += 1
                updated_since_save += 1
                print(
                    f"[{index}/{len(objects)}] object {object_id} {label}: "
                    f"missing snapshot image {image_index}"
                )
            elif bbox is None:
                set_description(root, object_id, existing)
                stats["bad_bbox"] += 1
                updated_since_save += 1
                print(f"[{index}/{len(objects)}] object {object_id} {label}: bad bbox")
            else:
                try:
                    assert agent is not None
                    description = describe_object(agent, image_paths[image_index], bbox, args)
                    set_description(root, object_id, description)
                    stats["described"] += 1
                    updated_since_save += 1
                    print(
                        f"[{index}/{len(objects)}] object {object_id} {label}: "
                        f"{description[:120]}"
                    )
                except Exception as exc:
                    stats["errors"] += 1
                    set_description(root, object_id, existing)
                    updated_since_save += 1
                    print(
                        f"[{index}/{len(objects)}] object {object_id} {label}: "
                        f"ERROR {exc}",
                        file=sys.stderr,
                    )
                    if args.stop_on_error:
                        atomic_write_json(output_path, root)
                        raise

        if updated_since_save >= save_every:
            atomic_write_json(output_path, root)
            updated_since_save = 0

    if updated_since_save or not output_path.exists():
        atomic_write_json(output_path, root)

    print("Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"Wrote: {output_path}")
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
