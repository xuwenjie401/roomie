#!/usr/bin/env python3
"""Resident framed-JSON worker for Roomie's durable DAM tasks.

The process loads DAM once, reads immutable AssetStore paths supplied by the
C++ adapter, and returns model text. It never reads or writes a scene graph.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import traceback
from typing import Any, NoReturn


PROTOCOL = "roomie.dam-worker.v1"
DEFAULT_QUERY = (
    "<image>\nDescribe only the visible object inside the masked region in detail. "
    "Include its color, material, shape, parts, pose, and distinctive visual "
    "details. Do not describe unrelated background."
)

# The public demo exposes ``focal_prompt`` as a UI-facing name, then maps it
# to the crop mode consumed by DescribeAnythingModel. Accept both spellings so
# existing Roomie YAML remains readable while the durable prompt provenance
# records the exact configured value.
PROMPT_MODES = {
    "focal_prompt": "full+focal_crop",
    "full+focal_crop": "full+focal_crop",
}


class WorkerFailure(Exception):
    """An explicitly classified request or startup failure."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--dam-src",
        dest="dam_src",
        type=Path,
        default=None,
        help="official NVlabs/describe-anything checkout added to PYTHONPATH",
    )
    parser.add_argument("--conv-mode", default="v1")
    parser.add_argument("--prompt-mode", default="focal_prompt")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--bbox-pad-px", type=float, default=2.0)
    parser.add_argument("--max-message-bytes", type=int, default=8 * 1024 * 1024)
    return parser.parse_args()


def read_exact(size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(0, remaining)
        if not chunk:
            raise EOFError("DAM request pipe closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(max_message_bytes: int) -> dict[str, Any]:
    size = struct.unpack("<I", read_exact(4))[0]
    if size > max_message_bytes:
        raise WorkerFailure(
            f"DAM request frame is too large: {size} > {max_message_bytes}",
            retryable=False,
        )
    try:
        value = json.loads(read_exact(size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerFailure(f"DAM request is not valid UTF-8 JSON: {exc}", retryable=False)
    if not isinstance(value, dict):
        raise WorkerFailure("DAM request JSON root must be an object", retryable=False)
    return value


def write_message(value: dict[str, Any], max_message_bytes: int) -> None:
    body = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(body) > max_message_bytes:
        raise RuntimeError("DAM response frame exceeds configured size limit")
    data = struct.pack("<I", len(body)) + body
    offset = 0
    while offset < len(data):
        written = os.write(1, data[offset:])
        if written <= 0:
            raise BrokenPipeError("DAM response pipe closed")
        offset += written


def startup_retryable(exc: BaseException) -> bool:
    text = str(exc).lower()
    transient_markers = (
        "out of memory",
        "cuda error",
        "temporarily unavailable",
        "resource busy",
        "connection",
        "timeout",
    )
    return any(marker in text for marker in transient_markers)


def prepare_dam_agent(args: argparse.Namespace) -> Any:
    if args.dam_src is not None:
        dam_src = args.dam_src.expanduser()
        if not dam_src.is_dir():
            raise WorkerFailure(
                f"configured DAM source directory does not exist: {dam_src}",
                retryable=False,
            )
        sys.path.insert(0, str(dam_src.resolve()))

    resolved_prompt_mode = PROMPT_MODES.get(args.prompt_mode)
    if resolved_prompt_mode is None:
        raise WorkerFailure(
            f"unsupported official DAM prompt mode: {args.prompt_mode}",
            retryable=False,
        )
    if "<image>" not in args.query:
        raise WorkerFailure(
            "official DAM query must contain the <image> token",
            retryable=False,
        )

    model_path = Path(args.model_path).expanduser()
    if model_path.is_absolute() or args.model_path.startswith("."):
        if not model_path.exists():
            raise WorkerFailure(
                f"configured local DAM model does not exist: {model_path}",
                retryable=False,
            )
    if model_path.exists():
        args.model_path = str(model_path.resolve())
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        # stdout is reserved exclusively for the framed protocol. Redirect
        # import/model diagnostics emitted by third-party libraries to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from PIL import Image, ImageDraw  # noqa: F401
            from dam import DescribeAnythingModel, disable_torch_init
    except (ImportError, ModuleNotFoundError) as exc:
        raise WorkerFailure(
            f"DAM Python dependency is unavailable: {exc}", retryable=False
        ) from exc

    print(
        "Roomie DAM startup: "
        f"model_id={args.model_id} model_path={args.model_path} "
        f"cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()} cuda={torch.version.cuda}",
        file=sys.stderr,
        flush=True,
    )
    try:
        with contextlib.redirect_stdout(sys.stderr):
            disable_torch_init()
            return DescribeAnythingModel(
                model_path=args.model_path,
                conv_mode=args.conv_mode,
                prompt_mode=resolved_prompt_mode,
            )
    except Exception as exc:
        raise WorkerFailure(
            f"DAM model initialization failed: {type(exc).__name__}: {exc}",
            retryable=startup_retryable(exc),
        ) from exc


def finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerFailure("bbox coordinate is not numeric", retryable=False)
    result = float(value)
    if not math.isfinite(result):
        raise WorkerFailure("bbox coordinate is not finite", retryable=False)
    return result


def validate_request(value: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if value.get("protocol") != PROTOCOL or value.get("type") != "describe":
        raise WorkerFailure("unsupported DAM protocol envelope", retryable=False)
    request_id = value.get("request_id")
    if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
        raise WorkerFailure("DAM request_id is invalid", retryable=False)
    task = value.get("task")
    input_value = value.get("input")
    if not isinstance(task, dict) or not isinstance(input_value, dict):
        raise WorkerFailure("DAM task/input envelope is missing", retryable=False)
    snapshots = input_value.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise WorkerFailure("DAM request has no snapshots", retryable=False)
    return request_id, input_value


def make_bbox_mask(size: tuple[int, int],
                   bbox_xyxy: list[float],
                   pad_px: float) -> Any:
    from PIL import Image, ImageDraw

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
        raise WorkerFailure("snapshot bbox is empty after clipping", retryable=False)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x0_i, y0_i, x1_i, y1_i], fill=255)
    return mask


def _json_object_candidate(raw_output: str) -> dict[str, Any] | None:
    cleaned = raw_output.strip()
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first < 0 or last <= first:
        return None
    try:
        candidate = json.loads(cleaned[first : last + 1])
    except json.JSONDecodeError:
        return None
    return candidate if isinstance(candidate, dict) else None


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return [item.strip() for item in value if item.strip()]


def normalize_structured_output(
    raw_output: str, input_value: dict[str, Any]
) -> str:
    """Fence model-owned facts and inject immutable Roomie provenance.

    Official DAM is primarily a region captioner, so ordinary free text is
    conservatively repaired into the durable schema. The repair keeps visual
    attribute arrays empty, uses zero confidence, and carries an explicit
    marker; it never invents parsed visual facts. C++ records that marker as a
    schema-repaired path while retaining the exact model text.
    """

    candidate = _json_object_candidate(raw_output)
    canonical_name = candidate.get("canonical_name") if candidate else None
    short_description = candidate.get("short_description") if candidate else None
    attributes = candidate.get("visual_attributes") if candidate else None
    uncertain = (
        _string_list(candidate.get("uncertain_or_not_visible"))
        if candidate
        else None
    )
    confidence = candidate.get("confidence") if candidate else None
    model_schema_valid = (
        isinstance(canonical_name, str)
        and bool(canonical_name.strip())
        and isinstance(short_description, str)
        and bool(short_description.strip())
        and isinstance(attributes, dict)
        and uncertain is not None
        and not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and math.isfinite(float(confidence))
        and 0.0 <= float(confidence) <= 1.0
    )

    normalized_attributes: dict[str, list[str]] = {}
    attribute_fields = (
        "colors",
        "materials",
        "shape",
        "visible_parts",
        "state_or_pose",
        "distinctive_marks",
        "visible_text",
    )
    if model_schema_valid:
        for field in attribute_fields:
            normalized = _string_list(attributes.get(field))
            if normalized is None:
                model_schema_valid = False
                break
            normalized_attributes[field] = normalized

    if not model_schema_valid:
        label = input_value.get("label")
        canonical_name = (
            label.strip() if isinstance(label, str) and label.strip() else "object"
        )
        short_description = raw_output.strip()
        normalized_attributes = {field: [] for field in attribute_fields}
        uncertain = []
        confidence = 0.0

    snapshots = input_value.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        return raw_output
    primary = snapshots[0]
    if not isinstance(primary, dict):
        return raw_output
    mask_source = primary.get("mask_source")
    if mask_source not in ("instance_mask", "bbox_fallback"):
        return raw_output
    evidence_snapshot_ids = [
        snapshot["evidence_hash"]
        for snapshot in snapshots
        if isinstance(snapshot, dict)
        and isinstance(snapshot.get("evidence_hash"), str)
        and snapshot["evidence_hash"]
    ]

    # retrieval_text is a stable projection of validated visual fields. It is
    # not trusted as a second free-form answer from the model.
    retrieval_parts = [canonical_name.strip(), short_description.strip()]
    for field in attribute_fields:
        retrieval_parts.extend(normalized_attributes[field])
    retrieval_text = "; ".join(dict.fromkeys(retrieval_parts))
    normalized_output = {
        "canonical_name": canonical_name.strip(),
        "short_description": short_description.strip(),
        "retrieval_text": retrieval_text,
        "visual_attributes": normalized_attributes,
        "uncertain_or_not_visible": uncertain,
        "confidence": float(confidence),
        "evidence_snapshot_ids": evidence_snapshot_ids,
        "mask_source": mask_source,
        # The C++ normalized projection intentionally drops this extra field,
        # while raw_text keeps the exact model response for audit/debugging.
        "raw_model_output": raw_output,
    }
    if not model_schema_valid:
        normalized_output["_roomie_parse_path"] = "schema_repaired"
        normalized_output["_roomie_schema_error"] = (
            "official DAM output was conservatively wrapped; visual attribute "
            "arrays were not schema-extracted"
        )
    return json.dumps(
        normalized_output, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def describe(agent: Any, input_value: dict[str, Any], args: argparse.Namespace) -> str:
    # SnapshotBank orders the retained set with its primary/best evidence first.
    # Roomie's v1 artifact schema resolves and pins the entire set but invokes
    # the official DAM API on the primary view.
    snapshot = input_value["snapshots"][0]
    if not isinstance(snapshot, dict):
        raise WorkerFailure("primary snapshot is not an object", retryable=False)
    asset = snapshot.get("durable_asset")
    if not isinstance(asset, dict):
        raise WorkerFailure("primary snapshot has no durable asset", retryable=False)
    asset_id = snapshot.get("source_frame_asset_id")
    if (
        not isinstance(asset_id, str)
        or not asset_id
        or asset.get("asset_id") != asset_id
        or asset.get("access_mode") != "read_only"
        or asset.get("immutable") is not True
    ):
        raise WorkerFailure("durable asset envelope is inconsistent", retryable=False)
    path_value = asset.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise WorkerFailure("durable asset path is missing", retryable=False)
    image_path = Path(path_value)
    if not image_path.is_absolute():
        raise WorkerFailure("durable asset path must be absolute", retryable=False)
    bbox_value = snapshot.get("bbox_xyxy")
    if not isinstance(bbox_value, list) or len(bbox_value) != 4:
        raise WorkerFailure("primary snapshot bbox is invalid", retryable=False)
    bbox = [finite_number(value) for value in bbox_value]

    try:
        from PIL import Image

        with Image.open(image_path, "r") as stream:
            image = stream.convert("RGB")
    except FileNotFoundError as exc:
        raise WorkerFailure(
            f"durable DAM asset disappeared: {image_path}", retryable=True
        ) from exc
    except OSError as exc:
        raise WorkerFailure(
            f"cannot read durable DAM asset {image_path}: {exc}", retryable=True
        ) from exc
    mask = make_bbox_mask(image.size, bbox, args.bbox_pad_px)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            output = agent.get_description(
                image,
                mask,
                args.query,
                streaming=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
            )
    except Exception as exc:
        raise WorkerFailure(
            f"DAM inference failed: {type(exc).__name__}: {exc}", retryable=True
        ) from exc
    text = str(output or "").strip()
    if not text:
        raise WorkerFailure("DAM inference returned empty text", retryable=True)
    return normalize_structured_output(text, input_value)


def serve(agent: Any, args: argparse.Namespace) -> NoReturn:
    write_message(
        {"protocol": PROTOCOL, "type": "ready", "model_id": args.model_id},
        args.max_message_bytes,
    )
    while True:
        try:
            request = read_message(args.max_message_bytes)
        except EOFError:
            raise SystemExit(0)

        request_id: int | None = None
        try:
            request_id, input_value = validate_request(request)
            raw_output = describe(agent, input_value, args)
            response = {
                "protocol": PROTOCOL,
                "type": "result",
                "request_id": request_id,
                "success": True,
                "raw_output": raw_output,
            }
        except WorkerFailure as exc:
            candidate_id = request.get("request_id")
            if isinstance(candidate_id, int) and not isinstance(candidate_id, bool):
                request_id = candidate_id
            response = {
                "protocol": PROTOCOL,
                "type": "result",
                "request_id": request_id if request_id is not None else 0,
                "success": False,
                "retryable": exc.retryable,
                "error": str(exc),
            }
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "protocol": PROTOCOL,
                "type": "result",
                "request_id": request_id if request_id is not None else 0,
                "success": False,
                "retryable": True,
                "error": f"unexpected DAM worker error: {type(exc).__name__}: {exc}",
            }
        write_message(response, args.max_message_bytes)


def main() -> int:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    args = parse_args()
    if args.max_message_bytes < 1024 or args.max_new_tokens <= 0:
        print("invalid DAM worker limits", file=sys.stderr)
        return 2
    try:
        agent = prepare_dam_agent(args)
    except WorkerFailure as exc:
        try:
            write_message(
                {
                    "protocol": PROTOCOL,
                    "type": "startup_error",
                    "retryable": exc.retryable,
                    "error": str(exc),
                },
                args.max_message_bytes,
            )
        except Exception:
            traceback.print_exc(file=sys.stderr)
        return 1
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        try:
            write_message(
                {
                    "protocol": PROTOCOL,
                    "type": "startup_error",
                    "retryable": startup_retryable(exc),
                    "error": f"unexpected DAM startup failure: {type(exc).__name__}: {exc}",
                },
                args.max_message_bytes,
            )
        except Exception:
            traceback.print_exc(file=sys.stderr)
        return 1
    serve(agent, args)


if __name__ == "__main__":
    raise SystemExit(main())
