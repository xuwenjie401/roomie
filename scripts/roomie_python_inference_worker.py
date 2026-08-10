#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import struct
import sys
import time
import traceback


REQUEST_MAGIC = b"RIEQ2"
RESPONSE_MAGIC = b"RIRS2"
PATCH_VALUES = 60 * 60
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


_BINARY_STDOUT = None


def configure_binary_stdout() -> None:
    global _BINARY_STDOUT
    if _BINARY_STDOUT is None:
        _BINARY_STDOUT = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
        sys.stdout = sys.stderr


class ProtocolError(RuntimeError):
    pass


def load_text_prompt_file(path: str) -> list[str]:
    resolved_path = os.path.abspath(os.path.expanduser(path))
    with open(resolved_path, "r", encoding="utf-8") as stream:
        labels = [line.strip() for line in stream if line.strip()]
    if not labels:
        raise ValueError(f"text prompt file is empty: {resolved_path}")
    return labels


def load_label_confidence_thresholds(
    path: str, fallback_by_stage: dict[str, float]
) -> dict[str, dict]:
    if not path:
        return {
            stage: {"default": float(fallback), "labels": {}}
            for stage, fallback in fallback_by_stage.items()
        }
    resolved_path = os.path.abspath(os.path.expanduser(path))
    with open(resolved_path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(
            f"label confidence thresholds must be a JSON object: {resolved_path}"
        )

    def checked_threshold(value, description: str) -> float:
        if type(value) not in (int, float):
            raise ValueError(f"{description} must be a number: {resolved_path}")
        threshold = float(value)
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"{description} must be in [0, 1]: {resolved_path}")
        return threshold

    thresholds: dict[str, dict] = {}
    for stage in fallback_by_stage:
        stage_config = document.get(stage)
        if not isinstance(stage_config, dict):
            raise ValueError(
                f"label confidence thresholds must contain an object for stage "
                f"'{stage}': {resolved_path}"
            )
        if "default" not in stage_config:
            raise ValueError(
                f"label confidence threshold stage '{stage}' is missing default: "
                f"{resolved_path}"
            )
        labels = stage_config.get("labels")
        if not isinstance(labels, dict):
            raise ValueError(
                f"label confidence threshold stage '{stage}' must contain a "
                f"labels object: {resolved_path}"
            )
        checked_labels = {}
        for label, value in labels.items():
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"label confidence threshold labels must be non-empty: "
                    f"{resolved_path}"
                )
            checked_labels[label] = checked_threshold(
                value, f"label confidence threshold for '{label}'"
            )
        thresholds[stage] = {
            "default": checked_threshold(
                stage_config["default"], f"default threshold for stage '{stage}'"
            ),
            "labels": checked_labels,
        }
    return thresholds


def label_confidence_threshold(stage_config: dict, label: str) -> float:
    return stage_config["labels"].get(label, stage_config["default"])


class RequestReader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read_bytes(self, size: int) -> bytes:
        if size < 0 or self.offset + size > len(self.data):
            raise ProtocolError("request body is truncated")
        out = self.data[self.offset : self.offset + size]
        self.offset += size
        return out

    def read_struct(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        chunk = self.read_bytes(size)
        values = struct.unpack("<" + fmt, chunk)
        return values[0] if len(values) == 1 else values

    def read_string(self) -> str:
        size = self.read_struct("I")
        return self.read_bytes(size).decode("utf-8", errors="replace")

    def read_image(self) -> dict:
        width = self.read_struct("i")
        height = self.read_struct("i")
        channels = self.read_struct("i")
        encoding = self.read_string()
        data_size = self.read_struct("I")
        data = self.read_bytes(data_size)
        return {
            "width": width,
            "height": height,
            "channels": channels,
            "encoding": encoding,
            "data": data,
        }

    def read_provenance(self) -> dict:
        provenance = {
            "run_id": self.read_run_id(),
            "frame_id": self.read_struct("Q"),
            "request_id": self.read_struct("Q"),
            "sensor_time_ns": self.read_struct("q"),
            "map_mode": self.read_struct("B"),
            "includes_current_frame": self.read_struct("B"),
            "causality_verified": self.read_struct("B"),
            "map": {
                "map_epoch": self.read_run_id(),
                "map_revision": self.read_struct("Q"),
                "integrated_through_ns": self.read_struct("q"),
            },
            "surface": {
                "map_epoch": self.read_run_id(),
                "surface_revision": self.read_struct("Q"),
                "source_map_revision": self.read_struct("Q"),
            },
        }
        if provenance["map_mode"] not in (0, 1):
            raise ProtocolError("request provenance has invalid map mode")
        for name in ("includes_current_frame", "causality_verified"):
            if provenance[name] not in (0, 1):
                raise ProtocolError(f"request provenance has invalid {name}")
            provenance[name] = bool(provenance[name])
        return provenance

    def read_run_id(self) -> dict:
        return {
            "high": self.read_struct("Q"),
            "low": self.read_struct("Q"),
        }

    def read_pipeline_timing(self) -> dict:
        return {
            "serialize_ms": self.read_struct("d"),
            "pipe_write_ms": self.read_struct("d"),
            "pipe_read_ms": self.read_struct("d"),
            "worker_queue_ms": self.read_struct("d"),
            "response_forward_ms": self.read_struct("d"),
        }

    def require_end(self) -> None:
        if self.offset != len(self.data):
            raise ProtocolError("request body has trailing bytes")


class ResponseWriter:
    def __init__(self):
        self.body = bytearray()
        self.body.extend(RESPONSE_MAGIC)

    def write_struct(self, fmt: str, *values) -> None:
        self.body.extend(struct.pack("<" + fmt, *values))

    def write_string(self, value: str) -> None:
        data = str(value).encode("utf-8")
        self.write_struct("I", len(data))
        self.body.extend(data)

    def write_run_id(self, run_id: dict) -> None:
        self.write_struct("Q", int(run_id.get("high", 0)))
        self.write_struct("Q", int(run_id.get("low", 0)))

    def write_provenance(self, provenance: dict) -> None:
        self.write_run_id(provenance.get("run_id", {}))
        self.write_struct("Q", int(provenance.get("frame_id", 0)))
        self.write_struct("Q", int(provenance.get("request_id", 0)))
        self.write_struct("q", int(provenance.get("sensor_time_ns", 0)))
        self.write_struct("B", int(provenance.get("map_mode", 0)))
        self.write_struct(
            "B", 1 if provenance.get("includes_current_frame", False) else 0
        )
        self.write_struct(
            "B", 1 if provenance.get("causality_verified", False) else 0
        )
        map_stamp = provenance.get("map", {})
        self.write_run_id(map_stamp.get("map_epoch", {}))
        self.write_struct("Q", int(map_stamp.get("map_revision", 0)))
        self.write_struct("q", int(map_stamp.get("integrated_through_ns", 0)))
        surface_stamp = provenance.get("surface", {})
        self.write_run_id(surface_stamp.get("map_epoch", {}))
        self.write_struct("Q", int(surface_stamp.get("surface_revision", 0)))
        self.write_struct("Q", int(surface_stamp.get("source_map_revision", 0)))

    def write_pipeline_timing(self, timing: dict) -> None:
        self.write_struct("d", float(timing.get("serialize_ms", 0.0)))
        self.write_struct("d", float(timing.get("pipe_write_ms", 0.0)))
        self.write_struct("d", float(timing.get("pipe_read_ms", 0.0)))
        self.write_struct("d", float(timing.get("worker_queue_ms", 0.0)))
        self.write_struct("d", float(timing.get("response_forward_ms", 0.0)))

    def encode_response(self, response: dict) -> bytes:
        self.write_struct("q", int(response.get("time_ns", 0)))
        self.write_provenance(response.get("provenance", {}))
        self.write_pipeline_timing(response.get("pipeline_timing", {}))
        self.write_string(response.get("camera_id", ""))
        self.write_struct("B", 1 if response.get("ok", False) else 0)
        self.write_string(response.get("error", ""))
        timings = response.get("timings_ms", {})
        self.write_struct("f", float(timings.get("worker_total", 0.0)))
        self.write_struct("f", float(timings.get("preprocess", 0.0)))
        self.write_struct("f", float(timings.get("owl", 0.0)))
        self.write_struct("f", float(timings.get("robot_filter", 0.0)))
        self.write_struct("f", float(timings.get("boxernet", 0.0)))
        self.write_struct("f", float(timings.get("postprocess", 0.0)))
        filtered_2d = response.get("filtered_2d_detections", [])
        self.write_struct("I", len(filtered_2d))
        for det in filtered_2d:
            self.write_struct("f", float(det["score_2d"]))
            self.write_struct("4f", *det["box_xyxy"])
            self.write_struct("i", int(det["semantic_id"]))
            self.write_string(det["label"])

        detections = response.get("detections", [])
        self.write_struct("I", len(detections))
        for det in detections:
            self.write_struct("3f", *det["center_world"])
            self.write_struct("3f", *det["size_m"])
            self.write_struct("f", float(det["yaw_rad"]))
            self.write_struct("f", float(det["score_2d"]))
            self.write_struct("f", float(det["score_3d"]))
            self.write_struct("4f", *det["box_xyxy"])
            self.write_struct("i", int(det["semantic_id"]))
            self.write_string(det["label"])

        return bytes(self.body)

    def write_response(self, response: dict) -> None:
        body = self.encode_response(response)
        if _BINARY_STDOUT is None:
            raise RuntimeError("binary stdout is not configured")
        _BINARY_STDOUT.write(struct.pack("<I", len(body)))
        _BINARY_STDOUT.write(body)


def read_exact(size: int) -> bytes | None:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message() -> bytes | None:
    size_data = read_exact(4)
    if size_data is None:
        return None
    (size,) = struct.unpack("<I", size_data)
    if size > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"request body too large: {size}")
    body = read_exact(size)
    if body is None:
        raise ProtocolError("request body ended early")
    return body


def empty_provenance() -> dict:
    return {
        "run_id": {"high": 0, "low": 0},
        "frame_id": 0,
        "request_id": 0,
        "sensor_time_ns": 0,
        "map_mode": 0,
        "includes_current_frame": False,
        "causality_verified": False,
        "map": {
            "map_epoch": {"high": 0, "low": 0},
            "map_revision": 0,
            "integrated_through_ns": 0,
        },
        "surface": {
            "map_epoch": {"high": 0, "low": 0},
            "surface_revision": 0,
            "source_map_revision": 0,
        },
    }


def empty_pipeline_timing() -> dict:
    return {
        "serialize_ms": 0.0,
        "pipe_write_ms": 0.0,
        "pipe_read_ms": 0.0,
        "worker_queue_ms": 0.0,
        "response_forward_ms": 0.0,
    }


def parse_request(body: bytes) -> dict:
    reader = RequestReader(body)
    if reader.read_bytes(len(REQUEST_MAGIC)) != REQUEST_MAGIC:
        raise ProtocolError("invalid request magic")
    request = {
        "time_ns": reader.read_struct("q"),
        "provenance": reader.read_provenance(),
        "pipeline_timing": reader.read_pipeline_timing(),
        "camera_id": reader.read_string(),
        "rgb": reader.read_image(),
        "mask": reader.read_image(),
    }
    request["intrinsics"] = {
        "width": reader.read_struct("i"),
        "height": reader.read_struct("i"),
        "fx": reader.read_struct("f"),
        "fy": reader.read_struct("f"),
        "cx": reader.read_struct("f"),
        "cy": reader.read_struct("f"),
    }
    request["patch_depth_bytes"] = reader.read_bytes(PATCH_VALUES * 4)
    request["valid_patches"] = reader.read_struct("i")
    request["projected_points"] = reader.read_struct("i")
    request["map_version"] = reader.read_struct("Q")
    request["T_world_camera"] = reader.read_struct("16f")
    reader.require_end()
    return request


def empty_response(request: dict, ok: bool, error: str = "") -> dict:
    return {
        "time_ns": int(request.get("time_ns", 0)),
        "provenance": request.get("provenance", empty_provenance()),
        "pipeline_timing": dict(
            request.get("pipeline_timing", empty_pipeline_timing())
        ),
        "camera_id": request.get("camera_id", ""),
        "ok": ok,
        "error": error,
        "timings_ms": {
            "worker_total": 0.0,
            "preprocess": 0.0,
            "owl": 0.0,
            "robot_filter": 0.0,
            "boxernet": 0.0,
            "postprocess": 0.0,
        },
        "filtered_2d_detections": [],
        "detections": [],
    }


def choose_device(device_arg: str, torch_module) -> str:
    if device_arg != "auto":
        return device_arg
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


class InferenceRuntime:
    def __init__(self, args: argparse.Namespace):
        boxer_repo = os.path.abspath(os.path.expanduser(args.boxer_repo))
        if boxer_repo not in sys.path:
            sys.path.insert(0, boxer_repo)

        # OWLv2 and DINOv3 resolve their companion weights through
        # BOXER_CKPT_DIR at import time.  Derive it from the explicitly
        # configured BoxerNet checkpoint so the ROS-launched worker does not
        # depend on a shell-specific environment variable.
        boxer_ckpt = os.path.abspath(os.path.expanduser(args.ckpt))
        os.environ.setdefault("BOXER_CKPT_DIR", os.path.dirname(boxer_ckpt))

        import numpy as np
        import torch
        import torch.nn.functional as F

        from boxernet.boxernet import (
            BoxerNet,
            batch_dino,
            generate_plucker_encoding,
        )
        from owl.owl_wrapper import OwlWrapper
        from utils.gravity import gravity_align_T_world_cam
        from utils.taxonomy import load_text_labels
        from utils.tw.camera import CameraTW
        from utils.tw.pose import PoseTW

        self.np = np
        self.torch = torch
        self.F = F
        self.BoxerNet = BoxerNet
        self.OwlWrapper = OwlWrapper
        self.CameraTW = CameraTW
        self.PoseTW = PoseTW
        self.batch_dino = batch_dino
        self.generate_plucker_encoding = generate_plucker_encoding
        self.gravity_align_T_world_cam = gravity_align_T_world_cam

        self.device = choose_device(args.device, torch)
        self.precision = args.precision
        confidence_thresholds = load_label_confidence_thresholds(
            args.label_thresholds_file,
            {
                "owl": float(args.owl_min_confidence),
                "boxernet": float(args.boxernet_min_confidence),
            },
        )
        self.owl_confidence_thresholds = confidence_thresholds["owl"]
        self.boxernet_confidence_thresholds = confidence_thresholds["boxernet"]
        self.robot_bbox_mask_overlap = float(args.robot_bbox_mask_overlap)
        self.robot_bbox_center_overlap = float(args.robot_bbox_center_overlap)
        self.robot_mask_dilate_px = int(args.robot_mask_dilate_px)

        if args.text_prompt_file:
            self.text_labels = load_text_prompt_file(args.text_prompt_file)
        else:
            prompt_spec = args.text_prompt or ["lvisplus"]
            self.text_labels = load_text_labels(prompt_spec)
        owl_precision = None if args.precision == "auto" else args.precision
        owl_prefilter_confidence = min(
            [
                self.owl_confidence_thresholds["default"],
                *self.owl_confidence_thresholds["labels"].values(),
            ]
        )

        print(
            "roomie inference worker loading "
            f"{len(self.text_labels)} text prompts on {self.device}",
            flush=True,
        )
        self.owl = OwlWrapper(
            device=self.device,
            text_prompts=self.text_labels,
            min_confidence=owl_prefilter_confidence,
            precision=owl_precision,
            warmup=True,
            nms_iou_threshold=float(args.owl_nms_iou_threshold),
        )
        self.boxernet = BoxerNet.load_from_checkpoint(args.ckpt, device=self.device)

    def process(self, request: dict) -> dict:
        torch = self.torch
        t_total = time.perf_counter()
        response = empty_response(request, ok=True)

        t_pre = time.perf_counter()
        rgb_np = self._image_to_numpy(request["rgb"], require_rgb=True)
        mask_np = self._image_to_numpy(request["mask"], require_rgb=False)
        if rgb_np is None:
            response = empty_response(request, ok=False, error="request has no RGB image")
            response["timings_ms"]["worker_total"] = (time.perf_counter() - t_total) * 1000.0
            return response

        img_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float()[None] / 255.0
        img_torch_255 = img_tensor * 255.0
        response["timings_ms"]["preprocess"] = (time.perf_counter() - t_pre) * 1000.0

        self._sync_if_cuda()
        t0 = time.perf_counter()
        bb2d, scores2d, label_ints, _ = self.owl.forward(
            img_torch_255,
            False,
            resize_to_HW=(rgb_np.shape[0], rgb_np.shape[1]),
        )
        self._sync_if_cuda()
        t_owl_ms = (time.perf_counter() - t0) * 1000.0
        response["timings_ms"]["owl"] = t_owl_ms

        if bb2d.shape[0] == 0:
            response["timings_ms"]["worker_total"] = (time.perf_counter() - t_total) * 1000.0
            return response

        owl_thresholds = scores2d.new_tensor(
            [
                label_confidence_threshold(
                    self.owl_confidence_thresholds,
                    self.text_labels[int(label)],
                )
                for label in label_ints
            ]
        )
        keep_confidence = scores2d > owl_thresholds
        bb2d = bb2d[keep_confidence]
        scores2d = scores2d[keep_confidence]
        label_ints = label_ints[keep_confidence]
        if bb2d.shape[0] == 0:
            response["timings_ms"]["worker_total"] = (time.perf_counter() - t_total) * 1000.0
            return response

        t_filter = time.perf_counter()
        keep_robot = self._robot_bbox_keep_mask(
            bb2d,
            mask_np,
            rgb_np.shape[0],
            rgb_np.shape[1],
        )
        response["timings_ms"]["robot_filter"] = (time.perf_counter() - t_filter) * 1000.0
        bb2d = bb2d[keep_robot]
        scores2d = scores2d[keep_robot]
        label_ints = label_ints[keep_robot]
        if bb2d.shape[0] == 0:
            response["timings_ms"]["worker_total"] = (time.perf_counter() - t_total) * 1000.0
            return response

        t_post = time.perf_counter()
        labels2d = [self.text_labels[int(label)] for label in label_ints]
        response["filtered_2d_detections"] = self._format_filtered_2d(
            bb2d,
            scores2d,
            label_ints,
            labels2d,
        )
        datum = self._build_datum(request, img_tensor, bb2d)
        response["timings_ms"]["postprocess"] += (time.perf_counter() - t_post) * 1000.0

        self._sync_if_cuda()
        t1 = time.perf_counter()
        outputs = self._forward_boxernet_with_patch_depth(datum)
        self._sync_if_cuda()
        t_boxer_ms = (time.perf_counter() - t1) * 1000.0
        response["timings_ms"]["boxernet"] = t_boxer_ms

        t_post = time.perf_counter()
        obb_pr_w = outputs["obbs_pr_w"].cpu()[0]
        scores3d = obb_pr_w.prob.squeeze(-1).clone()
        boxernet_thresholds = scores3d.new_tensor(
            [
                label_confidence_threshold(
                    self.boxernet_confidence_thresholds,
                    label,
                )
                for label in labels2d
            ]
        )
        keepers = scores3d >= boxernet_thresholds

        detections = []
        for original_index in torch.nonzero(keepers, as_tuple=False).flatten().tolist():
            obb = obb_pr_w[original_index]
            label_int = int(label_ints[original_index])
            center = obb.bb3_center_world.detach().cpu().numpy().reshape(-1)
            size = obb.bb3_diagonal.detach().cpu().numpy().reshape(-1)
            yaw = obb.T_world_object.to_euler().detach().cpu().numpy().reshape(-1)[2]
            bb = bb2d[original_index].detach().cpu().numpy().reshape(-1)
            detections.append(
                {
                    "center_world": [
                        float(center[0]),
                        float(center[1]),
                        float(center[2]),
                    ],
                    "size_m": [
                        max(0.0, float(size[0])),
                        max(0.0, float(size[1])),
                        max(0.0, float(size[2])),
                    ],
                    "yaw_rad": float(yaw),
                    "score_2d": float(scores2d[original_index]),
                    "score_3d": float(scores3d[original_index]),
                    "box_xyxy": [
                        float(bb[0]),
                        float(bb[2]),
                        float(bb[1]),
                        float(bb[3]),
                    ],
                    "semantic_id": label_int,
                    "label": labels2d[original_index],
                }
            )

        response["detections"] = detections
        response["timings_ms"]["postprocess"] += (time.perf_counter() - t_post) * 1000.0
        response["timings_ms"]["worker_total"] = (time.perf_counter() - t_total) * 1000.0
        if os.environ.get("ROOMIE_INFERENCE_DEBUG", "0") == "1":
            print(
                "roomie inference "
                f"t={request['time_ns']} 2d={len(labels2d)} 3d={len(detections)} "
                f"owl={t_owl_ms:.1f}ms boxer={t_boxer_ms:.1f}ms",
                flush=True,
            )
        return response

    def _sync_if_cuda(self) -> None:
        if self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def _format_filtered_2d(self, bb2d, scores2d, label_ints, labels2d: list[str]):
        out = []
        for i in range(len(labels2d)):
            bb = bb2d[i].detach().cpu().numpy().reshape(-1)
            out.append(
                {
                    "score_2d": float(scores2d[i]),
                    "box_xyxy": [
                        float(bb[0]),
                        float(bb[2]),
                        float(bb[1]),
                        float(bb[3]),
                    ],
                    "semantic_id": int(label_ints[i]),
                    "label": labels2d[i],
                }
            )
        return out

    def _image_to_numpy(self, image: dict, require_rgb: bool):
        np = self.np
        width = int(image["width"])
        height = int(image["height"])
        channels = int(image["channels"])
        data = image["data"]
        if width <= 0 or height <= 0 or channels <= 0 or not data:
            return None
        expected = width * height * channels
        if len(data) < expected:
            raise ProtocolError("image data is shorter than width*height*channels")
        arr = np.frombuffer(data[:expected], dtype=np.uint8).copy()
        arr = arr.reshape(height, width, channels)
        if require_rgb and channels == 1:
            arr = np.repeat(arr, 3, axis=2)
        if require_rgb and channels > 3:
            arr = arr[:, :, :3]
        if not require_rgb and channels > 1:
            arr = np.max(arr, axis=2)
        if not require_rgb and channels == 1:
            arr = arr[:, :, 0]
        return arr

    def _robot_bbox_keep_mask(self, bb2d, robot_mask, img_h: int, img_w: int):
        torch = self.torch
        np = self.np
        if len(bb2d) == 0:
            return torch.zeros(0, dtype=torch.bool)
        if robot_mask is None:
            return torch.ones(len(bb2d), dtype=torch.bool)

        mask = np.asarray(robot_mask)
        if mask.ndim == 3:
            mask = np.max(mask, axis=2)
        if mask.shape[:2] != (img_h, img_w):
            import cv2

            mask = cv2.resize(
                mask.astype(np.uint8),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST,
            )

        robot = mask > 0
        if self.robot_mask_dilate_px > 0:
            import cv2

            k = self.robot_mask_dilate_px * 2 + 1
            kernel = np.ones((k, k), dtype=np.uint8)
            robot = cv2.dilate(robot.astype(np.uint8), kernel, iterations=1).astype(bool)

        keep = []
        boxes = bb2d.detach().cpu().numpy()
        for box in boxes:
            x0_f, x1_f, y0_f, y1_f = [float(v) for v in box]
            x0 = max(0, min(img_w, int(np.floor(min(x0_f, x1_f)))))
            x1 = max(0, min(img_w, int(np.ceil(max(x0_f, x1_f)))))
            y0 = max(0, min(img_h, int(np.floor(min(y0_f, y1_f)))))
            y1 = max(0, min(img_h, int(np.ceil(max(y0_f, y1_f)))))
            if x1 <= x0 or y1 <= y0:
                keep.append(False)
                continue
            crop = robot[y0:y1, x0:x1]
            overlap = float(np.count_nonzero(crop)) / float(crop.size)
            cx0 = x0 + (x1 - x0) // 4
            cx1 = x1 - (x1 - x0) // 4
            cy0 = y0 + (y1 - y0) // 4
            cy1 = y1 - (y1 - y0) // 4
            center = robot[cy0:cy1, cx0:cx1]
            center_overlap = (
                float(np.count_nonzero(center)) / float(center.size)
                if center.size > 0
                else overlap
            )
            keep.append(
                overlap < self.robot_bbox_mask_overlap
                and center_overlap < self.robot_bbox_center_overlap
            )
        return torch.tensor(keep, dtype=torch.bool)

    def _build_datum(self, request: dict, img_tensor, bb2d):
        torch = self.torch
        np = self.np
        intr = request["intrinsics"]
        cam = self.CameraTW.from_surreal(
            width=float(intr["width"]),
            height=float(intr["height"]),
            type_str="Pinhole",
            params=torch.tensor(
                [intr["fx"], intr["fy"], intr["cx"], intr["cy"]],
                dtype=torch.float32,
            ),
            valid_radius=float(np.hypot(intr["width"] * 0.5, intr["height"] * 0.5) + 1.0),
        )
        T_values = np.asarray(request["T_world_camera"], dtype=np.float32).reshape(4, 4)
        T_world_camera = self.PoseTW.from_matrix(torch.from_numpy(T_values))
        patch_depth = np.frombuffer(request["patch_depth_bytes"], dtype="<f4").copy()
        patch_depth = torch.from_numpy(patch_depth.reshape(1, 1, 60, 60)).float()
        return {
            "img0": img_tensor,
            "cam0": cam.float(),
            "T_world_rig0": T_world_camera.float(),
            "rotated0": torch.tensor(False).reshape(1),
            "bb2d": bb2d.float(),
            "patch_depth0": patch_depth,
            "time_ns0": int(request["time_ns"]),
        }

    def _prepare_inputs_with_patch_depth(self, datum: dict):
        torch = self.torch
        inputs = {}
        inputs.update(self.boxernet.process_camera(datum))
        bb2d = datum["bb2d"]
        if bb2d.ndim == 2:
            bb2d = bb2d.unsqueeze(0)
        inputs["bb2d"] = bb2d
        patch_depth = datum["patch_depth0"]
        if patch_depth.ndim == 2:
            patch_depth = patch_depth.reshape(
                1,
                1,
                patch_depth.shape[0],
                patch_depth.shape[1],
            )
        elif patch_depth.ndim == 3:
            patch_depth = patch_depth.unsqueeze(0)
        inputs["sdp_patch0"] = patch_depth
        for key, value in list(inputs.items()):
            if "rotated" in key:
                inputs[key] = value.to(self.device).bool()
            else:
                inputs[key] = value.to(self.device).float()
        if inputs["sdp_patch0"].ndim != 4:
            raise RuntimeError("patch_depth0 must have shape Bx1xHxW")
        return inputs

    def _encode_with_patch_depth(self, batch: dict):
        torch = self.torch
        out = {}
        torch_img = batch["img0"]
        cam = batch["cam0"]
        T_wr = batch["T_world_rig0"]
        T_wc = T_wr @ cam.T_camera_rig.inverse()
        if "T_world_voxel0" not in batch:
            T_wv = self.gravity_align_T_world_cam(T_wc, z_grav=True)
            rpy = T_wv.to_euler()
            if not torch.allclose(
                torch.zeros(1, device=rpy.device),
                rpy[:, :2],
                atol=1e-4,
            ):
                raise RuntimeError("gravity-aligned voxel frame has unexpected roll/pitch")
            batch["T_world_voxel0"] = T_wv
        T_wv = batch["T_world_voxel0"]
        rotated = batch["rotated0"]
        B = torch_img.shape[0]

        with torch.no_grad():
            dino_feat = self.batch_dino(self.boxernet.dino, torch_img, rotated)
            out["dino0"] = dino_feat.clone()
            _, _, fH, fW = dino_feat.shape
            dino_feat = dino_feat.reshape(B, -1, fH * fW).permute(0, 2, 1)

            sdp_median = batch["sdp_patch0"]
            if sdp_median.shape[-2:] != (fH, fW):
                sdp_median = self.F.interpolate(
                    sdp_median,
                    size=(fH, fW),
                    mode="nearest",
                )
            out["sdp_patch0"] = sdp_median.clone()
            sdp_input = sdp_median.reshape(B, -1, fH * fW).permute(0, 2, 1)
            x = torch.cat([dino_feat, sdp_input], dim=-1)

            if self.boxernet.with_ray:
                T_vc = T_wv.inverse() @ T_wc
                ray_enc = self.generate_plucker_encoding(
                    B,
                    fH,
                    fW,
                    self.boxernet.dino.patch_size,
                    cam,
                    T_vc,
                )
                out["ray_enc0"] = ray_enc.clone()
                x = torch.cat([x, ray_enc], dim=-1)

        input_enc = self.boxernet.input2emb(x)
        if self.boxernet.in_depth > 0:
            input_enc = self.boxernet.self_attn(input_enc)
        out["input_enc"] = input_enc
        return out

    def _forward_boxernet_with_patch_depth(self, datum: dict):
        torch = self.torch
        inputs = self._prepare_inputs_with_patch_depth(datum)

        def run_forward():
            out = self._encode_with_patch_depth(inputs)
            return self.boxernet.query(inputs, out)

        if self.device == "cuda":
            if self.precision == "bfloat16" or (
                self.precision == "auto" and torch.cuda.is_bf16_supported()
            ):
                with torch.no_grad(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    return run_forward()
        with torch.no_grad():
            return run_forward()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxer-repo", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto", choices=["auto", "float32", "bfloat16"])
    parser.add_argument("--text-prompt-file", default="")
    parser.add_argument("--text-prompt", action="append", default=[])
    parser.add_argument("--owl-min-confidence", type=float, default=0.25)
    parser.add_argument("--owl-nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--boxernet-min-confidence", type=float, default=0.5)
    parser.add_argument("--label-thresholds-file", default="")
    parser.add_argument("--robot-bbox-mask-overlap", type=float, default=0.25)
    parser.add_argument("--robot-bbox-center-overlap", type=float, default=0.5)
    parser.add_argument("--robot-mask-dilate-px", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    # The owning C++ actor coordinates pipe closure and bounded shutdown. A
    # launcher commonly sends SIGINT to the whole process group; do not let
    # that interrupt a framed read and corrupt the protocol during teardown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    configure_binary_stdout()
    args = parse_args()
    runtime = None
    init_error = ""
    try:
        runtime = InferenceRuntime(args)
    except Exception:
        init_error = traceback.format_exc(limit=20)
        print(init_error, flush=True)

    while True:
        body = read_message()
        if body is None:
            return 0
        try:
            request = parse_request(body)
            if runtime is None:
                response = empty_response(request, ok=False, error=init_error)
            else:
                response = runtime.process(request)
        except Exception:
            fallback = {
                "time_ns": 0,
                "provenance": empty_provenance(),
                "pipeline_timing": empty_pipeline_timing(),
                "camera_id": "",
            }
            try:
                fallback = parse_request(body)
            except Exception:
                pass
            response = empty_response(
                fallback,
                ok=False,
                error=traceback.format_exc(limit=20),
            )
        ResponseWriter().write_response(response)


if __name__ == "__main__":
    raise SystemExit(main())
