"""Configuration for Roomie scene QA."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .object_references import default_object_reference_root


DEFAULT_DESCRIBED_JSON = Path(
    "/home/lindenbot/Datasets/output/jarvis_home/instances/latest.dam_described.json"
)
DEFAULT_EMBEDDING_MODEL = Path("/home/lindenbot/hugging_face/sentence_t5_large")
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_DOUBAO_MODEL = "doubao-seed-2-0-lite-260215"
DEFAULT_DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_DOUBAO_THINKING_TYPE = "disabled"
DOUBAO_THINKING_TYPES = frozenset({"enabled", "disabled", "auto"})
DEFAULT_DOUBAO_SERVICE_TIER = "default"
DOUBAO_SERVICE_TIERS = frozenset({"default", "fast"})
DEFAULT_QA_CONFIG_NAME = "config.json"
DEFAULT_SYSTEM_PROMPT_PATH = Path("prompts/system.txt")
DEFAULT_NAVIGATION_SYSTEM_PROMPT_PATH = Path("prompts/navigation.txt")
DEFAULT_FIND_OBJECT_IN_VIEW_SYSTEM_PROMPT_PATH = Path(
    "prompts/find_object_in_view.txt"
)
FALLBACK_SYSTEM_PROMPT = """You are a scene understanding assistant for a Roomie 3D scene graph.

Use the provided tools to inspect the scene. Answer in the same language as the user.
Return concise JSON with "reasoning" and "answer" fields.
"""


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def candidate_qa_config_paths() -> list[Path]:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "config" / "scene_qa" / DEFAULT_QA_CONFIG_NAME,
    ]
    if len(here.parents) > 3:
        candidates.append(
            here.parents[3] / "share" / "roomie" / "config" / "scene_qa" / DEFAULT_QA_CONFIG_NAME
        )
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(
            Path(get_package_share_directory("roomie"))
            / "config"
            / "scene_qa"
            / DEFAULT_QA_CONFIG_NAME
        )
    except Exception:
        pass
    return candidates


def default_qa_config_path() -> Path:
    candidates = candidate_qa_config_paths()
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _path_or_default(value: Any, default: Path) -> Path:
    if value is None or value == "":
        return default
    return Path(value).expanduser()


def _int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _doubao_thinking_type_or_default(value: Any, default: str) -> str:
    thinking_type = str(value or default).strip().lower()
    if thinking_type not in DOUBAO_THINKING_TYPES:
        choices = ", ".join(sorted(DOUBAO_THINKING_TYPES))
        raise ValueError(f"doubao_thinking_type must be one of: {choices}")
    return thinking_type


def _doubao_service_tier_or_default(value: Any, default: str) -> str:
    service_tier = str(value or default).strip().lower()
    if service_tier not in DOUBAO_SERVICE_TIERS:
        choices = ", ".join(sorted(DOUBAO_SERVICE_TIERS))
        raise ValueError(f"doubao_service_tier must be one of: {choices}")
    return service_tier


@dataclass
class SceneQaConfig:
    """Runtime settings for the scene QA agent."""

    graph_json: Path = DEFAULT_DESCRIBED_JSON
    gemini_model: str = DEFAULT_GEMINI_MODEL
    doubao_model: str = DEFAULT_DOUBAO_MODEL
    doubao_base_url: str = DEFAULT_DOUBAO_BASE_URL
    doubao_thinking_type: str = DEFAULT_DOUBAO_THINKING_TYPE
    doubao_service_tier: str = DEFAULT_DOUBAO_SERVICE_TIER
    embedding_model: Path = DEFAULT_EMBEDDING_MODEL
    embedding_backend: str = "embedding"
    device: str = "auto"
    top_k: int = 12
    max_iterations: int = 10
    temperature: float = 0.2
    max_output_tokens: int = 4096
    snapshot_max_side_px: int = 1024
    snapshot_bbox_pad_px: int = 8
    object_reference_root: Path = field(default_factory=default_object_reference_root)
    object_reference_max_scene_objects: int = 4
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT_PATH
    navigation_system_prompt_path: Path = DEFAULT_NAVIGATION_SYSTEM_PROMPT_PATH
    find_object_in_view_system_prompt_path: Path = (
        DEFAULT_FIND_OBJECT_IN_VIEW_SYSTEM_PROMPT_PATH
    )
    in_view_image_topic: str = "/roomie/input/head_color/image_rect"
    in_view_camera_info_topic: str = "/roomie/input/head_color/camera_info"
    in_view_detection_image_topic: str = "/roomie/detections_2d_image"
    in_view_detection_result_topic: str = "/roomie/detections_2d"
    in_view_world_frame: str = "map"
    in_view_camera_frame: str = "head_color"
    in_view_min_depth_m: float = 0.1
    in_view_max_depth_m: float = 6.0
    in_view_sensor_timeout_s: float = 1.0
    in_view_tf_tolerance_s: float = 0.2
    _config_dir: Path | None = field(default=None, repr=False, compare=False)

    def resolved_graph_json(self) -> Path:
        return self.graph_json.expanduser().resolve()

    def resolved_embedding_model(self) -> Path:
        return self.embedding_model.expanduser().resolve()

    def resolved_object_reference_root(self) -> Path:
        path = self.object_reference_root.expanduser()
        if path.is_absolute():
            return path.resolve()
        base = self._config_dir or default_qa_config_path().parent
        return (base / path).resolve()

    def resolved_system_prompt_path(self) -> Path:
        path = self.system_prompt_path.expanduser()
        if path.is_absolute():
            return path.resolve()
        base = self._config_dir or default_qa_config_path().parent
        return (base / path).resolve()

    def system_prompt_path_for_task(self, task: str) -> Path:
        if task == "navigation":
            return self.navigation_system_prompt_path
        if task == "find_object_in_view":
            return self.find_object_in_view_system_prompt_path
        return self.system_prompt_path

    def load_system_prompt(self) -> str:
        path = self.resolved_system_prompt_path()
        if not path.exists():
            return FALLBACK_SYSTEM_PROMPT
        return path.read_text(encoding="utf-8").strip()

    @classmethod
    def from_json(cls, path: str | Path) -> "SceneQaConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError(f"scene QA config must be a JSON object: {config_path}")
        return cls.from_mapping(data, config_dir=config_path.parent)

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        config_dir: Path | None = None,
    ) -> "SceneQaConfig":
        defaults = cls()
        return cls(
            graph_json=_path_or_default(data.get("graph_json"), defaults.graph_json),
            gemini_model=str(data.get("gemini_model") or defaults.gemini_model),
            doubao_model=str(data.get("doubao_model") or defaults.doubao_model),
            doubao_base_url=str(
                data.get("doubao_base_url") or defaults.doubao_base_url
            ).rstrip("/"),
            doubao_thinking_type=_doubao_thinking_type_or_default(
                data.get("doubao_thinking_type"), defaults.doubao_thinking_type
            ),
            doubao_service_tier=_doubao_service_tier_or_default(
                data.get("doubao_service_tier"), defaults.doubao_service_tier
            ),
            embedding_model=_path_or_default(
                data.get("embedding_model"), defaults.embedding_model
            ),
            embedding_backend=str(
                data.get("embedding_backend") or defaults.embedding_backend
            ),
            device=str(data.get("device") or defaults.device),
            top_k=_int_or_default(data.get("top_k"), defaults.top_k),
            max_iterations=_int_or_default(
                data.get("max_iterations"), defaults.max_iterations
            ),
            temperature=_float_or_default(data.get("temperature"), defaults.temperature),
            max_output_tokens=_int_or_default(
                data.get("max_output_tokens"), defaults.max_output_tokens
            ),
            snapshot_max_side_px=_int_or_default(
                data.get("snapshot_max_side_px"), defaults.snapshot_max_side_px
            ),
            snapshot_bbox_pad_px=_int_or_default(
                data.get("snapshot_bbox_pad_px"), defaults.snapshot_bbox_pad_px
            ),
            object_reference_root=_path_or_default(
                data.get("object_reference_root"), defaults.object_reference_root
            ),
            object_reference_max_scene_objects=_int_or_default(
                data.get("object_reference_max_scene_objects"),
                defaults.object_reference_max_scene_objects,
            ),
            system_prompt_path=_path_or_default(
                data.get("system_prompt_path"), defaults.system_prompt_path
            ),
            navigation_system_prompt_path=_path_or_default(
                data.get("navigation_system_prompt_path"),
                defaults.navigation_system_prompt_path,
            ),
            find_object_in_view_system_prompt_path=_path_or_default(
                data.get("find_object_in_view_system_prompt_path"),
                defaults.find_object_in_view_system_prompt_path,
            ),
            in_view_image_topic=str(
                data.get("in_view_image_topic") or defaults.in_view_image_topic
            ),
            in_view_camera_info_topic=str(
                data.get("in_view_camera_info_topic")
                or defaults.in_view_camera_info_topic
            ),
            in_view_detection_image_topic=str(
                data.get("in_view_detection_image_topic")
                or defaults.in_view_detection_image_topic
            ),
            in_view_detection_result_topic=str(
                data.get("in_view_detection_result_topic")
                or defaults.in_view_detection_result_topic
            ),
            in_view_world_frame=str(
                data.get("in_view_world_frame") or defaults.in_view_world_frame
            ),
            in_view_camera_frame=str(
                data.get("in_view_camera_frame") or defaults.in_view_camera_frame
            ),
            in_view_min_depth_m=_float_or_default(
                data.get("in_view_min_depth_m"), defaults.in_view_min_depth_m
            ),
            in_view_max_depth_m=_float_or_default(
                data.get("in_view_max_depth_m"), defaults.in_view_max_depth_m
            ),
            in_view_sensor_timeout_s=_float_or_default(
                data.get("in_view_sensor_timeout_s"),
                defaults.in_view_sensor_timeout_s,
            ),
            in_view_tf_tolerance_s=_float_or_default(
                data.get("in_view_tf_tolerance_s"),
                defaults.in_view_tf_tolerance_s,
            ),
            _config_dir=config_dir,
        )
