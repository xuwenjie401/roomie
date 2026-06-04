"""Configuration for Roomie scene QA."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


DEFAULT_DESCRIBED_JSON = Path(
    "/home/agxi/Datasets/output/jarvis_home/instances/latest.dam_described.json"
)
DEFAULT_EMBEDDING_MODEL = Path("/home/agxi/huggingface/sentence_t5_large")
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_QA_CONFIG_NAME = "config.json"
DEFAULT_SYSTEM_PROMPT_PATH = Path("prompts/system.txt")
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


@dataclass
class SceneQaConfig:
    """Runtime settings for the scene QA agent."""

    graph_json: Path = DEFAULT_DESCRIBED_JSON
    gemini_model: str = DEFAULT_GEMINI_MODEL
    embedding_model: Path = DEFAULT_EMBEDDING_MODEL
    embedding_backend: str = "embedding"
    device: str = "auto"
    top_k: int = 12
    max_iterations: int = 10
    temperature: float = 0.2
    max_output_tokens: int = 4096
    snapshot_max_side_px: int = 1024
    snapshot_bbox_pad_px: int = 8
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT_PATH
    _config_dir: Path | None = field(default=None, repr=False, compare=False)

    def resolved_graph_json(self) -> Path:
        return self.graph_json.expanduser().resolve()

    def resolved_embedding_model(self) -> Path:
        return self.embedding_model.expanduser().resolve()

    def resolved_system_prompt_path(self) -> Path:
        path = self.system_prompt_path.expanduser()
        if path.is_absolute():
            return path.resolve()
        base = self._config_dir or default_qa_config_path().parent
        return (base / path).resolve()

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
            system_prompt_path=_path_or_default(
                data.get("system_prompt_path"), defaults.system_prompt_path
            ),
            _config_dir=config_dir,
        )
