#!/usr/bin/env python3
"""Ask Gemini questions about a Roomie scene graph JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from roomie_dsg_viewer import (  # noqa: E402
    DEFAULT_CONFIG,
    as_path,
    load_pipeline_params,
    resolve_json_from_config,
)
from scene_qa import (  # noqa: E402
    GeminiSceneQaAgent,
    GraphStore,
    SceneQaConfig,
    create_default_tool_registry,
)
from scene_qa.config import DEFAULT_DESCRIBED_JSON, DEFAULT_EMBEDDING_MODEL  # noqa: E402
from scene_qa.config import default_qa_config_path  # noqa: E402
from scene_qa.embeddings import ObjectSearchIndex  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="question to ask about the scene")
    parser.add_argument(
        "--qa-config",
        type=Path,
        default=None,
        help=f"scene QA JSON config (default: {default_qa_config_path()})",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Roomie DSG JSON path; overrides the QA JSON config",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="pipeline YAML used to resolve the DSG path if --json is omitted",
    )
    parser.add_argument("--model", default=None, help="Gemini model; overrides QA config")
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=None,
        help=f"local SentenceTransformer checkpoint; default from QA config ({DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["embedding", "lexical"],
        default=None,
        help="semantic backend; overrides QA config",
    )
    parser.add_argument("--device", default=None, help="embedding device: auto, cuda, or cpu")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--snapshot-max-side-px", type=int, default=None)
    parser.add_argument("--snapshot-bbox-pad-px", type=int, default=None)
    parser.add_argument(
        "--dump-history",
        type=Path,
        default=None,
        help="optional path to write the Gemini/tool interaction history JSON",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key; defaults to GEMINI_API_KEY, GOOGLE_API_KEY, or SDK default auth",
    )
    return parser.parse_args()


def load_qa_config(path_arg: Path | None) -> SceneQaConfig:
    path = as_path(path_arg) if path_arg is not None else default_qa_config_path()
    if path is not None and path.exists():
        return SceneQaConfig.from_json(path)
    return SceneQaConfig()


def resolve_graph_json(args: argparse.Namespace, config: SceneQaConfig) -> Path:
    explicit = as_path(args.json)
    if explicit is not None:
        return explicit
    configured = config.graph_json.expanduser()
    if str(configured):
        return configured.resolve()
    params = load_pipeline_params(as_path(args.config))
    resolved = resolve_json_from_config(params)
    if resolved is not None and resolved.exists():
        described = resolved.with_name(f"{resolved.stem}.dam_described{resolved.suffix}")
        if described.exists():
            return described
        return resolved
    return DEFAULT_DESCRIBED_JSON.expanduser().resolve()


def apply_cli_overrides(config: SceneQaConfig, args: argparse.Namespace) -> SceneQaConfig:
    return SceneQaConfig(
        graph_json=resolve_graph_json(args, config),
        gemini_model=args.model or config.gemini_model,
        embedding_model=args.embedding_model or config.embedding_model,
        embedding_backend=args.embedding_backend or config.embedding_backend,
        device=args.device or config.device,
        top_k=args.top_k if args.top_k is not None else config.top_k,
        max_iterations=(
            args.max_iterations if args.max_iterations is not None else config.max_iterations
        ),
        temperature=args.temperature if args.temperature is not None else config.temperature,
        max_output_tokens=(
            args.max_output_tokens
            if args.max_output_tokens is not None
            else config.max_output_tokens
        ),
        snapshot_max_side_px=(
            args.snapshot_max_side_px
            if args.snapshot_max_side_px is not None
            else config.snapshot_max_side_px
        ),
        snapshot_bbox_pad_px=(
            args.snapshot_bbox_pad_px
            if args.snapshot_bbox_pad_px is not None
            else config.snapshot_bbox_pad_px
        ),
    )


def main() -> int:
    args = parse_args()
    query = " ".join(args.query).strip()
    if not query:
        print("query must not be empty", file=sys.stderr)
        return 2

    config = apply_cli_overrides(load_qa_config(args.qa_config), args)

    graph = GraphStore.load(config.resolved_graph_json())
    search_index = ObjectSearchIndex(
        graph,
        model_path=config.resolved_embedding_model(),
        backend=config.embedding_backend,
        device=config.device,
    )
    registry = create_default_tool_registry(graph, search_index, config)
    agent = GeminiSceneQaAgent(graph, registry, config, api_key=args.api_key)

    response = agent.answer_query(query)
    print(json.dumps({"reasoning": response.reasoning, "answer": response.answer}, ensure_ascii=False, indent=2))

    if args.dump_history is not None:
        output_path = args.dump_history.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(response.history, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
