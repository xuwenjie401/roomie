#!/usr/bin/env python3
"""Ask Gemini or Doubao questions about Roomie's live scene query service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, TextIO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scene_qa import (  # noqa: E402
    DoubaoSceneQaAgent,
    GeminiSceneQaAgent,
    GraphStore,
    LiveSceneQueryClient,
    RosQuerySceneTransport,
    SceneQaConfig,
    create_default_tool_registry,
    create_live_tool_registry,
)
from scene_qa.config import DEFAULT_EMBEDDING_MODEL  # noqa: E402
from scene_qa.config import default_qa_config_path  # noqa: E402
from scene_qa.embeddings import ObjectSearchIndex  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        nargs="*",
        help=(
            "optional first question; the default mode keeps prompting until "
            "/exit, /quit, Ctrl-D, or Ctrl-C"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="answer the positional question once and exit (for scripts)",
    )
    parser.add_argument(
        "--qa-config",
        type=Path,
        default=None,
        help=f"scene QA JSON config (default: {default_qa_config_path()})",
    )
    parser.add_argument(
        "--offline-json",
        type=Path,
        default=None,
        help=(
            "explicitly use a static Roomie DSG JSON instead of the live service; "
            "there is no implicit JSON fallback"
        ),
    )
    parser.add_argument(
        "--service",
        default="/roomie/query_scene",
        help="live QueryScene service (default: /roomie/query_scene)",
    )
    parser.add_argument(
        "--session-ttl-ms",
        type=int,
        default=30_000,
        help="fixed server read-session TTL for one model answer (default: 30000)",
    )
    parser.add_argument(
        "--service-timeout-sec",
        type=float,
        default=10.0,
        help="ROS service discovery/call timeout (default: 10)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "doubao"],
        default="gemini",
        help="model provider (default: gemini)",
    )
    parser.add_argument("--model", default=None, help="Gemini model; overrides QA config")
    parser.add_argument(
        "--doubao-model", default=None, help="Doubao model; overrides QA config"
    )
    parser.add_argument(
        "--doubao-base-url",
        default=None,
        help="Ark API v3 base URL; overrides QA config",
    )
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
        help="optional path to write the model/tool interaction history JSON",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key; defaults to GEMINI_API_KEY or GOOGLE_API_KEY",
    )
    parser.add_argument(
        "--doubao-api-key",
        default=None,
        help="Doubao key; defaults to DOUBAO_API_KEY or ARK_API_KEY",
    )
    return parser.parse_args()


def load_qa_config(path_arg: Path | None) -> SceneQaConfig:
    path = path_arg.expanduser().resolve() if path_arg is not None else default_qa_config_path()
    if path is not None and path.exists():
        return SceneQaConfig.from_json(path)
    return SceneQaConfig()


def apply_cli_overrides(config: SceneQaConfig, args: argparse.Namespace) -> SceneQaConfig:
    graph_json = (
        args.offline_json.expanduser().resolve()
        if args.offline_json is not None
        else config.graph_json
    )
    return SceneQaConfig(
        graph_json=graph_json,
        gemini_model=args.model or config.gemini_model,
        doubao_model=args.doubao_model or config.doubao_model,
        doubao_base_url=(args.doubao_base_url or config.doubao_base_url).rstrip("/"),
        doubao_thinking_type=config.doubao_thinking_type,
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
        system_prompt_path=config.system_prompt_path,
        _config_dir=config._config_dir,
    )


def build_agent(
    args: argparse.Namespace,
    config: SceneQaConfig,
) -> tuple[Any, RosQuerySceneTransport | None]:
    transport = None
    if args.offline_json is not None:
        graph = GraphStore.load(config.resolved_graph_json())
        search_index = ObjectSearchIndex(
            graph,
            model_path=config.resolved_embedding_model(),
            backend=config.embedding_backend,
            device=config.device,
        )
        registry = create_default_tool_registry(graph, search_index, config)
    else:
        graph = None
        transport = RosQuerySceneTransport(
            service_name=args.service,
            timeout_sec=args.service_timeout_sec,
        )
        live_client = LiveSceneQueryClient(
            transport,
            session_ttl_ms=args.session_ttl_ms,
        )
        registry = create_live_tool_registry(live_client, config)
    try:
        if args.provider == "doubao":
            agent = DoubaoSceneQaAgent(
                graph,
                registry,
                config,
                api_key=args.doubao_api_key,
            )
        else:
            agent = GeminiSceneQaAgent(
                graph,
                registry,
                config,
                api_key=args.api_key,
            )
    except Exception:
        if transport is not None:
            transport.close()
        raise
    return agent, transport


def response_payload(response: Any) -> dict[str, Any]:
    return {"reasoning": response.reasoning, "answer": response.answer}


def write_history(path: Path, histories: list[dict[str, Any]], *, once: bool) -> None:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] | list[dict[str, Any]]
    payload = histories[0] if once and len(histories) == 1 else histories
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_query_loop(
    agent: Any,
    *,
    initial_query: str = "",
    once: bool = False,
    dump_history: Path | None = None,
    input_fn: Callable[[str], str] = input,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one scripted query or a persistent, stateless interactive prompt."""

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    pending_query = initial_query.strip()
    if once and not pending_query:
        print("--once requires a non-empty positional query", file=stderr)
        return 2

    if not once:
        print(
            "Roomie Scene QA interactive mode. "
            "Type /exit or /quit to leave; each question reads the latest scene.",
            file=stdout,
        )

    histories: list[dict[str, Any]] = []
    while True:
        if pending_query:
            query = pending_query
            pending_query = ""
        elif once:
            break
        else:
            try:
                query = input_fn("roomie> ").strip()
            except EOFError:
                print(file=stdout)
                break
            except KeyboardInterrupt:
                print("\nInterrupted.", file=stdout)
                break
            if query in {"/exit", "/quit", ":q"}:
                break
            if not query:
                continue

        try:
            response = agent.answer_query(query)
        except KeyboardInterrupt:
            print("\nQuestion interrupted.", file=stderr)
            return 130
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=stderr)
            if once:
                return 1
            continue

        print(
            json.dumps(response_payload(response), ensure_ascii=False, indent=2),
            file=stdout,
        )
        histories.append(response.history)
        if dump_history is not None:
            write_history(dump_history, histories, once=once)
        if once:
            break
    return 0


def main() -> int:
    args = parse_args()
    query = " ".join(args.query).strip()
    if args.once and not query:
        print("--once requires a non-empty positional query", file=sys.stderr)
        return 2

    config = apply_cli_overrides(load_qa_config(args.qa_config), args)
    agent, transport = build_agent(args, config)

    try:
        return run_query_loop(
            agent,
            initial_query=query,
            once=args.once,
            dump_history=args.dump_history,
        )
    finally:
        if transport is not None:
            transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
