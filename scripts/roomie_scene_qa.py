#!/usr/bin/env python3
"""Ask Gemini or Doubao questions about Roomie's live scene query service."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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
    RosCameraViewSampler,
    RosLatest2dDetectionsSampler,
    RosQuerySceneTransport,
    SceneQaConfig,
    StructuredTaskAgent,
    TASK_FIND_OBJECT_IN_VIEW,
    TASK_NAMES,
    TASK_NAVIGATION,
    TASK_SCENE_QA,
    TaskEvidence,
    create_default_tool_registry,
    create_find_object_in_view_task_registry,
    create_live_tool_registry,
    create_navigation_task_registry,
)
from scene_qa.config import DEFAULT_EMBEDDING_MODEL  # noqa: E402
from scene_qa.config import default_qa_config_path  # noqa: E402
from scene_qa.embeddings import ObjectSearchIndex  # noqa: E402
from scene_qa.object_references import (  # noqa: E402
    try_load_object_reference_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=TASK_NAMES,
        default=TASK_SCENE_QA,
        help="task profile and bounded tool set (default: scene_qa)",
    )
    parser.add_argument(
        "--no-follow-up-question",
        action="store_true",
        help="navigation: choose the best destination instead of asking for confirmation",
    )
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
    parser.add_argument("--in-view-image-topic", default=None)
    parser.add_argument("--in-view-camera-info-topic", default=None)
    parser.add_argument("--in-view-detection-image-topic", default=None)
    parser.add_argument("--in-view-detection-result-topic", default=None)
    parser.add_argument("--in-view-world-frame", default=None)
    parser.add_argument("--in-view-camera-frame", default=None)
    parser.add_argument("--in-view-min-depth-m", type=float, default=None)
    parser.add_argument("--in-view-max-depth-m", type=float, default=None)
    parser.add_argument("--in-view-sensor-timeout-s", type=float, default=None)
    parser.add_argument("--in-view-tf-tolerance-s", type=float, default=None)
    parser.add_argument(
        "--object-reference-root",
        type=Path,
        default=None,
        help="local curated object-reference root; overrides QA config",
    )
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
        doubao_service_tier=config.doubao_service_tier,
        embedding_model=args.embedding_model or config.embedding_model,
        embedding_backend=args.embedding_backend or config.embedding_backend,
        device=args.device or config.device,
        top_k=args.top_k if args.top_k is not None else config.top_k,
        max_iterations=(
            args.max_iterations
            if args.max_iterations is not None
            else (
                2
                if args.task == TASK_FIND_OBJECT_IN_VIEW
                else (3 if args.task == TASK_NAVIGATION else config.max_iterations)
            )
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
        object_reference_root=(
            args.object_reference_root
            if args.object_reference_root is not None
            else config.object_reference_root
        ),
        object_reference_max_scene_objects=config.object_reference_max_scene_objects,
        system_prompt_path=config.system_prompt_path_for_task(args.task),
        navigation_system_prompt_path=config.navigation_system_prompt_path,
        find_object_in_view_system_prompt_path=(
            config.find_object_in_view_system_prompt_path
        ),
        in_view_image_topic=(
            args.in_view_image_topic or config.in_view_image_topic
        ),
        in_view_camera_info_topic=(
            args.in_view_camera_info_topic or config.in_view_camera_info_topic
        ),
        in_view_detection_image_topic=(
            args.in_view_detection_image_topic
            or config.in_view_detection_image_topic
        ),
        in_view_detection_result_topic=(
            args.in_view_detection_result_topic
            or config.in_view_detection_result_topic
        ),
        in_view_world_frame=(
            args.in_view_world_frame or config.in_view_world_frame
        ),
        in_view_camera_frame=(
            args.in_view_camera_frame
            if args.in_view_camera_frame is not None
            else config.in_view_camera_frame
        ),
        in_view_min_depth_m=(
            args.in_view_min_depth_m
            if args.in_view_min_depth_m is not None
            else config.in_view_min_depth_m
        ),
        in_view_max_depth_m=(
            args.in_view_max_depth_m
            if args.in_view_max_depth_m is not None
            else config.in_view_max_depth_m
        ),
        in_view_sensor_timeout_s=(
            args.in_view_sensor_timeout_s
            if args.in_view_sensor_timeout_s is not None
            else config.in_view_sensor_timeout_s
        ),
        in_view_tf_tolerance_s=(
            args.in_view_tf_tolerance_s
            if args.in_view_tf_tolerance_s is not None
            else config.in_view_tf_tolerance_s
        ),
        _config_dir=config._config_dir,
    )


@dataclass
class RuntimeResources:
    values: list[Any] = field(default_factory=list)

    def add(self, value: Any) -> Any:
        self.values.append(value)
        return value

    def close(self) -> None:
        for value in reversed(self.values):
            close = getattr(value, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def build_agent(
    args: argparse.Namespace,
    config: SceneQaConfig,
) -> tuple[Any, RuntimeResources]:
    if args.task == TASK_FIND_OBJECT_IN_VIEW and args.offline_json is not None:
        raise ValueError("find_object_in_view requires the live scene and current ROS camera")
    resources = RuntimeResources()
    reference_catalog_load = try_load_object_reference_catalog(
        config.resolved_object_reference_root()
    )
    print(
        f"Object reference catalog: {reference_catalog_load.status}",
        file=sys.stderr,
    )
    if args.offline_json is not None:
        graph = GraphStore.load(config.resolved_graph_json())
        search_index = ObjectSearchIndex(
            graph,
            model_path=config.resolved_embedding_model(),
            backend=config.embedding_backend,
            device=config.device,
        )
        base_registry = create_default_tool_registry(
            graph, search_index, config, reference_catalog_load
        )
    else:
        graph = None
        transport = resources.add(
            RosQuerySceneTransport(
                service_name=args.service,
                timeout_sec=args.service_timeout_sec,
            )
        )
        try:
            live_client = LiveSceneQueryClient(
                transport,
                session_ttl_ms=args.session_ttl_ms,
            )
            base_registry = create_live_tool_registry(
                live_client, config, reference_catalog_load
            )
        except Exception:
            resources.close()
            raise
    try:
        evidence = None
        if args.task == TASK_NAVIGATION:
            evidence = TaskEvidence(args.task)
            registry = create_navigation_task_registry(
                base_registry, config, evidence, reference_catalog_load
            )
        elif args.task == TASK_FIND_OBJECT_IN_VIEW:
            evidence = TaskEvidence(args.task)
            camera_sampler = resources.add(
                RosCameraViewSampler(
                    image_topic=config.in_view_image_topic,
                    camera_info_topic=config.in_view_camera_info_topic,
                    world_frame=config.in_view_world_frame,
                    camera_frame=config.in_view_camera_frame,
                    timeout_sec=config.in_view_sensor_timeout_s,
                    tf_tolerance_sec=config.in_view_tf_tolerance_s,
                )
            )
            detection_sampler = resources.add(
                RosLatest2dDetectionsSampler(
                    image_base_topic=config.in_view_detection_image_topic,
                    result_base_topic=config.in_view_detection_result_topic,
                    timeout_sec=config.in_view_sensor_timeout_s,
                )
            )
            registry = create_find_object_in_view_task_registry(
                base_registry,
                config,
                evidence,
                camera_sampler,
                reference_catalog_load,
                detection_sampler=detection_sampler,
            )
        else:
            registry = base_registry
    except Exception:
        resources.close()
        raise
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
        if evidence is not None:
            agent = StructuredTaskAgent(
                agent,
                evidence,
                task=args.task,
                allow_follow_up_question=not args.no_follow_up_question,
                world_frame=config.in_view_world_frame,
            )
    except Exception:
        resources.close()
        raise
    return agent, resources


def response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response.answer, dict) and response.history.get("task") in {
        TASK_NAVIGATION,
        TASK_FIND_OBJECT_IN_VIEW,
    }:
        return response.answer
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
    agent, resources = build_agent(args, config)

    try:
        return run_query_loop(
            agent,
            initial_query=query,
            once=args.once,
            dump_history=args.dump_history,
        )
    finally:
        resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
