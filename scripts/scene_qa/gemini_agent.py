"""Gemini tool-calling loop for Roomie scene QA."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
import threading
from typing import Any, Callable

from .config import SceneQaConfig
from .graph_store import GraphStore
from .tools import ToolRegistry, ToolResult


PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


def sanitize_proxy_environment() -> None:
    """Normalize proxy env vars for httpx/google-genai.

    Some shell configs use ``socks://``. httpx only accepts explicit SOCKS schemes
    when socks support is installed, and this jarvis env currently does not have
    socksio. Local proxy tools such as Clash commonly expose the same mixed port
    over HTTP, so fall back to http:// for http(s) proxy vars and drop all_proxy.
    """
    socksio_available = importlib.util.find_spec("socksio") is not None
    for name in PROXY_ENV_NAMES:
        value = os.environ.get(name)
        if not value or not value.lower().startswith("socks://"):
            continue
        suffix = value.split("://", 1)[1]
        if socksio_available:
            os.environ[name] = f"socks5://{suffix}"
        elif name.lower() == "all_proxy":
            os.environ.pop(name, None)
        else:
            os.environ[name] = f"http://{suffix}"


@dataclass
class QaResponse:
    reasoning: str
    answer: Any
    raw_text: str
    history: dict[str, Any]


class SceneQaCancelledError(RuntimeError):
    """Raised when the browser resets an in-flight question."""


class _CompatValue:
    """Small google.genai.types-shaped value used by injected fake clients."""

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _CompatPart(_CompatValue):
    @classmethod
    def from_text(cls, *, text: str) -> "_CompatPart":
        return cls(text=text)

    @classmethod
    def from_function_response(
        cls, *, name: str, response: dict[str, Any]
    ) -> "_CompatPart":
        return cls(name=name, response=response)

    @classmethod
    def from_bytes(cls, *, data: bytes, mime_type: str) -> "_CompatPart":
        return cls(data=data, mime_type=mime_type)


class _CompatTypes:
    Part = _CompatPart
    Content = _CompatValue
    FunctionDeclaration = _CompatValue
    Tool = _CompatValue
    GenerateContentConfig = _CompatValue


class GeminiSceneQaAgent:
    """Manual Gemini function-calling loop with auditable local tool execution."""

    def __init__(
        self,
        graph: GraphStore | None,
        registry: ToolRegistry,
        config: SceneQaConfig,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        history_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            if client is None:
                raise RuntimeError(
                    "google-genai is required. Activate the jarvis conda environment "
                    "or install google-genai."
                ) from exc
            genai = None
            types = _CompatTypes

        self.graph = graph
        self.registry = registry
        self.config = config
        self._types = types
        self._progress_callback = progress_callback
        self._history_callback = history_callback
        self._cancel_event = cancel_event
        self._active_history: dict[str, Any] | None = None
        if client is None:
            sanitize_proxy_environment()
        if client is not None:
            self._client = client
        else:
            assert genai is not None
            self._client = genai.Client(
                api_key=api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )

    def answer_query(self, query: str) -> QaResponse:
        # The live read session is deliberately created lazily by the first
        # tool call, after the model's initial planning latency.
        self.registry.end_answer()
        try:
            response = self._answer_query_in_session(query, None)
            self._refresh_read_session(response.history)
            self._emit_history(response.history)
            return response
        except Exception as exc:
            if self._active_history is not None:
                self._refresh_read_session(self._active_history)
                self._active_history["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                self._emit_history(self._active_history)
                try:
                    setattr(exc, "scene_qa_history", self._active_history)
                except Exception:
                    pass
            raise
        finally:
            self._active_history = None
            self.registry.end_answer()

    def _answer_query_in_session(
        self,
        query: str,
        session: dict[str, Any] | None,
    ) -> QaResponse:
        types = self._types
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)],
            )
        ]
        history: dict[str, Any] = {
            "query": query,
            "graph_json": str(self.graph.json_path) if self.graph is not None else None,
            "source": "offline_json" if self.graph is not None else "live_query_service",
            "provider": "gemini",
            "model": self.config.gemini_model,
            "tools": self.registry.list_tools(),
            "iterations": [],
        }
        self._active_history = history
        if session is not None:
            history["read_session"] = session
        self._emit_history(history)

        for iteration in range(1, self.config.max_iterations + 1):
            self._check_cancelled()
            self._emit_progress(
                "gemini_start",
                "Gemini planning the next step...",
                iteration=iteration,
            )
            response = self._client.models.generate_content(
                model=self.config.gemini_model,
                contents=contents,
                config=self._build_generate_config(include_tools=True),
            )
            self._emit_progress(
                "gemini_done",
                "Gemini response received.",
                iteration=iteration,
            )
            model_content = self._candidate_content(response)
            if model_content is not None:
                contents.append(model_content)

            function_calls = self._extract_function_calls(response)
            iter_data: dict[str, Any] = {
                "iteration": iteration,
                "model_text": self._response_text(response),
                "function_calls": [],
            }
            history["iterations"].append(iter_data)
            self._emit_history(history)
            self._check_cancelled()

            if not function_calls:
                raw_text = self._response_text(response)
                parsed = self._parse_final_text(raw_text)
                history["final_response"] = parsed
                self._refresh_read_session(history)
                self._emit_history(history)
                self._emit_progress(
                    "final_answer",
                    "Final answer received.",
                    iteration=iteration,
                )
                return QaResponse(
                    reasoning=str(parsed.get("reasoning", "")),
                    answer=parsed.get("answer", raw_text),
                    raw_text=raw_text,
                    history=history,
                )

            response_parts = []
            for call in function_calls:
                self._check_cancelled()
                name = str(getattr(call, "name", ""))
                args = dict(getattr(call, "args", {}) or {})
                self._emit_progress(
                    "tool_start",
                    self._tool_start_message(name, args),
                    iteration=iteration,
                    tool=name,
                    args=args,
                )
                try:
                    tool_result = self.registry.call_tool(name, args)
                except Exception as exc:
                    failed_result = ToolResult(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "status": str(getattr(exc, "status", "query_failed")),
                        }
                    )
                    iter_data["function_calls"].append(
                        self._history_call(
                            name,
                            args,
                            failed_result,
                            call_id=str(getattr(call, "id", "") or ""),
                        )
                    )
                    self._refresh_read_session(history)
                    self._emit_history(history)
                    self._emit_progress(
                        "tool_done",
                        self._tool_done_message(name, failed_result),
                        iteration=iteration,
                        tool=name,
                        status=str(getattr(exc, "status", "query_failed")),
                    )
                    raise
                self._emit_progress(
                    "tool_done",
                    self._tool_done_message(name, tool_result),
                    iteration=iteration,
                    tool=name,
                    **self._tool_result_summary(tool_result),
                )
                call_data = self._history_call(
                    name,
                    args,
                    tool_result,
                    call_id=str(getattr(call, "id", "") or ""),
                )
                iter_data["function_calls"].append(call_data)
                self._refresh_read_session(history)
                self._emit_history(history)
                self._check_cancelled()

                response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response=tool_result.response,
                    )
                )
                for media in tool_result.media:
                    media_label = str(media.summary.get("label") or "").strip()
                    if media_label:
                        response_parts.append(types.Part.from_text(text=media_label))
                    response_parts.append(
                        types.Part.from_bytes(
                            data=media.data,
                            mime_type=media.mime_type,
                        )
                    )

            contents.append(types.Content(role="user", parts=response_parts))

        final_text = self._force_final_answer(contents)
        parsed = self._parse_final_text(final_text)
        history["iterations"].append(
            {
                "iteration": self.config.max_iterations + 1,
                "phase": "forced_final",
                "model_text": final_text,
                "function_calls": [],
            }
        )
        history["final_response"] = parsed
        history["max_iterations_reached"] = True
        self._refresh_read_session(history)
        self._emit_history(history)
        self._emit_progress("final_answer", "Final answer received after max iterations.")
        return QaResponse(
            reasoning=str(parsed.get("reasoning", "")),
            answer=parsed.get("answer", final_text),
            raw_text=final_text,
            history=history,
        )

    def _build_generate_config(self, *, include_tools: bool):
        types = self._types
        kwargs: dict[str, Any] = {
            "system_instruction": self.registry.system_prompt(
                self.config.load_system_prompt()
            ),
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
        }
        if include_tools:
            declarations = [
                types.FunctionDeclaration(
                    name=decl["name"],
                    description=decl["description"],
                    parameters_json_schema=decl["parameters_json_schema"],
                )
                for decl in self.registry.declarations()
            ]
            kwargs["tools"] = [types.Tool(function_declarations=declarations)]
        return types.GenerateContentConfig(**kwargs)

    def _force_final_answer(self, contents: list[Any]) -> str:
        types = self._types
        self._emit_progress(
            "final_start",
            "Requesting final answer from gathered tool results...",
        )
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Maximum tool iterations reached. Based only on the "
                            "tool results already gathered, provide the final JSON answer now."
                        )
                    )
                ],
            )
        )
        response = self._client.models.generate_content(
            model=self.config.gemini_model,
            contents=contents,
            config=self._build_generate_config(include_tools=False),
        )
        self._emit_progress("final_done", "Forced final response received.")
        return self._response_text(response)

    def _emit_progress(self, phase: str, message: str, **payload: Any) -> None:
        if self._progress_callback is None:
            return
        event = {"phase": phase, "message": message, **payload}
        try:
            self._progress_callback(event)
        except Exception:
            pass

    def _emit_history(self, history: dict[str, Any]) -> None:
        if self._history_callback is None:
            return
        try:
            self._history_callback(history)
        except Exception:
            pass

    def _refresh_read_session(self, history: dict[str, Any]) -> None:
        session = self.registry.answer_metadata()
        if session is not None and session.get("session_id"):
            history["read_session"] = session

    def _check_cancelled(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise SceneQaCancelledError("question was reset")

    def _tool_start_message(self, name: str, args: dict[str, Any]) -> str:
        if name == "search_objects":
            description = str(args.get("description") or "").strip()
            return f"sentence querying: {description}" if description else "sentence querying objects..."
        if name == "get_objects_in_room" and args.get("description"):
            return (
                f"sentence querying room {args.get('room_id')}: "
                f"{str(args.get('description')).strip()}"
            )
        if name == "inspect_snapshot":
            return f"loading snapshot for object #{args.get('object_id')}"
        if name == "inspect_object_reference":
            return f"loading object reference {args.get('reference_id')}"
        if name == "compare_scene_objects_to_reference":
            return (
                f"loading reference {args.get('reference_id')} with scene objects "
                f"{args.get('object_ids')}"
            )
        if name == "get_object":
            return f"loading object metadata for #{args.get('object_id')}"
        if name == "list_rooms":
            return "loading room list..."
        return f"running local tool: {name}"

    def _tool_done_message(self, name: str, result: ToolResult) -> str:
        if "error" in result.response:
            return f"{name} failed: {result.response['error']}"
        summary = self._tool_result_summary(result)
        object_count = len(summary["object_ids"])
        media_count = int(summary["media_count"])
        if object_count and media_count:
            return f"{name} returned {object_count} object(s) and {media_count} image attachment(s)."
        if object_count:
            return f"{name} returned {object_count} object candidate(s)."
        if media_count:
            return f"{name} returned {media_count} image attachment(s)."
        return f"{name} completed."

    def _tool_result_summary(self, result: ToolResult) -> dict[str, Any]:
        object_ids: list[int] = []
        seen: set[int] = set()

        def add(value: Any) -> None:
            if isinstance(value, bool):
                return
            try:
                object_id = int(value)
            except (TypeError, ValueError):
                return
            if object_id not in seen:
                seen.add(object_id)
                object_ids.append(object_id)

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, item in node.items():
                    if key == "object_id":
                        add(item)
                    else:
                        walk(item)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(result.response)
        return {
            "object_ids": object_ids,
            "media_count": len(result.media),
            "response_count": result.response.get("count"),
        }

    def _candidate_content(self, response: Any) -> Any | None:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        content = getattr(candidates[0], "content", None)
        if content is not None and getattr(content, "role", None) is None:
            content.role = "model"
        return content

    def _extract_function_calls(self, response: Any) -> list[Any]:
        calls = getattr(response, "function_calls", None)
        if calls:
            return list(calls)
        calls = []
        for part in getattr(response, "parts", None) or []:
            call = getattr(part, "function_call", None)
            if call is not None:
                calls.append(call)
        content = self._candidate_content(response)
        for part in getattr(content, "parts", None) or []:
            call = getattr(part, "function_call", None)
            if call is not None and call not in calls:
                calls.append(call)
        return calls

    def _response_text(self, response: Any) -> str:
        content = self._candidate_content(response)
        content_parts = getattr(content, "parts", None) or []
        texts = [
            getattr(part, "text", None)
            for part in content_parts
            if getattr(part, "text", None)
        ]
        if texts:
            return "\n".join(str(text) for text in texts)
        if self._extract_function_calls(response):
            return ""
        try:
            text = response.text
            if text:
                return str(text)
        except Exception:
            pass
        parts = getattr(response, "parts", None) or []
        texts = [getattr(part, "text", None) for part in parts if getattr(part, "text", None)]
        return "\n".join(str(text) for text in texts)

    def _history_call(
        self,
        name: str,
        args: dict[str, Any],
        result: ToolResult,
        *,
        call_id: str = "",
    ) -> dict[str, Any]:
        data = {
            "name": name,
            "args": args,
            "response": result.response,
        }
        if call_id:
            data["call_id"] = call_id
        if result.media:
            data["media"] = [media.summary for media in result.media]
        return data

    def _parse_final_text(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if not stripped:
            return {"reasoning": "", "answer": ""}
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(stripped[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"reasoning": "", "answer": stripped}
