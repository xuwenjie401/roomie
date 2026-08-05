"""Doubao Responses API tool-calling loop for Roomie scene QA."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

from .config import SceneQaConfig
from .gemini_agent import QaResponse, sanitize_proxy_environment
from .graph_store import GraphStore
from .tools import ToolRegistry, ToolResult


class _JsonResponse:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = int(status_code)
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _UrllibJsonClient:
    """Tiny dependency-free HTTP client with an httpx-like ``post`` seam."""

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> _JsonResponse:
        payload = json_module_dumps(json)
        request = urllib_request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                status = int(getattr(response, "status", 200))
        except urllib_error.HTTPError as exc:
            body = exc.read()
            status = int(exc.code)
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Doubao API connection failed: {exc.reason}") from exc
        try:
            decoded = json_module_loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Doubao API returned non-JSON data (HTTP {status})"
            ) from exc
        return _JsonResponse(status, decoded)


def json_module_dumps(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def json_module_loads(value: bytes) -> Any:
    return json.loads(value.decode("utf-8"))


class DoubaoSceneQaAgent:
    """Manual Doubao Responses API loop with auditable local tool execution.

    The implementation calls Ark's OpenAI-compatible HTTP endpoint directly,
    so it does not require the OpenAI or Volcengine Python SDK.
    """

    def __init__(
        self,
        graph: GraphStore | None,
        registry: ToolRegistry,
        config: SceneQaConfig,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        request_timeout_sec: float = 180.0,
    ):
        if request_timeout_sec <= 0:
            raise ValueError("request_timeout_sec must be positive")
        resolved_key = (
            api_key
            or os.environ.get("DOUBAO_API_KEY")
            or os.environ.get("doubao_api_key")
            or os.environ.get("ARK_API_KEY")
        )
        if not resolved_key:
            raise RuntimeError(
                "Doubao API key is missing; export DOUBAO_API_KEY or ARK_API_KEY."
            )
        if client is None:
            sanitize_proxy_environment()
        self.graph = graph
        self.registry = registry
        self.config = config
        self._api_key = resolved_key
        self._client = client or _UrllibJsonClient()
        self._progress_callback = progress_callback
        self._request_timeout_sec = float(request_timeout_sec)
        self._endpoint = f"{config.doubao_base_url.rstrip('/')}/responses"

    def answer_query(self, query: str) -> QaResponse:
        session = self.registry.begin_answer()
        try:
            return self._answer_query_in_session(query, session)
        finally:
            self.registry.end_answer()

    def _answer_query_in_session(
        self,
        query: str,
        session: dict[str, Any] | None,
    ) -> QaResponse:
        next_input: Any = query
        previous_response_id: str | None = None
        history: dict[str, Any] = {
            "query": query,
            "graph_json": str(self.graph.json_path) if self.graph is not None else None,
            "source": "offline_json" if self.graph is not None else "live_query_service",
            "provider": "doubao",
            "model": self.config.doubao_model,
            "tools": self.registry.list_tools(),
            "iterations": [],
        }
        if session is not None:
            history["read_session"] = session

        for iteration in range(1, self.config.max_iterations + 1):
            self._emit_progress(
                "doubao_start",
                "Doubao planning the next step...",
                iteration=iteration,
            )
            payload = self._request_payload(
                next_input,
                previous_response_id=previous_response_id,
                include_tools=True,
            )
            response = self._post_response(payload)
            self._emit_progress(
                "doubao_done",
                "Doubao response received.",
                iteration=iteration,
            )
            response_id = str(response.get("id") or "")
            if not response_id:
                raise RuntimeError("Doubao API response is missing id")
            calls = self._extract_function_calls(response)
            model_text = self._response_text(response)
            iter_data: dict[str, Any] = {
                "iteration": iteration,
                "response_id": response_id,
                "model_text": model_text,
                "function_calls": [],
            }

            if not calls:
                parsed = self._parse_final_text(model_text)
                history["iterations"].append(iter_data)
                history["final_response"] = parsed
                self._emit_progress(
                    "final_answer",
                    "Final answer received.",
                    iteration=iteration,
                )
                return QaResponse(
                    reasoning=str(parsed.get("reasoning", "")),
                    answer=parsed.get("answer", model_text),
                    raw_text=model_text,
                    history=history,
                )

            tool_outputs: list[dict[str, Any]] = []
            media_inputs: list[dict[str, Any]] = []
            for call in calls:
                call_id = str(call.get("call_id") or "")
                name = str(call.get("name") or "")
                if not call_id or not name:
                    raise RuntimeError(
                        "Doubao function_call is missing call_id or name"
                    )
                args, argument_error = self._decode_arguments(call.get("arguments"))
                self._emit_progress(
                    "tool_start",
                    self._tool_start_message(name, args),
                    iteration=iteration,
                    tool=name,
                    args=args,
                )
                tool_result = (
                    ToolResult({"error": argument_error})
                    if argument_error
                    else self.registry.call_tool(name, args)
                )
                self._emit_progress(
                    "tool_done",
                    self._tool_done_message(name, tool_result),
                    iteration=iteration,
                    tool=name,
                    **self._tool_result_summary(tool_result),
                )
                iter_data["function_calls"].append(
                    self._history_call(name, args, tool_result)
                )
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(
                            tool_result.response,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
                if tool_result.media:
                    content: list[dict[str, Any]] = [
                        {
                            "type": "input_text",
                            "text": (
                                f"Local tool {name} returned the following visual "
                                "evidence. Use it together with its JSON output."
                            ),
                        }
                    ]
                    for media in tool_result.media:
                        encoded = base64.b64encode(media.data).decode("ascii")
                        content.append(
                            {
                                "type": "input_image",
                                "image_url": (
                                    f"data:{media.mime_type};base64,{encoded}"
                                ),
                            }
                        )
                    media_inputs.append(
                        {"type": "message", "role": "user", "content": content}
                    )

            history["iterations"].append(iter_data)
            previous_response_id = response_id
            next_input = [*tool_outputs, *media_inputs]

        final_text = self._force_final_answer(previous_response_id)
        parsed = self._parse_final_text(final_text)
        history["final_response"] = parsed
        history["max_iterations_reached"] = True
        self._emit_progress("final_answer", "Final answer received after max iterations.")
        return QaResponse(
            reasoning=str(parsed.get("reasoning", "")),
            answer=parsed.get("answer", final_text),
            raw_text=final_text,
            history=history,
        )

    def _request_payload(
        self,
        input_value: Any,
        *,
        previous_response_id: str | None,
        include_tools: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.doubao_model,
            "input": input_value,
            "instructions": self.config.load_system_prompt(),
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            # Ark's documented function-call continuation uses
            # previous_response_id, which requires stored responses.
            "store": True,
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if include_tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": declaration["name"],
                    "description": declaration["description"],
                    "parameters": declaration["parameters_json_schema"],
                }
                for declaration in self.registry.declarations()
            ]
        return payload

    def _post_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(
            self._endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._request_timeout_sec,
        )
        if isinstance(response, dict):
            status = 200
            data = response
        else:
            status = int(getattr(response, "status_code", 200))
            try:
                data = response.json()
            except Exception as exc:
                raise RuntimeError(
                    f"Doubao API returned invalid JSON (HTTP {status})"
                ) from exc
        if not isinstance(data, dict):
            raise RuntimeError("Doubao API response must be a JSON object")
        if status >= 400 or data.get("error"):
            error = data.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("code") or error)
            else:
                message = str(error or data.get("message") or "request failed")
            raise RuntimeError(f"Doubao API request failed (HTTP {status}): {message}")
        return data

    def _force_final_answer(self, previous_response_id: str | None) -> str:
        self._emit_progress(
            "final_start",
            "Requesting final answer from gathered tool results...",
        )
        payload = self._request_payload(
            (
                "Maximum tool iterations reached. Based only on the tool results "
                "already gathered, provide the final JSON answer now."
            ),
            previous_response_id=previous_response_id,
            include_tools=False,
        )
        response = self._post_response(payload)
        self._emit_progress("final_done", "Forced final response received.")
        return self._response_text(response)

    @staticmethod
    def _extract_function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
        output = response.get("output")
        if not isinstance(output, list):
            return []
        return [
            item
            for item in output
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]

    @staticmethod
    def _response_text(response: dict[str, Any]) -> str:
        direct = response.get("output_text")
        if isinstance(direct, str) and direct:
            return direct
        texts: list[str] = []
        output = response.get("output")
        if not isinstance(output, list):
            return ""
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if isinstance(content, str):
                texts.append(content)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_text", "text"} and isinstance(
                    part.get("text"), str
                ):
                    texts.append(part["text"])
        return "\n".join(texts)

    @staticmethod
    def _decode_arguments(value: Any) -> tuple[dict[str, Any], str]:
        if isinstance(value, dict):
            return value, ""
        if value is None or value == "":
            return {}, ""
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:
            return {}, f"invalid function arguments: {exc.msg}"
        if not isinstance(parsed, dict):
            return {}, "function arguments must decode to a JSON object"
        return parsed, ""

    def _emit_progress(self, phase: str, message: str, **payload: Any) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback({"phase": phase, "message": message, **payload})
        except Exception:
            pass

    @staticmethod
    def _tool_start_message(name: str, args: dict[str, Any]) -> str:
        if name == "search_objects":
            description = str(args.get("description") or "").strip()
            return (
                f"sentence querying: {description}"
                if description
                else "sentence querying objects..."
            )
        if name == "get_objects_in_room" and args.get("description"):
            return (
                f"sentence querying room {args.get('room_id')}: "
                f"{str(args.get('description')).strip()}"
            )
        if name == "inspect_snapshot":
            return f"loading snapshot for object #{args.get('object_id')}"
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
            return (
                f"{name} returned {object_count} object(s) and "
                f"{media_count} image attachment(s)."
            )
        if object_count:
            return f"{name} returned {object_count} object candidate(s)."
        if media_count:
            return f"{name} returned {media_count} image attachment(s)."
        return f"{name} completed."

    @staticmethod
    def _tool_result_summary(result: ToolResult) -> dict[str, Any]:
        object_ids: list[int] = []
        seen: set[int] = set()

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, item in node.items():
                    if key == "object_id" and not isinstance(item, bool):
                        try:
                            object_id = int(item)
                        except (TypeError, ValueError):
                            pass
                        else:
                            if object_id not in seen:
                                seen.add(object_id)
                                object_ids.append(object_id)
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

    @staticmethod
    def _history_call(
        name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": name,
            "args": args,
            "response": result.response,
        }
        if result.media:
            data["media"] = [media.summary for media in result.media]
        return data

    @staticmethod
    def _parse_final_text(text: str) -> dict[str, Any]:
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
