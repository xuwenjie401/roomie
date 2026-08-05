"""Live client for the Roomie ``/roomie/query_scene`` JSON service."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable


REQUEST_SCHEMA = "roomie.query_scene.v1"


class LiveSceneQueryError(RuntimeError):
    """Structured service/session/call failure."""

    def __init__(self, status: str, message: str, payload: dict[str, Any] | None = None):
        self.status = status
        self.payload = payload or {}
        super().__init__(f"{status}: {message}" if status else message)


class LiveSceneQueryClient:
    """One-answer client that keeps all tool calls on one pinned revision.

    ``transport`` accepts a JSON-compatible request dict and returns the parsed
    service response dict. This small boundary keeps ROS out of unit tests.
    """

    def __init__(
        self,
        transport: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        session_ttl_ms: int = 30_000,
    ):
        if session_ttl_ms <= 0 or session_ttl_ms > 300_000:
            raise ValueError("session_ttl_ms must be between 1 and 300000")
        self._transport = transport
        self.session_ttl_ms = int(session_ttl_ms)
        self.session_id: str | None = None
        self.scene_revision: int | None = None
        self.durable_scene_revision: int | None = None
        self.index_generation: int | None = None

    def begin_answer(self) -> dict[str, Any]:
        # A new model answer must never inherit the previous answer's lease.
        self.end_answer()
        response = self._request(
            {
                "schema_version": REQUEST_SCHEMA,
                "operation": "begin_session",
                "ttl_ms": self.session_ttl_ms,
            }
        )
        session = response.get("session")
        if not isinstance(session, dict):
            session = {}
        session_id = response.get("session_id") or session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise LiveSceneQueryError(
                "invalid_response", "begin_session did not return session_id", response
            )
        read_token = response.get("read_token")
        if not isinstance(read_token, dict):
            read_token = session
        try:
            scene_revision = int(read_token["scene_revision"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LiveSceneQueryError(
                "invalid_response",
                "begin_session did not return scene_revision",
                response,
            ) from exc
        self.session_id = session_id
        self.scene_revision = scene_revision
        self.durable_scene_revision = _optional_int(
            read_token.get("durable_scene_revision")
        )
        self.index_generation = _optional_int(read_token.get("index_generation"))
        return self.session_metadata()

    def end_answer(self) -> None:
        # The server lease is deliberately fixed-TTL. Dropping the local id is
        # enough; the bounded server table reclaims it on expiry.
        self.session_id = None
        self.scene_revision = None
        self.durable_scene_revision = None
        self.index_generation = None

    def session_metadata(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "scene_revision": self.scene_revision,
            "durable_scene_revision": self.durable_scene_revision,
            "index_generation": self.index_generation,
            "ttl_ms": self.session_ttl_ms,
        }

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        calls = self.call_many([{"method": method, "params": params or {}}])
        return calls[0]

    def call_many(self, calls: list[dict[str, Any]]) -> list[Any]:
        if not calls:
            return []
        if self.session_id is None or self.scene_revision is None:
            self.begin_answer()
        request = {
            "schema_version": REQUEST_SCHEMA,
            "session_id": self.session_id,
            "expected_scene_revision": self.scene_revision,
            "calls": calls,
        }
        response = self._request(request)
        returned_session = response.get("session_id")
        if returned_session is not None and returned_session != self.session_id:
            raise LiveSceneQueryError(
                "stale_session", "service returned a different session_id", response
            )
        read_token = response.get("read_token")
        if not isinstance(read_token, dict) or _optional_int(
            read_token.get("scene_revision")
        ) != self.scene_revision:
            raise LiveSceneQueryError(
                "stale_session",
                "service response revision does not match the answer session",
                response,
            )
        call_results = response.get("calls")
        if not isinstance(call_results, list) or len(call_results) != len(calls):
            raise LiveSceneQueryError(
                "invalid_response", "service returned an invalid calls array", response
            )
        values: list[Any] = []
        for call_result in call_results:
            if not isinstance(call_result, dict):
                raise LiveSceneQueryError(
                    "invalid_response", "service returned a non-object call result", response
                )
            if not call_result.get("success"):
                raise LiveSceneQueryError(
                    str(call_result.get("status") or "query_failed"),
                    str(call_result.get("message") or "scene query failed"),
                    call_result,
                )
            metadata = call_result.get("metadata")
            if not isinstance(metadata, dict) or _optional_int(
                metadata.get("scene_revision")
            ) != self.scene_revision:
                raise LiveSceneQueryError(
                    "stale_session",
                    "tool result revision does not match the answer session",
                    call_result,
                )
            values.append(call_result.get("result"))
        return values

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._transport(payload)
        except LiveSceneQueryError:
            raise
        except Exception as exc:
            raise LiveSceneQueryError("transport_error", str(exc)) from exc
        if not isinstance(response, dict):
            raise LiveSceneQueryError(
                "invalid_response", "query transport must return a JSON object"
            )
        if response.get("success") is False:
            raise LiveSceneQueryError(
                str(response.get("status") or "query_failed"),
                str(response.get("error") or "scene query failed"),
                response,
            )
        return response


class RosQuerySceneTransport:
    """ROS 2 transport used by the command-line live path.

    It uses rclpy in-process when ABI-compatible. If the Gemini environment's
    Python version differs from ROS Humble's system Python, it automatically
    starts the tiny bundled JSON-lines worker with ``/usr/bin/python3``.
    """

    def __init__(
        self,
        *,
        service_name: str = "/roomie/query_scene",
        timeout_sec: float = 10.0,
    ):
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        self._timeout_sec = float(timeout_sec)
        self.service_name = service_name
        self._rclpy = None
        self._query_scene_type = None
        self._owns_rclpy = False
        self._node = None
        self._client = None
        self._worker: subprocess.Popen[str] | None = None
        self._direct_import_error = ""
        try:
            import rclpy
            from roomie.srv import QueryScene

            self._rclpy = rclpy
            self._query_scene_type = QueryScene
            self._owns_rclpy = not rclpy.ok()
            if self._owns_rclpy:
                rclpy.init(args=None)
            self._node = rclpy.create_node(f"roomie_scene_qa_{os.getpid()}")
            self._client = self._node.create_client(QueryScene, service_name)
        except Exception as exc:
            self._direct_import_error = f"{type(exc).__name__}: {exc}"
            self._start_worker()

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._worker is not None:
            return self._call_worker(payload)
        if self._client is None or self._rclpy is None or self._query_scene_type is None:
            raise LiveSceneQueryError("transport_error", "ROS transport is not initialized")
        if not self._client.wait_for_service(timeout_sec=self._timeout_sec):
            raise LiveSceneQueryError(
                "service_unavailable",
                f"service {self.service_name} was not available within {self._timeout_sec:g}s",
            )
        request = self._query_scene_type.Request()
        request.request_json = json.dumps(payload, ensure_ascii=False)
        future = self._client.call_async(request)
        self._rclpy.spin_until_future_complete(
            self._node, future, timeout_sec=self._timeout_sec
        )
        if not future.done():
            raise LiveSceneQueryError(
                "service_timeout",
                f"service {self.service_name} did not respond within {self._timeout_sec:g}s",
            )
        response = future.result()
        if response is None:
            exception = future.exception()
            raise LiveSceneQueryError(
                "transport_error", str(exception or "service call failed")
            )
        return self._decode_service_response(
            bool(response.success), response.response_json, response.error
        )

    def _start_worker(self) -> None:
        worker_path = Path(__file__).with_name("_ros_query_worker.py")
        ros_python = os.environ.get("ROOMIE_ROS_PYTHON", "/usr/bin/python3")
        try:
            self._worker = subprocess.Popen(
                [
                    ros_python,
                    str(worker_path),
                    self.service_name,
                    str(self._timeout_sec),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise RuntimeError(
                "could not start the ROS QueryScene worker; source ROS/Roomie setup "
                f"and set ROOMIE_ROS_PYTHON if needed (direct import: "
                f"{self._direct_import_error}; worker: {exc})"
            ) from exc

    def _call_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        worker = self._worker
        if worker is None or worker.stdin is None or worker.stdout is None:
            raise LiveSceneQueryError("transport_error", "ROS worker is unavailable")
        if worker.poll() is not None:
            raise LiveSceneQueryError("transport_error", self._worker_exit_message(worker))
        try:
            worker.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            worker.stdin.flush()
            line = worker.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise LiveSceneQueryError("transport_error", str(exc)) from exc
        if not line:
            raise LiveSceneQueryError("transport_error", self._worker_exit_message(worker))
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LiveSceneQueryError(
                "invalid_response", "ROS worker returned invalid JSON"
            ) from exc
        if not isinstance(response, dict):
            raise LiveSceneQueryError(
                "invalid_response", "ROS worker response was not an object"
            )
        if not response.get("ok"):
            raise LiveSceneQueryError(
                str(response.get("status") or "transport_error"),
                str(response.get("error") or "ROS worker failed"),
                response,
            )
        return self._decode_service_response(
            bool(response.get("success")),
            str(response.get("response_json") or ""),
            str(response.get("error") or ""),
        )

    def _worker_exit_message(self, worker: subprocess.Popen[str]) -> str:
        detail = ""
        if worker.poll() is not None and worker.stderr is not None:
            try:
                detail = worker.stderr.read().strip()
            except OSError:
                detail = ""
        message = f"ROS worker exited with code {worker.poll()}"
        if detail:
            message += f": {detail}"
        if self._direct_import_error:
            message += f" (direct rclpy import also failed: {self._direct_import_error})"
        return message

    @staticmethod
    def _decode_service_response(
        success: bool,
        response_json: str,
        error: str,
    ) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        if response_json:
            try:
                candidate = json.loads(response_json)
            except json.JSONDecodeError as exc:
                raise LiveSceneQueryError(
                    "invalid_response", "service returned invalid response_json"
                ) from exc
            if isinstance(candidate, dict):
                parsed = candidate
        if not success:
            if parsed:
                parsed.setdefault("success", False)
                parsed.setdefault("error", error)
                return parsed
            raise LiveSceneQueryError(
                "query_failed", error or "scene query service rejected request"
            )
        if not parsed:
            raise LiveSceneQueryError(
                "invalid_response", "successful service response_json was empty"
            )
        parsed.setdefault("success", True)
        return parsed

    def close(self) -> None:
        node = getattr(self, "_node", None)
        if node is not None:
            node.destroy_node()
            self._node = None
        if self._owns_rclpy and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
        worker = self._worker
        self._worker = None
        if worker is not None:
            if worker.stdin is not None:
                try:
                    worker.stdin.close()
                except OSError:
                    pass
            try:
                worker.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=1.0)

    def __enter__(self) -> "RosQuerySceneTransport":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
