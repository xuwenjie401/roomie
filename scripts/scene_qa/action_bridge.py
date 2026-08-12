"""JSON-lines bridge between the QA Python runtime and ROS 2 actions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import uuid
from typing import Any, Callable


class QaActionBridgeError(RuntimeError):
    def __init__(self, message: str, *, status: str = "action_error", http_status: int = 500):
        self.status = status
        self.http_status = int(http_status)
        super().__init__(message)


class RosQaActionBridge:
    """Own a system-Python ROS worker and route goals back to QA agents."""

    def __init__(
        self,
        *,
        execute_callback: Callable[[dict[str, Any]], dict[str, Any]],
        cancel_callback: Callable[[dict[str, Any]], None],
        action_prefix: str = "/roomie",
        server_timeout_sec: float = 5.0,
    ):
        prefix = str(action_prefix).strip().rstrip("/")
        if not prefix or not prefix.startswith("/"):
            raise ValueError("action_prefix must be an absolute ROS name")
        if server_timeout_sec <= 0.0:
            raise ValueError("action server timeout must be positive")
        self.action_prefix = prefix
        self.server_timeout_sec = float(server_timeout_sec)
        self._execute_callback = execute_callback
        self._cancel_callback = cancel_callback
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._fatal_error = ""
        worker_path = Path(__file__).with_name("_ros_action_bridge_worker.py")
        ros_python = os.environ.get("ROOMIE_ROS_PYTHON", "/usr/bin/python3")
        try:
            self._worker = subprocess.Popen(
                [
                    ros_python,
                    "-u",
                    str(worker_path),
                    self.action_prefix,
                    str(self.server_timeout_sec),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise QaActionBridgeError(
                f"could not start ROS QA action worker: {exc}",
                status="action_worker_startup_error",
            ) from exc
        self._reader = threading.Thread(
            target=self._read_loop,
            name="roomie-qa-action-reader",
            daemon=True,
        )
        self._reader.start()
        if not self._ready.wait(timeout=self.server_timeout_sec):
            message = self._worker_exit_message()
            self.close()
            raise QaActionBridgeError(
                message or "ROS QA action worker did not become ready",
                status="action_worker_startup_error",
            )
        if self._fatal_error or self._worker.poll() is not None:
            message = self._fatal_error or self._worker_exit_message()
            self.close()
            raise QaActionBridgeError(
                message,
                status="action_worker_startup_error",
            )

    @property
    def action_names(self) -> dict[str, str]:
        return {
            task: f"{self.action_prefix}/{task}"
            for task in ("scene_qa", "navigation", "find_object_in_view")
        }

    def request(
        self,
        *,
        task: str,
        query: str,
        provider: str,
        allow_follow_up_question: bool,
        max_duration_s: float = 0.0,
    ) -> dict[str, Any]:
        if self._closed.is_set():
            raise QaActionBridgeError(
                "ROS QA action bridge is closed",
                status="action_worker_unavailable",
                http_status=503,
            )
        request_id = uuid.uuid4().hex
        event = threading.Event()
        response: dict[str, Any] = {}
        with self._pending_lock:
            self._pending[request_id] = (event, response)
        try:
            self._write(
                {
                    "type": "send_goal",
                    "request_id": request_id,
                    "task": task,
                    "query": query,
                    "provider": provider,
                    "allow_follow_up_question": bool(allow_follow_up_question),
                    "max_duration_s": max(0.0, float(max_duration_s)),
                }
            )
            while not event.wait(timeout=0.25):
                if self._closed.is_set() or self._worker.poll() is not None:
                    raise QaActionBridgeError(
                        self._worker_exit_message(),
                        status="action_worker_unavailable",
                        http_status=503,
                    )
            if not response.get("ok"):
                raise QaActionBridgeError(
                    str(response.get("error") or "ROS QA action request failed"),
                    status=str(response.get("status") or "action_request_failed"),
                    http_status=int(response.get("http_status") or 500),
                )
            body = response.get("body")
            if not isinstance(body, dict):
                raise QaActionBridgeError(
                    "ROS QA action result omitted the Web response envelope",
                    status="action_invalid_result",
                )
            return body
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def _read_loop(self) -> None:
        worker = self._worker
        stdout = worker.stdout
        if stdout is None:
            self._fail_all("ROS QA action worker has no stdout")
            return
        try:
            for line in stdout:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                message_type = str(message.get("type") or "")
                if message_type == "ready":
                    self._ready.set()
                elif message_type == "fatal":
                    self._fatal_error = str(message.get("error") or "action worker failed")
                    self._ready.set()
                elif message_type == "goal_result":
                    self._resolve_pending(message)
                elif message_type == "execute_goal":
                    threading.Thread(
                        target=self._execute_goal,
                        args=(message,),
                        name="roomie-qa-action-execute",
                        daemon=True,
                    ).start()
                elif message_type == "cancel_goal":
                    try:
                        self._cancel_callback(message)
                    except Exception:
                        pass
        finally:
            self._closed.set()
            self._ready.set()
            self._fail_all(self._worker_exit_message())

    def _resolve_pending(self, message: dict[str, Any]) -> None:
        request_id = str(message.get("request_id") or "")
        with self._pending_lock:
            pending = self._pending.get(request_id)
        if pending is None:
            return
        event, response = pending
        response.update(message)
        event.set()

    def _execute_goal(self, message: dict[str, Any]) -> None:
        execution_id = str(message.get("execution_id") or "")
        if not execution_id:
            return
        try:
            result = self._execute_callback(message)
            if not isinstance(result, dict):
                raise TypeError("QA action execute callback must return an object")
            payload = {"ok": True, **result}
        except QaActionBridgeError as exc:
            payload = {
                "ok": False,
                "error": str(exc),
                "status": exc.status,
                "http_status": exc.http_status,
            }
        except Exception as exc:
            payload = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "status": "qa_execution_failed",
                "http_status": 500,
            }
        try:
            self._write(
                {
                    "type": "execute_result",
                    "execution_id": execution_id,
                    **payload,
                }
            )
        except QaActionBridgeError:
            pass

    def _write(self, message: dict[str, Any]) -> None:
        worker = self._worker
        if worker.poll() is not None or worker.stdin is None:
            raise QaActionBridgeError(
                self._worker_exit_message(),
                status="action_worker_unavailable",
                http_status=503,
            )
        try:
            line = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            with self._write_lock:
                worker.stdin.write(line + "\n")
                worker.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise QaActionBridgeError(
                str(exc),
                status="action_worker_unavailable",
                http_status=503,
            ) from exc

    def _fail_all(self, error: str) -> None:
        with self._pending_lock:
            pending_values = list(self._pending.values())
        for event, response in pending_values:
            response.update(
                {
                    "ok": False,
                    "error": error,
                    "status": "action_worker_unavailable",
                    "http_status": 503,
                }
            )
            event.set()

    def _worker_exit_message(self) -> str:
        worker = self._worker
        detail = ""
        if worker.poll() is not None and worker.stderr is not None:
            try:
                detail = worker.stderr.read().strip()
            except OSError:
                detail = ""
        message = f"ROS QA action worker exited with code {worker.poll()}"
        return f"{message}: {detail}" if detail else message

    def close(self) -> None:
        if self._closed.is_set() and self._worker.poll() is not None:
            return
        self._closed.set()
        try:
            self._write({"type": "shutdown"})
        except QaActionBridgeError:
            pass
        worker = self._worker
        if worker.stdin is not None:
            try:
                worker.stdin.close()
            except OSError:
                pass
        try:
            worker.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:
                worker.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=1.0)
        self._fail_all("ROS QA action bridge closed")
