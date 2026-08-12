#!/usr/bin/python3
"""ROS 2 QA action servers plus an action client, bridged over JSON lines."""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import uuid
from typing import Any


def emit(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with EMIT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def wait_future(future: Any, timeout_sec: float | None = None) -> Any:
    completed = threading.Event()
    future.add_done_callback(lambda _future: completed.set())
    if not completed.wait(timeout=timeout_sec):
        raise TimeoutError("ROS action operation timed out")
    exception = future.exception()
    if exception is not None:
        raise exception
    return future.result()


def goal_key(goal_handle: Any) -> str:
    return bytes(goal_handle.goal_id.uuid).hex()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


EMIT_LOCK = threading.Lock()


def main() -> int:
    if len(sys.argv) != 3:
        emit(
            {
                "type": "fatal",
                "error": "usage: worker ACTION_PREFIX SERVER_TIMEOUT_SEC",
            }
        )
        return 2
    action_prefix = sys.argv[1].rstrip("/")
    try:
        server_timeout_sec = float(sys.argv[2])
        import rclpy
        from action_msgs.msg import GoalStatus
        from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from roomie_msgs.action import FindObjectInView, RoomieNavigation, RoomieSceneQA
        from rosidl_runtime_py import message_to_ordereddict, set_message_fields
    except Exception as exc:
        emit({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        return 2

    action_types = {
        "scene_qa": RoomieSceneQA,
        "navigation": RoomieNavigation,
        "find_object_in_view": FindObjectInView,
    }
    action_names = {
        task: f"{action_prefix}/{task}" for task in action_types
    }
    rclpy.init(args=None)
    node = rclpy.create_node(f"roomie_qa_actions_{os.getpid()}")
    callback_group = ReentrantCallbackGroup()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    stopping = threading.Event()
    execute_queues: dict[str, queue.Queue[dict[str, Any]]] = {}
    execute_lock = threading.Lock()
    goal_to_execution: dict[str, str] = {}

    def feedback(action_type: Any, phase: int, event: dict[str, Any]) -> Any:
        value = action_type.Feedback()
        value.phase = int(phase)
        value.event_json = compact_json(event)
        return value

    def error_body(response: dict[str, Any]) -> dict[str, Any]:
        return {
            "error": str(response.get("error") or "QA action execution failed"),
            "status": str(response.get("status") or "qa_execution_failed"),
            "http_status": int(response.get("http_status") or 500),
        }

    def action_result(
        task: str,
        response: dict[str, Any],
    ) -> Any:
        action_type = action_types[task]
        result = action_type.Result()
        ok = bool(response.get("ok"))
        body = response.get("body") if isinstance(response.get("body"), dict) else {}
        structured = response.get("structured_result")
        if not isinstance(structured, dict):
            answer = body.get("answer")
            structured = answer if isinstance(answer, dict) else {}
        result.response_json = compact_json(body if ok else error_body(response))
        if task == "scene_qa":
            result.status = result.STATUS_SUCCESS if ok else result.STATUS_FAILED
            result.reason = str(
                structured.get("reason")
                or body.get("reasoning")
                or response.get("error")
                or ""
            )
            answer = body.get("answer", "")
            result.answer = answer if isinstance(answer, str) else compact_json(answer)
            return result

        fields = {
            key: value
            for key, value in structured.items()
            if value is not None and key != "response_json"
        }
        if not ok:
            fields["status"] = 1
            fields["reason"] = str(response.get("error") or "QA action failed")
        set_message_fields(result, fields)
        return result

    def goal_callback(task: str, request: Any) -> Any:
        del task
        if not str(request.task_description).strip():
            return GoalResponse.REJECT
        if float(request.max_duration_s) < 0.0:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(goal_handle: Any) -> Any:
        key = goal_key(goal_handle)
        with execute_lock:
            execution_id = goal_to_execution.get(key)
        if execution_id:
            emit(
                {
                    "type": "cancel_goal",
                    "execution_id": execution_id,
                    "reason": "action client requested cancellation",
                }
            )
        return CancelResponse.ACCEPT

    def execute_callback(task: str, goal_handle: Any) -> Any:
        action_type = action_types[task]
        execution_id = uuid.uuid4().hex
        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        key = goal_key(goal_handle)
        with execute_lock:
            execute_queues[execution_id] = response_queue
            goal_to_execution[key] = execution_id
        goal_handle.publish_feedback(
            feedback(
                action_type,
                action_type.Feedback.PHASE_STARTING,
                {"phase": "starting", "task": task},
            )
        )
        request = goal_handle.request
        emit(
            {
                "type": "execute_goal",
                "execution_id": execution_id,
                "task": task,
                "query": str(request.task_description),
                "provider": str(request.provider),
                "allow_follow_up_question": bool(
                    getattr(request, "allow_follow_up_question", True)
                ),
                "max_duration_s": float(request.max_duration_s),
            }
        )
        goal_handle.publish_feedback(
            feedback(
                action_type,
                action_type.Feedback.PHASE_EXPLORING,
                {"phase": "exploring", "task": task},
            )
        )
        timeout = float(request.max_duration_s)
        timeout = timeout if timeout > 0.0 else None
        try:
            response = response_queue.get(timeout=timeout)
        except queue.Empty:
            response = {
                "ok": False,
                "error": f"QA action exceeded max_duration_s={timeout:g}",
                "status": "action_deadline_exceeded",
                "http_status": 504,
            }
            emit(
                {
                    "type": "cancel_goal",
                    "execution_id": execution_id,
                    "reason": "action deadline exceeded",
                }
            )
        finally:
            with execute_lock:
                execute_queues.pop(execution_id, None)
                goal_to_execution.pop(key, None)

        goal_handle.publish_feedback(
            feedback(
                action_type,
                action_type.Feedback.PHASE_SAVING_RESULT,
                {"phase": "saving_result", "task": task},
            )
        )
        result = action_result(task, response)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif response.get("ok"):
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    servers = []
    clients = {}
    for task, action_type in action_types.items():
        servers.append(
            ActionServer(
                node,
                action_type,
                action_names[task],
                lambda goal_handle, task=task: execute_callback(task, goal_handle),
                callback_group=callback_group,
                goal_callback=lambda request, task=task: goal_callback(task, request),
                cancel_callback=cancel_callback,
            )
        )
        clients[task] = ActionClient(
            node,
            action_type,
            action_names[task],
            callback_group=callback_group,
        )

    spin_thread = threading.Thread(target=executor.spin, name="qa-actions-spin", daemon=True)
    spin_thread.start()

    def send_goal(command: dict[str, Any]) -> None:
        request_id = str(command.get("request_id") or "")
        task = str(command.get("task") or "")
        try:
            if task not in action_types:
                raise ValueError(f"unknown QA action task: {task}")
            client = clients[task]
            if not client.wait_for_server(timeout_sec=server_timeout_sec):
                raise TimeoutError(f"action server {action_names[task]} is unavailable")
            goal = action_types[task].Goal()
            goal.task_description = str(command.get("query") or "")
            goal.provider = str(command.get("provider") or "")
            goal.max_duration_s = float(command.get("max_duration_s") or 0.0)
            if task == "navigation":
                goal.allow_follow_up_question = bool(
                    command.get("allow_follow_up_question", True)
                )

            def on_feedback(feedback_message: Any) -> None:
                value = feedback_message.feedback
                emit(
                    {
                        "type": "goal_feedback",
                        "request_id": request_id,
                        "task": task,
                        "phase": int(value.phase),
                        "event_json": str(value.event_json),
                    }
                )

            goal_handle = wait_future(
                client.send_goal_async(goal, feedback_callback=on_feedback),
                timeout_sec=server_timeout_sec,
            )
            if goal_handle is None or not goal_handle.accepted:
                emit(
                    {
                        "type": "goal_result",
                        "request_id": request_id,
                        "ok": False,
                        "error": f"action goal was rejected by {action_names[task]}",
                        "status": "action_goal_rejected",
                        "http_status": 503,
                    }
                )
                return
            wrapped = wait_future(goal_handle.get_result_async())
            result = wrapped.result
            try:
                body = json.loads(str(result.response_json))
            except json.JSONDecodeError:
                body = None
            succeeded = int(wrapped.status) == int(GoalStatus.STATUS_SUCCEEDED)
            if not succeeded:
                error_value = body if isinstance(body, dict) else {}
                emit(
                    {
                        "type": "goal_result",
                        "request_id": request_id,
                        "ok": False,
                        "error": str(error_value.get("error") or "QA action failed"),
                        "status": str(error_value.get("status") or "action_failed"),
                        "http_status": int(error_value.get("http_status") or 500),
                        "action_status": int(wrapped.status),
                        "result": message_to_ordereddict(result),
                    }
                )
                return
            emit(
                {
                    "type": "goal_result",
                    "request_id": request_id,
                    "ok": True,
                    "body": body,
                    "action_status": int(wrapped.status),
                    "result": message_to_ordereddict(result),
                }
            )
        except Exception as exc:
            emit(
                {
                    "type": "goal_result",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": "action_transport_error",
                    "http_status": 503,
                }
            )

    emit({"type": "ready", "action_names": action_names})
    try:
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(command, dict):
                continue
            command_type = str(command.get("type") or "")
            if command_type == "shutdown":
                break
            if command_type == "execute_result":
                execution_id = str(command.get("execution_id") or "")
                with execute_lock:
                    target = execute_queues.get(execution_id)
                if target is not None:
                    try:
                        target.put_nowait(command)
                    except queue.Full:
                        pass
            elif command_type == "send_goal":
                threading.Thread(
                    target=send_goal,
                    args=(command,),
                    name="qa-actions-client",
                    daemon=True,
                ).start()
    finally:
        stopping.set()
        executor.shutdown(timeout_sec=2.0)
        for server in servers:
            server.destroy()
        for client in clients.values():
            client.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
