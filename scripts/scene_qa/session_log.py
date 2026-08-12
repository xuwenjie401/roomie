"""Persistent multi-turn conversation logs for Roomie Scene QA."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any


SCHEMA_VERSION = "roomie.scene_qa.conversation.v1"


def default_scene_qa_log_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "roomie_scene_qa_viewer.py").is_file():
            return parent / "logs" / "scene_qa"
    return (Path.cwd() / "logs" / "scene_qa").resolve()


def _timestamp(value: float | None = None) -> str:
    stamp = datetime.fromtimestamp(
        time.time() if value is None else value,
        tz=timezone.utc,
    )
    return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class SceneQaSessionLog:
    """One atomically updated JSON document for a viewer process."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        started_time_s: float | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._assistant_positions: dict[int, int] = {}
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        started = time.time() if started_time_s is None else float(started_time_s)
        local_stamp = datetime.fromtimestamp(started).astimezone()
        self.session_id = (
            f"{local_stamp.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"
        )
        self.path = self.log_dir / f"conversation_{self.session_id}.json"
        self.latest_path = self.log_dir / "latest.json"
        self._document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at": _timestamp(started),
            "updated_at": _timestamp(started),
            "ended_at": None,
            "context_mode": "stateless_per_turn",
            "metadata": dict(metadata or {}),
            "turn_count": 0,
            "messages": [],
        }
        with self._lock:
            self._write_locked()

    def begin_turn(
        self,
        *,
        query: str,
        provider: str,
        model: str | None,
        started_time_s: float,
        task: str | None = None,
    ) -> int:
        started = float(started_time_s)
        with self._lock:
            turn_index = int(self._document["turn_count"]) + 1
            user_message = {
                "turn": turn_index,
                "role": "user",
                "timestamp": _timestamp(started),
                "content": query,
            }
            assistant_message: dict[str, Any] = {
                "turn": turn_index,
                "role": "assistant",
                "timestamp": None,
                "provider": provider,
                "model": model,
                "status": "running",
                "duration_ms": None,
                "vlm_trace": {},
            }
            if task:
                user_message["task"] = task
                assistant_message["task"] = task
            self._document["messages"].extend([user_message, assistant_message])
            self._assistant_positions[turn_index] = len(self._document["messages"]) - 1
            self._document["turn_count"] = turn_index
            self._document["updated_at"] = _timestamp(started)
            self._write_locked()
            return turn_index

    def checkpoint_turn(
        self,
        turn_index: int,
        *,
        history: dict[str, Any],
        progress_events: list[dict[str, Any]] | None = None,
        checkpoint_time_s: float | None = None,
    ) -> bool:
        checkpoint = time.time() if checkpoint_time_s is None else checkpoint_time_s
        with self._lock:
            position = self._assistant_positions.get(int(turn_index))
            if position is None:
                return False
            message = self._document["messages"][position]
            if message.get("status") != "running":
                return False
            message["vlm_trace"] = history
            if progress_events is not None:
                message["progress_events"] = progress_events
            message["last_checkpoint_at"] = _timestamp(checkpoint)
            self._document["updated_at"] = _timestamp(checkpoint)
            self._write_locked()
            return True

    def finish_turn(
        self,
        turn_index: int,
        *,
        started_time_s: float,
        response: dict[str, Any] | None = None,
        history: dict[str, Any] | None = None,
        error: BaseException | None = None,
        progress_events: list[dict[str, Any]] | None = None,
        completed_time_s: float | None = None,
        status: str | None = None,
    ) -> bool:
        completed = time.time() if completed_time_s is None else float(completed_time_s)
        started = float(started_time_s)
        with self._lock:
            position = self._assistant_positions.get(int(turn_index))
            if position is None:
                return False
            message = self._document["messages"][position]
            if message.get("status") != "running":
                return False
            message.update(
                {
                    "timestamp": _timestamp(completed),
                    "status": status or ("error" if error is not None else "ok"),
                    "duration_ms": round(
                        max(0.0, completed - started) * 1000.0,
                        3,
                    ),
                    "vlm_trace": history or message.get("vlm_trace") or {},
                }
            )
            if response is not None:
                message.update(
                    {
                        "reasoning": response.get("reasoning"),
                        "content": response.get("answer"),
                        "raw_text": response.get("raw_text"),
                    }
                )
            if error is not None:
                message["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            if progress_events is not None:
                message["progress_events"] = progress_events
            self._document["updated_at"] = _timestamp(completed)
            self._write_locked()
            return True

    def record_turn(
        self,
        *,
        query: str,
        provider: str,
        model: str | None,
        started_time_s: float,
        response: dict[str, Any] | None = None,
        history: dict[str, Any] | None = None,
        error: BaseException | None = None,
        progress_events: list[dict[str, Any]] | None = None,
        completed_time_s: float | None = None,
        task: str | None = None,
    ) -> int:
        turn_index = self.begin_turn(
            query=query,
            provider=provider,
            model=model,
            started_time_s=started_time_s,
            task=task,
        )
        self.finish_turn(
            turn_index,
            started_time_s=started_time_s,
            response=response,
            history=history,
            error=error,
            progress_events=progress_events,
            completed_time_s=completed_time_s,
        )
        return turn_index

    def close(self, *, ended_time_s: float | None = None) -> None:
        ended = time.time() if ended_time_s is None else float(ended_time_s)
        with self._lock:
            self._document["ended_at"] = _timestamp(ended)
            self._document["updated_at"] = _timestamp(ended)
            self._write_locked()

    def _write_locked(self) -> None:
        _atomic_write_json(self.path, self._document)
        _atomic_write_json(self.latest_path, self._document)
