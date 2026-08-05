#!/usr/bin/env python3
"""Run a Roomie bag replay and emit a deterministic JSON metrics report.

Commands run after the caller has sourced the required ROS workspaces. The
same program can analyze an existing RunLogger directory without launching a
bag.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


_FIELD_RE = re.compile(r"(?:^|\s)([A-Za-z][A-Za-z0-9_.-]*)=([^\s]+)")
_DROP_RE = re.compile(
    r"(?:drop|dropped|replaced|supersed|backpressure|deadline|expired|skip|"
    r"evict|not_joined|rejected|not_enqueued)",
    re.IGNORECASE,
)


def _scalar(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        number = float(value)
        return number if math.isfinite(number) else value
    except ValueError:
        return value


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _parse_log(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            timestamp = None
            message = line
            try:
                timestamp = dt.datetime.strptime(
                    line[:23], "%Y-%m-%d %H:%M:%S.%f"
                ).replace(tzinfo=dt.timezone.utc)
                message = line[24:]
            except (ValueError, IndexError):
                pass
            yield {
                "module": path.stem,
                "line": line_number,
                "timestamp": timestamp,
                "event": message.split(maxsplit=1)[0] if message else "",
                "message": message,
                "fields": {
                    match.group(1): _scalar(match.group(2))
                    for match in _FIELD_RE.finditer(message)
                },
            }


def analyze_run(run_directory: Path) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    if not run_directory.is_dir():
        raise ValueError(f"RunLogger directory does not exist: {run_directory}")
    log_files = sorted(run_directory.glob("*.log"))
    records = [record for path in log_files for record in _parse_log(path)]
    timestamps = [record["timestamp"] for record in records if record["timestamp"]]

    events: collections.Counter[str] = collections.Counter()
    drops: collections.Counter[str] = collections.Counter()
    artifact_events: collections.Counter[str] = collections.Counter()
    latencies: dict[str, list[float]] = collections.defaultdict(list)
    queues: dict[str, list[float]] = collections.defaultdict(list)
    latest: dict[str, Any] = {}
    map_timeline: list[dict[str, Any]] = []
    incremental_blocks = 0
    full_blocks = 0

    for record in records:
        module = record["module"]
        event = record["event"]
        fields = record["fields"]
        events[f"{module}.{event}"] += 1
        # Status lines contain counter names such as skip_rate; only the event
        # identity denotes an admission decision.
        if _DROP_RE.search(event):
            drops[str(fields.get("reason", event or "unspecified"))] += 1
        for key, value in fields.items():
            qualified = f"{module}.{key}"
            latest[qualified] = value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Absolute wall-clock fields are provenance, not durations.
                if key.endswith("_ms") and not key.endswith("_unix_ms"):
                    latencies[qualified].append(float(value))
                if "queue" in key or key.endswith("_depth") or key.startswith("pending_"):
                    queues[qualified].append(float(value))

        selected = {
            key: fields[key]
            for key in (
                "map_version",
                "latest_map_version",
                "map_revision",
                "surface_revision",
                "source_map_revision",
                "dirty_blocks",
                "cached_blocks",
                "mode",
            )
            if key in fields
        }
        if selected and module in {"map", "map_thread", "geometry"}:
            selected.update(
                timestamp=(
                    record["timestamp"].isoformat()
                    if record["timestamp"]
                    else None
                ),
                module=module,
                event=event,
            )
            map_timeline.append(selected)
        if fields.get("mode") == "incremental":
            incremental_blocks += int(fields.get("dirty_blocks", 0))
        elif fields.get("mode") == "full":
            full_blocks += int(fields.get("cached_blocks", fields.get("dirty_blocks", 0)))

        if any(token in module for token in ("artifact", "embedding", "snapshot", "query")):
            artifact_events[f"{module}.{event}"] += 1

    latency_summary = {
        key: {
            "count": len(values),
            "p50": _quantile(values, 0.50),
            "p95": _quantile(values, 0.95),
            "p99": _quantile(values, 0.99),
            "max": max(values),
        }
        for key, values in sorted(latencies.items())
    }
    queue_summary = {
        key: {"samples": len(values), "latest": values[-1], "max": max(values)}
        for key, values in sorted(queues.items())
    }
    total_blocks = incremental_blocks + full_blocks
    latest_counters = {
        key: value
        for key, value in sorted(latest.items())
        if any(
            token in key
            for token in (
                "revision",
                "lag",
                "frames",
                "responses",
                "errors",
                "scheduled",
                "accepted",
                "superseded",
                "fresh",
                "generation",
            )
        )
    }
    return {
        "schema": "roomie.bag_replay.metrics.v1",
        "run_directory": str(run_directory),
        "log_files": [path.name for path in log_files],
        "record_count": len(records),
        "time_window": {
            "first": min(timestamps).isoformat() if timestamps else None,
            "last": max(timestamps).isoformat() if timestamps else None,
        },
        "frame_admission": {"drop_reason_counts": dict(sorted(drops.items()))},
        "map_surface": {
            "timeline": map_timeline,
            "incremental_blocks": incremental_blocks,
            "full_blocks": full_blocks,
            "incremental_block_ratio": (
                incremental_blocks / total_blocks if total_blocks else None
            ),
        },
        "latency_ms": latency_summary,
        "queues": queue_summary,
        "latest_counters": latest_counters,
        "artifact_freshness_events": dict(sorted(artifact_events.items())),
        "event_counts": dict(sorted(events.items())),
    }


def _stop_process(process: subprocess.Popen[Any] | None, grace: float) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        # Signal only the session leader first.  Supervisors such as
        # ``ros2 launch`` forward SIGINT to their children themselves; sending
        # it to the entire process group would make those children receive the
        # signal twice and can abort their graceful checkpoint path.
        process.send_signal(signal.SIGINT)
        process.wait(timeout=grace)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _launch(command: str, output_path: Path) -> tuple[subprocess.Popen[Any], Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream = output_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", "-lc", command],
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, stream


def _newest_run(log_root: Path, started_ns: int) -> Path:
    candidates = [
        path
        for path in log_root.glob("run_*")
        if path.is_dir() and path.stat().st_mtime_ns >= started_ns
    ]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)
    latest = log_root / "latest"
    if latest.exists():
        return latest.resolve()
    raise ValueError(f"No replay RunLogger directory appeared under {log_root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Analyze an existing run")
    parser.add_argument("--pipeline-command", help="Command that starts Roomie")
    parser.add_argument("--player-command", help="Command that replays the bag")
    parser.add_argument("--log-root", type=Path, help="RunLogger root for launch mode")
    parser.add_argument("--startup-seconds", type=float, default=3.0)
    parser.add_argument(
        "--player-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Stop a player that keeps its UI/event loop alive after EOF; "
            "zero waits indefinitely"
        ),
    )
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--shutdown-grace-seconds", type=float, default=10.0)
    parser.add_argument("--process-log-dir", type=Path, default=Path("/tmp/roomie_replay"))
    parser.add_argument("--report", type=Path, help="Write JSON here; stdout otherwise")
    args = parser.parse_args(argv)

    launch_mode = bool(args.pipeline_command or args.player_command or args.log_root)
    if launch_mode and not all((args.pipeline_command, args.player_command, args.log_root)):
        parser.error("launch mode requires --pipeline-command, --player-command, and --log-root")
    if args.run_dir and launch_mode:
        parser.error("--run-dir cannot be combined with launch mode")
    if not args.run_dir and not launch_mode:
        parser.error("provide --run-dir or all launch-mode arguments")

    process_metadata = None
    if launch_mode:
        started_ns = time.time_ns()
        pipeline = player = None
        pipeline_stream = player_stream = None
        player_code = None
        player_termination_reason = "exited"
        try:
            pipeline, pipeline_stream = _launch(
                args.pipeline_command, args.process_log_dir / "pipeline.out"
            )
            time.sleep(max(0.0, args.startup_seconds))
            if pipeline.poll() is not None:
                raise RuntimeError(
                    f"pipeline exited during startup with code {pipeline.returncode}"
                )
            player, player_stream = _launch(
                args.player_command, args.process_log_dir / "player.out"
            )
            try:
                if args.player_timeout_seconds > 0.0:
                    player_code = player.wait(timeout=args.player_timeout_seconds)
                else:
                    player_code = player.wait()
            except subprocess.TimeoutExpired:
                player_termination_reason = "timeout"
                _stop_process(player, args.shutdown_grace_seconds)
                player_code = player.returncode
            time.sleep(max(0.0, args.settle_seconds))
        finally:
            _stop_process(player, args.shutdown_grace_seconds)
            _stop_process(pipeline, args.shutdown_grace_seconds)
            if player_stream:
                player_stream.close()
            if pipeline_stream:
                pipeline_stream.close()
        run_directory = _newest_run(args.log_root.resolve(), started_ns)
        process_metadata = {
            "pipeline_exit_code": pipeline.returncode if pipeline else None,
            "player_exit_code": player_code,
            "player_termination_reason": player_termination_reason,
            "pipeline_output": str((args.process_log_dir / "pipeline.out").resolve()),
            "player_output": str((args.process_log_dir / "player.out").resolve()),
        }
    else:
        run_directory = args.run_dir

    report = analyze_run(run_directory)
    if process_metadata is not None:
        report["processes"] = process_metadata
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
