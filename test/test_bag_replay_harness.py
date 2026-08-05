import importlib.util
import json
import shlex
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "roomie_bag_replay_harness.py"
SPEC = importlib.util.spec_from_file_location("roomie_bag_replay_harness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_machine_readable_metrics_cover_replay_contract(tmp_path):
    (tmp_path / "map.log").write_text(
        "2026-08-04 10:00:00.000 refreshed_nvblox_surface_cache "
        "mode=full dirty_blocks=0 cached_blocks=10 map_version=1 build_ms=8.0\n"
        "2026-08-04 10:00:00.100 refreshed_nvblox_surface_cache "
        "mode=incremental dirty_blocks=2 cached_blocks=11 map_version=2 build_ms=2.0\n",
        encoding="utf-8",
    )
    (tmp_path / "detection.log").write_text(
        "2026-08-04 10:00:00.200 perception_drop reason=deadline "
        "roundtrip_ms=10 queue_depth=3\n"
        "2026-08-04 10:00:00.300 response roundtrip_ms=30 queue_depth=1\n",
        encoding="utf-8",
    )
    (tmp_path / "embedding_runtime.log").write_text(
        "2026-08-04 10:00:00.400 indexed object_id=4 generation=7 fresh=true "
        "completed_at_unix_ms=1785830000000\n",
        encoding="utf-8",
    )

    report = MODULE.analyze_run(tmp_path)
    assert report["schema"] == "roomie.bag_replay.metrics.v1"
    assert report["frame_admission"]["drop_reason_counts"] == {"deadline": 1}
    assert report["map_surface"]["incremental_blocks"] == 2
    assert report["map_surface"]["full_blocks"] == 10
    assert report["map_surface"]["incremental_block_ratio"] == 2 / 12
    latency = report["latency_ms"]["detection.roundtrip_ms"]
    assert latency["count"] == 2
    assert latency["p50"] == 20
    assert latency["p99"] > 29
    assert report["queues"]["detection.queue_depth"]["max"] == 3
    assert report["artifact_freshness_events"] == {
        "embedding_runtime.indexed": 1
    }
    assert "embedding_runtime.completed_at_unix_ms" not in report["latency_ms"]
    json.dumps(report)


def test_launch_timeout_stops_gui_style_player_and_still_writes_report(tmp_path):
    log_root = tmp_path / "logs"
    process_logs = tmp_path / "process"
    report_path = tmp_path / "report.json"
    run_dir = log_root / "run_timeout"
    pipeline_program = (
        "from pathlib import Path; import time; "
        f"p=Path({str(run_dir)!r}); p.mkdir(parents=True); "
        "(p/'map.log').write_text('status map_revision=1\\n'); "
        "time.sleep(30)"
    )
    player_program = "import time; time.sleep(30)"
    pipeline_command = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(pipeline_program)}"
    )
    player_command = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(player_program)}"
    )

    result = MODULE.main(
        [
            "--pipeline-command",
            pipeline_command,
            "--player-command",
            player_command,
            "--log-root",
            str(log_root),
            "--startup-seconds",
            "0.05",
            "--player-timeout-seconds",
            "0.05",
            "--settle-seconds",
            "0",
            "--shutdown-grace-seconds",
            "0.2",
            "--process-log-dir",
            str(process_logs),
            "--report",
            str(report_path),
        ]
    )

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["processes"]["player_termination_reason"] == "timeout"
    assert report["processes"]["player_exit_code"] is not None
    assert report["event_counts"]["map.status"] == 1


def test_graceful_stop_signals_only_the_session_leader(monkeypatch):
    class GracefulSupervisor:
        pid = 1234

        def __init__(self):
            self.signals = []
            self.finished = False

        def poll(self):
            return 0 if self.finished else None

        def send_signal(self, requested_signal):
            self.signals.append(requested_signal)

        def wait(self, timeout=None):
            self.finished = True
            return 0

    supervisor = GracefulSupervisor()

    def unexpected_process_group_signal(*_args):
        raise AssertionError("graceful shutdown must not signal the process group")

    monkeypatch.setattr(MODULE.os, "killpg", unexpected_process_group_signal)
    MODULE._stop_process(supervisor, grace=0.2)

    assert supervisor.signals == [MODULE.signal.SIGINT]
    assert supervisor.poll() == 0
