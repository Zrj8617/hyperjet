from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_clean_mainline


REQUIRED_METRIC_FIELDS = [
    "generated_DAG_count",
    "completed_DAG_count",
    "DAG_completion_rate",
    "Average_DAG_flowtime",
    "DAG_throughput",
    "Average_critical_path_task_completion_delay",
    "Energy_per_completed_DAG",
    "total_executed_slots",
    "arrival_slots_executed",
    "arrival_generated_DAG_count",
    "arrival_completed_DAG_count",
    "arrival_DAG_completion_rate",
    "arrival_active_DAG_count",
    "arrival_active_task_count",
    "drain_slots_executed",
    "invalid_assignment_count",
    "invalid_assignment_rate",
    "action_executed_rate",
    "movement_action_distribution",
    "movement_frozen",
    "freeze_ue_mobility",
    "initial_hotspot_ue_count_mean",
    "offloading_action_count",
    "final_active_DAG_count",
    "final_active_task_count",
    "task_lifecycle_counts",
    "service_phase_counts",
    "ready_task_count_mean",
    "ready_task_count_max",
    "skipped_ready_due_to_no_legal_candidate_count",
    "assignment_buffer_entry_count",
    "successfully_committed_assignment_count",
    "reward_completed_task_count",
    "completed_non_sink_task_count",
    "returning_sink_task_count",
    "completed_sink_task_count",
    "unfinished_DAG_progress_samples",
    "executor_queue_summary",
    "drain_end_reason",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = eval_clean_mainline.build_arg_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            str(ROOT / "missing_clean_eval_checkpoint.pt"),
            "--episodes",
            "2",
            "--arrival-steps",
            "3",
            "--max-drain-steps",
            "4",
            "--output-dir",
            str(ROOT / ".codex_tmp_eval_entrypoint"),
            "--run-name",
            "eval_smoke",
            "--no-render",
            "--freeze-movement",
        ]
    )
    _assert(args.arrival_steps == 3, "CLI should parse --arrival-steps.")
    _assert(args.max_drain_steps == 4, "CLI should parse --max-drain-steps.")
    _assert(args.deterministic is True, "deterministic evaluation should be default.")
    _assert(args.no_render is True, "evaluation should support no-render mode.")
    _assert(args.freeze_movement is True, "evaluation should parse --freeze-movement.")

    config_payload = eval_clean_mainline.build_eval_config(args)
    _assert(
        config_payload["protocol"]["throughput_denominator"] == "total_executed_slots * TIME_SLOT_DURATION",
        "throughput denominator should use total executed slots.",
    )
    _assert("DAG arrivals disabled" in config_payload["protocol"]["drain_phase"], "drain protocol should disable DAG arrivals.")
    _assert(config_payload["protocol"]["default_action"] == "masked_argmax_deterministic", "default action should be deterministic masked argmax.")
    _assert(config_payload["protocol"]["movement_mode"] == "forced_hover", "frozen eval config should record forced hover.")

    sample_summary = eval_clean_mainline._aggregate_summary(
        {
            "episodes": [
                {
                    "generated_DAG_count": 2.0,
                    "completed_DAG_count": 1.0,
                    "DAG_completion_rate": 0.5,
                    "Average_DAG_flowtime": 10.0,
                    "DAG_throughput": 1.0 / 25.0,
                    "Average_critical_path_task_completion_delay": 5.0,
                    "Energy_per_completed_DAG": 3.0,
                    "total_executed_slots": 5,
                    "arrival_slots_executed": 3,
                    "arrival_generated_DAG_count": 2.0,
                    "arrival_completed_DAG_count": 1.0,
                    "arrival_DAG_completion_rate": 0.5,
                    "arrival_active_DAG_count": 1,
                    "arrival_active_task_count": 2,
                    "drain_slots_executed": 2,
                    "invalid_assignment_count": 0.0,
                    "invalid_assignment_rate": 0.0,
                    "action_executed_rate": 1.0,
                    "movement_action_distribution": {"hover": 1.0},
                    "movement_frozen": True,
                    "offloading_action_count": 1,
                    "final_active_DAG_count": 0,
                    "final_active_task_count": 0,
                    "task_lifecycle_counts": {"COMPLETED": 1},
                    "service_phase_counts": {"QUEUED": 0},
                    "ready_task_count_mean": 1.0,
                    "ready_task_count_max": 2,
                    "skipped_ready_due_to_no_legal_candidate_count": 0,
                    "assignment_buffer_entry_count": 1,
                    "successfully_committed_assignment_count": 1,
                    "reward_completed_task_count": 1,
                    "completed_non_sink_task_count": 0,
                    "returning_sink_task_count": 0,
                    "completed_sink_task_count": 1,
                    "unfinished_DAG_progress_samples": [],
                    "executor_queue_summary": {
                        "mean_queue_length": 0.0,
                        "max_queue_length": 0,
                        "total_queued_workload": 0.0,
                        "max_available_time": 1.0,
                    },
                    "drain_end_reason": "all_completed",
                }
            ],
            "movement_counts": {"hover": 1.0},
        },
        episode_count=1,
    )
    for field in REQUIRED_METRIC_FIELDS:
        _assert(field in sample_summary, f"eval summary schema should include {field}.")
    _assert(sample_summary["total_executed_slots"] == 5, "summary should preserve total executed slots.")
    _assert(np.isclose(sample_summary["DAG_throughput"], 1.0 / 25.0), "throughput should reflect total executed time.")
    _assert(sample_summary["drain_end_reason"] == "all_completed", "summary should include drain end reason.")
    _assert(sample_summary["arrival_DAG_completion_rate"] == 0.5, "summary should preserve arrival completion.")
    _assert(sample_summary["movement_frozen"] is True, "summary should preserve frozen movement mode.")

    weighted_rows = []
    for generated, completed in [(2.0, 1.0), (8.0, 2.0)]:
        row = dict(sample_summary)
        row.update(
            {
                "arrival_generated_DAG_count": generated,
                "arrival_completed_DAG_count": completed,
                "arrival_DAG_completion_rate": completed / generated,
            }
        )
        weighted_rows.append(row)
    weighted_summary = eval_clean_mainline._aggregate_summary(
        {"episodes": weighted_rows, "movement_counts": {"hover": 2.0}},
        episode_count=2,
    )
    _assert(
        np.isclose(weighted_summary["arrival_DAG_completion_rate"], 3.0 / 10.0),
        "arrival completion should aggregate counts rather than average episode rates.",
    )

    class _FreezeProbe:
        movement_observation = SimpleNamespace(uav_ids=[10, 11])

        @property
        def movement_logits(self):
            raise AssertionError("frozen movement must not inspect movement logits")

    frozen_actions = eval_clean_mainline._select_deterministic_movement_actions(
        _FreezeProbe(),
        freeze_movement=True,
    )
    _assert(frozen_actions == {10: 0, 11: 0}, "frozen evaluation should force the configured hover action.")

    metrics_calls = []

    class _ArrivalMetrics:
        def to_info(self, slots, *, total_time_seconds):
            metrics_calls.append((slots, total_time_seconds))
            return {"generated_dag_count": 4.0, "completed_dag_count": 3.0}

    snapshot_env = SimpleNamespace(
        metrics=_ArrivalMetrics(),
        task_manager=SimpleNamespace(
            jobs={"done": SimpleNamespace(completed=True), "active": SimpleNamespace(completed=False)},
            tasks={"done": SimpleNamespace(state="COMPLETED"), "active": SimpleNamespace(state="IN_SERVICE")},
        ),
    )
    arrival_snapshot = eval_clean_mainline._snapshot_arrival_metrics(env=snapshot_env, arrival_slots=7)
    _assert(arrival_snapshot["arrival_DAG_completion_rate"] == 0.75, "arrival snapshot completion mismatch.")
    _assert(arrival_snapshot["arrival_active_DAG_count"] == 1, "arrival snapshot active DAG count mismatch.")
    _assert(arrival_snapshot["arrival_active_task_count"] == 1, "arrival snapshot active task count mismatch.")
    _assert(metrics_calls == [(7, 7.0 * eval_clean_mainline.config.TIME_SLOT_DURATION)], "arrival snapshot time boundary mismatch.")

    class _ModernTorch:
        calls = []

        @classmethod
        def load(cls, path, **kwargs):
            cls.calls.append((path, kwargs))
            return {"format": "modern"}

    modern_payload = eval_clean_mainline._load_trusted_checkpoint(_ModernTorch, Path("trusted.pt"))
    _assert(modern_payload == {"format": "modern"}, "trusted checkpoint payload mismatch.")
    _assert(_ModernTorch.calls[0][1].get("weights_only") is False, "PyTorch 2.6 load should opt out of weights-only mode.")

    class _LegacyTorch:
        calls = []

        @classmethod
        def load(cls, path, **kwargs):
            cls.calls.append((path, kwargs))
            if "weights_only" in kwargs:
                raise TypeError("load() got an unexpected keyword argument 'weights_only'")
            return {"format": "legacy"}

    legacy_payload = eval_clean_mainline._load_trusted_checkpoint(_LegacyTorch, Path("trusted.pt"))
    _assert(legacy_payload == {"format": "legacy"}, "legacy checkpoint fallback payload mismatch.")
    _assert(len(_LegacyTorch.calls) == 2, "legacy checkpoint load should retry exactly once.")
    _assert("weights_only" not in _LegacyTorch.calls[1][1], "legacy retry should omit weights_only.")

    zero_completed_summary = eval_clean_mainline._aggregate_summary(
        {
            "episodes": [
                {
                    "generated_DAG_count": 180.0,
                    "completed_DAG_count": 0.0,
                    "DAG_completion_rate": 0.0,
                    "Average_DAG_flowtime": None,
                    "DAG_throughput": 0.0,
                    "Average_critical_path_task_completion_delay": 235.21,
                    "Energy_per_completed_DAG": None,
                    "total_executed_slots": 1500,
                    "arrival_slots_executed": 600,
                    "arrival_generated_DAG_count": 180.0,
                    "arrival_completed_DAG_count": 0.0,
                    "arrival_DAG_completion_rate": 0.0,
                    "arrival_active_DAG_count": 180,
                    "arrival_active_task_count": 500,
                    "drain_slots_executed": 900,
                    "invalid_assignment_count": 0.0,
                    "invalid_assignment_rate": 0.0,
                    "action_executed_rate": 1.0,
                    "movement_action_distribution": {"hover": 0.0, "+x": 1.0},
                    "movement_frozen": False,
                    "offloading_action_count": 465,
                    "final_active_DAG_count": 180,
                    "final_active_task_count": 500,
                    "task_lifecycle_counts": {"READY_UNSCHEDULED": 100, "IN_SERVICE": 50, "COMPLETED": 350},
                    "service_phase_counts": {"QUEUED": 10, "COMPUTING": 40},
                    "ready_task_count_mean": 2.0,
                    "ready_task_count_max": 8,
                    "skipped_ready_due_to_no_legal_candidate_count": 3,
                    "assignment_buffer_entry_count": 465,
                    "successfully_committed_assignment_count": 465,
                    "reward_completed_task_count": 350,
                    "completed_non_sink_task_count": 330,
                    "returning_sink_task_count": 20,
                    "completed_sink_task_count": 0,
                    "unfinished_DAG_progress_samples": [{"dag_id": "dag_1", "total_tasks": 5}],
                    "executor_queue_summary": {
                        "mean_queue_length": 1.0,
                        "max_queue_length": 3,
                        "total_queued_workload": 100.0,
                        "max_available_time": 999.0,
                    },
                    "drain_end_reason": "max_drain_steps_reached",
                }
            ],
            "movement_counts": {"hover": 0.0, "+x": 1.0},
        },
        episode_count=1,
    )
    _assert(zero_completed_summary["Average_DAG_flowtime"] is None, "zero completed DAGs should have no eval flowtime.")
    _assert(zero_completed_summary["Energy_per_completed_DAG"] is None, "zero completed DAGs should have no eval energy.")
    _assert(zero_completed_summary["drain_end_reason"] == "max_drain_steps_reached", "zero-completion diagnostics should keep drain reason.")
    _assert(zero_completed_summary["final_active_DAG_count"] == 180, "zero-completion diagnostics should keep active DAG count.")

    source = (ROOT / "scripts" / "eval_clean_mainline.py").read_text(encoding="utf-8")
    for token in ["clean_" + "mappo", "clean_assignment_" + "policy", "train_clean_assignment_" + "mappo"]:
        _assert(token not in source, f"eval entrypoint should not reference legacy clean token: {token}")
    for token in ["torch.argmax", "deterministic=True", "_dag_arrival_enabled(allow_dag_arrivals)"]:
        _assert(token in source, f"eval entrypoint missing deterministic/drain implementation token: {token}")

    with _workspace_temp_dir("eval") as tmp_dir:
        code = eval_clean_mainline.main(
            [
                "--checkpoint",
                str(ROOT / "missing_clean_eval_checkpoint.pt"),
                "--episodes",
                "1",
                "--arrival-steps",
                "1",
                "--max-drain-steps",
                "1",
                "--output-dir",
                str(tmp_dir),
                "--run-name",
                "main_eval_smoke",
            ]
        )
        _assert(code == 2, "missing torch or missing checkpoint should return a clear unavailable code.")
        run_dirs = [path for path in Path(tmp_dir).iterdir() if path.is_dir()]
        _assert(run_dirs, "eval entrypoint should create run directory and schema files before failing.")
        latest = sorted(run_dirs)[-1]
        _assert((latest / "config.json").is_file(), "eval setup should write config.json.")
        _assert((latest / "eval_summary.json").is_file(), "eval setup should write eval_summary.json.")
        summary_payload = json.loads((latest / "eval_summary.json").read_text(encoding="utf-8"))
        _assert(summary_payload["torch_required_for_evaluation"] is True, "summary should mark torch as required.")

    if _torch_available():
        print("smoke_clean_eval_entrypoint passed; real deterministic checkpoint evaluation not run without a checkpoint")
    else:
        print("smoke_clean_eval_entrypoint passed; torch evaluation branch skipped")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class _workspace_temp_dir:
    def __init__(self, name: str) -> None:
        self.path = ROOT / ".codex_tmp_eval_entrypoint" / f"{name}_{os.getpid()}_{np.random.randint(0, 1_000_000)}"

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        parent = ROOT / ".codex_tmp_eval_entrypoint"
        if parent.exists():
            try:
                if not any(parent.iterdir()):
                    parent.rmdir()
            except PermissionError:
                pass


if __name__ == "__main__":
    main()
