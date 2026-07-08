from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import sys

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
    "drain_slots_executed",
    "invalid_assignment_count",
    "invalid_assignment_rate",
    "action_executed_rate",
    "movement_action_distribution",
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
        ]
    )
    _assert(args.arrival_steps == 3, "CLI should parse --arrival-steps.")
    _assert(args.max_drain_steps == 4, "CLI should parse --max-drain-steps.")
    _assert(args.deterministic is True, "deterministic evaluation should be default.")
    _assert(args.no_render is True, "evaluation should support no-render mode.")

    config_payload = eval_clean_mainline.build_eval_config(args)
    _assert(
        config_payload["protocol"]["throughput_denominator"] == "total_executed_slots * TIME_SLOT_DURATION",
        "throughput denominator should use total executed slots.",
    )
    _assert("DAG arrivals disabled" in config_payload["protocol"]["drain_phase"], "drain protocol should disable DAG arrivals.")
    _assert(config_payload["protocol"]["default_action"] == "masked_argmax_deterministic", "default action should be deterministic masked argmax.")

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
                    "drain_slots_executed": 2,
                    "invalid_assignment_count": 0.0,
                    "invalid_assignment_rate": 0.0,
                    "action_executed_rate": 1.0,
                    "movement_action_distribution": {"hover": 1.0},
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
                    "drain_slots_executed": 900,
                    "invalid_assignment_count": 0.0,
                    "invalid_assignment_rate": 0.0,
                    "action_executed_rate": 1.0,
                    "movement_action_distribution": {"hover": 0.0, "+x": 1.0},
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
