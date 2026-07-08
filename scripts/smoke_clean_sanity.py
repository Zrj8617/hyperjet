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

from scripts import run_clean_sanity


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = run_clean_sanity.build_arg_parser()
    default_args = parser.parse_args([])
    _assert(default_args.episodes == 100, "default sanity episodes should be 100, not full experiment scale.")
    _assert(default_args.max_steps_per_episode == 200, "default sanity steps should be 200.")
    _assert(default_args.rollout_horizon == 20, "default sanity rollout horizon should be 20.")
    _assert(default_args.seed == 0, "default sanity seed should be 0.")
    _assert(default_args.episodes <= 100, "sanity defaults must not be full experiment scale.")

    skip_args = parser.parse_args(["--skip-smoke", "--skip-eval", "--skip-plot"])
    commands = run_clean_sanity.build_workflow_commands(skip_args, ROOT / "dummy_sanity")
    command_text = "\n".join(item["command"] for item in commands)
    _assert("scripts/smoke_clean_server_torch.py" not in command_text, "--skip-smoke should remove torch smoke command.")
    _assert("scripts/eval_clean_mainline.py" not in command_text, "--skip-eval should remove eval command.")
    _assert("scripts/plot_clean_metrics.py" not in command_text, "--skip-plot should remove plot command.")
    _assert("scripts/train_clean_mainline.py" in command_text, "sanity should call clean train entrypoint.")

    args = parser.parse_args(["--episodes", "50", "--max-steps-per-episode", "120", "--rollout-horizon", "12"])
    commands = run_clean_sanity.build_workflow_commands(args, ROOT / "dummy_sanity")
    command_text = "\n".join(item["command"] for item in commands)
    for token in [
        "scripts/smoke_clean_server_torch.py",
        "scripts/train_clean_mainline.py --smoke",
        "scripts/train_clean_mainline.py --episodes 50",
        "scripts/eval_clean_mainline.py",
        "scripts/plot_clean_metrics.py",
    ]:
        _assert(token in command_text, f"sanity workflow missing clean command token: {token}")
    for forbidden in ["train_clean_assignment_" + "mappo", "clean_" + "mappo", "clean_assignment_" + "policy", " main.py", " train.py", " tune.py"]:
        _assert(forbidden not in command_text, f"sanity command should not reference legacy token: {forbidden}")

    schema = run_clean_sanity.report_schema()
    for key in [
        "commands",
        "return_codes",
        "train_run_dir",
        "eval_run_dir",
        "plot_paths",
        "final_reward",
        "recent_reward",
        "generated_DAG_count",
        "completed_DAG_count",
        "completion_rate",
        "throughput",
        "average_DAG_flowtime",
        "energy_per_completed_DAG",
        "invalid_assignment_rate",
        "action_executed_rate",
        "movement_action_distribution",
        "movement_hover_rate",
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
        "train_final_metrics_source",
        "eval_final_metrics_source",
        "checkpoint_path",
        "pass_fail",
        "overall_pass",
    ]:
        _assert(key in schema, f"sanity report schema missing {key}.")

    zero_train_report = run_clean_sanity.report_schema()
    run_clean_sanity._merge_train_metrics(
        zero_train_report,
        {"latest_info": {"generated_dag_count": 5, "completed_dag_count": 0}},
        [
            {
                "reward": -1.0,
                "generated_DAG_count": 5,
                "completed_DAG_count": 0,
                "DAG_completion_rate": 0.0,
                "DAG_throughput": 0.0,
                "Average_DAG_flowtime": 123.0,
                "Energy_per_completed_DAG": 1772.686,
                "movement_action_distribution": {"hover": 0.0, "+x": 1.0},
            }
        ],
    )
    _assert(zero_train_report["generated_DAG_count"] == 5, "generated DAG count should be preserved.")
    _assert(zero_train_report["completed_DAG_count"] == 0, "completed DAG count should be preserved.")
    _assert(zero_train_report["average_DAG_flowtime"] is None, "zero completed DAGs should have no flowtime.")
    _assert(zero_train_report["energy_per_completed_DAG"] is None, "zero completed DAGs should have no energy per completed DAG.")
    _assert(zero_train_report["movement_action_distribution"].get("+x") == 1.0, "movement distribution should be reported.")
    _assert(zero_train_report["train_final_metrics_source"]["train_metrics_jsonl"] == "last row", "train metrics source should be reported.")

    zero_eval_report = run_clean_sanity.report_schema()
    run_clean_sanity._merge_eval_metrics(
        zero_eval_report,
        {
            "generated_DAG_count": 3,
            "completed_DAG_count": 0,
            "DAG_completion_rate": 0.0,
            "DAG_throughput": 0.0,
            "Average_DAG_flowtime": 99.0,
            "Energy_per_completed_DAG": 88.0,
            "movement_action_distribution": {"hover": 0.25, "+y": 0.75},
            "final_active_DAG_count": 3,
            "final_active_task_count": 12,
            "task_lifecycle_counts": {"READY_UNSCHEDULED": 4, "IN_SERVICE": 3, "RETURNING": 1, "COMPLETED": 4},
            "service_phase_counts": {"QUEUED": 1, "COMPUTING": 2},
            "ready_task_count_mean": 2.5,
            "ready_task_count_max": 5,
            "skipped_ready_due_to_no_legal_candidate_count": 7,
            "assignment_buffer_entry_count": 9,
            "successfully_committed_assignment_count": 8,
            "reward_completed_task_count": 4,
            "completed_non_sink_task_count": 4,
            "returning_sink_task_count": 1,
            "completed_sink_task_count": 0,
            "unfinished_DAG_progress_samples": [{"dag_id": "dag_1", "total_tasks": 4}],
            "executor_queue_summary": {"mean_queue_length": 1.0, "max_queue_length": 3},
            "drain_end_reason": "max_drain_steps_reached",
        },
    )
    _assert(zero_eval_report["average_DAG_flowtime"] is None, "zero completed eval DAGs should have no flowtime.")
    _assert(zero_eval_report["energy_per_completed_DAG"] is None, "zero completed eval DAGs should have no energy.")
    _assert(zero_eval_report["eval_final_metrics_source"]["eval_summary"] == "summary", "eval metrics source should be reported.")
    _assert(zero_eval_report["final_active_DAG_count"] == 3, "sanity report should carry final active DAG count.")
    _assert(zero_eval_report["task_lifecycle_counts"]["READY_UNSCHEDULED"] == 4, "sanity report should carry lifecycle counts.")
    _assert(zero_eval_report["drain_end_reason"] == "max_drain_steps_reached", "sanity report should carry drain end reason.")

    source = (ROOT / "scripts" / "run_clean_sanity.py").read_text(encoding="utf-8")
    for forbidden in ["train_clean_assignment_" + "mappo", "clean_" + "mappo", "clean_assignment_" + "policy"]:
        _assert(forbidden not in source, f"run_clean_sanity should not reference legacy clean token: {forbidden}")

    with _workspace_temp_dir("sanity") as tmp_dir:
        smoke_args = parser.parse_args(
            [
                "--episodes",
                "1",
                "--max-steps-per-episode",
                "1",
                "--rollout-horizon",
                "1",
                "--output-dir",
                str(tmp_dir),
                "--run-name",
                "sanity_smoke",
                "--skip-smoke",
                "--skip-eval",
                "--skip-plot",
            ]
        )
        report = run_clean_sanity.run_sanity(smoke_args)
        report_path = Path(report["run_dir"]) / "sanity_report.json"
        _assert(report_path.is_file(), "sanity helper should always write sanity_report.json.")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if _torch_available():
            _assert("short_sanity_training" in payload["return_codes"], "torch smoke should attempt short sanity training.")
        else:
            _assert(payload["return_codes"].get("torch_required") == 2, "non-torch sanity should exit as unavailable.")
            _assert(payload["overall_pass"] is False, "non-torch sanity must not pretend to pass.")
            print("smoke_clean_sanity passed; torch sanity branch skipped")
            return

    print("smoke_clean_sanity passed")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class _workspace_temp_dir:
    def __init__(self, name: str) -> None:
        self.path = ROOT / ".codex_tmp_clean_sanity" / f"{name}_{os.getpid()}_{np.random.randint(0, 1_000_000)}"

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        parent = ROOT / ".codex_tmp_clean_sanity"
        if parent.exists():
            try:
                if not any(parent.iterdir()):
                    parent.rmdir()
            except PermissionError:
                pass


if __name__ == "__main__":
    main()
