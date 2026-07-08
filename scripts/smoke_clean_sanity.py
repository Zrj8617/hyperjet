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
        "completion_rate",
        "throughput",
        "average_DAG_flowtime",
        "energy_per_completed_DAG",
        "invalid_assignment_rate",
        "action_executed_rate",
        "movement_hover_rate",
        "offloading_action_count",
        "checkpoint_path",
        "pass_fail",
        "overall_pass",
    ]:
        _assert(key in schema, f"sanity report schema missing {key}.")

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
