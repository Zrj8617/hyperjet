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

from scripts import plot_clean_metrics


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = plot_clean_metrics.build_arg_parser()
    args = parser.parse_args(["--run-dir", "dummy", "--prefix", "x_", "--format", "png", "--smooth-window", "3", "--no-show"])
    _assert(args.prefix == "x_", "CLI should parse --prefix.")
    _assert(args.format == "png", "CLI should parse --format png.")
    _assert(args.smooth_window == 3, "CLI should parse --smooth-window.")

    source = (ROOT / "scripts" / "plot_clean_metrics.py").read_text(encoding="utf-8")
    for token in ["PLOTTING_MODULE", "plot_assignment_rl_training", "plot_dagaware_results"]:
        _assert(token not in source, f"clean plotting should not reference legacy plotting token: {token}")
    for token in [
        "Average_DAG_flowtime",
        "DAG_completion_rate",
        "DAG_throughput",
        "Average_critical_path_task_completion_delay",
        "Energy_per_completed_DAG",
        "invalid_assignment_rate",
        "ppo_value_loss",
        "ppo_offloading_action_value_loss",
    ]:
        _assert(token in source, f"clean plotting source should reference clean metric: {token}")

    with _workspace_temp_dir("plotting") as tmp_dir:
        train_dir = Path(tmp_dir) / "train_run"
        eval_dir = Path(tmp_dir) / "eval_run"
        train_dir.mkdir(parents=True)
        eval_dir.mkdir(parents=True)
        _write_jsonl(train_dir / "train_metrics.jsonl", _fake_train_rows())
        _write_jsonl(eval_dir / "eval_metrics.jsonl", _fake_eval_rows())
        (eval_dir / "eval_summary.json").write_text(json.dumps(_fake_eval_summary(), indent=2), encoding="utf-8")

        train_paths = plot_clean_metrics.resolve_inputs(parser.parse_args(["--run-dir", str(train_dir)]))
        eval_paths = plot_clean_metrics.resolve_inputs(parser.parse_args(["--run-dir", str(eval_dir)]))
        _assert(train_paths["output_dir"] == train_dir / "plots", "default train output should be run_dir/plots.")
        _assert(eval_paths["output_dir"] == eval_dir / "plots", "default eval output should be run_dir/plots.")
        _assert(plot_clean_metrics.expected_plot_files("train")[0] == "train_reward.png", "train filenames should be stable.")
        _assert(plot_clean_metrics.expected_plot_files("eval")[0] == "eval_summary.png", "eval filenames should be stable.")

        if not _matplotlib_available():
            code = plot_clean_metrics.main(["--run-dir", str(train_dir), "--no-show"])
            _assert(code == 2, "missing matplotlib should produce a clear plotting-unavailable code.")
            print("smoke_clean_plotting passed; matplotlib plotting branch skipped")
            return

        train_code = plot_clean_metrics.main(["--run-dir", str(train_dir), "--no-show"])
        eval_code = plot_clean_metrics.main(["--run-dir", str(eval_dir), "--no-show"])
        _assert(train_code == 0, "train plotting should succeed when matplotlib is available.")
        _assert(eval_code == 0, "eval plotting should succeed when matplotlib is available.")
        for filename in plot_clean_metrics.expected_plot_files("train"):
            _assert((train_dir / "plots" / filename).is_file(), f"missing train plot {filename}.")
        for filename in plot_clean_metrics.expected_plot_files("eval"):
            _assert((eval_dir / "plots" / filename).is_file(), f"missing eval plot {filename}.")

    print("smoke_clean_plotting passed")


def _fake_train_rows() -> list[dict[str, object]]:
    rows = []
    for idx in range(1, 6):
        rows.append(
            {
                "episode": 0,
                "global_slot": idx,
                "reward": float(idx),
                "reward_time_penalty": -0.1 * idx,
                "reward_task_energy_penalty": -0.05 * idx,
                "reward_movement_energy_penalty": -0.01 * idx,
                "reward_completed_dag_bonus": 0.2 * idx,
                "DAG_completion_rate": 0.1 * idx,
                "DAG_throughput": 0.01 * idx,
                "Average_DAG_flowtime": 5.0 + idx,
                "Average_critical_path_task_completion_delay": 2.0 + idx,
                "Energy_per_completed_DAG": 10.0 + idx,
                "invalid_assignment_rate": 0.0,
                "action_executed_rate": 1.0,
                "movement_action_distribution": {"hover": 0.5, "+x": 0.5},
                "offloading_action_count": idx,
                "ppo_movement_loss": 0.1 * idx,
                "ppo_offloading_loss": 0.2 * idx,
                "ppo_value_loss": 0.3 * idx,
                "ppo_offloading_action_value_loss": 0.04 * idx,
                "ppo_movement_entropy": 0.01 * idx,
                "ppo_offloading_entropy": 0.02 * idx,
            }
        )
    return rows


def _fake_eval_rows() -> list[dict[str, object]]:
    rows = []
    for episode in range(3):
        rows.append(
            {
                "episode": episode,
                "generated_DAG_count": 2.0,
                "completed_DAG_count": 1.0,
                "DAG_completion_rate": 0.5,
                "Average_DAG_flowtime": 10.0 + episode,
                "DAG_throughput": 0.02,
                "Average_critical_path_task_completion_delay": 3.0,
                "Energy_per_completed_DAG": 4.0,
                "total_executed_slots": 5,
                "arrival_slots_executed": 3,
                "drain_slots_executed": 2,
                "invalid_assignment_rate": 0.0,
                "action_executed_rate": 1.0,
                "movement_action_distribution": {"hover": 0.4, "+y": 0.6},
                "offloading_action_count": 2,
            }
        )
    return rows


def _fake_eval_summary() -> dict[str, object]:
    return {
        "DAG_completion_rate": 0.5,
        "Average_DAG_flowtime": 11.0,
        "DAG_throughput": 0.02,
        "Average_critical_path_task_completion_delay": 3.0,
        "Energy_per_completed_DAG": 4.0,
        "invalid_assignment_rate": 0.0,
        "action_executed_rate": 1.0,
        "movement_action_distribution": {"hover": 0.4, "+y": 0.6},
        "arrival_slots_executed": 9,
        "drain_slots_executed": 6,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class _workspace_temp_dir:
    def __init__(self, name: str) -> None:
        self.path = ROOT / ".codex_tmp_clean_plotting" / f"{name}_{os.getpid()}_{np.random.randint(0, 1_000_000)}"

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        parent = ROOT / ".codex_tmp_clean_plotting"
        if parent.exists():
            try:
                if not any(parent.iterdir()):
                    parent.rmdir()
            except PermissionError:
                pass


if __name__ == "__main__":
    main()
