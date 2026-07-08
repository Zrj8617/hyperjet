"""Dry-run launcher template for HyperUAV clean full experiments.

This script intentionally does not start full training by default.  It writes an
experiment plan with clean train/eval/plot commands so the server run can be
reviewed after the smoke and sanity gates pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


BASELINE_ABLATION_STATUS = "not_implemented"
IMPLEMENTED_METHOD = "hypergraph_mainline_only"


def _command(parts: Iterable[Any]) -> str:
    return " ".join(str(part) for part in parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a dry-run launch plan for HyperUAV clean full experiments."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--arrival-steps", type=int, default=500)
    parser.add_argument("--max-drain-steps", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/clean_experiments"))
    parser.add_argument("--run-prefix", type=str, default="hypergraph")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Default mode: write and print the plan without launching training.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Reserved for a future guarded launcher. T18 still refuses execution.",
    )
    return parser


def create_launch_run_dir(output_dir: Path, run_prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_prefix)
    run_dir = output_dir / f"{timestamp}_{safe_prefix}_plan"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _gate_definitions() -> List[Dict[str, str]]:
    return [
        {
            "name": "Gate 1: torch/model smoke",
            "required_status": "passed",
            "command": "python scripts/smoke_clean_server_torch.py",
        },
        {
            "name": "Gate 2: minimal training smoke",
            "required_status": "passed",
            "command": (
                "python scripts/train_clean_mainline.py --smoke --episodes 3 "
                "--max-steps-per-episode 20 --rollout-horizon 5 --run-name smoke"
            ),
        },
        {
            "name": "Gate 3: short sanity run",
            "required_status": "overall_pass=true",
            "command": (
                "python scripts/run_clean_sanity.py --episodes 100 "
                "--max-steps-per-episode 200 --rollout-horizon 20 --seed 0 "
                "--run-name sanity_seed0"
            ),
        },
    ]


def _failure_return_package() -> List[str]:
    return [
        "command",
        "stdout/stderr",
        "traceback",
        "git log -2 --oneline",
        "git status --short",
        "torch version/cuda availability",
        "config.json",
        "run_summary.json",
        "last 20 lines of train_metrics.jsonl",
        "eval_summary.json",
        "sanity_report.json",
        "NaN/Inf diagnostics",
        "tensor shape/device/mask dtype diagnostics",
    ]


def build_experiment_plan(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    experiments: List[Dict[str, Any]] = []
    flat_commands: List[str] = []

    for seed in args.seeds:
        run_name = f"{args.run_prefix}_seed{seed}"
        train_output_dir = Path("logs/clean_mainline")
        eval_output_dir = Path("logs/clean_eval")
        train_run_dir_placeholder = f"<train_run_dir_{run_name}>"
        eval_run_dir_placeholder = f"<eval_run_dir_{run_name}>"
        checkpoint_placeholder = f"{train_run_dir_placeholder}/checkpoints/latest.pt"

        train_command = _command(
            [
                "python",
                "scripts/train_clean_mainline.py",
                "--episodes",
                args.episodes,
                "--max-steps-per-episode",
                args.max_steps_per_episode,
                "--rollout-horizon",
                args.rollout_horizon,
                "--seed",
                seed,
                "--run-name",
                run_name,
                "--output-dir",
                train_output_dir,
            ]
        )
        eval_command = _command(
            [
                "python",
                "scripts/eval_clean_mainline.py",
                "--checkpoint",
                checkpoint_placeholder,
                "--episodes",
                args.eval_episodes,
                "--arrival-steps",
                args.arrival_steps,
                "--max-drain-steps",
                args.max_drain_steps,
                "--seed",
                seed,
                "--run-name",
                f"{run_name}_eval",
                "--output-dir",
                eval_output_dir,
            ]
        )
        plot_train_command = _command(
            ["python", "scripts/plot_clean_metrics.py", "--run-dir", train_run_dir_placeholder]
        )
        plot_eval_command = _command(
            ["python", "scripts/plot_clean_metrics.py", "--run-dir", eval_run_dir_placeholder]
        )
        seed_commands = [train_command, eval_command, plot_train_command, plot_eval_command]
        flat_commands.extend(seed_commands)

        experiments.append(
            {
                "seed": seed,
                "run_name": run_name,
                "method": IMPLEMENTED_METHOD,
                "train_command": train_command,
                "eval_command": eval_command,
                "plot_commands": [plot_train_command, plot_eval_command],
                "checkpoint_placeholder": checkpoint_placeholder,
                "train_run_dir_placeholder": train_run_dir_placeholder,
                "eval_run_dir_placeholder": eval_run_dir_placeholder,
            }
        )

    return {
        "schema": "hyperuav_clean_experiment_plan_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "dry_run",
        "dry_run": True,
        "execute_requested": bool(args.execute),
        "execution_supported": False,
        "implemented_method": IMPLEMENTED_METHOD,
        "baseline_ablation_status": BASELINE_ABLATION_STATUS,
        "future_baselines": [
            "only P_i^t",
            "ordinary graph embedding",
            "hypergraph embedding",
        ],
        "launch_run_dir": str(run_dir),
        "gates": _gate_definitions(),
        "prelaunch_checks": [
            "git status --short is empty",
            "git log -2 --oneline recorded",
            "torch version and CUDA availability recorded",
            "config.json saved in each run directory",
            "seed saved in each run config",
            "output directories are independent and writable",
            "checkpoint/log/plot directories are writable",
            "only clean entrypoints are used",
        ],
        "experiments": experiments,
        "commands": flat_commands,
        "failure_return_package": _failure_return_package(),
    }


def write_experiment_plan(plan: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_dir = create_launch_run_dir(args.output_dir, args.run_prefix)
    plan = build_experiment_plan(args, run_dir)
    plan_path = run_dir / "experiment_plan.json"
    write_experiment_plan(plan, plan_path)

    print(json.dumps({"experiment_plan": str(plan_path), "dry_run": True}, indent=2))
    print("Planned clean commands:")
    for command in plan["commands"]:
        print(command)

    if args.execute:
        print(
            "Execution mode is intentionally not implemented in T18; review the dry-run plan first.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
