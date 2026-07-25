from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("random_hash", "greedy_eft", "mappo_mlp", "mappo_hgnn")
DEFAULT_SEEDS = (42, 86, 1042)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or launch the 4x3 HyperUAV environment sanity matrix."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--completed-dag-weight", type=float, default=16.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs") / "four_way_environment_sanity",
    )
    parser.add_argument("--run-prefix", type=str, default="env_sanity")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--execute", action="store_true", default=False)
    return parser


def build_matrix(args: argparse.Namespace, matrix_root: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in args.seeds:
            cell_id = f"{method}_seed{int(seed)}"
            common = {
                "cell_id": cell_id,
                "method": method,
                "seed": int(seed),
                "episodes": int(args.episodes),
                "max_steps_per_episode": int(args.max_steps_per_episode),
                "movement_frozen": True,
                "completed_dag_weight": float(args.completed_dag_weight),
            }
            if method in {"random_hash", "greedy_eft"}:
                command = [
                    sys.executable,
                    "scripts/run_clean_policy_baseline.py",
                    "--policy",
                    method,
                    "--episodes",
                    str(int(args.episodes)),
                    "--max-steps-per-episode",
                    str(int(args.max_steps_per_episode)),
                    "--seed",
                    str(int(seed)),
                    "--completed-dag-weight",
                    str(float(args.completed_dag_weight)),
                    "--output-dir",
                    str(matrix_root / "runs" / method),
                    "--run-name",
                    cell_id,
                ]
                kind = "no_learning"
            else:
                encoder = "mlp" if method == "mappo_mlp" else "hgnn"
                command = [
                    sys.executable,
                    "scripts/train_clean_mainline.py",
                    "--episodes",
                    str(int(args.episodes)),
                    "--max-steps-per-episode",
                    str(int(args.max_steps_per_episode)),
                    "--rollout-horizon",
                    str(int(args.rollout_horizon)),
                    "--ppo-epochs",
                    str(int(args.ppo_epochs)),
                    "--seed",
                    str(int(seed)),
                    "--completed-dag-weight",
                    str(float(args.completed_dag_weight)),
                    "--freeze-movement",
                    "--task-encoder",
                    encoder,
                    "--no-normalize-value-targets",
                    "--value-clip-epsilon",
                    "0",
                    "--offloading-counterfactual-coef",
                    "0",
                    "--offloading-action-value-loss-coef",
                    "0",
                    "--offloading-lagged-q-coef",
                    "0",
                    "--offloading-lagged-q-loss-coef",
                    "0",
                    "--checkpoint-interval",
                    "100",
                    "--device",
                    str(args.device),
                    "--output-dir",
                    str(matrix_root / "runs" / method),
                    "--run-name",
                    cell_id,
                ]
                kind = "training"
                common["task_encoder"] = encoder
                common["rollout_horizon"] = int(args.rollout_horizon)
                common["ppo_epochs"] = int(args.ppo_epochs)
                common["normalize_value_targets"] = False
                common["value_clip_epsilon"] = 0.0
            cells.append(
                {
                    **common,
                    "kind": kind,
                    "command": command,
                    "stdout_log": str(matrix_root / "launcher_logs" / f"{cell_id}.log"),
                }
            )
    return cells


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)
    matrix_root = _create_matrix_root(args.output_dir, args.run_prefix)
    cells = build_matrix(args, matrix_root)
    manifest = {
        "schema": "four_way_environment_sanity_matrix_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "matrix_root": str(matrix_root),
        "execute": bool(args.execute),
        "max_parallel": int(args.max_parallel),
        "methods": list(METHODS),
        "seeds": [int(seed) for seed in args.seeds],
        "cell_count": len(cells),
        "protocol": {
            "episodes": int(args.episodes),
            "max_steps_per_episode": int(args.max_steps_per_episode),
            "movement_frozen": True,
            "completed_dag_weight": float(args.completed_dag_weight),
            "max_active_dags_per_ue": 1,
            "rollout_horizon": int(args.rollout_horizon),
            "ppo_epochs": int(args.ppo_epochs),
            "normalize_value_targets": False,
            "value_clip_epsilon": 0.0,
            "counterfactual_enabled": False,
            "lagged_residual_q_enabled": False,
        },
        "cells": cells,
    }
    _write_json(matrix_root / "manifest.json", manifest)
    _write_json(
        matrix_root / "launcher_status.json",
        {
            "status": "planned" if not args.execute else "starting",
            "matrix_root": str(matrix_root),
            "cell_count": len(cells),
        },
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    if not args.execute:
        return 0
    return _execute_cells(matrix_root, cells, int(args.max_parallel))


def _execute_cells(
    matrix_root: Path,
    cells: list[dict[str, Any]],
    max_parallel: int,
) -> int:
    pending = list(cells)
    active: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    log_dir = matrix_root / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    while pending or active:
        while pending and len(active) < max_parallel:
            cell = pending.pop(0)
            log_handle = Path(cell["stdout_log"]).open("w", encoding="utf-8")
            process = subprocess.Popen(
                cell["command"],
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            cell["pid"] = int(process.pid)
            cell["started_at"] = datetime.now().isoformat(timespec="seconds")
            active.append((cell, process, log_handle))
            _write_launch_state(matrix_root, pending, active, completed, failed)

        time.sleep(1.0)
        still_active: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
        for cell, process, log_handle in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((cell, process, log_handle))
                continue
            log_handle.close()
            result = {
                **cell,
                "return_code": int(return_code),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            if int(return_code) == 0:
                completed.append(result)
            else:
                failed.append(result)
                pending.clear()
            _write_launch_state(
                matrix_root, pending, still_active, completed, failed
            )
        active = still_active
        if failed and not active:
            break

    return 1 if failed else 0


def _write_launch_state(
    matrix_root: Path,
    pending: list[dict[str, Any]],
    active: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]],
    completed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> None:
    _write_json(
        matrix_root / "launcher_status.json",
        {
            "status": (
                "failed"
                if failed
                else "completed"
                if not pending and not active
                else "running"
            ),
            "pending_cell_ids": [cell["cell_id"] for cell in pending],
            "active": [
                {"cell_id": cell["cell_id"], "pid": int(process.pid)}
                for cell, process, _ in active
            ],
            "completed_cell_ids": [cell["cell_id"] for cell in completed],
            "failed": failed,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _validate_args(args: argparse.Namespace) -> None:
    if len(set(int(seed) for seed in args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if int(args.episodes) <= 0 or int(args.max_steps_per_episode) <= 0:
        raise ValueError("episodes and max-steps-per-episode must be positive")
    if int(args.rollout_horizon) <= 0 or int(args.ppo_epochs) <= 0:
        raise ValueError("rollout-horizon and ppo-epochs must be positive")
    if int(args.max_parallel) <= 0:
        raise ValueError("max-parallel must be positive")


def _create_matrix_root(output_dir: Path, run_prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(run_prefix)
    ).strip("_")
    base = Path(output_dir)
    if not base.is_absolute():
        base = ROOT / base
    root = base / f"{timestamp}_{safe_prefix or 'matrix'}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
