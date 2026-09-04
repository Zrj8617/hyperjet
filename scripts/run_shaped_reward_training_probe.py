"""Run and summarize the approved 2 arms x 3 seeds shaped-reward probe."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        help="Run the six probe cells concurrently, one per listed GPU.",
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _single_run_dir(parent: Path) -> Path:
    rows = [path for path in parent.iterdir() if path.is_dir()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one run directory below {parent}, found {len(rows)}")
    return rows[0]


def _run_cell(
    *,
    output_root: Path,
    arm: str,
    seed: int,
    episodes: int,
    max_steps: int,
    rollout_horizon: int,
    device: str,
    gpu: int | None = None,
) -> tuple[Path, list[str]]:
    parent = output_root / "runs" / arm / f"seed{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_clean_mainline.py"),
        "--episodes", str(episodes),
        "--max-steps-per-episode", str(max_steps),
        "--rollout-horizon", str(rollout_horizon),
        "--seed", str(seed),
        "--task-encoder", "mlp",
        "--device", str(device),
        "--output-dir", str(parent),
        "--run-name", f"shaped_reward_probe_{arm}",
        "--checkpoint-interval", str(episodes),
        (
            "--dag-progress-potential-shaping"
            if arm == "on"
            else "--no-dag-progress-potential-shaping"
        ),
    ]
    log_path = output_root / "logs" / f"{arm}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process_env = os.environ.copy()
    recorded_command = command
    if gpu is not None:
        process_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        recorded_command = [f"CUDA_VISIBLE_DEVICES={gpu}", *command]
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=process_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return _single_run_dir(parent), recorded_command


def _metric(row: dict[str, Any], name: str) -> float | None:
    if name == "offloading_entropy_normalized":
        value = row.get("ppo_diagnostics", {}).get(
            "rollout_offloading_entropy_normalized_mean"
        )
    else:
        value = row.get(name)
    return None if value is None else float(value)


def _curve(rows: list[dict[str, Any]], name: str) -> list[dict[str, float]]:
    values = []
    for row in rows:
        value = _metric(row, name)
        update = row.get("ppo_update_step")
        if value is not None and update is not None:
            values.append({"update": int(update), "value": value})
    return values


def _terminal_curve(rows: list[dict[str, Any]], name: str) -> list[dict[str, float]]:
    values = []
    for row in rows:
        if not bool(row.get("episode_terminal_record", False)):
            continue
        value = _metric(row, name)
        if value is not None:
            values.append({"episode": int(row["episode"]), "value": value})
    return values


def _tail_mean(curve: list[dict[str, float]], fraction: float = 0.2) -> float | None:
    if not curve:
        return None
    count = max(1, int(np.ceil(len(curve) * fraction)))
    return float(np.mean([row["value"] for row in curve[-count:]]))


def _summarize(output_root: Path, runs: dict[str, dict[str, str]], commands: list[list[str]]) -> dict[str, Any]:
    update_metrics = (
        "ppo_shaped_explained_variance",
        "ppo_de_shaped_explained_variance",
        "offloading_entropy_normalized",
    )
    system_metrics = (
        "DAG_completion_rate",
        "average_dag_flowtime",
        "DAG_throughput",
    )
    result: dict[str, Any] = {
        "schema": "shaped_reward_training_probe_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "arms": ["off", "on"],
            "seeds": [0, 1, 2],
            "single_change": "cumulative DAG-progress potential shaping",
            "task_encoder": "mlp",
        },
        "commands": commands,
        "runs": {},
    }
    for arm in ("off", "on"):
        result["runs"][arm] = {}
        for seed in (0, 1, 2):
            run_dir = Path(runs[arm][str(seed)])
            rows = _read_jsonl(run_dir / "train_metrics.jsonl")
            curves = {name: _curve(rows, name) for name in update_metrics}
            terminal = {name: _terminal_curve(rows, name) for name in system_metrics}
            result["runs"][arm][str(seed)] = {
                "run_dir": str(run_dir),
                "update_curves": curves,
                "episode_curves": terminal,
                "tail_update_means": {
                    name: _tail_mean(curves[name]) for name in update_metrics
                },
                "tail_episode_means": {
                    name: _tail_mean(terminal[name]) for name in system_metrics
                },
            }
    result["paired_tail_deltas_on_minus_off"] = {
        str(seed): {
            **{
                name: (
                    result["runs"]["on"][str(seed)]["tail_update_means"][name]
                    - result["runs"]["off"][str(seed)]["tail_update_means"][name]
                )
                for name in update_metrics
            },
            **{
                name: (
                    result["runs"]["on"][str(seed)]["tail_episode_means"][name]
                    - result["runs"]["off"][str(seed)]["tail_episode_means"][name]
                )
                for name in system_metrics
            },
        }
        for seed in (0, 1, 2)
    }
    return result


def _plot(result: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("update_curves", "ppo_shaped_explained_variance", "Shaped-target critic EV"),
        ("update_curves", "ppo_de_shaped_explained_variance", "De-shaped critic EV"),
        ("update_curves", "offloading_entropy_normalized", "Offloading entropy (normalized)"),
        ("episode_curves", "DAG_completion_rate", "DAG completion rate"),
        ("episode_curves", "average_dag_flowtime", "Average DAG flowtime"),
        ("episode_curves", "DAG_throughput", "DAG throughput"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    colors = {"off": "#4C78A8", "on": "#E45756"}
    for axis, (group, metric, title) in zip(axes.flat, panels):
        for arm in ("off", "on"):
            per_seed = []
            for seed in (0, 1, 2):
                curve = result["runs"][arm][str(seed)][group][metric]
                x_key = "update" if group == "update_curves" else "episode"
                x = [row[x_key] for row in curve]
                y = [row["value"] for row in curve]
                per_seed.append(y)
                axis.plot(x, y, color=colors[arm], alpha=0.22, linewidth=0.8)
            if per_seed and len({len(values) for values in per_seed}) == 1:
                axis.plot(
                    x,
                    np.mean(np.asarray(per_seed, dtype=np.float64), axis=0),
                    color=colors[arm],
                    linewidth=2.0,
                    label=arm.upper(),
                )
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    args = _parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, str]] = {"off": {}, "on": {}}
    commands: list[list[str]] = []
    cells = [(seed, arm) for seed in (0, 1, 2) for arm in ("off", "on")]

    def record_completed(arm: str, seed: int, run_dir: Path, command: list[str]) -> None:
        runs[arm][str(seed)] = str(run_dir)
        commands.append(command)
        (args.output_root / "progress.json").write_text(
            json.dumps({"runs": runs, "commands": commands}, indent=2),
            encoding="utf-8",
        )

    if args.gpus:
        if len(args.gpus) != len(cells):
            raise ValueError(f"--gpus requires exactly {len(cells)} device indices")
        with ThreadPoolExecutor(max_workers=len(cells)) as executor:
            futures = {
                executor.submit(
                    _run_cell,
                    output_root=args.output_root,
                    arm=arm,
                    seed=seed,
                    episodes=int(args.episodes),
                    max_steps=int(args.max_steps_per_episode),
                    rollout_horizon=int(args.rollout_horizon),
                    device=str(args.device),
                    gpu=gpu,
                ): (arm, seed)
                for (seed, arm), gpu in zip(cells, args.gpus)
            }
            for future in as_completed(futures):
                arm, seed = futures[future]
                run_dir, command = future.result()
                record_completed(arm, seed, run_dir, command)
    else:
        for seed, arm in cells:
            run_dir, command = _run_cell(
                output_root=args.output_root,
                arm=arm,
                seed=seed,
                episodes=int(args.episodes),
                max_steps=int(args.max_steps_per_episode),
                rollout_horizon=int(args.rollout_horizon),
                device=str(args.device),
            )
            record_completed(arm, seed, run_dir, command)
    result = _summarize(args.output_root, runs, commands)
    result_path = args.output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _plot(result, args.output_root / "curves" / "training_probe_curves.png")
    print(json.dumps({"result": str(result_path), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
