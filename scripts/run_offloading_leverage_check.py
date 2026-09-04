"""Run and summarize the approved deterministic offloading leverage audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("random", "eft_greedy", "eft_worst")
METRICS = {
    "dag_completion_rate": "higher",
    "average_dag_flowtime": "lower",
    "dag_throughput": "higher",
    "energy_per_completed_dag": "lower",
    "avg_uav_queue_length": "lower",
    "load_balance": "lower",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--completed-dag-weight", type=float, default=8.0)
    parser.add_argument("--parallelism", type=int, default=8)
    return parser


def _run_cell(
    *,
    output_root: Path,
    policy: str,
    seed: int,
    max_steps: int,
    completed_dag_weight: float,
) -> tuple[str, int, Path, list[str]]:
    parent = output_root / "runs" / policy / f"seed{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_clean_policy_baseline.py"),
        "--policy", policy,
        "--episodes", "1",
        "--max-steps-per-episode", str(max_steps),
        "--seed", str(seed),
        "--completed-dag-weight", str(completed_dag_weight),
        "--output-dir", str(parent),
        "--run-name", f"offloading_leverage_{policy}",
    ]
    log_path = output_root / "logs" / f"{policy}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    run_dirs = [path for path in parent.iterdir() if path.is_dir()]
    if len(run_dirs) != 1:
        raise RuntimeError(f"expected one run directory below {parent}, found {len(run_dirs)}")
    return policy, seed, run_dirs[0], command


def _read_episode(run_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "episode_metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one episode row in {run_dir}, found {len(rows)}")
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise RuntimeError(f"run did not complete: {run_dir}")
    return rows[0]


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1)) if len(values) > 1 else 0.0


def _improvement(candidate: float, reference: float, direction: str) -> dict[str, float]:
    raw_delta = float(candidate - reference)
    signed_improvement = raw_delta if direction == "higher" else -raw_delta
    relative = (
        float(signed_improvement / abs(reference))
        if abs(reference) > 1e-12
        else 0.0
    )
    return {
        "raw_delta": raw_delta,
        "relative_improvement": relative,
        "relative_improvement_percent": 100.0 * relative,
    }


def _summarize(
    *,
    output_root: Path,
    seeds: list[int],
    max_steps: int,
    run_dirs: dict[str, dict[str, str]],
    commands: list[list[str]],
) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {policy: {} for policy in POLICIES}
    configs: dict[str, dict[str, dict[str, Any]]] = {
        policy: {} for policy in POLICIES
    }
    for policy in POLICIES:
        for seed in seeds:
            run_dir = Path(run_dirs[policy][str(seed)])
            episode = _read_episode(run_dir)
            configs[policy][str(seed)] = json.loads(
                (run_dir / "config.json").read_text(encoding="utf-8")
            )
            cells[policy][str(seed)] = {
                "run_dir": str(run_dir),
                "metrics": {
                    metric: float(episode[metric])
                    for metric in METRICS
                },
            }

    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    for policy in POLICIES:
        aggregates[policy] = {}
        for metric in METRICS:
            values = [cells[policy][str(seed)]["metrics"][metric] for seed in seeds]
            aggregates[policy][metric] = {
                "mean": _mean(values),
                "sample_std": _std(values),
            }

    comparisons: dict[str, Any] = {
        "eft_greedy_vs_random": {},
        "eft_greedy_vs_eft_worst_span": {},
    }
    for metric, direction in METRICS.items():
        greedy_random_pairs = []
        greedy_worst_pairs = []
        for seed in seeds:
            greedy = cells["eft_greedy"][str(seed)]["metrics"][metric]
            random = cells["random"][str(seed)]["metrics"][metric]
            worst = cells["eft_worst"][str(seed)]["metrics"][metric]
            greedy_random_pairs.append(_improvement(greedy, random, direction))
            greedy_worst_pairs.append(_improvement(greedy, worst, direction))
        comparisons["eft_greedy_vs_random"][metric] = {
            "direction": direction,
            "aggregate": _improvement(
                aggregates["eft_greedy"][metric]["mean"],
                aggregates["random"][metric]["mean"],
                direction,
            ),
            "paired_by_seed": {
                str(seed): row for seed, row in zip(seeds, greedy_random_pairs)
            },
            "favorable_seed_count": int(
                sum(row["relative_improvement"] > 0.0 for row in greedy_random_pairs)
            ),
        }
        comparisons["eft_greedy_vs_eft_worst_span"][metric] = {
            "direction": direction,
            "aggregate": _improvement(
                aggregates["eft_greedy"][metric]["mean"],
                aggregates["eft_worst"][metric]["mean"],
                direction,
            ),
            "paired_by_seed": {
                str(seed): row for seed, row in zip(seeds, greedy_worst_pairs)
            },
            "favorable_seed_count": int(
                sum(row["relative_improvement"] > 0.0 for row in greedy_worst_pairs)
            ),
        }

    primary = comparisons["eft_greedy_vs_random"]
    large = (
        primary["average_dag_flowtime"]["aggregate"]["relative_improvement"] >= 0.15
        or primary["dag_completion_rate"]["aggregate"]["relative_improvement"] >= 0.10
    )
    small = (
        primary["average_dag_flowtime"]["aggregate"]["relative_improvement"] < 0.05
        and primary["dag_completion_rate"]["aggregate"]["relative_improvement"] < 0.05
    )
    verdict = "large_leverage" if large else "limited_leverage" if small else "intermediate"
    frozen_config_fields = (
        "seed",
        "episodes",
        "max_steps_per_episode",
        "movement_frozen",
        "completed_dag_weight",
        "max_active_dags_per_ue",
        "dag_base_arrival_probability",
        "optimizer_step_count",
        "model_parameter_tensor_count",
    )
    paired_config_mismatches = []
    for seed in seeds:
        reference = configs["random"][str(seed)]
        for policy in ("eft_greedy", "eft_worst"):
            candidate = configs[policy][str(seed)]
            for field in frozen_config_fields:
                if candidate.get(field) != reference.get(field):
                    paired_config_mismatches.append(
                        {
                            "seed": int(seed),
                            "policy": policy,
                            "field": field,
                            "random": reference.get(field),
                            "candidate": candidate.get(field),
                        }
                    )
    return {
        "schema": "offloading_leverage_check_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "policies": list(POLICIES),
            "seeds": seeds,
            "episodes_per_seed": 1,
            "max_steps_per_episode": int(max_steps),
            "movement_policy": "forced_hover",
            "deterministic": True,
            "crn_scope": "same seed and independent policy RNG; shared initial RNG stream",
            "optimizer_step_count": 0,
            "model_parameter_tensor_count": 0,
            "load_balance_definition": "coefficient of variation of completed workload by UAV; lower is better",
        },
        "commands": commands,
        "protocol_checks": {
            "paired_seed_sets_equal": all(
                set(run_dirs[policy]) == {str(seed) for seed in seeds}
                for policy in POLICIES
            ),
            "paired_frozen_config_fields": list(frozen_config_fields),
            "paired_config_mismatches": paired_config_mismatches,
            "paired_config_check_passed": not paired_config_mismatches,
            "optimizer_step_count": 0,
            "model_parameter_tensor_count": 0,
        },
        "runs": run_dirs,
        "cells": cells,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "pre_registered_verdict": verdict,
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds requires at least five unique values")
    if int(args.parallelism) <= 0:
        raise ValueError("--parallelism must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "runs").exists() or (args.output_root / "result.json").exists():
        raise FileExistsError(f"output root already contains evaluation data: {args.output_root}")
    run_dirs: dict[str, dict[str, str]] = {policy: {} for policy in POLICIES}
    commands: list[list[str]] = []
    cells = [(policy, seed) for seed in seeds for policy in POLICIES]
    with ThreadPoolExecutor(max_workers=min(int(args.parallelism), len(cells))) as executor:
        futures = {
            executor.submit(
                _run_cell,
                output_root=args.output_root,
                policy=policy,
                seed=seed,
                max_steps=int(args.max_steps_per_episode),
                completed_dag_weight=float(args.completed_dag_weight),
            ): (policy, seed)
            for policy, seed in cells
        }
        for future in as_completed(futures):
            policy, seed, run_dir, command = future.result()
            run_dirs[policy][str(seed)] = str(run_dir)
            commands.append(command)
            (args.output_root / "progress.json").write_text(
                json.dumps({"runs": run_dirs, "commands": commands}, indent=2),
                encoding="utf-8",
            )
    result = _summarize(
        output_root=args.output_root,
        seeds=seeds,
        max_steps=int(args.max_steps_per_episode),
        run_dirs=run_dirs,
        commands=commands,
    )
    result_path = args.output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "result": str(result_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
