"""Offline reward-only sweep for the R2 reward-ablation bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("/data2/zrj2025/HyperUAV-r1b-20260827/logs/r1b_reward_alignment.json")
DEFAULT_OUTPUT = ROOT / "logs" / "r2_reward_offline_sweep.json"
LOW_CANCEL_CANDIDATES = (8.0, 6.0, 4.0, 2.0)
ENERGY_CANDIDATES = (0.05, 0.10, 0.25, 0.50, 1.00)
POLICIES = ("random_legal", "nearest_legal", "greedy_eft")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _rank(values: dict[str, float], *, higher: bool) -> list[str]:
    return sorted(values, key=lambda key: ((-1.0 if higher else 1.0) * values[key], key))


def _spearman(left: list[str], right: list[str]) -> float:
    if set(left) != set(right) or len(left) < 2:
        raise ValueError("rankings must contain the same policies")
    positions = {name: index for index, name in enumerate(right)}
    squared = sum((index - positions[name]) ** 2 for index, name in enumerate(left))
    count = len(left)
    return float(1.0 - 6.0 * squared / (count * (count * count - 1)))


def _rescore(run: dict[str, Any], *, dag_weight: float, energy_weight: float) -> dict[str, Any]:
    components = run["components"]
    scaled = {
        "time_penalty": float(components["time_penalty"]["cumulative_sum"]),
        "task_energy_penalty": (
            float(components["task_energy_penalty"]["cumulative_sum"])
            * float(energy_weight)
            / 0.05
        ),
        "movement_energy_penalty": float(
            components["movement_energy_penalty"]["cumulative_sum"]
        ),
        "completed_dag_bonus": (
            float(components["completed_dag_bonus"]["cumulative_sum"])
            * float(dag_weight)
            / 8.0
        ),
        "movement_position_bonus": float(
            components["movement_position_bonus"]["cumulative_sum"]
        ),
    }
    total = float(sum(scaled.values()))
    absolute = float(sum(abs(value) for value in scaled.values()))
    return {
        **run,
        "rescored_reward": total,
        "rescored_components": scaled,
        "cancellation_ratio": 1.0 - abs(total) / max(absolute, 1e-12),
        "task_energy_absolute_share": abs(scaled["task_energy_penalty"]) / max(absolute, 1e-12),
    }


def _alignment(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    output: list[dict[str, Any]] = []
    any_reverse = False
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        reward_rank = _rank(
            {str(row["policy"]): float(row["rescored_reward"]) for row in seed_rows},
            higher=True,
        )
        metrics = {
            "completed_dags": (
                {str(row["policy"]): float(row["completed_dag_count"]) for row in seed_rows},
                True,
            ),
            "avg_flowtime": (
                {str(row["policy"]): float(row["avg_dag_flowtime"]) for row in seed_rows},
                False,
            ),
            "critical_path_delay": (
                {str(row["policy"]): float(row["avg_critical_path_delay"]) for row in seed_rows},
                False,
            ),
        }
        correlations: dict[str, float] = {}
        for name, (values, higher) in metrics.items():
            correlation = _spearman(reward_rank, _rank(values, higher=higher))
            correlations[name] = correlation
            any_reverse = any_reverse or correlation < 0.0
        output.append(
            {
                "seed": seed,
                "reward_ranking_best_to_worst": reward_rank,
                "spearman_reward_vs_better_metric": correlations,
            }
        )
    return output, any_reverse


def _pareto_contradictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contradictions: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = {str(row["policy"]): row for row in rows if int(row["seed"]) == seed}
        for policy_a in POLICIES:
            for policy_b in POLICIES:
                if policy_a == policy_b:
                    continue
                a = seed_rows[policy_a]
                b = seed_rows[policy_b]
                no_worse = (
                    float(a["completed_dag_count"]) >= float(b["completed_dag_count"])
                    and float(a["avg_dag_flowtime"]) <= float(b["avg_dag_flowtime"])
                    and float(a["task_energy_per_completed_dag"])
                    <= float(b["task_energy_per_completed_dag"])
                )
                strictly_better = (
                    float(a["completed_dag_count"]) > float(b["completed_dag_count"])
                    or float(a["avg_dag_flowtime"]) < float(b["avg_dag_flowtime"])
                    or float(a["task_energy_per_completed_dag"])
                    < float(b["task_energy_per_completed_dag"])
                )
                if no_worse and strictly_better and float(a["rescored_reward"]) < float(
                    b["rescored_reward"]
                ):
                    contradictions.append(
                        {
                            "seed": seed,
                            "better_policy": policy_a,
                            "worse_policy": policy_b,
                            "better_reward": float(a["rescored_reward"]),
                            "worse_reward": float(b["rescored_reward"]),
                        }
                    )
    return contradictions


def _candidate_row(
    runs: list[dict[str, Any]], *, dag_weight: float, energy_weight: float
) -> dict[str, Any]:
    rescored = [
        _rescore(run, dag_weight=dag_weight, energy_weight=energy_weight) for run in runs
    ]
    alignment, reverse = _alignment(rescored)
    contradictions = _pareto_contradictions(rescored)
    return {
        "completed_dag_weight": float(dag_weight),
        "energy_weight": float(energy_weight),
        "mean_cancellation_ratio": statistics.fmean(
            float(row["cancellation_ratio"]) for row in rescored
        ),
        "mean_task_energy_absolute_share": statistics.fmean(
            float(row["task_energy_absolute_share"]) for row in rescored
        ),
        "alignment_by_seed": alignment,
        "obvious_reverse_ranking": bool(reverse),
        "pareto_contradiction_count": len(contradictions),
        "pareto_contradictions": contradictions,
    }


def run_sweep(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "r1b_reward_alignment_v1":
        raise ValueError("R1-B input schema mismatch")
    runs = list(payload.get("runs", []))
    if len(runs) != 9:
        raise ValueError(f"expected 9 R1-B runs, got {len(runs)}")

    low_cancel = [
        _candidate_row(runs, dag_weight=weight, energy_weight=0.05)
        for weight in LOW_CANCEL_CANDIDATES
    ]
    baseline_cancellation = float(low_cancel[0]["mean_cancellation_ratio"])
    for row in low_cancel:
        reduction = (
            (baseline_cancellation - float(row["mean_cancellation_ratio"]))
            / max(baseline_cancellation, 1e-12)
        )
        row["cancellation_reduction_vs_weight_8"] = float(reduction)
        row["selection_constraints_pass"] = bool(
            float(row["completed_dag_weight"]) < 8.0
            and reduction >= 0.20
            and int(row["pareto_contradiction_count"]) == 0
            and not bool(row["obvious_reverse_ranking"])
        )
    eligible_low_cancel = [row for row in low_cancel if row["selection_constraints_pass"]]
    if eligible_low_cancel:
        selected_low_cancel = eligible_low_cancel[0]
        low_cancel_rule = "first lower-than-8 candidate in [6,4,2] satisfying all constraints"
    else:
        valid_direction = [
            row
            for row in low_cancel
            if int(row["pareto_contradiction_count"]) == 0
            and not bool(row["obvious_reverse_ranking"])
        ]
        pool = valid_direction or low_cancel
        selected_low_cancel = min(pool, key=lambda row: float(row["mean_cancellation_ratio"]))
        low_cancel_rule = "fallback: valid direction/Pareto candidate with minimum cancellation"

    energy = [
        _candidate_row(runs, dag_weight=8.0, energy_weight=weight)
        for weight in ENERGY_CANDIDATES
    ]
    for row in energy:
        row["distance_to_target_share_0_03"] = abs(
            float(row["mean_task_energy_absolute_share"]) - 0.03
        )
        row["selection_constraints_pass"] = bool(
            int(row["pareto_contradiction_count"]) == 0
            and not bool(row["obvious_reverse_ranking"])
        )
    eligible_energy = [row for row in energy if row["selection_constraints_pass"]]
    if not eligible_energy:
        raise RuntimeError("no energy candidate preserves Pareto and reward ranking direction")
    selected_energy = min(
        eligible_energy,
        key=lambda row: (
            float(row["distance_to_target_share_0_03"]),
            float(row["energy_weight"]),
        ),
    )
    return {
        "schema": "r2_reward_offline_sweep_v1",
        "source_schema": payload["schema"],
        "source_server_code_commit": payload.get("server_code_commit"),
        "candidate_sets": {
            "completed_dag_weight": list(LOW_CANCEL_CANDIDATES),
            "energy_weight": list(ENERGY_CANDIDATES),
        },
        "low_cancel_candidates": low_cancel,
        "selected_low_cancel_weight": float(selected_low_cancel["completed_dag_weight"]),
        "low_cancel_selection_rule_applied": low_cancel_rule,
        "energy_candidates": energy,
        "selected_energy_weight": float(selected_energy["energy_weight"]),
        "energy_selection_target_absolute_share": 0.03,
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    source = json.loads(args.input.resolve().read_text(encoding="utf-8"))
    result = run_sweep(source)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
