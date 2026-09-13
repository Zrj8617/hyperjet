from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


ARMS = ("N0", "A2", "B1", "B2", "B2D", "D1", "D2")
SEEDS = (0, 1, 2)
EXPECTED_TAGS = (
    "offload/entropy",
    "eval/completion_rate",
    "eval/avg_dag_flowtime",
    "eval/throughput",
    "eval/agreement_eft_greedy",
    "train/critic_ev",
    "train/episode_reward",
    "train/actor_loss",
    "train/critic_loss",
    "train/approx_kl",
    "move/move_rate_m",
    "forecast/wall_time_frac",
)
REFERENCES = {
    "random_20_seed": {
        "completion_rate": 0.770195,
        "avg_dag_flowtime": 624.2100094962527,
        "throughput": 0.05858,
    },
    "eft_greedy_20_seed": {
        "completion_rate": 0.841262,
        "avg_dag_flowtime": 331.66414271046784,
        "throughput": 0.09714,
    },
    "stage1_eft_anchor_20_env_seed_pooled": {
        "completion_rate": 0.8910942062881869,
        "avg_dag_flowtime": 246.4510303020856,
        "throughput": 0.11522,
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize completed reward-redesign runs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _summary(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    return {
        "values": data,
        "mean": mean(data),
        "sample_std": stdev(data) if len(data) > 1 else 0.0,
        "min": min(data),
        "max": max(data),
    }


def _paired_summary(values: Iterable[float]) -> dict[str, Any]:
    result = _summary(values)
    data = result["values"]
    if len(data) == 3:
        half_width = 4.302652729911275 * result["sample_std"] / math.sqrt(3.0)
        result["paired_t_95ci"] = [result["mean"] - half_width, result["mean"] + half_width]
    return result


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _tail_mean(rows: list[dict[str, Any]], key: str, fraction: float = 0.2) -> float:
    count = max(int(len(rows) * fraction), 1)
    values = [float(row[key]) for row in rows[-count:] if _finite(row.get(key))]
    return mean(values)


def _terminal_tail_mean(rows: list[dict[str, Any]], key: str, fraction: float = 0.2) -> float:
    terminal = [row for row in rows if bool(row.get("episode_terminal_record", False))]
    count = max(int(len(terminal) * fraction), 1)
    values = [float(row[key]) for row in terminal[-count:] if _finite(row.get(key))]
    return mean(values)


def _relative(candidate: float, control: float, *, lower_better: bool) -> float:
    if control == 0.0:
        return float("nan")
    numerator = control - candidate if lower_better else candidate - control
    return numerator / abs(control)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cells: list[dict[str, Any]] = []
    failures: list[str] = []
    version_counts: Counter[str] = Counter()
    actual_parameters_by_run: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            run_name = f"20260906_{arm}_seed{seed}"
            run_root = args.input_root / run_name
            result_path = run_root / "result.json"
            if not result_path.is_file():
                failures.append(f"missing result: {run_name}")
                continue
            result = _read_json(result_path)
            metric_paths = list(run_root.glob("train/*/train_metrics.jsonl"))
            if len(metric_paths) != 1:
                failures.append(f"expected one train metrics file: {run_name}")
                continue
            rows = _read_jsonl(metric_paths[0])
            actual = result["actual_parameters"]
            actual_parameters_by_run[run_name] = actual
            train = result["train"]
            evaluation = result["evaluation"]
            expected = {
                "status": result.get("status") == "completed",
                "arm": result.get("arm") == arm,
                "seed": int(result.get("seed", -1)) == seed,
                "episodes": int(actual.get("episodes", -1)) == 500,
                "steps": int(actual.get("max_steps_per_episode", -1)) == 500,
                "updates": int(train.get("completed_update_count", -1)) == 2000,
                "slots": int(train.get("global_slot", -1)) == 250000,
                "metric_rows": len(rows) == 2000,
                "tensorboard_declared_tags": tuple(result["tensorboard"]["tags"]) == EXPECTED_TAGS,
            }
            for name, passed in expected.items():
                if not passed:
                    failures.append(f"{run_name}: {name}")
            version = result["version"]
            version_key = json.dumps(
                {
                    "head": version.get("head"),
                    "runner": version.get("runner_git_object"),
                    "trainer": version.get("trainer_git_object"),
                },
                sort_keys=True,
            )
            version_counts[version_key] += 1
            forecast = sum(float(row.get("ppo_forecast_wall_seconds", 0.0)) for row in rows)
            collection = sum(float(row.get("ppo_collection_wall_seconds", 0.0)) for row in rows)
            early_count = max(len(rows) // 5, 1)
            early_entropy = mean(
                float(row["ppo_diagnostics"]["rollout_offloading_entropy_normalized_mean"])
                for row in rows[:early_count]
            )
            late_entropy = mean(
                float(row["ppo_diagnostics"]["rollout_offloading_entropy_normalized_mean"])
                for row in rows[-early_count:]
            )
            cells.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "run_name": run_name,
                    "wall_seconds": float(result["wall_seconds"]),
                    "evaluation": {
                        "completion_rate": float(evaluation["DAG_completion_rate"]),
                        "avg_dag_flowtime": float(evaluation["Average_DAG_flowtime"]),
                        "throughput": float(evaluation["DAG_throughput"]),
                        "agreement_eft_greedy": float(evaluation["actor_greedy_agreement_rate"]),
                        "normalized_entropy": float(evaluation["actor_normalized_entropy_mean"]),
                        "completed_dags": int(evaluation["completed_DAG_count"]),
                        "generated_dags": int(evaluation["generated_DAG_count"]),
                        "energy_per_completed_dag": float(evaluation["Energy_per_completed_DAG"]),
                    },
                    "training": {
                        "early_20pct_normalized_entropy": early_entropy,
                        "late_20pct_normalized_entropy": late_entropy,
                        "late_20pct_critic_ev": _tail_mean(rows, "ppo_explained_variance"),
                        "late_20pct_actor_loss": mean(
                            float(row["ppo_movement_loss"]) + float(row["ppo_offloading_loss"])
                            for row in rows[-early_count:]
                        ),
                        "late_20pct_critic_loss": _tail_mean(rows, "ppo_value_loss"),
                        "late_20pct_approx_kl": _tail_mean(rows, "ppo_approx_kl"),
                        "late_20pct_episode_reward": _terminal_tail_mean(rows, "episode_reward_total"),
                        "late_20pct_move_rate": 1.0
                        - _terminal_tail_mean(rows, "hover_action_ratio"),
                        "forecast_wall_seconds": forecast,
                        "collection_wall_seconds": collection,
                        "forecast_over_collection": forecast / max(collection, 1e-12),
                        "forecast_over_run_wall": forecast / max(float(result["wall_seconds"]), 1e-12),
                    },
                }
            )

    aggregates: dict[str, Any] = {}
    for arm in ARMS:
        arm_cells = [cell for cell in cells if cell["arm"] == arm]
        aggregates[arm] = {
            "evaluation": {
                key: _summary(cell["evaluation"][key] for cell in arm_cells)
                for key in arm_cells[0]["evaluation"]
            },
            "training": {
                key: _summary(cell["training"][key] for cell in arm_cells)
                for key in arm_cells[0]["training"]
            },
        }

    comparisons = {}
    for candidate, control, label in (
        ("B2", "B1", "B2_vs_B1_incremental_ledger"),
        ("B2D", "B2", "B2D_vs_B2_decision_credit"),
        ("D2", "B1", "D2_vs_B1_move_lambda_004"),
        ("A2", "N0", "A2_vs_N0_original_reward_group"),
        ("D1", "B2", "D1_vs_B2_global_eta_duplicate"),
    ):
        comparison = {}
        for metric, lower in (
            ("completion_rate", False),
            ("avg_dag_flowtime", True),
            ("throughput", False),
            ("energy_per_completed_dag", True),
        ):
            candidate_values = aggregates[candidate]["evaluation"][metric]["values"]
            control_values = aggregates[control]["evaluation"][metric]["values"]
            deltas = [left - right for left, right in zip(candidate_values, control_values)]
            comparison[metric] = {
                "raw_delta": _paired_summary(deltas),
                "relative_beneficial_change_of_means": _relative(
                    mean(candidate_values), mean(control_values), lower_better=lower
                ),
                "favorable_seeds": sum(delta < 0.0 if lower else delta > 0.0 for delta in deltas),
            }
        comparisons[label] = {"candidate": candidate, "control": control, "metrics": comparison}

    reference_comparisons: dict[str, Any] = {}
    for arm in ARMS:
        reference_comparisons[arm] = {}
        for reference_name, reference in REFERENCES.items():
            reference_comparisons[arm][reference_name] = {
                metric: {
                    "raw_delta": aggregates[arm]["evaluation"][metric]["mean"] - value,
                    "relative_beneficial_change": _relative(
                        aggregates[arm]["evaluation"][metric]["mean"],
                        value,
                        lower_better=metric == "avg_dag_flowtime",
                    ),
                }
                for metric, value in reference.items()
            }

    ignored_config_keys = {"reward_redesign_arm", "output_dir", "run_name"}
    for seed in SEEDS:
        reference = actual_parameters_by_run[f"20260906_N0_seed{seed}"]
        normalized_reference = {
            key: value for key, value in reference.items() if key not in ignored_config_keys
        }
        for arm in ARMS[1:]:
            candidate = actual_parameters_by_run[f"20260906_{arm}_seed{seed}"]
            normalized_candidate = {
                key: value for key, value in candidate.items() if key not in ignored_config_keys
            }
            if normalized_candidate != normalized_reference:
                failures.append(f"seed{seed}: frozen actual-parameter mismatch for {arm}")

    output = {
        "schema": "reward_redesign_formal_summary_v1",
        "status": "descriptive_only_protocol_deviations" if not failures else "invalid",
        "data_integrity_status": "valid" if not failures else "invalid",
        "run_count": len(cells),
        "failures": failures,
        "protocol": {
            "training_seeds": list(SEEDS),
            "episodes_per_run": 500,
            "steps_per_episode": 500,
            "updates_per_run": 2000,
            "evaluation": "one forced-hover deterministic 500-slot episode at the matching run seed",
            "expected_tensorboard_tags": list(EXPECTED_TAGS),
        },
        "protocol_audit": {
            "all_runs_complete_and_shape_valid": not failures,
            "actual_tensorboard_event_tag_check": "performed separately: 21/21 exact 12-tag match",
            "formal_evaluation_cells_per_checkpoint": 1,
            "stage1_evaluation_cells_per_checkpoint": 20,
            "evaluation_protocol_matches_stage1": False,
            "fixed_external_offer_tape_used": False,
            "admission_conditional_end_to_end_rates_reported": False,
            "b2d_implemented_credit": "batch-standardized analytic -delta_phi/500 actor advantage",
            "b2d_uses_registered_variable_discount_decision_gae_and_microstate_critic": False,
            "tensorboard_semantic_notes": {
                "train/episode_reward": "records PPO rollout reward total",
                "move/move_rate_m": "records the update row's last-slot movement rate",
                "forecast/wall_time_frac": "records the update row's last-slot forecast/collection ratio",
            },
        },
        "version_counts": dict(version_counts),
        "references": REFERENCES,
        "cells": cells,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "reference_comparisons": reference_comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": output["status"], "runs": len(cells), "failures": failures}))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
