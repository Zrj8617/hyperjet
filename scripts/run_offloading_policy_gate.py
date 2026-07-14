from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.offloading_policy_gate import OFFLOADING_POLICIES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or execute the paired offloading policy gate.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="MODEL_SEED=PATH",
        help="Repeat exactly once per model checkpoint.",
    )
    parser.add_argument("--environment-seeds", type=int, nargs="+", default=[4242, 4243, 4244, 4245, 4246])
    parser.add_argument("--policies", nargs="+", choices=OFFLOADING_POLICIES, default=list(OFFLOADING_POLICIES))
    parser.add_argument("--arrival-steps", type=int, default=200)
    parser.add_argument("--max-drain-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, default=None, help="Required with --execute; must be a new directory.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--execute", action="store_true", help="Actually launch cells; default is a no-write dry run.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate already completed cells below output-root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.aggregate_only:
        summary = aggregate_gate_root(args.output_root)
        _write_aggregate_outputs(args.output_root, summary)
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
        return 0
    checkpoints = _parse_checkpoints(args.checkpoint)
    if not checkpoints:
        raise ValueError("at least one --checkpoint MODEL_SEED=PATH is required")

    cells = build_gate_cells(
        checkpoints=checkpoints,
        environment_seeds=args.environment_seeds,
        policies=args.policies,
        python=args.python,
        output_root=args.output_root,
        log_root=args.log_root,
        arrival_steps=args.arrival_steps,
        max_drain_steps=args.max_drain_steps,
        device=args.device,
    )
    plan = {
        "created_at": datetime.now().isoformat(),
        "execute": bool(args.execute),
        "cell_count": len(cells),
        "closed_loop_pairing": (
            "same seeds pair initial conditions only; policies may induce different later trajectories"
        ),
        "cells": cells,
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    if args.log_root is None:
        raise ValueError("--log-root is required with --execute")
    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")
    if args.log_root.exists():
        raise FileExistsError(f"log root already exists: {args.log_root}")

    args.output_root.mkdir(parents=True, exist_ok=False)
    args.log_root.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_root / "gate_manifest.json", plan)
    for cell in cells:
        log_path = Path(cell["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                cell["command"],
                cwd=ROOT,
                check=False,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode != 0:
            failure = {
                "status": "failed",
                "failed_cell": cell,
                "returncode": int(completed.returncode),
            }
            _write_json(args.output_root / "gate_summary.json", failure)
            return int(completed.returncode)

    summary = aggregate_gate_root(args.output_root)
    _write_aggregate_outputs(args.output_root, summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


def build_gate_cells(
    *,
    checkpoints: dict[int, Path],
    environment_seeds: list[int],
    policies: list[str],
    python: Path,
    output_root: Path,
    log_root: Path | None,
    arrival_steps: int,
    max_drain_steps: int,
    device: str,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    eval_script = ROOT / "scripts" / "eval_clean_mainline.py"
    for model_seed, checkpoint in sorted(checkpoints.items()):
        for environment_seed in environment_seeds:
            for policy in policies:
                cell_id = f"model{model_seed}_env{int(environment_seed)}_{policy}"
                cell_output = output_root / "cells" / cell_id
                log_path = None if log_root is None else log_root / f"{cell_id}.log"
                command = [
                    str(python),
                    str(eval_script),
                    "--checkpoint",
                    str(checkpoint),
                    "--episodes",
                    "1",
                    "--arrival-steps",
                    str(int(arrival_steps)),
                    "--max-drain-steps",
                    str(int(max_drain_steps)),
                    "--seed",
                    str(int(environment_seed)),
                    "--device",
                    str(device),
                    "--output-dir",
                    str(cell_output),
                    "--run-name",
                    cell_id,
                    "--offloading-policy",
                    str(policy),
                    "--no-render",
                ]
                cells.append(
                    {
                        "cell_id": cell_id,
                        "model_seed": int(model_seed),
                        "environment_seed": int(environment_seed),
                        "policy": str(policy),
                        "checkpoint": str(checkpoint),
                        "output_dir": str(cell_output),
                        "log_path": None if log_path is None else str(log_path),
                        "command": command,
                    }
                )
    return cells


def aggregate_gate_root(output_root: Path) -> dict[str, Any]:
    episode_rows: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    manifest_path = output_root / "gate_manifest.json"
    manifest: dict[str, Any] = {}
    expected_cells: dict[str, dict[str, Any]] = {}
    expected_episode_count = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_episode_count = int(manifest.get("cell_count", 0))
        expected_cells = {
            str(cell["cell_id"]): cell for cell in manifest.get("cells", []) if isinstance(cell, dict)
        }
    else:
        validation_errors.append("missing gate_manifest.json")
    summaries = sorted(output_root.glob("cells/*/*/eval_summary.json"))
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise ValueError(f"incomplete gate cell: {summary_path}")
        metrics_path = summary_path.with_name("eval_metrics.jsonl")
        lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(f"gate cell must contain exactly one episode row: {metrics_path}")
        row = json.loads(lines[0])
        cell_id = str(summary_path.parent.parent.name)
        expected_cell = expected_cells.get(cell_id, {})
        manifest_model_seed = expected_cell.get("model_seed")
        checkpoint_model_seed = row.get("checkpoint_model_seed")
        if checkpoint_model_seed is not None and manifest_model_seed is not None and int(checkpoint_model_seed) != int(manifest_model_seed):
            validation_errors.append(f"checkpoint/model seed mismatch in {cell_id}")
        row["gate_cell_id"] = cell_id
        row["gate_model_seed"] = (
            int(checkpoint_model_seed)
            if checkpoint_model_seed is not None
            else int(manifest_model_seed)
            if manifest_model_seed is not None
            else None
        )
        decision_path = summary_path.with_name("offloading_decisions.jsonl")
        decision_count = len(
            [line for line in decision_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ) if decision_path.is_file() else 0
        if decision_count != int(row.get("offloading_action_count", 0)):
            validation_errors.append(
                f"decision count mismatch at {summary_path.parent}: {decision_count} != {row.get('offloading_action_count')}"
            )
        if not _all_finite(row):
            validation_errors.append(f"NaN/Inf in metrics row: {metrics_path}")
        if int(row.get("kahypar_degraded_slot_count", 0)) > 0:
            validation_errors.append(f"KaHyPar degraded slots in {summary_path.parent}")
        row["run_dir"] = str(summary_path.parent)
        episode_rows.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        grouped[str(row["offloading_policy"])].append(row)
    by_policy = {policy: _summarize_rows(rows) for policy, rows in sorted(grouped.items())}
    expected_policies = set(OFFLOADING_POLICIES)
    observed_policies = set(by_policy)
    cell_keys = [
        row.get("gate_cell_id")
        for row in episode_rows
    ]
    if expected_episode_count is not None and len(episode_rows) != expected_episode_count:
        validation_errors.append(f"episode count mismatch: {len(episode_rows)} != {expected_episode_count}")
    duplicate_count = int(len(cell_keys) - len(set(cell_keys)))
    if duplicate_count > 0:
        validation_errors.append(f"duplicate gate cells: {duplicate_count}")
    if observed_policies != expected_policies:
        validation_errors.append(
            f"policy set mismatch: observed={sorted(observed_policies)} expected={sorted(expected_policies)}"
        )
    observed_cell_ids = set(str(value) for value in cell_keys)
    expected_cell_ids = set(expected_cells)
    if expected_cell_ids and observed_cell_ids != expected_cell_ids:
        validation_errors.append(
            f"gate cell id mismatch: missing={sorted(expected_cell_ids - observed_cell_ids)} "
            f"unexpected={sorted(observed_cell_ids - expected_cell_ids)}"
        )
    return {
        "status": "completed" if episode_rows and not validation_errors else "incomplete",
        "expected_episode_count": expected_episode_count,
        "episode_count": int(len(episode_rows)),
        "unique_cell_count": int(len(set(cell_keys))),
        "duplicate_cell_count": duplicate_count,
        "validation_errors": validation_errors,
        "policies": sorted(observed_policies),
        "by_policy": by_policy,
        "paired_vs_actor": _paired_vs_actor(episode_rows),
        "_episode_rows": episode_rows,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated = sum(float(row.get("arrival_generated_DAG_count", 0.0)) for row in rows)
    arrival_completed = sum(float(row.get("arrival_completed_DAG_count", 0.0)) for row in rows)
    final_completed = sum(float(row.get("completed_DAG_count", 0.0)) for row in rows)
    total_evaluation_time = sum(float(row.get("total_evaluation_time", 0.0)) for row in rows)
    flowtimes = [
        float(value)
        for row in rows
        for value in row.get("_dag_flowtime_samples", [])
    ]
    estimator_errors = [
        float(value)
        for row in rows
        for value in row.get("_estimator_error_samples", [])
    ]
    return {
        "episode_count": int(len(rows)),
        "arrival_generated_DAG_count": float(generated),
        "arrival_completed_DAG_count": float(arrival_completed),
        "arrival_DAG_completion_rate": float(arrival_completed / max(generated, 1.0)),
        "drain_final_completed_DAG_count": float(final_completed),
        "drain_final_DAG_completion_rate": float(final_completed / max(generated, 1.0)),
        "DAG_throughput": float(final_completed / total_evaluation_time) if total_evaluation_time > 0.0 else None,
        "DAG_throughput_episode_mean": _mean(row.get("DAG_throughput") for row in rows),
        "DAG_flowtime_mean": _mean(flowtimes),
        "DAG_flowtime_median": _percentile(flowtimes, 50.0),
        "DAG_flowtime_p90": _percentile(flowtimes, 90.0),
        "arrival_backlog_DAG_count": int(sum(int(row.get("arrival_backlog_DAG_count", 0)) for row in rows)),
        "arrival_backlog_task_count": int(sum(int(row.get("arrival_backlog_task_count", 0)) for row in rows)),
        "final_backlog_DAG_count": int(sum(int(row.get("final_backlog_DAG_count", 0)) for row in rows)),
        "final_backlog_task_count": int(sum(int(row.get("final_backlog_task_count", 0)) for row in rows)),
        "offloading_action_count": int(sum(int(row.get("offloading_action_count", 0)) for row in rows)),
        "actor_normalized_entropy_mean": _weighted_mean(rows, "actor_normalized_entropy_mean", "actor_entropy_sample_count"),
        "actor_top1_top2_margin_mean": _weighted_mean(rows, "actor_top1_top2_margin_mean", "actor_margin_sample_count"),
        "actor_greedy_agreement_rate": _ratio(rows, "actor_greedy_agreement_count", "actor_greedy_comparison_count"),
        "selected_estimated_regret_mean": _mean(
            value for row in rows for value in row.get("_selected_estimated_regret_samples", [])
        ),
        "estimator_calibration_count": int(len(estimator_errors)),
        "estimator_calibration_mae": _mean(abs(value) for value in estimator_errors),
        "estimator_calibration_bias": _mean(estimator_errors),
        "estimator_calibration_p90_abs_error": _percentile([abs(value) for value in estimator_errors], 90.0),
        "realized_cross_uav_transfer_time": float(sum(float(row.get("realized_cross_uav_transfer_time", 0.0)) for row in rows)),
        "realized_queue_resource_wait": float(sum(float(row.get("realized_queue_resource_wait", 0.0)) for row in rows)),
        "hover_action_ratio_mean": _mean(row.get("hover_action_ratio") for row in rows),
        "mean_uav_displacement_per_slot": _mean(row.get("mean_uav_displacement_per_slot") for row in rows),
        "movement_action_distribution": _weighted_distributions(
            rows, "movement_action_distribution", "total_executed_slots"
        ),
        "kahypar_degraded_slot_count": int(sum(int(row.get("kahypar_degraded_slot_count", 0)) for row in rows)),
        "provenance": {
            "checkpoint_model_seeds": sorted({int(row["gate_model_seed"]) for row in rows if row.get("gate_model_seed") is not None}),
            "environment_seeds": sorted({int(row["environment_seed"]) for row in rows}),
            "checkpoint_paths": sorted({str(row["checkpoint_path"]) for row in rows if row.get("checkpoint_path") is not None}),
            "git_commits": sorted({str(row["git_commit"]) for row in rows if row.get("git_commit") is not None}),
            "completed_dag_weights": sorted({float(row["completed_dag_weight"]) for row in rows if row.get("completed_dag_weight") is not None}),
            "detach_critic_hgnn": sorted({bool(row.get("detach_critic_hgnn")) for row in rows}),
            "freeze_ue_mobility": sorted({bool(row.get("freeze_ue_mobility")) for row in rows}),
        },
    }


def _parse_checkpoints(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator or not path_text:
            raise ValueError(f"checkpoint must use MODEL_SEED=PATH: {value}")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate model seed: {seed}")
        result[seed] = Path(path_text)
    return result


def _paired_vs_actor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {
        (row.get("gate_model_seed"), row.get("environment_seed"), row.get("offloading_policy")): row
        for row in rows
    }
    metrics = (
        "arrival_DAG_completion_rate",
        "DAG_completion_rate",
        "DAG_throughput",
        "Average_DAG_flowtime",
        "arrival_backlog_DAG_count",
        "final_backlog_DAG_count",
        "selected_estimated_regret_mean",
        "realized_cross_uav_transfer_time",
        "realized_queue_resource_wait",
    )
    result: dict[str, Any] = {}
    for policy in OFFLOADING_POLICIES:
        if policy == "actor_argmax":
            continue
        cell_deltas: list[dict[str, Any]] = []
        for (model_seed, environment_seed, observed_policy), row in sorted(
            indexed.items(), key=lambda item: (str(item[0][0]), int(item[0][1]), str(item[0][2]))
        ):
            if observed_policy != policy:
                continue
            actor = indexed.get((model_seed, environment_seed, "actor_argmax"))
            if actor is None:
                continue
            deltas = {
                metric: (
                    float(row[metric]) - float(actor[metric])
                    if row.get(metric) is not None and actor.get(metric) is not None
                    else None
                )
                for metric in metrics
            }
            cell_deltas.append(
                {
                    "checkpoint_model_seed": model_seed,
                    "environment_seed": environment_seed,
                    "deltas_policy_minus_actor": deltas,
                }
            )
        metric_summary = {}
        for metric in metrics:
            values = [
                float(cell["deltas_policy_minus_actor"][metric])
                for cell in cell_deltas
                if cell["deltas_policy_minus_actor"][metric] is not None
            ]
            metric_summary[metric] = {
                "paired_count": len(values),
                "mean_delta": _mean(values),
                "median_delta": _percentile(values, 50.0),
                "positive_count": int(sum(value > 0.0 for value in values)),
                "negative_count": int(sum(value < 0.0 for value in values)),
                "zero_count": int(sum(value == 0.0 for value in values)),
            }
        result[policy] = {
            "paired_cell_count": len(cell_deltas),
            "metric_summary": metric_summary,
            "cells": cell_deltas,
        }
    return result


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _mean(values: Any) -> float | None:
    resolved = [float(value) for value in values if value is not None]
    return float(np.mean(resolved)) if resolved else None


def _percentile(values: Any, percentile: float) -> float | None:
    resolved = [float(value) for value in values if value is not None]
    return float(np.percentile(np.asarray(resolved, dtype=np.float64), percentile)) if resolved else None


def _weighted_mean(rows: list[dict[str, Any]], value_key: str, weight_key: str) -> float | None:
    pairs = [
        (float(row[value_key]), int(row.get(weight_key, 0)))
        for row in rows
        if row.get(value_key) is not None and int(row.get(weight_key, 0)) > 0
    ]
    total_weight = sum(weight for _, weight in pairs)
    return float(sum(value * weight for value, weight in pairs) / total_weight) if total_weight > 0 else None


def _ratio(rows: list[dict[str, Any]], numerator_key: str, denominator_key: str) -> float | None:
    numerator = sum(int(row.get(numerator_key, 0)) for row in rows)
    denominator = sum(int(row.get(denominator_key, 0)) for row in rows)
    return float(numerator / denominator) if denominator > 0 else None


def _weighted_distributions(rows: list[dict[str, Any]], distribution_key: str, weight_key: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    total_weight = 0.0
    for row in rows:
        weight = float(row.get(weight_key, 0.0))
        distribution = row.get(distribution_key, {})
        if weight <= 0.0 or not isinstance(distribution, dict):
            continue
        total_weight += weight
        for key, value in distribution.items():
            totals[str(key)] += float(value) * weight
    return {key: value / total_weight for key, value in sorted(totals.items())} if total_weight > 0.0 else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _write_aggregate_outputs(output_root: Path, summary: dict[str, Any]) -> None:
    episode_rows = summary.pop("_episode_rows")
    with (output_root / "gate_episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in episode_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    _write_json(output_root / "gate_summary.json", summary)


if __name__ == "__main__":
    raise SystemExit(main())
