"""Read-only post-hoc audit of completed R2 training scalar curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


ARMS = ("control", "no_position", "low_cancel", "energy_balanced")
SEEDS = (42, 86, 1042)
SAMPLE_FIELDS = (
    "arm",
    "seed",
    "update",
    "reward",
    "completed_dag",
    "completion_rate",
    "flowtime",
    "critical_path_delay",
    "avg_queue",
    "energy_per_dag",
    "offloading_normalized_entropy",
    "logit_spread",
    "probability_spread",
    "critic_explained_variance",
    "value_loss",
    "return_std",
    "predicted_value_std",
    "offloading_actor_grad_norm",
    "critic_grad_norm",
    "global_preclip_norm",
    "grad_clip_scale",
    "actor_parameter_update_norm",
)
SCALAR_FIELDS = SAMPLE_FIELDS[3:]
INTERVALS = (
    (0, 500, "0-500"),
    (500, 1000, "500-1000"),
    (1000, 2000, "1000-2000"),
    (2000, 3000, "2000-3000"),
    (3000, 4000, "3000-4000"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("logs/r2_reward_bundle_summary.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/r2_training_curve_audit.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("logs/r2_training_curve_audit.csv"),
    )
    parser.add_argument("--sample-every", type=int, default=50)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _energy_per_dag(record: dict[str, Any], energy_weight: float) -> float | None:
    completed = _finite(record.get("completed_DAG_count"))
    penalty = _finite(record.get("episode_task_energy_penalty_so_far"))
    if completed is None or completed <= 0.0 or penalty is None or energy_weight <= 0.0:
        return None
    return -penalty / energy_weight / completed


def _extract(record: dict[str, Any], energy_weight: float) -> dict[str, float | None]:
    diag = record.get("ppo_diagnostics") or {}
    return {
        "reward": _finite(record.get("ppo_rollout_reward_mean")),
        "completed_dag": _finite(record.get("completed_DAG_count")),
        "completion_rate": _finite(record.get("DAG_completion_rate")),
        "flowtime": _finite(record.get("average_dag_flowtime")),
        # No per-update critical-path scalar is emitted by the completed R2 runs.
        "critical_path_delay": None,
        "avg_queue": _finite(record.get("avg_uav_queue_length")),
        "energy_per_dag": _energy_per_dag(record, energy_weight),
        "offloading_normalized_entropy": _finite(
            diag.get("rollout_offloading_entropy_normalized_mean")
        ),
        "logit_spread": _finite(diag.get("rollout_offloading_logit_spread_mean")),
        "probability_spread": _finite(
            diag.get("rollout_offloading_probability_spread_mean")
        ),
        "critic_explained_variance": _finite(record.get("ppo_explained_variance")),
        "value_loss": _finite(record.get("ppo_value_loss")),
        "return_std": _finite(record.get("ppo_returns_std")),
        "predicted_value_std": _finite(record.get("ppo_value_pred_std")),
        "offloading_actor_grad_norm": _finite(diag.get("grad_pre_clip_offloading")),
        "critic_grad_norm": _finite(diag.get("grad_pre_clip_critic")),
        "global_preclip_norm": _finite(diag.get("grad_pre_clip_global")),
        "grad_clip_scale": _finite(diag.get("grad_clip_scale")),
        "actor_parameter_update_norm": _finite(
            diag.get("offloading_actor_parameter_update_norm")
        ),
    }


def _nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [prefix] if not math.isfinite(float(value)) else []
    if isinstance(value, dict):
        paths: list[str] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_nonfinite_paths(item, child))
        return paths
    if isinstance(value, list):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_nonfinite_paths(item, f"{prefix}[{index}]"))
        return paths
    return []


def _read_updates(
    metrics_path: Path, energy_weight: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_update: dict[int, dict[str, Any]] = {}
    nonfinite: dict[str, list[int]] = {}
    malformed_lines = 0
    duplicate_updates = 0
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            update = record.get("ppo_update_step")
            if not isinstance(update, int) or update <= 0:
                continue
            if update in by_update:
                duplicate_updates += 1
                continue
            values = _extract(record, energy_weight)
            by_update[update] = {"update": update, **values}
            for path in _nonfinite_paths(record):
                nonfinite.setdefault(path, []).append(update)
    rows = [by_update[key] for key in sorted(by_update)]
    expected = list(range(1, 4001))
    observed = [row["update"] for row in rows]
    return rows, {
        "malformed_json_lines": malformed_lines,
        "unique_update_count": len(rows),
        "first_update": observed[0] if observed else None,
        "last_update": observed[-1] if observed else None,
        "missing_updates": sorted(set(expected).difference(observed)),
        "duplicate_updates_ignored": duplicate_updates,
        "nonfinite_updates_by_scalar": nonfinite,
    }


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def _scalar_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [
        float(row[field])
        for row in rows
        if row[field] is not None and math.isfinite(row[field])
    ]
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _slope(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    points = [
        (float(row["update"]), float(row[field]))
        for row in rows
        if row[field] is not None and math.isfinite(row[field])
    ]
    if len(points) < 2:
        return {
            "point_count": len(points),
            "slope_per_update": None,
            "projected_change_over_window": None,
            "value_std": None,
            "normalized_projected_change": None,
        }
    x_mean = sum(x for x, _ in points) / len(points)
    y_mean = sum(y for _, y in points) / len(points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    span = points[-1][0] - points[0][0]
    change = slope * span
    std = statistics.pstdev(y for _, y in points)
    normalized = abs(change) / std if std > 0.0 else (0.0 if change == 0.0 else None)
    return {
        "point_count": len(points),
        "slope_per_update": slope,
        "projected_change_over_window": change,
        "value_std": std,
        "normalized_projected_change": normalized,
    }


def _jump_audit(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    points = [
        (int(row["update"]), float(row[field]))
        for row in rows
        if row[field] is not None and math.isfinite(row[field])
    ]
    differences = [
        (right_update, right - left)
        for (_, left), (right_update, right) in zip(points, points[1:])
    ]
    if not differences:
        return {
            "max_abs_step_change": None,
            "max_abs_step_change_update": None,
            "robust_jump_score": None,
            "abrupt_jump_flag": False,
        }
    update, difference = max(differences, key=lambda item: abs(item[1]))
    magnitudes = [abs(value) for _, value in differences]
    median = statistics.median(magnitudes)
    deviations = [abs(value - median) for value in magnitudes]
    mad = statistics.median(deviations)
    score = (abs(difference) - median) / mad if mad > 0.0 else None
    return {
        "max_abs_step_change": abs(difference),
        "max_abs_step_change_update": update,
        "robust_jump_score": score,
        "abrupt_jump_flag": bool(score is not None and score >= 20.0),
    }


def _event_audit(run_dir: Path) -> dict[str, Any]:
    event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
    result: dict[str, Any] = {
        "event_file_count": len(event_files),
        "event_files": [str(path.resolve()) for path in event_files],
        "scalar_tags": {},
    }
    if not event_files:
        return result
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    for path in event_files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            events = accumulator.Scalars(tag)
            result["scalar_tags"][tag] = {
                "count": len(events),
                "first_step": events[0].step if events else None,
                "last_step": events[-1].step if events else None,
            }
    return result


def _best_observed_update(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row["completed_dag"] is not None
        and row["completion_rate"] is not None
        and row["flowtime"] is not None
    ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda row: (
            row["completed_dag"],
            row["completion_rate"],
            -row["flowtime"],
        ),
    )
    return {
        "update": best["update"],
        "selection_basis": "training_log_proxy: max completed_dag, then completion_rate, then min flowtime",
        "completed_dag": best["completed_dag"],
        "completion_rate": best["completion_rate"],
        "flowtime": best["flowtime"],
    }


def _run_audit(job: dict[str, Any], sample_every: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arm = str(job["arm"])
    seed = int(job["seed"])
    run_dir = Path(job["training_run_dir"])
    metrics_path = Path(job["training_diagnostics"]["train_metrics_jsonl"])
    controls = job.get("controls") or {}
    energy_weight = float(controls["energy_weight"])
    rows, continuity = _read_updates(metrics_path, energy_weight)
    sampled = [
        {"arm": arm, "seed": seed, **row}
        for row in rows
        if int(row["update"]) % sample_every == 0
    ]
    interval_means: dict[str, Any] = {}
    for start, end, label in INTERVALS:
        subset = [row for row in rows if start < int(row["update"]) <= end]
        interval_means[label] = {
            "update_count": len(subset),
            **{field: _mean(row[field] for row in subset) for field in SCALAR_FIELDS},
        }
    trends: dict[str, Any] = {}
    platform: dict[str, Any] = {}
    for field in SCALAR_FIELDS:
        last_500 = [row for row in rows if int(row["update"]) > 3500]
        last_1000 = [row for row in rows if int(row["update"]) > 3000]
        trends[field] = {
            "last_500": _slope(last_500, field),
            "last_1000": _slope(last_1000, field),
        }
        normalized = trends[field]["last_1000"]["normalized_projected_change"]
        platform[field] = {
            "normalized_projected_change": normalized,
            "plateau_flag": bool(normalized is not None and normalized <= 0.10),
        }
    jump_audit = {field: _jump_audit(rows, field) for field in SCALAR_FIELDS}
    run_summary = _load_json(Path(job["training_diagnostics"]["run_summary_json"]))
    event = _event_audit(run_dir)
    return {
        "arm": arm,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "event_audit": event,
        "continuity": continuity,
        "training_status": run_summary.get("status"),
        "completed_update_count": run_summary.get("completed_update_count"),
        "scalar_summary": {
            field: _scalar_summary(rows, field) for field in SCALAR_FIELDS
        },
        "interval_means": interval_means,
        "trends": trends,
        "platform_last_1000": platform,
        "best_observed_update": _best_observed_update(rows),
        "best_evaluated_checkpoint": {
            "update": 4000,
            "checkpoint": job.get("final_checkpoint"),
            "note": "Only the final checkpoint has deterministic evaluation metrics; intermediate checkpoints were not evaluated.",
        },
        "anomalies": {
            "nan_or_inf": bool(continuity["nonfinite_updates_by_scalar"]),
            "training_interrupted": not (
                run_summary.get("status") == "completed"
                and run_summary.get("completed_update_count") == 4000
                and continuity["unique_update_count"] == 4000
                and not continuity["missing_updates"]
            ),
            "abrupt_jump_by_scalar": jump_audit,
        },
    }, sampled


def main() -> int:
    args = _parser().parse_args()
    if args.sample_every <= 0:
        raise ValueError("--sample-every must be positive")
    summary = _load_json(args.summary.resolve())
    jobs = summary.get("jobs") or []
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed = {(str(job.get("arm")), int(job.get("seed"))) for job in jobs}
    if observed != expected:
        raise ValueError(f"R2 job matrix mismatch: expected={expected}, observed={observed}")

    run_audits: list[dict[str, Any]] = []
    sampled_rows: list[dict[str, Any]] = []
    for job in sorted(jobs, key=lambda row: (ARMS.index(row["arm"]), int(row["seed"]))):
        audit, sampled = _run_audit(job, args.sample_every)
        run_audits.append(audit)
        sampled_rows.extend(sampled)

    event_run_count = sum(
        audit["event_audit"]["event_file_count"] > 0 for audit in run_audits
    )
    payload = {
        "schema": "r2_training_curve_audit_v1",
        "source_summary": str(args.summary.resolve()),
        "data_source": "train_metrics.jsonl unique PPO update records",
        "sampling": {
            "every_updates": args.sample_every,
            "sampled_row_count": len(sampled_rows),
            "interval_convention": "start < update <= end",
        },
        "tensorboard_integrity": {
            "expected_run_count": 12,
            "runs_with_event_files": event_run_count,
            "runs_without_event_files": 12 - event_run_count,
            "complete": event_run_count == 12,
        },
        "scalar_availability": {
            field: sum(
                row[field] is not None for row in sampled_rows
            )
            for field in SCALAR_FIELDS
        },
        "platform_rule": {
            "numeric_measure": "abs(OLS slope * update span) / population std over the last 1000 updates",
            "plateau_flag_threshold": "normalized_projected_change <= 0.10",
            "interpretation": "mechanical numeric flag only; not a scientific conclusion",
        },
        "abrupt_jump_rule": {
            "numeric_measure": "(maximum absolute adjacent-update change - median absolute change) / MAD",
            "flag_threshold": ">= 20",
            "interpretation": "mechanical numeric flag only",
        },
        "run_audits": run_audits,
        "sampled_rows": sampled_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(sampled_rows)
    print(
        json.dumps(
            {
                "json": str(args.output_json.resolve()),
                "csv": str(args.output_csv.resolve()),
                "runs": len(run_audits),
                "sampled_rows": len(sampled_rows),
                "tensorboard_runs": event_run_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
