from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.assignment import (
    CleanAssignmentBuffer,
    TemporaryReservationState,
    build_offloading_candidate_components,
    freeze_ready_tasks,
)
from environment.env import Env
from environment.forecast import ForecastContext


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate the compute-only forecast DP.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default=",".join(str(value) for value in range(20)))
    parser.add_argument("--steps", type=int, default=500)
    return parser


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout.rstrip()


def _run_seed(seed: int, steps: int, *, forecast_enabled: bool) -> dict[str, Any]:
    np.random.seed(int(seed))
    env = Env(freeze_ue_mobility=False)
    env.reset()
    deltas: list[dict[str, Any]] = []
    assignments_trace: list[tuple[int, str, int]] = []
    forecast_wall = 0.0
    collection_started = perf_counter()
    info: dict[str, Any] = {}
    for slot in range(int(steps)):
        env.prepare_slot_state()
        env.apply_movement({int(uav.id): 0 for uav in env.uavs})
        ready = freeze_ready_tasks(env.task_manager)
        policy_clock = TemporaryReservationState.from_executor(env.uavs, env.executor)
        context = None
        if forecast_enabled:
            started = perf_counter()
            context = ForecastContext(
                task_manager=env.task_manager,
                executor=env.executor,
                uavs=env.uavs,
                ues=env.ues,
                current_time_seconds=env.current_time_seconds,
                uav_service_positions=env.uav_service_positions,
                ue_service_positions=env.ue_service_positions,
            )
            forecast_wall += perf_counter() - started
        buffer = CleanAssignmentBuffer()
        for order, task in enumerate(ready):
            _, _, mask, uav_ids, estimates = build_offloading_candidate_components(
                task=task,
                uavs=env.uavs,
                task_manager=env.task_manager,
                executor=env.executor,
                state_view=policy_clock,
                current_time_seconds=env.current_time_seconds,
                uav_service_positions=env.uav_service_positions,
                ue_service_positions=env.ue_service_positions,
                ues=env.ues,
            )
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                continue
            chosen = min(
                legal.tolist(),
                key=lambda idx: (float(estimates[idx].estimated_finish_time), int(uav_ids[idx])),
            )
            estimate = estimates[chosen]
            uav_id = int(uav_ids[chosen])
            if context is not None:
                delta = context.delta_for_reservation(
                    dag_id=str(task.dag_id),
                    uav_id=uav_id,
                    compute_finish_time=float(estimate.estimated_compute_finish_time),
                )
                forecast_wall += float(delta.wall_seconds)
                deltas.append(asdict(delta))
            policy_clock.reserve(
                str(task.task_id),
                uav_id,
                estimated_available_time=float(estimate.estimated_finish_time),
                estimated_queued_workload=float(estimate.estimated_queued_workload),
            )
            buffer.append(str(task.task_id), uav_id, order)
            assignments_trace.append((slot, str(task.task_id), uav_id))
        _, _, done, info = env.commit_and_advance(assignment_buffer=buffer)
        if done:
            break
    collection_wall = perf_counter() - collection_started
    return {
        "seed": int(seed),
        "deltas": deltas,
        "assignments_trace": assignments_trace,
        "forecast_wall_seconds": float(forecast_wall),
        "collection_wall_seconds": float(collection_wall),
        "latest_metrics": {
            key: info.get(key)
            for key in (
                "generated_dag_count",
                "completed_dag_count",
                "dag_completion_rate",
                "avg_dag_flowtime",
                "dag_throughput",
                "invalid_assignment_count",
            )
        },
    }


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    return {
        f"p{percentile}": (
            float(np.percentile(values, percentile)) if values.size else None
        )
        for percentile in (25, 50, 75, 90, 95, 99)
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seeds = [int(token) for token in str(args.seeds).split(",") if token.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    gate_off = _run_seed(seeds[0], min(int(args.steps), 100), forecast_enabled=False)
    gate_on = _run_seed(seeds[0], min(int(args.steps), 100), forecast_enabled=True)
    gate_equal = (
        gate_off["assignments_trace"] == gate_on["assignments_trace"]
        and gate_off["latest_metrics"] == gate_on["latest_metrics"]
    )
    if not gate_equal:
        raise AssertionError("forecast gate changed the fixed-policy environment trace")

    cells = [_run_seed(seed, int(args.steps), forecast_enabled=True) for seed in seeds]
    rows = [row for cell in cells for row in cell["deltas"]]
    values = np.asarray([float(row["total"]) for row in rows], dtype=np.float64)
    absolute = np.abs(values)
    positive = values[values > 1e-9]
    target = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    cross = np.asarray([float(row["cross"]) for row in rows], dtype=np.float64)
    affected = np.asarray([int(row["affected_dags"]) for row in rows], dtype=np.int64)
    forecast_wall = sum(float(cell["forecast_wall_seconds"]) for cell in cells)
    collection_wall = sum(float(cell["collection_wall_seconds"]) for cell in cells)
    wall_frac = forecast_wall / max(collection_wall, 1e-12)
    zero_fraction = float(np.mean(absolute <= 1e-9)) if absolute.size else 1.0
    abs_median = float(np.median(absolute)) if absolute.size else 0.0
    stop_reasons: list[str] = []
    if wall_frac > 0.10:
        stop_reasons.append("forecast_wall_time_frac_gt_0.10")
    if abs_median > 200.0:
        stop_reasons.append("absolute_delta_phi_median_gt_200s")
    if zero_fraction > 0.90:
        stop_reasons.append("zero_fraction_gt_0.90")
    result = {
        "schema": "reward_forecast_calibration_v1",
        "status": "stop" if stop_reasons else "pass",
        "scale_seconds": None if stop_reasons else 500.0,
        "stop_reasons": stop_reasons,
        "design": {
            "policy": "eft_greedy",
            "movement": "forced_hover",
            "training": False,
            "seeds": seeds,
            "steps": int(args.steps),
            "zero_threshold": 1e-9,
        },
        "gate_off_exact": gate_equal,
        "metrics": {
            "decision_count": int(values.size),
            "zero_fraction": zero_fraction,
            "delta_phi_percentiles": _percentiles(values),
            "absolute_delta_phi_percentiles": _percentiles(absolute),
            "mean_positive": float(positive.mean()) if positive.size else None,
            "target_dag_delta": {"mean": float(target.mean()) if target.size else None, **_percentiles(target)},
            "cross_dag_delta": {"mean": float(cross.mean()) if cross.size else None, **_percentiles(cross)},
            "affected_dag_count": {"mean": float(affected.mean()) if affected.size else None, **_percentiles(affected)},
            "forecast_wall_seconds": float(forecast_wall),
            "collection_wall_seconds": float(collection_wall),
            "forecast_wall_time_frac": float(wall_frac),
        },
        "version": {
            "head": _git("rev-parse", "HEAD"),
            "dirty": _git("status", "--porcelain").splitlines(),
            "script_git_object": _git("hash-object", str(Path(__file__).resolve())),
        },
        "cells": [
            {key: value for key, value in cell.items() if key != "deltas"}
            for cell in cells
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "metrics": result["metrics"]}))
    return 3 if stop_reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
