"""Clean-mainline load diagnostic and Phase 3 calibration sweep.

Runs the clean Env with fixed non-learning offloading baselines and reports
throughput/backlog numbers. This script must NOT change environment behavior:
it only drives the public prepare/apply/commit API the same way the clean
trainer does.

Baselines:
  greedy: each frozen ready task -> legal UAV minimizing estimated finish time
  random: each frozen ready task -> uniform random legal UAV
Movement is always hover (baseline lower bound; no learned positioning).

Per-run output fields (human-readable table + one machine line "DIAG_JSON {...}"):
  policy / slots / seed
  generated_dags, completed_dags, completion_rate
  generated_tasks, completed_tasks
  avg_flowtime_raw            raw env time units (unit semantics fixed in Phase 1)
  active_dag_backlog_end/max  unfinished DAG count at end / max over slots
  ready_backlog_end/mean/max  frozen ready-set size statistics
  queue_len_end               per-UAV executor queue lengths at end
  queue_len_mean              mean total queue length per slot
  offloading_skipped_total    ready tasks skipped for having no legal candidate
  avail_drift_vs_slot_index   max(uav_available_time) - time_step   (current convention)
  avail_drift_vs_seconds      max(uav_available_time) - time_step * TIME_SLOT_DURATION

The two drift fields intentionally report both time conventions so the same
script documents the unit bug before Phase 1 and verifies the fix after it.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.assignment import (
    TemporaryReservationState,
    estimate_offloading_candidate,
    legal_candidate_uav_ids,
)
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean mainline load diagnostic (heuristic baselines).")
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--policies", nargs="+", default=["greedy", "random"], choices=["greedy", "random"])
    parser.add_argument(
        "--drain-slots",
        type=int,
        default=0,
        help="Diagnostic-only drain phase: after the arrival slots, disable DAG "
        "arrivals (same protocol as eval: DAG_BASE_ARRIVAL_PROB=0) and keep "
        "executing up to this many extra slots or until all active DAGs finish.",
    )
    parser.add_argument("--sweep", action="store_true", help="Run an in-memory Phase 3 load sweep.")
    parser.add_argument(
        "--arrival-probs",
        type=str,
        default="0.015,0.02,0.025,0.03",
        help="Comma-separated DAG_BASE_ARRIVAL_PROB values for --sweep.",
    )
    parser.add_argument(
        "--input-ranges",
        type=str,
        default="0.5:12,0.75:15,1.0:15,1.0:18",
        help="Comma-separated min:max MB ranges for --sweep, e.g. 0.5:12,1:18.",
    )
    parser.add_argument(
        "--output-ranges",
        type=str,
        default="0.25:6,0.5:8,0.5:10",
        help="Comma-separated min:max MB ranges for --sweep, e.g. 0.25:6,0.5:10.",
    )
    parser.add_argument(
        "--task-constant-ranges",
        type=str,
        default=None,
        help="Optional comma-separated integer min:max TASK_CONSTANT_RANGE values for --sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional new/empty directory for an auditable sweep manifest, progress, rows, and reports.",
    )
    return parser


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def _distribution_summary(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": round(float(np.mean(values)), 4) if values else 0.0,
        f"{prefix}_p50": round(_percentile(values, 50), 4),
        f"{prefix}_p90": round(_percentile(values, 90), 4),
        f"{prefix}_max": round(max(values), 4) if values else 0.0,
    }


def run_baseline(policy: str, slots: int, seed: int, drain_slots: int = 0) -> dict:
    np.random.seed(int(seed))
    # Keep policy sampling out of the environment RNG stream so greedy/random
    # diagnostics do not diverge merely because random offloading consumes RNG.
    policy_rng = np.random.default_rng(int(seed) + 1_000_003)
    env = Env()
    env.reset()
    graph_builder = CleanGraphBuilder()

    ready_backlog_samples: list[int] = []
    active_dag_samples: list[int] = []
    queue_total_samples: list[int] = []
    offloading_skipped_total = 0
    arrival_created_dags: list[float] = []
    arrival_eligible_ues: list[float] = []
    arrival_suppressed_ues: list[float] = []
    arrival_active_dags: list[float] = []
    arrival_ready_tasks: list[float] = []
    arrival_assignment_counts: list[float] = []
    arrival_skipped_counts: list[float] = []
    arrival_queue_pressure: list[float] = []
    arrival_dag_hyperedges: list[float] = []
    arrival_khop_hyperedges: list[float] = []
    arrival_attribute_hyperedges: list[float] = []
    arrival_partition_hyperedges: list[float] = []
    arrival_total_hyperedges: list[float] = []
    partition_status_counts: Counter[str] = Counter()

    def _execute_one_slot(*, arrival_phase: bool) -> None:
        nonlocal offloading_skipped_total
        eligible_ues = sum(
            1
            for ue in env.ues
            if ue.active_dag_id is None
            and env.task_manager.get_active_job_for_ue(ue.id) is None
        )
        context = env.prepare_slot_state()
        graph_snapshot = graph_builder.build(
            task_manager=env.task_manager,
            uavs=env.uavs,
            current_time_step=env.time_step,
            executor=env.executor,
            frozen_ready_task_ids=context["frozen_ready_task_ids"],
            new_dag_arrived=bool(context["new_dag_arrived"]),
            dag_arrival_version=int(context["dag_arrival_version"]),
        )
        env.apply_movement({})  # hover baseline

        ready_tasks = [env.task_manager.get_task(task_id) for task_id in context["frozen_ready_task_ids"]]
        ready_tasks = [task for task in ready_tasks if task is not None]
        ready_backlog_samples.append(len(ready_tasks))

        reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
        uav_map = {int(uav.id): uav for uav in env.uavs}
        ordered_uav_ids = sorted(uav_map)
        assignments: dict[str, int] = {}
        for task in ready_tasks:
            legal = legal_candidate_uav_ids(
                task=task,
                uav_ids=ordered_uav_ids,
                state_view=reservation,
                executor=env.executor,
                service_positions=env.uav_service_positions,
            )
            if not legal:
                continue
            if policy == "greedy":
                best_uav_id = None
                best_finish = None
                for uav_id in legal:
                    estimate = estimate_offloading_candidate(
                        task=task,
                        uav_id=uav_id,
                        uav_map=uav_map,
                        task_manager=env.task_manager,
                        executor=env.executor,
                        state_view=reservation,
                        current_time_seconds=env.current_time_seconds,
                        uav_service_positions=env.uav_service_positions,
                        ue_service_positions=env.ue_service_positions,
                        ues=env.ues,
                        legal=True,
                    )
                    if best_finish is None or estimate.estimated_finish_time < best_finish:
                        best_uav_id = uav_id
                        best_finish = estimate.estimated_finish_time
                selected_uav_id = int(best_uav_id)
            else:
                selected_uav_id = int(policy_rng.choice(legal))
            assignments[task.task_id] = selected_uav_id
            reservation.reserve(task.task_id, selected_uav_id)

        _, _, _, info = env.commit_and_advance(assignments=assignments)
        skipped_this_slot = int(info.get("offloading_skipped_no_candidate", 0))
        offloading_skipped_total += skipped_this_slot
        active_dag_samples.append(int(info.get("active_dags", 0)))
        queue_total_samples.append(
            sum(len(env.executor.uav_queues.get(int(uav.id), [])) for uav in env.uavs)
        )
        if arrival_phase:
            queue_total = float(queue_total_samples[-1])
            queue_capacity = max(float(config.NUM_UAVS * config.CLEAN_MAX_QUEUE_PER_UAV), 1.0)
            arrival_created_dags.append(float(context["created_dags"]))
            arrival_eligible_ues.append(float(eligible_ues))
            arrival_suppressed_ues.append(float(len(env.ues) - eligible_ues))
            arrival_active_dags.append(float(info.get("active_dags", 0)))
            arrival_ready_tasks.append(float(len(ready_tasks)))
            arrival_assignment_counts.append(float(len(assignments)))
            arrival_skipped_counts.append(float(skipped_this_slot))
            arrival_queue_pressure.append(queue_total / queue_capacity)
            arrival_dag_hyperedges.append(float(len(graph_snapshot.dag_hyperedges)))
            arrival_khop_hyperedges.append(float(len(graph_snapshot.khop_hyperedges)))
            arrival_attribute_hyperedges.append(float(len(graph_snapshot.attribute_hyperedges)))
            arrival_partition_hyperedges.append(float(len(graph_snapshot.partition_hyperedges)))
            arrival_total_hyperedges.append(float(len(graph_snapshot.hyperedges)))
            partition_status_counts[str(graph_snapshot.partition_status)] += 1

    def _completion_snapshot() -> tuple[int, int]:
        generated = len(env.task_manager.jobs)
        completed = sum(1 for job in env.task_manager.jobs.values() if job.completed)
        return generated, completed

    try:
        for _ in range(int(slots)):
            _execute_one_slot(arrival_phase=True)

        # Snapshot at the end of the arrival phase (before any drain).
        arrival_generated, arrival_completed = _completion_snapshot()
        arrival_snapshot = {
            "completion_rate_arrival_end": round(arrival_completed / max(arrival_generated, 1), 4),
            "active_dag_backlog_arrival_end": int(arrival_generated - arrival_completed),
            "ready_backlog_arrival_end": ready_backlog_samples[-1] if ready_backlog_samples else 0,
            "queue_len_arrival_end": [len(env.executor.uav_queues.get(int(uav.id), [])) for uav in env.uavs],
        }

        # Diagnostic-only drain phase: disable arrivals exactly like eval does
        # (DAG_BASE_ARRIVAL_PROB=0) and keep executing the same baseline policy.
        drain_executed = 0
        drain_end_reason = "disabled"
        if int(drain_slots) > 0:
            original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
            drain_end_reason = "max_drain"
            try:
                config.DAG_BASE_ARRIVAL_PROB = 0.0
                while drain_executed < int(drain_slots):
                    if sum(1 for job in env.task_manager.jobs.values() if not job.completed) == 0:
                        drain_end_reason = "all_completed"
                        break
                    _execute_one_slot(arrival_phase=False)
                    drain_executed += 1
                else:
                    if sum(1 for job in env.task_manager.jobs.values() if not job.completed) == 0:
                        drain_end_reason = "all_completed"
            finally:
                config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob
    finally:
        graph_builder.close()

    kahypar_worker_alive_after_close = bool(graph_builder.kahypar_worker_alive)
    kahypar_cleanup_failed = bool(graph_builder.kahypar_cleanup_failed)
    kahypar_circuit_open = bool(graph_builder.kahypar_circuit_open)
    kahypar_last_failure_reason = graph_builder.kahypar_last_failure_reason

    task_manager = env.task_manager
    generated_dags = len(task_manager.jobs)
    completed_dags = sum(1 for job in task_manager.jobs.values() if job.completed)
    flowtimes = [
        float(job.return_complete_time - job.arrival_time)
        for job in task_manager.jobs.values()
        if job.completed and job.return_complete_time is not None
    ]
    generated_tasks = len(task_manager.tasks)
    completed_tasks = sum(1 for task in task_manager.tasks.values() if task.state == "COMPLETED")
    queue_len_end = [len(env.executor.uav_queues.get(int(uav.id), [])) for uav in env.uavs]
    scheduled_records = list(env.executor.task_records.values())
    service_times = [
        float(record.upload_time)
        + float(record.inter_transfer_time)
        + float(record.compute_time)
        + float(record.return_time)
        for record in scheduled_records
    ]
    compute_times = [float(record.compute_time) for record in scheduled_records]
    avg_service_time = float(np.mean(service_times)) if service_times else 0.0
    avg_compute_time = float(np.mean(compute_times)) if compute_times else 0.0
    p50_compute_time = float(np.percentile(compute_times, 50)) if compute_times else 0.0
    p95_compute_time = float(np.percentile(compute_times, 95)) if compute_times else 0.0
    capacity_seconds = float(config.NUM_UAVS) * float(slots) * float(config.TIME_SLOT_DURATION)
    rho_service_time_est = float(generated_tasks) * avg_service_time / max(capacity_seconds, 1.0)
    max_available = max(
        (float(env.executor.uav_available_time.get(int(uav.id), 0.0)) for uav in env.uavs),
        default=0.0,
    )

    return {
        "policy": policy,
        "slots": int(slots),
        "seed": int(seed),
        "dag_base_arrival_prob": float(config.DAG_BASE_ARRIVAL_PROB),
        "dag_hotspot_arrival_multiplier": float(config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER),
        "drain_slots_max": int(drain_slots),
        "drain_slots_executed": int(drain_executed),
        "drain_end_reason": drain_end_reason,
        **arrival_snapshot,
        "generated_dags": generated_dags,
        "completed_dags": completed_dags,
        "completion_rate": round(completed_dags / max(generated_dags, 1), 4),
        "generated_tasks": generated_tasks,
        "completed_tasks": completed_tasks,
        "avg_flowtime_raw": round(float(np.mean(flowtimes)), 2) if flowtimes else None,
        "active_dag_backlog_end": active_dag_samples[-1] if active_dag_samples else 0,
        "active_dag_backlog_max": max(active_dag_samples) if active_dag_samples else 0,
        "ready_backlog_end": ready_backlog_samples[-1] if ready_backlog_samples else 0,
        "ready_backlog_mean": round(float(np.mean(ready_backlog_samples)), 2) if ready_backlog_samples else 0.0,
        "ready_backlog_max": max(ready_backlog_samples) if ready_backlog_samples else 0,
        "queue_len_end": queue_len_end,
        "queue_len_mean": round(float(np.mean(queue_total_samples)), 2) if queue_total_samples else 0.0,
        "queue_capacity": int(config.NUM_UAVS) * int(config.CLEAN_MAX_QUEUE_PER_UAV),
        "queue_pressure_mean": round(
            (float(np.mean(queue_total_samples)) / max(float(config.NUM_UAVS * config.CLEAN_MAX_QUEUE_PER_UAV), 1.0))
            if queue_total_samples
            else 0.0,
            4,
        ),
        "offloading_skipped_total": offloading_skipped_total,
        "arrival_zero_dag_slot_ratio": round(
            sum(value == 0.0 for value in arrival_created_dags) / max(len(arrival_created_dags), 1), 4
        ),
        "arrival_multi_dag_slot_ratio": round(
            sum(value >= 2.0 for value in arrival_created_dags) / max(len(arrival_created_dags), 1), 4
        ),
        **_distribution_summary(arrival_created_dags, "new_dags_per_arrival_slot"),
        **_distribution_summary(arrival_eligible_ues, "eligible_ues_per_arrival_slot"),
        **_distribution_summary(arrival_suppressed_ues, "suppressed_ues_per_arrival_slot"),
        **_distribution_summary(arrival_active_dags, "active_dags_per_arrival_slot"),
        **_distribution_summary(arrival_ready_tasks, "ready_tasks_per_arrival_slot"),
        **_distribution_summary(arrival_assignment_counts, "assignments_per_arrival_slot"),
        **_distribution_summary(arrival_skipped_counts, "skipped_ready_per_arrival_slot"),
        "arrival_offloading_skipped_rate": round(
            sum(arrival_skipped_counts)
            / max(sum(arrival_assignment_counts) + sum(arrival_skipped_counts), 1.0),
            6,
        ),
        **_distribution_summary(arrival_queue_pressure, "queue_pressure_per_arrival_slot"),
        **_distribution_summary(arrival_dag_hyperedges, "dag_hyperedges_per_arrival_slot"),
        **_distribution_summary(arrival_khop_hyperedges, "khop_hyperedges_per_arrival_slot"),
        **_distribution_summary(arrival_attribute_hyperedges, "attribute_hyperedges_per_arrival_slot"),
        **_distribution_summary(arrival_partition_hyperedges, "partition_hyperedges_per_arrival_slot"),
        **_distribution_summary(arrival_total_hyperedges, "total_hyperedges_per_arrival_slot"),
        "partition_hyperedge_nonzero_slot_ratio": round(
            sum(value > 0.0 for value in arrival_partition_hyperedges)
            / max(len(arrival_partition_hyperedges), 1),
            4,
        ),
        "partition_status_counts": dict(sorted(partition_status_counts.items())),
        "kahypar_degraded_slot_count": int(
            partition_status_counts.get("degraded_cache", 0)
            + partition_status_counts.get("degraded_no_cache", 0)
        ),
        "kahypar_circuit_open": kahypar_circuit_open,
        "kahypar_cleanup_failed": kahypar_cleanup_failed,
        "kahypar_worker_alive_after_close": kahypar_worker_alive_after_close,
        "kahypar_last_failure_reason": kahypar_last_failure_reason,
        "avg_task_service_time_s": round(avg_service_time, 2),
        "avg_compute_time_s": round(avg_compute_time, 3),
        "p50_compute_time_s": round(p50_compute_time, 3),
        "p95_compute_time_s": round(p95_compute_time, 3),
        "rho_service_time_est": round(rho_service_time_est, 3),
        "avail_drift_vs_slot_index": round(max_available - float(env.time_step), 2),
        "avail_drift_vs_seconds": round(
            max_available - float(env.time_step) * float(config.TIME_SLOT_DURATION), 2
        ),
    }


def parse_float_csv(raw: str) -> list[float]:
    values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def parse_range_csv(raw: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Range '{item}' must use min:max syntax.")
        low_raw, high_raw = item.split(":", 1)
        low = float(low_raw)
        high = float(high_raw)
        if low <= 0.0 or high <= 0.0 or low > high:
            raise ValueError(f"Invalid positive range: {item}")
        ranges.append((low, high))
    if not ranges:
        raise ValueError("Expected at least one range.")
    return ranges


def parse_int_range_csv(raw: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Integer range '{item}' must use min:max syntax.")
        low_raw, high_raw = item.split(":", 1)
        low = int(low_raw)
        high = int(high_raw)
        if low <= 0 or high <= 0 or low > high:
            raise ValueError(f"Invalid positive integer range: {item}")
        ranges.append((low, high))
    if not ranges:
        raise ValueError("Expected at least one integer range.")
    return ranges


def _with_sweep_config(
    *,
    arrival_prob: float,
    input_range: tuple[float, float],
    output_range: tuple[float, float],
    task_constant_range: tuple[int, int],
) -> None:
    config.DAG_BASE_ARRIVAL_PROB = float(arrival_prob)
    config.DAG_ARRIVAL_PROB = float(arrival_prob)
    config.INPUT_DATA_SIZE_MB_RANGE = (float(input_range[0]), float(input_range[1]))
    config.OUTPUT_DATA_SIZE_MB_RANGE = (float(output_range[0]), float(output_range[1]))
    config.TASK_CONSTANT_RANGE = (int(task_constant_range[0]), int(task_constant_range[1]))


def _scenario_key(row: dict) -> tuple[float, tuple[float, float], tuple[float, float], tuple[int, int]]:
    return (
        float(row["arrival_prob"]),
        tuple(float(value) for value in row["input_range"]),
        tuple(float(value) for value in row["output_range"]),
        tuple(int(value) for value in row["task_constant_range"]),
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize_sweep(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, tuple[float, float], tuple[float, float], tuple[int, int]], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_scenario_key(row)].append(row)

    summaries: list[dict] = []
    for (arrival_prob, input_range, output_range, task_constant_range), group in sorted(grouped.items()):
        by_policy = {policy: [row for row in group if row["policy"] == policy] for policy in ("greedy", "random")}
        greedy_rows = by_policy.get("greedy", [])
        random_rows = by_policy.get("random", [])
        all_rows = greedy_rows + random_rows
        summary = {
            "arrival_prob": arrival_prob,
            "input_range": list(input_range),
            "output_range": list(output_range),
            "task_constant_range": list(task_constant_range),
            "greedy_completion_mean": round(_mean([float(row["completion_rate"]) for row in greedy_rows]), 4),
            "random_completion_mean": round(_mean([float(row["completion_rate"]) for row in random_rows]), 4),
            "greedy_queue_pressure_mean": round(_mean([float(row["queue_pressure_mean"]) for row in greedy_rows]), 4),
            "random_queue_pressure_mean": round(_mean([float(row["queue_pressure_mean"]) for row in random_rows]), 4),
            "ready_backlog_mean": round(_mean([float(row["ready_backlog_mean"]) for row in all_rows]), 2),
            "rho_service_time_est_mean": round(_mean([float(row["rho_service_time_est"]) for row in all_rows]), 3),
            "avg_compute_time_s_mean": round(_mean([float(row["avg_compute_time_s"]) for row in all_rows]), 3),
            "p95_compute_time_s_mean": round(_mean([float(row["p95_compute_time_s"]) for row in all_rows]), 3),
            "offloading_skipped_total_mean": round(_mean([float(row["offloading_skipped_total"]) for row in all_rows]), 1),
            "arrival_offloading_skipped_rate_max": round(
                max((float(row["arrival_offloading_skipped_rate"]) for row in all_rows), default=0.0), 6
            ),
            "drain_all_completed": bool(all(row.get("drain_end_reason") == "all_completed" for row in all_rows)),
            "queue_pressure_p90_max": round(
                max((float(row["queue_pressure_per_arrival_slot_p90"]) for row in all_rows), default=0.0), 4
            ),
            "active_dags_mean": round(
                _mean([float(row["active_dags_per_arrival_slot_mean"]) for row in all_rows]), 3
            ),
            "new_dags_per_slot_mean": round(
                _mean([float(row["new_dags_per_arrival_slot_mean"]) for row in all_rows]), 3
            ),
            "multi_dag_slot_ratio_mean": round(
                _mean([float(row["arrival_multi_dag_slot_ratio"]) for row in all_rows]), 4
            ),
            "assignments_per_slot_mean": round(
                _mean([float(row["assignments_per_arrival_slot_mean"]) for row in all_rows]), 3
            ),
            "partition_hyperedges_mean": round(
                _mean([float(row["partition_hyperedges_per_arrival_slot_mean"]) for row in all_rows]), 3
            ),
            "partition_nonzero_slot_ratio_mean": round(
                _mean([float(row["partition_hyperedge_nonzero_slot_ratio"]) for row in all_rows]), 4
            ),
            "kahypar_degraded_slot_count": int(
                sum(int(row["kahypar_degraded_slot_count"]) for row in all_rows)
            ),
            "kahypar_integrity_ok": bool(
                all(
                    not bool(row["kahypar_cleanup_failed"])
                    and not bool(row["kahypar_worker_alive_after_close"])
                    and int(row["kahypar_degraded_slot_count"]) == 0
                    for row in all_rows
                )
            ),
        }
        random_ok = 0.50 <= summary["random_completion_mean"] <= 0.70
        greedy_ok = 0.80 <= summary["greedy_completion_mean"] <= 0.90
        queue_ok = max(summary["greedy_queue_pressure_mean"], summary["random_queue_pressure_mean"]) < 0.90
        summary["gate3_pass"] = bool(random_ok and greedy_ok and queue_ok)
        summary["coarse_safe"] = bool(
            summary["drain_all_completed"]
            and summary["queue_pressure_p90_max"] < 0.95
            and summary["arrival_offloading_skipped_rate_max"] < 0.01
            and summary["kahypar_integrity_ok"]
        )
        # Smaller score is better; keep it simple and transparent for calibration.
        summary["gate3_score"] = round(
            abs(summary["random_completion_mean"] - 0.60)
            + abs(summary["greedy_completion_mean"] - 0.85)
            + max(max(summary["greedy_queue_pressure_mean"], summary["random_queue_pressure_mean"]) - 0.90, 0.0),
            4,
        )
        summaries.append(summary)
    return sorted(summaries, key=lambda row: (not row["gate3_pass"], row["gate3_score"]))


def print_sweep_table(summaries: list[dict]) -> None:
    header = (
        "arrival | input_MB | output_MB | greedy | random | q_g | q_r | "
        "task_c | ready | active | new_dag | actions | q_p90 | part | drain | coarse_safe"
    )
    print("SWEEP_TABLE " + header)
    for row in summaries:
        print(
            "SWEEP_TABLE "
            f"{row['arrival_prob']:.4g} | "
            f"{row['input_range'][0]:g}-{row['input_range'][1]:g} | "
            f"{row['output_range'][0]:g}-{row['output_range'][1]:g} | "
            f"{row['greedy_completion_mean']:.3f} | "
            f"{row['random_completion_mean']:.3f} | "
            f"{row['greedy_queue_pressure_mean']:.3f} | "
            f"{row['random_queue_pressure_mean']:.3f} | "
            f"{row['task_constant_range'][0]}-{row['task_constant_range'][1]} | "
            f"{row['ready_backlog_mean']:.1f} | "
            f"{row['active_dags_mean']:.2f} | "
            f"{row['new_dags_per_slot_mean']:.2f} | "
            f"{row['assignments_per_slot_mean']:.2f} | "
            f"{row['queue_pressure_p90_max']:.3f} | "
            f"{row['partition_hyperedges_mean']:.2f} | "
            f"{row['drain_all_completed']} | "
            f"{row['coarse_safe']}"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _prepare_sweep_output(args: argparse.Namespace, cells: list[dict]) -> Path | None:
    if args.output_dir is None:
        return None
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"--output-dir must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "python": sys.executable,
        "arguments": vars(args) | {"output_dir": str(output_dir)},
        "scene": {
            "dag_hotspot_arrival_multiplier": float(config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER),
            "num_ues": int(config.NUM_UES),
            "num_uavs": int(config.NUM_UAVS),
            "time_slot_duration": float(config.TIME_SLOT_DURATION),
            "queue_capacity_per_uav": int(config.CLEAN_MAX_QUEUE_PER_UAV),
            "kahypar_enabled": bool(config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES),
        },
        "cell_count": len(cells),
        "cells": cells,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(
        output_dir / "progress.json",
        {"status": "running", "completed_cells": 0, "total_cells": len(cells), "started_at_utc": _utc_now()},
    )
    return output_dir


def _write_sweep_reports(output_dir: Path, rows: list[dict], summaries: list[dict]) -> None:
    _write_json(output_dir / "sweep_summary.json", summaries)
    if summaries:
        with (output_dir / "sweep_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    lines = [
        "# Clean Load Boundary Scan",
        "",
        f"Cells: {len(rows)}",
        "",
        "| arrival | greedy | random | active DAG | new DAG/slot | actions/slot | queue P90 max | partition edges | drained | coarse safe |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['arrival_prob']:.4g} | {row['greedy_completion_mean']:.3f} | "
            f"{row['random_completion_mean']:.3f} | {row['active_dags_mean']:.2f} | "
            f"{row['new_dags_per_slot_mean']:.2f} | {row['assignments_per_slot_mean']:.2f} | "
            f"{row['queue_pressure_p90_max']:.3f} | {row['partition_hyperedges_mean']:.2f} | "
            f"{row['drain_all_completed']} | {row['coarse_safe']} |"
        )
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> int:
    arrival_probs = parse_float_csv(args.arrival_probs)
    input_ranges = parse_range_csv(args.input_ranges)
    output_ranges = parse_range_csv(args.output_ranges)
    task_constant_ranges = (
        parse_int_range_csv(args.task_constant_ranges)
        if args.task_constant_ranges is not None
        else [tuple(int(value) for value in config.TASK_CONSTANT_RANGE)]
    )
    seeds = list(args.seeds if args.seeds is not None else [args.seed])

    cells = [
        {
            "cell_index": index,
            "arrival_prob": float(arrival_prob),
            "input_range": [float(value) for value in input_range],
            "output_range": [float(value) for value in output_range],
            "task_constant_range": [int(value) for value in task_constant_range],
            "seed": int(seed),
            "policy": str(policy),
            "arrival_slots": int(args.slots),
            "drain_slots": int(args.drain_slots),
        }
        for index, (arrival_prob, input_range, output_range, task_constant_range, seed, policy) in enumerate(
            (
                (arrival_prob, input_range, output_range, task_constant_range, seed, policy)
                for arrival_prob in arrival_probs
                for input_range in input_ranges
                for output_range in output_ranges
                for task_constant_range in task_constant_ranges
                for seed in seeds
                for policy in args.policies
            ),
            start=1,
        )
    ]
    output_dir = _prepare_sweep_output(args, cells)

    original = {
        "arrival_prob": config.DAG_BASE_ARRIVAL_PROB,
        "arrival_alias": getattr(config, "DAG_ARRIVAL_PROB", config.DAG_BASE_ARRIVAL_PROB),
        "input_range": config.INPUT_DATA_SIZE_MB_RANGE,
        "output_range": config.OUTPUT_DATA_SIZE_MB_RANGE,
        "task_constant_range": config.TASK_CONSTANT_RANGE,
    }
    rows: list[dict] = []
    try:
        for arrival_prob in arrival_probs:
            for input_range in input_ranges:
                for output_range in output_ranges:
                    for task_constant_range in task_constant_ranges:
                        _with_sweep_config(
                            arrival_prob=arrival_prob,
                            input_range=input_range,
                            output_range=output_range,
                            task_constant_range=task_constant_range,
                        )
                        for seed in seeds:
                            for policy in args.policies:
                                result = run_baseline(
                                    policy=policy,
                                    slots=args.slots,
                                    seed=seed,
                                    drain_slots=args.drain_slots,
                                )
                                result["arrival_prob"] = float(arrival_prob)
                                result["input_range"] = [float(input_range[0]), float(input_range[1])]
                                result["output_range"] = [float(output_range[0]), float(output_range[1])]
                                result["task_constant_range"] = [
                                    int(task_constant_range[0]),
                                    int(task_constant_range[1]),
                                ]
                                rows.append(result)
                                print("SWEEP_JSON " + json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
                                if output_dir is not None:
                                    with (output_dir / "sweep_rows.jsonl").open("a", encoding="utf-8") as handle:
                                        handle.write(json.dumps(result, ensure_ascii=True, sort_keys=True) + "\n")
                                    _write_json(
                                        output_dir / "progress.json",
                                        {
                                            "status": "running",
                                            "completed_cells": len(rows),
                                            "total_cells": len(cells),
                                            "last_cell": cells[len(rows) - 1],
                                            "updated_at_utc": _utc_now(),
                                        },
                                    )
    except Exception as exc:
        if output_dir is not None:
            _write_json(
                output_dir / "progress.json",
                {
                    "status": "failed",
                    "completed_cells": len(rows),
                    "total_cells": len(cells),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at_utc": _utc_now(),
                },
            )
        raise
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original["arrival_prob"]
        config.DAG_ARRIVAL_PROB = original["arrival_alias"]
        config.INPUT_DATA_SIZE_MB_RANGE = original["input_range"]
        config.OUTPUT_DATA_SIZE_MB_RANGE = original["output_range"]
        config.TASK_CONSTANT_RANGE = original["task_constant_range"]

    summaries = summarize_sweep(rows)
    if output_dir is not None:
        _write_sweep_reports(output_dir, rows, summaries)
        _write_json(
            output_dir / "progress.json",
            {
                "status": "completed",
                "completed_cells": len(rows),
                "total_cells": len(cells),
                "completed_at_utc": _utc_now(),
            },
        )
    print_sweep_table(summaries)
    for summary in summaries:
        print("SWEEP_SUMMARY_JSON " + json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.sweep:
        return run_sweep(args)

    seeds = list(args.seeds) if args.seeds else [args.seed]
    for policy in args.policies:
      for seed in seeds:
        result = run_baseline(policy=policy, slots=args.slots, seed=seed, drain_slots=args.drain_slots)
        print(
            f"[{result['policy']}] slots={result['slots']} seed={result['seed']} "
            f"gen={result['generated_dags']} comp={result['completed_dags']} "
            f"rate={result['completion_rate']} flow_raw={result['avg_flowtime_raw']} "
            f"active_end={result['active_dag_backlog_end']} ready_mean={result['ready_backlog_mean']} "
            f"queues_end={result['queue_len_end']} skipped={result['offloading_skipped_total']} "
            f"drift_slot={result['avail_drift_vs_slot_index']} drift_sec={result['avail_drift_vs_seconds']}"
            + (
                f" | drain_exec={result['drain_slots_executed']} reason={result['drain_end_reason']} "
                f"arrival_rate={result['completion_rate_arrival_end']}"
                if int(args.drain_slots) > 0
                else ""
            )
        )
        print("DIAG_JSON " + json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
