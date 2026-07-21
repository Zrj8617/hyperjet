"""Clean-mainline Phase 2B ready-to-offload funnel diagnostic.

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
  arrival/final completion, drain usage, and drain completions
  generated_tasks, completed_tasks
  avg_flowtime_raw            raw env time units (unit semantics fixed in Phase 1)
  active_dag_backlog_end/max  unfinished DAG count at end / max over slots
  ready_backlog_end/mean/max  frozen ready-set size statistics
  queue_len_end               per-UAV executor queue lengths at end
  pre/post decision and post-execution queue occupancy and saturation
  capacity_blocked_task_slot_ratio
  ready -> graph -> legal -> selected -> buffer -> executor funnel
  legality-conditioned capacitated matching ceiling and utilization
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
    parser.add_argument("--sweep", action="store_true", help="Run a Phase 2B funnel sweep.")
    parser.add_argument(
        "--arrival-probs",
        type=str,
        default="0.0145,0.029,0.0435",
        help="Comma-separated DAG_BASE_ARRIVAL_PROB values for --sweep.",
    )
    parser.add_argument(
        "--active-dag-caps",
        type=str,
        default="1",
        help="Comma-separated positive per-UE active-DAG caps for --sweep.",
    )
    parser.add_argument(
        "--input-ranges",
        type=str,
        default=f"{config.INPUT_DATA_SIZE_MB_RANGE[0]}:{config.INPUT_DATA_SIZE_MB_RANGE[1]}",
        help="Comma-separated min:max MB ranges for --sweep, e.g. 0.5:12,1:18.",
    )
    parser.add_argument(
        "--output-ranges",
        type=str,
        default=f"{config.OUTPUT_DATA_SIZE_MB_RANGE[0]}:{config.OUTPUT_DATA_SIZE_MB_RANGE[1]}",
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


def _longest_true_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _tail_trend(values: list[float], prefix: str, window: int = 50) -> dict[str, float | int]:
    tail = [float(value) for value in values[-max(int(window), 1):]]
    if not tail:
        return {
            f"{prefix}_window": 0,
            f"{prefix}_start": 0.0,
            f"{prefix}_end": 0.0,
            f"{prefix}_delta": 0.0,
            f"{prefix}_slope_per_slot": 0.0,
        }
    if len(tail) < 2:
        slope = 0.0
    else:
        x = np.arange(len(tail), dtype=np.float64)
        centered_x = x - float(np.mean(x))
        denominator = float(np.dot(centered_x, centered_x))
        slope = float(np.dot(centered_x, np.asarray(tail) - float(np.mean(tail))) / denominator)
    return {
        f"{prefix}_window": len(tail),
        f"{prefix}_start": round(tail[0], 4),
        f"{prefix}_end": round(tail[-1], 4),
        f"{prefix}_delta": round(tail[-1] - tail[0], 4),
        f"{prefix}_slope_per_slot": round(slope, 6),
    }


def _capacitated_matching_size(
    candidate_uav_ids_by_task: dict[str, list[int]],
    residual_capacity_by_uav: dict[int, int],
) -> int:
    """Return a deterministic maximum task-to-capacity-slot matching size."""
    capacity_slots = {
        int(uav_id): [(int(uav_id), slot) for slot in range(max(int(capacity), 0))]
        for uav_id, capacity in residual_capacity_by_uav.items()
    }
    matched_task_by_slot: dict[tuple[int, int], str] = {}

    def _augment(task_id: str, visited_slots: set[tuple[int, int]]) -> bool:
        for uav_id in sorted(set(int(value) for value in candidate_uav_ids_by_task.get(task_id, []))):
            for capacity_slot in capacity_slots.get(uav_id, []):
                if capacity_slot in visited_slots:
                    continue
                visited_slots.add(capacity_slot)
                incumbent = matched_task_by_slot.get(capacity_slot)
                if incumbent is None or _augment(incumbent, visited_slots):
                    matched_task_by_slot[capacity_slot] = task_id
                    return True
        return False

    matched = 0
    for task_id in sorted(str(task_id) for task_id in candidate_uav_ids_by_task):
        if _augment(task_id, set()):
            matched += 1
    return matched


def _per_uav_aggregate(samples: list[list[int]], uav_count: int, prefix: str) -> dict[str, list[float | int]]:
    if not samples:
        return {
            f"{prefix}_per_uav_mean": [0.0 for _ in range(int(uav_count))],
            f"{prefix}_per_uav_max": [0 for _ in range(int(uav_count))],
        }
    values = np.asarray(samples, dtype=np.float64)
    return {
        f"{prefix}_per_uav_mean": [round(float(value), 4) for value in np.mean(values, axis=0)],
        f"{prefix}_per_uav_max": [int(value) for value in np.max(values, axis=0)],
    }


def run_baseline(
    policy: str,
    slots: int,
    seed: int,
    drain_slots: int = 0,
    max_active_dags_per_ue: int = 1,
) -> dict:
    np.random.seed(int(seed))
    # Keep policy sampling out of the environment RNG stream so greedy/random
    # diagnostics do not diverge merely because random offloading consumes RNG.
    policy_rng = np.random.default_rng(int(seed) + 1_000_003)
    env = Env(max_active_dags_per_ue=max_active_dags_per_ue)
    env.reset()
    graph_builder = CleanGraphBuilder()

    ready_backlog_samples: list[int] = []
    active_dag_samples: list[int] = []
    queue_total_samples: list[int] = []
    capacity_blocked_task_slot_total = 0
    arrival_created_dags: list[float] = []
    arrival_eligible_ues: list[float] = []
    arrival_suppressed_ues: list[float] = []
    arrival_active_dags: list[float] = []
    arrival_ready_tasks: list[float] = []
    arrival_assignment_counts: list[float] = []
    arrival_executor_accepted_counts: list[float] = []
    arrival_capacity_blocked_counts: list[float] = []
    arrival_queue_pressure: list[float] = []
    arrival_queue_pre_decision: list[list[int]] = []
    arrival_queue_post_decision: list[list[int]] = []
    arrival_queue_post_execution: list[list[int]] = []
    arrival_initially_full_uav_counts: list[float] = []
    arrival_pre_decision_all_full: list[bool] = []
    arrival_post_decision_all_full: list[bool] = []
    arrival_completed_and_released_counts: list[float] = []
    arrival_legal_capacity_ceilings: list[float] = []
    arrival_legal_capacity_utilizations: list[float] = []
    arrival_zero_legal_capacity: list[bool] = []
    arrival_executor_conversion_rates: list[float] = []
    arrival_slot_funnel: list[dict] = []
    funnel_reason_totals: Counter[str] = Counter()
    funnel_monotonicity_violation_count = 0
    executor_record_count_mismatch_count = 0
    executor_invalid_reason_mismatch_count = 0
    arrival_dag_hyperedges: list[float] = []
    arrival_khop_hyperedges: list[float] = []
    arrival_attribute_hyperedges: list[float] = []
    arrival_partition_hyperedges: list[float] = []
    arrival_total_hyperedges: list[float] = []
    partition_status_counts: Counter[str] = Counter()

    def _execute_one_slot(*, arrival_phase: bool) -> None:
        nonlocal capacity_blocked_task_slot_total, funnel_monotonicity_violation_count
        nonlocal executor_record_count_mismatch_count, executor_invalid_reason_mismatch_count
        eligible_ues = sum(
            1 for ue in env.ues if env.task_manager.can_accept_dag_for_ue(ue.id)
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

        uav_map = {int(uav.id): uav for uav in env.uavs}
        ordered_uav_ids = sorted(uav_map)
        queue_pre_decision = [len(env.executor.uav_queues.get(uav_id, [])) for uav_id in ordered_uav_ids]
        initial_reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
        initial_legal_by_task = {
            task.task_id: legal_candidate_uav_ids(
                task=task,
                uav_ids=ordered_uav_ids,
                state_view=initial_reservation,
                executor=env.executor,
                service_positions=env.uav_service_positions,
            )
            for task in ready_tasks
        }
        residual_capacity_by_uav = {
            uav_id: initial_reservation.remaining_slots(uav_id) for uav_id in ordered_uav_ids
        }
        legal_capacity_ceiling = _capacitated_matching_size(
            initial_legal_by_task,
            residual_capacity_by_uav,
        )
        graph_ready_ids = set(str(task_id) for task_id in graph_snapshot.ready_task_ids)
        graph_ready_tasks = [task for task in ready_tasks if task.task_id in graph_ready_ids]
        graph_missing_count = len(ready_tasks) - len(graph_ready_tasks)
        no_initial_legal_capacity_count = sum(not initial_legal_by_task[task.task_id] for task in ready_tasks)
        graph_ready_with_initial_candidate_count = sum(
            bool(initial_legal_by_task[task.task_id]) for task in graph_ready_tasks
        )

        reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
        assignments: dict[str, int] = {}
        same_slot_reservation_lost_count = 0
        policy_opportunity_count = 0
        candidate_count_by_decision: list[dict] = []
        for task in graph_ready_tasks:
            legal = legal_candidate_uav_ids(
                task=task,
                uav_ids=ordered_uav_ids,
                state_view=reservation,
                executor=env.executor,
                service_positions=env.uav_service_positions,
            )
            if not legal:
                if initial_legal_by_task[task.task_id]:
                    same_slot_reservation_lost_count += 1
                candidate_count_by_decision.append(
                    {
                        "task_id": task.task_id,
                        "initial_candidate_count": len(initial_legal_by_task[task.task_id]),
                        "sequential_candidate_count": 0,
                        "selected_uav_id": None,
                    }
                )
                continue
            policy_opportunity_count += 1
            selected_estimate = None
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
                        selected_estimate = estimate
                selected_uav_id = int(best_uav_id)
            else:
                selected_uav_id = int(policy_rng.choice(legal))
            assignments[task.task_id] = selected_uav_id
            if selected_estimate is None:
                reservation.reserve(task.task_id, selected_uav_id)
            else:
                reservation.reserve(
                    task.task_id,
                    selected_uav_id,
                    estimated_available_time=selected_estimate.estimated_finish_time,
                    estimated_queued_workload=selected_estimate.estimated_queued_workload,
                )
            candidate_count_by_decision.append(
                {
                    "task_id": task.task_id,
                    "initial_candidate_count": len(initial_legal_by_task[task.task_id]),
                    "sequential_candidate_count": len(legal),
                    "selected_uav_id": selected_uav_id,
                }
            )

        policy_selected_count = len(assignments)
        policy_omitted_count = max(policy_opportunity_count - policy_selected_count, 0)
        records_before_commit = set(str(task_id) for task_id in env.executor.task_records)

        _, _, _, info = env.commit_and_advance(assignments=assignments)
        assignment_buffer_accepted_count = int(info.get("assignment_buffer_entry_count", 0))
        executor_accepted_count = int(info.get("newly_assigned_tasks", 0))
        executor_invalid_count = int(info.get("invalid_assignments", 0))
        executor_invalid_reasons = Counter(
            {str(key): int(value) for key, value in info.get("invalid_assignment_reasons", {}).items()}
        )
        new_record_ids = set(str(task_id) for task_id in env.executor.task_records) - records_before_commit
        new_executor_record_count = len(new_record_ids)
        if new_executor_record_count != executor_accepted_count:
            executor_record_count_mismatch_count += 1
        if sum(executor_invalid_reasons.values()) != executor_invalid_count:
            executor_invalid_reason_mismatch_count += 1
        accepted_by_uav: Counter[int] = Counter(
            int(env.executor.task_records[task_id].uav_id) for task_id in new_record_ids
        )
        queue_post_decision = [
            queue_pre_decision[index] + accepted_by_uav.get(uav_id, 0)
            for index, uav_id in enumerate(ordered_uav_ids)
        ]
        queue_post_execution = [len(env.executor.uav_queues.get(uav_id, [])) for uav_id in ordered_uav_ids]
        completed_and_released_count = sum(
            max(before_execution - after_execution, 0)
            for before_execution, after_execution in zip(queue_post_decision, queue_post_execution)
        )
        capacity_blocked_this_slot = no_initial_legal_capacity_count + same_slot_reservation_lost_count
        capacity_blocked_task_slot_total += capacity_blocked_this_slot
        active_dag_samples.append(int(info.get("active_dags", 0)))
        queue_total_samples.append(sum(queue_post_execution))
        if arrival_phase:
            queue_total = float(queue_total_samples[-1])
            queue_capacity = max(float(config.NUM_UAVS * config.CLEAN_MAX_QUEUE_PER_UAV), 1.0)
            initially_full_uav_count = sum(
                queue_len >= int(config.CLEAN_MAX_QUEUE_PER_UAV) for queue_len in queue_pre_decision
            )
            pre_decision_all_full = bool(ordered_uav_ids) and initially_full_uav_count == len(ordered_uav_ids)
            post_decision_all_full = bool(ordered_uav_ids) and all(
                queue_len >= int(config.CLEAN_MAX_QUEUE_PER_UAV) for queue_len in queue_post_decision
            )
            legal_capacity_utilization = (
                float(executor_accepted_count) / float(legal_capacity_ceiling)
                if legal_capacity_ceiling > 0
                else None
            )
            executor_conversion_rate = float(executor_accepted_count) / max(float(len(ready_tasks)), 1.0)
            buffer_rejected_count = max(policy_selected_count - assignment_buffer_accepted_count, 0)
            monotone = (
                executor_accepted_count
                <= assignment_buffer_accepted_count
                <= policy_selected_count
                <= len(ready_tasks)
            )
            if not monotone:
                funnel_monotonicity_violation_count += 1
            slot_reason_counts = {
                "graph_missing": graph_missing_count,
                "no_initial_legal_capacity": no_initial_legal_capacity_count,
                "same_slot_reservation_lost": same_slot_reservation_lost_count,
                "policy_omitted": policy_omitted_count,
                "buffer_rejected": buffer_rejected_count,
                "executor_malformed_uav_id": executor_invalid_reasons.get("malformed_uav_id", 0),
                "executor_illegal_assignment": executor_invalid_reasons.get("illegal_assignment", 0),
                "executor_schedule_record_failure": executor_invalid_reasons.get("schedule_record_failure", 0),
            }
            funnel_reason_totals.update(slot_reason_counts)
            arrival_created_dags.append(float(context["created_dags"]))
            arrival_eligible_ues.append(float(eligible_ues))
            arrival_suppressed_ues.append(float(len(env.ues) - eligible_ues))
            arrival_active_dags.append(float(info.get("active_dags", 0)))
            arrival_ready_tasks.append(float(len(ready_tasks)))
            arrival_assignment_counts.append(float(policy_selected_count))
            arrival_executor_accepted_counts.append(float(executor_accepted_count))
            arrival_capacity_blocked_counts.append(float(capacity_blocked_this_slot))
            arrival_queue_pressure.append(queue_total / queue_capacity)
            arrival_queue_pre_decision.append(queue_pre_decision)
            arrival_queue_post_decision.append(queue_post_decision)
            arrival_queue_post_execution.append(queue_post_execution)
            arrival_initially_full_uav_counts.append(float(initially_full_uav_count))
            arrival_pre_decision_all_full.append(pre_decision_all_full)
            arrival_post_decision_all_full.append(post_decision_all_full)
            arrival_completed_and_released_counts.append(float(completed_and_released_count))
            arrival_legal_capacity_ceilings.append(float(legal_capacity_ceiling))
            arrival_zero_legal_capacity.append(legal_capacity_ceiling == 0)
            if legal_capacity_utilization is not None:
                arrival_legal_capacity_utilizations.append(legal_capacity_utilization)
            arrival_executor_conversion_rates.append(executor_conversion_rate)
            arrival_slot_funnel.append(
                {
                    "slot": int(context["slot_index"]),
                    "frozen_ready_count": len(ready_tasks),
                    "ready_in_graph_count": len(graph_ready_tasks),
                    "ready_with_initial_legal_candidate_count": graph_ready_with_initial_candidate_count,
                    "policy_selected_count": policy_selected_count,
                    "assignment_buffer_accepted_count": assignment_buffer_accepted_count,
                    "executor_accepted_count": executor_accepted_count,
                    "new_executor_record_count": new_executor_record_count,
                    "executor_invalid_count": executor_invalid_count,
                    "legal_capacity_ceiling": legal_capacity_ceiling,
                    "legal_capacity_utilization": (
                        round(legal_capacity_utilization, 6) if legal_capacity_utilization is not None else None
                    ),
                    "executor_to_frozen_ready_conversion": round(executor_conversion_rate, 6),
                    "queue_pre_decision": queue_pre_decision,
                    "queue_post_decision": queue_post_decision,
                    "queue_post_execution": queue_post_execution,
                    "initially_full_uav_count": initially_full_uav_count,
                    "pre_decision_all_uav_full": pre_decision_all_full,
                    "post_decision_all_uav_full": post_decision_all_full,
                    "completed_and_released_queue_positions": completed_and_released_count,
                    "initial_candidate_count_by_task": {
                        task_id: len(candidate_ids) for task_id, candidate_ids in initial_legal_by_task.items()
                    },
                    "candidate_count_by_decision": candidate_count_by_decision,
                    "reason_counts": slot_reason_counts,
                    "funnel_monotone": monotone,
                }
            )
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
        arrival_completion_rate = round(arrival_completed / max(arrival_generated, 1), 4)
        arrival_snapshot = {
            "arrival_generated_dags": int(arrival_generated),
            "arrival_completed_dags": int(arrival_completed),
            "arrival_completion_rate": arrival_completion_rate,
            "completion_rate_arrival_end": arrival_completion_rate,
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
    final_ready_backlog = len(env.task_manager.get_ready_tasks())
    final_active_dag_backlog = sum(
        1 for job in env.task_manager.jobs.values() if not job.completed
    )
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
    final_completion_rate = round(completed_dags / max(generated_dags, 1), 4)
    drain_completed = max(int(completed_dags) - int(arrival_completed), 0)
    queue_pre_decision_totals = [float(sum(values)) for values in arrival_queue_pre_decision]
    queue_post_decision_totals = [float(sum(values)) for values in arrival_queue_post_decision]
    queue_post_execution_totals = [float(sum(values)) for values in arrival_queue_post_execution]
    frozen_ready_total = sum(arrival_ready_tasks)
    capacity_blocked_task_slot_ratio = round(
        sum(arrival_capacity_blocked_counts) / max(frozen_ready_total, 1.0),
        6,
    )
    funnel_fields = (
        "frozen_ready_count",
        "ready_in_graph_count",
        "ready_with_initial_legal_candidate_count",
        "policy_selected_count",
        "assignment_buffer_accepted_count",
        "executor_accepted_count",
        "executor_invalid_count",
    )
    funnel_summaries: dict[str, float] = {}
    for field_name in funnel_fields:
        funnel_summaries.update(
            _distribution_summary(
                [float(row[field_name]) for row in arrival_slot_funnel],
                f"{field_name}_per_arrival_slot",
            )
        )

    return {
        "policy": policy,
        "slots": int(slots),
        "seed": int(seed),
        "max_active_dags_per_ue": int(env.max_active_dags_per_ue),
        "dag_base_arrival_prob": float(config.DAG_BASE_ARRIVAL_PROB),
        "dag_hotspot_arrival_multiplier": float(config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER),
        "drain_slots_max": int(drain_slots),
        "drain_slots_used": int(drain_executed),
        "drain_slots_executed": int(drain_executed),
        "drain_end_reason": drain_end_reason,
        "drain_completed": drain_completed,
        **arrival_snapshot,
        "generated_dags": generated_dags,
        "completed_dags": completed_dags,
        "final_completion_rate": final_completion_rate,
        "completion_rate": final_completion_rate,
        "generated_tasks": generated_tasks,
        "completed_tasks": completed_tasks,
        "avg_flowtime_raw": round(float(np.mean(flowtimes)), 2) if flowtimes else None,
        "active_dag_backlog_end": int(final_active_dag_backlog),
        "active_dag_backlog_max": max(active_dag_samples) if active_dag_samples else 0,
        "ready_backlog_end": int(final_ready_backlog),
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
        **_distribution_summary(queue_pre_decision_totals, "queue_total_pre_decision_per_arrival_slot"),
        **_distribution_summary(queue_post_decision_totals, "queue_total_post_decision_per_arrival_slot"),
        **_distribution_summary(queue_post_execution_totals, "queue_total_post_execution_per_arrival_slot"),
        **_per_uav_aggregate(arrival_queue_pre_decision, len(env.uavs), "queue_len_pre_decision"),
        **_per_uav_aggregate(arrival_queue_post_decision, len(env.uavs), "queue_len_post_decision"),
        **_per_uav_aggregate(arrival_queue_post_execution, len(env.uavs), "queue_len_post_execution"),
        **_distribution_summary(arrival_initially_full_uav_counts, "initially_full_uavs_per_arrival_slot"),
        "pre_decision_all_uav_full_slot_fraction": round(
            sum(arrival_pre_decision_all_full) / max(len(arrival_pre_decision_all_full), 1), 6
        ),
        "pre_decision_longest_consecutive_all_uav_full_slots": _longest_true_run(
            arrival_pre_decision_all_full
        ),
        "post_decision_all_uav_full_slot_fraction": round(
            sum(arrival_post_decision_all_full) / max(len(arrival_post_decision_all_full), 1), 6
        ),
        "post_decision_longest_consecutive_all_uav_full_slots": _longest_true_run(
            arrival_post_decision_all_full
        ),
        **_distribution_summary(
            arrival_completed_and_released_counts,
            "completed_and_released_queue_positions_per_arrival_slot",
        ),
        **_tail_trend(arrival_ready_tasks, "ready_backlog_final50"),
        **_tail_trend(arrival_active_dags, "active_dag_backlog_final50"),
        "capacity_blocked_task_slot_total": int(sum(arrival_capacity_blocked_counts)),
        "capacity_blocked_task_slot_total_all_phases": int(capacity_blocked_task_slot_total),
        "capacity_blocked_task_slot_ratio": capacity_blocked_task_slot_ratio,
        "offloading_skipped_total": int(sum(arrival_capacity_blocked_counts)),
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
        **_distribution_summary(arrival_executor_accepted_counts, "executor_accepted_per_arrival_slot"),
        **_distribution_summary(arrival_capacity_blocked_counts, "capacity_blocked_tasks_per_arrival_slot"),
        **_distribution_summary(arrival_capacity_blocked_counts, "skipped_ready_per_arrival_slot"),
        "arrival_offloading_skipped_rate": capacity_blocked_task_slot_ratio,
        **_distribution_summary(arrival_legal_capacity_ceilings, "legal_capacity_ceiling_per_arrival_slot"),
        **_distribution_summary(
            arrival_legal_capacity_utilizations,
            "legal_capacity_utilization_nonzero_ceiling",
        ),
        "zero_legal_capacity_slot_count": int(sum(arrival_zero_legal_capacity)),
        "zero_legal_capacity_slot_fraction": round(
            sum(arrival_zero_legal_capacity) / max(len(arrival_zero_legal_capacity), 1), 6
        ),
        **_distribution_summary(
            arrival_executor_conversion_rates,
            "executor_to_frozen_ready_conversion_per_arrival_slot",
        ),
        **funnel_summaries,
        "funnel_reason_totals": dict(sorted(funnel_reason_totals.items())),
        "funnel_monotonicity_violation_count": int(funnel_monotonicity_violation_count),
        "executor_record_count_mismatch_count": int(executor_record_count_mismatch_count),
        "executor_invalid_reason_mismatch_count": int(executor_invalid_reason_mismatch_count),
        "arrival_slot_funnel": arrival_slot_funnel,
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


def parse_positive_int_csv(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("Expected at least one positive integer value.")
    if len(set(values)) != len(values):
        raise ValueError("Active-DAG caps must not contain duplicates.")
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


def _scenario_key(row: dict) -> tuple[int, float, tuple[float, float], tuple[float, float], tuple[int, int]]:
    return (
        int(row["max_active_dags_per_ue"]),
        float(row["arrival_prob"]),
        tuple(float(value) for value in row["input_range"]),
        tuple(float(value) for value in row["output_range"]),
        tuple(int(value) for value in row["task_constant_range"]),
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _cap_pair_key(row: dict) -> tuple:
    """Identify one paired cap cell while excluding the cap itself."""
    return (
        float(row["arrival_prob"]),
        tuple(float(value) for value in row["input_range"]),
        tuple(float(value) for value in row["output_range"]),
        tuple(int(value) for value in row["task_constant_range"]),
        int(row["seed"]),
        str(row["policy"]),
    )


def summarize_sweep(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, float, tuple[float, float], tuple[float, float], tuple[int, int]], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_scenario_key(row)].append(row)

    cap_one_by_pair = {
        _cap_pair_key(row): row
        for row in rows
        if int(row["max_active_dags_per_ue"]) == 1
    }
    summaries: list[dict] = []
    for (active_dag_cap, arrival_prob, input_range, output_range, task_constant_range), group in sorted(grouped.items()):
        by_policy = {policy: [row for row in group if row["policy"] == policy] for policy in ("greedy", "random")}
        greedy_rows = by_policy.get("greedy", [])
        random_rows = by_policy.get("random", [])
        all_rows = greedy_rows + random_rows
        paired_deltas = [
            float(row["executor_accepted_per_arrival_slot_mean"])
            - float(cap_one_by_pair[_cap_pair_key(row)]["executor_accepted_per_arrival_slot_mean"])
            for row in all_rows
            if _cap_pair_key(row) in cap_one_by_pair
        ]
        greedy_paired_deltas = [
            float(row["executor_accepted_per_arrival_slot_mean"])
            - float(cap_one_by_pair[_cap_pair_key(row)]["executor_accepted_per_arrival_slot_mean"])
            for row in greedy_rows
            if _cap_pair_key(row) in cap_one_by_pair
        ]
        random_paired_deltas = [
            float(row["executor_accepted_per_arrival_slot_mean"])
            - float(cap_one_by_pair[_cap_pair_key(row)]["executor_accepted_per_arrival_slot_mean"])
            for row in random_rows
            if _cap_pair_key(row) in cap_one_by_pair
        ]
        reason_keys = sorted(
            {
                str(reason)
                for row in all_rows
                for reason in row.get("funnel_reason_totals", {})
            }
        )
        reason_totals = {
            reason: int(sum(int(row.get("funnel_reason_totals", {}).get(reason, 0)) for row in all_rows))
            for reason in reason_keys
        }
        summary = {
            "max_active_dags_per_ue": active_dag_cap,
            "arrival_prob": arrival_prob,
            "input_range": list(input_range),
            "output_range": list(output_range),
            "task_constant_range": list(task_constant_range),
            "greedy_arrival_completion_mean": round(
                _mean([float(row["arrival_completion_rate"]) for row in greedy_rows]), 4
            ),
            "random_arrival_completion_mean": round(
                _mean([float(row["arrival_completion_rate"]) for row in random_rows]), 4
            ),
            "greedy_final_completion_mean": round(
                _mean([float(row["final_completion_rate"]) for row in greedy_rows]), 4
            ),
            "random_final_completion_mean": round(
                _mean([float(row["final_completion_rate"]) for row in random_rows]), 4
            ),
            "greedy_completion_mean": round(
                _mean([float(row["final_completion_rate"]) for row in greedy_rows]), 4
            ),
            "random_completion_mean": round(
                _mean([float(row["final_completion_rate"]) for row in random_rows]), 4
            ),
            "greedy_queue_pressure_mean": round(_mean([float(row["queue_pressure_mean"]) for row in greedy_rows]), 4),
            "random_queue_pressure_mean": round(_mean([float(row["queue_pressure_mean"]) for row in random_rows]), 4),
            "ready_backlog_mean": round(_mean([float(row["ready_backlog_mean"]) for row in all_rows]), 2),
            "rho_service_time_est_mean": round(_mean([float(row["rho_service_time_est"]) for row in all_rows]), 3),
            "avg_compute_time_s_mean": round(_mean([float(row["avg_compute_time_s"]) for row in all_rows]), 3),
            "p95_compute_time_s_mean": round(_mean([float(row["p95_compute_time_s"]) for row in all_rows]), 3),
            "capacity_blocked_task_slot_total_mean": round(
                _mean([float(row["capacity_blocked_task_slot_total"]) for row in all_rows]), 1
            ),
            "capacity_blocked_task_slot_ratio_mean": round(
                _mean([float(row["capacity_blocked_task_slot_ratio"]) for row in all_rows]), 6
            ),
            "capacity_blocked_task_slot_ratio_max": round(
                max((float(row["capacity_blocked_task_slot_ratio"]) for row in all_rows), default=0.0), 6
            ),
            "drain_all_completed": bool(all(row.get("drain_end_reason") == "all_completed" for row in all_rows)),
            "drain_slots_used_mean": round(_mean([float(row["drain_slots_used"]) for row in all_rows]), 2),
            "drain_completed_mean": round(_mean([float(row["drain_completed"]) for row in all_rows]), 2),
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
            "executor_accepted_per_slot_mean": round(
                _mean([float(row["executor_accepted_per_arrival_slot_mean"]) for row in all_rows]), 3
            ),
            "paired_executor_accepted_delta_vs_cap1_mean": round(_mean(paired_deltas), 4),
            "greedy_paired_executor_accepted_delta_vs_cap1_mean": round(_mean(greedy_paired_deltas), 4),
            "random_paired_executor_accepted_delta_vs_cap1_mean": round(_mean(random_paired_deltas), 4),
            "paired_executor_accepted_delta_cell_count": len(paired_deltas),
            "greedy_executor_accepted_per_slot_mean": round(
                _mean([float(row["executor_accepted_per_arrival_slot_mean"]) for row in greedy_rows]), 3
            ),
            "random_executor_accepted_per_slot_mean": round(
                _mean([float(row["executor_accepted_per_arrival_slot_mean"]) for row in random_rows]), 3
            ),
            "legal_capacity_utilization_mean": round(
                _mean(
                    [
                        float(row["legal_capacity_utilization_nonzero_ceiling_mean"])
                        for row in all_rows
                    ]
                ),
                4,
            ),
            "zero_legal_capacity_slot_fraction_mean": round(
                _mean([float(row["zero_legal_capacity_slot_fraction"]) for row in all_rows]), 6
            ),
            "pre_decision_all_uav_full_slot_fraction_mean": round(
                _mean([float(row["pre_decision_all_uav_full_slot_fraction"]) for row in all_rows]), 6
            ),
            "pre_decision_longest_all_full_run_max": max(
                (int(row["pre_decision_longest_consecutive_all_uav_full_slots"]) for row in all_rows),
                default=0,
            ),
            "post_decision_all_uav_full_slot_fraction_mean": round(
                _mean([float(row["post_decision_all_uav_full_slot_fraction"]) for row in all_rows]), 6
            ),
            "post_decision_longest_all_full_run_max": max(
                (int(row["post_decision_longest_consecutive_all_uav_full_slots"]) for row in all_rows),
                default=0,
            ),
            "arrival_end_ready_backlog_mean": round(
                _mean([float(row["ready_backlog_arrival_end"]) for row in all_rows]), 2
            ),
            "arrival_end_active_dag_backlog_mean": round(
                _mean([float(row["active_dag_backlog_arrival_end"]) for row in all_rows]), 2
            ),
            "final_ready_backlog_mean": round(
                _mean([float(row["ready_backlog_end"]) for row in all_rows]), 2
            ),
            "final_active_dag_backlog_mean": round(
                _mean([float(row["active_dag_backlog_end"]) for row in all_rows]), 2
            ),
            "ready_backlog_final50_slope_mean": round(
                _mean([float(row["ready_backlog_final50_slope_per_slot"]) for row in all_rows]), 6
            ),
            "active_dag_backlog_final50_slope_mean": round(
                _mean([float(row["active_dag_backlog_final50_slope_per_slot"]) for row in all_rows]), 6
            ),
            "funnel_reason_totals": reason_totals,
            "funnel_monotonicity_violation_count": int(
                sum(int(row["funnel_monotonicity_violation_count"]) for row in all_rows)
            ),
            "executor_record_count_mismatch_count": int(
                sum(int(row["executor_record_count_mismatch_count"]) for row in all_rows)
            ),
            "executor_invalid_reason_mismatch_count": int(
                sum(int(row["executor_invalid_reason_mismatch_count"]) for row in all_rows)
            ),
            "partition_hyperedges_mean": round(
                _mean([float(row["partition_hyperedges_per_arrival_slot_mean"]) for row in all_rows]), 3
            ),
            "total_hyperedges_mean": round(
                _mean([float(row["total_hyperedges_per_arrival_slot_mean"]) for row in all_rows]), 3
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
            "feasible": bool(
                all(
                    row.get("drain_end_reason") == "all_completed"
                    and int(row["active_dag_backlog_end"]) == 0
                    and int(row["ready_backlog_end"]) == 0
                    and np.isfinite(float(row["executor_accepted_per_arrival_slot_mean"]))
                    and np.isfinite(float(row["capacity_blocked_task_slot_ratio"]))
                    for row in all_rows
                )
            ),
        }
        summaries.append(summary)
    return summaries


def print_sweep_table(summaries: list[dict]) -> None:
    header = (
        "cap | arrival | arr_g | arr_r | final_g | final_r | accept_g | accept_r | delta1 | "
        "legal_util | blocked | full_post | full_run | ready_end | drain"
    )
    print("SWEEP_TABLE " + header)
    for row in summaries:
        print(
            "SWEEP_TABLE "
            f"{row['max_active_dags_per_ue']} | "
            f"{row['arrival_prob']:.4g} | "
            f"{row['greedy_arrival_completion_mean']:.3f} | "
            f"{row['random_arrival_completion_mean']:.3f} | "
            f"{row['greedy_final_completion_mean']:.3f} | "
            f"{row['random_final_completion_mean']:.3f} | "
            f"{row['greedy_executor_accepted_per_slot_mean']:.2f} | "
            f"{row['random_executor_accepted_per_slot_mean']:.2f} | "
            f"{row['paired_executor_accepted_delta_vs_cap1_mean']:+.3f} | "
            f"{row['legal_capacity_utilization_mean']:.3f} | "
            f"{row['capacity_blocked_task_slot_ratio_mean']:.3f} | "
            f"{row['post_decision_all_uav_full_slot_fraction_mean']:.3f} | "
            f"{row['post_decision_longest_all_full_run_max']} | "
            f"{row['arrival_end_ready_backlog_mean']:.1f} | "
            f"{row['drain_all_completed']}"
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
        "schema_version": 3,
        "diagnostic": "phase2b_ready_to_offload_funnel_stage1",
        "created_at_utc": _utc_now(),
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "python": sys.executable,
        "arguments": vars(args) | {"output_dir": str(output_dir)},
        "scene": {
            "movement": "fixed_hover",
            "learning": False,
            "policy_rng_isolated": True,
            "dag_hotspot_arrival_multiplier": float(config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER),
            "num_ues": int(config.NUM_UES),
            "num_uavs": int(config.NUM_UAVS),
            "time_slot_duration": float(config.TIME_SLOT_DURATION),
            "queue_capacity_per_uav": int(config.CLEAN_MAX_QUEUE_PER_UAV),
            "active_dag_caps": parse_positive_int_csv(args.active_dag_caps),
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
        "# Phase 2B Ready-to-Offload Funnel",
        "",
        f"Cells: {len(rows)}",
        "",
        "| cap | arrival | arrival comp. G/R | final comp. G/R | accepted/slot G/R | paired delta vs cap 1 | legal util. | blocked ratio | post-full frac. | longest full | ready end | feasible |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['max_active_dags_per_ue']} | {row['arrival_prob']:.4g} | "
            f"{row['greedy_arrival_completion_mean']:.3f}/{row['random_arrival_completion_mean']:.3f} | "
            f"{row['greedy_final_completion_mean']:.3f}/{row['random_final_completion_mean']:.3f} | "
            f"{row['greedy_executor_accepted_per_slot_mean']:.2f}/{row['random_executor_accepted_per_slot_mean']:.2f} | "
            f"{row['paired_executor_accepted_delta_vs_cap1_mean']:+.3f} | "
            f"{row['legal_capacity_utilization_mean']:.3f} | "
            f"{row['capacity_blocked_task_slot_ratio_mean']:.3f} | "
            f"{row['post_decision_all_uav_full_slot_fraction_mean']:.3f} | "
            f"{row['post_decision_longest_all_full_run_max']} | "
            f"{row['arrival_end_ready_backlog_mean']:.1f} | {row['feasible']} |"
        )
    lines.extend(
        [
            "",
            "The table is diagnostic evidence only. Branch 1/2/3 selection requires reviewed funnel reasons and is not automatic.",
        ]
    )
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> int:
    active_dag_caps = parse_positive_int_csv(args.active_dag_caps)
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
            "max_active_dags_per_ue": int(active_dag_cap),
            "arrival_prob": float(arrival_prob),
            "input_range": [float(value) for value in input_range],
            "output_range": [float(value) for value in output_range],
            "task_constant_range": [int(value) for value in task_constant_range],
            "seed": int(seed),
            "policy": str(policy),
            "arrival_slots": int(args.slots),
            "drain_slots": int(args.drain_slots),
        }
        for index, (active_dag_cap, arrival_prob, input_range, output_range, task_constant_range, seed, policy) in enumerate(
            (
                (active_dag_cap, arrival_prob, input_range, output_range, task_constant_range, seed, policy)
                for active_dag_cap in active_dag_caps
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
        for active_dag_cap in active_dag_caps:
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
                                        max_active_dags_per_ue=active_dag_cap,
                                    )
                                    result["arrival_prob"] = float(arrival_prob)
                                    result["input_range"] = [float(input_range[0]), float(input_range[1])]
                                    result["output_range"] = [float(output_range[0]), float(output_range[1])]
                                    result["task_constant_range"] = [
                                        int(task_constant_range[0]),
                                        int(task_constant_range[1]),
                                    ]
                                    rows.append(result)
                                    console_result = {
                                        key: value for key, value in result.items() if key != "arrival_slot_funnel"
                                    }
                                    print(
                                        "SWEEP_JSON " + json.dumps(console_result, ensure_ascii=True, sort_keys=True),
                                        flush=True,
                                    )
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
            f"arrival_rate={result['arrival_completion_rate']} "
            f"final_rate={result['final_completion_rate']} flow_raw={result['avg_flowtime_raw']} "
            f"active_end={result['active_dag_backlog_end']} ready_mean={result['ready_backlog_mean']} "
            f"accepted_per_slot={result['executor_accepted_per_arrival_slot_mean']} "
            f"legal_util={result['legal_capacity_utilization_nonzero_ceiling_mean']} "
            f"blocked_ratio={result['capacity_blocked_task_slot_ratio']} "
            f"queues_end={result['queue_len_end']} "
            f"drift_slot={result['avail_drift_vs_slot_index']} drift_sec={result['avail_drift_vs_seconds']}"
            + (
                f" | drain_used={result['drain_slots_used']} drain_completed={result['drain_completed']} "
                f"reason={result['drain_end_reason']}"
                if int(args.drain_slots) > 0
                else ""
            )
        )
        print("DIAG_JSON " + json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
