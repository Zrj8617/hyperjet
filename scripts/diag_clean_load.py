"""Phase 0 load diagnostic for the clean mainline (read-only baselines).

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
import json
from pathlib import Path
import sys

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean mainline load diagnostic (heuristic baselines).")
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policies", nargs="+", default=["greedy", "random"], choices=["greedy", "random"])
    return parser


def run_baseline(policy: str, slots: int, seed: int) -> dict:
    np.random.seed(int(seed))
    env = Env()
    env.reset()

    ready_backlog_samples: list[int] = []
    active_dag_samples: list[int] = []
    queue_total_samples: list[int] = []
    offloading_skipped_total = 0

    for _ in range(int(slots)):
        context = env.prepare_slot_state()
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
                        current_time_step=env.time_step,
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
                selected_uav_id = int(np.random.choice(legal))
            assignments[task.task_id] = selected_uav_id
            reservation.reserve(task.task_id, selected_uav_id)

        _, _, _, info = env.commit_and_advance(assignments=assignments)
        offloading_skipped_total += int(info.get("offloading_skipped_no_candidate", 0))
        active_dag_samples.append(int(info.get("active_dags", 0)))
        queue_total_samples.append(
            sum(len(env.executor.uav_queues.get(int(uav.id), [])) for uav in env.uavs)
        )

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
    max_available = max(
        (float(env.executor.uav_available_time.get(int(uav.id), 0.0)) for uav in env.uavs),
        default=0.0,
    )

    return {
        "policy": policy,
        "slots": int(slots),
        "seed": int(seed),
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
        "offloading_skipped_total": offloading_skipped_total,
        "avail_drift_vs_slot_index": round(max_available - float(env.time_step), 2),
        "avail_drift_vs_seconds": round(
            max_available - float(env.time_step) * float(config.TIME_SLOT_DURATION), 2
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    for policy in args.policies:
        result = run_baseline(policy=policy, slots=args.slots, seed=args.seed)
        print(
            f"[{result['policy']}] slots={result['slots']} seed={result['seed']} "
            f"gen={result['generated_dags']} comp={result['completed_dags']} "
            f"rate={result['completion_rate']} flow_raw={result['avg_flowtime_raw']} "
            f"active_end={result['active_dag_backlog_end']} ready_mean={result['ready_backlog_mean']} "
            f"queues_end={result['queue_len_end']} skipped={result['offloading_skipped_total']} "
            f"drift_slot={result['avail_drift_vs_slot_index']} drift_sec={result['avail_drift_vs_seconds']}"
        )
        print("DIAG_JSON " + json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
