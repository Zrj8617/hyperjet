"""Stage R1-A environment/load-feedback audit.

This diagnostic compares three fixed offloading heuristics under identical
environment configuration and forced-hover UAV movement.  It does not train a
network or modify environment behavior.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.assignment import (  # noqa: E402
    CleanAssignmentBuffer,
    TemporaryReservationState,
    build_offloading_candidate_components,
)
from environment.dag_tasks import TASK_STATE_PENDING  # noqa: E402
from environment.env import Env  # noqa: E402


SEEDS = (42, 86, 1042)
POLICIES = ("random_legal", "nearest_legal", "greedy_eft")
SLOTS = 500
OUTPUT_PATH = ROOT / "logs" / "r1a_environment_load_feedback.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed Stage R1-A load-feedback audit.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Result JSON path. Seeds, policies, episodes, and slots are intentionally fixed.",
    )
    return parser


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _queue_lengths(env: Env) -> list[int]:
    return [
        len(env.executor.uav_queues.get(int(uav.id), []))
        for uav in sorted(env.uavs, key=lambda item: int(item.id))
    ]


def _pending_task_count(env: Env) -> int:
    return sum(
        1
        for task in env.task_manager.tasks.values()
        if str(getattr(task, "state", "")) == TASK_STATE_PENDING
    )


def _select_assignments(
    *,
    env: Env,
    frozen_ready_task_ids: list[str],
    policy: str,
    policy_rng: np.random.Generator,
) -> tuple[CleanAssignmentBuffer, int]:
    reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
    assignments = CleanAssignmentBuffer()
    skipped = 0

    for decision_order, task_id in enumerate(frozen_ready_task_ids):
        task = env.task_manager.get_task(task_id)
        if task is None or not task.is_ready:
            skipped += 1
            continue

        _, _, candidate_mask, candidate_uav_ids, estimates = build_offloading_candidate_components(
            task=task,
            uavs=env.uavs,
            task_manager=env.task_manager,
            executor=env.executor,
            state_view=reservation,
            current_time_seconds=env.current_time_seconds,
            uav_service_positions=env.uav_service_positions,
            ue_service_positions=env.ue_service_positions,
            ues=env.ues,
        )
        legal_indices = [
            index for index, legal in enumerate(candidate_mask.tolist()) if bool(legal)
        ]
        if not legal_indices:
            skipped += 1
            continue

        if policy == "random_legal":
            selected_index = int(policy_rng.choice(legal_indices))
        elif policy == "nearest_legal":
            source = np.asarray(task.source_pos, dtype=np.float64).reshape(-1)[:2]

            def nearest_key(index: int) -> tuple[float, int]:
                uav_id = int(candidate_uav_ids[index])
                target = np.asarray(
                    env.uav_service_positions[uav_id], dtype=np.float64
                ).reshape(-1)[:2]
                return float(np.linalg.norm(source - target)), uav_id

            selected_index = min(legal_indices, key=nearest_key)
        elif policy == "greedy_eft":
            selected_index = min(
                legal_indices,
                key=lambda index: (
                    float(estimates[index].estimated_finish_time),
                    int(candidate_uav_ids[index]),
                ),
            )
        else:
            raise ValueError(f"Unknown fixed policy: {policy}")

        selected_uav_id = int(candidate_uav_ids[selected_index])
        selected_estimate = estimates[selected_index]
        assignments.append(str(task.task_id), selected_uav_id, int(decision_order))
        reservation.reserve(
            str(task.task_id),
            selected_uav_id,
            estimated_available_time=float(selected_estimate.estimated_finish_time),
            estimated_queued_workload=float(selected_estimate.estimated_queued_workload),
        )

    return assignments, skipped


def run_one(*, seed: int, policy: str) -> dict[str, Any]:
    _set_seed(seed)
    policy_rng = np.random.default_rng(int(seed) + 1_000_003)
    env = Env()
    env.reset()

    active_dag_samples: list[float] = []
    ready_task_samples: list[float] = []
    pending_task_samples: list[float] = []
    queue_samples: list[float] = []
    executed_assignments = 0
    invalid_assignments = 0
    latest_info: dict[str, Any] = {}

    for _slot in range(SLOTS):
        context = env.prepare_slot_state()
        frozen_ready_task_ids = [str(value) for value in context["frozen_ready_task_ids"]]
        ready_task_samples.append(float(len(frozen_ready_task_ids)))
        pending_task_samples.append(float(_pending_task_count(env)))

        env.apply_movement({})
        assignments, skipped = _select_assignments(
            env=env,
            frozen_ready_task_ids=frozen_ready_task_ids,
            policy=policy,
            policy_rng=policy_rng,
        )
        _, _, _, latest_info = env.commit_and_advance(
            assignment_buffer=assignments,
            offloading_skip_count=skipped,
        )
        executed_assignments += int(latest_info["newly_assigned_tasks"])
        invalid_assignments += int(latest_info["invalid_assignments"])
        active_dag_samples.append(float(latest_info["active_dags"]))
        queue_samples.extend(float(value) for value in _queue_lengths(env))

    arrival_attempt_count = int(latest_info["arrival_attempt_count"])
    arrival_draw_count = int(latest_info["arrival_draw_count"])
    arrival_admitted_count = int(latest_info["arrival_admitted_count"])
    arrival_blocked_count = int(latest_info["arrival_blocked_count"])
    active_dag_cap_blocked_count = int(
        latest_info.get("arrival_blocked_reasons", {}).get("active_dag_cap", 0)
    )
    generated_dag_count = int(round(float(latest_info["generated_dag_count"])))
    completed_dag_count = int(round(float(latest_info["completed_dag_count"])))
    blocked_fraction = float(arrival_blocked_count / max(arrival_attempt_count, 1))

    return {
        "seed": int(seed),
        "policy": str(policy),
        "slots": SLOTS,
        "arrival_attempt_count": arrival_attempt_count,
        "arrival_draw_count": arrival_draw_count,
        "arrival_admitted_count": arrival_admitted_count,
        "arrival_blocked_count": arrival_blocked_count,
        "active_dag_cap_blocked_count": active_dag_cap_blocked_count,
        "generated_dag_count": generated_dag_count,
        "completed_dag_count": completed_dag_count,
        "completion_rate": float(latest_info["dag_completion_rate"]),
        "throughput": float(latest_info["dag_throughput"]),
        "avg_dag_flowtime": float(latest_info["average_dag_flowtime"]),
        "avg_critical_path_delay": float(
            latest_info["average_critical_path_task_completion_delay"]
        ),
        "mean_active_dag": _mean(active_dag_samples),
        "max_active_dag": max(active_dag_samples, default=0.0),
        "mean_ready_tasks": _mean(ready_task_samples),
        "mean_pending_tasks": _mean(pending_task_samples),
        "avg_uav_queue": _mean(queue_samples),
        "max_uav_queue": max(queue_samples, default=0.0),
        "executed_assignment_count": int(executed_assignments),
        "invalid_assignment_count": int(invalid_assignments),
        "task_energy": float(latest_info["total_task_energy"]),
        "movement_energy": float(latest_info["uav_movement_energy_total"]),
        "episode_reward": float(latest_info["episode_reward"]),
        "blocked_fraction": blocked_fraction,
        "net_active_growth": int(generated_dag_count - completed_dag_count),
        "admitted_per_completed": float(
            arrival_admitted_count / max(completed_dag_count, 1)
        ),
    }


def _main_row(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": run["seed"],
        "policy": run["policy"],
        "completed": run["completed_dag_count"],
        "admitted": run["arrival_admitted_count"],
        "generated": run["generated_dag_count"],
        "blocked_fraction": run["blocked_fraction"],
        "completion_rate": run["completion_rate"],
        "avg_flowtime": run["avg_dag_flowtime"],
        "avg_queue": run["avg_uav_queue"],
        "episode_reward": run["episode_reward"],
    }


def _paired_deltas(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(int(run["seed"]), str(run["policy"])): run for run in runs}
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline = by_key[(seed, "random_legal")]
        for policy, label in (
            ("greedy_eft", "greedy_eft - random"),
            ("nearest_legal", "nearest - random"),
        ):
            candidate = by_key[(seed, policy)]
            rows.append(
                {
                    "seed": seed,
                    "comparison": label,
                    "completed_delta": int(
                        candidate["completed_dag_count"] - baseline["completed_dag_count"]
                    ),
                    "admitted_delta": int(
                        candidate["arrival_admitted_count"] - baseline["arrival_admitted_count"]
                    ),
                    "generated_delta": int(
                        candidate["generated_dag_count"] - baseline["generated_dag_count"]
                    ),
                    "blocked_fraction_delta": float(
                        candidate["blocked_fraction"] - baseline["blocked_fraction"]
                    ),
                }
            )
    return rows


def _print_main_table(rows: list[dict[str, Any]]) -> None:
    columns = (
        "seed",
        "policy",
        "completed",
        "admitted",
        "generated",
        "blocked_fraction",
        "completion_rate",
        "avg_flowtime",
        "avg_queue",
        "episode_reward",
    )
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(str(row[column]) for column in columns))


def main() -> int:
    args = build_arg_parser().parse_args()
    code_commit = _git_head()

    print("Running determinism smoke: seed=42, policy=greedy_eft, repeat=2", flush=True)
    determinism_first = run_one(seed=42, policy="greedy_eft")
    determinism_second = run_one(seed=42, policy="greedy_eft")
    determinism_passed = determinism_first == determinism_second
    if not determinism_passed:
        raise AssertionError("R1-A determinism smoke mismatch; formal 9-run audit not started")
    print("R1-A determinism smoke: PASS", flush=True)

    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        for policy in POLICIES:
            print(f"Running seed={seed} policy={policy} slots={SLOTS}", flush=True)
            runs.append(run_one(seed=seed, policy=policy))

    main_table = [_main_row(run) for run in runs]
    paired = _paired_deltas(runs)
    payload = {
        "schema": "r1a_environment_load_feedback_v1",
        "server_code_commit": code_commit,
        "protocol": {
            "hypothesis": "policy performance changes workload through active-DAG-cap release",
            "seeds": list(SEEDS),
            "policies": list(POLICIES),
            "episodes_per_cell": 1,
            "slots_per_episode": SLOTS,
            "movement": "forced_hover",
            "ready_task_order": "environment_frozen_order",
            "reservation": "sequential_immediate",
            "nearest_legal": "min euclidean distance(task.source_pos, candidate_uav_service_position), tie min uav_id",
            "blocked_fraction": "arrival_blocked_count / max(arrival_attempt_count, 1)",
        },
        "determinism_smoke": {
            "passed": True,
            "seed": 42,
            "policy": "greedy_eft",
            "repeat_count": 2,
            "summary": determinism_first,
        },
        "runs": runs,
        "main_table": main_table,
        "paired_deltas": paired,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_main_table(main_table)
    print(json.dumps(paired, indent=2), flush=True)
    print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
