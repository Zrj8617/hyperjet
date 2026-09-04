"""Potential-based DAG-progress reward shaping for the approved short probe."""

from __future__ import annotations

import math
from typing import Any

from environment.dag_tasks import TASK_STATE_COMPLETED


def cumulative_dag_progress_potential(
    task_manager: Any,
    *,
    completed_dag_weight: float,
) -> float:
    """Return W * (completed DAGs + fractional progress of active DAGs)."""
    weight = float(completed_dag_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("completed-DAG weight must be finite and non-negative")
    progress = 0.0
    for job in task_manager.jobs.values():
        if bool(job.completed):
            progress += 1.0
            continue
        task_ids = [str(task_id) for task_id in job.task_ids]
        if not task_ids:
            continue
        completed = sum(
            1
            for task_id in task_ids
            if task_manager.tasks[task_id].state == TASK_STATE_COMPLETED
        )
        progress += float(completed) / float(len(task_ids))
    return float(weight * progress)


def shape_transition_reward(
    original_reward: float,
    *,
    potential_before: float,
    potential_after: float,
    gamma: float,
    done: bool,
) -> tuple[float, float]:
    """Apply F=gamma*Phi(s')-Phi(s), except F=0 on the baseline done step."""
    reward = float(original_reward)
    discount = float(gamma)
    if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("training gamma must be finite and in [0, 1]")
    shaping = (
        0.0
        if bool(done)
        else discount * float(potential_after) - float(potential_before)
    )
    shaped = reward + shaping
    if not math.isfinite(shaped):
        raise FloatingPointError("non-finite progress-shaped reward")
    return float(shaped), float(shaping)
