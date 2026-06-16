from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.dag_tasks import TASK_STATE_FINISHED, TASK_STATE_RETURNED
from environment.env import Env


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    np.random.seed(7)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    env = Env()
    try:
        env.reset()

        center = np.array([350.0, 350.0], dtype=np.float32)
        env.hotspot_center = center.copy()
        for idx, uav in enumerate(env.uavs):
            uav.pos[:2] = center + np.array([float(idx), 0.0], dtype=np.float32)
        ue = env.ues[0]
        ue.pos[:2] = center.copy()

        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=ue.pos[:2].copy(),
            current_time_step=env.time_step,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()

        ready_tasks = env.task_manager.get_ready_tasks()
        _assert(ready_tasks, "DAG should expose ready entry tasks after creation.")

        for _ in range(1000):
            ready_tasks = env.task_manager.get_ready_tasks()
            assignments = {
                task.task_id: int(task.level % config.NUM_UAVS)
                for task in ready_tasks
            }
            env.step(assignments)
            if job.completed:
                break

        _assert(job.completed, "DAG should complete within 1000 clean execution steps.")
        _assert(job.return_complete_time is not None, "Completed DAG should record return_complete_time.")
        _assert(not ue.service_waiting, "UE should leave service-waiting after DAG completion.")
        _assert(ue.active_dag_id is None, "UE active_dag_id should be released after DAG completion.")

        job_tasks = env.task_manager.get_job_tasks(job.dag_id)
        sink_ids = set(job.sink_task_ids)
        for task in job_tasks:
            if task.task_id in sink_ids:
                _assert(task.state == TASK_STATE_RETURNED, f"Sink task {task.task_id} should be returned.")
            else:
                _assert(
                    task.state in {TASK_STATE_FINISHED, TASK_STATE_RETURNED},
                    f"Non-sink task {task.task_id} should be finished or returned.",
                )

        _assert(any(task.compute_energy > 0.0 for task in job_tasks), "At least one task should record compute energy.")
        _assert(
            any(task.communication_energy > 0.0 for task in job_tasks),
            "At least one task should record upload or inter-transfer communication energy.",
        )
        _assert(
            any(task.return_energy > 0.0 for task in job_tasks if task.task_id in sink_ids),
            "At least one sink task should record return energy.",
        )
        _assert(sum(task.total_energy for task in job_tasks) > 0.0, "Total task energy should be positive.")

        records = env.executor.task_records
        cross_uav_edges = []
        for task in job_tasks:
            for parent_id in task.predecessors:
                parent = env.task_manager.get_task(parent_id)
                if parent is not None and parent.assigned_uav != task.assigned_uav:
                    cross_uav_edges.append((parent_id, task.task_id))
        if cross_uav_edges:
            _assert(
                any(record.inter_transfer_time > 0.0 for record in records.values()),
                "Cross-UAV parent-child execution should record inter-transfer time.",
            )
            _assert(
                any(record.inter_transfer_energy > 0.0 for record in records.values()),
                "Cross-UAV parent-child execution should record inter-transfer energy.",
            )
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob

    print("smoke_clean_execution passed")


if __name__ == "__main__":
    main()
