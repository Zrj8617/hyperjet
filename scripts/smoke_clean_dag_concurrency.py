from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.dag_tasks import DAGTaskManager
from environment.env import Env


def _job_signature(manager: DAGTaskManager, dag_id: str) -> tuple:
    job = manager.get_job(dag_id)
    assert job is not None
    tasks = manager.get_job_tasks(dag_id)
    return (
        job.dag_id,
        job.base_upload_bandwidth_mbps,
        job.base_download_bandwidth_mbps,
        tuple(
            (
                task.level,
                task.input_data_size_mb,
                task.output_data_size_mb,
                task.task_complexity,
                task.task_constant,
                tuple(task.predecessors),
                tuple(task.successors),
            )
            for task in tasks
        ),
    )


def _assert_cap_one_reproduces_default() -> None:
    signatures = []
    for manager in (DAGTaskManager(), DAGTaskManager(max_active_dags_per_ue=1)):
        np.random.seed(20260721)
        job = manager.create_dag_for_ue(7, np.array([12.0, 34.0]), arrival_time=10.0)
        signatures.append(_job_signature(manager, job.dag_id))
        try:
            manager.create_dag_for_ue(7, np.array([12.0, 34.0]), arrival_time=11.0)
        except ValueError:
            pass
        else:
            raise AssertionError("cap=1 admitted a second active DAG")
    assert signatures[0] == signatures[1]


def _assert_cap_two_coexists_and_releases_independently() -> None:
    np.random.seed(20260721)
    env = Env(max_active_dags_per_ue=2)
    env.reset()
    ue = env.ues[0]
    first = env.task_manager.create_dag_for_ue(ue.id, ue.pos[:2], arrival_time=1.0)
    ue.enter_service_waiting(first.dag_id)
    second = env.task_manager.create_dag_for_ue(ue.id, ue.pos[:2], arrival_time=2.0)
    ue.enter_service_waiting(second.dag_id)
    assert env.task_manager.active_dag_count_for_ue(ue.id) == 2

    first.completed = True
    env.release_ue_after_dag_completed(first.dag_id)
    assert ue.service_waiting
    assert ue.active_dag_id == second.dag_id
    assert env.task_manager.active_dag_count_for_ue(ue.id) == 1

    second.completed = True
    env.release_ue_after_dag_completed(second.dag_id)
    assert not ue.service_waiting
    assert ue.active_dag_id is None


def _assert_cap_three_rejects_fourth() -> None:
    np.random.seed(20260721)
    manager = DAGTaskManager(max_active_dags_per_ue=3)
    for arrival_time in (1.0, 2.0, 3.0):
        manager.create_dag_for_ue(9, np.array([1.0, 2.0]), arrival_time=arrival_time)
    assert manager.active_dag_count_for_ue(9) == 3
    assert not manager.can_accept_dag_for_ue(9)
    try:
        manager.create_dag_for_ue(9, np.array([1.0, 2.0]), arrival_time=4.0)
    except ValueError:
        pass
    else:
        raise AssertionError("cap=3 admitted a fourth active DAG")
    assert manager.active_dag_count_for_ue(9) == 3


def main() -> int:
    _assert_cap_one_reproduces_default()
    _assert_cap_two_coexists_and_releases_independently()
    _assert_cap_three_rejects_fourth()
    print("clean DAG concurrency smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
