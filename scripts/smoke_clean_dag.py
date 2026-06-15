from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import config
from environment.dag_tasks import DAGTaskManager


def _assert_acyclic(manager: DAGTaskManager, dag_id: str) -> None:
    tasks = manager.get_job_tasks(dag_id)
    task_map = {task.task_id: task for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise AssertionError(f"cycle detected at {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for child_id in task_map[task_id].successors:
            visit(child_id)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in task_map:
        visit(task_id)


def _assert_finished_task_is_not_active() -> None:
    manager = DAGTaskManager()
    job = manager.create_dag_for_ue(
        ue_id=0,
        source_pos=np.array([10.0, 20.0], dtype=np.float32),
        arrival_time=0,
    )
    ready_task = manager.get_ready_tasks()[0]
    manager.mark_task_finished(ready_task.task_id, current_time_step=1.0)

    active_task_ids = {task.task_id for task in manager.get_active_tasks()}
    assert ready_task.task_id not in active_task_ids
    assert manager.tasks[ready_task.task_id] is ready_task
    if hasattr(manager, "get_all_non_returned_tasks"):
        non_returned_ids = {task.task_id for task in manager.get_all_non_returned_tasks()}
        assert ready_task.task_id in non_returned_ids


def main() -> None:
    np.random.seed(config.SEED)
    _assert_finished_task_is_not_active()

    manager = DAGTaskManager()
    for idx in range(100):
        source_pos = np.array([float(idx % 10) * 10.0, float(idx // 10) * 10.0], dtype=np.float32)
        job = manager.create_dag_for_ue(ue_id=idx % config.NUM_UES, source_pos=source_pos, arrival_time=idx)
        tasks = manager.get_job_tasks(job.dag_id)

        assert config.DAG_MIN_TASKS <= len(tasks) <= config.DAG_MAX_TASKS
        assert max(task.level for task in tasks) < config.DAG_MAX_LEVELS
        assert job.base_upload_bandwidth_mbps in config.BASE_UPLOAD_BANDWIDTH_MBPS
        assert job.base_download_bandwidth_mbps in config.BASE_DOWNLOAD_BANDWIDTH_MBPS
        assert job.source_pos.shape == (2,)
        assert len(job.sink_task_ids) >= 1
        assert all(not manager.tasks[task_id].successors for task_id in job.sink_task_ids)

        critical_tasks = [task for task in tasks if task.is_critical_path]
        assert len(critical_tasks) >= 1

        for task in tasks:
            assert task.task_complexity in config.TASK_COMPLEXITY_PROBS
            assert config.INPUT_DATA_SIZE_MB_RANGE[0] <= task.input_data_size_mb <= config.INPUT_DATA_SIZE_MB_RANGE[1]
            assert config.OUTPUT_DATA_SIZE_MB_RANGE[0] <= task.output_data_size_mb <= config.OUTPUT_DATA_SIZE_MB_RANGE[1]
            assert config.TASK_CONSTANT_RANGE[0] <= task.task_constant <= config.TASK_CONSTANT_RANGE[1]
            assert task.num_operation > 0.0
            assert task.source_pos.shape == (2,)
            assert getattr(task, "deadline") is None
            assert getattr(task, "task_type") is None
            if task.level == 0:
                assert len(task.predecessors) == 0
            else:
                assert 1 <= len(task.predecessors) <= config.DAG_MAX_PARENTS
                for parent_id in task.predecessors:
                    parent = manager.tasks[parent_id]
                    assert parent.level < task.level
            for child_id in task.successors:
                child = manager.tasks[child_id]
                assert task.task_id in child.predecessors
                assert task.level < child.level

        _assert_acyclic(manager, job.dag_id)

    print("clean DAG smoke passed")


if __name__ == "__main__":
    main()
