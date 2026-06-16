from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import config
import numpy as np


TASK_STATE_PENDING = "pending"
TASK_STATE_READY = "ready"
TASK_STATE_QUEUED = "queued"
TASK_STATE_RUNNING = "running"
TASK_STATE_FINISHED = "finished"
TASK_STATE_RETURNED = "returned"
TASK_STATE_DROPPED = "dropped"  # Deprecated compatibility state; clean mainline does not deadline-drop tasks.

# Deprecated aliases kept for old imports.
TASK_STATE_WAITING = TASK_STATE_PENDING


@dataclass(slots=True)
class TaskNode:
    task_id: str
    dag_id: str
    ue_id: int
    input_data_size_mb: float
    output_data_size_mb: float
    task_complexity: str
    task_constant: int
    num_operation: float
    level: int
    source_pos: np.ndarray
    arrival_time: float = 0.0
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    ready_time: float | None = None
    start_time: float | None = None
    finish_time: float | None = None
    assigned_uav: int | None = None
    state: str = TASK_STATE_PENDING
    is_critical_path: bool = False
    compute_energy: float = 0.0
    communication_energy: float = 0.0
    return_energy: float = 0.0
    total_energy: float = 0.0
    enqueue_time: float | None = None
    # Deprecated compatibility fields. They must not participate in clean reward,
    # graph construction, candidate filtering, critical path, or main metrics.
    deadline: float | None = None
    task_type: int | None = None

    @property
    def input_size(self) -> int:
        """Deprecated byte-size alias for old modules."""
        return int(round(self.input_data_size_mb * 1024 * 1024))

    @property
    def output_size(self) -> int:
        """Deprecated byte-size alias for old modules."""
        return int(round(self.output_data_size_mb * 1024 * 1024))

    @property
    def cpu_cycles(self) -> float:
        """Deprecated compute alias; clean mainline uses num_operation."""
        return float(self.num_operation)

    @property
    def is_ready(self) -> bool:
        return self.state == TASK_STATE_READY

    @property
    def is_terminal(self) -> bool:
        return self.state in {TASK_STATE_FINISHED, TASK_STATE_RETURNED, TASK_STATE_DROPPED}

    @property
    def is_computation_finished(self) -> bool:
        return self.state in {TASK_STATE_FINISHED, TASK_STATE_RETURNED}

    @property
    def is_fully_completed(self) -> bool:
        return self.state in {TASK_STATE_FINISHED, TASK_STATE_RETURNED}

    def remaining_slack(self, current_time_step: float) -> float:
        """Deprecated compatibility helper. Clean mainline does not use slack."""
        if self.deadline is None:
            return float("inf")
        return float(self.deadline - current_time_step)


@dataclass(slots=True)
class DAGJob:
    dag_id: str
    ue_id: int
    arrival_time: float
    source_pos: np.ndarray
    base_upload_bandwidth_mbps: float
    base_download_bandwidth_mbps: float
    task_ids: list[str]
    sink_task_ids: list[str]
    return_complete_time: float | None = None
    completed: bool = False


class DAGTaskManager:
    """Clean mainline DAG generator and task-state manager for zrj_3."""

    def __init__(self) -> None:
        self._jobs: dict[str, DAGJob] = {}
        self._tasks: dict[str, TaskNode] = {}
        self._tasks_by_ue: dict[int, list[str]] = {}
        self._job_counter: int = 0
        self._task_counter: int = 0
        self._last_arrival_step: int = -1

    @property
    def jobs(self) -> dict[str, DAGJob]:
        return self._jobs

    @property
    def tasks(self) -> dict[str, TaskNode]:
        return self._tasks

    def reset(self) -> None:
        self._jobs.clear()
        self._tasks.clear()
        self._tasks_by_ue.clear()
        self._job_counter = 0
        self._task_counter = 0
        self._last_arrival_step = -1

    def create_dag_for_ue(
        self,
        ue_id: int,
        source_pos: np.ndarray,
        arrival_time: float | None = None,
        current_time_step: float | None = None,
    ) -> DAGJob:
        """Create one clean-mainline DAG for a UE.

        A UE may have at most one active DAG. `source_pos` is copied and remains
        fixed for the DAG lifetime.
        """
        actual_arrival = float(current_time_step if current_time_step is not None else (arrival_time or 0.0))
        if self._ue_has_active_dag(ue_id):
            raise ValueError(f"UE {ue_id} already has an active DAG.")

        self._job_counter += 1
        dag_id = f"dag_{int(actual_arrival)}_{ue_id}_{self._job_counter}"
        source_xy = np.asarray(source_pos, dtype=np.float32).reshape(-1)[:2].copy()

        base_upload = float(
            np.random.choice(config.BASE_UPLOAD_BANDWIDTH_MBPS, p=config.BANDWIDTH_LEVEL_PROBS)
        )
        base_download = float(
            np.random.choice(config.BASE_DOWNLOAD_BANDWIDTH_MBPS, p=config.BANDWIDTH_LEVEL_PROBS)
        )

        level_sizes = self._sample_level_sizes()
        task_ids: list[str] = []
        levels: list[list[str]] = []
        for level_idx, level_size in enumerate(level_sizes):
            level_task_ids: list[str] = []
            for _ in range(level_size):
                self._task_counter += 1
                task_id = f"task_{self._task_counter}"
                task = self._create_task_node(
                    task_id=task_id,
                    dag_id=dag_id,
                    ue_id=ue_id,
                    level=level_idx,
                    source_pos=source_xy,
                    arrival_time=actual_arrival,
                )
                self._tasks[task_id] = task
                self._tasks_by_ue.setdefault(ue_id, []).append(task_id)
                task_ids.append(task_id)
                level_task_ids.append(task_id)
            levels.append(level_task_ids)

        self._connect_levels(levels)
        sink_task_ids = [task_id for task_id in task_ids if not self._tasks[task_id].successors]
        job = DAGJob(
            dag_id=dag_id,
            ue_id=ue_id,
            arrival_time=actual_arrival,
            source_pos=source_xy.copy(),
            base_upload_bandwidth_mbps=base_upload,
            base_download_bandwidth_mbps=base_download,
            task_ids=task_ids,
            sink_task_ids=sink_task_ids,
        )
        self._jobs[dag_id] = job
        self._refresh_ready_states()
        self._mark_critical_path(dag_id)
        return job

    def observe_time_step(self, ues: list[Any], current_time_step: int) -> None:
        """Compatibility arrival hook.

        Phase 1 does not implement hotspot logic. This method uses the clean
        base arrival probability and skips UEs that already have an active DAG.
        It does not mutate existing DAG source positions.
        """
        if current_time_step == self._last_arrival_step:
            self._refresh_ready_states()
            return
        self._last_arrival_step = current_time_step
        for ue in ues:
            ue_id = int(getattr(ue, "id"))
            if self._ue_has_active_dag(ue_id):
                continue
            arrival_prob = float(np.clip(config.DAG_BASE_ARRIVAL_PROB, 0.0, 1.0))
            if np.random.random() >= arrival_prob:
                continue
            pos = np.asarray(getattr(ue, "pos"), dtype=np.float32).reshape(-1)[:2]
            self.create_dag_for_ue(ue_id=ue_id, source_pos=pos, current_time_step=current_time_step)
        self._refresh_ready_states()

    def get_active_tasks(self) -> list[TaskNode]:
        return [
            task
            for task in self._tasks.values()
            if task.state not in {TASK_STATE_FINISHED, TASK_STATE_RETURNED, TASK_STATE_DROPPED}
        ]

    def get_all_non_returned_tasks(self) -> list[TaskNode]:
        """Compatibility view for old callers that still expect finished tasks before return."""
        return [task for task in self._tasks.values() if task.state not in {TASK_STATE_RETURNED, TASK_STATE_DROPPED}]

    def get_ready_tasks(self) -> list[TaskNode]:
        return [task for task in self._tasks.values() if task.state == TASK_STATE_READY]

    def refresh_ready_states(self) -> None:
        self._refresh_ready_states()

    def get_tasks_for_ue(self, ue_id: int) -> list[TaskNode]:
        task_ids = self._tasks_by_ue.get(ue_id, [])
        return [self._tasks[task_id] for task_id in task_ids]

    def get_job(self, dag_id: str) -> DAGJob | None:
        return self._jobs.get(dag_id)

    def get_active_job_for_ue(self, ue_id: int) -> DAGJob | None:
        for job in self._jobs.values():
            if job.ue_id == ue_id and not job.completed:
                return job
        return None

    def get_job_tasks(self, dag_id: str) -> list[TaskNode]:
        job = self._jobs.get(dag_id)
        if job is None:
            return []
        return [self._tasks[task_id] for task_id in job.task_ids if task_id in self._tasks]

    def mark_task_queued(self, task_id: str, uav_id: int, current_time_step: float) -> None:
        task = self._tasks[task_id]
        if task.state != TASK_STATE_READY:
            raise ValueError(f"Task {task_id} is not ready and cannot be queued.")
        task.state = TASK_STATE_QUEUED
        task.assigned_uav = int(uav_id)
        task.enqueue_time = float(current_time_step)

    def mark_task_running(self, task_id: str, current_time_step: float) -> None:
        task = self._tasks[task_id]
        if task.state not in {TASK_STATE_READY, TASK_STATE_QUEUED}:
            raise ValueError(f"Task {task_id} is not queueable and cannot start.")
        task.state = TASK_STATE_RUNNING
        task.start_time = float(current_time_step)

    def mark_task_finished(self, task_id: str, current_time_step: float) -> None:
        task = self._tasks[task_id]
        task.state = TASK_STATE_FINISHED
        task.finish_time = float(current_time_step)
        task.total_energy = task.compute_energy + task.communication_energy + task.return_energy
        self._refresh_ready_states()

    def mark_task_returned(self, task_id: str, current_time_step: float) -> None:
        task = self._tasks[task_id]
        task.state = TASK_STATE_RETURNED
        task.finish_time = float(current_time_step)
        task.total_energy = task.compute_energy + task.communication_energy + task.return_energy
        self.mark_dag_completed_if_ready(task.dag_id, current_time_step)

    def mark_task_dropped(self, task_id: str) -> None:
        # Deprecated compatibility method. Clean mainline does not deadline-drop tasks.
        self._tasks[task_id].state = TASK_STATE_DROPPED

    def mark_dag_completed_if_ready(self, dag_id: str, current_time_step: float | None = None) -> bool:
        job = self._jobs.get(dag_id)
        if job is None or job.completed:
            return False
        job_tasks = self.get_job_tasks(dag_id)
        if not job_tasks:
            return False
        non_sink_finished = all(
            task.state in {TASK_STATE_FINISHED, TASK_STATE_RETURNED}
            for task in job_tasks
            if task.task_id not in set(job.sink_task_ids)
        )
        sink_returned = all(self._tasks[task_id].state == TASK_STATE_RETURNED for task_id in job.sink_task_ids)
        if non_sink_finished and sink_returned:
            job.completed = True
            complete_time = current_time_step
            if complete_time is None:
                finish_times = [task.finish_time for task in job_tasks if task.finish_time is not None]
                complete_time = max(finish_times) if finish_times else job.arrival_time
            job.return_complete_time = float(complete_time)
            return True
        return False

    def get_job_summary(self) -> dict[str, float]:
        total_jobs = len(self._jobs)
        completed_jobs = sum(1 for job in self._jobs.values() if job.completed)
        incomplete_jobs = total_jobs - completed_jobs
        generated_tasks = len(self._tasks)
        finished_tasks = sum(
            1 for task in self._tasks.values() if task.state in {TASK_STATE_FINISHED, TASK_STATE_RETURNED}
        )
        returned_tasks = sum(1 for task in self._tasks.values() if task.state == TASK_STATE_RETURNED)
        critical_tasks = sum(1 for task in self._tasks.values() if task.is_critical_path)
        critical_finished = sum(
            1
            for task in self._tasks.values()
            if task.is_critical_path and task.state in {TASK_STATE_FINISHED, TASK_STATE_RETURNED}
        )
        flowtimes = [
            float(job.return_complete_time - job.arrival_time)
            for job in self._jobs.values()
            if job.completed and job.return_complete_time is not None
        ]
        total_task_energy = sum(float(task.total_energy) for task in self._tasks.values())
        return {
            "dag_total_jobs": float(total_jobs),
            "dag_completed_jobs": float(completed_jobs),
            "dag_successful_jobs": float(completed_jobs),  # Deprecated alias for old logs.
            "dag_incomplete_jobs": float(incomplete_jobs),
            "dag_success_rate": completed_jobs / max(total_jobs, 1),
            "dag_generated_tasks": float(generated_tasks),
            "dag_finished_tasks": float(finished_tasks),
            "dag_returned_tasks": float(returned_tasks),
            "dag_task_finish_rate": finished_tasks / max(generated_tasks, 1),
            "dag_critical_path_tasks": float(critical_tasks),
            "dag_critical_path_finished_tasks": float(critical_finished),
            "dag_critical_path_finish_rate": critical_finished / max(critical_tasks, 1),
            "dag_avg_flowtime": float(np.mean(flowtimes)) if flowtimes else 0.0,
            "dag_total_task_energy": float(total_task_energy),
        }

    def get_dag_completion_ratio(self, dag_id: str) -> float:
        job_tasks = self.get_job_tasks(dag_id)
        if not job_tasks:
            return 0.0
        done = sum(1 for task in job_tasks if task.state in {TASK_STATE_FINISHED, TASK_STATE_RETURNED})
        return float(done / max(len(job_tasks), 1))

    def get_dag_remaining_slack(self, dag_id: str, current_time_step: float) -> float:
        """Deprecated compatibility helper. Clean mainline does not use deadline/slack."""
        return float("inf")

    def get_descendant_count(self, task_id: str) -> int:
        if task_id not in self._tasks:
            return 0
        visited: set[str] = set()
        stack = list(self._tasks[task_id].successors)
        while stack:
            child_id = stack.pop()
            if child_id in visited:
                continue
            visited.add(child_id)
            child = self._tasks.get(child_id)
            if child is not None:
                stack.extend(child.successors)
        return len(visited)

    def is_critical_path_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        return bool(task is not None and task.is_critical_path)

    def is_high_risk_job(self, dag_id: str) -> bool:
        """Deprecated compatibility helper. Deadline risk is not a clean-mainline concept."""
        return False

    def build_task_features(self, current_time_step: float) -> dict[str, np.ndarray]:
        """Compatibility feature builder.

        Phase 1 does not implement clean graph construction. This returns a
        stable numeric vector without deadline/task_type semantics so old imports
        can continue to compile until graph_builder is migrated.
        """
        features: dict[str, np.ndarray] = {}
        max_input = max(float(config.INPUT_DATA_SIZE_MB_RANGE[1]), 1.0)
        max_output = max(float(config.OUTPUT_DATA_SIZE_MB_RANGE[1]), 1.0)
        max_ops = max((task.num_operation for task in self._tasks.values()), default=1.0)
        max_level = float(max(config.DAG_MAX_LEVELS - 1, 1))
        for task in self.get_active_tasks():
            raw = np.array(
                [
                    np.clip(task.input_data_size_mb / max_input, 0.0, 1.0),
                    np.clip(task.output_data_size_mb / max_output, 0.0, 1.0),
                    np.clip(task.num_operation / max_ops, 0.0, 1.0),
                    0.0,
                    np.clip(task.level / max_level, 0.0, 1.0),
                    1.0 if task.is_ready else 0.0,
                    np.clip(task.source_pos[0] / float(config.AREA_WIDTH), 0.0, 1.0),
                    np.clip(task.source_pos[1] / float(config.AREA_HEIGHT), 0.0, 1.0),
                    1.0 if task.is_critical_path else 0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            dim = int(getattr(config, "DAG_TASK_FEATURE_DIM", len(raw)))
            if dim <= len(raw):
                features[task.task_id] = raw[:dim]
            else:
                padded = np.zeros((dim,), dtype=np.float32)
                padded[: len(raw)] = raw
                features[task.task_id] = padded
        return features

    def _create_job_for_ue(self, ue_id: int, source_pos: np.ndarray, current_time_step: int) -> None:
        """Deprecated wrapper for old callers."""
        self.create_dag_for_ue(ue_id=ue_id, source_pos=source_pos, current_time_step=current_time_step)

    def _spawn_new_jobs(self, ues: list[Any], current_time_step: int) -> None:
        """Deprecated wrapper used by old observe paths."""
        for ue in ues:
            ue_id = int(getattr(ue, "id"))
            if self._ue_has_active_dag(ue_id):
                continue
            if np.random.random() < float(np.clip(config.DAG_BASE_ARRIVAL_PROB, 0.0, 1.0)):
                pos = np.asarray(getattr(ue, "pos"), dtype=np.float32).reshape(-1)[:2]
                self.create_dag_for_ue(ue_id=ue_id, source_pos=pos, current_time_step=current_time_step)

    def _get_ue_arrival_prob(self, ue: Any) -> float:
        """Deprecated compatibility helper. Hotspot region logic is not implemented in Phase 1."""
        return float(np.clip(config.DAG_BASE_ARRIVAL_PROB, 0.0, 1.0))

    def _drop_overdue_tasks(self, current_time_step: int) -> None:
        """Deprecated no-op. Clean mainline does not deadline-drop tasks."""
        return

    def _ue_has_active_dag(self, ue_id: int) -> bool:
        for job in self._jobs.values():
            if job.ue_id == ue_id and not job.completed:
                return True
        return False

    def _sample_level_sizes(self) -> list[int]:
        num_tasks = int(np.random.randint(config.DAG_MIN_TASKS, config.DAG_MAX_TASKS + 1))
        max_levels = min(int(config.DAG_MAX_LEVELS), num_tasks)
        level_count = int(np.random.randint(2, max_levels + 1))
        level_sizes = [1] * level_count
        for _ in range(num_tasks - level_count):
            level_sizes[int(np.random.randint(0, level_count))] += 1
        return level_sizes

    def _create_task_node(
        self,
        task_id: str,
        dag_id: str,
        ue_id: int,
        level: int,
        source_pos: np.ndarray,
        arrival_time: float,
    ) -> TaskNode:
        input_mb = float(np.random.uniform(*config.INPUT_DATA_SIZE_MB_RANGE))
        output_mb = float(np.random.uniform(*config.OUTPUT_DATA_SIZE_MB_RANGE))
        task_constant = int(np.random.randint(config.TASK_CONSTANT_RANGE[0], config.TASK_CONSTANT_RANGE[1] + 1))
        complexity_keys = list(config.TASK_COMPLEXITY_PROBS.keys())
        complexity_probs = list(config.TASK_COMPLEXITY_PROBS.values())
        task_complexity = str(np.random.choice(complexity_keys, p=complexity_probs))
        num_operation = self._calculate_num_operation(input_mb, task_complexity, task_constant)
        return TaskNode(
            task_id=task_id,
            dag_id=dag_id,
            ue_id=ue_id,
            input_data_size_mb=input_mb,
            output_data_size_mb=output_mb,
            task_complexity=task_complexity,
            task_constant=task_constant,
            num_operation=num_operation,
            level=level,
            source_pos=np.asarray(source_pos, dtype=np.float32).reshape(-1)[:2].copy(),
            arrival_time=float(arrival_time),
        )

    def _calculate_num_operation(self, input_data_size_mb: float, task_complexity: str, task_constant: int) -> float:
        input_data_size_bytes = float(input_data_size_mb) * 1024.0 * 1024.0
        n = input_data_size_bytes / float(config.BASE_UNIT_BYTES)
        log_n = math.log2(max(n, 2.0))
        if task_complexity == "n":
            complexity_value = n
        elif task_complexity == "nlogn":
            complexity_value = n * log_n
        elif task_complexity == "nlog2n":
            complexity_value = n * (log_n**2)
        else:
            raise ValueError(f"Unknown clean task complexity: {task_complexity}")
        return float(complexity_value * int(task_constant))

    def _connect_levels(self, levels: list[list[str]]) -> None:
        for level_idx in range(1, len(levels)):
            parent_candidates = [task_id for previous in levels[:level_idx] for task_id in previous]
            for task_id in levels[level_idx]:
                max_parent_count = min(len(parent_candidates), int(config.DAG_MAX_PARENTS))
                parent_count = int(np.random.randint(1, max_parent_count + 1))
                chosen_parents = np.random.choice(parent_candidates, size=parent_count, replace=False)
                for parent_id in sorted(chosen_parents.tolist()):
                    self._tasks[task_id].predecessors.append(parent_id)
                    self._tasks[parent_id].successors.append(task_id)

    def _refresh_ready_states(self) -> None:
        for task in self._tasks.values():
            if task.state in {TASK_STATE_FINISHED, TASK_STATE_RETURNED, TASK_STATE_RUNNING, TASK_STATE_QUEUED, TASK_STATE_DROPPED}:
                continue
            if not task.predecessors:
                task.state = TASK_STATE_READY
                if task.ready_time is None:
                    task.ready_time = float(task.arrival_time)
                continue
            if all(self._tasks[parent_id].state in {TASK_STATE_FINISHED, TASK_STATE_RETURNED} for parent_id in task.predecessors):
                task.state = TASK_STATE_READY
                if task.ready_time is None:
                    parent_finish = [
                        self._tasks[parent_id].finish_time
                        for parent_id in task.predecessors
                        if self._tasks[parent_id].finish_time is not None
                    ]
                    task.ready_time = max(parent_finish) if parent_finish else float(task.arrival_time)
            else:
                task.state = TASK_STATE_PENDING

    def _mark_critical_path(self, dag_id: str) -> None:
        job_tasks = self.get_job_tasks(dag_id)
        if not job_tasks:
            return
        task_map = {task.task_id: task for task in job_tasks}
        for task in job_tasks:
            task.is_critical_path = False

        dp: dict[str, float] = {}
        next_on_path: dict[str, str | None] = {}
        for task in sorted(job_tasks, key=lambda item: (-item.level, item.task_id)):
            successors = [child_id for child_id in task.successors if child_id in task_map]
            if not successors:
                dp[task.task_id] = float(task.num_operation)
                next_on_path[task.task_id] = None
                continue
            best_child = max(sorted(successors), key=lambda child_id: (dp[child_id], -_stable_sort_value(child_id)))
            dp[task.task_id] = float(task.num_operation) + dp[best_child]
            next_on_path[task.task_id] = best_child

        start_task_id = max(sorted(dp), key=lambda task_id: (dp[task_id], -_stable_sort_value(task_id)))
        current: str | None = start_task_id
        while current is not None:
            task_map[current].is_critical_path = True
            current = next_on_path[current]


def _stable_sort_value(value: str) -> int:
    """Return a deterministic numeric tie-breaker for ids like task_12."""
    suffix = value.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return sum(ord(char) for char in value)
