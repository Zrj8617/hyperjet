from __future__ import annotations

from dataclasses import dataclass, field
import config
import numpy as np


TASK_STATE_WAITING = "waiting"
TASK_STATE_READY = "ready"
TASK_STATE_QUEUED = "queued"
TASK_STATE_RUNNING = "running"
TASK_STATE_FINISHED = "finished"
TASK_STATE_DROPPED = "dropped"


@dataclass(slots=True)
class TaskNode:
    task_id: str
    dag_id: str
    ue_id: int
    arrival_time: int
    input_size: int
    output_size: int
    cpu_cycles: float
    deadline: int
    level: int
    task_type: int
    source_pos: np.ndarray
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    state: str = TASK_STATE_WAITING
    assigned_uav: int | None = None
    enqueue_time: float | None = None
    start_time: float | None = None
    finish_time: float | None = None

    @property
    def is_ready(self) -> bool:
        return self.state == TASK_STATE_READY

    @property
    def is_terminal(self) -> bool:
        return self.state in {TASK_STATE_FINISHED, TASK_STATE_DROPPED}

    def remaining_slack(self, current_time_step: float) -> float:
        return float(self.deadline - current_time_step)


@dataclass(slots=True)
class DAGJob:
    dag_id: str
    ue_id: int
    arrival_time: int
    task_ids: list[str]


class DAGTaskManager:
    """Maintains active DAG jobs and task state transitions for phase one."""

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

    def observe_time_step(self, ues: list, current_time_step: int) -> None:
        """Updates source positions and spawns fresh DAG jobs once per environment step."""
        for task in self._tasks.values():
            task.source_pos = np.array(ues[task.ue_id].pos[:2], dtype=np.float32)

        if current_time_step == self._last_arrival_step:
            self._refresh_ready_states()
            self._drop_overdue_tasks(current_time_step)
            return

        self._last_arrival_step = current_time_step
        self._spawn_new_jobs(ues, current_time_step)
        self._refresh_ready_states()
        self._drop_overdue_tasks(current_time_step)

    def get_ready_tasks(self) -> list[TaskNode]:
        return [task for task in self._tasks.values() if task.state == TASK_STATE_READY]

    def get_active_tasks(self) -> list[TaskNode]:
        return [task for task in self._tasks.values() if not task.is_terminal]

    def get_tasks_for_ue(self, ue_id: int) -> list[TaskNode]:
        task_ids = self._tasks_by_ue.get(ue_id, [])
        return [self._tasks[task_id] for task_id in task_ids]

    def get_job_summary(self) -> dict[str, float]:
        total_jobs = len(self._jobs)
        successful_jobs = 0
        failed_jobs = 0
        incomplete_jobs = 0
        on_time_successful_jobs = 0

        for job in self._jobs.values():
            job_tasks = [self._tasks[task_id] for task_id in job.task_ids]
            if all(task.state == TASK_STATE_FINISHED for task in job_tasks):
                successful_jobs += 1
                if all(task.finish_time is not None and task.finish_time <= task.deadline for task in job_tasks):
                    on_time_successful_jobs += 1
            elif any(task.state == TASK_STATE_DROPPED for task in job_tasks):
                failed_jobs += 1
            else:
                incomplete_jobs += 1

        return {
            "dag_total_jobs": float(total_jobs),
            "dag_successful_jobs": float(successful_jobs),
            "dag_failed_jobs": float(failed_jobs),
            "dag_incomplete_jobs": float(incomplete_jobs),
            "dag_on_time_successful_jobs": float(on_time_successful_jobs),
            "dag_success_rate": successful_jobs / max(total_jobs, 1),
            "dag_failure_rate": failed_jobs / max(total_jobs, 1),
            "dag_incomplete_rate": incomplete_jobs / max(total_jobs, 1),
            "dag_on_time_success_rate": on_time_successful_jobs / max(total_jobs, 1),
        }

    def mark_task_queued(self, task_id: str, uav_id: int, current_time_step: float) -> None:
        task = self._tasks[task_id]
        if task.state != TASK_STATE_READY:
            raise ValueError(f"Task {task_id} is not ready and cannot be queued.")
        task.state = TASK_STATE_QUEUED
        task.assigned_uav = uav_id
        task.enqueue_time = current_time_step

    def mark_task_running(self, task_id: str, current_time_step: float) -> None:
        task = self._tasks[task_id]
        if task.state not in {TASK_STATE_READY, TASK_STATE_QUEUED}:
            raise ValueError(f"Task {task_id} is not queueable and cannot start.")
        task.state = TASK_STATE_RUNNING
        task.start_time = current_time_step

    def mark_task_finished(self, task_id: str, current_time_step: float) -> None:
        task = self._tasks[task_id]
        task.state = TASK_STATE_FINISHED
        task.finish_time = current_time_step
        self._refresh_ready_states()

    def mark_task_dropped(self, task_id: str) -> None:
        self._tasks[task_id].state = TASK_STATE_DROPPED

    def build_task_features(self, current_time_step: float) -> dict[str, np.ndarray]:
        features: dict[str, np.ndarray] = {}
        for task in self.get_active_tasks():
            features[task.task_id] = np.array(
                [
                    task.input_size / float(config.DAG_MAX_INPUT_SIZE),
                    task.output_size / float(config.DAG_MAX_OUTPUT_SIZE),
                    task.cpu_cycles / float(config.DAG_MAX_CPU_CYCLES),
                    max(task.remaining_slack(current_time_step), 0.0) / float(config.DAG_MAX_DEADLINE_OFFSET),
                    task.level / float(max(config.DAG_MAX_TASK_LEVELS - 1, 1)),
                    1.0 if task.is_ready else 0.0,
                    task.source_pos[0] / float(config.AREA_WIDTH),
                    task.source_pos[1] / float(config.AREA_HEIGHT),
                    1.0 if task.task_type == config.TASK_TYPE_PREPROCESS else 0.0,
                    1.0 if task.task_type == config.TASK_TYPE_COMPUTE else 0.0,
                    1.0 if task.task_type == config.TASK_TYPE_AGGREGATION else 0.0,
                ],
                dtype=np.float32,
            )
        return features

    def _spawn_new_jobs(self, ues: list, current_time_step: int) -> None:
        for ue in ues:
            arrival_prob = self._get_ue_arrival_prob(ue)
            if np.random.random() >= arrival_prob:
                continue
            self._create_job_for_ue(ue.id, np.array(ue.pos[:2], dtype=np.float32), current_time_step)

    def _get_ue_arrival_prob(self, ue) -> float:
        arrival_prob = config.DAG_ARRIVAL_PROB
        if getattr(ue, "is_hotspot", False):
            arrival_prob *= config.HOTSPOT_DAG_ARRIVAL_MULTIPLIER
        return float(np.clip(arrival_prob, 0.0, 1.0))

    def _create_job_for_ue(self, ue_id: int, source_pos: np.ndarray, current_time_step: int) -> None:
        self._job_counter += 1
        dag_id = f"dag_{current_time_step}_{ue_id}_{self._job_counter}"

        num_tasks = int(np.random.randint(config.DAG_MIN_TASKS, config.DAG_MAX_TASKS + 1))
        level_count = int(np.random.randint(2, min(config.DAG_MAX_TASK_LEVELS, num_tasks) + 1))
        level_sizes = [1] * level_count
        for _ in range(num_tasks - level_count):
            level_sizes[np.random.randint(0, level_count)] += 1

        task_ids: list[str] = []
        levels: list[list[str]] = []
        for level_idx, level_size in enumerate(level_sizes):
            level_task_ids: list[str] = []
            for _ in range(level_size):
                self._task_counter += 1
                task_id = f"task_{self._task_counter}"
                task_type = self._task_type_for_level(level_idx, level_count)
                input_size, output_size, cpu_cycles, deadline_offset = self._sample_task_attributes(task_type)
                deadline = current_time_step + deadline_offset
                task = TaskNode(
                    task_id=task_id,
                    dag_id=dag_id,
                    ue_id=ue_id,
                    arrival_time=current_time_step,
                    input_size=input_size,
                    output_size=output_size,
                    cpu_cycles=cpu_cycles,
                    deadline=deadline,
                    level=level_idx,
                    task_type=task_type,
                    source_pos=source_pos.copy(),
                )
                self._tasks[task_id] = task
                self._tasks_by_ue.setdefault(ue_id, []).append(task_id)
                task_ids.append(task_id)
                level_task_ids.append(task_id)
            levels.append(level_task_ids)

        for level_idx in range(1, len(levels)):
            for task_id in levels[level_idx]:
                parent_candidates = levels[level_idx - 1]
                parent_count = int(np.random.randint(1, min(len(parent_candidates), config.DAG_MAX_PARENTS) + 1))
                chosen_parents = np.random.choice(parent_candidates, size=parent_count, replace=False)
                for parent_id in chosen_parents.tolist():
                    self._tasks[task_id].predecessors.append(parent_id)
                    self._tasks[parent_id].successors.append(task_id)

        self._jobs[dag_id] = DAGJob(dag_id=dag_id, ue_id=ue_id, arrival_time=current_time_step, task_ids=task_ids)

    def _task_type_for_level(self, level_idx: int, level_count: int) -> int:
        if level_idx == 0:
            return config.TASK_TYPE_PREPROCESS
        if level_idx == level_count - 1:
            return config.TASK_TYPE_AGGREGATION
        return config.TASK_TYPE_COMPUTE

    def _sample_task_attributes(self, task_type: int) -> tuple[int, int, float, int]:
        if task_type == config.TASK_TYPE_PREPROCESS:
            input_range = config.DAG_TYPE0_INPUT_RANGE
            output_range = config.DAG_TYPE0_OUTPUT_RANGE
            cpu_range = config.DAG_TYPE0_CPU_RANGE
            deadline_range = config.DAG_TYPE0_DEADLINE_RANGE
        elif task_type == config.TASK_TYPE_COMPUTE:
            input_range = config.DAG_TYPE1_INPUT_RANGE
            output_range = config.DAG_TYPE1_OUTPUT_RANGE
            cpu_range = config.DAG_TYPE1_CPU_RANGE
            deadline_range = config.DAG_TYPE1_DEADLINE_RANGE
        elif task_type == config.TASK_TYPE_AGGREGATION:
            input_range = config.DAG_TYPE2_INPUT_RANGE
            output_range = config.DAG_TYPE2_OUTPUT_RANGE
            cpu_range = config.DAG_TYPE2_CPU_RANGE
            deadline_range = config.DAG_TYPE2_DEADLINE_RANGE
        else:
            raise ValueError(f"Unknown DAG task type: {task_type}")

        input_size = int(np.random.randint(input_range[0], input_range[1] + 1))
        output_size = int(np.random.randint(output_range[0], output_range[1] + 1))
        cpu_cycles = float(np.random.randint(cpu_range[0], cpu_range[1] + 1))
        deadline_offset = int(np.random.randint(deadline_range[0], deadline_range[1] + 1))
        return input_size, output_size, cpu_cycles, deadline_offset

    def _refresh_ready_states(self) -> None:
        for task in self._tasks.values():
            if task.state in {TASK_STATE_FINISHED, TASK_STATE_RUNNING, TASK_STATE_QUEUED, TASK_STATE_DROPPED}:
                continue
            if not task.predecessors:
                task.state = TASK_STATE_READY
                continue
            if all(self._tasks[parent_id].state == TASK_STATE_FINISHED for parent_id in task.predecessors):
                task.state = TASK_STATE_READY
            else:
                task.state = TASK_STATE_WAITING

    def _drop_overdue_tasks(self, current_time_step: int) -> None:
        for task in self._tasks.values():
            if task.is_terminal:
                continue
            if current_time_step > task.deadline:
                task.state = TASK_STATE_DROPPED
