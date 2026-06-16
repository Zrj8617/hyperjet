from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
from environment.dag_tasks import DAGTaskManager, TaskNode


@dataclass(slots=True)
class CleanStepReward:
    reward_total: float = 0.0
    time_penalty: float = 0.0
    energy_penalty: float = 0.0
    completed_dag_bonus: float = 0.0
    completed_tasks: int = 0
    completed_dags: int = 0


@dataclass(slots=True)
class CleanEpisodeMetrics:
    episode_reward: float = 0.0
    completed_dag_count: int = 0
    generated_dag_count: int = 0
    completed_task_count: int = 0
    total_task_energy: float = 0.0
    total_compute_energy: float = 0.0
    total_communication_energy: float = 0.0
    total_return_energy: float = 0.0
    invalid_assignment_count: int = 0
    action_count: int = 0
    executed_action_count: int = 0
    task_execution_delays: list[float] = field(default_factory=list)
    dag_flowtimes: list[float] = field(default_factory=list)
    compute_time_by_uav: dict[int, float] = field(default_factory=dict)
    completed_workload_by_uav: dict[int, float] = field(default_factory=dict)
    queue_length_total: float = 0.0
    queue_length_samples: int = 0
    uav_movement_energy_total: float = 0.0
    uav_movement_energy_by_uav: dict[int, float] = field(default_factory=dict)


class CleanMetricsTracker:
    def __init__(self) -> None:
        self.metrics = CleanEpisodeMetrics()
        self._counted_dag_ids: set[str] = set()

    def reset(self, uav_ids: list[int]) -> None:
        self.metrics = CleanEpisodeMetrics(
            compute_time_by_uav={int(uav_id): 0.0 for uav_id in uav_ids},
            completed_workload_by_uav={int(uav_id): 0.0 for uav_id in uav_ids},
            uav_movement_energy_by_uav={int(uav_id): 0.0 for uav_id in uav_ids},
        )
        self._counted_dag_ids = set()

    def calculate_step_reward(
        self,
        task_manager: DAGTaskManager,
        execution_stats: Any,
    ) -> CleanStepReward:
        time_cost = 0.0
        energy_cost = 0.0
        for task_id in getattr(execution_stats, "completed_task_ids", []):
            task = task_manager.get_task(task_id)
            if task is None or task.finish_time is None:
                continue
            delay = self._task_incremental_delay(task, task_manager)
            weight = (
                float(config.CRITICAL_TASK_WEIGHT)
                if task.is_critical_path
                else float(config.NONCRITICAL_TASK_WEIGHT)
            )
            time_cost += weight * delay
            energy_cost += float(task.compute_energy + task.communication_energy + task.return_energy)

        completed_dags = int(getattr(execution_stats, "completed_dags", 0))
        time_penalty = -float(config.REWARD_TIME_WEIGHT) * time_cost
        energy_penalty = -float(config.REWARD_ENERGY_WEIGHT) * energy_cost
        completed_dag_bonus = float(config.REWARD_COMPLETED_DAG_WEIGHT) * completed_dags
        return CleanStepReward(
            reward_total=time_penalty + energy_penalty + completed_dag_bonus,
            time_penalty=time_penalty,
            energy_penalty=energy_penalty,
            completed_dag_bonus=completed_dag_bonus,
            completed_tasks=int(getattr(execution_stats, "completed_tasks", 0)),
            completed_dags=completed_dags,
        )

    def update(
        self,
        task_manager: DAGTaskManager,
        execution_stats: Any,
        step_reward: CleanStepReward,
        created_dags: int,
        action_count: int,
        queue_lengths: list[int],
        elapsed_steps: int,
    ) -> None:
        self.metrics.episode_reward += float(step_reward.reward_total)
        self.metrics.generated_dag_count += int(created_dags)
        self.metrics.completed_task_count += int(getattr(execution_stats, "completed_tasks", 0))
        self.metrics.completed_dag_count += int(getattr(execution_stats, "completed_dags", 0))
        self.metrics.total_task_energy += float(getattr(execution_stats, "step_task_energy", 0.0))
        self.metrics.total_compute_energy += float(getattr(execution_stats, "step_compute_energy", 0.0))
        self.metrics.total_communication_energy += float(getattr(execution_stats, "step_communication_energy", 0.0))
        self.metrics.total_return_energy += float(getattr(execution_stats, "step_return_energy", 0.0))
        self.metrics.invalid_assignment_count += int(getattr(execution_stats, "invalid_assignments", 0))
        self.metrics.action_count += int(action_count)
        self.metrics.executed_action_count += int(getattr(execution_stats, "newly_assigned_tasks", 0))

        for task_id in getattr(execution_stats, "completed_task_ids", []):
            task = task_manager.get_task(task_id)
            if task is not None:
                self.metrics.task_execution_delays.append(self._task_execution_delay(task))

        for dag_id in getattr(execution_stats, "completed_dag_ids", []):
            job = task_manager.get_job(dag_id)
            if job is None or job.return_complete_time is None or dag_id in self._counted_dag_ids:
                continue
            self.metrics.dag_flowtimes.append(float(job.return_complete_time - job.arrival_time))
            self._counted_dag_ids.add(dag_id)

        for uav_id, compute_time in getattr(execution_stats, "compute_time_by_uav", {}).items():
            key = int(uav_id)
            self.metrics.compute_time_by_uav[key] = self.metrics.compute_time_by_uav.get(key, 0.0) + float(compute_time)
        for uav_id, workload in getattr(execution_stats, "completed_workload_by_uav", {}).items():
            key = int(uav_id)
            self.metrics.completed_workload_by_uav[key] = (
                self.metrics.completed_workload_by_uav.get(key, 0.0) + float(workload)
            )

        self.metrics.queue_length_total += float(sum(queue_lengths))
        self.metrics.queue_length_samples += len(queue_lengths)

    def to_info(self, elapsed_steps: int) -> dict[str, float]:
        steps = max(int(elapsed_steps), 1)
        num_uavs = max(int(config.NUM_UAVS), 1)
        compute_total = sum(self.metrics.compute_time_by_uav.values())
        workload_values = np.asarray(list(self.metrics.completed_workload_by_uav.values()), dtype=np.float64)
        workload_mean = float(workload_values.mean()) if workload_values.size else 0.0
        workload_std = float(workload_values.std()) if workload_values.size else 0.0
        load_balance = workload_std / workload_mean if workload_mean > 0.0 else 0.0
        action_total = max(self.metrics.action_count, 1)
        return {
            "avg_dag_flowtime": float(np.mean(self.metrics.dag_flowtimes)) if self.metrics.dag_flowtimes else 0.0,
            "completed_dag_count": float(self.metrics.completed_dag_count),
            "generated_dag_count": float(self.metrics.generated_dag_count),
            "dag_throughput": float(self.metrics.completed_dag_count / steps),
            "avg_task_execution_delay": (
                float(np.mean(self.metrics.task_execution_delays)) if self.metrics.task_execution_delays else 0.0
            ),
            "total_task_energy": float(self.metrics.total_task_energy),
            "total_compute_energy": float(self.metrics.total_compute_energy),
            "total_communication_energy": float(self.metrics.total_communication_energy),
            "total_return_energy": float(self.metrics.total_return_energy),
            "uav_computation_utilization": float(
                compute_total / (num_uavs * steps * float(config.TIME_SLOT_DURATION))
            ),
            "avg_uav_queue_length": float(
                self.metrics.queue_length_total / max(self.metrics.queue_length_samples, 1)
            ),
            "load_balance": float(load_balance),
            "episode_reward": float(self.metrics.episode_reward),
            "action_executed_rate": float(self.metrics.executed_action_count / action_total),
            "invalid_assignment_rate": float(self.metrics.invalid_assignment_count / action_total),
            "uav_movement_energy_total": float(self.metrics.uav_movement_energy_total),
            "uav_movement_energy_by_uav": dict(self.metrics.uav_movement_energy_by_uav),
            "uav_movement_energy_ratio": 0.0,
        }

    def _task_incremental_delay(self, task: TaskNode, task_manager: DAGTaskManager) -> float:
        if task.finish_time is None:
            return 0.0
        if not task.predecessors:
            job = task_manager.get_job(task.dag_id)
            start_reference = task.arrival_time if job is None else job.arrival_time
            return max(0.0, float(task.finish_time - start_reference))
        parent_finish_times = [
            float(parent.finish_time)
            for parent_id in task.predecessors
            if (parent := task_manager.get_task(parent_id)) is not None and parent.finish_time is not None
        ]
        parent_reference = max(parent_finish_times) if parent_finish_times else float(task.arrival_time)
        return max(0.0, float(task.finish_time - parent_reference))

    def _task_execution_delay(self, task: TaskNode) -> float:
        if task.finish_time is None:
            return 0.0
        ready_time = task.ready_time if task.ready_time is not None else task.arrival_time
        return max(0.0, float(task.finish_time - ready_time))
