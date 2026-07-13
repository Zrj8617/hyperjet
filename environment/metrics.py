from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

import config
from environment.dag_tasks import DAGTaskManager, TaskNode


@dataclass(slots=True)
class CleanStepReward:
    """保存一个时隙的总奖励及各个奖励分量。"""
    reward_total: float = 0.0
    time_penalty: float = 0.0
    energy_penalty: float = 0.0
    task_energy_penalty: float = 0.0
    movement_energy_penalty: float = 0.0
    completed_dag_bonus: float = 0.0
    # Optional movement position shaping (0.0 unless ENABLE_MOVEMENT_POSITION_SHAPING).
    movement_position_bonus: float = 0.0
    completed_tasks: int = 0
    completed_dags: int = 0


@dataclass(slots=True)
class CleanEpisodeMetrics:
    """累计一个回合内的完成率、时延、能耗、队列和移动统计。"""
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
    executed_slots: int = 0
    task_execution_delays: list[float] = field(default_factory=list)
    critical_path_task_completion_delays: list[float] = field(default_factory=list)
    dag_flowtimes: list[float] = field(default_factory=list)
    compute_time_by_uav: dict[int, float] = field(default_factory=dict)
    completed_workload_by_uav: dict[int, float] = field(default_factory=dict)
    queue_length_total: float = 0.0
    queue_length_samples: int = 0
    uav_movement_energy_total: float = 0.0
    uav_movement_energy_by_uav: dict[int, float] = field(default_factory=dict)
    movement_hover_action_count: int = 0
    movement_action_total: int = 0
    movement_displacement_total: float = 0.0


class CleanMetricsTracker:
    def __init__(self, completed_dag_weight: float | None = None) -> None:
        """创建空的回合指标，并准备 DAG 去重集合。"""
        resolved_weight = (
            float(config.REWARD_COMPLETED_DAG_WEIGHT)
            if completed_dag_weight is None
            else float(completed_dag_weight)
        )
        if not math.isfinite(resolved_weight) or resolved_weight < 0.0:
            raise ValueError("completed_dag_weight must be finite and non-negative")
        self.completed_dag_weight = resolved_weight
        self.metrics = CleanEpisodeMetrics()
        self._counted_dag_ids: set[str] = set()

    def reset(self, uav_ids: list[int]) -> None:
        """开始新回合时清空统计，并为每架 UAV 建立独立累计项。"""
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
        movement_energy_slot: float = 0.0,
        movement_position_signal: float = 0.0,
    ) -> CleanStepReward:
        """根据本时隙刚完成的任务和 DAG 计算共享奖励。

        每个任务和 DAG 只结算一次；奖励由时延惩罚、任务/移动能耗惩罚、
        DAG 完成奖励以及可选的位置塑形奖励组成。
        """
        time_cost = 0.0
        task_energy_cost = 0.0
        reward_completed_task_count = 0
        # 只对本时隙首次进入奖励完成状态的任务结算，避免重复扣时延和能耗。
        for task_id in getattr(execution_stats, "reward_completed_task_ids", getattr(execution_stats, "completed_task_ids", [])):
            task = task_manager.get_task(task_id)
            if task is None or task.reward_settled or task.reward_completion_time is None:
                continue
            delay = self._task_incremental_delay(task, task_manager)
            weight = (
                float(config.CRITICAL_TASK_WEIGHT)
                if task.is_critical_path
                else float(config.NONCRITICAL_TASK_WEIGHT)
            )
            time_cost += weight * self._norm_time(delay)
            task_energy = float(task.compute_energy + task.communication_energy + task.return_energy)
            task_energy_cost += self._norm_task_energy(task_energy)
            task.reward_settled = True
            reward_completed_task_count += 1

        # DAG 奖励要等出口结果真正返回 UE 后才结算。
        completed_dags = 0
        for dag_id in getattr(execution_stats, "reward_completed_dag_ids", getattr(execution_stats, "completed_dag_ids", [])):
            job = task_manager.get_job(dag_id)
            if job is None or job.completion_reward_settled:
                continue
            if not job.completed or job.return_complete_time is None:
                continue
            job.completion_reward_settled = True
            completed_dags += 1
        time_penalty = -float(config.REWARD_TIME_WEIGHT) * time_cost
        task_energy_penalty = -float(config.REWARD_ENERGY_WEIGHT) * task_energy_cost
        movement_energy_penalty = -float(config.REWARD_MOVEMENT_ENERGY_WEIGHT) * self._norm_move_energy(
            movement_energy_slot
        )
        energy_penalty = task_energy_penalty + movement_energy_penalty
        completed_dag_bonus = self.completed_dag_weight * completed_dags
        # 位置塑形默认关闭；开关关闭或权重为 0 时严格返回 0，不改变基线奖励。
        movement_position_bonus = 0.0
        if bool(getattr(config, "ENABLE_MOVEMENT_POSITION_SHAPING", False)):
            movement_position_bonus = float(
                getattr(config, "REWARD_MOVEMENT_POSITION_WEIGHT", 0.0)
            ) * float(movement_position_signal)
        return CleanStepReward(
            reward_total=(
                time_penalty
                + task_energy_penalty
                + movement_energy_penalty
                + completed_dag_bonus
                + movement_position_bonus
            ),
            time_penalty=time_penalty,
            energy_penalty=energy_penalty,
            task_energy_penalty=task_energy_penalty,
            movement_energy_penalty=movement_energy_penalty,
            completed_dag_bonus=completed_dag_bonus,
            movement_position_bonus=movement_position_bonus,
            completed_tasks=reward_completed_task_count,
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
        movement_energy_by_uav: dict[int, float] | None = None,
        movement_hover_count: int = 0,
        movement_action_count: int = 0,
        movement_displacement_total: float = 0.0,
    ) -> None:
        """把一个时隙的执行结果累加到当前回合指标中。

        除计数和能耗外，还会记录任务时延、DAG 流时间、UAV 工作量、队列长度和移动统计。
        """
        self.metrics.episode_reward += float(step_reward.reward_total)
        self.metrics.generated_dag_count = max(
            int(self.metrics.generated_dag_count + int(created_dags)),
            len(task_manager.jobs),
        )
        self.metrics.completed_task_count += int(getattr(execution_stats, "completed_tasks", 0))
        self.metrics.completed_dag_count += int(getattr(execution_stats, "completed_dags", 0))
        self.metrics.total_task_energy += float(getattr(execution_stats, "step_task_energy", 0.0))
        self.metrics.total_compute_energy += float(getattr(execution_stats, "step_compute_energy", 0.0))
        self.metrics.total_communication_energy += float(getattr(execution_stats, "step_communication_energy", 0.0))
        self.metrics.total_return_energy += float(getattr(execution_stats, "step_return_energy", 0.0))
        self.metrics.invalid_assignment_count += int(getattr(execution_stats, "invalid_assignments", 0))
        self.metrics.action_count += int(action_count)
        self.metrics.executed_action_count += int(getattr(execution_stats, "newly_assigned_tasks", 0))
        self.metrics.executed_slots = max(self.metrics.executed_slots, int(elapsed_steps))

        # 保存任务级样本，回合结束时再计算平均值。
        for task_id in getattr(execution_stats, "reward_completed_task_ids", getattr(execution_stats, "completed_task_ids", [])):
            task = task_manager.get_task(task_id)
            if task is not None:
                self.metrics.task_execution_delays.append(self._task_execution_delay(task))
                if task.is_critical_path:
                    self.metrics.critical_path_task_completion_delays.append(
                        self._task_incremental_delay(task, task_manager)
                    )

        # 同一个 DAG 只记录一次从到达到最终回传完成的流时间。
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
        if movement_energy_by_uav is not None:
            for uav_id, energy in movement_energy_by_uav.items():
                key = int(uav_id)
                value = float(energy)
                self.metrics.uav_movement_energy_by_uav[key] = (
                    self.metrics.uav_movement_energy_by_uav.get(key, 0.0) + value
                )
                self.metrics.uav_movement_energy_total += value
        self.metrics.movement_hover_action_count += int(movement_hover_count)
        self.metrics.movement_action_total += int(movement_action_count)
        self.metrics.movement_displacement_total += float(movement_displacement_total)

    def to_info(self, elapsed_steps: int, total_time_seconds: float) -> dict[str, float]:
        """把当前累计值整理成训练和评估日志使用的指标字典。"""
        steps = max(int(elapsed_steps), 1)
        # 时隙到秒的换算由 Env 统一完成，这里直接使用传入的物理时间，避免重复乘时隙长度。
        total_evaluation_time = max(float(total_time_seconds), float(config.TIME_SLOT_DURATION))
        num_uavs = max(int(config.NUM_UAVS), 1)
        compute_total = sum(self.metrics.compute_time_by_uav.values())
        workload_values = np.asarray(list(self.metrics.completed_workload_by_uav.values()), dtype=np.float64)
        workload_mean = float(workload_values.mean()) if workload_values.size else 0.0
        workload_std = float(workload_values.std()) if workload_values.size else 0.0
        load_balance = workload_std / workload_mean if workload_mean > 0.0 else 0.0
        action_total = max(self.metrics.action_count, 1)
        generated = max(int(self.metrics.generated_dag_count), len(self._counted_dag_ids))
        completed = int(self.metrics.completed_dag_count)
        avg_flowtime = float(np.mean(self.metrics.dag_flowtimes)) if self.metrics.dag_flowtimes else 0.0
        completion_rate = float(completed / max(generated, 1))
        throughput = float(completed / total_evaluation_time)
        # 回合总能耗包含任务执行和 UAV 移动两部分，单 DAG 能耗用这个总量除以完成数。
        total_episode_energy = float(self.metrics.total_task_energy) + float(self.metrics.uav_movement_energy_total)
        energy_per_completed_dag = float(total_episode_energy / max(completed, 1))
        avg_critical_delay = (
            float(np.mean(self.metrics.critical_path_task_completion_delays))
            if self.metrics.critical_path_task_completion_delays
            else 0.0
        )
        return {
            "avg_dag_flowtime": avg_flowtime,
            "average_dag_flowtime": avg_flowtime,
            "completed_dag_count": float(completed),
            "generated_dag_count": float(generated),
            "dag_completion_rate": completion_rate,
            "dag_throughput": throughput,
            "average_critical_path_task_completion_delay": avg_critical_delay,
            "energy_per_completed_dag": energy_per_completed_dag,
            "avg_task_execution_delay": (
                float(np.mean(self.metrics.task_execution_delays)) if self.metrics.task_execution_delays else 0.0
            ),
            "total_task_energy": float(self.metrics.total_task_energy),
            "total_episode_energy": float(total_episode_energy),
            "total_compute_energy": float(self.metrics.total_compute_energy),
            "total_communication_energy": float(self.metrics.total_communication_energy),
            "total_return_energy": float(self.metrics.total_return_energy),
            "uav_computation_utilization": float(
                compute_total / (num_uavs * total_evaluation_time)
            ),
            "avg_uav_queue_length": float(
                self.metrics.queue_length_total / max(self.metrics.queue_length_samples, 1)
            ),
            "load_balance": float(load_balance),
            "episode_reward": float(self.metrics.episode_reward),
            "action_executed_rate": float(self.metrics.executed_action_count / action_total),
            "invalid_assignment_count": float(self.metrics.invalid_assignment_count),
            "invalid_assignment_rate": float(self.metrics.invalid_assignment_count / action_total),
            "uav_movement_energy_total": float(self.metrics.uav_movement_energy_total),
            "uav_movement_energy_by_uav": dict(self.metrics.uav_movement_energy_by_uav),
            "uav_movement_energy_ratio": float(
                self.metrics.uav_movement_energy_total
                / max(float(config.CLEAN_REWARD_MOVE_ENERGY_REF) * steps, 1.0)
            ),
            "movement_action_total": float(self.metrics.movement_action_total),
            "hover_action_ratio": (
                float(self.metrics.movement_hover_action_count) / float(self.metrics.movement_action_total)
                if self.metrics.movement_action_total > 0
                else None
            ),
            "mean_uav_displacement_per_slot": float(
                self.metrics.movement_displacement_total / max(num_uavs * steps, 1)
            ),
        }

    def _task_incremental_delay(self, task: TaskNode, task_manager: DAGTaskManager) -> float:
        """计算任务相对 DAG 到达或最晚前驱完成时刻新增的等待与执行时延。"""
        if task.reward_completion_time is None:
            return 0.0
        if not task.predecessors:
            job = task_manager.get_job(task.dag_id)
            start_reference = task.arrival_time if job is None else job.arrival_time
            return max(0.0, float(task.reward_completion_time - start_reference))
        parent_finish_times = [
            float(parent.compute_finish_time)
            for parent_id in task.predecessors
            if (parent := task_manager.get_task(parent_id)) is not None and parent.compute_finish_time is not None
        ]
        parent_reference = max(parent_finish_times) if parent_finish_times else float(task.arrival_time)
        return max(0.0, float(task.reward_completion_time - parent_reference))

    def _task_execution_delay(self, task: TaskNode) -> float:
        """计算任务从就绪到奖励完成之间的实际执行时延。"""
        if task.reward_completion_time is None:
            return 0.0
        ready_time = task.ready_time if task.ready_time is not None else task.arrival_time
        return max(0.0, float(task.reward_completion_time - ready_time))

    def _norm_time(self, value: float) -> float:
        """按参考时间归一化奖励时延，并截断过大的长尾值。"""
        # 截断只影响奖励计算，日志中的原始任务时延和 DAG 流时间保持不变。
        normalized = max(float(value), 0.0) / max(float(config.CLEAN_REWARD_TIME_REF), 1.0)
        return min(normalized, float(getattr(config, "CLEAN_REWARD_TIME_CLIP", float("inf"))))

    def _norm_task_energy(self, value: float) -> float:
        """把任务执行能耗换算为无量纲奖励成本。"""
        return max(float(value), 0.0) / max(float(config.CLEAN_REWARD_TASK_ENERGY_REF), 1.0)

    def _norm_move_energy(self, value: float) -> float:
        """把本时隙 UAV 移动能耗换算为无量纲奖励成本。"""
        return max(float(value), 0.0) / max(float(config.CLEAN_REWARD_MOVE_ENERGY_REF), 1.0)
