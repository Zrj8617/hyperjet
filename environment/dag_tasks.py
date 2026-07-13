from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any

import config
import numpy as np


TASK_STATE_WAITING_DEPENDENCY = "WAITING_DEPENDENCY"
TASK_STATE_READY_UNSCHEDULED = "READY_UNSCHEDULED"
TASK_STATE_IN_SERVICE = "IN_SERVICE"
TASK_STATE_RETURNING = "RETURNING"
TASK_STATE_COMPLETED = "COMPLETED"
TASK_STATE_DROPPED = "DROPPED"  # Deprecated compatibility state; clean mainline does not deadline-drop tasks.

SERVICE_PHASE_UPLOADING_OR_TRANSFERRING = "UPLOADING_OR_TRANSFERRING"
SERVICE_PHASE_QUEUED = "QUEUED"
SERVICE_PHASE_COMPUTING = "COMPUTING"

# Deprecated aliases kept for old imports. Clean task states still use the five
# canonical clean lifecycle values above.
TASK_STATE_PENDING = TASK_STATE_WAITING_DEPENDENCY
TASK_STATE_READY = TASK_STATE_READY_UNSCHEDULED
TASK_STATE_QUEUED = TASK_STATE_IN_SERVICE
TASK_STATE_RUNNING = TASK_STATE_IN_SERVICE
TASK_STATE_FINISHED = TASK_STATE_COMPLETED
TASK_STATE_RETURNED = TASK_STATE_COMPLETED
TASK_STATE_WAITING = TASK_STATE_WAITING_DEPENDENCY


@dataclass(slots=True)
class TaskNode:
    """保存一个 DAG 子任务的依赖关系、运行状态、时间戳和能耗。"""
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
    # Static per-DAG topological rank assigned at DAG creation (level-major creation
    # order is a valid topological order). Used as the spec ready-sort key component.
    topological_index: int = 0
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    ready_time: float | None = None
    start_time: float | None = None
    finish_time: float | None = None
    compute_finish_time: float | None = None
    reward_completion_time: float | None = None
    assigned_uav: int | None = None
    state: str = TASK_STATE_WAITING_DEPENDENCY
    service_phase: str | None = None
    is_critical_path: bool = False
    compute_energy: float = 0.0
    communication_energy: float = 0.0
    return_energy: float = 0.0
    total_energy: float = 0.0
    enqueue_time: float | None = None
    reward_settled: bool = False
    # Deprecated compatibility fields. They must not participate in clean reward,
    # graph construction, candidate filtering, critical path, or main metrics.
    deadline: float | None = None
    task_type: int | None = None

    @property
    def input_size(self) -> int:
        """Deprecated byte-size alias for old modules.

        中文：旧模块使用的字节单位输入大小；clean 主线直接使用 MB 字段。
        """
        return int(round(self.input_data_size_mb * 1024 * 1024))

    @property
    def output_size(self) -> int:
        """Deprecated byte-size alias for old modules.

        中文：旧模块使用的字节单位输出大小；clean 主线直接使用 MB 字段。
        """
        return int(round(self.output_data_size_mb * 1024 * 1024))

    @property
    def cpu_cycles(self) -> float:
        """Deprecated compute alias; clean mainline uses num_operation.

        中文：旧代码的计算量别名，数值与 `num_operation` 相同。
        """
        return float(self.num_operation)

    @property
    def is_ready(self) -> bool:
        """判断任务是否已经满足依赖且尚未分配。"""
        return self.state == TASK_STATE_READY

    @property
    def is_terminal(self) -> bool:
        """判断任务是否已经完成，或进入旧版丢弃终态。"""
        return self.state in {TASK_STATE_COMPLETED, TASK_STATE_DROPPED}

    @property
    def is_computation_finished(self) -> bool:
        """判断任务计算是否结束，包括正在回传结果的出口任务。"""
        return self.state in {TASK_STATE_RETURNING, TASK_STATE_COMPLETED}

    @property
    def is_fully_completed(self) -> bool:
        """判断任务是否连同必要的结果回传一起全部完成。"""
        return self.state == TASK_STATE_COMPLETED

    def remaining_slack(self, current_time_step: float) -> float:
        """Deprecated compatibility helper. Clean mainline does not use slack.

        中文：旧截止期逻辑的剩余宽裕时间；没有截止期时返回无穷大。
        """
        if self.deadline is None:
            return float("inf")
        return float(self.deadline - current_time_step)


@dataclass(slots=True)
class DAGJob:
    """保存一个 UE 产生的完整 DAG 及其入口位置、带宽和完成状态。"""
    dag_id: str
    ue_id: int
    arrival_time: float
    source_pos: np.ndarray
    base_upload_bandwidth_mbps: float
    base_download_bandwidth_mbps: float
    task_ids: list[str]
    sink_task_ids: list[str]
    khop_hyperedges_global: list[list[str]] = field(default_factory=list)
    return_complete_time: float | None = None
    completed: bool = False
    completion_reward_settled: bool = False


class DAGTaskManager:
    """Clean mainline DAG generator and task-state manager for zrj_3.

    中文：负责生成 DAG、维护任务依赖和状态转换，并统计 DAG 是否真正完成。
    所有任务时间戳都使用秒，时隙编号只在 Env 中统一换算一次。

    Time semantics (Phase 1): every physical timestamp stored on TaskNode /
    DAGJob (arrival_time, ready_time, start_time, finish_time,
    compute_finish_time, reward_completion_time, return_complete_time, and the
    `current_time_step` arguments of the mark_* / create methods) is in SECONDS
    on the episode clock. The slot-index -> seconds conversion happens only in
    environment/env.py.
    """

    def __init__(self) -> None:
        """创建空的 DAG、任务和 UE 索引，并初始化编号和版本计数器。"""
        self._jobs: dict[str, DAGJob] = {}
        self._tasks: dict[str, TaskNode] = {}
        self._tasks_by_ue: dict[int, list[str]] = {}
        self._job_counter: int = 0
        self._task_counter: int = 0
        self._last_arrival_step: int = -1
        self._dag_arrival_version: int = 0
        self._last_created_dag_ids: list[str] = []

    @property
    def jobs(self) -> dict[str, DAGJob]:
        """返回当前保存的全部 DAG。"""
        return self._jobs

    @property
    def tasks(self) -> dict[str, TaskNode]:
        """返回当前保存的全部任务节点。"""
        return self._tasks

    @property
    def dag_arrival_version(self) -> int:
        """返回 DAG 到达版本；每创建一个新 DAG 就递增一次。"""
        return self._dag_arrival_version

    @property
    def last_created_dag_ids(self) -> list[str]:
        """返回最近一次到达检查中新建的 DAG ID 副本。"""
        return list(self._last_created_dag_ids)

    def reset(self) -> None:
        """清空全部 DAG、任务、索引和版本号，开始一个新回合。"""
        self._jobs.clear()
        self._tasks.clear()
        self._tasks_by_ue.clear()
        self._job_counter = 0
        self._task_counter = 0
        self._last_arrival_step = -1
        self._dag_arrival_version = 0
        self._last_created_dag_ids = []

    def create_dag_for_ue(
        self,
        ue_id: int,
        source_pos: np.ndarray,
        arrival_time: float | None = None,
        current_time_step: float | None = None,
    ) -> DAGJob:
        """Create one clean-mainline DAG for a UE.

        中文：为指定 UE 随机生成一个分层 DAG，采样任务属性和链路带宽，
        建立依赖、k 跳超边和关键路径；同一 UE 同时只能有一个活动 DAG。

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

        # 先确定每层任务数，再逐层创建节点，保证层序天然是合法拓扑序。
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
        # 按层序写入静态拓扑编号，后续就绪任务排序不再依赖任务 ID 猜测顺序。
        for topological_index, task_id in enumerate(task_ids):
            self._tasks[task_id].topological_index = int(topological_index)
        sink_task_ids = [task_id for task_id in task_ids if not self._tasks[task_id].successors]
        khop_hyperedges_global = self._precompute_khop_hyperedges(task_ids)
        job = DAGJob(
            dag_id=dag_id,
            ue_id=ue_id,
            arrival_time=actual_arrival,
            source_pos=source_xy.copy(),
            base_upload_bandwidth_mbps=base_upload,
            base_download_bandwidth_mbps=base_download,
            task_ids=task_ids,
            sink_task_ids=sink_task_ids,
            khop_hyperedges_global=khop_hyperedges_global,
        )
        self._jobs[dag_id] = job
        self._dag_arrival_version += 1
        self._last_created_dag_ids = [dag_id]
        # DAG 登记完成后立即刷新入口任务，并标出按计算量估计的关键路径。
        self._refresh_ready_states()
        self._mark_critical_path(dag_id)
        return job

    def observe_time_step(self, ues: list[Any], current_time_step: int) -> None:
        """Compatibility arrival hook.

        中文：旧调用入口会在每个时刻为没有活动 DAG 的 UE 采样一次基础到达概率；
        clean Env 已自行处理热点到达，这里主要用于兼容旧脚本。

        Phase 1 does not implement hotspot logic. This method uses the clean
        base arrival probability and skips UEs that already have an active DAG.
        It does not mutate existing DAG source positions.
        """
        if current_time_step == self._last_arrival_step:
            self._refresh_ready_states()
            return
        self._last_arrival_step = current_time_step
        created_dag_ids: list[str] = []
        for ue in ues:
            ue_id = int(getattr(ue, "id"))
            if self._ue_has_active_dag(ue_id):
                continue
            arrival_prob = float(np.clip(config.DAG_BASE_ARRIVAL_PROB, 0.0, 1.0))
            if np.random.random() >= arrival_prob:
                continue
            pos = np.asarray(getattr(ue, "pos"), dtype=np.float32).reshape(-1)[:2]
            job = self.create_dag_for_ue(ue_id=ue_id, source_pos=pos, current_time_step=current_time_step)
            created_dag_ids.append(job.dag_id)
        self._last_created_dag_ids = created_dag_ids
        self._refresh_ready_states()

    def get_active_tasks(self) -> list[TaskNode]:
        """返回所有尚未完成或丢弃的活动任务。"""
        return [
            task
            for task in self._tasks.values()
            if task.state not in {TASK_STATE_COMPLETED, TASK_STATE_DROPPED}
        ]

    def get_all_non_returned_tasks(self) -> list[TaskNode]:
        """Compatibility view for old callers that still expect finished tasks before return.

        中文：兼容旧调用，返回尚未进入最终完成状态的任务。
        """
        return [task for task in self._tasks.values() if task.state not in {TASK_STATE_COMPLETED, TASK_STATE_DROPPED}]

    def get_ready_tasks(self) -> list[TaskNode]:
        """返回依赖已满足、等待分配的任务。"""
        return [task for task in self._tasks.values() if task.state == TASK_STATE_READY]

    def refresh_ready_states(self) -> None:
        """公开触发一次任务依赖检查和就绪状态刷新。"""
        self._refresh_ready_states()

    def get_tasks_for_ue(self, ue_id: int) -> list[TaskNode]:
        """返回指定 UE 历史上创建的全部任务。"""
        task_ids = self._tasks_by_ue.get(ue_id, [])
        return [self._tasks[task_id] for task_id in task_ids]

    def get_job(self, dag_id: str) -> DAGJob | None:
        """按 ID 查找 DAG，不存在时返回 `None`。"""
        return self._jobs.get(dag_id)

    def get_task(self, task_id: str) -> TaskNode | None:
        """按 ID 查找任务，不存在时返回 `None`。"""
        return self._tasks.get(task_id)

    def get_active_job_for_ue(self, ue_id: int) -> DAGJob | None:
        """查找 UE 当前尚未完成的 DAG。"""
        for job in self._jobs.values():
            if job.ue_id == ue_id and not job.completed:
                return job
        return None

    def get_job_tasks(self, dag_id: str) -> list[TaskNode]:
        """按 DAG 中记录的顺序返回其仍然存在的任务节点。"""
        job = self._jobs.get(dag_id)
        if job is None:
            return []
        return [self._tasks[task_id] for task_id in job.task_ids if task_id in self._tasks]

    def mark_task_queued(self, task_id: str, uav_id: int, current_time_step: float) -> None:
        """把就绪任务标为已进入指定 UAV 队列，并记录入队时间。"""
        task = self._tasks[task_id]
        if task.state != TASK_STATE_READY:
            raise ValueError(f"Task {task_id} is not ready and cannot be queued.")
        task.state = TASK_STATE_IN_SERVICE
        task.service_phase = SERVICE_PHASE_QUEUED
        task.assigned_uav = int(uav_id)
        task.enqueue_time = float(current_time_step)

    def mark_task_running(self, task_id: str, current_time_step: float) -> None:
        """把可启动任务标为计算中，并记录实际开始时间。"""
        task = self._tasks[task_id]
        if task.state not in {TASK_STATE_READY, TASK_STATE_IN_SERVICE}:
            raise ValueError(f"Task {task_id} is not queueable and cannot start.")
        task.state = TASK_STATE_IN_SERVICE
        task.service_phase = SERVICE_PHASE_COMPUTING
        task.start_time = float(current_time_step)

    def mark_task_finished(self, task_id: str, current_time_step: float) -> None:
        """完成普通任务并刷新其后继任务的就绪状态。"""
        task = self._tasks[task_id]
        task.state = TASK_STATE_COMPLETED
        task.service_phase = None
        task.compute_finish_time = float(current_time_step)
        task.finish_time = float(current_time_step)
        task.reward_completion_time = float(current_time_step)
        task.total_energy = task.compute_energy + task.communication_energy + task.return_energy
        self._refresh_ready_states()

    def mark_task_returning(self, task_id: str, current_time_step: float) -> None:
        """把出口任务标为结果回传中；此时计算已完成但奖励尚未结算。"""
        task = self._tasks[task_id]
        if task.state == TASK_STATE_COMPLETED:
            return
        task.state = TASK_STATE_RETURNING
        task.service_phase = None
        task.compute_finish_time = float(current_time_step)
        task.finish_time = float(current_time_step)
        task.total_energy = task.compute_energy + task.communication_energy + task.return_energy

    def mark_task_returned(self, task_id: str, current_time_step: float) -> None:
        """标记出口结果已回到 UE，并尝试完成整个 DAG。"""
        task = self._tasks[task_id]
        task.state = TASK_STATE_COMPLETED
        task.service_phase = None
        task.finish_time = float(current_time_step)
        task.reward_completion_time = float(current_time_step)
        task.total_energy = task.compute_energy + task.communication_energy + task.return_energy
        self.mark_dag_completed_if_ready(task.dag_id, current_time_step)

    def mark_task_dropped(self, task_id: str) -> None:
        """兼容旧逻辑，把任务标成丢弃；clean 主线不会调用该方法。"""
        self._tasks[task_id].state = TASK_STATE_DROPPED

    def mark_dag_completed_if_ready(self, dag_id: str, current_time_step: float | None = None) -> bool:
        """检查 DAG 是否全部完成，并记录最终结果返回 UE 的时间。

        所有非出口任务要计算完成，所有出口任务还要完成回传；首次完成返回 `True`。
        """
        job = self._jobs.get(dag_id)
        if job is None or job.completed:
            return False
        job_tasks = self.get_job_tasks(dag_id)
        if not job_tasks:
            return False
        non_sink_finished = all(
            task.state == TASK_STATE_COMPLETED
            for task in job_tasks
            if task.task_id not in set(job.sink_task_ids)
        )
        sink_returned = all(self._tasks[task_id].state == TASK_STATE_COMPLETED for task_id in job.sink_task_ids)
        if non_sink_finished and sink_returned:
            job.completed = True
            complete_time = current_time_step
            if complete_time is None:
                finish_times = [task.reward_completion_time for task in job_tasks if task.reward_completion_time is not None]
                complete_time = max(finish_times) if finish_times else job.arrival_time
            job.return_complete_time = float(complete_time)
            return True
        return False

    def get_job_summary(self) -> dict[str, float]:
        """汇总当前全部 DAG 的完成数、任务数、关键路径、流时间和能耗。"""
        total_jobs = len(self._jobs)
        completed_jobs = sum(1 for job in self._jobs.values() if job.completed)
        incomplete_jobs = total_jobs - completed_jobs
        generated_tasks = len(self._tasks)
        finished_tasks = sum(
            1 for task in self._tasks.values() if task.state == TASK_STATE_COMPLETED
        )
        returned_tasks = sum(
            1
            for job in self._jobs.values()
            for task_id in job.sink_task_ids
            if self._tasks[task_id].state == TASK_STATE_COMPLETED
        )
        critical_tasks = sum(1 for task in self._tasks.values() if task.is_critical_path)
        critical_finished = sum(
            1
            for task in self._tasks.values()
            if task.is_critical_path and task.state == TASK_STATE_COMPLETED
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
        """返回指定 DAG 中已完成任务所占的比例。"""
        job_tasks = self.get_job_tasks(dag_id)
        if not job_tasks:
            return 0.0
        done = sum(1 for task in job_tasks if task.state == TASK_STATE_COMPLETED)
        return float(done / max(len(job_tasks), 1))

    def get_dag_remaining_slack(self, dag_id: str, current_time_step: float) -> float:
        """Deprecated compatibility helper. Clean mainline does not use deadline/slack.

        中文：旧截止期接口的占位实现，clean 主线始终返回无穷大。
        """
        return float("inf")

    def get_descendant_count(self, task_id: str) -> int:
        """遍历任务后继关系，统计不重复的所有下游任务数量。"""
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
        """判断指定任务是否位于该 DAG 的关键路径上。"""
        task = self._tasks.get(task_id)
        return bool(task is not None and task.is_critical_path)

    def is_high_risk_job(self, dag_id: str) -> bool:
        """Deprecated compatibility helper. Deadline risk is not a clean-mainline concept.

        中文：旧高风险 DAG 接口的占位实现，clean 主线固定返回 `False`。
        """
        return False

    def build_task_features(self, current_time_step: float) -> dict[str, np.ndarray]:
        """Compatibility feature builder.

        中文：为旧模块生成稳定的任务特征字典，不包含 clean 主线已移除的截止期和任务类型语义。

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
        """Deprecated wrapper for old callers.

        中文：旧调用名的薄封装，实际工作交给 `create_dag_for_ue`。
        """
        self.create_dag_for_ue(ue_id=ue_id, source_pos=source_pos, current_time_step=current_time_step)

    def _spawn_new_jobs(self, ues: list[Any], current_time_step: int) -> None:
        """Deprecated wrapper used by old observe paths.

        中文：旧观测路径按基础概率为没有活动 DAG 的 UE 创建任务。
        """
        for ue in ues:
            ue_id = int(getattr(ue, "id"))
            if self._ue_has_active_dag(ue_id):
                continue
            if np.random.random() < float(np.clip(config.DAG_BASE_ARRIVAL_PROB, 0.0, 1.0)):
                pos = np.asarray(getattr(ue, "pos"), dtype=np.float32).reshape(-1)[:2]
                self.create_dag_for_ue(ue_id=ue_id, source_pos=pos, current_time_step=current_time_step)

    def _get_ue_arrival_prob(self, ue: Any) -> float:
        """Deprecated compatibility helper. Hotspot region logic is not implemented in Phase 1.

        中文：旧接口只返回基础 DAG 到达概率，不处理热点倍率。
        """
        return float(np.clip(config.DAG_BASE_ARRIVAL_PROB, 0.0, 1.0))

    def _drop_overdue_tasks(self, current_time_step: int) -> None:
        """Deprecated no-op. Clean mainline does not deadline-drop tasks.

        中文：兼容旧调用的空操作，clean 主线不会因超时直接丢弃任务。
        """
        return

    def _ue_has_active_dag(self, ue_id: int) -> bool:
        """判断 UE 是否已经有一个尚未完成的 DAG。"""
        for job in self._jobs.values():
            if job.ue_id == ue_id and not job.completed:
                return True
        return False

    def _sample_level_sizes(self) -> list[int]:
        """随机决定 DAG 的层数以及每一层包含的任务数。"""
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
        """随机采样数据量、复杂度和常数，创建一个尚未连接依赖的任务节点。"""
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
        """根据输入规模和复杂度类型估算任务需要的运算次数。"""
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
        """为每个非首层任务随机选择前面层中的父任务，保证图保持无环。"""
        for level_idx in range(1, len(levels)):
            parent_candidates = [task_id for previous in levels[:level_idx] for task_id in previous]
            for task_id in levels[level_idx]:
                max_parent_count = min(len(parent_candidates), int(config.DAG_MAX_PARENTS))
                parent_count = int(np.random.randint(1, max_parent_count + 1))
                chosen_parents = np.random.choice(parent_candidates, size=parent_count, replace=False)
                for parent_id in sorted(chosen_parents.tolist()):
                    self._tasks[task_id].predecessors.append(parent_id)
                    self._tasks[parent_id].successors.append(task_id)

    def _precompute_khop_hyperedges(self, task_ids: list[str]) -> list[list[str]]:
        """从每个任务出发收集最多 K 跳可达节点，并去重形成依赖超边。"""
        if not config.ENABLE_KHOP_DEPENDENCY_HYPEREDGES:
            return []
        max_hops = max(int(config.KHOP_K), 0)
        if max_hops <= 0:
            return []

        task_id_set = set(task_ids)
        dedup: set[tuple[str, ...]] = set()
        output: list[list[str]] = []
        for root_id in sorted(task_ids, key=_stable_sort_value):
            reachable: set[str] = {root_id}
            queue: deque[tuple[str, int]] = deque(
                (child_id, 1)
                for child_id in self._tasks[root_id].successors
                if child_id in task_id_set
            )
            while queue:
                task_id, depth = queue.popleft()
                if depth > max_hops or task_id not in task_id_set:
                    continue
                reachable.add(task_id)
                if depth == max_hops:
                    continue
                for child_id in self._tasks[task_id].successors:
                    if child_id in task_id_set:
                        queue.append((child_id, depth + 1))
            if len(reachable) < 2:
                continue
            group = tuple(sorted(reachable, key=_stable_sort_value))
            if group in dedup:
                continue
            dedup.add(group)
            output.append(list(group))
        return output

    def _refresh_ready_states(self) -> None:
        """根据前驱完成情况刷新所有未执行任务的等待或就绪状态。"""
        for task in self._tasks.values():
            if task.state in {TASK_STATE_COMPLETED, TASK_STATE_RETURNING, TASK_STATE_IN_SERVICE, TASK_STATE_DROPPED}:
                continue
            if not task.predecessors:
                task.state = TASK_STATE_READY
                task.service_phase = None
                if task.ready_time is None:
                    task.ready_time = float(task.arrival_time)
                continue
            if all(self._tasks[parent_id].state == TASK_STATE_COMPLETED for parent_id in task.predecessors):
                task.state = TASK_STATE_READY
                task.service_phase = None
                if task.ready_time is None:
                    parent_finish = [
                        self._tasks[parent_id].finish_time
                        for parent_id in task.predecessors
                        if self._tasks[parent_id].finish_time is not None
                    ]
                    task.ready_time = max(parent_finish) if parent_finish else float(task.arrival_time)
            else:
                task.state = TASK_STATE_WAITING_DEPENDENCY
                task.service_phase = None

    def _mark_critical_path(self, dag_id: str) -> None:
        """按累计运算量找出 DAG 的最长路径，并标记路径上的任务。"""
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
    """Return a deterministic numeric tie-breaker for ids like task_12.

    中文：优先取 ID 的数字后缀；没有数字时用字符和值生成稳定排序数。
    """
    suffix = value.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return sum(ord(char) for char in value)
