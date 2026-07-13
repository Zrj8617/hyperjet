from __future__ import annotations

from typing import Any

import config
from environment.assignment import (
    CleanAssignmentBuffer,
    TemporaryReservationState,
    freeze_ready_tasks,
    is_assignment_legal,
    legal_candidate_uav_ids,
)
from environment.dag_tasks import DAGTaskManager
from environment.metrics import CleanMetricsTracker
from environment.task_execution import CleanTaskExecutor
from environment.user_equipments import UE
from environment.uavs import UAV
import numpy as np


class Env:
    """zrj_3 clean mainline environment.

    中文：组织用户移动、DAG 到达、无人机调度、任务执行和奖励结算，
    对外提供按时隙推进的强化学习环境接口。

    This hard-cleanup environment intentionally exposes only the clean reset,
    UE mobility, hotspot-based DAG arrival, and minimal observation loop. Legacy
    graph construction, task execution, score-head scheduling, and request/cache
    paths are no longer part of this main environment entrypoint.
    """

    def __init__(self) -> None:
        """初始化环境状态、任务管理器和指标组件。"""
        self._time_step: int = 0
        self.hotspot_center: np.ndarray | None = None
        self.hotspot_radius: float = float(config.HOTSPOT_RADIUS)
        self._ues: list[UE] = []
        self._uavs: list[UAV] = []
        self._task_manager: DAGTaskManager = DAGTaskManager()
        self._executor: CleanTaskExecutor = CleanTaskExecutor()
        self._metrics: CleanMetricsTracker = CleanMetricsTracker()
        self._latest_info: dict[str, Any] = {}
        self._ue_service_positions: dict[int, np.ndarray] = {}
        self._uav_pre_move_positions: dict[int, np.ndarray] = {}
        self._uav_service_positions: dict[int, np.ndarray] = {}
        self._latest_slot_boundary: dict[str, Any] = {}
        self._slot_service_positions_frozen: bool = False
        self._frozen_ready_task_ids: list[str] = []
        self._last_assignment_buffer: CleanAssignmentBuffer = CleanAssignmentBuffer()
        self._last_new_dag_arrived: bool = False
        self._latest_dag_arrival_version: int = 0
        self._last_movement_energy_by_uav: dict[int, float] = {}
        self._last_movement_energy_total: float = 0.0
        self._last_movement_distance_by_uav: dict[int, float] = {}
        self._last_movement_distance_total: float = 0.0
        self._last_movement_hover_count: int = 0
        self._last_movement_action_count: int = 0
        self._prepared_slot_open: bool = False
        self._prepared_slot_context: dict[str, Any] = {}

    @property
    def uavs(self) -> list[UAV]:
        """返回环境中的无人机列表。"""
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        """返回环境中的用户设备列表。"""
        return self._ues

    @property
    def task_manager(self) -> DAGTaskManager:
        """返回 DAG 任务管理器。"""
        return self._task_manager

    @property
    def executor(self) -> CleanTaskExecutor:
        """返回任务执行器。"""
        return self._executor

    @property
    def metrics(self) -> CleanMetricsTracker:
        """返回环境指标跟踪器。"""
        return self._metrics

    @property
    def time_step(self) -> int:
        """返回当前时间步。"""
        return self._time_step

    @property
    def current_time_seconds(self) -> float:
        """Physical simulation time in seconds for the current slot.

        中文：把离散时隙编号统一换算为物理仿真时间，供执行器、指标和任务时间戳使用。

        Canonical clean time semantics: one slot advances physical time by
        TIME_SLOT_DURATION seconds. The slot-index -> seconds conversion happens
        ONLY here; executor, assignment estimator, DAG timestamps, and metrics
        all consume seconds. slot_index (`time_step`) remains for the episode
        loop, graph update intervals, and log labels.
        """
        return float(self._time_step) * float(config.TIME_SLOT_DURATION)

    @property
    def latest_info(self) -> dict[str, Any]:
        """返回最近一个时间步的信息副本。"""
        return dict(self._latest_info)

    @property
    def ue_service_positions(self) -> dict[int, np.ndarray]:
        """返回各用户设备的服务位置副本。"""
        return {ue_id: pos.copy() for ue_id, pos in self._ue_service_positions.items()}

    @property
    def uav_service_positions(self) -> dict[int, np.ndarray]:
        """返回各无人机的服务位置副本。"""
        return {uav_id: pos.copy() for uav_id, pos in self._uav_service_positions.items()}

    @property
    def latest_slot_boundary(self) -> dict[str, Any]:
        """返回最近时隙的状态边界信息。"""
        return dict(self._latest_slot_boundary)

    @property
    def frozen_ready_task_ids(self) -> list[str]:
        """返回当前冻结的就绪任务 ID。"""
        return list(self._frozen_ready_task_ids)

    @property
    def last_assignment_buffer(self) -> CleanAssignmentBuffer:
        """返回最近一次任务分配缓冲区的副本。"""
        return CleanAssignmentBuffer(entries=list(self._last_assignment_buffer.entries))

    @property
    def new_dag_arrived(self) -> bool:
        """判断当前时隙是否有新的 DAG 到达。"""
        return bool(self._last_new_dag_arrived)

    @property
    def dag_arrival_version(self) -> int:
        """返回 DAG 到达状态的版本号。"""
        return int(self._latest_dag_arrival_version)

    def reset(self) -> list[np.ndarray]:
        """重置环境并返回初始观测。

        该方法会重新创建实体、清空任务和统计状态，并建立第 0 个时隙的状态边界。
        """
        self._time_step = 0
        self.hotspot_radius = float(config.HOTSPOT_RADIUS)
        self.hotspot_center = self._sample_episode_hotspot()

        # 重新创建本回合的无人机和用户，并同步重置任务、执行器与指标组件。
        self._uavs = self._init_uavs_uniform()
        self._ues = self._init_ues_uniform()
        self._task_manager.reset()
        self._executor.reset(self._uavs)
        self._metrics.reset([uav.id for uav in self._uavs])

        # 保存时隙开始时的服务位置，后续移动和传输计算都基于这些位置快照。
        self._ue_service_positions = {int(ue.id): ue.pos[:2].copy() for ue in self._ues}
        self._uav_pre_move_positions = {int(uav.id): uav.pos[:2].copy() for uav in self._uavs}
        self._uav_service_positions = {int(uav.id): uav.pos[:2].copy() for uav in self._uavs}

        # 清除上一回合遗留的就绪任务、分配结果和移动统计。
        self._slot_service_positions_frozen = False
        self._frozen_ready_task_ids = []
        self._last_assignment_buffer = CleanAssignmentBuffer()
        self._last_new_dag_arrived = False
        self._latest_dag_arrival_version = self._task_manager.dag_arrival_version
        self._last_movement_energy_by_uav = {int(uav.id): 0.0 for uav in self._uavs}
        self._last_movement_energy_total = 0.0
        self._last_movement_distance_by_uav = {int(uav.id): 0.0 for uav in self._uavs}
        self._last_movement_distance_total = 0.0
        self._last_movement_hover_count = 0
        self._last_movement_action_count = 0
        self._prepared_slot_open = False
        self._prepared_slot_context = {}

        # 用 x_0^- 表示任何决策和执行发生前的初始内部状态。
        self._latest_slot_boundary = {
            "internal_state": "x_0^-",
            "decision_state": None,
            "post_execution_state": "x_0^-",
        }
        self._latest_info = {
            "time_step": self._time_step,
            "created_dags": 0,
            "service_waiting_ues": 0,
            "step_reward": 0.0,
            "episode_reward": 0.0,
        }
        return self._get_obs()

    def step(self, actions: Any | None = None) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        """执行一个完整时隙并返回环境交互结果。

        依次解析动作、冻结决策状态、应用移动动作，再提交任务分配并结算该时隙。
        """
        parsed_actions = self._parse_clean_actions(actions)
        self.prepare_slot_state()
        self.apply_movement(parsed_actions["movement_actions"])
        return self.commit_and_advance(assignments=parsed_actions["assignments"])

    def prepare_slot_state(self) -> dict[str, Any]:
        """Prepare clean decision state s_t exactly once for the next slot.

        中文：推进用户位置和 DAG 到达状态，冻结本时隙的就绪任务集合，
        返回供策略模型编码的决策上下文；同一时隙只能调用一次。

        This stage is environment/parameter independent: UE movement, DAG
        arrivals, ready refresh, and frozen ready-set construction happen here.
        Model encoding is intentionally outside this method.
        """
        if self._prepared_slot_open:
            raise RuntimeError("A prepared clean slot is already open; commit it before preparing the next slot.")

        # 先推进时钟和用户移动，再冻结用户服务位置，保证本时隙内位置口径一致。
        slot_index = self._time_step
        previous_internal_state = f"x_{slot_index}^-"
        self._time_step += 1
        for ue in self._ues:
            ue.update_position()
        self._ue_service_positions = {int(ue.id): ue.pos[:2].copy() for ue in self._ues}
        self._slot_service_positions_frozen = True

        # 处理新任务到达并冻结就绪集合，避免策略决策期间任务集合发生变化。
        created_dags = self._process_clean_dag_arrivals()
        self._task_manager.refresh_ready_states()
        ready_tasks = freeze_ready_tasks(self._task_manager)
        self._frozen_ready_task_ids = [task.task_id for task in ready_tasks]
        self._uav_pre_move_positions = {int(uav.id): uav.pos[:2].copy() for uav in self._uavs}

        # 记录决策前后的状态标签和快照，供训练轨迹与诊断日志复用。
        decision_state = f"s_{slot_index}"
        self._latest_slot_boundary = {
            "slot": slot_index,
            "started_from": previous_internal_state,
            "decision_state": decision_state,
            "post_execution_state": None,
            "next_decision_state": None,
        }
        self._prepared_slot_context = {
            "slot_index": slot_index,
            "time_step": self._time_step,
            "previous_internal_state": previous_internal_state,
            "decision_state": decision_state,
            "created_dags": created_dags,
            "new_dag_arrived": self._last_new_dag_arrived,
            "dag_arrival_version": self._latest_dag_arrival_version,
            "frozen_ready_task_ids": list(self._frozen_ready_task_ids),
            "ue_service_positions": self.ue_service_positions,
            "uav_pre_move_positions": {uav_id: pos.copy() for uav_id, pos in self._uav_pre_move_positions.items()},
        }
        self._prepared_slot_open = True
        return self._copy_slot_context(self._prepared_slot_context)

    def apply_movement(self, movement_actions: dict[int, int | str] | None = None) -> dict[str, Any]:
        """Apply movement after R_t is frozen and expose current service positions.

        中文：在就绪任务集合冻结后应用无人机移动动作，并返回移动能耗及移动前后的服务位置。
        未提供动作的无人机默认悬停。
        """
        if not self._prepared_slot_open:
            raise RuntimeError("prepare_slot_state() must be called before apply_movement().")
        self._apply_clean_movement(movement_actions or {})
        return {
            "movement_energy_by_uav": dict(self._last_movement_energy_by_uav),
            "movement_energy_total": float(self._last_movement_energy_total),
            "uav_pre_move_positions": {uav_id: pos.copy() for uav_id, pos in self._uav_pre_move_positions.items()},
            "uav_service_positions": self.uav_service_positions,
        }

    def commit_and_advance(
        self,
        *,
        assignments: dict[str, int] | None = None,
        assignment_buffer: CleanAssignmentBuffer | None = None,
        offloading_skip_count: int = 0,
    ) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        """Commit clean assignments, advance executor one slot, and close reward/metrics.

        中文：校验并提交冻结任务的分配结果，推进任务执行器一个时隙，
        随后结算奖励、累计指标并返回下一观测。调用后当前准备时隙会被关闭。
        """
        if not self._prepared_slot_open:
            raise RuntimeError("prepare_slot_state() must be called before commit_and_advance().")
        if not self._uav_service_positions:
            self.apply_movement({})

        # 恢复 prepare_slot_state 冻结的上下文，确保分配只针对同一决策状态。
        context = self._prepared_slot_context
        slot_index = int(context["slot_index"])
        previous_internal_state = str(context["previous_internal_state"])
        decision_state = str(context["decision_state"])
        created_dags = int(context["created_dags"])

        # 若调用方未提供预构建缓冲区，则按当前容量和合法性规则过滤原始分配请求。
        if assignment_buffer is None:
            ready_tasks = self._frozen_ready_tasks_from_ids(self._frozen_ready_task_ids)
            assignment_buffer, offloading_skip_count = self._build_assignment_buffer(ready_tasks, assignments or {})
        else:
            assignment_buffer = CleanAssignmentBuffer(entries=list(assignment_buffer.entries))
        self._last_assignment_buffer = assignment_buffer

        # 先把合法任务写入各无人机队列，再统一推进计算、通信和结果回传。
        self._executor.assign_tasks(
            assignments=assignment_buffer,
            task_manager=self._task_manager,
            uavs=self._uavs,
            ues=self._ues,
            current_time_seconds=self.current_time_seconds,
            uav_service_positions=self._uav_service_positions,
            ue_service_positions=self._ue_service_positions,
        )
        execution_stats = self._executor.advance_one_slot(
            task_manager=self._task_manager,
            uavs=self._uavs,
            ues=self._ues,
            current_time_seconds=self.current_time_seconds,
            uav_service_positions=self._uav_service_positions,
            ue_service_positions=self._ue_service_positions,
        )
        for dag_id in execution_stats.completed_dag_ids:
            self.release_ue_after_dag_completed(dag_id)

        # 执行完成后，内部状态从决策状态 s_t 转移到下一时隙前的 x_(t+1)^-。
        post_execution_state = f"x_{slot_index + 1}^-"
        self._latest_slot_boundary = {
            "slot": slot_index,
            "started_from": previous_internal_state,
            "decision_state": decision_state,
            "post_execution_state": post_execution_state,
            "next_decision_state": None,
        }

        # 奖励使用本时隙执行结果、任务能耗、移动能耗及可选的位置塑形信号。
        movement_position_signal = self._compute_movement_position_signal()
        step_reward = self._metrics.calculate_step_reward(
            self._task_manager,
            execution_stats,
            movement_energy_slot=self._last_movement_energy_total,
            movement_position_signal=movement_position_signal,
        )
        queue_lengths = [len(self._executor.uav_queues.get(int(uav.id), [])) for uav in self._uavs]
        self._metrics.update(
            task_manager=self._task_manager,
            execution_stats=execution_stats,
            step_reward=step_reward,
            created_dags=created_dags,
            action_count=assignment_buffer.entry_count,
            queue_lengths=queue_lengths,
            elapsed_steps=self._time_step,
            movement_energy_by_uav=self._last_movement_energy_by_uav,
            movement_hover_count=self._last_movement_hover_count,
            movement_action_count=self._last_movement_action_count,
            movement_displacement_total=self._last_movement_distance_total,
        )
        metric_info = self._metrics.to_info(self._time_step, total_time_seconds=self.current_time_seconds)
        metric_info["generated_dag_count"] = float(len(self._task_manager.jobs))

        # info 汇总训练、评估和诊断所需的时隙级明细，不参与环境状态更新。
        info = {
            "time_step": self._time_step,
            "created_dags": created_dags,
            "newly_assigned_tasks": execution_stats.newly_assigned_tasks,
            "invalid_assignments": execution_stats.invalid_assignments,
            "completed_tasks": execution_stats.completed_tasks,
            "completed_dags": execution_stats.completed_dags,
            "step_reward": step_reward.reward_total,
            "step_time_penalty": step_reward.time_penalty,
            "step_energy_penalty": step_reward.energy_penalty,
            "step_task_energy_penalty": step_reward.task_energy_penalty,
            "step_movement_energy_penalty": step_reward.movement_energy_penalty,
            "step_completed_dag_bonus": step_reward.completed_dag_bonus,
            "step_movement_position_bonus": step_reward.movement_position_bonus,
            "movement_position_signal": movement_position_signal,
            "step_task_energy": execution_stats.step_task_energy,
            "step_movement_energy": self._last_movement_energy_total,
            "step_compute_energy": execution_stats.step_compute_energy,
            "step_communication_energy": execution_stats.step_communication_energy,
            "step_return_energy": execution_stats.step_return_energy,
            "active_dags": sum(1 for job in self._task_manager.jobs.values() if not job.completed),
            "service_waiting_ues": sum(1 for ue in self._ues if ue.service_waiting),
            "hotspot_center": None if self.hotspot_center is None else self.hotspot_center.copy(),
            "hotspot_radius": self.hotspot_radius,
            "slot_started_from": previous_internal_state,
            "slot_decision_state": decision_state,
            "slot_post_execution_state": post_execution_state,
            "frozen_ready_task_count": len(self._frozen_ready_task_ids),
            "assignment_buffer_entry_count": assignment_buffer.entry_count,
            "offloading_skipped_no_candidate": offloading_skip_count,
        }
        info.update(metric_info)
        self._latest_info = info

        # 所有无人机共享全局奖励，并按智能体数量平均以保持奖励尺度稳定。
        reward_per_uav = step_reward.reward_total / float(max(len(self._uavs), 1))
        rewards = [reward_per_uav for _ in self._uavs]
        done = self._time_step >= int(config.EPISODE_LENGTH)
        self._prepared_slot_open = False
        self._prepared_slot_context = {}
        return self._get_obs(), rewards, done, dict(info)

    def _sample_episode_hotspot(self) -> np.ndarray:
        """在地图有效范围内随机生成本回合的热点中心。

        采样前会检查热点半径，确保生成的圆形热点不会越过地图边界。
        """
        radius = float(config.HOTSPOT_RADIUS)
        width = float(config.AREA_WIDTH)
        height = float(config.AREA_HEIGHT)
        if radius <= 0.0:
            raise ValueError("HOTSPOT_RADIUS must be positive.")
        if 2.0 * radius > width or 2.0 * radius > height:
            raise ValueError("HOTSPOT_RADIUS is too large for the configured map.")
        return np.array(
            [
                np.random.uniform(radius, width - radius),
                np.random.uniform(radius, height - radius),
            ],
            dtype=np.float32,
        )

    def _init_uavs_uniform(self) -> list[UAV]:
        """按配置数量初始化无人机。"""
        return [UAV(i) for i in range(config.NUM_UAVS)]

    def _init_ues_uniform(self) -> list[UE]:
        """按配置数量初始化并重置用户设备。"""
        ues = [UE(i) for i in range(config.NUM_UES)]
        for ue in ues:
            ue.reset_episode_state()
        return ues

    def _process_clean_dag_arrivals(self) -> int:
        """为当前空闲用户采样新的 DAG 到达事件。

        已有活动 DAG 的用户会被跳过；返回本时隙新建的 DAG 数量，并同步更新到达版本。
        """
        if not self._slot_service_positions_frozen:
            self._ue_service_positions = {int(ue.id): ue.pos[:2].copy() for ue in self._ues}
        version_before = self._task_manager.dag_arrival_version
        created_count = 0

        # 每个空闲用户根据其相对热点的位置独立计算任务到达概率。
        for ue in self._ues:
            if ue.active_dag_id is not None:
                continue
            if self._task_manager.get_active_job_for_ue(ue.id) is not None:
                continue
            arrival_prob = ue.get_arrival_probability(self.hotspot_center, self.hotspot_radius)
            if np.random.random() >= arrival_prob:
                continue

            # DAG 的源位置使用本时隙冻结的用户位置，避免移动后位置造成时间口径混乱。
            job = self._task_manager.create_dag_for_ue(
                ue_id=ue.id,
                source_pos=self._ue_service_positions.get(int(ue.id), ue.pos[:2]).copy(),
                current_time_step=self.current_time_seconds,  # DAG timestamps are seconds
            )
            ue.enter_service_waiting(job.dag_id)
            created_count += 1

        # 版本号供图构建器等下游组件判断是否需要刷新缓存。
        version_after = self._task_manager.dag_arrival_version
        self._last_new_dag_arrived = version_after > version_before or created_count > 0
        self._latest_dag_arrival_version = version_after
        return created_count

    def release_ue_after_dag_completed(self, dag_id: str) -> None:
        """在 DAG 完成后解除对应用户设备的等待状态。"""
        for ue in self._ues:
            if ue.active_dag_id == dag_id:
                ue.release_service_waiting(dag_id)
                return

    def _get_obs(self) -> list[np.ndarray]:
        """构建所有无人机的归一化环境观测。

        每个观测依次包含无人机二维位置和热点中心二维位置，数值均缩放到地图比例。
        """
        hotspot = (
            np.zeros((2,), dtype=np.float32)
            if self.hotspot_center is None
            else self.hotspot_center.astype(np.float32, copy=False)
        )
        hotspot_norm = hotspot / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32)
        obs: list[np.ndarray] = []
        for uav in self._uavs:
            pos_norm = np.clip(
                uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32),
                0.0,
                1.0,
            )
            obs.append(np.concatenate([pos_norm, hotspot_norm]).astype(np.float32, copy=False))
        return obs

    def _parse_clean_actions(self, actions: Any | None) -> dict[str, Any]:
        """将输入动作统一解析为任务分配和移动动作。

        同时兼容完整动作字典和仅包含任务分配的旧格式；无效字段会安全退化为空字典。
        """
        if not isinstance(actions, dict):
            return {"assignments": {}, "movement_actions": {}}
        if "assignments" in actions or "offloading_assignments" in actions or "movement_actions" in actions:
            assignments = actions.get("assignments", actions.get("offloading_assignments", {}))
            movement_actions = actions.get("movement_actions", {})
            return {
                "assignments": assignments if isinstance(assignments, dict) else {},
                "movement_actions": movement_actions if isinstance(movement_actions, dict) else {},
            }
        return {"assignments": actions, "movement_actions": {}}

    def _build_assignment_buffer(
        self,
        ready_tasks: list[Any],
        requested_assignments: dict[str, int],
    ) -> tuple[CleanAssignmentBuffer, int]:
        """筛选合法分配并构建任务分配缓冲区。

        返回按冻结任务顺序排列的合法分配，以及因没有候选无人机而跳过的任务数量。
        """
        # 临时预留状态模拟本轮前序分配对队列容量的占用，防止批量决策超额分配。
        reservation = TemporaryReservationState.from_executor(self._uavs, self._executor)
        valid_uav_ids = {int(uav.id) for uav in self._uavs}
        ordered_uav_ids = sorted(valid_uav_ids)
        buffer = CleanAssignmentBuffer()
        skipped_no_candidate = 0
        for decision_order, task in enumerate(ready_tasks):
            # 先计算当前临时状态下的合法候选集；没有候选时单独计入诊断指标。
            legal_candidates = legal_candidate_uav_ids(
                task=task,
                uav_ids=ordered_uav_ids,
                state_view=reservation,
                executor=self._executor,
                service_positions=self._uav_service_positions,
            )
            if not legal_candidates:
                skipped_no_candidate += 1
                continue
            if task.task_id not in requested_assignments:
                continue
            try:
                selected_uav_id = int(requested_assignments[task.task_id])
            except (TypeError, ValueError):
                continue

            # 即使策略给出了无人机，也要再次校验容量、连接和执行状态等硬约束。
            if not is_assignment_legal(
                task=task,
                uav_id=selected_uav_id,
                state_view=reservation,
                valid_uav_ids=valid_uav_ids,
                executor=self._executor,
                service_positions=self._uav_service_positions,
            ):
                continue
            buffer.append(task.task_id, selected_uav_id, decision_order)
            # 立即预留资源，使后续任务看到本轮已经接受的分配。
            reservation.reserve(task.task_id, selected_uav_id)
        return buffer, skipped_no_candidate

    def _frozen_ready_tasks_from_ids(self, task_ids: list[str]) -> list[Any]:
        """根据 ID 获取仍然存在的冻结就绪任务。"""
        tasks = []
        for task_id in task_ids:
            task = self._task_manager.get_task(task_id)
            if task is not None:
                tasks.append(task)
        return tasks

    def _copy_slot_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """复制时隙上下文并隔离其中的可变数据。

        位置数组和任务 ID 列表会额外复制，避免调用方修改环境内部快照。
        """
        copied = dict(context)
        for key in ("ue_service_positions", "uav_pre_move_positions"):
            copied[key] = {int(item_id): np.asarray(pos, dtype=np.float32).copy() for item_id, pos in context.get(key, {}).items()}
        copied["frozen_ready_task_ids"] = list(context.get("frozen_ready_task_ids", []))
        return copied

    def _apply_clean_movement(self, movement_actions: dict[int, int | str]) -> None:
        """执行无人机移动并统计实际位移、悬停次数和移动能耗。

        越界动作会退化为原地不动，但仍按原始动作类别记录是否为悬停。
        """
        # 每个时隙重新建立移动前位置快照，并清空上一时隙的移动统计。
        self._uav_pre_move_positions = {int(uav.id): uav.pos[:2].copy() for uav in self._uavs}
        self._uav_service_positions = {}
        self._last_movement_energy_by_uav = {}
        self._last_movement_energy_total = 0.0
        self._last_movement_distance_by_uav = {}
        self._last_movement_distance_total = 0.0
        self._last_movement_hover_count = 0
        self._last_movement_action_count = 0
        for uav in self._uavs:
            uav_id = int(uav.id)
            pre_pos = self._uav_pre_move_positions[uav_id]
            action = movement_actions.get(uav_id, movement_actions.get(str(uav_id), "hover"))
            action_name = self._movement_action_name(action)
            delta = self._movement_delta(action)
            candidate = pre_pos + delta

            # 地图外的候选位置不执行移动，服务位置保持在时隙开始处。
            if not self._inside_map(candidate):
                candidate = pre_pos.copy()
            self._uav_service_positions[uav_id] = candidate.astype(np.float32, copy=True)
            move_distance = float(np.linalg.norm(candidate - pre_pos))
            move_energy = float(config.CLEAN_POWER_MOVE) * float(config.TIME_SLOT_DURATION) if move_distance > 0.0 else 0.0
            self._last_movement_energy_by_uav[uav_id] = move_energy
            self._last_movement_energy_total += move_energy
            # Displacement/hover diagnostics: hover ratio is by chosen ACTION (a blocked
            # boundary move is not a hover); displacement is the actual distance moved.
            self._last_movement_distance_by_uav[uav_id] = move_distance
            self._last_movement_distance_total += move_distance
            self._last_movement_action_count += 1
            if action_name == str(config.CLEAN_MOVEMENT_HOVER_ACTION):
                self._last_movement_hover_count += 1
            uav.update_position(candidate)

    def _movement_action_name(self, action: int | str) -> str:
        """将移动动作索引转换为动作名称。"""
        if isinstance(action, str):
            return action
        action_names = tuple(getattr(config, "CLEAN_MOVEMENT_ACTIONS", ("hover", "+x", "-x", "+y", "-y")))
        index = int(action)
        return action_names[index] if 0 <= index < len(action_names) else "hover"

    def _compute_movement_position_signal(self) -> float:
        """Coverage signal for the OPTIONAL movement position shaping term.

        中文：计算就绪任务被无人机覆盖的比例信号。

        Returns the fraction of current frozen ready tasks whose demand origin
        (task source position) is within UAV coverage radius of at least one UAV
        service position. Only computed when shaping is enabled; otherwise 0.0 so
        the clean spec baseline reward is untouched.
        """
        if not bool(getattr(config, "ENABLE_MOVEMENT_POSITION_SHAPING", False)):
            return 0.0
        ready_task_ids = self._frozen_ready_task_ids
        if not ready_task_ids or not self._uav_service_positions:
            return 0.0
        coverage_radius = float(config.UAV_COVERAGE_RADIUS)
        uav_positions = [
            np.asarray(pos, dtype=np.float32).reshape(-1)[:2] for pos in self._uav_service_positions.values()
        ]
        covered = 0
        counted = 0
        for task_id in ready_task_ids:
            task = self._task_manager.get_task(task_id)
            if task is None:
                continue
            counted += 1
            source_xy = np.asarray(task.source_pos, dtype=np.float32).reshape(-1)[:2]
            nearest = min(float(np.linalg.norm(pos - source_xy)) for pos in uav_positions)
            if nearest <= coverage_radius:
                covered += 1
        if counted == 0:
            return 0.0
        return float(covered) / float(counted)

    def _movement_delta(self, action: int | str) -> np.ndarray:
        """把悬停或四个方向的离散动作换成本时隙实际移动的二维距离。"""
        step_distance = float(config.CLEAN_UAV_MOVEMENT_SPEED) * float(config.TIME_SLOT_DURATION)
        if isinstance(action, str):
            action_name = action
        else:
            action_names = tuple(getattr(config, "CLEAN_MOVEMENT_ACTIONS", ("hover", "+x", "-x", "+y", "-y")))
            action_name = action_names[int(action)] if 0 <= int(action) < len(action_names) else "hover"
        if action_name == "+x":
            return np.array([step_distance, 0.0], dtype=np.float32)
        if action_name == "-x":
            return np.array([-step_distance, 0.0], dtype=np.float32)
        if action_name == "+y":
            return np.array([0.0, step_distance], dtype=np.float32)
        if action_name == "-y":
            return np.array([0.0, -step_distance], dtype=np.float32)
        return np.zeros((2,), dtype=np.float32)

    def _inside_map(self, position: np.ndarray) -> bool:
        """判断二维位置是否仍在包含边界的矩形地图内。"""
        return (
            0.0 <= float(position[0]) <= float(config.AREA_WIDTH)
            and 0.0 <= float(position[1]) <= float(config.AREA_HEIGHT)
        )
