# 导入用户设备(UE)类，用于表示终端用户设备
from dataclasses import replace
from typing import Any

from environment.user_equipments import UE
# 导入无人机(UAV)类，用于表示无人机执行节点
from environment.uavs import UAV
from environment.dag_tasks import DAGTaskManager, TASK_STATE_QUEUED, TASK_STATE_RUNNING
from environment.graph_builder import HeteroGraphBuilder, HeteroGraphSnapshot
from environment.task_execution import PhaseOneTaskExecutor
from marl_models.hgnn.scheduler import PhaseOneGraphScheduler, GraphSchedulingOutput
# 导入全局配置参数（如区域大小、无人机数量、速度、覆盖半径等）
import config
# 导入数值计算库，用于数组、矩阵、距离、归一化等计算
import numpy as np
import os
import torch


def _load_scheduler_state_compatible(scheduler: PhaseOneGraphScheduler, state_dict: dict) -> None:
    model_state = scheduler.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    incompatible = sorted(
        key
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    )
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    skipped = sorted(set(state_dict) - set(compatible_state))
    critical_prefixes = ("encoder.", "score_head.")
    critical_skipped = [
        key
        for key in sorted(set(incompatible) | set(missing) | set(unexpected))
        if key.startswith(critical_prefixes)
    ]
    if critical_skipped:
        raise RuntimeError(
            "HGNN checkpoint is incompatible for critical scheduler parameters: "
            + ", ".join(critical_skipped[:12])
            + (" ..." if len(critical_skipped) > 12 else "")
        )
    scheduler.load_state_dict(compatible_state, strict=False)
    if skipped:
        print(
            "HGNN checkpoint partially loaded; skipped incompatible keys: "
            + ", ".join(skipped[:8])
            + (" ..." if len(skipped) > 8 else "")
        )


class Env:
    """强化学习仿真环境类：负责无人机调度、状态更新、动作执行、奖励计算等核心逻辑"""
    def __init__(self) -> None:
        """环境初始化构造函数"""
        # 宏基站(MBS)位置，从配置文件读取
        self._mbs_pos: np.ndarray = config.MBS_POS
        # 初始化UE类的静态参数
        UE.initialize_ue_class()
        # 创建所有UE对象列表，数量由配置文件指定
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)]
        # 创建所有UAV对象列表，数量由配置文件指定
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self.hotspot_center: np.ndarray | None = None
        self.hotspot_radius: float = float(config.HOTSPOT_RADIUS)
        # 初始化环境时间步为0
        self._time_step: int = 0
        # 阶段一动态DAG任务系统
        self._task_manager: DAGTaskManager = DAGTaskManager()
        self._graph_builder: HeteroGraphBuilder = HeteroGraphBuilder()
        self._latest_graph_snapshot: HeteroGraphSnapshot | None = None
        self._latest_assignment_graph_snapshot: HeteroGraphSnapshot | None = None
        self._task_executor: PhaseOneTaskExecutor = PhaseOneTaskExecutor()
        self._graph_scheduler: PhaseOneGraphScheduler | None = None
        self._latest_graph_scheduling_output: GraphSchedulingOutput | None = None
        self._latest_selective_score_stats: dict[str, float] = {}
        self._latest_phase_one_diagnostics: dict[str, float] = {}
        self._last_phase_one_job_counts: dict[str, float] = {
            "dag_successful_jobs": 0.0,
            "dag_on_time_successful_jobs": 0.0,
            "dag_failed_jobs": 0.0,
        }
        self._latest_phase_one_reward_terms: dict[str, float] = {}
        self._assignment_policy: Any | None = None
        self._latest_assignment_rl_records: list[dict[str, Any]] = []
        if config.USE_HGNN_SCORE_ASSIGNMENT:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._graph_scheduler = PhaseOneGraphScheduler(device=device)
            if config.HGNN_SCORE_CHECKPOINT:
                if not os.path.exists(config.HGNN_SCORE_CHECKPOINT):
                    raise FileNotFoundError(f"HGNN score checkpoint not found: {config.HGNN_SCORE_CHECKPOINT}")
                state_dict = torch.load(config.HGNN_SCORE_CHECKPOINT, map_location=device)
                _load_scheduler_state_compatible(self._graph_scheduler, state_dict)
                self._graph_scheduler.eval()

    @property
    def uavs(self) -> list[UAV]:
        """外部获取无人机列表（只读属性）"""
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        """外部获取用户设备列表（只读属性）"""
        return self._ues

    @property
    def task_manager(self) -> DAGTaskManager:
        return self._task_manager

    @property
    def latest_graph_snapshot(self) -> HeteroGraphSnapshot | None:
        return self._latest_graph_snapshot

    @property
    def latest_assignment_graph_snapshot(self) -> HeteroGraphSnapshot | None:
        return self._latest_assignment_graph_snapshot

    @property
    def task_executor(self) -> PhaseOneTaskExecutor:
        return self._task_executor

    @property
    def latest_graph_scheduling_output(self) -> GraphSchedulingOutput | None:
        return self._latest_graph_scheduling_output

    @property
    def latest_phase_one_diagnostics(self) -> dict[str, float]:
        return self._latest_phase_one_diagnostics

    @property
    def latest_assignment_rl_records(self) -> list[dict[str, Any]]:
        return self._latest_assignment_rl_records

    def set_assignment_policy(self, policy: Any | None) -> None:
        self._assignment_policy = policy

    def reset(self) -> list[np.ndarray]:
        """
        重置环境到初始状态
        返回：每个UAV的初始观测值列表
        """
        self.hotspot_center = self._sample_episode_hotspot()
        self.hotspot_radius = float(config.HOTSPOT_RADIUS)
        self._ues = self._init_ues_uniform()
        self._uavs = self._init_uavs_uniform()
        # 时间步归零
        self._time_step = 0
        self._task_manager.reset()
        self._latest_graph_snapshot = None
        self._latest_assignment_graph_snapshot = None
        self._latest_graph_scheduling_output = None
        self._latest_selective_score_stats = {}
        self._latest_phase_one_diagnostics = {}
        self._last_phase_one_job_counts = {
            "dag_successful_jobs": 0.0,
            "dag_on_time_successful_jobs": 0.0,
            "dag_failed_jobs": 0.0,
        }
        self._latest_phase_one_reward_terms = {}
        self._latest_assignment_rl_records = []
        self._task_executor.reset(self._uavs)
        return self._get_clean_phase2_obs()

    def _sample_episode_hotspot(self) -> np.ndarray:
        radius = float(config.HOTSPOT_RADIUS)
        if radius * 2.0 > float(config.AREA_WIDTH) or radius * 2.0 > float(config.AREA_HEIGHT):
            raise ValueError("HOTSPOT_RADIUS is too large for the configured map.")
        return np.array(
            [
                np.random.uniform(radius, float(config.AREA_WIDTH) - radius),
                np.random.uniform(radius, float(config.AREA_HEIGHT) - radius),
            ],
            dtype=np.float32,
        )

    def _init_ues_uniform(self) -> list[UE]:
        UE.initialize_ue_class()
        ues = [UE(i) for i in range(config.NUM_UES)]
        for ue in ues:
            ue.reset_episode_state(uniform_position=True)
        return ues

    def _init_uavs_uniform(self) -> list[UAV]:
        return [UAV(i) for i in range(config.NUM_UAVS)]

    def _process_clean_dag_arrivals(self) -> int:
        created_count = 0
        for ue in self._ues:
            if ue.active_dag_id is not None:
                continue
            if self._task_manager.get_active_job_for_ue(ue.id) is not None:
                continue
            arrival_prob = ue.get_arrival_probability(self.hotspot_center, self.hotspot_radius)
            if np.random.random() >= arrival_prob:
                continue
            job = self._task_manager.create_dag_for_ue(
                ue_id=ue.id,
                source_pos=ue.pos[:2].copy(),
                current_time_step=self._time_step,
            )
            ue.enter_service_waiting(job.dag_id)
            created_count += 1
        return created_count

    def release_ue_after_dag_completed(self, dag_id: str) -> None:
        for ue in self._ues:
            if ue.active_dag_id == dag_id:
                ue.release_service_waiting(dag_id)
                return

    def _get_clean_phase2_obs(self) -> list[np.ndarray]:
        return [
            np.clip(
                uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32),
                0.0,
                1.0,
            ).astype(np.float32, copy=False)
            for uav in self._uavs
        ]

    def step(self, actions: np.ndarray) -> tuple[list[np.ndarray], list[float], tuple[float, float, float, float]]:
        """
        执行一步环境交互（强化学习核心步骤）
        输入：actions -> 所有UAV的动作（位置移动等）
        返回：下一时刻观测、奖励、性能指标
        """
        # 时间步 +1
        self._time_step += 1

        phase_two_clean_mode = (
            config.ENABLE_DYNAMIC_DAG
            and config.ENABLE_PHASE_ONE_EXECUTION
            and not config.ENABLE_LEGACY_REQUEST_PIPELINE
        )
        if phase_two_clean_mode:
            for ue in self._ues:
                ue.update_position()
            created_count = self._process_clean_dag_arrivals()
            self._task_manager._refresh_ready_states()  # noqa: SLF001 - Phase 2 keeps graph/executor out of the path.
            self._latest_phase_one_diagnostics = {
                "clean_phase2_created_dags": float(created_count),
                "clean_phase2_service_waiting_ues": float(sum(1 for ue in self._ues if ue.service_waiting)),
            }
            rewards = [0.0 for _ in self._uavs]
            metrics = (0.0, 0.0, 0.0, 0.0)
            return self._get_clean_phase2_obs(), rewards, metrics

        if config.ENABLE_LEGACY_REQUEST_PIPELINE:
            # 1. 每架无人机计算初始负载状态
            for uav in self._uavs:
                uav.calculate_initial_load()

            # 2. 无人机处理用户任务请求
            for uav in self._uavs:
                uav.process_requests()

        # 更新阶段一任务图并尝试分配ready任务
        if config.ENABLE_DYNAMIC_DAG and config.ENABLE_PHASE_ONE_EXECUTION:
            self._task_manager.observe_time_step(self._ues, self._time_step)
            self._latest_graph_snapshot = self._graph_builder.build(
                self._task_manager,
                self._uavs,
                self._time_step,
                self._task_executor,
            )
            self._latest_assignment_graph_snapshot = self._latest_graph_snapshot
            self._latest_graph_scheduling_output = None
            self._latest_selective_score_stats = {}
            edge_scores: dict[tuple[str, int], float] | None = None
            score_provider = None
            self._latest_assignment_rl_records = []
            if config.USE_RL_ASSIGNMENT:
                if self._assignment_policy is None:
                    raise RuntimeError("USE_RL_ASSIGNMENT=True requires an assignment policy.")
                self._assign_ready_tasks_sequential_rl()
            else:
                if (
                    config.USE_HGNN_SCORE_ASSIGNMENT
                    and config.USE_HGNN_PER_TASK_RESCORING
                    and self._graph_scheduler is not None
                ):
                    score_provider = self._build_per_task_score_provider()
                elif (
                    config.USE_HGNN_SCORE_ASSIGNMENT
                    and self._graph_scheduler is not None
                    and self._latest_graph_snapshot is not None
                ):
                    score_snapshot = self._build_selective_score_snapshot(self._latest_graph_snapshot)
                    self._latest_graph_scheduling_output = self._graph_scheduler.score_graph(score_snapshot)
                    edge_scores = self._latest_graph_scheduling_output.edge_scores
                allowed_edges = (
                    set(self._latest_graph_snapshot.task_uav_edges) if self._latest_graph_snapshot is not None else None
                )
                self._task_executor.assign_ready_tasks(
                    self._task_manager,
                    self._uavs,
                    self._time_step,
                    allowed_edges,
                    edge_scores=edge_scores,
                    score_provider=score_provider,
                )

        # 3. 更新UE电池与服务覆盖状态
        for ue in self._ues:
            # 未被分配的UE，电池消耗为0
            if not ue.assigned:
                ue.update_battery(0.0, 0.0)
            # 更新服务覆盖时长
            if config.ENABLE_LEGACY_REQUEST_PIPELINE:
                ue.update_service_coverage(self._time_step)

        if config.ENABLE_DYNAMIC_DAG and config.ENABLE_PHASE_ONE_EXECUTION:
            self._task_executor.advance_one_slot(self._task_manager, self._uavs, self._time_step)

        # 4. 更新无人机指数移动平均(EMA)与缓存状态
        for uav in self._uavs:
            if config.ENABLE_LEGACY_REQUEST_PIPELINE:
                uav.update_ema_and_cache()
            uav.update_energy_consumption()

        # 5. 计算当前步的奖励与系统指标
        if config.ENABLE_PHASE_ONE_EXECUTION and not config.ENABLE_LEGACY_REQUEST_PIPELINE:
            rewards, metrics = self._get_phase_one_rewards_and_metrics()
            self._latest_phase_one_diagnostics = self._build_phase_one_diagnostics()
        else:
            rewards, metrics = self._get_rewards_and_metrics()
            self._latest_phase_one_diagnostics = {}

        # 6. 每隔固定步长，执行一次缓存更新（GDSF策略）
        if config.ENABLE_LEGACY_REQUEST_PIPELINE and self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            for uav in self._uavs:
                uav.gdsf_cache_update()

        # 7. 为下一步做准备：更新所有UE位置
        for ue in self._ues:
            ue.update_position()

        # 8. 无人机重置临时状态（如覆盖UE列表）
        for uav in self._uavs:
            uav.reset_for_next_step()

        # 9. 执行智能体输出的动作（更新无人机位置、避撞、越界处理）
        assigned_distances_before_action = None
        coverages_before_action = None
        if config.ENABLE_PHASE_ONE_EXECUTION and config.USE_STAGE_B_MOVEMENT_REWARD:
            assigned_distances_before_action = self._get_assigned_task_distances_by_uav()
            coverages_before_action = self._get_stage_b_coverages_by_uav()
        self._apply_actions_to_env(actions)
        if config.ENABLE_PHASE_ONE_EXECUTION and config.USE_STAGE_B_MOVEMENT_REWARD:
            rewards = self._get_stage_b_movement_rewards(assigned_distances_before_action, coverages_before_action)
            self._latest_phase_one_diagnostics.update(self._latest_phase_one_reward_terms)

        # 10. 获取下一步观测
        next_obs: list[np.ndarray] = self._get_obs()
        # 返回：下一观测、奖励、系统指标
        return next_obs, rewards, metrics

    def _assign_ready_tasks_sequential_rl(self) -> None:
        """Runs RL assignment against fresh executor state one task at a time."""
        self._task_executor.begin_sequential_rl_step()
        ready_tasks = sorted(
            list(self._task_manager.get_ready_tasks()),
            key=lambda task: (task.deadline, task.level, task.task_id),
        )
        for task in ready_tasks:
            candidate_uav_ids, rejected_uav_reasons = self._task_executor.get_sequential_rl_feasible_uav_ids(
                task,
                self._task_manager,
                self._uavs,
                self._time_step,
            )
            if not candidate_uav_ids:
                self._task_executor.record_sequential_rl_no_feasible_candidate(
                    task,
                    self._time_step,
                    rejected_uav_reasons,
                )
                self._latest_assignment_rl_records.append(
                    {
                        "task_id": task.task_id,
                        "task_type": int(task.task_type),
                        "env_step_id": int(self._time_step),
                        "actor_called": False,
                        "candidate_uav_ids": [],
                        "candidate_count": 0,
                        "executor_candidate_uav_ids": [],
                        "executor_candidate_count": 0,
                        "actor_selected_uav": None,
                        "executor_selected_uav": None,
                        "action_executed": False,
                        "fallback_used": False,
                        "failure_reason": "no_feasible_candidate",
                        "selected_uav_failure_reason": None,
                        "non_executed_reason": "no_feasible_candidate",
                    }
                )
                continue

            assignment_snapshot = self._graph_builder.build(
                self._task_manager,
                self._uavs,
                self._time_step,
                self._task_executor,
            )
            self._latest_assignment_graph_snapshot = assignment_snapshot
            rl_record = self._assignment_policy.act_for_task(
                assignment_snapshot,
                task.task_id,
                candidate_uav_ids,
                exploration=True,
            )
            action_executed, executor_record, failure_reason = (
                self._task_executor.commit_sequential_rl_assignment(
                    task,
                    self._task_manager,
                    self._uavs,
                    self._time_step,
                    int(rl_record["actor_selected_uav"]),
                    candidate_uav_ids,
                )
            )
            actor_selected_uav = int(rl_record["actor_selected_uav"])
            executor_selected_uav = executor_record.executor_selected_uav
            rl_record.update(
                {
                    "task_type": int(task.task_type),
                    "env_step_id": int(self._time_step),
                    "candidate_count": len(candidate_uav_ids),
                    "executor_candidate_uav_ids": [int(candidate.uav_id) for candidate in executor_record.candidates],
                    "executor_candidate_count": len(executor_record.candidates),
                    "executor_selected_uav": executor_selected_uav,
                    "executor_selection_mode": executor_record.selection_mode,
                    "action_executed": bool(action_executed),
                    "fallback_used": False,
                    "failure_reason": failure_reason,
                    "selected_uav_failure_reason": failure_reason,
                    "non_executed_reason": None if action_executed else "invalid_actor_action",
                }
            )
            executor_record.actor_selected_uav = actor_selected_uav
            executor_record.action_executed = bool(action_executed)
            self._latest_assignment_rl_records.append(rl_record)

    def _align_assignment_rl_records(self) -> None:
        executor_records = {
            record.task_id: record
            for record in self._task_executor.latest_assignment_records
        }
        for rl_record in self._latest_assignment_rl_records:
            task_id = str(rl_record["task_id"])
            executor_record = executor_records.get(task_id)
            task = self._task_manager.tasks.get(task_id)
            actor_selected_uav_raw = rl_record.get("actor_selected_uav")
            actor_selected_uav = None if actor_selected_uav_raw is None else int(actor_selected_uav_raw)
            executor_selected_uav = None if executor_record is None else executor_record.selected_uav
            action_executed = actor_selected_uav is not None and executor_selected_uav == actor_selected_uav
            executor_candidate_uav_ids = (
                []
                if executor_record is None
                else [int(candidate.uav_id) for candidate in executor_record.candidates]
            )
            fallback_used = (
                executor_record is not None
                and executor_record.selection_mode in {"fallback", "guard_fallback"}
            )
            selected_uav_failure_reason = (
                None
                if executor_record is None or actor_selected_uav is None
                else executor_record.rejected_uav_reasons.get(actor_selected_uav)
            )
            failure_reason = (
                "executor_record_missing"
                if executor_record is None
                else selected_uav_failure_reason
                if selected_uav_failure_reason is not None
                else "executor_no_feasible_candidate"
                if executor_selected_uav is None
                else "executor_selected_different_uav"
                if executor_selected_uav != actor_selected_uav
                else None
            )
            non_executed_reason = (
                None
                if action_executed
                else "no_actor_selected_uav"
                if actor_selected_uav is None
                else "no_executor_assignment"
                if executor_selected_uav is None
                else "fallback_after_actor"
                if fallback_used
                else "executor_override"
                if executor_selected_uav != actor_selected_uav
                else "selected_uav_later_infeasible"
                if selected_uav_failure_reason is not None
                else "unknown_non_executed"
            )
            rl_record["env_step_id"] = int(self._time_step)
            rl_record["executor_selected_uav"] = executor_selected_uav
            rl_record["action_executed"] = bool(action_executed)
            rl_record["task_type"] = None if task is None else int(task.task_type)
            rl_record["candidate_count"] = len(rl_record.get("candidate_uav_ids", []))
            rl_record["executor_candidate_count"] = len(executor_candidate_uav_ids)
            rl_record["executor_candidate_uav_ids"] = executor_candidate_uav_ids
            rl_record["executor_selection_mode"] = None if executor_record is None else executor_record.selection_mode
            rl_record["fallback_used"] = bool(fallback_used)
            rl_record["failure_reason"] = failure_reason
            rl_record["selected_uav_failure_reason"] = selected_uav_failure_reason
            rl_record["non_executed_reason"] = non_executed_reason
            if executor_record is not None:
                executor_record.actor_selected_uav = actor_selected_uav
                executor_record.executor_selected_uav = executor_selected_uav
                executor_record.action_executed = bool(action_executed)

    def _build_selective_score_snapshot(self, snapshot: HeteroGraphSnapshot) -> HeteroGraphSnapshot:
        if not config.USE_SELECTIVE_HGNN_SCORING:
            self._latest_selective_score_stats = {
                "selective_scoring_enabled": 0.0,
                "selective_ready_tasks": float(len(self._task_manager.get_ready_tasks())),
                "selective_high_risk_tasks": 0.0,
                "selective_normal_tasks": 0.0,
                "selective_high_risk_ratio": 0.0,
                "selective_score_edges": float(len(snapshot.task_uav_edges)),
            }
            return snapshot

        high_risk_task_ids = self._selective_high_risk_task_ids(snapshot)
        selected_indices = [
            edge_idx
            for edge_idx, (task_id, _) in enumerate(snapshot.task_uav_edges)
            if task_id in high_risk_task_ids
        ]
        selected_edges = [snapshot.task_uav_edges[edge_idx] for edge_idx in selected_indices]
        if selected_indices:
            selected_edge_features = snapshot.task_uav_edge_features[selected_indices]
        else:
            feature_dim = snapshot.task_uav_edge_features.shape[1] if snapshot.task_uav_edge_features.ndim == 2 else 0
            selected_edge_features = np.zeros((0, feature_dim), dtype=np.float32)

        ready_count = len(self._task_manager.get_ready_tasks())
        high_risk_count = len(high_risk_task_ids)
        self._latest_selective_score_stats = {
            "selective_scoring_enabled": 1.0,
            "selective_ready_tasks": float(ready_count),
            "selective_high_risk_tasks": float(high_risk_count),
            "selective_normal_tasks": float(max(ready_count - high_risk_count, 0)),
            "selective_high_risk_ratio": high_risk_count / float(max(ready_count, 1)),
            "selective_score_edges": float(len(selected_edges)),
        }
        return replace(
            snapshot,
            task_uav_edges=selected_edges,
            task_uav_edge_features=selected_edge_features,
        )

    def _build_per_task_score_provider(self):
        initial_snapshot = self._latest_graph_snapshot
        ready_count = len(self._task_manager.get_ready_tasks())
        if config.USE_SELECTIVE_HGNN_SCORING and initial_snapshot is not None:
            initial_high_risk_task_ids = self._selective_high_risk_task_ids(initial_snapshot)
            high_risk_count = len(initial_high_risk_task_ids)
            self._latest_selective_score_stats = {
                "selective_scoring_enabled": 1.0,
                "selective_ready_tasks": float(ready_count),
                "selective_high_risk_tasks": float(high_risk_count),
                "selective_normal_tasks": float(max(ready_count - high_risk_count, 0)),
                "selective_high_risk_ratio": high_risk_count / float(max(ready_count, 1)),
                "selective_score_edges": 0.0,
            }
        elif initial_snapshot is not None:
            self._latest_selective_score_stats = {
                "selective_scoring_enabled": 0.0,
                "selective_ready_tasks": float(ready_count),
                "selective_high_risk_tasks": 0.0,
                "selective_normal_tasks": 0.0,
                "selective_high_risk_ratio": 0.0,
                "selective_score_edges": 0.0,
            }

        def provide(task) -> tuple[set[tuple[str, int]] | None, dict[tuple[str, int], float] | None]:
            snapshot = self._graph_builder.build(
                self._task_manager,
                self._uavs,
                self._time_step,
                self._task_executor,
            )
            self._latest_assignment_graph_snapshot = snapshot
            allowed_edges = set(snapshot.task_uav_edges)
            if self._graph_scheduler is None:
                return allowed_edges, None

            if config.USE_SELECTIVE_HGNN_SCORING:
                high_risk_task_ids = self._selective_high_risk_task_ids(snapshot)
                if task.task_id not in high_risk_task_ids:
                    return allowed_edges, None
                selected_indices = [
                    edge_idx
                    for edge_idx, (task_id, _) in enumerate(snapshot.task_uav_edges)
                    if task_id == task.task_id
                ]
                if selected_indices:
                    selected_edge_features = snapshot.task_uav_edge_features[selected_indices]
                else:
                    feature_dim = snapshot.task_uav_edge_features.shape[1] if snapshot.task_uav_edge_features.ndim == 2 else 0
                    selected_edge_features = np.zeros((0, feature_dim), dtype=np.float32)
                score_snapshot = replace(
                    snapshot,
                    task_uav_edges=[snapshot.task_uav_edges[edge_idx] for edge_idx in selected_indices],
                    task_uav_edge_features=selected_edge_features,
                )
            else:
                selected_indices = [
                    edge_idx
                    for edge_idx, (task_id, _) in enumerate(snapshot.task_uav_edges)
                    if task_id == task.task_id
                ]
                if selected_indices:
                    selected_edge_features = snapshot.task_uav_edge_features[selected_indices]
                else:
                    feature_dim = snapshot.task_uav_edge_features.shape[1] if snapshot.task_uav_edge_features.ndim == 2 else 0
                    selected_edge_features = np.zeros((0, feature_dim), dtype=np.float32)
                score_snapshot = replace(
                    snapshot,
                    task_uav_edges=[snapshot.task_uav_edges[edge_idx] for edge_idx in selected_indices],
                    task_uav_edge_features=selected_edge_features,
                )

            self._latest_graph_scheduling_output = self._graph_scheduler.score_graph(score_snapshot)
            self._latest_selective_score_stats["selective_score_edges"] = (
                self._latest_selective_score_stats.get("selective_score_edges", 0.0)
                + float(len(score_snapshot.task_uav_edges))
            )
            return allowed_edges, self._latest_graph_scheduling_output.edge_scores

        return provide

    def _selective_high_risk_task_ids(self, snapshot: HeteroGraphSnapshot) -> set[str]:
        candidate_counts: dict[str, int] = {}
        for task_id, _ in snapshot.task_uav_edges:
            candidate_counts[task_id] = candidate_counts.get(task_id, 0) + 1

        high_risk_candidates: list[tuple[float, str]] = []
        for task in self._task_manager.get_ready_tasks():
            candidate_count = candidate_counts.get(task.task_id, 0)
            if candidate_count <= 0:
                continue
            task_slack = task.remaining_slack(self._time_step)
            dag_slack = self._task_manager.get_dag_remaining_slack(task.dag_id, self._time_step)
            dag_completion = self._task_manager.get_dag_completion_ratio(task.dag_id)
            context_slack_threshold = (
                config.SELECTIVE_HGNN_SLACK_THRESHOLD
                * config.SELECTIVE_HGNN_CONTEXT_SLACK_MULTIPLIER
            )
            critical_path_task = (
                config.SELECTIVE_HGNN_USE_CRITICAL_PATH
                and self._task_manager.is_critical_path_task(task.task_id)
            )
            successor_unlock_task = (
                config.SELECTIVE_HGNN_USE_SUCCESSOR_UNLOCK
                and len(task.successors) > 0
            )
            is_high_risk = (
                task_slack <= config.SELECTIVE_HGNN_SLACK_THRESHOLD
                or dag_slack <= config.SELECTIVE_HGNN_SLACK_THRESHOLD
            )
            if config.USE_STRICT_SELECTIVE_HGNN_SCORING:
                if is_high_risk:
                    risk_score = 0.0
                    risk_score += max(0.0, config.SELECTIVE_HGNN_SLACK_THRESHOLD - task_slack) * 4.0
                    risk_score += max(0.0, config.SELECTIVE_HGNN_SLACK_THRESHOLD - dag_slack) * 4.0
                    high_risk_candidates.append((risk_score, task.task_id))
                continue
            if config.SELECTIVE_HGNN_USE_CANDIDATE_SCARCITY:
                is_high_risk = is_high_risk or candidate_count <= config.SELECTIVE_HGNN_CANDIDATE_THRESHOLD
            if critical_path_task:
                is_high_risk = (
                    is_high_risk
                    or dag_slack <= context_slack_threshold
                    or dag_completion >= config.SELECTIVE_HGNN_COMPLETION_THRESHOLD
                )
            if successor_unlock_task:
                is_high_risk = is_high_risk or dag_slack <= context_slack_threshold
            if config.SELECTIVE_HGNN_USE_DAG_COMPLETION:
                is_high_risk = (
                    is_high_risk
                    or (
                        dag_completion >= config.SELECTIVE_HGNN_COMPLETION_THRESHOLD
                        and (critical_path_task or successor_unlock_task)
                    )
                )
            if is_high_risk:
                risk_score = 0.0
                risk_score += max(0.0, config.SELECTIVE_HGNN_SLACK_THRESHOLD - task_slack) * 4.0
                risk_score += max(0.0, config.SELECTIVE_HGNN_SLACK_THRESHOLD - dag_slack) * 4.0
                risk_score += max(0.0, context_slack_threshold - dag_slack)
                risk_score += 40.0 if critical_path_task else 0.0
                risk_score += 30.0 if candidate_count <= config.SELECTIVE_HGNN_CANDIDATE_THRESHOLD else 0.0
                risk_score += 20.0 if successor_unlock_task else 0.0
                risk_score += 20.0 * dag_completion
                risk_score += float(len(task.successors))
                high_risk_candidates.append((risk_score, task.task_id))

        max_tasks = max(int(config.SELECTIVE_HGNN_MAX_TASKS_PER_STEP), 0)
        high_risk_candidates.sort(reverse=True)
        if max_tasks > 0:
            high_risk_candidates = high_risk_candidates[:max_tasks]
        return {task_id for _, task_id in high_risk_candidates}

    def _get_obs(self) -> list[np.ndarray]:
        """
        构建每个UAV的局部观测向量（多智能体观测）
        返回：所有UAV的观测列表
        """
        # 1. 所有UE生成当前时间步的服务请求
        phase_one_mode = config.ENABLE_DYNAMIC_DAG and config.ENABLE_PHASE_ONE_EXECUTION and not config.ENABLE_LEGACY_REQUEST_PIPELINE
        for ue in self._ues:
            ue.assigned = False
            if not phase_one_mode:
                ue.generate_request()
        # 2. 将UE关联到覆盖它的UAV（最近UAV）
        self._associate_ues_to_uavs()
        # 3. 为每个UAV设置邻居无人机列表
        for uav in self._uavs:
            uav.set_neighbors(self._uavs)

        if config.ENABLE_DYNAMIC_DAG:
            self._task_manager.observe_time_step(self._ues, self._time_step)
            self._latest_graph_snapshot = self._graph_builder.build(
                self._task_manager,
                self._uavs,
                self._time_step,
                self._task_executor if config.ENABLE_PHASE_ONE_EXECUTION else None,
            )
            self._latest_graph_scheduling_output = None
            self._latest_selective_score_stats = {}
            if (
                not config.USE_RL_ASSIGNMENT
                and config.USE_HGNN_SCORE_ASSIGNMENT
                and self._graph_scheduler is not None
                and self._latest_graph_snapshot is not None
            ):
                score_snapshot = self._build_selective_score_snapshot(self._latest_graph_snapshot)
                self._latest_graph_scheduling_output = self._graph_scheduler.score_graph(score_snapshot)

        if phase_one_mode and config.USE_MAPPO_COMPACT_OBS:
            return self._get_mappo_compact_obs()

        # 存储所有UAV的观测
        all_obs: list[np.ndarray] = []

        # 逐个构建UAV观测
        for uav in self._uavs:
            # ==================== Part 1: 无人机自身状态 ====================
            # 位置归一化（除以区域宽高）
            own_pos: np.ndarray = np.clip(
                uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32),
                0.0,
                1.0,
            )
            if phase_one_mode and config.USE_PHASE_ONE_DEDICATED_OBS:
                own_state = np.array(
                    [
                        own_pos[0],
                        own_pos[1],
                        min(self._task_executor.get_queue_length(uav.id) / float(max(config.DAG_MAX_QUEUE_PER_UAV, 1)), 1.0),
                        1.0 if self._task_executor.is_uav_busy(uav.id) else 0.0,
                        uav.remaining_energy_ratio,
                        len(uav.neighbors) / float(max(config.NUM_UAVS - 1, 1)),
                        config.UAV_COMPUTING_CAPACITY[uav.id] / float(np.max(config.UAV_COMPUTING_CAPACITY)),
                    ],
                    dtype=np.float32,
                )
            elif phase_one_mode:
                own_cache = np.zeros(config.NUM_FILES, dtype=np.float32)
                own_cache[0] = min(self._task_executor.get_queue_length(uav.id) / float(max(config.DAG_MAX_QUEUE_PER_UAV, 1)), 1.0)
                own_cache[1] = 1.0 if self._task_executor.is_uav_busy(uav.id) else 0.0
                own_cache[2] = uav.remaining_energy_ratio
                own_cache[3] = len(uav.neighbors) / float(max(config.NUM_UAVS - 1, 1))
                own_cache[4] = config.UAV_COMPUTING_CAPACITY[uav.id] / float(np.max(config.UAV_COMPUTING_CAPACITY))
                own_state = np.concatenate([own_pos, own_cache])
            else:
                # 缓存状态
                own_cache = uav.cache.astype(np.float32)
                own_state = np.concatenate([own_pos, own_cache])

            # ==================== Part 2: 邻居无人机状态 ====================
            # 初始化邻居状态矩阵（最大邻居数 × 邻居观测维度）
            neighbor_obs_dim = config.PHASE_ONE_NEIGHBOR_OBS_DIM if phase_one_mode and config.USE_PHASE_ONE_DEDICATED_OBS else config.LEGACY_NEIGHBOR_OBS_DIM
            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, neighbor_obs_dim), dtype=np.float32)
            # 按距离排序邻居，取前K个
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[: config.MAX_UAV_NEIGHBORS]
            # 逐个填入邻居相对位置
            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = np.clip((neighbor.pos[:2] - uav.pos[:2]) / config.UAV_SENSING_RANGE, -1.0, 1.0)
                if phase_one_mode and config.USE_PHASE_ONE_DEDICATED_OBS:
                    neighbor_states[i, :] = np.array(
                        [
                            relative_pos[0],
                            relative_pos[1],
                            min(self._task_executor.get_queue_length(neighbor.id) / float(max(config.DAG_MAX_QUEUE_PER_UAV, 1)), 1.0),
                            1.0 if self._task_executor.is_uav_busy(neighbor.id) else 0.0,
                        ],
                        dtype=np.float32,
                    )
                else:
                    neighbor_states[i, :] = relative_pos

            # ==================== Part 3: 关联UE / 阶段一任务摘要 ====================
            # 初始化UE状态矩阵
            ue_obs_dim = config.PHASE_ONE_TASK_OBS_DIM if phase_one_mode and config.USE_PHASE_ONE_DEDICATED_OBS else config.LEGACY_UE_OBS_DIM
            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, ue_obs_dim), dtype=np.float32)
            if phase_one_mode:
                task_summaries = self._get_phase_one_task_summaries(uav.id)
                for i, task in enumerate(task_summaries[: config.MAX_ASSOCIATED_UES]):
                    delta_pos: np.ndarray = np.clip((task.source_pos - uav.pos[:2]) / config.AREA_WIDTH, -1.0, 1.0)
                    state_norm = {
                        "ready": 0.0,
                        "queued": 1.0 / 3.0,
                        "running": 2.0 / 3.0,
                        "waiting": 1.0,
                    }.get(task.state, 1.0)
                    task_info = np.array(
                        [
                            np.clip(task.remaining_slack(self._time_step) / float(max(config.DAG_MAX_DEADLINE_OFFSET, 1)), -1.0, 1.0),
                            np.clip(task.input_size / float(max(config.DAG_MAX_INPUT_SIZE, 1)), 0.0, 1.0),
                            np.clip(task.level / float(max(config.DAG_MAX_TASK_LEVELS - 1, 1)), 0.0, 1.0),
                            state_norm,
                        ],
                        dtype=np.float32,
                    )
                    ue_states[i, :] = np.concatenate([delta_pos, task_info])
            else:
                # 按距离排序关联的UE，取前M个
                ues: list[UE] = sorted(
                    uav.current_covered_ues,
                    key=lambda u: float(np.linalg.norm(uav.pos[:2] - u.pos[:2])),
                )[: config.MAX_ASSOCIATED_UES]

                # 填入UE信息：相对位置、请求类型、大小、ID、电池等
                for i, ue in enumerate(ues):
                    delta_pos: np.ndarray = (ue.pos[:2] - uav.pos[:2]) / config.AREA_WIDTH
                    req_type, req_size, req_id = ue.current_request
                    norm_type: float = float(req_type) / 2.0
                    norm_id: float = float(req_id) / float(config.NUM_FILES)
                    norm_size: float = float(req_size) / float(config.MAX_INPUT_SIZE)
                    norm_battery: float = ue.battery_level / config.UE_BATTERY_CAPACITY
                    request_info: np.ndarray = np.array([norm_type, norm_size, norm_id, norm_battery], dtype=np.float32)
                    ue_states[i, :] = np.concatenate([delta_pos, request_info])

            # ==================== 拼接成最终观测 ====================
            obs: np.ndarray = np.concatenate([own_state, neighbor_states.flatten(), ue_states.flatten()])
            all_obs.append(obs)

        return all_obs

    def _get_mappo_compact_obs(self) -> list[np.ndarray]:
        """Builds a compact phase-one observation for MAPPO UAV movement control."""
        all_obs: list[np.ndarray] = []
        edge_counts_by_uav: dict[int, int] = {uav.id: 0 for uav in self._uavs}
        candidate_counts_by_task: dict[str, int] = {}
        if self._latest_graph_snapshot is not None:
            for task_id, uav_id in self._latest_graph_snapshot.task_uav_edges:
                edge_counts_by_uav[uav_id] = edge_counts_by_uav.get(uav_id, 0) + 1
                candidate_counts_by_task[task_id] = candidate_counts_by_task.get(task_id, 0) + 1
        score_values_by_uav: dict[int, list[float]] = {uav.id: [] for uav in self._uavs}
        if self._latest_graph_scheduling_output is not None:
            for (_, uav_id), score in self._latest_graph_scheduling_output.edge_scores.items():
                score_values_by_uav.setdefault(uav_id, []).append(float(np.tanh(score)))

        service_participation_by_uav: dict[int, float] = {uav.id: 0.0 for uav in self._uavs}
        resource_participation_by_uav: dict[int, float] = {uav.id: 0.0 for uav in self._uavs}
        critical_participation_by_uav: dict[int, float] = {uav.id: 0.0 for uav in self._uavs}
        feasible_tasks_by_uav: dict[int, set[str]] = {uav.id: set() for uav in self._uavs}
        if self._latest_graph_snapshot is not None:
            for task_id, uav_id in self._latest_graph_snapshot.task_uav_edges:
                feasible_tasks_by_uav.setdefault(uav_id, set()).add(task_id)
            service_total = float(max(len(self._latest_graph_snapshot.service_domain_hyperedges), 1))
            resource_total = float(max(len(self._latest_graph_snapshot.resource_competition_hyperedges), 1))
            critical_total = float(
                max(
                    len(self._latest_graph_snapshot.critical_hyperedges)
                    + len(self._latest_graph_snapshot.critical_support_hyperedges),
                    1,
                )
            )
            for _, uav_ids in self._latest_graph_snapshot.service_domain_hyperedges:
                for uav_id in uav_ids:
                    service_participation_by_uav[uav_id] = service_participation_by_uav.get(uav_id, 0.0) + 1.0 / service_total
            for _, uav_ids in self._latest_graph_snapshot.resource_competition_hyperedges:
                for uav_id in uav_ids:
                    resource_participation_by_uav[uav_id] = resource_participation_by_uav.get(uav_id, 0.0) + 1.0 / resource_total
            for task_ids in self._latest_graph_snapshot.critical_hyperedges:
                critical_task_set = set(task_ids)
                for uav_id, feasible_task_ids in feasible_tasks_by_uav.items():
                    if critical_task_set & feasible_task_ids:
                        critical_participation_by_uav[uav_id] = critical_participation_by_uav.get(uav_id, 0.0) + 1.0 / critical_total
            for task_ids, uav_ids in self._latest_graph_snapshot.critical_support_hyperedges:
                critical_task_set = set(task_ids)
                for uav_id in uav_ids:
                    if uav_id in feasible_tasks_by_uav and critical_task_set & feasible_tasks_by_uav[uav_id]:
                        critical_participation_by_uav[uav_id] = critical_participation_by_uav.get(uav_id, 0.0) + 1.0 / critical_total

        max_capacity = float(max(np.max(config.UAV_COMPUTING_CAPACITY), 1))
        max_queue = float(max(config.DAG_MAX_QUEUE_PER_UAV, 1))
        max_ready_ref = float(max(config.MAX_ASSOCIATED_UES, 1))
        max_slack = float(max(config.DAG_MAX_DEADLINE_OFFSET, 1))

        for uav in self._uavs:
            own_pos = np.clip(
                uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32),
                0.0,
                1.0,
            )
            own_state = np.array(
                [
                    own_pos[0],
                    own_pos[1],
                    min(self._task_executor.get_queue_length(uav.id) / max_queue, 1.0),
                    1.0 if self._task_executor.is_uav_busy(uav.id) else 0.0,
                    uav.remaining_energy_ratio,
                    len(uav.neighbors) / float(max(config.NUM_UAVS - 1, 1)),
                    config.UAV_COMPUTING_CAPACITY[uav.id] / max_capacity,
                ],
                dtype=np.float32,
            )

            neighbor_states = np.zeros(
                (config.MAX_UAV_NEIGHBORS, config.PHASE_ONE_NEIGHBOR_OBS_DIM),
                dtype=np.float32,
            )
            neighbors = sorted(
                uav.neighbors,
                key=lambda n: float(np.linalg.norm(uav.pos - n.pos)),
            )[: config.MAX_UAV_NEIGHBORS]
            for i, neighbor in enumerate(neighbors):
                relative_pos = np.clip((neighbor.pos[:2] - uav.pos[:2]) / config.UAV_SENSING_RANGE, -1.0, 1.0)
                neighbor_states[i, :] = np.array(
                    [
                        relative_pos[0],
                        relative_pos[1],
                        min(self._task_executor.get_queue_length(neighbor.id) / max_queue, 1.0),
                        1.0 if self._task_executor.is_uav_busy(neighbor.id) else 0.0,
                    ],
                    dtype=np.float32,
                )

            ready_local = []
            active_local = []
            assigned_local = []
            for task in self._task_manager.get_active_tasks():
                distance = float(np.linalg.norm(task.source_pos - uav.pos[:2]))
                if distance <= config.DAG_TASK_UAV_MAX_DISTANCE:
                    active_local.append(task)
                    if task.is_ready:
                        ready_local.append(task)
                if task.assigned_uav == uav.id and task.state in {TASK_STATE_QUEUED, TASK_STATE_RUNNING}:
                    assigned_local.append(task)

            if ready_local:
                slacks = np.array([task.remaining_slack(self._time_step) for task in ready_local], dtype=np.float32)
                avg_slack = float(np.clip(np.mean(slacks) / max_slack, -1.0, 1.0))
                min_slack = float(np.clip(np.min(slacks) / max_slack, -1.0, 1.0))
                avg_input = float(np.clip(np.mean([task.input_size for task in ready_local]) / float(max(config.DAG_MAX_INPUT_SIZE, 1)), 0.0, 1.0))
                avg_output = float(np.clip(np.mean([task.output_size for task in ready_local]) / float(max(config.DAG_MAX_OUTPUT_SIZE, 1)), 0.0, 1.0))
                avg_cpu = float(np.clip(np.mean([task.cpu_cycles for task in ready_local]) / float(max(config.DAG_MAX_CPU_CYCLES, 1)), 0.0, 1.0))
                urgent_ratio = float(np.mean(slacks <= config.DAG_CRITICAL_SLACK_THRESHOLD))
                compute_heavy_ratio = float(
                    np.mean([task.task_type == config.TASK_TYPE_COMPUTE for task in ready_local])
                )
                scarce_ratio = float(
                    np.mean([candidate_counts_by_task.get(task.task_id, 0) <= 2 for task in ready_local])
                )
            else:
                avg_slack = 0.0
                min_slack = 0.0
                avg_input = 0.0
                avg_output = 0.0
                avg_cpu = 0.0
                urgent_ratio = 0.0
                compute_heavy_ratio = 0.0
                scarce_ratio = 0.0

            has_critical_task = float(
                any(
                    task.remaining_slack(self._time_step) <= config.DAG_CRITICAL_SLACK_THRESHOLD
                    for task in assigned_local
                )
            )
            score_values = score_values_by_uav.get(uav.id, [])
            if score_values:
                score_array = np.array(score_values, dtype=np.float32)
                score_mean = float(np.clip(np.mean(score_array), -1.0, 1.0))
                score_max = float(np.clip(np.max(score_array), -1.0, 1.0))
                score_std = float(np.clip(np.std(score_array), 0.0, 1.0))
            else:
                score_mean = 0.0
                score_max = 0.0
                score_std = 0.0
            assigned_scores = []
            if self._latest_graph_scheduling_output is not None:
                for task in assigned_local:
                    score = self._latest_graph_scheduling_output.edge_scores.get((task.task_id, uav.id))
                    if score is not None:
                        assigned_scores.append(float(np.tanh(score)))
            if assigned_scores:
                assigned_score_array = np.array(assigned_scores, dtype=np.float32)
                assigned_score_mean = float(np.clip(np.mean(assigned_score_array), -1.0, 1.0))
                assigned_score_max = float(np.clip(np.max(assigned_score_array), -1.0, 1.0))
            else:
                assigned_score_mean = 0.0
                assigned_score_max = 0.0

            covered_ue_count = sum(
                1
                for ue in self._ues
                if float(np.linalg.norm(ue.pos[:2] - uav.pos[:2])) <= config.UAV_COVERAGE_RADIUS
            )
            local_summary = np.array(
                [
                    min(len(ready_local) / max_ready_ref, 1.0),
                    min(len(active_local) / max_ready_ref, 1.0),
                    min(len(assigned_local) / max_queue, 1.0),
                    has_critical_task,
                    min(edge_counts_by_uav.get(uav.id, 0) / max_ready_ref, 1.0),
                    score_mean,
                    score_max,
                    score_std,
                    assigned_score_mean,
                    assigned_score_max,
                    float(np.clip(service_participation_by_uav.get(uav.id, 0.0), 0.0, 1.0)),
                    float(np.clip(resource_participation_by_uav.get(uav.id, 0.0), 0.0, 1.0)),
                    float(np.clip(critical_participation_by_uav.get(uav.id, 0.0), 0.0, 1.0)),
                    avg_slack,
                    min_slack,
                    avg_input,
                    avg_output,
                    avg_cpu,
                    urgent_ratio,
                    compute_heavy_ratio,
                    scarce_ratio,
                    min(covered_ue_count / float(max(config.NUM_UES, 1)), 1.0),
                ],
                dtype=np.float32,
            )
            obs = np.concatenate([own_state, neighbor_states.flatten(), local_summary])
            all_obs.append(obs.astype(np.float32, copy=False))
        return all_obs

    def _get_phase_one_task_summaries(self, uav_id: int) -> list:
        relevant_states = {"queued", "running"}
        tasks = [
            task
            for task in self._task_manager.get_active_tasks()
            if task.assigned_uav == uav_id or (task.is_ready and float(np.linalg.norm(task.source_pos - self._uavs[uav_id].pos[:2])) <= config.DAG_TASK_UAV_MAX_DISTANCE)
        ]
        tasks.sort(
            key=lambda task: (
                0 if task.state in relevant_states else 1,
                task.deadline,
                task.level,
            )
        )
        return tasks

    def _apply_actions_to_env(self, actions: np.ndarray) -> None:
        """
        执行智能体动作：计算无人机新位置 + 避撞 + 越界保护
        """
        # 当前所有无人机位置
        current_positions: np.ndarray = np.array([uav.pos[:2] for uav in self._uavs], dtype=np.float32)
        # 单步最大移动距离 = 速度 × 时间片
        max_dist: float = config.UAV_SPEED * config.TIME_SLOT_DURATION

        # 动作解析为 (x,y) 移动向量
        delta_vec_raw: np.ndarray = np.array(actions, dtype=np.float32)

        # 计算向量模长（距离）
        raw_magnitude: np.ndarray = np.linalg.norm(delta_vec_raw, axis=1, keepdims=True)
        # 限制最大长度为1（归一化）
        clipped_magnitude: np.ndarray = np.minimum(raw_magnitude, 1.0)
        # 实际可移动距离
        distances: np.ndarray = clipped_magnitude * max_dist
        # 防止除0
        denom: np.ndarray = raw_magnitude + float(config.EPSILON)
        # 移动方向
        directions: np.ndarray = delta_vec_raw / denom
        # 最终位移
        delta_pos: np.ndarray = directions * distances

        # 计算提议位置
        proposed_positions: np.ndarray = current_positions + delta_pos

        # 边界保护：距离边界至少保留半个覆盖半径
        min_boundary_gap: float = config.UAV_COVERAGE_RADIUS / 2.0
        for i, uav in enumerate(self._uavs):
            # 检查是否越界
            if not (min_boundary_gap <= proposed_positions[i, 0] <= config.AREA_WIDTH - min_boundary_gap and
                    min_boundary_gap <= proposed_positions[i, 1] <= config.AREA_HEIGHT - min_boundary_gap):
                uav.boundary_violation = True
        # 裁剪到安全区域内
        next_positions: np.ndarray = np.clip(proposed_positions, [min_boundary_gap, min_boundary_gap],
                                             [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap])

        # ==================== 碰撞避免迭代 ====================
        min_sep_sq: float = config.MIN_UAV_SEPARATION**2
        for _ in range(config.COLLISION_AVOIDANCE_ITERATIONS + 1):
            collision_detected_in_iter: bool = False
            # 遍历所有无人机对
            for i in range(config.NUM_UAVS):
                for j in range(i + 1, config.NUM_UAVS):
                    pos_i: np.ndarray = next_positions[i]
                    pos_j: np.ndarray = next_positions[j]
                    dist_sq: float = np.sum((pos_i - pos_j) ** 2)
                    # 距离小于最小安全距离
                    if dist_sq < min_sep_sq:
                        # 标记碰撞违规
                        self._uavs[i].collision_violation = True
                        self._uavs[j].collision_violation = True
                        collision_detected_in_iter = True
                        # 计算分离向量
                        dist: float = np.sqrt(dist_sq) if dist_sq > 0 else config.EPSILON
                        overlap: float = config.MIN_UAV_SEPARATION - dist
                        direction: np.ndarray = (pos_i - pos_j) / dist
                        # 相互推开
                        next_positions[i] += direction * overlap * 0.5
                        next_positions[j] -= direction * overlap * 0.5
            # 无碰撞则提前退出
            if not collision_detected_in_iter:
                break

        # 最终位置再次边界裁剪
        final_positions: np.ndarray = np.clip(next_positions, [min_boundary_gap, min_boundary_gap],
                                              [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap])
        # 更新无人机位置
        for i, uav in enumerate(self._uavs):
            uav.update_position(final_positions[i])

    def _associate_ues_to_uavs(self) -> None:
        """
        UE关联规则：每个UE关联到距离最近、且在覆盖半径内的UAV
        实现：覆盖重叠区选择信号最好（最近）的UAV
        """
        for ue in self._ues:
            covering_uavs: list[tuple[UAV, float]] = []
            # 遍历所有UAV
            for uav in self._uavs:
                distance: float = float(np.linalg.norm(uav.pos[:2] - ue.pos[:2]))
                # 在覆盖半径内
                if distance <= config.UAV_COVERAGE_RADIUS:
                    covering_uavs.append((uav, distance))

            # 无UAV覆盖则跳过
            if not covering_uavs:
                continue
            # 选择距离最近的UAV
            best_uav, _ = min(covering_uavs, key=lambda x: x[1])
            # UE加入该UAV的服务列表
            best_uav.current_covered_ues.append(ue)
            ue.assigned = True

    def _get_rewards_and_metrics(self) -> tuple[list[float], tuple[float, float, float, float]]:
        """
        计算奖励函数与系统指标
        返回：每个UAV奖励、系统指标(时延,能耗,公平性,离线率)
        """
        # 总时延：已服务UE使用当前时延，未服务使用惩罚值
        total_latency: float = sum(ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY for ue in self._ues)
        # 总能耗：所有UAV能耗之和
        total_energy: float = sum(uav.energy for uav in self._uavs)
        # 服务覆盖时长指标
        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues], dtype=np.float32)
        # 计算Jain公平性指数
        jfi: float = 0.0
        if sc_metrics.size > 0 and np.sum(sc_metrics**2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics**2))
        # 低电量关机UE数量
        offline_count: int = sum(1 for ue in self._ues if ue.battery_level < config.UE_CRITICAL_THRESHOLD)
        # 离线率
        offline_rate: float = offline_count / config.NUM_UES

        # ==================== 奖励组成 ====================
        # 公平性奖励
        r_fairness: float = config.ALPHA_3 * np.log(jfi + config.EPSILON)
        # 时延惩罚
        r_latency: float = config.ALPHA_1 * np.log(total_latency + config.EPSILON)
        # 能耗惩罚
        r_energy: float = config.ALPHA_2 * np.log(total_energy + config.EPSILON)
        # 离线惩罚
        r_offline: float = config.ALPHA_4 * np.log(1.0 + offline_rate)

        # 总奖励 = 公平性 - 时延 - 能耗 - 离线
        reward: float = r_fairness - r_latency - r_energy - r_offline
        # 所有UAV共享全局奖励
        rewards: list[float] = [reward] * config.NUM_UAVS

        # 碰撞惩罚
        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY
            # 越界惩罚
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY

        # 奖励缩放
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards]
        # 返回奖励与系统指标
        return rewards, (total_latency, total_energy, jfi, offline_rate)

    def _get_phase_one_rewards_and_metrics(self) -> tuple[list[float], tuple[float, float, float, float]]:
        stats = self._task_executor.latest_stats
        summary = self._task_executor.get_summary()
        job_summary = self._task_manager.get_job_summary()
        new_successful_jobs = max(
            job_summary["dag_successful_jobs"] - self._last_phase_one_job_counts["dag_successful_jobs"],
            0.0,
        )
        new_on_time_successful_jobs = max(
            job_summary["dag_on_time_successful_jobs"]
            - self._last_phase_one_job_counts["dag_on_time_successful_jobs"],
            0.0,
        )
        new_failed_jobs = max(
            job_summary["dag_failed_jobs"] - self._last_phase_one_job_counts["dag_failed_jobs"],
            0.0,
        )
        self._last_phase_one_job_counts = {
            "dag_successful_jobs": job_summary["dag_successful_jobs"],
            "dag_on_time_successful_jobs": job_summary["dag_on_time_successful_jobs"],
            "dag_failed_jobs": job_summary["dag_failed_jobs"],
        }
        total_energy = sum(uav.energy for uav in self._uavs)
        task_finish_reward = config.PHASE_ONE_FINISH_REWARD * stats.on_time_completed_tasks
        task_late_penalty = config.PHASE_ONE_DEADLINE_PENALTY * max(stats.completed_tasks - stats.on_time_completed_tasks, 0)
        energy_penalty = config.PHASE_ONE_ENERGY_PENALTY * total_energy
        invalid_penalty = config.PHASE_ONE_INVALID_PENALTY * stats.invalid_actions
        dag_success_reward = 0.0
        dag_on_time_bonus = 0.0
        dag_failure_penalty = 0.0
        if config.USE_PHASE_ONE_DAG_REWARD_SHAPING:
            dag_success_reward = config.PHASE_ONE_DAG_SUCCESS_REWARD * new_successful_jobs
            dag_on_time_bonus = config.PHASE_ONE_DAG_ON_TIME_BONUS * new_on_time_successful_jobs
            dag_failure_penalty = config.PHASE_ONE_DAG_FAILURE_PENALTY * new_failed_jobs
        reward = (
            task_finish_reward
            - task_late_penalty
            - energy_penalty
            - invalid_penalty
            + dag_success_reward
            + dag_on_time_bonus
            - dag_failure_penalty
        )
        self._latest_phase_one_reward_terms = {
            "task_finish_reward": float(task_finish_reward),
            "task_late_penalty": float(task_late_penalty),
            "energy_penalty": float(energy_penalty),
            "invalid_penalty": float(invalid_penalty),
            "dag_success_reward": float(dag_success_reward),
            "dag_on_time_bonus": float(dag_on_time_bonus),
            "dag_failure_penalty": float(dag_failure_penalty),
            "new_successful_jobs": float(new_successful_jobs),
            "new_on_time_successful_jobs": float(new_on_time_successful_jobs),
            "new_failed_jobs": float(new_failed_jobs),
        }
        rewards = [reward] * config.NUM_UAVS
        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards]
        avg_delay = stats.step_delay / max(stats.completed_tasks, 1)
        on_time_ratio = summary["on_time_ratio"]
        deadline_violation_rate = summary["deadline_violations"] / max(summary["finished_count"], 1.0)
        invalid_rate = stats.invalid_actions / max(len(self._task_manager.get_ready_tasks()) + stats.newly_assigned_tasks, 1)
        return rewards, (avg_delay, total_energy, on_time_ratio, invalid_rate + deadline_violation_rate)

    def _get_assigned_task_distances_by_uav(self) -> dict[int, float]:
        distances: dict[int, float] = {}
        active_tasks = self._task_manager.get_active_tasks()
        for uav in self._uavs:
            assigned_distances = [
                float(np.linalg.norm(task.source_pos - uav.pos[:2]))
                for task in active_tasks
                if task.assigned_uav == uav.id and task.state in {TASK_STATE_QUEUED, TASK_STATE_RUNNING}
            ]
            if assigned_distances:
                distances[uav.id] = float(np.mean(assigned_distances))
        return distances

    def _get_stage_b_coverages_by_uav(self) -> dict[int, tuple[float, float]]:
        coverages: dict[int, tuple[float, float]] = {}
        max_ready_ref = float(max(config.MAX_ASSOCIATED_UES, 1))
        max_queue_ref = float(max(config.DAG_MAX_QUEUE_PER_UAV, 1))
        active_tasks = self._task_manager.get_active_tasks()
        for uav in self._uavs:
            local_ready_count = 0
            local_assigned_count = 0
            for task in active_tasks:
                distance = float(np.linalg.norm(task.source_pos - uav.pos[:2]))
                if task.is_ready and distance <= config.DAG_TASK_UAV_MAX_DISTANCE:
                    local_ready_count += 1
                if (
                    task.assigned_uav == uav.id
                    and task.state in {TASK_STATE_QUEUED, TASK_STATE_RUNNING}
                    and distance <= config.DAG_TASK_UAV_MAX_DISTANCE
                ):
                    local_assigned_count += 1
            coverages[uav.id] = (
                min(local_ready_count / max_ready_ref, 1.0),
                min(local_assigned_count / max_queue_ref, 1.0),
            )
        return coverages

    def _get_stage_b_movement_rewards(
        self,
        assigned_distances_before_action: dict[int, float] | None = None,
        coverages_before_action: dict[int, tuple[float, float]] | None = None,
    ) -> list[float]:
        """Stage B MAPPO reward for UAV movement control only.

        The global DAG/task assignment reward is still logged, but the actor
        receives local signals that the current movement can directly affect.
        """
        edge_counts_by_uav: dict[int, int] = {uav.id: 0 for uav in self._uavs}
        if self._latest_graph_snapshot is not None:
            for _, uav_id in self._latest_graph_snapshot.task_uav_edges:
                edge_counts_by_uav[uav_id] = edge_counts_by_uav.get(uav_id, 0) + 1

        max_ready_ref = float(max(config.MAX_ASSOCIATED_UES, 1))
        max_queue_ref = float(max(config.DAG_MAX_QUEUE_PER_UAV, 1))
        max_edge_ref = float(max(config.MAX_ASSOCIATED_UES, 1))
        max_move_energy = float(max(config.POWER_MOVE * config.TIME_SLOT_DURATION, config.EPSILON))
        rewards: list[float] = []
        ready_coverages: list[float] = []
        assigned_coverages: list[float] = []
        ready_coverage_deltas: list[float] = []
        assigned_coverage_deltas: list[float] = []
        assigned_distance_rewards: list[float] = []
        feasible_edge_scores: list[float] = []
        progress_rewards: list[float] = []
        local_finish_rewards: list[float] = []
        local_on_time_finish_rewards: list[float] = []
        move_penalties: list[float] = []
        collision_penalties: list[float] = []
        boundary_penalties: list[float] = []

        active_tasks = self._task_manager.get_active_tasks()
        stats = self._task_executor.latest_stats
        assigned_distances_after_action = self._get_assigned_task_distances_by_uav()
        new_successful_jobs = float(self._latest_phase_one_reward_terms.get("new_successful_jobs", 0.0))
        new_failed_jobs = float(self._latest_phase_one_reward_terms.get("new_failed_jobs", 0.0))
        job_summary = self._task_manager.get_job_summary()
        active_job_denominator = max(
            float(job_summary.get("dag_incomplete_jobs", 0.0)) + new_successful_jobs + new_failed_jobs,
            1.0,
        )
        dag_success_delta = float(np.clip(new_successful_jobs / active_job_denominator, 0.0, 1.0))
        dag_failure_delta = float(np.clip(new_failed_jobs / active_job_denominator, 0.0, 1.0))
        dag_shaping_reward = (
            config.STAGE_B_DAG_SUCCESS_DELTA_REWARD * dag_success_delta
            - config.STAGE_B_DAG_FAILURE_DELTA_PENALTY * dag_failure_delta
        )
        for uav in self._uavs:
            local_ready_count = 0
            local_assigned_count = 0
            for task in active_tasks:
                distance = float(np.linalg.norm(task.source_pos - uav.pos[:2]))
                if task.is_ready and distance <= config.DAG_TASK_UAV_MAX_DISTANCE:
                    local_ready_count += 1
                if (
                    task.assigned_uav == uav.id
                    and task.state in {TASK_STATE_QUEUED, TASK_STATE_RUNNING}
                    and distance <= config.DAG_TASK_UAV_MAX_DISTANCE
                ):
                    local_assigned_count += 1

            ready_coverage = min(local_ready_count / max_ready_ref, 1.0)
            assigned_coverage = min(local_assigned_count / max_queue_ref, 1.0)
            ready_coverage_delta = 0.0
            assigned_coverage_delta = 0.0
            if coverages_before_action is not None and uav.id in coverages_before_action:
                ready_before, assigned_before = coverages_before_action[uav.id]
                ready_coverage_delta = float(np.clip(ready_coverage - ready_before, -1.0, 1.0))
                assigned_coverage_delta = float(np.clip(assigned_coverage - assigned_before, -1.0, 1.0))
            assigned_distance_reward = 0.0
            if assigned_distances_before_action is not None and uav.id in assigned_distances_before_action:
                after_distance = assigned_distances_after_action.get(
                    uav.id,
                    assigned_distances_before_action[uav.id],
                )
                distance_improvement = assigned_distances_before_action[uav.id] - after_distance
                assigned_distance_reward = float(
                    np.clip(distance_improvement / max(config.DAG_TASK_UAV_MAX_DISTANCE, config.EPSILON), -1.0, 1.0)
                )
            feasible_edge_score = min(edge_counts_by_uav.get(uav.id, 0) / max_edge_ref, 1.0)
            progress_reward = min(float(stats.progress_by_uav.get(uav.id, 0.0)), 1.0)
            local_finish_reward = float(stats.completed_tasks_by_uav.get(uav.id, 0))
            local_on_time_finish_reward = float(stats.on_time_completed_tasks_by_uav.get(uav.id, 0))
            time_moving = float(getattr(uav, "_dist_moved", 0.0)) / float(max(config.UAV_SPEED, config.EPSILON))
            move_energy = config.POWER_MOVE * min(time_moving, config.TIME_SLOT_DURATION)
            move_penalty = min(move_energy / max_move_energy, 1.0)
            collision_penalty = 1.0 if uav.collision_violation else 0.0
            boundary_penalty = 1.0 if uav.boundary_violation else 0.0
            reward = (
                config.STAGE_B_READY_COVERAGE_REWARD * ready_coverage
                + config.STAGE_B_ASSIGNED_COVERAGE_REWARD * assigned_coverage
                + config.STAGE_B_READY_COVERAGE_DELTA_REWARD * ready_coverage_delta
                + config.STAGE_B_ASSIGNED_COVERAGE_DELTA_REWARD * assigned_coverage_delta
                + config.STAGE_B_ASSIGNED_DISTANCE_REWARD * assigned_distance_reward
                + config.STAGE_B_FEASIBLE_EDGE_REWARD * feasible_edge_score
                + config.STAGE_B_PROGRESS_REWARD * progress_reward
                + config.STAGE_B_LOCAL_FINISH_REWARD * local_finish_reward
                + config.STAGE_B_LOCAL_ON_TIME_FINISH_REWARD * local_on_time_finish_reward
                + dag_shaping_reward
                - config.STAGE_B_MOVE_ENERGY_PENALTY * move_penalty
                - config.STAGE_B_COLLISION_PENALTY * collision_penalty
                - config.STAGE_B_BOUNDARY_PENALTY * boundary_penalty
            )
            rewards.append(float(reward))
            ready_coverages.append(float(ready_coverage))
            assigned_coverages.append(float(assigned_coverage))
            ready_coverage_deltas.append(float(ready_coverage_delta))
            assigned_coverage_deltas.append(float(assigned_coverage_delta))
            assigned_distance_rewards.append(float(assigned_distance_reward))
            feasible_edge_scores.append(float(feasible_edge_score))
            progress_rewards.append(float(progress_reward))
            local_finish_rewards.append(float(local_finish_reward))
            local_on_time_finish_rewards.append(float(local_on_time_finish_reward))
            move_penalties.append(float(move_penalty))
            collision_penalties.append(float(collision_penalty))
            boundary_penalties.append(float(boundary_penalty))

        self._latest_phase_one_reward_terms.update(
            {
                "stage_b_reward_enabled": 1.0,
                "stage_b_ready_coverage": float(np.mean(ready_coverages)) if ready_coverages else 0.0,
                "stage_b_assigned_coverage": float(np.mean(assigned_coverages)) if assigned_coverages else 0.0,
                "stage_b_ready_coverage_delta": float(np.mean(ready_coverage_deltas)) if ready_coverage_deltas else 0.0,
                "stage_b_assigned_coverage_delta": float(np.mean(assigned_coverage_deltas)) if assigned_coverage_deltas else 0.0,
                "stage_b_assigned_distance_reward": float(np.mean(assigned_distance_rewards)) if assigned_distance_rewards else 0.0,
                "stage_b_feasible_edge_score": float(np.mean(feasible_edge_scores)) if feasible_edge_scores else 0.0,
                "stage_b_progress_reward": float(np.mean(progress_rewards)) if progress_rewards else 0.0,
                "stage_b_local_finish_reward": float(np.mean(local_finish_rewards)) if local_finish_rewards else 0.0,
                "stage_b_local_on_time_finish_reward": float(np.mean(local_on_time_finish_rewards)) if local_on_time_finish_rewards else 0.0,
                "stage_b_dag_success_delta": dag_success_delta,
                "stage_b_dag_failure_delta": dag_failure_delta,
                "stage_b_dag_shaping_reward": dag_shaping_reward,
                "stage_b_move_penalty": float(np.mean(move_penalties)) if move_penalties else 0.0,
                "stage_b_collision_penalty": float(np.mean(collision_penalties)) if collision_penalties else 0.0,
                "stage_b_boundary_penalty": float(np.mean(boundary_penalties)) if boundary_penalties else 0.0,
            }
        )
        return rewards

    def _build_phase_one_diagnostics(self) -> dict[str, float]:
        stats = self._task_executor.latest_stats
        diagnostics: dict[str, float] = {
            "ready_tasks": float(len(self._task_manager.get_ready_tasks())),
            "active_tasks": float(len(self._task_manager.get_active_tasks())),
            "feasible_edges": float(len(self._latest_graph_snapshot.task_uav_edges)) if self._latest_graph_snapshot is not None else 0.0,
            "score_edge_count": (
                float(self._latest_selective_score_stats.get("selective_score_edges", 0.0))
                if config.USE_HGNN_PER_TASK_RESCORING
                else float(len(self._latest_graph_scheduling_output.edge_scores))
                if self._latest_graph_scheduling_output is not None
                else 0.0
            ),
            "score_selected_assignments": float(stats.score_selected_assignments),
            "fallback_selected_assignments": float(stats.fallback_selected_assignments),
            "score_heuristic_disagreements": float(stats.score_heuristic_disagreements),
            "score_raw_disagreements": float(stats.score_raw_disagreements),
            "score_guard_fallback_assignments": float(stats.score_guard_fallback_assignments),
            "agreement_guard_rejections": float(stats.agreement_guard_rejections),
            "bounded_guard_rejections": float(stats.bounded_guard_rejections),
            "bounded_guard_clamps": float(stats.bounded_guard_clamps),
            "invalid_assignments": float(stats.invalid_actions),
        }
        diagnostics.update(self._latest_selective_score_stats)
        diagnostics.update(self._latest_phase_one_reward_terms)
        if config.USE_RL_ASSIGNMENT:
            decision_count = sum(
                1
                for record in self._latest_assignment_rl_records
                if record.get("actor_called", True)
            )
            executed_count = sum(
                1
                for record in self._latest_assignment_rl_records
                if record.get("action_executed", False)
            )
            diagnostics.update(
                {
                    "rl_assignment_decisions": float(decision_count),
                    "rl_assignment_executed_decisions": float(executed_count),
                    "rl_assignment_non_executed_decisions": float(decision_count - executed_count),
                    "rl_assignment_action_executed_rate": executed_count / float(max(decision_count, 1)),
                    "rl_assignment_invalid_actor_actions": float(stats.invalid_actor_actions),
                    "rl_assignment_no_feasible_candidates": float(stats.no_feasible_candidates),
                }
            )
        return diagnostics
