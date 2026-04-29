# 导入用户设备(UE)类，用于表示终端用户设备
from environment.user_equipments import UE
# 导入无人机(UAV)类，用于表示无人机执行节点
from environment.uavs import UAV
from environment.dag_tasks import DAGTaskManager
from environment.graph_builder import HeteroGraphBuilder, HeteroGraphSnapshot
from environment.task_execution import PhaseOneTaskExecutor
from marl_models.hgnn.scheduler import PhaseOneGraphScheduler, GraphSchedulingOutput
# 导入全局配置参数（如区域大小、无人机数量、速度、覆盖半径等）
import config
# 导入数值计算库，用于数组、矩阵、距离、归一化等计算
import numpy as np
import os
import torch


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
        # 初始化环境时间步为0
        self._time_step: int = 0
        # 阶段一动态DAG任务系统
        self._task_manager: DAGTaskManager = DAGTaskManager()
        self._graph_builder: HeteroGraphBuilder = HeteroGraphBuilder()
        self._latest_graph_snapshot: HeteroGraphSnapshot | None = None
        self._task_executor: PhaseOneTaskExecutor = PhaseOneTaskExecutor()
        self._graph_scheduler: PhaseOneGraphScheduler | None = None
        self._latest_graph_scheduling_output: GraphSchedulingOutput | None = None
        self._latest_phase_one_diagnostics: dict[str, float] = {}
        if config.USE_HGNN_SCORE_ASSIGNMENT:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._graph_scheduler = PhaseOneGraphScheduler(device=device)
            if config.HGNN_SCORE_CHECKPOINT:
                if not os.path.exists(config.HGNN_SCORE_CHECKPOINT):
                    raise FileNotFoundError(f"HGNN score checkpoint not found: {config.HGNN_SCORE_CHECKPOINT}")
                state_dict = torch.load(config.HGNN_SCORE_CHECKPOINT, map_location=device)
                self._graph_scheduler.load_state_dict(state_dict)
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
    def task_executor(self) -> PhaseOneTaskExecutor:
        return self._task_executor

    @property
    def latest_graph_scheduling_output(self) -> GraphSchedulingOutput | None:
        return self._latest_graph_scheduling_output

    @property
    def latest_phase_one_diagnostics(self) -> dict[str, float]:
        return self._latest_phase_one_diagnostics

    def reset(self) -> list[np.ndarray]:
        """
        重置环境到初始状态
        返回：每个UAV的初始观测值列表
        """
        # 重新生成所有UE
        self._ues = [UE(i) for i in range(config.NUM_UES)]
        # 重新生成所有UAV
        self._uavs = [UAV(i) for i in range(config.NUM_UAVS)]
        # 时间步归零
        self._time_step = 0
        self._task_manager.reset()
        self._latest_graph_snapshot = None
        self._latest_graph_scheduling_output = None
        self._latest_phase_one_diagnostics = {}
        self._task_executor.reset(self._uavs)
        # 返回重置后的初始观测
        return self._get_obs()

    def step(self, actions: np.ndarray) -> tuple[list[np.ndarray], list[float], tuple[float, float, float, float]]:
        """
        执行一步环境交互（强化学习核心步骤）
        输入：actions -> 所有UAV的动作（位置移动等）
        返回：下一时刻观测、奖励、性能指标
        """
        # 时间步 +1
        self._time_step += 1

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
            self._latest_graph_scheduling_output = None
            edge_scores: dict[tuple[str, int], float] | None = None
            if config.USE_HGNN_SCORE_ASSIGNMENT and self._graph_scheduler is not None and self._latest_graph_snapshot is not None:
                self._latest_graph_scheduling_output = self._graph_scheduler.score_graph(self._latest_graph_snapshot)
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
        self._apply_actions_to_env(actions)

        # 10. 获取下一步观测
        next_obs: list[np.ndarray] = self._get_obs()
        # 返回：下一观测、奖励、系统指标
        return next_obs, rewards, metrics

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
            if config.USE_HGNN_SCORE_ASSIGNMENT and self._graph_scheduler is not None and self._latest_graph_snapshot is not None:
                self._latest_graph_scheduling_output = self._graph_scheduler.score_graph(self._latest_graph_snapshot)

        # 存储所有UAV的观测
        all_obs: list[np.ndarray] = []

        # 逐个构建UAV观测
        for uav in self._uavs:
            # ==================== Part 1: 无人机自身状态 ====================
            # 位置归一化（除以区域宽高）
            own_pos: np.ndarray = uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32)
            if phase_one_mode:
                own_cache = np.zeros(config.NUM_FILES, dtype=np.float32)
                own_cache[0] = min(self._task_executor.get_queue_length(uav.id) / float(max(config.DAG_MAX_QUEUE_PER_UAV, 1)), 1.0)
                own_cache[1] = 1.0 if self._task_executor.is_uav_busy(uav.id) else 0.0
                own_cache[2] = min(uav.energy / max(config.POWER_MOVE + config.POWER_HOVER, config.EPSILON), 1.0)
                own_cache[3] = len(uav.neighbors) / float(max(config.NUM_UAVS - 1, 1))
                own_cache[4] = config.UAV_COMPUTING_CAPACITY[uav.id] / float(np.max(config.UAV_COMPUTING_CAPACITY))
            else:
                # 缓存状态
                own_cache = uav.cache.astype(np.float32)
            # 拼接自身状态
            own_state: np.ndarray = np.concatenate([own_pos, own_cache])

            # ==================== Part 2: 邻居无人机状态 ====================
            # 初始化邻居状态矩阵（最大邻居数 × 邻居观测维度）
            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, config.NEIGHBOR_OBS_DIM), dtype=np.float32)
            # 按距离排序邻居，取前K个
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[: config.MAX_UAV_NEIGHBORS]
            # 逐个填入邻居相对位置
            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = (neighbor.pos[:2] - uav.pos[:2]) / config.UAV_SENSING_RANGE
                neighbor_states[i, :] = relative_pos

            # ==================== Part 3: 关联UE / 阶段一任务摘要 ====================
            # 初始化UE状态矩阵
            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, config.UE_OBS_DIM), dtype=np.float32)
            if phase_one_mode:
                task_summaries = self._get_phase_one_task_summaries(uav.id)
                for i, task in enumerate(task_summaries[: config.MAX_ASSOCIATED_UES]):
                    delta_pos: np.ndarray = (task.source_pos - uav.pos[:2]) / config.AREA_WIDTH
                    state_norm = {
                        "ready": 0.0,
                        "queued": 1.0 / 3.0,
                        "running": 2.0 / 3.0,
                        "waiting": 1.0,
                    }.get(task.state, 1.0)
                    task_info = np.array(
                        [
                            max(task.remaining_slack(self._time_step), 0.0) / float(config.DAG_MAX_DEADLINE_OFFSET),
                            task.input_size / float(config.DAG_MAX_INPUT_SIZE),
                            task.level / float(max(config.DAG_MAX_TASK_LEVELS - 1, 1)),
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
        total_energy = sum(uav.energy for uav in self._uavs)
        reward = (
            config.PHASE_ONE_FINISH_REWARD * stats.on_time_completed_tasks
            - config.PHASE_ONE_DEADLINE_PENALTY * max(stats.completed_tasks - stats.on_time_completed_tasks, 0)
            - config.PHASE_ONE_ENERGY_PENALTY * total_energy
            - config.PHASE_ONE_INVALID_PENALTY * stats.invalid_actions
        )
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

    def _build_phase_one_diagnostics(self) -> dict[str, float]:
        stats = self._task_executor.latest_stats
        diagnostics: dict[str, float] = {
            "ready_tasks": float(len(self._task_manager.get_ready_tasks())),
            "active_tasks": float(len(self._task_manager.get_active_tasks())),
            "feasible_edges": float(len(self._latest_graph_snapshot.task_uav_edges)) if self._latest_graph_snapshot is not None else 0.0,
            "score_edge_count": float(len(self._latest_graph_scheduling_output.edge_scores))
            if self._latest_graph_scheduling_output is not None
            else 0.0,
            "score_selected_assignments": float(stats.score_selected_assignments),
            "fallback_selected_assignments": float(stats.fallback_selected_assignments),
            "score_heuristic_disagreements": float(stats.score_heuristic_disagreements),
            "invalid_assignments": float(stats.invalid_actions),
        }
        return diagnostics
