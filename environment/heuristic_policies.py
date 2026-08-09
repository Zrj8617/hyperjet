"""Pluggable, training-free heuristic offloading policies for frozen-tape evaluation.

中文：在冻结 tape 上运行「不训练」的启发式策略。本模块**不导入 torch**，
不加载任何 checkpoint，不使用 GPU。决策循环照抄
`environment/stage1_temperature_diagnostic.py::act_with_temperature` 的骨架，
只把「编码器 + scorer + 温度采样」换成两个可替换钩子：

    OrderPolicy   决定本时隙 ready 任务的处理顺序（默认恒等）
    SelectPolicy  为单个任务选一架 UAV，返回候选数组下标

顺序预留语义（`TemporaryReservationState`）原样保留：每选一个立刻 reserve，
下一个任务基于更新后的状态重算候选与 EFT。

注意：`assignment._ready_sort_key` 一个字符都没有改动。OrderPolicy 只在本模块
内部对已经冻结的 ready 列表做重排，主线的决策顺序不受影响。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

import config
from environment import comm_model
from environment.assignment import (
    CleanAssignmentBuffer,
    OffloadingCandidateEstimate,
    TemporaryReservationState,
    build_offloading_candidate_components,
)
from environment.dag_tasks import DAGTaskManager, TaskNode
from environment.stage1_temperature_sampling import keyed_uniform_from_key


HEURISTIC_POLICY_SCHEMA = "heuristic_policy_v1"
RANDOM_SELECT_KEY_SCHEMA = "heuristic_policy_random_select_v1"

# rank_u 口径标识。写进每条决策记录，保证事后分析知道用的是哪把尺子。
#   meanpair_distinct_uav -> c(t,s) 取「有序异机 UAV 对」上的传输时间均值（方案 B）
RANK_U_CONVENTION_BASE = "upward_rank_meanpair_distinct_uav_v1"


def rank_u_convention(include_return: bool) -> str:
    return f"{RANK_U_CONVENTION_BASE}:include_return={'true' if include_return else 'false'}"


# --------------------------------------------------------------------------------------
# 几何：UAV 两两距离因子 / UAV 到某点的距离因子
# --------------------------------------------------------------------------------------


def _inverse_distance_factor(distance_m: float) -> float:
    """返回 1 + (d/100)^2，即 `comm_model.clean_distance_factor` 的倒数。

    刻意通过 `comm_model` 取倒数而不是重写公式：信道模型一旦变化，这里自动跟随。
    """
    factor = float(comm_model.clean_distance_factor(float(distance_m)))
    if not (factor > 0.0):
        raise ValueError("clean_distance_factor must be strictly positive")
    return 1.0 / factor


def transmission_seconds_from_factor(data_size_mb: float, base_bandwidth_mbps: float, factor: float) -> float:
    """tx = mb * 8 / bw * factor。与 `comm_model.clean_transmission_time_seconds` 代数等价。

    因为 tx 对 mb 线性、对 bw 反比，把两者提到均值外面是**精确恒等**而非近似。
    `smoke_heuristic_policy_baseline.py` 对此有数值断言。
    """
    bandwidth = float(base_bandwidth_mbps)
    if not (bandwidth > 0.0):
        raise ValueError("base bandwidth must be strictly positive")
    return float(data_size_mb) * 8.0 / bandwidth * float(factor)


@dataclass(frozen=True, slots=True)
class UavGeometry:
    """本 episode 内恒定的 UAV 服务位置几何。

    Stage 1 冻结评估里 `env.apply_movement({})` 表示全体悬停，所以 5 架 UAV 的
    服务位置 200 个时隙不变，两两距离**精确已知**，不需要任何面积统计近似。
    `assert_positions_unchanged` 用来挡住「将来有人给这个框架传了移动动作」的情况。
    """

    uav_ids: tuple[int, ...]
    positions: tuple[tuple[float, float], ...]
    mean_distinct_pair_factor: float

    @classmethod
    def from_service_positions(cls, uav_service_positions: dict[int, Any]) -> "UavGeometry":
        uav_ids = tuple(sorted(int(key) for key in uav_service_positions))
        if len(uav_ids) < 2:
            raise ValueError("UAV geometry requires at least two UAVs")
        positions = tuple(
            tuple(float(value) for value in np.asarray(uav_service_positions[uav_id], dtype=np.float64).reshape(-1)[:2])
            for uav_id in uav_ids
        )
        factors: list[float] = []
        for i in range(len(uav_ids)):
            for j in range(i + 1, len(uav_ids)):
                distance = comm_model.clean_distance_2d(positions[i], positions[j])
                factors.append(_inverse_distance_factor(distance))
        # 距离对称，所以无序对均值 == 有序异机对均值。同机不参与平均（方案 B）。
        mean_factor = float(sum(factors) / float(len(factors)))
        return cls(uav_ids=uav_ids, positions=positions, mean_distinct_pair_factor=mean_factor)

    def mean_uav_to_point_factor(self, point: Any) -> float:
        """UAV -> 某个固定点（回传时用 DAG 到达时冻结的 `job.source_pos`）的平均距离因子。"""
        target = np.asarray(point, dtype=np.float64).reshape(-1)[:2]
        factors = [_inverse_distance_factor(comm_model.clean_distance_2d(position, target)) for position in self.positions]
        return float(sum(factors) / float(len(factors)))

    def assert_positions_unchanged(self, uav_service_positions: dict[int, Any], *, where: str) -> None:
        current = UavGeometry.from_service_positions(uav_service_positions)
        if current.uav_ids != self.uav_ids or current.positions != self.positions:
            raise AssertionError(
                f"UAV service positions changed ({where}); the cached rank_u mean-pair factor is stale. "
                "This framework assumes hover-only movement (env.apply_movement({}))."
            )


# --------------------------------------------------------------------------------------
# HEFT upward rank
# --------------------------------------------------------------------------------------


def task_computation_seconds(task: TaskNode) -> float:
    """w(t) = num_operation / UAV_COMPUTE_RATE_OPS_PER_SEC。UAV 同质，所以这是精确值。"""
    return float(task.num_operation) / float(config.UAV_COMPUTE_RATE_OPS_PER_SEC)


def compute_upward_ranks(
    *,
    dag_id: str,
    task_manager: DAGTaskManager,
    geometry: UavGeometry,
    include_return: bool,
) -> dict[str, float]:
    """标准 HEFT upward rank，按 `_mark_critical_path` 的逆拓扑序结构实现。

        rank_u(t)    = w(t) + max_{s in succ(t)} ( c(t,s) + rank_u(s) )
        rank_u(sink) = w(sink)  [+ 回传项，当 include_return=True]

    c(t,s) = t.output_data_size_mb * 8 / job.base_upload_bandwidth_mbps
             * mean_{a != b} (1 + d(a,b)^2/100^2)          （方案 B）

    注意 c 只依赖父节点 t（传的是 t 的输出），与 s 无关 —— 这是本项目通信模型的性质。
    回传项用 `base_download_bandwidth_mbps` 和 `job.source_pos`（DAG 到达时冻结）。
    """
    job = task_manager.get_job(str(dag_id))
    if job is None:
        raise ValueError(f"unknown dag_id for upward rank: {dag_id}")
    tasks = task_manager.get_job_tasks(str(dag_id))
    if not tasks:
        raise ValueError(f"dag has no tasks: {dag_id}")
    task_map = {task.task_id: task for task in tasks}
    sink_ids = set(str(value) for value in job.sink_task_ids)

    return_factor = geometry.mean_uav_to_point_factor(job.source_pos) if include_return else 0.0
    upload_bandwidth = float(job.base_upload_bandwidth_mbps)
    download_bandwidth = float(job.base_download_bandwidth_mbps)

    ranks: dict[str, float] = {}
    # 与 `_mark_critical_path` 完全同构的逆拓扑序：level 降序，同 level 按 task_id。
    for task in sorted(tasks, key=lambda item: (-int(item.level), str(item.task_id))):
        successors = [child_id for child_id in task.successors if child_id in task_map]
        if not successors:
            value = task_computation_seconds(task)
            if include_return and task.task_id in sink_ids:
                value += transmission_seconds_from_factor(
                    task.output_data_size_mb, download_bandwidth, return_factor
                )
            ranks[task.task_id] = float(value)
            continue
        missing = [child_id for child_id in successors if child_id not in ranks]
        if missing:
            raise AssertionError(
                f"reverse-topological order violated for {task.task_id}: successors {missing} not ranked yet"
            )
        communication = transmission_seconds_from_factor(
            task.output_data_size_mb, upload_bandwidth, geometry.mean_distinct_pair_factor
        )
        ranks[task.task_id] = float(
            task_computation_seconds(task) + max(communication + ranks[child_id] for child_id in successors)
        )
    if len(ranks) != len(task_map):
        raise AssertionError("upward rank did not cover every task in the DAG")
    return ranks


class UpwardRankCache:
    """按 dag_id 缓存 rank_u。UAV 位置恒定，所以缓存在整个 episode 内有效。"""

    __slots__ = ("_geometry", "_include_return", "_cache", "_convention")

    def __init__(self, *, geometry: UavGeometry, include_return: bool) -> None:
        self._geometry = geometry
        self._include_return = bool(include_return)
        self._cache: dict[str, dict[str, float]] = {}
        self._convention = rank_u_convention(bool(include_return))

    @property
    def convention(self) -> str:
        return self._convention

    @property
    def include_return(self) -> bool:
        return self._include_return

    @property
    def dag_count(self) -> int:
        return len(self._cache)

    def ranks_for_dag(self, dag_id: str, task_manager: DAGTaskManager) -> dict[str, float]:
        key = str(dag_id)
        cached = self._cache.get(key)
        if cached is None:
            cached = compute_upward_ranks(
                dag_id=key,
                task_manager=task_manager,
                geometry=self._geometry,
                include_return=self._include_return,
            )
            self._cache[key] = cached
        return cached

    def rank_for_task(self, task: TaskNode, task_manager: DAGTaskManager) -> float:
        ranks = self.ranks_for_dag(task.dag_id, task_manager)
        value = ranks.get(task.task_id)
        if value is None:
            raise AssertionError(f"task {task.task_id} missing from its DAG rank table")
        return float(value)


# --------------------------------------------------------------------------------------
# 钩子协议
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """一次卸载决策的只读上下文，供 SelectPolicy 做 keyed 哈希派生随机数。"""

    evaluation_scenario_seed: int
    slot_index: int
    stable_task_id: str
    decision_order: int
    policy_replicate: int


class OrderPolicy(Protocol):
    """钩子 1：决定本时隙任务的处理顺序。返回值必须是入参的一个排列。"""

    name: str

    def __call__(
        self,
        *,
        frozen_ready_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
        env: Any,
    ) -> list[TaskNode]: ...


class SelectPolicy(Protocol):
    """钩子 2：为单个任务选一架 UAV。返回候选数组下标，必须落在 mask 为 True 的位置。"""

    name: str

    def __call__(
        self,
        *,
        task: TaskNode,
        candidate_uav_ids: list[int],
        candidate_mask: np.ndarray,
        estimates: list[OffloadingCandidateEstimate],
        state_view: TemporaryReservationState,
        task_manager: DAGTaskManager,
        env: Any,
        context: DecisionContext,
    ) -> int: ...


# --------------------------------------------------------------------------------------
# OrderPolicy 实现
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdentityOrderPolicy:
    """恒等：保持 `assignment.freeze_ready_tasks` 给出的顺序。"""

    name: str = "identity"

    def __call__(
        self,
        *,
        frozen_ready_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
        env: Any,
    ) -> list[TaskNode]:
        del task_manager, env
        return list(frozen_ready_tasks)


class HeftUpwardRankOrderPolicy:
    """HEFT：按 rank_u 降序处理本时隙的 ready 任务。

    跨并发 DAG 直接比较原始秒数（不归一化）—— 这是刻意的适配，论文里必须显式声明：
    "HEFT adapted to the multi-DAG online setting by comparing upward ranks across
    concurrent DAGs."

    排序键是全序，无平局歧义：(-rank_u, DAG 到达时间, dag_id, 拓扑序号, task_id)。
    后四项与 `_ready_sort_key` 一致，保证 rank_u 相等时退化为主线顺序。
    """

    __slots__ = ("name", "_cache")

    def __init__(self, *, rank_cache: UpwardRankCache) -> None:
        self.name = "heft_ret1" if rank_cache.include_return else "heft_ret0"
        self._cache = rank_cache

    @property
    def rank_cache(self) -> UpwardRankCache:
        return self._cache

    def __call__(
        self,
        *,
        frozen_ready_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
        env: Any,
    ) -> list[TaskNode]:
        del env

        def sort_key(task: TaskNode) -> tuple[float, float, str, int, str]:
            job = task_manager.get_job(task.dag_id)
            arrival = float(job.arrival_time if job is not None else task.arrival_time)
            topological_index = int(getattr(task, "topological_index", 0) or 0)
            return (
                -self._cache.rank_for_task(task, task_manager),
                arrival,
                str(task.dag_id),
                topological_index,
                str(task.task_id),
            )

        return sorted(frozen_ready_tasks, key=sort_key)


# --------------------------------------------------------------------------------------
# SelectPolicy 实现
# --------------------------------------------------------------------------------------


def _legal_indices(candidate_mask: np.ndarray) -> np.ndarray:
    legal = np.flatnonzero(np.asarray(candidate_mask, dtype=bool).reshape(-1))
    if legal.size == 0:
        raise ValueError("select policy requires at least one legal candidate")
    return legal


def _assert_legal_finite_eft(legal: np.ndarray, estimates: Sequence[OffloadingCandidateEstimate]) -> None:
    """防御：所有合法候选的 EFT 必须有限。这类 bug 一旦发生是静默的。"""
    for index in legal:
        value = float(estimates[int(index)].estimated_finish_time)
        if not np.isfinite(value):
            raise AssertionError(f"legal candidate {int(index)} has non-finite EFT: {value}")


@dataclass(frozen=True, slots=True)
class GreedyEFTSelectPolicy:
    """在合法候选中取 EFT 最小；并列时取最小 uav_id。

    必须先按 mask 过滤：`estimate_offloading_candidate` 对非法候选返回
    `estimated_finish_time == 0.0`（不是 +inf），不过滤会稳定选中非法候选。
    与 `act_with_temperature` 里 corpus 的 `greedy_eft_uav_id` 计算逐行等价
    （精确浮点相等判并列，再取最小 uav_id）。
    """

    name: str = "greedy_eft"

    def __call__(
        self,
        *,
        task: TaskNode,
        candidate_uav_ids: list[int],
        candidate_mask: np.ndarray,
        estimates: list[OffloadingCandidateEstimate],
        state_view: TemporaryReservationState,
        task_manager: DAGTaskManager,
        env: Any,
        context: DecisionContext,
    ) -> int:
        del task, state_view, task_manager, env, context
        legal = _legal_indices(candidate_mask)
        _assert_legal_finite_eft(legal, estimates)
        eft = [float(value.estimated_finish_time) for value in estimates]
        minimum = float(np.min(np.asarray(eft, dtype=np.float64)[legal]))
        tied = [int(index) for index in legal if float(eft[index]) == minimum]
        return min(tied, key=lambda index: int(candidate_uav_ids[index]))


@dataclass(frozen=True, slots=True)
class RandomSelectPolicy:
    """在合法候选中均匀随机取一个。keyed 哈希派生，不依赖全局 RNG。

    key 里带 `policy_name`，保证不同策略之间的随机流互相独立，不产生伪相关。
    """

    name: str = "random"

    def __call__(
        self,
        *,
        task: TaskNode,
        candidate_uav_ids: list[int],
        candidate_mask: np.ndarray,
        estimates: list[OffloadingCandidateEstimate],
        state_view: TemporaryReservationState,
        task_manager: DAGTaskManager,
        env: Any,
        context: DecisionContext,
    ) -> int:
        del task, state_view, task_manager, env
        legal = _legal_indices(candidate_mask)
        _assert_legal_finite_eft(legal, estimates)
        key = [
            RANDOM_SELECT_KEY_SCHEMA,
            str(self.name),
            int(context.evaluation_scenario_seed),
            int(context.slot_index),
            str(context.stable_task_id),
            int(context.decision_order),
            int(context.policy_replicate),
        ]
        uniform = keyed_uniform_from_key(key)
        position = int(uniform * float(legal.size))
        position = min(max(position, 0), int(legal.size) - 1)
        return int(legal[position])


# --------------------------------------------------------------------------------------
# 策略组合与注册表
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeuristicPolicy:
    """一个完整策略 = OrderPolicy + SelectPolicy。"""

    order_policy: Any
    select_policy: Any
    heft_include_return: bool | None

    @property
    def name(self) -> str:
        return f"{self.order_policy.name}+{self.select_policy.name}"


ORDER_POLICY_NAMES = ("identity", "heft_ret0", "heft_ret1")
SELECT_POLICY_NAMES = ("greedy_eft", "random")
POLICY_NAMES = tuple(f"{order}+{select}" for order in ORDER_POLICY_NAMES for select in SELECT_POLICY_NAMES)


def build_policy(policy_name: str, *, geometry: UavGeometry) -> HeuristicPolicy:
    """按 `<order>+<select>` 组装策略。每个 episode 重建一次（rank 缓存绑定几何）。"""
    name = str(policy_name)
    if name not in POLICY_NAMES:
        raise ValueError(f"unknown policy {name!r}; expected one of {POLICY_NAMES}")
    order_name, select_name = name.split("+", 1)
    if order_name == "identity":
        order_policy: Any = IdentityOrderPolicy()
        include_return: bool | None = None
    else:
        include_return = order_name == "heft_ret1"
        order_policy = HeftUpwardRankOrderPolicy(
            rank_cache=UpwardRankCache(geometry=geometry, include_return=include_return)
        )
    select_policy = GreedyEFTSelectPolicy() if select_name == "greedy_eft" else RandomSelectPolicy()
    return HeuristicPolicy(order_policy=order_policy, select_policy=select_policy, heft_include_return=include_return)


# --------------------------------------------------------------------------------------
# 决策循环
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class HeuristicSlotResult:
    assignments: CleanAssignmentBuffer
    decision_records: list[dict[str, Any]] = field(default_factory=list)
    skip_count: int = 0


def _assert_permutation(original: list[TaskNode], ordered: list[TaskNode]) -> None:
    if len(ordered) != len(original):
        raise AssertionError("OrderPolicy must return a permutation of the frozen ready tasks (length changed)")
    if sorted(str(task.task_id) for task in ordered) != sorted(str(task.task_id) for task in original):
        raise AssertionError("OrderPolicy must return a permutation of the frozen ready tasks (contents changed)")


def act_with_heuristic_policy(
    *,
    policy: HeuristicPolicy,
    frozen_ready_tasks: list[TaskNode],
    graph_snapshot: Any,
    task_manager: DAGTaskManager,
    uavs: list[Any],
    executor: Any,
    current_time_seconds: float,
    uav_service_positions: dict[int, Any],
    ue_service_positions: dict[int, Any],
    ues: list[Any],
    env: Any,
    evaluation_scenario_seed: int,
    episode_index: int,
    slot_index: int,
    policy_replicate: int,
    analysis_rank_cache: UpwardRankCache | None = None,
    record_decisions: bool = False,
) -> HeuristicSlotResult:
    """无 torch 的顺序卸载决策循环。结构与 `act_with_temperature` 逐行对应。"""
    reservation = TemporaryReservationState.from_executor(uavs, executor)
    assignments, records, skips = CleanAssignmentBuffer(), [], 0

    ordered_tasks = policy.order_policy(
        frozen_ready_tasks=list(frozen_ready_tasks), task_manager=task_manager, env=env
    )
    _assert_permutation(list(frozen_ready_tasks), list(ordered_tasks))

    for decision_order, task in enumerate(ordered_tasks):
        task_idx = graph_snapshot.task_id_to_idx.get(task.task_id)
        if task_idx is None:
            skips += 1
            continue
        dynamic, pair, mask, uav_ids, estimates = build_offloading_candidate_components(
            task=task,
            uavs=uavs,
            task_manager=task_manager,
            executor=executor,
            state_view=reservation,
            current_time_seconds=float(current_time_seconds),
            uav_service_positions=uav_service_positions,
            ue_service_positions=ue_service_positions,
            ues=ues,
        )
        if dynamic.shape[0] == 0 or not bool(mask.any()):
            skips += 1
            continue

        context = DecisionContext(
            evaluation_scenario_seed=int(evaluation_scenario_seed),
            slot_index=int(slot_index),
            stable_task_id=str(task.task_id),
            decision_order=int(decision_order),
            policy_replicate=int(policy_replicate),
        )
        selected = int(
            policy.select_policy(
                task=task,
                candidate_uav_ids=list(uav_ids),
                candidate_mask=mask,
                estimates=list(estimates),
                state_view=reservation,
                task_manager=task_manager,
                env=env,
                context=context,
            )
        )
        # 防御断言：选中的下标必须落在 legal 集合里。静默越界会污染整批结果。
        if not 0 <= selected < int(mask.shape[0]):
            raise AssertionError(f"SelectPolicy returned out-of-range index {selected}")
        if not bool(mask[selected]):
            raise AssertionError(f"SelectPolicy returned illegal candidate index {selected}")

        estimate = estimates[selected]
        selected_uav_id = int(uav_ids[selected])
        assignments.append(task.task_id, selected_uav_id, decision_order)
        reservation.reserve(
            task.task_id,
            selected_uav_id,
            estimated_available_time=estimate.estimated_finish_time,
            estimated_queued_workload=estimate.estimated_queued_workload,
        )

        if record_decisions:
            rank_cache = getattr(policy.order_policy, "rank_cache", None) or analysis_rank_cache
            rank_value = float(rank_cache.rank_for_task(task, task_manager)) if rank_cache is not None else None
            job = task_manager.get_job(task.dag_id)
            records.append(
                {
                    "schema": "heuristic_policy_decision_v1",
                    "policy_name": policy.name,
                    "order_policy": policy.order_policy.name,
                    "select_policy": policy.select_policy.name,
                    "heft_include_return": policy.heft_include_return,
                    "policy_replicate": int(policy_replicate),
                    "evaluation_scenario_seed": int(evaluation_scenario_seed),
                    "episode_index": int(episode_index),
                    "slot_index": int(slot_index),
                    "stable_task_id": str(task.task_id),
                    "decision_order": int(decision_order),
                    "dag_id": str(task.dag_id),
                    "candidate_uav_ids": [int(value) for value in uav_ids],
                    "candidate_mask": [bool(value) for value in mask],
                    "eft": [float(value.estimated_finish_time) for value in estimates],
                    "selected_index": selected,
                    "selected_uav_id": selected_uav_id,
                    "legal_candidate_count": int(np.count_nonzero(mask)),
                    "rank_u": rank_value,
                    "rank_u_convention": rank_cache.convention if rank_cache is not None else None,
                    "base_upload_bandwidth_mbps": float(job.base_upload_bandwidth_mbps) if job is not None else None,
                }
            )

    return HeuristicSlotResult(assignments=assignments, decision_records=records, skip_count=skips)
