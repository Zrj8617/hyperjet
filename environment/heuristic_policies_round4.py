"""第 4 轮新增的两个启发式钩子：`shortest_queue` 与 `dag_remaining_asc`。

中文：**只新增，不修改** `environment/heuristic_policies.py`。本模块复用它的
`HeuristicPolicy` 容器、`_legal_indices` / `_assert_legal_finite_eft` 防御，
以及全部既有策略——老名字**原样委托回原 `build_policy`**，保证第 3 轮已经跑出来的
结果逐 bit 不受影响。

同样不 import torch、不加载 checkpoint、不用 GPU、不碰 `assignment._ready_sort_key`。

----------------------------------------------------------------------------------
两个新钩子分别回答什么
----------------------------------------------------------------------------------

**`shortest_queue`（SelectPolicy）** —— 检验「B1 之前的策略是不是退化成了负载均衡」。
这是目前唯一能直接检验该假设的基线：如果纯负载均衡就能拿到 pre-B1 的分数，
那 108.48 里就没有多少「真调度」的成分。

**`dag_remaining_asc`（OrderPolicy）** —— 检验 **DAG 级**优先级有没有 leverage。
它与第 3 轮跑过的 HEFT `rank_u` 降序**机制不同、方向相反**：

| | 排序依据 | 层级 | 方向 |
| --- | --- | --- | --- |
| `heft_ret0/1` | `rank_u` 降序 | DAG 内（关键路径） | 优先推进**离完成最远**的 DAG |
| `dag_remaining_asc` | DAG 剩余未完成任务数升序 | 跨 DAG | 优先推进**离完成最近**的 DAG |

主指标 `completed_dag_count` 只在一个 DAG 的**全部**任务完成时才 +1，
所以「先做快完成的 DAG」直接最大化该指标，先验很强。
第 3 轮「rank_u 没有 leverage」的结论**推不出** DAG 级优先级没有 leverage。

> ⚠️ 一个必须记住的对照事实（第 3 轮实测）：环境默认顺序
> `_ready_sort_key` 首键是 DAG 到达时间 → 老 DAG 优先 → 而老 DAG 通常离完成最近。
> 也就是说**默认顺序本身已经是「先做快完成的 DAG」的粗糙版本**
> （实测它与 `rank_u` 降序的 Kendall τ-b = −0.22，61% 逆序对）。
> `dag_remaining_asc` 要超越的不是中性基线，而是一个已经在做类似事情的顺序，
> 增量可能小于先验。它的价值在于是同一机制的**干净直接版**。

----------------------------------------------------------------------------------
已钉死的设计决策（2026-08-08 确认，不要自行更改）
----------------------------------------------------------------------------------

1. **`shortest_queue` 并列破法 = EFT 最小 → 再取最小 `uav_id`。**
   队列长度是 0–16 的整数且只有 5 架 UAV，并列极常见；直接取最小 `uav_id` 会
   系统性偏向 UAV 0，反而破坏负载均衡、测出一个不能代表「负载均衡」的假基线。
   取 EFT 做次级键后，**队列长度仍然严格主导**，与 `greedy_eft`（EFT 主导）
   构成干净对照，且保持确定性、无随机流。
2. **「剩余任务数」= 该 DAG 中 `is_fully_completed` 为假的任务数。**
   与主指标口径一致：DAG 必须全部任务完成（含 sink 回传）才计数，
   所以 `RETURNING` 态的任务仍然算「剩余」。
3. **两个新钩子的并列尾键一律回落到 `_ready_sort_key` 的后四项**
   （`arrival_time, dag_id, topological_index, task_id`），保证全序、无并列，
   与 `HeftUpwardRankOrderPolicy` 的约定一致。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from environment.assignment import OffloadingCandidateEstimate, TemporaryReservationState
from environment.dag_tasks import DAGTaskManager, TaskNode
from environment.heuristic_policies import (
    ORDER_POLICY_NAMES,
    SELECT_POLICY_NAMES,
    DecisionContext,
    HeuristicPolicy,
    UavGeometry,
    _assert_legal_finite_eft,
    _legal_indices,
    build_policy,
)

ROUND4_POLICY_SCHEMA = "heuristic_policy_round4_ext_v1"

NEW_ORDER_POLICY_NAMES = ("dag_remaining_asc",)
NEW_SELECT_POLICY_NAMES = ("shortest_queue",)

ORDER_POLICY_NAMES_EXT = tuple(ORDER_POLICY_NAMES) + NEW_ORDER_POLICY_NAMES
SELECT_POLICY_NAMES_EXT = tuple(SELECT_POLICY_NAMES) + NEW_SELECT_POLICY_NAMES
POLICY_NAMES_EXT = tuple(
    f"{order}+{select}" for order in ORDER_POLICY_NAMES_EXT for select in SELECT_POLICY_NAMES_EXT
)


# --------------------------------------------------------------------------------------
# OrderPolicy：DAG 剩余未完成任务数升序
# --------------------------------------------------------------------------------------


def dag_remaining_task_count(dag_id: str, task_manager: DAGTaskManager) -> int:
    """该 DAG 还有多少个任务没有「完全完成」。

    口径与主指标 `completed_dag_count` 对齐：`is_fully_completed` 才算完成，
    正在回传结果（`RETURNING`）的任务仍然计入剩余。
    """
    tasks = task_manager.get_job_tasks(str(dag_id))
    return sum(1 for task in tasks if not bool(task.is_fully_completed))


class DagRemainingAscOrderPolicy:
    """跨 DAG 优先级：剩余未完成任务数少的 DAG 先做。

    并列时依次回落到 `(arrival_time, dag_id, topological_index, task_id)`，
    与 `_ready_sort_key` 的后四项一致，保证全序。
    """

    name: str = "dag_remaining_asc"

    def __call__(
        self,
        *,
        frozen_ready_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
        env: Any,
    ) -> list[TaskNode]:
        del env
        # 剩余数每个时隙都在变，缓存只在本次调用（= 本时隙）内有效。
        cache: dict[str, int] = {}

        def remaining(dag_id: str) -> int:
            key = str(dag_id)
            if key not in cache:
                cache[key] = dag_remaining_task_count(key, task_manager)
            return cache[key]

        def sort_key(task: TaskNode) -> tuple[int, float, str, int, str]:
            job = task_manager.get_job(task.dag_id)
            arrival = float(job.arrival_time if job is not None else task.arrival_time)
            topological_index = int(getattr(task, "topological_index", 0) or 0)
            return (
                remaining(str(task.dag_id)),
                arrival,
                str(task.dag_id),
                topological_index,
                str(task.task_id),
            )

        return sorted(frozen_ready_tasks, key=sort_key)


# --------------------------------------------------------------------------------------
# SelectPolicy：最短队列
# --------------------------------------------------------------------------------------


class ShortestQueueSelectPolicy:
    """在合法候选中取**当前队列最短**的 UAV；并列取 EFT 最小，再并列取最小 uav_id。

    关键：读的是 `state_view.queue_lengths`，也就是**顺序预留之后**的队列长度。
    同一时隙内每选一架 UAV，它的队列长度立刻 +1，下一个任务看到的是更新后的状态。
    这正是负载均衡该有的语义——否则同一时隙里所有任务会一起涌向同一架最空的 UAV。
    """

    name: str = "shortest_queue"

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
        del task, task_manager, env, context
        legal = _legal_indices(candidate_mask)
        _assert_legal_finite_eft(legal, estimates)

        def sort_key(index: int) -> tuple[int, float, int]:
            uav_id = int(candidate_uav_ids[int(index)])
            return (
                int(state_view.queue_lengths.get(uav_id, 0)),
                float(estimates[int(index)].estimated_finish_time),
                uav_id,
            )

        return int(min((int(value) for value in legal), key=sort_key))


# --------------------------------------------------------------------------------------
# 扩展注册表
# --------------------------------------------------------------------------------------


def build_policy_ext(policy_name: str, *, geometry: UavGeometry) -> HeuristicPolicy:
    """扩展版组装器。

    **老名字原样委托回 `heuristic_policies.build_policy`**——第 3 轮的结果因此
    逐 bit 不受本模块影响（有冒烟测试守住这条）。
    """
    name = str(policy_name)
    if name not in POLICY_NAMES_EXT:
        raise ValueError(f"unknown policy {name!r}; expected one of {POLICY_NAMES_EXT}")
    order_name, select_name = name.split("+", 1)

    if order_name in ORDER_POLICY_NAMES and select_name in SELECT_POLICY_NAMES:
        return build_policy(name, geometry=geometry)

    if order_name in ORDER_POLICY_NAMES:
        # 老 order + 新 select：借原组装器造出 order 部分，避免重复实现 rank 缓存。
        base = build_policy(f"{order_name}+greedy_eft", geometry=geometry)
        order_policy: Any = base.order_policy
        include_return: bool | None = base.heft_include_return
    else:
        order_policy = DagRemainingAscOrderPolicy()
        include_return = None

    select_policy: Any = (
        ShortestQueueSelectPolicy()
        if select_name == "shortest_queue"
        else build_policy(f"identity+{select_name}", geometry=geometry).select_policy
    )
    return HeuristicPolicy(
        order_policy=order_policy, select_policy=select_policy, heft_include_return=include_return
    )
