"""环境动作与 DAG 到达记录；不负责执行。"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .types import TaskKey
from .workload.dag_generator import DAG


FlightActions: TypeAlias = NDArray[np.float64]
# 形状 (UAV 数量, 2)，元素在 [-1, 1]；由动作接收入口校验。
Placement: TypeAlias = tuple[TaskKey, int | None]
Placements: TypeAlias = list[Placement]
# None 表示该 DAG 的 UE；列表按 ERS 全局排序，决定入口输入的进队顺序。
# 动作入口校验后补齐覆盖外 DAG 的 UE 分配；非法 DAG 不进入 runtime。


@dataclass(frozen=True)
class DAGOffer:
    """本时隙生成并立即生效的 DAG 到达记录。"""

    dag_id: int
    ue_id: int
    arrival_time_s: float


@dataclass(frozen=True)
class Snapshot:
    """本时隙的方法层只读视图；位置数组直接引用实体，留存需自行拷贝。"""

    slot: int
    uav_positions: NDArray[np.float64]
    ue_positions: NDArray[np.float64]
    new_dags: list[DAG]
    manager_uav: NDArray[np.int64]
    coverage: NDArray[np.bool_]
    uav_adjacency: NDArray[np.bool_]
    candidates: dict[int, tuple[int | None, ...]]
