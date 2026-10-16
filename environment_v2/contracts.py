"""环境的动作和时隙物理结果；不负责执行。"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .types import TaskKey


FlightActions: TypeAlias = NDArray[np.float64]
# 形状 (UAV 数量, 2)，元素在 [-1, 1]；由动作接收入口校验。
Placement: TypeAlias = tuple[TaskKey, int]
Placements: TypeAlias = list[Placement]
# task 顺序为冻结 ready 列表的子序列；未选择的节点保持 READY。
# 入口检查 UAV 存在、task 就绪且不重复、队列未满；非法项跳过并计数。


@dataclass(frozen=True)
class DAGOffer:
    """本时隙生成并立即生效的 DAG 到达记录。"""

    dag_id: int
    ue_id: int
    arrival_time_s: float


@dataclass(frozen=True)
class CompletedDAG:
    """按时完成的 DAG；完成时刻为最后一个 sink 回到 UE 的时刻。"""

    dag_id: int
    arrival_time_s: float
    finish_time_s: float


@dataclass(frozen=True)
class TimedOutDAG:
    """超时 DAG；截止时刻由到达时刻和统一相对截止长度计算。"""

    dag_id: int
    arrival_time_s: float


@dataclass(frozen=True)
class SlotResult:
    """时隙物理结果；服务能耗在承诺时计入，飞行能耗为本时隙实际值。"""

    completed_dags: tuple[CompletedDAG, ...]
    timed_out_dags: tuple[TimedOutDAG, ...]
    compute_energy_j: float
    tx_energy_j: float
    return_energy_j: float
    flight_energy_j: float
    invalid_placements: int
    offers: tuple[DAGOffer, ...]
    done: bool
