"""供时隙接口组装的 DAG 结果和全额物理能耗。"""

from dataclasses import dataclass

from ..contracts import DAGOffer


@dataclass(frozen=True)
class CompletedDAG:
    """按时完成的 DAG，完成时刻包含开启的 sink 回传。"""

    dag_id: int
    arrival_time_s: float
    finish_time_s: float


@dataclass(frozen=True)
class TimedOutDAG:
    """超时 DAG 的实际完成时刻；非法分配未执行时为 None。"""

    dag_id: int
    arrival_time_s: float
    finish_time_s: float | None


@dataclass(frozen=True)
class SlotResult:
    """时隙物理结果；服务能耗含截止后算完的部分，飞行能耗由调用方填入。"""

    offers: tuple[DAGOffer, ...]
    completed_dags: tuple[CompletedDAG, ...]
    timed_out_dags: tuple[TimedOutDAG, ...]
    compute_energy_j: float
    tx_energy_j: float
    return_energy_j: float
    flight_energy_j: float
    invalid_placements: int
    done: bool
