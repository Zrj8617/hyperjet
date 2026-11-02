"""子任务唯一的执行状态、数据到达时刻和计算记录。"""

from dataclasses import dataclass, field

from ..types import TaskKey, TaskStatus
from .channel_queue import TransferRecord


@dataclass
class TaskRuntime:
    """已分配的子任务；None 执行节点表示所属 DAG 的 UE。"""

    key: TaskKey
    execution_node: int | None
    cpu_cycles: float
    input_bits: float
    predecessors: set[TaskKey]
    status: TaskStatus = field(default=TaskStatus.WAITING_DATA, init=False)
    input_record: TransferRecord | None = None
    predecessor_records: dict[TaskKey, TransferRecord] = field(default_factory=dict)
    return_record: TransferRecord | None = None
    input_arrival_s: float | None = None
    predecessor_arrival_s: dict[TaskKey, float] = field(default_factory=dict)
    ready_time_s: float | None = None
    start_time_s: float | None = None
    finish_time_s: float | None = None
    compute_energy_j: float = 0.0
