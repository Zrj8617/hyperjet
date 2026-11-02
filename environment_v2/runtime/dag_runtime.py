"""DAG 的静态拓扑索引、任务记录与全部 sink 的完成状态。"""

from dataclasses import dataclass, field

from ..types import DAGStatus
from ..workload.dag_generator import DAG
from .channel_queue import TransferRecord
from .task_runtime import TaskRuntime


@dataclass
class DAGRuntime:
    """本时隙一个 DAG 的运行记录，状态只由事件运行时修改。"""

    dag: DAG
    access_uav: int
    status: DAGStatus = field(default=DAGStatus.ACTIVE, init=False)
    finish_time_s: float | None = None
    tasks: dict[int, TaskRuntime] = field(default_factory=dict)
    transfers: list[TransferRecord] = field(default_factory=list)
    successors: dict[int, list[int]] = field(init=False)
    sinks: tuple[int, ...] = field(init=False)
    remaining_sinks: int = field(init=False)

    def __post_init__(self):
        """从依赖边建立后继和 sink 索引。"""
        self.successors = {node: [] for node in range(self.dag.num_tasks)}
        for parent, child in self.dag.edges:
            self.successors[parent].append(child)
        self.sinks = tuple(node for node, children in self.successors.items() if not children)
        self.remaining_sinks = len(self.sinks)

    @property
    def compute_energy_j(self) -> float:
        """汇总本 DAG 在 UAV 上的实际计算能耗。"""
        return sum(task.compute_energy_j for task in self.tasks.values())

    @property
    def tx_energy_j(self) -> float:
        """按唯一传输记录汇总输入中继和前驱结果的 UAV 发送能耗。"""
        return sum(record.tx_energy_j for record in self.transfers if record.purpose != "return")

    @property
    def return_energy_j(self) -> float:
        """汇总 sink 回传的 UAV 发送能耗。"""
        return sum(record.tx_energy_j for record in self.transfers if record.purpose == "return")
