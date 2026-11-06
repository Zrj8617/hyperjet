"""DAG 的静态拓扑索引、任务记录与全部 sink 的完成状态。"""

from dataclasses import dataclass, field

from ..types import DAGStatus
from ..workload.dag_generator import DAG
from .channel_queue import TransferRecord
from .task_runtime import TaskRuntime
# 主要负责三件事：
# 1. DAG 的静态拓扑索引：记录子任务之间的依赖关系。
# 2. 任务记录：保存 DAG 中所有子任务的运行记录。
# 3. 全部 sink 的完成状态：判断 DAG 中所有终点任务是否完成

@dataclass
class DAGRuntime:# 保存一一个DAG从开始执行到最终完成的全部运行记录
    """本时隙一个 DAG 的运行记录，状态只由事件运行时修改。"""

    dag: DAG
    access_uav: int
    status: DAGStatus = field(default=DAGStatus.ACTIVE, init=False)
    finish_time_s: float | None = None
    tasks: dict[int, TaskRuntime] = field(default_factory=dict)# 用于保存 DAG 中所有子任务的运行记录
    transfers: list[TransferRecord] = field(default_factory=list)# 记录这个 DAG 运行过程中发生的全部数据传输
    successors: dict[int, list[int]] = field(init=False)# 记录 DAG 中每个子任务的后继任务索引
    sinks: tuple[int, ...] = field(init=False)# 记录 DAG 中所有终点任务的索引
    remaining_sinks: int = field(init=False)# 记录剩余未完成的终点任务数量

    def __post_init__(self):
        """从依赖边建立后继和 sink 索引。"""
        self.successors = {node: [] for node in range(self.dag.num_tasks)}
        for parent, child in self.dag.edges:
            self.successors[parent].append(child) # 把原始 DAG 的边关系转换成了方便查询的后继字典
        self.sinks = tuple(node for node, children in self.successors.items() if not children)
        self.remaining_sinks = len(self.sinks)

    @property
    def compute_energy_j(self) -> float:
        """汇总本 DAG 在 UAV 上的实际计算能耗。"""
        return sum(task.compute_energy_j for task in self.tasks.values()) # self.tasks.values：获取 DAG 中所有子任务的运行记录
    @property
    def tx_energy_j(self) -> float:
        """按唯一传输记录汇总输入中继和前驱结果的 UAV 发送能耗。"""
        return sum(record.tx_energy_j for record in self.transfers if record.purpose != "return")

    @property
    def return_energy_j(self) -> float:
        """汇总 sink 回传的 UAV 发送能耗。"""
        return sum(record.tx_energy_j for record in self.transfers if record.purpose == "return")
