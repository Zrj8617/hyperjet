"""UAV 和 UE 共用的单核 FIFO 计算队列。"""

from collections import deque
from dataclasses import dataclass, field

from ..types import TaskKey
from .task_runtime import TaskRuntime


@dataclass(frozen=True)
class ComputeRecord:
    """一次计算的任务键、起止时刻和计算能耗。"""

    task_key: TaskKey
    start_time_s: float
    finish_time_s: float
    energy_j: float


@dataclass
class ServerState:
    """一个单核执行节点，UE 的计算能耗系数取零。"""

    frequency_hz: float
    kappa: float
    queued: deque[TaskRuntime] = field(default_factory=deque)
    running: ComputeRecord | None = None

    def enqueue(self, task: TaskRuntime) -> None:
        """将输入已经到齐的子任务追加到队尾。"""
        self.queued.append(task)

    def dispatch(self, now: float) -> ComputeRecord | None:
        """单核空闲时启动队首并计算完成时刻与能耗。"""
        if self.running is not None or not self.queued:
            return None
        task = self.queued.popleft()
        self.running = ComputeRecord(
            task.key, now, now + task.cpu_cycles / self.frequency_hz,
            self.kappa * task.cpu_cycles * self.frequency_hz**2,
        )
        return self.running

    def finish(self) -> ComputeRecord:
        """释放单核并返回刚完成的计算记录。"""
        record = self.running
        self.running = None
        return record
