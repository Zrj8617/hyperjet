"""单跳传输记录与有向链路的 FIFO 队列。"""

from collections import deque
from dataclasses import dataclass, field

from ..communication.routing import Hop
from ..types import TransferStatus


@dataclass
class TransferJob:# 这是单跳
    """一跳的剩余数据与资源份额，状态和时刻由事件运行时填写。"""

    transfer_id: int
    hop: Hop
    payload_bits: float
    channel_gain: float
    source_ready_s: float | None = None # 数据源准备好发送的时间
    remaining_bits: float = field(init=False)
    bandwidth_hz: float = 0.0
    power_w: float = 0.0
    rate_bps: float = 0.0
    predicted_finish_s: float | None = None
    tx_energy_j: float = 0.0
    status: TransferStatus = field(default=TransferStatus.QUEUED, init=False)
    start_time_s: float | None = None
    finish_time_s: float | None = None

    def __post_init__(self):
        """将本跳剩余数据初始化为完整载荷。"""
        self.remaining_bits = self.payload_bits


@dataclass
class TransferRecord:# 把多跳结合起来
    """一次输入、前驱分发或回传的数据搬运，包含全部逐跳作业。"""

    jobs: tuple[TransferJob, ...]
    purpose: str
    start_time_s: float | None = None
    finish_time_s: float | None = None

    @property
    def tx_energy_j(self) -> float:
        """汇总此路径中 UAV 发送各跳的实际能耗（J）。"""
        return sum(job.tx_energy_j for job in self.jobs)


@dataclass
class DirectedChannelState:# 管理某一条链路上的传输排队
    """一个有向链路的队列与当前在传作业。"""

    queued: deque[TransferJob] = field(default_factory=deque)# 等待区
    active: TransferJob | None = None # 当前在传作业区

    def enqueue(self, job: TransferJob) -> None:
        """将数据已就绪的跳追加到队尾。"""
        self.queued.append(job)

    def start_head(self) -> TransferJob | None:# 这里只是将作业设为active，并没有真正开始传输
        """链路空闲且队列非空时取出队首。"""
        if self.active is not None or not self.queued: # 链路不空闲或队列为空 返回none
            return None
        self.active = self.queued.popleft()
        return self.active

    def finish_active(self) -> TransferJob:
        """释放当前在传作业并返回它。"""
        job = self.active
        self.active = None
        return job
# 同一条有向链路在这份代码中，同一时刻最多只有一个active作业，也就是独占带宽
