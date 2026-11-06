"""各层共享的子任务键和运行状态。"""

from enum import Enum
from typing import TypeAlias


TaskKey: TypeAlias = tuple[int, int]


class TaskStatus(str, Enum):
    """子任务的唯一运行状态；由 runtime 修改。"""

    WAITING_DATA = "waiting_data"
    QUEUED_COMPUTE = "queued_compute"
    RUNNING = "running"
    FINISHED = "finished"


class TransferStatus(str, Enum):
    """单跳传输的唯一运行状态；由 runtime 修改。"""

    QUEUED = "queued"
    ACTIVE = "active"
    FINISHED = "finished"


class DAGStatus(str, Enum):
    """DAG 的唯一运行状态；由 runtime 修改。"""

    ACTIVE = "active"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
# 只有runtime会修改状态，其他层只读