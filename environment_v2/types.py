"""各层共享的子任务键和运行状态。"""

from enum import Enum
from typing import TypeAlias


TaskKey: TypeAlias = tuple[int, int]


class TaskStatus(str, Enum):
    """子任务的唯一运行状态；由 runtime 修改。"""

    WAITING = "waiting"
    READY = "ready"
    SCHEDULED = "scheduled"
    RETURNING = "returning"
    DONE = "done"
    CANCELLED = "cancelled"


class DAGStatus(str, Enum):
    """DAG 的唯一运行状态；由 runtime 修改。"""

    ACTIVE = "active"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
