"""只读位置并计算上传、前驱结果和回传的逐跳路径。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .channel_model import ChannelModel, LinkType


@dataclass(frozen=True)
class Hop:
    """一跳的端点下标与物理结果；端点属于 UE 或 UAV 由链路类型决定。"""

    sender: int
    receiver: int
    link_type: LinkType
    time_s: float
    energy_j: float


@dataclass(frozen=True)
class Route:
    """按存储转发顺序排列的传输路径。"""

    hops: tuple[Hop, ...]

    @property
    def time_s(self) -> float:
        """返回各跳传输时间之和（s），无跳时为 0。"""
        return sum((hop.time_s for hop in self.hops), 0.0)

    @property
    def energy_j(self) -> float:
        """返回各跳发送能耗之和（J），无跳时为 0。"""
        return sum((hop.energy_j for hop in self.hops), 0.0)


class Router:
    """根据当前位置选择最近接入 UAV 并组合链路，不修改环境状态。"""

    def __init__(self, channel: ChannelModel, ue_power_w: float, uav_power_w: float):
        """保存信道模型和两类发送方功率。"""
        self.channel = channel
        self.ue_power_w = ue_power_w
        self.uav_power_w = uav_power_w

    def nearest_uav(
        self, ue_position: NDArray[np.float64], uav_positions: NDArray[np.float64]
    ) -> int:
        """返回三维距离最近的 UAV 下标。"""
        return int(np.argmin(np.linalg.norm(uav_positions - ue_position, axis=1)))

    def can_upload(
        self, ue_position: NDArray[np.float64], uav_positions: NDArray[np.float64]
    ) -> bool:
        """返回 UE 到最近 UAV 的接收功率是否达到上传门限。"""
        access = self.nearest_uav(ue_position, uav_positions)
        received_power = self.channel.received_power_w(
            ue_position, uav_positions[access], self.ue_power_w, LinkType.GROUND_TO_AIR,
        )
        return received_power >= self.channel.min_received_power_w

    def upload(
        self, ue_id: int, exec_uav: int, bits: float,
        ue_positions: NDArray[np.float64], uav_positions: NDArray[np.float64],
    ) -> Route:
        """从 UE 经最近接入 UAV 上传到执行 UAV。"""
        access = self.nearest_uav(ue_positions[ue_id], uav_positions)
        time_s, energy_j = self.channel.transfer(
            bits, ue_positions[ue_id], uav_positions[access],
            self.ue_power_w, LinkType.GROUND_TO_AIR,
        )
        first = Hop(ue_id, access, LinkType.GROUND_TO_AIR, time_s, energy_j)
        onward = self.predecessor(access, exec_uav, bits, uav_positions)
        return Route((first,) + onward.hops)

    def predecessor(
        self, src_uav: int, dst_uav: int, bits: float,
        uav_positions: NDArray[np.float64],
    ) -> Route:
        """传输前驱结果；源与目标为同一架 UAV 时无跳。"""
        if src_uav == dst_uav:
            return Route(())
        time_s, energy_j = self.channel.transfer(
            bits, uav_positions[src_uav], uav_positions[dst_uav],
            self.uav_power_w, LinkType.AIR_TO_AIR,
        )
        return Route((Hop(src_uav, dst_uav, LinkType.AIR_TO_AIR, time_s, energy_j),))

    def back(
        self, sink_uav: int, ue_id: int, bits: float,
        ue_positions: NDArray[np.float64], uav_positions: NDArray[np.float64],
    ) -> Route:
        """从 sink UAV 经离 UE 最近的 UAV 回传到 UE。"""
        access = self.nearest_uav(ue_positions[ue_id], uav_positions)
        onward = self.predecessor(sink_uav, access, bits, uav_positions)
        time_s, energy_j = self.channel.transfer(
            bits, uav_positions[access], ue_positions[ue_id],
            self.uav_power_w, LinkType.AIR_TO_GROUND,
        )
        last = Hop(access, ue_id, LinkType.AIR_TO_GROUND, time_s, energy_j)
        return Route(onward.hops + (last,))
