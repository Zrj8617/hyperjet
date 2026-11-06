"""只读位置并计算入口输入、前驱结果和回传的逐跳路径。"""
# 根据ue，uav的位置算出数据该怎么走
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .channel_model import ChannelModel, LinkType


@dataclass(frozen=True)
class Hop:
    """一跳的端点下标；端点属于 UE 或 UAV 由链路类型决定。"""

    sender: int
    receiver: int
    link_type: LinkType


@dataclass(frozen=True)
class Route:
    """按存储转发顺序排列的传输路径。"""

    hops: tuple[Hop, ...]


class Router:
    """按覆盖与独占参考时长选路；执行节点 None 表示该 DAG 的 UE。"""

    def __init__(self, channel: ChannelModel, ue_power_w: float, uav_power_w: float):
        """保存信道模型和两类发送方的总功率。"""
        self.channel = channel
        self.ue_power_w = ue_power_w
        self.uav_power_w = uav_power_w

    def nearest_uav(
        self, ue_position: NDArray[np.float64], uav_positions: NDArray[np.float64]
    ) -> int:
        # 输入：一个UE坐标、全部无人机坐标数组
        # 算出这个UE到每台UAV的距离，返回最近那台UAV的id
        """返回三维距离最近的管理 UAV 下标。"""
        return int(np.argmin(np.linalg.norm(uav_positions - ue_position, axis=1)))

    def can_upload(
        self, ue_position: NDArray[np.float64], uav_position: NDArray[np.float64]
    ) -> bool:
        """用 UE 总功率判断到指定 UAV 的接收功率是否达到门限。"""
        return self.channel.received_power_w(
            ue_position, uav_position, self.ue_power_w, LinkType.GROUND_TO_AIR,
        ) >= self.channel.min_received_power_w

    def can_download(
        self, uav_position: NDArray[np.float64], ue_position: NDArray[np.float64]
    ) -> bool:
        """用 UAV 总功率判断到指定 UE 的接收功率是否达到门限。"""
        return self.channel.received_power_w(
            uav_position, ue_position, self.uav_power_w, LinkType.AIR_TO_GROUND,
        ) >= self.channel.min_received_power_w

    def in_air_range(
        self, src_position: NDArray[np.float64], dst_position: NDArray[np.float64]
    ) -> bool:
        """判断两架 UAV 的三维距离是否不超过通信范围。"""
        return bool(np.linalg.norm(dst_position - src_position) <= self.channel.config.air_range_m)

    def transfer_duration_s(
        self, hop: Hop, bits: float,
        ue_positions: NDArray[np.float64], uav_positions: NDArray[np.float64],
    ) -> float:
        """返回单跳独占全部带宽、发送方用满功率的参考时长（s）。"""
        if hop.link_type == LinkType.GROUND_TO_AIR:
            tx_position = ue_positions[hop.sender]
            rx_position = uav_positions[hop.receiver]
            power_w = self.ue_power_w
        else:
            tx_position = uav_positions[hop.sender]
            rx_position = (
                ue_positions[hop.receiver] if hop.link_type == LinkType.AIR_TO_GROUND
                else uav_positions[hop.receiver]
            )
            power_w = self.uav_power_w
        gain = self.channel.average_channel_gain(tx_position, rx_position, hop.link_type)
        return bits / self.channel.rate_bps(gain, power_w, self.channel.config.total_bandwidth_hz)

    def _select_route(
        self, direct: Hop, relay: Route, bits: float,
        ue_positions: NDArray[np.float64], uav_positions: NDArray[np.float64],
    ) -> Route:
        """在可达的直连和中继之间选参考时长较短者，相等取直连。"""
        direct_time = self.transfer_duration_s(direct, bits, ue_positions, uav_positions)
        relay_time = sum(
            self.transfer_duration_s(hop, bits, ue_positions, uav_positions)
            for hop in relay.hops
        )
        return Route((direct,)) if direct_time <= relay_time else relay

    def upload(
        self, ue_id: int, exec_uav: int | None, bits: float,
        ue_positions: NDArray[np.float64], uav_positions: NDArray[np.float64],
    ) -> Route:
        """返回入口或 UE 前驱结果的上传路径，UE 本地执行时无跳。"""
        if exec_uav is None:
            return Route(())
        access = self.nearest_uav(ue_positions[ue_id], uav_positions)
        first = Hop(ue_id, access, LinkType.GROUND_TO_AIR)
        if exec_uav == access:
            return Route((first,))
        relay = Route((first, Hop(access, exec_uav, LinkType.AIR_TO_AIR)))
        if not self.can_upload(ue_positions[ue_id], uav_positions[exec_uav]):
            return relay
        direct = Hop(ue_id, exec_uav, LinkType.GROUND_TO_AIR)
        return self._select_route(direct, relay, bits, ue_positions, uav_positions)

    def predecessor(
        self, src_uav: int | None, dst_uav: int | None, ue_id: int, bits: float,
        ue_positions: NDArray[np.float64], uav_positions: NDArray[np.float64],
    ) -> Route:
        """返回前驱结果路径，涉及 UE 时复用上传或回传，同节点无跳。"""
        if src_uav == dst_uav:
            return Route(())
        if src_uav is None:
            return self.upload(ue_id, dst_uav, bits, ue_positions, uav_positions)
        if dst_uav is None:
            return self.back(src_uav, ue_id, bits, ue_positions, uav_positions)
        if self.in_air_range(uav_positions[src_uav], uav_positions[dst_uav]):
            return Route((Hop(src_uav, dst_uav, LinkType.AIR_TO_AIR),))
        access = self.nearest_uav(ue_positions[ue_id], uav_positions)
        return Route((
            Hop(src_uav, access, LinkType.AIR_TO_AIR),
            Hop(access, dst_uav, LinkType.AIR_TO_AIR),
        ))

    def back(
        self, sink_uav: int | None, ue_id: int, bits: float,
        ue_positions: NDArray[np.float64], uav_positions: NDArray[np.float64],
    ) -> Route:
        """返回 UAV 到 UE 的回传或前驱结果路径，UE 本地结果无跳。"""
        if sink_uav is None:
            return Route(())
        access = self.nearest_uav(ue_positions[ue_id], uav_positions)
        last = Hop(access, ue_id, LinkType.AIR_TO_GROUND)
        if sink_uav == access:
            return Route((last,))
        relay = Route((Hop(sink_uav, access, LinkType.AIR_TO_AIR), last))
        if not self.can_download(uav_positions[sink_uav], ue_positions[ue_id]):
            return relay
        direct = Hop(sink_uav, ue_id, LinkType.AIR_TO_GROUND)
        return self._select_route(direct, relay, bits, ue_positions, uav_positions)
