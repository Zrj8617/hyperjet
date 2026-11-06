"""由三维位置计算平均信道增益，并按分配的功率与带宽计算速率。"""
#模型只考虑了热噪声，没有加干扰（其他 UAV、其他 UE 的同频干扰），属于简化模型
from enum import Enum
import math

import numpy as np
from numpy.typing import NDArray

from ..settings import ChannelConfig


class LinkType(str, Enum):
    """地对空、空对空和空对地三类有向链路。"""

    GROUND_TO_AIR = "ground_to_air"
    AIR_TO_AIR = "air_to_air"
    AIR_TO_GROUND = "air_to_ground"


class ChannelModel:
    """提供 LoS/NLoS 平均信道增益、接收功率与香农速率。"""

    def __init__(self, config: ChannelConfig):
        """保存信道配置，将噪声谱密度和接收功率门限换算为线性值。"""
        self.config = config
        self.noise_density_w_per_hz = (
            10 ** ((config.noise_density_dbm_per_hz - 30) / 10)
        )
        self.min_received_power_w = 10 ** ((config.min_received_power_dbm - 30) / 10)

    def average_channel_gain(
        self,
        tx_position: NDArray[np.float64],
        rx_position: NDArray[np.float64],
        link_type: LinkType,
    ) -> float:
        """由三维位置与链路类型返回平均功率增益。"""
        displacement = rx_position - tx_position
        horizontal_distance = float(np.linalg.norm(displacement[:2]))
        height_difference = abs(float(displacement[2]))
        distance = math.hypot(horizontal_distance, height_difference)
        elevation = math.degrees(math.atan2(height_difference, horizontal_distance))
        if link_type == LinkType.AIR_TO_AIR:
            los_probability = 1.0
        else:
            los_probability = 1.0 / (
                1.0 + self.config.los_alpha
                * math.exp(-self.config.los_beta * (elevation - self.config.los_alpha))
            )
        return (
            los_probability + (1.0 - los_probability) * self.config.nlos_attenuation_factor
        ) * self.config.reference_channel_gain / max(distance**2, 1e-12)

    def received_power_w(
        self,
        tx_position: NDArray[np.float64],
        rx_position: NDArray[np.float64],
        power_w: float,
        link_type: LinkType,
    ) -> float:
        """返回发送功率乘以平均信道增益得到的接收功率（W）。"""
        return power_w * self.average_channel_gain(
            tx_position, rx_position, link_type,
        )

    def rate_bps(self, channel_gain: float, power_w: float, bandwidth_hz: float) -> float:
        """按给定增益、发送功率（W）和带宽（Hz）返回速率（bit/s）。"""
        return bandwidth_hz * math.log2(
            1.0 + power_w * channel_gain / (self.noise_density_w_per_hz * bandwidth_hz)
        )
