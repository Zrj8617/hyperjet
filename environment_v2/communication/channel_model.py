"""由三维位置计算 LoS/NLoS 链路的传输时间与发送能耗。"""

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
    """使用子信道带宽与噪声功率谱密度计算单跳传输。"""

    def __init__(self, config: ChannelConfig):
        """保存信道配置，将噪声与上传接收功率门限换算为 W。"""
        self.config = config
        self.noise_power_w = (
            10 ** ((config.noise_density_dbm_per_hz - 30) / 10)
            * config.subchannel_bandwidth_hz
        )
        self.min_received_power_w = 10 ** ((config.min_received_power_dbm - 30) / 10)

    def received_power_w(
        self,
        tx_position: NDArray[np.float64],
        rx_position: NDArray[np.float64],
        power_w: float,
        link_type: LinkType,
    ) -> float:
        """由三维位置与链路类型返回平均接收功率（W）。"""
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
        los_gain = self.config.reference_channel_gain / distance**2
        average_gain = (
            los_probability * los_gain
            + (1.0 - los_probability) * self.config.nlos_attenuation_factor * los_gain
        )
        return power_w * average_gain

    def transfer(
        self,
        bits: float,
        tx_position: NDArray[np.float64],
        rx_position: NDArray[np.float64],
        power_w: float,
        link_type: LinkType,
    ) -> tuple[float, float]:
        """返回指定数据量的传输时间（s）与发送方能耗（J）。"""
        received_power = self.received_power_w(tx_position, rx_position, power_w, link_type)
        rate = self.config.subchannel_bandwidth_hz * math.log2(
            1.0 + received_power / self.noise_power_w
        )
        time_s = bits / rate
        return time_s, power_w * time_s
