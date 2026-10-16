"""同构无人机集合的位置、飞行与推进能耗。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..contracts import FlightActions
from ..settings import RegionConfig, SchedulingConfig, UAVConfig


@dataclass
class UAV:
    """全部 UAV 的三维位置；编号等于数组下标。"""

    positions: NDArray[np.float64]

    @classmethod
    def create(
        cls, config: UAVConfig, region: RegionConfig, rng: np.random.Generator
    ) -> "UAV":
        """用场景流在地图内均匀采样位置，高度固定。"""
        positions = np.empty((config.count, 3))
        positions[:, :2] = rng.uniform(0.0, region.side_length_m, (config.count, 2))
        positions[:, 2] = config.altitude_m
        return cls(positions)

    def apply_flight_actions(
        self, actions: FlightActions, config: UAVConfig, region: RegionConfig
    ) -> NDArray[np.bool_]:
        """应用水平飞行动作；越界者保持原位，返回每架 UAV 的越界标志。"""
        candidates = ( #计算候选位置
            self.positions[:, :2] + actions * config.max_speed * config.flight_duration_s
        )
        in_bounds = ((candidates >= 0.0) & (candidates <= region.side_length_m)).all(axis=1) #判断候选位置是否在地图内
        self.positions[in_bounds, :2] = candidates[in_bounds]
        return ~in_bounds

    @staticmethod
    def propulsion_power( # 计算推进功率
        speed: float | NDArray[np.float64], config: UAVConfig
    ) -> float | NDArray[np.float64]:
        """按水平速度大小（m/s）计算旋翼推进功率（W）。"""
        profile = config.profile_power_w * (1.0 + 3.0 * speed**2 / config.tip_speed**2)
        induced = config.induced_power_w * (
            np.sqrt(1.0 + speed**4 / (4.0 * config.hover_induced_speed**4))
            - speed**2 / (2.0 * config.hover_induced_speed**2)
        ) ** 0.5
        drag = (
            0.5 * config.drag_ratio * config.air_density
            * config.rotor_solidity * config.rotor_area * speed**3
        )
        return profile + induced + drag

    def flight_energy( # 计算飞行能耗
        self,
        actions: FlightActions,
        out_of_bounds: NDArray[np.bool_], #越界标志
        config: UAVConfig,
        scheduling: SchedulingConfig,
    ) -> NDArray[np.float64]:
        """按本次飞行的动作和越界结果返回各 UAV 的时隙能耗（J）。"""
        speeds = np.linalg.norm(actions * config.max_speed, axis=1) #把动作换算成实际速度大小
        speeds[out_of_bounds] = 0.0 #越界的 UAV 速度为 0
        return (
            self.propulsion_power(speeds, config) * config.flight_duration_s
            + self.propulsion_power(0.0, config) #悬停也耗能
            * (scheduling.slot_duration_s - config.flight_duration_s) #一个时隙里"飞行一段时间 + 悬停剩余时间"的总能量
        )
