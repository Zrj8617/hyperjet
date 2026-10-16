"""地面用户集合的数据与高斯—马尔可夫移动。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..settings import RegionConfig, UserConfig
from .region import Region


@dataclass
class UE:
    """全部地面用户的位置、速度和方向；编号等于数组下标。"""

    positions: NDArray[np.float64]
    speeds: NDArray[np.float64]
    headings: NDArray[np.float64] #移动方位角

    @classmethod
    def create(
        cls, config: UserConfig, region_config: RegionConfig,
        region: Region, rng: np.random.Generator
    ) -> "UE":
        """用面积权重按区域分配人数、均匀撒点并采样初速度与方向。"""
        hotspot_count = len(region.centers)
        hotspot_area = np.pi * region.radius_m**2
        weights = np.empty(hotspot_count + 1)
        weights[:hotspot_count] = hotspot_area * config.hotspot_density #热点区域权重
        weights[-1] = region_config.side_length_m**2 - hotspot_count * hotspot_area #非热点区域权重
        weights *= rng.uniform( #添加随机波动（目前这个应该是可选
            1.0 - config.area_fluctuation,
            1.0 + config.area_fluctuation,
            size=len(weights),
        )
        quotas = config.count * weights / weights.sum() #配额
        counts = np.floor(quotas).astype(np.int64)
        remainder = config.count - int(counts.sum())
        counts[np.argsort(-(quotas - counts), kind="stable")[:remainder]] += 1
        positions = np.zeros((config.count, 3)) #创建用户数量*三维位置
        start = 0
        for center, count in zip(region.centers, counts[:hotspot_count]): #热点内撒点
            radii = region.radius_m * np.sqrt(rng.uniform(size=count))
            angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
            end = start + count
            positions[start:end, :2] = center + radii[:, None] * np.column_stack(
                (np.cos(angles), np.sin(angles))
            )
            start = end
        for index in range(start, config.count): #非热点内撒点
            while True:
                positions[index, :2] = rng.uniform(0.0, region_config.side_length_m, 2)
                if region.get_region_ids(positions[index:index + 1])[0] == 0:
                    break
        speeds = rng.uniform(0.0, 2.0 * config.mean_speed, config.count)
        headings = rng.uniform(0.0, 2.0 * np.pi, config.count)
        return cls(positions, speeds, headings)

    def move(#一个时间槽内用户移动一次
        self, config: UserConfig, region_config: RegionConfig,
        region: Region, rng: np.random.Generator
    ) -> None:
        """按高斯—马尔可夫模型移动一步，在地图边界镜像反弹。"""
        noise_scale = np.sqrt(1.0 - config.gm_alpha**2) #alpha控制惯性，表示速度变化程度
        self.speeds[:] = np.maximum(
            config.gm_alpha * self.speeds
            + (1.0 - config.gm_alpha) * config.mean_speed
            + noise_scale * config.speed_sigma * rng.normal(size=len(self.speeds)),
            0.0,
        )
        self.headings[:] = (#更新方向
            self.headings
            + noise_scale * config.heading_sigma * rng.normal(size=len(self.headings))
        ) % (2.0 * np.pi)
        velocities = self.speeds[:, None] * np.column_stack(
            (np.cos(self.headings), np.sin(self.headings))
        )
        speed_scales = np.where(
            region.get_region_ids(self.positions) == 1, config.hotspot_speed_scale, 1.0
        )
        self.positions[:, :2] += (
            velocities * config.move_duration_s * speed_scales[:, None]
        ) #更新位置
        for axis in range(2):#边界反弹
            coordinates = self.positions[:, axis]
            while np.any((coordinates < 0.0) | (coordinates > region_config.side_length_m)):
                below = coordinates < 0.0
                coordinates[below] *= -1.0
                velocities[below, axis] = np.abs(velocities[below, axis])
                above = coordinates > region_config.side_length_m
                coordinates[above] = 2.0 * region_config.side_length_m - coordinates[above]
                velocities[above, axis] = -np.abs(velocities[above, axis])
        moving = self.speeds > 0 #只给速度大于 0 的 UE 重算方向（边界反弹会翻转速度向量）
        self.headings[moving] = np.arctan2(
            velocities[moving, 1], velocities[moving, 0]
        ) % (2.0 * np.pi)
