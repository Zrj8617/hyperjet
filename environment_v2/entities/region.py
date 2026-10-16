"""圆形热点的数据、采样与区域判定。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..settings import RegionConfig


@dataclass
class Region:
    """多个等半径圆形热点；中心为 K×2 水平坐标，半径单位为米。"""

    centers: NDArray[np.float64] # 存储 K 个热点的中心坐标
    radius_m: float #所有热点共用的半径

    @classmethod
    def create(cls, config: RegionConfig, rng: np.random.Generator) -> "Region":
        """采样热点个数及完全位于地图内、互不重叠的圆形热点。"""
        count = int(rng.integers(config.min_hotspots, config.max_hotspots + 1)) #从区间均匀随机抽取热点个数
        centers = np.empty((count, 2))
        for index in range(count): #逐个生成热点
            while True: #采用拒绝采样
                center = rng.uniform(
                    config.hotspot_radius_m,
                    config.side_length_m - config.hotspot_radius_m,
                    size=2,
                )
                distances = np.linalg.norm(centers[:index] - center, axis=1) #算一下候选热点中心和已有热点中心的距离是否合法
                if np.any(distances < 2.0 * config.hotspot_radius_m):
                    continue
                centers[index] = center
                break
        return cls(centers, config.hotspot_radius_m)

    def get_region_ids(self, positions: NDArray[np.float64]) -> NDArray[np.int64]: #判断用户是不是在热点内
        """按水平距离返回区域编号，热点内含边界为 1，外部为 0。"""
        distances = np.linalg.norm(
            positions[:, None, :2] - self.centers[None, :, :], axis=2
        )
        return (distances <= self.radius_m).any(axis=1).astype(np.int64)
