"""地图与圆形热点参数的 TOML 读取。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class RegionConfig:
    """地图与圆形热点参数。"""

    side_length_m: float
    min_hotspots: int
    max_hotspots: int
    hotspot_radius_m: float
    arrival_multiplier: float

    @classmethod
    def from_toml(cls, path: str | Path) -> "RegionConfig":
        """从 TOML 的 region 段直接构造配置。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)["region"]
        return cls(**data)
