"""用户数量、移动与到达参数的 TOML 读取。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class UserConfig:
    """用户数量、移动与到达参数。"""

    count: int
    hotspot_density: float
    area_fluctuation: float
    hotspot_speed_scale: float
    mean_speed: float
    gm_alpha: float
    speed_sigma: float
    heading_sigma: float
    move_duration_s: float
    transmit_power_w: float
    cpu_frequency_hz: float
    arrival_probability: float

    @classmethod
    def from_toml(cls, path: str | Path) -> "UserConfig":
        """从 TOML 的 user 段直接构造配置。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)["user"]
        return cls(**data)
