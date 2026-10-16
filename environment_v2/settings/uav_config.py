"""同构单核 UAV 与旋翼推进参数的 TOML 读取。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class UAVConfig:
    """同构单核 UAV 与旋翼推进参数。"""

    count: int
    altitude_m: float
    max_speed: float
    flight_duration_s: float
    cpu_frequency_hz: float
    kappa: float
    transmit_power_w: float
    queue_limit: int
    profile_power_w: float
    induced_power_w: float
    tip_speed: float
    hover_induced_speed: float
    drag_ratio: float
    air_density: float
    rotor_solidity: float
    rotor_area: float

    @classmethod
    def from_toml(cls, path: str | Path) -> "UAVConfig":
        """从 TOML 的 uav 段直接构造配置。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)["uav"]
        return cls(**data)
