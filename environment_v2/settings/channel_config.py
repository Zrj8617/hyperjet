"""子信道、噪声、上传覆盖与 LoS/NLoS 传播参数的 TOML 读取。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ChannelConfig:
    """子信道、噪声、上传覆盖与 LoS/NLoS 传播参数。"""

    subchannel_bandwidth_hz: float
    subchannels_per_uav: int
    reference_channel_gain: float
    nlos_attenuation_factor: float
    noise_density_dbm_per_hz: float
    min_received_power_dbm: float
    los_alpha: float
    los_beta: float

    @classmethod
    def from_toml(cls, path: str | Path) -> "ChannelConfig":
        """从 TOML 的 channel 段直接构造配置。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)["channel"]
        return cls(**data)
