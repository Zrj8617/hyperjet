"""时隙、到达窗口与相对截止参数的 TOML 读取。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class SchedulingConfig:
    """时隙、到达窗口与相对截止参数。"""

    slot_duration_s: float
    arrival_slots: int
    relative_deadline_slots: int

    @classmethod
    def from_toml(cls, path: str | Path) -> "SchedulingConfig":
        """从 TOML 的 scheduling 段直接构造配置。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)["scheduling"]
        return cls(**data)
