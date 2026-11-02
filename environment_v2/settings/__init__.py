"""统一导出六类配置，并在读取入口检查跨配置关系。"""

from pathlib import Path
from typing import NamedTuple

from .channel_config import ChannelConfig
from .dag_config import DAGConfig
from .region_config import RegionConfig
from .scheduling_config import SchedulingConfig
from .uav_config import UAVConfig
from .user_config import UserConfig

__all__ = [
    "ChannelConfig", "DAGConfig", "RegionConfig", "SchedulingConfig",
    "UAVConfig", "UserConfig", "Settings", "load_settings",
]


class Settings(NamedTuple):
    """六类配置的具名集合，供调用方按职责注入。"""

    region: RegionConfig
    user: UserConfig
    uav: UAVConfig
    dag: DAGConfig
    channel: ChannelConfig
    scheduling: SchedulingConfig


def load_settings(directory: str | Path) -> Settings:
    """读取六个 TOML，并检查时长和截止长度的四项约束。"""
    directory = Path(directory)
    settings = Settings(
        region=RegionConfig.from_toml(directory / "region.toml"),
        user=UserConfig.from_toml(directory / "user.toml"),
        uav=UAVConfig.from_toml(directory / "uav.toml"),
        dag=DAGConfig.from_toml(directory / "dag.toml"),
        channel=ChannelConfig.from_toml(directory / "channel.toml"),
        scheduling=SchedulingConfig.from_toml(directory / "scheduling.toml"),
    )
    if settings.scheduling.slot_duration_s <= 0:
        raise ValueError("slot_duration_s 必须大于 0")
    if settings.uav.flight_duration_s <= 0:
        raise ValueError("flight_duration_s 必须大于 0")
    if settings.uav.flight_duration_s > settings.scheduling.slot_duration_s:
        raise ValueError("flight_duration_s 不得超过 slot_duration_s")
    if not 0 < settings.scheduling.deadline_s <= settings.scheduling.slot_duration_s:
        raise ValueError("deadline_s 必须大于 0 且不超过 slot_duration_s")
    return settings
