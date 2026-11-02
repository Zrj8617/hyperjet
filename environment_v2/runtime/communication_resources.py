"""活动传输均分全网带宽，同一发送端均分其总功率。"""

from collections import Counter

from ..communication.channel_model import ChannelModel, LinkType
from ..communication.routing import Hop
from .channel_queue import TransferJob


def allocate_transfers(
    active: dict[Hop, TransferJob], model: ChannelModel,
    ue_power_w: float, uav_power_w: float, at: float,
) -> None:
    """更新在传跳的带宽、功率、速率及预计完成时刻。"""
    if not active:
        return
    bandwidth = model.config.total_bandwidth_hz / len(active)
    senders = {
        hop: (hop.link_type == LinkType.GROUND_TO_AIR, hop.sender) for hop in active
    }
    outgoing = Counter(senders.values())
    for hop, job in active.items():
        is_ue, _ = senders[hop]
        power = ue_power_w if is_ue else uav_power_w
        job.bandwidth_hz = bandwidth
        job.power_w = power / outgoing[senders[hop]]
        job.rate_bps = model.rate_bps(job.channel_gain, job.power_w, bandwidth)
        job.predicted_finish_s = at + job.remaining_bits / job.rate_bps
