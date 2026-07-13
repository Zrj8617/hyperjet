import config
import numpy as np


def calculate_channel_gain(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculates channel gain based on the free-space path loss model.

    中文：根据两个节点的三维距离估算信道增益，距离越远，增益越小。
    """
    distance_sq: float = np.sum((pos1 - pos2) ** 2)
    return (config.G_CONSTS_PRODUCT) / (distance_sq + config.EPSILON)


def calculate_ue_uav_rate(channel_gain: float, num_associated_ues: int) -> float:
    """Calculates data rate between a UE and a UAV.

    中文：计算 UE 到 UAV 的传输速率；同一 UAV 服务的 UE 越多，每个 UE 分到的带宽越少。
    """
    assert num_associated_ues != 0
    bandwidth_per_ue: float = config.BANDWIDTH_EDGE / num_associated_ues
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return bandwidth_per_ue * np.log2(1 + snr)


def calculate_g2a_rate(ue_pos: np.ndarray, uav_pos: np.ndarray, num_associated_ues: int = 1) -> float:
    """Calculates G2A data rate from positions.

    中文：直接根据 UE 和 UAV 的位置，算出地面到空中的传输速率。
    """
    source = np.array([ue_pos[0], ue_pos[1], 0.0], dtype=np.float32)
    target = np.array(uav_pos, dtype=np.float32)
    return float(calculate_ue_uav_rate(calculate_channel_gain(source, target), num_associated_ues))


def calculate_uav_mbs_rate(channel_gain: float) -> float:
    """Calculates data rate between a UAV and the MBS.

    中文：根据给定信道增益计算 UAV 与宏基站之间的回传速率。
    """
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_BACKHAUL * np.log2(1 + snr)


def calculate_uav_uav_rate(channel_gain: float) -> float:
    """Calculates data rate between two UAVs.

    中文：根据给定信道增益计算两架 UAV 之间的通信速率。
    """
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_INTER * np.log2(1 + snr)


def is_a2a_link_available(pos_a: np.ndarray, pos_b: np.ndarray) -> bool:
    """Returns whether a single-hop A2A link is available under the phase-one range rule.

    中文：判断两架 UAV 的距离是否在单跳空空通信范围内。
    """
    distance = float(np.linalg.norm(np.array(pos_a, dtype=np.float32) - np.array(pos_b, dtype=np.float32)))
    return distance <= config.A2A_MAX_RANGE


def calculate_a2a_rate(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Calculates single-hop A2A data rate from positions; returns 0 if unavailable.

    中文：根据两架 UAV 的位置计算单跳速率；距离超出范围时直接返回 0。
    """
    if not is_a2a_link_available(pos_a, pos_b):
        return 0.0
    source = np.array(pos_a, dtype=np.float32)
    target = np.array(pos_b, dtype=np.float32)
    return float(calculate_uav_uav_rate(calculate_channel_gain(source, target)))


def calculate_a2a_transfer_time(data_size: float, pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Calculates A2A transfer time with fixed phase-one control overhead.

    中文：计算 UAV 间传完指定数据所需的时间，并加上固定的链路控制开销。
    """
    rate = calculate_a2a_rate(pos_a, pos_b)
    if rate <= 0.0:
        return float("inf")
    return float(data_size / rate + config.A2A_CTRL_OVERHEAD)


def clean_distance_2d(pos_a, pos_b) -> float:
    """Return 2D Euclidean distance in meters for clean mainline links.

    中文：只取平面坐标，计算两个服务位置之间的直线距离，单位为米。
    """
    a = np.asarray(pos_a, dtype=np.float32).reshape(-1)[:2]
    b = np.asarray(pos_b, dtype=np.float32).reshape(-1)[:2]
    return float(np.linalg.norm(a - b))


def clean_distance_factor(distance_m: float) -> float:
    """Clean mainline distance attenuation: 1 / (1 + (d / 100)^2).

    中文：把距离换成 0 到 1 之间的衰减系数，距离越大，系数越接近 0。
    """
    distance = max(float(distance_m), 0.0)
    return float(1.0 / (1.0 + (distance / 100.0) ** 2))


def clean_effective_rate_mbps(base_bandwidth_mbps: float, distance_m: float) -> float:
    """Return clean effective rate in Mbps.

    中文：用基础带宽乘以距离衰减，得到实际可用速率；结果必须大于 0。
    """
    rate = float(base_bandwidth_mbps) * clean_distance_factor(distance_m)
    if rate <= 0.0:
        raise ValueError(
            f"Clean effective rate must be positive: base={base_bandwidth_mbps}, distance={distance_m}"
        )
    return float(rate)


def clean_transmission_time_seconds(
    data_size_mb: float,
    base_bandwidth_mbps: float,
    distance_m: float,
) -> float:
    """Return clean transmission time in seconds: data_MB * 8 / effective_rate_Mbps.

    中文：按“数据量除以有效速率”计算传输秒数，输入数据量使用 MB。
    """
    effective_rate = clean_effective_rate_mbps(base_bandwidth_mbps, distance_m)
    return float(float(data_size_mb) * 8.0 / effective_rate)
