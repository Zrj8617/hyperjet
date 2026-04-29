import config
import numpy as np


def calculate_channel_gain(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculates channel gain based on the free-space path loss model."""
    distance_sq: float = np.sum((pos1 - pos2) ** 2)
    return (config.G_CONSTS_PRODUCT) / (distance_sq + config.EPSILON)


def calculate_ue_uav_rate(channel_gain: float, num_associated_ues: int) -> float:
    """Calculates data rate between a UE and a UAV."""
    assert num_associated_ues != 0
    bandwidth_per_ue: float = config.BANDWIDTH_EDGE / num_associated_ues
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return bandwidth_per_ue * np.log2(1 + snr)


def calculate_g2a_rate(ue_pos: np.ndarray, uav_pos: np.ndarray, num_associated_ues: int = 1) -> float:
    """Calculates G2A data rate from positions."""
    source = np.array([ue_pos[0], ue_pos[1], 0.0], dtype=np.float32)
    target = np.array(uav_pos, dtype=np.float32)
    return float(calculate_ue_uav_rate(calculate_channel_gain(source, target), num_associated_ues))


def calculate_uav_mbs_rate(channel_gain: float) -> float:
    """Calculates data rate between a UAV and the MBS."""
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_BACKHAUL * np.log2(1 + snr)


def calculate_uav_uav_rate(channel_gain: float) -> float:
    """Calculates data rate between two UAVs."""
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_INTER * np.log2(1 + snr)


def is_a2a_link_available(pos_a: np.ndarray, pos_b: np.ndarray) -> bool:
    """Returns whether a single-hop A2A link is available under the phase-one range rule."""
    distance = float(np.linalg.norm(np.array(pos_a, dtype=np.float32) - np.array(pos_b, dtype=np.float32)))
    return distance <= config.A2A_MAX_RANGE


def calculate_a2a_rate(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Calculates single-hop A2A data rate from positions; returns 0 if unavailable."""
    if not is_a2a_link_available(pos_a, pos_b):
        return 0.0
    source = np.array(pos_a, dtype=np.float32)
    target = np.array(pos_b, dtype=np.float32)
    return float(calculate_uav_uav_rate(calculate_channel_gain(source, target)))


def calculate_a2a_transfer_time(data_size: float, pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Calculates A2A transfer time with fixed phase-one control overhead."""
    rate = calculate_a2a_rate(pos_a, pos_b)
    if rate <= 0.0:
        return float("inf")
    return float(data_size / rate + config.A2A_CTRL_OVERHEAD)
