"""决定每个时隙哪些 UE 会产生新的 DAG，以及 DAG 什么时候到达。"""
import numpy as np
from numpy.typing import NDArray

from ..entities.region import Region
from ..random_streams import RandomStreams
from ..settings import Settings
from .dag_generator import DAG, generate_dag


class ArrivalProcess:
    """保存一个 episode 的到达流、DAG 流与递增 DAG 编号。"""

    def __init__(self, settings: Settings, region: Region, streams: RandomStreams):
        """保存配置、热点和随机流，从 DAG 编号 0 开始。"""
        self.settings = settings
        self.region = region
        self.streams = streams
        self.next_id = 0 #同一个UE多次生成DAG，也不会重复使用DAG编号

    def generate(self, slot_index: int, ue_positions: NDArray[np.float64]) -> list[DAG]:
        """逐 UE 抽样，按 UE 下标生成该时隙的新 DAG。"""
        # 同一个时隙可能有多个 UE 同时产生任务，也可能没有任何UE产生任务
        probabilities = self.settings.user.arrival_probability * np.where(
            self.region.get_region_ids(ue_positions) == 1,
            self.settings.region.arrival_multiplier, 1.0,
        )
        arrivals = self.streams.arrival.binomial(1, probabilities)# 每个UE是否产生DAG，1表示产生，0表示不产生
        at = slot_index * self.settings.scheduling.slot_duration_s
        dags = []
        for ue_id in np.flatnonzero(arrivals):
            dags.append(generate_dag(
                self.next_id, int(ue_id), at, self.settings.dag, self.streams.dag,
            ))
            self.next_id += 1
        return dags
