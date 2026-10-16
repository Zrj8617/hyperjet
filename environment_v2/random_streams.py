"""从实验种子与回合编号派生四条独立环境随机流。"""

import numpy as np


class RandomStreams:
    """分别供场景、移动、到达和 DAG 内容使用的生成器。"""

    def __init__(self, seed: int, episode: int):
        """按固定顺序派生本回合的四条随机流。"""
        scene, mobility, arrival, dag = np.random.SeedSequence(
            [seed, episode]
        ).spawn(4)
        self.scene = np.random.default_rng(scene)
        self.mobility = np.random.default_rng(mobility)
        self.arrival = np.random.default_rng(arrival)
        self.dag = np.random.default_rng(dag)
