"""静态 DAG 数据及分层拓扑、数据量和计算量生成。"""
# 1、抽任务数 num_tasks；
# 2、抽层数 level_count，每层先占 1 个，剩余任务随机分配 → 得 levels（天然拓扑序）；
# 3、采输入数据量 input_bits、常数因子 constants、对数幂次 log_powers，按公式算 cpu_cycles；
# 4、按层连边：每个非 0 层节点从更浅层随机选 1~max_parents 个父节点；
# 5、每个节点采输出数据量 output_bits（后继共用，sink 的即回传量）；
# 6、返回 DAG。
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..settings import DAGConfig


@dataclass
class DAG:
    """一个 DAG 的静态数据；节点编号即拓扑序，不含运行状态。"""

    id: int
    ue_id: int
    arrival_time_s: float
    num_tasks: int
    edges: list[tuple[int, int]] #有向边
    input_bits: NDArray[np.float64]
    output_bits: NDArray[np.float64]
    cpu_cycles: NDArray[np.float64]
    levels: NDArray[np.int64]


def generate_dag(
    dag_id: int,
    ue_id: int,
    arrival_time_s: float,
    config: DAGConfig,
    rng: np.random.Generator,
) -> DAG:
    """用 DAG 内容流生成分层拓扑、节点输入输出数据量及计算量。"""
    num_tasks = int(rng.integers(config.min_tasks, config.max_tasks + 1)) #随机生成任务数量 上下区间自定义
    level_count = int(rng.integers(2, min(config.max_levels, num_tasks) + 1)) #随机生成层数，至少两层，最多 min(最大层数, 任务数)
    level_sizes = np.ones(level_count, dtype=np.int64)
    for _ in range(num_tasks - level_count):#随机分配剩余任务到各层
        level_sizes[rng.integers(level_count)] += 1
    levels = np.repeat(np.arange(level_count), level_sizes)
    # 为每个子任务随机生成输入数据量和计算复杂度
    input_bits = rng.uniform(config.min_input_bits, config.max_input_bits, num_tasks) #每个任务的输入数据量在范围内均匀采样
    constants = rng.uniform(config.min_task_constant, config.max_task_constant, num_tasks)#用来制造同规模节点计算量不同"的随机差异
    log_powers = rng.choice(
        3, size=num_tasks,
        p=[config.linear_probability, config.log_probability, config.squared_log_probability],
    )
    n = input_bits / config.base_unit_bits
    cpu_cycles = n * np.log2(np.maximum(n, 2.0)) ** log_powers * constants

    edges = [] #构建边
    for node in range(num_tasks):
        if levels[node] == 0:
            continue
        candidates = np.flatnonzero(levels < levels[node])#找到所有比当前节点浅的节点
        parent_count = int(rng.integers(1, min(len(candidates), config.max_parents) + 1)) # 随机选择父节点数量，至少1个，最多为浅层节点数和最大父节点数的最小值
        parents = rng.choice(candidates, size=parent_count, replace=False)# 随机选择父节点
        edges.extend((int(parent), node) for parent in sorted(parents)) # 将父节点和当前节点组成有向边，加入边列表
    output_bits = rng.uniform(config.min_output_bits, config.max_output_bits, num_tasks)
    return DAG(
        id=dag_id, ue_id=ue_id, arrival_time_s=arrival_time_s, num_tasks=num_tasks,
        edges=edges, input_bits=input_bits, output_bits=output_bits,
        cpu_cycles=cpu_cycles, levels=levels,
    )
