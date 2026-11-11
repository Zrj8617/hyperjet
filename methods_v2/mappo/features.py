"""从时隙快照构造局部观测、候选特征和集中式状态。"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


TASK_CYCLES = 1.5e8
BITS = 1e6
GRID_SIDE = 4
USER_INFO_DROP_COLUMNS = {
    "full": (),
    "quadrant": (23, 24),
    "count": (23, 24, 25, 26, 27, 28),
}


def get_feature_scales(settings):# 定义归一化尺度 （分母）
    """集中定义固定缩放，不使用运行时统计量。"""
    return SimpleNamespace(
        users=settings.user.count,
        dags=settings.user.count,
        tasks=settings.user.count * settings.dag.max_tasks,# 一个时隙内所有新 DAG 的最大子任务数量估计值
        dag_tasks=settings.dag.max_tasks,
        frequency=settings.uav.cpu_frequency_hz,
        task_cycles=TASK_CYCLES,
        bits=BITS,
        duration=settings.scheduling.deadline_s,
        position=settings.region.side_length_m,
        associated_position=settings.channel.air_range_m,
        dag_cycles=settings.user.cpu_frequency_hz * settings.scheduling.deadline_s,
        uav_cycles=settings.uav.cpu_frequency_hz * settings.scheduling.slot_duration_s,
        global_cycles=settings.uav.count * settings.uav.cpu_frequency_hz * settings.scheduling.slot_duration_s,
    )


@dataclass
class Features:
    """一个时隙的定长观测及紧凑子任务列表。"""

    observations: np.ndarray
    state: np.ndarray
    task_features: np.ndarray
    available_actions: np.ndarray
    owners: np.ndarray
    keys: list
    ranks: dict


class FeatureBuilder:
    """共享特征定义，训练和评估使用同一入口。"""

    def __init__(self, settings, ers, user_info):
        self.settings, self.ers = settings, ers
        self.scales = get_feature_scales(settings)
        # 下面这三个都没懂
        full_observation_dim = 2 + (settings.uav.count - 1) * 5 + 7 + 3
        self.observation_dim = full_observation_dim - len(USER_INFO_DROP_COLUMNS[user_info])
        self.state_dim = settings.uav.count * full_observation_dim + 3 + GRID_SIDE ** 2 + 1
        self.task_dim = 16
        self.drop_columns = USER_INFO_DROP_COLUMNS[user_info]

    def build(self, snapshot):
        """候选和邻居负载均来自本时隙快照，不含分配后的信息。"""
        scales, count = self.scales, self.settings.uav.count # 统计每架 UAV 的任务负载
        loads = np.zeros((count, 3)) # 记录管理的DAG数量，子任务数量，DAG总计算量
        for dag in snapshot.new_dags: # 遍历本时隙新生成的DAG
            if dag.id in snapshot.candidates: # 是否在候选集合中
                owner = snapshot.manager_uav[dag.ue_id] # 获取产生这个DAG的UE所对应的管理UAV
                loads[owner] += (1, dag.num_tasks, dag.cpu_cycles.sum()) # 这里统计的是 DAG 的管理归属负载，不是说这些任务最终全部在这架 UAV 上执行

        # 负载归一化
        scaled_loads = loads / (scales.dags, scales.tasks, scales.uav_cycles)
        # 构建观测
        observations = []
        for owner in range(count):
            position = snapshot.uav_positions[owner, :2] # 获取位置 取前两个坐标
            values = list(position / scales.position)
            for neighbor in range(count): #获取邻居信息
                if neighbor == owner:
                    continue
                if snapshot.uav_adjacency[owner, neighbor]:
                    relative = (snapshot.uav_positions[neighbor, :2] - position) / scales.position # 计算相对位置
                    values.extend((1, *relative, *scaled_loads[neighbor, 1:])) # 添加邻居的五个特征：是否相连，相对x，相对y，邻居任务数量负载，邻居计算量负载
                else:
                    values.extend((0, 0, 0, 0, 0))
            # 获取当前UAV管理的UE
            associated = (snapshot.manager_uav == owner) & snapshot.coverage[:, owner] # 筛选出了真正由当前 UAV 管理并且位于覆盖范围内的 UE
            relative = snapshot.ue_positions[associated, :2] - position
            centroid = relative.mean(axis=0) if len(relative) else np.zeros(2) # 计算相对位置质心，所有UE相对坐标的平均值
            # 以每架uav为中心 统计UE的分布 四象限统计
            quadrants = np.bincount((relative[:, 0] >= 0).astype(int) * 2
                                   + (relative[:, 1] >= 0).astype(int), minlength=4)
            proportions = quadrants / len(relative) if len(relative) else np.zeros(4)
            # 该UAV的所有局部信息加入观测：ue数量，ue质心位置，ue四象限分布，当前uav自身管理负载
            values.extend((len(relative) / scales.users, *(centroid / scales.associated_position),
                           *proportions, *scaled_loads[owner]))
            observations.append(values)
        full_observations = np.asarray(observations, dtype=np.float32)
        # 构建ue空间分布
        grid, _, _ = np.histogram2d(
            snapshot.ue_positions[:, 0], snapshot.ue_positions[:, 1],
         
           bins=GRID_SIDE, range=((0, scales.position), (0, scales.position)),
        )
        # 构建集中式全局状态 state
        state = np.concatenate((full_observations.ravel(), ( # 拼接所有uav的局部观测
            len(snapshot.new_dags) / scales.dags, # 统计当前时隙新产生的 DAG 数量，并归一化
            (len(snapshot.new_dags) - len(snapshot.candidates)) / scales.dags, # 非候选
            sum(dag.cpu_cycles.sum() for dag in snapshot.new_dags) / scales.global_cycles, # 当前时隙所产生的所有DAG的总计算量/理论上所有UAV的计算能力
        ), grid.ravel() / scales.users, ( # 全局ue空间分布
            (self.settings.scheduling.arrival_slots - snapshot.slot) / self.settings.scheduling.arrival_slots, #这里没看懂
        ))).astype(np.float32)
        # 计算ERS任务优先级
        ranks = self.ers.rank_tasks(snapshot)
        # 初始化子任务特征容器
        keys, owners, task_features, masks = [], [], [], [] # 子任务标识，所属UAV，子任务在各个候选执行节点上的特征，子任务允许选择的执行节点
        for dag in snapshot.new_dags:
            if dag.id not in snapshot.candidates:
                continue
            owner = int(snapshot.manager_uav[dag.ue_id])
            # 识别DAG的出入口
            has_parent = {child for _, child in dag.edges}
            has_child = {parent for parent, _ in dag.edges}
            for task in range(dag.num_tasks):
                # 子任务不是入口或者出口节点
                entry, sink = task not in has_parent, task not in has_child
                # 初始化候选节点的特征矩阵
                candidates = np.zeros((count + 1, self.task_dim), dtype=np.float32)
                mask = np.zeros(count + 1, dtype=bool)
                # 构建子任务的公共特征 均已归一化
                common = (
                    dag.cpu_cycles[task] / scales.task_cycles, dag.output_bits[task] / scales.bits,
                    float(entry), float(sink), ranks[dag.id, task] / scales.duration,
                    dag.input_bits[task] / scales.bits if entry else 0,
                    dag.cpu_cycles.sum() / scales.dag_cycles, dag.num_tasks / scales.dag_tasks,
                ) # 子任务计算量，子任务输出数据量，是否入口，是否出口，ERS排序值，子任务输入数据量 /（如果是入口），DAG总计算量，DAG总子任务数
                
                # 遍历子任务的候选执行节点
                for node in snapshot.candidates[dag.id]:
                    slot = 0 if node is None else node + 1
                    frequency = (self.settings.user.cpu_frequency_hz if node is None
                                 else self.settings.uav.cpu_frequency_hz)
                    load = (0, 0) if node is None else scaled_loads[node, 1:]
                    # 构建子任务在候选执行节点上的特征 16维
                    candidates[slot] = (*common, float(node is None), float(node == owner),
                        frequency / scales.frequency, *load,
                        self.ers.compute_duration(dag.cpu_cycles[task], node) / scales.duration, # 预计时间都是ers来算
                        self.ers.upload_duration(snapshot, dag.ue_id, node, dag.input_bits[task])
                        / scales.duration if entry else 0,
                        self.ers.return_duration(snapshot, dag.ue_id, node, dag.output_bits[task])
                        / scales.duration if sink else 0)
                    mask[slot] = True
                keys.append((dag.id, task)) # 保存子任务信息
                owners.append(owner) # 保存子任务所属UAV
                task_features.append(candidates) # 保存子任务在各个候选执行节点上的特征
                masks.append(mask)
        observations = np.delete(full_observations, self.drop_columns, axis=1)
        return Features(observations, state,
                        np.asarray(task_features, dtype=np.float32).reshape(-1, count + 1, self.task_dim),
                        np.asarray(masks, dtype=bool).reshape(-1, count + 1),
                        np.asarray(owners, dtype=np.int64), keys, ranks)
