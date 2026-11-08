"""从时隙快照构造局部观测、候选特征和集中式状态。"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


TASK_CYCLES = 1.5e8
BITS = 1e6
GRID_SIDE = 4


def get_feature_scales(settings):
    """集中定义固定缩放，不使用运行时统计量。"""
    return SimpleNamespace(
        users=settings.user.count,
        dags=settings.user.count,
        tasks=settings.user.count * settings.dag.max_tasks,
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

    def __init__(self, settings, ers):
        self.settings, self.ers = settings, ers
        self.scales = get_feature_scales(settings)
        self.observation_dim = 2 + (settings.uav.count - 1) * 5 + 7 + 3
        self.state_dim = settings.uav.count * self.observation_dim + 3 + GRID_SIDE ** 2 + 1
        self.task_dim = 16

    def build(self, snapshot):
        """候选和邻居负载均来自本时隙快照，不含分配后的信息。"""
        scales, count = self.scales, self.settings.uav.count
        loads = np.zeros((count, 3))
        for dag in snapshot.new_dags:
            if dag.id in snapshot.candidates:
                owner = snapshot.manager_uav[dag.ue_id]
                loads[owner] += (1, dag.num_tasks, dag.cpu_cycles.sum())
        scaled_loads = loads / (scales.dags, scales.tasks, scales.uav_cycles)
        observations = []
        for owner in range(count):
            position = snapshot.uav_positions[owner, :2]
            values = list(position / scales.position)
            for neighbor in range(count):
                if neighbor == owner:
                    continue
                if snapshot.uav_adjacency[owner, neighbor]:
                    relative = (snapshot.uav_positions[neighbor, :2] - position) / scales.position
                    values.extend((1, *relative, *scaled_loads[neighbor, 1:]))
                else:
                    values.extend((0, 0, 0, 0, 0))
            associated = (snapshot.manager_uav == owner) & snapshot.coverage[:, owner]
            relative = snapshot.ue_positions[associated, :2] - position
            centroid = relative.mean(axis=0) if len(relative) else np.zeros(2)
            quadrants = np.bincount((relative[:, 0] >= 0).astype(int) * 2
                                   + (relative[:, 1] >= 0).astype(int), minlength=4)
            proportions = quadrants / len(relative) if len(relative) else np.zeros(4)
            values.extend((len(relative) / scales.users, *(centroid / scales.associated_position),
                           *proportions, *scaled_loads[owner]))
            observations.append(values)
        observations = np.asarray(observations, dtype=np.float32)
        grid, _, _ = np.histogram2d(
            snapshot.ue_positions[:, 0], snapshot.ue_positions[:, 1],
            bins=GRID_SIDE, range=((0, scales.position), (0, scales.position)),
        )
        state = np.concatenate((observations.ravel(), (
            len(snapshot.new_dags) / scales.dags,
            (len(snapshot.new_dags) - len(snapshot.candidates)) / scales.dags,
            sum(dag.cpu_cycles.sum() for dag in snapshot.new_dags) / scales.global_cycles,
        ), grid.ravel() / scales.users, (
            (self.settings.scheduling.arrival_slots - snapshot.slot) / self.settings.scheduling.arrival_slots,
        ))).astype(np.float32)
        ranks = self.ers.rank_tasks(snapshot)
        keys, owners, task_features, masks = [], [], [], []
        for dag in snapshot.new_dags:
            if dag.id not in snapshot.candidates:
                continue
            owner = int(snapshot.manager_uav[dag.ue_id])
            has_parent = {child for _, child in dag.edges}
            has_child = {parent for parent, _ in dag.edges}
            for task in range(dag.num_tasks):
                entry, sink = task not in has_parent, task not in has_child
                candidates = np.zeros((count + 1, self.task_dim), dtype=np.float32)
                mask = np.zeros(count + 1, dtype=bool)
                common = (
                    dag.cpu_cycles[task] / scales.task_cycles, dag.output_bits[task] / scales.bits,
                    float(entry), float(sink), ranks[dag.id, task] / scales.duration,
                    dag.input_bits[task] / scales.bits if entry else 0,
                    dag.cpu_cycles.sum() / scales.dag_cycles, dag.num_tasks / scales.dag_tasks,
                )
                for node in snapshot.candidates[dag.id]:
                    slot = 0 if node is None else node + 1
                    frequency = (self.settings.user.cpu_frequency_hz if node is None
                                 else self.settings.uav.cpu_frequency_hz)
                    load = (0, 0) if node is None else scaled_loads[node, 1:]
                    candidates[slot] = (*common, float(node is None), float(node == owner),
                        frequency / scales.frequency, *load,
                        self.ers.compute_duration(dag.cpu_cycles[task], node) / scales.duration,
                        self.ers.upload_duration(snapshot, dag.ue_id, node, dag.input_bits[task])
                        / scales.duration if entry else 0,
                        self.ers.return_duration(snapshot, dag.ue_id, node, dag.output_bits[task])
                        / scales.duration if sink else 0)
                    mask[slot] = True
                keys.append((dag.id, task))
                owners.append(owner)
                task_features.append(candidates)
                masks.append(mask)
        return Features(observations, state,
                        np.asarray(task_features, dtype=np.float32).reshape(-1, count + 1, self.task_dim),
                        np.asarray(masks, dtype=bool).reshape(-1, count + 1),
                        np.asarray(owners, dtype=np.int64), keys, ranks)
