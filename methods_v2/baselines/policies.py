"""四个一次性完整分配基线，以及使用独立随机流的随机飞行。"""

from itertools import permutations

import numpy as np

from environment_v2.communication.routing import Router
from environment_v2.contracts import FlightActions, Placements, Snapshot
from environment_v2.entities.region import Region
from environment_v2.settings import Settings
from methods_v2.ers import ERS, sort_placements


def choose_random(snapshot: Snapshot, ers: ERS, rng: np.random.Generator) -> Placements:
    """每个子任务在其候选元组中均匀选节点，再按 ERS 排序。"""
    placements = []
    for dag in snapshot.new_dags:
        if dag.id not in snapshot.candidates:
            continue
        candidates = snapshot.candidates[dag.id]
        for node in range(dag.num_tasks):
            placements.append(((dag.id, node), candidates[int(rng.integers(len(candidates)))]))
    return sort_placements(placements, ers.rank_tasks(snapshot))


def choose_local(snapshot: Snapshot, ers: ERS) -> Placements:
    """将全部待决策子任务放在各自 UE，再按 ERS 排序。"""
    ranks = ers.rank_tasks(snapshot)
    return sort_placements([(key, None) for key in ranks], ranks)


def choose_greedy(snapshot: Snapshot, ers: ERS) -> Placements:
    """按最大 rank 逐个 DAG 选择预计完成最早的单个执行节点。"""
    ranks = ers.rank_tasks(snapshot)
    dags = sorted(
        (dag for dag in snapshot.new_dags if dag.id in snapshot.candidates),
        key=lambda dag: (-max(ranks[dag.id, node] for node in range(dag.num_tasks)), dag.id),
    )
    uav_load = np.zeros(len(snapshot.uav_positions))
    ue_load = np.zeros(len(snapshot.ue_positions))
    placements = []
    for dag in dags:
        children = {child for _, child in dag.edges}
        parents = {parent for parent, _ in dag.edges}
        entries = [node for node in range(dag.num_tasks) if node not in children]
        sinks = [node for node in range(dag.num_tasks) if node not in parents]
        cycles = float(sum(dag.cpu_cycles))

        def estimate(node):
            upload = max(ers.upload_duration(
                snapshot, dag.ue_id, node, float(dag.input_bits[entry]),
            ) for entry in entries)
            back = max(ers.return_duration(
                snapshot, dag.ue_id, node, float(dag.output_bits[sink]),
            ) for sink in sinks)
            load = ue_load[dag.ue_id] if node is None else uav_load[node]
            return (max(upload, ers.compute_duration(load, node))
                    + ers.compute_duration(cycles, node) + back)

        selected = min(snapshot.candidates[dag.id], key=estimate)
        if selected is None:
            ue_load[dag.ue_id] += cycles
        else:
            uav_load[selected] += cycles
        placements.extend(((dag.id, node), selected) for node in range(dag.num_tasks))
    return sort_placements(placements, ranks)


def choose_heft(snapshot: Snapshot, ers: ERS) -> Placements:
    """按 ERS 逐个子任务选择预计完成最早的候选，并更新节点空闲时刻。"""
    uav_available = np.zeros(len(snapshot.uav_positions))
    ue_available = np.zeros(len(snapshot.ue_positions))
    tables = [(uav_available, ue_available)] * len(snapshot.uav_positions)
    return _choose_heft(snapshot, ers, tables)


def choose_heft_uav(snapshot: Snapshot, ers: ERS) -> Placements:
    """各管理 UAV 独立记账，用邻居管理负载初始化其空闲时刻。"""
    count = len(snapshot.uav_positions)
    loads = np.zeros(count)
    for dag in snapshot.new_dags:
        if dag.id in snapshot.candidates:
            loads[snapshot.manager_uav[dag.ue_id]] += dag.cpu_cycles.sum()
    available = loads / ers.settings.uav.cpu_frequency_hz
    tables = []
    for owner in range(count):
        uav_available = available.copy()
        uav_available[owner] = 0.0
        tables.append((uav_available, np.zeros(len(snapshot.ue_positions))))
    return _choose_heft(snapshot, ers, tables)


def _choose_heft(snapshot: Snapshot, ers: ERS, tables) -> Placements:
    """共用 ERS 决策循环，按 DAG 管理者选取节点空闲时刻表。"""
    ranks = ers.rank_tasks(snapshot)
    order = sort_placements([(key, None) for key in ranks], ranks)
    dags = {dag.id: dag for dag in snapshot.new_dags if dag.id in snapshot.candidates}
    predecessors = {key: [] for key in ranks}
    parents = set()
    for dag in dags.values():
        for parent, child in dag.edges:
            predecessors[dag.id, child].append(parent)
            parents.add((dag.id, parent))
    finishes, assignments = {}, {}
    placements = []
    for key, _ in order:
        dag_id, task = key
        dag = dags[dag_id]
        uav_available, ue_available = tables[snapshot.manager_uav[dag.ue_id]]
        estimates = []
        for node in snapshot.candidates[dag_id]:
            if predecessors[key]:
                ready = max(
                    finishes[dag_id, parent] + ers.predecessor_duration(
                        snapshot, dag.ue_id, assignments[dag_id, parent], node,
                        float(dag.output_bits[parent]),
                    ) for parent in predecessors[key]
                )
            else:
                ready = ers.upload_duration(snapshot, dag.ue_id, node, float(dag.input_bits[task]))
            available = ue_available[dag.ue_id] if node is None else uav_available[node]
            finish = max(ready, available) + ers.compute_duration(float(dag.cpu_cycles[task]), node)
            score = finish  # 回传只用于选节点，不占 CPU
            if key not in parents:
                score += ers.return_duration(snapshot, dag.ue_id, node, float(dag.output_bits[task]))
            estimates.append((node, finish, score))
        selected, finish, _ = min(estimates, key=lambda item: item[2])
        assignments[key], finishes[key] = selected, finish
        if selected is None:
            ue_available[dag.ue_id] = finish
        else:
            uav_available[selected] = finish
        placements.append((key, selected))
    return placements


def sample_flight(snapshot: Snapshot, rng: np.random.Generator) -> FlightActions:
    """用独立生成器为每架 UAV 的两个飞行分量均匀采样 [-1, 1]。"""
    return rng.uniform(-1.0, 1.0, (len(snapshot.uav_positions), 2))


def fly_to(snapshot: Snapshot, targets: np.ndarray, settings: Settings) -> FlightActions:
    """按最大水平速度飞向各自目标点。"""
    step = settings.uav.max_speed * settings.uav.flight_duration_s
    return np.clip((targets - snapshot.uav_positions[:, :2]) / step, -1.0, 1.0)


def follow_users(
    snapshot: Snapshot, settings: Settings, rng: np.random.Generator,
) -> FlightActions:
    """飞向关联且被覆盖 UE 的重心，无关联时随机飞。"""
    targets = snapshot.uav_positions[:, :2].copy()
    following = np.zeros(len(snapshot.uav_positions), dtype=bool)
    for uav in range(len(snapshot.uav_positions)):
        associated = (snapshot.manager_uav == uav) & snapshot.coverage[:, uav]
        if associated.any():
            targets[uav] = snapshot.ue_positions[associated, :2].mean(axis=0)
            following[uav] = True
    flight = sample_flight(snapshot, rng)
    flight[following] = fly_to(snapshot, targets, settings)[following]
    return flight


def _pair_targets(snapshot: Snapshot, targets: np.ndarray) -> np.ndarray:
    """枚举目标排列，选择水平总距离最小的配对。"""
    order = min(
        permutations(range(len(targets))),
        key=lambda indices: sum(
            np.linalg.norm(snapshot.uav_positions[uav, :2] - targets[target])
            for uav, target in enumerate(indices)
        ),
    )
    return targets[np.asarray(order)]


def plan_grid(snapshot: Snapshot, settings: Settings) -> np.ndarray:
    """生成四个四分点和中心点，并按最短总距离配对。"""
    side = settings.region.side_length_m
    targets = side * np.asarray([
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5),
    ])
    return _pair_targets(snapshot, targets)


def plan_static(
    snapshot: Snapshot, router: Router, region: Region, settings: Settings,
) -> np.ndarray:
    """按新增覆盖权重贪心选择五个静态目标点，再配对。"""
    coordinates = np.arange(0.0, settings.region.side_length_m + 1.0, 10.0)
    targets = np.asarray([(x, y) for y in coordinates for x in coordinates])
    positions = np.column_stack((
        targets, np.full(len(targets), settings.uav.altitude_m),
    ))
    coverage = np.asarray([
        [router.can_upload(ue, uav) for ue in snapshot.ue_positions]
        for uav in positions
    ])
    weights = np.where(
        region.get_region_ids(snapshot.ue_positions) == 1,
        settings.region.arrival_multiplier,
        1.0,
    )
    uncovered = np.ones(len(snapshot.ue_positions), dtype=bool)
    selected = []
    for _ in range(settings.uav.count):
        gains = (coverage[:, uncovered] * weights[uncovered]).sum(axis=1)
        gains[selected] = -1.0
        target = int(np.argmax(gains))
        selected.append(target)
        uncovered &= ~coverage[target]
    return _pair_targets(snapshot, targets[np.asarray(selected)])
