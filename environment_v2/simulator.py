"""连接到达、动作校验、时隙执行与实体移动，向方法层返回快照和物理结果。"""

import numpy as np

from .communication.channel_model import ChannelModel
from .communication.routing import Router
from .contracts import DAGOffer, FlightActions, Placements, Snapshot
from .entities.region import Region
from .entities.uav import UAV
from .entities.user import UE
from .random_streams import RandomStreams
from .runtime.event_runtime import EventRuntime
from .runtime.slot_result import CompletedDAG, SlotResult, TimedOutDAG
from .settings import Settings
from .types import DAGStatus
from .workload.arrival import ArrivalProcess


class Simulator:
    """按 reset/step 推进固定数量的时隙，每个时隙使用独立的 runtime。"""

    def __init__(self, settings: Settings):
        """保存已读取的环境配置并建立只读路由计算器。"""
        self.settings = settings
        self.router = Router(
            ChannelModel(settings.channel),
            settings.user.transmit_power_w, settings.uav.transmit_power_w,
        )
    # 初始化一个episode
    def reset(self, seed: int, episode: int) -> Snapshot:
        """重建随机流和实体，生成第零时隙到达并返回快照。"""
        self.streams = RandomStreams(seed, episode) 
        self.region = Region.create(self.settings.region, self.streams.scene) 
        self.uav = UAV.create(self.settings.uav, self.settings.region, self.streams.scene)
        self.ue = UE.create(
            self.settings.user, self.settings.region, self.region, self.streams.scene,
        )
        self.arrival = ArrivalProcess(self.settings, self.region, self.streams)
        self.snapshot = self._make_snapshot(0)
        return self.snapshot
    # 为指定时隙生成快照
    def _make_snapshot(self, slot: int) -> Snapshot: #准备当前时隙给方法层看的信息
        dags = self.arrival.generate(slot, self.ue.positions)
        manager_uav = np.array([ # 计算每个UE的管理UAV
            self.router.nearest_uav(position, self.uav.positions) for position in self.ue.positions
        ], dtype=np.int64)
        coverage = np.array([# 每个UE能否上传到每架UAV，行UE、列UAV
            [self.router.can_upload(ue, uav) for uav in self.uav.positions]
            for ue in self.ue.positions
        ], dtype=bool)
        adjacency = np.array([# 计算每个UAV之间是否能互相通信，行和列都代表UAV
            [self.router.in_air_range(source, target) for target in self.uav.positions]
            for source in self.uav.positions
        ], dtype=bool)
        candidates = {} # 为每个DAG计算其候选执行节点
        for dag in dags:
            manager = int(manager_uav[dag.ue_id]) #dag.ue_id是这个DAG所属的UE编号
            if coverage[dag.ue_id, manager]:
                neighbors = tuple(
                    int(index) for index in np.flatnonzero(adjacency[manager]) if index != manager
                )
                candidates[dag.id] = (None, manager, *neighbors) #None是本地
        return Snapshot(
            slot, self.uav.positions, self.ue.positions, dags,
            manager_uav, coverage, adjacency, candidates,
        )# 返回构造的快照，包括时隙号、UAV/UE位置、新DAG、管理UAV、覆盖矩阵、邻接矩阵、候选执行节点。方法层据此做放置决策。

    def _validate_placements(self, placements: Placements) -> tuple[Placements, set[int], int]:
        snapshot = self.snapshot
        dag_ids = {dag.id for dag in snapshot.new_dags} # 记录本时隙新到达的DAG ID集合
        expected = {
            (dag.id, node) for dag in snapshot.new_dags if dag.id in snapshot.candidates
            for node in range(dag.num_tasks)
        }
        seen = set()
        invalid_dags = set()
        errors = 0
        for key, execution_node in placements:#检查执行分配是否合法
            dag_id, _ = key
            if key not in expected: # 这是一条不该提交的子任务分配
                errors += 1
                if dag_id in dag_ids:# 判断它能否归属到本时隙某个 DAG
                    invalid_dags.add(dag_id)
                continue
            if key in seen or execution_node not in snapshot.candidates[dag_id]:
                errors += 1
                invalid_dags.add(dag_id)
            seen.add(key)
        missing = expected - seen # 找到完全没提交的任务
        errors += len(missing)
        invalid_dags.update(dag_id for dag_id, _ in missing)
        valid = [
            (key, node) for key, node in placements
            if key in expected and key[0] not in invalid_dags
        ]# 只留合法DAG的条目，保持原顺序
        for dag in snapshot.new_dags:# 覆盖外的DAG不由方法层分配，全部放UE（None）
            if dag.id not in snapshot.candidates and dag.id not in invalid_dags:
                valid.extend(((dag.id, node), None) for node in range(dag.num_tasks))
        return valid, invalid_dags, errors

    def step(# 真正执行一个时隙
        self, flight_actions: FlightActions, placements: Placements,
    ) -> tuple[SlotResult, Snapshot | None]:
        """校验并执行本时隙动作，移动实体，返回物理结果和下一时隙快照。"""
        if flight_actions.shape != (self.settings.uav.count, 2):
            raise ValueError("飞行动作形状必须为 (UAV 数量, 2)")
        if not np.all((flight_actions >= -1.0) & (flight_actions <= 1.0)):
            raise ValueError("飞行动作各分量必须位于 [-1, 1]")
        valid, invalid_dags, errors = self._validate_placements(placements) # 校验任务分配的合法性
        snapshot = self.snapshot # 获取当前时隙的快照
        start = snapshot.slot * self.settings.scheduling.slot_duration_s #计算当前时隙的起始时间
        runtime = EventRuntime(# 为当前时隙建立一个执行器
            self.settings, snapshot.ue_positions, snapshot.uav_positions,
            start, start + self.settings.scheduling.deadline_s,
        )
        records = runtime.run([# 为每个新到达的 DAG 创建一个执行记录，并运行模拟
            (dag, int(snapshot.manager_uav[dag.ue_id])) for dag in snapshot.new_dags
            if dag.id not in invalid_dags
        ], valid)
        out_of_bounds = self.uav.apply_flight_actions(
            flight_actions, self.settings.uav, self.settings.region, # 检查飞行动作是否越界，越界需要约束，有惩罚之类的
        )
        flight_energy = self.uav.flight_energy(# 计算飞行能耗
            flight_actions, out_of_bounds, self.settings.uav, self.settings.scheduling,
        )
        self.ue.move(self.settings.user, self.settings.region, self.region, self.streams.mobility) # 更新UE位置
        completed, timed_out = [], [] # 初始化两个列表：保存本时隙成功完成 DAG、超时 / 失败 DAG
        for dag in snapshot.new_dags: # 遍历本时隙新来的所有 DAG
            if dag.id in invalid_dags:# 
                timed_out.append(TimedOutDAG(dag.id, dag.arrival_time_s, None))
            elif records[dag.id].status == DAGStatus.COMPLETED:# 放置合法且competed
                completed.append(CompletedDAG(
                    dag.id, dag.arrival_time_s, records[dag.id].finish_time_s,
                ))
            else:
                timed_out.append(TimedOutDAG(
                    dag.id, dag.arrival_time_s, records[dag.id].finish_time_s,
                ))
        done = snapshot.slot == self.settings.scheduling.arrival_slots - 1
        result = SlotResult(# 构造本时隙的执行结果，包括新到达的 DAG、成功完成的 DAG、超时的 DAG、能耗统计、越界飞行次数和是否结束
            offers=tuple(DAGOffer(dag.id, dag.ue_id, dag.arrival_time_s) for dag in snapshot.new_dags),
            completed_dags=tuple(completed), timed_out_dags=tuple(timed_out),
            compute_energy_j=sum(record.compute_energy_j for record in records.values()),
            tx_energy_j=sum(record.tx_energy_j for record in records.values()),
            return_energy_j=sum(record.return_energy_j for record in records.values()),
            flight_energy_j=float(flight_energy.sum()), invalid_placements=errors, done=done,
        )
        if done:
            return result, None
        self.snapshot = self._make_snapshot(snapshot.slot + 1)
        return result, self.snapshot
    
# 1. 校验动作（飞行动作 + 放置方案）
# 2. 拿当前状态快照，启动事件引擎跑 DAG 通信计算（非法放置任务直接不跑）
# 3. 更新无人机位置、计算飞行能耗
# 4. 更新地面 UE 位置
# 5. 统计 DAG 成功 / 失败
# 6. 汇总全部能耗、错误信息，打包 SlotResult
# 7. 判断是否仿真结束；没结束就生成下一时隙快照返回