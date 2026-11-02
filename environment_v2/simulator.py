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

    def _make_snapshot(self, slot: int) -> Snapshot:
        dags = self.arrival.generate(slot, self.ue.positions)
        manager_uav = np.array([
            self.router.nearest_uav(position, self.uav.positions) for position in self.ue.positions
        ], dtype=np.int64)
        coverage = np.array([
            [self.router.can_upload(ue, uav) for uav in self.uav.positions]
            for ue in self.ue.positions
        ], dtype=bool)
        adjacency = np.array([
            [self.router.in_air_range(source, target) for target in self.uav.positions]
            for source in self.uav.positions
        ], dtype=bool)
        candidates = {}
        for dag in dags:
            manager = int(manager_uav[dag.ue_id])
            if coverage[dag.ue_id, manager]:
                neighbors = tuple(
                    int(index) for index in np.flatnonzero(adjacency[manager]) if index != manager
                )
                candidates[dag.id] = (None, manager, *neighbors)
        return Snapshot(
            slot, self.uav.positions, self.ue.positions, dags,
            manager_uav, coverage, adjacency, candidates,
        )

    def _validate_placements(self, placements: Placements) -> tuple[Placements, set[int], int]:
        snapshot = self.snapshot
        dag_ids = {dag.id for dag in snapshot.new_dags}
        expected = {
            (dag.id, node) for dag in snapshot.new_dags if dag.id in snapshot.candidates
            for node in range(dag.num_tasks)
        }
        seen = set()
        invalid_dags = set()
        errors = 0
        for key, execution_node in placements:
            dag_id, _ = key
            if key not in expected:
                errors += 1
                if dag_id in dag_ids:
                    invalid_dags.add(dag_id)
                continue
            if key in seen or execution_node not in snapshot.candidates[dag_id]:
                errors += 1
                invalid_dags.add(dag_id)
            seen.add(key)
        missing = expected - seen
        errors += len(missing)
        invalid_dags.update(dag_id for dag_id, _ in missing)
        valid = [
            (key, node) for key, node in placements
            if key in expected and key[0] not in invalid_dags
        ]
        for dag in snapshot.new_dags:
            if dag.id not in snapshot.candidates and dag.id not in invalid_dags:
                valid.extend(((dag.id, node), None) for node in range(dag.num_tasks))
        return valid, invalid_dags, errors

    def step(
        self, flight_actions: FlightActions, placements: Placements,
    ) -> tuple[SlotResult, Snapshot | None]:
        """校验并执行本时隙动作，移动实体，返回物理结果和下一时隙快照。"""
        if flight_actions.shape != (self.settings.uav.count, 2):
            raise ValueError("飞行动作形状必须为 (UAV 数量, 2)")
        if not np.all((flight_actions >= -1.0) & (flight_actions <= 1.0)):
            raise ValueError("飞行动作各分量必须位于 [-1, 1]")
        valid, invalid_dags, errors = self._validate_placements(placements)
        snapshot = self.snapshot
        start = snapshot.slot * self.settings.scheduling.slot_duration_s
        runtime = EventRuntime(
            self.settings, snapshot.ue_positions, snapshot.uav_positions,
            start, start + self.settings.scheduling.deadline_s,
        )
        records = runtime.run([
            (dag, int(snapshot.manager_uav[dag.ue_id])) for dag in snapshot.new_dags
            if dag.id not in invalid_dags
        ], valid)
        out_of_bounds = self.uav.apply_flight_actions(
            flight_actions, self.settings.uav, self.settings.region,
        )
        flight_energy = self.uav.flight_energy(
            flight_actions, out_of_bounds, self.settings.uav, self.settings.scheduling,
        )
        self.ue.move(self.settings.user, self.settings.region, self.region, self.streams.mobility)
        completed, timed_out = [], []
        for dag in snapshot.new_dags:
            if dag.id in invalid_dags:
                timed_out.append(TimedOutDAG(dag.id, dag.arrival_time_s, None))
            elif records[dag.id].status == DAGStatus.COMPLETED:
                completed.append(CompletedDAG(
                    dag.id, dag.arrival_time_s, records[dag.id].finish_time_s,
                ))
            else:
                timed_out.append(TimedOutDAG(
                    dag.id, dag.arrival_time_s, records[dag.id].finish_time_s,
                ))
        done = snapshot.slot == self.settings.scheduling.arrival_slots - 1
        result = SlotResult(
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
