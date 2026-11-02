"""单时隙内按事件推进共享信道和单核计算，直到全部 DAG 做完。"""

from collections import defaultdict
from dataclasses import dataclass
import heapq
from itertools import count

import numpy as np
from numpy.typing import NDArray

from ..communication.channel_model import ChannelModel, LinkType
from ..communication.routing import Hop, Route, Router
from ..contracts import Placements
from ..settings import Settings
from ..types import DAGStatus, TaskKey, TaskStatus, TransferStatus
from ..workload.dag_generator import DAG
from .channel_queue import DirectedChannelState, TransferJob, TransferRecord
from .communication_resources import allocate_transfers
from .dag_runtime import DAGRuntime
from .server_queue import ServerState
from .task_runtime import TaskRuntime


@dataclass
class _TransferBinding:
    """一跳所属的数据搬运及最终到达后要通知的任务。"""

    record: TransferRecord
    hop_index: int
    targets: tuple[TaskKey, ...]
    predecessor: TaskKey | None


class EventRuntime:
    """每时隙一个实例；接收完整分配，只写本实例持有的运行状态。"""

    def __init__(
        self, settings: Settings,
        ue_positions: NDArray[np.float64], uav_positions: NDArray[np.float64],
        slot_start_s: float, deadline_s: float,
    ):
        """保存冻结位置、配置和绝对开始及截止时刻，建立空队列。"""
        self.settings = settings
        self.ue_positions = ue_positions
        self.uav_positions = uav_positions
        self.now = slot_start_s
        self.deadline_s = deadline_s
        self.channel_model = ChannelModel(settings.channel)
        self.router = Router(
            self.channel_model, settings.user.transmit_power_w, settings.uav.transmit_power_w,
        )
        self.servers = {
            ("uav", index): ServerState(settings.uav.cpu_frequency_hz, settings.uav.kappa)
            for index in range(len(uav_positions))
        }
        self.servers.update({
            ("ue", index): ServerState(settings.user.cpu_frequency_hz, 0.0)
            for index in range(len(ue_positions))
        })
        self.channels: dict[Hop, DirectedChannelState] = {}
        self.tasks: dict[TaskKey, TaskRuntime] = {}
        self.dag_runtimes: dict[int, DAGRuntime] = {}
        self._channel_gains: dict[Hop, float] = {}
        self._active_transfers: dict[Hop, TransferJob] = {}
        self._bindings: dict[int, _TransferBinding] = {}
        self._ready_for_server: list[TaskKey] = []
        self._events: list[tuple[float, int, str, object]] = []
        self._event_counter = count()
        self._transfer_counter = count()
        self._communication_changed = False

    def run(self, dags: list[tuple[DAG, int]], placements: Placements) -> dict[int, DAGRuntime]:
        """按完整分配跑完本时隙，返回各 DAG 记录；dags 的第二项为管理 UAV。"""
        assignments = dict(placements)
        for dag, access_uav in dags:
            runtime = DAGRuntime(dag, access_uav)
            self.dag_runtimes[dag.id] = runtime
            predecessors = {node: set() for node in range(dag.num_tasks)}
            for parent, child in dag.edges:
                predecessors[child].add((dag.id, parent))
            for node in range(dag.num_tasks):
                key = (dag.id, node)
                task = TaskRuntime(
                    key, assignments[key], dag.cpu_cycles[node], dag.input_bits[node],
                    predecessors[node],
                )
                runtime.tasks[node] = self.tasks[key] = task
        for key, _ in placements:
            task = self.tasks[key]
            if not task.predecessors:
                dag = self.dag_runtimes[key[0]].dag
                route = self.router.upload(
                    dag.ue_id, task.execution_node, task.input_bits,
                    self.ue_positions, self.uav_positions,
                )
                task.input_record = self._submit_data(route, task.input_bits, "input", (key,))
        self._dispatch()
        while self._events:
            timestamp = self._events[0][0]
            self._advance_clock(timestamp)
            while self._events and self._events[0][0] == timestamp:
                _, _, kind, payload = heapq.heappop(self._events)
                if kind == "transfer_finish":
                    self._handle_transfer_finish(payload)
                else:
                    self._handle_compute_finish(payload)
            self._dispatch()
        return self.dag_runtimes

    def _server_key(self, task: TaskRuntime) -> tuple[str, int]:
        if task.execution_node is None:
            return ("ue", self.dag_runtimes[task.key[0]].dag.ue_id)
        return ("uav", task.execution_node)

    def _channel_gain(self, hop: Hop) -> float:
        if hop not in self._channel_gains:
            tx = (self.ue_positions if hop.link_type == LinkType.GROUND_TO_AIR
                  else self.uav_positions)
            rx = (self.ue_positions if hop.link_type == LinkType.AIR_TO_GROUND
                  else self.uav_positions)
            self._channel_gains[hop] = self.channel_model.average_channel_gain(
                tx[hop.sender], rx[hop.receiver], hop.link_type,
            )
        return self._channel_gains[hop]

    def _submit_data(
        self, route: Route, bits: float, purpose: str, targets: tuple[TaskKey, ...],
        predecessor: TaskKey | None = None,
    ) -> TransferRecord:
        jobs = tuple(
            TransferJob(next(self._transfer_counter), hop, bits, self._channel_gain(hop))
            for hop in route.hops
        )
        record = TransferRecord(jobs, purpose)
        self.dag_runtimes[targets[0][0]].transfers.append(record)
        for index, job in enumerate(jobs):
            self._bindings[job.transfer_id] = _TransferBinding(record, index, targets, predecessor)
        if jobs:
            self._enqueue_transfer(jobs[0])
        else:
            record.start_time_s = record.finish_time_s = self.now
            self._deliver(record, targets, predecessor)
        return record

    def _enqueue_transfer(self, job: TransferJob) -> None:
        job.source_ready_s = self.now
        if job.hop not in self.channels:
            self.channels[job.hop] = DirectedChannelState()
        self.channels[job.hop].enqueue(job)

    def _deliver(
        self, record: TransferRecord, targets: tuple[TaskKey, ...],
        predecessor: TaskKey | None,
    ) -> None:
        if record.purpose == "return":
            self._finish_sink(targets[0][0])
            return
        for key in targets:
            task = self.tasks[key]
            if record.purpose == "input":
                task.input_arrival_s = self.now
            else:
                task.predecessor_arrival_s[predecessor] = self.now
            if task.predecessors:
                if len(task.predecessor_arrival_s) != len(task.predecessors):
                    continue
                task.ready_time_s = max(task.predecessor_arrival_s.values())
            else:
                task.ready_time_s = task.input_arrival_s
            task.status = TaskStatus.QUEUED_COMPUTE
            self._ready_for_server.append(key)

    def _advance_clock(self, at: float) -> None:
        elapsed = at - self.now
        for job in self._active_transfers.values():
            job.remaining_bits = max(0.0, job.remaining_bits - job.rate_bps * elapsed)
            if job.hop.link_type != LinkType.GROUND_TO_AIR:
                job.tx_energy_j += job.power_w * elapsed
        self.now = at

    def _handle_transfer_finish(self, transfer_id: int) -> None:
        binding = self._bindings[transfer_id]
        record = binding.record
        job = record.jobs[binding.hop_index]
        self.channels[job.hop].finish_active()
        del self._active_transfers[job.hop]
        self._communication_changed = True
        job.status = TransferStatus.FINISHED
        job.finish_time_s = self.now
        job.remaining_bits = 0.0
        job.bandwidth_hz = job.power_w = job.rate_bps = 0.0
        next_index = binding.hop_index + 1
        if next_index < len(record.jobs):
            self._enqueue_transfer(record.jobs[next_index])
        else:
            record.finish_time_s = self.now
            self._deliver(record, binding.targets, binding.predecessor)

    def _handle_compute_finish(self, server_key: tuple[str, int]) -> None:
        record = self.servers[server_key].finish()
        task = self.tasks[record.task_key]
        task.status = TaskStatus.FINISHED
        task.finish_time_s = self.now
        task.compute_energy_j = record.energy_j
        runtime = self.dag_runtimes[task.key[0]]
        dag = runtime.dag
        groups = defaultdict(list)
        for child in runtime.successors[task.key[1]]:
            successor = runtime.tasks[child]
            groups[successor.execution_node].append(successor.key)
        bits = dag.output_bits[task.key[1]]
        for destination, targets in groups.items():
            route = self.router.predecessor(
                task.execution_node, destination, dag.ue_id, bits,
                self.ue_positions, self.uav_positions,
            )
            transfer = self._submit_data(route, bits, "predecessor", tuple(targets), task.key)
            for key in targets:
                self.tasks[key].predecessor_records[task.key] = transfer
        if not runtime.successors[task.key[1]]:
            if self.settings.scheduling.result_return and task.execution_node is not None:
                route = self.router.back(
                    task.execution_node, dag.ue_id, bits, self.ue_positions, self.uav_positions,
                )
                task.return_record = self._submit_data(route, bits, "return", (task.key,))
            else:
                self._finish_sink(dag.id)

    def _finish_sink(self, dag_id: int) -> None:
        runtime = self.dag_runtimes[dag_id]
        runtime.remaining_sinks -= 1
        if runtime.remaining_sinks == 0:
            runtime.finish_time_s = self.now
            runtime.status = (
                DAGStatus.COMPLETED if self.now <= self.deadline_s else DAGStatus.TIMED_OUT
            )

    def _dispatch(self) -> None:
        for key in sorted(self._ready_for_server, key=lambda key: (self.tasks[key].ready_time_s, key)):
            task = self.tasks[key]
            self.servers[self._server_key(task)].enqueue(task)
        self._ready_for_server.clear()
        for hop in sorted(self.channels, key=lambda hop: (hop.link_type.value, hop.sender, hop.receiver)):
            job = self.channels[hop].start_head()
            if job is None:
                continue
            job.status = TransferStatus.ACTIVE
            job.start_time_s = self.now
            record = self._bindings[job.transfer_id].record
            if record.start_time_s is None:
                record.start_time_s = self.now
            self._active_transfers[hop] = job
            self._communication_changed = True
        if self._communication_changed:
            self._reallocate_transfers()
        for server_key, server in self.servers.items():
            record = server.dispatch(self.now)
            if record is None:
                continue
            task = self.tasks[record.task_key]
            task.status = TaskStatus.RUNNING
            task.start_time_s = self.now
            heapq.heappush(self._events, (
                record.finish_time_s, next(self._event_counter), "compute_finish", server_key,
            ))

    def _reallocate_transfers(self) -> None:
        allocate_transfers(
            self._active_transfers, self.channel_model,
            self.settings.user.transmit_power_w, self.settings.uav.transmit_power_w, self.now,
        )
        self._events = [event for event in self._events if event[2] != "transfer_finish"]
        self._events.extend(
            (job.predicted_finish_s, next(self._event_counter), "transfer_finish", job.transfer_id)
            for job in self._active_transfers.values()
        )
        heapq.heapify(self._events)
        self._communication_changed = False
