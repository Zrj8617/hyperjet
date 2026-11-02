"""一次性核对到达复现、DAG 数据和手工场景的通信计算时间线。"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment_v2.entities.region import Region
from environment_v2.entities.user import UE
from environment_v2.random_streams import RandomStreams
from environment_v2.runtime.event_runtime import EventRuntime
from environment_v2.settings import load_settings
from environment_v2.types import DAGStatus, TaskStatus, TransferStatus
from environment_v2.workload.arrival import ArrivalProcess
from environment_v2.workload.dag_generator import DAG


class TraceRuntime(EventRuntime):
    """仅在检查脚本中打印实际资源重分配。"""

    def _reallocate_transfers(self):
        super()._reallocate_transfers()
        shares = [(j.transfer_id, j.bandwidth_hz / 1e6, j.power_w)
                  for j in self._active_transfers.values()]
        print(f"分配 t={self.now:.6f}s (跳编号, MHz, W): {shares}")


def sample_arrivals(settings, policy_seed):
    """改变外部策略随机数消耗，采集完整到达窗口及 UE 轨迹。"""
    streams = RandomStreams(7, 1)
    region = Region.create(settings.region, streams.scene)
    ue = UE.create(settings.user, settings.region, region, streams.scene)
    arrival = ArrivalProcess(settings, region, streams)
    policy = np.random.default_rng(policy_seed)
    offers, samples, trajectory, inputs, outputs = [], [], [], [], []
    for slot in range(settings.scheduling.arrival_slots):
        trajectory.append(ue.positions.copy())
        dags = arrival.generate(slot, ue.positions)
        offers.append([(d.id, d.ue_id, d.arrival_time_s) for d in dags])
        for d in dags:
            parents = np.bincount([child for _, child in d.edges], minlength=d.num_tasks)
            assert settings.dag.min_tasks <= d.num_tasks <= settings.dag.max_tasks
            assert all(p < c and d.levels[p] < d.levels[c] for p, c in d.edges)
            assert all(1 <= parents[i] <= settings.dag.max_parents
                       for i in range(d.num_tasks) if d.levels[i] > 0)
            samples.append((d.edges, d.input_bits.tolist(), d.output_bits.tolist(), d.cpu_cycles.tolist()))
            inputs.extend(d.input_bits[parents == 0])
            outputs.extend(d.output_bits)
        policy.integers(0, settings.uav.count, size=int(policy.integers(1, 50)))
        ue.move(settings.user, settings.region, region, streams.mobility)
    print(f"策略流 {policy_seed} 的完整 offer 序列: {offers}")
    print("入口输入/节点输出 bit 的 min、mean、max:",
          [(min(values), float(np.mean(values)), max(values)) for values in (inputs, outputs)])
    return offers, samples, np.array(trajectory)


def main():
    """打印并核对手工事件、共享结果和外生随机过程。"""
    settings = load_settings(ROOT / "config")
    ue = np.array([[0., 0., 0.], [10., 0., 0.], [600., 600., 0.]])
    uavs = np.array([[100., 0., 100.], [200., 0., 100.], [110., 0., 100.]])
    dags = [
        DAG(0, 0, 0., 4, [(0, 2), (0, 3), (1, 2)], np.array([8e6, 4e6, 0., 0.]),
            np.array([4e6, 2e6, 1e6, 1e6]), np.array([1e8, 2e8, 1e8, 2e8]), np.array([0, 0, 1, 1])),
        DAG(1, 1, 0., 1, [], np.array([8e6]), np.array([1e6]), np.array([1e8]), np.array([0])),
        DAG(2, 2, 0., 2, [(0, 1)], np.zeros(2), np.ones(2), np.array([8e8, 8e8]), np.array([0, 1])),
    ]
    placements = [((0, 1), 1), ((1, 0), 2), ((0, 0), 1), ((0, 2), 2),
                  ((0, 3), 2), ((2, 0), None), ((2, 1), None)]
    runtime = TraceRuntime(settings, ue, uavs, 0., settings.scheduling.deadline_s)
    result = runtime.run([(d, runtime.router.nearest_uav(ue[d.ue_id], uavs)) for d in dags], placements)
    timeline = []
    for dag_id, d in result.items():
        for node, task in d.tasks.items():
            assert task.status == TaskStatus.FINISHED
            timeline.extend([(task.start_time_s, 2, f"计算开始 {dag_id, node}"),
                             (task.finish_time_s, 0, f"计算完成 {dag_id, node}")])
        for record in d.transfers:
            for job in record.jobs:
                assert job.status == TransferStatus.FINISHED
                timeline.extend([(job.source_ready_s, 1, f"跳 {job.transfer_id} 入队 {record.purpose} {job.hop}"),
                                 (job.start_time_s, 2, f"跳 {job.transfer_id} 开始"),
                                 (job.finish_time_s, 0, f"跳 {job.transfer_id} 完成")])
        print(f"DAG {dag_id}: {d.status.value}, 完成={d.finish_time_s:.6f}s, "
              f"计算/传输/回传 J={d.compute_energy_j:.6f}/{d.tx_energy_j:.6f}/{d.return_energy_j:.6f}")
    for at, _, event in sorted(timeline, key=lambda row: row[:2]):
        print(f"t={at:.6f}s {event}")
    shared = result[0].tasks[2].predecessor_records[(0, 0)]
    assert shared is result[0].tasks[3].predecessor_records[(0, 0)]
    assert sum(r is shared for r in result[0].transfers) == 1
    assert len(result[0].tasks[0].input_record.jobs) == 2
    assert len(result[1].tasks[0].input_record.jobs) == 1
    assert result[0].tasks[1].input_record.jobs[0].finish_time_s == result[0].tasks[0].input_record.jobs[0].start_time_s
    assert result[2].status == DAGStatus.TIMED_OUT and result[2].finish_time_s == 1.6
    assert result[2].compute_energy_j == result[2].tx_energy_j == result[2].return_energy_j == 0.
    print("共享前驱只发一份；中继/直连、FIFO、超时后 1.6s 算完：通过")
    first, second = sample_arrivals(settings, 1), sample_arrivals(settings, 99)
    assert first[:2] == second[:2] and np.array_equal(first[2], second[2])
    print("同一 (seed=7, episode=1) 下完整到达、DAG 内容、UE 轨迹完全相同：通过")


if __name__ == "__main__":
    main()
