"""一次性核对完整 episode、非法分配和不同策略下的外生过程复现。"""

from dataclasses import fields
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment_v2.contracts import Snapshot
from environment_v2.settings import load_settings
from environment_v2.simulator import Simulator


def run_episode(settings, policy_seed):
    """用独立策略流随机分配和飞行，记录完整 episode 的结果与外生数据。"""
    simulator = Simulator(settings)
    snapshot = simulator.reset(7, 1)
    policy = np.random.default_rng(policy_seed)
    offers, contents, trajectory = [], [], []
    completed, total = 0, 0
    energy = np.zeros(4)
    while snapshot is not None:
        assert snapshot.slot == len(offers)
        assert snapshot.uav_positions is simulator.uav.positions
        assert snapshot.ue_positions is simulator.ue.positions
        trajectory.append(snapshot.ue_positions.copy())
        contents.append([
            (d.id, d.ue_id, d.arrival_time_s, d.num_tasks, d.edges, d.input_bits.tolist(),
             d.output_bits.tolist(), d.cpu_cycles.tolist(), d.levels.tolist())
            for d in snapshot.new_dags
        ])
        placements = []
        for dag in snapshot.new_dags:
            if dag.id in snapshot.candidates:
                candidates = snapshot.candidates[dag.id]
                placements.extend(
                    ((dag.id, node), candidates[int(policy.integers(len(candidates)))])
                    for node in range(dag.num_tasks)
                )
        policy.shuffle(placements)
        actions = policy.uniform(-1., 1., (settings.uav.count, 2))
        result, next_snapshot = simulator.step(actions, placements)
        assert result.invalid_placements == 0
        assert len(result.completed_dags) + len(result.timed_out_dags) == len(snapshot.new_dags)
        assert result.done == (snapshot.slot == settings.scheduling.arrival_slots - 1)
        assert (next_snapshot is None) == result.done
        offers.append(result.offers)
        completed += len(result.completed_dags)
        total += len(result.offers)
        energy += [result.compute_energy_j, result.tx_energy_j,
                   result.return_energy_j, result.flight_energy_j]
        snapshot = next_snapshot
    trajectory.append(simulator.ue.positions.copy())
    assert len(offers) == settings.scheduling.arrival_slots
    print(f"策略种子 {policy_seed}: {len(offers)} 时隙，{total} offers，"
          f"按时完成 {completed}，完成率 {completed / total:.6%}")
    print("计算 / 上传与前驱传输 / 回传 / 飞行能耗（J）:", energy.tolist())
    return offers, contents, np.array(trajectory)


def check_invalid(settings):
    """分别核对漏子任务和非法执行节点导致的整 DAG 超时。"""
    for case in ("漏子任务", "执行节点不在候选内"):
        simulator = Simulator(settings)
        snapshot = simulator.reset(7, 1)
        placements = [
            ((d.id, node), None) for d in snapshot.new_dags if d.id in snapshot.candidates
            for node in range(d.num_tasks)
        ]
        key, _ = placements[0]
        if case == "漏子任务":
            placements.pop(0)
        else:
            placements[0] = (key, settings.uav.count)
        result, _ = simulator.step(np.zeros((settings.uav.count, 2)), placements)
        failed = next(d for d in result.timed_out_dags if d.dag_id == key[0])
        assert result.invalid_placements == 1 and failed.finish_time_s is None
        assert result.compute_energy_j == result.tx_energy_j == result.return_energy_j == 0.
        print(f"{case}: invalid_placements={result.invalid_placements}, "
              f"DAG {failed.dag_id}=TIMED_OUT, finish_time_s={failed.finish_time_s}, 服务能耗=0 J")


def main():
    """运行随机 episode、非法动作案例及外生过程精确比较。"""
    settings = load_settings(ROOT / "config")
    assert [field.name for field in fields(Snapshot)] == [
        "slot", "uav_positions", "ue_positions", "new_dags",
        "manager_uav", "coverage", "uav_adjacency", "candidates",
    ]
    first = run_episode(settings, 1)
    second = run_episode(settings, 99)
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert np.array_equal(first[2], second[2])
    print("相同 (seed=7, episode=1)：offers、DAG 内容、UE 轨迹完全相同。")
    check_invalid(settings)


if __name__ == "__main__":
    main()
