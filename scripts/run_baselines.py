"""运行四个基线的完整 episode，打印指标并核对相同到达和 UAV 轨迹。"""

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment_v2.settings import Settings, load_settings
from environment_v2.simulator import Simulator
from methods.baselines.policies import (
    choose_greedy, choose_heft, choose_local, choose_random, sample_flight,
)
from methods.ers import ERS


METRICS = (
    "on_time_rate", "timed_out", "compute_energy_j", "tx_energy_j",
    "return_energy_j", "flight_energy_j", "invalid_placements",
)


def run_episode(settings: Settings, seed: int, episode: int, policy):
    """仅用 reset/step 运行完整 episode，返回指标、逐时隙 offer 和位置副本。"""
    simulator = Simulator(settings)
    snapshot = simulator.reset(seed, episode)
    ers = ERS(simulator.router, settings)
    flight_rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 101]))
    placement_rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 102]))
    totals = {name: 0 for name in METRICS if name != "on_time_rate"}
    offers, positions = [], [snapshot.uav_positions.copy()]
    offered = completed = 0
    while snapshot is not None:
        placements = (policy(snapshot, ers, placement_rng) if policy is choose_random
                      else policy(snapshot, ers))
        result, next_snapshot = simulator.step(sample_flight(snapshot, flight_rng), placements)
        offers.append(result.offers)
        positions.append(snapshot.uav_positions.copy())
        offered += len(result.offers)
        completed += len(result.completed_dags)
        totals["timed_out"] += len(result.timed_out_dags)
        for name in METRICS[2:]:
            totals[name] += getattr(result, name)
        snapshot = next_snapshot
    totals["on_time_rate"] = completed / offered
    return totals, offers, np.array(positions)


def main() -> int:
    """读取配置与 seed/episode 对，打印各回合及总体均值和总体标准差。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config")
    parser.add_argument("--pair", type=int, nargs=2, action="append", metavar=("SEED", "EPISODE"))
    args = parser.parse_args()
    pairs = args.pair if args.pair is not None else [(42, 0), (42, 1), (43, 0)]
    settings = load_settings(args.config)
    policies = (choose_random, choose_local, choose_greedy, choose_heft)
    names = ("random", "local", "greedy", "heft")
    rows = {name: [] for name in names}
    matched = True
    print(f"arrival_slots={settings.scheduling.arrival_slots}; std=population (ddof=0)")
    print("seed episode policy " + " ".join(METRICS), flush=True)
    for seed, episode in pairs:
        reference_offers = reference_positions = None
        for name, policy in zip(names, policies):
            metrics, offers, positions = run_episode(settings, seed, episode, policy)
            rows[name].append([metrics[field] for field in METRICS])
            print(f"{seed} {episode} {name} " + " ".join(
                f"{metrics[field]:.6f}" for field in METRICS
            ), flush=True)
            if name == "random":
                reference_offers, reference_positions = offers, positions
            else:
                same_offers = offers == reference_offers
                same_positions = np.array_equal(positions, reference_positions)
                matched = matched and same_offers and same_positions
                print(f"check {seed} {episode} {name}: offers={same_offers} "
                      f"uav_positions={same_positions}", flush=True)
    for name in names:
        values = np.array(rows[name])
        for label, aggregate in (("mean", values.mean(axis=0)), ("std", values.std(axis=0))):
            print(f"{name} {label} " + " ".join(f"{value:.6f}" for value in aggregate))
    invalid = sum(row[-1] for episodes in rows.values() for row in episodes)
    print(f"invalid_placements_total={invalid}; offers_and_uav_positions_equal={matched}")
    return 0 if matched and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
