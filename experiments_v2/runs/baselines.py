"""运行四个基线，保存评估指标并核对相同到达与 UAV 轨迹。"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from environment_v2.settings import Settings, load_settings
from environment_v2.simulator import Simulator
from experiments_v2.runs.metrics import EpisodeMetrics, FlightStats, summarize
from methods_v2.baselines.policies import (
    choose_greedy, choose_heft, choose_local, choose_random,
    fly_to, follow_users, plan_grid, plan_static, sample_flight,
)
from methods_v2.ers import ERS


NAMES = ("random", "local", "greedy", "heft")
POLICIES = dict(zip(NAMES, (choose_random, choose_local, choose_greedy, choose_heft)))
FLIGHTS = ("random", "hover", "follow", "grid", "static")
METRICS = (
    "on_time_rate", "average_delay_s", "covered_share",
    "covered_on_time_rate", "covered_average_delay_s",
    "uncovered_on_time_rate", "uncovered_average_delay_s",
    "covered_task_ue_share", "covered_task_manager_share", "covered_task_neighbor_share",
    "timed_out", "compute_energy_j", "tx_energy_j", "return_energy_j", "flight_energy_j",
    "invalid_placements",
)


def run_episode(settings: Settings, seed: int, episode: int, policy, flight_name: str):
    """运行完整回合，返回统一指标、逐时隙 offer 和位置副本。"""
    simulator = Simulator(settings)
    snapshot = simulator.reset(seed, episode)
    ers = ERS(simulator.router, settings)
    flight_rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 101]))
    placement_rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 102]))
    metrics = EpisodeMetrics()
    flight_metrics = FlightStats(
        settings, simulator.region.centers, simulator.region.radius_m,
    )
    targets = None
    if flight_name == "grid":
        targets = plan_grid(snapshot, settings)
    elif flight_name == "static":
        targets = plan_static(snapshot, simulator.router, simulator.region, settings)
    offers, positions = [], [snapshot.uav_positions.copy()]
    while snapshot is not None:
        placements = (policy(snapshot, ers, placement_rng) if policy is choose_random
                      else policy(snapshot, ers))
        if flight_name == "random":
            flight = sample_flight(snapshot, flight_rng)
        elif flight_name == "hover":
            flight = np.zeros((settings.uav.count, 2))
        elif flight_name == "follow":
            flight = follow_users(snapshot, settings, flight_rng)
        else:
            flight = fly_to(snapshot, targets, settings)
        before = replace(
            snapshot,
            uav_positions=snapshot.uav_positions.copy(),
            ue_positions=snapshot.ue_positions.copy(),
        )
        result, following = simulator.step(flight, placements)
        positions_after = simulator.uav.positions.copy()
        metrics.add(before, result, placements)
        flight_metrics.add(before, flight, positions_after)
        offers.append(result.offers)
        positions.append(positions_after)
        snapshot = following
    return metrics, flight_metrics, offers, np.asarray(positions)


def copy_config(source, output):
    """保存本次运行使用的环境 TOML。"""
    target = output / "config"
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.toml"):
        shutil.copy2(path, target / path.name)


def main() -> int:
    """在固定评估集运行四个基线并写入正式实验目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config")
    parser.add_argument("--pair", type=int, nargs=2, action="append", metavar=("SEED", "EPISODE"))
    parser.add_argument("--policies", nargs="+", choices=NAMES, default=NAMES)
    parser.add_argument("--flight", choices=FLIGHTS, default="random")
    options = parser.parse_args()
    pairs = options.pair if options.pair is not None else [(1, episode) for episode in range(10)]
    settings = load_settings(options.config)
    output = PROJECT_ROOT / "output/runs" / options.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    copy_config(options.config, output)
    rows = {name: [] for name in options.policies}
    flight_rows = {name: [] for name in options.policies}
    matched = True
    print(f"arrival_slots={settings.scheduling.arrival_slots}; std=population (ddof=0)")
    print("seed episode policy " + " ".join(METRICS), flush=True)
    for seed in sorted({seed for seed, _ in pairs}):
        seed_output = output / f"seed{seed}"
        (seed_output / "checkpoints").mkdir(parents=True, exist_ok=True)
        seed_rows = {name: [] for name in options.policies}
        seed_flights = {name: [] for name in options.policies}
        with (seed_output / "metrics.jsonl").open("w", encoding="utf-8") as log:
            def emit(row):
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()

            emit(dict(kind="configuration", name=options.name, command="baselines", seed=seed,
                      flight=options.flight, policies=options.policies,
                      environment={name: vars(value) for name, value in settings._asdict().items()}))
            for _, episode in (pair for pair in pairs if pair[0] == seed):
                reference_offers = reference_positions = None
                for index, name in enumerate(options.policies):
                    policy = POLICIES[name]
                    started = perf_counter()
                    metrics, flight_metrics, offers, positions = run_episode(
                        settings, seed, episode, policy, options.flight,
                    )
                    rows[name].append(metrics)
                    flight_rows[name].append(flight_metrics)
                    seed_rows[name].append(metrics)
                    seed_flights[name].append(flight_metrics)
                    report = metrics.report()
                    emit(dict(kind="baseline_episode", policy=name, seed=seed, episode=episode,
                              elapsed_s=perf_counter() - started,
                              flight_segments=flight_metrics.report(), **report))
                    print(f"{seed} {episode} {name} " + " ".join(
                        f"{report[field]:.6f}" for field in METRICS
                    ), flush=True)
                    if index == 0:
                        reference_offers, reference_positions = offers, positions
                    else:
                        same_offers = offers == reference_offers
                        same_positions = np.array_equal(positions, reference_positions)
                        matched = matched and same_offers and same_positions
                        emit(dict(kind="baseline_comparison", policy=name, seed=seed, episode=episode,
                                  offers_equal=same_offers, positions_equal=same_positions))
                        print(f"check {seed} {episode} {name}: offers={same_offers} "
                              f"uav_positions={same_positions}", flush=True)
            for name in options.policies:
                emit(dict(kind="baseline_summary", policy=name, seed=seed,
                          episodes=len(seed_rows[name]),
                          flight_segments=FlightStats.combine(seed_flights[name]),
                          **summarize(seed_rows[name])))
    summaries = {
        name: dict(flight_segments=FlightStats.combine(flight_rows[name]), **summarize(rows[name]))
        for name in options.policies
    }
    for name in options.policies:
        print(f"{name} pooled " + " ".join(f"{summaries[name][field]:.6f}" for field in METRICS))
        values = np.asarray([[item.report()[field] for field in METRICS] for item in rows[name]])
        for label, aggregate in (("mean", values.mean(axis=0)), ("std", values.std(axis=0))):
            print(f"{name} {label} " + " ".join(f"{value:.6f}" for value in aggregate))
    invalid = sum(summary["invalid_placements"] for summary in summaries.values())
    summary = dict(name=options.name, pairs=pairs, flight=options.flight, policies=summaries,
                   invalid_placements=invalid, offers_and_uav_positions_equal=matched)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"invalid_placements_total={invalid}; offers_and_uav_positions_equal={matched}")
    print(f"output={output}")
    return 0 if matched and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
