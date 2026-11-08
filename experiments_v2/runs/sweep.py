"""并行扫描到达概率与带宽，保存四个基线的逐回合结果并打印汇总。"""

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from environment_v2.entities.region import Region
from environment_v2.entities.uav import UAV
from environment_v2.entities.user import UE
from environment_v2.random_streams import RandomStreams
from environment_v2.settings import Settings, load_settings
from environment_v2.workload.arrival import ArrivalProcess
from experiments_v2.runs.baselines import run_episode
from methods_v2.baselines.policies import choose_random, choose_local, choose_greedy, choose_heft


POLICIES = (choose_random, choose_local, choose_greedy, choose_heft)
NAMES = ("random", "local", "greedy", "heft")
FIELDS = (
    "arrival_probability", "bandwidth_mhz", "seed", "episode", "policy",
    "on_time_rate", "average_delay_s", "covered_share",
    "covered_on_time_rate", "covered_average_delay_s",
    "uncovered_on_time_rate", "uncovered_average_delay_s",
    "covered_task_ue_share", "covered_task_manager_share", "covered_task_neighbor_share",
    "timed_out", "offers", "compute_energy_j", "tx_energy_j",
    "return_energy_j", "flight_energy_j", "invalid_placements",
)


def measure_load(settings: Settings, seed: int, episode: int) -> float:
    """只生成场景、移动和到达，以每时隙计算量除以 UAV 总主频。"""
    streams = RandomStreams(seed, episode)
    region = Region.create(settings.region, streams.scene)
    # UAV 初始位置也消耗场景流，保持 UE 初始化与 reset 一致。
    UAV.create(settings.uav, settings.region, streams.scene)
    users = UE.create(settings.user, settings.region, region, streams.scene)
    arrival = ArrivalProcess(settings, region, streams)
    cycles = 0.0
    for slot in range(settings.scheduling.arrival_slots):
        cycles += sum(float(dag.cpu_cycles.sum()) for dag in arrival.generate(slot, users.positions))
        users.move(settings.user, settings.region, region, streams.mobility)
    return cycles / settings.scheduling.arrival_slots / (settings.uav.count * settings.uav.cpu_frequency_hz)


def run_group(job):
    """对一个参数组合与 seed 运行四个基线，并独立计算外生负载。"""
    settings, probability, bandwidth, seed = job
    settings = settings._replace(
        user=replace(settings.user, arrival_probability=probability),
        channel=replace(settings.channel, total_bandwidth_hz=bandwidth),
    )
    rows = []
    for name, policy in zip(NAMES, POLICIES):
        metrics, _, _ = run_episode(settings, seed, 0, policy)
        report = metrics.report()
        rows.append(dict(arrival_probability=probability, bandwidth_mhz=bandwidth / 1e6,
                         seed=seed, episode=0, policy=name,
                         **{field: report[field] for field in FIELDS[5:]}))
    return probability, bandwidth / 1e6, measure_load(settings, seed, 0), rows


def main():
    """按组合与 seed 并行，逐行保存 CSV，打印十组回合的均值和总体标准差。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config")
    parser.add_argument("--processes", type=int, default=mp.cpu_count())
    args = parser.parse_args()
    started = perf_counter()
    settings = load_settings(args.config)
    combinations = list(product(
        (.06, .09, .12, .15, .18, .21, .24, .27, .30, .36, .42, .48), (12e6, 20e6, 30e6),
    ))
    jobs = [(settings, probability, bandwidth, seed)
            for probability, bandwidth in combinations for seed in range(10)]
    output = PROJECT_ROOT / "output/runs" / args.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    config_output = output / "config"
    config_output.mkdir(exist_ok=True)
    for path in args.config.glob("*.toml"):
        shutil.copy2(path, config_output / path.name)
    logs = {}
    for seed in range(10):
        seed_output = output / f"seed{seed}"
        (seed_output / "checkpoints").mkdir(parents=True, exist_ok=True)
        logs[seed] = (seed_output / "metrics.jsonl").open("w", encoding="utf-8")
        logs[seed].write(json.dumps(dict(kind="configuration", name=args.name, command="sweep",
                                         seed=seed), ensure_ascii=False) + "\n")
    groups = defaultdict(list)
    loads = defaultdict(list)
    summaries = []
    print(f"Python={platform.python_version()} numpy={np.__version__} "
          f"CPU={mp.cpu_count()} processes={args.processes} arrival_slots={settings.scheduling.arrival_slots}", flush=True)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as file, mp.Pool(args.processes) as pool:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for index, (probability, bandwidth, load, rows) in enumerate(pool.imap_unordered(run_group, jobs), 1):
            writer.writerows(rows)
            file.flush()
            groups[probability, bandwidth].extend(rows)
            loads[probability, bandwidth].append(load)
            for row in rows:
                logs[row["seed"]].write(json.dumps(dict(kind="sweep_episode", **row),
                                                      ensure_ascii=False) + "\n")
                logs[row["seed"]].flush()
            print(f"已完成 {index}/{len(jobs)} 组", flush=True)
    print("占比与完成率单位 %；策略列为 总/覆盖内；差值单位百分点；std 为总体标准差（ddof=0）。")
    print("p bandwidth_mhz load covered_share uncovered_on_time_rate "
          "random local greedy heft greedy-random(total/covered) invalid_placements")
    for probability, bandwidth in combinations:
        key = probability, bandwidth / 1e6
        rows = groups[key]
        common = np.array([[row["covered_share"], row["uncovered_on_time_rate"]]
                           for row in rows if row["policy"] == "random"]) * 100
        values = [np.array([[row["on_time_rate"], row["covered_on_time_rate"]]
                           for row in rows if row["policy"] == name]) * 100 for name in NAMES]
        shared = " ".join(f"{column.mean():.4f}±{column.std():.4f}" for column in common.T)
        rates = " ".join("/".join(f"{column.mean():.4f}±{column.std():.4f}"
                                 for column in value.T) for value in values)
        differences = values[2].mean(axis=0) - values[0].mean(axis=0)
        invalid = sum(row["invalid_placements"] for row in rows)
        summaries.append(dict(arrival_probability=probability, bandwidth_mhz=key[1],
                              load=float(np.mean(loads[key])), invalid_placements=invalid))
        print(f"{probability:.2f} {key[1]:.0f} {np.mean(loads[key]):.6f} {shared} {rates} "
              f"{differences[0]:+.4f}/{differences[1]:+.4f} {invalid}")
    for log in logs.values():
        log.close()
    (output / "summary.json").write_text(json.dumps(
        dict(name=args.name, combinations=summaries), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CSV={output / 'results.csv'}")
    print(f"总耗时={perf_counter()-started:.3f} s", flush=True)


if __name__ == "__main__":
    main()
