"""重放多个 S1 checkpoint，诊断飞行方向、轨迹与覆盖变化。"""

import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from environment_v2.settings import load_settings
from environment_v2.simulator import Simulator
from experiments_v2.runs.metrics import EpisodeMetrics
from methods_v2.baselines.policies import choose_heft
from methods_v2.ers import ERS
from methods_v2.mappo.features import FeatureBuilder
from methods_v2.mappo.policy import Policy


def load_policy(checkpoint_path, builder):
    """按 checkpoint 保存的有效参数建立飞行策略。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**checkpoint["args"])
    policy = Policy(builder, args)
    policy.actor.load_state_dict(checkpoint["actor"])
    policy.critic.load_state_dict(checkpoint["critic"])
    policy.actor.eval()
    policy.critic.eval()
    return policy, args


def checkpoint_label(path):
    """用实验、种子和更新编号生成不冲突的标签。"""
    return f"{path.parents[2].name}_{path.parents[1].name}_{path.stem}"


def flight_mean(policy, features):
    """读取五架 UAV 的未裁剪飞行均值。"""
    with torch.no_grad():
        return policy.actor(torch.from_numpy(features.observations)).mean.numpy()


def pair_distances(positions):
    """计算 UAV 两两水平距离。"""
    values = [np.linalg.norm(positions[left, :2] - positions[right, :2])
              for left in range(len(positions)) for right in range(left + 1, len(positions))]
    return float(np.min(values)), float(np.mean(values))


def direction_cosines(snapshot, means):
    """计算飞行均值与各自关联被覆盖 UE 重心方向的余弦。"""
    values = []
    for uav, mean in enumerate(means):
        associated = (snapshot.manager_uav == uav) & snapshot.coverage[:, uav]
        if not associated.any():
            values.append(None)
            continue
        direction = snapshot.ue_positions[associated, :2].mean(axis=0) - snapshot.uav_positions[uav, :2]
        denominator = np.linalg.norm(mean) * np.linalg.norm(direction)
        values.append(float(np.dot(mean, direction) / denominator) if denominator > 0 else None)
    return values


def summarize_case(label, log_std, slot_rows, metrics, final_positions, hotspots):
    """汇总一个 checkpoint 的固定评估回合。"""
    means = np.asarray([[uav["flight_mean"] for uav in row["uavs"]] for row in slot_rows])
    cosine_by_uav = []
    all_cosines = []
    for uav in range(means.shape[1]):
        values = [row["uavs"][uav]["flight_user_cosine"] for row in slot_rows
                  if row["uavs"][uav]["flight_user_cosine"] is not None]
        cosine_by_uav.append(float(np.mean(values)) if values else None)
        all_cosines.extend(values)
    early, late = slot_rows[:50], slot_rows[-50:]
    episode = metrics.report()

    def covered_share(rows):
        total = sum(row["new_dags"] for row in rows)
        return sum(row["covered_dags"] for row in rows) / total

    return {
        "checkpoint": label,
        "hotspots": hotspots,
        "log_std_by_uav": log_std.tolist(),
        "std_by_uav": np.exp(log_std).tolist(),
        "mean_flight_by_uav": means.mean(axis=0).tolist(),
        "flight_user_cosine_by_uav": cosine_by_uav,
        "flight_user_cosine": float(np.mean(all_cosines)),
        "out_of_bounds_uav_slot_share": float(np.mean([
            uav["out_of_bounds"] for row in slot_rows for uav in row["uavs"]
        ])),
        "boundary_lt_30m_uav_slot_share": float(np.mean([
            uav["boundary_distance_m"] < 30 for row in slot_rows for uav in row["uavs"]
        ])),
        "final_positions": final_positions[:, :2].tolist(),
        "pair_distance_m_first50": float(np.mean([row["pair_mean_distance_m"] for row in early])),
        "pair_distance_m_last50": float(np.mean([row["pair_mean_distance_m"] for row in late])),
        "covered_share_first50": covered_share(early),
        "covered_share_last50": covered_share(late),
        "covered": {"on_time_rate": episode["covered_on_time_rate"],
                    "average_delay_s": episode["covered_average_delay_s"]},
        "uncovered": {"on_time_rate": episode["uncovered_on_time_rate"],
                      "average_delay_s": episode["uncovered_average_delay_s"]},
        "task_placement_share": {
            "ue": episode["covered_task_ue_share"],
            "self": episode["covered_task_manager_share"],
            "neighbor": episode["covered_task_neighbor_share"],
        },
        "episode": episode,
    }


def run_case(settings, policy, label, raw_stream):
    """确定性重放评估 seed 1、episode 0 并写逐时隙记录。"""
    simulator = Simulator(settings)
    builder = FeatureBuilder(settings, ERS(simulator.router, settings))
    snapshot = simulator.reset(1, 0)
    metrics = EpisodeMetrics()
    hotspots = {"centers": simulator.region.centers.tolist(), "radius_m": simulator.region.radius_m}
    slot_rows, trajectory, ue_frames = [], [], {}
    while snapshot is not None:
        positions = snapshot.uav_positions.copy()
        trajectory.append(positions[:, :2].copy())
        if snapshot.slot in (0, settings.scheduling.arrival_slots - 1):
            ue_frames[snapshot.slot] = snapshot.ue_positions[:, :2].copy()
        features = builder.build(snapshot)
        means = flight_mean(policy, features)
        cosines = direction_cosines(snapshot, means)
        flight, sample, _ = policy.act(features, None, deterministic=True)
        placements = choose_heft(snapshot, builder.ers)
        candidate = positions[:, :2] + flight * settings.uav.max_speed * settings.uav.flight_duration_s
        out_of_bounds = ((candidate < 0) | (candidate > settings.region.side_length_m)).any(axis=1)
        pair_min, pair_mean = pair_distances(positions)
        associated = [int(np.sum((snapshot.manager_uav == uav) & snapshot.coverage[:, uav]))
                      for uav in range(settings.uav.count)]
        managed = [sum(dag.id in snapshot.candidates
                       and int(snapshot.manager_uav[dag.ue_id]) == uav for dag in snapshot.new_dags)
                   for uav in range(settings.uav.count)]
        covered_ids = set(snapshot.candidates)
        result, following = simulator.step(flight, placements)
        metrics.add(snapshot, result, placements)
        distances = np.linalg.norm(
            snapshot.ue_positions[:, None, :2] - simulator.region.centers[None, :, :], axis=2)
        hotspot_membership = distances <= simulator.region.radius_m
        row = {
            "checkpoint": label,
            "seed": 1,
            "episode": 0,
            "slot": snapshot.slot,
            "covered_dags": len(covered_ids),
            "new_dags": len(snapshot.new_dags),
            "pair_min_distance_m": pair_min,
            "pair_mean_distance_m": pair_mean,
            "hotspot_ue_counts": hotspot_membership.sum(axis=0).tolist(),
            "outside_hotspot_ues": int((~hotspot_membership.any(axis=1)).sum()),
            "uavs": [{
                "uav": uav,
                "position": positions[uav, :2].tolist(),
                "position_after": simulator.uav.positions[uav, :2].tolist(),
                "boundary_distance_m": float(np.min((
                    positions[uav, 0], positions[uav, 1],
                    settings.region.side_length_m - positions[uav, 0],
                    settings.region.side_length_m - positions[uav, 1],
                ))),
                "flight_mean": means[uav].tolist(),
                "flight_user_cosine": cosines[uav],
                "sampled_preclip_action": sample["flight"][uav].tolist(),
                "environment_action": flight[uav].tolist(),
                "out_of_bounds": bool(out_of_bounds[uav]),
                "associated_covered_ues": associated[uav],
                "managed_dags": managed[uav],
            } for uav in range(settings.uav.count)],
        }
        raw_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        slot_rows.append(row)
        snapshot = following
    trajectory.append(simulator.uav.positions[:, :2].copy())
    summary = summarize_case(
        label, policy.actor.log_std_values().detach().numpy(), slot_rows,
        metrics, simulator.uav.positions.copy(), hotspots,
    )
    return summary, np.asarray(trajectory), ue_frames


def draw_trajectory(output, label, trajectory, ue_frames, hotspots, side_length):
    """绘制确定性回合的 UAV 轨迹、热点与首末 UE 位置。"""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 7))
    for index, center in enumerate(hotspots["centers"]):
        axis.add_patch(plt.Circle(
            center, hotspots["radius_m"], facecolor="tab:orange", alpha=0.08,
            edgecolor="tab:orange", linestyle="--", linewidth=1.5, zorder=0,
            label=f"Hotspot r={hotspots['radius_m']:.0f} m" if index == 0 else None,
        ))
        axis.scatter(*center, marker="+", s=45, c="tab:orange", zorder=1)
        axis.annotate(f"H{index}", center, xytext=(5, 5), textcoords="offset points",
                      color="tab:orange", fontsize=9)
    axis.scatter(ue_frames[0][:, 0], ue_frames[0][:, 1], s=12, c="0.65", label="UE slot 0")
    last_slot = max(ue_frames)
    axis.scatter(ue_frames[last_slot][:, 0], ue_frames[last_slot][:, 1], s=18, marker="+",
                 c="black", label=f"UE slot {last_slot}")
    for uav in range(trajectory.shape[1]):
        line = axis.plot(trajectory[:, uav, 0], trajectory[:, uav, 1], label=f"UAV {uav}")[0]
        axis.scatter(*trajectory[0, uav], marker="o", s=45, color=line.get_color())
        axis.scatter(*trajectory[-1, uav], marker="X", s=65, color=line.get_color())
    axis.set(xlim=(0, side_length), ylim=(0, side_length), aspect="equal",
             xlabel="x / m", ylabel="y / m", title=label)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    path = output / f"trajectory_{label}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def print_table(rows):
    """打印每架 UAV 的平均飞行均值与朝向用户的余弦。"""
    print("| checkpoint | mean flight UAV0..4 | cosine UAV0..4 | overall cosine | covered share | delay/s | on-time |")
    print("|---|---|---|---:|---:|---:|---:|")
    for row in rows:
        means = "; ".join(f"({x:.4f},{y:.4f})" for x, y in row["mean_flight_by_uav"])
        cosines = ", ".join("n/a" if value is None else f"{value:.4f}"
                            for value in row["flight_user_cosine_by_uav"])
        episode = row["episode"]
        print(f"| {row['checkpoint']} | {means} | {cosines} | {row['flight_user_cosine']:.4f} | "
              f"{episode['covered_share']:.4f} | {episode['average_delay_s']:.4f} | "
              f"{episode['on_time_rate']:.4f} |")


def main():
    """一次重放多个最终 checkpoint，并覆盖固定检查目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--plot-seed", type=int, default=10)
    options = parser.parse_args()
    output = PROJECT_ROOT / "output/checks/replay"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    settings = load_settings(options.config)
    simulator = Simulator(settings)
    builder = FeatureBuilder(settings, ERS(simulator.router, settings))
    raw_path = output / "slot_records.jsonl"
    summaries, plots = [], {}
    with raw_path.open("w", encoding="utf-8") as raw_stream:
        for checkpoint_path in options.checkpoint:
            label = checkpoint_label(checkpoint_path)
            policy, args = load_policy(checkpoint_path, builder)
            torch.set_num_threads(args.training_threads)
            summary, trajectory, ue_frames = run_case(settings, policy, label, raw_stream)
            summaries.append(summary)
            if checkpoint_path.parents[1].name == f"seed{options.plot_seed}":
                plots[label] = draw_trajectory(
                    output, label, trajectory, ue_frames,
                    summary["hotspots"], settings.region.side_length_m,
                )
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(
        {"summaries": summaries, "plots": plots, "raw_records": str(raw_path)},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    print_table(summaries)
    print(f"raw={raw_path}")
    print(f"summary={summary_path}")
    for path in plots.values():
        print(f"plot={path}")


if __name__ == "__main__":
    main()
