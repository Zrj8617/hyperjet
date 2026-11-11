"""把多个正式实验的评估曲线和训练健康度画在同一组图中。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASELINE_COLORS = {
    "random": "0.45", "local": "tab:brown", "greedy": "tab:green", "heft": "tab:red",
}
SEGMENT_FIELDS = (
    ("covered_share", "Covered DAG share"),
    ("speed_mps", "Mean speed / (m/s)"),
    ("user_cosine", "User direction cosine"),
    ("lost_share", "Lost UAV share"),
    ("crowded_share", "Crowded UAV share"),
    ("empty_hotspot_share", "Empty hotspot share"),
)


def load_seed_rows(run):
    """按 seed 读取新格式 metrics.jsonl。"""
    rows = {}
    for seed in sorted(run.glob("seed*")):
        kinds = {}
        with (seed / "metrics.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                kinds.setdefault(row["kind"], []).append(row)
        rows[seed.name] = kinds
    return rows


def aggregate(seed_rows, kind, field):
    """按更新编号聚合多种子的标量字段。"""
    updates = sorted({row["update"] for kinds in seed_rows.values() for row in kinds[kind]})
    values = np.asarray([[next(row[field] for row in kinds[kind] if row["update"] == update)
                          for update in updates] for kinds in seed_rows.values()])
    return np.asarray(updates), values.mean(axis=0), values.std(axis=0)


def plot_series(axis, updates, mean, std, label, color, linestyle="-"):
    """单种子画线，多种子增加均值标准差阴影。"""
    axis.plot(updates, mean, label=label, color=color, linestyle=linestyle)
    if np.any(std):
        axis.fill_between(updates, mean - std, mean + std, alpha=0.16, color=color)


def add_baselines(axis, summaries, field):
    """增加四个固定评估基线的水平线。"""
    for name, values in summaries.items():
        axis.axhline(values[field], linestyle=":", linewidth=1.1,
                     color=BASELINE_COLORS[name], label=name)


def save_evaluation(output, runs, baselines, field, ylabel, filename):
    """绘制多个方案的一个评估主指标及基线。"""
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for index, (name, rows) in enumerate(runs):
        updates, mean, std = aggregate(rows, "evaluation", field)
        plot_series(axis, updates, mean, std, name, f"C{index}")
    add_baselines(axis, baselines, field)
    axis.set(xlabel="Update", ylabel=ylabel)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path = output / "figures" / filename
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_coverage(output, runs, baselines):
    """并排绘制覆盖占比与覆盖内完成率。"""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, field, title in zip(
        axes, ("covered_share", "covered_on_time_rate"),
        ("Covered DAG share", "Covered DAG on-time rate"),
    ):
        for index, (name, rows) in enumerate(runs):
            updates, mean, std = aggregate(rows, "evaluation", field)
            plot_series(axis, updates, mean, std, name, f"C{index}")
        add_baselines(axis, baselines, field)
        axis.set(xlabel="Update", ylabel=title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    path = output / "figures/evaluation_coverage.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def moving_average(values, window):
    """计算固定窗口训练回报滑动平均。"""
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_health(output, runs):
    """绘制三个方案的六项训练健康度。"""
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    fields = (
        ("value_loss", "Value loss"), ("entropy", "Entropy"),
        ("approx_kl", "Approximate KL"), ("clip_fraction", "Clip fraction"),
    )
    scalar_axes = (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
    for axis, (field, title) in zip(scalar_axes, fields):
        for index, (name, rows) in enumerate(runs):
            updates, mean, std = aggregate(rows, "update", field)
            plot_series(axis, updates, mean, std, name, f"C{index}")
        axis.set(xlabel="Update", ylabel=title)
        axis.grid(alpha=0.25)
    for index, (name, rows) in enumerate(runs):
        updates = sorted({row["update"] for kinds in rows.values() for row in kinds["update"]})
        log_std = np.asarray([[next(row["flight_log_std"] for row in kinds["update"]
                                          if row["update"] == update)
                               for update in updates] for kinds in rows.values()])
        for component, linestyle in ((0, "-"), (1, "--")):
            per_seed = log_std[:, :, :, component].mean(axis=2)
            plot_series(axes[0, 2], np.asarray(updates), per_seed.mean(axis=0),
                        per_seed.std(axis=0), f"{name} dim {component}", f"C{index}", linestyle)
        returns = np.asarray([[next(row["episode_return"] for row in kinds["update"]
                                         if row["update"] == update)
                              for update in updates] for kinds in rows.values()])
        window = min(5, len(updates))
        smoothed = np.asarray([moving_average(values, window) for values in returns])
        smooth_updates = np.asarray(updates)[window - 1:]
        plot_series(axes[1, 2], smooth_updates, smoothed.mean(axis=0), smoothed.std(axis=0),
                    name, f"C{index}")
    axes[0, 2].set(xlabel="Update", ylabel="Mean flight log_std")
    axes[0, 2].grid(alpha=0.25)
    axes[0, 2].legend(fontsize=7)
    axes[1, 2].set(xlabel="Update", ylabel="Training return moving average")
    axes[1, 2].grid(alpha=0.25)
    for axis in scalar_axes:
        axis.legend(fontsize=8)
    axes[1, 2].legend(fontsize=8)
    figure.tight_layout()
    path = output / "figures/training_health.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def load_references(paths):
    """读取单放置基线的平均时延。"""
    references = []
    for path in paths or []:
        data = json.loads(path.read_text(encoding="utf-8"))
        values = next(iter(data["policies"].values()))
        references.append((data["name"], values["average_delay_s"]))
    return references


def save_convergence(output, runs, baselines, references):
    """分开绘制训练与评估时延，并汇总末段收敛数字。"""
    figure, axes = plt.subplots(
        2, len(runs), figsize=(5.2 * len(runs), 8), sharey="row", squeeze=False,
    )
    convergence = {}
    header_printed = False
    for column, (name, seed_rows) in enumerate(runs):
        convergence[name] = {}
        for index, (seed, kinds) in enumerate(seed_rows.items()):
            color = f"C{index}"
            train_updates = np.asarray([row["update"] for row in kinds["update"]])
            train_delays = np.asarray([row["average_delay_s"] for row in kinds["update"]])
            axes[0, column].plot(
                train_updates, train_delays, color=color, linewidth=0.8, alpha=0.3,
            )
            smooth = moving_average(train_delays, 5)
            axes[0, column].plot(
                train_updates[4:], smooth, color=color, linewidth=2, label=seed,
            )

            evaluation = kinds["evaluation"]
            eval_updates = np.asarray([row["update"] for row in evaluation])
            eval_delays = np.asarray([row["average_delay_s"] for row in evaluation])
            axes[1, column].plot(
                eval_updates, eval_delays, color=color, marker="o", markersize=4, label=seed,
            )

            slope = float(np.polyfit(train_updates[-20:], train_delays[-20:], 1)[0] * 10)
            eval_values = {
                row["update"]: float(row["average_delay_s"])
                for row in evaluation[-3:]
            }
            if not header_printed:
                print(
                    "experiment\tseed\tslope_s_per_10_updates\t"
                    + "\t".join(f"eval_{update}_s" for update in eval_values)
                )
                header_printed = True
            convergence[name][seed] = {
                "slope_s_per_10_updates": slope,
                "evaluation_delay_s": {str(update): value for update, value in eval_values.items()},
            }
            print(
                f"{name}\t{seed}\t{slope:.9f}\t"
                + "\t".join(f"{value:.9f}" for value in eval_values.values())
            )

        axes[0, column].set(title=name, xlabel="Update")
        axes[0, column].grid(alpha=0.25)
        axes[0, column].legend(fontsize=8)
        axes[1, column].axhline(
            baselines["heft"]["average_delay_s"], color="0.2", linestyle="--",
            linewidth=1.1, label="heft+random",
        )
        for index, (label, value) in enumerate(references):
            axes[1, column].axhline(
                value, color=f"C{index + 3}", linestyle="--", linewidth=1.1, label=label,
            )
        axes[1, column].set(xlabel="Update")
        axes[1, column].grid(alpha=0.25)
        axes[1, column].legend(fontsize=7)
    axes[0, 0].set_ylabel("Training average delay / s")
    axes[1, 0].set_ylabel("Evaluation average delay / s")
    figure.tight_layout()
    path = output / "figures/convergence.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path, convergence


def load_segments(path):
    """读取一个基线或重放汇总，并合并同实验的 checkpoint。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "policies" in data:
        rows = next(iter(data["policies"].values()))["flight_segments"]
        return data["name"], rows
    checkpoints = data["summaries"]
    rows = []
    for index, source in enumerate(checkpoints[0]["flight_segments"]):
        row = {"start_slot": source["start_slot"]}
        for field, _ in SEGMENT_FIELDS:
            values = np.asarray([
                checkpoint["flight_segments"][index][field] for checkpoint in checkpoints
            ], dtype=float)
            row[field] = float(np.nanmean(values))
        rows.append(row)
    return data["name"], rows


def save_segments(output, paths):
    """绘制多个飞行实验的六项分段统计。"""
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")
    linestyles = ("-", "--", "-.", ":", "-", "--", "-.", ":")
    for index, path in enumerate(paths):
        label, rows = load_segments(path)
        slots = np.asarray([row["start_slot"] for row in rows])
        for axis, (field, title) in zip(axes.flat, SEGMENT_FIELDS):
            values = np.asarray([
                np.nan if row[field] is None else row[field] for row in rows
            ])
            axis.plot(
                slots, values, label=label, color=f"C{index}",
                linestyle=linestyles[index], marker=markers[index], markersize=4,
            )
    for axis, (_, title) in zip(axes.flat, SEGMENT_FIELDS):
        axis.set(xlabel="Start slot", ylabel=title)
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    path = output / "figures/segments.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main():
    """读取正式实验，绘制学习曲线或飞行分段对比。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+")
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--references", type=Path, nargs="+")
    parser.add_argument("--segments", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    figures = []
    convergence = {}
    if options.run:
        (options.output / "figures").mkdir(exist_ok=True)
        runs = [(path.name, load_seed_rows(path)) for path in options.run]
        baselines = json.loads(
            (options.baselines / "summary.json").read_text(encoding="utf-8")
        )["policies"]
        references = load_references(options.references)
        figures.extend([
            save_evaluation(options.output, runs, baselines, "average_delay_s",
                            "Evaluation average delay / s", "evaluation_delay.png"),
            save_evaluation(options.output, runs, baselines, "on_time_rate",
                            "Evaluation on-time rate", "evaluation_on_time.png"),
            save_coverage(options.output, runs, baselines),
            save_health(options.output, runs),
        ])
        convergence_path, convergence = save_convergence(
            options.output, runs, baselines, references,
        )
        figures.append(convergence_path)
    if options.segments:
        (options.output / "figures").mkdir(exist_ok=True)
        figures.append(save_segments(options.output, options.segments))
    (options.output / "summary.json").write_text(json.dumps({
        "runs": [str(path) for path in options.run] if options.run else [],
        "baselines": str(options.baselines) if options.baselines else None,
        "segments": [str(path) for path in options.segments] if options.segments else [],
        "figures": [str(path) for path in figures],
        "convergence": convergence,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()
