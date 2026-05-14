from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _series(rows: list[dict], key: str) -> list[float]:
    return [float(row.get(key, 0.0)) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAPPO phase-one training metrics.")
    parser.add_argument("--log_json", type=str, default="")
    parser.add_argument("--tag", type=str, default="rl_mid_ckpt_seed43_2026-05-06")
    parser.add_argument("--log_dir", type=str, default="train_logs/mappo_short")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    log_json = Path(args.log_json) if args.log_json else Path(args.log_dir) / f"log_data_{args.tag}.json"
    if not log_json.exists():
        raise FileNotFoundError(f"MAPPO log JSON not found: {log_json}")

    rows = json.loads(log_json.read_text())
    if not rows:
        raise ValueError(f"MAPPO log JSON is empty: {log_json}")

    output = Path(args.output) if args.output else Path("/data2/zrj2025/各类文档") / f"{args.tag}_metrics.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    episodes = _series(rows, "episode")
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes = axes.flatten()

    plots = [
        ("reward", "Episode reward"),
        ("critic_loss", "Critic loss"),
        ("dag_success_rate", "DAG success rate"),
        ("dag_on_time_success_rate", "DAG on-time success rate"),
        ("dag_task_finish_rate", "Task finish rate"),
        ("dag_task_drop_rate", "Task drop rate"),
    ]
    for ax, (key, title) in zip(axes, plots):
        values = _series(rows, key)
        ax.plot(episodes, values, marker="o", linewidth=1.2, markersize=3, label=key)
        if args.window > 1 and len(values) >= args.window:
            smooth_x = episodes[args.window - 1 :]
            smooth_y = [
                sum(values[i - args.window + 1 : i + 1]) / float(args.window)
                for i in range(args.window - 1, len(values))
            ]
            ax.plot(smooth_x, smooth_y, linewidth=2.0, label=f"{args.window}-episode mean")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f"MAPPO phase-one training metrics: {args.tag}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
