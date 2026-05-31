from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_METRICS = (
    "episode_reward",
    "mean_step_reward",
    "dag_on_time_success_rate",
    "dag_success_rate",
    "dag_failure_rate",
    "dag_task_finish_rate",
    "dag_task_drop_rate",
    "episode_latency",
    "episode_energy",
    "action_executed_rate",
    "num_non_executed_decisions",
    "entropy",
    "critic_loss",
    "actor_loss",
)


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"warning: skipped malformed JSON at {path}:{line_no}: {exc}")
                continue
            if not isinstance(row, dict):
                print(f"warning: skipped non-object JSON at {path}:{line_no}")
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No valid training records found: {path}")
    return rows


def _infer_task_type(paths: list[Path]) -> str:
    joined = " ".join(str(path).lower() for path in paths)
    for pattern, label in (
        ("critical_plus_attribute", "critical_plus_attribute_rl"),
        ("attribute_blind", "attribute_blind_rl"),
        ("attribute_only", "attribute_only_rl"),
        ("critical_only", "critical_only_rl"),
    ):
        if pattern in joined:
            return label
    return "assignment_rl"


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_.-") or "assignment_rl"


def _series(rows: list[dict[str, Any]], metric: str, seed: int) -> tuple[list[float], list[float]]:
    episodes: list[float] = []
    values: list[float] = []
    missing = 0
    for index, row in enumerate(rows, 1):
        value = _safe_float(row.get(metric))
        if value is None:
            missing += 1
            continue
        episode = _safe_float(row.get("episode"))
        episodes.append(float(index) if episode is None else episode)
        values.append(value)
    if not values:
        print(f"warning: seed{seed}: missing field '{metric}', skipped")
    elif missing:
        print(f"warning: seed{seed}: field '{metric}' missing or invalid in {missing} records")
    return episodes, values


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        count = min(index + 1, window)
        result.append(running_sum / float(count))
    return result


def _plot_metric(
    axis: Any,
    rows: list[dict[str, Any]],
    seed: int,
    metric: str,
    window: int,
    *,
    color: str | None = None,
    label: str | None = None,
) -> bool:
    episodes, values = _series(rows, metric, seed)
    if not values:
        return False
    line_label = label or metric
    axis.plot(episodes, values, color=color, alpha=0.14, linewidth=0.7)
    axis.plot(episodes, _rolling_mean(values, window), color=color, linewidth=1.8, label=line_label)
    return True


def _finish_axis(axis: Any, title: str, ylabel: str = "") -> None:
    axis.set_title(title)
    axis.set_xlabel("Episode")
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def _add_legend(axis: Any, extra_axis: Any | None = None) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if extra_axis is not None:
        extra_handles, extra_labels = extra_axis.get_legend_handles_labels()
        handles.extend(extra_handles)
        labels.extend(extra_labels)
    if handles:
        axis.legend(handles, labels, fontsize=8, loc="best")


def _plot_seed(
    seed: int,
    rows: list[dict[str, Any]],
    output_path: Path,
    task_type: str,
    window: int,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(19, 14))

    reward_axis = axes[0, 0]
    reward_extra = reward_axis.twinx()
    _plot_metric(reward_axis, rows, seed, "episode_reward", window, color="tab:blue")
    _plot_metric(reward_extra, rows, seed, "mean_step_reward", window, color="tab:orange")
    _finish_axis(reward_axis, "Reward Convergence", "episode_reward")
    reward_extra.set_ylabel("mean_step_reward")
    _add_legend(reward_axis, reward_extra)

    dag_axis = axes[0, 1]
    for metric, color in (
        ("dag_on_time_success_rate", "tab:green"),
        ("dag_success_rate", "tab:blue"),
        ("dag_failure_rate", "tab:red"),
    ):
        _plot_metric(dag_axis, rows, seed, metric, window, color=color)
    _finish_axis(dag_axis, "DAG Outcomes", "Rate")
    _add_legend(dag_axis)

    task_axis = axes[0, 2]
    _plot_metric(task_axis, rows, seed, "dag_task_finish_rate", window, color="tab:green")
    _plot_metric(task_axis, rows, seed, "dag_task_drop_rate", window, color="tab:red")
    _finish_axis(task_axis, "Task Execution Quality", "Rate")
    _add_legend(task_axis)

    latency_axis = axes[1, 0]
    _plot_metric(latency_axis, rows, seed, "episode_latency", window, color="tab:purple")
    _finish_axis(latency_axis, "Episode Latency", "Latency")
    _add_legend(latency_axis)

    energy_axis = axes[1, 1]
    _plot_metric(energy_axis, rows, seed, "episode_energy", window, color="tab:brown")
    _finish_axis(energy_axis, "Episode Energy", "Energy")
    _add_legend(energy_axis)

    execution_axis = axes[1, 2]
    execution_extra = execution_axis.twinx()
    _plot_metric(execution_axis, rows, seed, "action_executed_rate", window, color="tab:green")
    _plot_metric(execution_extra, rows, seed, "num_non_executed_decisions", window, color="tab:red")
    _finish_axis(execution_axis, "RL Execution Stability", "action_executed_rate")
    execution_extra.set_ylabel("num_non_executed_decisions")
    _add_legend(execution_axis, execution_extra)

    entropy_axis = axes[2, 0]
    _plot_metric(entropy_axis, rows, seed, "entropy", window, color="tab:cyan")
    _finish_axis(entropy_axis, "Policy Entropy", "Entropy")
    _add_legend(entropy_axis)

    loss_axis = axes[2, 1]
    _plot_metric(loss_axis, rows, seed, "actor_loss", window, color="tab:blue")
    _plot_metric(loss_axis, rows, seed, "critic_loss", window, color="tab:orange")
    _finish_axis(loss_axis, "PPO Losses", "Loss")
    _add_legend(loss_axis)

    ppo_axis = axes[2, 2]
    ppo_extra = ppo_axis.twinx()
    _plot_metric(ppo_axis, rows, seed, "approx_kl", window, color="tab:purple")
    _plot_metric(ppo_extra, rows, seed, "clip_fraction", window, color="tab:gray")
    _finish_axis(ppo_axis, "PPO Update Diagnostics", "approx_kl")
    ppo_extra.set_ylabel("clip_fraction")
    _add_legend(ppo_axis, ppo_extra)

    fig.suptitle(f"Assignment-only PPO Training Trends | seed{seed} | {task_type}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"saved_plot={output_path}")


def _metric_mean(rows: list[dict[str, Any]], metric: str, seed: int) -> float | None:
    values = [_safe_float(row.get(metric)) for row in rows]
    valid = [value for value in values if value is not None]
    if not valid:
        print(f"warning: seed{seed}: missing summary field '{metric}', wrote empty value")
        return None
    return float(mean(valid))


def _summary_row(seed: int, task_type: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_rows = rows[-50:]
    summary: dict[str, Any] = {
        "seed": seed,
        "task_type": task_type,
        "total_episodes": len(rows),
    }
    for metric in SUMMARY_METRICS:
        summary[f"final_50_mean_{metric}"] = _metric_mean(final_rows, metric, seed)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={path}")


def _format_value(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "NA"
    if abs(numeric) >= 1000:
        return f"{numeric:.2f}"
    return f"{numeric:.4f}"


def _window_mean(rows: list[dict[str, Any]], metric: str, size: int = 50) -> float | None:
    values = [_safe_float(row.get(metric)) for row in rows[-size:]]
    valid = [value for value in values if value is not None]
    return float(mean(valid)) if valid else None


def _trend_judgments(seed: int, rows: list[dict[str, Any]]) -> list[str]:
    first = rows[: min(50, len(rows))]
    last = rows[-min(50, len(rows)) :]

    def window_delta(metric: str) -> tuple[float | None, float | None, float | None]:
        before = _window_mean(first, metric, len(first))
        after = _window_mean(last, metric, len(last))
        return before, after, None if before is None or after is None else after - before

    reward_before, reward_after, reward_delta = window_delta("episode_reward")
    on_time_before, on_time_after, on_time_delta = window_delta("dag_on_time_success_rate")
    success_before, success_after, success_delta = window_delta("dag_success_rate")
    executed_before, executed_after, executed_delta = window_delta("action_executed_rate")
    entropy_before, entropy_after, entropy_delta = window_delta("entropy")
    executed_values = [_safe_float(row.get("action_executed_rate")) for row in last]
    executed_valid = [value for value in executed_values if value is not None]
    critic_values = [_safe_float(row.get("critic_loss")) for row in last]
    critic_valid = [value for value in critic_values if value is not None]

    reward_text = "insufficient data"
    if reward_delta is not None:
        reward_text = "increased" if reward_delta > 0 else "decreased or flat"
    dag_text = "insufficient data"
    if on_time_delta is not None and success_delta is not None:
        dag_text = "improved" if on_time_delta > 0 and success_delta > 0 else "mixed or not improved"
    executed_text = "insufficient data"
    if executed_delta is not None:
        stable_variance = bool(executed_valid) and pstdev(executed_valid) <= 0.1
        executed_text = "stable" if abs(executed_delta) <= 0.05 and stable_variance else "changed materially or remains volatile"
    entropy_text = "insufficient data"
    if entropy_before is not None and entropy_after is not None:
        entropy_text = "possible collapse" if entropy_after < 0.1 or entropy_after < 0.1 * max(entropy_before, 1e-8) else "no collapse detected"
    critic_text = "insufficient data"
    if critic_valid:
        sorted_critic = sorted(critic_valid)
        median_critic = sorted_critic[len(sorted_critic) // 2]
        spike_threshold = max(10.0, 10.0 * median_critic)
        if len(critic_valid) != len(critic_values):
            critic_text = "abnormal non-finite value detected"
        elif max(critic_valid) > spike_threshold:
            critic_text = "large spike detected"
        else:
            critic_text = "no obvious anomaly detected"

    return [
        f"- seed{seed}: reward {reward_text} ({_format_value(reward_before)} -> {_format_value(reward_after)}).",
        (
            f"- seed{seed}: DAG metrics {dag_text}; on-time success {_format_value(on_time_before)} -> "
            f"{_format_value(on_time_after)}, success {_format_value(success_before)} -> {_format_value(success_after)}."
        ),
        (
            f"- seed{seed}: action_executed_rate {executed_text} "
            f"({_format_value(executed_before)} -> {_format_value(executed_after)})."
        ),
        (
            f"- seed{seed}: entropy {entropy_text} "
            f"({_format_value(entropy_before)} -> {_format_value(entropy_after)})."
        ),
        (
            f"- seed{seed}: critic_loss {critic_text}; final-50 mean "
            f"{_format_value(_window_mean(last, 'critic_loss', len(last)))}."
        ),
    ]


def _write_report(
    path: Path,
    logs: dict[int, Path],
    plots: dict[int, Path],
    summaries: list[dict[str, Any]],
    rows_by_seed: dict[int, list[dict[str, Any]]],
) -> None:
    columns = (
        "seed",
        "final_50_mean_episode_reward",
        "final_50_mean_dag_on_time_success_rate",
        "final_50_mean_dag_success_rate",
        "final_50_mean_dag_failure_rate",
        "final_50_mean_action_executed_rate",
        "final_50_mean_entropy",
        "final_50_mean_critic_loss",
    )
    lines = [
        "# Assignment-only PPO Training Report",
        "",
        "## Input Logs",
        "",
        *[f"- seed{seed}: `{logs[seed]}`" for seed in sorted(logs)],
        "",
        "## Output Figures",
        "",
        *[f"- seed{seed}: `{plots[seed]}`" for seed in sorted(plots)],
        "",
        "## Final 50 Episodes",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(str(summary.get(column)) if column == "seed" else _format_value(summary.get(column)) for column in columns)
            + " |"
        )
    lines.extend(["", "## Trend Assessment", ""])
    for seed in sorted(rows_by_seed):
        lines.extend(_trend_judgments(seed, rows_by_seed[seed]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_report={path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot assignment-only PPO training trends for three seeds.")
    parser.add_argument("--seed42_log", required=True)
    parser.add_argument("--seed43_log", required=True)
    parser.add_argument("--seed44_log", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--smooth_window", type=int, default=20)
    parser.add_argument("--task_type", default="")
    args = parser.parse_args()

    if args.smooth_window <= 0:
        raise ValueError("--smooth_window must be greater than 0.")

    logs = {
        42: Path(args.seed42_log),
        43: Path(args.seed43_log),
        44: Path(args.seed44_log),
    }
    task_type = _sanitize_name(args.task_type or _infer_task_type(list(logs.values())))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_root) / f"{timestamp}-seed42_43_44-{task_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_seed = {seed: _load_jsonl(path) for seed, path in logs.items()}
    plots: dict[int, Path] = {}
    for seed, rows in rows_by_seed.items():
        plot_path = output_dir / f"{timestamp}-seed{seed}-{task_type}.png"
        _plot_seed(seed, rows, plot_path, task_type, args.smooth_window)
        plots[seed] = plot_path

    summaries = [_summary_row(seed, task_type, rows_by_seed[seed]) for seed in sorted(rows_by_seed)]
    _write_csv(output_dir / "assignment_rl_training_summary.csv", summaries)
    _write_report(
        output_dir / "assignment_rl_training_report.md",
        logs,
        plots,
        summaries,
        rows_by_seed,
    )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
