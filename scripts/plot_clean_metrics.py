from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TRAIN_PLOTS = [
    "train_reward.png",
    "train_completion_throughput.png",
    "train_energy_flowtime.png",
    "train_losses.png",
    "train_actions_correctness.png",
]
EVAL_PLOTS = [
    "eval_summary.png",
    "eval_per_episode_metrics.png",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot HyperUAV clean train/eval metrics.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--format", type=str, default="png", choices=["png"])
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--no-show", action="store_true", default=False)
    return parser


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path | None]:
    run_dir = Path(args.run_dir) if args.run_dir is not None else None
    metrics_jsonl = Path(args.metrics_jsonl) if args.metrics_jsonl is not None else None
    summary_json = Path(args.summary_json) if args.summary_json is not None else None
    if metrics_jsonl is None and run_dir is not None:
        train_metrics = run_dir / "train_metrics.jsonl"
        eval_metrics = run_dir / "eval_metrics.jsonl"
        if train_metrics.exists():
            metrics_jsonl = train_metrics
        elif eval_metrics.exists():
            metrics_jsonl = eval_metrics
    if summary_json is None and run_dir is not None:
        eval_summary = run_dir / "eval_summary.json"
        run_summary = run_dir / "run_summary.json"
        if eval_summary.exists():
            summary_json = eval_summary
        elif run_summary.exists():
            summary_json = run_summary
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is None:
        output_dir = (run_dir / "plots") if run_dir is not None else Path("plots")
    return {"run_dir": run_dir, "metrics_jsonl": metrics_jsonl, "summary_json": summary_json, "output_dir": output_dir}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    paths = resolve_inputs(args)
    try:
        generated = plot_clean_metrics(
            metrics_jsonl=paths["metrics_jsonl"],
            summary_json=paths["summary_json"],
            output_dir=paths["output_dir"],
            prefix=str(args.prefix),
            file_format=str(args.format),
            smooth_window=int(args.smooth_window),
            no_show=bool(args.no_show),
        )
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib" or "matplotlib" in str(exc).lower():
            print("clean plotting requires matplotlib; install/use an environment with matplotlib to generate figures.", file=sys.stderr)
            return 2
        raise
    print(json.dumps({"generated": [str(path) for path in generated]}, ensure_ascii=True, sort_keys=True))
    return 0


def plot_clean_metrics(
    *,
    metrics_jsonl: Path | None,
    summary_json: Path | None,
    output_dir: Path | None,
    prefix: str = "",
    file_format: str = "png",
    smooth_window: int = 1,
    no_show: bool = False,
) -> list[Path]:
    plt = _require_matplotlib()
    output_dir = Path(output_dir or "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(metrics_jsonl) if metrics_jsonl is not None and Path(metrics_jsonl).exists() else []
    summary = read_json(summary_json) if summary_json is not None and Path(summary_json).exists() else {}
    if not rows and not summary:
        raise FileNotFoundError("No clean metrics JSONL or summary JSON found for plotting.")
    # Synchronous multisample logs write one row per environment per update;
    # only the first row of each update carries the ppo_* statistics, while the
    # other rows leave them None (rendered as 0.0 by _number). Deduplicate to
    # one row per update so reward/entropy curves do not oscillate between real
    # values and zero placeholders. Eval logs (no ppo_update_step at all) and
    # single-env training logs are unaffected.
    if rows and any(row.get("ppo_update_step") is not None for row in rows) and any(
        row.get("ppo_update_step") is None for row in rows
    ):
        rows = [row for row in rows if row.get("ppo_update_step") is not None]
    is_eval = _looks_like_eval(rows, summary, metrics_jsonl)
    generated = (
        _plot_eval(rows=rows, summary=summary, output_dir=output_dir, prefix=prefix, file_format=file_format, plt=plt)
        if is_eval
        else _plot_train(rows=rows, output_dir=output_dir, prefix=prefix, file_format=file_format, smooth_window=smooth_window, plt=plt)
    )
    if not no_show:
        plt.show()
    return generated


def read_jsonl(path: Path | str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def read_json(path: Path | str | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_plot_files(kind: str, *, prefix: str = "", file_format: str = "png") -> list[str]:
    names = EVAL_PLOTS if kind == "eval" else TRAIN_PLOTS
    prefix_value = str(prefix)
    return [f"{prefix_value}{name[:-3]}{file_format}" for name in names]


def _plot_train(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    prefix: str,
    file_format: str,
    smooth_window: int,
    plt: Any,
) -> list[Path]:
    if not rows:
        raise ValueError("Training plotting requires train_metrics.jsonl rows.")
    x = _series(rows, "global_slot", default_start=1)
    generated: list[Path] = []
    generated.append(
        _line_plot(
            plt,
            output_dir / f"{prefix}train_reward.{file_format}",
            x,
            {
                # Prefer the per-rollout mean reward (proper per-update
                # statistic); fall back to the legacy per-slot "reward" field
                # for logs that predate ppo_rollout_reward_mean.
                "reward": _smooth(
                    _first_available(rows, ["ppo_rollout_reward_mean", "reward"]),
                    smooth_window,
                ),
                "time": _smooth(_series(rows, "reward_time_penalty"), smooth_window),
                "task_energy": _smooth(_series(rows, "reward_task_energy_penalty"), smooth_window),
                "move_energy": _smooth(_series(rows, "reward_movement_energy_penalty"), smooth_window),
                "dag_bonus": _smooth(_series(rows, "reward_completed_dag_bonus"), smooth_window),
            },
            title="Clean Training Reward",
            xlabel="global slot",
        )
    )
    generated.append(
        _line_plot(
            plt,
            output_dir / f"{prefix}train_completion_throughput.{file_format}",
            x,
            {
                "completion_rate": _series(rows, "DAG_completion_rate"),
                "throughput": _series(rows, "DAG_throughput"),
            },
            title="Clean DAG Completion / Throughput",
            xlabel="global slot",
        )
    )
    generated.append(
        _line_plot(
            plt,
            output_dir / f"{prefix}train_energy_flowtime.{file_format}",
            x,
            {
                "avg_flowtime": _first_available(rows, ["Average_DAG_flowtime", "average_dag_flowtime", "avg_dag_flowtime"]),
                "critical_delay": _first_available(rows, ["Average_critical_path_task_completion_delay", "average_critical_path_task_completion_delay"]),
                "energy_per_dag": _first_available(rows, ["Energy_per_completed_DAG", "energy_per_completed_dag"]),
            },
            title="Clean Flowtime / Energy",
            xlabel="global slot",
        )
    )
    generated.append(
        _line_plot(
            plt,
            output_dir / f"{prefix}train_losses.{file_format}",
            x,
            {
                "policy_loss": _sum_series(rows, ["ppo_movement_loss", "ppo_offloading_loss"]),
                "value_loss": _series(rows, "ppo_value_loss"),
                "offloading_action_value_loss": _series(rows, "ppo_offloading_action_value_loss"),
                "movement_entropy": _series(rows, "ppo_movement_entropy"),
                "offloading_entropy": _series(rows, "ppo_offloading_entropy"),
            },
            title="Clean PPO Loss / Entropy",
            xlabel="global slot",
        )
    )
    generated.append(
        _line_plot(
            plt,
            output_dir / f"{prefix}train_actions_correctness.{file_format}",
            x,
            {
                "invalid_assignment_rate": _series(rows, "invalid_assignment_rate"),
                "action_executed_rate": _series(rows, "action_executed_rate"),
                "hover_rate": _movement_rate(rows, "hover"),
                "offloading_action_count": _series(rows, "offloading_action_count"),
            },
            title="Clean Actions / Correctness",
            xlabel="global slot",
        )
    )
    return generated


def _plot_eval(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    prefix: str,
    file_format: str,
    plt: Any,
) -> list[Path]:
    generated: list[Path] = []
    if rows:
        x = _series(rows, "episode", default_start=0)
        generated.append(
            _line_plot(
                plt,
                output_dir / f"{prefix}eval_per_episode_metrics.{file_format}",
                x,
                {
                    "completion_rate": _series(rows, "DAG_completion_rate"),
                    "flowtime": _series(rows, "Average_DAG_flowtime"),
                    "throughput": _series(rows, "DAG_throughput"),
                    "energy_per_dag": _series(rows, "Energy_per_completed_DAG"),
                    "arrival_slots": _series(rows, "arrival_slots_executed"),
                    "drain_slots": _series(rows, "drain_slots_executed"),
                    "invalid_rate": _series(rows, "invalid_assignment_rate"),
                    "action_executed_rate": _series(rows, "action_executed_rate"),
                },
                title="Clean Evaluation Per-Episode Metrics",
                xlabel="episode",
            )
        )
    summary_source = summary or _summary_from_rows(rows)
    generated.append(
        _bar_plot(
            plt,
            output_dir / f"{prefix}eval_summary.{file_format}",
            {
                "completion": summary_source.get("DAG_completion_rate", 0.0),
                "throughput": summary_source.get("DAG_throughput", 0.0),
                "flowtime": summary_source.get("Average_DAG_flowtime", 0.0),
                "critical_delay": summary_source.get("Average_critical_path_task_completion_delay", 0.0),
                "energy_per_dag": summary_source.get("Energy_per_completed_DAG", 0.0),
                "invalid_rate": summary_source.get("invalid_assignment_rate", 0.0),
                "action_rate": summary_source.get("action_executed_rate", 0.0),
                "hover_rate": (summary_source.get("movement_action_distribution") or {}).get("hover", 0.0),
            },
            title="Clean Evaluation Summary",
        )
    )
    return generated


def _line_plot(plt: Any, path: Path, x: np.ndarray, series: dict[str, np.ndarray], *, title: str, xlabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, y in series.items():
        if y.size == 0:
            continue
        ax.plot(x[: y.size], y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _bar_plot(plt: Any, path: Path, values: dict[str, Any], *, title: str) -> Path:
    labels = list(values)
    heights = [float(value) if value is not None else 0.0 for value in values.values()]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, heights)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _series(rows: list[dict[str, Any]], key: str, default_start: int | None = None) -> np.ndarray:
    if key not in rows[0] and default_start is not None:
        return np.arange(default_start, default_start + len(rows), dtype=np.float32)
    return np.asarray([_number(row.get(key)) for row in rows], dtype=np.float32)


def _first_available(rows: list[dict[str, Any]], keys: list[str]) -> np.ndarray:
    for key in keys:
        if key in rows[0]:
            return _series(rows, key)
    return np.zeros((len(rows),), dtype=np.float32)


def _sum_series(rows: list[dict[str, Any]], keys: list[str]) -> np.ndarray:
    values = np.zeros((len(rows),), dtype=np.float32)
    for key in keys:
        if key in rows[0]:
            values += _series(rows, key)
    return values


def _movement_rate(rows: list[dict[str, Any]], action_name: str) -> np.ndarray:
    values = []
    for row in rows:
        distribution = row.get("movement_action_distribution") or {}
        values.append(_number(distribution.get(action_name)))
    return np.asarray(values, dtype=np.float32)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if int(window) <= 1 or values.size < int(window):
        return values
    kernel = np.ones((int(window),), dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [
        "DAG_completion_rate",
        "DAG_throughput",
        "Average_DAG_flowtime",
        "Average_critical_path_task_completion_delay",
        "Energy_per_completed_DAG",
        "invalid_assignment_rate",
        "action_executed_rate",
    ]
    summary = {key: float(np.mean([_number(row.get(key)) for row in rows])) for key in keys}
    actions = {}
    for row in rows:
        for action, value in (row.get("movement_action_distribution") or {}).items():
            actions[action] = actions.get(action, 0.0) + _number(value) / max(float(len(rows)), 1.0)
    summary["movement_action_distribution"] = actions
    return summary


def _looks_like_eval(rows: list[dict[str, Any]], summary: dict[str, Any], metrics_jsonl: Path | None) -> bool:
    if metrics_jsonl is not None and "eval_metrics" in Path(metrics_jsonl).name:
        return True
    if rows and "arrival_slots_executed" in rows[0]:
        return True
    return "arrival_slots_executed" in summary or "drain_slots_executed" in summary


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise ModuleNotFoundError("matplotlib is required for clean plotting") from exc
        raise
    return plt


if __name__ == "__main__":
    raise SystemExit(main())
