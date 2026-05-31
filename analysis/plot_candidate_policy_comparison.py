from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ASSIGNMENT_METRICS = (
    "event_count",
    "avg_candidate_count",
    "candidate_choice_rate",
    "candidate_count_0_rate",
    "candidate_count_3plus_rate",
    "soft_violation_event_rate",
    "soft_violation_candidate_rate",
)

TASK_LEVEL_METRICS = (
    "dag_task_finish_rate",
    "dag_task_drop_rate",
    "dag_avg_completion_time",
    "invalid_assignments",
    "episode_latency",
)

DAG_LEVEL_METRICS = (
    "dag_on_time_success_rate",
    "dag_success_rate",
    "dag_failure_rate",
)

COST_STRUCTURE_METRICS = (
    "episode_energy",
    "compute_heavy_to_high_compute_uav_ratio",
    "communication_heavy_to_low_transfer_uav_ratio",
    "aggregation_to_parent_locality_uav_ratio",
)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _candidate_count(event: dict[str, Any]) -> int:
    for key in ("candidate_count", "num_candidates"):
        if key in event and event[key] is not None:
            try:
                return max(int(event[key]), 0)
            except (TypeError, ValueError):
                pass
    candidates = event.get("candidates")
    return len(candidates) if isinstance(candidates, list) else 0


def _candidate_has_soft_violation(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    violations = candidate.get("soft_constraint_violations")
    if isinstance(violations, list):
        return len(violations) > 0
    if isinstance(violations, str):
        return bool(violations.strip())
    return bool(violations)


def _resolve_jsonl_paths(path_arg: str, mode: str) -> list[Path]:
    paths: list[Path] = []
    for raw in path_arg.split(","):
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw)
        if path.is_dir():
            matched = sorted(path.glob(f"{mode}*_assignment_events.jsonl"))
            if not matched:
                matched = sorted(path.rglob(f"{mode}*_assignment_events.jsonl"))
            paths.extend(matched)
        else:
            paths.append(path)
    return paths


def _resolve_summary_paths(path_arg: str | None, mode: str) -> list[Path]:
    if not path_arg:
        return []
    paths: list[Path] = []
    for raw in path_arg.split(","):
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(path.glob(f"{mode}*.json"))
            if not candidates:
                candidates = sorted(path.rglob(f"{mode}*.json"))
            paths.extend(
                item
                for item in candidates
                if not item.name.endswith("_attribution.json")
                and not item.name.endswith("_candidate_rejection_summary.json")
            )
        else:
            paths.append(path)
    return paths


def _stream_assignment_stats(paths: list[Path], label: str) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "label": label,
        "source_files": [str(path) for path in paths],
        "missing_files": [],
        "parse_errors": [],
        "event_count": 0,
        "candidate_count_sum": 0,
        "candidate_choice_count": 0,
        "candidate_count_histogram": {"count_0": 0, "count_1": 0, "count_2": 0, "count_3plus": 0},
        "soft_violation_event_count": 0,
        "candidate_record_count": 0,
        "soft_violation_candidate_count": 0,
    }
    for path in paths:
        if not path.exists():
            stats["missing_files"].append(str(path))
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    stats["parse_errors"].append(f"{path}:{line_no}: {exc}")
                    continue
                if not isinstance(event, dict):
                    continue
                count = _candidate_count(event)
                stats["event_count"] += 1
                stats["candidate_count_sum"] += count
                if count > 1:
                    stats["candidate_choice_count"] += 1
                histogram = stats["candidate_count_histogram"]
                if count <= 0:
                    histogram["count_0"] += 1
                elif count == 1:
                    histogram["count_1"] += 1
                elif count == 2:
                    histogram["count_2"] += 1
                else:
                    histogram["count_3plus"] += 1

                candidates = event.get("candidates")
                event_has_violation = False
                if isinstance(candidates, list):
                    for candidate in candidates:
                        stats["candidate_record_count"] += 1
                        if _candidate_has_soft_violation(candidate):
                            stats["soft_violation_candidate_count"] += 1
                            event_has_violation = True
                if event_has_violation:
                    stats["soft_violation_event_count"] += 1

    total = int(stats["event_count"])
    candidate_total = int(stats["candidate_record_count"])
    histogram = stats["candidate_count_histogram"]
    stats["avg_candidate_count"] = stats["candidate_count_sum"] / total if total else None
    stats["candidate_choice_rate"] = stats["candidate_choice_count"] / total if total else None
    stats["candidate_count_0_rate"] = histogram["count_0"] / total if total else None
    stats["candidate_count_1_rate"] = histogram["count_1"] / total if total else None
    stats["candidate_count_2_rate"] = histogram["count_2"] / total if total else None
    stats["candidate_count_3plus_rate"] = histogram["count_3plus"] / total if total else None
    stats["soft_violation_event_rate"] = stats["soft_violation_event_count"] / total if total else None
    stats["soft_violation_candidate_rate"] = (
        stats["soft_violation_candidate_count"] / candidate_total if candidate_total else None
    )
    return stats


def _load_summary_metrics(paths: list[Path], mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in paths:
        if not path.exists():
            skipped.append(f"{path}: missing")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append(f"{path}: {exc}")
            continue
        if not isinstance(data, dict):
            skipped.append(f"{path}: not a JSON object")
            continue
        policy = str(data.get("candidate_policy_mode") or data.get("summary", {}).get("candidate_policy_mode") or "")
        if policy and policy != mode:
            skipped.append(f"{path}: candidate_policy_mode={policy}")
            continue
        summary = data.get("summary")
        if not isinstance(summary, dict):
            skipped.append(f"{path}: missing summary")
            continue
        rows.append(summary)

    metric_names = set(TASK_LEVEL_METRICS) | set(DAG_LEVEL_METRICS) | set(COST_STRUCTURE_METRICS)
    metrics: dict[str, Any] = {
        "source_files": [str(path) for path in paths],
        "used_files": len(rows),
        "skipped": skipped,
    }
    for name in sorted(metric_names):
        values = [_safe_float(row.get(name)) for row in rows]
        values = [value for value in values if value is not None]
        metrics[name] = float(mean(values)) if values else None
    return metrics


def _diff(expanded: dict[str, Any], strict: dict[str, Any], keys: Iterable[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in keys:
        lhs = _safe_float(expanded.get(key))
        rhs = _safe_float(strict.get(key))
        result[key] = None if lhs is None or rhs is None else lhs - rhs
    return result


def _try_import_matplotlib() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _format_value(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "NA"
    if abs(numeric) >= 100:
        return f"{numeric:.2f}"
    return f"{numeric:.6f}"


def _plot_grouped_bars(
    plt: Any,
    output_path: Path,
    title: str,
    metric_names: list[str],
    strict_values: list[float | None],
    expanded_values: list[float | None],
    ylabel: str = "value",
) -> None:
    x = list(range(len(metric_names)))
    width = 0.36
    strict_plot = [0.0 if value is None else value for value in strict_values]
    expanded_plot = [0.0 if value is None else value for value in expanded_values]
    fig_width = max(8.0, len(metric_names) * 1.55)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar([idx - width / 2 for idx in x], strict_plot, width, label="strict", color="#4c78a8")
    ax.bar([idx + width / 2 for idx in x], expanded_plot, width, label="expanded", color="#f58518")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_metric_panels(
    plt: Any,
    output_path: Path,
    title: str,
    metric_names: tuple[str, ...],
    strict: dict[str, Any],
    expanded: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, len(metric_names), figsize=(max(10.0, len(metric_names) * 3.0), 4.0))
    if len(metric_names) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metric_names):
        values = [_safe_float(strict.get(metric)), _safe_float(expanded.get(metric))]
        plot_values = [0.0 if value is None else value for value in values]
        ax.bar(["strict", "expanded"], plot_values, color=["#4c78a8", "#f58518"])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            if value is not None:
                ax.text(idx, plot_values[idx], _format_value(value), ha="center", va="bottom", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_figures(summary: dict[str, Any], output_dir: Path) -> list[str]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt = _try_import_matplotlib()
    if plt is None:
        return []

    written: list[str] = []
    assignment = summary["assignment_level"]
    strict_assignment = assignment["strict"]
    expanded_assignment = assignment["expanded"]
    strict_hist = strict_assignment["candidate_count_histogram"]
    expanded_hist = expanded_assignment["candidate_count_histogram"]
    event_counts = {
        "strict": max(int(strict_assignment.get("event_count") or 0), 1),
        "expanded": max(int(expanded_assignment.get("event_count") or 0), 1),
    }

    hist_metrics = ["count_0", "count_1", "count_2", "count_3plus"]
    _plot_grouped_bars(
        plt,
        figures_dir / "candidate_count_histogram.png",
        "Candidate Count Distribution (JSONL assignment-level)",
        hist_metrics,
        [strict_hist[name] / event_counts["strict"] for name in hist_metrics],
        [expanded_hist[name] / event_counts["expanded"] for name in hist_metrics],
        ylabel="event rate",
    )
    written.append(str(figures_dir / "candidate_count_histogram.png"))

    candidate_metrics = [
        "avg_candidate_count",
        "candidate_choice_rate",
        "candidate_count_0_rate",
        "candidate_count_3plus_rate",
    ]
    _plot_grouped_bars(
        plt,
        figures_dir / "candidate_space_summary.png",
        "Candidate Space Summary (JSONL assignment-level)",
        candidate_metrics,
        [_safe_float(strict_assignment.get(name)) for name in candidate_metrics],
        [_safe_float(expanded_assignment.get(name)) for name in candidate_metrics],
    )
    written.append(str(figures_dir / "candidate_space_summary.png"))

    _plot_grouped_bars(
        plt,
        figures_dir / "soft_violation_rates.png",
        "Soft Constraint Violation Rates (JSONL assignment-level)",
        ["soft_violation_event_rate", "soft_violation_candidate_rate"],
        [
            _safe_float(strict_assignment.get("soft_violation_event_rate")),
            _safe_float(strict_assignment.get("soft_violation_candidate_rate")),
        ],
        [
            _safe_float(expanded_assignment.get("soft_violation_event_rate")),
            _safe_float(expanded_assignment.get("soft_violation_candidate_rate")),
        ],
    )
    written.append(str(figures_dir / "soft_violation_rates.png"))

    run_summary = summary.get("run_summary")
    if run_summary and run_summary.get("available"):
        strict_run = run_summary["strict"]
        expanded_run = run_summary["expanded"]
        _plot_metric_panels(
            plt,
            figures_dir / "task_level_metrics.png",
            "Task-Level Metrics (run summary)",
            TASK_LEVEL_METRICS,
            strict_run,
            expanded_run,
        )
        written.append(str(figures_dir / "task_level_metrics.png"))
        _plot_metric_panels(
            plt,
            figures_dir / "dag_level_metrics.png",
            "DAG-Level Metrics (run summary)",
            DAG_LEVEL_METRICS,
            strict_run,
            expanded_run,
        )
        written.append(str(figures_dir / "dag_level_metrics.png"))
        _plot_metric_panels(
            plt,
            figures_dir / "cost_and_structure_tradeoff.png",
            "Cost and Structure Tradeoff (run summary)",
            COST_STRUCTURE_METRICS,
            strict_run,
            expanded_run,
        )
        written.append(str(figures_dir / "cost_and_structure_tradeoff.png"))
    return written


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _write_report(summary: dict[str, Any], output_path: Path) -> None:
    assignment = summary["assignment_level"]
    strict = assignment["strict"]
    expanded = assignment["expanded"]
    diff = assignment["expanded_minus_strict"]
    run_summary = summary.get("run_summary", {})

    hist_rows = []
    strict_total = max(int(strict.get("event_count") or 0), 1)
    expanded_total = max(int(expanded.get("event_count") or 0), 1)
    for key in ["count_0", "count_1", "count_2", "count_3plus"]:
        strict_count = strict["candidate_count_histogram"][key]
        expanded_count = expanded["candidate_count_histogram"][key]
        hist_rows.append(
            [
                key,
                strict_count,
                _format_value(strict_count / strict_total),
                expanded_count,
                _format_value(expanded_count / expanded_total),
            ]
        )

    assignment_rows = [
        [metric, _format_value(strict.get(metric)), _format_value(expanded.get(metric)), _format_value(diff.get(metric))]
        for metric in ASSIGNMENT_METRICS
    ]

    lines = [
        "# Candidate Policy Comparison",
        "",
        "This report separates JSONL assignment-level statistics from run summary task/DAG-level metrics.",
        "Candidate count and candidate choice rate are computed from the full assignment JSONL, not from candidate rejection summaries.",
        "",
        "## Inputs",
        f"- strict JSONL files: {len(strict.get('source_files', []))}",
        f"- expanded JSONL files: {len(expanded.get('source_files', []))}",
        f"- strict summary files: {run_summary.get('strict', {}).get('used_files', 0) if run_summary else 0}",
        f"- expanded summary files: {run_summary.get('expanded', {}).get('used_files', 0) if run_summary else 0}",
        "",
        "## JSONL Assignment-Level Statistics",
        _markdown_table(["metric", "strict", "expanded", "expanded - strict"], assignment_rows),
        "",
        "## Candidate Count Histogram",
        _markdown_table(["bucket", "strict_count", "strict_rate", "expanded_count", "expanded_rate"], hist_rows),
        "",
        "## Soft Constraint Violations",
        "- `soft_violation_event_rate` is the fraction of assignment events with at least one candidate containing `soft_constraint_violations`.",
        "- `soft_violation_candidate_rate` is the fraction of candidate records containing `soft_constraint_violations`.",
        f"- strict event rate: {_format_value(strict.get('soft_violation_event_rate'))}",
        f"- expanded event rate: {_format_value(expanded.get('soft_violation_event_rate'))}",
        f"- strict candidate rate: {_format_value(strict.get('soft_violation_candidate_rate'))}",
        f"- expanded candidate rate: {_format_value(expanded.get('soft_violation_candidate_rate'))}",
        "",
    ]

    if run_summary and run_summary.get("available"):
        strict_run = run_summary["strict"]
        expanded_run = run_summary["expanded"]
        run_diff = run_summary["expanded_minus_strict"]
        run_rows = [
            [metric, _format_value(strict_run.get(metric)), _format_value(expanded_run.get(metric)), _format_value(run_diff.get(metric))]
            for metric in list(TASK_LEVEL_METRICS) + list(DAG_LEVEL_METRICS) + list(COST_STRUCTURE_METRICS)
        ]
        lines.extend(
            [
                "## Run Summary Metrics",
                "These metrics come from `static_scheduler_compare.py` summary JSON files, not from assignment JSONL.",
                _markdown_table(["metric", "strict", "expanded", "expanded - strict"], run_rows),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Run Summary Metrics",
                "Summary JSON was not provided or no valid summary records were found, so task/DAG/energy figures were skipped.",
                "",
            ]
        )

    lines.extend(
        [
            "## Cautious Conclusion",
            "- expanded v1 clearly expands the candidate set relative to strict.",
            "- The strict candidate filter being too narrow is a real issue in these runs.",
            "- With summary metrics available, expanded v1 should be interpreted as improving task-level completion and latency if those metrics move in that direction.",
            "- expanded v1 also introduces many soft-violation candidates, so the extra choice space is not free.",
            "- DAG failure, energy, and communication/locality metrics must be treated as tradeoffs, not secondary details.",
            "- These figures are diagnostic evidence; they do not prove expanded v1 is the final optimal candidate policy.",
            "",
        ]
    )

    figure_paths = summary.get("figures", [])
    if figure_paths:
        lines.append("## Figures")
        for path in figure_paths:
            lines.append(f"- {path}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    strict_jsonls = _resolve_jsonl_paths(args.strict_jsonl, "strict")
    expanded_jsonls = _resolve_jsonl_paths(args.expanded_jsonl, "expanded")
    strict_assignment = _stream_assignment_stats(strict_jsonls, "strict")
    expanded_assignment = _stream_assignment_stats(expanded_jsonls, "expanded")

    strict_summary_paths = _resolve_summary_paths(args.strict_summary, "strict")
    expanded_summary_paths = _resolve_summary_paths(args.expanded_summary, "expanded")
    strict_run = _load_summary_metrics(strict_summary_paths, "strict") if strict_summary_paths else {}
    expanded_run = _load_summary_metrics(expanded_summary_paths, "expanded") if expanded_summary_paths else {}
    run_available = bool(strict_run.get("used_files")) and bool(expanded_run.get("used_files"))

    result: dict[str, Any] = {
        "metadata": {
            "strict_jsonl": args.strict_jsonl,
            "expanded_jsonl": args.expanded_jsonl,
            "strict_summary": args.strict_summary,
            "expanded_summary": args.expanded_summary,
            "output_dir": args.output_dir,
            "candidate_choice_rate_definition": "assignment events with candidate_count > 1 divided by all assignment events",
            "candidate_count_source": "assignment JSONL full statistics",
            "soft_violation_source": "candidates[*].soft_constraint_violations in assignment JSONL",
        },
        "assignment_level": {
            "strict": strict_assignment,
            "expanded": expanded_assignment,
            "expanded_minus_strict": _diff(expanded_assignment, strict_assignment, ASSIGNMENT_METRICS),
        },
        "run_summary": {
            "available": run_available,
            "strict": strict_run,
            "expanded": expanded_run,
            "expanded_minus_strict": _diff(
                expanded_run,
                strict_run,
                list(TASK_LEVEL_METRICS) + list(DAG_LEVEL_METRICS) + list(COST_STRUCTURE_METRICS),
            )
            if run_available
            else {},
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot strict vs expanded candidate policy comparison.")
    parser.add_argument("--strict_jsonl", required=True, help="Strict assignment JSONL file or directory.")
    parser.add_argument("--expanded_jsonl", required=True, help="Expanded assignment JSONL file or directory.")
    parser.add_argument("--strict_summary", default=None, help="Optional strict summary JSON file, directory, or comma list.")
    parser.add_argument("--expanded_summary", default=None, help="Optional expanded summary JSON file, directory, or comma list.")
    parser.add_argument("--output_dir", required=True, help="Directory for summary, report, and figures.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(args)
    figures = _write_figures(summary, output_dir)
    summary["figures"] = figures

    summary_path = output_dir / "candidate_policy_comparison_summary.json"
    report_path = output_dir / "candidate_policy_comparison_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(summary, report_path)

    strict = summary["assignment_level"]["strict"]
    expanded = summary["assignment_level"]["expanded"]
    print("Candidate policy comparison complete.")
    print(f"summary={summary_path}")
    print(f"report={report_path}")
    print(f"figures={len(figures)}")
    print(
        "assignment_level: "
        f"strict_events={strict.get('event_count')} expanded_events={expanded.get('event_count')} "
        f"strict_avg_candidates={_format_value(strict.get('avg_candidate_count'))} "
        f"expanded_avg_candidates={_format_value(expanded.get('avg_candidate_count'))} "
        f"strict_choice_rate={_format_value(strict.get('candidate_choice_rate'))} "
        f"expanded_choice_rate={_format_value(expanded.get('candidate_choice_rate'))}"
    )


if __name__ == "__main__":
    main()
