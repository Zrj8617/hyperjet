from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


Number = int | float


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _rate(count: float, total: float) -> float | None:
    return count / total if total > 0 else None


def _load_jsonl(path: Path, ablation: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            event.setdefault("ablation", ablation)
            event["_source_file"] = str(path)
            events.append(event)
    return events


def _soft_violations(event: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for candidate in event.get("candidates", []) or []:
        for violation in candidate.get("soft_constraint_violations", []) or []:
            violations.append(str(violation))
    return violations


def _load_rejection_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]]
    if isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
    elif isinstance(data, dict):
        obj = data.get(key, data.get("outcomes", []))
        rows = [row for row in obj if isinstance(row, dict)] if isinstance(obj, list) else []
    else:
        rows = []
    result: dict[tuple[int, str], dict[str, Any]] = {}
    id_field = "task_id" if key == "task_outcomes" else "dag_id"
    for row in rows:
        episode = row.get("episode")
        item_id = row.get(id_field)
        if episode is None or item_id is None:
            continue
        result[(int(episode), str(item_id))] = row
    return result


def _join_outcomes(
    events: list[dict[str, Any]],
    task_outcomes: dict[tuple[int, str], dict[str, Any]],
    dag_outcomes: dict[tuple[int, str], dict[str, Any]],
) -> int:
    joined = 0
    for event in events:
        episode = event.get("episode")
        if episode is None:
            continue
        task = task_outcomes.get((int(episode), str(event.get("task_id"))))
        dag = dag_outcomes.get((int(episode), str(event.get("dag_id"))))
        if task is not None:
            event["observed_task_final_state"] = task.get("final_state")
            event["observed_task_finished_on_time"] = task.get("finished_on_time")
            event["observed_task_dropped"] = task.get("dropped")
            event["observed_task_completion_time"] = task.get("completion_time")
            joined += 1
        if dag is not None:
            event["observed_dag_successful"] = dag.get("successful")
            event["observed_dag_on_time_successful"] = dag.get("on_time_successful")
            event["observed_dag_failed"] = dag.get("failed")
    return joined


def _candidate_for(event: dict[str, Any], uav_id: Any) -> dict[str, Any] | None:
    if uav_id is None:
        return None
    for candidate in event.get("candidates", []) or []:
        if candidate.get("uav_id") == uav_id:
            return candidate
    return None


def _selected_candidate(event: dict[str, Any]) -> dict[str, Any] | None:
    return _candidate_for(event, event.get("selected_uav"))


def _selection_mode(event: dict[str, Any]) -> str:
    return str(event.get("selection_mode") or "none")


def _is_score_selected(event: dict[str, Any]) -> bool:
    value = _safe_bool(event.get("score_selected"))
    return value if value is not None else _selection_mode(event) == "score"


def _is_fallback_selected(event: dict[str, Any]) -> bool:
    value = _safe_bool(event.get("fallback_used"))
    return value if value is not None else _selection_mode(event) in {"fallback", "guard_fallback"}


def _is_guard_clamped(event: dict[str, Any]) -> bool:
    value = _safe_bool(event.get("guard_clamped"))
    return value if value is not None else event.get("guard_reason") == "runtime_bounded_guard_clamp"


def _is_guard_rejected(event: dict[str, Any]) -> bool:
    value = _safe_bool(event.get("guard_rejected"))
    return value if value is not None else _selection_mode(event) == "guard_fallback"


def _event_metric(event: dict[str, Any], name: str) -> float | None:
    if name == "selected_uav_compute_capacity":
        candidate = _selected_candidate(event)
        return _safe_float(None if candidate is None else candidate.get("compute_capacity"))
    if name == "selected_candidate_transmission_time":
        candidate = _selected_candidate(event)
        return _safe_float(None if candidate is None else candidate.get("transmission_time"))
    if name == "selected_candidate_queue_length":
        candidate = _selected_candidate(event)
        return _safe_float(None if candidate is None else candidate.get("queue_length"))
    if name == "selected_planned_finish":
        return _safe_float(event.get("selected_planned_finish"))
    if name == "selected_deadline_margin":
        value = _safe_float(event.get("selected_deadline_margin"))
        if value is not None:
            return value
        deadline = _safe_float(event.get("task_deadline"))
        finish = _safe_float(event.get("selected_planned_finish"))
        return None if deadline is None or finish is None else deadline - finish
    if name == "selected_score":
        return _safe_float(event.get("selected_score"))
    return _safe_float(event.get(name))


def _bucket_value(event: dict[str, Any], bucket: str) -> str | None:
    if bucket == "task_type":
        return str(event.get("task_type", "missing"))
    if bucket == "candidate_count_bucket":
        count = int(event.get("candidate_count") or event.get("num_candidates") or 0)
        if count <= 1:
            return "1"
        if count == 2:
            return "2"
        return "3+"
    if bucket == "critical_path":
        value = _safe_bool(event.get("is_critical_path_task"))
        return "missing" if value is None else str(value).lower()
    if bucket == "slack_bucket":
        slack = _safe_float(event.get("task_slack"))
        if slack is None:
            return None
        if slack <= 0:
            return "negative_or_zero"
        if slack <= 4:
            return "low"
        if slack <= 12:
            return "medium"
        return "high"
    if bucket == "compute_demand_bucket":
        value = _safe_float(event.get("task_cpu_cycles"))
        return _tertile_bucket(value, "compute")
    if bucket == "data_size_bucket":
        value = _safe_float(event.get("task_data_size"))
        if value is None:
            input_size = _safe_float(event.get("task_input_size"))
            output_size = _safe_float(event.get("task_output_size"))
            value = None if input_size is None or output_size is None else input_size + output_size
        return _tertile_bucket(value, "data")
    return None


def _tertile_bucket(value: float | None, prefix: str) -> str | None:
    if value is None:
        return None
    # Coarse fixed thresholds are enough for behavioral diagnostics and avoid
    # making bucket definitions depend on both compared runs.
    if prefix == "compute":
        if value < 2.0e9:
            return "low"
        if value < 5.0e9:
            return "medium"
        return "high"
    if value < 8.0e5:
        return "low"
    if value < 2.0e6:
        return "medium"
    return "high"


def _summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    total = float(len(events))
    numeric_fields = {
        "candidate_count_mean": [
            _safe_float(event.get("candidate_count") or event.get("num_candidates"))
            for event in events
        ],
        "selected_uav_compute_capacity_mean": [_event_metric(event, "selected_uav_compute_capacity") for event in events],
        "selected_candidate_transmission_time_mean": [
            _event_metric(event, "selected_candidate_transmission_time") for event in events
        ],
        "selected_candidate_queue_length_mean": [_event_metric(event, "selected_candidate_queue_length") for event in events],
        "selected_planned_finish_mean": [_event_metric(event, "selected_planned_finish") for event in events],
        "selected_deadline_margin_mean": [_event_metric(event, "selected_deadline_margin") for event in events],
        "selected_score_mean": [_event_metric(event, "selected_score") for event in events],
        "delta_planned_finish_vs_heuristic_mean": [_safe_float(event.get("delta_planned_finish")) for event in events],
        "delta_transmission_time_vs_heuristic_mean": [_safe_float(event.get("delta_transmission_time")) for event in events],
        "delta_execution_time_vs_heuristic_mean": [_safe_float(event.get("delta_execution_time")) for event in events],
        "delta_queue_length_vs_heuristic_mean": [_safe_float(event.get("delta_queue_length")) for event in events],
    }
    deltas = [value for value in numeric_fields["delta_planned_finish_vs_heuristic_mean"] if value is not None]
    summary: dict[str, Any] = {
        "event_count": int(total),
        "score_selected_rate": _rate(sum(1 for event in events if _is_score_selected(event)), total),
        "fallback_selected_rate": _rate(sum(1 for event in events if _is_fallback_selected(event)), total),
        "guard_clamp_rate": _rate(sum(1 for event in events if _is_guard_clamped(event)), total),
        "guard_reject_rate": _rate(sum(1 for event in events if _is_guard_rejected(event)), total),
        "candidate_choice_rate": _rate(
            sum(1 for event in events if int(event.get("candidate_count") or event.get("num_candidates") or 0) > 1),
            total,
        ),
        "candidate_count_histogram": {
            "count_0": sum(1 for event in events if int(event.get("candidate_count") or event.get("num_candidates") or 0) <= 0),
            "count_1": sum(1 for event in events if int(event.get("candidate_count") or event.get("num_candidates") or 0) == 1),
            "count_2": sum(1 for event in events if int(event.get("candidate_count") or event.get("num_candidates") or 0) == 2),
            "count_3_plus": sum(1 for event in events if int(event.get("candidate_count") or event.get("num_candidates") or 0) >= 3),
        },
        "soft_constraint_violation_counts": {},
        "raw_score_vs_heuristic_disagreement_rate": _rate(
            sum(1 for event in events if _safe_bool(event.get("raw_disagrees_with_heuristic")) is True),
            total,
        ),
        "selected_vs_heuristic_disagreement_rate": _rate(
            sum(1 for event in events if _safe_bool(event.get("disagrees_with_heuristic")) is True),
            total,
        ),
        "delta_planned_finish_vs_heuristic_p50": _median(deltas),
        "delta_planned_finish_vs_heuristic_p90": _percentile(deltas, 0.9),
    }
    for violation in sorted(_soft_violations(event) for event in []):
        pass
    violation_counts: dict[str, int] = {}
    kept_soft_violation_candidates = 0
    total_candidates = 0
    for event in events:
        for candidate in event.get("candidates", []) or []:
            total_candidates += 1
            candidate_violations = candidate.get("soft_constraint_violations", []) or []
            if candidate_violations:
                kept_soft_violation_candidates += 1
            for violation in candidate_violations:
                key = str(violation)
                violation_counts[key] = violation_counts.get(key, 0) + 1
    summary["soft_constraint_violation_counts"] = violation_counts
    summary["kept_soft_violation_candidate_rate"] = _rate(kept_soft_violation_candidates, total_candidates)
    for key, values in numeric_fields.items():
        clean = [value for value in values if value is not None]
        summary[key] = _mean(clean)
        summary[f"{key}_available"] = len(clean)
    task_outcomes = [_safe_bool(event.get("observed_task_finished_on_time")) for event in events]
    task_drop = [_safe_bool(event.get("observed_task_dropped")) for event in events]
    if any(value is not None for value in task_outcomes):
        valid = [value for value in task_outcomes if value is not None]
        summary["observed_task_on_time_rate"] = sum(1 for value in valid if value) / max(len(valid), 1)
    if any(value is not None for value in task_drop):
        valid = [value for value in task_drop if value is not None]
        summary["observed_task_drop_rate"] = sum(1 for value in valid if value) / max(len(valid), 1)
    return summary


def _build_bucket_rows(events_by_ablation: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    bucket_names = [
        "task_type",
        "candidate_count_bucket",
        "critical_path",
        "slack_bucket",
        "compute_demand_bucket",
        "data_size_bucket",
    ]
    rows: list[dict[str, Any]] = []
    for bucket_name in bucket_names:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for ablation, events in events_by_ablation.items():
            for event in events:
                bucket_value = _bucket_value(event, bucket_name)
                if bucket_value is None:
                    continue
                grouped[(ablation, bucket_value)].append(event)
        for (ablation, bucket_value), events in sorted(grouped.items()):
            summary = _summarize_events(events)
            rows.append(
                {
                    "bucket": bucket_name,
                    "bucket_value": bucket_value,
                    "ablation": ablation,
                    **summary,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(
    path: Path,
    summary: dict[str, Any],
    bucket_rows: list[dict[str, Any]],
    outcomes_joined: dict[str, int],
    rejection_summaries: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = summary["overall"]
    lines = [
        "# Assignment Behavior Report",
        "",
        "## Inputs",
        f"- attribute_only: `{summary['metadata']['attribute_only_jsonl']}`",
        f"- attribute_blind: `{summary['metadata']['attribute_blind_jsonl']}`",
        "",
        "## Sample Size",
    ]
    for ablation, obj in overall.items():
        lines.append(f"- {ablation}: {obj.get('event_count', 0)} assignment events")
    lines.extend(["", "## Overall Metrics"])
    for metric in [
        "score_selected_rate",
        "fallback_selected_rate",
        "guard_clamp_rate",
        "guard_reject_rate",
        "candidate_count_mean",
        "candidate_choice_rate",
        "selected_vs_heuristic_disagreement_rate",
        "selected_uav_compute_capacity_mean",
        "selected_candidate_transmission_time_mean",
        "selected_candidate_queue_length_mean",
        "selected_planned_finish_mean",
        "selected_deadline_margin_mean",
        "delta_planned_finish_vs_heuristic_mean",
        "delta_planned_finish_vs_heuristic_p90",
    ]:
        values = ", ".join(f"{ablation}={obj.get(metric)}" for ablation, obj in overall.items())
        lines.append(f"- {metric}: {values}")
    lines.extend(["", "## Bucket Highlights"])
    for bucket in ["task_type", "candidate_count_bucket", "critical_path", "slack_bucket"]:
        candidates = [row for row in bucket_rows if row["bucket"] == bucket]
        lines.append(f"- {bucket}: {len(candidates)} rows written to CSV")
    lines.extend(["", "## Soft Constraint Violations"])
    for ablation, obj in overall.items():
        lines.append(
            f"- {ablation}: kept_soft_violation_candidate_rate={obj.get('kept_soft_violation_candidate_rate')}, "
            f"soft_constraint_violation_counts={obj.get('soft_constraint_violation_counts')}"
        )
    lines.extend(["", "## Rejection Summaries"])
    if any(rejection_summaries.values()):
        for ablation, rejection in rejection_summaries.items():
            if not rejection:
                continue
            lines.append(
                f"- {ablation}: avg_candidate_count={rejection.get('avg_candidate_count')}, "
                f"candidate_choice_rate={rejection.get('candidate_choice_rate')}, "
                f"rejection_reason_counts={rejection.get('rejection_reason_counts')}"
            )
    else:
        lines.append("- no rejection summary json provided")
    lines.extend(["", "## Counterfactual Estimate Notes"])
    lines.append(
        "- `planned_finish`, `deadline_margin`, transmission, execution, queue and energy deltas are assignment-time counterfactual estimates from candidate schedules."
    )
    lines.append("- They are not observed outcomes for unselected UAVs.")
    lines.extend(["", "## Observed Outcome Join"])
    if any(count > 0 for count in outcomes_joined.values()):
        for ablation, count in outcomes_joined.items():
            lines.append(f"- {ablation}: joined {count} task outcome rows into assignment events")
    else:
        lines.append("- observed outcome unavailable in v1 run: no task/DAG outcome files were provided or no records matched.")
    lines.extend(["", "## Interpretation Boundary"])
    lines.append(
        "Independent-run analysis cannot strictly prove that attribute_only and attribute_blind choose different UAVs under the same context."
    )
    lines.append("It only shows whether their aggregate assignment behavior distributions differ.")
    lines.append("Strict assignment difference rate requires a second-stage paired/shadow comparison.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_figures(output_dir: Path, events_by_ablation: dict[str, list[dict[str, Any]]]) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # Figure 1: task_type resource preference.
    rows = []
    for ablation, events in events_by_ablation.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            grouped[str(event.get("task_type", "missing"))].append(event)
        for task_type, group_events in grouped.items():
            values = [_event_metric(event, "selected_uav_compute_capacity") for event in group_events]
            clean = [value for value in values if value is not None]
            rows.append((ablation, task_type, _mean(clean)))
    labels = sorted({task_type for _, task_type, _ in rows})
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, ablation in enumerate(sorted(events_by_ablation)):
        vals = [next((value for a, task_type, value in rows if a == ablation and task_type == label), None) for label in labels]
        ax.bar([pos + idx * width for pos in x], [0 if value is None else value for value in vals], width, label=ablation)
    ax.set_xticks([pos + width / 2 for pos in x], labels)
    ax.set_ylabel("selected UAV compute capacity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "task_type_selected_resource_preference.png")
    plt.close(fig)

    # Figure 2: assignment source stacked bar.
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = sorted(events_by_ablation)
    bottoms = [0.0 for _ in labels]
    for metric, color in [
        ("score_selected_rate", "#4c78a8"),
        ("fallback_selected_rate", "#f58518"),
        ("guard_clamp_rate", "#54a24b"),
        ("guard_reject_rate", "#e45756"),
    ]:
        vals = [_summarize_events(events_by_ablation[label]).get(metric) or 0.0 for label in labels]
        ax.bar(labels, vals, bottom=bottoms, label=metric, color=color)
        bottoms = [bottom + val for bottom, val in zip(bottoms, vals)]
    ax.set_ylabel("rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "assignment_source_stacked_bar.png")
    plt.close(fig)

    # Figure 3: delta planned finish distribution.
    fig, ax = plt.subplots(figsize=(8, 4))
    for ablation, events in sorted(events_by_ablation.items()):
        vals = [_safe_float(event.get("delta_planned_finish")) for event in events]
        clean = [value for value in vals if value is not None]
        if clean:
            ax.hist(clean, bins=40, alpha=0.5, label=ablation)
    ax.set_xlabel("delta planned finish vs heuristic")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "delta_planned_finish_distribution.png")
    plt.close(fig)
    return True


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    attr_only = _load_jsonl(Path(args.attribute_only_jsonl), "attribute_only")
    attr_blind = _load_jsonl(Path(args.attribute_blind_jsonl), "attribute_blind")
    events_by_ablation = {
        "attribute_only": attr_only,
        "attribute_blind": attr_blind,
    }

    outcomes_joined = {
        "attribute_only": _join_outcomes(
            attr_only,
            _load_outcomes(Path(args.task_outcomes_attribute_only) if args.task_outcomes_attribute_only else None, "task_outcomes"),
            _load_outcomes(Path(args.dag_outcomes_attribute_only) if args.dag_outcomes_attribute_only else None, "dag_outcomes"),
        ),
        "attribute_blind": _join_outcomes(
            attr_blind,
            _load_outcomes(Path(args.task_outcomes_attribute_blind) if args.task_outcomes_attribute_blind else None, "task_outcomes"),
            _load_outcomes(Path(args.dag_outcomes_attribute_blind) if args.dag_outcomes_attribute_blind else None, "dag_outcomes"),
        ),
    }

    rejection_summaries = {
        "attribute_only": _load_rejection_summary(
            Path(args.attribute_only_rejection_summary) if args.attribute_only_rejection_summary else None
        ),
        "attribute_blind": _load_rejection_summary(
            Path(args.attribute_blind_rejection_summary) if args.attribute_blind_rejection_summary else None
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_rows = _build_bucket_rows(events_by_ablation)
    summary = {
        "metadata": {
            "attribute_only_jsonl": str(args.attribute_only_jsonl),
            "attribute_blind_jsonl": str(args.attribute_blind_jsonl),
            "observed_outcome_joined": outcomes_joined,
            "observed_outcome_note": "Observed outcome fields are only present when optional outcome files are provided and join succeeds.",
        },
        "overall": {ablation: _summarize_events(events) for ablation, events in events_by_ablation.items()},
    }
    (output_dir / "assignment_behavior_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "assignment_behavior_by_bucket.csv", bucket_rows)
    plotted = _plot_figures(output_dir, events_by_ablation)
    summary["metadata"]["figures_written"] = plotted
    (output_dir / "assignment_behavior_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(output_dir / "assignment_behavior_report.md", summary, bucket_rows, outcomes_joined, rejection_summaries)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare independent-run assignment behavior for attribute ablations.")
    parser.add_argument("--attribute_only_jsonl", required=True)
    parser.add_argument("--attribute_blind_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_outcomes_attribute_only", default="")
    parser.add_argument("--task_outcomes_attribute_blind", default="")
    parser.add_argument("--dag_outcomes_attribute_only", default="")
    parser.add_argument("--dag_outcomes_attribute_blind", default="")
    parser.add_argument("--attribute_only_rejection_summary", default="")
    parser.add_argument("--attribute_blind_rejection_summary", default="")
    args = parser.parse_args()
    summary = analyze(args)
    print(f"saved_summary={Path(args.output_dir) / 'assignment_behavior_summary.json'}")
    print(f"saved_bucket_csv={Path(args.output_dir) / 'assignment_behavior_by_bucket.csv'}")
    print(f"saved_report={Path(args.output_dir) / 'assignment_behavior_report.md'}")
    for ablation, values in summary["overall"].items():
        print(
            f"{ablation}: events={values.get('event_count')} "
            f"score_rate={values.get('score_selected_rate')} "
            f"fallback_rate={values.get('fallback_selected_rate')} "
            f"guard_clamp_rate={values.get('guard_clamp_rate')}"
        )


if __name__ == "__main__":
    main()
