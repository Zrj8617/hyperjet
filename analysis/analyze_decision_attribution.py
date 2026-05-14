from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_attribution(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    dag_rows: list[dict[str, Any]] = []
    files = sorted(input_dir.glob("*_attribution.json"))
    if not files:
        raise FileNotFoundError(f"No *_attribution.json files found in {input_dir}")
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        mode = data.get("mode")
        seed = data.get("seed")
        for row in data.get("assignments", []):
            row = dict(row)
            row["mode"] = mode
            row["seed"] = seed
            assignment_rows.append(row)
        for row in data.get("task_outcomes", []):
            row = dict(row)
            row["mode"] = mode
            row["seed"] = seed
            task_rows.append(row)
        for row in data.get("dag_outcomes", []):
            row = dict(row)
            row["mode"] = mode
            row["seed"] = seed
            dag_rows.append(row)
    return pd.DataFrame(assignment_rows), pd.DataFrame(task_rows), pd.DataFrame(dag_rows)


def _as_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 0.0
    return float(mask.fillna(False).mean())


def _quantile(series: pd.Series, q: float) -> float:
    values = _as_num(series).dropna()
    if values.empty:
        return 0.0
    return float(values.quantile(q))


def _cost_stats(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "count": 0.0,
            "mean_delta_planned_finish": 0.0,
            "median_delta_planned_finish": 0.0,
            "p90_delta_planned_finish": 0.0,
            "p95_delta_planned_finish": 0.0,
            "pct_delta_planned_finish_gt_0": 0.0,
            "pct_delta_planned_finish_gt_0.5": 0.0,
            "pct_delta_planned_finish_gt_1.0": 0.0,
            "mean_delta_deadline_margin": 0.0,
            "median_delta_deadline_margin": 0.0,
            "pct_delta_deadline_margin_lt_0": 0.0,
            "mean_delta_transmission_time": 0.0,
            "mean_delta_execution_time": 0.0,
            "mean_delta_queue_length": 0.0,
            "mean_delta_total_energy": 0.0,
            "teacher_mean_delta_planned_finish": 0.0,
            "teacher_pct_delta_planned_finish_gt_0": 0.0,
            "student_vs_teacher_mean_delta_finish": 0.0,
        }
    finish = _as_num(df.get("delta_planned_finish", pd.Series(dtype=float)))
    margin = _as_num(df.get("delta_deadline_margin", pd.Series(dtype=float)))
    teacher_finish = _as_num(df.get("teacher_delta_planned_finish", pd.Series(dtype=float)))
    student_teacher = _as_num(df.get("student_delta_planned_finish_vs_teacher", pd.Series(dtype=float)))
    return {
        "count": float(len(df)),
        "mean_delta_planned_finish": float(finish.mean()) if finish.notna().any() else 0.0,
        "median_delta_planned_finish": float(finish.median()) if finish.notna().any() else 0.0,
        "p90_delta_planned_finish": _quantile(finish, 0.90),
        "p95_delta_planned_finish": _quantile(finish, 0.95),
        "pct_delta_planned_finish_gt_0": _rate(finish > 0.0),
        "pct_delta_planned_finish_gt_0.5": _rate(finish > 0.5),
        "pct_delta_planned_finish_gt_1.0": _rate(finish > 1.0),
        "mean_delta_deadline_margin": float(margin.mean()) if margin.notna().any() else 0.0,
        "median_delta_deadline_margin": float(margin.median()) if margin.notna().any() else 0.0,
        "pct_delta_deadline_margin_lt_0": _rate(margin < 0.0),
        "mean_delta_transmission_time": float(_as_num(df.get("delta_transmission_time", pd.Series(dtype=float))).mean()),
        "mean_delta_execution_time": float(_as_num(df.get("delta_execution_time", pd.Series(dtype=float))).mean()),
        "mean_delta_queue_length": float(_as_num(df.get("delta_queue_length", pd.Series(dtype=float))).mean()),
        "mean_delta_total_energy": float(_as_num(df.get("delta_total_energy", pd.Series(dtype=float))).mean()),
        "teacher_mean_delta_planned_finish": float(teacher_finish.mean()) if teacher_finish.notna().any() else 0.0,
        "teacher_pct_delta_planned_finish_gt_0": _rate(teacher_finish > 0.0),
        "student_vs_teacher_mean_delta_finish": float(student_teacher.mean()) if student_teacher.notna().any() else 0.0,
    }


def _coverage_stats(df: pd.DataFrame) -> dict[str, float]:
    attempts = len(df)
    score_selected = df["selection_mode"].eq("score") if "selection_mode" in df else pd.Series(dtype=bool)
    fallback_selected = df["selection_mode"].eq("fallback") if "selection_mode" in df else pd.Series(dtype=bool)
    disagreements = df["disagrees_with_heuristic"].fillna(False).astype(bool) if "disagrees_with_heuristic" in df else pd.Series(dtype=bool)
    teacher_disagree = df.get("teacher_disagrees_with_heuristic", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    student_teacher = df.get("student_disagrees_with_teacher", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    return {
        "assignment_attempts": float(attempts),
        "score_selected_count": float(score_selected.sum()),
        "fallback_selected_count": float(fallback_selected.sum()),
        "score_selected_rate": float(score_selected.mean()) if attempts else 0.0,
        "disagreement_count": float(disagreements.sum()),
        "disagreement_rate": float(disagreements.mean()) if attempts else 0.0,
        "teacher_disagreement_count": float(teacher_disagree.sum()),
        "teacher_disagreement_rate": float(teacher_disagree.mean()) if attempts else 0.0,
        "student_teacher_disagreement_count": float(student_teacher.sum()),
        "student_teacher_disagreement_rate": float(student_teacher.mean()) if attempts else 0.0,
    }


def _bucket_frames(assignments: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    frames: list[tuple[str, str, pd.DataFrame]] = [("all", "all", assignments)]
    if assignments.empty:
        return frames
    frames.extend(
        [
            ("risk", "high_risk_task", assignments[assignments["is_high_risk_task"].fillna(False).astype(bool)]),
            ("risk", "non_high_risk_task", assignments[~assignments["is_high_risk_task"].fillna(False).astype(bool)]),
            ("critical", "critical_path_task", assignments[assignments["is_critical_path_task"].fillna(False).astype(bool)]),
            ("critical", "non_critical_path_task", assignments[~assignments["is_critical_path_task"].fillna(False).astype(bool)]),
        ]
    )
    for task_type, group in assignments.groupby("task_type", dropna=False):
        frames.append(("task_type", str(task_type), group))
    candidate = _as_num(assignments["candidate_count"])
    frames.extend(
        [
            ("candidate_count_bucket", "1", assignments[candidate == 1]),
            ("candidate_count_bucket", "2", assignments[candidate == 2]),
            ("candidate_count_bucket", "3+", assignments[candidate >= 3]),
        ]
    )
    slack = _as_num(assignments["task_slack"])
    frames.extend(
        [
            ("task_slack_bucket", "<=2", assignments[slack <= 2]),
            ("task_slack_bucket", "3-5", assignments[(slack > 2) & (slack <= 5)]),
            ("task_slack_bucket", "6-8", assignments[(slack > 5) & (slack <= 8)]),
            ("task_slack_bucket", ">8", assignments[slack > 8]),
        ]
    )
    return frames


def _task_outcome_stats(assignments: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty or tasks.empty:
        return pd.DataFrame()
    merged = assignments.merge(
        tasks,
        on=["mode", "seed", "episode", "task_id", "dag_id"],
        how="left",
        suffixes=("", "_outcome"),
    )
    full = merged[merged["mode"].eq("full")].copy()
    groups = {
        "score_selected_and_agree": full[full["selection_mode"].eq("score") & ~full["disagrees_with_heuristic"].fillna(False).astype(bool)],
        "score_selected_and_disagree": full[full["selection_mode"].eq("score") & full["disagrees_with_heuristic"].fillna(False).astype(bool)],
        "fallback_selected_in_full": full[full["selection_mode"].eq("fallback")],
        "fallback_baseline": merged[merged["mode"].eq("fallback")],
    }
    rows: list[dict[str, float | str]] = []
    for name, group in groups.items():
        rows.append(
            {
                "group": name,
                "bucket": "all",
                "count": float(len(group)),
                "task_finish_rate": _rate(group.get("finished", pd.Series(dtype=bool))),
                "task_drop_rate": _rate(group.get("dropped", pd.Series(dtype=bool))),
                "task_on_time_rate": _rate(group.get("finished_on_time", pd.Series(dtype=bool))),
                "avg_task_completion_time": float(_as_num(group.get("completion_time", pd.Series(dtype=float))).mean()),
            }
        )
    score = full[full["selection_mode"].eq("score")].copy()
    delta = _as_num(score["delta_planned_finish"])
    delta_buckets = {
        "<=0": score[delta <= 0],
        "(0,0.5]": score[(delta > 0) & (delta <= 0.5)],
        "(0.5,1.0]": score[(delta > 0.5) & (delta <= 1.0)],
        ">1.0": score[delta > 1.0],
    }
    for name, group in delta_buckets.items():
        rows.append(
            {
                "group": "score_selected_by_delta_finish",
                "bucket": name,
                "count": float(len(group)),
                "task_finish_rate": _rate(group.get("finished", pd.Series(dtype=bool))),
                "task_drop_rate": _rate(group.get("dropped", pd.Series(dtype=bool))),
                "task_on_time_rate": _rate(group.get("finished_on_time", pd.Series(dtype=bool))),
                "avg_task_completion_time": float(_as_num(group.get("completion_time", pd.Series(dtype=float))).mean()),
            }
        )
    return pd.DataFrame(rows)


def _dag_effects(assignments: pd.DataFrame, dags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if assignments.empty or dags.empty:
        return pd.DataFrame(), pd.DataFrame()
    full = assignments[assignments["mode"].eq("full")].copy()
    full["is_disagreement"] = full["disagrees_with_heuristic"].fillna(False).astype(bool)
    full["is_critical_disagreement"] = full["is_disagreement"] & full["is_critical_path_task"].fillna(False).astype(bool)
    full["is_high_risk_disagreement"] = full["is_disagreement"] & full["is_high_risk_task"].fillna(False).astype(bool)
    full["is_large_delta"] = _as_num(full["delta_planned_finish"]) > 1.0
    grouped = full.groupby(["mode", "seed", "episode", "dag_id"], dropna=False).agg(
        num_score_selected_assignments=("selection_mode", lambda x: float((x == "score").sum())),
        num_disagreements=("is_disagreement", "sum"),
        num_critical_disagreements=("is_critical_disagreement", "sum"),
        num_high_risk_disagreements=("is_high_risk_disagreement", "sum"),
        max_delta_planned_finish=("delta_planned_finish", "max"),
        mean_delta_planned_finish=("delta_planned_finish", "mean"),
        has_large_delta_finish=("is_large_delta", "max"),
    ).reset_index()
    dag_full = dags[dags["mode"].eq("full")].merge(grouped, on=["mode", "seed", "episode", "dag_id"], how="left")
    fill_cols = [
        "num_score_selected_assignments",
        "num_disagreements",
        "num_critical_disagreements",
        "num_high_risk_disagreements",
        "has_large_delta_finish",
    ]
    for col in fill_cols:
        dag_full[col] = dag_full[col].fillna(0)
    groups = {
        "no_disagreement_dag": dag_full[dag_full["num_disagreements"] <= 0],
        "any_disagreement_dag": dag_full[dag_full["num_disagreements"] > 0],
        "critical_disagreement_dag": dag_full[dag_full["num_critical_disagreements"] > 0],
        "high_risk_disagreement_dag": dag_full[dag_full["num_high_risk_disagreements"] > 0],
        "large_delta_disagreement_dag": dag_full[dag_full["has_large_delta_finish"].astype(bool)],
    }
    rows = []
    for name, group in groups.items():
        rows.append(
            {
                "group": name,
                "count": float(len(group)),
                "dag_success_rate": _rate(group.get("successful", pd.Series(dtype=bool))),
                "dag_failure_rate": _rate(group.get("failed", pd.Series(dtype=bool))),
                "dag_on_time_success_rate": _rate(group.get("on_time_successful", pd.Series(dtype=bool))),
                "avg_dag_completion_time": float(_as_num(group.get("completion_time", pd.Series(dtype=float))).mean()),
                "avg_dag_tardiness": float(_as_num(group.get("tardiness", pd.Series(dtype=float))).mean()),
            }
        )
    return dag_full, pd.DataFrame(rows)


def _gate_stats(assignments: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame()
    merged = assignments.merge(
        tasks,
        on=["mode", "seed", "episode", "task_id", "dag_id"],
        how="left",
        suffixes=("", "_outcome"),
    )
    full = merged[merged["mode"].eq("full")]
    groups = {
        "score_selected_high_risk": full[full["selection_mode"].eq("score") & full["is_high_risk_task"].fillna(False).astype(bool)],
        "score_selected_non_high_risk": full[full["selection_mode"].eq("score") & ~full["is_high_risk_task"].fillna(False).astype(bool)],
        "fallback_only_in_full": full[full["selection_mode"].eq("fallback")],
    }
    rows = []
    for name, group in groups.items():
        rows.append(
            {
                "group": name,
                "count": float(len(group)),
                "mean_candidate_count": float(_as_num(group.get("candidate_count", pd.Series(dtype=float))).mean()),
                "mean_task_slack": float(_as_num(group.get("task_slack", pd.Series(dtype=float))).mean()),
                "mean_dag_remaining_slack": float(_as_num(group.get("dag_remaining_slack", pd.Series(dtype=float))).mean()),
                "mean_num_successors": float(_as_num(group.get("num_successors", pd.Series(dtype=float))).mean()),
                "mean_dag_completion_ratio": float(_as_num(group.get("dag_completion_ratio", pd.Series(dtype=float))).mean()),
                "mean_heuristic_planned_finish": float(_as_num(group.get("heuristic_planned_finish", pd.Series(dtype=float))).mean()),
                "mean_heuristic_deadline_margin": float(_as_num(group.get("heuristic_deadline_margin", pd.Series(dtype=float))).mean()),
                "disagreement_rate": _rate(group.get("disagrees_with_heuristic", pd.Series(dtype=bool))),
                "drop_rate": _rate(group.get("dropped", pd.Series(dtype=bool))),
                "on_time_rate": _rate(group.get("finished_on_time", pd.Series(dtype=bool))),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    output_path: Path,
    summary: pd.DataFrame,
    by_bucket: pd.DataFrame,
    task_stats: pd.DataFrame,
    dag_summary: pd.DataFrame,
    gate: pd.DataFrame,
) -> None:
    def val(metric: str) -> float:
        row = summary[summary["metric"].eq(metric)]
        if row.empty:
            return 0.0
        return float(row.iloc[0]["value"])

    any_dag = dag_summary[dag_summary["group"].eq("any_disagreement_dag")]
    no_dag = dag_summary[dag_summary["group"].eq("no_disagreement_dag")]
    crit_dag = dag_summary[dag_summary["group"].eq("critical_disagreement_dag")]
    any_fail = float(any_dag.iloc[0]["dag_failure_rate"]) if not any_dag.empty else 0.0
    no_fail = float(no_dag.iloc[0]["dag_failure_rate"]) if not no_dag.empty else 0.0
    crit_fail = float(crit_dag.iloc[0]["dag_failure_rate"]) if not crit_dag.empty else 0.0

    teacher_bias = val("teacher_disagreement_rate") > 0.05 and val("teacher_mean_delta_planned_finish_score_selected") > 0
    student_instability = val("student_teacher_disagreement_rate") > 0.05
    gate_exposure = False
    if not gate.empty and {"score_selected_high_risk", "fallback_only_in_full"}.issubset(set(gate["group"])):
        high = gate[gate["group"].eq("score_selected_high_risk")].iloc[0]
        fb = gate[gate["group"].eq("fallback_only_in_full")].iloc[0]
        gate_exposure = float(high["mean_candidate_count"]) < float(fb["mean_candidate_count"]) or float(high["mean_task_slack"]) < float(fb["mean_task_slack"])
    static_strength = abs(val("mean_delta_planned_finish_score_selected_disagree")) < 0.5 and val("pct_delta_planned_finish_gt_0_score_selected_disagree") > 0.5

    likely = []
    if teacher_bias:
        likely.append("teacher bias")
    if student_instability:
        likely.append("student ranking instability")
    if gate_exposure:
        likely.append("gate hardest-case exposure")
    if static_strength:
        likely.append("static baseline strength")
    likely_text = ", ".join(likely) if likely else "mixed / not decisive"

    lines = [
        "# Decision Attribution Report",
        "",
        "## Key MVP Metrics",
        "",
        f"- score_selected_rate: {val('score_selected_rate'):.4f}",
        f"- disagreement_rate: {val('disagreement_rate'):.4f}",
        f"- mean_delta_planned_finish(score selected): {val('mean_delta_planned_finish_score_selected'):.4f}",
        f"- pct_delta_planned_finish_gt_0(score selected): {val('pct_delta_planned_finish_gt_0_score_selected'):.4f}",
        f"- mean_delta_deadline_margin(score selected): {val('mean_delta_deadline_margin_score_selected'):.4f}",
        f"- task_drop_rate disagreement vs agree: {val('task_drop_rate_score_disagree'):.4f} vs {val('task_drop_rate_score_agree'):.4f}",
        f"- DAG failure rate any_disagreement vs no_disagreement: {any_fail:.4f} vs {no_fail:.4f}",
        f"- DAG failure rate critical_disagreement vs no_disagreement: {crit_fail:.4f} vs {no_fail:.4f}",
        "",
        "## Evidence-Based Diagnosis",
        "",
        f"The current degradation is most consistent with: **{likely_text}**.",
        "",
        "This is a diagnostic hint, not a proof. Use the CSV files to decide whether to change teacher constraints, ranking imitation, or selective gate thresholds.",
        "",
        "## Tables",
        "",
        "Detailed CSV outputs are written next to this report:",
        "",
        "- decision_attribution_summary.csv",
        "- decision_attribution_by_bucket.csv",
        "- decision_attribution_task_outcomes.csv",
        "- decision_attribution_dag_effects.csv",
        "- decision_attribution_dag_groups.csv",
        "- decision_attribution_gate_quality.csv",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze static scheduler decision attribution outputs.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "decision_attribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments, tasks, dags = _load_attribution(input_dir)
    full = assignments[assignments["mode"].eq("full")].copy()
    score_selected = full[full["selection_mode"].eq("score")]
    score_disagree = score_selected[score_selected["disagrees_with_heuristic"].fillna(False).astype(bool)]
    score_agree = score_selected[~score_selected["disagrees_with_heuristic"].fillna(False).astype(bool)]

    summary_items: dict[str, float] = {}
    summary_items.update(_coverage_stats(full))
    for prefix, frame in {
        "score_selected": score_selected,
        "score_selected_disagree": score_disagree,
        "score_selected_agree": score_agree,
        "score_selected_high_risk": score_selected[score_selected["is_high_risk_task"].fillna(False).astype(bool)],
        "score_selected_critical": score_selected[score_selected["is_critical_path_task"].fillna(False).astype(bool)],
        "score_selected_candidate_scarce": score_selected[_as_num(score_selected["candidate_count"]) <= 2],
    }.items():
        for key, value in _cost_stats(frame).items():
            summary_items[f"{key}_{prefix}"] = value

    task_stats = _task_outcome_stats(assignments, tasks)
    for group_name, metric_prefix in [
        ("score_selected_and_disagree", "score_disagree"),
        ("score_selected_and_agree", "score_agree"),
    ]:
        row = task_stats[(task_stats["group"].eq(group_name)) & (task_stats["bucket"].eq("all"))]
        if not row.empty:
            summary_items[f"task_drop_rate_{metric_prefix}"] = float(row.iloc[0]["task_drop_rate"])
            summary_items[f"task_on_time_rate_{metric_prefix}"] = float(row.iloc[0]["task_on_time_rate"])

    dag_effects, dag_groups = _dag_effects(assignments, dags)
    gate = _gate_stats(assignments, tasks)

    summary = pd.DataFrame([{"metric": key, "value": value} for key, value in sorted(summary_items.items())])
    bucket_rows = []
    for bucket_type, bucket, frame in _bucket_frames(full):
        row: dict[str, float | str] = {"bucket_type": bucket_type, "bucket": bucket}
        row.update(_coverage_stats(frame))
        score_frame = frame[frame["selection_mode"].eq("score")]
        disagree_frame = score_frame[score_frame["disagrees_with_heuristic"].fillna(False).astype(bool)]
        for key, value in _cost_stats(score_frame).items():
            row[f"score_{key}"] = value
        for key, value in _cost_stats(disagree_frame).items():
            row[f"disagree_{key}"] = value
        bucket_rows.append(row)
    by_bucket = pd.DataFrame(bucket_rows)

    summary.to_csv(output_dir / "decision_attribution_summary.csv", index=False)
    by_bucket.to_csv(output_dir / "decision_attribution_by_bucket.csv", index=False)
    task_stats.to_csv(output_dir / "decision_attribution_task_outcomes.csv", index=False)
    dag_effects.to_csv(output_dir / "decision_attribution_dag_effects.csv", index=False)
    dag_groups.to_csv(output_dir / "decision_attribution_dag_groups.csv", index=False)
    gate.to_csv(output_dir / "decision_attribution_gate_quality.csv", index=False)
    _write_report(output_dir / "decision_attribution_report.md", summary, by_bucket, task_stats, dag_groups, gate)
    print(f"assignments={len(assignments)} tasks={len(tasks)} dags={len(dags)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
