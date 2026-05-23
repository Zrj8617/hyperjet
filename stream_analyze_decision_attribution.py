from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


JSONScalar = str | int | float | bool | None


def _parse_mode_seed(path: Path) -> tuple[str, int]:
    name = path.name
    mode = "full" if "_full_" in name else "fallback"
    match = re.search(r"_seed(\d+)", name)
    seed = int(match.group(1)) if match else 0
    return mode, seed


def iter_json_array(path: Path, key: str, chunk_size: int = 1024 * 1024) -> Iterable[dict[str, Any]]:
    """Yield objects from a top-level JSON array without loading the full file."""
    decoder = json.JSONDecoder()
    needle = f'"{key}"'
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        pos = 0
        found = False
        eof = False

        while not found:
            chunk = f.read(chunk_size)
            if not chunk:
                return
            buf += chunk
            idx = buf.find(needle)
            if idx < 0:
                keep = max(len(needle) + 32, 128)
                buf = buf[-keep:]
                continue
            while True:
                bracket = buf.find("[", idx + len(needle))
                if bracket >= 0:
                    pos = bracket + 1
                    found = True
                    break
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                buf += chunk

        while True:
            while True:
                if pos >= len(buf):
                    chunk = f.read(chunk_size)
                    if not chunk:
                        return
                    buf += chunk
                while pos < len(buf) and buf[pos] in " \r\n\t,":
                    pos += 1
                if pos < len(buf):
                    break

            if buf[pos] == "]":
                return

            while True:
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                    yield obj
                    pos = end
                    if pos > chunk_size:
                        buf = buf[pos:]
                        pos = 0
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    chunk = f.read(chunk_size)
                    if not chunk:
                        eof = True
                    else:
                        buf += chunk


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


@dataclass
class CoverageAgg:
    attempts: int = 0
    score_selected: int = 0
    fallback_selected: int = 0
    disagreements: int = 0
    teacher_disagreements: int = 0
    student_teacher_disagreements: int = 0

    def update(self, row: dict[str, Any]) -> None:
        self.attempts += 1
        mode = row.get("selection_mode")
        if mode == "score":
            self.score_selected += 1
        elif mode == "fallback":
            self.fallback_selected += 1
        if _bool(row.get("disagrees_with_heuristic")):
            self.disagreements += 1
        if _bool(row.get("teacher_disagrees_with_heuristic")):
            self.teacher_disagreements += 1
        if _bool(row.get("student_disagrees_with_teacher")):
            self.student_teacher_disagreements += 1

    def as_dict(self) -> dict[str, float]:
        denom = max(self.attempts, 1)
        return {
            "assignment_attempts": float(self.attempts),
            "score_selected_count": float(self.score_selected),
            "fallback_selected_count": float(self.fallback_selected),
            "score_selected_rate": self.score_selected / denom,
            "disagreement_count": float(self.disagreements),
            "disagreement_rate": self.disagreements / denom,
            "teacher_disagreement_count": float(self.teacher_disagreements),
            "teacher_disagreement_rate": self.teacher_disagreements / denom,
            "student_teacher_disagreement_count": float(self.student_teacher_disagreements),
            "student_teacher_disagreement_rate": self.student_teacher_disagreements / denom,
        }


@dataclass
class CostAgg:
    count: int = 0
    delta_finish: list[float] = field(default_factory=list)
    delta_margin: list[float] = field(default_factory=list)
    delta_transmission_sum: float = 0.0
    delta_transmission_count: int = 0
    delta_execution_sum: float = 0.0
    delta_execution_count: int = 0
    delta_queue_sum: float = 0.0
    delta_queue_count: int = 0
    delta_energy_sum: float = 0.0
    delta_energy_count: int = 0
    teacher_delta_finish: list[float] = field(default_factory=list)
    student_teacher_delta_finish: list[float] = field(default_factory=list)

    def update(self, row: dict[str, Any]) -> None:
        self.count += 1
        if (value := _safe_float(row.get("delta_planned_finish"))) is not None:
            self.delta_finish.append(value)
        if (value := _safe_float(row.get("delta_deadline_margin"))) is not None:
            self.delta_margin.append(value)
        if (value := _safe_float(row.get("delta_transmission_time"))) is not None:
            self.delta_transmission_sum += value
            self.delta_transmission_count += 1
        if (value := _safe_float(row.get("delta_execution_time"))) is not None:
            self.delta_execution_sum += value
            self.delta_execution_count += 1
        if (value := _safe_float(row.get("delta_queue_length"))) is not None:
            self.delta_queue_sum += value
            self.delta_queue_count += 1
        if (value := _safe_float(row.get("delta_total_energy"))) is not None:
            self.delta_energy_sum += value
            self.delta_energy_count += 1
        if (value := _safe_float(row.get("teacher_delta_planned_finish"))) is not None:
            self.teacher_delta_finish.append(value)
        if (value := _safe_float(row.get("student_delta_planned_finish_vs_teacher"))) is not None:
            self.student_teacher_delta_finish.append(value)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _median(values: list[float]) -> float:
        return float(np.median(values)) if values else 0.0

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        return float(np.quantile(values, q)) if values else 0.0

    @staticmethod
    def _rate(values: list[float], pred) -> float:
        return float(np.mean([pred(value) for value in values])) if values else 0.0

    @staticmethod
    def _safe_avg(total: float, count: int) -> float:
        return total / max(count, 1)

    def as_dict(self) -> dict[str, float]:
        return {
            "count": float(self.count),
            "mean_delta_planned_finish": self._mean(self.delta_finish),
            "median_delta_planned_finish": self._median(self.delta_finish),
            "p90_delta_planned_finish": self._quantile(self.delta_finish, 0.90),
            "p95_delta_planned_finish": self._quantile(self.delta_finish, 0.95),
            "pct_delta_planned_finish_gt_0": self._rate(self.delta_finish, lambda value: value > 0.0),
            "pct_delta_planned_finish_gt_0.5": self._rate(self.delta_finish, lambda value: value > 0.5),
            "pct_delta_planned_finish_gt_1.0": self._rate(self.delta_finish, lambda value: value > 1.0),
            "mean_delta_deadline_margin": self._mean(self.delta_margin),
            "median_delta_deadline_margin": self._median(self.delta_margin),
            "pct_delta_deadline_margin_lt_0": self._rate(self.delta_margin, lambda value: value < 0.0),
            "mean_delta_transmission_time": self._safe_avg(self.delta_transmission_sum, self.delta_transmission_count),
            "mean_delta_execution_time": self._safe_avg(self.delta_execution_sum, self.delta_execution_count),
            "mean_delta_queue_length": self._safe_avg(self.delta_queue_sum, self.delta_queue_count),
            "mean_delta_total_energy": self._safe_avg(self.delta_energy_sum, self.delta_energy_count),
            "teacher_mean_delta_planned_finish": self._mean(self.teacher_delta_finish),
            "teacher_pct_delta_planned_finish_gt_0": self._rate(self.teacher_delta_finish, lambda value: value > 0.0),
            "student_vs_teacher_mean_delta_finish": self._mean(self.student_teacher_delta_finish),
        }


@dataclass
class OutcomeAgg:
    count: int = 0
    finished: int = 0
    dropped: int = 0
    on_time: int = 0
    completion_sum: float = 0.0
    completion_count: int = 0

    def update(self, outcome: dict[str, Any] | None) -> None:
        self.count += 1
        if not outcome:
            return
        if _bool(outcome.get("finished")):
            self.finished += 1
        if _bool(outcome.get("dropped")):
            self.dropped += 1
        if _bool(outcome.get("finished_on_time")):
            self.on_time += 1
        if (value := _safe_float(outcome.get("completion_time"))) is not None:
            self.completion_sum += value
            self.completion_count += 1

    def as_dict(self) -> dict[str, float]:
        denom = max(self.count, 1)
        return {
            "count": float(self.count),
            "task_finish_rate": self.finished / denom,
            "task_drop_rate": self.dropped / denom,
            "task_on_time_rate": self.on_time / denom,
            "avg_task_completion_time": self.completion_sum / max(self.completion_count, 1),
        }


@dataclass
class GateAgg:
    count: int = 0
    candidate_sum: float = 0.0
    slack_sum: float = 0.0
    dag_slack_sum: float = 0.0
    successors_sum: float = 0.0
    completion_ratio_sum: float = 0.0
    heuristic_finish_sum: float = 0.0
    heuristic_finish_count: int = 0
    heuristic_margin_sum: float = 0.0
    heuristic_margin_count: int = 0
    disagreements: int = 0
    dropped: int = 0
    on_time: int = 0

    def update(self, row: dict[str, Any], outcome: dict[str, Any] | None) -> None:
        self.count += 1
        self.candidate_sum += _safe_float(row.get("candidate_count")) or 0.0
        self.slack_sum += _safe_float(row.get("task_slack")) or 0.0
        self.dag_slack_sum += _safe_float(row.get("dag_remaining_slack")) or 0.0
        self.successors_sum += _safe_float(row.get("num_successors")) or 0.0
        self.completion_ratio_sum += _safe_float(row.get("dag_completion_ratio")) or 0.0
        if (value := _safe_float(row.get("heuristic_planned_finish"))) is not None:
            self.heuristic_finish_sum += value
            self.heuristic_finish_count += 1
        if (value := _safe_float(row.get("heuristic_deadline_margin"))) is not None:
            self.heuristic_margin_sum += value
            self.heuristic_margin_count += 1
        if _bool(row.get("disagrees_with_heuristic")):
            self.disagreements += 1
        if outcome and _bool(outcome.get("dropped")):
            self.dropped += 1
        if outcome and _bool(outcome.get("finished_on_time")):
            self.on_time += 1

    def as_dict(self) -> dict[str, float]:
        denom = max(self.count, 1)
        return {
            "count": float(self.count),
            "mean_candidate_count": self.candidate_sum / denom,
            "mean_task_slack": self.slack_sum / denom,
            "mean_dag_remaining_slack": self.dag_slack_sum / denom,
            "mean_num_successors": self.successors_sum / denom,
            "mean_dag_completion_ratio": self.completion_ratio_sum / denom,
            "mean_heuristic_planned_finish": self.heuristic_finish_sum / max(self.heuristic_finish_count, 1),
            "mean_heuristic_deadline_margin": self.heuristic_margin_sum / max(self.heuristic_margin_count, 1),
            "disagreement_rate": self.disagreements / denom,
            "drop_rate": self.dropped / denom,
            "on_time_rate": self.on_time / denom,
        }


@dataclass
class DagAssignAgg:
    num_score_selected: int = 0
    num_disagreements: int = 0
    num_critical_disagreements: int = 0
    num_high_risk_disagreements: int = 0
    delta_sum: float = 0.0
    delta_count: int = 0
    max_delta: float | None = None
    has_large_delta: bool = False

    def update(self, row: dict[str, Any]) -> None:
        if row.get("selection_mode") == "score":
            self.num_score_selected += 1
        is_disagreement = _bool(row.get("disagrees_with_heuristic"))
        if is_disagreement:
            self.num_disagreements += 1
            if _bool(row.get("is_critical_path_task")):
                self.num_critical_disagreements += 1
            if _bool(row.get("is_high_risk_task")):
                self.num_high_risk_disagreements += 1
        if (value := _safe_float(row.get("delta_planned_finish"))) is not None:
            self.delta_sum += value
            self.delta_count += 1
            self.max_delta = value if self.max_delta is None else max(self.max_delta, value)
            if value > 1.0:
                self.has_large_delta = True


@dataclass
class DagOutcomeAgg:
    count: int = 0
    success: int = 0
    failed: int = 0
    on_time: int = 0
    completion_sum: float = 0.0
    completion_count: int = 0
    tardiness_sum: float = 0.0
    tardiness_count: int = 0

    def update(self, outcome: dict[str, Any]) -> None:
        self.count += 1
        if _bool(outcome.get("successful")):
            self.success += 1
        if _bool(outcome.get("failed")):
            self.failed += 1
        if _bool(outcome.get("on_time_successful")):
            self.on_time += 1
        if (value := _safe_float(outcome.get("completion_time"))) is not None:
            self.completion_sum += value
            self.completion_count += 1
        if (value := _safe_float(outcome.get("tardiness"))) is not None:
            self.tardiness_sum += value
            self.tardiness_count += 1

    def as_dict(self) -> dict[str, float]:
        denom = max(self.count, 1)
        return {
            "count": float(self.count),
            "dag_success_rate": self.success / denom,
            "dag_failure_rate": self.failed / denom,
            "dag_on_time_success_rate": self.on_time / denom,
            "avg_dag_completion_time": self.completion_sum / max(self.completion_count, 1),
            "avg_dag_tardiness": self.tardiness_sum / max(self.tardiness_count, 1),
        }


def _task_key(mode: str, seed: int, row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (mode, seed, int(row["episode"]), str(row["task_id"]))


def _dag_key(mode: str, seed: int, row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (mode, seed, int(row["episode"]), str(row["dag_id"]))


def _delta_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value <= 0.0:
        return "<=0"
    if value <= 0.5:
        return "(0,0.5]"
    if value <= 1.0:
        return "(0.5,1.0]"
    return ">1.0"


def _bucket_keys(row: dict[str, Any]) -> list[tuple[str, str]]:
    keys = [("all", "all")]
    high = _bool(row.get("is_high_risk_task"))
    critical = _bool(row.get("is_critical_path_task"))
    keys.append(("risk", "high_risk_task" if high else "non_high_risk_task"))
    keys.append(("critical", "critical_path_task" if critical else "non_critical_path_task"))
    keys.append(("task_type", str(row.get("task_type"))))
    candidate = int(_safe_float(row.get("candidate_count")) or 0)
    if candidate <= 1:
        keys.append(("candidate_count_bucket", "1"))
    elif candidate == 2:
        keys.append(("candidate_count_bucket", "2"))
    else:
        keys.append(("candidate_count_bucket", "3+"))
    slack = _safe_float(row.get("task_slack")) or 0.0
    if slack <= 2:
        keys.append(("task_slack_bucket", "<=2"))
    elif slack <= 5:
        keys.append(("task_slack_bucket", "3-5"))
    elif slack <= 8:
        keys.append(("task_slack_bucket", "6-8"))
    else:
        keys.append(("task_slack_bucket", ">8"))
    return keys


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_outcomes(files: list[Path]) -> tuple[dict[tuple[str, int, int, str], dict[str, Any]], dict[tuple[str, int, int, str], dict[str, Any]]]:
    task_outcomes: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    dag_outcomes: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for path in files:
        mode, seed = _parse_mode_seed(path)
        print(f"[outcomes] {path.name}", flush=True)
        for row in iter_json_array(path, "task_outcomes"):
            task_outcomes[_task_key(mode, seed, row)] = row
        for row in iter_json_array(path, "dag_outcomes"):
            dag_outcomes[_dag_key(mode, seed, row)] = row
    return task_outcomes, dag_outcomes


def analyze(files: list[Path], output_dir: Path) -> None:
    task_outcomes, dag_outcomes = _load_outcomes(files)
    print(f"[outcomes] tasks={len(task_outcomes)} dags={len(dag_outcomes)}", flush=True)

    full_coverage = CoverageAgg()
    bucket_cov: dict[tuple[str, str], CoverageAgg] = defaultdict(CoverageAgg)
    bucket_score_cost: dict[tuple[str, str], CostAgg] = defaultdict(CostAgg)
    bucket_disagree_cost: dict[tuple[str, str], CostAgg] = defaultdict(CostAgg)
    cost_groups: dict[str, CostAgg] = defaultdict(CostAgg)
    task_groups: dict[tuple[str, str], OutcomeAgg] = defaultdict(OutcomeAgg)
    gate_groups: dict[str, GateAgg] = defaultdict(GateAgg)
    dag_assign: dict[tuple[str, int, int, str], DagAssignAgg] = defaultdict(DagAssignAgg)
    assignment_count = 0

    for path in files:
        mode, seed = _parse_mode_seed(path)
        print(f"[assignments] {path.name}", flush=True)
        for row in iter_json_array(path, "assignments"):
            assignment_count += 1
            if assignment_count % 500000 == 0:
                print(f"  processed assignments={assignment_count}", flush=True)

            outcome = task_outcomes.get(_task_key(mode, seed, row))
            if mode == "fallback":
                task_groups[("fallback_baseline", "all")].update(outcome)
                continue

            full_coverage.update(row)
            for key in _bucket_keys(row):
                bucket_cov[key].update(row)

            is_score = row.get("selection_mode") == "score"
            is_disagree = _bool(row.get("disagrees_with_heuristic"))
            is_high = _bool(row.get("is_high_risk_task"))
            is_critical = _bool(row.get("is_critical_path_task"))
            is_scarce = (_safe_float(row.get("candidate_count")) or 0.0) <= 2

            dag_id = row.get("dag_id")
            if dag_id is not None:
                dag_assign[_dag_key(mode, seed, row)].update(row)

            if is_score:
                cost_groups["score_selected"].update(row)
                if is_high:
                    cost_groups["score_selected_high_risk"].update(row)
                if is_critical:
                    cost_groups["score_selected_critical"].update(row)
                if is_scarce:
                    cost_groups["score_selected_candidate_scarce"].update(row)
                for key in _bucket_keys(row):
                    bucket_score_cost[key].update(row)
                if is_disagree:
                    cost_groups["score_selected_disagree"].update(row)
                    task_groups[("score_selected_and_disagree", "all")].update(outcome)
                    for key in _bucket_keys(row):
                        bucket_disagree_cost[key].update(row)
                else:
                    cost_groups["score_selected_agree"].update(row)
                    task_groups[("score_selected_and_agree", "all")].update(outcome)

                delta = _safe_float(row.get("delta_planned_finish"))
                task_groups[("score_selected_by_delta_finish", _delta_bucket(delta))].update(outcome)
                if is_high:
                    gate_groups["score_selected_high_risk"].update(row, outcome)
                else:
                    gate_groups["score_selected_non_high_risk"].update(row, outcome)
            elif row.get("selection_mode") == "fallback":
                task_groups[("fallback_selected_in_full", "all")].update(outcome)
                gate_groups["fallback_only_in_full"].update(row, outcome)

    dag_group_aggs: dict[str, DagOutcomeAgg] = defaultdict(DagOutcomeAgg)
    dag_effect_rows: list[dict[str, Any]] = []
    for key, outcome in dag_outcomes.items():
        mode, seed, episode, dag_id = key
        if mode != "full":
            continue
        agg = dag_assign.get(key, DagAssignAgg())
        row = {
            "mode": mode,
            "seed": seed,
            "episode": episode,
            "dag_id": dag_id,
            "num_score_selected_assignments": agg.num_score_selected,
            "num_disagreements": agg.num_disagreements,
            "num_critical_disagreements": agg.num_critical_disagreements,
            "num_high_risk_disagreements": agg.num_high_risk_disagreements,
            "max_delta_planned_finish": 0.0 if agg.max_delta is None else agg.max_delta,
            "mean_delta_planned_finish": agg.delta_sum / max(agg.delta_count, 1),
            "has_large_delta_finish": agg.has_large_delta,
            "dag_success": _bool(outcome.get("successful")),
            "dag_failure": _bool(outcome.get("failed")),
            "dag_on_time_success": _bool(outcome.get("on_time_successful")),
            "dag_completion_time": outcome.get("completion_time"),
            "dag_tardiness": outcome.get("tardiness"),
        }
        dag_effect_rows.append(row)
        if agg.num_disagreements <= 0:
            dag_group_aggs["no_disagreement_dag"].update(outcome)
        else:
            dag_group_aggs["any_disagreement_dag"].update(outcome)
        if agg.num_critical_disagreements > 0:
            dag_group_aggs["critical_disagreement_dag"].update(outcome)
        if agg.num_high_risk_disagreements > 0:
            dag_group_aggs["high_risk_disagreement_dag"].update(outcome)
        if agg.has_large_delta:
            dag_group_aggs["large_delta_disagreement_dag"].update(outcome)

    summary_rows: list[dict[str, Any]] = []
    for key, value in full_coverage.as_dict().items():
        summary_rows.append({"metric": key, "value": value})
    for group, agg in cost_groups.items():
        for key, value in agg.as_dict().items():
            summary_rows.append({"metric": f"{key}_{group}", "value": value})
    for (group, bucket), agg in task_groups.items():
        if bucket == "all" and group in {"score_selected_and_disagree", "score_selected_and_agree"}:
            prefix = "score_disagree" if group.endswith("disagree") else "score_agree"
            stats = agg.as_dict()
            summary_rows.append({"metric": f"task_drop_rate_{prefix}", "value": stats["task_drop_rate"]})
            summary_rows.append({"metric": f"task_on_time_rate_{prefix}", "value": stats["task_on_time_rate"]})

    by_bucket_rows = []
    for key in sorted(bucket_cov):
        bucket_type, bucket = key
        row: dict[str, Any] = {"bucket_type": bucket_type, "bucket": bucket}
        row.update(bucket_cov[key].as_dict())
        row.update({f"score_{k}": v for k, v in bucket_score_cost[key].as_dict().items()})
        row.update({f"disagree_{k}": v for k, v in bucket_disagree_cost[key].as_dict().items()})
        by_bucket_rows.append(row)

    task_rows = []
    for (group, bucket), agg in sorted(task_groups.items()):
        row = {"group": group, "bucket": bucket}
        row.update(agg.as_dict())
        task_rows.append(row)

    dag_group_rows = []
    for group, agg in sorted(dag_group_aggs.items()):
        row = {"group": group}
        row.update(agg.as_dict())
        dag_group_rows.append(row)

    gate_rows = []
    for group, agg in sorted(gate_groups.items()):
        row = {"group": group}
        row.update(agg.as_dict())
        gate_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "decision_attribution_summary.csv", summary_rows)
    _write_csv(output_dir / "decision_attribution_by_bucket.csv", by_bucket_rows)
    _write_csv(output_dir / "decision_attribution_task_outcomes.csv", task_rows)
    _write_csv(output_dir / "decision_attribution_dag_effects.csv", dag_effect_rows)
    _write_csv(output_dir / "decision_attribution_dag_groups.csv", dag_group_rows)
    _write_csv(output_dir / "decision_attribution_gate_quality.csv", gate_rows)
    _write_report(output_dir / "decision_attribution_report.md", summary_rows, task_rows, dag_group_rows, gate_rows)
    print(f"[done] assignments={assignment_count} output_dir={output_dir}", flush=True)


def _metric(summary_rows: list[dict[str, Any]], name: str) -> float:
    for row in summary_rows:
        if row["metric"] == name:
            return float(row["value"])
    return 0.0


def _group_metric(rows: list[dict[str, Any]], group: str, metric: str) -> float:
    for row in rows:
        if row.get("group") == group:
            return float(row.get(metric, 0.0))
    return 0.0


def _write_report(path: Path, summary_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]], dag_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> None:
    score_rate = _metric(summary_rows, "score_selected_rate")
    disagreement_rate = _metric(summary_rows, "disagreement_rate")
    mean_delta = _metric(summary_rows, "mean_delta_planned_finish_score_selected")
    pct_gt0 = _metric(summary_rows, "pct_delta_planned_finish_gt_0_score_selected")
    mean_margin = _metric(summary_rows, "mean_delta_deadline_margin_score_selected")
    drop_disagree = _group_metric(task_rows, "score_selected_and_disagree", "task_drop_rate")
    drop_agree = _group_metric(task_rows, "score_selected_and_agree", "task_drop_rate")
    any_fail = _group_metric(dag_rows, "any_disagreement_dag", "dag_failure_rate")
    no_fail = _group_metric(dag_rows, "no_disagreement_dag", "dag_failure_rate")
    critical_fail = _group_metric(dag_rows, "critical_disagreement_dag", "dag_failure_rate")
    high_risk_fail = _group_metric(dag_rows, "high_risk_disagreement_dag", "dag_failure_rate")
    teacher_rate = _metric(summary_rows, "teacher_disagreement_rate")
    student_teacher_rate = _metric(summary_rows, "student_teacher_disagreement_rate")
    teacher_delta = _metric(summary_rows, "teacher_mean_delta_planned_finish_score_selected")

    likely = []
    if teacher_rate > 0.05 and teacher_delta > 0:
        likely.append("teacher bias")
    if student_teacher_rate > 0.05:
        likely.append("student ranking instability")
    high_candidate = _group_metric(gate_rows, "score_selected_high_risk", "mean_candidate_count")
    fb_candidate = _group_metric(gate_rows, "fallback_only_in_full", "mean_candidate_count")
    high_slack = _group_metric(gate_rows, "score_selected_high_risk", "mean_task_slack")
    fb_slack = _group_metric(gate_rows, "fallback_only_in_full", "mean_task_slack")
    if high_candidate and fb_candidate and (high_candidate < fb_candidate or high_slack < fb_slack):
        likely.append("gate hardest-case exposure")
    if pct_gt0 > 0.5 and abs(mean_delta) < 0.5:
        likely.append("static baseline strength")
    likely_text = ", ".join(likely) if likely else "mixed / not decisive"

    lines = [
        "# Decision Attribution Report",
        "",
        "## MVP Metrics",
        "",
        f"- score_selected_rate: {score_rate:.6f}",
        f"- disagreement_rate: {disagreement_rate:.6f}",
        f"- mean_delta_planned_finish(score selected): {mean_delta:.6f}",
        f"- pct_delta_planned_finish_gt_0(score selected): {pct_gt0:.6f}",
        f"- mean_delta_deadline_margin(score selected): {mean_margin:.6f}",
        f"- task_drop_rate disagreement vs agree: {drop_disagree:.6f} vs {drop_agree:.6f}",
        f"- dag_failure_rate any_disagreement vs no_disagreement: {any_fail:.6f} vs {no_fail:.6f}",
        f"- dag_failure_rate critical/high-risk disagreement vs no_disagreement: {critical_fail:.6f} / {high_risk_fail:.6f} vs {no_fail:.6f}",
        "",
        "## Diagnosis",
        "",
        f"The degradation is most consistent with: **{likely_text}**.",
        "",
        f"Teacher disagreement rate is {teacher_rate:.6f}; student-teacher disagreement rate is {student_teacher_rate:.6f}.",
        f"High-risk selected tasks have mean candidate count {high_candidate:.3f} and mean slack {high_slack:.3f}; fallback-only tasks have mean candidate count {fb_candidate:.3f} and mean slack {fb_slack:.3f}.",
        "",
        "Detailed CSV files are in this directory.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-analyze huge decision attribution JSON files.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "decision_attribution_stream"
    files = sorted(input_dir.glob("*_attribution.json"))
    if not files:
        raise FileNotFoundError(f"No *_attribution.json files found in {input_dir}")
    analyze(files, output_dir)


if __name__ == "__main__":
    main()
