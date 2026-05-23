from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stream_analyze_decision_attribution import iter_json_array


def _parse_mode_seed(path: Path) -> tuple[str, int]:
    name = path.name
    mode = "full" if "_full_" in name else "fallback"
    match = re.search(r"_seed(\d+)", name)
    seed = int(match.group(1)) if match else 0
    return mode, seed


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _task_key(row: dict[str, Any], seed: int | None = None) -> tuple[int, int, str]:
    row_seed = seed if seed is not None else _safe_int(row.get("seed"))
    return int(row_seed or 0), int(row.get("episode", 0)), str(row.get("task_id", ""))


def _dag_key(row: dict[str, Any], seed: int | None = None) -> tuple[int, int, str]:
    row_seed = seed if seed is not None else _safe_int(row.get("seed"))
    return int(row_seed or 0), int(row.get("episode", 0)), str(row.get("dag_id", ""))


def _delta_bucket(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    if delta <= 0.0:
        return "<=0"
    if delta <= 0.1:
        return "(0,0.1]"
    if delta <= 0.3:
        return "(0.1,0.3]"
    if delta <= 0.5:
        return "(0.3,0.5]"
    if delta <= 1.0:
        return "(0.5,1.0]"
    return ">1.0"


def _candidate_bucket(candidate_count: int | None) -> str:
    if candidate_count is None:
        return "unknown"
    if candidate_count <= 1:
        return "1"
    if candidate_count == 2:
        return "2"
    return "3+"


def _slack_bucket(slack: float | None) -> str:
    if slack is None:
        return "unknown"
    if slack <= 2:
        return "<=2"
    if slack <= 5:
        return "3-5"
    if slack <= 8:
        return "6-8"
    return ">8"


def classify_disagreement(
    row: dict[str, Any],
    task_outcome: dict[str, Any] | None,
    dag_outcome: dict[str, Any] | None,
    *,
    good_delta_tolerance: float = 0.1,
    strong_delta_threshold: float = 0.3,
) -> tuple[str, str]:
    """Classify one score-vs-heuristic disagreement for outcome-aware diagnostics.

    This is intentionally conservative. BAD labels are higher confidence than GOOD labels
    because the data is observational rather than counterfactual.
    """
    delta = _safe_float(row.get("delta_planned_finish"))
    selected_margin = _safe_float(row.get("selected_deadline_margin"))
    heuristic_margin = _safe_float(row.get("heuristic_deadline_margin"))
    is_high_risk = _bool(row.get("is_high_risk_task")) or _bool(row.get("dag_is_high_risk"))
    is_critical = _bool(row.get("is_critical_path_task"))

    task_dropped = _bool(task_outcome.get("dropped")) if task_outcome else False
    task_finished = _bool(task_outcome.get("finished")) if task_outcome else False
    task_on_time = _bool(task_outcome.get("finished_on_time")) if task_outcome else False
    missed_deadline = task_finished and not task_on_time
    dag_failed = _bool(dag_outcome.get("failed")) if dag_outcome else False
    dag_success = _bool(dag_outcome.get("successful")) if dag_outcome else False
    dag_on_time = _bool(dag_outcome.get("on_time_successful")) if dag_outcome else False

    if (
        delta is not None
        and delta > 0.0
        and (task_dropped or missed_deadline)
        and heuristic_margin is not None
        and selected_margin is not None
        and heuristic_margin > selected_margin
    ):
        if delta > strong_delta_threshold and (task_dropped or dag_failed) and (is_high_risk or is_critical):
            return "BAD_SCORE_DECISION_STRONG", "score slower, bad task/DAG outcome, high-risk/critical"
        return "BAD_SCORE_DECISION", "score slower and task outcome bad"

    if (
        task_on_time
        and (dag_on_time or dag_success)
        and (
            (delta is not None and delta <= good_delta_tolerance)
            or (selected_margin is not None and selected_margin >= 0.0)
        )
    ):
        return "GOOD_SCORE_DECISION", "score disagreement finished on-time with successful DAG context"

    return "AMBIGUOUS", "observational outcome is not decisive"


@dataclass
class LabelAgg:
    total: int = 0
    bad: int = 0
    bad_strong: int = 0
    good: int = 0
    ambiguous: int = 0
    high_risk: int = 0
    critical: int = 0
    candidate_scarce: int = 0
    task_dropped: int = 0
    task_on_time: int = 0
    dag_failed: int = 0
    dag_on_time: int = 0
    delta_sum: float = 0.0
    delta_count: int = 0

    def update(
        self,
        label: str,
        row: dict[str, Any],
        task_outcome: dict[str, Any] | None,
        dag_outcome: dict[str, Any] | None,
    ) -> None:
        self.total += 1
        if label == "BAD_SCORE_DECISION":
            self.bad += 1
        elif label == "BAD_SCORE_DECISION_STRONG":
            self.bad_strong += 1
        elif label == "GOOD_SCORE_DECISION":
            self.good += 1
        else:
            self.ambiguous += 1
        if _bool(row.get("is_high_risk_task")) or _bool(row.get("dag_is_high_risk")):
            self.high_risk += 1
        if _bool(row.get("is_critical_path_task")):
            self.critical += 1
        candidate_count = _safe_int(row.get("candidate_count"))
        if candidate_count is not None and candidate_count <= 2:
            self.candidate_scarce += 1
        if task_outcome:
            if _bool(task_outcome.get("dropped")):
                self.task_dropped += 1
            if _bool(task_outcome.get("finished_on_time")):
                self.task_on_time += 1
        if dag_outcome:
            if _bool(dag_outcome.get("failed")):
                self.dag_failed += 1
            if _bool(dag_outcome.get("on_time_successful")):
                self.dag_on_time += 1
        delta = _safe_float(row.get("delta_planned_finish"))
        if delta is not None:
            self.delta_sum += delta
            self.delta_count += 1

    def as_dict(self) -> dict[str, float]:
        denom = max(self.total, 1)
        return {
            "count": float(self.total),
            "bad_count": float(self.bad),
            "bad_strong_count": float(self.bad_strong),
            "bad_total_count": float(self.bad + self.bad_strong),
            "good_count": float(self.good),
            "ambiguous_count": float(self.ambiguous),
            "bad_rate": (self.bad + self.bad_strong) / denom,
            "good_rate": self.good / denom,
            "ambiguous_rate": self.ambiguous / denom,
            "high_risk_rate": self.high_risk / denom,
            "critical_rate": self.critical / denom,
            "candidate_scarce_rate": self.candidate_scarce / denom,
            "task_drop_rate": self.task_dropped / denom,
            "task_on_time_rate": self.task_on_time / denom,
            "dag_failure_rate": self.dag_failed / denom,
            "dag_on_time_rate": self.dag_on_time / denom,
            "mean_delta_planned_finish": self.delta_sum / max(self.delta_count, 1),
        }


def _iter_attribution_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in input_dir.glob("*_attribution.json") if "_full_" in path.name)


def load_outcome_maps(paths: list[Path]) -> tuple[dict[tuple[int, int, str], dict[str, Any]], dict[tuple[int, int, str], dict[str, Any]]]:
    task_outcomes: dict[tuple[int, int, str], dict[str, Any]] = {}
    dag_outcomes: dict[tuple[int, int, str], dict[str, Any]] = {}
    for path in paths:
        _, seed = _parse_mode_seed(path)
        print(f"[outcomes] {path.name}", flush=True)
        for row in iter_json_array(path, "task_outcomes"):
            task_outcomes[_task_key(row, seed)] = row
        for row in iter_json_array(path, "dag_outcomes"):
            dag_outcomes[_dag_key(row, seed)] = row
    return task_outcomes, dag_outcomes


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _add_bucket_rows(
    rows: list[dict[str, Any]],
    bucket_type: str,
    buckets: dict[str, LabelAgg],
) -> None:
    for bucket, agg in sorted(buckets.items(), key=lambda item: item[0]):
        row: dict[str, Any] = {"bucket_type": bucket_type, "bucket": bucket}
        row.update(agg.as_dict())
        rows.append(row)


def analyze(
    input_dir: Path,
    output_dir: Path,
    *,
    preview_limit: int,
    good_delta_tolerance: float,
    strong_delta_threshold: float,
) -> None:
    paths = list(_iter_attribution_files(input_dir))
    if not paths:
        raise FileNotFoundError(f"No full attribution files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    task_outcomes, dag_outcomes = load_outcome_maps(paths)

    summary = LabelAgg()
    by_seed: dict[str, LabelAgg] = defaultdict(LabelAgg)
    by_label: dict[str, LabelAgg] = defaultdict(LabelAgg)
    by_bucket: dict[str, dict[str, LabelAgg]] = {
        "risk": defaultdict(LabelAgg),
        "critical": defaultdict(LabelAgg),
        "candidate_count": defaultdict(LabelAgg),
        "task_type": defaultdict(LabelAgg),
        "task_slack": defaultdict(LabelAgg),
        "delta_planned_finish": defaultdict(LabelAgg),
    }
    preview_path = output_dir / "outcome_rerank_dataset_preview.jsonl"
    preview_written = 0

    with preview_path.open("w", encoding="utf-8") as preview:
        for path in paths:
            _, seed = _parse_mode_seed(path)
            seed_key = f"seed{seed}"
            print(f"[assignments] {path.name}", flush=True)
            for row in iter_json_array(path, "assignments"):
                if row.get("selection_mode") != "score" or not _bool(row.get("disagrees_with_heuristic")):
                    continue
                task_outcome = task_outcomes.get(_task_key(row, seed))
                dag_outcome = dag_outcomes.get(_dag_key(row, seed))
                label, reason = classify_disagreement(
                    row,
                    task_outcome,
                    dag_outcome,
                    good_delta_tolerance=good_delta_tolerance,
                    strong_delta_threshold=strong_delta_threshold,
                )
                summary.update(label, row, task_outcome, dag_outcome)
                by_seed[seed_key].update(label, row, task_outcome, dag_outcome)
                by_label[label].update(label, row, task_outcome, dag_outcome)

                risk_bucket = "high_risk" if _bool(row.get("is_high_risk_task")) or _bool(row.get("dag_is_high_risk")) else "non_high_risk"
                critical_bucket = "critical_path" if _bool(row.get("is_critical_path_task")) else "non_critical_path"
                candidate_bucket = _candidate_bucket(_safe_int(row.get("candidate_count")))
                task_type_bucket = str(row.get("task_type", "unknown"))
                slack_bucket = _slack_bucket(_safe_float(row.get("task_slack")))
                delta_bucket = _delta_bucket(_safe_float(row.get("delta_planned_finish")))
                for bucket_type, bucket in [
                    ("risk", risk_bucket),
                    ("critical", critical_bucket),
                    ("candidate_count", candidate_bucket),
                    ("task_type", task_type_bucket),
                    ("task_slack", slack_bucket),
                    ("delta_planned_finish", delta_bucket),
                ]:
                    by_bucket[bucket_type][bucket].update(label, row, task_outcome, dag_outcome)

                if preview_written < preview_limit and label != "AMBIGUOUS":
                    payload = {
                        "label": label,
                        "reason": reason,
                        "episode": row.get("episode"),
                        "seed": seed,
                        "task_id": row.get("task_id"),
                        "dag_id": row.get("dag_id"),
                        "task_type": row.get("task_type"),
                        "is_high_risk_task": row.get("is_high_risk_task"),
                        "is_critical_path_task": row.get("is_critical_path_task"),
                        "candidate_count": row.get("candidate_count"),
                        "score_uav": row.get("score_uav"),
                        "heuristic_uav": row.get("heuristic_uav"),
                        "delta_planned_finish": row.get("delta_planned_finish"),
                        "selected_deadline_margin": row.get("selected_deadline_margin"),
                        "heuristic_deadline_margin": row.get("heuristic_deadline_margin"),
                        "task_finished_on_time": None if task_outcome is None else task_outcome.get("finished_on_time"),
                        "task_dropped": None if task_outcome is None else task_outcome.get("dropped"),
                        "dag_successful": None if dag_outcome is None else dag_outcome.get("successful"),
                        "dag_failed": None if dag_outcome is None else dag_outcome.get("failed"),
                        "dag_on_time_successful": None if dag_outcome is None else dag_outcome.get("on_time_successful"),
                    }
                    preview.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    preview_written += 1

    summary_rows: list[dict[str, Any]] = []
    row = {"scope": "all"}
    row.update(summary.as_dict())
    summary_rows.append(row)
    for seed, agg in sorted(by_seed.items()):
        row = {"scope": seed}
        row.update(agg.as_dict())
        summary_rows.append(row)
    for label, agg in sorted(by_label.items()):
        row = {"scope": f"label:{label}"}
        row.update(agg.as_dict())
        summary_rows.append(row)

    summary_fields = ["scope"] + list(summary.as_dict().keys())
    _write_csv(output_dir / "outcome_rerank_candidate_summary.csv", summary_rows, summary_fields)

    bucket_rows: list[dict[str, Any]] = []
    for bucket_type, buckets in by_bucket.items():
        _add_bucket_rows(bucket_rows, bucket_type, buckets)
    bucket_fields = ["bucket_type", "bucket"] + list(summary.as_dict().keys())
    _write_csv(output_dir / "outcome_rerank_by_bucket.csv", bucket_rows, bucket_fields)

    report_path = output_dir / "outcome_rerank_report.md"
    report_path.write_text(_build_report(summary_rows, bucket_rows, paths, preview_written), encoding="utf-8")
    print(f"[done] output_dir={output_dir}", flush=True)


def _find_scope(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    for row in rows:
        if row.get("scope") == scope:
            return row
    return {}


def _build_report(
    summary_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    paths: list[Path],
    preview_written: int,
) -> str:
    overall = _find_scope(summary_rows, "all")
    total = int(float(overall.get("count", 0.0)))
    bad_total = int(float(overall.get("bad_total_count", 0.0)))
    good = int(float(overall.get("good_count", 0.0)))
    ambiguous = int(float(overall.get("ambiguous_count", 0.0)))
    bad_rate = float(overall.get("bad_rate", 0.0))
    ambiguous_rate = float(overall.get("ambiguous_rate", 0.0))

    seed_lines = []
    for row in summary_rows:
        scope = str(row.get("scope", ""))
        if scope.startswith("seed"):
            seed_lines.append(
                f"- {scope}: total={int(float(row['count']))}, bad_total={int(float(row['bad_total_count']))}, "
                f"good={int(float(row['good_count']))}, ambiguous={int(float(row['ambiguous_count']))}, "
                f"bad_rate={float(row['bad_rate']):.4f}"
            )

    risk_rows = [row for row in bucket_rows if row.get("bucket_type") == "risk"]
    critical_rows = [row for row in bucket_rows if row.get("bucket_type") == "critical"]
    candidate_rows = [row for row in bucket_rows if row.get("bucket_type") == "candidate_count"]

    def fmt_bucket(rows: list[dict[str, Any]]) -> str:
        lines = []
        for row in rows:
            lines.append(
                f"- {row['bucket_type']}={row['bucket']}: total={int(float(row['count']))}, "
                f"bad_total={int(float(row['bad_total_count']))}, bad_rate={float(row['bad_rate']):.4f}, "
                f"ambiguous_rate={float(row['ambiguous_rate']):.4f}"
            )
        return "\n".join(lines)

    if bad_total < 1000:
        recommendation = "高置信负样本偏少，暂不建议训练 outcome-aware rerank。"
    elif ambiguous_rate > 0.85:
        recommendation = "ambiguous 占比过高，暂不建议直接训练；应先收紧或重设标签规则。"
    else:
        recommendation = "高置信负样本具备初步训练价值；建议第一版只用 BAD 样本做小权重 fine-tune。"

    sources = "\n".join(f"- {path}" for path in paths)
    return f"""# Outcome-Aware Re-Ranking Dataset Diagnostics

## Inputs

{sources}

## Overall

- disagreement samples: {total}
- bad_total: {bad_total}
- good: {good}
- ambiguous: {ambiguous}
- bad_rate: {bad_rate:.6f}
- ambiguous_rate: {ambiguous_rate:.6f}
- preview_non_ambiguous_written: {preview_written}

## By Seed

{chr(10).join(seed_lines)}

## Key Buckets

### Risk

{fmt_bucket(risk_rows)}

### Critical

{fmt_bucket(critical_rows)}

### Candidate Count

{fmt_bucket(candidate_rows)}

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose outcome-aware reranking candidates from attribution files.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--preview_limit", type=int, default=2000)
    parser.add_argument("--good_delta_tolerance", type=float, default=0.1)
    parser.add_argument("--strong_delta_threshold", type=float, default=0.3)
    args = parser.parse_args()

    analyze(
        Path(args.input_dir),
        Path(args.output_dir),
        preview_limit=args.preview_limit,
        good_delta_tolerance=args.good_delta_tolerance,
        strong_delta_threshold=args.strong_delta_threshold,
    )


if __name__ == "__main__":
    main()
