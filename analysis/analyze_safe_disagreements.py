from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CATEGORIES = [
    "NO_SCORE_DECISION",
    "AGREES_WITH_HEURISTIC",
    "RAW_UNSAFE_CLAMPED",
    "RAW_AGREEMENT_GUARDED",
    "FINAL_SAFE_BAD_STRONG",
    "FINAL_SAFE_BAD_WEAK",
    "FINAL_SAFE_NOT_BAD",
    "FINAL_GOOD_CANDIDATE_STRICT",
    "AMBIGUOUS",
]

REQUIRED_FIELDS = [
    "selected_uav",
    "heuristic_uav",
    "score_uav",
    "raw_score_uav",
    "selection_mode",
    "guard_reason",
    "disagrees_with_heuristic",
    "raw_disagrees_with_heuristic",
    "delta_planned_finish",
    "selected_deadline_margin",
    "heuristic_deadline_margin",
    "candidate_count",
    "is_high_risk_task",
    "is_critical_path_task",
]

CSV_FIELDS = [
    "source_file",
    "episode",
    "step",
    "task_id",
    "dag_id",
    "selected_uav",
    "heuristic_uav",
    "score_uav",
    "raw_score_uav",
    "selection_mode",
    "guard_reason",
    "category",
    "raw_disagrees_with_heuristic",
    "disagrees_with_heuristic",
    "delta_planned_finish",
    "selected_deadline_margin",
    "heuristic_deadline_margin",
    "candidate_count",
    "is_high_risk_task",
    "is_critical_path_task",
    "task_finished_on_time",
    "task_dropped",
    "dag_success",
    "dag_on_time_success",
    "dag_failed",
    "inferred_fields",
    "notes",
]


@dataclass
class ParsedRecord:
    source_file: str
    raw: dict[str, Any]
    fields: dict[str, Any]
    category: str
    inferred_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        fval = _safe_float(value)
        return None if fval is None else int(fval)


def _safe_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _ratio(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001 - report per-file parse errors without aborting directory scans.
        return None, str(exc)


def _iter_json_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.json"))
    return []


def _records_from_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ("records", "attribution_records", "assignment_records", "decisions"):
        value = episode.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    return records


def _extract_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []

    records: list[dict[str, Any]] = []
    for key in ("assignments", "attribution_records", "assignment_records", "decisions", "records"):
        value = data.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    episodes = data.get("episodes")
    if isinstance(episodes, list):
        for episode in episodes:
            if isinstance(episode, dict):
                records.extend(_records_from_episode(episode))
    return records


def _build_outcome_maps(data: Any) -> tuple[dict[tuple[Any, Any], dict[str, Any]], dict[tuple[Any, Any], dict[str, Any]]]:
    task_map: dict[tuple[Any, Any], dict[str, Any]] = {}
    dag_map: dict[tuple[Any, Any], dict[str, Any]] = {}
    if not isinstance(data, dict):
        return task_map, dag_map
    for task in data.get("task_outcomes", []) or []:
        if isinstance(task, dict):
            task_map[(task.get("episode"), task.get("task_id"))] = task
    for dag in data.get("dag_outcomes", []) or []:
        if isinstance(dag, dict):
            dag_map[(dag.get("episode"), dag.get("dag_id"))] = dag
    return task_map, dag_map


def _candidate_by_uav(record: dict[str, Any], uav_id: int | None) -> dict[str, Any] | None:
    if uav_id is None:
        return None
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if isinstance(candidate, dict) and _safe_int(candidate.get("uav_id")) == uav_id:
            return candidate
    return None


def _candidate_finish(record: dict[str, Any], uav_id: int | None) -> float | None:
    candidate = _candidate_by_uav(record, uav_id)
    if candidate is None:
        return None
    return _safe_float(candidate.get("planned_finish"))


def _normalize_record(
    record: dict[str, Any],
    source_file: str,
    task_map: dict[tuple[Any, Any], dict[str, Any]],
    dag_map: dict[tuple[Any, Any], dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    missing = [field_name for field_name in REQUIRED_FIELDS if field_name not in record]
    inferred: list[str] = []
    notes: list[str] = []

    fields: dict[str, Any] = {
        "source_file": source_file,
        "episode": record.get("episode"),
        "step": record.get("step"),
        "task_id": record.get("task_id"),
        "dag_id": record.get("dag_id"),
        "selected_uav": _safe_int(record.get("selected_uav")),
        "heuristic_uav": _safe_int(record.get("heuristic_uav")),
        "score_uav": _safe_int(record.get("score_uav")),
        "raw_score_uav": _safe_int(record.get("raw_score_uav")),
        "selection_mode": str(record.get("selection_mode") or "none"),
        "guard_reason": str(record.get("guard_reason") or "none"),
        "disagrees_with_heuristic": _safe_bool(record.get("disagrees_with_heuristic")),
        "raw_disagrees_with_heuristic": _safe_bool(record.get("raw_disagrees_with_heuristic")),
        "delta_planned_finish": _safe_float(record.get("delta_planned_finish")),
        "selected_deadline_margin": _safe_float(record.get("selected_deadline_margin")),
        "heuristic_deadline_margin": _safe_float(record.get("heuristic_deadline_margin")),
        "candidate_count": _safe_int(record.get("candidate_count", record.get("num_candidates"))),
        "is_high_risk_task": _safe_bool(record.get("is_high_risk_task")),
        "is_critical_path_task": _safe_bool(record.get("is_critical_path_task")),
    }

    if fields["raw_score_uav"] is None and fields["score_uav"] is not None:
        fields["raw_score_uav"] = fields["score_uav"]
        inferred.append("raw_score_uav")
    if fields["raw_disagrees_with_heuristic"] is None:
        raw_score = fields["raw_score_uav"]
        heuristic = fields["heuristic_uav"]
        if raw_score is not None and heuristic is not None:
            fields["raw_disagrees_with_heuristic"] = raw_score != heuristic
            inferred.append("raw_disagrees_with_heuristic")
    if fields["disagrees_with_heuristic"] is None:
        selected = fields["selected_uav"]
        heuristic = fields["heuristic_uav"]
        if selected is not None and heuristic is not None:
            fields["disagrees_with_heuristic"] = selected != heuristic
            inferred.append("disagrees_with_heuristic")

    raw_finish = _candidate_finish(record, fields["raw_score_uav"])
    heuristic_finish = _safe_float(record.get("heuristic_planned_finish"))
    if heuristic_finish is None:
        heuristic_finish = _candidate_finish(record, fields["heuristic_uav"])
    fields["raw_planned_finish"] = raw_finish
    fields["heuristic_planned_finish"] = heuristic_finish
    if raw_finish is None and fields["raw_score_uav"] is not None:
        notes.append("raw_finish_unknown")

    task_outcome = task_map.get((fields["episode"], fields["task_id"]), {})
    dag_outcome = dag_map.get((fields["episode"], fields["dag_id"]), {})
    fields["task_finished_on_time"] = _safe_bool(
        record.get("task_finished_on_time", task_outcome.get("finished_on_time"))
    )
    fields["task_dropped"] = _safe_bool(record.get("task_dropped", task_outcome.get("dropped")))
    fields["dag_success"] = _safe_bool(record.get("dag_success", dag_outcome.get("successful")))
    fields["dag_on_time_success"] = _safe_bool(
        record.get("dag_on_time_success", dag_outcome.get("on_time_successful"))
    )
    fields["dag_failed"] = _safe_bool(record.get("dag_failed", dag_outcome.get("failed")))
    fields["critical_path_on_time"] = _safe_bool(record.get("critical_path_on_time"))
    fields["critical_path_finished"] = _safe_bool(record.get("critical_path_finished"))

    return fields, inferred, missing, notes


def _has_bad_outcome(fields: dict[str, Any]) -> bool:
    bad_values = [
        fields.get("task_dropped") is True,
        fields.get("task_finished_on_time") is False,
        fields.get("dag_failed") is True,
        fields.get("dag_on_time_success") is False,
    ]
    return any(bad_values)


def _has_good_outcome(fields: dict[str, Any]) -> bool:
    return (
        fields.get("task_finished_on_time") is True
        and fields.get("dag_on_time_success") is True
        and fields.get("dag_failed") is not True
        and fields.get("task_dropped") is not True
    )


def _classify(fields: dict[str, Any], args: argparse.Namespace) -> str:
    selected = fields.get("selected_uav")
    heuristic = fields.get("heuristic_uav")
    score = fields.get("score_uav")
    raw_score = fields.get("raw_score_uav")
    selection_mode = fields.get("selection_mode") or "none"
    guard_reason = fields.get("guard_reason") or "none"
    raw_disagrees = fields.get("raw_disagrees_with_heuristic")
    final_disagrees = (
        (selected is not None and heuristic is not None and selected != heuristic)
        or (score is not None and heuristic is not None and score != heuristic)
        or fields.get("disagrees_with_heuristic") is True
    )

    score_related = selection_mode in {"score", "guard_fallback"} or score is not None or raw_score is not None
    if not score_related:
        return "NO_SCORE_DECISION"

    if (
        raw_disagrees is False
        and final_disagrees is False
    ) or (selected is not None and heuristic is not None and selected == heuristic and raw_score == heuristic):
        return "AGREES_WITH_HEURISTIC"

    if (
        raw_disagrees is True
        and guard_reason == "runtime_bounded_guard_clamp"
        and raw_score is not None
        and selected is not None
        and raw_score != selected
    ):
        raw_finish = fields.get("raw_planned_finish")
        heuristic_finish = fields.get("heuristic_planned_finish")
        raw_finish_unknown = raw_finish is None or heuristic_finish is None
        raw_is_unsafe = (
            raw_finish is not None
            and heuristic_finish is not None
            and raw_finish > heuristic_finish + args.delta_tolerance
        )
        if raw_finish_unknown or raw_is_unsafe:
            return "RAW_UNSAFE_CLAMPED"

    if raw_disagrees is True and guard_reason == "agreement_only" and selected == heuristic:
        return "RAW_AGREEMENT_GUARDED"

    if not final_disagrees:
        return "AMBIGUOUS"

    delta = fields.get("delta_planned_finish")
    selected_margin = fields.get("selected_deadline_margin")
    heuristic_margin = fields.get("heuristic_deadline_margin")
    margin_worse = (
        selected_margin is not None
        and heuristic_margin is not None
        and heuristic_margin > selected_margin + args.min_margin_gap
    )
    bad_outcome = _has_bad_outcome(fields)
    high_priority = fields.get("is_high_risk_task") is True or fields.get("is_critical_path_task") is True

    if (
        delta is not None
        and delta > args.strong_bad_delta
        and bad_outcome
        and margin_worse
        and high_priority
    ):
        return "FINAL_SAFE_BAD_STRONG"

    if (
        delta is not None
        and delta > args.bad_delta_threshold
        and (bad_outcome or margin_worse)
    ):
        return "FINAL_SAFE_BAD_WEAK"

    critical_ok = fields.get("critical_path_on_time") is not False and fields.get("critical_path_finished") is not False
    if (
        delta is not None
        and delta <= args.good_delta_tolerance
        and selected_margin is not None
        and selected_margin >= 0.0
        and _has_good_outcome(fields)
        and critical_ok
    ):
        return "FINAL_GOOD_CANDIDATE_STRICT"

    if not bad_outcome:
        return "FINAL_SAFE_NOT_BAD"

    return "AMBIGUOUS"


def _delta_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value <= -0.1:
        return "<= -0.1"
    if value <= 0.0:
        return "(-0.1, 0]"
    if value <= 0.1:
        return "(0, 0.1]"
    if value <= 0.3:
        return "(0.1, 0.3]"
    if value <= 0.5:
        return "(0.3, 0.5]"
    return "> 0.5"


def _candidate_bucket(value: int | None) -> str:
    if value is None:
        return "missing"
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    return "4+"


def _bool_bucket(value: bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "missing"


def _guard_bucket(value: str | None) -> str:
    if not value or value == "none":
        return "none"
    if value in {"runtime_bounded_guard_clamp", "runtime_bounded_guard", "agreement_only"}:
        return value
    return "other"


def _selection_bucket(value: str | None) -> str:
    if value in {"score", "fallback", "guard_fallback"}:
        return value
    return "other"


def _bucket_values(record: ParsedRecord) -> dict[str, str]:
    fields = record.fields
    return {
        "delta_planned_finish": _delta_bucket(fields.get("delta_planned_finish")),
        "candidate_count": _candidate_bucket(fields.get("candidate_count")),
        "high_risk": _bool_bucket(fields.get("is_high_risk_task")),
        "critical_path": _bool_bucket(fields.get("is_critical_path_task")),
        "guard_reason": _guard_bucket(fields.get("guard_reason")),
        "selection_mode": _selection_bucket(fields.get("selection_mode")),
    }


def _final_disagrees(fields: dict[str, Any]) -> bool:
    selected = fields.get("selected_uav")
    score = fields.get("score_uav")
    heuristic = fields.get("heuristic_uav")
    return bool(
        fields.get("disagrees_with_heuristic") is True
        or (selected is not None and heuristic is not None and selected != heuristic)
        or (score is not None and heuristic is not None and score != heuristic)
    )


def _aggregate_records(records: list[ParsedRecord]) -> dict[str, Any]:
    values_delta: list[float] = []
    selected_margins: list[float] = []
    heuristic_margins: list[float] = []
    task_on_time = 0
    task_on_time_known = 0
    task_dropped = 0
    task_dropped_known = 0
    dag_success = 0
    dag_success_known = 0
    dag_on_time = 0
    dag_on_time_known = 0
    high_risk = 0
    high_risk_known = 0
    critical = 0
    critical_known = 0
    for record in records:
        fields = record.fields
        if fields.get("delta_planned_finish") is not None:
            values_delta.append(fields["delta_planned_finish"])
        if fields.get("selected_deadline_margin") is not None:
            selected_margins.append(fields["selected_deadline_margin"])
        if fields.get("heuristic_deadline_margin") is not None:
            heuristic_margins.append(fields["heuristic_deadline_margin"])
        if fields.get("task_finished_on_time") is not None:
            task_on_time_known += 1
            task_on_time += int(fields["task_finished_on_time"] is True)
        if fields.get("task_dropped") is not None:
            task_dropped_known += 1
            task_dropped += int(fields["task_dropped"] is True)
        if fields.get("dag_success") is not None:
            dag_success_known += 1
            dag_success += int(fields["dag_success"] is True)
        if fields.get("dag_on_time_success") is not None:
            dag_on_time_known += 1
            dag_on_time += int(fields["dag_on_time_success"] is True)
        if fields.get("is_high_risk_task") is not None:
            high_risk_known += 1
            high_risk += int(fields["is_high_risk_task"] is True)
        if fields.get("is_critical_path_task") is not None:
            critical_known += 1
            critical += int(fields["is_critical_path_task"] is True)
    return {
        "count": len(records),
        "avg_delta_planned_finish": _mean(values_delta),
        "median_delta_planned_finish": _median(values_delta),
        "avg_selected_margin": _mean(selected_margins),
        "avg_heuristic_margin": _mean(heuristic_margins),
        "task_on_time_rate": _ratio(task_on_time, task_on_time_known),
        "task_drop_rate": _ratio(task_dropped, task_dropped_known),
        "dag_success_rate": _ratio(dag_success, dag_success_known),
        "dag_on_time_success_rate": _ratio(dag_on_time, dag_on_time_known),
        "high_risk_ratio": _ratio(high_risk, high_risk_known),
        "critical_path_ratio": _ratio(critical, critical_known),
    }


def _bucket_summary(records: list[ParsedRecord]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in ("delta_planned_finish", "candidate_count", "high_risk", "critical_path", "guard_reason", "selection_mode"):
        groups: dict[str, list[ParsedRecord]] = defaultdict(list)
        for record in records:
            groups[_bucket_values(record)[dimension]].append(record)
        result[dimension] = {}
        for bucket, bucket_records in sorted(groups.items()):
            total = len(bucket_records)
            result[dimension][bucket] = {
                "total": total,
                "raw_disagree_ratio": _ratio(
                    sum(1 for item in bucket_records if item.fields.get("raw_disagrees_with_heuristic") is True),
                    total,
                ),
                "final_disagree_ratio": _ratio(sum(1 for item in bucket_records if _final_disagrees(item.fields)), total),
                "clamp_ratio": _ratio(
                    sum(1 for item in bucket_records if item.fields.get("guard_reason") == "runtime_bounded_guard_clamp"),
                    total,
                ),
                "bad_strong_ratio": _ratio(sum(1 for item in bucket_records if item.category == "FINAL_SAFE_BAD_STRONG"), total),
                "bad_weak_ratio": _ratio(sum(1 for item in bucket_records if item.category == "FINAL_SAFE_BAD_WEAK"), total),
                "safe_not_bad_ratio": _ratio(sum(1 for item in bucket_records if item.category == "FINAL_SAFE_NOT_BAD"), total),
                "good_candidate_ratio": _ratio(
                    sum(1 for item in bucket_records if item.category == "FINAL_GOOD_CANDIDATE_STRICT"),
                    total,
                ),
                "dag_on_time_success_rate": _aggregate_records(bucket_records)["dag_on_time_success_rate"],
                "task_on_time_rate": _aggregate_records(bucket_records)["task_on_time_rate"],
            }
    return result


def _decision_counts(records: list[ParsedRecord]) -> dict[str, int]:
    return {
        "raw_disagrees_count": sum(1 for item in records if item.fields.get("raw_disagrees_with_heuristic") is True),
        "final_disagrees_count": sum(1 for item in records if _final_disagrees(item.fields)),
        "guard_clamp_count": sum(1 for item in records if item.fields.get("guard_reason") == "runtime_bounded_guard_clamp"),
        "agreement_guard_count": sum(1 for item in records if item.fields.get("guard_reason") == "agreement_only"),
        "score_selected_count": sum(1 for item in records if item.fields.get("selection_mode") == "score"),
        "guard_fallback_count": sum(1 for item in records if item.fields.get("selection_mode") == "guard_fallback"),
        "fallback_selected_count": sum(1 for item in records if item.fields.get("selection_mode") == "fallback"),
    }


def _top_buckets(bucket_analysis: dict[str, Any], ratio_key: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension, buckets in bucket_analysis.items():
        for bucket, stats in buckets.items():
            rows.append(
                {
                    "dimension": dimension,
                    "bucket": bucket,
                    "ratio": stats.get(ratio_key, 0.0),
                    "total": stats.get("total", 0),
                }
            )
    rows.sort(key=lambda item: (item["ratio"], item["total"]), reverse=True)
    return rows[:limit]


def _write_csv(path: Path, records: list[ParsedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in records:
            fields = item.fields
            writer.writerow(
                {
                    "source_file": fields.get("source_file"),
                    "episode": fields.get("episode"),
                    "step": fields.get("step"),
                    "task_id": fields.get("task_id"),
                    "dag_id": fields.get("dag_id"),
                    "selected_uav": fields.get("selected_uav"),
                    "heuristic_uav": fields.get("heuristic_uav"),
                    "score_uav": fields.get("score_uav"),
                    "raw_score_uav": fields.get("raw_score_uav"),
                    "selection_mode": fields.get("selection_mode"),
                    "guard_reason": fields.get("guard_reason"),
                    "category": item.category,
                    "raw_disagrees_with_heuristic": fields.get("raw_disagrees_with_heuristic"),
                    "disagrees_with_heuristic": fields.get("disagrees_with_heuristic"),
                    "delta_planned_finish": fields.get("delta_planned_finish"),
                    "selected_deadline_margin": fields.get("selected_deadline_margin"),
                    "heuristic_deadline_margin": fields.get("heuristic_deadline_margin"),
                    "candidate_count": fields.get("candidate_count"),
                    "is_high_risk_task": fields.get("is_high_risk_task"),
                    "is_critical_path_task": fields.get("is_critical_path_task"),
                    "task_finished_on_time": fields.get("task_finished_on_time"),
                    "task_dropped": fields.get("task_dropped"),
                    "dag_success": fields.get("dag_success"),
                    "dag_on_time_success": fields.get("dag_on_time_success"),
                    "dag_failed": fields.get("dag_failed"),
                    "inferred_fields": ";".join(item.inferred_fields),
                    "notes": ";".join(item.notes),
                }
            )


def _print_report(summary: dict[str, Any], print_top_examples: int) -> None:
    total = summary["metadata"]["parsed_records"]
    decisions = summary["decision_counts"]
    counts = summary["overall_counts"]
    print("Safe disagreement attribution")
    print(f"records: {total}")
    print(f"raw disagreement ratio: {_ratio(decisions['raw_disagrees_count'], total):.6f}")
    print(f"final disagreement ratio: {_ratio(decisions['final_disagrees_count'], total):.6f}")
    print(f"bounded guard clamp ratio: {_ratio(decisions['guard_clamp_count'], total):.6f}")
    print(f"FINAL_SAFE_BAD_STRONG ratio: {counts.get('FINAL_SAFE_BAD_STRONG', {}).get('ratio', 0.0):.6f}")
    print(f"FINAL_SAFE_BAD_WEAK ratio: {counts.get('FINAL_SAFE_BAD_WEAK', {}).get('ratio', 0.0):.6f}")
    print(f"FINAL_SAFE_NOT_BAD ratio: {counts.get('FINAL_SAFE_NOT_BAD', {}).get('ratio', 0.0):.6f}")
    print(f"FINAL_GOOD_CANDIDATE_STRICT ratio: {counts.get('FINAL_GOOD_CANDIDATE_STRICT', {}).get('ratio', 0.0):.6f}")

    bad_top = _top_buckets(summary["bucket_analysis"], "bad_strong_ratio", print_top_examples)
    if not any(item["ratio"] > 0 for item in bad_top):
        bad_top = _top_buckets(summary["bucket_analysis"], "bad_weak_ratio", print_top_examples)
    good_top = _top_buckets(summary["bucket_analysis"], "good_candidate_ratio", print_top_examples)
    print("highest bad buckets:")
    for item in bad_top:
        print(f"  {item['dimension']}={item['bucket']} ratio={item['ratio']:.6f} total={item['total']}")
    print("highest good-candidate buckets:")
    for item in good_top:
        print(f"  {item['dimension']}={item['bucket']} ratio={item['ratio']:.6f} total={item['total']}")

    good_ratio = counts.get("FINAL_GOOD_CANDIDATE_STRICT", {}).get("ratio", 0.0)
    bad_ratio = (
        counts.get("FINAL_SAFE_BAD_STRONG", {}).get("ratio", 0.0)
        + counts.get("FINAL_SAFE_BAD_WEAK", {}).get("ratio", 0.0)
    )
    ambiguous_ratio = counts.get("AMBIGUOUS", {}).get("ratio", 0.0)
    print("recommendation:")
    if good_ratio < 0.01:
        print("  GOOD_CANDIDATE is rare; do not start positive outcome fine-tune yet.")
    if bad_ratio > 0.05:
        print("  BAD disagreements are common enough for bad-suppression fine-tune.")
    if ambiguous_ratio > 0.5:
        print("  AMBIGUOUS dominates; observational attribution may need counterfactual rollout.")
    print("  Treat GOOD_CANDIDATE as candidate positives only, not oracle positives.")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    files = _iter_json_files(input_path)
    parsed_records: list[ParsedRecord] = []
    file_status: list[dict[str, Any]] = []
    missing_counter: Counter[str] = Counter()
    total_records = 0

    for path in files:
        data, error = _read_json(path)
        if error is not None:
            file_status.append({"path": str(path), "parsed": False, "error": error, "records": 0})
            continue
        task_map, dag_map = _build_outcome_maps(data)
        records = _extract_records(data)
        total_records += len(records)
        file_status.append({"path": str(path), "parsed": True, "error": None, "records": len(records)})
        for record in records:
            fields, inferred, missing, notes = _normalize_record(record, str(path), task_map, dag_map)
            missing_counter.update(missing)
            category = _classify(fields, args)
            parsed_records.append(
                ParsedRecord(
                    source_file=str(path),
                    raw=record,
                    fields=fields,
                    category=category,
                    inferred_fields=inferred,
                    missing_fields=missing,
                    notes=notes,
                )
            )

    category_counter = Counter(item.category for item in parsed_records)
    overall_counts = {
        category: {"count": category_counter.get(category, 0), "ratio": _ratio(category_counter.get(category, 0), len(parsed_records))}
        for category in CATEGORIES
    }
    by_category = {
        category: _aggregate_records([item for item in parsed_records if item.category == category])
        for category in CATEGORIES
    }
    bucket_analysis = _bucket_summary(parsed_records)
    summary = {
        "metadata": {
            "input_path": str(input_path),
            "num_files": len(files),
            "total_records": total_records,
            "parsed_records": len(parsed_records),
            "skipped_records": max(total_records - len(parsed_records), 0),
            "missing_fields_summary": dict(sorted(missing_counter.items())),
            "delta_tolerance": args.delta_tolerance,
            "good_delta_tolerance": args.good_delta_tolerance,
            "bad_delta_threshold": args.bad_delta_threshold,
            "strong_bad_delta": args.strong_bad_delta,
            "min_margin_gap": args.min_margin_gap,
            "file_status": file_status,
        },
        "overall_counts": overall_counts,
        "decision_counts": _decision_counts(parsed_records),
        "outcome_by_category": by_category,
        "bucket_analysis": bucket_analysis,
    }
    if args.csv_output:
        _write_csv(Path(args.csv_output), parsed_records)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze safe HGNN-vs-heuristic disagreement attribution.")
    parser.add_argument("--input", required=True, help="Input attribution JSON file or directory.")
    parser.add_argument("--output", required=True, help="Output summary JSON path.")
    parser.add_argument("--csv_output", default="", help="Optional per-record classification CSV path.")
    parser.add_argument("--delta_tolerance", type=float, default=0.1)
    parser.add_argument("--good_delta_tolerance", type=float, default=0.0)
    parser.add_argument("--bad_delta_threshold", type=float, default=0.1)
    parser.add_argument("--strong_bad_delta", type=float, default=0.3)
    parser.add_argument("--min_margin_gap", type=float, default=0.1)
    parser.add_argument("--print_top_examples", type=int, default=20)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = analyze(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_report(summary, args.print_top_examples)
    print(f"summary={output_path}")
    if args.csv_output:
        print(f"csv={args.csv_output}")


if __name__ == "__main__":
    main()
