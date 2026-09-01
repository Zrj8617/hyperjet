#!/usr/bin/env python3
"""Stream a frozen Stage-1 static corpus into the Tier-2 summary JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    groups: dict[str, dict] = defaultdict(
        lambda: {
            "minimums": [],
            "buckets": defaultdict(list),
            "bucket_saturated": Counter(),
            "bucket_candidates": Counter(),
            "hist": Counter(),
            "saturated": 0,
            "margin20": 0,
            "margin20_saturated": 0,
        }
    )
    source_hash = hashlib.sha256()
    source_records = 0

    with args.corpus.open("rb") as handle:
        for raw_line in handle:
            source_hash.update(raw_line)
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            source_records += 1
            sha = record["checkpoint_sha256"]
            slot = int(record["slot_index"])
            current_time = (slot + 1) * 5.0
            legal_efts = [
                float(eft)
                for eft, legal in zip(record["eft"], record["candidate_mask"])
                if legal
            ]
            if not legal_efts:
                raise ValueError(f"record {source_records} has no legal candidate")
            delays = [max(eft - current_time, 0.0) for eft in legal_efts]
            minimum = min(delays)
            saturated = all(delay >= 40.0 for delay in delays)
            margin20 = (
                len(legal_efts) >= 2
                and sorted(legal_efts)[1] - sorted(legal_efts)[0] >= 20.0
            )
            bucket = slot // 20
            if not 0 <= bucket < 10:
                raise ValueError(f"slot_index out of frozen 0..199 range: {slot}")

            group = groups[sha]
            group["minimums"].append(minimum)
            group["buckets"][bucket].append(minimum)
            group["bucket_saturated"][bucket] += int(saturated)
            group["bucket_candidates"][bucket] += len(legal_efts)
            group["hist"][str(len(legal_efts))] += 1
            group["saturated"] += int(saturated)
            group["margin20"] += int(margin20)
            group["margin20_saturated"] += int(margin20 and saturated)

    summaries = []
    for sha in sorted(groups):
        group = groups[sha]
        count = len(group["minimums"])
        buckets = []
        for bucket in range(10):
            values = group["buckets"][bucket]
            bucket_count = len(values)
            buckets.append(
                {
                    "slot_lo": bucket * 20,
                    "slot_hi": bucket * 20 + 19,
                    "p50": percentile(values, 0.50),
                    "p90": percentile(values, 0.90),
                    "saturation_rate": group["bucket_saturated"][bucket] / bucket_count,
                    "mean_legal_candidates": group["bucket_candidates"][bucket] / bucket_count,
                    "decision_count": bucket_count,
                }
            )
        summaries.append(
            {
                "checkpoint_sha256": sha,
                "decision_count": count,
                "min_legal_incremental_delay": distribution(group["minimums"]),
                "by_slot_bucket": buckets,
                "legal_candidate_count_hist": {
                    key: group["hist"][key] for key in ("2", "3", "4", "5")
                },
                "saturation_rate_overall": group["saturated"] / count,
                "saturation_rate_margin20": (
                    group["margin20_saturated"] / group["margin20"]
                    if group["margin20"]
                    else None
                ),
                "margin20_decision_count": group["margin20"],
            }
        )

    output = {
        "schema": "stage1_static_corpus_load_summary_v1",
        "source_path": str(args.corpus),
        "source_sha256": source_hash.hexdigest(),
        "source_record_count": source_records,
        "time_formula": "current_time=(slot_index+1)*5.0",
        "saturation_threshold_seconds": 40.0,
        "checkpoint_summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
