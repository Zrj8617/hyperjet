from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


SATURATION_REF = 40.0
EPSILON = 1e-9


def _percentiles(values: list[float], points: tuple[int, ...]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("percentiles require at least one value")
    return {f"p{point}": float(np.percentile(array, point)) for point in points}


def _accuracy(matches: list[bool]) -> float:
    if not matches:
        raise ValueError("accuracy requires at least one record")
    return float(np.mean(np.asarray(matches, dtype=np.float64)))


def _transform_specs() -> list[tuple[str, float | None, Callable[..., np.ndarray]]]:
    specs: list[tuple[str, float | None, Callable[..., np.ndarray]]] = [
        ("clip", 40.0, lambda x, ref, **_: np.clip(x / ref, 0.0, 1.0))
    ]
    specs.extend(
        ("clip", float(ref), lambda x, ref, **_: np.clip(x / ref, 0.0, 1.0))
        for ref in (80, 160, 200, 320)
    )
    specs.extend(
        ("x/(x+ref)", float(ref), lambda x, ref, **_: x / (x + ref))
        for ref in (20, 30, 50, 80, 120, 160, 200, 300)
    )
    specs.extend(
        ("log1p", float(ref), lambda x, ref, **_: np.log1p(x / ref))
        for ref in (5, 10, 20, 50)
    )
    specs.append(
        (
            "relative_upper_bound",
            None,
            lambda x, ref, d1, dmax: (x - d1) / (dmax - d1 + EPSILON),
        )
    )
    return specs


def _evaluate_transform(
    rows: list[dict[str, Any]],
    *,
    transform: str,
    ref: float | None,
    function: Callable[..., np.ndarray],
    delta_star: float,
    delta_scale: float = 1.0,
    phi_at_x_hi: float | None = None,
) -> dict[str, Any]:
    margin20 = [row for row in rows if row["margin20"]]
    deltas: list[float] = []
    saturated_deltas: list[float] = []
    unsaturated_deltas: list[float] = []
    preserved: list[bool] = []
    for row in margin20:
        pair = np.asarray([row["d1"], row["d2"]], dtype=np.float64)
        encoded = function(pair, ref=ref, d1=row["d1"], dmax=row["dmax"])
        delta = float(encoded[1] - encoded[0]) / float(delta_scale)
        deltas.append(delta)
        preserved.append(bool(delta > 0.0))
        (saturated_deltas if row["saturated"] else unsaturated_deltas).append(delta)
    if not saturated_deltas or not unsaturated_deltas:
        raise ValueError("both saturated and unsaturated margin20 subsets are required")
    return {
        "transform": transform,
        "ref": ref,
        "phi_at_x_hi": phi_at_x_hi,
        "argmin_preserved_margin20": _accuracy(preserved),
        "delta_percentiles_unsat_m20": _percentiles(unsaturated_deltas, (10, 50, 90)),
        "delta_percentiles_sat_m20": _percentiles(saturated_deltas, (10, 50, 90)),
        "fraction_sat_m20_meeting_delta_star": float(
            np.mean(np.asarray(saturated_deltas, dtype=np.float64) >= delta_star)
        ),
        "delta_percentiles_all_m20": _percentiles(deltas, (10, 50, 90)),
        "selection_candidate": transform not in ("relative_upper_bound",) and not (
            transform == "clip" and ref == SATURATION_REF
        ),
    }


def _candidate_key(entry: dict[str, Any]) -> str:
    return f"{entry['transform']}@{entry['ref']:g}"


def _rank_candidates(candidates: list[dict[str, Any]]) -> list[str]:
    remaining = sorted(candidates, key=lambda item: item["ranking_score_min_p10"], reverse=True)
    ranking: list[str] = []
    while remaining:
        best = float(remaining[0]["ranking_score_min_p10"])
        tied = [
            item
            for item in remaining
            if best == 0.0
            or (best - float(item["ranking_score_min_p10"])) / best < 0.05
        ]
        tied.sort(
            key=lambda item: (
                not bool(item["natural_unit_range"]),
                float(item["ref"]),
                str(item["transform"]),
            )
        )
        ranking.extend(_candidate_key(item) for item in tied)
        tied_keys = {_candidate_key(item) for item in tied}
        remaining = [item for item in remaining if _candidate_key(item) not in tied_keys]
    return ranking


def _build_range_normalized_result(
    *,
    args: argparse.Namespace,
    grouped: dict[str, list[dict[str, Any]]],
    base_result: dict[str, Any],
) -> dict[str, Any]:
    if args.baseline_result is None:
        raise ValueError("--baseline-result is required with --range-normalized")
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    if baseline.get("source_sha256") != base_result["source_sha256"]:
        raise ValueError("baseline result source SHA-256 mismatch")
    for checkpoint, value in base_result["delta_star"].items():
        if not math.isclose(
            float(value), float(baseline["delta_star"][checkpoint]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"delta_star cross-check failed for {checkpoint}")

    x_hi_variants = (150.0, 400.0)
    by_x_hi: dict[str, Any] = {}
    rankings: dict[str, list[str]] = {}
    qualifiers_by_x_hi: dict[str, list[dict[str, Any]]] = {}
    for x_hi in x_hi_variants:
        x_key = str(x_hi)
        checkpoint_entries: dict[str, list[dict[str, Any]]] = {}
        for checkpoint, rows in sorted(grouped.items()):
            evaluated = []
            for transform, ref, function in _transform_specs():
                if transform == "relative_upper_bound":
                    continue
                phi_at_x_hi = float(
                    function(
                        np.asarray([x_hi], dtype=np.float64),
                        ref=ref,
                        d1=0.0,
                        dmax=x_hi,
                    )[0]
                )
                entry = _evaluate_transform(
                    rows,
                    transform=transform,
                    ref=ref,
                    function=function,
                    delta_star=float(base_result["delta_star"][checkpoint]),
                    delta_scale=phi_at_x_hi,
                    phi_at_x_hi=phi_at_x_hi,
                )
                entry["delta_norm_percentiles_unsat_m20"] = entry.pop(
                    "delta_percentiles_unsat_m20"
                )
                entry["delta_norm_percentiles_sat_m20"] = entry.pop(
                    "delta_percentiles_sat_m20"
                )
                entry["delta_norm_percentiles_all_m20"] = entry.pop(
                    "delta_percentiles_all_m20"
                )
                evaluated.append(entry)
            checkpoint_entries[checkpoint] = evaluated
        by_x_hi[x_key] = checkpoint_entries

        candidate_keys = [
            (entry["transform"], entry["ref"])
            for entry in next(iter(checkpoint_entries.values()))
            if entry["selection_candidate"]
        ]
        qualifiers: list[dict[str, Any]] = []
        for transform, ref in candidate_keys:
            matches = {
                checkpoint: next(
                    entry
                    for entry in entries
                    if entry["transform"] == transform and entry["ref"] == ref
                )
                for checkpoint, entries in checkpoint_entries.items()
            }
            if all(
                entry["argmin_preserved_margin20"] == 1.0
                and entry["fraction_sat_m20_meeting_delta_star"] >= 0.90
                for entry in matches.values()
            ):
                p10_by_checkpoint = {
                    checkpoint: entry["delta_norm_percentiles_sat_m20"]["p10"]
                    for checkpoint, entry in matches.items()
                }
                qualifiers.append(
                    {
                        "transform": transform,
                        "ref": ref,
                        "natural_unit_range": transform in ("clip", "x/(x+ref)"),
                        "fraction_by_checkpoint": {
                            checkpoint: entry["fraction_sat_m20_meeting_delta_star"]
                            for checkpoint, entry in matches.items()
                        },
                        "normalized_sat_m20_p10_by_checkpoint": p10_by_checkpoint,
                        "ranking_score_min_p10": min(p10_by_checkpoint.values()),
                    }
                )
        qualifiers_by_x_hi[x_key] = qualifiers
        rankings[x_key] = _rank_candidates(qualifiers)

    ranking_stable = rankings[str(x_hi_variants[0])] == rankings[str(x_hi_variants[1])]
    no_qualifiers = any(not qualifiers_by_x_hi[str(x_hi)] for x_hi in x_hi_variants)
    selected = None
    if ranking_stable and not no_qualifiers:
        selected_key = rankings[str(x_hi_variants[0])][0]
        selected = next(
            item
            for item in qualifiers_by_x_hi[str(x_hi_variants[0])]
            if _candidate_key(item) == selected_key
        )
        selected = {
            "transform": selected["transform"],
            "ref": selected["ref"],
            "shared_features": [
                "incremental_delay_pair_5",
                "queue_waiting_time_pair_2",
                "available_delta_dynamic_4",
            ],
        }
    return {
        "schema": "normalization_transform_selection_range_normalized_v1",
        "source_path": base_result["source_path"],
        "source_sha256": base_result["source_sha256"],
        "source_record_count": base_result["source_record_count"],
        "x_hi_variants": list(x_hi_variants),
        "delta_star": base_result["delta_star"],
        "delta_star_cross_check_pass": True,
        "by_x_hi": by_x_hi,
        "ranking_score_definition": "minimum normalized saturated-margin20 p10 across checkpoints",
        "rankings_by_x_hi": rankings,
        "ranking_stable_across_x_hi": ranking_stable,
        "qualifying_candidates": qualifiers_by_x_hi,
        "selected_candidate": selected,
        "selection_status": (
            "none_qualified" if no_qualifiers else "selected" if ranking_stable else "inconclusive"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline normalization transform selection")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--range-normalized", action="store_true")
    parser.add_argument("--baseline-result", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"create-only output already exists: {args.output}")

    digest = hashlib.sha256()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_record_count = 0
    with args.source.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            source_record_count += 1
            mask = np.asarray(record["candidate_mask"], dtype=bool)
            legal_indices = np.flatnonzero(mask)
            if legal_indices.size < 2:
                raise ValueError(f"record {source_record_count} has fewer than two legal candidates")
            eft = np.asarray(record["eft"], dtype=np.float64)
            current_time = (int(record["slot_index"]) + 1) * 5.0
            legal_delays = np.maximum(eft[legal_indices] - current_time, 0.0)
            ordered = np.sort(legal_delays)
            d1, d2 = float(ordered[0]), float(ordered[1])
            gap = d2 - d1
            grouped[str(record["checkpoint_sha256"])].append(
                {
                    "d1": d1,
                    "d2": d2,
                    "dmax": float(ordered[-1]),
                    "gap": gap,
                    "legal_k": int(legal_indices.size),
                    "margin20": bool(gap >= 20.0),
                    "saturated": bool(np.all(legal_delays >= SATURATION_REF)),
                    "correct": int(record["deterministic_actor_uav_id"])
                    == int(record["greedy_eft_uav_id"]),
                }
            )

    part_a: dict[str, Any] = {}
    part_a_pass = True
    for checkpoint, rows in sorted(grouped.items()):
        margin20 = [row for row in rows if row["margin20"]]
        saturated = [row for row in margin20 if row["saturated"]]
        unsaturated = [row for row in margin20 if not row["saturated"]]
        by_k = {
            str(k): _accuracy([row["correct"] for row in margin20 if row["legal_k"] == k])
            for k in (2, 3, 4, 5)
            if any(row["legal_k"] == k for row in margin20)
        }
        accuracy_saturated = _accuracy([row["correct"] for row in saturated])
        accuracy_unsaturated = _accuracy([row["correct"] for row in unsaturated])
        checkpoint_pass = (
            0.95 <= accuracy_unsaturated <= 0.99
            and 0.53 <= accuracy_saturated <= 0.61
        )
        part_a_pass = part_a_pass and checkpoint_pass
        part_a[checkpoint] = {
            "margin20_count": len(margin20),
            "saturated_fraction": len(saturated) / len(margin20),
            "accuracy_overall": _accuracy([row["correct"] for row in margin20]),
            "accuracy_saturated": accuracy_saturated,
            "accuracy_unsaturated": accuracy_unsaturated,
            "accuracy_by_legal_k": by_k,
            "expected_range_pass": checkpoint_pass,
        }

    result: dict[str, Any] = {
        "schema": "normalization_transform_selection_v1",
        "source_path": str(args.source),
        "source_sha256": digest.hexdigest(),
        "source_record_count": source_record_count,
        "saturation_ref": SATURATION_REF,
        "part_a": part_a,
        "part_a_pass": part_a_pass,
    }
    if not part_a_pass:
        result.update(
            {
                "selection_status": "stopped_part_a_outside_preregistered_ranges",
                "delta_star": {},
                "part_b": {},
                "part_c": {},
            }
        )
    else:
        delta_star: dict[str, float] = {}
        part_b: dict[str, list[dict[str, Any]]] = {}
        part_c: dict[str, Any] = {}
        for checkpoint, rows in sorted(grouped.items()):
            baseline_deltas = []
            for row in rows:
                if row["margin20"] and not row["saturated"]:
                    encoded = np.clip(
                        np.asarray([row["d1"], row["d2"]], dtype=np.float64)
                        / SATURATION_REF,
                        0.0,
                        1.0,
                    )
                    baseline_deltas.append(float(encoded[1] - encoded[0]))
            checkpoint_delta_star = _percentiles(baseline_deltas, (10,))["p10"]
            delta_star[checkpoint] = checkpoint_delta_star
            part_b[checkpoint] = [
                _evaluate_transform(
                    rows,
                    transform=transform,
                    ref=ref,
                    function=function,
                    delta_star=checkpoint_delta_star,
                )
                for transform, ref, function in _transform_specs()
            ]
            part_c[checkpoint] = {
                "delay_rank1_percentiles": _percentiles(
                    [row["d1"] for row in rows], (10, 25, 50, 75, 90, 99, 100)
                ),
                "delay_rank2_percentiles": _percentiles(
                    [row["d2"] for row in rows], (10, 50, 90)
                ),
                "gap_percentiles": _percentiles(
                    [row["gap"] for row in rows], (10, 25, 50, 75, 90, 99)
                ),
                "legal_candidate_count_hist": {
                    str(k): count
                    for k, count in sorted(Counter(row["legal_k"] for row in rows).items())
                },
            }
            part_c[checkpoint]["delay_rank1_percentiles"]["max"] = part_c[checkpoint][
                "delay_rank1_percentiles"
            ].pop("p100")

        candidate_keys = [
            (entry["transform"], entry["ref"])
            for entry in next(iter(part_b.values()))
            if entry["selection_candidate"]
        ]
        qualifiers = []
        for transform, ref in candidate_keys:
            fractions = {
                checkpoint: next(
                    entry["fraction_sat_m20_meeting_delta_star"]
                    for entry in entries
                    if entry["transform"] == transform and entry["ref"] == ref
                )
                for checkpoint, entries in part_b.items()
            }
            if all(value >= 0.90 for value in fractions.values()):
                qualifiers.append(
                    {"transform": transform, "ref": ref, "fractions_by_checkpoint": fractions}
                )
        minimum_ref = min((entry["ref"] for entry in qualifiers), default=None)
        minimum_ref_qualifiers = [
            entry for entry in qualifiers if entry["ref"] == minimum_ref
        ]
        result.update(
            {
                "delta_star": delta_star,
                "part_b": part_b,
                "part_c": part_c,
                "qualifying_candidates": qualifiers,
                "minimum_ref_qualifiers": minimum_ref_qualifiers,
                "selected_candidate": (
                    minimum_ref_qualifiers[0] if len(minimum_ref_qualifiers) == 1 else None
                ),
                "selection_status": (
                    "selected"
                    if len(minimum_ref_qualifiers) == 1
                    else "no_candidate_meets_0.90"
                    if not qualifiers
                    else "minimum_ref_tie_requires_preregistered_tiebreak"
                ),
            }
        )

    if args.range_normalized:
        result = _build_range_normalized_result(
            args=args, grouped=grouped, base_result=result
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"part_a_pass": part_a_pass, "selection_status": result["selection_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
