from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_models.mappo.clean_decision_td_dataset import (
    PHASE4_RHO_ZERO_TOLERANCE,
    clean_decision_slot_position_flags,
    load_clean_decision_td_raw_dataset,
)


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_rank(left), _rank(right))


def _mean_std(values: np.ndarray) -> dict[str, float]:
    rows = np.asarray(values, dtype=np.float64).reshape(-1)
    return {"mean": float(np.mean(rows)), "std": float(np.std(rows))}


def _q_ev(target: np.ndarray, prediction: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    variance = float(np.var(target))
    if target.size < 2 or variance <= 1e-12:
        return None
    return 1.0 - float(np.var(target - prediction)) / variance


def _group_stats(dataset: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    rho = np.asarray(dataset["rho"], dtype=np.float64)[selected]
    bootstrap = np.asarray(dataset["bootstrap_value"], dtype=np.float64)[selected]
    target = np.asarray(dataset["td_target"], dtype=np.float64)[selected]
    prediction = np.asarray(
        dataset["online_selected_q_prediction"], dtype=np.float64
    )[selected]
    if target.size == 0:
        return {"count": 0}
    return {
        "count": int(target.size),
        "rho": _mean_std(rho),
        "bootstrap": _mean_std(bootstrap),
        "target": _mean_std(target),
        "q_prediction": _mean_std(prediction),
        "q_explained_variance": _q_ev(target, prediction),
        "td_error_abs_mean": float(np.mean(np.abs(target - prediction))),
    }


def _slot_groups(dataset: dict[str, np.ndarray]) -> dict[tuple[int, int, int], list[int]]:
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, values in enumerate(
        zip(
            dataset["episode_index"].tolist(),
            dataset["lane_index"].tolist(),
            dataset["slot_index"].tolist(),
        )
    ):
        groups.setdefault(tuple(int(value) for value in values), []).append(index)
    return groups


def _within_slot_std_mean(
    values: np.ndarray,
    groups: dict[tuple[int, int, int], list[int]],
    include: np.ndarray,
) -> tuple[float | None, int]:
    rows = np.asarray(values, dtype=np.float64)
    include = np.asarray(include, dtype=bool)
    deviations: list[float] = []
    for indices in groups.values():
        selected = [index for index in indices if bool(include[index])]
        if len(selected) >= 2:
            deviations.append(float(np.std(rows[selected])))
    if not deviations:
        return None, 0
    return float(np.mean(deviations)), len(deviations)


def analyze_dataset(path: str | Path) -> dict[str, Any]:
    dataset, metadata = load_clean_decision_td_raw_dataset(path)
    flags = clean_decision_slot_position_flags(dataset)
    count = int(dataset["td_target"].shape[0])
    all_rows = np.ones(count, dtype=bool)
    delta_zero = np.asarray(dataset["delta"], dtype=np.int64) == 0
    rho = np.asarray(dataset["rho"], dtype=np.float64)
    bootstrap = np.asarray(dataset["bootstrap_value"], dtype=np.float64)
    target = np.asarray(dataset["td_target"], dtype=np.float64)
    prediction = np.asarray(dataset["online_selected_q_prediction"], dtype=np.float64)
    tolerance = float(metadata["rho_zero_absolute_tolerance"])
    if tolerance != PHASE4_RHO_ZERO_TOLERANCE:
        raise ValueError("dataset rho-zero tolerance differs from the approved diagnostic value")
    rho_zero = np.abs(rho) <= tolerance
    groups = _slot_groups(dataset)
    within_all, within_all_groups = _within_slot_std_mean(target, groups, all_rows)
    within_nonlast, within_nonlast_groups = _within_slot_std_mean(
        target, groups, ~flags["is_last_decision"]
    )
    within_zero_rho, within_zero_rho_groups = _within_slot_std_mean(
        target, groups, rho_zero
    )

    target_variance = float(np.var(target))
    rho_variance = float(np.var(rho))
    bootstrap_variance = float(np.var(bootstrap))
    covariance = float(np.mean((rho - np.mean(rho)) * (bootstrap - np.mean(bootstrap))))
    identity_rhs = rho_variance + bootstrap_variance + 2.0 * covariance
    same_slot_identity_error = float(
        np.max(np.abs(target[delta_zero] - bootstrap[delta_zero]))
    ) if bool(delta_zero.any()) else None
    report = {
        "schema": "hyperuav_phase4_decision_td_raw_analysis_v1",
        "dataset_path": str(Path(path)),
        "metadata": metadata,
        "sample_count": count,
        "rho_zero_absolute_tolerance": tolerance,
        "delta_split": {
            "delta_eq_0": _group_stats(dataset, delta_zero),
            "delta_gt_0": _group_stats(dataset, ~delta_zero),
        },
        "rho_split": {
            "rho_eq_0": _group_stats(dataset, rho_zero),
            "rho_ne_0": _group_stats(dataset, ~rho_zero),
        },
        "decision_position": {
            "first": _group_stats(dataset, flags["is_first_decision"]),
            "middle": _group_stats(dataset, flags["is_middle_decision"]),
            "last": _group_stats(dataset, flags["is_last_decision"]),
            "singleton": _group_stats(dataset, flags["is_singleton"]),
            "multi_decision_slot": _group_stats(dataset, flags["is_multi_decision_slot"]),
        },
        "within_slot": {
            "within_slot_phase4_target_std_mean": within_all,
            "within_slot_phase4_group_count": within_all_groups,
            "within_slot_nonlast_target_std_mean": within_nonlast,
            "within_slot_nonlast_group_count": within_nonlast_groups,
            "within_slot_zero_rho_target_std_mean": within_zero_rho,
            "within_slot_zero_rho_group_count": within_zero_rho_groups,
            "same_slot_bootstrap_std": float(np.std(bootstrap[delta_zero])),
            "same_slot_target_std": float(np.std(target[delta_zero])),
            "same_slot_target_bootstrap_max_abs_error": same_slot_identity_error,
        },
        "variance_decomposition": {
            "var_target": target_variance,
            "var_rho": rho_variance,
            "var_bootstrap": bootstrap_variance,
            "cov_rho_bootstrap": covariance,
            "identity_rhs": identity_rhs,
            "identity_abs_error": abs(target_variance - identity_rhs),
            "var_rho_over_var_target": rho_variance / target_variance,
            "var_bootstrap_over_var_target": bootstrap_variance / target_variance,
        },
        "correlations": {
            "pearson_target_rho": _pearson(target, rho),
            "spearman_target_rho": _spearman(target, rho),
            "pearson_target_bootstrap": _pearson(target, bootstrap),
            "spearman_target_bootstrap": _spearman(target, bootstrap),
            "pearson_target_is_last": _pearson(target, flags["is_last_decision"]),
            "spearman_target_is_last": _spearman(target, flags["is_last_decision"]),
            "pearson_abs_target_is_last": _pearson(
                np.abs(target), flags["is_last_decision"]
            ),
            "pearson_q_prediction_target": _pearson(prediction, target),
            "spearman_q_prediction_target": _spearman(prediction, target),
            "pearson_rho_bootstrap": _pearson(rho, bootstrap),
            "spearman_rho_bootstrap": _spearman(rho, bootstrap),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = analyze_dataset(args.dataset)
    output = args.output or (args.dataset / "raw_decomposition.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
