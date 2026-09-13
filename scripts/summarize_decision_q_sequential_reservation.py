"""Compare archived order-0 and sequential Decision-Q multi-root audits."""

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

from scripts.diagnose_decision_q_v2_ranking_crn import _multi_root_calibration


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = _multi_root_calibration(rows)
    return {
        "decision_count": len(rows),
        "usable_decision_count": result["usable_decision_count"],
        "censor_fraction": result["censor_fraction"],
        "q_vs_target": result["q_vs_training_target"],
        "target_vs_crn_truth": result["target_vs_expected_truth"],
        "q_vs_crn_truth": result["q_vs_expected_truth"],
    }


def _breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pooled": _metric(rows),
        "same_slot": _metric([row for row in rows if row["same_slot_primary"]]),
        "delta_positive": _metric([row for row in rows if not row["same_slot_primary"]]),
        "by_phase": {
            phase: _metric([row for row in rows if row["phase"] == phase])
            for phase in ("early", "mid", "late")
        },
        "by_seed": {
            str(seed): _metric([row for row in rows if int(row["seed"]) == seed])
            for seed in (0, 1, 2)
        },
    }


def _reservation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [row["sequential_reservation_audit"] for row in rows]
    l2 = [float(row["candidate_feature_delta_l2"]) for row in audits]
    maximum = [float(row["candidate_feature_delta_max_abs"]) for row in audits]
    changed = [int(row["candidate_feature_changed_count"]) for row in audits]
    orders = [int(row["decision_order"]) for row in rows]
    reconstructed = [float(row["current_state_reconstruction_max_abs_error"]) for row in audits]
    legal_changed = [
        list(row["slot_start_legal_mask"]) != list(row["before_current_legal_mask"])
        for row in audits
    ]
    return {
        "decision_order_distribution": {
            str(order): int(sum(value == order for value in orders))
            for order in sorted(set(orders))
        },
        "prefix_decision_count_mean": float(
            np.mean([int(row["prefix_decision_count"]) for row in audits])
        ),
        "candidate_feature_delta_l2_mean": float(np.mean(l2)),
        "candidate_feature_delta_l2_max": float(np.max(l2)),
        "candidate_feature_delta_max_abs_mean": float(np.mean(maximum)),
        "candidate_feature_changed_count_mean": float(np.mean(changed)),
        "candidate_feature_changed_fraction": float(np.mean(np.asarray(changed) > 0)),
        "legal_mask_changed_fraction": float(np.mean(legal_changed)),
        "state_reconstruction_max_abs_error": float(np.max(reconstructed)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order0", type=Path, required=True)
    parser.add_argument("--order-positive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    order0 = _read(args.order0)
    order_positive = _read(args.order_positive)
    if len(order0) != 27 or len(order_positive) != 27:
        raise ValueError("comparison requires 27 order-0 and 27 order-positive decisions")
    if any(int(row["decision_order"]) != 0 for row in order0):
        raise ValueError("order-0 source contains a sequential decision")
    if any(int(row["decision_order"]) <= 0 for row in order_positive):
        raise ValueError("sequential source contains a non-sequential decision")
    payload = {
        "schema": "decision_q_v2_sequential_reservation_comparison_v1",
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "historical_training_snapshot_recovered": False,
        "order0": _breakdown(order0),
        "order_positive": _breakdown(order_positive),
        "reservation_feature_effect": _reservation(order_positive),
        "gates": {
            "order0_decision_count": len(order0),
            "order_positive_decision_count": len(order_positive),
            "optimizer_steps_total": sum(int(row["optimizer_steps"]) for row in order_positive),
            "semantic_mismatch_count": sum(
                int(row["gates"]["semantic_mismatch_count"]) for row in order_positive
            ),
            "unrecognized_environment_rng_calls": sum(
                int(row["gates"]["unrecognized_environment_rng_calls"])
                for row in order_positive
            ),
            "all_training_parameters_unchanged": all(
                row["gates"].get("training_parameters_unchanged", False)
                for row in order_positive
            ),
            "all_strict_suffix_feasible": all(
                row["gates"]["strict_suffix_feasible_all"] for row in order_positive
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gates": payload["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
