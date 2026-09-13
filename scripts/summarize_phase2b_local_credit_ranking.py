"""Phase 2B-lite read-only ranking audit for Phase 1 local credit outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-decisions", type=Path, required=True)
    parser.add_argument("--multi-root-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(row["seed"]),
        int(row["checkpoint_update"]),
        int(row["slot_index"]),
        int(row["decision_order"]),
        str(row["task_id"]),
    )


def _sign(value: float) -> int:
    return int(value > 0.0) - int(value < 0.0)


def _tie_aware_score(left: float, right: float) -> float:
    left_sign, right_sign = _sign(left), _sign(right)
    if left_sign == 0 or right_sign == 0:
        return 0.5
    return float(left_sign == right_sign)


def _rank(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty_like(array)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_rank, right_rank = _rank(left), _rank(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _truth_vectors(source: dict[str, Any]) -> dict[str, Any]:
    legal_ids = [int(value) for value in source["legal_uav_ids"]]
    truth_by_id = {
        int(row["uav_id"]): float(row["mean_continuation_return"])
        for row in source["final_statistics"]["action_statistics"]
    }
    target_by_id = {
        int(row["uav_id"]): float(row["target"])
        for row in source["forced_action_targets"]
    }
    resolved_by_pair: dict[tuple[int, int], bool] = {}
    for row in source["final_statistics"]["paired_action_differences"]:
        left, right = int(row["left_uav_id"]), int(row["right_uav_id"])
        resolved_by_pair[(left, right)] = bool(row["truth_order_resolved"])
        resolved_by_pair[(right, left)] = bool(row["truth_order_resolved"])
    return {
        "legal_ids": legal_ids,
        "truth": [truth_by_id[value] for value in legal_ids],
        "target": [target_by_id[value] for value in legal_ids],
        "q": [float(value) for value in source["legal_q_vector"]],
        "actor": [float(value) for value in source["actor_legal_probabilities"]],
        "truth_by_id": truth_by_id,
        "target_by_id": target_by_id,
        "resolved_by_pair": resolved_by_pair,
    }


def _decision_row(local: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    vectors = _truth_vectors(source)
    actual, alternative = int(local["actual_uav_id"]), int(local["alternative_uav_id"])
    ids = vectors["legal_ids"]
    q_by_id = dict(zip(ids, vectors["q"]))
    actor_by_id = dict(zip(ids, vectors["actor"]))
    local_difference = float(local["credit_statistics"]["mean_c_i_local"])
    first_root = local["root_results"][0]
    suffix_changed = (
        first_root["actual"]["suffix_assignments"]
        != first_root["counterfactual"]["suffix_assignments"]
    )
    differences = {
        "local": local_difference,
        "q": float(q_by_id[actual] - q_by_id[alternative]),
        "actor": float(actor_by_id[actual] - actor_by_id[alternative]),
        "recursive_target": float(
            vectors["target_by_id"][actual] - vectors["target_by_id"][alternative]
        ),
        "truth": float(
            vectors["truth_by_id"][actual] - vectors["truth_by_id"][alternative]
        ),
    }
    return {
        "seed": int(local["seed"]),
        "phase": str(local["phase"]),
        "checkpoint_update": int(local["checkpoint_update"]),
        "slot_index": int(local["slot_index"]),
        "decision_order": int(local["decision_order"]),
        "task_id": str(local["task_id"]),
        "actual_uav_id": actual,
        "alternative_uav_id": alternative,
        "c_local": local_difference,
        "abs_c_local": abs(local_difference),
        "actual_reward_mean": float(
            local["credit_statistics"]["actual_reward_mean"]
        ),
        "counterfactual_reward_mean": float(
            local["credit_statistics"]["counterfactual_reward_mean"]
        ),
        "suffix_changed_after_intervention": bool(suffix_changed),
        "local_order_resolved": _sign(local_difference) != 0,
        "truth_pair_resolved": bool(vectors["resolved_by_pair"].get((actual, alternative), False)),
        "pair_differences": differences,
        "sampled_pair_agreement": {
            name: _tie_aware_score(local_difference, differences[name])
            for name in ("q", "actor", "recursive_target", "truth")
        },
        "full_legal_vectors": {
            "legal_uav_ids": ids,
            "q": vectors["q"],
            "actor": vectors["actor"],
            "recursive_target": vectors["target"],
            "truth": vectors["truth"],
            "c_local": None,
        },
    }


def _sampled_pair_metrics(rows: list[dict[str, Any]], comparator: str) -> dict[str, Any]:
    resolved_local = [
        row
        for row in rows
        if row["local_order_resolved"]
        and _sign(float(row["pair_differences"][comparator])) != 0
    ]
    truth_resolved = (
        [row for row in resolved_local if row["truth_pair_resolved"]]
        if comparator == "truth"
        else resolved_local
    )
    scores = [
        _tie_aware_score(
            float(row["pair_differences"]["local"]),
            float(row["pair_differences"][comparator]),
        )
        for row in rows
    ]
    resolved_scores = [
        float(
            _sign(float(row["pair_differences"]["local"]))
            == _sign(float(row["pair_differences"][comparator]))
        )
        for row in resolved_local
    ]
    truth_resolved_scores = [
        float(
            _sign(float(row["pair_differences"]["local"]))
            == _sign(float(row["pair_differences"][comparator]))
        )
        for row in truth_resolved
    ]
    return {
        "decision_count": len(rows),
        "tie_aware_pairwise_accuracy": None if not scores else float(np.mean(scores)),
        "resolved_pair_count": len(resolved_local),
        "resolved_pair_top1_agreement": (
            None if not resolved_scores else float(np.mean(resolved_scores))
        ),
        "resolved_pair_spearman": (
            None
            if not resolved_scores
            else float(np.mean([1.0 if score == 1.0 else -1.0 for score in resolved_scores]))
        ),
        "truth_ci_resolved_pair_count": len(truth_resolved) if comparator == "truth" else None,
        "truth_ci_resolved_accuracy": (
            None
            if comparator != "truth" or not truth_resolved_scores
            else float(np.mean(truth_resolved_scores))
        ),
        "scope": "actual_vs_one_uniformly_sampled_legal_alternative",
    }


def _comparator_vs_truth_pair(rows: list[dict[str, Any]], comparator: str) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if _sign(float(row["pair_differences"][comparator])) != 0
        and _sign(float(row["pair_differences"]["truth"])) != 0
    ]
    resolved = [row for row in usable if row["truth_pair_resolved"]]
    def accuracy(selected: list[dict[str, Any]]) -> float | None:
        if not selected:
            return None
        return float(np.mean([
            _sign(float(row["pair_differences"][comparator]))
            == _sign(float(row["pair_differences"]["truth"]))
            for row in selected
        ]))
    return {
        "non_tied_pair_count": len(usable),
        "pairwise_accuracy": accuracy(usable),
        "truth_ci_resolved_pair_count": len(resolved),
        "truth_ci_resolved_accuracy": accuracy(resolved),
    }


def _same_local_resolved_subset_vs_truth(
    rows: list[dict[str, Any]], comparator: str
) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row["local_order_resolved"]
        and _sign(float(row["pair_differences"][comparator])) != 0
        and _sign(float(row["pair_differences"]["truth"])) != 0
    ]
    resolved = [row for row in usable if row["truth_pair_resolved"]]

    def accuracy(selected: list[dict[str, Any]]) -> float | None:
        if not selected:
            return None
        return float(np.mean([
            _sign(float(row["pair_differences"][comparator]))
            == _sign(float(row["pair_differences"]["truth"]))
            for row in selected
        ]))

    return {
        "pair_count": len(usable),
        "pairwise_accuracy": accuracy(usable),
        "truth_ci_resolved_pair_count": len(resolved),
        "truth_ci_resolved_accuracy": accuracy(resolved),
        "scope": "same decisions where C_local resolves actual-vs-alternative order",
    }


def _full_legal_metrics(rows: list[dict[str, Any]], comparator: str) -> dict[str, Any]:
    top1: list[float] = []
    spearman: list[float] = []
    pairwise: list[float] = []
    for row in rows:
        vectors = row["full_legal_vectors"]
        predicted = [float(value) for value in vectors[comparator]]
        truth = [float(value) for value in vectors["truth"]]
        top1.append(float(int(np.argmax(predicted)) == int(np.argmax(truth))))
        correlation = _spearman(predicted, truth)
        if correlation is not None:
            spearman.append(correlation)
        for left in range(len(predicted)):
            for right in range(left + 1, len(predicted)):
                pairwise.append(
                    _tie_aware_score(
                        predicted[left] - predicted[right], truth[left] - truth[right]
                    )
                )
    return {
        "decision_count": len(rows),
        "top1_accuracy": None if not top1 else float(np.mean(top1)),
        "spearman_mean": None if not spearman else float(np.mean(spearman)),
        "pairwise_accuracy": None if not pairwise else float(np.mean(pairwise)),
    }


def _density(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([float(row["c_local"]) for row in rows], dtype=np.float64)
    positive = int(np.count_nonzero(values > 0.0))
    negative = int(np.count_nonzero(values < 0.0))
    nonzero = positive + negative
    return {
        "decision_count": int(values.size),
        "nonzero_count": nonzero,
        "nonzero_fraction": None if values.size == 0 else float(np.mean(values != 0.0)),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": int(values.size) - nonzero,
        "positive_fraction_of_nonzero": (
            None if nonzero == 0 else float(positive / nonzero)
        ),
        "negative_fraction_of_nonzero": (
            None if nonzero == 0 else float(negative / nonzero)
        ),
        "mean_abs_c_local": None if values.size == 0 else float(np.mean(np.abs(values))),
        "std_c_local": None if values.size < 2 else float(np.std(values, ddof=1)),
        "mean_c_local": None if values.size == 0 else float(np.mean(values)),
    }


def _action_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [row for row in rows if row["suffix_changed_after_intervention"]]
    unchanged = [row for row in rows if not row["suffix_changed_after_intervention"]]

    def subset(selected: list[dict[str, Any]]) -> dict[str, Any]:
        values = np.asarray([float(row["c_local"]) for row in selected], dtype=np.float64)
        return {
            "decision_count": int(values.size),
            "nonzero_count": int(np.count_nonzero(values)),
            "nonzero_fraction": (
                None if values.size == 0 else float(np.mean(values != 0.0))
            ),
            "mean_abs_c_local": (
                None if values.size == 0 else float(np.mean(np.abs(values)))
            ),
        }

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{int(row['actual_uav_id'])}->{int(row['alternative_uav_id'])}"
        groups.setdefault(key, []).append(row)
    return {
        "suffix_changed": subset(changed),
        "suffix_unchanged": subset(unchanged),
        "by_actual_to_alternative_pair": {
            key: subset(selected) for key, selected in sorted(groups.items())
        },
        "interpretation": (
            "The intervention includes both the forced current action and the "
            "frozen-policy suffix response; it is not a direct-effect-only estimate."
        ),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "credit_density": _density(rows),
        "action_sensitivity": _action_sensitivity(rows),
        "c_local_sampled_pair_ranking": {
            comparator: _sampled_pair_metrics(rows, comparator)
            for comparator in ("q", "actor", "recursive_target", "truth")
        },
        "sampled_pair_vs_truth": {
            comparator: _comparator_vs_truth_pair(rows, comparator)
            for comparator in ("local", "q", "actor", "recursive_target")
        },
        "same_local_resolved_subset_vs_truth": {
            comparator: _same_local_resolved_subset_vs_truth(rows, comparator)
            for comparator in ("local", "q", "actor", "recursive_target")
        },
        "full_legal_ranking_vs_truth": {
            "c_local": {
                "top1_accuracy": None,
                "spearman_mean": None,
                "pairwise_accuracy": None,
                "reason": "Phase 1 sampled only one alternative; a full legal C_local vector does not exist",
            },
            **{
                comparator: _full_legal_metrics(rows, comparator)
                for comparator in ("q", "actor", "recursive_target")
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    local_rows = _read_jsonl(args.phase1_decisions)
    truth_rows = {_key(row): row for row in _read_jsonl(args.multi_root_truth)}
    if len(local_rows) != 27 or len(truth_rows) != 27:
        raise ValueError("Phase 2B-lite requires the existing 27-decision audits")
    decisions = []
    for local in local_rows:
        source = truth_rows.get(_key(local))
        if source is None:
            raise ValueError(f"missing multi-root truth row for {_key(local)}")
        decisions.append(_decision_row(local, source))
    summary = {
        "schema": "phase2b_lite_local_credit_ranking_summary_v1",
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "new_rollout_generated": False,
        "training_performed": False,
        "decision_count": len(decisions),
        "pooled": _summarize(decisions),
        "by_phase": {
            phase: _summarize([row for row in decisions if row["phase"] == phase])
            for phase in ("early", "mid", "late")
        },
        "by_seed": {
            str(seed): _summarize([row for row in decisions if row["seed"] == seed])
            for seed in (0, 1, 2)
        },
        "interpretation_boundary": {
            "full_legal_c_local_ranking_available": False,
            "sampled_pair_ranking_available": True,
            "reason": "one uniform legal alternative was sampled per decision in Phase 1",
        },
        "decision_rows": decisions,
    }
    pooled_same_subset = summary["pooled"]["same_local_resolved_subset_vs_truth"]
    local_accuracy = pooled_same_subset["local"]["pairwise_accuracy"]
    target_accuracy = pooled_same_subset["recursive_target"]["pairwise_accuracy"]
    q_accuracy = pooled_same_subset["q"]["pairwise_accuracy"]
    actor_accuracy = pooled_same_subset["actor"]["pairwise_accuracy"]
    summary["supervision_candidate_assessment"] = {
        "supports_c_local_over_existing_signals": False,
        "local_vs_truth_accuracy_on_same_resolved_subset": local_accuracy,
        "recursive_target_vs_truth_accuracy_on_same_subset": target_accuracy,
        "decision_q_vs_truth_accuracy_on_same_subset": q_accuracy,
        "actor_vs_truth_accuracy_on_same_subset": actor_accuracy,
        "conclusion": (
            "The available sampled-pair evidence does not show C_local to be closer "
            "to continuation truth than the existing recursive target, Decision-Q, "
            "or actor preference."
        ),
        "evidence_limit": (
            "Only one alternative exists per decision and only two sampled truth "
            "pairs have resolved confidence intervals."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision_count": len(decisions)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
