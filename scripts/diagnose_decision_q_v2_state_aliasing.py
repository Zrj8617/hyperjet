"""State-sufficiency audit for frozen Decision-Q v2 replay diagnostics.

This script joins the existing root-cause state records with multi-root CRN
expectations.  It never restores a model, steps an environment, or trains.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-jsonl", type=Path, required=True)
    parser.add_argument("--multi-root-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _key(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(row["seed"]),
        int(row["checkpoint_update"]),
        int(row["slot_index"]),
        int(row["decision_order"]),
        str(row["task_id"]),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, ddof=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (matrix - mean) / scale


def _rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def _similarity(distance: float) -> float:
    return float(1.0 / (1.0 + max(float(distance), 0.0)))


def _rank(values: list[float], ids: list[int]) -> list[int]:
    return [
        int(uav_id)
        for uav_id, _ in sorted(zip(ids, values), key=lambda item: (-item[1], item[0]))
    ]


def _pairwise_agreement(
    left: dict[int, float], right: dict[int, float], overlap: list[int]
) -> float | None:
    scores = []
    for first in range(len(overlap)):
        for second in range(first + 1, len(overlap)):
            a, b = overlap[first], overlap[second]
            x = left[a] - left[b]
            y = right[a] - right[b]
            scores.append(0.5 if x == 0.0 or y == 0.0 else float((x > 0.0) == (y > 0.0)))
    return None if not scores else float(np.mean(scores))


def _truth_resolved_agreement(
    left_stats: dict[str, Any], right_stats: dict[str, Any]
) -> tuple[float | None, int]:
    def resolved_map(stats: dict[str, Any]) -> dict[tuple[int, int], int]:
        result = {}
        for pair in stats["paired_action_differences"]:
            if not pair["truth_order_resolved"]:
                continue
            a, b = int(pair["left_uav_id"]), int(pair["right_uav_id"])
            result[(min(a, b), max(a, b))] = int(float(pair["mean_difference"]) > 0.0) if a < b else int(float(pair["mean_difference"]) < 0.0)
        return result

    left = resolved_map(left_stats)
    right = resolved_map(right_stats)
    common = sorted(set(left).intersection(right))
    if not common:
        return None, 0
    return float(np.mean([left[pair] == right[pair] for pair in common])), len(common)


def _values(row: dict[str, Any], field: str) -> dict[int, float]:
    ids = [int(value) for value in row["legal_uav_ids"]]
    if field == "q":
        return dict(zip(ids, [float(value) for value in row["legal_q_vector"]]))
    if field == "target":
        return {int(item["uav_id"]): float(item["target"]) for item in row["forced_action_targets"]}
    if field == "truth":
        return {
            int(item["uav_id"]): float(item["mean_continuation_return"])
            for item in row["final_statistics"]["action_statistics"]
        }
    raise ValueError(field)


def _semantic_report() -> dict[str, Any]:
    return {
        "schema": "decision_q_v2_state_semantic_report_v1",
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "historical_training_snapshot_recovered": False,
        "decision_q_action_input_dim": 183,
        "global_context": {
            "dimension": 102,
            "time_semantics": "assembled at slot start before movement; reused for every offloading decision in the slot",
            "fields": [
                {"indices": [0, 64], "name": "pooled task embeddings", "source": "slot-start graph snapshot/task encoder", "time": "pre-move; graph frozen within slot"},
                {"indices": [64, 94], "name": "five UAV global rows", "source": "pre-move x/y, executor queue length/capacity, available-time delta, aggregate queued workload", "time": "pre-move and before same-slot reservations"},
                {"indices": [94, 97], "name": "active/ready/pending task summaries", "source": "slot-start graph snapshot", "time": "pre-move"},
                {"indices": [97, 102], "name": "queue summaries", "source": "executor mean/max queue, mean/max available delta, total workload", "time": "before same-slot reservations"},
            ],
            "not_present": [
                "all-UAV post-movement positions",
                "same-slot temporary reservations",
                "ordered executor queue/task identities",
                "reserved task identities",
            ],
        },
        "candidate_rows": {
            "dimension": 79,
            "one_row_per_candidate_uav": True,
            "fields": [
                {"indices": [0, 64], "name": "current task embedding", "time": "slot-start graph embedding; repeated across UAV rows"},
                {"indices": [64, 66], "name": "candidate UAV x/y", "time": "post-movement"},
                {"indices": [66, 71], "name": "queue length, remaining slots, available delta, queued workload, slot-assigned count", "time": "post-movement and after prior sequential reservations"},
                {"indices": [71, 79], "name": "transfer time/energy, waiting, compute time/energy, EFT-like incremental delay, return time/energy", "time": "post-movement and reservation-aware"},
            ],
            "legal_mask_time": "post-movement and after prior sequential reservations",
        },
        "per_action_q_input": {
            "layout": [
                {"indices": [0, 79], "name": "current action candidate row"},
                {"indices": [79, 181], "name": "slot-start global context"},
                {"indices": [181, 182], "name": "legal fraction"},
                {"indices": [182, 183], "name": "normalized decision order"},
            ],
            "important_boundary": "Q(s,a) sees only action a's post-move/reservation-aware row; other candidates' post-move and reservation-aware rows are absent from that action input",
        },
        "source_references": [
            "marl_models/mappo/clean_slot_orchestrator.py: prepare before movement and reuse critic_global_input",
            "marl_models/mappo/clean_ppo.py: build_clean_critic_non_graph_input/_critic_uav_global/_critic_queue_summary",
            "marl_models/mappo/clean_offloading_actor.py: sequential reservation and candidate record",
            "environment/assignment.py: 7 dynamic and 8 pair features",
            "marl_models/mappo/clean_offloading_decision_q_credit.py: candidate row + global context + legal/order suffix",
        ],
    }


def main() -> int:
    args = _parser().parse_args()
    states = {_key(row): row for row in _read_jsonl(args.state_jsonl)}
    roots = _read_jsonl(args.multi_root_jsonl)
    if len(states) != 27 or len(roots) != 27:
        raise ValueError("state aliasing audit requires the existing 27 decisions")
    joined = []
    for root in roots:
        state = states.get(_key(root))
        if state is None:
            raise KeyError(f"missing state row for {_key(root)}")
        row = dict(root)
        for field in ("critic_global_context", "candidate_features", "legal_q_inputs", "state_scalar_suffix"):
            row[field] = state[field]
        joined.append(row)

    globals_raw = np.asarray([row["critic_global_context"] for row in joined], dtype=np.float64)
    candidates_raw = np.asarray([row["candidate_features"] for row in joined], dtype=np.float64)
    suffix_raw = np.asarray([
        [row["state_scalar_suffix"]["legal_fraction"], row["state_scalar_suffix"]["normalized_decision_order"]]
        for row in joined
    ], dtype=np.float64)
    global_z = _zscore_rows(globals_raw)
    candidate_z = _zscore_rows(candidates_raw.reshape(len(joined), -1))
    candidate_task_z = _zscore_rows(candidates_raw[:, :, 0:64].reshape(len(joined), -1))
    candidate_position_z = _zscore_rows(candidates_raw[:, :, 64:66].reshape(len(joined), -1))
    candidate_reservation_z = _zscore_rows(candidates_raw[:, :, 66:71].reshape(len(joined), -1))
    candidate_pair_z = _zscore_rows(candidates_raw[:, :, 71:79].reshape(len(joined), -1))
    full_z = _zscore_rows(np.concatenate([globals_raw, candidates_raw.reshape(len(joined), -1), suffix_raw], axis=1))

    pairs = []
    for first in range(len(joined)):
        for second in range(first + 1, len(joined)):
            left, right = joined[first], joined[second]
            left_ids = [int(value) for value in left["legal_uav_ids"]]
            right_ids = [int(value) for value in right["legal_uav_ids"]]
            overlap = sorted(set(left_ids).intersection(right_ids))
            union = set(left_ids).union(right_ids)
            q_agreement = _pairwise_agreement(_values(left, "q"), _values(right, "q"), overlap)
            target_agreement = _pairwise_agreement(_values(left, "target"), _values(right, "target"), overlap)
            truth_agreement = _pairwise_agreement(_values(left, "truth"), _values(right, "truth"), overlap)
            resolved_agreement, resolved_count = _truth_resolved_agreement(left["final_statistics"], right["final_statistics"])
            pairs.append({
                "left": {"seed": left["seed"], "update": left["checkpoint_update"], "slot": left["slot_index"], "order": left["decision_order"], "task": left["task_id"]},
                "right": {"seed": right["seed"], "update": right["checkpoint_update"], "slot": right["slot_index"], "order": right["decision_order"], "task": right["task_id"]},
                "same_checkpoint_policy": int(left["seed"]) == int(right["seed"]) and int(left["checkpoint_update"]) == int(right["checkpoint_update"]),
                "global_distance": _rms_distance(global_z[first], global_z[second]),
                "candidate_distance": _rms_distance(candidate_z[first], candidate_z[second]),
                "task_embedding_distance": _rms_distance(candidate_task_z[first], candidate_task_z[second]),
                "post_move_position_distance": _rms_distance(candidate_position_z[first], candidate_position_z[second]),
                "reservation_state_distance": _rms_distance(candidate_reservation_z[first], candidate_reservation_z[second]),
                "pair_estimate_distance": _rms_distance(candidate_pair_z[first], candidate_pair_z[second]),
                "full_state_distance": _rms_distance(full_z[first], full_z[second]),
                "legal_action_jaccard": float(len(overlap) / max(len(union), 1)),
                "q_ranking_agreement": q_agreement,
                "target_ranking_agreement": target_agreement,
                "truth_ranking_agreement": truth_agreement,
                "truth_resolved_pair_agreement": resolved_agreement,
                "truth_resolved_pair_count": resolved_count,
                "q_top1_agreement": _rank(list(_values(left, "q").values()), list(_values(left, "q").keys()))[0] == _rank(list(_values(right, "q").values()), list(_values(right, "q").keys()))[0],
                "target_top1_agreement": _rank(list(_values(left, "target").values()), list(_values(left, "target").keys()))[0] == _rank(list(_values(right, "target").values()), list(_values(right, "target").keys()))[0],
                "truth_top1_agreement": int(left["final_statistics"]["mean_return_top1"]) == int(right["final_statistics"]["mean_return_top1"]),
            })
    full_distances = np.asarray([row["full_state_distance"] for row in pairs], dtype=np.float64)
    p10, p25, p50 = [float(np.quantile(full_distances, value)) for value in (0.10, 0.25, 0.50)]
    for row in pairs:
        row["global_similarity"] = _similarity(row["global_distance"])
        row["candidate_similarity"] = _similarity(row["candidate_distance"])
        row["full_state_similarity"] = _similarity(row["full_state_distance"])
        row["high_similarity_p10"] = row["full_state_distance"] <= p10 and row["legal_action_jaccard"] >= 0.8
        row["high_similarity_p25"] = row["full_state_distance"] <= p25 and row["legal_action_jaccard"] >= 0.8

    # Action-level audit: compare the exact per-action Q input against information
    # from the other candidate rows that is excluded from Q(s,a).
    action_rows = []
    for decision_index, row in enumerate(joined):
        ids = [int(value) for value in row["legal_uav_ids"]]
        candidates = np.asarray(row["candidate_features"], dtype=np.float64)
        q_inputs = np.asarray(row["legal_q_inputs"], dtype=np.float64)
        truth = {int(item["uav_id"]): item for item in row["final_statistics"]["action_statistics"]}
        for action_index, uav_id in enumerate(ids):
            others = np.delete(candidates[:, 64:79], action_index, axis=0).reshape(-1)
            action_rows.append({
                "decision_index": decision_index,
                "seed": int(row["seed"]),
                "update": int(row["checkpoint_update"]),
                "uav_id": uav_id,
                "q_input": q_inputs[action_index],
                "other_candidate_state": others,
                "truth_mean": float(truth[uav_id]["mean_continuation_return"]),
                "truth_ci": truth[uav_id]["ci95"],
            })
    q_input_z = _zscore_rows(np.stack([row["q_input"] for row in action_rows]))
    other_z = _zscore_rows(np.stack([row["other_candidate_state"] for row in action_rows]))
    action_pairs = []
    for first in range(len(action_rows)):
        for second in range(first + 1, len(action_rows)):
            left, right = action_rows[first], action_rows[second]
            if left["uav_id"] != right["uav_id"]:
                continue
            q_distance = _rms_distance(q_input_z[first], q_input_z[second])
            hidden_distance = _rms_distance(other_z[first], other_z[second])
            left_ci, right_ci = left["truth_ci"], right["truth_ci"]
            ci_nonoverlap = bool(left_ci and right_ci and (left_ci[1] < right_ci[0] or right_ci[1] < left_ci[0]))
            action_pairs.append({
                "uav_id": left["uav_id"],
                "left_decision_index": left["decision_index"],
                "right_decision_index": right["decision_index"],
                "same_checkpoint_policy": left["seed"] == right["seed"] and left["update"] == right["update"],
                "q_input_distance": q_distance,
                "q_input_similarity": _similarity(q_distance),
                "excluded_other_candidate_distance": hidden_distance,
                "truth_mean_absolute_difference": abs(left["truth_mean"] - right["truth_mean"]),
                "truth_mean_ci_nonoverlap": ci_nonoverlap,
            })
    q_distances = np.asarray([row["q_input_distance"] for row in action_pairs])
    truth_differences = np.asarray([row["truth_mean_absolute_difference"] for row in action_pairs])
    q_p10 = float(np.quantile(q_distances, 0.10))
    truth_p75 = float(np.quantile(truth_differences, 0.75))
    for row in action_pairs:
        row["action_alias_candidate"] = bool(
            row["q_input_distance"] <= q_p10
            and row["truth_mean_absolute_difference"] >= truth_p75
            and row["truth_mean_ci_nonoverlap"]
        )

    same_checkpoint_pairs = [row for row in pairs if row["same_checkpoint_policy"]]
    same_checkpoint_distances = np.asarray(
        [row["full_state_distance"] for row in same_checkpoint_pairs], dtype=np.float64
    )
    same_checkpoint_p25 = float(np.quantile(same_checkpoint_distances, 0.25))
    same_checkpoint_near = [
        row
        for row in same_checkpoint_pairs
        if row["full_state_distance"] <= same_checkpoint_p25
    ]
    same_checkpoint_action_pairs = [
        row for row in action_pairs if row["same_checkpoint_policy"]
    ]
    same_checkpoint_q_distances = np.asarray(
        [row["q_input_distance"] for row in same_checkpoint_action_pairs],
        dtype=np.float64,
    )
    same_checkpoint_truth_differences = np.asarray(
        [row["truth_mean_absolute_difference"] for row in same_checkpoint_action_pairs],
        dtype=np.float64,
    )
    same_checkpoint_q_p25 = float(np.quantile(same_checkpoint_q_distances, 0.25))
    same_checkpoint_truth_p75 = float(
        np.quantile(same_checkpoint_truth_differences, 0.75)
    )
    same_checkpoint_action_aliases = [
        row
        for row in same_checkpoint_action_pairs
        if row["q_input_distance"] <= same_checkpoint_q_p25
        and row["truth_mean_absolute_difference"] >= same_checkpoint_truth_p75
        and row["truth_mean_ci_nonoverlap"]
    ]
    same_checkpoint_case_a = [
        row
        for row in same_checkpoint_near
        if row["truth_ranking_agreement"] >= 0.8
        and row["q_ranking_agreement"] < 0.5
    ]
    same_checkpoint_case_b = [
        row
        for row in same_checkpoint_near
        if row["truth_ranking_agreement"] < 0.5
    ]

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "pair_count": len(selected),
            "mean_truth_ranking_agreement": None if not selected else float(np.mean([row["truth_ranking_agreement"] for row in selected])),
            "truth_top1_disagreement_fraction": None if not selected else float(np.mean([not row["truth_top1_agreement"] for row in selected])),
            "mean_target_ranking_agreement": None if not selected else float(np.mean([row["target_ranking_agreement"] for row in selected])),
            "target_top1_disagreement_fraction": None if not selected else float(np.mean([not row["target_top1_agreement"] for row in selected])),
            "mean_q_ranking_agreement": None if not selected else float(np.mean([row["q_ranking_agreement"] for row in selected])),
            "q_top1_disagreement_fraction": None if not selected else float(np.mean([not row["q_top1_agreement"] for row in selected])),
            "resolved_truth_comparable_pair_count": sum(row["truth_resolved_pair_count"] > 0 for row in selected),
        }

    high10 = [row for row in pairs if row["high_similarity_p10"]]
    high25 = [row for row in pairs if row["high_similarity_p25"]]
    alias_actions = [row for row in action_pairs if row["action_alias_candidate"]]
    summary = {
        "schema": "decision_q_v2_state_aliasing_summary_v1",
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "historical_training_snapshot_recovered": False,
        "decision_count": len(joined),
        "decision_pair_count": len(pairs),
        "distance_thresholds": {"full_state_p10": p10, "full_state_p25": p25, "full_state_median": p50, "action_q_input_p10": q_p10, "truth_difference_p75": truth_p75},
        "high_similarity_p10": aggregate(high10),
        "high_similarity_p25": aggregate(high25),
        "all_pairs": aggregate(pairs),
        "case_counts_p10": {
            "A_state_similar_truth_consistent_q_different": sum(row["truth_ranking_agreement"] >= 0.8 and row["q_ranking_agreement"] < 0.5 for row in high10),
            "B_state_similar_truth_different": sum(row["truth_ranking_agreement"] < 0.5 for row in high10),
        },
        "action_level": {
            "same_uav_pair_count": len(action_pairs),
            "near_q_input_pair_count": sum(row["q_input_distance"] <= q_p10 for row in action_pairs),
            "near_q_input_ci_nonoverlap_count": sum(row["q_input_distance"] <= q_p10 and row["truth_mean_ci_nonoverlap"] for row in action_pairs),
            "strict_alias_candidate_count": len(alias_actions),
            "strict_alias_candidates": alias_actions,
            "q_distance_truth_difference_correlation": float(np.corrcoef(q_distances, truth_differences)[0, 1]),
        },
        "policy_consistent_same_checkpoint": {
            "decision_pairs": aggregate(same_checkpoint_pairs),
            "near_state_p25_threshold": same_checkpoint_p25,
            "near_state_pairs": aggregate(same_checkpoint_near),
            "near_state_case_counts": {
                "A_state_similar_truth_consistent_q_different": sum(
                    row["truth_ranking_agreement"] >= 0.8
                    and row["q_ranking_agreement"] < 0.5
                    for row in same_checkpoint_near
                ),
                "B_state_similar_truth_different": sum(
                    row["truth_ranking_agreement"] < 0.5
                    for row in same_checkpoint_near
                ),
            },
            "near_state_case_A_rows": same_checkpoint_case_a,
            "near_state_case_B_rows": same_checkpoint_case_b,
            "state_distance_truth_ranking_agreement_correlation": float(
                np.corrcoef(
                    same_checkpoint_distances,
                    np.asarray(
                        [row["truth_ranking_agreement"] for row in same_checkpoint_pairs],
                        dtype=np.float64,
                    ),
                )[0, 1]
            ),
            "action_pairs": {
                "pair_count": len(same_checkpoint_action_pairs),
                "q_input_p25_threshold": same_checkpoint_q_p25,
                "truth_difference_p75_threshold": same_checkpoint_truth_p75,
                "near_q_input_pair_count": sum(
                    row["q_input_distance"] <= same_checkpoint_q_p25
                    for row in same_checkpoint_action_pairs
                ),
                "near_q_input_ci_nonoverlap_count": sum(
                    row["q_input_distance"] <= same_checkpoint_q_p25
                    and row["truth_mean_ci_nonoverlap"]
                    for row in same_checkpoint_action_pairs
                ),
                "strict_alias_candidate_count": len(same_checkpoint_action_aliases),
                "strict_alias_candidates": same_checkpoint_action_aliases,
                "q_distance_truth_difference_correlation": float(
                    np.corrcoef(
                        same_checkpoint_q_distances,
                        same_checkpoint_truth_differences,
                    )[0, 1]
                ),
            },
        },
        "interpretation_guard": "Near-neighbor disagreement is evidence of observational aliasing, not proof that any one omitted variable is causal.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "state_semantic_report.json").write_text(json.dumps(_semantic_report(), indent=2, sort_keys=True), encoding="utf-8")
    (args.output / "state_aliasing_pairs.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sorted(pairs, key=lambda item: item["full_state_distance"])), encoding="utf-8")
    (args.output / "state_action_aliasing_pairs.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sorted(action_pairs, key=lambda item: item["q_input_distance"])), encoding="utf-8")
    (args.output / "state_aliasing_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
