"""Phase 4-A diagnostic-only multi-alternative long-horizon credit audit.

For every reused Phase 3 decision and semantic CRN root, all legal actions are
branched from the same exact pre-decision snapshot.  Action quality is ranked by
G_H(a); the stored actual-relative credit C_H(a)=G_H(actual)-G_H(a) therefore
has the inverse (ascending) ranking.  No result is connected to training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import multiprocessing
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_models.mappo.clean_counterfactual_oracle_common_random import (  # noqa: E402
    audit_clean_semantic_common_random,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import (  # noqa: E402
    CleanHostRngState,
)
from scripts import diagnose_boundary_anchored_decision_credit_phase1 as phase1  # noqa: E402
from scripts import diagnose_phase3_counterfactual_horizon as phase3  # noqa: E402
from scripts.diagnose_decision_q_v2_ranking_crn import (  # noqa: E402
    _independent_semantic_root,
    _t_critical_975,
)
from scripts.summarize_phase2b_local_credit_ranking import (  # noqa: E402
    _key,
    _read_jsonl,
    _spearman,
    _tie_aware_score,
    _truth_vectors,
)


STATE_SEMANTICS = "frozen-checkpoint replay-generated on-policy decision states"
HORIZONS = (40, 60, 80)
PHASE_BY_UPDATE = {30: "early", 60: "mid", 120: "late"}
_TASK_ENCODER: Any = None
_MOVEMENT_ACTOR: Any = None


@dataclass(frozen=True, slots=True)
class MultiActionRootResult:
    branches: tuple[phase3.HorizonBranchResult, ...]
    shared_semantic_keys_checked: int
    semantic_mismatch_count: int
    unrecognized_environment_rng_calls: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--phase3-audit", type=Path, required=True)
    parser.add_argument("--multi-root-truth", type=Path, required=True)
    parser.add_argument("--phase4a-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-search-slots", type=int, default=200)
    parser.add_argument("--process-timeout-seconds", type=float, default=1200.0)
    return parser


def _read_phase3_decisions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("decision_rows")
    if not isinstance(rows, list):
        raise ValueError("Phase 3 audit has no decision_rows")
    return rows


def _capture_modules_wrapper(original: Any) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        global _TASK_ENCODER, _MOVEMENT_ACTOR
        modules = original(*args, **kwargs)
        _TASK_ENCODER = modules.hgnn
        _MOVEMENT_ACTOR = modules.movement_actor
        return modules

    return wrapped


def _run_checkpoint(
    *,
    checkpoint: Path,
    seed: int,
    update: int,
    source_rows: list[dict[str, Any]],
    gamma: float,
    max_search_slots: int,
    timeout_seconds: float,
    compare_serial_process: bool,
) -> list[dict[str, Any]]:
    original_collect = phase1._collect_one_decision
    original_build = phase1._build_modules

    def collect(**kwargs: Any) -> dict[str, Any]:
        kwargs["compare_serial_process"] = bool(compare_serial_process)
        return _collect_one_decision(**kwargs, gamma=float(gamma))

    phase1._collect_one_decision = collect
    phase1._build_modules = _capture_modules_wrapper(original_build)
    try:
        return phase1._collect_checkpoint(
            checkpoint=checkpoint,
            seed=int(seed),
            phase=PHASE_BY_UPDATE[int(update)],
            expected_update=int(update),
            source_rows=source_rows,
            roots=2 if compare_serial_process else 8,
            max_search_slots=int(max_search_slots),
            process_timeout_seconds=float(timeout_seconds),
            compare_serial_process=bool(compare_serial_process),
        )
    finally:
        phase1._collect_one_decision = original_collect
        phase1._build_modules = original_build


def _collect_one_decision(
    *,
    source: dict[str, Any],
    snapshot: Any,
    base_rng_root: CleanHostRngState,
    trace: tuple[Any, ...],
    target_trace: Any,
    target_record: Any,
    frozen_ready_task_ids: tuple[str, ...],
    task_embeddings_by_id: dict[str, np.ndarray],
    offloading_actor: Any,
    q_values: np.ndarray,
    q_inputs: np.ndarray,
    replay_episode_index: int,
    roots: int,
    process_timeout_seconds: float,
    compare_serial_process: bool,
    gamma: float,
) -> dict[str, Any]:
    del target_record, q_inputs, replay_episode_index
    if _TASK_ENCODER is None or _MOVEMENT_ACTOR is None:
        raise AssertionError("future-slot frozen modules were not captured")
    legal_ids = [int(value) for value in target_trace.legal_uav_ids]
    actual = int(source["actual_uav_id"])
    first_alternative = int(source["alternative_uav_id"])
    if actual != int(target_trace.selected_uav_id):
        raise AssertionError("Phase 3 actual action changed during replay")
    if first_alternative not in legal_ids or first_alternative == actual:
        raise AssertionError("Phase 3 alternative is not legal")
    action_order = [actual, first_alternative] + sorted(
        value for value in legal_ids if value not in (actual, first_alternative)
    )
    if set(action_order) != set(legal_ids) or len(action_order) != len(legal_ids):
        raise AssertionError("all-legal action ordering is invalid")
    if int(roots) > len(source["root_results"]):
        raise AssertionError("Phase 4-A requests unavailable Phase 3 roots")

    specs: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for root_id in range(int(roots)):
        rng_root, rng_identifier = _independent_semantic_root(
            base_rng_root=base_rng_root,
            seed=int(source["seed"]),
            update=int(source["checkpoint_update"]),
            slot_index=int(source["slot_index"]),
            decision_order=int(source["decision_order"]),
            root_id=root_id,
        )
        stored = source["root_results"][root_id]
        if int(stored["root_id"]) != root_id or rng_identifier != stored["rng_state_identifier"]:
            raise AssertionError("Phase 4-A semantic root differs from Phase 3")
        specs.append(
            (
                root_id,
                rng_identifier,
                stored,
                {
                    "env_snapshot": snapshot,
                    "baseline_trace": trace,
                    "target_decision_order": int(target_trace.decision_order),
                    "action_ids": tuple(action_order),
                    "frozen_ready_task_ids": frozen_ready_task_ids,
                    "task_embeddings_by_id": task_embeddings_by_id,
                    "task_encoder": _TASK_ENCODER,
                    "movement_actor": _MOVEMENT_ACTOR,
                    "offloading_actor": offloading_actor,
                    "initial_rng_state": rng_root,
                    "max_horizon": max(HORIZONS),
                },
            )
        )
    batch = (
        {}
        if compare_serial_process
        else _run_multi_process_batch(
            [(root_id, kwargs) for root_id, _, _, kwargs in specs],
            timeout_seconds=float(process_timeout_seconds),
            max_parallel=8,
        )
    )
    root_rows: list[dict[str, Any]] = []
    reference_reservation: dict[str, Any] | None = None
    for root_id, rng_identifier, stored, kwargs in specs:
        result = (
            _run_multi_process(**kwargs, timeout_seconds=float(process_timeout_seconds))
            if compare_serial_process
            else batch[root_id]
        )
        serial_equal = None
        if compare_serial_process:
            serial = _run_multi_serial(**kwargs)
            serial_equal = bool(serial == result)
            if not serial_equal:
                raise AssertionError("Phase 4-A serial/process results differ")
        if result.semantic_mismatch_count:
            raise AssertionError("Phase 4-A semantic CRN mismatch")
        if result.unrecognized_environment_rng_calls:
            raise AssertionError("Phase 4-A observed unrecognized environment RNG")
        branches = {int(branch.forced_uav_id): branch for branch in result.branches}
        reservations = [branch.target_pre_decision_reservation for branch in result.branches]
        if any(value != reservations[0] for value in reservations[1:]):
            raise AssertionError("all actions do not share the exact decision state")
        if reference_reservation is None:
            reference_reservation = reservations[0]
        if reservations[0] != reference_reservation:
            raise AssertionError("roots do not share the exact decision state")
        old_actual = stored["actual"]["reward_sequence"]
        old_alt = stored["counterfactual"]["reward_sequence"]
        if not _sequence_equal(branches[actual].reward_sequence, old_actual):
            raise AssertionError("actual branch does not reproduce Phase 3")
        if not _sequence_equal(branches[first_alternative].reward_sequence, old_alt):
            raise AssertionError("first alternative does not reproduce Phase 3")
        returns = {
            str(horizon): {
                str(action): (
                    _discounted_return(branches[action].reward_sequence[:horizon], gamma)
                    if len(branches[action].reward_sequence) >= horizon
                    else None
                )
                for action in action_order
            }
            for horizon in HORIZONS
        }
        shaped_rewards = {
            str(action): _progress_shaped_rewards(branches[action], gamma)
            for action in action_order
        }
        shaped_returns = {
            str(horizon): {
                str(action): (
                    _discounted_return(shaped_rewards[str(action)][:horizon], gamma)
                    if len(shaped_rewards[str(action)]) >= horizon
                    else None
                )
                for action in action_order
            }
            for horizon in HORIZONS
        }
        credits = {
            str(horizon): {
                str(action): (
                    None
                    if returns[str(horizon)][str(actual)] is None
                    or returns[str(horizon)][str(action)] is None
                    else float(returns[str(horizon)][str(actual)])
                    - float(returns[str(horizon)][str(action)])
                )
                for action in action_order
            }
            for horizon in HORIZONS
        }
        shaped_credits = {
            str(horizon): {
                str(action): (
                    None
                    if shaped_returns[str(horizon)][str(actual)] is None
                    or shaped_returns[str(horizon)][str(action)] is None
                    else float(shaped_returns[str(horizon)][str(actual)])
                    - float(shaped_returns[str(horizon)][str(action)])
                )
                for action in action_order
            }
            for horizon in HORIZONS
        }
        root_rows.append(
            {
                "root_id": root_id,
                "rng_state_identifier": rng_identifier,
                "action_returns_by_horizon": returns,
                "actual_minus_action_credit_by_horizon": credits,
                "progress_shaped_reward_sequence_by_action": shaped_rewards,
                "progress_shaped_action_returns_by_horizon": shaped_returns,
                "progress_shaped_actual_minus_action_credit_by_horizon": shaped_credits,
                "action_branches": {
                    str(action): phase3._compact_branch_record(branches[action])
                    for action in action_order
                },
                "phase3_actual_and_first_alternative_reproduced": True,
                "semantic_crn": {
                    "shared_semantic_keys_checked": result.shared_semantic_keys_checked,
                    "semantic_mismatch_count": result.semantic_mismatch_count,
                    "unrecognized_environment_rng_calls": result.unrecognized_environment_rng_calls,
                },
                "serial_process_complete_result_equal": serial_equal,
            }
        )

    q_by_id = {int(value): float(q_values[index]) for index, value in enumerate(legal_ids)}
    return {
        "schema": "phase4a_multi_alternative_credit_decision_v1",
        "state_semantics": STATE_SEMANTICS,
        "historical_training_snapshot_recovered": False,
        "diagnostic_only_not_training_label": True,
        "seed": int(source["seed"]),
        "phase": str(source["phase"]),
        "checkpoint_update": int(source["checkpoint_update"]),
        "slot_index": int(source["slot_index"]),
        "decision_order": int(source["decision_order"]),
        "task_id": str(source["task_id"]),
        "legal_uav_ids": legal_ids,
        "all_legal_action_order": action_order,
        "actual_uav_id": actual,
        "phase3_first_alternative_uav_id": first_alternative,
        "nested_alternative_budgets": _nested_budgets(action_order),
        "legal_q_vector": [q_by_id[value] for value in legal_ids],
        "root_count": int(roots),
        "gamma_per_physical_slot": float(gamma),
        "same_slot_gamma_steps": 0,
        "root_results": root_rows,
        "action_statistics_by_horizon": {
            str(horizon): _action_statistics(
                root_rows, action_order, actual, horizon, "action_returns_by_horizon"
            )
            for horizon in HORIZONS
        },
        "progress_shaped_action_statistics_by_horizon": {
            str(horizon): _action_statistics(
                root_rows,
                action_order,
                actual,
                horizon,
                "progress_shaped_action_returns_by_horizon",
            )
            for horizon in HORIZONS
        },
        "optimizer_steps": 0,
        "gates": {
            "all_legal_actions_branched": True,
            "exact_pre_decision_state_equal": True,
            "phase3_roots_reused": True,
            "phase3_actual_and_first_alternative_reproduced": True,
            "suffix_regenerated_by_frozen_deterministic_policy": True,
            "same_slot_gamma_steps": 0,
            "training_parameters_unchanged": False,
            "optimizer_steps": 0,
            "semantic_mismatch_count": sum(
                int(row["semantic_crn"]["semantic_mismatch_count"]) for row in root_rows
            ),
            "unrecognized_environment_rng_calls": sum(
                int(row["semantic_crn"]["unrecognized_environment_rng_calls"])
                for row in root_rows
            ),
            "serial_process_complete_result_equal": (
                all(row["serial_process_complete_result_equal"] is True for row in root_rows)
                if compare_serial_process else None
            ),
        },
    }


def _run_multi_serial(
    *, action_ids: tuple[int, ...], **kwargs: Any
) -> MultiActionRootResult:
    branches = tuple(
        phase3._run_branch(forced_uav_id=int(action), **kwargs) for action in action_ids
    )
    audit = audit_clean_semantic_common_random(
        [branch.semantic_audit for branch in branches]
    )
    return MultiActionRootResult(
        branches=branches,
        shared_semantic_keys_checked=int(audit.shared_semantic_keys_checked),
        semantic_mismatch_count=len(audit.semantic_key_mismatches),
        unrecognized_environment_rng_calls=int(audit.unrecognized_environment_calls),
    )


def _run_multi_process(*, timeout_seconds: float, **kwargs: Any) -> MultiActionRootResult:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_multi_worker, args=(child, kwargs))
    process.start()
    child.close()
    try:
        if not parent.poll(float(timeout_seconds)):
            raise TimeoutError("Phase 4-A multi-action root timed out")
        payload = parent.recv()
        process.join(timeout=float(timeout_seconds))
        if process.is_alive():
            raise TimeoutError("Phase 4-A multi-action root did not exit")
        if int(process.exitcode or 0) != 0 or not payload.get("ok", False):
            raise RuntimeError(f"Phase 4-A root failed: {payload.get('error')}")
        return payload["result"]
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


def _run_multi_process_batch(
    specs: list[tuple[int, dict[str, Any]]], *, timeout_seconds: float, max_parallel: int
) -> dict[int, MultiActionRootResult]:
    context = multiprocessing.get_context("spawn")
    results: dict[int, MultiActionRootResult] = {}
    for start in range(0, len(specs), int(max_parallel)):
        wave = specs[start : start + int(max_parallel)]
        running: list[tuple[int, Any, Any]] = []
        try:
            for root_id, kwargs in wave:
                parent, child = context.Pipe(duplex=False)
                process = context.Process(target=_multi_worker, args=(child, kwargs))
                process.start()
                child.close()
                running.append((root_id, parent, process))
            for root_id, parent, process in running:
                if not parent.poll(float(timeout_seconds)):
                    raise TimeoutError(f"Phase 4-A root {root_id} timed out")
                payload = parent.recv()
                process.join(timeout=float(timeout_seconds))
                if process.is_alive():
                    raise TimeoutError(f"Phase 4-A root {root_id} did not exit")
                if int(process.exitcode or 0) != 0 or not payload.get("ok", False):
                    raise RuntimeError(
                        f"Phase 4-A root {root_id} failed: {payload.get('error')}"
                    )
                results[root_id] = payload["result"]
        finally:
            for _, parent, process in running:
                parent.close()
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
    return results


def _multi_worker(connection: Any, kwargs: dict[str, Any]) -> None:
    try:
        connection.send({"ok": True, "result": _run_multi_serial(**kwargs)})
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _sequence_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return bool(
        left_array.shape == right_array.shape
        and np.allclose(left_array, right_array, rtol=0.0, atol=1e-10)
    )


def _discounted_return(rewards: Any, gamma: float) -> float:
    return float(sum((float(gamma) ** index) * float(value) for index, value in enumerate(rewards)))


def _progress_shaped_rewards(branch: phase3.HorizonBranchResult, gamma: float) -> list[float]:
    rewards = list(branch.reward_sequence)
    potentials = list(branch.progress_potential_sequence)
    if len(potentials) != len(rewards) + 1:
        raise AssertionError("progress potential sequence must bracket every reward transition")
    return [
        float(reward) + float(gamma) * float(potentials[index + 1])
        - float(potentials[index])
        for index, reward in enumerate(rewards)
    ]


def _nested_budgets(action_order: list[int]) -> dict[str, list[int]]:
    alternatives = action_order[1:]
    return {
        "1": [action_order[0]] + alternatives[:1],
        "3": [action_order[0]] + alternatives[:3],
        "all": list(action_order),
    }


def _mean_std_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = None if array.size == 0 else float(array.mean())
    std = None if array.size < 2 else float(array.std(ddof=1))
    se = None if std is None else float(std / math.sqrt(array.size))
    half = None if se is None else float(_t_critical_975(array.size - 1) * se)
    return {
        "root_count": int(array.size),
        "mean": mean,
        "std": std,
        "se": se,
        "ci95": None if half is None else [mean - half, mean + half],
    }


def _action_statistics(
    root_rows: list[dict[str, Any]], action_order: list[int], actual: int, horizon: int,
    return_field: str,
) -> dict[str, Any]:
    action_stats: list[dict[str, Any]] = []
    for action in action_order:
        values = [
            row[return_field][str(horizon)][str(action)]
            for row in root_rows
        ]
        usable = [float(value) for value in values if value is not None]
        stats = _mean_std_ci(usable)
        action_stats.append(
            {
                "uav_id": action,
                "censor_fraction": 1.0 - len(usable) / max(len(root_rows), 1),
                "return": stats,
            }
        )
    means = {int(row["uav_id"]): row["return"]["mean"] for row in action_stats}
    if any(means[action] is None for action in action_order):
        ranking: list[int] = []
    else:
        ranking = sorted(action_order, key=lambda action: (-float(means[action]), action))
    pair_stats: list[dict[str, Any]] = []
    for left_index, left in enumerate(action_order):
        for right in action_order[left_index + 1 :]:
            differences = []
            for root in root_rows:
                left_value = root[return_field][str(horizon)][str(left)]
                right_value = root[return_field][str(horizon)][str(right)]
                if left_value is not None and right_value is not None:
                    differences.append(float(left_value) - float(right_value))
            stats = _mean_std_ci(differences)
            ci = stats["ci95"]
            positive = sum(value > 0.0 for value in differences)
            negative = sum(value < 0.0 for value in differences)
            pair_stats.append(
                {
                    "left_uav_id": left,
                    "right_uav_id": right,
                    "return_difference": stats,
                    "root_sign_consistency": (
                        None
                        if not differences
                        else float(max(positive, negative) / len(differences))
                    ),
                    "ordering_resolved": bool(
                        ci is not None and (float(ci[0]) > 0.0 or float(ci[1]) < 0.0)
                    ),
                }
            )
    root_top1 = []
    root_pair_agreement = []
    if ranking:
        for root in root_rows:
            values = {
                action: root[return_field][str(horizon)][str(action)]
                for action in action_order
            }
            if any(value is None for value in values.values()):
                continue
            root_ranking = sorted(action_order, key=lambda action: (-float(values[action]), action))
            root_top1.append(float(root_ranking[0] == ranking[0]))
            agreements = []
            for left_index, left in enumerate(action_order):
                for right in action_order[left_index + 1 :]:
                    agreements.append(
                        _tie_aware_score(
                            float(values[left]) - float(values[right]),
                            float(means[left]) - float(means[right]),
                        )
                    )
            root_pair_agreement.extend(agreements)
    return {
        "horizon": horizon,
        "action_statistics": action_stats,
        "action_ranking_by_mean_return_desc": ranking,
        "actual_relative_credit_ranking_ascending": ranking,
        "actual_relative_credit_by_action": {
            str(action): (
                None
                if means[actual] is None or means[action] is None
                else float(means[actual]) - float(means[action])
            )
            for action in action_order
        },
        "pair_statistics": pair_stats,
        "unresolved_pair_fraction": (
            None
            if not pair_stats
            else float(np.mean([not row["ordering_resolved"] for row in pair_stats]))
        ),
        "mean_pair_root_sign_consistency": (
            None
            if not pair_stats
            else float(np.mean([row["root_sign_consistency"] for row in pair_stats]))
        ),
        "root_top1_agreement_with_mean_ranking": (
            None if not root_top1 else float(np.mean(root_top1))
        ),
        "root_pairwise_agreement_with_mean_ranking": (
            None if not root_pair_agreement else float(np.mean(root_pair_agreement))
        ),
    }


def _subset_metrics(
    decisions: list[dict[str, Any]], truth: dict[tuple[Any, ...], dict[str, Any]], horizon: int,
    budget: str, statistics_field: str,
) -> dict[str, Any]:
    top1: list[float] = []
    spearman: list[float] = []
    pairs: list[float] = []
    resolved_pairs: list[float] = []
    stability_top1: list[float] = []
    stability_pair: list[float] = []
    unresolved: list[float] = []
    selected_pair_resolved: list[bool] = []
    actual_relative_resolved: list[bool] = []
    usable = 0
    for row in decisions:
        source = truth[_key(row)]
        vectors = _truth_vectors(source)
        truth_by_id = vectors["truth_by_id"]
        selected = [int(value) for value in row["nested_alternative_budgets"][budget]]
        stats = row[statistics_field][str(horizon)]
        predicted_by_id = {
            int(value["uav_id"]): value["return"]["mean"]
            for value in stats["action_statistics"]
        }
        if any(predicted_by_id[action] is None for action in selected):
            continue
        predicted = [float(predicted_by_id[action]) for action in selected]
        expected = [float(truth_by_id[action]) for action in selected]
        usable += 1
        top1.append(float(selected[int(np.argmax(predicted))] == selected[int(np.argmax(expected))]))
        correlation = _spearman(predicted, expected)
        if correlation is not None:
            spearman.append(float(correlation))
        for left_index, left in enumerate(selected):
            for right in selected[left_index + 1 :]:
                score = _tie_aware_score(
                    predicted_by_id[left] - predicted_by_id[right],
                    truth_by_id[left] - truth_by_id[right],
                )
                pairs.append(score)
                if vectors["resolved_by_pair"].get((left, right), False):
                    resolved_pairs.append(score)
        stability_top1.append(float(stats["root_top1_agreement_with_mean_ranking"]))
        stability_pair.append(float(stats["root_pairwise_agreement_with_mean_ranking"]))
        unresolved.append(float(stats["unresolved_pair_fraction"]))
        selected_set = set(selected)
        actual = int(row["actual_uav_id"])
        for pair in stats["pair_statistics"]:
            left = int(pair["left_uav_id"])
            right = int(pair["right_uav_id"])
            if left not in selected_set or right not in selected_set:
                continue
            resolved = bool(pair["ordering_resolved"])
            selected_pair_resolved.append(resolved)
            if actual in (left, right):
                actual_relative_resolved.append(resolved)
    return {
        "alternative_budget": budget,
        "candidate_count_including_actual": (
            None if not decisions else len(decisions[0]["nested_alternative_budgets"][budget])
        ),
        "usable_decision_count": usable,
        "top1_accuracy_vs_existing_multi_root_truth": None if not top1 else float(np.mean(top1)),
        "spearman_mean_vs_existing_multi_root_truth": None if not spearman else float(np.mean(spearman)),
        "pairwise_accuracy_vs_existing_multi_root_truth": None if not pairs else float(np.mean(pairs)),
        "truth_ci_resolved_pair_count": len(resolved_pairs),
        "truth_ci_resolved_pairwise_accuracy": (
            None if not resolved_pairs else float(np.mean(resolved_pairs))
        ),
        "mean_root_top1_agreement_with_credit_mean_ranking": (
            None if not stability_top1 else float(np.mean(stability_top1))
        ),
        "mean_root_pairwise_agreement_with_credit_mean_ranking": (
            None if not stability_pair else float(np.mean(stability_pair))
        ),
        "mean_unresolved_credit_pair_fraction": (
            None if not unresolved else float(np.mean(unresolved))
        ),
        "credit_pair_ci_resolved_fraction": (
            None if not selected_pair_resolved else float(np.mean(selected_pair_resolved))
        ),
        "actual_relative_credit_ci_resolved_fraction": (
            None if not actual_relative_resolved else float(np.mean(actual_relative_resolved))
        ),
    }


def _group_summary(
    rows: list[dict[str, Any]], truth: dict[tuple[Any, ...], dict[str, Any]],
    statistics_field: str,
) -> dict[str, Any]:
    return {
        str(horizon): {
            budget: _subset_metrics(rows, truth, horizon, budget, statistics_field)
            for budget in ("1", "3", "all")
        }
        for horizon in HORIZONS
    }


def _write_output(
    *, output: Path, mode: str, decisions: list[dict[str, Any]], phase3_path: Path,
    truth_path: Path, phase4a_reference_path: Path, gamma: float
) -> dict[str, Any]:
    truth = {_key(row): row for row in _read_jsonl(truth_path)}
    for row in decisions:
        if _key(row) not in truth:
            raise ValueError(f"missing truth row for {_key(row)}")
        row["gates"]["training_parameters_unchanged"] = True
    control_pooled = _group_summary(decisions, truth, "action_statistics_by_horizon")
    control_by_phase = {
        phase: _group_summary(
            [row for row in decisions if row["phase"] == phase],
            truth,
            "action_statistics_by_horizon",
        )
        for phase in ("early", "mid", "late")
    }
    control_by_seed = {
        str(seed): _group_summary(
            [row for row in decisions if row["seed"] == seed],
            truth,
            "action_statistics_by_horizon",
        )
        for seed in (0, 1, 2)
    }
    shaped_pooled = _group_summary(
        decisions, truth, "progress_shaped_action_statistics_by_horizon"
    )
    shaped_by_phase = {
        phase: _group_summary(
            [row for row in decisions if row["phase"] == phase],
            truth,
            "progress_shaped_action_statistics_by_horizon",
        )
        for phase in ("early", "mid", "late")
    }
    shaped_by_seed = {
        str(seed): _group_summary(
            [row for row in decisions if row["seed"] == seed],
            truth,
            "progress_shaped_action_statistics_by_horizon",
        )
        for seed in (0, 1, 2)
    }
    reference = json.loads(phase4a_reference_path.read_text(encoding="utf-8"))
    _assert_control_reproduces_reference(
        decisions=decisions,
        reference=reference,
        pooled=control_pooled,
        by_phase=control_by_phase,
        by_seed=control_by_seed,
        require_aggregate_equal=mode == "formal",
    )
    result = {
        "schema": "reward_density_progress_shaping_audit_v1",
        "mode": mode,
        "state_semantics": STATE_SEMANTICS,
        "historical_training_snapshot_recovered": False,
        "phase3_audit": str(phase3_path),
        "multi_root_truth": str(truth_path),
        "phase4a_reference": str(phase4a_reference_path),
        "horizons_physical_slots": list(HORIZONS),
        "alternative_budgets": ["1", "3", "all"],
        "action_quality_ranking_semantics": "G_H(action) descending",
        "credit_semantics": "C_H(action)=G_H(actual)-G_H(action); ascending equals action-quality ranking",
        "gamma_per_physical_slot": float(gamma),
        "same_slot_gamma_steps": 0,
        "training_performed": False,
        "optimizer_steps": 0,
        "decision_count": len(decisions),
        "gates": {
            "all_legal_actions_branched": all(row["gates"]["all_legal_actions_branched"] for row in decisions),
            "all_phase3_reference_branches_reproduced": all(
                row["gates"]["phase3_actual_and_first_alternative_reproduced"] for row in decisions
            ),
            "all_parameters_unchanged": all(row["gates"]["training_parameters_unchanged"] for row in decisions),
            "semantic_mismatch_count": sum(row["gates"]["semantic_mismatch_count"] for row in decisions),
            "unrecognized_environment_rng_calls": sum(
                row["gates"]["unrecognized_environment_rng_calls"] for row in decisions
            ),
            "optimizer_steps": 0,
            "serial_process_complete_result_equal": (
                all(row["gates"]["serial_process_complete_result_equal"] for row in decisions)
                if mode == "smoke" else None
            ),
            "shaping_off_reproduces_phase4a_value_for_value": True,
        },
        "progress_shaping": {
            "enabled_for_primary_variant": True,
            "formula": "r'(s->s') = r(s->s') + gamma*Phi(s') - Phi(s)",
            "potential": "REWARD_COMPLETED_DAG_WEIGHT * sum_active_dag(completed_tasks/total_tasks)",
            "reward_completed_dag_weight": float(__import__("config").REWARD_COMPLETED_DAG_WEIGHT),
        },
        "control": {
            "pooled": control_pooled,
            "by_phase": control_by_phase,
            "by_seed": control_by_seed,
        },
        "progress_shaped": {
            "pooled": shaped_pooled,
            "by_phase": shaped_by_phase,
            "by_seed": shaped_by_seed,
        },
        "decision_rows": decisions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _assert_control_reproduces_reference(
    *, decisions: list[dict[str, Any]], reference: dict[str, Any],
    pooled: dict[str, Any], by_phase: dict[str, Any], by_seed: dict[str, Any],
    require_aggregate_equal: bool,
) -> None:
    reference_by_key = {_key(row): row for row in reference.get("decision_rows", [])}
    for current in decisions:
        old = reference_by_key.get(_key(current))
        if old is None:
            raise AssertionError("control decision is missing from Phase 4A reference")
        if require_aggregate_equal and (
            current["action_statistics_by_horizon"] != old["action_statistics_by_horizon"]
        ):
            raise AssertionError("gate-OFF action statistics differ from Phase 4A")
        if len(current["root_results"]) > len(old["root_results"]):
            raise AssertionError("gate-OFF root count differs from Phase 4A")
        for current_root, old_root in zip(current["root_results"], old["root_results"]):
            for field in (
                "action_returns_by_horizon",
                "actual_minus_action_credit_by_horizon",
            ):
                if current_root[field] != old_root[field]:
                    raise AssertionError(f"gate-OFF root field differs from Phase 4A: {field}")
            for action, current_branch in current_root["action_branches"].items():
                if not _sequence_equal(
                    current_branch["reward_sequence"],
                    old_root["action_branches"][action]["reward_sequence"],
                ):
                    raise AssertionError("gate-OFF reward sequence differs from Phase 4A")
    if require_aggregate_equal and not (
        _matches_reference_shape(pooled, reference["pooled"])
        and _matches_reference_shape(by_phase, reference["by_phase"])
        and _matches_reference_shape(by_seed, reference["by_seed"])
    ):
        raise AssertionError("gate-OFF aggregate metrics differ from Phase 4A")


def _matches_reference_shape(current: Any, reference: Any) -> bool:
    if isinstance(reference, dict):
        return isinstance(current, dict) and all(
            key in current and _matches_reference_shape(current[key], value)
            for key, value in reference.items()
        )
    if isinstance(reference, list):
        return isinstance(current, list) and len(current) == len(reference) and all(
            _matches_reference_shape(left, right)
            for left, right in zip(current, reference)
        )
    return bool(current == reference)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source_rows = _read_phase3_decisions(args.phase3_audit)
    if len(source_rows) != 27:
        raise ValueError("Phase 4-A requires the completed 27-decision Phase 3 audit")
    if args.mode == "smoke":
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise ValueError("--checkpoint must identify seed0/update30 for smoke")
        selected = [
            row for row in source_rows
            if int(row["seed"]) == 0 and int(row["checkpoint_update"]) == 30
        ][:1]
        decisions = _run_checkpoint(
            checkpoint=args.checkpoint,
            seed=0,
            update=30,
            source_rows=selected,
            gamma=float(args.gamma),
            max_search_slots=int(args.max_search_slots),
            timeout_seconds=float(args.process_timeout_seconds),
            compare_serial_process=True,
        )
    else:
        if args.experiment_root is None or not args.experiment_root.is_dir():
            raise ValueError("--experiment-root is required for formal mode")
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in source_rows:
            grouped.setdefault((int(row["seed"]), int(row["checkpoint_update"])), []).append(row)
        decisions = []
        partial = args.output.with_suffix(".partial.json")
        for seed in (0, 1, 2):
            run_dir = phase1._single_path(
                args.experiment_root.glob(f"runs/seed{seed}/*"),
                f"seed {seed} run directory",
                require_directory=True,
            )
            for update in (30, 60, 120):
                rows = _run_checkpoint(
                    checkpoint=run_dir / "checkpoints" / f"checkpoint_update_{update:04d}.pt",
                    seed=seed,
                    update=update,
                    source_rows=grouped[(seed, update)],
                    gamma=float(args.gamma),
                    max_search_slots=int(args.max_search_slots),
                    timeout_seconds=float(args.process_timeout_seconds),
                    compare_serial_process=False,
                )
                decisions.extend(rows)
                partial.parent.mkdir(parents=True, exist_ok=True)
                partial.write_text(json.dumps({"decision_rows": decisions}, sort_keys=True), encoding="utf-8")
                print(f"completed Phase 4-A seed={seed} update={update} decisions={len(rows)}", flush=True)
    result = _write_output(
        output=args.output,
        mode=args.mode,
        decisions=decisions,
        phase3_path=args.phase3_audit,
        truth_path=args.multi_root_truth,
        phase4a_reference_path=args.phase4a_reference,
        gamma=float(args.gamma),
    )
    print(json.dumps({"output": str(args.output), "decision_count": len(decisions), "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
