"""Calibrate Decision-Q same-slot E_pi Q(next) against exact-state CRN V(next)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.diagnose_decision_q_v2_ranking_crn as base
from marl_models.mappo.clean_counterfactual_oracle import (
    CleanCounterfactualDecisionTrace,
    capture_clean_counterfactual_baseline_trace,
    clone_post_movement_pre_offloading_env,
    run_clean_counterfactual_branches_serial,
)
from marl_models.mappo.clean_counterfactual_oracle_common_random import (
    run_clean_common_random_completion_serial,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--source-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--updates", type=int, nargs="+", default=(30, 60, 120))
    parser.add_argument("--decisions-per-checkpoint", type=int, default=3)
    parser.add_argument("--max-search-slots", type=int, default=300)
    parser.add_argument("--max-future-slots", type=int, default=100)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if int(args.max_future_slots) != 100:
        raise ValueError("same-slot bootstrap calibration requires H=100")
    source = [
        row
        for row in (
            json.loads(line)
            for line in args.source_decisions.read_text(encoding="utf-8").splitlines()
        )
        if bool(row.get("same_slot_primary", False))
        and int(row["seed"]) in set(args.seeds)
        and int(row["checkpoint_update"]) in set(args.updates)
    ]
    by_checkpoint: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in source:
        by_checkpoint.setdefault(
            (int(row["seed"]), int(row["checkpoint_update"])), []
        ).append(row)
    phase = {30: "early", 60: "mid", 120: "late"}
    args.output.mkdir(parents=True, exist_ok=True)
    partial = args.output / "same_slot_bootstrap_decisions.partial.jsonl"
    decisions: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_dir = base._single_path(
            args.experiment_root.glob(f"runs/seed{seed}/*"),
            f"seed {seed} run directory",
            require_directory=True,
        )
        for update in args.updates:
            selected = by_checkpoint.get((int(seed), int(update)), [])[
                : int(args.decisions_per_checkpoint)
            ]
            if not selected:
                continue
            rows = _checkpoint(
                checkpoint=run_dir / "checkpoints" / f"checkpoint_update_{update:04d}.pt",
                seed=int(seed),
                phase=phase[int(update)],
                update=int(update),
                source_rows=selected,
                max_search_slots=int(args.max_search_slots),
                max_future_slots=int(args.max_future_slots),
            )
            decisions.extend(rows)
            partial.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
                encoding="utf-8",
            )
            print(
                f"completed seed={seed} update={update} decisions={len(rows)} "
                f"budgets={[row['root_budget_final'] for row in rows]}",
                flush=True,
            )
    summary = _summary(decisions)
    decision_path = args.output / "same_slot_bootstrap_decisions.jsonl"
    summary_path = args.output / "same_slot_bootstrap_summary.json"
    decision_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "decision_count": len(decisions),
        "decision_jsonl": str(decision_path),
        "summary_json": str(summary_path),
        "gates": summary["gates"],
    }


def _checkpoint(
    *,
    checkpoint: Path,
    seed: int,
    phase: str,
    update: int,
    source_rows: list[dict[str, Any]],
    max_search_slots: int,
    max_future_slots: int,
) -> list[dict[str, Any]]:
    import torch
    from torch.distributions import Categorical

    payload = base._load_trusted_checkpoint(torch, checkpoint)
    if int(payload.get("update_step", -1)) != update:
        raise ValueError("checkpoint update mismatch")
    controls = base.checkpoint_experiment_controls(payload)
    device = torch.device("cpu")
    dims = base._module_dims_from_checkpoint(
        payload, argparse.Namespace(task_embedding_dim=None, hidden_dim=None)
    )
    modules = base._build_modules(dims=dims, experiment_controls=controls, device=device)
    base._load_module_state(modules, payload)
    base._set_eval_mode(modules)
    q_state = payload["extra_state"]["offloading_decision_q_credit"]
    first_weight = q_state["critic"]["net.0.weight"]
    q_critic = base.CleanDecisionCritic(
        input_dim=int(first_weight.shape[1]), hidden_dim=int(first_weight.shape[0])
    ).to(device)
    q_critic.load_state_dict(q_state["critic"])
    frozen = {
        "task_encoder": modules.hgnn,
        "movement_actor": modules.movement_actor,
        "offloading_actor": modules.offloading_actor,
        "main_critic": modules.critic,
        "decision_q_critic": q_critic,
    }
    for module in frozen.values():
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    before = base._clone_parameters(frozen)
    env = base.Env(
        completed_dag_weight=float(controls["completed_dag_weight"]),
        freeze_ue_mobility=bool(controls.get("freeze_ue_mobility", False)),
    )
    builder = base.CleanGraphBuilder()
    builder.reset()
    base._set_rng_state(payload.get("rng_state", {}))
    env.reset()
    source_by_slot = {int(row["slot_index"]): row for row in source_rows}
    if len(source_by_slot) != len(source_rows):
        raise ValueError("source has more than one decision per slot")
    audited: list[dict[str, Any]] = []
    try:
        for _ in range(max_search_slots):
            with torch.no_grad():
                prepared = base.prepare_slot_state(env=env, graph_builder=builder)
                encoded = base.encode_prepared_slot(
                    prepared_state=prepared,
                    env=env,
                    hgnn=modules.hgnn,
                    critic=modules.critic,
                    movement_actor=modules.movement_actor,
                    device=device,
                    detach_critic_hgnn=bool(controls.get("detach_critic_hgnn", False)),
                )
                movement = Categorical(logits=encoded.movement_logits).sample()
                env.apply_movement(
                    {
                        int(uav_id): int(movement[index].cpu().item())
                        for index, uav_id in enumerate(encoded.movement_observation.uav_ids)
                    }
                )
                snapshot = clone_post_movement_pre_offloading_env(env)
                rng_root = base.capture_clean_host_rng_state()
                ready = [
                    env.task_manager.get_task(task_id)
                    for task_id in prepared.frozen_ready_task_ids
                ]
                ready = [task for task in ready if task is not None and task.is_ready]
                assignments = modules.offloading_actor.act(
                    frozen_ready_tasks=ready,
                    task_embeddings=encoded.task_embeddings.detach(),
                    graph_snapshot=prepared.graph_snapshot,
                    task_manager=env.task_manager,
                    uavs=env.uavs,
                    executor=env.executor,
                    current_time_seconds=env.current_time_seconds,
                    uav_service_positions=env.uav_service_positions,
                    ue_service_positions=env.ue_service_positions,
                    ues=env.ues,
                    deterministic=False,
                )
            slot = int(prepared.slot_index)
            if slot in source_by_slot:
                source = source_by_slot[slot]
                records = tuple(modules.offloading_actor.latest_records)
                trace = capture_clean_counterfactual_baseline_trace(
                    decision_records=records, assignment_buffer=assignments
                )
                record = next(
                    row for row in records
                    if int(row.decision_order) == int(source["decision_order"])
                )
                state = base._decision_state(
                    record,
                    critic_global_context=encoded.critic_global_input,
                    episode_index=int(payload.get("episode", -1)) + 1,
                    slot_index=slot,
                )
                target_trace = next(
                    row for row in trace
                    if int(row.decision_order) == int(source["decision_order"])
                )
                target_rows, target_audit = base._reconstruct_forced_action_targets(
                    state=state,
                    snapshot=snapshot,
                    branch_rng_root=rng_root,
                    trace=trace,
                    target_trace=target_trace,
                    q_critic=q_critic,
                    q_state=q_state,
                    modules=modules,
                    prepared=prepared,
                    encoded=encoded,
                    max_future_slots=max_future_slots,
                )
                audited.append(
                    _one_decision(
                        source=source,
                        snapshot=snapshot,
                        rng_root=rng_root,
                        original_trace=trace,
                        target_rows=target_rows,
                        target_audit=target_audit,
                        modules=modules,
                        gamma=float(q_state["gamma"]),
                        max_future_slots=max_future_slots,
                    )
                )
            _, _, done, _ = env.commit_and_advance(assignment_buffer=assignments)
            if len(audited) == len(source_rows) or done:
                break
    finally:
        builder.close()
    if len(audited) != len(source_rows):
        raise RuntimeError("replay did not recover all same-slot source decisions")
    if not base._parameters_equal(before, base._clone_parameters(frozen)):
        raise AssertionError("same-slot calibration changed a frozen parameter")
    for row in audited:
        row["gates"]["training_parameters_unchanged"] = True
    return audited


def _nested_trace(
    original: tuple[Any, ...], reconstructed: list[dict[str, Any]]
) -> tuple[CleanCounterfactualDecisionTrace, ...]:
    rebuilt = {int(row["decision_order"]): row for row in reconstructed}
    result = []
    for row in original:
        data = rebuilt.get(int(row.decision_order))
        result.append(
            row
            if data is None
            else CleanCounterfactualDecisionTrace(
                task_id=str(data["task_id"]),
                decision_order=int(data["decision_order"]),
                selected_uav_id=int(data["selected_uav_id"]),
                candidate_uav_ids=tuple(int(value) for value in data["candidate_uav_ids"]),
                candidate_mask=tuple(bool(value) for value in data["candidate_mask"]),
            )
        )
    return tuple(result)


def _one_decision(
    *,
    source: dict[str, Any],
    snapshot: Any,
    rng_root: Any,
    original_trace: tuple[Any, ...],
    target_rows: list[dict[str, Any]],
    target_audit: dict[str, Any],
    modules: Any,
    gamma: float,
    max_future_slots: int,
) -> dict[str, Any]:
    legal_ids = [int(value) for value in source["legal_uav_ids"]]
    if [int(row["uav_id"]) for row in target_rows] != legal_ids:
        raise AssertionError("current action/target IDs are misaligned")
    contexts: dict[int, dict[str, Any]] = {}
    for target in target_rows:
        current_id = int(target["uav_id"])
        if target["target_censored"] or int(target["delta"]) != 0:
            raise AssertionError("source decision is not an exact same-slot transition")
        if not math.isclose(float(target["rho"]), 0.0, abs_tol=1e-12):
            raise AssertionError("same-slot transition has nonzero rho")
        if not math.isclose(
            float(target["bootstrap"]), float(target["expected_next_q"]), abs_tol=1e-8
        ):
            raise AssertionError("same-slot bootstrap consumed gamma")
        nested = _nested_trace(
            original_trace, list(target["reconstructed_trace_through_next"])
        )
        next_state = target["exact_next_state"]
        next_order = int(next_state["decision_order"])
        feasibility = run_clean_counterfactual_branches_serial(
            env_snapshot=snapshot,
            baseline_trace=nested,
            target_decision_order=next_order,
        )
        if not all(row.suffix_replay_feasible for row in feasibility):
            raise AssertionError("nested next-decision branch lost strict suffix")
        next_ids = [int(value) for value in target["next_legal_uav_ids"]]
        if [int(row.forced_uav_id) for row in feasibility] != next_ids:
            raise AssertionError("exact next-state legal IDs changed in nested branch")
        contexts[current_id] = {
            "target": target,
            "trace": nested,
            "next_order": next_order,
            "current_rewards": {
                int(row.forced_uav_id): float(row.current_slot_reward)
                for row in feasibility
            },
        }
    root_rows: list[dict[str, Any]] = []
    serial_checks = 0
    for start, stop in ((0, 8), (8, 16), (16, 32)):
        if start > 0 and not base._multi_root_needs_expansion(
            base._multi_root_decision_statistics(root_rows, legal_ids)
        ):
            break
        for root_id in range(start, stop):
            semantic_root, fingerprint = base._independent_semantic_root(
                base_rng_root=rng_root,
                seed=int(source["seed"]),
                update=int(source["checkpoint_update"]),
                slot_index=int(source["slot_index"]),
                decision_order=int(source["decision_order"]),
                root_id=root_id,
            )
            current_actions = []
            root_censored = False
            for current_id in legal_ids:
                context = contexts[current_id]
                kwargs = {
                    "env_snapshot": snapshot,
                    "baseline_trace": context["trace"],
                    "target_decision_order": context["next_order"],
                    "initial_rng_state": semantic_root,
                    "gamma": float(gamma),
                    "task_encoder": modules.hgnn,
                    "movement_actor": modules.movement_actor,
                    "offloading_actor": modules.offloading_actor,
                    "max_future_slots": int(max_future_slots),
                    "device": "cpu",
                }
                result = base._run_clean_common_random_completion_server_process(**kwargs)
                serial_equal = None
                if root_id == 0:
                    serial = run_clean_common_random_completion_serial(**kwargs)
                    serial_equal = bool(
                        serial.decision == result.decision and serial.audit == result.audit
                    )
                    if not serial_equal:
                        raise AssertionError("nested serial/process results differ")
                    serial_checks += 1
                if result.audit.semantic_key_mismatches:
                    raise AssertionError("nested semantic CRN mismatch")
                if result.audit.unrecognized_environment_calls:
                    raise AssertionError("nested unrecognized environment RNG")
                target = context["target"]
                probabilities = [
                    float(value) for value in target["next_legal_behavior_probabilities"]
                ]
                next_ids = [int(value) for value in target["next_legal_uav_ids"]]
                branches = []
                for branch in result.decision.branches:
                    next_id = int(branch.forced_uav_id)
                    value = (
                        None
                        if branch.common_discounted_return is None
                        else context["current_rewards"][next_id]
                        + float(gamma) * float(branch.common_discounted_return)
                    )
                    branches.append(
                        {
                            "next_action_id": next_id,
                            "continuation_return": value,
                            "censored": value is None,
                            "target_dag_completion_horizon": branch.target_dag_completion_horizon,
                            "target_dag_completion_time": branch.target_dag_completion_time,
                        }
                    )
                by_id = {int(row["next_action_id"]): row for row in branches}
                censored = any(by_id[next_id]["censored"] for next_id in next_ids)
                true_v = (
                    None
                    if censored
                    else float(
                        sum(
                            probability * float(by_id[next_id]["continuation_return"])
                            for next_id, probability in zip(next_ids, probabilities)
                        )
                    )
                )
                root_censored = root_censored or censored
                current_actions.append(
                    {
                        "uav_id": current_id,
                        "e_pi_q_next": float(target["expected_next_q"]),
                        "true_v_next": true_v,
                        "exact_next_state": target["exact_next_state"],
                        "next_legal_uav_ids": next_ids,
                        "next_legal_q_vector": list(target["next_legal_q_vector"]),
                        "next_behavior_probabilities": probabilities,
                        "next_q_spread": float(target["next_q_spread"]),
                        "next_action_crn_branches": branches,
                        "serial_process_equal": serial_equal,
                        "semantic_mismatch_count": 0,
                        "unrecognized_environment_rng_calls": 0,
                    }
                )
            root_rows.append(
                {
                    "root_id": root_id,
                    "rng_fingerprint": fingerprint,
                    "root_censored": root_censored,
                    "branches": [
                        {
                            "uav_id": int(row["uav_id"]),
                            "continuation_return": row["true_v_next"],
                        }
                        for row in current_actions
                    ],
                    "current_actions": current_actions,
                }
            )
    statistics = base._multi_root_decision_statistics(root_rows, legal_ids)
    current_truth = {
        int(row["uav_id"]): float(row["mean_continuation_return"])
        for row in source["final_statistics"]["action_statistics"]
    }
    target_by_id = {int(row["uav_id"]): row for row in target_rows}
    return {
        "schema": "decision_q_v2_same_slot_bootstrap_calibration_v1",
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "historical_training_snapshot_recovered": False,
        "seed": int(source["seed"]),
        "phase": str(source["phase"]),
        "checkpoint_update": int(source["checkpoint_update"]),
        "slot_index": int(source["slot_index"]),
        "task_id": str(source["task_id"]),
        "decision_order": int(source["decision_order"]),
        "legal_uav_ids": legal_ids,
        "current_action_rows": [
            {
                "current_action_id": current_id,
                "current_target": float(target_by_id[current_id]["target"]),
                "e_pi_q_next": float(target_by_id[current_id]["expected_next_q"]),
                "current_action_crn_truth": current_truth[current_id],
                "next_q_spread": float(target_by_id[current_id]["next_q_spread"]),
                "exact_next_state": target_by_id[current_id]["exact_next_state"],
                "next_legal_uav_ids": list(target_by_id[current_id]["next_legal_uav_ids"]),
                "next_legal_q_vector": list(target_by_id[current_id]["next_legal_q_vector"]),
                "next_behavior_probabilities": list(
                    target_by_id[current_id]["next_legal_behavior_probabilities"]
                ),
                "true_v_next_mean": next(
                    float(row["mean_continuation_return"])
                    for row in statistics["action_statistics"]
                    if int(row["uav_id"]) == current_id
                ),
            }
            for current_id in legal_ids
        ],
        "root_budget_final": len(root_rows),
        "root_results": root_rows,
        "true_v_next_statistics": statistics,
        "optimizer_steps": 0,
        "gates": {
            "same_slot_delta_zero_rho_zero": True,
            "same_slot_no_extra_gamma": True,
            "exact_next_state_from_forced_current_action_prefix": True,
            "next_state_q_probability_candidate_ids_aligned": True,
            "nested_strict_suffix_feasible_all": True,
            "single_post_movement_pre_offloading_root": True,
            "serial_process_equal": serial_checks == len(legal_ids),
            "semantic_mismatch_count": int(target_audit["semantic_mismatch_count"]),
            "unrecognized_environment_rng_calls": int(
                target_audit["unrecognized_environment_rng_calls"]
            ),
            "policy_and_critics_frozen": True,
        },
    }


def _relation(left: list[float], right: list[float]) -> dict[str, Any]:
    pair = []
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            pair.append(
                base._ordering_score(
                    left[first] - left[second], right[first] - right[second]
                )
            )
    return {
        "spearman": base._spearman(left, right),
        "pairwise_ordering_accuracy": base._mean_or_none(pair),
    }


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bootstrap_relations = []
    current_relations = []
    errors = []
    signed = []
    spreads = []
    decision_mae = []
    decision_spread = []
    configured = 0
    usable = 0
    for row in rows:
        actions = row["current_action_rows"]
        bootstrap = [float(item["e_pi_q_next"]) for item in actions]
        true_v = [float(item["true_v_next_mean"]) for item in actions]
        current_target = [float(item["current_target"]) for item in actions]
        current_truth = [float(item["current_action_crn_truth"]) for item in actions]
        bootstrap_relations.append(_relation(bootstrap, true_v))
        current_relations.append(_relation(current_target, current_truth))
        local_errors = [abs(left - right) for left, right in zip(bootstrap, true_v)]
        errors.extend(local_errors)
        signed.extend(left - right for left, right in zip(bootstrap, true_v))
        local_spreads = [float(item["next_q_spread"]) for item in actions]
        spreads.extend(local_spreads)
        decision_mae.append(float(np.mean(local_errors)))
        decision_spread.append(float(np.mean(local_spreads)))
        configured += int(row["root_budget_final"])
        usable += int(row["true_v_next_statistics"]["usable_root_count"])
    return {
        "decision_count": len(rows),
        "configured_root_count": configured,
        "usable_root_count": usable,
        "censor_fraction": None if configured == 0 else 1.0 - usable / configured,
        "e_pi_q_next_vs_true_v_next": {
            "spearman_mean": base._mean_or_none(
                [item["spearman"] for item in bootstrap_relations if item["spearman"] is not None]
            ),
            "pairwise_ordering_accuracy": base._mean_or_none(
                [item["pairwise_ordering_accuracy"] for item in bootstrap_relations]
            ),
            "mae": base._mean_or_none(errors),
            "rmse": None if not errors else float(math.sqrt(np.mean(np.square(errors)))),
            "mean_signed_error": base._mean_or_none(signed),
        },
        "current_target_vs_current_action_crn_truth": {
            "spearman_mean": base._mean_or_none(
                [item["spearman"] for item in current_relations if item["spearman"] is not None]
            ),
            "pairwise_ordering_accuracy": base._mean_or_none(
                [item["pairwise_ordering_accuracy"] for item in current_relations]
            ),
        },
        "next_q_spread_mean": base._mean_or_none(spreads),
        "absolute_error_vs_next_q_spread_pearson": _pearson(errors, spreads),
        "decision_mae_vs_next_q_spread_pearson": _pearson(decision_mae, decision_spread),
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "decision_q_v2_same_slot_bootstrap_calibration_summary_v1",
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "historical_training_snapshot_recovered": False,
        "decision_count": len(rows),
        "pooled": _group(rows),
        "by_phase": {
            phase: _group([row for row in rows if row["phase"] == phase])
            for phase in ("early", "mid", "late")
        },
        "by_seed": {
            str(seed): _group([row for row in rows if int(row["seed"]) == seed])
            for seed in (0, 1, 2)
        },
        "by_phase_seed": {
            phase: {
                str(seed): _group(
                    [
                        row for row in rows
                        if row["phase"] == phase and int(row["seed"]) == seed
                    ]
                )
                for seed in (0, 1, 2)
            }
            for phase in ("early", "mid", "late")
        },
        "gates": {
            "all_decision_gates": all(
                all(
                    bool(value)
                    for key, value in row["gates"].items()
                    if key not in (
                        "semantic_mismatch_count",
                        "unrecognized_environment_rng_calls",
                    )
                )
                and int(row["gates"]["semantic_mismatch_count"]) == 0
                and int(row["gates"]["unrecognized_environment_rng_calls"]) == 0
                for row in rows
            ),
            "all_training_parameters_unchanged": all(
                row["gates"].get("training_parameters_unchanged", False) for row in rows
            ),
            "optimizer_steps_total": sum(int(row["optimizer_steps"]) for row in rows),
            "semantic_mismatch_count": sum(
                int(row["gates"]["semantic_mismatch_count"]) for row in rows
            ),
            "unrecognized_environment_rng_calls": sum(
                int(row["gates"]["unrecognized_environment_rng_calls"]) for row in rows
            ),
        },
    }


def main() -> int:
    result = run(_parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
