"""Decision-Q v2 ranking audit on replay-generated on-policy decision states.

This is a diagnostic-only entrypoint.  It restores a frozen checkpoint at the
project's new-episode resume boundary, captures a post-movement/pre-offloading
snapshot, and reuses Scheme-B2 exact-state semantic-CRN branches.  Branch
outcomes never enter a training target or optimizer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.assignment import (
    CleanAssignmentBuffer,
    TemporaryReservationState,
    build_offloading_candidate_components,
)
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_counterfactual_oracle import (
    capture_clean_counterfactual_baseline_trace,
    clone_post_movement_pre_offloading_env,
    run_clean_counterfactual_branches_serial,
)
from marl_models.mappo.clean_counterfactual_oracle_common_random import (
    CleanSemanticCommonRandom,
    audit_clean_semantic_common_random,
    run_clean_common_random_completion_process,
    run_clean_common_random_completion_serial,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import (
    CleanHostRngState,
    capture_clean_host_rng_state,
)
from marl_models.mappo.clean_decision_transitions import CleanDecisionState
from marl_models.mappo.clean_decision_transitions import CleanDecisionTransitionTracker
from marl_models.mappo.clean_offloading_decision_credit import decision_state_key
from marl_models.mappo.clean_offloading_decision_q_credit import (
    encode_decision_candidate_rows,
    expected_behavior_q,
)
from marl_models.mappo.clean_ppo import CleanDecisionCritic
from marl_models.mappo.clean_slot_orchestrator import (
    encode_prepared_slot,
    prepare_slot_state,
)
from marl_models.mappo.clean_trainer import _set_rng_state
from scripts.eval_clean_mainline import (
    _build_modules,
    _load_module_state,
    _load_trusted_checkpoint,
    _module_dims_from_checkpoint,
    _set_eval_mode,
)
from scripts.train_clean_mainline import checkpoint_experiment_controls


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke/audit frozen Decision-Q rankings with Scheme-B2 CRN."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "root-cause", "multi-root", "multi-root-summary"),
        default="smoke",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--max-search-slots", type=int, default=200)
    parser.add_argument("--max-future-slots", type=int, default=2)
    parser.add_argument("--min-legal-actions", type=int, default=3)
    parser.add_argument("--decisions-per-checkpoint", type=int, default=3)
    parser.add_argument("--source-decisions", type=Path)
    parser.add_argument("--multi-root-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.distributions import Categorical

    if bool(config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES):
        raise ValueError("Scheme-B2 ranking audit requires KaHyPar OFF")
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in smoke mode")
    payload = _load_trusted_checkpoint(torch, args.checkpoint)
    if payload.get("resume_semantics") != "restart_from_new_episode_only":
        raise ValueError("checkpoint does not use the expected new-episode resume semantics")
    if int(payload.get("update_step", -1)) != 30:
        raise ValueError("the minimal smoke is restricted to update 30")
    controls = checkpoint_experiment_controls(payload)
    if not bool(controls.get("offloading_decision_q_credit", False)):
        raise ValueError("checkpoint does not enable Decision-Q credit")
    if str(controls.get("task_encoder", "hgnn")) != "mlp":
        raise ValueError("frozen Scheme-B2 infrastructure requires an MLP checkpoint")

    device = torch.device("cpu")
    dims = _module_dims_from_checkpoint(
        payload, argparse.Namespace(task_embedding_dim=None, hidden_dim=None)
    )
    modules = _build_modules(dims=dims, experiment_controls=controls, device=device)
    _load_module_state(modules, payload)
    _set_eval_mode(modules)

    q_state = payload.get("extra_state", {}).get("offloading_decision_q_credit")
    if q_state is None:
        raise ValueError("checkpoint is missing the Decision-Q critic state")
    first_weight = q_state["critic"]["net.0.weight"]
    q_critic = CleanDecisionCritic(
        input_dim=int(first_weight.shape[1]), hidden_dim=int(first_weight.shape[0])
    ).to(device)
    q_critic.load_state_dict(q_state["critic"])
    q_critic.eval()

    frozen_modules = {
        "task_encoder": modules.hgnn,
        "movement_actor": modules.movement_actor,
        "offloading_actor": modules.offloading_actor,
        "main_critic": modules.critic,
        "decision_q_critic": q_critic,
    }
    for module in frozen_modules.values():
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    parameters_before = _clone_parameters(frozen_modules)

    env = Env(
        completed_dag_weight=float(controls["completed_dag_weight"]),
        freeze_ue_mobility=bool(controls.get("freeze_ue_mobility", False)),
    )
    builder = CleanGraphBuilder()
    builder.reset()
    _set_rng_state(payload.get("rng_state", {}))
    env.reset()
    replay_episode = int(payload.get("episode", -1)) + 1
    selected: dict[str, Any] | None = None
    try:
        for _ in range(int(args.max_search_slots)):
            with torch.no_grad():
                prepared = prepare_slot_state(env=env, graph_builder=builder)
                encoded = encode_prepared_slot(
                    prepared_state=prepared,
                    env=env,
                    hgnn=modules.hgnn,
                    critic=modules.critic,
                    movement_actor=modules.movement_actor,
                    device=device,
                    detach_critic_hgnn=bool(controls.get("detach_critic_hgnn", False)),
                )
                movement_dist = Categorical(logits=encoded.movement_logits)
                movement = movement_dist.sample()
                env.apply_movement(
                    {
                        int(uav_id): int(movement[index].cpu().item())
                        for index, uav_id in enumerate(
                            encoded.movement_observation.uav_ids
                        )
                    }
                )
                snapshot = clone_post_movement_pre_offloading_env(env)
                branch_rng_root = capture_clean_host_rng_state()
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
            records = tuple(modules.offloading_actor.latest_records)
            trace = capture_clean_counterfactual_baseline_trace(
                decision_records=records,
                assignment_buffer=assignments,
            )
            record_by_order = {int(record.decision_order): record for record in records}
            trace_by_order = {int(row.decision_order): row for row in trace}
            ordered = sorted(trace_by_order)
            for position, decision_order in enumerate(ordered[:-1]):
                target_trace = trace_by_order[decision_order]
                if len(target_trace.legal_uav_ids) < int(args.min_legal_actions):
                    continue
                feasibility = run_clean_counterfactual_branches_serial(
                    env_snapshot=snapshot,
                    baseline_trace=trace,
                    target_decision_order=decision_order,
                )
                if not all(row.suffix_replay_feasible for row in feasibility):
                    continue
                next_order = ordered[position + 1]
                state = _decision_state(
                    record_by_order[decision_order],
                    critic_global_context=encoded.critic_global_input,
                    episode_index=replay_episode,
                    slot_index=int(prepared.slot_index),
                )
                next_state = _decision_state(
                    record_by_order[next_order],
                    critic_global_context=encoded.critic_global_input,
                    episode_index=replay_episode,
                    slot_index=int(prepared.slot_index),
                )
                selected = {
                    "prepared": prepared,
                    "snapshot": snapshot,
                    "branch_rng_root": branch_rng_root,
                    "assignments": assignments,
                    "trace": trace,
                    "target_trace": target_trace,
                    "state": state,
                    "next_state": next_state,
                    "feasibility": feasibility,
                }
                break
            if selected is not None:
                break
            _, _, done, _ = env.commit_and_advance(assignment_buffer=assignments)
            if done:
                raise RuntimeError("replay episode ended before a smoke decision was found")
    finally:
        builder.close()
    if selected is None:
        raise RuntimeError("no strict-suffix-feasible multicandidate decision found")

    state = selected["state"]
    next_state = selected["next_state"]
    legal_indices, q_inputs = encode_decision_candidate_rows(state)
    next_legal_indices, next_q_inputs = encode_decision_candidate_rows(next_state)
    with torch.no_grad():
        q_values = q_critic(
            torch.as_tensor(q_inputs, dtype=torch.float32, device=device)
        ).cpu().numpy()
        next_q_values = q_critic(
            torch.as_tensor(next_q_inputs, dtype=torch.float32, device=device)
        ).cpu().numpy()
    legal_candidate_ids = [state.candidate_uav_ids[int(index)] for index in legal_indices]
    if tuple(legal_candidate_ids) != selected["target_trace"].legal_uav_ids:
        raise AssertionError("Q legal vector and Scheme-B2 candidate IDs are misaligned")
    bootstrap = expected_behavior_q(
        legal_indices=next_legal_indices,
        probabilities=next_state.old_masked_probabilities,
        q_values=next_q_values,
    )
    rho = 0.0
    delta = 0
    target = rho + (float(q_state["gamma"]) ** delta) * bootstrap

    kwargs = {
        "env_snapshot": selected["snapshot"],
        "baseline_trace": selected["trace"],
        "target_decision_order": selected["target_trace"].decision_order,
        "initial_rng_state": selected["branch_rng_root"],
        "gamma": float(q_state["gamma"]),
        "task_encoder": modules.hgnn,
        "movement_actor": modules.movement_actor,
        "offloading_actor": modules.offloading_actor,
        "max_future_slots": int(args.max_future_slots),
        "device": device,
    }
    serial = run_clean_common_random_completion_serial(**kwargs)
    parallel = run_clean_common_random_completion_process(**kwargs)
    if serial.decision != parallel.decision or serial.audit != parallel.audit:
        raise AssertionError("Scheme-B2 serial/process complete results differ")
    if serial.audit.semantic_key_mismatches:
        raise AssertionError("Scheme-B2 semantic RNG keys differ across branches")
    if serial.audit.unrecognized_environment_calls:
        raise AssertionError("Scheme-B2 observed an unrecognized environment RNG call")

    baseline_assignments = tuple(
        (int(row.decision_order), str(row.task_id), int(row.selected_uav_id))
        for row in selected["trace"]
    )
    exact_state_rows = []
    current_reward_by_uav = {}
    for branch in selected["feasibility"]:
        differing_orders = [
            baseline[0]
            for baseline, replayed in zip(
                baseline_assignments, branch.replayed_assignments
            )
            if baseline != replayed
        ]
        expected_differences = (
            []
            if int(branch.forced_uav_id) == int(branch.baseline_uav_id)
            else [int(branch.decision_order)]
        )
        if differing_orders != expected_differences:
            raise AssertionError("exact-state branch changed more than the target decision")
        exact_state_rows.append(
            {
                "forced_uav_id": int(branch.forced_uav_id),
                "differing_decision_orders": differing_orders,
                "strict_suffix_feasible": bool(branch.suffix_replay_feasible),
            }
        )
        current_reward_by_uav[int(branch.forced_uav_id)] = float(
            branch.current_slot_reward
        )

    gamma = float(q_state["gamma"])
    branch_rows = []
    for branch in serial.decision.branches:
        future_discounted = float(
            sum(
                (gamma**index) * reward
                for index, reward in enumerate(branch.reward_sequence)
            )
        )
        current_reward = current_reward_by_uav[int(branch.forced_uav_id)]
        branch_rows.append(
            {
                "uav_id": int(branch.forced_uav_id),
                "current_slot_reward": current_reward,
                "future_reward_sequence": list(branch.reward_sequence),
                "diagnostic_truncated_return": current_reward
                + gamma * future_discounted,
                "completion_scale_return": (
                    None
                    if branch.common_discounted_return is None
                    else current_reward + gamma * float(branch.common_discounted_return)
                ),
                "target_dag_completion_horizon": branch.target_dag_completion_horizon,
            }
        )

    parameters_unchanged = _parameters_equal(
        parameters_before, _clone_parameters(frozen_modules)
    )
    if not parameters_unchanged:
        raise AssertionError("a frozen training parameter changed during the smoke")
    if not bool(getattr(selected["snapshot"], "_prepared_slot_open", False)):
        raise AssertionError("branch root snapshot is not an open prepared slot")
    if int(getattr(selected["snapshot"], "_last_movement_action_count", 0)) != len(
        selected["snapshot"].uavs
    ):
        raise AssertionError("branch root snapshot is not post-movement")

    q_pairs = list(zip(legal_candidate_ids, (float(value) for value in q_values)))
    q_ranking = [
        int(uav_id)
        for uav_id, _ in sorted(q_pairs, key=lambda item: (-item[1], item[0]))
    ]
    report = {
        "schema": "decision_q_v2_ranking_crn_smoke_v1",
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "historical_training_snapshot_recovered": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": int(payload["update_step"]),
        "checkpoint_episode": int(payload["episode"]),
        "replay_episode": replay_episode,
        "optimizer_steps": 0,
        "policy_and_critics_frozen": True,
        "slot_index": int(selected["prepared"].slot_index),
        "decision": {
            "task_id": state.task_id,
            "decision_order": int(state.decision_order),
            "candidate_ids": list(state.candidate_uav_ids),
            "legal_candidate_ids": legal_candidate_ids,
            "legal_q_vector": [float(value) for value in q_values],
            "q_ranking": q_ranking,
            "q_top1": q_ranking[0],
            "q_spread": float(np.max(q_values) - np.min(q_values)),
            "rho": rho,
            "delta": delta,
            "bootstrap": float(bootstrap),
            "target": float(target),
            "same_slot_suffix_discounted": False,
            "branches": branch_rows,
        },
        "gates": {
            "exact_state_branch_correct": True,
            "strict_suffix_feasible_all": True,
            "serial_process_complete_result_equal": True,
            "semantic_mismatch_count": len(serial.audit.semantic_key_mismatches),
            "unrecognized_environment_rng_calls": int(
                serial.audit.unrecognized_environment_calls
            ),
            "shared_semantic_keys_checked": int(
                serial.audit.shared_semantic_keys_checked
            ),
            "q_vector_candidate_ids_aligned": True,
            "single_post_movement_pre_offloading_branch_root": True,
            "training_parameters_unchanged": parameters_unchanged,
        },
        "exact_state_branch_proof": exact_state_rows,
        "serial_decision": asdict(serial.decision),
        "serial_audit": asdict(serial.audit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.experiment_root is None:
        raise ValueError("--experiment-root is required in pilot mode")
    if int(args.max_future_slots) != 100:
        raise ValueError("the formal ranking pilot requires H=100")
    if int(args.decisions_per_checkpoint) != 3:
        raise ValueError("the formal ranking pilot is capped at 3 decisions/checkpoint")
    phases = (("early", 30), ("mid", 60), ("late", 120))
    root_cause = args.mode == "root-cause"
    decisions: list[dict[str, Any]] = []
    ev_rows: dict[tuple[int, int], dict[str, float]] = {}
    metrics_paths: dict[int, Path] = {}
    for seed in (0, 1, 2):
        run_dir = _single_path(
            args.experiment_root.glob(f"runs/seed{seed}/*"),
            f"seed {seed} run directory",
            require_directory=True,
        )
        metrics_path = run_dir / "train_metrics.jsonl"
        metrics_paths[seed] = metrics_path
        for phase, update in phases:
            checkpoint = run_dir / "checkpoints" / f"checkpoint_update_{update:04d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            ev_rows[(seed, update)] = _training_q_ev(metrics_path, update)
            rows = _audit_checkpoint(
                checkpoint=checkpoint,
                seed=seed,
                phase=phase,
                expected_update=update,
                decision_limit=int(args.decisions_per_checkpoint),
                max_search_slots=int(args.max_search_slots),
                max_future_slots=int(args.max_future_slots),
                min_legal_actions=int(args.min_legal_actions),
                include_forced_action_targets=root_cause,
            )
            decisions.extend(rows)
            print(
                f"completed seed={seed} phase={phase} update={update} "
                f"decisions={len(rows)}",
                flush=True,
            )
    summary = (
        _root_cause_summary(decisions, ev_rows, metrics_paths)
        if root_cause
        else _pilot_summary(decisions, ev_rows)
    )
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "root_cause" if root_cause else "pilot"
    decision_path = output_dir / f"{prefix}_decisions.jsonl"
    summary_path = output_dir / f"{prefix}_summary.json"
    decision_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    row = {
        "decision_jsonl": str(decision_path),
        "summary_json": str(summary_path),
        "decision_count": len(decisions),
        "summary": summary,
    }
    return row


def run_multi_root(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.experiment_root is None:
        raise ValueError("--experiment-root is required in multi-root mode")
    if args.source_decisions is None or not args.source_decisions.is_file():
        raise ValueError("--source-decisions must identify the root-cause JSONL")
    if int(args.max_future_slots) != 100:
        raise ValueError("multi-root diagnostic requires H=100")
    source_rows = [
        json.loads(line)
        for line in args.source_decisions.read_text(encoding="utf-8").splitlines()
    ]
    if len(source_rows) != 27:
        raise ValueError("multi-root diagnostic requires the existing 27 decisions")
    by_checkpoint: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in source_rows:
        by_checkpoint.setdefault(
            (int(row["seed"]), int(row["checkpoint_update"])), []
        ).append(row)
    phase_updates = (("early", 30), ("mid", 60), ("late", 120))
    selected_seeds = tuple(int(value) for value in args.multi_root_seeds)
    if any(seed not in (0, 1, 2) for seed in selected_seeds):
        raise ValueError("multi-root seeds must be selected from 0/1/2")
    args.output.mkdir(parents=True, exist_ok=True)
    partial_path = args.output / "multi_root_decisions.partial.jsonl"
    decisions: list[dict[str, Any]] = []
    for seed in selected_seeds:
        run_dir = _single_path(
            args.experiment_root.glob(f"runs/seed{seed}/*"),
            f"seed {seed} run directory",
            require_directory=True,
        )
        for phase, update in phase_updates:
            checkpoint = run_dir / "checkpoints" / f"checkpoint_update_{update:04d}.pt"
            rows = _multi_root_checkpoint(
                checkpoint=checkpoint,
                seed=seed,
                phase=phase,
                expected_update=update,
                source_rows=by_checkpoint[(seed, update)],
                max_search_slots=int(args.max_search_slots),
                max_future_slots=int(args.max_future_slots),
            )
            decisions.extend(rows)
            partial_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
                encoding="utf-8",
            )
            print(
                f"completed multi-root seed={seed} phase={phase} "
                f"budgets={[row['root_budget_final'] for row in rows]}",
                flush=True,
            )
    summary = _multi_root_summary(decisions)
    decision_path = args.output / "multi_root_decisions.jsonl"
    summary_path = args.output / "multi_root_summary.json"
    decision_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "decision_count": len(decisions),
        "decision_jsonl": str(decision_path),
        "summary_json": str(summary_path),
        "summary": summary,
    }


def run_multi_root_summary_only(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_decisions is None or not args.source_decisions.is_file():
        raise ValueError("--source-decisions must identify the 27-decision multi-root JSONL")
    decisions = [
        json.loads(line)
        for line in args.source_decisions.read_text(encoding="utf-8").splitlines()
    ]
    if len(decisions) != 27:
        raise ValueError("multi-root summary requires exactly 27 decisions")
    summary = _multi_root_summary(decisions)
    args.output.mkdir(parents=True, exist_ok=True)
    decision_path = args.output / "multi_root_decisions.jsonl"
    summary_path = args.output / "multi_root_summary.json"
    decision_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "decision_count": len(decisions),
        "decision_jsonl": str(decision_path),
        "summary_json": str(summary_path),
        "summary": summary,
    }


def _multi_root_checkpoint(
    *,
    checkpoint: Path,
    seed: int,
    phase: str,
    expected_update: int,
    source_rows: list[dict[str, Any]],
    max_search_slots: int,
    max_future_slots: int,
) -> list[dict[str, Any]]:
    import torch
    from torch.distributions import Categorical

    payload = _load_trusted_checkpoint(torch, checkpoint)
    if int(payload.get("update_step", -1)) != int(expected_update):
        raise ValueError("checkpoint update mismatch")
    controls = checkpoint_experiment_controls(payload)
    device = torch.device("cpu")
    dims = _module_dims_from_checkpoint(
        payload, argparse.Namespace(task_embedding_dim=None, hidden_dim=None)
    )
    modules = _build_modules(dims=dims, experiment_controls=controls, device=device)
    _load_module_state(modules, payload)
    _set_eval_mode(modules)
    q_state = payload.get("extra_state", {}).get("offloading_decision_q_credit")
    first_weight = q_state["critic"]["net.0.weight"]
    q_critic = CleanDecisionCritic(
        input_dim=int(first_weight.shape[1]), hidden_dim=int(first_weight.shape[0])
    ).to(device)
    q_critic.load_state_dict(q_state["critic"])
    frozen_modules = {
        "task_encoder": modules.hgnn,
        "movement_actor": modules.movement_actor,
        "offloading_actor": modules.offloading_actor,
        "main_critic": modules.critic,
        "decision_q_critic": q_critic,
    }
    for module in frozen_modules.values():
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    parameters_before = _clone_parameters(frozen_modules)
    env = Env(
        completed_dag_weight=float(controls["completed_dag_weight"]),
        freeze_ue_mobility=bool(controls.get("freeze_ue_mobility", False)),
    )
    builder = CleanGraphBuilder()
    builder.reset()
    _set_rng_state(payload.get("rng_state", {}))
    env.reset()
    source_by_slot = {int(row["slot_index"]): row for row in source_rows}
    if len(source_by_slot) != len(source_rows):
        raise ValueError("multi-root source has more than one decision per slot")
    audited: list[dict[str, Any]] = []
    try:
        for _ in range(max_search_slots):
            with torch.no_grad():
                prepared = prepare_slot_state(env=env, graph_builder=builder)
                encoded = encode_prepared_slot(
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
                        for index, uav_id in enumerate(
                            encoded.movement_observation.uav_ids
                        )
                    }
                )
                snapshot = clone_post_movement_pre_offloading_env(env)
                base_rng_root = capture_clean_host_rng_state()
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
            slot_index = int(prepared.slot_index)
            if slot_index in source_by_slot:
                source = source_by_slot[slot_index]
                records = tuple(modules.offloading_actor.latest_records)
                trace = capture_clean_counterfactual_baseline_trace(
                    decision_records=records, assignment_buffer=assignments
                )
                target_matches = [
                    row
                    for row in trace
                    if int(row.decision_order) == int(source["decision_order"])
                    and str(row.task_id) == str(source["task_id"])
                ]
                if len(target_matches) != 1:
                    raise AssertionError("multi-root replay did not recover source decision")
                target_trace = target_matches[0]
                feasibility = run_clean_counterfactual_branches_serial(
                    env_snapshot=snapshot,
                    baseline_trace=trace,
                    target_decision_order=int(target_trace.decision_order),
                )
                if not all(row.suffix_replay_feasible for row in feasibility):
                    raise AssertionError("multi-root source lost strict suffix feasibility")
                record = next(
                    row
                    for row in records
                    if int(row.decision_order) == int(source["decision_order"])
                )
                state = _decision_state(
                    record,
                    critic_global_context=encoded.critic_global_input,
                    episode_index=int(payload.get("episode", -1)) + 1,
                    slot_index=slot_index,
                )
                legal_indices, q_inputs = encode_decision_candidate_rows(state)
                legal_ids = [
                    int(state.candidate_uav_ids[int(index)]) for index in legal_indices
                ]
                with torch.no_grad():
                    q_values = q_critic(
                        torch.as_tensor(q_inputs, dtype=torch.float32)
                    ).cpu().numpy()
                if legal_ids != [int(value) for value in source["legal_uav_ids"]]:
                    raise AssertionError("multi-root legal candidate IDs changed")
                if not np.allclose(
                    q_values,
                    np.asarray(source["legal_q_vector"], dtype=np.float32),
                    rtol=0.0,
                    atol=1e-6,
                ):
                    raise AssertionError("multi-root Q vector changed")
                audited.append(
                    _multi_root_one_decision(
                        source=source,
                        snapshot=snapshot,
                        base_rng_root=base_rng_root,
                        trace=trace,
                        target_trace=target_trace,
                        feasibility=feasibility,
                        modules=modules,
                        gamma=float(q_state["gamma"]),
                        max_future_slots=max_future_slots,
                    )
                )
            _, _, done, _ = env.commit_and_advance(assignment_buffer=assignments)
            if len(audited) == len(source_rows):
                break
            if done:
                break
    finally:
        builder.close()
    if len(audited) != len(source_rows):
        raise RuntimeError("multi-root replay did not recover all source decisions")
    if not _parameters_equal(parameters_before, _clone_parameters(frozen_modules)):
        raise AssertionError("multi-root diagnostic changed a frozen parameter")
    for row in audited:
        row["gates"]["training_parameters_unchanged"] = True
    return audited


def _multi_root_one_decision(
    *,
    source: dict[str, Any],
    snapshot: Any,
    base_rng_root: CleanHostRngState,
    trace: Any,
    target_trace: Any,
    feasibility: Any,
    modules: Any,
    gamma: float,
    max_future_slots: int,
) -> dict[str, Any]:
    current_reward = {
        int(row.forced_uav_id): float(row.current_slot_reward) for row in feasibility
    }
    root_rows: list[dict[str, Any]] = []
    serial_process_roots: list[int] = []
    for start, stop in ((0, 8), (8, 16), (16, 32)):
        if start > 0 and not _multi_root_needs_expansion(
            _multi_root_decision_statistics(
                root_rows,
                [int(value) for value in source["legal_uav_ids"]],
            )
        ):
            break
        for root_id in range(start, stop):
            rng_root, fingerprint = _independent_semantic_root(
                base_rng_root=base_rng_root,
                seed=int(source["seed"]),
                update=int(source["checkpoint_update"]),
                slot_index=int(source["slot_index"]),
                decision_order=int(source["decision_order"]),
                root_id=root_id,
            )
            kwargs = {
                "env_snapshot": snapshot,
                "baseline_trace": trace,
                "target_decision_order": int(target_trace.decision_order),
                "initial_rng_state": rng_root,
                "gamma": float(gamma),
                "task_encoder": modules.hgnn,
                "movement_actor": modules.movement_actor,
                "offloading_actor": modules.offloading_actor,
                "max_future_slots": int(max_future_slots),
                "device": "cpu",
            }
            branch_result = _run_clean_common_random_completion_server_process(**kwargs)
            serial_equal = None
            if root_id == 0:
                serial_result = run_clean_common_random_completion_serial(**kwargs)
                serial_equal = bool(
                    serial_result.decision == branch_result.decision
                    and serial_result.audit == branch_result.audit
                )
                if not serial_equal:
                    raise AssertionError("multi-root serial/process results differ")
                serial_process_roots.append(root_id)
            if branch_result.audit.semantic_key_mismatches:
                raise AssertionError("multi-root semantic CRN mismatch")
            if branch_result.audit.unrecognized_environment_calls:
                raise AssertionError("multi-root unrecognized environment RNG")
            branches = []
            for branch in branch_result.decision.branches:
                uav_id = int(branch.forced_uav_id)
                future_discounted = float(
                    sum(
                        (float(gamma) ** index) * float(reward)
                        for index, reward in enumerate(branch.reward_sequence)
                    )
                )
                branches.append(
                    {
                        "uav_id": uav_id,
                        "continuation_return": (
                            None
                            if branch.common_discounted_return is None
                            else current_reward[uav_id]
                            + float(gamma) * float(branch.common_discounted_return)
                        ),
                        "h100_truncated_return": current_reward[uav_id]
                        + float(gamma) * future_discounted,
                        "target_dag_completion_horizon": branch.target_dag_completion_horizon,
                        "target_dag_completion_time": branch.target_dag_completion_time,
                        "censored": branch.target_dag_completion_horizon is None,
                    }
                )
            root_rows.append(
                {
                    "root_id": root_id,
                    "rng_fingerprint": fingerprint,
                    "branches": branches,
                    "root_censored": any(row["censored"] for row in branches),
                    "semantic_crn": {
                        "shared_semantic_keys_checked": int(
                            branch_result.audit.shared_semantic_keys_checked
                        ),
                        "semantic_mismatch_count": 0,
                        "unrecognized_environment_rng_calls": 0,
                    },
                    "serial_process_complete_result_equal": serial_equal,
                }
            )
    legal_ids = [int(value) for value in source["legal_uav_ids"]]
    checkpoints = {
        str(budget): _multi_root_decision_statistics(root_rows[:budget], legal_ids)
        for budget in (8, 16, 32)
        if len(root_rows) >= budget
    }
    row = {
        "schema": "decision_q_v2_multi_root_crn_decision_v1",
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "historical_training_snapshot_recovered": False,
        "seed": int(source["seed"]),
        "phase": str(source["phase"]),
        "checkpoint_update": int(source["checkpoint_update"]),
        "slot_index": int(source["slot_index"]),
        "task_id": str(source["task_id"]),
        "decision_order": int(source["decision_order"]),
        "legal_uav_ids": legal_ids,
        "legal_q_vector": list(source["legal_q_vector"]),
        "actor_legal_probabilities": list(source["actor_legal_probabilities"]),
        "forced_action_targets": list(source["forced_action_targets"]),
        "same_slot_primary": all(
            bool(item["same_slot_bootstrap_only"])
            for item in source["forced_action_targets"]
        ),
        "single_root_branches": list(source["branches"]),
        "root_budget_final": len(root_rows),
        "root_results": root_rows,
        "statistics_by_budget": checkpoints,
        "final_statistics": checkpoints[str(len(root_rows))],
        "optimizer_steps": 0,
        "gates": {
            "exact_state_branch_correct": True,
            "strict_suffix_feasible_all": True,
            "single_post_movement_pre_offloading_branch_root": True,
            "candidate_q_target_actor_ids_aligned": True,
            "serial_process_complete_result_equal": bool(serial_process_roots),
            "same_slot_gamma_semantics_unchanged": all(
                int(item["delta"]) != 0
                or math.isclose(
                    float(item["bootstrap"]), float(item["expected_next_q"])
                )
                for item in source["forced_action_targets"]
            ),
            "semantic_mismatch_count": 0,
            "unrecognized_environment_rng_calls": 0,
            "policy_and_critics_frozen": True,
        },
    }
    return row


def _run_clean_common_random_completion_server_process(**kwargs: Any) -> Any:
    """Use the existing process runner without repeated spawn imports on Linux."""
    import marl_models.mappo.clean_counterfactual_oracle_common_random as module

    original_get_context = module.multiprocessing.get_context
    if sys.platform != "win32":
        module.multiprocessing.get_context = lambda _: original_get_context("fork")
    try:
        return run_clean_common_random_completion_process(**kwargs)
    finally:
        module.multiprocessing.get_context = original_get_context


def _independent_semantic_root(
    *,
    base_rng_root: CleanHostRngState,
    seed: int,
    update: int,
    slot_index: int,
    decision_order: int,
    root_id: int,
) -> tuple[CleanHostRngState, dict[str, Any]]:
    sequence = np.random.SeedSequence(
        [20260831, int(seed), int(update), int(slot_index), int(decision_order), int(root_id)]
    )
    root_seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
    random_state = np.random.RandomState(root_seed).get_state()
    root = CleanHostRngState(
        python_state=base_rng_root.python_state,
        numpy_state=random_state,
    )
    return root, {
        "root_seed": root_seed,
        "numpy_generator": str(random_state[0]),
        "state_position": int(random_state[2]),
        "has_cached_gaussian": int(random_state[3]),
        "cached_gaussian": float(random_state[4]),
        "state_words_head": [int(value) for value in random_state[1][:8]],
    }


def _multi_root_decision_statistics(
    root_rows: list[dict[str, Any]], legal_ids: list[int]
) -> dict[str, Any]:
    usable = [row for row in root_rows if not row["root_censored"]]
    values_by_action = {
        uav_id: [
            float(next(item for item in root["branches"] if int(item["uav_id"]) == uav_id)["continuation_return"])
            for root in usable
        ]
        for uav_id in legal_ids
    }
    action_stats = []
    for uav_id in legal_ids:
        values = np.asarray(values_by_action[uav_id], dtype=np.float64)
        mean = None if values.size == 0 else float(values.mean())
        std = None if values.size < 2 else float(values.std(ddof=1))
        se = None if std is None else float(std / math.sqrt(values.size))
        half_width = None if se is None else _t_critical_975(values.size - 1) * se
        action_stats.append(
            {
                "uav_id": uav_id,
                "root_count": int(values.size),
                "mean_continuation_return": mean,
                "std": std,
                "se": se,
                "ci95": None if half_width is None else [mean - half_width, mean + half_width],
                "best_action_frequency": None,
            }
        )
    best_counts = {uav_id: 0 for uav_id in legal_ids}
    for root in usable:
        returns = {
            int(item["uav_id"]): float(item["continuation_return"])
            for item in root["branches"]
        }
        best = max(returns.values())
        best_ids = [uav_id for uav_id, value in returns.items() if value == best]
        for uav_id in best_ids:
            best_counts[uav_id] += 1.0 / len(best_ids)
    for item in action_stats:
        item["best_action_frequency"] = (
            None
            if not usable
            else float(best_counts[int(item["uav_id"])] / len(usable))
        )
    ranking = (
        None
        if not usable
        else [
            int(item["uav_id"])
            for item in sorted(
                action_stats,
                key=lambda item: (-float(item["mean_continuation_return"]), int(item["uav_id"])),
            )
        ]
    )
    pairs = []
    for first in range(len(legal_ids)):
        for second in range(first + 1, len(legal_ids)):
            left_id, right_id = legal_ids[first], legal_ids[second]
            differences = np.asarray(
                [
                    values_by_action[left_id][index] - values_by_action[right_id][index]
                    for index in range(len(usable))
                ],
                dtype=np.float64,
            )
            mean = None if differences.size == 0 else float(differences.mean())
            std = None if differences.size < 2 else float(differences.std(ddof=1))
            se = None if std is None else float(std / math.sqrt(differences.size))
            half_width = None if se is None else _t_critical_975(differences.size - 1) * se
            ci = None if half_width is None else [mean - half_width, mean + half_width]
            resolved = bool(ci is not None and (ci[0] > 0.0 or ci[1] < 0.0))
            pairs.append(
                {
                    "left_uav_id": left_id,
                    "right_uav_id": right_id,
                    "root_count": int(differences.size),
                    "mean_difference": mean,
                    "se": se,
                    "ci95": ci,
                    "empirical_left_win_fraction": (
                        None
                        if differences.size == 0
                        else float(np.mean(differences > 0.0))
                    ),
                    "truth_order_resolved": resolved,
                }
            )
    unresolved = sum(not item["truth_order_resolved"] for item in pairs)
    return {
        "configured_root_count": len(root_rows),
        "usable_root_count": len(usable),
        "root_censor_fraction": 1.0 - len(usable) / max(len(root_rows), 1),
        "action_statistics": action_stats,
        "mean_return_ranking": ranking,
        "mean_return_top1": None if ranking is None else ranking[0],
        "paired_action_differences": pairs,
        "unresolved_pair_fraction": unresolved / max(len(pairs), 1),
    }


def _multi_root_needs_expansion(statistics: dict[str, Any]) -> bool:
    configured = int(statistics["configured_root_count"])
    usable = int(statistics["usable_root_count"])
    if usable < max(6, configured // 2):
        return True
    best = statistics["mean_return_top1"]
    if best is None:
        return True
    best_pairs = [
        row
        for row in statistics["paired_action_differences"]
        if int(row["left_uav_id"]) == int(best)
        or int(row["right_uav_id"]) == int(best)
    ]
    return any(not row["truth_order_resolved"] for row in best_pairs) or float(
        statistics["unresolved_pair_fraction"]
    ) > 0.5


def _t_critical_975(degrees_of_freedom: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        31: 2.040,
    }
    return float(table.get(max(int(degrees_of_freedom), 1), 1.960))


def _multi_root_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    subsets = {
        "same_slot_primary": [row for row in decisions if row["same_slot_primary"]],
        "delta_positive_secondary": [
            row for row in decisions if not row["same_slot_primary"]
        ],
    }
    calibration: dict[str, Any] = {}
    for subset_name, subset_rows in subsets.items():
        phases: dict[str, Any] = {}
        for phase in ("early", "mid", "late"):
            phase_rows = [row for row in subset_rows if row["phase"] == phase]
            phases[phase] = {
                "pooled": _multi_root_calibration(phase_rows),
                "by_seed": {
                    str(seed): _multi_root_calibration(
                        [row for row in phase_rows if int(row["seed"]) == seed]
                    )
                    for seed in (0, 1, 2)
                },
            }
        calibration[subset_name] = {
            "decision_count": len(subset_rows),
            "pooled": _multi_root_calibration(subset_rows),
            "phases": phases,
        }
    gates = {
        "decision_count_is_27": len(decisions) == 27,
        "optimizer_steps_total": sum(int(row["optimizer_steps"]) for row in decisions),
        "all_training_parameters_unchanged": all(
            row["gates"].get("training_parameters_unchanged", False)
            for row in decisions
        ),
        "all_policy_and_critics_frozen": all(
            row["gates"]["policy_and_critics_frozen"] for row in decisions
        ),
        "all_exact_state_branch_correct": all(
            row["gates"]["exact_state_branch_correct"] for row in decisions
        ),
        "all_strict_suffix_feasible": all(
            row["gates"]["strict_suffix_feasible_all"] for row in decisions
        ),
        "all_single_branch_root": all(
            row["gates"]["single_post_movement_pre_offloading_branch_root"]
            for row in decisions
        ),
        "all_candidate_ids_aligned": all(
            row["gates"]["candidate_q_target_actor_ids_aligned"]
            for row in decisions
        ),
        "all_serial_process_equal": all(
            row["gates"]["serial_process_complete_result_equal"]
            for row in decisions
        ),
        "all_same_slot_gamma_semantics_unchanged": all(
            row["gates"]["same_slot_gamma_semantics_unchanged"]
            for row in decisions
        ),
        "semantic_mismatch_count": sum(
            int(row["gates"]["semantic_mismatch_count"]) for row in decisions
        ),
        "unrecognized_environment_rng_calls": sum(
            int(row["gates"]["unrecognized_environment_rng_calls"])
            for row in decisions
        ),
    }
    return {
        "schema": "decision_q_v2_multi_root_crn_summary_v1",
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "historical_training_snapshot_recovered": False,
        "diagnostic_only_not_training_labels": True,
        "adaptive_root_cap": 32,
        "decision_count": len(decisions),
        "same_slot_primary_count": len(subsets["same_slot_primary"]),
        "delta_positive_secondary_count": len(subsets["delta_positive_secondary"]),
        "final_root_budget_distribution": {
            str(budget): sum(
                int(row["root_budget_final"]) == budget for row in decisions
            )
            for budget in (8, 16, 32)
        },
        "calibration": calibration,
        "root_budget_stability": _multi_root_stability(decisions),
        "single_root_vs_multi_root": _single_root_vs_multi_root(decisions),
        "gates": gates,
    }


def _multi_root_calibration(
    rows: list[dict[str, Any]], *, budget: int | None = None
) -> dict[str, Any]:
    usable: list[tuple[dict[str, Any], dict[str, Any]]] = []
    configured_roots = 0
    usable_roots = 0
    for row in rows:
        statistics = (
            row["final_statistics"]
            if budget is None
            else row["statistics_by_budget"].get(str(budget))
        )
        if statistics is None:
            continue
        configured_roots += int(statistics["configured_root_count"])
        usable_roots += int(statistics["usable_root_count"])
        if statistics["mean_return_ranking"] is not None:
            usable.append((row, statistics))
    return {
        "selected_decision_count": len(rows),
        "usable_decision_count": len(usable),
        "configured_root_count": configured_roots,
        "usable_root_count": usable_roots,
        "censor_fraction": (
            None
            if configured_roots == 0
            else 1.0 - usable_roots / configured_roots
        ),
        "unresolved_pair_fraction": _mean_or_none(
            [float(statistics["unresolved_pair_fraction"]) for _, statistics in usable]
        ),
        "target_vs_expected_truth": _expected_truth_relation(usable, "target"),
        "q_vs_expected_truth": _expected_truth_relation(usable, "q"),
        "actor_vs_expected_truth": _expected_truth_relation(usable, "actor"),
        "q_vs_training_target": _fixed_vector_relation(
            [row for row, _ in usable], "q", "target"
        ),
    }


def _expected_truth_relation(
    usable: list[tuple[dict[str, Any], dict[str, Any]]], predictor: str
) -> dict[str, Any]:
    top1: list[float] = []
    spearman: list[float] = []
    pair_scores: list[float] = []
    resolved_scores: list[float] = []
    total_pairs = 0
    resolved_pairs = 0
    for row, statistics in usable:
        ids = [int(value) for value in row["legal_uav_ids"]]
        predicted = _multi_root_vector(row, predictor)
        truth_by_id = {
            int(item["uav_id"]): float(item["mean_continuation_return"])
            for item in statistics["action_statistics"]
        }
        truth = [truth_by_id[uav_id] for uav_id in ids]
        top1.append(float(ids[int(np.argmax(predicted))] == int(statistics["mean_return_top1"])))
        value = _spearman(predicted, truth)
        if value is not None:
            spearman.append(float(value))
        for pair in statistics["paired_action_differences"]:
            left = ids.index(int(pair["left_uav_id"]))
            right = ids.index(int(pair["right_uav_id"]))
            score = _ordering_score(
                predicted[left] - predicted[right], float(pair["mean_difference"])
            )
            pair_scores.append(score)
            total_pairs += 1
            if bool(pair["truth_order_resolved"]):
                resolved_scores.append(score)
                resolved_pairs += 1
    return {
        "top1_accuracy": _mean_or_none(top1),
        "spearman_mean": _mean_or_none(spearman),
        "spearman_decision_count": len(spearman),
        "pairwise_ordering_accuracy": _mean_or_none(pair_scores),
        "pair_count": total_pairs,
        "resolved_pair_only_accuracy": _mean_or_none(resolved_scores),
        "resolved_pair_count": resolved_pairs,
    }


def _fixed_vector_relation(
    rows: list[dict[str, Any]], left_name: str, right_name: str
) -> dict[str, Any]:
    top1: list[float] = []
    spearman: list[float] = []
    pair_scores: list[float] = []
    for row in rows:
        left = _multi_root_vector(row, left_name)
        right = _multi_root_vector(row, right_name)
        top1.append(float(int(np.argmax(left)) == int(np.argmax(right))))
        value = _spearman(left, right)
        if value is not None:
            spearman.append(float(value))
        for first in range(len(left)):
            for second in range(first + 1, len(left)):
                pair_scores.append(
                    _ordering_score(
                        left[first] - left[second], right[first] - right[second]
                    )
                )
    return {
        "top1_accuracy": _mean_or_none(top1),
        "spearman_mean": _mean_or_none(spearman),
        "spearman_decision_count": len(spearman),
        "pairwise_ordering_accuracy": _mean_or_none(pair_scores),
        "pair_count": len(pair_scores),
        "resolved_pair_only_accuracy": _mean_or_none(pair_scores),
        "resolved_pair_count": len(pair_scores),
    }


def _multi_root_vector(row: dict[str, Any], name: str) -> list[float]:
    ids = [int(value) for value in row["legal_uav_ids"]]
    if name == "q":
        return [float(value) for value in row["legal_q_vector"]]
    if name == "actor":
        return [float(value) for value in row["actor_legal_probabilities"]]
    if name == "target":
        by_id = {
            int(item["uav_id"]): float(item["target"])
            for item in row["forced_action_targets"]
        }
        return [by_id[uav_id] for uav_id in ids]
    raise ValueError(f"unknown multi-root vector: {name}")


def _ordering_score(predicted_difference: float, truth_difference: float) -> float:
    if predicted_difference == 0.0 or truth_difference == 0.0:
        return 0.5
    return float((predicted_difference > 0.0) == (truth_difference > 0.0))


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _multi_root_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    budgets: dict[str, Any] = {}
    for budget in (8, 16, 32):
        available = [row for row in rows if str(budget) in row["statistics_by_budget"]]
        budgets[str(budget)] = {
            "decision_count": len(available),
            "same_slot_primary": _multi_root_calibration(
                [row for row in available if row["same_slot_primary"]], budget=budget
            ),
            "delta_positive_secondary": _multi_root_calibration(
                [row for row in available if not row["same_slot_primary"]], budget=budget
            ),
        }
    flips: dict[str, Any] = {}
    for earlier, later in ((8, 16), (16, 32)):
        common = [
            row
            for row in rows
            if str(earlier) in row["statistics_by_budget"]
            and str(later) in row["statistics_by_budget"]
        ]
        changed = sum(
            row["statistics_by_budget"][str(earlier)]["mean_return_top1"]
            != row["statistics_by_budget"][str(later)]["mean_return_top1"]
            for row in common
        )
        flips[f"{earlier}_to_{later}"] = {
            "decision_count": len(common),
            "best_action_flip_count": int(changed),
            "best_action_flip_fraction": None if not common else float(changed / len(common)),
        }
    return {"budgets": budgets, "best_action_flips": flips}


def _single_root_vs_multi_root(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        top1: list[float] = []
        spearman: list[float] = []
        pairwise: list[float] = []
        target_single: list[tuple[list[float], list[float]]] = []
        target_multi: list[tuple[list[float], list[float]]] = []
        q_single: list[tuple[list[float], list[float]]] = []
        q_multi: list[tuple[list[float], list[float]]] = []
        for row in selected:
            ids = [int(value) for value in row["legal_uav_ids"]]
            single_by_id = {
                int(item["uav_id"]): item["long_term_return"]
                for item in row["single_root_branches"]
            }
            if any(single_by_id.get(uav_id) is None for uav_id in ids):
                continue
            single = [float(single_by_id[uav_id]) for uav_id in ids]
            mean_by_id = {
                int(item["uav_id"]): item["mean_continuation_return"]
                for item in row["final_statistics"]["action_statistics"]
            }
            if any(mean_by_id.get(uav_id) is None for uav_id in ids):
                continue
            multi = [float(mean_by_id[uav_id]) for uav_id in ids]
            target = _multi_root_vector(row, "target")
            q_values = _multi_root_vector(row, "q")
            top1.append(float(int(np.argmax(single)) == int(np.argmax(multi))))
            value = _spearman(single, multi)
            if value is not None:
                spearman.append(float(value))
            for first in range(len(ids)):
                for second in range(first + 1, len(ids)):
                    pairwise.append(_ordering_score(
                        single[first] - single[second], multi[first] - multi[second]
                    ))
            target_single.append((target, single))
            target_multi.append((target, multi))
            q_single.append((q_values, single))
            q_multi.append((q_values, multi))
        return {
            "usable_decision_count": len(top1),
            "single_vs_multi_top1_agreement": _mean_or_none(top1),
            "single_vs_multi_spearman_mean": _mean_or_none(spearman),
            "single_vs_multi_pairwise_agreement": _mean_or_none(pairwise),
            "target_vs_single_root": _vector_pair_aggregate(target_single),
            "target_vs_multi_root": _vector_pair_aggregate(target_multi),
            "q_vs_single_root": _vector_pair_aggregate(q_single),
            "q_vs_multi_root": _vector_pair_aggregate(q_multi),
        }

    return {
        "same_slot_primary": summarize([row for row in rows if row["same_slot_primary"]]),
        "delta_positive_secondary": summarize([row for row in rows if not row["same_slot_primary"]]),
    }


def _vector_pair_aggregate(
    pairs: list[tuple[list[float], list[float]]]
) -> dict[str, Any]:
    top1: list[float] = []
    spearman: list[float] = []
    ordering: list[float] = []
    for left, right in pairs:
        top1.append(float(int(np.argmax(left)) == int(np.argmax(right))))
        value = _spearman(left, right)
        if value is not None:
            spearman.append(float(value))
        for first in range(len(left)):
            for second in range(first + 1, len(left)):
                ordering.append(_ordering_score(
                    left[first] - left[second], right[first] - right[second]
                ))
    return {
        "top1_accuracy": _mean_or_none(top1),
        "spearman_mean": _mean_or_none(spearman),
        "pairwise_ordering_accuracy": _mean_or_none(ordering),
    }


def _audit_checkpoint(
    *,
    checkpoint: Path,
    seed: int,
    phase: str,
    expected_update: int,
    decision_limit: int,
    max_search_slots: int,
    max_future_slots: int,
    min_legal_actions: int,
    include_forced_action_targets: bool = False,
) -> list[dict[str, Any]]:
    import torch
    from torch.distributions import Categorical

    payload = _load_trusted_checkpoint(torch, checkpoint)
    if payload.get("resume_semantics") != "restart_from_new_episode_only":
        raise ValueError("checkpoint does not use new-episode resume semantics")
    if int(payload.get("update_step", -1)) != int(expected_update):
        raise ValueError("checkpoint update does not match the requested phase")
    controls = checkpoint_experiment_controls(payload)
    if not bool(controls.get("offloading_decision_q_credit", False)):
        raise ValueError("checkpoint does not enable Decision-Q")
    if str(controls.get("task_encoder", "hgnn")) != "mlp":
        raise ValueError("Scheme-B2 requires an MLP checkpoint")
    device = torch.device("cpu")
    dims = _module_dims_from_checkpoint(
        payload, argparse.Namespace(task_embedding_dim=None, hidden_dim=None)
    )
    modules = _build_modules(dims=dims, experiment_controls=controls, device=device)
    _load_module_state(modules, payload)
    _set_eval_mode(modules)
    q_state = payload.get("extra_state", {}).get("offloading_decision_q_credit")
    if q_state is None:
        raise ValueError("checkpoint is missing Decision-Q state")
    first_weight = q_state["critic"]["net.0.weight"]
    q_critic = CleanDecisionCritic(
        input_dim=int(first_weight.shape[1]), hidden_dim=int(first_weight.shape[0])
    ).to(device)
    q_critic.load_state_dict(q_state["critic"])
    frozen_modules = {
        "task_encoder": modules.hgnn,
        "movement_actor": modules.movement_actor,
        "offloading_actor": modules.offloading_actor,
        "main_critic": modules.critic,
        "decision_q_critic": q_critic,
    }
    for module in frozen_modules.values():
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    parameters_before = _clone_parameters(frozen_modules)

    env = Env(
        completed_dag_weight=float(controls["completed_dag_weight"]),
        freeze_ue_mobility=bool(controls.get("freeze_ue_mobility", False)),
    )
    builder = CleanGraphBuilder()
    builder.reset()
    _set_rng_state(payload.get("rng_state", {}))
    env.reset()
    replay_episode = int(payload.get("episode", -1)) + 1
    tracker = CleanDecisionTransitionTracker(gamma=float(q_state["gamma"]), lane_index=0)
    tracker.start_episode(replay_episode)
    completed_transitions: dict[Any, Any] = {}
    audited: list[dict[str, Any]] = []
    by_key: dict[Any, dict[str, Any]] = {}
    try:
        for _ in range(max_search_slots):
            with torch.no_grad():
                prepared = prepare_slot_state(env=env, graph_builder=builder)
                encoded = encode_prepared_slot(
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
                        for index, uav_id in enumerate(
                            encoded.movement_observation.uav_ids
                        )
                    }
                )
                snapshot = clone_post_movement_pre_offloading_env(env)
                branch_rng_root = capture_clean_host_rng_state()
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
            records = tuple(modules.offloading_actor.latest_records)
            transition_records = tuple(
                _transition_record(record, encoded.critic_global_input)
                for record in records
            )
            tracker.record_decisions(
                slot_index=int(prepared.slot_index), records=transition_records
            )
            for transition in tracker.pop_completed():
                completed_transitions[decision_state_key(transition.state)] = transition
            trace = capture_clean_counterfactual_baseline_trace(
                decision_records=records, assignment_buffer=assignments
            )
            states = {
                int(record.decision_order): CleanDecisionState.from_rollout_record(
                    transition_record,
                    episode_index=replay_episode,
                    lane_index=0,
                    slot_index=int(prepared.slot_index),
                )
                for record, transition_record in zip(records, transition_records)
            }
            if len(audited) < decision_limit:
                for target_trace in trace:
                    if len(target_trace.legal_uav_ids) < min_legal_actions:
                        continue
                    feasibility = run_clean_counterfactual_branches_serial(
                        env_snapshot=snapshot,
                        baseline_trace=trace,
                        target_decision_order=int(target_trace.decision_order),
                    )
                    if not all(row.suffix_replay_feasible for row in feasibility):
                        continue
                    state = states[int(target_trace.decision_order)]
                    row = _audit_one_decision(
                        checkpoint=checkpoint,
                        checkpoint_payload=payload,
                        seed=seed,
                        phase=phase,
                        state=state,
                        snapshot=snapshot,
                        branch_rng_root=branch_rng_root,
                        trace=trace,
                        target_trace=target_trace,
                        feasibility=feasibility,
                        q_critic=q_critic,
                        q_state=q_state,
                        modules=modules,
                        max_future_slots=max_future_slots,
                        prepared=prepared,
                        encoded=encoded,
                        include_forced_action_targets=include_forced_action_targets,
                    )
                    key = decision_state_key(state)
                    audited.append(row)
                    by_key[key] = row
                    if key in completed_transitions:
                        _attach_target(row, completed_transitions[key], q_critic, q_state)
                    break
            _, _, done, info = env.commit_and_advance(assignment_buffer=assignments)
            tracker.record_slot_reward(float(info["step_reward"]))
            if done:
                tracker.close_terminated()
                for transition in tracker.pop_completed():
                    completed_transitions[decision_state_key(transition.state)] = transition
            for key, row in by_key.items():
                if "target_diagnostics" not in row and key in completed_transitions:
                    _attach_target(row, completed_transitions[key], q_critic, q_state)
            if len(audited) >= decision_limit and all(
                "target_diagnostics" in row for row in audited
            ):
                break
            if done:
                break
    finally:
        builder.close()
    if len(audited) != decision_limit:
        raise RuntimeError(
            f"seed={seed} update={expected_update}: found {len(audited)} of "
            f"{decision_limit} requested decisions"
        )
    if not all("target_diagnostics" in row for row in audited):
        raise RuntimeError("selected decision transition target remained unresolved")
    if not _parameters_equal(parameters_before, _clone_parameters(frozen_modules)):
        raise AssertionError("a frozen parameter changed during the formal audit")
    for row in audited:
        row["gates"]["training_parameters_unchanged"] = True
        row["optimizer_steps"] = 0
    return audited


def _audit_one_decision(
    *,
    checkpoint: Path,
    checkpoint_payload: dict[str, Any],
    seed: int,
    phase: str,
    state: CleanDecisionState,
    snapshot: Any,
    branch_rng_root: Any,
    trace: Any,
    target_trace: Any,
    feasibility: Any,
    q_critic: Any,
    q_state: dict[str, Any],
    modules: Any,
    max_future_slots: int,
    prepared: Any,
    encoded: Any,
    include_forced_action_targets: bool,
) -> dict[str, Any]:
    import torch

    legal_indices, q_inputs = encode_decision_candidate_rows(state)
    with torch.no_grad():
        q_values = q_critic(
            torch.as_tensor(q_inputs, dtype=torch.float32)
        ).cpu().numpy()
    legal_ids = [int(state.candidate_uav_ids[int(index)]) for index in legal_indices]
    if tuple(legal_ids) != target_trace.legal_uav_ids:
        raise AssertionError("Q legal vector/candidate IDs are misaligned")
    q_pairs = list(zip(legal_ids, (float(value) for value in q_values)))
    q_ranking = [
        uav_id for uav_id, _ in sorted(q_pairs, key=lambda item: (-item[1], item[0]))
    ]
    baseline_assignments = tuple(
        (int(row.decision_order), str(row.task_id), int(row.selected_uav_id))
        for row in trace
    )
    current_reward_by_uav: dict[int, float] = {}
    exact_state_proof = []
    for branch in feasibility:
        differing_orders = [
            baseline[0]
            for baseline, replayed in zip(baseline_assignments, branch.replayed_assignments)
            if baseline != replayed
        ]
        expected = (
            []
            if int(branch.forced_uav_id) == int(branch.baseline_uav_id)
            else [int(branch.decision_order)]
        )
        if differing_orders != expected:
            raise AssertionError("exact-state branch changed more than the target decision")
        current_reward_by_uav[int(branch.forced_uav_id)] = float(
            branch.current_slot_reward
        )
        exact_state_proof.append(
            {
                "forced_uav_id": int(branch.forced_uav_id),
                "differing_decision_orders": differing_orders,
            }
        )
    result = run_clean_common_random_completion_process(
        env_snapshot=snapshot,
        baseline_trace=trace,
        target_decision_order=int(target_trace.decision_order),
        initial_rng_state=branch_rng_root,
        gamma=float(q_state["gamma"]),
        task_encoder=modules.hgnn,
        movement_actor=modules.movement_actor,
        offloading_actor=modules.offloading_actor,
        max_future_slots=max_future_slots,
        device="cpu",
    )
    if result.audit.semantic_key_mismatches:
        raise AssertionError("semantic CRN mismatch")
    if result.audit.unrecognized_environment_calls:
        raise AssertionError("unrecognized environment RNG call")
    gamma = float(q_state["gamma"])
    branches = []
    for branch in result.decision.branches:
        future_discounted = float(
            sum(
                (gamma**index) * reward
                for index, reward in enumerate(branch.reward_sequence)
            )
        )
        current_reward = current_reward_by_uav[int(branch.forced_uav_id)]
        branches.append(
            {
                "uav_id": int(branch.forced_uav_id),
                "current_slot_reward": current_reward,
                "future_reward_sequence": list(branch.reward_sequence),
                "h100_truncated_return": current_reward + gamma * future_discounted,
                "long_term_return": (
                    None
                    if branch.common_discounted_return is None
                    else current_reward + gamma * float(branch.common_discounted_return)
                ),
                "target_dag_completion_horizon": branch.target_dag_completion_horizon,
                "target_dag_completion_time": branch.target_dag_completion_time,
                "censored": branch.target_dag_completion_horizon is None,
            }
        )
    censored = any(branch["censored"] for branch in branches)
    top1_correct = None
    spearman = None
    true_ranking = None
    true_best_ids = None
    if not censored:
        returns = {int(row["uav_id"]): float(row["long_term_return"]) for row in branches}
        best = max(returns.values())
        true_best_ids = [uav_id for uav_id, value in returns.items() if value == best]
        true_ranking = [
            uav_id
            for uav_id, _ in sorted(
                returns.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        top1_correct = int(q_ranking[0] in true_best_ids)
        spearman = _spearman(
            [float(value) for value in q_values],
            [returns[uav_id] for uav_id in legal_ids],
        )
    row = {
        "schema": "decision_q_v2_ranking_crn_pilot_decision_v1",
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "historical_training_snapshot_recovered": False,
        "seed": int(seed),
        "phase": str(phase),
        "checkpoint_update": int(checkpoint_payload["update_step"]),
        "checkpoint_episode": int(checkpoint_payload["episode"]),
        "replay_episode": int(state.episode_index),
        "checkpoint": str(checkpoint),
        "slot_index": int(state.slot_index),
        "task_id": state.task_id,
        "decision_order": int(state.decision_order),
        "legal_uav_ids": legal_ids,
        "legal_q_vector": [float(value) for value in q_values],
        "q_ranking": q_ranking,
        "q_top1": int(q_ranking[0]),
        "q_spread": float(np.max(q_values) - np.min(q_values)),
        "baseline_selected_action_index": int(state.selected_action),
        "baseline_selected_uav_id": int(state.selected_uav_id),
        "branches": branches,
        "decision_censored": bool(censored),
        "usable_for_main_ranking": not bool(censored),
        "true_long_term_ranking": true_ranking,
        "true_long_term_best_uav_ids": true_best_ids,
        "q_top1_correct": top1_correct,
        "q_vs_long_term_spearman": spearman,
        "semantic_crn": {
            "execution_backend": "process",
            "serial_process_consistency_validated_by_smoke": True,
            "shared_semantic_keys_checked": int(
                result.audit.shared_semantic_keys_checked
            ),
            "semantic_mismatch_count": 0,
            "unrecognized_environment_rng_calls": 0,
            "stop_reason": result.decision.stop_reason,
            "common_completion_horizon": result.decision.common_completion_horizon,
        },
        "gates": {
            "exact_state_branch_correct": True,
            "strict_suffix_feasible_all": True,
            "q_vector_candidate_ids_aligned": True,
            "single_post_movement_pre_offloading_branch_root": True,
            "policy_and_critics_frozen": True,
        },
        "exact_state_branch_proof": exact_state_proof,
    }
    if include_forced_action_targets:
        legal_probabilities = np.asarray(
            state.old_masked_probabilities, dtype=np.float64
        )[np.asarray(legal_indices, dtype=np.int64)]
        legal_probabilities /= legal_probabilities.sum()
        actor_pairs = list(zip(legal_ids, legal_probabilities.tolist()))
        row.update(
            {
                "schema": "decision_q_v2_root_cause_decision_v1",
                "legal_q_inputs": np.asarray(q_inputs, dtype=np.float32).tolist(),
                "candidate_features": np.asarray(
                    state.candidate_features, dtype=np.float32
                ).tolist(),
                "critic_global_context": np.asarray(
                    state.critic_global_context, dtype=np.float32
                ).reshape(-1).tolist(),
                "actor_legal_probabilities": [
                    float(value) for value in legal_probabilities
                ],
                "actor_ranking": [
                    int(uav_id)
                    for uav_id, _ in sorted(
                        actor_pairs, key=lambda item: (-item[1], item[0])
                    )
                ],
                "actor_probability_spread": float(
                    np.max(legal_probabilities) - np.min(legal_probabilities)
                ),
                "actor_normalized_entropy": _normalized_entropy(
                    legal_probabilities
                ),
                "state_scalar_suffix": {
                    "legal_fraction": float(len(legal_ids))
                    / float(max(len(state.candidate_uav_ids), 1)),
                    "normalized_decision_order": float(
                        np.asarray(q_inputs, dtype=np.float32)[0, -1]
                    ),
                },
            }
        )
        target_rows, target_audit = _reconstruct_forced_action_targets(
            state=state,
            snapshot=snapshot,
            branch_rng_root=branch_rng_root,
            trace=trace,
            target_trace=target_trace,
            q_critic=q_critic,
            q_state=q_state,
            modules=modules,
            prepared=prepared,
            encoded=encoded,
            max_future_slots=max_future_slots,
        )
        row["forced_action_targets"] = target_rows
        row["target_semantic_crn"] = target_audit
        row["gates"]["forced_target_candidate_ids_aligned"] = (
            [int(item["uav_id"]) for item in target_rows] == legal_ids
        )
        row["gates"]["forced_target_same_slot_gamma_semantics"] = all(
            not (
                int(item["delta"]) == 0
                and not math.isclose(float(item["bootstrap"]), float(item["expected_next_q"]))
            )
            for item in target_rows
            if not item["target_censored"]
        )
        if not all(row["gates"].values()):
            raise AssertionError("root-cause forced-action target gate failed")
    return row


def _attach_target(
    row: dict[str, Any], transition: Any, q_critic: Any, q_state: dict[str, Any]
) -> None:
    import torch

    if transition.truncated or transition.unresolved:
        raise ValueError("formal audit selected an unresolved transition")
    bootstrap = 0.0
    if not transition.terminated:
        if transition.next_state is None:
            raise ValueError("nonterminal transition is missing next state")
        legal_indices, inputs = encode_decision_candidate_rows(transition.next_state)
        with torch.no_grad():
            next_q = q_critic(torch.as_tensor(inputs, dtype=torch.float32)).cpu().numpy()
        bootstrap = (float(q_state["gamma"]) ** int(transition.delta)) * expected_behavior_q(
            legal_indices=legal_indices,
            probabilities=transition.next_state.old_masked_probabilities,
            q_values=next_q,
        )
    target = float(transition.rho) + float(bootstrap)
    row["target_diagnostics"] = {
        "rho": float(transition.rho),
        "delta": int(transition.delta),
        "bootstrap": float(bootstrap),
        "target": target,
        "terminated": bool(transition.terminated),
        "same_slot_no_gamma": bool(int(transition.delta) == 0),
    }


def _reconstruct_forced_action_targets(
    *,
    state: CleanDecisionState,
    snapshot: Any,
    branch_rng_root: Any,
    trace: Any,
    target_trace: Any,
    q_critic: Any,
    q_state: dict[str, Any],
    modules: Any,
    prepared: Any,
    encoded: Any,
    max_future_slots: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    semantic_audits: list[dict[str, Any]] = []
    for forced_uav_id in target_trace.legal_uav_ids:
        row, semantic_audit = _reconstruct_one_forced_action_target(
            state=state,
            snapshot=snapshot,
            branch_rng_root=branch_rng_root,
            trace=trace,
            target_trace=target_trace,
            forced_uav_id=int(forced_uav_id),
            q_critic=q_critic,
            q_state=q_state,
            modules=modules,
            prepared=prepared,
            encoded=encoded,
            max_future_slots=max_future_slots,
        )
        rows.append(row)
        semantic_audits.append(semantic_audit)
    audit = audit_clean_semantic_common_random(semantic_audits)
    if audit.semantic_key_mismatches:
        raise AssertionError("forced-action target semantic CRN mismatch")
    if audit.unrecognized_environment_calls:
        raise AssertionError("forced-action target used unrecognized environment RNG")
    return rows, {
        "shared_semantic_keys_checked": int(audit.shared_semantic_keys_checked),
        "semantic_mismatch_count": 0,
        "unrecognized_environment_rng_calls": 0,
    }


def _reconstruct_one_forced_action_target(
    *,
    state: CleanDecisionState,
    snapshot: Any,
    branch_rng_root: Any,
    trace: Any,
    target_trace: Any,
    forced_uav_id: int,
    q_critic: Any,
    q_state: dict[str, Any],
    modules: Any,
    prepared: Any,
    encoded: Any,
    max_future_slots: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    gamma = float(q_state["gamma"])
    env = clone_post_movement_pre_offloading_env(snapshot)
    tracker = CleanDecisionTransitionTracker(gamma=gamma, lane_index=0)
    tracker.start_episode(int(state.episode_index))
    reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
    assignments = CleanAssignmentBuffer()
    task_embeddings = np.asarray(
        encoded.task_embeddings.detach().cpu(), dtype=np.float32
    )
    target_transition = None
    current_probability_max_error = None
    ordered_trace = tuple(sorted(trace, key=lambda item: int(item.decision_order)))
    for trace_row in ordered_trace:
        task = env.task_manager.get_task(trace_row.task_id)
        if task is None:
            raise ValueError("forced-action target replay lost a baseline task")
        dynamic, pair, mask, candidate_ids, estimates = (
            build_offloading_candidate_components(
                task=task,
                uavs=env.uavs,
                task_manager=env.task_manager,
                executor=env.executor,
                state_view=reservation,
                current_time_seconds=float(env.current_time_seconds),
                uav_service_positions=env.uav_service_positions,
                ue_service_positions=env.ue_service_positions,
                ues=env.ues,
            )
        )
        task_index = prepared.graph_snapshot.task_id_to_idx[str(task.task_id)]
        features = np.concatenate(
            [
                np.repeat(
                    task_embeddings[int(task_index)].reshape(1, -1),
                    len(candidate_ids),
                    axis=0,
                ),
                dynamic,
                pair,
            ],
            axis=1,
        ).astype(np.float32)
        with torch.no_grad():
            logits = modules.offloading_actor.scorer(
                torch.as_tensor(features, dtype=torch.float32)
            )
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
            probabilities = torch.softmax(
                logits.masked_fill(~mask_tensor, torch.finfo(torch.float32).min),
                dim=0,
            ).cpu().numpy()
        selected_uav_id = (
            int(forced_uav_id)
            if int(trace_row.decision_order) == int(target_trace.decision_order)
            else int(trace_row.selected_uav_id)
        )
        candidate_ids = [int(value) for value in candidate_ids]
        if selected_uav_id not in candidate_ids:
            raise AssertionError("forced-action target suffix candidate disappeared")
        selected_action = candidate_ids.index(selected_uav_id)
        if not bool(mask[selected_action]):
            raise AssertionError("forced-action target suffix became infeasible")
        record = SimpleNamespace(
            task_id=str(task.task_id),
            task_local_index=int(task_index),
            dag_id=str(task.dag_id),
            decision_order=int(trace_row.decision_order),
            selected_action=int(selected_action),
            selected_uav_id=int(selected_uav_id),
            candidate_uav_ids=tuple(candidate_ids),
            candidate_mask=np.asarray(mask, dtype=bool),
            candidate_features=features,
            critic_global_context=np.asarray(
                encoded.critic_global_input.detach().cpu(), dtype=np.float32
            ),
            old_masked_probabilities=np.asarray(probabilities, dtype=np.float32),
            old_log_probability=float(math.log(max(float(probabilities[selected_action]), 1e-30))),
        )
        if int(trace_row.decision_order) == int(target_trace.decision_order):
            if tuple(candidate_ids) != tuple(state.candidate_uav_ids):
                raise AssertionError("forced target current candidate IDs changed")
            if not np.array_equal(np.asarray(mask, dtype=bool), state.candidate_mask):
                raise AssertionError("forced target current legal mask changed")
            if not np.allclose(features, state.candidate_features, rtol=0.0, atol=1e-6):
                raise AssertionError("forced target current candidate features changed")
            current_probability_max_error = float(
                np.max(
                    np.abs(
                        np.asarray(probabilities, dtype=np.float64)
                        - np.asarray(state.old_masked_probabilities, dtype=np.float64)
                    )
                )
            )
        tracker.record_decisions(
            slot_index=int(prepared.slot_index), records=(record,)
        )
        for transition in tracker.pop_completed():
            if (
                str(transition.state.task_id) == str(state.task_id)
                and int(transition.state.decision_order) == int(state.decision_order)
            ):
                target_transition = transition
        estimate = estimates[selected_action]
        assignments.append(str(task.task_id), selected_uav_id, int(trace_row.decision_order))
        reservation.reserve(
            str(task.task_id),
            selected_uav_id,
            estimated_available_time=float(estimate.estimated_finish_time),
            estimated_queued_workload=float(estimate.estimated_queued_workload),
        )
        if target_transition is not None:
            break

    semantic = CleanSemanticCommonRandom(branch_rng_root)
    if target_transition is None:
        _, _, done, info = env.commit_and_advance(assignment_buffer=assignments)
        tracker.record_slot_reward(float(info["step_reward"]))
        if done:
            tracker.close_terminated()
        target_transition = _pop_target_transition(tracker, state)

    builder = CleanGraphBuilder()
    builder.reset()
    try:
        for future_slot in range(1, int(max_future_slots) + 1):
            if target_transition is not None:
                break
            with semantic.scoped_environment_calls(future_slot):
                future_prepared = prepare_slot_state(env=env, graph_builder=builder)
                future_encoded = encode_prepared_slot(
                    prepared_state=future_prepared,
                    env=env,
                    hgnn=modules.hgnn,
                    critic=modules.critic,
                    movement_actor=modules.movement_actor,
                    device="cpu",
                )
                movement = torch.argmax(future_encoded.movement_logits, dim=-1)
                env.apply_movement(
                    {
                        int(uav_id): int(movement[index].cpu().item())
                        for index, uav_id in enumerate(
                            future_encoded.movement_observation.uav_ids
                        )
                    }
                )
                ready = [
                    env.task_manager.get_task(task_id)
                    for task_id in future_prepared.frozen_ready_task_ids
                ]
                ready = [task for task in ready if task is not None and task.is_ready]
                future_assignments = modules.offloading_actor.act(
                    frozen_ready_tasks=ready,
                    task_embeddings=future_encoded.task_embeddings.detach(),
                    graph_snapshot=future_prepared.graph_snapshot,
                    task_manager=env.task_manager,
                    uavs=env.uavs,
                    executor=env.executor,
                    current_time_seconds=env.current_time_seconds,
                    uav_service_positions=env.uav_service_positions,
                    ue_service_positions=env.ue_service_positions,
                    ues=env.ues,
                    deterministic=True,
                )
                future_records = tuple(
                    _transition_record(record, future_encoded.critic_global_input)
                    for record in modules.offloading_actor.latest_records
                )
                tracker.record_decisions(
                    slot_index=int(future_prepared.slot_index),
                    records=future_records,
                )
                target_transition = _pop_target_transition(tracker, state)
                if target_transition is None:
                    _, _, done, info = env.commit_and_advance(
                        assignment_buffer=future_assignments
                    )
                    tracker.record_slot_reward(float(info["step_reward"]))
                    if done:
                        tracker.close_terminated()
                    target_transition = _pop_target_transition(tracker, state)
    finally:
        builder.close()

    audit_snapshot = semantic.audit_snapshot()
    if target_transition is None:
        return (
            {
                "uav_id": int(forced_uav_id),
                "target_censored": True,
                "rho": None,
                "delta": None,
                "expected_next_q": None,
                "bootstrap": None,
                "target": None,
                "same_slot_bootstrap_only": False,
                "current_actor_probability_max_error": current_probability_max_error,
            },
            audit_snapshot,
        )
    bootstrap = 0.0
    expected_next_q = 0.0
    next_legal_ids: list[int] = []
    next_q_spread = 0.0
    if not target_transition.terminated:
        if target_transition.next_state is None:
            raise ValueError("forced-action target is missing its next state")
        next_legal_indices, next_inputs = encode_decision_candidate_rows(
            target_transition.next_state
        )
        with torch.no_grad():
            next_q = q_critic(
                torch.as_tensor(next_inputs, dtype=torch.float32)
            ).cpu().numpy()
        expected_next_q = expected_behavior_q(
            legal_indices=next_legal_indices,
            probabilities=target_transition.next_state.old_masked_probabilities,
            q_values=next_q,
        )
        bootstrap = (gamma ** int(target_transition.delta)) * expected_next_q
        next_legal_ids = [
            int(target_transition.next_state.candidate_uav_ids[int(index)])
            for index in next_legal_indices
        ]
        next_q_spread = float(np.max(next_q) - np.min(next_q))
    target = float(target_transition.rho) + float(bootstrap)
    return (
        {
            "uav_id": int(forced_uav_id),
            "target_censored": False,
            "rho": float(target_transition.rho),
            "delta": int(target_transition.delta),
            "expected_next_q": float(expected_next_q),
            "bootstrap": float(bootstrap),
            "target": float(target),
            "terminated": bool(target_transition.terminated),
            "same_slot_bootstrap_only": bool(
                int(target_transition.delta) == 0
                and math.isclose(float(target_transition.rho), 0.0, abs_tol=1e-12)
            ),
            "next_legal_uav_ids": next_legal_ids,
            "next_q_spread": next_q_spread,
            "current_actor_probability_max_error": current_probability_max_error,
        },
        audit_snapshot,
    )


def _pop_target_transition(tracker: Any, state: CleanDecisionState) -> Any:
    result = None
    for transition in tracker.pop_completed():
        if (
            str(transition.state.task_id) == str(state.task_id)
            and int(transition.state.decision_order) == int(state.decision_order)
        ):
            result = transition
    return result


def _transition_record(record: Any, critic_global_context: Any) -> Any:
    return SimpleNamespace(
        task_id=str(record.task_id),
        task_local_index=int(record.task_local_index),
        dag_id=str(record.dag_id),
        decision_order=int(record.decision_order),
        selected_action=int(record.selected_action),
        selected_uav_id=int(record.selected_uav_id),
        candidate_uav_ids=tuple(int(value) for value in record.candidate_uav_ids),
        candidate_mask=record.candidate_mask,
        candidate_features=record.candidate_features,
        critic_global_context=critic_global_context,
        old_masked_probabilities=record.old_masked_probabilities,
        old_log_probability=float(record.old_log_prob),
    )


def _root_cause_summary(
    decisions: list[dict[str, Any]],
    ev_rows: dict[tuple[int, int], dict[str, float]],
    metrics_paths: dict[int, Path],
) -> dict[str, Any]:
    phase_updates = {"early": 30, "mid": 60, "late": 120}
    phases: dict[str, Any] = {}
    for phase, update in phase_updates.items():
        phase_rows = [row for row in decisions if row["phase"] == phase]
        by_seed = {}
        for seed in (0, 1, 2):
            seed_rows = [row for row in phase_rows if int(row["seed"]) == seed]
            by_seed[str(seed)] = {
                **_three_layer_calibration(seed_rows),
                "q_spread_mean": _mean_or_none(
                    [float(row["q_spread"]) for row in seed_rows]
                ),
                "training_q_ev": ev_rows[(seed, update)],
            }
        phases[phase] = {
            "checkpoint_update": update,
            "by_seed": by_seed,
            "pooled": {
                **_three_layer_calibration(phase_rows),
                "q_spread_mean": _mean_or_none(
                    [float(row["q_spread"]) for row in phase_rows]
                ),
                "training_q_ev_pre_mean": _mean_or_none(
                    [ev_rows[(seed, update)]["training_q_ev_pre"] for seed in (0, 1, 2)]
                ),
                "training_q_ev_post_mean": _mean_or_none(
                    [ev_rows[(seed, update)]["training_q_ev_post"] for seed in (0, 1, 2)]
                ),
            },
        }
    metrics = {
        seed: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for seed, path in metrics_paths.items()
    }
    return {
        "schema": "decision_q_v2_root_cause_summary_v1",
        "pilot_only_not_final_statistics": True,
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "historical_training_snapshot_recovered": False,
        "selected_decision_count": len(decisions),
        "phases": phases,
        "bootstrap_audit": _bootstrap_audit(decisions),
        "state_nearest_pairs": _state_nearest_pairs(decisions),
        "training_timeline": _training_timeline(metrics),
        "gates": {
            "all_decision_gates": all(
                all(bool(value) for value in row["gates"].values())
                for row in decisions
            ),
            "semantic_mismatch_count": sum(
                int(row["semantic_crn"]["semantic_mismatch_count"])
                + int(row["target_semantic_crn"]["semantic_mismatch_count"])
                for row in decisions
            ),
            "unrecognized_environment_rng_calls": sum(
                int(row["semantic_crn"]["unrecognized_environment_rng_calls"])
                + int(row["target_semantic_crn"]["unrecognized_environment_rng_calls"])
                for row in decisions
            ),
            "optimizer_steps": sum(int(row["optimizer_steps"]) for row in decisions),
        },
    }


def _three_layer_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "selected_decision_count": len(rows),
        "truth_censored_fraction": (
            sum(bool(row["decision_censored"]) for row in rows) / max(len(rows), 1)
        ),
        "target_censored_fraction": (
            sum(
                any(item["target_censored"] for item in row["forced_action_targets"])
                for row in rows
            )
            / max(len(rows), 1)
        ),
    }
    for name, prediction, reference in (
        ("q_prediction_vs_crn_truth", "q", "truth"),
        ("training_target_vs_crn_truth", "target", "truth"),
        ("q_prediction_vs_training_target", "q", "target"),
        ("actor_preference_vs_crn_truth", "actor", "truth"),
        ("actor_preference_vs_training_target", "actor", "target"),
        ("actor_preference_vs_q_prediction", "actor", "q"),
    ):
        result[name] = _relation_calibration(rows, prediction, reference)
    result["actor_normalized_entropy_mean"] = _mean_or_none(
        [float(row["actor_normalized_entropy"]) for row in rows]
    )
    result["actor_probability_spread_mean"] = _mean_or_none(
        [float(row["actor_probability_spread"]) for row in rows]
    )
    return result


def _relation_calibration(
    rows: list[dict[str, Any]], prediction: str, reference: str
) -> dict[str, Any]:
    top1: list[float] = []
    spearman: list[float] = []
    pairwise_correct = 0.0
    pairwise_count = 0
    for row in rows:
        vectors = _decision_vectors(row)
        left = vectors.get(prediction)
        right = vectors.get(reference)
        if left is None or right is None:
            continue
        reference_max = float(np.max(right))
        reference_best = {
            index
            for index, value in enumerate(right)
            if math.isclose(float(value), reference_max, rel_tol=0.0, abs_tol=1e-12)
        }
        top1.append(float(int(np.argmax(left)) in reference_best))
        value = _spearman([float(v) for v in left], [float(v) for v in right])
        if value is not None:
            spearman.append(float(value))
        correct, count = _pairwise_ordering(left, right)
        pairwise_correct += correct
        pairwise_count += count
    return {
        "usable_decision_count": len(top1),
        "censored_fraction": 1.0 - len(top1) / max(len(rows), 1),
        "top1_accuracy": _mean_or_none(top1),
        "spearman_mean": _mean_or_none(spearman),
        "pairwise_ordering_accuracy": (
            None if pairwise_count == 0 else float(pairwise_correct / pairwise_count)
        ),
        "pairwise_comparison_count": int(pairwise_count),
    }


def _decision_vectors(row: dict[str, Any]) -> dict[str, np.ndarray | None]:
    legal_ids = [int(value) for value in row["legal_uav_ids"]]
    truth_by_id = {
        int(item["uav_id"]): item["long_term_return"] for item in row["branches"]
    }
    target_by_id = {
        int(item["uav_id"]): item["target"]
        for item in row["forced_action_targets"]
    }
    truth = None
    if all(truth_by_id[uav_id] is not None for uav_id in legal_ids):
        truth = np.asarray([truth_by_id[uav_id] for uav_id in legal_ids], dtype=np.float64)
    target = None
    if all(target_by_id[uav_id] is not None for uav_id in legal_ids):
        target = np.asarray([target_by_id[uav_id] for uav_id in legal_ids], dtype=np.float64)
    return {
        "q": np.asarray(row["legal_q_vector"], dtype=np.float64),
        "actor": np.asarray(row["actor_legal_probabilities"], dtype=np.float64),
        "truth": truth,
        "target": target,
    }


def _pairwise_ordering(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    correct = 0.0
    count = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            reference_delta = float(right[first] - right[second])
            if math.isclose(reference_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
                continue
            prediction_delta = float(left[first] - left[second])
            count += 1
            if math.isclose(prediction_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
                correct += 0.5
            elif prediction_delta * reference_delta > 0.0:
                correct += 1.0
    return correct, count


def _bootstrap_audit(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("early", "mid", "late"):
        phase_rows = [row for row in decisions if row["phase"] == phase]
        phase_result: dict[str, Any] = {}
        for group in ("bootstrap_only", "delta_positive"):
            selected = []
            grouped_decisions = []
            for row in phase_rows:
                action_rows = [
                    item for item in row["forced_action_targets"]
                    if not item["target_censored"]
                ]
                if group == "bootstrap_only":
                    matching = [item for item in action_rows if item["same_slot_bootstrap_only"]]
                else:
                    matching = [item for item in action_rows if int(item["delta"]) > 0]
                selected.extend(matching)
                if len(matching) == len(row["forced_action_targets"]):
                    grouped_decisions.append(row)
            rho = np.asarray([item["rho"] for item in selected], dtype=np.float64)
            bootstrap = np.asarray(
                [item["bootstrap"] for item in selected], dtype=np.float64
            )
            target = np.asarray([item["target"] for item in selected], dtype=np.float64)
            phase_result[group] = {
                "action_row_count": len(selected),
                "decision_count_all_actions_in_group": len(grouped_decisions),
                "target_vs_truth": _relation_calibration(
                    grouped_decisions, "target", "truth"
                ),
                "q_vs_truth": _relation_calibration(grouped_decisions, "q", "truth"),
                **_variance_decomposition(rho, bootstrap, target),
                "mean_abs_bootstrap_over_mean_abs_target": (
                    None
                    if target.size == 0
                    else float(
                        np.mean(np.abs(bootstrap))
                        / max(float(np.mean(np.abs(target))), 1e-12)
                    )
                ),
                "q_spread_mean": _mean_or_none(
                    [float(row["q_spread"]) for row in grouped_decisions]
                ),
                "bootstrap_action_spread_mean": _mean_or_none(
                    [
                        float(
                            np.ptp(
                                [
                                    item["bootstrap"]
                                    for item in row["forced_action_targets"]
                                ]
                            )
                        )
                        for row in grouped_decisions
                    ]
                ),
            }
        all_targets = [
            item
            for row in phase_rows
            for item in row["forced_action_targets"]
            if not item["target_censored"]
        ]
        phase_result["bootstrap_only_action_row_fraction"] = (
            sum(bool(item["same_slot_bootstrap_only"]) for item in all_targets)
            / max(len(all_targets), 1)
        )
        result[phase] = phase_result
    return result


def _variance_decomposition(
    rho: np.ndarray, bootstrap: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    if target.size == 0:
        return {
            "target_variance": None,
            "rho_variance": None,
            "bootstrap_variance": None,
            "two_cov_rho_bootstrap": None,
            "variance_identity_error": None,
        }
    rho_variance = float(np.var(rho, ddof=0))
    bootstrap_variance = float(np.var(bootstrap, ddof=0))
    target_variance = float(np.var(target, ddof=0))
    covariance_twice = float(
        2.0 * np.mean((rho - rho.mean()) * (bootstrap - bootstrap.mean()))
    )
    return {
        "target_variance": target_variance,
        "rho_variance": rho_variance,
        "bootstrap_variance": bootstrap_variance,
        "two_cov_rho_bootstrap": covariance_twice,
        "variance_identity_error": float(
            target_variance - rho_variance - bootstrap_variance - covariance_twice
        ),
    }


def _state_nearest_pairs(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for seed in (0, 1, 2):
        for phase in ("early", "mid", "late"):
            rows = [
                row for row in decisions
                if int(row["seed"]) == seed and row["phase"] == phase
                and not row["decision_censored"]
            ]
            pairs = []
            for first in range(len(rows)):
                for second in range(first + 1, len(rows)):
                    left = np.asarray(rows[first]["legal_q_inputs"], dtype=np.float64).reshape(-1)
                    right = np.asarray(rows[second]["legal_q_inputs"], dtype=np.float64).reshape(-1)
                    distance = float(np.sqrt(np.mean((left - right) ** 2)))
                    left_truth = _decision_vectors(rows[first])["truth"]
                    right_truth = _decision_vectors(rows[second])["truth"]
                    truth_spearman = (
                        None
                        if left_truth is None or right_truth is None
                        else _spearman(left_truth.tolist(), right_truth.tolist())
                    )
                    pairs.append((distance, truth_spearman, first, second))
            if pairs:
                distance, truth_spearman, first, second = min(pairs, key=lambda item: item[0])
                result.append(
                    {
                        "seed": seed,
                        "phase": phase,
                        "pair": [
                            {"slot_index": rows[first]["slot_index"], "task_id": rows[first]["task_id"]},
                            {"slot_index": rows[second]["slot_index"], "task_id": rows[second]["task_id"]},
                        ],
                        "q_input_rms_distance": distance,
                        "crn_return_vector_spearman": truth_spearman,
                        "candidate_pair_count": len(pairs),
                    }
                )
    return result


def _training_timeline(metrics: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    fields = {
        "movement_entropy": "rollout_movement_entropy_normalized_mean",
        "offloading_entropy": "rollout_offloading_entropy_normalized_mean",
        "q_ev_pre": "decision_q_ev_pre_update",
        "q_ev_post": "decision_q_ev_post_update",
        "q_spread": "decision_q_legal_action_spread_mean",
        "target_std": "decision_q_target_std",
        "scaled_actor_advantage_std": "decision_q_scaled_actor_advantage_std",
        "q_preclip_grad_norm_mean": "decision_q_preclip_grad_norm_mean",
        "q_preclip_grad_norm_max": "decision_q_preclip_grad_norm_max",
        "q_value_clip_fraction": "decision_q_value_clip_fraction",
        "q_normalized_loss": "decision_q_normalized_loss",
        "offloading_actor_parameter_update_norm": "offloading_actor_parameter_update_norm",
        "offloading_actor_preclip_grad_norm": "grad_pre_clip_offloading",
        "global_grad_clip_scale": "grad_clip_scale",
    }
    flattened: list[dict[str, float]] = []
    by_seed: dict[str, Any] = {}
    for seed, records in metrics.items():
        normalized = []
        for record in records:
            diagnostics = record["ppo_diagnostics"]
            row = {"update": int(record["ppo_update_step"])}
            for name, key in fields.items():
                row[name] = float(diagnostics[key])
            normalized.append(row)
            flattened.append(row)
        by_seed[str(seed)] = _timeline_one(normalized)
    return {"by_seed": by_seed, "pooled": _timeline_one(flattened)}


def _timeline_one(rows: list[dict[str, float]]) -> dict[str, Any]:
    windows = {"early_1_30": (1, 30), "mid_31_60": (31, 60), "late_61_120": (61, 120)}
    result: dict[str, Any] = {"windows": {}}
    value_fields = [key for key in rows[0] if key != "update"] if rows else []
    for name, (start, end) in windows.items():
        selected = [row for row in rows if start <= int(row["update"]) <= end]
        result["windows"][name] = {
            key: _mean_or_none([row[key] for row in selected]) for key in value_fields
        }
    result["correlations"] = {
        "movement_entropy_vs_q_ev_post": _pearson(rows, "movement_entropy", "q_ev_post"),
        "movement_entropy_vs_q_spread": _pearson(rows, "movement_entropy", "q_spread"),
        "q_spread_vs_offloading_entropy": _pearson(rows, "q_spread", "offloading_entropy"),
        "q_spread_vs_actor_parameter_update": _pearson(
            rows, "q_spread", "offloading_actor_parameter_update_norm"
        ),
        "q_ev_post_vs_offloading_entropy": _pearson(rows, "q_ev_post", "offloading_entropy"),
    }
    result["first_update_below_movement_entropy"] = {
        str(threshold): next(
            (int(row["update"]) for row in sorted(rows, key=lambda item: item["update"])
             if row["movement_entropy"] < threshold),
            None,
        )
        for threshold in (0.8, 0.5, 0.2)
    }
    result["mean_q_ev_single_update_gain"] = _mean_or_none(
        [row["q_ev_post"] - row["q_ev_pre"] for row in rows]
    )
    return result


def _pearson(rows: list[dict[str, float]], left: str, right: str) -> float | None:
    if len(rows) < 2:
        return None
    x = np.asarray([row[left] for row in rows], dtype=np.float64)
    y = np.asarray([row[right] for row in rows], dtype=np.float64)
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _normalized_entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = values[values > 0.0]
    if values.size <= 1:
        return 0.0
    return float(-np.sum(values * np.log(values)) / math.log(values.size))


def _pilot_summary(
    decisions: list[dict[str, Any]],
    ev_rows: dict[tuple[int, int], dict[str, float]],
) -> dict[str, Any]:
    phase_updates = {"early": 30, "mid": 60, "late": 120}
    phases: dict[str, Any] = {}
    for phase, update in phase_updates.items():
        by_seed = {}
        for seed in (0, 1, 2):
            rows = [
                row
                for row in decisions
                if row["phase"] == phase and int(row["seed"]) == seed
            ]
            by_seed[str(seed)] = {
                **_aggregate_ranking(rows),
                **ev_rows[(seed, update)],
            }
        pooled_rows = [row for row in decisions if row["phase"] == phase]
        phases[phase] = {
            "checkpoint_update": update,
            "by_seed": by_seed,
            "pooled": {
                **_aggregate_ranking(pooled_rows),
                "training_q_ev_pre_mean": float(
                    np.mean([ev_rows[(seed, update)]["training_q_ev_pre"] for seed in (0, 1, 2)])
                ),
                "training_q_ev_post_mean": float(
                    np.mean([ev_rows[(seed, update)]["training_q_ev_post"] for seed in (0, 1, 2)])
                ),
            },
        }
    return {
        "schema": "decision_q_v2_ranking_crn_pilot_summary_v1",
        "pilot_only_not_final_statistics": True,
        "state_semantics": "frozen-checkpoint replay-generated on-policy decision states",
        "historical_training_snapshot_recovered": False,
        "configured_decision_cap": 27,
        "selected_decision_count": len(decisions),
        "phases": phases,
    }


def _aggregate_ranking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["usable_for_main_ranking"]]
    spearman_values = [
        float(row["q_vs_long_term_spearman"])
        for row in usable
        if row["q_vs_long_term_spearman"] is not None
    ]
    return {
        "selected_decision_count": len(rows),
        "usable_decision_count": len(usable),
        "q_top1_accuracy": (
            None
            if not usable
            else float(np.mean([int(row["q_top1_correct"]) for row in usable]))
        ),
        "spearman_mean": (
            None if not spearman_values else float(np.mean(spearman_values))
        ),
        "spearman_decision_count": len(spearman_values),
        "q_spread_mean_all": (
            None if not rows else float(np.mean([row["q_spread"] for row in rows]))
        ),
        "q_spread_mean_usable": (
            None
            if not usable
            else float(np.mean([row["q_spread"] for row in usable]))
        ),
        "censored_fraction": (
            None
            if not rows
            else float(np.mean([int(row["decision_censored"]) for row in rows]))
        ),
    }


def _training_q_ev(metrics_path: Path, update: int) -> dict[str, float]:
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["ppo_update_step"]) == int(update):
            diagnostics = record["ppo_diagnostics"]
            return {
                "training_q_ev_pre": float(diagnostics["decision_q_ev_pre_update"]),
                "training_q_ev_post": float(diagnostics["decision_q_ev_post_update"]),
            }
    raise ValueError(f"training metrics do not contain update {update}: {metrics_path}")


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x_rank = _average_ranks(np.asarray(left, dtype=np.float64))
    y_rank = _average_ranks(np.asarray(right, dtype=np.float64))
    if float(x_rank.std()) == 0.0 or float(y_rank.std()) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _single_path(paths: Any, label: str, *, require_directory: bool = False) -> Path:
    values = [path for path in paths if not require_directory or path.is_dir()]
    if len(values) != 1:
        raise ValueError(f"expected one {label}, found {len(values)}")
    return values[0]


def _decision_state(
    record: Any,
    *,
    critic_global_context: Any,
    episode_index: int,
    slot_index: int,
) -> CleanDecisionState:
    return CleanDecisionState(
        episode_index=int(episode_index),
        lane_index=0,
        slot_index=int(slot_index),
        task_id=str(record.task_id),
        task_local_index=int(record.task_local_index),
        dag_id=str(record.dag_id),
        decision_order=int(record.decision_order),
        selected_action=int(record.selected_action),
        selected_uav_id=int(record.selected_uav_id),
        candidate_uav_ids=tuple(int(value) for value in record.candidate_uav_ids),
        candidate_mask=np.asarray(record.candidate_mask.cpu(), dtype=bool).copy(),
        candidate_features=np.asarray(
            record.candidate_features.cpu(), dtype=np.float32
        ).copy(),
        critic_global_context=np.asarray(
            critic_global_context.detach().cpu(), dtype=np.float32
        ).reshape(-1).copy(),
        old_masked_probabilities=np.asarray(
            record.old_masked_probabilities.cpu(), dtype=np.float32
        ).reshape(-1).copy(),
        old_log_probability=float(record.old_log_prob),
    )


def _clone_parameters(modules: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }
        for name, module in modules.items()
    }


def _parameters_equal(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> bool:
    import torch

    return all(
        torch.equal(left[module][key], right[module][key])
        for module in left
        for key in left[module]
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "smoke":
        result = run_smoke(args)
    elif args.mode == "multi-root":
        result = run_multi_root(args)
    elif args.mode == "multi-root-summary":
        result = run_multi_root_summary_only(args)
    else:
        result = run_pilot(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
