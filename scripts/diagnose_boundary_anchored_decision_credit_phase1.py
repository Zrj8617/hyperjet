"""Phase 1 paired local counterfactual-credit feasibility audit.

This diagnostic replays frozen checkpoints from a new episode, reconstructs
the existing Decision-Q source decisions, and measures

    C_i^local = r_t^real - r_{t,i}^cf

from paired exact-state Scheme-B2 semantic-CRN branches.  The paired branch
outcomes are diagnostic only: this module has no training or loss integration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

import config
from environment.assignment import (
    CleanAssignmentBuffer,
    TemporaryReservationState,
    build_offloading_candidate_components,
)
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_counterfactual_oracle import (
    CleanCounterfactualDecisionTrace,
    capture_clean_counterfactual_baseline_trace,
    clone_post_movement_pre_offloading_env,
)
from marl_models.mappo.clean_counterfactual_oracle_common_random import (
    CleanSemanticCommonRandom,
    audit_clean_semantic_common_random,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import (
    CleanHostRngState,
    capture_clean_host_rng_state,
)
from marl_models.mappo.clean_offloading_decision_q_credit import (
    encode_decision_candidate_rows,
)
from marl_models.mappo.clean_ppo import CleanDecisionCritic
from marl_models.mappo.clean_slot_orchestrator import (
    encode_prepared_slot,
    prepare_slot_state,
)
from marl_models.mappo.clean_trainer import _set_rng_state
from scripts.diagnose_decision_q_v2_ranking_crn import (
    _clone_parameters,
    _decision_state,
    _independent_semantic_root,
    _parameters_equal,
    _spearman,
    _t_critical_975,
)
from scripts.eval_clean_mainline import (
    _build_modules,
    _load_module_state,
    _load_trusted_checkpoint,
    _module_dims_from_checkpoint,
    _set_eval_mode,
)
from scripts.train_clean_mainline import checkpoint_experiment_controls


STATE_SEMANTICS = "frozen-checkpoint replay-generated on-policy decision states"
PHASE_BY_UPDATE = {30: "early", 60: "mid", 120: "late"}
REWARD_COMPONENT_NAMES = (
    "step_time_penalty",
    "step_energy_penalty",
    "step_task_energy_penalty",
    "step_movement_energy_penalty",
    "step_completed_dag_bonus",
    "step_movement_position_bonus",
)


@dataclass(frozen=True, slots=True)
class LocalCreditBranchResult:
    forced_uav_id: int
    target_pre_decision_reservation: dict[str, Any]
    current_slot_reward: float | None
    reward_components: dict[str, float]
    replayed_assignments: tuple[tuple[int, str, int], ...]
    suffix_assignments: tuple[tuple[int, str, int], ...]
    completed_task_ids: tuple[str, ...]
    completed_dag_ids: tuple[str, ...]
    next_time_seconds: float | None
    censored: bool
    censor_reason: str | None
    semantic_audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LocalCreditPairResult:
    actual: LocalCreditBranchResult
    counterfactual: LocalCreditBranchResult
    shared_semantic_keys_checked: int
    semantic_mismatch_count: int
    unrecognized_environment_rng_calls: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1 diagnostic-only paired local counterfactual credit audit."
    )
    parser.add_argument("--mode", choices=("smoke", "formal", "summary"), default="smoke")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--source-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-search-slots", type=int, default=200)
    parser.add_argument("--roots", type=int, default=None)
    parser.add_argument("--process-timeout-seconds", type=float, default=120.0)
    return parser


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise ValueError("--checkpoint must identify seed0/update30 for smoke")
    source_rows = _read_source_rows(args.source_decisions)
    matches = [
        row
        for row in source_rows
        if int(row["seed"]) == 0 and int(row["checkpoint_update"]) == 30
    ]
    if not matches:
        raise ValueError("source JSONL has no seed0/update30 decision")
    roots = 2 if args.roots is None else int(args.roots)
    if roots != 2:
        raise ValueError("Phase 1 smoke uses exactly two CRN roots")
    rows = _collect_checkpoint(
        checkpoint=args.checkpoint,
        seed=0,
        phase="early",
        expected_update=30,
        source_rows=[matches[0]],
        roots=roots,
        max_search_slots=int(args.max_search_slots),
        process_timeout_seconds=float(args.process_timeout_seconds),
        compare_serial_process=True,
    )
    output_dir = args.output / "smoke"
    return _write_outputs(
        output_dir=output_dir,
        decisions=rows,
        mode="smoke",
        source_path=args.source_decisions,
        experiment_root=None,
    )


def run_formal(args: argparse.Namespace) -> dict[str, Any]:
    if args.experiment_root is None or not args.experiment_root.is_dir():
        raise ValueError("--experiment-root is required for formal mode")
    source_rows = _read_source_rows(args.source_decisions)
    if len(source_rows) != 27:
        raise ValueError("Phase 1 formal audit reuses exactly 27 source decisions")
    roots = 8 if args.roots is None else int(args.roots)
    if roots != 8:
        raise ValueError("Phase 1 formal audit uses exactly eight CRN roots")
    by_checkpoint: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in source_rows:
        key = (int(row["seed"]), int(row["checkpoint_update"]))
        by_checkpoint.setdefault(key, []).append(row)
    expected_keys = {(seed, update) for seed in (0, 1, 2) for update in (30, 60, 120)}
    if set(by_checkpoint) != expected_keys:
        raise ValueError("source JSONL must cover seeds 0/1/2 and updates 30/60/120")

    output_dir = args.output / "formal"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "decision_records.partial.jsonl"
    decisions: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        run_dir = _single_path(
            args.experiment_root.glob(f"runs/seed{seed}/*"),
            f"seed {seed} run directory",
            require_directory=True,
        )
        for update in (30, 60, 120):
            checkpoint = run_dir / "checkpoints" / f"checkpoint_update_{update:04d}.pt"
            rows = _collect_checkpoint(
                checkpoint=checkpoint,
                seed=seed,
                phase=PHASE_BY_UPDATE[update],
                expected_update=update,
                source_rows=by_checkpoint[(seed, update)],
                roots=roots,
                max_search_slots=int(args.max_search_slots),
                process_timeout_seconds=float(args.process_timeout_seconds),
                compare_serial_process=False,
            )
            decisions.extend(rows)
            partial_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
                encoding="utf-8",
            )
            print(
                f"completed Phase 1 seed={seed} update={update} decisions={len(rows)}",
                flush=True,
            )
    return _write_outputs(
        output_dir=output_dir,
        decisions=decisions,
        mode="formal",
        source_path=args.source_decisions,
        experiment_root=args.experiment_root,
    )


def run_summary(args: argparse.Namespace) -> dict[str, Any]:
    decisions = _read_source_rows(args.source_decisions)
    return _write_outputs(
        output_dir=args.output,
        decisions=decisions,
        mode="summary",
        source_path=args.source_decisions,
        experiment_root=args.experiment_root,
    )


def _collect_checkpoint(
    *,
    checkpoint: Path,
    seed: int,
    phase: str,
    expected_update: int,
    source_rows: list[dict[str, Any]],
    roots: int,
    max_search_slots: int,
    process_timeout_seconds: float,
    compare_serial_process: bool,
) -> list[dict[str, Any]]:
    import torch
    from torch.distributions import Categorical

    if bool(config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES):
        raise ValueError("Scheme-B2 Phase 1 audit requires KaHyPar OFF")
    payload = _load_trusted_checkpoint(torch, checkpoint)
    if payload.get("resume_semantics") != "restart_from_new_episode_only":
        raise ValueError("checkpoint does not use the expected new-episode resume semantics")
    if int(payload.get("update_step", -1)) != int(expected_update):
        raise ValueError("checkpoint update mismatch")
    controls = checkpoint_experiment_controls(payload)
    if not bool(controls.get("offloading_decision_q_credit", False)):
        raise ValueError("checkpoint does not contain the Decision-Q experiment")
    if str(controls.get("task_encoder", "hgnn")) != "mlp":
        raise ValueError("existing Scheme-B2 replay infrastructure requires an MLP checkpoint")

    device = torch.device("cpu")
    dims = _module_dims_from_checkpoint(
        payload, argparse.Namespace(task_embedding_dim=None, hidden_dim=None)
    )
    modules = _build_modules(dims=dims, experiment_controls=controls, device=device)
    _load_module_state(modules, payload)
    _set_eval_mode(modules)
    q_state = payload.get("extra_state", {}).get("offloading_decision_q_credit")
    if q_state is None:
        raise ValueError("checkpoint is missing the Decision-Q critic")
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
    source_by_slot = {int(row["slot_index"]): row for row in source_rows}
    if len(source_by_slot) != len(source_rows):
        raise ValueError("Phase 1 source has more than one decision in a slot")
    collected: list[dict[str, Any]] = []
    try:
        for _ in range(int(max_search_slots)):
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
                        for index, uav_id in enumerate(encoded.movement_observation.uav_ids)
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
                target_records = [
                    record
                    for record in records
                    if int(record.decision_order) == int(source["decision_order"])
                    and str(record.task_id) == str(source["task_id"])
                ]
                if len(target_records) != 1:
                    raise AssertionError("checkpoint replay did not recover the source decision")
                target_record = target_records[0]
                target_trace = next(
                    row for row in trace if int(row.decision_order) == int(source["decision_order"])
                )
                state = _decision_state(
                    target_record,
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
                        torch.as_tensor(q_inputs, dtype=torch.float32, device=device)
                    ).detach().cpu().numpy()
                source_legal = [int(value) for value in source["legal_uav_ids"]]
                if legal_ids != source_legal:
                    raise AssertionError("Phase 1 replay legal candidate IDs changed")
                if "legal_q_vector" in source and not np.allclose(
                    q_values,
                    np.asarray(source["legal_q_vector"], dtype=np.float32),
                    rtol=0.0,
                    atol=1e-6,
                ):
                    raise AssertionError("Phase 1 replay Q legal vector changed")
                embeddings_by_task = {
                    str(task_id): np.asarray(
                        encoded.task_embeddings[
                            int(prepared.graph_snapshot.task_id_to_idx[str(task_id)])
                        ].detach().cpu(),
                        dtype=np.float32,
                    ).copy()
                    for task_id in prepared.frozen_ready_task_ids
                    if str(task_id) in prepared.graph_snapshot.task_id_to_idx
                }
                collected.append(
                    _collect_one_decision(
                        source=source,
                        snapshot=snapshot,
                        base_rng_root=base_rng_root,
                        trace=trace,
                        target_trace=target_trace,
                        target_record=target_record,
                        frozen_ready_task_ids=tuple(
                            str(value) for value in prepared.frozen_ready_task_ids
                        ),
                        task_embeddings_by_id=embeddings_by_task,
                        offloading_actor=modules.offloading_actor,
                        q_values=q_values,
                        q_inputs=q_inputs,
                        replay_episode_index=int(payload.get("episode", -1)) + 1,
                        roots=roots,
                        process_timeout_seconds=process_timeout_seconds,
                        compare_serial_process=compare_serial_process,
                    )
                )
            _, _, done, _ = env.commit_and_advance(assignment_buffer=assignments)
            if len(collected) == len(source_rows):
                break
            if done:
                break
    finally:
        builder.close()
    if len(collected) != len(source_rows):
        raise RuntimeError("checkpoint replay did not recover all Phase 1 source decisions")
    if not _parameters_equal(parameters_before, _clone_parameters(frozen_modules)):
        raise AssertionError("Phase 1 diagnostic changed a frozen parameter")
    for row in collected:
        row["checkpoint_path"] = str(checkpoint)
        row["gates"]["training_parameters_unchanged"] = True
    return collected


def _collect_one_decision(
    *,
    source: dict[str, Any],
    snapshot: Any,
    base_rng_root: CleanHostRngState,
    trace: tuple[CleanCounterfactualDecisionTrace, ...],
    target_trace: CleanCounterfactualDecisionTrace,
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
) -> dict[str, Any]:
    legal_ids = [int(value) for value in target_trace.legal_uav_ids]
    actual_uav_id = int(target_trace.selected_uav_id)
    alternative_uav_id, alternative_rng = _sample_uniform_alternative(
        legal_ids=legal_ids,
        actual_uav_id=actual_uav_id,
        seed=int(source["seed"]),
        update=int(source["checkpoint_update"]),
        slot_index=int(source["slot_index"]),
        decision_order=int(source["decision_order"]),
    )
    candidate_ids = [int(value) for value in target_record.candidate_uav_ids]
    actual_index = candidate_ids.index(actual_uav_id)
    alternative_index = candidate_ids.index(alternative_uav_id)
    probabilities = np.asarray(
        target_record.old_masked_probabilities.cpu(), dtype=np.float64
    )
    efts = np.asarray(
        target_record.candidate_estimated_finish_times.cpu(), dtype=np.float64
    )
    legal_probabilities = [float(probabilities[candidate_ids.index(value)]) for value in legal_ids]

    root_rows: list[dict[str, Any]] = []
    exact_decision_state_equal = True
    reference_pre_decision_reservation: dict[str, Any] | None = None
    for root_id in range(int(roots)):
        rng_root, rng_identifier = _independent_semantic_root(
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
            "actual_uav_id": actual_uav_id,
            "alternative_uav_id": alternative_uav_id,
            "frozen_ready_task_ids": frozen_ready_task_ids,
            "task_embeddings_by_id": task_embeddings_by_id,
            "offloading_actor": offloading_actor,
            "initial_rng_state": rng_root,
        }
        process_result = _run_pair_process(
            **kwargs, timeout_seconds=float(process_timeout_seconds)
        )
        serial_equal = None
        if compare_serial_process:
            serial_result = _run_pair_serial(**kwargs)
            serial_equal = bool(serial_result == process_result)
            if not serial_equal:
                raise AssertionError("Phase 1 serial/process paired results differ")
        if process_result.semantic_mismatch_count:
            raise AssertionError("Phase 1 semantic CRN mismatch")
        if process_result.unrecognized_environment_rng_calls:
            raise AssertionError("Phase 1 observed an unrecognized environment RNG call")
        actual = process_result.actual
        counterfactual = process_result.counterfactual
        if reference_pre_decision_reservation is None:
            reference_pre_decision_reservation = actual.target_pre_decision_reservation
        exact_decision_state_equal = bool(
            exact_decision_state_equal
            and actual.target_pre_decision_reservation
            == counterfactual.target_pre_decision_reservation
            and actual.target_pre_decision_reservation
            == reference_pre_decision_reservation
        )
        if not exact_decision_state_equal:
            raise AssertionError(
                "actual/counterfactual branches do not share the exact pre-decision state"
            )
        credit = None
        if not actual.censored and not counterfactual.censored:
            credit = float(actual.current_slot_reward) - float(
                counterfactual.current_slot_reward
            )
        root_rows.append(
            {
                "root_id": root_id,
                "rng_state_identifier": rng_identifier,
                "actual": asdict(actual),
                "counterfactual": asdict(counterfactual),
                "c_i_local": credit,
                "paired_root_censored": bool(actual.censored or counterfactual.censored),
                "semantic_crn": {
                    "shared_semantic_keys_checked": int(
                        process_result.shared_semantic_keys_checked
                    ),
                    "semantic_mismatch_count": int(
                        process_result.semantic_mismatch_count
                    ),
                    "unrecognized_environment_rng_calls": int(
                        process_result.unrecognized_environment_rng_calls
                    ),
                },
                "serial_process_complete_result_equal": serial_equal,
            }
        )

    q_by_id = {uav_id: float(q_values[index]) for index, uav_id in enumerate(legal_ids)}
    decision_statistics = _decision_credit_statistics(root_rows)
    return {
        "schema": "boundary_anchored_decision_credit_phase1_decision_v1",
        "state_semantics": STATE_SEMANTICS,
        "historical_training_snapshot_recovered": False,
        "diagnostic_only_not_training_label": True,
        "seed": int(source["seed"]),
        "phase": str(source["phase"]),
        "checkpoint_update": int(source["checkpoint_update"]),
        "replay_episode_index": int(replay_episode_index),
        "slot_index": int(source["slot_index"]),
        "task_id": str(source["task_id"]),
        "dag_id": str(target_record.dag_id),
        "decision_order": int(source["decision_order"]),
        "branch_root_snapshot_timing": "post-movement/pre-offloading",
        "branch_root_snapshot_identity": _snapshot_identity(snapshot),
        "exact_decision_state_timing": (
            "after baseline prefix reservations/current decision not yet applied"
        ),
        "target_pre_decision_reservation": reference_pre_decision_reservation,
        "legal_uav_ids": legal_ids,
        "actual_uav_id": actual_uav_id,
        "alternative_uav_id": alternative_uav_id,
        "alternative_proposal_probability": 1.0 / float(len(legal_ids) - 1),
        "alternative_rng": alternative_rng,
        "legal_q_vector": [q_by_id[value] for value in legal_ids],
        "q_ranking": sorted(legal_ids, key=lambda value: (-q_by_id[value], value)),
        "q_spread": float(max(q_by_id.values()) - min(q_by_id.values())),
        "q_actual_minus_alternative": float(
            q_by_id[actual_uav_id] - q_by_id[alternative_uav_id]
        ),
        "actor_legal_probabilities": legal_probabilities,
        "actor_actual_probability": float(probabilities[actual_index]),
        "actor_alternative_probability": float(probabilities[alternative_index]),
        "actor_actual_minus_alternative_probability": float(
            probabilities[actual_index] - probabilities[alternative_index]
        ),
        "actor_actual_minus_alternative_log_probability": float(
            math.log(max(float(probabilities[actual_index]), 1e-300))
            - math.log(max(float(probabilities[alternative_index]), 1e-300))
        ),
        "actual_eft": float(efts[actual_index]),
        "alternative_eft": float(efts[alternative_index]),
        "eft_advantage_actual_over_alternative": float(
            efts[alternative_index] - efts[actual_index]
        ),
        "legal_q_inputs": np.asarray(q_inputs, dtype=np.float32).tolist(),
        "root_count": int(roots),
        "root_results": root_rows,
        "credit_statistics": decision_statistics,
        "optimizer_steps": 0,
        "gates": {
            "exact_pre_decision_state_reconstructed_equal": exact_decision_state_equal,
            "single_post_movement_pre_offloading_branch_root": True,
            "current_action_forced_before_suffix": True,
            "suffix_regenerated_by_frozen_deterministic_policy": True,
            "suffix_not_fixed_to_baseline": True,
            "same_slot_gamma_steps": 0,
            "candidate_q_actor_ids_aligned": True,
            "uniform_alternative_fixed_across_roots": True,
            "collector_rng_isolated": True,
            "independent_root_seeds": len(
                {
                    int(row["rng_state_identifier"]["root_seed"])
                    for row in root_rows
                }
            )
            == len(root_rows),
            "policy_and_critics_frozen": True,
            "training_parameters_unchanged": False,
            "optimizer_steps": 0,
            "semantic_mismatch_count": sum(
                int(row["semantic_crn"]["semantic_mismatch_count"])
                for row in root_rows
            ),
            "unrecognized_environment_rng_calls": sum(
                int(row["semantic_crn"]["unrecognized_environment_rng_calls"])
                for row in root_rows
            ),
            "serial_process_complete_result_equal": (
                all(row["serial_process_complete_result_equal"] is True for row in root_rows)
                if compare_serial_process
                else None
            ),
        },
    }


def _sample_uniform_alternative(
    *,
    legal_ids: list[int],
    actual_uav_id: int,
    seed: int,
    update: int,
    slot_index: int,
    decision_order: int,
) -> tuple[int, dict[str, Any]]:
    alternatives = [value for value in legal_ids if value != int(actual_uav_id)]
    if not alternatives:
        raise ValueError("Phase 1 decision has no legal counterfactual alternative")
    sequence = np.random.SeedSequence(
        [20260902, 1, int(seed), int(update), int(slot_index), int(decision_order)]
    )
    collector_seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
    isolated_rng = np.random.RandomState(collector_seed)
    selected_index = int(isolated_rng.randint(len(alternatives)))
    return int(alternatives[selected_index]), {
        "collector_seed": collector_seed,
        "eligible_alternatives": alternatives,
        "selected_alternative_index": selected_index,
        "isolated_from_environment_and_policy_rng": True,
    }


def _run_pair_serial(
    *,
    env_snapshot: Any,
    baseline_trace: tuple[CleanCounterfactualDecisionTrace, ...],
    target_decision_order: int,
    actual_uav_id: int,
    alternative_uav_id: int,
    frozen_ready_task_ids: tuple[str, ...],
    task_embeddings_by_id: dict[str, np.ndarray],
    offloading_actor: Any,
    initial_rng_state: CleanHostRngState,
) -> LocalCreditPairResult:
    actual = _run_policy_suffix_branch(
        env_snapshot=env_snapshot,
        baseline_trace=baseline_trace,
        target_decision_order=target_decision_order,
        forced_uav_id=actual_uav_id,
        frozen_ready_task_ids=frozen_ready_task_ids,
        task_embeddings_by_id=task_embeddings_by_id,
        offloading_actor=offloading_actor,
        initial_rng_state=initial_rng_state,
    )
    counterfactual = _run_policy_suffix_branch(
        env_snapshot=env_snapshot,
        baseline_trace=baseline_trace,
        target_decision_order=target_decision_order,
        forced_uav_id=alternative_uav_id,
        frozen_ready_task_ids=frozen_ready_task_ids,
        task_embeddings_by_id=task_embeddings_by_id,
        offloading_actor=offloading_actor,
        initial_rng_state=initial_rng_state,
    )
    audit = audit_clean_semantic_common_random(
        [actual.semantic_audit, counterfactual.semantic_audit]
    )
    return LocalCreditPairResult(
        actual=actual,
        counterfactual=counterfactual,
        shared_semantic_keys_checked=int(audit.shared_semantic_keys_checked),
        semantic_mismatch_count=len(audit.semantic_key_mismatches),
        unrecognized_environment_rng_calls=int(audit.unrecognized_environment_calls),
    )


def _run_policy_suffix_branch(
    *,
    env_snapshot: Any,
    baseline_trace: tuple[CleanCounterfactualDecisionTrace, ...],
    target_decision_order: int,
    forced_uav_id: int,
    frozen_ready_task_ids: tuple[str, ...],
    task_embeddings_by_id: dict[str, np.ndarray],
    offloading_actor: Any,
    initial_rng_state: CleanHostRngState,
) -> LocalCreditBranchResult:
    import torch

    env = clone_post_movement_pre_offloading_env(env_snapshot)
    reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
    assignments = CleanAssignmentBuffer()
    trace_by_order = {int(row.decision_order): row for row in baseline_trace}
    target = trace_by_order.get(int(target_decision_order))
    if target is None:
        raise ValueError("target decision is missing from the replay trace")
    if int(forced_uav_id) not in target.legal_uav_ids:
        raise ValueError("forced action is not legal at the exact target decision")
    replayed: list[tuple[int, str, int]] = []
    suffix: list[tuple[int, str, int]] = []
    target_seen = False
    target_pre_decision_reservation: dict[str, Any] | None = None
    with torch.no_grad():
        for decision_order, task_id in enumerate(frozen_ready_task_ids):
            task = env.task_manager.get_task(task_id)
            task_embedding = task_embeddings_by_id.get(str(task_id))
            if task is None or not task.is_ready or task_embedding is None:
                continue
            dynamic, pair, mask, candidate_ids, estimates = build_offloading_candidate_components(
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
            candidate_ids = [int(value) for value in candidate_ids]
            legal_mask = np.asarray(mask, dtype=bool)
            if dynamic.shape[0] == 0 or not bool(legal_mask.any()):
                continue
            baseline = trace_by_order.get(decision_order)
            if decision_order < int(target_decision_order):
                if baseline is None:
                    raise AssertionError("exact prefix contains an unexpected actionable task")
                if (
                    tuple(candidate_ids) != baseline.candidate_uav_ids
                    or tuple(bool(value) for value in legal_mask) != baseline.candidate_mask
                ):
                    raise AssertionError("exact prefix candidate state changed")
                selected_uav_id = int(baseline.selected_uav_id)
            elif decision_order == int(target_decision_order):
                if str(task_id) != str(target.task_id):
                    raise AssertionError("target decision order resolved to a different task")
                if (
                    tuple(candidate_ids) != target.candidate_uav_ids
                    or tuple(bool(value) for value in legal_mask) != target.candidate_mask
                ):
                    raise AssertionError("exact target candidate state changed")
                selected_uav_id = int(forced_uav_id)
                target_pre_decision_reservation = _reservation_state(reservation)
                target_seen = True
            else:
                candidate_features = np.concatenate(
                    [
                        np.repeat(
                            np.asarray(task_embedding, dtype=np.float32).reshape(1, -1),
                            dynamic.shape[0],
                            axis=0,
                        ),
                        dynamic,
                        pair,
                    ],
                    axis=1,
                ).astype(np.float32)
                logits = offloading_actor.scorer(
                    torch.as_tensor(
                        candidate_features,
                        dtype=torch.float32,
                        device=next(offloading_actor.parameters()).device,
                    )
                )
                masked_logits = logits.masked_fill(
                    ~torch.as_tensor(
                        legal_mask,
                        dtype=torch.bool,
                        device=logits.device,
                    ),
                    torch.finfo(logits.dtype).min,
                )
                selected_uav_id = int(candidate_ids[int(torch.argmax(masked_logits).item())])

            if selected_uav_id not in candidate_ids:
                raise AssertionError("selected UAV disappeared from candidate ordering")
            selected_index = candidate_ids.index(selected_uav_id)
            if not bool(legal_mask[selected_index]):
                raise AssertionError("selected UAV is illegal in reconstructed reservation state")
            estimate = estimates[selected_index]
            assignments.append(str(task_id), selected_uav_id, decision_order)
            reservation.reserve(
                str(task_id),
                selected_uav_id,
                estimated_available_time=float(estimate.estimated_finish_time),
                estimated_queued_workload=float(estimate.estimated_queued_workload),
            )
            assignment = (decision_order, str(task_id), selected_uav_id)
            replayed.append(assignment)
            if decision_order > int(target_decision_order):
                suffix.append(assignment)

    if not target_seen:
        raise AssertionError("target decision was not applied")
    if target_pre_decision_reservation is None:
        raise AssertionError("target pre-decision reservation was not captured")
    common_random = CleanSemanticCommonRandom(initial_rng_state)
    with common_random.scoped_environment_calls(0):
        _, _, done, info = env.commit_and_advance(assignment_buffer=assignments)
    del done
    reward = info.get("step_reward")
    censored = reward is None or not math.isfinite(float(reward))
    latest_stats = env.executor.latest_stats
    return LocalCreditBranchResult(
        forced_uav_id=int(forced_uav_id),
        target_pre_decision_reservation=target_pre_decision_reservation,
        current_slot_reward=None if censored else float(reward),
        reward_components={
            name: float(info[name]) for name in REWARD_COMPONENT_NAMES if name in info
        },
        replayed_assignments=tuple(replayed),
        suffix_assignments=tuple(suffix),
        completed_task_ids=tuple(str(value) for value in latest_stats.completed_task_ids),
        completed_dag_ids=tuple(str(value) for value in latest_stats.completed_dag_ids),
        next_time_seconds=None if censored else float(env.current_time_seconds),
        censored=bool(censored),
        censor_reason="invalid_step_reward" if censored else None,
        semantic_audit=common_random.audit_snapshot(),
    )


def _run_pair_process(*, timeout_seconds: float, **kwargs: Any) -> LocalCreditPairResult:
    context = multiprocessing.get_context("spawn" if sys.platform == "win32" else "fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_pair_process_worker,
        args=(child, kwargs),
        name="phase1-local-credit-pair",
    )
    process.start()
    child.close()
    try:
        if not parent.poll(float(timeout_seconds)):
            raise TimeoutError("Phase 1 paired branch process timed out")
        payload = parent.recv()
        process.join(timeout=float(timeout_seconds))
        if process.is_alive():
            raise TimeoutError("Phase 1 paired branch process did not exit")
        if int(process.exitcode or 0) != 0 or not payload.get("ok", False):
            raise RuntimeError(f"Phase 1 paired branch failed: {payload.get('error')}")
        return payload["result"]
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


def _pair_process_worker(connection: Any, kwargs: dict[str, Any]) -> None:
    try:
        connection.send({"ok": True, "result": _run_pair_serial(**kwargs)})
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _decision_credit_statistics(root_rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in root_rows if row["c_i_local"] is not None]
    values = np.asarray([float(row["c_i_local"]) for row in usable], dtype=np.float64)
    actual = np.asarray(
        [float(row["actual"]["current_slot_reward"]) for row in usable],
        dtype=np.float64,
    )
    counterfactual = np.asarray(
        [float(row["counterfactual"]["current_slot_reward"]) for row in usable],
        dtype=np.float64,
    )
    mean = None if values.size == 0 else float(values.mean())
    std = None if values.size < 2 else float(values.std(ddof=1))
    se = None if std is None else float(std / math.sqrt(values.size))
    half_width = None if se is None else _t_critical_975(values.size - 1) * se
    reward_scale = (
        None
        if values.size == 0
        else float(np.median(np.maximum(np.abs(actual), np.abs(counterfactual))))
    )
    near_zero_tolerance = (
        None if reward_scale is None else max(1e-12, 1e-6 * reward_scale)
    )
    positive_fraction = None if values.size == 0 else float(np.mean(values > 0.0))
    negative_fraction = None if values.size == 0 else float(np.mean(values < 0.0))
    return {
        "configured_root_count": len(root_rows),
        "usable_root_count": int(values.size),
        "censor_fraction": 1.0 - float(values.size) / max(len(root_rows), 1),
        "mean_c_i_local": mean,
        "std_c_i_local": std,
        "variance_c_i_local": None if std is None else float(std * std),
        "se_c_i_local": se,
        "ci95_c_i_local": (
            None if half_width is None else [mean - half_width, mean + half_width]
        ),
        "mean_abs_c_i_local": (
            None if values.size == 0 else float(np.mean(np.abs(values)))
        ),
        "actual_reward_mean": None if values.size == 0 else float(actual.mean()),
        "counterfactual_reward_mean": (
            None if values.size == 0 else float(counterfactual.mean())
        ),
        "positive_fraction": positive_fraction,
        "negative_fraction": negative_fraction,
        "sign_consistency": (
            None
            if values.size == 0
            else float(max(positive_fraction, negative_fraction))
        ),
        "exact_zero_fraction": (
            None if values.size == 0 else float(np.mean(values == 0.0))
        ),
        "reward_scale_median": reward_scale,
        "near_zero_tolerance": near_zero_tolerance,
        "near_zero_fraction": (
            None
            if values.size == 0
            else float(np.mean(np.abs(values) <= float(near_zero_tolerance)))
        ),
        "ci_resolved_away_from_zero": bool(
            half_width is not None and (mean - half_width > 0.0 or mean + half_width < 0.0)
        ),
    }


def _summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase = {
        phase: {
            "pooled": _aggregate(selected),
            "by_seed": {
                str(seed): _aggregate(
                    [row for row in selected if int(row["seed"]) == seed]
                )
                for seed in (0, 1, 2)
            },
        }
        for phase in ("early", "mid", "late")
        for selected in [[row for row in decisions if str(row["phase"]) == phase]]
    }
    return {
        "schema": "boundary_anchored_decision_credit_phase1_summary_v1",
        "state_semantics": STATE_SEMANTICS,
        "historical_training_snapshot_recovered": False,
        "diagnostic_only_not_training_labels": True,
        "decision_count": len(decisions),
        "root_budget": sorted({int(row["root_count"]) for row in decisions}),
        "pooled": _aggregate(decisions),
        "by_seed": {
            str(seed): _aggregate([row for row in decisions if int(row["seed"]) == seed])
            for seed in (0, 1, 2)
        },
        "by_phase": by_phase,
        "gates": {
            "optimizer_steps_total": sum(int(row["optimizer_steps"]) for row in decisions),
            "all_diagnostic_only": all(
                bool(row["diagnostic_only_not_training_label"]) for row in decisions
            ),
            "all_parameters_unchanged": all(
                bool(row["gates"]["training_parameters_unchanged"])
                for row in decisions
            ),
            "all_exact_decision_states_equal": all(
                bool(row["gates"]["exact_pre_decision_state_reconstructed_equal"])
                for row in decisions
            ),
            "all_suffixes_policy_regenerated": all(
                bool(row["gates"]["suffix_regenerated_by_frozen_deterministic_policy"])
                for row in decisions
            ),
            "same_slot_gamma_steps": sorted(
                {int(row["gates"]["same_slot_gamma_steps"]) for row in decisions}
            ),
            "semantic_mismatch_count": sum(
                int(row["gates"]["semantic_mismatch_count"]) for row in decisions
            ),
            "unrecognized_environment_rng_calls": sum(
                int(row["gates"]["unrecognized_environment_rng_calls"])
                for row in decisions
            ),
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row["credit_statistics"]["mean_c_i_local"] is not None
    ]
    means = np.asarray(
        [float(row["credit_statistics"]["mean_c_i_local"]) for row in usable],
        dtype=np.float64,
    )
    within_variances = [
        float(row["credit_statistics"]["variance_c_i_local"])
        for row in usable
        if row["credit_statistics"]["variance_c_i_local"] is not None
    ]
    configured_roots = sum(int(row["root_count"]) for row in rows)
    usable_roots = sum(
        int(row["credit_statistics"]["usable_root_count"]) for row in rows
    )
    q_differences = [float(row["q_actual_minus_alternative"]) for row in usable]
    eft_differences = [
        float(row["eft_advantage_actual_over_alternative"]) for row in usable
    ]
    actor_differences = [
        float(row["actor_actual_minus_alternative_log_probability"]) for row in usable
    ]
    credits = [float(value) for value in means]
    return {
        "selected_decision_count": len(rows),
        "usable_decision_count": len(usable),
        "configured_root_count": configured_roots,
        "usable_root_count": usable_roots,
        "censor_fraction": 1.0 - usable_roots / max(configured_roots, 1),
        "mean_c_i_local": None if means.size == 0 else float(means.mean()),
        "std_across_decision_means": (
            None if means.size < 2 else float(means.std(ddof=1))
        ),
        "variance_across_decision_means": (
            None if means.size < 2 else float(means.var(ddof=1))
        ),
        "median_c_i_local": None if means.size == 0 else float(np.median(means)),
        "iqr_c_i_local": (
            None
            if means.size == 0
            else [float(np.quantile(means, 0.25)), float(np.quantile(means, 0.75))]
        ),
        "mean_abs_c_i_local": (
            None if means.size == 0 else float(np.mean(np.abs(means)))
        ),
        "mean_within_decision_root_variance": (
            None if not within_variances else float(np.mean(within_variances))
        ),
        "ci_resolved_decision_fraction": (
            None
            if not usable
            else float(
                np.mean(
                    [
                        bool(row["credit_statistics"]["ci_resolved_away_from_zero"])
                        for row in usable
                    ]
                )
            )
        ),
        "mean_sign_consistency": (
            None
            if not usable
            else float(
                np.mean(
                    [float(row["credit_statistics"]["sign_consistency"]) for row in usable]
                )
            )
        ),
        "mean_near_zero_fraction": (
            None
            if not usable
            else float(
                np.mean(
                    [float(row["credit_statistics"]["near_zero_fraction"]) for row in usable]
                )
            )
        ),
        "relations": {
            "credit_vs_eft_advantage": _relation(credits, eft_differences),
            "credit_vs_q_difference": _relation(credits, q_differences),
            "credit_vs_actor_log_probability_difference": _relation(
                credits, actor_differences
            ),
        },
        "selected_alternative_groups": _group_pairs(usable),
    }


def _relation(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        return {"count": 0, "pearson": None, "spearman": None}
    pearson = None
    if len(left) >= 2 and np.std(left) > 0.0 and np.std(right) > 0.0:
        pearson = float(np.corrcoef(left, right)[0, 1])
    return {
        "count": len(left),
        "pearson": pearson,
        "spearman": _spearman(left, right),
    }


def _group_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        key = f"{int(row['actual_uav_id'])}->{int(row['alternative_uav_id'])}"
        groups.setdefault(key, []).append(
            float(row["credit_statistics"]["mean_c_i_local"])
        )
    return {
        key: {
            "decision_count": len(values),
            "mean_c_i_local": float(np.mean(values)),
            "std_c_i_local": None if len(values) < 2 else float(np.std(values, ddof=1)),
        }
        for key, values in sorted(groups.items())
    }


def _snapshot_identity(env: Any) -> dict[str, Any]:
    return {
        "prepared_slot_open": bool(getattr(env, "_prepared_slot_open", False)),
        "movement_action_count": int(getattr(env, "_last_movement_action_count", 0)),
        "current_time_seconds": float(env.current_time_seconds),
        "uav_service_positions": {
            str(key): [float(value[0]), float(value[1])]
            for key, value in sorted(env.uav_service_positions.items())
        },
        "executor_queue_lengths": {
            str(int(uav.id)): len(env.executor.uav_queues.get(int(uav.id), []))
            for uav in env.uavs
        },
    }


def _reservation_state(reservation: TemporaryReservationState) -> dict[str, Any]:
    return {
        "queue_lengths": {
            str(key): int(value) for key, value in sorted(reservation.queue_lengths.items())
        },
        "available_times": {
            str(key): float(value)
            for key, value in sorted(reservation.available_times.items())
        },
        "queued_workloads": {
            str(key): float(value)
            for key, value in sorted(reservation.queued_workloads.items())
        },
        "slot_assigned_counts": {
            str(key): int(value)
            for key, value in sorted(reservation.slot_assigned_counts.items())
        },
        "reserved_task_ids": sorted(str(value) for value in reservation.reserved_task_ids),
    }


def _write_outputs(
    *,
    output_dir: Path,
    decisions: list[dict[str, Any]],
    mode: str,
    source_path: Path,
    experiment_root: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / "decision_records.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "run_manifest.json"
    summary = _summary(decisions)
    decision_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema": "boundary_anchored_decision_credit_phase1_manifest_v1",
        "mode": str(mode),
        "state_semantics": STATE_SEMANTICS,
        "source_decisions": str(source_path),
        "experiment_root": None if experiment_root is None else str(experiment_root),
        "checkpoint_paths": sorted(
            {
                str(row["checkpoint_path"])
                for row in decisions
                if "checkpoint_path" in row
            }
        ),
        "decision_count": len(decisions),
        "root_counts": sorted({int(row["root_count"]) for row in decisions}),
        "diagnostic_only_not_training_labels": True,
        "optimizer_steps": 0,
        "execution": {
            "branch_mode": "process",
            "smoke_serial_process_comparison": str(mode) == "smoke",
            "same_slot_gamma_steps": 0,
            "suffix_policy": "frozen_deterministic_masked_argmax",
        },
        "decision_roots_and_alternatives": [
            {
                "seed": int(row["seed"]),
                "checkpoint_update": int(row["checkpoint_update"]),
                "slot_index": int(row["slot_index"]),
                "decision_order": int(row["decision_order"]),
                "actual_uav_id": int(row["actual_uav_id"]),
                "alternative_uav_id": int(row["alternative_uav_id"]),
                "alternative_rng": row["alternative_rng"],
                "roots": [
                    {
                        "root_id": int(root["root_id"]),
                        "rng_state_identifier": root["rng_state_identifier"],
                    }
                    for root in row["root_results"]
                ],
            }
            for row in decisions
        ],
        "gates": summary["gates"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "decision_jsonl": str(decision_path),
        "summary_json": str(summary_path),
        "manifest_json": str(manifest_path),
        "decision_count": len(decisions),
        "summary": summary,
    }


def _read_source_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _single_path(paths: Any, label: str, *, require_directory: bool = False) -> Path:
    values = [path for path in paths if not require_directory or path.is_dir()]
    if len(values) != 1:
        raise ValueError(f"expected one {label}, found {len(values)}")
    return values[0]


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "smoke":
        result = run_smoke(args)
    elif args.mode == "formal":
        result = run_formal(args)
    else:
        result = run_summary(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
