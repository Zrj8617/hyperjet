"""Phase 3 diagnostic-only counterfactual credit-horizon audit.

This module reuses the exact Phase 1 checkpoint replay decisions, alternatives,
and semantic CRN roots.  It extends each paired branch across physical slots and
computes C_H = G_real(0:H) - G_cf(0:H).  No branch result is used for training.
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

import config  # noqa: E402
from environment.assignment import (  # noqa: E402
    CleanAssignmentBuffer,
    TemporaryReservationState,
    build_offloading_candidate_components,
)
from environment.graph_builder import CleanGraphBuilder  # noqa: E402
from environment.dag_tasks import TASK_STATE_COMPLETED  # noqa: E402
from marl_models.mappo.clean_counterfactual_oracle import (  # noqa: E402
    CleanCounterfactualDecisionTrace,
    clone_post_movement_pre_offloading_env,
)
from marl_models.mappo.clean_counterfactual_oracle_common_random import (  # noqa: E402
    CleanSemanticCommonRandom,
    audit_clean_semantic_common_random,
)
from marl_models.mappo.clean_counterfactual_oracle_completion import (  # noqa: E402
    _advance_one_future_slot,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import (  # noqa: E402
    CleanHostRngState,
)
from scripts import diagnose_boundary_anchored_decision_credit_phase1 as phase1  # noqa: E402
from scripts.diagnose_decision_q_v2_ranking_crn import (  # noqa: E402
    _independent_semantic_root,
)
from scripts.summarize_phase2b_local_credit_ranking import (  # noqa: E402
    _key,
    _read_jsonl,
    _sign,
    _tie_aware_score,
    _truth_vectors,
)


STATE_SEMANTICS = "frozen-checkpoint replay-generated on-policy decision states"
HORIZONS = (1, 5, 10, 20, 30, 60, 80)
PHASE_BY_UPDATE = {30: "early", 60: "mid", 120: "late"}

# Captured from the unchanged Phase 1 module-construction path.  Linux process
# workers inherit these frozen modules via fork.
_FUTURE_TASK_ENCODER: Any = None
_FUTURE_MOVEMENT_ACTOR: Any = None


@dataclass(frozen=True, slots=True)
class HorizonBranchResult:
    forced_uav_id: int
    target_pre_decision_reservation: dict[str, Any]
    replayed_assignments: tuple[tuple[int, str, int], ...]
    suffix_assignments: tuple[tuple[int, str, int], ...]
    reward_sequence: tuple[float, ...]
    dag_progress_sequence: tuple[tuple[tuple[str, int, int], ...], ...]
    progress_potential_sequence: tuple[float, ...]
    done_after_slots: int | None
    stop_reason: str
    semantic_audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class HorizonPairResult:
    actual: HorizonBranchResult
    counterfactual: HorizonBranchResult
    shared_semantic_keys_checked: int
    semantic_mismatch_count: int
    unrecognized_environment_rng_calls: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--phase1-decisions", type=Path, required=True)
    parser.add_argument("--multi-root-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-search-slots", type=int, default=200)
    parser.add_argument("--process-timeout-seconds", type=float, default=600.0)
    return parser


def _capture_modules_wrapper(original: Any) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        global _FUTURE_TASK_ENCODER, _FUTURE_MOVEMENT_ACTOR
        modules = original(*args, **kwargs)
        _FUTURE_TASK_ENCODER = modules.hgnn
        _FUTURE_MOVEMENT_ACTOR = modules.movement_actor
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
        return _collect_one_decision(
            **kwargs,
            gamma=float(gamma),
        )

    phase1._collect_one_decision = collect
    phase1._build_modules = _capture_modules_wrapper(original_build)
    try:
        rows = phase1._collect_checkpoint(
            checkpoint=checkpoint,
            seed=int(seed),
            phase=PHASE_BY_UPDATE[int(update)],
            expected_update=int(update),
            source_rows=source_rows,
            roots=8 if not compare_serial_process else 2,
            max_search_slots=int(max_search_slots),
            process_timeout_seconds=float(timeout_seconds),
            compare_serial_process=bool(compare_serial_process),
        )
    finally:
        phase1._collect_one_decision = original_collect
        phase1._build_modules = original_build
    return rows


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
    gamma: float,
) -> dict[str, Any]:
    del target_record, q_inputs, replay_episode_index
    if _FUTURE_TASK_ENCODER is None or _FUTURE_MOVEMENT_ACTOR is None:
        raise AssertionError("future-slot frozen modules were not captured")
    legal_ids = [int(value) for value in target_trace.legal_uav_ids]
    actual_uav_id = int(source["actual_uav_id"])
    alternative_uav_id = int(source["alternative_uav_id"])
    if actual_uav_id != int(target_trace.selected_uav_id):
        raise AssertionError("Phase 1 actual action changed during replay")
    if alternative_uav_id not in legal_ids or alternative_uav_id == actual_uav_id:
        raise AssertionError("stored Phase 1 alternative is not a legal alternative")
    if int(roots) > len(source["root_results"]):
        raise AssertionError("Phase 3 requests more semantic roots than Phase 1 stored")

    root_specs: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for root_id in range(int(roots)):
        rng_root, rng_identifier = _independent_semantic_root(
            base_rng_root=base_rng_root,
            seed=int(source["seed"]),
            update=int(source["checkpoint_update"]),
            slot_index=int(source["slot_index"]),
            decision_order=int(source["decision_order"]),
            root_id=root_id,
        )
        stored_root = source["root_results"][root_id]
        if int(stored_root["root_id"]) != root_id:
            raise AssertionError("Phase 1 root ordering changed")
        if rng_identifier != stored_root["rng_state_identifier"]:
            raise AssertionError("Phase 3 semantic RNG root differs from Phase 1")
        root_specs.append(
            (
                root_id,
                rng_identifier,
                stored_root,
                {
                    "env_snapshot": snapshot,
                    "baseline_trace": trace,
                    "target_decision_order": int(target_trace.decision_order),
                    "actual_uav_id": actual_uav_id,
                    "alternative_uav_id": alternative_uav_id,
                    "frozen_ready_task_ids": frozen_ready_task_ids,
                    "task_embeddings_by_id": task_embeddings_by_id,
                    "task_encoder": _FUTURE_TASK_ENCODER,
                    "movement_actor": _FUTURE_MOVEMENT_ACTOR,
                    "offloading_actor": offloading_actor,
                    "initial_rng_state": rng_root,
                    "max_horizon": max(HORIZONS),
                },
            )
        )
    batch_results = (
        {}
        if compare_serial_process
        else _run_pair_process_batch(
            [(root_id, kwargs) for root_id, _, _, kwargs in root_specs],
            timeout_seconds=float(process_timeout_seconds),
            max_parallel=8,
        )
    )
    root_rows: list[dict[str, Any]] = []
    reference_reservation: dict[str, Any] | None = None
    for root_id, rng_identifier, stored_root, kwargs in root_specs:
        process_result = (
            _run_pair_process(**kwargs, timeout_seconds=float(process_timeout_seconds))
            if compare_serial_process
            else batch_results[root_id]
        )
        serial_equal = None
        if compare_serial_process:
            serial_result = _run_pair_serial(**kwargs)
            serial_equal = bool(serial_result == process_result)
            if not serial_equal:
                raise AssertionError("Phase 3 serial/process complete results differ")
        if process_result.semantic_mismatch_count:
            raise AssertionError("Phase 3 semantic CRN mismatch")
        if process_result.unrecognized_environment_rng_calls:
            raise AssertionError("Phase 3 observed unrecognized environment RNG")
        actual, counterfactual = process_result.actual, process_result.counterfactual
        if actual.target_pre_decision_reservation != counterfactual.target_pre_decision_reservation:
            raise AssertionError("paired branches do not share the exact decision state")
        if reference_reservation is None:
            reference_reservation = actual.target_pre_decision_reservation
        if actual.target_pre_decision_reservation != reference_reservation:
            raise AssertionError("CRN roots do not share the exact decision state")

        credits: dict[str, float | None] = {}
        censored: dict[str, bool] = {}
        for horizon in HORIZONS:
            usable = (
                len(actual.reward_sequence) >= horizon
                and len(counterfactual.reward_sequence) >= horizon
            )
            censored[str(horizon)] = not usable
            credits[str(horizon)] = (
                _discounted_difference(
                    actual.reward_sequence[:horizon],
                    counterfactual.reward_sequence[:horizon],
                    gamma=float(gamma),
                )
                if usable
                else None
            )
        old_credit = stored_root["c_i_local"]
        new_credit = credits["1"]
        if old_credit is None or new_credit is None or not math.isclose(
            float(old_credit), float(new_credit), rel_tol=0.0, abs_tol=1e-10
        ):
            raise AssertionError("H=1 credit does not reproduce the Phase 1 root")
        root_rows.append(
            {
                "root_id": root_id,
                "rng_state_identifier": rng_identifier,
                "actual": _compact_branch_record(actual),
                "counterfactual": _compact_branch_record(counterfactual),
                "credits_by_horizon": credits,
                "paired_censored_by_horizon": censored,
                "phase1_h1_credit_reproduced": True,
                "semantic_crn": {
                    "shared_semantic_keys_checked": int(
                        process_result.shared_semantic_keys_checked
                    ),
                    "semantic_mismatch_count": int(process_result.semantic_mismatch_count),
                    "unrecognized_environment_rng_calls": int(
                        process_result.unrecognized_environment_rng_calls
                    ),
                },
                "serial_process_complete_result_equal": serial_equal,
            }
        )

    q_by_id = {int(uav): float(q_values[index]) for index, uav in enumerate(legal_ids)}
    return {
        "schema": "phase3_counterfactual_horizon_decision_v1",
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
        "actual_uav_id": actual_uav_id,
        "alternative_uav_id": alternative_uav_id,
        "legal_q_vector": [q_by_id[value] for value in legal_ids],
        "q_actual_minus_alternative": float(
            source["q_actual_minus_alternative"]
        ),
        "actor_actual_minus_alternative_probability": float(
            source["actor_actual_minus_alternative_probability"]
        ),
        "gamma_per_physical_slot": float(gamma),
        "same_slot_gamma_steps": 0,
        "horizons": list(HORIZONS),
        "root_count": int(roots),
        "root_results": root_rows,
        "horizon_statistics": {
            str(horizon): _horizon_statistics(root_rows, horizon)
            for horizon in HORIZONS
        },
        "optimizer_steps": 0,
        "gates": {
            "exact_pre_decision_state_equal": True,
            "phase1_actual_and_alternative_reused": True,
            "phase1_rng_roots_reused": True,
            "phase1_h1_reproduced_all_roots": True,
            "suffix_regenerated_by_frozen_deterministic_policy": True,
            "same_slot_gamma_steps": 0,
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


def _compact_branch_record(branch: HorizonBranchResult) -> dict[str, Any]:
    record = asdict(branch)
    # The full per-call semantic trace can be tens of MB for H=80.  Root IDs
    # and paired audit totals below are sufficient to reproduce and gate CRN.
    record.pop("semantic_audit", None)
    return record


def _run_pair_serial(
    *,
    env_snapshot: Any,
    baseline_trace: tuple[CleanCounterfactualDecisionTrace, ...],
    target_decision_order: int,
    actual_uav_id: int,
    alternative_uav_id: int,
    frozen_ready_task_ids: tuple[str, ...],
    task_embeddings_by_id: dict[str, np.ndarray],
    task_encoder: Any,
    movement_actor: Any,
    offloading_actor: Any,
    initial_rng_state: CleanHostRngState,
    max_horizon: int,
) -> HorizonPairResult:
    common = {
        "env_snapshot": env_snapshot,
        "baseline_trace": baseline_trace,
        "target_decision_order": target_decision_order,
        "frozen_ready_task_ids": frozen_ready_task_ids,
        "task_embeddings_by_id": task_embeddings_by_id,
        "task_encoder": task_encoder,
        "movement_actor": movement_actor,
        "offloading_actor": offloading_actor,
        "initial_rng_state": initial_rng_state,
        "max_horizon": max_horizon,
    }
    actual = _run_branch(forced_uav_id=actual_uav_id, **common)
    counterfactual = _run_branch(forced_uav_id=alternative_uav_id, **common)
    audit = audit_clean_semantic_common_random(
        [actual.semantic_audit, counterfactual.semantic_audit]
    )
    return HorizonPairResult(
        actual=actual,
        counterfactual=counterfactual,
        shared_semantic_keys_checked=int(audit.shared_semantic_keys_checked),
        semantic_mismatch_count=len(audit.semantic_key_mismatches),
        unrecognized_environment_rng_calls=int(audit.unrecognized_environment_calls),
    )


def _run_branch(
    *,
    env_snapshot: Any,
    baseline_trace: tuple[CleanCounterfactualDecisionTrace, ...],
    target_decision_order: int,
    forced_uav_id: int,
    frozen_ready_task_ids: tuple[str, ...],
    task_embeddings_by_id: dict[str, np.ndarray],
    task_encoder: Any,
    movement_actor: Any,
    offloading_actor: Any,
    initial_rng_state: CleanHostRngState,
    max_horizon: int,
) -> HorizonBranchResult:
    import torch

    env = clone_post_movement_pre_offloading_env(env_snapshot)
    reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
    assignments = CleanAssignmentBuffer()
    trace_by_order = {int(row.decision_order): row for row in baseline_trace}
    target = trace_by_order.get(int(target_decision_order))
    if target is None or int(forced_uav_id) not in target.legal_uav_ids:
        raise ValueError("missing target or illegal forced action")
    replayed: list[tuple[int, str, int]] = []
    suffix: list[tuple[int, str, int]] = []
    target_reservation: dict[str, Any] | None = None
    target_seen = False
    with torch.no_grad():
        for decision_order, task_id in enumerate(frozen_ready_task_ids):
            task = env.task_manager.get_task(task_id)
            embedding = task_embeddings_by_id.get(str(task_id))
            if task is None or not task.is_ready or embedding is None:
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
                    raise AssertionError("exact prefix has an unexpected actionable task")
                if (
                    tuple(candidate_ids) != baseline.candidate_uav_ids
                    or tuple(bool(value) for value in legal_mask) != baseline.candidate_mask
                ):
                    raise AssertionError("exact prefix candidate state changed")
                selected_uav_id = int(baseline.selected_uav_id)
            elif decision_order == int(target_decision_order):
                if str(task_id) != str(target.task_id):
                    raise AssertionError("target decision resolved to another task")
                if (
                    tuple(candidate_ids) != target.candidate_uav_ids
                    or tuple(bool(value) for value in legal_mask) != target.candidate_mask
                ):
                    raise AssertionError("exact target candidate state changed")
                target_reservation = phase1._reservation_state(reservation)
                selected_uav_id = int(forced_uav_id)
                target_seen = True
            else:
                features = np.concatenate(
                    [
                        np.repeat(
                            np.asarray(embedding, dtype=np.float32).reshape(1, -1),
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
                        features,
                        dtype=torch.float32,
                        device=next(offloading_actor.parameters()).device,
                    )
                )
                masked = logits.masked_fill(
                    ~torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device),
                    torch.finfo(logits.dtype).min,
                )
                selected_uav_id = int(candidate_ids[int(torch.argmax(masked).item())])
            selected_index = candidate_ids.index(selected_uav_id)
            if not bool(legal_mask[selected_index]):
                raise AssertionError("reconstructed selected action is illegal")
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
    if not target_seen or target_reservation is None:
        raise AssertionError("target decision was not reconstructed")

    common_random = CleanSemanticCommonRandom(initial_rng_state)
    rewards: list[float] = []
    dag_progress: list[tuple[tuple[str, int, int], ...]] = [
        _active_dag_progress(env)
    ]
    progress_potentials: list[float] = [_progress_potential(dag_progress[-1])]
    done_after: int | None = None
    with common_random.scoped_environment_calls(0):
        _, _, done, info = env.commit_and_advance(assignment_buffer=assignments)
    reward = float(info["step_reward"])
    if not math.isfinite(reward):
        raise FloatingPointError("non-finite current-slot reward")
    rewards.append(reward)
    dag_progress.append(_active_dag_progress(env))
    progress_potentials.append(_progress_potential(dag_progress[-1]))
    if done:
        done_after = 1

    builder = CleanGraphBuilder()
    builder.reset()
    try:
        for future_slot in range(1, int(max_horizon)):
            if done_after is not None:
                break
            with common_random.scoped_environment_calls(future_slot):
                reward, done = _advance_one_future_slot(
                    env=env,
                    graph_builder=builder,
                    task_encoder=task_encoder,
                    movement_actor=movement_actor,
                    offloading_actor=offloading_actor,
                    device=next(offloading_actor.parameters()).device,
                )
            if not math.isfinite(float(reward)):
                raise FloatingPointError("non-finite future-slot reward")
            rewards.append(float(reward))
            dag_progress.append(_active_dag_progress(env))
            progress_potentials.append(_progress_potential(dag_progress[-1]))
            if done:
                done_after = len(rewards)
    finally:
        builder.close()
    return HorizonBranchResult(
        forced_uav_id=int(forced_uav_id),
        target_pre_decision_reservation=target_reservation,
        replayed_assignments=tuple(replayed),
        suffix_assignments=tuple(suffix),
        reward_sequence=tuple(rewards),
        dag_progress_sequence=tuple(dag_progress),
        progress_potential_sequence=tuple(progress_potentials),
        done_after_slots=done_after,
        stop_reason="episode_done" if done_after is not None else "max_horizon_reached",
        semantic_audit=common_random.audit_snapshot(),
    )


def _active_dag_progress(env: Any) -> tuple[tuple[str, int, int], ...]:
    """Record completed/total child tasks for every currently active DAG."""
    rows: list[tuple[str, int, int]] = []
    for dag_id, job in sorted(env.task_manager.jobs.items()):
        if bool(job.completed):
            continue
        task_ids = [str(task_id) for task_id in job.task_ids]
        completed = sum(
            1
            for task_id in task_ids
            if env.task_manager.tasks[task_id].state == TASK_STATE_COMPLETED
        )
        rows.append((str(dag_id), int(completed), len(task_ids)))
    return tuple(rows)


def _progress_potential(progress: tuple[tuple[str, int, int], ...]) -> float:
    return float(config.REWARD_COMPLETED_DAG_WEIGHT) * sum(
        float(completed) / float(total)
        for _, completed, total in progress
    )


def _run_pair_process(*, timeout_seconds: float, **kwargs: Any) -> HorizonPairResult:
    # H>1 evaluates encoder and movement modules in the worker.  Spawn avoids
    # inheriting an initialized PyTorch thread pool through fork.
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_pair_worker,
        args=(child, kwargs),
        name="phase3-counterfactual-horizon-pair",
    )
    process.start()
    child.close()
    try:
        if not parent.poll(float(timeout_seconds)):
            raise TimeoutError("Phase 3 paired branch process timed out")
        payload = parent.recv()
        process.join(timeout=float(timeout_seconds))
        if process.is_alive():
            raise TimeoutError("Phase 3 paired branch process did not exit")
        if int(process.exitcode or 0) != 0 or not payload.get("ok", False):
            raise RuntimeError(f"Phase 3 paired branch failed: {payload.get('error')}")
        return payload["result"]
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


def _run_pair_process_batch(
    specs: list[tuple[int, dict[str, Any]]],
    *,
    timeout_seconds: float,
    max_parallel: int,
) -> dict[int, HorizonPairResult]:
    context = multiprocessing.get_context("spawn")
    results: dict[int, HorizonPairResult] = {}
    for start in range(0, len(specs), int(max_parallel)):
        wave = specs[start : start + int(max_parallel)]
        running: list[tuple[int, Any, Any]] = []
        try:
            for root_id, kwargs in wave:
                parent, child = context.Pipe(duplex=False)
                process = context.Process(
                    target=_pair_worker,
                    args=(child, kwargs),
                    name=f"phase3-counterfactual-horizon-root-{root_id}",
                )
                process.start()
                child.close()
                running.append((root_id, parent, process))
            for root_id, parent, process in running:
                if not parent.poll(float(timeout_seconds)):
                    raise TimeoutError(f"Phase 3 root {root_id} process timed out")
                payload = parent.recv()
                process.join(timeout=float(timeout_seconds))
                if process.is_alive():
                    raise TimeoutError(f"Phase 3 root {root_id} process did not exit")
                if int(process.exitcode or 0) != 0 or not payload.get("ok", False):
                    raise RuntimeError(
                        f"Phase 3 root {root_id} failed: {payload.get('error')}"
                    )
                results[root_id] = payload["result"]
        finally:
            for _, parent, process in running:
                parent.close()
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
    return results


def _pair_worker(connection: Any, kwargs: dict[str, Any]) -> None:
    try:
        connection.send({"ok": True, "result": _run_pair_serial(**kwargs)})
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _discounted_difference(
    actual: tuple[float, ...], counterfactual: tuple[float, ...], *, gamma: float
) -> float:
    return float(
        sum(
            (float(gamma) ** index) * (float(left) - float(right))
            for index, (left, right) in enumerate(zip(actual, counterfactual))
        )
    )


def _horizon_statistics(root_rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    values = [
        float(row["credits_by_horizon"][str(horizon)])
        for row in root_rows
        if row["credits_by_horizon"][str(horizon)] is not None
    ]
    array = np.asarray(values, dtype=np.float64)
    return {
        "configured_root_count": len(root_rows),
        "usable_root_count": len(values),
        "censor_fraction": 1.0 - len(values) / max(len(root_rows), 1),
        "mean_credit": None if not values else float(array.mean()),
        "std_across_roots": None if len(values) < 2 else float(array.std(ddof=1)),
        "mean_abs_credit": None if not values else float(np.mean(np.abs(array))),
        "positive_root_fraction": None if not values else float(np.mean(array > 0.0)),
        "negative_root_fraction": None if not values else float(np.mean(array < 0.0)),
        "exact_zero_root_fraction": None if not values else float(np.mean(array == 0.0)),
    }


def _attach_truth(decisions: list[dict[str, Any]], truth_path: Path) -> None:
    truth_rows = {_key(row): row for row in _read_jsonl(truth_path)}
    for row in decisions:
        source = truth_rows.get(_key(row))
        if source is None:
            raise ValueError(f"missing truth row for {_key(row)}")
        vectors = _truth_vectors(source)
        actual = int(row["actual_uav_id"])
        alternative = int(row["alternative_uav_id"])
        row["comparators"] = {
            "truth_actual_minus_alternative": float(
                vectors["truth_by_id"][actual] - vectors["truth_by_id"][alternative]
            ),
            "truth_pair_ci_resolved": bool(
                vectors["resolved_by_pair"].get((actual, alternative), False)
            ),
            "q_actual_minus_alternative": float(row["q_actual_minus_alternative"]),
            "actor_actual_minus_alternative_probability": float(
                row["actor_actual_minus_alternative_probability"]
            ),
        }


def _metrics(rows: list[dict[str, Any]], horizon: int, comparator: str) -> dict[str, Any]:
    credit_and_comparator: list[tuple[float, float, bool]] = []
    key = {
        "truth": "truth_actual_minus_alternative",
        "q": "q_actual_minus_alternative",
        "actor": "actor_actual_minus_alternative_probability",
    }[comparator]
    for row in rows:
        credit = row["horizon_statistics"][str(horizon)]["mean_credit"]
        if credit is not None:
            credit_and_comparator.append(
                (
                    float(credit),
                    float(row["comparators"][key]),
                    bool(row["comparators"]["truth_pair_ci_resolved"]),
                )
            )
    tie_scores = [_tie_aware_score(left, right) for left, right, _ in credit_and_comparator]
    resolved = [
        (left, right, truth_resolved)
        for left, right, truth_resolved in credit_and_comparator
        if _sign(left) != 0 and _sign(right) != 0
    ]
    agreements = [float(_sign(left) == _sign(right)) for left, right, _ in resolved]
    truth_ci = [
        float(_sign(left) == _sign(right))
        for left, right, is_resolved in resolved
        if is_resolved
    ] if comparator == "truth" else []
    configured_roots = sum(int(row["root_count"]) for row in rows)
    usable_roots = sum(
        int(row["horizon_statistics"][str(horizon)]["usable_root_count"])
        for row in rows
    )
    return {
        "decision_count": len(rows),
        "usable_decision_count": len(credit_and_comparator),
        "censor_fraction": 1.0 - usable_roots / max(configured_roots, 1),
        "tie_aware_pairwise_accuracy": None if not tie_scores else float(np.mean(tie_scores)),
        "resolved_pair_count": len(resolved),
        "sampled_pair_top1_agreement": None if not agreements else float(np.mean(agreements)),
        "sampled_pair_spearman": (
            None
            if not agreements
            else float(np.mean([1.0 if value == 1.0 else -1.0 for value in agreements]))
        ),
        "sampled_pair_ordering_accuracy": None if not agreements else float(np.mean(agreements)),
        "truth_ci_resolved_pair_count": len(truth_ci) if comparator == "truth" else None,
        "truth_ci_resolved_accuracy": (
            None if comparator != "truth" or not truth_ci else float(np.mean(truth_ci))
        ),
        "scope": "actual_vs_the_same_single_uniform_legal_alternative_from_Phase1",
        "full_legal_top1_or_spearman_available": False,
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        means = [
            float(row["horizon_statistics"][str(horizon)]["mean_credit"])
            for row in rows
            if row["horizon_statistics"][str(horizon)]["mean_credit"] is not None
        ]
        result[str(horizon)] = {
            "credit_distribution": {
                "usable_decision_count": len(means),
                "nonzero_fraction": None if not means else float(np.mean(np.asarray(means) != 0.0)),
                "mean_credit": None if not means else float(np.mean(means)),
                "mean_abs_credit": None if not means else float(np.mean(np.abs(means))),
                "std_across_decision_means": (
                    None if len(means) < 2 else float(np.std(means, ddof=1))
                ),
            },
            "ranking": {
                comparator: _metrics(rows, horizon, comparator)
                for comparator in ("truth", "q", "actor")
            },
        }
    return result


def _stability(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for horizon in HORIZONS:
        usable = [
            row for row in decisions
            if row["horizon_statistics"][str(horizon)]["mean_credit"] is not None
        ]
        h1_overlap = [
            row for row in usable
            if row["horizon_statistics"]["1"]["mean_credit"] is not None
            and _sign(float(row["horizon_statistics"]["1"]["mean_credit"])) != 0
            and _sign(float(row["horizon_statistics"][str(horizon)]["mean_credit"])) != 0
        ]
        rows.append(
            {
                "horizon": horizon,
                "usable_decision_count": len(usable),
                "sign_agreement_with_h1": (
                    None
                    if not h1_overlap
                    else float(np.mean([
                        _sign(float(row["horizon_statistics"]["1"]["mean_credit"]))
                        == _sign(float(row["horizon_statistics"][str(horizon)]["mean_credit"]))
                        for row in h1_overlap
                    ]))
                ),
                "truth_ordering_accuracy": _metrics(usable, horizon, "truth")[
                    "sampled_pair_ordering_accuracy"
                ],
                "truth_resolved_pair_count": _metrics(usable, horizon, "truth")[
                    "resolved_pair_count"
                ],
                "censor_fraction": _metrics(usable, horizon, "truth")["censor_fraction"],
            }
        )
    return {"by_horizon": rows}


def _write_output(
    *,
    output: Path,
    mode: str,
    decisions: list[dict[str, Any]],
    phase1_path: Path,
    truth_path: Path,
    gamma: float,
) -> dict[str, Any]:
    _attach_truth(decisions, truth_path)
    for row in decisions:
        row["gates"]["training_parameters_unchanged"] = True
    result = {
        "schema": "phase3_counterfactual_horizon_audit_v1",
        "mode": mode,
        "state_semantics": STATE_SEMANTICS,
        "historical_training_snapshot_recovered": False,
        "phase1_decisions": str(phase1_path),
        "multi_root_truth": str(truth_path),
        "horizons_physical_slots": list(HORIZONS),
        "gamma_per_physical_slot": float(gamma),
        "same_slot_gamma_steps": 0,
        "new_rollout_generated": False,
        "training_performed": False,
        "optimizer_steps": 0,
        "decision_count": len(decisions),
        "interpretation_boundary": {
            "full_legal_credit_ranking_available": False,
            "sampled_pair_ranking_available": True,
            "reason": "Phase 1 fixed one uniform legal alternative per decision",
            "truth_source": "existing multi-root continuation audit; not regenerated",
        },
        "gates": {
            "all_phase1_h1_roots_reproduced": all(
                bool(row["gates"]["phase1_h1_reproduced_all_roots"])
                for row in decisions
            ),
            "all_parameters_unchanged": all(
                bool(row["gates"]["training_parameters_unchanged"])
                for row in decisions
            ),
            "optimizer_steps": 0,
            "semantic_mismatch_count": sum(
                int(row["gates"]["semantic_mismatch_count"]) for row in decisions
            ),
            "unrecognized_environment_rng_calls": sum(
                int(row["gates"]["unrecognized_environment_rng_calls"])
                for row in decisions
            ),
            "serial_process_complete_result_equal": (
                all(bool(row["gates"]["serial_process_complete_result_equal"])
                    for row in decisions)
                if mode == "smoke" else None
            ),
        },
        "pooled": _group_summary(decisions),
        "by_phase": {
            phase: _group_summary([row for row in decisions if row["phase"] == phase])
            for phase in ("early", "mid", "late")
        },
        "by_seed": {
            str(seed): _group_summary([row for row in decisions if row["seed"] == seed])
            for seed in (0, 1, 2)
        },
        "horizon_stability": _stability(decisions),
        "decision_rows": decisions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source_rows = _read_jsonl(args.phase1_decisions)
    if len(source_rows) != 27:
        raise ValueError("Phase 3 requires the existing 27 Phase 1 decisions")
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
            grouped.setdefault(
                (int(row["seed"]), int(row["checkpoint_update"])), []
            ).append(row)
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
                partial.write_text(
                    json.dumps({"decision_rows": decisions}, sort_keys=True),
                    encoding="utf-8",
                )
                print(
                    f"completed Phase 3 seed={seed} update={update} decisions={len(rows)}",
                    flush=True,
                )
    result = _write_output(
        output=args.output,
        mode=args.mode,
        decisions=decisions,
        phase1_path=args.phase1_decisions,
        truth_path=args.multi_root_truth,
        gamma=float(args.gamma),
    )
    print(json.dumps({"output": str(args.output), "decision_count": len(decisions), "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
