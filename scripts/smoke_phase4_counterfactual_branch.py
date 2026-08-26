from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.assignment import CleanAssignmentBuffer
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.hgnn.clean_incidence import build_clean_task_encoder
from marl_models.mappo.clean_counterfactual_oracle import (
    capture_clean_counterfactual_baseline_trace,
    clone_post_movement_pre_offloading_env,
    run_clean_counterfactual_branch,
    run_clean_counterfactual_branches_process,
    run_clean_counterfactual_branches_serial,
    select_first_multicandidate_decision,
)
from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
from marl_models.mappo.clean_slot_orchestrator import (
    encode_prepared_slot,
    prepare_slot_state,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_baseline_fixture(*, seed: int = 42) -> tuple[Env, tuple, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _assert(
        not bool(config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES),
        "Scheme-B0 smoke requires KaHyPar OFF",
    )
    env = Env()
    env.reset()
    graph_builder = CleanGraphBuilder()
    actor = CleanOffloadingActor(task_embedding_dim=64, hidden_dim=64)
    for parameter in actor.parameters():
        parameter.data.zero_()
    encoder = None
    try:
        for _ in range(100):
            prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
            if encoder is None:
                encoder = build_clean_task_encoder(
                    encoder_type="mlp",
                    task_feature_dim=int(prepared.graph_snapshot.task_features.shape[1]),
                    hidden_dim=128,
                    output_dim=64,
                )
            encoded = encode_prepared_slot(
                prepared_state=prepared,
                env=env,
                hgnn=encoder,
                device="cpu",
            )
            env.apply_movement({})
            env_snapshot = clone_post_movement_pre_offloading_env(env)
            frozen_tasks = [
                env.task_manager.get_task(task_id)
                for task_id in prepared.frozen_ready_task_ids
            ]
            frozen_tasks = [task for task in frozen_tasks if task is not None]
            assignment_buffer = actor.act(
                frozen_ready_tasks=frozen_tasks,
                task_embeddings=encoded.task_embeddings,
                graph_snapshot=prepared.graph_snapshot,
                task_manager=env.task_manager,
                uavs=env.uavs,
                executor=env.executor,
                current_time_seconds=env.current_time_seconds,
                uav_service_positions=env.uav_service_positions,
                ue_service_positions=env.ue_service_positions,
                ues=env.ues,
                deterministic=True,
            )
            trace = capture_clean_counterfactual_baseline_trace(
                decision_records=actor.latest_records,
                assignment_buffer=assignment_buffer,
            )
            if any(len(item.legal_uav_ids) >= 2 for item in trace):
                return env_snapshot, trace, assignment_buffer
            env.commit_and_advance(assignment_buffer=assignment_buffer)
    finally:
        graph_builder.close()
    raise AssertionError("could not construct a multi-candidate MLP baseline decision")


def test_clone_independence() -> None:
    snapshot, _, _ = _build_baseline_fixture()
    original_uav_position = snapshot.uavs[0].pos.copy()
    original_queue = list(snapshot.executor.uav_queues[0])
    clone_a = clone_post_movement_pre_offloading_env(snapshot)
    clone_b = clone_post_movement_pre_offloading_env(snapshot)
    clone_a.uavs[0].pos[0] += 7.0
    clone_a.executor.uav_queues[0].append("clone-a-only")
    _assert(
        np.array_equal(snapshot.uavs[0].pos, original_uav_position),
        "mutating clone A changed original UAV state",
    )
    _assert(
        np.array_equal(clone_b.uavs[0].pos, original_uav_position),
        "mutating clone A changed clone B UAV state",
    )
    _assert(snapshot.executor.uav_queues[0] == original_queue, "clone A changed original queue")
    _assert(clone_b.executor.uav_queues[0] == original_queue, "clone A changed clone B queue")
    print("Test 1 — Clone independence: PASS")


def test_identical_branch() -> None:
    snapshot, trace, _ = _build_baseline_fixture()
    target = select_first_multicandidate_decision(trace)
    kwargs = {
        "env_snapshot": snapshot,
        "baseline_trace": trace,
        "target_decision_order": target.decision_order,
        "forced_uav_id": target.selected_uav_id,
    }
    first = run_clean_counterfactual_branch(**kwargs)
    second = run_clean_counterfactual_branch(**kwargs)
    _assert(first == second, "identical branches produced different current-slot outcomes")
    _assert(first.suffix_replay_feasible, "baseline-identical branch must be feasible")
    print("Test 2 — Identical branch: PASS")


def test_single_intervention_and_censor() -> None:
    snapshot, trace, _ = _build_baseline_fixture()
    target = select_first_multicandidate_decision(trace)
    outcomes = run_clean_counterfactual_branches_serial(
        env_snapshot=snapshot,
        baseline_trace=trace,
        target_decision_order=target.decision_order,
    )
    baseline_assignments = tuple(
        (item.decision_order, item.task_id, item.selected_uav_id) for item in trace
    )
    for outcome in outcomes:
        if not outcome.suffix_replay_feasible:
            continue
        changed = [
            (baseline, replayed)
            for baseline, replayed in zip(baseline_assignments, outcome.replayed_assignments)
            if baseline != replayed
        ]
        expected_change_count = int(outcome.forced_uav_id != outcome.baseline_uav_id)
        _assert(
            len(changed) == expected_change_count,
            "a feasible branch changed more than the target decision",
        )
        if changed:
            _assert(
                int(changed[0][0][0]) == target.decision_order,
                "the only changed assignment was not the target decision",
            )

    original_capacity = int(config.CLEAN_MAX_QUEUE_PER_UAV)
    try:
        config.CLEAN_MAX_QUEUE_PER_UAV = 1
        capped_snapshot, capped_trace, _ = _build_baseline_fixture(seed=86)
        censor_case = None
        for candidate in capped_trace[:-1]:
            later_uavs = [
                item.selected_uav_id
                for item in capped_trace
                if item.decision_order > candidate.decision_order
            ]
            forced_options = [
                uav_id
                for uav_id in candidate.legal_uav_ids
                if uav_id != candidate.selected_uav_id and uav_id in later_uavs
            ]
            if forced_options:
                censor_case = (candidate, forced_options[0])
                break
        _assert(censor_case is not None, "failed to construct an infeasible suffix fixture")
        capped_target, forced_uav = censor_case
        censored = run_clean_counterfactual_branch(
            env_snapshot=capped_snapshot,
            baseline_trace=capped_trace,
            target_decision_order=capped_target.decision_order,
            forced_uav_id=forced_uav,
        )
        _assert(not censored.suffix_replay_feasible, "illegal suffix was not censored")
        _assert(censored.current_slot_reward is None, "censored branch must not commit")
        _assert(
            len(censored.replayed_assignments) < len(capped_trace),
            "censored branch silently replanned the suffix",
        )
    finally:
        config.CLEAN_MAX_QUEUE_PER_UAV = original_capacity
    print("Test 3 — Single intervention and suffix censor: PASS")


def test_serial_vs_parallel() -> tuple[list[dict], list[dict]]:
    snapshot, trace, _ = _build_baseline_fixture(seed=1042)
    target = select_first_multicandidate_decision(trace)
    serial = run_clean_counterfactual_branches_serial(
        env_snapshot=snapshot,
        baseline_trace=trace,
        target_decision_order=target.decision_order,
    )
    parallel = run_clean_counterfactual_branches_process(
        env_snapshot=snapshot,
        baseline_trace=trace,
        target_decision_order=target.decision_order,
    )
    _assert(serial == parallel, "serial and spawn-process branch outcomes differ")
    print("Test 4 — Serial vs parallel: PASS")
    return (
        [item.minimal_output() for item in serial],
        [item.minimal_output() for item in parallel],
    )


def main() -> int:
    test_clone_independence()
    test_identical_branch()
    test_single_intervention_and_censor()
    serial, parallel = test_serial_vs_parallel()
    print("Serial example:")
    print(json.dumps(serial[:2], ensure_ascii=False, sort_keys=True))
    print("Parallel example:")
    print(json.dumps(parallel[:2], ensure_ascii=False, sort_keys=True))
    print("Phase4 Scheme-B0 same-snapshot branching smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
