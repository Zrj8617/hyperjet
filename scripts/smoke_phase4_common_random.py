from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.hgnn.clean_incidence import build_clean_task_encoder
from marl_models.mappo.clean_counterfactual_oracle import (
    capture_clean_counterfactual_baseline_trace,
    clone_post_movement_pre_offloading_env,
    select_first_multicandidate_decision,
)
from marl_models.mappo.clean_counterfactual_oracle_common_random import (
    CleanSemanticCommonRandom,
    run_clean_common_random_completion_process,
    run_clean_common_random_completion_serial,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import capture_clean_host_rng_state
from marl_models.mappo.clean_movement_actor import CleanMovementActor
from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
from marl_models.mappo.clean_slot_orchestrator import encode_prepared_slot, prepare_slot_state


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_semantic_determinism() -> None:
    np.random.seed(42)
    root = capture_clean_host_rng_state()
    left = CleanSemanticCommonRandom(root)
    right = CleanSemanticCommonRandom(root)
    left_values = [
        left.draw(future_slot=3, ue_id=7, subsystem="mobility", function="normal")
        for _ in range(2)
    ]
    right_values = [
        right.draw(future_slot=3, ue_id=7, subsystem="mobility", function="normal")
        for _ in range(2)
    ]
    _assert(left_values == right_values, "same semantic key produced different streams")
    print("Test 1 — semantic determinism: PASS")


def test_skip_isolation() -> None:
    np.random.seed(42)
    root = capture_clean_host_rng_state()
    branch_a = CleanSemanticCommonRandom(root)
    branch_b = CleanSemanticCommonRandom(root)
    branch_a.draw(future_slot=5, ue_id=21, subsystem="arrival", function="random")
    a_value = branch_a.draw(
        future_slot=5, ue_id=23, subsystem="arrival", function="random"
    )
    b_value = branch_b.draw(
        future_slot=5, ue_id=23, subsystem="arrival", function="random"
    )
    _assert(a_value == b_value, "skipping another semantic key shifted arrival input")
    print("Test 2 — skip isolation: PASS")


def test_generation_isolation() -> None:
    np.random.seed(42)
    root = capture_clean_host_rng_state()
    branch_a = CleanSemanticCommonRandom(root)
    branch_b = CleanSemanticCommonRandom(root)
    branch_a.draw(
        future_slot=8,
        ue_id=36,
        subsystem="dag_generation",
        function="randint",
        args=(4, 10),
    )
    branch_a.draw(
        future_slot=8,
        ue_id=36,
        subsystem="dag_generation",
        function="uniform",
        args=(1.0, 4.0),
    )
    for subsystem, function in (("mobility", "normal"), ("arrival", "random")):
        a_value = branch_a.draw(
            future_slot=9,
            ue_id=23,
            subsystem=subsystem,
            function=function,
        )
        b_value = branch_b.draw(
            future_slot=9,
            ue_id=23,
            subsystem=subsystem,
            function=function,
        )
        _assert(a_value == b_value, "extra DAG generation shifted an independent key")
    print("Test 3 — generation isolation: PASS")


def test_serial_spawn() -> None:
    snapshot, trace, root, encoder, movement_actor, offloading_actor = _fixture()
    target = select_first_multicandidate_decision(trace)
    kwargs = {
        "env_snapshot": snapshot,
        "baseline_trace": trace,
        "target_decision_order": target.decision_order,
        "initial_rng_state": root,
        "gamma": 0.99,
        "task_encoder": encoder,
        "movement_actor": movement_actor,
        "offloading_actor": offloading_actor,
        "max_future_slots": 2,
        "device": "cpu",
    }
    original_random = np.random.random
    serial = run_clean_common_random_completion_serial(**kwargs)
    _assert(np.random.random is original_random, "NumPy wrapper was not restored after serial")
    parallel = run_clean_common_random_completion_process(**kwargs)
    _assert(serial.decision == parallel.decision, "serial/spawn CRN outcomes differ")
    _assert(not serial.audit.semantic_key_mismatches, "serial semantic audit mismatch")
    _assert(not parallel.audit.semantic_key_mismatches, "spawn semantic audit mismatch")
    _assert(serial.audit.unrecognized_environment_calls == 0, "unrecognized serial RNG call")
    _assert(parallel.audit.unrecognized_environment_calls == 0, "unrecognized spawn RNG call")
    print("Test 4 — serial / spawn: PASS")


def _fixture() -> tuple[Env, tuple, object, object, object, object]:
    random.seed(1042)
    np.random.seed(1042)
    torch.manual_seed(1042)
    _assert(not bool(config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES), "KaHyPar must be OFF")
    env = Env()
    env.reset()
    builder = CleanGraphBuilder()
    encoder = None
    movement_actor = CleanMovementActor(task_embedding_dim=64, hidden_dim=64)
    offloading_actor = CleanOffloadingActor(task_embedding_dim=64, hidden_dim=64)
    for module in (movement_actor, offloading_actor):
        for parameter in module.parameters():
            parameter.data.zero_()
        module.eval()
    try:
        for _ in range(100):
            prepared = prepare_slot_state(env=env, graph_builder=builder)
            if encoder is None:
                encoder = build_clean_task_encoder(
                    encoder_type="mlp",
                    task_feature_dim=int(prepared.graph_snapshot.task_features.shape[1]),
                    hidden_dim=128,
                    output_dim=64,
                )
                encoder.eval()
            encoded = encode_prepared_slot(
                prepared_state=prepared,
                env=env,
                hgnn=encoder,
                movement_actor=movement_actor,
                device="cpu",
            )
            movement = torch.argmax(encoded.movement_logits, dim=-1)
            env.apply_movement(
                {
                    int(uav_id): int(movement[index].item())
                    for index, uav_id in enumerate(encoded.movement_observation.uav_ids)
                }
            )
            snapshot = clone_post_movement_pre_offloading_env(env)
            root = capture_clean_host_rng_state()
            ready = [env.task_manager.get_task(task_id) for task_id in prepared.frozen_ready_task_ids]
            ready = [task for task in ready if task is not None and task.is_ready]
            assignments = offloading_actor.act(
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
                deterministic=True,
            )
            trace = capture_clean_counterfactual_baseline_trace(
                decision_records=offloading_actor.latest_records,
                assignment_buffer=assignments,
            )
            if any(len(item.legal_uav_ids) >= 2 for item in trace):
                return snapshot, trace, root, encoder, movement_actor, offloading_actor
            env.commit_and_advance(assignment_buffer=assignments)
    finally:
        builder.close()
    raise AssertionError("failed to build common-random smoke fixture")


def main() -> int:
    test_semantic_determinism()
    test_skip_isolation()
    test_generation_isolation()
    test_serial_spawn()
    print("Phase4 Scheme-B2 semantic common-random smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
