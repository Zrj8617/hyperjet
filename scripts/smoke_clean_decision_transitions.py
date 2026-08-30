from __future__ import annotations

from pathlib import Path
import math
import random
import tempfile
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_decision_transitions import (
    CleanDecisionTransitionTracker,
)
from marl_models.mappo.clean_slot_orchestrator import CleanSlotRolloutBuffer


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _record(name: str, order: int, probabilities: np.ndarray | None = None) -> SimpleNamespace:
    probs = (
        np.asarray([0.25, 0.75], dtype=np.float32)
        if probabilities is None
        else probabilities
    )
    return SimpleNamespace(
        task_id=name,
        task_local_index=order,
        dag_id=f"dag_{name}",
        decision_order=order,
        selected_action=1,
        selected_uav_id=1,
        candidate_uav_ids=[0, 1],
        candidate_mask=np.asarray([True, True]),
        candidate_features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        critic_global_context=np.asarray([5.0, 6.0], dtype=np.float32),
        old_masked_probabilities=probs,
        old_log_probability=float(math.log(float(probs[1]))),
    )


def _unit_transition_checks() -> None:
    tracker = CleanDecisionTransitionTracker(gamma=0.9)
    tracker.start_episode(0)
    tracker.record_decisions(slot_index=3, records=[_record("d1", 0), _record("d2", 1)])
    first = tracker.completed_transitions[0]
    _assert(first.rho == 0.0 and first.delta == 0, "Test 1 same-slot rho/delta mismatch")
    _assert(first.next_state is not None and first.next_state.task_id == "d2", "Test 1 next decision mismatch")
    print("Test 1 PASS: rho=0.0 delta=0 next=d2")

    tracker.record_slot_reward(2.0)
    tracker.record_decisions(slot_index=4, records=[_record("d3", 0)])
    second = tracker.completed_transitions[1]
    _assert(np.isclose(second.rho, 2.0) and second.delta == 1, "Test 2 one-slot transition mismatch")
    print("Test 2 PASS: rho=2.0 delta=1")

    tracker.record_slot_reward(3.0)
    tracker.record_decisions(slot_index=5, records=[])
    tracker.record_slot_reward(4.0)
    tracker.record_decisions(slot_index=6, records=[_record("d4", 0)])
    third = tracker.completed_transitions[2]
    expected = 3.0 + 0.9 * 4.0
    _assert(np.isclose(third.rho, expected) and third.delta == 2, "Test 3 no-decision slot mismatch")
    print(f"Test 3 PASS: rho={third.rho:.6f} delta=2")

    conservation = CleanDecisionTransitionTracker(gamma=1.0)
    conservation.start_episode(0)
    conservation.record_decisions(slot_index=0, records=[_record("c1", 0), _record("c2", 1)])
    conservation.record_slot_reward(1.5)
    conservation.record_decisions(slot_index=1, records=[])
    conservation.record_slot_reward(-0.5)
    conservation.record_decisions(slot_index=2, records=[_record("c3", 0)])
    mapped = conservation.completed_transitions
    _assert([row.rho for row in mapped] == [0.0, 1.0], "Test 4 reward conservation mismatch")
    print("Test 4 PASS: c1->c2 maps []; c2->c3 maps [1.5,-0.5], sum=1.0")

    boundary = CleanDecisionTransitionTracker(gamma=0.5)
    boundary.start_episode(0)
    old_buffer = CleanSlotRolloutBuffer()
    boundary.record_decisions(slot_index=0, records=[_record("b1", 0)])
    boundary.record_slot_reward(8.0)
    old_buffer.close(bootstrap_value=0.0)
    new_buffer = CleanSlotRolloutBuffer()
    _assert(old_buffer is not new_buffer and boundary.pending, "Test 5 pending lost at buffer rollover")
    boundary.record_decisions(slot_index=1, records=[_record("b2", 0)])
    crossed = boundary.completed_transitions[0]
    _assert(np.isclose(crossed.rho, 8.0) and crossed.delta == 1, "Test 5 cross-buffer transition mismatch")
    _assert(not hasattr(old_buffer, "decision_transitions"), "Test 5 transitions must not live in slot buffer")
    print("Test 5 PASS: pending survived independent buffer replacement")

    on_policy_boundary = CleanDecisionTransitionTracker(gamma=0.5)
    on_policy_boundary.start_episode(0)
    on_policy_boundary.record_decisions(slot_index=0, records=[_record("g1", 0)])
    on_policy_boundary.record_slot_reward(8.0)
    on_policy_boundary.close_rollout_boundary()
    censored = on_policy_boundary.completed_transitions[0]
    _assert(
        censored.truncated and censored.unresolved and not on_policy_boundary.pending,
        "Test 5B rollout boundary did not censor and clear pending",
    )
    on_policy_boundary.record_decisions(slot_index=1, records=[_record("g2", 0)])
    _assert(on_policy_boundary.pending, "Test 5B rollout boundary ended the episode")
    print("Test 5B PASS: decision-GAE boundary censors pending without ending episode")

    terminal = CleanDecisionTransitionTracker(gamma=0.9)
    terminal.start_episode(0)
    terminal.record_decisions(slot_index=0, records=[_record("t1", 0)])
    terminal.record_slot_reward(7.0)
    terminal.close_terminated()
    terminal_row = terminal.completed_transitions[0]
    _assert(terminal_row.terminated and terminal_row.next_state is None, "Test 6 terminal flags mismatch")
    _assert(terminal_row.future_bootstrap == 0.0, "Test 6 terminal bootstrap must be zero")
    print("Test 6 PASS: terminal=True next=None future_bootstrap=0")

    truncated = CleanDecisionTransitionTracker(gamma=0.9)
    truncated.start_episode(0)
    truncated.record_decisions(slot_index=0, records=[_record("x1", 0)])
    truncated.record_slot_reward(6.0)
    truncated.close_truncated()
    truncated_row = truncated.completed_transitions[0]
    _assert(not truncated_row.terminated and truncated_row.truncated, "Test 7 truncation treated as terminal")
    _assert(truncated_row.unresolved and truncated_row.future_bootstrap is None, "Test 7 truncation must be censored")
    print("Test 7 PASS: truncated=True unresolved=True terminated=False")

    source_probs = np.asarray([0.4, 0.6], dtype=np.float32)
    frozen = CleanDecisionTransitionTracker(gamma=0.9)
    frozen.start_episode(0)
    frozen.record_decisions(slot_index=0, records=[_record("p1", 0, source_probs)])
    source_probs[:] = [0.9, 0.1]
    frozen.record_decisions(slot_index=0, records=[_record("p2", 1)])
    stored = frozen.completed_transitions[0].state.old_masked_probabilities
    _assert(np.allclose(stored, [0.4, 0.6]) and not stored.flags.writeable, "Test 8 behavior probabilities were not frozen")
    print("Test 8 PASS: rollout old_masked_probabilities frozen at [0.4,0.6]")


def _center_scene(env: Env) -> None:
    center = np.asarray([250.0, 250.0], dtype=np.float32)
    env.hotspot_center = center.copy()
    for index, uav in enumerate(env.uavs):
        uav.pos[:2] = center + np.asarray([float(index), 0.0], dtype=np.float32)
    for ue in env.ues:
        ue.pos[:2] = center.copy()


def _add_dag(env: Env, ue_index: int) -> None:
    ue = env.ues[ue_index]
    job = env.task_manager.create_dag_for_ue(
        ue_id=ue.id,
        source_pos=ue.pos[:2].copy(),
        current_time_step=env.current_time_seconds,
    )
    ue.enter_service_waiting(job.dag_id)
    env.task_manager.refresh_ready_states()


def _build_modules(task_feature_dim: int, *, torch: object, include_q: bool = True):
    from marl_models.hgnn import CleanIncidenceHGNN
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_action_value import (
        build_rng_neutral_clean_counterfactual_q,
    )
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim
    from marl_models.mappo.clean_trainer import CleanTrainingModules

    embedding_dim = 8
    hidden_dim = 16
    offloading_actor = CleanOffloadingActor(
        task_embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    critic_dim = clean_critic_input_dim(embedding_dim, config.NUM_UAVS)
    return CleanTrainingModules(
        hgnn=CleanIncidenceHGNN(
            task_feature_dim=task_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
        ),
        movement_actor=CleanMovementActor(
            task_embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ),
        offloading_actor=offloading_actor,
        critic=CleanCentralizedCritic(input_dim=critic_dim, hidden_dim=hidden_dim),
        offloading_action_value_critic=(
            build_rng_neutral_clean_counterfactual_q(
                input_dim=int(offloading_actor.candidate_feature_dim) + int(critic_dim),
                hidden_dim=hidden_dim,
            )
            if include_q
            else None
        ),
    )


def _module_snapshot(modules: object, *, torch: object) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for module_name in (
        "hgnn",
        "movement_actor",
        "offloading_actor",
        "critic",
        "offloading_action_value_critic",
    ):
        module = getattr(modules, module_name)
        if module is None:
            continue
        for key, value in module.state_dict().items():
            snapshot[f"{module_name}.{key}"] = value.detach().cpu().clone()
    return snapshot


def _run_disabled_signature(*, explicit_none: bool) -> dict[str, object]:
    import torch
    from torch.distributions import Categorical

    from environment.dag_tasks import TASK_STATE_READY_UNSCHEDULED
    from marl_models.mappo.clean_slot_orchestrator import encode_prepared_slot, prepare_slot_state
    from marl_models.mappo.clean_trainer import (
        CleanPPOUpdateConfig,
        CleanPPOUpdater,
        build_single_optimizer,
        close_rollout_with_bootstrap,
    )
    from scripts.train_clean_mainline import _collect_clean_slot

    random.seed(701)
    np.random.seed(701)
    torch.manual_seed(701)
    env = Env()
    env.reset()
    _center_scene(env)
    _add_dag(env, 0)
    graph_builder = CleanGraphBuilder()
    prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
    modules = _build_modules(prepared.graph_snapshot.task_features.shape[1], torch=torch)
    initial = _module_snapshot(modules, torch=torch)
    encoded = encode_prepared_slot(
        prepared_state=prepared,
        env=env,
        hgnn=modules.hgnn,
        critic=modules.critic,
        movement_actor=modules.movement_actor,
        device="cpu",
    )
    kwargs = dict(
        env=env,
        modules=modules,
        encoded_state=encoded,
        categorical_cls=Categorical,
        device="cpu",
        task_state_ready=TASK_STATE_READY_UNSCHEDULED,
        freeze_movement=False,
        lagged_q_enabled=False,
    )
    if explicit_none:
        kwargs["decision_transition_tracker"] = None
    slot, done, _ = _collect_clean_slot(**kwargs)
    slot.terminated = bool(done)
    slot.truncated = False
    buffer = CleanSlotRolloutBuffer()
    buffer.append(slot)
    next_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
    next_encoded = encode_prepared_slot(
        prepared_state=next_prepared,
        env=env,
        hgnn=modules.hgnn,
        critic=modules.critic,
        movement_actor=modules.movement_actor,
        device="cpu",
    )
    close_rollout_with_bootstrap(
        buffer=buffer,
        next_encoded_state=next_encoded,
        terminated=bool(done),
    )
    updater = CleanPPOUpdater(
        modules=modules,
        optimizer=build_single_optimizer(modules, lr=1e-3),
        config=CleanPPOUpdateConfig(
            ppo_epochs=1,
            clean_counterfactual_credit=True,
        ),
        device="cpu",
    )
    stats = updater.update(buffer)
    signature = {
        "initial": initial,
        "movement_actions": [row.selected_action for row in slot.movement_records],
        "offloading_actions": [row.selected_action for row in slot.offloading_records],
        "movement_log_probs": [row.old_log_probability for row in slot.movement_records],
        "offloading_log_probs": [row.old_log_probability for row in slot.offloading_records],
        "reward": slot.reward,
        "terminated": slot.terminated,
        "truncated": slot.truncated,
        "loss": stats.total_loss,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().clone(),
    }
    graph_builder.close()
    return signature


def _rng_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _behavior_equivalence_check() -> None:
    import torch
    from scripts.train_clean_mainline import build_arg_parser

    args = build_arg_parser().parse_args([])
    _assert(args.record_decision_transitions is False, "Test 9 gate default must be disabled")
    _assert(args.offloading_decision_gae is False, "Test 9 decision-GAE default must be disabled")
    legacy = _run_disabled_signature(explicit_none=False)
    compatible = _run_disabled_signature(explicit_none=True)
    _assert(legacy["initial"].keys() == compatible["initial"].keys(), "Test 9 module key mismatch")
    for key, value in legacy["initial"].items():
        _assert(torch.equal(value, compatible["initial"][key]), f"Test 9 initial parameter mismatch: {key}")
    for key in (
        "movement_actions",
        "offloading_actions",
        "movement_log_probs",
        "offloading_log_probs",
        "reward",
        "terminated",
        "truncated",
        "loss",
        "python_rng",
    ):
        _assert(legacy[key] == compatible[key], f"Test 9 behavior mismatch: {key}")
    _assert(_rng_equal(legacy["numpy_rng"], compatible["numpy_rng"]), "Test 9 NumPy RNG mismatch")
    _assert(torch.equal(legacy["torch_rng"], compatible["torch_rng"]), "Test 9 Torch RNG mismatch")
    print("Test 9 PASS: params/actions/log-probs/reward/flags/PPO loss/Python-NumPy-Torch RNG identical")


def _mainline_integration_check() -> None:
    import torch
    from torch.distributions import Categorical

    from environment.dag_tasks import TASK_STATE_READY_UNSCHEDULED
    from marl_models.mappo.clean_slot_orchestrator import encode_prepared_slot, prepare_slot_state
    from marl_models.mappo.clean_trainer import (
        CleanPPOUpdateConfig,
        CleanPPOUpdater,
        build_single_optimizer,
        close_rollout_with_bootstrap,
        reencode_prepared_after_update,
    )
    from scripts.train_clean_mainline import _collect_clean_slot

    random.seed(809)
    np.random.seed(809)
    torch.manual_seed(809)
    env = Env()
    env.reset()
    _center_scene(env)
    _add_dag(env, 0)
    graph_builder = CleanGraphBuilder()
    first_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
    modules = _build_modules(
        first_prepared.graph_snapshot.task_features.shape[1],
        torch=torch,
        include_q=False,
    )
    optimizer = build_single_optimizer(modules, lr=1e-3)
    updater = CleanPPOUpdater(
        modules=modules,
        optimizer=optimizer,
        config=CleanPPOUpdateConfig(ppo_epochs=1),
        device="cpu",
    )
    tracker = CleanDecisionTransitionTracker(gamma=0.9)
    tracker.start_episode(0)
    first_encoded = encode_prepared_slot(
        prepared_state=first_prepared,
        env=env,
        hgnn=modules.hgnn,
        critic=modules.critic,
        movement_actor=modules.movement_actor,
        device="cpu",
    )
    first_slot, first_done, _ = _collect_clean_slot(
        env=env,
        modules=modules,
        encoded_state=first_encoded,
        categorical_cls=Categorical,
        device="cpu",
        task_state_ready=TASK_STATE_READY_UNSCHEDULED,
        freeze_movement=False,
        decision_transition_tracker=tracker,
    )
    _assert(first_slot.offloading_records and tracker.pending, "Test 10 first real decision missing")
    first_buffer = CleanSlotRolloutBuffer()
    first_buffer.append(first_slot)

    _add_dag(env, 1)
    second_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
    second_encoded_old = encode_prepared_slot(
        prepared_state=second_prepared,
        env=env,
        hgnn=modules.hgnn,
        critic=modules.critic,
        movement_actor=modules.movement_actor,
        device="cpu",
    )
    close_rollout_with_bootstrap(
        buffer=first_buffer,
        next_encoded_state=second_encoded_old,
        terminated=bool(first_done),
    )
    updater.update(first_buffer)
    second_buffer = CleanSlotRolloutBuffer()
    _assert(first_buffer is not second_buffer and tracker.pending, "Test 10 pending did not survive PPO rollover")
    second_encoded = reencode_prepared_after_update(
        prepared_state=second_prepared,
        env=env,
        modules=modules,
        device="cpu",
    )
    second_slot, _, _ = _collect_clean_slot(
        env=env,
        modules=modules,
        encoded_state=second_encoded,
        categorical_cls=Categorical,
        device="cpu",
        task_state_ready=TASK_STATE_READY_UNSCHEDULED,
        freeze_movement=False,
        decision_transition_tracker=tracker,
    )
    second_buffer.append(second_slot)
    cross_buffer = [
        row
        for row in tracker.completed_transitions
        if row.state.slot_index == first_slot.slot_index
        and row.next_state is not None
        and row.next_state.slot_index == second_slot.slot_index
    ]
    _assert(cross_buffer, "Test 10 real next decision did not close the pre-rollover pending transition")
    row = cross_buffer[-1]
    _assert(row.delta == 1 and np.isclose(row.rho, first_slot.reward), "Test 10 real reward/delta mismatch")
    _assert(not hasattr(first_buffer, "decision_transitions"), "Test 10 slot buffer was coupled to transitions")
    tracker.close_truncated()
    graph_builder.close()
    print(
        "Test 10 PASS: real _collect_clean_slot decision -> commit reward "
        f"({first_slot.reward:.6f}) -> PPO buffer rollover -> next decision; "
        f"rho={row.rho:.6f} delta={row.delta}"
    )

    import scripts.train_clean_mainline as train_clean_mainline

    tracker_instances: list[CleanDecisionTransitionTracker] = []

    class _SpyTracker(CleanDecisionTransitionTracker):
        def __init__(self, *, gamma: float, lane_index: int = 0) -> None:
            super().__init__(gamma=gamma, lane_index=lane_index)
            tracker_instances.append(self)

    original_tracker_class = train_clean_mainline.CleanDecisionTransitionTracker
    original_arrival_probability = config.DAG_BASE_ARRIVAL_PROB
    try:
        train_clean_mainline.CleanDecisionTransitionTracker = _SpyTracker
        config.DAG_BASE_ARRIVAL_PROB = 1.0
        with tempfile.TemporaryDirectory(
            prefix=".codex_tmp_phase4a_mainline_",
            dir=ROOT,
        ) as temp_dir:
            args = train_clean_mainline.build_arg_parser().parse_args(
                [
                    "--smoke",
                    "--episodes",
                    "1",
                    "--max-steps-per-episode",
                    "2",
                    "--rollout-horizon",
                    "1",
                    "--checkpoint-interval",
                    "0",
                    "--device",
                    "cpu",
                    "--task-encoder",
                    "mlp",
                    "--output-dir",
                    temp_dir,
                    "--run-name",
                    "phase4a_mainline_integration",
                    "--record-decision-transitions",
                ]
            )
            result = train_clean_mainline.run_training(args)
        _assert(tracker_instances, "Test 10 run_training did not instantiate the tracker")
        live_tracker = tracker_instances[0]
        _assert(
            live_tracker.registered_decision_count > 0
            and live_tracker.completed_transition_count > 0,
            "Test 10 run_training did not drive real decision transitions",
        )
        _assert(
            result["record_decision_transitions"] is True
            and result["decision_transition_summary"]["completed_transition_count"]
            == live_tracker.completed_transition_count,
            "Test 10 run_training transition summary mismatch",
        )
        print(
            "Test 10 mainline-loop PASS: run_training registered "
            f"{live_tracker.registered_decision_count} decisions and closed "
            f"{live_tracker.completed_transition_count} transitions"
        )
    finally:
        train_clean_mainline.CleanDecisionTransitionTracker = original_tracker_class
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_probability


def main() -> None:
    original_arrival_probability = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        _unit_transition_checks()
        _behavior_equivalence_check()
        _mainline_integration_check()
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_probability
    print("smoke_clean_decision_transitions passed")


if __name__ == "__main__":
    main()
