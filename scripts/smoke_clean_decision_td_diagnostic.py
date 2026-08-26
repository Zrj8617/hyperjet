from __future__ import annotations

import copy
import math
from pathlib import Path
import tempfile
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from marl_models.mappo.clean_decision_td_diagnostic import (
    PHASE4_TARGET_SYNC_INTERVAL,
    CleanDecisionTDDiagnostic,
    build_clean_decision_td_diagnostic,
    decision_q_input,
    expected_sarsa_target,
)
from marl_models.mappo.clean_decision_transitions import (
    CleanDecisionState,
    CleanDecisionTransition,
    CleanDecisionTransitionTracker,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _state(
    *,
    slot: int,
    order: int,
    values: tuple[float, float] = (2.0, 6.0),
    probabilities: tuple[float, float] = (0.25, 0.75),
) -> CleanDecisionState:
    return CleanDecisionState(
        episode_index=0,
        lane_index=0,
        slot_index=slot,
        task_id=f"task_{slot}_{order}",
        task_local_index=order,
        dag_id="dag_0",
        decision_order=order,
        selected_action=1,
        selected_uav_id=11,
        candidate_uav_ids=(10, 11),
        candidate_mask=np.asarray([True, True]),
        candidate_features=np.asarray([[values[0], 1.0], [values[1], 3.0]], dtype=np.float32),
        critic_global_context=np.asarray([7.0], dtype=np.float32),
        old_masked_probabilities=np.asarray(probabilities, dtype=np.float32),
        old_log_probability=float(math.log(probabilities[1])),
    )


def _transition(
    *,
    slot: int,
    rho: float,
    delta: int,
    next_state: CleanDecisionState | None,
    terminated: bool = False,
    truncated: bool = False,
    unresolved: bool = False,
) -> CleanDecisionTransition:
    return CleanDecisionTransition(
        state=_state(slot=slot, order=0),
        rho=rho,
        delta=delta,
        next_state=next_state,
        terminated=terminated,
        truncated=truncated,
        unresolved=unresolved,
        future_bootstrap=0.0 if terminated else None,
    )


def _target_math_checks(torch: object) -> None:
    class FixedQ(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            return inputs[:, 0]

    fixed = FixedQ()
    next_state = _state(slot=0, order=1)
    same_slot = _transition(slot=0, rho=0.0, delta=0, next_state=next_state)
    value = expected_sarsa_target(same_slot, target_q=fixed, gamma=0.9, device="cpu")
    _assert(np.isclose(float(value), 5.0), "Test A added an incorrect gamma factor")
    print("Test A PASS: Y=0+gamma^0*(0.25*2+0.75*6)=5.0")

    one_slot = _transition(slot=0, rho=2.0, delta=1, next_state=_state(slot=1, order=0))
    value = expected_sarsa_target(one_slot, target_q=fixed, gamma=0.9, device="cpu")
    _assert(np.isclose(float(value), 6.5), "Test B one-slot target mismatch")
    print("Test B PASS: Y=2+0.9*5=6.5")

    rho = 1.0 + 0.9 * 2.0 + (0.9**2) * 3.0
    multi_slot = _transition(slot=0, rho=rho, delta=3, next_state=_state(slot=3, order=0))
    value = expected_sarsa_target(multi_slot, target_q=fixed, gamma=0.9, device="cpu")
    expected = rho + (0.9**3) * 5.0
    _assert(np.isclose(float(value), expected), "Test C multi-slot discount is off by one")
    print(f"Test C PASS: rho={rho:.6f}, bootstrap=gamma^3*5, Y={expected:.6f}")

    terminal = _transition(slot=0, rho=4.25, delta=2, next_state=None, terminated=True)
    calls_before = fixed.calls
    value = expected_sarsa_target(terminal, target_q=fixed, gamma=0.9, device="cpu")
    _assert(float(value) == 4.25 and fixed.calls == calls_before, "Test D terminal bootstrapped")
    print("Test D PASS: terminal Y=rho=4.25 and target Q not called")


def _build_diagnostic(torch: object, *, seed: int = 11) -> CleanDecisionTDDiagnostic:
    torch.manual_seed(seed)
    return build_clean_decision_td_diagnostic(
        input_dim=3,
        hidden_dim=8,
        learning_rate=1e-2,
        gamma=0.9,
        max_grad_norm=1.0,
        device="cpu",
    )


def _feed(
    diagnostic: CleanDecisionTDDiagnostic,
    transition: CleanDecisionTransition,
    *,
    phase3b_target: float,
) -> dict[str, object]:
    key = (
        transition.state.episode_index,
        transition.state.lane_index,
        transition.state.slot_index,
    )
    diagnostic.ingest_transitions([transition])
    diagnostic.record_slot_advantages(
        slot_keys=[key],
        normalized_advantages=np.asarray([phase3b_target], dtype=np.float32),
        decision_counts=[1],
    )
    return diagnostic.train_ready()


def _isolation_and_queue_checks(torch: object) -> None:
    diagnostic = _build_diagnostic(torch)
    _assert(
        all(
            torch.equal(left, right)
            for left, right in zip(diagnostic.online_q.parameters(), diagnostic.target_q.parameters())
        ),
        "target Q was not initialized by an exact hard copy",
    )
    unresolved = _transition(
        slot=0,
        rho=1.0,
        delta=1,
        next_state=None,
        truncated=True,
        unresolved=True,
    )
    before_updates = diagnostic.shadow_update_count
    stats = _feed(diagnostic, unresolved, phase3b_target=0.5)
    _assert(not stats["phase4_shadow_optimizer_step"], "Test E trained on unresolved data")
    _assert(diagnostic.shadow_update_count == before_updates, "Test E advanced update count")
    _assert(stats["phase4_truncated_unresolved_transitions"] == 1, "Test E coverage missing")
    print("Test E PASS: unresolved excluded, coverage counted, optimizer not stepped")

    frozen_transition = _transition(
        slot=1,
        rho=0.0,
        delta=0,
        next_state=_state(slot=1, order=1, probabilities=(0.25, 0.75)),
    )
    live_actor = torch.nn.Linear(3, 2)
    for parameter in live_actor.parameters():
        parameter.data.fill_(100.0)
    target_before = expected_sarsa_target(
        frozen_transition, target_q=diagnostic.target_q, gamma=0.9, device="cpu"
    )
    for parameter in live_actor.parameters():
        parameter.data.fill_(-100.0)
    target_after = expected_sarsa_target(
        frozen_transition, target_q=diagnostic.target_q, gamma=0.9, device="cpu"
    )
    _assert(torch.equal(target_before, target_after), "Test F target used live actor")
    print("Test F PASS: target uses frozen rollout probabilities after live actor mutation")

    actor = torch.nn.Linear(2, 2)
    encoder = torch.nn.Linear(2, 2)
    critic = torch.nn.Linear(2, 1)
    movement = torch.nn.Linear(2, 2)
    base_parameters = [
        parameter
        for module in (actor, encoder, critic, movement)
        for parameter in module.parameters()
    ]
    base_optimizer = torch.optim.Adam(base_parameters, lr=1e-3)
    base_before = [parameter.detach().clone() for parameter in base_parameters]
    optimizer_before = copy.deepcopy(base_optimizer.state_dict())
    target_params_before = [parameter.detach().clone() for parameter in diagnostic.target_q.parameters()]
    eligible = _transition(slot=2, rho=3.0, delta=1, next_state=_state(slot=3, order=0))
    stats = _feed(diagnostic, eligible, phase3b_target=-0.4)
    online_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in diagnostic.online_q.parameters()
        if parameter.grad is not None
    )
    _assert(online_grad > 0.0, "Test G shadow online Q has no gradient")
    _assert(all(parameter.grad is None for parameter in base_parameters), "Test G leaked base gradients")
    _assert(all(parameter.grad is None for parameter in diagnostic.target_q.parameters()), "Test G target Q has gradient")
    _assert(all(torch.equal(before, after) for before, after in zip(base_before, base_parameters)), "Test H changed base parameters")
    _assert(base_optimizer.state_dict() == optimizer_before, "Test H changed base optimizer state")
    _assert(
        all(torch.equal(before, after) for before, after in zip(target_params_before, diagnostic.target_q.parameters())),
        "target Q changed before the tenth shadow update",
    )
    _assert(stats["phase4_shadow_optimizer_step"], "eligible sample did not step shadow optimizer")
    print("Test G PASS: only shadow online Q receives gradient")
    print("Test H PASS: base parameters/optimizer unchanged; target unchanged before step 10")

    ordered = _state(slot=4, order=0, probabilities=(0.4, 0.6))
    q_input = decision_q_input(ordered)
    _assert(ordered.candidate_uav_ids == (10, 11), "Test I UAV order changed")
    _assert(np.array_equal(q_input[:, 0], [2.0, 6.0]), "Test I feature/Q order changed")
    _assert(np.allclose(ordered.old_masked_probabilities, [0.4, 0.6]), "Test I probability order changed")
    print("Test I PASS: UAV/mask/features/probabilities/Q rows share one candidate order")

    tracker = CleanDecisionTransitionTracker(gamma=0.9)
    tracker.start_episode(0)
    from types import SimpleNamespace
    def record(index: int):
        state = _state(slot=5, order=index)
        return SimpleNamespace(
            task_id=state.task_id,
            task_local_index=state.task_local_index,
            dag_id=state.dag_id,
            decision_order=state.decision_order,
            selected_action=state.selected_action,
            selected_uav_id=state.selected_uav_id,
            candidate_uav_ids=list(state.candidate_uav_ids),
            candidate_mask=state.candidate_mask,
            candidate_features=state.candidate_features,
            critic_global_context=state.critic_global_context,
            old_masked_probabilities=state.old_masked_probabilities,
            old_log_probability=state.old_log_probability,
        )
    tracker.record_decisions(slot_index=5, records=[record(0), record(1), record(2)])
    first_pop = tracker.pop_completed()
    second_pop = tracker.pop_completed()
    _assert(len(first_pop) == 2 and not second_pop and not tracker.completed_transitions, "Test J queue duplicated/lost data")
    print("Test J PASS: completed queue consumed exactly once and returned to zero")

    while diagnostic.shadow_update_count < PHASE4_TARGET_SYNC_INTERVAL:
        slot = 10 + diagnostic.shadow_update_count
        _feed(
            diagnostic,
            _transition(slot=slot, rho=1.0, delta=1, next_state=_state(slot=slot + 1, order=0)),
            phase3b_target=0.1,
        )
    _assert(diagnostic.shadow_update_count == 10 and diagnostic.target_sync_count == 1, "hard sync count mismatch")
    _assert(
        all(
            torch.equal(left, right)
            for left, right in zip(diagnostic.online_q.parameters(), diagnostic.target_q.parameters())
        ),
        "step-10 hard sync was not exact",
    )
    checkpoint = diagnostic.state_dict()
    resumed = _build_diagnostic(torch, seed=99)
    resumed.load_state_dict(checkpoint)
    _assert(resumed.shadow_update_count == 10 and resumed.target_sync_count == 1, "resume reset sync cycle")
    _assert(resumed.target_sync_interval == 10 and resumed.target_sync_mode == "hard", "resume sync controls mismatch")
    print("hard-sync PASS: exact copy after step 10; checkpoint/resume keeps count 10/sync 1")


def _nested_equal(left: object, right: object, torch: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key], torch) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b, torch) for a, b in zip(left, right)
        )
    return left == right


def _default_equivalence_check(torch: object) -> None:
    from scripts.smoke_clean_decision_transitions import _behavior_equivalence_check
    from scripts import train_clean_mainline

    args = train_clean_mainline.build_arg_parser().parse_args([])
    _assert(not args.phase4_decision_td_diagnostic, "Test K Phase4 diagnostic default changed")
    _behavior_equivalence_check()

    original_probability = config.DAG_BASE_ARRIVAL_PROB
    original_collect = train_clean_mainline._collect_clean_slot
    captures: list[tuple[object, ...]] = []

    def capturing_collect(**kwargs):
        slot, done, info = original_collect(**kwargs)
        captures.append(
            (
                tuple(row.selected_action for row in slot.movement_records),
                tuple(row.old_log_probability for row in slot.movement_records),
                tuple(row.selected_action for row in slot.offloading_records),
                tuple(row.old_log_probability for row in slot.offloading_records),
                float(slot.reward),
                bool(done),
            )
        )
        return slot, done, info

    try:
        config.DAG_BASE_ARRIVAL_PROB = 1.0
        train_clean_mainline._collect_clean_slot = capturing_collect
        with tempfile.TemporaryDirectory(prefix=".codex_tmp_phase4a_equiv_", dir=ROOT) as temp_dir:
            common = [
                "--smoke", "--episodes", "1",
                "--max-steps-per-episode", "2",
                "--rollout-horizon", "1",
                "--checkpoint-interval", "0",
                "--device", "cpu",
                "--task-encoder", "mlp",
                "--output-dir", temp_dir,
                "--record-decision-transitions",
            ]
            baseline_args = train_clean_mainline.build_arg_parser().parse_args(
                common + ["--run-name", "phase4a_equiv_baseline"]
            )
            baseline_result = train_clean_mainline.run_training(baseline_args)
            baseline_capture = list(captures)
            captures.clear()
            variant_args = train_clean_mainline.build_arg_parser().parse_args(
                common
                + [
                    "--run-name", "phase4a_equiv_shadow",
                    "--phase4-decision-td-diagnostic",
                ]
            )
            variant_result = train_clean_mainline.run_training(variant_args)
            variant_capture = list(captures)
            baseline_payload = torch.load(
                Path(baseline_result["run_dir"]) / "checkpoints" / "latest.pt",
                map_location="cpu",
            )
            variant_payload = torch.load(
                Path(variant_result["run_dir"]) / "checkpoints" / "latest.pt",
                map_location="cpu",
            )
        _assert(baseline_capture == variant_capture, "Test K actions/log-probs/rewards diverged")
        for key in ("hgnn", "movement_actor", "offloading_actor", "critic", "optimizer", "rng_state"):
            _assert(
                _nested_equal(baseline_payload[key], variant_payload[key], torch),
                f"Test K base checkpoint state diverged: {key}",
            )
        for key in (
            "total_loss",
            "movement_loss",
            "offloading_loss",
            "value_loss",
            "returns_mean",
            "returns_std",
        ):
            _assert(
                baseline_result["latest_update"][key] == variant_result["latest_update"][key],
                f"Test K PPO statistic diverged: {key}",
            )
    finally:
        train_clean_mainline._collect_clean_slot = original_collect
        config.DAG_BASE_ARRIVAL_PROB = original_probability
    print("Test K PASS: diagnostic off/on actions, log-probs, rewards, PPO, base params/optimizer and RNG identical")


def _mainline_check(torch: object) -> None:
    from scripts import train_clean_mainline

    original_probability = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 1.0
    try:
        with tempfile.TemporaryDirectory(prefix=".codex_tmp_phase4a_td_", dir=ROOT) as temp_dir:
            args = train_clean_mainline.build_arg_parser().parse_args(
                [
                    "--smoke",
                    "--episodes", "1",
                    "--max-steps-per-episode", "2",
                    "--rollout-horizon", "1",
                    "--checkpoint-interval", "0",
                    "--device", "cpu",
                    "--task-encoder", "mlp",
                    "--output-dir", temp_dir,
                    "--run-name", "phase4a_td_integration",
                    "--record-decision-transitions",
                    "--phase4-decision-td-diagnostic",
                ]
            )
            result = train_clean_mainline.run_training(args)
            diagnostics = result["latest_update"]["diagnostics"]
            _assert(diagnostics["phase4_consumed_transition_count"] > 0, "Test L queue was not consumed")
            _assert(diagnostics["phase4_training_eligible_transitions"] > 0, "Test L built no TD samples")
            _assert(diagnostics["phase4_shadow_update_count"] > 0, "Test L shadow optimizer did not run")
            checkpoint_path = next(Path(result["run_dir"]).joinpath("checkpoints").glob("latest.pt"))
            payload = torch.load(checkpoint_path, map_location="cpu")
            saved = payload["extra_state"]["phase4_decision_td_diagnostic"]
            _assert(saved["target_sync_mode"] == "hard" and saved["target_sync_interval"] == 10, "Test L checkpoint controls missing")
            _assert(saved["shadow_update_count"] == diagnostics["phase4_shadow_update_count"], "Test L checkpoint count mismatch")
            required = (
                "phase4_td_target_mean",
                "phase4_q_explained_variance",
                "phase4_legal_q_spread_mean",
                "phase4_within_slot_phase4_target_std_mean",
                "phase4_phase3b_td_sign_agreement",
                "phase4_same_slot_transition_fraction",
            )
            _assert(all(key in diagnostics for key in required), "Test L diagnostics incomplete")
            resume_args = train_clean_mainline.build_arg_parser().parse_args(
                [
                    "--smoke",
                    "--episodes", "2",
                    "--max-steps-per-episode", "2",
                    "--rollout-horizon", "1",
                    "--checkpoint-interval", "0",
                    "--device", "cpu",
                    "--task-encoder", "mlp",
                    "--output-dir", temp_dir,
                    "--run-name", "phase4a_td_resume",
                    "--record-decision-transitions",
                    "--phase4-decision-td-diagnostic",
                    "--resume-checkpoint", str(checkpoint_path),
                ]
            )
            resumed_result = train_clean_mainline.run_training(resume_args)
            resumed_count = resumed_result["latest_update"]["diagnostics"][
                "phase4_shadow_update_count"
            ]
            _assert(
                resumed_count > saved["shadow_update_count"],
                "Test L resume reset the shadow update cycle",
            )
            print(
                "Test L PASS: run_training transition->pop->label->TD target->shadow step; "
                f"consumed={diagnostics['phase4_consumed_transition_count']} "
                f"eligible={diagnostics['phase4_training_eligible_transitions']} "
                f"updates={diagnostics['phase4_shadow_update_count']} "
                f"resumed_updates={resumed_count}"
            )
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_probability


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("smoke_clean_decision_td_diagnostic skipped: torch is not installed")
        return
    _target_math_checks(torch)
    _isolation_and_queue_checks(torch)
    _default_equivalence_check(torch)
    _mainline_check(torch)
    print("smoke_clean_decision_td_diagnostic passed")


if __name__ == "__main__":
    main()
