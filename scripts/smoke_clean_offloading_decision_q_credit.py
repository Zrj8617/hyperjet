from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_models.mappo.clean_decision_transitions import (
    CleanDecisionState,
    CleanDecisionTransition,
    CleanDecisionTransitionTracker,
)
from marl_models.mappo.clean_offloading_decision_q_credit import (
    CleanOffloadingDecisionQCredit,
    FrozenOffloadingDecisionQBatch,
    encode_decision_candidate_rows,
    expected_behavior_q,
    selected_legal_row,
)
from marl_models.mappo.clean_ppo import normalized_clipped_value_loss


def _state(name: str, slot: int, order: int, q0: float, q2: float, selected: int = 0):
    return CleanDecisionState(
        episode_index=0,
        lane_index=0,
        slot_index=slot,
        task_id=name,
        task_local_index=order,
        dag_id="dag",
        decision_order=order,
        selected_action=selected,
        selected_uav_id=selected,
        candidate_uav_ids=(0, 1, 2),
        candidate_mask=np.asarray([True, False, True]),
        candidate_features=np.asarray(
            [[q0, 2.0], [99.0, 99.0], [q2, 6.0]], dtype=np.float32
        ),
        critic_global_context=np.asarray([0.25, 0.75], dtype=np.float32),
        old_masked_probabilities=np.asarray([0.4, 0.0, 0.6], dtype=np.float32),
        old_log_probability=float(np.log(0.4 if selected == 0 else 0.6)),
    )


def _transition(state, *, rho, delta, next_state=None, terminated=False, unresolved=False):
    return CleanDecisionTransition(
        state=state,
        rho=float(rho),
        delta=int(delta),
        next_state=next_state,
        terminated=bool(terminated),
        truncated=bool(unresolved),
        unresolved=bool(unresolved),
        future_bootstrap=0.0 if terminated else None,
    )


class _FirstFeatureQ(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, 1, bias=False))
        with torch.no_grad():
            self.net[0].weight.zero_()
            self.net[0].weight[0, 0] = 1.0

    def forward(self, value):
        return self.net(value).squeeze(-1)


def _credit() -> CleanOffloadingDecisionQCredit:
    critic = _FirstFeatureQ(6)
    return CleanOffloadingDecisionQCredit(
        critic=critic,
        optimizer=torch.optim.Adam(critic.parameters(), lr=3e-4),
        gamma=0.9,
        max_grad_norm=0.5,
        ppo_epochs=3,
        value_clip_epsilon=0.2,
        device="cpu",
    )


def _candidate_and_expected_value_checks() -> None:
    state = _state("s", 0, 0, 1.0, 3.0, selected=2)
    indices, rows = encode_decision_candidate_rows(state)
    assert np.array_equal(indices, [0, 2])
    assert np.allclose(rows[:, 0], [1.0, 3.0])
    assert selected_legal_row(state, indices) == 1
    expected = expected_behavior_q(
        legal_indices=indices,
        probabilities=state.old_masked_probabilities,
        q_values=np.asarray([1.0, 3.0]),
    )
    assert np.isclose(expected, 2.2)
    permutation = np.asarray([2, 1, 0])
    permuted = replace(
        state,
        candidate_uav_ids=tuple(np.asarray(state.candidate_uav_ids)[permutation]),
        candidate_mask=np.asarray(state.candidate_mask)[permutation],
        candidate_features=np.asarray(state.candidate_features)[permutation],
        old_masked_probabilities=np.asarray(state.old_masked_probabilities)[permutation],
        selected_action=0,
    )
    perm_indices, perm_rows = encode_decision_candidate_rows(permuted)
    perm_expected = expected_behavior_q(
        legal_indices=perm_indices,
        probabilities=permuted.old_masked_probabilities,
        q_values=perm_rows[:, 0],
    )
    assert np.isclose(perm_expected, expected)
    assert np.isclose(perm_rows[selected_legal_row(permuted, perm_indices), 0], 3.0)
    print("candidate alignment PASS: permutation, selected row, expected V=sum(pi*Q)")


def _target_and_advantage_checks() -> None:
    credit = _credit()
    s0 = _state("s0", 0, 0, 1.0, 3.0, selected=0)
    s1 = _state("s1", 0, 1, 2.0, 4.0, selected=2)
    s2 = _state("s2", 3, 0, 5.0, 7.0, selected=0)
    rows = [
        _transition(s0, rho=0.0, delta=0, next_state=s1),
        _transition(s1, rho=5.0, delta=1, next_state=s2),
        _transition(s2, rho=1.0 + 0.9 * 2.0 + 0.9**2 * 3.0, delta=3, terminated=True),
        _transition(_state("censored", 4, 0, 1.0, 2.0), rho=4.0, delta=1, unresolved=True),
    ]
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    batch = credit.prepare_rollout(rows)
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0] and np.array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert batch.critic_targets.shape == (3,)
    assert np.allclose(
        batch.critic_targets,
        [3.2, 5.0 + 0.9 * 6.2, 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0],
    )
    raw_advantages = np.asarray([-1.2, 0.8, -1.2], dtype=np.float32)
    target_scale = max(float(batch.critic_targets.std(ddof=0)), 1e-8)
    scaled = raw_advantages / target_scale
    assert np.allclose(list(batch.actor_advantages.values()), scaled)
    assert np.isclose(batch.diagnostics["decision_q_actor_advantage_scale"], target_scale)
    assert np.isclose(
        batch.diagnostics["decision_q_scaled_actor_advantage_std"],
        scaled.std(ddof=0),
    )
    assert batch.diagnostics["decision_q_unresolved_count"] == 1
    assert np.isclose(batch.diagnostics["decision_q_advantage_std"], raw_advantages.std(ddof=0))
    assert np.isfinite(batch.critic_targets).all()
    assert np.isfinite(list(batch.actor_advantages.values())).all()
    assert all(
        np.isfinite(value)
        for value in batch.diagnostics.values()
        if isinstance(value, (int, float))
    )
    frozen_actor_advantages = dict(batch.actor_advantages)
    credit.train_frozen(batch)
    assert batch.actor_advantages == frozen_actor_advantages
    print(
        "target/advantage PASS: Delta=0/1/N, terminal, censor, "
        "Q-V credit scaled by frozen target std, frozen across critic epochs, RNG neutral"
    )


def _zero_target_scale_check() -> None:
    credit = _credit()
    rows = [
        _transition(_state("z0", 0, 0, 1.0, 3.0), rho=1.0, delta=1, terminated=True),
        _transition(_state("z1", 1, 0, 2.0, 4.0), rho=1.0, delta=1, terminated=True),
    ]
    batch = credit.prepare_rollout(rows)
    assert np.isclose(batch.diagnostics["decision_q_actor_advantage_scale"], 1e-8)
    assert np.isfinite(list(batch.actor_advantages.values())).all()
    assert np.isfinite(batch.diagnostics["decision_q_scaled_actor_advantage_std"])
    print("zero target-scale clamp PASS: scaled actor advantages remain finite")


def _tracker_boundary_check() -> None:
    class Record:
        pass
    source = _state("pending", 0, 0, 1.0, 2.0)
    record = Record()
    for field in source.__dataclass_fields__:
        setattr(record, field, getattr(source, field))
    tracker = CleanDecisionTransitionTracker(gamma=0.9)
    tracker.start_episode(0)
    tracker.record_decisions(slot_index=0, records=[record])
    tracker.record_slot_reward(7.0)
    tracker.close_rollout_boundary()
    row = tracker.pop_completed()[0]
    assert row.delta == 1 and np.isclose(row.rho, 7.0)
    assert row.truncated and row.unresolved and not tracker.pending
    print("rollout boundary PASS: unresolved sample excluded and pending cleared")


def _rng_and_optimizer_state_checks() -> None:
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    credit = CleanOffloadingDecisionQCredit.build_rng_neutral(
        input_dim=6,
        hidden_dim=8,
        learning_rate=3e-4,
        gamma=0.99,
        max_grad_norm=0.5,
        ppo_epochs=3,
        value_clip_epsilon=0.2,
        device="cpu",
    )
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0] and np.array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    inputs = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(4, 6)
    targets = np.asarray([1e6, 1.001e6, 0.999e6, 1.002e6], dtype=np.float32)
    with torch.no_grad():
        old = credit.critic(torch.as_tensor(inputs)).cpu().numpy().astype(np.float32)
    diagnostics = credit.train_frozen(FrozenOffloadingDecisionQBatch(
        actor_advantages={},
        critic_inputs=inputs,
        critic_targets=targets,
        critic_old_predictions=old,
        transitions=(),
        diagnostics={"decision_q_target_mean": float(targets.mean()), "decision_q_target_std": float(targets.std(ddof=0))},
    ))
    assert all(np.isfinite(value) for value in diagnostics.values() if isinstance(value, (int, float)))
    restored = CleanOffloadingDecisionQCredit.build_rng_neutral(
        input_dim=6,
        hidden_dim=8,
        learning_rate=3e-4,
        gamma=0.99,
        max_grad_norm=0.5,
        ppo_epochs=3,
        value_clip_epsilon=0.2,
        device="cpu",
    )
    restored.load_state_dict(credit.state_dict())
    for key, value in credit.critic.state_dict().items():
        assert torch.equal(value, restored.critic.state_dict()[key])
    left_optimizer = credit.optimizer.state_dict()
    right_optimizer = restored.optimizer.state_dict()
    assert left_optimizer["param_groups"] == right_optimizer["param_groups"]
    for parameter_id, left_values in left_optimizer["state"].items():
        right_values = right_optimizer["state"][parameter_id]
        for key, value in left_values.items():
            if torch.is_tensor(value):
                assert torch.equal(value, right_values[key])
            else:
                assert value == right_values[key]
    print(
        "RNG-neutral build and normalized Q training/checkpoint PASS: "
        f"loss={diagnostics['decision_q_normalized_loss']:.6f} "
        f"grad={diagnostics['decision_q_preclip_grad_norm_max']:.6f} "
        f"clip={diagnostics['decision_q_value_clip_fraction']:.6f}"
    )


def _normalized_clipping_check() -> None:
    loss, clipped = normalized_clipped_value_loss(
        value=torch.tensor([1050.0]),
        old_value=torch.tensor([1000.0]),
        target=torch.tensor([1100.0]),
        target_mean=torch.tensor(1000.0),
        target_scale=torch.tensor(100.0),
        clip_epsilon=0.2,
    )
    assert torch.allclose(loss, torch.tensor([0.32]), atol=1e-6)
    assert clipped.item() == 1.0
    print("normalized Q clipping PASS: PPO value-clipping semantics")


def main() -> None:
    _candidate_and_expected_value_checks()
    _target_and_advantage_checks()
    _zero_target_scale_check()
    _tracker_boundary_check()
    _rng_and_optimizer_state_checks()
    _normalized_clipping_check()
    print("smoke_clean_offloading_decision_q_credit passed")


if __name__ == "__main__":
    main()
