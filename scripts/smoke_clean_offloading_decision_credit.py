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
from marl_models.mappo.clean_offloading_decision_credit import (
    CleanOffloadingDecisionCredit,
    compute_smdp_decision_gae,
    decision_state_key,
    encode_decision_state,
)


def _state(name: str, slot: int, order: int, selected: int = 0) -> CleanDecisionState:
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
            [[1.0, 2.0], [99.0, 99.0], [3.0, 6.0]], dtype=np.float32
        ),
        critic_global_context=np.asarray([0.25, 0.75], dtype=np.float32),
        old_masked_probabilities=np.asarray([0.4, 0.0, 0.6], dtype=np.float32),
        old_log_probability=float(np.log(0.4 if selected == 0 else 0.6)),
    )


def _transition(
    state: CleanDecisionState,
    *,
    rho: float,
    delta: int,
    next_state: CleanDecisionState | None,
    terminated: bool = False,
    truncated: bool = False,
    unresolved: bool = False,
) -> CleanDecisionTransition:
    return CleanDecisionTransition(
        state=state,
        rho=rho,
        delta=delta,
        next_state=next_state,
        terminated=terminated,
        truncated=truncated,
        unresolved=unresolved,
        future_bootstrap=0.0 if terminated else None,
    )


def _state_encoder_checks() -> None:
    state = _state("s", 0, 0, selected=0)
    original = encode_decision_state(state)
    permutation = np.asarray([2, 1, 0])
    permuted = replace(
        state,
        candidate_uav_ids=tuple(np.asarray(state.candidate_uav_ids)[permutation]),
        candidate_mask=np.asarray(state.candidate_mask)[permutation],
        candidate_features=np.asarray(state.candidate_features)[permutation],
        old_masked_probabilities=np.asarray(state.old_masked_probabilities)[permutation],
        selected_action=2,
    )
    assert np.array_equal(original, encode_decision_state(permuted))
    changed_action = replace(state, selected_action=2, selected_uav_id=2)
    assert np.array_equal(original, encode_decision_state(changed_action))
    assert np.allclose(original[:2], [2.0, 4.0])
    print("state encoder PASS: legal-only pooling, permutation/action invariant")


def _toy_gae_check() -> None:
    gamma = 0.9
    lam = 0.8
    s1, s2, s3 = _state("s1", 0, 0), _state("s2", 0, 1), _state("s3", 1, 0)
    rho3 = 3.0 + gamma * 4.0 + gamma**2 * 5.0
    rows = [
        _transition(s1, rho=0.0, delta=0, next_state=s2),
        _transition(s2, rho=2.0, delta=1, next_state=s3),
        _transition(s3, rho=rho3, delta=3, next_state=None, terminated=True),
    ]
    values = {decision_state_key(s1): 1.0, decision_state_key(s2): 2.0, decision_state_key(s3): 3.0}
    td, advantages, targets = compute_smdp_decision_gae(
        rows, values=values, gamma=gamma, gae_lambda=lam
    )
    td3 = rho3 - 3.0
    adv3 = td3
    td2 = 2.0 + gamma * 3.0 - 2.0
    adv2 = td2 + (gamma * lam) * adv3
    td1 = 2.0 - 1.0
    adv1 = td1 + adv2
    expected = [adv1, adv2, adv3]
    actual = [advantages[decision_state_key(state)] for state in (s1, s2, s3)]
    assert np.allclose(actual, expected)
    assert np.allclose(
        [targets[decision_state_key(state)] for state in (s1, s2, s3)],
        np.asarray(expected) + np.asarray([1.0, 2.0, 3.0]),
    )
    assert np.isclose(td[decision_state_key(s1)], td1)
    print("toy SMDP GAE PASS: Delta=0/1/3 and (gamma*lambda)^Delta")


def _boundary_check() -> None:
    class Record:
        pass

    source = _state("pending", 0, 0)
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
    tracker.record_decisions(slot_index=1, records=[record])
    assert tracker.pending
    print("rollout boundary PASS: unresolved/censored, pending cleared, episode remains live")


def _rng_neutrality_check() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    credit = CleanOffloadingDecisionCredit.build_rng_neutral(
        input_dim=12,
        hidden_dim=8,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        device="cpu",
    )
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    if cuda_before is not None:
        assert all(
            torch.equal(left, right)
            for left, right in zip(cuda_before, torch.cuda.get_rng_state_all())
        )
    assert credit.update_count == 0
    restored = CleanOffloadingDecisionCredit.build_rng_neutral(
        input_dim=12,
        hidden_dim=8,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        device="cpu",
    )
    restored.load_state_dict(credit.state_dict())
    for key, value in credit.critic.state_dict().items():
        assert torch.equal(value, restored.critic.state_dict()[key])
    print("RNG neutrality PASS: Python/NumPy/Torch/CUDA state unchanged")


def main() -> None:
    _state_encoder_checks()
    _toy_gae_check()
    _boundary_check()
    _rng_neutrality_check()
    print("smoke_clean_offloading_decision_credit passed")


if __name__ == "__main__":
    main()
