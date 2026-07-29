from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("SKIP smoke_decision_ppo_bandit_objective: torch unavailable")
        return 0

    from marl_models.hgnn.clean_independent_mlp import CleanIndependentTaskMLP
    from marl_models.mappo.clean_decision_ppo_bandit import (
        DecisionBanditPPOUpdater,
        DecisionBanditRolloutBuffer,
        DecisionBanditUpdateConfig,
        build_decision_bandit_record,
        decision_ppo_surrogate,
    )
    from marl_models.mappo.clean_offloading_actor import SharedOffloadingCandidateScorer

    positive_loss, positive_ratio = decision_ppo_surrogate(
        new_log_prob=torch.tensor(math.log(0.8)),
        old_log_prob=torch.tensor(math.log(0.5)),
        advantage=torch.tensor(2.0),
        clip_epsilon=0.2,
    )
    assert math.isclose(float(positive_ratio), 1.6, rel_tol=1e-6)
    assert math.isclose(float(positive_loss), -2.4, rel_tol=1e-6)
    negative_loss, negative_ratio = decision_ppo_surrogate(
        new_log_prob=torch.tensor(math.log(0.2)),
        old_log_prob=torch.tensor(math.log(0.5)),
        advantage=torch.tensor(-2.0),
        clip_epsilon=0.2,
    )
    assert math.isclose(float(negative_ratio), 0.4, rel_tol=1e-6)
    assert math.isclose(float(negative_loss), 1.6, rel_tol=1e-6)

    action = SimpleNamespace(
        candidate_mask=torch.tensor([True, True, False]),
        candidate_estimated_finish_times=torch.tensor([10.0, 20.0, 999.0]),
        old_masked_probabilities=torch.tensor([0.25, 0.75, 0.0]),
        dynamic_uav_features=torch.zeros((3, 7)),
        pair_features=torch.zeros((3, 8)),
        candidate_uav_ids=[0, 1, 2],
        selected_action=1,
        selected_uav_id=1,
        old_log_prob=math.log(0.75),
        selected_estimated_finish_time=20.0,
        decision_order=0,
        task_id="task",
        dag_id="dag",
        task_local_index=0,
    )
    graph = SimpleNamespace(
        task_features=np.ones((1, 4), dtype=np.float32),
        task_id_to_idx={"task": 0},
        idx_to_task_id=["task"],
    )
    record = build_decision_bandit_record(
        action_record=action,
        graph_snapshot=graph,
        environment_id=0,
        trajectory_id="trajectory",
        episode_id=0,
        physical_slot=0,
        regret_scale=10.0,
    )
    # rewards are [0, -1, 0], exact baseline=-0.75, executed advantage=-0.25
    assert math.isclose(record.old_policy_baseline, -0.75, rel_tol=1e-6)
    assert math.isclose(record.advantage, -0.25, rel_tol=1e-6)
    assert not record.task_features.flags.writeable
    assert not record.old_masked_probabilities.flags.writeable

    encoder = CleanIndependentTaskMLP(4, 4, output_dim=2)
    scorer = SharedOffloadingCandidateScorer(2 + 7 + 8, hidden_dim=4)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(scorer.parameters()), lr=1e-3
    )
    updater = DecisionBanditPPOUpdater(
        encoder=encoder,
        scorer=scorer,
        optimizer=optimizer,
        config=DecisionBanditUpdateConfig(
            ppo_epochs=3, chunk_decisions=1, entropy_coef=0.0
        ),
    )
    buffer = DecisionBanditRolloutBuffer(records=[record])
    before_record = np.asarray(record.task_features).copy()
    stats = updater.update(buffer)
    assert stats.optimizer_step_count == 3
    assert stats.effective_decision_count == 1
    assert all(math.isfinite(float(row["actor_loss"])) for row in stats.epochs)
    assert np.array_equal(before_record, record.task_features)

    empty_encoder = CleanIndependentTaskMLP(4, 4, output_dim=2)
    empty_scorer = SharedOffloadingCandidateScorer(2 + 7 + 8, hidden_dim=4)
    empty_optimizer = torch.optim.Adam(
        list(empty_encoder.parameters()) + list(empty_scorer.parameters()), lr=1e-3
    )
    empty_updater = DecisionBanditPPOUpdater(
        encoder=empty_encoder,
        scorer=empty_scorer,
        optimizer=empty_optimizer,
        config=DecisionBanditUpdateConfig(ppo_epochs=3),
    )
    parameter_before = [value.detach().clone() for value in empty_encoder.parameters()]
    empty_stats = empty_updater.update(DecisionBanditRolloutBuffer())
    assert empty_stats.empty_actor_batch
    assert empty_stats.optimizer_step_count == 0
    assert all(row["actor_loss"] is None for row in empty_stats.epochs)
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameter_before, empty_encoder.parameters())
    )
    print("PASS smoke_decision_ppo_bandit_objective")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
