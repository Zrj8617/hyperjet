from __future__ import annotations

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_models.mappo.clean_eft_auxiliary import eft_auxiliary_lambda


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _non_torch_checks() -> None:
    initial = 2.2
    expected = {
        0: initial,
        8: initial,
        9: initial * 11.0 / 12.0,
        19: initial / 12.0,
        20: 0.0,
        21: 0.0,
        29: 0.0,
    }
    for update, value in expected.items():
        _assert(
            math.isclose(
                eft_auxiliary_lambda(update, lambda_initial=initial),
                value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"lambda schedule mismatch at update {update}",
        )

    trainer_source = (
        ROOT / "marl_models" / "mappo" / "clean_trainer.py"
    ).read_text(encoding="utf-8")
    loss_source = trainer_source[
        trainer_source.index("    def _loss(") :
        trainer_source.index("    def _stats_from_loss_parts(")
    ]
    forbidden = (
        "estimate_offloading_candidate",
        "build_offloading_candidate_components",
        "TemporaryReservationState",
    )
    _assert(
        not any(name in loss_source for name in forbidden),
        "PPO update must not call environment EFT/reservation helpers",
    )
    update_source = trainer_source[
        trainer_source.index("    def update_many(") :
        trainer_source.index("    def _precompute_lagged_q_corrections(")
    ]
    _assert(
        update_source.count("self.optimizer.step()") == 1,
        "the PPO epoch body must contain exactly one optimizer.step",
    )
    _assert(
        update_source.count("loss_parts[\"total_loss\"].backward()") == 1,
        "the PPO epoch body must contain exactly one combined backward",
    )


def _torch_checks() -> None:
    import torch
    from torch import nn
    from torch.distributions import Categorical

    from marl_models.mappo.clean_eft_auxiliary import (
        compute_eft_auxiliary_objective,
        summarize_historical_eft_items,
    )

    logits = torch.tensor([0.2, 100.0, -0.4], dtype=torch.float32, requires_grad=True)
    mask = torch.tensor([True, False, True])
    eft = torch.tensor([10.0, 999.0, 30.0])
    base = {
        "masked_logits": logits.masked_fill(~mask, torch.finfo(torch.float32).min),
        "candidate_mask": mask,
        "candidate_estimated_finish_times": eft,
        "decision_order": 0,
    }
    generator_a = torch.Generator().manual_seed(123)
    loss_a, diagnostics_a = compute_eft_auxiliary_objective(
        [{**base, "rollout_action": 0}],
        regret_scale=10.0,
        categorical_cls=Categorical,
        generator=generator_a,
        include_debug_vectors=True,
    )
    generator_b = torch.Generator().manual_seed(123)
    loss_b, diagnostics_b = compute_eft_auxiliary_objective(
        [{**base, "rollout_action": 2}],
        regret_scale=10.0,
        categorical_cls=Categorical,
        generator=generator_b,
        include_debug_vectors=True,
    )
    _assert(
        diagnostics_a["eft_aux_sampled_actions"]
        == diagnostics_b["eft_aux_sampled_actions"],
        "auxiliary action must not depend on the rollout PPO action",
    )
    _assert(
        torch.allclose(loss_a, loss_b),
        "auxiliary REINFORCE loss must not use the rollout PPO action",
    )
    _assert(
        diagnostics_a["eft_aux_invalid_action_count"] == 0,
        "masked auxiliary sampling selected an illegal action",
    )
    probs = torch.softmax(base["masked_logits"], dim=0)
    expected_baseline = float((probs * torch.tensor([0.0, 0.0, -2.0])).sum().item())
    _assert(
        math.isclose(
            diagnostics_a["eft_aux_exact_baselines"][0],
            expected_baseline,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "exact policy baseline does not match the masked manual value",
    )
    loss_a.backward()
    _assert(logits.grad is not None and torch.isfinite(logits.grad).all(), "logit gradient is not finite")
    _assert(float(logits.grad[1].abs().item()) == 0.0, "illegal candidate received gradient")

    single_logits = torch.tensor([0.0, 0.0], requires_grad=True)
    single_mask = torch.tensor([True, False])
    single_loss, single_diag = compute_eft_auxiliary_objective(
        [
            {
                "masked_logits": single_logits.masked_fill(
                    ~single_mask, torch.finfo(torch.float32).min
                ),
                "candidate_mask": single_mask,
                "candidate_estimated_finish_times": torch.tensor([5.0, 0.0]),
                "rollout_action": 0,
            }
        ],
        regret_scale=10.0,
        categorical_cls=Categorical,
        generator=torch.Generator().manual_seed(1),
    )
    _assert(single_diag["eft_aux_effective_decision_count"] == 0, "single legal candidate entered auxiliary denominator")
    _assert(single_diag["eft_aux_excluded_single_legal_count"] == 1, "single-candidate exclusion was not counted")
    _assert(float(single_loss.detach().item()) == 0.0, "trivial decision auxiliary loss must be zero")

    encoder = nn.Linear(2, 2, bias=False)
    scorer = nn.Linear(4, 1, bias=False)
    critic = nn.Linear(2, 1)
    movement = nn.Linear(2, 1)
    task = encoder(torch.tensor([[1.0, 2.0]]))[0]
    dynamic = torch.tensor([[0.0], [1.0], [2.0]])
    pair = torch.tensor([[0.5], [0.0], [1.0]])
    features = torch.cat([task.reshape(1, -1).expand(3, -1), dynamic, pair], dim=1)
    current_logits = scorer(features).reshape(-1)
    objective, objective_diag = compute_eft_auxiliary_objective(
        [
            {
                "masked_logits": current_logits,
                "candidate_mask": torch.tensor([True, True, True]),
                "candidate_estimated_finish_times": torch.tensor([30.0, 10.0, 50.0]),
                "rollout_action": 0,
            }
        ],
        regret_scale=10.0,
        categorical_cls=Categorical,
        generator=torch.Generator().manual_seed(7),
    )
    encoder_grads = torch.autograd.grad(
        objective, list(encoder.parameters()), retain_graph=True, allow_unused=True
    )
    scorer_grads = torch.autograd.grad(
        objective, list(scorer.parameters()), retain_graph=True, allow_unused=True
    )
    critic_grads = torch.autograd.grad(
        objective, list(critic.parameters()), retain_graph=True, allow_unused=True
    )
    movement_grads = torch.autograd.grad(
        objective, list(movement.parameters()), retain_graph=True, allow_unused=True
    )
    _assert(any(grad is not None for grad in encoder_grads), "EFT loss did not reach MLP encoder")
    _assert(any(grad is not None for grad in scorer_grads), "EFT loss did not reach scorer")
    _assert(all(grad is None for grad in critic_grads), "EFT loss directly reached critic")
    _assert(all(grad is None for grad in movement_grads), "EFT loss directly reached movement actor")
    _assert(objective_diag["eft_aux_invalid_action_count"] == 0, "direct gradient fixture sampled illegal action")

    summary = summarize_historical_eft_items(
        [
            {
                "candidate_mask": torch.tensor([True, True, False]),
                "candidate_estimated_finish_times": torch.tensor([12.0, 20.0, 0.0]),
                "rollout_action": 1,
            }
        ]
    )
    _assert(
        math.isclose(summary["eft_rollout_chosen_raw_regret_mean"], 8.0),
        "rollout chosen EFT regret is incorrect",
    )

    repeated_items = [
        {
            "masked_logits": torch.zeros(3),
            "candidate_mask": torch.tensor([True, True, True]),
            "candidate_estimated_finish_times": torch.tensor([0.0, 1.0, 2.0]),
            "rollout_action": 0,
        }
        for _ in range(64)
    ]
    shared_generator = torch.Generator().manual_seed(999)
    _, first = compute_eft_auxiliary_objective(
        repeated_items,
        regret_scale=1.0,
        categorical_cls=Categorical,
        generator=shared_generator,
        include_debug_vectors=True,
    )
    _, second = compute_eft_auxiliary_objective(
        repeated_items,
        regret_scale=1.0,
        categorical_cls=Categorical,
        generator=shared_generator,
        include_debug_vectors=True,
    )
    _assert(
        first["eft_aux_sampled_actions"] != second["eft_aux_sampled_actions"],
        "successive updates did not resample auxiliary actions",
    )


def main() -> int:
    _non_torch_checks()
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print("smoke_clean_mappo_eft_auxiliary: PASS (torch checks skipped)")
        return 0
    _torch_checks()
    print("smoke_clean_mappo_eft_auxiliary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
