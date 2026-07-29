from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import greedy_imitation_dataset as frozen_data
from scripts import run_contextual_bandit_gate as bandit
from scripts import smoke_greedy_imitation_grouped_batch as grouped_smoke
from scripts import train_greedy_imitation_gate as gate


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    samples = grouped_smoke._samples()
    for sample in samples:
        legal = [
            float(sample["estimated_finish_times"][index])
            for index, allowed in enumerate(sample["candidate_mask"])
            if allowed
        ]
        ordered = sorted(legal)
        sample["greedy_margin"] = ordered[1] - ordered[0] if len(ordered) >= 2 else None
    frozen_before = frozen_data.canonical_json(samples)
    scale = bandit.compute_train_regret_scale(samples)
    _assert(scale["method"] == "train_legal_candidate_raw_eft_regret_rms", "scale method drifted")
    _assert(scale["value"] > 0.0 and math.isfinite(scale["value"]), "scale is invalid")
    _assert(
        bandit.compute_train_regret_scale(copy.deepcopy(samples)) == scale,
        "train scale is not deterministic",
    )
    unrelated_val = copy.deepcopy(samples)
    unrelated_val[0]["estimated_finish_times"][0] += 1_000_000.0
    _assert(
        bandit.compute_train_regret_scale(samples) == scale,
        "validation/test data must not affect train-only scale",
    )
    three_seed_scales = [bandit.compute_train_regret_scale(samples)["value"] for _ in (42, 86, 1042)]
    _assert(len(set(three_seed_scales)) == 1, "three seeds must share one fixed scale")

    if not _torch_available():
        _assert(frozen_data.canonical_json(samples) == frozen_before, "scale mutated samples")
        print("smoke_contextual_bandit_gate skipped torch checks: torch is not installed")
        return 0

    import torch

    modules = _build_modules()
    for parameter in modules.offloading_actor.scorer.parameters():
        torch.nn.init.zeros_(parameter)
    torch.manual_seed(11)
    loss, diagnostics = bandit.contextual_bandit_objective(
        modules,
        samples,
        regret_scale=float(scale["value"]),
    )
    actions = diagnostics["actions"]
    gathered_mask = diagnostics["mask_matrix"].gather(1, actions.unsqueeze(1)).squeeze(1)
    _assert(bool(gathered_mask.all().item()), "masked policy sampled an illegal action")
    _assert(diagnostics["invalid_action_count"] == 0, "invalid action counter drifted")
    expected_log_probs = torch.log_softmax(
        diagnostics["forward"].masked_logits,
        dim=1,
    ).gather(1, actions.unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(diagnostics["log_probs"], expected_log_probs)
    expected_rewards = diagnostics["reward_matrix"].gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)
    expected_regrets = diagnostics["raw_regret_matrix"].gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)
    torch.testing.assert_close(diagnostics["selected_rewards"], expected_rewards)
    torch.testing.assert_close(diagnostics["selected_raw_regrets"], expected_regrets)
    uniform_baselines = []
    for row, mask in zip(diagnostics["reward_matrix"], diagnostics["mask_matrix"]):
        uniform_baselines.append(row[mask].mean())
    torch.testing.assert_close(
        diagnostics["exact_baseline"],
        torch.stack(uniform_baselines),
    )
    _assert(diagnostics["exact_baseline"].requires_grad, "exact baseline must use current policy")
    _assert(not diagnostics["advantage"].requires_grad, "bandit advantage baseline must be detached")
    _assert(not diagnostics["reward_matrix"].requires_grad, "EFT reward must never backpropagate")
    _assert(bool(torch.isfinite(loss).item()), "bandit loss is non-finite")
    modules.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in list(modules.task_encoder.parameters())
        + list(modules.offloading_actor.scorer.parameters())
        if parameter.grad is not None
    ]
    _assert(gradients, "bandit objective produced no gradients")
    _assert(
        all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients),
        "bandit gradient is non-finite",
    )

    torch.manual_seed(21)
    _, first = bandit.contextual_bandit_objective(
        modules, samples, regret_scale=float(scale["value"])
    )
    torch.manual_seed(22)
    _, second = bandit.contextual_bandit_objective(
        modules, samples, regret_scale=float(scale["value"])
    )
    _assert(
        first["actions"].detach().cpu().tolist()
        != second["actions"].detach().cpu().tolist(),
        "each update must resample actions from the current policy",
    )

    first_eval = bandit.evaluate_deterministic(
        modules,
        samples,
        regret_scale=float(scale["value"]),
        target_batch_decisions=8,
    )
    second_eval = bandit.evaluate_deterministic(
        modules,
        samples,
        regret_scale=float(scale["value"]),
        target_batch_decisions=8,
    )
    _assert(first_eval == second_eval, "validation/test argmax must be deterministic")
    _assert(first_eval["invalid_action_count"] == 0, "deterministic eval selected illegal action")
    _assert(first_eval["finite"], "deterministic eval is non-finite")
    _assert(
        frozen_data.canonical_json(samples) == frozen_before,
        "bandit objective or evaluation mutated frozen samples",
    )

    print("smoke_contextual_bandit_gate passed")
    return 0


def _build_modules() -> gate.GateModules:
    args = argparse.Namespace(
        seed=42,
        task_encoder="mlp",
        task_feature_dim=3,
        task_embedding_dim=4,
        hidden_dim=8,
        lr=1e-3,
        gradient_batch_decisions=8,
        max_grad_norm=0.5,
        completed_dag_weight=16.0,
        device="cpu",
    )
    return gate._build_modules(args, encoder_seed=42, scorer_seed=42)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
