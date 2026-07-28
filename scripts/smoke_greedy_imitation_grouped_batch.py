from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import greedy_imitation_dataset as frozen_data
from scripts import greedy_imitation_grouped_batch as grouped
from scripts import run_greedy_imitation_encoder_comparison as comparison
from scripts import train_greedy_imitation_gate as gate


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    samples = _samples()
    original_payload = frozen_data.canonical_json(samples)
    first = grouped.build_graph_aware_batch_plan(
        samples,
        target_batch_decisions=3,
        shuffle_seed=12345,
    )
    second = grouped.build_graph_aware_batch_plan(
        samples,
        target_batch_decisions=3,
        shuffle_seed=12345,
    )
    _assert(first.batch_plan_hash == second.batch_plan_hash, "batch plan is not deterministic")
    _assert(first.sample_ids == second.sample_ids, "deterministic plan changed sample order")
    _assert(first.decision_count == len(samples), "batch plan changed decision count")
    _assert(len(set(first.sample_ids)) == len(samples), "batch plan repeated a decision")
    _assert(set(first.sample_ids) == {sample["sample_id"] for sample in samples}, "batch plan dropped a decision")
    _assert(first.unique_graph_count == 2, "unique graph count mismatch")
    _assert(first.encoder_forward_count == 2, "plan split a graph across batches")

    oversized = grouped.build_graph_aware_batch_plan(
        samples,
        target_batch_decisions=2,
        shuffle_seed=12345,
        shuffle_graph_groups=False,
    )
    graph_a_batch = next(
        batch
        for batch in oversized.batches
        if any(group.graph_snapshot_id == "graph-a" for group in batch.groups)
    )
    _assert(
        graph_a_batch.decision_count == 3 and graph_a_batch.unique_graph_count == 1,
        "oversized graph group must remain one deterministic optimizer batch",
    )
    empty = grouped.build_graph_aware_batch_plan(
        [],
        target_batch_decisions=4,
        shuffle_seed=1,
    )
    _assert(empty.decision_count == 0 and empty.encoder_forward_count == 0, "empty plan failed")

    plan_hashes = {
        encoder: grouped.build_graph_aware_batch_plan(
            samples,
            target_batch_decisions=3,
            shuffle_seed=comparison._epoch_shuffle_seed(42, 0),
        ).batch_plan_hash
        for encoder in comparison.CANONICAL_ENCODERS
    }
    _assert(len(set(plan_hashes.values())) == 1, "four encoders received different batch plans")

    if not _torch_available():
        _assert(
            frozen_data.canonical_json(samples) == original_payload,
            "non-torch batch planning mutated frozen samples",
        )
        print("smoke_greedy_imitation_grouped_batch skipped torch equivalence: torch is not installed")
        return 0

    for encoder in comparison.CANONICAL_ENCODERS:
        _assert_grouped_equivalence(encoder, samples)

    modules = _build_modules("mlp")
    forward, results = gate._score_grouped_batch(modules, [], train_enabled=False)
    _assert(forward.batch_loss is None and results == [], "empty grouped forward failed")
    invalid = copy.deepcopy(samples[0])
    invalid["candidate_mask"] = [False] * len(invalid["candidate_mask"])
    try:
        gate._score_grouped_batch(modules, [invalid], train_enabled=False)
    except ValueError as exc:
        _assert("no legal candidate" in str(exc), "zero-candidate error was not explicit")
    else:
        raise AssertionError("zero-candidate sample must be rejected before supervised loss")

    empty_graph = copy.deepcopy(samples[0])
    empty_graph["task_features"] = []
    empty_graph["incidence_matrix"] = []
    try:
        gate._score_grouped_batch(modules, [empty_graph], train_enabled=False)
    except ValueError as exc:
        _assert("no task rows" in str(exc), "empty-graph error was not explicit")
    else:
        raise AssertionError("empty graph decision must be rejected explicitly")

    _assert(
        frozen_data.canonical_json(samples) == original_payload,
        "grouped forward or optimizer step mutated frozen samples",
    )
    print("smoke_greedy_imitation_grouped_batch passed")
    return 0


def _assert_grouped_equivalence(
    encoder: str,
    samples: list[dict[str, Any]],
) -> None:
    torch = _torch()
    legacy = _build_modules(encoder)
    batched = _build_modules(encoder)
    _assert_state_close(torch, legacy, batched, "initial")

    legacy_forward_count = {"value": 0}
    grouped_forward_count = {"value": 0}
    legacy_handle = legacy.task_encoder.register_forward_hook(
        lambda _module, _inputs, _output: legacy_forward_count.__setitem__(
            "value", legacy_forward_count["value"] + 1
        )
    )
    grouped_handle = batched.task_encoder.register_forward_hook(
        lambda _module, _inputs, _output: grouped_forward_count.__setitem__(
            "value", grouped_forward_count["value"] + 1
        )
    )
    try:
        legacy.optimizer.zero_grad(set_to_none=True)
        legacy_logits: list[Any] = []
        legacy_masked: list[Any] = []
        legacy_losses: list[Any] = []
        for sample in samples:
            logits = gate._sample_logits(legacy, sample)
            mask = torch.as_tensor(
                sample["candidate_mask"],
                dtype=torch.bool,
                device=legacy.device,
            )
            masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            label = torch.as_tensor(
                [int(sample["greedy_label_idx"])],
                dtype=torch.long,
                device=legacy.device,
            )
            loss = legacy.functional.cross_entropy(masked.unsqueeze(0), label)
            legacy_logits.append(logits)
            legacy_masked.append(masked)
            legacy_losses.append(loss)
        legacy_batch_loss = torch.stack(legacy_losses).mean()

        grouped_forward, grouped_results = gate._score_grouped_batch(
            batched,
            samples,
            train_enabled=True,
        )
        _assert(len(grouped_results) == len(samples), f"{encoder} grouped result count mismatch")
        _assert(
            grouped_forward.encoder_forward_count == 2,
            f"{encoder} grouped forward count metadata mismatch",
        )
        _assert(
            grouped_forward.scorer_forward_count == 1,
            f"{encoder} scorer was not truly batched",
        )
        for index, sample in enumerate(samples):
            candidate_count = len(sample["candidate_mask"])
            torch.testing.assert_close(
                grouped_forward.raw_logits[index],
                legacy_logits[index],
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            torch.testing.assert_close(
                grouped_forward.masked_logits[index, :candidate_count],
                legacy_masked[index],
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            torch.testing.assert_close(
                grouped_forward.per_decision_losses[index],
                legacy_losses[index],
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        torch.testing.assert_close(
            grouped_forward.batch_loss,
            legacy_batch_loss,
            rtol=1.0e-5,
            atol=1.0e-6,
        )

        legacy_batch_loss.backward()
        legacy_gradient_norm = torch.nn.utils.clip_grad_norm_(
            _parameters(legacy),
            float(legacy.max_grad_norm),
        )
        legacy.optimizer.step()
        grouped_gradient_norm = gate._apply_grouped_optimizer_step(
            batched,
            grouped_forward,
        )
        _assert(
            grouped_gradient_norm is not None
            and math.isfinite(grouped_gradient_norm)
            and math.isfinite(float(legacy_gradient_norm.item())),
            f"{encoder} gradient norm was non-finite",
        )
        _assert_gradients_close(torch, legacy, batched, encoder)
        _assert_state_close(torch, legacy, batched, f"{encoder} optimizer step")
    finally:
        legacy_handle.remove()
        grouped_handle.remove()

    _assert(
        legacy_forward_count["value"] == len(samples),
        f"{encoder} legacy reference did not forward once per decision",
    )
    _assert(
        grouped_forward_count["value"] == 2,
        f"{encoder} grouped path did not forward once per unique graph",
    )


def _build_modules(encoder: str) -> gate.GateModules:
    args = argparse.Namespace(
        seed=42,
        task_encoder=str(encoder),
        task_feature_dim=3,
        task_embedding_dim=4,
        hidden_dim=8,
        lr=1.0e-3,
        gradient_batch_decisions=8,
        max_grad_norm=1000.0,
        completed_dag_weight=16.0,
        device="cpu",
    )
    return gate._build_modules(args, encoder_seed=77, scorer_seed=99)


def _parameters(modules: gate.GateModules) -> list[Any]:
    return list(modules.task_encoder.parameters()) + list(
        modules.offloading_actor.scorer.parameters()
    )


def _named_parameters(modules: gate.GateModules) -> dict[str, Any]:
    values = {
        f"encoder.{name}": parameter
        for name, parameter in modules.task_encoder.named_parameters()
    }
    values.update(
        {
            f"scorer.{name}": parameter
            for name, parameter in modules.offloading_actor.scorer.named_parameters()
        }
    )
    return values


def _assert_gradients_close(
    torch: Any,
    legacy: gate.GateModules,
    batched: gate.GateModules,
    encoder: str,
) -> None:
    legacy_values = _named_parameters(legacy)
    batched_values = _named_parameters(batched)
    _assert(legacy_values.keys() == batched_values.keys(), f"{encoder} parameter names drifted")
    for name in legacy_values:
        legacy_grad = legacy_values[name].grad
        batched_grad = batched_values[name].grad
        _assert(
            (legacy_grad is None) == (batched_grad is None),
            f"{encoder} gradient presence drifted for {name}",
        )
        if legacy_grad is not None:
            torch.testing.assert_close(
                legacy_grad,
                batched_grad,
                rtol=2.0e-5,
                atol=2.0e-6,
            )


def _assert_state_close(
    torch: Any,
    legacy: gate.GateModules,
    batched: gate.GateModules,
    context: str,
) -> None:
    legacy_values = _named_parameters(legacy)
    batched_values = _named_parameters(batched)
    _assert(legacy_values.keys() == batched_values.keys(), f"{context} parameter names drifted")
    for name in legacy_values:
        post_step = "optimizer step" in context
        torch.testing.assert_close(
            legacy_values[name].detach(),
            batched_values[name].detach(),
            rtol=5.0e-3 if post_step else 2.0e-5,
            atol=6.0e-4 if post_step else 2.0e-6,
        )


def _samples() -> list[dict[str, Any]]:
    graph_a = {
        "task_features": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        "incidence_matrix": [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        "hyperedge_type_ids": [0, 2],
    }
    graph_b = {
        "task_features": [
            [0.15, 0.25, 0.35],
            [0.45, 0.55, 0.65],
        ],
        "incidence_matrix": [
            [1.0],
            [1.0],
        ],
        "hyperedge_type_ids": [3],
    }
    return [
        _sample("a0", "graph-a", graph_a, task_local_index=0, candidate_count=5, legal=(0, 2, 4), label=2, offset=0.0),
        _sample("a1", "graph-a", graph_a, task_local_index=1, candidate_count=4, legal=(1, 3), label=1, offset=0.3),
        _sample("a2", "graph-a", graph_a, task_local_index=2, candidate_count=3, legal=(0, 1, 2), label=0, offset=0.6),
        _sample("b0", "graph-b", graph_b, task_local_index=1, candidate_count=5, legal=(1, 2, 4), label=4, offset=0.9),
    ]


def _sample(
    sample_id: str,
    graph_id: str,
    graph: dict[str, Any],
    *,
    task_local_index: int,
    candidate_count: int,
    legal: tuple[int, ...],
    label: int,
    offset: float,
) -> dict[str, Any]:
    mask = [index in legal for index in range(candidate_count)]
    eft = [float(10.0 + offset + index * 1.5) for index in range(candidate_count)]
    eft[label] = float(1.0 + offset)
    return {
        "sample_id": str(sample_id),
        "graph_snapshot_id": str(graph_id),
        "task_features": copy.deepcopy(graph["task_features"]),
        "incidence_matrix": copy.deepcopy(graph["incidence_matrix"]),
        "hyperedge_type_ids": list(graph["hyperedge_type_ids"]),
        "task_local_index": int(task_local_index),
        "dynamic_uav_features": [
            [float(offset + row * 0.1 + col * 0.01) for col in range(7)]
            for row in range(candidate_count)
        ],
        "pair_features": [
            [float(offset + row * 0.2 + col * 0.02) for col in range(8)]
            for row in range(candidate_count)
        ],
        "candidate_mask": mask,
        "candidate_uav_ids": list(range(candidate_count)),
        "candidate_uav_id_mapping": list(range(candidate_count)),
        "greedy_label_idx": int(label),
        "estimated_finish_times": eft,
        "valid_candidate_count": int(sum(mask)),
        "trajectory_policy": "greedy_eft",
        "episode": 0,
        "slot": int(round(offset * 10)),
    }


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _torch() -> Any:
    import torch

    return torch


if __name__ == "__main__":
    raise SystemExit(main())
