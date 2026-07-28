from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import statistics
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class GraphDecisionGroup:
    graph_snapshot_id: str
    sample_indices: tuple[int, ...]
    sample_ids: tuple[str, ...]

    @property
    def decision_count(self) -> int:
        return len(self.sample_indices)


@dataclass(frozen=True, slots=True)
class GraphAwareOptimizerBatch:
    groups: tuple[GraphDecisionGroup, ...]

    @property
    def decision_count(self) -> int:
        return sum(group.decision_count for group in self.groups)

    @property
    def unique_graph_count(self) -> int:
        return len(self.groups)

    @property
    def sample_indices(self) -> tuple[int, ...]:
        return tuple(index for group in self.groups for index in group.sample_indices)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample_id for group in self.groups for sample_id in group.sample_ids)


@dataclass(frozen=True, slots=True)
class GraphAwareBatchPlan:
    batches: tuple[GraphAwareOptimizerBatch, ...]
    batch_plan_hash: str
    shuffle_seed: int
    target_batch_decisions: int
    decision_count: int
    unique_graph_count: int
    decisions_per_graph: dict[str, float | int | None]

    @property
    def encoder_forward_count(self) -> int:
        return sum(batch.unique_graph_count for batch in self.batches)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample_id for batch in self.batches for sample_id in batch.sample_ids)


@dataclass(slots=True)
class GroupedBatchForward:
    samples: list[dict[str, Any]]
    raw_logits: list[Any]
    masked_logits: Any | None
    per_decision_losses: Any | None
    batch_loss: Any | None
    encoder_forward_count: int
    scorer_forward_count: int


def build_graph_aware_batch_plan(
    samples: list[dict[str, Any]],
    *,
    target_batch_decisions: int,
    shuffle_seed: int,
    shuffle_graph_groups: bool = True,
) -> GraphAwareBatchPlan:
    """Build deterministic optimizer batches without splitting graph groups.

    A graph group larger than ``target_batch_decisions`` is emitted as one
    deterministic oversized optimizer batch. This preserves the invariant that
    every graph is encoded exactly once per plan.
    """

    target = int(target_batch_decisions)
    if target <= 0:
        raise ValueError("target_batch_decisions must be positive")

    grouped_indices: dict[str, list[int]] = {}
    grouped_ids: dict[str, list[str]] = {}
    seen_sample_ids: set[str] = set()
    for index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", ""))
        graph_id = str(sample.get("graph_snapshot_id", ""))
        if not sample_id:
            raise ValueError("every grouped sample must have a non-empty sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate grouped sample_id: {sample_id}")
        if not graph_id:
            raise ValueError(f"grouped sample {sample_id} has no graph_snapshot_id")
        seen_sample_ids.add(sample_id)
        grouped_indices.setdefault(graph_id, []).append(int(index))
        grouped_ids.setdefault(graph_id, []).append(sample_id)

    graph_ids = list(grouped_indices)
    if shuffle_graph_groups:
        random.Random(int(shuffle_seed)).shuffle(graph_ids)

    groups = [
        GraphDecisionGroup(
            graph_snapshot_id=graph_id,
            sample_indices=tuple(grouped_indices[graph_id]),
            sample_ids=tuple(grouped_ids[graph_id]),
        )
        for graph_id in graph_ids
    ]

    batches: list[GraphAwareOptimizerBatch] = []
    pending: list[GraphDecisionGroup] = []
    pending_count = 0
    for group in groups:
        if pending and pending_count + group.decision_count > target:
            batches.append(GraphAwareOptimizerBatch(groups=tuple(pending)))
            pending = []
            pending_count = 0
        if group.decision_count > target:
            if pending:
                batches.append(GraphAwareOptimizerBatch(groups=tuple(pending)))
                pending = []
                pending_count = 0
            batches.append(GraphAwareOptimizerBatch(groups=(group,)))
            continue
        pending.append(group)
        pending_count += group.decision_count
    if pending:
        batches.append(GraphAwareOptimizerBatch(groups=tuple(pending)))

    planned_ids = [sample_id for batch in batches for sample_id in batch.sample_ids]
    original_ids = [str(sample["sample_id"]) for sample in samples]
    if len(planned_ids) != len(original_ids):
        raise AssertionError("graph-aware plan changed the decision count")
    if len(set(planned_ids)) != len(planned_ids):
        raise AssertionError("graph-aware plan repeated a decision")
    if set(planned_ids) != set(original_ids):
        raise AssertionError("graph-aware plan dropped or replaced a decision")

    group_sizes = [group.decision_count for group in groups]
    payload = {
        "shuffle_seed": int(shuffle_seed),
        "target_batch_decisions": target,
        "batches": [
            [
                {
                    "graph_snapshot_id": group.graph_snapshot_id,
                    "sample_ids": list(group.sample_ids),
                }
                for group in batch.groups
            ]
            for batch in batches
        ],
    }
    plan_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    plan = GraphAwareBatchPlan(
        batches=tuple(batches),
        batch_plan_hash=plan_hash,
        shuffle_seed=int(shuffle_seed),
        target_batch_decisions=target,
        decision_count=len(samples),
        unique_graph_count=len(groups),
        decisions_per_graph=_distribution_summary(group_sizes),
    )
    if plan.encoder_forward_count != plan.unique_graph_count:
        raise AssertionError("graph-aware plan split a graph across optimizer batches")
    return plan


def samples_for_batch(
    samples: list[dict[str, Any]],
    batch: GraphAwareOptimizerBatch,
) -> list[dict[str, Any]]:
    return [samples[index] for index in batch.sample_indices]


def grouped_candidate_forward(
    modules: Any,
    samples: list[dict[str, Any]],
    *,
    train_enabled: bool,
) -> GroupedBatchForward:
    """Encode each graph once, then score every decision/candidate together."""

    if not samples:
        return GroupedBatchForward(
            samples=[],
            raw_logits=[],
            masked_logits=None,
            per_decision_losses=None,
            batch_loss=None,
            encoder_forward_count=0,
            scorer_forward_count=0,
        )

    torch = modules.torch
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        graph_id = str(sample.get("graph_snapshot_id", ""))
        if not graph_id:
            raise ValueError("grouped batch sample has no graph_snapshot_id")
        mask = list(sample.get("candidate_mask", []))
        if not mask or not any(bool(item) for item in mask):
            raise ValueError(f"grouped batch sample {sample.get('sample_id')} has no legal candidate")
        grouped.setdefault(graph_id, []).append(sample)

    task_embeddings: dict[str, Any] = {}
    encoder_forward_count = 0
    with torch.set_grad_enabled(bool(train_enabled)):
        for graph_id, graph_samples in grouped.items():
            reference = graph_samples[0]
            task_features = torch.as_tensor(
                reference["task_features"], dtype=torch.float32, device=modules.device
            )
            incidence = torch.as_tensor(
                reference["incidence_matrix"], dtype=torch.float32, device=modules.device
            )
            type_ids = torch.as_tensor(
                reference["hyperedge_type_ids"], dtype=torch.long, device=modules.device
            )
            if task_features.dim() != 2 or task_features.shape[0] == 0:
                raise ValueError(f"graph {graph_id} has no task rows for grouped decisions")
            embeddings = modules.task_encoder(task_features, incidence, type_ids)
            encoder_forward_count += 1
            for sample in graph_samples:
                local_index = int(sample["task_local_index"])
                if local_index < 0 or local_index >= int(embeddings.shape[0]):
                    raise ValueError(
                        f"sample {sample.get('sample_id')} task_local_index is outside graph rows"
                    )
                task_embeddings[str(sample["sample_id"])] = embeddings[local_index].reshape(1, -1)

        candidate_rows: list[Any] = []
        candidate_counts: list[int] = []
        masks: list[Any] = []
        labels: list[int] = []
        for sample in samples:
            dynamic = torch.as_tensor(
                sample["dynamic_uav_features"], dtype=torch.float32, device=modules.device
            )
            pair = torch.as_tensor(
                sample["pair_features"], dtype=torch.float32, device=modules.device
            )
            mask = torch.as_tensor(
                sample["candidate_mask"], dtype=torch.bool, device=modules.device
            )
            if dynamic.dim() != 2 or pair.dim() != 2 or mask.dim() != 1:
                raise ValueError("grouped candidate tensors have invalid dimensions")
            if dynamic.shape[0] != pair.shape[0] or dynamic.shape[0] != mask.shape[0]:
                raise ValueError("grouped candidate feature and mask row counts differ")
            count = int(dynamic.shape[0])
            embedding = task_embeddings[str(sample["sample_id"])]
            repeated = embedding.expand(count, -1)
            candidate_rows.append(torch.cat([repeated, dynamic, pair], dim=1))
            candidate_counts.append(count)
            masks.append(mask)
            labels.append(int(sample["greedy_label_idx"]))

        flat_candidate_features = torch.cat(candidate_rows, dim=0)
        flat_logits = modules.offloading_actor.scorer(flat_candidate_features)
        raw_logits = list(torch.split(flat_logits, candidate_counts))
        max_candidates = max(candidate_counts)
        padded_logits: list[Any] = []
        padded_masks: list[Any] = []
        for logits, mask, count in zip(raw_logits, masks, candidate_counts):
            pad = int(max_candidates - count)
            padded_logits.append(
                modules.functional.pad(
                    logits,
                    (0, pad),
                    value=float(torch.finfo(logits.dtype).min),
                )
            )
            padded_masks.append(modules.functional.pad(mask, (0, pad), value=False))
        logits_matrix = torch.stack(padded_logits, dim=0)
        mask_matrix = torch.stack(padded_masks, dim=0)
        masked_logits = logits_matrix.masked_fill(
            ~mask_matrix,
            torch.finfo(logits_matrix.dtype).min,
        )
        label_tensor = torch.as_tensor(labels, dtype=torch.long, device=modules.device)
        per_losses = modules.functional.cross_entropy(
            masked_logits,
            label_tensor,
            reduction="none",
        )
        batch_loss = per_losses.mean()

    if not bool(torch.isfinite(masked_logits).all().item()):
        raise FloatingPointError("non-finite grouped imitation logits")
    if not bool(torch.isfinite(per_losses).all().item()):
        raise FloatingPointError("non-finite grouped imitation loss")
    return GroupedBatchForward(
        samples=list(samples),
        raw_logits=raw_logits,
        masked_logits=masked_logits,
        per_decision_losses=per_losses,
        batch_loss=batch_loss,
        encoder_forward_count=encoder_forward_count,
        scorer_forward_count=1,
    )


def _distribution_summary(values: Iterable[int]) -> dict[str, float | int | None]:
    data = [int(value) for value in values]
    if not data:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(data)
    p95_index = max(int(math.ceil(0.95 * len(ordered))) - 1, 0)
    return {
        "count": len(data),
        "mean": float(statistics.fmean(data)),
        "median": float(statistics.median(data)),
        "p95": float(ordered[p95_index]),
        "max": int(ordered[-1]),
    }
