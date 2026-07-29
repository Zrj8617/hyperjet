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
        import marl_models.mappo.clean_offloading_actor as actor_module
    except ModuleNotFoundError:
        print("SKIP smoke_decision_ppo_bandit_collection: torch unavailable")
        return 0

    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor

    original = actor_module.build_offloading_candidate_components
    try:
        actor = CleanOffloadingActor(task_embedding_dim=2, hidden_dim=4)
        task = SimpleNamespace(task_id="task", dag_id="dag")
        graph = SimpleNamespace(task_id_to_idx={"task": 0})

        def no_legal(**_kwargs):
            return (
                np.zeros((2, 7), dtype=np.float32),
                np.zeros((2, 8), dtype=np.float32),
                np.asarray([False, False], dtype=bool),
                [0, 1],
                [],
            )

        actor_module.build_offloading_candidate_components = no_legal
        assignments = actor.act(
            frozen_ready_tasks=[task],
            task_embeddings=torch.zeros((1, 2)),
            graph_snapshot=graph,
            task_manager=object(),
            uavs=[],
            executor=object(),
            current_time_seconds=0.0,
        )
        assert assignments.entry_count == 0
        assert actor.latest_records == []
        assert len(actor.latest_skip_events) == 1
        event = actor.latest_skip_events[0]
        assert event.task_id == "task"
        assert event.decision_order == 0
        assert event.valid_candidate_count == 0
        assert event.skip_reason == "no_legal_candidate"

        estimates = [
            SimpleNamespace(
                estimated_finish_time=10.0,
                estimated_queued_workload=1.0,
            ),
            SimpleNamespace(
                estimated_finish_time=20.0,
                estimated_queued_workload=2.0,
            ),
        ]

        def two_legal(**_kwargs):
            return (
                np.zeros((2, 7), dtype=np.float32),
                np.zeros((2, 8), dtype=np.float32),
                np.asarray([True, True], dtype=bool),
                [0, 1],
                estimates,
            )

        actor_module.build_offloading_candidate_components = two_legal
        # Avoid depending on executor internals in this focused collection smoke.
        original_from_executor = actor_module.TemporaryReservationState.from_executor
        reservation = SimpleNamespace(reserve=lambda *_args, **_kwargs: None)
        actor_module.TemporaryReservationState.from_executor = lambda *_args, **_kwargs: reservation
        try:
            assignments = actor.act(
                frozen_ready_tasks=[task],
                task_embeddings=torch.zeros((1, 2)),
                graph_snapshot=graph,
                task_manager=object(),
                uavs=[],
                executor=object(),
                current_time_seconds=0.0,
            )
        finally:
            actor_module.TemporaryReservationState.from_executor = original_from_executor
        assert assignments.entry_count == 1
        assert len(actor.latest_records) == 1
        record = actor.latest_records[0]
        probabilities = record.old_masked_probabilities.numpy()
        assert np.isclose(probabilities.sum(), 1.0)
        assert math.isclose(
            record.old_log_prob,
            math.log(float(probabilities[record.selected_action])),
            rel_tol=1e-5,
            abs_tol=1e-5,
        )
        assert actor.latest_skip_events == []
    finally:
        actor_module.build_offloading_candidate_components = original
    print("PASS smoke_decision_ppo_bandit_collection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
