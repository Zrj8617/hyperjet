from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    try:
        import torch
        from marl_models.hgnn import CleanIncidenceHGNN
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print("smoke_clean_hgnn skipped: torch is not installed in this Python runtime")
            return
        raise

    np.random.seed(17)
    torch.manual_seed(17)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        env = Env()
        env.reset()
        ue = env.ues[0]
        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=ue.pos[:2].copy(),
            current_time_step=env.current_time_seconds,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()

        builder = CleanGraphBuilder()
        snapshot = builder.build(env.task_manager, env.uavs, env.time_step, executor=env.executor)
        model = CleanIncidenceHGNN(task_feature_dim=snapshot.task_features.shape[1], hidden_dim=16, output_dim=8)
        embeddings = model(snapshot.task_features, snapshot.incidence_matrix)
        _assert(embeddings.shape == (len(snapshot.task_ids), 8), "non-empty HGNN embedding shape mismatch.")
        _assert(torch.isfinite(embeddings).all().item(), "HGNN embeddings should be finite.")

        empty_env = Env()
        empty_env.reset()
        empty_snapshot = builder.build(
            empty_env.task_manager,
            empty_env.uavs,
            current_time_step=0,
            executor=empty_env.executor,
        )
        empty_embeddings = model(empty_snapshot.task_features, empty_snapshot.incidence_matrix)
        _assert(empty_embeddings.shape == (0, 8), "empty HGNN embedding shape mismatch.")
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob

    print("smoke_clean_hgnn passed")


if __name__ == "__main__":
    main()
