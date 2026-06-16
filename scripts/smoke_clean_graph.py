from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.dag_tasks import TASK_STATE_FINISHED
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    np.random.seed(11)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        env = Env()
        env.reset()
        ue = env.ues[0]
        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=ue.pos[:2].copy(),
            current_time_step=env.time_step,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()

        builder = CleanGraphBuilder()
        snapshot = builder.build(env.task_manager, env.uavs, env.time_step, executor=env.executor)

        active_task_ids = {task.task_id for task in env.task_manager.get_active_tasks()}
        _assert(set(snapshot.task_ids) == active_task_ids, "snapshot task_ids should be active unfinished tasks only.")
        _assert(len(snapshot.uav_ids) == config.NUM_UAVS, "snapshot should include all UAVs.")
        _assert(snapshot.task_features.shape[0] == len(snapshot.task_ids), "task feature row count mismatch.")
        _assert(snapshot.uav_features.shape[0] == config.NUM_UAVS, "uav feature row count mismatch.")
        _assert(snapshot.dag_dependency_edges.shape[0] == 2, "dependency edge array should have shape (2, E).")

        for parent_idx, child_idx in snapshot.dag_dependency_edges.T:
            parent_id = snapshot.task_ids[int(parent_idx)]
            child_id = snapshot.task_ids[int(child_idx)]
            parent = env.task_manager.get_task(parent_id)
            child = env.task_manager.get_task(child_id)
            _assert(parent is not None and child is not None, "dependency edge references missing task.")
            _assert(child_id in parent.successors, "dependency edge parent->child not in successors.")
            _assert(parent_id in child.predecessors, "dependency edge parent->child not in predecessors.")

        _assert(snapshot.khop_dependency_hyperedges, "k-hop dependency hyperedges should not be empty.")
        _assert(
            all(len(edge) >= 2 for edge in snapshot.khop_dependency_hyperedges),
            "each k-hop hyperedge should contain at least two tasks.",
        )
        if len(snapshot.task_ids) >= 2:
            _assert(
                len(snapshot.attribute_hyperedges) <= config.ATTRIBUTE_HYPEREDGE_CLUSTER_NUM,
                "attribute hyperedge count should not exceed configured cluster count.",
            )
        ready_count = len(env.task_manager.get_ready_tasks())
        _assert(
            snapshot.ready_task_to_uav_candidate_pairs.shape == (2, ready_count * config.NUM_UAVS),
            "candidate pair count should equal ready_task_count * NUM_UAVS.",
        )
        _assert(snapshot.task_features.shape[1] == 10, "task features should contain only clean feature fields.")

        finished_task = env.task_manager.get_ready_tasks()[0]
        env.task_manager.mark_task_finished(finished_task.task_id, current_time_step=1.0)
        refreshed = builder.build(env.task_manager, env.uavs, current_time_step=1, executor=env.executor)
        _assert(
            finished_task.task_id not in refreshed.task_ids,
            "finished task should not remain in clean graph snapshot.",
        )

    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob

    print("smoke_clean_graph passed")


if __name__ == "__main__":
    main()
