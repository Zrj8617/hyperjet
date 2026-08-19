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
    np.random.seed(11)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    original_kahypar_enabled = config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = True
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
        ready_task_ids = {task.task_id for task in env.task_manager.get_ready_tasks()}
        pending_task_ids = active_task_ids - ready_task_ids
        _assert(set(snapshot.active_task_ids) == active_task_ids, "snapshot active ids should be active unfinished tasks only.")
        _assert(set(snapshot.ready_task_ids) == ready_task_ids, "snapshot ready ids should match frozen/default ready set.")
        _assert(set(snapshot.pending_task_ids) == pending_task_ids, "snapshot pending ids should be active minus ready.")
        _assert(snapshot.task_ids == snapshot.active_task_ids, "task_ids compatibility alias should mirror active ids.")
        _assert(snapshot.task_features.shape[0] == len(snapshot.active_task_ids), "task feature row count mismatch.")
        _assert(snapshot.task_features.shape[1] == 12, "task features should contain clean task-only fields.")
        _assert(
            snapshot.incidence_matrix.shape == (len(snapshot.task_ids), len(snapshot.hyperedges)),
            "incidence matrix shape should match nodes x hyperedges.",
        )
        _assert(
            set(np.unique(snapshot.incidence_matrix)).issubset({0.0, 1.0}),
            "incidence matrix should be binary.",
        )
        _assert(
            isinstance(snapshot.partition_hyperedges, list),
            "partition hyperedges should be present even when KaHyPar degrades to an empty cache.",
        )
        _assert(not hasattr(snapshot, "uav_features"), "GraphSnapshot must not include UAV features.")
        _assert(not hasattr(snapshot, "new_dag_arrived"), "GraphSnapshot must not include slot event flags.")
        _assert(not hasattr(snapshot, "dag_arrival_version"), "GraphSnapshot must not include arrival versions.")
        _assert(
            not hasattr(snapshot, "ready_task_to_uav_candidate_pairs"),
            "GraphSnapshot must not include candidate pairs.",
        )
        _assert(not hasattr(snapshot, "uav_ids"), "GraphSnapshot must not include UAV ids.")
        _assert(
            all(snapshot.idx_to_task_id[idx] == task_id for task_id, idx in snapshot.task_id_to_idx.items()),
            "task_id/local_idx mappings should be reversible.",
        )

        for parent_idx, child_idx in snapshot.dag_hyperedges:
            parent_id = snapshot.task_ids[int(parent_idx)]
            child_id = snapshot.task_ids[int(child_idx)]
            parent = env.task_manager.get_task(parent_id)
            child = env.task_manager.get_task(child_id)
            _assert(parent is not None and child is not None, "DAG hyperedge references missing task.")
            _assert(
                child_id in parent.successors or parent_id in child.successors,
                "DAG size-2 hyperedge should come from an original dependency.",
            )

        _assert(snapshot.khop_hyperedges, "k-hop dependency hyperedges should not be empty.")
        _assert(
            all(len(edge) >= 2 for edge in snapshot.khop_hyperedges),
            "each k-hop hyperedge should contain at least two tasks.",
        )
        khop_groups_from_cache = {
            tuple(sorted(snapshot.task_id_to_idx[task_id] for task_id in group if task_id in snapshot.task_id_to_idx))
            for group in job.khop_hyperedges_global
        }
        _assert(
            all(tuple(edge) in khop_groups_from_cache for edge in snapshot.khop_hyperedges),
            "k-hop hyperedges should be filtered/remapped from DAG-level global task id cache.",
        )
        if len(snapshot.task_ids) >= 2:
            _assert(
                len(snapshot.attribute_hyperedges) <= config.ATTRIBUTE_HYPEREDGE_CLUSTER_NUM,
                "attribute hyperedge count should not exceed configured cluster count.",
            )

        second_ue = env.ues[1]
        second_job = env.task_manager.create_dag_for_ue(
            ue_id=second_ue.id,
            source_pos=second_ue.pos[:2].copy(),
            current_time_step=1,
        )
        second_ue.enter_service_waiting(second_job.dag_id)
        forced_snapshot = builder.build(
            env.task_manager,
            env.uavs,
            current_time_step=1,
            executor=env.executor,
            new_dag_arrived=True,
            dag_arrival_version=env.task_manager.dag_arrival_version,
        )
        _assert(second_job.dag_id in {env.task_manager.get_task(task_id).dag_id for task_id in forced_snapshot.task_ids}, "forced snapshot should include newly arrived DAG tasks.")
        _assert(builder.last_attribute_update_step == 1, "new DAG event should force attribute hyperedge update.")
        _assert(builder.last_partition_attempt_step == 1, "attribute update should trigger KaHyPar repartition attempt.")

        finished_task = env.task_manager.get_ready_tasks()[0]
        env.task_manager.mark_task_finished(finished_task.task_id, current_time_step=1.0)
        refreshed = builder.build(env.task_manager, env.uavs, current_time_step=1, executor=env.executor)
        _assert(
            finished_task.task_id not in refreshed.task_ids,
            "finished task should not remain in clean graph snapshot.",
        )

        empty_env = Env()
        empty_env.reset()
        empty_snapshot = builder.build(empty_env.task_manager, empty_env.uavs, current_time_step=0, executor=empty_env.executor)
        _assert(empty_snapshot.active_task_ids == [], "empty snapshot should have no active tasks.")
        _assert(empty_snapshot.ready_task_ids == [], "empty snapshot should have no ready tasks.")
        _assert(empty_snapshot.pending_task_ids == [], "empty snapshot should have no pending tasks.")
        _assert(empty_snapshot.task_features.shape == (0, 12), "empty task feature matrix shape mismatch.")
        _assert(empty_snapshot.task_id_to_idx == {}, "empty snapshot task_id_to_idx should be empty.")
        _assert(empty_snapshot.idx_to_task_id == {}, "empty snapshot idx_to_task_id should be empty.")
        _assert(empty_snapshot.hyperedges == [], "empty snapshot should have no hyperedges.")
        _assert(empty_snapshot.incidence_matrix.shape == (0, 0), "empty snapshot incidence matrix shape mismatch.")

    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = original_kahypar_enabled

    print("smoke_clean_graph passed")


if __name__ == "__main__":
    main()
