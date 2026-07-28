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
from marl_models.mappo.clean_slot_orchestrator import (
    assert_graph_snapshot_task_only,
    copy_clean_graph_snapshot,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expected_type_ids(snapshot) -> np.ndarray:
    return np.asarray(
        [0] * len(snapshot.dag_hyperedges)
        + [1] * len(snapshot.khop_hyperedges)
        + [2] * len(snapshot.attribute_hyperedges)
        + [3] * len(snapshot.partition_hyperedges),
        dtype=np.int64,
    )


def _assert_type_alignment(snapshot) -> None:
    type_ids = np.asarray(snapshot.hyperedge_type_ids, dtype=np.int64)
    expected = _expected_type_ids(snapshot)
    _assert(
        type_ids.shape == (snapshot.incidence_matrix.shape[1],),
        "hyperedge_type_ids must be 1D and aligned to incidence matrix columns.",
    )
    _assert(
        len(type_ids) == len(snapshot.hyperedges),
        "hyperedge_type_ids length must match the final hyperedge list.",
    )
    _assert(
        np.array_equal(type_ids, expected),
        "hyperedge_type_ids must follow dag/khop/attribute/partition final column order.",
    )
    _assert(
        set(type_ids.tolist()).issubset({0, 1, 2, 3}),
        "hyperedge type ids must be restricted to the declared type ids.",
    )


def _make_env_with_one_dag() -> tuple[Env, object]:
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
    return env, job


def main() -> None:
    np.random.seed(13)
    original_values = {
        "DAG_BASE_ARRIVAL_PROB": config.DAG_BASE_ARRIVAL_PROB,
        "ENABLE_DAG_DEPENDENCY_EDGES": config.ENABLE_DAG_DEPENDENCY_EDGES,
        "ENABLE_KHOP_DEPENDENCY_HYPEREDGES": config.ENABLE_KHOP_DEPENDENCY_HYPEREDGES,
        "ENABLE_ATTRIBUTE_HYPEREDGES": config.ENABLE_ATTRIBUTE_HYPEREDGES,
        "ENABLE_KAHYPAR_PARTITION_HYPEREDGES": config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES,
        "ATTRIBUTE_HYPEREDGE_CLUSTER_NUM": config.ATTRIBUTE_HYPEREDGE_CLUSTER_NUM,
    }
    try:
        config.DAG_BASE_ARRIVAL_PROB = 0.0
        config.ENABLE_DAG_DEPENDENCY_EDGES = True
        config.ENABLE_KHOP_DEPENDENCY_HYPEREDGES = True
        config.ENABLE_ATTRIBUTE_HYPEREDGES = True
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = False
        config.ATTRIBUTE_HYPEREDGE_CLUSTER_NUM = 1

        env, _job = _make_env_with_one_dag()
        builder = CleanGraphBuilder()
        snapshot = builder.build(env.task_manager, env.uavs, env.time_step, executor=env.executor)
        assert_graph_snapshot_task_only(snapshot)
        _assert_type_alignment(snapshot)
        _assert(snapshot.dag_hyperedges, "test fixture should expose DAG dependency hyperedges.")
        _assert(snapshot.khop_hyperedges, "test fixture should expose k-hop hyperedges.")
        _assert(snapshot.attribute_hyperedges, "test fixture should expose attribute hyperedges.")
        _assert(0 in set(snapshot.hyperedge_type_ids.tolist()), "DAG hyperedges should be tagged as type 0.")
        _assert(1 in set(snapshot.hyperedge_type_ids.tolist()), "k-hop hyperedges should be tagged as type 1.")
        _assert(2 in set(snapshot.hyperedge_type_ids.tolist()), "attribute hyperedges should be tagged as type 2.")

        copied = copy_clean_graph_snapshot(snapshot)
        _assert_type_alignment(copied)
        _assert(copied.hyperedge_type_ids is not snapshot.hyperedge_type_ids, "copy must own type id array.")
        _assert(not copied.hyperedge_type_ids.flags.writeable, "copied hyperedge_type_ids should be read-only.")
        if len(snapshot.hyperedge_type_ids) > 0:
            copied_before = copied.hyperedge_type_ids.copy()
            snapshot.hyperedge_type_ids[0] = 99
            _assert(
                np.array_equal(copied.hyperedge_type_ids, copied_before),
                "historical copy must not reflect later source type-id mutations.",
            )

        config.ENABLE_DAG_DEPENDENCY_EDGES = True
        config.ENABLE_KHOP_DEPENDENCY_HYPEREDGES = False
        config.ENABLE_ATTRIBUTE_HYPEREDGES = False
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = True
        partition_builder = CleanGraphBuilder()
        active_ids = [task.task_id for task in env.task_manager.get_active_tasks()]
        _assert(len(active_ids) >= 2, "partition fixture requires at least two active tasks.")
        partition_builder._cached_partition_groups_global = [active_ids[:2]]
        partition_snapshot = partition_builder.build(
            env.task_manager,
            env.uavs,
            env.time_step,
            executor=env.executor,
        )
        _assert_type_alignment(partition_snapshot)
        _assert(partition_snapshot.partition_hyperedges, "cached partition fixture should expose partition hyperedges.")
        _assert(3 in set(partition_snapshot.hyperedge_type_ids.tolist()), "partition hyperedges should be tagged as type 3.")

        empty_env = Env()
        empty_env.reset()
        empty_snapshot = CleanGraphBuilder().build(
            empty_env.task_manager,
            empty_env.uavs,
            current_time_step=0,
            executor=empty_env.executor,
        )
        assert_graph_snapshot_task_only(empty_snapshot)
        _assert(empty_snapshot.active_task_ids == [], "empty snapshot should have no active tasks.")
        _assert(empty_snapshot.incidence_matrix.shape == (0, 0), "empty incidence matrix should be 0x0.")
        _assert(
            np.asarray(empty_snapshot.hyperedge_type_ids, dtype=np.int64).shape == (0,),
            "empty hyperedge_type_ids should be a legal empty 1D array.",
        )
        _assert_type_alignment(empty_snapshot)

    finally:
        for name, value in original_values.items():
            setattr(config, name, value)

    print("smoke_clean_graph_hyperedge_types passed")


if __name__ == "__main__":
    main()
