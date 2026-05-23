from __future__ import annotations

import json

import numpy as np

import config
from environment.graph_builder import HeteroGraphSnapshot
from scripts.fine_tune_outcome_rerank import (
    OutcomePair,
    SnapshotOutcomeSample,
    build_outcome_pair_index,
    fine_tune,
)


def test_build_outcome_pair_index_bad_only(tmp_path):
    attribution_path = tmp_path / "static_full_seed42_attribution.json"
    payload = {
        "assignments": [
            {
                "episode": 1,
                "step": 7,
                "task_id": "task_1",
                "dag_id": "dag_1",
                "selection_mode": "score",
                "disagrees_with_heuristic": True,
                "score_uav": 2,
                "heuristic_uav": 3,
                "delta_planned_finish": 0.4,
                "selected_deadline_margin": -0.2,
                "heuristic_deadline_margin": 0.2,
                "is_high_risk_task": True,
                "is_critical_path_task": True,
            },
            {
                "episode": 1,
                "step": 8,
                "task_id": "task_2",
                "dag_id": "dag_1",
                "selection_mode": "score",
                "disagrees_with_heuristic": True,
                "score_uav": 4,
                "heuristic_uav": 5,
                "delta_planned_finish": 0.01,
                "selected_deadline_margin": 1.0,
                "heuristic_deadline_margin": 1.01,
            },
        ],
        "task_outcomes": [
            {
                "episode": 1,
                "task_id": "task_1",
                "dropped": True,
                "finished": False,
                "finished_on_time": False,
            },
            {
                "episode": 1,
                "task_id": "task_2",
                "dropped": False,
                "finished": True,
                "finished_on_time": True,
            },
        ],
        "dag_outcomes": [
            {
                "episode": 1,
                "dag_id": "dag_1",
                "successful": False,
                "failed": True,
                "on_time_successful": False,
            }
        ],
    }
    attribution_path.write_text(json.dumps(payload), encoding="utf-8")

    index = build_outcome_pair_index(
        tmp_path,
        use_good=False,
        good_delta_tolerance=0.1,
        strong_delta_threshold=0.3,
        max_label_files=0,
        max_pairs=0,
    )

    assert list(index) == [(42, 1, 7, "task_1")]
    pair = index[(42, 1, 7, "task_1")]
    assert pair.preferred_uav == 3
    assert pair.other_uav == 2
    assert pair.label == "BAD_SCORE_DECISION_STRONG"
    assert pair.weight == 2.0


def test_fine_tune_one_synthetic_snapshot(tmp_path):
    snapshot = HeteroGraphSnapshot(
        task_ids=["task_1"],
        uav_ids=[0, 1],
        task_features=np.zeros((1, config.DAG_TASK_FEATURE_DIM), dtype=np.float32),
        uav_features=np.zeros((2, 7), dtype=np.float32),
        dependency_edges=[],
        task_uav_edges=[("task_1", 0), ("task_1", 1)],
        task_uav_edge_features=np.zeros((2, config.BASE_TASK_UAV_PAIR_FEATURE_DIM), dtype=np.float32),
        uav_uav_edges=[(0, 1), (1, 0)],
        service_domain_hyperedges=[],
        resource_competition_hyperedges=[],
        collaborative_hyperedges=[],
        critical_hyperedges=[],
        critical_support_hyperedges=[],
        compute_attribute_hyperedges=[],
        communication_attribute_hyperedges=[],
        candidate_scarce_attribute_hyperedges=[],
        attribute_hyperedges=[],
    )
    sample = SnapshotOutcomeSample(
        snapshot=snapshot,
        pairs=[
            (
                "task_1",
                OutcomePair(
                    preferred_uav=0,
                    other_uav=1,
                    weight=1.0,
                    label="BAD_SCORE_DECISION",
                    delta_planned_finish=0.2,
                ),
            )
        ],
    )

    model_path, metrics = fine_tune(
        [sample],
        checkpoint="",
        output_dir=tmp_path,
        device="cpu",
        epochs=1,
        lr=1e-5,
        outcome_margin=0.05,
        lambda_outcome=0.2,
        lambda_distill=1.0,
    )

    assert metrics
    assert metrics[-1].pair_count == 1
    assert model_path.endswith("phase_one_graph_scheduler.pt")
