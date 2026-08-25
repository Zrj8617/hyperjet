from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.assignment import (
    TemporaryReservationState,
    _normalize_pair_features,
    build_offloading_candidate_components,
)
from environment.env import Env


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_no_episode_time_scale_in_feature_code() -> None:
    checked_files = [
        ROOT / "environment" / "assignment.py",
        ROOT / "marl_models" / "mappo" / "clean_movement_actor.py",
        ROOT / "marl_models" / "mappo" / "clean_ppo.py",
    ]
    forbidden = [
        "EPISODE_LENGTH * TIME_SLOT_DURATION",
        "TIME_SLOT_DURATION * EPISODE_LENGTH",
        "config.EPISODE_LENGTH) * float(config.TIME_SLOT_DURATION",
        "config.TIME_SLOT_DURATION) * float(config.EPISODE_LENGTH",
    ]
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            _assert(pattern not in text, f"{path.name} still uses episode-scale time normalization: {pattern}")


def main() -> None:
    _assert(float(config.CLEAN_NORM_PAIR_TIME_REF) < float(config.EPISODE_LENGTH) * float(config.TIME_SLOT_DURATION),
            "pair time norm should be smaller than full-episode time scale.")
    _assert(float(config.CLEAN_NORM_AVAIL_TIME_REF) < float(config.EPISODE_LENGTH) * float(config.TIME_SLOT_DURATION),
            "availability norm should be smaller than full-episode time scale.")
    _assert_no_episode_time_scale_in_feature_code()

    raw_pair_features = [4.0, 10.0, 40.0, 4.0, 10.0, 320.0, 4.0, 10.0]
    normalized_pair_features = _normalize_pair_features(raw_pair_features)
    _assert(
        np.isclose(normalized_pair_features[2], 1.0),
        "queue waiting time should retain x/40 clipping.",
    )
    _assert(
        np.isclose(normalized_pair_features[5], 320.0 / (320.0 + 160.0)),
        "incremental delay should use x/(x+160) normalization.",
    )
    _assert(
        normalized_pair_features.shape == (8,),
        "incremental delay normalization must not change pair feature dimension.",
    )

    np.random.seed(41)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        env = Env()
        env.reset()
        source = np.array([100.0, 100.0], dtype=np.float32)
        env.uavs[0].pos[:2] = source
        env.uavs[1].pos[:2] = np.array([450.0, 450.0], dtype=np.float32)
        ue = env.ues[0]
        ue.pos[:2] = source.copy()
        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=source.copy(),
            current_time_step=env.current_time_seconds,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()
        ready_tasks = env.task_manager.get_ready_tasks()
        _assert(ready_tasks, "test DAG should expose a ready task.")
        task = ready_tasks[0]
        task.input_data_size_mb = 20.0
        task.num_operation = float(config.UAV_COMPUTE_RATE_OPS_PER_SEC) * 2.0

        reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
        _, pair_features, candidate_mask, _, estimates = build_offloading_candidate_components(
            task=task,
            uavs=env.uavs[:2],
            task_manager=env.task_manager,
            executor=env.executor,
            state_view=reservation,
            current_time_seconds=env.current_time_seconds,
            uav_service_positions={int(uav.id): uav.pos[:2].copy() for uav in env.uavs[:2]},
            ue_service_positions={int(ue.id): ue.pos[:2].copy()},
            ues=[ue],
        )
        _assert(candidate_mask.shape == (2,), "candidate mask should cover both test UAVs.")
        _assert(bool(candidate_mask.all()), "both test UAVs should be legal candidates.")
        _assert(np.all(np.isfinite(pair_features)), "pair features must be finite.")
        _assert(pair_features.shape[0] == 2, "pair feature rows should match test UAVs.")

        feature_range = np.ptp(pair_features, axis=0)
        _assert(float(feature_range.max()) > 0.05, "pair features are still too compressed across candidates.")
        finish_times = [float(estimate.estimated_finish_time) for estimate in estimates]
        _assert(max(finish_times) > min(finish_times), "near/far candidates should produce different finish times.")
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob

    print("smoke_clean_feature_normalization passed")


if __name__ == "__main__":
    main()
