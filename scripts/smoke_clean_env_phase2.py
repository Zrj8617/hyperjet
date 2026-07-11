from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import config
from environment.env import Env


def _assert_xy_inside_map(pos: np.ndarray) -> None:
    assert 0.0 <= float(pos[0]) <= float(config.AREA_WIDTH)
    assert 0.0 <= float(pos[1]) <= float(config.AREA_HEIGHT)


def main() -> None:
    np.random.seed(config.SEED)
    env = Env()
    env.reset()

    # Assert against config only; hardcoded scene numbers (8/100/200.0) belonged
    # to an older scene revision and go stale whenever the clean scene changes.
    assert len(env.uavs) == config.NUM_UAVS
    assert len(env.ues) == config.NUM_UES
    assert env.hotspot_center is not None
    assert float(env.hotspot_radius) == float(config.HOTSPOT_RADIUS)
    assert config.HOTSPOT_RADIUS <= float(env.hotspot_center[0]) <= config.AREA_WIDTH - config.HOTSPOT_RADIUS
    assert config.HOTSPOT_RADIUS <= float(env.hotspot_center[1]) <= config.AREA_HEIGHT - config.HOTSPOT_RADIUS

    for ue in env.ues:
        _assert_xy_inside_map(ue.pos[:2])
        assert ue.active_dag_id is None
        assert not ue.service_waiting
        assert not getattr(ue, "is_hotspot", False)
    for uav in env.uavs:
        _assert_xy_inside_map(uav.pos[:2])

    old_base_prob = config.DAG_BASE_ARRIVAL_PROB
    try:
        config.DAG_BASE_ARRIVAL_PROB = 1.0
        ue = env.ues[0]
        for other_ue in env.ues[1:]:
            other_ue.active_dag_id = "preexisting_dag"
        ue.pos[:2] = env.hotspot_center.copy()
        source_at_arrival = ue.pos[:2].copy()
        created_count = env._process_clean_dag_arrivals()
        assert created_count == 1
        assert ue.service_waiting
        assert ue.active_dag_id is not None
        job = env.task_manager.get_job(ue.active_dag_id)
        assert job is not None
        np.testing.assert_allclose(job.source_pos, source_at_arrival)

        ue.pos[:2] = np.array([0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(job.source_pos, source_at_arrival)
        second_count = env._process_clean_dag_arrivals()
        assert second_count == 0
        assert len([job for job in env.task_manager.jobs.values() if job.ue_id == ue.id]) == 1
    finally:
        config.DAG_BASE_ARRIVAL_PROB = old_base_prob

    moving_ue = env.ues[1]
    moving_ue.service_waiting = False
    moving_ue.speed = float(config.UE_WALK_SPEED_MEAN)
    moving_ue.theta = 0.0
    moving_ue.velocity = moving_ue._velocity_from_polar()
    normal_velocity_norm = float(np.linalg.norm(moving_ue.velocity))

    moving_ue.enter_service_waiting("manual_dag")
    moving_ue.speed = float(config.UE_WALK_SPEED_MEAN)
    moving_ue.theta = 0.0
    moving_ue.velocity = moving_ue._velocity_from_polar()
    moving_ue.update_position()
    waiting_velocity_norm = float(np.linalg.norm(moving_ue.velocity))
    assert waiting_velocity_norm <= normal_velocity_norm * config.UE_SERVICE_WAITING_SPEED_SCALE + 1e-6
    assert waiting_velocity_norm > 0.0

    env.release_ue_after_dag_completed("manual_dag")
    assert moving_ue.active_dag_id is None
    assert not moving_ue.service_waiting

    print("clean env phase2 smoke passed")


if __name__ == "__main__":
    main()
