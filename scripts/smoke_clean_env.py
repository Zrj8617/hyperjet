from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import config
from environment.env import Env


def _inside_map_xy(pos: np.ndarray) -> bool:
    return 0.0 <= float(pos[0]) <= float(config.AREA_WIDTH) and 0.0 <= float(pos[1]) <= float(config.AREA_HEIGHT)


def main() -> None:
    np.random.seed(config.SEED)
    env = Env()
    obs = env.reset()

    assert len(obs) == config.NUM_UAVS
    assert len(env.uavs) == config.NUM_UAVS == 8
    assert len(env.ues) == config.NUM_UES == 100
    assert env.hotspot_center is not None
    assert float(env.hotspot_radius) == float(config.HOTSPOT_RADIUS) == 200.0
    assert config.HOTSPOT_RADIUS <= float(env.hotspot_center[0]) <= config.AREA_WIDTH - config.HOTSPOT_RADIUS
    assert config.HOTSPOT_RADIUS <= float(env.hotspot_center[1]) <= config.AREA_HEIGHT - config.HOTSPOT_RADIUS

    for uav in env.uavs:
        assert _inside_map_xy(uav.pos[:2])
    for ue in env.ues:
        assert _inside_map_xy(ue.pos[:2])
        assert ue.active_dag_id is None
        assert not ue.service_waiting

    old_base_prob = config.DAG_BASE_ARRIVAL_PROB
    try:
        config.DAG_BASE_ARRIVAL_PROB = 1.0
        target = env.ues[0]
        target.pos[:2] = env.hotspot_center.copy()
        source_pos = target.pos[:2].copy()
        for ue in env.ues[1:]:
            ue.active_dag_id = "blocked_for_smoke"

        created = env._process_clean_dag_arrivals()
        assert created == 1
        assert target.service_waiting
        assert target.active_dag_id is not None
        job = env.task_manager.get_job(target.active_dag_id)
        assert job is not None
        np.testing.assert_allclose(job.source_pos, source_pos)

        target.pos[:2] = np.array([0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(job.source_pos, source_pos)
        assert env._process_clean_dag_arrivals() == 0
        assert len([job for job in env.task_manager.jobs.values() if job.ue_id == target.id]) == 1

        outside = env.ues[1]
        outside.active_dag_id = None
        outside.pos[:2] = np.array([0.0, 0.0], dtype=np.float32)
        setattr(outside, "is_hotspot", True)
        config.DAG_BASE_ARRIVAL_PROB = 0.0
        assert env._process_clean_dag_arrivals() == 0
        assert outside.active_dag_id is None
    finally:
        config.DAG_BASE_ARRIVAL_PROB = old_base_prob

    for _ in range(20):
        obs, reward, done, info = env.step(np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32))
        assert len(obs) == config.NUM_UAVS
        assert len(reward) == config.NUM_UAVS
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    print("clean env smoke passed")


if __name__ == "__main__":
    main()
