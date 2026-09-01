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
    _dynamic_uav_features,
    _normalize_pair_features,
    _soft_delay_norm,
)


class _Uav:
    id = 0
    pos = np.asarray([100.0, 200.0, config.UAV_ALTITUDE], dtype=np.float32)


MAX_REACHABLE_DELAY_SECONDS = 1e6


def main() -> int:
    ref = float(config.CLEAN_NORM_DELAY_SOFT_REF)
    assert ref > 0.0

    # About 1000 times the roughly 1000-second episode horizon, so this is a
    # conservative physical upper bound for all reachable delay features.
    inputs = (0.0, 1.0, 39.0, 40.0, 41.0, 160.0, MAX_REACHABLE_DELAY_SECONDS)
    outputs = tuple(_soft_delay_norm(value, ref) for value in inputs)
    assert all(0.0 <= value < 1.0 for value in outputs)
    assert all(left < right for left, right in zip(outputs, outputs[1:]))
    assert _soft_delay_norm(-1.0, ref) == 0.0

    # At the float64 representation limit x + ref rounds back to x (the
    # crossover is about 1.4e18 seconds for this ref), far beyond reachability.
    assert _soft_delay_norm(1e30, ref) == 1.0

    # Features are ultimately stored as float32. A physically meaningful
    # 20-second candidate gap must remain distinguishable after that cast.
    resolution_inputs = (0.0, 10.0, 47.0, 100.0, 300.0, 1000.0, 10000.0)
    for value in resolution_inputs:
        normalized_pair = np.asarray(
            [
                _soft_delay_norm(value, ref),
                _soft_delay_norm(value + 20.0, ref),
            ],
            dtype=np.float32,
        )
        assert normalized_pair[0] != normalized_pair[1]
        assert int(np.argmin(normalized_pair)) == 0

    delays = np.asarray([140.0, 60.0, 80.0, 41.0], dtype=np.float64)
    normalized = np.asarray([_soft_delay_norm(value, ref) for value in delays])
    assert int(np.argmin(normalized)) == int(np.argmin(delays))

    raw_pair = np.asarray([10.0, 2.0, 20.0, 5.0, 3.0, 80.0, 4.0, 1.0])
    pair = _normalize_pair_features(raw_pair.tolist())
    old_pair = np.clip(
        raw_pair
        / np.asarray(
            [
                config.CLEAN_NORM_PAIR_TIME_REF,
                config.CLEAN_NORM_PAIR_ENERGY_REF,
                config.CLEAN_NORM_AVAIL_TIME_REF,
                config.CLEAN_NORM_PAIR_TIME_REF,
                config.CLEAN_NORM_PAIR_ENERGY_REF,
                config.CLEAN_NORM_AVAIL_TIME_REF,
                config.CLEAN_NORM_PAIR_TIME_REF,
                config.CLEAN_NORM_PAIR_ENERGY_REF,
            ],
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )
    assert pair.shape == (8,)
    assert np.allclose(pair[[0, 1, 3, 4, 6, 7]], old_pair[[0, 1, 3, 4, 6, 7]])
    assert np.isclose(pair[2], _soft_delay_norm(raw_pair[2], ref))
    assert np.isclose(pair[5], _soft_delay_norm(raw_pair[5], ref))

    state = TemporaryReservationState(
        queue_lengths={0: 4},
        available_times={0: 80.0},
        queued_workloads={0: 1_000_000.0},
        slot_assigned_counts={0: 3},
    )
    dynamic = _dynamic_uav_features(
        uav=_Uav(),
        uav_id=0,
        state_view=state,
        current_time_seconds=0.0,
        uav_service_positions={0: _Uav.pos[:2]},
    )
    old_dynamic = np.asarray(
        [
            100.0 / config.AREA_WIDTH,
            200.0 / config.AREA_HEIGHT,
            4.0 / config.CLEAN_MAX_QUEUE_PER_UAV,
            12.0 / config.CLEAN_MAX_QUEUE_PER_UAV,
            1.0,
            1_000_000.0 / config.CLEAN_NORM_QUEUE_WORKLOAD_REF,
            3.0 / config.CLEAN_MAX_QUEUE_PER_UAV,
        ],
        dtype=np.float32,
    )
    assert dynamic.shape == (7,)
    assert np.allclose(dynamic[[0, 1, 2, 3, 5, 6]], old_dynamic[[0, 1, 2, 3, 5, 6]])
    assert np.isclose(dynamic[4], _soft_delay_norm(80.0, ref))

    print("SMOKE_CLEAN_SOFT_DELAY_NORMALIZATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
