from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _advance_without_actions(env: Env) -> dict:
    env.prepare_slot_state()
    env.apply_movement({})
    _, _, _, info = env.commit_and_advance()
    return info


def main() -> int:
    original_probability = config.DAG_BASE_ARRIVAL_PROB
    try:
        config.DAG_BASE_ARRIVAL_PROB = 1.0
        env = Env(freeze_ue_mobility=True, max_active_dags_per_ue=1)
        env.reset()

        first = _advance_without_actions(env)
        ue_count = len(env.ues)
        _assert(first["step_arrival_attempt_count"] == ue_count, "every UE must be considered")
        _assert(first["step_arrival_draw_count"] == ue_count, "all idle UEs must receive a draw")
        _assert(first["step_arrival_sampled_event_count"] == ue_count, "probability one must sample every idle UE")
        _assert(first["step_arrival_admitted_count"] == ue_count, "every sampled idle UE must be admitted")
        _assert(first["step_arrival_blocked_count"] == 0, "first slot must not be cap-blocked")
        _assert(first["step_arrival_no_event_count"] == 0, "probability one cannot produce no-event")

        second = _advance_without_actions(env)
        _assert(second["step_arrival_attempt_count"] == ue_count, "attempt semantics drifted")
        _assert(second["step_arrival_draw_count"] == 0, "cap-blocked UEs must not consume arrival RNG")
        _assert(second["step_arrival_sampled_event_count"] == 0, "blocked UEs must not be sampled")
        _assert(second["step_arrival_admitted_count"] == 0, "blocked UEs cannot be admitted")
        _assert(second["step_arrival_blocked_count"] == ue_count, "all active UEs must be blocked")
        _assert(
            second["step_arrival_blocked_reasons"] == {"active_dag_cap": ue_count},
            "active-cap reason accounting drifted",
        )
        _assert(second["arrival_attempt_count"] == 2 * ue_count, "cumulative attempts drifted")
        _assert(second["arrival_admitted_count"] == ue_count, "cumulative admissions drifted")
        _assert(second["arrival_blocked_count"] == ue_count, "cumulative blocks drifted")
        _assert(
            second["arrival_attempt_count"]
            == second["arrival_blocked_count"] + second["arrival_draw_count"],
            "attempts must partition into blocked opportunities and eligible draws",
        )
        _assert(
            second["arrival_draw_count"]
            == second["arrival_sampled_event_count"] + second["arrival_no_event_count"],
            "eligible draws must partition into sampled events and no-events",
        )
        _assert(
            second["arrival_sampled_event_count"] == second["arrival_admitted_count"],
            "the current clean path must admit every sampled eligible event",
        )

        env.reset()
        reset_info = env.latest_info
        _assert(reset_info["arrival_attempt_count"] == 0, "episode reset must clear totals")
        _assert(reset_info["step_arrival_attempt_count"] == 0, "episode reset must clear step data")
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_probability

    print("smoke_clean_arrival_funnel passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
