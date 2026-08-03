from __future__ import annotations

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.capacity_factorial_diagnostic import RESULT_SCHEMA, analyze_factorial_rows


def _row(cell: str, value: float | None) -> dict:
    base = {
        "schema": RESULT_SCHEMA,
        "cell": cell,
        "policy": "random_hash",
        "scenario_seed": 42,
        "episode": 0,
        "scenario_checksum": "s",
        "offered_event_checksum": "e",
        "offered_template_checksum": "t",
        "full_tape_checksum": "f",
        "pilot_prefix_checksum": "p",
        "technical_pass": True,
        "finite": True,
        "active_cap_blocked_offered_count": 0,
        "queue_full_mask_count": 0,
        "invalid_assignment_reasons": {},
        "completed_dag_flowtime_sum": 0.0,
        "completed_dag_flowtime_count": 0,
    }
    metrics = (
        "choice_decision_fraction",
        "queue_full_mask_count",
        "all_uavs_full_decision_count",
        "repeated_ready_attempt_count",
        "completed_dag_per_slot",
        "dag_completion_rate_offered",
        "average_dag_flowtime",
        "episode_end_admitted_incomplete_count",
        "dag_completion_rate_admitted",
        "episode_reward_total",
        "choice_decision_count",
        "forced_decision_count",
        "skip_decision_count",
        "admitted_dag_count",
        "completed_dag_count",
        "avg_uav_queue_length",
    )
    base.update({metric: value for metric in metrics})
    base["queue_full_mask_count"] = 0
    return base


def main() -> int:
    rows = [_row("A", 1.0), _row("B", 3.0), _row("C", 5.0), _row("D", 11.0)]
    result = analyze_factorial_rows(
        rows,
        policies=("random_hash",),
        scenario_seeds=(42,),
        episode_count=1,
        expected_full_tape_checksum="f",
        expected_prefix_checksum="p",
    )
    effect = result["paired_episode_effects"][0]["effects"]["episode_reward_total"]
    assert effect == {"active": 4.0, "queue": 6.0, "interaction": 4.0}
    assert result["technical_pass"] is True

    null_rows = copy.deepcopy(rows)
    null_rows[2]["average_dag_flowtime"] = None
    null_result = analyze_factorial_rows(
        null_rows,
        policies=("random_hash",),
        scenario_seeds=(42,),
        episode_count=1,
    )
    null_effect = null_result["paired_episode_effects"][0]["effects"]["average_dag_flowtime"]
    assert null_effect == {"active": None, "queue": None, "interaction": None}
    counts = null_result["seed_level_effects"]["random_hash"]["42"]["average_dag_flowtime"]
    assert counts["valid_paired_episode_count"] == 0
    assert counts["null_paired_episode_count"] == 1

    broken = copy.deepcopy(rows)
    broken[1]["scenario_checksum"] = "different"
    try:
        analyze_factorial_rows(broken, policies=("random_hash",), scenario_seeds=(42,), episode_count=1)
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("paired checksum mismatch was accepted")
    try:
        analyze_factorial_rows(rows[:-1], policies=("random_hash",), scenario_seeds=(42,), episode_count=1)
    except ValueError as error:
        assert "row key mismatch" in str(error)
    else:
        raise AssertionError("missing factorial cell was accepted")
    print("SMOKE_ACTIVE_DAG_QUEUE_CAP_PAIRING_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
