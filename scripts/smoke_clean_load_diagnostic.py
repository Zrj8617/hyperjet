from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.task_execution import CleanExecutionStepStats
from scripts import diag_clean_load as diagnostic


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    summary = diagnostic._distribution_summary([0.0, 1.0, 2.0], "sample")
    _assert(summary["sample_mean"] == 1.0, "distribution mean mismatch")
    _assert(summary["sample_max"] == 2.0, "distribution max mismatch")
    _assert(diagnostic._distribution_summary([], "empty")["empty_p90"] == 0.0, "empty percentile mismatch")
    _assert(diagnostic._longest_true_run([False, True, True, False, True]) == 2, "run length mismatch")
    trend = diagnostic._tail_trend([1.0, 2.0, 3.0], "trend")
    _assert(trend["trend_delta"] == 2.0 and trend["trend_slope_per_slot"] == 1.0, "trend mismatch")

    _assert(
        diagnostic._capacitated_matching_size({"a": [0], "b": [1]}, {0: 1, 1: 1}) == 2,
        "disjoint matching mismatch",
    )
    _assert(
        diagnostic._capacitated_matching_size({"a": [0], "b": [0]}, {0: 1, 1: 1}) == 1,
        "overlapping matching double-counted capacity",
    )
    _assert(
        diagnostic._capacitated_matching_size(
            {"a": [0, 1], "b": [0], "c": [0], "d": [1]},
            {0: 2, 1: 1},
        )
        == 3,
        "unequal-capacity matching mismatch",
    )
    _assert(
        diagnostic._capacitated_matching_size({"a": [0]}, {0: 0}) == 0,
        "zero-capacity matching mismatch",
    )

    rejection_stats = CleanExecutionStepStats()
    rejection_stats.record_invalid_assignment("malformed_uav_id")
    rejection_stats.record_invalid_assignment("illegal_assignment")
    rejection_stats.record_invalid_assignment("schedule_record_failure")
    _assert(rejection_stats.invalid_assignments == 3, "invalid assignment aggregate mismatch")
    _assert(sum(rejection_stats.invalid_assignment_reasons.values()) == 3, "invalid reason conservation mismatch")

    original_partition = config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES
    original_base = config.DAG_BASE_ARRIVAL_PROB
    original_alias = config.DAG_ARRIVAL_PROB
    try:
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = False
        override = os.environ.get("HYPERUAV_SMOKE_TMP")
        temp_context = (
            nullcontext(override)
            if override
            else tempfile.TemporaryDirectory(prefix="hyperuav_load_diag_")
        )
        with temp_context as tmp:
            output_dir = Path(tmp) / "sweep"
            exit_code = diagnostic.main(
                [
                    "--sweep",
                    "--slots",
                    "3",
                    "--drain-slots",
                    "2",
                    "--seeds",
                    "7",
                    "--policies",
                    "greedy",
                    "random",
                    "--arrival-probs",
                    "0.5,1.0",
                    "--input-ranges",
                    "0.75:14",
                    "--output-ranges",
                    "0.6:10.5",
                    "--task-constant-ranges",
                    "6:60",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            _assert(exit_code == 0, "sweep returned nonzero")
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (output_dir / "sweep_rows.jsonl").read_text().splitlines()]
            summaries = json.loads((output_dir / "sweep_summary.json").read_text(encoding="utf-8"))
            _assert(manifest["cell_count"] == 4, "manifest cell count mismatch")
            _assert(manifest["schema_version"] == 2, "manifest schema mismatch")
            _assert(progress["status"] == "completed" and progress["completed_cells"] == 4, "progress mismatch")
            _assert(len(rows) == 4 and len(summaries) == 2, "persisted row/summary count mismatch")
            _assert(all(row["drain_slots_max"] == 2 for row in rows), "sweep did not forward drain slots")
            _assert(all("active_dags_per_arrival_slot_p90" in row for row in rows), "concurrency metrics missing")
            _assert(all("partition_status_counts" in row for row in rows), "partition provenance missing")
            _assert(all("arrival_completion_rate" in row for row in rows), "arrival completion missing")
            _assert(all("final_completion_rate" in row for row in rows), "final completion missing")
            _assert(
                all(row["arrival_completion_rate"] == row["completion_rate_arrival_end"] for row in rows),
                "arrival completion alias mismatch",
            )
            _assert(
                all(row["final_completion_rate"] == row["completion_rate"] for row in rows),
                "final completion alias mismatch",
            )
            _assert(
                all(row["drain_slots_used"] == row["drain_slots_executed"] for row in rows),
                "drain slot alias mismatch",
            )
            _assert(all("arrival_slot_funnel" in row for row in rows), "slot funnel missing")
            _assert(all(row["funnel_monotonicity_violation_count"] == 0 for row in rows), "funnel not monotone")
            _assert(all(row["executor_record_count_mismatch_count"] == 0 for row in rows), "executor record mismatch")
            _assert(all(row["executor_invalid_reason_mismatch_count"] == 0 for row in rows), "executor reason mismatch")
            _assert(all("capacity_blocked_task_slot_ratio" in row for row in rows), "blocking ratio missing")
            _assert(all("legal_capacity_utilization_nonzero_ceiling_mean" in row for row in rows), "utilization missing")
            for row in rows:
                for slot in row["arrival_slot_funnel"]:
                    _assert(
                        slot["executor_accepted_count"]
                        <= slot["assignment_buffer_accepted_count"]
                        <= slot["policy_selected_count"]
                        <= slot["frozen_ready_count"],
                        "slot funnel monotonicity mismatch",
                    )
                    _assert(
                        slot["executor_accepted_count"] == slot["new_executor_record_count"],
                        "newly-assigned/executor-record mismatch",
                    )
                    _assert(
                        sum(slot["queue_post_decision"])
                        == sum(slot["queue_pre_decision"]) + slot["executor_accepted_count"],
                        "post-decision queue reconstruction mismatch",
                    )
                    _assert(
                        sum(slot["reason_counts"].get(name, 0) for name in (
                            "executor_malformed_uav_id",
                            "executor_illegal_assignment",
                            "executor_schedule_record_failure",
                        ))
                        == slot["executor_invalid_count"],
                        "executor rejection reason mismatch",
                    )
            paired = {}
            for row in rows:
                key = (row["dag_base_arrival_prob"], row["seed"])
                paired.setdefault(key, []).append(row)
            _assert(
                all(len(group) == 2 and group[0]["generated_dags"] == group[1]["generated_dags"] for group in paired.values()),
                "policy RNG changed paired arrival trajectories",
            )
            _assert(all(not row["kahypar_worker_alive_after_close"] for row in rows), "graph worker leaked")
            _assert((output_dir / "sweep_summary.csv").is_file(), "summary CSV missing")
            _assert((output_dir / "analysis_report.md").is_file(), "analysis report missing")
            _assert(config.DAG_BASE_ARRIVAL_PROB == original_base, "sweep did not restore arrival probability")
    finally:
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = original_partition
        config.DAG_BASE_ARRIVAL_PROB = original_base
        config.DAG_ARRIVAL_PROB = original_alias

    print("clean load diagnostic smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
