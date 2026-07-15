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
from scripts import diag_clean_load as diagnostic


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    summary = diagnostic._distribution_summary([0.0, 1.0, 2.0], "sample")
    _assert(summary["sample_mean"] == 1.0, "distribution mean mismatch")
    _assert(summary["sample_max"] == 2.0, "distribution max mismatch")
    _assert(diagnostic._distribution_summary([], "empty")["empty_p90"] == 0.0, "empty percentile mismatch")

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
            _assert(progress["status"] == "completed" and progress["completed_cells"] == 4, "progress mismatch")
            _assert(len(rows) == 4 and len(summaries) == 2, "persisted row/summary count mismatch")
            _assert(all(row["drain_slots_max"] == 2 for row in rows), "sweep did not forward drain slots")
            _assert(all("active_dags_per_arrival_slot_p90" in row for row in rows), "concurrency metrics missing")
            _assert(all("partition_status_counts" in row for row in rows), "partition provenance missing")
            _assert(all(not row["kahypar_worker_alive_after_close"] for row in rows), "graph worker leaked")
            _assert((output_dir / "sweep_summary.csv").is_file(), "summary CSV missing")
            _assert((output_dir / "analysis_report.md").is_file(), "analysis report missing")
    finally:
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = original_partition
        config.DAG_BASE_ARRIVAL_PROB = original_base
        config.DAG_ARRIVAL_PROB = original_alias

    print("clean load diagnostic smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
