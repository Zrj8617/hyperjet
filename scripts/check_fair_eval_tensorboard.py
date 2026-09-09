"""Verify the nine fixed-tape fair-evaluation TensorBoard runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from export_fair_eval_tensorboard import ARMS, CHECKPOINT_LABELS, METRICS, PROTOCOLS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    expected_tags = {
        f"{protocol}/{checkpoint}/{metric}"
        for protocol in PROTOCOLS
        for checkpoint in CHECKPOINT_LABELS
        for metric in METRICS
    }
    failures = []
    runs = []
    for arm in ARMS:
        for seed in range(3):
            run_name = f"FormalEval/{arm}/seed{seed}"
            event_dir = args.root / arm / f"seed{seed}"
            accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
            actual_tags = set(accumulator.Tags().get("scalars", []))
            if actual_tags != expected_tags:
                failures.append(
                    {
                        "run": run_name,
                        "missing_tags": sorted(expected_tags - actual_tags),
                        "extra_tags": sorted(actual_tags - expected_tags),
                    }
                )
            bad_series = []
            for tag in actual_tags:
                events = accumulator.Scalars(tag)
                if len(events) != 50 or [event.step for event in events] != list(range(200, 250)):
                    bad_series.append(tag)
            if bad_series:
                failures.append({"run": run_name, "bad_series": sorted(bad_series)})
            runs.append(
                {
                    "run": run_name,
                    "scalar_tag_count": len(actual_tags),
                    "points_per_tag": 50,
                }
            )

    result = {
        "status": "pass" if not failures else "fail",
        "expected_run_count": 9,
        "expected_scalar_tags_per_run": len(expected_tags),
        "failures": failures,
        "runs": runs,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
