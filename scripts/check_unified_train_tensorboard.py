"""Verify UnifiedTrain scalar names, counts, and step semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from export_unified_train_tensorboard import RUN_NAMES, TAGS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    failures = []
    runs = []
    for run_name in RUN_NAMES:
        accumulator = EventAccumulator(
            str(args.root / run_name), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        actual_tags = set(accumulator.Tags().get("scalars", []))
        if actual_tags != set(TAGS):
            failures.append(
                {
                    "run": run_name,
                    "missing": sorted(set(TAGS) - actual_tags),
                    "extra": sorted(actual_tags - set(TAGS)),
                }
            )
            continue

        episode_events = accumulator.Scalars("train/episode_reward")
        critic_events = accumulator.Scalars("train/critic_loss")
        if (
            len(episode_events) != 500
            or episode_events[0].step != 500
            or episode_events[-1].step != 250000
        ):
            failures.append({"run": run_name, "bad_episode_series": True})
        if len(critic_events) != 2000 or critic_events[-1].step != 250000:
            failures.append({"run": run_name, "bad_critic_series": True})
        runs.append(
            {
                "run": f"UnifiedTrain/{run_name}",
                "episode_points": len(episode_events),
                "critic_points": len(critic_events),
            }
        )

    result = {
        "status": "pass" if not failures else "fail",
        "expected_run_count": 33,
        "failures": failures,
        "runs": runs,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
