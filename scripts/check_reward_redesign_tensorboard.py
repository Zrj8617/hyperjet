from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ARMS = ("N0", "A2", "B1", "B2", "B2D", "D1", "D2")
EXPECTED = {
    "offload/entropy",
    "eval/completion_rate",
    "eval/avg_dag_flowtime",
    "eval/throughput",
    "eval/agreement_eft_greedy",
    "train/critic_ev",
    "train/episode_reward",
    "train/actor_loss",
    "train/critic_loss",
    "train/approx_kl",
    "move/move_rate_m",
    "forecast/wall_time_frac",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    failures = []
    for arm in ARMS:
        for seed in range(3):
            run_name = f"20260906_{arm}_seed{seed}"
            event_dir = args.root / run_name / "tensorboard"
            accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
            actual = set(accumulator.Tags().get("scalars", []))
            if actual != EXPECTED:
                failures.append(
                    {
                        "run": run_name,
                        "missing": sorted(EXPECTED - actual),
                        "extra": sorted(actual - EXPECTED),
                    }
                )
            cells.append(
                {
                    "run": run_name,
                    "scalar_tags": sorted(actual),
                    "scalar_count": len(actual),
                }
            )
    print(json.dumps({"status": "pass" if not failures else "fail", "failures": failures, "cells": cells}))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
