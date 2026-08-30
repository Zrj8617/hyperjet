from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--movement-control", action="store_true")
    args = parser.parse_args()
    checkpoint_pattern = (
        "*/checkpoints/checkpoint_update_0000.pt"
        if args.initial_only
        else "*/checkpoints/latest.pt"
    )
    baseline_checkpoint = _single(args.baseline, checkpoint_pattern)
    current_checkpoint = _single(args.current, checkpoint_pattern)
    baseline = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
    current = torch.load(current_checkpoint, map_location="cpu", weights_only=False)
    if args.movement_control:
        baseline_summary = json.loads(_single(args.baseline, "*/run_summary.json").read_text())
        current_summary = json.loads(_single(args.current, "*/run_summary.json").read_text())
        left = baseline_summary["latest_update"]
        right = current_summary["latest_update"]
        for key in ("movement_action_count", "movement_entropy", "movement_loss"):
            if not np.isclose(float(left[key]), float(right[key]), rtol=0.0, atol=1e-7):
                raise AssertionError(f"movement control mismatch: {key}")
        for key in (
            "movement_position_advantage_count",
            "movement_position_advantage_mean",
            "movement_position_advantage_std",
            "grad_pre_clip_movement",
        ):
            if not np.isclose(
                float(left["diagnostics"][key]),
                float(right["diagnostics"][key]),
                rtol=0.0,
                atol=1e-7,
            ):
                raise AssertionError(f"movement control mismatch: {key}")
        movement_equal = _equal(baseline["movement_actor"], current["movement_actor"])
        print(
            "movement control PASS: advantage/loss input and preclip gradient unchanged; "
            f"post-update tensor equality={movement_equal}"
        )
        return
    for key in ("hgnn", "movement_actor", "offloading_actor", "critic", "optimizer", "rng_state"):
        if not _equal(baseline[key], current[key]):
            raise AssertionError(f"gate-OFF checkpoint mismatch: {key}")
    if args.initial_only:
        print("gate ON initialization neutrality PASS: base modules/optimizer/RNG identical")
        return
    baseline_summary = json.loads(_single(args.baseline, "*/run_summary.json").read_text())
    current_summary = json.loads(_single(args.current, "*/run_summary.json").read_text())
    for key in (
        "global_slot",
        "completed_update_count",
        "episode_reward",
        "latest_info",
        "latest_update",
    ):
        if not _equal(baseline_summary[key], current_summary[key]):
            raise AssertionError(f"gate-OFF run summary mismatch: {key}")
    print("gate OFF exact-commit equivalence PASS: modules/optimizer/RNG/reward/PPO update identical")


if __name__ == "__main__":
    main()
