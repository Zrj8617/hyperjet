from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.capacity_factorial_diagnostic import (
    CELL_FLAGS,
    EPISODE_SLOTS,
    FORMAL_EPISODES,
    FORMAL_SCENARIO_SEEDS,
    LOAD_SLOTS,
    PILOT_EPISODES,
    analyze_factorial_rows,
    index_tape_episodes,
    load_scenario_tape,
    load_stage1_actor_policy,
    run_factorial_episode,
)


POLICIES = ("random_hash", "greedy_eft", "stage1_actor")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed-policy active-DAG/queue-cap factorial.")
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=PILOT_EPISODES)
    parser.add_argument("--cells", nargs="+", choices=tuple(CELL_FLAGS), default=tuple(CELL_FLAGS))
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--scenario-seeds", nargs="+", type=int, default=list(FORMAL_SCENARIO_SEEDS))
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not 1 <= int(args.episodes) <= FORMAL_EPISODES:
        raise ValueError(f"--episodes must be in [1, {FORMAL_EPISODES}]")
    tape = load_scenario_tape(args.tape)
    _assert_formal_tape_controls(tape)
    requested_seeds = tuple(int(value) for value in args.scenario_seeds)
    if set(requested_seeds) - set(FORMAL_SCENARIO_SEEDS):
        raise ValueError("runner scenario seeds must be a subset of the frozen formal seeds")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = args.output_dir / "episode_rows.jsonl"
    tape_index = index_tape_episodes(tape)
    actor_cache: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with rows_path.open("x", encoding="utf-8") as handle:
        for policy in args.policies:
            for scenario_seed in requested_seeds:
                actor_policy = None
                if policy == "stage1_actor":
                    if scenario_seed not in actor_cache:
                        actor_cache[scenario_seed] = load_stage1_actor_policy(
                            root=ROOT,
                            scenario_seed=scenario_seed,
                            device=args.device,
                        )
                    actor_policy = actor_cache[scenario_seed]
                for episode in range(int(args.episodes)):
                    episode_payload = tape_index[(scenario_seed, episode)]
                    for cell in args.cells:
                        row = run_factorial_episode(
                            episode_payload=episode_payload,
                            cell=cell,
                            policy=policy,
                            full_tape_checksum=tape["full_tape_checksum"],
                            pilot_prefix_checksum=tape["pilot_prefix_checksum"],
                            actor_policy=actor_policy,
                        )
                        rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                        handle.flush()
    analysis = analyze_factorial_rows(
        rows,
        policies=args.policies,
        scenario_seeds=requested_seeds,
        episode_count=int(args.episodes),
        expected_full_tape_checksum=tape["full_tape_checksum"],
        expected_prefix_checksum=tape["pilot_prefix_checksum"],
    )
    summary = {
        "technical_pass": bool(analysis["technical_pass"]),
        "row_count": len(rows),
        "episodes": int(args.episodes),
        "cells": list(args.cells),
        "policies": list(args.policies),
        "scenario_seeds": list(requested_seeds),
        "full_tape_checksum": tape["full_tape_checksum"],
        "pilot_prefix_checksum": tape["pilot_prefix_checksum"],
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_cpu_rss_mb": _peak_rss_mb(),
        "peak_cuda_memory_mb": _peak_cuda_memory_mb(),
        "gate_errors": analysis["gate_errors"],
    }
    (args.output_dir / "paired_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not analysis["technical_pass"]:
        raise RuntimeError("factorial pilot gate failed: " + "; ".join(analysis["gate_errors"][:10]))
    return 0


def _assert_formal_tape_controls(tape: dict[str, Any]) -> None:
    controls = tape["controls"]
    expected = {
        "scenario_seeds": list(FORMAL_SCENARIO_SEEDS),
        "episodes": FORMAL_EPISODES,
        "load_slots": LOAD_SLOTS,
        "episode_slots": EPISODE_SLOTS,
        "num_ues": 60,
        "num_uavs": 5,
    }
    for key, value in expected.items():
        if controls.get(key) != value:
            raise ValueError(f"formal tape control mismatch for {key}: {controls.get(key)!r}")
    if int(tape["pilot_prefix_episode_count"]) != PILOT_EPISODES:
        raise ValueError("formal tape pilot prefix must contain episodes 0-4")


def _peak_rss_mb() -> float | None:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / 1024.0
    except (ImportError, AttributeError):
        return None


def _peak_cuda_memory_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except ModuleNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
