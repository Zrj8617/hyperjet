from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
GPUS = (0, 4, 6)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the approved C2 production runs once.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.rstrip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"existing result root required: {output_root}")
    smoke_result = _read_json(args.smoke_result.resolve())
    if smoke_result.get("status") != "completed" or smoke_result.get("arm") != "C2":
        raise ValueError("completed C2 smoke result required")
    smoke_episodes = int(smoke_result["actual_parameters"]["episodes"])
    if smoke_episodes != 5:
        raise ValueError("C2 smoke must contain exactly 5 episodes")
    training_seconds_per_episode = float(smoke_result["training_wall_seconds"]) / 5.0
    evaluation_seconds = float(smoke_result.get("evaluation_wall_seconds", 0.0))
    launched_at = datetime.now(timezone.utc)
    runs: list[dict[str, Any]] = []
    for seed, gpu in zip(SEEDS, GPUS):
        run_name = f"20260908_C2_seed{seed}"
        run_root = output_root / run_name
        log_path = output_root / f"{run_name}.log"
        if run_root.exists() or log_path.exists():
            raise FileExistsError(f"production target already exists: {run_name}")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_reward_redesign_arm.py"),
            "--arm", "C2",
            "--seed", str(seed),
            "--episodes", str(int(args.episodes)),
            "--steps", str(int(args.steps)),
            "--device", "cuda:0",
            "--output-root", str(run_root),
            "--run-name", run_name,
        ]
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        estimated_seconds = training_seconds_per_episode * float(args.episodes) + evaluation_seconds
        eta = launched_at + timedelta(seconds=estimated_seconds)
        runs.append(
            {
                "run_name": run_name,
                "arm": "C2",
                "seed": int(seed),
                "gpu": int(gpu),
                "pid": int(process.pid),
                "log_path": str(log_path),
                "result_path": str(run_root / "result.json"),
                "estimated_wall_seconds": float(estimated_seconds),
                "estimated_completion_utc": eta.isoformat(),
                "estimated_completion_asia_shanghai": eta.astimezone(
                    timezone(timedelta(hours=8))
                ).isoformat(),
            }
        )
    manifest = {
        "schema": "c2_production_launch_v1",
        "status": "launched_unmonitored",
        "launched_at_utc": launched_at.isoformat(),
        "allocation": "three independent processes on GPUs 0, 1, and 2",
        "episodes_per_run": int(args.episodes),
        "steps_per_episode": int(args.steps),
        "smoke_result": str(args.smoke_result.resolve()),
        "estimated_all_complete_utc": max(
            row["estimated_completion_utc"] for row in runs
        ),
        "estimated_all_complete_asia_shanghai": max(
            row["estimated_completion_asia_shanghai"] for row in runs
        ),
        "version": {
            "head": _git("rev-parse", "HEAD"),
            "dirty": _git("status", "--porcelain").splitlines(),
            "launcher_git_object": _git("hash-object", str(Path(__file__).resolve())),
            "runner_git_object": _git(
                "hash-object", str(ROOT / "scripts" / "run_reward_redesign_arm.py")
            ),
            "trainer_git_object": _git(
                "hash-object", str(ROOT / "scripts" / "train_clean_mainline.py")
            ),
            "reward_ledger_git_object": _git(
                "hash-object", str(ROOT / "environment" / "reward_redesign.py")
            ),
        },
        "runs": runs,
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
