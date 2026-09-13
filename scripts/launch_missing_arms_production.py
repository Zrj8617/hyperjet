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
ARMS = ("A1", "C1", "N0-local")
SEEDS = (0, 1, 2)
GPU_ASSIGNMENT = (0, 1, 2, 3, 4, 5, 6, 1, 2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the nine missing-arm production runs once.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
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
    probe = _read_json(args.probe_manifest.resolve())
    seconds_per_episode = float(probe["conservative_training_seconds_per_episode"])
    evaluation_seconds = float(probe["conservative_evaluation_seconds"])
    targets = [
        (arm, seed, GPU_ASSIGNMENT[index])
        for index, (arm, seed) in enumerate(
            (arm, seed) for arm in ARMS for seed in SEEDS
        )
    ]
    launched_at = datetime.now(timezone.utc)
    runs: list[dict[str, Any]] = []
    for arm, seed, gpu in targets:
        run_name = f"20260907_{arm}_seed{seed}"
        run_root = output_root / run_name
        log_path = output_root / f"{run_name}.log"
        if run_root.exists() or log_path.exists():
            raise FileExistsError(f"formal target already exists: {run_name}")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_reward_redesign_arm.py"),
            "--arm", arm.upper(),
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
        estimated_seconds = seconds_per_episode * float(args.episodes) + evaluation_seconds
        eta = launched_at + timedelta(seconds=estimated_seconds)
        runs.append(
            {
                "run_name": run_name,
                "arm": arm,
                "seed": seed,
                "gpu": gpu,
                "pid": int(process.pid),
                "log_path": str(log_path),
                "result_path": str(run_root / "result.json"),
                "estimated_wall_seconds": estimated_seconds,
                "estimated_completion_utc": eta.isoformat(),
                "estimated_completion_asia_shanghai": eta.astimezone(
                    timezone(timedelta(hours=8))
                ).isoformat(),
            }
        )
    latest_eta = max(datetime.fromisoformat(row["estimated_completion_utc"]) for row in runs)
    manifest = {
        "schema": "missing_arms_production_launch_v1",
        "status": "launched_unmonitored",
        "launched_at_utc": launched_at.isoformat(),
        "allocation": "nine processes across seven GPUs; GPUs 1 and 2 host two processes",
        "episodes_per_run": int(args.episodes),
        "steps_per_episode": int(args.steps),
        "run_count": len(runs),
        "probe": probe,
        "estimated_all_complete_utc": latest_eta.isoformat(),
        "estimated_all_complete_asia_shanghai": latest_eta.astimezone(
            timezone(timedelta(hours=8))
        ).isoformat(),
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
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
