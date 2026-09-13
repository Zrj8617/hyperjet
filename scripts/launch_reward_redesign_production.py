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
ARM_GPU = {
    "B1": 0,
    "D2": 1,
    "N0": 2,
    "A2": 3,
    "B2": 4,
    "B2D": 5,
    "D1": 6,
}
SEEDS = (0, 1, 2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the 21 reward-redesign production runs once.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--probe-prefix", default="20260906_densityprobe")
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

    probe_wall_by_arm: dict[str, list[float]] = {}
    for arm in ARM_GPU:
        values = []
        for seed in SEEDS:
            probe = output_root / f"{args.probe_prefix}_{arm}_seed{seed}" / "result.json"
            payload = _read_json(probe)
            if payload.get("status") != "completed":
                raise RuntimeError(f"time probe did not complete: {probe}")
            values.append(float(payload["wall_seconds"]))
        probe_wall_by_arm[arm] = values

    targets = []
    for arm, gpu in ARM_GPU.items():
        for seed in SEEDS:
            run_name = f"20260906_{arm}_seed{seed}"
            run_root = output_root / run_name
            log_path = output_root / f"{run_name}.log"
            if run_root.exists() or log_path.exists():
                raise FileExistsError(f"formal target already exists: {run_name}")
            targets.append((arm, gpu, seed, run_name, run_root, log_path))

    launched_at = datetime.now(timezone.utc)
    runs: list[dict[str, Any]] = []
    for arm, gpu, seed, run_name, run_root, log_path in targets:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_reward_redesign_arm.py"),
            "--arm", arm,
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
        seconds_per_episode = max(probe_wall_by_arm[arm]) / 5.0
        estimated_seconds = seconds_per_episode * float(args.episodes)
        eta_utc = launched_at + timedelta(seconds=estimated_seconds)
        runs.append(
            {
                "run_name": run_name,
                "arm": arm,
                "seed": seed,
                "gpu": gpu,
                "pid": int(process.pid),
                "log_path": str(log_path),
                "result_path": str(run_root / "result.json"),
                "run_root": str(run_root),
                "probe_seconds_per_episode_conservative": seconds_per_episode,
                "estimated_wall_seconds": estimated_seconds,
                "estimated_completion_utc": eta_utc.isoformat(),
                "estimated_completion_asia_shanghai": eta_utc.astimezone(
                    timezone(timedelta(hours=8))
                ).isoformat(),
            }
        )

    latest_eta = max(
        datetime.fromisoformat(row["estimated_completion_utc"]) for row in runs
    )
    manifest = {
        "schema": "reward_redesign_production_launch_v1",
        "status": "launched_unmonitored",
        "launched_at_utc": launched_at.isoformat(),
        "allocation": "three processes per GPU; one arm per GPU",
        "episodes_per_run": int(args.episodes),
        "steps_per_episode": int(args.steps),
        "run_count": len(runs),
        "memory_probe_mib_per_process": 610,
        "memory_snapshot_free_mib": {
            "0": 17188,
            "1": 19922,
            "2": 24199,
            "3": 22188,
            "4": 24199,
            "5": 24199,
            "6": 23582,
        },
        "time_probe_wall_seconds": probe_wall_by_arm,
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
            "forecast_git_object": _git(
                "hash-object", str(ROOT / "environment" / "forecast.py")
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
