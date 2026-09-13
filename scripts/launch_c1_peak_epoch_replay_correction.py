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
RUN_PREFIX = "20260909_C1_peak_epoch_diag_replay_fixed_seed"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch corrected C1 peak-epoch replays with the original teacher clock."
    )
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seconds-per-update", type=float, required=True)
    args = parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError(args.manifest)

    launched_at = datetime.now(timezone.utc)
    runs: list[dict[str, Any]] = []
    for seed, gpu in enumerate((0, 3, 6)):
        source_result_path = args.audit_root / f"20260907_C1_seed{seed}" / "result.json"
        source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
        source_train_dir = Path(source_result["train_dir"])
        if not (source_train_dir / "train_metrics.jsonl").is_file():
            raise FileNotFoundError(source_train_dir / "train_metrics.jsonl")

        run_name = f"{RUN_PREFIX}{seed}"
        run_root = args.audit_root / run_name
        log_path = args.audit_root / f"{run_name}.log"
        if run_root.exists() or log_path.exists():
            raise FileExistsError(run_name)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_reward_redesign_arm.py"),
            "--arm", "C1",
            "--seed", str(seed),
            "--episodes", "250",
            "--steps", "500",
            "--max-updates", "1000",
            "--teacher-anneal-total-updates", "2000",
            "--skip-eval",
            "--rollout-cost-diagnostics",
            "--device", "cuda:0",
            "--output-root", str(run_root),
            "--run-name", run_name,
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        handle = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            handle.close()
        runs.append(
            {
                "arm": "C1",
                "seed": seed,
                "gpu": gpu,
                "pid": int(process.pid),
                "log_path": str(log_path),
                "result_path": str(run_root / "result.json"),
                "source_result": str(source_result_path),
                "source_train_dir": str(source_train_dir),
                "max_updates": 1000,
                "teacher_anneal_total_updates": 2000,
                "estimated_completion_utc": (
                    launched_at + timedelta(seconds=float(args.seconds_per_update) * 1000.0)
                ).isoformat(),
            }
        )

    manifest = {
        "schema": "c1_peak_epoch_diagnostic_replay_correction_launch_v1",
        "status": "launched_unmonitored",
        "reason": "replace deleted C1 replays whose teacher schedule annealed after update 500",
        "launched_at_utc": launched_at.isoformat(),
        "runs": runs,
        "version": {
            "head": _git("rev-parse", "HEAD"),
            "dirty": _git("status", "--porcelain").splitlines(),
            "launcher_git_object": _git("hash-object", str(Path(__file__).resolve())),
            "runner_git_object": _git("hash-object", str(ROOT / "scripts" / "run_reward_redesign_arm.py")),
            "trainer_git_object": _git("hash-object", str(ROOT / "scripts" / "train_clean_mainline.py")),
            "metrics_git_object": _git("hash-object", str(ROOT / "environment" / "metrics.py")),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
