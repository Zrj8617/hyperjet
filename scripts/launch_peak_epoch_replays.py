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
AUDIT_RUNS = {
    "C1": "20260907_C1_seed{seed}",
    "B2": "20260906_B2_seed{seed}",
}
ALLOCATIONS = (("C1", 0, 0), ("C1", 1, 1), ("C1", 2, 2), ("B2", 0, 3), ("B2", 1, 5), ("B2", 2, 6))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the six approved peak-epoch diagnostic replays.")
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    smoke = json.loads(args.smoke_result.read_text(encoding="utf-8"))
    if smoke.get("status") != "completed" or not bool(
        smoke.get("actual_parameters", {}).get("rollout_cost_diagnostics", False)
    ):
        raise ValueError("completed rollout-cost diagnostic smoke required")
    seconds_per_update = float(smoke["wall_seconds"])
    launched_at = datetime.now(timezone.utc)
    runs: list[dict[str, Any]] = []
    for arm, seed, gpu in ALLOCATIONS:
        source_result_path = args.audit_root / AUDIT_RUNS[arm].format(seed=seed) / "result.json"
        source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
        source_train_dir = Path(source_result["train_dir"])
        source_metrics = source_train_dir / "train_metrics.jsonl"
        if not source_metrics.is_file():
            raise FileNotFoundError(source_metrics)
        run_name = f"20260908_{arm}_peak_epoch_diag_replay_seed{seed}"
        run_root = args.audit_root / run_name
        log_path = args.audit_root / f"{run_name}.log"
        if run_root.exists() or log_path.exists():
            raise FileExistsError(run_name)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_reward_redesign_arm.py"),
            "--arm", arm,
            "--seed", str(seed),
            "--episodes", "250",
            "--steps", "500",
            "--max-updates", "1000",
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
        eta = launched_at + timedelta(seconds=seconds_per_update * 1000.0)
        runs.append(
            {
                "arm": arm,
                "seed": seed,
                "gpu": gpu,
                "pid": int(process.pid),
                "log_path": str(log_path),
                "result_path": str(run_root / "result.json"),
                "source_result": str(source_result_path),
                "source_train_dir": str(source_train_dir),
                "estimated_completion_utc_conservative": eta.isoformat(),
            }
        )
    manifest = {
        "schema": "peak_epoch_diagnostic_replay_launch_v1",
        "status": "launched_unmonitored",
        "launched_at_utc": launched_at.isoformat(),
        "max_updates": 1000,
        "episodes_cap": 250,
        "smoke_result": str(args.smoke_result),
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
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.rstrip()


if __name__ == "__main__":
    main()
