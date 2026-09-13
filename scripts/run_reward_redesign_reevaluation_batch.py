from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OLD_CONFIGURATIONS = ("N0", "A2", "B1", "B2", "B2D", "D2")
SEEDS = (0, 1, 2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reevaluate the 18 existing distinct checkpoints.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(7)))
    return parser


def _checkpoint(output_root: Path, configuration: str, seed: int) -> Path:
    candidates = sorted(
        (output_root / f"20260906_{configuration}_seed{seed}" / "train").glob(
            "*/checkpoints/latest.pt"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one checkpoint for {configuration} seed {seed}, found {len(candidates)}"
        )
    return candidates[0].resolve()


def _write(path: Path, payload: dict[str, Any], lock: Lock) -> None:
    with lock:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"existing result root required: {output_root}")
    jobs = [
        {
            "configuration": configuration,
            "model_seed": seed,
            "checkpoint": str(_checkpoint(output_root, configuration, seed)),
            "gpu": int(args.gpus[index % len(args.gpus)]),
            "run_name": f"20260907_reeval_{configuration}_seed{seed}",
        }
        for index, (configuration, seed) in enumerate(
            (configuration, seed)
            for configuration in OLD_CONFIGURATIONS
            for seed in SEEDS
        )
    ]
    for job in jobs:
        run_root = output_root / str(job["run_name"])
        log_path = output_root / f"{job['run_name']}.log"
        if run_root.exists() or log_path.exists():
            raise FileExistsError(str(run_root))
        job["result_path"] = str(run_root / "result.json")
        job["log_path"] = str(log_path)
        job["status"] = "queued"
    lock = Lock()
    manifest: dict[str, Any] = {
        "schema": "reward_redesign_reevaluation_batch_v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "20 seeds (0..19), forced hover, deterministic masked argmax, training RNG prelude",
        "jobs": jobs,
    }
    _write(args.manifest, manifest, lock)

    def run(job: dict[str, Any]) -> dict[str, Any]:
        job["status"] = "running"
        _write(args.manifest, manifest, lock)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "reevaluate_reward_redesign_checkpoint.py"),
            "--checkpoint", str(job["checkpoint"]),
            "--configuration", str(job["configuration"]),
            "--model-seed", str(job["model_seed"]),
            "--device", "cuda:0",
            "--output-root", str(output_root / str(job["run_name"])),
            "--run-name", str(job["run_name"]),
        ]
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        with Path(str(job["log_path"])).open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        job["status"] = "completed" if completed.returncode == 0 else "failed"
        job["returncode"] = int(completed.returncode)
        _write(args.manifest, manifest, lock)
        return job

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
    manifest["status"] = (
        "completed" if all(job["status"] == "completed" for job in jobs) else "failed"
    )
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write(args.manifest, manifest, lock)
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
