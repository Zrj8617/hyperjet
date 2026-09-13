from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_clean_mainline import build_arg_parser, run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure exact forecast cost in real PPO training.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.30)
    return parser


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.rstrip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=False)
    train_args = build_arg_parser().parse_args(
        [
            "--episodes", "20",
            "--max-steps-per-episode", "500",
            "--rollout-horizon", "128",
            "--max-updates", str(int(args.updates)),
            "--seed", str(int(args.seed)),
            "--device", str(args.device),
            "--task-encoder", "mlp",
            "--output-dir", str(args.output_root / "train"),
            "--run-name", "20260906_forecast_training_timing_seed0",
            "--reward-redesign-arm", "B2",
            "--no-dag-progress-potential-shaping",
        ]
    )
    started_at = datetime.now(timezone.utc)
    wall_started = perf_counter()
    training = run_training(train_args)
    total_wall = float(perf_counter() - wall_started)
    train_dir = Path(training["run_dir"])
    rows = _read_jsonl(train_dir / "train_metrics.jsonl")
    update_rows = [row for row in rows if "ppo_update_step" in row]
    if len(update_rows) != int(args.updates):
        raise AssertionError(
            f"expected {int(args.updates)} logged updates, found {len(update_rows)}"
        )
    forecast_wall = float(sum(float(row["ppo_forecast_wall_seconds"]) for row in update_rows))
    collection_wall = float(sum(float(row["ppo_collection_wall_seconds"]) for row in update_rows))
    ratio = forecast_wall / max(total_wall, 1e-12)
    result = {
        "schema": "forecast_training_timing_gate_v2",
        "status": "pass" if ratio <= float(args.threshold) else "fail",
        "threshold": float(args.threshold),
        "arm": "B2",
        "seed": int(args.seed),
        "update_count": len(update_rows),
        "rollout_horizon": 128,
        "forecast_wall_seconds": forecast_wall,
        "collection_wall_seconds": collection_wall,
        "training_total_wall_seconds": total_wall,
        "forecast_over_collection": forecast_wall / max(collection_wall, 1e-12),
        "forecast_over_training_total": ratio,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "train": training,
        "actual_parameters": vars(train_args),
        "version": {
            "head": _git("rev-parse", "HEAD"),
            "dirty": _git("status", "--porcelain").splitlines(),
            "script_git_object": _git("hash-object", str(Path(__file__).resolve())),
            "trainer_git_object": _git("hash-object", str(ROOT / "scripts" / "train_clean_mainline.py")),
            "forecast_git_object": _git("hash-object", str(ROOT / "environment" / "forecast.py")),
        },
    }
    result_path = args.output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(result_path), "ratio": ratio}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
