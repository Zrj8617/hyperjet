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

from scripts.eval_clean_mainline import build_arg_parser as build_eval_parser, run_evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen 20-seed checkpoint reevaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    return parser


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.rstrip()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    started_wall = perf_counter()
    eval_args = build_eval_parser().parse_args(
        [
            "--checkpoint", str(args.checkpoint.resolve()),
            "--episodes", "20",
            "--arrival-steps", "500",
            "--max-drain-steps", "0",
            "--seed", "0",
            "--device", str(args.device),
            "--output-dir", str(output_root / "eval"),
            "--run-name", f"{args.run_name}_eval20",
            "--offloading-policy", "actor_argmax",
            "--freeze-movement",
            "--clean-training-rng-prelude",
            "--no-render",
        ]
    )
    evaluation = run_evaluation(eval_args)
    result: dict[str, Any] = {
        "schema": "reward_redesign_checkpoint_reevaluation_v1",
        "status": "completed",
        "configuration": str(args.configuration),
        "model_seed": int(args.model_seed),
        "checkpoint": str(args.checkpoint.resolve()),
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": float(perf_counter() - started_wall),
        "evaluation": evaluation,
        "protocol": {
            "environment_seeds": list(range(20)),
            "movement": "forced_hover",
            "offloading": "deterministic_masked_argmax",
            "arrival_steps": 500,
            "max_drain_steps": 0,
            "clean_training_rng_prelude": True,
        },
        "version": {
            "head": _git("rev-parse", "HEAD"),
            "dirty": _git("status", "--porcelain").splitlines(),
            "script_git_object": _git("hash-object", str(Path(__file__).resolve())),
            "evaluator_git_object": _git(
                "hash-object", str(ROOT / "scripts" / "eval_clean_mainline.py")
            ),
        },
    }
    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "result": str(output_root / "result.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
