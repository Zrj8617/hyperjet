from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B2", "C2A", "C2B", "C2")
ENCODERS = ("mlp", "typed_gated_hgnn")
SEEDS = (5, 86, 617)
ENCODER_HIDDEN_DIMS = {"mlp": 773, "typed_gated_hgnn": 128}


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one GPU is required")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the frozen 2026-09-18 EQ10 four-arm/two-encoder matrix."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", type=_csv_ints, required=True)
    parser.add_argument("--run-prefix", default="20260918")
    return parser


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.rstrip()


def _targets(output_root: Path, gpus: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for arm in ARMS:
        for encoder in ENCODERS:
            for seed in SEEDS:
                run_name = f"20260918_{arm}_{encoder.upper()}_EQ10_seed{seed}"
                rows.append(
                    {
                        "arm": arm,
                        "encoder": encoder,
                        "seed": seed,
                        "encoder_hidden_dim": ENCODER_HIDDEN_DIMS[encoder],
                        "gpu": gpus[index % len(gpus)],
                        "run_name": run_name,
                        "run_root": output_root / run_name,
                        "log_path": output_root / f"{run_name}.log",
                    }
                )
                index += 1
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.run_prefix) != "20260918":
        raise ValueError("--run-prefix must be 20260918")
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"existing result root required: {output_root}")
    if args.manifest.exists():
        raise FileExistsError(f"manifest already exists: {args.manifest}")

    targets = _targets(output_root, tuple(args.gpus))
    for target in targets:
        if target["run_root"].exists() or target["log_path"].exists():
            raise FileExistsError(f"refusing to overwrite {target['run_name']}")

    runs: list[dict[str, Any]] = []
    for target in targets:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_reward_redesign_arm.py"),
            "--arm",
            str(target["arm"]),
            "--seed",
            str(target["seed"]),
            "--episodes",
            "500",
            "--steps",
            "500",
            "--rollout-horizon",
            "125",
            "--device",
            "cuda:0",
            "--task-encoder",
            str(target["encoder"]),
            "--hidden-dim",
            "128",
            "--task-encoder-hidden-dim",
            str(target["encoder_hidden_dim"]),
            "--task-embedding-dim",
            "64",
            "--reward-energy-lambda",
            "1.0",
            "--output-root",
            str(target["run_root"]),
            "--run-name",
            str(target["run_name"]),
            "--skip-eval",
        ]
        if target["encoder"] == "typed_gated_hgnn":
            command.extend(
                [
                    "--rng-neutral-task-encoder-comparison",
                    "--rng-neutral-reference-encoder-hidden-dim",
                    str(ENCODER_HIDDEN_DIMS["mlp"]),
                ]
            )
        if target["arm"] in {"C2A", "C2B", "C2"}:
            command.extend(["--teacher-anneal-total-updates", "2000"])
        child_env = dict(os.environ)
        child_env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(target["gpu"]),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        with target["log_path"].open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        runs.append(
            {
                **{
                    key: value
                    for key, value in target.items()
                    if key not in {"run_root", "log_path"}
                },
                "run_root": str(target["run_root"]),
                "log_path": str(target["log_path"]),
                "result_path": str(target["run_root"] / "result.json"),
                "pid": int(process.pid),
                "command": command,
            }
        )

    manifest = {
        "schema": "eq10_four_arm_two_encoder_launch_v1",
        "status": "launched_unmonitored",
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(runs),
        "arms": list(ARMS),
        "encoders": list(ENCODERS),
        "seeds": list(SEEDS),
        "energy_lambda_task": 1.0,
        "energy_lambda_move": 1.0,
        "flowtime_ref_seconds": 500.0,
        "shared_hidden_dim": 128,
        "task_embedding_dim": 64,
        "encoder_hidden_dims": dict(ENCODER_HIDDEN_DIMS),
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
