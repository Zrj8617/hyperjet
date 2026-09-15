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
APPROVED_ARMS = ("B1", "B2", "C2A", "C2B", "C2", "C1")
APPROVED_SEEDS = (5, 86, 617)


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(token.strip().upper() for token in value.split(",") if token.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be non-empty and unique")
    return values


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be non-empty and unique")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the approved 2026-09-15 six-arm production matrix."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arms", type=_csv_strings, required=True)
    parser.add_argument("--seeds", type=_csv_ints, required=True)
    parser.add_argument("--gpus", type=_csv_ints, required=True)
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--rollout-horizon", type=int, default=125)
    return parser


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.rstrip()


def _targets(
    *,
    output_root: Path,
    run_prefix: str,
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
    gpus: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for arm in arms:
        for seed in seeds:
            run_name = f"{run_prefix}_{arm}_seed{seed}"
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
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
    arms = tuple(args.arms)
    seeds = tuple(args.seeds)
    if arms != APPROVED_ARMS:
        raise ValueError(f"--arms must be {','.join(APPROVED_ARMS)} in this order")
    if seeds != APPROVED_SEEDS:
        raise ValueError(f"--seeds must be {','.join(map(str, APPROVED_SEEDS))}")
    if str(args.run_prefix) != "20260915":
        raise ValueError("--run-prefix must be 20260915")
    if int(args.episodes) != 500 or int(args.steps) != 500:
        raise ValueError("production runs require exactly 500 episodes x 500 steps")
    if int(args.rollout_horizon) != 125:
        raise ValueError("production runs require --rollout-horizon 125")
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"existing result root required: {output_root}")
    if args.manifest.exists():
        raise FileExistsError(f"manifest already exists: {args.manifest}")
    targets = _targets(
        output_root=output_root,
        run_prefix=str(args.run_prefix),
        arms=arms,
        seeds=seeds,
        gpus=tuple(args.gpus),
    )
    for target in targets:
        if target["run_root"].exists() or target["log_path"].exists():
            raise FileExistsError(f"production target already exists: {target['run_name']}")

    runs: list[dict[str, Any]] = []
    launched_at = datetime.now(timezone.utc)
    for target in targets:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_reward_redesign_arm.py"),
            "--arm",
            str(target["arm"]),
            "--seed",
            str(target["seed"]),
            "--episodes",
            str(args.episodes),
            "--steps",
            str(args.steps),
            "--rollout-horizon",
            str(args.rollout_horizon),
            "--device",
            "cuda:0",
            "--output-root",
            str(target["run_root"]),
            "--run-name",
            str(target["run_name"]),
        ]
        child_env = dict(os.environ)
        child_env["CUDA_VISIBLE_DEVICES"] = str(target["gpu"])
        with target["log_path"].open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        runs.append(
            {
                **{key: value for key, value in target.items() if key not in {"run_root", "log_path"}},
                "run_root": str(target["run_root"]),
                "log_path": str(target["log_path"]),
                "result_path": str(target["run_root"] / "result.json"),
                "pid": int(process.pid),
                "command": command,
            }
        )

    manifest = {
        "schema": "six_arm_realign_production_launch_v1",
        "status": "launched_unmonitored",
        "launched_at_utc": launched_at.isoformat(),
        "run_count": len(runs),
        "arms_order": list(arms),
        "seeds": list(seeds),
        "episodes_per_run": int(args.episodes),
        "steps_per_episode": int(args.steps),
        "rollout_horizon": int(args.rollout_horizon),
        "task_encoder": "mlp",
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
