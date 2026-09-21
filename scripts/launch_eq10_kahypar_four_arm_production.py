from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B2", "C2A", "C2B", "C2")
SEEDS = (5, 86, 617)
ENCODER = "typed_gated_hgnn"
ENCODER_HIDDEN_DIM = 128
REFERENCE_MLP_HIDDEN_DIM = 773
RUN_PREFIX = "20260921"


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    expected = len(ARMS) * len(SEEDS)
    if len(values) != expected:
        raise argparse.ArgumentTypeError(
            f"exactly {expected} GPU assignments are required"
        )
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the frozen 2026-09-21 EQ10 KaHyPar-on four-arm matrix."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-audit-root", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-assignment", type=_csv_ints, required=True)
    parser.add_argument(
        "--launch-stagger-seconds",
        type=float,
        default=60.0,
        help=(
            "Delay between process starts so spawned KaHyPar workers do not "
            "cold-start against storage simultaneously."
        ),
    )
    return parser


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.rstrip()


def _name(arm: str, seed: int) -> str:
    return (
        f"{RUN_PREFIX}_{arm}_TYPED_GATED_HGNN_KAHYPAR_EQ10_seed{seed}"
    )


def _baseline_path(root: Path, arm: str, seed: int) -> Path:
    name = f"20260918_{arm}_TYPED_GATED_HGNN_EQ10_seed{seed}"
    return root / name / "result.json"


def _targets(
    output_root: Path,
    baseline_root: Path,
    gpu_assignment: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    index = 0
    for arm in ARMS:
        for seed in SEEDS:
            name = _name(arm, seed)
            baseline_path = _baseline_path(baseline_root, arm, seed)
            if not baseline_path.is_file():
                raise FileNotFoundError(baseline_path)
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            if baseline.get("status") != "completed":
                raise ValueError(f"incomplete baseline: {baseline_path}")
            actual = baseline["actual_parameters"]
            if (
                str(actual.get("reward_redesign_arm")) != arm
                or int(actual.get("seed")) != seed
                or str(actual.get("task_encoder")) != ENCODER
                or bool(actual.get("enable_kahypar"))
                or float(actual.get("reward_energy_lambda", -1.0)) != 1.0
            ):
                raise ValueError(f"baseline identity/config mismatch: {baseline_path}")
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "gpu": gpu_assignment[index],
                    "run_name": name,
                    "run_root": output_root / name,
                    "log_path": output_root / f"{name}.log",
                    "baseline_result_path": baseline_path,
                    "baseline_version": baseline.get("version"),
                    "baseline_parameter_counts": baseline.get("parameter_counts"),
                }
            )
            index += 1
    return rows


def _command(target: dict[str, Any]) -> list[str]:
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
        ENCODER,
        "--hidden-dim",
        "128",
        "--task-encoder-hidden-dim",
        str(ENCODER_HIDDEN_DIM),
        "--task-embedding-dim",
        "64",
        "--reward-energy-lambda",
        "1.0",
        "--enable-kahypar",
        "--rng-neutral-task-encoder-comparison",
        "--rng-neutral-reference-encoder-hidden-dim",
        str(REFERENCE_MLP_HIDDEN_DIM),
        "--output-root",
        str(target["run_root"]),
        "--run-name",
        str(target["run_name"]),
        "--skip-eval",
    ]
    if target["arm"] in {"C2A", "C2B", "C2"}:
        command.extend(["--teacher-anneal-total-updates", "2000"])
    return command


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    baseline_root = args.baseline_audit_root.resolve()
    if not output_root.is_dir() or not baseline_root.is_dir():
        raise FileNotFoundError("existing output and baseline audit roots are required")
    if args.manifest.exists():
        raise FileExistsError(f"manifest already exists: {args.manifest}")
    if float(args.launch_stagger_seconds) < 0.0:
        raise ValueError("--launch-stagger-seconds must be non-negative")

    smoke = json.loads(args.smoke_result.read_text(encoding="utf-8"))
    if smoke.get("schema") != "eq10_kahypar_four_arm_smoke_v1" or smoke.get(
        "status"
    ) != "pass":
        raise ValueError("a passing frozen KaHyPar smoke result is required")

    targets = _targets(
        output_root,
        baseline_root,
        tuple(args.gpu_assignment),
    )
    for target in targets:
        if target["run_root"].exists() or target["log_path"].exists():
            raise FileExistsError(f"refusing to overwrite {target['run_name']}")

    runs = []
    for index, target in enumerate(targets):
        command = _command(target)
        environment = dict(os.environ)
        environment.update(
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
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        runs.append(
            {
                "arm": target["arm"],
                "seed": target["seed"],
                "encoder": ENCODER,
                "enable_kahypar": True,
                "gpu": target["gpu"],
                "run_name": target["run_name"],
                "run_root": str(target["run_root"]),
                "log_path": str(target["log_path"]),
                "result_path": str(target["run_root"] / "result.json"),
                "baseline_result_path": str(target["baseline_result_path"]),
                "baseline_version": target["baseline_version"],
                "baseline_parameter_counts": target["baseline_parameter_counts"],
                "pid": int(process.pid),
                "command": command,
            }
        )
        if index + 1 < len(targets) and float(args.launch_stagger_seconds) > 0.0:
            time.sleep(float(args.launch_stagger_seconds))

    manifest = {
        "schema": "eq10_kahypar_four_arm_launch_v1",
        "status": "launched_unmonitored",
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(runs),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "task_encoder": ENCODER,
        "enable_kahypar": True,
        "energy_lambda_task": 1.0,
        "energy_lambda_move": 1.0,
        "flowtime_ref_seconds": 500.0,
        "shared_hidden_dim": 128,
        "task_encoder_hidden_dim": ENCODER_HIDDEN_DIM,
        "task_embedding_dim": 64,
        "smoke_result": str(args.smoke_result.resolve()),
        "gpu_assignment": list(args.gpu_assignment),
        "launch_stagger_seconds": float(args.launch_stagger_seconds),
        "version": {
            "head": _git("rev-parse", "HEAD"),
            "dirty": _git("status", "--porcelain").splitlines(),
            "launcher_git_object": _git("hash-object", str(Path(__file__).resolve())),
            "smoke_git_object": _git(
                "hash-object", str(ROOT / "scripts" / "smoke_eq10_kahypar_four_arm.py")
            ),
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
