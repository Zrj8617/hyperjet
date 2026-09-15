from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
APPROVED_ARMS = ("B1", "B2", "C2A", "C2B", "C2", "C1")
APPROVED_SEEDS = (5, 86, 617)
CANDIDATE_EPISODES = (320, 360, 400, 450, 500)
TEST_PROTOCOLS = ("joint", "forced_hover")
VALIDATION_PROTOCOL = "joint"


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
        description="Run six-arm fixed-tape joint validation and locked-checkpoint tests."
    )
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--tape-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=28)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arms", type=_csv_strings, required=True)
    parser.add_argument("--seeds", type=_csv_ints, required=True)
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--gpus", type=_csv_ints, required=True)
    return parser


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
    if args.output_root.exists() or args.manifest.exists():
        raise FileExistsError("fair-evaluation output or manifest already exists")
    args.output_root.mkdir(parents=True)
    state: dict[str, Any] = {
        "schema": "six_arm_fair_eval_orchestrator_v2",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_root": str(args.audit_root.resolve()),
        "tape_dir": str(args.tape_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "run_prefix": str(args.run_prefix),
        "arms_order": list(arms),
        "seeds": list(seeds),
        "validation_tape_ids": list(range(100, 120)),
        "test_tape_ids": list(range(200, 250)),
        "candidate_episodes": list(CANDIDATE_EPISODES),
        "selection_protocol": VALIDATION_PROTOCOL,
        "test_protocols": list(TEST_PROTOCOLS),
        "max_workers": int(args.max_workers),
        "gpus": list(args.gpus),
        "version": _version_record(),
        "arms": _audit_runs(
            args.audit_root,
            arms,
            seeds,
            run_prefix=str(args.run_prefix),
        ),
    }
    _write(args.manifest, state)
    _run_all(args=args, state=state, arms=arms, seeds=seeds)
    state["status"] = "completed"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write(args.manifest, state)
    return 0


def _run_all(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
) -> None:
    validation_jobs: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            run = state["arms"][arm][str(seed)]
            for episode in CANDIDATE_EPISODES:
                checkpoint = (
                    Path(run["train_dir"])
                    / "checkpoints"
                    / f"checkpoint_ep_{episode:04d}.pt"
                )
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                validation_jobs.append(
                    _job(
                        args=args,
                        arm=arm,
                        seed=seed,
                        checkpoint=checkpoint,
                        checkpoint_label=f"ep{episode:04d}",
                        split="validation",
                        tape_ids="100-119",
                        protocol=VALIDATION_PROTOCOL,
                    )
                )
    _run_jobs(
        validation_jobs,
        max_workers=int(args.max_workers),
        gpus=tuple(args.gpus),
    )

    selections: dict[str, Any] = {}
    for arm in arms:
        selections[arm] = {}
        for seed in seeds:
            candidates = []
            for episode in CANDIDATE_EPISODES:
                path = _output_path(
                    args,
                    arm,
                    seed,
                    f"ep{episode:04d}",
                    "validation",
                    VALIDATION_PROTOCOL,
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                score = statistics.fmean(
                    float(row["J_per_offer"]) for row in payload["rows"]
                )
                candidates.append((score, episode, str(payload["checkpoint"]), str(path)))
            score, episode, checkpoint, source = min(
                candidates, key=lambda item: (item[0], item[1])
            )
            selections[arm][str(seed)] = {
                "selected_episode": int(episode),
                "selected_checkpoint": checkpoint,
                "joint_validation_J_per_offer_mean": float(score),
                "selection_source": source,
                "selection_protocol": VALIDATION_PROTOCOL,
                "tie_break": "earlier_checkpoint",
                "candidate_sources": [item[3] for item in candidates],
            }
    state["checkpoint_selections"] = selections
    state["status"] = "selection_locked"
    _write(args.manifest, state)
    _write(args.output_root / "selection_six_arm.json", selections)

    test_jobs: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            selected = selections[arm][str(seed)]
            final_checkpoint = Path(state["arms"][arm][str(seed)]["checkpoint"])
            for label, checkpoint in (
                ("budget_selected", Path(selected["selected_checkpoint"])),
                ("final_ep0500", final_checkpoint),
            ):
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                for protocol in TEST_PROTOCOLS:
                    test_jobs.append(
                        _job(
                            args=args,
                            arm=arm,
                            seed=seed,
                            checkpoint=checkpoint,
                            checkpoint_label=label,
                            split="test",
                            tape_ids="200-249",
                            protocol=protocol,
                        )
                    )
    _run_jobs(
        test_jobs,
        max_workers=int(args.max_workers),
        gpus=tuple(args.gpus),
    )


def _audit_runs(
    audit_root: Path,
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
    *,
    run_prefix: str,
) -> dict[str, Any]:
    audited: dict[str, Any] = {}
    for arm in arms:
        audited[arm] = {}
        for seed in seeds:
            result_path = audit_root / f"{run_prefix}_{arm}_seed{seed}" / "result.json"
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed":
                raise ValueError(f"incomplete source run: {result_path}")
            actual = dict(result["actual_parameters"])
            if str(actual["reward_redesign_arm"]) != arm or int(actual["seed"]) != seed:
                raise ValueError(f"source run identity mismatch: {result_path}")
            train_dir = Path(result["train_dir"])
            rows = [
                json.loads(line)
                for line in (train_dir / "train_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            teacher_weights: dict[str, float] = {}
            checkpoints: dict[str, str] = {}
            for episode in CANDIDATE_EPISODES:
                eligible = [row for row in rows if int(row["episode"]) == episode - 1]
                if not eligible:
                    raise ValueError(
                        f"missing episode boundary for {arm} seed {seed} episode {episode}"
                    )
                teacher_weights[str(episode)] = float(
                    eligible[-1].get("ppo_diagnostics", {}).get("teacher_weight", 0.0)
                )
                checkpoint = train_dir / "checkpoints" / f"checkpoint_ep_{episode:04d}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                checkpoints[str(episode)] = str(checkpoint)
            audited[arm][str(seed)] = {
                "result_path": str(result_path),
                "train_dir": str(train_dir),
                "checkpoint": str(result["checkpoint"]),
                "actual_parameters": actual,
                "resolved_flags": result.get("resolved_flags"),
                "candidate_teacher_weights": teacher_weights,
                "candidate_checkpoints_read": checkpoints,
                "source_version": result.get("version"),
            }
    return audited


def _job(
    *,
    args: argparse.Namespace,
    arm: str,
    seed: int,
    checkpoint: Path,
    checkpoint_label: str,
    split: str,
    tape_ids: str,
    protocol: str,
) -> dict[str, Any]:
    output = _output_path(args, arm, seed, checkpoint_label, split, protocol)
    return {
        "arm": arm,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_label": checkpoint_label,
        "split": split,
        "tape_ids": tape_ids,
        "protocol": protocol,
        "tape_dir": str(args.tape_dir),
        "output": str(output),
        "log": str(output.with_suffix(".log")),
    }


def _output_path(
    args: argparse.Namespace,
    arm: str,
    seed: int,
    label: str,
    split: str,
    protocol: str,
) -> Path:
    return args.output_root / split / f"{arm}_seed{seed}_{label}_{protocol}.json"


def _run_jobs(
    jobs: list[dict[str, Any]], *, max_workers: int, gpus: tuple[int, ...]
) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one, job, index, gpus): job
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            future.result()


def _run_one(job: dict[str, Any], index: int, gpus: tuple[int, ...]) -> None:
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        raise FileExistsError(f"refusing to reuse evaluation output: {output}")
    gpu = gpus[index % len(gpus)]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_fair_eval_batch.py"),
        "--checkpoint",
        job["checkpoint"],
        "--tape-dir",
        job["tape_dir"],
        "--tape-ids",
        job["tape_ids"],
        "--protocol",
        job["protocol"],
        "--device",
        "cuda:0",
        "--output",
        job["output"],
        "--arm",
        job["arm"],
        "--model-seed",
        str(job["seed"]),
        "--checkpoint-label",
        job["checkpoint_label"],
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    with Path(job["log"]).open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _version_record() -> dict[str, Any]:
    return {
        "head": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain").splitlines(),
        "orchestrator_git_object": _git("hash-object", str(Path(__file__).resolve())),
        "evaluator_git_object": _git(
            "hash-object", str(ROOT / "scripts" / "run_fair_eval_batch.py")
        ),
        "tape_module_git_object": _git(
            "hash-object", str(ROOT / "environment" / "exogenous_tape.py")
        ),
        "env_git_object": _git("hash-object", str(ROOT / "environment" / "env.py")),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.rstrip()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
