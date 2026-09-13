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
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_EPISODES = (50, 100, 150, 200, 250)
PROTOCOLS = ("forced_hover", "joint")
GPUS = tuple(range(7))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-tape validation/selection/test for C1, B2, and C2.")
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--tape-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=28)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.manifest.exists():
        raise FileExistsError("fair-evaluation output or manifest already exists")
    args.output_root.mkdir(parents=True)
    state: dict[str, Any] = {
        "schema": "c1_b2_c2_fair_eval_orchestrator_v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_tape_ids": list(range(100, 120)),
        "test_tape_ids": list(range(200, 250)),
        "candidate_episodes": list(CANDIDATE_EPISODES),
        "protocols": list(PROTOCOLS),
        "max_workers": int(args.max_workers),
        "version": _version_record(),
        "arms": {},
    }
    _write(args.manifest, state)

    initial_runs = _audit_runs(args.audit_root, ("C1", "B2"), require_completed=True)
    state["arms"].update(initial_runs)
    _write(args.manifest, state)
    _run_arm_set(args=args, state=state, arms=("C1", "B2"))

    state["status"] = "waiting_for_c2"
    _write(args.manifest, state)
    _wait_for_c2(args.audit_root)
    c2_runs = _audit_runs(args.audit_root, ("C2",), require_completed=True)
    state["arms"].update(c2_runs)
    _write(args.manifest, state)
    _run_arm_set(args=args, state=state, arms=("C2",))

    state["status"] = "completed"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write(args.manifest, state)


def _run_arm_set(*, args: argparse.Namespace, state: dict[str, Any], arms: tuple[str, ...]) -> None:
    validation_jobs: list[dict[str, Any]] = []
    for arm in arms:
        for seed in range(3):
            run = state["arms"][arm][str(seed)]
            for episode in CANDIDATE_EPISODES:
                checkpoint = Path(run["train_dir"]) / "checkpoints" / f"checkpoint_ep_{episode:04d}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                if arm == "C1" and float(run["candidate_teacher_weights"][str(episode)]) != 1.0:
                    continue
                for protocol in PROTOCOLS:
                    validation_jobs.append(
                        _job(
                            args=args,
                            arm=arm,
                            seed=seed,
                            checkpoint=checkpoint,
                            checkpoint_label=f"ep{episode:04d}",
                            split="validation",
                            tape_ids="100-119",
                            protocol=protocol,
                        )
                    )
    _run_jobs(validation_jobs, max_workers=int(args.max_workers))
    selections: dict[str, Any] = {}
    for arm in arms:
        selections[arm] = {}
        for seed in range(3):
            candidates = []
            for episode in CANDIDATE_EPISODES:
                path = _output_path(args, arm, seed, f"ep{episode:04d}", "validation", "forced_hover")
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                score = statistics.fmean(float(row["J_per_offer"]) for row in payload["rows"])
                candidates.append((score, episode, str(payload["checkpoint"]), str(path)))
            if not candidates:
                raise RuntimeError(f"no valid validation candidates for {arm} seed {seed}")
            score, episode, checkpoint, source = min(candidates, key=lambda item: (item[0], item[1]))
            selections[arm][str(seed)] = {
                "selected_episode": int(episode),
                "selected_checkpoint": checkpoint,
                "forced_hover_validation_J_per_offer_mean": float(score),
                "selection_source": source,
                "tie_break": "earlier_checkpoint",
            }
    state.setdefault("checkpoint_selections", {}).update(selections)
    state["status"] = f"{'+'.join(arms)}_selection_locked"
    _write(args.manifest, state)
    selection_path = args.output_root / f"selection_{'_'.join(arms)}.json"
    _write(selection_path, selections)

    test_jobs: list[dict[str, Any]] = []
    for arm in arms:
        for seed in range(3):
            selected = state["checkpoint_selections"][arm][str(seed)]
            final_checkpoint = Path(state["arms"][arm][str(seed)]["checkpoint"])
            comparisons = (
                ("budget_selected", Path(selected["selected_checkpoint"])),
                ("final_ep0500", final_checkpoint),
            )
            for label, checkpoint in comparisons:
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                for protocol in PROTOCOLS:
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
    _run_jobs(test_jobs, max_workers=int(args.max_workers))
    state["status"] = f"{'+'.join(arms)}_completed"
    state.setdefault("completed_arm_sets", []).append(list(arms))
    _write(args.manifest, state)


def _audit_runs(audit_root: Path, arms: tuple[str, ...], *, require_completed: bool) -> dict[str, Any]:
    dates = {"C1": "20260907", "B2": "20260906", "C2": "20260908"}
    audited: dict[str, Any] = {}
    for arm in arms:
        audited[arm] = {}
        for seed in range(3):
            result_path = audit_root / f"{dates[arm]}_{arm}_seed{seed}" / "result.json"
            if not result_path.is_file():
                if require_completed:
                    raise FileNotFoundError(result_path)
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed":
                raise ValueError(f"incomplete source run: {result_path}")
            actual = dict(result["actual_parameters"])
            if str(actual["reward_redesign_arm"]) != arm or int(actual["seed"]) != seed:
                raise ValueError(f"source run identity mismatch: {result_path}")
            train_dir = Path(result["train_dir"])
            metrics_path = train_dir / "train_metrics.jsonl"
            rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            teacher_weights: dict[str, float] = {}
            for episode in CANDIDATE_EPISODES:
                eligible = [row for row in rows if int(row["episode"]) == episode - 1]
                if not eligible:
                    raise ValueError(f"missing episode boundary for {arm} seed {seed} episode {episode}")
                teacher_weights[str(episode)] = float(eligible[-1].get("ppo_diagnostics", {}).get("teacher_weight", 0.0))
            audited[arm][str(seed)] = {
                "result_path": str(result_path),
                "train_dir": str(train_dir),
                "checkpoint": str(result["checkpoint"]),
                "actual_parameters": actual,
                "candidate_teacher_weights": teacher_weights,
                "source_version": result.get("version"),
            }
    return audited


def _wait_for_c2(audit_root: Path) -> None:
    paths = [audit_root / f"20260908_C2_seed{seed}" / "result.json" for seed in range(3)]
    while True:
        if all(path.is_file() for path in paths):
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            if all(payload.get("status") == "completed" for payload in payloads):
                return
        time.sleep(60.0)


def _job(
    *, args: argparse.Namespace, arm: str, seed: int, checkpoint: Path,
    checkpoint_label: str, split: str, tape_ids: str, protocol: str,
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


def _output_path(args: argparse.Namespace, arm: str, seed: int, label: str, split: str, protocol: str) -> Path:
    return args.output_root / split / f"{arm}_seed{seed}_{label}_{protocol}.json"


def _run_jobs(jobs: list[dict[str, Any]], *, max_workers: int) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, job, index): job for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            future.result()


def _run_one(job: dict[str, Any], index: int) -> None:
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            return
        raise ValueError(f"existing incomplete evaluation output: {output}")
    gpu = GPUS[index % len(GPUS)]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_fair_eval_batch.py"),
        "--checkpoint", job["checkpoint"],
        "--tape-dir", job["tape_dir"],
        "--tape-ids", job["tape_ids"],
        "--protocol", job["protocol"],
        "--device", "cuda:0",
        "--output", job["output"],
        "--arm", job["arm"],
        "--model-seed", str(job["seed"]),
        "--checkpoint-label", job["checkpoint_label"],
    ]
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    with Path(job["log"]).open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _version_record() -> dict[str, Any]:
    return {
        "head": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain").splitlines(),
        "orchestrator_git_object": _git("hash-object", str(Path(__file__).resolve())),
        "evaluator_git_object": _git("hash-object", str(ROOT / "scripts" / "run_fair_eval_batch.py")),
        "tape_module_git_object": _git("hash-object", str(ROOT / "environment" / "exogenous_tape.py")),
        "env_git_object": _git("hash-object", str(ROOT / "environment" / "env.py")),
    }


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.rstrip()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
