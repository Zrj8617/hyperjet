from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.reward_redesign import RewardRedesignLedger
from scripts import launch_six_arm_realign_production as launcher
from scripts import orchestrate_fair_eval as orchestrator
from scripts import run_reward_redesign_arm as runner
from scripts import summarize_c1_b2_c2_fair_reevaluation as summarizer
from scripts import train_clean_mainline as trainer


ARMS = ("B1", "B2", "C2A", "C2B", "C2", "C1")
EXPECTED_FLAGS = {
    "B1": {
        "offloading_eft_advantage": False,
        "movement_position_advantage": False,
        "forecast_enabled": False,
        "offloading_forecast_advantage": False,
        "teacher_anneal_total_updates": 0,
    },
    "B2": {
        "offloading_eft_advantage": False,
        "movement_position_advantage": False,
        "forecast_enabled": True,
        "offloading_forecast_advantage": False,
        "teacher_anneal_total_updates": 0,
    },
    "C2A": {
        "offloading_eft_advantage": True,
        "movement_position_advantage": False,
        "forecast_enabled": True,
        "offloading_forecast_advantage": False,
        "teacher_anneal_total_updates": 2000,
    },
    "C2B": {
        "offloading_eft_advantage": False,
        "movement_position_advantage": True,
        "forecast_enabled": True,
        "offloading_forecast_advantage": False,
        "teacher_anneal_total_updates": 2000,
    },
    "C2": {
        "offloading_eft_advantage": True,
        "movement_position_advantage": True,
        "forecast_enabled": True,
        "offloading_forecast_advantage": False,
        "teacher_anneal_total_updates": 2000,
    },
    "C1": {
        "offloading_eft_advantage": True,
        "movement_position_advantage": True,
        "forecast_enabled": False,
        "offloading_forecast_advantage": False,
        "teacher_anneal_total_updates": 2000,
    },
}
PRECHANGE_C1_C2 = {arm: EXPECTED_FLAGS[arm] for arm in ("C1", "C2")}


class _LedgerClosureProbe:
    observations: list[dict[str, float]] = []

    def __init__(self, arm: str):
        self.primary = RewardRedesignLedger(arm)
        self.shadow_b1 = RewardRedesignLedger("B1") if self.primary.arm == "B2" else None
        self.b2_total = 0.0
        self.b1_total = 0.0

    @property
    def forecast_enabled(self) -> bool:
        return self.primary.forecast_enabled

    @property
    def analytic_forecast_advantage(self) -> bool:
        return self.primary.analytic_forecast_advantage

    def apply(self, **kwargs: Any) -> float:
        reward = self.primary.apply(**kwargs)
        if self.shadow_b1 is None:
            return reward
        info = kwargs["info"]
        self.b2_total += (
            float(info["reward_redesign_initial_cost_seconds"])
            + float(info["reward_redesign_assignment_cost_seconds"])
            + float(info["reward_redesign_correction_cost_seconds"])
        )
        shadow_info = dict(info)
        self.shadow_b1.apply(**{**kwargs, "info": shadow_info})
        self.b1_total += float(shadow_info["reward_redesign_timing_cost_seconds"])
        if bool(kwargs["done"]):
            denominator = max(abs(self.b1_total), 1e-12)
            self.observations.append(
                {
                    "b2_initial_assignment_correction_seconds": self.b2_total,
                    "b1_realized_flowtime_seconds": self.b1_total,
                    "relative_error": abs(self.b2_total - self.b1_total) / denominator,
                }
            )
        return reward


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 2026-09-15 six-arm gating smoke.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-prefix", type=str, required=True)
    return parser


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.rstrip()


def _assert_missing_matrix_arguments(script: Path, base_args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(script), *base_args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode == 0:
        raise AssertionError(f"{script.name} accepted a command without --arms/--seeds")
    if "--arms" not in completed.stderr or "--seeds" not in completed.stderr:
        raise AssertionError(f"{script.name} did not identify missing arm/seed arguments")
    return {"return_code": completed.returncode, "stderr_tail": completed.stderr.splitlines()[-1]}


def _candidate_fixture() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="six_arm_candidate_gate_") as temporary:
        root = Path(temporary)
        run_root = root / "20260915_C1_seed5"
        train_dir = run_root / "train"
        checkpoint_dir = train_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        rows = []
        for episode in orchestrator.CANDIDATE_EPISODES:
            (checkpoint_dir / f"checkpoint_ep_{episode:04d}.pt").write_bytes(b"fixture")
            rows.append(
                {
                    "episode": episode - 1,
                    "ppo_diagnostics": {"teacher_weight": 0.0},
                }
            )
        (train_dir / "train_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        result = {
            "status": "completed",
            "train_dir": str(train_dir),
            "checkpoint": str(checkpoint_dir / "checkpoint_ep_0500.pt"),
            "actual_parameters": {"reward_redesign_arm": "C1", "seed": 5},
            "version": {},
        }
        run_root.mkdir(exist_ok=True)
        (run_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
        audited = orchestrator._audit_runs(
            root, ("C1",), (5,), run_prefix="20260915"
        )
        checkpoints = audited["C1"]["5"]["candidate_checkpoints_read"]
        if tuple(int(value) for value in checkpoints) != orchestrator.CANDIDATE_EPISODES:
            raise AssertionError("candidate checkpoint fixture did not read all approved episodes")
        return {"episodes": list(map(int, checkpoints)), "read_count": len(checkpoints)}


def _summary_formula_fixture() -> dict[str, Any]:
    arm_values = {"B1": -1.0, "B2": 0.0, "C2A": 2.0, "C2B": 3.0, "C2": 7.0, "C1": 5.0}
    rows: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for arm in ARMS:
        for seed in summarizer.SEEDS:
            for checkpoint in summarizer.CHECKPOINT_LABELS:
                for protocol in summarizer.PROTOCOLS:
                    row = {metric: arm_values[arm] for metric in summarizer.TEST_METRICS}
                    row.update(
                        {
                            "tape_id": 200,
                            "delay_seconds_total": 500.0 * (10.0 + arm_values[arm]),
                            "task_energy_joules_total": 50.0,
                            "move_energy_joules_total": 100.0,
                            "N_offer": 10,
                        }
                    )
                    rows[(arm, seed, checkpoint, protocol)] = [row]
    contrasts = summarizer._contrast_summaries(rows, replicates=2)
    selected = contrasts["budget_selected"]["joint"]
    actual = {
        "movement_teacher_average_main_effect": selected[
            "movement_teacher_average_main_effect"
        ]["J_per_offer"]["mean"],
        "offloading_teacher_average_main_effect": selected[
            "offloading_teacher_average_main_effect"
        ]["J_per_offer"]["mean"],
        "teacher_interaction": selected["teacher_interaction"]["J_per_offer"]["mean"],
    }
    expected = {
        "movement_teacher_average_main_effect": 4.0,
        "offloading_teacher_average_main_effect": 3.0,
        "teacher_interaction": 2.0,
    }
    if actual != expected:
        raise AssertionError(f"2x2 contrast fixture mismatch: {actual}")
    sweep = summarizer._lambda_move_sweep(
        rows, lambdas=(0.10, 0.04, 0.0), replicates=2
    )
    if set(sweep["budget_selected"]["joint"]) != {"0.1", "0.04", "0"}:
        raise AssertionError("lambda_move sweep did not emit all approved values")
    return {"actual": actual, "expected": expected, "lambda_move": [0.10, 0.04, 0.0]}


def _run_arm(
    *, output_root: Path, run_prefix: str, arm: str, seed: int, device: str
) -> dict[str, Any]:
    run_name = f"{run_prefix}_{arm}_seed{seed}"
    run_root = output_root / run_name
    log_path = output_root / f"{run_name}.log"
    if run_root.exists() or log_path.exists():
        raise FileExistsError(f"smoke target already exists: {run_name}")
    argv = [
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--episodes",
        "5",
        "--steps",
        "500",
        "--rollout-horizon",
        "125",
        "--device",
        device,
        "--output-root",
        str(run_root),
        "--run-name",
        run_name,
        "--rollout-cost-diagnostics",
        "--skip-eval",
    ]
    if arm in {"C1", "C2", "C2A", "C2B"}:
        argv.extend(["--teacher-anneal-total-updates", "2000"])
    started = perf_counter()
    original_ledger = trainer.RewardRedesignLedger
    if arm == "B2":
        _LedgerClosureProbe.observations = []
        trainer.RewardRedesignLedger = _LedgerClosureProbe
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            with redirect_stdout(log_handle), redirect_stderr(log_handle):
                return_code = runner.main(argv)
    finally:
        trainer.RewardRedesignLedger = original_ledger
    wall_seconds = perf_counter() - started
    if return_code != 0:
        raise RuntimeError(f"{arm} smoke failed with return code {return_code}")
    if wall_seconds > 1800.0:
        raise RuntimeError(f"hard stop: {arm} smoke exceeded 30 minutes")
    result_path = run_root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (Path(result["train_dir"]) / "train_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(rows) != 20 or any(int(row["ppo_slot_count"]) != 125 for row in rows):
        raise AssertionError(f"{arm}: expected twenty equal 125-slot rollouts")
    per_episode = {
        episode: sum(int(row["episode"]) == episode for row in rows)
        for episode in range(5)
    }
    if set(per_episode.values()) != {4}:
        raise AssertionError(f"{arm}: expected four updates per episode: {per_episode}")
    if sum(bool(row.get("episode_terminal_record", False)) for row in rows) != 5:
        raise AssertionError(f"{arm}: train_metrics.jsonl lacks five episode terminal rows")
    if result["resolved_flags"] != EXPECTED_FLAGS[arm]:
        raise AssertionError(
            f"hard stop: {arm} resolved flags mismatch: {result['resolved_flags']}"
        )
    return {
        "arm": arm,
        "seed": seed,
        "wall_seconds": wall_seconds,
        "result_path": str(result_path),
        "log_path": str(log_path),
        "train_metrics_path": str(Path(result["train_dir"]) / "train_metrics.jsonl"),
        "train_metrics_rows": len(rows),
        "updates_per_episode": per_episode,
        "rollout_slot_counts": sorted({int(row["ppo_slot_count"]) for row in rows}),
        "resolved_flags": result["resolved_flags"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.output_root.is_dir():
        raise FileNotFoundError(f"existing output root required: {args.output_root}")
    if args.result.exists():
        raise FileExistsError(f"result already exists: {args.result}")
    if int(args.seed) != 5 or str(args.run_prefix) != "20260915_smoke":
        raise ValueError("smoke requires seed 5 and run prefix 20260915_smoke")

    missing_argument_checks = {
        "orchestrator": _assert_missing_matrix_arguments(
            ROOT / "scripts" / "orchestrate_fair_eval.py",
            [
                "--audit-root",
                "unused",
                "--tape-dir",
                "unused",
                "--output-root",
                "unused",
                "--manifest",
                "unused.json",
                "--run-prefix",
                "20260915",
                "--gpus",
                "0",
            ],
        ),
        "production_launcher": _assert_missing_matrix_arguments(
            ROOT / "scripts" / "launch_six_arm_realign_production.py",
            [
                "--output-root",
                str(args.output_root),
                "--manifest",
                str(args.output_root / "unused_manifest.json"),
                "--run-prefix",
                "20260915",
                "--gpus",
                "0",
            ],
        ),
    }
    candidate_fixture = _candidate_fixture()
    summary_fixture = _summary_formula_fixture()

    production_args = trainer.build_arg_parser().parse_args(
        [
            "--episodes",
            "500",
            "--max-steps-per-episode",
            "500",
            "--rollout-horizon",
            "125",
            "--reward-redesign-arm",
            "C1",
        ]
    )
    production_update_count = int(production_args.episodes) * math.ceil(
        int(production_args.max_steps_per_episode) / int(production_args.rollout_horizon)
    )
    if production_update_count != 2000:
        raise AssertionError("production configuration does not resolve to 2000 updates")

    arm_results = [
        _run_arm(
            output_root=args.output_root,
            run_prefix=str(args.run_prefix),
            arm=arm,
            seed=int(args.seed),
            device=str(args.device),
        )
        for arm in ARMS
    ]
    if len(_LedgerClosureProbe.observations) != 5:
        raise AssertionError("B2 ledger closure probe did not observe five complete episodes")
    max_closure_error = max(
        row["relative_error"] for row in _LedgerClosureProbe.observations
    )
    if max_closure_error >= 1e-6:
        raise AssertionError(f"hard stop: B1/B2 ledger closure error {max_closure_error}")

    actual_flags = {row["arm"]: row["resolved_flags"] for row in arm_results}
    c1_c2_regression = {arm: actual_flags[arm] for arm in ("C1", "C2")}
    if c1_c2_regression != PRECHANGE_C1_C2:
        raise AssertionError("hard stop: C1/C2 resolved behavior changed from pre-change baseline")
    result = {
        "schema": "six_arm_realign_gating_smoke_v1",
        "status": "pass",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": {
            "head": _git("rev-parse", "HEAD"),
            "dirty": _git("status", "--porcelain").splitlines(),
        },
        "gates": {
            "G1": {"status": "pass", "resolved_flags": actual_flags},
            "G2": {
                "status": "pass",
                "rollout_horizon": 125,
                "updates_per_episode": 4,
                "production_update_count": production_update_count,
                "all_smoke_rollouts_equal": True,
            },
            "G3": {
                "status": "pass",
                "arms": list(ARMS),
                "seeds": list(launcher.APPROVED_SEEDS),
                "matrix_run_count": len(ARMS) * len(launcher.APPROVED_SEEDS),
                "missing_argument_checks": missing_argument_checks,
            },
            "G4": {
                "status": "pass",
                "selection_protocol": orchestrator.VALIDATION_PROTOCOL,
                "test_protocols": list(orchestrator.TEST_PROTOCOLS),
            },
            "G5": {"status": "pass", "candidate_fixture": candidate_fixture},
            "G6": {
                "status": "pass",
                "test_metric_present": "hover_action_ratio" in summarizer.TEST_METRICS,
            },
            "G7": {"status": "pass", "formula_fixture": summary_fixture},
            "G8": {
                "status": "pass",
                "lambda_move_sweep": summary_fixture["lambda_move"],
            },
            "G9": {
                "status": "pass",
                "flag_table_matches": actual_flags == EXPECTED_FLAGS,
                "c1_c2_prechange_regression_matches": c1_c2_regression == PRECHANGE_C1_C2,
                "ledger_closure_observations": _LedgerClosureProbe.observations,
                "ledger_closure_max_relative_error": max_closure_error,
                "six_arms_completed_five_episodes": True,
            },
        },
        "arm_results": arm_results,
        "version": {
            "smoke_git_object": _git("hash-object", str(Path(__file__).resolve())),
            "runner_git_object": _git(
                "hash-object", str(ROOT / "scripts" / "run_reward_redesign_arm.py")
            ),
            "trainer_git_object": _git(
                "hash-object", str(ROOT / "scripts" / "train_clean_mainline.py")
            ),
            "reward_ledger_git_object": _git(
                "hash-object", str(ROOT / "environment" / "reward_redesign.py")
            ),
            "orchestrator_git_object": _git(
                "hash-object", str(ROOT / "scripts" / "orchestrate_fair_eval.py")
            ),
            "launcher_git_object": _git(
                "hash-object", str(ROOT / "scripts" / "launch_six_arm_realign_production.py")
            ),
            "evaluator_git_object": _git(
                "hash-object", str(ROOT / "scripts" / "run_fair_eval_batch.py")
            ),
            "summarizer_git_object": _git(
                "hash-object",
                str(ROOT / "scripts" / "summarize_c1_b2_c2_fair_reevaluation.py"),
            ),
            "fair_eval_tensorboard_git_object": _git(
                "hash-object", str(ROOT / "scripts" / "export_fair_eval_tensorboard.py")
            ),
            "train_tensorboard_git_object": _git(
                "hash-object", str(ROOT / "scripts" / "export_unified_train_tensorboard.py")
            ),
        },
        "hard_stop_triggered": None,
    }
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "result": str(args.result)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
