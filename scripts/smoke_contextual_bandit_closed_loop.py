from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_contextual_bandit_closed_loop as closed_loop
from scripts import run_contextual_bandit_gate as bandit
from scripts import train_greedy_imitation_gate as gate


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("smoke_contextual_bandit_closed_loop skipped: torch is not installed")
        return 0
    with tempfile.TemporaryDirectory(prefix="bandit_closed_loop_smoke_") as temp:
        root = Path(temp)
        bandit_dir = root / "bandit"
        seed_dir = bandit_dir / "seed_42"
        seed_dir.mkdir(parents=True)
        modules = _build_modules()
        scale = {
            "method": "train_legal_candidate_raw_eft_regret_rms",
            "value": 1.0,
        }
        for stage in ("initial", "trained"):
            torch.save(
                {
                    "schema": bandit.CHECKPOINT_SCHEMA,
                    "stage": stage,
                    "encoder": "mlp",
                    "training_seed": 42,
                    "dataset_checksum": "smoke_checksum",
                    "task_feature_dim": 12,
                    "task_embedding_dim": 4,
                    "hidden_dim": 8,
                    "regret_scale": scale,
                    "task_encoder_state_dict": modules.task_encoder.state_dict(),
                    "candidate_scorer_state_dict": modules.offloading_actor.scorer.state_dict(),
                },
                seed_dir / f"{stage}_checkpoint.pt",
            )
        (bandit_dir / "summary.json").write_text(
            json.dumps(
                {
                    "schema": bandit.SCHEMA,
                    "status": "completed",
                    "technical_pass": True,
                    "epochs": 5,
                    "training_seeds": [42],
                    "dataset_checksum": "smoke_checksum",
                }
            ),
            encoding="utf-8",
        )
        output_dir = root / "eval"
        exit_code = closed_loop.main(
            [
                "--bandit-dir",
                str(bandit_dir),
                "--output-dir",
                str(output_dir),
                "--training-seeds",
                "42",
                "--smoke",
            ]
        )
        _assert(exit_code == 0, "closed-loop smoke exited unsuccessfully")
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        _assert(summary["status"] == "completed", "closed-loop smoke did not complete")
        _assert(summary["group_count"] == 4, "closed-loop policy group count drifted")
        _assert(summary["finite"] is True, "closed-loop smoke metrics are non-finite")
        missing_failed = False
        try:
            closed_loop.load_bandit_checkpoint(
                root / "missing.pt",
                expected_seed=42,
                expected_stage="trained",
                expected_dataset_checksum="smoke_checksum",
                device="cpu",
            )
        except FileNotFoundError:
            missing_failed = True
        _assert(missing_failed, "missing checkpoint did not fail explicitly")
    print("smoke_contextual_bandit_closed_loop passed")
    return 0


def _build_modules() -> gate.GateModules:
    args = argparse.Namespace(
        seed=42,
        task_encoder="mlp",
        task_feature_dim=12,
        task_embedding_dim=4,
        hidden_dim=8,
        lr=3e-4,
        gradient_batch_decisions=8,
        max_grad_norm=0.5,
        completed_dag_weight=16.0,
        device="cpu",
    )
    return gate._build_modules(args, encoder_seed=42, scorer_seed=42)


if __name__ == "__main__":
    raise SystemExit(main())
