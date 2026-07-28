from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state
from scripts import eval_greedy_imitation_encoder_comparison as evaluator
from scripts import train_greedy_imitation_gate as gate


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = evaluator.build_arg_parser()
    parsed = parser.parse_args(["--comparison-dir", "comparison", "--smoke"])
    _assert(parsed.smoke, "smoke flag should parse")
    _assert(parsed.closed_loop_eval_episodes == 100, "parser defaults must remain formal")

    missing = ROOT / ".codex_missing_checkpoint.pt"
    try:
        evaluator._torch_load(_fake_torch(), missing, "cpu")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing checkpoint must raise FileNotFoundError")

    if not _torch_available():
        print("smoke_greedy_imitation_encoder_eval skipped torch evaluation: torch is not installed")
        return 0

    import torch

    with _workspace_temp_dir("checkpoint_eval") as temp_dir:
        comparison_dir = temp_dir / "comparison"
        output_dir = temp_dir / "eval"
        comparison_dir.mkdir(parents=True)
        dataset_checksum = "smoke-dataset-checksum"
        task_feature_dim = _task_feature_dimension()
        manifest_path = comparison_dir / "dataset_manifest.json"
        _write_json(
            manifest_path,
            {
                "dataset_checksum": dataset_checksum,
                "task_feature_dimension": task_feature_dim,
            },
        )
        seed = 17
        encoder = "mlp"
        variant_dir = comparison_dir / "variants" / encoder / f"seed_{seed}"
        variant_dir.mkdir(parents=True)
        hyperparameters = {
            "task_embedding_dim": 8,
            "hidden_dim": 16,
            "learning_rate": 3e-4,
            "gradient_batch_decisions": 4,
            "max_grad_norm": 0.5,
        }
        _write_json(
            variant_dir / "config.json",
            {
                "encoder": encoder,
                "training_seed": seed,
                "dataset_checksum": dataset_checksum,
                "hyperparameters": hyperparameters,
            },
        )
        module_args = argparse.Namespace(
            seed=seed,
            task_encoder=encoder,
            task_feature_dim=task_feature_dim,
            task_embedding_dim=8,
            hidden_dim=16,
            lr=3e-4,
            gradient_batch_decisions=4,
            max_grad_norm=0.5,
            completed_dag_weight=16.0,
            device="cpu",
        )
        modules = gate._build_modules(module_args, encoder_seed=seed, scorer_seed=123)
        checkpoint_path = variant_dir / "checkpoint.pt"
        torch.save(
            {
                "schema": evaluator.CHECKPOINT_SCHEMA,
                "encoder": encoder,
                "training_seed": seed,
                "dataset_checksum": dataset_checksum,
                "task_encoder_state_dict": modules.task_encoder.state_dict(),
                "candidate_scorer_state_dict": modules.offloading_actor.scorer.state_dict(),
            },
            checkpoint_path,
        )
        row = {
            "encoder": encoder,
            "training_seed": seed,
            "variant_dir": str(variant_dir),
            "dataset_checksum": dataset_checksum,
            "sample_id_hashes": {"train": "a", "val": "b", "test": "c"},
            "technical_pass": True,
            "finite_gradient_norm": True,
        }
        _write_json(
            comparison_dir / "comparison_config.json",
            {"cli": {"completed_dag_weight": 16.0}},
        )
        _write_json(
            comparison_dir / "comparison_summary.json",
            {
                "status": "completed",
                "technical_pass": True,
                "split_leakage_count": 0,
                "dataset_checksum": dataset_checksum,
                "dataset_manifest_path": str(manifest_path),
            },
        )
        (comparison_dir / "comparison_rows.jsonl").write_text(
            json.dumps(row, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        command = [
            "--comparison-dir",
            str(comparison_dir),
            "--output-dir",
            str(output_dir),
            "--training-seeds",
            str(seed),
            "--encoders",
            encoder,
            "--closed-loop-eval-seed",
            "1234",
            "--smoke",
        ]
        _assert(evaluator.main(command) == 0, "checkpoint-only smoke should complete")
        result_path = output_dir / "variants" / encoder / f"seed_{seed}" / "variant_result.json"
        metrics_path = (
            output_dir
            / "variants"
            / encoder
            / f"seed_{seed}"
            / "closed_loop_eval_metrics.jsonl"
        )
        _assert(result_path.is_file(), "variant result is missing")
        _assert(metrics_path.is_file(), "closed-loop metrics are missing")
        first_result_mtime = result_path.stat().st_mtime_ns
        first_metrics = metrics_path.read_text(encoding="utf-8")
        _assert(len(first_metrics.splitlines()) == 3, "one episode for three policies expected")

        _assert(evaluator.main(command) == 0, "completed evaluation should resume idempotently")
        _assert(result_path.stat().st_mtime_ns == first_result_mtime, "resume rewrote result")
        _assert(
            metrics_path.read_text(encoding="utf-8") == first_metrics,
            "resume duplicated closed-loop episode rows",
        )
        summary = json.loads(
            (output_dir / "closed_loop_summary.json").read_text(encoding="utf-8")
        )
        _assert(summary["status"] == "completed", "aggregate summary must complete")
        _assert(summary["completed_variant_count"] == 1, "aggregate variant count drifted")

        bad_checkpoint = temp_dir / "bad_checkpoint.pt"
        torch.save({"schema": "wrong"}, bad_checkpoint)
        try:
            evaluator._build_and_load_modules(
                args=argparse.Namespace(device="cpu"),
                comparison=evaluator._load_comparison(comparison_dir),
                row=row,
                checkpoint_path=bad_checkpoint,
                variant_config_path=variant_dir / "config.json",
            )
        except ValueError as exc:
            _assert("checkpoint schema mismatch" in str(exc), "invalid checkpoint error unclear")
        else:
            raise AssertionError("invalid checkpoint must fail strict validation")

    print("smoke_greedy_imitation_encoder_eval passed")
    return 0


def _task_feature_dimension() -> int:
    env = Env(completed_dag_weight=16.0, freeze_ue_mobility=True)
    graph_builder = CleanGraphBuilder()
    try:
        env.reset()
        graph_builder.reset()
        prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
        return int(prepared.graph_snapshot.task_features.shape[1])
    finally:
        graph_builder.close()


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class _fake_torch:
    @staticmethod
    def load(*args, **kwargs):
        raise AssertionError("torch.load should not run for a missing checkpoint")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class _workspace_temp_dir:
    def __init__(self, name: str) -> None:
        self.path = ROOT / ".codex_tmp_greedy_imitation_encoder_eval" / (
            f"{name}_{os.getpid()}_{random.randint(0, 1_000_000)}"
        )

    def __enter__(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        parent = ROOT / ".codex_tmp_greedy_imitation_encoder_eval"
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
