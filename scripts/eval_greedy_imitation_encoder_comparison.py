from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_greedy_imitation_gate as gate
from scripts.run_greedy_imitation_encoder_comparison import CANONICAL_ENCODERS


EVAL_SCHEMA = "greedy_imitation_checkpoint_closed_loop_eval_v1"
CHECKPOINT_SCHEMA = "greedy_imitation_encoder_comparison_checkpoint_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run checkpoint-only closed-loop evaluation for an existing greedy "
            "imitation encoder comparison. This entry point never trains models "
            "or regenerates the frozen dataset."
        )
    )
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[42, 86, 1042])
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=CANONICAL_ENCODERS,
        default=list(CANONICAL_ENCODERS),
    )
    parser.add_argument("--closed-loop-eval-episodes", type=int, default=100)
    parser.add_argument("--closed-loop-eval-seed", type=int, default=1_000_042)
    parser.add_argument("--max-steps-per-episode", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.smoke:
        args.closed_loop_eval_episodes = 1
        args.max_steps_per_episode = 1
        args.device = "cpu"
    _validate_args(args)

    comparison_dir = Path(args.comparison_dir).resolve()
    comparison = _load_comparison(comparison_dir)
    selected_rows = _select_rows(
        comparison["rows"],
        encoders=[str(value) for value in args.encoders],
        training_seeds=[int(value) for value in args.training_seeds],
    )
    _validate_selected_rows(comparison, selected_rows)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else comparison_dir / "checkpoint_closed_loop_eval"
    )
    eval_config = _build_eval_config(
        args=args,
        comparison_dir=comparison_dir,
        dataset_checksum=str(comparison["summary"]["dataset_checksum"]),
        selected_rows=selected_rows,
    )
    _initialize_output_dir(output_dir, eval_config)

    results: list[dict[str, Any]] = []
    for row in selected_rows:
        result = _evaluate_or_resume_variant(
            args=args,
            comparison=comparison,
            row=row,
            output_dir=output_dir,
            eval_config=eval_config,
        )
        results.append(result)
        _write_aggregate_outputs(output_dir, eval_config, results)
    _write_aggregate_outputs(output_dir, eval_config, results)
    return 0


def _load_comparison(comparison_dir: Path) -> dict[str, Any]:
    summary_path = comparison_dir / "comparison_summary.json"
    rows_path = comparison_dir / "comparison_rows.jsonl"
    config_path = comparison_dir / "comparison_config.json"
    for path in (summary_path, rows_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"required comparison artifact is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if summary.get("status") != "completed":
        raise ValueError("comparison status must be completed")
    if summary.get("technical_pass") is not True:
        raise ValueError("comparison technical_pass must be true")
    if int(summary.get("split_leakage_count", -1)) != 0:
        raise ValueError("comparison split leakage count must be zero")
    if not rows:
        raise ValueError("comparison_rows.jsonl contains no variants")
    return {
        "dir": comparison_dir,
        "summary": summary,
        "config": config,
        "rows": rows,
    }


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    encoders: list[str],
    training_seeds: list[int],
) -> list[dict[str, Any]]:
    requested = {(str(encoder), int(seed)) for seed in training_seeds for encoder in encoders}
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["encoder"]), int(row["training_seed"]))
        if key in indexed:
            raise ValueError(f"duplicate comparison row for {key}")
        indexed[key] = row
    missing = sorted(requested - set(indexed))
    if missing:
        raise ValueError(f"comparison is missing requested variants: {missing}")
    return [indexed[(encoder, seed)] for seed in training_seeds for encoder in encoders]


def _validate_selected_rows(
    comparison: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    expected_checksum = str(comparison["summary"]["dataset_checksum"])
    sample_hashes = {
        json.dumps(row.get("sample_id_hashes"), sort_keys=True)
        for row in rows
    }
    if len(sample_hashes) != 1:
        raise ValueError("selected variants do not share identical sample/split hashes")
    for row in rows:
        encoder = str(row["encoder"])
        training_seed = int(row["training_seed"])
        if row.get("technical_pass") is not True:
            raise ValueError(f"{encoder} seed {training_seed} did not technically pass")
        if row.get("finite_gradient_norm") is not True:
            raise ValueError(f"{encoder} seed {training_seed} has a non-finite gradient")
        if str(row.get("dataset_checksum")) != expected_checksum:
            raise ValueError(f"{encoder} seed {training_seed} dataset checksum mismatch")
        variant_dir = _resolve_variant_dir(comparison["dir"], row)
        for filename in ("config.json", "checkpoint.pt"):
            path = variant_dir / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"{encoder} seed {training_seed} is missing {filename}: {path}"
                )


def _build_eval_config(
    *,
    args: argparse.Namespace,
    comparison_dir: Path,
    dataset_checksum: str,
    selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": EVAL_SCHEMA,
        "comparison_dir": str(comparison_dir),
        "dataset_checksum": str(dataset_checksum),
        "encoders": [str(value) for value in args.encoders],
        "training_seeds": [int(value) for value in args.training_seeds],
        "closed_loop_eval_episodes": int(args.closed_loop_eval_episodes),
        "closed_loop_eval_seed": int(args.closed_loop_eval_seed),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "device": str(args.device),
        "movement_mode": "forced_fixed_empty_movement_action",
        "variant_keys": [
            f"{row['encoder']}:seed_{int(row['training_seed'])}"
            for row in selected_rows
        ],
    }
    payload["config_hash"] = _json_hash(payload)
    return payload


def _initialize_output_dir(output_dir: Path, eval_config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "eval_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != eval_config["config_hash"]:
            raise ValueError(
                "output directory contains a different evaluation config; "
                "refusing to mix results"
            )
        return
    _write_json_atomic(config_path, eval_config)


def _evaluate_or_resume_variant(
    *,
    args: argparse.Namespace,
    comparison: dict[str, Any],
    row: dict[str, Any],
    output_dir: Path,
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    encoder = str(row["encoder"])
    training_seed = int(row["training_seed"])
    variant_key = f"{encoder}:seed_{training_seed}"
    source_dir = _resolve_variant_dir(comparison["dir"], row)
    checkpoint_path = source_dir / "checkpoint.pt"
    variant_config_path = source_dir / "config.json"
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    variant_config_sha256 = _file_sha256(variant_config_path)
    final_dir = output_dir / "variants" / encoder / f"seed_{training_seed}"
    result_path = final_dir / "variant_result.json"
    expected_identity = {
        "eval_config_hash": eval_config["config_hash"],
        "variant_key": variant_key,
        "checkpoint_sha256": checkpoint_sha256,
        "variant_config_sha256": variant_config_sha256,
        "dataset_checksum": str(comparison["summary"]["dataset_checksum"]),
    }
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") != "completed":
            raise ValueError(f"existing result is incomplete: {result_path}")
        for key, value in expected_identity.items():
            if existing.get(key) != value:
                raise ValueError(f"existing result identity mismatch for {variant_key}: {key}")
        return existing
    if final_dir.exists():
        raise ValueError(f"variant output exists without a completed result: {final_dir}")

    attempt_dir = (
        output_dir
        / "_attempts"
        / f"{encoder}_seed_{training_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    modules = _build_and_load_modules(
        args=args,
        comparison=comparison,
        row=row,
        checkpoint_path=checkpoint_path,
        variant_config_path=variant_config_path,
    )
    eval_args = argparse.Namespace(
        seed=int(training_seed),
        task_encoder=encoder,
        completed_dag_weight=float(
            comparison["config"].get("cli", {}).get("completed_dag_weight", 16.0)
        ),
        max_steps_per_episode=int(args.max_steps_per_episode),
        closed_loop_eval_episodes=int(args.closed_loop_eval_episodes),
        closed_loop_eval_seed=int(args.closed_loop_eval_seed),
    )
    start = time.perf_counter()
    with modules.torch.no_grad():
        closed_loop = gate._run_closed_loop_eval(
            args=eval_args,
            run_dir=attempt_dir,
            modules=modules,
        )
    result = {
        "schema": EVAL_SCHEMA,
        "status": "completed",
        **expected_identity,
        "encoder": encoder,
        "training_seed": training_seed,
        "checkpoint_path": str(checkpoint_path),
        "closed_loop_eval_episodes": int(args.closed_loop_eval_episodes),
        "closed_loop_eval_seed": int(args.closed_loop_eval_seed),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "movement_mode": "forced_fixed_empty_movement_action",
        "duration_seconds": float(time.perf_counter() - start),
        "closed_loop": closed_loop,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(attempt_dir / "variant_result.json", result)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt_dir.replace(final_dir)
    return result


def _build_and_load_modules(
    *,
    args: argparse.Namespace,
    comparison: dict[str, Any],
    row: dict[str, Any],
    checkpoint_path: Path,
    variant_config_path: Path,
) -> gate.GateModules:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for checkpoint-only evaluation") from exc

    variant_config = json.loads(variant_config_path.read_text(encoding="utf-8"))
    manifest_path = Path(str(comparison["summary"]["dataset_manifest_path"]))
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_checksum = str(comparison["summary"]["dataset_checksum"])
    if str(manifest.get("dataset_checksum")) != expected_checksum:
        raise ValueError("dataset manifest checksum differs from comparison summary")
    hyperparameters = variant_config["hyperparameters"]
    module_args = argparse.Namespace(
        seed=int(row["training_seed"]),
        task_encoder=str(row["encoder"]),
        task_feature_dim=int(manifest["task_feature_dimension"]),
        task_embedding_dim=int(hyperparameters["task_embedding_dim"]),
        hidden_dim=int(hyperparameters["hidden_dim"]),
        lr=float(hyperparameters["learning_rate"]),
        gradient_batch_decisions=int(hyperparameters["gradient_batch_decisions"]),
        max_grad_norm=float(hyperparameters["max_grad_norm"]),
        completed_dag_weight=float(
            comparison["config"].get("cli", {}).get("completed_dag_weight", 16.0)
        ),
        device=str(args.device),
    )
    modules = gate._build_modules(
        module_args,
        encoder_seed=int(row["training_seed"]),
        scorer_seed=None,
    )
    checkpoint = _torch_load(torch, checkpoint_path, modules.device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint payload must be a dictionary: {checkpoint_path}")
    expected_fields = {
        "schema": CHECKPOINT_SCHEMA,
        "encoder": str(row["encoder"]),
        "training_seed": int(row["training_seed"]),
        "dataset_checksum": expected_checksum,
    }
    for key, value in expected_fields.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"checkpoint {key} mismatch for {row['encoder']} seed "
                f"{row['training_seed']}: expected {value!r}, got {checkpoint.get(key)!r}"
            )
    for key in ("task_encoder_state_dict", "candidate_scorer_state_dict"):
        if key not in checkpoint:
            raise ValueError(f"checkpoint is missing required state: {key}")
    modules.task_encoder.load_state_dict(
        checkpoint["task_encoder_state_dict"],
        strict=True,
    )
    modules.offloading_actor.scorer.load_state_dict(
        checkpoint["candidate_scorer_state_dict"],
        strict=True,
    )
    modules.task_encoder.eval()
    modules.offloading_actor.scorer.eval()
    return modules


def _torch_load(torch: Any, path: Path, device: Any) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _write_aggregate_outputs(
    output_dir: Path,
    eval_config: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    ordered = sorted(
        results,
        key=lambda row: (
            eval_config["training_seeds"].index(int(row["training_seed"])),
            eval_config["encoders"].index(str(row["encoder"])),
        ),
    )
    lines = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        for row in ordered
    )
    _write_text_atomic(output_dir / "closed_loop_rows.jsonl", lines)
    _write_json_atomic(
        output_dir / "closed_loop_summary.json",
        {
            "schema": EVAL_SCHEMA,
            "status": (
                "completed"
                if len(ordered) == len(eval_config["variant_keys"])
                else "in_progress"
            ),
            "eval_config_hash": eval_config["config_hash"],
            "dataset_checksum": eval_config["dataset_checksum"],
            "completed_variant_count": len(ordered),
            "expected_variant_count": len(eval_config["variant_keys"]),
            "rows": ordered,
        },
    )


def _resolve_variant_dir(comparison_dir: Path, row: dict[str, Any]) -> Path:
    configured = Path(str(row["variant_dir"]))
    if configured.is_absolute():
        return configured
    root_relative = ROOT / configured
    if root_relative.exists():
        return root_relative
    return comparison_dir / "variants" / str(row["encoder"]) / f"seed_{int(row['training_seed'])}"


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.closed_loop_eval_episodes) <= 0:
        raise ValueError("closed-loop eval episodes must be positive")
    if int(args.max_steps_per_episode) <= 0:
        raise ValueError("max steps per episode must be positive")
    if len(set(args.encoders)) != len(args.encoders):
        raise ValueError("encoders must not contain duplicates")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError("training seeds must not contain duplicates")


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
