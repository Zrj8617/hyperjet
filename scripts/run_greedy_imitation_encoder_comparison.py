from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from scripts import greedy_imitation_dataset as frozen_data
from scripts import greedy_imitation_grouped_batch as grouped_batch
from scripts import train_greedy_imitation_gate as gate


CANONICAL_ENCODERS = (
    "mlp",
    "current_mean_hgnn",
    "standard_weighted_hgnn",
    "typed_gated_hgnn",
)
COMPARISON_SCHEMA = "greedy_imitation_encoder_comparison_v1"
TYPE_NAMES = ("dag", "khop", "attribute", "partition")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare clean task encoders on one frozen greedy-EFT imitation dataset. "
            "This runner does not use PPO, reward learning, critic, GAE, or movement training."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--generate-dataset", action="store_true", default=False)
    parser.add_argument("--reuse-dataset", action="store_true", default=False)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps-per-episode", type=int, default=int(config.EPISODE_LENGTH))
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[42, 86, 1042])
    parser.add_argument(
        "--trajectory-policies",
        nargs="+",
        choices=gate.TRAJECTORY_POLICIES,
        default=list(gate.TRAJECTORY_POLICIES),
    )
    parser.add_argument("--encoders", nargs="+", choices=CANONICAL_ENCODERS, default=list(CANONICAL_ENCODERS))
    parser.add_argument("--task-embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--supervised-epochs", type=int, default=3)
    parser.add_argument("--gradient-batch-decisions", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--completed-dag-weight", type=float, default=16.0)
    parser.add_argument("--dag-base-arrival-prob", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs") / "greedy_imitation_encoder_comparison",
    )
    parser.add_argument("--run-name", type=str, default="encoder_comparison")
    parser.add_argument("--skip-closed-loop-eval", action="store_true", default=False)
    parser.add_argument("--closed-loop-eval-episodes", type=int, default=100)
    parser.add_argument("--closed-loop-eval-seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args = _apply_smoke_overrides(args)
    _validate_args(args)
    run_dir = _create_run_dir(args)
    frozen_data.write_json(
        run_dir / "comparison_config.json",
        {
            "schema": COMPARISON_SCHEMA,
            "git_commit": gate._git_commit(),
            "smoke_only": bool(args.smoke),
            "not_a_scientific_result": bool(args.smoke),
            "cli": gate._namespace_to_dict(args),
            "canonical_encoders": list(CANONICAL_ENCODERS),
            "actual_encoders": list(args.encoders),
        },
    )

    dataset_dir = _resolve_dataset_dir(args, run_dir)
    if bool(args.generate_dataset):
        samples, manifest = frozen_data.generate_frozen_dataset(
            dataset_dir=dataset_dir,
            dataset_seed=int(args.dataset_seed),
            episodes=int(args.episodes),
            max_steps_per_episode=int(args.max_steps_per_episode),
            trajectory_policies=list(args.trajectory_policies),
            train_fraction=float(args.train_fraction),
            val_fraction=float(args.val_fraction),
            completed_dag_weight=float(args.completed_dag_weight),
            dag_base_arrival_prob=args.dag_base_arrival_prob,
        )
    else:
        samples, manifest = frozen_data.load_frozen_dataset(dataset_dir)

    try:
        import torch
    except ModuleNotFoundError:
        frozen_data.write_json(
            run_dir / "comparison_summary.json",
            {
                "schema": COMPARISON_SCHEMA,
                "status": "torch_unavailable",
                "dataset_checksum": manifest["dataset_checksum"],
                "split_leakage_count": manifest["leakage_count"],
                "smoke_only": bool(args.smoke),
                "not_a_scientific_result": bool(args.smoke),
            },
        )
        print("greedy imitation encoder comparison skipped: torch is not installed")
        return 2

    split_samples = frozen_data.samples_by_split(samples)
    sample_ids = frozen_data.sample_ids_by_split(samples)
    rows: list[dict[str, Any]] = []
    for training_seed in args.training_seeds:
        for encoder in args.encoders:
            variant_dir = run_dir / "variants" / str(encoder) / f"seed_{int(training_seed)}"
            row = _run_variant(
                args=args,
                encoder=str(encoder),
                training_seed=int(training_seed),
                variant_dir=variant_dir,
                split_samples=split_samples,
                manifest=manifest,
                common_sample_ids=sample_ids,
            )
            rows.append(row)
    summary = _build_comparison_summary(
        args=args,
        run_dir=run_dir,
        manifest=manifest,
        rows=rows,
        sample_ids=sample_ids,
    )
    frozen_data.write_json(run_dir / "comparison_summary.json", summary)
    _write_jsonl(run_dir / "comparison_rows.jsonl", rows)
    return 0


def _run_variant(
    *,
    args: argparse.Namespace,
    encoder: str,
    training_seed: int,
    variant_dir: Path,
    split_samples: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    common_sample_ids: dict[str, list[str]],
) -> dict[str, Any]:
    import torch
    from marl_models.hgnn import count_trainable_parameters

    variant_dir.mkdir(parents=True, exist_ok=False)
    module_args = argparse.Namespace(
        seed=int(training_seed),
        task_encoder=str(encoder),
        task_feature_dim=int(manifest["task_feature_dimension"]),
        task_embedding_dim=int(args.task_embedding_dim),
        hidden_dim=int(args.hidden_dim),
        lr=float(args.lr),
        gradient_batch_decisions=int(args.gradient_batch_decisions),
        max_grad_norm=float(args.max_grad_norm),
        completed_dag_weight=float(args.completed_dag_weight),
        device=str(args.device),
    )
    scorer_seed = _scorer_seed(int(training_seed))
    modules = gate._build_modules(
        module_args,
        encoder_seed=int(training_seed),
        scorer_seed=scorer_seed,
    )
    scorer_initial_hash = _state_dict_hash(modules.offloading_actor.scorer.state_dict())
    trainable_parameter_count = (
        count_trainable_parameters(modules.task_encoder)
        + count_trainable_parameters(modules.offloading_actor.scorer)
    )
    sample_id_hashes = {
        split: _string_list_hash(common_sample_ids[split])
        for split in ("train", "val", "test")
    }
    config_payload = {
        "schema": "greedy_imitation_encoder_variant_v1",
        "encoder": encoder,
        "training_seed": int(training_seed),
        "encoder_initialization_seed": int(training_seed),
        "candidate_scorer_seed": scorer_seed,
        "candidate_scorer_initial_state_hash": scorer_initial_hash,
        "dataset_checksum": manifest["dataset_checksum"],
        "sample_id_hashes": sample_id_hashes,
        "hyperparameters": _common_hyperparameters(args),
        "trainable_parameter_count": int(trainable_parameter_count),
    }
    frozen_data.write_json(variant_dir / "config.json", config_payload)

    start_time = time.perf_counter()
    train_rows: list[dict[str, Any]] = []
    all_gradient_norms: list[float] = []
    batch_plan_hashes: list[str] = []
    total_encoder_forward_count = 0
    total_scorer_forward_count = 0
    if modules.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(modules.device)
    for epoch in range(int(args.supervised_epochs)):
        shuffle_seed = _epoch_shuffle_seed(int(training_seed), int(epoch))
        plan = grouped_batch.build_graph_aware_batch_plan(
            split_samples["train"],
            target_batch_decisions=int(args.gradient_batch_decisions),
            shuffle_seed=int(shuffle_seed),
            shuffle_graph_groups=True,
        )
        batch_plan_hashes.append(plan.batch_plan_hash)
        metrics = gate._make_metric_tree()
        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_decision_count = 0
        epoch_optimizer_steps = 0
        epoch_gradient_norms: list[float] = []
        epoch_encoder_forward_count = 0
        epoch_scorer_forward_count = 0
        for batch in plan.batches:
            batch_samples = grouped_batch.samples_for_batch(
                split_samples["train"],
                batch,
            )
            forward, results = gate._score_grouped_batch(
                modules,
                batch_samples,
                train_enabled=True,
            )
            gradient_norm = gate._apply_grouped_optimizer_step(modules, forward)
            if gradient_norm is not None:
                epoch_gradient_norms.append(float(gradient_norm))
                epoch_optimizer_steps += 1
            epoch_encoder_forward_count += int(forward.encoder_forward_count)
            epoch_scorer_forward_count += int(forward.scorer_forward_count)
            epoch_loss_sum += float(
                forward.per_decision_losses.detach().sum().cpu().item()
            )
            epoch_decision_count += len(results)
            for sample, result in zip(batch_samples, results):
                gate._update_metric_tree(metrics, "train", sample, result)
        epoch_wall_time = float(time.perf_counter() - epoch_start)
        if epoch_decision_count != plan.decision_count:
            raise AssertionError("grouped epoch decision count drifted from batch plan")
        if epoch_encoder_forward_count != plan.unique_graph_count:
            raise AssertionError(
                "grouped epoch encoder forward count differs from unique graph count"
            )
        _merge_manifest_skips(metrics, "train", manifest)
        all_gradient_norms.extend(epoch_gradient_norms)
        total_encoder_forward_count += epoch_encoder_forward_count
        total_scorer_forward_count += epoch_scorer_forward_count
        train_summary = gate._summarize_metric_tree(metrics)
        row = {
            "epoch": int(epoch),
            "shuffle_seed": int(shuffle_seed),
            "batch_plan_hash": plan.batch_plan_hash,
            "shuffle_order_hash": _string_list_hash(list(plan.sample_ids)),
            "sample_id_hash": sample_id_hashes["train"],
            "loss_mean": (
                float(epoch_loss_sum / epoch_decision_count)
                if epoch_decision_count
                else None
            ),
            "optimizer_steps": int(epoch_optimizer_steps),
            "finite_gradient_norm": all(
                math.isfinite(value) for value in epoch_gradient_norms
            ),
            "gradient_norm_mean": _mean_or_none(epoch_gradient_norms),
            "gradient_norm_max": (
                max(epoch_gradient_norms) if epoch_gradient_norms else None
            ),
            "decision_count": int(plan.decision_count),
            "unique_graph_count": int(plan.unique_graph_count),
            "encoder_forward_count": int(epoch_encoder_forward_count),
            "scorer_forward_count": int(epoch_scorer_forward_count),
            "decisions_per_graph": plan.decisions_per_graph,
            "optimizer_batch_count": len(plan.batches),
            "epoch_wall_time_seconds": epoch_wall_time,
            "peak_cpu_rss_bytes": _peak_cpu_rss_bytes(),
            "peak_gpu_memory_allocated_bytes": _peak_gpu_memory_allocated_bytes(
                torch,
                modules.device,
            ),
            "peak_gpu_memory_reserved_bytes": _peak_gpu_memory_reserved_bytes(
                torch,
                modules.device,
            ),
            "metrics": train_summary,
        }
        train_rows.append(row)

    val_metrics, val_grouped = _evaluate_split(
        modules,
        split_samples["val"],
        "val",
        manifest,
        target_batch_decisions=int(args.gradient_batch_decisions),
    )
    test_metrics, test_grouped = _evaluate_split(
        modules,
        split_samples["test"],
        "test",
        manifest,
        target_batch_decisions=int(args.gradient_batch_decisions),
    )
    typed_diagnostics = _typed_diagnostics(
        modules=modules,
        train_samples=split_samples["train"],
        evaluation_samples=split_samples["test"] or split_samples["val"] or split_samples["train"],
        target_batch_decisions=int(args.gradient_batch_decisions),
    )
    duration_seconds = float(time.perf_counter() - start_time)
    closed_loop = None
    if not bool(args.skip_closed_loop_eval):
        closed_loop_args = argparse.Namespace(
            seed=int(training_seed),
            task_encoder=str(encoder),
            completed_dag_weight=float(args.completed_dag_weight),
            max_steps_per_episode=int(args.max_steps_per_episode),
            closed_loop_eval_episodes=int(args.closed_loop_eval_episodes),
            closed_loop_eval_seed=(
                int(args.closed_loop_eval_seed)
                if args.closed_loop_eval_seed is not None
                else int(args.dataset_seed) + 1_000_000
            ),
        )
        closed_loop = gate._run_closed_loop_eval(
            args=closed_loop_args,
            run_dir=variant_dir,
            modules=modules,
        )

    checkpoint_path = variant_dir / "checkpoint.pt"
    torch.save(
        {
            "schema": "greedy_imitation_encoder_comparison_checkpoint_v1",
            "encoder": encoder,
            "training_seed": int(training_seed),
            "dataset_checksum": manifest["dataset_checksum"],
            "task_encoder_state_dict": modules.task_encoder.state_dict(),
            "candidate_scorer_state_dict": modules.offloading_actor.scorer.state_dict(),
        },
        checkpoint_path,
    )
    _write_jsonl(variant_dir / "train_metrics.jsonl", train_rows)
    metrics_payload = {
        "schema": "greedy_imitation_fixed_dataset_metrics_v1",
        "dataset_checksum": manifest["dataset_checksum"],
        "sample_id_hashes": sample_id_hashes,
        "batch_plan_hashes": batch_plan_hashes,
        "val": val_metrics,
        "test": test_metrics,
        "grouped_pipeline": {
            "train_encoder_forward_count": int(total_encoder_forward_count),
            "train_scorer_forward_count": int(total_scorer_forward_count),
            "val": val_grouped,
            "test": test_grouped,
            "peak_cpu_rss_bytes": _peak_cpu_rss_bytes(),
            "peak_gpu_memory_allocated_bytes": _peak_gpu_memory_allocated_bytes(
                torch,
                modules.device,
            ),
            "peak_gpu_memory_reserved_bytes": _peak_gpu_memory_reserved_bytes(
                torch,
                modules.device,
            ),
        },
        "typed_diagnostics": typed_diagnostics,
        "closed_loop_eval": closed_loop,
        "closed_loop_is_secondary_diagnostic": True,
        "fixed_test_dataset_is_primary": True,
        "finite_gradient_norm": all(math.isfinite(value) for value in all_gradient_norms),
        "training_duration_seconds": duration_seconds,
        "trainable_parameter_count": int(trainable_parameter_count),
    }
    frozen_data.write_json(variant_dir / "test_metrics.json", metrics_payload)

    test_overall = test_metrics.get("test/overall", {})
    technical_pass = bool(
        int(manifest["leakage_count"]) == 0
        and test_overall.get("finite_loss", True)
        and test_overall.get("finite_logits", True)
        and all(math.isfinite(value) for value in all_gradient_norms)
        and all(
            int(row["encoder_forward_count"]) == int(row["unique_graph_count"])
            for row in train_rows
        )
    )
    return {
        "encoder": encoder,
        "training_seed": int(training_seed),
        "variant_dir": str(variant_dir),
        "dataset_checksum": manifest["dataset_checksum"],
        "sample_id_hashes": sample_id_hashes,
        "candidate_scorer_initial_state_hash": scorer_initial_hash,
        "batch_plan_hashes": batch_plan_hashes,
        "shuffle_order_hashes": [
            str(row["shuffle_order_hash"]) for row in train_rows
        ],
        "trainable_parameter_count": int(trainable_parameter_count),
        "training_duration_seconds": duration_seconds,
        "finite_gradient_norm": all(math.isfinite(value) for value in all_gradient_norms),
        "typed_diagnostics": typed_diagnostics,
        "test_metrics": test_metrics,
        "test_overall": test_overall,
        "technical_pass": technical_pass,
    }


def _evaluate_split(
    modules: gate.GateModules,
    samples: list[dict[str, Any]],
    split: str,
    manifest: dict[str, Any],
    *,
    target_batch_decisions: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = gate._make_metric_tree()
    plan = grouped_batch.build_graph_aware_batch_plan(
        samples,
        target_batch_decisions=int(target_batch_decisions),
        shuffle_seed=_evaluation_plan_seed(split),
        shuffle_graph_groups=False,
    )
    encoder_forward_count = 0
    scorer_forward_count = 0
    start_time = time.perf_counter()
    for batch in plan.batches:
        batch_samples = grouped_batch.samples_for_batch(samples, batch)
        forward, results = gate._score_grouped_batch(
            modules,
            batch_samples,
            train_enabled=False,
        )
        encoder_forward_count += int(forward.encoder_forward_count)
        scorer_forward_count += int(forward.scorer_forward_count)
        for sample, result in zip(batch_samples, results):
            gate._update_metric_tree(metrics, split, sample, result)
    wall_time = float(time.perf_counter() - start_time)
    if encoder_forward_count != plan.unique_graph_count:
        raise AssertionError(
            f"{split} grouped encoder forward count differs from unique graph count"
        )
    _merge_manifest_skips(metrics, split, manifest)
    return gate._summarize_metric_tree(metrics), {
        "batch_plan_hash": plan.batch_plan_hash,
        "decision_count": int(plan.decision_count),
        "unique_graph_count": int(plan.unique_graph_count),
        "encoder_forward_count": int(encoder_forward_count),
        "scorer_forward_count": int(scorer_forward_count),
        "decisions_per_graph": plan.decisions_per_graph,
        "optimizer_batch_count": len(plan.batches),
        "wall_time_seconds": wall_time,
    }


def _merge_manifest_skips(
    metrics: dict[str, gate.MetricAccumulator],
    split: str,
    manifest: dict[str, Any],
) -> None:
    skipped = manifest.get("skipped_no_candidate", {}).get(split, {})
    by_policy = skipped.get("by_trajectory_policy", {})
    for policy, count in by_policy.items():
        gate._merge_skip(metrics, split, str(policy), int(count))


def _typed_diagnostics(
    *,
    modules: gate.GateModules,
    train_samples: list[dict[str, Any]],
    evaluation_samples: list[dict[str, Any]],
    target_batch_decisions: int,
) -> dict[str, Any] | None:
    layers = [
        layer
        for layer in getattr(modules.task_encoder, "layers", [])
        if hasattr(layer, "raw_type_weights") and hasattr(layer, "gate_network")
    ]
    if not layers:
        return None

    gradient_rows: list[dict[str, Any]] = []
    modules.optimizer.zero_grad(set_to_none=True)
    if train_samples:
        result = gate._score_sample(modules, train_samples[0], train_enabled=True)
        result["loss"].backward()
    for index, layer in enumerate(layers):
        grad = layer.raw_type_weights.grad
        gradient_rows.append(
            {
                "layer": int(index),
                "raw_type_weights_gradient_norm": (
                    float(grad.detach().norm().cpu().item()) if grad is not None else 0.0
                ),
                "raw_type_weights_gradient_finite": (
                    bool(modules.torch.isfinite(grad).all().item()) if grad is not None else True
                ),
            }
        )
    modules.optimizer.zero_grad(set_to_none=True)

    gate_values: dict[int, list[float]] = defaultdict(list)
    capture_multiplier = {"value": 1}
    handles = []
    for index, layer in enumerate(layers):
        def capture(_module: Any, _inputs: Any, output: Any, *, layer_index: int = index) -> None:
            values = modules.torch.sigmoid(output.detach()).reshape(-1).cpu().tolist()
            for _ in range(int(capture_multiplier["value"])):
                gate_values[layer_index].extend(float(item) for item in values)

        handles.append(layer.gate_network.register_forward_hook(capture))
    try:
        plan = grouped_batch.build_graph_aware_batch_plan(
            evaluation_samples,
            target_batch_decisions=int(target_batch_decisions),
            shuffle_seed=_evaluation_plan_seed("typed_diagnostics"),
            shuffle_graph_groups=False,
        )
        for batch in plan.batches:
            for group in batch.groups:
                capture_multiplier["value"] = int(group.decision_count)
                graph_samples = [
                    evaluation_samples[index] for index in group.sample_indices
                ]
                with modules.torch.no_grad():
                    gate._score_grouped_batch(
                        modules,
                        graph_samples,
                        train_enabled=False,
                    )
    finally:
        for handle in handles:
            handle.remove()

    layer_rows: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        weights = layer.normalized_type_weights().detach().cpu().tolist()
        values = gate_values[index]
        layer_rows.append(
            {
                "layer": int(index),
                "normalized_type_weights": {
                    name: float(weights[type_index])
                    for type_index, name in enumerate(TYPE_NAMES)
                },
                "normalized_type_weight_mean": float(statistics.fmean(weights)),
                "residual_gate_mean": _mean_or_none(values),
                "residual_gate_std": _std_or_none(values),
                "residual_gate_min": min(values) if values else None,
                "residual_gate_max": max(values) if values else None,
                "residual_gate_saturation_rate": (
                    float(sum(value <= 0.01 or value >= 0.99 for value in values) / len(values))
                    if values
                    else None
                ),
            }
        )
    return {
        "type_names": list(TYPE_NAMES),
        "layers": layer_rows,
        "gradient": gradient_rows,
    }


def _build_comparison_summary(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    sample_ids: dict[str, list[str]],
) -> dict[str, Any]:
    scorer_hashes_by_seed: dict[int, set[str]] = defaultdict(set)
    batch_plan_hashes_by_seed: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        training_seed = int(row["training_seed"])
        scorer_hashes_by_seed[training_seed].add(
            str(row["candidate_scorer_initial_state_hash"])
        )
        batch_plan_hashes_by_seed[training_seed].add(
            tuple(str(value) for value in row.get("batch_plan_hashes", []))
        )
    scorer_hash_match = all(len(values) == 1 for values in scorer_hashes_by_seed.values())
    batch_plan_hash_match = all(
        len(values) == 1 for values in batch_plan_hashes_by_seed.values()
    )
    common_id_hashes = {
        split: _string_list_hash(sample_ids[split])
        for split in ("train", "val", "test")
    }
    aggregate: dict[str, Any] = {}
    for encoder in args.encoders:
        encoder_rows = [row for row in rows if row["encoder"] == encoder]
        aggregate[str(encoder)] = _aggregate_encoder_rows(encoder_rows)
    representation_ranking = sorted(
        (
            {
                "encoder": encoder,
                "test_top1_accuracy_mean": aggregate[encoder]["test_top1_accuracy_mean"],
                "test_mean_eft_regret_mean": aggregate[encoder]["test_mean_eft_regret_mean"],
            }
            for encoder in aggregate
        ),
        key=lambda item: (
            -_finite_or_default(item["test_top1_accuracy_mean"], -1.0),
            _finite_or_default(item["test_mean_eft_regret_mean"], float("inf")),
            item["encoder"],
        ),
    )
    technical_pass = bool(
        int(manifest["leakage_count"]) == 0
        and scorer_hash_match
        and batch_plan_hash_match
        and all(row["dataset_checksum"] == manifest["dataset_checksum"] for row in rows)
        and all(row["sample_id_hashes"] == common_id_hashes for row in rows)
        and all(bool(row["technical_pass"]) for row in rows)
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "completed",
        "run_dir": str(run_dir),
        "git_commit": gate._git_commit(),
        "smoke_only": bool(args.smoke),
        "not_a_scientific_result": bool(args.smoke),
        "dataset_checksum": manifest["dataset_checksum"],
        "dataset_manifest_path": str(Path(manifest["dataset_file_path"]).parent / frozen_data.MANIFEST_FILENAME),
        "split_leakage_count": int(manifest["leakage_count"]),
        "common_sample_id_hashes": common_id_hashes,
        "common_hyperparameters": _common_hyperparameters(args),
        "actual_encoders": list(args.encoders),
        "training_seeds": [int(item) for item in args.training_seeds],
        "candidate_scorer_initial_hash_match": scorer_hash_match,
        "candidate_scorer_initial_hashes_by_seed": {
            str(seed): sorted(values)
            for seed, values in sorted(scorer_hashes_by_seed.items())
        },
        "batch_plan_hash_match_across_encoders": batch_plan_hash_match,
        "batch_plan_hashes_by_seed": {
            str(seed): [list(values) for values in sorted(hash_sets)]
            for seed, hash_sets in sorted(batch_plan_hashes_by_seed.items())
        },
        "checksum_validation": {
            **dict(manifest.get("_runtime_validation", {})),
            "post_training_dataset_scan_performed": False,
            "post_training_checksum_reused": str(manifest["dataset_checksum"]),
        },
        "variant_runs": rows,
        "encoder_aggregate": aggregate,
        "representation_ranking": representation_ranking,
        "ranking_is_descriptive_not_predeclared": True,
        "fixed_test_dataset_is_primary": True,
        "closed_loop_is_secondary_diagnostic": True,
        "technical_pass": technical_pass,
    }


def _aggregate_encoder_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top1 = [_metric(row, "top1_accuracy") for row in rows]
    regret = [_metric(row, "mean_eft_regret") for row in rows]
    durations = [float(row["training_duration_seconds"]) for row in rows]
    parameter_counts = sorted({int(row["trainable_parameter_count"]) for row in rows})
    return {
        "run_count": len(rows),
        "test_top1_accuracy_mean": _mean_or_none(top1),
        "test_top1_accuracy_std": _std_or_none(top1),
        "test_mean_eft_regret_mean": _mean_or_none(regret),
        "test_mean_eft_regret_std": _std_or_none(regret),
        "training_duration_seconds_mean": _mean_or_none(durations),
        "training_duration_seconds_std": _std_or_none(durations),
        "trainable_parameter_counts": parameter_counts,
        "all_technical_pass": all(bool(row["technical_pass"]) for row in rows),
    }


def _metric(row: dict[str, Any], key: str) -> float:
    value = row.get("test_overall", {}).get(key)
    return float(value) if value is not None else float("nan")


def _common_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task_embedding_dim": int(args.task_embedding_dim),
        "hidden_dim": int(args.hidden_dim),
        "optimizer": "Adam",
        "learning_rate": float(args.lr),
        "gradient_batch_decisions": int(args.gradient_batch_decisions),
        "supervised_epochs": int(args.supervised_epochs),
        "max_grad_norm": float(args.max_grad_norm),
        "epoch_shuffle_scheme": "sha256(training_seed, epoch), graph-group shuffle",
        "batch_plan": "indivisible graph groups packed by target decision count",
        "early_stopping": None,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if bool(args.generate_dataset) == bool(args.reuse_dataset):
        raise ValueError("select exactly one of --generate-dataset or --reuse-dataset")
    if bool(args.reuse_dataset) and args.dataset_dir is None:
        raise ValueError("--reuse-dataset requires --dataset-dir")
    if int(args.episodes) <= 0 or int(args.max_steps_per_episode) <= 0:
        raise ValueError("episodes and max-steps-per-episode must be positive")
    if int(args.supervised_epochs) <= 0 or int(args.gradient_batch_decisions) <= 0:
        raise ValueError("supervised epochs and gradient batch decisions must be positive")
    if float(args.train_fraction) < 0.0 or float(args.val_fraction) < 0.0:
        raise ValueError("split fractions must be non-negative")
    if float(args.train_fraction) + float(args.val_fraction) > 1.0:
        raise ValueError("train-fraction + val-fraction must be <= 1")
    if len(set(args.encoders)) != len(args.encoders):
        raise ValueError("encoders must not contain duplicates")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError("training-seeds must not contain duplicates")


def _apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not bool(args.smoke):
        return args
    args.episodes = min(int(args.episodes), 6)
    args.max_steps_per_episode = min(int(args.max_steps_per_episode), 8)
    args.supervised_epochs = min(int(args.supervised_epochs), 1)
    args.gradient_batch_decisions = min(int(args.gradient_batch_decisions), 4)
    args.training_seeds = [int(args.training_seeds[0])]
    args.train_fraction = 0.5
    args.val_fraction = 1.0 / 6.0
    args.skip_closed_loop_eval = True
    if args.dag_base_arrival_prob is None:
        args.dag_base_arrival_prob = 0.1
    if not bool(args.generate_dataset) and not bool(args.reuse_dataset):
        args.generate_dataset = True
    return args


def _resolve_dataset_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.dataset_dir is not None:
        return Path(args.dataset_dir)
    return run_dir / "dataset"


def _create_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(args.run_name)
    ).strip("_")
    run_dir = Path(args.output_dir) / (
        f"{timestamp}_{safe_name or 'encoder_comparison'}_dataset_seed{int(args.dataset_seed)}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _state_dict_hash(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _string_list_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _scorer_seed(training_seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"candidate-scorer:{int(training_seed)}".encode("utf-8")).digest()[:4],
        byteorder="big",
        signed=False,
    )


def _epoch_shuffle_seed(training_seed: int, epoch: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"shuffle:{int(training_seed)}:{int(epoch)}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )


def _evaluation_plan_seed(split: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"grouped-evaluation:{str(split)}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )


def _peak_cpu_rss_bytes() -> int | None:
    try:
        import resource
    except ModuleNotFoundError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


def _peak_gpu_memory_allocated_bytes(torch: Any, device: Any) -> int | None:
    if getattr(device, "type", None) != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _peak_gpu_memory_reserved_bytes(torch: Any, device: Any) -> int | None:
    if getattr(device, "type", None) != "cuda":
        return None
    return int(torch.cuda.max_memory_reserved(device))


def _mean_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else None


def _std_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(statistics.pstdev(finite))


def _finite_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(frozen_data.canonical_json(row) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
