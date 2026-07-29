from __future__ import annotations

import argparse
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

from scripts import greedy_imitation_dataset as frozen_data
from scripts import greedy_imitation_grouped_batch as grouped_batch
from scripts import train_greedy_imitation_gate as gate
from scripts.run_greedy_imitation_encoder_comparison import (
    _epoch_shuffle_seed,
    _scorer_seed,
)


SCHEMA = "greedy_eft_contextual_bandit_gate_v1"
CHECKPOINT_SCHEMA = "greedy_eft_contextual_bandit_checkpoint_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic per-decision contextual-bandit policy-gradient gate. "
            "It uses frozen historical contexts and never uses PPO, GAE, a "
            "critic, movement training, or environment reward."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[42, 86, 1042])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--task-embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gradient-batch-decisions", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample-limit-per-split", type=int, default=-1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs") / "contextual_bandit_gate",
    )
    parser.add_argument("--run-name", type=str, default="contextual_bandit_gate")
    parser.add_argument("--smoke", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.smoke:
        args.epochs = 1
        args.training_seeds = [int(args.training_seeds[0])]
        args.sample_limit_per_split = 64
        args.task_embedding_dim = min(int(args.task_embedding_dim), 16)
        args.hidden_dim = min(int(args.hidden_dim), 32)
        args.device = "cpu"
    _validate_args(args)

    samples, manifest = frozen_data.load_frozen_dataset(Path(args.dataset_dir))
    if int(manifest["leakage_count"]) != 0:
        raise ValueError("frozen dataset split leakage must be zero")
    split_samples = frozen_data.samples_by_split(samples)
    split_samples = {
        split: _limit_samples(rows, int(args.sample_limit_per_split))
        for split, rows in split_samples.items()
    }
    scale_info = compute_train_regret_scale(split_samples["train"])
    run_dir = _create_run_dir(args)
    frozen_data.write_json(
        run_dir / "config.json",
        {
            "schema": SCHEMA,
            "git_commit": gate._git_commit(),
            "dataset_dir": str(Path(args.dataset_dir).resolve()),
            "dataset_checksum": manifest["dataset_checksum"],
            "encoder": "mlp",
            "movement": "fixed_not_trained",
            "epochs": int(args.epochs),
            "training_seeds": [int(seed) for seed in args.training_seeds],
            "task_embedding_dim": int(args.task_embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "learning_rate": float(args.lr),
            "gradient_batch_decisions": int(args.gradient_batch_decisions),
            "max_grad_norm": float(args.max_grad_norm),
            "device": str(args.device),
            "sample_limit_per_split": int(args.sample_limit_per_split),
            "split_sample_counts": {
                split: len(rows) for split, rows in split_samples.items()
            },
            "regret_scale": scale_info,
            "uses_ppo": False,
            "uses_gae": False,
            "uses_critic": False,
            "uses_environment_reward": False,
        },
    )

    rows: list[dict[str, Any]] = []
    for seed in args.training_seeds:
        rows.append(
            _run_seed(
                args=args,
                training_seed=int(seed),
                split_samples=split_samples,
                manifest=manifest,
                scale_info=scale_info,
                seed_dir=run_dir / f"seed_{int(seed)}",
            )
        )
        _write_jsonl(run_dir / "seed_rows.jsonl", rows)
    summary = _build_summary(
        args=args,
        manifest=manifest,
        scale_info=scale_info,
        rows=rows,
    )
    frozen_data.write_json(run_dir / "summary.json", summary)
    return 0


def compute_train_regret_scale(train_samples: list[dict[str, Any]]) -> dict[str, Any]:
    squared_sum = 0.0
    value_count = 0
    positive_count = 0
    sample_ids: list[str] = []
    for sample in train_samples:
        sample_ids.append(str(sample["sample_id"]))
        mask = [bool(value) for value in sample["candidate_mask"]]
        eft = [float(value) for value in sample["estimated_finish_times"]]
        legal = [index for index, allowed in enumerate(mask) if allowed]
        if not legal:
            raise ValueError(f"train sample has no legal candidate: {sample['sample_id']}")
        if not all(math.isfinite(eft[index]) for index in legal):
            raise FloatingPointError(
                f"train sample has non-finite legal candidate EFT: {sample['sample_id']}"
            )
        best = min(eft[index] for index in legal)
        for index in legal:
            regret = max(float(eft[index] - best), 0.0)
            squared_sum += regret * regret
            value_count += 1
            positive_count += int(regret > 0.0)
    if value_count <= 0:
        raise ValueError("train split has no legal candidate EFT values")
    rms = math.sqrt(squared_sum / float(value_count))
    scale = float(rms if rms > 1e-12 else 1.0)
    return {
        "method": "train_legal_candidate_raw_eft_regret_rms",
        "value": scale,
        "legal_candidate_value_count": int(value_count),
        "positive_regret_count": int(positive_count),
        "train_sample_count": len(train_samples),
        "train_sample_id_hash": _string_list_hash(sample_ids),
    }


def contextual_bandit_objective(
    modules: gate.GateModules,
    samples: list[dict[str, Any]],
    *,
    regret_scale: float,
) -> tuple[Any, dict[str, Any]]:
    torch = modules.torch
    forward = grouped_batch.grouped_candidate_forward(
        modules,
        samples,
        train_enabled=True,
        compute_supervised_loss=False,
    )
    logits = forward.masked_logits
    if logits is None:
        raise ValueError("contextual-bandit batch cannot be empty")
    mask_rows: list[Any] = []
    reward_rows: list[Any] = []
    raw_regret_rows: list[Any] = []
    max_candidates = int(logits.shape[1])
    for sample in samples:
        mask = torch.as_tensor(
            sample["candidate_mask"], dtype=torch.bool, device=modules.device
        )
        eft = torch.as_tensor(
            sample["estimated_finish_times"], dtype=torch.float32, device=modules.device
        ).detach()
        legal_eft = eft[mask]
        if legal_eft.numel() <= 0:
            raise ValueError(f"sample has no legal candidate: {sample['sample_id']}")
        if not bool(torch.isfinite(legal_eft).all().item()):
            raise FloatingPointError(
                f"sample has non-finite legal candidate EFT: {sample['sample_id']}"
            )
        raw_regret = torch.zeros_like(eft)
        raw_regret[mask] = (legal_eft - legal_eft.min()).clamp_min(0.0)
        raw_regret = raw_regret.detach()
        rewards = (-raw_regret / float(regret_scale)).detach()
        pad = max_candidates - int(mask.numel())
        mask_rows.append(modules.functional.pad(mask, (0, pad), value=False))
        reward_rows.append(modules.functional.pad(rewards, (0, pad), value=0.0))
        raw_regret_rows.append(modules.functional.pad(raw_regret, (0, pad), value=0.0))
    mask_matrix = torch.stack(mask_rows)
    reward_matrix = torch.stack(reward_rows).detach()
    raw_regret_matrix = torch.stack(raw_regret_rows).detach()
    distribution = torch.distributions.Categorical(logits=logits)
    actions = distribution.sample()
    log_probs = distribution.log_prob(actions)
    selected_legal = mask_matrix.gather(1, actions.unsqueeze(1)).squeeze(1)
    invalid_count = int((~selected_legal).sum().detach().cpu().item())
    if invalid_count:
        raise AssertionError("masked contextual-bandit policy sampled an illegal action")
    selected_rewards = reward_matrix.gather(1, actions.unsqueeze(1)).squeeze(1)
    selected_raw_regrets = raw_regret_matrix.gather(1, actions.unsqueeze(1)).squeeze(1)
    exact_baseline = (distribution.probs * reward_matrix).sum(dim=1)
    advantage = (selected_rewards - exact_baseline.detach()).detach()
    per_decision_loss = -(advantage * log_probs)
    loss = per_decision_loss.mean()
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("contextual-bandit logits are non-finite")
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("contextual-bandit loss is non-finite")
    labels = torch.as_tensor(
        [int(sample["greedy_label_idx"]) for sample in samples],
        dtype=torch.long,
        device=modules.device,
    )
    diagnostics = {
        "actions": actions,
        "log_probs": log_probs,
        "selected_rewards": selected_rewards,
        "selected_raw_regrets": selected_raw_regrets,
        "exact_baseline": exact_baseline,
        "advantage": advantage,
        "entropy": distribution.entropy(),
        "agreement": actions.eq(labels),
        "invalid_action_count": invalid_count,
        "mask_matrix": mask_matrix,
        "reward_matrix": reward_matrix,
        "raw_regret_matrix": raw_regret_matrix,
        "forward": forward,
    }
    return loss, diagnostics


def _run_seed(
    *,
    args: argparse.Namespace,
    training_seed: int,
    split_samples: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    scale_info: dict[str, Any],
    seed_dir: Path,
) -> dict[str, Any]:
    import torch

    seed_dir.mkdir(parents=True, exist_ok=False)
    module_args = argparse.Namespace(
        seed=int(training_seed),
        task_encoder="mlp",
        task_feature_dim=int(manifest["task_feature_dimension"]),
        task_embedding_dim=int(args.task_embedding_dim),
        hidden_dim=int(args.hidden_dim),
        lr=float(args.lr),
        gradient_batch_decisions=int(args.gradient_batch_decisions),
        max_grad_norm=float(args.max_grad_norm),
        completed_dag_weight=16.0,
        device=str(args.device),
    )
    modules = gate._build_modules(
        module_args,
        encoder_seed=int(training_seed),
        scorer_seed=_scorer_seed(int(training_seed)),
    )
    _save_checkpoint(
        seed_dir / "initial_checkpoint.pt",
        modules=modules,
        training_seed=training_seed,
        manifest=manifest,
        args=args,
        scale_info=scale_info,
        stage="initial",
    )
    initial_val = evaluate_deterministic(
        modules,
        split_samples["val"],
        regret_scale=float(scale_info["value"]),
        target_batch_decisions=int(args.gradient_batch_decisions),
    )
    initial_test = evaluate_deterministic(
        modules,
        split_samples["test"],
        regret_scale=float(scale_info["value"]),
        target_batch_decisions=int(args.gradient_batch_decisions),
    )
    train_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for epoch in range(int(args.epochs)):
        plan = grouped_batch.build_graph_aware_batch_plan(
            split_samples["train"],
            target_batch_decisions=int(args.gradient_batch_decisions),
            shuffle_seed=_epoch_shuffle_seed(int(training_seed), int(epoch)),
            shuffle_graph_groups=True,
        )
        sampled_regrets: list[float] = []
        agreements: list[float] = []
        entropies: list[float] = []
        gradient_norms: list[float] = []
        action_tokens: list[str] = []
        invalid_count = 0
        epoch_start = time.perf_counter()
        for batch in plan.batches:
            batch_samples = grouped_batch.samples_for_batch(split_samples["train"], batch)
            loss, diagnostics = contextual_bandit_objective(
                modules,
                batch_samples,
                regret_scale=float(scale_info["value"]),
            )
            modules.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = list(modules.task_encoder.parameters()) + list(
                modules.offloading_actor.scorer.parameters()
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                float(args.max_grad_norm),
            )
            gradient_value = float(gradient_norm.detach().cpu().item())
            if not math.isfinite(gradient_value):
                raise FloatingPointError("contextual-bandit gradient norm is non-finite")
            modules.optimizer.step()
            gradient_norms.append(gradient_value)
            sampled_regrets.extend(
                diagnostics["selected_raw_regrets"].detach().cpu().tolist()
            )
            agreements.extend(
                diagnostics["agreement"].float().detach().cpu().tolist()
            )
            entropies.extend(diagnostics["entropy"].detach().cpu().tolist())
            invalid_count += int(diagnostics["invalid_action_count"])
            action_tokens.extend(
                f"{sample['sample_id']}:{int(action)}"
                for sample, action in zip(
                    batch_samples,
                    diagnostics["actions"].detach().cpu().tolist(),
                )
            )
        val_metrics = evaluate_deterministic(
            modules,
            split_samples["val"],
            regret_scale=float(scale_info["value"]),
            target_batch_decisions=int(args.gradient_batch_decisions),
        )
        row = {
            "epoch": int(epoch + 1),
            "batch_plan_hash": plan.batch_plan_hash,
            "sampled_action_hash": _string_list_hash(action_tokens),
            "sample_count": len(sampled_regrets),
            "sampled_raw_eft_regret_mean": statistics.fmean(sampled_regrets),
            "sampled_raw_eft_regret_p95": _percentile(sampled_regrets, 95.0),
            "sampled_greedy_agreement": statistics.fmean(agreements),
            "entropy_mean": statistics.fmean(entropies),
            "gradient_norm_mean": statistics.fmean(gradient_norms),
            "gradient_norm_max": max(gradient_norms),
            "invalid_action_count": int(invalid_count),
            "finite": all(
                math.isfinite(value)
                for value in sampled_regrets + entropies + gradient_norms
            ),
            "wall_time_seconds": float(time.perf_counter() - epoch_start),
            "val": val_metrics,
        }
        train_rows.append(row)
        _write_jsonl(seed_dir / "train_metrics.jsonl", train_rows)
    final_test = evaluate_deterministic(
        modules,
        split_samples["test"],
        regret_scale=float(scale_info["value"]),
        target_batch_decisions=int(args.gradient_batch_decisions),
    )
    _save_checkpoint(
        seed_dir / "trained_checkpoint.pt",
        modules=modules,
        training_seed=training_seed,
        manifest=manifest,
        args=args,
        scale_info=scale_info,
        stage="trained",
    )
    row = {
        "schema": SCHEMA,
        "training_seed": int(training_seed),
        "dataset_checksum": manifest["dataset_checksum"],
        "regret_scale": scale_info,
        "initial_val": initial_val,
        "initial_test": initial_test,
        "final_val": train_rows[-1]["val"],
        "final_test": final_test,
        "train_epochs": train_rows,
        "duration_seconds": float(time.perf_counter() - start_time),
        "technical_pass": bool(
            all(item["finite"] for item in train_rows)
            and sum(item["invalid_action_count"] for item in train_rows) == 0
            and final_test["finite"]
        ),
        "seed_dir": str(seed_dir),
    }
    frozen_data.write_json(seed_dir / "result.json", row)
    return row


def evaluate_deterministic(
    modules: gate.GateModules,
    samples: list[dict[str, Any]],
    *,
    regret_scale: float,
    target_batch_decisions: int,
) -> dict[str, Any]:
    torch = modules.torch
    plan = grouped_batch.build_graph_aware_batch_plan(
        samples,
        target_batch_decisions=int(target_batch_decisions),
        shuffle_seed=0,
        shuffle_graph_groups=False,
    )
    regrets: list[float] = []
    agreements: list[float] = []
    nontrivial: list[float] = []
    margin5: list[float] = []
    margin20: list[float] = []
    entropies: list[float] = []
    masked_random_accuracies: list[float] = []
    masked_random_regrets: list[float] = []
    invalid_count = 0
    with torch.no_grad():
        for batch in plan.batches:
            batch_samples = grouped_batch.samples_for_batch(samples, batch)
            forward = grouped_batch.grouped_candidate_forward(
                modules,
                batch_samples,
                train_enabled=False,
                compute_supervised_loss=False,
            )
            for index, sample in enumerate(batch_samples):
                count = len(sample["candidate_mask"])
                logits = forward.masked_logits[index, :count]
                mask = torch.as_tensor(
                    sample["candidate_mask"], dtype=torch.bool, device=modules.device
                )
                action = int(torch.argmax(logits).cpu().item())
                invalid_count += int(not bool(mask[action].item()))
                eft = [float(value) for value in sample["estimated_finish_times"]]
                legal = [i for i, allowed in enumerate(sample["candidate_mask"]) if allowed]
                best = min(eft[i] for i in legal)
                regrets.append(float(eft[action] - best))
                masked_random_accuracies.append(1.0 / float(len(legal)))
                masked_random_regrets.append(
                    statistics.fmean(float(eft[i] - best) for i in legal)
                )
                correct = float(action == int(sample["greedy_label_idx"]))
                agreements.append(correct)
                if int(sample["valid_candidate_count"]) >= 2:
                    nontrivial.append(correct)
                margin = sample.get("greedy_margin")
                if margin is not None and float(margin) >= 5.0:
                    margin5.append(correct)
                if margin is not None and float(margin) >= 20.0:
                    margin20.append(correct)
                entropies.append(
                    float(torch.distributions.Categorical(logits=logits).entropy().cpu().item())
                )
    values = regrets + agreements + entropies
    return {
        "sample_count": len(samples),
        "nontrivial_k_ge_2_sample_count": len(nontrivial),
        "margin_ge_5s_sample_count": len(margin5),
        "margin_ge_20s_sample_count": len(margin20),
        "mean_raw_eft_regret": statistics.fmean(regrets) if regrets else None,
        "p95_raw_eft_regret": _percentile(regrets, 95.0),
        "normalized_mean_eft_regret": (
            statistics.fmean(regrets) / float(regret_scale) if regrets else None
        ),
        "greedy_agreement": statistics.fmean(agreements) if agreements else None,
        "masked_random_accuracy": (
            statistics.fmean(masked_random_accuracies)
            if masked_random_accuracies
            else None
        ),
        "masked_random_mean_raw_eft_regret": (
            statistics.fmean(masked_random_regrets) if masked_random_regrets else None
        ),
        "nontrivial_k_ge_2_top1": statistics.fmean(nontrivial) if nontrivial else None,
        "margin_ge_5s_top1": statistics.fmean(margin5) if margin5 else None,
        "margin_ge_20s_top1": statistics.fmean(margin20) if margin20 else None,
        "entropy_mean": statistics.fmean(entropies) if entropies else None,
        "invalid_action_count": int(invalid_count),
        "finite": bool(all(math.isfinite(value) for value in values)),
    }


def _save_checkpoint(
    path: Path,
    *,
    modules: gate.GateModules,
    training_seed: int,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    scale_info: dict[str, Any],
    stage: str,
) -> None:
    modules.torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": str(stage),
            "encoder": "mlp",
            "training_seed": int(training_seed),
            "dataset_checksum": manifest["dataset_checksum"],
            "task_feature_dim": int(manifest["task_feature_dimension"]),
            "task_embedding_dim": int(args.task_embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "regret_scale": scale_info,
            "task_encoder_state_dict": modules.task_encoder.state_dict(),
            "candidate_scorer_state_dict": modules.offloading_actor.scorer.state_dict(),
        },
        path,
    )


def _build_summary(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    scale_info: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_keys = (
        "mean_raw_eft_regret",
        "p95_raw_eft_regret",
        "greedy_agreement",
        "nontrivial_k_ge_2_top1",
        "margin_ge_5s_top1",
        "margin_ge_20s_top1",
        "entropy_mean",
        "masked_random_accuracy",
        "masked_random_mean_raw_eft_regret",
    )
    aggregate: dict[str, Any] = {}
    for key in metric_keys:
        values = [float(row["final_test"][key]) for row in rows]
        aggregate[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
    reductions = [
        1.0
        - float(row["final_test"]["mean_raw_eft_regret"])
        / max(float(row["initial_test"]["mean_raw_eft_regret"]), 1e-12)
        for row in rows
    ]
    return {
        "schema": SCHEMA,
        "status": "completed",
        "dataset_checksum": manifest["dataset_checksum"],
        "training_seeds": [int(seed) for seed in args.training_seeds],
        "epochs": int(args.epochs),
        "regret_scale": scale_info,
        "rows": rows,
        "aggregate_final_test": aggregate,
        "relative_mean_regret_reduction": {
            "mean": statistics.fmean(reductions),
            "std": statistics.pstdev(reductions),
            "values": reductions,
        },
        "technical_pass": all(row["technical_pass"] for row in rows),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not bool(args.smoke) and int(args.epochs) != 5:
        raise ValueError("formal contextual-bandit gate requires exactly 5 epochs")
    if int(args.gradient_batch_decisions) <= 0:
        raise ValueError("gradient batch decisions must be positive")
    if float(args.lr) <= 0.0:
        raise ValueError("learning rate must be positive")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError("training seeds must be unique")


def _limit_samples(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return list(samples if int(limit) < 0 else samples[: int(limit)])


def _create_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(args.run_name)
    ).strip("_")
    run_dir = Path(args.output_dir) / f"{timestamp}_{safe or 'contextual_bandit'}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _string_list_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(int(math.ceil(float(percentile) / 100.0 * len(ordered))) - 1, 0)
    return float(ordered[index])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
