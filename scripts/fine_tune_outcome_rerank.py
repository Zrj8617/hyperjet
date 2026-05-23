from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from analysis.analyze_outcome_rerank_candidates import classify_disagreement
from environment.env import Env
from environment.graph_builder import HeteroGraphSnapshot
from marl_models.hgnn.scheduler import PhaseOneGraphScheduler
from stream_analyze_decision_attribution import iter_json_array
from utils.progress import TerminalProgress


@dataclass(slots=True)
class OutcomePair:
    preferred_uav: int
    other_uav: int
    weight: float
    label: str
    delta_planned_finish: float


@dataclass(slots=True)
class SnapshotOutcomeSample:
    snapshot: HeteroGraphSnapshot
    pairs: list[tuple[str, OutcomePair]]


@dataclass(slots=True)
class FineTuneMetrics:
    epoch: int
    avg_loss: float
    avg_pair_loss: float
    avg_distill_loss: float
    pair_accuracy: float
    pair_count: int


def _parse_seed(path: Path) -> int:
    match = re.search(r"_seed(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_scheduler_state_compatible(scheduler: PhaseOneGraphScheduler, checkpoint_path: str, device: str) -> None:
    if not checkpoint_path:
        return
    state_dict = torch.load(checkpoint_path, map_location=device)
    model_state = scheduler.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    scheduler.load_state_dict(compatible_state, strict=False)


def _refresh_dimension_config() -> None:
    phase_one_obs = (
        config.ENABLE_DYNAMIC_DAG
        and config.ENABLE_PHASE_ONE_EXECUTION
        and not config.ENABLE_LEGACY_REQUEST_PIPELINE
        and config.USE_PHASE_ONE_DEDICATED_OBS
    )
    compact_obs = phase_one_obs and config.USE_MAPPO_COMPACT_OBS
    config.MAX_UAV_NEIGHBORS = max(config.NUM_UAVS - 1, 1)
    config.MAX_ASSOCIATED_UES = min(30, config.NUM_UES // max(config.NUM_UAVS, 1) + 10)
    config.SELF_OBS_DIM = config.PHASE_ONE_SELF_OBS_DIM if phase_one_obs else config.LEGACY_SELF_OBS_DIM
    config.UE_OBS_DIM = (
        config.MAPPO_COMPACT_LOCAL_OBS_DIM
        if compact_obs
        else config.PHASE_ONE_TASK_OBS_DIM
        if phase_one_obs
        else config.LEGACY_UE_OBS_DIM
    )
    config.NEIGHBOR_OBS_DIM = config.PHASE_ONE_NEIGHBOR_OBS_DIM if phase_one_obs else config.LEGACY_NEIGHBOR_OBS_DIM
    config.OBS_DIM_SINGLE = (
        config.SELF_OBS_DIM + (config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM) + config.UE_OBS_DIM
        if compact_obs
        else config.SELF_OBS_DIM
        + (config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM)
        + (config.MAX_ASSOCIATED_UES * config.UE_OBS_DIM)
    )


def _override_num_uavs(num_uavs: int, seed: int) -> None:
    if num_uavs <= 1:
        raise ValueError("--num_uavs must be greater than 1.")
    rng = np.random.default_rng(seed)
    config.NUM_UAVS = int(num_uavs)
    config.UAV_STORAGE_CAPACITY = rng.choice(
        np.arange(40 * 10**6, 80 * 10**6, 10**6),
        size=config.NUM_UAVS,
    ).astype(np.int64)
    config.UAV_COMPUTING_CAPACITY = rng.choice(
        np.arange(5 * 10**9, 20 * 10**9, 10**9),
        size=config.NUM_UAVS,
    ).astype(np.int64)
    _refresh_dimension_config()


def _zero_actions() -> np.ndarray:
    return np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)


def _configure_static_replay(args: argparse.Namespace, run_seed: int | None = None) -> None:
    _override_num_uavs(args.num_uavs, args.seed if run_seed is None else run_seed)
    config.DAG_ARRIVAL_PROB = float(args.dag_arrival_prob)
    config.STEPS_PER_EPISODE = int(args.steps)
    config.USE_MAPPO_COMPACT_OBS = True
    config.USE_PHASE_ONE_DEDICATED_OBS = True
    config.USE_STAGE_B_MOVEMENT_REWARD = False
    config.USE_PHASE_ONE_DAG_REWARD_SHAPING = False
    config.USE_HGNN_SCORE_ASSIGNMENT = True
    config.USE_SELECTIVE_HGNN_SCORING = True
    config.HGNN_SCORE_CHECKPOINT = args.checkpoint
    config.USE_PHASE_ONE_HYPEREDGES = True
    config.USE_COLLABORATIVE_HYPEREDGES = True
    config.USE_SERVICE_DOMAIN_HYPEREDGES = True
    config.USE_RESOURCE_COMPETITION_HYPEREDGES = True
    config.USE_CRITICAL_HYPEREDGES = True
    config.USE_ATTRIBUTE_HYPEREDGES = False
    config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES = False
    config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES = False
    config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES = False


def build_outcome_pair_index(
    input_dir: Path,
    *,
    use_good: bool,
    good_delta_tolerance: float,
    strong_delta_threshold: float,
    max_label_files: int,
    max_pairs: int,
) -> dict[tuple[int, int, int, str], OutcomePair]:
    paths = sorted(input_dir.glob("*_full_*_attribution.json"))
    if not paths:
        raise FileNotFoundError(f"No full attribution files found in {input_dir}")
    if max_label_files > 0:
        paths = paths[:max_label_files]

    index: dict[tuple[int, int, int, str], OutcomePair] = {}
    for path in paths:
        seed = _parse_seed(path)
        task_outcomes: dict[tuple[int, str], dict[str, Any]] = {}
        dag_outcomes: dict[tuple[int, str], dict[str, Any]] = {}
        print(f"[labels:outcomes] {path.name}", flush=True)
        for row in iter_json_array(path, "task_outcomes"):
            task_outcomes[(int(row.get("episode", 0)), str(row.get("task_id", "")))] = row
        for row in iter_json_array(path, "dag_outcomes"):
            dag_outcomes[(int(row.get("episode", 0)), str(row.get("dag_id", "")))] = row

        print(f"[labels:assignments] {path.name}", flush=True)
        for row in iter_json_array(path, "assignments"):
            if row.get("selection_mode") != "score" or not bool(row.get("disagrees_with_heuristic")):
                continue
            label, _ = classify_disagreement(
                row,
                task_outcomes.get((int(row.get("episode", 0)), str(row.get("task_id", "")))),
                dag_outcomes.get((int(row.get("episode", 0)), str(row.get("dag_id", "")))),
                good_delta_tolerance=good_delta_tolerance,
                strong_delta_threshold=strong_delta_threshold,
            )
            if label not in {"BAD_SCORE_DECISION", "BAD_SCORE_DECISION_STRONG", "GOOD_SCORE_DECISION"}:
                continue
            if label == "GOOD_SCORE_DECISION" and not use_good:
                continue

            score_uav = row.get("score_uav")
            heuristic_uav = row.get("heuristic_uav")
            if score_uav is None or heuristic_uav is None:
                continue
            if label.startswith("BAD"):
                preferred_uav, other_uav = int(heuristic_uav), int(score_uav)
                weight = 2.0 if label == "BAD_SCORE_DECISION_STRONG" else 1.0
            else:
                preferred_uav, other_uav = int(score_uav), int(heuristic_uav)
                weight = 0.5

            key = (
                seed,
                int(row.get("episode", 0)),
                int(row.get("step", 0)),
                str(row.get("task_id", "")),
            )
            index[key] = OutcomePair(
                preferred_uav=preferred_uav,
                other_uav=other_uav,
                weight=weight,
                label=label,
                delta_planned_finish=_safe_float(row.get("delta_planned_finish")),
            )
            if max_pairs > 0 and len(index) >= max_pairs:
                return index
    return index


def collect_replay_samples(
    args: argparse.Namespace,
    pair_index: dict[tuple[int, int, int, str], OutcomePair],
) -> list[SnapshotOutcomeSample]:
    samples: list[SnapshotOutcomeSample] = []
    progress = TerminalProgress(len(pair_index), "collect:outcome_pairs")
    collected_pairs = 0
    seeds = sorted({key[0] for key in pair_index})
    max_episodes_by_seed = {
        seed: max(key[1] for key in pair_index if key[0] == seed)
        for seed in seeds
    }

    for seed in seeds:
        _configure_static_replay(args, run_seed=seed)
        for episode in range(1, max_episodes_by_seed[seed] + 1):
            episode_seed = seed + episode - 1
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            env = Env()
            env.reset()
            for step in range(1, args.steps + 1):
                env.step(_zero_actions())
                assignment_snapshot = env.latest_assignment_graph_snapshot
                if assignment_snapshot is None:
                    continue
                pairs: list[tuple[str, OutcomePair]] = []
                edge_set = set(assignment_snapshot.task_uav_edges)
                for record in env.task_executor.latest_assignment_records:
                    key = (seed, episode, step, record.task_id)
                    pair = pair_index.get(key)
                    if pair is None:
                        continue
                    if (record.task_id, pair.preferred_uav) not in edge_set:
                        continue
                    if (record.task_id, pair.other_uav) not in edge_set:
                        continue
                    pairs.append((record.task_id, pair))
                if pairs:
                    samples.append(SnapshotOutcomeSample(snapshot=assignment_snapshot, pairs=pairs))
                    collected_pairs += len(pairs)
                    progress.update(len(pairs), postfix=f"seed {seed} episode {episode} step {step} samples {len(samples)}")
                if args.max_pairs and collected_pairs >= args.max_pairs:
                    progress.finish(postfix=f"collected_pairs {collected_pairs}")
                    return samples
    progress.finish(postfix=f"collected_pairs {collected_pairs}")
    return samples


def fine_tune(
    samples: list[SnapshotOutcomeSample],
    checkpoint: str,
    output_dir: Path,
    *,
    device: str,
    epochs: int,
    lr: float,
    outcome_margin: float,
    lambda_outcome: float,
    lambda_distill: float,
) -> tuple[str, list[FineTuneMetrics]]:
    model = PhaseOneGraphScheduler(device=device)
    base_model = PhaseOneGraphScheduler(device=device)
    _load_scheduler_state_compatible(model, checkpoint, device)
    _load_scheduler_state_compatible(base_model, checkpoint, device)
    model.train()
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics: list[FineTuneMetrics] = []
    progress = TerminalProgress(epochs, "train:outcome_rerank")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_pair_loss = 0.0
        total_distill_loss = 0.0
        total_pairs = 0
        correct_pairs = 0
        updates = 0
        for sample_idx in torch.randperm(len(samples)).tolist():
            sample = samples[sample_idx]
            output = model.forward_graph(sample.snapshot)
            with torch.no_grad():
                base_output = base_model.forward_graph(sample.snapshot)
            score_lookup = {edge: output.edge_scores[idx] for idx, edge in enumerate(output.edge_keys)}
            base_scores = base_output.edge_scores.detach()

            pair_terms: list[torch.Tensor] = []
            for task_id, pair in sample.pairs:
                preferred_key = (task_id, pair.preferred_uav)
                other_key = (task_id, pair.other_uav)
                if preferred_key not in score_lookup or other_key not in score_lookup:
                    continue
                preferred_score = score_lookup[preferred_key]
                other_score = score_lookup[other_key]
                term = F.relu(outcome_margin - preferred_score + other_score) * float(pair.weight)
                pair_terms.append(term)
                total_pairs += 1
                if float(preferred_score.detach().item()) > float(other_score.detach().item()):
                    correct_pairs += 1
            if not pair_terms:
                continue
            pair_loss = torch.stack(pair_terms).mean()
            distill_loss = F.mse_loss(output.edge_scores, base_scores)
            loss = lambda_outcome * pair_loss + lambda_distill * distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            total_pair_loss += float(pair_loss.detach().item())
            total_distill_loss += float(distill_loss.detach().item())
            updates += 1

        metric = FineTuneMetrics(
            epoch=epoch,
            avg_loss=total_loss / max(updates, 1),
            avg_pair_loss=total_pair_loss / max(updates, 1),
            avg_distill_loss=total_distill_loss / max(updates, 1),
            pair_accuracy=correct_pairs / max(total_pairs, 1),
            pair_count=total_pairs,
        )
        metrics.append(metric)
        progress.update(postfix=f"loss {metric.avg_loss:.4f} pair_acc {metric.pair_accuracy:.4f}")

    progress.finish(postfix=f"final loss {metrics[-1].avg_loss:.4f}" if metrics else "done")
    run_dir = output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "phase_one_graph_scheduler.pt"
    torch.save(model.state_dict(), model_path)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(metric) for metric in metrics], f, ensure_ascii=False, indent=2)
    return str(model_path), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune HGNN score model with outcome-aware pairwise labels.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--num_uavs", type=int, default=8)
    parser.add_argument("--dag_arrival_prob", type=float, default=0.050)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--outcome_margin", type=float, default=0.05)
    parser.add_argument("--lambda_outcome", type=float, default=0.2)
    parser.add_argument("--lambda_distill", type=float, default=1.0)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--max_label_files", type=int, default=0)
    parser.add_argument("--use_good", action="store_true")
    parser.add_argument("--good_delta_tolerance", type=float, default=0.1)
    parser.add_argument("--strong_delta_threshold", type=float, default=0.3)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _configure_static_replay(args)
    pair_index = build_outcome_pair_index(
        Path(args.input_dir),
        use_good=args.use_good,
        good_delta_tolerance=args.good_delta_tolerance,
        strong_delta_threshold=args.strong_delta_threshold,
        max_label_files=args.max_label_files,
        max_pairs=args.max_pairs,
    )
    print(f"[labels] pairs={len(pair_index)} use_good={args.use_good}", flush=True)
    samples = collect_replay_samples(args, pair_index)
    print(f"[samples] snapshots={len(samples)} pairs={sum(len(sample.pairs) for sample in samples)}", flush=True)
    if not samples:
        raise RuntimeError("No replay samples matched outcome labels.")
    model_path, metrics = fine_tune(
        samples,
        args.checkpoint,
        Path(args.output_dir),
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        outcome_margin=args.outcome_margin,
        lambda_outcome=args.lambda_outcome,
        lambda_distill=args.lambda_distill,
    )
    print(f"[saved] {model_path}", flush=True)
    if metrics:
        print(f"[final] {dataclasses.asdict(metrics[-1])}", flush=True)


if __name__ == "__main__":
    main()
