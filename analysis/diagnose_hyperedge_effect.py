from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from statistics import mean
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

import config
from environment.dag_tasks import TASK_STATE_READY
from environment.env import Env
from environment.graph_builder import HeteroGraphSnapshot
from marl_models.hgnn.scheduler import PhaseOneGraphScheduler


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
    config.NUM_UAVS = num_uavs
    rng = np.random.default_rng(seed)
    config.UAV_STORAGE_CAPACITY = rng.choice(
        np.arange(40 * 10**6, 80 * 10**6, 10**6),
        size=config.NUM_UAVS,
    ).astype(np.int64)
    config.UAV_COMPUTING_CAPACITY = rng.choice(
        np.arange(5 * 10**9, 20 * 10**9, 10**9),
        size=config.NUM_UAVS,
    ).astype(np.int64)
    _refresh_dimension_config()


def _ablate_snapshot(snapshot: HeteroGraphSnapshot, mode: str) -> HeteroGraphSnapshot:
    if mode == "full":
        return snapshot
    if mode == "no_service":
        return replace(
            snapshot,
            service_domain_hyperedges=[],
            collaborative_hyperedges=snapshot.resource_competition_hyperedges,
        )
    if mode == "no_resource":
        return replace(
            snapshot,
            resource_competition_hyperedges=[],
            collaborative_hyperedges=snapshot.service_domain_hyperedges,
        )
    if mode == "no_critical":
        return replace(snapshot, critical_hyperedges=[], critical_support_hyperedges=[])
    if mode == "no_hyperedge":
        return replace(
            snapshot,
            service_domain_hyperedges=[],
            resource_competition_hyperedges=[],
            collaborative_hyperedges=[],
            critical_hyperedges=[],
            critical_support_hyperedges=[],
            compute_attribute_hyperedges=[],
            communication_attribute_hyperedges=[],
            candidate_scarce_attribute_hyperedges=[],
            attribute_hyperedges=[],
        )
    raise ValueError(f"Unsupported ablation mode: {mode}")


def _best_assignments(edge_scores: dict[tuple[str, int], float]) -> dict[str, tuple[int, float]]:
    best: dict[str, tuple[int, float]] = {}
    for (task_id, uav_id), score in edge_scores.items():
        if task_id not in best or score > best[task_id][1]:
            best[task_id] = (uav_id, score)
    return best


def _hyperedge_stats(snapshot: HeteroGraphSnapshot, ready_task_ids: set[str]) -> dict[str, float]:
    task_ids = set(snapshot.task_ids)
    uav_ids = set(snapshot.uav_ids)
    service_tasks = set()
    service_uavs = set()
    for tasks, uavs in snapshot.service_domain_hyperedges:
        service_tasks.update(tasks)
        service_uavs.update(uavs)
    resource_tasks = set()
    resource_uavs = set()
    for tasks, uavs in snapshot.resource_competition_hyperedges:
        resource_tasks.update(tasks)
        resource_uavs.update(uavs)
    critical_tasks = set()
    for tasks in snapshot.critical_hyperedges:
        critical_tasks.update(tasks)
    critical_support_tasks = set()
    critical_support_uavs = set()
    for tasks, uavs in snapshot.critical_support_hyperedges:
        critical_support_tasks.update(tasks)
        critical_support_uavs.update(uavs)
    critical_tasks.update(critical_support_tasks)
    return {
        "num_tasks": float(len(task_ids)),
        "num_ready_tasks": float(len(ready_task_ids)),
        "num_uavs": float(len(uav_ids)),
        "num_task_uav_edges": float(len(snapshot.task_uav_edges)),
        "num_service_edges": float(len(snapshot.service_domain_hyperedges)),
        "num_resource_edges": float(len(snapshot.resource_competition_hyperedges)),
        "num_critical_edges": float(len(snapshot.critical_hyperedges)),
        "num_critical_support_edges": float(len(snapshot.critical_support_hyperedges)),
        "service_ready_coverage": len(service_tasks & ready_task_ids) / max(len(ready_task_ids), 1),
        "resource_ready_coverage": len(resource_tasks & ready_task_ids) / max(len(ready_task_ids), 1),
        "critical_ready_coverage": len(critical_tasks & ready_task_ids) / max(len(ready_task_ids), 1),
        "critical_support_ready_coverage": len(critical_support_tasks & ready_task_ids) / max(len(ready_task_ids), 1),
        "service_uav_coverage": len(service_uavs) / max(len(uav_ids), 1),
        "resource_uav_coverage": len(resource_uavs) / max(len(uav_ids), 1),
        "critical_support_uav_coverage": len(critical_support_uavs) / max(len(uav_ids), 1),
    }


def _critical_support_teacher_alignment_stats(snapshot: HeteroGraphSnapshot, env: Env) -> dict[str, float]:
    if not snapshot.critical_support_hyperedges:
        return {
            "critical_support_anchor_task_pairs": 0.0,
            "critical_support_anchor_teacher_overlap_rate": 0.0,
            "critical_support_anchor_teacher_overlap_count": 0.0,
            "critical_support_anchor_teacher_eft_gap_mean": 0.0,
            "critical_support_anchor_teacher_eft_gap_max": 0.0,
            "critical_support_anchor_feasible_rate": 0.0,
            "critical_support_edge_any_teacher_overlap_rate": 0.0,
            "critical_support_edge_all_teacher_overlap_rate": 0.0,
        }

    targets = env.task_executor.build_supervision_targets(
        env.task_manager,
        env.uavs,
        float(env._time_step),  # Diagnostic script only; keep aligned with snapshot build time.
        allowed_edges=set(snapshot.task_uav_edges),
    )
    target_by_task = {target.task_id: target for target in targets}
    pair_count = 0
    feasible_count = 0
    overlap_count = 0
    eft_gaps: list[float] = []
    edge_any_overlap: list[bool] = []
    edge_all_overlap: list[bool] = []

    for task_ids, uav_ids in snapshot.critical_support_hyperedges:
        if not task_ids or not uav_ids:
            continue
        anchor_uav = int(uav_ids[0])
        edge_pair_count = 0
        edge_overlap_count = 0
        for task_id in task_ids:
            target = target_by_task.get(task_id)
            if target is None:
                continue
            pair_count += 1
            edge_pair_count += 1
            if anchor_uav in target.heuristic_eft_by_uav:
                feasible_count += 1
                best_eft = float(target.heuristic_eft_by_uav[target.heuristic_best_uav])
                anchor_eft = float(target.heuristic_eft_by_uav[anchor_uav])
                eft_gaps.append(max(anchor_eft - best_eft, 0.0))
            if anchor_uav == int(target.heuristic_best_uav):
                overlap_count += 1
                edge_overlap_count += 1
        if edge_pair_count > 0:
            edge_any_overlap.append(edge_overlap_count > 0)
            edge_all_overlap.append(edge_overlap_count == edge_pair_count)

    return {
        "critical_support_anchor_task_pairs": float(pair_count),
        "critical_support_anchor_teacher_overlap_rate": overlap_count / max(pair_count, 1),
        "critical_support_anchor_teacher_overlap_count": float(overlap_count),
        "critical_support_anchor_teacher_eft_gap_mean": float(np.mean(eft_gaps)) if eft_gaps else 0.0,
        "critical_support_anchor_teacher_eft_gap_max": float(np.max(eft_gaps)) if eft_gaps else 0.0,
        "critical_support_anchor_feasible_rate": feasible_count / max(pair_count, 1),
        "critical_support_edge_any_teacher_overlap_rate": float(np.mean(edge_any_overlap)) if edge_any_overlap else 0.0,
        "critical_support_edge_all_teacher_overlap_rate": float(np.mean(edge_all_overlap)) if edge_all_overlap else 0.0,
    }


def _score_diff_stats(
    full_scores: dict[tuple[str, int], float],
    nohyper_scores: dict[tuple[str, int], float],
    ready_task_ids: set[str],
    critical_task_ids: set[str],
) -> dict[str, float]:
    common_edges = sorted(set(full_scores) & set(nohyper_scores))
    diffs = np.array([full_scores[key] - nohyper_scores[key] for key in common_edges], dtype=np.float32)
    abs_diffs = np.abs(diffs)
    full_best = _best_assignments(full_scores)
    nohyper_best = _best_assignments(nohyper_scores)
    common_tasks = sorted(set(full_best) & set(nohyper_best) & ready_task_ids)
    changed = [full_best[task_id][0] != nohyper_best[task_id][0] for task_id in common_tasks]
    critical_common_tasks = [task_id for task_id in common_tasks if task_id in critical_task_ids]
    critical_changed = [
        full_best[task_id][0] != nohyper_best[task_id][0]
        for task_id in critical_common_tasks
    ]
    per_task_centered_abs: list[float] = []
    per_task_diff_std: list[float] = []
    full_margins: list[float] = []
    ablated_margins: list[float] = []
    margin_deltas: list[float] = []
    top1_diff_minus_mean: list[float] = []
    for task_id in common_tasks:
        task_edges = [
            edge_key
            for edge_key in common_edges
            if edge_key[0] == task_id
        ]
        if len(task_edges) < 2:
            continue
        task_full_scores = np.array([full_scores[edge_key] for edge_key in task_edges], dtype=np.float32)
        task_ablated_scores = np.array([nohyper_scores[edge_key] for edge_key in task_edges], dtype=np.float32)
        task_diffs = task_full_scores - task_ablated_scores
        centered = task_diffs - float(np.mean(task_diffs))
        per_task_centered_abs.append(float(np.mean(np.abs(centered))))
        per_task_diff_std.append(float(np.std(task_diffs)))

        full_sorted = np.sort(task_full_scores)[::-1]
        ablated_sorted = np.sort(task_ablated_scores)[::-1]
        full_margin = float(full_sorted[0] - full_sorted[1])
        ablated_margin = float(ablated_sorted[0] - ablated_sorted[1])
        full_margins.append(full_margin)
        ablated_margins.append(ablated_margin)
        margin_deltas.append(full_margin - ablated_margin)
        full_top_uav = full_best[task_id][0]
        top_edge = (task_id, full_top_uav)
        if top_edge in full_scores and top_edge in nohyper_scores:
            top1_diff_minus_mean.append(float((full_scores[top_edge] - nohyper_scores[top_edge]) - np.mean(task_diffs)))

    if abs_diffs.size == 0:
        return {
            "score_diff_mean": 0.0,
            "score_abs_diff_mean": 0.0,
            "score_abs_diff_max": 0.0,
            "assignment_change_rate": 0.0,
            "critical_assignment_change_rate": 0.0,
            "common_scored_edges": 0.0,
            "common_ready_tasks": 0.0,
            "common_critical_tasks": 0.0,
            "per_task_diff_std_mean": 0.0,
            "per_task_centered_abs_diff_mean": 0.0,
            "top1_margin_full_mean": 0.0,
            "top1_margin_ablated_mean": 0.0,
            "top1_margin_delta_mean": 0.0,
            "top1_diff_minus_task_mean": 0.0,
        }
    return {
        "score_diff_mean": float(np.mean(diffs)),
        "score_abs_diff_mean": float(np.mean(abs_diffs)),
        "score_abs_diff_max": float(np.max(abs_diffs)),
        "assignment_change_rate": float(np.mean(changed)) if changed else 0.0,
        "critical_assignment_change_rate": float(np.mean(critical_changed)) if critical_changed else 0.0,
        "common_scored_edges": float(len(common_edges)),
        "common_ready_tasks": float(len(common_tasks)),
        "common_critical_tasks": float(len(critical_common_tasks)),
        "per_task_diff_std_mean": float(np.mean(per_task_diff_std)) if per_task_diff_std else 0.0,
        "per_task_centered_abs_diff_mean": float(np.mean(per_task_centered_abs)) if per_task_centered_abs else 0.0,
        "top1_margin_full_mean": float(np.mean(full_margins)) if full_margins else 0.0,
        "top1_margin_ablated_mean": float(np.mean(ablated_margins)) if ablated_margins else 0.0,
        "top1_margin_delta_mean": float(np.mean(margin_deltas)) if margin_deltas else 0.0,
        "top1_diff_minus_task_mean": float(np.mean(top1_diff_minus_mean)) if top1_diff_minus_mean else 0.0,
    }


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(mean(row.get(key, 0.0) for row in rows)) for key in keys}


def _load_scheduler_state_compatible(scheduler: PhaseOneGraphScheduler, state_dict: dict) -> None:
    model_state = scheduler.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    skipped = sorted(set(state_dict) - set(compatible_state))
    scheduler.load_state_dict(compatible_state, strict=False)
    if skipped:
        print(
            "HGNN checkpoint partially loaded; skipped incompatible keys: "
            + ", ".join(skipped[:8])
            + (" ..." if len(skipped) > 8 else "")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether phase-one hyperedges change HGNN scores/assignments.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--num_uavs", type=int, default=config.NUM_UAVS)
    parser.add_argument("--dag_arrival_prob", type=float, default=config.DAG_ARRIVAL_PROB)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--sample_every", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--output", default="/tmp/hyperedge_effect_diagnostic.json")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["no_service", "no_resource", "no_critical", "no_hyperedge"],
        choices=["no_service", "no_resource", "no_critical", "no_hyperedge"],
        help="Ablation modes to compare against full on the same snapshots.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _override_num_uavs(args.num_uavs, args.seed)
    config.DAG_ARRIVAL_PROB = args.dag_arrival_prob
    config.USE_HGNN_SCORE_ASSIGNMENT = False
    config.HGNN_SCORE_CHECKPOINT = ""
    config.USE_MAPPO_COMPACT_OBS = True
    config.USE_PHASE_ONE_DEDICATED_OBS = True
    config.USE_PHASE_ONE_HYPEREDGES = True
    config.USE_COLLABORATIVE_HYPEREDGES = True
    config.USE_SERVICE_DOMAIN_HYPEREDGES = True
    config.USE_RESOURCE_COMPETITION_HYPEREDGES = True
    config.USE_CRITICAL_HYPEREDGES = True
    config.USE_ATTRIBUTE_HYPEREDGES = False
    config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES = False
    config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES = False
    config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES = False

    scheduler = PhaseOneGraphScheduler(device=args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    _load_scheduler_state_compatible(scheduler, state_dict)
    scheduler.eval()

    env = Env()
    env.reset()
    per_snapshot: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        actions = np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)
        env.step(actions)
        snapshot = env.latest_graph_snapshot
        if snapshot is None or step <= args.warmup_steps or step % args.sample_every != 0:
            continue
        ready_task_ids = {
            task.task_id
            for task in env.task_manager.tasks.values()
            if task.state == TASK_STATE_READY
        }
        critical_task_ids = {
            task_id
            for hyperedge in snapshot.critical_hyperedges
            for task_id in hyperedge
        }
        critical_task_ids.update(
            task_id
            for task_ids, _ in snapshot.critical_support_hyperedges
            for task_id in task_ids
        )
        full_output = scheduler.score_graph(snapshot)
        row = {}
        row.update(_hyperedge_stats(snapshot, ready_task_ids))
        row.update(_critical_support_teacher_alignment_stats(snapshot, env))
        for mode in args.modes:
            ablated_output = scheduler.score_graph(_ablate_snapshot(snapshot, mode))
            stats = _score_diff_stats(
                full_output.edge_scores,
                ablated_output.edge_scores,
                ready_task_ids,
                critical_task_ids,
            )
            row.update({f"{mode}_{key}": value for key, value in stats.items()})
        per_snapshot.append(row)

    summary = _mean_dict(per_snapshot) if per_snapshot else {}
    result = {
        "config": {
            "seed": args.seed,
            "num_uavs": config.NUM_UAVS,
            "dag_arrival_prob": config.DAG_ARRIVAL_PROB,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "sample_every": args.sample_every,
            "num_snapshots": len(per_snapshot),
            "checkpoint": args.checkpoint,
        },
        "summary": summary,
        "per_snapshot": per_snapshot,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result["config"], indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
