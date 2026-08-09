"""Frozen-tape evaluation entry point for training-free heuristic offloading policies.

中文：在冻结 tape 上运行启发式策略，产出与
`logs/stage1_temperature_followup/*/closed_loop/episodes.jsonl` **同构**的结果文件，
以便直接进现有的配对 bootstrap 分析。

不训练任何模型，不加载任何 checkpoint，不需要 GPU，全文件不 import torch。

用法示例：

    python scripts/run_heuristic_policy_baseline.py \
        --tape-dir /path/to/tape \
        --output-dir logs/heuristic_policy_baseline/20260807_greedy \
        --policies identity+greedy_eft identity+random \
        --scenario-indices 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.heuristic_policies import (
    HEURISTIC_POLICY_SCHEMA,
    POLICY_NAMES,
    UavGeometry,
    UpwardRankCache,
    act_with_heuristic_policy,
    build_policy,
)
from environment.stage1_temperature_sampling import canonical_sha256, file_sha256
from environment.stage1_temperature_tape import EPISODE_SLOTS, load_scenario_shard, validate_manifest


EPISODE_SCHEMA = "heuristic_policy_closed_loop_episode_v1"
RUN_MANIFEST_SCHEMA = "heuristic_policy_run_manifest_v1"

# 输出行里保留但不适用于启发式策略的字段（保留字段名置 null，保证列对齐）。
INAPPLICABLE_FIELDS = (
    "training_seed",
    "checkpoint_sha256",
    "sampling_replicate",
    "temperature",
    "encoder",
    "actor_uav_feature_dim",
)


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training-free heuristic policy baseline on a frozen tape")
    parser.add_argument("--tape-dir", type=Path, required=True, help="directory containing manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="create-only output directory")
    parser.add_argument("--policies", nargs="+", required=True, choices=POLICY_NAMES)
    parser.add_argument("--scenario-indices", nargs="+", type=int, required=True)
    parser.add_argument("--policy-replicates", nargs="+", type=int, default=[0])
    parser.add_argument("--max-physical-slots", type=int, default=EPISODE_SLOTS)
    parser.add_argument("--phase", choices=("pilot", "formal"), default="formal")
    parser.add_argument(
        "--record-decisions",
        action="store_true",
        help="write a per-decision trace (large; default off)",
    )
    parser.add_argument(
        "--record-decisions-max-slot",
        type=int,
        default=None,
        help="only record decisions for slot_index <= N (e.g. 0 for the corpus anchor)",
    )
    parser.add_argument(
        "--partition-hyperedges",
        choices=("on", "off"),
        default="off",
        help=(
            "KaHyPar E_part hyperedges. Heuristic policies never read the incidence matrix, "
            "so 'off' is pure speedup with no semantic effect on assignments."
        ),
    )
    parser.add_argument(
        "--validate-shards",
        action="store_true",
        help="re-parse every shard during manifest validation (slow, ~1.4 GB; sha256 already pins bytes)",
    )
    return parser.parse_args(argv)


def _cuda_initialized() -> bool:
    """只在 torch 已经被别的模块拉进来时查询，绝不主动 import。"""
    module = sys.modules.get("torch")
    if module is None:
        return False
    try:
        return bool(module.cuda.is_initialized())
    except Exception:  # pragma: no cover - torch build without CUDA support
        return False


def _write_line(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def _resolve_replicates(policy_name: str, requested: list[int]) -> list[int]:
    """确定性策略只跑 replicate 0；只有 random 才需要多个种子出误差棒。"""
    if policy_name.endswith("+random"):
        return sorted(set(int(value) for value in requested))
    return [0]


def run_episode(
    *,
    shard: dict[str, Any],
    episode_index: int,
    policy_name: str,
    policy_replicate: int,
    max_slots: int = EPISODE_SLOTS,
    record_decisions: bool = False,
    record_decisions_max_slot: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """跑完一个 episode，返回 (closed-loop 行的度量部分, 决策记录)。"""
    from environment.graph_builder import CleanGraphBuilder
    from environment.stage1_temperature_diagnostic import Stage1TemperatureDiagnosticEnv
    from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state

    scenario_seed = int(shard["evaluation_scenario_seed"])
    env = Stage1TemperatureDiagnosticEnv(scenario_shard=shard)
    builder = CleanGraphBuilder()
    env.reset()
    builder.reset()

    geometry = UavGeometry.from_service_positions(env.uav_service_positions)
    policy = build_policy(policy_name, geometry=geometry)
    # 统一口径：非 HEFT 策略的 rank_u 一律按 include_return=True 计算，
    # 保证所有策略的 rank_u 在同一把尺子上（用于事后的顺序重合度分析）。
    analysis_rank_cache = UpwardRankCache(geometry=geometry, include_return=True)

    reward_total = 0.0
    skip_total = 0
    decision_total = 0
    latest: dict[str, Any] = {}
    decision_records: list[dict[str, Any]] = []
    observed_slots: list[int] = []

    try:
        for slot_index in range(int(max_slots)):
            prepared = prepare_slot_state(env=env, graph_builder=builder)
            observed_slots.append(int(prepared.slot_index))
            env.apply_movement({})

            ready = [env.task_manager.get_task(task_id) for task_id in prepared.frozen_ready_task_ids]
            ready = [task for task in ready if task is not None]

            should_record = bool(record_decisions) and (
                record_decisions_max_slot is None or slot_index <= int(record_decisions_max_slot)
            )
            result = act_with_heuristic_policy(
                policy=policy,
                frozen_ready_tasks=ready,
                graph_snapshot=prepared.graph_snapshot,
                task_manager=env.task_manager,
                uavs=env.uavs,
                executor=env.executor,
                current_time_seconds=env.current_time_seconds,
                uav_service_positions=env.uav_service_positions,
                ue_service_positions=env.ue_service_positions,
                ues=env.ues,
                env=env,
                evaluation_scenario_seed=scenario_seed,
                episode_index=int(episode_index),
                slot_index=slot_index,
                policy_replicate=int(policy_replicate),
                analysis_rank_cache=analysis_rank_cache,
                record_decisions=should_record,
            )
            _, _, _, latest = env.commit_and_advance(
                assignment_buffer=result.assignments, offloading_skip_count=result.skip_count
            )
            reward_total += float(latest["step_reward"])
            skip_total += int(result.skip_count)
            decision_total += int(result.assignments.entry_count)
            decision_records.extend(result.decision_records)
    finally:
        builder.close()

    # 锚点 4：每个 episode 正好 max_slots 个时隙，且 slot_index 序列是 0..N-1。
    if observed_slots != list(range(int(max_slots))):
        raise AssertionError(f"slot index sequence mismatch: got {observed_slots[:5]}... expected 0..{int(max_slots) - 1}")
    # 锚点：UAV 全程悬停，缓存的 mean-pair 因子没有失效。
    geometry.assert_positions_unchanged(env.uav_service_positions, where="end of episode")
    # 锚点 3：admitted == generated，否则内部直接抛 AssertionError。
    arrival = env.arrival_identity_metrics()

    completed = int(latest.get("completed_dag_count", 0))
    generated = int(arrival["generated_dag_count"])
    invalid = int(latest.get("invalid_assignment_count", 0))
    # 锚点 2。
    if invalid != 0:
        raise AssertionError(f"invalid_assignment_count must be zero, got {invalid}")

    metrics = {
        "episode_reward_total": reward_total,
        "completed_dag_count": completed,
        "generated_dag_count": generated,
        "admitted_dag_count": int(arrival["admitted_dag_count"]),
        "arrival_blocked_count": int(arrival["arrival_blocked_count"]),
        "dag_completion_rate": latest.get("dag_completion_rate"),
        "average_dag_flowtime": latest.get("average_dag_flowtime"),
        "avg_uav_queue_length": latest.get("avg_uav_queue_length"),
        "admitted_incomplete_backlog": generated - completed,
        "invalid_assignment_count": invalid,
        "finite": bool(math.isfinite(reward_total)),
        "offloading_skip_count": skip_total,
        "assignment_count": decision_total,
        "uav_mean_distinct_pair_factor": float(geometry.mean_distinct_pair_factor),
        "policy_rank_u_convention": (
            policy.order_policy.rank_cache.convention
            if hasattr(policy.order_policy, "rank_cache")
            else analysis_rank_cache.convention
        ),
        "heft_include_return": policy.heft_include_return,
        "order_policy": policy.order_policy.name,
        "select_policy": policy.select_policy.name,
    }
    return metrics, decision_records


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    if int(args.max_physical_slots) != EPISODE_SLOTS:
        raise ValueError(f"heuristic baseline is frozen at {EPISODE_SLOTS} physical slots")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("output directory is create-only")

    tape_root = args.tape_dir.resolve()
    manifest = json.loads((tape_root / "manifest.json").read_text(encoding="utf-8"))
    # manifest 已对每个 shard 钉死 size_bytes + sha256，字节层面锁死；
    # validate_shards 只是额外的结构重解析（会把 ~1.4 GB 全部 parse 进内存），默认关闭。
    validate_manifest(manifest, root=tape_root, validate_shards=bool(args.validate_shards))

    if args.partition_hyperedges == "off":
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = False

    output.mkdir(parents=True)
    closed_dir = output / "closed_loop"
    closed_dir.mkdir()
    closed_path = closed_dir / "episodes.jsonl"
    decisions_path: Path | None = None
    if args.record_decisions:
        decisions_dir = output / "decisions"
        decisions_dir.mkdir()
        decisions_path = decisions_dir / "records.jsonl"

    scenarios = [int(value) for value in args.scenario_indices]
    rows_written = 0
    decisions_written = 0
    started = time.time()

    decisions_handle = decisions_path.open("x", encoding="utf-8", newline="\n") if decisions_path else None
    try:
        with closed_path.open("x", encoding="utf-8", newline="\n") as closed_handle:
            for policy_name in args.policies:
                for replicate in _resolve_replicates(policy_name, list(args.policy_replicates)):
                    for scenario_index in scenarios:
                        record = manifest["shards"][scenario_index]
                        shard = load_scenario_shard(tape_root / record["path"])
                        metrics, decision_records = run_episode(
                            shard=shard,
                            episode_index=scenario_index,
                            policy_name=policy_name,
                            policy_replicate=replicate,
                            max_slots=EPISODE_SLOTS,
                            record_decisions=bool(args.record_decisions),
                            record_decisions_max_slot=args.record_decisions_max_slot,
                        )
                        row: dict[str, Any] = {
                            "schema": EPISODE_SCHEMA,
                            "phase": args.phase,
                            "policy_name": policy_name,
                            "policy_replicate": int(replicate),
                            "logical_tape_sha256": manifest["logical_tape_sha256"],
                            "episode_index": int(scenario_index),
                            "evaluation_scenario_seed": int(shard["evaluation_scenario_seed"]),
                            "physical_slots": EPISODE_SLOTS,
                            "active_dag_cap": 1,
                            "queue_cap": 16,
                            "partition_hyperedges_enabled": args.partition_hyperedges == "on",
                        }
                        row.update({name: None for name in INAPPLICABLE_FIELDS})
                        row.update(metrics)
                        row["policy_config_sha256"] = canonical_sha256(
                            {
                                "schema": HEURISTIC_POLICY_SCHEMA,
                                "policy_name": policy_name,
                                "order_policy": row["order_policy"],
                                "select_policy": row["select_policy"],
                                "heft_include_return": row["heft_include_return"],
                                "policy_replicate": int(replicate),
                                "rank_u_convention": row["policy_rank_u_convention"],
                            }
                        )
                        _write_line(closed_handle, row)
                        rows_written += 1
                        if decisions_handle is not None:
                            for decision in decision_records:
                                _write_line(decisions_handle, decision)
                                decisions_written += 1
                        print(
                            f"[{rows_written}] {policy_name} rep={replicate} episode={scenario_index} "
                            f"completed={row['completed_dag_count']} reward={row['episode_reward_total']:.3f}",
                            flush=True,
                        )
    finally:
        if decisions_handle is not None:
            decisions_handle.close()

    summary: dict[str, Any] = {
        "schema": RUN_MANIFEST_SCHEMA,
        "phase": args.phase,
        "technical_pass": True,
        "policies": list(args.policies),
        "scenario_indices": scenarios,
        "policy_replicates": [int(value) for value in args.policy_replicates],
        "physical_slots": EPISODE_SLOTS,
        "closed_loop_rows": rows_written,
        "closed_loop_sha256": file_sha256(closed_path),
        "decision_records": decisions_written,
        "decisions_sha256": file_sha256(decisions_path) if decisions_path is not None else None,
        "record_decisions_max_slot": args.record_decisions_max_slot,
        "logical_tape_sha256": manifest["logical_tape_sha256"],
        "partition_hyperedges_enabled": args.partition_hyperedges == "on",
        "validate_shards": bool(args.validate_shards),
        # 本框架自身不 import torch，但 `marl_models.mappo.clean_slot_orchestrator`
        # 在模块顶层做了 `try: import torch`，所以在装了 torch 的机器上它会被动进入
        # sys.modules。如实记录，不做「torch 不在 sys.modules」这种会误报的断言。
        "torch_present_transitively": "torch" in sys.modules,
        "torch_used_by_policy": False,
        "cuda_initialized": _cuda_initialized(),
        "uses_checkpoint": False,
        "elapsed_seconds": round(time.time() - started, 3),
        "numpy_version": str(np.__version__),
        "uav_compute_rate_ops_per_sec": float(config.UAV_COMPUTE_RATE_OPS_PER_SEC),
        "clean_max_queue_per_uav": int(config.CLEAN_MAX_QUEUE_PER_UAV),
    }
    if _cuda_initialized():
        raise AssertionError("heuristic baseline must never initialize CUDA")
    (output / "run_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
