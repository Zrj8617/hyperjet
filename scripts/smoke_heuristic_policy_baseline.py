"""Smoke tests for the training-free heuristic policy baseline framework.

中文：不需要完整 tape（`build_manifest` 硬性要求 20 个 shard）。本脚本直接用
`generate_scenario_shard(i)` 在内存里造 shard 构造环境，秒级到分钟级完成。

    python scripts/smoke_heuristic_policy_baseline.py
    python scripts/smoke_heuristic_policy_baseline.py --episodes 3 --slots 200

覆盖第 2 轮的验证锚点 1–9（锚点 1 需要 `analysis_inbox/corpus_slot0_anchor.jsonl`，
拿不到会 SKIP 并明确说明）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment import comm_model
from environment.heuristic_policies import (
    GreedyEFTSelectPolicy,
    HeftUpwardRankOrderPolicy,
    IdentityOrderPolicy,
    UavGeometry,
    UpwardRankCache,
    build_policy,
    compute_upward_ranks,
    task_computation_seconds,
    transmission_seconds_from_factor,
)
from environment.stage1_temperature_sampling import canonical_sha256
from environment.stage1_temperature_tape import generate_scenario_shard, load_scenario_shard
from scripts.run_heuristic_policy_baseline import run_episode


ANCHOR_CORPUS_PATH = ROOT / "analysis_inbox" / "corpus_slot0_anchor.jsonl"

PASSED: list[str] = []
SKIPPED: list[str] = []


def _ok(name: str, detail: str = "") -> None:
    PASSED.append(name)
    print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def _skip(name: str, reason: str) -> None:
    SKIPPED.append(name)
    print(f"  SKIP  {name}  ({reason})", flush=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# A. 纯函数层：通信代价的代数恒等
# --------------------------------------------------------------------------------------


def test_factorization_is_exact() -> None:
    """锚点：把 mb 和 bw 提到均值外面是精确恒等，不是近似。"""
    rng = np.random.default_rng(20260807)
    positions = [tuple(float(value) for value in rng.random(2) * 500.0) for _ in range(5)]
    geometry = UavGeometry.from_service_positions({index: np.asarray(pos) for index, pos in enumerate(positions)})

    for data_mb, bandwidth in ((3.25, 20.0), (0.75, 50.0), (10.5, 100.0)):
        direct = []
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i == j:
                    continue
                distance = comm_model.clean_distance_2d(positions[i], positions[j])
                direct.append(comm_model.clean_transmission_time_seconds(data_mb, bandwidth, distance))
        reference = float(sum(direct) / len(direct))
        factored = transmission_seconds_from_factor(data_mb, bandwidth, geometry.mean_distinct_pair_factor)
        _require(
            abs(factored - reference) <= 1e-12 * max(abs(reference), 1.0),
            f"factorization mismatch: {factored!r} vs {reference!r}",
        )
    _ok("A1 c(t,s) factorization is algebraically exact", f"mean_factor={geometry.mean_distinct_pair_factor:.6f}")

    # 无序 10 对 与 有序 20 对 的均值必须相同（距离对称）。
    ordered = []
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i != j:
                ordered.append(1.0 / comm_model.clean_distance_factor(comm_model.clean_distance_2d(positions[i], positions[j])))
    _require(
        abs(float(np.mean(ordered)) - geometry.mean_distinct_pair_factor) <= 1e-12 * geometry.mean_distinct_pair_factor,
        "ordered/unordered pair means disagree",
    )
    _ok("A2 ordered-20-pair mean == unordered-10-pair mean")


# --------------------------------------------------------------------------------------
# B. HEFT upward rank
# --------------------------------------------------------------------------------------


def _brute_force_rank(
    *,
    task_id: str,
    task_map: dict[str, Any],
    sink_ids: set[str],
    geometry: UavGeometry,
    job: Any,
    include_return: bool,
    return_factor: float,
) -> float:
    """独立实现：穷举从 task_id 出发的所有路径，取代价最大者。与 DP 递推互为交叉验证。"""

    def enumerate_paths(current: str) -> list[list[str]]:
        successors = [child for child in task_map[current].successors if child in task_map]
        if not successors:
            return [[current]]
        return [[current] + tail for child in successors for tail in enumerate_paths(child)]

    best = -float("inf")
    for path in enumerate_paths(task_id):
        cost = sum(task_computation_seconds(task_map[node]) for node in path)
        for node in path[:-1]:
            cost += transmission_seconds_from_factor(
                task_map[node].output_data_size_mb,
                job.base_upload_bandwidth_mbps,
                geometry.mean_distinct_pair_factor,
            )
        if include_return and path[-1] in sink_ids:
            cost += transmission_seconds_from_factor(
                task_map[path[-1]].output_data_size_mb, job.base_download_bandwidth_mbps, return_factor
            )
        best = max(best, cost)
    return float(best)


def test_upward_rank(env: Any) -> None:
    task_manager = env.task_manager
    geometry = UavGeometry.from_service_positions(env.uav_service_positions)
    dag_ids = sorted({task.dag_id for task in task_manager.get_ready_tasks()})
    _require(bool(dag_ids), "no DAGs available for rank tests")

    checked = 0
    for dag_id in dag_ids:
        job = task_manager.get_job(dag_id)
        tasks = task_manager.get_job_tasks(dag_id)
        task_map = {task.task_id: task for task in tasks}
        sink_ids = set(str(value) for value in job.sink_task_ids)

        # B1：无后继任务集合 == job.sink_task_ids（DP 基例与回传项口径一致）
        no_successor = {task.task_id for task in tasks if not [c for c in task.successors if c in task_map]}
        _require(no_successor == sink_ids, f"sink set mismatch in {dag_id}: {no_successor} vs {sink_ids}")

        for include_return in (False, True):
            ranks = compute_upward_ranks(
                dag_id=dag_id, task_manager=task_manager, geometry=geometry, include_return=include_return
            )
            return_factor = geometry.mean_uav_to_point_factor(job.source_pos) if include_return else 0.0
            for task_id in task_map:
                brute = _brute_force_rank(
                    task_id=task_id,
                    task_map=task_map,
                    sink_ids=sink_ids,
                    geometry=geometry,
                    job=job,
                    include_return=include_return,
                    return_factor=return_factor,
                )
                _require(
                    abs(ranks[task_id] - brute) <= 1e-9 * max(abs(brute), 1.0),
                    f"rank mismatch {dag_id}/{task_id} include_return={include_return}: {ranks[task_id]} vs {brute}",
                )
                checked += 1
    _ok("B1 sink set == job.sink_task_ids", f"{len(dag_ids)} DAGs")
    _ok("B2 upward rank DP == brute-force path enumeration", f"{checked} task/convention pairs")

    # B3：令 c == 0，rank_u * COMPUTE_RATE 必须逐任务等于 _mark_critical_path 的 dp。
    zero_geometry = UavGeometry(
        uav_ids=geometry.uav_ids, positions=geometry.positions, mean_distinct_pair_factor=0.0
    )
    compared = 0
    for dag_id in dag_ids:
        tasks = task_manager.get_job_tasks(dag_id)
        task_map = {task.task_id: task for task in tasks}
        dp: dict[str, float] = {}
        for task in sorted(tasks, key=lambda item: (-item.level, item.task_id)):
            successors = [child for child in task.successors if child in task_map]
            dp[task.task_id] = float(task.num_operation) + (max(dp[c] for c in successors) if successors else 0.0)
        ranks = compute_upward_ranks(
            dag_id=dag_id, task_manager=task_manager, geometry=zero_geometry, include_return=False
        )
        for task_id, value in dp.items():
            scaled = ranks[task_id] * float(config.UAV_COMPUTE_RATE_OPS_PER_SEC)
            _require(
                abs(scaled - value) <= 1e-6 * max(abs(value), 1.0),
                f"c==0 degeneracy failed for {task_id}: {scaled} vs {value}",
            )
            compared += 1
    _ok("B3 c==0 degenerates to _mark_critical_path dp", f"{compared} tasks")


# --------------------------------------------------------------------------------------
# C. 钩子契约
# --------------------------------------------------------------------------------------


def test_order_policies(env: Any, frozen_ready_task_ids: list[str]) -> None:
    task_manager = env.task_manager
    ready = [task_manager.get_task(task_id) for task_id in frozen_ready_task_ids]
    ready = [task for task in ready if task is not None]
    _require(len(ready) >= 2, "need at least two ready tasks for order tests")

    identity = IdentityOrderPolicy()
    ordered = identity(frozen_ready_tasks=ready, task_manager=task_manager, env=env)
    _require(
        [task.task_id for task in ordered] == [task.task_id for task in ready],
        "identity order policy must preserve freeze_ready_tasks order",
    )
    _ok("C1 identity OrderPolicy == frozen_ready_task_ids", f"{len(ready)} tasks")

    geometry = UavGeometry.from_service_positions(env.uav_service_positions)
    for include_return in (False, True):
        heft = HeftUpwardRankOrderPolicy(rank_cache=UpwardRankCache(geometry=geometry, include_return=include_return))
        heft_ordered = heft(frozen_ready_tasks=ready, task_manager=task_manager, env=env)
        _require(
            sorted(t.task_id for t in heft_ordered) == sorted(t.task_id for t in ready),
            "HEFT order policy must return a permutation",
        )
        keys = []
        for task in heft_ordered:
            job = task_manager.get_job(task.dag_id)
            keys.append(
                (
                    -heft.rank_cache.rank_for_task(task, task_manager),
                    float(job.arrival_time),
                    str(task.dag_id),
                    int(task.topological_index),
                    str(task.task_id),
                )
            )
        _require(len(set(keys)) == len(keys), "HEFT sort key must be a total order (no ties)")
        _require(keys == sorted(keys), "HEFT output must be sorted by its own key")
        ranks = [-key[0] for key in keys]
        _require(all(a >= b for a, b in zip(ranks, ranks[1:])), "HEFT must be descending in rank_u")
    _ok("C2 HEFT OrderPolicy: permutation + total order + descending rank_u")


# --------------------------------------------------------------------------------------
# D. 端到端 episode
# --------------------------------------------------------------------------------------


def test_episode_invariants(shard: dict[str, Any], slots: int) -> dict[str, Any]:
    """锚点 2/3/4 由 run_episode 内部断言保证；这里额外验证决策级不变量。"""
    metrics, decisions = run_episode(
        shard=shard,
        episode_index=int(shard["episode_index"]),
        policy_name="identity+greedy_eft",
        policy_replicate=0,
        max_slots=slots,
        record_decisions=True,
        record_decisions_max_slot=None,
    )
    _require(int(metrics["invalid_assignment_count"]) == 0, "anchor 2: invalid_assignment_count must be 0")
    _ok("D1 invalid_assignment_count == 0 (anchor 2)")
    _ok("D2 arrival identity admitted == generated (anchor 3)", f"generated={metrics['generated_dag_count']}")
    _ok("D3 exactly %d slots, slot_index 0..%d (anchor 4)" % (slots, slots - 1))

    _require(bool(decisions), "expected at least one decision record")
    for record in decisions:
        eft = record["eft"]
        mask = record["candidate_mask"]
        legal = [index for index, flag in enumerate(mask) if flag]
        _require(bool(legal), "decision recorded with no legal candidate")
        _require(record["selected_index"] in legal, "greedy selected an illegal candidate")
        minimum = min(float(eft[index]) for index in legal)
        _require(
            float(eft[record["selected_index"]]) == minimum,
            f"greedy did not pick the minimum legal EFT in {record['stable_task_id']}",
        )
        tied = [index for index in legal if float(eft[index]) == minimum]
        _require(
            record["selected_uav_id"] == min(int(record["candidate_uav_ids"][i]) for i in tied),
            "greedy tie-break must take the smallest uav_id",
        )
        _require(record["rank_u"] is not None and np.isfinite(record["rank_u"]), "rank_u must be recorded and finite")
        _require(record["base_upload_bandwidth_mbps"] is not None, "bandwidth level must be recorded")
    _ok("D4 greedy always picks min legal EFT, tie-break min uav_id", f"{len(decisions)} decisions")
    _ok("D5 rank_u + bandwidth recorded for every decision (unified include_return=true yardstick)")
    return metrics


def test_reproducibility(shard: dict[str, Any], slots: int) -> None:
    """锚点 5：同一 (策略, 场景) 连跑两次，输出逐 bit 相同。"""
    digests = []
    for _ in range(2):
        metrics, decisions = run_episode(
            shard=shard,
            episode_index=int(shard["episode_index"]),
            policy_name="identity+random",
            policy_replicate=0,
            max_slots=slots,
            record_decisions=True,
            record_decisions_max_slot=None,
        )
        digests.append(canonical_sha256({"metrics": metrics, "decisions": decisions}))
    _require(digests[0] == digests[1], f"reruns differ: {digests}")
    _ok("D6 bit-identical reruns (anchor 5)", digests[0][:16])


def test_random_replicates_differ(shard: dict[str, Any], slots: int) -> None:
    digests = []
    for replicate in (0, 1):
        metrics, _ = run_episode(
            shard=shard,
            episode_index=int(shard["episode_index"]),
            policy_name="identity+random",
            policy_replicate=replicate,
            max_slots=slots,
            record_decisions=False,
        )
        digests.append(canonical_sha256(metrics))
    _require(digests[0] != digests[1], "random replicates 0 and 1 produced identical episodes")
    _ok("D7 random policy_replicate actually changes the stream")


def test_partition_hyperedges_are_inert(shard: dict[str, Any]) -> None:
    """E_part 对启发式策略无语义影响（没有编码器读 incidence 矩阵）。

    只有装了 kahypar 才有意义：没装时 worker 会反复 spawn-then-timeout（每次 10s），
    既慢又测不到真正的分区路径。所以未安装时直接 SKIP。
    """
    import importlib.util

    if importlib.util.find_spec("kahypar") is None:
        _skip("D8 --partition-hyperedges off is semantically inert", "kahypar not installed; run this on the server")
        return
    original = bool(config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES)
    results = {}
    try:
        for enabled in (False, True):
            config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = enabled
            metrics, _ = run_episode(
                shard=shard,
                episode_index=int(shard["episode_index"]),
                policy_name="identity+greedy_eft",
                policy_replicate=0,
                max_slots=20,
                record_decisions=False,
            )
            results[enabled] = canonical_sha256(metrics)
    finally:
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = original
    _require(results[False] == results[True], f"partition hyperedges changed the outcome: {results}")
    _ok("D8 --partition-hyperedges off is semantically inert", "20 slots")


def test_greedy_beats_random(shards: list[dict[str, Any]], slots: int) -> None:
    """锚点 6（预注册判据）：逐场景 greedy >= random，且均值提升 > 10%。"""
    greedy_values, random_values = [], []
    for shard in shards:
        greedy, _ = run_episode(
            shard=shard,
            episode_index=int(shard["episode_index"]),
            policy_name="identity+greedy_eft",
            policy_replicate=0,
            max_slots=slots,
            record_decisions=False,
        )
        random_metrics, _ = run_episode(
            shard=shard,
            episode_index=int(shard["episode_index"]),
            policy_name="identity+random",
            policy_replicate=0,
            max_slots=slots,
            record_decisions=False,
        )
        greedy_values.append(int(greedy["completed_dag_count"]))
        random_values.append(int(random_metrics["completed_dag_count"]))

    pairs = list(zip(greedy_values, random_values))
    mean_greedy = float(np.mean(greedy_values))
    mean_random = float(np.mean(random_values))
    improvement = (mean_greedy - mean_random) / max(mean_random, 1e-9)
    detail = f"greedy={greedy_values} random={random_values} mean {mean_greedy:.1f} vs {mean_random:.1f} (+{improvement:.1%})"
    _require(all(g >= r for g, r in pairs), f"anchor 6 failed: greedy lost on some scenario. {detail}")
    _require(improvement > 0.10, f"anchor 6 failed: mean improvement {improvement:.1%} <= 10%. {detail}")
    _ok("D9 greedy-EFT dominates random (anchor 6)", detail)


# --------------------------------------------------------------------------------------
# E. 锚点 1：与 static corpus 的 greedy_eft_uav_id 逐条一致
# --------------------------------------------------------------------------------------


def test_corpus_anchor(shards: list[dict[str, Any]]) -> None:
    if not ANCHOR_CORPUS_PATH.is_file():
        _skip(
            "E1 greedy-EFT vs static corpus (anchor 1)",
            f"{ANCHOR_CORPUS_PATH.relative_to(ROOT).as_posix()} not found",
        )
        return

    expected: dict[tuple[int, str], dict[str, Any]] = {}
    with ANCHOR_CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["slot_index"]) != 0 or int(record["decision_order"]) != 0:
                continue
            expected[(int(record["evaluation_scenario_seed"]), str(record["stable_task_id"]))] = record

    _require(bool(expected), "anchor corpus contains no slot 0 / decision_order 0 rows")

    matched = 0
    for shard in shards:
        _, decisions = run_episode(
            shard=shard,
            episode_index=int(shard["episode_index"]),
            policy_name="identity+greedy_eft",
            policy_replicate=0,
            max_slots=1,
            record_decisions=True,
            record_decisions_max_slot=0,
        )
        for record in decisions:
            if int(record["decision_order"]) != 0:
                continue
            key = (int(record["evaluation_scenario_seed"]), str(record["stable_task_id"]))
            reference = expected.get(key)
            if reference is None:
                continue
            _require(
                list(record["candidate_uav_ids"]) == list(reference["candidate_uav_ids"]),
                f"candidate id ordering differs at {key}",
            )
            _require(
                list(record["candidate_mask"]) == list(reference["candidate_mask"]),
                f"candidate mask differs at {key}",
            )
            for ours, theirs in zip(record["eft"], reference["eft"]):
                _require(float(ours) == float(theirs), f"EFT differs bitwise at {key}: {ours} vs {theirs}")
            _require(
                int(record["selected_uav_id"]) == int(reference["greedy_eft_uav_id"]),
                f"greedy choice differs at {key}: {record['selected_uav_id']} vs {reference['greedy_eft_uav_id']}",
            )
            matched += 1
    _require(matched > 0, "anchor corpus had no overlapping keys with the evaluated scenarios")
    _ok("E1 greedy-EFT + EFT vector match static corpus bitwise (anchor 1)", f"{matched} decisions")


# --------------------------------------------------------------------------------------


# A 通信代价代数 / B upward rank / C 钩子契约 / D episode 不变量
# R 可重复性 + replicate 独立性 + E_part 惰性 / G greedy vs random / E static corpus 锚点
SECTIONS = ("A", "B", "C", "D", "R", "G", "E")


def _load_shards(count: int, shard_dir: Path | None) -> list[dict[str, Any]]:
    """优先从已有 tape 目录读 shard；没有就在内存里现生成（每个约 12s）。"""
    shards: list[dict[str, Any]] = []
    for index in range(int(count)):
        candidate = (shard_dir / f"episode_{index:02d}.json") if shard_dir is not None else None
        if candidate is not None and candidate.is_file() and candidate.stat().st_size > 0:
            print(f"  loading {candidate.name} ...", flush=True)
            shards.append(load_scenario_shard(candidate))
        else:
            print(f"  generating episode {index:02d} in memory ...", flush=True)
            shards.append(generate_scenario_shard(index))
    return shards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke tests for the heuristic policy baseline")
    parser.add_argument("--episodes", type=int, default=3, help="number of tape episodes to use")
    parser.add_argument("--slots", type=int, default=200, help="physical slots per episode")
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="optional existing tape shards/ directory; missing shards are generated in memory",
    )
    parser.add_argument(
        "--warmup-slots",
        type=int,
        default=40,
        help="slots to advance with greedy before running the B/C hook tests (more concurrent DAGs)",
    )
    parser.add_argument(
        "--sections",
        default=",".join(SECTIONS),
        help="comma-separated subset of A,B,C,D,E (useful when wall-clock per invocation is limited)",
    )
    args = parser.parse_args(argv)
    selected = tuple(part.strip().upper() for part in str(args.sections).split(",") if part.strip())
    for part in selected:
        if part not in SECTIONS:
            raise ValueError(f"unknown section {part!r}; expected a subset of {SECTIONS}")

    started = time.time()
    print(f"preparing {args.episodes} scenario shard(s) ...", flush=True)
    shards = _load_shards(int(args.episodes), args.shard_dir)
    print(f"  done in {time.time() - started:.1f}s", flush=True)

    # E_part 对本框架无语义影响，关掉以避免每 5 slot spawn 一次 KaHyPar 子进程。
    config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = False

    if "A" in selected:
        print("\n[A] communication-cost algebra")
        test_factorization_is_exact()

    if "B" in selected or "C" in selected:
        warmup = int(args.warmup_slots)
        print(f"\n[B/C] upward rank + hook contracts (live env after {warmup} warm-up slots)")
        from environment.graph_builder import CleanGraphBuilder
        from environment.heuristic_policies import act_with_heuristic_policy
        from environment.stage1_temperature_diagnostic import Stage1TemperatureDiagnosticEnv
        from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state

        env = Stage1TemperatureDiagnosticEnv(scenario_shard=shards[0])
        builder = CleanGraphBuilder()
        env.reset()
        builder.reset()
        try:
            # 先用 greedy 推进若干时隙，让多个 DAG 并发存在，
            # 否则 slot 0 只有 1 个 DAG / 2 个 ready 任务，B/C 的覆盖度太弱。
            warm_policy = build_policy("identity+greedy_eft", geometry=UavGeometry.from_service_positions(env.uav_service_positions))
            prepared = None
            for slot_index in range(warmup):
                prepared = prepare_slot_state(env=env, graph_builder=builder)
                env.apply_movement({})
                ready = [env.task_manager.get_task(task_id) for task_id in prepared.frozen_ready_task_ids]
                ready = [task for task in ready if task is not None]
                if slot_index == warmup - 1:
                    break
                result = act_with_heuristic_policy(
                    policy=warm_policy,
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
                    evaluation_scenario_seed=int(shards[0]["evaluation_scenario_seed"]),
                    episode_index=0,
                    slot_index=slot_index,
                    policy_replicate=0,
                )
                env.commit_and_advance(assignment_buffer=result.assignments, offloading_skip_count=result.skip_count)
            if "B" in selected:
                test_upward_rank(env)
            if "C" in selected:
                test_order_policies(env, list(prepared.frozen_ready_task_ids))
        finally:
            builder.close()

    if "D" in selected:
        print("\n[D] end-to-end episode invariants")
        test_episode_invariants(shards[0], int(args.slots))

    if "R" in selected:
        print("\n[R] reproducibility")
        test_reproducibility(shards[0], int(args.slots))
        test_random_replicates_differ(shards[0], int(args.slots))
        test_partition_hyperedges_are_inert(shards[0])

    if "G" in selected:
        print("\n[G] greedy-EFT vs random sanity control")
        test_greedy_beats_random(shards, int(args.slots))

    if "E" in selected:
        print("\n[E] static corpus anchor")
        test_corpus_anchor(shards)

    print(f"\n{len(PASSED)} passed, {len(SKIPPED)} skipped in {time.time() - started:.1f}s")
    if SKIPPED:
        print("SKIPPED:")
        for name in SKIPPED:
            print(f"  - {name}")
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
