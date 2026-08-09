"""锚点 1 的**严格覆盖版**校验：static corpus vs greedy-EFT，逐场景零遗漏。

中文：`scripts/smoke_heuristic_policy_baseline.py --sections E` 已经做了逐条 bitwise
比对，但它的匹配是「按 `(seed, stable_task_id)` 查表，查不到就 `continue`」——
一旦我们复现出的 slot 0 首个任务与 corpus 记的不是同一个 `stable_task_id`，
它会**静默跳过**而不是失败，最后仍然 `matched > 0` 判 PASS。本脚本堵这个洞。

本脚本**只新增、不修改**任何已有文件，也不 import torch、不加载 checkpoint、不用 GPU。

----------------------------------------------------------------------------------
锚点文件的行数：**不是 20，是「文件里实际有多少场景」**
----------------------------------------------------------------------------------
static corpus 只记录「slot 0 有 ready 任务、且 ≥2 个合法候选」的决策。
20 个场景里有 6 个在 slot 0 完全没有 DAG 到达
（seed 424252 / 424253 / 424254 / 424255 / 424259 / 424260，即 episode
10 / 11 / 12 / 13 / 17 / 18），因此它们在 corpus 里没有 slot 0 记录。
按到达率算 P(slot 0 无到达) ≈ 0.17–0.42，实测 6/20 = 0.30，完全吻合。

所以期望场景数一律取自**文件本身**，绝不硬编码 20；缺场景不算失败。
但对文件中**存在**的每个场景，要求逐条完全一致，且必须**真的匹配上**
（present_matched 必须等于 present_in_corpus，不允许静默跳过）。
对文件中**缺失**的每个场景，必须给出可接受的解释：
slot 0 无 ready 任务，或所有 slot0/order0 决策的合法候选 < 2。
解释不出来的缺失 → FAIL（那说明是数据缺陷，不是采样效应）。

----------------------------------------------------------------------------------
上游前置检验：「首个决策与策略无关」是实证的，不是推理的
----------------------------------------------------------------------------------
锚点文件在服务器上由一次性提取脚本产出。该脚本对每个场景把 corpus 里全部 **15 条**
（3 checkpoint × 5 sampling replicate）slot0/order0 记录取出，比对
`candidate_uav_ids` / `candidate_mask` / `eft` / `greedy_eft_uav_id` / `stable_task_id`
五个字段，**只有当 15 条完全一致时才写出该场景**，否则整个锚点作废。自检输出：

    场景数 14，每场景记录数 [15]
    策略无关性检验通过：14 个场景内部全部一致

这条把「slot 0 第一个决策的状态完全由 tape 决定、与 checkpoint / replicate 无关」
从假设升级成了**实测事实**——因此拿它当训练无关的外部真值是站得住的。

provenance 的两处诚实说明：
1. 该检验**无法在本地复验**——源 corpus（`logs/stage1_temperature_followup/
   20260806_formal_v1/static_corpus/records.jsonl`，116,783 条）只在服务器上。
   本脚本只做记录与转述，不声称自己验过。
2. 提取脚本是临时文件，已在两侧清理，**不可再引用**。锚点与上游的绑定因此
   只剩「文件内容本身」——所以下面把 `corpus_sha256` 钉进 verdict，
   后续任何人拿到不同 sha 的锚点文件都必须重新走一遍上游检验。

用法：

    python scripts/smoke_heuristic_corpus_anchor_coverage.py \
        --shard-dir /path/to/tape/shards \
        --state-file /tmp/anchor_coverage_state.json \
        --budget-seconds 32

`--state-file` 用于分块续跑（单次调用有 wall-clock 上限时）：每次跑完把已完成的
episode 结果落盘，重复调用直到 `complete=true`，最后一次打印判定。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.stage1_temperature_sampling import file_sha256
from environment.stage1_temperature_tape import FORMAL_EPISODES, load_scenario_shard
from scripts.run_heuristic_policy_baseline import run_episode

ANCHOR_CORPUS_PATH = ROOT / "analysis_inbox" / "corpus_slot0_anchor.jsonl"

# corpus 与复现结果之间必须逐条 bitwise 相同的字段。
COMPARED_FIELDS = ("candidate_uav_ids", "candidate_mask", "eft")


def _load_corpus(path: Path) -> dict[int, dict[str, Any]]:
    """按 seed 索引 slot0/order0 记录；同一 seed 出现多次即为数据缺陷。"""
    by_seed: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["slot_index"]) != 0 or int(record["decision_order"]) != 0:
                raise AssertionError(f"{path.name}:{lineno} is not a slot0/order0 record")
            seed = int(record["evaluation_scenario_seed"])
            if seed in by_seed:
                raise AssertionError(f"{path.name}:{lineno} duplicate seed {seed} (extractor should dedup)")
            by_seed[seed] = record
    if not by_seed:
        raise AssertionError(f"{path} contains no slot0/order0 rows")
    return by_seed


def _replay_slot0(shard: dict[str, Any], episode_index: int) -> list[dict[str, Any]]:
    """用 greedy-EFT 复现该场景的 slot 0，返回全部 slot 0 决策记录。"""
    _, decisions = run_episode(
        shard=shard,
        episode_index=int(episode_index),
        policy_name="identity+greedy_eft",
        policy_replicate=0,
        max_slots=1,
        record_decisions=True,
        record_decisions_max_slot=0,
    )
    return list(decisions)


def _check_episode(
    *, episode_index: int, shard: dict[str, Any], corpus: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    seed = int(shard["evaluation_scenario_seed"])
    decisions = _replay_slot0(shard, episode_index)
    first = [record for record in decisions if int(record["decision_order"]) == 0]
    reference = corpus.get(seed)

    result: dict[str, Any] = {
        "episode_index": int(episode_index),
        "evaluation_scenario_seed": seed,
        "in_corpus": reference is not None,
        "slot0_decision_count": len(decisions),
        "status": None,
        "detail": None,
    }

    if reference is None:
        # 缺失场景必须能被解释，否则判失败。
        if not decisions:
            result["status"] = "absent_no_ready_task"
            result["detail"] = "slot 0 produced no offloading decision (no DAG arrival)"
            return result
        legal_counts = [sum(1 for flag in rec["candidate_mask"] if flag) for rec in first]
        if first and all(count < 2 for count in legal_counts):
            result["status"] = "absent_fewer_than_two_legal_candidates"
            result["detail"] = f"legal candidate counts {legal_counts}"
            return result
        result["status"] = "ABSENT_UNEXPLAINED"
        result["detail"] = (
            f"replay produced {len(first)} order-0 decision(s) with legal counts "
            f"{legal_counts} but corpus has no row for seed {seed}"
        )
        return result

    # 在 corpus 里的场景：必须真的匹配上，不允许静默跳过。
    if not first:
        result["status"] = "MISSING_REPLAY_DECISION"
        result["detail"] = "corpus has a row but replay produced no order-0 decision"
        return result
    if len(first) != 1:
        result["status"] = "MULTIPLE_ORDER0_DECISIONS"
        result["detail"] = f"replay produced {len(first)} order-0 decisions"
        return result

    ours = first[0]
    if str(ours["stable_task_id"]) != str(reference["stable_task_id"]):
        result["status"] = "STABLE_TASK_ID_MISMATCH"
        result["detail"] = f"{ours['stable_task_id']!r} vs corpus {reference['stable_task_id']!r}"
        return result

    for field in COMPARED_FIELDS:
        mine, theirs = list(ours[field]), list(reference[field])
        if len(mine) != len(theirs):
            result["status"] = f"{field.upper()}_LENGTH_MISMATCH"
            result["detail"] = f"{len(mine)} vs {len(theirs)}"
            return result
        for index, (a, b) in enumerate(zip(mine, theirs)):
            # eft 是浮点，这里刻意用 == 做 bitwise 比较（与 --sections E 同口径）。
            if (float(a) != float(b)) if field == "eft" else (a != b):
                result["status"] = f"{field.upper()}_MISMATCH"
                result["detail"] = f"index {index}: {a!r} vs {b!r}"
                return result

    if int(ours["selected_uav_id"]) != int(reference["greedy_eft_uav_id"]):
        result["status"] = "GREEDY_CHOICE_MISMATCH"
        result["detail"] = f"{ours['selected_uav_id']} vs corpus {reference['greedy_eft_uav_id']}"
        return result

    result["status"] = "matched"
    result["detail"] = (
        f"task={ours['stable_task_id']} uav={ours['selected_uav_id']} "
        f"legal={sum(1 for flag in ours['candidate_mask'] if flag)}/{len(ours['candidate_mask'])}"
    )
    return result


def _verdict(results: list[dict[str, Any]], corpus: dict[int, dict[str, Any]]) -> dict[str, Any]:
    evaluated_seeds = {int(item["evaluation_scenario_seed"]) for item in results}
    present = [item for item in results if item["in_corpus"]]
    absent = [item for item in results if not item["in_corpus"]]
    matched = [item for item in present if item["status"] == "matched"]
    failures = [item for item in results if str(item["status"]).isupper()]

    corpus_seeds_not_evaluated = sorted(set(corpus) - evaluated_seeds)

    return {
        "schema": "heuristic_corpus_anchor_coverage_v1",
        "anchor_corpus_path": ANCHOR_CORPUS_PATH.relative_to(ROOT).as_posix(),
        # 提取脚本是临时文件、已清理；锚点与上游的唯一绑定就是这个 sha256。
        "corpus_sha256": file_sha256(ANCHOR_CORPUS_PATH),
        "corpus_scenario_count": len(corpus),
        "corpus_scenarios": sorted(corpus),
        "evaluated_episode_count": len(results),
        # 期望场景数取自文件本身，不硬编码 20。
        "expected_matches": len(present),
        "actual_matches": len(matched),
        "absent_from_corpus": sorted(item["evaluation_scenario_seed"] for item in absent),
        "absent_reasons": {
            str(item["evaluation_scenario_seed"]): item["status"] for item in absent
        },
        "corpus_seeds_not_evaluated": corpus_seeds_not_evaluated,
        "failures": failures,
        "policy_independence_precheck": {
            "verified_by": "one-off server-side extraction script (temporary, since deleted)",
            "self_check_output": (
                "场景数 14，每场景记录数 [15] / "
                "策略无关性检验通过：14 个场景内部全部一致"
            ),
            "source_corpus": (
                "logs/stage1_temperature_followup/20260806_formal_v1/static_corpus/records.jsonl"
                " (server only)"
            ),
            "records_per_scenario": 15,
            "decomposition": "3 checkpoints x 5 sampling replicates",
            "fields_required_identical": [
                "candidate_uav_ids",
                "candidate_mask",
                "eft",
                "greedy_eft_uav_id",
                "stable_task_id",
            ],
            "outcome": "all scenarios internally identical; extractor refuses to emit otherwise",
            "implication": (
                "the slot-0 first decision is empirically independent of checkpoint and "
                "sampling replicate, so it is a legitimate training-free external ground truth"
            ),
            "locally_reverifiable": False,
            "note": (
                "source corpus lives only on the server and the extractor has been cleaned up; "
                "this field records the upstream check rather than re-performing it. "
                "corpus_sha256 is the only remaining binding to that check -- a different sha "
                "means the upstream verification must be redone."
            ),
        },
        "passed": (
            not failures
            and len(matched) == len(present)
            and not corpus_seeds_not_evaluated
            and len(present) > 0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict coverage check for anchor 1")
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=FORMAL_EPISODES)
    parser.add_argument("--state-file", type=Path, default=None, help="resume file for chunked runs")
    parser.add_argument("--budget-seconds", type=float, default=0.0, help="0 = run everything in one go")
    parser.add_argument("--summary-out", type=Path, default=None, help="create-only path for the verdict JSON")
    args = parser.parse_args(argv)

    corpus = _load_corpus(ANCHOR_CORPUS_PATH)
    print(
        f"anchor corpus: {len(corpus)} scenario(s) "
        f"(episodes {sorted(seed - 424242 for seed in corpus)})",
        flush=True,
    )

    done: dict[str, dict[str, Any]] = {}
    if args.state_file is not None and args.state_file.is_file():
        done = {str(k): v for k, v in json.loads(args.state_file.read_text(encoding="utf-8")).items()}

    started = time.time()
    budget = float(args.budget_seconds)
    for episode_index in range(int(args.episodes)):
        if str(episode_index) in done:
            continue
        if budget > 0.0 and (time.time() - started) > budget:
            break
        shard = load_scenario_shard(args.shard_dir / f"episode_{episode_index:02d}.json")
        result = _check_episode(episode_index=episode_index, shard=shard, corpus=corpus)
        done[str(episode_index)] = result
        flag = "FAIL" if str(result["status"]).isupper() else "ok  "
        print(
            f"  {flag} ep{episode_index:02d} seed={result['evaluation_scenario_seed']} "
            f"{result['status']}  {result['detail']}",
            flush=True,
        )
        del shard

    if args.state_file is not None:
        args.state_file.write_text(json.dumps(done, indent=2, sort_keys=True), encoding="utf-8")

    complete = len(done) == int(args.episodes)
    if not complete:
        print(f"\nINCOMPLETE {len(done)}/{args.episodes} episodes; rerun to resume", flush=True)
        return 2

    results = [done[str(index)] for index in range(int(args.episodes))]
    verdict = _verdict(results, corpus)
    verdict["per_episode"] = results
    print("\n" + json.dumps({k: v for k, v in verdict.items() if k != "per_episode"}, indent=2, sort_keys=True))

    if args.summary_out is not None:
        if args.summary_out.exists():
            raise FileExistsError("summary output is create-only")
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")

    if not verdict["passed"]:
        print("\nANCHOR 1 COVERAGE FAIL")
        return 1
    print(
        f"\nANCHOR 1 COVERAGE PASS "
        f"({verdict['actual_matches']}/{verdict['expected_matches']} corpus scenarios matched bitwise, "
        f"{len(verdict['absent_from_corpus'])} absent and explained)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
