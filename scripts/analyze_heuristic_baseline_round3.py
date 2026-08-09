"""第 3 轮启发式基线分析：逐场景配对对照 / 难度分层 / 顺序分歧度 / 辅助指标。

中文：只读输入、只写 create-only 输出，不训练、不加载 checkpoint、不改任何已有文件。

输入
----
- 启发式逐 episode：`--episodes-glob`（`run_heuristic_policy_baseline.py` 的
  `closed_loop/episodes.jsonl`，160 行 = 20 场景 × {random×5, greedy_eft, heft_ret0, heft_ret1}）
- RL 对照基准：`--rl-episodes`（`analysis_inbox/round2/e2/closed_loop/episodes.jsonl`，
  只取 `temperature == 1.0` 的 300 行 = 20 场景 × 3 seed × 5 replicate）
- 决策轨迹：`--decisions-glob`（三个确定性策略的 `decisions/records.jsonl`）

口径（按 CLAUDE.md）
--------------------
- 主指标 **`completed_dag_count`**。`dag_completion_rate` 只作上下文列出并标注污染，
  因为分母 `generated_dag_count` 依赖策略（完成越快 → UE 越早解除 active-DAG cap → 生成越多）。
- 所有对照在**同一条冻结 tape** 上，以 **tape episode（场景）为配对单位**做配对 bootstrap。
- 两层重采样：外层对 20 个场景有放回抽样；内层对该场景内的 RL 15 个值、
  random 5 个 replicate 有放回抽样（确定性策略贡献其单值）。这样同时吃进
  场景间方差与场景内采样噪声。

已知陷阱（结果里会显式标注）
----------------------------
`average_dag_flowtime` 只在**已完成**的 DAG 上取平均，因此它的样本集合依赖策略：
一个只完成了简单 DAG 的弱策略会得到虚低的 flowtime。random vs greedy 之间
（58 vs 147 完成数）这个偏差极大，**不能当作 random 更快的证据**。
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_TAPE_SHA = "cf0086e047e33931a867ff94104f4615627164d1afb29bbd6b7c9b133bbfacf4"
POLICIES = ("identity+greedy_eft", "heft_ret0+greedy_eft", "heft_ret1+greedy_eft", "identity+random")
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_DRAWS = 20000


# --------------------------------------------------------------------------------------
# 载入
# --------------------------------------------------------------------------------------


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_heuristic(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(globlib.glob(pattern)):
        rows.extend(_read_jsonl(path))
    if not rows:
        raise SystemExit(f"no heuristic rows matched {pattern!r}")
    tapes = {row["logical_tape_sha256"] for row in rows}
    if tapes != {EXPECTED_TAPE_SHA}:
        raise SystemExit(f"heuristic rows are not all on the frozen tape: {tapes}")
    if any(int(row["invalid_assignment_count"]) != 0 for row in rows):
        raise SystemExit("heuristic rows contain invalid assignments")
    if not all(bool(row["finite"]) for row in rows):
        raise SystemExit("heuristic rows contain non-finite rewards")
    return rows


def load_rl(path: str | Path) -> list[dict[str, Any]]:
    rows = [row for row in _read_jsonl(path) if float(row.get("temperature", -1.0)) == 1.0]
    if not rows:
        raise SystemExit("no RL rows at temperature 1.0")
    tapes = {row["logical_tape_sha256"] for row in rows}
    if tapes != {EXPECTED_TAPE_SHA}:
        raise SystemExit(f"RL rows are not on the frozen tape: {tapes}")
    return rows


# --------------------------------------------------------------------------------------
# 配对 bootstrap
# --------------------------------------------------------------------------------------


def paired_bootstrap(
    *,
    treatment: dict[int, list[float]],
    control: dict[int, list[float]],
    rng: np.random.Generator,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """外层抽场景、内层抽该场景内的重复观测；返回 mean(treatment - control) 的分布。"""
    scenarios = sorted(set(treatment) & set(control))
    if not scenarios:
        raise ValueError("no shared scenarios")
    point = float(
        np.mean([statistics.mean(treatment[s]) - statistics.mean(control[s]) for s in scenarios])
    )

    def resampled_means(values: list[float]) -> np.ndarray:
        """(draws,) 的内层 bootstrap 均值；单值（确定性策略）直接广播常数。"""
        array = np.asarray(values, dtype=float)
        if array.size == 1:
            return np.full(draws, float(array[0]))
        idx = rng.integers(0, array.size, size=(draws, array.size))
        return array[idx].mean(axis=1)

    # delta[s, draw]：第 s 个场景在第 draw 次抽样下的 treatment-control 差
    delta = np.stack(
        [resampled_means(treatment[s]) - resampled_means(control[s]) for s in scenarios], axis=0
    )
    # 外层：对场景有放回抽样，每次抽 len(scenarios) 个
    picks = rng.integers(0, len(scenarios), size=(draws, len(scenarios)))
    samples = delta[picks, np.arange(draws)[:, None]].mean(axis=1)

    low, high = np.percentile(samples, [2.5, 97.5])
    wins = sum(
        1 for s in scenarios if statistics.mean(treatment[s]) > statistics.mean(control[s])
    )
    return {
        "n_scenarios": len(scenarios),
        "mean_delta": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "scenarios_treatment_wins": wins,
        "scenarios_control_wins": len(scenarios) - wins,
        "bootstrap_draws": int(draws),
    }


# --------------------------------------------------------------------------------------
# 顺序分歧度：Kendall tau-b + 逆序对比例
# --------------------------------------------------------------------------------------


def kendall_tau_b(order_values: list[float]) -> tuple[float, int, int, int]:
    """输入 = 按实际执行顺序排列的 rank_u 序列。

    与「rank_u 降序」这一目标排列比较：
      concordant = 前面的 rank_u 更大（与 HEFT 一致）
      discordant = 前面的 rank_u 更小（HEFT 会把它们换过来）
    返回 (tau_b, concordant, discordant, ties)。
    """
    n = len(order_values)
    concordant = discordant = ties = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = order_values[i], order_values[j]
            if a > b:
                concordant += 1
            elif a < b:
                discordant += 1
            else:
                ties += 1
    total = concordant + discordant
    tau = (concordant - discordant) / total if total else float("nan")
    return tau, concordant, discordant, ties


def order_divergence(
    records: list[dict[str, Any]], per_slot_sink: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    slots: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        rank = record.get("rank_u")
        if rank is None or not math.isfinite(float(rank)):
            raise SystemExit("decision record without a finite rank_u")
        slots[(int(record["episode_index"]), int(record["slot_index"]))].append(
            (int(record["decision_order"]), float(rank))
        )

    taus: list[float] = []
    sizes: list[int] = []
    total_c = total_d = total_t = 0
    identical = 0
    multi = 0
    for key in sorted(slots):
        entries = sorted(slots[key])
        if [order for order, _ in entries] != list(range(len(entries))):
            raise SystemExit(f"decision_order is not 0..n-1 at {key}")
        if len(entries) < 2:
            continue
        multi += 1
        values = [rank for _, rank in entries]
        tau, c, d, t = kendall_tau_b(values)
        taus.append(tau)
        sizes.append(len(entries))
        total_c += c
        total_d += d
        total_t += t
        if d == 0:
            identical += 1
        if per_slot_sink is not None:
            per_slot_sink.append(
                {
                    "policy_name": str(records[0]["policy_name"]),
                    "episode_index": key[0],
                    "slot_index": key[1],
                    "ready_set_size": len(entries),
                    "concordant_pairs": c,
                    "discordant_pairs": d,
                    "tied_pairs": t,
                    "kendall_tau_b": tau,
                }
            )

    pairs = total_c + total_d
    return {
        "slots_with_at_least_two_decisions": multi,
        "slots_already_in_rank_order": identical,
        "fraction_slots_already_in_rank_order": (identical / multi) if multi else None,
        "mean_kendall_tau_b_per_slot": float(np.mean(taus)) if taus else None,
        "median_kendall_tau_b_per_slot": float(np.median(taus)) if taus else None,
        "pair_weighted_kendall_tau_b": ((total_c - total_d) / pairs) if pairs else None,
        "discordant_pair_fraction": (total_d / pairs) if pairs else None,
        "concordant_pairs": total_c,
        "discordant_pairs": total_d,
        "tied_rank_pairs": total_t,
        "mean_ready_set_size": float(np.mean(sizes)) if sizes else None,
        "max_ready_set_size": int(max(sizes)) if sizes else None,
    }


# --------------------------------------------------------------------------------------


def collect(rows: list[dict[str, Any]], metric: str, policy: str | None = None) -> dict[int, list[float]]:
    out: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if policy is not None and row.get("policy_name") != policy:
            continue
        value = row.get(metric)
        if value is None:
            continue
        out[int(row["episode_index"])].append(float(value))
    return dict(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Round 3 heuristic baseline analysis")
    parser.add_argument("--episodes-glob", required=True)
    parser.add_argument("--rl-episodes", required=True)
    parser.add_argument("--decisions-glob", required=True)
    parser.add_argument("--out", type=Path, required=True, help="create-only summary JSON")
    parser.add_argument(
        "--divergence-out",
        type=Path,
        default=None,
        help="create-only per-slot Kendall tau JSONL (small; lets the order claim be audited "
        "without keeping the ~42 MB decision trace)",
    )
    args = parser.parse_args(argv)

    if args.out.exists():
        raise FileExistsError("summary output is create-only")

    heuristic = load_heuristic(args.episodes_glob)
    rl = load_rl(args.rl_episodes)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rl_completed = collect(rl, "completed_dag_count")
    rl_scenario_mean = {s: statistics.mean(v) for s, v in rl_completed.items()}

    # ---- 1. 逐场景配对对照 -----------------------------------------------------------
    per_scenario: dict[str, Any] = {}
    for scenario in sorted(rl_scenario_mean):
        entry: dict[str, Any] = {"rl_mean": round(rl_scenario_mean[scenario], 3)}
        for policy in POLICIES:
            values = collect(heuristic, "completed_dag_count", policy).get(scenario, [])
            if not values:
                continue
            mean = statistics.mean(values)
            entry[policy] = {
                "mean": mean,
                "n": len(values),
                "delta_vs_rl": round(mean - rl_scenario_mean[scenario], 3),
                "values": values if len(values) > 1 else None,
            }
        per_scenario[str(scenario)] = entry

    paired: dict[str, Any] = {}
    for policy in POLICIES:
        paired[policy] = paired_bootstrap(
            treatment=collect(heuristic, "completed_dag_count", policy),
            control=rl_completed,
            rng=rng,
        )

    # greedy vs 两个 HEFT 变体（回答「决策顺序有没有 leverage」的直接对照）
    head_to_head: dict[str, Any] = {}
    for policy in ("heft_ret0+greedy_eft", "heft_ret1+greedy_eft", "identity+random"):
        head_to_head[f"{policy}_vs_identity+greedy_eft"] = paired_bootstrap(
            treatment=collect(heuristic, "completed_dag_count", policy),
            control=collect(heuristic, "completed_dag_count", "identity+greedy_eft"),
            rng=rng,
        )

    # ---- 2. 难度分层 ------------------------------------------------------------------
    ordered = sorted(rl_scenario_mean, key=lambda s: rl_scenario_mean[s])
    tiers = {"hard": ordered[:7], "medium": ordered[7:13], "easy": ordered[13:]}
    stratified: dict[str, Any] = {}
    for name, members in tiers.items():
        block: dict[str, Any] = {
            "scenarios": sorted(members),
            "rl_range": [
                round(min(rl_scenario_mean[s] for s in members), 2),
                round(max(rl_scenario_mean[s] for s in members), 2),
            ],
            "rl_mean": round(statistics.mean(rl_scenario_mean[s] for s in members), 2),
        }
        for policy in POLICIES:
            values = collect(heuristic, "completed_dag_count", policy)
            means = [statistics.mean(values[s]) for s in members if s in values]
            deltas = [
                statistics.mean(values[s]) - rl_scenario_mean[s] for s in members if s in values
            ]
            block[policy] = {
                "mean": round(statistics.mean(means), 2),
                "mean_delta_vs_rl": round(statistics.mean(deltas), 2),
                "relative_vs_rl": round(
                    statistics.mean(means) / statistics.mean(rl_scenario_mean[s] for s in members), 4
                ),
                "scenarios_beating_rl": sum(1 for d in deltas if d > 0),
            }
        stratified[name] = block

    # ---- 2b. 难度 × 优势 的交叉点 ------------------------------------------------------
    # 「greedy 在宽松场景赢、在拥堵场景输」如果成立，delta 应随 RL 基准单调上升，
    # 并存在一个 RL 值使 delta 过零。这里给出相关系数（带 bootstrap CI）与 OLS 交叉点。
    def _rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            shared = (position + end) / 2.0 + 1.0
            for k in range(position, end + 1):
                ranks[order[k]] = shared
            position = end + 1
        return ranks

    crossover: dict[str, Any] = {}
    for policy in POLICIES:
        values = collect(heuristic, "completed_dag_count", policy)
        scen = sorted(set(values) & set(rl_scenario_mean))
        x = np.asarray([rl_scenario_mean[s] for s in scen], dtype=float)
        y = np.asarray([statistics.mean(values[s]) - rl_scenario_mean[s] for s in scen], dtype=float)
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = float(np.corrcoef(_rank(list(x)), _rank(list(y)))[0, 1])
        slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
        boot = np.empty(BOOTSTRAP_DRAWS, dtype=float)
        picks = rng.integers(0, len(scen), size=(BOOTSTRAP_DRAWS, len(scen)))
        for draw in range(BOOTSTRAP_DRAWS):
            idx = picks[draw]
            xs, ys = x[idx], y[idx]
            boot[draw] = np.corrcoef(xs, ys)[0, 1] if xs.std() > 0 and ys.std() > 0 else np.nan
        finite = boot[np.isfinite(boot)]
        crossover[policy] = {
            "pearson_r_delta_vs_rl_difficulty": round(pearson, 4),
            "spearman_rho": round(spearman, 4),
            "pearson_ci95": [
                round(float(np.percentile(finite, 2.5)), 4),
                round(float(np.percentile(finite, 97.5)), 4),
            ],
            "ols_slope": round(slope, 4),
            "ols_intercept": round(intercept, 4),
            "crossover_rl_value": round(-intercept / slope, 2) if slope != 0.0 else None,
        }

    # ---- 3. 顺序分歧度 ----------------------------------------------------------------
    by_policy_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(globlib.glob(args.decisions_glob)):
        for record in _read_jsonl(path):
            by_policy_records[str(record["policy_name"])].append(record)
    per_slot: list[dict[str, Any]] = []
    divergence = {
        policy: order_divergence(recs, per_slot_sink=per_slot)
        for policy, recs in sorted(by_policy_records.items())
    }
    if args.divergence_out is not None:
        if args.divergence_out.exists():
            raise FileExistsError("divergence output is create-only")
        args.divergence_out.parent.mkdir(parents=True, exist_ok=True)
        with args.divergence_out.open("x", encoding="utf-8", newline="\n") as handle:
            for row in sorted(
                per_slot, key=lambda r: (r["policy_name"], r["episode_index"], r["slot_index"])
            ):
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    # ---- 4. 辅助指标 ------------------------------------------------------------------
    auxiliary: dict[str, Any] = {}
    for metric in ("average_dag_flowtime", "admitted_incomplete_backlog", "generated_dag_count"):
        control = collect(rl, metric)
        if not control:
            continue
        auxiliary[metric] = {
            policy: paired_bootstrap(
                treatment=collect(heuristic, metric, policy), control=control, rng=rng
            )
            for policy in POLICIES
        }

    summary = {
        "schema": "heuristic_baseline_round3_analysis_v1",
        "logical_tape_sha256": EXPECTED_TAPE_SHA,
        "primary_metric": "completed_dag_count",
        "heuristic_rows": len(heuristic),
        "rl_rows_at_T1": len(rl),
        "rl_global_mean": round(statistics.mean(rl_scenario_mean.values()), 4),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "pairing_unit": "tape episode (scenario)",
            "scheme": "two-level: outer resample of scenarios, inner resample of within-scenario replicates",
        },
        "per_scenario": per_scenario,
        "paired_vs_rl": paired,
        "head_to_head": head_to_head,
        "difficulty_strata": stratified,
        "difficulty_crossover": crossover,
        "order_divergence": divergence,
        "auxiliary": auxiliary,
        "caveats": {
            "average_dag_flowtime": (
                "averaged over COMPLETED DAGs only, so the sample set is policy-dependent; a weak "
                "policy that only finishes easy DAGs gets an artificially low flowtime. Not "
                "comparable across policies with very different completion counts."
            ),
            "dag_completion_rate": (
                "deliberately not used as a headline metric: the denominator generated_dag_count is "
                "policy-dependent (faster completion releases the per-UE active-DAG cap sooner)."
            ),
            "rank_u_convention": (
                "rank_u in the decision trace is always the include_return=True analysis yardstick, "
                "so tau for heft_ret0 measures ret0-vs-ret1 divergence, not a bug."
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_scenario",)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
