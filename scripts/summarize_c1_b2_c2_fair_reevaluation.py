from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


ARMS = ("B1", "B2", "C2A", "C2B", "C2", "C1")
SEEDS = (5, 86, 617)
PROTOCOLS = ("joint", "forced_hover")
CHECKPOINT_LABELS = ("budget_selected", "final_ep0500")
TEST_METRICS = (
    "J_episode",
    "J_per_offer",
    "J_delay_component",
    "J_task_energy_component",
    "J_move_energy_component",
    "J_per_offer_delay_component",
    "J_per_offer_task_energy_component",
    "J_per_offer_move_energy_component",
    "N_offer",
    "N_admitted",
    "N_completed",
    "admission_rate",
    "conditional_completion_rate",
    "end_to_end_completion_rate",
    "completed_DAG_flowtime_mean",
    "completed_DAG_flowtime_median",
    "throughput",
    "task_energy_joules_total",
    "move_energy_joules_total",
    "energy_per_completed_DAG",
    "hover_action_ratio",
)
LOWER_IS_BETTER = {
    "J_episode",
    "J_per_offer",
    "J_delay_component",
    "J_task_energy_component",
    "J_move_energy_component",
    "J_per_offer_delay_component",
    "J_per_offer_task_energy_component",
    "J_per_offer_move_energy_component",
    "completed_DAG_flowtime_mean",
    "completed_DAG_flowtime_median",
    "task_energy_joules_total",
    "move_energy_joules_total",
    "energy_per_completed_DAG",
}
CONTRASTS: tuple[tuple[str, str, dict[str, float]], ...] = (
    (
        "movement_teacher_average_main_effect",
        "移动老师平均主效应",
        {"C2B": 0.5, "B2": -0.5, "C2": 0.5, "C2A": -0.5},
    ),
    (
        "offloading_teacher_average_main_effect",
        "卸载老师平均主效应",
        {"C2A": 0.5, "B2": -0.5, "C2": 0.5, "C2B": -0.5},
    ),
    (
        "teacher_interaction",
        "教师交互效应",
        {"C2": 1.0, "C2A": -1.0, "C2B": -1.0, "B2": 1.0},
    ),
    (
        "movement_teacher_conditional_effect_given_offloading_teacher",
        "已有卸载老师时增加移动老师的条件效应",
        {"C2": 1.0, "C2A": -1.0},
    ),
    (
        "offloading_teacher_conditional_effect_given_movement_teacher",
        "已有移动老师时增加卸载老师的条件效应",
        {"C2": 1.0, "C2B": -1.0},
    ),
    (
        "complete_reward_definition_effect",
        "完整奖励定义的效应",
        {"C2": 1.0, "C1": -1.0},
    ),
    (
        "staged_accounting_effect_same_realized_cost",
        "同一实际代价的分期记账效应",
        {"B2": 1.0, "B1": -1.0},
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--lambda-move-sweep", type=str, default="0.10,0.04,0.0")
    return parser.parse_args(argv)


def _lambda_values(spec: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in spec.split(",") if token.strip())
    if not values or any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("--lambda-move-sweep requires finite nonnegative values")
    return values


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def quantile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [quantile(ordered, 0.025), quantile(ordered, 0.975)]


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": float(statistics.median(values)),
        "min": min(values),
        "max": max(values),
    }


def _hierarchical_ci(
    values_by_seed: dict[int, list[float]], *, replicates: int, rng: random.Random
) -> list[float]:
    samples: list[float] = []
    seeds = tuple(values_by_seed)
    for _ in range(replicates):
        sampled_seed_means: list[float] = []
        for sampled_seed in (rng.choice(seeds) for _ in seeds):
            values = values_by_seed[sampled_seed]
            sampled_seed_means.append(mean([rng.choice(values) for _ in values]))
        samples.append(mean(sampled_seed_means))
    return interval(samples)


def _strip_eval_row(row: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    result = {key: row[key] for key in TEST_METRICS}
    result.update(
        {
            "delay_seconds_total": row["delay_seconds_total"],
            "task_energy_joules_total": row["task_energy_joules_total"],
            "move_energy_joules_total": row["move_energy_joules_total"],
        }
    )
    result.update(
        {
            "arm": row["arm"],
            "model_seed": row["model_seed"],
            "tape_id": row["tape_id"],
            "tape_path": row["tape_path"],
            "protocol": row["protocol"],
            "checkpoint_label": row["checkpoint_label"],
            "raw_result_path": str(raw_path),
        }
    )
    return result


def _load_test_rows(
    manifest: dict[str, Any], *, evaluation_root: Path
) -> tuple[dict[tuple[str, int, str, str], list[dict[str, Any]]], dict[str, Any]]:
    rows: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    raw_paths: list[str] = []
    cost_identity_max_abs_error = 0.0
    for arm in ARMS:
        for seed in SEEDS:
            for checkpoint_label in CHECKPOINT_LABELS:
                for protocol in PROTOCOLS:
                    path = (
                        evaluation_root
                        / "test"
                        / f"{arm}_seed{seed}_{checkpoint_label}_{protocol}.json"
                    )
                    payload = read_json(path)
                    if payload.get("status") != "completed" or len(payload["rows"]) != 50:
                        raise RuntimeError(f"incomplete test result: {path}")
                    cleaned: list[dict[str, Any]] = []
                    for row in payload["rows"]:
                        component_error = abs(
                            float(row["J_episode"])
                            - float(row["J_delay_component"])
                            - float(row["J_task_energy_component"])
                            - float(row["J_move_energy_component"])
                        )
                        per_offer_error = abs(
                            float(row["J_per_offer"])
                            - float(row["J_episode"]) / max(int(row["N_offer"]), 1)
                        )
                        cost_identity_max_abs_error = max(
                            cost_identity_max_abs_error, component_error, per_offer_error
                        )
                        if protocol == "forced_hover" and float(row["hover_action_ratio"]) != 1.0:
                            raise AssertionError(
                                f"forced_hover hover ratio mismatch: {arm} seed {seed}"
                            )
                        cleaned.append(_strip_eval_row(row, path))
                    rows[(arm, seed, checkpoint_label, protocol)] = cleaned
                    raw_paths.append(str(path))
    if cost_identity_max_abs_error > 1e-9:
        raise RuntimeError(f"cost identity error {cost_identity_max_abs_error}")

    selections = manifest["checkpoint_selections"]
    validation_paths: list[str] = []
    for arm in ARMS:
        for seed in SEEDS:
            selected = selections[arm][str(seed)]
            if selected["selection_protocol"] != "joint":
                raise AssertionError("checkpoint selection did not use joint validation")
            if "_joint.json" not in selected["selection_source"]:
                raise AssertionError("selection_source is not a joint validation file")
            sources = list(selected["candidate_sources"])
            if len(sources) != 5:
                raise AssertionError("selection did not read all five candidate checkpoints")
            for source in sources:
                if not Path(source).is_file():
                    raise FileNotFoundError(source)
            validation_paths.extend(sources)
    return rows, {
        "raw_test_paths": raw_paths,
        "raw_validation_paths": validation_paths,
        "test_file_count": len(raw_paths),
        "validation_file_count": len(validation_paths),
        "test_row_count": sum(len(value) for value in rows.values()),
        "cost_identity_max_abs_error": cost_identity_max_abs_error,
    }


def _arm_summaries(
    rows: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    *,
    replicates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_summaries: dict[str, Any] = {}
    aggregate_summaries: dict[str, Any] = {}
    for checkpoint_index, checkpoint_label in enumerate(CHECKPOINT_LABELS):
        seed_summaries[checkpoint_label] = {}
        aggregate_summaries[checkpoint_label] = {}
        for protocol_index, protocol in enumerate(PROTOCOLS):
            seed_summaries[checkpoint_label][protocol] = {}
            aggregate_summaries[checkpoint_label][protocol] = {}
            for arm_index, arm in enumerate(ARMS):
                by_seed: dict[int, dict[str, Any]] = {}
                for seed in SEEDS:
                    arm_rows = rows[(arm, seed, checkpoint_label, protocol)]
                    by_seed[seed] = {
                        metric: summarize_values(
                            [float(row[metric]) for row in arm_rows]
                        )
                        for metric in TEST_METRICS
                    }
                seed_summaries[checkpoint_label][protocol][arm] = {
                    str(seed): value for seed, value in by_seed.items()
                }
                aggregate: dict[str, Any] = {}
                for metric_index, metric in enumerate(TEST_METRICS):
                    values_by_seed = {
                        seed: [
                            float(row[metric])
                            for row in rows[(arm, seed, checkpoint_label, protocol)]
                        ]
                        for seed in SEEDS
                    }
                    seed_means = [mean(values_by_seed[seed]) for seed in SEEDS]
                    rng = random.Random(
                        20260915
                        + checkpoint_index * 100_003
                        + protocol_index * 1_000_003
                        + arm_index * 10_000_019
                        + metric_index * 1009
                    )
                    aggregate[metric] = {
                        "mean_of_seed_means": mean(seed_means),
                        "seed_means": seed_means,
                        "hierarchical_bootstrap_95_ci": _hierarchical_ci(
                            values_by_seed, replicates=replicates, rng=rng
                        ),
                    }
                aggregate_summaries[checkpoint_label][protocol][arm] = aggregate
    return seed_summaries, aggregate_summaries


def _contrast_summaries(
    rows: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    *,
    replicates: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for checkpoint_index, checkpoint_label in enumerate(CHECKPOINT_LABELS):
        results[checkpoint_label] = {}
        for protocol_index, protocol in enumerate(PROTOCOLS):
            protocol_result: dict[str, Any] = {}
            for contrast_index, (name, label, coefficients) in enumerate(CONTRASTS):
                metric_result: dict[str, Any] = {}
                for metric_index, metric in enumerate(TEST_METRICS):
                    values_by_seed: dict[int, list[float]] = {}
                    for seed in SEEDS:
                        by_arm_tape = {
                            arm: {
                                int(row["tape_id"]): float(row[metric])
                                for row in rows[(arm, seed, checkpoint_label, protocol)]
                            }
                            for arm in coefficients
                        }
                        tape_ids = sorted(next(iter(by_arm_tape.values())))
                        values_by_seed[seed] = [
                            sum(
                                coefficient * by_arm_tape[arm][tape_id]
                                for arm, coefficient in coefficients.items()
                            )
                            for tape_id in tape_ids
                        ]
                    seed_means = [mean(values_by_seed[seed]) for seed in SEEDS]
                    rng = random.Random(
                        20260915
                        + checkpoint_index * 100_003
                        + protocol_index * 1_000_003
                        + contrast_index * 10_000_019
                        + metric_index * 1009
                    )
                    favorable = [
                        value < 0.0 if metric in LOWER_IS_BETTER else value > 0.0
                        for value in seed_means
                    ]
                    metric_result[metric] = {
                        "label": label,
                        "coefficients": coefficients,
                        "mean": mean(seed_means),
                        "seed_mean_differences": seed_means,
                        "favorable_seed_count": sum(favorable),
                        "seed_direction_counts": {
                            "negative": sum(value < 0.0 for value in seed_means),
                            "positive": sum(value > 0.0 for value in seed_means),
                            "zero": sum(value == 0.0 for value in seed_means),
                        },
                        "hierarchical_bootstrap_95_ci": _hierarchical_ci(
                            values_by_seed, replicates=replicates, rng=rng
                        ),
                    }
                protocol_result[name] = metric_result
            results[checkpoint_label][protocol] = protocol_result
    return results


def _late_training_hover(
    manifest: dict[str, Any], *, replicates: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS):
        values_by_seed: dict[int, list[float]] = {}
        for seed in SEEDS:
            metrics_path = (
                Path(manifest["arms"][arm][str(seed)]["train_dir"])
                / "train_metrics.jsonl"
            )
            rows = read_jsonl(metrics_path)
            values = [
                float(row["hover_action_ratio"])
                for row in rows
                if bool(row.get("episode_terminal_record", False))
                and 300 <= int(row["episode"]) + 1 <= 500
            ]
            if len(values) != 201:
                raise RuntimeError(
                    f"{arm} seed {seed}: expected 201 terminal hover rows for episodes 300-500"
                )
            values_by_seed[seed] = values
        seed_means = [mean(values_by_seed[seed]) for seed in SEEDS]
        result[arm] = {
            "episode_range_inclusive": [300, 500],
            "seed_means": seed_means,
            "mean_of_seed_means": mean(seed_means),
            "hierarchical_bootstrap_95_ci": _hierarchical_ci(
                values_by_seed,
                replicates=replicates,
                rng=random.Random(20260915 + arm_index * 10_000_019),
            ),
        }
    return result


def _lambda_move_sweep(
    rows: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    *,
    lambdas: tuple[float, ...],
    replicates: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for checkpoint_index, checkpoint_label in enumerate(CHECKPOINT_LABELS):
        result[checkpoint_label] = {}
        for protocol_index, protocol in enumerate(PROTOCOLS):
            protocol_result: dict[str, Any] = {}
            for lambda_index, lambda_move in enumerate(lambdas):
                arm_values: dict[str, Any] = {}
                for arm_index, arm in enumerate(ARMS):
                    values_by_seed = {
                        seed: [
                            (
                                float(row["delay_seconds_total"])
                                + float(row["task_energy_joules_total"])
                                + lambda_move * float(row["move_energy_joules_total"])
                            )
                            / 500.0
                            / max(int(row["N_offer"]), 1)
                            for row in rows[(arm, seed, checkpoint_label, protocol)]
                        ]
                        for seed in SEEDS
                    }
                    seed_means = [mean(values_by_seed[seed]) for seed in SEEDS]
                    arm_values[arm] = {
                        "J_per_offer_mean_of_seed_means": mean(seed_means),
                        "seed_means": seed_means,
                        "hierarchical_bootstrap_95_ci": _hierarchical_ci(
                            values_by_seed,
                            replicates=replicates,
                            rng=random.Random(
                                20260915
                                + checkpoint_index * 100_003
                                + protocol_index * 1_000_003
                                + lambda_index * 10_000_019
                                + arm_index * 1009
                            ),
                        ),
                    }
                ranking = sorted(
                    ARMS,
                    key=lambda arm: arm_values[arm]["J_per_offer_mean_of_seed_means"],
                )
                protocol_result[f"{lambda_move:g}"] = {
                    "lambda_move_seconds_per_joule": lambda_move,
                    "arms": arm_values,
                    "ranking_low_to_high_cost": ranking,
                }
            result[checkpoint_label][protocol] = protocol_result
    return result


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# 六臂公平评测汇总",
        "",
        "主协议为 deterministic joint-policy；forced_hover 仅作卸载隔离附表。所有 checkpoint 都由 joint validation 唯一选定并锁定。",
        "",
        "## Joint test：2×2 教师效应与奖励/记账对比",
        "",
        "| 对比 | ΔJ/offer | 两层 bootstrap 95% CI | 有利 seed |",
        "|---|---:|---:|---:|",
    ]
    comparisons = result["comparisons"]["budget_selected"]["joint"]
    for name, label, _ in CONTRASTS:
        value = comparisons[name]["J_per_offer"]
        ci = value["hierarchical_bootstrap_95_ci"]
        lines.append(
            f"| {label} | {_fmt(value['mean'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | "
            f"{value['favorable_seed_count']}/3 |"
        )
    lines.extend(
        [
            "",
            "`C2−C2A` 与 `C2−C2B` 是在已有另一个老师前提下增加本老师的条件效应，不是净效应。`C2−C1` 是完整奖励定义的效应；两臂差异还包含完成奖励、覆盖整形与参考尺度。`B2−B1` 只在门禁账本闭合通过后解释为同一实际代价的分期记账效应。",
            "",
            "## Hover ratio",
            "",
            "| 臂 | joint test | 两层 95% CI | 训练 episode 300–500 | 两层 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    joint = result["aggregate_summaries"]["budget_selected"]["joint"]
    late = result["late_training_hover_ratio"]
    for arm in ARMS:
        test = joint[arm]["hover_action_ratio"]
        lines.append(
            f"| {arm} | {_fmt(test['mean_of_seed_means'])} | "
            f"[{_fmt(test['hierarchical_bootstrap_95_ci'][0])}, {_fmt(test['hierarchical_bootstrap_95_ci'][1])}] | "
            f"{_fmt(late[arm]['mean_of_seed_means'])} | "
            f"[{_fmt(late[arm]['hierarchical_bootstrap_95_ci'][0])}, {_fmt(late[arm]['hierarchical_bootstrap_95_ci'][1])}] |"
        )
    lines.extend(
        [
            "",
            "forced_hover 下 hover ratio 恒为 1，只作协议自检，不参与方法比较。",
            "",
            "## λ_move 离线敏感性",
            "",
        ]
    )
    sweep = result["lambda_move_sweep"]["budget_selected"]["joint"]
    rankings = []
    for key, value in sweep.items():
        ranking = " < ".join(value["ranking_low_to_high_cost"])
        rankings.append(tuple(value["ranking_low_to_high_cost"]))
        lines.append(f"- λ_move={key}: {ranking}")
    lines.append("")
    lines.append(
        "三档排名一致，结论对 λ_move 稳健。"
        if len(set(rankings)) == 1
        else "三档排名发生翻转，谁赢依赖人为 λ_move 权重，结论必须显式保留该敏感性。"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = read_json(args.manifest)
    if manifest.get("status") != "completed":
        raise RuntimeError("fair-evaluation manifest is not completed")
    if tuple(manifest["arms_order"]) != ARMS or tuple(manifest["seeds"]) != SEEDS:
        raise ValueError("manifest arm/seed matrix does not match the approved design")
    if tuple(manifest["candidate_episodes"]) != (320, 360, 400, 450, 500):
        raise ValueError("manifest candidate checkpoints do not match the approved design")
    if manifest["selection_protocol"] != "joint":
        raise ValueError("manifest did not use joint validation selection")
    for arm in ARMS:
        for seed in SEEDS:
            weights = manifest["arms"][arm][str(seed)]["candidate_teacher_weights"]
            if any(float(value) != 0.0 for value in weights.values()):
                raise AssertionError(f"candidate checkpoint still has teacher weight: {arm} seed {seed}")

    replicates = int(args.bootstrap_replicates)
    rows, audit = _load_test_rows(
        manifest, evaluation_root=Path(manifest["output_root"])
    )
    seed_summaries, aggregate_summaries = _arm_summaries(
        rows, replicates=replicates
    )
    result = {
        "schema": "six_arm_fair_evaluation_result_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "protocol": {
            "primary": "deterministic joint-policy",
            "secondary": "forced_hover offloading-isolation diagnostic",
            "checkpoint_selection": "joint validation only; one locked checkpoint shared by both test protocols",
        },
        "checkpoint_selections": manifest["checkpoint_selections"],
        "seed_summaries": seed_summaries,
        "aggregate_summaries": aggregate_summaries,
        "comparisons": _contrast_summaries(rows, replicates=replicates),
        "late_training_hover_ratio": _late_training_hover(
            manifest, replicates=replicates
        ),
        "lambda_move_sweep": _lambda_move_sweep(
            rows,
            lambdas=_lambda_values(args.lambda_move_sweep),
            replicates=replicates,
        ),
        "test_rows": [row for key in sorted(rows) for row in rows[key]],
        "audit": {"manifest_path": str(args.manifest), **audit},
        "version": manifest["version"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "pass",
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
