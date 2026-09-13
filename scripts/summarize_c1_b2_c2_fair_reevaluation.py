from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARMS = ("C1", "B2", "C2")
PROTOCOLS = ("forced_hover", "joint")
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
)
COMPARISONS = (
    ("C1_vs_B2", "C1", "B2"),
    ("C2_vs_B2", "C2", "B2"),
    ("C2_vs_C1", "C2", "C1"),
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
REPLAY_IDENTITY_TOP_LEVEL = (
    "ppo_update_step",
    "global_slot",
    "episode",
    "ppo_slot_count",
    "ppo_rollout_reward_total",
    "ppo_rollout_reward_mean",
    "ppo_total_loss",
    "ppo_offloading_loss",
    "ppo_movement_loss",
    "ppo_value_loss",
    "ppo_approx_kl",
    "ppo_offloading_action_count",
    "ppo_movement_action_count",
    "completed_DAG_count",
    "generated_DAG_count",
    "arrival_admitted_count",
    "arrival_blocked_count",
    "reward",
    "reward_original",
    "episode_reward_so_far",
)
TRACK_B_FIELDS = (
    "rollout_slot_count",
    "rollout_start_global_slot",
    "rollout_end_global_slot",
    "rollout_incremental_delay_seconds",
    "rollout_incremental_delay_truncated_count",
    "rollout_censored_backlog_area_seconds",
    "rollout_actual_task_energy_joules",
    "rollout_movement_energy_joules",
    "rollout_reward_settled_task_energy_joules",
    "rollout_completed_task_count",
    "rollout_completed_dag_count",
    "rollout_probe_start_active_dag_count",
    "C_orig_epoch",
    "C_orig_epoch_delay",
    "C_orig_epoch_task_energy",
    "C_orig_epoch_move_energy",
    "C_orig_epoch_per_slot",
    "C_sec_epoch_incremental",
    "C_sec_epoch_incremental_per_slot",
    "C_sec_epoch_backlog_area",
    "C_sec_epoch_backlog_area_per_slot",
    "teacher_weight",
    "teacher_phase",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def strip_eval_row(row: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    result = {key: row[key] for key in TEST_METRICS}
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


def load_track_a(audit_root: Path, replicates: int) -> dict[str, Any]:
    root = audit_root / "20260908_C1_B2_C2_fair_eval"
    manifest_path = audit_root / "20260908_C1_B2_C2_fair_eval_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise RuntimeError("Track A manifest is not completed")

    selections: dict[str, Any] = {}
    for name in ("selection_C1_B2.json", "selection_C2.json"):
        selections.update(read_json(root / name))

    rows: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    raw_paths: list[str] = []
    cost_identity_max_abs_error = 0.0
    offer_ids_by_tape: dict[int, tuple[str, ...]] = {}
    tape_paths_by_id: dict[int, set[str]] = defaultdict(set)

    for arm in ARMS:
        for seed in range(3):
            for checkpoint_label in CHECKPOINT_LABELS:
                for protocol in PROTOCOLS:
                    path = root / "test" / f"{arm}_seed{seed}_{checkpoint_label}_{protocol}.json"
                    payload = read_json(path)
                    if payload.get("status") != "completed" or len(payload["rows"]) != 50:
                        raise RuntimeError(f"incomplete test result: {path}")
                    raw_paths.append(str(path))
                    cleaned: list[dict[str, Any]] = []
                    for row in payload["rows"]:
                        component_error = abs(
                            row["J_episode"]
                            - row["J_delay_component"]
                            - row["J_task_energy_component"]
                            - row["J_move_energy_component"]
                        )
                        per_offer_error = abs(
                            row["J_per_offer"] - row["J_episode"] / max(row["N_offer"], 1)
                        )
                        cost_identity_max_abs_error = max(
                            cost_identity_max_abs_error, component_error, per_offer_error
                        )
                        tape_id = int(row["tape_id"])
                        offer_ids = tuple(row["offer_ids"])
                        if tape_id in offer_ids_by_tape and offer_ids_by_tape[tape_id] != offer_ids:
                            raise RuntimeError(f"offer tape mismatch for tape {tape_id}")
                        offer_ids_by_tape[tape_id] = offer_ids
                        tape_paths_by_id[tape_id].add(row["tape_path"])
                        cleaned.append(strip_eval_row(row, path))
                    rows[(arm, seed, checkpoint_label, protocol)] = cleaned

    if cost_identity_max_abs_error > 1e-9:
        raise RuntimeError(f"Track A cost identity error {cost_identity_max_abs_error}")
    if any(len(paths) != 1 for paths in tape_paths_by_id.values()):
        raise RuntimeError("a tape id resolved to multiple tape paths")

    seed_summaries: dict[str, Any] = {}
    aggregate_summaries: dict[str, Any] = {}
    for checkpoint_label in CHECKPOINT_LABELS:
        seed_summaries[checkpoint_label] = {}
        aggregate_summaries[checkpoint_label] = {}
        for protocol in PROTOCOLS:
            seed_summaries[checkpoint_label][protocol] = {}
            aggregate_summaries[checkpoint_label][protocol] = {}
            for arm in ARMS:
                by_seed: dict[str, Any] = {}
                for seed in range(3):
                    arm_rows = rows[(arm, seed, checkpoint_label, protocol)]
                    by_seed[str(seed)] = {
                        metric: summarize_values([float(row[metric]) for row in arm_rows])
                        for metric in TEST_METRICS
                    }
                seed_summaries[checkpoint_label][protocol][arm] = by_seed
                aggregate_summaries[checkpoint_label][protocol][arm] = {
                    metric: {
                        "mean_of_seed_means": mean(
                            [by_seed[str(seed)][metric]["mean"] for seed in range(3)]
                        ),
                        "seed_means": [by_seed[str(seed)][metric]["mean"] for seed in range(3)],
                    }
                    for metric in TEST_METRICS
                }

    comparison_results: dict[str, Any] = {}
    for checkpoint_label in CHECKPOINT_LABELS:
        comparison_results[checkpoint_label] = {}
        for protocol in PROTOCOLS:
            comparison_results[checkpoint_label][protocol] = {}
            for comparison_name, candidate, control in COMPARISONS:
                metric_results: dict[str, Any] = {}
                for metric_index, metric in enumerate(TEST_METRICS):
                    differences_by_seed: dict[int, list[float]] = {}
                    for seed in range(3):
                        candidate_rows = rows[(candidate, seed, checkpoint_label, protocol)]
                        control_rows = rows[(control, seed, checkpoint_label, protocol)]
                        control_by_tape = {int(row["tape_id"]): row for row in control_rows}
                        differences_by_seed[seed] = [
                            float(row[metric]) - float(control_by_tape[int(row["tape_id"])][metric])
                            for row in candidate_rows
                        ]
                    seed_mean_differences = [mean(differences_by_seed[seed]) for seed in range(3)]
                    paired = [value for seed in range(3) for value in differences_by_seed[seed]]
                    rng = random.Random(
                        20260908
                        + metric_index * 1009
                        + CHECKPOINT_LABELS.index(checkpoint_label) * 100_003
                        + PROTOCOLS.index(protocol) * 1_000_003
                        + COMPARISONS.index((comparison_name, candidate, control)) * 10_000_019
                    )
                    paired_bootstrap: list[float] = []
                    hierarchical_bootstrap: list[float] = []
                    for _ in range(replicates):
                        paired_bootstrap.append(mean([rng.choice(paired) for _ in paired]))
                        sampled_seed_means: list[float] = []
                        for sampled_seed in [rng.randrange(3) for _ in range(3)]:
                            values = differences_by_seed[sampled_seed]
                            sampled_seed_means.append(mean([rng.choice(values) for _ in values]))
                        hierarchical_bootstrap.append(mean(sampled_seed_means))
                    favorable = [
                        value < 0.0 if metric in LOWER_IS_BETTER else value > 0.0
                        for value in seed_mean_differences
                    ]
                    metric_results[metric] = {
                        "difference_definition": f"{candidate}-{control}",
                        "mean": mean(paired),
                        "median": float(statistics.median(paired)),
                        "seed_mean_differences": seed_mean_differences,
                        "favorable_seed_count": sum(favorable),
                        "paired_bootstrap_95_ci": interval(paired_bootstrap),
                        "hierarchical_bootstrap_95_ci": interval(hierarchical_bootstrap),
                    }
                comparison_results[checkpoint_label][protocol][comparison_name] = metric_results

    validation_paths = sorted(str(path) for path in (root / "validation").glob("*.json"))
    validation_expected = len(ARMS) * 3 * 5 * len(PROTOCOLS)
    if len(validation_paths) != validation_expected:
        raise RuntimeError(f"expected {validation_expected} validation files, got {len(validation_paths)}")

    return {
        "status": "pass",
        "protocol": {
            "offer_slots": 500,
            "drain": False,
            "primary": "forced_hover deterministic masked argmax",
            "secondary": "complete deterministic joint policy",
            "validation_tape_ids": manifest["validation_tape_ids"],
            "test_tape_ids": manifest["test_tape_ids"],
            "selection_metric": "forced-hover validation mean J_per_offer; lower is better",
            "tape_independence": "validation 100-119; test 200-249",
        },
        "checkpoint_selections": selections,
        "seed_summaries": seed_summaries,
        "aggregate_summaries": aggregate_summaries,
        "comparisons": comparison_results,
        "test_rows": [
            row
            for key in sorted(rows)
            for row in rows[key]
        ],
        "audit": {
            "manifest_path": str(manifest_path),
            "manifest_status": manifest["status"],
            "manifest_completed_at_utc": manifest["completed_at_utc"],
            "validation_file_count": len(validation_paths),
            "test_file_count": len(raw_paths),
            "test_row_count": sum(len(value) for value in rows.values()),
            "shared_offer_tape_identity": True,
            "shared_tape_path_per_id": True,
            "shared_tape_count": len(tape_paths_by_id),
            "cost_identity_max_abs_error": cost_identity_max_abs_error,
            "raw_validation_paths": validation_paths,
            "raw_test_paths": raw_paths,
        },
        "version": manifest["version"],
        "run_actual_parameters": {
            arm: {
                str(seed): manifest["arms"][arm][str(seed)]["actual_parameters"]
                for seed in range(3)
            }
            for arm in ARMS
        },
        "run_sources": {
            arm: {
                str(seed): {
                    key: manifest["arms"][arm][str(seed)][key]
                    for key in ("result_path", "train_dir", "checkpoint", "source_version")
                }
                for seed in range(3)
            }
            for arm in ARMS
        },
    }


def is_local_maximum(rows: list[dict[str, Any]], index: int) -> bool:
    reward = float(rows[index]["ppo_rollout_reward_total"])
    if index == 0 or index == len(rows) - 1:
        return False
    left = float(rows[index - 1]["ppo_rollout_reward_total"])
    right = float(rows[index + 1]["ppo_rollout_reward_total"])
    return reward >= left and reward >= right and (reward > left or reward > right)


def select_peaks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    local_updates = {
        int(row["ppo_update_step"])
        for index, row in enumerate(rows)
        if is_local_maximum(rows, index)
    }
    ranked = sorted(
        rows,
        key=lambda row: (-float(row["ppo_rollout_reward_total"]), int(row["ppo_update_step"])),
    )
    selected: list[dict[str, Any]] = []
    fallback_considered: set[int] = set()
    for local_only in (True, False):
        for row in ranked:
            update = int(row["ppo_update_step"])
            is_local = update in local_updates
            if is_local != local_only or any(update == int(item["ppo_update_step"]) for item in selected):
                continue
            if not is_local:
                fallback_considered.add(update)
            too_close_to = [
                int(item["ppo_update_step"])
                for item in selected
                if abs(update - int(item["ppo_update_step"])) < 100
            ]
            if too_close_to:
                continue
            selected.append(row)
            if len(selected) == 5:
                break
        if len(selected) == 5:
            break
    if len(selected) != 5:
        raise RuntimeError("unable to select five spaced peaks")

    selected_updates = {int(row["ppo_update_step"]) for row in selected}
    audit: list[dict[str, Any]] = []
    for row in ranked:
        update = int(row["ppo_update_step"])
        is_local = update in local_updates
        if not is_local and update not in fallback_considered and update not in selected_updates:
            continue
        item: dict[str, Any] = {
            "update": update,
            "reward": row["ppo_rollout_reward_total"],
            "local_maximum": is_local,
        }
        if update in selected_updates:
            item["decision"] = "selected_local" if is_local else "selected_nonlocal_fallback"
        else:
            too_close_to = sorted(
                selected_update
                for selected_update in selected_updates
                if abs(update - selected_update) < 100
            )
            if too_close_to:
                item["decision"] = "rejected_spacing"
                item["too_close_to"] = too_close_to
            else:
                item["decision"] = "rejected_lower_reward_after_five_selected"
        audit.append(item)
    return sorted(selected, key=lambda row: int(row["ppo_update_step"])), audit


def replay_identity(original: list[dict[str, Any]], replay: list[dict[str, Any]]) -> dict[str, Any]:
    if len(original) < 1000 or len(replay) != 1000:
        raise RuntimeError("original/replay update count is insufficient")
    mismatches: list[dict[str, Any]] = []
    compared = 0
    for original_row, replay_row in zip(original[:1000], replay):
        for key in REPLAY_IDENTITY_TOP_LEVEL:
            compared += 1
            if original_row[key] != replay_row[key]:
                mismatches.append(
                    {
                        "update": replay_row["ppo_update_step"],
                        "field": key,
                        "original": original_row[key],
                        "replay": replay_row[key],
                    }
                )
        original_diag = original_row["ppo_diagnostics"]
        replay_diag = replay_row["ppo_diagnostics"]
        for key in sorted(original_diag.keys() & replay_diag.keys()):
            if "wall" in key:
                continue
            compared += 1
            if original_diag[key] != replay_diag[key]:
                mismatches.append(
                    {
                        "update": replay_row["ppo_update_step"],
                        "field": f"ppo_diagnostics.{key}",
                        "original": original_diag[key],
                        "replay": replay_diag[key],
                    }
                )
    return {"pass": not mismatches, "compared_value_count": compared, "mismatches": mismatches[:20]}


def load_track_b(audit_root: Path) -> dict[str, Any]:
    original_launch_manifest_path = audit_root / "20260908_C1_B2_peak_epoch_launch_manifest.json"
    correction_launch_manifest_path = audit_root / "20260909_C1_peak_epoch_diag_replay_fixed_launch_manifest.json"
    original_launch_manifest = read_json(original_launch_manifest_path)
    correction_launch_manifest = read_json(correction_launch_manifest_path)
    source_dates = {"C1": "20260907", "B2": "20260906"}
    arm_results: dict[str, Any] = {}
    hard_stop_mismatches: list[dict[str, Any]] = []

    for arm in ("C1", "B2"):
        seed_results: dict[str, Any] = {}
        for seed in range(3):
            source_result_path = audit_root / f"{source_dates[arm]}_{arm}_seed{seed}" / "result.json"
            replay_run_name = (
                f"20260909_C1_peak_epoch_diag_replay_fixed_seed{seed}"
                if arm == "C1"
                else f"20260908_B2_peak_epoch_diag_replay_seed{seed}"
            )
            replay_result_path = audit_root / replay_run_name / "result.json"
            source_result = read_json(source_result_path)
            replay_result = read_json(replay_result_path)
            if replay_result.get("status") != "completed":
                raise RuntimeError(f"incomplete Track B replay: {replay_result_path}")
            original_rows = read_jsonl(Path(source_result["train_dir"]) / "train_metrics.jsonl")
            replay_rows = read_jsonl(Path(replay_result["train_dir"]) / "train_metrics.jsonl")
            identity = replay_identity(original_rows, replay_rows)
            if not identity["pass"]:
                hard_stop_mismatches.append({"arm": arm, "seed": seed, **identity})

            eligible: list[dict[str, Any]] = []
            for row in original_rows[:1000]:
                if arm == "C1" and float(row["ppo_diagnostics"]["teacher_weight"]) != 1.0:
                    continue
                eligible.append(row)
            selected, candidate_audit = select_peaks(eligible)
            replay_by_update = {int(row["ppo_update_step"]): row for row in replay_rows}
            points: list[dict[str, Any]] = []
            for source_row in selected:
                update = int(source_row["ppo_update_step"])
                replay_row = replay_by_update[update]
                diagnostics = replay_row["ppo_diagnostics"]
                point = {
                    "arm": arm,
                    "model_seed": seed,
                    "ppo_update_step": update,
                    "episode": int(replay_row["episode"]),
                    "global_slot_at_log": int(replay_row["global_slot"]),
                    "ppo_rollout_reward_total": float(source_row["ppo_rollout_reward_total"]),
                    "raw_original_metrics_path": str(Path(source_result["train_dir"]) / "train_metrics.jsonl"),
                    "raw_replay_metrics_path": str(Path(replay_result["train_dir"]) / "train_metrics.jsonl"),
                }
                point.update({key: diagnostics[key] for key in TRACK_B_FIELDS})
                points.append(point)

            seed_mean_fields = (
                "C_orig_epoch",
                "C_orig_epoch_per_slot",
                "C_sec_epoch_incremental",
                "C_sec_epoch_incremental_per_slot",
                "C_sec_epoch_backlog_area",
                "C_sec_epoch_backlog_area_per_slot",
                "rollout_incremental_delay_seconds",
                "rollout_censored_backlog_area_seconds",
                "rollout_actual_task_energy_joules",
                "rollout_movement_energy_joules",
                "rollout_completed_task_count",
                "rollout_completed_dag_count",
                "rollout_slot_count",
            )
            seed_means = {
                key: mean([float(point[key]) for point in points])
                for key in seed_mean_fields
            }
            seed_results[str(seed)] = {
                "eligible_update_count": len(eligible),
                "selection_rule": "native rollout reward local maxima, descending reward, >=100 updates apart",
                "selected_updates": [int(point["ppo_update_step"]) for point in points],
                "used_nonlocal_fallback": any(
                    item["decision"] == "selected_nonlocal_fallback" for item in candidate_audit
                ),
                "candidate_audit": candidate_audit,
                "points": points,
                "five_point_means": seed_means,
                "replay_identity": identity,
                "source_result_path": str(source_result_path),
                "replay_result_path": str(replay_result_path),
                "source_actual_parameters": source_result["actual_parameters"],
                "source_version": source_result["version"],
                "replay_version": replay_result["version"],
            }
        arm_results[arm] = {"seeds": seed_results}

    if hard_stop_mismatches:
        raise RuntimeError(f"diagnostic replay changed original values: {hard_stop_mismatches}")

    aggregate_fields = list(arm_results["C1"]["seeds"]["0"]["five_point_means"])
    for arm in ("C1", "B2"):
        arm_results[arm]["three_seed_summary"] = {
            key: {
                "mean_of_seed_means": mean(
                    [arm_results[arm]["seeds"][str(seed)]["five_point_means"][key] for seed in range(3)]
                ),
                "seed_means": [
                    arm_results[arm]["seeds"][str(seed)]["five_point_means"][key]
                    for seed in range(3)
                ],
            }
            for key in aggregate_fields
        }
    difference = {
        key: {
            "definition": "C1-B2",
            "mean_of_seed_mean_differences": mean(
                [
                    arm_results["C1"]["seeds"][str(seed)]["five_point_means"][key]
                    - arm_results["B2"]["seeds"][str(seed)]["five_point_means"][key]
                    for seed in range(3)
                ]
            ),
            "seed_mean_differences": [
                arm_results["C1"]["seeds"][str(seed)]["five_point_means"][key]
                - arm_results["B2"]["seeds"][str(seed)]["five_point_means"][key]
                for seed in range(3)
            ],
        }
        for key in aggregate_fields
    }
    return {
        "status": "pass",
        "role": "auxiliary selected-training-trajectory diagnostic; not a substitute for Track A",
        "arms": arm_results,
        "C1_minus_B2": difference,
        "audit": {
            "original_launch_manifest_path": str(original_launch_manifest_path),
            "correction_launch_manifest_path": str(correction_launch_manifest_path),
            "max_updates": 1000,
            "episodes_cap": 250,
            "teacher_anneal_total_updates": 2000,
            "all_six_replays_completed": True,
            "all_replay_identities_passed": True,
        },
        "version": {
            "B2_replay": original_launch_manifest["version"],
            "C1_corrected_replay": correction_launch_manifest["version"],
        },
    }


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_report(result: dict[str, Any]) -> str:
    track_a = result["track_a"]
    track_b = result["track_b"]
    lines = [
        "# C1 / B2 / C2 公平重评与峰值 epoch 诊断结果（2026-09-08）",
        "",
        "**对应 spec：** `docs/superpowers/specs/2026-09-08-c1-b2-fair-reevaluation.md` 及用户批准的 C2 扩展。  ",
        "**执行状态：** PASS（Track A、Track B 与 C2 扩展均完成；未触发第 10 节硬停止条件）。  ",
        "**结论口径：** 正式结论只来自固定外生 tape 的 Track A；Track B 是事后按原生 rollout reward 选峰的辅助诊断。",
        "",
        "## 1. 结论先行",
        "",
    ]
    primary = track_a["comparisons"]["budget_selected"]["forced_hover"]
    final = track_a["comparisons"]["final_ep0500"]["forced_hover"]
    for label, comparisons in (("等预算 validation 选点", primary), ("最终 episode 500", final)):
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| 对比（候选−对照） | ΔJ/offer | 两层 95% CI | 有利 seed | Δ端到端完成率 |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, candidate, control in COMPARISONS:
            j = comparisons[name]["J_per_offer"]
            completion = comparisons[name]["end_to_end_completion_rate"]
            lines.append(
                f"| {candidate}−{control} | {fmt(j['mean'])} | "
                f"[{fmt(j['hierarchical_bootstrap_95_ci'][0])}, {fmt(j['hierarchical_bootstrap_95_ci'][1])}] | "
                f"{j['favorable_seed_count']}/3 | {fmt(completion['mean'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "正式主协议的方向由上表给出。区间将训练 seed 与 test tape 两层不确定性都计入；只有 3 个训练 seed，不能把 150 个 seed×tape 单元当成 150 个独立训练重复。C1 与 B2 同时改变奖励和信用通路，因此任何 C1−B2 差异都不是老师的独立因果效应。C2−B2 才隔离同为 B2 环境奖励时的老师预热，C2−C1 则比较同款老师下撤老师后的 B2 增量奖励与原始奖励。",
            "",
            "## 2. Track A：固定外生 tape 正式公平评估",
            "",
            "### 2.1 Checkpoint 选择",
            "",
            "| 臂 | seed 0 | seed 1 | seed 2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        lines.append(
            f"| {arm} | "
            + " | ".join(str(track_a["checkpoint_selections"][arm][str(seed)]["selected_episode"]) for seed in range(3))
            + " |"
        )

    for checkpoint_label, title in (
        ("budget_selected", "等预算 validation 选点"),
        ("final_ep0500", "最终 episode 500"),
    ):
        for protocol, protocol_title in (("forced_hover", "forced-hover 主协议"), ("joint", "deterministic joint-policy 次协议")):
            lines.extend(["", f"### 2.2 {title} · {protocol_title}", ""])
            lines.append("| 臂 | J/offer | J/episode | delay | task E | move E | admission | conditional | end-to-end |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            summary = track_a["aggregate_summaries"][checkpoint_label][protocol]
            for arm in ARMS:
                value = summary[arm]
                lines.append(
                    f"| {arm} | {fmt(value['J_per_offer']['mean_of_seed_means'])} | "
                    f"{fmt(value['J_episode']['mean_of_seed_means'], 2)} | "
                    f"{fmt(value['J_delay_component']['mean_of_seed_means'], 2)} | "
                    f"{fmt(value['J_task_energy_component']['mean_of_seed_means'], 2)} | "
                    f"{fmt(value['J_move_energy_component']['mean_of_seed_means'], 2)} | "
                    f"{fmt(value['admission_rate']['mean_of_seed_means'])} | "
                    f"{fmt(value['conditional_completion_rate']['mean_of_seed_means'])} | "
                    f"{fmt(value['end_to_end_completion_rate']['mean_of_seed_means'])} |"
                )

    lines.extend(
        [
            "",
            "### 2.3 可复核性",
            "",
            f"- validation 文件：{track_a['audit']['validation_file_count']}；test 文件：{track_a['audit']['test_file_count']}；逐 tape test 行：{track_a['audit']['test_row_count']}。",
            f"- 全部策略共享同一套 50 条 test tape，offer ID 逐值一致；每个 tape ID 只解析到一个预生成 tape 路径。",
            f"- `J_episode` 分量恒等式与 `J_per_offer` 恒等式最大绝对误差：{track_a['audit']['cost_identity_max_abs_error']:.3e}。",
            "- 训练奖励未参与跨臂正式排名。",
            "",
            "## 3. Track B：多峰值 PPO rollout/update epoch（辅助诊断）",
            "",
            "| 臂 | seed | 5 个 update | C_orig/epoch | C_orig/slot | C_sec incremental/epoch | C_sec incremental/slot | C_sec backlog/epoch |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ("C1", "B2"):
        for seed in range(3):
            seed_result = track_b["arms"][arm]["seeds"][str(seed)]
            means = seed_result["five_point_means"]
            lines.append(
                f"| {arm} | {seed} | {', '.join(map(str, seed_result['selected_updates']))} | "
                f"{fmt(means['C_orig_epoch'], 2)} | {fmt(means['C_orig_epoch_per_slot'])} | "
                f"{fmt(means['C_sec_epoch_incremental'], 2)} | {fmt(means['C_sec_epoch_incremental_per_slot'], 2)} | "
                f"{fmt(means['C_sec_epoch_backlog_area'], 2)} |"
            )
    lines.extend(["", "三 seed 先各自对 5 点平均，再汇总：", ""])
    lines.append("| 指标 | C1 | B2 | C1−B2 |")
    lines.append("|---|---:|---:|---:|")
    for key in (
        "C_orig_epoch",
        "C_orig_epoch_per_slot",
        "C_sec_epoch_incremental",
        "C_sec_epoch_incremental_per_slot",
        "C_sec_epoch_backlog_area",
        "C_sec_epoch_backlog_area_per_slot",
    ):
        c1 = track_b["arms"]["C1"]["three_seed_summary"][key]["mean_of_seed_means"]
        b2 = track_b["arms"]["B2"]["three_seed_summary"][key]["mean_of_seed_means"]
        diff = track_b["C1_minus_B2"][key]["mean_of_seed_mean_differences"]
        lines.append(f"| {key} | {fmt(c1, 2)} | {fmt(b2, 2)} | {fmt(diff, 2)} |")
    lines.extend(
        [
            "",
            "峰值结果有明确选择偏差，也来自各策略访问到的不同训练状态，只能解释训练轨迹，不能替代固定 tape 测试。`C_sec_epoch_incremental` 使用本 epoch 首次结算任务的可精确归属增量时延；`C_sec_epoch_backlog_area` 使用探针起点到终点的 censored backlog-area 边界。没有把完整 DAG flowtime 归到最后一个 slot，也没有用 episode reward 除法近似。",
            "",
            "## 4. 版本现实（AGENTS.md §3）",
            "",
            f"- 服务器实际 HEAD：`{track_a['version']['head']}`；Track A manifest 记录 dirty {len(track_a['version']['dirty'])} 行，完整列表见机器 JSON。",
            f"- Track A 脚本 git object：orchestrator `{track_a['version']['orchestrator_git_object']}`；evaluator `{track_a['version']['evaluator_git_object']}`；tape module `{track_a['version']['tape_module_git_object']}`；environment `{track_a['version']['env_git_object']}`。",
            f"- Track B B2 replay 脚本 git object：launcher `{track_b['version']['B2_replay']['launcher_git_object']}`；runner `{track_b['version']['B2_replay']['runner_git_object']}`；trainer `{track_b['version']['B2_replay']['trainer_git_object']}`；metrics `{track_b['version']['B2_replay']['metrics_git_object']}`。",
            f"- Track B 修正 C1 replay 脚本 git object：launcher `{track_b['version']['C1_corrected_replay']['launcher_git_object']}`；runner `{track_b['version']['C1_corrected_replay']['runner_git_object']}`；trainer `{track_b['version']['C1_corrected_replay']['trainer_git_object']}`；metrics `{track_b['version']['C1_corrected_replay']['metrics_git_object']}`。",
            "- C1/B2/C2 每个 run 的实际参数、源 result、train_dir、checkpoint 与源版本均逐 seed 保存在机器 JSON；没有根据命名猜路径。",
            "",
            "## 5. 结论边界",
            "",
            "- 正式胜负只按固定 tape 上的实际时延、任务能耗、移动能耗统一口径判断；不直接比较 C1/B2/C2 的训练 reward。",
            "- completed-DAG flowtime 仅是辅助指标；主成本把未接纳或未完成 offer 按 horizon 截尾纳入。",
            "- C2 的解释范围严格限定为已批准的两个对比，不新增奖励臂，不进入 HGNN vs MLP。",
            "- 机器 JSON 保存所有 1800 条精简逐 tape 数值行及其服务器原始 JSON 路径；峰值候选、剔除原因、15×2 个峰值明细和原始 JSONL 路径也全部可追溯。",
            "",
            "## 6. 原始产物",
            "",
            f"- Track A manifest：`{track_a['audit']['manifest_path']}`",
            f"- Track B 原 B2 launch manifest：`{track_b['audit']['original_launch_manifest_path']}`",
            f"- Track B 修正 C1 launch manifest：`{track_b['audit']['correction_launch_manifest_path']}`",
            "- 原始 validation/test JSON、训练 JSONL、诊断 replay JSONL 的逐文件路径见机器可读结果。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    result = {
        "schema": "c1_b2_c2_fair_reevaluation_result_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "track_a": load_track_a(args.audit_root, args.bootstrap_replicates),
        "track_b": load_track_b(args.audit_root),
        "conclusion_scope": {
            "formal_primary": "Track A fixed external tapes",
            "track_b_role": "auxiliary diagnostic only",
            "training_rewards_compared_across_arms": False,
            "hgnn_vs_mlp_entered": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"status": "pass", "output_json": str(args.output_json), "output_md": str(args.output_md)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
