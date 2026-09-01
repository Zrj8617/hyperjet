from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Sequence

import numpy as np

from environment.stage1_temperature_sampling import deterministic_masked_argmax, distribution_diagnostics, legal_temperature_probabilities


ANALYSIS_SCHEMA = "stage1_temperature_followup_analysis_v1"


def replay_static_record(record: dict[str, Any], temperature: float) -> dict[str, Any]:
    logits = np.asarray(record["raw_logits"], dtype=np.float64)
    mask = np.asarray(record["candidate_mask"], dtype=bool)
    uav_ids = np.asarray(record["candidate_uav_ids"], dtype=np.int64)
    eft = np.asarray(record["eft"], dtype=np.float64)
    gumbels = np.asarray(record["gumbels"], dtype=np.float64)
    legal = np.flatnonzero(mask)
    if legal.size < 2 or any(values.shape != logits.shape for values in (mask, uav_ids, eft, gumbels)):
        raise ValueError("static record must contain aligned nontrivial candidates")
    scores = logits[legal] / float(temperature) + gumbels[legal]
    best_score = float(np.max(scores))
    tied = [int(legal[position]) for position, score in enumerate(scores) if float(score) == best_score]
    sampled_index = min(tied, key=lambda index: int(uav_ids[index]))
    argmax_index = deterministic_masked_argmax(logits, mask, uav_ids)
    best_eft = float(np.min(eft[legal]))
    greedy_tied = [int(index) for index in legal if float(eft[index]) == best_eft]
    greedy_index = min(greedy_tied, key=lambda index: int(uav_ids[index]))
    sorted_eft = np.sort(eft[legal])
    margin = float(sorted_eft[1] - sorted_eft[0])
    probabilities = legal_temperature_probabilities(logits, mask, temperature)
    diagnostics = distribution_diagnostics(probabilities, mask)
    return {
        **{key: record[key] for key in ("checkpoint_sha256", "evaluation_scenario_seed", "episode_index", "slot_index", "stable_task_id", "decision_order", "sampling_replicate")},
        "temperature": float(temperature),
        "sampled_uav_id": int(uav_ids[sampled_index]),
        "deterministic_uav_id": int(uav_ids[argmax_index]),
        "greedy_eft_uav_id": int(uav_ids[greedy_index]),
        "sampled_eft_regret": float(eft[sampled_index] - best_eft),
        "deterministic_eft_regret": float(eft[argmax_index] - best_eft),
        "sampled_greedy_agreement": bool(sampled_index == greedy_index),
        "deterministic_greedy_agreement": bool(argmax_index == greedy_index),
        "eft_margin": margin,
        "margin5_eligible": bool(margin >= 5.0),
        "margin20_eligible": bool(margin >= 20.0),
        **diagnostics,
    }


def summarize_static(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("static summary requires records")
    regrets = np.asarray([float(row["sampled_eft_regret"]) for row in rows], dtype=np.float64)
    deterministic_regrets = np.asarray([float(row["deterministic_eft_regret"]) for row in rows], dtype=np.float64)
    margin20 = [row for row in rows if bool(row["margin20_eligible"])]
    margin5 = [row for row in rows if bool(row["margin5_eligible"])]
    return {
        "decision_count": len(rows),
        "sampled_mean_EFT_regret": float(np.mean(regrets)),
        "sampled_median_EFT_regret": float(np.median(regrets)),
        "sampled_p95_EFT_regret": float(np.percentile(regrets, 95)),
        "sampled_greedy_agreement": float(np.mean([row["sampled_greedy_agreement"] for row in rows])),
        "margin5_count": len(margin5),
        "margin5_accuracy": None if not margin5 else float(np.mean([row["sampled_greedy_agreement"] for row in margin5])),
        "margin20_count": len(margin20),
        "margin20_accuracy": None if not margin20 else float(np.mean([row["sampled_greedy_agreement"] for row in margin20])),
        "normalized_entropy": float(np.mean([row["normalized_entropy"] for row in rows])),
        "max_action_probability": float(np.mean([row["max_action_probability"] for row in rows])),
        "top1_top2_probability_margin": float(np.mean([row["top1_top2_probability_margin"] for row in rows])),
        "deterministic_mean_EFT_regret": float(np.mean(deterministic_regrets)),
        "deterministic_greedy_agreement": float(np.mean([row["deterministic_greedy_agreement"] for row in rows])),
        "deterministic_margin20_accuracy": None if not margin20 else float(np.mean([row["deterministic_greedy_agreement"] for row in margin20])),
    }


def deterministic_reachability(t1_summary: dict[str, Any]) -> dict[str, Any]:
    sampled = float(t1_summary["sampled_mean_EFT_regret"])
    deterministic = float(t1_summary["deterministic_mean_EFT_regret"])
    margin20 = t1_summary.get("deterministic_margin20_accuracy")
    if margin20 is None:
        raise ValueError("deterministic margin20 accuracy is undefined")
    if sampled == 0.0:
        if deterministic != 0.0:
            raise ValueError("zero sampled regret with positive deterministic regret is invalid")
        maximum = None
        regret_pass = True
        regret_applicable = False
    else:
        maximum = (sampled - deterministic) / sampled
        regret_pass = bool(maximum >= 0.50)
        regret_applicable = True
    reasons = []
    if float(margin20) < 0.90: reasons.append("ranking_limited_margin20")
    if not regret_pass: reasons.append("ranking_limited_regret")
    return {"deterministic_margin20_accuracy": float(margin20), "deterministic_mean_EFT_regret": deterministic, "max_achievable_regret_reduction": maximum, "regret_reduction_applicable": regret_applicable, "reachable": not reasons, "reasons": reasons}


def _relative_change(candidate: float, baseline: float) -> float:
    return (float(candidate) - float(baseline)) / max(abs(float(baseline)), 1.0)


def checkpoint_guardrail(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    completed_change = _relative_change(candidate["completed_dag_count"], baseline["completed_dag_count"])
    completion_delta = float(candidate["dag_completion_rate"]) - float(baseline["dag_completion_rate"])
    reward_change = _relative_change(candidate["episode_reward_total"], baseline["episode_reward_total"])
    flow_change = None if baseline.get("average_dag_flowtime") is None or candidate.get("average_dag_flowtime") is None else _relative_change(candidate["average_dag_flowtime"], baseline["average_dag_flowtime"])
    backlog_change = _relative_change(candidate["admitted_incomplete_backlog"], baseline["admitted_incomplete_backlog"])
    material = {"completed_dag_count": completed_change <= -0.05, "dag_completion_rate": completion_delta <= -0.05, "episode_reward_total": reward_change <= -0.10, "average_dag_flowtime": flow_change is not None and flow_change >= 0.10, "admitted_incomplete_backlog": backlog_change >= 0.10}
    catastrophic = {"completed_dag_count": completed_change <= -0.15, "dag_completion_rate": completion_delta <= -0.10, "episode_reward_total": reward_change <= -0.25, "average_dag_flowtime": flow_change is not None and flow_change >= 0.25, "admitted_incomplete_backlog": backlog_change >= 0.25}
    return {"material_metrics": [key for key, value in material.items() if value], "catastrophic_metrics": [key for key, value in catastrophic.items() if value], "material_degradation_triggered": any(material.values()), "catastrophic_regression": any(catastrophic.values())}


def combined_guardrail(checkpoint_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    catastrophic = [key for key, value in checkpoint_results.items() if value["catastrophic_regression"]]
    material = [key for key, value in checkpoint_results.items() if value["material_degradation_triggered"]]
    metrics = defaultdict(int)
    for result in checkpoint_results.values():
        for metric in result["material_metrics"]: metrics[metric] += 1
    same_metric_all_three = sorted(metric for metric, count in metrics.items() if count == 3)
    return {"pass": not catastrophic and len(material) < 2, "catastrophic_checkpoints": catastrophic, "material_degradation_checkpoints": material, "same_metric_all_three": same_metric_all_three}


def classify(*, technical_pass: bool, reachability_by_checkpoint: dict[str, dict[str, Any]], moderate_pass: bool, hard_pass: bool) -> str:
    if not technical_pass: return "invalid_experiment"
    if not reachability_by_checkpoint or any(not value["reachable"] for value in reachability_by_checkpoint.values()): return "ranking_limited"
    if moderate_pass: return "probability_scale_primary"
    if hard_pass: return "hard_sharpening_only"
    return "concentration_only_rejected"


def group_static_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[(str(row["checkpoint_sha256"]), float(row["temperature"]))].append(row)
    return {key: summarize_static(value) for key, value in groups.items()}


def paired_scenario_bootstrap(t1_rows: Sequence[dict[str, Any]], candidate_rows: Sequence[dict[str, Any]], *, resamples: int = 10000, seed: int = 20260804) -> dict[str, float]:
    identity = ("checkpoint_sha256", "evaluation_scenario_seed", "slot_index", "stable_task_id", "decision_order", "sampling_replicate")
    baseline = {tuple(row[key] for key in identity): float(row["sampled_eft_regret"]) for row in t1_rows}
    candidate = {tuple(row[key] for key in identity): float(row["sampled_eft_regret"]) for row in candidate_rows}
    if baseline.keys() != candidate.keys() or not baseline:
        raise ValueError("paired bootstrap record identities differ")
    scenario_replicate: dict[tuple[int, int], list[float]] = defaultdict(list)
    for key, value in baseline.items():
        scenario_replicate[(int(key[1]), int(key[5]))].append(value - candidate[key])
    scenario_values: dict[int, list[float]] = defaultdict(list)
    for (scenario, _replicate), values in scenario_replicate.items(): scenario_values[scenario].append(float(np.mean(values)))
    scenarios = sorted(scenario_values)
    block_values = np.asarray([np.mean(scenario_values[value]) for value in scenarios], dtype=np.float64)
    rng = np.random.default_rng(int(seed)); samples = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)): samples[index] = float(np.mean(block_values[rng.integers(0, len(block_values), len(block_values))]))
    return {"mean_improvement": float(np.mean(block_values)), "ci95_lower": float(np.percentile(samples, 2.5)), "ci95_upper": float(np.percentile(samples, 97.5)), "scenario_block_count": len(scenarios), "resamples": int(resamples), "seed": int(seed)}


def static_temperature_gate(rows: Sequence[dict[str, Any]], temperature: float) -> dict[str, Any]:
    checkpoints = sorted({str(row["checkpoint_sha256"]) for row in rows})
    if len(checkpoints) != 3: raise ValueError("static gate requires all three checkpoints")
    per_checkpoint: dict[str, Any] = {}; reductions=[]
    for checkpoint in checkpoints:
        t1=[row for row in rows if row["checkpoint_sha256"]==checkpoint and float(row["temperature"])==1.0]
        candidate=[row for row in rows if row["checkpoint_sha256"]==checkpoint and float(row["temperature"])==float(temperature)]
        base_summary, candidate_summary=summarize_static(t1),summarize_static(candidate)
        denominator=float(base_summary["sampled_mean_EFT_regret"])
        reduction=None if denominator==0.0 else (denominator-float(candidate_summary["sampled_mean_EFT_regret"]))/denominator
        bootstrap=paired_scenario_bootstrap(t1,candidate)
        directions=(candidate_summary["sampled_greedy_agreement"]>base_summary["sampled_greedy_agreement"] and candidate_summary["normalized_entropy"]<base_summary["normalized_entropy"] and candidate_summary["max_action_probability"]>base_summary["max_action_probability"] and candidate_summary["top1_top2_probability_margin"]>base_summary["top1_top2_probability_margin"])
        passed=(reduction is not None and reduction>0.0 and bootstrap["ci95_lower"]>0.0 and candidate_summary["margin20_accuracy"] is not None and candidate_summary["margin20_accuracy"]>=0.90 and directions)
        per_checkpoint[checkpoint]={"passed":passed,"regret_reduction":reduction,"bootstrap":bootstrap,"summary":candidate_summary}; reductions.append(reduction)
    common_pass=all(value["passed"] for value in per_checkpoint.values()) and sum(value is not None and value>=0.50 for value in reductions)>=2
    return {"temperature":float(temperature),"common_pass":common_pass,"per_checkpoint":per_checkpoint}
