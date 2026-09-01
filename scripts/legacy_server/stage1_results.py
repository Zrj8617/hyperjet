import glob
import json
import math
from collections import defaultdict

import numpy as np


TRAIN_UPDATES = (1, 5, 10, 20, 30)
EVAL_UPDATES = (0, 1, 5, 10, 20, 30)
SEEDS = (42, 86, 1042)
GROUPS = ("S1-A", "S1-B")
BEHAVIOR_METRICS = (
    "raw_eft_regret_mean",
    "raw_eft_regret_p95",
    "greedy_agreement",
    "margin5_accuracy",
    "margin20_accuracy",
    "normalized_entropy",
    "max_action_probability",
    "top1_top2_probability_margin",
)
EVAL_METRICS = (
    "episode_reward_total",
    "completed_dag_count",
    "generated_dag_count",
    "dag_completion_rate",
    "average_dag_flowtime",
    "avg_uav_queue_length",
)


def stats(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "mean": None if not values else float(np.mean(values)),
        "std": None if not values else float(np.std(values)),
        "n": len(values),
    }


training = {}
for directory in glob.glob("logs/decision_ppo_bandit/*_stage1_formal_*"):
    with open(directory + "/summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(directory + "/updates.jsonl", encoding="utf-8") as handle:
        updates = [json.loads(line) for line in handle if line.strip()]
    training[(summary["group"], int(summary["seed"]))] = {
        int(row["completed_update"]): row for row in updates
    }

behavior = {}
behavior_delta = {}
for update in TRAIN_UPDATES:
    behavior[str(update)] = {}
    behavior_delta[str(update)] = {}
    for group in GROUPS:
        behavior[str(update)][group] = {
            metric: stats(
                training[(group, seed)][update]["behavior"].get(metric)
                for seed in SEEDS
            )
            for metric in BEHAVIOR_METRICS
        }
    for metric in BEHAVIOR_METRICS:
        behavior_delta[str(update)][metric] = stats(
            training[("S1-B", seed)][update]["behavior"].get(metric)
            - training[("S1-A", seed)][update]["behavior"].get(metric)
            for seed in SEEDS
            if training[("S1-B", seed)][update]["behavior"].get(metric) is not None
            and training[("S1-A", seed)][update]["behavior"].get(metric) is not None
        )

with open(
    "logs/decision_ppo_bandit_closed_loop/stage1_formal/comparison_summary.json",
    encoding="utf-8",
) as handle:
    closed_payload = json.load(handle)
assert closed_payload["technical_pass"] is True
assert len(closed_payload["rows"]) == 36

closed_index = {
    (row["group"], int(row["seed"]), int(row["completed_update"])): row
    for row in closed_payload["rows"]
}
closed_loop = {}
closed_delta = {}
for update in EVAL_UPDATES:
    closed_loop[str(update)] = {}
    closed_delta[str(update)] = {}
    for mode in ("stochastic", "deterministic"):
        closed_loop[str(update)][mode] = {}
        closed_delta[str(update)][mode] = {}
        for group in GROUPS:
            closed_loop[str(update)][mode][group] = {
                metric: stats(
                    closed_index[(group, seed, update)]["summary"][mode][metric]["mean"]
                    for seed in SEEDS
                )
                for metric in EVAL_METRICS
            }
        for metric in EVAL_METRICS:
            closed_delta[str(update)][mode][metric] = stats(
                closed_index[("S1-B", seed, update)]["summary"][mode][metric]["mean"]
                - closed_index[("S1-A", seed, update)]["summary"][mode][metric]["mean"]
                for seed in SEEDS
            )

for row in closed_payload["rows"]:
    for mode in ("stochastic", "deterministic"):
        for metric in EVAL_METRICS:
            assert row["summary"][mode][metric]["count"] == 20
            value = row["summary"][mode][metric]["mean"]
            assert value is None or math.isfinite(float(value))

result = {
    "technical": {
        "training_cells": len(training),
        "closed_loop_cells": len(closed_payload["rows"]),
        "episodes_per_mode": 20,
        "technical_pass": True,
    },
    "behavior_three_seed": behavior,
    "behavior_same_seed_delta_B_minus_A": behavior_delta,
    "closed_loop_three_seed": closed_loop,
    "closed_loop_same_seed_delta_B_minus_A": closed_delta,
    "pairing_limitation": closed_payload["pairing_limitation"],
}
path = "logs/decision_ppo_bandit_closed_loop/stage1_formal/stage1_analysis.json"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
