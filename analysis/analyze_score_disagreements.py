from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

import config
from environment.env import Env
from scripts.score_experiment import temporary_score_config


def _zero_actions() -> np.ndarray:
    return np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)


def _candidate_by_uav(record: dict, uav_id: int | None) -> dict | None:
    if uav_id is None:
        return None
    for candidate in record["candidates"]:
        if candidate["uav_id"] == uav_id:
            return candidate
    return None


def _summarize(records: list[dict], total_decisions: int) -> dict:
    disagreements = [record for record in records if record["disagrees_with_heuristic"]]
    worse_finish_gaps: list[float] = []
    score_margins: list[float] = []
    slack_values: list[float] = []
    candidate_counts: list[int] = []
    multi_pred_count = 0
    tight_slack_count = 0
    high_level_count = 0

    for record in disagreements:
        heuristic_candidate = _candidate_by_uav(record, record["heuristic_uav"])
        score_candidate = _candidate_by_uav(record, record["score_uav"])
        if heuristic_candidate is not None and score_candidate is not None:
            worse_finish_gaps.append(float(score_candidate["planned_finish"] - heuristic_candidate["planned_finish"]))
            if heuristic_candidate["score"] is not None and score_candidate["score"] is not None:
                score_margins.append(float(score_candidate["score"] - heuristic_candidate["score"]))
        slack = float(record["task_slack"])
        slack_values.append(slack)
        candidate_counts.append(len(record["candidates"]))
        if int(record["num_predecessors"]) >= 2:
            multi_pred_count += 1
        if slack <= config.DAG_CRITICAL_SLACK_THRESHOLD:
            tight_slack_count += 1
        if int(record["task_level"]) >= 2:
            high_level_count += 1

    def avg(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "total_decisions": total_decisions,
        "disagreement_count": len(disagreements),
        "disagreement_rate": len(disagreements) / max(total_decisions, 1),
        "avg_score_minus_heuristic_finish_gap": avg(worse_finish_gaps),
        "avg_score_margin_over_heuristic": avg(score_margins),
        "avg_disagreement_slack": avg(slack_values),
        "avg_disagreement_candidate_count": avg([float(v) for v in candidate_counts]),
        "multi_predecessor_disagreement_count": multi_pred_count,
        "tight_slack_disagreement_count": tight_slack_count,
        "high_level_disagreement_count": high_level_count,
    }


def analyze_disagreements(
    *,
    checkpoint: str,
    seed: int,
    episodes: int,
    steps: int,
    output_dir: Path,
    max_records: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    total_decisions = 0

    with temporary_score_config(
        use_score=True,
        checkpoint_path=checkpoint,
        fallback_to_heuristic=True,
    ):
        for episode_idx in range(episodes):
            episode_seed = seed + episode_idx
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)

            env = Env()
            env.reset()
            for step_idx in range(steps):
                env.step(_zero_actions())
                records = [dataclasses.asdict(record) for record in env.task_executor.latest_assignment_records]
                total_decisions += len(records)
                for record in records:
                    if not record["disagrees_with_heuristic"]:
                        continue
                    heuristic_candidate = _candidate_by_uav(record, record["heuristic_uav"])
                    score_candidate = _candidate_by_uav(record, record["score_uav"])
                    record["episode"] = episode_idx
                    record["step"] = step_idx + 1
                    record["heuristic_planned_finish"] = None if heuristic_candidate is None else heuristic_candidate["planned_finish"]
                    record["score_planned_finish"] = None if score_candidate is None else score_candidate["planned_finish"]
                    if heuristic_candidate is not None and score_candidate is not None:
                        record["score_minus_heuristic_finish_gap"] = (
                            score_candidate["planned_finish"] - heuristic_candidate["planned_finish"]
                        )
                    else:
                        record["score_minus_heuristic_finish_gap"] = None
                    if len(all_records) < max_records:
                        all_records.append(record)

    payload = {
        "checkpoint": checkpoint,
        "seed": seed,
        "episodes": episodes,
        "steps": steps,
        "summary": _summarize(all_records, total_decisions),
        "records": all_records,
    }
    output_path = output_dir / f"score_disagreements_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--output_dir", type=str, default="/tmp/hyperuav_disagreement_analysis")
    parser.add_argument("--max_records", type=int, default=500)
    args = parser.parse_args()

    path = analyze_disagreements(
        checkpoint=args.checkpoint,
        seed=args.seed,
        episodes=args.episodes,
        steps=args.steps,
        output_dir=Path(args.output_dir),
        max_records=args.max_records,
    )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    print(f"Saved disagreement analysis to {path}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
