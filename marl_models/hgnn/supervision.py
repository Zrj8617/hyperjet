from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import torch

import config
from environment.env import Env
from environment.graph_builder import HeteroGraphSnapshot
from environment.task_execution import TaskSupervisionTarget
from utils.progress import TerminalProgress


@dataclass(slots=True)
class GraphSupervisionSample:
    snapshot: HeteroGraphSnapshot
    targets: list[TaskSupervisionTarget]

    def to_dict(self) -> dict:
        return {
            "snapshot": {
                "task_ids": self.snapshot.task_ids,
                "uav_ids": self.snapshot.uav_ids,
                "task_features": self.snapshot.task_features.tolist(),
                "uav_features": self.snapshot.uav_features.tolist(),
                "dependency_edges": self.snapshot.dependency_edges,
                "task_uav_edges": self.snapshot.task_uav_edges,
                "task_uav_edge_features": self.snapshot.task_uav_edge_features.tolist(),
                "uav_uav_edges": self.snapshot.uav_uav_edges,
                "collaborative_hyperedges": self.snapshot.collaborative_hyperedges,
                "critical_hyperedges": self.snapshot.critical_hyperedges,
                "attribute_hyperedges": self.snapshot.attribute_hyperedges,
            },
            "targets": [
                {
                    "task_id": target.task_id,
                    "feasible_uav_ids": target.feasible_uav_ids,
                    "heuristic_best_uav": target.heuristic_best_uav,
                    "heuristic_eft_by_uav": target.heuristic_eft_by_uav,
                }
                for target in self.targets
            ],
        }


def collect_score_supervision_dataset(
    num_episodes: int,
    steps_per_episode: int,
    action_mode: str = "zero",
) -> list[GraphSupervisionSample]:
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    config.USE_HGNN_SCORE_ASSIGNMENT = False
    env = Env()
    samples: list[GraphSupervisionSample] = []
    total_steps = num_episodes * steps_per_episode
    progress = TerminalProgress(total_steps, f"collect:{action_mode}")

    for episode_idx in range(num_episodes):
        env.reset()
        for step_idx in range(steps_per_episode):
            snapshot = env.latest_graph_snapshot
            if snapshot is not None:
                allowed_edges = set(snapshot.task_uav_edges)
                targets = env.task_executor.build_supervision_targets(
                    env.task_manager,
                    env.uavs,
                    env._time_step,
                    allowed_edges,
                )
                if targets:
                    samples.append(GraphSupervisionSample(snapshot=snapshot, targets=targets))

            if action_mode == "zero":
                actions = np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)
            else:
                actions = np.random.uniform(-1.0, 1.0, size=(config.NUM_UAVS, config.ACTION_DIM)).astype(np.float32)
            env.step(actions)
            progress.update(
                postfix=f"episode {episode_idx + 1}/{num_episodes} step {step_idx + 1}/{steps_per_episode} samples {len(samples)}"
            )

    progress.finish(postfix=f"collected samples {len(samples)}")
    return samples


def save_supervision_dataset(samples: list[GraphSupervisionSample], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([sample.to_dict() for sample in samples], f, ensure_ascii=False, indent=2)
