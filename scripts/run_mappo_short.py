from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

import config
from environment.env import Env
from marl_models.mappo.mappo import MAPPO
from train import train_on_policy
from utils.logger import Logger


def _refresh_dimension_config() -> None:
    phase_one_obs = (
        config.ENABLE_DYNAMIC_DAG
        and config.ENABLE_PHASE_ONE_EXECUTION
        and not config.ENABLE_LEGACY_REQUEST_PIPELINE
        and config.USE_PHASE_ONE_DEDICATED_OBS
    )
    compact_obs = phase_one_obs and config.USE_MAPPO_COMPACT_OBS
    config.MAX_UAV_NEIGHBORS = max(config.NUM_UAVS - 1, 1)
    config.MAX_ASSOCIATED_UES = min(30, config.NUM_UES // max(config.NUM_UAVS, 1) + 10)
    config.SELF_OBS_DIM = config.PHASE_ONE_SELF_OBS_DIM if phase_one_obs else config.LEGACY_SELF_OBS_DIM
    config.UE_OBS_DIM = (
        config.MAPPO_COMPACT_LOCAL_OBS_DIM
        if compact_obs
        else config.PHASE_ONE_TASK_OBS_DIM
        if phase_one_obs
        else config.LEGACY_UE_OBS_DIM
    )
    config.NEIGHBOR_OBS_DIM = config.PHASE_ONE_NEIGHBOR_OBS_DIM if phase_one_obs else config.LEGACY_NEIGHBOR_OBS_DIM
    config.OBS_DIM_SINGLE = (
        config.SELF_OBS_DIM + (config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM) + config.UE_OBS_DIM
        if compact_obs
        else config.SELF_OBS_DIM
        + (config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM)
        + (config.MAX_ASSOCIATED_UES * config.UE_OBS_DIM)
    )


def _override_num_uavs(num_uavs: int, seed: int) -> None:
    if num_uavs <= 1:
        raise ValueError("--num_uavs must be greater than 1.")
    config.NUM_UAVS = num_uavs
    rng = np.random.default_rng(seed)
    config.UAV_STORAGE_CAPACITY = rng.choice(
        np.arange(40 * 10**6, 80 * 10**6, 10**6),
        size=config.NUM_UAVS,
    ).astype(np.int64)
    config.UAV_COMPUTING_CAPACITY = rng.choice(
        np.arange(5 * 10**9, 20 * 10**9, 10**9),
        size=config.NUM_UAVS,
    ).astype(np.int64)
    _refresh_dimension_config()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short phase-one MAPPO training job.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--rollout", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--ppo_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num_uavs", type=int, default=None)
    parser.add_argument("--dag_arrival_prob", type=float, default=None)
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=[
            "full",
            "no_hyperedge",
            "no_service_domain",
            "no_resource_competition",
            "no_critical",
        ],
        help="HGNN hyperedge ablation used during RL training.",
    )
    parser.add_argument("--disable_hgnn_score", action="store_true")
    parser.add_argument("--selective_hgnn_score", action="store_true")
    parser.add_argument("--disable_dag_reward_shaping", action="store_true")
    parser.add_argument("--dag_failure_penalty", type=float, default=None)
    parser.add_argument("--stage_b_movement_reward", action="store_true")
    parser.add_argument("--tag", type=str, default="rl_short")
    parser.add_argument("--log_dir", type=str, default="train_logs/mappo_short")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    if args.num_uavs is not None:
        _override_num_uavs(args.num_uavs, args.seed)
    if args.dag_arrival_prob is not None:
        if not 0.0 <= args.dag_arrival_prob <= 1.0:
            raise ValueError("--dag_arrival_prob must be in [0, 1].")
        config.DAG_ARRIVAL_PROB = args.dag_arrival_prob

    config.STEPS_PER_EPISODE = args.steps
    config.PPO_ROLLOUT_LENGTH = args.rollout if args.rollout is not None else args.steps
    config.PPO_BATCH_SIZE = args.batch_size
    config.PPO_EPOCHS = args.ppo_epochs
    config.USE_HGNN_SCORE_ASSIGNMENT = not args.disable_hgnn_score
    config.USE_SELECTIVE_HGNN_SCORING = args.selective_hgnn_score and not args.disable_hgnn_score
    config.HGNN_SCORE_CHECKPOINT = "" if args.disable_hgnn_score else args.checkpoint
    config.USE_MAPPO_COMPACT_OBS = True
    config.USE_PHASE_ONE_DEDICATED_OBS = True
    config.USE_PHASE_ONE_DAG_REWARD_SHAPING = not args.disable_dag_reward_shaping
    if args.dag_failure_penalty is not None:
        config.PHASE_ONE_DAG_FAILURE_PENALTY = args.dag_failure_penalty
    config.USE_STAGE_B_MOVEMENT_REWARD = args.stage_b_movement_reward
    config.USE_PHASE_ONE_HYPEREDGES = args.ablation != "no_hyperedge"
    config.USE_COLLABORATIVE_HYPEREDGES = args.ablation != "no_hyperedge"
    config.USE_SERVICE_DOMAIN_HYPEREDGES = args.ablation not in {"no_hyperedge", "no_service_domain"}
    config.USE_RESOURCE_COMPETITION_HYPEREDGES = args.ablation not in {"no_hyperedge", "no_resource_competition"}
    config.USE_CRITICAL_HYPEREDGES = args.ablation not in {"no_hyperedge", "no_critical"}
    config.USE_ATTRIBUTE_HYPEREDGES = False
    config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES = False
    config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES = False
    config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES = False

    if config.USE_HGNN_SCORE_ASSIGNMENT and not config.HGNN_SCORE_CHECKPOINT:
        raise ValueError("--checkpoint is required unless --disable_hgnn_score is set.")
    if config.HGNN_SCORE_CHECKPOINT and not os.path.exists(config.HGNN_SCORE_CHECKPOINT):
        raise FileNotFoundError(f"HGNN checkpoint not found: {config.HGNN_SCORE_CHECKPOINT}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = Env()
    model = MAPPO(
        model_name="mappo",
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device=args.device,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    logger = Logger(args.log_dir, args.tag)
    logger.log_configs()
    print(f"tag={args.tag}")
    print(f"episodes={args.episodes} steps={config.STEPS_PER_EPISODE} rollout={config.PPO_ROLLOUT_LENGTH}")
    print(f"num_uavs={config.NUM_UAVS} dag_arrival_prob={config.DAG_ARRIVAL_PROB}")
    print(f"obs_dim={config.OBS_DIM_SINGLE} action_dim={config.ACTION_DIM}")
    print(f"use_hgnn_score={config.USE_HGNN_SCORE_ASSIGNMENT}")
    print(f"use_selective_hgnn_score={config.USE_SELECTIVE_HGNN_SCORING}")
    print(f"ablation={args.ablation}")
    print(
        "hyperedges="
        f"phase_one:{config.USE_PHASE_ONE_HYPEREDGES}, "
        f"service:{config.USE_SERVICE_DOMAIN_HYPEREDGES}, "
        f"resource:{config.USE_RESOURCE_COMPETITION_HYPEREDGES}, "
        f"critical:{config.USE_CRITICAL_HYPEREDGES}, "
        f"attribute:{config.USE_ATTRIBUTE_HYPEREDGES}"
    )
    print(f"hgnn_checkpoint={config.HGNN_SCORE_CHECKPOINT}")
    print(f"use_dag_reward_shaping={config.USE_PHASE_ONE_DAG_REWARD_SHAPING}")
    print(f"dag_failure_penalty={config.PHASE_ONE_DAG_FAILURE_PENALTY}")
    print(f"use_stage_b_movement_reward={config.USE_STAGE_B_MOVEMENT_REWARD}")

    progress_bar = None
    progress_callback = None
    if not args.no_progress:
        progress_bar = tqdm(
            total=args.episodes,
            desc=args.tag,
            unit="episode",
            dynamic_ncols=True,
            leave=True,
            mininterval=1.0,
        )

        def _progress_callback(episode: int, reward: float, metrics: dict[str, float], losses: dict) -> None:
            last_episode = getattr(_progress_callback, "last_episode", 0)
            progress_bar.update(max(episode - last_episode, 0))
            _progress_callback.last_episode = episode
            progress_bar.set_postfix(
                reward=f"{reward:.1f}",
                dag=f"{metrics.get('dag_success_rate', 0.0):.3f}",
                drop=f"{metrics.get('dag_task_drop_rate', 0.0):.3f}",
                critic=f"{losses.get('critic'):.2f}" if losses.get("critic") is not None else "n/a",
            )

        progress_callback = _progress_callback

    score = train_on_policy(env, model, logger, num_episodes=args.episodes, progress_callback=progress_callback)
    if progress_bar is not None:
        progress_bar.close()
    print(f"mappo_short_score={score}")


if __name__ == "__main__":
    main()
