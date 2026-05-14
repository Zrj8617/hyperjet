from marl_models.base_model import MARLModel
from marl_models.buffer_and_helpers import ReplayBuffer, RolloutBuffer, AttentionRolloutBuffer
from marl_models.utils import save_models
from environment.env import Env
from utils.logger import Logger, Log

# from utils.plot_snapshots import plot_snapshot  # snapshot plotting, comment if not needed

# from utils.plot_snapshots import update_trajectories, reset_trajectories  # trajectory tracking, comment if not needed
import config
import torch
import numpy as np
import time
import optuna


def _update_phase_one_diagnostics(accumulator: dict[str, float], diagnostics: dict[str, float]) -> None:
    for key, value in diagnostics.items():
        accumulator[key] = accumulator.get(key, 0.0) + float(value)


def _finalize_phase_one_diagnostics(accumulator: dict[str, float], episode_steps: int) -> dict[str, float]:
    if not accumulator or episode_steps <= 0:
        return {}
    averaged_keys = {
        "ready_tasks",
        "active_tasks",
        "feasible_edges",
        "score_edge_count",
        "selective_scoring_enabled",
        "selective_ready_tasks",
        "selective_high_risk_tasks",
        "selective_normal_tasks",
        "selective_high_risk_ratio",
        "selective_score_edges",
        "stage_b_reward_enabled",
        "stage_b_ready_coverage",
        "stage_b_assigned_coverage",
        "stage_b_ready_coverage_delta",
        "stage_b_assigned_coverage_delta",
        "stage_b_assigned_distance_reward",
        "stage_b_feasible_edge_score",
        "stage_b_progress_reward",
        "stage_b_local_finish_reward",
        "stage_b_local_on_time_finish_reward",
        "stage_b_dag_success_delta",
        "stage_b_dag_failure_delta",
        "stage_b_dag_shaping_reward",
        "stage_b_move_penalty",
        "stage_b_collision_penalty",
        "stage_b_boundary_penalty",
    }
    finalized: dict[str, float] = {}
    for key, value in accumulator.items():
        if key in averaged_keys:
            finalized[f"avg_{key}"] = value / float(episode_steps)
        else:
            finalized[key] = value
    return finalized


def _finalize_phase_one_episode_metrics(env: Env, accumulator: dict[str, float], episode_steps: int) -> dict[str, float]:
    metrics = _finalize_phase_one_diagnostics(accumulator, episode_steps)
    if config.ENABLE_DYNAMIC_DAG and config.ENABLE_PHASE_ONE_EXECUTION:
        metrics.update(env.task_manager.get_job_summary())
    return metrics


def train_on_policy(
    env: Env,
    model: MARLModel,
    logger: Logger,
    num_episodes: int,
    trial: optuna.Trial | None = None,
    progress_callback=None,
) -> float:
    start_time: float = time.time()
    BufferClass: type[RolloutBuffer] = AttentionRolloutBuffer if "attention" in model.model_name.lower() else RolloutBuffer
    buffer: RolloutBuffer = BufferClass(num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, buffer_size=config.PPO_ROLLOUT_LENGTH, device=model.device)
    max_time_steps: int = num_episodes * config.STEPS_PER_EPISODE
    num_updates: int = max_time_steps // config.PPO_ROLLOUT_LENGTH
    assert num_updates > 0, "num_updates is 0, please modify settings."
    save_freq: int = max(num_updates // 10, 100)

    recent_rewards: list[float] = []  # Tracking metrics for tuning

    recent_losses: dict = {"actor": None, "critic": None, "entropy": None}  # For logging most recent losses with episodes

    episode_log: Log = Log()
    episode: int = 1
    episode_step: int = 0
    episode_reward: float = 0.0
    episode_latency: float = 0.0
    episode_energy: float = 0.0
    episode_fairness: float = 0.0
    episode_offline_rate: float = 0.0
    episode_phase_one_diag: dict[str, float] = {}

    obs: list[np.ndarray] = env.reset()
    obs_arr: np.ndarray = np.asarray(obs, dtype=np.float32)
    state: np.ndarray = np.concatenate(obs, axis=0, dtype=np.float32)
    last_obs: list[np.ndarray] = obs

    # reset_trajectories(env)  # tracking code, comment if not needed
    # plot_snapshot(env, episode, 0, logger.log_dir, logger.timestamp, True)

    for update in range(1, num_updates + 1):
        for _ in range(1, config.PPO_ROLLOUT_LENGTH + 1):
            # if episode_step > 0 and episode_step % config.IMG_FREQ == 0:
            # plot_snapshot(env, episode, episode_step, logger.log_dir, logger.timestamp)

            raw_actions, log_probs, values = model.get_action_and_value(obs_arr, state)
            actions: np.ndarray = np.clip(raw_actions, -1.0, 1.0)

            next_obs, rewards, (total_latency, total_energy, jfi, offline_rate) = env.step(actions)
            next_state: np.ndarray = np.concatenate(next_obs, axis=0, dtype=np.float32)
            # update_trajectories(env)  # tracking code, comment if not needed

            episode_step += 1
            done: bool = episode_step >= config.STEPS_PER_EPISODE
            buffer.add(state, obs_arr, raw_actions, log_probs, rewards, done, values)
            obs = next_obs
            obs_arr = np.asarray(obs, dtype=np.float32)
            state = next_state
            last_obs = obs

            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness = jfi
            episode_offline_rate = offline_rate
            _update_phase_one_diagnostics(episode_phase_one_diag, env.latest_phase_one_diagnostics)

            if done:
                # plot_snapshot(env, episode, episode_step, logger.log_dir, logger.timestamp)  # Final snapshot of episode
                recent_rewards.append(episode_reward)
                episode_log.append(
                    episode_reward,
                    episode_latency,
                    episode_energy,
                    episode_fairness,
                    episode_offline_rate,
                    extra_metrics=_finalize_phase_one_episode_metrics(env, episode_phase_one_diag, episode_step),
                )

                # Optuna Pruning Check
                if trial:
                    current_avg_reward: float = float(np.mean(recent_rewards[-10:] if len(recent_rewards) >= 10 else recent_rewards))
                    trial.report(current_avg_reward, episode)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                if episode % config.LOG_FREQ == 0:
                    elapsed_time: float = time.time() - start_time
                    logger.log_metrics(episode, episode_log, config.LOG_FREQ, elapsed_time, losses=recent_losses)
                    if progress_callback is not None:
                        progress_callback(
                            episode,
                            episode_reward,
                            _finalize_phase_one_episode_metrics(env, episode_phase_one_diag, episode_step),
                            recent_losses,
                        )

                obs = env.reset()
                obs_arr = np.asarray(obs, dtype=np.float32)
                state = np.concatenate(obs, axis=0, dtype=np.float32)

                episode += 1
                episode_step = 0
                episode_reward, episode_latency, episode_energy, episode_fairness, episode_offline_rate = 0.0, 0.0, 0.0, 0.0, 0.0
                episode_phase_one_diag = {}
                # reset_trajectories(env)  # tracking code, comment if not needed
                # plot_snapshot(env, episode, 0, logger.log_dir, logger.timestamp, True)

        with torch.no_grad():
            last_obs_arr: np.ndarray = np.asarray(last_obs, dtype=np.float32)
            last_state: np.ndarray = np.concatenate(last_obs, axis=0, dtype=np.float32)
            _, _, last_values = model.get_action_and_value(last_obs_arr, last_state)

        buffer.compute_returns_and_advantages(last_values, config.DISCOUNT_FACTOR, config.PPO_GAE_LAMBDA)

        temp_losses: dict = {"actor": [], "critic": [], "entropy": []}  # Only for this update

        for _ in range(config.PPO_EPOCHS):
            for batch in buffer.get_batches(config.PPO_BATCH_SIZE):
                loss_dict = model.update(batch)
                if loss_dict:
                    temp_losses["actor"].append(loss_dict.get("actor"))
                    temp_losses["critic"].append(loss_dict.get("critic"))
                    temp_losses["entropy"].append(loss_dict.get("entropy"))

        buffer.clear()

        if temp_losses["actor"]:
            recent_losses = {
                "actor": float(np.mean([x for x in temp_losses["actor"] if x is not None])),
                "critic": float(np.mean([x for x in temp_losses["critic"] if x is not None])),
                "entropy": float(np.mean([x for x in temp_losses["entropy"] if x is not None])),
            }
        if update % save_freq == 0 and update < num_updates:
            save_models(model, update, "update", logger.timestamp)

    save_models(model, -1, "update", logger.timestamp, final=True)

    # Return average reward of last 10% of training for optimization score
    return float(np.mean(recent_rewards[-max(1, int(num_episodes * 0.1)) :]))


def train_off_policy(env: Env, model: MARLModel, logger: Logger, num_episodes: int, total_step_count: int, trial: optuna.Trial | None = None) -> float:
    start_time: float = time.time()
    buffer: ReplayBuffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
    save_freq: int = max(num_episodes // 10, 100)
    episode_log: Log = Log()

    accumulated_losses: dict = {"actor": [], "critic": []}
    has_alpha: bool = "sac" in model.model_name.lower()  # Only track alpha loss for SAC-based algorithms
    if has_alpha:
        accumulated_losses["alpha"] = []
    recent_rewards: list[float] = []  # Tracking metrics for tuning

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        model.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_offline_rate: float = 0.0
        episode_phase_one_diag: dict[str, float] = {}
        # reset_trajectories(env)  # tracking code, comment if not needed
        # plot_snapshot(env, episode, 0, logger.log_dir, logger.timestamp, True)

        for step in range(1, config.STEPS_PER_EPISODE + 1):
            # if step % config.IMG_FREQ == 0:
            # plot_snapshot(env, episode, step, logger.log_dir, logger.timestamp)

            total_step_count += 1
            obs_arr: np.ndarray = np.array(obs, dtype=np.float32)
            if total_step_count <= config.INITIAL_RANDOM_STEPS:
                actions: np.ndarray = np.array([np.random.uniform(-1, 1, config.ACTION_DIM) for _ in range(config.NUM_UAVS)])
            else:
                actions = model.select_actions(obs_arr, exploration=True)

            next_obs, rewards, (total_latency, total_energy, jfi, offline_rate) = env.step(actions)
            next_obs_arr: np.ndarray = np.array(next_obs, dtype=np.float32)
            # update_trajectories(env)  # tracking code, comment if not needed
            done: bool = step >= config.STEPS_PER_EPISODE
            buffer.add(obs_arr, actions, rewards, next_obs_arr, done)

            if (total_step_count > config.INITIAL_RANDOM_STEPS) and (step % config.LEARN_FREQ == 0) and (len(buffer) > config.REPLAY_BATCH_SIZE):
                batch = buffer.sample(config.REPLAY_BATCH_SIZE)
                loss_dict = model.update(batch)
                if loss_dict:
                    accumulated_losses["actor"].append(loss_dict.get("actor"))
                    accumulated_losses["critic"].append(loss_dict.get("critic"))
                    if has_alpha and "alpha" in loss_dict:
                        accumulated_losses["alpha"].append(loss_dict.get("alpha"))

            obs = next_obs

            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness = jfi
            episode_offline_rate = offline_rate
            _update_phase_one_diagnostics(episode_phase_one_diag, env.latest_phase_one_diagnostics)
            if done:
                break

        episode_log.append(
            episode_reward,
            episode_latency,
            episode_energy,
            episode_fairness,
            episode_offline_rate,
            extra_metrics=_finalize_phase_one_episode_metrics(env, episode_phase_one_diag, step),
        )
        if episode % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            # Prepare averaged losses for logging
            avg_losses: dict | None = None
            if accumulated_losses["actor"]:
                avg_losses = {
                    "actor": float(np.mean([x for x in accumulated_losses["actor"] if x is not None])),
                    "critic": float(np.mean([x for x in accumulated_losses["critic"] if x is not None])),
                }
                if has_alpha and accumulated_losses["alpha"]:
                    avg_losses["alpha"] = float(np.mean([x for x in accumulated_losses["alpha"] if x is not None]))
            logger.log_metrics(episode, episode_log, config.LOG_FREQ, elapsed_time, losses=avg_losses)
            # Reset accumulated losses for next logging interval
            accumulated_losses = {"actor": [], "critic": []}
            if has_alpha:
                accumulated_losses["alpha"] = []

        if episode % save_freq == 0 and episode < num_episodes:
            save_models(model, episode, "episode", logger.timestamp, total_steps=total_step_count)

        recent_rewards.append(episode_reward)
        if trial:
            # Report average of last 10 episodes
            current_avg_reward: float = float(np.mean(recent_rewards[-10:] if len(recent_rewards) >= 10 else recent_rewards))
            trial.report(current_avg_reward, episode)
            if trial.should_prune():
                raise optuna.TrialPruned()

    save_models(model, -1, "episode", logger.timestamp, final=True, total_steps=total_step_count)

    # Return average reward of last 10% of training for optimization score
    return float(np.mean(recent_rewards[-max(1, int(num_episodes * 0.1)) :]))


def train_random(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> float:
    start_time: float = time.time()
    episode_log: Log = Log()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_offline_rate: float = 0.0
        episode_phase_one_diag: dict[str, float] = {}
        # reset_trajectories(env)  # tracking code, comment if not needed
        # plot_snapshot(env, episode, 0, logger.log_dir, logger.timestamp, True)

        for step in range(1, config.STEPS_PER_EPISODE + 1):
            # if step % config.IMG_FREQ == 0:
            # plot_snapshot(env, episode, step, logger.log_dir, logger.timestamp)

            obs_arr: np.ndarray = np.array(obs, dtype=np.float32)
            actions: np.ndarray = model.select_actions(obs_arr, exploration=False)
            next_obs, rewards, (total_latency, total_energy, jfi, offline_rate) = env.step(actions)
            # update_trajectories(env)  # tracking code, comment if not needed
            done: bool = step >= config.STEPS_PER_EPISODE
            obs = next_obs

            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness = jfi
            episode_offline_rate = offline_rate
            _update_phase_one_diagnostics(episode_phase_one_diag, env.latest_phase_one_diagnostics)
            if done:
                break

        episode_log.append(
            episode_reward,
            episode_latency,
            episode_energy,
            episode_fairness,
            episode_offline_rate,
            extra_metrics=_finalize_phase_one_episode_metrics(env, episode_phase_one_diag, step),
        )
        if episode % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            logger.log_metrics(episode, episode_log, config.LOG_FREQ, elapsed_time, losses=None)

    return 0.0  # Random training does not need tuning
