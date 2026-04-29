import optuna
import argparse
import os
import numpy as np
import torch
import json
from datetime import datetime
import warnings
import optuna.visualization as vis

import config
from environment.env import Env
from marl_models.utils import get_model
from utils.logger import Logger
from train import train_on_policy, train_off_policy

# Suppress warnings for cleaner output during tuning
warnings.filterwarnings("ignore")


def objective(trial: optuna.Trial, stage: int, model_name: str, num_episodes: int) -> float:
    """
    Optuna Objective Function.
    Adjusts config based on 'stage' and runs a training session.

    Tuning Strategy:
    ================
    STAGE 1: Reward Function Weights (Objective Definition)
        - Tune ALPHA_1 (latency), ALPHA_2 (energy), ALPHA_3 (fairness), ALPHA_4 (offline rate)
        - Tune GDSF_SMOOTHING_FACTOR (caching decay)
        - Goal: Find the right balance of objectives

    STAGE 2: Agent Hyperparameters (Solver Tuning)
        - Tune ACTOR_LR, CRITIC_LR (learning rates)
        - Tune MLP_HIDDEN_DIM (network capacity)
        - Tune PPO_BATCH_SIZE (for on-policy) or REPLAY_BATCH_SIZE (for off-policy)
        - Tune DISCOUNT_FACTOR (temporal credit assignment)
        - Goal: Find optimal hyperparameters for the reward function from Stage 1

    STAGE 3: Architecture Hyperparameters (Attention-specific, if using attention models)
        - Tune ATTN_HIDDEN_DIM, ATTN_NUM_HEADS (attention layer capacity)
        - Goal: Optimize attention mechanism for the problem
    """

    # --- STAGE 1: Objective Tuning (Reward Weights & Caching) ---
    if stage == 1:
        # We tune the definition of "Success" first - what matters most in the problem?
        config.ALPHA_1 = trial.suggest_float("alpha_1", 1.0, 15.0, step=0.5)  # Latency penalty weight
        config.ALPHA_2 = trial.suggest_float("alpha_2", 0.1, 5.0, step=0.1)  # Energy penalty weight
        config.ALPHA_3 = trial.suggest_float("alpha_3", 1.0, 10.0, step=0.1)  # Fairness bonus weight
        config.ALPHA_4 = trial.suggest_float("alpha_4", 5.0, 100.0, step=5)  # Offline rate penalty (5-100)
        config.GDSF_SMOOTHING_FACTOR = trial.suggest_float("gdsf_beta", 0.1, 0.9, step=0.05)  # Cache EMA decay

    # --- STAGE 2: Agent Tuning (Hyperparameters) ---
    elif stage == 2:
        # We tune the solver to reach the success defined in Stage 1
        # NOTE: Before running Stage 2, you should load the best Stage 1 parameters:
        #   with open("tuning_logs/{model_name}/stage_1.json", "r") as f:
        #       best_params = json.load(f)["best_params"]
        #       config.ALPHA_1 = best_params["alpha_1"]
        #       config.ALPHA_2 = best_params["alpha_2"]
        #       ... (set all ALPHA and GDSF values)

        # --- Core Solver Parameters (All Algos) ---
        config.ACTOR_LR = trial.suggest_float("actor_lr", 1e-5, 5e-3, log=True)
        config.CRITIC_LR = trial.suggest_float("critic_lr", 1e-5, 5e-3, log=True)
        config.MLP_HIDDEN_DIM = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        config.DISCOUNT_FACTOR = trial.suggest_float("gamma", 0.95, 0.995)

        # --- Off-Policy Specific (MATD3, MASAC, MADDPG) ---
        if config.MODEL in ["matd3", "masac", "maddpg", "attention_matd3", "attention_masac", "attention_maddpg"]:
            config.REPLAY_BATCH_SIZE = trial.suggest_categorical("batch_size", [64, 128, 256])
            config.UPDATE_FACTOR = trial.suggest_float("tau", 0.005, 0.05)

            if config.MODEL in ["matd3", "attention_matd3"]:
                config.TARGET_POLICY_NOISE = trial.suggest_float("target_noise", 0.1, 0.3)

        # --- On-Policy Specific (MAPPO) ---
        elif config.MODEL in ["mappo", "attention_mappo"]:
            config.PPO_BATCH_SIZE = trial.suggest_categorical("batch_size", [64, 128, 256])
            config.PPO_CLIP_EPS = trial.suggest_float("clip_eps", 0.1, 0.3)
            config.PPO_ENTROPY_COEF = trial.suggest_float("entropy_coef", 0.001, 0.05, log=True)

    # --- STAGE 3: Attention Architecture (for attention-based models) ---
    elif stage == 3:
        if "attention" not in model_name.lower():
            raise ValueError(f"Stage 3 is only for attention models. Got: {model_name}")
        # Tune attention-specific hyperparameters
        # Note: ATTN_HIDDEN_DIM must be divisible by ATTN_NUM_HEADS (config.py validates this)
        config.ATTN_HIDDEN_DIM = trial.suggest_categorical("attn_hidden_dim", [32, 64, 128, 256])
        config.ATTN_NUM_HEADS = trial.suggest_categorical("attn_num_heads", [1, 2, 4, 8])
        # Ensure divisibility constraint
        while config.ATTN_HIDDEN_DIM % config.ATTN_NUM_HEADS != 0:
            config.ATTN_NUM_HEADS = trial.suggest_categorical("attn_num_heads", [1, 2, 4, 8])

    else:
        raise ValueError(f"Invalid stage: {stage}. Choose from [1, 2, 3]")

    # --- Setup Environment & Model ---
    np.random.seed(config.SEED + trial.number)  # Change seed per trial
    torch.manual_seed(config.SEED + trial.number)

    env = Env()
    model = get_model(model_name)

    # Minimal Logger for Tuning (Prevent cluttering disk with 100s of logs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_log_dir = f"tuning_logs/{model_name}/stage_{stage}/trial_{trial.number}"
    if not os.path.exists(tuning_log_dir):
        os.makedirs(tuning_log_dir)
    logger = Logger(tuning_log_dir, timestamp)

    # --- Execution ---
    try:
        final_score: float = 0.0
        if model_name in ["maddpg", "matd3", "masac", "attention_maddpg", "attention_matd3", "attention_masac"]:
            final_score = train_off_policy(env, model, logger, num_episodes, 0, trial)
        elif model_name in ["mappo", "attention_mappo"]:
            final_score = train_on_policy(env, model, logger, num_episodes, trial)
        else:
            raise ValueError(f"Unsupported model for tuning: {model_name}")

        return final_score

    except optuna.TrialPruned:
        raise  # Let Optuna handle the pruning exception
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float("-inf")  # Return lowest possible score on failure


def run_tuning(args):
    print(f"\n🎯 Starting Stage {args.stage} Tuning for {config.MODEL}...")
    print(f"📝 Episodes per trial: {args.episodes}")
    print(f"🔍 Trials: {args.trials}")

    # Timestamp used for per-run / per-trial logging
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Small JSON encoder to handle numpy types when saving trial summaries
    def _numpy_encoder(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # Callback executed after each trial completes (Optuna calls this)
    def _trial_logging_callback(study, trial):
        trial_log_dir = f"tuning_logs/{config.MODEL}/stage_{args.stage}/trial_{trial.number}"
        if not os.path.exists(trial_log_dir):
            os.makedirs(trial_log_dir)

        # Create a Logger for the trial and save current configs
        trial_logger = Logger(trial_log_dir, run_timestamp)
        try:
            trial_logger.log_configs()
        except Exception as e:
            print(f"⚠️ Could not save configs for trial {trial.number}: {e}")

        # Build trial summary and write to JSON
        summary = {
            "trial_number": trial.number,
            "params": trial.params,
            "value": None if trial.value is None else float(trial.value),
            "best_value": None if study.best_value is None else float(study.best_value),
            "best_params": study.best_params,
        }

        summary_path = os.path.join(trial_log_dir, f"trial_{trial.number}_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as sf:
                json.dump(summary, sf, indent=4, default=_numpy_encoder)
        except Exception as e:
            print(f"⚠️ Could not write trial summary for trial {trial.number}: {e}")

    # Use MedianPruner for Early Stopping
    # It stops a trial if its intermediate result is worse than the median of previous trials
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Don't prune the first 5 trials (let them complete)
        n_warmup_steps=10,  # Don't prune early steps of any trial
        interval_steps=1,  # Check for pruning at every report
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{config.MODEL}_stage_{args.stage}",
        pruner=pruner,
    )

    def objective_wrapper(trial):
        return objective(trial, args.stage, config.MODEL.lower(), args.episodes)

    study.optimize(objective_wrapper, n_trials=args.trials, callbacks=[_trial_logging_callback])

    print("\n🏆 Tuning Completed!")
    print(f"Best Trial Score: {study.best_value}")
    print(f"Best Trial Number: {study.best_trial.number}")
    print("Best Parameters:")
    print(json.dumps(study.best_params, indent=4))

    # Save best params and study summary
    save_path = f"tuning_logs/{config.MODEL}/stage_{args.stage}/stage_{args.stage}.json"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"💾 Saved best parameters to {save_path}")

    # Generate plots
    try:
        plot_tuning_results(study, config.MODEL, args.stage)
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")


def plot_tuning_results(study: optuna.Study, model_name: str, stage: int) -> None:
    """Generates and saves tuning result plots."""

    plot_dir = f"tuning_logs/{model_name}/plots_stage_{stage}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Parameter Importance
    fig = vis.plot_param_importances(study)
    fig.write_image(f"{plot_dir}/param_importance.png")

    # Optimization History
    fig = vis.plot_optimization_history(study)
    fig.write_image(f"{plot_dir}/optimization_history.png")

    # Slice Plot - Shows individual parameter effects
    fig = vis.plot_slice(study)
    fig.write_image(f"{plot_dir}/slice_plot.png")

    # Intermediate Values - Shows learning curves across trials
    fig = vis.plot_intermediate_values(study)
    fig.write_image(f"{plot_dir}/intermediate_values.png")

    print(f"📊 Saved tuning plots to {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning Module",
        epilog="""
Examples:
  Stage 1 (Objective tuning):
    python tune.py --stage 1 --episodes 500 --trials 50
    
  Stage 2 (Agent hyperparameter tuning - after loading Stage 1 best params):
    python tune.py --stage 2 --episodes 1000 --trials 50
    
  Stage 3 (Attention architecture tuning - attention models only):
    python tune.py --stage 3 --episodes 500 --trials 30
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="1: Tune Rewards/Env, 2: Tune Agent Hyperparams, 3: Tune Attention Architecture (attention models only)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Episodes per trial (Lower than full training)",
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to run")

    args = parser.parse_args()
    run_tuning(args)
