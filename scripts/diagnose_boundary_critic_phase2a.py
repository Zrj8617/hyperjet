"""Phase 2-A offline Boundary critic feasibility audit.

The script collects complete frozen-policy episodes at slot boundaries, trains
an independent critic on realized discounted slot returns, and evaluates it on
held-out episodes.  It never updates or feeds data into PPO, Decision-Q, the
actor, the encoder, or the rollout buffer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_ppo import CleanCentralizedCritic, CleanDecisionCritic
from marl_models.mappo.clean_slot_orchestrator import encode_prepared_slot, prepare_slot_state
from marl_models.mappo.clean_trainer import _set_rng_state
from scripts.diagnose_decision_q_v2_ranking_crn import (
    _clone_parameters,
    _parameters_equal,
)
from scripts.eval_clean_mainline import (
    _build_modules,
    _load_module_state,
    _load_trusted_checkpoint,
    _module_dims_from_checkpoint,
    _set_eval_mode,
)
from scripts.train_clean_mainline import checkpoint_experiment_controls


BOUNDARY_STATE_SEMANTICS = (
    "slot-start/slot-boundary state before movement and before all within-slot reservations"
)


@dataclass(slots=True)
class BoundaryEpisode:
    episode_index: int
    states: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    returns: np.ndarray


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 2-A diagnostic-only offline Boundary critic audit."
    )
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    return parser


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise ValueError("--checkpoint must identify seed0/update120 for smoke")
    episodes = 2 if args.episodes is None else int(args.episodes)
    epochs = 4 if args.epochs is None else int(args.epochs)
    if episodes != 2:
        raise ValueError("Phase 2-A smoke uses exactly two complete episodes")
    if epochs != 4:
        raise ValueError("Phase 2-A smoke uses exactly four offline epochs")
    result = _run_seed(
        seed=0,
        checkpoint=args.checkpoint,
        episodes=episodes,
        epochs=epochs,
        checkpoint_epochs=(1, 2, 4),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        hidden_dim=int(args.hidden_dim),
        output_dir=args.output / "smoke" / "seed0",
    )
    summary = _combined_summary([result], mode="smoke")
    return _write_combined(args.output / "smoke", summary, [result])


def run_formal(args: argparse.Namespace) -> dict[str, Any]:
    if args.experiment_root is None or not args.experiment_root.is_dir():
        raise ValueError("--experiment-root is required for formal mode")
    episodes = 8 if args.episodes is None else int(args.episodes)
    epochs = 60 if args.epochs is None else int(args.epochs)
    if episodes < 4:
        raise ValueError("formal audit requires at least four complete episodes per seed")
    if epochs < 3:
        raise ValueError("formal audit requires at least three epochs")
    checkpoint_epochs = _formal_checkpoint_epochs(epochs)
    results: list[dict[str, Any]] = []
    output_dir = args.output / "formal"
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in (0, 1, 2):
        run_dir = _single_path(
            args.experiment_root.glob(f"runs/seed{seed}/*"),
            f"seed {seed} run directory",
            require_directory=True,
        )
        checkpoint = run_dir / "checkpoints" / "checkpoint_update_0120.pt"
        result = _run_seed(
            seed=seed,
            checkpoint=checkpoint,
            episodes=episodes,
            epochs=epochs,
            checkpoint_epochs=checkpoint_epochs,
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            hidden_dim=int(args.hidden_dim),
            output_dir=output_dir / f"seed{seed}",
        )
        results.append(result)
        (output_dir / "boundary_critic_summary.partial.json").write_text(
            json.dumps(_combined_summary(results, mode="formal-partial"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"completed Boundary critic seed={seed}", flush=True)
    return _write_combined(
        output_dir,
        _combined_summary(results, mode="formal"),
        results,
    )


def _run_seed(
    *,
    seed: int,
    checkpoint: Path,
    episodes: int,
    epochs: int,
    checkpoint_epochs: tuple[int, int, int],
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    payload = _load_trusted_checkpoint(torch, checkpoint)
    if payload.get("resume_semantics") != "restart_from_new_episode_only":
        raise ValueError("checkpoint does not use new-episode replay semantics")
    if int(payload.get("update_step", -1)) != 120:
        raise ValueError("Phase 2-A uses the frozen update120 policy checkpoint")
    controls = checkpoint_experiment_controls(payload)
    device = torch.device("cpu")
    dims = _module_dims_from_checkpoint(
        payload, argparse.Namespace(task_embedding_dim=None, hidden_dim=None)
    )
    modules = _build_modules(dims=dims, experiment_controls=controls, device=device)
    _load_module_state(modules, payload)
    _set_eval_mode(modules)
    q_state = payload.get("extra_state", {}).get("offloading_decision_q_credit")
    if q_state is None:
        raise ValueError("checkpoint is missing the frozen Decision-Q critic")
    q_first_weight = q_state["critic"]["net.0.weight"]
    q_critic = CleanDecisionCritic(
        input_dim=int(q_first_weight.shape[1]),
        hidden_dim=int(q_first_weight.shape[0]),
    ).to(device)
    q_critic.load_state_dict(q_state["critic"])
    q_critic.eval()

    frozen_modules = {
        "task_encoder": modules.hgnn,
        "movement_actor": modules.movement_actor,
        "offloading_actor": modules.offloading_actor,
        "main_critic": modules.critic,
        "decision_q_critic": q_critic,
    }
    for module in frozen_modules.values():
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    frozen_before = _clone_parameters(frozen_modules)

    gamma = float(q_state.get("gamma", controls.get("gamma", 0.99)))
    collected = _collect_complete_episodes(
        payload=payload,
        controls=controls,
        modules=modules,
        episode_count=int(episodes),
        gamma=gamma,
        device=device,
    )
    input_dims = {int(episode.states.shape[1]) for episode in collected}
    if len(input_dims) != 1:
        raise AssertionError("B_t dimensions changed across episodes")
    input_dim = next(iter(input_dims))
    if not all(
        episode.states.shape[0] == episode.rewards.shape[0]
        == episode.terminals.shape[0] == episode.returns.shape[0]
        for episode in collected
    ):
        raise AssertionError("boundary transitions are not one-to-one")
    if not all(bool(episode.terminals[-1]) for episode in collected):
        raise AssertionError("a collected episode lacks a terminal boundary")
    if not all(int(np.count_nonzero(episode.terminals)) == 1 for episode in collected):
        raise AssertionError("terminal handling is not unique per complete episode")

    validation_count = max(1, int(episodes) // 4)
    train_episodes = collected[:-validation_count]
    validation_episodes = collected[-validation_count:]
    train_data = _flatten_episodes(train_episodes)
    validation_data = _flatten_episodes(validation_episodes)
    for data in (train_data, validation_data):
        terminal_rows = np.flatnonzero(data[2] > 0.5)
        if terminal_rows.size == 0 or not np.all(data[3][terminal_rows] == 0.0):
            raise AssertionError("terminal B_t transitions do not use zero next value state")
    if not all(np.isfinite(value).all() for value in (*train_data, *validation_data)):
        raise FloatingPointError("Boundary dataset contains NaN/Inf")

    torch.manual_seed(20260902 + int(seed))
    boundary_critic = CleanCentralizedCritic(
        input_dim=input_dim,
        hidden_dim=int(hidden_dim),
        task_pooling=str(getattr(modules.critic, "task_pooling", "mean")),
    ).to(device)
    optimizer = torch.optim.Adam(boundary_critic.parameters(), lr=float(learning_rate))
    boundary_parameter_ids = {id(parameter) for parameter in boundary_critic.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameter_ids != boundary_parameter_ids:
        raise AssertionError("optimizer contains parameters outside Boundary critic")
    if any(
        id(parameter) in optimizer_parameter_ids
        for module in frozen_modules.values()
        for parameter in module.parameters()
    ):
        raise AssertionError("optimizer includes a frozen mainline parameter")

    output_dir.mkdir(parents=True, exist_ok=True)
    training_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _train_one_epoch(
            critic=boundary_critic,
            optimizer=optimizer,
            data=train_data,
            batch_size=int(batch_size),
            seed=2026090200 + 1000 * int(seed) + epoch,
            device=device,
            gamma=gamma,
        )
        validation_metrics = _evaluate(
            critic=boundary_critic,
            data=validation_data,
            device=device,
            gamma=gamma,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        training_rows.append(row)
        if epoch in checkpoint_epochs:
            label = {checkpoint_epochs[0]: "early", checkpoint_epochs[1]: "mid", checkpoint_epochs[2]: "late"}[epoch]
            checkpoint_path = output_dir / f"boundary_critic_{label}_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "schema": "phase2a_boundary_critic_v1",
                    "seed": int(seed),
                    "checkpoint_label": label,
                    "epoch": int(epoch),
                    "input_dim": int(input_dim),
                    "hidden_dim": int(hidden_dim),
                    "gamma": float(gamma),
                    "state_semantics": BOUNDARY_STATE_SEMANTICS,
                    "critic": boundary_critic.state_dict(),
                },
                checkpoint_path,
            )
            checkpoint_rows.append(
                {
                    "seed": int(seed),
                    "checkpoint": label,
                    "epoch": int(epoch),
                    "path": str(checkpoint_path),
                    **validation_metrics,
                }
            )
    if not _parameters_equal(frozen_before, _clone_parameters(frozen_modules)):
        raise AssertionError("Boundary audit changed actor/encoder/main critic/Decision-Q")
    if not all(
        math.isfinite(float(value))
        for row in training_rows
        for scope in (row["train"], row["validation"])
        for value in scope.values()
        if value is not None
    ):
        raise FloatingPointError("Boundary training metrics contain NaN/Inf")

    metrics_path = output_dir / "boundary_critic_metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in training_rows),
        encoding="utf-8",
    )
    result = {
        "schema": "phase2a_boundary_critic_seed_summary_v1",
        "seed": int(seed),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_update": 120,
        "state_semantics": BOUNDARY_STATE_SEMANTICS,
        "input_dim": int(input_dim),
        "gamma": float(gamma),
        "training_target": "realized_smdp_slot_return_G_t=r_t+gamma*G_t_plus_1",
        "terminal_bootstrap": 0.0,
        "episode_count": int(episodes),
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "train_transition_count": int(train_data[0].shape[0]),
        "validation_transition_count": int(validation_data[0].shape[0]),
        "checkpoints": checkpoint_rows,
        "metrics_jsonl": str(metrics_path),
        "gates": {
            "boundary_dimensions_consistent": True,
            "terminal_next_value_zero": True,
            "all_finite": True,
            "optimizer_only_boundary_critic": True,
            "actor_decision_q_encoder_main_critic_unchanged": True,
            "rollout_buffer_used": False,
            "decision_q_target_used": False,
            "c_local_used": False,
            "counterfactual_credit_used": False,
        },
    }
    (output_dir / "seed_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _collect_complete_episodes(
    *,
    payload: dict[str, Any],
    controls: dict[str, Any],
    modules: Any,
    episode_count: int,
    gamma: float,
    device: Any,
) -> list[BoundaryEpisode]:
    import torch
    from torch.distributions import Categorical

    _set_rng_state(payload.get("rng_state", {}))
    episodes: list[BoundaryEpisode] = []
    replay_episode_start = int(payload.get("episode", -1)) + 1
    for episode_offset in range(int(episode_count)):
        env = Env(
            completed_dag_weight=float(controls["completed_dag_weight"]),
            freeze_ue_mobility=bool(controls.get("freeze_ue_mobility", False)),
        )
        builder = CleanGraphBuilder()
        builder.reset()
        env.reset()
        states: list[np.ndarray] = []
        rewards: list[float] = []
        terminals: list[bool] = []
        try:
            for _ in range(int(config.EPISODE_LENGTH)):
                with torch.no_grad():
                    prepared = prepare_slot_state(env=env, graph_builder=builder)
                    encoded = encode_prepared_slot(
                        prepared_state=prepared,
                        env=env,
                        hgnn=modules.hgnn,
                        critic=modules.critic,
                        movement_actor=modules.movement_actor,
                        device=device,
                        detach_critic_hgnn=bool(controls.get("detach_critic_hgnn", False)),
                    )
                    boundary_state = np.asarray(
                        encoded.critic_global_input.detach().cpu(), dtype=np.float32
                    ).reshape(-1).copy()
                    movement = Categorical(logits=encoded.movement_logits).sample()
                    env.apply_movement(
                        {
                            int(uav_id): int(movement[index].cpu().item())
                            for index, uav_id in enumerate(encoded.movement_observation.uav_ids)
                        }
                    )
                    ready = [
                        env.task_manager.get_task(task_id)
                        for task_id in prepared.frozen_ready_task_ids
                    ]
                    ready = [task for task in ready if task is not None and task.is_ready]
                    assignments = modules.offloading_actor.act(
                        frozen_ready_tasks=ready,
                        task_embeddings=encoded.task_embeddings.detach(),
                        graph_snapshot=prepared.graph_snapshot,
                        task_manager=env.task_manager,
                        uavs=env.uavs,
                        executor=env.executor,
                        current_time_seconds=env.current_time_seconds,
                        uav_service_positions=env.uav_service_positions,
                        ue_service_positions=env.ue_service_positions,
                        ues=env.ues,
                        deterministic=False,
                    )
                    _, _, done, info = env.commit_and_advance(
                        assignment_buffer=assignments
                    )
                states.append(boundary_state)
                rewards.append(float(info["step_reward"]))
                terminals.append(bool(done))
                if done:
                    break
        finally:
            builder.close()
        if not terminals or not terminals[-1]:
            raise RuntimeError("frozen replay did not reach the terminal slot boundary")
        rewards_array = np.asarray(rewards, dtype=np.float32)
        terminal_array = np.asarray(terminals, dtype=bool)
        returns = _discounted_returns(rewards_array, terminal_array, gamma)
        episodes.append(
            BoundaryEpisode(
                episode_index=replay_episode_start + episode_offset,
                states=np.asarray(states, dtype=np.float32),
                rewards=rewards_array,
                terminals=terminal_array,
                returns=returns,
            )
        )
    return episodes


def _discounted_returns(
    rewards: np.ndarray, terminals: np.ndarray, gamma: float
) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    next_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if bool(terminals[index]):
            next_return = 0.0
        next_return = float(rewards[index]) + float(gamma) * next_return
        returns[index] = float(next_return)
    return returns


def _flatten_episodes(
    episodes: list[BoundaryEpisode],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    terminals: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    for episode in episodes:
        zero_boundary = np.zeros((1, episode.states.shape[1]), dtype=np.float32)
        shifted = np.concatenate([episode.states[1:], zero_boundary], axis=0)
        states.append(episode.states)
        rewards.append(episode.rewards)
        terminals.append(episode.terminals.astype(np.float32))
        next_states.append(shifted)
        returns.append(episode.returns)
    return (
        np.concatenate(states, axis=0),
        np.concatenate(rewards, axis=0),
        np.concatenate(terminals, axis=0),
        np.concatenate(next_states, axis=0),
        np.concatenate(returns, axis=0),
    )


def _train_one_epoch(
    *,
    critic: Any,
    optimizer: Any,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    seed: int,
    device: Any,
    gamma: float,
) -> dict[str, float | None]:
    import torch

    states, _, _, _, returns = data
    order = np.random.RandomState(int(seed)).permutation(states.shape[0])
    critic.train()
    losses: list[float] = []
    for start in range(0, len(order), int(batch_size)):
        indices = order[start : start + int(batch_size)]
        state_tensor = torch.as_tensor(states[indices], dtype=torch.float32, device=device)
        target_tensor = torch.as_tensor(returns[indices], dtype=torch.float32, device=device)
        prediction = critic(state_tensor)
        loss = 0.5 * (prediction - target_tensor).pow(2).mean()
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Boundary critic loss is NaN/Inf")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    metrics = _evaluate(critic=critic, data=data, device=device, gamma=gamma)
    metrics["critic_loss"] = float(np.mean(losses))
    return metrics


def _evaluate(
    *,
    critic: Any,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: Any,
    gamma: float,
) -> dict[str, float | None]:
    import torch

    states, rewards, terminals, next_states, returns = data
    critic.eval()
    with torch.no_grad():
        predictions = critic(
            torch.as_tensor(states, dtype=torch.float32, device=device)
        ).detach().cpu().numpy().astype(np.float64)
        next_predictions = critic(
            torch.as_tensor(next_states, dtype=torch.float32, device=device)
        ).detach().cpu().numpy().astype(np.float64)
    targets = returns.astype(np.float64)
    td_targets = rewards.astype(np.float64) + float(gamma) * (
        1.0 - terminals.astype(np.float64)
    ) * next_predictions
    td_errors = td_targets - predictions
    residuals = targets - predictions
    target_variance = float(np.var(targets))
    explained_variance = (
        0.0
        if target_variance <= 1e-12
        else 1.0 - float(np.var(residuals)) / target_variance
    )
    correlation = None
    if predictions.size >= 2 and np.std(predictions) > 0.0 and np.std(targets) > 0.0:
        correlation = float(np.corrcoef(predictions, targets)[0, 1])
    return {
        "critic_loss": float(0.5 * np.mean(residuals * residuals)),
        "value_explained_variance": float(explained_variance),
        "td_error_mean": float(np.mean(td_errors)),
        "td_error_std": float(np.std(td_errors)),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions)),
        "target_mean": float(np.mean(targets)),
        "target_std": float(np.std(targets)),
        "calibration_mae": float(np.mean(np.abs(residuals))),
        "calibration_correlation": correlation,
        "calibration_explained_variance": float(explained_variance),
    }


def _combined_summary(results: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    checkpoint_rows = [row for result in results for row in result["checkpoints"]]
    labels = ("early", "mid", "late")
    by_checkpoint = {
        label: _aggregate_checkpoint(
            [row for row in checkpoint_rows if str(row["checkpoint"]) == label]
        )
        for label in labels
    }
    return {
        "schema": "phase2a_boundary_critic_summary_v1",
        "mode": str(mode),
        "state_semantics": BOUNDARY_STATE_SEMANTICS,
        "source_policy_checkpoint_update": 120,
        "training_target": "realized_smdp_slot_return_G_t=r_t+gamma*G_t_plus_1",
        "terminal_bootstrap": 0.0,
        "seed_count": len(results),
        "seeds": [int(result["seed"]) for result in results],
        "checkpoint_metrics": checkpoint_rows,
        "by_checkpoint": by_checkpoint,
        "gates": {
            "all_boundary_dimensions_consistent": all(
                result["gates"]["boundary_dimensions_consistent"] for result in results
            ),
            "all_terminal_next_value_zero": all(
                result["gates"]["terminal_next_value_zero"] for result in results
            ),
            "all_finite": all(result["gates"]["all_finite"] for result in results),
            "all_optimizers_boundary_only": all(
                result["gates"]["optimizer_only_boundary_critic"] for result in results
            ),
            "all_mainline_parameters_unchanged": all(
                result["gates"]["actor_decision_q_encoder_main_critic_unchanged"]
                for result in results
            ),
            "rollout_buffer_used": False,
            "decision_q_target_used": False,
            "c_local_used": False,
            "counterfactual_credit_used": False,
        },
    }


def _aggregate_checkpoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "critic_loss",
        "value_explained_variance",
        "td_error_mean",
        "td_error_std",
        "prediction_mean",
        "prediction_std",
        "target_mean",
        "target_std",
        "calibration_mae",
        "calibration_correlation",
        "calibration_explained_variance",
    )
    return {
        "seed_count": len(rows),
        **{
            name: (
                None
                if not [row[name] for row in rows if row[name] is not None]
                else float(np.mean([row[name] for row in rows if row[name] is not None]))
            )
            for name in names
        },
    }


def _write_combined(
    output_dir: Path,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "boundary_critic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "phase2a_boundary_critic_manifest_v1",
                "state_semantics": BOUNDARY_STATE_SEMANTICS,
                "source_checkpoints": [result["source_checkpoint"] for result in results],
                "seed_outputs": [result["metrics_jsonl"] for result in results],
                "gates": summary["gates"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "summary_json": str(summary_path),
        "manifest_json": str(manifest_path),
        "summary": summary,
    }


def _formal_checkpoint_epochs(epochs: int) -> tuple[int, int, int]:
    return (max(1, int(epochs) // 6), max(2, int(epochs) // 2), int(epochs))


def _single_path(paths: Any, label: str, *, require_directory: bool = False) -> Path:
    values = [path for path in paths if not require_directory or path.is_dir()]
    if len(values) != 1:
        raise ValueError(f"expected one {label}, found {len(values)}")
    return values[0]


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_smoke(args) if args.mode == "smoke" else run_formal(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
