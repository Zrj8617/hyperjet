from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.hgnn import build_clean_task_encoder
from marl_models.mappo.clean_decision_ppo_bandit import (
    DECISION_BANDIT_REGRET_SCALE,
    DecisionBanditPPOUpdater,
    DecisionBanditRolloutBuffer,
    DecisionBanditSkipEvent,
    DecisionBanditUpdateConfig,
    build_decision_bandit_record,
)
from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state


CHECKPOINT_SCHEMA = "decision_ppo_bandit_stage1_checkpoint_v1"
GROUPS = ("S1-A", "S1-B")
CHECKPOINT_UPDATES = (0, 1, 5, 10, 20, 30)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1 executed-action decision PPO gate.")
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--slots-per-update", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--chunk-decisions", type=int, default=64)
    parser.add_argument("--task-embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--completed-dag-weight", type=float, default=16.0)
    parser.add_argument("--freeze-ue-mobility", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("logs") / "decision_ppo_bandit")
    parser.add_argument("--run-name", type=str, default="stage1")
    parser.add_argument("--checkpoint-updates", type=str, default="0,1,5,10,20,30")
    parser.add_argument("--pilot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_training(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    import torch

    if bool(args.pilot):
        args.updates = min(int(args.updates), 2)
    _set_seed(int(args.seed), torch)
    device = torch.device(str(args.device))
    env = Env(
        completed_dag_weight=float(args.completed_dag_weight),
        freeze_ue_mobility=bool(args.freeze_ue_mobility),
    )
    graph_builder = CleanGraphBuilder()
    env.reset()
    graph_builder.reset()
    prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
    task_feature_dim = int(prepared.graph_snapshot.task_features.shape[1])
    encoder = build_clean_task_encoder(
        encoder_type="mlp",
        task_feature_dim=task_feature_dim,
        hidden_dim=int(args.hidden_dim),
        output_dim=int(args.task_embedding_dim),
    ).to(device)
    actor = CleanOffloadingActor(
        task_embedding_dim=int(args.task_embedding_dim),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(actor.scorer.parameters()),
        lr=float(args.lr),
    )
    updater = DecisionBanditPPOUpdater(
        encoder=encoder,
        scorer=actor.scorer,
        optimizer=optimizer,
        config=DecisionBanditUpdateConfig(
            clip_epsilon=float(args.clip_ratio),
            ppo_epochs=int(args.ppo_epochs),
            max_grad_norm=float(args.max_grad_norm),
            chunk_decisions=int(args.chunk_decisions),
            entropy_coef=0.0,
        ),
        device=device,
    )
    initialization_identity = {
        "encoder_hash": _state_hash(encoder.state_dict()),
        "scorer_hash": _state_hash(actor.scorer.state_dict()),
        "parameter_count": int(
            sum(parameter.numel() for parameter in encoder.parameters())
            + sum(parameter.numel() for parameter in actor.scorer.parameters())
        ),
    }
    run_dir = _create_run_dir(args)
    checkpoint_updates = _parse_updates(str(args.checkpoint_updates))
    config_payload = _config_payload(args, initialization_identity)
    _write_json(run_dir / "config.json", config_payload)
    _save_checkpoint(
        run_dir / "checkpoints" / "checkpoint_update_0000.pt",
        torch=torch,
        encoder=encoder,
        actor=actor,
        optimizer=optimizer,
        args=args,
        completed_update=0,
        global_slot=0,
        initialization_identity=initialization_identity,
    )

    global_slot = 0
    episode_id = 0
    trajectory_id = f"{args.group}_seed{int(args.seed)}"
    update_rows: list[dict[str, Any]] = []
    current_prepared = prepared
    for update_index in range(1, int(args.updates) + 1):
        rollout = DecisionBanditRolloutBuffer()
        reward_total = 0.0
        latest_info: dict[str, Any] = {}
        for _ in range(int(args.slots_per_update)):
            snapshot = current_prepared.graph_snapshot
            task_features = torch.as_tensor(
                np.array(snapshot.task_features, dtype=np.float32, copy=True),
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                task_embeddings = encoder(task_features)
            env.apply_movement({})
            ready_tasks = [
                env.task_manager.get_task(task_id)
                for task_id in current_prepared.frozen_ready_task_ids
            ]
            ready_tasks = [task for task in ready_tasks if task is not None]
            assignment_buffer = actor.act(
                frozen_ready_tasks=ready_tasks,
                task_embeddings=task_embeddings,
                graph_snapshot=snapshot,
                task_manager=env.task_manager,
                uavs=env.uavs,
                executor=env.executor,
                current_time_seconds=env.current_time_seconds,
                uav_service_positions=env.uav_service_positions,
                ue_service_positions=env.ue_service_positions,
                ues=env.ues,
                deterministic=False,
            )
            for action_record in actor.latest_records:
                record = build_decision_bandit_record(
                    action_record=action_record,
                    graph_snapshot=snapshot,
                    environment_id=0,
                    trajectory_id=trajectory_id,
                    episode_id=episode_id,
                    physical_slot=global_slot,
                )
                rollout.append_record(record)
            for event in actor.latest_skip_events:
                rollout.append_skip(
                    DecisionBanditSkipEvent(
                        environment_id=0,
                        trajectory_id=trajectory_id,
                        episode_id=episode_id,
                        physical_slot=global_slot,
                        task_id=str(event.task_id),
                        dag_id=str(event.dag_id),
                        decision_order=int(event.decision_order),
                        valid_candidate_count=0,
                        skip_reason=str(event.skip_reason),
                    )
                )
            _, _, done, latest_info = env.commit_and_advance(
                assignment_buffer=assignment_buffer,
                offloading_skip_count=len(actor.latest_skip_events),
            )
            reward_total += float(latest_info["step_reward"])
            rollout.physical_slot_count += 1
            global_slot += 1
            if done:
                episode_id += 1
                env.reset()
                graph_builder.reset()
            current_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)

        if str(args.group) == "S1-B":
            update_stats = updater.update(rollout)
        else:
            update_stats = _control_stats(rollout, int(args.ppo_epochs))
        row = {
            "group": str(args.group),
            "seed": int(args.seed),
            "completed_update": int(update_index),
            "global_slot": int(global_slot),
            "physical_slot_count": int(rollout.physical_slot_count),
            "choice_decision_count": int(rollout.choice_decision_count),
            "forced_decision_count": int(rollout.forced_decision_count),
            "skipped_no_candidate": int(rollout.skipped_no_candidate),
            "environment_reward_total": float(reward_total),
            "behavior": _behavior_diagnostics(rollout.records),
            "update": asdict(update_stats),
            "arrival_funnel": _arrival_subset(latest_info),
            "finite": True,
        }
        _append_jsonl(run_dir / "updates.jsonl", row)
        update_rows.append(row)
        if update_index in checkpoint_updates:
            _save_checkpoint(
                run_dir / "checkpoints" / f"checkpoint_update_{update_index:04d}.pt",
                torch=torch,
                encoder=encoder,
                actor=actor,
                optimizer=optimizer,
                args=args,
                completed_update=update_index,
                global_slot=global_slot,
                initialization_identity=initialization_identity,
            )

    summary = {
        "schema": "decision_ppo_bandit_stage1_summary_v1",
        "technical_pass": bool(all(row["finite"] for row in update_rows)),
        "group": str(args.group),
        "seed": int(args.seed),
        "completed_updates": len(update_rows),
        "global_slot": int(global_slot),
        "initialization_identity": initialization_identity,
        "run_dir": str(run_dir),
        "last_update": update_rows[-1] if update_rows else None,
        "active_dag_cap_pairing_limitation": (
            "same evaluator seeds are not strict counterfactual pairs because "
            "active_dag_cap makes eligibility and RNG consumption policy-dependent"
        ),
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def _control_stats(buffer: DecisionBanditRolloutBuffer, epochs: int) -> Any:
    from marl_models.mappo.clean_decision_ppo_bandit import DecisionBanditUpdateStats

    empty = not bool(buffer.records)
    rows = []
    for epoch_index in range(int(epochs)):
        rows.append(
            {
                "epoch_index": epoch_index,
                "actor_loss": None,
                "entropy": None,
                "approx_kl": None,
                "clip_fraction": None,
                "ratio_mean": None,
                "ratio_std": None,
                "ratio_min": None,
                "ratio_max": None,
                "control_no_update": True,
            }
        )
    return DecisionBanditUpdateStats(
        empty_actor_batch=empty,
        effective_decision_count=len(buffer.records),
        optimizer_step_count=0,
        epochs=rows,
    )


def _behavior_diagnostics(records: list[Any]) -> dict[str, Any]:
    if not records:
        return {
            "sample_count": 0,
            "raw_eft_regret_mean": None,
            "raw_eft_regret_median": None,
            "raw_eft_regret_p95": None,
            "greedy_agreement": None,
            "margin5_accuracy": None,
            "margin20_accuracy": None,
            "normalized_entropy": None,
            "max_action_probability": None,
            "top1_top2_probability_margin": None,
        }
    regrets = np.asarray([row.raw_eft_regret for row in records], dtype=np.float64)
    agreements: list[float] = []
    margin5: list[float] = []
    margin20: list[float] = []
    normalized_entropy: list[float] = []
    max_probabilities: list[float] = []
    probability_margins: list[float] = []
    for row in records:
        mask = np.asarray(row.candidate_mask, dtype=bool)
        eft = np.asarray(row.candidate_estimated_finish_times, dtype=np.float64)
        probabilities = np.asarray(row.old_masked_probabilities, dtype=np.float64)
        legal = np.flatnonzero(mask)
        order = legal[np.argsort(eft[legal])]
        agreement = float(int(row.executed_action) == int(order[0]))
        margin = float(eft[order[1]] - eft[order[0]])
        agreements.append(agreement)
        if margin >= 5.0:
            margin5.append(agreement)
        if margin >= 20.0:
            margin20.append(agreement)
        legal_prob = probabilities[legal]
        entropy = -float(np.sum(legal_prob * np.log(np.clip(legal_prob, 1e-12, 1.0))))
        normalized_entropy.append(entropy / max(math.log(len(legal)), 1e-12))
        sorted_prob = np.sort(legal_prob)
        max_probabilities.append(float(sorted_prob[-1]))
        probability_margins.append(float(sorted_prob[-1] - sorted_prob[-2]))
    return {
        "sample_count": len(records),
        "raw_eft_regret_mean": float(regrets.mean()),
        "raw_eft_regret_median": float(np.median(regrets)),
        "raw_eft_regret_p95": float(np.percentile(regrets, 95)),
        "greedy_agreement": float(np.mean(agreements)),
        "margin5_sample_count": len(margin5),
        "margin5_accuracy": None if not margin5 else float(np.mean(margin5)),
        "margin20_sample_count": len(margin20),
        "margin20_accuracy": None if not margin20 else float(np.mean(margin20)),
        "normalized_entropy": float(np.mean(normalized_entropy)),
        "max_action_probability": float(np.mean(max_probabilities)),
        "top1_top2_probability_margin": float(np.mean(probability_margins)),
    }


def _arrival_subset(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in info.items()
        if str(key).startswith("arrival_")
    }


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.updates) <= 0 or int(args.slots_per_update) <= 0:
        raise ValueError("updates and slots-per-update must be positive")
    if int(args.ppo_epochs) != 3:
        raise ValueError("Stage 1 requires exactly three PPO epochs")
    if float(args.lr) <= 0.0 or float(args.max_grad_norm) <= 0.0:
        raise ValueError("learning rate and max grad norm must be positive")
    if int(args.chunk_decisions) <= 0:
        raise ValueError("chunk-decisions must be positive")
    if not math.isclose(
        DECISION_BANDIT_REGRET_SCALE, 61.75621424202263, rel_tol=0.0, abs_tol=0.0
    ):
        raise AssertionError("Stage 1 regret scale changed")


def _parse_updates(text: str) -> set[int]:
    values = {int(value.strip()) for value in text.split(",") if value.strip()}
    if any(value < 0 for value in values):
        raise ValueError("checkpoint updates must be non-negative")
    return values


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _state_hash(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _create_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(args.output_dir) / f"{timestamp}_{args.run_name}_{args.group}_seed{int(args.seed)}"
    (path / "checkpoints").mkdir(parents=True, exist_ok=False)
    return path


def _config_payload(args: argparse.Namespace, identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "decision_ppo_bandit_stage1_config_v1",
        "group": str(args.group),
        "seed": int(args.seed),
        "updates": int(args.updates),
        "slots_per_update": int(args.slots_per_update),
        "ppo_epochs": int(args.ppo_epochs),
        "lr": float(args.lr),
        "clip_ratio": float(args.clip_ratio),
        "max_grad_norm": float(args.max_grad_norm),
        "chunk_decisions": int(args.chunk_decisions),
        "task_embedding_dim": int(args.task_embedding_dim),
        "hidden_dim": int(args.hidden_dim),
        "regret_scale": DECISION_BANDIT_REGRET_SCALE,
        "entropy_coef": 0.0,
        "num_envs": 1,
        "movement": "forced_hover",
        "encoder": "mlp",
        "initialization_identity": identity,
    }


def _save_checkpoint(
    path: Path,
    *,
    torch: Any,
    encoder: Any,
    actor: Any,
    optimizer: Any,
    args: argparse.Namespace,
    completed_update: int,
    global_slot: int,
    initialization_identity: dict[str, Any],
) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "group": str(args.group),
        "seed": int(args.seed),
        "completed_update": int(completed_update),
        "global_slot": int(global_slot),
        "encoder_state_dict": encoder.state_dict(),
        "scorer_state_dict": actor.scorer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _config_payload(args, initialization_identity),
        "initialization_identity": initialization_identity,
        "resume_semantics": "no exact resume; rerun an interrupted cell from its start",
    }
    torch.save(payload, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
