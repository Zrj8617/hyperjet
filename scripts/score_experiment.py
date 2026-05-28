from __future__ import annotations

import argparse
import dataclasses
import json
from contextlib import contextmanager
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
from environment.dag_tasks import TASK_STATE_DROPPED, TASK_STATE_FINISHED
from marl_models.hgnn.pretrain import save_pretrained_scheduler, train_score_imitation
from marl_models.hgnn.supervision import collect_score_supervision_dataset
from utils.progress import TerminalProgress


@dataclasses.dataclass(slots=True)
class ExperimentConfigState:
    use_hgnn_score_assignment: bool
    hgnn_score_checkpoint: str
    score_fallback_to_heuristic: bool
    use_phase_one_hyperedges: bool
    use_collaborative_hyperedges: bool
    use_service_domain_hyperedges: bool
    use_resource_competition_hyperedges: bool
    use_critical_hyperedges: bool
    use_critical_support_hyperedges: bool
    use_attribute_hyperedges: bool
    use_compute_attribute_hyperedges: bool
    use_communication_attribute_hyperedges: bool
    use_candidate_scarce_attribute_hyperedges: bool
    use_task_type_attribute_hyperedges: bool
    use_task_uav_pair_features: bool
    task_uav_pair_feature_mode: str
    use_pair_hyperedge_score_features: bool


@contextmanager
def temporary_score_config(
    *,
    use_score: bool,
    checkpoint_path: str = "",
    fallback_to_heuristic: bool = True,
):
    old_state = ExperimentConfigState(
        use_hgnn_score_assignment=config.USE_HGNN_SCORE_ASSIGNMENT,
        hgnn_score_checkpoint=config.HGNN_SCORE_CHECKPOINT,
        score_fallback_to_heuristic=config.SCORE_FALLBACK_TO_HEURISTIC,
        use_phase_one_hyperedges=config.USE_PHASE_ONE_HYPEREDGES,
        use_collaborative_hyperedges=config.USE_COLLABORATIVE_HYPEREDGES,
        use_service_domain_hyperedges=config.USE_SERVICE_DOMAIN_HYPEREDGES,
        use_resource_competition_hyperedges=config.USE_RESOURCE_COMPETITION_HYPEREDGES,
        use_critical_hyperedges=config.USE_CRITICAL_HYPEREDGES,
        use_critical_support_hyperedges=config.USE_CRITICAL_SUPPORT_HYPEREDGES,
        use_attribute_hyperedges=config.USE_ATTRIBUTE_HYPEREDGES,
        use_compute_attribute_hyperedges=config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES,
        use_communication_attribute_hyperedges=config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES,
        use_candidate_scarce_attribute_hyperedges=config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES,
        use_task_type_attribute_hyperedges=config.USE_TASK_TYPE_ATTRIBUTE_HYPEREDGES,
        use_task_uav_pair_features=config.USE_TASK_UAV_PAIR_FEATURES,
        task_uav_pair_feature_mode=config.TASK_UAV_PAIR_FEATURE_MODE,
        use_pair_hyperedge_score_features=config.USE_PAIR_HYPEREDGE_SCORE_FEATURES,
    )
    config.USE_HGNN_SCORE_ASSIGNMENT = use_score
    config.HGNN_SCORE_CHECKPOINT = checkpoint_path
    config.SCORE_FALLBACK_TO_HEURISTIC = fallback_to_heuristic
    try:
        yield
    finally:
        config.USE_HGNN_SCORE_ASSIGNMENT = old_state.use_hgnn_score_assignment
        config.HGNN_SCORE_CHECKPOINT = old_state.hgnn_score_checkpoint
        config.SCORE_FALLBACK_TO_HEURISTIC = old_state.score_fallback_to_heuristic
        config.USE_PHASE_ONE_HYPEREDGES = old_state.use_phase_one_hyperedges
        config.USE_COLLABORATIVE_HYPEREDGES = old_state.use_collaborative_hyperedges
        config.USE_SERVICE_DOMAIN_HYPEREDGES = old_state.use_service_domain_hyperedges
        config.USE_RESOURCE_COMPETITION_HYPEREDGES = old_state.use_resource_competition_hyperedges
        config.USE_CRITICAL_HYPEREDGES = old_state.use_critical_hyperedges
        config.USE_CRITICAL_SUPPORT_HYPEREDGES = old_state.use_critical_support_hyperedges
        config.USE_ATTRIBUTE_HYPEREDGES = old_state.use_attribute_hyperedges
        config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES = old_state.use_compute_attribute_hyperedges
        config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES = old_state.use_communication_attribute_hyperedges
        config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES = old_state.use_candidate_scarce_attribute_hyperedges
        config.USE_TASK_TYPE_ATTRIBUTE_HYPEREDGES = old_state.use_task_type_attribute_hyperedges
        config.USE_TASK_UAV_PAIR_FEATURES = old_state.use_task_uav_pair_features
        config.TASK_UAV_PAIR_FEATURE_MODE = old_state.task_uav_pair_feature_mode
        config.USE_PAIR_HYPEREDGE_SCORE_FEATURES = old_state.use_pair_hyperedge_score_features


def _zero_actions() -> np.ndarray:
    return np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)


def _update_diag(accumulator: dict[str, float], diagnostics: dict[str, float]) -> None:
    for key, value in diagnostics.items():
        accumulator[key] = accumulator.get(key, 0.0) + float(value)


def _finalize_diag(accumulator: dict[str, float], steps: int) -> dict[str, float]:
    if not accumulator or steps <= 0:
        return {}
    averaged = {"ready_tasks", "active_tasks", "feasible_edges", "score_edge_count"}
    result: dict[str, float] = {}
    for key, value in accumulator.items():
        if key in averaged:
            result[f"avg_{key}"] = value / float(steps)
        else:
            result[key] = value
    return result


def _task_type_name(task_type: int) -> str:
    if task_type == config.TASK_TYPE_PREPROCESS:
        return "preprocess"
    if task_type == config.TASK_TYPE_COMPUTE:
        return "compute"
    if task_type == config.TASK_TYPE_AGGREGATION:
        return "aggregation"
    return f"unknown_{task_type}"


def _add_type_metric(accumulator: dict[str, float], task_type: int, suffix: str, value: float = 1.0) -> None:
    key = f"type_{_task_type_name(task_type)}_{suffix}"
    accumulator[key] = accumulator.get(key, 0.0) + float(value)


def _update_type_step_diag(accumulator: dict[str, float], env: Env) -> None:
    for task in env.task_manager.tasks.values():
        if task.state not in {TASK_STATE_FINISHED, TASK_STATE_DROPPED}:
            _add_type_metric(accumulator, task.task_type, "active_tasks", 1.0)
            if task.is_ready:
                _add_type_metric(accumulator, task.task_type, "ready_tasks", 1.0)

    for record in env.task_executor.latest_assignment_records:
        _add_type_metric(accumulator, record.task_type, "assignment_attempts", 1.0)
        _add_type_metric(accumulator, record.task_type, "candidate_edges", len(record.candidates))
        _add_type_metric(accumulator, record.task_type, "slack_sum", record.task_slack)
        if record.selected_uav is not None:
            _add_type_metric(accumulator, record.task_type, "assigned", 1.0)
        if record.selection_mode == "score":
            _add_type_metric(accumulator, record.task_type, "score_selected", 1.0)
        elif record.selection_mode == "fallback":
            _add_type_metric(accumulator, record.task_type, "fallback_selected", 1.0)
        if record.disagrees_with_heuristic:
            _add_type_metric(accumulator, record.task_type, "score_heuristic_disagreements", 1.0)

        selected_candidate = next((candidate for candidate in record.candidates if candidate.uav_id == record.selected_uav), None)
        heuristic_candidate = next((candidate for candidate in record.candidates if candidate.uav_id == record.heuristic_uav), None)
        if selected_candidate is not None:
            selected_margin = record.task_deadline - selected_candidate.planned_finish
            _add_type_metric(accumulator, record.task_type, "selected_deadline_margin_sum", selected_margin)
            _add_type_metric(accumulator, record.task_type, "selected_planned_finish_sum", selected_candidate.planned_finish)
        if selected_candidate is not None and heuristic_candidate is not None:
            _add_type_metric(
                accumulator,
                record.task_type,
                "selected_minus_heuristic_finish_sum",
                selected_candidate.planned_finish - heuristic_candidate.planned_finish,
            )


def _finalize_type_diag(accumulator: dict[str, float], steps: int, env: Env) -> dict[str, float]:
    result: dict[str, float] = {}
    type_names = [
        _task_type_name(config.TASK_TYPE_PREPROCESS),
        _task_type_name(config.TASK_TYPE_COMPUTE),
        _task_type_name(config.TASK_TYPE_AGGREGATION),
    ]

    for task in env.task_manager.tasks.values():
        _add_type_metric(accumulator, task.task_type, "generated_tasks", 1.0)
        if task.state == TASK_STATE_FINISHED:
            _add_type_metric(accumulator, task.task_type, "finished_tasks", 1.0)
            if task.finish_time is not None and task.finish_time <= task.deadline:
                _add_type_metric(accumulator, task.task_type, "on_time_finished_tasks", 1.0)
            else:
                _add_type_metric(accumulator, task.task_type, "deadline_violations", 1.0)
        elif task.state == TASK_STATE_DROPPED:
            _add_type_metric(accumulator, task.task_type, "dropped_tasks", 1.0)

    for type_name in type_names:
        prefix = f"type_{type_name}_"
        attempts = accumulator.get(prefix + "assignment_attempts", 0.0)
        assigned = accumulator.get(prefix + "assigned", 0.0)
        generated = accumulator.get(prefix + "generated_tasks", 0.0)
        finished = accumulator.get(prefix + "finished_tasks", 0.0)
        selected = accumulator.get(prefix + "score_selected", 0.0)

        result[prefix + "avg_active_tasks"] = accumulator.get(prefix + "active_tasks", 0.0) / max(float(steps), 1.0)
        result[prefix + "avg_ready_tasks"] = accumulator.get(prefix + "ready_tasks", 0.0) / max(float(steps), 1.0)
        result[prefix + "assignment_attempts"] = attempts
        result[prefix + "assigned"] = assigned
        result[prefix + "assignment_success_rate"] = assigned / max(attempts, 1.0)
        result[prefix + "avg_candidate_edges"] = accumulator.get(prefix + "candidate_edges", 0.0) / max(attempts, 1.0)
        result[prefix + "avg_assignment_slack"] = accumulator.get(prefix + "slack_sum", 0.0) / max(attempts, 1.0)
        result[prefix + "score_selected"] = selected
        result[prefix + "fallback_selected"] = accumulator.get(prefix + "fallback_selected", 0.0)
        result[prefix + "score_heuristic_disagreements"] = accumulator.get(prefix + "score_heuristic_disagreements", 0.0)
        result[prefix + "disagreement_rate"] = accumulator.get(prefix + "score_heuristic_disagreements", 0.0) / max(selected, 1.0)
        result[prefix + "avg_selected_deadline_margin"] = accumulator.get(prefix + "selected_deadline_margin_sum", 0.0) / max(assigned, 1.0)
        result[prefix + "avg_selected_planned_finish"] = accumulator.get(prefix + "selected_planned_finish_sum", 0.0) / max(assigned, 1.0)
        result[prefix + "avg_selected_minus_heuristic_finish"] = (
            accumulator.get(prefix + "selected_minus_heuristic_finish_sum", 0.0) / max(selected, 1.0)
        )
        result[prefix + "generated_tasks"] = generated
        result[prefix + "finished_tasks"] = finished
        result[prefix + "on_time_finished_tasks"] = accumulator.get(prefix + "on_time_finished_tasks", 0.0)
        result[prefix + "deadline_violations"] = accumulator.get(prefix + "deadline_violations", 0.0)
        result[prefix + "dropped_tasks"] = accumulator.get(prefix + "dropped_tasks", 0.0)
        result[prefix + "finish_rate"] = finished / max(generated, 1.0)
        result[prefix + "on_time_rate"] = accumulator.get(prefix + "on_time_finished_tasks", 0.0) / max(finished, 1.0)

    return result


def _episode_seed(seed: int, episode_idx: int) -> int:
    return int(seed + episode_idx)


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


def _override_num_ues(num_ues: int) -> None:
    if num_ues <= 0:
        raise ValueError("--num_ues must be greater than 0.")
    config.NUM_UES = num_ues
    _refresh_dimension_config()


def _apply_ablation_config(ablation: str) -> None:
    mainline_modes = {
        "attribute_blind",
        "critical_only",
        "attribute_only",
        "critical_plus_attribute",
    }
    if ablation in mainline_modes:
        config.USE_PHASE_ONE_HYPEREDGES = ablation != "attribute_blind"
        config.USE_COLLABORATIVE_HYPEREDGES = False
        config.USE_SERVICE_DOMAIN_HYPEREDGES = False
        config.USE_RESOURCE_COMPETITION_HYPEREDGES = False
        config.USE_CRITICAL_SUPPORT_HYPEREDGES = False
        config.USE_CRITICAL_HYPEREDGES = ablation in {"critical_only", "critical_plus_attribute"}
        use_attribute = ablation in {
            "attribute_only",
            "critical_plus_attribute",
        }
        config.USE_ATTRIBUTE_HYPEREDGES = use_attribute
        config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES = use_attribute
        config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES = use_attribute
        config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES = use_attribute
        config.USE_TASK_TYPE_ATTRIBUTE_HYPEREDGES = False
        config.USE_TASK_UAV_PAIR_FEATURES = ablation != "no_pair_feature"
        config.USE_PAIR_HYPEREDGE_SCORE_FEATURES = False
        return

    no_attribute_modes = {
        "no_attribute",
        "no_attribute_no_service_domain",
        "no_attribute_no_resource_competition",
        "no_attribute_no_collaborative",
        "safe_hyperedge_only",
    }
    no_service_domain_modes = {"no_service_domain", "no_attribute_no_service_domain", "no_attribute_no_collaborative"}
    no_resource_competition_modes = {
        "no_resource_competition",
        "no_attribute_no_resource_competition",
        "no_attribute_no_collaborative",
        "safe_hyperedge_only",
    }
    config.USE_PHASE_ONE_HYPEREDGES = ablation != "no_hyperedge"
    config.USE_COLLABORATIVE_HYPEREDGES = ablation not in {"no_hyperedge", "no_collaborative"}
    config.USE_SERVICE_DOMAIN_HYPEREDGES = (
        ablation not in {"no_hyperedge", "no_collaborative"} and ablation not in no_service_domain_modes
    )
    config.USE_RESOURCE_COMPETITION_HYPEREDGES = (
        ablation not in {"no_hyperedge", "no_collaborative"} and ablation not in no_resource_competition_modes
    )
    config.USE_CRITICAL_HYPEREDGES = ablation not in {"no_hyperedge", "no_critical"}
    config.USE_CRITICAL_SUPPORT_HYPEREDGES = ablation not in {"no_hyperedge", "no_critical", "safe_hyperedge_only"}
    config.USE_ATTRIBUTE_HYPEREDGES = ablation not in {"no_hyperedge"} and ablation not in no_attribute_modes
    config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES = config.USE_ATTRIBUTE_HYPEREDGES
    config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES = config.USE_ATTRIBUTE_HYPEREDGES
    config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES = config.USE_ATTRIBUTE_HYPEREDGES
    config.USE_TASK_TYPE_ATTRIBUTE_HYPEREDGES = False
    if ablation in {"no_compute_attribute", "no_communication_attribute", "no_candidate_scarce_attribute"}:
        config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES = ablation != "no_compute_attribute"
        config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES = ablation != "no_communication_attribute"
        config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES = ablation != "no_candidate_scarce_attribute"
    elif ablation in {"only_compute_attribute", "only_communication_attribute", "only_candidate_scarce_attribute"}:
        config.USE_ATTRIBUTE_HYPEREDGES = True
        config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES = ablation == "only_compute_attribute"
        config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES = ablation == "only_communication_attribute"
        config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES = ablation == "only_candidate_scarce_attribute"
    config.USE_TASK_UAV_PAIR_FEATURES = ablation != "no_pair_feature"
    if not config.USE_TASK_UAV_PAIR_FEATURES:
        config.TASK_UAV_PAIR_FEATURE_MODE = "none"
    config.USE_PAIR_HYPEREDGE_SCORE_FEATURES = ablation not in {
        "no_pair_hyperedge_score_feature",
        "safe_hyperedge_only",
    }


def evaluate_scheduler(
    *,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    use_score: bool,
    checkpoint_path: str = "",
    fallback_to_heuristic: bool = True,
) -> dict[str, float]:
    episode_metrics: list[dict[str, float]] = []
    total_steps = num_episodes * steps_per_episode
    eval_title = "eval:score" if use_score else "eval:heuristic"
    progress = TerminalProgress(total_steps, eval_title)
    with temporary_score_config(
        use_score=use_score,
        checkpoint_path=checkpoint_path,
        fallback_to_heuristic=fallback_to_heuristic,
    ):
        for episode_idx in range(num_episodes):
            episode_seed = _episode_seed(seed, episode_idx)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)

            env = Env()
            env.reset()

            reward_total = 0.0
            latency_total = 0.0
            energy_total = 0.0
            on_time_ratio_last = 0.0
            offline_like_total = 0.0
            diag_acc: dict[str, float] = {}
            type_diag_acc: dict[str, float] = {}

            for step_idx in range(steps_per_episode):
                _, rewards, (avg_delay, total_energy, on_time_ratio, offline_like_rate) = env.step(_zero_actions())
                reward_total += float(np.sum(rewards))
                latency_total += float(avg_delay)
                energy_total += float(total_energy)
                on_time_ratio_last = float(on_time_ratio)
                offline_like_total += float(offline_like_rate)
                _update_diag(diag_acc, env.latest_phase_one_diagnostics)
                _update_type_step_diag(type_diag_acc, env)
                progress.update(
                    postfix=(
                        f"episode {episode_idx + 1}/{num_episodes} "
                        f"reward {reward_total:.1f} finished {env.task_executor.get_summary()['finished_count']:.0f}"
                    )
                )

            summary = env.task_executor.get_summary()
            episode_result = {
                "reward_total": reward_total,
                "latency_total": latency_total,
                "energy_total": energy_total,
                "offline_like_total": offline_like_total,
                "finished_count": float(summary["finished_count"]),
                "on_time_ratio": float(summary["on_time_ratio"]),
                "deadline_violations": float(summary["deadline_violations"]),
            }
            episode_result.update(env.task_manager.get_job_summary())
            episode_result.update(_finalize_diag(diag_acc, steps_per_episode))
            episode_result.update(_finalize_type_diag(type_diag_acc, steps_per_episode, env))
            episode_result["last_on_time_ratio"] = on_time_ratio_last
            episode_metrics.append(episode_result)

    progress.finish(postfix=f"episodes {num_episodes} complete")
    keys = sorted({key for metrics in episode_metrics for key in metrics})
    averaged_result: dict[str, float] = {}
    for key in keys:
        averaged_result[key] = float(np.mean([metrics.get(key, 0.0) for metrics in episode_metrics]))
    return averaged_result


def pretrain_mode(
    *,
    mode: str,
    output_dir: Path,
    episodes: int,
    steps_per_episode: int,
    epochs: int,
    learning_rate: float,
    device: str,
    seed: int,
    action_mode: str,
) -> tuple[str, list[dict[str, float]]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"[score-exp] Collecting supervision for mode={mode}, action_mode={action_mode}, seed={seed} on {device} ...")
    samples = collect_score_supervision_dataset(episodes, steps_per_episode, action_mode=action_mode, seed=seed)
    print(f"[score-exp] Collected {len(samples)} graph samples for mode={mode}.")
    scheduler, metrics = train_score_imitation(samples, epochs, learning_rate, device, mode=mode)
    run_dir = output_dir / f"{mode}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_path = save_pretrained_scheduler(scheduler, str(run_dir))
    metrics_payload = [dataclasses.asdict(metric) for metric in metrics]
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    return model_path, metrics_payload


def _delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(candidate) | set(baseline))
    return {key: float(candidate.get(key, 0.0) - baseline.get(key, 0.0)) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="score_experiments")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--num_ues", type=int, default=None)
    parser.add_argument("--num_uavs", type=int, default=None)
    parser.add_argument("--dag_arrival_prob", type=float, default=None)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--pretrain_episodes", type=int, default=8)
    parser.add_argument("--pretrain_steps", type=int, default=80)
    parser.add_argument("--pretrain_epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=config.SCORE_PRETRAIN_LR)
    parser.add_argument("--action_mode", type=str, default=config.SCORE_PRETRAIN_ACTION_MODE, choices=["zero", "random"])
    parser.add_argument("--finish_tolerance", type=float, default=config.SCORE_BOUNDED_RANKING_FINISH_TOLERANCE)
    parser.add_argument("--eval_episodes", type=int, default=6)
    parser.add_argument("--eval_steps", type=int, default=120)
    parser.add_argument("--skip_top1", action="store_true")
    parser.add_argument("--skip_ranking", action="store_true")
    parser.add_argument("--run_bounded_ranking", action="store_true")
    parser.add_argument("--skip_soft", action="store_true")
    parser.add_argument("--pair_feature_mode", choices=["full", "limited", "none"], default="full")
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=[
            "full",
            "no_hyperedge",
            "no_pair_feature",
            "no_collaborative",
            "no_service_domain",
            "no_resource_competition",
            "no_critical",
            "no_pair_hyperedge_score_feature",
            "safe_hyperedge_only",
            "attribute_blind",
            "critical_only",
            "attribute_only",
            "critical_plus_attribute",
            "no_attribute",
            "no_attribute_no_service_domain",
            "no_attribute_no_resource_competition",
            "no_attribute_no_collaborative",
            "no_compute_attribute",
            "no_communication_attribute",
            "no_candidate_scarce_attribute",
            "only_compute_attribute",
            "only_communication_attribute",
            "only_candidate_scarce_attribute",
        ],
        help=(
            "Ablation mode. full keeps all graph inputs; no_hyperedge disables all hyperedges; "
            "no_pair_feature zeros pair features; no_collaborative disables both split collaborative subtypes; "
            "no_service_domain/no_resource_competition/no_critical/no_attribute "
            "disable one hyperedge type while keeping pair features. no_attribute_no_service_domain, "
            "no_attribute_no_resource_competition, and no_attribute_no_collaborative support paired "
            "service/resource ablations under the no-attribute baseline. Attribute subtype modes can "
            "disable or keep only compute/communication/candidate-scarce attribute hyperedges."
        ),
    )
    args = parser.parse_args()
    if args.num_ues is not None:
        _override_num_ues(args.num_ues)
    if args.num_uavs is not None:
        _override_num_uavs(args.num_uavs, args.seed)
    if args.dag_arrival_prob is not None:
        if not 0.0 <= args.dag_arrival_prob <= 1.0:
            raise ValueError("--dag_arrival_prob must be in [0, 1].")
        config.DAG_ARRIVAL_PROB = args.dag_arrival_prob
    config.TASK_UAV_PAIR_FEATURE_MODE = args.pair_feature_mode
    config.USE_TASK_UAV_PAIR_FEATURES = args.pair_feature_mode != "none"
    config.SCORE_BOUNDED_RANKING_FINISH_TOLERANCE = float(args.finish_tolerance)
    _apply_ablation_config(args.ablation)
    if args.ablation == "no_pair_feature" or args.pair_feature_mode == "none":
        config.TASK_UAV_PAIR_FEATURE_MODE = "none"
        config.USE_TASK_UAV_PAIR_FEATURES = False
    else:
        config.TASK_UAV_PAIR_FEATURE_MODE = args.pair_feature_mode
        config.USE_TASK_UAV_PAIR_FEATURES = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "seed": args.seed,
        "device": args.device,
        "num_ues": config.NUM_UES,
        "num_uavs": config.NUM_UAVS,
        "dag_arrival_prob": config.DAG_ARRIVAL_PROB,
        "ablation": args.ablation,
        "pair_feature_mode": config.TASK_UAV_PAIR_FEATURE_MODE,
        "feature_switches": {
            "use_phase_one_hyperedges": config.USE_PHASE_ONE_HYPEREDGES,
            "use_collaborative_hyperedges": config.USE_COLLABORATIVE_HYPEREDGES,
            "use_service_domain_hyperedges": config.USE_SERVICE_DOMAIN_HYPEREDGES,
            "use_resource_competition_hyperedges": config.USE_RESOURCE_COMPETITION_HYPEREDGES,
            "use_critical_hyperedges": config.USE_CRITICAL_HYPEREDGES,
            "use_critical_support_hyperedges": config.USE_CRITICAL_SUPPORT_HYPEREDGES,
            "use_attribute_hyperedges": config.USE_ATTRIBUTE_HYPEREDGES,
            "use_compute_attribute_hyperedges": config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES,
            "use_communication_attribute_hyperedges": config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES,
            "use_candidate_scarce_attribute_hyperedges": config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES,
            "use_task_type_attribute_hyperedges": config.USE_TASK_TYPE_ATTRIBUTE_HYPEREDGES,
            "use_task_uav_pair_features": config.USE_TASK_UAV_PAIR_FEATURES,
            "task_uav_pair_feature_mode": config.TASK_UAV_PAIR_FEATURE_MODE,
            "use_pair_hyperedge_score_features": config.USE_PAIR_HYPEREDGE_SCORE_FEATURES,
        },
        "pretrain": {
            "episodes": args.pretrain_episodes,
            "steps_per_episode": args.pretrain_steps,
            "epochs": args.pretrain_epochs,
            "learning_rate": args.lr,
            "action_mode": args.action_mode,
            "bounded_ranking_finish_tolerance": config.SCORE_BOUNDED_RANKING_FINISH_TOLERANCE,
        },
        "evaluation": {
            "episodes": args.eval_episodes,
            "steps_per_episode": args.eval_steps,
        },
        "runs": {},
    }

    print("[score-exp] Evaluating heuristic baseline (score off) ...")
    off_metrics = evaluate_scheduler(
        num_episodes=args.eval_episodes,
        steps_per_episode=args.eval_steps,
        seed=args.seed,
        use_score=False,
    )
    payload["runs"]["off"] = {"metrics": off_metrics}

    if not args.skip_top1:
        top1_ckpt, top1_pretrain_metrics = pretrain_mode(
            mode="top1",
            output_dir=output_dir,
            episodes=args.pretrain_episodes,
            steps_per_episode=args.pretrain_steps,
            epochs=args.pretrain_epochs,
            learning_rate=args.lr,
            device=args.device,
            seed=args.seed,
            action_mode=args.action_mode,
        )
        top1_eval = evaluate_scheduler(
            num_episodes=args.eval_episodes,
            steps_per_episode=args.eval_steps,
            seed=args.seed,
            use_score=True,
            checkpoint_path=top1_ckpt,
            fallback_to_heuristic=True,
        )
        payload["runs"]["top1_on"] = {
            "checkpoint": top1_ckpt,
            "pretrain_metrics": top1_pretrain_metrics,
            "metrics": top1_eval,
            "delta_vs_off": _delta(top1_eval, off_metrics),
        }

    if not args.skip_ranking:
        ranking_ckpt, ranking_pretrain_metrics = pretrain_mode(
            mode="ranking",
            output_dir=output_dir,
            episodes=args.pretrain_episodes,
            steps_per_episode=args.pretrain_steps,
            epochs=args.pretrain_epochs,
            learning_rate=args.lr,
            device=args.device,
            seed=args.seed,
            action_mode=args.action_mode,
        )
        ranking_eval = evaluate_scheduler(
            num_episodes=args.eval_episodes,
            steps_per_episode=args.eval_steps,
            seed=args.seed,
            use_score=True,
            checkpoint_path=ranking_ckpt,
            fallback_to_heuristic=True,
        )
        payload["runs"]["ranking_on"] = {
            "checkpoint": ranking_ckpt,
            "pretrain_metrics": ranking_pretrain_metrics,
            "metrics": ranking_eval,
            "delta_vs_off": _delta(ranking_eval, off_metrics),
        }

    if args.run_bounded_ranking:
        bounded_ckpt, bounded_pretrain_metrics = pretrain_mode(
            mode="bounded_ranking",
            output_dir=output_dir,
            episodes=args.pretrain_episodes,
            steps_per_episode=args.pretrain_steps,
            epochs=args.pretrain_epochs,
            learning_rate=args.lr,
            device=args.device,
            seed=args.seed,
            action_mode=args.action_mode,
        )
        bounded_eval = evaluate_scheduler(
            num_episodes=args.eval_episodes,
            steps_per_episode=args.eval_steps,
            seed=args.seed,
            use_score=True,
            checkpoint_path=bounded_ckpt,
            fallback_to_heuristic=True,
        )
        payload["runs"]["bounded_ranking_on"] = {
            "checkpoint": bounded_ckpt,
            "pretrain_metrics": bounded_pretrain_metrics,
            "metrics": bounded_eval,
            "delta_vs_off": _delta(bounded_eval, off_metrics),
        }

    if not args.skip_soft:
        soft_ckpt, soft_pretrain_metrics = pretrain_mode(
            mode="soft",
            output_dir=output_dir,
            episodes=args.pretrain_episodes,
            steps_per_episode=args.pretrain_steps,
            epochs=args.pretrain_epochs,
            learning_rate=args.lr,
            device=args.device,
            seed=args.seed,
            action_mode=args.action_mode,
        )
        soft_eval = evaluate_scheduler(
            num_episodes=args.eval_episodes,
            steps_per_episode=args.eval_steps,
            seed=args.seed,
            use_score=True,
            checkpoint_path=soft_ckpt,
            fallback_to_heuristic=True,
        )
        payload["runs"]["soft_on"] = {
            "checkpoint": soft_ckpt,
            "pretrain_metrics": soft_pretrain_metrics,
            "metrics": soft_eval,
            "delta_vs_off": _delta(soft_eval, off_metrics),
        }

    if "top1_on" in payload["runs"] and "ranking_on" in payload["runs"]:
        payload["top1_vs_ranking"] = _delta(
            payload["runs"]["ranking_on"]["metrics"],  # type: ignore[index]
            payload["runs"]["top1_on"]["metrics"],  # type: ignore[index]
        )
    if "top1_on" in payload["runs"] and "soft_on" in payload["runs"]:
        payload["top1_vs_soft"] = _delta(
            payload["runs"]["soft_on"]["metrics"],  # type: ignore[index]
            payload["runs"]["top1_on"]["metrics"],  # type: ignore[index]
        )
    if "ranking_on" in payload["runs"] and "soft_on" in payload["runs"]:
        payload["ranking_vs_soft"] = _delta(
            payload["runs"]["soft_on"]["metrics"],  # type: ignore[index]
            payload["runs"]["ranking_on"]["metrics"],  # type: ignore[index]
        )
    if "ranking_on" in payload["runs"] and "bounded_ranking_on" in payload["runs"]:
        payload["ranking_vs_bounded_ranking"] = _delta(
            payload["runs"]["bounded_ranking_on"]["metrics"],  # type: ignore[index]
            payload["runs"]["ranking_on"]["metrics"],  # type: ignore[index]
        )

    result_path = output_dir / f"score_compare_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[score-exp] Saved comparison report to {result_path}")


if __name__ == "__main__":
    main()
