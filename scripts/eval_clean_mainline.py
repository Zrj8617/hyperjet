from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_slot_orchestrator import encode_prepared_slot, prepare_slot_state
from marl_models.mappo.clean_trainer import CleanTrainingModules
from scripts.train_clean_mainline import _movement_action_distribution


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic HyperUAV clean mainline evaluation.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--arrival-steps", type=int, default=200)
    parser.add_argument("--max-drain-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if _torch_cuda_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("logs") / "clean_eval")
    parser.add_argument("--run-name", type=str, default="eval")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-render", action="store_true", default=True)
    parser.add_argument("--task-embedding-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    return parser


def create_eval_run_directory(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(args.run_name)).strip("_")
    tag = clean_name or "eval"
    run_id = f"{timestamp}_{tag}_seed{int(args.seed)}"
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cli": _namespace_to_dict(args),
        "protocol": {
            "arrival_phase": "normal UE movement and DAG arrivals",
            "drain_phase": "UE movement continues; DAG arrivals disabled; executor continues",
            "throughput_denominator": "total_executed_slots * TIME_SLOT_DURATION",
            "default_action": "masked_argmax_deterministic",
        },
        "clean_scene": {
            "AREA_WIDTH": config.AREA_WIDTH,
            "AREA_HEIGHT": config.AREA_HEIGHT,
            "NUM_UAVS": config.NUM_UAVS,
            "NUM_UES": config.NUM_UES,
            "TIME_SLOT_DURATION": config.TIME_SLOT_DURATION,
            "HOTSPOT_RADIUS": config.HOTSPOT_RADIUS,
        },
    }


def initialize_eval_files(run_dir: Path, args: argparse.Namespace) -> None:
    _write_json(run_dir / "config.json", build_eval_config(args))
    _write_json(
        run_dir / "eval_summary.json",
        {
            "status": "initialized",
            "run_dir": str(run_dir),
            "torch_required_for_evaluation": True,
            "deterministic": bool(args.deterministic),
            "arrival_steps": int(args.arrival_steps),
            "max_drain_steps": int(args.max_drain_steps),
        },
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = create_eval_run_directory(args)
    initialize_eval_files(run_dir, args)
    torch = _require_torch()
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for deterministic clean evaluation.")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not bool(args.deterministic):
        raise ValueError("Clean evaluation defaults to deterministic masked argmax; stochastic eval is not implemented in T15.")

    _set_seed(int(args.seed), torch=torch)
    device = torch.device(str(args.device))
    checkpoint_payload = torch.load(Path(args.checkpoint), map_location="cpu")
    module_dims = _module_dims_from_checkpoint(checkpoint_payload, args)
    modules = _build_modules(dims=module_dims, device=device)
    _load_module_state(modules, checkpoint_payload)
    _set_eval_mode(modules)

    metrics_rows: list[dict[str, Any]] = []
    aggregate = _empty_aggregate()
    for episode in range(int(args.episodes)):
        episode_seed = int(args.seed) + episode
        _set_seed(episode_seed, torch=torch)
        env = Env()
        graph_builder = CleanGraphBuilder()
        env.reset()
        graph_builder.reset()
        episode_result = _run_eval_episode(
            env=env,
            graph_builder=graph_builder,
            modules=modules,
            device=device,
            arrival_steps=int(args.arrival_steps),
            max_drain_steps=int(args.max_drain_steps),
            episode=episode,
        )
        metrics_rows.append(episode_result)
        _write_jsonl(run_dir / "eval_metrics.jsonl", episode_result)
        _update_aggregate(aggregate, episode_result)

    summary = _aggregate_summary(aggregate, episode_count=int(args.episodes))
    summary.update(
        {
            "status": "completed",
            "run_dir": str(run_dir),
            "checkpoint": str(args.checkpoint),
            "deterministic": True,
            "episodes": int(args.episodes),
        }
    )
    _write_json(run_dir / "eval_summary.json", summary)
    return summary


def _run_eval_episode(
    *,
    env: Env,
    graph_builder: CleanGraphBuilder,
    modules: CleanTrainingModules,
    device: Any,
    arrival_steps: int,
    max_drain_steps: int,
    episode: int,
) -> dict[str, Any]:
    arrival_slots = 0
    drain_slots = 0
    movement_action_counts = {str(action): 0 for action in config.CLEAN_MOVEMENT_ACTIONS}
    offloading_action_count = 0
    last_info: dict[str, Any] = {}

    for _ in range(max(int(arrival_steps), 0)):
        done, info, movement_counts, off_count = _eval_one_slot(
            env=env,
            graph_builder=graph_builder,
            modules=modules,
            device=device,
            allow_dag_arrivals=True,
        )
        arrival_slots += 1
        offloading_action_count += off_count
        _merge_counts(movement_action_counts, movement_counts)
        last_info = info
        if done:
            break

    while drain_slots < max(int(max_drain_steps), 0) and _active_dag_count(env) > 0:
        done, info, movement_counts, off_count = _eval_one_slot(
            env=env,
            graph_builder=graph_builder,
            modules=modules,
            device=device,
            allow_dag_arrivals=False,
        )
        drain_slots += 1
        offloading_action_count += off_count
        _merge_counts(movement_action_counts, movement_counts)
        last_info = info
        if done:
            break

    total_slots = int(arrival_slots + drain_slots)
    info_metrics = env.metrics.to_info(total_slots)
    total_time = float(total_slots) * float(config.TIME_SLOT_DURATION)
    completed = float(info_metrics.get("completed_dag_count", 0.0))
    generated = float(info_metrics.get("generated_dag_count", 0.0))
    assignment_entries = float(env.metrics.metrics.action_count)
    movement_distribution = _normalized_distribution(movement_action_counts)
    return {
        "episode": int(episode),
        "generated_DAG_count": generated,
        "completed_DAG_count": completed,
        "DAG_completion_rate": float(completed / max(generated, 1.0)),
        "Average_DAG_flowtime": float(info_metrics.get("average_dag_flowtime", 0.0)),
        "DAG_throughput": float(completed / max(total_time, float(config.TIME_SLOT_DURATION))),
        "Average_critical_path_task_completion_delay": float(
            info_metrics.get("average_critical_path_task_completion_delay", 0.0)
        ),
        "Energy_per_completed_DAG": float(info_metrics.get("energy_per_completed_dag", 0.0)),
        "total_executed_slots": total_slots,
        "arrival_slots_executed": int(arrival_slots),
        "drain_slots_executed": int(drain_slots),
        "invalid_assignment_count": float(info_metrics.get("invalid_assignment_count", 0.0)),
        "invalid_assignment_rate": None if assignment_entries <= 0.0 else float(env.metrics.metrics.invalid_assignment_count / assignment_entries),
        "action_executed_rate": None if assignment_entries <= 0.0 else float(env.metrics.metrics.executed_action_count / assignment_entries),
        "movement_action_distribution": movement_distribution,
        "offloading_action_count": int(offloading_action_count),
        "total_evaluation_time": total_time,
        "active_dag_count_after_eval": int(_active_dag_count(env)),
        "last_info": _jsonable(last_info),
    }


def _eval_one_slot(
    *,
    env: Env,
    graph_builder: CleanGraphBuilder,
    modules: CleanTrainingModules,
    device: Any,
    allow_dag_arrivals: bool,
) -> tuple[bool, dict[str, Any], dict[str, int], int]:
    import torch

    with _dag_arrival_enabled(allow_dag_arrivals):
        prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
    with torch.no_grad():
        encoded = encode_prepared_slot(
            prepared_state=prepared,
            env=env,
            hgnn=modules.hgnn,
            critic=modules.critic,
            movement_actor=modules.movement_actor,
            device=device,
        )
        selected_movement = torch.argmax(encoded.movement_logits, dim=-1)
    movement_actions = {
        int(uav_id): int(selected_movement[idx].detach().cpu().item())
        for idx, uav_id in enumerate(encoded.movement_observation.uav_ids)
    }
    movement_records = [
        type("MovementEvalRecord", (), {"selected_action": action})()
        for action in movement_actions.values()
    ]
    env.apply_movement(movement_actions)
    ready_tasks = [env.task_manager.get_task(task_id) for task_id in prepared.frozen_ready_task_ids]
    ready_tasks = [task for task in ready_tasks if task is not None and task.is_ready]
    assignment_buffer = modules.offloading_actor.act(
        frozen_ready_tasks=ready_tasks,
        task_embeddings=encoded.task_embeddings.detach(),
        graph_snapshot=prepared.graph_snapshot,
        task_manager=env.task_manager,
        uavs=env.uavs,
        executor=env.executor,
        current_time_step=env.time_step,
        uav_service_positions=env.uav_service_positions,
        ue_service_positions=env.ue_service_positions,
        ues=env.ues,
        deterministic=True,
    )
    _, _, done, info = env.commit_and_advance(assignment_buffer=assignment_buffer)
    movement_distribution = _movement_action_distribution(movement_records)
    movement_counts = {
        action: int(round(float(ratio) * max(len(movement_records), 1)))
        for action, ratio in movement_distribution.items()
    }
    info["movement_action_distribution"] = movement_distribution
    info["offloading_action_count"] = len(modules.offloading_actor.latest_records)
    return bool(done), info, movement_counts, len(modules.offloading_actor.latest_records)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(args)
    except ModuleNotFoundError as exc:
        if exc.name == "torch" or "torch" in str(exc).lower():
            print("clean deterministic evaluation requires torch and a clean checkpoint.", file=sys.stderr)
            return 2
        raise
    except (FileNotFoundError, ValueError) as exc:
        print(f"clean deterministic evaluation unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


def _build_modules(*, dims: dict[str, int], device: Any) -> CleanTrainingModules:
    from marl_models.hgnn import CleanIncidenceHGNN
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim

    modules = CleanTrainingModules(
        hgnn=CleanIncidenceHGNN(
            task_feature_dim=int(dims["task_feature_dim"]),
            hidden_dim=int(dims["hidden_dim"]),
            output_dim=int(dims["task_embedding_dim"]),
        ).to(device),
        movement_actor=CleanMovementActor(
            task_embedding_dim=int(dims["task_embedding_dim"]),
            hidden_dim=int(dims["hidden_dim"]),
        ).to(device),
        offloading_actor=CleanOffloadingActor(
            task_embedding_dim=int(dims["task_embedding_dim"]),
            hidden_dim=int(dims["hidden_dim"]),
        ).to(device),
        critic=CleanCentralizedCritic(
            input_dim=clean_critic_input_dim(int(dims["task_embedding_dim"]), config.NUM_UAVS),
            hidden_dim=int(dims["hidden_dim"]),
        ).to(device),
    )
    return modules


def _load_module_state(modules: CleanTrainingModules, payload: dict[str, Any]) -> None:
    modules.hgnn.load_state_dict(payload["hgnn"])
    modules.movement_actor.load_state_dict(payload["movement_actor"])
    modules.offloading_actor.load_state_dict(payload["offloading_actor"])
    modules.critic.load_state_dict(payload["critic"])


def _module_dims_from_checkpoint(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, int]:
    cli_config = payload.get("config", {}).get("cli", {}) if isinstance(payload.get("config"), dict) else {}
    task_embedding_dim = args.task_embedding_dim or cli_config.get("task_embedding_dim") or _infer_hgnn_output_dim(payload)
    hidden_dim = args.hidden_dim or cli_config.get("hidden_dim") or _infer_hidden_dim(payload)
    return {
        "task_feature_dim": int(_infer_hgnn_task_feature_dim(payload)),
        "task_embedding_dim": int(task_embedding_dim),
        "hidden_dim": int(hidden_dim),
    }


def _infer_hgnn_task_feature_dim(payload: dict[str, Any]) -> int:
    state = payload.get("hgnn", {})
    weight = state.get("input_proj.weight")
    return int(weight.shape[1]) if hasattr(weight, "shape") else 12


def _infer_hgnn_output_dim(payload: dict[str, Any]) -> int:
    state = payload.get("hgnn", {})
    weight = state.get("output_proj.weight")
    return int(weight.shape[0]) if hasattr(weight, "shape") else 64


def _infer_hidden_dim(payload: dict[str, Any]) -> int:
    state = payload.get("hgnn", {})
    weight = state.get("input_proj.weight")
    return int(weight.shape[0]) if hasattr(weight, "shape") else 128


def _set_eval_mode(modules: CleanTrainingModules) -> None:
    modules.hgnn.eval()
    modules.movement_actor.eval()
    modules.offloading_actor.eval()
    modules.critic.eval()


@contextmanager
def _dag_arrival_enabled(enabled: bool):
    original = config.DAG_BASE_ARRIVAL_PROB
    if not enabled:
        config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        yield
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original


def _active_dag_count(env: Env) -> int:
    return sum(1 for job in env.task_manager.jobs.values() if not job.completed)


def _empty_aggregate() -> dict[str, Any]:
    return {
        "episodes": [],
        "movement_counts": {str(action): 0.0 for action in config.CLEAN_MOVEMENT_ACTIONS},
    }


def _update_aggregate(aggregate: dict[str, Any], row: dict[str, Any]) -> None:
    aggregate["episodes"].append(row)
    for action, ratio in row.get("movement_action_distribution", {}).items():
        aggregate["movement_counts"][action] = aggregate["movement_counts"].get(action, 0.0) + float(ratio)


def _aggregate_summary(aggregate: dict[str, Any], *, episode_count: int) -> dict[str, Any]:
    rows = aggregate["episodes"]
    return {
        "generated_DAG_count": float(sum(row["generated_DAG_count"] for row in rows)),
        "completed_DAG_count": float(sum(row["completed_DAG_count"] for row in rows)),
        "DAG_completion_rate": _mean(row["DAG_completion_rate"] for row in rows),
        "Average_DAG_flowtime": _mean(row["Average_DAG_flowtime"] for row in rows),
        "DAG_throughput": _mean(row["DAG_throughput"] for row in rows),
        "Average_critical_path_task_completion_delay": _mean(row["Average_critical_path_task_completion_delay"] for row in rows),
        "Energy_per_completed_DAG": _mean(row["Energy_per_completed_DAG"] for row in rows),
        "total_executed_slots": int(sum(row["total_executed_slots"] for row in rows)),
        "arrival_slots_executed": int(sum(row["arrival_slots_executed"] for row in rows)),
        "drain_slots_executed": int(sum(row["drain_slots_executed"] for row in rows)),
        "invalid_assignment_count": float(sum(row["invalid_assignment_count"] for row in rows)),
        "invalid_assignment_rate": _mean(row["invalid_assignment_rate"] for row in rows if row["invalid_assignment_rate"] is not None),
        "action_executed_rate": _mean(row["action_executed_rate"] for row in rows if row["action_executed_rate"] is not None),
        "movement_action_distribution": {
            action: float(total) / max(float(episode_count), 1.0)
            for action, total in aggregate["movement_counts"].items()
        },
        "offloading_action_count": int(sum(row["offloading_action_count"] for row in rows)),
    }


def _mean(values: Any) -> float:
    values_list = [float(value) for value in values]
    return float(np.mean(values_list)) if values_list else 0.0


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value)


def _normalized_distribution(counts: dict[str, int]) -> dict[str, float]:
    total = max(float(sum(counts.values())), 1.0)
    return {str(key): float(value) / total for key, value in counts.items()}


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ModuleNotFoundError("torch is required for clean deterministic evaluation") from exc
        raise
    return torch


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def _set_seed(seed: int, *, torch: Any) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _namespace_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), ensure_ascii=True, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
