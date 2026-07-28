from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import config
from environment.assignment import CleanAssignmentBuffer, TemporaryReservationState
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state
from scripts import train_greedy_imitation_gate as gate


DATASET_SCHEMA = "greedy_imitation_frozen_dataset_v1"
SAMPLE_SCHEMA = "greedy_imitation_decision_sample_v1"
SAMPLES_FILENAME = "decision_samples.jsonl"
GRAPH_SNAPSHOTS_FILENAME = "graph_snapshots.jsonl"
MANIFEST_FILENAME = "dataset_manifest.json"
GRAPH_FIELDS = (
    "task_features",
    "incidence_matrix",
    "hyperedge_type_ids",
    "task_id_to_idx",
    "idx_to_task_id",
    "active_task_ids",
    "ready_task_ids",
    "pending_task_ids",
    "hyperedges",
    "dag_hyperedges",
    "khop_hyperedges",
    "attribute_hyperedges",
    "partition_hyperedges",
    "frozen_ready_task_ids",
)


def generate_frozen_dataset(
    *,
    dataset_dir: Path,
    dataset_seed: int,
    episodes: int,
    max_steps_per_episode: int,
    trajectory_policies: list[str] | tuple[str, ...],
    train_fraction: float,
    val_fraction: float,
    completed_dag_weight: float = 16.0,
    dag_base_arrival_prob: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        raise FileExistsError(f"dataset directory is not empty: {dataset_dir}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    if int(max_steps_per_episode) <= 0:
        raise ValueError("max_steps_per_episode must be positive")
    policies = [str(item) for item in trajectory_policies]
    if not policies or any(item not in gate.TRAJECTORY_POLICIES for item in policies):
        raise ValueError(f"trajectory_policies must be a non-empty subset of {gate.TRAJECTORY_POLICIES}")

    split_bounds = gate._split_bounds(int(episodes), float(train_fraction), float(val_fraction))
    samples: list[dict[str, Any]] = []
    graph_snapshots: dict[str, dict[str, Any]] = {}
    skipped_by_split_policy: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    valid_slots_by_split: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    original_arrival_probability = config.DAG_BASE_ARRIVAL_PROB
    if dag_base_arrival_prob is not None:
        config.DAG_BASE_ARRIVAL_PROB = float(dag_base_arrival_prob)
    try:
        for policy in policies:
            gate._set_seed(int(dataset_seed))
            env = Env(completed_dag_weight=float(completed_dag_weight), freeze_ue_mobility=True)
            graph_builder = CleanGraphBuilder()
            try:
                for episode in range(int(episodes)):
                    env.reset()
                    graph_builder.reset()
                    split = _split_for_episode(int(episode), split_bounds)
                    trajectory_id = gate._trajectory_id(
                        trajectory_policy=policy,
                        environment_seed=int(dataset_seed),
                        episode=int(episode),
                    )
                    for slot in range(int(max_steps_per_episode)):
                        prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
                        env.apply_movement({})
                        ready_tasks = [
                            env.task_manager.get_task(task_id)
                            for task_id in prepared.frozen_ready_task_ids
                        ]
                        ready_tasks = [task for task in ready_tasks if task is not None and task.is_ready]
                        assignment_buffer = CleanAssignmentBuffer()
                        reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
                        slot_had_sample = False
                        slot_skipped_no_candidate = 0
                        for decision_order, task in enumerate(ready_tasks):
                            sample = gate._build_decision_sample(
                                trajectory_policy=policy,
                                prepared=prepared,
                                task=task,
                                decision_order=int(decision_order),
                                reservation=reservation,
                                env=env,
                                environment_seed=int(dataset_seed),
                                episode=int(episode),
                                slot=int(slot),
                            )
                            if sample is None:
                                skipped_by_split_policy[split][policy] += 1
                                slot_skipped_no_candidate += 1
                                continue
                            if str(sample["trajectory_id"]) != trajectory_id:
                                raise AssertionError("sample trajectory_id drifted during dataset generation")
                            sample["split"] = split
                            compact_sample, graph_snapshot = _normalize_sample_storage(sample)
                            graph_snapshot_id = str(compact_sample["graph_snapshot_id"])
                            existing_graph = graph_snapshots.get(graph_snapshot_id)
                            if existing_graph is not None and canonical_json(existing_graph) != canonical_json(graph_snapshot):
                                raise AssertionError("graph snapshot hash collision")
                            graph_snapshots.setdefault(graph_snapshot_id, graph_snapshot)
                            samples.append(compact_sample)
                            slot_had_sample = True

                            behavior_idx = int(sample["behavior_idx"])
                            selected_uav_id = int(sample["candidate_uav_ids"][behavior_idx])
                            assignment_buffer.append(str(task.task_id), selected_uav_id, int(decision_order))
                            reservation.reserve(
                                str(task.task_id),
                                selected_uav_id,
                                estimated_available_time=float(sample["estimated_finish_times"][behavior_idx]),
                                estimated_queued_workload=float(sample["estimated_queued_workloads"][behavior_idx]),
                            )
                        if slot_had_sample:
                            valid_slots_by_split[split].add((policy, int(episode), int(slot)))
                        _, _, done, _ = env.commit_and_advance(
                            assignment_buffer=assignment_buffer,
                            offloading_skip_count=int(slot_skipped_no_candidate),
                        )
                        if done:
                            break
            finally:
                graph_builder.close()
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_probability

    _assert_unique_sample_ids(samples)
    samples_path = dataset_dir / SAMPLES_FILENAME
    graph_snapshots_path = dataset_dir / GRAPH_SNAPSHOTS_FILENAME
    graph_checksum = write_graph_snapshots(graph_snapshots_path, list(graph_snapshots.values()))
    samples_checksum_value = write_samples(samples_path, samples)
    checksum = _combined_checksum(graph_checksum, samples_checksum_value)
    hydrated_samples = _hydrate_samples(samples, graph_snapshots)
    manifest = _build_manifest(
        samples=hydrated_samples,
        samples_path=samples_path,
        graph_snapshots_path=graph_snapshots_path,
        graph_snapshot_count=len(graph_snapshots),
        checksum=checksum,
        graph_checksum=graph_checksum,
        samples_checksum_value=samples_checksum_value,
        dataset_seed=int(dataset_seed),
        episodes=int(episodes),
        max_steps_per_episode=int(max_steps_per_episode),
        trajectory_policies=policies,
        split_bounds=split_bounds,
        skipped_by_split_policy=skipped_by_split_policy,
        valid_slots_by_split=valid_slots_by_split,
        completed_dag_weight=float(completed_dag_weight),
        dag_base_arrival_prob=dag_base_arrival_prob,
    )
    write_json(dataset_dir / MANIFEST_FILENAME, manifest)
    validate_frozen_dataset(
        samples=hydrated_samples,
        manifest=manifest,
        samples_path=samples_path,
        graph_snapshots_path=graph_snapshots_path,
    )
    return hydrated_samples, manifest


def load_frozen_dataset(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples_path = dataset_dir / SAMPLES_FILENAME
    graph_snapshots_path = dataset_dir / GRAPH_SNAPSHOTS_FILENAME
    compact_samples: list[dict[str, Any]] = []
    if samples_path.is_file():
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                compact_samples.append(json.loads(line))
    graph_snapshots: dict[str, dict[str, Any]] = {}
    if graph_snapshots_path.is_file():
        for line in graph_snapshots_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            graph = json.loads(line)
            graph_snapshots[str(graph["graph_snapshot_id"])] = graph
    samples = _hydrate_samples(compact_samples, graph_snapshots)
    validate_frozen_dataset(
        samples=samples,
        manifest=manifest,
        samples_path=samples_path,
        graph_snapshots_path=graph_snapshots_path,
    )
    return samples, manifest


def write_samples(path: Path, samples: list[dict[str, Any]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            line = canonical_json(sample)
            encoded = (line + "\n").encode("utf-8")
            handle.write(line + "\n")
            digest.update(encoded)
    return digest.hexdigest()


def write_graph_snapshots(path: Path, graph_snapshots: list[dict[str, Any]]) -> str:
    return write_samples(path, graph_snapshots)


def samples_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_dataset(
    *,
    samples: list[dict[str, Any]],
    manifest: dict[str, Any],
    samples_path: Path,
    graph_snapshots_path: Path,
) -> dict[str, Any]:
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"unsupported dataset schema: {manifest.get('schema')}")
    if int(manifest.get("sample_count", -1)) != len(samples):
        raise ValueError("dataset sample_count does not match decision_samples.jsonl")
    actual_samples_checksum = samples_checksum(samples_path)
    actual_graph_checksum = samples_checksum(graph_snapshots_path)
    actual_checksum = _combined_checksum(actual_graph_checksum, actual_samples_checksum)
    if str(manifest.get("dataset_checksum")) != actual_checksum:
        raise ValueError("dataset checksum mismatch")
    if str(manifest.get("decision_samples_checksum")) != actual_samples_checksum:
        raise ValueError("decision sample checksum mismatch")
    if str(manifest.get("graph_snapshots_checksum")) != actual_graph_checksum:
        raise ValueError("graph snapshot checksum mismatch")

    _assert_unique_sample_ids(samples)
    sample_ids_by_split: dict[str, set[str]] = defaultdict(set)
    episodes_by_split: dict[str, set[int]] = defaultdict(set)
    trajectories_by_split: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        _validate_sample(sample)
        split = str(sample.get("split"))
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid sample split: {split}")
        sample_ids_by_split[split].add(str(sample["sample_id"]))
        episodes_by_split[split].add(int(sample["episode"]))
        trajectories_by_split[split].add(str(sample["trajectory_id"]))

    leakage_count = _leakage_count(episodes_by_split) + _leakage_count(trajectories_by_split)
    if leakage_count != 0 or int(manifest.get("leakage_count", -1)) != 0:
        raise ValueError(f"dataset split leakage detected: {leakage_count}")
    expected_ids = manifest.get("split_sample_ids", {})
    for split in ("train", "val", "test"):
        if sorted(sample_ids_by_split[split]) != sorted(str(item) for item in expected_ids.get(split, [])):
            raise ValueError(f"{split} sample IDs do not match manifest")
    return {
        "sample_count": len(samples),
        "dataset_checksum": actual_checksum,
        "leakage_count": leakage_count,
    }


def sample_ids_by_split(samples: list[dict[str, Any]]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        output[split] = [
            str(sample["sample_id"])
            for sample in samples
            if str(sample.get("split")) == split
        ]
    return output


def samples_by_split(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [sample for sample in samples if str(sample.get("split")) == split]
        for split in ("train", "val", "test")
    }


def canonical_json(value: Any) -> str:
    return json.dumps(
        gate._jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(gate._jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _build_manifest(
    *,
    samples: list[dict[str, Any]],
    samples_path: Path,
    graph_snapshots_path: Path,
    graph_snapshot_count: int,
    checksum: str,
    graph_checksum: str,
    samples_checksum_value: str,
    dataset_seed: int,
    episodes: int,
    max_steps_per_episode: int,
    trajectory_policies: list[str],
    split_bounds: dict[str, tuple[int, int]],
    skipped_by_split_policy: dict[str, dict[str, int]],
    valid_slots_by_split: dict[str, set[tuple[str, int, int]]],
    completed_dag_weight: float,
    dag_base_arrival_prob: float | None,
) -> dict[str, Any]:
    ids_by_split = sample_ids_by_split(samples)
    episode_sets = gate._split_episode_sets(split_bounds)
    trajectory_ids = {
        split: sorted(
            gate._trajectory_id(
                trajectory_policy=policy,
                environment_seed=dataset_seed,
                episode=episode,
            )
            for policy in trajectory_policies
            for episode in sorted(episode_sets[split])
        )
        for split in ("train", "val", "test")
    }
    sample_counts = {split: len(ids_by_split[split]) for split in ("train", "val", "test")}
    first_sample = samples[0] if samples else None
    dynamic_dim = (
        int(np.asarray(first_sample["dynamic_uav_features"]).shape[1])
        if first_sample is not None and np.asarray(first_sample["dynamic_uav_features"]).ndim == 2
        else 0
    )
    pair_dim = (
        int(np.asarray(first_sample["pair_features"]).shape[1])
        if first_sample is not None and np.asarray(first_sample["pair_features"]).ndim == 2
        else 0
    )
    task_dim = (
        int(np.asarray(first_sample["task_features"]).shape[1])
        if first_sample is not None and np.asarray(first_sample["task_features"]).ndim == 2
        else int(_default_task_feature_dim())
    )
    return {
        "schema": DATASET_SCHEMA,
        "version": 1,
        "sample_schema": SAMPLE_SCHEMA,
        "git_commit": gate._git_commit(),
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_seed": int(dataset_seed),
        "episodes": int(episodes),
        "max_steps_per_episode": int(max_steps_per_episode),
        "trajectory_policies": list(trajectory_policies),
        "scenario_config_snapshot": {
            "completed_dag_weight": float(completed_dag_weight),
            "dag_base_arrival_prob": (
                float(config.DAG_BASE_ARRIVAL_PROB)
                if dag_base_arrival_prob is None
                else float(dag_base_arrival_prob)
            ),
            "freeze_ue_mobility": True,
            "config": _scenario_config_snapshot(),
        },
        "task_feature_dimension": task_dim,
        "pair_feature_dimension": pair_dim,
        "dynamic_uav_feature_dimension": dynamic_dim,
        "sample_count": len(samples),
        "graph_snapshot_count": int(graph_snapshot_count),
        "episode_count": int(episodes),
        "trajectory_count": int(episodes) * len(trajectory_policies),
        "split_mode": "episode_and_trajectory",
        "split_bounds": {key: [int(value[0]), int(value[1])] for key, value in split_bounds.items()},
        "train_episode_ids": sorted(episode_sets["train"]),
        "val_episode_ids": sorted(episode_sets["val"]),
        "test_episode_ids": sorted(episode_sets["test"]),
        "train_trajectory_ids": trajectory_ids["train"],
        "val_trajectory_ids": trajectory_ids["val"],
        "test_trajectory_ids": trajectory_ids["test"],
        "train_sample_count": sample_counts["train"],
        "val_sample_count": sample_counts["val"],
        "test_sample_count": sample_counts["test"],
        "split_sample_ids": ids_by_split,
        "skipped_no_candidate": {
            split: {
                "overall": int(sum(skipped_by_split_policy[split].values())),
                "by_trajectory_policy": {
                    policy: int(skipped_by_split_policy[split].get(policy, 0))
                    for policy in trajectory_policies
                },
            }
            for split in ("train", "val", "test")
        },
        "valid_slot_count": {
            split: int(len(valid_slots_by_split[split]))
            for split in ("train", "val", "test")
        },
        "leakage_count": 0,
        "dataset_file_path": str(samples_path.resolve()),
        "graph_snapshots_file_path": str(graph_snapshots_path.resolve()),
        "dataset_checksum": checksum,
        "decision_samples_checksum": samples_checksum_value,
        "graph_snapshots_checksum": graph_checksum,
        "checksum_algorithm": "sha256",
    }


def _validate_sample(sample: dict[str, Any]) -> None:
    required = (
        "sample_id",
        "graph_snapshot_id",
        "trajectory_id",
        "trajectory_policy",
        "environment_seed",
        "episode",
        "slot",
        "decision_order",
        "task_id",
        "task_local_index",
        "task_features",
        "incidence_matrix",
        "hyperedge_type_ids",
        "task_id_to_idx",
        "idx_to_task_id",
        "active_task_ids",
        "ready_task_ids",
        "pending_task_ids",
        "dynamic_uav_features",
        "pair_features",
        "candidate_mask",
        "candidate_uav_ids",
        "candidate_uav_id_mapping",
        "estimated_finish_times",
        "greedy_label_idx",
        "valid_candidate_count",
        "greedy_margin",
    )
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"frozen sample missing fields: {missing}")
    incidence = np.asarray(sample["incidence_matrix"])
    type_ids = np.asarray(sample["hyperedge_type_ids"])
    if incidence.ndim != 2 or type_ids.ndim != 1 or incidence.shape[1] != type_ids.shape[0]:
        raise ValueError("sample hyperedge_type_ids do not align with incidence columns")
    task_features = np.asarray(sample["task_features"])
    if task_features.ndim != 2 or task_features.shape[0] != incidence.shape[0]:
        raise ValueError("sample task features do not align with incidence rows")
    local_index = int(sample["task_local_index"])
    if not 0 <= local_index < task_features.shape[0]:
        raise ValueError("sample task_local_index is outside historical graph rows")
    mask = np.asarray(sample["candidate_mask"], dtype=bool)
    if mask.ndim != 1 or not bool(mask.any()):
        raise ValueError("frozen training sample must have at least one legal candidate")
    if int(mask.sum()) != int(sample["valid_candidate_count"]):
        raise ValueError("sample valid_candidate_count does not match candidate mask")
    label = int(sample["greedy_label_idx"])
    if not 0 <= label < mask.shape[0] or not bool(mask[label]):
        raise ValueError("sample greedy label is not a legal candidate")
    mapping = [int(item) for item in sample["candidate_uav_id_mapping"]]
    if mapping != [int(item) for item in sample["candidate_uav_ids"]]:
        raise ValueError("candidate_uav_id_mapping does not match candidate_uav_ids")


def _split_for_episode(episode: int, split_bounds: dict[str, tuple[int, int]]) -> str:
    for split in ("train", "val", "test"):
        start, end = split_bounds[split]
        if int(start) <= int(episode) < int(end):
            return split
    raise ValueError(f"episode {episode} does not belong to any split")


def _leakage_count(values_by_split: dict[str, set[Any]]) -> int:
    leakage = 0
    splits = ("train", "val", "test")
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            leakage += len(values_by_split[left].intersection(values_by_split[right]))
    return int(leakage)


def _assert_unique_sample_ids(samples: list[dict[str, Any]]) -> None:
    sample_ids = [str(sample.get("sample_id", "")) for sample in samples]
    if any(not item for item in sample_ids):
        raise ValueError("every frozen sample must have a non-empty sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample_id detected")


def _deep_jsonable_copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _normalize_sample_storage(
    sample: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    jsonable = _deep_jsonable_copy(sample)
    graph_payload = {
        key: jsonable.pop(key)
        for key in GRAPH_FIELDS
        if key in jsonable
    }
    graph_snapshot_id = hashlib.sha256(
        canonical_json(graph_payload).encode("utf-8")
    ).hexdigest()[:24]
    graph_record = {
        "schema": "greedy_imitation_graph_snapshot_v1",
        "graph_snapshot_id": graph_snapshot_id,
        **graph_payload,
    }
    jsonable["graph_snapshot_id"] = graph_snapshot_id
    return jsonable, graph_record


def _hydrate_samples(
    compact_samples: list[dict[str, Any]],
    graph_snapshots: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    shared_graph_fields: dict[str, dict[str, Any]] = {}
    for graph_id, graph_record in graph_snapshots.items():
        shared_graph_fields[graph_id] = {
            key: graph_record[key]
            for key in GRAPH_FIELDS
            if key in graph_record
        }
    for compact in compact_samples:
        graph_id = str(compact.get("graph_snapshot_id", ""))
        if graph_id not in shared_graph_fields:
            raise ValueError(f"decision sample references missing graph snapshot: {graph_id}")
        hydrated.append({**compact, **shared_graph_fields[graph_id]})
    return hydrated


def _combined_checksum(graph_checksum: str, decision_checksum: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"graph:{graph_checksum}\n".encode("ascii"))
    digest.update(f"decisions:{decision_checksum}\n".encode("ascii"))
    return digest.hexdigest()


def _scenario_config_snapshot() -> dict[str, Any]:
    names = (
        "AREA_WIDTH",
        "AREA_HEIGHT",
        "NUM_UAVS",
        "NUM_UES",
        "DAG_MAX_TASKS",
        "DAG_MAX_LEVELS",
        "DAG_MAX_PARENTS",
        "CLEAN_MAX_QUEUE_PER_UAV",
        "UAV_COMPUTE_RATE_OPS_PER_SEC",
        "ENABLE_ATTRIBUTE_HYPEREDGES",
        "ENABLE_KAHYPAR_PARTITION_HYPEREDGES",
        "ATTRIBUTE_HYPEREDGE_CLUSTER_NUM",
        "KHOP_K",
    )
    output: dict[str, Any] = {}
    for name in names:
        if hasattr(config, name):
            output[name] = gate._jsonable(getattr(config, name))
    return output


def _default_task_feature_dim() -> int:
    return 12
