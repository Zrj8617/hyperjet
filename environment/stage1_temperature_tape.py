from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterator, Sequence

import config
import numpy as np

from environment.dag_tasks import DAGJob, DAGTaskManager, TaskNode
from environment.stage1_temperature_sampling import canonical_json_bytes, canonical_sha256, file_sha256


TAPE_SCHEMA = "stage1_temperature_random_material_tape_v1"
SHARD_SCHEMA = "stage1_temperature_random_material_shard_v1"
CANONICAL_JSON_SCHEMA = "compact_sorted_utf8_json_v1"
FORMAL_EPISODES = 20
EPISODE_SLOTS = 200
BASE_SCENARIO_SEED = 424242


def scenario_seed_for_episode(episode_index: int) -> int:
    index = int(episode_index)
    if not 0 <= index < FORMAL_EPISODES:
        raise ValueError("episode_index must be in 0..19")
    return BASE_SCENARIO_SEED + index


def stable_dag_id(scenario_seed: int, slot_index: int, ue_id: int) -> str:
    return "stage1t_dag_" + canonical_sha256(
        ["stage1_temperature_dag_v1", int(scenario_seed), int(slot_index), int(ue_id)]
    )


def stable_task_id(dag_id: str, local_task_index: int) -> str:
    return f"{dag_id}_task_{int(local_task_index):04d}"


def _keyed_seed(namespace: str, *values: Any) -> int:
    digest = hashlib.sha256(canonical_json_bytes([namespace, *values])).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


@contextmanager
def isolated_rng(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) & 0xFFFFFFFF)
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _serialize_template(job: DAGJob, manager: DAGTaskManager, *, scenario_seed: int, slot_index: int, ue_id: int) -> dict[str, Any]:
    dag_id = stable_dag_id(scenario_seed, slot_index, ue_id)
    old_to_new = {
        old_id: stable_task_id(dag_id, local_index)
        for local_index, old_id in enumerate(job.task_ids)
    }
    tasks: list[dict[str, Any]] = []
    for local_index, old_id in enumerate(job.task_ids):
        task = manager.get_task(old_id)
        if task is None:
            raise RuntimeError("DAG generator returned a missing task")
        tasks.append(
            {
                "local_task_index": local_index,
                "task_id": old_to_new[old_id],
                "input_data_size_mb": float(task.input_data_size_mb),
                "output_data_size_mb": float(task.output_data_size_mb),
                "task_complexity": str(task.task_complexity),
                "task_constant": int(task.task_constant),
                "num_operation": float(task.num_operation),
                "level": int(task.level),
                "topological_index": int(task.topological_index),
                "predecessors": [old_to_new[value] for value in task.predecessors],
                "successors": [old_to_new[value] for value in task.successors],
                "is_critical_path": bool(task.is_critical_path),
            }
        )
    return {
        "dag_id": dag_id,
        "ue_id": int(ue_id),
        "slot_index": int(slot_index),
        "base_upload_bandwidth_mbps": float(job.base_upload_bandwidth_mbps),
        "base_download_bandwidth_mbps": float(job.base_download_bandwidth_mbps),
        "task_ids": [old_to_new[value] for value in job.task_ids],
        "sink_task_ids": [old_to_new[value] for value in job.sink_task_ids],
        "khop_hyperedges_global": [[old_to_new[value] for value in edge] for edge in job.khop_hyperedges_global],
        "tasks": tasks,
    }


def generate_potential_template(*, scenario_seed: int, slot_index: int, ue_id: int) -> dict[str, Any]:
    seed = _keyed_seed("stage1_temperature_dag_template", scenario_seed, slot_index, ue_id)
    with isolated_rng(seed):
        manager = DAGTaskManager(max_active_dags_per_ue=1)
        job = manager.create_dag_for_ue(
            ue_id=int(ue_id),
            source_pos=np.zeros(2, dtype=np.float32),
            current_time_step=0.0,
        )
        return _serialize_template(
            job,
            manager,
            scenario_seed=int(scenario_seed),
            slot_index=int(slot_index),
            ue_id=int(ue_id),
        )


def instantiate_potential_template(
    manager: DAGTaskManager,
    template: dict[str, Any],
    *,
    source_pos: Sequence[float],
    arrival_time: float,
) -> DAGJob:
    validate_template(template)
    ue_id = int(template["ue_id"])
    if not manager.can_accept_dag_for_ue(ue_id):
        raise ValueError("active DAG cap rejected potential template")
    dag_id = str(template["dag_id"])
    if manager.get_job(dag_id) is not None:
        raise ValueError(f"duplicate DAG instantiation: {dag_id}")
    source = np.asarray(source_pos, dtype=np.float32).reshape(-1)[:2].copy()
    tasks: list[TaskNode] = []
    for payload in template["tasks"]:
        tasks.append(TaskNode(
            task_id=str(payload["task_id"]), dag_id=dag_id, ue_id=ue_id,
            input_data_size_mb=float(payload["input_data_size_mb"]),
            output_data_size_mb=float(payload["output_data_size_mb"]),
            task_complexity=str(payload["task_complexity"]), task_constant=int(payload["task_constant"]),
            num_operation=float(payload["num_operation"]), level=int(payload["level"]),
            source_pos=source.copy(), arrival_time=float(arrival_time),
            topological_index=int(payload["topological_index"]),
            predecessors=[str(value) for value in payload["predecessors"]],
            successors=[str(value) for value in payload["successors"]],
            is_critical_path=bool(payload["is_critical_path"]),
        ))
    job = DAGJob(
        dag_id=dag_id, ue_id=ue_id, arrival_time=float(arrival_time), source_pos=source.copy(),
        base_upload_bandwidth_mbps=float(template["base_upload_bandwidth_mbps"]),
        base_download_bandwidth_mbps=float(template["base_download_bandwidth_mbps"]),
        task_ids=[str(value) for value in template["task_ids"]],
        sink_task_ids=[str(value) for value in template["sink_task_ids"]],
        khop_hyperedges_global=[[str(value) for value in edge] for edge in template["khop_hyperedges_global"]],
    )
    manager._jobs[dag_id] = job
    for task in tasks:
        if task.task_id in manager._tasks:
            raise ValueError(f"duplicate task instantiation: {task.task_id}")
        manager._tasks[task.task_id] = task
        manager._tasks_by_ue.setdefault(ue_id, []).append(task.task_id)
    manager._job_counter += 1
    manager._task_counter += len(tasks)
    manager._dag_arrival_version += 1
    manager._last_created_dag_ids = [dag_id]
    manager.refresh_ready_states()
    return job


def generate_scenario_shard(episode_index: int, *, num_ues: int | None = None, num_uavs: int | None = None, episode_slots: int = EPISODE_SLOTS) -> dict[str, Any]:
    index = int(episode_index)
    scenario_seed = scenario_seed_for_episode(index)
    resolved_ues = int(config.NUM_UES if num_ues is None else num_ues)
    resolved_uavs = int(config.NUM_UAVS if num_uavs is None else num_uavs)
    if int(episode_slots) != EPISODE_SLOTS:
        raise ValueError("formal Stage 1 temperature episodes must contain exactly 200 slots")
    rng = np.random.default_rng(_keyed_seed("stage1_temperature_scenario", scenario_seed))
    shard: dict[str, Any] = {
        "schema": SHARD_SCHEMA,
        "episode_index": index,
        "evaluation_scenario_seed": scenario_seed,
        "episode_slots": EPISODE_SLOTS,
        "num_ues": resolved_ues,
        "num_uavs": resolved_uavs,
        "hotspot_center_uniforms": rng.random(2).tolist(),
        "ue_initial_uniforms": rng.random((resolved_ues, 4)).tolist(),
        "uav_position_uniforms": rng.random((resolved_uavs, 2)).tolist(),
        "ue_mobility_standard_normals": rng.standard_normal((EPISODE_SLOTS, resolved_ues, 2)).tolist(),
        "arrival_uniforms": rng.random((EPISODE_SLOTS, resolved_ues)).tolist(),
        "potential_dag_templates": [],
    }
    shard["potential_dag_templates"] = [
        generate_potential_template(scenario_seed=scenario_seed, slot_index=slot_index, ue_id=ue_id)
        for slot_index in range(EPISODE_SLOTS)
        for ue_id in range(resolved_ues)
    ]
    validate_scenario_shard(shard)
    return shard


def validate_template(template: dict[str, Any]) -> None:
    forbidden = {"source_pos", "arrival_time", "admitted", "generated", "offered"}
    if forbidden.intersection(template):
        raise ValueError("potential template contains trajectory-dependent fields")
    dag_id = str(template["dag_id"])
    if not dag_id.startswith("stage1t_dag_"):
        raise ValueError("invalid stable DAG ID")
    seen: set[str] = set()
    for local_index, task in enumerate(template["tasks"]):
        expected = stable_task_id(dag_id, local_index)
        if int(task["local_task_index"]) != local_index or str(task["task_id"]) != expected:
            raise ValueError("invalid stable task mapping")
        if expected in seen:
            raise ValueError("duplicate task ID")
        seen.add(expected)


def validate_scenario_shard(shard: dict[str, Any]) -> None:
    if shard.get("schema") != SHARD_SCHEMA:
        raise ValueError("unsupported Stage 1 temperature shard schema")
    index = int(shard["episode_index"])
    if int(shard["evaluation_scenario_seed"]) != scenario_seed_for_episode(index):
        raise ValueError("scenario seed mapping mismatch")
    if int(shard["episode_slots"]) != EPISODE_SLOTS:
        raise ValueError("episode length mismatch")
    n_ues, n_uavs = int(shard["num_ues"]), int(shard["num_uavs"])
    if len(shard["hotspot_center_uniforms"]) != 2 or len(shard["ue_initial_uniforms"]) != n_ues or len(shard["uav_position_uniforms"]) != n_uavs:
        raise ValueError("initial primitive dimensions mismatch")
    if len(shard["ue_mobility_standard_normals"]) != EPISODE_SLOTS or len(shard["arrival_uniforms"]) != EPISODE_SLOTS:
        raise ValueError("slot primitive dimensions mismatch")
    templates = shard["potential_dag_templates"]
    if len(templates) != EPISODE_SLOTS * n_ues:
        raise ValueError("potential template count mismatch")
    forbidden_shard_fields = {"ue_positions", "arrival_bits", "hotspot_membership", "generated_bits", "admitted_bits"}
    if forbidden_shard_fields.intersection(shard):
        raise ValueError("shard contains trajectory-dependent final events")
    for slot_index in range(EPISODE_SLOTS):
        if len(shard["ue_mobility_standard_normals"][slot_index]) != n_ues or len(shard["arrival_uniforms"][slot_index]) != n_ues:
            raise ValueError("per-slot primitive dimensions mismatch")
        for ue_id in range(n_ues):
            template = templates[slot_index * n_ues + ue_id]
            validate_template(template)
            if int(template["slot_index"]) != slot_index or int(template["ue_id"]) != ue_id:
                raise ValueError("potential template index mismatch")
            expected = stable_dag_id(int(shard["evaluation_scenario_seed"]), slot_index, ue_id)
            if str(template["dag_id"]) != expected:
                raise ValueError("potential template stable DAG ID mismatch")


def save_json_create_only(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json_bytes(payload).decode("utf-8"))
        handle.write("\n")


def load_scenario_shard(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_scenario_shard(payload)
    return payload


def build_manifest(shard_paths: Sequence[Path], *, root: Path) -> dict[str, Any]:
    if len(shard_paths) != FORMAL_EPISODES:
        raise ValueError("formal tape manifest requires exactly 20 shards")
    records = []
    for index, path in enumerate(shard_paths):
        shard = load_scenario_shard(path)
        if int(shard["episode_index"]) != index:
            raise ValueError("ordered shard episode mismatch")
        records.append({
            "episode_index": index,
            "evaluation_scenario_seed": scenario_seed_for_episode(index),
            "path": path.resolve().relative_to(root.resolve()).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    manifest = {
        "schema": TAPE_SCHEMA,
        "canonical_json_schema": CANONICAL_JSON_SCHEMA,
        "episode_slots": EPISODE_SLOTS,
        "episode_indices": list(range(FORMAL_EPISODES)),
        "scenario_seed_mapping": "424242 + episode_index",
        "num_ues": int(config.NUM_UES), "num_uavs": int(config.NUM_UAVS),
        "active_dag_cap": 1, "queue_cap": 16,
        "area_width": float(config.AREA_WIDTH), "area_height": float(config.AREA_HEIGHT),
        "uav_altitude": float(config.UAV_ALTITUDE), "hotspot_radius": float(config.HOTSPOT_RADIUS),
        "time_slot_duration": float(config.TIME_SLOT_DURATION),
        "dag_base_arrival_prob": float(config.DAG_BASE_ARRIVAL_PROB),
        "dag_hotspot_arrival_multiplier": float(config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER),
        "ue_walk_speed_mean": float(config.UE_WALK_SPEED_MEAN), "ue_gm_min_speed": float(config.UE_GM_MIN_SPEED),
        "ue_gm_max_speed": float(config.UE_GM_MAX_SPEED), "ue_gm_alpha": float(config.UE_GM_ALPHA),
        "ue_gm_speed_sigma": float(config.UE_GM_SPEED_SIGMA), "ue_gm_theta_sigma": float(config.UE_GM_THETA_SIGMA),
        "ue_service_waiting_speed_scale": float(config.UE_SERVICE_WAITING_SPEED_SCALE),
        "shards": records,
    }
    manifest["logical_tape_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any], *, root: Path, validate_shards: bool = True) -> None:
    if manifest.get("schema") != TAPE_SCHEMA or int(manifest.get("episode_slots", -1)) != EPISODE_SLOTS:
        raise ValueError("invalid Stage 1 temperature tape manifest")
    expected = dict(manifest)
    stored = str(expected.pop("logical_tape_sha256"))
    if canonical_sha256(expected) != stored:
        raise ValueError("logical tape checksum mismatch")
    if len(manifest["shards"]) != FORMAL_EPISODES:
        raise ValueError("manifest shard count mismatch")
    for index, record in enumerate(manifest["shards"]):
        path = root / str(record["path"])
        if int(record["episode_index"]) != index or int(record["evaluation_scenario_seed"]) != scenario_seed_for_episode(index):
            raise ValueError("manifest episode mapping mismatch")
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]) or file_sha256(path) != str(record["sha256"]):
            raise ValueError("manifest shard identity mismatch")
        if validate_shards:
            load_scenario_shard(path)


def potential_template_at(shard: dict[str, Any], slot_index: int, ue_id: int) -> dict[str, Any]:
    n_ues = int(shard["num_ues"])
    return shard["potential_dag_templates"][int(slot_index) * n_ues + int(ue_id)]
