from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import config
import numpy as np

from environment.assignment import (
    CleanAssignmentBuffer,
    TemporaryReservationState,
    build_offloading_candidate_components,
)
from environment.dag_tasks import (
    DAGJob,
    DAGTaskManager,
    TASK_STATE_COMPLETED,
    TASK_STATE_IN_SERVICE,
    TaskNode,
)
from environment.diagnostic_capacity import DiagnosticCapacityContext
from environment.env import Env


TAPE_SCHEMA = "active_dag_queue_cap_factorial_tape_v1"
RESULT_SCHEMA = "active_dag_queue_cap_factorial_episode_v1"
ANALYSIS_SCHEMA = "active_dag_queue_cap_factorial_analysis_v1"
FORMAL_SCENARIO_SEEDS = (42, 86, 1042)
FORMAL_EPISODES = 20
PILOT_EPISODES = 5
LOAD_SLOTS = 150
EPISODE_SLOTS = 200
BASELINE_ACTIVE_CAP = 1
BASELINE_QUEUE_CAP = 16
REASON_PRECEDENCE = (
    "task_not_ready",
    "already_reserved",
    "already_scheduled",
    "invalid_uav",
    "queue_full",
    "other",
)
CELL_FLAGS = {
    "A": (False, False),
    "B": (True, False),
    "C": (False, True),
    "D": (True, True),
}
STAGE1_ACTOR_CHECKPOINTS = {
    42: "logs/decision_ppo_bandit/20260729_215923_stage1_formal_S1-B_seed42/checkpoints/checkpoint_update_0030.pt",
    86: "logs/decision_ppo_bandit/20260729_220604_stage1_formal_S1-B_seed86/checkpoints/checkpoint_update_0030.pt",
    1042: "logs/decision_ppo_bandit/20260729_221421_stage1_formal_S1-B_seed1042/checkpoints/checkpoint_update_0030.pt",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_dag_id(scenario_seed: int, episode: int, slot_index: int, ue_id: int) -> str:
    digest = canonical_sha256([int(scenario_seed), int(episode), int(slot_index), int(ue_id)])
    return "diag_dag_" + digest


def stable_task_id(dag_id: str, local_task_index: int) -> str:
    return f"{dag_id}_task_{int(local_task_index):04d}"


def random_hash_uav(
    *,
    scenario_seed: int,
    episode: int,
    slot_index: int,
    stable_task_id_value: str,
    legal_uav_ids: Iterable[int],
) -> int:
    candidates = []
    for raw_uav_id in legal_uav_ids:
        uav_id = int(raw_uav_id)
        payload = [
            int(scenario_seed),
            int(episode),
            int(slot_index),
            str(stable_task_id_value),
            uav_id,
        ]
        candidates.append((int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest(), "big"), uav_id))
    if not candidates:
        raise ValueError("random_hash requires at least one legal UAV")
    return min(candidates)[1]


def _keyed_seed(namespace: str, values: list[Any]) -> int:
    digest = hashlib.sha256(canonical_json_bytes([str(namespace), *values])).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _keyed_uniform(namespace: str, values: list[Any]) -> float:
    integer = _keyed_seed(namespace, values)
    return float(integer) / float(1 << 64)


def _scenario_payload(scenario_seed: int, episode: int, num_ues: int, num_uavs: int) -> dict[str, Any]:
    rng = np.random.default_rng(_keyed_seed("scenario", [scenario_seed, episode]))
    radius = float(config.HOTSPOT_RADIUS)
    hotspot = [
        float(rng.uniform(radius, float(config.AREA_WIDTH) - radius)),
        float(rng.uniform(radius, float(config.AREA_HEIGHT) - radius)),
    ]
    ue_positions = [
        [
            float(rng.uniform(0.0, float(config.AREA_WIDTH))),
            float(rng.uniform(0.0, float(config.AREA_HEIGHT))),
        ]
        for _ in range(int(num_ues))
    ]
    uav_positions = [
        [
            float(rng.uniform(0.0, float(config.AREA_WIDTH))),
            float(rng.uniform(0.0, float(config.AREA_HEIGHT))),
        ]
        for _ in range(int(num_uavs))
    ]
    return {
        "hotspot_center": hotspot,
        "hotspot_radius": radius,
        "ue_positions": ue_positions,
        "uav_positions": uav_positions,
        "ue_mobility_frozen": True,
        "uav_movement": "hover",
    }


def _arrival_probability(position: list[float], scenario: dict[str, Any]) -> float:
    point = np.asarray(position, dtype=np.float64)
    center = np.asarray(scenario["hotspot_center"], dtype=np.float64)
    probability = float(config.DAG_BASE_ARRIVAL_PROB)
    if float(np.linalg.norm(point - center)) <= float(scenario["hotspot_radius"]):
        probability *= float(config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER)
    return float(np.clip(probability, 0.0, 1.0))


def _template_from_current_generator(
    *,
    scenario_seed: int,
    episode: int,
    slot_index: int,
    ue_id: int,
    source_pos: list[float],
) -> dict[str, Any]:
    offer_key = [int(scenario_seed), int(episode), int(slot_index), int(ue_id)]
    state = np.random.get_state()
    try:
        np.random.seed(_keyed_seed("dag_template", offer_key) & 0xFFFFFFFF)
        manager = DAGTaskManager(max_active_dags_per_ue=2)
        job = manager.create_dag_for_ue(
            ue_id=int(ue_id),
            source_pos=np.asarray(source_pos, dtype=np.float32),
            current_time_step=float(slot_index + 1) * float(config.TIME_SLOT_DURATION),
        )
    finally:
        np.random.set_state(state)

    dag_id = stable_dag_id(scenario_seed, episode, slot_index, ue_id)
    old_to_new = {
        old_id: stable_task_id(dag_id, local_index)
        for local_index, old_id in enumerate(job.task_ids)
    }
    tasks = []
    for local_index, old_id in enumerate(job.task_ids):
        task = manager.get_task(old_id)
        assert task is not None
        tasks.append(
            {
                "local_task_index": int(local_index),
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
        "offer_key": offer_key,
        "dag_id": dag_id,
        "ue_id": int(ue_id),
        "arrival_time": float(slot_index + 1) * float(config.TIME_SLOT_DURATION),
        "source_pos": [float(source_pos[0]), float(source_pos[1])],
        "base_upload_bandwidth_mbps": float(job.base_upload_bandwidth_mbps),
        "base_download_bandwidth_mbps": float(job.base_download_bandwidth_mbps),
        "task_ids": [old_to_new[value] for value in job.task_ids],
        "sink_task_ids": [old_to_new[value] for value in job.sink_task_ids],
        "khop_hyperedges_global": [
            [old_to_new[value] for value in edge]
            for edge in job.khop_hyperedges_global
        ],
        "tasks": tasks,
    }


def _episode_checksums(episode_payload: dict[str, Any]) -> dict[str, str]:
    return {
        "scenario_checksum": canonical_sha256(episode_payload["scenario"]),
        "offered_event_checksum": canonical_sha256(
            [
                episode_payload["scenario_seed"],
                episode_payload["episode"],
                episode_payload["offer_bits_by_slot"],
            ]
        ),
        "offered_template_checksum": canonical_sha256(episode_payload["templates"]),
    }


def _tape_checksum_payload(tape: dict[str, Any], episode_limit: int | None = None) -> dict[str, Any]:
    selected = tape["episodes"]
    if episode_limit is not None:
        selected = [row for row in selected if int(row["episode"]) < int(episode_limit)]
    return {
        "schema": tape["schema"],
        "controls": tape["controls"],
        "episodes": selected,
    }


def generate_scenario_tape(
    *,
    scenario_seeds: Iterable[int] = FORMAL_SCENARIO_SEEDS,
    episodes: int = FORMAL_EPISODES,
    load_slots: int = LOAD_SLOTS,
    episode_slots: int = EPISODE_SLOTS,
    num_ues: int | None = None,
    num_uavs: int | None = None,
) -> dict[str, Any]:
    resolved_ues = int(config.NUM_UES if num_ues is None else num_ues)
    resolved_uavs = int(config.NUM_UAVS if num_uavs is None else num_uavs)
    seeds = tuple(int(seed) for seed in scenario_seeds)
    if not seeds or int(episodes) <= 0 or int(load_slots) <= 0:
        raise ValueError("tape dimensions must be positive")
    if int(episode_slots) < int(load_slots):
        raise ValueError("episode_slots must be at least load_slots")
    tape: dict[str, Any] = {
        "schema": TAPE_SCHEMA,
        "controls": {
            "scenario_seeds": list(seeds),
            "episodes": int(episodes),
            "load_slots": int(load_slots),
            "episode_slots": int(episode_slots),
            "num_ues": resolved_ues,
            "num_uavs": resolved_uavs,
            "time_slot_duration": float(config.TIME_SLOT_DURATION),
            "dag_base_arrival_prob": float(config.DAG_BASE_ARRIVAL_PROB),
            "dag_hotspot_arrival_multiplier": float(config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER),
        },
        "episodes": [],
    }
    for scenario_seed in seeds:
        for episode in range(int(episodes)):
            scenario = _scenario_payload(scenario_seed, episode, resolved_ues, resolved_uavs)
            bits_by_slot: list[str] = []
            templates: list[dict[str, Any]] = []
            offered_by_ue = [0 for _ in range(resolved_ues)]
            for slot_index in range(int(load_slots)):
                bits: list[str] = []
                for ue_id in range(resolved_ues):
                    offer_key = [scenario_seed, episode, slot_index, ue_id]
                    probability = _arrival_probability(scenario["ue_positions"][ue_id], scenario)
                    offered = _keyed_uniform("offered_event", offer_key) < probability
                    bits.append("1" if offered else "0")
                    if not offered:
                        continue
                    offered_by_ue[ue_id] += 1
                    templates.append(
                        _template_from_current_generator(
                            scenario_seed=scenario_seed,
                            episode=episode,
                            slot_index=slot_index,
                            ue_id=ue_id,
                            source_pos=scenario["ue_positions"][ue_id],
                        )
                    )
                bits_by_slot.append("".join(bits))
            offered_subtasks = sum(len(template["tasks"]) for template in templates)
            row: dict[str, Any] = {
                "scenario_seed": scenario_seed,
                "episode": episode,
                "load_slots": int(load_slots),
                "episode_slots": int(episode_slots),
                "scenario": scenario,
                "offer_bits_by_slot": bits_by_slot,
                "templates": templates,
                "arrival_opportunity_count": int(load_slots) * resolved_ues,
                "offered_dag_count": len(templates),
                "offered_subtask_count": int(offered_subtasks),
                "active_nonbinding_cap": max(offered_by_ue, default=0) + 1,
                "queue_nonbinding_cap": int(offered_subtasks) + 1,
            }
            row.update(_episode_checksums(row))
            tape["episodes"].append(row)
    tape["full_tape_checksum"] = canonical_sha256(_tape_checksum_payload(tape))
    prefix_limit = min(PILOT_EPISODES, int(episodes))
    tape["pilot_prefix_episode_count"] = prefix_limit
    tape["pilot_prefix_checksum"] = canonical_sha256(_tape_checksum_payload(tape, prefix_limit))
    validate_scenario_tape(tape)
    return tape


def save_scenario_tape_create_only(tape: dict[str, Any], path: Path) -> None:
    validate_scenario_tape(tape)
    if path.exists():
        existing = load_scenario_tape(path)
        if existing["full_tape_checksum"] != tape["full_tape_checksum"]:
            raise FileExistsError(f"refusing to overwrite different tape: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tape, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8")


def load_scenario_tape(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_scenario_tape(payload)
    return payload


def validate_scenario_tape(tape: dict[str, Any]) -> None:
    if tape.get("schema") != TAPE_SCHEMA:
        raise ValueError("unsupported factorial tape schema")
    controls = tape.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("tape controls are missing")
    episodes = tape.get("episodes")
    expected_rows = len(controls["scenario_seeds"]) * int(controls["episodes"])
    if not isinstance(episodes, list) or len(episodes) != expected_rows:
        raise ValueError("tape episode count mismatch")
    seen_episode_keys: set[tuple[int, int]] = set()
    seen_dag_ids: set[str] = set()
    seen_task_ids: set[str] = set()
    num_ues = int(controls["num_ues"])
    load_slots = int(controls["load_slots"])
    for row in episodes:
        key = (int(row["scenario_seed"]), int(row["episode"]))
        if key in seen_episode_keys:
            raise ValueError(f"duplicate tape episode {key}")
        seen_episode_keys.add(key)
        if len(row["offer_bits_by_slot"]) != load_slots:
            raise ValueError(f"offer slot count mismatch for {key}")
        offered_count = 0
        offered_by_ue = [0 for _ in range(num_ues)]
        for bit_string in row["offer_bits_by_slot"]:
            if len(bit_string) != num_ues or set(bit_string) - {"0", "1"}:
                raise ValueError(f"invalid offer bit string for {key}")
            offered_count += bit_string.count("1")
            for ue_id, bit in enumerate(bit_string):
                offered_by_ue[ue_id] += int(bit == "1")
        if offered_count != int(row["offered_dag_count"]) or offered_count != len(row["templates"]):
            raise ValueError(f"offered DAG count mismatch for {key}")
        subtask_count = 0
        for template in row["templates"]:
            expected_dag_id = stable_dag_id(*[int(value) for value in template["offer_key"]])
            if template["dag_id"] != expected_dag_id or template["dag_id"] in seen_dag_ids:
                raise ValueError("unstable or duplicate diagnostic DAG ID")
            seen_dag_ids.add(template["dag_id"])
            if json.loads(canonical_json_bytes(template).decode("utf-8")) != template:
                raise ValueError("template canonical round-trip mismatch")
            for local_index, task in enumerate(template["tasks"]):
                expected_task_id = stable_task_id(template["dag_id"], local_index)
                if task["local_task_index"] != local_index or task["task_id"] != expected_task_id:
                    raise ValueError("unstable task-local index or ID")
                if task["task_id"] in seen_task_ids:
                    raise ValueError("duplicate diagnostic task ID")
                seen_task_ids.add(task["task_id"])
            subtask_count += len(template["tasks"])
        if int(row["offered_subtask_count"]) != subtask_count:
            raise ValueError(f"offered subtask count mismatch for {key}")
        if int(row["arrival_opportunity_count"]) != load_slots * num_ues:
            raise ValueError(f"arrival opportunity count mismatch for {key}")
        if int(row["active_nonbinding_cap"]) != max(offered_by_ue, default=0) + 1:
            raise ValueError(f"active nonbinding cap mismatch for {key}")
        if int(row["queue_nonbinding_cap"]) != subtask_count + 1:
            raise ValueError(f"queue nonbinding cap mismatch for {key}")
        if _episode_checksums(row) != {
            name: row[name]
            for name in ("scenario_checksum", "offered_event_checksum", "offered_template_checksum")
        }:
            raise ValueError(f"episode checksum mismatch for {key}")
    if tape.get("full_tape_checksum") != canonical_sha256(_tape_checksum_payload(tape)):
        raise ValueError("full tape checksum mismatch")
    prefix_limit = int(tape.get("pilot_prefix_episode_count", 0))
    if tape.get("pilot_prefix_checksum") != canonical_sha256(_tape_checksum_payload(tape, prefix_limit)):
        raise ValueError("pilot prefix checksum mismatch")


def index_tape_episodes(tape: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row["scenario_seed"]), int(row["episode"])): row
        for row in tape["episodes"]
    }


def index_episode_templates(episode_payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(template["offer_key"][2]), int(template["offer_key"][3])): template
        for template in episode_payload["templates"]
    }


def instantiate_dag_template(manager: DAGTaskManager, template: dict[str, Any]) -> DAGJob:
    ue_id = int(template["ue_id"])
    if not manager.can_accept_dag_for_ue(ue_id):
        raise ValueError("active DAG cap rejected immutable template")
    dag_id = str(template["dag_id"])
    if manager.get_job(dag_id) is not None:
        raise ValueError(f"duplicate DAG instantiation: {dag_id}")
    source = np.asarray(template["source_pos"], dtype=np.float32).reshape(-1)[:2].copy()
    tasks: list[TaskNode] = []
    for task_payload in template["tasks"]:
        task = TaskNode(
            task_id=str(task_payload["task_id"]),
            dag_id=dag_id,
            ue_id=ue_id,
            input_data_size_mb=float(task_payload["input_data_size_mb"]),
            output_data_size_mb=float(task_payload["output_data_size_mb"]),
            task_complexity=str(task_payload["task_complexity"]),
            task_constant=int(task_payload["task_constant"]),
            num_operation=float(task_payload["num_operation"]),
            level=int(task_payload["level"]),
            source_pos=source.copy(),
            arrival_time=float(template["arrival_time"]),
            topological_index=int(task_payload["topological_index"]),
            predecessors=[str(value) for value in task_payload["predecessors"]],
            successors=[str(value) for value in task_payload["successors"]],
            is_critical_path=bool(task_payload["is_critical_path"]),
        )
        tasks.append(task)
    job = DAGJob(
        dag_id=dag_id,
        ue_id=ue_id,
        arrival_time=float(template["arrival_time"]),
        source_pos=source.copy(),
        base_upload_bandwidth_mbps=float(template["base_upload_bandwidth_mbps"]),
        base_download_bandwidth_mbps=float(template["base_download_bandwidth_mbps"]),
        task_ids=[str(value) for value in template["task_ids"]],
        sink_task_ids=[str(value) for value in template["sink_task_ids"]],
        khop_hyperedges_global=[
            [str(value) for value in edge]
            for edge in template["khop_hyperedges_global"]
        ],
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


class FactorialDiagnosticEnv(Env):
    """Clean environment adapter that consumes one immutable tape episode."""

    def __init__(self, *, episode_payload: dict[str, Any], cell: str) -> None:
        normalized_cell = str(cell).upper()
        if normalized_cell not in CELL_FLAGS:
            raise ValueError(f"unknown factorial cell: {cell}")
        active_nonbinding, queue_nonbinding = CELL_FLAGS[normalized_cell]
        active_cap = (
            int(episode_payload["active_nonbinding_cap"])
            if active_nonbinding
            else BASELINE_ACTIVE_CAP
        )
        queue_cap = (
            int(episode_payload["queue_nonbinding_cap"])
            if queue_nonbinding
            else BASELINE_QUEUE_CAP
        )
        self.episode_payload = episode_payload
        self.cell = normalized_cell
        self.capacity_context = DiagnosticCapacityContext(hard_queue_cap=queue_cap)
        self.templates_by_key = index_episode_templates(episode_payload)
        self.diagnostic_arrival_totals: dict[str, int] = {}
        super().__init__(
            completed_dag_weight=16.0,
            freeze_ue_mobility=True,
            max_active_dags_per_ue=active_cap,
            diagnostic_capacity_context=self.capacity_context,
        )

    def reset(self) -> list[np.ndarray]:
        observations = super().reset()
        scenario = self.episode_payload["scenario"]
        self.hotspot_center = np.asarray(scenario["hotspot_center"], dtype=np.float32)
        self.hotspot_radius = float(scenario["hotspot_radius"])
        if len(self.ues) != len(scenario["ue_positions"]) or len(self.uavs) != len(scenario["uav_positions"]):
            raise ValueError("tape entity count does not match environment")
        for ue, position in zip(self.ues, scenario["ue_positions"]):
            ue.pos[:2] = np.asarray(position, dtype=np.float32)
            ue.speed = 0.0
            ue.velocity = np.zeros((2,), dtype=np.float32)
        for uav, position in zip(self.uavs, scenario["uav_positions"]):
            uav.pos[:2] = np.asarray(position, dtype=np.float32)
        self._ue_service_positions = {int(ue.id): ue.pos[:2].copy() for ue in self.ues}
        self._uav_pre_move_positions = {int(uav.id): uav.pos[:2].copy() for uav in self.uavs}
        self._uav_service_positions = {int(uav.id): uav.pos[:2].copy() for uav in self.uavs}
        self._initial_hotspot_ue_count = sum(
            int(ue.is_inside_hotspot(self.hotspot_center, self.hotspot_radius))
            for ue in self.ues
        )
        self.diagnostic_arrival_totals = {
            "arrival_opportunity_count": 0,
            "offered_dag_count": 0,
            "offered_subtask_count": 0,
            "active_cap_blocked_offered_count": 0,
            "admitted_dag_count": 0,
            "admitted_subtask_count": 0,
            "no_arrival_event_count": 0,
        }
        return observations

    def _process_clean_dag_arrivals(self) -> int:
        slot_index = int(self._time_step) - 1
        version_before = self.task_manager.dag_arrival_version
        created_count = 0
        funnel = self._empty_arrival_funnel()
        if slot_index < int(self.episode_payload.get("load_slots", LOAD_SLOTS)):
            bit_string = self.episode_payload["offer_bits_by_slot"][slot_index]
            for ue in self.ues:
                ue_id = int(ue.id)
                self.diagnostic_arrival_totals["arrival_opportunity_count"] += 1
                funnel["arrival_attempt_count"] += 1
                if bit_string[ue_id] == "0":
                    self.diagnostic_arrival_totals["no_arrival_event_count"] += 1
                    funnel["arrival_no_event_count"] += 1
                    continue
                template = self.templates_by_key[(slot_index, ue_id)]
                self.diagnostic_arrival_totals["offered_dag_count"] += 1
                self.diagnostic_arrival_totals["offered_subtask_count"] += len(template["tasks"])
                funnel["arrival_draw_count"] += 1
                funnel["arrival_sampled_event_count"] += 1
                if not self.task_manager.can_accept_dag_for_ue(ue_id):
                    self.diagnostic_arrival_totals["active_cap_blocked_offered_count"] += 1
                    funnel["arrival_blocked_count"] += 1
                    funnel["arrival_blocked_reasons"]["active_dag_cap"] += 1
                    continue
                job = instantiate_dag_template(self.task_manager, template)
                ue.enter_service_waiting(job.dag_id)
                created_count += 1
                self.diagnostic_arrival_totals["admitted_dag_count"] += 1
                self.diagnostic_arrival_totals["admitted_subtask_count"] += len(template["tasks"])
                funnel["arrival_admitted_count"] += 1
        version_after = self.task_manager.dag_arrival_version
        self._latest_arrival_funnel = funnel
        self._accumulate_arrival_funnel(funnel)
        self._last_new_dag_arrived = version_after > version_before
        self._latest_dag_arrival_version = version_after
        return created_count


def candidate_legality_reasons(
    *,
    task: TaskNode | None,
    uav_id: int,
    reservation: TemporaryReservationState,
    valid_uav_ids: set[int],
    executor: Any,
    capacity_context: DiagnosticCapacityContext,
) -> frozenset[str]:
    reasons: set[str] = set()
    if task is None or not task.is_ready:
        reasons.add("task_not_ready")
    if int(uav_id) not in valid_uav_ids:
        reasons.add("invalid_uav")
    if task is not None and str(task.task_id) in reservation.reserved_task_ids:
        reasons.add("already_reserved")
    if task is not None and hasattr(executor, "is_task_scheduled") and executor.is_task_scheduled(task.task_id):
        reasons.add("already_scheduled")
    if reservation.remaining_slots(int(uav_id), capacity_context) <= 0:
        reasons.add("queue_full")
    return frozenset(reasons)


def primary_reason(reasons: Iterable[str]) -> str:
    normalized = set(str(value) for value in reasons)
    for reason in REASON_PRECEDENCE:
        if reason in normalized:
            return reason
    return "other"


@dataclass(slots=True)
class FactorialEpisodeTracker:
    uav_ids: tuple[int, ...]
    candidate_mask_reason_count: Counter[str] = field(default_factory=Counter)
    candidate_mask_reason_count_by_uav: dict[int, Counter[str]] = field(default_factory=dict)
    skip_reason_count: Counter[str] = field(default_factory=Counter)
    skip_reason_signature_count: Counter[str] = field(default_factory=Counter)
    max_executor_queue_length_by_uav: dict[int, int] = field(default_factory=dict)
    max_temporary_queue_length_by_uav: dict[int, int] = field(default_factory=dict)
    executor_queue_at_16_observation_count_by_uav: dict[int, int] = field(default_factory=dict)
    temporary_queue_at_16_observation_count_by_uav: dict[int, int] = field(default_factory=dict)
    queue_full_mask_count_by_uav: dict[int, int] = field(default_factory=dict)
    same_slot_queue_reached_hard_cap_count_by_uav: dict[int, int] = field(default_factory=dict)
    all_uavs_full_decision_count: int = 0
    ready_task_attempts: Counter[str] = field(default_factory=Counter)
    ready_decision_attempt_count: int = 0
    choice_decision_count: int = 0
    forced_decision_count: int = 0
    skip_decision_count: int = 0
    assignment_legality_records: list[dict[str, Any]] = field(default_factory=list)
    invalid_assignment_reasons: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        ids = tuple(sorted(int(value) for value in self.uav_ids))
        self.uav_ids = ids
        self.candidate_mask_reason_count_by_uav = {value: Counter() for value in ids}
        self.max_executor_queue_length_by_uav = {value: 0 for value in ids}
        self.max_temporary_queue_length_by_uav = {value: 0 for value in ids}
        self.executor_queue_at_16_observation_count_by_uav = {value: 0 for value in ids}
        self.temporary_queue_at_16_observation_count_by_uav = {value: 0 for value in ids}
        self.queue_full_mask_count_by_uav = {value: 0 for value in ids}
        self.same_slot_queue_reached_hard_cap_count_by_uav = {value: 0 for value in ids}

    def observe_executor_queues(self, executor: Any, *, count_at_16: bool = True) -> None:
        for uav_id in self.uav_ids:
            length = len(getattr(executor, "uav_queues", {}).get(uav_id, []))
            self.max_executor_queue_length_by_uav[uav_id] = max(
                self.max_executor_queue_length_by_uav[uav_id], length
            )
            if count_at_16 and length == BASELINE_QUEUE_CAP:
                self.executor_queue_at_16_observation_count_by_uav[uav_id] += 1

    def observe_temporary_reservation(
        self,
        reservation: TemporaryReservationState,
        *,
        count_at_16: bool = False,
    ) -> None:
        for uav_id in self.uav_ids:
            queue_length = int(reservation.queue_lengths.get(uav_id, 0))
            self.max_temporary_queue_length_by_uav[uav_id] = max(
                self.max_temporary_queue_length_by_uav[uav_id], queue_length
            )
            if count_at_16 and queue_length == BASELINE_QUEUE_CAP:
                self.temporary_queue_at_16_observation_count_by_uav[uav_id] += 1

    def observe_decision(
        self,
        *,
        task_id: str,
        reservation: TemporaryReservationState,
        reasons_by_uav: dict[int, frozenset[str]],
        legal_uav_ids: list[int],
    ) -> None:
        normalized_task_id = str(task_id)
        self.ready_task_attempts[normalized_task_id] += 1
        self.ready_decision_attempt_count += 1
        self.observe_temporary_reservation(reservation, count_at_16=True)
        for uav_id in self.uav_ids:
            reasons = reasons_by_uav.get(uav_id, frozenset({"other"}))
            for reason in reasons:
                self.candidate_mask_reason_count[str(reason)] += 1
                self.candidate_mask_reason_count_by_uav[uav_id][str(reason)] += 1
                if reason == "queue_full":
                    self.queue_full_mask_count_by_uav[uav_id] += 1
        legal_count = len(legal_uav_ids)
        if legal_count >= 2:
            self.choice_decision_count += 1
            return
        if legal_count == 1:
            self.forced_decision_count += 1
            return
        self.skip_decision_count += 1
        union = set().union(*(set(value) for value in reasons_by_uav.values()))
        primary = primary_reason(union)
        signature = "+".join(sorted(union)) if union else "other"
        self.skip_reason_count[primary] += 1
        self.skip_reason_signature_count[signature] += 1
        task_consistency = {"task_not_ready", "already_reserved", "already_scheduled"}
        all_queue_full = bool(reasons_by_uav) and all(
            "queue_full" in reasons_by_uav.get(uav_id, frozenset())
            for uav_id in self.uav_ids
        )
        if all_queue_full and not (union & task_consistency):
            self.all_uavs_full_decision_count += 1

    def record_assignment(self, *, slot_index: int, task_id: str, uav_id: int) -> int:
        self.assignment_legality_records.append(
            {
                "slot_index": int(slot_index),
                "physical_slot": int(slot_index) + 1,
                "task_id": str(task_id),
                "uav_id": int(uav_id),
                "decision_time_legal": True,
                "commit_time_legal": None,
            }
        )
        return len(self.assignment_legality_records) - 1

    def record_commit(self, indices: list[int], executor: Any) -> None:
        records = getattr(executor, "task_records", {})
        for index in indices:
            row = self.assignment_legality_records[index]
            committed = str(row["task_id"]) in records
            row["commit_time_legal"] = bool(committed)
            if not committed:
                self.invalid_assignment_reasons["queue_cap_mismatch"] += 1

    def record_executor_invalid(self, reasons: dict[str, int]) -> None:
        for reason, count in reasons.items():
            self.invalid_assignment_reasons[str(reason)] += int(count)

    def to_metrics(self) -> dict[str, Any]:
        repeated = sum(max(count - 1, 0) for count in self.ready_task_attempts.values())
        choice_fraction = (
            None
            if self.ready_decision_attempt_count == 0
            else float(self.choice_decision_count) / float(self.ready_decision_attempt_count)
        )
        return {
            "max_executor_queue_length_by_uav": _string_key_dict(self.max_executor_queue_length_by_uav),
            "max_temporary_queue_length_by_uav": _string_key_dict(self.max_temporary_queue_length_by_uav),
            "executor_queue_at_16_observation_count_by_uav": _string_key_dict(
                self.executor_queue_at_16_observation_count_by_uav
            ),
            "temporary_queue_at_16_observation_count_by_uav": _string_key_dict(
                self.temporary_queue_at_16_observation_count_by_uav
            ),
            "queue_full_mask_count_by_uav": _string_key_dict(self.queue_full_mask_count_by_uav),
            "same_slot_queue_reached_hard_cap_count_by_uav": _string_key_dict(
                self.same_slot_queue_reached_hard_cap_count_by_uav
            ),
            "queue_full_mask_count": int(sum(self.queue_full_mask_count_by_uav.values())),
            "all_uavs_full_decision_count": int(self.all_uavs_full_decision_count),
            "unique_ready_task_count": len(self.ready_task_attempts),
            "ready_decision_attempt_count": int(self.ready_decision_attempt_count),
            "repeated_ready_attempt_count": int(repeated),
            "choice_decision_count": int(self.choice_decision_count),
            "forced_decision_count": int(self.forced_decision_count),
            "skip_decision_count": int(self.skip_decision_count),
            "choice_decision_fraction": choice_fraction,
            "candidate_mask_reason_count": dict(sorted(self.candidate_mask_reason_count.items())),
            "candidate_mask_reason_count_by_uav": {
                str(uav_id): dict(sorted(counter.items()))
                for uav_id, counter in sorted(self.candidate_mask_reason_count_by_uav.items())
            },
            "skip_reason_count": dict(sorted(self.skip_reason_count.items())),
            "skip_reason_signature_count": dict(sorted(self.skip_reason_signature_count.items())),
            "invalid_assignment_reasons": dict(sorted(self.invalid_assignment_reasons.items())),
            "assignment_legality_records": self.assignment_legality_records,
        }


def _string_key_dict(values: dict[int, Any]) -> dict[str, Any]:
    return {str(key): values[key] for key in sorted(values)}


@dataclass(slots=True)
class FrozenActorPolicy:
    encoder: Any
    scorer: Any
    device: Any
    checkpoint_metadata: dict[str, Any]


def load_stage1_actor_policy(*, root: Path, scenario_seed: int, device: str) -> FrozenActorPolicy:
    import torch

    from marl_models.hgnn import build_clean_task_encoder
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from scripts.train_decision_ppo_bandit_gate import CHECKPOINT_SCHEMA

    if int(scenario_seed) not in STAGE1_ACTOR_CHECKPOINTS:
        raise ValueError(f"no frozen Stage 1 checkpoint for scenario seed {scenario_seed}")
    relative_path = STAGE1_ACTOR_CHECKPOINTS[int(scenario_seed)]
    path = (root / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checksum = sha256_file(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Stage 1 actor checkpoint schema mismatch")
    if str(payload.get("group")) != "S1-B":
        raise ValueError("Stage 1 actor checkpoint must be S1-B")
    if int(payload.get("seed")) != int(scenario_seed):
        raise ValueError("Stage 1 actor checkpoint training seed mismatch")
    if int(payload.get("completed_update")) != 30:
        raise ValueError("Stage 1 actor checkpoint must be update 30")
    model_config = payload.get("config")
    if not isinstance(model_config, dict) or str(model_config.get("encoder")) != "mlp":
        raise ValueError("Stage 1 actor checkpoint must contain an MLP config")
    encoder_state = payload.get("encoder_state_dict")
    if not isinstance(encoder_state, dict) or "input_proj.weight" not in encoder_state:
        raise ValueError("Stage 1 actor checkpoint lacks encoder input_proj.weight")
    input_weight = encoder_state["input_proj.weight"]
    if getattr(input_weight, "ndim", None) != 2:
        raise ValueError("Stage 1 actor checkpoint input_proj.weight must be rank two")
    checkpoint_task_feature_dim = int(input_weight.shape[1])
    graph_snapshot_task_feature_dim = _probe_graph_snapshot_task_feature_dim(torch)
    if checkpoint_task_feature_dim != graph_snapshot_task_feature_dim:
        raise ValueError(
            "Stage 1 actor task feature dimension mismatch: "
            f"checkpoint_dim={checkpoint_task_feature_dim}, "
            f"graph_snapshot_dim={graph_snapshot_task_feature_dim}"
        )
    resolved_task_feature_dim = checkpoint_task_feature_dim
    resolved_device = torch.device(str(device))
    encoder = build_clean_task_encoder(
        encoder_type="mlp",
        task_feature_dim=resolved_task_feature_dim,
        hidden_dim=int(model_config["hidden_dim"]),
        output_dim=int(model_config["task_embedding_dim"]),
    ).to(resolved_device)
    actor = CleanOffloadingActor(
        task_embedding_dim=int(model_config["task_embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
    ).to(resolved_device)
    encoder.load_state_dict(encoder_state, strict=True)
    actor.scorer.load_state_dict(payload["scorer_state_dict"], strict=True)
    encoder.eval()
    actor.scorer.eval()
    return FrozenActorPolicy(
        encoder=encoder,
        scorer=actor.scorer,
        device=resolved_device,
        checkpoint_metadata={
            "actor_checkpoint_path": relative_path,
            "actor_checkpoint_sha256": checksum,
            "actor_training_seed": int(payload["seed"]),
            "actor_completed_update": int(payload["completed_update"]),
            "checkpoint_task_feature_dim": checkpoint_task_feature_dim,
            "graph_snapshot_task_feature_dim": graph_snapshot_task_feature_dim,
            "resolved_task_feature_dim": resolved_task_feature_dim,
            "encoder_strict_load_pass": True,
        },
    )


def _probe_graph_snapshot_task_feature_dim(torch: Any) -> int:
    """Probe the current clean graph schema without perturbing caller RNG state."""
    from environment.graph_builder import CleanGraphBuilder
    from environment.env import Env
    from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    builder = CleanGraphBuilder()
    try:
        probe_env = Env(completed_dag_weight=16.0, freeze_ue_mobility=True)
        probe_env.reset()
        builder.reset()
        probe = prepare_slot_state(env=probe_env, graph_builder=builder)
        task_features = np.asarray(probe.graph_snapshot.task_features)
        if task_features.ndim != 2 or int(task_features.shape[1]) <= 0:
            raise ValueError("probe GraphSnapshot has an invalid task feature shape")
        return int(task_features.shape[1])
    finally:
        builder.close()
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def run_factorial_episode(
    *,
    episode_payload: dict[str, Any],
    cell: str,
    policy: str,
    full_tape_checksum: str,
    pilot_prefix_checksum: str,
    actor_policy: FrozenActorPolicy | None = None,
) -> dict[str, Any]:
    from environment.graph_builder import CleanGraphBuilder
    from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state

    normalized_policy = str(policy)
    if normalized_policy not in {"random_hash", "greedy_eft", "stage1_actor"}:
        raise ValueError(f"unknown fixed policy: {policy}")
    if normalized_policy == "stage1_actor" and actor_policy is None:
        raise ValueError("stage1_actor policy requires a loaded checkpoint")
    env = FactorialDiagnosticEnv(episode_payload=episode_payload, cell=cell)
    builder = CleanGraphBuilder()
    env.reset()
    builder.reset()
    initial_ue_positions = [np.asarray(value, dtype=np.float32) for value in episode_payload["scenario"]["ue_positions"]]
    initial_uav_positions = [np.asarray(value, dtype=np.float32) for value in episode_payload["scenario"]["uav_positions"]]
    tracker = FactorialEpisodeTracker(tuple(int(uav.id) for uav in env.uavs))
    reward_total = 0.0
    latest_info: dict[str, Any] = {}
    finite = True
    try:
        for expected_slot in range(EPISODE_SLOTS):
            tracker.observe_executor_queues(env.executor)
            prepared = prepare_slot_state(env=env, graph_builder=builder)
            if int(prepared.slot_index) != expected_slot:
                raise AssertionError("machine-readable slot_index is not zero-based and contiguous")
            env.apply_movement({})
            _assert_frozen_positions(env, initial_ue_positions, initial_uav_positions)
            embeddings = None
            if actor_policy is not None:
                import torch

                task_features = torch.as_tensor(
                    np.asarray(prepared.graph_snapshot.task_features, dtype=np.float32).copy(),
                    dtype=torch.float32,
                    device=actor_policy.device,
                )
                with torch.no_grad():
                    embeddings = actor_policy.encoder(task_features)
            reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
            assignments = CleanAssignmentBuffer()
            commit_record_indices: list[int] = []
            valid_uav_ids = {int(uav.id) for uav in env.uavs}
            ready_tasks = [env.task_manager.get_task(task_id) for task_id in prepared.frozen_ready_task_ids]
            for decision_order, task in enumerate(ready_tasks):
                if task is None:
                    continue
                dynamic, pair, mask, candidate_uav_ids, estimates = build_offloading_candidate_components(
                    task=task,
                    uavs=env.uavs,
                    task_manager=env.task_manager,
                    executor=env.executor,
                    state_view=reservation,
                    current_time_seconds=env.current_time_seconds,
                    uav_service_positions=env.uav_service_positions,
                    ue_service_positions=env.ue_service_positions,
                    ues=env.ues,
                    capacity_context=env.capacity_context,
                )
                reasons_by_uav: dict[int, frozenset[str]] = {}
                for candidate_index, uav_id in enumerate(candidate_uav_ids):
                    reasons = set(
                        candidate_legality_reasons(
                            task=task,
                            uav_id=uav_id,
                            reservation=reservation,
                            valid_uav_ids=valid_uav_ids,
                            executor=env.executor,
                            capacity_context=env.capacity_context,
                        )
                    )
                    if bool(mask[candidate_index]) is False and not reasons:
                        reasons.add("other")
                    if bool(mask[candidate_index]) and reasons:
                        raise AssertionError("legal candidate retained an explicit illegality reason")
                    reasons_by_uav[int(uav_id)] = frozenset(reasons)
                legal_indices = [index for index, legal in enumerate(mask.tolist()) if bool(legal)]
                legal_ids = [int(candidate_uav_ids[index]) for index in legal_indices]
                tracker.observe_decision(
                    task_id=task.task_id,
                    reservation=reservation,
                    reasons_by_uav=reasons_by_uav,
                    legal_uav_ids=legal_ids,
                )
                if not legal_indices:
                    continue
                if normalized_policy == "random_hash":
                    selected_uav_id = random_hash_uav(
                        scenario_seed=int(episode_payload["scenario_seed"]),
                        episode=int(episode_payload["episode"]),
                        slot_index=int(prepared.slot_index),
                        stable_task_id_value=str(task.task_id),
                        legal_uav_ids=legal_ids,
                    )
                    selected_index = candidate_uav_ids.index(selected_uav_id)
                elif normalized_policy == "greedy_eft":
                    selected_index = min(
                        legal_indices,
                        key=lambda index: (float(estimates[index].estimated_finish_time), int(candidate_uav_ids[index])),
                    )
                    selected_uav_id = int(candidate_uav_ids[selected_index])
                else:
                    selected_index = _select_stage1_actor_action(
                        actor_policy=actor_policy,
                        task=task,
                        graph_snapshot=prepared.graph_snapshot,
                        embeddings=embeddings,
                        dynamic=dynamic,
                        pair=pair,
                        mask=mask,
                    )
                    selected_uav_id = int(candidate_uav_ids[selected_index])
                if selected_index not in legal_indices:
                    raise AssertionError("fixed policy selected an illegal action")
                selected_estimate = estimates[selected_index]
                assignments.append(task.task_id, selected_uav_id, decision_order)
                queue_before = int(reservation.queue_lengths.get(selected_uav_id, 0))
                reservation.reserve(
                    task.task_id,
                    selected_uav_id,
                    estimated_available_time=selected_estimate.estimated_finish_time,
                    estimated_queued_workload=selected_estimate.estimated_queued_workload,
                )
                tracker.observe_temporary_reservation(reservation)
                queue_after = int(reservation.queue_lengths.get(selected_uav_id, 0))
                if queue_before < int(env.capacity_context.hard_queue_cap) <= queue_after:
                    tracker.same_slot_queue_reached_hard_cap_count_by_uav[selected_uav_id] += 1
                commit_record_indices.append(
                    tracker.record_assignment(
                        slot_index=prepared.slot_index,
                        task_id=task.task_id,
                        uav_id=selected_uav_id,
                    )
                )
            _, _, _, latest_info = env.commit_and_advance(assignment_buffer=assignments)
            reward_total += float(latest_info["step_reward"])
            tracker.record_commit(commit_record_indices, env.executor)
            tracker.record_executor_invalid(latest_info.get("invalid_assignment_reasons", {}))
            tracker.observe_executor_queues(env.executor, count_at_16=False)
            finite = finite and _all_finite(latest_info)
    finally:
        builder.close()
    row = _build_episode_result(
        env=env,
        tracker=tracker,
        episode_payload=episode_payload,
        policy=normalized_policy,
        full_tape_checksum=full_tape_checksum,
        pilot_prefix_checksum=pilot_prefix_checksum,
        reward_total=reward_total,
        latest_info=latest_info,
        finite=finite,
    )
    if actor_policy is not None:
        row.update(actor_policy.checkpoint_metadata)
    return row


def _select_stage1_actor_action(
    *,
    actor_policy: FrozenActorPolicy | None,
    task: TaskNode,
    graph_snapshot: Any,
    embeddings: Any,
    dynamic: np.ndarray,
    pair: np.ndarray,
    mask: np.ndarray,
) -> int:
    import torch

    if actor_policy is None or embeddings is None:
        raise ValueError("actor policy is not loaded")
    task_index = graph_snapshot.task_id_to_idx.get(task.task_id)
    if task_index is None:
        raise ValueError(f"Stage 1 actor task is missing from GraphSnapshot: {task.task_id}")
    task_embedding = embeddings[int(task_index)].reshape(1, -1)
    repeated = task_embedding.repeat(int(dynamic.shape[0]), 1)
    features = torch.cat(
        [
            repeated,
            torch.as_tensor(dynamic, dtype=torch.float32, device=actor_policy.device),
            torch.as_tensor(pair, dtype=torch.float32, device=actor_policy.device),
        ],
        dim=1,
    )
    if int(dynamic.shape[1]) != 7:
        raise AssertionError("diagnostic changed the seven-dimensional actor UAV input")
    with torch.no_grad():
        logits = actor_policy.scorer(features)
    torch_mask = torch.as_tensor(mask, dtype=torch.bool, device=actor_policy.device)
    masked = logits.masked_fill(~torch_mask, torch.finfo(logits.dtype).min)
    return int(torch.argmax(masked).item())


def _assert_frozen_positions(
    env: FactorialDiagnosticEnv,
    initial_ue_positions: list[np.ndarray],
    initial_uav_positions: list[np.ndarray],
) -> None:
    for ue, expected in zip(env.ues, initial_ue_positions):
        if not np.array_equal(np.asarray(ue.pos[:2], dtype=np.float32), expected):
            raise AssertionError("UE position changed during frozen diagnostic episode")
    for uav, expected in zip(env.uavs, initial_uav_positions):
        if not np.array_equal(np.asarray(uav.pos[:2], dtype=np.float32), expected):
            raise AssertionError("UAV position changed during hover diagnostic episode")


def _build_episode_result(
    *,
    env: FactorialDiagnosticEnv,
    tracker: FactorialEpisodeTracker,
    episode_payload: dict[str, Any],
    policy: str,
    full_tape_checksum: str,
    pilot_prefix_checksum: str,
    reward_total: float,
    latest_info: dict[str, Any],
    finite: bool,
) -> dict[str, Any]:
    completed = sum(int(job.completed) for job in env.task_manager.jobs.values())
    admitted = int(env.diagnostic_arrival_totals["admitted_dag_count"])
    offered = int(env.diagnostic_arrival_totals["offered_dag_count"])
    flowtimes = [float(value) for value in env.metrics.metrics.dag_flowtimes]
    incomplete = sum(int(not job.completed) for job in env.task_manager.jobs.values())
    ready_count = len(env.task_manager.get_ready_tasks())
    queued_task_count = sum(
        int(task.state == TASK_STATE_IN_SERVICE)
        for task in env.task_manager.tasks.values()
    )
    executor_queues = {
        str(uav_id): len(env.executor.uav_queues.get(uav_id, []))
        for uav_id in sorted(int(uav.id) for uav in env.uavs)
    }
    metrics = tracker.to_metrics()
    metrics.update(env.diagnostic_arrival_totals)
    metrics.update(
        {
            "completed_dag_count": int(completed),
            "completed_dag_per_slot": float(completed) / float(EPISODE_SLOTS),
            "dag_completion_rate_admitted": None if admitted == 0 else float(completed) / float(admitted),
            "dag_completion_rate_offered": None if offered == 0 else float(completed) / float(offered),
            "average_dag_flowtime": None if not flowtimes else float(np.mean(flowtimes)),
            "completed_dag_flowtime_sum": float(sum(flowtimes)),
            "completed_dag_flowtime_count": len(flowtimes),
            "episode_reward_total": float(reward_total),
            "avg_uav_queue_length": latest_info.get("avg_uav_queue_length"),
            "episode_end_admitted_incomplete_count": int(incomplete),
            "episode_end_active_dag_count": int(incomplete),
            "episode_end_ready_task_count": int(ready_count),
            "episode_end_queued_task_count": int(queued_task_count),
            "episode_end_executor_queue_count_by_uav": executor_queues,
        }
    )
    queue_mismatch = int(metrics["invalid_assignment_reasons"].get("queue_cap_mismatch", 0))
    technical_pass = bool(finite and queue_mismatch == 0)
    return {
        "schema": RESULT_SCHEMA,
        "cell": env.cell,
        "policy": str(policy),
        "scenario_seed": int(episode_payload["scenario_seed"]),
        "episode": int(episode_payload["episode"]),
        "episode_slots": EPISODE_SLOTS,
        "load_slots": LOAD_SLOTS,
        "active_dag_cap": int(env.max_active_dags_per_ue),
        "hard_queue_cap": int(env.capacity_context.hard_queue_cap),
        "scenario_checksum": str(episode_payload["scenario_checksum"]),
        "offered_event_checksum": str(episode_payload["offered_event_checksum"]),
        "offered_template_checksum": str(episode_payload["offered_template_checksum"]),
        "full_tape_checksum": str(full_tape_checksum),
        "pilot_prefix_checksum": str(pilot_prefix_checksum),
        "finite": bool(finite),
        "technical_pass": technical_pass,
        **metrics,
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


PAIRED_METRICS = (
    "choice_decision_fraction",
    "queue_full_mask_count",
    "all_uavs_full_decision_count",
    "repeated_ready_attempt_count",
    "completed_dag_per_slot",
    "dag_completion_rate_offered",
    "average_dag_flowtime",
    "episode_end_admitted_incomplete_count",
    "dag_completion_rate_admitted",
    "episode_reward_total",
    "choice_decision_count",
    "forced_decision_count",
    "skip_decision_count",
    "admitted_dag_count",
    "completed_dag_count",
    "avg_uav_queue_length",
)


def analyze_factorial_rows(
    rows: list[dict[str, Any]],
    *,
    policies: Iterable[str],
    scenario_seeds: Iterable[int],
    episode_count: int,
    expected_full_tape_checksum: str | None = None,
    expected_prefix_checksum: str | None = None,
) -> dict[str, Any]:
    policy_values = tuple(str(value) for value in policies)
    seed_values = tuple(int(value) for value in scenario_seeds)
    expected_keys = {
        (cell, policy, seed, episode)
        for cell in CELL_FLAGS
        for policy in policy_values
        for seed in seed_values
        for episode in range(int(episode_count))
    }
    indexed: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != RESULT_SCHEMA:
            raise ValueError("unsupported factorial result row schema")
        key = (
            str(row["cell"]),
            str(row["policy"]),
            int(row["scenario_seed"]),
            int(row["episode"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate factorial result row: {key}")
        indexed[key] = row
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        extra = sorted(set(indexed) - expected_keys)
        raise ValueError(f"factorial row key mismatch; missing={missing[:5]}, extra={extra[:5]}")

    gate_errors: list[str] = []
    paired_episode_effects: list[dict[str, Any]] = []
    for policy in policy_values:
        for seed in seed_values:
            for episode in range(int(episode_count)):
                cells = {
                    cell: indexed[(cell, policy, seed, episode)]
                    for cell in CELL_FLAGS
                }
                checksum_fields = (
                    "scenario_checksum",
                    "offered_event_checksum",
                    "offered_template_checksum",
                    "full_tape_checksum",
                    "pilot_prefix_checksum",
                )
                for field_name in checksum_fields:
                    values = {str(row[field_name]) for row in cells.values()}
                    if len(values) != 1:
                        raise ValueError(
                            f"paired checksum mismatch for {(policy, seed, episode)}: {field_name}"
                        )
                if expected_full_tape_checksum is not None and cells["A"]["full_tape_checksum"] != expected_full_tape_checksum:
                    raise ValueError("result full-tape checksum does not match requested tape")
                if expected_prefix_checksum is not None and cells["A"]["pilot_prefix_checksum"] != expected_prefix_checksum:
                    raise ValueError("result prefix checksum does not match requested tape")
                effects: dict[str, Any] = {
                    "policy": policy,
                    "scenario_seed": seed,
                    "episode": episode,
                    "effects": {},
                }
                for metric in PAIRED_METRICS:
                    values = {cell: cells[cell].get(metric) for cell in CELL_FLAGS}
                    if any(value is None for value in values.values()):
                        effects["effects"][metric] = {
                            "active": None,
                            "queue": None,
                            "interaction": None,
                        }
                        continue
                    numeric = {cell: float(value) for cell, value in values.items()}
                    effects["effects"][metric] = {
                        "active": ((numeric["B"] - numeric["A"]) + (numeric["D"] - numeric["C"])) / 2.0,
                        "queue": ((numeric["C"] - numeric["A"]) + (numeric["D"] - numeric["B"])) / 2.0,
                        "interaction": numeric["D"] - numeric["C"] - numeric["B"] + numeric["A"],
                    }
                paired_episode_effects.append(effects)
                for cell, row in cells.items():
                    if not bool(row.get("technical_pass")) or not bool(row.get("finite")):
                        gate_errors.append(f"non-finite or failed row: {(cell, policy, seed, episode)}")
                    invalid = row.get("invalid_assignment_reasons", {})
                    if sum(int(value) for value in invalid.values()) != 0:
                        gate_errors.append(f"invalid assignment: {(cell, policy, seed, episode)} {invalid}")
                    if cell in {"B", "D"} and int(row["active_cap_blocked_offered_count"]) != 0:
                        gate_errors.append(f"active nonbinding cap bound in {(cell, policy, seed, episode)}")
                    if cell in {"C", "D"} and int(row["queue_full_mask_count"]) != 0:
                        gate_errors.append(f"queue nonbinding cap bound in {(cell, policy, seed, episode)}")
                    if policy == "stage1_actor":
                        expected_path = STAGE1_ACTOR_CHECKPOINTS[seed]
                        if (
                            row.get("actor_checkpoint_path") != expected_path
                            or int(row.get("actor_training_seed", -1)) != seed
                            or int(row.get("actor_completed_update", -1)) != 30
                            or len(str(row.get("actor_checkpoint_sha256", ""))) != 64
                            or int(row.get("checkpoint_task_feature_dim", -1))
                            != int(row.get("graph_snapshot_task_feature_dim", -2))
                            or int(row.get("resolved_task_feature_dim", -1))
                            != int(row.get("checkpoint_task_feature_dim", -2))
                            or row.get("encoder_strict_load_pass") is not True
                        ):
                            gate_errors.append(f"Stage 1 checkpoint identity mismatch: {(cell, seed, episode)}")

    seed_level: dict[str, dict[str, dict[str, Any]]] = {}
    aggregate: dict[str, dict[str, Any]] = {}
    for policy in policy_values:
        seed_level[policy] = {}
        aggregate[policy] = {}
        for seed in seed_values:
            seed_level[policy][str(seed)] = {}
            selected = [
                row for row in paired_episode_effects
                if row["policy"] == policy and int(row["scenario_seed"]) == seed
            ]
            for metric in PAIRED_METRICS:
                metric_summary: dict[str, Any] = {}
                valid_count = 0
                null_count = 0
                for effect_name in ("active", "queue", "interaction"):
                    values = [row["effects"][metric][effect_name] for row in selected]
                    finite_values = [float(value) for value in values if value is not None]
                    valid_count = len(finite_values)
                    null_count = len(values) - valid_count
                    metric_summary[effect_name] = None if not finite_values else float(np.mean(finite_values))
                metric_summary["valid_paired_episode_count"] = valid_count
                metric_summary["null_paired_episode_count"] = null_count
                seed_level[policy][str(seed)][metric] = metric_summary
        for metric in PAIRED_METRICS:
            aggregate[policy][metric] = {}
            for effect_name in ("active", "queue", "interaction"):
                values = [
                    seed_level[policy][str(seed)][metric][effect_name]
                    for seed in seed_values
                    if seed_level[policy][str(seed)][metric][effect_name] is not None
                ]
                aggregate[policy][metric][effect_name] = {
                    "mean": None if not values else float(np.mean(values)),
                    "std": None if not values else float(np.std(values)),
                    "seed_count": len(values),
                }

    return {
        "schema": ANALYSIS_SCHEMA,
        "technical_pass": not gate_errors,
        "gate_errors": gate_errors,
        "expected_row_count": len(expected_keys),
        "actual_row_count": len(rows),
        "policies": list(policy_values),
        "scenario_seeds": list(seed_values),
        "episode_count": int(episode_count),
        "paired_episode_effects": paired_episode_effects,
        "seed_level_effects": seed_level,
        "three_seed_effects": aggregate,
        "cell_flowtime_aggregates": _flowtime_aggregates(rows),
        "stage1_actor_excluded_from_core_capacity_conclusion": True,
    }


def _flowtime_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["policy"]), str(row["cell"]))].append(row)
    for (policy, cell), selected in sorted(groups.items()):
        total_sum = sum(float(row["completed_dag_flowtime_sum"]) for row in selected)
        total_count = sum(int(row["completed_dag_flowtime_count"]) for row in selected)
        episode_values = [
            float(row["average_dag_flowtime"])
            for row in selected
            if row.get("average_dag_flowtime") is not None
        ]
        output[f"{policy}:{cell}"] = {
            "pooled_completed_dag_flowtime": None if total_count == 0 else total_sum / float(total_count),
            "pooled_completed_dag_count": total_count,
            "episode_mean_average_dag_flowtime": None if not episode_values else float(np.mean(episode_values)),
            "episode_mean_contributing_count": len(episode_values),
        }
    return output
