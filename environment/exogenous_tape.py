from __future__ import annotations

from dataclasses import asdict
import gzip
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

import config
from environment.dag_tasks import DAGJob, DAGTaskManager, TaskNode
from environment.uavs import UAV
from environment.user_equipments import UE


TAPE_SCHEMA = "hyperuav_exogenous_offer_mobility_tape_v1"


def generate_tape(*, tape_id: int, slots: int) -> dict[str, Any]:
    """Generate a policy-independent mobility path and all exogenous DAG offers."""
    if int(slots) <= 0:
        raise ValueError("slots must be positive")
    random.seed(int(tape_id))
    np.random.seed(int(tape_id))
    radius = float(config.HOTSPOT_RADIUS)
    hotspot = np.array(
        [
            np.random.uniform(radius, float(config.AREA_WIDTH) - radius),
            np.random.uniform(radius, float(config.AREA_HEIGHT) - radius),
        ],
        dtype=np.float32,
    )
    uavs = [UAV(i) for i in range(int(config.NUM_UAVS))]
    ues = [UE(i) for i in range(int(config.NUM_UES))]
    for ue in ues:
        ue.reset_episode_state()
    manager = DAGTaskManager(max_active_dags_per_ue=max(int(slots), 1) + 1)
    slot_rows: list[dict[str, Any]] = []
    offer_count = 0
    for slot_index in range(int(slots)):
        for ue in ues:
            # The external mobility law is deliberately independent of policy and
            # service state. Replay installs these absolute states verbatim.
            ue.service_waiting = False
            ue.active_dag_id = None
            ue.update_position(commit_position=True)
        offers: list[dict[str, Any]] = []
        arrival_seconds = float(slot_index + 1) * float(config.TIME_SLOT_DURATION)
        for ue in ues:
            probability = ue.get_arrival_probability(hotspot, radius)
            if float(np.random.random()) >= float(probability):
                continue
            job = manager.create_dag_for_ue(
                ue_id=int(ue.id),
                source_pos=ue.pos[:2].copy(),
                current_time_step=arrival_seconds,
            )
            offer_count += 1
            offers.append(_serialize_offer(job=job, manager=manager, offer_index=offer_count))
        slot_rows.append(
            {
                "slot_index": int(slot_index),
                "time_seconds": arrival_seconds,
                "ues": [_serialize_ue(ue) for ue in ues],
                "offers": offers,
            }
        )
    return {
        "schema": TAPE_SCHEMA,
        "tape_id": int(tape_id),
        "generation_seed": int(tape_id),
        "slots": int(slots),
        "time_slot_duration": float(config.TIME_SLOT_DURATION),
        "scene": {
            "area_width": float(config.AREA_WIDTH),
            "area_height": float(config.AREA_HEIGHT),
            "num_uavs": int(config.NUM_UAVS),
            "num_ues": int(config.NUM_UES),
            "hotspot_radius": radius,
        },
        "initial": {
            "hotspot_center": hotspot.tolist(),
            "uav_positions": [uav.pos.tolist() for uav in uavs],
            "ues": [_serialize_ue(ue) for ue in ues],
        },
        "offer_count": int(offer_count),
        "slot_rows": slot_rows,
    }


def write_tape(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def read_tape(path: Path) -> dict[str, Any]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != TAPE_SCHEMA:
        raise ValueError(f"unsupported exogenous tape schema: {payload.get('schema')}")
    return payload


class ExogenousTapeReplay:
    """Episode-local replay cursor consumed by ``Env`` through a narrow hook."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if payload.get("schema") != TAPE_SCHEMA:
            raise ValueError("invalid exogenous tape")
        self.payload = payload
        self.slot_rows = list(payload["slot_rows"])
        self.next_slot = 0
        self.admitted_offer_ids: set[str] = set()
        self.rejected_offer_ids: set[str] = set()

    @property
    def tape_id(self) -> int:
        return int(self.payload["tape_id"])

    @property
    def offers(self) -> list[dict[str, Any]]:
        return [offer for row in self.slot_rows for offer in row["offers"]]

    def reset_env(self, env: Any) -> None:
        self.next_slot = 0
        self.admitted_offer_ids.clear()
        self.rejected_offer_ids.clear()
        initial = self.payload["initial"]
        env.hotspot_center = np.asarray(initial["hotspot_center"], dtype=np.float32).copy()
        for uav, position in zip(env.uavs, initial["uav_positions"]):
            uav.pos = np.asarray(position, dtype=np.float32).copy()
        for ue, state in zip(env.ues, initial["ues"]):
            _restore_ue(ue, state)

    def apply_mobility(self, env: Any) -> None:
        if self.next_slot >= len(self.slot_rows):
            raise RuntimeError("exogenous tape exhausted")
        row = self.slot_rows[self.next_slot]
        if int(row["slot_index"]) != self.next_slot:
            raise RuntimeError("non-contiguous exogenous tape")
        for ue, state in zip(env.ues, row["ues"]):
            _restore_ue(ue, state, preserve_service_state=True)

    def process_offers(self, env: Any) -> int:
        row = self.slot_rows[self.next_slot]
        self.next_slot += 1
        funnel = env._empty_arrival_funnel()
        funnel["arrival_attempt_count"] = int(len(env.ues))
        funnel["arrival_draw_count"] = int(len(env.ues))
        funnel["arrival_sampled_event_count"] = int(len(row["offers"]))
        funnel["arrival_no_event_count"] = int(len(env.ues) - len(row["offers"]))
        admitted_ids: list[str] = []
        for offer in row["offers"]:
            offer_id = str(offer["offer_id"])
            ue_id = int(offer["job"]["ue_id"])
            if not env.task_manager.can_accept_dag_for_ue(ue_id):
                self.rejected_offer_ids.add(offer_id)
                funnel["arrival_blocked_count"] += 1
                funnel["arrival_blocked_reasons"]["active_dag_cap"] += 1
                continue
            _admit_offer(env.task_manager, offer)
            self.admitted_offer_ids.add(offer_id)
            admitted_ids.append(offer_id)
            funnel["arrival_admitted_count"] += 1
            for ue in env.ues:
                if int(ue.id) == ue_id:
                    ue.enter_service_waiting(offer_id)
                    break
        env.task_manager._last_created_dag_ids = list(admitted_ids)
        env._latest_arrival_funnel = funnel
        env._accumulate_arrival_funnel(funnel)
        env._last_new_dag_arrived = bool(admitted_ids)
        env._latest_dag_arrival_version = env.task_manager.dag_arrival_version
        return int(len(admitted_ids))


def _serialize_ue(ue: UE) -> dict[str, Any]:
    return {
        "id": int(ue.id),
        "pos": ue.pos.tolist(),
        "speed": float(ue.speed),
        "theta": float(ue.theta),
        "velocity": ue.velocity.tolist(),
    }


def _restore_ue(ue: UE, state: dict[str, Any], *, preserve_service_state: bool = False) -> None:
    if int(ue.id) != int(state["id"]):
        raise ValueError("UE order mismatch in exogenous tape")
    service_waiting = bool(ue.service_waiting)
    active_dag_id = ue.active_dag_id
    ue.pos = np.asarray(state["pos"], dtype=np.float32).copy()
    ue.speed = float(state["speed"])
    ue.theta = float(state["theta"])
    ue.velocity = np.asarray(state["velocity"], dtype=np.float32).copy()
    if preserve_service_state:
        ue.service_waiting = service_waiting
        ue.active_dag_id = active_dag_id


def _serialize_offer(*, job: DAGJob, manager: DAGTaskManager, offer_index: int) -> dict[str, Any]:
    return {
        "offer_id": str(job.dag_id),
        "offer_index": int(offer_index),
        "job": _jsonable(asdict(job)),
        "tasks": [_jsonable(asdict(manager.get_task(task_id))) for task_id in job.task_ids],
    }


def _admit_offer(manager: DAGTaskManager, offer: dict[str, Any]) -> None:
    job_payload = dict(offer["job"])
    job_payload["source_pos"] = np.asarray(job_payload["source_pos"], dtype=np.float32)
    job = DAGJob(**job_payload)
    if job.dag_id in manager._jobs:
        raise ValueError(f"duplicate taped DAG: {job.dag_id}")
    tasks: list[TaskNode] = []
    for raw in offer["tasks"]:
        task_payload = dict(raw)
        task_payload["source_pos"] = np.asarray(task_payload["source_pos"], dtype=np.float32)
        tasks.append(TaskNode(**task_payload))
    manager._jobs[job.dag_id] = job
    for task in tasks:
        if task.task_id in manager._tasks:
            raise ValueError(f"duplicate taped task: {task.task_id}")
        manager._tasks[task.task_id] = task
        manager._tasks_by_ue.setdefault(int(task.ue_id), []).append(task.task_id)
    manager._dag_arrival_version += 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
