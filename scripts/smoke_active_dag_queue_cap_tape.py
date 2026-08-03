from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.capacity_factorial_diagnostic import (
    canonical_json_bytes,
    canonical_sha256,
    generate_scenario_tape,
    instantiate_dag_template,
    random_hash_uav,
    stable_task_id,
    validate_scenario_tape,
)
from environment.dag_tasks import DAGTaskManager


def main() -> int:
    try:
        canonical_json_bytes({"bad": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("canonical JSON accepted NaN")
    value = {"z": [3, 2, 1], "a": {"y": 2, "x": 1}}
    assert canonical_json_bytes(value) == b'{"a":{"x":1,"y":2},"z":[3,2,1]}'
    assert canonical_sha256(value) == canonical_sha256(json.loads(canonical_json_bytes(value)))

    original_probability = config.DAG_BASE_ARRIVAL_PROB
    original_multiplier = config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER
    try:
        config.DAG_BASE_ARRIVAL_PROB = 1.0
        config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER = 1.0
        tape = generate_scenario_tape(
            scenario_seeds=(7,),
            episodes=2,
            load_slots=3,
            episode_slots=5,
            num_ues=3,
            num_uavs=2,
        )
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_probability
        config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER = original_multiplier
    validate_scenario_tape(tape)
    assert len(tape["episodes"]) == 2
    assert all(row["arrival_opportunity_count"] == 9 for row in tape["episodes"])
    assert all(row["offered_dag_count"] == 9 for row in tape["episodes"])
    assert all(row["active_nonbinding_cap"] == 4 for row in tape["episodes"])
    assert tape["pilot_prefix_checksum"] == canonical_sha256(
        {
            "schema": tape["schema"],
            "controls": tape["controls"],
            "episodes": tape["episodes"],
        }
    )
    all_dags = [template["dag_id"] for row in tape["episodes"] for template in row["templates"]]
    all_tasks = [task["task_id"] for row in tape["episodes"] for template in row["templates"] for task in template["tasks"]]
    assert len(all_dags) == len(set(all_dags))
    assert len(all_tasks) == len(set(all_tasks))
    template = tape["episodes"][0]["templates"][0]
    assert all(
        task["task_id"] == stable_task_id(template["dag_id"], index)
        for index, task in enumerate(template["tasks"])
    )
    manager = DAGTaskManager(max_active_dags_per_ue=10)
    job = instantiate_dag_template(manager, template)
    assert job.dag_id == template["dag_id"]
    assert job.task_ids == template["task_ids"]
    assert job.sink_task_ids == template["sink_task_ids"]
    for payload in template["tasks"]:
        task = manager.get_task(payload["task_id"])
        assert task is not None
        assert task.predecessors == payload["predecessors"]
        assert task.successors == payload["successors"]
        assert task.num_operation == payload["num_operation"]

    expected = random_hash_uav(
        scenario_seed=42,
        episode=3,
        slot_index=9,
        stable_task_id_value="diag_task",
        legal_uav_ids=(0, 1, 2, 3, 4),
    )
    code = (
        "from environment.capacity_factorial_diagnostic import random_hash_uav;"
        "print(random_hash_uav(scenario_seed=42,episode=3,slot_index=9,"
        "stable_task_id_value='diag_task',legal_uav_ids=(0,1,2,3,4)))"
    )
    outputs = []
    for hash_seed in ("1", "999"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        outputs.append(
            subprocess.check_output([sys.executable, "-B", "-c", code], cwd=ROOT, env=env, text=True).strip()
        )
    assert outputs == [str(expected), str(expected)]
    print("SMOKE_ACTIVE_DAG_QUEUE_CAP_TAPE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
