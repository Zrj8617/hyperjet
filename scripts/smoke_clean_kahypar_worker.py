from __future__ import annotations

import importlib.util
import multiprocessing
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.graph_builder import CleanGraphBuilder
from environment.kahypar_worker import (
    KaHyParWorkerClient,
    find_repository_root,
    resolve_repository_resource,
    terminate_worker_process,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _new_client() -> KaHyParWorkerClient:
    return KaHyParWorkerClient(
        response_timeout_seconds=float(config.KAHYPAR_WORKER_TIMEOUT_SECONDS),
        terminate_grace_seconds=float(config.KAHYPAR_WORKER_TERMINATE_GRACE_SECONDS),
        kill_grace_seconds=float(config.KAHYPAR_WORKER_KILL_GRACE_SECONDS),
        max_consecutive_failures=int(config.KAHYPAR_MAX_CONSECUTIVE_FAILURES),
    )


def _request(
    client: KaHyParWorkerClient,
    ini_path: Path,
    *,
    partition_count: int = 4,
) -> list[list[int]] | None:
    edges = [
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8],
        [8, 9, 10],
        [0, 10, 11],
        [1, 5, 9],
        [3, 7, 11],
    ]
    return client.partition(
        node_count=12,
        base_hyperedges=edges,
        partition_count=partition_count,
        epsilon=float(config.KAHYPAR_EPSILON),
        seed=int(config.KAHYPAR_SEED),
        ini_path=ini_path,
    )


def _stubborn_worker(ready: multiprocessing.synchronize.Event) -> None:
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(0.05)


def _assert_status_boundaries() -> None:
    active_ids = ["task_0", "task_1"]
    task_id_to_idx = {task_id: idx for idx, task_id in enumerate(active_ids)}
    idx_to_task_id = {idx: task_id for task_id, idx in task_id_to_idx.items()}

    success_builder = CleanGraphBuilder()
    try:
        success_builder._run_kahypar_partition = lambda **_: []  # type: ignore[method-assign]
        groups = success_builder._build_partition_hyperedges(
            active_task_ids=active_ids,
            task_id_to_idx=task_id_to_idx,
            idx_to_task_id=idx_to_task_id,
            base_hyperedges=[[0, 1]],
            current_time_step=0,
            force_update=True,
        )
        _assert(groups == [], "successful singleton filtering should emit no partition edges")
        _assert(success_builder.last_partition_status == "success", "successful empty result must remain success")
    finally:
        success_builder.close()

    failure_builder = CleanGraphBuilder()
    try:
        failure_builder._run_kahypar_partition = lambda **_: None  # type: ignore[method-assign]
        groups = failure_builder._build_partition_hyperedges(
            active_task_ids=active_ids,
            task_id_to_idx=task_id_to_idx,
            idx_to_task_id=idx_to_task_id,
            base_hyperedges=[[0, 1]],
            current_time_step=0,
            force_update=True,
        )
        _assert(groups == [], "failed partition without cache should emit no partition edges")
        _assert(
            failure_builder.last_partition_status == "degraded_no_cache",
            "execution failure must be degraded rather than success",
        )
    finally:
        failure_builder.close()


def _assert_bounded_kill() -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_stubborn_worker, args=(ready,), name="hyperuav-kahypar-stubborn-smoke")
    process.start()
    try:
        _assert(ready.wait(timeout=5.0), "stubborn worker did not start")
        result = terminate_worker_process(
            process,
            terminate_grace_seconds=0.2,
            kill_grace_seconds=1.0,
        )
        _assert(result.stopped, "terminate/kill escalation must stop the stubborn worker")
        if sys.platform != "win32":
            _assert(result.kill_used, "POSIX worker ignoring SIGTERM should require kill escalation")
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)


def _assert_circuit_breaker(ini_path: Path) -> None:
    client = _new_client()
    missing_ini = ini_path.with_name("missing_for_smoke.ini")
    try:
        for _ in range(int(config.KAHYPAR_MAX_CONSECUTIVE_FAILURES)):
            _assert(_request(client, missing_ini) is None, "missing INI must fail")
        _assert(client.circuit_open, "three consecutive failures must open the circuit")
        starts_before = client.worker_start_count
        _assert(_request(client, ini_path) is None, "open circuit must degrade without a request")
        _assert(client.worker_start_count == starts_before, "open circuit must not spawn another worker")
    finally:
        client.close()
    _assert(not client.worker_alive, "circuit-breaker client left a worker alive")


def _assert_installed_wheel(ini_path: Path) -> None:
    if importlib.util.find_spec("kahypar") is None:
        print("smoke_clean_kahypar_worker: KaHyPar unavailable; native success/abort checks skipped")
        return

    client = _new_client()
    try:
        results = [_request(client, ini_path) for _ in range(5)]
        _assert(all(result for result in results), "valid KaHyPar request should return non-empty groups")
        canonical = {tuple(tuple(group) for group in result or []) for result in results}
        _assert(len(canonical) == 1, "fixed seed should preserve canonical group membership")
        _assert(client.worker_start_count == 1, "successful requests should reuse one persistent worker")
        _assert(client.consecutive_failures == 0, "successful response should clear failure count")
    finally:
        client.close()
    _assert(not client.worker_alive, "successful client left a worker alive")

    recovery_client = _new_client()
    try:
        missing_ini = ini_path.with_name("missing_for_native_exit_smoke.ini")
        _assert(_request(recovery_client, missing_ini) is None, "missing INI should be isolated")
        _assert(not recovery_client.worker_alive, "native INI failure worker should be reaped")
        recovered = _request(recovery_client, ini_path)
        _assert(recovered is not None, "client should restart after one isolated worker failure")
        _assert(recovery_client.consecutive_failures == 0, "successful recovery should reset failure count")
    finally:
        recovery_client.close()

    invalid_k_client = _new_client()
    try:
        _assert(_request(invalid_k_client, ini_path, partition_count=0) is None, "invalid k must be isolated")
        _assert(not invalid_k_client.worker_alive, "invalid-k worker should be reaped")
    finally:
        invalid_k_client.close()

    singleton_client = _new_client()
    try:
        groups = singleton_client.partition(
            node_count=2,
            base_hyperedges=[[0, 1]],
            partition_count=2,
            epsilon=float(config.KAHYPAR_EPSILON),
            seed=int(config.KAHYPAR_SEED),
            ini_path=ini_path,
        )
        _assert(groups == [], "successful two-way singleton partition should return an empty group list")
    finally:
        singleton_client.close()


def main() -> None:
    root = find_repository_root(__file__)
    _assert(root == ROOT, "repository-root locator returned the wrong root")
    _assert(not Path(config.KAHYPAR_INI_RELATIVE_PATH).is_absolute(), "configured INI path must be relative")
    ini_path = resolve_repository_resource(config.KAHYPAR_INI_RELATIVE_PATH, anchor=__file__)
    _assert(ini_path.is_file(), "vendored KaHyPar INI is missing")

    _assert_status_boundaries()
    _assert_bounded_kill()
    _assert_circuit_breaker(ini_path)
    _assert_installed_wheel(ini_path)

    residual = [child for child in multiprocessing.active_children() if child.name.startswith("hyperuav-kahypar")]
    _assert(not residual, f"KaHyPar smoke left residual children: {residual}")
    print("smoke_clean_kahypar_worker passed")


if __name__ == "__main__":
    main()
