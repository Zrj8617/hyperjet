from __future__ import annotations

import atexit
from dataclasses import dataclass
from functools import lru_cache
import logging
import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path
import threading
from typing import Any


LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def find_repository_root(anchor: str | Path | None = None) -> Path:
    """Find the source-tree root without relying on cwd or fixed parent hops."""
    start = Path(__file__ if anchor is None else anchor).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / "config.py").is_file() and (candidate / "environment" / "graph_builder.py").is_file():
            return candidate
    raise FileNotFoundError(f"could not locate HyperUAV repository root from {start}")


def resolve_repository_resource(relative_path: str | Path, *, anchor: str | Path | None = None) -> Path:
    """Resolve and validate one repository-relative file resource."""
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"repository resource path must be relative: {relative}")
    root = find_repository_root(anchor)
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"repository resource escapes source tree: {relative}")
    if not resolved.is_file():
        raise FileNotFoundError(f"repository resource not found: {resolved}")
    return resolved


@dataclass(frozen=True, slots=True)
class WorkerCleanupResult:
    stopped: bool
    terminate_used: bool
    kill_used: bool
    exitcode: int | None


def terminate_worker_process(
    process: Any,
    *,
    terminate_grace_seconds: float,
    kill_grace_seconds: float,
) -> WorkerCleanupResult:
    """Stop a child with bounded terminate -> kill escalation."""
    terminate_used = False
    kill_used = False
    try:
        alive = bool(process.is_alive())
    except Exception:
        alive = False
    if alive:
        terminate_used = True
        try:
            process.terminate()
        except Exception:
            pass
        try:
            process.join(timeout=max(float(terminate_grace_seconds), 0.0))
        except Exception:
            pass
    try:
        alive = bool(process.is_alive())
    except Exception:
        alive = False
    if alive:
        kill_used = True
        try:
            if hasattr(process, "kill"):
                process.kill()
            else:
                process.terminate()
        except Exception:
            pass
        try:
            process.join(timeout=max(float(kill_grace_seconds), 0.0))
        except Exception:
            pass
    try:
        stopped = not bool(process.is_alive())
    except Exception:
        stopped = True
    return WorkerCleanupResult(
        stopped=stopped,
        terminate_used=terminate_used,
        kill_used=kill_used,
        exitcode=getattr(process, "exitcode", None),
    )


class KaHyParWorkerClient:
    """Persistent crash-isolated KaHyPar client for one graph-builder lifetime."""

    def __init__(
        self,
        *,
        response_timeout_seconds: float,
        terminate_grace_seconds: float,
        kill_grace_seconds: float,
        max_consecutive_failures: int,
    ) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._response_timeout_seconds = max(float(response_timeout_seconds), 0.0)
        self._terminate_grace_seconds = max(float(terminate_grace_seconds), 0.0)
        self._kill_grace_seconds = max(float(kill_grace_seconds), 0.0)
        self._max_consecutive_failures = max(int(max_consecutive_failures), 1)
        self._lock = threading.Lock()
        self._connection: Connection | None = None
        self._process: Any | None = None
        self._request_id = 0
        self._consecutive_failures = 0
        self._circuit_open = False
        self._warning_emitted = False
        self._last_failure_reason: str | None = None
        self._cleanup_failed = False
        self._worker_start_count = 0
        self._closed = False
        atexit.register(self.close)

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def circuit_open(self) -> bool:
        return self._circuit_open

    @property
    def last_failure_reason(self) -> str | None:
        return self._last_failure_reason

    @property
    def cleanup_failed(self) -> bool:
        return self._cleanup_failed

    @property
    def worker_start_count(self) -> int:
        return self._worker_start_count

    @property
    def worker_pid(self) -> int | None:
        return None if self._process is None else getattr(self._process, "pid", None)

    @property
    def worker_alive(self) -> bool:
        if self._process is None:
            return False
        try:
            return bool(self._process.is_alive())
        except Exception:
            return False

    def note_failure(self, reason: str) -> None:
        """Count a parent-side configuration failure without spawning a worker."""
        with self._lock:
            if self._closed or self._circuit_open:
                return
            self._record_failure_locked(str(reason))
            if self._circuit_open:
                self._dispose_worker_locked()

    def partition(
        self,
        *,
        node_count: int,
        base_hyperedges: list[list[int]],
        partition_count: int,
        epsilon: float,
        seed: int,
        ini_path: str | Path,
    ) -> list[list[int]] | None:
        with self._lock:
            if self._closed:
                self._last_failure_reason = "worker client is closed"
                return None
            if self._circuit_open:
                return None
            if not self._ensure_worker_locked():
                return None

            self._request_id += 1
            request_id = self._request_id
            request = {
                "op": "partition",
                "request_id": request_id,
                "node_count": int(node_count),
                "base_hyperedges": [[int(pin) for pin in edge] for edge in base_hyperedges],
                "partition_count": int(partition_count),
                "epsilon": float(epsilon),
                "seed": int(seed),
                "ini_path": str(ini_path),
            }
            try:
                assert self._connection is not None
                self._connection.send(request)
            except (BrokenPipeError, EOFError, OSError, ValueError) as exc:
                self._fail_worker_locked(f"worker send failed: {type(exc).__name__}")
                return None

            try:
                assert self._connection is not None
                ready = self._connection.poll(self._response_timeout_seconds)
            except (EOFError, OSError, ValueError) as exc:
                self._fail_worker_locked(f"worker poll failed: {type(exc).__name__}")
                return None
            if not ready:
                self._fail_worker_locked("worker response timed out")
                return None

            try:
                response = self._connection.recv()
            except (EOFError, OSError, ValueError) as exc:
                self._fail_worker_locked(f"worker exited before response: {type(exc).__name__}")
                return None
            if not isinstance(response, dict) or int(response.get("request_id", -1)) != request_id:
                self._fail_worker_locked("worker protocol response mismatch")
                return None
            if not bool(response.get("ok", False)):
                error = str(response.get("error", "unknown worker error"))
                self._fail_worker_locked(error)
                return None

            groups = response.get("groups")
            if not isinstance(groups, list):
                self._fail_worker_locked("worker returned non-list groups")
                return None
            canonical: set[tuple[int, ...]] = set()
            for group in groups:
                if not isinstance(group, list):
                    self._fail_worker_locked("worker returned malformed group")
                    return None
                cleaned = tuple(sorted({int(idx) for idx in group if 0 <= int(idx) < int(node_count)}))
                if len(cleaned) >= 2:
                    canonical.add(cleaned)
            self._consecutive_failures = 0
            self._last_failure_reason = None
            return [list(group) for group in sorted(canonical)]

    def close(self) -> None:
        unregister = False
        with self._lock:
            if self._closed and self._process is None:
                return
            self._closed = True
            if self._connection is not None and self.worker_alive:
                try:
                    self._connection.send({"op": "shutdown"})
                except Exception:
                    pass
                try:
                    assert self._process is not None
                    self._process.join(timeout=self._terminate_grace_seconds)
                except Exception:
                    pass
            self._dispose_worker_locked()
            unregister = self._process is None
        if unregister:
            try:
                atexit.unregister(self.close)
            except Exception:
                pass

    def _ensure_worker_locked(self) -> bool:
        if self.worker_alive and self._connection is not None:
            return True
        self._dispose_worker_locked()
        parent_connection: Connection | None = None
        child_connection: Connection | None = None
        try:
            parent_connection, child_connection = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=kahypar_worker_main,
                args=(child_connection,),
                name="hyperuav-kahypar-worker",
                daemon=True,
            )
            process.start()
            child_connection.close()
            self._connection = parent_connection
            self._process = process
            self._worker_start_count += 1
            return True
        except Exception as exc:
            for connection in (parent_connection, child_connection):
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
            self._record_failure_locked(f"worker start failed: {type(exc).__name__}: {exc}")
            return False

    def _fail_worker_locked(self, reason: str) -> None:
        self._dispose_worker_locked()
        self._record_failure_locked(reason)

    def _record_failure_locked(self, reason: str) -> None:
        self._last_failure_reason = str(reason)
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_consecutive_failures:
            self._circuit_open = True
            if not self._warning_emitted:
                LOGGER.warning(
                    "KaHyPar disabled for this graph builder after %d consecutive failures: %s",
                    self._consecutive_failures,
                    self._last_failure_reason,
                )
                self._warning_emitted = True

    def _dispose_worker_locked(self) -> None:
        process = self._process
        connection = self._connection
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        self._connection = None
        if process is None:
            return
        result = terminate_worker_process(
            process,
            terminate_grace_seconds=self._terminate_grace_seconds,
            kill_grace_seconds=self._kill_grace_seconds,
        )
        if result.stopped:
            self._process = None
        else:
            self._cleanup_failed = True
            self._circuit_open = True
            self._last_failure_reason = "worker remained alive after kill escalation"
            LOGGER.error("KaHyPar worker pid=%s remained alive after kill escalation", getattr(process, "pid", None))


def kahypar_worker_main(connection: Connection) -> None:
    """Minimal worker loop; native exits remain isolated from the parent."""
    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break
            if not isinstance(request, dict):
                continue
            if request.get("op") == "shutdown":
                break
            request_id = int(request.get("request_id", -1))
            try:
                groups = _partition_request(request)
                connection.send({"request_id": request_id, "ok": True, "groups": groups})
            except Exception as exc:
                connection.send(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _partition_request(request: dict[str, Any]) -> list[list[int]]:
    import kahypar  # type: ignore

    node_count = int(request["node_count"])
    partition_count = int(request["partition_count"])
    base_hyperedges = [
        sorted({int(idx) for idx in edge if 0 <= int(idx) < node_count})
        for edge in request["base_hyperedges"]
    ]
    base_hyperedges = [edge for edge in base_hyperedges if len(edge) >= 2]
    if node_count < 2 or not base_hyperedges:
        raise ValueError("KaHyPar request requires at least two nodes and one valid hyperedge")

    hyperedge_indices = [0]
    pins: list[int] = []
    for edge in base_hyperedges:
        pins.extend(edge)
        hyperedge_indices.append(len(pins))

    context = kahypar.Context()
    context.loadINIconfiguration(str(request["ini_path"]))
    context.setK(partition_count)
    context.setEpsilon(float(request["epsilon"]))
    if hasattr(context, "setSeed"):
        context.setSeed(int(request["seed"]))
    context.suppressOutput(True)

    try:
        hypergraph = kahypar.Hypergraph(
            node_count,
            len(base_hyperedges),
            hyperedge_indices,
            pins,
            partition_count,
            [1] * len(base_hyperedges),
            [1] * node_count,
        )
    except TypeError:
        hypergraph = kahypar.Hypergraph(
            node_count,
            len(base_hyperedges),
            hyperedge_indices,
            pins,
            partition_count,
        )

    kahypar.partition(hypergraph, context)
    groups_by_block: dict[int, list[int]] = {}
    for node_idx in range(node_count):
        if hasattr(hypergraph, "blockID"):
            block_id = int(hypergraph.blockID(node_idx))
        elif hasattr(hypergraph, "block_id"):
            block_id = int(hypergraph.block_id(node_idx))
        else:
            raise RuntimeError("installed KaHyPar Hypergraph has no block-id accessor")
        groups_by_block.setdefault(block_id, []).append(node_idx)

    canonical = {
        tuple(sorted(group))
        for group in groups_by_block.values()
        if len(group) >= 2
    }
    return [list(group) for group in sorted(canonical)]
