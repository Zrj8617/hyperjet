from __future__ import annotations

from dataclasses import dataclass
import math


DIAGNOSTIC_QUEUE_LENGTH_NORM_REF = 16
DIAGNOSTIC_REMAINING_SLOTS_FEATURE_REF = 16
DIAGNOSTIC_SLOT_ASSIGNED_NORM_REF = 16
DIAGNOSTIC_QUEUE_WORKLOAD_NORM_REF = 80_000_000.0


@dataclass(frozen=True, slots=True)
class DiagnosticCapacityContext:
    """One immutable source of queue legality for a diagnostic episode."""

    hard_queue_cap: int
    queue_length_norm_ref: int = DIAGNOSTIC_QUEUE_LENGTH_NORM_REF
    remaining_slots_feature_ref: int = DIAGNOSTIC_REMAINING_SLOTS_FEATURE_REF
    slot_assigned_norm_ref: int = DIAGNOSTIC_SLOT_ASSIGNED_NORM_REF
    queue_workload_norm_ref: float = DIAGNOSTIC_QUEUE_WORKLOAD_NORM_REF

    def __post_init__(self) -> None:
        for name in (
            "hard_queue_cap",
            "queue_length_norm_ref",
            "remaining_slots_feature_ref",
            "slot_assigned_norm_ref",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(float(self.queue_workload_norm_ref)):
            raise ValueError("queue_workload_norm_ref must be finite")
        if float(self.queue_workload_norm_ref) != DIAGNOSTIC_QUEUE_WORKLOAD_NORM_REF:
            raise ValueError("diagnostic queue workload reference must equal 80_000_000.0")


def queue_cap_from_context(context: DiagnosticCapacityContext | None, fallback: int) -> int:
    return int(context.hard_queue_cap) if context is not None else int(fallback)
