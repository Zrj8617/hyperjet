# Phase 2B Stage 0/1 Ready-to-Offload Funnel Implementation Plan

Date: 2026-07-21
Design: `docs/superpowers/specs/2026-07-15-phase2b-ready-offload-funnel-and-concurrency-design.md`
Baseline: `e5412a43fa158c97b53c0ef324bf417d98f761d5`

## Objective and boundary

Implement only the approved Phase 2B Stage 0 report corrections and Stage 1
fixed-hover, non-learning ready-to-offload funnel. The implementation must not
change environment arrivals, policy semantics, queue capacity, task execution,
reward, PPO training, per-UE DAG concurrency, or any Stage 2 factor.

The primary Stage 1 result is executor-accepted ready tasks per arrival slot.
The implementation-efficiency denominator is a policy-independent,
legality-conditioned capacitated matching ceiling, not raw queue space.

## Task 1: Focused pure-Python diagnostic primitives

Files:

- `scripts/diag_clean_load.py`
- `scripts/smoke_clean_load_diagnostic.py`

Add small deterministic helpers for:

1. longest consecutive true run;
2. final-window backlog level, delta, and least-squares slope;
3. per-UAV queue aggregation;
4. legality-conditioned capacitated bipartite matching;
5. funnel ratio aggregation with explicit zero-denominator counts.

The matching helper accepts task-to-UAV adjacency and residual capacity by UAV.
It must enforce task capacity one, support overlapping candidate sets and
unequal UAV capacities, and return zero for an empty or zero-capacity graph.
Use a deterministic augmenting-path implementation so no new dependency or RNG
stream is introduced.

## Task 2: Stage 0 completion and drain semantics

File: `scripts/diag_clean_load.py`

At the end of arrival slots, freeze generated/completed counts. Report:

- `arrival_completion_rate`;
- `final_completion_rate` after drain;
- `drain_slots_used`;
- `drain_completed`;
- final and arrival-end generated/completed counts.

Retain `completion_rate_arrival_end`, `completion_rate`, and
`drain_slots_executed` as deprecated machine-readable aliases for one revision.
Human-readable output and sweep summaries must use the corrected names.

## Task 3: Stage 0 queue, saturation, and backlog semantics

File: `scripts/diag_clean_load.py`

For every arrival slot capture queue occupancy at three boundaries:

1. pre-decision, directly from executor queues;
2. post-decision/pre-execution, reconstructed from pre-decision queues plus
   assignments actually accepted by the executor;
3. post-execution, directly from executor queues after the slot advances.

Report total-queue distributions, per-UAV mean/max occupancy, initially-full
UAV distributions, pre- and post-decision all-UAV-full fractions, and longest
consecutive all-full runs. Report arrival-end ready and active-DAG backlog plus
final-50-arrival-slot delta and slope for both backlog series.

Replace the report concept `offloading_skipped_rate` with
`capacity_blocked_task_slot_ratio`: frozen ready task-slot opportunities that
have no legal capacity before selection or lose their final candidate because
of same-slot reservations, divided by all frozen ready task-slot opportunities.
Retain old skip fields only as deprecated aliases.

## Task 4: Stage 1 slot funnel

File: `scripts/diag_clean_load.py`

Within each arrival slot record:

- frozen ready tasks;
- frozen ready tasks represented as ready in the graph;
- graph-ready tasks with at least one initial legal candidate;
- policy-selected assignments;
- assignment-buffer accepted entries;
- executor-accepted assignments;
- executor-invalid assignments.

Record reason counts for graph-missing tasks, no initial legal capacity,
same-slot loss of the last candidate, policy omission, buffer rejection,
executor legality rejection, schedule-record failure, and malformed UAV IDs.

The policy loop must consume only graph-ready tasks, matching the production
graph-to-policy path. Before sequential reservations, compute initial legality
for every frozen task from one unmodified pre-decision reservation snapshot.
This preserves graph-loss observability and supplies the matching ceiling.

## Task 5: Legal-capacity ceiling and conversion metrics

File: `scripts/diag_clean_load.py`

For each arrival slot build the matching graph from all valid frozen ready
tasks, the legality rules active in the current revision, and actual
pre-decision residual queue positions. Report:

- `legal_capacity_ceiling`;
- `legal_capacity_utilization = executor accepted / ceiling` for nonzero cells;
- zero-ceiling slot count/fraction separately;
- `executor_accepted / frozen_ready` learning-sample conversion;
- accepted assignments per arrival slot as the primary action-density metric.

Do not fold congestion, drain, or completion into the primary metric. Add an
integrity flag for `executor <= buffer <= selected <= frozen` and report any
violation count rather than silently clipping data.

## Task 6: Greedy same-slot reservation consistency

File: `scripts/diag_clean_load.py`

When greedy selects the lowest estimated-finish-time candidate, reserve the
selected candidate with both `estimated_finish_time` and
`estimated_queued_workload`. Random selection continues to reserve only the
queue position because it does not compute an estimate. This changes greedy
choice quality only; it does not change the matching ceiling.

## Task 7: Executor rejection provenance

Files:

- `environment/task_execution.py`
- `environment/env.py`
- relevant smoke scripts

Extend `CleanExecutionStepStats` with a reason-count dictionary. Increment one
of `malformed_uav_id`, `illegal_assignment`, or `schedule_record_failure` at
the existing rejection branches while preserving the aggregate
`invalid_assignments`. Expose a copied dictionary in environment `info`.
This is diagnostic provenance only and must not change acceptance behavior.

## Task 8: Sweep summaries and artifacts

File: `scripts/diag_clean_load.py`

Update console, JSON, CSV, and Markdown summaries to use Stage 0/1 names and to
include, at minimum:

- arrival and final completion;
- drain usage/completions;
- executor-accepted action density;
- legal-capacity utilization and zero-ceiling fraction;
- capacity-blocked task-slot ratio;
- pre/post queue saturation and longest all-full run;
- arrival-end backlog and final-50 trends;
- funnel loss reason totals;
- KaHyPar integrity.

Do not retain the Phase 2A `coarse_safe` continuation decision as the Phase 2B
branch decision. The report supplies evidence; branch selection remains a
reviewed interpretation after the 12-cell run.

## Task 9: Local validation without Torch

Files:

- `scripts/smoke_clean_load_diagnostic.py`
- `scripts/smoke_clean_execution.py` if executor provenance needs direct coverage

Add deterministic fixtures for disjoint candidates, overlapping candidates,
unequal capacities, and zero legal capacity. Assert funnel monotonicity,
executor/newly-assigned equality, corrected completion aliases, drain arrival
restoration, queue boundary consistency, reason-count conservation, new report
fields, RNG isolation assumptions, and absence of KaHyPar worker leaks.

Run only local checks that do not import Torch:

1. focused load diagnostic smoke;
2. focused execution/environment smokes affected by provenance changes;
3. AST/`py_compile` checks using a workspace-local temporary cache;
4. `git diff --check` and final scope review.

Torch/KaHyPar validation and the 12-cell Stage 1 execution are not part of this
implementation turn. They require an exact pushed HEAD and a separately
announced server action.

## Acceptance criteria

1. Existing environment behavior is unchanged for identical assignments.
2. Corrected completion/drain names are present and old names are aliases only.
3. Capacity blocking means deferred ready-task/slot opportunity, not rejection.
4. Matching fixtures prove the ceiling does not double-count shared candidates.
5. Every Stage 1 slot has auditable funnel counts and queue boundaries.
6. Greedy reservations propagate estimated finish time and workload.
7. Executor invalid-reason totals equal aggregate invalid assignments.
8. Local non-Torch tests pass and protected untracked files remain untouched.

## Deferred work

- Stage 1 server preflight and the approved 12-cell matrix;
- evidence-backed Branch 1/2/3 selection;
- any cap 1/2/3 implementation or Stage 2 grid;
- service-capacity changes;
- PPO training, counterfactual credit, or lagged residual-Q changes.
