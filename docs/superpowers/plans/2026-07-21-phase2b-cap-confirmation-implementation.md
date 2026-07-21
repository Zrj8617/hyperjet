# Phase 2B Branch 1 Cap 1/2/3 Confirmation Implementation Plan

Date: 2026-07-21
Design: `docs/superpowers/specs/2026-07-15-phase2b-ready-offload-funnel-and-concurrency-design.md`
Stage 1 evidence: service/queue throughput saturation (Branch 1)

## Objective and boundary

Implement and run only the Section 7.2 small confirmation. Add a strict
run-level maximum of 1, 2, or 3 active DAGs per UE, while leaving DAG shape,
task attributes, mobility, UAV compute, queue capacity, policies, graph
construction, execution timing, and reward unchanged.

The confirmation matrix is exactly 12 cells:

- caps `1`, `2`, `3`;
- arrival probability `0.0145`;
- policies `greedy`, `random`;
- seeds `4242`, `4243`;
- 200 arrival slots and at most 500 drain slots.

This is characterization, not a new load search. Do not run the adaptive grid,
increase arrival probability, alter service capacity, or start PPO training.

## Task 1: Authoritative concurrency limit

File: `environment/dag_tasks.py`

Add a positive-integer constructor value `max_active_dags_per_ue`, defaulting
to `1`. The task manager remains the authoritative source of active-DAG
membership. Expose deterministic helpers to list and count active DAGs and to
test whether another DAG may be admitted. Enforce the cap at every DAG creation
path. The default and explicit value `1` must preserve legacy behavior and RNG
consumption.

## Task 2: Environment admission and UE compatibility state

Files:

- `environment/env.py`
- `environment/user_equipments.py`

Pass the immutable run-level cap into the task manager and expose it for
provenance. Arrival admission consults task-manager capacity rather than the
single compatibility ID on the UE. When one DAG completes, recompute the UE's
compatibility marker from all remaining authoritative active DAGs: keep waiting
if another DAG remains, otherwise release the UE.

## Task 3: Diagnostic sweep and provenance

File: `scripts/diag_clean_load.py`

Add `--active-dag-caps` with strict positive, duplicate-free parsing. Include
the cap in every manifest cell, raw row, summary grouping, CSV, console table,
and Markdown report. Preserve isolated policy RNG streams and the Stage 1
funnel definitions.

Report the primary result as executor-accepted tasks per arrival slot. Compute
paired deltas against cap `1` for the same scene parameters, seed, and policy.
Keep drain, final backlog, capacity blocking, queue saturation, active DAGs,
hyperedges, and KaHyPar integrity as separate outcomes. Mark a grouped
configuration feasible only when every cell drains, leaves no final ready or
active-DAG backlog, and has finite primary/congestion metrics.

## Task 4: Focused tests

Files:

- `scripts/smoke_clean_dag_concurrency.py`
- `scripts/smoke_clean_load_diagnostic.py`
- affected environment smoke scripts

Verify:

1. default cap and explicit cap `1` generate the same DAG under a fixed seed
   and reject a second active DAG;
2. cap `2` admits two DAGs and completing one preserves the other UE wait;
3. cap `3` rejects a fourth active DAG;
4. cap provenance, paired deltas, cell counts, funnel monotonicity, executor
   accounting, RNG pairing, drain restoration, and artifact schemas;
5. existing DAG, environment, execution, assignment, graph, and diagnostic
   non-Torch smokes still pass.

## Task 5: Exact revision and server preflight

After local validation:

1. review the tracked diff and protected untracked files;
2. commit only implementation, tests, and this plan;
3. push the current branch;
4. synchronize the exact commit to `/data2/zrj2025/HyperUAV` without deleting
   server `logs/` or `runs/`;
5. verify server HEAD and run the focused concurrency, Torch/CUDA, KaHyPar, and
   diagnostic smokes.

No formal cell starts until the exact server revision passes preflight.

## Task 6: Server confirmation and stop condition

Create new timestamped paths under `/data2/zrj2025/HyperUAV/logs/` and
`/data2/zrj2025/HyperUAV/runs/`. Start the 12-cell matrix with persistent
`nohup`, record PID, log, and result directory, then monitor to terminal
completion. Preserve resumable per-cell progress.

After completion, verify all 12 cells, no non-finite values, no funnel or
executor accounting mismatch, no KaHyPar degradation, and exact cap/seed/policy
coverage. Compare cap `1` against the prior Stage 1 cells and report paired
accepted-action deltas plus congestion and structure outcomes. Stop Phase 2B
after this report; do not continue to higher arrival rates or a full grid.

## Acceptance criteria

1. Cap `1` preserves legacy behavior under the fixed-seed test.
2. Multi-DAG completion never releases a UE with another active DAG.
3. No UE exceeds its configured cap.
4. Every artifact records the cap and exact Git revision.
5. The server confirmation contains exactly 12 authorized cells.
6. The report includes paired action-density deltas and separate feasibility,
   congestion, backlog, structure, and integrity evidence.
7. Protected local files and server `logs/`/`runs/` remain untouched.
