# Phase 2A Load Boundary Scan Implementation Plan

Date: 2026-07-15
Design: `docs/superpowers/specs/2026-07-15-phase2-concurrency-load-calibration-design.md`

## Objective

Extend the existing non-learning clean-load diagnostic so it measures actual
arrival/concurrency/hypergraph pressure, honors drain during sweeps, persists an
auditable manifest and progress, and can launch the approved 20-cell coarse
boundary scan without changing training or environment behavior.

## Task 1: Focused diagnostic coverage

Add `scripts/smoke_clean_load_diagnostic.py` with tiny deterministic fake-free
environment runs. Cover:

1. per-slot created-DAG, eligible/suppressed UE, active DAG, ready-task,
   assignment, queue-pressure, and hyperedge aggregates;
2. zero-arrival and multi-arrival slot ratios;
3. finite percentile behavior for empty and nonempty samples;
4. graph-builder cleanup;
5. sweep forwarding of `--drain-slots`;
6. output manifest, JSONL rows, summary JSON/CSV, and progress lifecycle;
7. random-policy RNG isolation remains intact.

The local machine has no Torch runtime. These tests must not require Torch.
KaHyPar-enabled behavior is verified with the configured server runtime.

## Task 2: Extend slot diagnostics

Update `scripts/diag_clean_load.py` to build the same clean graph snapshot used
by the training path before every decision. Keep policy action selection and
environment advancement unchanged. Record arrival-phase samples separately from
drain samples and preserve existing output fields for compatibility.

Close the graph builder in `finally`, expose cleanup/circuit/failure state, and
treat degraded partition statuses as diagnostic failures in the formal runner.

## Task 3: Persist sweep artifacts

Add an optional `--output-dir`. When supplied, require a new or empty directory
and write before the first cell:

- `manifest.json` with git/config/runtime/arguments and fixed cell order;
- `progress.json` with total/completed/status;
- `sweep_rows.jsonl` appended after every completed cell.

After all cells, write:

- `sweep_summary.json`;
- `sweep_summary.csv`;
- `analysis_report.md` with coarse-gate status per probability.

Stop on the first exception and preserve progress as failed. Do not retry or
adapt the matrix during execution.

## Task 4: Validation

Run locally:

1. `python -m py_compile scripts/diag_clean_load.py scripts/smoke_clean_load_diagnostic.py`;
2. `python scripts/smoke_clean_load_diagnostic.py`;
3. existing clean environment, graph, execution, and policy-RNG smokes that do
   not require Torch.

Run on the server with the configured Python:

1. focused load-diagnostic smoke;
2. clean graph and KaHyPar smoke;
3. one baseline and one stress diagnostic cell using 30 arrival slots and up to
   100 drain slots, both greedy and random;
4. verify finite metrics, drain forwarding, KaHyPar status, artifact integrity,
   no traceback, and no residual worker/eval process.

## Task 5: Publish and synchronize

Commit the implementation separately from the design/plan, push
`zrj_3_static_ue`, and fast-forward the server repository to the exact GitHub
HEAD. Preserve local untracked `docs/session_handoff_phase4.md` and `runs/`, plus
all historical server `logs/` and `runs/`.

## Task 6: Launch coarse scan

Create new timestamped result and log roots. Freeze the manifest with:

- arrival probabilities `0.0145,0.029,0.0435,0.058,0.087`;
- seeds `4242,4243`;
- policies `greedy,random`;
- 200 arrival slots;
- 500 maximum drain slots;
- current data/computation ranges;
- fixed hover movement and full KaHyPar hyperedges.

Launch the single sequential sweep command in `tmux` when available, otherwise
with `nohup`. Record PID/session, command, Git HEAD, result/log paths, and initial
progress. After confirming the process is alive and the manifest is fixed, stop
active monitoring and return control to the user as requested.

## Completion boundary

This implementation turn is complete when the coarse scan is safely running and
its launch evidence has been reported. Result analysis and the adaptive formal
1000-slot manifest occur only after the user reports that the coarse scan has
finished.
