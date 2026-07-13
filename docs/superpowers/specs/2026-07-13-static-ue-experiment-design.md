# Static UE Shared-HGNN Experiment Design

## Scope

Add an opt-in clean-mainline experiment control that freezes each UE at its
episode-initial position while preserving the established moving-UE behavior by
default. The change is isolated on branch `zrj_3_static_ue`; branch `zrj_3`
remains at `e9ecb50` as the rollback point.

The formal experiment changes only UE mobility. It retains completed-DAG reward
weight 16, shared critic-to-HGNN gradients, enabled KaHyPar partition hyperedges,
and all other settings from the completed Phase 4 shared runs.

## Environment semantics and RNG boundary

Add `freeze_ue_mobility=False` to the environment. Every episode still samples
its hotspot and UE initial positions normally. The hotspot is static within an
episode. When freezing is enabled, every UE remains at its sampled initial
position for all slots in that episode.

`UE.update_position` receives a commit control. It always executes the existing
Gauss-Markov noise sampling, candidate-position calculation, and boundary
reflection path. In fixed mode it discards the candidate instead of writing it
to `UE.pos`. This preserves the mobility subsystem's local NumPy draw count,
including boundary cases.

This is deliberately a local guarantee only: it aligns episode-local mobility
random-number consumption. It does **not** guarantee that fixed and moving runs
retain the same global NumPy state for the full trajectory after DAG arrivals,
queue state, or service decisions diverge. Strict bitwise trajectory pairing
would require independent RNG substreams and is out of scope.

## Training, checkpoint, and evaluation controls

Add `--freeze-ue-mobility` to the clean training entrypoint and record the
resolved boolean in `config.json`, JSONL diagnostics/episode records,
`run_summary.json`, and checkpoint configuration provenance.

Resume validation treats checkpoints created before this field existed as
moving mode (`False`). A requested mode that differs from the checkpoint mode
is rejected before training resumes.

Deterministic evaluation derives the UE mobility mode from checkpoint
provenance. If an explicit evaluation override is exposed, it must be tri-state:
absence means inherit, while an explicit value that conflicts with the
checkpoint is rejected rather than silently overridden.

## Hotspot covariate

Record `initial_hotspot_ue_count` for every episode in both fixed and moving
modes. The count is measured after episode initialization and before the first
mobility update, using the episode's static hotspot. Recording it in both modes
makes the initial spatial-load covariate directly comparable.

## Compatibility and failure behavior

Default behavior is bitwise-neutral for callers that do not enable the new
control. Existing checkpoint tensor layouts do not change. Legacy checkpoint
configuration resolves the missing flag to `False`; malformed non-boolean
values fail with a clear error.

## Verification

Add behavior-local smoke coverage for:

- unchanged default movement behavior;
- fixed UE positions across multiple updates while mobility noise is consumed;
- a forced boundary-reflection case with equal one-call NumPy RNG states;
- training CLI/config/JSONL/summary/checkpoint provenance;
- legacy checkpoint compatibility and resume mismatch rejection;
- deterministic evaluation inheritance and conflict rejection;
- `initial_hotspot_ue_count` presence in both moving and fixed modes.

Run the affected NumPy and Torch clean-mainline smoke suites locally, then push
`zrj_3_static_ue`, fetch and switch the server checkout to that branch, and run
the server Torch/KaHyPar short validation before formal launch.

## Formal runs

Launch two independent fixed-UE shared-HGNN runs:

- seed 42;
- seed 1042.

Each run uses 100 episodes, 200 arrival slots per episode, checkpoint interval
20, completed-DAG reward weight 16, critic-to-HGNN detach disabled, learned UAV
movement enabled, and the current KaHyPar configuration with all supported
hyperedge families enabled. Use a new static-UE run root and distinct run names
so the moving-UE formal results remain untouched.

The immediate launch acceptance criterion is that both processes initialize,
write valid configuration and JSONL records showing fixed UE mode, report active
KaHyPar partitioning, and continue without startup errors. Completion analysis
is deferred until the user reports that the runs have finished.
