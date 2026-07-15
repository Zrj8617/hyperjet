# Offloading Checkpoint Isolation Execution Plan

Date: 2026-07-15
Design: `docs/superpowers/specs/2026-07-15-offloading-checkpoint-isolation-design.md`

## Objective

Run and audit the pre-registered 70-cell forced-hover checkpoint isolation without changing repository training or evaluation behavior. The experiment compares deterministic actor offloading at episodes 100, 300, 600, and 1000 for model seeds 42, 86, and 1042 against one paired `random_hash` reference and one diagnostic `greedy_eft_teacher` reference over five common environment seeds.

## Task 1: Local contract verification

1. Confirm branch and HEAD.
2. Confirm the only untracked user artifacts remain `docs/session_handoff_phase4.md` and `runs/`.
3. Inspect the eval CLI and policy selector to verify:
   - `--freeze-movement` reaches the deterministic movement selector;
   - `actor_argmax`, `random_hash`, and `greedy_eft_teacher` are available;
   - checkpoint provenance controls completed-DAG weight, detach mode, and UE mobility;
   - every output records policy and movement provenance.
4. Run local syntax checks and available no-GPU smoke tests for the eval entrypoint, policy selector, runner aggregation, and RNG pairing.
5. Do not edit repository source unless a failed invariant demonstrates a concrete defect. Any such defect requires a separately documented minimal change before server evaluation.

## Task 2: Publish the design baseline

1. Keep the design commit separate from any later implementation commit.
2. Push branch `zrj_3_static_ue`.
3. Verify the pushed commit hash.

## Task 3: Read-only server preflight

Run only under `/data2/zrj2025`:

1. Verify repository branch, HEAD, worktree status, and remote tracking state.
2. Locate the exact episode-100, 300, 600, and 1000 checkpoint paths for model seeds 42, 86, and 1042.
3. Verify every checkpoint is readable and record its size and SHA-256 digest.
4. Verify Python, PyTorch, CUDA, GPU availability, KaHyPar import/configuration, free disk space, and whether `tmux` is installed.
5. Check for active HyperUAV training or evaluation processes. Do not terminate anything.
6. Inspect existing result roots only to confirm provenance and checkpoint mapping. Do not modify historical `runs/` or `logs/`.

Stop before launch if branch/HEAD, checkpoint set, runtime, GPU availability, or disk space is inconsistent with the design.

## Task 4: Pairing and smoke gate

1. Run the existing RNG-pairing and offloading-policy smoke tests with the server Python.
2. Create an ephemeral initial-state probe under the new experiment output root, not in repository source.
3. For each common environment seed, construct the evaluation environment through the existing helper and record a stable fingerprint covering initial UAV and UE state, mobility configuration, hotspot count, and relevant RNG state.
4. Assert that policy selection does not change the initial fingerprint.
5. Run one disposable forced-hover eval cell and verify:
   - `movement_frozen=true`;
   - hover ratio is 1;
   - mean displacement is 0 within tolerance;
   - policy and checkpoint provenance are correct;
   - no invalid assignment or KaHyPar degradation occurs.

Do not launch the matrix if any assertion fails.

## Task 5: Materialize an immutable experiment manifest

Create a new timestamped root:

`/data2/zrj2025/HyperUAV/runs/phase5_checkpoint_offloading_isolation_<timestamp>`

and a matching log root:

`/data2/zrj2025/HyperUAV/logs/phase5_checkpoint_offloading_isolation_<timestamp>`

The manifest must record:

- repository HEAD and branch;
- Python, Torch, CUDA, GPU, and KaHyPar provenance;
- all checkpoint paths, sizes, and digests;
- model seeds `42, 86, 1042`;
- checkpoint episodes `100, 300, 600, 1000`;
- environment seeds `4242..4246`;
- policies and the fixed reference checkpoint;
- the exact 70 commands in execution order;
- initial-state fingerprints;
- output and log paths;
- creation time and selected GPU.

The manifest is fixed before the first matrix cell. Results must not alter it.

## Task 6: Launch the 70-cell matrix

1. Generate a run-local launcher below the new result root.
2. Execute cells sequentially on one selected GPU.
3. Use a named `tmux` session if available; otherwise use `nohup` and record the launcher PID and child Python PID.
4. Each command calls `scripts/eval_clean_mainline.py` directly with one checkpoint, one environment seed, one policy, `--freeze-movement`, 200 arrival slots, 500 maximum drain slots, and a unique output directory.
5. Stop on the first nonzero exit. Never silently retry, skip, substitute, or overwrite a cell.
6. Preserve all historical server `runs/` and `logs/`.

## Task 7: Monitor and audit

Monitor without changing the running process. At completion verify:

1. exactly 70 unique cells exist and all completed;
2. every cell matches the manifest;
3. all cells report forced movement, full hover, and zero displacement;
4. all assignments are valid and all required metrics are finite;
5. there are no tracebacks or unexplained KaHyPar degraded slots;
6. no evaluation process remains;
7. historical outputs were not overwritten.

If a cell fails, preserve its artifacts, diagnose the exact failure, and present the evidence before deciding whether a rerun is scientifically valid.

## Task 8: Aggregate and classify

Produce cell-level JSONL and summary tables with paired actor-minus-random deltas for completion, flowtime, and backlog. Include entropy, top1-top2 margin, estimated regret, actor/greedy agreement, queue wait, transfer time, and provenance.

Classify the result only under the design's pre-registered rules:

- reproducible weak ranking;
- no reproducible offloading learning;
- seed-specific degradation;
- ambiguous first stage.

Align seed1042 checkpoint quality with its historical hover trajectory as temporal association only. Do not infer movement causality from correlation.

## Task 9: Decision boundary

- If the first stage is decisive, design the evidence-supported next intervention before changing training.
- If it is ambiguous, prepare the pre-registered 160-cell expansion manifest; do not launch it without reporting why expansion is necessary.
- Do not revive EFT distillation or the vetoed unit-normalized counterfactual formulation without new evidence.
- Do not start the three formal 1000-episode runs until a repair has passed a separate short deterministic gate.

## Completion evidence

This diagnostic phase is complete only when the design commit is published, the valid 70-cell matrix is fully audited, paired summaries are produced, and one pre-registered interpretation is reported with paths and provenance.
