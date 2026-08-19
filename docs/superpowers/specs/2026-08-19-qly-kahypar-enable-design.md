# QLY KaHyPar Enablement Design

## Goal

Enable KaHyPar partition hyperedges explicitly for Linux formal runs in the
`qly` branch, using the same worker integration and runtime parameters already
used by `zrj_3multisample`, while keeping the Windows development path usable
without KaHyPar.

## Scope

- Declare `kahypar==1.3.7` as a Linux-only dependency.
- Keep `ENABLE_KAHYPAR_PARTITION_HYPEREDGES` disabled by default.
- Add an explicit `--enable-kahypar` control to the clean training and
  evaluation entrypoints. The flag enables partition hyperedges before graph
  builders are created and is recorded in run configuration output.
- Align the graph smoke with its intended condition so that it explicitly
  enables KaHyPar while testing partition-attempt behavior.
- Keep the existing worker, INI file, timeout, seed, epsilon, cache, and degraded
  execution behavior unchanged because the two branches already share that
  implementation.
- Do not change HGNN architecture, reward design, training hyperparameters, or
  the `zrj_3multisample` branch.

## Platform Behavior

Linux training environments install `kahypar==1.3.7`. Formal KaHyPar runs pass
`--enable-kahypar` and attempt partitioning when the existing graph-update
conditions are met. Windows skips the Linux-only dependency and omits the flag,
so QLY can continue local development without silently claiming a complete
four-type hypergraph experiment.

The `qly` branch also receives a repository rule that blocks branch deletion
and non-fast-forward pushes while still allowing ordinary collaborator pushes.
Repository administrators retain bypass access.

## Verification

On `/data2/zrj2025/HyperUAV-qly`:

1. Confirm `kahypar==1.3.7` imports in the training interpreter.
2. Run `scripts/smoke_clean_graph.py`, the KaHyPar worker smoke, and
   `scripts/smoke_clean_end_to_end.py`.
3. Run one GPU PPO update with the HGNN and decision critic configuration plus
   `--enable-kahypar`.
4. Require evidence of actual partition execution: a successful KaHyPar status
   and a nonzero partition-hyperedge count in at least one observed slot.
5. Confirm the original `/data2/zrj2025/HyperUAV` checkout remains on
   `zrj_3multisample`.
6. Confirm a normal update to `qly` remains allowed while the active repository
   rule blocks force pushes and deletion for non-admin collaborators.

The separate reward smoke mismatch is outside this change and remains reported
as an independent test/configuration issue.
