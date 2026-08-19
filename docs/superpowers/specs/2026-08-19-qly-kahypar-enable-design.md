# QLY KaHyPar Enablement Design

## Goal

Enable KaHyPar partition hyperedges in the `qly` branch using the same worker
integration and runtime parameters already used by `zrj_3multisample`, while
keeping the Windows development path usable when KaHyPar is unavailable.

## Scope

- Declare `kahypar==1.3.7` as a Linux-only dependency.
- Set `ENABLE_KAHYPAR_PARTITION_HYPEREDGES` to `True` in `config.py`.
- Keep the existing worker, INI file, timeout, seed, epsilon, cache, and degraded
  execution behavior unchanged because the two branches already share that
  implementation.
- Do not change HGNN architecture, reward design, training hyperparameters, or
  the `zrj_3multisample` branch.

## Platform Behavior

Linux training environments install `kahypar==1.3.7` and attempt partitioning
when the existing graph-update conditions are met. Windows does not install the
Linux-only package; the existing worker failure/degraded path remains available
for local development, but formal KaHyPar experiments run on the Linux server.

## Verification

On `/data2/zrj2025/HyperUAV-qly`:

1. Confirm `kahypar==1.3.7` imports in the training interpreter.
2. Run `scripts/smoke_clean_graph.py` and `scripts/smoke_clean_end_to_end.py`.
3. Run one GPU PPO update with the HGNN and decision critic configuration.
4. Require evidence of actual partition execution: a successful KaHyPar status
   and a nonzero partition-hyperedge count in at least one observed slot.
5. Confirm the original `/data2/zrj2025/HyperUAV` checkout remains on
   `zrj_3multisample`.

The separate reward smoke mismatch is outside this change and remains reported
as an independent test/configuration issue.
