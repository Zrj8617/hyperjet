# Offloading Counterfactual Critic Implementation Plan

Design: `docs/superpowers/specs/2026-07-14-offloading-counterfactual-critic-design.md`

## Task 1: Pure action-value model and counterfactual math

Files:

- add `marl_models/mappo/clean_offloading_action_value.py`;
- add `scripts/smoke_clean_offloading_counterfactual.py`.

Steps:

1. Add the zero-output-initialized candidate action-value MLP.
2. Add pure helpers for masked policy expectation, selected-minus-expected counterfactual values, and stable population normalization.
3. Test masks, one/no candidate, zero initialization, differing per-action values, detach behavior, shape failures, and finite checks.

## Task 2: PPO updater integration

Files:

- modify `marl_models/mappo/clean_trainer.py`;
- update existing PPO/trainer/server smoke scripts where module construction is required.

Steps:

1. Add optional action-value critic to `CleanTrainingModules` and configuration coefficients to `CleanPPOUpdateConfig`.
2. Validate coefficient pairs and module presence at updater construction.
3. In a two-pass loss calculation, build detached Q inputs, train selected Q values on normalized slot advantages, compute detached normalized counterfactual values, and supply the combined advantage only to offloading PPO.
4. Add Q loss to total loss, optimizer parameters, global clipping, update stats, and diagnostics.
5. Prove the Q loss has zero HGNN gradient and disabled mode is numerically identical.

## Task 3: Checkpoint, CLI, resume, and provenance

Files:

- modify `scripts/train_clean_mainline.py`;
- modify `marl_models/mappo/clean_trainer.py` checkpoint helpers;
- modify `scripts/eval_clean_mainline.py`;
- modify checkpoint/resume/eval smoke scripts.

Steps:

1. Add the two non-negative CLI coefficients and reject half-enabled pairs.
2. Conditionally construct the Q critic before optimizer creation.
3. Persist and restore enabled Q state; retain exact disabled legacy compatibility.
4. Reject resume coefficient mismatches and missing enabled state.
5. In eval, inherit both coefficients, conditionally instantiate/load the Q critic for provenance validation, but continue selecting actions only with the actor.
6. Add both fields to config, summaries, metrics, and checkpoint-derived eval output.

## Task 4: Diagnostics and plotting

Files:

- modify `marl_models/mappo/clean_trainer.py`;
- modify `scripts/plot_clean_metrics.py`;
- update diagnostics-neutrality and plotting smoke scripts.

Steps:

1. Emit Q target/value statistics, explained variance, legal Q spread, raw/normalized counterfactual statistics, effective sample count, Q gradient norms, and coefficient provenance.
2. Preserve existing metric names and disabled-mode behavior.
3. Add plotting support without requiring Q fields in legacy logs.

## Task 5: Local verification and focused commit

Steps:

1. Run syntax compilation with output outside tracked paths or in a verified temporary directory.
2. Run the new focused smoke.
3. Run all affected clean PPO, training-loop, checkpoint, resume, eval, detach, diagnostics-neutrality, plotting, server-Torch, end-to-end, feature, and gate smokes using the bundled Torch runtime.
4. Run `git diff --check`, inspect the complete diff, and verify only `docs/session_handoff_phase4.md` and `runs/` remain untracked.
5. Commit the plan, implementation, and tests as one focused implementation commit, then push `zrj_3_static_ue`.

## Task 6: Server smoke, short validation, and deterministic gate

Steps:

1. Report the exact synchronization and smoke commands, paths, GPU use, and output roots before execution.
2. Synchronize the pushed commit to `/data2/zrj2025/HyperUAV` without touching old `logs/` or `runs/`.
3. Run server Torch/KaHyPar and enabled/disabled checkpoint smokes.
4. Launch the nine 100-episode short-training cells in persistent tmux under new roots, using the approved coefficients and unchanged remaining protocol.
5. Monitor to completion and verify artifacts, processes, errors, gradients, entropy, movement, critic, and KaHyPar.
6. Run 45 paired ep100 deterministic evaluation cells and apply every predeclared gate condition from the design.
7. If no arm passes, stop formal progression and diagnose the failed action-value formulation. If an arm passes, select it exactly by the design rule.

## Task 7: Three formal runs and final audit

Steps:

1. Present exact commands, GPU allocation, persistence, run/log roots, and resource checks.
2. Launch seeds 42, 86, and 1042 with the one selected configuration for 1000 episodes × 200 slots.
3. Monitor until all three terminate normally; do not silently restart or alter parameters.
4. Audit 1000 terminal episodes, global slot 200000, checkpoint schedule, `latest.pt`, summaries, provenance, NaN/Inf, KaHyPar status, and residual processes.
5. Report all evidence and mark the Goal complete only if every active-Goal requirement is proven.
