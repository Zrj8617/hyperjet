# Lagged DAG-Outcome Residual-Q Implementation Plan

Date: 2026-07-15
Design: `docs/superpowers/specs/2026-07-15-lagged-dag-outcome-residual-q-design.md`

## Task 1: Add behavior-neutral controls and provenance

Files:

- `scripts/train_clean_mainline.py`
- `scripts/eval_clean_mainline.py`
- `marl_models/mappo/clean_trainer.py`
- existing CLI/checkpoint/eval smokes

Add the four v2 controls, strict validation, v1/v2 mutual exclusion, config/checkpoint/resume propagation, and deterministic-eval reporting. Prove old checkpoints resolve to disabled defaults and all-zero controls create baseline topology and optimizer parameters.

Build the enabled auxiliary Q module without advancing the CPU Torch RNG stream,
and test the RNG state for exact equality before and after construction.

## Task 2: Implement target and tracker primitives

Files:

- new `marl_models/mappo/clean_lagged_residual_q.py`
- new `scripts/smoke_clean_lagged_residual_q.py`

Implement pure target arithmetic, bounded finite validation, pending action records without old log-probability, completed/censored finalization, finalized-sample consumption, and episode clearing. Test these primitives before trainer integration.

## Task 3: Capture action-time provenance

Files:

- `marl_models/mappo/clean_offloading_actor.py`
- `marl_models/mappo/clean_slot_orchestrator.py`
- `scripts/train_clean_mainline.py`

Record DAG ID, assignment time, full detached candidate features, detached global context, selected estimated finish, and selected incremental delay. Preserve existing PPO records and RNG behavior when v2 is disabled.

## Task 4: Integrate episode tracker

File:

- `scripts/train_clean_mainline.py`

Create the tracker only when v2 is enabled. Register after slot commit, resolve completed DAGs every slot, carry pending actions only across rollouts inside the episode, censor at episode end, pass newly finalized samples to the next updater call, and assert the tracker is empty after terminal clearing.

## Task 5: Implement lagged Q regression and frozen correction

Files:

- `marl_models/mappo/clean_lagged_residual_q.py`
- `marl_models/mappo/clean_trainer.py`

Use weighted smooth-L1 on stored selected inputs. Precompute detached clamped candidate corrections once before PPO epochs. Keep historical samples out of `_ppo_action_loss`. Verify Q loss has no direct gradient to HGNN, actors, or centralized critic.

## Task 6: Checkpoint and eval compatibility

Files:

- `marl_models/mappo/clean_trainer.py`
- `scripts/train_clean_mainline.py`
- `scripts/eval_clean_mainline.py`

Use an explicit v2 Q checkpoint key, reject configuration mismatch, retain legacy defaults, and instantiate/load the v2 module for deterministic eval provenance without using Q for action selection.

## Task 7: Diagnostics

Files:

- `marl_models/mappo/clean_trainer.py`
- `scripts/train_clean_mainline.py`

Log target coverage/censoring, Q loss/EV/spread, frozen correction scale/saturation, pending/finalized counts, gradient isolation, and terminal tracker state. Keep existing v1 diagnostic fields unchanged when v1 is selected.

## Task 8: Static and server validation

1. Local: AST/compile-source checks, `git diff --check`, staged-scope review only.
2. Server: sync the implementation commit under `/data2/zrj2025/HyperUAV`.
3. Run all focused CPU smokes with `/data2/zrj2025/.conda/envs/uav322/bin/python`.
4. Run existing eval, policy, RNG, checkpoint, and default-neutrality smokes.
5. Run one tiny GPU baseline smoke and one tiny GPU v2 smoke on a free GPU.
6. Audit finite diagnostics, action counts, gradient isolation, checkpoint reload, and no residual processes.

## Task 9: Short gate

Launch three v2 runs, seeds 42/86/1042, 100 episodes × 200 slots, in persistent server sessions. After completion run normal deterministic and forced-hover common-scene evaluation, plus one forced-hover random reference. Apply only the pre-registered design criteria.

Do not tune coefficients or start formal runs if the first v2 arm fails.

## Task 10: Formal boundary

Only after a passing short gate, launch seeds 42/86/1042 for 1000 episodes × 200 slots with the approved v2 configuration. Audit all runs and deterministic eval before marking the Goal complete.
