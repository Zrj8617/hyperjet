# Reward calibration and critic-to-HGNN detach experiment design

## Scope

Add two independent clean-mainline training controls for the next Phase 4
experiment:

1. a run-level completed-DAG reward weight, with `16.0` used by the calibrated
   experiment and the existing `2.0` retained as the default baseline;
2. an optional critic-to-HGNN detach boundary that prevents value loss from
   updating the shared HGNN while preserving actor-to-HGNN gradients.

This change must not alter task load, environment dynamics, policy action
spaces, model parameter shapes, KaHyPar behavior, PPO clipping, entropy
coefficients, or the existing `w_c=2/shared-HGNN` default.

## Training interface

Add these CLI controls to `scripts/train_clean_mainline.py`:

- `--completed-dag-weight FLOAT`, defaulting to
  `config.REWARD_COMPLETED_DAG_WEIGHT` (`2.0`);
- `--detach-critic-hgnn`, a boolean flag that defaults to `False`.

Reject a completed-DAG weight that is non-finite or negative. Record both
resolved values in `config.json`, every checkpoint's config snapshot, the
training result, and the relevant PPO diagnostics. Run names remain explicit so
the experiment directory identifies `w16_shared` or `w16_detach` without
parsing checkpoint internals.

## Reward dependency injection

Do not mutate the process-global `config.REWARD_COMPLETED_DAG_WEIGHT` at runtime.
`Env` accepts an optional completed-DAG reward weight and passes it to
`CleanMetricsTracker`; the tracker stores the resolved value for its lifetime
and uses it when calculating `completed_dag_bonus`. Resetting an environment
does not re-resolve or change that value. Existing call sites that do not pass a
value retain the configured baseline of `2.0`.

This makes w2 and w16 environments safe to construct in the same Python process
and prevents smoke tests, evaluation, or later runs from inheriting a prior
experiment's global mutation.

## Critic-to-HGNN detach boundary

Add `detach_critic_hgnn: bool = False` to `CleanPPOUpdateConfig` and propagate
the training CLI value into the updater.

For each PPO loss recomputation, calculate one shared `task_embeddings` tensor
from the HGNN. The movement and offloading actors always consume that original
tensor. The critic consumes:

- `task_embeddings` in the default shared mode;
- `task_embeddings.detach()` in the detach arm.

The critic network itself remains fully trainable. Only the value-loss path into
HGNN parameters is cut. Actor losses and actor entropy terms continue to update
the HGNN. The optimizer parameter set, checkpoint tensors, and evaluation model
topology do not change.

Apply the same boundary to rollout-time critic input assembly for conceptual
consistency and reduced graph retention, although rollout values are converted
to detached scalars and do not perform backward. All public encoding helpers
default to shared mode so existing callers remain unchanged.

The existing loss-specific HGNN decomposition is retained. In detach mode its
value-loss HGNN norm must be `0.0` and cosine must be `None`; actor HGNN norm can
remain non-zero. Add a diagnostic field that states whether the critic-HGNN
boundary was detached for the update.

## Checkpoint resume and evaluation

On resume, compare the requested completed-DAG weight and detach flag with the
checkpoint's saved training CLI values. Treat old checkpoints without these
fields as `w_c=2.0` and shared mode. Reject a mismatch with a clear error instead
of silently changing the reward objective or gradient topology mid-run. Resolve
the saved values from `payload["config"]["cli"]`; compare the detach flag exactly
and compare the reward weight with `math.isclose` using zero relative tolerance
and `1e-12` absolute tolerance.

Deterministic evaluation reads the saved values for reporting. It creates its
environment with the checkpoint's completed-DAG weight so reward-component logs
match training. The detach flag changes no forward computation or model shape,
but is still reported in `config.json` and `eval_summary.json` for provenance.
Completion and flowtime evaluation remain policy/environment measurements and
are not reweighted. Evaluation exposes no CLI override for either training
control: both values are checkpoint-derived, with the same legacy defaults as
resume validation.

## Smoke and correctness coverage

Add focused Torch and reward smoke coverage for:

- baseline environments still use `w_c=2.0`;
- for identical completed DAGs, the w16 bonus minus the w2 bonus is exactly
  `14 * completed_dag_count`, with all other reward components equal;
- explicit shared mode matches the old/default update behavior;
- value-only backward in shared mode produces HGNN and critic gradients;
- value-only backward in detach mode produces zero HGNN gradient and non-zero
  critic gradient;
- actor loss in detach mode still produces a non-zero HGNN gradient;
- loss-specific diagnostics report value HGNN norm `0.0` and cosine `None` in
  detach mode;
- checkpoint resume accepts matching values and rejects either mismatch;
- config snapshots and deterministic evaluation provenance contain both values.

Run the existing diagnostic-neutrality, training-loop, server-Torch, reward,
entrypoint, RNG-pairing, and short-training smokes after the focused tests.

## Commit and rollout structure

Use independent commits:

1. design specification;
2. run-level completed-DAG reward injection, provenance, resume validation, and
   reward smoke;
3. critic-to-HGNN detach plumbing, diagnostics, and Torch smoke.

After local checks, push `zrj_3`, fast-forward the server, and run all Torch
smokes there. Do not start the four new training jobs until both implementation
commits and the server short-training smoke pass.

The subsequent controlled matrix is:

- existing `w2/shared` anchor, reused rather than rerun;
- `w16/shared`, seeds `42` and `1042`;
- `w16/detach`, seeds `42` and `1042`.

All new jobs use learned movement, 100 episodes, 200 slots per episode, rollout
horizon 128, 3 PPO epochs, learning rate `3e-4`, gamma `0.99`, GAE lambda
`0.95`, entropy coefficient `0.01`, value coefficient `0.5`, max grad norm
`0.5`, checkpoint interval 20, and the now-enabled KaHyPar partition edges.

## Non-goals

- No global default change from `w_c=2.0` to `16.0`.
- No value-coefficient sweep.
- No separate critic HGNN or new model parameters.
- No optimizer, gradient-clip, advantage, entropy, or reward-time-scale change.
- No claim that detach is a final architecture; it is a controlled diagnostic
  arm motivated by the measured value-to-actor HGNN gradient imbalance.
