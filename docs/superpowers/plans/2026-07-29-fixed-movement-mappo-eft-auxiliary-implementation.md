# Fixed-Movement MAPPO + Per-Decision EFT Auxiliary Implementation Plan

## 1. Preserve decision-k EFT history

Files:

- `marl_models/mappo/clean_offloading_actor.py`
- `marl_models/mappo/clean_slot_orchestrator.py`
- `scripts/train_clean_mainline.py`

Tasks:

1. Add a complete candidate EFT vector to the action and rollout records.
2. Construct it from the existing decision-k `estimates` before reservation.
3. Copy arrays at the actor-to-rollout boundary and make historical ownership
   explicit.
4. Keep original PPO action/log-prob fields and all GraphSnapshot fields
   unchanged.

Verification:

- mutation of environment/actor source values cannot modify rollout history;
- vector, mask, candidate UAV ids, and feature row counts match;
- the selected EFT agrees with the selected vector entry.

## 2. Implement a pure auxiliary objective and schedule

Files:

- `marl_models/mappo/clean_trainer.py`
- optionally a small dedicated helper under `marl_models/mappo/` if needed to
  keep the trainer readable.

Tasks:

1. Add default-off config for frozen regret scale and lambda schedule.
2. Implement the approved update-indexed lambda function.
3. While `_loss` rebuilds current offloading distributions, independently
   resample auxiliary actions from each nontrivial decision.
4. Compute detached EFT rewards, exact detached baseline, advantages, and
   mean decision loss.
5. Add the weighted loss to the existing total loss.
6. Expose loss, counts, sampled regret, agreement, entropy, invalid actions,
   and finite diagnostics.
7. Measure PPO-only and EFT-only gradients on the same encoder+scorer target
   set without writing `.grad`, then retain the existing single optimizer step.

Verification:

- no environment estimator is referenced by update code;
- resampling occurs per PPO epoch;
- old PPO action cannot select the auxiliary reward;
- critic and movement direct auxiliary gradients are zero;
- encoder/scorer gradients are finite;
- zero/single-candidate decisions do not enter the denominator.

## 3. Freeze movement and initialize treatments fairly

Files:

- `scripts/train_clean_mainline.py`
- a new diagnostic launcher, expected
  `scripts/run_fixed_movement_mappo_eft_aux_gate.py`

Tasks:

1. Require `--freeze-movement`, `--task-encoder mlp`, and disabled
   counterfactual/lagged-Q options in diagnostic mode.
2. Set movement parameters `requires_grad=False` before optimizer creation.
3. Strictly import matching-seed bandit encoder/scorer weights for B/C only.
4. Build every MAPPO optimizer fresh after initialization.
5. Capture and assert A/D and B/C initialization hashes.
6. Stop runs by outer update count and save the required update checkpoints.

Verification:

- forced-hover actions only;
- movement rollout/action count and movement gradient are zero;
- bandit optimizer state is absent;
- checkpoint seed/schema/dataset checksum/shape mismatches fail loudly;
- pairwise initialization hashes match as specified.

## 4. Add focused smokes

Expected new files:

- `scripts/smoke_clean_mappo_eft_auxiliary.py`
- `scripts/smoke_fixed_movement_mappo_eft_gate.py`

Coverage:

1. immutable EFT history and decision-k alignment;
2. no update-time environment EFT call;
3. current-policy resampling and old-action independence;
4. mask safety and exact baseline detach;
5. encoder/scorer-only direct auxiliary gradient;
6. one optimizer step per PPO epoch;
7. trivial-decision exclusion;
8. lambda values at updates 0, 8, 9, 19, 20, 21, and 29;
9. A/B/C/D initialization mapping and strict checkpoint import;
10. finite logits/loss/gradients and frozen movement.

Run existing related smokes for clean PPO, slot orchestration, offloading actor,
multisample behavior, arrival funnel, and contextual bandit.

## 5. Calibrate lambda on the server

1. Push only after local/schema smokes pass.
2. Fast-forward the server repository without reset/clean and preserve
   `logs/` and `runs/`.
3. Run a two-update seed-42 pilot on an available GPU.
4. Confirm checkpoint import, finite behavior, zero illegal actions, zero
   movement gradient, and comparable target-set gradient measurements.
5. Choose and freeze one `lambda_0` from the measured ratio.

No scientific metric may be used to tune lambda.

## 6. Run the formal diagnostic

1. Launch A/B/C/D for seeds 42, 86, and 1042, each for exactly 30 outer
   updates.
2. Use forced hover, MLP, identical settings, paired run/evaluation seeds,
   the frozen scale and lambda, and fresh optimizers.
3. Save control logs, per-update JSONL, required checkpoints, run manifest,
   initialization hashes, and resource/finite diagnostics.
4. Launch short paired fixed-scenario checkpoint-only closed-loop evaluation.
5. Stop active polling and schedule one low-frequency completion check.

## 7. Completion audit and report

Prove:

- all 12 variants completed 30 updates;
- required checkpoints and paired initialization hashes exist;
- no traceback, NaN/Inf, OOM, illegal action, split/checkpoint mismatch, or
  movement gradient occurred;
- lambda schedule and frozen scale match every run;
- arrival funnel and active-DAG admission effects are reported separately;
- three-seed mean/std and paired closed-loop results are complete.

Classify the result without claiming EFT is a formal reward or that delayed
reward is the unique cause.
