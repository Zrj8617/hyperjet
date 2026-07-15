# Lagged DAG-Outcome Residual-Q Design

Date: 2026-07-15
Branch: `zrj_3_static_ue`
Baseline commit: `cd6ef1d59bc03322929aaf15b362e9cd7966ea60`

## 1. Decision

Implement one default-disabled v2 experiment that adds an action-linked, delayed DAG-outcome residual-Q correction to the current offloading PPO advantage.

This is not a replacement for PPO, not an EFT behavior-cloning teacher, and not a reuse of stale actions in PPO. Historical actions supervise only an independent Q regressor. Only actions in the currently consumed rollout may enter the PPO ratio.

The existing v1 counterfactual implementation remains present, default-disabled, and mutually exclusive with v2.

## 2. Evidence that authorizes this experiment

The audited 70-cell forced-hover checkpoint isolation is stored at:

`/data2/zrj2025/HyperUAV/runs/phase5_checkpoint_offloading_isolation_20260715_111017`

Its audit passed with 70/70 cells, zero invalid assignments, zero KaHyPar degraded slots, full hover, zero movement displacement, and no residual evaluation process.

The forced-hover references were:

| policy | pooled arrival completion | weighted drain flowtime |
|---|---:|---:|
| random_hash | `0.568928` | `526.334 s` |
| greedy EFT | `0.798684` | `158.524 s` |

Checkpoint results established:

- seed 42 and seed 86 contain reproducible deterministic argmax rankings despite normalized entropy near one;
- seed 1042 is already below random at episode 100 and remains below random at episodes 300, 600, and 1000;
- seed 1042 therefore is not explained solely by a good offloading policy being destroyed after late movement collapse;
- reduced entropy is not the objective: seed 1042's growing top1-top2 margin accompanies poor ranking;
- the task load is sufficient for some seeds to learn a useful ranking, so increasing arrivals is not the first repair;
- greedy EFT is useful as a local baseline but was inconsistent in the earlier learned-movement system gate, so it is not a global training teacher.

The repair target is early, seed-stable offloading ranking.

## 3. Rejected alternatives

### 3.1 Increase completed-DAG reward

Rejected. `w_c=16` already repaired reward direction. The reward correlates positively with completion, and no current evidence supports increasing it.

### 3.2 Detach critic from HGNN

Rejected. The detach arm was implemented correctly and performed worse for both formal seeds. Large value gradients are not sufficient evidence that the actor is harmed.

### 3.3 Lower offloading entropy to force convergence

Rejected. Seeds 42 and 86 demonstrate useful deterministic rankings at high entropy. Seed 1042 demonstrates that amplifying a poor small bias can worsen deterministic behavior.

### 3.4 EFT distillation or pairwise EFT ranking

Rejected as the primary repair. EFT was inconsistent at system level under learned movement. It may be subtracted as a known local estimate but must not label the preferred action.

### 3.5 Re-enable v1 action-value counterfactual

Rejected. V1 trained every selected Q on the same slot advantage, produced Q explained variance near zero and tiny legal Q spreads, then forced those tiny differences to unit variance. Its short gate was vetoed.

### 3.6 Put historical actions back through PPO

Forbidden. No action from a previous consumed rollout may be passed to `_ppo_action_loss`, and no ordinary PPO ratio may be formed from a historical action after parameters have changed.

## 4. Causal scope

The v2 target does not claim a full counterfactual outcome for unselected UAVs. It learns the expected delayed residual for a candidate from outcomes of actions actually sampled by the stochastic behavior policy.

The current near-uniform sampling policy supplies broad candidate coverage. The learned Q may generalize across similar candidate states, but it is still an observational action-value estimate and may contain policy-distribution bias.

## 5. New controls

Add four training controls:

- `--offloading-lagged-q-coef` (`beta_lq`), default `0.0`;
- `--offloading-lagged-q-loss-coef` (`eta_lq`), default `0.0`;
- `--offloading-lagged-q-scale-seconds`, default `200.0`;
- `--offloading-lagged-q-censor-weight`, default `0.25`.

Validation:

- coefficients must be finite and non-negative;
- scale must be finite and strictly positive;
- censor weight must be finite and in `[0, 1]`;
- `beta_lq` and `eta_lq` must be enabled or disabled together;
- v1 and v2 may not be enabled in the same run;
- defaults must create no extra module, tracker, target, loss, or optimizer parameter.

When v2 is enabled, constructing its auxiliary Q module must save and restore the
CPU Torch RNG state. The Q initialization remains reproducible, but its mere
presence must not shift the actor's subsequent sampling stream before a learned
Q correction exists. This keeps paired baseline/v2 comparisons isolated at the
first action.

Recommended first arm:

- `beta_lq=0.25`;
- `eta_lq=0.5`;
- scale `200 s`;
- censor weight `0.25`.

No second coefficient arm is authorized before the first short gate is interpreted.

## 6. Action record additions

For every actually sampled offloading action, retain detached CPU copies of:

- episode index;
- slot index;
- task ID and DAG ID;
- assignment time in seconds;
- selected action and UAV ID;
- candidate mask and candidate UAV IDs;
- full candidate feature rows at action time;
- centralized critic context at action time;
- selected estimated finish time;
- selected estimated incremental delay.

The lagged outcome sample type must intentionally omit old log-probability. This makes accidental reuse in PPO structurally harder.

The existing current-rollout PPO record continues to retain old log-probability exactly as before.

## 7. Outcome tracker

Create one tracker per training episode.

After each slot commits:

1. register newly sampled actions;
2. inspect their DAG jobs;
3. finalize every pending action whose DAG now has `completed=true` and a finite `return_complete_time`;
4. retain unresolved actions across PPO rollout boundaries inside the same episode.

At episode termination or truncation:

1. finalize all completed DAG actions normally;
2. finalize all remaining actions as right-censored samples;
3. expose all finalized samples not yet consumed by a Q update;
4. clear the episode tracker after the final update.

Pending samples never cross an environment reset.

Existing resume semantics restart from a new episode. Therefore a resumed v2 run starts with an empty tracker. The checkpoint must record this policy as:

`lagged_q_resume_pending_policy=discard_restarted_episode`

All finalized samples are consumed at the next update and are not retained as a long replay buffer.

## 8. Target

For an action assigned at physical time `t_a`, let:

- `d_eft` be the selected candidate's estimated incremental delay at assignment;
- `T_dag` be the DAG's actual `return_complete_time` when completed;
- `T_end` be the physical episode end time for a censored DAG;
- `S` be `offloading_lagged_q_scale_seconds`.

For a completed DAG:

`residual_seconds = (T_dag - t_a) - d_eft`

`target = -tanh(residual_seconds / S)`

For a censored DAG:

`residual_lower_bound = max((T_end - t_a) - d_eft, 0)`

`target = -tanh(residual_lower_bound / S)`

Censored samples receive `offloading_lagged_q_censor_weight`; completed samples receive weight `1`.

Interpretation:

- EFT removes the known local finish estimate from the label;
- Q learns delayed downstream residual, including later queueing, cross-UAV dependencies, and sink return effects that actually materialize;
- higher Q is better;
- the fixed transform bounds every target to `[-1, 1]`;
- no batch or rollout standard-deviation normalization is applied.

The target is attached to an actually sampled action only. No target is invented for an unselected candidate.

## 9. Q loss

Reuse the small independent action-value MLP architecture, but select v2 behavior through explicit mode controls.

For each finalized sample, feed only its stored selected candidate feature row plus stored global context. Both are detached.

Use weighted smooth-L1 loss:

`L_lq = weighted_mean(smooth_l1(Q(stored_selected_input), target))`

The Q loss may update only Q parameters. It must have zero gradient to:

- HGNN;
- movement actor;
- offloading actor;
- centralized critic.

If no finalized samples are available for an update, `L_lq=0` and the update remains valid.

## 10. Current-rollout correction

At the start of each PPO update, before the first optimizer step:

1. recompute each current-rollout offloading candidate representation with the current pre-update HGNN and actor;
2. detach candidate inputs and global context;
3. evaluate Q for every legal candidate;
4. clamp Q values to `[-1, 1]` for correction only;
5. compute the behavior-policy expectation using pre-update actor probabilities;
6. compute:

`A_lq = Q(selected) - sum_a pi_behavior(a|s) Q(a)`

7. detach and freeze every `A_lq` for all PPO epochs in this update.

The offloading advantage becomes:

`A_off = A_slot + beta_lq * A_lq`

`A_slot` remains the normalized current-rollout slot GAE.

There is no unit-variance normalization of `A_lq`.

## 11. On-policy boundary

The implementation must make the following separation auditable:

| data | permitted use |
|---|---|
| current unconsumed rollout action | PPO ratio, entropy, frozen Q correction |
| finalized historical action outcome | Q regression only |
| unselected candidate in current rollout | Q baseline evaluation only |
| unselected historical candidate | no target and no loss |

Assertions/tests must prove:

1. lagged samples have no old log-probability field;
2. `_ppo_action_loss` receives exactly the current rollout action count;
3. Q regression can run with zero current rollout actions without invoking PPO;
4. changing a lagged target changes Q gradients but not actor/HGNN gradients directly;
5. frozen corrections are identical across PPO epochs within one update even while Q parameters change.

## 12. Optimizer and clipping

Keep the existing single Adam optimizer and global clipping for the first experiment. The Q loss uses detached stored inputs, so it does not directly update actor or HGNN parameters.

Record Q pre/post-clip gradients and the existing global clip scale as diagnostics. Do not introduce grouped clipping as a repair; Adam's approximate scale invariance makes it a diagnostic, not the current causal hypothesis.

## 13. Checkpoint and eval provenance

Checkpoint config and resume mismatch guards must include all four v2 controls.

When v2 is enabled, checkpoint the Q module under an explicit v2 key. A v2 checkpoint must not load into a v1 configuration or a disabled configuration.

Deterministic eval inherits and reports the v2 controls but does not use Q to choose actions. The trained offloading actor remains the evaluated policy.

Old checkpoints without v2 fields resolve to disabled defaults.

## 14. Diagnostics

Every update records:

- lagged pending count;
- newly finalized completed count;
- newly finalized censored count;
- Q training sample count and effective weighted count;
- completion-label coverage;
- censor fraction;
- target mean/std/min/max;
- selected Q mean/std;
- Q explained variance on finalized samples;
- legal Q spread mean/median/P90 for current actions;
- frozen correction mean/std/min/max;
- fraction of current actions with at least two legal candidates;
- Q loss;
- Q gradient norm;
- direct Q-loss gradient norms to Q, HGNN, actor, and centralized critic;
- current offloading PPO action count.

Episode-terminal logs also record unresolved actions before censoring and tracker count after clearing.

## 15. Default-off equivalence

With all v2 controls at defaults:

- module topology equals baseline;
- optimizer parameter identities equal baseline;
- RNG consumption equals baseline;
- rollout records and PPO losses equal baseline;
- checkpoint payload remains backward compatible;
- deterministic eval outputs equal baseline.

This requires an exact fixed-seed smoke, not a tolerance-only statistical claim.

## 16. Tests

Add or extend CPU smokes for:

1. CLI/default validation and v1/v2 mutual exclusion;
2. completed target arithmetic;
3. censored lower-bound target arithmetic and weight;
4. target bounds and finite-value rejection;
5. tracker registration, delayed completion, cross-rollout retention, censoring, and episode clearing;
6. no cross-episode task-ID collision;
7. lagged sample type contains no PPO log-probability;
8. weighted smooth-L1 loss and zero-sample loss;
9. zero Q initialization gives zero correction;
10. candidate-dependent Q gives candidate-dependent correction;
11. correction is frozen across PPO epochs;
12. no forced unit-variance normalization;
13. Q loss has zero HGNN/actor/critic gradient;
14. current PPO action count is unchanged by the number of lagged samples;
15. default-off exact baseline equivalence;
16. checkpoint save/load, mismatch refusal, legacy default, and eval provenance.

All runtime smokes run on the server Python because the local workspace has no NumPy/Torch environment.

## 17. Short training gate

Train exactly three v2 runs first:

- seeds `42, 86, 1042`;
- 100 episodes;
- 200 slots;
- rollout 128;
- PPO epochs 3;
- existing learning rate, reward, entropy, mobility, shared HGNN, and KaHyPar settings;
- recommended v2 coefficients only.

Use the existing default-disabled short baseline runs only if provenance matches; otherwise launch paired baseline runs under the new commit.

Evaluate every v2 episode-100 checkpoint on five common environment seeds under both:

- normal deterministic learned movement;
- forced-hover deterministic offloading isolation.

Also run paired `random_hash` under forced hover once per environment seed.

## 18. Gate criteria

The v2 arm passes only if all integrity checks pass and:

1. seed 1042 forced-hover actor improves over its historical episode-100 baseline on at least four of five scenes in the same primary direction;
2. seed 1042 forced-hover pooled completion exceeds forced-hover random and weighted flowtime is lower than random;
3. at least two of three seeds improve over their paired short baseline in normal deterministic evaluation by either at least `+0.03` pooled arrival completion or at least `10%` lower weighted flowtime, without crossing the material-regression boundary on the other metric;
4. neither seed 42 nor seed 86 loses more than `0.03` pooled forced-hover arrival completion or increases weighted forced-hover flowtime by more than `10%` versus its paired baseline;
5. no seed simultaneously increases terminal-20 training hover ratio by more than `0.20` and reduces terminal-20 mean displacement by more than `15 m` versus paired baseline;
6. across the final 20 training updates, finalized labels cover at least `25%` of sampled offloading actions and the weighted censored fraction is at most `0.75`;
7. across the final 20 training updates, median finalized-sample Q explained variance is greater than `0`, median legal Q spread is at least `0.005`, median frozen-correction standard deviation is at least `0.005`, and fewer than `5%` of legal Q values hit either correction clamp boundary;
8. there is no stale-ratio path, non-finite value, invalid assignment, KaHyPar degradation, or provenance mismatch.

For criteria 3 and 4, the material-regression boundary is the same `0.03` completion loss or `10%` flowtime increase. A run does not count as improved if it crosses either boundary.

If seed 1042 remains below random or Q remains uninformative, veto v2. Do not tune beta, eta, entropy, reward, detach, or arrival rate inside the same gate.

## 19. Formal-run boundary

Only a passing short gate authorizes three new formal runs:

- seeds `42, 86, 1042`;
- 1000 episodes;
- 200 slots;
- all other formal settings unchanged;
- persistent server sessions;
- checkpoints and diagnostics at the existing cadence.

If the short gate fails, the Goal terminates this branch with evidence and redirects investigation; it must not force formal training merely to satisfy a launch condition.

## 20. Safety

Never delete, modify, stage, or commit:

- local `docs/session_handoff_phase4.md`;
- local `runs/`;
- server historical `logs/`;
- server historical `runs/`.

All new server outputs use new timestamped roots below `/data2/zrj2025/HyperUAV/runs` and `/data2/zrj2025/HyperUAV/logs`.
