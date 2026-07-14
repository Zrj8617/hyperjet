# Offloading Counterfactual Critic Design

Date: 2026-07-14
Branch: `zrj_3_static_ue`
Baseline commit: `390c049f53bc66298120317555e4fd85430f4993`

## 1. Purpose

The offloading actor receives one normalized slot-level GAE advantage for every task assignment made in that slot. The loss then averages those assignments within the slot. This is valid as a joint-policy gradient estimator, but it gives every selected task–UAV action the same scalar teaching signal. It cannot distinguish a good assignment from a bad assignment in the same slot.

The objective of this change is to add a learned, action-conditioned counterfactual term to the offloading PPO advantage while retaining the existing on-policy PPO sample semantics. The change must stabilize offloading learning across seeds without changing reward, entropy, mobility, the shared HGNN default, or the existing centralized slot critic.

This design does not use the greedy EFT policy as a training teacher.

## 2. Evidence and gate conclusion

The 60-cell policy gate is stored at:

`/data2/zrj2025/HyperUAV/runs/offloading_policy_gate_20260714_215332`

It compared three ep1000 checkpoints, five common environment seeds, and four offloading policies. Greedy EFT always reduced its own local estimated regret, but its system-level effect depended on the checkpoint:

- model 42: greedy was approximately neutral or worse than actor;
- model 86: greedy reduced local regret but was systemically mixed or worse;
- model 1042: greedy strongly improved completion, throughput, flowtime, transfer time, and queue waiting.

The 30-cell frozen-movement isolation is stored at:

`/data2/zrj2025/HyperUAV/runs/offloading_movement_isolation_20260714_222309`

All 30 cells completed, all had `movement_frozen=true`, and all had zero KaHyPar degraded slots. Mean deterministic arrival completion was:

| Checkpoint | Actor | Greedy EFT |
|---|---:|---:|
| model 42 | 0.7992 | 0.7931 |
| model 86 | 0.8276 | 0.7931 |
| model 1042 | 0.4797 | 0.7931 |

Freezing movement removed trajectory divergence but did not make EFT consistently superior. Therefore the original EFT teacher is vetoed. The evidence is branch 2 of the policy gate: lowering local EFT regret does not consistently improve the global DAG objective.

The isolation also establishes that offloading policy differences matter independently of movement. Model 42 and model 86 contain useful system-level offloading behavior that the EFT heuristic does not reproduce, while model 1042 learned a poor offloading policy. The repair should learn long-term action values rather than distill EFT.

## 3. Rejected alternatives

### 3.1 Greedy EFT ranking auxiliary

Rejected because the gate shows that local EFT ranking is not a reliable system-level teacher. Forcing entropy down or increasing an EFT auxiliary coefficient could turn small, seed-specific logit differences into confident wrong rankings.

### 3.2 Lightweight DAG-only lookahead teacher

Rejected as the first repair after implementation review. The executor computes sink return at a later slot using the later UAV and UE positions. Learned UAV movement, moving UEs, future DAG arrivals, and future queue occupation are not known to a planner that only expands the current DAG at the current positions. A faithful candidate counterfactual would require cloning the full environment and rolling out future movement, arrivals, and offloading for every candidate. That is too expensive and still policy-dependent for use as a training teacher.

### 3.3 Delayed task outcome applied to an old PPO action

Rejected because task and DAG outcomes can cross a rollout boundary. Reusing an action and old log probability after one or more optimizer updates would create stale PPO ratios. Redesigning the rollout lifecycle to wait for every task outcome is outside the minimal repair and would conflict with the current fixed 128-slot rollout protocol.

### 3.4 Critic-to-HGNN detach, reward changes, or entropy forcing

Rejected for this experiment. Existing `w_c=16` shared-HGNN experiments showed shared outperforming detach for both paired seeds. Reward direction has already been corrected. Global clipping alone is not evidence that the offloading head is starved under Adam. These variables remain frozen.

## 4. Proposed architecture

### 4.1 Optional action-value module

Add an optional `CleanOffloadingActionValueCritic`. It is instantiated only when both of the following training configuration values are positive:

- `offloading_counterfactual_coef` (`beta`);
- `offloading_action_value_loss_coef` (`eta`).

The module is a small MLP with the same hidden width convention as the current offloading scorer. It produces one scalar for each legal task–UAV candidate. Its final linear layer is initialized to zero, so every counterfactual advantage starts at zero.

The input for candidate `u` at decision `i` is:

1. the current candidate feature row already used by the actor:
   - task HGNN embedding;
   - dynamic UAV features;
   - task–UAV pair features;
2. the current centralized critic input:
   - pooled active-task HGNN embedding;
   - existing non-graph global state, including UAV, active-task, and queue summaries.

The complete action-value input is detached before entering the action-value critic. Consequently:

- the action-value loss updates only the action-value critic;
- it cannot update the HGNN, offloading actor, movement actor, or existing slot critic;
- the existing shared HGNN actor/value behavior is unchanged.

When enabled, the action-value critic uses the existing optimizer learning rate and is added to the existing single optimizer and global `clip_grad_norm_` parameter set. It does not receive a separate learning rate, optimizer, or clipping rule. Disabled mode leaves the optimizer parameter list exactly unchanged.

The action-value module is not used for environment action selection and is not required for deterministic inference. The learned offloading actor remains the deployed policy.

### 4.2 Target

The existing trainer computes raw slot returns and slot GAE advantages, then normalizes the slot advantages across the closed rollout. Let the normalized slot advantage be `A_slot[t]`.

For every selected offloading action in slot `t`, train the selected candidate value toward the detached target:

`Q_phi(s_t, i, a_i) -> A_slot[t]`

All actions in one slot still share the observed Monte Carlo/GAE target, but the action-value critic conditions on the task, candidate UAV, temporary reservation state, decision order state encoded in the candidate features, and global context. Across on-policy exploration it estimates the conditional long-term value of each candidate.

Only the sampled action is supervised. Unselected candidates are not assigned synthetic EFT labels. The present near-uniform policy provides broad candidate coverage during early training.

The action-value regression loss is averaged first across actions in a slot and then across effective offloading slots, preserving the current invariant that adding more ready tasks does not automatically increase a slot's optimizer weight.

### 4.3 Counterfactual advantage

For each decision, recompute the current masked actor distribution and candidate values. Define:

`A_cf = Q_phi(s, i, a_selected) - sum_a pi_theta(a | s, i) Q_phi(s, i, a)`

Both the policy probabilities and action values are detached for this calculation. `A_cf` must never create a direct gradient path into either network.

Collect all valid `A_cf` values in the rollout loss pass and normalize them to zero mean and unit population standard deviation. If there are fewer than two valid samples or the standard deviation is below `1e-8`, use zero for the normalized counterfactual term. A single legal candidate always has zero counterfactual advantage.

The offloading PPO advantage becomes:

`A_off[t, i] = A_slot[t] + beta * normalize(A_cf[t, i])`

The movement actor continues to use `A_slot[t]` exactly as before. PPO clipping, old log probabilities, action masks, slot/action aggregation, and entropy remain unchanged.

### 4.4 Total loss

The enabled total loss is:

`L_total = L_move_PPO + L_off_PPO(A_off) + value_coef * L_value + eta * L_Q - movement_entropy_coef * H_move - offloading_entropy_coef * H_off`

`L_Q` is half mean squared error between selected candidate values and normalized slot advantages, aggregated per slot as described above.

The initial validation arms are fixed to:

| Arm | beta | eta |
|---|---:|---:|
| regression baseline | 0.0 | 0.0 |
| counterfactual-small | 0.25 | 0.5 |
| counterfactual-medium | 0.50 | 0.5 |

No other hyperparameter differs between arms. If both enabled arms pass, select the smaller `beta` to keep the original PPO return dominant.

## 5. On-policy and causality invariants

The implementation must preserve all of the following:

1. Actions and old log probabilities come only from the currently closed rollout.
2. Every rollout is consumed before collecting samples under the updated policy.
3. No task action is retained for a future PPO update.
4. No stale PPO ratio is evaluated across update boundaries.
5. The action-value target uses only the current rollout's existing GAE calculation and bootstrap semantics.
6. The action-value loss has no gradient path into HGNN or either actor.
7. The counterfactual term is detached before it is supplied to PPO.
8. Temporary reservations remain part of the sequential candidate features; no candidate is evaluated against an earlier, inconsistent reservation state.

This module is an action-dependent control variate for current on-policy PPO, not a delayed replay buffer and not a heuristic teacher.

## 6. Configuration and compatibility

Add CLI and persisted configuration fields:

- `--offloading-counterfactual-coef`, default `0.0`;
- `--offloading-action-value-loss-coef`, default `0.0`.

Both values must be finite and non-negative. Exactly one positive and one zero is an invalid configuration because it either trains an unused critic or applies an untrained critic. The enabled configuration requires both values to be positive.

Persist both fields in:

- resolved run config;
- checkpoint config;
- `run_summary.json`;
- training JSONL update records;
- resume compatibility checks;
- evaluation provenance.

Compatibility rules:

- `beta=0, eta=0` does not instantiate the new module, does not add optimizer parameters, and must reproduce the current checkpoint schema and update behavior apart from new zero-valued provenance fields.
- A legacy checkpoint without an action-value state is valid only when both resolved coefficients are zero.
- An enabled checkpoint must contain the action-value module state. Missing state is an error.
- Resume rejects either coefficient mismatch.
- Deterministic evaluation loads and validates enabled checkpoint provenance but selects actions only with the offloading actor.
- Existing old checkpoints remain evaluable without conversion.

The checkpoint payload key, when enabled, is `offloading_action_value_critic`.

## 7. Diagnostics

Add behavior-neutral update diagnostics:

- `offloading_action_value_loss`;
- `offloading_action_value_target_mean/std`;
- `offloading_action_value_selected_mean/std`;
- `offloading_action_value_explained_variance`;
- `offloading_legal_q_spread_mean`;
- `offloading_counterfactual_advantage_mean/std` before normalization;
- `offloading_counterfactual_advantage_normalized_std`;
- `offloading_counterfactual_effective_action_count`;
- action-value critic pre/post-clip gradient norm;
- configured `beta` and `eta`;
- an explicit diagnostic proving Q-loss-to-HGNN gradient norm is zero when enabled.

Retain the existing diagnostics for PPO losses, actor/HGNN/critic gradients, clipping scale, entropy, margin, agreement, estimated regret, completion, throughput, backlog, movement entropy, hover ratio, displacement, and explained variance.

Action-value explained variance is diagnostic rather than a standalone pass criterion. System performance remains authoritative.

## 8. Error handling

- No legal candidates: preserve the existing safe skip and create no action-value sample.
- One legal candidate: train its selected value if enabled, but use zero counterfactual advantage.
- Masked candidates: exclude from both the policy expectation and Q-spread diagnostics.
- Invalid task index or inconsistent candidate shape: fail the update with a descriptive error rather than silently train a misaligned sample.
- Any non-finite target, Q value, counterfactual advantage, loss, or gradient: fail the current validation run and report the first update and field.
- Disabled mode: execute no action-value forward pass.

## 9. Local verification

Add focused smoke coverage for:

1. zero-initialized Q output and zero initial counterfactual advantage;
2. masked policy-weighted baseline;
3. single-candidate and no-candidate behavior;
4. per-action counterfactual terms differ within one slot when Q values differ;
5. Q target is the normalized slot advantage;
6. per-slot loss aggregation is invariant to duplicating action count;
7. action-value loss updates Q parameters but not HGNN or either actor;
8. PPO actor loss receives detached counterfactual advantages;
9. `beta=0, eta=0` numerical regression against the current update path;
10. enabled checkpoint save/load;
11. legacy checkpoint compatibility in disabled mode;
12. missing enabled Q state and coefficient mismatch rejection;
13. deterministic eval provenance;
14. finite loss and gradients;
15. training and evaluation smoke with KaHyPar active where available.

Run the existing clean PPO, trainer, checkpoint, resume, eval, detach, diagnostics-neutrality, plotting, server-Torch, and end-to-end smoke suites in addition to the new focused tests.

## 10. Server short validation

After local tests, focused commit, push, and authorized server synchronization, run nine short training cells:

- three arms from Section 4.4;
- seeds 42, 86, and 1042;
- 100 episodes per cell;
- 200 slots per episode;
- rollout 128;
- PPO epochs 3;
- all remaining hyperparameters identical to the final protocol;
- moving UE and learned UAV movement;
- `w_c=16`;
- `detach_critic_hgnn=false`;
- KaHyPar full hyperedges enabled.

Use new run and log roots. Do not overwrite any earlier gate, diagnostic, or training output. Use a persistent tmux session and record commands, PIDs, GPU choice, logs, output directories, start time, and git commit. Do not silently restart failed cells or change parameters.

After all nine cells complete, evaluate each ep100 checkpoint deterministically on environment seeds 4242–4246 with 200 arrival slots and drain. This produces 45 actor-policy evaluation cells. Compare enabled arms with the same-seed disabled baseline; EFT remains a diagnostic reference only and is not a training target.

## 11. Short-validation gate

An enabled arm passes only if all integrity conditions and all primary performance conditions pass.

Integrity conditions:

1. all three training cells and all 15 deterministic evaluation cells complete;
2. no NaN/Inf, traceback, unexplained KaHyPar degradation, invalid action, or residual process;
3. Q gradients are finite and nonzero after initialization;
4. Q-loss-to-HGNN gradient is zero;
5. checkpoint resume and deterministic eval load correctly;
6. offloading normalized entropy does not fall below `0.50` in the final 20 training episodes;
7. no seed develops a new movement collapse relative to its paired baseline, defined as a hover-ratio increase greater than `0.20` together with a displacement decrease greater than `20%`.

Primary deterministic performance conditions, paired against the same-seed disabled baseline:

1. pooled arrival completion improves by at least `0.02`;
2. pooled mean drain flowtime improves by at least `5%`;
3. at least two of three checkpoint seeds improve mean arrival completion by at least `0.01`;
4. no checkpoint seed loses more than `0.02` mean arrival completion;
5. no checkpoint seed worsens mean drain flowtime by more than `10%`;
6. at least 10 of 15 paired scenes improve arrival completion or flowtime by more than `1e-9` without worsening the other metric by more than `1e-9`.

If neither enabled arm passes, do not start formal training. The result vetoes this counterfactual formulation and the next investigation must address the action-value target or critic capacity. Do not compensate by changing reward, entropy, `w_c`, detach, or unrelated hyperparameters.

If both enabled arms pass, choose `beta=0.25, eta=0.5`. If only one passes, choose that arm. The chosen arm must then pass checkpoint resume/eval smoke once more from a newly produced short-run checkpoint.

## 12. Formal training boundary

Only a configuration that passes Section 11 may enter the final three formal runs. The formal runs use one identical selected configuration with seeds 42, 86, and 1042 and the protocol required by the active Goal:

- 1000 episodes;
- 200 slots per episode;
- rollout 128;
- PPO epochs 3;
- learning rate `3e-4`;
- gamma `0.99`;
- GAE lambda `0.95`;
- PPO clip `0.2`;
- movement and offloading entropy coefficients `0.01`;
- value coefficient `0.5`;
- max gradient norm `0.5`;
- completed-DAG weight `16`;
- learned UAV movement;
- moving UEs;
- shared HGNN and `detach_critic_hgnn=false`;
- KaHyPar full hyperedges enabled;
- checkpoint every 20 episodes plus `latest.pt`;
- persisted config, metrics, summary, and complete provenance.

The three runs may begin only after presenting the exact commands, GPU allocation, persistent execution method, log/run roots, and resource check to the user. Starting is not completion. The Goal completes only after all three runs finish and satisfy the full artifact and process audit in the active Goal.

## 13. Research interpretation

The new module is a learned action-dependent critic/control variate. It is not an EFT teacher and does not select evaluation actions. PPO remains the policy objective, and the deployed offloading actor may outperform the action-value critic's instantaneous ranking.

Reports must distinguish:

- the unchanged environment reward and slot GAE;
- the existing centralized slot critic;
- the new action-conditioned critic;
- the counterfactual control-variate term;
- the learned offloading actor used at inference.

Training-sampled completion is not final deterministic performance. Formal model comparison still requires a later independent deterministic evaluation, which is outside the final three-run completion boundary of the active Goal.

## 14. Out of scope

- changing DAG arrival rate or DAG size;
- changing reward or `w_c`;
- setting entropy to zero or otherwise forcing confidence;
- enabling critic-to-HGNN detach;
- changing value coefficient or clipping policy;
- distilling EFT or another heuristic;
- retaining completed task actions across rollout updates;
- changing learned movement or UE mobility;
- claiming final model superiority from training logs alone.
