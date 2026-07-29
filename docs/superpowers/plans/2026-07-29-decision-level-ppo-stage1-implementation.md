# Stage 1 Decision-Level PPO Contextual-Bandit Gate Implementation Plan

Date: 2026-07-29  
Branch: `zrj_3multisample`  
Design authority:
`docs/superpowers/specs/2026-07-29-decision-level-ppo-microstep-mappo-design.md`  
Required starting commit: `2c358f7`

## 0. Scope, stop condition, and invariants

This plan implements only Stage 1:

```text
online sequential environment interaction
-> immutable executed-decision records
-> immediate EFT-regret advantage
-> old-action clipped PPO-style update
-> smoke
-> server pilot
-> matched three-seed 30-update gate
-> Stage 1 review
```

It does not implement Stage 2 transition types, boundary records,
decision-aware critic inputs, variable-discount GAE, or formal-reward
micro-step MAPPO.

The following remain unchanged:

- DAG arrival settings and active-DAG admission logic;
- formal environment reward and task lifecycle;
- one physical executor advance and one reward settlement per slot;
- frozen-ready ordering and sequential temporary reservation;
- candidate masks, dynamic UAV features, pair features, and EFT estimator;
- task-only `GraphSnapshot`;
- forced-hover movement;
- MLP task encoder and current offloading scorer architecture;
- optional counterfactual and lagged residual-Q paths remain disabled and
  are not baselines.

Stage 1 stops after the three-seed gate report. Stage 2 work is prohibited
unless the user reviews the result and explicitly authorizes it.

## 1. Establish the Stage 1 record and immutable buffer

Files:

- add `marl_models/mappo/clean_decision_ppo_bandit.py`
- modify `marl_models/mappo/clean_offloading_actor.py`

### 1.1 Add dedicated Stage 1 data types

In `clean_decision_ppo_bandit.py`, add:

- `DecisionBanditRecord`;
- `DecisionBanditRolloutBuffer`;
- `DecisionBanditUpdateConfig`;
- `DecisionBanditUpdateStats`.

`DecisionBanditRecord` contains:

- trajectory/environment/episode/slot/decision-order identifiers;
- task ID, DAG ID, task local index, and task-ID mappings;
- immutable MLP-consumed historical `task_features`;
- immutable historical `dynamic_uav_features` and `pair_features`;
- candidate UAV IDs, candidate mask, and candidate EFT vector;
- valid candidate count;
- actually executed action and UAV ID;
- frozen `old_masked_probabilities`;
- old action log-probability;
- best legal EFT, selected EFT, raw selected regret;
- frozen regret scale;
- frozen old-policy exact baseline and detached advantage.

Do not store incidence matrices or hyperedge type IDs in this Stage 1 record:
the selected encoder is MLP and those tensors are not consumed. Do not add
any of these fields to `CleanGraphSnapshot`.

The buffer owns immutable copies. It exposes counts for:

- `choice_decision_count`;
- `forced_decision_count`;
- `skipped_no_candidate`;
- physical slots collected;
- records per slot.

Only records with `valid_candidate_count >= 2` enter the actor-loss list.
Forced and skipped events remain diagnostics and reservation semantics, not
loss samples.

Zero-candidate tasks do not need a full actor record. A lightweight event or
collection-stat row must reliably store the environment/trajectory/episode,
physical slot, task and DAG IDs, decision order, `valid_candidate_count = 0`,
and the skip reason. It is emitted inside the sequential actor loop at the
point where the zero-candidate condition is observed.

### 1.2 Preserve the old behavior distribution at decision time

Extend `CleanOffloadingActionRecord` with an immutable CPU copy named
`old_masked_probabilities`. Capture it from the same `Categorical`
distribution that samples the environment action. Always save these
probabilities, `candidate_mask`, and `old_log_prob`; do not implement a
logits/probabilities union schema.

Do not extend `CleanOffloadingRolloutRecord` merely to carry this Stage 1
diagnostic data. The dedicated runner converts `latest_records` immediately
after `act()` and before the next slot can overwrite them. Existing clean
MAPPO rollout schema and updater behavior stay unchanged.

At collection time assert:

```text
selected_action == action used by reservation.reserve()
selected_uav_id == candidate_uav_ids[selected_action]
old_log_prob ~= log(old_probability[selected_action])
candidate rows == mask length == EFT length == UAV-ID length
selected EFT ~= EFT[selected_action]
```

The two floating-point consistency checks use explicit numerical tolerances,
not bitwise equality.

The actor must continue to:

1. process the frozen ready list in its existing order;
2. build features, mask, and EFT from the current reservation;
3. sample one action;
4. append the assignment;
5. update reservation immediately;
6. then move to the next task.

Do not move record construction outside this sequence.

### 1.3 Convert online action history into Stage 1 records

Add a pure conversion helper that combines:

- the slot's copied MLP graph input;
- the matching `CleanOffloadingActionRecord`;
- the fixed scale `61.75621424202263`;
- run/episode/slot identity.

For legal candidate `i`, compute once:

```text
raw_regret_i = EFT_i - min(EFT_legal)
reward_i = -raw_regret_i / scale
old_baseline = sum(old_probability_i * reward_i)
advantage = reward_executed - old_baseline
```

All reward, EFT, baseline, and advantage fields are detached scalar/array
copies. PPO epochs must never recompute them using updated parameters.

### Verification

Add direct unit-smoke assertions for:

- mutation of actor/environment tensors cannot change the buffer;
- feature/mask/EFT/action alignment;
- sequential reservation changes the second decision's context;
- old probability normalization over legal candidates;
- zero-candidate events create no action record;
- one-candidate events can execute and reserve but do not enter actor loss;
- no GraphSnapshot schema change.

Suggested commit boundary:

```text
Add decision PPO bandit rollout records
```

## 2. Implement the frozen old-action PPO-style objective

Files:

- `marl_models/mappo/clean_decision_ppo_bandit.py`
- add `scripts/smoke_decision_ppo_bandit_objective.py`

### 2.1 Rebuild only the current policy

For every historical record in an update:

```text
historical task_features
-> current MLP encoder
-> gather historical task_local_index
-> concatenate saved dynamic_uav_features and pair_features
-> current shared scorer
-> apply saved candidate mask
-> current masked distribution
```

The update path must not call the environment, executor, reservation builder,
candidate feature builder, or EFT estimator.

The executed historical action and old log-probability remain fixed through
all epochs. Do not call `Categorical.sample()` in the update path.

### 2.2 Implement the clipped objective

For each effective record:

```text
new_log_prob = current_dist.log_prob(executed_action)
ratio = exp(new_log_prob - old_log_prob)
surrogate_1 = ratio * frozen_advantage
surrogate_2 = clip(ratio, 1-epsilon, 1+epsilon) * frozen_advantage
loss_i = -min(surrogate_1, surrogate_2)
```

The full-rollout actor loss is the equal-weight mean over nontrivial
decisions. Do not normalize advantages. Set entropy coefficient to exactly
zero; entropy is diagnostics only.

### 2.3 Deterministic chunk accumulation

Allow a configurable decision chunk size for memory only. Every epoch:

1. zero gradients once;
2. visit each effective decision exactly once in deterministic record order;
3. scale each chunk sum by the full effective decision count;
4. accumulate gradients;
5. measure the fully accumulated pre-clip gradients once;
6. clip once;
7. measure post-clip gradients once;
8. optimizer step once.

No `drop_last`, sample duplication, per-chunk optimizer step, or per-chunk
mean is allowed.

Only the MLP encoder and scorer belong to the optimizer. Critic and movement
modules are absent from this updater rather than present with zero loss.

### 2.4 Empty actor batch

If an entire 128-slot rollout contains no nontrivial records:

- do not calculate an empty mean, standard deviation, KL, or entropy;
- set `empty_actor_batch=true`;
- return actor loss, entropy, KL, clip fraction, and ratio statistics as
  `null`;
- skip backward, clipping, and optimizer step;
- leave parameters and optimizer state unchanged;
- still emit physical-slot, forced, skipped, arrival, and environment
  diagnostics.

### 2.5 Diagnostics

Record per epoch and per outer update:

- effective decision count and unique physical-slot count;
- loss and frozen advantage mean/std/min/max;
- ratio mean/std/min/max, clip fraction, approximate KL;
- raw EFT regret mean/median/p95;
- nontrivial greedy agreement;
- margin >=5 s and >=20 s sample counts and accuracies;
- normalized entropy, maximum action probability, top1-top2 probability
  margin;
- invalid selected-action count;
- encoder/scorer pre- and post-clip gradient norms;
- optimizer step count;
- finite flags.

### Verification

`smoke_decision_ppo_bandit_objective.py` covers:

1. exact old-policy baseline by hand;
2. baseline and advantage are detached;
3. positive-advantage clipped PPO by hand;
4. negative-advantage clipped PPO by hand;
5. no resampling across three epochs;
6. executed action/log-probability consistency;
7. illegal actions have zero probability and cannot be selected;
8. equal per-decision weighting under uneven chunk sizes;
9. chunked gradient and full-batch gradient/parameter-step equivalence;
10. entropy coefficient contributes no gradient;
11. encoder/scorer receive finite gradients;
12. frozen records remain bitwise unchanged;
13. empty actor batch is a true no-op with null diagnostics;
14. zero/single-candidate events never enter the denominator.

Suggested commit boundary:

```text
Add executed-action decision PPO objective
```

## 3. Add a dedicated online Stage 1 runner

Files:

- add `scripts/train_decision_ppo_bandit_gate.py`
- add `scripts/run_decision_ppo_bandit_gate.py`
- add `scripts/smoke_decision_ppo_bandit_collection.py`
- add `scripts/smoke_decision_ppo_bandit_runner.py`
- minimally modify `scripts/train_clean_mainline.py` only if a reusable,
  behavior-neutral environment/module/checkpoint helper must be made public

Do not add a Stage 1 mode to `CleanPPOUpdater`. Keeping the diagnostic
updater separate prevents accidental critic, GAE, formal reward, or
resampled-EFT reuse.

### 3.1 Online collection loop

The dedicated trainer reuses the current clean orchestration in this order:

```text
prepare physical slot once
-> MLP forward once
-> force UAV hover
-> sequential stochastic offloading actor act()
-> copy DecisionBanditRecords
-> commit assignments and advance executor once
-> log formal environment result as diagnostics only
```

The formal slot reward may be logged but cannot enter the Stage 1 loss.
Do not run critic forward, movement actor forward, GAE, PPO ratio for slot
reward, or any auxiliary action sampler.

Each outer update collects exactly 128 physical slots with `num_envs=1`.
Episode termination/reset follows the current environment; the decision
buffer may contain records from more than one episode, and identifiers must
preserve that boundary.

### 3.2 Train/control groups

Implement:

- `S1-A`: stochastic masked actions from the frozen random initialization;
  collect the same interaction budget but perform no optimizer step;
- `S1-B`: identical initialization, stochastic masked actions, then the
  Stage 1 decision-PPO update.

For each seed, create one initialization payload and load it strictly into
both groups. Assert matching:

- encoder state hash;
- scorer state hash;
- parameter count;
- architecture/config hash.

Controls and treatments use fresh environment instances and the existing
fixed-scenario evaluator. Identical seeds do not make the trajectories strict
counterfactual pairs: `active_dag_cap` makes eligibility, later admission,
and RNG consumption policy-dependent after policies diverge.

### 3.3 Frozen experiment controls

The runner rejects incompatible arguments:

```text
task_encoder != mlp
num_envs != 1
freeze_movement != true
entropy_coef != 0
counterfactual or lagged-Q enabled
critic/GAE/formal-reward loss enabled
```

Freeze before the pilot:

- seeds: 42, 86, 1042;
- outer updates: 30;
- physical slots per update: 128;
- PPO epochs: exactly `3`;
- learning rate: `3e-4`;
- clip ratio: `0.2`;
- max gradient norm: `0.5`;
- task embedding dimension: `64`;
- hidden dimension: `128`;
- regret scale: `61.75621424202263`;
- entropy coefficient: `0`.

No result-dependent tuning is permitted after the pilot. The pilot may
validate only technical correctness and resource limits.

### 3.4 Checkpoints

Checkpoint:

- encoder and scorer states;
- optimizer state for S1-B;
- completed outer update count and physical-slot count;
- initialization identity;
- immutable experiment controls and scale;
- technical diagnostics.

Do not add a complete resume system in Stage 1. If existing code cannot
reliably restore the exact environment and RNG state, an interrupted cell is
rerun from that cell's initial checkpoint. Metadata must not claim exact
mid-episode continuation.

Save update checkpoints at:

```text
0, 1, 5, 10, 20, 30
```

### Verification

Collection and runner smokes prove:

- exactly 128 physical slots per normal update;
- one prepare/commit/executor/reward event per physical slot;
- frozen movement produces hover only and has no trainable parameters;
- formal reward cannot affect the objective;
- no critic/GAE/PPO-mainline updater call occurs;
- behavior action equals reservation and environment assignment;
- records span episode resets without cross-episode state leakage;
- S1-A parameters never change;
- S1-B performs one optimizer step per non-empty epoch;
- A/B initialization hashes match;
- empty actor rollout completes without NaN or parameter mutation;
- checkpoints reject schema/config/seed/shape mismatch.

Suggested commit boundary:

```text
Add online decision PPO bandit gate runner
```

## 4. Reuse the existing fixed-scenario closed-loop evaluator

Files:

- add `scripts/eval_decision_ppo_bandit_closed_loop.py` only if the existing
  evaluator cannot strictly load the Stage 1 encoder/scorer checkpoint;
- add `scripts/smoke_decision_ppo_bandit_closed_loop.py`.

Do not modify `environment/env.py`, `environment/dag_tasks.py`, UE movement
RNG, or any other environment random source in Stage 1.

For updates `0, 1, 5, 10, 20, 30`, evaluate both:

- stochastic masked sampling, primary;
- deterministic masked argmax, secondary.

Use fixed hover and the same existing evaluator seed/config. Report:

- episode reward total;
- generated/admitted/blocked/completed DAG counts and blocked reasons;
- completion rate, flowtime, throughput, queue length, unfinished work;
- executed raw EFT regret and agreement metrics;
- entropy/concentration metrics;
- per-seed S1-B minus S1-A same-seed deltas.

Use 20-30 episodes for the short gate, frozen before formal launch. Report
stochastic and deterministic results separately. Explicitly state that
`active_dag_cap` creates policy-dependent trajectory and RNG-consumption
differences, so these are not strict exogenous counterfactual pairs.

Suggested commit boundary:

```text
Add fixed-scenario evaluation for decision PPO gate
```

## 5. Run the complete local and server smoke suite

### 5.1 Local syntax and non-Torch checks

Run:

```bash
python -m py_compile \
  marl_models/mappo/clean_decision_ppo_bandit.py \
  scripts/train_decision_ppo_bandit_gate.py \
  scripts/run_decision_ppo_bandit_gate.py \
  scripts/eval_decision_ppo_bandit_closed_loop.py \
  scripts/smoke_decision_ppo_bandit_objective.py \
  scripts/smoke_decision_ppo_bandit_collection.py \
  scripts/smoke_decision_ppo_bandit_runner.py \
  scripts/smoke_decision_ppo_bandit_closed_loop.py
```

Run all new smokes. If local Torch is unavailable, Torch branches may report
a real skip; schema, immutability, sequential reservation, skip-event stats,
and empty-batch checks must still run with the available bundled environment.

### 5.2 Existing regression smokes

Run at least:

```bash
python scripts/smoke_clean_graph.py
python scripts/smoke_clean_graph_hyperedge_types.py
python scripts/smoke_clean_slot_orchestration.py
python scripts/smoke_clean_offloading_actor.py
python scripts/smoke_clean_training_loop.py
python scripts/smoke_contextual_bandit_gate.py
python scripts/smoke_fixed_movement_mappo_eft_gate.py
python scripts/smoke_clean_arrival_funnel.py
```

Use actual existing filenames discovered at implementation time; if a named
smoke has been renamed, record the current replacement rather than creating
a duplicate compatibility wrapper.

### 5.3 Server Torch verification

Before every server command, state the exact action. Work only under:

```text
/data2/zrj2025
```

Preserve server `logs/` and `runs/`. Sync only through normal push and
`git pull --ff-only`; never use force push, reset, or clean.

Run all new Torch smokes and the relevant regression suite in the server
repository. Do not start the pilot if any required smoke fails.

## 6. Run a technical pilot

Run one seed (`42`), both S1-A and S1-B, for two outer updates with the formal
128-slot horizon and the frozen three PPO epochs.

The pilot is accepted only if:

- both cells exit without traceback/OOM;
- actions, masks, EFT vectors, and reservations align;
- no update-time action sampling occurs;
- no illegal action is executed;
- ratios, losses, KL, gradients, and parameters are finite;
- S1-A has zero optimizer steps and unchanged hashes;
- S1-B has exactly three optimizer steps per non-empty update;
- empty actor batches, if encountered, use the specified no-op semantics;
- environment executor/reward counts equal physical-slot counts;
- critic, GAE, movement loss, and formal reward are absent from the update;
- GPU memory, CPU RSS, and wall time are acceptable.

Do not select hyperparameters from regret, accuracy, entropy, reward, or
completion results. If the pilot fails, stop and report the original error;
do not silently change the algorithm or restart.

## 7. Run the formal three-seed Stage 1 gate

After explicit pilot acceptance:

```text
groups: S1-A, S1-B
seeds: 42, 86, 1042
updates: 30
physical slots/update: 128
movement: fixed hover
encoder: MLP
entropy coefficient: 0
```

Launch in the server background with a control log and PID. Record:

- exact command;
- branch and commit;
- GPU;
- PID;
- output and checkpoint directories;
- fixed-scenario evaluator seed and configuration;
- expected completion time.

After successful launch, stop active monitoring and use one low-frequency
completion check after the estimated finish time. Do not launch duplicates.

For each group/seed verify:

- update counts and physical interaction budget;
- initialization and configuration hashes;
- decision/forced/skipped/empty-batch counts;
- optimizer-step counts;
- no traceback, NaN/Inf, OOM, or illegal action;
- all required checkpoints;
- complete stochastic and deterministic fixed-scenario closed-loop outputs.

## 8. Summarize and apply the Stage 1 gate

Report per group and seed, then three-seed mean/std:

- sampled raw EFT regret mean/median/p95;
- nontrivial greedy agreement;
- margin >=5 s and >=20 s accuracy with sample counts;
- entropy, maximum probability, probability margin;
- ratio/KL/clip and gradient diagnostics;
- stochastic fixed-scenario reward, completion, flowtime, throughput, queue,
  and
  unfinished work;
- deterministic equivalents as secondary metrics;
- S1-B minus S1-A same-seed deltas;
- arrival funnel counts and policy-dependent blocked/admitted differences.

The report must not call these strict paired counterfactuals. It must explain
that `active_dag_cap` can change later eligibility and RNG consumption.

Classification:

- **strong pass:** all three seeds improve sampled behavior; regret falls
  roughly 50%; large-margin sampled accuracy approaches 90%; entropy
  meaningfully falls; stochastic closed-loop beats control; stochastic and
  argmax performance narrow; no seed fails;
- **partial pass:** argmax improves but sampling remains diffuse, results are
  seed-fragile, or closed-loop improvement is not consistent;
- **fail:** executed-action PPO path does not improve despite technical
  correctness, or any technical invariant fails.

Only a strong pass permits a separate user authorization for Stage 2.
Partial pass or failure ends this implementation stage and triggers review,
not automatic reward, arrival, critic, Q, or HGNN changes.

## 9. Stage 2 interface reservations only

The following future impacts are recorded but must not be implemented in
Stage 1:

- `CHOICE_DECISION`, `FORCED_DECISION`, `SKIP_DECISION`, and
  `SLOT_BOUNDARY` transition schema;
- persistent prepared next-slot state across PPO optimization;
- full per-UAV reservation critic matrix;
- record-type/task-present critic slots;
- variable `gamma` and `gae_lambda`;
- GAE computed once and frozen;
- terminal/truncation only at boundaries;
- matched S2-A/S2-B critic architecture and initialization checks;
- actor-empty but critic-nonempty shared-optimizer semantics.

Do not add placeholder fields to current rollout records merely for these
future items. Stage 2 begins with a fresh schema review after Stage 1
authorization.

## 10. Git and artifact discipline

At each commit boundary:

1. inspect `git status --short`;
2. stage only the explicitly reviewed Stage 1 files;
3. run `git diff --cached --stat` and
   `git diff --cached --name-only`;
4. never stage datasets, `logs/`, `runs/`, checkpoints, handoff documents,
   or protected untracked files;
5. do not use `git add .`;
6. do not push until the user authorizes the push or a later execution
   instruction explicitly includes it.

Protected local and server artifacts remain untouched throughout.
