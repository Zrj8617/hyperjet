# Decision-Level PPO and Decision Micro-Step MAPPO Design

Date: 2026-07-29  
Branch baseline: `zrj_3multisample`  
Expected baseline commit: `e53947a`

## 1. Purpose

The current clean MAPPO implementation may process several sequential
offloading decisions inside one physical slot, while every offloading action
uses the same slot-level advantage. This design separates the diagnosis from
the formal algorithm change:

1. Stage 1 is a PPO-style contextual-bandit gate. It verifies the executed
   action, old log-probability, clipped ratio, mask, and optimizer path using
   one immediate EFT-regret advantage per offloading decision.
2. Stage 2 expands the already-sequential assignments into explicit
   decision micro-steps, then reconnects the original environment reward,
   a decision-aware critic, and variable-discount GAE.
3. Decision-level Q, outcome ledgers, counterfactual objectives, and lagged
   residual-Q remain out of scope. They may be designed separately only if
   both stages are technically correct and Stage 2 still fails.

Stage 2 removes the implementation error of copying one scalar advantage to
all decisions in a slot. It does not claim to calculate each assignment's
exact causal contribution. Decisions in one slot still share a downstream
return sample; their advantages differ mainly through their state-dependent
value baselines.

## 2. Fixed Boundaries

Both stages preserve:

- the current DAG generation and arrival-rate configuration;
- the current physical execution, reward formula, and task lifecycle;
- frozen-ready ordering;
- sequential temporary-reservation updates;
- candidate masks, dynamic UAV features, and pair features;
- the greedy-EFT estimator;
- the task-only `GraphSnapshot` boundary;
- MLP task encoding for the first implementation;
- forced UAV hover and a non-trainable movement actor;
- one synchronous environment for the first gate.

Both stages prohibit:

- changing arrival rate to make training easier;
- changing the formal reward;
- advancing arrival, execution, reward settlement, or physical time once per
  micro-step;
- adding HGNN changes;
- adding event-driven environment execution;
- introducing a second optimizer that owns the shared task encoder;
- treating EFT as the final formal reward;
- treating optional counterfactual or lagged-Q paths as baselines.

One physical slot still performs exactly one arrival/orchestration cycle, one
executor advance, one formal reward settlement, and one physical-time
increment.

## 3. Stage 1: PPO-Style Contextual-Bandit Gate

### 3.1 Scope

Stage 1 is not MAPPO and is not actor-critic. It uses no critic, GAE, formal
environment reward, cross-slot return, or movement optimization.

Its only question is:

> When every executed offloading decision receives an immediate independent
> advantage, can the standard old-action, old-log-probability, clipped-ratio,
> and optimizer path concentrate the sampled policy?

### 3.2 Historical Decision Record

Every offloading decision with at least two legal candidates stores an
immutable `DecisionBanditRecord`:

- environment, trajectory, episode, physical slot, and decision-order IDs;
- task and DAG IDs;
- the MLP-consumed historical graph tensors and task-ID mappings;
- task local index;
- historical dynamic UAV features;
- historical pair features;
- candidate mask and candidate UAV IDs;
- decision-time candidate EFT vector;
- valid-candidate count;
- the action actually executed in the environment;
- executed UAV ID;
- old masked logits or frozen old masked probabilities;
- old action log-probability;
- best legal EFT, selected EFT, and raw EFT regret;
- fixed regret scale;
- old-policy exact baseline;
- detached decision advantage.

Because Stage 1 is MLP-only, it does not duplicate unused incidence matrices
or hyperedge type IDs in this record. It stores only raw tensors actually
consumed by the MLP actor. This does not change `GraphSnapshot`; it avoids
unnecessary memory and copy risk in the Stage 1 buffer.

The candidate features, mask, EFT vector, executed action, and old
log-probability must all come from the same decision-k temporary-reservation
state. The executed action must be the action that updates reservation and is
later executed by the environment.

PPO epochs must not resample a different auxiliary action and must not
recompute old probabilities with an updated model.

### 3.3 Reward, Baseline, and Advantage

For each legal candidate:

\[
r_i = -\frac{\mathrm{EFT}_i-\mathrm{EFT}_{\min}}{\mathrm{scale}}.
\]

The scale is the train-only contextual-bandit scale already frozen at:

\[
\mathrm{scale}=61.75621424202263.
\]

The exact behavior-policy baseline and frozen advantage are:

\[
b_{\mathrm{old}}(s)=\sum_i \pi_{\mathrm{old}}(i\mid s)r_i,
\]

\[
A_{\mathrm{decision}}
=r_{\mathrm{executed}}-\operatorname{stopgrad}(b_{\mathrm{old}}).
\]

EFT values, reward vectors, baselines, and advantages do not receive
gradients. Stage 1 does not normalize advantages; the fixed EFT scale retains
its original meaning across all seeds and updates.

### 3.4 PPO-Style Objective

For the action that was actually executed:

\[
\rho=
\exp\left(
\log\pi_\theta(a_{\mathrm{old}}\mid s)
-\log\pi_{\mathrm{old}}(a_{\mathrm{old}}\mid s)
\right).
\]

\[
L_{\mathrm{decision\text{-}PPO}}
=-\operatorname{mean}
\left[
\min\left(
\rho A_{\mathrm{decision}},
\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)
A_{\mathrm{decision}}
\right)
\right].
\]

Every valid nontrivial decision has equal weight. Candidate count does not
change sample weight. Each decision occurs exactly once per epoch, and no
sample is silently dropped when batch sizes do not divide sample count.

Entropy is recorded but its coefficient is fixed to zero in the Stage 1 main
gate.

### 3.5 Zero and Single Legal Candidates

- `valid_candidate_count = 0`: no action is executed and reservation is not
  changed. The event is counted as `skipped_no_candidate` and does not enter
  Stage 1 actor loss.
- `valid_candidate_count = 1`: the only legal assignment may be executed and
  may update reservation. It is counted as a forced decision but does not
  enter Stage 1 actor loss because its exact advantage is zero.

These cases must not inflate accuracy or the actor-loss denominator.

### 3.6 Stage 1 Optimization

Only the shared MLP task encoder and offloading scorer are trainable. The
critic and movement actor are frozen and need not execute forward passes.

There is one optimizer. For each epoch it performs one logically combined
gradient accumulation over all deterministic chunks, followed by one clip
and one optimizer step. Chunking may reduce memory use but must be normalized
by the full effective decision count so that it is numerically equivalent to
the full-rollout mean.

The gate records:

- pre- and post-clip encoder and scorer gradient norms;
- PPO ratio mean, standard deviation, minimum, and maximum;
- clip fraction and approximate KL;
- loss and advantage statistics;
- invalid sampled-action count;
- normalized entropy, maximum action probability, and top-1/top-2
  probability margin.

### 3.7 Stage 1 Sampling and A/B Test

Each outer update collects a fixed 128 physical slots, not a fixed number of
decisions. The first gate uses:

- `num_envs = 1`;
- synchronous sampling;
- MLP encoder;
- forced hover;
- PPO epochs fixed before the experiment;
- seeds 42, 86, and 1042;
- 30 outer updates.

The matched comparison is:

- **S1-A, frozen random control:** paired random initialization, stochastic
  masked action sampling, no parameter update.
- **S1-B, decision-PPO EFT treatment:** identical initialization and physical
  interaction budget, executed-action decision advantages, zero entropy
  coefficient, and clipped PPO updates.

The earlier resampled EFT-auxiliary experiment is historical evidence, not a
new matched baseline.

Primary metrics use the stochastic sampled policy:

- raw EFT regret mean, median, and p95;
- nontrivial greedy agreement;
- margin-at-least-5-second and margin-at-least-20-second accuracy;
- normalized entropy, maximum action probability, and probability margin;
- stochastic paired closed-loop reward, completion, flowtime, and throughput.

Deterministic argmax metrics are secondary.

Stage 1 strongly passes only if all three seeds improve sampled behavior,
mean sampled regret falls by roughly 50%, large-margin sampled accuracy
approaches 90%, entropy meaningfully falls, stochastic closed-loop beats the
frozen control, deterministic/stochastic performance narrows, and no seed
fails technically. Argmax-only improvement is a partial pass and blocks
Stage 2.

Thirty updates form a diagnostic gate, not a final convergence claim.

## 4. Stage 2: Decision Micro-Step MAPPO

### 4.1 Scope

Stage 2 reconnects:

- the original formal slot reward;
- a decision-aware centralized value critic;
- variable-discount GAE;
- clipped offloading PPO.

It uses no EFT reward or auxiliary loss. A Stage 1 actor checkpoint may be
used only as a warm-start comparison; all formal Stage 2 EFT coefficients
remain exactly zero.

### 4.2 Unified Ordered Transition Stream

Each frozen-ready task produces one of three pre-boundary records:

1. `CHOICE_DECISION`
   - at least two legal candidates;
   - `actor_mask = 1`, `critic_mask = 1`;
   - sampled action updates reservation.
2. `FORCED_DECISION`
   - exactly one legal candidate;
   - `actor_mask = 0`, `critic_mask = 1`;
   - the unique action executes and updates reservation.
3. `SKIP_DECISION`
   - no legal candidate;
   - `actor_mask = 0`, `critic_mask = 1`;
   - no action and no reservation change.

Recording forced and skipped deterministic steps keeps state progression,
decision order, and remaining-ready count explicit. They never enter actor
loss, entropy, action accuracy, or actor-advantage normalization.

Every physical slot then has exactly one `SLOT_BOUNDARY`, including slots
with no frozen-ready tasks or no choice decisions.

A slot with `m` frozen-ready tasks has:

```text
task-step(t,1) -> ... -> task-step(t,m) -> B(t) -> next-slot-first-state
```

### 4.3 Boundary Semantics

`B(t)` is constructed strictly after the last temporary-reservation update
and strictly before executor advancement or reward settlement:

```text
last reservation update
-> build and copy B(t)
-> evaluate old V(B(t))
-> advance the physical executor once
-> settle the formal slot reward once
-> prepare the next physical slot using the current clean orchestration
-> connect to its first task-step, or directly to B(t+1) if none exists
```

The next state is the first pre-action micro-state after all current
pre-decision operations of the next physical slot have completed, including
the existing UE movement, DAG-arrival, ready refresh, and frozen-ready
construction order.

The boundary must not connect to:

- an executor intermediate state;
- a state constructed after observing the current reward;
- a reset state from a new episode.

### 4.4 Rewards, Discount, and GAE Parameters

For `CHOICE_DECISION`, `FORCED_DECISION`, and `SKIP_DECISION`:

```text
reward = 0
gamma = gamma_inner = 1
gae_lambda = lambda_inner = 1
terminal = false
```

For `SLOT_BOUNDARY`:

```text
reward = original formal slot reward
gamma = gamma_slot, initially 0.99
gae_lambda = lambda_slot, initially 0.95
```

No physical time passes between task-step records. Therefore the number or
order of ready tasks cannot introduce additional discount or lambda decay.

### 4.5 Terminal and Truncation

True termination and time-limit truncation can occur only on a boundary
transition after the physical slot advances.

For true termination:

\[
d_n=\gamma_n(1-\mathrm{true\_terminal}_n)=0.
\]

For time-limit truncation, the current clean-mainline final-observation
bootstrap semantics are preserved. Truncation must not be treated as a true
terminal and must not connect to the reset episode.

The final inner decision always connects to its pre-executor boundary; it
never directly receives a zero next value because the episode is about to
end.

### 4.6 Decision and Boundary Critic Inputs

Actor and critic continue to share the MLP task encoder. This is the frozen
first design choice to isolate decision granularity from parameter-sharing
changes.

The decision critic receives, outside `GraphSnapshot`:

- a fixed UAV-ID-ordered matrix for all five UAVs;
- per UAV: temporary available time, reserved workload, assigned count,
  queue/load features, and the dynamic resource features already consumed by
  the actor;
- current task representation and a task-present flag;
- decision order and remaining-ready count;
- simple fixed pooling of the unprocessed frozen-ready tasks;
- record-type and boundary flags.

The critic must retain per-UAV identity. Mean, maximum, or total summaries
alone are prohibited because different reservations can collide under those
summaries.

For a boundary state, the task-local input is zeroed with an explicit
task-present mask, while the complete post-reservation per-UAV matrix and
boundary flag remain.

The task-only `GraphSnapshot` schema does not change.

### 4.7 Variable-Discount GAE

Each collected transition stores immutable:

- old action log-probability when actor-masked;
- old value;
- reward;
- gamma;
- GAE lambda;
- true-terminal and truncation masks;
- next-state linkage.

After the rollout ends, GAE and return targets are calculated once:

\[
\delta_n
=r_n+d_nV_{\mathrm{old}}(s_{n+1})-V_{\mathrm{old}}(s_n),
\]

\[
A_n
=\delta_n+d_n\lambda_n A_{n+1},
\]

where:

\[
d_n=\gamma_n(1-\mathrm{true\_terminal}_n).
\]

Old values, advantages, and return targets are frozen for all PPO epochs.
They must never be recomputed with an updated critic inside an epoch.

With `gamma_inner = lambda_inner = 1`, inner value terms telescope. Decisions
in one slot share a downstream return sample and differ mainly through their
state-dependent old values. This is expected and must not be described as an
exact difference reward.

### 4.8 Advantage Normalization

Stage 2 actor advantages are normalized once at rollout-batch level using
only `CHOICE_DECISION` records with `actor_mask = 1`.

Normalization excludes:

- boundaries;
- forced decisions;
- skipped decisions.

It is never performed separately per slot, minibatch, decision order, or
seed. The frozen normalized advantages are reused for every PPO epoch.

Critic returns are not actor-advantage normalized.

### 4.9 Value Loss and Shared Optimization

The fixed first diagnostic uses:

\[
L_V
=0.5L_{\mathrm{choice/forced/skip}}
+0.5L_{\mathrm{boundary}}.
\]

Both terms are means within their own class. If a class is absent, only the
existing class contributes and the absence is logged. The 0.5/0.5 choice is
frozen before experiments and is not tuned per seed.

The report separates:

- sample count;
- value loss;
- explained variance;
- target mean and standard deviation

for decision-type records and boundaries.

The shared MLP encoder belongs to one optimizer. The combined actor and value
loss uses one logical zero-grad, backward, clip, and optimizer step per
epoch. Deterministic gradient accumulation is allowed but must reproduce the
full-rollout weighting.

Actor-only, value-only, and combined gradients on the shared encoder are
measured on a fixed diagnostic epoch, including their norms and cosine.

Stage 2 fixes `entropy_coef = 0` for all primary groups. Entropy sensitivity
is a later independent experiment.

### 4.10 Minibatch Rules

The first diagnostic prefers full-rollout logical batches implemented with
deterministic chunked gradient accumulation if needed.

Every epoch guarantees:

- every actor choice decision contributes exactly once to actor loss;
- every decision-type state contributes exactly once to its value-loss
  traversal;
- every boundary state contributes exactly once;
- no `drop_last`;
- no duplication to fill a batch;
- denominators use the total relevant class counts, not chunk sizes.

Optimizer step count, physical-slot count, choice/forced/skip counts, and
boundary count are reported.

## 5. Paired Exogenous Evaluation

Equal seeds alone are insufficient because `active_dag_cap` makes eligibility
policy-dependent and can change RNG consumption order.

Paired evaluation therefore uses an evaluation-only exogenous scenario tape
or keyed RNG:

- arrival random values keyed by episode, physical slot, and UE ID;
- DAG-content random values keyed by the corresponding potential arrival;
- independent keyed streams for hotspot generation, UE movement, and other
  exogenous processes.

Eligibility remains policy-dependent, but each policy sees the same
underlying potential random event for a given episode/slot/UE key. This
changes only the evaluation RNG source, not arrival probabilities, physical
dynamics, reward, or training.

The scenario-tape implementation requires an equivalence smoke showing that
its marginal distributions match the current evaluator on a large read-only
sample. Formal paired comparisons must report the scenario-tape checksum.

## 6. Stage 2 Controls

The scientifically matched primary comparison is:

- **S2-A, matched slot-step control:** slot-level advantage and current
  physical-slot rollout, but using the same shared encoder family, complete
  per-UAV critic information, entropy coefficient zero, and fixed optimizer
  settings as S2-B.
- **S2-B, decision micro-step MAPPO:** choice/forced/skip/boundary stream,
  variable-discount GAE, and no EFT.
- **S2-C, warm-start micro-step MAPPO:** identical to S2-B, but loads the
  matching Stage 1 actor encoder/scorer; optimizer and critic-specific
  parameters start fresh; EFT remains zero.

The current legacy slot-level MAPPO remains **S2-A0**, a historical or
separately rerun reference. S2-A versus S2-B is the causal comparison for
decision granularity; S2-A0 shows how the matched control relates to the
existing mainline.

All groups fix:

- three seeds: 42, 86, and 1042;
- identical actor initialization for S2-A and S2-B;
- the same formal reward and physical-slot interaction budget;
- the same MLP dimensions, movement behavior, arrival settings, PPO epochs,
  and paired scenario tapes;
- zero entropy coefficient;
- the same actor action-loss convention: one loss contribution per actual
  choice action, averaged over choice decisions.

Reports include parameter counts, optimizer steps, actual decision counts,
and all loss denominators.

S2-C is evaluated densely at updates 0, 1, 5, and 10 to record whether a
randomly initialized critic damages the warm-start actor. Results do not
trigger mid-run freezing or tuning.

The first run is a 30-update gate. It is not a final convergence experiment.

## 7. Required Gates and Smokes

### 7.1 Stage 1

Tests must prove:

- executed action, old log-probability, and reservation action are identical;
- PPO epochs do not resample actions;
- the exact old-policy baseline matches hand calculation and is detached;
- clipped PPO matches positive- and negative-advantage hand calculations;
- masks prevent illegal sampling;
- zero/single-candidate records do not enter actor loss;
- every effective decision has equal weight;
- entropy coefficient is zero and creates no loss gradient;
- critic and movement receive no gradients;
- historical tensors are immutable;
- sequential reservation and frozen-ready semantics remain correct;
- all losses, logits, ratios, KL values, and gradients are finite.

### 7.2 Stage 2

Before joint training, all six gates below must pass:

1. **Return equivalence:** on a fixed trajectory, the original slot-discounted
   Monte Carlo return equals the first micro-step discounted return within
   tolerance. This compares returns, not lambda-return targets.
2. **Telescoping hand calculation:** for `D1 -> D2 -> B -> next`, all deltas,
   advantages, and returns match an explicit manual calculation.
3. **Critic-state separability:** identical graph/task inputs with different
   per-UAV reservations produce different critic tensors, including cases
   where mean/max summaries would collide.
4. **Zero/one-candidate stream:** cover no ready task, skip-only, forced-only,
   forced-then-choice, and choice-then-skip slots.
5. **Frozen GAE:** old values, advantages, and returns remain bitwise or
   tolerance-equivalent through three PPO epochs.
6. **Critic-only fit:** on a fixed micro-step rollout with a frozen actor, the
   decision and boundary critic losses can decrease without target or state
   misalignment.

Additional smokes prove:

- exactly one physical executor advance and one reward settlement per
  boundary;
- exactly one boundary per physical slot;
- actionless slots retain their boundary reward;
- boundaries never enter actor loss;
- inner records use gamma and lambda one;
- terminal and truncation occur only at boundaries;
- final-observation bootstrap is correct;
- no stream crosses episode reset;
- scenario-tape keys and checksums are deterministic;
- no reward duplication, transition loss, silent sample drop, or NaN/Inf.

## 8. Metrics and Interpretation

Stage 2 reports:

- decision PPO loss, clip fraction, KL, entropy, probability concentration;
- decision advantage mean/std and within-slot advantage std;
- fraction of same-slot choice pairs with numerically identical advantages;
- decision and boundary value loss and explained variance;
- actor, value, and combined shared-encoder gradients and cosine;
- reward, generated/admitted/blocked/completed DAG counts, blocked reasons,
  completion, flowtime, throughput, queue length, and unfinished work;
- stochastic and deterministic paired closed-loop results.

Stage 2 strongly passes if S2-B is more stable than S2-A across all seeds,
within-slot advantages are not mechanically identical, critic metrics show a
coherent positive trend, and at least one primary environment dimension
improves without seed failure. S2-C should preserve or improve the Stage 1
policy.

A technically correct 30-update run with no clear environment difference is
a partial pass and may justify a longer fixed-configuration run. It does not
justify reward or arrival changes.

Only if Stage 1 strongly passes, every Stage 2 mathematical and implementation
gate passes, the critic can fit fixed data, and formal Stage 2 still fails may
a separate design consider decision-level Q or an outcome ledger.

## 9. Implementation Sequencing

Implementation must remain split:

1. Stage 1 record, objective, smoke, pilot, and three-seed gate.
2. Stop and review Stage 1. Stage 2 is prohibited unless Stage 1 strongly
   passes.
3. Stage 2 transition-schema and mathematical smokes.
4. Evaluation-only exogenous scenario tape and equivalence smoke.
5. Critic-state and critic-only fit gates.
6. Stage 2 short A/B/C gate.
7. Stop and review before any longer experiment or Q design.

No implementation stage may silently expand into the next.
