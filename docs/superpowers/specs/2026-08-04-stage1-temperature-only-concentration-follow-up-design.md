# Stage 1 Temperature-Only Concentration Follow-Up Design

Date: 2026-08-04  
Frozen revision: 2026-08-06  
Branch: `zrj_3multisample`  
Design baseline commit: `6b6cc75f8269a3e99a0e9145b540734be1d86504`  
Design status: `FROZEN_FOR_IMPLEMENTATION_PLANNING`

## 1. Purpose

Stage 1 produced a partial scientific pass. Deterministic masked argmax
improved strongly, but the stochastic sampled policy remained too diffuse:

- sampled EFT regret improved by about 32.7%, below the approximately 50%
  target;
- margin-at-least-20-second sampled accuracy was about 35.7%, far below the
  90% target;
- normalized entropy remained high;
- maximum action probability and the top-1/top-2 probability margin remained
  too small.

The completed active-DAG/queue-cap 2x2 diagnostic also showed that both
capacity factors bind and interact, but it did not explain away the Stage 1
concentration failure. The next question is therefore narrower:

> If the actor's learned ranking is held fixed and only the scale of its
> masked legal logits is sharpened, does the sampled policy approach the
> deterministic policy?

This is a checkpoint-only mechanism diagnostic. It does not retrain a model,
select a deployment temperature, or authorize Stage 2.

## 2. Frozen Scope and Prohibitions

This follow-up preserves:

- the three frozen S1-B update-30 MLP checkpoints;
- the MLP encoder and current offloading scorer;
- the actual `GraphSnapshot.task_features` consumed by the checkpoint;
- the seven-dimensional actor candidate input;
- actor masks, pair/EFT features, sequential temporary reservation, and
  stable task ordering;
- the original Stage 1 environment reward, DAG arrival distribution, DAG
  content distribution, mobility, task lifecycle, executor, and physical
  transition semantics;
- forced UAV hover and the existing movement boundary;
- one physical executor advance, reward settlement, and physical-time
  increment per physical slot.

It prohibits:

- changing actor, scorer, encoder, HGNN, `GraphSnapshot`, or feature shapes;
- changing reward, arrival probability, DAG templates, movement, execution,
  or capacity logic;
- changing the seven-dimensional actor input;
- changing checkpoint weights or using `strict=False`;
- PPO/MAPPO updates, critic execution, GAE, boundary transitions, or Stage 2
  fields;
- entropy-loss changes, optimizer changes, or any retraining;
- adding temperatures after seeing results;
- selecting a deployment temperature from the formal diagnostic set;
- pooling this experiment with the 2x2 capacity pilot or formal results.

Any technical gate failure stops the follow-up. It must not be repaired by
changing experimental semantics after pilot or formal results are visible.

## 3. Primary Environment

The primary diagnostic uses the checkpoint's original Stage 1 capacity
environment:

```text
active_dag_cap = 1
hard_queue_cap = 16
```

This is referred to as capacity environment A. In this document, A means only
these capacity semantics inside the original Stage 1 evaluator. It does not
mean reusing the 2x2 capacity diagnostic runner or its frozen-position tape.
The original Stage 1 mobility, arrival, reward, execution, and episode
semantics remain authoritative.

No C-cell run is part of the primary experiment. A later optional migration
appendix may use:

```text
active_dag_cap = 1
hard_queue_cap = episode-local nonbinding
```

Such a C-cell appendix requires separate authorization, output directories,
and interpretation. It cannot change or be pooled with the primary A result.

## 4. Frozen Checkpoints

The follow-up uses exactly these checkpoints:

```text
training seed 42
logs/decision_ppo_bandit/20260729_215923_stage1_formal_S1-B_seed42/
checkpoints/checkpoint_update_0030.pt
SHA-256 b4f0b84afa5ad443e901ceaa58bdd49f10fe4570655d7ca89f65812ead78c668

training seed 86
logs/decision_ppo_bandit/20260729_220604_stage1_formal_S1-B_seed86/
checkpoints/checkpoint_update_0030.pt
SHA-256 abb7c3c09d61be51860c7020e2555339982b042b4b8de260546e112123a2d643

training seed 1042
logs/decision_ppo_bandit/20260729_221421_stage1_formal_S1-B_seed1042/
checkpoints/checkpoint_update_0030.pt
SHA-256 95d7f1a13dbc671e214ef9162d89c9768a58ab4bc8df13e712817f5931804ac9
```

The loader uses an isolated probe environment and `CleanGraphBuilder` to
prepare one real `GraphSnapshot`, then verifies:

```text
checkpoint_task_feature_dim
= encoder_state_dict["input_proj.weight"].shape[1]

graph_snapshot_task_feature_dim
= probe.graph_snapshot.task_features.shape[1]

checkpoint_task_feature_dim == graph_snapshot_task_feature_dim == 12
```

Only after this equality passes may it construct the MLP encoder and load the
encoder and scorer with `strict=True`. Probe resources and RNG state are
isolated and restored before any tape or formal record is generated.

Every output records the checkpoint path, SHA-256, training seed, completed
update, both observed input dimensions, resolved input dimension, and strict
load result.

## 5. Temperature Semantics

The frozen temperature set is:

```text
T = 1.0, 0.75, 0.5, 0.25
```

No interpolation, additional temperature, or adaptive refinement is allowed
after pilot or formal metrics are inspected.

For every decision:

```text
construct the legal candidate mask using the unchanged environment state
-> extract only legal candidate logits
-> divide legal logits by positive T
-> apply a numerically stable softmax over legal candidates
```

Equivalently, illegal candidates may be set to negative infinity before the
division, provided they remain negative infinity. Temperature must never be
applied to probabilities after softmax.

Required invariants are:

- every temperature is strictly positive;
- every static replay record uses exactly the same legal candidate set;
- masked argmax and its UAV-ID tie-break are identical for every temperature;
- no temperature path changes task order, candidate construction, features,
  EFT, or reservation before action selection;
- normalized entropy is calculated only over legal candidates;
- zero- and one-legal-candidate decisions are excluded from stochastic
  concentration metrics exactly as in Stage 1.

## 6. Keyed Common Sampling Noise

The diagnostic does not use a sequential Python, NumPy, or Torch random stream
for sampled offloading actions. Each legal candidate receives deterministic
Gumbel noise derived from a canonical typed key:

```text
sampling_noise_key = [
  schema_version,
  checkpoint_sha256,
  evaluation_scenario_seed,
  slot_index,
  stable_task_id,
  decision_order,
  sampling_replicate,
  candidate_uav_id
]
```

The schema version is the literal string
`stage1_temperature_gumbel_v1`. The key must not contain temperature. It is
serialized as a canonical compact JSON UTF-8 array and hashed with SHA-256.
The conversion is frozen as:

```text
digest = SHA256(canonical_key_bytes)
h = unsigned_big_endian_integer(digest[0:8])
m = h >> 11
u = (m + 0.5) / 2^53
gumbel = -log(-log(u))
```

This gives an open-interval float64 uniform value without rounding to zero or
one. Legal actor logits are copied without mutation, converted to CPU float64,
and used for temperature softmax and Gumbel-max selection. These formulas are
covered by hand calculations and cross-process smoke tests.

For temperature `T`, the sampled legal action is:

```text
argmax(logit_i / T + keyed_gumbel_i)
```

with UAV ID as the final exact-tie break. This is the Gumbel-max categorical
sample from `softmax(legal_logits / T)`.

The same checkpoint, scenario, state, task, decision order, sampling replicate,
and candidate UAV therefore use the same noise at all temperatures. The
scenario seed uniquely identifies the episode, so no redundant episode field
is present in the sampling key. Different checkpoints and sampling replicates
remain independent.

## 7. Evaluation Scenario Tape

Equal initial seeds alone are insufficient because sampled decisions change
active-DAG eligibility, `service_waiting`, later state, and legacy RNG
consumption. Before any pilot result is inspected, the implementation
generates and freezes one evaluation-only random-material tape for the
original Stage 1 environment.

The formal scenario seeds are fixed as the existing checkpoint evaluator's
20 scenarios:

```text
424242, 424243, ..., 424261
```

The mapping and episode horizon are frozen as:

```text
episode_index = 0, 1, ..., 19
evaluation_scenario_seed = 424242 + episode_index
slot_index = 0, 1, ..., 199
maximum physical slots per episode = 200
```

Every evaluation scenario is one independently reset episode. A successful
formal row executes 200 physical slots unless the unchanged environment emits
a genuine terminal earlier; an exception or technical abort is not a shorter
valid episode. This follow-up does not use the 2x2 diagnostic's 150-slot load
plus 50-slot drain schedule. Machine-readable result rows retain both
`episode_index` and `evaluation_scenario_seed` and assert the mapping above,
but the sampling-noise key uses only the unique scenario seed.

The tape freezes random material, never trajectory-dependent final outcomes.
It contains:

- raw uniforms for the episode hotspot center;
- raw initial UE position, walk-speed, and heading draws for the authoritative
  post-reset UE state;
- raw initial UAV position draws for the authoritative post-reset UAV state;
- per-slot/per-UE Gaussian speed and heading innovations for the unchanged
  Gaussian-Markov movement update;
- one per-slot/per-UE arrival uniform, whether or not that draw later becomes
  eligible for use;
- one complete potential DAG template per slot and UE, generated from isolated
  keyed DAG random material but not instantiated unless an arrival actually
  occurs;
- any other exogenous random primitive whose sequential consumption could
  otherwise diverge after an action change.

The tape must not store a precomputed final UE position, hotspot-membership
bit, arrival-event bit, generated-DAG bit, or admission result. For each
temperature trajectory, the original Stage 1 order is executed exactly:

```text
read the slot/UE movement innovations
-> update walk speed and heading from that trajectory's previous UE state
-> apply that trajectory's current service_waiting speed scale
-> reflect and commit the resulting UE position
-> freeze the current service position
-> check active_dag_cap using that trajectory's current DAG state
   -> ineligible: increment arrival_blocked_count and do not evaluate arrival
   -> eligible: calculate hotspot arrival probability from the current UE position
      -> compare the frozen arrival uniform u with the current probability p
      -> u >= p: record no event
      -> u < p: instantiate the already-frozen potential DAG template
```

The potential arrival uniform and potential DAG template exist even when the
active cap prevents their use, but an ineligible UE must not increment
`arrival_draw_count`, inspect `u < p`, instantiate a template, or create a DAG.
This preserves the original Stage 1 cap-before-draw semantics and explicitly
forbids the 2x2 diagnostic's offered-before-cap semantics.

The potential DAG template freezes only random DAG content: structure,
task-local attributes, dependency topology, bandwidth draws, and other values
normally sampled by `create_dag_for_ue`. It must not freeze the
trajectory-dependent DAG `source_pos` or `arrival_time`. When `u < p`, the
template is instantiated with that trajectory's current frozen UE service
position and current physical time, exactly where the original Stage 1 path
would create the DAG.

Eligibility, final UE positions, hotspot membership, realized arrivals,
admission, task readiness, queues, execution, and reward remain
trajectory-dependent and may differ across temperatures. The tape changes
only the source of random primitives, not their distributions, state
dependence, ordering, or transition rules.

Required tape gates include:

- canonical SHA-256 checksum and exact round-trip;
- no key reuse across scenario seed, slot, UE, event type, or DAG-local task;
- identical random-material checksum for all temperatures sharing a scenario;
- equivalence smokes against the current Stage 1 evaluator for per-UE arrival,
  eligible arrival-draw counts, blocked-before-draw counts,
  hotspot-conditioned arrival, service-waiting-dependent movement, DAG type,
  DAG size, and DAG-content distributions;
- exact reproducibility under the same tape checksum.

Every potential DAG template has a trajectory-independent stable ID:

```text
stable_dag_id
= SHA-256(canonical JSON [
    "stage1_temperature_dag_v1",
    evaluation_scenario_seed,
    slot_index,
    ue_id
  ])

stable_task_id
= stable_dag_id + "_task_" + zero_padded_DAG_local_task_index
```

The original environment has one arrival opportunity and therefore at most one
potential DAG-template key per UE and slot, so this key is unique. IDs are
assigned in the tape before active-cap
eligibility is evaluated and therefore do not depend on admission history or
temperature. A stable ID does not mean a DAG was offered, generated, or
admitted. Global uniqueness, round-trip, and admission-independence are hard
gates.

The 2x2 capacity tape must not be reused because its movement and diagnostic
semantics are not the original Stage 1 environment.

## 8. Two Separate Analyses

### 8.1 Static masked-logit replay

For each checkpoint, evaluation scenario, and sampling replicate, the keyed
`T=1.0` closed-loop rollout records an immutable corpus of every nontrivial
decision state:

- checkpoint identity;
- scenario seed, derived episode index, slot, stable task, and decision-order
  identifiers;
- exact legal candidate UAV IDs and mask;
- immutable actor-consumed tensors or their verified replay reference;
- raw legal logits before temperature;
- decision-time EFT for every legal candidate;
- deterministic greedy-EFT action and masked actor argmax;
- keyed per-candidate Gumbel noise;
- complete record checksum.

The corpus is frozen before replay metrics for lower temperatures are
inspected. Every temperature is then evaluated on these exact records without
advancing an environment. The state, candidates, logits, EFT, and noise are
identical; only `logits / T` changes.

This is the primary concentration-mechanism analysis and supports strict
within-record comparisons across temperatures.

### 8.2 Closed-loop temperature rollout

Each temperature also executes its sampled actions in the original Stage 1 A
environment using the same random-material tape and keyed-noise contract. It
reports whether sharpening the distribution improves or damages actual
trajectories.

Once two temperatures choose different actions, their later states and legal
candidates may diverge. Closed-loop rows may therefore be described only as:

```text
same initial scenario
same random-material tape
same keyed-noise rule
matched-start comparison
```

They must not be called strict per-decision counterfactual pairs. The static
replay and closed-loop results use separate schemas, files, summaries, and
scientific conclusions.

## 9. T=1 Baseline Compatibility

The new keyed `T=1.0` condition is the sole internal baseline for all formal
temperature reductions. It must pass these compatibility checks:

- on an identical static record, unscaled masked logits equal the actor's
  current logits;
- `softmax(logits / 1.0)` equals the current masked softmax with maximum
  absolute probability error at most `1e-6`;
- deterministic argmax and tie-break equal the current actor;
- masks, candidate EFT, and actor-consumed features equal the current path;
- the checkpoint and environment configuration match the historical Stage 1
  checkpoint-only evaluator.

The historical evaluator used sequential Torch sampling. The new keyed
Gumbel sampler therefore is not required to reproduce its stochastic action
sequence or closed-loop trajectory exactly. Historical T=1 summaries remain
context only and are never used as the denominator of a formal temperature
effect.

## 10. Sampling Replicates and Formal Matrix

Sampling replicates are frozen as:

```text
sampling_replicate = 0, 1, 2, 3, 4
```

The formal closed-loop matrix is:

```text
3 frozen checkpoints
x 20 evaluation scenarios
x 5 keyed sampling replicates
x 4 temperatures
= 1200 closed-loop episodes
```

The `T=1.0` members of this matrix also generate the static decision corpus.
Static replay evaluates all four temperatures on every corpus record without
additional environment rollouts.

Before formal execution, an isolated technical pilot uses the already-frozen
formal tape prefix:

```text
3 checkpoints
x scenarios 424242 and 424243
x sampling replicates 0 and 1
x 4 temperatures
= 48 closed-loop pilot episodes
```

Pilot outputs are technical evidence only. They are stored separately and
are not pooled with formal results or used to change temperatures, thresholds,
replicate count, or scenario count.

## 11. Static Replay Metrics

For nontrivial decisions with at least two legal candidates, report separately
for every checkpoint, temperature, scenario, and sampling replicate:

- sampled raw EFT regret mean, median, and p95;
- sampled greedy-EFT agreement;
- margin-at-least-5-second sampled accuracy;
- margin-at-least-20-second sampled accuracy;
- normalized entropy;
- maximum legal action probability;
- top-1/top-2 legal probability margin;
- valid decision count and each margin subset count;
- deterministic masked-argmax EFT regret and greedy agreement as invariants.

Definitions remain identical to Stage 1. The greedy action is the legal action
with minimum decision-time EFT, with UAV ID as its stable tie-break. The EFT
margin is the second-lowest legal EFT minus the lowest legal EFT. A sampled
margin accuracy is one only when the sampled action equals the greedy-EFT
action.

Normalized entropy for `K` legal candidates is:

```text
-sum_i(p_i log p_i) / log(K)
```

The maximum probability and probability margin are calculated directly from
the temperature-scaled legal distribution, not estimated from samples.

For each checkpoint and candidate temperature, mean-regret reduction is
defined relative to that same checkpoint's keyed T=1 records:

```text
regret_reduction(T)
= (mean_regret(T=1) - mean_regret(T)) / mean_regret(T=1)
```

A zero T=1 denominator is reported as null and cannot contribute to a pass.
Metrics are first aggregated within scenario and sampling replicate, then by
checkpoint. Decisions are not treated as independent replicates.

The paired bootstrap resamples the 20 evaluation scenarios with replacement,
retains all five sampling replicates inside each selected scenario block, and
uses 10,000 resamples generated from the literal analysis seed `20260804`.
It never resamples individual decisions as independent observations.

## 12. Closed-Loop Metrics

Closed-loop summaries report, by checkpoint and temperature:

- episode reward total;
- completed DAG count;
- the original Stage 1 `generated_dag_count`;
- the original Stage 1 `arrival_admitted_count` and
  `arrival_blocked_count`;
- `arrival_attempt_count`, `arrival_draw_count`,
  `arrival_sampled_event_count`, and `arrival_no_event_count`;
- completion rate;
- average DAG flowtime with explicit null handling;
- average UAV queue length;
- end-of-episode active, ready, queued, and admitted-incomplete backlog;
- choice, forced, and skipped decision counts;
- invalid assignment and non-finite counters.

The report must not use `offered_dag_count` or describe a potential template
as an offered DAG. If a human-readable `admitted_dag_count` field is added, it
is only a read-only alias and must satisfy exactly:

```text
admitted_dag_count == arrival_admitted_count == generated_dag_count
```

for the original Stage 1 funnel. No `potential_arrival_count` is introduced
in this follow-up.

The report retains all checkpoint-level results. It may report overall means
only after the three checkpoint results are shown and must not use one strong
checkpoint to hide another checkpoint's degradation.

Closed-loop matched-start deltas use the keyed T=1 condition with the same
checkpoint, scenario, and sampling replicate. These deltas reduce noise but
remain non-counterfactual after trajectory divergence.

## 13. Deterministic Reachability Gate

Temperature scaling cannot change masked-logit ordering. Before interpreting
any sampled temperature result, the static corpus therefore calculates the
following ceiling separately for every checkpoint:

```text
deterministic_margin20_accuracy
deterministic_mean_EFT_regret

max_achievable_regret_reduction
= (T1_sampled_mean_EFT_regret - deterministic_mean_EFT_regret)
  / T1_sampled_mean_EFT_regret
```

All terms use the same immutable T=1 static records, legal masks, EFT vectors,
and Stage 1 metric definitions. The deterministic action is the unchanged
masked argmax with the frozen UAV-ID tie-break. Exact top-logit ties retain the
current tie-break; no additional scientific metric is introduced.

The reachability result is checkpoint-local:

- if `deterministic_margin20_accuracy < 0.90`, that checkpoint is
  `ranking_limited_margin20`;
- if `max_achievable_regret_reduction < 0.50`, that checkpoint is
  `ranking_limited_regret`;
- if either condition holds for any checkpoint, the three-checkpoint formal
  conclusion is `ranking_limited` and probability scale cannot be declared
  the primary explanation of the original Stage 1 failure;
- the temperature sweep still runs and reports how much of the reachable gap
  it closes, but it cannot receive the moderate-temperature or
  hard-sharpening concentration-primary labels in Section 14.

When the T=1 sampled mean regret is exactly zero, the regret-reduction ceiling
is null. If deterministic regret is also zero, the checkpoint has no remaining
mean-regret gap and the 0.50 regret-reduction requirement is marked not
applicable rather than failed. Any other zero-denominator case is a hard
analysis error.

This gate does not add a rollout or model evaluation. It uses the deterministic
metrics already required in the static corpus.

## 14. Predeclared Interpretation Gates

### 14.1 Moderate-temperature concentration evidence

A single common moderate temperature, either `T=0.75` or `T=0.5`, supports
the conclusion that probability scale is the primary Stage 1 limitation only
if all of the following hold:

1. all three checkpoints have positive mean sampled-EFT-regret reduction;
2. a scenario-block paired bootstrap 95% interval for each checkpoint's
   static mean-regret improvement has a lower bound above zero;
3. at least two checkpoints have regret reduction at least 0.50;
4. all three checkpoints have margin-at-least-20-second sampled accuracy at
   least 0.90;
5. sampled greedy agreement improves for all three checkpoints;
6. normalized entropy decreases, while maximum probability and probability
   margin increase, without any argmax or mask change;
7. the deterministic reachability gate in Section 13 permits a
   concentration-primary conclusion;
8. the closed-loop guardrail in Section 14.4 passes.

The same temperature must satisfy the gates across checkpoints. A different
temperature may not be selected separately for each checkpoint.

### 14.2 Hard-sharpening-only evidence

If neither moderate temperature passes but `T=0.25` satisfies items 1 through
6 of Section 14.1 with `T=0.25`, the Section 13 reachability gate, and the
Section 14.4 closed-loop guardrail,
the conclusion is limited to:

```text
ranking partially validated
original stochastic concentration gate failed
near-hard sharpening was required
```

This is not a Stage 1 strong pass and does not select `T=0.25` for deployment.

### 14.3 Concentration-only hypothesis rejected

This category is available only when the Section 13 deterministic reachability
gate passes for all three checkpoints. Under that condition, the
concentration-only hypothesis is rejected if `T=0.25` fails any of the
following for any checkpoint, or fails the shared closed-loop guardrail:

- positive sampled-regret improvement with a positive static paired interval;
- margin-at-least-20-second accuracy of 0.90;
- improved sampled greedy agreement;
- deterministic argmax invariance;
- the closed-loop guardrail.

The resulting conclusion is that flat probability scale is not sufficient to
explain the Stage 1 stochastic failure. Follow-up analysis may inspect scorer
separation, pair/EFT feature use, state-dependent ranking errors, and training
versus closed-loop state distribution. It still does not automatically
authorize Stage 2.

If Section 13 fails, `ranking_limited` is the authoritative conclusion instead
of this category, regardless of the T=0.25 result.

### 14.4 Closed-loop non-degradation guardrail

For a checkpoint and temperature, material degradation relative to its keyed
T=1 mean is predeclared as any of:

- completed DAG count decreases by at least 5%;
- completion rate decreases by at least 0.05 absolute;
- normalized reward change is at most -10%, using
  `delta / max(abs(T1 reward), 1)`;
- average flowtime increases by at least 10%;
- end-of-episode admitted-incomplete backlog increases by at least 10%.

For each checkpoint, `material_degradation_triggered` is true when at least one
material threshold above is met. The guardrail fails when either:

1. any one checkpoint has a catastrophic regression; or
2. at least two of the three checkpoints have
   `material_degradation_triggered = true`, even when they degrade on different
   metrics.

A same-metric material regression across all three checkpoints remains a
mandatory highlighted report item, but it is not the only cross-checkpoint
failure rule. A catastrophic single-checkpoint regression is any of:

- completed DAG count decreases by at least 15%;
- completion rate decreases by at least 0.10 absolute;
- normalized reward change is at most -25%;
- flowtime or admitted-incomplete backlog increases by at least 25%.

Null flowtime is handled using the existing completed-count semantics and may
not be converted to zero. Guardrail thresholds are frozen mechanism-safety
rules, not claims of statistical equivalence.

## 15. No Temperature Selection on the Formal Set

The sweep answers whether systematic sharpening closes the sampled-versus-
deterministic gap. It does not choose an inference or deployment temperature.

The report may state:

- moderate sharpening succeeds;
- only near-hard sharpening succeeds;
- sharpening fails;
- lower temperatures show a monotonic or non-monotonic response.

It must not state that the temperature with the highest formal reward is the
new production setting. Any later proposal to adopt a temperature requires a
separately authorized confirmation on evaluation scenarios not used in this
diagnostic, with the temperature fixed before those scenarios are opened.

## 16. Technical Gates

Before the pilot:

1. all three checkpoint paths and SHA-256 values match Section 4;
2. checkpoint and actual graph dimensions both equal 12;
3. encoder and scorer strict loads pass with no missing or unexpected keys;
4. probe RNG state is restored and probe resources are closed;
5. the random-material tape passes checksum, round-trip, uniqueness,
   distribution, and reproducibility smokes;
6. primary environment capacity is exactly active 1 and queue 16;
7. actor input remains seven-dimensional and the encoder remains MLP;
8. legal masks, raw logits, EFT vectors, argmax, and tie-break match the
   current Stage 1 path at T=1;
9. temperatures act on legal logits before softmax;
10. every temperature is positive and the set is exactly frozen;
11. keyed Gumbel values are identical across temperatures and stable across
    repeated processes;
12. keyed Gumbel values differ across sampling replicate or candidate UAV;
13. the key does not contain temperature;
14. static replay never advances or mutates an environment;
15. static record checksums are stable and all required tensors are immutable;
16. deterministic argmax and candidate masks are identical at all T;
17. T=1 probabilities match the current masked softmax within the frozen
    tolerance;
18. static metric hand calculations pass for constructed logits, masks, EFT,
    ties, zero-regret, and margin subsets;
19. closed-loop arrival, movement, reward, transition, and capacity semantics
    match the original Stage 1 evaluator;
20. no training optimizer, critic, GAE, boundary, HGNN, or Stage 2 path runs.
21. the tape contains movement innovations and arrival uniforms, not final UE
    positions, hotspot-membership bits, arrival bits, or admission outcomes;
22. constructed service-waiting cases prove the same movement innovation can
    produce different valid final positions through the unchanged speed scale;
23. active-cap rejection occurs before arrival-uniform evaluation and DAG
    instantiation, with original funnel counters matching hand calculations;
24. every result satisfies
    `evaluation_scenario_seed == 424242 + episode_index`, uses slot indices
    `0..199`, and contains no 2x2 load/drain phase;
25. deterministic reachability metrics and zero-denominator handling match
    constructed ceiling calculations;
26. closed-loop guardrail smokes include different material-degradation
    metrics on two checkpoints and a catastrophic single-checkpoint case.

Pilot pass additionally requires:

```text
48/48 unique closed-loop pilot rows
all expected static records and replay rows exist exactly once
no NaN or Inf
no invalid assignment
no traceback or OOM
all tape and checkpoint checksums match
all T=1 compatibility gates pass
all-temperature argmax and mask invariance passes
static replay does not change its source corpus checksum
every successful pilot episode executes 200 physical slots
no offered-before-cap or offered_dag_count semantics appear
pilot and formal output directories are distinct
```

The pilot does not apply the scientific thresholds in Sections 13 and 14 and
cannot be used to change the design.

## 17. Reporting and Statistical Units

Static comparisons align by the exact immutable record key plus sampling
replicate and are strict within-record temperature comparisons. Scenario is
the primary resampling block; individual decisions are not treated as
independent experimental replicates.

Closed-loop summaries retain the exact key:

```text
checkpoint training seed
+ evaluation scenario seed
+ derived episode_index
+ sampling replicate
+ temperature
```

Matched-start differences align temperature with keyed T=1 by checkpoint,
scenario seed, derived episode index, and sampling replicate. Reports must
repeat that subsequent states are not strict counterfactual pairs after the
first action divergence.

Every formal report contains:

- code commit and clean/dirty status;
- runner arguments and environment constants;
- checkpoint paths and SHA-256 values;
- random-material-tape and static-corpus checksums;
- exact row counts and duplicate/missing-key audit;
- per-checkpoint results before any aggregate;
- all four temperatures, including failed conditions;
- static and closed-loop conclusions in separate sections;
- the deterministic reachability result from Section 13;
- the final category from Section 14;
- an explicit statement that no deployment temperature or Stage 2 transition
  was authorized.

## 18. Output Isolation

Pilot and formal outputs use new, non-overlapping directories under protected
`logs/` or `runs/`. The required logical layout is:

```text
stage1_temperature_followup/
  frozen_inputs/
  pilot/
    static/
    closed_loop/
  formal/
    static_corpus/
    static_replay/
    closed_loop/
    analysis/
```

Existing Stage 1 logs, 2x2 capacity logs, tapes, and runs are never deleted,
overwritten, or merged. Formal runners must refuse to start when their target
output directory already exists.

## 19. Frozen Execution Order

```text
approve this design
-> write and approve a separate implementation plan
-> implement isolated tape, keyed sampling, static-record, replay, and runner paths
-> run local non-Torch structural checks
-> sync only explicitly approved files to the server
-> run isolated Torch and distribution-equivalence smokes
-> generate and checksum the complete formal random-material tape
-> run the 48-episode technical pilot on the formal tape prefix
-> stop on any technical gate failure without changing semantics
-> freeze code, checkpoints, tape, corpus rules, temperatures, and runner arguments
-> run the 1200-episode formal matrix
-> verify technical gates
-> calculate deterministic reachability ceilings for all three checkpoints
-> analyze static concentration metrics
-> analyze closed-loop guardrails separately
-> assign ranking_limited or exactly one reachable predeclared temperature category
-> stop for review
```

No implementation, pilot, formal experiment, retraining, deployment-temperature
selection, or Stage 2 work is authorized by this design document alone.
