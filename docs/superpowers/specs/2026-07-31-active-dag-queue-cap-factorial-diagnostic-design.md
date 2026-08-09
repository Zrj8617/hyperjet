# Active-DAG and Queue-Cap 2x2 Factorial Diagnostic Design

Date: 2026-07-31  
Last revised: 2026-08-03  
Status: frozen and implementation-ready; implementation not started

## 1. Purpose

This diagnostic separates two capacity mechanisms that currently interact in
the clean HyperUAV environment:

1. the upstream per-UE active-DAG admission limit; and
2. the downstream per-UAV hard queue limit.

The experiment answers two distinct questions:

- Which constraint most reduces the number of offloading decisions from which
  an actor can learn?
- Which constraint most limits offered-load service, DAG completion, and
  flowtime?

It does not select the final physical system model by performance alone. A
constraint may be scientifically necessary even if removing it improves an
experimental metric.

## 2. Existing Evidence and Motivation

The current clean baseline uses:

```text
max_active_dags_per_ue = 1
CLEAN_MAX_QUEUE_PER_UAV = 16
NUM_UES = 60
NUM_UAVS = 5
```

In the completed Stage 1 S1-B training cells, 88,076 of 113,616 ready-task
decision attempts had no legal candidate. Only 17,458, or about 15.37%, were
choice decisions that entered actor optimization. Across all six S1-A/S1-B
cells, `skipped_no_candidate` and environment reward had a pooled correlation
of approximately -0.726.

Those observations establish that capacity constraints participate in the
current behavior, but they do not causally separate the upstream active-DAG
cap from the downstream queue cap. The two-by-two experiment is intended to
provide that separation before any Stage 2 MAPPO work.

## 3. Fixed Scope and Non-Goals

This phase is an environment diagnostic using fixed policies. It does not:

- train or retrain PPO/MAPPO;
- implement Stage 2 critic, GAE, or boundary transitions;
- change the formal reward;
- change HGNN, MLP, actor, scorer, GraphSnapshot, or candidate feature
  dimensions;
- use the results to tune a policy;
- infer that a capacity constraint is physically correct merely because it
  improves a metric;
- delete or overwrite existing logs, runs, datasets, or checkpoints.

The first strict factorial freezes UE mobility and UAV movement. Normal UE
mobility is reserved for a separately authorized robustness check after the
capacity semantics have been reviewed.

## 4. Factorial Cells

The experiment has four capacity cells:

| Cell | Per-UE active-DAG cap | Per-UAV hard queue cap |
|---|---:|---:|
| A | 1 | 16 |
| B | nonbinding | 16 |
| C | 1 | nonbinding |
| D | nonbinding | nonbinding |

The cells differ only in the two hard capacity checks. Feature dimensions,
feature normalization, policies, initial positions, hotspot, offered workload,
DAG contents, episode length, drain interval, and metrics are identical.

## 5. Immutable Scenario and Offered-Arrival Tape

### 5.1 Frozen scenario state

For each `(scenario_seed, episode)` the tape fixes:

- hotspot center and radius;
- all initial UE positions;
- all initial UAV positions;
- UE mobility disabled for the entire episode;
- UAV movement fixed to hover for the entire episode.

This prevents service-waiting state from changing UE movement, hotspot
membership, arrival probability, or communication distance differently across
capacity cells.

### 5.2 Offer key

Every possible external arrival during the load phase has the stable key:

```text
OfferKey = (scenario_seed, episode, slot_index, ue_id)
```

The load phase contains one opportunity for each UE in each load slot. Offer
generation happens before and independently of active-DAG admission.

All implementation-facing slot identifiers are zero-based:

```text
Env slot_index 0-149   = documented physical slots 1-150, load phase
Env slot_index 150-199 = documented physical slots 151-200, drain phase
documented physical_slot = Env slot_index + 1
```

`OfferKey`, stable DAG/task IDs, tape lookup, checksums, and machine-readable
result rows use the zero-based `slot_index`. A human-facing report may also
include the one-based `physical_slot`, but it must derive it as
`slot_index + 1` rather than storing an independent counter.

`arrival_opportunity_count` counts only the first 150 load slots. The drain
phase creates neither arrival opportunities nor `no_arrival_event_count`
increments.

For every diagnostic episode:

```text
arrival_opportunity_count = 150 * NUM_UES
```

### 5.3 Full immutable DAG template

Each offered event is bound to one immutable `DAGTemplate` containing every
random quantity needed to instantiate the DAG, including:

- DAG layer count and tasks per layer;
- dependency edges;
- stable DAG ID and stable task IDs derived from `OfferKey`;
- task input and output sizes;
- task computation requirements;
- upload and download bandwidth values;
- every other random DAG or task attribute consumed by the current generator.

Stable identifiers use the canonical JSON and SHA-256 contract:

```text
stable_dag_digest
= SHA-256(UTF-8 canonical JSON array:
    [scenario_seed, episode, slot_index, ue_id]
  ).hexdigest()

stable_dag_id
= "diag_dag_" + stable_dag_digest

stable_task_id
= stable_dag_id + "_task_" + zero-padded local_task_index
```

`local_task_index` is the unique zero-based task index inside the immutable
template. Across the complete three-seed, 20-episode tape, the generator must
assert:

```text
all stable_dag_id values are globally unique
all stable_task_id values are globally unique
template round-trip preserves every DAG/task ID exactly
```

The template is generated whether or not a capacity cell later admits the
offer. Rejection must not change any subsequent random-number consumption,
IDs, or templates.

All hashes use one canonical serialization contract:

```text
UTF-8 JSON
arrays retain declared element order
objects sort keys recursively
separators are exactly (",", ":") with no extra whitespace
NaN and Infinity are forbidden
```

Canonical serialization is used to calculate:

```text
scenario_tape_checksum
offered_event_checksum
offered_template_checksum
```

All policies and all four cells for the same scenario must report identical
checksums.

### 5.4 Stable hash input

Every keyed policy or tape hash consumes typed values through the canonical
JSON contract rather than ad-hoc string concatenation. In particular,
`random_hash` hashes the UTF-8 canonical JSON array:

```json
[scenario_seed, episode, slot_index, stable_task_id, uav_id]
```

Scenario and DAG-template checksums use the same canonical JSON rules.

### 5.5 Correct admission funnel

The diagnostic arrival order is:

```text
read OfferKey and offered-event bit
-> no offer: increment no_arrival_event_count
-> offer: read the already-fixed DAGTemplate
-> check the active-DAG cap
   -> capacity available: instantiate template and admit it
   -> capacity unavailable: record blocked offered event; do not instantiate
```

Required counters are:

```text
arrival_opportunity_count
offered_dag_count
offered_subtask_count
active_cap_blocked_offered_count
admitted_dag_count
admitted_subtask_count
no_arrival_event_count
```

The legacy `arrival_blocked_count` is not interpreted as offered demand,
because the current baseline checks eligibility before drawing an arrival.

### 5.6 Formal tape generated before the pilot

Before any pilot cell runs, the implementation generates and freezes the full
20-episode tape for every scenario seed:

```text
scenario seed 42   -> episodes 0-19
scenario seed 86   -> episodes 0-19
scenario seed 1042 -> episodes 0-19
```

It then calculates all formal checksums and every episode-local nonbinding cap.
The pilot reads episodes `0-4` from this already-frozen formal tape. It must not
generate a separate five-episode tape. The formal run later reads episodes
`0-19` from exactly the same artifact.

The pilot records both the full-tape checksum and the checksum of its selected
prefix. A mandatory gate verifies:

```text
pilot prefix checksum
= checksum of formal tape episodes 0-4 under the same canonical serialization
```

No tape, template, nonbinding cap, or workload distribution is regenerated or
changed after pilot outcomes are inspected.

## 6. Episode-Local Nonbinding Caps

The environment resets between episodes. Nonbinding values are therefore
calculated from each episode's tape, not by accumulating offers across
episodes.

For active DAGs:

```text
active_nonbinding_cap(scenario_seed, episode)
= max over ue_id(
    offered DAG count for that UE in that episode
  ) + 1
```

For the per-UAV queue:

```text
queue_nonbinding_cap(scenario_seed, episode)
= offered subtask count in that episode + 1
```

The queue bound is deliberately conservative. At any instant, one UAV cannot
contain more admitted tasks than the total number of subtasks offered during
that episode, including executor entries and same-slot temporary
reservations.

Each row records the actual episode-local bounds. The following assertions are
mandatory:

```text
B and D: active_cap_blocked_offered_count == 0
C and D: queue_full_mask_count == 0
```

A violation invalidates the pilot or formal cell. It must not be silently
treated as a nonbinding condition.

## 7. Hard Queue Capacity and Feature Normalization

The hard legality cap is separated from all feature normalization references.
Changing the C/D treatment must not rescale the actor input.

The diagnostic keeps the current seven-dimensional UAV feature interface and
freezes the original references:

```text
HARD_QUEUE_CAP = 16 or episode-local nonbinding value
QUEUE_LENGTH_NORM_REF = 16
REMAINING_SLOTS_FEATURE_REF = 16
SLOT_ASSIGNED_NORM_REF = 16
QUEUE_WORKLOAD_NORM_REF = 80_000_000.0 operations
```

The workload reference is frozen as a literal diagnostic constant. It equals
the current baseline calculation:

```text
1_000_000 operations/second x 5.0 seconds x 16
= 80_000_000.0 operations
```

The implementation asserts at runtime:

```text
QUEUE_WORKLOAD_NORM_REF == 80_000_000.0
```

It must not silently change if UAV compute rate, slot duration, or a production
queue configuration changes later.

The diagnostic feature values remain:

```text
queue_length_feature
= clip(queue_length / QUEUE_LENGTH_NORM_REF, 0, 1)

remaining_slots_feature
= clip(max(REMAINING_SLOTS_FEATURE_REF - queue_length, 0)
       / REMAINING_SLOTS_FEATURE_REF, 0, 1)

slot_assigned_feature
= clip(slot_assigned / SLOT_ASSIGNED_NORM_REF, 0, 1)
```

Only `HARD_QUEUE_CAP` controls legality. In nonbinding cells, queue features
may saturate beyond the original range. This is an expected distribution
shift for the existing actor and is one reason its result is auxiliary only.

No candidate dimension, actor dimension, or checkpoint tensor shape changes.

### 7.1 One capacity context for every legality path

Each episode creates one immutable diagnostic capacity context containing its
`HARD_QUEUE_CAP`. That same context is injected into every queue-legality path,
including:

- actor candidate construction and mask generation;
- environment assignment-buffer filtering;
- executor assignment commit validation.

No diagnostic path may read a separate legacy value of 16 for legality after
the context is created. This requirement does not mandate a production-code
refactor; it mandates one source of truth within the diagnostic path.

Every accepted assignment records its decision-time and commit-time legality.
The following invariants are hard gates:

```text
decision-time legality == commit-time legality
invalid_assignment_reasons["queue_cap_mismatch"] == 0
```

The fixed normalization references remain outside the capacity context and do
not change between cells.

## 8. Candidate Mask and Skip Reasons

Instrumentation must evaluate and retain explicit reasons rather than only a
single aggregate skip count.

At minimum, reason-level counters distinguish:

```text
queue_full
task_not_ready
already_reserved
already_scheduled
invalid_uav
other
```

Required outputs include:

```text
candidate_mask_reason_count
candidate_mask_reason_count_by_uav
skip_reason_count
```

If more than one condition is true, the diagnostic retains the complete reason
set for correctness analysis and also uses one documented deterministic
primary-reason precedence for aggregate reporting. It must not discard an
invariant violation merely because `queue_full` is also true.

`all_uavs_full_decision_count` increases only when:

1. all five UAV candidates are illegal;
2. every UAV is explicitly marked `queue_full`; and
3. no task-level consistency reason such as `task_not_ready`,
   `already_reserved`, or `already_scheduled` is present.

All other zero-candidate cases are reported under their actual reason
signature.

## 9. Fixed Policies

### 9.1 Primary policy: random_hash

`random_hash` never calls Python's built-in `hash()` and never consumes a
sequential random stream. For every legal task-UAV pair it computes:

```text
SHA-256(UTF-8 canonical JSON array:
  [scenario_seed, episode, slot_index, stable_task_id, uav_id]
)
```

The digest is converted to an unsigned integer. The legal UAV with the
smallest value is selected, with UAV ID as the final deterministic tie-break.
Changing a candidate mask only removes candidates; it does not change the
relative priority of the remaining candidates.

### 9.2 Primary policy: greedy_eft

`greedy_eft` selects the legal candidate with minimum decision-time EFT under
the current sequential temporary reservation. Exact EFT ties use UAV ID as a
stable tie-break. It receives no future information from the tape.

### 9.3 Auxiliary policy: Stage 1 deterministic actor

The S1-B update-30 MLP checkpoint is selected by scenario seed using the fixed
mapping:

```text
scenario seed 42
-> logs/decision_ppo_bandit/20260729_215923_stage1_formal_S1-B_seed42/
   checkpoints/checkpoint_update_0030.pt

scenario seed 86
-> logs/decision_ppo_bandit/20260729_220604_stage1_formal_S1-B_seed86/
   checkpoints/checkpoint_update_0030.pt

scenario seed 1042
-> logs/decision_ppo_bandit/20260729_221421_stage1_formal_S1-B_seed1042/
   checkpoints/checkpoint_update_0030.pt
```

Each checkpoint is loaded strictly and uses masked argmax. Every result row for
this policy records:

```text
actor_checkpoint_path
actor_checkpoint_sha256
actor_training_seed
actor_completed_update = 30
```

Missing files, SHA changes within one comparison, seed mismatches, partial
loads, or a completed update other than 30 are hard failures. The actor was
trained only under cell A semantics and is therefore reported as a migration
or out-of-distribution diagnostic in B/C/D.

Its results are excluded from the core causal conclusion about capacity main
effects. No policy is retrained in any cell.

## 10. Sequential Offloading Semantics

Every cell preserves the current clean ordering:

```text
freeze ready tasks
-> process them in stable order
-> construct mask/features/EFT from the current reservation
-> choose one action when legal
-> update temporary reservation immediately
-> next ready task sees the updated reservation
-> commit assignments once
-> advance the executor once
```

The tracker distinguishes:

- executor queue length at slot preparation;
- temporary reservation queue length at each decision;
- same-slot assignments that caused a queue to reach its hard limit.

No tracker value enters state, reward, action selection, GraphSnapshot, or DAG
execution.

## 11. Load and Drain Schedule

Each diagnostic episode contains exactly 200 physical slots:

```text
slots 1-150: offered-arrival tape enabled
slots 151-200: no new offers; drain admitted work only
```

These one-based documentation labels map to `Env slot_index` 0-149 and 150-199
respectively, as defined in Section 5.2.

The drain phase reduces right-censoring but is not assumed to clear every
cell. Required end-of-episode backlog metrics are:

```text
episode_end_admitted_incomplete_count
episode_end_active_dag_count
episode_end_ready_task_count
episode_end_queued_task_count
episode_end_executor_queue_count_by_uav
```

## 12. Required Metrics

### 12.1 Offered and admitted workload

```text
arrival_opportunity_count
offered_dag_count
offered_subtask_count
active_cap_blocked_offered_count
admitted_dag_count
admitted_subtask_count
```

### 12.2 Queue and decision availability

```text
max_executor_queue_length_by_uav
max_temporary_queue_length_by_uav
executor_queue_at_16_observation_count_by_uav
temporary_queue_at_16_observation_count_by_uav
queue_full_mask_count_by_uav
all_uavs_full_decision_count
unique_ready_task_count
ready_decision_attempt_count
repeated_ready_attempt_count
choice_decision_count
forced_decision_count
skip_decision_count
choice_decision_fraction
candidate_mask_reason_count
skip_reason_count
```

`repeated_ready_attempt_count` counts attempts after the first attempt for the
same stable task ID in an episode. It is not interpreted as a unique subtask.

### 12.3 Service outcomes

```text
completed_dag_count
completed_dag_per_slot
dag_completion_rate_admitted
dag_completion_rate_offered
average_dag_flowtime
completed_dag_flowtime_sum
completed_dag_flowtime_count
episode_reward_total
avg_uav_queue_length
end-of-episode backlog metrics
```

The completion rates are:

\[
completion_{admitted}=\frac{completed\ DAG}{admitted\ DAG},
\]

\[
completion_{offered}=\frac{completed\ DAG}{offered\ DAG}.
\]

Throughput per physical slot is fixed as:

\[
completed\_dag\_per\_slot
=\frac{completed\_dag\_count}{200}.
\]

It always uses all 200 physical simulation slots, including the 50-slot drain
phase; it is never divided by only the 150 load slots.

Flowtime uses explicit missing-data semantics:

```text
completed_dag_count == 0
-> average_dag_flowtime = null
-> completed_dag_flowtime_sum = 0.0
-> completed_dag_flowtime_count = 0
```

It must never report zero average flowtime merely because no DAG completed.
Formal aggregation reports both:

```text
pooled_completed_dag_flowtime
= sum(completed_dag_flowtime_sum)
  / sum(completed_dag_flowtime_count)

episode_mean_average_dag_flowtime
= mean(non-null episode average_dag_flowtime values)
```

The pooled value is null when the pooled completed count is zero. The episode
mean excludes null episodes and reports its contributing episode count.

Zero-denominator rows are explicitly marked null and excluded only from that
rate's aggregate; counts remain reported.

## 13. Experimental Matrix

### 13.1 Pilot

```text
4 capacity cells
x 3 fixed policies
x 3 scenario seeds (42, 86, 1042)
x 5 episodes
```

The pilot checks technical validity and resource use. It is not a scientific
result and must not be used to tune policies or select a favored cell.
It reads episodes 0-4 from the full 20-episode formal tape generated before the
pilot.

### 13.2 Formal diagnostic

After every pilot gate passes:

```text
4 capacity cells
x 3 fixed policies
x 3 scenario seeds (42, 86, 1042)
x 20 episodes
```

All cells reuse the frozen pilot-approved semantics. No parameter is changed
after inspecting pilot outcome metrics except stopping for a technical failure.

## 14. Paired Statistical Analysis

For each metric, first align A/B/C/D by the exact key:

```text
(policy, scenario_seed, episode)
```

Unmatched or checksum-mismatched rows invalidate the paired comparison.

For each matched episode, calculate:

\[
Effect_{active}=\frac{(B-A)+(D-C)}{2},
\]

\[
Effect_{queue}=\frac{(C-A)+(D-B)}{2},
\]

\[
Interaction=D-C-B+A.
\]

Null handling is metric-specific and strictly paired. For a given metric and
matched `(policy, scenario_seed, episode)` key, if any one of A/B/C/D is null,
then all three quantities for that episode and metric are null:

```text
Effect_active = null
Effect_queue = null
Interaction = null
```

The analysis must not drop only the null cell and apply a reduced or unpaired
formula. Every metric reports separately:

```text
valid_paired_episode_count
null_paired_episode_count
```

These counts are reported per policy and seed before aggregation.

The analysis then:

1. averages episode effects within each `policy x scenario_seed`;
2. reports all three seed-level effects;
3. reports the three-seed mean and standard deviation;
4. optionally reports a seed-stratified hierarchical bootstrap interval,
   clearly labeled exploratory.

The 60 episodes are not treated as 60 fully independent replicates. With only
three scenario seeds, no confidence interval is interpreted as strong
confirmatory evidence. Primary conclusions rely on direction, magnitude,
cross-seed consistency, and the predeclared metrics.

The Stage 1 actor is analyzed separately and is not pooled with the two primary
fixed policies.

## 15. Interpretation Rules

### 15.1 Actor learnability

The primary metrics are:

```text
choice_decision_fraction
queue_full_mask_count
all_uavs_full_decision_count
repeated_ready_attempt_count
```

The constraint with the larger paired effect on these measures is considered
the larger obstruction to obtaining learnable offloading actions. A reduction
in skips without an increase in choice decisions is not automatically treated
as improved learnability; it may merely convert skips into forced actions.

### 15.2 System capacity

The primary metrics are:

```text
completed_dag_per_slot
dag_completion_rate_offered
average_dag_flowtime
episode_end_admitted_incomplete_count
```

`dag_completion_rate_admitted` and reward are secondary because the active cap
changes which offered work enters their denominator or trajectory.

### 15.3 Interaction

A large interaction means one cap masks or amplifies the other. In that case,
the report must not rank the two constraints using baseline A-to-B and A-to-C
differences alone.

The final report gives separate answers for learnability and service capacity.
It does not force metrics with different units into one composite score.

## 16. Required Technical Gates

Before the pilot:

1. identical scenario and offered-template checksums across A/B/C/D and all
   policies;
2. stable SHA-256 random-hash choices across processes and machines;
3. stable DAG/task IDs independent of admission history;
4. stable DAG/task IDs are globally unique across the complete tape, include
   the task-local index, and survive template round-trip exactly;
5. complete DAGTemplate round-trip and instantiation equivalence;
6. offer generation occurs before active-cap admission;
7. episode-local nonbinding-cap formulas match hand calculations;
8. queue legality uses only the hard cap while normalization references remain
   frozen;
9. runtime `QUEUE_WORKLOAD_NORM_REF` equals exactly `80_000_000.0` operations;
10. reason-level mask and skip counters match constructed cases;
11. `all_uavs_full_decision_count` rejects non-queue consistency failures;
12. executor and temporary-reservation queue counters are distinct;
13. zero-completion episodes emit null average flowtime plus zero sum/count;
14. metric-level null pairing invalidates all A/B/C/D effects for that matched
    episode and reports valid/null paired counts;
15. UE and UAV positions remain frozen throughout an episode;
16. the drain interval creates no new offered events;
17. primary policies do not consume sequential RNG;
18. current active-DAG concurrency smoke passes, and a read-only consumer audit
    confirms `UE.active_dag_id` is not authoritative for admission, completion,
    reward, or task lookup;
19. no tracker value changes observations, rewards, actions, or execution;
20. machine-readable rows use zero-based `slot_index`, while any one-based
    `physical_slot` equals `slot_index + 1`;
21. the pilot prefix checksum equals the checksum of formal-tape episodes 0-4;
22. actor mask, environment filtering, and executor commit read the same
    episode-local capacity context;
23. decision-time and commit-time legality agree for every assignment;
24. every Stage 1 actor row matches the frozen scenario-seed checkpoint map and
    records its checkpoint SHA-256, training seed, and completed update.

Pilot technical pass additionally requires:

```text
no NaN or Inf
no illegal executed action
no traceback or OOM
B/D active blocked offered count = 0
C/D queue-full mask count = 0
invalid_assignment_reasons["queue_cap_mismatch"] = 0
all paired checksum assertions pass
all expected rows exist exactly once
```

Any failure stops the phase for review. The implementation must not silently
change caps, episode length, tape, policy, reward, or feature semantics to make
the pilot pass.

## 17. Implementation Boundaries

Implementation should prefer isolated diagnostic components rather than
changing the production training path:

- an immutable scenario/DAG-template tape helper;
- a diagnostic arrival adapter that instantiates current DAG objects from
  templates;
- a diagnostic capacity configuration that separates hard legality from fixed
  normalization;
- reason-level candidate legality instrumentation;
- a fixed-policy factorial runner and a separate paired-analysis helper;
- focused smokes for tape, caps, reasons, hashing, pairing, and non-interference.

The implementation must reuse current clean sequential assignment and executor
semantics. It must not fork or duplicate the old legacy MAPPO path.

Experiment outputs remain under protected `logs/` or `runs/` and are never
committed. Existing outputs are never deleted.

## 18. Frozen Execution Order

```text
generate and validate the complete 20-episode immutable scenario/template tape
-> calculate episode-local nonbinding caps
-> add stable SHA-256 random_hash and reason-level tracker
-> run isolated smokes
-> run 2x2 pilot on formal-tape episodes 0-4
-> verify paired checksums and nonbinding assertions
-> run formal fixed-policy diagnostic on the same episodes 0-19
-> compute paired seed-level main effects and interactions
-> judge actor learnability and system capacity separately
-> review and decide the final environment semantics
```

No PPO modification, MAPPO retraining, or Stage 2 implementation begins during
this phase.
