# Phase 2B Ready-to-Offload Funnel and Conditional DAG Concurrency Design

Date: 2026-07-15
Branch: `zrj_3_static_ue`
Baseline commit: `e66f1bb6a63fba2828fe4f220224cb6e7b0764cf`

## 1. Decision and scope

Phase 2B maximizes the density of **successfully accepted offloading decisions**:

> executor-accepted ready tasks per arrival slot

New DAGs per slot are an input and characterization metric, not the objective.
Two or three new DAGs per slot is not a pass condition. A candidate load is
useful only if it increases accepted decisions while the system still drains,
capacity blocking remains controlled, and UAV queues do not stay permanently
saturated.

This design supersedes the automatic formal-load continuation in Sections 6.3
and 7 of the earlier Phase 2 design. It does not authorize PPO training, a
service-capacity change, or a full concurrency grid before the funnel gate.

## 2. Current evidence

The completed Phase 2A coarse scan shows 7--11 ready tasks in many slots but
only about 2.5--4.9 assignments per slot. Raising arrival probability primarily
increased active-DAG and ready-task backlog; it did not proportionally increase
new DAGs or accepted assignments.

Read-only code inspection found no explicit per-slot assignment count limit:

- the environment freezes all ready tasks rather than truncating the set;
- the actor and diagnostic traverse the full frozen ready set;
- same-slot reservations consume queue positions sequentially;
- each UAV queue has a capacity of 16;
- completed tasks release queue positions only after execution advances.

The working hypothesis is therefore that accepted assignments are limited by
queue headroom and service throughput. This remains a hypothesis until the
runtime funnel confirms it. DAG dependency release can also delay successors by
one slot, but it does not explain a ready-to-assignment gap when many tasks are
already ready.

The current implementation's legality helper does not enforce geometric
reachability, but the diagnostic must not rely on that implementation detail.
All capacity ceilings are conditioned on the legality rules actually active in
the evaluated revision so future reachability or resource checks cannot create
a false implementation-defect signal.

## 3. Alternatives considered

### A. Funnel first, then conditional concurrency calibration

Instrument the decision pipeline, identify the limiting stage, and run only the
concurrency cells justified by that result. This is selected because it
preserves causal interpretation and avoids spending cells on known saturation.

### B. Instrument and immediately run the full concurrency grid

This reduces calendar latency only if the system is not already saturated. The
existing evidence makes that unlikely, so it risks measuring the same backlog
failure repeatedly. It is rejected.

### C. Increase queue capacity, UAV compute, or shorten tasks first

These changes may raise accepted assignments, but they alter the service system
and can move backlog from the ready set into UAV queues without increasing
completed work. They require a separate service-capacity design and are not part
of Phase 2B.

## 4. Stage 0: repair summary semantics

Stage 0 changes diagnostics and reports only. It must not change environment,
policy, arrival, queue, execution, or reward behavior.

Report these completion measures separately:

- `arrival_completion_rate`: DAGs completed by the end of arrival slots divided
  by DAGs generated during arrival;
- `final_completion_rate`: DAGs completed after drain divided by total generated
  DAGs;
- `drain_slots_used` and `drain_completed`.

Replace the ambiguous `offloading_skipped_rate` report label with
`capacity_blocked_task_slot_ratio`. It counts ready-task/slot opportunities
deferred because no legal capacity remained. It is not a dropped-task rate and
must not be described as rejection of the DAG.

Also report pre- and post-decision queue occupancy, all-UAV-full slot fraction,
longest consecutive all-full run, arrival-end ready and active-DAG backlog, and
the backlog trend over the final 50 arrival slots. Existing machine-readable
field names may be retained as deprecated aliases for one revision only if a
consumer test requires compatibility; the report must use the corrected names.

## 5. Stage 1: ready-to-offload funnel

### 5.1 Slot-level funnel

For every arrival slot, record the following counts without consuming policy or
environment RNG:

1. `frozen_ready_count`;
2. `ready_in_graph_count`;
3. `ready_with_initial_legal_candidate_count`;
4. `policy_selected_count`;
5. `assignment_buffer_accepted_count`;
6. `executor_accepted_count`, taken from `newly_assigned_tasks`;
7. `executor_invalid_count`, split by reason where observable.

Capacity diagnostics must include:

- per-UAV queue occupancy and free positions before decisions;
- initially full UAV count and all-UAV-full status;
- candidate count at each sequential decision;
- tasks with no candidate in the unmodified pre-decision state;
- tasks that lose their last candidate only after same-slot reservations;
- tasks omitted by the policy or missing from the graph;
- tasks rejected while constructing a schedule record;
- tasks completed and queue positions released during the slot;
- all-UAV-full fraction and longest consecutive all-full run.

Training and evaluation paths that pass a prebuilt assignment buffer currently
do not reconstruct the no-candidate count. The funnel must measure actor-side
capacity blocking explicitly rather than infer zero blocking from the existing
environment field.

### 5.2 Legality-conditioned capacity ceiling

The utilization denominator must not be raw total queue space. It is the maximum
number of frozen ready tasks that could be accepted in the slot under the
current legality relation, per-UAV residual capacities, duplicate/scheduled-task
rules, and same-slot reservation semantics.

Compute this `legal_capacity_ceiling` with a policy-independent capacitated
bipartite matching or an equivalent deterministic max-flow calculation:

- left nodes: frozen ready tasks that are valid before policy selection;
- right nodes: UAVs with their pre-decision residual queue capacities;
- edges: task--UAV pairs accepted by the current legality checks;
- capacity: one per task and the actual residual queue positions per UAV.

This is stricter than
`min(ready_with_legal_candidate_count, legal-UAV free positions)`, which can
double-count overlapping candidate sets. It also makes the diagnostic correct
if reachability constraints are added later.

Define:

`legal_capacity_utilization = executor_accepted_count / legal_capacity_ceiling`

with a separately reported zero-ceiling case. Also retain
`executor_accepted_count / frozen_ready_count` as the learning-sample conversion
rate; it is not the implementation-efficiency gate.

### 5.3 Greedy reservation consistency

When the diagnostic greedily minimizes estimated finish time, each same-slot
reservation must propagate the selected candidate's estimated finish time and
queued workload. This matches the production actor's sequential reservation
semantics. The correction affects greedy choice quality but must not alter the
definition of the legal capacity ceiling.

### 5.4 Diagnostic matrix

Run fixed-hover, non-learning cells with:

- arrival probabilities `0.0145`, `0.0290`, `0.0435`;
- policies `greedy`, `random`;
- environment seeds `4242`, `4243`;
- 200 arrival slots and at most 500 drain slots.

This is 12 cells. Actor checkpoints are not loaded, and no GPU is required.
Random-policy choice uses an isolated RNG stream. Initial environment state and
geometry must be paired across policies for each probability and seed.

## 6. Stage 1 decision branches

### Branch 1: service/queue throughput saturation

Select this branch when legality-conditioned utilization is consistently near
one, accepted assignments track newly released queue capacity, and the
ready-to-accepted gap occurs primarily while legal queue capacity is exhausted.

Do **not** run the full Stage 2 grid. Run only the small cap 1/2/3 confirmation
in Section 7.2, then stop Phase 2B. The next research phase must separately
characterize service capacity, including task duration, UAV compute, queue
semantics, and execution timing. It must not silently change those variables in
this experiment.

### Branch 2: pipeline loss below the legal ceiling

Select this branch when meaningful legal capacity exists but tasks are lost at
graph inclusion, candidate construction, policy selection, assignment-buffer,
or executor commitment. Stop before changing DAG concurrency. Produce a
reason-counted defect report and design the smallest relevant correction. Do not
label a legality-constrained task as an implementation loss.

### Branch 3: concurrency/arrival suppression without hard saturation

Select this branch when legal capacity is regularly unused, the pipeline uses
it efficiently, and the one-active-DAG-per-UE rule materially suppresses new
work. Proceed to the adaptive Stage 2 grid.

Ambiguous evidence does not default to Branch 3. Extend Stage 1 seeds or slots
before changing concurrency.

## 7. Stage 2: conditional per-UE DAG concurrency experiment

### 7.1 Controlled factor

Introduce a run-level maximum active-DAG count per UE with values `1`, `2`, and
`3`. Value `1` must reproduce current behavior. The task manager is the
authoritative source for active-DAG membership; any UE compatibility state must
remain consistent when one of several DAGs completes.

Keep fixed:

- DAG size, topology, task attributes, and communication distributions;
- number, geometry, and mobility rules of UEs and UAVs;
- UAV compute rate and queue capacity 16;
- hotspot multiplier;
- greedy/random policy definitions and fixed-hover movement;
- arrival/drain duration and paired scene seeds;
- graph and KaHyPar configuration.

The implementation requires strict validation and provenance for the new
run-level value. No training/checkpoint compatibility work is authorized until
a load is selected for a later learning experiment.

### 7.2 Small confirmation for Branch 1

If Stage 1 confirms hard saturation, run only:

- concurrency caps `1`, `2`, `3`;
- arrival probability `0.0145`;
- policies `greedy`, `random`;
- seeds `4242`, `4243`;
- 200 arrival slots plus at most 500 drain slots.

This 12-cell confirmation tests whether higher concurrency increases active
DAGs and hypergraph structure while accepted assignments remain service-bound.
It is characterization, not a search for a higher load. Stop after the report.

### 7.3 Adaptive grid for Branch 3

If Stage 1 supports unsaturated arrival suppression, first run:

- concurrency caps `1`, `2`, `3`;
- arrival probabilities `0.0145`, `0.0290`;
- policies `greedy`, `random`;
- seeds `4242`, `4243`;
- 200 arrival slots plus at most 500 drain slots.

This is 24 cells. Add the 12 cells at `0.0435` only if at least one higher-cap
configuration at `0.0290` drains and shows no persistent queue saturation or
runaway capacity blocking.

### 7.4 Outcome measures and feasibility

The primary outcome is mean `executor_accepted_count` per arrival slot. Report
paired deltas relative to cap 1 at the same probability, seed, and policy.

A configuration is infeasible if it fails to drain within 500 slots, leaves
nonzero final backlog, exhibits non-finite metrics, or shows a sustained
all-UAV-full tail. Capacity-blocked task-slot ratio, all-full fraction and run
length, queue/backlog slopes, drain duration, and flowtime remain separate
congestion outcomes; they must not be folded into the primary action-density
number.

Among feasible configurations, select by:

1. highest executor-accepted action density;
2. shorter drain;
3. lower capacity-blocked task-slot ratio;
4. less persistent queue saturation;
5. richer concurrent-DAG and nontrivial hyperedge structure.

A higher cap may be scientifically useful even if action density does not rise:
it can prove that arrival suppression was removed while service throughput
remained limiting. That result redirects the project; it does not justify still
higher arrival probability.

## 8. Main-problem priority boundary

Load calibration is secondary characterization. It does not repair or explain
the two stronger Phase 1 findings:

1. seed 1042 has poor forced-hover offloading quality from the earliest measured
   checkpoint rather than a demonstrated late collapse from a good policy;
2. learned movement has negative value for seeds 42 and 86 under the completed
   paired evaluations and is independently unstable.

The load is shared across seeds, so it cannot by itself explain why seed 1042
diverges under the same scenario. Phase 2B must not indefinitely postpone the
early-divergence diagnostic at checkpoints 20/40/60/80 or the controlled
movement-value upper-bound experiment. Those tracks may run in parallel when
they do not contend for the same code or experimental resources. No Phase 2B
result may be presented as resolving seed stability without direct paired
evidence.

## 9. Tests and integrity checks

Before server execution, verify locally where dependencies allow and on the
server where Torch/KaHyPar are required:

- cap 1 reproduces legacy arrivals and release behavior under a fixed seed;
- cap 2 permits two active DAGs and releasing one does not release the other;
- cap 3 never permits a fourth active DAG;
- max-flow ceiling fixtures cover disjoint candidates, overlapping candidates,
  unequal UAV capacities, and zero legal capacity;
- funnel counts are monotone:
  `executor <= buffer <= selected <= frozen ready`;
- executor-accepted counts equal `newly_assigned_tasks`;
- random diagnostics do not change environment RNG trajectories;
- arrival and final completion are reported separately;
- drain sets arrivals to zero temporarily and restores the configured value;
- no KaHyPar degradation or unexplained invalid assignment occurs.

The server preflight must use the exact pushed HEAD. Each run uses a new
timestamped result and log root, a frozen manifest, persistent process handling,
and resumable per-cell progress.

## 10. Safety and artifacts

All server commands and artifacts remain under `/data2/zrj2025`. Existing
server `logs/` and `runs/` are preserved. Local untracked
`docs/session_handoff_phase4.md` and `runs/` are never staged or committed.

Design, implementation, diagnostic results, and any later service-capacity
study use separate commits. Phase 2B ends with an evidence-backed branch
decision; it is not required to force a concurrency or arrival-rate change.
