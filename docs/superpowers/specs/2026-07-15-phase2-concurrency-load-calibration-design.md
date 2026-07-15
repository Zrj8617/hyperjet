# Phase 2 Concurrency and Load Calibration Design

Date: 2026-07-15
Branch: `zrj_3_static_ue`
Baseline commit: `3b51523cd57b31b711a175c5270b662c67ddb173`

## 1. Purpose

Phase 1 established two facts under the current scenario. First, model seeds 42
and 86 contain reproducible but weak deterministic offloading rankings, while
seed 1042 does not beat forced-hover random assignment at any measured
checkpoint. Second, deterministic movement is itself unstable and can collapse
to nearly all hover or nearly all movement. Phase 1 did not establish why these
seed differences occur.

Phase 2 tests one bounded hypothesis:

> Does increasing useful concurrent DAG/task experience, without saturating the
> physical system, make the learning problem better conditioned and reduce
> between-seed instability?

Higher arrival probability is an experimental variable, not a presumed repair.
The phase must distinguish new DAG arrivals from active-DAG concurrency and must
not select a load merely because it creates more work.

## 2. Existing evidence and constraints

The current scenario was already calibrated over 14 seeds. At base arrival
probability `0.0145`, the historical fixed-hover baselines were approximately:

- greedy completion `0.8085`;
- random completion `0.6463`;
- greedy mean queue pressure `0.5833`;
- random mean queue pressure `0.8523`.

This means the current load is already close to the historical random-policy
queue-pressure boundary. Doubling the probability is therefore a stress point,
not an automatically valid new baseline.

The Phase 1 closed-loop evaluations generated about `0.43-0.74` new DAGs per
arrival slot, depending on policy weights. This is lower than the all-UE
Bernoulli expectation because one UE cannot create another DAG while its current
DAG is active. New arrivals per slot therefore cannot be used as a proxy for
concurrent active DAGs or useful hypergraph structure.

The repository already contains `scripts/diag_clean_load.py`, which runs
fixed-hover greedy and RNG-isolated random baselines while overriding scenario
parameters in memory. It is the correct base for Phase 2A, but it does not yet
record all required concurrency/hypergraph metrics, and its sweep path does not
currently apply the requested drain protocol.

## 3. Alternatives considered

### A. Change the global arrival constant and immediately retrain

This is the smallest textual diff but confounds load selection with learning,
has no checkpoint provenance boundary, and can silently turn the task into a
permanently overloaded queue. It is rejected.

### B. Add the training CLI control first, then explore loads with PPO

This provides provenance but spends GPU time before establishing physical
feasibility. It also makes a failed run ambiguous between overload and learning
instability. It is deferred until after calibration.

### C. Staged non-learning calibration, then training A/B

Enhance the existing diagnostic, select or veto a higher load using fixed
policies, then add a strict run-level control only for the selected training A/B.
This is the selected approach.

## 4. Frozen scenario variables

Phase 2 changes only `DAG_BASE_ARRIVAL_PROB` during calibration. It keeps fixed:

- hotspot arrival multiplier `2.0`;
- number and mobility model of UEs and UAVs;
- one-active-DAG-per-UE rule;
- DAG size and topology distribution;
- input/output data-size ranges;
- task-computation range and communication model;
- queue capacity;
- reward and completed-DAG weight;
- KaHyPar configuration and all hyperedge builders;
- fixed-hover movement for Phase 2A;
- greedy and RNG-isolated random offloading definitions.

No PPO model is loaded or trained during Phase 2A.

## 5. Phase 2A diagnostic changes

Extend `scripts/diag_clean_load.py` without changing environment dynamics.
Every arrival slot must record:

- newly created DAG count;
- fraction of zero-arrival slots and multi-arrival slots;
- arrival-eligible UE count before sampling;
- UE count suppressed by an active DAG;
- active DAG count;
- frozen ready-task count;
- committed offloading-action count;
- total queue occupancy and pressure;
- DAG, k-hop, attribute, and KaHyPar partition hyperedge counts;
- KaHyPar status and degradation reason.

Aggregate mean, P50, P90, and maximum where meaningful. Report generated DAGs
per arrival slot, actions per arrival slot, active-DAG concurrency, hyperedge
nonzero-slot ratios, completion/backlog at the end of arrival, and drain
completion/flowtime separately.

The sweep path must honor `--drain-slots`. Arrival probability is set to zero
only during drain and restored in a `finally` block. The diagnostic must preserve
the existing invariant that random policy selection does not consume the
environment RNG stream.

Machine-readable JSONL, a frozen manifest, and an aggregate CSV/Markdown report
are required. Repository source must not contain result-specific paths.

## 6. Calibration matrix

### 6.1 Smoke gate

Before the coarse matrix, run one seed and both policies for 30 arrival slots
and up to 100 drain slots at the baseline and highest stress probabilities.
Verify finite metrics, correct arrival provenance, functioning hyperedge
counters, no RNG-pairing regression, and no residual process.

### 6.2 Coarse 200-slot boundary scan

Use probabilities:

- `0.0145` -- current baseline;
- `0.0290` -- two times the baseline;
- `0.0435` -- three times the baseline;
- `0.0580` -- four times the baseline;
- `0.0870` -- six times the baseline stress point.

Use environment seeds `4242, 4243`, policies `greedy` and `random`, 200 arrival
slots, and up to 500 drain slots:

`5 probabilities x 2 seeds x 2 policies = 20 cells`.

This scan locates the transition from useful additional concurrency to obvious
saturation. It is not used for the final load claim.

### 6.3 Formal 1000-slot confirmation

The formal matrix contains exactly three probabilities:

1. the current baseline `0.0145`;
2. the highest coarse-scan probability that satisfies the coarse safety gate;
3. the next lower non-baseline probability from the fixed coarse list.

If the highest safe candidate is `0.0290`, use `0.0145`, `0.0290`, and `0.0435`
so the first unsafe neighbor is formally confirmed. If no probability above
baseline satisfies the coarse gate, use only `0.0145` and `0.0290` to confirm
the first overload boundary; no higher load may be selected. If every tested
probability is provisionally safe, use `0.0145`, `0.0580`, and `0.0870`.

Use environment seeds `4242, 4243, 4244, 4245, 4246`, policies `greedy` and
`random`, 1000 arrival slots, and up to 1000 drain slots. The normal case is:

`3 probabilities x 5 seeds x 2 policies = 30 cells`.

Coarse and formal cells run sequentially in persistent server sessions and write to new
timestamped roots. The manifest is fixed before the first cell. A failed cell is
preserved and stops the matrix; it is never silently skipped or substituted.

## 7. Load-selection gate

The coarse scan marks a probability provisionally safe only when both policies
and both seeds finish drain within 500 slots, all metrics are finite, P90 queue
pressure is below `0.95`, and skipped-ready-task counts do not indicate
persistent capacity rejection. This relaxed coarse gate only selects the formal
matrix; it never selects the final load.

A higher probability is viable only if all integrity checks pass and, for both
policies:

1. all five scenes finish drain within the 1000-slot limit;
2. no unexplained invalid assignment or KaHyPar degradation occurs;
3. P90 queue pressure stays below `0.90` and the queue is not pinned at capacity;
4. skipped-ready-task counts do not indicate persistent capacity rejection;
5. arrival-end backlog and drain flowtime remain finite and do not show runaway
   growth across the 1000-slot window;
6. greedy retains a meaningful paired advantage over random;
7. active-DAG concurrency, ready tasks, offloading actions, or nontrivial
   hyperedges increase materially relative to `0.0145`.

The selected load is the highest candidate satisfying the safety conditions,
not the candidate with the largest arrival count. If none passes, retain
`0.0145`; Phase 2A is still a successful diagnostic and must not force a new
load. A fixed target such as two or three new DAGs per slot is descriptive only
and cannot override the saturation veto.

## 8. Phase 2B run-level control

Only after a higher load passes Phase 2A, add run-level controls to the clean
training/evaluation path:

- `--dag-base-arrival-prob`;
- checkpoint/config/run-summary provenance;
- strict finite `[0,1]` validation;
- resume rejection on mismatch;
- deterministic evaluation inherits the checkpoint value and rejects an
  explicit mismatch;
- legacy checkpoints resolve to the current `0.0145` default;
- drain temporarily disables arrivals without losing the run-level value.

The hotspot multiplier remains fixed in Phase 2. It must not be tuned in the
same experiment. Default invocation must reproduce the current behavior and
checkpoint topology apart from additive provenance fields.

## 9. Phase 2C short learning A/B

Compare the current load and the selected higher load using model seeds:

`42, 86, 1042, 2042, 3042`.

Every run uses 200 episodes x 200 slots, checkpoint interval 20, the current
shared-HGNN baseline, completed-DAG weight 16, learned movement, and no v1/v2
counterfactual module. Evaluate checkpoints 20/40/60/80/100/150/200 on the same
five common environment seeds using both normal deterministic movement and
forced-hover offloading isolation.

The higher load advances only if it:

- remains physically viable under training;
- does not materially reduce deterministic completion or worsen drain flowtime;
- reduces between-seed dispersion in offloading quality or movement collapse;
- increases useful offloading/hypergraph decision counts;
- improves more than one seed without producing a large regression in another.

If higher load increases data volume but not stability, it is not a learning
repair. The next experiment then becomes a separately designed optimizer/shared-
representation or hypergraph ablation. Full-hypergraph versus disabled-edge A/B
is deliberately excluded from this phase so load and representation are not
changed simultaneously.

## 10. Interpretation boundaries

Phase 2 can establish whether higher feasible concurrency changes learning
stability. It cannot by itself prove that the hypergraph causes seed variance,
that more episodes guarantee convergence, or that task randomness is the sole
cause. A later hypergraph ablation is required for causal attribution to HGNN or
KaHyPar edges.

No Phase 2 result directly authorizes the three formal 1000-episode runs. Those
remain gated by the 200-episode deterministic A/B. The formal 1000-slot load
matrix in Phase 2A is a non-learning physical calibration, not a training claim.

## 11. Safety and artifacts

All server actions remain under `/data2/zrj2025`. New result/log roots are used;
historical `runs/` and `logs/` are never deleted, moved, or overwritten. Local
`docs/session_handoff_phase4.md` and `runs/` remain untracked and must not be
staged or committed.

The design, diagnostic implementation, run-level implementation, and experiment
reports use separate commits. Server execution begins only after the relevant
commit is pushed and the server fast-forwards to the exact approved HEAD.
