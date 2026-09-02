# Boundary-Anchored Decision Credit: Phase 1 Local Credit Audit

**Date:** 2026-09-02

**Status:** Approved design awaiting implementation review

**Scope:** Diagnostic-only feasibility audit; not a training target and not a final paper method

## 1. Objective

Phase 1 answers one narrow question:

> Does a decision-level, paired counterfactual intervention produce a stable and non-degenerate local slot-reward difference in the current environment?

For decision `i`, the diagnostic quantity is:

```text
C_i^local = r_t^real - r_{t,i}^cf
```

Both rewards are obtained by replaying from the same exact pre-decision snapshot and executing to the next slot boundary. `C_i^local` is diagnostic output only. It must never be written into the rollout buffer, used as a Decision-Q target, or consumed by an actor/critic loss.

This phase does not claim that `C_i^local` is the final decision credit. Formal boundary-anchored credit remains reserved for a later phase:

```text
U^real = r_t^real + gamma * V_B(B_{t+1}^real)
U_i^cf = r_{t,i}^cf + gamma * V_B(B_{t+1,i}^cf)
C_i = U^real - U_i^cf
```

No boundary critic is implemented or trained in Phase 1.

## 2. Frozen Experimental Source

The audit reuses the existing Decision-Q v2 ranking infrastructure and its 27 source decisions:

- seeds `0`, `1`, and `2`;
- checkpoints at updates `30`, `60`, and `120`;
- up to three existing source decisions per checkpoint;
- source states from checkpoint replay started according to the project's existing resume semantics from a new episode;
- states described only as **frozen-checkpoint replay-generated on-policy decision states**.

The audit must not claim to recover historical rollout snapshots or the historical training distribution.

Each decision snapshot is the exact environment and policy-side state after movement and after any earlier within-slot decisions/reservations, but before applying the current offloading decision. The actor, encoder, main critic, and Decision-Q critic remain frozen throughout collection.

The existing source selection is reused rather than recollected where its saved snapshot can be reconstructed exactly. A source decision must have at least two legal UAV actions so that a counterfactual alternative exists.

## 3. Reused Infrastructure

Implementation should extend the current diagnostic path rather than create a separate simulator:

- checkpoint replay and post-movement/pre-offloading exact-state reconstruction from the Decision-Q ranking audit;
- existing Scheme-B2 exact-state branch restoration;
- existing semantic common-random-number handling;
- existing counterfactual oracle and common-random helpers where applicable;
- existing serial/process consistency checks.

Only diagnostic scripts and diagnostic outputs may change. Training modules, training targets, checkpoint format, rollout behavior, and environment semantics remain untouched.

## 4. Intervention Definition

For each source decision:

1. Let `a_i` be the frozen policy's selected legal UAV.
2. Sample exactly one alternative:

   ```text
   a_i' ~ Uniform(A_legal \ {a_i})
   ```

3. Use an isolated, deterministic collector RNG for this draw. The draw must not consume or alter environment, actor, framework-global, or continuation RNG state.
4. Sample `a_i'` once per decision and hold it fixed across every CRN root. Root-to-root variation therefore estimates continuation noise for one paired intervention, not variation in the alternative proposal.
5. Record the legal candidate IDs, selected action, alternative action, uniform proposal probability, and the collector draw identifier needed for exact reproduction.

The alternative must not be fixed to top-2, EFT-best, Q-best, or any other policy-dependent comparator.

## 5. Paired Branch Semantics

For each decision and independent CRN root, create two branches from the same exact pre-decision snapshot.

### Actual branch

- Restore the exact snapshot.
- Force the original selected action `a_i`.
- Apply its reservation/state transition.
- Regenerate all remaining decisions in the slot using the frozen Scheme-B2 deterministic policy continuation.
- Execute through the current physical slot to the next slot boundary.
- Save the existing environment slot reward `r_t^real` and its already-available components.

### Counterfactual branch

- Restore the same exact snapshot.
- Force the sampled alternative `a_i'`.
- Apply its reservation/state transition.
- Regenerate all remaining decisions in the slot using the same frozen Scheme-B2 deterministic policy continuation, now conditioned on the counterfactual reservation state.
- Execute through the same current physical slot to the next slot boundary.
- Save the existing environment slot reward `r_{t,i}^cf` and its already-available components.

The actual reward must also be branch-replayed. An ordinary observed replay reward must not be paired with a separately executed counterfactual reward.

Changing the current action may legitimately change later legal sets and suffix actions. That is the intended policy-conditioned intervention, not a semantic mismatch. The collector must not force the original suffix. Both branches must nevertheless start from the identical snapshot, use the same frozen policy rule, execute the same physical-slot interval, and reach the corresponding next slot boundary under normal environment semantics.

The current decision and remaining same-slot decisions consume no extra discount step. Phase 1 records the undiscounted paired difference in the existing slot reward; `gamma` is not applied within the slot.

## 6. Semantic CRN and Root Budget

Within one root, actual and counterfactual branches share aligned semantic randomness for corresponding environment events. Different roots are independent and reproducible.

The Phase 1 formal audit uses eight roots per decision. The sampled alternative remains fixed across those roots. There is no automatic expansion to 16 or 32 roots in this phase; the purpose is a minimum-cost feasibility decision.

Before the formal audit, run a minimal smoke on seed `0`, update `30`, one decision, and two roots. The smoke compares serial and process execution and verifies the gates in Section 10.

With 27 decisions and eight roots, the upper-bound formal workload is 432 branch-to-boundary continuations: two branches per paired root.

Each root records an explicit `root_id` plus the reproducible RNG seed/state identifiers already supported by the semantic CRN infrastructure. No new digest or hashing mechanism is introduced.

## 7. Diagnostic Records

### Decision-level JSONL

Each decision record contains:

- source identity: seed, checkpoint update, replay phase/episode, slot, decision order, task/DAG identity;
- state timing label and exact snapshot identity used by the existing replay infrastructure;
- legal UAV IDs, selected UAV ID, alternative UAV ID, and alternative proposal probability;
- frozen Q legal vector, Q ranking/spread, and actor probabilities;
- available EFT quantities for the selected and alternative actions, including their paired difference or regret convention;
- per-root actual reward, counterfactual reward, reward components, and `C_i^local`;
- per-root completion/censor state and reason;
- per-root semantic CRN alignment and recognized-RNG status;
- aggregate `C_i^local` mean, standard deviation, standard error, 95% confidence interval, empirical positive fraction, sign consistency, and observed zero/near-zero frequency.

Near-zero summaries must be reported relative to the observed slot-reward scale, with the raw values retained. They are descriptive diagnostics rather than a hard algorithmic threshold.

### Summary JSON

Summaries are produced by seed, checkpoint phase (`early=30`, `mid=60`, `late=120`), and pooled:

- usable and censored decision/root counts;
- mean, standard deviation, median, interquartile range, variance, and mean absolute `C_i^local`;
- within-decision root variability versus between-decision variability;
- confidence-interval-resolved fraction and sign consistency;
- association between paired credit and EFT difference;
- association between paired credit and `Q(s_i,a_i) - Q(s_i,a_i')`;
- association between paired credit and selected-versus-alternative actor probability or log-probability difference;
- descriptive grouping by selected/alternative UAV ID where sample support exists.

These associations test whether the counterfactual signal adds information beyond existing EFT, Q, and policy preference. They are not causal estimates and do not turn `C_i^local` into a value target.

## 8. Interpretation Rule

Phase 1 can support proceeding to the boundary-critic phases when paired reward differences are reproducibly non-degenerate across multiple seeds/checkpoint phases and their between-decision variation is not overwhelmed by within-decision CRN uncertainty. Evidence that the signal differs from existing EFT/Q/actor preference strengthens that case but is not mandatory.

The result is mixed when effects exist but are dominated by continuation uncertainty, censoring, or checkpoint-specific behavior.

The counterfactual direction is not supported when most paired differences remain observably zero or unresolved at the eight-root budget and within-decision uncertainty dominates the available between-decision signal.

These are evidence-based judgments, not fixed pass/fail constants. Phase 1 only decides whether the signal is stable enough to justify Phase 2/3 work; it cannot prove that the eventual boundary-anchored credit is correct.

## 9. Output Location

The proposed server output root is:

```text
/data2/zrj2025/uav-results/audits/boundary-anchored-decision-credit/phase1_local_credit_27
```

Expected artifacts are:

```text
smoke/decision_records.jsonl
smoke/summary.json
formal/decision_records.jsonl
formal/summary.json
formal/run_manifest.json
```

The manifest records checkpoint/source paths, root identifiers, alternative draws, execution mode, and gate results using plain structured fields.

## 10. Engineering Gates

The smoke and formal collector must demonstrate:

- actor, encoder, main critic, and Decision-Q critic parameters are tensor-for-tensor unchanged;
- optimizer step count is zero;
- no rollout target, Decision-Q target, advantage, or training buffer is written;
- actual and counterfactual branches begin from the same exact pre-decision snapshot;
- each paired branch uses the same semantic CRN root;
- different root IDs are independent and reproducible;
- semantic mismatch count is zero;
- unrecognized RNG count is zero;
- serial and process results are exactly consistent for the smoke;
- both branches preserve the existing same-slot timing convention and reach the next slot boundary without an intra-slot gamma step;
- the alternative is uniform over legal actions excluding the selected action and remains fixed across roots;
- the collector does not modify training code, checkpoint state, or policy parameters.

If a branch cannot reach the corresponding next slot boundary under the normal frozen-policy continuation, record it as censored with a concrete reason and exclude that paired root from credit aggregation. Do not substitute a fixed suffix to rescue it.

## 11. Explicit Non-Goals

Phase 1 does not:

- train a Decision critic or actor;
- replace any existing Decision-Q target;
- implement or train a boundary critic;
- alter the 183-dimensional Decision-Q input;
- change PPO, reward, movement, environment, action space, optimizer, entropy, `gamma`, or `lambda`;
- enumerate all legal counterfactual actions;
- compute Shapley values;
- feed branch outcomes into training;
- claim final credit-assignment validity or final-paper readiness.

Implementation stops after producing and interpreting the smoke and formal diagnostic outputs. Phase 2/3 requires a separate reviewed design and explicit authorization.
