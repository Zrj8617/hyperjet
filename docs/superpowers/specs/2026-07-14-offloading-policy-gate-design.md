# Offloading policy gate and diagnosis design

## 1. Scope and invariants

This change adds an evaluation-only, closed-loop offloading policy gate. It does
not change training, PPO, rewards, entropy coefficients, movement selection,
checkpoint weights, environment dynamics, DAG arrivals, or KaHyPar behavior.

The default `actor_argmax` path must remain behaviorally identical to the
current deterministic evaluation path. Checkpoint-derived
`completed_dag_weight`, `detach_critic_hgnn`, and `freeze_ue_mobility` remain
authoritative. A moving-UE checkpoint therefore continues with moving UEs.

All policies consume the same stable ready-task order, legality mask, candidate
estimator, and sequential temporary reservation mechanism. A shared environment
seed pairs only the initial random conditions. Different offloading decisions
change later task completion, queues, arrivals, and actor observations, so the
four runs are closed-loop paired interventions and are not identical trajectories.

## 2. Architecture options

### Option A: add policy modes to `CleanOffloadingActor.act`

This minimizes call-site changes, but puts hand-written eval policies inside the
training actor and makes accidental training-path coupling likely. Rejected.

### Option B: implement all policy selection in `eval_clean_mainline.py`

This is the smallest file-count change, but duplicates the actor's candidate
assembly and creates a large entrypoint with weak unit boundaries. Rejected.

### Option C: a dedicated eval-only sequential policy selector

Add a small module under `scripts/` that accepts the loaded offloading actor and
the existing environment objects. It reuses
`build_offloading_candidate_components`, `TemporaryReservationState`, and the
actor scorer, but owns policy selection and diagnostics. Training continues to
call `CleanOffloadingActor.act` unchanged. Selected.

The selector returns a `CleanAssignmentBuffer`, decision trace records, and
episode aggregation inputs. `eval_clean_mainline.py` only chooses the policy,
passes stable episode/slot context, writes outputs, and aggregates metrics.

## 3. Unified policy contract

The CLI adds:

```text
--offloading-policy {actor_argmax,greedy_eft_teacher,shortest_queue,random_hash}
```

The default is `actor_argmax`. For every frozen ready task, the selector:

1. resolves the task embedding from the graph snapshot;
2. builds all candidates with the current temporary reservation;
3. skips safely if there is no legal candidate;
4. computes actor logits and probabilities for diagnostics regardless of the
   selected gate policy;
5. computes all three deterministic reference choices;
6. selects the configured policy choice;
7. appends the assignment and updates the same reservation state;
8. emits one decision record.

Policy semantics are:

- `actor_argmax`: masked actor-logit argmax. Ties follow the existing candidate
  ordering, which is sorted UAV ID order.
- `greedy_eft_teacher`: minimum legal `estimated_finish_time`, then UAV ID.
  It is an estimator-backed teacher, not a true oracle.
- `shortest_queue`: minimum current reserved queue length, then estimated finish
  time, then UAV ID. The queue length is captured before reserving this task.
- `random_hash`: SHA-256 over a canonical UTF-8 string containing evaluation
  seed, episode, slot, task ID, and the ordered legal UAV IDs. An integer derived
  from the digest indexes the legal candidates. It consumes no Python, NumPy,
  Torch, environment, or CUDA RNG state.

`selected_estimated_regret` is the selected candidate's estimated finish time
minus the minimum legal estimated finish time. It is non-negative apart from a
small numerical tolerance.

## 4. Decision and realization data model

The run directory retains one `eval_metrics.jsonl` row per episode for backward
compatibility and adds `offloading_decisions.jsonl`, one row per accepted
offloading decision. Each decision receives a stable key consisting of episode,
slot, decision order, and task ID.

At decision time the trace records:

- checkpoint path and checkpoint/model seed when recoverable from provenance;
- environment seed, episode, slot, task/DAG ID, and policy;
- ordered candidate UAV IDs and legality mask;
- pre-decision queue lengths, available times, and queued workloads;
- actor logits, masked probabilities, normalized entropy, and top-1/top-2
  probability margin;
- candidate estimated finish times and delays;
- actor, greedy EFT, shortest-queue, and actual selections;
- selected estimated regret and finish time;
- task ready time and decision/enqueue time.

Realized fields are attached after the slot and finalized at episode end by
looking up the selected task in the executor and task manager. They include
enqueue/assignment time, start time when available, compute finish, final finish,
upload, inter-UAV transfer, queue/resource wait, compute, return, and selected
estimated-versus-realized finish error.

Only the selected action has a realized outcome. Calibration therefore uses
selected candidates only and is labeled accordingly. Unfinished selected tasks
retain `null` realized finish/error plus a terminal status; they are not silently
dropped or treated as zero error. No counterfactual realization is claimed for
unselected candidates.

If executor records do not expose a direct start time, it is derived only when
the identity is exact from existing timestamps; otherwise it remains `null`.
Queue/resource wait follows the executor's realized scheduling timestamps, not
the estimator feature value.

## 5. Episode and aggregate outputs

Every episode row adds policy and provenance plus compact diagnostic summaries:

- arrival generated/completed/completion and arrival DAG/task backlog;
- drain-final generated/completed/completion, throughput, active DAG/task and
  final backlog;
- DAG flowtime mean, median, and P90 from actually completed DAGs;
- offloading action count and valid-candidate-count distribution;
- actor entropy, top-1/top-2 margin, actor/teacher agreement, and selected
  estimated regret;
- selected-candidate estimator calibration count, MAE, signed bias, and P90
  absolute error;
- realized inter-UAV transfer time and queue/resource wait;
- hover ratio, mean UAV displacement, and movement distribution;
- KaHyPar success/degraded counters;
- checkpoint path, policy, environment seed, git commit, and effective config.

`eval_summary.json` aggregates counts by summing and rates from summed
denominators. Distributional metrics aggregate underlying samples rather than
averaging episode percentiles. Existing keys and meanings remain intact.

The config records the exact policy semantics, hash version, closed-loop pairing
caveat, checkpoint controls, git commit, and output schema version.

## 6. Gate runner and provenance

A small local orchestration script may invoke the existing eval entrypoint once
per `(checkpoint, environment seed, policy)`. It does not load checkpoints or
alter evaluation semantics itself. Each invocation runs one episode and writes
to a unique directory beneath a new gate root. A separate read-only aggregation
mode validates and combines the 60 completed summaries.

The fixed gate matrix is three ep1000 checkpoints by environment seeds
`4242..4246` by four policies, for 60 episodes. Each episode uses 200 arrival
slots and one common maximum drain limit. Movement remains deterministic from
the same checkpoint and UEs remain moving because checkpoint provenance says
`freeze_ue_mobility=False`.

Historical completion and flowtime numbers are sanity references only. They are
not pass thresholds because their version and protocol differ.

## 7. Gate interpretation and veto rules

The primary paired comparisons use direction consistency and effect sizes across
checkpoint/environment-seed cells, not five-seed significance claims.

1. If greedy reduces estimated regret, selected-candidate calibration is
   acceptable, and system completion/throughput or flowtime/backlog improves,
   the estimator is useful and training credit/representation becomes the next
   target.
2. If greedy reduces estimated regret but system outcomes do not improve, EFT is
   too local for the DAG objective. EFT distillation is vetoed; downstream
   dependency, transfer, ordering, and congestion terms are investigated.
3. If greedy does not reduce estimated regret, the estimator, mask, reservation,
   or gate implementation is faulty. PPO changes are vetoed until it is fixed.
4. If actor agrees with greedy and both are poor, offloading credit is not the
   leading bottleneck. Movement collapse, coverage, capacity, candidate
   degeneration, arrival feedback, and execution are investigated.

Ambiguous effects may justify a larger gate, but that is a new server run and
requires separate authorization. No branch permits forcing a preferred training
change merely to reach formal training.

## 8. Repair admission rules

Reward, `completed_dag_weight=16`, entropy coefficient, and detach remain frozen
during the gate. Module-specific clipping is diagnostic only.

An EFT ranking auxiliary is admissible only under branch 1 and only if both
local regret and system metrics support the teacher with acceptable selected
candidate calibration. It must remain auxiliary to PPO, start with a small
explicit coefficient, preserve coefficient-zero behavior, and log its loss,
accuracy, margin, regret, gradient norms, and gradient cosine relative to PPO.
If it dominates PPO or merely copies the teacher, the coefficient is reduced and
the result is described as heuristic distillation rather than RL innovation.

Detach remains false for the first repair because the existing shared arm won
and near-zero gradient cosine does not prove harmful interference. It becomes a
single-variable interaction test only after a new credit signal succeeds and
stable negative conflict is demonstrated.

Any delayed action-return design requires an explicit cross-rollout, on-policy
construction. Stale PPO ratios are forbidden.

## 9. Test plan

Behavior-local smoke coverage will verify:

- default actor argmax matches the legacy deterministic selection;
- all policies honor legality, stable ready-task ordering, and capacity;
- EFT minimum and both tie-break rules;
- sequential reservation changes later decisions;
- random hash reproducibility and unchanged Python/NumPy/Torch RNG states;
- movement selection and arrival/drain behavior remain unchanged;
- checkpoint control inheritance, including moving UE behavior;
- decision realization/calibration null handling and numerical aggregation;
- config, episode JSONL, decision JSONL, and summary provenance;
- backward-compatible actor-default schema and KaHyPar lifecycle.

Tests use synthetic candidates and local smoke environments, never production
checkpoints or server state. Relevant existing clean eval, actor, training-loop,
checkpoint, static-UE, graph-builder, and KaHyPar smokes will also run.

## 10. Implementation sequence

1. Add the eval-only selector and focused unit/smoke probes.
2. Add CLI/config/provenance and route only eval assignments through it.
3. Add pending-decision realization finalization and episode aggregation.
4. Add the 60-cell command planner/read-only aggregator.
5. Run focused then broader local smokes and inspect the diff.
6. Commit only the design, implementation, and tests; exclude user artifacts.
7. Push the target branch.
8. Request separate authorization for server sync and the 60-episode gate.
9. Interpret the gate using the four branches, implement only the supported
   repair, then test and request authorization for short server validation.
10. Admit one fixed configuration to formal training only after the gate and
    short validation pass.

## 11. Formal-training admission and completion

Formal training is not admitted until the gate-supported repair has finite
losses and gradients, improves the intended diagnostics, preserves checkpoint
resume/eval provenance, avoids new movement collapse and KaHyPar failures, and
passes a server short validation.

The final three runs use seeds 42, 86, and 1042 with 1000 episodes by 200 slots
and identical configuration apart from seed/name/path. Before launch, local,
remote, and server commits must match; GPU, disk, Python, CUDA, KaHyPar, and new
output paths must be checked. Launch requires fresh explicit authorization and a
persistent tmux session (nohup only if tmux is unavailable), with commands,
sessions, PIDs, logs, paths, start times, and commit recorded.

Starting the jobs is not completion. Completion requires all three summaries to
say `completed`, 1000 terminal episode records and global slot 200000 per run,
final and latest checkpoints, complete provenance, no NaN/Inf or unexplained
KaHyPar degradation, and no residual training process. Training-sampled
completion is not reported as final deterministic performance.

