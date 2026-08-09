# Active-DAG and Queue-Cap 2x2 Factorial Diagnostic Implementation Plan

Date: 2026-08-03  
Branch: `zrj_3multisample`  
Required starting commit: `cc9c4d51b44ba7bc09a08f3812b701e8f2abc67f`  
Design authority: `docs/superpowers/specs/2026-07-31-active-dag-queue-cap-factorial-diagnostic-design.md`

## 0. Scope and stop condition

Implement only the fixed-policy environment diagnostic:

```text
full immutable 20-episode tape
-> episode-local active/queue nonbinding caps
-> one queue-capacity context across decision/filter/commit
-> reason-level candidate and skip instrumentation
-> deterministic random_hash / greedy_eft / frozen Stage-1 argmax
-> four-cell episode runner
-> strict paired analysis
-> isolated smokes
-> five-episode-prefix pilot
-> stop for review
```

Do not modify PPO/MAPPO objectives, critic, GAE, reward, HGNN,
`GraphSnapshot`, actor input dimensions, or training behavior. Do not train or
retrain a model. Do not start the formal 20-episode diagnostic after the pilot.

## 1. Add the diagnostic capacity context and neutral plumbing

Files:

- add `environment/diagnostic_capacity.py`
- minimally modify `environment/assignment.py`
- minimally modify `environment/task_execution.py`
- minimally modify `environment/env.py`

Add an immutable `DiagnosticCapacityContext` with:

- `hard_queue_cap`;
- fixed `queue_length_norm_ref = 16`;
- fixed `remaining_slots_feature_ref = 16`;
- fixed `slot_assigned_norm_ref = 16`;
- literal `queue_workload_norm_ref = 80_000_000.0`.

Validate all values and assert the workload reference exactly. Thread the same
object through candidate legality, environment-side assignment filtering, and
executor commit validation. Default `None` preserves the existing
`config.CLEAN_MAX_QUEUE_PER_UAV` behavior.

In diagnostic mode only, compute the existing seven dynamic UAV features with
the fixed references. The remaining-slots feature uses the frozen reference,
not the nonbinding hard cap. Add no feature and change no tensor shape.

Verification:

- default-path candidate masks and features remain unchanged;
- hard cap changes legality only;
- feature references do not change between caps 16 and nonbinding;
- decision, environment filtering, and commit all receive object-identical
  context;
- no accepted assignment is rejected because of a cap mismatch.

## 2. Build the canonical immutable tape

Files:

- add `environment/capacity_factorial_diagnostic.py`
- add `scripts/generate_active_dag_queue_cap_tape.py`

Implement canonical UTF-8 JSON with recursively sorted keys, compact
separators, ordered arrays, and `allow_nan=False`. Use it for every checksum,
stable DAG/task identifier, keyed scenario draw, offer draw, template draw,
and `random_hash` choice.

For seeds `42, 86, 1042` and episodes `0..19`, freeze:

- hotspot center/radius;
- 60 initial UE positions;
- 5 initial UAV positions;
- 150 x 60 offered-event bits;
- one complete immutable DAG template for every offered event.

Generate a template with an OfferKey-derived isolated NumPy seed while saving
and restoring the caller's global NumPy RNG state. Reuse the current clean DAG
generator, then remap all DAG/task/dependency/hyperedge identifiers to the
stable SHA-256 contract. Serialize every random DAG/job/task attribute needed
for exact reconstruction.

Store compact slot-ordered offer bit strings plus slot/UE-keyed templates.
Calculate and persist:

- per-episode scenario, offered-event, and offered-template checksums;
- per-episode offered DAG/subtask counts;
- episode-local active and queue nonbinding caps;
- full-tape checksum;
- exact episodes-0..4 prefix checksum.

The writer is create-only: an existing tape is accepted only when validation
and requested controls match; it is never silently overwritten.

Verification:

- canonical hashes reject NaN/Inf;
- all stable DAG/task IDs are globally unique over the complete tape;
- task IDs include zero-padded local indices;
- JSON round-trip preserves every field and identifier;
- template instantiation reconstructs current `DAGJob`/`TaskNode` objects;
- generated and instantiated templates are attribute-equivalent;
- prefix checksum equals a fresh canonical slice of episodes 0..4.

## 3. Add the diagnostic arrival adapter

File:

- `environment/capacity_factorial_diagnostic.py`

Add an isolated `FactorialDiagnosticEnv` subclass. One instance represents one
`scenario_seed x episode x capacity cell` and receives the tape episode plus
its immutable capacity context.

On reset, call the clean reset, then replace the sampled hotspot and entity
positions with tape values, freeze UE mobility, and retain hover UAV movement.
Override only clean arrival processing:

```text
read slot/UE offered bit
-> no offer counter
-> retrieve fixed template
-> active-cap admission check
-> instantiate and admit, or record active-cap blocked offer
```

Slots `0..149` are load slots. Slots `150..199` are drain slots with no
opportunity, offer, or no-event increments. The adapter must never regenerate
a template or consume arrival/DAG RNG.

## 4. Add reason-level legality and episode tracking

File:

- `environment/capacity_factorial_diagnostic.py`

Implement a tracker with deterministic primary precedence and complete reason
sets. Minimum reasons:

```text
task_not_ready
already_reserved
already_scheduled
invalid_uav
queue_full
other
```

Record candidate reason totals and per-UAV totals, skip primary reasons and
full signatures, executor versus temporary queue observations, maxima, at-16
counts, unique/repeated ready attempts, choice/forced/skip counts, and
`all_uavs_full_decision_count` under the exact three-condition rule.

For every accepted assignment record zero-based `slot_index`, derived
one-based `physical_slot`, task/UAV ID, decision-time legality, and commit-time
legality. Any mismatch increments `queue_cap_mismatch` and fails the row.

At episode end report offered/admitted workload, service outcomes, explicit
zero-completion flowtime null semantics, and all backlog metrics required by
the design.

## 5. Implement the three fixed policies and the 2x2 runner

Files:

- `environment/capacity_factorial_diagnostic.py`
- add `scripts/run_active_dag_queue_cap_factorial.py`

Use one shared sequential decision loop:

```text
prepare clean slot once
-> force hover
-> freeze ready order
-> build candidate mask/features/EFT from current reservation and capacity context
-> track all legality reasons
-> choose fixed action
-> reserve immediately
-> commit once with the same context
-> advance executor/reward once
```

Policies:

- `random_hash`: exact canonical JSON array and smallest unsigned SHA-256
  integer, with UAV ID tie-break;
- `greedy_eft`: minimum legal decision-time EFT, then UAV ID;
- `stage1_actor`: strict scenario-seed checkpoint mapping, MLP-only strict
  load, update 30, masked argmax, checkpoint SHA recorded.

The runner requires an already-existing validated full formal tape. It accepts
an episode prefix count so the pilot uses exactly episodes 0..4. It writes to a
new output directory and refuses to overwrite existing rows or summaries.

Each row contains its cell, policy, seed, episode, capacity values, all tape
checksums, policy/checkpoint identity, metrics, reason counters, legality
records, and finite/technical flags.

## 6. Implement strict paired analysis

Files:

- `environment/capacity_factorial_diagnostic.py`
- add `scripts/analyze_active_dag_queue_cap_factorial.py`

Require exactly one A/B/C/D row for every
`policy x scenario_seed x episode`. Reject duplicate, missing, or checksum-
mismatched rows.

For every declared numeric metric compute episode-level active main effect,
queue main effect, and interaction. If any of A/B/C/D is null, emit all three
as null for that metric and increment the null-paired count. Aggregate first
within policy/seed, then report all three seed values and three-seed mean/std.
Keep the Stage-1 actor policy separate from the two primary policies.

The analysis also enforces pilot gates:

- B/D active blocked offered count is zero;
- C/D queue-full mask count is zero;
- no queue-cap mismatch, invalid execution, NaN, Inf, or traceback marker;
- identical scenario/event/template/full/prefix checksums;
- expected row keys occur exactly once.

## 7. Add isolated smokes

Files:

- add `scripts/smoke_active_dag_queue_cap_tape.py`
- add `scripts/smoke_active_dag_queue_cap_semantics.py`
- add `scripts/smoke_active_dag_queue_cap_pairing.py`

Cover all 24 pre-pilot technical gates with small constructed fixtures where
possible, including:

- canonical JSON and cross-process random-hash stability;
- global ID uniqueness and template round-trip/instantiation equivalence;
- offer-before-admission and drain behavior;
- hand-calculated nonbinding caps;
- fixed seven-dimensional features and literal workload reference;
- constructed multi-reason masks and all-UAV-full exclusion;
- separate executor/temporary counters;
- null flowtime and strict null pairing;
- frozen UE/UAV positions;
- no sequential policy RNG;
- active-DAG concurrency plus read-only `UE.active_dag_id` consumer audit;
- no tracker feedback into observation, reward, choice, or execution;
- zero/one-based slot identity;
- object-identical capacity context at all three legality paths;
- strict Stage-1 checkpoint metadata helper checks.

Run local syntax compilation with the bundled Python. Run non-Torch smokes
locally. Torch/checkpoint branches must execute on the server.

## 8. Server verification and pilot

Before each server operation, state the exact action. Work only under
`/data2/zrj2025`. Preserve `logs/` and `runs/` and sync code without reset,
clean, or overwrite.

Sequence:

1. transfer the reviewed implementation without touching protected outputs;
2. run isolated smokes and relevant clean regressions with server Torch;
3. generate the complete three-seed, 20-episode tape once under a new
   protected output path;
4. validate its full checksum and 0..4 prefix checksum;
5. run the 4 x 3 x 3 x 5 pilot into a new output directory;
6. run paired analysis and all pilot gates;
7. record wall time, CPU RSS, GPU use, paths, commit, and checksums;
8. stop and report.

If any smoke or pilot gate fails, preserve the original traceback/artifacts,
stop, and report. Do not change caps, tape, schedule, policy, reward, feature
semantics, or episode count in response to pilot outcomes.

## 9. Git and artifact discipline

- Preserve every pre-existing untracked file and directory.
- Never use `git add .`, reset, clean, force push, or overwrite an experiment
  directory.
- Do not stage, commit, or push without separate user authorization.
- Tape, pilot rows, summaries, checkpoints, and logs remain untracked under
  protected `logs/` or `runs/` paths.
