# Stage 1 Temperature-Only Concentration Follow-Up Implementation Plan

Date: 2026-08-06  
Branch: `zrj_3multisample`  
Required starting commit: `6b6cc75f8269a3e99a0e9145b540734be1d86504`  
Design authority: `docs/superpowers/specs/2026-08-04-stage1-temperature-only-concentration-follow-up-design.md`  
Frozen design SHA-256: `84a551f6d997c7251313a617a5f66bcc743aa326f7fa9f48d95cf07f71c50606`

## 0. Scope, authorization, and stop conditions

Implement only the frozen checkpoint-only temperature mechanism diagnostic:

```text
original Stage 1 A environment
-> random-material evaluation tape
-> SHA-256 keyed common Gumbel noise
-> keyed T=1 static corpus
-> static replay at T=1.0, 0.75, 0.5, 0.25
-> separate closed-loop rollouts at the same temperatures
-> deterministic reachability ceiling
-> predeclared static and closed-loop analysis
```

Do not modify checkpoint weights, model architecture, reward, arrival
probabilities, capacity values, seven-dimensional actor input, HGNN,
`GraphSnapshot`, PPO/MAPPO, critic, GAE, or Stage 2 fields. Do not train or
retrain a model. Do not reuse the 2x2 offered-before-cap adapter or its tape.

This plan does not itself authorize implementation, server synchronization,
pilot execution, or formal execution. Each later execution phase requires the
authorization stated by the user at that phase. Any failed gate stops the
phase without changing experimental semantics.

## 1. Freeze repository and input identities before implementation

Read-only checks:

- confirm branch `zrj_3multisample`;
- confirm HEAD `6b6cc75f8269a3e99a0e9145b540734be1d86504`;
- record `git status --short`, preserving every unrelated untracked file;
- hash the frozen design document;
- verify the three checkpoint paths and SHA-256 values from the design;
- verify `config.CLEAN_MAX_QUEUE_PER_UAV == 16`, default active cap `1`, and
  `config.EPISODE_LENGTH >= 200`;
- record the existing Stage 1 evaluator defaults and current environment
  arrival/movement order.

Create one implementation manifest under a new protected local diagnostic
directory. It records the baseline commit, design hash, anticipated files,
checkpoint map, temperatures, episode mapping, sampling replicates, bootstrap
seed, and all planned output roots. It is not added to training inputs.

Verification:

- no tracked file differs before implementation;
- no existing `logs/` or `runs/` path is deleted or overwritten;
- no commit, push, or server write occurs without separate authorization.

## 2. Add canonical hashing and temperature sampling primitives

File:

- add `environment/stage1_temperature_sampling.py`

Implement small pure functions with no Torch dependency:

- canonical compact UTF-8 JSON serialization with `allow_nan=False`;
- canonical SHA-256;
- exact `stage1_temperature_gumbel_v1` key validation;
- digest-prefix to 53-bit open uniform conversion;
- float64 Gumbel conversion;
- stable legal-logit softmax;
- temperature-scaled keyed Gumbel-max action selection;
- deterministic masked argmax with UAV-ID tie-break;
- static distribution diagnostics.

The sampling key contains exactly:

```text
[
  "stage1_temperature_gumbel_v1",
  checkpoint_sha256,
  evaluation_scenario_seed,
  slot_index,
  stable_task_id,
  decision_order,
  sampling_replicate,
  candidate_uav_id
]
```

Temperature is not a key field. Legal logits are copied to CPU float64 before
softmax and Gumbel-max. Illegal candidates never enter the float64 legal
vector. Ties use UAV ID.

Verification:

- hand-calculate canonical bytes, digest prefix, `u`, and Gumbel for fixed
  keys;
- repeated processes return byte-identical keyed noise;
- changing only temperature leaves all Gumbels unchanged;
- changing replicate or UAV ID changes its Gumbel;
- `T=1` probability differs from the current masked softmax by at most `1e-6`;
- positive temperatures preserve deterministic masked argmax;
- applying temperature to probabilities is absent from the implementation.

## 3. Add the sharded random-material tape

Files:

- add `environment/stage1_temperature_tape.py`
- add `scripts/generate_stage1_temperature_tape.py`

Represent one logical tape as a create-only manifest plus 20 immutable scenario
shards. Sharding avoids loading all 240,000 potential DAG templates into memory
at once while preserving one ordered logical checksum.

The manifest freezes:

- schema and canonical serialization version;
- episode indices `0..19`;
- mapping `evaluation_scenario_seed = 424242 + episode_index`;
- slot indices `0..199`;
- UE/UAV counts and environment constants;
- ordered scenario-shard paths, sizes, and SHA-256 values;
- ordered full logical tape checksum.

Each scenario shard freezes random primitives, not final events:

- hotspot-center uniforms;
- authoritative post-reset UE position, speed, and heading uniforms;
- authoritative post-reset UAV position uniforms;
- 200 x NUM_UES pairs of Gaussian-Markov speed/heading standard normals;
- 200 x NUM_UES arrival uniforms;
- 200 x NUM_UES potential DAG templates derived from isolated keyed DAG random
  material.

The template contains random DAG content only. It excludes source position,
arrival time, admission status, and final runtime IDs that depend on counters.
Remap DAG and task identifiers to the frozen stable contract and preserve
task-local indices, dependencies, sink IDs, k-hop edges, critical-path flags,
bandwidth draws, and every task attribute needed for exact instantiation.

Generate templates with an isolated NumPy RNG context. Save and restore the
caller's Python and NumPy RNG states. Never advance the environment's runtime
RNG while creating or loading a template.

The writer is create-only. Existing manifest or shard paths cause a hard
failure unless the caller explicitly requests validation-only mode and every
control matches.

Verification:

- canonical round-trip and checksum validation for every shard;
- all stable DAG/task IDs are globally unique across all shards;
- stable IDs do not depend on active-cap eligibility or temperature;
- no shard contains final UE positions, hotspot-membership bits, arrival bits,
  generated/admitted bits, source position, or arrival time;
- template generation/instantiation equivalence covers all DAG/task fields;
- full logical checksum is reproducible across processes;
- pilot scenarios are exactly shards 0 and 1 of the already-complete formal
  tape, never a separately generated tape.

## 4. Add neutral explicit UE-noise plumbing without changing defaults

Files:

- minimally modify `environment/user_equipments.py`
- minimally modify `environment/env.py`

Extend `UE.update_position` with optional explicit standard-normal inputs for
speed and heading. When omitted, execute the current two `np.random.normal()`
calls in the current order. When supplied, use the values in the unchanged
Gaussian-Markov formula, including:

- previous walk speed and heading;
- configured alpha and sigmas;
- clipping;
- trajectory-current `service_waiting` speed scale;
- position reflection and heading correction;
- position commit flag.

Extract the UE loop in `Env.prepare_slot_state` behind one protected hook whose
default implementation calls `ue.update_position` exactly as before. Do not
change the surrounding order: increment time, move UEs, freeze service
positions, process arrivals, refresh ready states, and freeze ready tasks.

Verification:

- default path produces the same seeded UE states, positions, arrival funnel,
  ready IDs, rewards, and metrics as the pre-refactor path;
- explicit normals reproduce the default path when fed the captured draws;
- the same innovations with `service_waiting=false` versus true produce the
  expected different positions through only the existing speed scale;
- no new observation, actor feature, or `GraphSnapshot` field appears.

## 5. Add the original-semantics Stage 1 tape environment

File:

- add `environment/stage1_temperature_diagnostic.py`

Add an isolated `Stage1TemperatureDiagnosticEnv` subclass that consumes one
validated scenario shard. It may override only reset-time random-state binding,
the UE-noise hook, and clean DAG arrival processing.

Reset behavior:

- call the clean reset under an isolated/restored RNG context;
- derive hotspot center, authoritative UE state, and UAV positions from raw
  tape primitives;
- reset task manager, executor, metrics, service-waiting flags, and slot
  counters exactly as the clean environment requires;
- assert active cap 1, queue cap 16, UE mobility enabled, and hover UAV
  movement.

Per-slot movement uses that shard's two Gaussian innovations per UE but the
trajectory's current UE speed, heading, position, and `service_waiting` flag.
It calculates final positions at runtime; the tape never supplies them.

Override arrival processing with the original Stage 1 funnel order:

```text
increment arrival_attempt_count
-> check can_accept_dag_for_ue
   -> blocked: increment arrival_blocked_count/reason and continue
-> increment arrival_draw_count
-> compute p from the runtime UE position and fixed episode hotspot
-> read the slot/UE arrival uniform
   -> u >= p: increment arrival_no_event_count
   -> u < p: increment arrival_sampled_event_count
      -> instantiate the frozen potential DAG template
         with runtime source_pos and current physical time
      -> enter service_waiting
      -> increment arrival_admitted_count
```

Do not introduce offered-before-cap counters or semantics. If output exposes
`admitted_dag_count`, assert it equals both `arrival_admitted_count` and
`generated_dag_count`.

Verification:

- constructed active-cap-blocked cases do not inspect the arrival comparison,
  instantiate a template, or increment draw/no-event/sample/admitted counts;
- constructed eligible cases match hand-calculated hotspot probabilities;
- the same raw tape may produce different valid positions and arrivals after
  temperature trajectories diverge;
- runtime DAG source position equals the current frozen UE service position;
- runtime arrival time equals current physical time;
- every successful episode uses slots `0..199` and has no 150/50 load-drain
  branch;
- distribution-equivalence smokes compare original versus taped movement,
  eligible arrival, blocked arrival, hotspot conditioning, and DAG attributes.

## 6. Add the strict frozen-checkpoint loader

File:

- `environment/stage1_temperature_diagnostic.py`

Reuse the validated loader principle from the capacity diagnostic without
importing its capacity runner:

```text
isolated probe Env and CleanGraphBuilder
-> actual GraphSnapshot task feature dimension
-> checkpoint input_proj weight dimension
-> require both equal 12
-> construct MLP encoder and scorer
-> strict=True load
```

Save and restore Python, NumPy, Torch CPU, and Torch CUDA RNG state around the
probe. Close the builder in `finally`. Validate checkpoint schema, group S1-B,
training-seed mapping, completed update 30, file SHA-256, MLP encoder type, and
all state-dict keys.

Verification:

- all three formal checkpoints strict-load;
- altered dimension, hash, seed, update, missing key, or unexpected key is a
  hard failure;
- probe state never enters tape, static corpus, or rollout checksums.

## 7. Add the isolated sequential temperature policy

File:

- `environment/stage1_temperature_diagnostic.py`

Implement a diagnostic-only sequential actor loop using the current
`build_offloading_candidate_components`, frozen task order, current MLP
encoder/scorer, and `TemporaryReservationState`.

For each ready task:

```text
build the unchanged candidate components from current reservation
-> construct actor candidate features with the unchanged 7D candidate input
-> score once with the frozen MLP
-> copy raw logits, mask, legal UAV IDs, and EFT vector
-> select with keyed Gumbel-max at the requested T
-> reserve the selected assignment immediately
```

Zero and one legal candidate behavior remains identical to Stage 1 and is
excluded from concentration metrics. The diagnostic loop records immutable
nontrivial-decision records when requested.

Do not change `CleanOffloadingActor`, its scorer, or production sampling. Add
equivalence smokes comparing the diagnostic T=1 path against the production
actor on identical live states:

- candidate IDs, masks, features, EFT, and raw logits equal;
- deterministic argmax and tie-break equal;
- masked probabilities differ by at most `1e-6`;
- reservations and committed deterministic assignments equal.

## 8. Add immutable static-corpus records and replay

Files:

- add `environment/stage1_temperature_analysis.py`
- `environment/stage1_temperature_diagnostic.py`

The keyed T=1 rollout writes one immutable record for every nontrivial decision
with:

- complete checkpoint/scenario/slot/task/decision/replicate identity;
- derived episode index and asserted seed mapping;
- stable legal candidate ordering and mask;
- copied raw legal logits and all-candidate EFT vector;
- keyed candidate Gumbels;
- sampled T=1 action;
- deterministic actor argmax and greedy-EFT action;
- immutable actor-consumed tensors or a content-addressed tensor reference;
- record checksum and source-corpus checksum.

Make arrays read-only before serialization. Use create-only writers and refuse
duplicate record keys. Static replay loads the frozen corpus, verifies its
checksum, calculates all four temperatures without creating or advancing an
environment, and verifies the corpus checksum again after replay.

Verification:

- attempted array mutation fails;
- replay does not construct `Env`, call prepare/commit, or mutate a source
  record;
- T=1 replay reproduces the recorded action and distribution;
- all temperatures share identical state, mask, logits, EFT, and Gumbels;
- argmax and mask invariance holds for every replay row;
- zero/single candidate records never enter concentration denominators.

## 9. Add the pilot/formal closed-loop runner

Files:

- add `scripts/run_stage1_temperature_followup.py`
- add `scripts/generate_stage1_temperature_tape.py`

The runner requires an existing validated tape manifest and explicit mode:

```text
--phase pilot | formal
--temperatures 1.0 0.75 0.5 0.25
--sampling-replicates ...
--scenario-indices ...
--max-physical-slots 200
--device cuda
--output-dir NEW_PATH
```

Reject any controls that do not match the frozen phase:

- pilot: scenarios `0,1`, replicates `0,1`, 48 closed-loop rows;
- formal: scenarios `0..19`, replicates `0..4`, 1200 closed-loop rows.

Run keyed T=1 for each checkpoint/scenario/replicate and write its static
corpus before lower-temperature static replay is analyzed. Closed-loop lower
temperatures run in separately reset environments. Every closed-loop row
records checkpoint/tape/code identities, temperature, replicate, scenario
seed, derived episode index, 200-slot count, original Stage 1 arrival funnel,
service outcomes, backlog, decision counts, legality, and finite flags.

Use separate create-only directories for pilot static corpus, pilot static
replay, pilot closed loop, formal static corpus, formal static replay, formal
closed loop, and formal analysis. Never merge pilot and formal rows.

Verification:

- runner refuses existing output directories;
- duplicate or missing keys fail immediately;
- every successful row executes exactly 200 slots;
- `scenario_seed == 424242 + episode_index` for every row;
- all rows use active cap 1, queue cap 16, MLP, and seven-dimensional actor
  candidates;
- no training optimizer, critic, GAE, boundary, or Stage 2 object is created.

## 10. Implement static metrics and deterministic reachability

File:

- `environment/stage1_temperature_analysis.py`

Calculate Stage 1-compatible metrics by checkpoint, temperature, scenario,
and replicate:

- sampled EFT regret mean/median/p95;
- greedy/sample agreement;
- margin-5 and margin-20 counts and accuracy;
- normalized entropy, maximum probability, and probability margin;
- deterministic EFT regret, greedy agreement, and margin-20 accuracy.

Aggregate decisions within scenario/replicate first. Calculate checkpoint
means and the exact T1-relative regret-reduction formula. Bootstrap 20 scenario
blocks with replacement for 10,000 resamples under analysis seed `20260804`,
retaining all five replicates within each selected block.

Before classifying temperature response, calculate per checkpoint:

```text
deterministic_margin20_accuracy
deterministic_mean_EFT_regret
max_achievable_regret_reduction
```

Apply zero-denominator semantics exactly. If any checkpoint fails either
reachability threshold, emit authoritative `ranking_limited` and prohibit a
concentration-primary label while still reporting the sweep.

Verification:

- hand-built logits/EFT cases cover correct and incorrect argmax, 5/20-second
  margins, exact logit ties, zero T1 regret, and unreachable 50% reduction;
- bootstrap is byte-reproducible and never resamples decisions independently;
- pooled checkpoint means cannot replace checkpoint-local gates.

## 11. Implement closed-loop guardrail and final classification

Files:

- `environment/stage1_temperature_analysis.py`
- add `scripts/analyze_stage1_temperature_followup.py`

Audit exact row counts, identities, tape/corpus checksums, finite values,
invalid assignments, 200-slot horizons, and phase separation before analyzing
outcomes.

For every checkpoint/temperature, compare closed-loop means with its keyed T=1
matched-start rows and calculate the frozen material/catastrophic thresholds.
Set `material_degradation_triggered` when any material metric triggers.

Fail the guardrail when:

1. any checkpoint has a catastrophic regression; or
2. at least two checkpoints trigger material degradation, even on different
   metrics.

Also report same-metric three-checkpoint degradation as a highlighted pattern.
Do not describe matched-start closed-loop differences as strict
counterfactual pairs.

Apply final classification in this order:

```text
technical failure -> invalid experiment; stop without scientific analysis
reachability failure -> ranking_limited
reachable + common moderate T passes -> probability_scale_primary
reachable + only T=0.25 passes -> hard_sharpening_only
reachable + T=0.25 fails -> concentration_only_rejected
```

The analyzer reports every temperature and never emits a deployment
temperature recommendation.

Verification:

- two checkpoints degrading on different material metrics fail;
- one catastrophic checkpoint fails;
- one non-catastrophic material checkpoint alone does not fail;
- all-three same-metric degradation is highlighted and fails;
- ranking-limited preempts all temperature-primary labels;
- analysis refuses pilot/formal pooling and any missing checkpoint.

## 12. Add isolated smokes

Files:

- add `scripts/smoke_stage1_temperature_sampling.py`
- add `scripts/smoke_stage1_temperature_tape.py`
- add `scripts/smoke_stage1_temperature_env.py`
- add `scripts/smoke_stage1_temperature_replay.py`
- add `scripts/smoke_stage1_temperature_analysis.py`
- add `scripts/smoke_stage1_temperature_closed_loop.py`

Smoke responsibilities:

1. sampling: canonical keys, exact Gumbel formula, temperature exclusion,
   float64 softmax, action/tie behavior;
2. tape: raw-material-only schema, sharding, stable IDs, template round-trip,
   no source position/time, create-only behavior;
3. environment: service-waiting movement, cap-before-draw funnel, dynamic
   hotspot membership, runtime source/time binding, 200-slot semantics;
4. replay: immutable records, no environment advancement, T1 reproduction,
   cross-temperature invariants;
5. analysis: reachability ceilings, bootstrap units, mixed-metric guardrail,
   catastrophic regression, classification precedence;
6. closed loop: one tiny CPU checkpoint fixture, exact row identity, strict
   load, separate static/closed-loop outputs, no optimizer or critic.

Each smoke prints a single PASS line and raises the raw exception on failure.
No smoke silently skips Torch on the server. Local Torch-unavailable skips are
explicit and do not count as server validation.

## 13. Run local structural and non-Torch checks

Local environment has no Torch. Run only:

- `python -m py_compile` for every changed/new Python file using an isolated
  temporary cache under the workspace;
- pure sampling, tape-schema, and analysis smokes that do not import Torch;
- repository searches proving no PPO/MAPPO, critic, GAE, HGNN, Stage 2, reward,
  or seven-dimensional feature change;
- default-path diffs for `Env` and `UE` limited to neutral explicit-noise
  plumbing;
- `git diff --check` and explicit intended-file inventory.

Remove only Codex-created temporary compile caches after resolving and checking
their absolute paths. Preserve every pre-existing untracked file.

Stop and report the raw failure if any local check fails.

## 14. Prepare server synchronization only after authorization

When separately authorized:

- create a tar containing only the reviewed implementation and smoke files;
- record its SHA-256 and file list;
- upload only to an explicitly authorized path under `/data2/zrj2025`;
- inspect server branch, HEAD, status, and target-file hashes before extraction;
- preserve server `logs/`, `runs/`, checkpoints, and existing tapes;
- never use reset, clean, force push, or paths outside `/data2/zrj2025`.

Before every server operation, state the exact command purpose. Do not commit,
push, run a pilot, or run formal merely because synchronization succeeds.

## 15. Run server pre-pilot gates only after authorization

Use the server Torch environment and the explicitly authorized physical GPU.
Run, in order:

1. pure sampling smoke;
2. UE/default-Env equivalence smoke;
3. tape/template smoke;
4. strict loader smoke for all three checkpoints;
5. static replay smoke;
6. reachability/guardrail analysis smoke;
7. tiny closed-loop smoke;
8. large read-only distribution-equivalence audit;
9. resource preflight for a complete 20-scenario sharded tape.

All checkpoint dimensions must resolve to 12 and strict-load. Any mismatch,
default-environment regression, cap-before-draw violation, distribution
mismatch, traceback, NaN/Inf, OOM, or unexpected resource growth stops the
phase without tape generation or pilot.

## 16. Generate the complete formal tape before pilot

When authorized and all pre-pilot gates pass:

- generate all 20 scenario shards and the final manifest once;
- validate every shard and the full logical checksum;
- record file sizes, generation time, peak memory, and checksum manifest;
- make the tape read-only by process convention and never regenerate it after
  pilot metrics are visible;
- confirm pilot uses only scenario indices 0 and 1 from this complete tape.

Do not start the pilot if any shard, template, stable ID, distribution, or
checksum gate fails.

## 17. Run the 48-episode technical pilot, then stop

When separately authorized, run exactly:

```text
3 checkpoints
x 2 formal-tape scenarios
x 2 sampling replicates
x 4 temperatures
= 48 closed-loop episodes
```

Generate its T1 static corpus and four-temperature static replay in separate
pilot paths. Validate every technical gate in the design, including exact
200-slot rows, raw-material checksum equality, strict loads, T1 compatibility,
argmax/mask invariance, corpus immutability, and absence of offered-before-cap
semantics.

Classify the pilot only as technical pass/fail. Do not inspect scientific
thresholds, alter controls, or start the 1200-episode formal run automatically.
Stop and report raw errors on any failure.

## 18. Formal execution remains a later explicit decision

Only after the user reviews and explicitly authorizes a technically passing
pilot may the 1200-episode formal matrix run. During formal execution, freeze
code, checkpoints, tape, temperatures, scenario mapping, replicates, commands,
and output paths.

After completion:

```text
technical row/checksum/legality audit
-> deterministic reachability ceiling
-> static replay concentration analysis
-> separate closed-loop guardrail analysis
-> predeclared final classification
-> stop for review
```

No outcome authorizes retraining, a deployment temperature, or Stage 2 without
a new explicit user decision.

## 19. Expected implementation file inventory

Expected tracked changes:

```text
environment/user_equipments.py                         minimal explicit-noise hook
environment/env.py                                     minimal UE-advance hook
environment/stage1_temperature_sampling.py             new pure sampling primitives
environment/stage1_temperature_tape.py                 new random-material tape
environment/stage1_temperature_diagnostic.py           new loader/env/policy/runner core
environment/stage1_temperature_analysis.py             new metrics/gates/analysis
scripts/generate_stage1_temperature_tape.py             new create-only generator
scripts/run_stage1_temperature_followup.py              new pilot/formal runner
scripts/analyze_stage1_temperature_followup.py          new strict analyzer
scripts/smoke_stage1_temperature_sampling.py            new isolated smoke
scripts/smoke_stage1_temperature_tape.py                new isolated smoke
scripts/smoke_stage1_temperature_env.py                 new isolated smoke
scripts/smoke_stage1_temperature_replay.py              new isolated smoke
scripts/smoke_stage1_temperature_analysis.py            new isolated smoke
scripts/smoke_stage1_temperature_closed_loop.py         new isolated smoke
```

Any additional tracked file requires review before implementation. Formal
tapes, corpora, rows, summaries, manifests, and analysis outputs remain under
protected `logs/` or `runs/` and are never committed.
