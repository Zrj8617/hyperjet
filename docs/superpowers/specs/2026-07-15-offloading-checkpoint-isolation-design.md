# Forced-Hover Offloading Checkpoint Isolation Design

Date: 2026-07-15
Branch: `zrj_3_static_ue`
Baseline commit: `42e319e4b7c64b0963da0d2c010c4638a29b9e5b`

## 1. Purpose

The current evidence does not establish whether the three trained offloading actors learned a reproducible policy improvement over random assignment. Seed 42 and seed 86 retain normalized offloading entropy near one, while seed 1042 decreases only from approximately `0.9995` to `0.983` and therefore also remains close to uniform. Small logit differences can nevertheless produce very different deterministic argmax policies.

This experiment evaluates the offloading inference path at multiple historical checkpoints under one fixed movement intervention. Its primary question is:

> Under the same forced-hover geometry and common environment seeds, does each checkpoint's deterministic offloading actor consistently outperform a reproducible random policy?

Secondary questions are:

1. Does deterministic offloading quality improve or degrade between episodes 100 and 1000?
2. Are the large differences among model seeds 42, 86, and 1042 larger than scene-to-scene variability?
3. Does any degradation in the seed-1042 offloading inference path occur at the same training stage as its historical hover increase?

The experiment is diagnostic. It does not select a repair, change training, or prove a causal effect of movement on offloading.

## 2. Existing evidence

### 2.1 Main policy gate

The completed 60-cell policy gate is stored at:

`/data2/zrj2025/HyperUAV/runs/offloading_policy_gate_20260714_215332`

Greedy EFT reduced its own local estimated regret for all three checkpoint seeds, but its system-level effect was inconsistent:

| Model seed | Greedy-minus-actor arrival completion | Greedy-minus-actor mean flowtime |
|---|---:|---:|
| 42 | `+0.0032` | `+16.09 s` |
| 86 | `-0.0527` | `+44.04 s` |
| 1042 | `+0.1354` | `-207.75 s` |

This is Phase-F branch 2: EFT is not a reliable global-DAG training teacher.

### 2.2 Existing forced-hover isolation

The completed 30-cell actor-versus-greedy isolation is stored at:

`/data2/zrj2025/HyperUAV/runs/offloading_movement_isolation_20260714_222309`

At the final checkpoints, mean arrival completion was:

| Model seed | Actor | Greedy EFT |
|---|---:|---:|
| 42 | `0.7992` | `0.7931` |
| 86 | `0.8276` | `0.7931` |
| 1042 | `0.4797` | `0.7931` |

All cells had `movement_frozen=true`, zero invalid assignments, and zero KaHyPar degraded slots. This proves that final deterministic offloading behavior differs across trained checkpoints even when evaluation movement is forced to hover. It does not show whether the actor policies beat random assignment, and it does not locate when their differences emerged.

### 2.3 Vetoed counterfactual formulation

The first action-value counterfactual formulation at commit `42e319e` failed its short deterministic gate. Its optional module remains default-disabled with:

- `offloading_counterfactual_coef=0`;
- `offloading_action_value_loss_coef=0`.

This experiment must keep both values zero and must not modify that implementation.

## 3. Interpretation of forced hover

`--freeze-movement` forces every UAV to execute hover. It does not freeze or replay the learned movement policy.

For a common environment seed, forced hover removes evaluation-time movement actions as a confounder and pins UAV geometry to the environment initialization. Historical checkpoints are therefore compared from the same initial geometry without their movement heads changing UAV positions.

This is a strong but bounded isolation:

- it tests the deterministic offloading inference path, including the shared HGNN and offloading head;
- it does not isolate the offloading head from the shared HGNN;
- after different offloading decisions occur, queues, completion times, UE activity, later DAG arrivals, and later observations may diverge;
- it therefore compares closed-loop offloading policies from matched initial conditions, not identical full trajectories.

If seed 1042 performs well at episode 100 and poorly at episode 1000 under the same forced-hover initial geometry, the offloading inference path has degraded during training. Aligning that curve with the historical training hover curve can establish temporal association, but not that movement caused the degradation.

## 4. Frozen experiment variables

The experiment must not change:

- environment reward;
- completed-DAG weight `16`;
- entropy coefficients;
- DAG arrival process or DAG size;
- PPO or GAE;
- shared-HGNN architecture;
- critic detach controls;
- checkpoint weights;
- UE mobility provenance;
- KaHyPar configuration;
- counterfactual coefficients, which remain zero for the historical checkpoints.

No training process is launched.

## 5. Checkpoints and policies

### 5.1 Actor checkpoints

Use all three formal moving-UE model seeds:

- `42`;
- `86`;
- `1042`.

For every model seed, use the same fixed training episodes:

- episode 100;
- episode 300;
- episode 600;
- episode 1000.

The checkpoint list is fixed before evaluation and must not be changed in response to completion, flowtime, entropy, margin, or regret results.

### 5.2 Reference policies

Run two forced-hover references on the same environment seeds:

- `random_hash` as the primary no-learning baseline;
- `greedy_eft_teacher` as a diagnostic heuristic reference only.

The reference policies use one fixed compatible checkpoint solely to construct the evaluation stack and record provenance. Because movement is forced and the actual offloading assignment is supplied by the reference policy, reference system performance is not repeated for all actor checkpoints.

The historical pooled random completion of approximately `0.607` used learned movement. It must not be compared as an absolute baseline for this forced-hover experiment.

## 6. Evaluation protocol

Use environment seeds:

`4242, 4243, 4244, 4245, 4246`

Every cell uses:

- one deterministic evaluation episode;
- 200 arrival slots;
- drain until all DAGs complete or 500 drain slots;
- forced-hover UAV movement;
- moving UE behavior inherited from checkpoint provenance;
- `freeze_ue_mobility=false` for these moving checkpoints;
- completed-DAG weight `16` inherited from checkpoint provenance;
- shared HGNN and the checkpoint's detach provenance;
- full KaHyPar hyperedges;
- no rendering.

Actor cells:

`3 model seeds x 4 checkpoints x 5 environment seeds = 60`

Reference cells:

`1 random policy x 5 environment seeds = 5`

`1 greedy policy x 5 environment seeds = 5`

Total first-stage size:

`70 cells`

All cells run sequentially on one selected GPU. They write to new, timestamped roots and never overwrite earlier results.

## 7. Initial-state pairing invariant

Before launching the matrix, verify that repeated initialization with one environment seed produces identical:

- UAV IDs and initial coordinates;
- UE IDs, initial coordinates, activity state, and mobility configuration;
- initial hotspot count;
- relevant environment RNG state before the first policy action.

Also verify that selecting `actor_argmax`, `random_hash`, or `greedy_eft_teacher` does not alter environment initialization. `random_hash` must retain its existing invariant that it does not mutate NumPy global RNG, Torch RNG, or environment RNG.

The verification must run without editing the training or evaluation path. Use the existing RNG-pairing smoke coverage plus an ephemeral preflight assertion based on the existing evaluation environment-construction helper. Record the check result and a stable initial-state fingerprint in the experiment manifest.

If initial-state equality fails, do not launch the 70-cell matrix.

## 8. Execution boundary

The existing `scripts/run_offloading_policy_gate.py` does not forward `--freeze-movement`. It must not be used as though it did.

To avoid a repository code change, generate a run-local manifest and launcher under the new result root. The launcher invokes `scripts/eval_clean_mainline.py` directly for every cell with `--freeze-movement`. It is an experiment artifact, not repository source, and must never be committed.

Use a persistent named `tmux` session if available. If `tmux` is unavailable, use `nohup` and record:

- launcher PID;
- active Python PID;
- selected GPU;
- full command matrix;
- checkpoint paths;
- run and log roots;
- git HEAD;
- start time.

The launcher stops on the first nonzero cell exit. It must not silently retry, change a checkpoint, change an environment seed, or skip a failed cell.

## 9. Command template

Every actor cell follows:

```bash
/data2/zrj2025/.conda/envs/uav322/bin/python \
  /data2/zrj2025/HyperUAV/scripts/eval_clean_mainline.py \
  --checkpoint <absolute-checkpoint-path> \
  --episodes 1 \
  --arrival-steps 200 \
  --max-drain-steps 500 \
  --seed <environment-seed> \
  --device <selected-cuda-device> \
  --output-dir <new-cell-output-directory> \
  --run-name <unique-model-episode-environment-policy-name> \
  --offloading-policy actor_argmax \
  --freeze-movement \
  --no-render
```

Reference cells differ only in `--offloading-policy` and their unique names and output directories.

## 10. Outputs and integrity checks

Use new roots of the form:

`/data2/zrj2025/HyperUAV/runs/phase5_checkpoint_offloading_isolation_<timestamp>`

`/data2/zrj2025/HyperUAV/logs/phase5_checkpoint_offloading_isolation_<timestamp>`

For each cell retain the existing eval config, metrics, summary, and offloading-decision trace. The aggregate report must include:

- arrival generated and completed DAGs;
- arrival completion;
- throughput;
- arrival and final backlog;
- mean, median, and P90 drain DAG flowtime;
- queue/resource wait;
- cross-UAV transfer time;
- normalized actor entropy;
- top1-top2 probability margin;
- actor/greedy agreement;
- selected estimated regret;
- valid-candidate distribution;
- invalid assignment count;
- forced-hover flag, hover ratio, and displacement;
- KaHyPar status;
- checkpoint, model seed, environment seed, policy, and git provenance.

The matrix is valid only if:

1. all 70 cells complete;
2. every cell has `movement_frozen=true`;
3. every cell has hover ratio `1` and mean displacement `0`, within numerical tolerance;
4. invalid assignment count is zero;
5. there is no NaN, Inf, traceback, or unexplained KaHyPar degradation;
6. every checkpoint and policy provenance field matches the manifest;
7. there is no residual evaluation process;
8. earlier `runs/` and `logs/` content is not deleted, moved, or overwritten.

## 11. Analysis

For every actor checkpoint, compare it with `random_hash` under the same environment seed. Report both cell-level paired deltas and pooled counts. Do not rely on pooled completion alone because policy decisions can change later DAG generation.

At each checkpoint report:

- actor-minus-random arrival completion;
- actor-minus-random weighted drain flowtime improvement;
- actor-minus-random backlog change;
- the count of common scenes with consistent improvement;
- entropy and top1-top2 margin;
- estimated regret and actor/greedy agreement.

For seed 1042, align these evaluation values with its historical training hover trajectory. Do not relabel entropy near `0.983` as confident behavior without supporting probability margins.

## 12. Pre-registered interpretations

### 12.1 Stable weak ranking in seeds 42 or 86

Evidence requires all of the following for a model seed:

- at least four of five common scenes improve over random in the same primary direction;
- at least two of arrival completion, weighted drain flowtime, and arrival backlog improve;
- the effect appears at more than one checkpoint rather than in one isolated cell group.

If a near-uniform actor satisfies these conditions, conclude that small logit differences contain a reproducible weak ranking. Do not claim that the stochastic policy has converged.

### 12.2 No reproducible offloading learning

If actor-minus-random effects are small, inconsistent across scenes, or reverse across checkpoints for all three model seeds, conclude that no actor has demonstrated reproducible offloading improvement. Large training-run differences may then be dominated by movement, task randomness, small argmax biases, and closed-loop trajectory amplification.

This result motivates a separately designed action-credit or load experiment. It does not by itself authorize changing DAG arrival rate.

### 12.3 Seed-specific degradation

If one model seed is competitive with random at early checkpoints but degrades consistently at later checkpoints while control seeds remain stable, conclude that its offloading inference path degraded during training.

For seed 1042, temporal alignment with increasing training hover supports coupling between the two phenomena. It does not prove that movement caused the offloading degradation.

### 12.4 Ambiguous first-stage result

The first stage is considered ambiguous if the main actor-versus-random direction is not consistent in at least four of five scenes, or if completion, flowtime, and backlog materially disagree.

Only then expand to a second stage:

- actor checkpoints: episode 100 and episode 1000;
- environment seeds: 20 fixed common seeds;
- actor cells: `3 x 2 x 20 = 120`;
- random reference: 20 cells;
- greedy reference: 20 cells;
- total: 160 cells.

The second stage requires a separately presented execution manifest but no training-code change.

## 13. What the experiment cannot prove

This experiment cannot prove:

- movement caused offloading degradation;
- offloading caused hover collapse;
- the offloading head rather than the shared HGNN is responsible;
- random task attributes or one-DAG-per-slot load is the sole cause;
- greedy EFT is a valid training teacher;
- a specific PPO repair will work;
- any checkpoint is a final superior model.

It locates whether deterministic offloading quality exists, changes over training, and exceeds a forced-hover random baseline.

## 14. Next-step boundaries

After the first-stage report:

- reproducible weak ranking plus seed-specific degradation leads to a separately reviewed stability or causal-intervention design;
- no actor beating random leads to a separately reviewed action-credit and load diagnosis;
- an ambiguous result leads only to the pre-registered 20-seed expansion;
- no outcome directly authorizes formal 1000-episode training.

The three final formal runs remain blocked until a repair is independently designed, tested, and passes its own short deterministic gate.

## 15. Repository and server safety

Never modify, delete, stage, or commit:

- local `docs/session_handoff_phase4.md`;
- local `runs/`;
- server historical `logs/`;
- server historical `runs/`.

The run-local launcher, manifest, logs, and aggregate analysis are experiment artifacts and must not be committed. All server actions remain confined to `/data2/zrj2025` and use new paths.
