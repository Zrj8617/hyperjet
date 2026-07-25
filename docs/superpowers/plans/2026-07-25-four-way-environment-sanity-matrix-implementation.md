# Four-Way Environment Sanity Matrix Implementation Plan

## Objective

Run a controlled four-method comparison over model/environment seeds
`42`, `86`, and `1042`, with `1000` episodes per cell:

1. `random_hash` fixed-hover, no learning;
2. `greedy_eft` fixed-hover, no learning;
3. `mappo_mlp` fixed-hover training;
4. `mappo_hgnn` fixed-hover training.

The matrix diagnoses whether the clean environment responds to assignment
quality and whether the MAPPO learning path works without hypergraph message
passing. It does not promote greedy EFT to a formal learning teacher.

## Fixed Protocol

- seeds: `42, 86, 1042`;
- episodes per cell: `1000`;
- slots per episode: `500`;
- movement: forced hover;
- UE mobility: enabled;
- per-UE active-DAG cap: `1`;
- base arrival probability: `0.0145`;
- completed-DAG reward weight: `16`;
- rollout horizon for trainable methods: `128`;
- gamma: `0.99`;
- GAE lambda: `0.95`;
- PPO clip: `0.2`;
- PPO epochs per rollout: `3`;
- value target normalization: disabled explicitly;
- value clipping: disabled explicitly;
- counterfactual action-value correction: disabled;
- lagged residual-Q correction: disabled.

Random and greedy cells execute the same 500-slot episode semantics as
training but do not create an optimizer or update a model. `random_hash` must
remain RNG-neutral. Because actions change task completion and later per-UE
arrival eligibility, pairing is guaranteed only at episode initialization,
not for complete closed-loop trajectories.

## Task 1: Independent Task MLP Encoder

Add a task-only encoder with the same call contract and output shape as the
clean incidence HGNN:

- input: task features `[N, 12]` and an incidence matrix argument;
- output: embeddings `[N, task_embedding_dim]`;
- architecture: `Linear -> ReLU -> Linear`;
- incidence values must not affect output;
- empty task batches must return `[0, task_embedding_dim]`.

Keep the historical `modules.hgnn` checkpoint key and module field for
compatibility, but record `task_encoder=mlp` in the run configuration. The MLP
mode is an ablation of message passing, not a claim that no task encoder exists.

## Task 2: Training and Evaluation Encoder Selection

Add `--task-encoder {hgnn,mlp}` to the clean training entrypoint.

- build the requested encoder;
- write the encoder type to `config.json`, training logs, summaries, and
  checkpoints through the existing config snapshot;
- reject resume when the requested encoder differs from the checkpoint;
- preserve legacy checkpoints by resolving a missing encoder field as `hgnn`;
- make deterministic evaluation construct the encoder declared by the
  checkpoint;
- keep actor, critic, optimizer, reward, and rollout behavior unchanged.

## Task 3: No-Learning Baseline Runner

Add a clean baseline entrypoint that:

- supports `random_hash` and `greedy_eft`;
- runs exactly `episodes x max_steps_per_episode`;
- forces all UAV movement actions to hover;
- uses the existing frozen-ready order, candidate estimator, and sequential
  temporary reservation;
- does not require a checkpoint, HGNN, actor, critic, optimizer, or Torch;
- writes `config.json`, `episode_metrics.jsonl`, `run_summary.json`, and
  TensorBoard-compatible scalar events when TensorBoard is available;
- records reward components, DAG completion, flowtime, backlog, queue pressure,
  accepted assignments, no-candidate skips, and KaHyPar status;
- stops on the first exception and records a failed summary.

The greedy policy minimizes estimated finish time with UAV ID as the stable
tie-breaker. The random policy uses the existing stable hash over environment
seed, episode, slot, task ID, and legal UAV IDs and must not consume global RNG.

## Task 4: Matrix Manifest and Guarded Launcher

Add a matrix planner/launcher that creates twelve independent cells:

- three `random_hash`;
- three `greedy_eft`;
- three `mappo_mlp`;
- three `mappo_hgnn`.

Requirements:

- dry-run by default;
- explicit `--execute` required to launch;
- new output root only;
- exact commands and immutable protocol saved in a manifest;
- per-cell stdout/stderr logs;
- fail-fast for preflight/smoke failures;
- no deletion or reuse of historical run directories;
- configurable safe concurrency, chosen only after server GPU/process checks.

## Task 5: Verification

Local checks that do not require Torch:

1. Python syntax compilation;
2. CLI/parser and manifest checks;
3. baseline runner short smoke for both policies;
4. random-hash RNG neutrality;
5. greedy sequential reservation;
6. protected untracked files remain untouched.

Server checks on the exact deployed revision:

1. Torch/CUDA and KaHyPar smoke;
2. MLP output shape and incidence-invariance test;
3. one short cell for each of the four methods;
4. checkpoint save/load/eval for both encoder types;
5. manifest contains exactly twelve unique cells;
6. no NaN/Inf or degraded KaHyPar state.

No 1000-episode cell starts until all gates pass.

## Task 6: Server Launch and Monitoring

All server commands and new artifacts must remain below `/data2/zrj2025`.
Preserve historical `logs/` and `runs/`.

Before launch:

- verify local, GitHub, and server revision equality;
- inspect active processes, GPUs, memory, and disk;
- select a concurrency that does not interfere with existing work;
- create a new timestamped matrix root;
- record manifest, revision, environment, launcher PID, and child PIDs.

After launch:

- verify every expected process and output directory;
- expose the new root through the existing persistent TensorBoard service;
- report the permanent link and the exact run-name mapping;
- monitor first updates for failures before leaving the long matrix running.

## Interpretation Gate

- `greedy_eft > random_hash`: environment responds to assignment quality;
- `mappo_mlp` improves over its early checkpoints/random: MAPPO learning path is
  viable without hypergraph message passing;
- `mappo_mlp` learns while `mappo_hgnn` does not: inspect hypergraph encoding or
  shared HGNN gradients;
- neither no-learning baseline separates: inspect environment action execution
  and reward linkage before tuning neural networks;
- neither trainable method learns while greedy separates strongly: inspect
  MAPPO credit assignment, observation sufficiency, and optimizer path.
