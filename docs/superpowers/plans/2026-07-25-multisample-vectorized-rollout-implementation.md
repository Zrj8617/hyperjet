# Multisample Vectorized Rollout Implementation Plan

## Goal

Add synchronous multi-environment sampling to the clean MAPPO training path without
changing the meaning of `--episodes`, mixing GAE trajectories, or affecting the
currently running `ce6075b` experiments.

The supported gate sizes are `num_envs=1,2,4,8`. Throughput artifacts and run names
must carry the `multisample` label.

## Required semantics

1. `--episodes N` means exactly N completed episodes in total, independent of
   `--num-envs`.
2. Each sampler lane owns an independent `Env`, `CleanGraphBuilder`, environment RNG
   stream, rollout buffer, episode totals, and optional lagged-outcome tracker.
3. A lane contributes at most one temporally contiguous trajectory to a PPO batch.
4. GAE is computed independently for every closed lane buffer. Returns and
   advantages are concatenated only after the per-lane backward recursions finish.
5. A PPO update occurs after each active lane reaches the rollout horizon or its
   episode boundary. Continuing lane states are re-encoded after the update.
6. `num_envs=1` remains the compatibility reference.
7. Checkpoints are written only after closed rollout buffers. Resume continues from
   the next episode; exact mid-episode resume remains unsupported.
8. No counterfactual or lagged residual-Q option is enabled by the gate.

## Implementation steps

### 1. Multi-buffer PPO update

- Add `CleanPPOUpdater.update_many`.
- Validate that every input buffer is non-empty and closed.
- Compute GAE separately using each buffer's own bootstrap value.
- Concatenate records, returns, and advantages for the existing loss path.
- Preserve `update(buffer)` as a one-buffer wrapper.
- Log the number of rollout environments and per-environment record counts.

### 2. Independent sampler lanes

- Add `--num-envs` with a positive-integer validator.
- Derive stable environment seeds from the training seed.
- Preserve Python and NumPy RNG state per lane around environment/graph operations;
  keep Torch policy sampling on the shared training RNG stream.
- Give each lane its own environment, graph builder, buffer, prepared/encoded state,
  reward totals, and terminal state.
- Allocate total episode identifiers in waves of at most `num_envs`.

### 3. Training and logging

- Count `global_slot` as total environment transitions across all lanes.
- Record `num_envs`, active lane count, lane index, environment seed, aggregate
  environment slots/sec, and elapsed time.
- Keep episode terminal metrics separate for each lane.
- Aggregate KaHyPar shutdown health across all graph builders.
- Include multisample controls in config, summaries, checkpoint validation, and
  resume metadata.

### 4. Gates

- Static smoke:
  - CLI validation for 1/2/4/8.
  - Matrix commands contain the requested environment counts and multisample label.
  - Synthetic multi-buffer GAE proves there is no cross-environment leakage.
- Server equivalence:
  - `num_envs=1` smoke remains healthy.
  - For deterministic fixed-action environment traces, vector lanes match isolated
    runs with the same derived seeds.
- Server throughput:
  - Run 1/2/4/8 sequentially on one non-production GPU.
  - Use the same total number of environment slots and PPO settings.
  - Report environment slots/sec, updates/sec, wall time, GPU memory, and failures.
  - Mark results `multisample_background_load` because the 12 formal experiments
    remain active.

## Safety

- Do not switch or modify the checkout used by the active server experiments.
- Use a separate server worktree under `/data2/zrj2025`.
- Use a GPU other than physical GPU 6.
- Store all gate outputs in new `runs/multisample_*` and `logs/multisample_*`
  directories.
- Preserve existing server `logs/`, `runs/` and protected local untracked files.
