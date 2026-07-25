# Multisample Process-Worker Background-Load Gate

## Identity

- Branch: `zrj_3multisample`
- Tested commit: `1c025cc42bc83a20943e79ca74403a70c703963d`
- Label: `multisample_process_background_load`
- Server checkout:
  `/data2/zrj2025/HyperUAV_multisample_process_1c025cc`
- Artifact root:
  `/data2/zrj2025/HyperUAV_multisample_process_1c025cc/runs/multisample_process_throughput_gate/20260725_115849_multisample_process_background_load`
- Device: physical GPU 5
- Background condition: the 12-cell `ce6075b` matrix remained active on GPU 6.

## Architecture exercised

- One persistent spawned process per environment.
- Each worker owns its Env, GraphBuilder, KaHyPar worker, RNG stream, and CPU
  inference model.
- The learner broadcasts one frozen parameter snapshot to every active worker.
- Workers collect their rollout horizons concurrently.
- Closed rollout buffers return to the main process.
- GAE is computed separately per worker trajectory and concatenated only afterward.
- The main process performs one GPU PPO update and broadcasts the new parameters
  before the next collection round.

## Protocol

Every case used 8 total episodes, 64 slots per episode, 512 total environment
slots, a per-worker rollout horizon of 16, seed 42, HGNN, forced hover, and one PPO
epoch. Value normalization, value clipping, counterfactual, and lagged residual-Q
were disabled.

## Results

| num_envs | elapsed seconds | environment slots/s | speedup vs process-1 | peak GPU MiB |
|---:|---:|---:|---:|---:|
| 1 | 39.74 | 12.88 | 1.00x | 498 |
| 2 | 30.02 | 17.05 | 1.32x | 510 |
| 4 | 29.63 | 17.28 | 1.34x | 520 |
| 8 | 25.93 | 19.75 | 1.53x | 552 |

All cases processed 512/512 slots and returned zero. There were no tracebacks,
runtime errors, NaNs, infinities, CUDA OOMs, missing worker-health reports, KaHyPar
circuit openings, cleanup failures, or live KaHyPar workers after shutdown.

## Verdict

True concurrent sampling is working and provides measurable throughput improvement
under the recorded server background load. `num_envs=8` is the fastest tested
configuration, while `num_envs=2` captures most of the gain with fewer processes.

This is a throughput gate, not a learning-quality gate. With a fixed per-environment
horizon, increasing `num_envs` increases the PPO batch from
`horizon` to `num_envs * horizon` and reduces the number of optimizer updates per
fixed number of environment slots. A longer learning-equivalence gate is required
before choosing 8 for formal 1000-episode training. Recommended next comparison:
`num_envs=1,2,4,8` with equal total environment slots, fixed evaluation checkpoints,
and explicit update-count/batch-size reporting.
