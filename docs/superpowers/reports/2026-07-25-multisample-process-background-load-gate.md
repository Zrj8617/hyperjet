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

## Extended scale gate: 10/15/20

An extended gate was run from commit
`87e2f7a1c2ffe601857e5dd56971219ab499067a` with same-batch controls at
`num_envs=1` and `8`.

Artifact root:

`/data2/zrj2025/HyperUAV_multisample_scale_87e2f7a/runs/multisample_process_scale_gate/20260725_122024_multisample_process_background_load`

Every case used 20 total episodes, 64 slots per episode, and 1280 total
environment slots. All other model and PPO controls matched the earlier process
gate.

| num_envs | elapsed seconds | environment slots/s | speedup vs process-1 | peak GPU MiB |
|---:|---:|---:|---:|---:|
| 1 | 92.25 | 13.87 | 1.00x | 500 |
| 8 | 42.12 | 30.39 | 2.19x | 554 |
| 10 | 38.88 | 32.92 | 2.37x | 570 |
| 15 | 38.37 | 33.36 | 2.40x | 614 |
| 20 | 38.89 | 32.91 | 2.37x | 652 |

All five cases processed 1280/1280 slots and returned zero. No traceback, runtime
error, NaN, infinity, CUDA OOM, missing worker health, KaHyPar circuit opening,
cleanup failure, or surviving worker was observed.

Raw throughput peaks at `num_envs=15`, but scaling is effectively flat from 10 to
20. Compared with 10, 15 uses 50% more environment processes for only about 1.3%
more throughput; 20 is slightly slower than 15. Therefore:

- `15` is the raw-throughput winner in this gate;
- `10` is the more resource-efficient candidate;
- `20` should be rejected for the current server/load;
- none should enter formal learning runs until the larger effective PPO batch and
  lower optimizer-update frequency pass a learning-quality gate.
