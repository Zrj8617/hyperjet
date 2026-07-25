# Multisample Background-Load Gate Report

## Identity

- Branch: `zrj_3multisample`
- Tested commit: `be62b7c5f6f06937442c720eb87da0337a44b005`
- Label: `multisample_background_load`
- Server checkout:
  `/data2/zrj2025/HyperUAV_multisample_gate_be62b7c`
- Artifact root:
  `/data2/zrj2025/HyperUAV_multisample_gate_be62b7c/runs/multisample_throughput_gate/20260725_113525_multisample_background_load`
- Device: physical GPU 5
- Background condition: the 12-cell `ce6075b` matrix remained active on GPU 6.

## Protocol

Every case used:

- 8 total episodes;
- 64 slots per episode;
- 512 total environment slots;
- rollout horizon 16 per environment;
- seed 42;
- HGNN task encoder;
- forced hover;
- one PPO epoch;
- value-target normalization disabled;
- value clip disabled;
- counterfactual and lagged residual-Q disabled.

Cases ran sequentially for `num_envs=1,2,4,8`.

## Correctness gates

- Multi-trajectory GAE was computed separately for each closed environment buffer.
- Synthetic bootstrap isolation passed.
- Python and NumPy RNG lane isolation and outer-state restoration passed.
- Every case processed 512/512 environment slots.
- All cases exited with return code zero.
- No traceback, runtime error, NaN, infinity, or CUDA OOM was found.
- All per-lane KaHyPar circuit, cleanup, and worker-after-close health checks passed.

## Throughput

| num_envs | elapsed seconds | environment slots/s | speedup vs 1 | peak GPU MiB |
|---:|---:|---:|---:|---:|
| 1 | 37.55 | 13.64 | 1.00x | 506 |
| 2 | 38.05 | 13.46 | 0.99x | 514 |
| 4 | 45.25 | 11.31 | 0.83x | 536 |
| 8 | 53.51 | 9.57 | 0.70x | 580 |

## Verdict

The synchronous multi-lane implementation is a valid correctness prototype but is
rejected as a training-speed optimization.

The environment lanes are isolated, and PPO receives correctly separated GAE
trajectories, but lane stepping and model action selection still execute serially in
one Python process. Additional environments therefore add Env/GraphBuilder/KaHyPar
lifecycle overhead without parallelizing the dominant CPU work.

Do not use `num_envs>1` for formal training from this commit.

The next performance implementation should use process-isolated environment workers
with rollout-boundary parameter synchronization, or redesign action preparation for
batched inference plus parallel environment advancement. That architecture requires
a separate design/review gate.
