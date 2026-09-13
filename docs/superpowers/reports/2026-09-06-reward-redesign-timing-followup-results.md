# Reward redesign forecast timing follow-up — 2026-09-06

> Superseded gate decision: the user and Claude later raised the threshold to
> 30%. After the two approved optimizations, the rerun measured 17.30% and the
> experiment proceeded to production. See `2026-09-06-reward-redesign-production-launch-results.md`.

## Outcome

**STOP after profiling.** With forecast enabled in a real B2 training run, 20 PPO updates spent **27.564 s** in forecast out of **136.786 s** total training wall time, so `forecast / total training wall = 20.15%`. This exceeds the revised 10% gate.

Per the registered branch, the formal 20-seed full-episode calibration was not run. No seven-arm smoke, production training, memory probe, or background process was started.

## Revised timing gate

The run used the actual B2 arm, one synchronous environment, MLP task encoder, seed 0, 128-slot rollout horizon, one PPO epoch, and GPU forward/backward. It completed 20 updates and 2,500 environment slots. Timing was accumulated from the same rollout records consumed by each update; total wall time surrounds the complete `run_training` call and therefore includes collection and PPO update work.

| Measurement | Value |
|---|---:|
| completed PPO updates | 20 |
| environment slots | 2,500 |
| forecast wall | 27.5636 s |
| collection wall | 43.3802 s |
| total training wall, including updates | 136.7855 s |
| forecast / collection | 63.54% |
| **forecast / total training wall** | **20.15% — FAIL** |
| revised threshold | <= 10% |

The previous `forecast / collection` denominator was indeed too strict as a training-overhead estimate, but correcting it does not change the decision: the exact implementation is still about 2.02 times the allowed fraction.

## Profile

Profile protocol: CPU-only, fixed EFT-greedy policy, forced hover, seed 0, one complete 500-slot episode, plus the calibration script's 100-slot gate-OFF and 100-slot gate-ON equality check. Python's built-in deterministic profiler recorded 24,542,827 calls over 20.247 s. The diagnostic run itself measured 10.059 s of forecast wall time and made 600 `ForecastContext` constructions and 3,108 reservation-delta forecasts.

The two principal forecast kernels have nearly equal cumulative cost under the profiler:

| Forecast component | Calls | cProfile cumulative | Interpretation |
|---|---:|---:|---|
| `ForecastContext._build_arrays` | 600 | 5.647 s | arrays and transfer matrices are rebuilt once per slot |
| `ForecastContext._forecast_dag` | 64,686 | 5.430 s | all active DAGs are recomputed after construction and after every reservation |
| `_tx_vector` | 103,566 | 4.801 s | 85.0% of `_build_arrays`; scalar communication calls dominate construction |
| `numpy.fromiter` inside `_tx_vector` | 103,566 | 4.425 s | wraps 2,127,375 scalar communication-model evaluations |
| `_forecast_dag` self time | 64,686 | 2.873 s | 52.9% of `_forecast_dag`; primarily per-task and per-parent Python control flow |
| `numpy.min` | 355,621 | 1.322 s | mainly the `U x U` parent-arrival reductions, about 24.3% of `_forecast_dag` |

The cumulative rows overlap: `_tx_vector` is inside `_build_arrays`, and `_forecast_dag` is called by both context construction and reservation deltas. They must not be added as independent wall times. The profile nevertheless identifies two concrete bottlenecks:

1. Transfer data are cached only for the lifetime of one slot context. Rebuilding the context invokes the scalar communication model millions of times across slots.
2. `_forecast_dag` iterates in Python over every active DAG, task, and `parents.tolist()` entry. The inner parent transfer is array-based, but dispatching hundreds of thousands of small NumPy reductions from Python remains expensive.

## Exact optimization plan

No approximate forecast is proposed. The next implementation should retain the same optimistic topological DP, fixed scheduled-task finish times, private compute-only queue clocks, and actor reservation semantics.

1. Split immutable DAG topology from slot-varying state. Cache task order, predecessor indices/masks, sink indices, task compute durations, edge payloads, and bandwidth keys across slots. Rebuild only active/fixed-state views.
2. Vectorize the communication formula over distance arrays and cache per-slot transfer tensors by `(payload, bandwidth)` rather than invoking `clean_transmission_time_seconds` once per matrix cell. This is algebraically identical and removes the 2.13 million scalar calls observed in the profile.
3. Group active DAGs by task count/topology shape. Store padded parent indices plus a parent-valid mask, gather parent finishes in one array operation, reduce source UAV with `min`, then reduce valid parents with `max`. This removes `for pidx in parents.tolist()` while preserving `O(E*U^2)`; it does not enumerate parent-location combinations.
4. Maintain `last_times` and per-DAG changes as aligned NumPy arrays instead of rebuilding `before`, `after`, and `changes` dictionaries after every decision. Convert to keyed data only at the external diagnostic boundary.
5. After each exact change, rerun the same gate: 20 real-training updates first. Only if `forecast / total training wall <= 10%` should the 20-seed full-episode distribution probe be resumed.

## Artifacts and version record

- Training timing JSON: `/data2/zrj2025/uav-results/audits/reward-redesign-training-timing-gate-20260906-v2/result.json`
- Training run directory: `/data2/zrj2025/uav-results/audits/reward-redesign-training-timing-gate-20260906-v2/train/20260906_173939_20260906_forecast_training_timing_seed0_seed0`
- Raw cProfile data: `/data2/zrj2025/uav-results/audits/reward-redesign-forecast-profile-20260906-v1/forecast.prof`
- Profile-side seed-0 JSON: `/data2/zrj2025/uav-results/audits/reward-redesign-forecast-profile-20260906-v1/calibration-seed0.json`
- Machine-readable repository result: `docs/superpowers/reports/2026-09-06-reward-redesign-timing-followup-result.json`
- Executed worktree: `/data2/zrj2025/HyperUAV-reward-redesign`
- Executed HEAD: `644227bb90cd5eb87b89e8cafb66af305f56bd4d`

Executed dirty files:

```text
 M environment/assignment.py
 M environment/env.py
 M marl_models/mappo/clean_offloading_actor.py
 M marl_models/mappo/clean_slot_orchestrator.py
 M marl_models/mappo/clean_trainer.py
 M scripts/train_clean_mainline.py
?? environment/forecast.py
?? environment/reward_redesign.py
?? scripts/run_forecast_training_timing_gate.py
?? scripts/run_reward_forecast_calibration.py
?? scripts/run_reward_redesign_arm.py
```

Executed Git object identifiers:

```text
run_forecast_training_timing_gate.py  cc2a65635503922becb179bc1c74425ac8423753
run_reward_forecast_calibration.py    e2332afee417154abe9a3a25f2bfa0e70760b546
train_clean_mainline.py               1df8731efb383451a91b0cc7a795e3a1794e5077
environment/forecast.py               be8ee622a6f2224f49bbfe37a3175b7d7278d335
```

## Boundary

This follow-up supports only the runtime-gate conclusion and the profile-guided exact optimization plan. The seed-0 profile-side distribution is not the requested formal distribution and must not be used as the 20-seed calibration result.
