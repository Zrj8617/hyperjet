# Reward redesign experiment results — 2026-09-06

> Follow-up: the revised real-training denominator was measured in
> `2026-09-06-reward-redesign-timing-followup-results.md`. It also failed
> (`forecast / total training wall = 20.15%`) and therefore triggered an exact-implementation profile.

## Outcome

**STOP at the forecast-cost checkpoint.** The preflight fixed-policy calibration measured forecast time at **48.76%** of collection time, above the frozen **10%** threshold. The formal 20-seed calibration, seven-arm smoke, and 21 formal runs were not started.

This is an early-stop preflight result, not the planned 20-seed M2 estimate. It is enough to reject the executed exact-global implementation on the registered cost gate, but it is not a 20-seed distributional result.

## Implemented before the checkpoint

- P1 adds `estimated_compute_finish_time` and `estimated_return_finish_time` while retaining the legacy return-inclusive `estimated_finish_time`. Actor reservation and environment behavior remain on the legacy value.
- P2 adds a separate compute-only forecast queue initialized from executor schedules. It caches topology and communication matrices and uses NumPy topological DP with vectorized parent-UAV to child-UAV minimization; sink forecasting stops at compute finish.
- Forecast collection is gated. Gate-OFF and gate-ON fixed-policy assignment traces and final metrics matched exactly.
- The seven arm definitions, reward ledger, analytic `ΔΦ` actor-credit switch, run wrapper, forced-hover evaluation call, and exactly 12 TensorBoard tags are implemented, but were not exercised as an experiment because the checkpoint stopped the workflow.

## Checkpoint measurements

Protocol: no training, `eft_greedy`, forced hover, seed 0, 20 slots, 91 offloading decisions.

| Measurement | Value |
|---|---:|
| zero fraction (`|ΔΦ| <= 1e-9`) | 20.88% |
| `|ΔΦ|` median | 1.423 s |
| `ΔΦ` p25 / p50 / p75 | 0.025 / 1.423 / 5.573 s |
| `ΔΦ` p90 / p95 / p99 | 8.203 / 10.319 / 34.185 s |
| mean positive `ΔΦ` | 4.746 s |
| target-DAG mean delta | 1.139 s |
| cross-DAG mean delta | 2.617 s |
| affected DAG count mean | 2.198 |
| forecast wall time | 0.07768 s |
| collection wall time | 0.15929 s |
| **forecast / collection** | **48.76% — STOP** |

The scale was therefore not advanced to `F_ref=500` for training.

## Verification and version record

- Existing actor smoke: PASS on the server.
- Python compilation of all changed implementation and runner files: PASS on the server.
- Gate-OFF fixed-policy trace comparison: PASS, exact equality.
- Server execution worktree: `/data2/zrj2025/HyperUAV-reward-redesign`.
- Server raw JSON: `/data2/zrj2025/uav-results/audits/reward-redesign-20260906-calibration-smoke.json`.
- Machine-readable repository result: `docs/superpowers/reports/2026-09-06-reward-redesign-experiment-result.json`.
- Executed HEAD: `644227bb90cd5eb87b89e8cafb66af305f56bd4d` plus the dirty files listed below.

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
?? scripts/run_reward_forecast_calibration.py
?? scripts/run_reward_redesign_arm.py
```

Executed Git object identifiers:

```text
run_reward_forecast_calibration.py  e2332afee417154abe9a3a25f2bfa0e70760b546
run_reward_redesign_arm.py          cdb72c9495415f947224b2e32fca0d16a6fa48e6
train_clean_mainline.py             b97283bb094f2d7f8835031e9e4422636110f8bd
environment/forecast.py             be8ee622a6f2224f49bbfe37a3175b7d7278d335
```

## Boundary

No claim is made about any reward arm, convergence, or policy quality. No GPU-memory probe, ETA probe, or background formal run was started. The only experimental conclusion is that the executed exact-global forecast implementation failed the registered wall-time gate.
