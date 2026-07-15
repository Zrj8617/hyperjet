# Lagged DAG-Outcome Residual-Q Short-Gate Results

Date: 2026-07-15

Branch: `zrj_3_static_ue`

Implementation commit: `a08486f18f2816ec8f1e512df11b10fb5aa5d782`

Decision: **VETO v2; do not launch formal 1000-episode training**

## 1. Audited artifacts

Training root:

`/data2/zrj2025/HyperUAV/runs/phase5_lagged_q_short_gate_20260715_122828`

Training log root:

`/data2/zrj2025/HyperUAV/logs/phase5_lagged_q_short_gate_20260715_122828`

Deterministic evaluation root:

`/data2/zrj2025/HyperUAV/runs/phase5_lagged_q_short_gate_eval_20260715_130448`

Evaluation log root:

`/data2/zrj2025/HyperUAV/logs/phase5_lagged_q_short_gate_eval_20260715_130448`

The training matrix contains seeds 42, 86, and 1042. Every run completed 100
episodes x 200 slots, produced 200 update rows, 100 terminal episode rows, and
checkpoints at episodes 20, 40, 60, 80, and 100 plus `latest.pt`.

The evaluation matrix contains 13 completed runs and 65 episode rows:

- baseline and v2 episode-100 checkpoints for three model seeds;
- five common environment seeds, 4242 through 4246;
- normal learned movement and forced-hover isolation;
- one five-scene forced-hover `random_hash` reference.

All training and evaluation values were finite. Invalid assignments, KaHyPar
degraded slots, circuit-open events, and residual train/eval processes were all
zero. Every evaluation drain ended with `all_completed`.

## 2. Training integrity and Q mechanism

The last-20-update diagnostics passed the mechanism requirements:

| seed | label coverage | weighted censor fraction | median Q EV | median legal spread | median correction std | clamp fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.000 | 0.132 | 0.233 | 0.00794 | 0.00674 | 0.000 |
| 86 | 1.000 | 0.121 | 0.170 | 0.00871 | 0.00630 | 0.000 |
| 1042 | 1.000 | 0.108 | 0.190 | 0.01209 | 0.00772 | 0.000 |

For all three seeds, the direct lagged-Q loss gradient to HGNN, movement actor,
offloading actor, and centralized critic was exactly zero. The historical Q
sample type contains no PPO log probability, and current-rollout corrections
were frozen before PPO epochs. The on-policy boundary therefore passed.

The movement-collapse guard also passed:

| seed | baseline last-20 hover | v2 hover | baseline displacement | v2 displacement |
|---:|---:|---:|---:|---:|
| 42 | 0.509 | 0.313 | 36.84 m | 51.54 m |
| 86 | 0.234 | 0.212 | 57.43 m | 59.09 m |
| 1042 | 0.562 | 0.185 | 32.84 m | 61.12 m |

Thus the v2 failure is not explained by a new hover collapse.

## 3. Deterministic aggregate results

### Normal learned movement

| seed | baseline completion | v2 completion | delta | baseline flowtime | v2 flowtime | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6821 | 0.6914 | +0.0093 | 308.83 s | 315.03 s | +2.0% |
| 86 | 0.6188 | 0.5915 | -0.0272 | 450.36 s | 404.64 s | -10.2% |
| 1042 | 0.5556 | 0.6710 | +0.1154 | 496.05 s | 331.76 s | -33.1% |

Seeds 86 and 1042 satisfy the normal-evaluation improvement rule. Seed 42 does
not materially regress, but also does not reach an improvement threshold.

### Forced-hover offloading isolation

| seed | baseline completion | v2 completion | delta | baseline flowtime | v2 flowtime | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8225 | 0.6691 | **-0.1534** | 182.87 s | 353.09 s | **+93.1%** |
| 86 | 0.6381 | 0.6040 | **-0.0342** | 429.42 s | 430.08 s | +0.2% |
| 1042 | 0.4884 | 0.5575 | +0.0691 | 590.58 s | 511.36 s | -13.4% |

The current forced-hover random reference is completion `0.5423` and weighted
flowtime `508.97 s`.

Seed 1042 improves completion in all five paired scenes, but its aggregate
flowtime remains slightly worse than random (`511.36 s` versus `508.97 s`). More
importantly, v2 destroys much of seed 42's previously strong isolated offloading
ranking and crosses both material-regression boundaries by a wide margin.

## 4. Pre-registered gate verdict

| criterion | result | evidence |
|---|---|---|
| 1. 1042 forced-hover improves in at least 4/5 scenes | PASS | completion improves in 5/5 scenes |
| 2. 1042 forced-hover beats random on completion and flowtime | **FAIL** | completion wins, flowtime is 0.47% worse |
| 3. At least two normal-eval seed improvements | PASS | seeds 86 and 1042 |
| 4. No material forced-hover regression for 42 or 86 | **FAIL** | seed42: -0.153 completion and +93.1% flowtime; seed86: -0.034 completion |
| 5. No new movement collapse | PASS | all three hover/displacement comparisons pass |
| 6. Label coverage and weighted censoring | PASS | coverage 1.0; weighted censor fraction 0.108-0.132 |
| 7. Q information and correction scale | PASS | all EV/spread/correction thresholds pass; zero clamp saturation |
| 8. Integrity/on-policy/provenance | PASS | finite, valid, no stale ratio path, no degraded KaHyPar |

Because every criterion is mandatory, failures in criteria 2 and 4 veto v2.
The criterion-4 failure is large, so the veto does not depend on the narrowly
missed random-flowtime comparison.

## 5. Interpretation

Lagged residual Q learned a real delayed-outcome signal and repaired the weak
seed-1042 offloading ranking. It is nevertheless not seed-stable: the same
correction severely damages seed 42, whose baseline offloader was already good.

The most likely remaining design risk is target confounding. Every selected task
action receives a residual derived from the final DAG return time. That residual
contains later decisions and downstream congestion shared by many actions in the
same DAG. The Q regressor can predict this observational residual (positive EV)
without learning a correction that is causally beneficial for each action. The
opposite effects on seeds 42 and 1042 are consistent with that failure mode.

This is a diagnosis, not proof of a unique cause. The next investigation should
measure, per seed and action, whether the learned correction's sign agrees with
realized candidate regret and whether Q-driven PPO updates systematically move
the actor away from the already-good baseline ranking. A future repair needs a
more local or causally isolated delayed target, not a coefficient sweep of this
v2 arm.

## 6. Formal-run boundary

No v2 coefficient tuning, entropy/reward/detach/arrival experiment, or formal
1000-episode run is authorized from this result. In accordance with the design,
the branch terminates with this evidence and redirects to delayed-target causal
attribution. No formal v2 training task was launched.
