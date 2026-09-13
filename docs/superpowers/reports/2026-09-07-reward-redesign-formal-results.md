# Reward redesign formal results — 2026-09-07

## Outcome

All **21/21** production runs completed: 7 arms × 3 seeds × 500 episodes, with 250,000 slots and 2,000 PPO updates per run. All original PIDs have exited, every run has a completed `result.json`, all 21 runs used the same recorded HEAD/runner/trainer objects, and the actual TensorBoard event files contain exactly the registered 12 scalar tags.

**Experimental verdict: descriptive result only; do not accept as the preregistered formal comparison.** Two protocol deviations prevent a decisive arm ranking:

1. Each checkpoint received only one forced-hover evaluation episode at its matching training seed. Stage-1 used 20 environment seeds per checkpoint, and the spec additionally called for a common fixed offer tape and admission/conditional/end-to-end rates. Those were not produced here.
2. B2D did not use the registered variable-discount decision reward/GAE plus microstate critic. It used the same batch-standardized analytic `-delta_phi/500` actor advantage mechanism as N0, on top of the B2 slot reward. B2D must therefore be interpreted as **B2 + analytic delta-phi advantage**, not as the intended decision-reward arm.

The completed checkpoints remain usable. The six arms not affected by the B2D implementation mismatch do not need retraining merely to correct the evaluation sample size.

## Integrity and actual protocol

- Formal result root: `/data2/zrj2025/uav-results/audits`
- Runs: `20260906_<arm>_seed{0,1,2}` for N0, A2, B1, B2, B2D, D1, D2.
- Training: 500 × 500 slots, MLP encoder, learned movement, rollout horizon 128, PPO epochs 1, gamma 0.99, GAE lambda 0.95.
- Evaluation actually run: one 500-slot deterministic actor-argmax episode, forced hover, no drain, training RNG prelude, environment seed equal to model seed.
- Actual-parameter comparison: within each seed, all fields match across arms except the registered arm, run name, and output path.
- Finite/shape checks: 21 completed results, 21 × 2,000 training metric rows, 21 completed evaluations; no missing cell.
- Version identity: one HEAD/runner/trainer combination across all 21 runs.
- TensorBoard: external event-file readback passed for 21/21; exactly 12 scalar tags, no missing or extra tag.

## Descriptive forced-hover evaluation

Mean and sample standard deviation across the three matching-seed evaluation episodes. Higher completion/throughput and lower flowtime/energy are favorable.

| Arm | Completion | Avg DAG flowtime (s) | Throughput | Energy/completed DAG (J) | EFT agreement | Eval normalized entropy |
|---|---:|---:|---:|---:|---:|---:|
| N0 | 0.7560 ± 0.0482 | 682.3 ± 157.4 | 0.05227 | 201.93 | 0.5338 | 0.1516 |
| **A2** | **0.8228 ± 0.0540** | **416.3 ± 155.9** | **0.07640** | **181.27** | 0.5708 | 0.9992 |
| B1 | 0.7379 ± 0.0539 | 665.3 ± 144.6 | 0.05173 | 195.41 | 0.5862 | 0.9863 |
| B2 | 0.7681 ± 0.0936 | 607.1 ± 213.4 | 0.05613 | 198.11 | 0.5456 | 0.8451 |
| B2D | 0.7497 ± 0.0379 | 655.0 ± 131.1 | 0.05280 | 205.37 | 0.5110 | 0.1257 |
| D1 | 0.7681 ± 0.0936 | 607.1 ± 213.4 | 0.05613 | 198.11 | 0.5456 | 0.8451 |
| D2 | 0.6861 ± 0.0720 | 761.9 ± 144.4 | 0.04360 | 219.54 | 0.6312 | 0.9858 |

D1 and B2 are exactly equal seed by seed on every behavioral result. This is expected under the user's later eta=1 override: both labels execute the same global incremental treatment. Their only measured difference is wall-clock noise.

### Relation to prior reference bands

The prior 20-seed reference means were random = `(0.7702, 624.2 s, 0.05858)`, EFT-greedy = `(0.8413, 331.7 s, 0.09714)`, and Stage-1 EFT-anchor = `(0.8911, 246.5 s, 0.11522)` for completion, flowtime, and throughput.

- A2 is the only current arm whose three-point means improve over the random reference on all three measures: completion +6.8%, flowtime −33.3%, throughput +30.4%. It remains below EFT-greedy by 2.2%, 25.5%, and 21.4% respectively.
- B2/D1 are approximately random-level: completion −0.3%, flowtime 2.7% better, throughput 4.2% worse than the random 20-seed mean.
- N0, B1, B2D, and D2 are worse than the random reference on all three mean measures.
- No arm reaches EFT-greedy, much less the prior Stage-1 EFT-anchor result, on the three joint system measures.

These are contextual comparisons between three evaluation cells here and 20/60-cell historical aggregates, not paired formal tests.

## Registered within-experiment contrasts

Paired by model/environment seed. The 95% intervals are unadjusted paired-t intervals with only 2 degrees of freedom; they are shown to make the uncertainty explicit, not to claim adequate power.

### B2 versus B1: complete incremental ledger versus completion settlement

| Metric | Mean B2−B1 | Beneficial relative change | Favorable seeds | Paired 95% interval |
|---|---:|---:|---:|---:|
| Completion | +0.0302 | +4.1% | 2/3 | [−0.0972, +0.1576] |
| Flowtime | −58.2 s | +8.8% | 2/3 | [−242.4, +125.9] s |
| Throughput | +0.00440 | +8.5% | 2/3 | [−0.00993, +0.01873] |
| Energy/completed DAG | +2.70 J | −1.4% | 1/3 | [−13.98, +19.38] J |

Directionally, the complete incremental ledger is mildly better on the three delay/system measures, but the seed variance is larger than the observed effect. This contrast is **inconclusive**, not a pass.

### B2D versus B2: analytic delta-phi actor advantage

B2D lowered late training normalized entropy from 0.931 to 0.159 and evaluation entropy from 0.845 to 0.126, so the analytic signal strongly changes the actor. Nevertheless, B2D worsened mean completion by 2.4%, flowtime by 7.9%, throughput by 5.9%, and energy/DAG by 3.7%; only 1/3 seeds favored B2D on each metric, and every interval crosses zero.

This supports a narrow conclusion: **analytic global delta-phi is learnable enough to collapse entropy, but lower entropy did not produce better system behavior.** It does not evaluate the registered B2D decision-GAE mechanism because that mechanism was not implemented.

### D2 versus B1: movement coefficient 0.04 versus 0.10

- Late-episode movement rate increased from **0.55%** to **35.51%**, so lambda_move=0.04 reaches the intended approximately 0.30 mobility regime while 0.10 nearly freezes movement.
- D2 was worse in 3/3 seeds on completion, flowtime, throughput, and energy/DAG.
- Completion delta was −0.0518 with paired 95% interval [−0.0968, −0.0069]. Energy/DAG increased 24.14 J with interval [+9.96, +38.31]. Flowtime and throughput intervals still cross zero.

Thus 0.04 fixes the movement-rate calibration but does not improve the current forced-hover offloading evaluation; training under more movement may itself change the state distribution seen by the offloading actor.

### A2 versus N0: descriptive only

A2 improves all four displayed system measures in 3/3 seeds, including flowtime by 266.0 s and throughput by 46.2% on the means. However this is not a clean single-variable comparison: N0 uses analytic delta-phi actor advantage, while A2 uses slot GAE and changes the time-reward decomposition. Moreover A2 offloading entropy remains 0.9992. The result cannot establish that A2 learned a useful offloading ranking.

## Learning diagnostics

Tail statistics use the last 20% of 2,000 updates or the last 20% of complete episodes.

| Arm | Early → late normalized offload entropy | Tail critic EV | Tail movement rate | Forecast/run wall |
|---|---:|---:|---:|---:|
| N0 | 0.709 → 0.162 | 0.059 | 74.67% | 15.19% |
| A2 | 1.000 → 0.999 | 0.077 | 78.51% | 11.82% |
| B1 | 1.000 → 0.996 | 0.264 | 0.55% | 0% |
| B2 | 1.000 → 0.931 | 0.153 | 1.00% | 15.26% |
| B2D | 0.876 → 0.159 | 0.188 | 0.64% | 14.74% |
| D1 | 1.000 → 0.931 | 0.153 | 1.00% | 15.35% |
| D2 | 1.000 → 0.995 | 0.301 | 35.51% | 0% |

- Entropy reduction is not sufficient evidence of learning the desired policy: N0 and B2D have the lowest entropy but poor system metrics.
- Critic EV is also not an action-ranking measure. B1/D2 have the highest EV while producing weak forced-hover results.
- Across the 15 forecast-enabled runs, forecast/run-wall fraction ranges **11.69%–15.67%** with mean **14.47%**, safely below the revised 30% gate.
- Recorded `train/approx_kl` is approximately zero for every arm. With one PPO epoch, the implementation computes it from the pre-update distribution, so it is not a useful convergence diagnostic in these runs.

## TensorBoard semantic caveat

The event files contain exactly the requested tag names, but three tags do not contain the aggregate implied by their labels:

- `train/episode_reward` stores the PPO rollout reward total, not the completed episode reward.
- `move/move_rate_m` stores the last slot represented by an update row, not the episode movement rate.
- `forecast/wall_time_frac` stores that row's slot-level forecast/collection fraction, not forecast/total training wall.

The raw JSONL allowed the episode/tail and cumulative ratios above to be reconstructed correctly. The TensorBoard curves should not be interpreted using the stronger aggregate meanings.

## Verdict and next decision boundary

The strongest supported findings are:

1. The exact forecast implementation remains within the approved runtime budget at full scale.
2. Complete incremental timing (B2) shows only a small, high-variance directional gain over completion settlement (B1).
3. Direct analytic delta-phi credit strongly changes policy entropy but does not recover EFT-greedy-level system performance; this is evidence against using it as a drop-in replacement for EFT regret.
4. Lambda_move=0.10 nearly suppresses all movement; 0.04 restores the target movement rate but worsens the current system results.
5. No executed reward arm provides evidence comparable to the previously successful Stage-1 EFT-anchor.

Before any paper-level arm selection, the existing checkpoints need the registered multi-seed evaluation protocol. B2D additionally requires either relabeling as “B2 + analytic delta-phi advantage” or a corrected implementation and retraining. This report does not authorize either action or advance to HGNN-versus-MLP conclusions.

## Artifacts and version record

- Server formal runs: `/data2/zrj2025/uav-results/audits/20260906_<arm>_seed{0,1,2}`
- Server launch manifest: `/data2/zrj2025/uav-results/audits/reward-redesign-production-launch-20260906.json`
- Repository machine summary: `docs/superpowers/reports/2026-09-07-reward-redesign-formal-result.json`
- Reproducible summarizer: `scripts/summarize_reward_redesign_results.py`
- TensorBoard verifier: `scripts/check_reward_redesign_tensorboard.py`

Executed server version:

```text
HEAD                                644227bb90cd5eb87b89e8cafb66af305f56bd4d
launch_reward_redesign_production   dad6b11641af6b7cc44ced4e6f044c0c8c0098e4
run_reward_redesign_arm             cdb72c9495415f947224b2e32fca0d16a6fa48e6
train_clean_mainline                1df8731efb383451a91b0cc7a795e3a1794e5077
environment/forecast                4f91002bd4ba3bb0a643e157587e717a4f64f634
environment/reward_redesign         de5873bba7865cbf2f570c4dbacea54cf03c8ce7
summarize_reward_redesign_results   9b331f93cb1437d95c20a46f12afb80c2ca9809d
check_reward_redesign_tensorboard   0d4a27d0c7b6b8f1643ac8cc2eba06476e0b59db
```

All 21 result files record the same HEAD and runner/trainer objects. The complete 12-line launch dirty-file list is preserved in the server manifest and the production-launch report.
