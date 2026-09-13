# Missing arms + 20-seed reevaluation results (2026-09-07)

## Conclusion

**C1 is the decisive winner.** The EFT/centroid teacher creates a useful policy, and the policy retains that preference after the teacher weight reaches zero. C1 beats A1, A2, random, and EFT-greedy on completion rate, flowtime, and throughput with paired 95% intervals excluding zero. Its three model seeds are unusually consistent.

**A2 is better than A1 on average, but less stable across model seeds.** This is the first comparable answer to the original-reward question: incremental timing credit improves all three system metrics in the 3-seed mean, although model seed 0 reverses the result. The conclusion is promising rather than fully variance-proof with only three training seeds.

**N0-local does not rescue N0.** All N0-local minus N0 intervals cross zero, the per-seed directions are mixed, and both policies remain near random on system outcomes. The poor N0 result is therefore not explained by cross-DAG externality alone; the analytic ΔΦ actor-advantage route should be closed in its current form.

## Unified deterministic evaluation

Each learned configuration is 3 model seeds × 20 paired environment seeds. Values are mean with a 95% bootstrap interval over the 20 paired environment seeds after averaging the three model seeds. Lower flowtime is better.

| Configuration | Completion rate | Flowtime (s) | Throughput | EFT agreement |
|---|---:|---:|---:|---:|
| **C1** | **0.918 [0.897, 0.937]** | **182 [144, 222]** | **0.1286 [0.1162, 0.1412]** | 0.775 |
| Stage-1 EFT-anchored | 0.891 [0.861, 0.920] | 246 [186, 309] | 0.1152 [0.0996, 0.1310] | 0.776 |
| A2 | 0.868 [0.840, 0.895] | 305 [245, 367] | 0.1055 [0.0925, 0.1189] | 0.498 |
| A1 | 0.848 [0.816, 0.879] | 367 [290, 447] | 0.0922 [0.0788, 0.1068] | 0.598 |
| EFT-greedy | 0.841 [0.801, 0.883] | 332 [249, 414] | 0.0971 [0.0798, 0.1160] | 1.000 |
| B2 | 0.813 [0.789, 0.837] | 488 [424, 553] | 0.0700 [0.0629, 0.0776] | 0.467 |
| B2D | 0.785 [0.758, 0.812] | 560 [487, 635] | 0.0626 [0.0557, 0.0699] | 0.420 |
| N0-local | 0.784 [0.757, 0.811] | 578 [507, 650] | 0.0610 [0.0543, 0.0684] | 0.420 |
| N0 | 0.782 [0.757, 0.808] | 573 [503, 643] | 0.0614 [0.0546, 0.0691] | 0.426 |
| Random | 0.770 [0.736, 0.804] | 624 [537, 712] | 0.0586 [0.0507, 0.0674] | — |
| B1 | 0.769 [0.739, 0.798] | 599 [524, 673] | 0.0591 [0.0524, 0.0664] | 0.524 |
| D2 | 0.752 [0.719, 0.785] | 663 [582, 744] | 0.0545 [0.0478, 0.0620] | 0.545 |

## Pre-registered paired contrasts

Differences are candidate minus baseline. Negative flowtime is beneficial.

| Contrast | Δ completion | Δ flowtime (s) | Δ throughput | Interpretation |
|---|---:|---:|---:|---|
| A1 − A2 | −0.0199 [−0.0312, −0.0088] | +62.5 [36.2, 91.0] | −0.01329 [−0.01712, −0.00932] | A2 wins on average |
| C1 − A1 | +0.0702 [0.0521, 0.0886] | −184.8 [−238.2, −135.5] | +0.03638 [0.03005, 0.04287] | C1 decisively wins |
| C1 − A2 | +0.0503 [0.0364, 0.0652] | −122.3 [−160.9, −86.0] | +0.02309 [0.01740, 0.02952] | C1 decisively wins |
| N0-local − N0 | +0.0017 [−0.0080, 0.0108] | +5.0 [−12.5, 22.7] | −0.00039 [−0.00207, 0.00121] | indistinguishable |
| C1 − EFT-greedy | +0.0768 [0.0469, 0.1068] | −149.3 [−213.8, −85.3] | +0.03149 [0.01866, 0.04333] | C1 wins all system metrics |
| C1 − Stage-1 anchor | +0.0270 [0.0102, 0.0461] | −64.1 [−106.1, −26.8] | +0.01341 [0.00639, 0.02118] | C1 is better, but budgets differ |

The Stage-1 comparison is a reference comparison, not a clean treatment contrast: Stage-1 trained for 100 episodes, while C1 trained for 500.

## C1 teacher-retention result

The criterion was whether normalized entropy returned to approximately 0.999 after teacher removal.

- First 100 updates: 0.996–0.997 across model seeds.
- Pre-anneal phase mean: 0.535; entropy falls progressively while the teacher is fully active.
- Annealing phase mean: 0.245.
- Post-anneal phase mean: 0.317.
- Final 20% means by seed: 0.351, 0.379, 0.332.
- Final-update values: 0.306, 0.354, 0.384.

There is a modest rebound after teacher removal, but nothing close to 0.999. The learned preference is retained. Evaluation stability supports the same conclusion: C1 completion rates are 0.920/0.916/0.917, flowtimes 186/184/177 s, and throughputs 0.1291/0.1277/0.1291 across the three model seeds.

## What the training diagnostics mean

- C1 late normalized entropy is 0.354, but N0, N0-local, and B2D also reach low entropy (approximately 0.16–0.17) while producing near-random outcomes. Entropy collapse is not evidence of learning by itself.
- A1 and A2 remain near maximum entropy (approximately 0.999), yet deterministic evaluation is materially better than random. Small logit differences can therefore still induce a useful masked-argmax policy.
- Late critic EV is only 0.010 for C1, versus 0.15–0.30 for several worse B/D arms. Critic EV is not ranking policy quality here, consistent with the existing charter warning.
- Episode rewards are only comparable within identical reward definitions. A1 and C1 share the original reward: their late means are about 12.6 versus 1682.1, agreeing with the large evaluation advantage of C1. A2/B/D reward magnitudes must not be compared directly with A1/C1.

## Decision

1. Accept C1 as the current strongest MLP learnability result and the only reward-redesign arm that clearly and stably exceeds both heuristic reference lines.
2. Record A2 as a positive but training-seed-sensitive improvement over A1; more training seeds would be needed before making a tight effect-size claim.
3. Close both global and local analytic ΔΦ actor-credit variants in their current form.
4. Do not select arms from entropy or critic EV alone; deterministic 20-seed system metrics remain the decision criterion.

## Provenance

- Raw unified JSON: `/data2/zrj2025/uav-results/audits/20260907_missing_arms_unified_analysis.json`
- Old reevaluation manifest: `/data2/zrj2025/uav-results/audits/20260907_old_reevaluation_manifest.json`
- Production manifest: `/data2/zrj2025/uav-results/audits/20260907_missing_arms_launch_manifest.json`
- Analysis script: `scripts/summarize_missing_arms_and_reeval.py`
- HEAD: `644227bb90cd5eb87b89e8cafb66af305f56bd4d`; dirty worktree (58 local entries at analysis time); analysis script object `96344bae19c03ec6394ac0e676461a3719f7d388`.
