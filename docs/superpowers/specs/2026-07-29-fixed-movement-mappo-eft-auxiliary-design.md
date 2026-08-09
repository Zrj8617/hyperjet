# Fixed-Movement MAPPO + Per-Decision EFT Auxiliary Diagnostic Design

## Status and scope

This design records the user-approved Phase 1 diagnostic on
`zrj_3multisample@5156789`. It is an opt-in diagnostic extension of the current
clean MAPPO path. It does not replace the clean mainline specification and does
not make EFT a formal environment reward.

The diagnostic asks whether current-policy, per-decision, low-variance EFT
feedback can improve the offloading policy when it is trained jointly with the
existing slot-level MAPPO loss.

## Invariants

- Movement is forced to hover and the movement actor receives no optimizer
  gradient.
- The environment reward, slot critic, GAE, PPO ratio, task arrival process,
  transition timing, candidate features/mask, HGNN, and GraphSnapshot are
  unchanged.
- The task encoder is MLP for every treatment.
- No action-value critic, lagged residual-Q model, outcome ledger, or
  retroactive reward reassignment is enabled.
- A/B/C/D use the same interaction, rollout, update, and paired evaluation
  seeds.
- EFT metadata is diagnostic-only rollout history and never enters state,
  reward, critic input, or GraphSnapshot.

## Historical decision record

At decision `k`, `CleanOffloadingActor.act()` already constructs candidate
features, the legal mask, and every candidate's EFT from the same current
`TemporaryReservationState`. Before reserving the sampled action, it will copy
the complete candidate EFT vector into `CleanOffloadingActionRecord`.

The corresponding `CleanOffloadingRolloutRecord` will own immutable historical
copies of:

- candidate UAV ids;
- dynamic UAV features;
- pair features;
- candidate mask;
- candidate EFT values;
- task local index and sequential decision order;
- the original PPO action and old log probability.

The next decision sees the reservation made by the current sampled action, as
it does today. PPO update code must only consume the saved values. It must not
call the environment EFT estimator or advance a reservation.

## Auxiliary objective

For every recorded decision with at least two legal candidates, the updater
reconstructs current candidate logits from the historical graph and dynamic
features. It creates a new masked distribution and resamples an auxiliary
action during every PPO epoch:

```text
r_i = -(EFT_i - min_legal(EFT)) / scale
b(s) = sum_i pi(i | s) r_i
A_aux = stopgrad(r_a - stopgrad(b(s)))
L_EFT = mean(-A_aux * log pi(a | s))
```

The rollout PPO action is used only by the original PPO term. It is never used
as the auxiliary action. EFT, rewards, and the exact baseline are detached.
Illegal actions must never be sampled. Decisions with zero or one legal
candidate are excluded from the auxiliary denominator and reported.

`scale` is the frozen train-split RMS legal-candidate regret definition already
used by the contextual-bandit gate. The formal diagnostic uses the previously
validated value and records its provenance.

## Optimization

The total loss is:

```text
L_total =
    L_move_PPO
  + L_offload_PPO
  + lambda(update) * L_EFT
  + value_coef * L_value
  - movement_entropy_coef * H_move
  - offloading_entropy_coef * H_offload
```

The existing shared optimizer remains the only optimizer. Each PPO epoch has
exactly one `zero_grad`, one `backward`, one global clip, and one `step`.
`L_EFT` can update only the shared MLP task encoder and offloading scorer.
The critic and frozen movement actor receive no direct auxiliary gradient.

The approved schedule is indexed by the outer rollout update:

```text
updates 0..8:  lambda = lambda_0
updates 9..19: linearly decrease from 11/12 lambda_0 to 1/12 lambda_0
update 20:     lambda = 0
updates 21..29 lambda = 0
```

All PPO epochs within one outer update use the same lambda.

## Lambda calibration

An independent seed-42 two-update pilot measures, on the same target parameter
set (MLP encoder plus scorer):

- offloading PPO gradient norm;
- raw EFT auxiliary gradient norm;
- weighted EFT auxiliary gradient norm;
- weighted-EFT / PPO gradient ratio.

`lambda_0` is selected once so weighted EFT is approximately equal to or
slightly stronger than the PPO actor gradient, never tens of times stronger.
It is then frozen for every treatment and seed. Scientific outcomes do not
trigger retuning.

## Treatments and initialization

| Group | Encoder/scorer initialization | PPO | EFT auxiliary |
|---|---|---:|---:|
| A | random seed-specific initialization | yes | no |
| B | matching-seed trained bandit checkpoint | yes | no |
| C | same matching-seed checkpoint as B | yes | decays to zero |
| D | exact same random initialization as A | yes | decays to zero |

Only MLP encoder and scorer weights are loaded from bandit checkpoints, with
strict key and shape checks. Critic and optimizer are freshly initialized.
A/D and B/C initialization hashes are asserted pairwise equal.

## Gate and evaluation

The formal gate runs 30 outer updates for 12 variants
(`A/B/C/D x 42/86/1042`). It records update checkpoints at 0, 1, 5, 10, the
first zero-lambda update, and the final update.

Training diagnostics include selected EFT regret, greedy agreement, margin
accuracy, entropy, PPO/EFT losses, lambda, critic explained variance,
module/target gradient norms, illegal auxiliary samples, and arrival funnel
metrics.

Short closed-loop evaluation uses 20--30 paired fixed scenarios and real
sequential reservations for:

- initial policy;
- masked random;
- trained treatment;
- greedy EFT.

Frozen imitation samples are not chained into closed-loop evaluation.

## Interpretation boundary

A successful diagnostic supports only:

> Per-decision, immediate, low-variance feedback can drive the offloading actor;
> the formal MAPPO bottleneck is more likely in delayed credit, slot-shared
> advantage, critic, GAE, or reward scale.

It does not prove delayed reward is the unique cause, and EFT does not become
the formal baseline or final reward.
