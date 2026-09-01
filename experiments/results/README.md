# HyperUAV experiment results archive

This directory contains compact, reviewable results from the R1/R2 and
offloading-credit experiments. Large checkpoints, TensorBoard event files,
per-decision traces, and repeated source snapshots remain in the server-side
raw archive and are intentionally excluded from Git history.

## Layout

- `r1/`: R1-A native workload feedback, R1-A2 strict-CRN control, and R1-B reward alignment.
- `r2/`: reward sweep, bundle summary, initialization equality, and training-curve audit.
- `bridge/`: old-convergence-recipe EFT single-variable bridge experiment.
- `decision_gae_v1/`: paired shared-slot GAE versus Decision-GAE-v1 results.
- `decision_gae_v2/`: three-seed stabilized Decision-GAE critic results.
- `decision_q_v1/`: three-seed environment-return Decision-Q results and checkpoint evaluations.
- `decision_q_v2/`: three-seed target-spread-scaled Decision-Q training results.
- `audits/`: Decision-Q ranking, state-aliasing, and root-cause summaries.

## Code identity

- R2 bundle baseline: `8070bc4fdd12ee4a513216bcd1bed3a87f6e2bed`
- Decision-GAE-v2: `a7d6bb0645b4c37b7b1e6eacac561ef369f39429`
- Decision-Q-v1: `6eaa0de2a81ffad928d3414205380b94dd4b7565`
- Decision-Q-v2 training code: `a9501872a450d937786de1ecb88b969dbbf0c1ab`
- Ranking-audit script commit: `10eb3cad8e1a6d9ceae6c6427eec8e2a14978b26`

The complete raw archive is retained on the experiment server under
`/data2/zrj2025/uav-results`.
