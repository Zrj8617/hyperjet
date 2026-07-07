# HyperUAV Clean Mainline

This repository is currently organized around the HyperUAV clean mainline.
The authoritative design document is:

- `docs/hyperuav_clean_mainline_design.md`

If older code, old README content, or older specs disagree with that document,
the clean mainline design document wins.

## Current Status

The T1-T10 clean skeleton has been implemented and committed. The implemented
scope includes:

- clean scenario parameters and clean-only configuration;
- clean task lifecycle and slot/service-position semantics;
- ready-set freezing, assignment buffer, and temporary reservation boundaries;
- task-only `GraphSnapshot`, clean hyperedge/incidence representation, and
  minimal incidence-matrix HGNN module;
- clean movement actor, offloading actor structures, centralized critic/PPO
  record/loss helpers;
- clean reward accounting, metrics tracking, and end-to-end smoke coverage.

The non-torch smoke path passes in the current local bundled Python runtime.
That runtime does not have `torch`, so local training cannot start here yet.
Before training, switch to a torch-enabled Python environment and rerun the
model smoke checks that are skipped without torch.

## Recommended Smoke Checks

Use the same Python environment you plan to train with.

```bash
python scripts/smoke_clean_env.py
python scripts/smoke_clean_execution.py
python scripts/smoke_clean_graph.py
python scripts/smoke_clean_reward_metrics.py
python scripts/smoke_clean_offloading_actor.py
python scripts/smoke_clean_ppo.py
python scripts/smoke_clean_end_to_end.py
```

In a torch-enabled environment, also run:

```bash
python scripts/smoke_clean_hgnn.py
python scripts/smoke_clean_movement_actor.py
python scripts/smoke_clean_offloading_actor.py
python scripts/smoke_clean_ppo.py
python scripts/smoke_clean_end_to_end.py
```

Without torch, the HGNN/actor/critic/PPO forward checks are expected to skip;
that is not a failure of the non-torch environment logic.

## Clean Mainline Entrypoints

The current clean mainline validation entrypoints are the `scripts/smoke_clean_*`
scripts listed above. A clean training entrypoint has not been finalized yet.

The old `main.py`, `train.py`, `tune.py`, legacy MAPPO, legacy HGNN scheduler,
and old experiment scripts are not clean mainline entrypoints. They may remain
in the repository as legacy paths, but they must not be used to validate or run
the clean mainline unless they are explicitly migrated to the clean design.

## Main Clean Modules

- Environment: `environment/env.py`
- Task lifecycle: `environment/dag_tasks.py`
- Executor: `environment/task_execution.py`
- Assignment/reservation helpers: `environment/assignment.py`
- Task graph snapshot: `environment/graph_builder.py`
- Reward and metrics: `environment/metrics.py`
- Clean HGNN: `marl_models/hgnn/clean_incidence.py`
- Movement actor: `marl_models/mappo/clean_movement_actor.py`
- Offloading actor: `marl_models/mappo/clean_offloading_actor.py`
- Centralized critic/PPO helpers: `marl_models/mappo/clean_ppo.py`
