# Clean Model Boundary

The HyperUAV clean mainline uses a small, explicit model surface. The model
modules in this folder should be interpreted through the clean design document:

- `docs/hyperuav_clean_mainline_design.md`

## Clean Model Modules

### Clean HGNN

- `marl_models/hgnn/clean_incidence.py`

This module contains the clean incidence-matrix-based HGNN forward path. It
consumes task features and incidence/hyperedge structure derived from the
task-only `GraphSnapshot`. It must not consume UAV features, candidate masks,
reward fields, metrics, or profiling data.

### Movement Actor

- `marl_models/mappo/clean_movement_actor.py`

This module builds the clean movement observation and shared movement actor
structure. Boundary action masks are applied to logits, not concatenated into
network input. Movement observations and masks are not part of `GraphSnapshot`.

### Offloading Actor

- `marl_models/mappo/clean_offloading_actor.py`

This module contains the shared candidate scorer and action-record structure for
clean offloading. Candidate features and masks are rollout/action data, not task
graph snapshot data.

### PPO And Critic

- `marl_models/mappo/clean_ppo.py`

This module defines clean rollout records, centralized critic input helpers, GAE
helpers, and PPO loss aggregation rules. Movement and offloading ratios are
computed per action. Offloading slots with `M_t = 0` are excluded from the
offloading loss denominator.

## Legacy Model Paths

The legacy MADDPG, MATD3, MAPPO, MASAC, attention variants, old HGNN scheduler,
and older assignment MAPPO files are retained only as legacy paths. They are not
clean mainline model entrypoints.

Do not connect legacy model modules to the clean mainline unless they are
explicitly migrated to the clean `GraphSnapshot`, clean actor observations,
clean candidate records, and clean PPO/critic contracts.
