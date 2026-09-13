# Missing arms + 20-seed reevaluation — production handoff (2026-09-07)

## Outcome

Implementation and the three-arm smoke gate passed. Nine 500-episode training runs and the 18 existing-checkpoint reevaluation batch were launched in the background. Monitoring stopped immediately after launch.

The nine new checkpoints each run the frozen 20-environment-seed evaluation internally after training. Together with the 18 old-checkpoint reevaluations, this produces all 27 required checkpoint cells. The unified comparison table and paired bootstrap intervals are intentionally deferred until those artifacts exist.

## Implemented protocol corrections

- Evaluation now uses environment seeds 0–19 in one evaluator call, with forced hover, deterministic masked argmax, the training RNG prelude, 500 arrival slots, and zero drain slots.
- `train/episode_reward` and `move/move_rate_m` are written only from episode-terminal cumulative records.
- `forecast/wall_time_frac` is the sum of forecast wall time divided by the measured total training wall time.
- A1 returns the original environment reward with every new forecast/teacher gate off.
- C1 reuses the existing EFT and movement-centroid injection points. Teacher weight is 1 through 50% of updates, linearly reaches 0 at 60%, and remains 0 thereafter; every update log records `teacher_weight` and `teacher_phase` alongside entropy.
- N0-local reuses the N0 analytic advantage path with `forecast_advantage_eta=0`, so only target-DAG Δ contributes.

## Smoke and resource gate

- A1, C1, and N0-local each completed 5 training episodes and the 20-seed evaluation.
- TensorBoard inspection found exactly the prescribed 12 scalar tags.
- C1 smoke contained `pre_anneal`, `annealing`, and `post_anneal` entropy records.
- Measured GPU memory was 580–590 MiB per process.
- Seven RTX 4090 GPUs were verified; free memory after the probe was 17.2–22.3 GiB.
- At the formal maximum density of two processes on one GPU, the conservative measured training time was 66.13 seconds/episode. Adding the fixed 20-seed evaluation gives 33,397 seconds (9.28 hours) per run.

Raw probe: `docs/superpowers/reports/2026-09-07-missing-arms-probe-result.json`.

## Production runs

| Run | GPU | PID | Log | Result |
|---|---:|---:|---|---|
| `20260907_A1_seed0` | 0 | 791866 | `/data2/zrj2025/uav-results/audits/20260907_A1_seed0.log` | `/data2/zrj2025/uav-results/audits/20260907_A1_seed0/result.json` |
| `20260907_A1_seed1` | 1 | 791867 | `/data2/zrj2025/uav-results/audits/20260907_A1_seed1.log` | `/data2/zrj2025/uav-results/audits/20260907_A1_seed1/result.json` |
| `20260907_A1_seed2` | 2 | 791868 | `/data2/zrj2025/uav-results/audits/20260907_A1_seed2.log` | `/data2/zrj2025/uav-results/audits/20260907_A1_seed2/result.json` |
| `20260907_C1_seed0` | 3 | 791869 | `/data2/zrj2025/uav-results/audits/20260907_C1_seed0.log` | `/data2/zrj2025/uav-results/audits/20260907_C1_seed0/result.json` |
| `20260907_C1_seed1` | 4 | 791870 | `/data2/zrj2025/uav-results/audits/20260907_C1_seed1.log` | `/data2/zrj2025/uav-results/audits/20260907_C1_seed1/result.json` |
| `20260907_C1_seed2` | 5 | 791871 | `/data2/zrj2025/uav-results/audits/20260907_C1_seed2.log` | `/data2/zrj2025/uav-results/audits/20260907_C1_seed2/result.json` |
| `20260907_N0-local_seed0` | 6 | 791872 | `/data2/zrj2025/uav-results/audits/20260907_N0-local_seed0.log` | `/data2/zrj2025/uav-results/audits/20260907_N0-local_seed0/result.json` |
| `20260907_N0-local_seed1` | 1 | 791873 | `/data2/zrj2025/uav-results/audits/20260907_N0-local_seed1.log` | `/data2/zrj2025/uav-results/audits/20260907_N0-local_seed1/result.json` |
| `20260907_N0-local_seed2` | 2 | 791874 | `/data2/zrj2025/uav-results/audits/20260907_N0-local_seed2.log` | `/data2/zrj2025/uav-results/audits/20260907_N0-local_seed2/result.json` |

All runs have a conservative estimated completion time of **2026-09-07 21:27:59 Asia/Shanghai**.

Old-checkpoint reevaluation batch: PID `792799`; log `/data2/zrj2025/uav-results/audits/20260907_old_reevaluation_batch.log`; manifest `/data2/zrj2025/uav-results/audits/20260907_old_reevaluation_manifest.json`.

## Version record

- Server HEAD: `644227bb90cd5eb87b89e8cafb66af305f56bd4d`
- Server worktree: dirty, 19 entries at launch; exact list is embedded in `/data2/zrj2025/uav-results/audits/20260907_missing_arms_launch_manifest.json`.
- Actual launch-time script object IDs are in `2026-09-07-missing-arms-production-result.json` and the server launch manifest. Parameters are also embedded from each run's actual parsed namespace, not inferred from a config template.
