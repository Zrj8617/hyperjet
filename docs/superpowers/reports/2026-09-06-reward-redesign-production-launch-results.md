# Reward redesign production launch — 2026-09-06

## Outcome

The revised timing gate passed and all 21 formal runs were launched in the background. Monitoring stopped immediately after the launcher returned the process IDs and manifest.

- Launch time: **2026-09-06 20:32:08 Asia/Shanghai**
- Allocation: three processes per GPU, one arm per GPU
- Formal workload: 7 arms × 3 seeds × 500 episodes
- Conservative estimated all-complete time: **2026-09-07 06:46:57 Asia/Shanghai**
- Server launch manifest: `/data2/zrj2025/uav-results/audits/reward-redesign-production-launch-20260906.json`

## Implementation and gates

- `_tx_vector` now evaluates `data_MB * 8 * (1 + (distance/100)^2) / base_bw` over a complete NumPy distance array. It no longer calls the scalar communication model per matrix cell.
- Immutable task order, predecessor arrays, sinks, and compute durations are cached for each concrete DAG object across slots. Position-dependent transfer arrays and fixed task schedules remain slot-current and exact.
- Incremental arms use the complete three-part ledger: admission forecast, global `sum(delta_phi)` with eta fixed to 1, and actual-completion/episode-tail correction. Admission cost remains a slot-boundary reward and is not attached to the first offloading decision.
- No padded-parent batching, topology-shape batching, or aligned result-array rewrite was introduced.
- Optimized 20-update timing gate: **forecast / total training wall = 17.30%**, passing the revised 30% threshold.
- Seven-arm smoke: all seven arms completed 5 training episodes, forced-hover evaluation, result JSON, and TensorBoard event writing. The first smoke attempt completed training but exposed a missing TensorBoard package; the dependency was installed and the complete second smoke passed for all arms.

## Capacity and ETA probe

The single in-flight runner occupied 610 MiB. The one permitted `nvidia-smi` snapshot reported free memory of 17,188 / 19,922 / 24,199 / 22,188 / 24,199 / 24,199 / 23,582 MiB on GPUs 0–6. Three processes require about 1,830 MiB per card, leaving at least about 15.4 GiB on the most occupied card at the snapshot.

At the actual density of three processes per GPU, the conservative time estimate for each arm uses the slowest of its three 5-episode probes:

| Arm | GPU | Conservative seconds/episode | Estimated completion (Asia/Shanghai) |
|---|---:|---:|---|
| B1 | 0 | 73.778 | 2026-09-07 06:46:57 |
| D2 | 1 | 42.806 | 2026-09-07 02:28:51 |
| N0 | 2 | 35.879 | 2026-09-07 01:31:07 |
| A2 | 3 | 44.929 | 2026-09-07 02:46:32 |
| B2 | 4 | 36.667 | 2026-09-07 01:37:41 |
| B2D | 5 | 35.978 | 2026-09-07 01:31:57 |
| D1 | 6 | 36.048 | 2026-09-07 01:32:32 |

## Formal run launch table

All paths are under the existing result root `/data2/zrj2025/uav-results/audits`.

| Run | GPU | PID | Log | Result |
|---|---:|---:|---|---|
| `20260906_B1_seed0` | 0 | 78204 | `/data2/zrj2025/uav-results/audits/20260906_B1_seed0.log` | `/data2/zrj2025/uav-results/audits/20260906_B1_seed0/result.json` |
| `20260906_B1_seed1` | 0 | 78205 | `/data2/zrj2025/uav-results/audits/20260906_B1_seed1.log` | `/data2/zrj2025/uav-results/audits/20260906_B1_seed1/result.json` |
| `20260906_B1_seed2` | 0 | 78206 | `/data2/zrj2025/uav-results/audits/20260906_B1_seed2.log` | `/data2/zrj2025/uav-results/audits/20260906_B1_seed2/result.json` |
| `20260906_D2_seed0` | 1 | 78207 | `/data2/zrj2025/uav-results/audits/20260906_D2_seed0.log` | `/data2/zrj2025/uav-results/audits/20260906_D2_seed0/result.json` |
| `20260906_D2_seed1` | 1 | 78208 | `/data2/zrj2025/uav-results/audits/20260906_D2_seed1.log` | `/data2/zrj2025/uav-results/audits/20260906_D2_seed1/result.json` |
| `20260906_D2_seed2` | 1 | 78209 | `/data2/zrj2025/uav-results/audits/20260906_D2_seed2.log` | `/data2/zrj2025/uav-results/audits/20260906_D2_seed2/result.json` |
| `20260906_N0_seed0` | 2 | 78210 | `/data2/zrj2025/uav-results/audits/20260906_N0_seed0.log` | `/data2/zrj2025/uav-results/audits/20260906_N0_seed0/result.json` |
| `20260906_N0_seed1` | 2 | 78211 | `/data2/zrj2025/uav-results/audits/20260906_N0_seed1.log` | `/data2/zrj2025/uav-results/audits/20260906_N0_seed1/result.json` |
| `20260906_N0_seed2` | 2 | 78212 | `/data2/zrj2025/uav-results/audits/20260906_N0_seed2.log` | `/data2/zrj2025/uav-results/audits/20260906_N0_seed2/result.json` |
| `20260906_A2_seed0` | 3 | 78213 | `/data2/zrj2025/uav-results/audits/20260906_A2_seed0.log` | `/data2/zrj2025/uav-results/audits/20260906_A2_seed0/result.json` |
| `20260906_A2_seed1` | 3 | 78214 | `/data2/zrj2025/uav-results/audits/20260906_A2_seed1.log` | `/data2/zrj2025/uav-results/audits/20260906_A2_seed1/result.json` |
| `20260906_A2_seed2` | 3 | 78215 | `/data2/zrj2025/uav-results/audits/20260906_A2_seed2.log` | `/data2/zrj2025/uav-results/audits/20260906_A2_seed2/result.json` |
| `20260906_B2_seed0` | 4 | 78216 | `/data2/zrj2025/uav-results/audits/20260906_B2_seed0.log` | `/data2/zrj2025/uav-results/audits/20260906_B2_seed0/result.json` |
| `20260906_B2_seed1` | 4 | 78217 | `/data2/zrj2025/uav-results/audits/20260906_B2_seed1.log` | `/data2/zrj2025/uav-results/audits/20260906_B2_seed1/result.json` |
| `20260906_B2_seed2` | 4 | 78218 | `/data2/zrj2025/uav-results/audits/20260906_B2_seed2.log` | `/data2/zrj2025/uav-results/audits/20260906_B2_seed2/result.json` |
| `20260906_B2D_seed0` | 5 | 78219 | `/data2/zrj2025/uav-results/audits/20260906_B2D_seed0.log` | `/data2/zrj2025/uav-results/audits/20260906_B2D_seed0/result.json` |
| `20260906_B2D_seed1` | 5 | 78220 | `/data2/zrj2025/uav-results/audits/20260906_B2D_seed1.log` | `/data2/zrj2025/uav-results/audits/20260906_B2D_seed1/result.json` |
| `20260906_B2D_seed2` | 5 | 78221 | `/data2/zrj2025/uav-results/audits/20260906_B2D_seed2.log` | `/data2/zrj2025/uav-results/audits/20260906_B2D_seed2/result.json` |
| `20260906_D1_seed0` | 6 | 78222 | `/data2/zrj2025/uav-results/audits/20260906_D1_seed0.log` | `/data2/zrj2025/uav-results/audits/20260906_D1_seed0/result.json` |
| `20260906_D1_seed1` | 6 | 78223 | `/data2/zrj2025/uav-results/audits/20260906_D1_seed1.log` | `/data2/zrj2025/uav-results/audits/20260906_D1_seed1/result.json` |
| `20260906_D1_seed2` | 6 | 78224 | `/data2/zrj2025/uav-results/audits/20260906_D1_seed2.log` | `/data2/zrj2025/uav-results/audits/20260906_D1_seed2/result.json` |

## TensorBoard and evaluation

The runner writes only the registered 12 scalar tags: `offload/entropy`, four `eval/*` measures, five `train/*` convergence measures, `move/move_rate_m`, and `forecast/wall_time_frac`. Evaluation uses actor argmax, forced hover, the training RNG prelude, the run seed, 500 arrival steps, and no drain, matching the existing leverage/Stage-1 runner protocol.

## Executed version

- HEAD: `644227bb90cd5eb87b89e8cafb66af305f56bd4d`
- Dirty files at launch: 12
- `launch_reward_redesign_production.py`: `dad6b11641af6b7cc44ced4e6f044c0c8c0098e4`
- `run_reward_redesign_arm.py`: `cdb72c9495415f947224b2e32fca0d16a6fa48e6`
- `train_clean_mainline.py`: `1df8731efb383451a91b0cc7a795e3a1794e5077`
- `environment/forecast.py`: `4f91002bd4ba3bb0a643e157587e717a4f64f634`
- `environment/reward_redesign.py`: `de5873bba7865cbf2f570c4dbacea54cf03c8ce7`

The complete dirty-file list is stored in the server launch manifest. Each completed run will also record its actual arguments and version block in its own `result.json`.

## Boundary

This report establishes successful launch, not successful completion. No log tail, process poll, or post-launch GPU query was performed.
