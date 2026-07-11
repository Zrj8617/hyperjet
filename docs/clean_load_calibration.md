# Clean Mainline 负载标定记录

日期：2026-07-11  
分支：`zrj_3`  
阶段：Phase 3，负载标定，仅修改场景参数与诊断脚本。

## 目标

Phase 1 修复了执行时间单位，Phase 2 修复了特征归一化。Phase 3 的目标是把 clean mainline 的任务负载调到一个适合强化学习训练的区间：

- random 基线不要太高，目标约 `50%~70%`；
- greedy 基线不要太低，目标约 `80%~90%`；
- UAV 队列不要长期贴满 `CLEAN_MAX_QUEUE_PER_UAV=16`；
- 计算时间不能继续接近 0，否则 offloading 实验会退化成纯通信/排队问题。

## 方法

使用 `scripts/diag_clean_load.py --sweep` 扫描非学习基线：

- `greedy`：每个 ready task 选择估计完成时间最短的 UAV；
- `random`：每个 ready task 在合法 UAV 中随机选择；
- movement 固定 hover；
- 每个候选场景使用 `slots=300`、`seeds=0,1`；
- sweep 只在内存中覆盖 `config`，不改文件；
- 输出 completion、queue pressure、ready backlog、经验负载系数 `rho_service_time_est`、compute time 分布。

经验负载系数定义为：

```text
rho_service_time_est ~= generated_tasks * avg_task_service_time / (NUM_UAVS * slots * TIME_SLOT_DURATION)
```

它不是严格排队论证明，只用于横向比较场景压力。

## 扫描结论

### 默认场景过重

默认参数：

```python
DAG_BASE_ARRIVAL_PROB = 0.05
INPUT_DATA_SIZE_MB_RANGE = (1.0, 30.0)
OUTPUT_DATA_SIZE_MB_RANGE = (0.5, 20.0)
TASK_CONSTANT_RANGE = (1, 10)
```

即使把 arrival prob 降到 `0.01`，队列仍经常接近满，flowtime 很大。说明问题不只是任务来得太快，而是单个任务通信负载太重。

### 轻量场景过易

候选：

```python
DAG_BASE_ARRIVAL_PROB = 0.02
INPUT_DATA_SIZE_MB_RANGE = (0.5, 8.0)
OUTPUT_DATA_SIZE_MB_RANGE = (0.25, 4.0)
```

random 基线已经在 `90%` 左右，greedy 接近 `98%`。这个场景适合 sanity，但不适合作为主实验场景，因为策略区分度不足。

### 中间区域 sweep

第一轮中间 sweep 显示，`input=(0.5,12)` 到 `(0.75,15)`、`output=(0.25,6)` 到 `(0.5,8)` 比较接近目标，但 random 仍偏高。随后加入 `TASK_CONSTANT_RANGE` 扫描，将计算时间从原来的近 0 提升到 `p95 ~= 1~2s` 区间。

最接近 Gate 3 的候选如下：

| arrival | input MB | output MB | task constant | greedy | random | greedy q | random q | p95 compute |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.016 | 0.75-15 | 0.6-11 | 6-60 | 0.799 | 0.704 | 0.737 | 0.828 | 1.733s |
| 0.0175 | 0.75-15 | 0.5-10 | 6-60 | 0.809 | 0.728 | 0.772 | 0.845 | 1.969s |
| 0.016 | 0.75-16 | 0.5-10 | 7-70 | 0.819 | 0.739 | 0.752 | 0.846 | 2.292s |
| 0.015 | 0.75-15 | 0.5-8 | 6-60 | 0.865 | 0.756 | 0.612 | 0.730 | 1.685s |

没有候选严格同时满足 random `<=0.70`、greedy `>=0.80`、queue pressure `<0.90`。最接近的点是第一行：

- greedy 约 `0.799`，贴近 `0.80` 下界；
- random 约 `0.704`，略高于 `0.70` 上界；
- 队列压力没有长期贴满；
- `p95 compute ~= 1.73s`，计算量级进入合理区间。

因此将 Gate 3 视为“近似通过”，选择该点作为 clean mainline 主线场景。

注意：该点在两个 seed 上有明显方差。seed 0 的 greedy/random 为 `0.865/0.714`，seed 1 为 `0.732/0.693`，均值落在边界附近。因此后续 quick RL 不能只看单 seed，至少应使用 `2` 个 seed 判断趋势。

## 最终参数

```python
DAG_BASE_ARRIVAL_PROB = 0.016
INPUT_DATA_SIZE_MB_RANGE = (0.75, 15.0)
OUTPUT_DATA_SIZE_MB_RANGE = (0.6, 11.0)
TASK_CONSTANT_RANGE = (6, 60)
```

选择理由：

- 比默认场景轻，避免系统物理上做不完；
- 比轻量场景重，避免 random 轻松 90%+；
- greedy 和 random 有可见差距；
- 队列压力仍高但未长期贴满；
- 计算时间不再接近 0，offloading 决策同时包含通信和计算因素。

## 后续使用

正式训练前建议先跑 quick RL：

```text
30~50 episodes x 200 slots x 1~2 seeds
```

观察：

- completion rate 是否高于 random/接近 greedy；
- ready backlog 和 queue pressure 是否稳定；
- reward 分量是否被完成惩罚主导；
- actor/critic loss 是否数值稳定。

如果 quick RL 中 random/greedy 仍区分不足，再考虑 Phase 4 的 reward 标定或观测增强，而不要继续反复改场景参数。
