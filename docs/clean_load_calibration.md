# Clean Mainline 负载标定记录

日期：2026-07-11  
分支：`zrj_3`  
阶段：Phase 3，负载标定，只修改场景参数与诊断记录。

## 目标

Phase 1 统一了执行时间单位，Phase 2 修复了特征归一化。Phase 3 的目标是把 clean mainline 的任务负载调到适合强化学习训练的区间：

- `random` 基线不要太高，目标约 `50%~70%`；
- `greedy` 基线不要太低，目标约 `80%~90%`；
- UAV 队列不要长期贴满 `CLEAN_MAX_QUEUE_PER_UAV=16`；
- 计算时间不能继续接近 0，否则 offloading 实验会退化成纯通信/排队问题。

## 方法

使用 `scripts/diag_clean_load.py --sweep` 扫描非学习基线：

- `greedy`：每个 ready task 选择估计完成时间最短的 UAV；
- `random`：每个 ready task 在合法 UAV 中随机选择；
- movement 固定 hover；
- sweep 只在内存中覆盖 `config`，不改文件；
- 输出 completion、queue pressure、ready backlog、经验负载系数 `rho_service_time_est`、compute time 分布。

经验负载系数定义为：

```text
rho_service_time_est ~= generated_tasks * avg_task_service_time / (NUM_UAVS * slots * TIME_SLOT_DURATION)
```

它不是严格排队论证明，只用于横向比较场景压力。

## 初筛结论

原始默认场景过重：

```python
DAG_BASE_ARRIVAL_PROB = 0.05
INPUT_DATA_SIZE_MB_RANGE = (1.0, 30.0)
OUTPUT_DATA_SIZE_MB_RANGE = (0.5, 20.0)
TASK_CONSTANT_RANGE = (1, 10)
```

即使降低 arrival，队列仍经常接近满，flowtime 很大。问题不只是任务来得太快，而是单个任务通信负载较重。

过轻场景也不适合主实验：

```python
DAG_BASE_ARRIVAL_PROB = 0.02
INPUT_DATA_SIZE_MB_RANGE = (0.5, 8.0)
OUTPUT_DATA_SIZE_MB_RANGE = (0.25, 4.0)
```

该场景下 random 基线可达到约 `90%`，greedy 接近 `98%`，适合 sanity check，但不适合作为主实验场景，因为策略区分度不足。

中间区域初筛后，将 `TASK_CONSTANT_RANGE` 提升到 `6-60`，使计算时间进入 `p95 ~= 1~2s` 区间，避免计算量继续接近 0。

## Targeted Sweep 复核

在修复 `diag_clean_load.py` 的 policy RNG 隔离后，使用 14 个 seed 重新做 targeted sweep：

```text
seeds = 0,101,202,303,404,505,606,707,808,909,1001,1102,1203,1304
slots = 300
policies = greedy, random
arrival = 0.0145, 0.015, 0.0155, 0.016
input = 0.75:14, 0.75:15
output = 0.6:10.5, 0.6:11
task_constant = 6:60
```

严格通过 Gate 3 的候选有两个：

| arrival | input MB | output MB | task constant | greedy | random | greedy q | random q | p95 compute | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.0145 | 0.75-14 | 0.6-10.5 | 6-60 | 0.8085 | 0.6463 | 0.5833 | 0.8523 | 1.734s | PASS |
| 0.0150 | 0.75-14 | 0.6-10.5 | 6-60 | 0.8020 | 0.6774 | 0.6074 | 0.8521 | 1.608s | PASS |

最终选择第一组：

```python
DAG_BASE_ARRIVAL_PROB = 0.0145
INPUT_DATA_SIZE_MB_RANGE = (0.75, 14.0)
OUTPUT_DATA_SIZE_MB_RANGE = (0.6, 10.5)
TASK_CONSTANT_RANGE = (6, 60)
```

选择理由：

- `random=0.6463` 稳定落在 `0.50~0.70` 目标区间，保留足够学习空间；
- `greedy=0.8085` 刚好超过 `0.80` 下界，说明场景不是物理上做不完；
- `random_queue_pressure=0.8523 < 0.90`，队列压力较高但未长期贴满；
- `p95_compute=1.734s`，计算时间保持在 1-2 秒量级，offloading 决策仍同时包含通信与计算因素；
- 第二组 `arrival=0.0150` 虽然也通过，但 `random=0.6774` 更靠近 `0.70` 上界，部分 seed 偏容易，因此不作为主线默认值。

## 后续使用

正式训练前先跑 quick RL：

```text
30~50 episodes x 200 slots x 2 seeds
```

观察：

- completion rate 是否高于 random、逐步接近 greedy；
- ready backlog 和 queue pressure 是否稳定；
- reward 分量是否被完成惩罚/时间惩罚主导；
- actor/critic loss 是否数值稳定；
- hover ratio 是否符合当前 baseline/improved 配置预期。

如果 quick RL 中 random/greedy 仍区分不足，再考虑 Phase 4 的 reward 标定或观察增强，不要继续反复修改场景参数。

## Phase 4 P1:reward 时间尺度重标(2026-07-11)

依据 P0 的 200-slot 基线(greedy 0.7850±0.0991,random 0.5261±0.0989;drain 协议下两基线完成率均为 1.0,判别指标改为 flowtime:greedy 188s vs random 566s)以及 quick RL 日志中 time_penalty 分布(均值 -57/slot,p5 -156,尾部 -335;DAG bonus 均值 +0.8),将:

- `CLEAN_REWARD_TIME_REF`: 5.0s -> **60.0s**
- 新增 `CLEAN_REWARD_TIME_CLIP = 10.0`(仅作用于 reward 的 norm_time,不影响 metrics 原始 delay/flowtime)

预期:step reward 收敛到 O(1) 量级,|V| 从 ~4e3 降至 O(1e2),pre-clip grad norm 下降 3-4 个数量级,actor 梯度不再被全局 clip 吞没。验证字段:`ppo_returns_mean/std`、`ppo_value_pred_mean`、`ppo_explained_variance`。
