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

## 参考论文后的场景重标定（2026-09-14，分支 `param-realign-20260914`）

背景：任务属性与算力属性要对齐两篇参考文献——Deng et al., *Task Offloading in Internet of
Vehicles: A DRL-Based Approach With Representation Learning for DAG Scheduling*, IEEE TMC
24(6), 2025（DVTP，DAG 结构/节点属性口径）与 Xu et al., *Trajectory Planning and Resource
Allocation for Multi-UAV Cooperative Computation*, IEEE TCOM 72(7), 2024（cycles 与能耗口径）。

**只改参数，不改模型结构与调度内核。**

改前的问题：`num_operation` 的量纲是自造的“运算次数”，与数据量之比只有 **0.006 cycles/bit**，
而 DVTP 是 2.5~250 cycles/bit、Xu 等是 1000 cycles/bit，差 4~5 个数量级。

### 改动（config.py，6 行）

| 参数 | 旧值 | 新值 | 依据 |
|---|---|---|---|
| `INPUT_DATA_SIZE_MB_RANGE` | (0.75, 14.0) | (0.1, 1.0) | DVTP `D_i` 50~500 KB |
| `OUTPUT_DATA_SIZE_MB_RANGE` | (0.6, 10.5) | (0.15, 0.75) | DVTP 边数据 100~500 KB |
| `TASK_CONSTANT_RANGE` | (6, 60) | (500_000, 1_500_000) | 量纲改为每基本算子 cycles |
| `BASE_UPLOAD_BANDWIDTH_MBPS` | [20, 50, 100] | [1.75, 3.5, 7.0] | 锚 DVTP VE↔VES 2 Mbps |
| `BASE_DOWNLOAD_BANDWIDTH_MBPS` | [50, 100, 200] | [3.5, 7.0, 14.0] | 同上，下行取 2 倍 |
| `UAV_COMPUTE_RATE_OPS_PER_SEC` | 1e6 | 1e9 | DVTP VE 档 1~2 GHz（量纲现为 cycles/s） |

改后任务量纲（30 万次采样）：单任务 cycles p5/p50/p95 = 2.8e7 / 2.5e8 / 1.8e9，
**cycles/bit p50 = 62、mean = 90**，落入 DVTP 的 2.5~250 区间。

### Gate 复核（`scripts/diag_clean_load.py`，300 slots × 14 seeds，与 2026-07-11 同协议）

| 指标 | 旧场景 | 新场景 | Gate | 结论 |
|---|---:|---:|---|---|
| greedy completion | 0.8128 | 0.8044 | 0.80~0.90 | PASS |
| random completion | 0.6441 | 0.6569 | 0.50~0.70 | PASS |
| random queue pressure | 0.8527 | 0.8653 | < 0.90 | PASS |
| p95 compute (greedy) | 1.759 s | 1.685 s | 1~2 s | PASS |
| greedy flowtime | 196.53 | 196.48 | — | 等难度 |
| random flowtime | 524.69 | 523.84 | — | 等难度 |

结论：场景难度、策略区分度、计算/通信配比全部保持不变。

口径说明：本次**不是严格复现两篇论文的原值**，而是以论文给出合理量级为参考、按本课题目标
重新标定。任务计算量比 DVTP 原始范围（1e7~1e8 cycles）大一个量级，是为了让计算时间相对
5 秒时隙有意义、且完整 DAG 具备跨时隙特性——greedy flowtime 196 s ≈ 39 个时隙，
random 524 s ≈ 105 个时隙，跨时隙特性明确成立。

旧 checkpoint 与旧固定 tape 在新场景下**全部作废**，需重新生成 tape 并重训。

### 未做（需师兄另行确认）

- 能耗仍是常功率 `E = P_UAV_COMPUTE * t`，未改成 Xu 等的 `E = kappa * f^2 * C`（属模型改动）。
- UAV 仍是单处理器 FIFO（等同 DVTP 的 VE 档），未实现 DVTP 的 VES 多处理器 FAT 队列（属调度内核改动）。

## 规模上限扫描（2026-09-14，同分支）

### 「不超时」的操作化定义

clean 主线没有 deadline/drop，所以「不超时」按三条同时成立判定：

1. 停止到达后 drain 阶段能清空：`final_completion_rate >= 0.95`；
2. 队列不长期贴满：`queue_pressure_mean < 0.90`；
3. drain 预算没被吃满：`drain_slots_used <= 0.5 * drain_budget`。

三条全过记 **OK**；只有 1 过记 **MARGINAL**（能做完但撑满）；1 不过记 **OVERLOAD**。

方法：`scripts/diag_clean_load.py`，200 arrival slots + 150 drain slots，seeds 0/101，
greedy 策略（容量上界探针；random 只用于区分度，不用于容量判定）。

### 扫描结果

| 配置（UAV / UE / DAG 节点数） | arrival 末完成率 | drain 后 | queue press | flowtime | drain slots | 最大并发 DAG | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 / 30 / 5-8 | 0.773 | 1.000 | 0.453 | 178 | 29 | 18.5 | **OK**（小规模） |
| 3 / 36 / 5-8 | 0.812 | 1.000 | 0.618 | 247 | 65 | 24.5 | **OK**（小规模上限） |
| 3 / 60 / 5-8 | 0.633 | 0.922 | 0.884 | 599 | 142 | — | OVERLOAD |
| 4 / 60 / 5-8 | 0.673 | 1.000 | 0.850 | 548 | 122 | — | MARGINAL |
| **5 / 60 / 5-8（主实验）** | **0.842** | **1.000** | **0.465** | **158** | **60** | **28** | **OK（拐点）** |
| 6 / 60 / 5-8 | 0.967 | 1.000 | 0.185 | 54 | 8 | — | OK（偏松） |
| 8 / 60 / 5-8 | 0.973 | 1.000 | 0.093 | 39 | 6 | — | OK（rho<1，过松） |
| 10 / 60 / 5-8 | 0.980 | 1.000 | 0.038 | 26 | 6 | — | OK（过松） |
| 5 / 60 / 8-12 | 0.552 | 0.930 | 0.829 | 481 | 122 | 47 | OVERLOAD |
| 5 / 60 / 10-15 | 0.503 | 0.863 | 0.924 | 851 | 178 | — | OVERLOAD |
| 5 / 60 / 15-20 | 0.374 | 0.654 | 0.961 | 1124 | 200 | — | OVERLOAD |
| 8 / 60 / 10-15 | 0.529 | 0.958 | 0.808 | 562 | 144 | 46 | MARGINAL |
| 10 / 60 / 10-15 | 0.771 | 1.000 | 0.539 | 283 | 62 | 37 | **OK** |
| 10 / 60 / 15-20 | 0.390 | 0.755 | 0.890 | 750 | 150 | 52 | OVERLOAD |
| 10 / 120 / 5-8 | 0.806 | 1.000 | 0.493 | 164 | 56 | 60.5 | **OK（大规模）** |
| 10 / 120 / 10-15 | 0.332 | 0.559 | 0.946 | 693 | 150 | 105 | OVERLOAD |
| 16 / 120 / 10-15 | 0.609 | 1.000 | 0.746 | 439 | 150 | 85.5 | MARGINAL |

到达率扫描（5 UAV / 60 UE / 5-8 节点，`DAG_BASE_ARRIVAL_PROB`）：

| arrival | 生成 DAG 数 | 最大并发 DAG | queue press | 判定 |
|---:|---:|---:|---:|---|
| 0.0145 | 156 | 28.0 | 0.346 | OK |
| 0.020 | 134 | 48.5 | 0.726 | MARGINAL |
| 0.030 | 119 | 54.0 | 0.821 | MARGINAL |
| 0.045 | 116 | 57.0 | 0.858 | MARGINAL |

注意：arrival 超过 0.02 后**吞吐反而下降**（156 → 116 个 DAG）。这是「每 UE 最多一个活动
DAG」的背压：系统堵住以后新 DAG 根本进不来。所以 arrival 不是有效的加压旋钮，
**加压要靠 UE 数或 DAG 节点数**。

### 结论：容量经验规律

- **UE:UAV 比是主控量。** 5-8 节点 DAG 的上限约 **12 UE / UAV**（60/5、120/10、36/3 都在
  这个比例上且判 OK；3/60 = 20:1 就 OVERLOAD）。
- **DAG 变大，比例要同步收紧。** 10-15 节点 DAG 的上限约 **6 UE / UAV**（10 UAV / 60 UE 判 OK，
  10 UAV / 120 UE 判 OVERLOAD）。15-20 节点在任何测过的配置下都没通过。
- **饱和吞吐 ≈ 1.0 个子任务 / UAV / 时隙**（5-8 节点 DAG）；大 DAG 因依赖链等待降到约 0.8。
- **5 UAV 的最大并发 DAG 约 55 个**（受每 UE 单活动 DAG 上限约束，实测饱和在 48~57）。
- DAG 层数从 4 加到 6（更深、更窄）会略微更难（8-12 节点：完成率 0.552 → 0.490），
  但不改变判定档位。

### 推荐的三档实验规模

| 档位 | UAV | UE | DAG 节点 | arrival | 说明 |
|---|---:|---:|---|---:|---|
| 小 | 3 | 30 | 5-8 | 0.0145 | 快速 sanity / 调试 |
| **中（主实验）** | **5** | **60** | **5-8** | **0.0145** | 拐点附近，greedy/random 区分度最大 |
| 大 | 10 | 120 | 5-8 | 0.0145 | 规模泛化，同一 UE:UAV 比 |
| 大（重 DAG） | 10 | 60 | 10-15 | 0.0145 | 验证更复杂 DAG 结构 |

**不要**用 8 或 10 UAV 配 60 UE / 5-8 节点做主实验：rho < 1、queue pressure < 0.1，
场景太松，卸载策略没有区分空间。

### 尚未扫描

- random 策略下的规模上限（本次只用 greedy 作容量上界探针）。
- 每 UE 活动 DAG 上限 > 1 的情形（`--active-dag-caps` 支持，但会改变背压语义）。
- 训练期实际墙钟随规模的增长（大规模点的候选枚举成本明显上升，10-15 节点 + 16 UAV 时
  单次扫描已接近本地超时）。
