# HyperUAV 研究上下文交接文档 v2

> **用途**：把当前对话窗口的必要上下文交接给新窗口。
> **生成时间**：2026-09-12（v1 为 2026-09-08，**已作废，勿用**）
> **本地代码版本**：`HEAD = 053d93d`（工作区 dirty）
> **服务器版本（最近一次正式评测 Track A）**：`644227bb90cd5eb87b89e8cafb66af305f56bd4d`，dirty 29 行
> **代码位置**：`D:\CodeFile\HyperUAV-qly`（设备 `desktop-mnqnchi`）

---

## 0. 一分钟速读

- **课题**：HyperUAV —— 多无人机 MEC 系统中的动态 DAG 任务卸载，HGNN + MAPPO。
- **论文核心主张**：在环境/PPO/奖励/信用分配/评测**完全相同**的条件下，HGNN/超图表示是否优于一个**公平对照的 MLP**。
- **当前状态**：已完成 11 个奖励臂 × 3 seed 的训练（33 个 run），并对 **C1 / B2 / C2** 三臂完成了**固定外生 tape 的正式公平评测**。
- **当前结论**：**EFT 教师预热是主效应**；在有教师的前提下，增量式奖励 vs 原始奖励在最终 checkpoint 上**差异不显著**。
- **下一步**：见 §8。HGNN vs MLP 正式对比尚未启动。

---

## 1. 工作流与协作纪律

### 1.1 Claude ↔ Codex 循环

```
① Claude 设计方案 → 写成 spec
     docs/superpowers/specs/YYYY-MM-DD-<名字>.md

② Claude 写给 Codex 的提示词 → 用户复制到本地 Codex

③ Codex 实现 + 跑实验 → 产出同名配对的两个文件
     docs/superpowers/reports/YYYY-MM-DD-<名字>-results.md   (人读)
     docs/superpowers/reports/YYYY-MM-DD-<名字>-result.json  (机读，全量)

④ 用户返回报告 → Claude 分析 → 下一轮 spec

⑤ 结果追加进 Living Charter
     docs/research/HyperUAV_research_master_roadmap.md
```

**同步机制 = spec↔report 同名配对。**

> **给新窗口的硬要求**：任何时候怀疑上下文过期，**自己按 mtime 列 `docs/superpowers/reports/` 并读掉没见过的报告**，不要问用户"有什么新进展"。用户不应该承担同步的责任。

### 1.2 角色

| 角色 | 职责 |
|---|---|
| **Claude** | 方案设计、代码审查、结果分析、写给 Codex 的提示词 |
| **Codex**（本地 CLI） | 实现、跑实验、返回报告 |
| **用户（Caiyurun）** | 审批算法改动和阶段推进 |

**单一事实来源**：`docs/research/HyperUAV_research_master_roadmap.md`（Living Charter）。不建第二份 master roadmap。

**已作废的约束**：不再以"师兄同意"为推进条件。

### 1.3 版本纪律（AGENTS.md §3）

- 本地 = 编辑源 / 服务器 = 执行副本 / GitHub = 已冻结版本
- 每份报告必须记录：**服务器实际 HEAD、dirty 行数、每个脚本的 git object hash**（Codex 一直照做，报告 §4 有）

### 1.4 给 Codex 写提示词时的固定要求

1. 每个实验跑 500 轮次
2. 用满 7 张卡最快跑完，Codex 自己检查显存并分配（可一卡多进程）
3. 规范化命名含年月日 + 接 TensorBoard，只保留判断"是否学会"和"是否收敛"的字段
4. 结果存已有目录，不新建
5. 挂上服务器后停止监控，返回预计完成时间，据此定定时任务
6. 尽可能节省额度，不要浪费时间在小地方
7. spec 里写**硬停止条件**，Codex 触发就停，不自行判断

---

## 2. 用户最初要验证的 4 件事（锚点，不可丢失）

1. **统一量纲**——时延（秒）和能耗（焦耳）单位不同，用系数统一
2. **只把时延改成增量式**，其余不动，与最原始实验对比 → **A1 vs A2**
3. **把最原始奖励删减到只剩时延+能耗**，再把时延改成增量式 → **B1 vs B2**
4. **先给 EFT 学习信号和移动信号，之后关掉**，看能不能学会 → **C1**

> ⚠️ **注意**：目标 2 和 3 的现有结论**只建立在旧评测口径上**（见 §6.1），尚未在新的固定-tape 公平协议下重做。这是一个待决口子。

---

## 3. 系统与算法（已核对代码）

### 3.1 场景常量（`config.py`）

```
AREA_WIDTH/HEIGHT = 500          NUM_UAVS = 5            NUM_UES = 60
EPISODE_LENGTH = 500             TIME_SLOT_DURATION = 5.0 s
HOTSPOT_RADIUS = 150.0           DAG_BASE_ARRIVAL_PROB = 0.0145
DAG_HOTSPOT_ARRIVAL_MULTIPLIER = 2.0
DAG_MIN_TASKS = 5                DAG_MAX_TASKS = 8
UAV_COMPUTE_RATE_OPS_PER_SEC = 1_000_000
CLEAN_MAX_QUEUE_PER_UAV = 16     CLEAN_UAV_MOVEMENT_SPEED = 15.0
CLEAN_POWER_MOVE = 100.0
```

> ⚠️ **铁律**：`config.py` 写 `DISCOUNT_FACTOR = 0.96`，实际 run 用的是 **γ = 0.99**。
> **已跑实验的参数一律以 run 的实际记录为准，绝不能从 config 推断。**

### 3.2 两类智能体

- **飞行智能体（movement actor）**：参数共享，每架无人机一个实例
- **卸载智能体（offloading actor）**：**单一的、任务中心的中心化策略**——遍历"就绪任务"列表，不是遍历无人机

> **推论**：卸载的信用分配是**时序/顺序**问题，**不是多智能体**问题。

- **奖励全局共享**：`reward_per_uav = reward_total / len(uavs)`
- **卸载动作没有 no-op**——只在结构性原因下跳过

### 3.3 两个"排队时间估价器"（务必区分，曾经答错过）

| | **估价器 A**：候选评估 | **估价器 B**：全局预测 DP |
|---|---|---|
| 文件 | `environment/assignment.py` | `environment/forecast.py` |
| 算什么 | 单任务派给单架无人机的时间/能耗分解 | 全局所有未完成 DAG 的预测完成时间 Φ 及增量 ΔΦ |
| 用在哪 | **策略网络的输入特征** | **奖励（B2/C2）或优势（B2D/N0）** |
| 时序 | 决策**之前** | 决策**之后**（`scorer()` → `argmax` → `delta_for_reservation()`） |
| **部署时** | **必须有，框架的一部分** | **不需要，纯训练脚手架** |
| 开销 | 可忽略 | **训练墙钟 20.15%**，部署 0 |

**估价器 A 的输入维度**（策略实际看到的）：
- pair 特征 8 维：`[传输时间, 传输能耗, 排队时间, 计算时间, 计算能耗, 增量时延, 回传时间, 回传能耗]`
- dynamic 特征 7 维：`[x, y, 队列长度, 剩余槽位, 还要等多久空闲, 排队工作量, 本时隙已分配数]`

> pair 特征第 6 维就是**"增量时延"**（`estimated_finish_time − now`，归一化 `x/(x+160)`）。
> **"增量"概念本来就在策略观测里**，B2 只是把它也搬进了奖励——可能是 B2 收益有限的原因之一。

**估价器 B 的机制**（贪心 + 乐观的影子调度 DP）：
- 影子时钟从真实 `compute_finish_time` 初始化，**故意不含 return time**
- `ready[u] = max over 父 ( min over 父UAV ( finish + 传输 ) )` ← **乐观**
- `finish = max(ready, local_queue) + compute_time`，取 `argmin` 占位 ← **贪心**
- DAG 完成时间 = `max over sinks ( min over UAV )`
- `Φ = Σ_{未完成 DAG} T̂_G`；`ΔΦ` 拆成 `target` + `cross`
- 传输模型简化：`data_mb × 8 × (1 + (d/100)²) / bandwidth`

### 3.4 已知的场景问题（声明了但未强制执行）

- **C7（MIN_UAV_SEPARATION）**：`_apply_clean_movement` 只检查地图边界
- **C8（DAG_TASK_UAV_MAX_DISTANCE）**：`is_assignment_legal` **无通信可达性检查**（代码注释自认 "T7 should extend this helper with communication reachability"）
- 无人机算力异质性参数存在但未使用
- UE 速度矛盾：`UE_GM_MAX_SPEED = 0.6 < UE_WALK_SPEED_MEAN = 1.2`
- `task_execution.py` 分配时把 `uav_available_time` 设为 `compute_finish`，回程时间稍后才补 → 同时隙后续任务可能被排进"算完但没飞回来"的窗口

---

## 4. 奖励公式（每组的确切定义）

### 4.1 最原始奖励（= A1 = C1 的奖励部分）

```
r_t = −0.5  · Σ_{k∈K_t} w_k · min(d_k / 60, 10)     时延惩罚
      −0.05 · Σ_{k∈K_t} (e_k / 250)                  计算能耗
      −0.08 · (E_move,t / 2500)                      飞行能耗
      +8.0  · n_completed_DAG,t                      DAG 完成奖励
      +0.5  · q_t                                    覆盖整形
```

- `K_t` = 本时隙**新结算**的子任务，由 `task.reward_settled` 保证只算一次
- `w_k` = 关键路径 1.0 / 非关键 0.5
- `/60` = `CLEAN_REWARD_TIME_REF`，clip 到 10
- `/250` = `CLEAN_REWARD_TASK_ENERGY_REF` = `P_UAV_COMPUTE × 5.0`
- `/2500` = `CLEAN_REWARD_MOVE_ENERGY_REF` = `5 × 100 × 5.0`
- `q_t` 覆盖整形：`ENABLE_MOVEMENT_POSITION_SHAPING = True` **本来就开着，不是新加的**

> ⚠️ **常见误解**：原始奖励**已经归一化了**（三个 ref 常数早就存在）。
> "统一量纲"这轮改的是**换参考尺度和权重**（时延 ref 60 → 500），**不是从无到有引入归一化**。
> 因此 A1/A2 的差异**不能**归因于"加了归一化"。

### 4.2 重构后的量纲基准（`environment/reward_redesign.py`）

```
flowtime_ref_seconds       = 500.0
lambda_task_seconds/joule  = 1.0
lambda_move_seconds/joule  = 0.10    # D2 组特例：0.04
```

### 4.3 全部 11 个臂

| Arm | 奖励定义 | 教师 | 估价器 B | 解析优势 |
|---|---|:-:|:-:|:-:|
| **A1** | 原始奖励，原样透传 | ✗ | ✗ | ✗ |
| **A2** | `原始 − step_time_penalty + time_reward`（只换时延项为增量式） | ✗ | ✓ | ✗ |
| **B1** | `time_reward − task_cost − move_cost`，时延用**真实 flowtime** | ✗ | ✗ | ✗ |
| **B2** | 同 B1 结构，时延用**增量式三段账** | ✗ | ✓ | ✗ |
| **B2D** | 同 B2，但 `−ΔΦ/500` 标准化后**直接当优势** | ✗ | ✓ | ✓ |
| **C1** | 奖励 = **原始奖励**，加 EFT 教师 + 移动信号，中途退火撤掉 | ✓ | ✗ | ✗ |
| **C2** | 奖励 = **B2 奖励**，加同款 EFT 教师并退火 ← **新增臂** | ✓ | ✓ | ✗ |
| **D1** | （已作废，见 §9 错误 #5） | ✗ | ✓ | ✗ |
| **D2** | 同 B1，`λ_move = 0.04` | ✗ | ✗ | ✗ |
| **N0** | 原始奖励 + `−ΔΦ/500` 当解析优势 | ✗ | ✓ | ✓ |
| **N0-LOCAL** | 同 N0，只用 `target` 不用 `cross` | ✗ | ✓ | ✓ |

> **C2 的存在理由**：C1 相对 B2 同时改了**奖励**和**信用通路**，所以 C1−B2 **不是教师的独立因果效应**。
> **C2−B2 才隔离教师**（同为 B2 奖励，只差教师）；**C2−C1 才隔离奖励形式**（同款教师，只差奖励）。

### 4.4 B2 的"增量式三段账"（望远镜求和）

```
timing_cost = initial_cost + assignment_cost + correction_cost
time_reward = −timing_cost / 500
```

| 项 | 何时记 | 记什么 |
|---|---|---|
| `initial_cost` | DAG **首次出现**，仅一次（`admitted` 集合） | `T̂_G − arrival_time` |
| `assignment_cost` | **每次分配** | `Σ ΔΦ` |
| `correction_cost` | DAG **真正完成** | `真实 return_complete_time − 最后预测锚点`；episode 末未完成的按 `now − 锚点` 补 |

> ⚠️ **2026-09-15 更正：下面这句原话是错的，已证伪，保留原文仅为留痕。**
> ~~三段加起来恰好等于真实 flowtime，与 B1 的 episode 总回报数学上相等。
> → B2 不换目标函数，只是把账从"完成时一次记"改成"分期摊到每次决策"。~~
>
> **实测：B1/B2 同轨迹账本最大相对误差 `0.4891791541`（要求 <1e-6），恒等式从未成立。**
> 原因（代码推导 + 实测）：要闭合必须满足 `Σ ΔΦ_G = anchor_G − T̂⁰_G`，即预测值的**全部**漂移
> 都由分配决策造成。但预测还会因外生原因移动——UE 移动、新 DAG 到达改变排队、以及影子 DP
> 本身是**贪心 + 乐观**的（`environment/forecast.py`，影子时钟故意不含 return time），
> 这部分漂移**完全没有被记账**。
> 因此 **闭合缺口 = 未被记账的预测漂移**；`b2_total ≈ 0.51 × b1_total`，即 B2 账本里约有一半
> 的量来自任何策略都无法控制的漂移。
>
> **推论（必须一并更正）**：
> 1. B2 与 B1 **不是**同一个目标函数的两种记账方式，而是**两个不同的目标函数**。
> 2. 曾据此推出的"B1/B2 优化同一目标、两者都差 ⇒ 问题在目标函数本身"**作废**。
> 3. `B2 − B1` 的对比不得称为"同一实际代价的分期记账效应"。
> 4. 这同时**坐实了** §4.4 原本标注为"仍未坐实"的 B2 失败机制假说，且证据比原假说更硬：
>    不是某一项量级大，而是整个账本有约一半对不上，对不上的部分正好是策略管不着的。
>
> 证据：`docs/superpowers/reports/2026-09-15-six-arm-realign-gating-results.md`

**B2 弱的机制假说（仍未坐实）**：`initial_cost` 量级几百秒且**与策略无关**（纯看到达了什么活儿），`assignment_cost`（真正携带决策质量的）只有几秒到几十秒 → 运气淹没本事。
**验证方法**：三项在 `_diagnostics` 里分开记，读日志即可，不用重跑。**这个检查仍未做。**

### 4.5 教师退火时刻表（C1 / C2 共用）

```python
teacher_anneal_hold_fraction = 0.50
teacher_anneal_end_fraction  = 0.60
w = 1.0                        if progress <= 0.50
  = (0.60 − progress) / 0.10   if 0.50 < progress < 0.60
  = 0.0                        if progress >= 0.60
```

**折算到 500 episode × 500 step/ep = 250k step**：
- episode 250（125k step）开始退火
- **episode 300（150k step）教师权重归零**

---

## 5. 正式评测协议（2026-09-08 起，**这是当前唯一有效的口径**）

### 5.1 统一代价 J

```
J_episode  = [ Σ_{G∈offers} L_G  +  1.0·E_task  +  0.10·E_move ] / 500
J_per_offer = J_episode / max(N_offer, 1)

L_G = C_G − a_G          若 G 在 horizon 内完成回传
    = T_end − a_G        若 G 未完成或未被接纳（截尾）
```

- **越低越好**（这是成本，不是奖励）
- 1.0 秒/J 和 0.10 秒/J 把能耗换成"等价秒"
- **不使用** C1 的 bonus/coverage，**不使用** B2 的 forecast —— 纯用实际结果
- **未接纳/未完成的 offer 按 horizon 截尾计入**（旧口径漏掉了这部分）

### 5.2 协议要点

- **固定外生 tape**：offer/mobility 序列与策略无关，预生成。validation tape ID 100–119（20 条），**独立** test tape ID 200–249（50 条）
- **checkpoint 选择**：每个训练 seed **独立**在 validation tape 上选统一代价最低的点，平局取更早的；锁定后在 test tape 上**只测一次**
- **主协议 = forced-hover**（无人机不移动）；**次协议 = deterministic joint-policy**（完整联合策略）
- **两层 95% bootstrap CI**：先重采样 3 个训练 seed，再在每个 seed 内重采样 test tape
- **只有 3 个训练 seed**，必须报 3/3 / 2/3 / 1/3 的 seed 方向，**不能把 50 个 tape 当成独立训练重复**
- 完成率拆三层：`admission = N_admitted/N_offer`、`conditional = N_completed/N_admitted`、`end-to-end = N_completed/N_offer`
- **训练奖励不参与跨臂正式排名**

---

## 6. 实验结果

### 6.1 旧口径结果（2026-09-07，**已被 §5 协议取代，仅作历史参考**）

> ⚠️ 旧协议：20 个**评测** seed、实时到达（非固定 tape）、无未接纳 offer 分母、无 validation 选点。
> **这些数字不能与 §6.2 的 J/offer 混用。** 但它们是 A1/A2/B1/N0/D2 等臂**目前唯一的数据**。

| 配置 | Completion | Flowtime ↓ | Throughput ↑ |
|---|---|---|---|
| C1 | 0.9181 | 182.4 | 0.1286 |
| Stage-1 参考（100 ep） | 0.8911 | 246.5 | 0.1152 |
| **A2** | 0.8677 | 304.7 | 0.1055 |
| EFT-greedy 参考 | 0.8413 | 331.7 | 0.0971 |
| **A1** | 0.8478 | 367.2 | 0.0922 |
| B2 | 0.8131 | 487.6 | 0.0700 |
| **B2D** | 0.7850 | 559.9 | 0.0626 |
| **N0** | 0.7818 | 573.4 | 0.0614 |
| **N0-LOCAL** | 0.7835 | 578.4 | 0.0610 |
| **B1** | 0.7688 | 598.5 | 0.0591 |
| Random 参考 | 0.7702 | 624.2 | 0.0586 |
| **D2** | 0.7523 | 662.8 | 0.0545 |

旧口径下的读数：A2 优于 A1（用户目标 2）；B1/B2 都远差于 A1（用户目标 3）；所有解析优势方案（N0/N0-LOCAL/B2D）全部失败，且 N0 vs N0-LOCAL 无差异 → **跨 DAG 外部性不是失败原因**。

### 6.2 正式公平评测（2026-09-08/09，C1 / B2 / C2）

**checkpoint 选择结果**：

| 臂 | seed0 | seed1 | seed2 |
|---|---:|---:|---:|
| C1 | 250 | **50** | **50** |
| B2 | 200 | 50 | 250 |
| C2 | 50 | 50 | 50 |

> C1 有两个 seed 选在 episode 50 —— 教师全开期（退火从 250 才开始）。

**配对差（越负越有利）**：

| 对比 | budget 选点 ΔJ/offer | CI | final ΔJ/offer | CI | 有利 seed |
|---|---:|---|---:|---|:-:|
| C1−B2 | −0.4847 | [−0.6758, −0.1639] | −0.5898 | [−0.6245, −0.5562] | 3/3 |
| **C2−B2**（教师效应） | **−0.6061** | [−0.7971, −0.3058] | **−0.6048** | [−0.6387, −0.5686] | 3/3 |
| **C2−C1**（奖励形式） | −0.1214 | [−0.1549, −0.0840] | **−0.0150** | **[−0.0342, 0.0027] 跨 0** | 3/3 |

**绝对值 · forced-hover 主协议**：

| checkpoint | 臂 | J/offer ↓ | delay | task E | move E | admission | conditional | end-to-end ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| budget | C1 | 1.2386 | 590.16 | 103.05 | 0.00 | 0.6263 | 0.9199 | 0.5815 |
| budget | B2 | 1.7234 | 890.19 | 72.71 | 0.00 | 0.4431 | 0.8377 | 0.3788 |
| budget | **C2** | **1.1173** | 515.03 | 110.27 | 0.00 | 0.6735 | 0.9320 | **0.6328** |
| final | C1 | 1.2859 | 619.20 | 100.14 | 0.00 | 0.6085 | 0.9140 | 0.5615 |
| final | B2 | 1.8757 | 982.44 | 65.16 | 0.00 | 0.3903 | 0.8174 | 0.3205 |
| final | **C2** | **1.2709** | 610.14 | 101.01 | 0.00 | 0.6145 | 0.9148 | 0.5677 |

**绝对值 · deterministic joint-policy 次协议**：

| checkpoint | 臂 | J/offer ↓ | delay | task E | move E | end-to-end ↑ |
|---|---|---:|---:|---:|---:|---:|
| budget | **C1** | **1.0075** | 443.22 | 119.28 | 0.95 | 0.6956 |
| budget | B2 | 1.7234 | 890.19 | 72.71 | 0.00 | 0.3788 |
| budget | C2 | 1.2020 | 449.26 | 119.60 | 102.89 | 0.6925 |
| final | C1 | 1.5206 | 668.40 | 97.26 | **84.16** | 0.5355 |
| final | B2 | 1.8832 | 986.05 | 65.38 | 0.48 | 0.3200 |
| final | **C2** | **1.1941** | 560.37 | 106.60 | 0.91 | 0.6050 |

### 6.3 当前可以说的结论

1. **EFT 教师预热是主效应。** C2−B2 在两个 checkpoint 上都是 −0.60 左右，CI 不跨 0，3/3 seed 有利。
2. **在有教师的前提下，增量式奖励 vs 原始奖励的差异很小。** C2−C1 在 budget 选点上 −0.1214 显著，但在 final 上 −0.0150 且 **CI 跨 0**。
3. **教师撤掉后有退化但不致命。** C1 从 1.2386（budget，教师期）退到 1.2859（final，无教师），约 3.8%；C2 从 1.1173 退到 1.2709，约 13.7%。但两者在 final 仍远优于 B2（1.8757）。
4. **移动能耗在不同 checkpoint × 协议下极不稳定**（C1 final joint 的 84.16 vs budget 的 0.95；C2 反向）。**3 个 seed 不足以解读**，但值得警惕"撤教师后飞行策略变乱"。
5. **C1−B2 不能解读为教师的因果效应**（同时改了奖励和信用通路）。这是 spec §5.4 明文规定。

---

## 7. 已确立的负面结论（本课题最硬的证据链）

### 7.1 「针对长时程全局回报的逐决策卸载信用分配，在本环境中不可行」

三条独立证据：
1. **Phase 4A oracle**：只能分辨 63/270 = **23%** 的真值对
2. **奖励密度审计 + PBRS 训练探针**：PBRS（Ng et al. 1999）因望远镜性质**保持回报不变**，不可能给信噪比本就极低的价值函数增加信息；实测 critic EV **变差**
3. **解析 ΔΦ（N0）**：熵降到 0.15 但性能 ≈ 随机 → **锐利但方向错**

### 7.2 但卸载策略的杠杆是**大**的

EFT-greedy vs Random：flowtime **−46.87%**，throughput **+65.82%**，**20/20 seed 全胜**。
→ 问题不是"卸载不重要"，是"从全局回报学不到卸载"。
→ 解法 = **启发式锚定的信用**，即教师预热。§6.2 的 C2−B2 是对这条的正式验证。

### 7.3 方法论结论（值得写进论文）

- **熵下降 ≠ 学会了**——网络可以非常自信地做蠢事
- **critic EV 不能给策略排序**
- **训练奖励曲线不能当裁判**——它测的是"奖励怎么定义的"，不是"策略多好"
  - C1：训练曲线难看（震荡）但评测好
  - B2：训练曲线好看（平稳收敛）但评测差
  - B2 收敛得平稳，恰恰是"奖励与策略无关"的**预测结果**，不是反例
- **旧 TensorBoard 的 `train/episode_reward` 记的是 rollout 值，不是 episode 值**（见 §8.3）——曾据此产生过错误推断

### 7.4 「C1/C2 还算不算强化学习？」

算。教师是**被退火掉的**，最终策略是纯网络推理，标准课程学习。
估价器 B 和 critic 一样是训练期组件——**没人会说 PPO 因为训练用了 critic 就部署依赖 critic**。
分界线是：**部署的策略需不需要它**。答案是不需要（§3.3 已逐行核对）。

---

## 8. TensorBoard 现状（2026-09-09 / 09-10 接入）

**地址**：`http://127.0.0.1:16008/`（持久 tunnel，登录任务 `HyperUAV-RewardRedesign-TensorBoard`）

**共 75 个 run = 33 原始训练 + 9 FormalEval + 33 UnifiedTrain**

### 8.1 `FormalEval/{arm}/seed{seed}`（9 个，C1/B2/C2 × 3 seed）

固定-tape 正式评测结果，横轴是 test tape ID 200–249。
- `forced_hover/budget_selected/*` ← **正式主协议**
- `forced_hover/final_ep0500/*`
- `joint/budget_selected/*`、`joint/final_ep0500/*`
- 每 run 80 个固定-tape tag × 50 点，另加 `train/episode_reward`、`train/critic_loss` 两张上下文卡片

**正式比较以 `FormalEval/*/forced_hover/budget_selected/J_per_offer` 为主。**

### 8.2 `UnifiedTrain/<原始 run>`（33 个）

从 `train_metrics.jsonl` 重建的统一训练口径：
- `train/episode_reward`：只取 `episode_terminal_record=true` 的 `episode_reward_total`，每 run **恰好 500 点**，step 500→250000
- `train/critic_loss`：每 update 的 `ppo_value_loss`，每 run **恰好 2000 点**

覆盖：20260906 批（N0/A2/B1/B2/B2D/D1/D2 × 3 = 21）+ 20260907 批（A1/C1/N0-local × 3 = 9）+ 20260908 批（C2 × 3）= 33

### 8.3 ⚠️ 旧 event 的已知错误

> 旧 2026-09-06 event 里的 `train/episode_reward` **实际写入的是 rollout/update 总奖励**（2000 点），**不是完整 episode 总奖励**。

**后果**：曾据此推断"C1 三 seed 最终训练奖励差 7 倍（374/2684/1214）但评测几乎相同"是个矛盾——那个矛盾相当一部分是**记录 bug**，不是策略行为。`UnifiedTrain/` 视图已修复。**看训练曲线一律用 `UnifiedTrain/`。**

### 8.4 使用注意

- Runs 过滤 `UnifiedTrain/`，Tags 过滤 `train/episode_reward|train/critic_loss`
- **严格读数时把 smoothing 调为 0**
- **不同奖励臂的 `train/episode_reward` 不能直接比高低**——公式不同，不是一把尺子

---

## 9. 未决问题（按优先级）

### P0 —— A1 / A2 / B1 从未在公平协议下评过

§6.2 的固定-tape 正式评测**只覆盖了 C1 / B2 / C2**。
**用户目标 2（A1 vs A2）和目标 3（B1 vs B2）的现有结论全部建立在 §6.1 的旧口径上**——旧口径没有固定 tape、没有 validation 选点、没有未接纳 offer 的截尾分母。

**这是论文里最容易被审稿人打的地方。** 需要决定：是补跑 A1/A2/B1 的固定-tape 评测（不用重训，checkpoint 都在），还是在论文里把这两组降级为探索性结果。

### P1 —— C2−C1 在 final 上 CI 跨 0 的解读

3 个训练 seed 太少。`C2−C1 = −0.0150, CI [−0.0342, 0.0027], 3/3 seed 有利`——方向一致但区间跨 0。
要么增加训练 seed，要么在论文里如实说"有教师时奖励形式的影响未达显著"。

### P1 —— 移动策略在撤教师后的不稳定

C1 final 在 joint 协议下移动能耗 84.16，budget 选点只有 0.95；C2 方向相反。
需要一个针对性诊断（看移动 actor 的熵/动作分布随 episode 的变化），或直接测试"退火后冻结"的修法。

**候选修法**（此前讨论过，未实施）：
1. 退火结束时把 actor 学习率降到 0（冻结）
2. 退火更慢：`hold` 0.50 → 0.70，`end` 0.60 → 0.95
3. 保留残余教师权重 0.05–0.1（代价：削弱"完全撤掉老师"的论文卖点）

### P1 —— B2 失败机制的坐实

读日志即可，不用重跑：拉 `initial_cost / assignment_cost / correction_cost` 的量级比例（见 §4.4）。

### P2 —— HGNN vs MLP 正式对比（尚未启动）

spec：`docs/superpowers/specs/2026-09-06-hgnn-vs-mlp-formal-comparison.md`
要求：**≥5 训练 seed**、**容量匹配**、用 §5 的固定-tape 协议、配方待定（C1 还是 C2？取决于 P1 的结论）。

> **spec 里 C2 的解释范围被明确限定为已批准的两个对比，不进入 HGNN vs MLP。** 若要改用 C2 配方，需要先解除这条限制。

### P3 —— 场景问题
见 §3.4：C7 / C8 未强制、算力异质性未用、UE 速度矛盾。

---

## 10. 我（Claude）犯过的错误 —— 新窗口请引以为戒

用户明确担心："我主要会担心你随着聊天上下文变长会产生幻觉或者忘记关键信息"。

| # | 错误 | 教训 |
|---|---|---|
| 1 | 用 config 的 γ=0.96 算残差，实际 run 是 0.99 → 残差 8.42 ≈ **一整个 DAG bonus** | **参数以 run 实际值为准，不能信 config** |
| 2 | PBRS 探针选错势函数——用了"DAG 完成进度"（不携带质量信息的计数器） | 势函数必须**测量决策质量** |
| 3 | 时序门限用错分母（10% 设在 `forecast/collection` 上，而 collection 只有 ~8ms/slot） | 重测为 `forecast/总训练墙钟 = 20.15%`；**并承认 10% 是我拍的、过严**，已放宽到 30% |
| 4 | 夸大"拖延"风险——用 holding-cost 替换了用户的"不增量"定义 | 卸载 actor **没有 no-op**，该 exploit 大概率不可达。**已撤回** |
| 5 | η 指令自相矛盾，**浪费 3 个 run**（D1 与 B2 逐字节相同） | **被浪费的恰恰是唯一能诊断 ΔΦ 为何失败的那组** |
| 6 | A1 基线不可比——标"已有，不重跑"，拿 100ep/20-seed 去比 500ep/1-seed。**用户抓到** | 用户目标 2 当时根本没被测过 |
| 7 | "决定性胜出"过度断言——只看评测数字，没交代训练稳定性。**用户从图上抓到** | 报结论必须同时报稳定性 |
| 8 | 「策略网络输入里没有排队估价」——对估价器 B 成立，**对 A 不成立**。用户追问"你确定吗"才查出来 | **"排队时间估价器"一词有歧义，先问清是哪一个** |
| 9 | 把 C1−B2 的差异说成"教师的效应"——**C1 同时改了奖励和信用通路，这是混杂的**。Codex 的 spec §5.4 明文纠正 | **隔离教师要看 C2−B2，隔离奖励要看 C2−C1** |
| 10 | 上下文过期时等用户来同步。正确做法是**自己按 mtime 读 reports 目录** | 同步是我的责任，不是用户的 |

**行为准则**：
- 任何关于代码的断言，**先读代码再说**
- 用户挑战时（"你确定吗"），**去查，不要重复断言**
- 怀疑上下文过期，**自己读 reports 目录**
- 承认错误直接说，不要绕

---

## 11. 关键文件索引

### 代码
| 路径 | 内容 |
|---|---|
| `config.py` | 场景 + 算法参数 —— **但已跑实验的参数以 run 记录为准** |
| `environment/metrics.py` | `calculate_step_reward`，原始奖励 |
| `environment/env.py` | 全局共享奖励；`_apply_clean_movement` |
| `environment/assignment.py` | **估价器 A**；`is_assignment_legal`；`build_offloading_candidate_components` |
| `environment/forecast.py` | **估价器 B**：`ForecastContext` |
| `environment/reward_redesign.py` | 各 arm 的奖励拼装；`RewardRedesignLedger`（含 C2） |
| `environment/task_execution.py` | L151 分配时设 `uav_available_time`；L351 `_maybe_start_return` |
| `marl_models/mappo/clean_offloading_actor.py` | 卸载 actor；决策—预测时序在此 |
| `marl_models/mappo/clean_trainer.py` | 教师退火 `_teacher_weight`；解析优势注入点 |
| `scripts/train_clean_mainline.py` | 训练主入口；`_collect_clean_slot` |
| `scripts/eval_clean_mainline.py` | **旧**评测入口（20-seed 口径）—— `forecast` 出现 0 次 |
| `scripts/offloading_policy_gate.py` | 评测期动作选择 —— 无 forecast |
| `scripts/export_fair_eval_tensorboard.py` | FormalEval 导出 |
| `scripts/append_fair_eval_training_tags.py` | FormalEval 训练上下文追加 |
| `scripts/check_fair_eval_tensorboard.py` | FormalEval 校验 |
| `scripts/export_unified_train_tensorboard.py` | UnifiedTrain 导出 |
| `scripts/check_unified_train_tensorboard.py` | UnifiedTrain 校验 |

### 文档
| 路径 | 内容 |
|---|---|
| `AGENTS.md` | Codex 入口：角色、单一事实来源、任务循环、版本纪律、10 条工程原则、禁做实验清单 |
| `docs/research/HyperUAV_research_master_roadmap.md` | **Living Charter（唯一事实来源）** |
| `docs/research/HyperUAV_problem_formulation.md` | (P1) 目标 + 约束 C1–C8，标注了 C7/C8 声明但未执行 |
| `docs/superpowers/specs/` | 各实验 spec |
| `docs/superpowers/reports/` | Codex 返回的报告（**怀疑过期就按 mtime 读这里**） |

### 最近三份报告（v1 交接文档之后新增）
- `2026-09-08-c1-b2-fair-reevaluation-results.md` ← **最重要，§5/§6.2 全部来自它**
- `2026-09-09-fair-eval-tensorboard-integration-results.md`
- `2026-09-10-unified-train-tensorboard-results.md`

### 服务器产物路径
- Track A manifest：`/data2/zrj2025/uav-results/audits/20260908_C1_B2_C2_fair_eval_manifest.json`
- FormalEval 事件根：`/data2/zrj2025/uav-results/audits/20260908_C1_B2_C2_fair_eval/tensorboard_formal_eval`
- UnifiedTrain 事件根：`/data2/zrj2025/uav-results/audits/20260910_unified_train_tensorboard`
- TensorBoard view 根：`/data2/zrj2025/uav-results/tensorboard_views/reward_redesign/`

---

## 12. 给新窗口的开场建议

1. 读本文档 → 读 `docs/research/HyperUAV_research_master_roadmap.md`
2. **按 mtime 扫一遍 `docs/superpowers/reports/`**，确认本文档之后有没有新报告（本文档截止 2026-09-10 的报告）
3. 处理 §9 的 **P0：A1/A2/B1 从未在公平协议下评过**——先跟用户确认是补跑还是降级
4. 然后 P1 三项，再启动 HGNN vs MLP

**不要做的事**：
- 不要在没查代码的情况下断言代码行为
- 不要从 `config.py` 推断已跑实验的参数
- 不要跨奖励臂比训练奖励曲线
- 不要把 §6.1 的旧口径数字和 §6.2 的 J/offer 混用
- 不要把 C1−B2 说成教师的因果效应
- 不要重新引入"等师兄同意"作为推进条件
