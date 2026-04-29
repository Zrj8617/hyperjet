# HyperUAV 新窗口接力包

生成时间：2026-04-29 23:20:37

当前工作目录：`/data2/zrj2025/HyperUAV`

## 1. 当前项目目标

项目方向是多无人机边缘计算中的动态 DAG 任务卸载与调度。当前阶段不是直接做完整联训，而是先把阶段一系统做稳：动态 DAG 任务进入环境、构造异构超图快照、用 HGNN 输出 task-UAV score、通过 score 接管上层任务分配，并通过 imitation learning 和消融实验验证超图结构是否真的有贡献。

阶段一核心目标：

- 动态 DAG 任务真实接入多 UAV 环境。
- 任务节点、UAV 节点、依赖边、可行卸载边、UAV-UAV 边和超边形成动态图快照。
- HGNN encoder + score head 能对 feasible task-UAV edge 打分。
- score 能真实影响任务分配，不只是旁路分析。
- 在不急着联训的前提下，通过 heuristic imitation、soft target 和消融实验验证 learned score 的价值。
- 重点回答：在高密度动态 DAG 卸载场景下，超图结构是否能在 pair feature 之外提供额外收益。

当前定位：

- 已经结束“接口接通阶段”。
- 已经进入“score 行为验证与学习校正阶段”。
- 当前不是“能不能跑”的问题，而是“结果是否稳定、收益来源是否解释清楚”的问题。

## 2. 当前代码目录结构

主要代码目录：

```text
/data2/zrj2025/HyperUAV
├── config.py
├── score_experiment.py
├── pretrain_score.py
├── analyze_score_disagreements.py
├── environment/
│   ├── env.py
│   ├── dag_tasks.py
│   ├── graph_builder.py
│   ├── task_execution.py
│   ├── comm_model.py
│   ├── uavs.py
│   └── user_equipments.py
├── marl_models/hgnn/
│   ├── layers.py
│   ├── encoder.py
│   ├── scheduler.py
│   ├── score_head.py
│   ├── pretrain.py
│   └── supervision.py
└── utils/progress.py
```

相关文档目录：

```text
/data2/zrj2025/各类文档/阶段一实现规格.md
/data2/zrj2025/各类文档/命令集合.md
/data2/zrj2025/各类文档/env.py
/data2/zrj2025/各类文档/方案1.0/合并版.md
```

实验结果目录：

```text
/data2/zrj2025/Result_Hyperuav
```

参考论文与代码：

```text
/data2/zrj2025/各类文档/Huang 等 - 2025 - HyperJet Joint Communication and Computation Scheduling for Hypergraph Tasks in Distributed Edge Co.pdf
/data2/zrj2025/HGNN/HyperJet-main/HyperJet-main
```

## 3. 已经修改过的文件

已经明显参与阶段一主线的文件包括：

```text
config.py
environment/dag_tasks.py
environment/env.py
environment/graph_builder.py
environment/task_execution.py
environment/comm_model.py
environment/user_equipments.py
marl_models/hgnn/layers.py
marl_models/hgnn/encoder.py
marl_models/hgnn/scheduler.py
marl_models/hgnn/score_head.py
marl_models/hgnn/pretrain.py
marl_models/hgnn/supervision.py
score_experiment.py
utils/progress.py
```

文档文件也多次追加记录：

```text
/data2/zrj2025/各类文档/阶段一实现规格.md
/data2/zrj2025/各类文档/命令集合.md
```

## 4. 每个文件具体改了什么

### `config.py`

新增或调整了阶段一相关配置：

- `ENABLE_DYNAMIC_DAG`
- `ENABLE_PHASE_ONE_EXECUTION`
- `ENABLE_LEGACY_REQUEST_PIPELINE = False`

DAG 任务规模：

- `DAG_ARRIVAL_PROB = 0.05`
- `DAG_MIN_TASKS = 5`
- `DAG_MAX_TASKS = 10`
- `DAG_MAX_TASK_LEVELS = 5`
- `DAG_MAX_PARENTS = 2`
- `DAG_TASK_FEATURE_DIM = 11`

任务类型：

- `TASK_TYPE_PREPROCESS`
- `TASK_TYPE_COMPUTE`
- `TASK_TYPE_AGGREGATION`
- `NUM_DAG_TASK_TYPES = 3`

三类任务使用不同 input/output/cpu/deadline 范围，使 DAG 从随机小任务变成轻量工作流语义：

```text
preprocess / sensing
compute-heavy inference
aggregation / output-heavy
```

UE 移动模型改为 Gaussian-Markov：

- `UE_GM_ALPHA = 0.85`
- `UE_GM_MEAN_SPEED = 6.0`
- `UE_GM_SPEED_SIGMA = 1.5`
- `UE_GM_THETA_SIGMA = 0.30`
- `UE_GM_MIN_SPEED = 0.0`
- `UE_GM_MAX_SPEED = UE_MAX_DIST / TIME_SLOT_DURATION`
- `UE_BOUNDARY_MODE = "reflect"`

热点场景：

- `ENABLE_UE_HOTSPOTS = True`
- `NUM_HOTSPOTS = 2`
- `HOTSPOT_UE_RATIO = 0.6`
- `HOTSPOT_STD = 60.0`
- `HOTSPOT_DAG_ARRIVAL_MULTIPLIER = 2.0`

通信模型：

- `A2A_MAX_RANGE = UAV_SENSING_RANGE`
- `A2A_CTRL_OVERHEAD = 0.05`

HGNN / score 相关：

- `USE_HGNN_SCORE_ASSIGNMENT`
- `HGNN_SCORE_CHECKPOINT`
- `SCORE_FALLBACK_TO_HEURISTIC`
- `USE_PHASE_ONE_HYPEREDGES`
- `USE_TASK_UAV_PAIR_FEATURES`
- HGNN hidden dim、embedding dim、score pretrain 参数。

超边开关：

- `USE_COLLABORATIVE_HYPEREDGES`
- `USE_CRITICAL_HYPEREDGES`
- `USE_ATTRIBUTE_HYPEREDGES`

### `environment/dag_tasks.py`

实现动态 DAG 任务系统：

- `TaskNode`
- `DAGJob`
- `DAGTaskManager`

任务状态：

```text
waiting / ready / queued / running / finished / dropped
```

主要功能：

- UE 按概率生成 DAG。
- hotspot UE 使用更高 DAG 到达概率。
- DAG 有 5-10 个子任务。
- DAG 按 level 生成，最多 5 层，最多 2 个父任务。
- 任务类型按 level 固定分配：第一层 preprocess，中间层 compute，最后一层 aggregation。
- 任务特征从 8 维扩展到 11 维：input、output、cpu、slack、level、ready flag、source x/y、task type one-hot。

新增 DAG 作业级统计：

- `dag_total_jobs`
- `dag_successful_jobs`
- `dag_failed_jobs`
- `dag_incomplete_jobs`
- `dag_on_time_successful_jobs`
- `dag_success_rate`
- `dag_failure_rate`
- `dag_incomplete_rate`
- `dag_on_time_success_rate`

判定规则：

- DAG 成功：该 DAG 的所有子任务都 finished。
- DAG 失败：该 DAG 至少一个子任务 dropped。
- DAG 未完成：没有 dropped，但仍有 waiting/ready/queued/running 子任务。

注意：旧实验 JSON 没有这些 DAG 级指标，需要重新跑实验才能得到。

### `environment/env.py`

阶段一主线已经接入：

- 初始化 `DAGTaskManager`。
- 初始化 `HeteroGraphBuilder`。
- 初始化 `PhaseOneTaskExecutor`。
- 可选初始化 `PhaseOneGraphScheduler`。
- `reset()` 时重置任务管理器、执行器和图快照。
- `step()` 中构造 graph snapshot、可选 score assignment、执行任务推进、记录 phase-one reward 和 diagnostics。

重要变化：

- phase-one 模式下 `_get_obs()` 不再调用 legacy `ue.generate_request()`。
- observation 仍保留兼容旧维度的过渡结构，但内容已经偏 task-centric。
- reward 包括完成奖励、deadline penalty、energy penalty、invalid assignment penalty。
- diagnostics 包括 ready/active tasks、feasible edges、score edge count、score/fallback/disagreement/invalid。

### `environment/graph_builder.py`

构造 `HeteroGraphSnapshot`，包含：

- task ids / task features
- UAV ids / UAV features
- DAG dependency edges
- task-UAV feasible edges
- task-UAV edge features
- UAV-UAV edges
- collaborative hyperedges
- critical hyperedges
- attribute hyperedges

feasible task-UAV 判断考虑：

- task 是否 ready。
- task 与 UAV 距离。
- G2A rate 是否大于 0。
- UAV queue 是否未满。
- 前驱任务是否完成。
- 若前驱在其他 UAV 上，则 A2A transfer 是否有限。
- planned finish 是否满足 deadline + tolerance。

pair feature 当前 9 维：

```text
upload_time
predecessor_transfer_time
compute_time
available_time - current_time
planned_finish - current_time
deadline - planned_finish
cross-UAV parent ratio
queue length
distance
```

当前三类超边：

- collaborative：局部 ready tasks + top-K candidate UAVs。当前最主要作用是表达局部共享 UAV 资源池和竞争上下文，但语义还偏粗。
- critical：slack 小于阈值的 active tasks 形成紧急任务超边，目前比较粗，容易变成“一大包紧急任务”。
- attribute：按 input/cpu/slack/level/task_type 相似性构造任务属性超边，稳定辅助，但容易被质疑为普通聚类。

支持以下 ablation 开关：

- `USE_COLLABORATIVE_HYPEREDGES`
- `USE_CRITICAL_HYPEREDGES`
- `USE_ATTRIBUTE_HYPEREDGES`

### `environment/task_execution.py`

实现任务分配与执行：

- `ScheduledTask`
- `PhaseOneStepStats`
- `TaskSupervisionTarget`
- `AssignmentCandidateRecord`
- `AssignmentDecisionRecord`
- `PhaseOneTaskExecutor`

任务分配逻辑：

- 如果有 learned score：在 feasible UAV 中选择 score 最大的 UAV。
- 如果没有 learned score 或 fallback：使用 heuristic。
- heuristic 本质是 EFT，即枚举可行 UAV，估计 planned_finish，选择最早完成的 UAV。

`_estimate_schedule()` 考虑：

- G2A upload time。
- predecessor finish time。
- 如果前驱在不同 UAV，考虑 A2A transfer time。
- UAV available time。
- compute time。
- deadline feasibility。
- energy estimate。

`advance_one_slot()` 支持一个 slot 内连续完成多个短任务，finish time 是浮点数。这个修复了原来一步只能粗粒度推进、短任务被拖一拍的问题。

当前任务卸载语义：

- 子任务只能分配给 UAV。
- 没有本地 UE 执行。
- 不是二元卸载。
- 没有 running-task migration。
- 有跨 UAV 前驱结果传输，即 DAG 子任务如果前驱在不同 UAV 执行，需要 A2A 数据传输。

### `environment/comm_model.py`

通信模型已统一：

- G2A / A2A 使用 Shannon rate。
- A2A 加入距离可用性。
- A2A 加固定控制开销。
- A2A 不可用时 transfer time 为无限大，对应不可行。

阶段一暂不做：

- LoS/NLoS 复杂模型。
- 多跳 A2A 路由。
- 复杂接触窗口预测。
- 动态功率控制。

### `environment/user_equipments.py`

UE 移动从 random waypoint 改为 Gaussian-Markov：

- 每个 UE 有 speed、theta、velocity 状态。
- 速度和方向按 Gaussian-Markov 更新。
- 使用边界反弹，避免 UE 卡在边界。
- 支持 hotspot 初始化和 `is_hotspot` 标记。

### `marl_models/hgnn/layers.py`

实现基础图/超图层：

- `EdgeAggregationLayer`
- `HyperedgePoolingLayer`

`HyperedgePoolingLayer` 的机制：

- 对同一条超边内的节点 embedding 做 pooling。
- 得到 hyperedge context。
- 再把这个 context broadcast 回超边内每个节点。

因此节点不是显式保存“我和哪些节点在一起”，而是通过 embedding 接收同超边上下文。

### `marl_models/hgnn/encoder.py`

`PhaseOneHGNNEncoder` 输入：

- task features
- UAV features
- dependency edges
- task-UAV edges
- UAV-UAV edges
- task hyperedges
- UAV hyperedges

输出：

- task embeddings
- UAV embeddings

当前重要问题：

- collaborative / critical / attribute 三类 task-side hyperedges 在 encoder 中都进入同一组 `task_hyper_layers`。
- encoder 只区分 task-side hyperedge 和 UAV-side hyperedge。
- encoder 没有真正区分 collaborative、critical、attribute 的类型语义。

这就是后续应该做 type-aware hyperedge encoder 的原因。

### `marl_models/hgnn/scheduler.py`

负责将 `HeteroGraphSnapshot` 转成 encoder 输入。

当前映射关系：

```text
collaborative task_ids -> task_hyperedges
collaborative uav_ids -> uav_hyperedges
critical task_ids -> task_hyperedges
attribute task_ids -> task_hyperedges
```

然后调用 encoder，再用 score head 对每条 feasible task-UAV edge 打分。

当前问题：

- 数据结构里有三类任务超边。
- 但 encoder 处理时任务侧三类超边被合并进同一类 task hyperedge。
- 这会混掉一部分类型语义。

### `marl_models/hgnn/score_head.py`

score head 是一个 MLP：

```text
[task_embedding, uav_embedding, pair_feature] -> MLP -> scalar score
```

其中 task_embedding 和 uav_embedding 来自 HGNN encoder。

pair feature 是显式的 task-UAV 局部调度代价信息。

### `marl_models/hgnn/pretrain.py`

实现 score imitation learning：

- `top1`
- `ranking`
- `soft`

当前最强路线是：

```text
soft imitation + pair feature + hyperedges
```

注意：

- soft 只作用在训练 loss 上。
- 推理时仍然是 `argmax(score)`。
- 所以 soft 的分布信息没有直接在部署策略中被采样使用。

### `marl_models/hgnn/supervision.py`

收集 heuristic supervision：

- graph snapshot
- feasible UAV ids
- heuristic top-1 UAV
- heuristic EFT / planned_finish by UAV

当前训练数据来自 heuristic rollout，部署时 learned score 自己接管，会存在 distribution shift。

### `score_experiment.py`

实验主脚本，支持：

- heuristic baseline。
- supervision collection。
- top1/ranking/soft 训练。
- score_on evaluation。
- 输出 JSON 对照报告。

支持 ablation：

- `full`
- `no_hyperedge`
- `no_pair_feature`
- `no_collaborative`
- `no_critical`
- `no_attribute`

已有 terminal progress UI。

现在已接入 `env.task_manager.get_job_summary()`，未来新跑出来的 JSON 会包含 `dag_*` 指标。

### `utils/progress.py`

实现终端进度条，让用户能看到 collect/train/eval 的进度、elapsed、eta、loss、top1 等。

## 5. 当前已经确定不要改的设计

阶段一暂时不要改：

- 不做本地 UE 执行。
- 不做二元卸载。
- 不做混合动作空间。
- UAV 下层动作仍然是二维连续移动 `[dx, dy]`。
- 不做 running-task migration。
- 不急着做上层 score 和下层 RL 的复杂联训。
- 不继续盲目堆更多超边。
- 不把旧 `/data2/zrj2025/各类文档/env.py` 整体迁移到当前环境。
- 不把功率、CPU 分配、缓存迁移等旧语义重新塞回阶段一主线。

原因：

- 当前阶段最重要的是先把上层图调度和 DAG 任务卸载说清楚。
- 如果现在引入混合动作空间、二元卸载、功率控制或联训，会显著增加解释难度。
- 当前最大问题不是系统不够复杂，而是结果稳定性和超图贡献解释还需要收敛。

## 6. 当前还没完成的任务

### 任务 1：重新跑带 DAG 成功率指标的关键实验

因为 `dag_*` 指标是后来加入的，旧 JSON 没有 DAG job success/failure rate。

至少需要重新跑：

- `full seed42/43/44`
- `no_hyperedge seed42/43/44`

如果时间允许，再跑：

- `no_collaborative seed42/43/44`
- `no_critical seed42/43/44`
- `no_attribute seed42/43/44`

### 任务 2：实现 type-aware hyperedge encoder

这是当前最优先的结构改进。

目标：

- 不再把 collaborative、critical、attribute 全部混进同一组 `task_hyper_layers`。
- 让 encoder 对不同超边类型使用独立参数。

建议最小改法：

- 修改 `scheduler.py`，单独传：
  - `collab_task_hyperedges`
  - `critical_hyperedges`
  - `attribute_hyperedges`
  - `uav_hyperedges`
- 修改 `encoder.py`，增加：
  - `collab_task_hyper_layers`
  - `critical_hyper_layers`
  - `attribute_hyper_layers`
  - `uav_hyper_layers`
- forward 中分别更新，再合并。

先不要上复杂 attention/gating。独立 layer 已经足够验证“类型语义是否重要”。

### 任务 3：进一步细化超边语义

当前 collaborative 和 attribute 仍然容易被质疑为 grouping / clustering。

后续可以考虑，但不要一次全做：

- candidate-overlap competition hyperedge。
- A2A transfer-coupling hyperedge。
- resource-demand similarity hyperedge。
- deadline-risk hyperedge。

### 任务 4：补 DAG 完成时间指标

可参考旧 `/data2/zrj2025/各类文档/env.py` 的 DAG finish time 统计思想。

建议新增：

- `dag_avg_completion_time`
- `dag_avg_success_completion_time`
- `dag_avg_on_time_completion_time`

### 任务 5：后续固定上层调度 + 下层 MAPPO 训练

当前不要直接联训。

更稳的路线是：

- heuristic assignment + MAPPO 2D movement。
- full score assignment + MAPPO 2D movement。
- 对比上层调度在下层移动学习中的影响。

## 7. 已经尝试过但效果不好、后续不要再走的方案

### 不要继续只换 imitation loss

已尝试：

- top1 imitation
- ranking imitation
- soft target imitation

结论：

- soft 当前最好，但主要收益不来自“loss 换了一个名字”。
- 当前瓶颈更像 teacher 局部目标、闭环分布偏移、结构表达和推理 argmax 压缩。
- 继续只调 top1/ranking/soft 的 loss，不如做结构消融和 type-aware encoder。

### 不要把 score 是否超过 heuristic 当成唯一验收标准

heuristic 已经不弱，包含上传、前驱传输、UAV 可用时间、计算时间和 deadline 粗判。

阶段一更合理的验收：

- 动态 DAG 闭环跑通。
- 图结构能真实影响 assignment。
- learned score 接近或部分超过 heuristic。
- 超图版相对无图/普通图/去超边版本有可解释收益。

### 不要过早联训

当前上层 learned score 还没有完全稳定。

如果现在直接联训：

- 训练噪声会变大。
- 问题归因会变困难。
- 可能无法判断是 score、hyperedge、pair feature 还是 lower policy 在起作用。

### 不要盲目加更多超边

用户已经指出当前超边定义偏粗，这是合理担忧。

后续重点不是“更多超边”，而是：

- 更清楚的超边语义。
- type-aware 编码。
- 每类超边的消融证据。

### 不要把旧 env.py 主线整体搬回当前环境

旧文件可借鉴部分指标和调度思想，但整体流程与当前 phase-one DAG 主线不一致。

## 8. 当前运行命令

### full 实验示例

```bash
CUDA_VISIBLE_DEVICES=0 python3 score_experiment.py \
  --device cuda \
  --seed 42 \
  --output_dir /data2/zrj2025/Result_Hyperuav/dag_job_full_seed42 \
  --pretrain_episodes 12 \
  --pretrain_steps 120 \
  --pretrain_epochs 10 \
  --eval_episodes 8 \
  --eval_steps 150 \
  --skip_top1 \
  --skip_ranking \
  --ablation full
```

### no_hyperedge 实验示例

```bash
CUDA_VISIBLE_DEVICES=0 python3 score_experiment.py \
  --device cuda \
  --seed 42 \
  --output_dir /data2/zrj2025/Result_Hyperuav/dag_job_nohyper_seed42 \
  --pretrain_episodes 12 \
  --pretrain_steps 120 \
  --pretrain_epochs 10 \
  --eval_episodes 8 \
  --eval_steps 150 \
  --skip_top1 \
  --skip_ranking \
  --ablation no_hyperedge
```

### 单独超边消融示例

```bash
CUDA_VISIBLE_DEVICES=0 python3 score_experiment.py \
  --device cuda \
  --seed 42 \
  --output_dir /data2/zrj2025/Result_Hyperuav/dag_job_nocollab_seed42 \
  --pretrain_episodes 12 \
  --pretrain_steps 120 \
  --pretrain_epochs 10 \
  --eval_episodes 8 \
  --eval_steps 150 \
  --skip_top1 \
  --skip_ranking \
  --ablation no_collaborative
```

将 `--seed` 和 `--output_dir` 改成 43/44 即可继续多 seed。

## 9. 当前报错或待验证点

### 已知 OOM 风险

之前在 GPU 上跑 soft 时出现过 CUDA out of memory。

可规避方式：

- 确认 GPU 空闲。
- 使用 `CUDA_VISIBLE_DEVICES=0` 或 `CUDA_VISIBLE_DEVICES=1` 指定空闲卡。
- 减小 pretrain episodes / steps / eval episodes。
- 不要多个大实验同时占同一张卡。

### 旧 JSON 没有 DAG 作业级成功率

旧实验结果无法直接统计 DAG success/failure rate。

原因：

- 旧 JSON 只记录 task-level/episode-level 指标。
- 没保存每个 DAG job 的完整状态。

解决：

- 重新跑实验。
- 使用新版 `score_experiment.py` 输出 `dag_*` 指标。

### 当前超边语义仍需谨慎表述

不能过度声称：

- 当前 collaborative 同时明确建模了协同、竞争、迁移耦合。

更严谨说法：

- 当前 collaborative 建模局部 ready tasks 与候选 UAV 资源池之间的高阶上下文。
- 从实验看，它对 reward、deadline、invalid 和 disagreement 有贡献。
- 但其内部语义仍较粗，后续需要 type-aware 和更细粒度子语义验证。

### critical hyperedge 贡献不完全稳定

已有结果显示：

- no_critical 时 deadline/invalid 变差。
- 但 finished/latency 可能反而变好。

说明当前 critical hyperedge 可能更偏保守，有 deadline 防护作用，但可能牺牲吞吐或延迟。

## 10. 已有实验结论

### full vs no_hyperedge 三 seed 平均

```text
full - no_hyperedge:
reward +2.1908
finished +3.2500
deadline violations -1.5833
invalid assignments -62.5833
latency -2.3954
disagreements -21.3333
```

结论：

- 超边在 pair feature 之外有额外贡献。
- full 比 no_hyperedge 更稳。
- 超图不是完全摆设。

### 细粒度超边消融三 seed 平均

去 collaborative 损失最大：

```text
full - no_collaborative:
reward +2.5216
deadline violations -5.6250
invalid assignments -48.0833
disagreements -24.8333
```

去 attribute：

```text
full - no_attribute:
reward +1.7576
deadline -3.5833
invalid -44.0000
latency -1.4657
disagreements -2.7500
```

去 critical：

```text
full - no_critical:
reward +1.5197
deadline -4.8750
invalid -23.5833
但 finished -2.9583，latency +0.8174，disagreements +1.5417
```

解释：

- collaborative 是当前贡献最大的超边。
- attribute 是稳定辅助。
- critical 对 deadline/invalid 有帮助，但目前过粗，可能牺牲 throughput/latency。

### pair feature 的作用

pair feature 很重要，而且贡献稳定。

但论文或方案叙事不能变成“pair feature 是主要创新”。

合理表述：

- pair feature 提供局部 task-UAV 代价显式信息。
- HGNN hyperedge 提供任务组和 UAV 资源池的高阶上下文。
- 二者互补。

## 11. 关键概念回答记录

### `latest_graph_snapshot` 是什么

`latest_graph_snapshot` 是某一个时间步的异构超图快照。

它不是静态全局图，而是动态系统在当前时刻的图结构帧。

每个 step 会根据当前 UAV 位置、UE 位置、DAG 任务状态、队列、通信可用性重新构建 snapshot。

因此当前实验是动态图实验，不是静态图实验。

### imitation learning 是什么

这里的 imitation learning 是让 score head 模仿 heuristic teacher。

具体做法：

- 对每个 ready task 枚举 feasible UAV。
- heuristic 计算每个 UAV 的 planned_finish。
- planned_finish 最小的 UAV 作为 teacher top-1。
- score head 学习给 teacher 更高分。

### 为什么用 heuristic 为准

因为当前没有直接的最优标签。

heuristic 是一个可解释、可运行的 teacher：

- 考虑上传时间。
- 考虑前驱跨 UAV 传输时间。
- 考虑 UAV 可用时间。
- 考虑计算时间。
- 考虑 deadline 可行性。

它不是全局最优，但适合作为 warm-start teacher。

### 当前任务怎么卸载

当前是 task-to-UAV assignment。

没有本地 UE 执行。

不是二元卸载。

不存在“本地 or UAV”的 binary decision。

### 图快照和超图的区别

超图是结构类型：一条超边可以连接多个节点。

图快照是时间概念：当前时刻的图/超图状态。

所以当前可以说：

```text
latest_graph_snapshot 是动态异构超图在某个时间步的快照。
```

## 12. HyperJet 参考情况

参考论文：

```text
HyperJet: Joint Communication and Computation Scheduling for Hypergraph Tasks in Distributed Edge Computing
IEEE INFOCOM 2025
DOI: 10.1109/INFOCOM55648.2025.11044587
```

本地 PDF：

```text
/data2/zrj2025/各类文档/Huang 等 - 2025 - HyperJet Joint Communication and Computation Scheduling for Hypergraph Tasks in Distributed Edge Co.pdf
```

本地代码：

```text
/data2/zrj2025/HGNN/HyperJet-main/HyperJet-main
```

HyperJet 可借鉴点：

- hypergraph task scheduling。
- k-hop neighborhood hyperedges。
- attribute hyperedges。
- partitioning hyperedges。
- DAG / hypergraph task 与资源调度结合。

不能照搬点：

- 它有 local / resource pool 离散动作。
- 当前 HyperUAV 阶段一不做本地 UE 二元卸载。
- 当前重点是多 UAV 动态 DAG 场景，不是直接复现 HyperJet。

## 13. 新窗口 Codex 应该从哪里继续

最推荐的继续顺序：

1. 先读取当前代码确认状态：

```bash
pwd
python3 -m py_compile config.py environment/dag_tasks.py environment/graph_builder.py environment/task_execution.py score_experiment.py
```

2. 如果用户要继续实验，优先跑带 `dag_*` 指标的 full/no_hyperedge 多 seed。

3. 如果用户要继续改代码，优先实现 type-aware hyperedge encoder，不要先加更多超边。

4. 如果用户问论文表述，必须谨慎：当前超边确实有效，但 collaborative 和 attribute 的语义还需要进一步细化，不能过度声称已经明确分离协同/竞争/迁移耦合。

5. 如果用户问下一步研究路线，建议：

```text
先稳定当前最强路线 -> 补 DAG 成功率指标 -> type-aware hyperedge encoder -> 细化超边语义 -> 再考虑固定上层调度 + 下层 MAPPO -> 最后再考虑联训。
```

## 14. 额外提醒

用户明确要求：如果用户的提议有风险，必须指出风险，再让用户做决断。

因此后续不要一味顺着用户提议直接做。尤其是这些方向需要提醒风险：

- 过早联训。
- 过早加二元卸载。
- 过早加混合动作空间。
- 继续堆超边但不做消融。
- 把 pair feature 当成主创新。
- 过度声称当前 collaborative hyperedge 已经清楚表达协同/竞争/迁移三种语义。

当前最应该收敛的问题是：

```text
超图到底有没有用；哪类超边最有用；为什么它比普通 pair feature 多提供了信息。
```
