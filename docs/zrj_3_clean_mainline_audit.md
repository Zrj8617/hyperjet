# zrj_3 clean mainline 代码审计报告

审计范围：`config.py`、`environment/` 核心环境文件、`marl_models/hgnn/`、`marl_models/mappo/`、`scripts/train_assignment_mappo.py`、`scripts/static_scheduler_compare.py`、`scripts/pretrain_score.py`。

约束：本阶段只审计，不修改业务代码。本报告用于下一阶段重构 HyperUAV clean mainline。

## 0. 当前结论

当前 `zrj_3` 不是 clean mainline，而是多条旧实验主线叠加后的版本：

- 环境仍是旧的“热点用户身份”模型，不是“episode 重采样固定圆形热点区域，任意 UE 进入后 DAG 到达概率提高”的地图属性模型。证据：`config.py:52-62`，`environment/user_equipments.py:37-76`。
- DAG 任务仍强依赖 deadline、task_type、slack、高风险任务流，不是我们已经收口的“deadline 不进入主奖励，只作为 QoS 对比指标”的主线。证据：`environment/dag_tasks.py:16-28`，`environment/dag_tasks.py:115-161`，`environment/task_execution.py:524-608`。
- 图构建仍保留 service-domain、resource-competition、critical、critical-support、candidate-scarce、task-type 等旧超边，不是“DAG 依赖 + K-hop + 属性相似超边”的 clean mainline。证据：`environment/graph_builder.py:12-31`，`environment/graph_builder.py:65-126`。
- HGNN/MAPPO 模型层直接依赖旧 snapshot 字段和旧特征维度，因此不能先删模型字段，必须先重写接口再迁移模型。证据：`marl_models/hgnn/scheduler.py:62-178`，`marl_models/hgnn/encoder.py:29-42`，`marl_models/mappo/assignment_mappo.py:82-190`。

## 1. 新主线必须重写的文件

### `config.py`

必须重写为 clean mainline 的参数中心。当前配置混合了旧缓存请求、旧 deadline 奖励、旧超边消融、旧 Stage B 移动奖励、旧 HGNN score 预训练参数。

关键问题：

- UAV 数量仍是 5，不是已收口的 8：`config.py:25-27`。
- UE 速度明显偏大，当前 `UE_GM_MEAN_SPEED=6.0 m/s`、`UE_MAX_DIST=15.0 m/slot`：`config.py:35-48`。
- 热点仍是“热点 UE 比例 + 高斯初始化”，不是“episode 内固定圆形热点区域”：`config.py:52-62`。
- DAG 父节点最大数仍为 2，不是已讨论的 3：`config.py:73-79`。
- task_type、deadline range、slack、critical threshold 仍是主线配置：`config.py:88-110`，`config.py:123-148`。
- 旧超边开关数量过多，已经干扰主线判断：`config.py:214-247`。

建议重写方向：

- 分成 `ENV`, `HOTSPOT`, `UE_MOBILITY`, `DAG`, `COMM`, `GRAPH`, `REWARD`, `METRICS` 七组。
- 保留 deadline 参数但归入 `METRICS/QOS_EVAL`，不得作为默认 reward 或候选过滤条件。
- 删除旧超边开关的主线地位，只保留 clean mainline 三类结构开关：`USE_DAG_DEP_HYPEREDGE`、`USE_KHOP_HYPEREDGE`、`USE_ATTRIBUTE_HYPEREDGE`。

### `environment/user_equipments.py`

必须重写 UE 状态机。当前 UE 在构造时被固定标记为 hotspot 用户，并围绕热点高斯初始化：`environment/user_equipments.py:37-76`。这与已确定场景不一致。

必须支持：

- episode 重采样一个固定圆形热点区域；
- UE 普通行人速度；
- UE 进入热点区域后有一次吸引判定；
- 被吸引或处于 service-waiting 期间低速移动，速度比例 `0.2x`；
- DAG 完整完成后 UE 释放，恢复随机移动；
- 任意 UE 进入热点区域后 DAG 到达概率倍率 `2x`。

当前缺口：

- 没有 `service_waiting`、`active_dag_id`、`hotspot_slowdown`、`was_attracted` 等状态。
- `update_position()` 永远按同一高斯马尔科夫速度走：`environment/user_equipments.py:78-88`。
- 旧 `generate_request()` 是缓存/文件请求逻辑，不属于新 DAG 主线：`environment/user_equipments.py:90-102`。

### `environment/dag_tasks.py`

必须重写 DAG 生成与生命周期管理。

当前问题：

- `TaskNode` 把 `deadline` 和 `task_type` 作为核心字段：`environment/dag_tasks.py:16-28`。
- `observe_time_step()` 每步把所有 active task 的 `source_pos` 更新成 UE 实时位置：`environment/dag_tasks.py:84-88`。这与“DAG 产生时固定 source_pos，最终回传看 UE 实时位置”的设定冲突。
- DAG 生成仍基于旧热点 UE 身份提高到达率：`environment/dag_tasks.py:84-97` 负责刷新/生成，热点身份来自 `environment/user_equipments.py:37-40`。
- ready、job summary、critical path 都绑定 deadline/slack：`environment/dag_tasks.py:115-161`，`environment/dag_tasks.py:163-283`。
- 任务特征包含 slack 和 task_type one-hot：`environment/dag_tasks.py:309-320` 后续特征构造。

必须支持：

- 每个 UE 同时最多一个 active DAG；
- DAG 产生时记录 `source_pos_at_arrival`，后续不随 UE 移动更新；
- DAG 完成时记录 `ue_current_pos_for_return`，用于最终结果回传；
- DAG 级固定属性：`bandwidth_up`、`bandwidth_down`；
- 子任务属性：`input_size`、`output_size`、`cpu_cycles`、`level`、`predecessors/successors`；
- 任务复杂度分布先按已定比例 `0.2, 0.7, 0.1`；
- 父节点可从所有前序层选择，最大父节点数 3；
- critical path 作为评价/分析指标，不以 deadline/slack 作为主线定义。

### `environment/task_execution.py`

必须重写执行时延模型与奖励统计接口。

当前问题：

- `ScheduledTask` 只有总 transmission、execution、energy，无法区分上传、跨 UAV、回传、计算能耗：`environment/task_execution.py:14-23`。
- step stats 仍以 on-time/deadline violation 为核心：`environment/task_execution.py:25-47`。
- 候选估计字段包含 `deadline_margin` 和 `deadline_violation_estimated`：`environment/task_execution.py:58-72`。
- ready task 排序按 deadline：`environment/task_execution.py:226-238`。
- `_estimate_schedule_result()` 对每个任务都计算 UE-UAV upload，而不是只对入口任务上传；并且用 deadline 过滤候选：`environment/task_execution.py:558-595`。
- 能耗只粗略合并 compute 和 tx，没有最终回传能耗：`environment/task_execution.py:597-608`。
- task 完成时只标记子任务 finished，没有 DAG 完成回调和 UE service-waiting 释放：`environment/task_execution.py:699-722`。

必须支持：

- entry task：UE 到 UAV 上传；
- non-entry task：只等待父结果，跨 UAV 时发生 UAV-UAV 中间结果传输；
- exit task / DAG 完成：UAV 到 UE 实时位置回传最终结果；
- 记录 `upload_time`、`inter_uav_transfer_time`、`compute_time`、`return_time`、`total_delay`；
- 记录 `compute_energy`、`communication_energy`、`return_energy`；
- 移动能耗只 logging，不进入默认 reward；
- 默认候选过滤不使用 deadline。

### `environment/graph_builder.py`

必须重写 snapshot schema 和超边构建。

当前问题：

- Snapshot 暴露大量旧超边字段：`environment/graph_builder.py:12-31`。
- 构图主流程仍受旧开关控制：`environment/graph_builder.py:65-126`。
- task-UAV 可行性仍用 deadline 过滤：`environment/graph_builder.py:187-225`。
- pair feature 里包含 planned finish、deadline margin 等“答案型”字段：`environment/graph_builder.py:253-307`。
- service-domain 是空间邻域超边，已被本轮讨论放弃：`environment/graph_builder.py:309-354`。
- resource/critical/critical-support 超边大量依赖 deadline、task_type、slack：`environment/graph_builder.py:356-620`。

必须支持：

- DAG 依赖结构；
- K-hop 局部结构；
- 属性相似超边：周期性观察全局 active unfinished DAG tasks，按属性向量聚类；
- 属性向量当前暂定：`input_size`、`output_size`、`cpu_cycles`、`bandwidth_up`、`bandwidth_down`；
- 属性超边更新周期先设 5 step；
- 删除空间密度超边主线；
- clean snapshot 字段要与 HGNN/MAPPO 同步迁移，不能只改 builder。

### `environment/env.py`

必须重写 step/reset 主流程。

当前问题：

- `__init__()` 初始化 UE 类静态热点，但 `reset()` 不重新调用 `UE.initialize_ue_class()`：`environment/env.py:58-67`，`environment/env.py:139-165`。这会导致热点是否按 episode 重采样不清晰。
- step 中 DAG 生成、构图、分配发生在 UE 位置更新之前：`environment/env.py:173-230`，`environment/env.py:262-264`。新主线需要明确“先移动再根据当前位置判断热点”，否则热点区域语义混乱。
- legacy request pipeline 仍在主流程中：`environment/env.py:176-184`，`environment/env.py:243-260`。
- 阶段一奖励仍是 deadline/on-time/energy/invalid 的组合：`environment/env.py:249-255`，reward 函数后续还会读取 executor deadline stats。
- 动作在 step 末尾才应用：`environment/env.py:270-279`。如果保留这个顺序，需要文档化动作作用于下一时隙；否则建议 clean mainline 改为 step 开头应用动作。

必须支持：

- reset 时重采样一个热点区域；
- episode 内热点固定；
- UE 移动、热点判断、DAG 到达、图构建、调度执行、DAG 完成释放 UE 的顺序固定；
- 记录 movement energy，但默认 reward 不包含它；
- metrics 输出本轮已定主指标、系统指标、辅助指标。

### `environment/comm_model.py`

必须补 clean mainline 通信模型接口。

当前问题：

- 现有 UE-UAV 速率是 Shannon/free-space 路径损耗模型：`environment/comm_model.py:5-23`。
- 当前 DAG 执行直接调用 `calculate_g2a_rate()`：`environment/task_execution.py:558-561`。
- A2A 传输可保留，但需要和新 DAG 任务的中间结果传输语义对齐：`environment/comm_model.py:38-58`。

必须支持：

- DAG 固定 `base_upload_bandwidth/base_download_bandwidth`；
- 距离修正：例如 `effective_rate = base_bandwidth * distance_factor(distance)`；
- 上行、下行分离；
- A2A 传输保持单独接口，避免混用 UE-UAV 逻辑。

### `environment/uavs.py`

必须拆分 UAV clean 状态与旧缓存请求逻辑。

当前问题：

- UAV 初始化包含 cache、file popularity、MBS rate 等旧逻辑状态：`environment/uavs.py:49-55`。
- legacy service/content request 处理占据主体：`environment/uavs.py:102-260`。
- movement energy 已经有基础记录：`environment/uavs.py:40-45`，`environment/uavs.py:87-91`，但目前会和 task energy 混在同一个 `uav.energy` 中被环境奖励读取。

建议：

- clean mainline 保留 `id/pos/remaining_energy/current_slot_energy/dist_moved/queue info`。
- 旧 cache/request 逻辑 deprecated，不第一批物理删除。
- movement energy 记录成单独字段，例如 `movement_energy_current_slot`，默认只进 diagnostics。

### `marl_models/hgnn/`

必须在环境 schema 稳定后迁移，不建议第一批直接重写。

当前依赖：

- `scheduler.py` 固定读取旧 snapshot 字段：`marl_models/hgnn/scheduler.py:62-178`。
- pair hyperedge score features 统计 service/resource/critical 三类旧超边：`marl_models/hgnn/scheduler.py:205-275`。
- encoder 内部固定有 service/resource/critical-support/critical/candidate-scarce 等层：`marl_models/hgnn/encoder.py:29-42`。
- forward 仍把旧超边全部送入 encoder：`marl_models/hgnn/scheduler.py:277-356`。

建议：

- 新建或重写为 clean encoder：DAG dependency、K-hop、attribute hyperedge、task-UAV edges、UAV peer edges。
- 在迁移前保留旧类名，避免 scripts 和 MAPPO 立刻断裂。

### `marl_models/mappo/`

暂时不重写算法主体，但必须在 graph schema 变更后同步调整输入维度。

当前依赖：

- `AssignmentMAPPO` 强依赖 `PhaseOneGraphScheduler` 编码结果：`marl_models/mappo/assignment_mappo.py:32-48`。
- actor 输入直接拼接 task embedding、UAV embedding、edge pair features：`marl_models/mappo/assignment_mappo.py:82-143`，`marl_models/mappo/assignment_mappo.py:145-212`。

风险：

- 如果先改 `BASE_TASK_UAV_PAIR_FEATURE_DIM` 或 snapshot 字段，MAPPO 会直接维度不匹配。
- 第一阶段建议只审计，第二阶段先让 env smoke pass，再迁移 MAPPO。

## 2. 可以删除的旧逻辑

这里的“可以删除”指 clean mainline 不应继续保留为主路径；实际删除应等引用迁移完成后执行。

1. 旧热点用户身份逻辑
   - `ENABLE_UE_HOTSPOTS`、`NUM_HOTSPOTS`、`HOTSPOT_UE_RATIO`、`HOTSPOT_STD` 作为主线参数应删除：`config.py:52-62`。
   - `UE.is_hotspot`、`hotspot_id`、按 UE id 前 60% 标记热点用户应删除：`environment/user_equipments.py:37-40`，`environment/user_equipments.py:55-64`。
   - 按热点中心高斯初始化 UE 应删除：`environment/user_equipments.py:66-76`。

2. deadline 主奖励与主候选过滤
   - `PHASE_ONE_DEADLINE_PENALTY`、`PHASE_ONE_DAG_ON_TIME_BONUS` 等不应作为 clean reward 主线：`config.py:155-164`。
   - `deadline_margin`、`deadline_violation_estimated` 不应作为主候选字段：`environment/task_execution.py:58-72`。
   - deadline 过滤候选应删除或移到 QoS eval 模式：`environment/task_execution.py:588-595`，`environment/graph_builder.py:221-223`。

3. 旧 task_type 主线
   - `TASK_TYPE_PREPROCESS/COMPUTE/AGGREGATION` 不符合新任务属性主线：`config.py:91-110`。
   - 基于 task_type 的记录与统计应从主线移除：`environment/task_execution.py:95-104`，`scripts/train_assignment_mappo.py:69-70`，`scripts/train_assignment_mappo.py:103-144`。

4. 空间密度/服务域超边
   - service-domain 超边把空间邻近 ready tasks 圈起来，已经在讨论中放弃：`environment/graph_builder.py:309-354`。
   - 对应开关应从主线删除：`config.py:216-221`。

5. resource/critical/candidate-scarce/task-type 旧超边
   - resource competition 依赖 deadline、queue、候选稀缺：`environment/graph_builder.py:356-432`。
   - critical/critical-support 依赖 slack、deadline、task_type：`environment/graph_builder.py:434-620`。
   - candidate-scarce/task-type attribute 不属于当前 clean mainline：`environment/graph_builder.py:110-126`。

6. selective HGNN scoring 风险启发式
   - 以 slack、critical path、candidate scarcity 判定高风险任务：`config.py:189-213`。
   - 当前不应作为 clean mainline 的默认训练路径。

## 3. 暂时不能删除、只能 deprecated 的文件/模块

1. `environment/uavs.py` 的 cache/request 旧逻辑
   - 虽然不属于新 DAG 主线，但 `Env` 仍直接实例化 `UAV`：`environment/env.py:64-75`。
   - legacy 分支仍可能被开关触发：`environment/env.py:176-184`，`environment/env.py:243-260`。
   - 建议标记 deprecated，等 clean UAV 类稳定后再删除。

2. `marl_models/hgnn/`
   - 旧 encoder/scheduler 与 snapshot 强绑定：`marl_models/hgnn/scheduler.py:62-178`，`marl_models/hgnn/encoder.py:29-99`。
   - `scripts/pretrain_score.py` 仍直接调用 HGNN 预训练入口：`scripts/pretrain_score.py:17-19`，`scripts/pretrain_score.py:47-58`。
   - 不能先删，否则 MAPPO 和 pretrain 脚本直接不可运行。

3. `marl_models/mappo/`
   - assignment MAPPO 通过 HGNN scheduler 编码图：`marl_models/mappo/assignment_mappo.py:32-48`。
   - actor 输入使用旧 edge pair feature：`marl_models/mappo/assignment_mappo.py:176-190`。
   - 只能等 clean graph schema 定型后迁移。

4. `scripts/static_scheduler_compare.py`
   - 当前是大量消融配置和诊断统计的集合：`scripts/static_scheduler_compare.py:76-124`。
   - `scripts/train_assignment_mappo.py` 还 import 它的 `_apply_ablation_config/_override_num_uavs/_override_num_ues`：`scripts/train_assignment_mappo.py:18-22`。
   - 不能直接删，先 deprecated 并抽出仍需要的 override 工具。

5. `scripts/train_assignment_mappo.py`
   - 当前训练入口能跑旧 assignment MAPPO，但引用 deadline tolerance、旧 ablation、task_type 统计：`scripts/train_assignment_mappo.py:152-179`，`scripts/train_assignment_mappo.py:316-329`。
   - 建议 deprecated 后新建 clean 训练入口，或重写为 clean 入口。

6. `scripts/pretrain_score.py`
   - 与旧 score imitation 强绑定，不属于新主线，但可作为历史 baseline/pretrain 参考。
   - 标记 deprecated，等 clean HGNN encoder 完成后再决定是否重写。

## 4. 旧逻辑的引用位置

### hotspot 旧逻辑

- 配置：`config.py:52-62`。
- UE 静态热点中心：`environment/user_equipments.py:20-35`。
- UE 身份标记：`environment/user_equipments.py:37-40`。
- 通过 UE id 判断热点用户：`environment/user_equipments.py:55-64`。
- 热点 UE 初始位置高斯采样：`environment/user_equipments.py:66-76`。

### deadline/slack 旧逻辑

- 配置 deadline range：`config.py:88-110`。
- critical/slack 配置：`config.py:123-148`。
- TaskNode deadline 字段：`environment/dag_tasks.py:16-28`。
- `remaining_slack()`：`environment/dag_tasks.py:45-47`。
- DAG slack/高风险判断：`environment/dag_tasks.py:115-161`。
- job summary on-time/tardiness：`environment/dag_tasks.py:163-283`。
- executor 候选 deadline 字段：`environment/task_execution.py:58-72`。
- executor deadline 排序和过滤：`environment/task_execution.py:237-238`，`environment/task_execution.py:588-595`。
- graph 可行边 deadline 判断：`environment/graph_builder.py:221-223`。

### task_type 旧逻辑

- 配置 task type：`config.py:91-110`。
- TaskNode 字段：`environment/dag_tasks.py:16-28`。
- assignment record 字段：`environment/task_execution.py:95-104`。
- train assignment 按 task_type 汇总：`scripts/train_assignment_mappo.py:66-144`。
- static compare task_type 诊断：`scripts/static_scheduler_compare.py:158-165`。

### 旧超边

- Snapshot 字段：`environment/graph_builder.py:12-31`。
- build 开关：`environment/graph_builder.py:65-126`。
- service-domain：`environment/graph_builder.py:309-354`。
- resource-competition：`environment/graph_builder.py:356-432`。
- critical/critical-support：`environment/graph_builder.py:434-620`。
- HGNN scheduler 读取旧超边：`marl_models/hgnn/scheduler.py:95-178`。
- HGNN encoder 旧超边层：`marl_models/hgnn/encoder.py:29-42`，`marl_models/hgnn/encoder.py:74-99`。
- static compare 旧消融开关：`scripts/static_scheduler_compare.py:76-124`。

### 旧通信/执行模型

- UE-UAV Shannon rate：`environment/comm_model.py:11-23`。
- A2A transfer：`environment/comm_model.py:38-58`。
- executor 对所有 task 都加 upload：`environment/task_execution.py:558-586`。
- executor 未建模最终回传：`environment/task_execution.py:699-722`。

### legacy cache/request pipeline

- Env legacy 分支：`environment/env.py:176-184`，`environment/env.py:243-260`。
- UE 旧 request：`environment/user_equipments.py:90-102`。
- UAV 旧 cache/request：`environment/uavs.py:49-55`，`environment/uavs.py:102-260`。

## 5. 新主线建议调用链

建议 clean mainline 调用链如下：

1. `Env.reset(seed)`
   - 设置随机种子；
   - episode 级重采样一个圆形热点区域，半径先 200m；
   - 初始化 100 个 UE，普通行人速度；
   - 初始化 8 个 UAV，位置均匀随机部署；
   - 清空 DAG manager、executor、graph builder cache、metrics logger。

2. `Env.step(actions)`
   - 应用 UAV movement action，记录 movement energy；
   - 更新 UE 位置；
   - 判断 UE 是否在热点区域；
   - 对进入热点区域的 UE 执行一次吸引判定；
   - service-waiting UE 以 `0.2x` 速度移动；
   - `DAGTaskManager.observe_time_step(ues, hotspot_region, t)`：
     - 任意 UE 进入热点区域后 DAG 到达概率提高 2x；
     - UE 若已有 active DAG，不生成新 DAG；
     - 新 DAG 固定 `source_pos_at_arrival`；
     - 新 DAG 采样 `bandwidth_up/down`、任务属性、DAG 依赖。
   - `HeteroGraphBuilder.build(...)`：
     - active unfinished tasks；
     - DAG dependency edges；
     - K-hop local structure；
     - 每 5 step 更新一次 attribute hyperedges；
     - task-UAV feasible edges 不用 deadline 过滤。
   - `AssignmentPolicy / heuristic / MAPPO` 选择 task-UAV assignment。
   - `PhaseOneTaskExecutor.advance_one_slot(...)`：
     - entry upload；
     - predecessor cross-UAV transfer；
     - compute；
     - exit return to current UE position；
     - DAG 完成后释放 UE service-waiting。
   - `Reward/Metrics`：
     - reward 默认基于子任务增量、DAG 完成、flowtime/energy 的轻量项；
     - movement energy 只记录；
     - deadline 只在 QoS eval 中计算完成率。

## 6. 第一批最小可改文件清单

第一批目标不是让完整训练最优，而是让 clean env 主线可运行、可测、语义闭合。

建议第一批只改这些文件：

1. `config.py`
   - 收口 clean mainline 参数；
   - 保留旧参数兼容但标记 deprecated；
   - 不一次性删除所有旧常量，避免 import 崩溃。

2. `environment/user_equipments.py`
   - 实现 UE clean 状态机；
   - 保留旧 `generate_request()` 但标记 deprecated。

3. `environment/dag_tasks.py`
   - 重写 DAG 生成、固定 source_pos、active DAG 限制、任务属性生成、依赖生成。

4. `environment/comm_model.py`
   - 增加 clean DAG 通信接口，不先删除旧接口。

5. `environment/task_execution.py`
   - 重写候选估计和执行统计；
   - 去掉默认 deadline 过滤；
   - 增加 DAG 完成回调。

6. `environment/graph_builder.py`
   - 先产出 clean snapshot；
   - 可暂时保留旧字段为空，降低 HGNN/MAPPO 迁移风险。

7. `environment/env.py`
   - 重排 reset/step 主流程；
   - 建立 clean metrics 输出。

8. `environment/uavs.py`
   - 最小改 movement energy logging；
   - 不第一批删除旧 cache/request 逻辑。

暂不第一批改：

- `marl_models/hgnn/`
- `marl_models/mappo/`
- `scripts/train_assignment_mappo.py`
- `scripts/static_scheduler_compare.py`
- `scripts/pretrain_score.py`

原因：这些文件高度依赖旧 snapshot 和旧特征维度。先让 clean env smoke pass，再迁移模型入口，风险更低。

## 7. 删除风险最高的文件

1. `config.py`
   - 风险最高。几乎所有模块 import 常量。直接删旧常量会造成 import-time failure。

2. `environment/graph_builder.py`
   - Snapshot schema 被 HGNN scheduler、MAPPO assignment 直接消费：`marl_models/hgnn/scheduler.py:62-178`，`marl_models/mappo/assignment_mappo.py:82-190`。
   - 不能先删旧字段，建议先空字段兼容，再逐步迁移。

3. `marl_models/hgnn/scheduler.py`
   - Env 可直接加载 `PhaseOneGraphScheduler`：`environment/env.py:11`，`environment/env.py:88-97`。
   - 训练/预训练脚本也依赖它。

4. `environment/task_execution.py`
   - Env、graph builder、MAPPO sequential assignment 都依赖 executor 的候选逻辑。
   - 如果 schedule/result dataclass 字段变更，脚本日志和 supervision 会同步断。

5. `environment/uavs.py`
   - 虽然旧 cache/request 应废弃，但 Env 初始化仍直接使用 `UAV`：`environment/env.py:64-75`。
   - 不应整文件删除。

6. `scripts/static_scheduler_compare.py`
   - 看似脚本，但 `scripts/train_assignment_mappo.py` import 其中工具函数：`scripts/train_assignment_mappo.py:18-22`。
   - 直接删除会影响训练入口。

7. `marl_models/mappo/assignment_mappo.py`
   - actor 输入维度和旧 graph encoding 绑定：`marl_models/mappo/assignment_mappo.py:45-48`，`marl_models/mappo/assignment_mappo.py:176-190`。
   - 必须等 clean graph feature 维度稳定后再改。

## 8. smoke test 设计

smoke test 目标：验证 clean env 语义闭合，而不是验证学习效果。

### A. import/reset smoke

检查：

- `import config`
- `from environment.env import Env`
- `env = Env(); obs = env.reset()`
- `len(env.uavs) == 8`
- `len(env.ues) == 100`
- 热点区域存在且 episode 内固定；
- UAV 位置在 `[0, 700] x [0, 700]`。

### B. hotspot/UE mobility smoke

构造一个 UE 放入热点区域，强制 DAG 到达概率为 1。

检查：

- UE 进入热点后 DAG 到达倍率生效；
- UE 只有一次吸引判定；
- service-waiting 期间速度比例为 `0.2x`；
- DAG 完成后 UE 释放并恢复普通随机移动。

### C. DAG generation smoke

强制生成一个 DAG。

检查：

- task 数在 5-8；
- level 数不超过 4；
- 非入口任务父节点数在 1-3；
- 父节点来自所有前序层，不限相邻上一层；
- 图无环；
- `source_pos_at_arrival` 在 UE 后续移动后保持不变；
- DAG 有固定 `bandwidth_up/down`。

### D. communication/execution smoke

构造一个包含 entry、middle、exit 的小 DAG。

检查：

- entry task 计算 UE-UAV upload；
- middle task 不重复 UE upload；
- 父任务和子任务分配到不同 UAV 时产生 A2A transfer；
- exit task / DAG 完成时按 UE 实时位置计算 return；
- 记录 upload/inter_uav/compute/return 分项 delay；
- 记录 compute/communication/return energy；
- movement energy 只出现在 diagnostics，不进入默认 reward。

### E. graph smoke

在 active DAG tasks 存在时构图。

检查：

- snapshot 包含 dependency edges；
- snapshot 包含 K-hop 结构；
- attribute hyperedges 每 5 step 更新一次；
- attribute 聚类对象是全局 active unfinished tasks；
- 不产生 service-domain/spatial-density/resource/critical-support/candidate-scarce/task-type 旧主线超边。

### F. reward/metrics smoke

运行 5-10 step。

检查 metrics 至少包含：

- `average_dag_flowtime`
- `completed_dag_count`
- `dag_throughput`
- `deadline_completion_rate`
- `average_task_execution_delay`
- `total_task_energy`
- `uav_computation_utilization`
- `average_uav_queue_length`
- `load_balance_across_uavs`
- `average_episode_reward`
- `action_executed_rate`
- `invalid_assignment_rate`
- `uav_movement_energy`
- `qos_deadline_completion_rate`

检查 reward：

- 默认 reward 不使用 deadline；
- 默认 reward 不使用 movement energy；
- deadline 只进入 QoS 指标。

### G. short CLI smoke

第二阶段重构后建议提供一个最小脚本或 pytest：

```bash
python -m pytest tests/test_clean_env_smoke.py -q
```

测试内容只跑 1 个 episode、10 step、固定 seed，不训练模型。

## 9. 推荐下一步

下一阶段建议先做 `clean env`，不要同时改训练脚本和模型：

1. 先重写 `config.py`、`user_equipments.py`、`dag_tasks.py`、`comm_model.py`。
2. 再重写 `task_execution.py` 和 `graph_builder.py`，让 smoke test 通过。
3. 最后重排 `env.py` 的 step/reset。
4. 等 clean env 稳定后，再迁移 `marl_models/hgnn/` 和 `marl_models/mappo/`。

这样做的核心原因：当前模型层与旧 graph schema 强绑定。如果先动 HGNN/MAPPO，会在环境语义尚未闭合时引入维度、字段、训练入口三类错误，调试成本会明显变高。
