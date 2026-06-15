# zrj_3 Clean Mainline Specification

本文件用于锁定 `zrj_3 clean mainline` 的实验定义。后续所有 Phase 的代码修改都必须严格遵守该规格文件。

如果旧代码、旧注释、Phase 0 审计报告与该规格文件冲突，以该规格文件为准。

如果实现过程中发现规格不清楚，先停止并询问，不允许自行发挥。

## 1. 环境场景

* 地图大小：700m × 700m。
* UAV 数量：8。
* UE 数量：100。
* episode 长度：500 steps。
* 每个 episode reset 时：

  * UAV 均匀随机初始化在地图内；
  * UE 均匀随机初始化在地图内；
  * 随机生成 1 个圆形热点；
  * 热点半径为 200m；
  * 热点圆必须完整落在地图内；
  * episode 内热点固定不变。
* 热点只影响 DAG 到达概率和 UE 的 service-waiting 状态。
* 热点不直接吸引 UAV。
* 热点不直接改变 UAV 的移动策略。

## 2. UE 移动与热点行为

* UE 普通状态下按高斯马尔科夫行人速度移动。
* 如果当前项目没有合适的行人速度设定，则普通速度均值使用约 1.2 m/s。
* 如果 UE 当前位于热点区域内，则 DAG 到达概率提高为普通概率的 2 倍。
* 如果 UE 在热点内成功产生 DAG，则进入 service-waiting 状态。
* service-waiting 期间，UE 仍然移动，但速度为普通速度的 0.2 倍。
* 一个 UE 同时最多只能持有一个 active DAG。
* UE 如果已经有 active DAG，则不能再生成新的 DAG。
* 当该 UE 的 active DAG 完整完成，并且最终结果已经回传给 UE 后，UE 退出 service-waiting 状态，恢复普通移动。
* DAG.source_pos 固定为 DAG 生成时 UE 的位置。
* DAG.ue_id 必须保留，因为最终回传时需要根据 ue_id 查找 UE 的当前实时位置。

重要禁止项：

* 不要实现“hotspot UE 身份模型”。
* 不要把固定比例的 UE 标记为热点用户。
* 不要使用固定多个热点中心。
* 不要实现“热点吸引 UE”。
* 热点是地图区域，不是 UE 类别。

## 3. DAG 到达逻辑

* clean mainline 不使用 deadline 作为主线机制。
* deadline 不得进入默认 reward。
* deadline 不得进入默认 graph construction。
* deadline 不得进入默认 candidate filtering。
* deadline 不得用于定义 critical path。
* deadline 只允许作为后续 QoS 实验扩展字段，默认关闭。
* DAG 到达概率由 UE 当前是否位于 episode 热点区域内决定。
* 如果 UE 不在热点内，使用 `DAG_BASE_ARRIVAL_PROB`。
* 如果 UE 在热点内，使用 `DAG_BASE_ARRIVAL_PROB × DAG_HOTSPOT_ARRIVAL_MULTIPLIER`。
* `DAG_HOTSPOT_ARRIVAL_MULTIPLIER = 2.0`。
* 最终概率必须 clip 到 `[0, 1]`。

## 4. DAG 结构

* 每个 DAG 有 5 到 8 个子任务。
* 最大层数为 4。
* 每个非入口任务至少有 1 个父节点。
* 父节点可以从所有前序层任务中选择，不限定必须来自上一层。
* 每个任务最大父节点数为 3。
* 必须保证 DAG 无环。
* 必须保留原始有向 DAG 依赖边。
* DAG 依赖边用于：

  * 判断任务是否 ready；
  * 维护父子依赖；
  * 计算中间结果传输；
  * 构造 k-hop dependency hyperedge。
* 允许一个 DAG 有多个 sink task。
* DAG 完整完成的定义是：

  * 所有子任务已经完成；
  * 所有 sink task 的 output 已经回传给 UE。

## 5. 任务属性

### DAG 级属性

* 每个 DAG 生成时采样固定基础带宽档位。
* `base_upload_bandwidth_mbps ∈ {20, 50, 100}`。
* `base_download_bandwidth_mbps ∈ {50, 100, 200}`。
* 采样概率为 `{0.3, 0.5, 0.2}`。
* 采样后的带宽在 DAG 生命周期内固定不变。
* 第一版中，UAV-UAV 中间传输可以使用 `base_upload_bandwidth_mbps`，后续再扩展单独的 inter-UAV bandwidth。

### 子任务级属性

* `input_data_size_mb ~ Uniform[1, 30]`。
* `output_data_size_mb ~ Uniform[0.5, 20]`。
* `task_complexity` 分布：

  * `O(n)`：0.2；
  * `O(nlogn)`：0.7；
  * `O(nlog^2n)`：0.1。
* `task_constant` 从 1 到 10 均匀采样。
* `BASE_UNIT_BYTES = 10 KB`。
* `n = input_data_size_bytes / BASE_UNIT_BYTES`。
* `num_operation = complexity(n) × task_constant`。
* clean mainline 不再使用旧的 preprocess / compute / aggregation 三类任务语义。

## 6. 通信、执行与能耗

### 实际通信速率

* `effective_rate = base_bandwidth × distance_factor(d)`。
* `distance_factor(d) = 1 / (1 + (d / 100)^2)`。
* 数据大小单位使用 MB。
* 带宽单位使用 Mbps。
* 传输时间计算公式：

  * `transmission_time = data_size_MB × 8 / effective_rate_Mbps`。

### 通信流程

入口任务：

* UE 将 `input_data_size` 上传到被选择的 UAV。
* 上传完成后，UAV 计算该任务。

非入口任务：

* 必须等待所有父任务完成。
* 如果父任务执行 UAV 与当前任务执行 UAV 不同，则需要将父任务的 `output_data_size` 从父任务 UAV 传输到当前 UAV。
* 所有必要的中间结果传输完成后，当前 UAV 执行计算。

sink task / 出口任务：

* UAV 计算完成后，将 sink task 的 `output_data_size` 回传给 UE 当前实时位置。
* 回传完成后，该 sink task 的结果才算完成。

完整 DAG 完成：

* 所有子任务完成；
* 所有 sink task 输出均已经回传给 UE。

### 计算时间

* `compute_time = num_operation / UAV_compute_rate`。
* 第一版 UAV 可以使用同构计算能力，但必须保持配置化，便于后续扩展异构 UAV。

### 能耗

* `task_energy = compute_energy + communication_energy`。
* `compute_energy = compute_time × P_UAV_COMPUTE`。
* UE 到 UAV 上传能耗：

  * `upload_energy = upload_time × P_UE_TX`。
* UAV 到 UAV 中间传输能耗：

  * `inter_transfer_energy = inter_transfer_time × P_UAV_TX`。
* UAV 到 UE 回传能耗：

  * `return_energy = return_time × P_UAV_TX`。
* UAV movement energy 必须计算和记录。
* UAV movement energy 不进入默认主 reward，只进入 logging / diagnostics。

## 7. 图结构与超边

clean mainline 保留：

* 原始 DAG 有向依赖边；
* k-hop dependency hyperedge；
* attribute similarity hyperedge。

clean mainline 废弃或 deprecated：

* spatial density hyperedge；
* service-domain hyperedge；
* resource-competition hyperedge；
* critical-support hyperedge；
* candidate-scarce hyperedge；
* task-type hyperedge；
* 旧的复杂超边开关体系。

clean graph 开关收敛为：

* `ENABLE_DAG_DEPENDENCY_EDGES`
* `ENABLE_KHOP_DEPENDENCY_HYPEREDGES`
* `ENABLE_ATTRIBUTE_HYPEREDGES`
* `ATTRIBUTE_HYPEREDGE_UPDATE_INTERVAL`
* `ATTRIBUTE_HYPEREDGE_CLUSTER_NUM`
* `KHOP_K`

### attribute similarity hyperedge

* 聚类对象是当前全局 active unfinished DAG sub-tasks。
* 不包含 finished tasks。
* 不包含 future tasks。
* 属性超边每 5 steps 更新一次。
* 属性向量为：

```text
[base_upload_bandwidth_mbps,
 base_download_bandwidth_mbps,
 input_data_size_mb,
 output_data_size_mb,
 num_operation]
```

* KMeans 前必须归一化。
* `cluster_num = min(4, active_unfinished_task_num)`。
* 如果 `active_unfinished_task_num < 2`，不生成属性超边。
* 每次更新时覆盖旧的 attribute hyperedges。

## 8. Critical Path

* critical path 不得基于 deadline。
* critical path 不得基于 slack。
* critical path 定义为 DAG 拓扑中的最长路径。
* 第一版路径权重使用 `num_operation`。
* 后续版本可以加入通信估计。
* `TaskNode.is_critical_path` 必须根据该定义设置。
* reward 中：

  * 如果任务在 critical path 上，`c_i = 1.0`；
  * 如果任务不在 critical path 上，`c_i = 0.5`。

## 9. Reward

默认主 reward：

```text
r = - w_t × c_i × ΔT_i - w_e × ΔE_task + w_c × completed_DAG
```

定义：

* `ΔT_i` 在子任务完成时结算一次。
* `ΔE_task` 在子任务完成时结算一次。
* 不要对同一个未完成任务在每个 step 重复扣 delay。
* 入口任务：

  * `ΔT_i = task_finish_time - dag_arrival_time`。
* 非入口任务：

  * `ΔT_i = task_finish_time - max(parent_finish_time)`。
* `ΔE_task = 当前任务 compute energy + 当前任务 communication energy`。
* `completed_DAG` 是当前 step 新完成的 DAG 数。
* `completed_DAG reward` 只在 DAG 完整完成并最终结果回传的 step 结算一次。
* UAV movement energy 只记录，不进入默认 reward。
* deadline 不进入默认 reward。

## 10. 评价指标

### 主指标

* `Average DAG flowtime = mean(dag_return_complete_time - dag_arrival_time)`。
* `Completed DAG count`。
* `DAG throughput = completed_DAG_count / episode_steps`。
* `Average task execution delay = mean(task_finish_time - task_ready_time)`。
* `Total task energy = compute energy + upload energy + inter-UAV transfer energy + return energy`。

### 系统指标

* `UAV computation utilization = sum(actual_compute_time_by_uav) / (NUM_UAVS × episode_steps × TIME_SLOT_DURATION)`。
* `Average UAV queue length = 对所有 step、所有 UAV 的 queue length 求均值`。
* `Load balance across UAVs = per-UAV completed workload 的 coefficient of variation`。

### 辅助指标

* `Average episode reward`。
* `Action executed rate`。
* `Invalid assignment rate`。
* `UAV movement energy`：

  * total；
  * per-UAV；
  * ratio。
* `QoS deadline completion rate`：

  * 仅作为后续扩展；
  * 默认关闭；
  * 不作为主指标。

## 11. 清理规则

以下内容不能继续作为 clean mainline 逻辑：

* hotspot UE identity model；
* 固定比例 hotspot UE model；
* 固定多个 hotspot centers；
* 旧 preprocess / compute / aggregation 任务类型主线；
* deadline-driven reward；
* deadline-driven graph；
* deadline-driven teacher score；
* deadline-driven candidate filtering；
* spatial density hyperedge；
* 旧 service-domain / resource-competition / critical-support hyperedges；
* 旧 selective HGNN scoring / bounded guard / score fallback 主线。

删除策略：

* 删除旧代码前必须先搜索引用。
* 如果某个旧文件仍被入口脚本 import，不要直接删除。
* 如果暂时无法删除，则标记 deprecated。
* 可以在后续阶段迁移到 legacy，但必须先保证 import 不崩。
* 不要删除结果目录、checkpoint、日志或 `.git` 元数据。
* 不要自动 push。

## 12. 实现纪律

后续每个 Phase 必须遵守：

1. 先读取本规格文件。
2. 明确说明本阶段实现规格文件中的哪些章节。
3. 修改前列出本阶段将修改的文件。
4. 只修改本阶段允许修改的文件。
5. 修改后运行 py_compile 或 smoke test。
6. 输出是否偏离本规格文件。
7. 如果实现需要改变本规格文件，必须先停止并询问。
8. 不允许用旧代码注释、旧 config 名称或旧实验逻辑覆盖本规格文件。

严禁在未说明的情况下重新引入：

* deadline 主奖励；
* deadline 候选过滤；
* deadline critical path；
* hotspot UE 身份模型；
* preprocess / compute / aggregation 任务类型主线；
* 旧空间超边；
* 旧 service/resource/critical-support 超边。
