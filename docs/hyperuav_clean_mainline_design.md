# HyperUAV Clean Mainline Design

本文档记录 HyperUAV clean 第一版主线方案。该主线目标是：在用户随机移动并形成高任务密度热点的动态场景下，通过任务超图建模 DAG 子任务之间的高阶关系，帮助多 UAV 学习移动与卸载策略，从而缩短 DAG 任务从到达到最终结果返回 UE 的整体处理时间，并提升 DAG 任务卸载效率。

## 1. 论文主线与场景设定

核心贡献是：

```text
超图辅助多 UAV DAG 任务卸载
```

UAV movement 用于支撑动态 UAV 服务位置调整，使 UAV 更好服务当前任务分布和热点场景；它不是单独的飞行控制贡献，也不做连续飞行-通信联合物理建模。

clean 第一版场景固定为：

```text
地图大小：500m x 500m
UAV 数量：5
UE 数量：60
slot 时长：5s
hotspot 半径：150m
hotspot 每个 episode 采样一次
hotspot 在 episode 内固定
UAV 高度固定
UAV 只做二维水平移动
通信距离第一版使用二维水平距离
本地执行第一版不加入
```

UE 在每个 slot 开始时移动一次。hotspot 影响 UE 的 DAG 到达概率，但不直接吸引 UAV，也不是固定用户身份标签。

UE service-waiting state 定义为 UE 已经产生一个 active DAG 且该 DAG 尚未完整完成并返回结果的等待状态。UE 在 service-waiting 期间不再产生新的 DAG；当其 active DAG 完整完成且所有 sink 输出返回 UE 后，UE 退出 service-waiting。service-waiting 是环境生命周期状态，不进入 clean 第一版 reward，也不作为 deadline / QoS / slack 逻辑。

`deadline / QoS / slack` 不进入 clean 第一版算法主线：不进入状态、动作、图结构、排序、mask、critic、reward 和主实验指标。代码可以保留独立、默认关闭的 evaluation-only 指标计算，用于在存在合理 deadline 数据时统计 DAG deadline satisfaction 和 tardiness；若当前不存在有依据的 deadline 生成规则，第一轮实现不强制增加该评价。

总体算法结构采用：

```text
一个任务超图表示模块
两个 actor head
一个 centralized critic
```

movement actor、offloading actor、centralized critic 与任务超图表示模块共同构成一个联合策略。所有 UAV 共享同一个 movement actor；所有 ready task 共享同一个 offloading actor。UAV 数量从 config 读取，不在网络或训练代码中写死为 5。

## 2. 任务超图与 HGNN 表示

任务超图表示模块第一版采用传统 incidence-matrix-based HGNN。每个 slot 中，当前 active unfinished tasks 构成超图节点集合：

```text
V_t = active unfinished tasks
```

无向超边集合为：

```text
E_t = E_DAG^t ∪ E_khop^t ∪ E_attr^t ∪ E_part^t
```

四类超边分别为：

```text
E_DAG^t：DAG 二元依赖边转换得到的 size=2 无向超边
E_khop^t：k-hop 邻域超边
E_attr^t：属性相似超边
E_part^t：KaHyPar partition 超边
```

根据节点与超边的包含关系构建节点-超边关联矩阵：

```text
H[v, e] = 1  if task node v belongs to hyperedge e
H[v, e] = 0  otherwise
```

HGNN 基于该关联矩阵执行节点-超边-节点消息传递，输出每个 active task 的 task embedding：

```text
task_features + incidence matrix H -> task embeddings
```

超图表示层不显式建模 DAG 边方向。原始 DAG 依赖边进入 HGNN 时被转换为 size=2 的无向超边，因此 incidence matrix 本身不能区分父任务到子任务的方向。DAG 方向性通过两条路径保留：

```text
1. task features 中包含方向相关节点特征：
   is_entry
   is_sink
   topological_level
   parent_count
   child_count

2. 环境逻辑保留原始有向 DAG，用于：
   ready task 判断
   父子依赖释放
   critical path 计算
   reward 结算
```

将有向 DAG 依赖边转换为无向 size=2 超边，只影响 HGNN 的消息传递表示，不改变任务实际执行顺序，也不允许子任务绕过父任务依赖。

任务编号采用双编号：

```text
全局 task id：
用于日志、历史摘要、跨 slot 追踪、debug、缓存超边。

局部 task index：
用于当前 GraphSnapshot、task feature matrix、hyperedges、KaHyPar 输入、HGNN 输入。
每个 slot 根据当前 active unfinished tasks 重新 remap。
```

### 2.1 k-hop 超边

k-hop 超边表达原始 DAG 拓扑中的局部多任务依赖关系。k-hop 关系在 DAG 生成时基于原始 DAG 拓扑预计算，并以全局 task id 缓存。每个 slot 仅根据当前 active unfinished tasks 过滤非 active 成员，删除规模不足的超边，并 remap 到局部 task index。不在每个 slot 的 active induced DAG 上重新搜索 k-hop。

### 2.2 属性相似超边

属性相似超边表达任务静态或慢变化属性相似性，基于如下属性生成：

```text
input_data_size
output_data_size
num_operation
base_upload_bandwidth
base_download_bandwidth
```

不用于属性相似聚类的动态信息：

```text
is_ready
当前 source position
当前 UAV 距离
实时通信速率
队列状态
临时容量
reservation state
```

这些动态信息分别通过 task features、dynamic UAV features、pair features 和 candidate mask 提供给 actor。这样 attribute hyperedge 可以跨多个 slot 缓存，不会因为动态属性过期导致超边语义不一致。

属性相似超边可按固定 interval 或 active task 集合显著变化触发更新。若当前 slot 有新 DAG 到达，则必须触发 attribute similarity 超边更新。

### 2.3 KaHyPar partition 超边

KaHyPar partition 超边属于 clean 主方法定义中的超边类型。KaHyPar 输入固定为：

```text
k-hop 邻域超边 + 属性相似超边
```

不放 DAG 二元边。

新 DAG 到达时，attribute similarity 更新完成后，必须触发 KaHyPar 重新划分。更新顺序固定为：

```text
新 DAG 到达
-> 更新 active task 集合和局部编号
-> 更新 attribute similarity 超边
-> 使用最新 k-hop + attribute 超边运行 KaHyPar
-> 将 KaHyPar partition 结果转换为全局 task id 形式并缓存
-> 构建当前 GraphSnapshot
-> HGNN forward
```

KaHyPar 在当前 slot 的 active-task 局部编号映射，以及 `E_khop^t ∪ E_attr^t` 构成的基础超图上运行。KaHyPar 输出 `E_part^t` 后，再构建最终 GraphSnapshot。划分结果生成后，必须通过当前 `idx_to_task_id` 映射转换回全局 task id，再写入 partition cache。后续 slot 使用缓存时，根据当前 `task_id_to_idx` 过滤并 remap，禁止直接跨 slot 复用旧的局部 task index。

KaHyPar 按 `partition_update_interval` 或事件触发更新；新 DAG 到达时强制触发；未触发的中间 slot 使用全局 task id 缓存过滤并 remap。

正式主方法实验应启用 KaHyPar，并记录实际使用情况。工程降级规则为：

```text
成功重新划分：使用新 partition 并更新缓存
失败但有有效缓存：使用缓存过滤/remap
不可用且无缓存：本 slot 不生成 partition 超边
```

如果 KaHyPar 不可用或经常失败，该实验不能静默声称为完整主方法实验，应标记为 no-KaHyPar / degraded。

### 2.4 GraphSnapshot 边界与更新频率

GraphSnapshot 每个 slot 刷新：

```text
active / ready / pending 标记
task features
当前有效超边
全局 task id 与局部 task index 映射
incidence matrix 或可构造 incidence matrix 的超边列表
```

不属于任务 GraphSnapshot 的动态卸载特征：

```text
UAV features
pair features
candidate mask
profiling results
metrics
reward components
```

HGNN 每个 slot 对当前 GraphSnapshot forward 一次，得到当前 slot 固定 task embeddings。task embeddings 在 slot 内固定；movement 后不重跑 HGNN；顺序 offloading 中更新的是 dynamic UAV features、pair features、candidate mask 和 temporary reservation state。

当 active unfinished tasks 为空时，构造合法空 GraphSnapshot；不运行需要非空 incidence matrix 的 HGNN 消息传递；task embedding tensor 为空；task mean pooling、ready context 和 pending context 使用零向量；`M_t = 0`，不产生 offloading action；movement actor 仍可根据 UAV 自身状态产生动作。

## 3. Slot 时间流程

clean 第一版采用离散 slot 执行流程：

```text
决策准备阶段
-> movement 阶段
-> 顺序 offloading 阶段
-> executor 推进与 reward/metrics 结算阶段
```

每个 slot 从上一 slot 执行后的内部状态 `x_t^-` 开始。当前 slot 开始时先完成 UE 移动、DAG 到达和 ready 状态刷新，之后形成环境决策状态 `s_t`。GraphSnapshot 和 HGNN forward 是基于 `s_t` 构造当前决策 observation / encoding，不属于额外环境状态转移。

当前 slot 执行结束后得到 `x_{t+1}^-`，它不是 `s_{t+1}`。真正的 `s_{t+1}` 在下一 slot 完成 UE 移动、DAG 到达和 ready 刷新后形成，随后构图并执行 HGNN forward，得到计算 `V(s_{t+1})` 所需编码。

PPO transition 语义为：

```text
s_t
-> joint action a_t
-> reward r_t
-> x_{t+1}^-
-> next slot exogenous update
-> s_{t+1}
```

TD / GAE bootstrap 使用：

```text
δ_t = r_t + γ(1 - terminated_t) V(s_{t+1}) - V(s_t)
```

自然终止 `terminated=True` 时不 bootstrap。固定 horizon 截断 `truncated=True` 但可构造下一决策状态时，GAE 仍使用 `V(s_{t+1})` bootstrap。

### 3.1 决策准备阶段

当前 slot 的准备阶段顺序为：

```text
1. 从上一 slot 的 x_t^- 开始
2. UE 执行一次边界移动，得到当前 slot 的 UE service position
3. UE 在当前 slot executor 推进期间位置保持不变
4. 根据 UE 当前 slot 位置和 hotspot 判断 DAG 到达
5. 新 DAG 到达后，生成 DAG 与子任务
6. 根据上一 slot completion 结果和新到达 DAG，刷新 task lifecycle / ready 状态
7. 冻结当前 slot 初始 ready 集合 R_t
8. 按稳定排序得到 ready task 决策顺序
9. 基于 s_t 构建当前 GraphSnapshot
10. HGNN forward 一次，得到当前 slot 固定 task embeddings
11. 构造 centralized critic 输入并计算 V(s_t)
```

ready 集合冻结后，本 slot 内 executor 推进期间新变 ready 的任务，不进入本 slot offloading 决策。

UE 在每个 slot 开始移动一次。移动完成后得到当前 slot 的 UE service position `p_UE^t`，在当前 slot executor 推进期间保持不变。当前 slot 的上传、下载和 sink return 均使用该固定 UE 位置。下一 slot 开始时 UE 才再次移动。对于跨多个 slot 进行的 sink return，每个新 slot 可以根据该 slot 开始后的最新 UE 位置重新计算通信状态，但同一 slot 内不连续更新 UE 位置。

### 3.2 Movement 阶段

movement 采用“先移动、后服务”的离散时间抽象。对每架 UAV：

```text
pre_move_position = slot 决策前 UAV 位置
movement action -> service_position
```

movement actor 输出：

```text
a_i^move(t) ∈ {hover, +x, -x, +y, -y}
```

动作映射为固定速度大小 `V_UAV` 下的二维速度向量：

```text
hover -> (0, 0)
+x    -> ( V_UAV, 0)
-x    -> (-V_UAV, 0)
+y    -> (0,  V_UAV)
-y    -> (0, -V_UAV)
```

当前 slot 的 UAV service position 定义为：

```text
p_{i,srv}^t = p_{i,pre}^t + v_i^t * Δt
```

offloading pair features、candidate mask 和 executor 通信模型均使用 `p_{i,srv}^t`。slot 结束后：

```text
p_i^{t+1,-} = p_{i,srv}^t
```

第一版不建模 movement 对当前 slot service duration 的压缩。该抽象用于动态服务位置调整，不作为连续飞行控制贡献。movement energy 在 movement 阶段只计算并暂存为 `E_move^t`，总 reward 在 reward 结算阶段统一生成。

### 3.3 顺序 offloading 阶段

当前 slot 的 ready task 集合为：

```text
R_t = (j_1, j_2, ..., j_|R_t|)
```

排序必须稳定且确定。第一版排序键为：

```text
(dag_arrival_time, dag_id, topological_index, task_id)
```

该排序不使用 critical path、任务计算量、距离或其他调度启发式，避免把人工优先级混入环境。

如果当前任务没有任何合法候选 UAV：

```text
不调用 offloading actor
不产生 action
不产生 log probability
不写入 assignment_buffer
不改变权威任务生命周期状态
任务保持 READY_UNSCHEDULED
等待下一 slot
```

因此：

```text
|R_t| = slot 开始冻结的 ready task 数
M_t = 本 slot 实际产生 offloading policy action 的任务数
0 <= M_t <= |R_t|
```

顺序 offloading 阶段写入 `assignment_buffer` 时，只更新临时 reservation state，不立即修改任务的权威 lifecycle state，不立即进入 `IN_SERVICE`，不立即提交 executor。所有决策结束后，`assignment_buffer` 统一提交 executor；只有 executor 接受提交后，任务才正式从 `READY_UNSCHEDULED` 转为 `IN_SERVICE`。

### 3.4 Executor 与结算阶段

executor 使用当前 slot 的 UAV service positions、UE service positions 和最终 assignment / reservation 结果推进一个 slot，包括通信、排队、计算、sink return 和任务状态更新。executor 更新：

```text
compute_finish_time
return_finish_time
reward_completion_time
task lifecycle state
DAG completion
UE/DAG release 状态
```

随后构造当前 slot 新进入 reward 结算的任务集合 `C_t` 和新完成 DAG 数 `N_completed_DAG(t)`，统一计算 reward components 并写入 metrics，最后生成执行后内部状态 `x_{t+1}^-`。

## 4. 任务生命周期

第一版生命周期状态固定为：

```text
WAITING_DEPENDENCY
READY_UNSCHEDULED
IN_SERVICE
RETURNING
COMPLETED
```

`IN_SERVICE` 内部阶段由独立字段区分：

```text
service_phase ∈ {UPLOADING_OR_TRANSFERRING, QUEUED, COMPUTING}
```

定义：

```text
active unfinished tasks = 已到达系统且未 COMPLETED 的任务
ready task = READY_UNSCHEDULED，依赖满足，尚未分配，允许进入 offloading 决策
IN_SERVICE = 任务已经正式提交 executor，正在输入通信、跨 UAV 传输、排队或计算
RETURNING = sink 已计算完成，但输出尚未返回 UE
COMPLETED = 非 sink 计算完成，或 sink 输出返回 UE
```

`RETURNING` sink 仍属于 active unfinished task，但不能再次进入 ready 集合，不能再次被调度，不能提前 reward-completed，也不能提前触发 DAG completion。

movement actor 中的 pending task 集合定义为：

```text
P_t = active unfinished tasks - R_t
```

`P_t` 在 movement 决策前定义为当前 active unfinished tasks 中不属于冻结 ready 集合 `R_t` 的任务，包括等待依赖、已分配或服务中、排队中、计算中、返回中。本 slot 因无合法候选 UAV 而未产生 offloading action 的任务，仍属于当前冻结的 `R_t`，保持 `READY_UNSCHEDULED`，并在下一 slot ready 刷新时重新进入 `R_{t+1}`。

## 5. Movement Actor

movement actor 用于在每个 slot 内为每架 UAV 选择当前 slot 的服务位置。所有 UAV 共享同一个 movement actor。对每架 UAV `i`，movement actor 输出 5 个 movement action logits，对应：

```text
{hover, +x, -x, +y, -y}
```

boundary action mask 不拼接进网络输入，而是在 logits 输出后、构造动作分布前直接应用。`hover` 始终合法。boundary mask 根据动作对应的候选 service position 是否位于地图边界内生成，并单独记录在 rollout 中用于 PPO 重新计算 log probability。不能通过执行后 clipping 替代 action mask。

movement actor 网络输入为：

```text
o_i^move =
[
  u_i
  || c_i^ready
  || c_i^pending
  || n_ready_normalized
  || n_pending_normalized
]
```

UAV 自身状态 `u_i` 第一版固定为已归一化字段：

```text
pre_move_position
current queue length
current remaining capacity
current available time
current queued workload
```

movement actor 采用 UAV-specific 单头缩放点积 cross-attention。对 UAV `i`：

```text
q_i = W_q u_i
K_ready = H_ready W_k
V_ready = H_ready W_v
alpha_i^ready = softmax(q_i K_ready^T / sqrt(d_k))
c_i^ready = alpha_i^ready V_ready
```

pending 集合同理。第一版 ready 和 pending attention 共享 `W_q, W_k, W_v`。两类集合均使用当前 slot HGNN 输出的固定 task embeddings。空集合 context 为与正常 context 相同维度的零向量。数量归一化使用固定参考尺度，例如配置中的最大 active task 数或明确的固定归一化常数，不能根据当前集合自身大小做无意义归一化。

attention weights 可选保存，用于后续解释 UAV movement actor 关注了哪些任务。默认不进入 reward，不进入 GraphSnapshot，不改变环境转移。

movement PPO loss 对当前配置中的 `N_UAV` 个实际 movement actions 取 mean。`N_UAV` 从 config 读取，不在网络或训练代码中写死为 5。

## 6. Offloading Actor

offloading actor 用于在 movement 阶段确定当前 slot UAV service positions 后，为当前 slot 冻结的 ready tasks 逐个选择执行 UAV。所有 ready task 共用同一个 offloading actor。

对第 `k` 个 ready task `j_k` 和候选 UAV `i`，构造 candidate input：

```text
x_{j_k,i}^{off} =
[
  h_{j_k}^t
  || u_i^{dyn,t,k}
  || p_{j_k,i}^{dyn,t,k}
]
```

其中：

```text
h_{j_k}^t：当前 slot HGNN 输出的固定 task embedding
u_i^{dyn,t,k}：第 k 次顺序决策时的动态 UAV 状态
p_{j_k,i}^{dyn,t,k}：当前 task-UAV pair feature
```

不加入 optional assignment summary。临时 assignment / reservation 对后续决策的影响通过动态 UAV 状态和 pair features 体现，避免重复计数。

动态 UAV 状态第一版固定包括归一化字段：

```text
UAV service position
temporary queue length
temporary remaining capacity
temporary available time
temporary queued workload
temporary slot assigned count
```

pair features 第一版固定包括归一化字段：

```text
estimated input / inter-UAV transfer time
estimated input / inter-UAV communication energy
estimated queue waiting time
estimated compute time
estimated compute energy
estimated incremental completion delay
estimated return time（非 sink 为 0）
estimated return energy（非 sink 为 0）
```

`estimated incremental completion delay` 表示当前任务分配给候选 UAV 后，按照当前临时 reservation state 和 executor 的时间模型，预计还需要多长时间达到该任务的 completion 定义。非 sink 的完成点为 compute finish，sink 的完成点为 return finish。

entry task 的输入数据源为当前 slot 的 UE service position。non-entry task 的输入数据源为各父任务计算输出当前所在的 UAV。父任务与候选 UAV 相同时，对应父任务输出不产生 inter-UAV transfer；父任务与候选 UAV 不同时，按照通信模型估计 inter-UAV transfer。存在多个父任务时，transfer time 和 transfer energy 的聚合规则必须与 executor 的实际多父任务通信规则完全一致。pair feature estimator、candidate mask、reservation state 和 executor 共用同一套通信与完成时间估计规则。

offloading actor 使用同一个共享 candidate scorer 处理所有 task-UAV 候选对：

```text
l_{j_k,i}^{t,k} = f_off(x_{j_k,i}^{off})
```

`f_off` 的参数在所有 ready task 和所有 UAV 候选之间共享。按照稳定的 `uav_id` 顺序拼接所有候选 logit，随后应用 candidate mask，并构造 categorical action distribution。

candidate mask、temporary reservation update 和 executor commit 必须共享同一套 assignment legality constraints：

```text
is_legal(task, uav, state_view, service_positions)
```

其中：

```text
顺序决策阶段：
state_view = 当前 temporary reservation state

executor commit 阶段：
state_view = 最终提交的 reservation / authoritative state
```

三处必须对容量、通信可达性、任务需求和 executor 接收约束采用完全相同的判定口径。正常情况下，进入 `assignment_buffer` 的 assignment 不应被 executor 拒绝；invalid assignment 非零时，应视为规则实现不一致。

对每个实际产生 offloading action 的 task，rollout 保存当次实际使用的 candidate feature tensor、candidate mask、selected UAV action、old log probability、entropy、task id、candidate uav id mapping 和 decision order `k`。禁止在 PPO 更新时根据已经变化的环境状态重新构造旧 observation。

如果 `M_t = 0`，不计算该 slot 的 offloading policy loss，不计算该 slot 的 offloading entropy term，该 slot 不进入 offloading action-level mean。日志可记录数值 0，但必须同时带有：

```text
offloading_action_count = 0
offloading_loss_valid = false
```

## 7. Centralized Critic 与 PPO

clean 第一版采用单一 centralized critic。critic 在当前 slot 的决策状态 `s_t` 上计算 slot-level value：

```text
V_\phi(s_t)
```

movement 与 offloading 是同一联合策略的两个动作头，不分别设置独立 critic。

critic 可以看比 actor 更完整的全局摘要，但不能看未来信息、动作后的完成结果、reward 结算结果或 `x_{t+1}^-`。第一版 critic 输入固定为：

```text
s_t^critic =
[
  Pool(H_task^t)
  || U_global^t
  || n_active_normalized
  || n_ready_normalized
  || n_pending_normalized
  || Q_summary^t
]
```

其中 `Pool(H_task^t)` 第一版使用 mean pooling；active task 为空则使用零向量。

`U_global^t` 按稳定 `uav_id` 顺序拼接每架 UAV 的以下归一化字段：

```text
pre_move_position
current queue length
current remaining capacity
current available time
current queued workload
```

`Q_summary^t` 固定为以下全局队列/负载摘要：

```text
mean queue length
max queue length
mean available time
max available time
total queued workload
```

数量归一化使用固定尺度。

critic 输入不包含：

```text
本 slot movement actions
本 slot offloading assignments
本 slot reward components
本 slot completed tasks
本 slot completed DAG count
future DAG arrivals
deadline / QoS / slack
```

当前 slot 的联合动作包括：

```text
a_t =
(
  {a_{i,t}^{move}} for all UAV i,
  {a_{j_k,t}^{off}} for actual offloading actions k = 1..M_t
)
```

movement actor 与 offloading actor 共享同一个 slot-level advantage `A_t`。`-A_t log π` 只作为策略梯度原理说明，正式实现采用 PPO clipped surrogate。

每个 slot 内先计算：

```text
L_move,t = mean over N_UAV movement actions
L_off,t = mean over M_t offloading actions, only valid if M_t > 0
```

PPO batch 内：

```text
movement loss = mean over valid slots
offloading loss = mean over slots with M_t > 0
```

`M_t = 0` 的 slot 不进入 offloading loss / entropy 分母，不能当成一个数值为 0 的训练样本。

movement 和 offloading 分别按单个 action 计算 PPO ratio。禁止将全部 movement/offloading 动作概率相乘后构造 joint PPO ratio。

总 actor loss：

```text
L_actor = λ_move L_move + λ_off L_off
```

第一版默认：

```text
λ_move = 1.0
λ_off = 1.0
```

critic value loss：

```text
L_value = 0.5 * (V_\phi(s_t) - R_t^target)^2
```

总 loss：

```text
L_total =
  L_actor
  + c_v L_value
  - c_ent_move H_move
  - c_ent_off H_off
```

HGNN 与 movement actor、offloading actor 和 centralized critic 在 PPO 更新中联合训练。actor loss 和 critic loss 均可以通过 task embeddings 向 HGNN 参数反向传播。因此 rollout 不能只保存 detached task embeddings。每个 slot 必须保存产生 task embeddings 所需的历史 GraphSnapshot：

```text
task feature matrix
incidence matrix / hyperedge structure
task id 与 local index 映射
```

PPO 更新时使用同一份历史 GraphSnapshot 重新执行当前 HGNN forward，不能使用已经变化后的环境状态重建历史图。`old log probability` 仍保存 rollout 时的数值，不重新用当前参数计算。

movement action record 保存：

```text
movement observation
movement mask
selected action
old log probability
entropy
UAV id
```

critic / slot record 保存：

```text
历史 GraphSnapshot
critic 使用的非图全局输入
V(s_t)
reward r_t
terminated
truncated
```

正式 evaluation 默认使用 masked argmax deterministic action。随机采样评估只能作为补充实验，并明确随机种子。

## 8. Reward

clean 第一版 reward 鼓励更快完成 DAG、降低任务执行能耗、约束 UAV 无成本移动，并在完整 DAG 结果返回 UE 后给予一次性完成奖励。

reward 不使用：

```text
deadline
QoS
slack
invalid assignment penalty
每 slot unfinished waiting penalty
```

当前 slot reward 为：

```text
r_t =
  - w_t * Σ_{j in C_t} c_j * norm_time(ΔT_j)
  - w_e * Σ_{j in C_t} norm_task_energy(ΔE_j)
  - w_m * norm_move_energy(E_move^t)
  + w_c * N_completed_DAG(t)
```

非 sink task：

```text
reward completion = compute_finish
```

sink task：

```text
reward completion = return_finish
```

因此非 sink task 在计算完成时进入 `C_t`；sink task 只有在输出成功返回 UE 后进入 `C_t`。

时间成本：

```text
ΔT_j =
  reward_completion_time_j - dag_arrival_time,
  if entry task

ΔT_j =
  reward_completion_time_j - max(parent compute_finish_time),
  if non-entry task
```

non-entry task 的时间起点始终是父任务 compute_finish_time 的最大值，不是父任务 reward_completion_time。

critical path 权重：

```text
c_j = 1.0  if task j is on critical path
c_j = 0.5  otherwise
```

critical path 使用 DAG 生成时基于原始 DAG 拓扑和 `num_operation` 计算得到的静态标记。该标记不随执行过程动态变化。

任务能耗：

```text
ΔE_j =
  input / inter-UAV communication energy
  + compute energy
  + I(j is sink) * return energy
```

`ΔE_j` 使用 executor 在任务生命周期内实际累计的通信、计算和回传能耗，不使用 offloading pair features 中的 estimated energy。每一笔能耗只能归属并结算一次，避免跨 slot 累计或多父任务传输时重复计数。

clean 第一版统一沿用当前 clean communication model 实际输出的能耗边界。reward、pair features、episode energy metrics 和 Energy per completed DAG 均使用同一口径，不另行混用 UAV-only 与 system-total energy。具体能耗边界在实现计划阶段通过读取当前 clean communication / execution code 验证并写入配置说明。

完整 DAG 完成定义：

```text
所有非 sink task 已计算完成
所有 sink task 输出均已返回 UE
```

DAG completion bonus 只在 DAG 首次满足完整完成定义的 slot 发放一次。

movement energy 每 slot 结算一次：

```text
r_move = - w_m * norm_move_energy(E_move^t)
```

invalid assignment 不进入 reward。如果 invalid assignment 非零，应优先检查 candidate mask、temporary reservation、assignment legality rule 和 executor commit validation，而不是通过 reward penalty 让策略自己学习。

任务和 DAG reward 均必须一次性结算。建议保留：

```text
task.reward_settled
dag.completion_reward_settled
```

固定 horizon 截断时，不额外添加 terminal reward 或 terminal energy penalty。尚未 reward-completed 的任务，其累计 task energy 不进入本 episode reward，但必须完整进入 episode energy metrics。已按 slot 结算的 movement energy 不重复结算。

reward 归一化调试初值：

```text
T_ref = TIME_SLOT_DURATION
E_task_ref = P_UAV_COMPUTE * TIME_SLOT_DURATION
E_move_ref = N_UAV * POWER_MOVE * TIME_SLOT_DURATION
```

首次训练参数可设为：

```text
w_t = 1.0
w_e = 0.1
w_m = 0.05
w_c = 2.0
```

这些只作为 bootstrap / smoke training 初值。正式训练前必须基于 random policy / heuristic policy 下的 reward component 分布固定 reference scale 和权重；正式训练期间 reference scale 不应持续在线变化。

## 9. Metrics 与 Evaluation

metrics 只用于训练诊断、实验评价和论文结果展示，不进入 RL state、actor input、critic input、reward、GraphSnapshot，也不改变环境转移。实现上使用独立轻量 `MetricsTracker / EpisodeStats` 汇总器。

第一版主指标：

```text
Average DAG flowtime
DAG completion rate
DAG throughput
Average critical-path task completion delay
Energy per completed DAG
```

DAG flowtime：

```text
DAG flowtime = dag_return_complete_time - dag_arrival_time
dag_return_complete_time = max(return_finish_time of all sink tasks)
```

Average DAG flowtime 只统计 completed DAG，必须同时记录 generated_DAG_count、completed_DAG_count、DAG completion rate 和 DAG throughput。

DAG completion rate：

```text
completed_DAG_count / generated_DAG_count
```

若 `generated_DAG_count = 0`，记为 `None / NaN`。

DAG throughput：

```text
completed_DAG_count / total_evaluation_time
total_evaluation_time = total executed slots * TIME_SLOT_DURATION
```

正式 evaluation 中包含 arrival phase 和实际执行的 drain phase。可以额外报告 DAG/slot，但主结果单位必须统一。

critical-path task completion delay：

```text
task completion delay = reward_completion_time - task_ready_time
```

`task_ready_time` 表示任务第一次满足全部依赖并进入 `READY_UNSCHEDULED` 的时间，只记录一次。任务因无合法候选 UAV 跨多个 slot 保持 `READY_UNSCHEDULED` 时，不得在后续 slot 重写 `task_ready_time`。

Energy per completed DAG：

```text
total_episode_energy / completed_DAG_count
```

其中：

```text
total_episode_energy =
task energy actually consumed in the episode
+ movement energy actually consumed in the episode
```

该分子包括已完成 DAG 的能耗、未完成 DAG 已经实际消耗的 task energy、全部 movement energy。若 `completed_DAG_count = 0`，记为 `None / NaN`，不强行记 0。

reward 诊断指标：

```text
episode reward
mean step reward
r_time
r_task_energy
r_move
r_DAG
```

系统与正确性指标：

```text
invalid_assignment_count
invalid_assignment_rate
action_executed_rate
offloading_action_count
movement action distribution
hover action ratio
mean UAV displacement per slot
movement energy total
movement energy per completed DAG
average UAV queue length
load balance CV
UAV computation utilization
```

正确性指标分母固定为：

```text
invalid_assignment_rate =
invalid_assignment_count / assignment_buffer_entry_count

action_executed_rate =
successfully_committed_assignment_count / assignment_buffer_entry_count
```

若 `assignment_buffer_entry_count = 0`，两项记为 `None / NaN`。正常情况下：

```text
invalid_assignment_rate = 0
action_executed_rate = 1
```

由于 sink task 的 completion 包含 return finish，原先容易歧义的 `Average task execution delay` 改为 `Average task completion delay`。若需要纯计算维度，可单独记录 `Average compute time` 和 `Average sink return time`，但不作为第一版主指标。

正式 evaluation 建议采用：

```text
arrival phase：正常产生 DAG，持续固定 slot 数
drain phase：停止新 DAG 到达，继续执行已有 DAG
termination：全部 active DAG 完成，或达到最大 drain horizon
```

训练 episode 不要求 drain。drain phase 是正式测试协议，用于减少固定 horizon 对 DAG flowtime 和 completion rate 的右删失偏差。正式测试应使用多个随机种子、多个独立 hotspot、独立 UE 移动轨迹、独立 DAG 到达序列，并报告均值和标准差或置信区间。

## 10. Profiling

profiling 默认关闭，只作为 debug / 实验开销分析 hook。profiling 不进入 GraphSnapshot、actor input、critic input、reward，也不改变环境转移。

profiling 最多统计：

```text
graph_build_time
attribute_update_time
kahypar_partition_time
hgnn_forward_time
sequential_offloading_total_time

active_task_count
ready_task_count
total_hyperedge_count
offloading_decision_count
```

计时口径：

```text
graph_build_time 表示完整 GraphSnapshot 构建总耗时。
attribute_update_time 和 kahypar_partition_time 是 graph_build_time 中的子阶段耗时。
未触发 attribute 或 KaHyPar 更新时，统一记录为 0。
```

实现上使用独立轻量 profiler helper 或 context manager，避免在核心流程中散落大量计时代码。若 HGNN 在 GPU 上运行，profiling 实现应注意 CUDA 异步计时问题；这是 debug 实现注意事项，不改变算法逻辑。

## 11. 必须边界与排除项

必须遵守的边界：

```text
deadline / QoS / slack 不进入 clean 第一版算法主线
本地执行不进入第一版
UAV 高度固定，第一版不做 z 轴移动
movement 是动态服务位置调整，不是飞行控制贡献
GraphSnapshot 只承载任务超图表示所需信息
task embedding 在 slot 内固定
无合法候选 UAV 时不强行动作
assignment legality constraints 必须在 mask / reservation / commit 中共享
任务生命周期状态唯一
RETURNING sink 不能提前完成
reward 与 metrics 必须分离
profiling 默认关闭
KaHyPar 主方法与工程降级分离
reward component 归一化尺度正式训练前固定
PPO 更新不能重建已变化的历史 observation
训练与测试分离
```

暂不进入 clean 第一版默认主线：

```text
本地执行
deadline / QoS / slack 主优化
z 轴移动
连续飞行-通信联合建模
movement 压缩 service duration
多 hotspot 动态刷新
hotspot UE 固定身份模型
deadline-driven candidate filtering
deadline-driven critical path
旧 preprocess / compute / aggregation 任务类型语义
旧 spatial density hyperedge
旧 service-domain hyperedge
旧 resource-competition hyperedge
旧 critical-support hyperedge
旧 candidate-scarce hyperedge
旧 selective HGNN scoring / fallback / guard 逻辑
```

第一轮实现不阻塞项：

```text
baseline 具体集合
完整消融实验矩阵
attention 可解释性指标
partition cut/connectivity 展示
不同 UAV 数量扩展实验
不同 DAG 负载扩展实验
网络层数和 hidden dimension 调参
更复杂 direction-aware HGNN
更复杂 reward terminal settlement
```

## 12. 第一轮实现最低完成条件

第一轮 clean 实现至少需要完成：

```text
1. config 场景参数切到 clean 第一版口径
2. slot 时序与 s_t / x_{t+1}^- / s_{t+1} 边界
3. UE / UAV service position 语义
4. task lifecycle 状态机
5. GraphSnapshot + incidence HGNN + 超边缓存
6. movement actor
7. offloading actor + sequential reservation
8. centralized critic + PPO rollout/update
9. reward 公式与一次性结算
10. MetricsTracker 主指标和诊断指标
11. invalid assignment 一致性检查
12. smoke tests 跑通若干 episode
```
