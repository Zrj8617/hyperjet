# HyperUAV 环境重构：新窗口启动上下文

> **性质：非权威上下文摘要，不是第二份 roadmap、正式 spec 或实施结果。**
>
> **创建日期：2026-10-13。**
> 新窗口必须重新读取仓库根 `AGENTS.md`、`docs/research/HyperUAV_research_master_roadmap.md` 和 `docs/superpowers/specs/2026-10-10-hyperuav-env-rebuild-requirements-draft.md`。三者与本文冲突时，以 `AGENTS.md`、Living Research Charter、需求文档 §0.4 的顺序为准。

## 1. 当前任务是什么

用户准备从头重构 HyperUAV 的物理环境框架，主要原因是当前 `environment/` 同时混合了实体、仿真生命周期、通信、任务执行、reward、graph、诊断和历史分支，难以理解和继续修改。

本轮目标不是重做论文算法，也不是把师兄的 MRDTS 一比一复制过来，而是：

1. 参考 MRDTS 清楚的实体、配置、contract 和 simulator 分层；
2. 保留 HyperUAV 已确认的业务语义；
3. 在并行目录 `environment_v2/` 中重新实现；
4. 每完成一个 Gate 就停下，让用户和师兄审核后再进入下一 Gate；
5. 最终形成用户能够理解、调试和继续维护的清爽环境代码。

用户代码基础较弱，希望在每个 Gate 中同时理解：实体与 runtime、数据类与服务类、状态所有权、依赖方向、contract 和配置注入等基础概念。因此交付说明要结合 HyperUAV 的真实代码，不能只给抽象术语。

## 2. 当前仓库与参考仓库

### 2.1 本地编辑源

- 仓库：`D:\CodeFile\HyperUAV-qly`
- 当前分支：`param-realign-20260914`
- 本文创建时 HEAD：`d2214c99fe8fcbb1861a7bc2492b1e6157c51760`
- 工作树不是 clean，存在用户已有的 modified/untracked 文档、`output/` 和 `tools/` 等内容。
- 不得清理、覆盖、移动或顺手提交这些已有改动。

### 2.2 师兄的参考代码

- 本地：`D:\CodeFile\mrdts`
- 服务器：`/data2/zrj2025/mrdts`
- 已核查版本：`main`，HEAD `17b256c8de3c15da36f7c2140825b87270227a80`
- 主要参考目录：`env/types.py`、`env/contracts.py`、`env/settings/`、`env/entities/`、`env/communication/`、`env/workload/`、`env/runtime/`、`env/simulator.py`、`config/*.toml`。

MRDTS 只提供架构与部分公式参考。不要复制它的 Master/Member/BS 世界模型、每时隙任务失败语义、信道队列、`EntityRef`、大量内部校验或与 HyperUAV 无关的抽象。

## 3. 当前阶段和授权边界

需求文档已经经过多轮 Claude 修改和 Codex 只读复核。最后一轮复核结果是“没有阻塞问题”。需求文档 Gate 0 停止条件处已经记录：用户确认 §0.4，师兄同意先实现后评审，可以进入 Gate 1–2。

用户在本轮进一步明确“文档已经看过，准备开改”。下一窗口应从 **Gate 1** 开始，完成 Gate 1 后停止并提交审核材料；不要自动继续 Gate 2。

需求文档首页和末尾仍有早期的 DRAFT/未批准文字，那是历史状态，没有同步覆盖；实施时以需求文档 §0.4、Gate 0 的 2026-10-13 注记和用户本轮授权为准。不要为了修正文档状态而顺手修改需求文档。

## 4. 长期工程约束

1. 先读 `AGENTS.md` 和 Living Research Charter。
2. 本地是编辑源，服务器是执行副本，GitHub 是冻结版本。
3. 保护当前 dirty worktree；尤其不要碰 `output/`、`tools/` 和其他无关未提交文件。
4. 不执行 `git reset --hard`、`git checkout --` 或其他清理用户改动的操作。
5. 不写自定义 hash、SHA-256、配置指纹或模型指纹。
6. 不为基本不可能发生的 case 编写重复防御逻辑。
7. v2 只在两个外部入口校验：读取配置、接收算法动作。
8. 实体、通信和 runtime 内部断言为 0，不写内部 shape/type/finite 检查、失败回滚、fallback 或兼容层。
9. 不引入阶段对象、`EntityRef`、`validate_dag`、快照复制或 `freeze_ue_mobility`。
10. 每个 Gate 只做本 Gate 的职责，完成后停止等待用户和师兄审核。
11. 不在环境重构中顺便改 actor 数量、PPO、HGNN、reward 架构或训练逻辑。

## 5. 已确认的总体架构

新代码并行放在 `environment_v2/`，旧 `environment/` 在新环境跑通、正式入口切换并获得删除批准前保持不动。目标依赖方向为：

```text
types/settings
    ↓
entities
    ↓
communication/workload/runtime
    ↓
simulator
    ↓
method adapter
    ↓
RL/heuristic methods
    ↓
training/evaluation/experiments
```

环境内核不导入 actor、HGNN、trainer、reward 或实验脚本。状态所有权为：

- UE/UAV/DAG/Region 保存静态或物理实体数据；
- runtime 唯一拥有 `TaskStatus`、`DAGStatus`、UAV 队列、空闲时刻和执行记录；
- simulator 只协调生命周期，不复制第二份状态；
- method/agent 只读 snapshot、提交动作，不直接改环境内部状态。

## 6. 实体与字段决定

### 6.1 实体集合

- 实体只有 UE、UAV、DAG、Region；没有 BS、MBS、云或 Master/Member 分层。
- 子任务不是独立实体类，而是 DAG 内节点，用 `(dag_id, node_id)` 标识。
- UE 和 UAV 都采用集合写法：一个对象保存全部实例的数组。

### 6.2 UAV

- `positions` 形状为 `(N, 3)`，UAV 高度为固定配置值。
- UAV 编号固定等于数组下标。
- 不保存 `region_ids`，需要时由 `Region.get_region_ids(positions)` 现算。
- 飞行动作为每架 UAV 一个 `[-1, 1]^2` 连续二维向量，乘最大水平速度和飞行时长；越界则保持原位并返回越界标记。
- UAV 同构、单核；不设核数字段，也不保存逐 UAV 主频数组。主频为 `UAVConfig.cpu_frequency_hz` 标量。
- 不保存剩余能量、`slot_energy`、`coverage_radius` 或 Master/Member 字段。
- 计算能耗为 `kappa * cpu_cycles * frequency_hz^2`。
- 飞行能耗使用标准旋翼推进功率公式；悬停也消耗能量。不要采用 MRDTS 的 `u1`–`u5` 字段。

### 6.3 UE

- 集合字段为 `positions`、`speeds`、`headings`；不单独保存速度向量。
- 高斯—马尔可夫移动；每时隙位移为 `speed * move_duration_s`。
- 去掉旧 `UE_MAX_DIST` 和 `UE_GM_MAX_SPEED` 上限，只保留速度不小于 0 与地图边界反弹。
- 取消“有活动 DAG 时降速”。
- UE 每个时隙只在飞行阶段移动一次；之后的上传和回传使用更新后的位置。

### 6.4 Region

- 只有热点区和非热点区；热点为圆形。
- `center`、`radius_m`、`get_region_ids(positions)`。
- 热点中心每个 episode 重新采样。

### 6.5 DAG 与运行状态

DAG 静态数据主要为：

- `id`、`ue_id`、`arrival_time_s`、`num_tasks`；
- `input_bits[node]`；
- `cpu_cycles[node]`；
- `levels[node]`；
- `edges`；
- 与 `edges` 一一对应的 `edge_bits`；
- sink 节点的 `return_bits[node]`，非 sink 为 0。

前驱、后继和 sink 由 `edges` 推出，不重复保存。`num_operation` 只改名为 `cpu_cycles`，数值不重新缩放。

运行时状态：

- `TaskStatus`：`WAITING / READY / SCHEDULED / RETURNING / DONE / CANCELLED`；
- `DAGStatus`：`ACTIVE / COMPLETED / TIMED_OUT`；
- `DAGRuntime.finish_time_s` 只在 COMPLETED 时写入，表示最后一个 sink 回传完成时刻。

## 7. 配置决定

新环境使用六个 TOML 文件和六个 frozen dataclass settings：

1. `region.toml`：地图边长、热点半径、热点到达倍率；
2. `user.toml`：UE 数量、移动参数、`move_duration_s`、发射功率、基础到达概率；
3. `uav.toml`：UAV 数量、高度、最大速度、飞行时长、主频、电容系数、发射功率、队列上限、旋翼推进参数；不含核数；
4. `dag.toml`：任务数、层数、父节点数、输入/边/回传数据量与计算量参数；
5. `channel.toml`：MRDTS ChannelModel 需要的信道参数和三类链路带宽；
6. `scheduling.toml`：`slot_duration_s`、`arrival_slots`、`relative_deadline_slots`。

读取使用 Python 3.12 的 `tomllib`，每个 dataclass 从对应 TOML 段取出字典后直接 `cls(**data)`。缺键和多键由 Python 自然报错，不手写逐键校验。

只检查以下配置关系：

- 时隙长度大于 0；
- 飞行时长大于 0；
- 飞行时长不超过时隙长度；
- `relative_deadline_slots` 大于 DAG 最大层数。

地图、高度、速度、带宽、功率、数据量范围、主频、kappa、飞行时长、D 和到达窗口等具体数值仍属于“最后再定”。Gate 1 不做调参，也不要把临时默认值写成论文最终参数。

## 8. 随机流、到达和截止时间

### 8.1 随机流

`reset(seed, episode)` 从 `(seed, episode)` 派生四条独立 `np.random.Generator`：

- 场景：热点中心、UAV/UE 初始位置、UE 初始速度和方向；
- 移动：UE 高斯—马尔可夫移动；
- 到达：每个 UE 每个到达时隙的 Bernoulli 抽样；
- DAG 内容：DAG 结构和属性生成。

方法层和随机策略使用自己的 RNG，不得消费环境的随机流。新环境不使用全局 `np.random`，也不需要重写 exogenous tape。

### 8.2 到达语义 D'

- 到达窗口内，每个 UE 每个时隙都抽一次到达，与策略和当前活动 DAG 数无关。
- 每个 offer 立即成为活动 DAG，不做接收/拒绝；同一 UE 可以同时有多个活动 DAG。
- DAG id 按 offer 生成顺序递增。
- 热点内到达概率为基础概率乘热点倍率。

### 8.3 截止时间和 episode

- 所有 DAG 使用统一相对截止 `D = relative_deadline_slots`。
- 截止时刻为 `arrival_time_s + D * slot_duration_s`，落在时隙边界。
- 所有 sink 结果在截止时刻之前或恰好在截止时刻回到 UE，DAG 才是 COMPLETED。
- 截止时仍未成功则为 TIMED_OUT；未分配节点变为 CANCELLED。
- 已分配工作的承诺时间线不撤销、不抢占，相关服务能耗已经记账。
- 开始时刻不早于截止时刻的回传不安排；已经安排的回传继续，但 DAG 仍是 TIMED_OUT。
- 前 `arrival_slots` 个时隙产生到达，之后只收尾；所有 DAG 进入 COMPLETED/TIMED_OUT 后 `done=True`，最多再 D 个时隙。
- episode 不等待已承诺但尚未执行完的工作推进到 DONE。
- 主指标为按时完成率：`COMPLETED DAG 数 / offer 总数`。

## 9. 环境接口与时隙顺序

对外接口：

- `reset(seed, episode)`：初始化随机流、实体和热点，完成第一个时隙开头的到达与 ready 冻结，返回 snapshot；
- `begin_slot(flight_actions)`：执行 UAV 飞行和 UE 一次移动，返回供卸载决策读取的 snapshot；
- `end_slot(placements)`：提交 placement、推进到时隙末并结算，返回 `SlotResult` 和下一时隙 snapshot；`done=True` 时下一 snapshot 为 `None`。

时隙顺序：

```text
时隙开头产生到达（UE 不动）
→ 冻结 ready 节点
→ UAV 飞行 flight_duration_s，UE 同时移动一步
→ 新 placement 从 slot_start + flight_duration_s 开始排程
→ 已承诺时间线推进到时隙末
→ 判定 DAG 成功
→ 判定 DAG 超时
→ 未结束则准备下一时隙的到达和 ready snapshot
```

placement 是 `((dag_id, node_id), uav_id)` 列表，只能从本时隙冻结 ready 列表中选一个保持原顺序的子序列。未选节点保持 READY。环境不改变节点排序；ordering 智能体属于 Gate 9。

动作入口只校验：

- 飞行动作形状和值域；
- `uav_id` 存在；
- task 属于本时隙冻结 ready 集合且未重复；
- UAV 队列未满。

不合法 placement 不执行并计数。

## 10. 通信、路由和单服务台时间线

### 10.1 ChannelModel

全部采用 MRDTS 的三维 ChannelModel：3D 距离、仰角 LoS 概率、NLoS 衰减、`g0 / d^2` 和 Shannon 速率。带宽按地对空、空对空、空对地链路类型配置。删除旧 DAG 随机上下行带宽字段。

本轮没有信道排队。中转 UAV 不占计算服务台，可以同时转发多份数据；这是已知简化，不要在本轮擅自补信道队列。

### 10.2 路由

- 入口节点：UE → 当前离 UE 最近的接入 UAV → 执行 UAV；接入 UAV 与执行 UAV 相同时只有一跳。
- 前驱结果：前驱所在 UAV → 后继执行 UAV；同 UAV 不传。
- sink 回传：sink UAV → 安排回传时离 UE 最近的 UAV → UE；相同 UAV 时只有空对地一跳。
- 多跳采用存储转发，各跳时间相加；能耗按每跳发送方计算。

服务台占用：

- 上传和前驱传输占用执行 UAV；
- 回传占用 sink UAV；
- 中转 UAV 不占服务台。

### 10.3 FIFO 与回传的最终决定

- 每架 UAV 是单服务台。
- 接受 placement 时立即计算上传、前驱传输、排队和计算时间线。
- 分配 sink 时也必须把 UAV 空闲时刻推进到该 sink 的计算完成时刻，不能复刻旧代码中 sink 计算与后续任务重叠的错误。
- sink 计算完成所在时隙的末尾，才安排回传。
- 回传作为一个新的服务项排在该 UAV 已经分配的工作之后：`return_start = max(compute_finish, current_uav_available_time)`。
- 已排好的记录不回头顺延。
- 回传链路、时长和能耗在“安排回传”的时隙末按当时位置重算并冻结；即使真正服务开始更晚，也不再次重算。
- 若 `return_start >= deadline`，不安排回传。

估计器和执行器必须共用 executor 的纯 `plan()` 计算入口；`plan()` 返回分段时间与能耗，不修改 runtime 状态。

## 11. 能耗记账的最终决定

服务能耗采用“承诺时记账”，不是完成时记账：

- 接受 placement 时，记录该节点的计算能耗、上传和前驱传输能耗；
- 安排回传时，记录回传能耗；
- 任务实际完成时只更新状态，不再次累计；
- 飞行能耗仍是当前时隙实际飞行/悬停产生的能耗。

因此 `SlotResult` 的计算、上传与前驱、回传三类能耗表示本时隙新承诺的服务能耗；飞行能耗表示本时隙实际飞行能耗。该变化使逐时隙 reward 中的服务能耗更接近作出 placement 的时刻，但会改变折扣回报的时间分布。

如果后续某个 reward 明确要恢复“完成时扣能耗”，由 reward 层读取 `TaskRuntime` 自行处理，环境的记账语义不变。

## 12. Snapshot 与 SlotResult

- snapshot 不复制数组或 DAG 记录，直接提供只读视图；调用方承诺只读。
- `SlotResult` 不放 reward 和指标。
- `SlotResult` 包含：本时隙完成 DAG、超时 DAG、四类能耗、不合法 placement 数、offer 和 `done`。
- 完成 DAG 记录到达时刻与完成时刻。
- 超时 DAG 只记录到达时刻；截止时刻由到达时刻和 D 计算，不另存。
- offer 至少记录 DAG id、UE id 和到达时刻。

## 13. 方法层和智能体后置

当前环境设计不依赖最终智能体数量。师兄所说“两个智能体不够”指当前 movement/offloading 两个 actor 的设计问题，放到环境稳定后的 Gate 9。

Gate 7 才处理：

- graph/observation/reward/forecast/assignment adapter；
- `task_id` 改为 `(dag_id, node_id)`；
- 集合式 UE/UAV 访问；
- 删除 DAG 带宽特征；
- 输出数据量特征改为出边 `edge_bits` 之和；
- 删除 `remaining_energy_ratio`；
- reward 能耗归一化与新推进功率；
- 一个 UE 多个活动 DAG 的 graph/observation；
- 旧离散 movement actor 到连续环境动作的 5 行外部映射。

不要在 Gate 1–6 中提前修改这些方法层内容，也不要改 `scripts/train_clean_mainline.py`。

## 14. Gate 1 的准确范围

Gate 1 的目标是定义各层共享的最小语言，不实现完整仿真。

### 14.1 应交付

- `environment_v2/types.py`：最小 ID/value types、`TaskStatus`、`DAGStatus`；不要实现 `EntityRef`。
- `environment_v2/contracts.py`：movement action、placement、slot snapshot/result 的中立 contract；不包含 reward 或特定 actor 概念。
- `environment_v2/settings/`：六个 frozen dataclass 配置类和统一导出。
- `config/region.toml`、`user.toml`、`uav.toml`、`dag.toml`、`channel.toml`、`scheduling.toml`。
- 配置读取与已批准的四项配置关系校验。
- 用一条命令读取并打印全部配置，证明 TOML 与 dataclass 可以工作。

### 14.2 不应交付

- 不实现 UE/UAV/DAG/Region 实体；它们属于 Gate 2。
- 不实现 ChannelModel、routing、DAG generator、runtime、simulator 或方法 adapter。
- 不创建 contract tests；需求文档明确 Gate 1 只要求配置加载打印。
- 不修改旧 `environment/`、`config.py`、训练入口或算法代码。
- 不开始调参，不把临时 TOML 数值宣称为最终论文数值。
- 不提前进入 Gate 2。

### 14.3 Gate 1 可参考但不能照搬的文件

MRDTS：

- `D:\CodeFile\mrdts\env\types.py`
- `D:\CodeFile\mrdts\env\contracts.py`
- `D:\CodeFile\mrdts\env\settings\*.py`
- `D:\CodeFile\mrdts\config\*.toml`
- `D:\CodeFile\mrdts\tests\test_settings.py`

HyperUAV 旧值与字段来源：

- `config.py`
- `environment/env.py`
- `environment/uavs.py`
- `environment/user_equipments.py`
- `environment/dag_tasks.py`
- `environment/assignment.py`
- `environment/task_execution.py`

旧代码只能用于核实当前事实和暂定数值来源，不能把旧耦合结构搬进 v2。

## 15. Gate 1 完成时应返回什么

完成 Gate 1 后停止，不进入 Gate 2，并向用户报告：

1. 新增或修改的文件；
2. 每个类型、contract、配置类的职责；
3. 哪些数值只是暂定、来源是什么；
4. 配置加载命令及输出结果；
5. `git diff --check` 结果；
6. 当前分支、HEAD、`git status --short`，明确区分本次文件和用户原有 dirty 文件；
7. 需要用户或师兄在进入 Gate 2 前确认的问题。

不要提交、推送、清理工作树或更新 Living Charter，除非用户另行明确要求。

## 16. 新窗口交接提示词

下面内容可以直接复制到新的 Codex/Claude 窗口：

```text
你现在接手 HyperUAV 环境重构，工作目录是 D:\CodeFile\HyperUAV-qly。

先完整读取：
1. 仓库根 AGENTS.md；
2. docs/research/HyperUAV_research_master_roadmap.md；
3. docs/superpowers/specs/2026-10-10-hyperuav-env-rebuild-requirements-draft.md，尤其是 §0.4 和 Gate 1；
4. docs/superpowers/2026-10-13-hyperuav-env-rebuild-context-brief.md。

权威顺序是 AGENTS.md → Living Research Charter → 需求文档 §0.4；context brief 只是帮助理解，不是新的事实源。需求文档已经过多轮只读复核，最后结论是没有阻塞问题。Gate 0 已通过，用户已经明确表示准备开始修改。

本轮只实施 Gate 1：公共类型、单位、配置与 contract。完成后必须停止，等待用户和师兄审核，不要进入 Gate 2。

Gate 1 目标：
- 在并行目录 environment_v2/ 中建立最小 types.py、contracts.py 和 settings/；
- 新建六个 config/*.toml：region、user、uav、dag、channel、scheduling；
- frozen dataclass 用 tomllib 读取对应 TOML 段并 cls(**data) 构造；
- 只做需求文档批准的配置关系校验；
- 定义中立的 movement/placement/snapshot/SlotResult contract，不放 reward、不绑定 actor 数量；
- TaskStatus 和 DAGStatus 按需求文档定义；子任务标识是 (dag_id, node_id)，不要 EntityRef；
- 用一条命令加载并打印全部配置。不写 contract tests。

严格边界：
- 不修改旧 environment/、config.py、训练入口或算法代码；
- 不实现实体、通信、runtime、simulator；
- 不碰 output/、tools/ 和其他已有未提交文件；
- 不清理 dirty worktree，不 reset/checkout 用户改动；
- 不写 hash/SHA-256、assert、内部校验、fallback、兼容层、阶段对象或假想需求；
- 不调论文最终数值，TOML 数值如需暂定必须注明来源并在交付中列出；
- 不提交或推送，除非用户明确要求。

开始前先检查 branch、HEAD、git status，并只读核对 MRDTS 的 env/types.py、env/contracts.py、env/settings/ 和 config/*.toml，以及 HyperUAV 的 config.py 与相关旧环境字段。按仓库技能规则先形成最小 Gate 1 实施计划，再直接实现和验证；不要因为非关键细节停下来反复询问。

完成后返回：改动文件、设计说明、暂定数值及来源、配置加载结果、git diff --check、branch/HEAD/status，并明确已经停在 Gate 1。
```
