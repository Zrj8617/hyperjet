# HyperUAV 环境重构：需求文档

> 本文是环境重构的唯一方案文件，只写当前规则，交给 Codex 实现时以本文为准。修改历史见 git。偏离师兄 MRDTS 的地方用 ⚠️ 标出。

## 0. 总则与当前规则

### 0.1 工作约束

1. 本文只管环境重构；项目整体的研究状态仍以 `docs/research/HyperUAV_research_master_roadmap.md` 为准。
2. 分阶段推进：每个 Gate 先写设计说明 → 用户和师兄审核 → 用户批准 → 实现 → 结果说明 → 用户和师兄复核 → 用户批准后进入下一个 Gate。不自动推进；每个 Gate 只做本 Gate 的事，不顺带重构其他层。
3. 不过度防御、不写成屎山：
   - 不写哈希、SHA-256、文件或模型指纹校验；
   - 不为基本不可能出现的内部情况加保护。判断标准：说不出哪条真实调用路径会触发它，就不写；
   - 不为假想需求提前抽象：没有第二种真实实现时，不加 Protocol / ABC / Factory、事件总线、注册表、依赖注入、任意拓扑路由；
   - 一个事实只有一个实现：通信公式只有一套；任务状态只存一份；配置和实体不各存一份参数；observation 不改环境状态；simulator 和 runtime 不各有一套时间；
   - 不永久保留兼容层，过渡期最多一个薄 adapter；不堆 `legacy`、`compat`、fallback、feature flag。
4. 每个阶段交付时，用户只检查五个问题：这个模块能否一句话说清职责？同一个公式或状态是否在两处重复？是否新增了大量 legacy、compat、fallback 或 feature flag？是否出现哈希或不必要的校验？删除这个模块是否只影响一种明确功能？任何一个说不清，就不进入下一阶段。
5. 用户工作树里与本任务无关的未提交文件不清理、不覆盖、不顺手提交。旧代码的删除放在 Gate 8，用户明确批准后才做。
6. 正式实验按 `AGENTS.md` 记录 Git 分支、HEAD 和 dirty 状态；这是版本纪律，不在运行时代码里做校验。

## 0.4 环境规则

### 0.4.1 总体做法

1. 并行新建 `environment_v2/`（名称切换时再定），旧 `environment/` 在新环境跑通前不改。
2. 架构参照师兄 MRDTS 来写，但去掉其防御性检查：只在“读取配置”和“接收算法动作”两个入口做校验；实体、通信、runtime 内部不写 shape / 类型 / 有限值检查，不写失败回滚。
3. 验收标准：新环境跑通、debug 无问题即可；本轮不跑基线对比、不跑训练实验。
4. 旧 MARL（MADDPG / MATD3 / MASAC 等）与训练侧（`scripts/train_clean_mainline.py`）本轮不动，新环境不为它们做兼容（§0.5.7 第 4 条）。

### 0.4.2 实体

1. 实体为 UE、UAV、DAG，外加 Region；没有基站（BS）。子任务不是独立实体，而是 DAG 内的节点，用 `(dag_id, node_id)` 标识（同 MRDTS）。
2. 属性以 §0.4.10 字段表为准。
3. Region：借用 MRDTS“给位置返回区域 id”的接口思路（方法名 `get_region_ids`），不搬其 Voronoi 网格实现。区域只有两类：热点区（id=1）与非热点区（id=0）；热点为若干个半径相同的完整圆（整个圆在地图内），个数每个 episode 在 [`min_hotspots`, `max_hotspots`] 中均匀抽取（两者相等即固定个数），中心每个 episode 重新采样、episode 内不动，任意两个中心距离不小于 2 倍半径（互不重叠）；位于任一热点内即为热点区。
4. UAV：只有一种 UAV，写成单个 `UAV` 类：
   - 集合写法：一个对象保存全部 UAV，`positions` 为 (N, 3) 数组，按下标访问；UE 的 z=0，UAV 的 z 为飞行高度；
   - 不保存区域 id：需要时用 `Region.get_region_ids(uav.positions)` 现算；
   - 飞行采用 MRDTS `apply_flight_actions` 的连续动作：每架 UAV 一个 [-1,1]^2 归一化速度 × 最大水平速度 × 时隙长度，越界则保持原位并返回越界标记；位置在时隙末更新（§0.5.1 第 7 条）；
   - 单核：主频为 `UAVConfig.cpu_frequency_hz` 一个标量（UAV 同构）；
   - 计算能耗用 MRDTS 的电容模型 κ·C·f²，计算量字段为 `cpu_cycles`；
   - 飞行能耗：旋翼无人机推进功率公式 P(V) = P0·(1 + 3V²/U_tip²) + Pi·(√(1 + V⁴/(4v0⁴)) − V²/(2v0²))^{1/2} + ½·d0·ρ·s·A·V³（Zeng et al., IEEE TWC 2019），参数按公式中的物理量命名（叶片轮廓功率 P0、诱导功率 Pi、叶尖速度 U_tip、悬停诱导速度 v0、机身阻力比 d0、空气密度 ρ、旋翼实度 s、旋翼盘面积 A）。每时隙飞行能耗 = P(V) × 时隙长度：整个时隙都在飞，没有单独的悬停阶段；越界未动的 UAV 按 V = 0 计，即 P(0)。按现参数 P(0) = 160 W、P(10 m/s) ≈ 123 W、P(15 m/s) ≈ 137 W，中速飞行比原地不动省电（Claude 按 `config/uav.toml` 算）。写成 `UAV` 的方法（师兄建议能耗写进实体）；
   - 不保存剩余能量；没有覆盖半径属性（覆盖由 §0.4.3 第 4 条的接收功率决定）；
   - 初始位置：地图内均匀随机。
5. UE：集合写法。初始位置参照 MRDTS `User` 按区域分配人数：各区域权重 = 面积 × 密度倍数（热点为 `hotspot_density`，非热点为 1）× U(1 − `area_fluctuation`, 1 + `area_fluctuation`)，归一化后按 MRDTS 的取整方法（先向下取整，余数按小数部分从大到小补）分配人数；热点内的 UE 在圆内均匀采样，非热点的 UE 在地图内、所有热点外均匀采样；不向区域中心聚集。高斯-马尔可夫移动，每个时隙的位移 = 速度 × `move_duration_s`（取时隙长度）；不设每时隙最大移动距离，只保留速度不小于 0 和地图边界反弹。热点内减速（开关：`hotspot_speed_scale` 取 1.0 即关闭，暂定 0.5）：移动前位于任一热点内的 UE，本步位移乘 `hotspot_speed_scale`（速度状态本身不变），只看位置、与策略无关；稳定后热点密度约为非热点的 1/`hotspot_speed_scale` 倍，取 1/`hotspot_density` 时开局分布即为稳定分布。热点内 UE 的到达概率为非热点的 `arrival_multiplier` 倍（`region.toml`，暂定 2）。UE 位置每个时隙在时隙末与 UAV 一起更新一次，时隙内的传输都用时隙开始时的位置（§0.5.1）。

### 0.4.3 通信

1. 信道公式沿用 MRDTS `ChannelModel`：3D 距离、仰角 LoS 概率（Al-Hourani 城市参数 a=9.61, b=0.16）、NLoS 衰减系数 η、g0/d²、香农速率。空对空链路按纯 LoS（LoS 概率取 1；师兄确认空中不考虑遮挡），地对空、空对地按仰角公式。
   - 平均信道增益 G = [P_LoS + (1 − P_LoS)·η] × g0 / d²，g0 = −60 dB（1e-6）；平均接收功率 = 发射功率 × G；
   - 噪声 = 噪声功率谱密度 N0 × 这一跳当前分到的带宽 b，N0 = −174 dBm/Hz（b = 1 MHz 时 −114 dBm）。⚠️ MRDTS `serial_scheduling` 用 2.5e-20 W/Hz（约 −166 dBm/Hz）；
   - 速率 = b × log2(1 + p·G / (N0 × b))，b、p 为这一跳当前分到的带宽和功率（第 3 条）；覆盖判断（第 4 条）用设备总功率；
   - 不考虑干扰：各传输之间理想正交，也不考虑不同 UAV 之间的同频干扰（参考 Zhao TMC 2025、VEC TMC 2024；同 MRDTS）。
2. 带宽只由第 3 条的共享带宽决定，DAG 和 UE 上不带带宽字段。
3. 多址与带宽（照师兄 MRDTS `serial_scheduling` 分支 `52d6ba8`）：
   - 全网一个总带宽 `total_bandwidth_hz`，此刻所有正在传的有向传输平分（理想正交，不建子载波；排队的不占）；
   - 每台设备（UAV、UE）有一个总发射功率（UAV 1 W、UE 0.1 W，同 MRDTS 配置），在它正在发的传输之间平分；
   - 每当有传输开始或结束，重新分配带宽和功率、重算所有在传传输的速率，并按剩余数据量更新预计完成时刻（照 MRDTS `communication_resources.py` 与 `event_runtime.py` 的 `_reallocate_transfers`）；
   - 同一发送端到同一接收端的有向链路一次只传一条，先来先到（照 MRDTS `channel_queue.py`）；不同链路同时传，只是各自变慢；多跳逐跳进队（前一跳传完，下一跳才进队）；设备可以同时收和发；
   - 排队等待不耗能；传输能耗 = 发送功率对时间累加（功率随分配变化）；只计 UAV 发送的跳，UE 发送的跳不计（同 MRDTS 只统计 UAV 能耗）；
   - 总带宽数值待模拟定（先试 12 / 20 / 30 MHz）：本项目每时隙一次性分配，入口子任务在时隙开始同时上传（每时隙最多约 17 路：7.3 个 DAG × 平均 2.34 个入口；覆盖外的和放在 UE 上的不上传，实际更少），并发比 MRDTS 的逐个决策多，不能直接用其 12 MHz；
   - 距离平方取 max(d², 1e-12)（照师兄 `4be9834`，保持一致）：只在两架 UAV 重合时起作用。
   例：某时刻 3 条传输在传，各分到 1/3 总带宽；其中一条传完，剩下两条各分到 1/2，速率变快，完成时刻重算。UAV 同时发两条，每条 0.5 W。细则见 §0.5.9 第 5 条。
4. 覆盖：UE 到 UAV 的接收功率不低于 `min_received_power_dbm`（暂定 −86 dBm；UE 0.1 W、UAV 高 100 m 时，水平约 154 m，三维约 183 m；参考 Zhao TMC 2025：覆盖外的用户不能卸载）才能上传。离 UE 最近的 UAV 也覆盖不到时，该 UE 本时隙的新 DAG 只能在 UE 本地执行（§0.5.1 第 3 条）。UAV 之间距离不超过 `air_range_m` 才能直接通信，超出的经管理 UAV 中转（§0.5.2 第 3、6 条）。UAV 发给 UE（回传、前驱结果）时，发送的 UAV 不是管理 UAV 就在中继（经管理 UAV）与直连之间选，直连要求该 UAV 到 UE 的接收功率不低于同一门限（按 UAV 总功率 1 W 算，水平约 366 m；上下行用同一门限，即假设 UE 与 UAV 的接收灵敏度相同）。判断都用时隙开始时的位置。
5. 路由（按 §0.5.2）：
   - 入口子任务上传三种方式：管理 UAV u1 执行（UE → u1）、中继（UE → u1 → u2）、直连（UE → u2，要求覆盖）；动作只选执行节点（UAV 或该 DAG 的 UE，入口在 UE 上时不上传），中继还是直连由环境按独占参考时长选短的（相同时取直连，§0.5.9 第 6 条）；
   - 前驱结果：每个子任务有一个输出量 `output_bits`，它的所有后继共用这一份结果，边上不放数据量。前驱算完，把它的后继按执行节点分组：和前驱同一节点的当场到达；其他每个节点只发一份 `output_bits`，到达后该节点上用到它的后继都记这个前驱到达。不同前驱的结果各自传、不合并。UAV 之间距离超过 `air_range_m` 时经 u1 中转；涉及 UE 的路径见 §0.5.2 第 6 条。各传输同时进行，按第 3 条分享带宽；后继的数据就绪取它各前驱中最晚到达者。⚠️ MRDTS 每条边一个数据量（`edge_features`）、逐边传；前驱结果在前驱算完时进队，MRDTS 逐个决策，在后继被分配时进队；
   - 结果回传（开关 `result_return`，默认打开）：sink 计算完成后立即回传 sink 的 `output_bits`：sink 在 u1 上一跳 u1 → UE；在其他 UAV x 上同入口上传的规则，在中继（x → u1 → UE）与直连（x → UE，要求覆盖）中选（§0.5.2 第 7 条）；sink 在 UE 上不回传；回传在截止前到达才算按时完成；
   - 多跳传输先收完再转发，每一跳各自按第 3 条分享带宽与功率；能耗按各跳发送方功率计算（只计 UAV 发送的跳）；
   - 计算与传输分开：每架 UAV、每个 UE 一条 CPU 时间线（单核，只排计算；输入全部到达后按到达先后计算，照 MRDTS），传输走第 3 条的共享信道，互不占用；子任务计算开始 = max(所有输入数据到达时刻, CPU 空闲时刻)；
   - 位置用时隙开始时的 `positions`（时隙内冻结），DAG 不保存位置。
6. 时间估计：ERS 排序只用单跳参考时长（独占全部带宽和功率，同 MRDTS）和计算时长（候选含 UE，按候选平均）；中继/直连按同样的参考时长比较（§0.5.9 第 6 条），不另写独立的估计器。
7. Gate 3 与 Gate 4 的分工：Gate 3 只提供只读计算——单跳计算（两端位置、链路类型 → 平均信道增益与接收功率；增益、功率、带宽 → 速率，照 MRDTS `serial_scheduling` 的 `rate_bps`）、UE 与 UAV 之间的覆盖判断（第 4 条）、UAV 之间是否在 `air_range_m` 内、路由计算（上传 / 前驱结果 / 回传三类，含 UE 执行时的路径，返回每一跳的发送方、接收方、链路类型；中继还是直连按独占参考时长选）；带宽与功率的动态分配、有向链路与 CPU 的排队叫号、前驱结果按目的节点分发，由 Gate 4 的 runtime 负责（§0.5.9）。

### 0.4.4 时间线

见 §0.5.1。

### 0.4.5 配置

1. 仿照 MRDTS：`config/*.toml` 存数值，`environment_v2/settings/*_config.py` 用 dataclass + `tomllib` 读取（训练环境为 Python 3.12）。
2. 文件划分：`region.toml`（地图边长、热点个数上下限 `min_hotspots`/`max_hotspots`、热点半径、热点到达倍数）、`user.toml`（UE 数量、热点密度倍数 `hotspot_density`、人数浮动 `area_fluctuation`、热点内速度倍数 `hotspot_speed_scale`、移动参数、`move_duration_s`、发射功率、基础到达概率、主频）、`uav.toml`（数量、高度、最大速度、飞行时长、主频、电容系数、发射功率、推进功率公式参数）、`dag.toml`（任务数、层数、父节点数、入口输入数据量、子任务输出数据量、计算量参数）、`channel.toml`（全网总带宽 `total_bandwidth_hz`、噪声功率谱密度 `noise_density_dbm_per_hz`、最小接收功率 `min_received_power_dbm`、g0、NLoS 系数、LoS 参数、UAV 间通信距离 `air_range_m`）、`scheduling.toml`（时隙长度、到达时隙数 `arrival_slots`、截止秒数 `deadline_s`、回传开关 `result_return`，见 §0.5.4）。
3. 旧 `config.py` 保留给旧代码；训练 / 算法参数不进入新配置。
4. 读取方式：每个 dataclass 用 `cls(**data)` 直接构造，缺键、多键由 Python 报错，不另写逐键校验。只检查会让仿真静默出错的几个值：时隙长度、飞行时长大于 0，飞行时长不超过时隙长度，0 < `deadline_s` ≤ 时隙长度。

### 0.4.6 数值

暂定数值见 §0.5.5（版本 D）。

### 0.4.7 智能体

见 §0.5.1 第 4 条与 §0.5.3。环境不依赖 actor 数量。

### 0.4.8 环境接口与随机数

1. 接口见 §0.5.4（`reset` / `step`）。快照不复制数据，直接给出实体数组和 DAG 记录的只读视图，调用方只读不写（同 MRDTS）。
2. `SlotResult` 只放物理结果：本时隙完成的 DAG（带到达、完成时刻）、超时的 DAG（带到达时刻和算完的时刻）、四类能耗（计算、上传与前驱传输、回传、飞行，焦耳；按实际执行全额计入，含超时 DAG 截止后为算完而执行的部分；传输只计 UAV 发送的跳，计算只计 UAV 上的）、不合法分配个数、本时隙的 offer、`done`。reward 和指标由方法层、实验层从中计算。
3. 入口校验只做在动作上：飞行动作的形状与取值范围；placement 按 §0.5.4 第 2 条。
4. 随机数：见 §0.4.14；不依赖导入 `config.py` 时设置的全局 `np.random`。seed 由实验入口提供，不写进 toml。

### 0.4.9 内部检查

1. 内部断言 0 条。非法内部状态通过数据结构和单一持有者消除，而不是事后检查：
   - 每个任务只有一个 `status` 字段（`TaskStatus` 枚举）；
   - 状态只由 runtime 修改，不对外公开 `mark_task_*` 之类的改状态方法；
   - 队列、空闲时刻、任务执行记录只存在于唯一的 runtime 中，其他模块只读快照；
   - 时隙接口只有 `step` 一个方法（§0.5.4），由唯一的调用方（训练 / 评估循环）调用，不做检查，也不为防误用增加阶段对象等额外结构。
2. 校验只在两个入口：读取配置、接收算法动作（见 §0.4.8 第 3 条）。
3. UE、UAV 的编号固定等于数组下标，reset 后不变。

### 0.4.10 字段表

| 对象 | 字段 | 说明 |
|---|---|---|
| 子任务 | `(dag_id, node_id)` | DAG 内的节点（同 MRDTS），节点编号即拓扑序 |
| 子任务 | DAG 的 `input_bits[node]` | 入口子任务从 UE 上传的输入量；非入口节点的值只用来生成计算量，不传输 |
| 子任务 | DAG 的 `output_bits[node]` | 每个子任务一个输出量：非 sink 的输出发给后继，同一前驱发往同一节点只发一份，该节点上的后继共用；sink 的输出就是回传 UE 的结果。⚠️ MRDTS 每条边一个数据量（`edge_features`），无回传 |
| 子任务 | DAG 的 `cpu_cycles[node]` | 计算量 |
| 子任务 | DAG 的 `levels[node]` | 只有 graph 使用，生成后无法从 `edges` 推出 |
| DAG | `edges` | 依赖关系只存一份，边不带数据量；前驱、后继和 sink 由 runtime 从 `edges` 推出 |
| DAG | `id`、`ue_id`、`arrival_time_s`、`num_tasks` | `id` 按 offer 顺序递增；每个 offer 都成为 DAG（§0.4.15）；截止时刻 = `arrival_time_s` + `deadline_s`，不另存 |
| 子任务运行 | `TaskRuntime.status: TaskStatus` | WAITING_DATA / QUEUED_COMPUTE / RUNNING / FINISHED（照 MRDTS，不设 UNSCHEDULED、FAILED）；传输另有 `TransferStatus`：QUEUED / ACTIVE / FINISHED（同 MRDTS）；见 §0.5.9 第 11 条 |
| 子任务运行 | `TaskRuntime` 的执行节点、`ready_time_s`、`start_time_s`、`finish_time_s`、`input_arrival_s`、`predecessor_arrival_s`、`compute_energy_j` | 照 MRDTS `TaskRuntime`，只由 runtime 写；时刻加 `_s` 以区别时隙序号；执行节点为 UAV 或本 DAG 的 UE；传输能耗记在各次传输的 `TransferRecord` |
| DAG 运行 | `DAGRuntime.status: DAGStatus`、`DAGRuntime.finish_time_s` | ACTIVE / COMPLETED / TIMED_OUT；`finish_time_s` 为最后一个 sink 回传完成的时刻（回传关闭或 sink 在 UE 上时为 sink 算完的时刻），COMPLETED 和 TIMED_OUT 都写入 |
| UE | `positions`、`speeds`、`headings` | 集合写法；速度向量现算，不另存；一个 UE 可同时有多个活动 DAG（§0.4.15） |
| UAV | `positions` | (N, 3)；高度、最大速度、飞行时长、主频 `cpu_frequency_hz`、发射功率、电容系数放在 `UAVConfig` |
| Region | `centers`（K×2）、`radius_m`、`get_region_ids(positions)` | 热点个数上下限、到达倍数放在 `region.toml` |

reward、关键路径、超图等只在方法层计算，不存进实体。

### 0.4.11 命名规范

1. 名字里不带来历和阶段：不出现 `clean`、`zrj`、`phase`、`v2`、`legacy`、`new` 等词（目录名 `environment_v2/` 切换时改回正式名）。
2. 一般 1–3 个英文单词，snake_case；类名用名词，函数名以动词开头。
3. 只在容易混淆单位的物理量后加单位后缀：`_s`、`_m`、`_w`、`_j`、`_hz`、`_bits`。
4. toml 键名不重复段名：`[uav]` 下写 `count = 5`，不写 `uav_count`。
5. 注释风格与 MRDTS 保持一致，中文、简短：
   - 每个文件开头一行模块 docstring，说明这个文件负责什么，例：`"""仿真模块共享的实体、信道与任务标识；不依赖队列或方法实现。"""`；
   - 每个类、每个对外方法一行 docstring，说明它是什么、做什么；只有参数含义不直观时，才按 `Args:` / `Returns:` 补充（参考 MRDTS `apply_flight_actions`）；
   - 行内注释只解释不显然的逻辑，以一行为限；较长的文件可用 `# ---- 小节名 ---- #` 分段（参考 MRDTS `dag_generator.py`）；
   - toml 每个键上方一行中文注释，写含义和单位，例：`# 全局正方形仿真区域的边长，单位：m。`；
   - 不写修改历史、标定来源、阶段标签，这些写进 docs。

### 0.4.12 验收

1. 一个一次性检查脚本（约 100 行，放在 `experiments_v2/checks/` 下，不被环境 import，不设统计门槛）：
   - DAG 结构性质逐样本检查：无环、父任务来自更早的层、任务数与父节点数在配置范围内；
   - 固定输入下精确比较：热点判定、理论到达概率；
   - 新旧环境并排打印分布摘要：任务数、数据量、计算量、UE 位移、热点内外到达率；两边使用同一组对照参数，并先统一换算为 bit、CPU cycles、米、秒再打印；
   - 打印两类数据量 `input_bits`（只看入口子任务）、`output_bits` 的分布；
   - 手工小场景只看先后顺序和端点：有向链路先来先到与带宽动态分配、依赖解锁、前驱结果按目的节点只发一份、sink 回传、每次传输的跳路径（上传三种方式、超距中转、回传的中继 / 直连、UE 本地执行时的路径，见 §0.5.2）、截止后照样算完并记下超时 DAG 的完成时刻；
   - 同一 (seed, episode) 下用两个不同的随机策略各跑一次，打印两边前 `arrival_slots` 个时隙的 offer 序列和 UE 轨迹，应完全相同。
2. 新环境用随机连续动作跑若干 episode，打印传输时间、队列长度、按时完成率、超时数，看量级是否合理。同时跑“随机挑一架 UAV”和“贪心挑整个 DAG 最早完成的 UAV”两个简单策略，打印两者的按时完成率，确认两者有明显差距。
3. 不比较通信时间、能耗、轨迹、完整 episode 结果；不跑基线、不跑训练。

### 0.4.13 能耗与奖励的约束（供方法层）

1. 环境只输出物理焦耳值；reward 与指标中的能耗按 κ·C·f² 和 §0.4.2 第 4 条的飞行能耗标定。
2. 能耗量级（版本 D 参数，Claude 估算）：每架 UAV 每时隙飞行能耗约 120–160 J（P(V) × 1 s），5 架约 0.6–0.8 kJ；一个子任务在 UAV 上的计算能耗约 0.4 J（κ·C·f²，C = 1.5e8、f = 5 GHz），一跳传输能耗为 1 W × 传输时长，约 0.1 J 量级；飞行能耗比服务能耗大约两个数量级。reward 不得直接用总能耗：飞行能耗与服务能耗（计算 + 传输）分别归一化（或飞行能耗只作指标不进 reward），分别对应移动与卸载决策；不得为拉近量级而改动 κ 等物理参数。

### 0.4.14 随机流与外生过程

1. 参照 MRDTS `RandomStreams`，由 (seed, episode) 派生四条独立的 `np.random.Generator`：场景（热点个数与中心、UE 人数分配、UAV / UE 初始位置、UE 初始速度与方向）、移动（UE 移动）、到达（到达抽样）、DAG 内容。
2. 到达窗口内每个 UE 每个时隙都抽一次到达，与它是否已有活动 DAG 无关；抽中即从 DAG 流生成一个 DAG，立即生效，不存在接收或拒绝（§0.4.15）。DAG `id` 按生成顺序递增。每条流一个 Generator，整个 episode 只用它，不按时隙或 UE 再拆。
3. UE 不因等待服务而降速；热点内减速（§0.4.2 第 5 条）只取决于 UE 自身位置，与策略无关。
4. 效果：UE 轨迹、到达和 DAG 内容与策略无关，同一 (seed, episode) 下所有策略面对相同的外部环境，不需要 tape。
5. `SlotResult` 记录本时隙的 offer（DAG 编号、UE、到达时刻），供 offer 总数和按时完成率使用。
6. 方法层和随机策略使用自己的随机数生成器，不使用环境的随机流。

### 0.4.15 截止时间与 episode 结束

1. 每个 offer 都立即成为 DAG，不判断 UE 是否空闲；同一 UE 可同时有多个活动 DAG。
2. 截止时间、超时处理与 episode 结束：见 §0.5.1 第 6 条、§0.5.4。
3. 成功：所有 sink 在截止前完成（回传打开时为回传完成），DAG 记为 COMPLETED。
4. 时隙开始时离 UE 最近的 UAV 也覆盖不到该 UE 时（§0.4.3 第 4 条），它本时隙的新 DAG 只能在 UE 本地执行，按同样的截止判定完成或超时（§0.5.1 第 3 条）。
5. 主指标：按时完成率 = COMPLETED 的 DAG 数 ÷ offer 总数（师兄要求加截止时间，实验加完成率指标）。

### 0.4.16 明确不做

1. 不设 `EntityRef` 之类的实体引用类型：UE、UAV 用整数下标，子任务用 `(dag_id, node_id)`。
2. 不写 `validate_dag`：DAG 只由本项目生成器产生，结构性质由 §0.4.12 的一次性脚本检查。
3. 不设冻结 UE 移动的开关：UE 移动与策略无关（§0.4.14）。
4. 快照不复制，`SlotResult` 不放 reward 和指标（§0.4.8）。

## 0.5 时隙流程、通信与 Gate 4 规则

> 时隙流程来自倪师兄提出、韩师兄整理的 procedure：时隙初的状态代表整个时隙，一次输出整个 DAG 的卸载。

### 0.5.1 时隙流程

1. 时隙开始：UAV、UE 位置和信道冻结，整个时隙按这一刻计算（准静态）。
2. 到达：同 §0.4.14（到达窗口内每个 UE 每个时隙抽一次，新 DAG 立即生效）。
3. 关联：每个 UE 关联接收功率最强（即最近）的 UAV，称为该 UE 及其 DAG 的管理 UAV；有 UAV 能覆盖这个 UE 时，最近那架一定能覆盖。UE 测量对各 UAV 的接收功率并上报给管理 UAV（控制信令时延忽略，论文中作为假设写明）。
   用户计算（照 MRDTS）：每个 UE 单核，主频 `cpu_frequency_hz` = 1 GHz（同 JTPDTS 表 II、MRDTS `config/user.toml`），是它自己 DAG 中每个子任务的候选执行节点之一（同 MRDTS `candidate_execution_nodes` 含用户自己的设备）；覆盖外的 UE 候选只有它自己；UE 计算能耗不计（同 MRDTS 只统计 UAV 能耗）；UE 只算自己的 DAG（每时隙最多一个新 DAG）。数据在 UE 和 UAV 之间怎么走见 §0.5.2 第 6 条。依据：子任务平均约 1.5e8 cycles，在 UE 上约 0.15 s、在 UAV 上约 0.03 s，卸载还要加传输时间，两者同一量级，负载高时本地执行是真实的选项（Claude 按版本 D 参数估算）。
4. 决策（方法层，见 §0.5.3）：每架 UAV 一个 actor，一次输出飞行动作，以及它管理的本时隙新 DAG 中每个子任务的执行节点（候选 UAV 或该 DAG 的 UE）。入口输入的进队顺序由 ERS（韩师兄 MRDTS 中的 HEFT 式排序）给出，不由动作决定。⚠️ 偏离 MRDTS：MRDTS 的 `run_online_schedule` 是每个 Member 同时最多一个活动子任务、上一个真实完成后才观测并决定下一个；本项目每时隙一次性给出全部分配。
5. 执行：时隙内事件驱动、排队叫号，参照 MRDTS `env/runtime/event_runtime.py` 与 `methods/scheduling.py` 的 `run_online_schedule`（细则见 §0.5.9）。信道按 §0.4.3 第 3 条：全网共享带宽动态均分、设备功率均分、同一有向链路先来先到。
6. 截止：DAG 截止时刻 = 时隙开始 + `deadline_s`，`deadline_s` 不超过时隙长度（暂定等于时隙长度 1.0 s，即截止就是时隙末，见 §0.5.5）。截止时仍未完成的 DAG 记 TIMED_OUT，但不丢弃：本时隙的事件继续推进，直到本时隙所有 DAG 都做完，记下实际完成时刻；超时在所有任务算完后、算奖励时结算（师兄确认）。⚠️ 偏离 MRDTS `_close_slot()`（截止时停止并清空）。简化假设（论文中写明）：超时 DAG 截止后的剩余部分只用于算出完成时间，不占用下一时隙的资源；下一时隙所有队列从空开始，不跨时隙。真正跨时隙（剩余部分与下一时隙新任务一起排队）留作以后的扩展。
7. 时隙末：UAV 按飞行动作更新位置（位移 = 动作 × 最大速度 × 时隙长度，整个时隙都在飞，位置在时隙末才更新），UE 移动一步（`move_duration_s` 取时隙长度）。飞行能耗 = P(V) × 时隙长度（§0.4.2 第 4 条）。
8. 能耗：计算、传输能耗按实际执行全额计入，含超时 DAG 截止后为算完而执行的部分。

### 0.5.2 通信与路由

1. 信道公式、噪声见 §0.4.3 第 1 条；多址为全网共享带宽动态均分（§0.4.3 第 3 条，照师兄 MRDTS `serial_scheduling`）。
2. 上传覆盖：UE 到 UAV 的接收功率不低于 −86 dBm（§0.4.3 第 4 条）。
3. UAV 之间：两架 UAV 距离不超过 `air_range_m`（300 m，`channel.toml`）才能直接通信；按我们的信道公式，等价于空对空最小接收功率约 −79.5 dBm（用满 1 W 时，Claude 计算）。依据：读过的 4 篇 CCF-A UAV-MEC 论文里，只有 JTPDTS 建模了 UAV 之间的链路（空对空，“主要为视距”），它不限距离，也没有说明原因；Zhan TMC 2026（子任务在关联 UAV 上执行）、Dai TMC 2024（单架 UAV）、Zhou INFOCOM 2022（用户直接卸载给 UAV）没有 UAV 之间的通信。限距离主要影响谁能和谁协作、每架 UAV 能看到谁；明确限距离的 AirFed（arXiv 2510.23053，1000 m 地图取 400 m）与 Li 等（arXiv 2609.14476，1000 m 地图取 500 m）按地图边长换算到 600 m 为 240–300 m（Claude 换算，两篇发表处未核实）。5 架 UAV 均匀分布时，300 m 内平均约 1.9 个邻居，约 10% 的 UAV 没有邻居（自写模拟）。
4. 候选执行节点：该 DAG 的 UE、管理 UAV u1、u1 的 `air_range_m` 内的邻居（含 UE 同 MRDTS）。覆盖外的 UE 只能自己执行；没有邻居的 UAV 管的子任务只能在它自己或 UE 上执行。
5. 入口子任务上传有三种方式：u1 执行（UE → u1）；中继（UE → u1 → u2）；直连（UE → u2，要求 UE 到 u2 的接收功率不低于 −86 dBm）。动作只选执行节点；中继还是直连由环境按规则选：按独占全部带宽和功率时的参考时长比较两条路（同 MRDTS ERS 的参考时延），取短的（相同时取直连）。入口子任务放在 UE 上时不上传。⚠️ 直连偏离 MRDTS（MRDTS 只经接入 UAV）。
6. 前驱结果（前驱算完就发，发往同一节点只发一份 `output_bits`，同 §0.4.3 第 5 条）：UAV → UAV 空对空，距离超过 `air_range_m` 时经 u1 中转（候选都在 u1 范围内，最多两跳）；UAV → UE 同回传的规则（第 7 条）；UE → UAV 按入口上传的规则（UE → u1，或中继 / 直连）；同一节点不传。
7. 回传：开关 `result_return`，默认打开。关闭时 sink 计算完成即算完成；打开时 sink 计算完成后立即回传它的 `output_bits`，回传在截止前到达才算按时完成。路径和入口上传同一规则：sink 在 u1 上一跳 u1 → UE；在其他 UAV x 上，在中继（x → u1 → UE）与直连（x → UE，要求 x 到 UE 的接收功率不低于 −86 dBm，按 x 的 1 W 算）中选，按独占参考时长比较，取短的，相等取直连，直连连不上只能中继；sink 在 UE 上不用回传。⚠️ 偏离 MRDTS（`routing.py` 的 UAV → 用户只经接入 UAV）。
8. CPU：单核；子任务输入全部到达后进入所在节点（UAV 或 UE）的 CPU 队列，按输入到齐的先后计算（照 MRDTS `_dispatch` 按 `data_ready_at` 排序）。

### 0.5.3 方法层（不进环境内核；记录在此，便于环境提供所需信息）

1. 算法：MAPPO（集中训练、分布式执行），所有 UAV 共享团队奖励；critic 输入全局超图（由环境状态按同一规则直接建），池化后得到价值。
2. 超图节点三类：UAV、子任务、用户（UE）。节点类型：每类节点用自己的输入层把特征映射到同一维度，再加类型标记（异构图的常用做法，如 HAN（WWW 2019）、HGT（WWW 2020）按节点类型分别投影，二者均超出近五年）。节点特征：位置只放在 UE 和 UAV 节点上，子任务不带位置（师兄意见：子任务通过归属超边从 UE 得到位置）；子任务的“输出量”特征取 `output_bits`；UE 节点带“是否被覆盖”或“覆盖它的 UAV 数”。超边类型：六类超边拼进同一个关联矩阵（HGNN，AAAI 2019 的做法），每类超边一个可学习权重（沿用用户旧代码 `marl_models/hgnn/clean_incidence.py` 的 typed-gated HGNN，类型数为 6）。
   UE 节点：actor 的局部超图放该 UAV 关联的全部被覆盖 UE，不管本时隙有没有新 DAG（在覆盖内能和 UAV 通信，UAV 知道它在哪；没来任务的 UE 也反映人群分布，供飞行用）；覆盖外的 UE 不放（和任何 UAV 都通不了信，UAV 无从知道；它的 DAG 只能本地算，没有要决策的子任务）。critic 的全局超图放全部 UE，含覆盖外的（集中训练时可用全局信息；覆盖外 DAG 的完成情况也计入奖励）；覆盖外的 UE 只通过归属超边连着自己的子任务，本时隙也没任务时是孤立点，critic 池化时仍能看到它的特征。代价：actor 看不到离所有 UAV 都远的人群，只能靠探索和奖励找到；若成问题，再考虑“热点位置事先已知”或开局把 UAV 均匀铺开。
3. 超边六类：
   - DAG 依赖：每条 DAG 边一条 {父, 子}（HyperJet，IEEE INFOCOM 2025，式(13)）；
   - k 邻域：k = 2 层，节点 + 父 + 祖父（HyperJet 代码 `get_k_neighborhood`）；
   - 归属：每个有新 DAG 的用户一条 = {用户, 它本时隙新 DAG 的全部子任务}（子任务的位置、可在本地执行，都通过它连到用户；没有新任务的用户不建）；
   - 覆盖：每个被覆盖的用户一条 = {用户, 能和它通信（不低于 −86 dBm）的所有 UAV}；覆盖外的用户不建（只剩它自己的单点超边在卷积里只相当于自环，传不出信息），“是否被覆盖”改作 UE 节点特征（师兄同意）；
   - 服务：每架 UAV 一条 = {这架 UAV, 它关联的被覆盖用户, 这些用户本时隙新 DAG 的全部子任务, 它 `air_range_m` 内的邻居 UAV}；
   - 划分：KaHyPar k 路划分，每块一条超边。以其余五类超边为输入，三类节点都参与（局部也要切：actor 只看自己的局部超图，划分超边要进 actor 的输入，又不能用别的 UAV 的信息）；每架 UAV 的局部超图各切一次（k = 候选 UAV 数，只有 1 架时不切），全局超图切一次（k = UAV 数）；ε = 0.03、km1 目标（同 HyperJet）、不加权、固定种子；每个时隙一次；进程内直接调用，不加子进程、超时、熔断。kahypar 只有 Linux/macOS 安装包，在 Linux 服务器上跑。
   局部超图只用该 UAV 能得到的信息（自己、关联的被覆盖用户及其子任务、邻居 UAV）。
4. 动作：节点打分器（所有子任务共用一个小 MLP）输出每个子任务在候选执行节点（候选 UAV 和该 DAG 的用户）上的分布，不可选的掩码；飞行用高斯头；两部分对数概率相加。
5. 邻居负载摘要：开关，默认开。时隙开始时邻居告知本时隙要管的子任务数和总计算量，作为局部超图中邻居 UAV 节点的特征（信令假设同 §0.5.1 第 3 条）。
6. 奖励：MLP 版先只做时延，公式见 §0.5.11 第 5 条；能耗和子任务级奖励以后再加。
7. 环境快照：时隙开始、到达与关联做完后给方法层的只读视图（§0.4.8），不复制数组；每个时隙从空队列开始，快照里没有队列状态。字段：
   - `slot`：时隙序号（截止 = 时隙开始 + `deadline_s`，现算）；
   - `uav_positions` (N, 3)、`ue_positions` (M, 3)；
   - `new_dags`：本时隙全部新 DAG 的静态数据（`id`、`ue_id`、`edges`、`input_bits`、`output_bits`、`cpu_cycles`、`levels`），含覆盖外 UE 的（critic 全局超图要用）；
   - `manager_uav` (M,)：每个 UE 的管理 UAV；
   - `coverage` (M, N) 布尔：UE 按 0.1 W 发到各 UAV 是否不低于 −86 dBm（同 `Router.can_upload`）；
   - `uav_adjacency` (N, N) 布尔：两架 UAV 是否在 `air_range_m` 内；
   - `candidates`：每个需要决策的 DAG 一个候选执行节点元组 `(None, u1, u1 的邻居按编号升序)`；覆盖外的 DAG 不出现（环境直接放 UE）。
   不放：接收功率矩阵（有 `coverage`；ERS 的参考时长由方法层调 `Router.transfer_duration_s`，信道公式只有一处实现）、热点位置（不假设事先已知）、队列与 runtime 状态、邻居负载摘要（方法层由 `new_dags` 与 `manager_uav` 算）、UE 速度与方向、奖励和上一时隙结果（在 `SlotResult`）。位置数组在 `step` 末原地更新，快照只在本时隙内有效，方法层要留存需自己拷贝。
8. 实验时照 HyperJet 做超边类型消融（HyperJet 实验里三类与两类超边效果几乎一样，多加不一定更好）。

### 0.5.4 接口、截止与 episode

1. 接口：`reset(seed, episode)` 返回第一个时隙的快照；`step(flight_actions, placements)` 执行本时隙，返回 `SlotResult` 和下一时隙快照（`done=True` 时为 `None`）。⚠️ 偏离 MRDTS 的 `begin_slot`/`end_slot` 两个方法：飞行和卸载同时决策、飞行在时隙末才执行，分成两个方法没有意义。
2. `placements`：((dag_id, node_id), 执行节点) 列表（执行节点为某架 UAV 的编号，`None` 表示该 DAG 的 UE；UAV 0 和 `None` 在 Python 里都为假，判断一律写 `is None`，不写 `if node:`），覆盖本时隙全部需要决策的子任务（覆盖外 UE 的 DAG 由环境直接放在 UE 上，不在其中）；列表顺序即入口输入的进队顺序，方法层把各 UAV 的分配按 ERS 值统一排序后再交（§0.5.9 第 6 条）。入口校验：子任务属于本时隙新 DAG、每个恰好出现一次、执行节点在该子任务的候选集合内。不合法的分配计数，所在 DAG 直接记 TIMED_OUT。
3. 不设 UAV 队列上限：每个时隙从空队列开始（§0.5.1 第 6 条），积压已被限住；设上限只会在任务集中的热点 UAV 上额外拒掉分配。
4. 截止：配置为 `deadline_s`（`scheduling.toml`）；读取配置时检查 0 < `deadline_s` ≤ 时隙长度。成功 = 所有 sink 在截止前完成（回传打开时为回传完成）；超时的 DAG 也记下算完的时刻（§0.5.1 第 6 条）。
5. episode 恰好 `arrival_slots` 个时隙，最后一个时隙结束时 `done=True`，没有收尾段。
6. 主指标：按时完成率 = COMPLETED 的 DAG 数 ÷ offer 总数。

### 0.5.5 数值（版本 D，暂定）

| 文件 · 字段 | 取值 |
|---|---|
| `config/region.toml` `side_length_m` | 600（按师兄的 UAV 覆盖比例） |
| `config/region.toml` 热点个数 / `hotspot_radius_m` | 3 / 90 |
| `config/uav.toml` `count` | 5 |
| `config/uav.toml` `cpu_frequency_hz` | 5e9（参考 Dai TMC 2024 的 5 GHz、Zhan TMC 2026 的 4.5 GHz） |
| `config/uav.toml` `flight_duration_s` | 1.0（等于时隙长度） |
| `config/uav.toml` `max_speed` | 15（每时隙每个方向 ≤ 15 m） |
| `config/user.toml` `count` | 60 |
| `config/user.toml` `cpu_frequency_hz` | 1e9（JTPDTS、MRDTS） |
| `config/user.toml` `move_duration_s` | 1.0 |
| `config/user.toml` `arrival_probability` | 0.24（热点内 ×2；每时隙约 19.5 个 DAG；每个 UE 每时隙一次伯努利，即泊松到达的离散时隙形式，同 Tang & Wong TMC 2022；选点见下方“扫参依据”） |
| `config/channel.toml` `air_range_m` | 300 |
| `config/channel.toml` `total_bandwidth_hz` | 20e6（师兄 MRDTS 为 12 MHz；师兄同意适当提高） |
| `config/scheduling.toml` `slot_duration_s` | 1.0 |
| `config/scheduling.toml` `deadline_s` | 1.0（等于时隙长度） |
| `config/scheduling.toml` `result_return` | true |
| `config/scheduling.toml` `arrival_slots` | 200 |
| `config/dag.toml` `min_input_bits` / `max_input_bits` | 279,620.3 / 2,796,202.7 |
| `config/dag.toml` `min_output_bits` / `max_output_bits` | 419,430.4 / 2,097,152 |
| `config/dag.toml` `base_unit_bits` | 27,306.67 |
| `config/dag.toml` `min_task_constant` / `max_task_constant` | 166,666.7 / 500,000 |

负载依据：负载 = 每时隙到达计算量 / UAV 总算力。论文一般不直接给负载，Claude 按其参数推算（只算 UAV、不含基站）：JTPDTS 每个设备每时隙一个 DAG，4 UAV/10 设备约 7%、8/40 约 14%、12/80 约 18%，完成率各方法 75%–98%（表 IV）；MRDTS 代码（100 用户每时隙各一个 DAG）约 19%–23%；Zhou INFOCOM 2022 约 1%。本项目 p = 0.24 时约 76%（按全部 DAG 计；随机飞行下约 41% 的 DAG 覆盖不到、只在 UE 上算，UAV 实际负载约其六成），比师兄偏重；实验扫规模（如负载 20%–75%）。子任务计算量在文献范围内：每子任务约 1.5e8 cycles（师兄 1e7–1e8，Zhan TMC 2026 1e8–5e8）；DAG 总计算量均值 9.8e8 cycles，UE 1 GHz 单独算一个 DAG 约 65% 能在 1 s 内做完。数据量比师兄大约 3–7 倍：入口输入 35–350 KB、子任务输出 52–262 KB，师兄是 100–500 Kbit（12.5–62.5 KB）、100–300 Kbit（12.5–37.5 KB）（MRDTS `event_runtime.py` 把配置值 ×1000 换成 bit）。KaHyPar 实测每次 15 ms（30 节点）到 99 ms（115 节点）。完成率在新模型下还没有模拟，Gate 4 实现后按 §0.5.7 第 9 条扫参数再定。

扫参依据（四个基线、随机飞行，12 个到达概率 0.06–0.48 × 12 / 20 / 30 MHz，每组 10 个 seed，`experiments_v2/runs/sweep.py`，真实代码）：随机飞行下约 59% 的 DAG 被覆盖，覆盖外的只能本地算、约 65% 按时完成；贪心在覆盖内的 DAG 上几乎做满（20 MHz 下 p ≤ 0.21 时 ≥ 99%），卸载难度主要随带宽变窄、负载升高出现。选点规则只看基线、事先定好：贪心与 HEFT 式未做满、四个基线拉得开。20 MHz、p = 0.24 时：全本地 65.6%、随机 70.8%、贪心 85.2%、HEFT 式 85.7%（覆盖内：随机 75.7%、贪心 98.7%、HEFT 式 99.5%）。学会的飞行提高覆盖后 UAV 负载会上升，卸载会更难。

### 0.5.6 各 Gate 内容

1. Gate 0–2：系统事实、公共类型与配置、实体，已完成。
2. Gate 3：信道改为照师兄 `serial_scheduling` 的“平均信道增益 + 速率函数”；新增 `air_range_m` 与 UAV 邻接判断、UE 与 UAV 双向覆盖判断（只读）；细则见 §0.5.9 第 12 条。
3. Gate 4：整 DAG 提交 + 时隙内事件驱动 + 截止后照样算完，参照 MRDTS `env/runtime/`（文件名也照 MRDTS），信道为共享带宽，加入中继 / 直连规则和超距中转。详细规则见 §0.5.9。
4. Gate 5：Simulator，流程按 §0.5.1，对外接口按 §0.5.4；只协调流程，不算 reward、不建超图、不构造 observation、不加载 checkpoint、不记录训练 loss。
5. Gate 6：验收按 §0.4.12。
6. Gate 7：方法层按 §0.5.3 新写（§0.5.7 第 4 条）；先做不接超图的 MLP 版 MAPPO 跑通（§0.5.11，师兄 11-06 意见），再接超图。
7. Gate 8：正式切换并通过审核后删除旧代码：不建永久 `legacy/`，由 Git 历史负责恢复；删除前只检查实际引用和研究证据用途；不删用户未提交文件；删除单独形成可审核 diff；用户明确批准后执行。

### 0.5.7 已记录、先不处理

1. 划分超边 ε = 0.03 要求各块几乎一样大，若各 UAV 手上任务数相差悬殊，可能拆开本该在一起的一群任务；先照 HyperJet，看实验再定（可放大 ε 或用计算量做节点权重）。
2. 孤立 UAV（300 m 内没有邻居，约 10%）：训练时统计；偏高再考虑加 UAV 或给 actor 全体 UAV 的位置。
3. 奖励：MLP 版只做时延（§0.5.11 第 5 条）；能耗、子任务级奖励（第 6 条）以后再加。
4. Gate 7 不给旧训练代码做适配：方法层按 §0.5.3 新写，旧训练代码不动；以后若要旧方法当基线，在新框架里按其思路重写。
5. 对比实验（JTPDTS、HYGMA、TGCNet、HyperJet 等）以后再做。
6. 用户考虑给每个子任务单独的奖励（奖励设计时再定；可参考 One-shot DAG（NeurIPS 2025）的节点级密集奖励）。
7. 覆盖外的 UE：600 m 地图下 UAV 随机摆放时约 41% 的 UE 不在覆盖内（自写模拟），它们的 DAG 只能本地执行（§0.5.1 第 3 条）。文献依据：JTPDTS 覆盖外的设备本地计算（1 GHz）；Zhou INFOCOM 2022 覆盖外卸载到远端云；MRDTS 候选执行节点含用户自己的设备和基站；Zhao TMC 2025 只说覆盖外不能卸载（是否本地计算未核对）。
8. 中继 / 直连：按独占参考时长比，只要直连连得上，直连总比中继快（自写模拟，600 m、5 架 UAV 随机摆放：上传中继最少慢 19%，约 22% 能直连、78% 只能中继；回传 UAV 用 1 W，水平约 366 m 内能直连，约 96% 能直连，中继最少慢 16%；12 / 20 / 30 MHz 都是如此），这一步比较实际上不改变结果；先保留，跑实验后再看是否简化成“能直连就直连，否则中继”。
9. 到达概率：0.09 时每时隙约 7.3 个 DAG、约 47 个子任务（在 HyperJet 实验的 20–60 个任务之间），但每架 UAV 只分到 1–2 个 DAG，抢资源的情况少，覆盖、服务这几类超边可能显不出作用。Gate 4 跑通后，到达概率和总带宽一起扫（如 0.06–0.24 × 12 / 20 / 30 MHz），每组跑随机、贪心、全本地、ERS，选贪心明显做不满、几个基线拉得开的点做默认值。不走“DAG 改小、个数变多”：DAG 小到 UE 自己能算完时，完成率分不出好坏（师兄的设置下一个 DAG 全本地最多 1.0 s）。
10. 飞行基线：热点不动、UE 每时隙约走 1.2 m，最好的 UAV 位置几乎不变。按到达 DAG 加权的覆盖率（自写模拟：600 m、热点 90 m、60 个 UE、1500 次）：UAV 随机摆放 61%，3 架放热点中心 + 2 架随机 76%，知道全部 UE 位置的静态贪心部署 97%（上界参考）。飞行要和随机、网格、往覆盖内用户重心飞等启发式比；动态场景先记录、不实现：episode 内热点变化（例如第 100 个时隙重抽热点中心）、到达率随时间变化（例如某个热点进入高峰），考的是“重新发现热点”；现在只做“热点每个 episode 重抽、episode 内不动”的场景，动态场景放在 §0.5.11 第 11 条 S4 之后作鲁棒性实验。
11. 非入口子任务的计算量：现在由它自己抽的 `input_bits` 推出（规模 n = `input_bits` ÷ `base_unit_bits`），这个 `input_bits` 只决定计算量、不传输，与前驱的 `output_bits` 无关（相关系数 0.00，Claude 自写模拟）。另一种做法是按各前驱 `output_bits` 之和算规模（更符合“收到的数据越多、要算的越多”），DAG 总计算量均值会从 9.83e8 升到 1.36e9 cycles（+38%，自写模拟），需重新校准常数。先不改；方法层能训练后，两种生成方式各训练一次，看完成率和超边消融结论是否变化再定。方法层不把非入口子任务的 `input_bits` 作为网络特征（与 `cpu_cycles` 重复）。

### 0.5.8 代码目录（照 MRDTS）

新代码的目录照 MRDTS（`env/`、`methods/`、`scripts/`、`config/`），旧代码不动、不混放：

```text
config/                      # 环境数值（已有）；算法超参放 config/methods/（同 MRDTS）
environment_v2/              # 环境，切换时改为正式名（对应 MRDTS env/）
  types.py  contracts.py  random_streams.py        # 已有（Gate 1）
  settings/  entities/  workload/dag_generator.py  # 已有（Gate 1–2）
  communication/channel_model.py  routing.py       # 已有（Gate 3）
  workload/arrival.py        # 到达（Gate 4）
  runtime/                   # Gate 4，照 MRDTS env/runtime/
    task_runtime.py          #   子任务运行状态
    dag_runtime.py           #   DAG 状态、截止判定
    server_queue.py          #   单核 CPU 队列（照 MRDTS）
    channel_queue.py         #   有向链路先来先到（照 MRDTS）
    communication_resources.py  # 全网带宽与设备功率均分（照 MRDTS serial_scheduling）
    event_runtime.py         #   时隙内事件推进、截止后照样算完
    slot_result.py           #   SlotResult
  simulator.py               # Gate 5：reset / step（对应 MRDTS simulator.py）
methods_v2/                  # 方法层（对应 MRDTS methods/），环境不 import 它；11-06 由 methods/ 改名，与 environment_v2/ 成对
  ers.py                     #   ERS 排序、参考时长、按 ERS 排序提交（⚠️ MRDTS 为 Member 内串行逐个决策）
  baselines/policies.py      #   随机、全本地、贪心、HEFT 式四个基线和随机飞行（§0.5.10）
  learning/                  #   通用学习代码，不 import 环境（照 MRDTS codex/development 的 methods/learning/；§0.5.11）
  mappo/                     #   本项目的观测、actor、critic、奖励（§0.5.11）
  hypergraph/                #   以后：六类超边、KaHyPar 划分、HGNN 编码器（移植旧 clean_incidence.py）
experiments_v2/              # 实验入口（对应 MRDTS scripts/）
  checks/                    #   小实验、小测试：几分钟跑完，只回答“对不对”（§0.4.12）
    env.py  simulator.py  slot_runtime.py  replay.py
  runs/                      #   正式实验：多种子、多进程，产出进论文的数据
    baselines.py  sweep.py  train.py  plot.py  metrics.py   # metrics.py：训练、评估、基线共用的指标统计
```

1. 与 MRDTS 的差别：reward、observation 放在 `methods_v2/`（MRDTS 放在 `env/step/`），保持“环境只出物理结果”（§0.4.8）；不建 MRDTS 那样的大量 `tests/`，检查用 §0.4.12 的一次性脚本。
2. 仓库根目录的旧代码（`environment/`、`marl_models/`、`train.py` 等）本轮不动，Gate 8 再删。
3. 新代码不 import 旧代码；要用的旧实现（如 typed-gated HGNN）复制到 `methods_v2/` 改写。
4. 命名：文件名一个词，不加目录已表达的前缀（写 `checks/env.py`，不写 `check_env.py`）。正式实验名为“内容_月日”，内容一个词（如 `joint_1106`、`flight_1110`、`sweep_1105`），同一天再跑加 `b`、`c`；同一组实验的不同方案在内容后加“-方案名”（如 `s1-shared_1109`、`s1-id_1109`）。检查类子命令（如 `runs/train.py check-reward`）算小实验，按子命令名写到 `output/checks/<子命令名>/`。
5. 产物：`output/` 不进 git（写入 `.gitignore`）。小实验写到 `output/checks/<脚本名>/`，每次覆盖、不留历史；正式实验写到 `output/runs/<实验名>/`，内含 `config/`（运行时环境与方法配置的副本）、`seed<k>/`（`metrics.jsonl`、`checkpoints/`）、`figures/`、`summary.json`。Python 虚拟环境放在仓库外。所有实验（含小实验）都在服务器上跑，产物只留在服务器的 `output/`，不拷回本地；本地与服务器只保持代码同步。仓库目录：本地 `D:\CodeFile\HyperUAV-zdq`，服务器 `/data2/zrj2025/HyperUAV-zdq`。
6. 旧路线的脚本留在 `scripts/`，不动，Gate 8 随旧代码一起删。

### 0.5.9 Gate 4 运行规则

> 对照 MRDTS `e3fedcf` 的 `env/runtime/`、`methods/random/ers.py`；信道部分对照师兄 `serial_scheduling` 分支（`7180b9b`）的 `env/runtime/{communication_resources, channel_queue, event_runtime}.py`。⚠️ 标出偏离 MRDTS 的地方。

1. 范围：Gate 4 写 `workload/arrival.py`，改 `workload/dag_generator.py`（`edge_bits`、`return_bits` 合成每个节点一个 `output_bits`），写 `runtime/{task_runtime, dag_runtime, server_queue, channel_queue, communication_resources, event_runtime, slot_result}.py`，并把 `config/` 改成版本 D（第 13 条）；`simulator.py`（reset/step、快照、动作校验、飞行、UE 移动）属于 Gate 5。Gate 3 先按第 12 条改信道、补几处只读计算。
2. 排队叫号（照 MRDTS `event_runtime.py`）：
   - 队：每条有向链路（发送方, 接收方）一条先来先到的队，队头在传、其余等（同 `serial_scheduling` 的 `DirectedChannelState`）；每个执行节点（UAV、UE）一条 CPU 队；
   - 一跳在它的数据就绪时进队；子任务在输入全部到齐时进 CPU 队；
   - 事件只有“某一跳传完”“某子任务算完”两种，按时间放进堆；同一时刻的事件按压入先后全部处理完，再叫号一次（同 `_process_events_until`）；
   - 叫号：t0 一次，之后每处理完一个时刻的事件一次。顺序同 MRDTS `_dispatch`：先把这一刻新到齐的子任务按 (到齐时刻, dag_id, node_id) 放进 CPU 队；再让每条链路队头能开始的开始；正在传的跳有增减时，重新分配带宽和功率（第 5 条），重算所有正在传的跳的预计传完时刻，整批替换堆里的“传完”事件（同 `_reallocate_transfers`）；最后叫 CPU；
   - 截止后照常叫号，直到本时隙所有 DAG 做完（第 8 条；⚠️ MRDTS `_dispatch` 到截止直接返回）。
3. 到达（`workload/arrival.py`）：t0 = 时隙序号 × 时隙长度，截止 = t0 + `deadline_s`；每个 UE（按下标）用到达流抽一次伯努利，概率 = `arrival_probability`，时隙开始时在热点内的 × `arrival_multiplier`，不截到 1；抽中的 UE 按下标顺序用 DAG 流生成 DAG，`id` 全局递增，`arrival_time_s` = t0。
4. 新 DAG 分流（按时隙开始的位置）：
   - 管理 UAV u1 = 离 UE 最近的 UAV；
   - u1 对 UE 的接收功率低于 −86 dBm（覆盖外）：整个 DAG 放在 UE 上执行，由环境直接填，不交给智能体、不算进 placements，照常进 runtime（第 9 条）；
   - 否则每个子任务的候选执行节点 = {该 DAG 的 UE} ∪ {u1} ∪ {与 u1 距离 ≤ `air_range_m` 的 UAV}，DAG 内所有子任务共用这一个集合（含 UE 同 MRDTS `candidate_execution_nodes`；MRDTS 还有 BS，本项目没有）；
   - 动作校验由 Gate 5 做（§0.5.4 第 2 条）；不合法的 DAG 整个 TIMED_OUT，不进 runtime、能耗 0、没有完成时刻；`invalid_placements` = 错误条目数 + 漏掉的子任务数。
5. 信道（`runtime/communication_resources.py`、`runtime/channel_queue.py`，照 `serial_scheduling`）：
   - 全网所有正在传的跳均分总带宽 B：每跳 b = B / 正在传的跳数；
   - 每个发送方（UAV 1 W、UE 0.1 W）的功率在它正在发的跳之间均分；
   - 速率 = b·log2(1 + p·G / (N0·b))，G 为这一跳的平均信道增益（§0.4.3 第 1 条）；时隙内位置冻结，每跳的 G 算一次后缓存（同 `_channel_gain`）；
   - 只在正在传的跳有增减时重新分配；两次之间速率不变，按速率扣剩余比特、按功率 × 时长累加能耗（同 `_advance_clock`）；
   - 同一有向链路先来先到，队头传完下一个才开始；不同链路互不排队，只通过分带宽、分功率互相影响；
   - 例：B = 12 MHz，t0 有 UE_a → u1、UE_b → u1、u1 → u2 三跳在传 → 每跳 4 MHz；u1 只发一跳，1 W 全给 u1 → u2；UE_a → u1 在 0.1 s 传完 → 剩下两跳各 6 MHz，重算速率和预计传完时刻；
   - 传完即退出；排队不耗能；
   - ⚠️ MRDTS 原版每条链路各用满带宽；与 `serial_scheduling` 的差别只有 N0（§0.4.3 第 1 条）。总带宽待模拟定（§0.5.5）。
6. 数据怎么走（逐跳排队，前一跳传完下一跳才进队，同 MRDTS；中转节点收一跳、发一跳）：
   - 入口子任务的输入（t0 进队）：进队顺序 = placements 列表顺序；方法层把所有 UAV 的分配按 ERS 值统一排序后再交（大的先，相等按 (dag_id, node_id)，同 MRDTS `ERS.plan` 的全局 `order`），不按 UAV 编号拼接；只有入口输入受列表顺序影响，其他传输和计算按事件先后进队。每个入口子任务各传自己的 `input_bits`，不合并。执行节点 = UE：不传，t0 到达；= u1：UE → u1；= u2 ≠ u1：在中继（UE → u1 → u2）与直连（UE → u2，要求 UE 到 u2 不低于 −86 dBm）中选，按独占全部带宽、发送方全部功率时的参考时长比较（同 MRDTS `transfer_duration_s`，ERS 也用它），取短的，相等取直连，直连不覆盖只能中继；参考时长不看排队和分带宽，实际可能更长。⚠️ MRDTS 每个子任务都从用户设备上传自己的输入、只经接入 UAV；本项目只有入口子任务上传（§0.4.10），且有直连；
   - 前驱结果（前驱算完时进队；数据量 = 前驱的 `output_bits`；⚠️ MRDTS 每条边一个数据量、逐边传）：把前驱 p 的后继按执行节点分组；和 p 同一节点的当场到达；其他每个节点只发一份（一个 `TransferRecord`），到达后该节点上用到 p 的后继都记 p 到达；不同前驱各自传、不合并。路径：UAV x → UAV y：距离 ≤ `air_range_m` 一跳，超过经 u1 两跳（候选都在 u1 范围内）；UAV x → UE：同下面的结果回传；UE → UAV y：同入口输入的规则。后继的所有前驱都到达 → 到齐（到齐时刻取最晚那个），进 CPU 队。⚠️ MRDTS 逐个决策，后继被分配时（那时前驱已算完）才把前驱结果送进队；本项目每时隙一次性分配，后继在哪 t0 就知道，所以前驱一算完就进队；
   - 结果回传（`result_return` 开；sink 算完时进队；数据量 = sink 的 `output_bits`）：sink 在 UAV x：x = u1 时一跳 x → UE；否则同入口输入，在中继（x → u1 → UE）与直连（x → UE，要求 x 到 UE 不低于 −86 dBm，按 x 的 1 W 算）中按参考时长选，取短的，相等取直连，直连连不上只能中继（⚠️ MRDTS `routing.py` 的 UAV → 用户只经接入 UAV）；sink 在 UE：不传；每个 sink 各自回传。DAG 完成 = 所有 sink 的回传都到 UE（回传关时 = 所有 sink 都算完），完成时刻取最晚者。
7. CPU（`runtime/server_queue.py`）：照搬 MRDTS `ServerState`，改成单核——一条 FIFO 队、一个“正在算”的记录；`enqueue`、`dispatch`（CPU 空着且队里有 → 队头开始，时长 = `cpu_cycles` / 该节点主频）、`finish`。UAV 和 UE 用同一个类：UAV 5 GHz，能耗 κ·C·f² 开始时算好；UE 1 GHz，能耗不计（同 MRDTS `_settle_compute_energy` 只记 UAV）。删 MRDTS 的多核编号和 `compute_wait`（ERS 在 t0 排序时队都是空的，用不到）。
8. 事件处理与结束（`runtime/event_runtime.py`）：
   - t0：为进 runtime 的 DAG 建好子任务（WAITING_DATA，执行节点已知），入口输入的第一跳进队（入口在 UE 上的直接到齐），叫号；
   - 某一跳传完：退出链路（能耗已随时钟累加）；还有下一跳 → 下一跳进队；是最后一跳 → 按用途记输入到达、某前驱结果到达（该节点上用到它的后继都记，可能使子任务到齐）或某 sink 回传到 UE（可能使 DAG 完成）；
   - 某子任务算完：释放 CPU，记计算能耗，子任务 FINISHED；按第 6 条发给每个后继；是 sink 时，回传开且 sink 在 UAV 上 → 回传第一跳进队，否则这个 sink 完成；
   - DAG 完成 → 记 `finish_time_s`；不晚于截止为 COMPLETED，否则 TIMED_OUT；
   - 一直处理到事件堆空，即本时隙所有 DAG 都做完，截止时不停（⚠️ 偏离 MRDTS `_close_slot()` 截止时停止、清空、记 FAILED）；所有子任务最后都会 FINISHED；
   - 每个时隙新建一个 runtime（同 MRDTS 每时隙一个 `SchedulingRuntime`），不跨时隙：超时 DAG 截止后的部分只为算出完成时刻，不占下一时隙的资源（简化假设，论文写明）；不提前判断“来不及就不发”（同 MRDTS）。
9. UE 执行：UE 的 CPU 队只有它自己 DAG 的子任务（每时隙最多一个新 DAG）。覆盖外的 UE 整个 DAG 在 UE 上，没有传输，单核依次算，完成时刻 = t0 + DAG 全部 `cpu_cycles` 之和 / 1 GHz；和其他 DAG 一样进 runtime，不另写函数。参考（Claude 自写模拟，版本 D 的 DAG 参数、2 万个样本）：DAG 总计算量均值 9.8e8 cycles，1 GHz 下约 65% 能在 1 s 内算完。
10. 能耗：计算 κ·C·f²（只 UAV）；传输 = 分到的功率 × 时长，分配变了就分段累加（同 `_settle_transfer_energy`）；只计 UAV 发出的跳，UE 发出的不计（同 MRDTS）；全额计入，含超时 DAG 截止后的部分；排队等待不耗能。归类：`tx_energy_j` = 输入中继的第二跳 + 前驱结果中 UAV 发出的各跳；`return_energy_j` = 回传中 UAV 发出的各跳；`compute_energy_j`；`flight_energy_j` 由 Gate 5 加。
11. 状态与记录（照 MRDTS，删用不到的；枚举放在已有的 `types.py`）：
   - `TaskStatus`：WAITING_DATA（已分配，等输入）→ QUEUED_COMPUTE（到齐，在 CPU 队）→ RUNNING → FINISHED；不设 MRDTS 的 UNSCHEDULED（t0 建立时已有执行节点）和 FAILED（截止不停，都会算完）；替换 `types.py` 现有的 WAITING / READY / SCHEDULED / RETURNING / DONE / CANCELLED（回传进度看回传那次传输的状态）；
   - `TransferStatus`：QUEUED → ACTIVE → FINISHED（同 MRDTS，它放在 `channel_queue.py`）；
   - `DAGStatus`：ACTIVE → COMPLETED / TIMED_OUT；
   - `TaskRuntime`（照 MRDTS，时刻字段加 `_s`）：`key`、`execution_node`（UAV 编号，`None` 表示该 DAG 的 UE）、`cpu_cycles`、`input_bits`、`predecessors`、`status`、`input_record`、`predecessor_records`、`input_arrival_s`、`predecessor_arrival_s`、`ready_time_s`、`start_time_s`、`finish_time_s`、`compute_energy_j`，加 `return_record`（sink 在 UAV 上且回传开时）；删 MRDTS 的 `decision_at`（都是 t0）、`compute_queue_enter_at`（= `ready_time_s`）、`execution_core_id`（单核）、`failed_at`（没有 FAILED）；状态只由 `event_runtime` 修改，不设 MRDTS 的 `mark_*` 方法（§0.4.9 第 1 条）；到齐判定：入口子任务 = 输入到达，其他 = 所有前驱的结果到达（MRDTS 两者都要，因为它每个子任务都有输入）；同一节点上的几个后继可共用同一个前驱的 `TransferRecord`；
   - 传输记录（`channel_queue.py`，照 `serial_scheduling`）：每次数据搬运（一个输入 / 一个前驱发往一个节点 / 一次回传）一个 `TransferRecord`（各跳作业、开始、结束、发送能耗）；每一跳一个 `TransferJob`：`transfer_id`、`source_ready_s`、`payload_bits`、`channel_gain`、`remaining_bits`、`bandwidth_hz`、`power_w`、`rate_bps`、`predicted_finish_s`、`tx_energy_j`、`status`、`start_time_s`、`finish_time_s`；下一跳的作业在前一跳传完时进队；剩余比特扣减时取 max(0, ·)（同 MRDTS，恰好传完时浮点舍入会出现略小于 0）；
   - `DAGRuntime`：DAG 静态数据、u1、`status`、`finish_time_s`（COMPLETED、TIMED_OUT 都写）、由 `edges` 推出的后继和 sink、还没完成的 sink 数；删 MRDTS 的 `remaining_predecessors` / `ready_node_ids`（逐个决策时用来判断哪些子任务能决策）和 `topological_order`（节点编号就是拓扑序）；
   - `SlotResult`（Gate 5 组装）：`offers`、`completed_dags`、`timed_out_dags`（都含覆盖外 UE 的 DAG）、各 DAG 的完成时刻（奖励按超时长短惩罚用）、`compute_energy_j`、`tx_energy_j`、`return_energy_j`、`flight_energy_j`、`invalid_placements`、`done`；同时改掉 `contracts.py` 中已过时的注释（“冻结 ready 列表”“队列未满”“承诺时计入”“统一相对截止长度”）；
   - `event_runtime` 对外：每个时隙新建，一个方法跑完整个时隙；输入 t0、截止、冻结的位置、进 runtime 的 DAG（含 u1，覆盖外的已全部放在 UE 上）、已校验并按 ERS 排好的 placements、配置；返回各 DAG 的 `DAGRuntime`（含子任务和传输记录，§0.4.12 的检查脚本用它看先后顺序）。
12. Gate 3 先改：
   - `ChannelModel`：照 `serial_scheduling` 改成“平均信道增益 + 速率函数”：增益 G = [P_LoS + (1 − P_LoS)·η]·g0 / max(d², 1e-12)（同 `4be9834`，§0.4.3 第 3 条），速率 `rate_bps(G, p, b)` = b·log2(1 + p·G / (N0·b))；覆盖判断（−86 dBm）用 P_tx·G；
   - `channel.toml`：`total_bandwidth_hz`（替换现有的子信道带宽键），`air_range_m = 300`；加“UAV 两两是否在范围内”；
   - `Router`：加“UE 到某架 UAV 是否覆盖”（现在的 `can_upload` 只看最近那架，直连要查 u2）和“某架 UAV 到 UE 是否覆盖”（回传直连，按 UAV 功率算）；只给出逐跳的（发送方, 接收方）：入口输入、UE 发给 UAV 的前驱结果、UAV 发给 UE 的前驱结果、回传（都是中继或直连，按第 6 条的参考时长选）和 UAV 之间的前驱结果；不再算每跳时长和能耗（共享带宽下由 runtime 算）。
13. 配置改成版本 D：按 §0.5.5 改 `config/*.toml` 和对应 dataclass；`dag.toml` 的边数据量、回传数据量两组键合成 `min_output_bits` / `max_output_bits`；`region.toml` `side_length_m = 600`、`hotspot_radius_m = 90`；`channel.toml` 见第 12 条；`scheduling.toml` 删 `relative_deadline_slots`、加 `deadline_s`、`result_return`；`user.toml` 加 `cpu_frequency_hz`；`uav.toml` 删 `queue_limit`；`load_settings` 里“`relative_deadline_slots` > `max_levels`”换成 0 < `deadline_s` ≤ 时隙长度。
14. 明确不做：内部断言、浮点容差（第 11 条剩余比特取 max(0, ·) 除外）、阶段对象、子信道 / 子载波、带宽或功率分配给智能体、“来不及就不发”的提前判断、跨时隙、干扰、UE 能耗。

### 0.5.10 基线（方法层）

1. 范围：`methods_v2/ers.py`（ERS 排序和参考时长小函数，照 MRDTS `methods/random/ers.py` 改写）、`methods_v2/baselines/policies.py`（四个基线和随机飞行，随机部分照 MRDTS `methods/random/run.py`；HEFT 没有可复用的现成实现，按下面规则写）、`experiments_v2/runs/baselines.py`。方法层只通过 `Simulator.reset` / `step` 交动作，不直接调用 runtime，不改环境代码；用自己的随机数生成器（§0.4.14 第 6 条）。
2. ERS（照 MRDTS `methods/random/ers.py`，只排序、不选执行节点）：
   - 每个要决策的 DAG 用快照里它的候选集合；平均计算时长 = `cpu_cycles` × 各候选主频倒数的平均（UAV 5 GHz、UE 1 GHz）；
   - 平均每比特传输时长 = 候选两两之间（含同一节点，同节点为 0）按 `Router.predecessor` 的路径、各跳 `transfer_duration_s(跳, 1 bit)` 之和的平均；
   - rank(i) = 平均计算时长(i) + max over 后继 c [`output_bits`(i) × 平均每比特传输时长 + rank(c)]，sink 只有平均计算时长；入口输入的上传不计入（同 MRDTS）；
   - placements 按 rank 从大到小统一排序，相等按 (dag_id, node_id)。⚠️ MRDTS 通信项用每条边的数据量、按核心数加权；这里用前驱的 `output_bits`、单核。
3. 四个基线，都按 ERS 排好序再交给 `step`：
   - 随机：每个子任务在候选里均匀随机选（同 MRDTS Random）；
   - 全本地：全部子任务放 UE（`None`）；
   - 贪心（整个 DAG 一个节点）：按 DAG 内最大 rank 从大到小逐个 DAG，放到预计完成最早的候选（含 UE）。预计完成 = max(入口上传参考时长（各入口取最大；UE 为 0）, 该节点本时隙已分到的计算量 ÷ 主频) + 本 DAG 总计算量 ÷ 主频 + 回传参考时长（各 sink 取最大；UE 为 0）（上传和排队同时进行，取较大者）；放完把本 DAG 计算量记到该节点；
   - HEFT 式（逐个子任务）：按 ERS 顺序逐个子任务（rank 大的先，前驱一定排在后继前面），对每个候选算预计完成 = max(数据就绪, 该节点空闲时刻) + `cpu_cycles` ÷ 主频；数据就绪 = 入口取上传参考时长，非入口取各前驱(预计完成 + 前驱节点到本节点的参考传输时长)的最大；节点空闲时刻 = 该节点上已放子任务的最晚预计完成（单核）；sink 选节点时再加上回传参考时长（UE 为 0；完成以回传到达为准），回传不占 CPU，节点空闲时刻只记计算完成；选最小者，相等取候选元组里靠前的。
   - 参考时长都用 `Router.transfer_duration_s` 沿 Router 给出的路径求和（独占全部带宽和功率，不看排队，同 ERS）。
4. 飞行：四个基线都随机飞行，每个分量在 [-1, 1] 均匀抽（同 MRDTS Random）。飞行用单独的随机数生成器，四个基线对同一 (seed, episode) 用同一个种子，因而 UAV 轨迹相同；随机基线选节点用另一个生成器，不影响飞行动作。
5. `experiments_v2/runs/baselines.py`：对给定的若干 (seed, episode)，每个基线跑完整 episode，打印按时完成率（COMPLETED ÷ offers）、超时数、四类能耗，以及跨 episode 的均值和标准差、`invalid_placements` 合计（应为 0）；核对同一 (seed, episode) 下四个基线的 offer 序列和每时隙 UAV 位置完全相同。用于 §0.4.12 第 2 条（随机与贪心要有明显差距）和 §0.5.7 第 9 条扫参。

### 0.5.11 MAPPO：MLP 版（方法层，先跑通）

> 师兄意见（11-06）：先不接超图，用 MLP 把强化学习跑通。算法照官方 MAPPO（Yu 等，NeurIPS 2022；代码 https://github.com/marlbenchmark/on-policy ，commit `de66d7a`）；通用学习代码的放法照 MRDTS `codex/development` 分支（`1714042`）的 `methods/learning/`。旧 `marl_models/mappo/` 不搬、不 import（§0.5.7 第 4 条）。⚠️ 超参照官方 MAPPO，不照 MRDTS 的 PPO（MRDTS：γ = 1、4 轮、熵系数 0.001、KL 提前停）。

1. 目录：
   - `methods/` 整体改名为 `methods_v2/`（与 `environment_v2/` 成对），同步改各调用处的 import；
   - 新增：
     ```text
     methods_v2/learning/        # 通用，不 import 环境
       valuenorm.py              #   照搬官方 onpolicy/utils/valuenorm.py
       distributions.py          #   照官方 algorithms/utils/distributions.py：带 mask 的 Categorical、DiagGaussian
       networks.py               #   照官方 algorithms/utils/mlp.py 与 r_actor_critic.py 的 critic：输入 LayerNorm、两层 64、ReLU + LayerNorm、正交初始化
       buffer.py                 #   一次更新的样本（按完整 episode 存）+ GAE（照官方 utils/shared_buffer.py 的 compute_returns，只留 GAE + ValueNorm 分支）
       mappo.py                  #   照官方 algorithms/r_mappo/r_mappo.py：ppo_update、cal_value_loss、train
     methods_v2/mappo/           # 本项目专用
       features.py               #   从快照构造局部观测、打分特征、mask、全局状态（第 3、4 条）
       policy.py                 #   actor（飞行头 + 子任务打分器）与 critic
       reward.py                 #   第 5 条
     config/methods/mappo.toml   # 第 6 条超参与 N̄
     experiments_v2/runs/train.py  # 采样、更新、评估、日志、保存（第 7 条）
     ```
   - 照搬的代码只取用到的类和函数，文件头写明来源文件和 commit；不搬 RNN、PopArt、active_masks、并行环境、学习率衰减；不用 `torch.compile`；超参从 toml 读，不用全局 `config`。
2. 智能体与动作：
   - 5 架 UAV 各是一个智能体，共用一套 actor 参数。每个时隙每架 UAV 输出：飞行动作（2 维）；它管理的本时隙新 DAG（出现在 `snapshot.candidates` 里、且 `manager_uav[ue_id]` 是它）中每个子任务的执行节点。覆盖外的 DAG 不决策（环境放 UE）；没有要管的 DAG 时只有飞行动作。
   - 执行节点：固定 N + 1 = 6 格，第 0 格是该 DAG 的 UE（交给 `step` 时写 `None`），第 1 + j 格是 UAV j；不在 `snapshot.candidates[dag_id]` 里的格 logit 置 −1e10（照官方 `distributions.py` 第 64–67 行）。
   - 各子任务独立采样。一架 UAV 的对数概率 = 飞行对数概率 + 它全部子任务的对数概率之和（§0.5.3 第 4 条；同官方 `act.py` 混合动作分支），PPO 比值按这个和算；熵同样相加。
   - 交给 `step` 前，placements 用 `methods_v2/ers.py` 的 `sort_placements` 排序（§0.5.9 第 6 条）；飞行动作裁到 [−1, 1] 再交，样本里存裁剪前的值（照官方连续动作）。
   - 评估用确定性动作：飞行取均值（裁到 [−1, 1]），执行节点取概率最大的格。
3. Actor 输入（只用该 UAV 能拿到的信息，范围同 §0.5.3 第 2 条；缩放见本条最后一项，第 4 条 critic 的特征同样缩放）：
   - 局部观测 o_i（定长，5 架 UAV 时 32 维）：自己位置（2）；其余 4 架 UAV 按编号各 5 维：是否邻居、相对位置（2）、邻居负载摘要（它本时隙管理的子任务数、总 cycles，§0.5.3 第 5 条），非邻居全填 0（20）；自己关联的被覆盖 UE：个数、相对重心（2）、相对自己四个象限各占这些 UE 的比例（没有关联 UE 时全 0）（7）；自己本时隙管理的新 DAG：DAG 数、子任务数、总 cycles（3）。
   - 打分特征（每个（子任务, 候选）一条）：子任务的 `cpu_cycles`、`output_bits`、是否入口、是否 sink、ERS rank、入口的 `input_bits`（非入口填 0，§0.5.7 第 11 条）、所在 DAG 的总 cycles 和子任务数；候选的是否 UE、是否自己、主频、负载摘要（自己用自己的，邻居用摘要，UE 填 0）；子任务在该候选上的计算时长、入口的上传参考时长、sink 的回传参考时长（用 `ERS.compute_duration`、`upload_duration`、`return_duration`，不适用的填 0）。
   - 网络：o_i → 主干 MLP → h_i；飞行头 = DiagGaussian(h_i)（均值线性层 gain 0.01，log_std 为可学习参数、初值 0，照官方）；打分器 = 小 MLP([h_i, 打分特征]) → 一个 logit，所有 UAV、所有子任务共用。两个 MLP 都照官方 `networks.py`。
   - 缩放（只在 `features.py` 顶部定义一处；分母能由 settings 算的都由 settings 算；不做运行时统计归一化）：
     - 个数除以最大可能值：UE 数、网格人数 ÷ `user.count`（60）；DAG 数 ÷ `user.count`（每个 UE 每时隙最多一个 DAG）；子任务数 ÷ (`user.count` × `dag.max_tasks`)（480）；单个 DAG 的子任务数 ÷ `dag.max_tasks`（8）；
     - 主频 ÷ `uav.cpu_frequency_hz`；单个子任务的 `cpu_cycles` ÷ 1.5e8；比特 ÷ 1e6；时长 ÷ `deadline_s`；位置、相对位置 ÷ `side_length_m`，其中关联 UE 的相对重心 ÷ `air_range_m`（让“用户在哪个方向”这类线索量级接近 1）；是否类特征取 0/1；
     - 三个总 cycles 除以对应算力（若也除以 1.5e8，典型值约 6.5 / 15 / 127，会压扁其他特征；Claude 按 §0.5.5 均值估算）：所在 DAG 的总 cycles ÷ (`user.cpu_frequency_hz` × `deadline_s`)，即全本地算完要几个截止期；自己管理的、邻居负载摘要里的总 cycles ÷ (`uav.cpu_frequency_hz` × `slot_duration_s`)，即单架 UAV 的负载率；critic 全局状态里的新 DAG 总 cycles ÷ (`uav.count` × `uav.cpu_frequency_hz` × `slot_duration_s`)，即全体 UAV 的负载（§0.5.5）。
4. Critic（集中式）：
   - 全局状态 s_t（定长，180 维）：5 架 UAV 的 o_i 拼接（照官方 `use_centralized_V`，160）；本时隙新 DAG 总数（含覆盖外）、其中覆盖外的个数、新 DAG 总 cycles（3）；全部 60 个 UE 在 4 × 4 网格里的人数（16）；剩余时隙比例 (T − t) / T（1）。
   - 团队奖励相同，所有 UAV 共用一个价值：每个时隙算一次，不加智能体 one-hot（官方 MPE 同；旧代码加了）。
   - 必须含新 DAG 总数：到达多的时隙奖励更负，与策略无关，靠 critic 扣掉。
5. 奖励（只做时延，所有 UAV 共用）：

   $$r_t = -\frac{1}{\bar N}\sum_{d\in\mathcal D_t}\frac{f_d-a_d}{D}$$

   - $\mathcal D_t$：本时隙新到的全部 DAG（`SlotResult.completed_dags` + `timed_out_dags`，含覆盖外的）；$a_d$ = `arrival_time_s`（时隙开始）；$f_d$ = `finish_time_s`（超时的 DAG 也有，环境照样算完）；$D$ = `deadline_s` = 1.0 s；$\bar N$ = 每时隙期望到达数，p = 0.24 时 19.5（§0.5.5），写在 `config/methods/mappo.toml`，扫 p 时同步改。
   - 求和后除以常数，不对本时隙求平均：每个 DAG 权重相同，对应“全部 DAG 平均时延”；没有 DAG 的时隙为 0，不用特判。$\bar N$ 只是缩放，不改变最优策略。
   - 不设上限（20 MHz、p = 0.24 下超过 3 s 的 DAG 约 0.6%，最长 5.19 s，自写统计；量纲由 ValueNorm 处理；上限会让最慢的 DAG 没有改进动力）；超时不另罚；越界不罚（越界者留在原地，`environment_v2/entities/uav.py` 第 31–37 行）；能耗只作指标（§0.4.13 第 2 条）。
   - `invalid_placements` 必须为 0；不为 0 说明 mask 有错，直接报错，不进奖励。
   - 依据：只做时延的 DAG 工作主流是“负时延 / 负时延增量，按全本地时延或截止期归一化”（DRLTO，IEEE TC 2022；Qi 等，NeurIPS 2025；见项目文献清单“11-06 时延奖励补充调研”）。环境每时隙从空队列开始、当场算完（§0.5.1 第 6 条），奖励在本时隙就能结算，不需要 Decima 式跨步记账。
6. 超参（`config/methods/mappo.toml`，全部照官方 on-policy `onpolicy/config.py` 默认）：

   | 项 | 值 |
   |---|---|
   | 学习率（actor、critic） | 5e-4，Adam eps 1e-5，weight_decay 0 |
   | ppo_epoch / num_mini_batch / clip | 15 / 1 / 0.2 |
   | entropy_coef / value_loss_coef / max_grad_norm | 0.01 / 1 / 10 |
   | γ / GAE λ | 0.99 / 0.95；episode 末 `done` 不自举（官方 `use_proper_time_limits` = False） |
   | ValueNorm、Huber（δ = 10）、value 裁剪（在归一化量纲里）、优势按批归一化 | 全开 |
   | 网络 | hidden 64、layer_N 1、ReLU、输入 LayerNorm、正交初始化、动作输出层 gain 0.01 |
   | 每次更新的样本 | 官方 MPE 是 32 个并行环境 × 200 步；本项目每次更新跑 `episodes_per_update` = 32 个完整 episode（每个 200 时隙），由 `rollout_workers` = 32 个子进程同时跑（第 7 条），样本量与官方相同 |
   | 更新时 torch 线程数 `training_threads` | 8（只影响速度；采样子进程各用 1 个线程） |
   | $\bar N$ | 19.5（第 5 条） |

7. 训练与评估（`experiments_v2/runs/train.py`）：
   - 训练用 (seed = 训练种子, episode = 0, 1, 2, …)，episode 不重复；训练种子不用 1（1 留给评估集），多种子时取 10、11、12；评估固定用 (seed = 1, episode = 0–9)，与训练不重叠，确定性动作；基线也在同一组 (seed, episode) 上跑一次（`experiments_v2/runs/baselines.py`），同口径比较。
   - 方法层用自己的随机数生成器（§0.4.14 第 6 条）：采样用每个 episode 单独派生的 torch 生成器（见下一项）；minibatch 打乱用由训练 seed 派生的 numpy 生成器。CPU 即可。
   - 并行采样（11-06 起）：每次更新把当前 actor、critic 参数发给子进程，每个子进程跑一个完整 episode，主进程按 episode 编号顺序收样本；32 个 episode 用同一套参数（参数要等全部跑完才更新），与顺序跑等价。每个 episode 的采样随机数由 `np.random.SeedSequence([train_seed, episode])` 单独派生，结果与进程调度无关、可复现。评估的 10 个 episode 同样并行（确定性动作，不用随机数）。理由：服务器上一个 episode 约 6.5 s，顺序跑一次更新约 4.5 min（采样约 3.5 min + 更新约 1 min，按 11-06 冒烟外推），300 次约 22 h；并行后一次更新约 1 min 多。
   - 每次更新打印：训练 episode 的平均回报、按时完成率、平均时延、被覆盖 DAG 占比；value loss、policy loss、熵、近似 KL、裁剪比例、actor 与 critic 梯度范数；每个 episode 用时。每 10 次更新评估一次，打印评估集上的按时完成率、平均时延、被覆盖 DAG 占比、四类能耗。
   - checkpoint 和日志写到 `output/runs/<实验名>/seed<k>/`（§0.5.8 第 5 条）。
8. 验收：
   - 冒烟（服务器，写到 `output/checks/`）：`episodes_per_update` = 1、2 次更新能跑完；`invalid_placements` 合计为 0；采样到的执行节点都在候选里；ValueNorm 的均值、方差随更新变化；打印每个 episode 用时，据此定服务器上的更新次数。
   - 奖励核对：用贪心基线跑一个 episode，−Σ r_t × $\bar N$ ÷ 该 episode 的 DAG 总数，应等于该 episode 的平均（时延 ÷ 截止期），只差浮点误差（11-06 冒烟：0.6746593930957117 与 0.674659393095712）。
   - 并行核对：`episodes_per_update` = 4 时，`rollout_workers` = 1 与 = 4 各跑 2 次更新，每个 episode 的指标和每次更新的损失应完全相同。
   - 服务器训练：先跑 50 次更新，看回报曲线再定总次数；回报曲线上升、评估按时完成率明显高于随机基线（20 MHz、p = 0.24 时 70.8%，§0.5.5）即算跑通（11-06 评估集 seed = 1、episode 0–9 上：随机 71.1%、全本地 65.5%、贪心 85.8%、HEFT 式 86.3%）。HEFT 式 85.7%、其中覆盖内 99.5%，覆盖是主要瓶颈，超过它要靠学会的飞行，不作为跑通标准。
9. 先记下、出现再处理（不预先写代码）：
   - 一架 UAV 管很多子任务时，对数概率之和的比值容易被裁剪（裁剪比例高、KL 大）；出现时改成官方 multi-discrete 的逐子任务比值。
10. 指标与图（正式实验都按这个记、按这个画）：
   - 主指标：平均时延（奖励由它算）、按时完成率（§0.4.15）。
   - 解释指标：被覆盖 DAG 占比（看飞行）；覆盖内 DAG 的按时完成率与平均时延（看放置）；覆盖外 DAG 的按时完成率与平均时延（不受放置影响，作对照）；覆盖内子任务放到 UE、管理 UAV、邻居 UAV 的比例。
   - 训练健康度：value loss、熵、飞行 log_std（两个分量）、近似 KL、裁剪比例。
   - 能耗四项只记录，暂不画。
   - 训练 episode 和评估 episode 都记以上指标；`runs/baselines.py` 在评估集上记同一套主指标和解释指标。
   - `runs/plot.py` 读一个实验目录（可含多个种子）和基线结果，出四张图：① 评估集平均时延–更新次数；② 评估集按时完成率–更新次数；③ 被覆盖 DAG 占比、覆盖内按时完成率–更新次数（①–③ 多种子画均值 ± 标准差，基线画水平线）；④ 训练健康度 2 × 3（value loss、熵、log_std、近似 KL、裁剪比例、训练回报滑动平均）。轨迹图由 `checks/replay.py` 出。
   - 判断“学会了”：评估集平均时延持续下降并稳定；稳定后好于对应基线；至少 3 个种子一致；解释指标和轨迹说得通。只看训练回报不够：11-06 的 50 次更新里回报几乎不动，是放置变好与覆盖下降相互抵消。
   - 论文用图（主对比柱状图、时延 CDF、扫参曲线、消融）等方案定型后再加。
11. 验证路线（只看时延）：
   - S1 放置固定为 HEFT 式、只学飞行。目标：评估集平均时延低于 HEFT 式 + 随机飞行（11-06 评估集 0.683 s），被覆盖 DAG 占比明显高于随机飞行（约 61%）；以后再和 §0.5.7 第 10 条的简单飞行启发式比。设置：
     - 放置每个时隙直接用 `methods_v2/baselines/policies.py` 的 `choose_heft`，不学；飞行单独一个 actor（o_i → 主干 MLP → DiagGaussian），不再和放置共用主干；PPO 的对数概率、比值和熵只算飞行；critic、奖励、超参、评估集同前几条。依据：11-06 的 50 次更新里放置在学（覆盖内按时完成率 75% → 90%），飞行几乎没学到（log_std 0 → 0.2，5 架飞行均值几乎相同，诊断见 `docs/superpowers/reports/2026-11-06-mlp-mappo-results.md`）；师兄 MRDTS `codex/development` 也是飞行、调度两个 actor 分阶段训练，奖励只用时延、不加越界惩罚。越界规则不变、不罚。
     - 三个方案，只差飞行 actor 的参数怎么共享，用 `config/methods/mappo.toml` 的一项 `flight_actor` 切换：`shared`（5 架共用一套）；`per_uav`（每架一套）；`shared_id`（共用一套，o_i 末尾加 5 维 UAV 编号 one-hot，同官方 on-policy SMAC 环境的 `obs_agent_id`）。
     - 每个方案 3 个训练种子（10、11、12），各 100 次更新；实验名 `s1-shared_<月日>`、`s1-peruav_<月日>`、`s1-id_<月日>`。
     - 判据：§0.5.11 第 10 条；另看 5 架的飞行均值是否各不相同、是否朝自己关联用户的方向（`checks/replay.py` 输出飞行均值方向与“指向关联用户重心”方向的平均余弦）。`runs/plot.py` 能把多个方案画在同一张图上对比。
   - S2 飞行与放置一起学（MLP）：不差于 S1。
   - S3 换超图编码（论文方法）：优于 MLP 版，负载越高差距越大。
   - S4 扫负载、带宽，超边类型消融。
