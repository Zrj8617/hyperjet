# HyperUAV 环境框架重构：需求汇总与初步方案草案

> **文档状态：§0.4 已由用户确认（2026-10-13）；师兄同意先实现、再逐阶段评审**  
> **日期：2026-10-10（2026-10-12 更新：新增并补充 §0.4 已确认决定；2026-10-13 再次更新）**  
> **用途：供用户先自行审核，再与师兄讨论；同时作为后续 AI 理解需求的结构化输入。**  
> **不是：第二份 master roadmap 或代码删除授权。2026-10-13 起授权进入 Gate 1–2（类型、配置、实体）。**

## 0. 文档地位与阅读规则

### 0.1 单一事实源不变

本项目唯一活动状态文档仍是：

`docs/research/HyperUAV_research_master_roadmap.md`

本文只整理“环境框架重构”的需求、代码核查事实、初步建议和开放问题，不替代 Living Research Charter，不记录新的研究阶段结论。

### 0.2 本文使用的结论标签

- **[用户需求]**：用户已经明确表达的目标、担忧或约束。
- **[已验证]**：已经通过读取当前代码、项目文档或师兄仓库代码核实的事实。
- **[初步建议]**：当前推荐方向，尚未获得用户和师兄批准。
- **[待确认]**：目前无法仅从代码确定，需要用户或师兄裁决。
- **[禁止]**：在当前阶段明确不能执行的事项。

任何 AI 接手本文时，不得把“初步建议”写成用户已经批准的决定，也不得把“待确认”自动补成默认答案。

### 0.3 当前版本背景

本文起草时，本地仓库状态为：

- 分支：`param-realign-20260914`
- HEAD：`d2214c99fe8fcbb1861a7bc2492b1e6157c51760`
- 工作树不是 clean；存在用户已有的 modified/untracked 文档、计划、报告、`output/` 和 `tools/` 等内容。

这些未提交内容不属于本任务，不得清理、覆盖或顺手提交。

师兄仓库的只读核查对象为：

- 服务器路径：`/data2/zrj2025/mrdts`
- 分支：`main`
- 核查时 HEAD：`17b256c8de3c15da36f7c2140825b87270227a80`

## 0.4 已确认决定（2026-10-12，用户与师兄讨论后确认）

> 本节优先于本文其他章节中与之冲突的表述。其他章节中的“待确认/初步建议”如与本节冲突，以本节为准。

### 0.4.1 总体做法

1. 并行新建 `environment_v2/`（名称切换时再定），旧 `environment/` 在新环境跑通前不改。
2. 架构原则上参照师兄 MRDTS 来写，但去掉其防御性检查：只在“读取配置”和“接收算法动作”两个入口做校验；实体、通信、runtime 内部不写 shape/类型/有限值检查，不写失败回滚。
3. 验收标准：新环境跑通、debug 无问题即可；本轮不跑基线对比、不跑训练实验。
4. 旧 MARL（MADDPG/MATD3/MASAC 等）与训练侧（`scripts/train_clean_mainline.py`）本轮不动，但新环境不为它们做任何兼容。

### 0.4.2 实体

1. 实体为 UE、UAV、DAG，外加 Region；没有基站（BS）。子任务不是独立实体，而是 DAG 内的节点，用 `(dag_id, node_id)` 标识（同 MRDTS）。
2. 属性大体沿用当前代码；如有更清爽的写法（包括 MRDTS 的写法），采用更好的版本。逐字段的保留、删除、改名、迁出以 §0.4.10 字段表为准。
3. Region：借用 MRDTS“给位置返回区域 id”的接口思路（方法名 `get_region_ids`），不搬其 Voronoi 网格实现。区域只有两类：热点区（id=1）与非热点区（id=0）；热点为若干个半径相同的完整圆（整个圆在地图内），个数每个 episode 在 [`min_hotspots`, `max_hotspots`] 中均匀抽取（两者相等即固定个数），中心每个 episode 重新采样，任意两个中心距离不小于 2 倍半径（互不重叠）；位于任一热点内即为热点区（2026-10-15 改）。
4. UAV：只有一种 UAV，写成单个 `UAV` 类，合并 MRDTS 的 `UAV` 父类与 `MemberUAV` 中需要的部分：
   - 集合写法：一个对象保存全部 UAV，`positions` 为 (N, 3) 数组，按下标访问；
   - 位置沿用现有三维 (x, y, z)：UE 的 z=0，UAV 的 z 为飞行高度（当前 UE/UAV 位置本来就是三维；二维的只是 clean 通信公式和 DAG/Task 的 `source_pos`，二者都在本轮改掉）；
   - 不保存 `region_ids`：需要时用 `Region.get_region_ids(uav.positions)` 现算（2026-10-13 改）；
   - 飞行采用 MRDTS `apply_flight_actions` 的连续动作：每架 UAV 一个 [-1,1]^2 归一化速度 × 最大水平速度 × 飞行时长，越界则保持原位并返回越界标记；弃用当前 5 个离散动作（`CLEAN_MOVEMENT_ACTIONS`）；
   - 单核：不设核数和逐 UAV 主频数组，主频为 `UAVConfig.cpu_frequency_hz` 一个标量（UAV 同构；2026-10-13 改，以后需要多核或异构时再加）；
   - 计算能耗改用 MRDTS 的电容模型 κ·C·f²；计算量 `num_operation` 自 2026-09-14 起已按 CPU cycles 标定，只改名为 `cpu_cycles`，数值不重新缩放；
   - 保留高度、发射功率；
   - 保留飞行能耗（用户 2026-10-12 决定）：采用标准旋翼无人机推进功率公式 P(V) = P0·(1 + 3V²/U_tip²) + Pi·(√(1 + V⁴/(4v0⁴)) − V²/(2v0²))^{1/2} + ½·d0·ρ·s·A·V³（Zeng et al., IEEE TWC 2019）。MRDTS 的 `u1–u5` 在其代码中没有对应公式，不沿用；参数直接按公式中的物理量命名（叶片轮廓功率 P0、诱导功率 Pi、叶尖速度 U_tip、悬停诱导速度 v0、机身阻力比 d0、空气密度 ρ、旋翼实度 s、旋翼盘面积 A），数值最后再定。环境每时隙按各 UAV 的飞行速度输出飞行能耗（焦耳）；悬停也按该模型计耗能（当前代码是“移动则按常数功率 × 时隙长度计、悬停为 0”，见 `env.py:717`，这一语义随之改变）；每时隙飞行能耗 = P(V)×`flight_duration_s` + P(0)×(时隙长度 − `flight_duration_s`)，越界未动的 UAV 按 V=0 计（悬停整个时隙），写成 `UAV` 的方法（2026-10-13：师兄建议能耗写进实体；结构同旧 `uavs.py:354` 的飞行+悬停分段，常数功率换成上式）；
   - 去掉：BS 相关、`slot_energy`、Master 相关与按区域分配 Member 的逻辑、`coverage_radius`；
   - UAV 不保存剩余能量：clean 主线从未调用扣减能量的函数，`remaining_energy_ratio` 恒为 1.0（graph 特征 `graph_builder.py:291` 读到的是常数），删除不影响结果；覆盖半径 `UAV_COVERAGE_RADIUS` 只被位置塑形奖励（`env.py:752`）使用，不是 UAV 的物理属性，移到 reward 配置（Gate 7，见 §0.4.13）；
   - 初始位置：地图内均匀随机（沿用当前做法）。
5. UE：集合写法；初始位置（2026-10-15 改，参照 MRDTS `User` 按区域分配人数）：各区域权重 = 面积 × 密度倍数（热点为 `hotspot_density`，非热点为 1）× U(1 − `area_fluctuation`, 1 + `area_fluctuation`)，归一化后按 MRDTS 的取整方法（先向下取整，余数按小数部分从大到小补）分配人数；热点内的 UE 在圆内均匀采样，非热点的 UE 在地图内、所有热点外均匀采样；不采用 MRDTS 的向区域中心聚集；热点内减速（2026-10-15 定）：移动前位于任一热点内的 UE，本步位移乘 `hotspot_speed_scale`（高斯-马尔可夫的速度状态本身不变），只看位置、与策略无关；稳定后热点密度约为非热点的 1/`hotspot_speed_scale` 倍，取 1/`hotspot_density` 时开局分布即为稳定分布（不减速时开局多放在热点的 UE 约 50 个时隙后走散）。高斯-马尔可夫移动，每个时隙的位移 = 速度 × `move_duration_s`（`user.toml`，默认取时隙长度；设为飞行时长即“只在飞行阶段移动”，换算法只改这个数，不保留两套代码）；去掉每时隙最大移动距离限制（`UE_MAX_DIST`、`UE_GM_MAX_SPEED`），只保留速度不小于 0 和地图边界反弹；取消等待服务降速（见 §0.4.14）。UE 位置每个时隙只在飞行阶段更新一次，本时隙之后的上传、回传都用这个位置。

### 0.4.3 通信

1. 信道公式沿用 MRDTS `ChannelModel`：3D 距离、仰角 LoS 概率（Al-Hourani 城市参数 a=9.61, b=0.16）、NLoS 衰减系数 η、g0/d^2、香农速率。空对空链路按纯 LoS（LoS 概率取 1，不用仰角公式；师兄 2026-10-16 确认空中不考虑遮挡），地对空、空对地按仰角公式。2026-10-17 改：
   - 接收功率显式计算：P_rx = P_tx × [P_LoS + (1 − P_LoS)·η] × g0 / d^2，g0 = −60 dB（1e-6）；
   - 噪声 = 噪声功率谱密度 N0 × 子信道带宽 b，N0 = −174 dBm/Hz（b = 1 MHz 时噪声为 −114 dBm）；取代原固定 `noise_power_w`；
   - 速率 = b × log2(1 + P_rx / (N0 × b))；所有链路类型用同一个子信道带宽 b；
   - 不考虑干扰：子信道之间正交，也不考虑不同 UAV 之间的同频干扰（参考 Zhao TMC 2025、VEC TMC 2024）。
2. 删除 DAG 上按 DAG 随机抽取的 `base_upload/download_bandwidth_mbps`。
3. 多址与带宽上限（2026-10-17，师兄确认“把总带宽限制住”；取代原“本轮不加信道排队”）：采用 OFDMA，每架 UAV 有 K 个正交子信道（暂定 K = 4、b = 1 MHz，即每架 UAV 总带宽 4 MHz），收和发共用这 K 个。MRDTS 每条有向链路独立、各用满带宽，相当于子信道无限多，本项目不沿用。每一跳传输从开始到结束占用一个子信道：
   - UE → UAV 占接收 UAV 的一个；UAV → UE 占发送 UAV 的一个；UAV → UAV 发送方、接收方各占一个，两边都有空闲才开始；UE 一端不设限制；
   - 子信道全忙时排队：按安排顺序 FIFO，取最早空闲的子信道，不插空；一跳的开始时刻 = max(这一跳数据就绪时刻, 所需子信道的最早空闲时刻)；
   - 只要还有空闲子信道，一架 UAV 可以同时收和发（用不同子信道，不涉及同频全双工的自干扰）；
   - 排队等待不耗能，能耗 = 传输时长 × 发送方功率。
   例：K = 4，UAV A 正在收 3 个 UE 的上传，剩 1 个子信道，可以同时把一个结果发给 UAV B；此时第 4 个 UE 要上传，就得等前面有一个传完。
4. 覆盖（2026-10-17）：只对 UE 上传检查。UE 到离它最近的 UAV 的接收功率低于 `min_received_power_dbm`（暂定 −86 dBm；UE 0.1 W、UAV 高 100 m 时，水平覆盖约 154 m，三维约 183 m）时，该 UE 的入口子任务本时隙不能分配，下个时隙再判断（参考 Zhao TMC 2025：覆盖外的用户不能卸载）。空对空和回传不检查（若空对空也检查，地图对角上的两架 UAV 会互相不可达）。判断用分配时的位置（同第 6 条）。
5. 路由（2026-10-13 采用中转；2026-10-17 按第 3 条更新）：
   - 入口子任务上传：UE →（地对空）接入 UAV →（空对空）执行 UAV；接入 UAV 为分配时离 UE 最近的 UAV，若它就是执行 UAV 则只有一跳；
   - 前驱结果：前驱子任务所在的 UAV 直接发给后继子任务所在的 UAV（空对空），同一架则不传。一个后继有多个前驱时：同一源 UAV 的多份结果合并成一次传输（数据量相加），占一个子信道；不同源 UAV 各自一次传输，子信道够就同时传，不够就按第 3 条排队；后继的数据就绪取各传输中最晚到达者；
   - 结果回传：在 sink 计算完成的时隙计算，sink 所在 UAV →（空对空）当时离 UE 最近的 UAV →（空对地）UE，同一架则一跳；
   - 多跳传输先收完再转发，每一跳各自按第 3 条占用子信道（中转 UAV 收一跳、发一跳，都占用它的子信道）；能耗按各跳发送方功率计算；
   - 计算与传输分开（2026-10-17，取代原“服务台占用”规则）：每架 UAV 有一条 CPU 时间线（单服务台、按分配顺序 FIFO，只排计算）和 K 个子信道（只排传输），两者互不占用；子任务计算开始 = max(所有输入数据到达时刻, CPU 空闲时刻)；
   - 位置直接读当前 `positions` 数组，DAG 不保存 `source_pos`。
6. 估计器（assignment）与执行器（executor）共用 executor 的 `plan()`：给定子任务、执行 UAV 和当前位置，返回各段时间与能耗，不改状态。估计器对回传也按当前位置估算；执行器在安排回传时（sink 计算完成的时隙末，见 §0.4.4 第 3 条）按当时的位置重新计算链路、时长和能耗并冻结，之后不再改（那时离 UE 最近的 UAV 可能已与分配时不同）。与上传、前驱传输一样，传输实际开始可能晚于计算时刻，仍用计算时刻的位置（2026-10-13 改）。
7. Gate 3 与 Gate 4 的分工（2026-10-16；2026-10-17 更新）：Gate 3 只提供只读计算——单跳计算（两端位置、数据量、发射功率、链路类型 → 接收功率、速率、时间、能耗）、上传覆盖判断（第 4 条）、路由计算（上传 / 前驱结果 / 回传三类，返回每一跳的发送方、接收方、链路类型、时间、能耗）；子信道占用与排队、每跳开始时刻、多个前驱按源 UAV 合并、CPU 时间线，由 Gate 4 的 `plan()` 与 runtime 负责。

### 0.4.4 时间线

1. 保持当前时间线：每架 UAV 单服务台、按分配顺序 FIFO、分配时即算出完整时间线、子任务与 DAG 可跨时隙执行。2026-10-17 起单服务台只排计算，传输占用子信道、不占用 CPU（§0.4.3 第 3、5 条）。
2. 时隙内顺序（2026-10-13 更新）：时隙开头产生到达（UE 不动）并冻结就绪子任务 → UAV 飞行 `flight_duration_s`、UE 同时移动一步 → 本时隙的新分配从“时隙起点 + `flight_duration_s`”开始，直到时隙结束。时钟连续，飞行期间已分配的子任务照常推进；时隙开头到达的子任务等待的这段时间计入完成时延（MRDTS 的任务不跨时隙，所以能把飞行时间排除在时延之外，本项目不能）。
3. 以下两点本轮原样保留现有行为（用户 2026-10-12 批准，不阻塞 Gate 0→Gate 1），以后再与师兄确认是否修改：(a) 依赖只在时隙边界释放；(b) sink 回传在计算完成的时隙才补算（2026-10-17 起占用子信道，不再后推 CPU 空闲时刻）。具体规则（2026-10-13 明确，改正旧代码中计算与回传的重叠）：
   - 分配 sink 时与其他子任务一样，把执行 UAV 的空闲时刻推到它计算完成的时刻（旧代码 `task_execution.py:296` 对 sink 不推，会让后续任务与 sink 计算重叠）；
   - 回传在 sink 计算完成的时隙末安排，每一跳按 §0.4.3 第 3 条占用子信道：开始时刻 = max(这一跳数据就绪时刻, 所需子信道最早空闲时刻)，第一跳的数据就绪时刻为 sink 计算完成时刻；已排好的记录不顺延（2026-10-17 改）；
   - 若第一跳开始时刻不早于所属 DAG 的截止时刻，则不安排回传（即 §0.4.15 第 4 条的“未开始的回传不再开始”）。

### 0.4.5 配置

1. 仿照 MRDTS：`config/*.toml` 存数值，`environment_v2/settings/*_config.py` 用 dataclass + `tomllib` 读取（训练环境为 Python 3.12）。
2. 文件划分：`region.toml`（地图边长、热点个数上下限 `min_hotspots`/`max_hotspots`、热点半径、热点到达倍数）、`user.toml`（UE 数量、热点密度倍数 `hotspot_density`、人数浮动 `area_fluctuation`、热点内速度倍数 `hotspot_speed_scale`、移动参数、`move_duration_s`、发射功率、基础到达概率）、`uav.toml`（数量、高度、最大速度、飞行时长、主频、电容系数、发射功率、队列上限、推进功率公式参数）、`dag.toml`（任务数、层数、父节点数、输入数据量、边数据量、回传数据量、计算量参数）、`channel.toml`（子信道带宽 `subchannel_bandwidth_hz`、子信道个数 `subchannels_per_uav`、噪声功率谱密度 `noise_density_dbm_per_hz`、最小接收功率 `min_received_power_dbm`、g0、NLoS 系数、LoS 参数；2026-10-17）、`scheduling.toml`（时隙长度、到达时隙数 `arrival_slots`、截止时隙数 `relative_deadline_slots`，见 §0.4.15）。
3. 旧 `config.py` 保留给旧代码；训练/算法参数不进入新配置。
4. 读取方式：每个 dataclass 用 `cls(**data)` 直接构造，缺键、多键由 Python 报错，不另写逐键校验。只检查会让仿真静默出错的几个值：时隙长度、飞行时长大于 0，飞行时长不超过时隙长度，`relative_deadline_slots` 大于 DAG 最大层数。

### 0.4.6 数值最后再定（先写公式，不在本轮调参）

地图边长（调大）、热点半径、UE 移动速度（去掉上限后需重新设定）、UAV 高度、最大速度与飞行时长（MRDTS 时隙为 0.5 s 飞行 + 1.0 s 调度，本项目时隙为 5 s，数值含义不同）、子信道带宽与个数、最小接收功率、发射功率、数据量单位（MB 或 bit 统一一种）、核频率、电容系数、推进功率参数、截止时隙数 D 与到达时隙数。注意（2026-10-16）：时隙 5 s 而飞行只占 0.5 s，悬停占每时隙推进能耗的绝大部分（每架约 800 J/时隙，飞行决策只能改变约 ±2%），定飞行时长、时隙长度时要一并考虑。

### 0.4.7 智能体

1. 师兄所说“两个智能体不够”指当前的两个 actor（movement、offloading）；属于算法设计，后置。
2. 环境只接收：每架 UAV 一个连续二维飞行动作；从本时隙冻结的就绪任务列表中选出的子序列（必须保持冻结列表中的先后顺序，FIFO 顺序由该列表决定），每个子任务指定一个执行 UAV（未被分配的子任务保持 ready，沿用当前行为）。任意改变任务顺序属于以后的智能体设计（Gate 9）。环境不依赖 actor 数量。
3. movement actor 由离散输出改为连续输出属于算法层工作，不在环境重构中顺带完成。在此之前，入口切换（Gate 7）时由环境外的 adapter 用一张 5 行映射把离散动作翻译成连续向量：hover→(0,0)、+x→(1,0)、-x→(-1,0)、+y→(0,1)、-y→(0,-1)；actor 改为连续输出后删除该映射。环境内只有连续动作一套代码，不用注释或开关保留离散版本。

### 0.4.8 环境接口与随机数

1. 时隙接口照 MRDTS 用两个方法，时隙内顺序见 §0.4.4 第 2 条：
   - `reset(seed, episode)`：按 §0.4.14 创建随机流，初始化实体与热点，完成第一个时隙开头的到达（UE 不动）和就绪子任务冻结，返回快照；
   - `begin_slot(flight_actions)`：`flight_actions` 为 (N, 2) 数组，元素在 [-1, 1]；UAV 飞行 `flight_duration_s`，UE 同时按 `move_duration_s` 移动一步，返回供卸载决策使用的快照；
   - `end_slot(placements)`：`placements` 为 ((dag_id, node_id), uav_id) 列表，子任务顺序须保持冻结就绪列表的顺序（见 §0.4.7 第 2 条）；从“时隙起点 + `flight_duration_s`”提交分配、推进到时隙末并结算，返回本时隙结果 `SlotResult`（含结束标志 `done`）。未结束时进入下一时隙开头（到达窗口内产生到达；冻结就绪子任务；UE 不动）并返回其快照；`done=True` 时不再进入下一时隙（不产生到达、不消耗随机数），下一时隙快照为 `None`。`done` 的判定见 §0.4.15 第 6 条。
   - 快照不复制数据，直接给出实体数组和 DAG 记录的只读视图，调用方只读不写（同 MRDTS）。
   - `SlotResult` 只放物理结果：本时隙完成的 DAG（带到达、完成时刻）和超时的 DAG（带到达时刻；截止时刻由到达时刻算出）、四类能耗（焦耳）、不合法分配个数、本时隙的 offer、`done`。四类能耗中，计算、上传与前驱传输、回传三类是本时隙新承诺的服务能耗：计算、上传与前驱传输在本时隙接受 placement 时按算好的时间线计入，回传在本时隙安排回传时计入，任务完成时只更新状态、不再累计；飞行能耗是本时隙实际飞行的能耗（2026-10-13 改）。reward 和指标由方法层、实验层从中计算。
2. 入口校验只做在动作上：飞行动作的形状与取值范围；placement 中的 uav_id 存在、task 属于本时隙冻结的就绪集合且未重复、UAV 队列未满。不合法的分配不执行并计数（沿用当前行为）。
3. 随机数：见 §0.4.14；不再依赖导入 `config.py` 时设置的全局 `np.random`。seed 由实验入口提供，不写进 toml。

### 0.4.9 内部检查（取代 §10.3）

1. 内部断言 0 条。非法内部状态通过数据结构和单一持有者消除，而不是事后检查：
   - 每个任务只有一个 `status` 字段（`TaskStatus` 枚举），不再同时维护 `state` 和 `service_phase`；
   - 状态只由 runtime 修改，不对外公开 `mark_task_*` 之类的改状态方法；
   - UAV 队列、空闲时刻、任务执行记录只存在于唯一的 runtime 中，其他模块只读快照；
   - 时隙顺序：`begin_slot` / `end_slot` 两个方法并列（同 MRDTS），由唯一的调用方（训练/评估循环）按固定顺序调用，不做检查，也不为防误用增加阶段对象等额外结构。
2. 校验只在两个入口：读取配置、接收算法动作（见 §0.4.8 第 2 条）。
3. UE、UAV 的编号固定等于数组下标，reset 后不变。

### 0.4.10 字段表（v2）

| 实体 | 现字段 | v2 | 说明 |
|---|---|---|---|
| 子任务 | `task_id`、`dag_id` | `(dag_id, node_id)` | 子任务不再是独立实体，是 DAG 内的节点（同 MRDTS）；节点编号即拓扑序 |
| 子任务 | `ue_id`、`arrival_time` | — | 属于 DAG，不在子任务上重复保存 |
| 子任务 | `input_data_size_mb` | DAG 的 `input_bits[node]` | 只有入口子任务从 UE 上传；也用于生成计算量；单位最后定 |
| 子任务 | `output_data_size_mb` | DAG 的 `edge_bits[edge]`、`return_bits[node]` | 数据量改放在边上（每条依赖边一个值，与 `edges` 一一对应，同 MRDTS）；sink 回传 UE 的数据量单独存为 `return_bits`（非 sink 为 0，MRDTS 无回传）；取值范围沿用现在的输出数据量范围 |
| 子任务 | `num_operation` | DAG 的 `cpu_cycles[node]` | 改名，数值不变 |
| 子任务 | `level` | DAG 的 `levels[node]` | 保留：只有 graph 使用，但生成后无法从 `edges` 推出 |
| 子任务 | `topological_index` | — | 节点编号本身就是拓扑序 |
| 子任务 | `predecessors`、`successors` | DAG 的 `edges` | 依赖关系只存一份；前驱、后继和 sink 由 runtime 从 `edges` 推出 |
| 子任务 | `state` + `service_phase` | `TaskRuntime.status: TaskStatus` | WAITING / READY / SCHEDULED / RETURNING / DONE / CANCELLED；SCHEDULED 表示已分配、时间线已算好（可能仍在上传、排队或计算），细分阶段由时间线推出；CANCELLED 表示所属 DAG 超时时尚未分配（§0.4.15） |
| 子任务 | `ready_time`、`start_time`、`finish_time`、`compute_finish_time` | `TaskRuntime` 的 `ready_time_s`、`start_time_s`、`finish_time_s`、`compute_finish_time_s` | 只由 runtime 写；加 `_s` 以区别时隙序号 |
| 子任务 | `assigned_uav` | `TaskRuntime.uav_id` | 只由 runtime 写 |
| 子任务 | `compute_energy`、`communication_energy`、`return_energy`、`total_energy` | `TaskRuntime` 的 `compute_energy_j`、`tx_energy_j`、`return_energy_j` | total 删除（可由三者求和） |
| 子任务 | `is_critical_path` | — | 迁到方法层计算 |
| 子任务 | `reward_completion_time`、`reward_settled` | — | 迁到 reward 层 |
| 子任务 | `task_complexity`、`task_constant` | — | 删除（只在生成计算量时使用） |
| 子任务 | `source_pos`、`enqueue_time`、`deadline`、`task_type`、`input_size`/`output_size`/`cpu_cycles` 别名、`remaining_slack` | — | 删除 |
| DAG | `dag_id`、`ue_id`、`arrival_time`、`task_ids` | `id`、`ue_id`、`arrival_time_s`、`num_tasks` | `id` 按 offer 顺序递增；每个 offer 都成为 DAG（§0.4.15）；截止时刻由 `arrival_time_s` 和 D 算出，不另存 |
| DAG | `sink_task_ids` | — | 由 runtime 从 `edges` 推出 |
| DAG | `return_complete_time` + `completed` | `DAGRuntime.status: DAGStatus`、`DAGRuntime.finish_time_s` | 运行状态移入 runtime；ACTIVE / COMPLETED / TIMED_OUT；`finish_time_s` 为最后一个 sink 回传完成的时刻，只在 COMPLETED 时写入 |
| DAG | `base_upload_bandwidth_mbps`、`base_download_bandwidth_mbps`、`source_pos` | — | 删除（带宽由子信道决定，见 §0.4.3 第 3 条；位置取 UE 当前位置） |
| DAG | `khop_hyperedges_global` | — | 迁到方法层（graph） |
| DAG | `completion_reward_settled` | — | 迁到 reward 层 |
| UE | `pos`、`speed`、`theta`、`velocity` | `positions`、`speeds`、`headings` | 与 UAV 一致采用集合写法（用户 2026-10-12 确认）；速度向量由 `speeds`、`headings` 现算，不另存（2026-10-13） |
| UE | `service_waiting`、`active_dag_id` | — | 删除；等待服务降速已取消（§0.4.14），UE 是否有活动 DAG 由 DAG 记录推出；一个 UE 可同时有多个活动 DAG（§0.4.15） |
| UAV | 见 §0.4.2 第 4 条 | `positions` | 高度、最大速度、飞行时长、主频 `cpu_frequency_hz`、发射功率、电容系数放在 `UAVConfig`；区域 id 现算 |
| Region | — | `centers`（K×2）、`radius_m`、`get_region_ids(positions)` | 热点个数上下限、到达倍数放在 `region.toml`（2026-10-15） |

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

### 0.4.12 验收（取代原 Gate 6 的新旧逐项对照）

1. 一个一次性检查脚本（约 100 行，放在 `scripts/` 下，不被环境 import，不设统计门槛）：
   - DAG 结构性质逐样本检查：无环、父任务来自更早的层、任务数与父节点数在配置范围内；
   - 固定输入下精确比较：热点判定、理论到达概率；
   - 新旧环境并排打印分布摘要：任务数、数据量、计算量、UE 位移、热点内外到达率；新旧两边使用同一组对照参数，并先统一换算为 bit、CPU cycles、米、秒再打印；
   - 打印三类数据量 `input_bits`、`edge_bits`、`return_bits` 的分布；
   - 手工小场景只看先后顺序和端点：FIFO 排队、依赖解锁、sink 回传、每次传输的跳路径（UE→接入 UAV→执行 UAV；sink UAV→最近 UAV→UE）、超时后的取消；
   - 同一 (seed, episode) 下用两个不同的随机策略各跑一次，打印两边前 `arrival_slots` 个时隙的 offer 序列和 UE 轨迹，应完全相同（不同策略 `done` 的时隙不同，收尾段不比较）。
2. 新环境用随机连续动作跑若干 episode，打印传输时间、队列长度、按时完成率、超时数，看量级是否合理。
3. 不比较通信时间、能耗、轨迹、完整 episode 结果；不跑基线、不跑训练。

### 0.4.13 留给方法层（Gate 7）处理的连带影响

1. 候选特征、12 维 task feature、属性聚类向量、forecast 中用到的 DAG 带宽字段随之删除（forecast 的输出数据量改用 `edge_bits`），相应特征维度在方法 adapter 阶段调整。
2. 能耗归一化（如 `CLEAN_NORM_PAIR_ENERGY_REF`）、候选能耗特征和 reward 按 κ·C·f² 的焦耳值重新标定；reward 与指标中的移动能耗改用新的飞行能耗（悬停也耗能）；环境只输出物理焦耳值。
3. graph 特征中的 `remaining_energy_ratio`（恒为 1.0）直接删除；位置塑形奖励只属于 N0/A1/C1 的默认 reward（B2/C2 不用），其覆盖半径移到该 reward 的配置，v2 环境不涉及；graph 中“每个子任务的输出量”特征改为该子任务全部出边 `edge_bits` 之和。
4. assignment、executor、graph、forecast、movement observation 改为按下标读取集合式 UAV/UE（UAV 编号 = 数组下标）。
5. v2 不需要 exogenous tape（见 §0.4.14）；旧 tape 与相关脚本保留，仅用于复现历史 B2/C2 结果，不为 v2 做兼容。
6. movement actor 改为连续输出之前，用 §0.4.7 第 3 条的映射过渡。
7. 方法层中以 `task_id` 索引的地方（assignment 候选、graph 节点、placement）改用 `(dag_id, node_id)`。
8. 一个 UE 可同时有多个活动 DAG（§0.4.15）：C2 质心 teacher 的“UE 有活动 DAG”改为从 DAG 记录推出，graph、observation 中按 UE 只取一个活动 DAG 的地方相应调整。
9. 服务能耗改为在承诺的时隙计入（§0.4.8；旧代码在任务完成时计入）：逐时隙 reward 中能耗出现的时隙随之提前，同一批工作的总能耗不变。若要保留“完成时扣能耗”，由 reward 层读 `TaskRuntime` 的能耗在完成时隙扣除，环境不变。
10. 能耗量级（2026-10-16 实测，暂定参数）：每时隙 5 架 UAV 推进能耗约 4000 J，其中约 98% 是与策略无关的悬停底数；计算约 0.4 J、传输约 0.8 J，推进能耗约为服务能耗的 3500 倍。reward 不得直接用总能耗：推进能耗应扣除悬停底数后单独归一化（或只作指标不进 reward），服务能耗（计算 + 传输）单独归一化，分别对应移动与卸载决策；不得为拉近量级而改动 κ 等物理参数。悬停占比大与时隙设置（5 s 时隙、0.5 s 飞行）直接相关，见 §0.4.6。
11. 覆盖（2026-10-17）：本时隙未被覆盖的 UE，其入口子任务不能分配（§0.4.3 第 4 条），assignment 候选要按环境给出的覆盖判断过滤。

### 0.4.14 随机流与外生过程（2026-10-13 决定）

1. 参照 MRDTS `RandomStreams`，由 (seed, episode) 派生四条独立的 `np.random.Generator`：场景（热点个数与中心、UE 人数分配、UAV/UE 初始位置、UE 初始速度与方向）、移动（UE 移动）、到达（到达抽样）、DAG 内容。
2. 到达窗口内每个 UE 每个时隙都抽一次到达，与它是否已有活动 DAG 无关；抽中即从 DAG 流生成一个 DAG，立即生效，不存在接收或拒绝（§0.4.15）。DAG `id` 按生成顺序递增。每条流一个 Generator，整个 episode 只用它，不按时隙或 UE 再拆。
3. 取消等待服务降速（`UE_SERVICE_WAITING_SPEED_SCALE`）。另设热点内减速（§0.4.2 第 5 条，2026-10-15）：只取决于 UE 自身位置，不破坏下一条的与策略无关。
4. 效果：UE 轨迹、到达和 DAG 内容与策略无关，同一 (seed, episode) 下所有策略面对相同的外部环境，v2 不需要 tape。该语义与历史正式公平评估（固定 tape：不降速、固定 offer）一致。
5. `SlotResult` 记录本时隙的 offer（DAG 编号、UE、到达时刻），供 offer 总数和按时完成率使用。
6. 方法层和随机策略使用自己的随机数生成器，不使用环境的随机流。

### 0.4.15 截止时间与 episode 结束（2026-10-13 决定）

> 师兄选择“加截止时间，实验加完成率指标”；用户选择不限制每个 UE 的活动 DAG 数（D'）。

1. 每个 offer 都立即成为活动 DAG，不判断 UE 是否空闲；同一 UE 可同时有多个活动 DAG，其子任务与其他 DAG 一样进入就绪列表。积压由截止时间限制。
2. 截止时间：所有 DAG 统一用相对截止 `relative_deadline_slots`（记 D），截止时刻 = 到达时刻 + D × 时隙长度，正好落在时隙边界。依赖只在时隙边界释放，L 层的 DAG 至少要 L 个时隙，所以 D 必须大于最大层数（读取配置时检查，见 §0.4.5 第 4 条）。
3. 成功：所有 sink 的结果都在截止时刻之前（含恰好等于）回到 UE，DAG 记为 COMPLETED。
4. 超时：到截止时刻仍未成功，DAG 记为 TIMED_OUT：
   - 尚未分配的子任务记为 CANCELLED，不再进入就绪列表；
   - 已分配的子任务不抢占：承诺的时间线不撤销，照常占用 UAV，能耗已在分配时计入（§0.4.8）；
   - 开始时刻不早于截止时刻的回传不再安排（§0.4.4 第 3 条）；已安排的回传照常进行，但 DAG 仍记为超时。
5. `end_slot` 内的结算顺序：推进时间线到时隙末 → 判定成功 → 判定超时。
6. episode 分两段：前 `arrival_slots` 个时隙产生到达；之后不再产生到达，只推进已有 DAG，直到所有 DAG 都是 COMPLETED 或 TIMED_OUT（最多再 D 个时隙），此时 `done=True`；episode 不等已承诺但未执行完的工作推进到 DONE（其能耗已计入）。
7. 主指标：按时完成率 = COMPLETED 的 DAG 数 ÷ offer 总数。

### 0.4.16 明确不做（2026-10-13）

1. 不设 `EntityRef` 之类的实体引用类型：UE、UAV 用整数下标，子任务用 `(dag_id, node_id)`。
2. 不写 `validate_dag`：DAG 只由本项目生成器产生，结构性质由 §0.4.12 的一次性脚本检查。
3. 不保留 `freeze_ue_mobility` 开关（旧 `env.py:36`）：UE 移动已与策略无关（§0.4.14），不需要冻结。
4. 快照不复制，`SlotResult` 不放 reward 和指标（§0.4.8）。

---

# 1. 背景与核心问题

## 1.1 用户当前处境

**[用户需求]** 用户认为当前 HyperUAV 环境代码较乱，希望借鉴师兄 MRDTS 项目的 env 架构，重新建立一套清晰、可理解、可扩展、便于测试的环境框架。

**[用户需求]** 用户目前对师兄项目和自己项目的代码结构都缺乏完整理解，不能仅凭经验判断一个重构方案是否合理，也无法独立识别：

- 方案是否过度设计；
- 是否存在不必要的防御逻辑；
- 是否引入哈希、SHA-256 或复杂版本校验；
- 是否会在重构过程中再次形成屎山代码；
- 哪些旧代码可以删除，哪些仍属于当前主线；
- “两个智能体不够”究竟是在说什么。

因此，后续流程不能要求用户通过逐行代码审核来保障质量。设计必须使用用户、师兄和 AI 都能检查的职责边界、数据流、阶段产物和停止条件。

## 1.2 用户希望实现的最终结果

**[用户需求]** 最终希望得到一套“清爽版本”的 HyperUAV 环境框架，至少具有以下性质：

1. 能清楚解释系统里有哪些实体、状态和事件；
2. 环境、算法、智能体、训练和实验管理职责分离；
3. 一个物理公式或状态事实只有一个权威实现；
4. 模块可以独立理解和测试；
5. 后续调整智能体数量或策略组织方式时，不需要重写物理环境；
6. 不永久保留大量 legacy/compat 分支；
7. 旧支线留下的无用代码最终从活动代码树中删除；
8. 用户能在过程中学习实体、配置、接口、依赖、状态机、仿真器和测试等基础知识。

## 1.3 本任务不是简单的目录整理

**[初步建议]** 本任务应被视为一次“领域模型与环境边界重建”，而不是：

- 把大文件机械拆成小文件；
- 把现有函数移动到与师兄相同的目录名；
- 复制 MRDTS 的 env 后逐个修改特例；
- 在现有 `Env` 外面再套一层包装器；
- 一边编码一边临时决定系统语义。

如果只改变目录，不先冻结系统事实、状态所有权和时间流程，旧耦合会原样进入新目录。

---

# 2. 用户明确提出的需求与约束

## 2.1 参考师兄结构，但不一比一复制

**[用户需求]** 师兄建议“先写实体，再写其他部分”。用户希望理解这句话在两个项目中的真实含义，然后决定 HyperUAV 是否以及如何采用类似架构。

**[初步建议]** “先写实体”不应解释为立刻创建 `UE.py`、`UAV.py` 等文件。更准确的顺序是：

1. 先确定系统事实、单位、时间语义和可控决策；
2. 再定义公共类型与交互契约；
3. 再实现实体及实体自身的不变量；
4. 然后实现通信、workload、runtime 和 simulator；
5. 最后接入方法、智能体和训练。

## 2.2 每一步都需要和师兄讨论

**[用户需求]** 每个重构阶段都需要先向师兄说明，师兄认为可行后才能进入下一步。

**[初步建议]** 每个阶段设置两个评审门：

```text
阶段设计说明
→ 用户与师兄审核
→ 用户明确批准
→ 实现该阶段
→ 测试与结果说明
→ 用户与师兄复核
→ 用户明确批准进入下一阶段
```

不建议按“每写几行代码就讨论一次”，而应按职责完整、能够单独审核的模块设置阶段。

## 2.3 智能体数量尚未确定

**[用户需求]** 师兄提出当前“两个智能体不够”，但具体含义尚未确认；师兄当前让用户优先修改环境。

**[用户需求]** 用户希望确认：

- 当前 env 是否已经混入智能体设计；
- 混入了多少；
- 是否可以暂时不决定最终智能体数量；
- 是否能先把环境改好，再讨论智能体结构。

## 2.4 防止再次形成屎山代码

**[用户需求]** 用户担心一开始批准的方案在实施过程中不断堆兼容分支、fallback、feature flag、诊断逻辑和临时抽象，最终形成新的屎山。

用户希望方案本身包含能够被非代码专家理解的质量约束，而不是要求用户相信 AI 的判断。

## 2.5 删除旧支线遗留代码

**[用户需求]** 用户希望利用本轮重构，最终删除以前旧支线留下的无用代码，而不是继续让 clean、legacy、兼容层和历史实验逻辑混在活动代码树中。

**[待确认]** “旧支线”可能包括不同类型的对象，后续必须分别处理：

- Git 分支名称；
- 当前分支中遗留的旧入口；
- 旧环境逻辑；
- 旧 MARL 算法实现；
- 已关闭研究路线的训练/诊断模块；
- 历史实验脚本；
- spec/report/结果索引；
- 服务器上的原始结果。

这些对象不能一刀切采用同一种删除策略。

## 2.6 禁止过度防御和自定义哈希校验

**[用户需求][禁止]** 后续新代码不得：

- 编写哈希或 SHA-256 校验；
- 为基本不可能出现的内部 case 反复增加保护逻辑；
- 为假想未来构造过度通用的框架；
- 用复杂 rubric 代替对任务目标的合理判断。

正式实验继续按 `AGENTS.md` 记录 Git 分支、HEAD、dirty 状态和脚本版本。这属于版本纪律，不意味着在运行时代码中增加自定义文件/模型 SHA-256 校验。

---

# 3. HyperUAV 当前代码结构：已验证事实

## 3.1 当前核心环境模块

**[已验证]** 当前 `environment/` 主要包括：

| 文件 | 当前主要职责 | 已观察到的问题 |
|---|---|---|
| `environment/env.py` | 环境生命周期、实体初始化、slot 准备、移动、分配提交、执行推进、reward、metrics、观测和 info | 协调器同时承担物理环境、RL 接口和诊断职责 |
| `environment/dag_tasks.py` | Task/DAG 数据、DAG 生成、到达、状态推进、ready 刷新、关键路径、特征和超边预计算 | 数据模型、workload、runtime 和表示特征混合 |
| `environment/assignment.py` | 分配缓冲、临时预留、合法候选、候选时间/能耗估计、神经网络输入特征 | 环境合法性与方法专属特征混合 |
| `environment/task_execution.py` | UAV 队列、任务时间线、计算、通信、回传和结算 | 执行事实与候选估计存在重复路径，需统一权威实现 |
| `environment/comm_model.py` | legacy 信道公式、A2A 范围、clean 距离衰减、传输时间 | 同一文件并存两套通信语义 |
| `environment/metrics.py` | reward component、episode metrics、info 汇总 | reward 与公共评估指标仍位于 environment 包 |
| `environment/graph_builder.py` | task graph/hypergraph snapshot 与特征 | 更接近方法表示层，而不是纯环境事实 |
| `environment/uavs.py` | UAV 状态、移动、旧请求处理、缓存和能量逻辑 | clean 当前实际使用与旧业务逻辑混合 |
| `environment/user_equipments.py` | UE 位置、移动、hotspot 到达概率和 service-waiting | 直接依赖全局 `config.py` |
| `environment/exogenous_tape.py` | 外生随机轨迹生成与回放 | 与现有实体具体内部字段强耦合 |
| `environment/forecast.py`、`reward_redesign.py` | 历史诊断/奖励重构相关能力 | 不应自然进入新环境内核 |

## 3.2 当前 `Env` 承担了过多职责

**[已验证]** `environment/env.py` 的 `Env` 当前同时负责：

- 创建 UAV 和 UE；
- 创建并重置 DAG manager、executor 和 metrics；
- 处理 UE 移动和 DAG 到达；
- 冻结 ready task；
- 解析 movement/offloading 动作字典；
- 应用 UAV 移动；
- 构建 assignment buffer；
- 推进 task executor；
- 计算 reward；
- 按 UAV 数量复制全局 reward；
- 生成环境 observation；
- 汇总训练、评估和诊断 `info`；
- 计算部分 movement shaping signal。

这说明当前问题不是“文件数量不够”，而是环境协调器没有保持最小职责。

## 3.3 当前并不是只有两个物理智能体

**[已验证]** 当前配置为：

- `NUM_UAVS = 5`
- `NUM_UES = 60`

**[已验证]** clean 主线设计中所谓“两个”是：

- movement actor；
- offloading actor。

其中：

- 所有 UAV 共享同一个 movement actor 参数，但每架 UAV 每个 slot 都产生一次移动动作；
- offloading actor 是任务中心的共享 scorer，为当前 ready task 依次选择 UAV；
- 当前整体还包含 task encoder/HGNN 与 centralized critic。

因此必须区分：

1. 物理实体数量；
2. 每个 slot 的动作数量；
3. 参数共享的 actor head 数量；
4. 逻辑 agent 数量；
5. 决策阶段数量。

师兄所说“两个智能体不够”目前不能从代码推断其真实含义。

## 3.4 当前 env 已经混入智能体/方法设计

**[已验证]** 当前耦合至少包括：

- `Env.step()` 直接接受 movement/offloading 联合动作格式；
- `_get_obs()` 按 UAV 生成 RL observation；
- `commit_and_advance()` 直接生成 reward list；
- `assignment.py` 生成 offloading actor 使用的归一化 candidate features；
- `graph_builder.py` 在 environment 下生成 HGNN/MLP 表示输入；
- `clean_slot_orchestrator.py` 和 `train_clean_mainline.py` 直接读取 `env.task_manager`、`env.executor`、`env.uavs`、service positions 等内部对象；
- reward、diagnostics、evaluation 数据通过一个大型 `info` 字典混合输出。

结论：最终智能体数量可以后置，但环境与方法的边界不能保持现状。

## 3.5 当前通信存在两套语义

**[已验证]** `environment/comm_model.py` 中至少并存：

1. legacy 自由空间距离平方衰减、SNR 和带宽模型；
2. clean 主线的二维距离因子：`1 / (1 + (d / 100)^2)`；
3. A2A 最大距离与固定控制开销；
4. clean 的 MB/Mbps 传输时间换算。

**[已验证]** clean assignment estimator 与 executor 都会计算：

- UE 到 UAV 的入口数据传输；
- 父任务所在 UAV 到目标 UAV 的中间结果传输；
- sink UAV 到 UE 的结果回传；
- 传输时间和发送能耗。

重构通信前必须先由用户和师兄确认哪一套公式、链路和单位才是目标系统模型。不能因为师兄的通信代码更整洁，就默认其物理公式适用于 HyperUAV。

## 3.6 当前旧入口已经被项目文档判定为非 clean 主线

**[已验证]** 当前 README 和 clean runbook 已明确把以下内容列为旧入口或禁止入口：

- 根目录 `main.py`；
- 根目录 `train.py`；
- `scripts/tune.py`；
- 旧 assignment MAPPO；
- legacy MAPPO；
- legacy HGNN scheduler；
- 旧实验脚本。

**[已验证]** 根目录旧入口仍通过 `marl_models/utils.py` 连接 MADDPG、MATD3、MASAC、旧 MAPPO、attention 版本和 random model。

这说明删除旧算法和旧入口在方向上是可行的，但仍需先核对当前主线和历史结果的引用关系。

## 3.7 当前确实存在自定义 SHA-256 和哈希逻辑

**[已验证]** 当前代码包含：

- `scripts/train_clean_mainline.py` 中的模型状态 SHA-256；
- 同文件中的 checkpoint 文件 SHA-256；
- 若干历史 evaluation 脚本中的文件 SHA-256；
- `scripts/offloading_policy_gate.py` 中基于 SHA-256 的 `random_hash` 策略。

**[初步建议]** 新环境与新主线不得继承这些做法。历史实验的旧行为在 Git 历史和报告中保留；是否修改仍在使用的评估脚本，需要作为单独决策，不在环境重构中静默改变实验协议。

## 3.8 当前测试结构不足以直接支撑大规模原地重构

**[已验证]** `test/` 下常规单元测试数量很少，环境验证大量依赖 `scripts/smoke_clean_*`。

这并不意味着要建立庞大测试体系，但意味着直接在旧 `environment/` 内大规模移动代码会失去可靠参照。并行构建新环境更容易做新旧行为对照。

---

# 4. 师兄 MRDTS 架构：已验证事实

## 4.1 目录与职责

**[已验证]** MRDTS 当前采用以下主要分层：

```text
env/
├── entities/        区域、用户、Master/Member UAV 等实体
├── settings/        类型化配置
├── types.py         公共标识和值类型
├── contracts.py     环境接受的请求和放置决策
├── workload/        DAG 数据和生成
├── communication/   信道模型与路由
├── runtime/         传输、计算队列、任务状态和事件推进
└── simulator.py     时隙与资源生命周期协调

methods/
├── contracts.py     排序、飞行、调度和 Trainer 接口
├── components/      可复用决策组件
├── solutions/       完整方法
└── learning/        共享学习支持

experiments/         运行、训练、配置、记录、指标和分析
```

## 4.2 最值得借鉴的原则

**[已验证]** MRDTS 明确规定：

- `env` 不导入 `methods`；
- 环境只维护物理事实和执行规则；
- RL observation、reward 和可见信息由具体方法负责；
- 基础标识集中定义，不在子模块重复；
- channel model、route、queue/runtime 分离；
- Simulator 协调生命周期，不负责选择策略；
- 方法通过公共 contract 向环境提交决策；
- experiments 负责正式运行、训练和结果管理。

这些边界适合 HyperUAV 参考。

## 4.3 不能直接复制的系统差异

> 2026-10-12 更新：其中“物理信道”一行已决定改用 MRDTS 公式，见 §0.4.3。

**[已验证]** 两个项目至少存在以下关键差异：

| 维度 | HyperUAV 当前 clean 主线 | MRDTS 当前实现 | 后果 |
|---|---|---|---|
| UAV 角色 | 同类 UAV | Master/Member UAV 与 BS | 实体和路由不能照搬 |
| DAG 生命周期 | DAG/任务可跨多个 slot 执行 | 当前调度 runtime 在单个截止窗口内结算，未完成任务失败 | runtime 语义不能照搬 |
| UE | 移动，hotspot 影响 DAG arrival | 当前文档描述的用户位置在回合内固定 | mobility 和场景流程不同 |
| 入口路由 | UE 到被选 UAV | Ground 先到 owner Member，再到执行节点 | RoutePlanner 不能照搬 |
| 中间结果 | 父 UAV 到子 UAV | 根据 Member/BS/Ground 规则选 route | 拓扑规则不同 |
| sink 返回 | sink UAV 返回 UE | 使用其既定 route 规则 | 必须重新定义 HyperUAV route |
| 物理信道 | 当前 clean 使用二维经验衰减 | 三维 LoS/NLoS 与 Shannon rate | 公式和参数不能默认复用 |
| 方法结构 | movement + task-centric offloading + encoder/critic | ordering + flight + scheduling 组件 | agent/决策分工不能照搬 |

结论：HyperUAV 可以采用 MRDTS 的分层原则，但必须建立自己的领域模型和生命周期。

---

# 5. 当前初步结论

## 5.1 最终智能体数量可以后置

**[初步建议]** 可以先不决定最终逻辑 agent 数量，优先改环境。

但环境必须先明确系统有哪些客观可控自由度，例如：

```text
MovementDecision:
    UAV ID -> 移动动作

PlacementDecision:
    Task ID -> 执行 UAV ID
```

这些 contract 不规定动作由一个 agent、多个 agent、共享参数 actor、启发式或人工规则产生。

新环境应当可以被以下任一方法驱动而不改变内部实现：

- random policy；
- scripted policy；
- heuristic；
- 中心化单智能体；
- 每 UAV 一个智能体；
- 参数共享多智能体；
- 将来重新设计的层级式或多阶段智能体。

## 5.2 “从最开始改”的真实起点是 Gate 0

**[初步建议]** 第一项工作不是创建实体文件，而是形成并审核“系统事实与生命周期说明”。

必须先回答：

- 有哪些实体；
- 各实体拥有什么状态；
- 谁拥有和修改这些状态；
- slot 开始和结束分别发生什么；
- DAG 如何产生、阻塞、释放和完成；
- task 如何从 pending 进入 ready、queued、running、returning、completed；
- 通信和计算是否并行；
- UAV 队列和信道队列如何工作；
- 移动前后使用哪一时刻的位置；
- 哪些结果是物理事实，哪些只是 reward 或 observation；
- 哪些现有行为是目标语义，哪些是历史遗留或 bug。

## 5.3 通信可以较多参考师兄，但不是复制

> 2026-10-12 更新：已决定通信公式全部采用 MRDTS，本节中“公式不能默认复用”的表述作废；下方“独立 channel queue/runtime”本轮不建；“必须重新确认”列表中的维度、公式、带宽、可达规则、路线、多父传输、并发与排队、跨 slot、sink 返回均已由 §0.4.3–§0.4.4 回答，仅功率、噪声、带宽等数值按 §0.4.6 最后再定。

**[初步建议]** 可以参考或重用的主要是结构：

- 类型化 `ChannelConfig`；
- `LinkType`；
- `LinkMetrics`；
- 由位置和参数计算 rate/delay/energy 的纯 `ChannelModel`；
- 独立 `RoutePlanner`；
- 独立 channel queue/runtime；
- estimator 和真实 executor 共用同一通信实现。

必须重新确认并按 HyperUAV 实现的内容包括：

- 二维或三维位置；
- LoS/NLoS 公式；
- transmit power、noise、bandwidth 和单位；
- 带宽共享方式；
- coverage/A2A 可达规则；
- UE→UAV、UAV→UAV、UAV→UE 的路线；
- 多父任务传输聚合；
- 通信并发和排队；
- 通信是否跨 slot 保留进度；
- sink result 返回语义。

## 5.4 旧代码最终应删除，但不应在重写前全部删除

**[初步建议]** 采用“并行新建 → 对照 → 切换 → 删除旧代码”的方式。

原因：

- 旧环境目前仍是行为参照；
- 当前常规单元测试覆盖不足；
- 训练/评估/诊断脚本大量直接访问 env 内部对象；
- 立刻删除会同时失去参照和暴露大量无关迁移问题；
- 永久保留 `legacy/` 又会继续制造混乱。

Git 历史已经提供恢复能力，不需要额外创建 legacy 复制目录、压缩备份或自定义哈希清单。

---

# 6. 初步范围

## 6.1 第一阶段在范围内

**[初步建议]** 第一阶段仅聚焦环境架构基础：

- 系统事实整理；
- 时间、位置、数据量、计算量、能量单位；
- 公共类型；
- 决策和结果 contract；
- 实体；
- 配置注入；
- 通信；
- DAG/workload；
- task/runtime；
- simulator 生命周期；
- 最小 scripted/random driver；
- 新旧环境行为对照；
- env 与 method 的依赖边界。

## 6.2 第一阶段不在范围内

> 2026-10-12 更新：通信公式、计算能耗公式与 UAV 连续飞行动作按 §0.4 替换，不属于本节所禁止的“顺便修改物理参数”；具体数值仍按 §0.4.6 最后再定。下列“不做”事项中涉及 state、reward、特征的条目，以 §0.4.13 已批准的连带变化为例外；除此之外不增加其他方法重设计。

**[禁止，除非用户另行批准]** 第一阶段不做：

- 决定最终智能体数量；
- 重新设计 movement/offloading actor；
- 修改 PPO/MAPPO；
- 修改 HGNN/MLP 最终对照方案；
- 修改 183 维 state；
- reward redesign；
- 新的 credit assignment；
- 新的训练实验；
- 重新开启已被 `AGENTS.md` 禁止的旧实验；
- 为了重构顺便修改物理参数；
- 为了测试方便改变 DAG/task 生命周期；
- 清理用户当前未提交文件。

---

# 7. 方案选择

## 7.1 方案 A：并行建立 HyperUAV 新环境（推荐）

建议暂用路径名：`environment_v2/`。最终名称在切换时再决定，不要求永久带 `v2`。

优点：

- 旧环境保持可运行和可对照；
- 可以从公共类型、实体和 contract 开始；
- 每个阶段可独立让师兄审核；
- 不需要在旧 `Env` 内继续堆兼容分支；
- 切换成功后可以一次性删除旧实现并将新环境改回简洁正式名称。

风险：

- 切换前会短期存在两套环境；
- 必须严格限制过渡期长度；
- 训练/评估 adapter 需要一次明确迁移。

## 7.2 方案 B：在现有 `environment/` 内渐进拆分

优点：

- 表面改动路径少；
- 不需要最终大规模改 import。

风险：

- clean/legacy、reward/metrics、graph/env 和 trainer/env 耦合会持续干扰；
- 需要大量临时兼容路径；
- 很难判断每次修改是否改变行为；
- 更容易形成新的 feature flag 和 fallback。

不推荐。

## 7.3 方案 C：复制 MRDTS env 后改成 HyperUAV

优点：

- 初始目录看起来整洁；
- 可以快速获得现成类和文件。

风险：

- 两个项目的实体、路由、slot、DAG 生命周期和执行节点不同；
- 后续需要不断添加 HyperUAV 特例；
- 容易形成“师兄架构外壳 + HyperUAV 条件分支”；
- 用户难以真正理解自己的系统模型。

不推荐。

---

# 8. 初步目标架构（未批准）

> 2026-10-12 更新：本轮不建 `channel_queue.py`、`server_queue.py` 和事件推进模块（时间线保持现状，见 §0.4.4）；实体与配置按 §0.4.2、§0.4.5。

以下只表示职责边界，不是已经冻结的文件清单：

```text
environment_v2/
├── types.py
│   └── 子任务标识 (dag_id, node_id)、TaskStatus、DAGStatus
├── contracts.py
│   └── SlotResult（动作格式见 §0.4.8）
├── settings/
│   └── 类型化场景、UAV、UE、DAG、通信和 runtime 配置
├── entities/
│   ├── ue.py
│   ├── uav.py
│   ├── dag.py
│   └── region.py
├── communication/
│   ├── channel_model.py
│   └── routing.py
├── workload/
│   └── DAG 生成与到达（随机流见 §0.4.14）
├── runtime/
│   └── executor.py（单服务台 FIFO 计算时间线、子信道占用、plan()、任务与 DAG 状态推进、超时结算）
└── simulator.py
    └── 只协调生命周期，不选择策略、不计算 RL reward

methods/ 或后续适配层
├── observation/
├── reward/
├── graph/
├── movement/
├── offloading/
└── learning/

experiments/
├── train
├── evaluate
├── metrics
└── artifacts
```

## 8.1 依赖方向

**[初步建议]** 依赖保持单向：

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

底层不得反向导入 actor、HGNN、trainer 或实验脚本。

## 8.2 状态所有权原则

**[初步建议]** 每类状态只能有一个所有者：

- UAV 实体只拥有位置，物理参数在 `UAVConfig`；
- runtime（executor）拥有任务状态（`TaskStatus`）、DAG 状态（`DAGStatus`）、UAV 队列、UAV 空闲时刻、子信道空闲时刻和任务执行记录；
- simulator 只协调，不复制成为第二状态源；
- method/agent 只读取 snapshot 并提交 decision，不直接修改内部状态。

---

# 9. 建议的阶段与师兄评审门

## Gate 0：系统事实与生命周期冻结

### 目标

在不写业务代码的前提下，明确 HyperUAV 的真实世界模型。

### 产物

- 实体清单；
- 状态所有权表；
- 单位表；
- task/DAG 状态机；
- 一个 slot 的时间流程；
- 通信链路和路由表；
- 计算/通信并发规则；
- 决策自由度；
- 当前代码行为与目标行为差异表；
- 需要师兄裁决的问题清单；
- 明确的“不做事项”。

### 停止条件

用户和师兄没有明确批准前，不进入 Gate 1，不创建新实体和 runtime 代码。（2026-10-13：用户确认 §0.4，师兄同意先实现后评审，进入 Gate 1–2。）

## Gate 1：公共类型、单位、配置与 contract

### 目标

定义环境各层共同使用的最小语言，但不实现完整仿真。

### 产物

- ID/value types；
- 类型化 settings；
- movement/placement contract；
- slot snapshot/result contract；
- 决策合法性边界说明；
- 不写 contract tests（2026-10-13）：用一条命令读入全部配置并打印即可；行为检查见 §0.4.12。

## Gate 2：实体

### 目标

实现 UE、UAV、DAG、Region 的数据。子任务是 DAG 内的节点，不单独建类；不写局部不变量检查（§0.4.9）。

### 不做

- reward；
- observation；
- actor；
- 全局 slot 流程；
- 通信队列；
- 训练接口。

## Gate 3：通信

### 目标

实现唯一的 channel model 和 HyperUAV route planner（只读计算；子信道占用与排队在 Gate 4，见 §0.4.3 第 7 条）。

### 关键要求

- estimator 与真实执行共用同一计算入口；
- 明确单位；
- 不同时保留 legacy/clean 两套正式公式；
- 不复制 MRDTS 的 Master/Member/BS 路由；
- 不加入与当前系统无关的复杂链路类型。

## Gate 4：workload 与 runtime

### 目标

实现 DAG arrival、依赖释放、task 状态机、单服务台 FIFO 计算时间线、子信道占用与排队、跨 slot 执行（见 §0.4.4）和截止时间结算（见 §0.4.15）。

### 关键要求

- 任务状态只有一个权威 runtime；
- 不在 DAG 数据类中混入 neural feature；
- 不在 runtime 中选择策略；
- 保持已批准的跨 slot 语义。

## Gate 5：Simulator

### 目标

仅协调以下流程（对外接口为 `reset` / `begin_slot` / `end_slot`，见 §0.4.8）：

```text
reset
→ prepare slot
→ freeze snapshot
→ accept movement decision
→ accept placement decision
→ advance communication/computation
→ settle physical results
→ return next snapshot/result
```

### 不做

- 计算 PPO reward；
- 生成 HGNN 图；
- 构造特定网络 observation；
- 复制每 agent reward；
- 加载 checkpoint；
- 记录训练 loss。

## Gate 6：最小 driver 与行为对照

> 2026-10-12 更新：本 Gate 的验收内容以 §0.4.12 为准，下方“最小对照内容”中的逐项对照作废。

### 目标

使用 scripted/random/heuristic driver 运行新环境，不接正式 RL。

### 最小对照内容

- 相同固定输入下的 DAG arrival；
- ready task；
- assignment legality；
- task 起止时间；
- UAV queue；
- 通信与计算能耗；
- DAG 完成状态；
- slot 边界。

不追求穷举所有异常情况，也不使用文件或模型哈希校验。

## Gate 7：方法 adapter 与正式入口切换

> 2026-10-12 更新：本 Gate 处理 §0.4.13 列出的连带影响；离散 movement actor 用 §0.4.7 第 3 条的映射过渡。下文“暂时保持算法语义”指：除 §0.4.13 已批准的变化外，不增加其他方法重设计。方法层中的 `task_id` 改用 `(dag_id, node_id)`。

### 目标

- 把 graph/observation/reward 放在方法层；
- 让当前必要的训练/评估代码通过 adapter 使用新 Simulator；
- 确认环境不依赖最终 agent 数量；
- 暂时保持算法语义，不同时重做 agent 架构。

## Gate 8：旧代码删除

### 目标

正式入口切换并通过审核后，删除活动代码树中的旧实现和失效路径。

### 原则

- 不建立永久 `legacy/`；
- Git 历史负责恢复；
- 不制作 SHA-256 清单；
- 删除前只检查实际引用和研究证据用途；
- 不删除用户未提交文件；
- 删除动作单独形成可审核 diff；
- 用户明确批准后执行。

## Gate 9：智能体架构设计

环境稳定后，再与师兄讨论：

- 最终逻辑 agent 数量；
- 是否按 UAV、区域、task 或决策阶段拆分；
- ordering 是否需要独立决策模块；
- movement/offloading 是否继续作为两个 actor head；
- 参数共享与集中训练/分散执行方式；
- observation/reward/credit 如何匹配新的 agent 定义。

Gate 9 属于算法设计，必须由用户批准，不能在环境重构中顺带决定。

---

# 10. 防止过度防御与屎山的强制约束

## 10.1 不使用自定义哈希校验

**[禁止]** 新环境、新 adapter 和新测试中不实现：

- 文件 SHA-256；
- 模型参数 SHA-256；
- checkpoint 指纹；
- 配置内容哈希；
- 用哈希代替正常 RNG 的策略。

正式实验的 Git HEAD/dirty 状态记录继续遵守项目版本纪律。

## 10.2 只保留最小必要校验

> 2026-10-12 更新：v2 中校验只在两个入口——读取配置、接收算法动作；核心不变量由结构保证，不做内部校验（见 §0.4.9）。下列第 2、3 条不适用于 v2。

校验只允许用于：

1. 外部输入是否合法；
2. 会破坏物理/研究正确性的核心不变量；
3. 代码中已经真实发生过、且修复成本明显低于后果的错误。

不为基本不可能出现的内部情况连续增加 fallback。

## 10.3 非法内部状态由结构消除（2026-10-12 改写）

不写内部断言。互斥状态、资源重复持有、生命周期乱序分别由单一 `status` 字段、单一 runtime 持有者、`begin_slot` / `end_slot` 接口在结构上消除；placement 指向不存在的实体属于算法动作入口校验。详见 §0.4.8、§0.4.9。

## 10.4 不提前抽象假想需求

只有存在明确的第二种真实实现或清楚的边界需要时，才增加 Protocol/ABC/Factory。

不为了“以后也许支持”提前设计：

- 任意数量环境插件；
- 通用事件总线；
- 多层注册表；
- 复杂依赖注入容器；
- 任意拓扑路由框架；
- 与当前研究无关的实体类型。

## 10.5 一个事实只有一个实现

必须避免：

- estimator 一套通信公式、executor 另一套；
- task 和 DAG manager 各保存一份状态；
- config 和实体各保存一份参数；
- observation 为方便训练偷偷修改环境状态；
- Simulator 与 runtime 各维护一套时间。

## 10.6 不永久保留兼容层

过渡期最多存在一个薄 adapter。正式切换后删除 adapter 或将其变成正式方法接口，不允许继续出现：

```text
legacy → compat → v2 → new wrapper
```

## 10.7 非代码专家也能检查的五个问题

每个阶段交付时，用户只需要检查：

1. 这个模块能否用一句话解释职责？
2. 同一个公式或状态是否在两个地方重复？
3. 是否新增了大量 `legacy`、`compat`、fallback 或 feature flag？
4. 是否出现哈希、SHA-256、文件指纹或不必要校验？
5. 删除该模块时，是否只影响一种明确功能？

任何问题解释不清，就不进入下一阶段。

---

# 11. 最小充分的验证方案

## 11.1 需要的验证

**[初步建议]** 每个模块只要求与风险相称的验证：

- 正常主路径；
- 一个真实有意义的边界情况；
- 关键状态转换；
- 必要的集成生命周期；
- 不变语义的定向核对与量级检查（见 §0.4.12）；
- env 不反向依赖 methods/marl 的结构检查。

## 11.2 明确不做的验证

**[禁止]** 不做：

- 自定义哈希一致性检查；
- 针对理论上任意损坏内存/文件的防御；
- 大量重复类型校验；
- 对每个私有函数做机械化单测；
- 没有现实来源的异常组合穷举；
- 为了追求测试覆盖率数字而制造无意义测试。

## 11.3 行为等价的边界

> 2026-10-12 更新：已由 §0.4.12 取代，不再要求新旧行为等价。

**[待确认]** 新旧环境对照不能盲目要求所有旧行为完全一致，因为旧行为中可能包含：

- 历史遗留逻辑；
- 已知不一致参数；
- clean 与 legacy 混合；
- 本轮本来就准备修正的通信/生命周期问题。

因此 Gate 0 必须把旧行为分为：

- 应保留的目标语义；
- 确认要修正的问题；
- 与新主线无关、直接删除的遗留行为。

只有“应保留语义”进入等价验证。

---

# 12. 旧代码删除初步分类

## 12.1 高概率删除候选

以下是候选，不是当前删除授权：

- 根目录旧 `main.py`、`train.py`、`test.py`；
- 旧 `marl_models` 下 MADDPG、MATD3、MASAC 和 attention 变体；
- 旧 MAPPO 入口与旧 buffer/helper；
- UAV 中未被 clean 主线使用的内容缓存、旧请求、能量请求和 GDSF 逻辑；
- legacy observation/action 配置；
- 被新环境完整替代的旧 `environment/`；
- 已关闭研究路线在正式主线中的运行时分支；
- 自定义 checkpoint/module/file SHA-256；
- 不再需要的 `random_hash` 评估路径；
- 与最终论文/当前评估无关的旧启动脚本。

## 12.2 切换前需要逐项核对的内容

- `scripts/train_clean_mainline.py` 当前直接导入的诊断/credit 模块；
- `scripts/eval_clean_mainline.py` 对训练脚本内部函数的反向依赖；
- 当前 graph builder、encoder、actor、critic 和 trainer；
- 当前正式 evaluation 和结果导出脚本；
- exogenous tape 与严格 CRN 诊断基础设施；
- 论文最终仍需要复核的实验脚本。

## 12.3 原则上保留的历史证据

- `AGENTS.md`；
- Living Research Charter；
- 已冻结 spec/report；
- 支撑论文结论的结果说明；
- 原始 JSON/log 的路径记录；
- 用户当前未提交文档和产物。

历史代码可以从 Git 恢复，不代表所有历史报告和研究决策都应从仓库删除。

---

# 13. 用户学习目标

**[用户需求]** 本轮重构不仅要交付代码，也希望帮助用户补足基础知识。

**[初步建议]** 每个 Gate 的说明应包含一个简短“为什么这样设计”部分，解释：

- 实体与 runtime 的区别；
- 数据类与服务类的区别；
- 状态所有权；
- 依赖方向；
- contract/interface 的用途；
- 配置注入为什么优于全局 config；
- simulator 为什么不应包含 RL reward；
- 单元测试、集成测试和行为对照分别解决什么问题；
- 什么是参数共享，以及为什么 actor head 数量不等于 agent 数量。

说明应围绕实际 HyperUAV 代码，不写成脱离项目的大段通用教程。

---

# 14. 每个阶段给师兄的评审材料模板

每个 Gate 的评审材料控制在能够一次读完的范围，至少包含：

1. 本阶段要解决的问题；
2. 当前代码的对应问题；
3. 借鉴 MRDTS 的哪条原则；
4. 哪些地方因 HyperUAV 不同而没有照搬；
5. 模块职责和数据流；
6. 明确不做什么；
7. 关键接口；
8. 最小测试或对照结果；
9. 已知限制；
10. 需要师兄裁决的问题；
11. 是否建议进入下一 Gate。

师兄的意见属于重要输入，但算法改动和阶段推进仍需用户明确批准。

---

# 15. 当前开放问题

> 2026-10-12 更新：15.1–15.3 中大部分问题已在 §0.4 决定；仍未决定的见 §0.4.4 第 3 条与 §0.4.6。

以下问题尚未解决，不能由 AI 自动假设：

## 15.1 系统模型

- HyperUAV 的最终实体是否只有 UE、同类 UAV、DAG 和 Task？（已决定：UE、UAV、DAG、Region；子任务是 DAG 内的节点，见 §0.4）
- 是否存在 MBS/BS、云、边缘服务器或需要保留的其他节点？（已决定：没有，见 §0.4）
- UAV 是否同构？未来异构性是否属于环境基础还是后续 scenario？（已决定：本轮同构，见 §0.4）
- task 是否允许跨 slot 继续执行？（已决定：允许，沿用现状，见 §0.4）
- communication 是否跨 slot 保留进度？（已决定：沿用现有解析时间线，见 §0.4）
- communication 和 computation 是否并行？（2026-10-17 改：并行，传输占子信道、计算占 CPU，见 §0.4.3 第 5 条）
- 一个 UAV 是否单核串行，还是支持并行核心？（已决定：本轮单核，见 §0.4）
- 多个传输是否共享带宽或进入链路队列？（2026-10-17 决定：OFDMA，每架 UAV K 个子信道、收发共用，满了排队，见 §0.4.3 第 3 条）

## 15.2 通信

- 采用当前 clean 二维经验衰减，还是师兄的三维 LoS/NLoS 模型，或另一个论文依据模型？（已决定：师兄的模型，见 §0.4）
- entry task 是否可以直接传到任意 UAV？（已决定：可以，经离 UE 最近的 UAV 中转，见 §0.4.3）
- A2A 是否有硬距离限制？（已决定：没有，见 §0.4）
- sink result 是否直接返回 UE？（已决定：经回传时离 UE 最近的 UAV 中转回 UE，见 §0.4.3）
- 多父任务结果如何并行或串行传输？（已决定：沿用现有串行累加，见 §0.4）
- candidate estimator 是否需要显式估计通信排队？（已决定：不需要，见 §0.4）

## 15.3 智能体

- 师兄所说“两个智能体不够”是指 actor head、逻辑 agent、决策阶段还是 UAV 实体数量？（已决定：指当前两个 actor，算法层后置，见 §0.4）
- 是否需要独立 ordering 决策模块？
- movement 与 offloading 是否由同一逻辑 agent 联合决策？
- 是否每架 UAV 都应成为独立 agent？

这些问题允许延后到环境稳定后回答，但 Gate 0 需要确认哪些属于环境可控自由度。

## 15.4 删除范围

- 已关闭实验路线的脚本是否全部从活动树删除，只保留 spec/report？
- 当前论文复核还需要哪些历史诊断脚本？
- 旧 MARL baseline 是否还有论文对比用途？
- `plotting/`、`analysis/` 和结果导出工具哪些仍属于最终交付？

---

# 16. 初步成功标准

环境框架重构最终应满足：

1. 用户能够解释一个 slot 的完整数据流；
2. 每个模块可以用一句话说明职责；
3. env 不导入 actor、HGNN、trainer 或 experiment；
4. reward、observation 和 graph representation 不属于物理环境内核；
5. 最终 agent 数量变化不要求修改 communication/runtime/simulator；
6. 通信时间、能耗和路由只有一个权威实现；
7. task/DAG 状态只有一个权威 runtime；
8. 配置按领域组织，不继续无限堆在单个全局 `config.py`；
9. 未改变的语义通过 §0.4.12 的定向核对，新环境通过量级检查；
10. 活动代码树不再保留永久 legacy/compat 层；
11. 旧入口、旧算法和失效分支按批准清单删除；
12. 新代码没有自定义哈希/SHA-256 校验；
13. 没有针对基本不可能 case 的重复防御逻辑；
14. 每个阶段都经过用户与师兄评审；
15. 未经用户批准不改变算法和研究阶段。

---

# 17. 推荐但尚未批准的总路线

```text
先审核本文
→ 修订需求
→ Gate 0：系统事实与生命周期
→ 师兄审核 + 用户批准
→ Gate 1：types/settings/contracts
→ 双重审核
→ Gate 2：entities
→ 双重审核
→ Gate 3：communication
→ 双重审核
→ Gate 4：workload/runtime
→ 双重审核
→ Gate 5：simulator
→ 双重审核
→ Gate 6：最小 driver 与行为对照
→ 双重审核
→ Gate 7：adapter 与入口切换
→ 双重审核
→ Gate 8：删除旧代码
→ 双重审核
→ Gate 9：重新设计智能体
```

---

# 18. 用户审核清单

用户审核本文时，可以逐项标注：

- `同意`
- `不同意`
- `需要师兄回答`
- `需要代码进一步核查`
- `暂时不做`

建议重点审核：

1. 是否准确表达了“先改环境、智能体后置”；
2. 是否同意先做 Gate 0，而不是直接写实体；
3. 是否同意采用并行新环境方案；
4. 是否同意最终不保留永久 legacy 目录；
5. 是否同意切换成功后再统一删除旧代码；
6. 是否同意新代码完全禁止自定义哈希/SHA-256；
7. 是否同意只做最小必要验证；
8. 是否有遗漏的系统实体或业务流程；
9. 哪些开放问题必须先问师兄；
10. 哪些内容表述过度、复杂或偏离用户真实目标。

---

# 19. 对后续 AI 的明确指令

1. 先读取仓库根 `AGENTS.md` 和 Living Research Charter；
2. 把本文视为待审核草案，不是批准的 spec；
3. 不得仅依据本文中的旧文档描述建立定量断言；
4. 需要使用旧行为作为前提时，必须读代码或日志验证；
5. 在用户明确批准本文或其修订版之前，不实现新环境；
6. 在用户明确批准删除清单之前，不删除旧代码；
7. 不清理当前工作树中的用户修改和未跟踪文件；
8. 不编写自定义哈希或 SHA-256 校验；
9. 不对基本不可能出现的 case 增加重复防御；
10. 智能体数量未定时，使用中立 decision contract，不替用户做算法决定；
11. 每个 Gate 只解决本 Gate 的职责，不顺带重构其他层；
12. 每个阶段结束后等待用户和师兄审核，不自动推进。

---

# 20. 当前状态

截至本文创建时：

- 只完成了需求梳理与只读源码核查；
- 没有创建 `environment_v2/`；
- 没有修改环境、算法、配置或训练代码；
- 没有删除任何旧文件；
- 没有启动训练或评估；
- 没有获得用户对初步方案的最终批准；
- 2026-10-12：用户与师兄讨论后确认的决定已写入 §0.4；仍未创建 `environment_v2/`，未修改任何代码；
- 2026-10-12（第二次）：根据 Codex 自查补充 §0.4.8–§0.4.13，更正 3 处事实，并同步修改 §8、§9、§10.3、§11、§15、§16；
- 2026-10-12（第三次）：根据 Codex 第二轮复核修改 11 条（未采纳“阶段对象”建议）；保留飞行能耗，采用标准旋翼推进功率公式，不沿用 MRDTS 的 `u1–u5`；
- 2026-10-13：子任务改为 DAG 内节点、数据量放在边上；时隙内顺序与飞行时长；UE 移动规则；中转路由；随机流拆分（§0.4.14）；注释风格；剩余能量与覆盖半径的处理更正；
- 2026-10-17：通信改为 OFDMA（每架 UAV K 个子信道、收发共用、满了排队）、噪声按功率谱密度、显式接收功率与上传覆盖、计算与传输分开两条时间线（§0.4.3、§0.4.4）；
- 下一步仅应是用户审核本文并提出修改意见。
