# MLP MAPPO 实现与验收结果（2026-11-06）

完成方法层实现、目录改名、两次本地更新、奖励核对与固定评估集基线。未 commit；output 未暂存。

用户随后要求实验改在服务器执行。10.12.54.24 上的两次更新、奖励核对及评估集基线已正常完成，总耗时 201 秒；日志和两个 checkpoint 已取回。以下保留先前本地记录，文末追加服务器完整验收数据，以服务器结果为本次最终验收依据。启动与完成信息见 output/mappo-check/server_job.json。本地未再运行实验。

**版本与范围**

- 仓库 HEAD：`c86b21ef819989e01b95ac0f256508fffd39b1da`，分支 `param-realign-20260914`。
- 官方仓库：`D:/CodeFile/on-policy`，commit `de66d7a4b23fac2513f56f96f73b3f5cb96695ac`。
- 执行器：`output/mappo-env/Scripts/python.exe`；Python 3.12.7、NumPy 1.26.4、PyTorch 2.15.0+cpu。
- 诊断与训练脚本版本以 HEAD + dirty/new 标识；未生成文件哈希。
- 本次没有修改 environment_v2、旧环境、旧模型和既有 config 文件。既有未提交的 channel/user 配置、需求文档和脚本改动保留。
- 运行参数：p=0.24，20 MHz，200 时隙，截止期 1 s，期望每时隙到达数 19.5。
- 下列状态在生成本报告和 result.json 前采集：

```text
 M config/channel.toml
 M config/user.toml
 M docs/superpowers/specs/2026-10-10-hyperuav-env-rebuild-requirements-draft.md
RM methods/baselines/policies.py -> methods_v2/baselines/policies.py
R  methods/ers.py -> methods_v2/ers.py
 M scripts/check_environment.py
 M scripts/run_baselines.py
?? config/methods/
?? methods_v2/learning/
?? methods_v2/mappo/
?? output/
?? scripts/sweep_baselines.py
?? scripts/train_mappo.py
```

**文件清单**

| 文件 | 当前行数 | 本次职责/改动 |
|---|---:|---|
| `methods_v2/ers.py` | 84 | 目录改名，文件内容不变；提供 ERS 排名和参考时长。 |
| `methods_v2/baselines/policies.py` | 111 | 目录改名，仅改 1 行 import；四个基线和随机飞行。 |
| `scripts/run_baselines.py` | 108 | 仅改 2 行 import，保留用户已有覆盖分组统计。 |
| `scripts/sweep_baselines.py` | 118 | 仅改 1 行 import，保留用户已有扫参实现。 |
| `scripts/check_environment.py` | 157 | 仅改 2 行 import，检查逻辑不变。 |
| `methods_v2/learning/valuenorm.py` | 45 | 官方默认分支的 ValueNorm 均值、方差和归一化。 |
| `methods_v2/learning/distributions.py` | 68 | 带候选 mask 的离散分布和独立随机流的高斯分布。 |
| `methods_v2/learning/networks.py` | 57 | 输入 LayerNorm、两层 MLP 和集中式价值头。 |
| `methods_v2/learning/buffer.py` | 55 | 完整 episode 缓存、终止处不自举的 GAE 和时隙批采样。 |
| `methods_v2/learning/mappo.py` | 68 | 官方裁剪 PPO、归一化价值损失、Adam 更新及诊断指标。 |
| `methods_v2/learning/LICENSE` | 21 | 保留官方 MIT 许可。 |
| `methods_v2/mappo/features.py` | 134 | 统一缩放，构造 32 维局部观测、16 维候选特征、180 维状态。 |
| `methods_v2/mappo/policy.py` | 76 | 共享飞行头和候选打分器，联合概率、确定性动作及 ERS 排序。 |
| `methods_v2/mappo/reward.py` | 14 | 唯一奖励实现；非法分配报错，统计全部 DAG 实际时延。 |
| `config/methods/mappo.toml` | 47 | 官方默认超参、本项目采样评估安排及期望到达数。 |
| `scripts/train_mappo.py` | 235 | 顺序采样、训练、评估、日志、checkpoint、奖励核对及基线评估。 |

新代码 10 个文件，另带 1 个 MIT LICENSE；原有 4 个调用文件共替换 6 行 import。以上行数含空行。

**与官方实现的差异及理由**

1. 仅保留 CPU、前馈 MLP、GAE、ValueNorm、Huber、价值裁剪和梯度裁剪分支，去掉 RNN、PopArt、active_masks、并行环境及学习率衰减；按本次需求裁剪。
2. actor 采用一个飞行头和逐子任务/候选共享的标量打分头；固定 6 格、可变任务数、按 UAV 合并 log probability，满足本环境动作定义。
3. 联合熵直接相加。官方混合分支为连续熵 / 2 + 离散熵 / 0.98；用户已明确选择需求文档公式。
4. critic 每时隙只存一个团队价值；GAE 每个完整 episode 单独计算。32 个并行环境改为顺序采样 32 个完整 episode，终止处不自举。
5. buffer 按时隙打乱，保留五个 actor 样本及一个 critic 样本；紧凑任务列表避免 padding/active_masks。默认 1 个 minibatch 时使用同一整批样本。
6. torch 与 NumPy 使用各自独立生成器；Gaussian/Categorical 显式接收生成器，初始化在 fork_rng 中进行。高斯 log_std 直接存为 Parameter，替代官方 AddBias 辅助类，数学形式相同。
7. 增加近似 KL（mean[(ratio-1)-log_ratio]）和裁剪比例日志，不添加 KL 早停。超参改从 TOML 读取，项目专用的特征、ERS、奖励和环境动作放在 mappo 目录。
8. ValueNorm 的 beta、去偏与方差下限，MLP 的输入 LayerNorm、ReLU、正交初始化，GAE、裁剪价值损失与 Adam 超参均保留官方指定分支。官方 MIT 许可一并保留。

**用户确认的需求补充**

- 奖励保持负时延；核对时在奖励还原式前加负号。
- 所有缩放只在 features.py 顶部定义：个数/主频/时长/位置分母从 settings 取，单任务 cycles 和 bits 用 1.5e8 与 1e6；三类总 cycles 分别按 UE 截止期算力、单 UAV 时隙算力和全体 UAV 时隙算力缩放。
- 熵直接相加。不存在尚待用户裁决的算法定义问题。

**改名前完整输出**

```text
arrival_slots=200; std=population (ddof=0)
seed episode policy on_time_rate covered_share covered_on_time_rate uncovered_on_time_rate timed_out compute_energy_j tx_energy_j return_energy_j flight_energy_j invalid_placements
0 0 random 0.737111 0.613050 0.780552 0.668286 979.000000 3980.011429 323.032830 119.262521 135051.852875 0.000000
0 0 local 0.662460 0.613050 0.658782 0.668286 1257.000000 0.000000 0.000000 0.000000 135051.852875 0.000000
check 0 0 local: offers=True uav_positions=True
0 0 greedy 0.862513 0.613050 0.985107 0.668286 512.000000 5132.020452 57.832571 99.603984 135051.852875 0.000000
check 0 0 greedy: offers=True uav_positions=True
0 0 heft 0.867078 0.613050 0.992554 0.668286 495.000000 4088.946796 167.194508 62.841431 135051.852875 0.000000
check 0 0 heft: offers=True uav_positions=True
random mean 0.737111 0.613050 0.780552 0.668286 979.000000 3980.011429 323.032830 119.262521 135051.852875 0.000000
random std 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
local mean 0.662460 0.613050 0.658782 0.668286 1257.000000 0.000000 0.000000 0.000000 135051.852875 0.000000
local std 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
greedy mean 0.862513 0.613050 0.985107 0.668286 512.000000 5132.020452 57.832571 99.603984 135051.852875 0.000000
greedy std 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
heft mean 0.867078 0.613050 0.992554 0.668286 495.000000 4088.946796 167.194508 62.841431 135051.852875 0.000000
heft std 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
invalid_placements_total=0; offers_and_uav_positions_equal=True
```

**改名后完整输出**

```text
arrival_slots=200; std=population (ddof=0)
seed episode policy on_time_rate covered_share covered_on_time_rate uncovered_on_time_rate timed_out compute_energy_j tx_energy_j return_energy_j flight_energy_j invalid_placements
0 0 random 0.737111 0.613050 0.780552 0.668286 979.000000 3980.011429 323.032830 119.262521 135051.852875 0.000000
0 0 local 0.662460 0.613050 0.658782 0.668286 1257.000000 0.000000 0.000000 0.000000 135051.852875 0.000000
check 0 0 local: offers=True uav_positions=True
0 0 greedy 0.862513 0.613050 0.985107 0.668286 512.000000 5132.020452 57.832571 99.603984 135051.852875 0.000000
check 0 0 greedy: offers=True uav_positions=True
0 0 heft 0.867078 0.613050 0.992554 0.668286 495.000000 4088.946796 167.194508 62.841431 135051.852875 0.000000
check 0 0 heft: offers=True uav_positions=True
random mean 0.737111 0.613050 0.780552 0.668286 979.000000 3980.011429 323.032830 119.262521 135051.852875 0.000000
random std 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
local mean 0.662460 0.613050 0.658782 0.668286 1257.000000 0.000000 0.000000 0.000000 135051.852875 0.000000
local std 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
greedy mean 0.862513 0.613050 0.985107 0.668286 512.000000 5132.020452 57.832571 99.603984 135051.852875 0.000000
greedy std 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
heft mean 0.867078 0.613050 0.992554 0.668286 495.000000 4088.946796 167.194508 62.841431 135051.852875 0.000000
heft std 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
invalid_placements_total=0; offers_and_uav_positions_equal=True
```

逐字比较：`True`。相同输入为 `(seed, episode)=(0,0)`。

**最终冒烟输出**

命令：

```powershell
Set-Location D:\CodeFile\HyperUAV-qly
& .\output\mappo-env\Scripts\python.exe -B -X utf8 .\scripts\train_mappo.py train --updates 2 --episodes-per-update 1
```

下面为最终 float64 环境动作版本的两次更新；此前 float32 接口版本的初次冒烟不用于本报告。后续仅整理函数命名、TOML 中文注释及移除基线未计算的回报列。

```json
{"kind": "train_episode", "update": 1, "seed": 0, "episode": 0, "elapsed_s": 10.380578399985097, "episode_return": -165.05015328670598, "offers": 3724, "on_time_rate": 0.7473147153598282, "average_delay_s": 0.8642529508836643, "covered_share": 0.5128893662728249, "invalid_placements": 0, "compute_energy_j": 3598.500806647936, "tx_energy_j": 306.4264710109892, "return_energy_j": 108.22839007751212, "flight_energy_j": 145908.10966682792}
{"kind": "update", "update": 1, "elapsed_s": 13.002081099897623, "episode_return": -165.05015328670598, "on_time_rate": 0.7473147153598282, "average_delay_s": 0.8642529508836643, "covered_share": 0.5128893662728249, "invalid_placements": 0, "compute_energy_j": 3598.500806647936, "tx_energy_j": 306.4264710109892, "return_energy_j": 108.22839007751212, "flight_energy_j": 145908.10966682792, "value_loss": 0.5232578039169311, "policy_loss": -0.014290559907878494, "entropy": 21.624261347452798, "approx_kl": 0.005996585101820528, "clip_fraction": 0.08000000029181441, "actor_grad_norm": 0.3722371498743693, "critic_grad_norm": 1.1571246753136317, "valuenorm_mean": [-12.65754222869873], "valuenorm_variance": [6.088836669921875]}
{"kind": "train_episode", "update": 2, "seed": 0, "episode": 1, "elapsed_s": 14.448480800027028, "episode_return": -186.8784490735886, "offers": 4006, "on_time_rate": 0.6884672990514229, "average_delay_s": 0.9096679373277534, "covered_share": 0.5591612581128308, "invalid_placements": 0, "compute_energy_j": 4390.156297970041, "tx_energy_j": 410.5116613354627, "return_energy_j": 143.26683020677564, "flight_energy_j": 146798.69636325783}
{"kind": "update", "update": 2, "elapsed_s": 17.6521487000864, "episode_return": -186.8784490735886, "on_time_rate": 0.6884672990514229, "average_delay_s": 0.9096679373277534, "covered_share": 0.5591612581128308, "invalid_placements": 0, "compute_energy_j": 4390.156297970041, "tx_energy_j": 410.5116613354627, "return_energy_j": 143.26683020677564, "flight_energy_j": 146798.69636325783, "value_loss": 0.8413454949855804, "policy_loss": -0.01065107394382352, "entropy": 26.8233154296875, "approx_kl": 0.002711074918261147, "clip_fraction": 0.025466666960467895, "actor_grad_norm": 0.39039788444836937, "critic_grad_norm": 4.643683401743571, "valuenorm_mean": [-18.73809051513672], "valuenorm_variance": [57.532135009765625]}
```

- 两回合 invalid_placements 合计为 0。环境动作校验逐项验证候选合法性和每个待决策子任务恰好出现一次。
- loss、熵、KL、裁剪比例及梯度范数是每次更新内 15 轮的均值；梯度范数为裁剪前的范数。
- ValueNorm 依官方逻辑每个 minibatch 更新，日志记录每次 PPO 更新结束后的去偏均值与方差。
- 单时隙接口检查：观测 (5,32)、状态 (180,)、候选特征 (75,6,16)，75 个任务全部在候选内，确定性动作复现且不推进采样 RNG；环境动作 float64，缓存原始高斯动作 float32。
- 从指定官方源码提取对应类/函数，在相同输入上比较 ValueNorm、GAE、MLP、价值损失，最大差均为 0。未导入官方环境包；首次整包导入因缺少 absl 失败，未为此安装无关依赖。

**奖励核对**

- `-sum(reward) * expected_arrivals / DAG_count = 0.6746593930957117`
- `mean(delay / deadline) = 0.6746593930957120`
- 贪心 `(0,0)`，3724 个 DAG，invalid_placements=0。差异来自浮点累加次序，未设置容差或断言门槛。

**评估集基线**

使用 scripts/run_baselines.py 的 run_episode 和原四个策略，seed=1、episode=0–9，每个策略共 38,716 个 DAG。通过调用侧 Simulator 子类收集时延；未修改环境源码或基线运行逻辑。

下表按全部 DAG 汇总，与本训练入口的 MAPPO 评估口径相同。平均时延包含超时 DAG 的实际完成时刻。

| 基线 | 按时完成率 | 平均时延/s | 被覆盖 DAG 占比 |
|---|---:|---:|---:|
| choose_random | 0.7113338155 | 0.8892395440 | 0.6076040913 |
| choose_local | 0.6550263457 | 0.9788106597 | 0.6076040913 |
| choose_greedy | 0.8576815787 | 0.6935180236 | 0.6076040913 |
| choose_heft | 0.8632606674 | 0.6832516238 | 0.6076040913 |

以下是逐 episode 等权均值；完成率和覆盖占比对应原 run_baselines.py 的 mean 输出口径。

| 基线 | 按时完成率 | 平均时延/s | 被覆盖 DAG 占比 |
|---|---:|---:|---:|
| choose_random | 0.7113118579 | 0.8892355027 | 0.6079605784 |
| choose_local | 0.6550664259 | 0.9787404069 | 0.6079605784 |
| choose_greedy | 0.8577534664 | 0.6933939278 | 0.6079605784 |
| choose_heft | 0.8633457752 | 0.6830829993 | 0.6079605784 |

40 个基线回合采样耗时合计 272.529813 s；invalid_placements 合计 0；30 组与随机策略的到达序列、轨迹比较全部相同。
原始基线日志的早期 summary 带 episode_return=0 占位列，未计算且不作为结果；当前脚本已移除该列，所有逐回合物理指标及上表统计不受影响。

**证据与边界**

- 最终冒烟日志：`D:\CodeFile\HyperUAV-qly\output\mappo\20261106_212719_186155\metrics.jsonl`；checkpoint_0001.pt 与 checkpoint_0002.pt 在同目录。
- 奖励日志：`D:\CodeFile\HyperUAV-qly\output\mappo\20261106_212104_751694\metrics.jsonl`。
- 基线逐回合日志：`D:\CodeFile\HyperUAV-qly\output\mappo\20261106_212105_399609\metrics.jsonl`。
- 机器可读结果：`D:\CodeFile\HyperUAV-qly\output\mappo-check\result.json`。
- 改名前后日志、官方数值对照与接口检查保存在 output/mappo-check/。
- 仅完成本地两次更新。未验证长期回报趋势、每十次更新评估调度的实际触发、默认每批 32 个 episode 的资源消耗；不据此声称已学会或已超过基线。
- 不选择服务器训练更新次数；以本报告回合与更新耗时供用户决定。

**服务器验收结果（最终依据，2026-11-06）**

- 服务器：10.12.54.24；执行目录 /data2/zrj2025/hyperjet-baselines-20261106。
- PID 3673508 已退出；日志最终标记 server_checks_complete elapsed_s=201。启动脚本使用 set -e，三项命令均成功后才写此完成标记。
- 总耗时 201 秒；CPU 执行，PyTorch 单线程。Python 3.12.13、NumPy 1.26.4、PyTorch 2.2.0+cu118。
- 本地先前使用 PyTorch 2.15.0+cpu；两端训练采样结果分别记录，不将其当作同一软件环境下的逐动作复现验证。
- 未提交代码，未修改环境或旧方法。服务器 HEAD 为 c86b21ef819989e01b95ac0f256508fffd39b1da；实际 dirty 状态如下：

```text
 M config/channel.toml
 M config/user.toml
RM methods/baselines/policies.py -> methods_v2/baselines/policies.py
R  methods/ers.py -> methods_v2/ers.py
 M scripts/check_environment.py
 M scripts/run_baselines.py
?? config/methods/
?? methods_v2/learning/
?? methods_v2/mappo/
?? output/
?? scripts/sweep_baselines.py
?? scripts/train_mappo.py
```

两次更新的完整输出：

```json
{"kind": "train_episode", "update": 1, "seed": 0, "episode": 0, "elapsed_s": 7.426481362432241, "episode_return": -163.94833780590693, "offers": 3724, "on_time_rate": 0.7607411385606875, "average_delay_s": 0.8584835089192228, "covered_share": 0.5107411385606875, "invalid_placements": 0, "compute_energy_j": 3597.9500775710226, "tx_energy_j": 309.4439678335155, "return_energy_j": 107.17094423622649, "flight_energy_j": 146732.74626019728}
{"kind": "update", "update": 1, "elapsed_s": 9.854840371757746, "episode_return": -163.94833780590693, "on_time_rate": 0.7607411385606875, "average_delay_s": 0.8584835089192228, "covered_share": 0.5107411385606875, "invalid_placements": 0, "compute_energy_j": 3597.9500775710226, "tx_energy_j": 309.4439678335155, "return_energy_j": 107.17094423622649, "flight_energy_j": 146732.74626019728, "value_loss": 0.4646834433078766, "policy_loss": -0.010950077216451366, "entropy": 22.487023162841798, "approx_kl": 0.0041606567450799044, "clip_fraction": 0.042200000025331974, "actor_grad_norm": 0.31266262630621594, "critic_grad_norm": 1.7796807368596395, "valuenorm_mean": [-12.600173950195312], "valuenorm_variance": [5.84637451171875]}
{"kind": "train_episode", "update": 2, "seed": 0, "episode": 1, "elapsed_s": 5.766415484249592, "episode_return": -178.8378831681904, "offers": 4006, "on_time_rate": 0.7241637543684474, "average_delay_s": 0.870528887114257, "covered_share": 0.37943085371942087, "invalid_placements": 0, "compute_energy_j": 2917.2382196590693, "tx_energy_j": 209.8693869639643, "return_energy_j": 81.32063248812568, "flight_energy_j": 147311.7579688454}
{"kind": "update", "update": 2, "elapsed_s": 7.5386674627661705, "episode_return": -178.8378831681904, "on_time_rate": 0.7241637543684474, "average_delay_s": 0.870528887114257, "covered_share": 0.37943085371942087, "invalid_placements": 0, "compute_energy_j": 2917.2382196590693, "tx_energy_j": 209.8693869639643, "return_energy_j": 81.32063248812568, "flight_energy_j": 147311.7579688454, "value_loss": 0.7447043816248576, "policy_loss": -0.012685920813058733, "entropy": 17.649089177449543, "approx_kl": 0.0036711525836532624, "clip_fraction": 0.04200000039612253, "actor_grad_norm": 0.39167732099692026, "critic_grad_norm": 4.281144561370214, "valuenorm_mean": [-18.837970733642578], "valuenorm_variance": [56.088775634765625]}
```

- 每次更新采样 1 个完整 episode，依次使用 (seed, episode)=(0,0)、(0,1)。
- 回合用时分别 7.426481 s、5.766415 s；含采样的更新总用时分别 9.854840 s、7.538667 s。
- ValueNorm 去偏均值/方差：更新 1 为 -12.600173950195312 / 5.84637451171875；更新 2 为 -18.837970733642578 / 56.088775634765625。
- 两次更新均有有限的损失与梯度值；所有时隙 invalid_placements=0，环境已校验候选合法性、缺项和重复项。
- 这些结果确认两次更新能完成；没有据此作出长期学习效果判断。

奖励核对：

```json
{"kind": "reward_check", "seed": 0, "episode": 0, "offers": 3724, "restored_delay": 0.6746593930957117, "mean_normalized_delay": 0.674659393095712, "invalid_placements": 0}
```

评估集 seed=1、episode=0–9；每个策略 38,716 个 DAG。下表按全部 DAG 汇总；平均时延包括超时 DAG 的实际完成时延。

| 基线 | 按时完成率 | 平均时延/s | 被覆盖 DAG 占比 |
|---|---:|---:|---:|
| choose_random | 0.7113338155 | 0.8892395440 | 0.6076040913 |
| choose_local | 0.6550263457 | 0.9788106597 | 0.6076040913 |
| choose_greedy | 0.8576815787 | 0.6935180236 | 0.6076040913 |
| choose_heft | 0.8632606674 | 0.6832516238 | 0.6076040913 |

逐 episode 等权均值（原 run_baselines.py 的 mean 口径）：

| 基线 | 按时完成率 | 平均时延/s | 被覆盖 DAG 占比 |
|---|---:|---:|---:|
| choose_random | 0.7113118579 | 0.8892355027 | 0.6079605784 |
| choose_local | 0.6550664259 | 0.9787404069 | 0.6079605784 |
| choose_greedy | 0.8577534664 | 0.6933939278 | 0.6079605784 |
| choose_heft | 0.8633457752 | 0.6830829993 | 0.6079605784 |

- 基线 40/40 回合完成，回合耗时合计 171.376579 秒。
- 30 组与随机基线的比较均 offers_equal=True、positions_equal=True。
- 两次训练回合、奖励核对和全部基线回合的 invalid_placements 合计为 0。
- 服务器基线日志已移除未计算的 episode_return 占位列。
- 保留最初改名前后的完整基线输出和官方数值对照记录，见前文。

已同步回本地的产物：

- `D:\CodeFile\HyperUAV-qly\output\mappo-check\server.log`
- `D:\CodeFile\HyperUAV-qly\output\mappo\20261106_213910_116861\metrics.jsonl`
- `D:\CodeFile\HyperUAV-qly\output\mappo\20261106_213932_063010\metrics.jsonl`
- `D:\CodeFile\HyperUAV-qly\output\mappo\20261106_213937_853579\metrics.jsonl`
- `D:\CodeFile\HyperUAV-qly\output\mappo\20261106_213910_116861\checkpoint_0001.pt`（446562 字节）
- `D:\CodeFile\HyperUAV-qly\output\mappo\20261106_213910_116861\checkpoint_0002.pt`（446562 字节）
- `D:\CodeFile\HyperUAV-qly\output\mappo-check\server_result.json`：服务器机器可读结果。
- `D:\CodeFile\HyperUAV-qly\output\mappo-check\result.json`：合并服务器和先前本地记录，primary_execution=server。

**两处修正、并行采样与 50 次更新（2026-11-06）**

本次按用户要求修改并在本地验收；全部本地检查已通过。服务器首轮计时已启动，未达到 50 次更新；后续状态与结果在本节继续追加。未 commit，output 未暂存。

执行基线版本：HEAD `c86b21ef819989e01b95ac0f256508fffd39b1da` + dirty/new。未计算文件哈希。

**文件变化**

| 文件 | 修改前行数 | 修改后行数 | 增/删行 |
|---|---:|---:|---:|
| `scripts/run_baselines.py` | 108 | 113 | +6/−1 |
| `scripts/train_mappo.py` | 235 | 277 | +124/−82 |
| `scripts/sweep_baselines.py` | 118 | 119 | +2/−1 |
| `methods_v2/mappo/features.py` | 134 | 134 | +3/−3 |
| `methods_v2/mappo/policy.py` | 76 | 76 | +1/−1 |
| `config/methods/mappo.toml` | 47 | 49 | +4/−2 |

- run_baselines.run_episode 复用 compute_total_delay，返回 total_delay_s 及已有原始计数；原 METRICS、打印与检查逻辑保持不变。evaluate_baselines 直接使用这些数据，删掉 MeasuredSimulator 和 unittest.mock.patch。
- features.py 仅将 parents/successors 改名为 has_parent/has_child；缩放仍集中在文件顶部并使用 settings 分母。
- 常驻 spawn 进程池中，每个进程只建一次 Simulator、FeatureBuilder、Policy，torch 单线程。训练与确定性评估共用该池；参数在每批采样结束后才更新，样本由 Pool.map 按 episode 顺序收集。
- 采样 torch RNG 由 SeedSequence([train_seed, episode]) 派生；minibatch 使用独立 NumPy RNG。checkpoint 保存模型、优化器、ValueNorm、NumPy RNG、next_episode 和配置；--resume 的 --updates 表示累计更新次数目标，日志追加到 checkpoint 所在目录。
- sampling_s 包含采样进程启动（首批）、任务传递、回合采样和收集；update_s 是 PPO 更新耗时；elapsed_s 是两者所在的整次训练更新耗时，不含之后的评估与 checkpoint 保存。
- 直接受影响的 sweep_baselines.py 改为只展开原有 CSV 列，避免附加计数/时延字段进入 CSV；实际 CSV 写入检查确认列名和顺序不变。
- 首次并行对照发现 actor 的 hidden[owners] 在 CPU 8 线程反向累加时不确定。相同输入重复反向的最大差为 3.814697265625e-06；改为数学等价的 hidden.index_select(0, owners) 后为 0。仅修正该索引操作，没有增添算法开关或浮点容差。

**本地重跑结果**

奖励核对（与上次两个数逐值相同）：0.6746593930957117 / 0.674659393095712。

基线 seed=1、episode=0–9，按全体 DAG 汇总；四个基线的全部现有 summary 字段与此前本地记录逐值相同。

| 基线 | 按时完成率 | 平均时延/s | 覆盖占比 | 与上次完全相同 |
|---|---:|---:|---:|---|
| choose_random | 0.7113338155 | 0.8892395440 | 0.6076040913 | True |
| choose_local | 0.6550263457 | 0.9788106597 | 0.6076040913 | True |
| choose_greedy | 0.8576815787 | 0.6935180236 | 0.6076040913 | True |
| choose_heft | 0.8632606674 | 0.6832516238 | 0.6076040913 | True |

**并行与续训精确核对**

两组均 episodes_per_update=4、training_threads=8、updates=2；分别 rollout_workers=1 和 4。比较不使用容差，排除 elapsed_s/sampling_s/update_s，其他全部字段逐值比较。

| episode | 1 进程回报 | 4 进程回报 | 全部非时间字段相同 |
|---:|---:|---:|---|
| 0 | -164.08138520769305 | -164.08138520769305 | True |
| 1 | -180.39084019636422 | -180.39084019636422 | True |
| 2 | -180.03343831643903 | -180.03343831643903 | True |
| 3 | -193.3261581030998 | -193.3261581030998 | True |
| 4 | -173.7400694176366 | -173.7400694176366 | True |
| 5 | -169.40969441887904 | -169.40969441887904 | True |
| 6 | -165.18919391195996 | -165.18919391195996 | True |
| 7 | -167.0201854881602 | -167.0201854881602 | True |

| 更新 | 指标 | 1 进程 | 4 进程 | 完全相同 |
|---:|---|---:|---:|---|
| 1 | value_loss | 0.6955239693323771 | 0.6955239693323771 | True |
| 1 | policy_loss | -0.005622441001391584 | -0.005622441001391584 | True |
| 1 | entropy | 23.728568522135415 | 23.728568522135415 | True |
| 1 | approx_kl | 0.0024491418414982037 | 0.0024491418414982037 | True |
| 1 | clip_fraction | 0.020300000091083348 | 0.020300000091083348 | True |
| 1 | actor_grad_norm | 0.14244410693645476 | 0.14244410693645476 | True |
| 1 | critic_grad_norm | 1.0076734145482382 | 1.0076734145482382 | True |
| 1 | valuenorm_mean | [-13.87136173248291] | [-13.87136173248291] | True |
| 1 | valuenorm_variance | [9.150848388671875] | [9.150848388671875] | True |
| 2 | value_loss | 0.9535459280014038 | 0.9535459280014038 | True |
| 2 | policy_loss | -0.006792242320254616 | -0.006792242320254616 | True |
| 2 | entropy | 20.79958699544271 | 20.79958699544271 | True |
| 2 | approx_kl | 0.004160522379000516 | 0.004160522379000516 | True |
| 2 | clip_fraction | 0.04903333314384024 | 0.04903333314384024 | True |
| 2 | actor_grad_norm | 0.19892862538496653 | 0.19892862538496653 | True |
| 2 | critic_grad_norm | 3.9870426456133523 | 3.9870426456133523 | True |
| 2 | valuenorm_mean | [-19.184654235839844] | [-19.184654235839844] | True |
| 2 | valuenorm_variance | [45.96343994140625] | [45.96343994140625] | True |

- 最终 8/8 个 train_episode 行及 2/2 个 update 行的全部非时间字段相同；invalid_placements 均为 0。
- 将 4 进程第一次更新 checkpoint 复制到独立验收目录，再以 1 进程续训到第 2 次：episode 4–7 和第二次更新全部非时间字段与连续训练精确相同。
- 初次本地 spawn 被沙箱命名管道权限拦截，使用沙箱外执行完成；一次同时启动的输出目录重名导致 1 进程未启动，随后单独重跑。均未添加兼容层或回退。
- 所有比较结果与原始数值见 output/mappo-parallel/comparison.json、regression.json；失败的初次索引版本日志单独保留为 workers_1_index.log / workers_4_index.log。

本地证据路径：

- `D:\CodeFile\HyperUAV-qly\output\mappo-parallel\reward.log`
- `D:\CodeFile\HyperUAV-qly\output\mappo-parallel\baselines.log`
- `D:\CodeFile\HyperUAV-qly\output\mappo-parallel\workers_1.log`
- `D:\CodeFile\HyperUAV-qly\output\mappo-parallel\workers_4.log`
- `D:\CodeFile\HyperUAV-qly\output\mappo-parallel\resume.log`
- `D:\CodeFile\HyperUAV-qly\output\mappo-parallel\index_gradient.log`
- `D:\CodeFile\HyperUAV-qly\output\mappo-parallel\result.json`：本次机器可读记录。

**服务器执行安排**

使用 episodes_per_update=32、rollout_workers=32、training_threads=8。先执行 train --updates 1 检查首次更新 elapsed_s；超过 180 秒则停止并报告，否则从该 checkpoint 续训到累计 50 次，episode 不重复。用户要求后台启动后停止监控，待通知完成后回收日志并追加完整 50 次统计和评估表。

服务器首轮与后台续训启动记录：

- 首轮 sampling_s=22.941807381808758，update_s=73.97231094166636，elapsed_s=96.91426552459598；小于 180 秒，按用户授权继续。invalid_placements=0。
- 2026-11-06 22:48:37（北京时间）后台启动续训，PID 3753927，从 checkpoint_0001.pt 接着运行至累计第 50 次更新。
- 参数：episodes_per_update=32，rollout_workers=32，training_threads=8；episode 从 32 继续，未重新采样前 32 个 episode。
- 按首轮外推，剩余约 80–100 分钟，预计北京时间 2026-11-07 00:10–00:30 完成；这是估计，尚无后续更新实测。
- 按用户要求，后台启动后停止监控；没有读取后续训练进度。50 次逐轮耗时均值、每 5 次训练表和第 10/20/30/40/50 次评估表，待用户通知完成后回收并追加。
- metrics.jsonl：`/data2/zrj2025/hyperjet-baselines-20261106/output/mappo/20261106_224525_845330/metrics.jsonl`。
- 后台标准输出：`/data2/zrj2025/hyperjet-baselines-20261106/output/mappo-parallel/training.log`。
- 首轮日志已取回本地 output/mappo-parallel/first.log；启动与版本记录见同目录 server_job.json、server-status.txt、server-head.txt。
- 本次仍未 commit，output 未暂存。

服务器运行故障与恢复（2026-11-06 23:00，北京时间）：

- 用户未看到进程后单次查询确认，原 PID 3753927 已退出，报 OSError: [Errno 24] Too many open files；不是仍在运行。
- 当时软限制为 1024、硬限制为 1048576。完整保存到 update 2（elapsed_s=134.6680473536253），尚无 update 3 记录。
- 仅在新启动 shell 中将 ulimit -n 提高至 65536，没有修改训练源码、参数或系统全局配置。
- 23:00:20 从 checkpoint_0002.pt 恢复到累计 50 次更新，新 PID 3770550；一次性确认进程存活且文件句柄软限制为 65536。
- 新标准输出为 /data2/zrj2025/hyperjet-baselines-20261106/output/mappo-parallel/training_resume.log；原 training.log 保留。metrics.jsonl 仍在原目录，继续追加。先前完成时间估计已失效，恢复后约需两小时，尚未实测剩余更新。
- 按用户要求，恢复并确认启动后不持续监控。

**50 次服务器训练完整结果（2026-11-07 回收）**

已确认训练主进程退出、update 1–50 和 evaluation 10/20/30/40/50 完整存在，checkpoint_0050.pt 存在。全部日志及 50 个 checkpoint 已同步回本地。
- 训练 episode：1600 个，编号 0–1599，唯一编号 1600 个；评估 5×10=50 个回合。
- 训练及评估时隙的 invalid_placements 合计：0。
- 参数仍为每批 32 个完整 episode、32 个采样进程、PPO 主进程 8 线程，采样进程各 1 线程。曾因文件句柄限制退出，从第 2 次 checkpoint 恢复；前文保留故障及恢复记录。本次没有修改算法或训练超参。
- 采样用时均值：15.588433 s；PPO 更新用时均值：54.276297 s；每次训练更新总用时均值：69.865217 s。
- 50 次训练更新累计计时：3493.260836 s（约 58 分 13 秒），不含进程重启间隔、确定性评估和 checkpoint 保存；不当作整次任务的墙钟耗时。

每次更新的采样与 PPO 更新用时：

| 更新 | 采样/s | PPO 更新/s | 合计/s |
|---:|---:|---:|---:|
| 1 | 22.942 | 73.972 | 96.914 |
| 2 | 37.784 | 96.884 | 134.668 |
| 3 | 31.056 | 95.258 | 126.314 |
| 4 | 25.081 | 92.236 | 117.317 |
| 5 | 25.657 | 85.160 | 110.836 |
| 6 | 12.696 | 58.142 | 70.837 |
| 7 | 20.565 | 62.930 | 83.495 |
| 8 | 18.548 | 66.975 | 85.523 |
| 9 | 20.773 | 62.483 | 83.256 |
| 10 | 20.936 | 60.627 | 81.563 |
| 11 | 20.580 | 40.380 | 60.960 |
| 12 | 12.530 | 39.588 | 52.118 |
| 13 | 14.139 | 48.683 | 62.822 |
| 14 | 25.850 | 94.076 | 119.926 |
| 15 | 22.317 | 90.443 | 112.760 |
| 16 | 22.951 | 79.375 | 102.326 |
| 17 | 15.642 | 51.115 | 66.757 |
| 18 | 13.388 | 51.189 | 64.577 |
| 19 | 11.416 | 51.211 | 62.627 |
| 20 | 14.552 | 48.414 | 62.966 |
| 21 | 11.833 | 46.163 | 57.996 |
| 22 | 11.661 | 44.693 | 56.354 |
| 23 | 12.610 | 44.080 | 56.690 |
| 24 | 11.379 | 44.744 | 56.124 |
| 25 | 10.537 | 43.757 | 54.294 |
| 26 | 13.506 | 44.275 | 57.782 |
| 27 | 13.647 | 45.035 | 58.682 |
| 28 | 10.129 | 39.926 | 50.055 |
| 29 | 13.163 | 42.208 | 55.371 |
| 30 | 11.619 | 40.312 | 51.931 |
| 31 | 10.192 | 42.705 | 52.897 |
| 32 | 10.377 | 40.647 | 51.024 |
| 33 | 12.582 | 42.652 | 55.235 |
| 34 | 12.316 | 45.177 | 57.493 |
| 35 | 10.921 | 40.189 | 51.111 |
| 36 | 12.659 | 40.872 | 53.531 |
| 37 | 10.829 | 38.146 | 48.975 |
| 38 | 13.936 | 42.337 | 56.273 |
| 39 | 10.098 | 38.387 | 48.485 |
| 40 | 8.744 | 25.965 | 34.710 |
| 41 | 9.314 | 26.979 | 36.292 |
| 42 | 10.181 | 24.680 | 34.861 |
| 43 | 9.134 | 31.806 | 40.940 |
| 44 | 9.924 | 62.043 | 71.967 |
| 45 | 20.048 | 80.206 | 100.254 |
| 46 | 18.301 | 69.532 | 87.833 |
| 47 | 18.589 | 81.889 | 100.479 |
| 48 | 19.271 | 77.222 | 96.493 |
| 49 | 9.978 | 42.650 | 52.628 |
| 50 | 12.539 | 35.400 | 47.939 |
| 均值 | 15.588 | 54.276 | 69.865 |

每 5 次更新的训练指标；回报为 32 个 episode 的平均回报，完成率、时延及覆盖占比按全部 DAG 汇总。损失、熵、KL、裁剪比例取该次更新 15 轮 PPO 的均值。

| 更新 | 训练回报 | 按时完成率/% | 平均时延/s | 覆盖占比/% | value loss | 熵 | 近似 KL | 裁剪比例/% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | -177.306730 | 70.7222 | 0.892757 | 56.9136 | 0.306353 | 22.660454 | 0.002740 | 2.3996 |
| 10 | -177.148040 | 71.5053 | 0.886671 | 53.4174 | 0.173167 | 21.078042 | 0.002390 | 2.0631 |
| 15 | -174.962670 | 71.9272 | 0.880514 | 50.7678 | 0.101228 | 20.094585 | 0.003768 | 4.1665 |
| 20 | -177.211960 | 71.4239 | 0.883780 | 43.6270 | 0.038183 | 17.690392 | 0.003746 | 4.0715 |
| 25 | -172.845983 | 72.6222 | 0.872310 | 37.9226 | 0.021823 | 15.638093 | 0.003725 | 4.2198 |
| 30 | -176.629360 | 71.8088 | 0.880311 | 36.4818 | 0.016148 | 15.800436 | 0.004293 | 4.7754 |
| 35 | -175.227766 | 71.9254 | 0.879203 | 37.4422 | 0.011812 | 15.637085 | 0.004080 | 4.7883 |
| 40 | -172.785927 | 72.2730 | 0.874397 | 37.6454 | 0.010069 | 15.544408 | 0.004433 | 5.3465 |
| 45 | -175.149029 | 71.9509 | 0.878766 | 35.5501 | 0.007473 | 14.695928 | 0.005004 | 6.0512 |
| 50 | -175.962139 | 71.8980 | 0.878916 | 33.0801 | 0.005549 | 13.905874 | 0.004609 | 5.5921 |

评估与四个基线放在同一表中：固定 seed=1、episode=0–9，每个策略/评估点均 38,716 个 DAG，按全体 DAG 汇总。MAPPO 为确定性飞行均值和候选 argmax；基线沿用各自随机飞行，因此覆盖占比随策略而不同。

| 方法/更新 | 按时完成率/% | 平均时延/s | 覆盖占比/% | 与随机基线之差/百分点 |
|---|---:|---:|---:|---:|
| choose_random | 71.1334 | 0.889240 | 60.7604 | +0.0000 |
| choose_local | 65.5026 | 0.978811 | 60.7604 | -5.6307 |
| choose_greedy | 85.7682 | 0.693518 | 60.7604 | +14.6348 |
| choose_heft | 86.3261 | 0.683252 | 60.7604 | +15.1927 |
| MAPPO / 10 | 71.4330 | 0.873547 | 50.2041 | +0.2996 |
| MAPPO / 20 | 70.4050 | 0.892422 | 36.5740 | -0.7284 |
| MAPPO / 30 | 72.0322 | 0.864284 | 37.4238 | +0.8989 |
| MAPPO / 40 | 71.9186 | 0.869451 | 32.2683 | +0.7852 |
| MAPPO / 50 | 72.0684 | 0.865627 | 33.6373 | +0.9350 |

评估四类能耗，单位 J，均为每个完整 episode 的平均值：

| 更新 | 计算能耗 | 上传/中间传输能耗 | 回传能耗 | 飞行能耗 |
|---:|---:|---:|---:|---:|
| 10 | 2253.652862 | 42.893005 | 43.364544 | 152714.567749 |
| 20 | 1640.752578 | 31.891472 | 29.645780 | 154088.110001 |
| 30 | 1922.922042 | 36.714848 | 37.341285 | 154315.415981 |
| 40 | 1911.782131 | 32.039622 | 32.903552 | 154810.078518 |
| 50 | 2022.725322 | 37.493245 | 33.803080 | 155804.362693 |

结果与边界：

- 前 5 次更新的平均训练回报为 -178.720737，后 5 次为 -176.745031，整体略有改善但存在波动。前后批次的 episode 不同，此比较仅描述训练曲线。
- 第 50 次确定性评估完成率 72.0684%，比随机基线高 0.9350 个百分点；比贪心低 13.6998 个百分点，比 HEFT 式低 14.2577 个百分点。
- 第 20 次评估低于随机；第 10–50 次评估覆盖占比从 50.2041% 降到 33.6373%。覆盖下降与性能差距同时存在，但本实验没有单独隔离飞行与卸载决策的因果贡献。
- 本次证实并行采样、PPO 更新、续训和评估能完成；50 次结果未显示出明确、持续超过随机基线的优势，不能据此判定“明显高于随机”已达成。没有追加训练、调参或改变奖励。

本地完整产物：

- metrics.jsonl：`D:\CodeFile\HyperUAV-qly\output\mappo\20261106_224525_845330\metrics.jsonl`。
- 最终 checkpoint：`D:\CodeFile\HyperUAV-qly\output\mappo\20261106_224525_845330\checkpoint_0050.pt`。
- 原始失败日志：`D:\CodeFile\HyperUAV-qly\output\mappo-parallel\training.log`；恢复日志：`D:\CodeFile\HyperUAV-qly\output\mappo-parallel\training_resume.log`。
- 结构化结果（含全部 50 行用时、指标和 5 次评估）：`D:\CodeFile\HyperUAV-qly\output\mappo-parallel\training_result.json`。
- 服务器 metrics.jsonl：`/data2/zrj2025/hyperjet-baselines-20261106/output/mappo/20261106_224525_845330/metrics.jsonl`。
- 代码版本及本次修改行数、奖励/基线回归、1/4 进程精确对比、续训对比见同一节前文。未 commit；output 未暂存。

## 覆盖下降诊断

本节只重放已有 checkpoint，没有修改训练代码、配置或方法实现，也没有重新训练。服务器执行目录为 `/data2/zrj2025/hyperjet-baselines-20261106`，HEAD 仍为 `c86b21ef819989e01b95ac0f256508fffd39b1da`；工作树保留前述 dirty/new 文件，并新增未跟踪的 `scripts/diagnose_mappo.py` 和诊断输出。诊断脚本 301 行，服务器运行 58 秒，8 个完整 episode 的 `invalid_placements` 均为 0。

固定重放口径：

- evaluation：seed=1、episode=0，确定性飞行均值和候选 argmax；
- training：seed=0、episode=2000，按训练方式采样，torch 生成器由 `SeedSequence([0, 2000])` 派生。该 episode 未出现在 0–1599 的训练集中；每个 checkpoint 都按同一规则重新初始化环境和采样生成器。由于各策略产生的候选任务数不同，后续随机数消耗不作为跨 checkpoint 的逐时隙 common-random 对齐；
- “平均绝对均值”是全部 UAV、全部时隙、两个飞行分量的 `abs(mean)` 平均；越界和边界比例的分母均为 1000 个 UAV-时隙；
- 前/后 50 时隙的覆盖占比按各段覆盖 DAG 总数除以新 DAG 总数，不是逐时隙比例的平均；“第 200 时隙位置”取 slot 199 动作执行后的 episode 终点。

重放核对：checkpoint 10、20、50 的 evaluation episode 0 在 offers、总完成率、平均时延、覆盖占比和 invalid_placements 五项上与原 `metrics.jsonl` 的对应行逐值相同；checkpoint 1 当时不在 evaluation interval 上，没有对应旧日志行。

### Evaluation 口径

| checkpoint | log_std (x, y) | std (x, y) | 飞行均值向量 | 平均绝对均值 | 均值分量绝对值 > 1 | 越界 UAV-时隙 | 边界 < 30 m | 两两平均距离前/后 50 /m | 覆盖占比前/后 50 | 覆盖内完成率/时延 | 覆盖外完成率/时延 | 子任务 UE/自己/邻居 | 第 200 时隙 5 架 UAV 位置 /m |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|
| 0001 | 0.0068, 0.0026 | 1.0068, 1.0026 | (-0.0048, 0.0113) | 0.0326 | 0.00% | 19.20% | 24.50% | 286.31 / 343.90 | 49.00% / 53.23% | 75.15% / 0.8077 s | 65.19% / 0.9896 s | 41.36% / 13.84% / 44.80% | (356.7,272.8); (402.7,468.5); (599.5,261.1); (454.3,512.0); (0.7,109.1) |
| 0010 | 0.0324, 0.0102 | 1.0330, 1.0103 | (0.0142, -0.2684) | 0.1892 | 0.00% | 48.80% | 66.10% | 253.67 / 210.00 | 49.00% / 14.90% | 83.78% / 0.6517 s | 65.06% / 0.9851 s | 59.98% / 23.68% / 16.34% | (465.8,1.9); (580.3,0.0); (589.2,1.1); (558.3,1.1); (112.1,0.2) |
| 0020 | 0.0853, 0.0462 | 1.0891, 1.0473 | (0.2013, -0.3396) | 0.2775 | 0.00% | 80.00% | 86.20% | 248.72 / 251.26 | 29.61% / 14.27% | 74.16% / 0.8217 s | 65.42% / 0.9838 s | 82.39% / 11.35% / 6.26% | (595.5,3.1); (599.6,169.5); (599.9,143.5); (599.0,225.2); (172.8,0.0) |
| 0050 | 0.1943, 0.2025 | 1.2144, 1.2244 | (0.7176, -0.5459) | 0.6318 | 0.00% | 93.50% | 94.80% | 247.05 / 242.38 | 17.81% / 13.65% | 90.10% / 0.5452 s | 65.21% / 0.9865 s | 57.99% / 22.96% / 19.05% | (595.8,150.8); (594.2,348.2); (599.7,259.0); (599.2,296.5); (238.1,2.0) |

### Training 口径

| checkpoint | log_std (x, y) | std (x, y) | 飞行均值向量 | 平均绝对均值 | 均值分量绝对值 > 1 | 越界 UAV-时隙 | 边界 < 30 m | 两两平均距离前/后 50 /m | 覆盖占比前/后 50 | 覆盖内完成率/时延 | 覆盖外完成率/时延 | 子任务 UE/自己/邻居 | 第 200 时隙 5 架 UAV 位置 /m |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|
| 0001 | 0.0068, 0.0026 | 1.0068, 1.0026 | (-0.0465, 0.0119) | 0.0401 | 0.00% | 6.70% | 41.00% | 318.27 / 372.23 | 52.78% / 45.65% | 85.65% / 0.6839 s | 66.22% / 0.9688 s | 37.61% / 37.75% / 24.65% | (544.9,188.5); (424.7,512.3); (598.4,48.6); (549.4,139.1); (102.3,289.9) |
| 0010 | 0.0324, 0.0102 | 1.0330, 1.0103 | (-0.0693, -0.1140) | 0.1995 | 0.00% | 15.60% | 50.10% | 321.15 / 355.16 | 45.50% / 39.85% | 85.67% / 0.6544 s | 66.71% / 0.9645 s | 40.00% / 42.17% / 17.82% | (443.5,35.3); (268.3,520.7); (528.7,27.7); (397.7,39.5); (282.4,397.4) |
| 0020 | 0.0853, 0.0462 | 1.0891, 1.0473 | (0.1938, -0.3137) | 0.2677 | 0.00% | 28.60% | 69.60% | 295.93 / 160.23 | 43.58% / 17.94% | 82.30% / 0.6916 s | 66.90% / 0.9600 s | 40.63% / 39.47% / 19.89% | (534.9,5.7); (542.3,35.1); (566.6,4.6); (590.3,5.4); (290.6,34.4) |
| 0050 | 0.1943, 0.2025 | 1.2144, 1.2244 | (0.6518, -0.6226) | 0.6494 | 0.05% | 57.20% | 83.60% | 278.54 / 14.42 | 35.69% / 2.15% | 89.70% / 0.5443 s | 65.87% / 0.9784 s | 32.28% / 33.13% / 34.59% | (582.3,11.9); (585.9,8.5); (597.1,6.7); (597.8,14.8); (598.7,5.1) |

### 诊断结论与边界

覆盖下降的直接行为原因已经坐实：飞行头逐步学出所有 UAV 共享的向右、向下偏置，把 UAV 推到地图右边界和下边界，并在训练采样口径下进一步聚成一团。

- 确定性 evaluation 的平均飞行均值由 checkpoint 1 的 `(-0.0048, 0.0113)` 变为 checkpoint 50 的 `(0.7176, -0.5459)`；离边界不足 30 m 的比例由 24.50% 升至 94.80%，越界留在原地由 19.20% 升至 93.50%。终点中 4 架 UAV 的 x 已在 594–600 m，另 1 架的 y 约为 2 m。
- 训练采样口径复现同一方向偏置。checkpoint 50 后 50 时隙的两两平均距离只有 14.42 m，5 架 UAV 最终都位于右下角附近；后 50 时隙覆盖占比只有 2.15%，而 checkpoint 1 为 45.65%。
- 飞行均值分量绝对值大于 1 的比例始终几乎为 0，因此覆盖下降不是“均值先超过动作范围、再被 `np.clip` 截断”造成的。问题是方向一致的中等幅度动作长期累积，抵达边界后又被环境的越界保持原位规则困住。`log_std` 对应的标准差还从约 1.00 增至约 1.22，并没有出现探索方差塌缩；较大的采样噪声会增加边界附近的越界，但确定性回合已经单独复现覆盖下降，故它不是必要原因。
- 覆盖内 DAG 的策略质量并未同步恶化：同一个 evaluation 回合中，覆盖内完成率由 75.15% 升到 90.10%，平均时延由 0.8077 s 降到 0.5452 s；覆盖外完成率和时延基本停在约 65% 和 0.99 s。覆盖归属在 placement 之前已经由 UAV 位置决定，所以卸载动作不能解释当前时隙覆盖减少。结果更像是 placement 在越来越小的覆盖子集上变好，但飞行几何退化使大多数 DAG 失去覆盖。

这项重放只能识别覆盖下降的近因，不能单独回答 PPO 为什么把共同的东南方向偏置学成高概率动作；奖励权重、特征对称性或优化过程中的哪一项导致该偏置，需要另行设计隔离实验。本节也不把单 episode 的绝对覆盖率与前文 10 个评估 episode 的 50.20%/33.64% 直接等同；这里只验证相同 episode 跨 checkpoint 的行为变化。

### 产物

- 原始逐时隙 JSONL，共 1600 行：本地 `D:\CodeFile\HyperUAV-qly\output\mappo-diagnosis\slot_records.jsonl`；服务器 `/data2/zrj2025/hyperjet-baselines-20261106/output/mappo-diagnosis/slot_records.jsonl`。
- 机器可读汇总：本地 `D:\CodeFile\HyperUAV-qly\output\mappo-diagnosis\summary.json`；服务器同目录 `summary.json`。
- 完整终端表：本地 `D:\CodeFile\HyperUAV-qly\output\mappo-diagnosis\diagnose.log`；服务器同目录 `diagnose.log`。
- 轨迹图：`trajectory_checkpoint_0001_evaluation.png`、`trajectory_checkpoint_0010_evaluation.png`、`trajectory_checkpoint_0020_evaluation.png`、`trajectory_checkpoint_0050_evaluation.png`，本地和服务器均位于各自的 `output/mappo-diagnosis/`。
- `output/diag/` 已从本地仓库和服务器执行副本删除。未 commit，`output/` 未暂存。

### 热点叠图与更新次数补充（2026-11-08）

按用户选择，在四张 evaluation 轨迹图中加入本 episode 的三个热点：橙色半透明圆为半径 90 m 的热点范围，橙色“+”及 H0–H2 为圆心；灰点为 slot 0 UE，黑色“+”为 slot 199 UE；空心圆和黑色 X 分别表示 UAV 起点和终点。脚本同时把每个重放回合的热点圆心与半径写入 `summary.json`。服务器重新运行 8 个重放 episode 用时 44 秒，原指标保持不变。

seed=1、episode=0 的三个固定热点为 H0 `(383.59, 163.22)`、H1 `(130.68, 431.28)`、H2 `(385.62, 457.66)`。slot 0 每个热点内各有 7 个 UE，共 21/60；slot 199 分别有 4、11、10 个，共 25/60。热点内单位面积初始人数密度是热点外的 2 倍，但每个半径 90 m 的圆初始只有 7 个 UE，因此不画圆时视觉聚集并不明显；热点只影响初始撒点权重、热点内移动速度倍率和到达概率倍率，不会吸引 UE 向圆心移动。

热点布局在每次 `Simulator.reset(seed, episode)` 时重新生成，在一个 200-slot episode 内保持不变。50 次更新的原始 `metrics.jsonl` 记录了 1600 个训练 episode 和 50 个 evaluation episode，因此：

- 仅 PPO rollout：1600 次热点布局生成，即 4800 个热点圆；
- 训练命令实际执行的 rollout 加 evaluation：1650 次热点布局生成，即 4950 个热点圆；
- 不是“每次 PPO update 更新一次热点”：每个 update 的 32 个训练 episode 各有一套独立热点；第 10、20、30、40、50 次更新还各运行 10 套 evaluation 热点。

checkpoint 50 的 evaluation 中，UAV 2 初始位置为 `(599.726, 258.961)`，距右边界约 0.274 m。slot 0 的确定性动作 `(0.750, -0.817)` 已使候选 x 越过 600 m；环境对越界动作保持整个二维位置不变，而不是把 x 裁到边界。其后 200/200 个时隙的 x 动作均向右，全部被判越界，因此 UAV 2 的移动次数为 0，起点与终点完全相同。图中的绿色轨迹退化成一个点正是这一行为。

更新后的四张图仍位于：

- `D:\CodeFile\HyperUAV-qly\output\mappo-diagnosis\trajectory_checkpoint_0001_evaluation.png`
- `D:\CodeFile\HyperUAV-qly\output\mappo-diagnosis\trajectory_checkpoint_0010_evaluation.png`
- `D:\CodeFile\HyperUAV-qly\output\mappo-diagnosis\trajectory_checkpoint_0020_evaluation.png`
- `D:\CodeFile\HyperUAV-qly\output\mappo-diagnosis\trajectory_checkpoint_0050_evaluation.png`

服务器对应目录为 `/data2/zrj2025/hyperjet-baselines-20261106/output/mappo-diagnosis/`。未重新训练，未修改训练代码，未 commit。
