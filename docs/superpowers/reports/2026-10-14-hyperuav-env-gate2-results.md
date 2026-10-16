# HyperUAV Gate 1 修订与 Gate 2 实施结果

日期：2026-10-14。用户已批准 Gate 1 修订和 Gate 2 实施。**当前停在 Gate 2，等待师兄评审实体；没有进入后续 Gate。**

## 本轮范围

Gate 1：删除三个 Protocol、SlotSnapshot、SlotTransition 和四个整数 ID 别名；保留 TaskKey=tuple[int,int]、动作契约和结果记录；六份 TOML 每个数值的注释末尾加“（暂定）”，未改变任何数值。

Gate 2：
- random_streams.py：RandomStreams(seed, episode)，用 SeedSequence([seed, episode]).spawn(4) 依次派生 scene、mobility、arrival、dag。这里只创建到达 RNG，没有实现到达过程。
- entities.py：Region 只有 center/radius_m；UE 只有 positions/speeds/headings；UAV 只有 positions。配置和 RNG 作为方法参数传入，不在实体里复制配置、队列、能耗或运行状态。
- dag.py：DAG 的十个静态字段及 generate_dag()；没有 runtime 状态、前驱/后继缓存、sink 缓存、通信或计算能耗。

无新增 assert、输入检查、fallback、阶段对象、Protocol 或兼容层。UE 的速度下界、边界反弹，UAV 的越界判定，以及生成 DAG 时限制父节点选择范围，均为已批准的物理/生成规则，不是额外防御。没有测试文件。

## 与依据的对应关系

- 热点：沿用 environment/env.py 的 _sample_episode_hotspot，中心在 [radius, side-radius] 中均匀采样，整个圆在地图内；未搬旧参数检查。热点判定只用水平距离，含边界。
- UE：沿用 environment/user_equipments.py 的初速度 [0, 2×mean_speed]、初方向 [0,2π] 及速度/方向噪声递推；速度只截到不小于 0，位移乘 move_duration_s，不用等待降速。边界反弹同时改变运动方向；速度向量只作为本次计算局部变量。
- UAV：按 MRDTS env/entities/uav.py 的 apply_flight_actions 逐分量缩放，不把对角线动作再归一化。因此动作 (1,1) 的速度大小是 sqrt(2)×max_speed，和参考实现一致。越界整步不执行，返回每架 UAV 的 bool 标志。
- propulsion_power() 精确实现需求 §0.4 给出的 P(V)，返回 W；flight_energy() 接收同一次动作和越界标志，越界速度取 0，返回每架 UAV 的 J；不在 UAV 上存能耗。
- DAG：沿用 environment/dag_tasks.py:625–689 的节点数、层数、每层至少一个节点、额外节点随机分层、从所有更早层随机无放回选父、均匀输入量、闭区间常数及三类复杂度概率。边排序按子节点遍历、父编号升序，节点编号即拓扑序。
- 边数据按 §0.4 引用的 MRDTS generate_edge_features 逐边采样，但沿用 HyperUAV 的连续均匀数据分布和配置范围；sink 独立采样 return_bits，非 sink 为零。sink 由“无出边”判定，可能位于较早层，未强加所有末端必须在最后一层的规则。
- 计算量 n=input_bits/base_unit_bits，CPU cycles=n×log2(max(n,2))^k×constant，k 按 0/1/2 的原概率选择；未增加计算能耗。DAG 的 id、ue_id、arrival_time_s 由调用方明确传入，无到达逻辑。
- 用 Generator 和集合式抽样保持上述分布，不声称与旧全局 np.random 在同一 seed 下逐样本相等。

## 一条 PowerShell 命令

从仓库根执行下方完整 here-string 命令。固定 seed=42、episode=0，依次创建热点、UAV、UE，双方各移动一步，再生成 3 个 DAG；打印配置、实体前后数据和 DAG 全部字段。

本机默认 Python 3.12.9 没有 NumPy，本轮已将 NumPy 2.3.3 安装到系统临时目录 hyperuav-gate2-python312，仅此命令将该路径加入 sys.path；没有修改全局 Python、训练环境、requirements 或环境代码。这是本机演示依赖路径，不是产品 fallback。其他已安装 NumPy 的 Python 3.12 可省略临时路径相关三行。

```powershell
@'
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(tempfile.gettempdir()) / "hyperuav-gate2-python312"))
import numpy as np
from environment_v2.settings import load_settings
from environment_v2.random_streams import RandomStreams
from environment_v2.entities import Region, UE, UAV
from environment_v2.dag import generate_dag

np.set_printoptions(precision=6, linewidth=120)
cfg = load_settings("config")
print(*cfg, sep="\n")
streams = RandomStreams(seed=42, episode=0)
region = Region.create(cfg.region, streams.scene)
uav = UAV.create(cfg.uav, cfg.region, streams.scene)
ue = UE.create(cfg.user, cfg.region, streams.scene)
print("seed=42 episode=0")
print(region)
print("UAV before:", uav.positions)
print("UE before:", ue)
actions = np.tile([1.0, 1.0], (cfg.uav.count, 1))
actions[0] = [0.0, 0.0]
blocked = uav.apply_flight_actions(actions, cfg.uav, cfg.region)
ue.move(cfg.user, cfg.region, streams.mobility)
print("Actions:", actions)
print("UAV after:", uav.positions)
print("Out of bounds:", blocked)
print("Flight energy J:", uav.flight_energy(actions, blocked, cfg.uav, cfg.scheduling))
print("UE after:", ue)
print("UE region IDs:", region.get_region_ids(ue.positions))
for i in range(3):
    print(generate_dag(i, i, 0.0, cfg.dag, streams.dag))
'@ | python -B -
```

## 实测结果

上述命令退出码 0。**完整原始输出**在同目录 `2026-10-14-hyperuav-env-gate2-result.json` 的 `demo.output` 中，包含 60 个 UE 前后坐标、速度、方向以及三个 DAG 的全部字段，未截断。

摘要：

```text
seed=42 episode=0
Region center=[333.348832, 332.197334], radius=150
Out of bounds: [False False True False False]
Flight energy J: [800.000000 815.852194 800.000000 815.852194 815.852194]
DAG 0: num_tasks=8, edges=11, sinks=4, levels=[0,0,1,2,2,3,3,3]
DAG 1: num_tasks=8, edges=4, sinks=4, levels=[0,0,0,0,0,0,1,1]
DAG 2: num_tasks=7, edges=6, sinks=2, levels=[0,0,0,1,2,2,3]
```

UAV 0 主动悬停，UAV 2 越界后原位悬停，均为 P(0)=160 W × 5 s=800 J。所有参数仍暂定，这些数字只证明公式和调用路径跑通，不是物理标定结果。

额外一次性命令核对（未创建测试文件）：

```text
Reflection positions: [[496.0, 250.0, 0.0], [4.0, 250.0, 0.0]] headings: [3.141592653589793, 1.2246467991473532e-16]
Hover power W: 160.0
DAG stream independent of mobility: True
Fixed seed DAG replay: True
DAG node/edge/sink counts: 8 11 4
Syntax OK: 13 Python files
```

13 个 environment_v2 Python 文件在 Python 3.12.9 下语法解析成功。配置、契约、实体和 DAG 已在 Python 3.12.9 + NumPy 2.3.3 联合导入运行，消除了前轮仅能分开导入的验证限制。

## Git 与边界

分支：param-realign-20260914。
HEAD：670ba0dfd549d8b28606a3b85df4ad04fbffbf39；本轮前后相同。

git diff --check 通过；本轮改动仍未跟踪，另逐文件用 git diff --no-index --check 与 NUL 比较，无空白诊断（该命令隐含 exit-code，返回 1 表示有内容差异，不代表空白错误）。

原有 dirty 文件保留，旧 environment/、config.py、旧 MARL、训练脚本、output/、tools/ 均未改动。Gate 1 原报告作为当时交付记录保留，本报告说明本轮批准的删减，不改旧记录。没有提交、推送或更新 charter。

## 评审重点

请师兄评审实体字段与方法接口、UE 反弹与 UAV 越界动作语义、DAG 结构/属性生成。快照留到 Gate 5，运行状态留到 Gate 4，其他明确排除的内容均未实现。**Gate 2 实施结束，等待评审，不自动推进。**
