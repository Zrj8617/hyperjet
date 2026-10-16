# HyperUAV 环境重构 Gate 1 结果

日期：2026-10-14。实施和配置加载验证完成，**停在 Gate 1，等待用户和师兄审核**。未进入 Gate 2，未运行仿真或训练；不据此声明物理行为已经验证。

## 改动文件

本次全部为新增文件：

- environment_v2/__init__.py
- environment_v2/types.py
- environment_v2/contracts.py
- environment_v2/settings/__init__.py
- environment_v2/settings/region_config.py
- environment_v2/settings/user_config.py
- environment_v2/settings/uav_config.py
- environment_v2/settings/dag_config.py
- environment_v2/settings/channel_config.py
- environment_v2/settings/scheduling_config.py
- config/region.toml
- config/user.toml
- config/uav.toml
- config/dag.toml
- config/channel.toml
- config/scheduling.toml
- docs/superpowers/plans/2026-10-14-hyperuav-env-gate1.md
- 本报告
- docs/superpowers/reports/2026-10-14-hyperuav-env-gate1-result.json

没有修改旧 environment/、config.py、训练/算法代码、已有文档、output/ 或 tools/；未提交、推送、清理工作树或更新 charter。

## 设计说明

types 中 UEId/UAVId/DAGId/NodeId 都是整数别名，不包装实体。TaskKey 就是 (dag_id, node_id)。TaskStatus 和 DAGStatus 精确采用 §0.4 的六种和三种状态；这里只声明枚举，没有状态转换。

FlightActions 是 (N, 2) 的数组契约，Placement 是 ((dag_id, node_id), uav_id)，Placements 是列表。contract 注释明确保持 ready 子序列顺序，以及动作接收入口的合法性边界；本 Gate 不实现接收或检查动作。

SlotSnapshot 保存数组和记录的借用引用，frozen 仅阻止重绑字段，**不提供深度不可变保证**。调用方承诺不写数组、映射或记录，也不能拿它当历史快照。DAGView、TaskView、DAGStateView 三个 Protocol 仅描述跨层读取字段，不构造实体、不生成副本、不执行校验，也没有 runtime 状态写方法。未来实体/runtime 可直接满足这些接口，不必继承；这样 contracts 不反向导入尚未实现的上层模块。队列和空闲时刻仍由未来 runtime 唯一持有。

DAGOffer 记录 DAG/UE/到达时刻；CompletedDAG 记录 DAG/到达/完成时刻；TimedOutDAG 记录 DAG/到达时刻。SlotResult 汇集这些事件、四类焦耳能耗、非法 placement 数和 done，不含 reward 或指标。tx_energy_j 合计上传与前驱传输；compute/tx/return 为当时隙新承诺服务能耗，flight 为当时隙实际飞行/悬停能耗。SlotTransition 约定 end_slot 返回 (结果, 下一快照)，done 时下一快照为 None。

六类配置各自负责地图、UE、UAV、DAG、信道、时隙，都是 frozen dataclass，字段全部为标量。from_toml 直接取同名段并 cls(**data)，不补默认值、不处理缺键多键。统一 load_settings(directory) 返回 Settings 具名元组，集中执行批准的四项关系检查；单类 from_toml 只读取，正式组合入口为 load_settings。没有全局 config 导入、随机数调用或配置隐式加载。

例如将来 UAV 只需接收 settings.uav，移动时读取 flight_duration_s；它不需要知道 PPO 参数，也不需要加载旧 config.py。这就是本项目中的配置注入。

## 暂定数值与已读源码来源

**所有 TOML 数值均为开发暂定值，不是最终论文参数；本次不标定、不调参。**

| 配置 | 数值与来源 |
|---|---|
| region | 500 m、150 m、倍率 2：config.py:14–27 的当前地图、热点和到达倍率；地图调大尚未执行。 |
| user | 60、1.2 m/s、0.85、0.3 m/s、0.30 rad、0.5 W、0.0145：config.py:17、26、31、72、164–170；移动时长 5 s 暂取当前时隙长度，符合 §0.4.2 的默认选择。取消旧上限和等待降速所需行为仍留给实体阶段。 |
| uav 基础 | 5 架、100 m、15 m/s、1e9 cycles/s、0.5 W、队列 16：config.py:16、73–76、83、393；environment/task_execution.py:280 实际使用该计算速率。不是 charter 历史段落中的 1e6。 |
| uav 新参数 | flight_duration_s=0.5、kappa=1e-28、tip_speed=110：读取 D:/CodeFile/mrdts/config/uav.toml 的 member 参数后暂借用。它们未针对 HyperUAV 标定；kappa 与旧 50 W 常功率计算模型并不等价。 |
| uav 推进参数 | P0=80 W、Pi=80 W、v0=4 m/s、d0=0.6、rho=1.225 kg/m³、s=0.05、A=0.5 m²：本轮为配置可加载而选择的占位数值，**假设，未坐实**；不声称来自 Zeng 论文表格或 MRDTS。没有沿用 u1–u5，也没有实现推进公式。 |
| dag 拓扑 | 节点 5–8、最大层数 4、最大父节点数 3：config.py:36–39。 |
| dag 数据 | 输入 0.1–1.0、输出 0.15–0.75 取自 config.py:49–50；统一乘 1024²×8 得 bit：838860.8–8388608、1258291.2–6291456。后者同时用于 edge 和 return 的取值范围；尚未实现生成器。 |
| dag 计算量 | 基准 10×1024 byte 转为 81920 bit；常数 500000–1500000，复杂度概率 0.2/0.7/0.1，来自 config.py:40、51–56 和 environment/dag_tasks.py:665–678。以后生成器仍算 n=input_bits/base_unit_bits 和原复杂度函数，CPU cycles 不重新缩放。 |
| channel | 三类带宽 4/6/4 MHz 转为 4e6/6e6/4e6 Hz；gain=1e-6、NLoS=0.2、noise=1e-13 W、alpha=9.61、beta=0.16：D:/CodeFile/mrdts/config/channel.toml，公式读取 env/communication/channel_model.py 核对；本轮只配置，不实现通信。 |
| scheduling | 时隙 5 s 沿用 config.py:19；到达窗口暂取旧 EPISODE_LENGTH=500，但新 episode 还包含收尾，因此不声称长度等价。D=40 为本轮占位选择，**假设，未坐实**，不是旧 deadline 字段的迁移，也不是实验结论。 |

单位注意：旧 environment/dag_tasks.py 的计算量使用 MB×1024² 得 byte，但旧 environment/comm_model.py 的传输时间用 MB×8/Mbps，两个口径并不一致。本轮选择 bit 后以原计算量的二进制换算为准，使 n 与 CPU cycles 不变；不宣称保持旧通信时长。新的通信公式和参数本来就按 §0.4 替换。

## 验证

仓库根执行：

```powershell
python -B -c "from environment_v2.settings import load_settings; print(*load_settings('config'), sep='\n')"
```

Python 3.12.9 成功打印六个配置，完整输出存于同目录 result.json 的 config_load_stdout。关键组合为 slot=5、flight=0.5、D=40、max_levels=4，四项关系通过。

- 10 个新增 Python 文件在 Python 3.12 下完成 AST 语法解析。
- contracts/types 在现有 Python 3.10 + NumPy 1.26.4 下导入成功。本机默认 Python 3.12 没有 NumPy，所以未完成 **Python 3.12 + NumPy** 的 contract 联合导入验证；没有为此改依赖、安装包或写 fallback。
- git diff --check 通过；另外对全部本次未跟踪文件执行 git diff --no-index --check，避免普通 diff 漏掉新文件。命令使用临时 -c core.safecrlf=false 抑制 CRLF 转换提示，不修改 Git 配置。
- 新代码仅 load_settings 有四处 ValueError；无内部 assert、hash、自定义指纹、fallback、兼容层或阶段对象；没有新增 contract tests。

## Git 状态与保护边界

开始与完成分支：param-realign-20260914。

HEAD：670ba0dfd549d8b28606a3b85df4ad04fbffbf39（未改变）。

初始 8 项均保留：

```text
 M docs/superpowers/specs/2026-09-23-hyperuav-chapter3-word-design.md
 M docs/superpowers/specs/2026-10-10-hyperuav-env-rebuild-requirements-draft.md
?? docs/superpowers/2026-10-13-hyperuav-env-rebuild-context-brief.md
?? docs/superpowers/plans/2026-09-23-hyperuav-chapter3-word-implementation.md
?? docs/superpowers/plans/2026-09-28-pdf-code-consistency-audit.md
?? docs/superpowers/reports/2026-09-28-pdf-code-consistency-audit.md
?? output/
?? tools/
```

本次额外状态为 config/、environment_v2/、本次计划、报告、result.json，全部 untracked。完整完成状态见 result.json。未枚举或读取 output/、tools/ 内容。

## Gate 2 前的审核

请用户与师兄审核字段命名、snapshot 只读借用约定、三个记录接口，以及数据量换算选择。飞行时长、推进参数、kappa、D 等继续标为暂定，最终数值按原计划后定。没有重新打开 Gate 0 阻塞项。

本轮交付只说明 Gate 1 已实施并完成上述验证，**阶段验收及进入 Gate 2 仍等待用户和师兄审核、用户明确批准**。
