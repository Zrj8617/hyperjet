# HyperUAV Gate 2 实施计划

**目标：** 落实用户批准的 Gate 1 精简，新增四随机流、Region/UE/UAV 和静态 DAG 生成；演示后停止等待实体评审。

**依据：** 2026-10-14 用户审核意见与需求 §0.4。本轮授权已包含实施，不重复设计审批。

**结构：** random_streams.py 只派生四条 Generator；entities.py 只保存物理数据，通过方法参数接收配置和 RNG；dag.py 只保存静态 DAG 数据并生成结构和属性。

**技术：** Python 3.12、NumPy、dataclasses、tomllib。

## 步骤

- [x] 精简 contracts.py/types.py，六个 TOML 的数值注释加“（暂定）”。
- [x] RandomStreams(seed, episode)：SeedSequence([seed, episode]).spawn(4)，依次为 scene、mobility、arrival、dag。
- [x] Region.create(config, rng)：按旧环境的 [radius, side-radius] 采样中心；get_region_ids 用水平距离 ≤ radius。
- [x] UE.create(config, region_config, rng)：地面均匀位置、[0, 2×mean_speed] 初速、[0, 2π] 初方向；move(config, region_config, rng) 使用旧高斯—马尔可夫递推，去掉速度上限和等待降速，保留镜像反弹。
- [x] UAV.create(config, region_config, rng)：均匀水平位置与固定高度；apply_flight_actions(actions, config, region_config) 更新位置并返回越界标志。propulsion_power(speed, config) 实现指定公式；flight_energy(actions, out_of_bounds, config, scheduling_config) 由实际执行速度计算每架 UAV 的焦耳值，不存能耗状态。
- [x] DAG dataclass 和 generate_dag(dag_id, ue_id, arrival_time_s, config, rng)：沿用旧分层、选父、输入和计算量分布；边数据逐边均匀采样，sink 独立采样回传量，非 sink 为零。
- [x] 用固定 (seed, episode) 加载配置、创建实体、UE/UAV 各移动一步、打印飞行能耗并生成 3 个 DAG；保存命令和完整输出。无测试文件。
- [x] 检查源码语法、diff 空白及最终状态，报告文件行数，停止在 Gate 2。

## 边界

无运行状态、到达过程、通信、计算能耗、simulator、快照或 reward。不增加内部断言、输入检查、fallback、Protocol、兼容层或 hash。物理规则中的速度下界、越界判定、反弹、拓扑父节点范围属于所要求的运算本身。

不修改旧环境、config.py、算法、训练脚本和无关 dirty 文件，不提交、不更新 charter。沿用本会话已读取的 brainstorming / writing-plans 工作流，用户本轮批准范围优先。
