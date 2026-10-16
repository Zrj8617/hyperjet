# HyperUAV Gate 1 Implementation Plan

**Goal:** 只建立公共类型、配置和 contract，配置加载验证后停止。

**Architecture:** types 定义标识和状态；contracts 定义动作、只读数据接口和物理结果；settings 从六个 TOML 段直接构造六个 frozen dataclass。只读接口是实体/runtime 与调用方的边界，不持有或修改状态。

**Tech Stack:** Python 3.12、tomllib、dataclasses、NumPy 类型标注。

**Spec:** `docs/superpowers/specs/2026-10-10-hyperuav-env-rebuild-requirements-draft.md` §0.4、Gate 1；本轮用户明确授权直接实施。

## 约束与审核重点

- 只新增 environment_v2、六个 config/*.toml 和本次计划/报告；不提交、不清理、不更新 charter。
- 不实现实体、通信、runtime 或 simulator；不写 contract tests。
- 只检查时隙 > 0、飞行时长 > 0、飞行时长 ≤ 时隙、D > 最大层数。
- snapshot 引用只读使用的数据，不能变成第二个状态所有者；结果不含 reward。
- 数据使用 bit、计算量使用 CPU cycles；暂定值与来源在报告中列出。

## 执行步骤

- [x] 新增 types.py：整数 ID、二元组 TaskKey、两组状态枚举。
- [x] 新增 contracts.py：FlightActions、Placement(s)、DAG/运行记录只读接口、SlotSnapshot、offer/完成/超时记录、SlotResult。接口只描述字段，不实现实体或状态转换。
- [x] 新增 settings/{region,user,uav,dag,channel,scheduling}_config.py，各自 from_toml(path) 使用 `cls(**tomllib.load(file)[段名])`；__init__.py 的 load_settings(directory) 返回六配置的具名元组，并在读取入口执行四项关系检查。
- [x] 新增六个 TOML，每键中文含义及单位；保持配置字段为标量，避免 frozen dataclass 内仍可变的范围列表。
- [x] 运行 `python -B -c "from environment_v2.settings import load_settings; print(*load_settings('config'), sep='\n')"`；导入 contracts/types，检查新增文件语法、git diff --check 和最终状态。
- [x] 写 Gate 1 结果报告与 result.json，记录来源、输出和审核边界，停止等待用户与师兄审核。

执行方式：本会话直接实施；不创建新工作树、不委派、不重复设计审批。遵从用户要求，验证仅覆盖本 Gate，不提前写行为测试。
