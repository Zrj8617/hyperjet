# SPEC: <date>-<name>

**状态：** DRAFT → (用户批准后) FROZEN
**作者：** Claude　**批准人：** 用户　**执行：** Codex

## 1. 目标（一句话）
<这次实验/改动要回答的唯一问题>

## 2. 单一变量
<只改这一个东西；其它一律不动>

## 3. 精确改动
- 文件：`<path>`（改什么，最小 diff）
- gate/开关：默认 <ON/OFF>；**gate OFF 必须严格复现旧行为**

## 4. 样本 / 设定
- 决策数 / roots / seeds / checkpoints / horizons：<...>
- RNG：诊断/采样 RNG 独立，不污染 environment/training RNG

## 5. 指标
- <Top-1 / Spearman / pairwise / mean±CI / EV / ...>

## 6. 停止 / 通过条件（量化）
- 通过：<...>　失败：<...>　→ 失败则 <下一步 / 回退>

## 7. 结果产物路径（约定，Codex 必须写到这里）
- report：`docs/superpowers/reports/<date>-<name>-results.md`
- 机器可读：`<result.json 路径>`
- 原始 log/JSON：`<路径>`

## 8. 不做
<明确排除的东西，防止 scope 漂移>
