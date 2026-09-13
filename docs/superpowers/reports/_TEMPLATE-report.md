# REPORT: <date>-<name>

**对应 spec：** `docs/superpowers/specs/<date>-<name>-design.md`
**裁决：** PASS / FAIL / PARTIAL（由 Claude 复核后填）

## 1. 版本现实（AGENTS.md §3，必填）
- `git rev-parse HEAD`：<...>
- `git status --porcelain`：<dirty 行数 + 关键文件，或 "clean">
- 诊断/训练脚本版本：<`git hash-object <script>` 或 commit+dirty>
- 运行环境：<本地 / 服务器路径>；如后台：PID / log path

## 2. 结果（原始数字）
<指标值；不解释，先给数>

## 3. 结论边界
<这些数字支持什么、不支持什么；样本量/置信区间>

## 4. 原始产物路径
- result.json：<...>　log：<...>　checkpoint：<...>

## 5. Codex 自检
- gate OFF 复现旧行为：<是/否 + 证据>
- 单变量：<是/否>　RNG-neutral：<是/否>　参数未变：<是/否>
