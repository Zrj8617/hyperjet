# mini-spec: shaped-reward 小训练探针（2026-09-05, v3 · 修正 Φ 与终止处理）

**状态：** APPROVED（用户 + 师兄 + Codex 复核）· 已放行,可实跑　**执行：** Codex　**类型：** 训练探针（首次触及训练 + reward）

## 核心（4 行）
- **目标**：测 audit 测不到的 Question B —— 训练时加 potential-based 进度塑形,**critic 能否真正学起来、offloading actor entropy 能否稳**,对比 baseline。
- **做什么**：新 gate `USE_DAG_PROGRESS_POTENTIAL_SHAPING`(默认 OFF=严格复现现基线);ON 时训练 reward 每步加 `r' = r + γ·Φ(s') − Φ(s)`。其余全不动。
- **看什么数**：两个 primary gate + de-shaped EV 解释性校验（见下)。
- **通过**：两个 EV 都改善 + entropy 更稳 + 真实系统指标不恶化 → 才提交正式 shaped training 批准;否则停止,转 state / estimand。

## Φ 定义（不掉落版,替代旧 active-only）
- `Φ(s) = REWARD_COMPLETED_DAG_WEIGHT × [ 已完成 DAG 数 + Σ_active DAG(该 DAG 已完成子任务 / 总子任务) ]`
- DAG 完成时由 active fraction(→1)转成 completed count(+1),potential **连续不掉**,不会抵消 +8 完成信号。
- `Φ` 只依赖当前环境状态,禁用未来信息 / branch oracle。reset 时无 DAG → `Φ(s_0)=0`。
- ⚠️ **注意**:这与 2026-09-04 诊断用的 active-only Φ **不同**(那个会掉);本训练探针以本节的不掉落公式为准。

## 终止处理（关键,避免 spike）
- EPISODE_LENGTH 末端本质是 time-limit。**不掉落 Φ 会随 episode 累加,若在末端强制 `Φ(s')=0`,会注入 ≈ −Φ(s_{T-1}) 的巨大负 spike → 禁止。**
- 正确做法:**保持基线原有终止/bootstrap 逻辑不动**;在终止(done)那一步把 **shaping 增量 F 置 0**(不是强制 Φ=0)。因 `Φ(s_0)=0`,整段 discounted return 只差 `γ^{T-1}·Φ(s_{T-1}) ≈ 1e-9` → 目标守恒,shaping 只在 episode 内密集重分配奖励,无 spike。
- 真正的 mid-episode truncation(若有)按训练器现有方式正常 bootstrap,保留正常 `Φ(s')`。

## 正确性条件
- γ 与训练实际 discount **完全一致**(用 run 的 γ)。
- **单变量**:不同时调 LR / GAE / entropy 系数 / 网络 / state;gate-OFF 时 reward、trajectory、RNG、loss、参数更新**逐值复现基线**。
- RNG-neutral:塑形不抽 RNG;ON/OFF 共享 seed 序列,唯一差异是 reward 项。

## 实验设计（配对 short gate）
- **6 runs**:control(OFF) 与 shaping(ON) 各 `100 episodes × 3 seeds`,相同 seed / 初始化 / 训练预算。

## 判据
- **Primary gate 1 — Critic EV**:shaped 组是否从 ~0 持续爬升,并在多数 seed 后半程保持为正。
- **Primary gate 2 — Offloading entropy**:是否避免早期 collapse / 剧烈震荡,又不长期完全均匀。
- **de-shaped EV 解释性校验**:同时记录 shaped-target EV 与 de-shaped EV(`V_original = V_shaped + Φ`,与本 spec 的终止约定一致计算)。用于排除"critic 只学会了易预测的 Φ 偏置、EV 假性变好、原始长期价值仍没学会"。**只 shaped-EV 好不算 PASS。**
- **Guard metrics**(否决/护栏,不作 PASS 依据):系统指标(completion / flowtime / throughput)、reward、参数 / RNG 门禁。不能仅凭 entropy 下降判 PASS;系统指标恶化则否决。
- 3 seeds 仅方向探针,不做正式显著性结论。

## 工程门禁
freeze commit;记录版本(HEAD + dirty + 脚本 hash);服务器后台跑,返回 PID / log / result path;不做 1000-episode 正式训练。

## 结果产物
- report：`docs/superpowers/reports/2026-09-05-shaped-reward-training-probe-results.md`
- JSON + 曲线:shaped-EV 与 de-shaped-EV vs update、offloading entropy vs update、系统指标(ON vs OFF)。

## 不做
不改环境目标(PBRS 保证)、不接入 Phase 4A credit 分支、不动 state / PPO 结构、不启动正式长训练。**进入本探针 ≠ 批准正式 shaped training。**
