# HyperUAV 窗口交接（2026-08-31）

## 1. 当前状态

HyperUAV 已完成以下研究阶段：

- R0：冻结 Scheme-B2 semantic common-random 机制。
- R1-A：Environment / Load Feedback Audit。
- R1-A2：strict semantic CRN workload-feedback control。
- R1-B：Reward component / timing audit。
- R2：Reward-Ablation MLP Learnability Bundle。
- Shared-slot GAE 与 Decision-GAE-v1 的 3-seed、120-update 配对实验。
- Decision-GAE-v2 的 3-seed、120-update 实验。
- Decision-Q 的 3-seed、120-update 实验。

上述实验均已完成，不要自动重复运行，也不要自动进入下一阶段。

## 2. Roadmap

Canonical roadmap：

`docs/research/HyperUAV_research_master_roadmap.md`

不要未经明确要求修改 roadmap。

## 3. 当前最新源码

服务器：`10.12.54.24`

当前最新源码仓库：

`/data2/zrj2025/HyperUAV-bridge-eft-ablation-8070bc4`

当前版本：

- Branch：`codex/offloading-decision-q-20260831`
- HEAD：`6eaa0de2a81ffad928d3414205380b94dd4b7565`
- Commit：`Add environment-return offloading Decision-Q credit`
- 上次检查时工作树干净。

最近研究版本的演进：

- R2 基线：`8070bc4fdd12ee4a513216bcd1bed3a87f6e2bed`
- Decision-GAE-v2：`a7d6bb0645b4c37b7b1e6eacac561ef369f39429`
- Decision-Q：`6eaa0de2a81ffad928d3414205380b94dd4b7565`

`/data2/zrj2025/HyperUAV-r2-formal-8070bc4` 是较早的 R2 正式目录和永久 TensorBoard 日志根目录，不是最新 Decision-Q 源码仓库。

普通的 `/data2/zrj2025/HyperUAV` 是更早的历史仓库，最近实验不是从该目录启动的。

## 4. 主要实验目录

Shared-GAE 与 Decision-GAE-v1：

`/data2/zrj2025/HyperUAV-decision-gae-paired-120u-20260830`

Decision-GAE-v2：

`/data2/zrj2025/HyperUAV-decision-gae-v2-3seed-120u-a7d6bb0-20260830`

Decision-Q：

`/data2/zrj2025/HyperUAV-decision-q-3seed-120u-6eaa0de-20260831`

R2 正式实验及汇总：

`/data2/zrj2025/HyperUAV-r2-formal-8070bc4`

Decision-Q 三个 run 均已完成，每个 `train_metrics.jsonl` 包含 120 updates。目前没有 `train_clean_mainline` 训练进程运行。

## 5. R2 已冻结事实

R2 offline sweep 已完成，不要重做：

- `low_cancel` completed-DAG weight：`2.0`
- `energy_balanced` energy weight：`0.50`

R2 正式 bundle：

- 4 arms × 3 seeds。
- 4000 updates/run。
- 12 个 training runs。
- 12 checkpoints × 3 eval seeds，共 36 个 strict-CRN evaluations。
- 已完成。

## 6. Decision-GAE 与 Decision-Q

Decision-GAE 将 offloading credit 从 shared slot GAE 改成基于真实 environment return 的 decision-level SMDP GAE，同时保持 movement-position advantage 原路径不变。

Decision-GAE-v2 仅稳定 decision value critic：采用 frozen rollout targets、normalized-space critic loss/value clipping、独立 optimizer 和独立 gradient clipping。

Decision-Q 的主 gate：

`--offloading-decision-q-credit`

Decision-Q 使用真实 environment return 训练 action-conditioned `Q(s,a)`。其 actor credit 为：

`A_Q(s,a_selected) = Q_old(s,a_selected) - sum_a pi_old(a|s) Q_old(s,a)`

Q bootstrap 为：

`V_old(s_next) = sum_a pi_old(a|s_next) Q_old(s_next,a)`

target 为：

`y_d = rho_d + gamma^Delta * V_old(s_next)`

terminal 时 bootstrap 为 0；rollout-boundary unresolved 样本不训练。Actor advantage 在 PPO update 前冻结、standardize、detach，不叠加 slot GAE、EFT 或其他 credit。

Decision-Q 不使用 EFT、auxiliary teacher、counterfactual、lagged-Q、branching oracle 或旧 decision critic 训练逻辑。

## 7. 最近对比实验共同配置

- Task encoder：MLP。
- Seeds：0、1、2。
- Episodes：200。
- Max steps/episode：200。
- Rollout horizon：128。
- Num envs：1。
- Sampler：synchronous。
- PPO epochs：3。
- LR：`3e-4`。
- Gamma：`0.99`。
- GAE lambda：`0.95`。
- Clip ratio：`0.2`。
- Entropy coefficient：`0.01`。
- Value coefficient：`0.5`。
- Value clip epsilon：`0.2`。
- Normalize value targets：ON。
- Max grad norm：`0.5`。
- Offloading LR scale：10。
- Movement LR scale：5。
- Movement-position advantage：ON。
- Detach critic HGNN：ON。
- Completed-DAG reward weight：8。
- Movement energy penalty：`0.0045`。
- Max updates：120。
- Checkpoints：0、30、60、90、120。

Movement 路径是控制变量，不能因为 offloading credit 改动而变化。

## 8. TensorBoard

永久访问地址：

`http://127.0.0.1:16060/`

服务器日志根目录：

`/data2/zrj2025/HyperUAV-r2-formal-8070bc4/logs/tensorboard/r2_reward_bundle_8070bc4`

Decision-Q 已接入为：

- `Decision-Q_seed0`
- `Decision-Q_seed1`
- `Decision-Q_seed2`

每组有 120 updates。奖励曲线另设 `00_reward/*` 标签组，包括 update reward、episode reward、rollout reward 和 reward components。

Decision-Q 关键诊断位于 `ppo/*`，包括 Q critic EV、normalized loss、preclip gradient、value clip fraction、raw target、TD error、legal Q spread、Q advantage、within-slot advantage、offloading PPO target gradient、entropy、logit/probability spread，以及 movement/global clipping diagnostics。

## 9. 后续分析原则

- 不要凭交接文档猜实验数值；必须读取 JSON、`train_metrics.jsonl`、evaluation JSON 或 TensorBoard 数据。
- 不要重复 R0、R1、R2、Shared-GAE、Decision-GAE-v1/v2 或 Decision-Q 已完成实验。
- 跨方法结论应使用 seeds 0/1/2 和 checkpoints 30/60/90/120 的 paired 比较。
- 不要只依据 entropy 判断学习。
- 至少同时检查 critic EV/TD error、advantage、actor PPO gradient、entropy、logit/probability spread、global clipping、movement coupling，以及 deterministic evaluation 的 flowtime/completion/energy。
- 不要把 heuristic 当 teacher。
- 实验只能在服务器运行；本地环境不能运行正式实验。
- 未经明确要求，不修改代码、不提交、不 push、不清理实验目录、不启动新实验。

## 10. 新窗口开始方式

新窗口应先：

1. 阅读本文档和 canonical roadmap。
2. 只读检查服务器最新源码 branch/HEAD/status。
3. 阅读相关实验目录中的配置、JSON、`train_metrics.jsonl` 和 evaluation 结果。
4. 简短汇报恢复出的当前研究状态。
5. 等待用户下一条具体研究指令，不自动执行后续实验。
