# HyperUAV Clean Server Runbook

本文档固定 HyperUAV clean mainline 在服务器上的验证顺序。完整实验不能直接开始；必须先通过第一阶段 torch/model smoke、第二阶段最小训练 smoke、第三阶段短训练 sanity run。

权威算法设计仍以 `docs/hyperuav_clean_mainline_design.md` 为准。本文档只说明服务器验证流程，不定义新算法口径。

## 0. 环境检查

进入项目目录并确认分支：

```bash
cd /data2/zrj2025/HyperUAV
git fetch origin
git checkout zrj_3
git reset --hard origin/zrj_3
git status --short
git log -2 --oneline
```

检查 Python 和 torch：

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果 `torch.cuda.is_available()` 为 `False`，仍可先跑 CPU smoke，但不能直接进入正式 GPU 实验。

## 第一阶段：torch/model smoke

先运行：

```bash
python scripts/smoke_clean_server_torch.py
```

该 smoke 检查：

- `import torch`
- `torch.__version__`
- `torch.cuda.is_available()`
- selected device
- HGNN forward
- movement actor forward
- offloading scorer forward
- centralized critic forward
- PPO loss shape
- tiny backward
- optimizer.step
- checkpoint save/load
- NaN/Inf
- tensor device
- mask dtype

通过标准：

- 不因为 torch 缺失 skip；
- 输出 torch version、CUDA 状态和 selected device；
- HGNN / movement actor / offloading scorer / critic / PPO update 全部通过；
- 无 device mismatch、mask dtype、shape error、NaN/Inf；
- checkpoint safe-boundary 保存和恢复通过。

## 第二阶段：最小训练 smoke

运行固定命令：

```bash
python scripts/train_clean_mainline.py --smoke --episodes 3 --max-steps-per-episode 20 --rollout-horizon 5 --run-name smoke
```

目标不是让 reward 变好，只检查：

- rollout 收集；
- slot-level GAE；
- PPO update；
- optimizer.step；
- checkpoint；
- JSONL logging；
- metrics 输出；
- invalid assignment rate 是否为 0 或有明确解释。

结果目录默认在：

```text
logs/clean_mainline/<timestamp>_smoke_seed<seed>/
```

目录应至少包含：

```text
train_metrics.jsonl
checkpoints/latest.pt
config.json
run_summary.json
plots/
```

## 第三阶段：短训练 sanity run

第一、第二阶段通过后，才运行短训练 sanity run。第一轮只跑 hypergraph 主方法，不跑 baseline/消融。

建议命令：

```bash
python scripts/train_clean_mainline.py --episodes 100 --max-steps-per-episode 200 --rollout-horizon 128 --seed 42 --run-name sanity_seed42
```

如果服务器时间紧，可以先用 50 episodes：

```bash
python scripts/train_clean_mainline.py --episodes 50 --max-steps-per-episode 200 --rollout-horizon 128 --seed 42 --run-name sanity50_seed42
```

观察项：

- reward components 数量级；
- movement actor 是否退化为全 hover；
- DAG completion rate 是否长期为 0；
- Energy per completed DAG 是否正常统计；
- ready/offloading action 数量是否正常；
- invalid assignment 是否为 0；
- loss 是否有限；
- checkpoint 是否正常生成。

## 何时可以启动完整实验

只有当以下条件全部满足，才可以启动完整实验：

1. 第一阶段 `python scripts/smoke_clean_server_torch.py` 通过；
2. 第二阶段最小训练 smoke 通过；
3. 第三阶段 50-100 episodes sanity run 没有 NaN/Inf、device mismatch、mask dtype、shape error；
4. `invalid_assignment_rate` 为 0，或有明确代码层原因和修复计划；
5. `train_metrics.jsonl`、`run_summary.json` 和 checkpoint 均正常；
6. DAG completion rate、offloading action count、movement action distribution 没有明显异常。

完整实验仍只使用 clean mainline 入口：

```bash
python scripts/train_clean_mainline.py ...
```

## 失败时回传信息

失败时请回传：

- 完整 traceback；
- 失败命令；
- stdout/stderr；
- `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"` 输出；
- `git log -2 --oneline` 输出；
- `git status --short` 输出；
- tensor shape、mask dtype、device 信息；
- 对应 run 目录中的 `run_summary.json`；
- `train_metrics.jsonl` 最后 20 行；
- 如果 checkpoint 相关失败，回传 `ls -lh logs/clean_mainline/<run_id>/checkpoints`。

建议辅助命令：

```bash
tail -n 20 logs/clean_mainline/<run_id>/train_metrics.jsonl
cat logs/clean_mainline/<run_id>/run_summary.json
```

## 常见失败类型

- torch 缺失：进入包含 torch 的 conda/env 后重试。
- CUDA 不可用：先跑 CPU smoke，正式训练前确认 GPU/driver/torch CUDA 匹配。
- device mismatch：回传 smoke 输出中的 selected device 和 traceback。
- mask dtype 错误：candidate mask / movement mask 必须是 bool。
- shape error：回传 task feature、incidence、movement logits、candidate feature shape。
- NaN/Inf：回传最新 `train_metrics.jsonl`、loss、reward components 和 traceback。
- checkpoint 失败：确认只在 rollout update 完成后或 episode 边界保存。

## 禁止入口

clean mainline 验证和训练禁止直接使用旧入口：

```text
main.py
train.py
scripts/tune.py
scripts/train_clean_assignment_mappo.py
marl_models/mappo/clean_mappo.py
marl_models/mappo/clean_assignment_policy.py
```

也不要启动 baseline、消融、plotting 或 T15 evaluation/drain。它们属于后续阶段。
