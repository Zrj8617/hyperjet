# HyperUAV Clean Full Experiment Launch Checklist

本文档只用于完整实验启动前检查和命令模板固化，不定义新算法，不启动 baseline/ablation，不替代 `docs/hyperuav_clean_mainline_design.md`。

## 启动前硬性三道门

完整训练实验只能在以下三道门全部通过后启动：

### Gate 1: torch/model smoke 通过

必须成功运行：

```bash
python scripts/smoke_clean_server_torch.py
```

要求无 torch 缺失、device mismatch、mask dtype、shape error、NaN/Inf；HGNN、movement actor、offloading scorer、critic、PPO tiny backward、optimizer.step、checkpoint save/load 全部通过。

### Gate 2: minimal training smoke 通过

必须成功运行：

```bash
python scripts/train_clean_mainline.py --smoke --episodes 3 --max-steps-per-episode 20 --rollout-horizon 5 --run-name smoke
```

只检查 rollout -> GAE -> PPO update -> checkpoint -> JSONL，不要求 reward 变好。

### Gate 3: short sanity run 通过

必须成功运行 T17 sanity helper，并确认 `sanity_report.json` 中：

```text
overall_pass = true
```

建议命令：

```bash
python scripts/run_clean_sanity.py --episodes 100 --max-steps-per-episode 200 --rollout-horizon 20 --seed 0 --run-name sanity_seed0
```

如果任一 gate 失败，不得启动完整实验。

## 启动前记录项

启动完整实验前必须记录：

```bash
git status --short
git log -2 --oneline
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

还必须确认：

- `git status --short` 为空；
- `config.json` 会写入每个 run 目录；
- run seed 写入命令和 config；
- output dir 独立；
- checkpoint/log/plot 目录可写；
- 只使用 clean mainline 入口；
- 不使用旧 `main.py`、`train.py`、`scripts/tune.py`、`scripts/train_clean_assignment_mappo.py`、`marl_models/mappo/clean_mappo.py`、`marl_models/mappo/clean_assignment_policy.py`。

## Full Training Command Template

第一版完整实验只允许 hypergraph 主方法：

```bash
python scripts/train_clean_mainline.py \
  --episodes <N> \
  --max-steps-per-episode <M> \
  --rollout-horizon <H> \
  --seed <S> \
  --run-name hypergraph_seed<S> \
  --output-dir logs/clean_mainline
```

示例：

```bash
python scripts/train_clean_mainline.py --episodes 500 --max-steps-per-episode 500 --rollout-horizon 128 --seed 0 --run-name hypergraph_seed0 --output-dir logs/clean_mainline
```

## Evaluation Command Template

```bash
python scripts/eval_clean_mainline.py \
  --checkpoint <checkpoint> \
  --episodes <EVAL_EPISODES> \
  --arrival-steps <ARRIVAL_STEPS> \
  --max-drain-steps <MAX_DRAIN_STEPS> \
  --seed <S> \
  --run-name hypergraph_seed<S>_eval
```

示例：

```bash
python scripts/eval_clean_mainline.py --checkpoint logs/clean_mainline/<run_id>/checkpoints/latest.pt --episodes 20 --arrival-steps 500 --max-drain-steps 500 --seed 0 --run-name hypergraph_seed0_eval
```

## Plotting Command Template

```bash
python scripts/plot_clean_metrics.py --run-dir <train_run_dir>
python scripts/plot_clean_metrics.py --run-dir <eval_run_dir>
```

## Launcher Dry-Run

可以先生成完整实验计划，不执行训练：

```bash
python scripts/launch_clean_experiments.py --seeds 0 1 2 --episodes 500 --max-steps-per-episode 500 --rollout-horizon 128 --eval-episodes 20 --arrival-steps 500 --max-drain-steps 500 --run-prefix hypergraph --dry-run
```

第一版 launcher 默认 dry-run。实际执行完整实验必须显式传 `--execute`，且仍需先确认三道门全部通过。

## 结果目录

训练：

```text
logs/clean_mainline/<timestamp>_hypergraph_seed<S>/
  train_metrics.jsonl
  config.json
  run_summary.json
  checkpoints/
  plots/
```

评估：

```text
logs/clean_eval/<timestamp>_hypergraph_seed<S>_eval/
  eval_metrics.jsonl
  eval_summary.json
  config.json
  plots/
```

实验计划：

```text
logs/clean_experiments/<timestamp>_<run-prefix>/
  experiment_plan.json
```

## 失败回传清单

失败时回传：

- command；
- stdout/stderr；
- traceback；
- `git log -2 --oneline`；
- `git status --short`；
- `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"` 输出；
- `config.json`；
- `run_summary.json`；
- `train_metrics.jsonl` 最后 20 行；
- `eval_summary.json`；
- `sanity_report.json`；
- NaN/Inf、shape、device、mask dtype 信息；
- checkpoint 目录列表。

辅助命令：

```bash
tail -n 20 <train_run_dir>/train_metrics.jsonl
cat <train_run_dir>/run_summary.json
cat <eval_run_dir>/eval_summary.json
cat <sanity_run_dir>/sanity_report.json
```

## Baseline/Ablation 状态

T18 不实现 baseline/ablation。以下对照实验只作为后续阶段：

Baseline/ablation status: not implemented in T18.

- only `P_i^t`
- ordinary graph embedding
- hypergraph embedding

当前 launcher 和 runbook 只允许 clean hypergraph 主方法。
