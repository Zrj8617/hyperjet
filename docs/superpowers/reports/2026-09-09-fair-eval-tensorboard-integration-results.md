# C1 / B2 / C2 公平评估 TensorBoard 接入结果（2026-09-09）

**状态：PASS**

固定-tape Track A 正式评估已接入 `http://127.0.0.1:16008/`。原训练 run 未修改；新增 9 个 `FormalEval/{arm}/seed{seed}` run。随后接入 C2 三个原始训练 run，TensorBoard API 当前共返回 42 个 run。

## 接入内容

- `forced_hover/budget_selected/*`：正式主协议。
- `forced_hover/final_ep0500/*`：最终 checkpoint 敏感性分析。
- `joint/budget_selected/*`、`joint/final_ep0500/*`：deterministic joint-policy 次协议。
- 横轴严格使用独立 test tape ID 200--249。
- 每个 `FormalEval` run 有 80 个固定-tape scalar tag，每个 tag 有 50 个 test-tape 点。
- 每个 `FormalEval` run 另有 `train/episode_reward` 和 `train/critic_loss` 两张训练上下文卡片；值和 step 直接复制自对应原始训练 event。C1/C2 的前者为 500 个 episode 点；旧 B2 的前者原生记录为 2000 个 rollout/update 点，不能按 episode 曲线解释；后者均为 2000 个 PPO update 点。
- 指标覆盖 `J_episode`、`J_per_offer`、时延/任务能耗/移动能耗分量、offer/admission/completion 计数与比例，以及流时、吞吐和能耗辅助指标。
- Track B 多峰值 epoch 诊断未放入 `FormalEval`。

## 验证

- EventAccumulator：9/9 run 通过；82 个标签的集合、点数和固定-tape step 序列全部通过。
- 本机 TensorBoard API：9 个 `FormalEval` run 全部可见，9/9 均包含 `forced_hover/budget_selected/J_per_offer`。
- 页面渲染：Scalars 页面正常显示 `forced_hover` 分组及曲线。
- 持久 tunnel：复用 `HyperUAV-RewardRedesign-TensorBoard` 登录任务，本机 `127.0.0.1:16008` 已监听。

## 路径

- 服务器事件根：`/data2/zrj2025/uav-results/audits/20260908_C1_B2_C2_fair_eval/tensorboard_formal_eval`
- TensorBoard view：`/data2/zrj2025/uav-results/tensorboard_views/reward_redesign/FormalEval`
- 导出脚本：`scripts/export_fair_eval_tensorboard.py`
- 训练上下文追加脚本：`scripts/append_fair_eval_training_tags.py`
- 校验脚本：`scripts/check_fair_eval_tensorboard.py`
- 页面验证截图：`output/formal_eval_tensorboard.png`

## 版本状态

- 接入前设计 commit：`a6142a76e98ce5432436067ecfdd5298612c0856`
- 工作树原本已有大量用户/实验改动；接入过程没有清理或覆盖这些文件。
