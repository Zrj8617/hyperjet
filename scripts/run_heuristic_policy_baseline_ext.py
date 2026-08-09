"""第 4 轮跑批入口：在原 runner 上挂载扩展策略注册表。

中文：`scripts/run_heuristic_policy_baseline.py` 一个字符都没改。本脚本只在
**导入之后、调用 main 之前**把该模块里的两个名字换掉：

    build_policy   -> environment.heuristic_policies_round4.build_policy_ext
    POLICY_NAMES   -> POLICY_NAMES_EXT   （argparse 的 choices 在调用时才查模块全局）

这样做的理由：`run_episode` 里那一整套指标计算、`policy_config_sha256`、
输出行 schema、锚点 2/3/4 的内部断言**全部原样复用**，不产生任何重复实现，
也就不存在「复制过来的那份悄悄漂移」的风险。代价是用了 monkeypatch，
所以配套冒烟测试里有一条**惰性守卫**：老策略经扩展注册表跑出来的 episode
必须与经原注册表跑出来的逐 bit 相同。

用法与原脚本完全一致，只是 `--policies` 多了新名字：

    python scripts/run_heuristic_policy_baseline_ext.py \
        --tape-dir /path/to/tape \
        --output-dir /path/to/out \
        --policies identity+shortest_queue dag_remaining_asc+greedy_eft \
        --scenario-indices 0 1 2
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_heuristic_policy_baseline as base
from environment.heuristic_policies_round4 import POLICY_NAMES_EXT, build_policy_ext


def install() -> None:
    """把扩展注册表挂到原 runner 上（幂等）。"""
    base.build_policy = build_policy_ext
    base.POLICY_NAMES = POLICY_NAMES_EXT


if __name__ == "__main__":
    install()
    raise SystemExit(base.main())
