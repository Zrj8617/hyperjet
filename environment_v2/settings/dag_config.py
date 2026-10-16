"""DAG 拓扑、数据量与计算量生成参数的 TOML 读取。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DAGConfig:
    """DAG 拓扑、数据量与计算量生成参数。"""

    min_tasks: int
    max_tasks: int
    max_levels: int
    max_parents: int
    min_input_bits: float
    max_input_bits: float
    min_edge_bits: float
    max_edge_bits: float
    min_return_bits: float
    max_return_bits: float
    base_unit_bits: float
    min_task_constant: int
    max_task_constant: int
    linear_probability: float
    log_probability: float
    squared_log_probability: float

    @classmethod
    def from_toml(cls, path: str | Path) -> "DAGConfig":
        """从 TOML 的 dag 段直接构造配置。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)["dag"]
        return cls(**data)
