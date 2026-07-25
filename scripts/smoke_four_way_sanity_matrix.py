from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import launch_four_way_sanity_matrix as launcher


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = launcher.build_arg_parser().parse_args([])
    matrix = launcher.build_matrix(args, Path("logs") / "matrix_placeholder")
    _assert(len(matrix) == 12, "default matrix must contain exactly 12 cells")
    _assert(
        len({cell["cell_id"] for cell in matrix}) == 12,
        "matrix cell IDs must be unique",
    )
    by_method = {
        method: [cell for cell in matrix if cell["method"] == method]
        for method in launcher.METHODS
    }
    for method, cells in by_method.items():
        _assert(len(cells) == 3, f"{method} must contain three seeds")
        _assert(
            {cell["seed"] for cell in cells} == {42, 86, 1042},
            f"{method} seed set mismatch",
        )
    for cell in by_method["mappo_mlp"]:
        command = cell["command"]
        _assert(_option_value(command, "--task-encoder") == "mlp", "MLP cell encoder mismatch")
        _assert("--freeze-movement" in command, "MLP cell must freeze movement")
        _assert("--no-normalize-value-targets" in command, "MLP cell must disable value normalization")
        _assert(_option_value(command, "--value-clip-epsilon") == "0", "MLP cell must disable value clip")
    for cell in by_method["mappo_hgnn"]:
        _assert(_option_value(cell["command"], "--task-encoder") == "hgnn", "HGNN cell encoder mismatch")
    for method in ("random_hash", "greedy_eft"):
        for cell in by_method[method]:
            _assert(
                "scripts/run_clean_policy_baseline.py" in cell["command"],
                f"{method} must use the no-learning runner",
            )

    mlp_path = ROOT / "marl_models" / "hgnn" / "clean_independent_mlp.py"
    tree = ast.parse(mlp_path.read_text(encoding="utf-8"))
    forward = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
    )
    _assert(
        any(
            isinstance(node, ast.Delete)
            and any(isinstance(target, ast.Name) and target.id == "incidence_matrix" for target in node.targets)
            for node in ast.walk(forward)
        ),
        "independent MLP forward must explicitly ignore incidence_matrix",
    )
    print("smoke_four_way_sanity_matrix passed")
    return 0


def _option_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return str(command[index + 1])


if __name__ == "__main__":
    raise SystemExit(main())
