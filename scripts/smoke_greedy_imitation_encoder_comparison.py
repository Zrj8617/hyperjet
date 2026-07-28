from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_greedy_imitation_encoder_comparison as comparison
import config


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = comparison.build_arg_parser()
    parsed = parser.parse_args(["--smoke"])
    parsed = comparison._apply_smoke_overrides(parsed)
    _assert(parsed.generate_dataset, "smoke should explicitly select frozen dataset generation")
    _assert(parsed.skip_closed_loop_eval, "smoke must skip closed-loop evaluation")
    _assert(parsed.episodes <= 6, "smoke must cap dataset episodes")
    _assert(parsed.supervised_epochs == 1, "smoke must cap supervised epochs")
    _assert(tuple(comparison.CANONICAL_ENCODERS) == (
        "mlp",
        "current_mean_hgnn",
        "standard_weighted_hgnn",
        "typed_gated_hgnn",
    ), "canonical encoder list drifted")
    try:
        unsafe_defaults = parser.parse_args([])
        comparison._validate_args(unsafe_defaults)
    except ValueError:
        pass
    else:
        raise AssertionError("comparison defaults must not silently start dataset generation/training")

    _assert(
        comparison._epoch_shuffle_seed(42, 0) == comparison._epoch_shuffle_seed(42, 0),
        "epoch shuffle seed must be deterministic",
    )
    _assert(
        comparison._scorer_seed(42) == comparison._scorer_seed(42),
        "candidate scorer seed must be deterministic",
    )

    if not _torch_available():
        print("smoke_greedy_imitation_encoder_comparison skipped torch comparison: torch is not installed")
        return 0

    with _workspace_temp_dir("comparison") as temp_dir:
        original_partition = config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = False
        try:
            exit_code = comparison.main(
                [
                    "--smoke",
                    "--generate-dataset",
                "--episodes",
                "3",
                "--max-steps-per-episode",
                "1",
                    "--training-seeds",
                    "17",
                "--encoders",
                *comparison.CANONICAL_ENCODERS,
                "--trajectory-policies",
                "greedy_eft",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(temp_dir),
                    "--run-name",
                    "phase5_smoke",
                ]
            )
        finally:
            config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = original_partition
        _assert(exit_code == 0, "minimal four-encoder comparison should complete")
        run_dirs = [path for path in temp_dir.iterdir() if path.is_dir()]
        _assert(len(run_dirs) == 1, "smoke should create one comparison run")
        run_dir = run_dirs[0]
        summary = json.loads((run_dir / "comparison_summary.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (run_dir / "comparison_rows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _assert(summary["schema"] == comparison.COMPARISON_SCHEMA, "comparison schema mismatch")
        _assert(summary["smoke_only"] is True, "smoke summary must be marked smoke_only")
        _assert(
            summary["not_a_scientific_result"] is True,
            "smoke summary must state it is not a scientific result",
        )
        _assert(summary["split_leakage_count"] == 0, "comparison dataset leaked across splits")
        _assert(
            summary["candidate_scorer_initial_hash_match"] is True,
            "all encoders must share the candidate scorer initial state",
        )
        _assert(summary["technical_pass"] is True, "comparison technical checks should pass")
        _assert(len(rows) == 4, "smoke should emit one row for each canonical encoder")
        _assert(
            len({row["dataset_checksum"] for row in rows}) == 1,
            "all encoders must consume the same dataset checksum",
        )
        _assert(
            len({json.dumps(row["sample_id_hashes"], sort_keys=True) for row in rows}) == 1,
            "all encoders must consume identical train/val/test sample IDs",
        )
        _assert(
            len({row["candidate_scorer_initial_state_hash"] for row in rows}) == 1,
            "candidate scorer initial state hashes must match",
        )
        _assert(
            len({tuple(row["shuffle_order_hashes"]) for row in rows}) == 1,
            "all encoders must use identical epoch shuffle order",
        )
        typed = next(row for row in rows if row["encoder"] == "typed_gated_hgnn")
        _assert(typed["typed_diagnostics"] is not None, "typed diagnostics must be present")
        _assert(typed["typed_diagnostics"]["layers"], "typed layer diagnostics must be non-empty")
        for row in rows:
            _assert(row["technical_pass"] is True, f"{row['encoder']} technical checks failed")
            _assert(row["finite_gradient_norm"] is True, f"{row['encoder']} gradient was non-finite")
            variant_dir = Path(row["variant_dir"])
            for filename in ("config.json", "train_metrics.jsonl", "test_metrics.json", "checkpoint.pt"):
                _assert((variant_dir / filename).is_file(), f"missing variant artifact {filename}")

    print("smoke_greedy_imitation_encoder_comparison passed")
    return 0


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class _workspace_temp_dir:
    def __init__(self, name: str) -> None:
        self.path = ROOT / ".codex_tmp_greedy_imitation_encoder_comparison" / (
            f"{name}_{os.getpid()}_{random.randint(0, 1_000_000)}"
        )

    def __enter__(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        parent = ROOT / ".codex_tmp_greedy_imitation_encoder_comparison"
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
