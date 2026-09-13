from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


CONFIGURATIONS = ("N0", "A2", "B1", "B2", "B2D", "D2", "A1", "C1", "N0-local")
OLD_CONFIGURATIONS = {"N0", "A2", "B1", "B2", "B2D", "D2"}
METRICS = {
    "completion_rate": "DAG_completion_rate",
    "flowtime_seconds": "Average_DAG_flowtime",
    "throughput": "DAG_throughput",
    "agreement_eft_greedy": "actor_greedy_agreement_rate",
}
COMPARISONS = (
    ("A1", "A2"),
    ("C1", "A1"),
    ("C1", "A2"),
    ("N0-local", "N0"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _result_path(root: Path, configuration: str, model_seed: int) -> Path:
    prefix = "20260907_reeval_" if configuration in OLD_CONFIGURATIONS else "20260907_"
    return root / f"{prefix}{configuration}_seed{model_seed}" / "result.json"


def _cell_rows(root: Path, configuration: str, model_seed: int) -> list[dict[str, Any]]:
    payload = _read_json(_result_path(root, configuration, model_seed))
    evaluation = payload["evaluation"]
    run_dir = Path(evaluation["run_dir"])
    rows = _read_jsonl(run_dir / "eval_metrics.jsonl")
    if len(rows) != 20:
        raise RuntimeError(f"{configuration} seed {model_seed}: expected 20 eval rows, got {len(rows)}")
    if [int(row["environment_seed"]) for row in rows] != list(range(20)):
        raise RuntimeError(f"{configuration} seed {model_seed}: environment seed mismatch")
    return rows


def _ci(values: np.ndarray, indices: np.ndarray) -> list[float]:
    draws = values[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _reference_rows(leverage: dict[str, Any], name: str) -> dict[str, np.ndarray]:
    cells = leverage["cells"][name]
    return {
        metric_name: np.asarray(
            [float(cells[str(seed)]["metrics"][source.lower()]) for seed in range(20)],
            dtype=np.float64,
        )
        for metric_name, source in {
            "completion_rate": "dag_completion_rate",
            "flowtime_seconds": "average_dag_flowtime",
            "throughput": "dag_throughput",
        }.items()
    }


def _c1_entropy(root: Path) -> dict[str, Any]:
    phase_values: dict[str, list[float]] = defaultdict(list)
    per_seed: dict[str, Any] = {}
    for seed in range(3):
        result = _read_json(root / f"20260907_C1_seed{seed}" / "result.json")
        rows = _read_jsonl(Path(result["train_dir"]) / "train_metrics.jsonl")
        values: list[float] = []
        phase_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            diagnostics = row.get("ppo_diagnostics", {})
            phase = str(diagnostics.get("teacher_phase"))
            value = float(diagnostics["rollout_offloading_entropy_normalized_mean"])
            values.append(value)
            phase_values[phase].append(value)
            phase_counts[phase] += 1
        per_seed[str(seed)] = {
            "update_count": len(values),
            "phase_counts": dict(phase_counts),
            "first_100_mean": float(np.mean(values[:100])),
            "last_100_mean": float(np.mean(values[-100:])),
            "last_20pct_mean": float(np.mean(values[int(0.8 * len(values)) :])),
            "final": float(values[-1]),
        }
    return {
        "metric": "rollout_offloading_entropy_normalized_mean",
        "by_phase": {
            phase: {
                "count": len(values),
                "mean": float(np.mean(values)),
                "p10": float(np.percentile(values, 10)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
            }
            for phase, values in sorted(phase_values.items())
        },
        "per_model_seed": per_seed,
    }


def _training_late_diagnostics(root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for configuration in CONFIGURATIONS:
        per_seed: dict[str, Any] = {}
        for seed in range(3):
            date = "20260906" if configuration in OLD_CONFIGURATIONS else "20260907"
            result = _read_json(root / f"{date}_{configuration}_seed{seed}" / "result.json")
            rows = _read_jsonl(Path(result["train_dir"]) / "train_metrics.jsonl")
            late_rows = rows[int(0.8 * len(rows)) :]
            terminal_rows = [row for row in rows if bool(row.get("episode_terminal_record", False))]
            late_terminal = terminal_rows[int(0.8 * len(terminal_rows)) :]
            per_seed[str(seed)] = {
                "normalized_offloading_entropy": float(
                    np.mean(
                        [
                            row["ppo_diagnostics"]["rollout_offloading_entropy_normalized_mean"]
                            for row in late_rows
                        ]
                    )
                ),
                "critic_explained_variance": float(
                    np.mean([row["ppo_explained_variance"] for row in late_rows])
                ),
                "approx_kl": float(np.mean([row["ppo_approx_kl"] for row in late_rows])),
                "episode_reward": float(
                    np.mean([row["episode_reward_total"] for row in late_terminal])
                ),
                "move_rate": float(
                    np.mean([1.0 - float(row["hover_action_ratio"]) for row in late_terminal])
                ),
            }
        output[configuration] = {
            metric: {
                "mean": float(np.mean([per_seed[str(seed)][metric] for seed in range(3)])),
                "per_model_seed": [per_seed[str(seed)][metric] for seed in range(3)],
            }
            for metric in next(iter(per_seed.values()))
        }
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.result_root.resolve()
    rng = np.random.default_rng(20260907)
    indices = rng.integers(0, 20, size=(int(args.bootstrap_resamples), 20))
    arrays: dict[str, dict[str, np.ndarray]] = {}
    table: dict[str, Any] = {}
    for configuration in CONFIGURATIONS:
        by_model = [_cell_rows(root, configuration, seed) for seed in range(3)]
        arrays[configuration] = {}
        table[configuration] = {"checkpoint_count": 3, "eval_cell_count": 60}
        for metric_name, source in METRICS.items():
            matrix = np.asarray(
                [[float(row[source]) for row in rows] for rows in by_model],
                dtype=np.float64,
            )
            paired_env = matrix.mean(axis=0)
            arrays[configuration][metric_name] = paired_env
            table[configuration][metric_name] = {
                "mean": float(matrix.mean()),
                "paired_environment_seed_ci95": _ci(paired_env, indices),
                "per_model_seed_mean": [float(row.mean()) for row in matrix],
            }

    leverage = _read_json(
        root / "offloading-leverage-check" / "formal_20260904_230441" / "result.json"
    )
    references: dict[str, Any] = {}
    reference_arrays: dict[str, dict[str, np.ndarray]] = {}
    for name in ("random", "eft_greedy"):
        reference = _reference_rows(leverage, name)
        reference_arrays[name] = reference
        references[name] = {
            metric: {
                "mean": float(values.mean()),
                "paired_environment_seed_ci95": _ci(values, indices),
            }
            for metric, values in reference.items()
        }
    stage1 = _read_json(
        args.repo_root.resolve()
        / "docs/superpowers/reports/2026-09-06-stage1-eft-anchored-credit-probe-result.json"
    )
    anchored = stage1["evaluation"]["on"]["pooled"]
    stage1_root = root / "stage1-eft-anchored-credit-probe" / "formal_20260905_104527" / "eval" / "on"
    stage1_arrays: dict[str, np.ndarray] = {}
    for metric_name, source in METRICS.items():
        matrix = np.empty((3, 20), dtype=np.float64)
        for model_seed in range(3):
            for env_seed in range(20):
                matches = list(
                    (stage1_root / f"model_seed{model_seed}" / f"env_seed{env_seed}").glob(
                        "*/eval_metrics.jsonl"
                    )
                )
                if len(matches) != 1:
                    raise RuntimeError("Stage-1 raw evaluation cell is missing or ambiguous")
                matrix[model_seed, env_seed] = float(_read_jsonl(matches[0])[0][source])
        stage1_arrays[metric_name] = matrix.mean(axis=0)
    references["stage1_eft_anchored"] = {
        metric: {
            "mean": float(values.mean()),
            "paired_environment_seed_ci95": _ci(values, indices),
        }
        for metric, values in stage1_arrays.items()
    }
    # Cross-check raw-cell reconstruction against the frozen Stage-1 summary.
    if abs(references["stage1_eft_anchored"]["completion_rate"]["mean"] - float(anchored["dag_completion_rate"]["mean"])) > 1e-12:
        raise RuntimeError("Stage-1 completion reconstruction mismatch")

    comparisons: dict[str, Any] = {}
    for candidate, baseline in COMPARISONS:
        key = f"{candidate}_minus_{baseline}"
        comparisons[key] = {}
        for metric in METRICS:
            delta = arrays[candidate][metric] - arrays[baseline][metric]
            comparisons[key][metric] = {
                "mean_delta": float(delta.mean()),
                "paired_environment_seed_ci95": _ci(delta, indices),
                "paired_seed_win_count": int(
                    np.sum(delta < 0.0 if metric == "flowtime_seconds" else delta > 0.0)
                ),
                "paired_seed_count": 20,
            }
    for baseline in ("random", "eft_greedy"):
        key = f"C1_minus_{baseline}"
        comparisons[key] = {}
        for metric, baseline_values in reference_arrays[baseline].items():
            delta = arrays["C1"][metric] - baseline_values
            comparisons[key][metric] = {
                "mean_delta": float(delta.mean()),
                "paired_environment_seed_ci95": _ci(delta, indices),
                "paired_seed_win_count": int(
                    np.sum(delta < 0.0 if metric == "flowtime_seconds" else delta > 0.0)
                ),
                "paired_seed_count": 20,
            }
    comparisons["C1_minus_stage1_eft_anchored"] = {}
    for metric, baseline_values in stage1_arrays.items():
        delta = arrays["C1"][metric] - baseline_values
        comparisons["C1_minus_stage1_eft_anchored"][metric] = {
            "mean_delta": float(delta.mean()),
            "paired_environment_seed_ci95": _ci(delta, indices),
            "paired_seed_win_count": int(
                np.sum(delta < 0.0 if metric == "flowtime_seconds" else delta > 0.0)
            ),
            "paired_seed_count": 20,
        }

    result = {
        "schema": "missing_arms_unified_analysis_v1",
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "checkpoint_count": 27,
            "environment_seeds": list(range(20)),
            "model_seeds": [0, 1, 2],
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "bootstrap_unit": "paired environment seed after averaging three model seeds",
            "interval": "percentile 95%",
        },
        "table": table,
        "references": references,
        "pre_registered_comparisons": comparisons,
        "c1_entropy": _c1_entropy(root),
        "training_late_20pct": _training_late_diagnostics(root),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
