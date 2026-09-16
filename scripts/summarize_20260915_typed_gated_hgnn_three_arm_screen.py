from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


ARMS = ("B2", "C1", "C2")
SEEDS = (5, 86, 617)
PROTOCOLS = ("joint", "forced_hover")
LABELS = ("selected", "final_ep0500")
METRICS = (
    "J_per_offer",
    "J_per_offer_delay_component",
    "J_per_offer_task_energy_component",
    "J_per_offer_move_energy_component",
    "admission_rate",
    "conditional_completion_rate",
    "end_to_end_completion_rate",
    "completed_DAG_flowtime_mean",
    "completed_DAG_flowtime_median",
    "throughput",
    "energy_per_completed_DAG",
    "hover_action_ratio",
)
LOWER_IS_BETTER = {
    "J_per_offer",
    "J_per_offer_delay_component",
    "J_per_offer_task_energy_component",
    "J_per_offer_move_energy_component",
    "completed_DAG_flowtime_mean",
    "completed_DAG_flowtime_median",
    "energy_per_completed_DAG",
}
GRADIENT_KEYS = (
    "grad_clip_scale",
    "grad_pre_clip_hgnn",
    "grad_post_clip_hgnn",
    "grad_pre_clip_movement",
    "grad_post_clip_movement",
    "grad_pre_clip_offloading",
    "grad_post_clip_offloading",
    "grad_pre_clip_critic",
    "grad_post_clip_critic",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize paired typed-gated HGNN vs MLP results.")
    parser.add_argument("--hgnn-manifest", type=Path, required=True)
    parser.add_argument("--mlp-manifest", type=Path, required=True)
    parser.add_argument("--hgnn-launch-manifest", type=Path, required=True)
    parser.add_argument("--mlp-launch-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_hierarchical_ci(
    values_by_seed: dict[int, list[float]], *, replicates: int, rng: random.Random
) -> list[float]:
    seeds = tuple(values_by_seed)
    samples = []
    for _ in range(replicates):
        seed_means = []
        for sampled_seed in (rng.choice(seeds) for _ in seeds):
            values = values_by_seed[sampled_seed]
            seed_means.append(_mean([rng.choice(values) for _ in values]))
        samples.append(_mean(seed_means))
    return [_quantile(samples, 0.025), _quantile(samples, 0.975)]


def _hgnn_path(root: Path, arm: str, seed: int, label: str, protocol: str) -> Path:
    return root / arm / f"seed{seed}" / "test" / f"{label}_{protocol}.json"


def _mlp_path(root: Path, arm: str, seed: int, label: str, protocol: str) -> Path:
    mlp_label = "budget_selected" if label == "selected" else label
    return root / "test" / f"{arm}_seed{seed}_{mlp_label}_{protocol}.json"


def _rows_by_tape(path: Path) -> dict[int, dict[str, Any]]:
    payload = _read(path)
    if payload.get("status") != "completed" or len(payload.get("rows", [])) != 50:
        raise RuntimeError(f"incomplete evaluation result: {path}")
    if payload.get("enable_kahypar") and not payload.get("kahypar_health", {}).get("pass"):
        raise RuntimeError(f"unhealthy KaHyPar evaluation result: {path}")
    return {int(row["tape_id"]): dict(row) for row in payload["rows"]}


def _paired_results(
    *, hgnn_root: Path, mlp_root: Path, replicates: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label_index, label in enumerate(LABELS):
        output[label] = {}
        for protocol_index, protocol in enumerate(PROTOCOLS):
            output[label][protocol] = {}
            for arm_index, arm in enumerate(ARMS):
                metrics: dict[str, Any] = {}
                cached: dict[int, tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]] = {}
                for seed in SEEDS:
                    hgnn_rows = _rows_by_tape(_hgnn_path(hgnn_root, arm, seed, label, protocol))
                    mlp_rows = _rows_by_tape(_mlp_path(mlp_root, arm, seed, label, protocol))
                    if set(hgnn_rows) != set(mlp_rows):
                        raise ValueError(f"tape pairing mismatch for {arm}/{seed}/{label}/{protocol}")
                    cached[seed] = (hgnn_rows, mlp_rows)
                for metric_index, metric in enumerate(METRICS):
                    values_by_seed = {
                        seed: [
                            float(cached[seed][0][tape_id][metric])
                            - float(cached[seed][1][tape_id][metric])
                            for tape_id in sorted(cached[seed][0])
                        ]
                        for seed in SEEDS
                    }
                    seed_means = [_mean(values_by_seed[seed]) for seed in SEEDS]
                    favorable = [
                        value < 0.0 if metric in LOWER_IS_BETTER else value > 0.0
                        for value in seed_means
                    ]
                    metrics[metric] = {
                        "estimand": "typed_gated_hgnn_plus_kahypar_minus_mlp",
                        "mean": _mean(seed_means),
                        "seed_mean_differences": seed_means,
                        "favorable_seed_count": int(sum(favorable)),
                        "hierarchical_bootstrap_95_ci": _paired_hierarchical_ci(
                            values_by_seed,
                            replicates=replicates,
                            rng=random.Random(
                                20260916 + label_index * 100_003 + protocol_index * 1_000_003
                                + arm_index * 10_000_019 + metric_index * 1009
                            ),
                        ),
                    }
                output[label][protocol][arm] = metrics
    return output


def _lambda_sweep(hgnn_root: Path, mlp_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in (0.10, 0.04, 0.0):
        arm_result: dict[str, Any] = {}
        for arm in ARMS:
            seed_differences = []
            for seed in SEEDS:
                hgnn = _rows_by_tape(_hgnn_path(hgnn_root, arm, seed, "selected", "joint"))
                mlp = _rows_by_tape(_mlp_path(mlp_root, arm, seed, "selected", "joint"))
                differences = []
                for tape_id in sorted(hgnn):
                    def score(row: dict[str, Any]) -> float:
                        numerator = (
                            float(row["delay_seconds_total"])
                            + float(row["task_energy_joules_total"])
                            + value * float(row["move_energy_joules_total"])
                        ) / 500.0
                        return numerator / max(int(row["N_offer"]), 1)
                    differences.append(score(hgnn[tape_id]) - score(mlp[tape_id]))
                seed_differences.append(_mean(differences))
            arm_result[arm] = {
                "mean_difference": _mean(seed_differences),
                "seed_mean_differences": seed_differences,
            }
        result[f"{value:g}"] = arm_result
    return result


def _gradient_summary(hgnn_launch: dict[str, Any], mlp_launch: dict[str, Any]) -> dict[str, Any]:
    summaries: dict[str, Any] = {"typed_gated_hgnn": {}, "mlp": {}}
    for label, launch in (("typed_gated_hgnn", hgnn_launch), ("mlp", mlp_launch)):
        wanted = {
            (arm, seed): row for row in launch.get("runs", [])
            for arm in ARMS for seed in SEEDS
            if str(row.get("arm")) == arm and int(row.get("seed")) == seed
        }
        for arm in ARMS:
            summaries[label][arm] = {}
            for seed in SEEDS:
                result = _read(Path(wanted[(arm, seed)]["result_path"]))
                metrics_path = Path(result["train_dir"]) / "train_metrics.jsonl"
                values = {key: [] for key in GRADIENT_KEYS}
                for line in metrics_path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    diagnostics = dict(row.get("ppo_diagnostics", {}))
                    for key in GRADIENT_KEYS:
                        if diagnostics.get(key) is not None:
                            values[key].append(float(diagnostics[key]))
                summaries[label][arm][str(seed)] = {
                    "parameter_counts": result.get("parameter_counts"),
                    "gradients": {
                        key: {
                            "mean": _mean(items),
                            "median": float(statistics.median(items)),
                            "min": min(items),
                            "max": max(items),
                        }
                        for key, items in values.items() if items
                    },
                }
    return summaries


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# typed-gated HGNN 三臂筛选结果",
        "",
        f"状态：{result['status']}",
        "",
        "本报告只比较完整 typed-gated HGNN + KaHyPar treatment 与现有 MLP；不把效应单独归因给内部组件。",
        "",
        "## 主结果：selected checkpoint / joint",
        "",
        "| Arm | ΔJ/offer | 95% CI | 有利 seed | Δflowtime | Δthroughput |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    primary = result["paired_effects"]["selected"]["joint"]
    for arm in ARMS:
        j = primary[arm]["J_per_offer"]
        flow = primary[arm]["completed_DAG_flowtime_mean"]
        throughput = primary[arm]["throughput"]
        lines.append(
            f"| {arm} | {j['mean']:.6g} | [{j['hierarchical_bootstrap_95_ci'][0]:.6g}, "
            f"{j['hierarchical_bootstrap_95_ci'][1]:.6g}] | {j['favorable_seed_count']}/3 | "
            f"{flow['mean']:.6g} | {throughput['mean']:.6g} |"
        )
    lines.extend(["", "完整 JSON 包含全部协议、final checkpoint、lambda_move 与 gradient clipping 诊断。", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    hgnn_manifest = _read(args.hgnn_manifest)
    mlp_manifest = _read(args.mlp_manifest)
    if hgnn_manifest.get("status") != "completed" or mlp_manifest.get("status") != "completed":
        raise ValueError("both evaluation manifests must be completed")
    hgnn_root = args.hgnn_manifest.parent
    mlp_root = Path(mlp_manifest["output_root"])
    result = {
        "schema": "typed_gated_hgnn_three_arm_screen_result_v1",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conclusion_boundary": "three-seed non-capacity-matched screen; not a final HGNN-vs-MLP claim",
        "paired_effects": _paired_results(
            hgnn_root=hgnn_root, mlp_root=mlp_root,
            replicates=int(args.bootstrap_replicates),
        ),
        "lambda_move_sweep": _lambda_sweep(hgnn_root, mlp_root),
        "gradient_diagnostics": _gradient_summary(
            _read(args.hgnn_launch_manifest), _read(args.mlp_launch_manifest)
        ),
        "sources": {
            "hgnn_manifest": str(args.hgnn_manifest),
            "mlp_manifest": str(args.mlp_manifest),
            "hgnn_launch_manifest": str(args.hgnn_launch_manifest),
            "mlp_launch_manifest": str(args.mlp_launch_manifest),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render(result), encoding="utf-8")
    print(json.dumps({"status": "pass", "output_json": str(args.output_json)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
