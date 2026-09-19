"""Publish the 20260918 EQ10 training and fixed-tape evaluation runs to TensorBoard."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

TREATMENTS = (
    "B2_MLP_EQ10",
    "B2_TYPED_GATED_HGNN_EQ10",
    "C2A_MLP_EQ10",
    "C2A_TYPED_GATED_HGNN_EQ10",
    "C2B_MLP_EQ10",
    "C2B_TYPED_GATED_HGNN_EQ10",
    "C2_MLP_EQ10",
    "C2_TYPED_GATED_HGNN_EQ10",
)
SEEDS = (5, 86, 617)
CANDIDATE_EPISODES = (320, 360, 400, 450, 500)
PROTOCOLS = ("joint", "forced_hover")
CHECKPOINT_LABELS = ("budget_selected", "final_ep0500")
SENSITIVITY_LAMBDAS = (0.1, 0.7, 1.0)
BASE_METRICS = (
    "J_episode",
    "J_per_offer",
    "J_delay_component",
    "J_task_energy_component",
    "J_move_energy_component",
    "J_per_offer_delay_component",
    "J_per_offer_task_energy_component",
    "J_per_offer_move_energy_component",
    "N_offer",
    "N_admitted",
    "N_rejected",
    "N_completed",
    "admission_rate",
    "conditional_completion_rate",
    "end_to_end_completion_rate",
    "delay_seconds_total",
    "completed_DAG_flowtime_mean",
    "completed_DAG_flowtime_median",
    "throughput",
    "task_energy_joules_total",
    "move_energy_joules_total",
    "energy_per_completed_DAG",
    "hover_action_ratio",
    "m_action",
    "m_effective",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--run-prefix", default="20260918")
    return parser


def _summary_writer(log_dir: Path, *, suffix: str):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=str(log_dir), filename_suffix=suffix)


def _identity(treatment: str) -> tuple[str, str]:
    if treatment.endswith("_MLP_EQ10"):
        return treatment[: -len("_MLP_EQ10")], "mlp"
    if treatment.endswith("_TYPED_GATED_HGNN_EQ10"):
        return treatment[: -len("_TYPED_GATED_HGNN_EQ10")], "typed_gated_hgnn"
    raise ValueError(f"unknown treatment: {treatment}")


def _visible_name(*, run_prefix: str, treatment: str, seed: int) -> str:
    arm, encoder = _identity(treatment)
    if encoder == "mlp":
        return f"{run_prefix}_{arm}_seed{seed}"
    return f"{run_prefix}_TYPED_GATED_HGNN_{arm}_seed{seed}"


def _lambda_label(value: float) -> str:
    return str(value).replace(".", "p")


def _read_result(path: Path, *, expected_rows: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "six_arm_fixed_tape_evaluation_v2":
        raise ValueError(f"unexpected evaluation schema: {path}")
    if payload.get("status") != "completed":
        raise ValueError(f"incomplete evaluation result: {path}")
    if len(payload.get("rows", [])) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows: {path}")
    return payload


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _flatten_numeric(payload: dict[str, Any], *, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        numeric = _numeric(value)
        if numeric is not None:
            flattened[name] = numeric
        elif isinstance(value, dict):
            flattened.update(_flatten_numeric(value, prefix=name))
    return flattened


def _augmented_metrics(row: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in BASE_METRICS:
        if metric == "m_action":
            values[metric] = 1.0 - float(row["hover_action_ratio"])
        elif metric == "m_effective":
            values[metric] = float(row["move_energy_joules_total"]) / (500.0 * 5.0 * 500.0)
        else:
            numeric = _numeric(row.get(metric))
            if numeric is not None:
                values[metric] = numeric
    for key, value in _flatten_numeric(dict(row.get("policy_metrics", {}))).items():
        values[f"policy/{key}"] = value
    return values


def _sensitivity(row: dict[str, Any], energy_lambda: float) -> dict[str, float]:
    offers = float(max(int(row["N_offer"]), 1))
    delay = float(row["delay_seconds_total"]) / 500.0
    task = energy_lambda * float(row["task_energy_joules_total"]) / 500.0
    move = energy_lambda * float(row["move_energy_joules_total"]) / 500.0
    total = delay + task + move
    return {
        "J_episode": total,
        "J_per_offer": total / offers,
        "J_per_offer_delay_component": delay / offers,
        "J_per_offer_task_energy_component": task / offers,
        "J_per_offer_move_energy_component": move / offers,
    }


def _link_training_events(*, source: Path, target: Path) -> list[str]:
    event_files = sorted(source.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard events in {source}")
    linked: list[str] = []
    for index, event_file in enumerate(event_files):
        link = target / f"training_{index:02d}_{event_file.name}"
        link.symlink_to(event_file.resolve())
        linked.append(str(event_file.resolve()))
    return linked


def _write_rows(writer: Any, *, prefix: str, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        step = int(row["tape_id"])
        for metric, value in _augmented_metrics(row).items():
            writer.add_scalar(f"{prefix}/{metric}", value, step)


def _validate_manifest(payload: dict[str, Any]) -> None:
    if payload.get("status") != "completed":
        raise ValueError("fair evaluation is not completed")
    if payload.get("test_tapes_read_after_selection_locked") is not True:
        raise ValueError("test tapes were not read under the locked-selection protocol")
    if tuple(payload.get("arms_order", ())) != TREATMENTS:
        raise ValueError("evaluation treatment order differs from the frozen matrix")
    if tuple(payload.get("seeds", ())) != SEEDS:
        raise ValueError("evaluation seeds differ from the frozen matrix")


def _verify_run(run_dir: Path) -> dict[str, Any]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    required_counts = {
        "train/episode_reward": 500,
        "validation/ep0320/J_per_offer": 20,
        "validation/mean/J_per_offer": 5,
        "selection/selected_episode": 1,
        "joint/budget_selected/J_per_offer": 50,
        "joint/final_ep0500/J_per_offer": 50,
        "forced_hover/budget_selected/J_per_offer": 50,
        "forced_hover/final_ep0500/J_per_offer": 50,
        "sensitivity/lambda_0p1/J_per_offer": 50,
        "sensitivity/lambda_0p7/J_per_offer": 50,
        "sensitivity/lambda_1p0/J_per_offer": 50,
    }
    failures = {}
    for tag, expected in required_counts.items():
        actual = len(accumulator.Scalars(tag)) if tag in tags else 0
        if actual != expected:
            failures[tag] = {"actual": actual, "expected": expected}
    if failures:
        raise ValueError(f"TensorBoard verification failed for {run_dir}: {failures}")
    return {"scalar_tag_count": len(tags), "required_counts": required_counts}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.run_prefix) != "20260918":
        raise ValueError("--run-prefix must be 20260918")
    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")
    if not args.view_root.is_dir():
        raise FileNotFoundError(f"existing TensorBoard view root required: {args.view_root}")

    evaluation_manifest = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    _validate_manifest(evaluation_manifest)
    selections = json.loads(
        (args.evaluation_root / "selection_six_arm.json").read_text(encoding="utf-8")
    )
    visible_names = [
        _visible_name(run_prefix=args.run_prefix, treatment=treatment, seed=seed)
        for treatment in TREATMENTS
        for seed in SEEDS
    ]
    summary_name = f"{args.run_prefix}_SUMMARY"
    for name in (*visible_names, summary_name):
        if (args.view_root / name).exists() or (args.view_root / name).is_symlink():
            raise FileExistsError(f"TensorBoard view target already exists: {name}")

    args.output_root.mkdir(parents=True)
    run_records: list[dict[str, Any]] = []
    summary_values: dict[str, list[float]] = {}
    try:
        for treatment in TREATMENTS:
            arm, encoder = _identity(treatment)
            for seed in SEEDS:
                visible_name = _visible_name(
                    run_prefix=args.run_prefix, treatment=treatment, seed=seed
                )
                source_result_path = (
                    args.audit_root
                    / f"{args.run_prefix}_{treatment}_seed{seed}"
                    / "result.json"
                )
                source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
                if source_result.get("status") != "completed":
                    raise ValueError(f"incomplete training result: {source_result_path}")
                source_events = Path(source_result["tensorboard"]["directory"])
                run_dir = args.output_root / visible_name
                run_dir.mkdir()
                training_event_files = _link_training_events(
                    source=source_events, target=run_dir
                )
                writer = _summary_writer(run_dir, suffix=".eq10-evaluation")

                validation_means: dict[str, list[tuple[int, float]]] = {}
                for episode in CANDIDATE_EPISODES:
                    path = (
                        args.evaluation_root
                        / "validation"
                        / f"{treatment}_seed{seed}_ep{episode:04d}_validation_joint.json"
                    )
                    payload = _read_result(path, expected_rows=20)
                    _write_rows(
                        writer,
                        prefix=f"validation/ep{episode:04d}",
                        rows=payload["rows"],
                    )
                    metric_rows = [_augmented_metrics(row) for row in payload["rows"]]
                    for metric in BASE_METRICS:
                        values = [row[metric] for row in metric_rows if metric in row]
                        if values:
                            validation_means.setdefault(metric, []).append(
                                (episode, statistics.fmean(values))
                            )
                for metric, points in validation_means.items():
                    for episode, value in points:
                        writer.add_scalar(f"validation/mean/{metric}", value, episode)

                selected = selections[treatment][str(seed)]
                writer.add_scalar(
                    "selection/selected_episode", float(selected["selected_episode"]), 0
                )
                writer.add_scalar(
                    "selection/joint_validation_J_per_offer_mean",
                    float(selected["joint_validation_J_per_offer_mean"]),
                    0,
                )

                joint_budget_rows: list[dict[str, Any]] = []
                for protocol in PROTOCOLS:
                    for checkpoint_label in CHECKPOINT_LABELS:
                        path = (
                            args.evaluation_root
                            / "test"
                            / f"{treatment}_seed{seed}_{checkpoint_label}_test_{protocol}.json"
                        )
                        payload = _read_result(path, expected_rows=50)
                        rows = list(payload["rows"])
                        _write_rows(
                            writer,
                            prefix=f"{protocol}/{checkpoint_label}",
                            rows=rows,
                        )
                        if protocol == "joint" and checkpoint_label == "budget_selected":
                            joint_budget_rows = rows

                for energy_lambda in SENSITIVITY_LAMBDAS:
                    label = _lambda_label(energy_lambda)
                    for row in joint_budget_rows:
                        for metric, value in _sensitivity(row, energy_lambda).items():
                            writer.add_scalar(
                                f"sensitivity/lambda_{label}/{metric}",
                                value,
                                int(row["tape_id"]),
                            )
                writer.close()

                main_values = [float(row["J_per_offer"]) for row in joint_budget_rows]
                summary_values.setdefault(f"main/{encoder}/{arm}/J_per_offer", []).extend(
                    main_values
                )
                for energy_lambda in SENSITIVITY_LAMBDAS:
                    label = _lambda_label(energy_lambda)
                    values = [
                        _sensitivity(row, energy_lambda)["J_per_offer"]
                        for row in joint_budget_rows
                    ]
                    summary_values.setdefault(
                        f"sensitivity/lambda_{label}/{encoder}/{arm}/J_per_offer", []
                    ).extend(values)

                run_records.append(
                    {
                        "visible_name": visible_name,
                        "treatment": treatment,
                        "arm": arm,
                        "encoder": encoder,
                        "seed": seed,
                        "source_result": str(source_result_path),
                        "source_event_files": training_event_files,
                        "event_dir": str(run_dir),
                    }
                )

        summary_dir = args.output_root / summary_name
        summary_writer = _summary_writer(summary_dir, suffix=".eq10-summary")
        summary_means = {
            tag: statistics.fmean(values) for tag, values in summary_values.items()
        }
        for tag, value in summary_means.items():
            summary_writer.add_scalar(tag, value, 0)
        for encoder in ("mlp", "typed_gated_hgnn"):
            arm_value = {
                arm: summary_means[f"main/{encoder}/{arm}/J_per_offer"]
                for arm in ("B2", "C2A", "C2B", "C2")
            }
            summary_writer.add_scalar(
                f"effects/{encoder}/movement_teacher/J_per_offer",
                0.5 * ((arm_value["C2B"] - arm_value["B2"]) + (arm_value["C2"] - arm_value["C2A"])),
                0,
            )
            summary_writer.add_scalar(
                f"effects/{encoder}/offloading_teacher/J_per_offer",
                0.5 * ((arm_value["C2A"] - arm_value["B2"]) + (arm_value["C2"] - arm_value["C2B"])),
                0,
            )
            summary_writer.add_scalar(
                f"effects/{encoder}/interaction/J_per_offer",
                arm_value["C2"] - arm_value["C2A"] - arm_value["C2B"] + arm_value["B2"],
                0,
            )
        for arm in ("B2", "C2A", "C2B", "C2"):
            summary_writer.add_scalar(
                f"encoder_effect/{arm}/HGNN_minus_MLP/J_per_offer",
                summary_means[f"main/typed_gated_hgnn/{arm}/J_per_offer"]
                - summary_means[f"main/mlp/{arm}/J_per_offer"],
                0,
            )
        summary_writer.close()

        verification = {
            record["visible_name"]: _verify_run(Path(record["event_dir"]))
            for record in run_records
        }
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        summary_accumulator = EventAccumulator(
            str(summary_dir), size_guidance={"scalars": 0}
        )
        summary_accumulator.Reload()
        summary_tag_count = len(summary_accumulator.Tags().get("scalars", []))
        if summary_tag_count == 0:
            raise ValueError("summary TensorBoard run contains no scalar tags")

        published = []
        for record in run_records:
            link = args.view_root / record["visible_name"]
            link.symlink_to(Path(record["event_dir"]).resolve(), target_is_directory=True)
            published.append(str(link))
        summary_link = args.view_root / summary_name
        summary_link.symlink_to(summary_dir.resolve(), target_is_directory=True)
        published.append(str(summary_link))

        manifest = {
            "schema": "eq10_tensorboard_export_v1",
            "status": "pass",
            "run_prefix": str(args.run_prefix),
            "run_count": len(run_records),
            "summary_run": summary_name,
            "summary_tag_count": summary_tag_count,
            "evaluation_manifest": str(args.evaluation_manifest.resolve()),
            "evaluation_root": str(args.evaluation_root.resolve()),
            "view_root": str(args.view_root.resolve()),
            "published_links": published,
            "runs": run_records,
            "verification": verification,
        }
        (args.output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    except Exception:
        for name in (*visible_names, summary_name):
            link = args.view_root / name
            if link.is_symlink():
                link.unlink()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
