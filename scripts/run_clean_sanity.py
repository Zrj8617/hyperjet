from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_clean_mainline, plot_clean_metrics, smoke_clean_server_torch, train_clean_mainline


DEFAULT_EPISODES = 100
DEFAULT_MAX_STEPS = 200
DEFAULT_ROLLOUT_HORIZON = 20
DEFAULT_SEED = 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run clean mainline short sanity workflow before full experiments.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--rollout-horizon", type=int, default=DEFAULT_ROLLOUT_HORIZON)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda" if _torch_cuda_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("logs") / "clean_sanity")
    parser.add_argument("--run-name", type=str, default="sanity")
    parser.add_argument("--arrival-steps", type=int, default=200)
    parser.add_argument("--max-drain-steps", type=int, default=300)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser


def create_sanity_run_directory(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(args.run_name)).strip("_") or "sanity"
    run_dir = Path(args.output_dir) / f"{timestamp}_{clean_name}_seed{int(args.seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_workflow_commands(args: argparse.Namespace, run_dir: Path | None = None) -> list[dict[str, Any]]:
    run_dir = Path(run_dir) if run_dir is not None else Path("<sanity_run_dir>")
    train_root = run_dir / "train"
    eval_root = run_dir / "eval"
    commands: list[dict[str, Any]] = []
    if not bool(args.skip_smoke):
        commands.append({"step": "torch_model_smoke", "command": "python scripts/smoke_clean_server_torch.py"})
        commands.append(
            {
                "step": "minimal_training_smoke",
                "command": (
                    "python scripts/train_clean_mainline.py --smoke --episodes 3 "
                    "--max-steps-per-episode 20 --rollout-horizon 5 --run-name smoke"
                ),
            }
        )
    commands.append(
        {
            "step": "short_sanity_training",
            "command": (
                "python scripts/train_clean_mainline.py "
                f"--episodes {int(args.episodes)} --max-steps-per-episode {int(args.max_steps_per_episode)} "
                f"--rollout-horizon {int(args.rollout_horizon)} --seed {int(args.seed)} "
                f"--device {args.device} --output-dir {train_root} --run-name {args.run_name}"
            ),
        }
    )
    if not bool(args.skip_eval):
        commands.append(
            {
                "step": "deterministic_evaluation",
                "command": (
                    "python scripts/eval_clean_mainline.py --checkpoint <train_run_dir>/checkpoints/latest.pt "
                    f"--episodes 3 --arrival-steps {int(args.arrival_steps)} --max-drain-steps {int(args.max_drain_steps)} "
                    f"--seed {int(args.seed)} --device {args.device} --output-dir {eval_root} --run-name {args.run_name}_eval"
                ),
            }
        )
    if not bool(args.skip_plot):
        commands.append({"step": "plot_training", "command": "python scripts/plot_clean_metrics.py --run-dir <train_run_dir>"})
        if not bool(args.skip_eval):
            commands.append({"step": "plot_evaluation", "command": "python scripts/plot_clean_metrics.py --run-dir <eval_run_dir>"})
    return commands


def report_schema() -> dict[str, Any]:
    return {
        "commands": [],
        "return_codes": {},
        "train_run_dir": None,
        "eval_run_dir": None,
        "plot_paths": [],
        "final_reward": None,
        "recent_reward": None,
        "generated_DAG_count": None,
        "completed_DAG_count": None,
        "completion_rate": None,
        "throughput": None,
        "average_DAG_flowtime": None,
        "energy_per_completed_DAG": None,
        "invalid_assignment_rate": None,
        "action_executed_rate": None,
        "movement_action_distribution": {},
        "movement_hover_rate": None,
        "offloading_action_count": None,
        "train_final_metrics_source": None,
        "eval_final_metrics_source": None,
        "checkpoint_path": None,
        "pass_fail": {},
        "overall_pass": False,
    }


def run_sanity(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = create_sanity_run_directory(args)
    report = report_schema()
    report["run_dir"] = str(run_dir)
    report["config"] = _namespace_to_dict(args)
    report["commands"] = build_workflow_commands(args, run_dir)
    report_path = run_dir / "sanity_report.json"

    if not _torch_available():
        report["return_codes"]["torch_required"] = 2
        report["pass_fail"] = _pass_fail(report)
        _write_json(report_path, report)
        print("clean sanity requires torch; run this helper on the server torch environment.", file=sys.stderr)
        return report

    if not bool(args.skip_smoke):
        report["return_codes"]["torch_model_smoke"] = int(smoke_clean_server_torch.main() or 0)
        minimal_code = train_clean_mainline.main(
            [
                "--smoke",
                "--episodes",
                "3",
                "--max-steps-per-episode",
                "20",
                "--rollout-horizon",
                "5",
                "--run-name",
                "smoke",
                "--device",
                str(args.device),
                "--output-dir",
                str(run_dir / "minimal_smoke"),
            ]
        )
        report["return_codes"]["minimal_training_smoke"] = int(minimal_code)

    train_args = train_clean_mainline.build_arg_parser().parse_args(
        [
            "--episodes",
            str(int(args.episodes)),
            "--max-steps-per-episode",
            str(int(args.max_steps_per_episode)),
            "--rollout-horizon",
            str(int(args.rollout_horizon)),
            "--seed",
            str(int(args.seed)),
            "--device",
            str(args.device),
            "--output-dir",
            str(run_dir / "train"),
            "--run-name",
            str(args.run_name),
            "--checkpoint-interval",
            "1",
        ]
    )
    try:
        train_result = train_clean_mainline.run_training(train_args)
        report["return_codes"]["short_sanity_training"] = 0
        train_run_dir = Path(train_result["run_dir"])
    except Exception as exc:  # noqa: BLE001 - report must capture server smoke failures.
        report["return_codes"]["short_sanity_training"] = 1
        report["train_error"] = repr(exc)
        train_run_dir = None
    if train_run_dir is not None:
        report["train_run_dir"] = str(train_run_dir)
        train_summary = _read_json(train_run_dir / "run_summary.json")
        train_rows = _read_jsonl(train_run_dir / "train_metrics.jsonl")
        checkpoint_path = train_run_dir / "checkpoints" / "latest.pt"
        report["checkpoint_path"] = str(checkpoint_path) if checkpoint_path.exists() else None
        _merge_train_metrics(report, train_summary, train_rows)

        if not bool(args.skip_eval) and checkpoint_path.exists():
            eval_args = eval_clean_mainline.build_arg_parser().parse_args(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--episodes",
                    "3",
                    "--arrival-steps",
                    str(int(args.arrival_steps)),
                    "--max-drain-steps",
                    str(int(args.max_drain_steps)),
                    "--seed",
                    str(int(args.seed)),
                    "--device",
                    str(args.device),
                    "--output-dir",
                    str(run_dir / "eval"),
                    "--run-name",
                    f"{args.run_name}_eval",
                ]
            )
            try:
                eval_result = eval_clean_mainline.run_evaluation(eval_args)
                report["return_codes"]["deterministic_evaluation"] = 0
                report["eval_run_dir"] = str(eval_result.get("run_dir"))
                _merge_eval_metrics(report, eval_result)
            except Exception as exc:  # noqa: BLE001
                report["return_codes"]["deterministic_evaluation"] = 1
                report["eval_error"] = repr(exc)

        if not bool(args.skip_plot):
            report["plot_paths"] = []
            report["return_codes"]["plot_training"] = _plot_run_dir(train_run_dir, report)
            eval_run_dir = Path(report["eval_run_dir"]) if report.get("eval_run_dir") else None
            if eval_run_dir is not None:
                report["return_codes"]["plot_evaluation"] = _plot_run_dir(eval_run_dir, report)

    report["pass_fail"] = _pass_fail(report)
    report["overall_pass"] = bool(all(report["pass_fail"].values()))
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = run_sanity(args)
    print(json.dumps({"sanity_report": str(Path(report["run_dir"]) / "sanity_report.json"), "overall_pass": report["overall_pass"]}, sort_keys=True))
    return 0 if report["overall_pass"] else 2


def _plot_run_dir(run_dir: Path, report: dict[str, Any]) -> int:
    try:
        generated = plot_clean_metrics.plot_clean_metrics(
            metrics_jsonl=plot_clean_metrics.resolve_inputs(argparse.Namespace(run_dir=run_dir, metrics_jsonl=None, summary_json=None, output_dir=None))["metrics_jsonl"],
            summary_json=plot_clean_metrics.resolve_inputs(argparse.Namespace(run_dir=run_dir, metrics_jsonl=None, summary_json=None, output_dir=None))["summary_json"],
            output_dir=run_dir / "plots",
            no_show=True,
        )
    except ModuleNotFoundError:
        report.setdefault("plot_skipped", "matplotlib unavailable")
        return 2
    except Exception as exc:  # noqa: BLE001
        report.setdefault("plot_errors", []).append(repr(exc))
        return 1
    report["plot_paths"].extend(str(path) for path in generated)
    return 0


def _pass_fail(report: dict[str, Any]) -> dict[str, bool]:
    return_codes = report.get("return_codes", {})
    plot_codes = [code for key, code in return_codes.items() if key.startswith("plot_")]
    command_codes = [code for key, code in return_codes.items() if not key.startswith("plot_")]
    checkpoint_path = report.get("checkpoint_path")
    train_rows_ok = bool(report.get("train_metrics_nonempty"))
    invalid_rate = report.get("invalid_assignment_rate")
    return {
        "commands_zero": bool(command_codes) and all(int(code) == 0 for code in command_codes),
        "no_nan_inf": not bool(report.get("nan_or_inf_detected")),
        "checkpoint_exists": bool(checkpoint_path and Path(checkpoint_path).exists()),
        "train_metrics_nonempty": train_rows_ok,
        "invalid_assignment_recorded": invalid_rate is not None,
        "completion_fields_present": report.get("completion_rate") is not None,
        "plots_ok_or_skipped": all(int(code) in {0, 2} for code in plot_codes),
    }


def _merge_train_metrics(report: dict[str, Any], summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    report["train_metrics_nonempty"] = bool(rows)
    latest = rows[-1] if rows else {}
    recent = rows[-min(len(rows), 5) :] if rows else []
    rewards = [_number(row.get("reward")) for row in recent]
    latest_info = summary.get("latest_info", {}) if isinstance(summary, dict) else {}
    report["train_final_metrics_source"] = {
        "train_metrics_jsonl": "last row" if rows else None,
        "run_summary": "latest_info" if latest_info else None,
    }
    report["final_reward"] = _number(latest.get("reward", latest_info.get("step_reward")))
    report["recent_reward"] = float(np.mean(rewards)) if rewards else None
    report["generated_DAG_count"] = _first_not_none(latest.get("generated_DAG_count"), latest_info.get("generated_dag_count"))
    report["completed_DAG_count"] = _first_not_none(latest.get("completed_DAG_count"), latest_info.get("completed_dag_count"))
    report["completion_rate"] = _first_not_none(latest.get("DAG_completion_rate"), latest_info.get("dag_completion_rate"))
    report["throughput"] = _first_not_none(latest.get("DAG_throughput"), latest_info.get("dag_throughput"))
    report["average_DAG_flowtime"] = _first_not_none(latest.get("Average_DAG_flowtime"), latest_info.get("average_dag_flowtime"))
    report["energy_per_completed_DAG"] = _first_not_none(latest.get("Energy_per_completed_DAG"), latest_info.get("energy_per_completed_dag"))
    report["invalid_assignment_rate"] = _first_not_none(latest.get("invalid_assignment_rate"), latest_info.get("invalid_assignment_rate"))
    report["action_executed_rate"] = _first_not_none(latest.get("action_executed_rate"), latest_info.get("action_executed_rate"))
    movement = latest.get("movement_action_distribution") or latest_info.get("movement_action_distribution") or {}
    report["movement_action_distribution"] = movement if isinstance(movement, dict) else {}
    report["movement_hover_rate"] = movement.get("hover") if isinstance(movement, dict) else None
    report["offloading_action_count"] = _first_not_none(latest.get("offloading_action_count"), latest_info.get("offloading_action_count"))
    report["nan_or_inf_detected"] = _has_nan_inf(rows)
    _apply_zero_completed_metric_policy(report)


def _merge_eval_metrics(report: dict[str, Any], eval_summary: dict[str, Any]) -> None:
    report["eval_run_dir"] = eval_summary.get("run_dir")
    report["eval_final_metrics_source"] = {"eval_summary": "summary" if eval_summary else None}
    for output_key, eval_key in [
        ("generated_DAG_count", "generated_DAG_count"),
        ("completed_DAG_count", "completed_DAG_count"),
        ("completion_rate", "DAG_completion_rate"),
        ("throughput", "DAG_throughput"),
        ("average_DAG_flowtime", "Average_DAG_flowtime"),
        ("energy_per_completed_DAG", "Energy_per_completed_DAG"),
        ("invalid_assignment_rate", "invalid_assignment_rate"),
        ("action_executed_rate", "action_executed_rate"),
        ("offloading_action_count", "offloading_action_count"),
    ]:
        if eval_summary.get(eval_key) is not None:
            report[output_key] = eval_summary.get(eval_key)
    movement = eval_summary.get("movement_action_distribution") or {}
    if isinstance(movement, dict) and movement.get("hover") is not None:
        report["movement_action_distribution"] = movement
        report["movement_hover_rate"] = movement.get("hover")
    _apply_zero_completed_metric_policy(report)


def _apply_zero_completed_metric_policy(report: dict[str, Any]) -> None:
    completed = _optional_float(report.get("completed_DAG_count"))
    if completed is not None and completed <= 0.0:
        report["average_DAG_flowtime"] = None
        report["energy_per_completed_DAG"] = None


def _has_nan_inf(rows: list[dict[str, Any]]) -> bool:
    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            return any(walk(item) for item in value.values())
        if isinstance(value, list):
            return any(walk(item) for item in value)
        if isinstance(value, (int, float, np.number)):
            return not np.isfinite(float(value))
        return False

    return any(walk(row) for row in rows)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _namespace_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
