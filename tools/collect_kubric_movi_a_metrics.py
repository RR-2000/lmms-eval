#!/usr/bin/env python3
"""Collect Kubric MOVi-A result metrics as spreadsheet-ready TSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs")

# The default set is intentionally compact: headline correctness, geometry,
# and the combined score.  Use --all-metrics to emit every non-stderr metric.
CORE_METRICS = (
    "viewpoint_answer_accuracy",
    "viewpoint_axis_sign_accuracy",
    "viewpoint_answer_and_direction_correct",
    "viewpoint_reference_to_camera_cosine",
    "viewpoint_reference_to_camera_distance_score",
    "viewpoint_vector_cosine",
    "viewpoint_scale_score",
    "viewpoint_combined_score",
    "camera_relative_position_viewpoint_combined_score",
    "camera_distance_viewpoint_combined_score",
    "height_relative_3d_viewpoint_combined_score",
    "object_centric_relative_position_viewpoint_combined_score",
    "object_centric_relative_position_multi_viewpoint_combined_score",
    "object_centric_direction_binary_viewpoint_combined_score",
    "object_centric_camera_pose_viewpoint_combined_score",
    "easy_viewpoint_combined_score",
    "hard_viewpoint_combined_score",
    "very_hard_viewpoint_combined_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print Kubric MOVi-A result metrics as tab-separated values."
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        type=Path,
        help="Result JSON files or directories. Defaults to outputs/kubric_movi_a_*/.",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Include every non-stderr, non-empty metric instead of core metrics.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal places for numeric values (default: 6).",
    )
    return parser.parse_args()


def result_files(paths: list[Path]) -> list[Path]:
    if not paths:
        paths = [DEFAULT_ROOT]

    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            files.add(path)
        elif path.is_dir():
            files.update(path.rglob("*_results.json"))
    return sorted(files)


def metric_name(key: str) -> str:
    """Remove lmms-eval's aggregation suffix from a result key."""
    return key.removesuffix(",none")


def format_value(value: Any, precision: int) -> Any:
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, (dict, list)):
        return ""
    return value


def load_rows(path: Path, all_metrics: bool, precision: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    model = payload.get("model_name") or payload.get("model_source", "")
    # run = payload.get("date") or path.stem.removesuffix("_results")
    run = payload.get("config").get("resolved_cli_args").get("output_path").split("/")[-1]

    rows: list[dict[str, Any]] = []
    discovered: set[str] = set()
    for task, raw_metrics in results.items():
        metrics: dict[str, Any] = {}
        for key, value in raw_metrics.items():
            name = metric_name(key)
            if name == "alias" or "stderr" in name or name == "submission":
                continue
            if value in ([], None) or isinstance(value, (dict, list)):
                continue
            if all_metrics or name in CORE_METRICS:
                metrics[name] = format_value(value, precision)
                discovered.add(name)
        rows.append({"model": model, "run": run, "task": task, **metrics})

    # Keep a stable core-metric column order; full output is alphabetical.
    ordered = [m for m in CORE_METRICS if m in discovered]
    if all_metrics:
        ordered = sorted(discovered)
    for row in rows:
        for name in ordered:
            row.setdefault(name, "")
    return {"rows": rows, "metrics": ordered}


def main() -> int:
    args = parse_args()
    if args.precision < 0:
        print("--precision must be non-negative", file=sys.stderr)
        return 2

    files = result_files(args.paths)
    if not files:
        print("No *_results.json files found.", file=sys.stderr)
        return 1

    all_rows: list[dict[str, Any]] = []
    metric_names: set[str] = set()
    for path in files:
        loaded = load_rows(path, args.all_metrics, args.precision)
        all_rows.extend(loaded["rows"])
        metric_names.update(loaded["metrics"])

    metrics = (
        sorted(metric_names)
        if args.all_metrics
        else [m for m in CORE_METRICS if m in metric_names]
    )
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["model", "run", "task", *metrics],
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
