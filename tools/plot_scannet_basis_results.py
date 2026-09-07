#!/usr/bin/env python3
"""Collect and plot ScanNet camera/object-basis experiment submissions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DEFAULT_INPUT = Path("/home/ramanathan/VLM/lmms-eval/outputs/scannet_basis_all_8")
OUTPUT_DIR_NAME = "scannet_basis_analysis"
FILE_PREFIXES = ("scannet_camera_basis_", "scannet_object_basis_")
FORMATS = ("direction", "vector", "combined")
FRAME_ORDER = ("camera", "object_facing_camera")
FRAME_LABELS = {
    "camera": "Camera frame",
    "object_facing_camera": "Object facing camera",
}

METRICS = (
    "raw_accuracy",
    "direction_accuracy",
    "vector_dominant_direction_accuracy",
    "vector_cosine",
    "vector_l2_score",
    "vector_angle_30_accuracy",
    "front_sign_accuracy",
    "up_sign_accuracy",
    "right_sign_accuracy",
    "full_sign_accuracy",
    "direction_parse_success",
    "vector_parse_success",
    "combined_parse_success",
)
METRIC_LABELS = {
    "raw_accuracy": "Raw accuracy",
    "direction_accuracy": "Direction accuracy",
    "vector_dominant_direction_accuracy": "Vector dominant direction",
    "vector_cosine": "Vector cosine",
    "vector_l2_score": "Unit-vector L2 score",
    "vector_angle_30_accuracy": "Within 30 degrees",
    "front_sign_accuracy": "Front sign",
    "up_sign_accuracy": "Up sign",
    "right_sign_accuracy": "Right sign",
    "full_sign_accuracy": "All signs",
    "direction_parse_success": "Direction parsed",
    "vector_parse_success": "Vector parsed",
    "combined_parse_success": "Both parsed",
}

PRIMARY_OUTCOMES = ("success", "incorrect", "parse_failure")
PRIMARY_LABELS = {
    "success": "Primary prediction correct",
    "incorrect": "Parsed but incorrect",
    "parse_failure": "Parse failure / zero vector",
}
PRIMARY_COLORS = {
    "success": "#2A9D8F",
    "incorrect": "#E76F51",
    "parse_failure": "#6C757D",
}
COMBINED_OUTCOMES = (
    "both_correct",
    "direction_correct_vector_wrong",
    "direction_wrong_vector_correct",
    "both_wrong",
    "parse_failure",
)
COMBINED_LABELS = {
    "both_correct": "Direction + vector correct",
    "direction_correct_vector_wrong": "Direction correct, vector wrong",
    "direction_wrong_vector_correct": "Direction wrong, vector correct",
    "both_wrong": "Direction + vector wrong",
    "parse_failure": "Either output failed to parse",
}
COMBINED_COLORS = {
    "both_correct": "#2A9D8F",
    "direction_correct_vector_wrong": "#E9C46A",
    "direction_wrong_vector_correct": "#F4A261",
    "both_wrong": "#E76F51",
    "parse_failure": "#6C757D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect ScanNet basis submission JSON files and generate metric "
            "tables, raw-accuracy bars, and 100% stacked outcome plots."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[DEFAULT_INPUT],
        help="Submission JSON files or directories searched recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults below the first input directory.",
    )
    return parser.parse_args()


def discover(inputs: Iterable[Path]) -> list[Path]:
    files = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*.json")
                if candidate.is_file() and candidate.name.startswith(FILE_PREFIXES)
            )
        else:
            raise FileNotFoundError(path)
    unique = sorted(set(files))
    if not unique:
        raise FileNotFoundError("No ScanNet basis submission JSON files found")
    return unique


def load_submission(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        records = payload.get("records")
        metadata = payload
    elif isinstance(payload, list):
        records = payload
        metadata = {}
    else:
        raise ValueError(f"Unsupported JSON structure: {path}")
    if not isinstance(records, list) or not records:
        raise ValueError(f"No sample records in {path}")
    rows = [row for row in records if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No dictionary sample records in {path}")
    return metadata, rows


def infer_format(metadata: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> str:
    values = {str(row.get("prediction_format")) for row in rows if row.get("prediction_format")}
    if metadata.get("prediction_format"):
        values.add(str(metadata["prediction_format"]))
    valid = values & set(FORMATS)
    if len(valid) == 1:
        return next(iter(valid))
    for value in FORMATS:
        if f"_basis_{value}_" in path.name:
            return value
    raise ValueError(f"Cannot infer prediction format for {path}: {sorted(values)}")


def infer_frame(metadata: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> str:
    values = {str(row.get("coordinate_frame")) for row in rows if row.get("coordinate_frame")}
    if values == {"camera"}:
        return "camera"
    if values == {"object_facing_camera"}:
        return "object_facing_camera"
    task = str(metadata.get("task", path.name)).lower()
    if "scannet_camera_basis_" in task or path.name.startswith("scannet_camera_basis_"):
        return "camera"
    if "scannet_object_basis_" in task or path.name.startswith("scannet_object_basis_"):
        return "object_facing_camera"
    raise ValueError(f"Cannot infer coordinate frame for {path}: {sorted(values)}")


def model_name(metadata: dict[str, Any], path: Path) -> str:
    task = str(metadata.get("task", ""))
    marker = f"{task}_" if task else ""
    return path.stem.split(marker, 1)[1] if marker and marker in path.stem else path.stem


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field, 0.0)) for row in rows) / len(rows)


def summarize(path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    prediction_format = infer_format(metadata, rows, path)
    coordinate_frame = infer_frame(metadata, rows, path)
    model = model_name(metadata, path)
    result: dict[str, Any] = {
        "label": f"{FRAME_LABELS[coordinate_frame]}/{prediction_format}: {model}",
        "model": model,
        "coordinate_frame": coordinate_frame,
        "prediction_format": prediction_format,
        "path": str(path),
        "num_records": len(rows),
        "num_scenes": len({str(row.get("scene_id")) for row in rows}),
        **{metric: None for metric in METRICS},
    }
    applicable = {
        "direction": {"direction_accuracy", "direction_parse_success"},
        "vector": {
            "vector_dominant_direction_accuracy",
            "vector_cosine",
            "vector_l2_score",
            "vector_angle_30_accuracy",
            "front_sign_accuracy",
            "up_sign_accuracy",
            "right_sign_accuracy",
            "full_sign_accuracy",
            "vector_parse_success",
        },
        "combined": set(METRICS) - {"raw_accuracy"},
    }[prediction_format]
    result.update({metric: mean(rows, metric) for metric in applicable})
    result["raw_accuracy"] = {
        "direction": result["direction_accuracy"],
        "vector": result["vector_dominant_direction_accuracy"],
        "combined": mean(rows, "both_correct"),
    }[prediction_format]
    return result


def primary_outcome(row: dict[str, Any], prediction_format: str) -> str:
    if prediction_format == "direction":
        if not row.get("direction_parse_success"):
            return "parse_failure"
        return "success" if row.get("direction_accuracy") else "incorrect"
    if prediction_format == "vector":
        if not row.get("vector_parse_success"):
            return "parse_failure"
        return "success" if row.get("vector_dominant_direction_accuracy") else "incorrect"
    if not row.get("combined_parse_success"):
        return "parse_failure"
    return "success" if row.get("both_correct") else "incorrect"


def combined_outcome(row: dict[str, Any], _: str = "combined") -> str:
    if not row.get("combined_parse_success"):
        return "parse_failure"
    for outcome in COMBINED_OUTCOMES[:-1]:
        if row.get(outcome):
            return outcome
    return "both_wrong"


def _sort_key(experiment: dict[str, Any]) -> tuple[int, int, str]:
    summary = experiment["summary"]
    return (
        FRAME_ORDER.index(summary["coordinate_frame"]),
        FORMATS.index(summary["prediction_format"]),
        summary["model"],
    )


def _stacked_plot(
    groups: list[tuple[str, list[dict[str, Any]], str]],
    outcomes: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    classifier,
    title: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(13, max(3.5, 1.1 + 0.82 * len(groups))))
    for y, (label, rows, prediction_format) in enumerate(groups):
        counts = Counter(classifier(row, prediction_format) for row in rows)
        left = 0.0
        for outcome in outcomes:
            width = counts[outcome] / len(rows)
            ax.barh(y, width, left=left, height=0.62, color=colors[outcome], edgecolor="white", linewidth=0.7)
            if width >= 0.045:
                ax.text(left + width / 2, y, f"{width:.1%}", ha="center", va="center", fontsize=9)
            left += width
        ax.text(-0.012, y, f"{label} (n={len(rows)})", ha="right", va="center", fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.7, len(groups) - 0.3)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of questions")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[name]) for name in outcomes]
    ax.legend(handles, [labels[name] for name in outcomes], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(3, len(outcomes)), frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_primary_outcomes(experiments: list[dict[str, Any]], output_dir: Path) -> Path:
    groups = [(item["summary"]["label"], item["rows"], item["summary"]["prediction_format"]) for item in experiments]
    return _stacked_plot(
        groups,
        PRIMARY_OUTCOMES,
        PRIMARY_LABELS,
        PRIMARY_COLORS,
        primary_outcome,
        "ScanNet Basis Primary Outcomes",
        output_dir / "primary_outcomes_100pct.png",
    )


def plot_combined_outcomes(experiments: list[dict[str, Any]], output_dir: Path) -> Path | None:
    combined = [item for item in experiments if item["summary"]["prediction_format"] == "combined"]
    if not combined:
        return None
    groups = [(item["summary"]["label"], item["rows"], "combined") for item in combined]
    return _stacked_plot(
        groups,
        COMBINED_OUTCOMES,
        COMBINED_LABELS,
        COMBINED_COLORS,
        combined_outcome,
        "ScanNet Combined Direction/Vector Outcomes",
        output_dir / "combined_outcomes_100pct.png",
    )


def plot_raw_accuracy(summaries: list[dict[str, Any]], output_dir: Path) -> Path:
    labels = [row["label"] for row in summaries]
    values = [float(row["raw_accuracy"]) for row in summaries]
    colors = ["#457B9D" if row["coordinate_frame"] == "camera" else "#E9C46A" for row in summaries]
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.65 * len(summaries) + 1.5)))
    y = list(range(len(summaries)))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Raw accuracy")
    ax.set_title("ScanNet Basis Raw Accuracy")
    ax.grid(axis="x", alpha=0.2)
    for index, value in enumerate(values):
        kwargs = {"ha": "right", "x": value - 0.012} if value > 0.9 else {"ha": "left", "x": value + 0.012}
        ax.text(kwargs.pop("x"), index, f"{value:.1%}", va="center", fontsize=9, **kwargs)
    fig.tight_layout()
    path = output_dir / "raw_accuracy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_summary(summaries: list[dict[str, Any]], output_dir: Path) -> Path:
    selected = (
        "direction_accuracy",
        "vector_dominant_direction_accuracy",
        "vector_angle_30_accuracy",
        "vector_cosine",
        "full_sign_accuracy",
    )
    width = 0.8 / max(1, len(summaries))
    x_positions = list(range(len(selected)))
    plotted_values = []
    fig, ax = plt.subplots(figsize=(13, 6))
    for index, summary in enumerate(summaries):
        offset = (index - (len(summaries) - 1) / 2) * width
        values = [float(summary[name]) if summary[name] is not None else math.nan for name in selected]
        plotted_values.extend(value for value in values if math.isfinite(value))
        ax.bar([x + offset for x in x_positions], values, width=width, label=summary["label"])
    ax.set_xticks(x_positions, [METRIC_LABELS[name] for name in selected], rotation=18)
    lower_bound = min(-0.1, min(plotted_values, default=0.0) - 0.05)
    ax.set_ylim(lower_bound, 1.05)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_ylabel("Score")
    ax.set_title("ScanNet Basis Metric Summary")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "metric_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_frame_comparison(summaries: list[dict[str, Any]], output_dir: Path) -> Path:
    models = sorted({row["model"] for row in summaries})
    groups = [(model, prediction_format) for model in models for prediction_format in FORMATS]
    x = list(range(len(groups)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(10, 2.0 * len(groups)), 6))
    for offset_index, frame in enumerate(FRAME_ORDER):
        values = []
        for model, prediction_format in groups:
            match = next((row for row in summaries if row["model"] == model and row["prediction_format"] == prediction_format and row["coordinate_frame"] == frame), None)
            values.append(float(match["raw_accuracy"]) if match else math.nan)
        offset = (-0.5 if offset_index == 0 else 0.5) * width
        bars = ax.bar([value + offset for value in x], values, width, label=FRAME_LABELS[frame])
        for bar, score in zip(bars, values):
            if math.isfinite(score):
                ax.text(bar.get_x() + bar.get_width() / 2, min(1.02, score + 0.015), f"{score:.1%}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x, [f"{model}\n{prediction_format}" for model, prediction_format in groups])
    ax.set_ylim(0.0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Raw accuracy")
    ax.set_title("ScanNet Camera vs Object-Facing-Camera Frames")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "camera_vs_object_frame_accuracy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_tables(summaries: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    fields = ["label", "model", "coordinate_frame", "prediction_format", "path", "num_records", "num_scenes", *METRICS]
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in summaries)

    def formatted(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.3f}"

    lines = [
        "# ScanNet basis results",
        "",
        "| Experiment | n | Raw accuracy | Direction | Vector dominant | Cosine | Within 30° | Full sign | Direction parsed | Vector parsed | Both parsed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['label']} | {row['num_records']} | {formatted(row['raw_accuracy'])} | "
            f"{formatted(row['direction_accuracy'])} | {formatted(row['vector_dominant_direction_accuracy'])} | "
            f"{formatted(row['vector_cosine'])} | {formatted(row['vector_angle_30_accuracy'])} | "
            f"{formatted(row['full_sign_accuracy'])} | {formatted(row['direction_parse_success'])} | "
            f"{formatted(row['vector_parse_success'])} | {formatted(row['combined_parse_success'])} |"
        )
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, csv_path, markdown_path]


def resolve_output_dir(inputs: list[Path], requested: Path | None) -> Path:
    if requested is not None:
        result = requested.expanduser().resolve()
    else:
        first = inputs[0].expanduser().resolve()
        result = (first if first.is_dir() else first.parent) / OUTPUT_DIR_NAME
    result.mkdir(parents=True, exist_ok=True)
    return result


def main() -> int:
    args = parse_args()
    paths = discover(args.inputs)
    experiments = []
    errors = []
    for path in paths:
        try:
            metadata, rows = load_submission(path)
            summary = summarize(path, metadata, rows)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{path}: {error}")
            continue
        experiments.append({"summary": summary, "rows": rows})
    if not experiments:
        raise ValueError("No valid ScanNet basis submissions were loaded:\n" + "\n".join(errors))
    experiments.sort(key=_sort_key)
    summaries = [item["summary"] for item in experiments]
    output_dir = resolve_output_dir(args.inputs, args.output_dir)
    outputs = [
        *write_tables(summaries, output_dir),
        plot_raw_accuracy(summaries, output_dir),
        plot_metric_summary(summaries, output_dir),
        plot_primary_outcomes(experiments, output_dir),
        plot_frame_comparison(summaries, output_dir),
    ]
    combined_path = plot_combined_outcomes(experiments, output_dir)
    if combined_path is not None:
        outputs.append(combined_path)
    print(f"Loaded {len(experiments)} submissions ({sum(row['num_records'] for row in summaries)} records).")
    for output in outputs:
        print(output)
    if errors:
        print("Skipped unrelated/invalid files:")
        for error in errors:
            print(f"- {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
