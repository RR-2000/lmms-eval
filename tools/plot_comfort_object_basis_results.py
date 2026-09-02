#!/usr/bin/env python3
"""Collect and plot COMFORT object-basis direction/vector experiments."""

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


DEFAULT_INPUT = Path("/home/ramanathan/VLM/lmms-eval/outputs/comfort_multi_3d_object_basis_0")
OUTPUT_DIR_NAME = "comfort_object_basis_analysis"
FILE_PREFIXES = (
    "comfort_multi_3d_object_basis_",
    "comfort_multi_3d_camera_basis_",
)

METRICS = (
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
    "format_accuracy",
    "object_answer_accuracy",
    "direction_answer_accuracy",
    "object_minus_direction",
    "format_parse_success",
)
METRIC_LABELS = {
    "direction_accuracy": "Direction accuracy",
    "vector_dominant_direction_accuracy": "Vector dominant-direction",
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
    "format_accuracy": "Paired-format accuracy",
    "object_answer_accuracy": "Object-answer accuracy",
    "direction_answer_accuracy": "Direction-answer accuracy",
    "object_minus_direction": "Object minus direction",
    "format_parse_success": "Paired-format parsed",
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

PAIR_OUTCOMES = (
    "both_correct",
    "object_only_correct",
    "direction_only_correct",
    "both_wrong",
    "parse_failure",
)
PAIR_LABELS = {
    "both_correct": "Object + direction correct",
    "object_only_correct": "Object correct, direction wrong",
    "direction_only_correct": "Direction correct, object wrong",
    "both_wrong": "Object + direction wrong",
    "parse_failure": "Either answer failed to parse",
}
PAIR_COLORS = {
    "both_correct": "#2A9D8F",
    "object_only_correct": "#E9C46A",
    "direction_only_correct": "#F4A261",
    "both_wrong": "#E76F51",
    "parse_failure": "#6C757D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Collect COMFORT object-basis submission JSON files and generate " "metric summaries plus 100% stacked outcome plots."))
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
            files.extend(candidate for candidate in path.rglob("*.json") if candidate.is_file() and candidate.name.startswith(FILE_PREFIXES))
        else:
            raise FileNotFoundError(path)
    unique = sorted(set(files))
    if not unique:
        raise FileNotFoundError("No COMFORT object-basis submission JSON files found")
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
    values = {str(row.get("prediction_format")) for row in rows if row.get("prediction_format") is not None}
    if metadata.get("prediction_format") is not None:
        values.add(str(metadata["prediction_format"]))
    values.discard("None")
    if len(values) == 1:
        value = next(iter(values))
        if value in {"direction", "vector", "combined", "object_direction"}:
            return value
    for value in ("object_direction", "combined", "direction", "vector"):
        if f"_basis_{value}_" in path.name:
            return value
    raise ValueError(f"Cannot infer prediction format for {path}: {sorted(values)}")


def experiment_label(path: Path, metadata: dict[str, Any], prediction_format: str, coordinate_frame: str) -> str:
    task = str(metadata.get("task", ""))
    marker = f"{task}_" if task else ""
    model = path.stem.split(marker, 1)[1] if marker and marker in path.stem else path.stem
    return f"{coordinate_frame}/{prediction_format}: {model}"


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field, 0.0)) for row in rows) / len(rows)


def summarize(path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    prediction_format = infer_format(metadata, rows, path)
    frame_values = {str(row.get("coordinate_frame")) for row in rows if row.get("coordinate_frame") is not None}
    coordinate_frame = str(metadata.get("coordinate_frame", "")).lower()
    if coordinate_frame not in {"object", "camera"} and len(frame_values) == 1:
        coordinate_frame = next(iter(frame_values)).lower()
    if coordinate_frame not in {"object", "camera"}:
        task_name = str(metadata.get("task", path.name)).lower()
        coordinate_frame = "camera" if "_camera_basis_" in task_name else "object"
    summary = {
        "label": experiment_label(path, metadata, prediction_format, coordinate_frame),
        "coordinate_frame": coordinate_frame,
        "prediction_format": prediction_format,
        "path": str(path),
        "num_records": len(rows),
        "num_scenes": len({str(row.get("scene_id")) for row in rows}),
    }
    summary.update({metric: None for metric in METRICS})
    applicable = {
        "direction": {"direction_accuracy", "direction_parse_success"},
        "vector": {metric for metric in METRICS if metric.startswith("vector_") or metric.endswith("_sign_accuracy")},
        "combined": {
            metric
            for metric in METRICS
            if metric
            not in {
                "format_accuracy",
                "object_answer_accuracy",
                "direction_answer_accuracy",
                "object_minus_direction",
                "format_parse_success",
            }
        },
        "object_direction": set(),
    }[prediction_format]
    summary.update({metric: mean(rows, metric) for metric in applicable})
    if prediction_format == "object_direction":
        object_rows = [row for row in rows if row.get("answer_format") == "object"]
        direction_rows = [row for row in rows if row.get("answer_format") == "direction"]
        if not object_rows or not direction_rows:
            raise ValueError(f"Object/direction submission must contain both answer formats: {path}")
        object_accuracy = mean(object_rows, "answer_accuracy")
        direction_accuracy = mean(direction_rows, "answer_accuracy")
        summary.update(
            {
                "num_object_answers": len(object_rows),
                "num_direction_answers": len(direction_rows),
                "format_accuracy": mean(rows, "answer_accuracy"),
                "object_answer_accuracy": object_accuracy,
                "direction_answer_accuracy": direction_accuracy,
                "object_minus_direction": object_accuracy - direction_accuracy,
                "format_parse_success": mean(rows, "parse_success"),
            }
        )
    return summary


def primary_outcome(row: dict[str, Any], prediction_format: str) -> str:
    if prediction_format == "object_direction":
        if not row.get("parse_success"):
            return "parse_failure"
        return "success" if row.get("answer_accuracy") else "incorrect"
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


def combined_outcome(row: dict[str, Any]) -> str:
    if not row.get("combined_parse_success"):
        return "parse_failure"
    for outcome in COMBINED_OUTCOMES[:-1]:
        if row.get(outcome):
            return outcome
    return "both_wrong"


def _stacked_plot(
    groups: list[tuple[str, list[dict[str, Any]], str]],
    outcomes: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    classifier,
    title: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(13, max(3.2, 1.0 + 0.85 * len(groups))))
    for y, (label, rows, prediction_format) in enumerate(groups):
        counts = Counter(classifier(row, prediction_format) for row in rows)
        left = 0.0
        for outcome in outcomes:
            width = counts[outcome] / len(rows)
            ax.barh(
                y,
                width,
                left=left,
                height=0.62,
                color=colors[outcome],
                edgecolor="white",
                linewidth=0.7,
            )
            if width >= 0.045:
                ax.text(
                    left + width / 2,
                    y,
                    f"{width:.1%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            left += width
        ax.text(-0.012, y, f"{label} (n={len(rows)})", ha="right", va="center", fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.7, len(groups) - 0.3)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of questions")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[name]) for name in outcomes]
    ax.legend(
        handles,
        [labels[name] for name in outcomes],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=min(3, len(outcomes)),
        frameon=False,
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_primary_outcomes(experiments: list[dict[str, Any]], output_dir: Path) -> Path:
    groups = [(experiment["summary"]["label"], experiment["rows"], experiment["summary"]["prediction_format"]) for experiment in experiments]
    return _stacked_plot(
        groups,
        PRIMARY_OUTCOMES,
        PRIMARY_LABELS,
        PRIMARY_COLORS,
        primary_outcome,
        "COMFORT Object-Basis Primary Outcomes",
        output_dir / "primary_outcomes_100pct.png",
    )


def plot_combined_outcomes(experiments: list[dict[str, Any]], output_dir: Path) -> Path | None:
    combined = [experiment for experiment in experiments if experiment["summary"]["prediction_format"] == "combined"]
    if not combined:
        return None
    groups = [(experiment["summary"]["label"], experiment["rows"], "combined") for experiment in combined]
    return _stacked_plot(
        groups,
        COMBINED_OUTCOMES,
        COMBINED_LABELS,
        COMBINED_COLORS,
        lambda row, _: combined_outcome(row),
        "COMFORT Combined Direction/Vector Outcomes",
        output_dir / "combined_outcomes_100pct.png",
    )


def _paired_format_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row.get("pair_id")), {})[str(row.get("answer_format"))] = row
    pairs = []
    for pair in grouped.values():
        if not {"object", "direction"} <= set(pair):
            continue
        pairs.append(
            {
                "parse_success": float(bool(pair["object"].get("parse_success")) and bool(pair["direction"].get("parse_success"))),
                "object_correct": float(pair["object"].get("answer_accuracy", 0.0)),
                "direction_correct": float(pair["direction"].get("answer_accuracy", 0.0)),
            }
        )
    return pairs


def pair_outcome(row: dict[str, Any], _: str) -> str:
    if not row.get("parse_success"):
        return "parse_failure"
    object_correct = bool(row.get("object_correct"))
    direction_correct = bool(row.get("direction_correct"))
    if object_correct and direction_correct:
        return "both_correct"
    if object_correct:
        return "object_only_correct"
    if direction_correct:
        return "direction_only_correct"
    return "both_wrong"


def plot_object_direction_outcomes(experiments: list[dict[str, Any]], output_dir: Path) -> Path | None:
    paired = [experiment for experiment in experiments if experiment["summary"]["prediction_format"] == "object_direction"]
    if not paired:
        return None
    groups = [
        (
            experiment["summary"]["label"],
            _paired_format_rows(experiment["rows"]),
            "object_direction",
        )
        for experiment in paired
    ]
    return _stacked_plot(
        groups,
        PAIR_OUTCOMES,
        PAIR_LABELS,
        PAIR_COLORS,
        pair_outcome,
        "COMFORT Object-vs-Direction Paired Outcomes",
        output_dir / "object_direction_outcomes_100pct.png",
    )


def plot_object_vs_direction_accuracy(summaries: list[dict[str, Any]], output_dir: Path) -> Path | None:
    paired = [summary for summary in summaries if summary["prediction_format"] == "object_direction"]
    if not paired:
        return None

    x_positions = list(range(len(paired)))
    width = 0.34
    object_scores = [float(row["object_answer_accuracy"]) for row in paired]
    direction_scores = [float(row["direction_answer_accuracy"]) for row in paired]
    fig, ax = plt.subplots(figsize=(max(8, 2.8 * len(paired)), 5.8))
    object_bars = ax.bar(
        [x - width / 2 for x in x_positions],
        object_scores,
        width,
        label="Answer with object",
        color="#457B9D",
    )
    direction_bars = ax.bar(
        [x + width / 2 for x in x_positions],
        direction_scores,
        width,
        label="Answer with direction",
        color="#E9C46A",
    )
    for bars in (object_bars, direction_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.025, value + 0.018),
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x_positions, [row["label"] for row in paired], rotation=12)
    ax.set_ylim(0.0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Accuracy")
    ax.set_title("COMFORT Answer-with-Object vs Answer-with-Direction Accuracy")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "object_vs_direction_accuracy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_summary(summaries: list[dict[str, Any]], output_dir: Path) -> Path | None:
    summaries = [summary for summary in summaries if summary["prediction_format"] != "object_direction"]
    if not summaries:
        return None
    selected = (
        "direction_accuracy",
        "vector_dominant_direction_accuracy",
        "vector_angle_30_accuracy",
        "vector_cosine",
        "full_sign_accuracy",
    )
    width = 0.8 / max(1, len(summaries))
    x_positions = list(range(len(selected)))
    fig, ax = plt.subplots(figsize=(13, 6))
    for index, summary in enumerate(summaries):
        offset = (index - (len(summaries) - 1) / 2) * width
        values = [float(summary[name]) if summary[name] is not None else math.nan for name in selected]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=summary["label"],
        )
        for bar, value in zip(bars, values):
            if math.isfinite(value) and value > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(1.03, value + 0.015),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
    ax.set_xticks(x_positions, [METRIC_LABELS[name] for name in selected], rotation=18)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("COMFORT Object-Basis Metric Summary")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "metric_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_tables(summaries: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "summary.csv"
    fields = [
        "label",
        "coordinate_frame",
        "prediction_format",
        "path",
        "num_records",
        "num_scenes",
        *METRICS,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in summaries)
    markdown_path = output_dir / "summary.md"
    lines = [
        "# COMFORT object-basis results",
        "",
        "| Experiment | n | Direction | Vector dominant | Cosine | Within 30° | Full sign | Pair overall | Object answer | Direction answer | Object−direction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def formatted(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.3f}"

    for row in summaries:
        lines.append(
            f"| {row['label']} | {row['num_records']} | {formatted(row['direction_accuracy'])} | "
            f"{formatted(row['vector_dominant_direction_accuracy'])} | {formatted(row['vector_cosine'])} | "
            f"{formatted(row['vector_angle_30_accuracy'])} | {formatted(row['full_sign_accuracy'])} | "
            f"{formatted(row['format_accuracy'])} | {formatted(row['object_answer_accuracy'])} | "
            f"{formatted(row['direction_answer_accuracy'])} | {formatted(row['object_minus_direction'])} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, csv_path, markdown_path]


def write_object_direction_report(summaries: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    paired = [summary for summary in summaries if summary["prediction_format"] == "object_direction"]
    if not paired:
        return []

    rows = [
        {
            "experiment": summary["label"],
            "coordinate_frame": summary["coordinate_frame"],
            "num_object_answers": summary["num_object_answers"],
            "num_direction_answers": summary["num_direction_answers"],
            "object_answer_accuracy": summary["object_answer_accuracy"],
            "direction_answer_accuracy": summary["direction_answer_accuracy"],
            "object_minus_direction": summary["object_minus_direction"],
        }
        for summary in paired
    ]
    csv_path = output_dir / "object_vs_direction_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "object_vs_direction_accuracy.md"
    lines = [
        "# Object-answer vs direction-answer accuracy",
        "",
        "| Experiment | Object n | Object accuracy | Direction n | Direction accuracy | Object−direction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['num_object_answers']} | " f"{row['object_answer_accuracy']:.1%} | {row['num_direction_answers']} | " f"{row['direction_answer_accuracy']:.1%} | " f"{row['object_minus_direction']:+.1%} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, markdown_path]


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
        raise ValueError("No valid object-basis submissions were loaded:\n" + "\n".join(errors))
    experiments.sort(
        key=lambda item: (
            item["summary"]["coordinate_frame"],
            item["summary"]["prediction_format"],
            item["summary"]["label"],
        )
    )
    output_dir = resolve_output_dir(args.inputs, args.output_dir)
    summaries = [experiment["summary"] for experiment in experiments]
    outputs = [
        *write_tables(summaries, output_dir),
        *write_object_direction_report(summaries, output_dir),
    ]
    for output in (
        plot_metric_summary(summaries, output_dir),
        plot_primary_outcomes(experiments, output_dir),
        plot_combined_outcomes(experiments, output_dir),
        plot_object_direction_outcomes(experiments, output_dir),
        plot_object_vs_direction_accuracy(summaries, output_dir),
    ):
        if output is not None:
            outputs.append(output)
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
