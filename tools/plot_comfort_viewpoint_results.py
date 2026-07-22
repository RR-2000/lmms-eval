#!/usr/bin/env python3
"""Plot COMFORT viewpoint metrics and summarize likely error modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(
    "/home/ramanathan/VLM/lmms-eval/outputs/comfort_viewpoint_0/"
    "Qwen__Qwen3-VL-4B-Instruct/20260722_115910_results.json"
)
TASK = "comfort_viewpoint"

OUTCOME_METRICS = [
    "comfort_answer_and_direction_correct",
    "comfort_answer_correct_direction_wrong",
    "comfort_answer_wrong_direction_correct",
    "comfort_answer_and_direction_wrong",
]
OUTCOME_LABELS = {
    OUTCOME_METRICS[0]: "Answer + direction correct",
    OUTCOME_METRICS[1]: "Answer correct, direction wrong",
    OUTCOME_METRICS[2]: "Answer wrong, direction correct",
    OUTCOME_METRICS[3]: "Answer + direction wrong",
}
OUTCOME_COLORS = ["#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]

METRIC_LABELS = {
    "comfort_answer_accuracy": "Overall answer",
    "comfort_relation_axis_accuracy": "Relation-axis",
    "comfort_vector_cosine": "Vector cosine",
    "comfort_camera_answer_accuracy": "Camera question",
    "comfort_reference_answer_accuracy": "Reference question",
    "comfort_addressee_answer_accuracy": "Addressee question",
}
SUMMARY_METRICS = [
    "comfort_answer_accuracy",
    "comfort_relation_axis_accuracy",
    "comfort_vector_cosine",
    "comfort_camera_answer_accuracy",
    "comfort_reference_answer_accuracy",
    "comfort_addressee_answer_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot COMFORT viewpoint metrics and infer failure modes."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--task", default=TASK, help="Task key inside the result JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots and reports; defaults next to the input JSON.",
    )
    return parser.parse_args()


def load_results(path: Path, task: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    results = payload.get("results", {}).get(task)
    if not isinstance(results, dict):
        raise KeyError(f"Task '{task}' not found in {path}")
    return results


def metric(results: dict[str, Any], name: str) -> float:
    value = results.get(f"{name},none")
    if not isinstance(value, (int, float)):
        raise KeyError(f"Metric '{name}' is missing or non-numeric")
    return float(value)


def output_dir(input_path: Path, requested: Path | None) -> Path:
    directory = requested or input_path.parent / f"{input_path.stem}_analysis"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def annotate_bars(ax: Any, bars: Any) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_summary(values: dict[str, float], directory: Path) -> Path:
    labels = [METRIC_LABELS[name] for name in SUMMARY_METRICS]
    scores = [values[name] for name in SUMMARY_METRICS]
    colors = ["#457B9D", "#8D99AE", "#3A86FF", "#2A9D8F", "#E9C46A", "#F4A261"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, scores, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("COMFORT Viewpoint Metric Summary")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, bars)
    fig.tight_layout()
    path = directory / "metric_summary.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_outcomes(values: dict[str, float], directory: Path) -> list[Path]:
    scores = [values[name] for name in OUTCOME_METRICS]
    labels = [OUTCOME_LABELS[name] for name in OUTCOME_METRICS]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, scores, color=OUTCOME_COLORS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Answer/Direction Outcome Breakdown")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, bars)
    fig.tight_layout()
    bar_path = directory / "outcome_breakdown.png"
    fig.savefig(bar_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.pie(
        scores,
        labels=labels,
        colors=OUTCOME_COLORS,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=110,
        counterclock=False,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    ax.set_title("Overall Outcome Split")
    fig.tight_layout()
    pie_path = directory / "outcome_split.png"
    fig.savefig(pie_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [bar_path, pie_path]


def plot_question_types(values: dict[str, float], directory: Path) -> Path:
    names = [
        "comfort_camera_answer_accuracy",
        "comfort_reference_answer_accuracy",
        "comfort_addressee_answer_accuracy",
    ]
    labels = [METRIC_LABELS[name] for name in names]
    scores = [values[name] for name in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, scores, color=["#457B9D", "#E9C46A", "#F4A261"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Answer accuracy")
    ax.set_title("COMFORT Question-Type Accuracy")
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, bars)
    fig.tight_layout()
    path = directory / "question_type_accuracy.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def infer(values: dict[str, float]) -> list[str]:
    notes = []
    answer = values["comfort_answer_accuracy"]
    direction = values["comfort_relation_axis_accuracy"]
    vector = values["comfort_vector_cosine"]
    if answer - direction > 0.10:
        notes.append(f"Answer accuracy ({answer:.3f}) exceeds relation-axis accuracy ({direction:.3f}), suggesting weak spatial consistency after choosing an answer.")
    if values["comfort_answer_correct_direction_wrong"] > values["comfort_answer_wrong_direction_correct"] + 0.05:
        notes.append("The dominant asymmetric error is correct answer with wrong direction, consistent with answer priors or shortcut reasoning.")
    if vector < 0.10:
        notes.append(f"Vector cosine is low ({vector:.3f}), indicating poor continuous geometric alignment.")
    question_scores = {name: values[name] for name in (
        "comfort_camera_answer_accuracy", "comfort_reference_answer_accuracy", "comfort_addressee_answer_accuracy")}
    hardest = min(question_scores, key=question_scores.get)
    notes.append(f"The weakest question type is {METRIC_LABELS[hardest].lower()} ({question_scores[hardest]:.3f}).")
    return notes


def write_report(values: dict[str, float], directory: Path, input_path: Path) -> Path:
    lines = ["# COMFORT Viewpoint Result Analysis", "", f"Input: `{input_path}`", "", "## Metrics", ""]
    lines.extend(f"- `{name}`: `{values[name]:.6f}`" for name in SUMMARY_METRICS + OUTCOME_METRICS)
    lines.extend(["", "## Inferences", ""])
    lines.extend(f"- {note}" for note in infer(values))
    path = directory / "inference_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    args = parse_args()
    results = load_results(args.input, args.task)
    names = SUMMARY_METRICS + OUTCOME_METRICS
    values = {name: metric(results, name) for name in names}
    directory = output_dir(args.input, args.output_dir)
    outputs = [plot_summary(values, directory), *plot_outcomes(values, directory), plot_question_types(values, directory), write_report(values, directory, args.input)]
    print("Generated analysis artifacts:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
