#!/usr/bin/env python3
"""Plot COMFORT viewpoint metrics and summarize likely error modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


DEFAULT_INPUT = Path(
    "/home/ramanathan/VLM/lmms-eval/outputs/comfort_viewpoint_0/Qwen__Qwen3-VL-4B-Instruct/20260722_142023_results.json"
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
INVALID_DIRECTION_OUTCOME = "invalid_zero_direction"
INVALID_DIRECTION_LABEL = "Invalid direction (0, 0, 0)"
INVALID_DIRECTION_COLOR = "#6C757D"

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
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="Sample-level submission JSON. Auto-detected from the input run when omitted.",
    )
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


def find_submission(input_path: Path, requested: Path | None) -> Path | None:
    """Find the sample-level submission written by the COMFORT aggregator."""
    if requested is not None:
        return requested
    submission_dir = input_path.parent.parent / "submissions"
    candidates = sorted(submission_dir.glob("comfort_viewpoint_*.json"))
    return candidates[-1] if candidates else None


def load_samples(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of sample records in {path}")
    return [row for row in payload if isinstance(row, dict)]


def sample_outcome(row: dict[str, Any]) -> str:
    if row.get("answer_and_direction_correct"):
        return OUTCOME_METRICS[0]
    if row.get("answer_correct_direction_wrong"):
        return OUTCOME_METRICS[1]
    if row.get("answer_wrong_direction_correct"):
        return OUTCOME_METRICS[2]
    return OUTCOME_METRICS[3]


def has_invalid_zero_direction(row: dict[str, Any]) -> bool:
    """Return whether the submitted direction vector is exactly (0, 0, 0)."""
    prediction = row.get("parsed_prediction")
    if not isinstance(prediction, dict):
        return False
    vector = prediction.get("between_objects", prediction.get("relative_vector", {}))
    if not isinstance(vector, dict):
        return False
    try:
        return all(abs(float(vector.get(axis, 0.0))) <= 1e-8 for axis in ("right", "up", "front"))
    except (TypeError, ValueError):
        return False


def sample_outcome_with_invalid_direction(row: dict[str, Any]) -> str:
    """Reserve a mutually exclusive outcome class for invalid zero vectors."""
    return INVALID_DIRECTION_OUTCOME if has_invalid_zero_direction(row) else sample_outcome(row)


def plot_split_rectangles(samples: list[dict[str, Any]], directory: Path) -> Path | None:
    """Draw proportional colored rectangles for each question viewpoint."""
    if not samples:
        return None
    groups = [("All questions", samples)]
    for viewpoint, label in (("camera", "Camera"), ("reference", "Reference"), ("addressee", "Addressee")):
        rows = [row for row in samples if row.get("viewpoint") == viewpoint]
        if rows:
            groups.append((label, rows))

    fig, ax = plt.subplots(figsize=(12, 1.25 + 0.8 * len(groups)))
    for y, (label, rows) in enumerate(groups):
        counts = {name: sum(sample_outcome(row) == name for row in rows) for name in OUTCOME_METRICS}
        total = len(rows)
        left = 0.0
        for name, color in zip(OUTCOME_METRICS, OUTCOME_COLORS):
            width = counts[name] / total
            ax.barh(y, width, left=left, height=0.62, color=color, edgecolor="white", linewidth=0.6)
            if width >= 0.045:
                ax.text(left + width / 2, y, f"{width:.1%}", ha="center", va="center", fontsize=9)
            left += width
        ax.text(-0.012, y, f"{label} (n={total})", ha="right", va="center", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(groups) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of questions")
    ax.set_title("COMFORT Answer/Direction Ratios")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in OUTCOME_COLORS]
    ax.legend(handles, [OUTCOME_LABELS[name] for name in OUTCOME_METRICS],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = directory / "answer_direction_split_rectangles.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_split_rectangles_with_invalid_direction(
    samples: list[dict[str, Any]], directory: Path
) -> Path | None:
    """Draw 100% outcome rectangles with zero direction vectors shown separately."""
    if not samples:
        return None
    groups = [("All questions", samples)]
    for viewpoint, label in (("camera", "Camera"), ("reference", "Reference"), ("addressee", "Addressee")):
        rows = [row for row in samples if row.get("viewpoint") == viewpoint]
        if rows:
            groups.append((label, rows))

    outcomes = [*OUTCOME_METRICS, INVALID_DIRECTION_OUTCOME]
    labels = {**OUTCOME_LABELS, INVALID_DIRECTION_OUTCOME: INVALID_DIRECTION_LABEL}
    colors = [*OUTCOME_COLORS, INVALID_DIRECTION_COLOR]
    fig, ax = plt.subplots(figsize=(12, 1.25 + 0.8 * len(groups)))
    for y, (label, rows) in enumerate(groups):
        counts = {name: sum(sample_outcome_with_invalid_direction(row) == name for row in rows) for name in outcomes}
        total = len(rows)
        left = 0.0
        for name, color in zip(outcomes, colors):
            width = counts[name] / total
            ax.barh(y, width, left=left, height=0.62, color=color, edgecolor="white", linewidth=0.6)
            if width >= 0.045:
                ax.text(left + width / 2, y, f"{width:.1%}", ha="center", va="center", fontsize=9)
            left += width
        ax.text(-0.012, y, f"{label} (n={total})", ha="right", va="center", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(groups) - 0.3)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("Share of questions")
    ax.set_title("COMFORT Outcomes (Zero Directions Shown as Invalid)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax.legend(handles, [labels[name] for name in outcomes],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = directory / "answer_direction_split_rectangles_with_invalid_zero_direction.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_question_outcome_strip(samples: list[dict[str, Any]], directory: Path) -> Path | None:
    """Show one colored cell per question, ordered by viewpoint."""
    if not samples:
        return None
    ordered = sorted(samples, key=lambda row: (str(row.get("viewpoint", "")), str(row.get("qid", ""))))
    colors = {name: color for name, color in zip(OUTCOME_METRICS, OUTCOME_COLORS)}
    rgb = [to_rgb(colors[sample_outcome(row)]) for row in ordered]
    fig, ax = plt.subplots(figsize=(15, 2.4))
    ax.imshow([rgb], aspect="auto", interpolation="nearest")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(f"{len(ordered)} questions; ordered by viewpoint and question id")
    ax.set_title("Per-Question Answer/Direction Outcome")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in OUTCOME_COLORS]
    ax.legend(handles, [OUTCOME_LABELS[name] for name in OUTCOME_METRICS],
              loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
    fig.tight_layout()
    path = directory / "per_question_outcome_strip.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


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
    samples_path = find_submission(args.input, args.samples)
    samples = load_samples(samples_path)
    outputs = [
        plot_summary(values, directory),
        *plot_outcomes(values, directory),
        plot_question_types(values, directory),
    ]
    for path in (
        plot_split_rectangles(samples, directory),
        plot_split_rectangles_with_invalid_direction(samples, directory),
        plot_question_outcome_strip(samples, directory),
    ):
        if path is not None:
            outputs.append(path)
    outputs.append(write_report(values, directory, args.input))
    print("Generated analysis artifacts:")
    for path in outputs:
        print(path)
    if samples_path is None:
        print("Sample-level submission not found; split-rectangle plots were skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
