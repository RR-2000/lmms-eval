#!/usr/bin/env python3
"""Plot COMFORT direction/object confusion matrices from an answer-case report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


DIRECTIONS = ("left", "right", "front", "behind")
MATRIX_SPECS = {
    "direction_tasks": "direction_tasks_confusion_matrix.png",
    "object_tasks": "object_tasks_confusion_matrix.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to answer_case_report.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: the report's directory)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output resolution")
    return parser.parse_args()


def load_distributions(report_path: Path) -> dict[str, Any]:
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    distributions = report.get("selected_direction_distributions")
    if not isinstance(distributions, dict):
        raise ValueError("Report has no 'selected_direction_distributions' object")
    return distributions


def counts_matrix(distributions: dict[str, Any], task_key: str) -> np.ndarray:
    task_counts = distributions.get(task_key)
    if not isinstance(task_counts, dict):
        raise ValueError(f"Report has no '{task_key}' distribution")

    matrix = np.zeros((len(DIRECTIONS), len(DIRECTIONS)), dtype=int)
    for row_index, true_direction in enumerate(DIRECTIONS):
        row = task_counts.get(true_direction)
        if not isinstance(row, dict):
            raise ValueError(f"Missing distribution for {task_key}/{true_direction}")
        for column_index, selected_direction in enumerate(DIRECTIONS):
            value = row.get(selected_direction, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"Invalid count for {task_key}/{true_direction}/{selected_direction}: {value!r}"
                )
            matrix[row_index, column_index] = value
    return matrix


def annotations(counts: np.ndarray) -> np.ndarray:
    labels = np.empty(counts.shape, dtype=object)
    for row_index, column_index in np.ndindex(counts.shape):
        labels[row_index, column_index] = f"{counts[row_index, column_index]:,}"
    return labels


def save_confusion_matrix(
    counts: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    row_totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    ) * 100.0

    sns.set_theme(style="white", context="talk")
    figure, axis = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        percentages,
        annot=annotations(counts),
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        linewidths=1,
        linecolor="white",
        square=True,
        xticklabels=[direction.title() for direction in DIRECTIONS],
        yticklabels=[direction.title() for direction in DIRECTIONS],
        cbar_kws={"shrink": 0.75, "format": "%.0f%%"},
        ax=axis,
    )
    axis.set_xlabel("Selected direction")
    axis.set_ylabel("True direction")
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    distributions = load_distributions(args.report)
    output_dir = args.output_dir or args.report.parent
    for task_key, filename in MATRIX_SPECS.items():
        output_path = output_dir / filename
        save_confusion_matrix(
            counts_matrix(distributions, task_key),
            output_path,
            args.dpi,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
