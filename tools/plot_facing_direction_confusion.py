#!/usr/bin/env python3
"""Create a confusion matrix for an eight-way facing-direction submission."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DIRECTIONS = (
    "right",
    "down-right",
    "down",
    "down-left",
    "left",
    "up-left",
    "up",
    "up-right",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the gold-versus-predicted facing-direction confusion matrix."
    )
    parser.add_argument("submission", type=Path, help="Submission JSON file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run directory>/facing_direction_confusion.",
    )
    return parser.parse_args()


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
        raise ValueError("Submission must be a JSON object or list")
    if not isinstance(records, list) or not records:
        raise ValueError("Submission contains no records")
    return metadata, [record for record in records if isinstance(record, dict)]


def build_matrix(records: list[dict[str, Any]]) -> tuple[list[list[int]], Counter[str]]:
    index = {direction: position for position, direction in enumerate(DIRECTIONS)}
    matrix = [[0 for _ in DIRECTIONS] for _ in DIRECTIONS]
    unparsed: Counter[str] = Counter()
    for record in records:
        gold = str(record.get("gold_answer", record.get("target", ""))).strip().lower()
        prediction = str(record.get("parsed_answer", "")).strip().lower()
        if gold not in index:
            raise ValueError(f"Unknown gold direction {gold!r} in {record.get('qid')}")
        if prediction not in index:
            unparsed[gold] += 1
            continue
        matrix[index[gold]][index[prediction]] += 1
    return matrix, unparsed


def write_csv_files(matrix: list[list[int]], output_dir: Path) -> list[Path]:
    counts_path = output_dir / "confusion_counts.csv"
    normalized_path = output_dir / "confusion_row_normalized.csv"
    with counts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\prediction", *DIRECTIONS])
        for direction, row in zip(DIRECTIONS, matrix):
            writer.writerow([direction, *row])
    with normalized_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\prediction", *DIRECTIONS])
        for direction, row in zip(DIRECTIONS, matrix):
            total = sum(row)
            writer.writerow(
                [direction, *[(value / total if total else 0.0) for value in row]]
            )
    return [counts_path, normalized_path]


def plot_matrix(matrix: list[list[int]], task: str, output_dir: Path) -> Path:
    normalized = []
    for row in matrix:
        total = sum(row)
        normalized.append([value / total if total else 0.0 for value in row])

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(DIRECTIONS)), DIRECTIONS, rotation=35, ha="right")
    ax.set_yticks(range(len(DIRECTIONS)), DIRECTIONS)
    ax.set_xlabel("Predicted direction")
    ax.set_ylabel("Ground-truth direction")
    ax.set_title(f"{task}: facing-direction confusion matrix")
    for row_index, row in enumerate(matrix):
        total = sum(row)
        for column_index, count in enumerate(row):
            label = "—" if not total else f"{count}\n{count / total:.1%}"
            color = "white" if total and count / total >= 0.5 else "black"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized share")
    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    matrix: list[list[int]],
    unparsed: Counter[str],
    output_dir: Path,
) -> Path:
    total = len(records)
    parsed = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(DIRECTIONS)))
    gold_counts = {
        direction: sum(matrix[index]) + unparsed[direction]
        for index, direction in enumerate(DIRECTIONS)
    }
    supported = [direction for direction, count in gold_counts.items() if count]
    lines = [
        "# Facing-direction confusion matrix",
        "",
        f"- Task: `{metadata.get('task', 'unknown')}`",
        f"- Records: {total}",
        f"- Parsed predictions: {parsed}/{total} ({parsed / total:.2%})",
        f"- Exact accuracy: {correct}/{total} ({correct / total:.2%})",
        f"- Ground-truth classes represented: {len(supported)}/8 ({', '.join(supported)})",
        "",
    ]
    if len(supported) < len(DIRECTIONS):
        lines.extend(
            [
                "> **Dataset warning:** the ground-truth labels do not cover all eight classes. "
                "Rows marked with an em dash have no samples, so this run cannot measure "
                "eight-way class performance.",
                "",
            ]
        )
    lines.extend(
        [
            "Cells show `count (percentage within the ground-truth row)`.",
            "",
            "| Gold \\ Predicted | " + " | ".join(DIRECTIONS) + " |",
            "|---|" + "---:|" * len(DIRECTIONS),
        ]
    )
    for direction, row in zip(DIRECTIONS, matrix):
        row_total = sum(row)
        cells = [
            (f"{value} ({value / row_total:.2%})" if row_total else "—")
            for value in row
        ]
        lines.append(f"| {direction} | " + " | ".join(cells) + " |")
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    submission = args.submission.expanduser().resolve()
    metadata, records = load_submission(submission)
    matrix, unparsed = build_matrix(records)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else submission.parent.parent / "facing_direction_confusion"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        *write_csv_files(matrix, output_dir),
        plot_matrix(matrix, str(metadata.get("task", submission.stem)), output_dir),
        write_summary(metadata, records, matrix, unparsed, output_dir),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
