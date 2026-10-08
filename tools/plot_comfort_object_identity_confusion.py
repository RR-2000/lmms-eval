#!/usr/bin/env python3
"""Plot filtered COMFORT object-answer identity confusion.

Unlike the direction-resolved object diagnostic, rows are true object labels and
columns are the literal object option selected by the model.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, default=Path("/home/ramanathan/data/COMFORT_Multi_3D/scenes.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts-output", type=Path)
    parser.add_argument("--labels", nargs=2, default=("horsel", "horser"), metavar=("LABEL_A", "LABEL_B"))
    return parser.parse_args()


def excluded_scenes(path: Path, labels: set[str]) -> set[str]:
    excluded = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        scene = json.loads(line)
        scene_labels = {str(item.get("label", "")).lower() for item in scene.get("objects", [])}
        if labels <= scene_labels:
            excluded.add(str(scene["scene_id"]))
    return excluded


def selected_object(row: dict) -> str | None:
    letter = str(row.get("predicted_option_letter") or "").upper()
    options = row.get("options")
    if len(letter) != 1 or not "A" <= letter <= "D" or not isinstance(options, list):
        return None
    index = ord(letter) - ord("A")
    return str(options[index]) if index < len(options) else None


def main() -> None:
    args = parse_args()
    labels_to_exclude = {label.lower() for label in args.labels}
    excluded = excluded_scenes(args.scene_manifest, labels_to_exclude)
    payload = json.loads(args.submission.read_text(encoding="utf-8"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    rows = [
        row for row in records
        if isinstance(row, dict)
        and row.get("answer_format") == "object"
        and str(row.get("scene_id")) not in excluded
    ]
    observations = [(str(row.get("gold_answer") or row.get("target") or "<missing>"), selected_object(row) or "unmapped / parse") for row in rows]
    names = sorted({item for pair in observations for item in pair})
    matrix = np.zeros((len(names), len(names)), dtype=int)
    index = {name: position for position, name in enumerate(names)}
    for gold, selected in observations:
        matrix[index[gold], index[selected]] += 1
    totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0) * 100.0
    annotations = np.empty(matrix.shape, dtype=object)
    for row, column in np.ndindex(matrix.shape):
        annotations[row, column] = f"{percentages[row, column]:.1f}%\n({matrix[row, column]})" if matrix[row, column] else ""

    sns.set_theme(style="white", context="talk")
    size = max(9, min(18, 0.82 * len(names) + 5))
    figure, axis = plt.subplots(figsize=(size, size))
    sns.heatmap(
        percentages, annot=annotations, fmt="", cmap="Blues", vmin=0, vmax=100,
        linewidths=0.75, linecolor="white", square=True,
        xticklabels=names, yticklabels=names,
        cbar_kws={"shrink": 0.75, "format": "%.0f%%"}, ax=axis,
    )
    axis.set_xlabel("Selected object")
    axis.set_ylabel("True object")
    axis.set_title("COMFORT object-answer identity confusion\n(horsel/horser-pair scenes excluded)")
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "rows": names,
        "columns": names,
        "counts": matrix.tolist(),
        "object_answer_records": len(rows),
        "excluded_scenes": len(excluded),
        "excluded_when_all_labels_present": sorted(labels_to_exclude),
        "submission": str(args.submission),
    }
    counts_path = args.counts_output or args.output.with_suffix(".json")
    counts_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {args.output}")
    print(f"Saved: {counts_path}")


if __name__ == "__main__":
    main()
