#!/usr/bin/env python3
"""Render ego-ground-truth versus predicted-direction confusion matrices.

Direction-answer rows are read directly from the selected option.  For
object-answer rows, the selected object is resolved to its reference-relative
direction through the original COMFORT scene manifest.  Each row is normalized
within its ground-truth ego direction, so matrices are comparable across
models despite small class-count differences.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DIRECTIONS = ("left", "right", "front", "behind")
PREDICTION_LABELS = (*DIRECTIONS, "unmapped / parse")
DEFAULT_SCENES = Path("/home/ramanathan/data/COMFORT_Multi_3D/scenes.jsonl")
MODEL_LABELS = {
    "cambrians": "Cambrian-S-7B",
    "internvideo3": "InternVideo3-8B",
    "internvl3_5": "InternVL3.5-8B",
    "llava_onevision2": "LLaVA-OneVision-2-8B",
    "openai": "GPT-5.6-luna",
    "sat": "Qwen2.5-VL-SAT",
    "spatialladder": "SpatialLadder-3B",
    "vst": "VST-7B-RL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/final_comfort_direction_object"),
        help="Root containing one model directory per COMFORT submission.",
    )
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_object_directions(path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            scene = json.loads(line)
            positions = scene.get("objects_at_reference_directions", {})
            lookup[str(scene["scene_id"])] = {
                str(object_name): direction
                for direction, object_name in positions.items()
                if direction in DIRECTIONS
            }
    return lookup


def selected_option(row: dict[str, Any]) -> str | None:
    letter = str(row.get("predicted_option_letter") or "").upper()
    options = row.get("options")
    if len(letter) != 1 or not ("A" <= letter <= "D") or not isinstance(options, list):
        return None
    index = ord(letter) - ord("A")
    return str(options[index]) if index < len(options) else None


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"No record list in {path}")
    return [row for row in records if isinstance(row, dict)]


def confusion_counts(records: list[dict[str, Any]], object_directions: dict[str, dict[str, str]]) -> dict[str, np.ndarray]:
    counts = {answer_format: np.zeros((len(DIRECTIONS), len(PREDICTION_LABELS)), dtype=int) for answer_format in ("direction", "object")}
    for row in records:
        answer_format = str(row.get("answer_format") or row.get("variant") or "")
        gold = str(row.get("relation") or "")
        if answer_format not in counts or gold not in DIRECTIONS:
            continue
        selected = selected_option(row)
        if answer_format == "direction":
            predicted = selected
        else:
            predicted = object_directions.get(str(row.get("scene_id")), {}).get(str(selected)) if selected else None
        column = PREDICTION_LABELS.index(predicted) if predicted in DIRECTIONS else len(PREDICTION_LABELS) - 1
        counts[answer_format][DIRECTIONS.index(gold), column] += 1
    return counts


def row_percentages(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0) * 100.0


def plot_grid(matrices: list[tuple[str, np.ndarray]], answer_format: str, output: Path) -> None:
    rows = int(np.ceil(len(matrices) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(15, 4.4 * rows), squeeze=False, layout="constrained")
    image = None
    for axis, (model, counts) in zip(axes.flat, matrices):
        percentages = row_percentages(counts)
        image = axis.imshow(percentages, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for row, column in np.ndindex(counts.shape):
            axis.text(
                column,
                row,
                f"{percentages[row, column]:.1f}%\n({counts[row, column]})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if percentages[row, column] >= 55 else "black",
            )
        axis.set_title(MODEL_LABELS.get(model, model), fontsize=12)
        axis.set_xticks(range(len(PREDICTION_LABELS)), [label.title() for label in PREDICTION_LABELS], rotation=24, ha="right")
        axis.set_yticks(range(len(DIRECTIONS)), [label.title() for label in DIRECTIONS])
        axis.set_xlabel("Predicted ego-relative direction")
        axis.set_ylabel("Gold ego-relative direction")
    for axis in axes.flat[len(matrices):]:
        axis.set_visible(False)
    figure.suptitle(f"COMFORT: Gold Ego Direction vs Prediction ({answer_format.title()} Answers)", fontsize=16)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, label="Within-row percentage")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir / "ego_vs_predicted_confusions").expanduser().resolve()
    object_directions = load_object_directions(args.scenes.expanduser().resolve())
    submissions = sorted(input_dir.glob("*/submissions/comfort_direction_object_*.json"))
    if not submissions:
        raise FileNotFoundError(f"No COMFORT direction/object submissions under {input_dir}")

    by_model: dict[str, dict[str, np.ndarray]] = {}
    for submission in submissions:
        model = submission.parents[1].name
        by_model[model] = confusion_counts(load_records(submission), object_directions)

    ordered = sorted(by_model.items())
    for answer_format in ("direction", "object"):
        plot_grid(
            [(model, matrices[answer_format]) for model, matrices in ordered],
            answer_format,
            output_dir / f"ego_vs_predicted_{answer_format}_answers.png",
        )

    serializable = {
        model: {answer_format: matrix.tolist() for answer_format, matrix in matrices.items()}
        for model, matrices in ordered
    }
    (output_dir / "confusion_counts.json").write_text(
        json.dumps(
            {
                "rows": list(DIRECTIONS),
                "columns": list(PREDICTION_LABELS),
                "counts": serializable,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(by_model)} model confusion matrices to {output_dir}")


if __name__ == "__main__":
    main()
