#!/usr/bin/env python3
"""Plot answer-direction distributions for a COMFORT direction/object submission.

The direction-task plot compares each ground-truth direction with the direction
selected by the model.  For object tasks, the selected object is resolved via
the original COMFORT ``scenes.jsonl`` manifest, then plotted by its actual
direction relative to the scene's reference object.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DIRECTIONS = ("left", "right", "front", "behind")
DEFAULT_SUBMISSION = Path("/home/ramanathan/VLM/lmms-eval/outputs/comfort_direction_object_0/" "submissions/comfort_direction_object_qwen3_vl_experiments.json")
DEFAULT_SCENES = Path("/home/ramanathan/data/COMFORT_Multi_3D/scenes.jsonl")


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload if isinstance(payload, list) else payload.get("records")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("Submission must be a record list or an object with a 'records' list")
    return records


def load_object_directions(path: Path) -> dict[str, dict[str, str]]:
    """Map each scene's object name to its reference-object-relative direction."""
    lookup: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            scene = json.loads(line)
            scene_id = scene.get("scene_id")
            positions = scene.get("objects_at_reference_directions")
            if not isinstance(scene_id, str) or not isinstance(positions, dict):
                raise ValueError(f"Invalid scene metadata at {path}:{line_number}")
            lookup[scene_id] = {str(object_name): direction for direction, object_name in positions.items() if direction in DIRECTIONS and isinstance(object_name, str)}
    return lookup


def selected_option(row: dict[str, Any]) -> str | None:
    letter = str(row.get("predicted_option_letter") or "").upper()
    options = row.get("options")
    if len(letter) != 1 or letter not in "ABCD" or not isinstance(options, list):
        return None
    index = ord(letter) - ord("A")
    return str(options[index]) if index < len(options) else None


def count_distributions(records: list[dict[str, Any]], object_directions: dict[str, dict[str, str]]) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]], dict[str, int]]:
    direction_counts = {direction: Counter() for direction in DIRECTIONS}
    object_counts = {direction: Counter() for direction in DIRECTIONS}
    skipped = Counter()
    for row in records:
        variant = row.get("answer_format", row.get("variant"))
        gold_direction = str(row.get("relation") or row.get("gold_answer") or "")
        if gold_direction not in DIRECTIONS:
            skipped["missing_or_invalid_gold_direction"] += 1
            continue
        prediction = selected_option(row)
        if variant == "direction":
            direction_counts[gold_direction][prediction if prediction in DIRECTIONS else "parse failure"] += 1
        elif variant == "object":
            if prediction is None:
                object_counts[gold_direction]["parse failure"] += 1
                continue
            scene_objects = object_directions.get(str(row.get("scene_id")))
            if scene_objects is None:
                skipped["scene_missing_from_manifest"] += 1
                object_counts[gold_direction]["unmapped object"] += 1
                continue
            object_counts[gold_direction][scene_objects.get(prediction, "unmapped object")] += 1
    return direction_counts, object_counts, dict(skipped)


def save_plot(counts: dict[str, Counter[str]], title: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [*DIRECTIONS]
    categories = [*DIRECTIONS, "unmapped object", "parse failure"]
    colors = {
        "left": "#4C78A8",
        "right": "#F58518",
        "front": "#54A24B",
        "behind": "#E45756",
        "unmapped object": "#B8B8B8",
        "parse failure": "#6B6B6B",
    }
    figure, axis = plt.subplots(figsize=(9, 6))
    bottom = [0] * len(labels)
    for category in categories:
        values = [counts[label][category] for label in labels]
        if not any(values):
            continue
        axis.bar(labels, values, bottom=bottom, label=category, color=colors[category])
        bottom = [current + value for current, value in zip(bottom, values)]
    axis.set_xlabel("Ground-truth direction")
    axis.set_ylabel("Number of predictions")
    axis.set_title(title)
    axis.legend(title="Model-selected direction", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def serialise(counts: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {direction: dict(counts[direction]) for direction in DIRECTIONS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", nargs="?", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES, help="Original COMFORT scenes.jsonl manifest")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <submission parent>/<stem>_direction_plots")
    args = parser.parse_args()

    output_dir = args.output_dir or args.submission.parent / f"{args.submission.stem}_direction_plots"
    direction_counts, object_counts, skipped = count_distributions(load_records(args.submission), load_object_directions(args.scenes))
    direction_plot = output_dir / "direction_gt_vs_predicted_direction.png"
    object_plot = output_dir / "object_gt_vs_predicted_object_direction.png"
    save_plot(direction_counts, "Direction tasks: ground truth vs model-selected direction", direction_plot)
    save_plot(object_counts, "Object tasks: ground truth vs direction of selected object", object_plot)
    summary_path = output_dir / "direction_distributions.json"
    summary_path.write_text(
        json.dumps(
            {
                "direction_tasks": serialise(direction_counts),
                "object_tasks": serialise(object_counts),
                "skipped": skipped,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {direction_plot}\nSaved: {object_plot}\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
