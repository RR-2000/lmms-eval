#!/usr/bin/env python3
"""Build a portable Markdown report for the COMFORT facing diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


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
OBJECTS = ("bicycle", "car", "dog", "duck", "horse")
DISPLAY_NAMES = {
    "cambrians": "Cambrian-S-7B",
    "internvideo3": "InternVideo3-8B-Instruct",
    "internvl3_5": "InternVL3.5-8B",
    "llava_onevision2": "LLaVA-OneVision-2-8B",
    "openai": "GPT-5.6 Luna (OpenAI)",
    "sat": "SAT (Qwen2.5-VL-SAT)",
    "spatialladder": "SpatialLadder-3B",
    "vst": "VST-7B-RL",
}


def load_submission(path: Path, model: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["records"]
    counts = Counter(str(record["parsed_answer"]) for record in records)
    gold_counts = Counter(str(record["gold_answer"]) for record in records)
    by_object = {
        obj: [record for record in records if record["reference_object"] == obj]
        for obj in OBJECTS
    }
    correct = sum(float(record["accuracy"]) for record in records)
    parsed = sum(float(record["parse_success"]) for record in records)
    return {
        "model": model,
        "path": path,
        "n": len(records),
        "counts": counts,
        "gold_counts": gold_counts,
        "by_object": by_object,
        "correct": int(correct),
        "accuracy": correct / len(records),
        "parse_success": parsed / len(records),
    }


def collect(root: Path, qwen_path: Path) -> list[dict]:
    results = []
    for path in sorted(root.glob("*/submissions/*.json")):
        key = path.parents[1].name
        results.append(load_submission(path, DISPLAY_NAMES.get(key, key)))
    results.append(load_submission(qwen_path, "Qwen3-VL-8B-Instruct"))
    return sorted(results, key=lambda result: (-result["accuracy"], result["model"]))


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def count_percent(count: int, total: int) -> str:
    return f"{count} ({100.0 * count / total:.2f}%)"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def build_report(results: list[dict], root: Path, qwen_path: Path) -> str:
    total_predictions = sum(result["n"] for result in results)
    sample = results[0]
    gold_counts = sample["gold_counts"]
    if set(gold_counts) != {"down-left"}:
        raise ValueError(f"Expected a single down-left gold class, found {gold_counts}")

    lines = [
        "# COMFORT reference-object facing-direction diagnostic",
        "",
        "## Scope",
        "",
        "This report summarizes every model submission in ",
        f"`{root}` plus the separate Qwen3-VL experiment at `{qwen_path}`.",
        "It is self-contained so it can be uploaded directly to ChatGPT for further analysis.",
        "",
        "- Dataset: `COMFORT_Multi_3D`",
        "- Task: `comfort_gt_component_facing_direction_no_bbox`",
        f"- Models: {len(results)}",
        f"- Records per model: {sample['n']}",
        f"- Total evaluated predictions: {total_predictions}",
        "- Allowed answers: `right`, `down-right`, `down`, `down-left`, `left`, `up-left`, `up`, `up-right`",
        f"- Gold-label distribution: `down-left` = {sample['n']}/{sample['n']} (100%)",
        "- Visual mode: plain image, without a bounding box",
        "",
        "## Important interpretation caveat",
        "",
        "Every record has the same gold answer, `down-left`. Therefore, exact-match accuracy is numerically identical to the fraction of times a model predicts `down-left`. This diagnostic is especially useful for exposing orientation conventions and systematic answer biases, but it is not a balanced eight-class direction benchmark.",
        "",
        "## Overall performance",
        "",
    ]

    overall_rows = []
    for rank, result in enumerate(results, start=1):
        dominant, dominant_count = result["counts"].most_common(1)[0]
        overall_rows.append(
            [
                str(rank),
                result["model"],
                f"{result['correct']}/{result['n']}",
                percent(result["accuracy"]),
                percent(result["parse_success"]),
                dominant,
                count_percent(dominant_count, result["n"]),
            ]
        )
    lines += table(
        ["Rank", "Model", "Correct", "Accuracy", "Parse success", "Most common prediction", "Count (share)"],
        overall_rows,
    )

    lines += [
        "",
        "## Complete prediction distribution",
        "",
        "Each cell is `count (percentage of 441 records)`. The gold direction is **down-left**.",
        "",
    ]
    distribution_rows = []
    for result in results:
        distribution_rows.append(
            [result["model"]]
            + [count_percent(result["counts"][direction], result["n"]) for direction in DIRECTIONS]
        )
    lines += table(["Model", *DIRECTIONS], distribution_rows)

    object_counts = {
        obj: len(sample["by_object"][obj])
        for obj in OBJECTS
    }
    lines += [
        "",
        "## Accuracy by reference-object category",
        "",
        "Because the gold answer is always `down-left`, these values are also the within-object rates of predicting `down-left`. Each cell is `correct/total (accuracy)`.",
        "",
        "Reference-object sample counts: "
        + ", ".join(f"`{obj}` = {count}" for obj, count in object_counts.items())
        + ".",
        "",
    ]
    object_accuracy_rows = []
    for result in results:
        row = [result["model"]]
        for obj in OBJECTS:
            records = result["by_object"][obj]
            correct = sum(float(record["accuracy"]) for record in records)
            row.append(f"{int(correct)}/{len(records)} ({100.0 * correct / len(records):.2f}%)")
        object_accuracy_rows.append(row)
    lines += table(["Model", *[obj.capitalize() for obj in OBJECTS]], object_accuracy_rows)

    lines += [
        "",
        "## Dominant prediction by reference-object category",
        "",
        "Each cell gives the most frequent parsed answer and its within-category share.",
        "",
    ]
    dominant_rows = []
    for result in results:
        row = [result["model"]]
        for obj in OBJECTS:
            records = result["by_object"][obj]
            counter = Counter(str(record["parsed_answer"]) for record in records)
            direction, count = counter.most_common(1)[0]
            row.append(f"{direction}: {count}/{len(records)} ({100.0 * count / len(records):.1f}%)")
        dominant_rows.append(row)
    lines += table(["Model", *[obj.capitalize() for obj in OBJECTS]], dominant_rows)

    nonzero = [result for result in results if result["correct"]]
    zero = [result["model"] for result in results if not result["correct"]]
    lines += [
        "",
        "## Main observations",
        "",
        f"- The strongest result is **{results[0]['model']}** at **{percent(results[0]['accuracy'])}** ({results[0]['correct']}/{results[0]['n']}).",
        f"- **{results[1]['model']}** is second at **{percent(results[1]['accuracy'])}** ({results[1]['correct']}/{results[1]['n']}).",
        "- Only " + ", ".join(f"**{result['model']}**" for result in nonzero) + " predict `down-left` at least once.",
        "- Zero exact matches: " + ", ".join(f"**{model}**" for model in zero) + ".",
        "- Several models collapse onto one or two horizontal or diagonal labels rather than using all eight classes, indicating strong direction-convention or projection biases.",
        "- A high parse-success rate only means the output could be mapped to an allowed label; it does not imply directional correctness.",
        "",
        "## Model-specific patterns",
        "",
    ]
    for result in results:
        frequent = result["counts"].most_common(3)
        pattern = ", ".join(
            f"`{direction}` {count_percent(count, result['n'])}"
            for direction, count in frequent
        )
        lines.append(
            f"- **{result['model']}**: accuracy {percent(result['accuracy'])}; leading predictions: {pattern}."
        )

    lines += [
        "",
        "## Suggested questions for follow-up analysis",
        "",
        "1. Which models appear to use a mirrored, rotated, or image-horizontal direction convention?",
        "2. Are the object-category differences consistent with failures to infer the intrinsic front of particular object types?",
        "3. How much of each model's behavior is explained by predicting only its one or two preferred labels?",
        "4. How should this degenerate-label diagnostic be combined with a balanced facing-direction evaluation?",
        "",
        "## Source fields used",
        "",
        "The report was recomputed from record-level `parsed_answer`, `gold_answer`, `accuracy`, `parse_success`, and `reference_object` fields. Percentages use all 441 records as the denominator unless explicitly reported within an object category.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/final_comfort_gt_component_facing_direction_no_bbox"),
    )
    parser.add_argument(
        "--qwen",
        type=Path,
        default=Path(
            "outputs/comfort_gt_component_facing_direction_no_bbox_8/submissions/"
            "comfort_gt_component_facing_direction_no_bbox_qwen3_vl_experiments.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(build_report(collect(args.root, args.qwen), args.root, args.qwen))


if __name__ == "__main__":
    main()
