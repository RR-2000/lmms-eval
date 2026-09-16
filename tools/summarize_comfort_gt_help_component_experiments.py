#!/usr/bin/env python3
"""Summarize COMFORT GT_HELP component diagnostic submissions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


PREFIX = "comfort_gt_component_"
DEFAULT_INPUT = Path("outputs/comfort_gt_help_components_8")
TASK_INFO = {
    "comfort_gt_component_bbox_prediction": ("Localization", "BBox prediction", "bbox_iou"),
    "comfort_gt_component_bbox_naming": ("Localization", "Name boxed object", "accuracy"),
    "comfort_gt_component_facing_direction": ("Orientation", "8-way facing", "accuracy"),
    "comfort_gt_component_front_arrow": ("Orientation", "Predict front arrow", "arrow_cosine"),
    "comfort_gt_component_front_arrow_reading": ("Orientation", "Use supplied front arrow", "accuracy"),
    "comfort_gt_component_left_arrow": ("Orientation", "Predict left arrow", "arrow_cosine"),
    "comfort_gt_component_left_arrow_reading": ("Orientation", "Use supplied left arrow", "accuracy"),
    "comfort_gt_component_symbol_to_object": ("Symbol mapping", "Symbol to object", "accuracy"),
    "comfort_gt_component_object_to_symbol": ("Symbol mapping", "Object to symbol", "accuracy"),
    "comfort_gt_component_long_arrow_to_symbol": ("Symbol mapping", "Long arrows to symbol", "accuracy"),
    "comfort_gt_component_short_arrow_to_symbol": ("Symbol mapping", "Short arrows to symbol", "accuracy"),
    "comfort_gt_component_vector_to_direction": ("Vector semantics", "Vector to direction", "accuracy"),
    "comfort_gt_component_direction_to_vector": ("Vector semantics", "Direction to vector", "vector_cosine"),
    "comfort_gt_component_projected_axes_prediction": ("Vector semantics", "Predict projected axes", "basis_mean_cosine"),
    "comfort_gt_component_text_axes_direction": ("Vector semantics", "Text axes applied", "accuracy"),
    "comfort_gt_component_overlay_axes_direction": ("Vector semantics", "Overlay axes applied", "accuracy"),
}

TASK_ROLES = {
    "comfort_gt_component_bbox_prediction": ("BBox", "generation"),
    "comfort_gt_component_bbox_naming": ("BBox", "utilization"),
    "comfort_gt_component_facing_direction": ("Orientation label", "generation"),
    "comfort_gt_component_front_arrow": ("Front arrow", "generation"),
    "comfort_gt_component_front_arrow_reading": ("Front arrow", "utilization"),
    "comfort_gt_component_left_arrow": ("Left arrow", "generation"),
    "comfort_gt_component_left_arrow_reading": ("Left arrow", "utilization"),
    "comfort_gt_component_symbol_to_object": ("Abstract symbols", "utilization"),
    "comfort_gt_component_object_to_symbol": ("Abstract symbols", "generation"),
    "comfort_gt_component_long_arrow_to_symbol": ("Long direction arrows", "utilization"),
    "comfort_gt_component_short_arrow_to_symbol": ("Short direction arrows", "utilization"),
    "comfort_gt_component_vector_to_direction": ("Direction vector", "utilization"),
    "comfort_gt_component_direction_to_vector": ("Direction vector", "generation"),
    "comfort_gt_component_projected_axes_prediction": ("Projected axes", "generation"),
    "comfort_gt_component_text_axes_direction": ("Projected axes (text)", "utilization"),
    "comfort_gt_component_overlay_axes_direction": ("Projected axes (overlay)", "utilization"),
}

PAIR_SPECS = (
    ("BBox", "comfort_gt_component_bbox_prediction", "comfort_gt_component_bbox_naming"),
    ("Front arrow", "comfort_gt_component_front_arrow", "comfort_gt_component_front_arrow_reading"),
    ("Left arrow", "comfort_gt_component_left_arrow", "comfort_gt_component_left_arrow_reading"),
    ("Abstract symbols", "comfort_gt_component_object_to_symbol", "comfort_gt_component_symbol_to_object"),
    ("Direction vector", "comfort_gt_component_direction_to_vector", "comfort_gt_component_vector_to_direction"),
    ("Projected axes → text use", "comfort_gt_component_projected_axes_prediction", "comfort_gt_component_text_axes_direction"),
    ("Projected axes → overlay use", "comfort_gt_component_projected_axes_prediction", "comfort_gt_component_overlay_axes_direction"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[DEFAULT_INPUT], help="Submission files or directories searched recursively.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <first input>/comfort_gt_component_analysis.")
    return parser.parse_args()


def discover(inputs: Iterable[Path]) -> list[Path]:
    found = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(candidate for candidate in path.rglob("*.json") if candidate.name.startswith(PREFIX))
        else:
            raise FileNotFoundError(path)
    unique = sorted(set(found))
    if not unique:
        raise FileNotFoundError("No comfort_gt_component_*.json submissions found")
    return unique


def load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        rows = payload.get("records")
        metadata = payload
    elif isinstance(payload, list):
        rows, metadata = payload, {}
    else:
        raise ValueError(f"Unsupported JSON structure: {path}")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No records in {path}")
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No dictionary records in {path}")
    return metadata, rows


def mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def infer_task(metadata: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> str:
    values = {str(row.get("diagnostic_task")) for row in rows if row.get("diagnostic_task")}
    if metadata.get("task"):
        values.add(str(metadata["task"]))
    values = {value for value in values if value in TASK_INFO}
    if len(values) == 1:
        return next(iter(values))
    for task in TASK_INFO:
        if path.name.startswith(task + "_"):
            return task
    raise ValueError(f"Could not infer one known task from {path}: {sorted(values)}")


def model_name(task: str, path: Path) -> str:
    stem = path.stem
    return stem[len(task) + 1:] if stem.startswith(task + "_") else stem


def summarize_one(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata, rows = load(path)
    task = infer_task(metadata, rows, path)
    capability, label, primary_field = TASK_INFO[task]
    fields = (
        "accuracy", "raw_accuracy", "parse_success", "bbox_iou", "bbox_acc_0_5",
        "arrow_cosine", "arrow_angle_30_accuracy", "arrow_start_score",
        "vector_cosine", "vector_angle_30_accuracy", "vector_full_sign_accuracy",
        "basis_mean_cosine", "basis_all_angle_30_accuracy",
    )
    component, component_role = TASK_ROLES[task]
    result = {
        "task": task,
        "capability": capability,
        "label": label,
        "component": component,
        "component_role": component_role,
        "model": model_name(task, path),
        "submission": str(path),
        "num_records": len(rows),
        "num_scenes": len({str(row.get("scene_id")) for row in rows}),
        "primary_field": primary_field,
        **{field: mean(rows, field) for field in fields},
    }
    result["primary_score"] = result[primary_field]
    if result["raw_accuracy"] is None:
        # Backward-compatible derivation for submissions produced before the
        # uniform raw_accuracy field was added.
        if result["accuracy"] is not None:
            result["raw_accuracy"] = result["accuracy"]
        elif result["bbox_acc_0_5"] is not None:
            result["raw_accuracy"] = result["bbox_acc_0_5"]
        elif result["arrow_angle_30_accuracy"] is not None:
            result["raw_accuracy"] = result["arrow_angle_30_accuracy"]
        elif result["vector_angle_30_accuracy"] is not None:
            result["raw_accuracy"] = result["vector_angle_30_accuracy"]

    answer_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gold = row.get("gold_answer")
        if gold is not None:
            answer_groups[str(gold)].append(row)
    breakdown = []
    for answer, group in sorted(answer_groups.items()):
        breakdown.append({
            "task": task,
            "model": result["model"],
            "gold_answer": answer,
            "count": len(group),
            "accuracy": mean(group, "accuracy"),
            "parse_success": mean(group, "parse_success"),
        })
    return result, breakdown


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# COMFORT GT_HELP component results", "",
        "| Capability | Component | Role | Diagnostic | Model | N | Primary metric | Score | Raw accuracy | Parse success |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['capability']} | {row['component']} | {row['component_role']} | {row['label']} | {row['model']} | {row['num_records']} | {row['primary_field']} | {fmt(row['primary_score'])} | {fmt(row['raw_accuracy'])} | {fmt(row['parse_success'])} |")
    lines.extend([
        "", "## Recommended paired comparisons", "",
        "- BBox prediction vs name-boxed-object: producing versus consuming localization.",
        "- Front-arrow vs left-arrow: orientation versus handedness.",
        "- Symbol-to-object vs object-to-symbol: both directions of abstract grounding.",
        "- Long vs short arrows: isolated GT_HELP 6 versus GT_HELP 36 cue-length effect.",
        "- Vector-to-direction vs direction-to-vector: decoding versus encoding axis semantics.",
        "- Text axes vs overlay axes: numeric versus visual use of the same projected basis.",
        "", "## Generation versus utilization", "",
        "All paired bars use the common binary `raw_accuracy`: IoU ≥ 0.5 for boxes, angular error ≤ 30° for arrows/vectors/axes, and exact match for classification. This makes the pair readable, while the primary metrics above retain finer-grained performance.",
    ])
    pairs = generation_utilization_rows(summaries)
    if pairs:
        lines.extend((
            "",
            "| Component pair | Model | Generation | Utilization | Use−generation |",
            "|---|---|---:|---:|---:|",
        ))
        for row in pairs:
            delta = row["utilization_accuracy"] - row["generation_accuracy"]
            lines.append(
                f"| {row['pair']} | {row['model']} | {fmt(row['generation_accuracy'])} | "
                f"{fmt(row['utilization_accuracy'])} | {100.0 * delta:+.2f} pp |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generation_utilization_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    models = sorted({row["model"] for row in summaries})
    for model in models:
        by_task = {row["task"]: row for row in summaries if row["model"] == model}
        for pair, generation_task, utilization_task in PAIR_SPECS:
            generation = by_task.get(generation_task)
            utilization = by_task.get(utilization_task)
            if not generation or not utilization:
                continue
            if generation["raw_accuracy"] is None or utilization["raw_accuracy"] is None:
                continue
            rows.append({
                "pair": pair,
                "model": model,
                "generation_task": generation_task,
                "utilization_task": utilization_task,
                "generation_accuracy": float(generation["raw_accuracy"]),
                "utilization_accuracy": float(utilization["raw_accuracy"]),
            })
    return rows


def plot_generation_utilization(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [f"{row['pair']} · {row['model']}" for row in rows]
    y = list(range(len(rows)))
    height = max(5.0, 0.55 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(12, height))
    bar_height = 0.36
    ax.barh([value - bar_height / 2 for value in y], [row["generation_accuracy"] for row in rows], height=bar_height, label="Generate component", color="#457B9D")
    ax.barh([value + bar_height / 2 for value in y], [row["utilization_accuracy"] for row in rows], height=bar_height, label="Use component", color="#E9C46A")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Raw accuracy")
    ax.set_title("COMFORT component generation versus utilization")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_primary(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = [row for row in summaries if row["primary_score"] is not None]
    height = max(5.0, 0.42 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    y = list(range(len(rows)))
    colors = {"Localization": "#457B9D", "Orientation": "#E76F51", "Symbol mapping": "#E9C46A", "Vector semantics": "#2A9D8F"}
    values = [float(row["primary_score"]) for row in rows]
    ax.barh(y, values, color=[colors[row["capability"]] for row in rows])
    ax.set_yticks(y, [f"{row['label']} · {row['model']}" for row in rows])
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xlim(min(-0.05, min(values, default=0.0) - 0.05), 1.05)
    ax.set_xlabel("Primary score (accuracy/IoU/cosine)")
    ax.set_title("COMFORT GT_HELP component diagnostics")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_raw_accuracy(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = [row for row in summaries if row["raw_accuracy"] is not None]
    height = max(5.0, 0.42 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    y = list(range(len(rows)))
    values = [float(row["raw_accuracy"]) for row in rows]
    colors = {"Localization": "#457B9D", "Orientation": "#E76F51", "Symbol mapping": "#E9C46A", "Vector semantics": "#2A9D8F"}
    ax.barh(y, values, color=[colors[row["capability"]] for row in rows])
    ax.set_yticks(y, [f"{row['label']} · {row['model']}" for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Raw accuracy")
    ax.set_title("COMFORT GT_HELP raw accuracy by task")
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(values):
        if value > 0.9:
            ax.text(value - 0.012, index, f"{100.0 * value:.1f}%", va="center", ha="right")
        else:
            ax.text(value + 0.012, index, f"{100.0 * value:.1f}%", va="center", ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_comparisons(path: Path, summaries: list[dict[str, Any]]) -> None:
    # One panel per model; task order preserves the intended paired comparisons.
    models = sorted({row["model"] for row in summaries})
    fig, axes = plt.subplots(len(models), 1, figsize=(13, max(5, 4.2 * len(models))), squeeze=False)
    order = list(TASK_INFO)
    for axis, model in zip(axes[:, 0], models):
        by_task = {row["task"]: row for row in summaries if row["model"] == model}
        rows = [by_task[task] for task in order if task in by_task and by_task[task]["primary_score"] is not None]
        x = list(range(len(rows)))
        axis.bar(x, [row["primary_score"] for row in rows], color="#4C78A8")
        axis.set_xticks(x, [row["label"] for row in rows], rotation=35, ha="right")
        axis.set_ylim(min(-0.05, min((row["primary_score"] for row in rows), default=0.0) - 0.05), 1.05)
        axis.set_ylabel("Score")
        axis.set_title(model)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Representation and cue comparisons", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = discover(args.inputs)
    summaries, breakdown = [], []
    for path in paths:
        summary, answers = summarize_one(path)
        summaries.append(summary)
        breakdown.extend(answers)
    summaries.sort(key=lambda row: (row["model"], list(TASK_INFO).index(row["task"])))
    first = args.inputs[0].expanduser().resolve()
    default_parent = first if first.is_dir() else first.parent
    output = (args.output_dir.expanduser().resolve() if args.output_dir else default_parent / "comfort_gt_component_analysis")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({"experiments": summaries, "answer_breakdown": breakdown}, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "answer_breakdown.csv", breakdown)
    pair_rows = generation_utilization_rows(summaries)
    write_csv(output / "generation_vs_utilization.csv", pair_rows)
    write_markdown(output / "summary.md", summaries)
    plot_primary(output / "primary_scores.png", summaries)
    plot_raw_accuracy(output / "raw_accuracy.png", summaries)
    plot_comparisons(output / "representation_comparisons.png", summaries)
    plot_generation_utilization(output / "generation_vs_utilization.png", pair_rows)
    print(f"Analyzed {len(paths)} submissions; wrote results to {output}")


if __name__ == "__main__":
    main()
