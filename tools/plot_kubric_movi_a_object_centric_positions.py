#!/usr/bin/env python3
"""Visualize Kubric MOVi-A object-centric position experiment submissions.

The script consumes the per-example submission JSON emitted by the task
aggregation functions and groups examples by ``task_family`` and by the four
mutually-exclusive answer/direction outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


OUTCOME_KEYS = (
    "answer_and_direction_correct",
    "answer_correct_direction_wrong",
    "answer_wrong_direction_correct",
    "answer_and_direction_wrong",
)
OUTCOME_LABELS = {
    "answer_and_direction_correct": "Answer + direction correct",
    "answer_correct_direction_wrong": "Answer correct, direction wrong",
    "answer_wrong_direction_correct": "Answer wrong, direction correct",
    "answer_and_direction_wrong": "Answer + direction wrong",
}
OUTCOME_COLORS = ("#2A9D8F", "#E9C46A", "#F4A261", "#E76F51")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group Kubric MOVi-A results by task family and correctness."
    )
    parser.add_argument("--input", type=Path, required=True, help="One per-example submission JSON.")
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            f"{path} must be the per-example submission JSON (a non-empty list); "
            "the aggregate *_results.json does not contain task_family records."
        )
    required = {"task_family", *OUTCOME_KEYS}
    records = [record for record in payload if isinstance(record, dict)]
    missing = required - records[0].keys()
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(sorted(missing))}")
    return records


def aggregate_by_task_family(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        family = str(record.get("task_family") or "unknown")
        entry = grouped.setdefault(family, {"count": 0, "outcomes": {key: 0 for key in OUTCOME_KEYS}})
        entry["count"] += 1
        for key in OUTCOME_KEYS:
            value = record.get(key)
            if isinstance(value, (int, float)):
                entry["outcomes"][key] += float(value)

    for entry in grouped.values():
        count = entry["count"]
        entry["outcomes"] = {key: value / count for key, value in entry["outcomes"].items()}
    return dict(sorted(grouped.items()))


def save(fig: plt.Figure, path: Path, output_format: str) -> Path:
    destination = path.with_suffix(f".{output_format}")
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_stacked(families: dict[str, dict[str, Any]], output_dir: Path, fmt: str) -> Path:
    labels = list(families)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.2), 3.8))
    left = [0.0] * len(labels)
    for key, color in zip(OUTCOME_KEYS, OUTCOME_COLORS):
        values = [families[label]["outcomes"][key] for label in labels]
        bars = ax.barh(labels, values, left=left, color=color, label=OUTCOME_LABELS[key])
        for bar, amount in zip(bars, values):
            if amount >= 0.045:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        f"{amount:.1%}", ha="center", va="center", fontsize=9)
        left = [x + y for x, y in zip(left, values)]
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("Share of examples")
    ax.set_title("Outcomes by Task Family")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return save(fig, output_dir / "task_family_outcomes_100_percent_stacked", fmt)


def plot_pies(families: dict[str, dict[str, Any]], output_dir: Path, fmt: str) -> Path:
    labels = list(families)
    columns = min(3, len(labels))
    rows = (len(labels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.4 * rows), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        values = [families[label]["outcomes"][key] for key in OUTCOME_KEYS]
        ax.pie(values, colors=OUTCOME_COLORS, startangle=110, counterclock=False,
               autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
               wedgeprops={"linewidth": 1, "edgecolor": "white"})
        ax.set_title(label)
    for ax in list(axes.flat)[len(labels):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in OUTCOME_COLORS]
    fig.legend(handles, [OUTCOME_LABELS[key] for key in OUTCOME_KEYS],
               loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    fig.suptitle("Outcome Distributions by Task Family", y=0.99)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return save(fig, output_dir / "task_family_outcome_pies", fmt)


def plot_metric_comparison(families: dict[str, dict[str, Any]], output_dir: Path, fmt: str) -> Path:
    labels = list(families)
    metric_names = ("answer_accuracy", "direction_accuracy", "both_correct")
    metric_labels = {
        "answer_accuracy": "Answer accuracy",
        "direction_accuracy": "Direction accuracy",
        "both_correct": "Answer + direction correct",
    }
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.2), 4.8))
    width = 0.8 / len(metric_names)
    positions = list(range(len(labels)))
    for index, metric in enumerate(metric_names):
        values = []
        for label in labels:
            outcomes = families[label]["outcomes"]
            if metric == "answer_accuracy":
                values.append(outcomes["answer_and_direction_correct"] + outcomes["answer_correct_direction_wrong"])
            elif metric == "direction_accuracy":
                values.append(outcomes["answer_and_direction_correct"] + outcomes["answer_wrong_direction_correct"])
            else:
                values.append(outcomes["answer_and_direction_correct"])
        offsets = [position - 0.4 + width / 2 + index * width for position in positions]
        ax.bar(offsets, values, width=width, label=metric_labels[metric])
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_ylabel("Accuracy")
    ax.set_title("Correctness Comparison by Task Family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save(fig, output_dir / "task_family_correctness_comparison", fmt)


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    records = load_records(input_path)
    families = aggregate_by_task_family(records)

    output_dir = input_path.parent / f"{input_path.stem}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_stacked(families, output_dir, args.format),
        plot_pies(families, output_dir, args.format),
        plot_metric_comparison(families, output_dir, args.format),
    ]
    summary = {
        family: {
            "count": entry["count"],
            "outcomes": {key: round(value, 8) for key, value in entry["outcomes"].items()},
        }
        for family, entry in families.items()
    }
    (output_dir / "summary.json").write_text(
        json.dumps({"input": str(input_path), "total_records": len(records), "task_families": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    for output in outputs:
        print(output)
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
