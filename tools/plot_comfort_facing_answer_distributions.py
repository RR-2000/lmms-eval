#!/usr/bin/env python3
"""Plot answer distributions for the COMFORT facing-direction diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


DIRECTIONS = [
    "right",
    "down-right",
    "down",
    "down-left",
    "left",
    "up-left",
    "up",
    "up-right",
]

# Distinct, report-friendly colors. The gold class (down-left) is highlighted.
COLORS = {
    "right": "#0072B2",
    "down-right": "#56B4E9",
    "down": "#009E73",
    "down-left": "#E69F00",
    "left": "#D55E00",
    "up-left": "#CC79A7",
    "up": "#8C8C8C",
    "up-right": "#6A3D9A",
}

ABBREVIATIONS = {
    "right": "R",
    "down-right": "DR",
    "down": "D",
    "down-left": "DL",
    "left": "L",
    "up-left": "UL",
    "up": "U",
    "up-right": "UR",
}

DISPLAY_NAMES = {
    "openai": "OpenAI",
    "internvideo3": "InternVideo3",
    "cambrians": "CambrianS",
    "internvl3_5": "InternVL3.5",
    "llava_onevision2": "LLaVA-OneVision2",
    "sat": "SAT",
    "spatialladder": "SpatialLadder",
    "vst": "VST",
}


def load_submission(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["records"]
    counts = Counter(record["parsed_answer"] for record in records)
    objects = sorted({record["reference_object"] for record in records})
    by_object = {
        obj: Counter(
            record["parsed_answer"]
            for record in records
            if record["reference_object"] == obj
        )
        for obj in objects
    }
    object_totals = {
        obj: sum(by_object[obj].values())
        for obj in objects
    }
    return {
        "n": len(records),
        "counts": counts,
        "by_object": by_object,
        "object_totals": object_totals,
        "accuracy": float(payload["metrics"]["accuracy"]),
        "parse_success": float(payload["metrics"]["parse_success"]),
    }


def collect_results(root: Path, qwen_path: Path) -> dict[str, dict]:
    results = {}
    for path in sorted(root.glob("*/submissions/*.json")):
        key = path.parents[1].name
        results[DISPLAY_NAMES.get(key, key)] = load_submission(path)
    results["Qwen3-VL-8B"] = load_submission(qwen_path)
    return results


def make_figure(results: dict[str, dict], output_dir: Path) -> None:
    preferred_order = [
        "OpenAI",
        "InternVideo3",
        "CambrianS",
        "Qwen3-VL-8B",
        "InternVL3.5",
        "LLaVA-OneVision2",
        "SAT",
        "SpatialLadder",
        "VST",
    ]
    models = [name for name in preferred_order if name in results]
    models.extend(sorted(set(results) - set(models)))
    objects = ["bicycle", "car", "dog", "duck", "horse"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(10.6, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[1.16, 1.0])

    # Panel (a): complete answer distribution for each model and the gold labels.
    ax = figure.add_subplot(grid[0])
    rows = models + ["Gold labels"]
    y_positions = np.arange(len(rows))
    left = np.zeros(len(rows))
    n = results[models[0]]["n"]

    for direction in DIRECTIONS:
        values = []
        for model in models:
            values.append(100.0 * results[model]["counts"][direction] / results[model]["n"])
        values.append(100.0 if direction == "down-left" else 0.0)
        values_array = np.asarray(values)
        ax.barh(
            y_positions,
            values_array,
            left=left,
            height=0.70,
            color=COLORS[direction],
            edgecolor="white",
            linewidth=0.45,
            label=direction,
        )
        left += values_array

    for row, model in enumerate(models):
        accuracy = 100.0 * results[model]["accuracy"]
        ax.text(101.4, row, f"{accuracy:.1f}%", va="center", ha="left", fontsize=8.5)
    ax.text(101.4, len(models), "reference", va="center", ha="left", fontsize=8.5)
    ax.text(101.4, -0.95, "Exact acc.", va="bottom", ha="left", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y_positions, rows)
    ax.invert_yaxis()
    ax.set_xlim(0, 114)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Share of parsed answers (%)")
    ax.set_title(f"(a) Overall parsed-answer distribution ($n={n}$ per model)", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=COLORS[d], label=d) for d in DIRECTIONS]
    ax.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.19),
        frameon=False,
        columnspacing=1.3,
        handlelength=1.5,
    )

    # Panel (b): dominant answer and its mass for each object category.
    ax2 = figure.add_subplot(grid[1])
    dominant_index = np.zeros((len(models), len(objects)), dtype=int)
    dominant_share = np.zeros_like(dominant_index, dtype=float)
    for row, model in enumerate(models):
        for col, obj in enumerate(objects):
            counter = results[model]["by_object"][obj]
            direction, count = max(
                ((direction, counter[direction]) for direction in DIRECTIONS),
                key=lambda item: item[1],
            )
            dominant_index[row, col] = DIRECTIONS.index(direction)
            dominant_share[row, col] = 100.0 * count / results[model]["object_totals"][obj]

    cmap = ListedColormap([COLORS[direction] for direction in DIRECTIONS])
    ax2.imshow(dominant_index, cmap=cmap, vmin=-0.5, vmax=7.5, aspect="auto")
    for row in range(len(models)):
        for col in range(len(objects)):
            direction = DIRECTIONS[dominant_index[row, col]]
            share = dominant_share[row, col]
            text_color = "black" if direction in {"down-right", "down-left", "up"} else "white"
            ax2.text(
                col,
                row,
                f"{ABBREVIATIONS[direction]}  {share:.0f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="bold",
            )

    ax2.set_xticks(np.arange(len(objects)), [obj.capitalize() for obj in objects])
    ax2.set_yticks(np.arange(len(models)), models)
    ax2.set_xticks(np.arange(-0.5, len(objects), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=2)
    ax2.tick_params(which="minor", bottom=False, left=False)
    ax2.tick_params(axis="both", length=0)
    ax2.set_title(
        "(b) Dominant answer by reference-object category (label and within-category share)",
        loc="left",
        fontweight="bold",
    )
    for spine in ax2.spines.values():
        spine.set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "comfort_facing_answer_distributions.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "comfort_facing_answer_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_comfort_gt_component_facing_direction_no_bbox/figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = collect_results(args.root, args.qwen)
    make_figure(results, args.output_dir)


if __name__ == "__main__":
    main()
