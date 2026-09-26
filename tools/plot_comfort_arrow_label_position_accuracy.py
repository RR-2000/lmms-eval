#!/usr/bin/env python3
"""Plot object and direction accuracy across arrow-label positions."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def read_scores(path: Path) -> dict[str, dict[float, tuple[float, int]]]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)["records"]

    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for record in records:
        position = float(record["experiment_condition"].removeprefix("position_"))
        grouped[(record["answer_format"], position)].append(float(record["score"]))

    result: dict[str, dict[float, tuple[float, int]]] = defaultdict(dict)
    for (answer_format, position), values in grouped.items():
        result[answer_format][position] = (sum(values) / len(values), len(values))
    return dict(result)


def wilson_half_width(accuracy: float, count: int, z: float = 1.96) -> float:
    """Return the approximate half-width of a 95% Wilson interval."""
    denominator = 1.0 + z * z / count
    return (
        z
        * math.sqrt(accuracy * (1.0 - accuracy) / count + z * z / (4.0 * count * count))
        / denominator
    )


def plot(scores: dict[str, dict[float, tuple[float, int]]], output_dir: Path) -> None:
    positions = sorted(scores["object"])
    series = [
        ("object", "Object", "#0072B2", "o"),
        ("direction", "Direction", "#E69F00", "s"),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.35), constrained_layout=True)
    for answer_format, label, color, marker in series:
        values = np.array([scores[answer_format][position][0] for position in positions])
        counts = np.array([scores[answer_format][position][1] for position in positions])
        errors = np.array(
            [wilson_half_width(value, int(count)) for value, count in zip(values, counts)]
        )
        ax.errorbar(
            positions,
            values,
            yerr=errors,
            label=label,
            color=color,
            marker=marker,
            markersize=6.5,
            linewidth=2.2,
            capsize=3,
            capthick=1.2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        for position, value in zip(positions, values):
            if answer_format == "object" or value < 0.08:
                offset = (0, 9)
                vertical_alignment = "bottom"
            else:
                offset = (0, -13)
                vertical_alignment = "top"
            ax.annotate(
                f"{100 * value:.1f}%",
                (position, value),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
                color=color,
                fontsize=8.2,
                fontweight="bold",
            )

    ax.axhline(
        0.25,
        color="#777777",
        linestyle=(0, (4, 3)),
        linewidth=1.1,
        label="Chance (4 choices)",
        zorder=1,
    )
    ax.set_xticks(positions, [f"{position:.1f}" for position in positions])
    ax.set_xlabel("Label position along arrow (0 = origin, 1 = arrowhead)")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_xlim(-0.035, 1.035)
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("COMFORT arrow-label position sweep", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "arrow_label_position_vs_accuracy"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/comfort_arrow_label_endpoint_8/submissions/"
            "comfort_arrow_label_endpoint_qwen3_vl_experiments.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comfort_arrow_label_endpoint_8/analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(read_scores(args.input), args.output_dir)


if __name__ == "__main__":
    main()
