#!/usr/bin/env python3
"""Plot object and direction accuracy across COMFORT arrow-length scales."""

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
    """Return mean score and sample count by answer format and arrow length."""
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)["records"]

    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for record in records:
        length = float(record["experiment_condition"])
        grouped[(record["answer_format"], length)].append(float(record["score"]))

    result: dict[str, dict[float, tuple[float, int]]] = defaultdict(dict)
    for (answer_format, length), values in grouped.items():
        result[answer_format][length] = (sum(values) / len(values), len(values))
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
    lengths = sorted(scores["object"])
    if set(lengths) != set(scores["direction"]):
        raise ValueError("Object and direction records do not share the same lengths")

    series = [
        ("object", "Object", "#0072B2", "o"),
        ("direction", "Direction", "#E69F00", "s"),
    ]
    accuracies = {
        answer_format: np.array(
            [scores[answer_format][length][0] for length in lengths]
        )
        for answer_format, *_ in series
    }

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
        values = accuracies[answer_format]
        counts = np.array(
            [scores[answer_format][length][1] for length in lengths]
        )
        errors = np.array(
            [wilson_half_width(value, int(count)) for value, count in zip(values, counts)]
        )
        ax.errorbar(
            lengths,
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

        other_format = "direction" if answer_format == "object" else "object"
        for index, (length, value) in enumerate(zip(lengths, values)):
            above_other = value >= accuracies[other_format][index]
            offset = (0, 9) if above_other else (0, -13)
            vertical_alignment = "bottom" if above_other else "top"
            ax.annotate(
                f"{100 * value:.1f}%",
                (length, value),
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
    ax.set_xticks(lengths, [f"{length:g}" for length in lengths])
    ax.set_xlabel("Arrow length (reference bounding-box diagonals)")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_xlim(min(lengths) - 0.055, max(lengths) + 0.055)
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("COMFORT arrow-length sweep", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "arrow_length_vs_accuracy"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/comfort_arrow_length_sweep_8/submissions/"
            "comfort_arrow_length_sweep_qwen3_vl_experiments.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comfort_arrow_length_sweep_8/analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(read_scores(args.input), args.output_dir)


if __name__ == "__main__":
    main()
