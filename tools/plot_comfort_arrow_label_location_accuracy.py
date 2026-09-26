#!/usr/bin/env python3
"""Plot object and direction accuracy against arrow-label location."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def read_scores(path: Path) -> dict[tuple[str, str], tuple[float, int]]:
    scores = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores[(row["experiment_condition"], row["answer_format"])] = (
                float(row["score"]),
                int(row["count"]),
            )
    return scores


def plot(scores: dict[tuple[str, str], tuple[float, int]], output_dir: Path) -> None:
    conditions = ["arrowhead_label", "tail_label"]
    condition_labels = ["Arrowhead", "Tail"]
    series = [
        ("object", "Object", "#0072B2", ""),
        ("direction", "Direction", "#E69F00", "///"),
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

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    x = np.arange(len(conditions))
    width = 0.34

    for index, (answer_format, label, color, hatch) in enumerate(series):
        values = [scores[(condition, answer_format)][0] for condition in conditions]
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width,
            label=label,
            color=color,
            edgecolor="white" if not hatch else "#6A4A00",
            linewidth=0.8,
            hatch=hatch,
            zorder=3,
        )
        for bar, value, condition in zip(bars, values, conditions):
            count = scores[(condition, answer_format)][1]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.022,
                f"{100 * value:.1f}%\n($n={count:,}$)",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_xticks(x, condition_labels)
    ax.set_xlabel("Label location on arrow")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("COMFORT arrow-label endpoint diagnostic", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "arrow_label_location_vs_accuracy"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/comfort_arrow_label_endpoint_8/analysis/arrow_label_endpoint.csv"
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
