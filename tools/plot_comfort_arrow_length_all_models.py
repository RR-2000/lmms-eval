#!/usr/bin/env python3
"""Plot COMFORT arrow-length sweep accuracy for every completed model."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


MODEL_NAMES = {
    "cambrians": "Cambrian-S-7B",
    "internvideo3": "InternVideo3-8B",
    "internvl3_5": "InternVL3.5-8B",
    "llava_onevision2": "LLaVA-OneVision2-8B",
    "sat": "SAT (Qwen2.5-VL)",
    "spatialladder": "SpatialLadder-3B",
    "vst": "VST-7B-RL",
}

MODEL_STYLES = {
    "cambrians": ("#0072B2", "o"),
    "internvideo3": ("#E69F00", "s"),
    "internvl3_5": ("#009E73", "^"),
    "llava_onevision2": ("#D55E00", "D"),
    "sat": ("#CC79A7", "P"),
    "spatialladder": ("#7A7A7A", "X"),
    "vst": ("#56B4E9", "v"),
}


def read_submission(path: Path) -> dict[str, dict[float, tuple[float, int]]]:
    """Return mean score and count by answer format and arrow length."""
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)["records"]

    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for record in records:
        key = (record["answer_format"], float(record["experiment_condition"]))
        grouped[key].append(float(record["score"]))

    scores: dict[str, dict[float, tuple[float, int]]] = defaultdict(dict)
    for (answer_format, length), values in grouped.items():
        scores[answer_format][length] = (sum(values) / len(values), len(values))
    return dict(scores)


def discover_submissions(
    input_dir: Path,
) -> dict[str, dict[str, dict[float, tuple[float, int]]]]:
    """Discover one submission JSON for every model directory."""
    submissions: dict[str, dict[str, dict[float, tuple[float, int]]]] = {}
    for path in sorted(input_dir.glob("*/submissions/*.json")):
        model_key = path.parent.parent.name
        if model_key in submissions:
            raise ValueError(f"Multiple submission files found for {model_key}")
        submissions[model_key] = read_submission(path)

    if not submissions:
        raise FileNotFoundError(f"No submission JSON files found under {input_dir}")

    unknown = sorted(set(submissions) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Add display names and styles for models: {', '.join(unknown)}")
    return submissions


def wilson_half_width(accuracy: float, count: int, z: float = 1.96) -> float:
    """Return the approximate half-width of a 95% Wilson interval."""
    denominator = 1.0 + z * z / count
    return (
        z
        * math.sqrt(accuracy * (1.0 - accuracy) / count + z * z / (4.0 * count * count))
        / denominator
    )


def validate_and_get_lengths(
    submissions: dict[str, dict[str, dict[float, tuple[float, int]]]],
) -> list[float]:
    first_model = next(iter(submissions.values()))
    lengths = sorted(first_model["object"])
    expected = set(lengths)
    for model_key, scores in submissions.items():
        for answer_format in ("object", "direction"):
            actual = set(scores.get(answer_format, {}))
            if actual != expected:
                raise ValueError(
                    f"{model_key}/{answer_format} has lengths {sorted(actual)}, "
                    f"expected {lengths}"
                )
    return lengths


def plot(
    submissions: dict[str, dict[str, dict[float, tuple[float, int]]]],
    output_dir: Path,
) -> None:
    lengths = validate_and_get_lengths(submissions)

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

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.7),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.28, wspace=0.03)

    ordered_models = [key for key in MODEL_NAMES if key in submissions]
    for ax, answer_format, panel_title in zip(
        axes, ("object", "direction"), ("Object questions", "Direction questions")
    ):
        for model_key in ordered_models:
            color, marker = MODEL_STYLES[model_key]
            values = np.array(
                [submissions[model_key][answer_format][length][0] for length in lengths]
            )
            counts = np.array(
                [submissions[model_key][answer_format][length][1] for length in lengths]
            )
            errors = np.array(
                [
                    wilson_half_width(value, int(count))
                    for value, count in zip(values, counts)
                ]
            )
            ax.errorbar(
                lengths,
                values,
                yerr=errors,
                color=color,
                marker=marker,
                markersize=5.7,
                linewidth=2.0,
                capsize=2.3,
                capthick=0.9,
                markeredgecolor="white",
                markeredgewidth=0.65,
                label=MODEL_NAMES[model_key],
                zorder=3,
            )

        ax.axhline(
            0.25,
            color="#666666",
            linestyle=(0, (4, 3)),
            linewidth=1.1,
            zorder=1,
        )
        ax.set_title(panel_title, loc="left", fontweight="bold")
        ax.set_xticks(lengths, [f"{length:g}" for length in lengths])
        ax.set_xlim(min(lengths) - 0.055, max(lengths) + 0.055)
        ax.set_ylim(0.1, 0.86)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Exact-match accuracy")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.supxlabel("Arrow length (reference bounding-box diagonals)", y=0.195)
    fig.suptitle(
        "COMFORT arrow-length sweep across models",
        x=0.075,
        y=0.96,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], color="#666666", linestyle=(0, (4, 3)), linewidth=1.1)
    )
    labels.append("Chance (4 choices)")
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        columnspacing=1.5,
        handlelength=2.5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "arrow_length_sweep_all_models"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/diagnosis_final_comfort_arrow_length_sweep"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnosis_final_comfort_arrow_length_sweep/analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(discover_submissions(args.input_dir), args.output_dir)


if __name__ == "__main__":
    main()
