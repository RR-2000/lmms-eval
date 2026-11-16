#!/usr/bin/env python3
"""Build plots and human-readable reports for the split MOVi-A diagnostic run."""

from __future__ import annotations

import csv
import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "kubric_movi_a_motion_diagnostics_full_8"
REPORT_DIR = RUN_DIR / "analysis"
FIGURE_DIR = REPORT_DIR / "figures"

DISPLAY_NAMES = {
    "kubric_movi_a_motion_oracle_ladder": "Oracle ladder",
    "kubric_movi_a_motion_perception_diagnostics": "Perception",
    "kubric_movi_a_motion_temporal_diagnostics": "Temporal",
    "kubric_movi_a_motion_coordinate_diagnostics": "Coordinate",
    "kubric_movi_a_motion_calculation_diagnostics": "Calculation",
    "kubric_movi_a_motion_representation_diagnostics": "Representation",
    "kubric_movi_a_motion_precision_diagnostics": "Precision",
    "kubric_movi_a_motion_robustness_controls": "Robustness",
}

SUITE_DESCRIPTIONS = {
    "Oracle ladder": "Progressively stronger ground-truth assistance, through an answer-letter oracle.",
    "Perception": "Bounding boxes, masks, visibility, and object tracks.",
    "Temporal": "Labels, endpoints, query triplets, intervals, events, and timing.",
    "Coordinate": "World/camera state, basis, distance trace, trajectory plot, and direction conventions.",
    "Calculation": "Formulas and intermediate arithmetic quantities.",
    "Representation": "Equivalent table, JSON, and graphical state representations.",
    "Precision": "Ground-truth state at two, four, and eight decimal places.",
    "Robustness": "Coordinate noise, reversed frames, irrelevant state, and mismatched state.",
}

COLORS = {
    "baseline": "#6B7280",
    "overall": "#2563EB",
    "assisted": "#0F9D76",
    "positive": "#0F9D76",
    "negative": "#C43C39",
    "error": "#D97706",
    "runtime": "#7C3AED",
}


def load_rows() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    runs: list[dict] = []
    result_files = sorted(RUN_DIR.glob("part_*/Qwen__Qwen3-VL-8B-Instruct/*_results.json"))
    if len(result_files) != 4:
        raise RuntimeError(f"Expected four split result files, found {len(result_files)}")

    for path in result_files:
        with path.open() as handle:
            payload = json.load(handle)
        sample_counts = payload["n-samples"]
        part = path.parents[1].name
        runtime_seconds = float(payload["total_evaluation_time_seconds"])
        runs.append(
            {
                "part": part,
                "tasks": len(payload["results"]),
                "samples": sum(int(v["effective"]) for v in sample_counts.values()),
                "runtime_seconds": runtime_seconds,
                "runtime_days": runtime_seconds / 86400,
                "tokens_per_second": float(payload["throughput"]["avg_speed"]),
                "source": str(path.relative_to(ROOT)),
            }
        )
        for task, metrics in payload["results"].items():
            rows.append(
                {
                    "task": task,
                    "suite": DISPLAY_NAMES[task],
                    "samples": int(sample_counts[task]["effective"]),
                    "accuracy": float(metrics["accuracy,none"]),
                    "accuracy_stderr": float(metrics["accuracy_stderr_clt,none"]),
                    "baseline_accuracy": float(metrics["baseline_accuracy,none"]),
                    "assisted_accuracy": float(metrics["assisted_accuracy,none"]),
                    "paired_gain": float(metrics["paired_scaffold_gain,none"]),
                    "parse_success": float(metrics["parse_success,none"]),
                    "numeric_error": float(metrics["numeric_option_error,none"]),
                    "part": part,
                    "source": str(path.relative_to(ROOT)),
                    "model": payload["model_name"],
                    "git_hash": payload["git_hash"],
                }
            )
    return rows, runs


def save_csv(rows: list[dict]) -> None:
    path = REPORT_DIR / "results_summary.csv"
    fields = [
        "suite", "task", "samples", "accuracy", "accuracy_stderr",
        "baseline_accuracy", "assisted_accuracy", "paired_gain",
        "parse_success", "numeric_error", "part", "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["accuracy"], reverse=True))


def style_axis(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=16)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color="#4B5563")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)


def plot_accuracy(rows: list[dict]) -> Path:
    ordered = sorted(rows, key=lambda row: row["assisted_accuracy"])
    names = [row["suite"] for row in ordered]
    y = np.arange(len(ordered))
    height = 0.25
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.barh(y - height, [100 * row["baseline_accuracy"] for row in ordered], height, label="Baseline", color=COLORS["baseline"])
    ax.barh(y, [100 * row["accuracy"] for row in ordered], height, label="All conditions", color=COLORS["overall"])
    bars = ax.barh(y + height, [100 * row["assisted_accuracy"] for row in ordered], height, label="Assisted only", color=COLORS["assisted"])
    for bar in bars:
        ax.text(bar.get_width() + 0.35, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1f}%", va="center", fontsize=8)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 43)
    ax.set_xlabel("Accuracy (%) — higher is better")
    style_axis(ax, "Accuracy by diagnostic suite", "Baseline is the same deterministic video-only question set repeated in each suite.")
    ax.legend(frameon=False, ncols=3, loc="lower right")
    fig.tight_layout()
    path = FIGURE_DIR / "suite_accuracy.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gain(rows: list[dict]) -> Path:
    ordered = sorted(rows, key=lambda row: row["paired_gain"])
    values = [100 * row["paired_gain"] for row in ordered]
    colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in values]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh([row["suite"] for row in ordered], values, color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    for bar, value in zip(bars, values):
        offset = 0.35 if value >= 0 else -0.35
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:+.1f} pp", ha="left" if value >= 0 else "right", va="center", fontsize=9)
    ax.set_xlim(min(-3, min(values) - 2), max(values) + 3)
    ax.set_xlabel("Mean paired change from baseline (percentage points) — higher is better")
    style_axis(ax, "Paired scaffold gain", "Each assisted response is compared with the baseline response for the same source question.")
    fig.tight_layout()
    path = FIGURE_DIR / "paired_scaffold_gain.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error(rows: list[dict]) -> Path:
    ordered = sorted(rows, key=lambda row: row["numeric_error"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh([row["suite"] for row in ordered], [row["numeric_error"] for row in ordered], color=COLORS["error"])
    for bar in bars:
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("Mean absolute numeric error of selected option — lower is better")
    style_axis(ax, "Numeric option error", "Computed when the selected answer option contains a numeric value.")
    fig.tight_layout()
    path = FIGURE_DIR / "numeric_option_error.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_runs(runs: list[dict]) -> Path:
    ordered = sorted(runs, key=lambda run: run["part"])
    labels = [run["part"].replace("part_", "").replace("_", "\n") for run in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(labels, [run["samples"] for run in ordered], color=COLORS["overall"])
    axes[0].set_ylabel("Effective evaluations")
    axes[0].set_title("Work assigned", loc="left", fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].tick_params(axis="x", labelsize=8)
    axes[1].bar(labels, [run["runtime_days"] for run in ordered], color=COLORS["runtime"])
    axes[1].set_ylabel("Evaluation time (days)")
    axes[1].set_title("Observed runtime", loc="left", fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].tick_params(axis="x", labelsize=8)
    fig.suptitle("Four-way PBS split", x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIGURE_DIR / "workload_runtime.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def markdown_report(rows: list[dict], runs: list[dict]) -> str:
    ranked = sorted(rows, key=lambda row: row["accuracy"], reverse=True)
    gain_ranked = sorted(rows, key=lambda row: row["paired_gain"], reverse=True)
    total_samples = sum(row["samples"] for row in rows)
    micro_accuracy = sum(row["samples"] * row["accuracy"] for row in rows) / total_samples
    baseline = rows[0]["baseline_accuracy"]
    max_runtime = max(run["runtime_days"] for run in runs)
    serial_runtime = sum(run["runtime_days"] for run in runs)
    accuracy_error_corr = float(np.corrcoef([row["accuracy"] for row in rows], [row["numeric_error"] for row in rows])[0, 1])

    table_lines = [
        "| Rank | Suite | N | Overall accuracy | Assisted accuracy | Paired gain | Numeric error |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, 1):
        table_lines.append(
            f"| {rank} | {row['suite']} | {row['samples']:,} | {pct(row['accuracy'])} | "
            f"{pct(row['assisted_accuracy'])} | {100 * row['paired_gain']:+.2f} pp | {row['numeric_error']:.3f} |"
        )

    run_lines = [
        "| PBS part | Evaluations | Runtime | Generation speed |",
        "|---|---:|---:|---:|",
    ]
    for run in sorted(runs, key=lambda item: item["part"]):
        run_lines.append(
            f"| `{run['part']}` | {run['samples']:,} | {run['runtime_days']:.2f} days | {run['tokens_per_second']:.3f} tokens/s |"
        )

    descriptions = "\n".join(f"- **{name}:** {description}" for name, description in SUITE_DESCRIPTIONS.items())
    sources = "\n".join(f"- `{run['source']}`" for run in sorted(runs, key=lambda item: item["part"]))

    return f"""# MOVi-A Motion Diagnostics: Qwen3-VL-8B-Instruct

Generated from the completed four-way PBS evaluation. This report summarizes the aggregate metrics saved by `lmms-eval`; sample-level model outputs were not logged for these runs.

## Executive summary

- **Best overall suite:** {ranked[0]['suite']} at **{pct(ranked[0]['accuracy'])}** overall and **{pct(ranked[0]['assisted_accuracy'])}** on assisted conditions.
- **Largest paired improvements:** {gain_ranked[0]['suite']} (**{100 * gain_ranked[0]['paired_gain']:+.2f} pp**), {gain_ranked[1]['suite']} (**{100 * gain_ranked[1]['paired_gain']:+.2f} pp**), and {gain_ranked[2]['suite']} (**{100 * gain_ranked[2]['paired_gain']:+.2f} pp**).
- **Baseline:** **{pct(baseline)}**. Representation, perception, temporal, and coordinate scaffolds remain close to it, suggesting that exposing more visual/state information alone does not reliably resolve the task.
- **Negative control result:** robustness assistance produces **{100 * gain_ranked[-1]['paired_gain']:+.2f} pp** paired gain and {pct(gain_ranked[-1]['assisted_accuracy'])} assisted accuracy, below baseline. The corrupting/mismatched controls therefore behave as intended and do not provide a shortcut.
- **Numeric reasoning:** the oracle ladder has the lowest numeric-option error ({ranked[0]['numeric_error']:.3f}). Accuracy and numeric error are strongly inversely associated across the eight suites (Pearson r = {accuracy_error_corr:.3f}), though eight suites are too few for a strong inferential claim.
- **Coverage:** {total_samples:,} expanded evaluations, all with **100% parse success**. The micro-average accuracy across the expanded records is **{pct(micro_accuracy)}**; it should not be interpreted as an independent-question accuracy because baseline questions recur across suites.

![Accuracy by suite](figures/suite_accuracy.png)

## Full results

{chr(10).join(table_lines)}

Definitions:

- **Overall accuracy** averages baseline and every assisted condition in that suite.
- **Assisted accuracy** excludes the video-only baseline condition.
- **Paired gain** is the mean assisted-minus-baseline correctness change for the same source question. This is the clearest measure of whether a scaffold helps.
- **Numeric error** is the mean absolute difference between the numeric value in the selected option and the gold value, when a numeric option can be parsed. Lower is better.

![Paired scaffold gain](figures/paired_scaffold_gain.png)

![Numeric option error](figures/numeric_option_error.png)

## Interpretation

### 1. The main bottleneck is not uniform

The oracle ladder, precision, and calculation suites show substantial gains. Their assisted accuracies are {pct(gain_ranked[0]['assisted_accuracy'])}, {pct(gain_ranked[1]['assisted_accuracy'])}, and {pct(gain_ranked[2]['assisted_accuracy'])}, respectively. This indicates that many failures can be repaired by sufficiently direct ground-truth assistance, more precise state, or explicit intermediate quantities.

### 2. Raw representations and perceptual aids have weak leverage

Representation, perception, temporal, and coordinate suites yield paired gains between {100 * min(row['paired_gain'] for row in rows if row['suite'] not in {'Robustness'}):.2f} and {100 * max(row['paired_gain'] for row in rows if row['suite'] in {'Representation', 'Perception', 'Temporal', 'Coordinate'}):.2f} percentage points. The model is often unable to turn these aids into the correct five-choice answer. This is consistent with a downstream integration/reasoning bottleneck, although aggregate results cannot isolate the exact failure stage.

### 3. Precision and arithmetic information are especially useful

Precision and calculation are the strongest non-oracle suites. This supports testing whether errors arise from rounding, distance comparisons near decision boundaries, or failure to carry intermediate calculations through to the final option.

### 4. Robustness controls validate the diagnostic design

Robustness is the only suite with negative paired gain. Noise, reversal, irrelevant information, and explicitly mismatched state do not inflate performance; together they reduce it. A condition-level breakdown would show which corruption is responsible, but those values are not stored in the aggregate result JSON.

## What each suite tests

{descriptions}

## Run coverage and performance

{chr(10).join(run_lines)}

![Workload and runtime](figures/workload_runtime.png)

If the four jobs ran concurrently, the observed critical-path evaluation time was about **{max_runtime:.2f} days**, versus **{serial_runtime:.2f} GPU-days** summed across all parts. Part 0 was the largest and slowest partition.

## Limitations and recommended follow-up

1. The runs used `log_samples=false`, so this report cannot break metrics down by scaffold condition, source subtask, sequence, or individual failure example.
2. The result files report suite-level CLT standard errors only for the overall accuracy. The plotted comparisons therefore emphasize effect sizes rather than formal significance tests.
3. Baseline questions are repeated in each suite, so the 40,900 rows are expanded conditions, not 40,900 independent source questions.
4. For the next run, enable `--log_samples` and retain `source_qid`, `subtask`, and `condition`. Then plot rescue/regression counts and condition-level paired gains, especially within oracle, calculation, precision, and robustness.

## Provenance

- Model: `{rows[0]['model']}`
- `lmms-eval` Git revision: `{rows[0]['git_hash']}`
- Model arguments: `max_num_frames=32,pretrained=Qwen/Qwen3-VL-8B-Instruct`
- Batch size: 1
- Result sources:
{sources}
"""


def add_wrapped_text(ax, text: str, x: float, y: float, width: int = 105, size: float = 10, line_spacing: float = 0.038) -> float:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if paragraph:
            lines.extend(textwrap.wrap(paragraph, width=width))
        else:
            lines.append("")
    for line in lines:
        ax.text(x, y, line, transform=ax.transAxes, fontsize=size, va="top")
        y -= line_spacing
    return y


def pdf_report(rows: list[dict], runs: list[dict], figures: list[Path]) -> None:
    path = REPORT_DIR / "MOVi_A_motion_diagnostics_analysis.pdf"
    ranked = sorted(rows, key=lambda row: row["accuracy"], reverse=True)
    total_samples = sum(row["samples"] for row in rows)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes((0.08, 0.06, 0.84, 0.88))
        ax.axis("off")
        ax.text(0, 1, "MOVi-A Motion Diagnostics", fontsize=24, fontweight="bold", va="top")
        ax.text(0, 0.955, "Qwen3-VL-8B-Instruct · four-way PBS run", fontsize=13, color="#4B5563", va="top")
        y = 0.89
        summary = (
            f"The completed experiment contains {total_samples:,} expanded evaluations across eight diagnostic suites. "
            f"The video-only baseline is {pct(rows[0]['baseline_accuracy'])}. The oracle ladder performs best "
            f"({pct(ranked[0]['accuracy'])} overall; {100 * ranked[0]['paired_gain']:+.2f} percentage points paired gain), "
            "followed by calculation and precision assistance. Representation, perception, temporal, and coordinate "
            "aids have limited effect, while robustness controls reduce performance below baseline."
        )
        y = add_wrapped_text(ax, summary, 0, y, width=92, size=11, line_spacing=0.035)
        y -= 0.035
        ax.text(0, y, "Key findings", fontsize=15, fontweight="bold", va="top")
        y -= 0.045
        findings = [
            f"Oracle ladder: {pct(ranked[0]['assisted_accuracy'])} assisted accuracy, {100 * ranked[0]['paired_gain']:+.2f} pp paired gain.",
            "Calculation and precision are the strongest non-oracle interventions.",
            "All responses parse successfully, ruling out answer-format failure as the explanation.",
            "Robustness is the only negative-gain suite, supporting the intended role of the controls.",
            "No sample logs were saved; condition-level and example-level diagnosis requires a follow-up run with --log_samples.",
        ]
        for item in findings:
            y = add_wrapped_text(ax, "• " + item, 0.02, y, width=88, size=10.5, line_spacing=0.034)
            y -= 0.012
        ax.text(0, 0.05, f"Generated from lmms-eval revision {rows[0]['git_hash']}", fontsize=8, color="#6B7280")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for figure_path in figures[:3]:
            image = plt.imread(figure_path)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes((0.04, 0.04, 0.92, 0.92))
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes((0.03, 0.10, 0.94, 0.82))
        ax.axis("off")
        ax.set_title("Complete suite metrics", loc="left", fontsize=18, fontweight="bold", pad=18)
        table_rows = []
        for rank, row in enumerate(ranked, 1):
            table_rows.append([
                rank, row["suite"], f"{row['samples']:,}", pct(row["accuracy"]),
                pct(row["assisted_accuracy"]), f"{100 * row['paired_gain']:+.2f} pp", f"{row['numeric_error']:.3f}",
            ])
        table = ax.table(
            cellText=table_rows,
            colLabels=["Rank", "Suite", "N", "Overall", "Assisted", "Paired gain", "Numeric error"],
            cellLoc="center", colLoc="center", loc="upper center",
            colWidths=[0.06, 0.20, 0.10, 0.12, 0.12, 0.14, 0.14],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.7)
        for (row_index, _), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#DCEAFE")
                cell.set_text_props(weight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#F3F4F6")
        ax.text(0.02, 0.20, "Overall includes baseline and all assisted conditions. Paired gain compares assisted and baseline correctness for the same source question.", transform=ax.transAxes, fontsize=9)
        ax.text(0.02, 0.15, "Numeric error is lower-is-better. Parse success is 100% for every suite.", transform=ax.transAxes, fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        image = plt.imread(figures[3])
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes((0.04, 0.04, 0.92, 0.92))
        ax.imshow(image)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows, runs = load_rows()
    save_csv(rows)
    figures = [plot_accuracy(rows), plot_gain(rows), plot_error(rows), plot_runs(runs)]
    report = markdown_report(rows, runs)
    (REPORT_DIR / "ANALYSIS.md").write_text(report)
    pdf_report(rows, runs, figures)
    print(f"Wrote {REPORT_DIR / 'ANALYSIS.md'}")
    print(f"Wrote {REPORT_DIR / 'MOVi_A_motion_diagnostics_analysis.pdf'}")


if __name__ == "__main__":
    main()
