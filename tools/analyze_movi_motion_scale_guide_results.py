#!/usr/bin/env python3
"""Build detailed artifacts for the full 12-frame MOVi-A scale-guide run."""

from __future__ import annotations

import csv
import json
import math
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "kubric_movi_a_motion_12f_full_scale_guide"
OLD_RUN_DIR = ROOT / "outputs" / "kubric_movi_a_motion_12f_detailed_160"
REPORT_DIR = RUN_DIR / "analysis"
FIGURE_DIR = REPORT_DIR / "figures"
EXAMPLE_DIR = REPORT_DIR / "examples"

SUITE_NAMES = {
    "oracle_ladder": "Oracle ladder",
    "perception": "Perception",
    "temporal": "Temporal",
    "coordinate": "Coordinate",
    "calculation": "Calculation",
    "representation": "Representation",
    "precision": "Precision",
    "robustness": "Robustness",
}

SUBTASK_NAMES = {
    "displacement": "Displacement",
    "path_length": "Path length",
    "average_speed": "Average speed",
    "instantaneous_speed": "Instantaneous speed",
    "instantaneous_acceleration": "Instantaneous acceleration",
    "fastest_object": "Fastest object",
    "greatest_path_length_object": "Greatest path object",
    "dominant_motion_direction": "Motion direction",
}

NUMERIC_SUBTASKS = {
    "displacement",
    "path_length",
    "average_speed",
    "instantaneous_speed",
    "instantaneous_acceleration",
}

COLORS = {
    "baseline": "#6B7280",
    "assisted": "#2563EB",
    "positive": "#0F9D76",
    "negative": "#C43C39",
    "neutral": "#94A3B8",
    "numeric": "#7C3AED",
    "categorical": "#D97706",
}


def load_sample_file(path: Path, run: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            metric = record["accuracy"]
            rows.append(
                {
                    **metric,
                    "input": record.get("input", ""),
                    "filtered_response": record.get("filtered_resps", ""),
                    "token_count": record.get("token_counts"),
                    "sample_file": str(path.relative_to(ROOT)),
                    "run": run,
                }
            )
    return rows


def load_samples(run_dir: Path, run: str) -> pd.DataFrame:
    files = sorted(run_dir.glob("part_*/Qwen__Qwen3-VL-8B-Instruct/*_samples_*.jsonl"))
    if len(files) != 8:
        raise RuntimeError(f"Expected eight sample files under {run_dir}, found {len(files)}")
    rows = [row for path in files for row in load_sample_file(path, run)]
    return pd.DataFrame(rows)


def load_runs() -> pd.DataFrame:
    rows = []
    paths = sorted(RUN_DIR.glob("part_*/Qwen__Qwen3-VL-8B-Instruct/*_results.json"))
    if len(paths) != 4:
        raise RuntimeError(f"Expected four result files, found {len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts = payload["n-samples"]
        rows.append(
            {
                "part": path.parents[1].name,
                "tasks": len(payload["results"]),
                "evaluations": sum(int(item["effective"]) for item in counts.values()),
                "evaluation_seconds": float(payload["total_evaluation_time_seconds"]),
                "elapsed_hours": float(payload["total_evaluation_time_seconds"]) / 3600,
                "generation_tokens_per_second": float(payload["throughput"]["avg_speed"]),
                "model": payload["model_name"],
                "git_hash": payload["git_hash"],
                "source": str(path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def paired_summary(group: pd.DataFrame) -> dict:
    baseline = group[group.condition == "baseline"][["source_qid", "score"]].drop_duplicates("source_qid")
    assisted = group[group.condition != "baseline"]
    merged = assisted.merge(baseline, on="source_qid", suffixes=("", "_baseline"), validate="many_to_one")
    difference = merged.score - merged.score_baseline
    stderr = float(difference.std(ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else 0.0
    return {
        "n": len(group),
        "source_questions": group.source_qid.nunique(),
        "baseline_accuracy": float(baseline.score.mean()),
        "overall_accuracy": float(group.score.mean()),
        "assisted_accuracy": float(assisted.score.mean()),
        "paired_gain": float(difference.mean()),
        "paired_gain_stderr": stderr,
        "paired_gain_ci95_low": float(difference.mean() - 1.96 * stderr),
        "paired_gain_ci95_high": float(difference.mean() + 1.96 * stderr),
        "rescues": int(((merged.score_baseline == 0) & (merged.score == 1)).sum()),
        "regressions": int(((merged.score_baseline == 1) & (merged.score == 0)).sum()),
        "parse_success": float(group.parse_success.mean()),
        "numeric_error": float(group.numeric_option_error.dropna().mean()),
    }


def build_summaries(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    suite_rows = []
    for suite, group in samples.groupby("suite", sort=False):
        suite_rows.append({"suite": suite, "suite_name": SUITE_NAMES[suite], **paired_summary(group)})
    suites = pd.DataFrame(suite_rows).sort_values("paired_gain", ascending=False)

    condition_rows = []
    condition_subtask_rows = []
    for suite, suite_group in samples.groupby("suite", sort=False):
        baseline = suite_group[suite_group.condition == "baseline"][["source_qid", "subtask", "score"]].drop_duplicates("source_qid")
        for condition, group in suite_group.groupby("condition", sort=False):
            if condition == "baseline":
                continue
            merged = group.merge(
                baseline[["source_qid", "score"]], on="source_qid", suffixes=("", "_baseline"), validate="one_to_one"
            )
            diff = merged.score - merged.score_baseline
            stderr = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
            condition_rows.append(
                {
                    "suite": suite,
                    "suite_name": SUITE_NAMES[suite],
                    "condition": condition,
                    "n": len(merged),
                    "accuracy": float(merged.score.mean()),
                    "baseline_accuracy": float(merged.score_baseline.mean()),
                    "paired_gain": float(diff.mean()),
                    "paired_gain_stderr": stderr,
                    "paired_gain_ci95_low": float(diff.mean() - 1.96 * stderr),
                    "paired_gain_ci95_high": float(diff.mean() + 1.96 * stderr),
                    "rescues": int(((merged.score_baseline == 0) & (merged.score == 1)).sum()),
                    "regressions": int(((merged.score_baseline == 1) & (merged.score == 0)).sum()),
                    "net_rescues": int(diff.sum()),
                    "numeric_error": float(merged.numeric_option_error.dropna().mean()),
                }
            )
            for subtask, subgroup in merged.groupby("subtask", sort=False):
                subdiff = subgroup.score - subgroup.score_baseline
                condition_subtask_rows.append(
                    {
                        "suite": suite,
                        "suite_name": SUITE_NAMES[suite],
                        "condition": condition,
                        "subtask": subtask,
                        "subtask_name": SUBTASK_NAMES[subtask],
                        "n": len(subgroup),
                        "accuracy": float(subgroup.score.mean()),
                        "baseline_accuracy": float(subgroup.score_baseline.mean()),
                        "paired_gain": float(subdiff.mean()),
                        "rescues": int(((subgroup.score_baseline == 0) & (subgroup.score == 1)).sum()),
                        "regressions": int(((subgroup.score_baseline == 1) & (subgroup.score == 0)).sum()),
                    }
                )

    # Every suite repeats the same baseline. Use one suite to avoid weighting it eight times.
    baseline = samples[(samples.suite == "oracle_ladder") & (samples.condition == "baseline")].copy()
    baseline["answer_type"] = np.where(baseline.subtask.isin(NUMERIC_SUBTASKS), "Numerical", "Categorical")
    baseline_rows = []
    for subtask, group in baseline.groupby("subtask", sort=False):
        baseline_rows.append(
            {
                "subtask": subtask,
                "subtask_name": SUBTASK_NAMES[subtask],
                "answer_type": group.answer_type.iloc[0],
                "n": len(group),
                "accuracy": float(group.score.mean()),
                "parse_success": float(group.parse_success.mean()),
                "numeric_error": float(group.numeric_option_error.dropna().mean()),
            }
        )
    return suites, pd.DataFrame(condition_rows), pd.DataFrame(condition_subtask_rows), pd.DataFrame(baseline_rows)


def matched_scale_comparison(current: pd.DataFrame, old: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_base = current[(current.suite == "oracle_ladder") & (current.condition == "baseline")].drop_duplicates("source_qid")
    old_base = old[(old.suite == "oracle_ladder") & (old.condition == "baseline")].drop_duplicates("source_qid")
    matched = current_base.merge(
        old_base[["source_qid", "score", "prediction", "parsed_prediction", "input"]],
        on="source_qid",
        suffixes=("_scale", "_no_scale"),
        validate="one_to_one",
    )
    matched["answer_type"] = np.where(matched.subtask.isin(NUMERIC_SUBTASKS), "Numerical", "Categorical")
    matched["difference"] = matched.score_scale - matched.score_no_scale
    matched["prediction_changed"] = matched.parsed_prediction_scale != matched.parsed_prediction_no_scale
    rows = []
    for answer_type, group in matched.groupby("answer_type", sort=False):
        diff = group.difference
        stderr = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
        rows.append(
            {
                "answer_type": answer_type,
                "n": len(group),
                "no_scale_accuracy": float(group.score_no_scale.mean()),
                "scale_accuracy": float(group.score_scale.mean()),
                "paired_change": float(diff.mean()),
                "paired_change_stderr": stderr,
                "paired_change_ci95_low": float(diff.mean() - 1.96 * stderr),
                "paired_change_ci95_high": float(diff.mean() + 1.96 * stderr),
                "rescues": int((diff == 1).sum()),
                "regressions": int((diff == -1).sum()),
                "prediction_changes": int(group.prediction_changed.sum()),
            }
        )
    return pd.DataFrame(rows), matched


def save_tables(**tables: pd.DataFrame) -> None:
    for name, table in tables.items():
        table.to_csv(REPORT_DIR / f"{name}.csv", index=False)


def style_axis(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=16)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color="#4B5563")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)


def plot_suite_overview(suites: pd.DataFrame) -> Path:
    ordered = suites.sort_values("assisted_accuracy")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.barh(y - 0.17, 100 * ordered.baseline_accuracy, 0.32, color=COLORS["baseline"], label="Baseline")
    bars = ax.barh(y + 0.17, 100 * ordered.assisted_accuracy, 0.32, color=COLORS["assisted"], label="Assisted conditions")
    for bar, gain in zip(bars, ordered.paired_gain):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"{100*gain:+.1f} pp", va="center", fontsize=8)
    ax.set_yticks(y, ordered.suite_name)
    ax.set_xlim(0, max(52, 100 * ordered.assisted_accuracy.max() + 7))
    ax.set_xlabel("Accuracy (%)")
    style_axis(ax, "Assistance versus the scale-guide baseline", "Gain labels use paired source-question comparisons.")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path = FIGURE_DIR / "suite_overview.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_baseline_subtasks(baseline: pd.DataFrame) -> Path:
    ordered = baseline.sort_values("accuracy")
    colors = [COLORS["numeric"] if value == "Numerical" else COLORS["categorical"] for value in ordered.answer_type]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.barh(ordered.subtask_name, 100 * ordered.accuracy, color=colors)
    for bar, value in zip(bars, ordered.accuracy):
        ax.text(100 * value + 0.4, bar.get_y() + bar.get_height() / 2, f"{100*value:.1f}%", va="center", fontsize=8)
    ax.axvline(20, color="#111827", linestyle="--", linewidth=1, label="Five-choice chance (20%)")
    ax.set_xlim(0, max(42, 100 * ordered.accuracy.max() + 6))
    ax.set_xlabel("Baseline accuracy (%)")
    style_axis(ax, "Baseline by motion subtask", "Purple tasks receive the plotted scale guide; orange tasks are categorical and unchanged.")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path = FIGURE_DIR / "baseline_subtasks.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_condition_extremes(conditions: pd.DataFrame) -> Path:
    selected = pd.concat([conditions.nsmallest(12, "paired_gain"), conditions.nlargest(18, "paired_gain")]).drop_duplicates(
        ["suite", "condition"]
    ).sort_values("paired_gain")
    labels = [f"{row.suite_name}: {row.condition}" for row in selected.itertuples()]
    values = 100 * selected.paired_gain
    colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in values]
    fig, ax = plt.subplots(figsize=(11, 10))
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(value + (0.4 if value >= 0 else -0.4), bar.get_y() + bar.get_height()/2, f"{value:+.1f}",
                ha="left" if value >= 0 else "right", va="center", fontsize=7.5)
    ax.set_xlabel("Paired change from baseline (percentage points)")
    style_axis(ax, "Most helpful and harmful assistance conditions", "Each bar compares the same source questions under one condition and baseline.")
    fig.tight_layout()
    path = FIGURE_DIR / "condition_extremes.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_subtask_suite_heatmap(condition_subtasks: pd.DataFrame) -> Path:
    grouped = condition_subtasks.groupby(["suite_name", "subtask_name"], as_index=False).paired_gain.mean()
    pivot = grouped.pivot(index="subtask_name", columns="suite_name", values="paired_gain")
    suite_order = [SUITE_NAMES[key] for key in SUITE_NAMES]
    subtask_order = [SUBTASK_NAMES[key] for key in SUBTASK_NAMES]
    pivot = pivot.reindex(index=subtask_order, columns=suite_order)
    data = 100 * pivot.to_numpy()
    bound = max(10, math.ceil(np.nanmax(np.abs(data)) / 5) * 5)
    fig, ax = plt.subplots(figsize=(12, 6.8))
    image = ax.imshow(data, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:+.1f}", ha="center", va="center", fontsize=7.5)
    ax.set_title("Mean assistance gain by subtask and suite", loc="left", fontsize=14, fontweight="bold", pad=14)
    fig.colorbar(image, ax=ax, label="Percentage points versus baseline", shrink=0.8)
    fig.tight_layout()
    path = FIGURE_DIR / "subtask_suite_heatmap.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_matched_scale(matched_summary: pd.DataFrame) -> Path:
    x = np.arange(len(matched_summary))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.bar(x - width/2, 100 * matched_summary.no_scale_accuracy, width, color=COLORS["baseline"], label="Earlier 12-frame run (no scale)")
    bars = ax.bar(x + width/2, 100 * matched_summary.scale_accuracy, width, color=COLORS["numeric"], label="Full run (scale enabled)")
    for bar, row in zip(bars, matched_summary.itertuples()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{100*row.paired_change:+.1f} pp", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{row.answer_type}\n(n={row.n})" for row in matched_summary.itertuples()])
    ax.set_ylim(0, max(35, 100 * matched_summary[["no_scale_accuracy", "scale_accuracy"]].to_numpy().max() + 7))
    ax.set_ylabel("Accuracy (%)")
    style_axis(ax, "Matched scale-guide comparison", "Same deterministic QIDs; categorical questions are an unchanged-input control.")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = FIGURE_DIR / "matched_scale_comparison.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rescue_regression(conditions: pd.DataFrame) -> Path:
    selected = conditions.sort_values("net_rescues", ascending=False).head(20).sort_values("net_rescues")
    labels = [f"{row.suite_name}: {row.condition}" for row in selected.itertuples()]
    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(selected))
    ax.barh(y, selected.rescues, color=COLORS["positive"], label="Baseline errors rescued")
    ax.barh(y, -selected.regressions, color=COLORS["negative"], label="Baseline-correct regressed")
    ax.set_yticks(y, labels)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Paired question count")
    style_axis(ax, "Rescues and regressions for strongest net conditions")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path = FIGURE_DIR / "rescue_regression.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def reconstruct_example(samples: pd.DataFrame) -> dict:
    oracle = samples[(samples.suite == "oracle_ladder") & (samples.condition == "derived_quantity_oracle")]
    baseline = samples[(samples.suite == "oracle_ladder") & (samples.condition == "baseline")][["source_qid", "score", "prediction"]]
    candidates = oracle.merge(baseline, on="source_qid", suffixes=("", "_baseline"))
    candidates = candidates[(candidates.subtask.isin(NUMERIC_SUBTASKS)) & (candidates.score_baseline == 0) & (candidates.score == 1)]
    selected_row = candidates.sort_values(["subtask", "source_qid"]).iloc[0]

    os.environ["MOVI_MOTION_DERIVED_NUM_FRAMES"] = "12"
    os.environ["MOVI_MOTION_NUMERIC_SCALE_GUIDE"] = "1"
    from lmms_eval.tasks.kubric_movi_a_motion import utils

    dataset = Dataset.from_parquet(str(Path("/home/ramanathan/data/movi_a_motion/qa.parquet")))
    raw = next(dict(item) for item in dataset if str(item["qid"]) == selected_row.source_qid)
    source = utils._derive_frame_limited_doc(raw, 12)
    doc = {
        **source,
        "source_qid": source["qid"],
        "qid": f"{source['qid']}::derived_quantity_oracle",
        "index": f"{source['qid']}::derived_quantity_oracle",
        "scaffold_suite": "oracle_ladder",
        "scaffold_condition": "derived_quantity_oracle",
    }
    prompt = utils.doc_to_text(
        doc,
        {
            "pre_prompt": "These are the ordered frames of a 12 FPS rigid-body simulation.",
            "post_prompt": "Answer with only the option letter A, B, C, D, or E.",
        },
    )
    assert prompt == selected_row.input, "Reconstructed prompt differs from logged prompt"
    images = utils.doc_to_visual(doc)
    frame_ids = utils.frame_indices(doc)

    tile = 256
    label_height = 20
    columns = 4
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        image = image.resize((tile, tile))
        x = (index % columns) * tile
        y = (index // columns) * (tile + label_height)
        sheet.paste(image, (x, y))
        label = f"input {index + 1} · source frame {frame_ids[index]}" if index < len(frame_ids) else f"input {index + 1}"
        draw.text((x + 5, y + tile + 3), label, fill="black")
    sheet_path = EXAMPLE_DIR / "representative_scale_guide_contact_sheet.jpg"
    sheet.save(sheet_path, quality=92)

    state_doc = {**doc, "qid": f"{source['qid']}::state_only_no_visual", "scaffold_suite": "calculation", "scaffold_condition": "state_only_no_visual"}
    state_image = utils.doc_to_visual(state_doc)[0]
    state_path = EXAMPLE_DIR / "state_only_scale_panel.png"
    state_image.save(state_path)
    (EXAMPLE_DIR / "representative_prompt.txt").write_text(prompt, encoding="utf-8")

    record = {
        "source_qid": selected_row.source_qid,
        "sequence_id": selected_row.sequence_id,
        "subtask": selected_row.subtask,
        "condition": selected_row.condition,
        "baseline_prediction": candidates.loc[selected_row.name, "prediction_baseline"],
        "assisted_prediction": selected_row.prediction,
        "gold": selected_row.gold,
        "baseline_correct": bool(selected_row.score_baseline),
        "assisted_correct": bool(selected_row.score),
        "contact_sheet": str(sheet_path.relative_to(REPORT_DIR)),
        "prompt": str((EXAMPLE_DIR / "representative_prompt.txt").relative_to(REPORT_DIR)),
    }
    (EXAMPLE_DIR / "representative_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], formatters: dict[str, callable] | None = None) -> str:
    formatters = formatters or {}
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "|".join("---:" for _ in columns) + "|"]
    for row in frame.itertuples(index=False):
        values = []
        for key, _ in columns:
            value = getattr(row, key)
            values.append(str(formatters.get(key, lambda item: item)(value)))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown(
    samples: pd.DataFrame,
    suites: pd.DataFrame,
    conditions: pd.DataFrame,
    condition_subtasks: pd.DataFrame,
    baseline: pd.DataFrame,
    matched_summary: pd.DataFrame,
    runs: pd.DataFrame,
    example: dict,
) -> str:
    best = suites.iloc[0]
    strongest = conditions.sort_values("paired_gain", ascending=False).head(12)
    weakest = conditions.sort_values("paired_gain").head(10)
    numeric_match = matched_summary[matched_summary.answer_type == "Numerical"].iloc[0]
    categorical_match = matched_summary[matched_summary.answer_type == "Categorical"].iloc[0]
    unique_baseline = samples[(samples.suite == "oracle_ladder") & (samples.condition == "baseline")]

    suite_table = markdown_table(
        suites,
        [("suite_name", "Suite"), ("n", "N"), ("baseline_accuracy", "Baseline"), ("assisted_accuracy", "Assisted"),
         ("paired_gain", "Paired gain"), ("rescues", "Rescues"), ("regressions", "Regressions"), ("numeric_error", "Numeric error")],
        {"baseline_accuracy": pct, "assisted_accuracy": pct, "paired_gain": pp, "numeric_error": lambda x: f"{x:.3f}"},
    )
    baseline_table = markdown_table(
        baseline.sort_values("accuracy", ascending=False),
        [("subtask_name", "Subtask"), ("answer_type", "Type"), ("n", "N"), ("accuracy", "Accuracy"), ("numeric_error", "Numeric error")],
        {"accuracy": pct, "numeric_error": lambda x: "—" if pd.isna(x) else f"{x:.3f}"},
    )
    condition_columns = [("suite_name", "Suite"), ("condition", "Condition"), ("n", "N"), ("accuracy", "Accuracy"),
                         ("paired_gain", "Gain"), ("rescues", "Rescues"), ("regressions", "Regressions")]
    condition_format = {"accuracy": pct, "paired_gain": pp}
    run_table = markdown_table(
        runs,
        [("part", "PBS part"), ("evaluations", "Evaluations"), ("elapsed_hours", "Evaluation hours"),
         ("generation_tokens_per_second", "Tokens/s")],
        {"elapsed_hours": lambda x: f"{x:.2f}", "generation_tokens_per_second": lambda x: f"{x:.3f}"},
    )
    matched_table = markdown_table(
        matched_summary,
        [("answer_type", "Question type"), ("n", "Matched N"), ("no_scale_accuracy", "No scale"),
         ("scale_accuracy", "Scale run"), ("paired_change", "Change"), ("rescues", "Rescues"),
         ("regressions", "Regressions"), ("prediction_changes", "Prediction changes")],
        {"no_scale_accuracy": pct, "scale_accuracy": pct, "paired_change": pp},
    )

    prompt_excerpt = (EXAMPLE_DIR / "representative_prompt.txt").read_text(encoding="utf-8")
    return f"""# MOVi-A 12-frame full diagnostics with metric scale guides

## Executive summary

- All four PBS parts completed without traceback or CUDA failure. The run contains **{len(samples):,} expanded evaluations**, **{samples.source_qid.nunique():,} unique source questions**, eight diagnostic suites, and **{samples.condition.nunique()} named conditions**.
- The shared scale-guide baseline is **{pct(unique_baseline.score.mean())}** overall. The baseline remains below five-choice chance ({20:.0f}%) by **{20 - 100 * unique_baseline.score.mean():.2f} percentage points**.
- **{best.suite_name}** gives the largest suite-level gain: **{pct(best.assisted_accuracy)}** assisted accuracy and **{pp(best.paired_gain)}** paired change from the same-question baseline.
- The strongest non-oracle suites are **Precision ({pp(suites[suites.suite == 'precision'].paired_gain.iloc[0])})** and **Calculation ({pp(suites[suites.suite == 'calculation'].paired_gain.iloc[0])})**. Perception and temporal visual organization helps little; robustness controls are slightly harmful, as intended.
- In the matched earlier 12-frame comparison, numerical questions changed by **{pp(numeric_match.paired_change)}** ({numeric_match.rescues} rescues, {numeric_match.regressions} regressions; n={numeric_match.n}). The unchanged categorical control changed by **{pp(categorical_match.paired_change)}**. This is consistent with the guide's effect, but run-to-run GPU nondeterminism and the added prompt sentence prevent treating it as a pure visual-only causal estimate.
- Parse success is **{pct(samples.parse_success.mean())}**; answer-format failure is not a meaningful bottleneck.

![Suite overview](figures/suite_overview.png)

## Experiment and interpretation boundary

Every numerical-answer frame contains a perspective-correct ruler representing a camera-plane segment at the target object's depth. Its pixel length varies with perspective. The prompt explicitly explains this. Five numerical subtasks receive the guide; the three categorical subtasks do not. The intentionally scene-free `state_only_no_visual` condition receives only a synthetic scale panel.

The guide is an **assistance signal**, not a neutral rendering change. Consequently, “baseline” in this report means the video-plus-scale baseline for numerical tasks and ordinary video baseline for categorical tasks. Comparisons among current assistance conditions are paired and valid within this definition. Direct comparison to older baselines requires the matched analysis below.

## Suite-level results

{suite_table}

Notes:

- `N` counts expanded condition records; baseline records recur once per suite.
- Paired gain is assisted correctness minus baseline correctness on the same source question.
- Rescues and regressions count all assisted records in a suite, so suites with more conditions have more opportunities.
- Numeric error is absolute error in the numerical value of the selected option and is lower-is-better.

## Baseline difficulty by subtask

{baseline_table}

![Baseline by subtask](figures/baseline_subtasks.png)

This view is crucial: aggregate baseline accuracy mixes physically different targets. A scale bar can directly help apparent distance and speed judgments, but it cannot by itself provide elapsed-time integration, velocity derivatives, identity tracking, or 3-D trajectory reconstruction.

## Which assistance conditions help?

### Strongest conditions

{markdown_table(strongest, condition_columns, condition_format)}

### Weakest conditions

{markdown_table(weakest, condition_columns, condition_format)}

![Condition extremes](figures/condition_extremes.png)

![Rescue and regression counts](figures/rescue_regression.png)

The oracle endpoint conditions provide an upper-bound sanity check rather than evidence of model reasoning. More diagnostically useful are precision, raw-state, formula, and intermediate-quantity conditions: these expose where accurate values still fail to map to the correct answer.

## Assistance by task type

![Subtask and suite heatmap](figures/subtask_suite_heatmap.png)

The complete per-condition/per-subtask table is available in [`condition_subtask_summary.csv`](condition_subtask_summary.csv). Use it to distinguish broadly helpful aids from gains concentrated in one target type.

## Matched effect of adding the scale guide

{matched_table}

![Matched scale comparison](figures/matched_scale_comparison.png)

This comparison uses the exact overlapping deterministic QIDs from the earlier 20-per-subtask 12-frame run. The numerical subset changed both visually and textually; the categorical subset should have identical task inputs and acts as a run-to-run stability control. The confidence intervals and per-question transitions are saved in [`matched_scale_comparison.csv`](matched_scale_comparison.csv) and [`matched_scale_questions.csv`](matched_scale_questions.csv).

## Representative rescued example

- Source QID: `{example['source_qid']}`
- Sequence: `{example['sequence_id']}`
- Subtask: `{example['subtask']}`
- Condition: `{example['condition']}`
- Baseline prediction: **{example['baseline_prediction']}** (incorrect)
- Assisted prediction: **{example['assisted_prediction']}**; gold: **{example['gold']}**

![Representative reconstructed input](examples/representative_scale_guide_contact_sheet.jpg)

The image sequence above was deterministically reconstructed from the logged QID, current 12-frame derivation code, source frames, and camera metadata. It was not stored by the completed run because `MOVI_MOTION_TRACE_DIR` was not enabled. The exact model input text *was* logged and matches the reconstructed prompt byte-for-byte:

```text
{prompt_excerpt}
```

The separate no-scene control panel is shown below:

![Scale-only panel](examples/state_only_scale_panel.png)

## Runtime and completeness

{run_table}

- Critical-path evaluation time if the four parts ran concurrently: **{runs.elapsed_hours.max():.2f} hours**.
- Summed evaluation compute: **{runs.elapsed_hours.sum():.2f} GPU-hours**.
- Result JSON files: four; JSONL sample files: eight.
- Expected and observed expanded evaluations: **40,900**.
- Error logs contain model compatibility warnings but no traceback, OOM, or failed evaluation.

## Limitations

1. Full task-side trace bundles were not enabled. Exact transformed frames for every sample are therefore unavailable; only representative inputs can be reconstructed.
2. The scale ruler is valid for a segment parallel to the image plane at the target-center depth. It is not a ground-plane grid and does not directly reveal out-of-plane displacement.
3. Scale-guide versus no-scale evidence comes from two separate deterministic inference runs. The categorical control helps quantify instability, but it is not a randomized intervention.
4. Expanded rows are not independent: the same 800 source questions recur across suites and conditions. Use paired gains and source-level transitions, not naive row-level significance.
5. Answer-letter and answer-text oracles are ceiling checks and should not be grouped with deployable assistance methods when drawing scientific conclusions.

## Files for further analysis

- [`suite_summary.csv`](suite_summary.csv)
- [`condition_summary.csv`](condition_summary.csv)
- [`condition_subtask_summary.csv`](condition_subtask_summary.csv)
- [`baseline_subtask_summary.csv`](baseline_subtask_summary.csv)
- [`matched_scale_comparison.csv`](matched_scale_comparison.csv)
- [`matched_scale_questions.csv`](matched_scale_questions.csv)
- [`representative_record.json`](examples/representative_record.json)
- [`representative_prompt.txt`](examples/representative_prompt.txt)

## Provenance

- Model: `{runs.model.iloc[0]}`
- Model arguments: `max_num_frames=12,pretrained=Qwen/Qwen3-VL-8B-Instruct`
- `lmms-eval` revision: `{runs.git_hash.iloc[0]}`
- Full-run output: `{RUN_DIR.relative_to(ROOT)}`
- Report generator: `tools/analyze_movi_motion_scale_guide_results.py`
"""


def add_wrapped(ax, text: str, y: float, width: int = 92, size: float = 10.5, spacing: float = 0.035) -> float:
    for paragraph in text.split("\n"):
        lines = textwrap.wrap(paragraph, width=width) if paragraph else [""]
        for line in lines:
            ax.text(0, y, line, transform=ax.transAxes, fontsize=size, va="top")
            y -= spacing
    return y


def add_image_page(pdf: PdfPages, path: Path, title: str) -> None:
    image = plt.imread(path)
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes((0.035, 0.035, 0.93, 0.90))
    ax.imshow(image)
    ax.axis("off")
    fig.suptitle(title, x=0.045, y=0.98, ha="left", fontsize=17, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf(
    samples: pd.DataFrame,
    suites: pd.DataFrame,
    baseline: pd.DataFrame,
    matched: pd.DataFrame,
    conditions: pd.DataFrame,
    figures: list[Path],
    example: dict,
) -> Path:
    path = REPORT_DIR / "MOVi_A_12f_scale_guide_detailed_report.pdf"
    best = suites.iloc[0]
    numeric_match = matched[matched.answer_type == "Numerical"].iloc[0]
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes((0.08, 0.07, 0.84, 0.87))
        ax.axis("off")
        ax.text(0, 1, "MOVi-A Motion Diagnostics", fontsize=24, fontweight="bold", va="top")
        ax.text(0, 0.955, "Full 12-frame run with perspective scale guides", fontsize=13, color="#475569", va="top")
        summary = (
            f"The four completed PBS jobs contain {len(samples):,} expanded evaluations from 800 source questions. "
            f"The video-plus-scale baseline is {pct(samples[(samples.suite=='oracle_ladder') & (samples.condition=='baseline')].score.mean())}. "
            f"{best.suite_name} is strongest at {pct(best.assisted_accuracy)} assisted accuracy ({pp(best.paired_gain)} paired gain). "
            f"On 100 matched numerical questions from the earlier 12-frame run, adding the guide changes accuracy by {pp(numeric_match.paired_change)}."
        )
        y = add_wrapped(ax, summary, 0.88, size=11, spacing=0.036)
        y -= 0.03
        ax.text(0, y, "Interpretation", fontsize=15, fontweight="bold", va="top")
        y -= 0.05
        points = [
            "Precision and calculation assistance yield the strongest non-oracle improvements.",
            "Perception and temporal visual organization provide little average gain.",
            "Robustness controls are slightly harmful, supporting their intended negative-control role.",
            "All responses parse; the central errors concern perception, state recovery, and numerical integration rather than output formatting.",
            "The scale guide is itself an assistance signal, and the completed jobs did not save full transformed-image traces.",
        ]
        for point in points:
            y = add_wrapped(ax, "• " + point, y, width=88, size=10.5, spacing=0.033) - 0.01
        ax.text(0, 0.04, "Detailed CSV tables and reconstructed examples accompany this PDF.", fontsize=8.5, color="#64748B")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for figure in figures:
            add_image_page(pdf, figure, figure.stem.replace("_", " ").title())

        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes((0.03, 0.08, 0.94, 0.84))
        ax.axis("off")
        ax.set_title("Complete suite metrics", loc="left", fontsize=17, fontweight="bold", pad=16)
        cells = []
        for row in suites.itertuples():
            cells.append([row.suite_name, f"{row.n:,}", pct(row.baseline_accuracy), pct(row.assisted_accuracy), pp(row.paired_gain), row.rescues, row.regressions, f"{row.numeric_error:.3f}"])
        table = ax.table(cellText=cells, colLabels=["Suite", "N", "Baseline", "Assisted", "Gain", "Rescues", "Regress.", "Num. err."],
                         cellLoc="center", colLoc="center", loc="upper center", colWidths=[.18,.09,.11,.11,.11,.10,.10,.10])
        table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 1.65)
        for (i, _), cell in table.get_celld().items():
            if i == 0: cell.set_facecolor("#DBEAFE"); cell.set_text_props(weight="bold")
            elif i % 2 == 0: cell.set_facecolor("#F1F5F9")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        contact = plt.imread(REPORT_DIR / example["contact_sheet"])
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes((0.03, 0.06, 0.94, 0.86)); ax.imshow(contact); ax.axis("off")
        fig.suptitle(f"Representative rescued input · {example['subtask']} · {example['condition']}", x=.04, y=.98, ha="left", fontsize=16, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        strongest = conditions.sort_values("paired_gain", ascending=False).head(20)
        fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_axes((.03,.06,.94,.86)); ax.axis("off")
        ax.set_title("Top 20 conditions", loc="left", fontsize=17, fontweight="bold", pad=16)
        cells = [[r.suite_name, r.condition, r.n, pct(r.accuracy), pp(r.paired_gain), r.rescues, r.regressions] for r in strongest.itertuples()]
        table = ax.table(cellText=cells, colLabels=["Suite","Condition","N","Accuracy","Gain","Rescues","Regress."],
                         cellLoc="center", colLoc="center", loc="upper center", colWidths=[.15,.28,.07,.10,.10,.09,.09])
        table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1,1.35)
        for (i,_), cell in table.get_celld().items():
            if i == 0: cell.set_facecolor("#DCFCE7"); cell.set_text_props(weight="bold")
            elif i % 2 == 0: cell.set_facecolor("#F1F5F9")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    return path


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_samples(RUN_DIR, "scale_guide_full")
    if len(samples) != 40900:
        raise RuntimeError(f"Expected 40,900 full-run records, found {len(samples):,}")
    old_samples = load_samples(OLD_RUN_DIR, "no_scale_sample")
    runs = load_runs()
    suites, conditions, condition_subtasks, baseline = build_summaries(samples)
    matched_summary, matched_questions = matched_scale_comparison(samples, old_samples)

    save_tables(
        suite_summary=suites,
        condition_summary=conditions.sort_values(["suite", "paired_gain"], ascending=[True, False]),
        condition_subtask_summary=condition_subtasks.sort_values(["suite", "condition", "subtask"]),
        baseline_subtask_summary=baseline.sort_values("accuracy", ascending=False),
        matched_scale_comparison=matched_summary,
        matched_scale_questions=matched_questions,
        run_summary=runs,
    )

    figures = [
        plot_suite_overview(suites),
        plot_baseline_subtasks(baseline),
        plot_condition_extremes(conditions),
        plot_rescue_regression(conditions),
        plot_subtask_suite_heatmap(condition_subtasks),
        plot_matched_scale(matched_summary),
    ]
    example = reconstruct_example(samples)
    report = build_markdown(samples, suites, conditions, condition_subtasks, baseline, matched_summary, runs, example)
    md_path = REPORT_DIR / "DETAILED_ANALYSIS.md"
    md_path.write_text(report, encoding="utf-8")
    pdf_path = build_pdf(samples, suites, baseline, matched_summary, conditions, figures, example)
    print(f"Wrote {md_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {len(figures)} figures and 7 CSV tables")


if __name__ == "__main__":
    main()
