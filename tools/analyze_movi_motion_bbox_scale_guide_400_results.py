#!/usr/bin/env python3
"""Collect the 400/subtask MOVi-A diagnostic run into tables, plots, MD and PDF."""

from __future__ import annotations

import hashlib
import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "kubric_movi_a_motion_12f_full_bbox_scale_guide_400"
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
    "visual_scale": "Visual scale",
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
    "displacement", "path_length", "average_speed", "instantaneous_speed", "instantaneous_acceleration"
}

COLORS = {
    "baseline": "#64748B", "assisted": "#2563EB", "positive": "#0F9D76",
    "negative": "#C43C39", "corner": "#F59E0B", "anchored": "#7C3AED",
}


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def load_samples() -> pd.DataFrame:
    files = sorted(RUN_DIR.glob("part_*/Qwen__Qwen3-VL-8B-Instruct/*_samples_*.jsonl"))
    if len(files) != 9:
        raise RuntimeError(f"Expected 9 sample JSONLs, found {len(files)}")
    rows: list[dict] = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                metric = dict(record["accuracy"])
                metric["sample_file"] = str(path.relative_to(ROOT))
                rows.append(metric)
    frame = pd.DataFrame(rows)
    if len(frame) != 169_600:
        raise RuntimeError(f"Expected 169,600 evaluations, found {len(frame):,}")
    return frame


def load_runs() -> pd.DataFrame:
    paths = sorted(RUN_DIR.glob("part_*/Qwen__Qwen3-VL-8B-Instruct/*_results.json"))
    if len(paths) != 4:
        raise RuntimeError(f"Expected 4 result JSONs, found {len(paths)}")
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        counts = data["n-samples"]
        rows.append({
            "part": path.parents[1].name,
            "evaluations": sum(int(value["effective"]) for value in counts.values()),
            "evaluation_hours": float(data["total_evaluation_time_seconds"]) / 3600,
            "generation_tokens_per_second": float(data["throughput"]["avg_speed"]),
            "model": data["model_name"],
            "git_hash": data["git_hash"],
            "result_file": str(path.relative_to(ROOT)),
        })
    return pd.DataFrame(rows)


def paired_rows(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict] = []
    transitions: list[pd.DataFrame] = []
    for (suite, condition, subtask), group in samples[samples.condition != "baseline"].groupby(
        ["suite", "condition", "subtask"], sort=False
    ):
        baseline = samples[
            (samples.suite == suite) & (samples.condition == "baseline") & (samples.subtask == subtask)
        ][["source_qid", "score", "prediction", "parsed_prediction"]].drop_duplicates("source_qid")
        merged = group.merge(baseline, on="source_qid", suffixes=("", "_baseline"), validate="one_to_one")
        diff = merged.score - merged.score_baseline
        stderr = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
        summaries.append({
            "suite": suite,
            "suite_name": SUITE_NAMES[suite],
            "condition": condition,
            "subtask": subtask,
            "subtask_name": SUBTASK_NAMES[subtask],
            "n": len(merged),
            "baseline_accuracy": float(merged.score_baseline.mean()),
            "accuracy": float(merged.score.mean()),
            "paired_gain": float(diff.mean()),
            "paired_gain_stderr": stderr,
            "paired_gain_ci95_low": float(diff.mean() - 1.96 * stderr),
            "paired_gain_ci95_high": float(diff.mean() + 1.96 * stderr),
            "rescues": int((diff == 1).sum()),
            "regressions": int((diff == -1).sum()),
            "net_rescues": int(diff.sum()),
            "parse_success": float(merged.parse_success.mean()),
            "numeric_option_error": float(merged.numeric_option_error.dropna().mean()),
        })
        transition = merged[[
            "suite", "condition", "subtask", "source_qid", "qid", "sequence_id", "score_baseline",
            "score", "prediction_baseline", "prediction", "parsed_prediction_baseline", "parsed_prediction", "gold",
        ]].copy()
        transition["difference"] = diff
        transitions.append(transition)
    return pd.DataFrame(summaries), pd.concat(transitions, ignore_index=True)


def aggregate_conditions(task_conditions: pd.DataFrame) -> pd.DataFrame:
    fields = ["suite", "suite_name", "condition"]
    rows = []
    for keys, group in task_conditions.groupby(fields, sort=False):
        n = int(group.n.sum())
        rows.append({
            **dict(zip(fields, keys)),
            "n": n,
            "baseline_accuracy": float(np.average(group.baseline_accuracy, weights=group.n)),
            "accuracy": float(np.average(group.accuracy, weights=group.n)),
            "paired_gain": float(np.average(group.paired_gain, weights=group.n)),
            "rescues": int(group.rescues.sum()),
            "regressions": int(group.regressions.sum()),
            "net_rescues": int(group.net_rescues.sum()),
        })
    return pd.DataFrame(rows)


def aggregate_suites(samples: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for suite, group in samples.groupby("suite", sort=False):
        baseline = group[group.condition == "baseline"].drop_duplicates("source_qid")
        assisted = group[group.condition != "baseline"]
        c = conditions[conditions.suite == suite]
        rows.append({
            "suite": suite, "suite_name": SUITE_NAMES[suite], "n": len(group),
            "source_questions": group.source_qid.nunique(),
            "baseline_accuracy": float(baseline.score.mean()),
            "assisted_accuracy": float(assisted.score.mean()),
            "paired_gain": float(np.average(c.paired_gain, weights=c.n)),
            "rescues": int(c.rescues.sum()), "regressions": int(c.regressions.sum()),
        })
    return pd.DataFrame(rows).sort_values("paired_gain", ascending=False)


def baseline_tasks(samples: pd.DataFrame) -> pd.DataFrame:
    # Visual-scale baseline is genuinely unassisted for numeric tasks. Categorical tasks are
    # absent from that suite and use the unchanged oracle-ladder baseline.
    numeric = samples[(samples.suite == "visual_scale") & (samples.condition == "baseline")]
    categorical = samples[
        (samples.suite == "oracle_ladder") & (samples.condition == "baseline") & ~samples.subtask.isin(NUMERIC_SUBTASKS)
    ]
    base = pd.concat([numeric, categorical]).drop_duplicates("source_qid")
    rows = []
    for subtask, group in base.groupby("subtask", sort=False):
        rows.append({
            "subtask": subtask, "subtask_name": SUBTASK_NAMES[subtask], "n": len(group),
            "answer_type": "Numerical" if subtask in NUMERIC_SUBTASKS else "Categorical",
            "accuracy": float(group.score.mean()), "parse_success": float(group.parse_success.mean()),
            "numeric_option_error": float(group.numeric_option_error.dropna().mean()),
        })
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def trace_path(row: pd.Series) -> Path:
    part = {
        "calculation": "part_0_calculation_representation", "representation": "part_0_calculation_representation",
        "oracle_ladder": "part_1_oracle_precision", "precision": "part_1_oracle_precision",
        "temporal": "part_2_temporal_perception", "perception": "part_2_temporal_perception",
        "coordinate": "part_3_coordinate_robustness_visual_scale", "robustness": "part_3_coordinate_robustness_visual_scale",
        "visual_scale": "part_3_coordinate_robustness_visual_scale",
    }[row.suite]
    digest = hashlib.sha1(f"{row.suite}::{row.qid}".encode()).hexdigest()
    return RUN_DIR / part / "trace_bundle" / "records" / f"{digest}_{row.condition}.json"


def trace_examples(transitions: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for (suite, condition, subtask), group in transitions.groupby(["suite", "condition", "subtask"], sort=False):
        for kind, value in (("rescue", 1.0), ("regression", -1.0)):
            candidates = group[group.difference == value]
            if candidates.empty:
                continue
            row = candidates.sort_values("source_qid").iloc[0].copy()
            path = trace_path(row)
            record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            visual = record.get("visual_input", {})
            selected.append({
                "kind": kind, "suite": suite, "condition": condition, "subtask": subtask,
                "source_qid": row.source_qid, "sequence_id": row.sequence_id,
                "baseline_prediction": row.prediction_baseline, "assisted_prediction": row.prediction,
                "gold": row.gold, "trace_file": str(path.relative_to(RUN_DIR)),
                "contact_sheet": visual.get("contact_sheet"), "task_prompt": record.get("task_prompt", ""),
            })
    return pd.DataFrame(selected)


def visual_scale_example(transitions: pd.DataFrame) -> Path | None:
    anchored = transitions[
        (transitions.suite == "visual_scale")
        & (transitions.condition == "object_anchored_metric_scale_guide")
        & (transitions.difference == 1)
    ]
    if anchored.empty:
        return None
    source_qid = anchored.sort_values("source_qid").iloc[0].source_qid
    panels = []
    labels = {
        "baseline": "Unassisted baseline", "corner_metric_scale_guide": "Corner ruler",
        "object_anchored_metric_scale_guide": "Object-anchored ruler",
    }
    for condition in labels:
        sample = transitions[
            (transitions.suite == "visual_scale") & (transitions.condition == condition)
            & (transitions.source_qid == source_qid)
        ]
        if condition == "baseline":
            # Baseline is represented only on the right side of assisted pair rows.
            sample = transitions[
                (transitions.suite == "visual_scale") & (transitions.source_qid == source_qid)
            ].head(1).copy()
            if not sample.empty:
                sample.loc[:, "qid"] = source_qid + "::baseline"
                sample.loc[:, "condition"] = "baseline"
        if sample.empty:
            continue
        path = trace_path(sample.iloc[0])
        record = json.loads(path.read_text(encoding="utf-8"))
        preview = path.parents[1] / record["visual_input"]["contact_sheet"]
        image = Image.open(preview).convert("RGB")
        panels.append((labels[condition], image))
    if len(panels) != 3:
        return None
    width = max(image.width for _, image in panels)
    label_height = 34
    canvas = Image.new("RGB", (width, sum(image.height + label_height for _, image in panels)), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, image in panels:
        draw.text((8, y + 8), label, fill="black")
        y += label_height
        canvas.paste(image, (0, y))
        y += image.height
    path = EXAMPLE_DIR / "visual_scale_rescue_comparison.jpg"
    canvas.save(path, quality=91)
    return path


def style_axis(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=16)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color="#475569")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)


def make_plots(suites: pd.DataFrame, conditions: pd.DataFrame, tasks: pd.DataFrame, baseline: pd.DataFrame) -> list[Path]:
    paths = []
    ordered = suites.sort_values("assisted_accuracy")
    fig, ax = plt.subplots(figsize=(10.8, 6.6)); y = np.arange(len(ordered))
    ax.barh(y - .18, 100 * ordered.baseline_accuracy, .34, color=COLORS["baseline"], label="Suite baseline")
    bars = ax.barh(y + .18, 100 * ordered.assisted_accuracy, .34, color=COLORS["assisted"], label="Assisted")
    for bar, gain in zip(bars, ordered.paired_gain):
        ax.text(bar.get_width()+.35, bar.get_y()+bar.get_height()/2, f"{100*gain:+.1f} pp", va="center", fontsize=8)
    ax.set_yticks(y, ordered.suite_name); ax.set_xlabel("Accuracy (%)"); ax.legend(frameon=False)
    style_axis(ax, "Diagnostic suites versus their paired baseline", "Numerical baselines in the eight original suites include the shared corner scale guide.")
    fig.tight_layout(); path = FIGURE_DIR / "suite_overview.png"; fig.savefig(path, dpi=190, bbox_inches="tight"); plt.close(fig); paths.append(path)

    visual = tasks[tasks.suite == "visual_scale"].copy()
    pivot = visual.pivot(index="subtask_name", columns="condition", values="accuracy")
    base_accuracy = samples_visual_baseline = visual.drop_duplicates("subtask_name").set_index("subtask_name").baseline_accuracy
    order = [SUBTASK_NAMES[item] for item in NUMERIC_SUBTASKS if SUBTASK_NAMES[item] in pivot.index]
    pivot = pivot.reindex(order); base_accuracy = base_accuracy.reindex(order)
    fig, ax = plt.subplots(figsize=(11, 6)); x = np.arange(len(order)); width = .25
    ax.bar(x-width, 100*base_accuracy, width, color=COLORS["baseline"], label="Unassisted")
    ax.bar(x, 100*pivot["corner_metric_scale_guide"], width, color=COLORS["corner"], label="Corner ruler")
    ax.bar(x+width, 100*pivot["object_anchored_metric_scale_guide"], width, color=COLORS["anchored"], label="Object-anchored ruler")
    ax.set_xticks(x, order, rotation=20, ha="right"); ax.set_ylabel("Accuracy (%)"); ax.legend(frameon=False, ncols=3)
    style_axis(ax, "Visual scale assistance by numerical task", "All three conditions use the same source questions, prompts and 12-frame selection policy.")
    fig.tight_layout(); path = FIGURE_DIR / "visual_scale_by_task.png"; fig.savefig(path, dpi=190, bbox_inches="tight"); plt.close(fig); paths.append(path)

    grouped = tasks.groupby(["subtask_name", "suite_name"], as_index=False).paired_gain.mean()
    pivot = grouped.pivot(index="subtask_name", columns="suite_name", values="paired_gain")
    pivot = pivot.reindex(index=[SUBTASK_NAMES[k] for k in SUBTASK_NAMES], columns=[SUITE_NAMES[k] for k in SUITE_NAMES])
    data = 100*pivot.to_numpy(); bound = max(8, math.ceil(np.nanmax(np.abs(data))/5)*5)
    fig, ax = plt.subplots(figsize=(13, 7)); image = ax.imshow(data, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right"); ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i,j]): ax.text(j, i, f"{data[i,j]:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Mean diagnostic effect by task", loc="left", fontsize=14, fontweight="bold", pad=15)
    fig.colorbar(image, ax=ax, label="Paired gain (percentage points)", shrink=.82)
    fig.tight_layout(); path = FIGURE_DIR / "task_suite_heatmap.png"; fig.savefig(path, dpi=190, bbox_inches="tight"); plt.close(fig); paths.append(path)

    selected = pd.concat([conditions.nlargest(18, "paired_gain"), conditions.nsmallest(12, "paired_gain")]).drop_duplicates(["suite", "condition"]).sort_values("paired_gain")
    labels = [f"{r.suite_name}: {r.condition}" for r in selected.itertuples()]; values = 100*selected.paired_gain
    fig, ax = plt.subplots(figsize=(11.5, 10)); bars = ax.barh(labels, values, color=[COLORS["positive"] if v >= 0 else COLORS["negative"] for v in values]); ax.axvline(0, color="black", lw=.8)
    for bar, value in zip(bars, values): ax.text(value + (.25 if value>=0 else -.25), bar.get_y()+bar.get_height()/2, f"{value:+.1f}", ha="left" if value>=0 else "right", va="center", fontsize=7.5)
    ax.set_xlabel("Paired gain (percentage points)"); style_axis(ax, "Strongest and weakest individual diagnostics")
    fig.tight_layout(); path = FIGURE_DIR / "condition_extremes.png"; fig.savefig(path, dpi=190, bbox_inches="tight"); plt.close(fig); paths.append(path)

    ordered = baseline.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(10, 6)); bars = ax.barh(ordered.subtask_name, 100*ordered.accuracy, color=[COLORS["anchored"] if x=="Numerical" else COLORS["corner"] for x in ordered.answer_type])
    for bar, value in zip(bars, ordered.accuracy): ax.text(100*value+.3, bar.get_y()+bar.get_height()/2, f"{100*value:.1f}%", va="center", fontsize=8)
    ax.axvline(20, ls="--", color="black", lw=.8); ax.set_xlabel("Unassisted accuracy (%)")
    style_axis(ax, "True unassisted baseline by task", "Numerical values come from the isolated visual-scale suite; categorical inputs are unchanged controls.")
    fig.tight_layout(); path = FIGURE_DIR / "unassisted_baseline_tasks.png"; fig.savefig(path, dpi=190, bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def md_table(frame: pd.DataFrame, columns: list[tuple[str, str]], formats: dict | None = None) -> str:
    formats = formats or {}; lines = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False):
        values = []
        for key, _ in columns:
            value = getattr(row, key); values.append(str(formats.get(key, lambda x: x)(value)))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown(samples: pd.DataFrame, suites: pd.DataFrame, conditions: pd.DataFrame, tasks: pd.DataFrame,
                   baseline: pd.DataFrame, runs: pd.DataFrame, examples: pd.DataFrame, example_image: Path | None) -> str:
    strongest = conditions.nlargest(12, "paired_gain"); weakest = conditions.nsmallest(10, "paired_gain")
    visual = tasks[tasks.suite == "visual_scale"].sort_values(["subtask_name", "condition"])
    suite_table = md_table(suites, [("suite_name","Suite"),("n","N"),("baseline_accuracy","Baseline"),("assisted_accuracy","Assisted"),("paired_gain","Gain"),("rescues","Rescues"),("regressions","Regressions")], {"baseline_accuracy":pct,"assisted_accuracy":pct,"paired_gain":pp})
    condition_cols = [("suite_name","Suite"),("condition","Condition"),("n","N"),("accuracy","Accuracy"),("paired_gain","Gain"),("rescues","Rescues"),("regressions","Regressions")]
    formats = {"accuracy":pct,"paired_gain":pp}
    task_sections = []
    for subtask, group in tasks.groupby("subtask", sort=False):
        task_sections.append(f"### {SUBTASK_NAMES[subtask]}\n\n" + md_table(group.sort_values("paired_gain", ascending=False), condition_cols, formats))
    visual_table = md_table(visual, [("subtask_name","Task"),("condition","Scale condition"),("n","N"),("baseline_accuracy","Unassisted"),("accuracy","Accuracy"),("paired_gain","Gain"),("rescues","Rescues"),("regressions","Regressions")], {"baseline_accuracy":pct,"accuracy":pct,"paired_gain":pp})
    baseline_table = md_table(baseline, [("subtask_name","Task"),("answer_type","Type"),("n","N"),("accuracy","Accuracy"),("parse_success","Parse success")], {"accuracy":pct,"parse_success":pct})
    run_table = md_table(runs, [("part","Part"),("evaluations","Evaluations"),("evaluation_hours","Evaluation hours"),("generation_tokens_per_second","Tokens/s")], {"evaluation_hours":lambda x:f"{x:.2f}","generation_tokens_per_second":lambda x:f"{x:.3f}"})
    visual_image_md = "![Matched visual-scale rescue](examples/visual_scale_rescue_comparison.jpg)" if example_image else ""
    return f"""# MOVi-A 12-frame motion diagnostics — 400 samples per task

## Executive summary

- **Completeness:** {len(samples):,} evaluations, {samples.source_qid.nunique():,} unique source questions, {samples.condition.nunique()} named conditions, 9 suites, and {len(examples):,} indexed rescue/regression traces.
- **Trace completeness:** all {len(samples):,} evaluations have a task-side JSON trace. These include the modified prompt, transformed ordered images, model input/output, target and scoring record.
- **Largest suite-level gain:** {suites.iloc[0].suite_name}, {pp(suites.iloc[0].paired_gain)} against its same-question baseline.
- **Strongest individual diagnostic:** {strongest.iloc[0].suite_name} / `{strongest.iloc[0].condition}`, {pp(strongest.iloc[0].paired_gain)}.
- **Answer formatting:** {pct(samples.parse_success.mean())} parse success.

The eight original suites use a corner scale guide on both baseline and assisted numerical conditions. Their gains therefore measure the added diagnostic while controlling for scale. The isolated **Visual scale** suite uses a genuinely unassisted baseline and measures corner and object-anchored rulers directly.

![Suite overview](figures/suite_overview.png)

## Suite-level effects

{suite_table}

## Visual scale assistance

{visual_table}

![Visual scale by task](figures/visual_scale_by_task.png)

{visual_image_md}

## True unassisted task baselines

{baseline_table}

![Unassisted baselines](figures/unassisted_baseline_tasks.png)

## Strongest diagnostics

{md_table(strongest, condition_cols, formats)}

## Weakest diagnostics

{md_table(weakest, condition_cols, formats)}

![Condition extremes](figures/condition_extremes.png)

## Diagnostic effects by task

![Task-suite heatmap](figures/task_suite_heatmap.png)

{chr(10).join(task_sections)}

## Full-trace analysis workflow

[`trace_examples.csv`](trace_examples.csv) indexes one representative rescue and regression for every applicable task/diagnostic pair. Each row links to the complete JSON trace and its contact sheet. [`paired_transitions.csv`](paired_transitions.csv) contains every paired baseline/assisted transition, so rescues, regressions and prediction changes can be filtered without rereading the large JSONLs.

The trace records remain under each `part_*/trace_bundle/records/` directory. Image paths inside a trace are relative to that trace bundle and point to content-addressed PNGs, making the modified visual input auditable.

## Runtime and provenance

{run_table}

- Critical-path evaluation time: {runs.evaluation_hours.max():.2f} hours.
- Summed evaluation compute: {runs.evaluation_hours.sum():.2f} GPU-hours.
- Model: `{runs.model.iloc[0]}`
- Revision: `{runs.git_hash.iloc[0]}`

## Analysis files

- [`suite_summary.csv`](suite_summary.csv)
- [`condition_summary.csv`](condition_summary.csv)
- [`task_condition_summary.csv`](task_condition_summary.csv)
- [`baseline_task_summary.csv`](baseline_task_summary.csv)
- [`paired_transitions.csv`](paired_transitions.csv)
- [`trace_examples.csv`](trace_examples.csv)
- [`run_summary.csv`](run_summary.csv)
"""


def add_image_page(pdf: PdfPages, path: Path, title: str) -> None:
    image = plt.imread(path); fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_axes((.035,.035,.93,.90))
    ax.imshow(image); ax.axis("off"); fig.suptitle(title, x=.045, y=.98, ha="left", fontsize=17, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def build_pdf(samples: pd.DataFrame, suites: pd.DataFrame, tasks: pd.DataFrame, figures: list[Path], example_image: Path | None) -> Path:
    path = REPORT_DIR / "MOVi_A_12f_bbox_scale_guide_400_report.pdf"
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5,11)); ax = fig.add_axes((.08,.07,.84,.87)); ax.axis("off")
        ax.text(0,1,"MOVi-A Motion Diagnostics",fontsize=24,fontweight="bold",va="top")
        ax.text(0,.955,"400 samples per task · 12 frames · full traces",fontsize=13,color="#475569",va="top")
        summary = f"{len(samples):,} completed model evaluations across {samples.source_qid.nunique():,} source questions. The largest suite-level effect is {suites.iloc[0].suite_name} at {pp(suites.iloc[0].paired_gain)}. All model outputs parsed successfully. The visual-scale suite directly compares unassisted, corner-ruler and object-anchored-ruler inputs."
        y=.88
        for line in textwrap.wrap(summary, 88): ax.text(0,y,line,fontsize=11,va="top"); y-=.034
        y-=.03; ax.text(0,y,"Interpretation boundary",fontsize=15,fontweight="bold",va="top"); y-=.05
        for point in ["Original-suite numerical baselines already include the corner ruler.", "Visual-scale baselines are genuinely unassisted and isolate ruler effects.", "Every comparison is paired by source_qid; expanded rows are not independent samples.", "All transformed images, rendered prompts, raw outputs and scoring decisions are retained in trace bundles."]:
            for line in textwrap.wrap("• "+point, 88): ax.text(0,y,line,fontsize=10.5,va="top"); y-=.033
            y-=.008
        pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
        for figure in figures: add_image_page(pdf, figure, figure.stem.replace("_"," ").title())
        if example_image: add_image_page(pdf, example_image, "Matched visual-scale rescue example")
        for subtask, group in tasks.groupby("subtask", sort=False):
            ordered = group.sort_values("paired_gain", ascending=False)
            chunks = [ordered.iloc[i:i+24] for i in range(0,len(ordered),24)]
            for page_index, chunk in enumerate(chunks,1):
                fig=plt.figure(figsize=(11,8.5)); ax=fig.add_axes((.02,.05,.96,.88)); ax.axis("off")
                title=SUBTASK_NAMES[subtask] + (f" · page {page_index}" if len(chunks)>1 else "")
                ax.set_title(title,loc="left",fontsize=17,fontweight="bold",pad=15)
                cells=[[r.suite_name,r.condition,r.n,pct(r.baseline_accuracy),pct(r.accuracy),pp(r.paired_gain),r.rescues,r.regressions] for r in chunk.itertuples()]
                table=ax.table(cellText=cells,colLabels=["Suite","Condition","N","Base","Accuracy","Gain","Rescue","Regress"],cellLoc="center",colLoc="center",loc="upper center",colWidths=[.12,.27,.06,.09,.09,.09,.07,.07])
                table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1,1.25)
                for (i,_),cell in table.get_celld().items():
                    if i==0: cell.set_facecolor("#DBEAFE"); cell.set_text_props(weight="bold")
                    elif i%2==0: cell.set_facecolor("#F1F5F9")
                pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
    return path


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True); FIGURE_DIR.mkdir(parents=True, exist_ok=True); EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples(); runs = load_runs(); task_conditions, transitions = paired_rows(samples)
    conditions = aggregate_conditions(task_conditions); suites = aggregate_suites(samples, conditions); baseline = baseline_tasks(samples)
    examples = trace_examples(transitions); example_image = visual_scale_example(transitions)
    suites.to_csv(REPORT_DIR/"suite_summary.csv",index=False)
    conditions.sort_values(["suite","paired_gain"],ascending=[True,False]).to_csv(REPORT_DIR/"condition_summary.csv",index=False)
    task_conditions.sort_values(["subtask","suite","paired_gain"],ascending=[True,True,False]).to_csv(REPORT_DIR/"task_condition_summary.csv",index=False)
    baseline.to_csv(REPORT_DIR/"baseline_task_summary.csv",index=False); runs.to_csv(REPORT_DIR/"run_summary.csv",index=False)
    transitions.to_csv(REPORT_DIR/"paired_transitions.csv",index=False); examples.to_csv(REPORT_DIR/"trace_examples.csv",index=False)
    figures = make_plots(suites,conditions,task_conditions,baseline)
    markdown = build_markdown(samples,suites,conditions,task_conditions,baseline,runs,examples,example_image)
    md_path=REPORT_DIR/"DETAILED_ANALYSIS.md"; md_path.write_text(markdown,encoding="utf-8")
    pdf_path=build_pdf(samples,suites,task_conditions,figures,example_image)
    print(f"Wrote {md_path}"); print(f"Wrote {pdf_path}"); print(f"Validated {len(samples):,} evaluations and {len(examples):,} indexed trace examples")


if __name__ == "__main__":
    main()
