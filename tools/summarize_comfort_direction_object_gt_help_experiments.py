#!/usr/bin/env python3
"""Compare COMFORT direction/object GT-help experiment submissions.

The tool discovers ``gt_help_<mode>/evaluation/submissions/*.json`` below an
experiment root, applies the same per-submission analysis as
``analyze_comfort_direction_object_submission.py``, and writes combined JSON,
CSV, text, and accuracy-plot artifacts.  When mode 0 is present, each other
mode is also paired by qid against that baseline to report correctness
transitions and mean accuracy/parse-success gains. Modes 0--14 plus short-arrow
controls 36 and 37 are selected by default; use ``--modes`` to choose another
range or comma-separated set.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from analyze_comfort_direction_object_submission import (
    _load_records,
    _save_paired_outcomes_plot as save_single_paired_outcomes_plot,
    analyze,
    format_report,
)
from analyze_kubric_movi_e_direction_object_submission import ANSWER_FORMATS, RELATIONS
from plot_comfort_direction_object_distributions import (
    DEFAULT_SCENES,
    count_distributions,
    load_object_directions,
    save_plot as save_distribution_plot,
    serialise,
)


DEFAULT_ROOT = Path("/home/ramanathan/VLM/lmms-eval/outputs/" "comforter_comfort_direction_object_gt_help_all_variants_debug")
MODE_LABELS = {
    "0": "baseline",
    "1": "reference bbox",
    "2": "answer bbox",
    "3": "all-object bboxes",
    "4": "reference-front arrow",
    "5": "plain-text GT",
    "6": "four direction arrows",
    "7": "arrows + perspective text",
    "8": "perspective text",
    "9": "reference vs camera example",
    "10": "example + arrows",
    "11": "named top-down map",
    "12": "response-format scaffold",
    "13": "neutral worked example",
    "14": "color-only top-down map",
    "15": "fully labeled top-down map",
    "16": "reference vs camera maps",
    "17": "reference identity text",
    "18": "reference bbox + crop",
    "19": "numbered object bboxes",
    "20": "reference bbox + heading",
    "21": "reference bbox + labeled front",
    "22": "color-coded direction arrows",
    "23": "canonical numeric layout",
    "24": "top-down map only",
    "25": "query-pair map",
    "26": "highlighted-query map",
    "27": "camera map control",
    "28": "dual maps without frame command",
    "29": "dual maps + reference command",
    "30": "text camera heading",
    "31": "text axis mapping",
    "32": "object-row relation oracle",
    "33": "direction-row target localization",
    "34": "answer-text oracle",
    "35": "answer-letter oracle",
    "36": "short four direction arrows",
    "37": "short front/left arrows",
}
OUTPUT_STEM = "comfort_direction_object_gt_help_experiment_summary"
DEFAULT_MODES = "0-14,36-37"


def _mode_sort_key(mode: str) -> tuple[int, int | str]:
    return (0, int(mode)) if mode.isdigit() else (1, mode)


def parse_mode_spec(spec: str) -> list[str]:
    """Parse values such as ``0-14`` or ``0,4,6-10`` into ordered modes."""
    modes: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"Invalid mode range {item!r}")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Mode range must be ascending: {item!r}")
            modes.update(range(start, end + 1))
        elif item.isdigit():
            modes.add(int(item))
        else:
            raise ValueError(f"Invalid mode {item!r}")
    if not modes:
        raise ValueError("At least one GT_HELP mode must be selected")
    return [str(mode) for mode in sorted(modes)]


def _read_payload(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _infer_mode(path: Path, records: list[dict[str, Any]]) -> str | None:
    payload = _read_payload(path)
    if isinstance(payload, dict) and payload.get("gt_help_mode") is not None:
        return str(payload["gt_help_mode"])

    record_modes = {str(row["gt_help_mode"]) for row in records if row.get("gt_help_mode") is not None}
    if len(record_modes) == 1:
        return next(iter(record_modes))
    if len(record_modes) > 1:
        raise ValueError(f"Submission mixes GT-help modes {sorted(record_modes)}: {path}")

    for part in reversed(path.parts):
        match = re.fullmatch(r"gt_help_(\d+)", part)
        if match:
            return match.group(1)
    match = re.search(r"comfort_direction_object_gt_help_(\d+)_", path.name)
    return match.group(1) if match else None


def _parse_explicit_submissions(values: Iterable[str]) -> dict[str, Path]:
    submissions: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--submission expects MODE=PATH, got {value!r}")
        mode, raw_path = value.split("=", 1)
        mode = mode.strip()
        path = Path(raw_path).expanduser().resolve()
        if not mode or not path.is_file():
            raise ValueError(f"Invalid explicit submission {value!r}")
        if mode in submissions:
            raise ValueError(f"Mode {mode!r} was supplied more than once")
        submissions[mode] = path
    return submissions


def discover_submissions(root: Path, expected_modes: Iterable[str] | None = None) -> tuple[dict[str, Path], list[str]]:
    """Find one wrapped task submission per mode and report absent modes."""
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment root not found: {root}")
    candidates = sorted(path for path in root.rglob("*.json") if path.parent.name == "submissions" and path.name.startswith("comfort_direction_object_gt_help_"))
    expected = set(expected_modes) if expected_modes is not None else None
    by_mode: dict[str, list[Path]] = {}
    for path in candidates:
        records = _load_records(path)
        mode = _infer_mode(path, records)
        if mode is None:
            continue
        if expected is not None and mode not in expected:
            continue
        by_mode.setdefault(mode, []).append(path)

    duplicates = {mode: paths for mode, paths in by_mode.items() if len(paths) > 1}
    if duplicates:
        details = "\n".join(f"  mode {mode}: " + ", ".join(str(path) for path in paths) for mode, paths in sorted(duplicates.items(), key=lambda item: _mode_sort_key(item[0])))
        raise ValueError("Multiple submissions were found for a mode. Use repeated " f"--submission MODE=PATH to select explicitly:\n{details}")
    submissions = {mode: paths[0] for mode, paths in by_mode.items()}
    checked_modes = list(expected_modes) if expected_modes is not None else list(MODE_LABELS)
    missing = [mode for mode in checked_modes if mode not in submissions]
    return submissions, missing


def _qid_map(records: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in records:
        qid = str(row.get("qid") or row.get("index") or "")
        if not qid:
            raise ValueError(f"A record in {path} has no qid/index")
        if qid in rows:
            raise ValueError(f"Duplicate qid {qid!r} in {path}")
        rows[qid] = row
    return rows


def _score(row: dict[str, Any]) -> float:
    try:
        return float(row["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Record {row.get('qid', '<unknown>')} has invalid score") from exc


def compare_to_baseline(
    baseline: list[dict[str, Any]],
    variant: list[dict[str, Any]],
    baseline_path: Path,
    variant_path: Path,
) -> dict[str, Any]:
    """Pair two runs by qid and summarize their correctness transitions."""
    base_by_qid = _qid_map(baseline, baseline_path)
    variant_by_qid = _qid_map(variant, variant_path)
    common = sorted(set(base_by_qid) & set(variant_by_qid))
    outcomes = {
        "improved": 0,
        "worsened": 0,
        "unchanged_correct": 0,
        "unchanged_incorrect": 0,
    }
    score_differences = []
    parse_differences = []
    for qid in common:
        baseline_row = base_by_qid[qid]
        variant_row = variant_by_qid[qid]
        baseline_score = _score(baseline_row)
        variant_score = _score(variant_row)
        score_differences.append(variant_score - baseline_score)
        parse_differences.append(float(bool(variant_row.get("parse_success"))) - float(bool(baseline_row.get("parse_success"))))
        baseline_correct = baseline_score == 1.0
        variant_correct = variant_score == 1.0
        if variant_correct and not baseline_correct:
            outcomes["improved"] += 1
        elif baseline_correct and not variant_correct:
            outcomes["worsened"] += 1
        elif variant_correct:
            outcomes["unchanged_correct"] += 1
        else:
            outcomes["unchanged_incorrect"] += 1
    return {
        "paired_count": len(common),
        "baseline_only_count": len(set(base_by_qid) - set(variant_by_qid)),
        "variant_only_count": len(set(variant_by_qid) - set(base_by_qid)),
        "accuracy_gain": (sum(score_differences) / len(score_differences) if score_differences else None),
        "parse_success_gain": (sum(parse_differences) / len(parse_differences) if parse_differences else None),
        **outcomes,
    }


def build_summary(submissions: dict[str, Path], expected_modes: Iterable[str] | None = None) -> dict[str, Any]:
    if not submissions:
        raise ValueError("No GT-help submissions were supplied or discovered")
    records_by_mode: dict[str, list[dict[str, Any]]] = {}
    variants = {}
    for mode, path in sorted(submissions.items(), key=lambda item: _mode_sort_key(item[0])):
        records = _load_records(path)
        inferred_mode = _infer_mode(path, records)
        if inferred_mode is not None and inferred_mode != mode:
            raise ValueError(f"Submission selected as mode {mode} identifies itself as mode " f"{inferred_mode}: {path}")
        records_by_mode[mode] = records
        variants[mode] = {
            "label": MODE_LABELS.get(mode, f"mode {mode}"),
            "submission": str(path),
            "analysis": analyze(records),
        }

    baseline_comparisons = {}
    if "0" in records_by_mode:
        for mode, records in records_by_mode.items():
            if mode == "0":
                continue
            baseline_comparisons[mode] = compare_to_baseline(
                records_by_mode["0"],
                records,
                submissions["0"],
                submissions[mode],
            )
    return {
        "requested_modes": list(expected_modes) if expected_modes is not None else sorted(variants, key=_mode_sort_key),
        "modes": sorted(variants, key=_mode_sort_key),
        "missing_expected_modes": [mode for mode in (expected_modes if expected_modes is not None else MODE_LABELS) if mode not in variants],
        "variants": variants,
        "comparisons_to_mode_0": baseline_comparisons,
    }


def _metric_text(value: Any, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.4f}" if signed else f"{float(value):.4f}"


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "COMFORT direction/object GT-help experiment summary",
        "",
        (f"{'mode':<5} {'aid':<23} {'records':>8} {'overall':>9} " f"{'direction':>10} {'object':>9} {'parse':>9} {'dir-p':>8} " f"{'obj-p':>8} {'obj-dir':>9} {'delta(0)':>9}"),
        "-" * 125,
    ]
    comparisons = summary["comparisons_to_mode_0"]
    for mode in summary["modes"]:
        variant = summary["variants"][mode]
        analysis = variant["analysis"]
        comparison = comparisons.get(mode)
        delta = 0.0 if mode == "0" else comparison.get("accuracy_gain") if comparison else None
        lines.append(
            f"{mode:<5} {variant['label']:<23} {analysis['records']:>8} "
            f"{_metric_text(analysis['overall']['accuracy']):>9} "
            f"{_metric_text(analysis['by_answer_format']['direction']['accuracy']):>10} "
            f"{_metric_text(analysis['by_answer_format']['object']['accuracy']):>9} "
            f"{_metric_text(analysis['overall']['parse_success_rate']):>9} "
            f"{_metric_text(analysis['by_answer_format']['direction']['parse_success_rate']):>8} "
            f"{_metric_text(analysis['by_answer_format']['object']['parse_success_rate']):>8} "
            f"{_metric_text(analysis['paired']['object_minus_direction'], signed=True):>9} "
            f"{_metric_text(delta, signed=True):>9}"
        )

    if comparisons:
        lines.extend(["", "Per-question transitions versus mode 0"])
        for mode in sorted(comparisons, key=_mode_sort_key):
            row = comparisons[mode]
            lines.append(
                f"  {mode} ({MODE_LABELS.get(mode, mode)}): "
                f"improved={row['improved']}, worsened={row['worsened']}, "
                f"unchanged-correct={row['unchanged_correct']}, "
                f"unchanged-incorrect={row['unchanged_incorrect']}, "
                f"paired={row['paired_count']}"
            )
    elif "0" not in summary["modes"]:
        lines.extend(["", "Mode 0 is absent; baseline transition comparisons were skipped."])

    lines.extend(["", "Accuracy by relation"])
    for mode in summary["modes"]:
        analysis = summary["variants"][mode]["analysis"]
        relation_values = " ".join(f"{relation}={_metric_text(analysis['by_relation'][relation]['overall']['accuracy'])}" for relation in RELATIONS)
        lines.append(f"  {mode}: {relation_values}")
    if summary["missing_expected_modes"]:
        lines.extend(["", f"Missing expected modes: {', '.join(summary['missing_expected_modes'])}"])
    return "\n".join(lines) + "\n"


def _csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    comparisons = summary["comparisons_to_mode_0"]
    for mode in summary["modes"]:
        variant = summary["variants"][mode]
        analysis = variant["analysis"]
        comparison = comparisons.get(mode, {})
        row = {
            "mode": mode,
            "aid": variant["label"],
            "submission": variant["submission"],
            "records": analysis["records"],
            "overall_accuracy": analysis["overall"]["accuracy"],
            "direction_accuracy": analysis["by_answer_format"]["direction"]["accuracy"],
            "object_accuracy": analysis["by_answer_format"]["object"]["accuracy"],
            "parse_success_rate": analysis["overall"]["parse_success_rate"],
            "direction_parse_success_rate": analysis["by_answer_format"]["direction"]["parse_success_rate"],
            "object_parse_success_rate": analysis["by_answer_format"]["object"]["parse_success_rate"],
            "object_minus_direction": analysis["paired"]["object_minus_direction"],
            "format_switch_gain": analysis["paired"]["object_minus_direction"],
            "object_correct_direction_wrong": analysis["paired"]["improved"] / analysis["paired"]["paired_count"],
            "direction_correct_object_wrong": analysis["paired"]["worsened"] / analysis["paired"]["paired_count"],
            "both_correct": analysis["paired"]["unchanged_correct"] / analysis["paired"]["paired_count"],
            "both_wrong": analysis["paired"]["unchanged_incorrect"] / analysis["paired"]["paired_count"],
            "accuracy_gain_vs_0": 0.0 if mode == "0" else comparison.get("accuracy_gain"),
            "parse_success_gain_vs_0": 0.0 if mode == "0" else comparison.get("parse_success_gain"),
            "paired_with_0": comparison.get("paired_count"),
            "improved_vs_0": comparison.get("improved"),
            "worsened_vs_0": comparison.get("worsened"),
        }
        for relation in RELATIONS:
            row[f"{relation}_accuracy"] = analysis["by_relation"][relation]["overall"]["accuracy"]
        rows.append(row)
    return rows


def format_markdown(summary: dict[str, Any]) -> str:
    """Return a compact report suitable for direct experiment comparison."""
    csv_rows = _csv_rows(summary)
    lines = [
        "# COMFORT direction/object GT_HELP summary",
        "",
        "| Mode | Aid | Records | Overall | Direction | Object | Parse | Object−direction | Gain vs 0 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in csv_rows:
        gain = row["accuracy_gain_vs_0"]
        gain_text = "N/A" if gain is None else f"{gain:+.1%}"
        lines.append(
            f"| {row['mode']} | {row['aid']} | {row['records']} | "
            f"{row['overall_accuracy']:.1%} | {row['direction_accuracy']:.1%} | "
            f"{row['object_accuracy']:.1%} | {row['parse_success_rate']:.1%} | "
            f"{row['object_minus_direction']:+.1%} | {gain_text} |"
        )

    lines.extend(
        [
            "",
            "## Paired object-versus-direction outcomes",
            "",
            "| Mode | Both correct | Object only | Direction only | Both wrong |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in csv_rows:
        lines.append(f"| {row['mode']} | {row['both_correct']:.1%} | " f"{row['object_correct_direction_wrong']:.1%} | " f"{row['direction_correct_object_wrong']:.1%} | " f"{row['both_wrong']:.1%} |")

    if summary["missing_expected_modes"]:
        lines.extend(
            [
                "",
                "## Missing runs",
                "",
                ", ".join(summary["missing_expected_modes"]),
            ]
        )
    return "\n".join(lines) + "\n"


def save_accuracy_plot(summary: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; use --no-plot to skip it") from exc

    modes = summary["modes"]
    labels = [f"{mode}\n{summary['variants'][mode]['label']}" for mode in modes]
    series = {
        "overall": [summary["variants"][mode]["analysis"]["overall"]["accuracy"] for mode in modes],
        "direction": [summary["variants"][mode]["analysis"]["by_answer_format"]["direction"]["accuracy"] for mode in modes],
        "object": [summary["variants"][mode]["analysis"]["by_answer_format"]["object"]["accuracy"] for mode in modes],
    }
    figure, axis = plt.subplots(figsize=(max(9, len(modes) * 1.7), 6))
    width = 0.24
    centers = list(range(len(modes)))
    colors = {"overall": "#4C78A8", "direction": "#F58518", "object": "#54A24B"}
    for offset, (name, values) in zip((-width, 0.0, width), series.items()):
        positions = [center + offset for center in centers]
        clean_values = [0.0 if value is None else value for value in values]
        bars = axis.bar(positions, clean_values, width=width, label=name, color=colors[name])
        for bar, value in zip(bars, values):
            if value is not None:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(0.99, value + 0.015),
                    f"{value:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
    axis.set_xticks(centers, labels, rotation=25, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Accuracy")
    axis.set_title("COMFORT direction/object accuracy by GT-help mode")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_paired_outcomes_plot(summary: dict[str, Any], path: Path) -> None:
    """Plot the mutually exclusive object/direction outcomes for every mode."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; use --no-plot to skip it") from exc

    outcome_specs = (
        ("unchanged_correct", "Both correct", "#2A9D8F"),
        ("improved", "Object correct, direction wrong", "#E9C46A"),
        ("worsened", "Direction correct, object wrong", "#F4A261"),
        ("unchanged_incorrect", "Both wrong", "#E76F51"),
    )
    modes = summary["modes"]
    labels = [f"{mode}: {summary['variants'][mode]['label']}" for mode in modes]
    figure, axis = plt.subplots(figsize=(13, max(5, 0.55 * len(modes) + 2)))
    left = [0.0] * len(modes)
    for key, label, color in outcome_specs:
        values = []
        for mode in modes:
            paired = summary["variants"][mode]["analysis"]["paired"]
            count = paired["paired_count"]
            values.append(paired[key] / count if count else 0.0)
        bars = axis.barh(labels, values, left=left, label=label, color=color)
        for bar, value, offset in zip(bars, values, left):
            if value >= 0.045:
                axis.text(
                    offset + value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        left = [current + value for current, value in zip(left, values)]
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Share of matched object/direction pairs")
    axis.set_title("COMFORT paired outcomes by GT_HELP mode")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    axis.invert_yaxis()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_baseline_transition_plot(summary: dict[str, Any], path: Path) -> None:
    """Plot per-question improvements and regressions relative to mode 0."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; use --no-plot to skip it") from exc

    comparisons = summary["comparisons_to_mode_0"]
    modes = [mode for mode in summary["modes"] if mode in comparisons]
    if not modes:
        return
    labels = [f"{mode}: {summary['variants'][mode]['label']}" for mode in modes]
    improved = [comparisons[mode]["improved"] / comparisons[mode]["paired_count"] for mode in modes]
    worsened = [-comparisons[mode]["worsened"] / comparisons[mode]["paired_count"] for mode in modes]
    positions = list(range(len(modes)))
    figure, axis = plt.subplots(figsize=(13, max(5, 0.5 * len(modes) + 2)))
    axis.barh(positions, improved, label="Mode 0 wrong → variant correct", color="#2A9D8F")
    axis.barh(positions, worsened, label="Mode 0 correct → variant wrong", color="#E76F51")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(positions, labels)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Share of matched questions (regressions shown negative)")
    axis.set_title("COMFORT correctness transitions relative to GT_HELP=0")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False)
    axis.invert_yaxis()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _case_rows(
    mode: str,
    answer_format: str,
    counts: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    categories = [*RELATIONS, "unmapped object", "parse failure"]
    for gold_direction in RELATIONS:
        for selected_direction in categories:
            rows.append(
                {
                    "mode": mode,
                    "answer_format": answer_format,
                    "ground_truth_direction": gold_direction,
                    "selected_direction": selected_direction,
                    "count": counts[gold_direction][selected_direction],
                }
            )
    return rows


def _confusion_markdown(title: str, counts: dict[str, Any]) -> list[str]:
    categories = [*RELATIONS, "unmapped object", "parse failure"]
    lines = [
        f"### {title}",
        "",
        "| Ground truth | " + " | ".join(categories) + " | Total |",
        "|---|" + "---:|" * (len(categories) + 1),
    ]
    for gold_direction in RELATIONS:
        values = [counts[gold_direction][category] for category in categories]
        lines.append(f"| {gold_direction} | " + " | ".join(str(value) for value in values) + f" | {sum(values)} |")
    return lines


def format_mode_markdown(
    mode: str,
    label: str,
    analysis: dict[str, Any],
    direction_counts: dict[str, Any],
    object_counts: dict[str, Any],
    skipped: dict[str, int],
) -> str:
    lines = [
        f"# GT_HELP={mode}: {label}",
        "",
        "## Accuracy by answer case",
        "",
        "| Ground-truth direction | Direction accuracy | Object accuracy | Object−direction | Both correct | Object only | Direction only | Both wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for relation in RELATIONS:
        report = analysis["by_relation"][relation]
        paired = report["paired"]
        count = paired["paired_count"]

        def rate(key: str) -> float:
            return paired[key] / count if count else 0.0

        lines.append(
            f"| {relation} | {report['direction']['accuracy']:.1%} | "
            f"{report['object']['accuracy']:.1%} | "
            f"{paired['object_minus_direction']:+.1%} | "
            f"{rate('unchanged_correct'):.1%} | {rate('improved'):.1%} | "
            f"{rate('worsened'):.1%} | {rate('unchanged_incorrect'):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Selected-direction confusion tables",
            "",
            "For object-answer questions, the selected option is resolved back to its actual reference-relative direction using `scenes.jsonl`.",
            "",
            *_confusion_markdown("Direction-answer questions", direction_counts),
            "",
            *_confusion_markdown("Object-answer questions", object_counts),
            "",
            "## Option-letter distributions",
            "",
            f"- Gold: `{analysis['gold_option_distribution']}`",
            f"- Predicted: `{analysis['predicted_option_distribution']}`",
        ]
    )
    if skipped:
        lines.extend(["", f"Skipped/unmapped metadata cases: `{skipped}`"])
    return "\n".join(lines) + "\n"


def save_per_mode_details(
    summary: dict[str, Any],
    output_dir: Path,
    scenes_path: Path,
    plot: bool,
) -> list[Path]:
    """Write analyzer-style reports and answer-direction plots for each mode."""
    object_directions = load_object_directions(scenes_path)
    per_mode_root = output_dir / "per_mode"
    per_mode_root.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "# Per-mode GT_HELP answer-case reports",
        "",
        "| Mode | Aid | Detailed report |",
        "|---:|---|---|",
    ]
    case_rows = []
    generated_dirs = []
    for mode in summary["modes"]:
        variant = summary["variants"][mode]
        analysis = variant["analysis"]
        records = _load_records(Path(variant["submission"]))
        direction_counts, object_counts, skipped = count_distributions(records, object_directions)
        mode_dir = per_mode_root / f"gt_help_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": mode,
            "label": variant["label"],
            "submission": variant["submission"],
            "analysis": analysis,
            "selected_direction_distributions": {
                "direction_tasks": serialise(direction_counts),
                "object_tasks": serialise(object_counts),
                "skipped": skipped,
            },
        }
        (mode_dir / "answer_case_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (mode_dir / "answer_case_report.txt").write_text(format_report(analysis), encoding="utf-8")
        (mode_dir / "answer_case_report.md").write_text(
            format_mode_markdown(
                mode,
                variant["label"],
                analysis,
                direction_counts,
                object_counts,
                skipped,
            ),
            encoding="utf-8",
        )
        case_rows.extend(_case_rows(mode, "direction", direction_counts))
        case_rows.extend(_case_rows(mode, "object", object_counts))
        if plot:
            save_single_paired_outcomes_plot(analysis, mode_dir / "paired_outcomes_by_relation.png")
            save_distribution_plot(
                direction_counts,
                f"GT_HELP={mode} direction tasks: ground truth vs selected direction",
                mode_dir / "direction_gt_vs_predicted_direction.png",
            )
            save_distribution_plot(
                object_counts,
                f"GT_HELP={mode} object tasks: ground truth vs selected-object direction",
                mode_dir / "object_gt_vs_predicted_object_direction.png",
            )
        index_lines.append(f"| {mode} | {variant['label']} | " f"[report](gt_help_{mode}/answer_case_report.md) |")
        generated_dirs.append(mode_dir)

    case_csv = per_mode_root / "all_modes_answer_cases.csv"
    with case_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    index_path = per_mode_root / "README.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return [index_path, case_csv, *generated_dirs]


def save_summary(
    summary: dict[str, Any],
    output_dir: Path,
    plot: bool,
    scenes_path: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{OUTPUT_STEM}.json"
    text_path = output_dir / f"{OUTPUT_STEM}.txt"
    markdown_path = output_dir / f"{OUTPUT_STEM}.md"
    csv_path = output_dir / f"{OUTPUT_STEM}.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(format_summary(summary), encoding="utf-8")
    markdown_path.write_text(format_markdown(summary), encoding="utf-8")
    rows = _csv_rows(summary)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifacts = [json_path, text_path, markdown_path, csv_path]
    if plot:
        plot_path = output_dir / f"{OUTPUT_STEM}_accuracy.png"
        save_accuracy_plot(summary, plot_path)
        paired_path = output_dir / f"{OUTPUT_STEM}_paired_outcomes_100pct.png"
        save_paired_outcomes_plot(summary, paired_path)
        artifacts.extend([plot_path, paired_path])
        if summary["comparisons_to_mode_0"]:
            transition_path = output_dir / f"{OUTPUT_STEM}_transitions_vs_0.png"
            save_baseline_transition_plot(summary, transition_path)
            artifacts.append(transition_path)
    artifacts.extend(save_per_mode_details(summary, output_dir, scenes_path=scenes_path, plot=plot))
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Experiment root to scan (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--submission",
        action="append",
        default=[],
        metavar="MODE=PATH",
        help="Select a submission explicitly; repeat for multiple modes",
    )
    parser.add_argument(
        "--modes",
        default=DEFAULT_MODES,
        metavar="SPEC",
        help=("Modes to include, as a range or comma-separated values " f"(default: {DEFAULT_MODES}; example: 0,4,6-10,14)"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Summary destination (default: <root>/summary)",
    )
    parser.add_argument(
        "--scenes",
        type=Path,
        default=DEFAULT_SCENES,
        help=("COMFORT scenes.jsonl used to resolve selected objects to their " f"directions (default: {DEFAULT_SCENES})"),
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not generate the PNG chart")
    parser.add_argument("--json", action="store_true", dest="print_json", help="Print JSON instead of text")
    args = parser.parse_args()

    try:
        expected_modes = parse_mode_spec(args.modes)
        if args.submission:
            submissions = _parse_explicit_submissions(args.submission)
            unexpected = sorted(set(submissions) - set(expected_modes), key=_mode_sort_key)
            if unexpected:
                raise ValueError("Explicit submissions outside --modes: " + ", ".join(unexpected))
        else:
            submissions, _ = discover_submissions(args.root.resolve(), expected_modes=expected_modes)
        summary = build_summary(submissions, expected_modes=expected_modes)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    output_dir = (args.output_dir or args.root / "summary").resolve()
    artifacts = save_summary(
        summary,
        output_dir,
        plot=not args.no_plot,
        scenes_path=args.scenes.resolve(),
    )
    print(json.dumps(summary, indent=2) if args.print_json else format_summary(summary), end="")
    for artifact in artifacts:
        print(f"Saved: {artifact}")


if __name__ == "__main__":
    main()
