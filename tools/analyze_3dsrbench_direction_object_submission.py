#!/usr/bin/env python3
"""Analyze a 3DSRBench direction/object submission and plot paired changes.

The object-answer ``inverse`` row is compared to the direction-answer
``native`` row for every ``source_qid``.  The plot reports how often changing
the prompt answer format improved, worsened, or left correctness unchanged.

Example:
    python tools/analyze_3dsrbench_direction_object_submission.py \
        outputs/3dsrbench_direction_object_0/submissions/\
3dsrbench_direction_object_qwen3_vl_experiments.json \
        --plot outputs/3dsrbench_direction_object_0/paired_outcomes.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("native", "inverse")
OUTCOME_KEYS = ("improved", "worsened", "unchanged_correct", "unchanged_incorrect")


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("results", payload.get("submissions"))
    else:
        records = None
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("Expected a JSON list of submission records")
    return records


def _score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Record {row.get('qid', '<unknown>')} has no numeric score") from exc


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_score(row) for row in rows]
    return {
        "count": len(scores),
        "correct": sum(score == 1.0 for score in scores),
        "accuracy": sum(scores) / len(scores) if scores else None,
    }


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unknown_variants: set[str] = set()
    for row in records:
        variant = str(row.get("variant", ""))
        if variant not in VARIANTS:
            unknown_variants.add(variant)
        grouped[variant].append(row)
        by_source[str(row.get("source_qid", ""))][variant] = row

    outcomes = {key: 0 for key in OUTCOME_KEYS}
    for variants in by_source.values():
        if not {"native", "inverse"} <= variants.keys():
            continue
        native_correct = _score(variants["native"]) == 1.0
        inverse_correct = _score(variants["inverse"]) == 1.0
        if inverse_correct and not native_correct:
            outcomes["improved"] += 1
        elif native_correct and not inverse_correct:
            outcomes["worsened"] += 1
        elif inverse_correct:
            outcomes["unchanged_correct"] += 1
        else:
            outcomes["unchanged_incorrect"] += 1
    outcomes["paired_count"] = sum(outcomes.values())

    report: dict[str, Any] = {
        "records": len(records),
        "by_variant": {variant: _summary(grouped[variant]) for variant in VARIANTS},
        "paired_gain_inverse_minus_native": (
            (outcomes["improved"] - outcomes["worsened"]) / outcomes["paired_count"]
            if outcomes["paired_count"]
            else None
        ),
        "paired_outcomes_inverse_vs_native": outcomes,
    }
    if unknown_variants:
        report["unknown_variants"] = sorted(unknown_variants)
    return report


def _save_plot(report: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; install it to use --plot.") from exc

    counts = report["paired_outcomes_inverse_vs_native"]
    colors = {
        "improved": "#2a9d8f",
        "worsened": "#e76f51",
        "unchanged_correct": "#457b9d",
        "unchanged_incorrect": "#9aa0a6",
    }
    labels = {
        "improved": "Improved\n(wrong → correct)",
        "worsened": "Worsened\n(correct → wrong)",
        "unchanged_correct": "Unchanged:\ncorrect",
        "unchanged_incorrect": "Unchanged:\nincorrect",
    }
    figure, axis = plt.subplots(figsize=(8, 5))
    keys = list(OUTCOME_KEYS)
    bars = axis.bar(
        [labels[key] for key in keys],
        [counts[key] for key in keys],
        color=[colors[key] for key in keys],
    )
    for bar, key in zip(bars, keys):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(counts[key]), ha="center", va="bottom")
    axis.set_ylabel("Number of paired prompts")
    axis.set_title("3DSRBench direction/object prompt-variation outcomes\n(inverse object answer versus native direction answer)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _print_report(report: dict[str, Any]) -> None:
    print(f"Records: {report['records']}")
    print("\nAccuracy by variant")
    for variant, values in report["by_variant"].items():
        accuracy = "N/A" if values["accuracy"] is None else f"{values['accuracy']:.4f}"
        print(f"  {variant}: {values['correct']}/{values['count']} ({accuracy})")
    print("\nPaired correctness transitions: inverse object answer versus native direction answer")
    for key in OUTCOME_KEYS:
        print(f"  {key}: {report['paired_outcomes_inverse_vs_native'][key]}")
    gain = report["paired_gain_inverse_minus_native"]
    print(f"  mean gain: {'N/A' if gain is None else f'{gain:+.4f}'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to the JSON submission list")
    parser.add_argument("--plot", type=Path, metavar="PATH", help="Write the paired-outcomes bar chart")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON")
    args = parser.parse_args()

    report = analyze(_load_records(args.submission))
    if args.plot:
        _save_plot(report, args.plot)
        print(f"Saved paired-outcomes plot to {args.plot}", file=sys.stderr)
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
