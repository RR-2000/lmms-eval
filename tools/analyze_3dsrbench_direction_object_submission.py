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
RELATIONS = ("left", "right", "front", "behind")


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


def _paired_outcomes(by_source: dict[str, dict[str, dict[str, Any]]]) -> dict[str, int]:
    """Count the effect of replacing a direction answer with an object answer."""
    outcomes = {key: 0 for key in OUTCOME_KEYS}
    for variants in by_source.values():
        if not {"native", "inverse"} <= variants.keys():
            continue
        direction_correct = _score(variants["native"]) == 1.0
        object_correct = _score(variants["inverse"]) == 1.0
        if object_correct and not direction_correct:
            outcomes["improved"] += 1
        elif direction_correct and not object_correct:
            outcomes["worsened"] += 1
        elif object_correct:
            outcomes["unchanged_correct"] += 1
        else:
            outcomes["unchanged_incorrect"] += 1
    outcomes["paired_count"] = sum(outcomes.values())
    return outcomes


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

    outcomes = _paired_outcomes(by_source)
    by_direction: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    for source_qid, variants in by_source.items():
        direction = next((str(row.get("direction", "")) for row in variants.values()), "")
        if direction:
            by_direction[direction][source_qid] = variants
    paired_outcomes_by_direction = {
        "overall": outcomes,
        **{
            direction: _paired_outcomes(by_direction[direction])
            for direction in RELATIONS
            if direction in by_direction
        },
    }

    report: dict[str, Any] = {
        "records": len(records),
        "by_variant": {variant: _summary(grouped[variant]) for variant in VARIANTS},
        "paired_gain_inverse_minus_native": (
            (outcomes["improved"] - outcomes["worsened"]) / outcomes["paired_count"]
            if outcomes["paired_count"]
            else None
        ),
        "paired_outcomes_inverse_vs_native": outcomes,
        "paired_outcomes_object_vs_direction_by_direction": paired_outcomes_by_direction,
    }
    if unknown_variants:
        report["unknown_variants"] = sorted(unknown_variants)
    return report


def _save_plot(report: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; install it to use --plot.") from exc

    outcomes_by_direction = report["paired_outcomes_object_vs_direction_by_direction"]
    labels = ["overall", *(direction for direction in RELATIONS if direction in outcomes_by_direction)]
    labels = [label for label in labels if outcomes_by_direction[label]["paired_count"]]
    if not labels:
        raise ValueError("No complete direction/object pairs were found for plotting.")
    colors = {
        "improved": "#2a9d8f",
        "worsened": "#e76f51",
        "unchanged_correct": "#457b9d",
        "unchanged_incorrect": "#9aa0a6",
    }
    display = {
        "improved": "Improved (direction wrong → object correct)",
        "worsened": "Worsened (direction correct → object wrong)",
        "unchanged_correct": "Unchanged: correct",
        "unchanged_incorrect": "Unchanged: incorrect",
    }
    figure, axis = plt.subplots(figsize=(11, 6))
    bottom = [0] * len(labels)
    for key in OUTCOME_KEYS:
        values = [outcomes_by_direction[label][key] for label in labels]
        axis.bar(labels, values, bottom=bottom, label=display[key], color=colors[key])
        bottom = [current + value for current, value in zip(bottom, values)]
    axis.set_ylabel("Number of matched source relations")
    axis.set_title("3DSRBench direction/object paired outcomes")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _default_plot_path(submission: Path) -> Path:
    """Place the chart in the run directory when given its submissions file."""
    if submission.parent.name == "submissions":
        return submission.parent.parent / "3dsrbench_paired_outcomes.png"
    return submission.with_name(f"{submission.stem}_paired_outcomes.png")


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
    print("\nDirection-to-object paired outcomes by direction")
    for direction, counts in report["paired_outcomes_object_vs_direction_by_direction"].items():
        print(
            f"  {direction}: improved={counts['improved']}, worsened={counts['worsened']}, "
            f"unchanged-correct={counts['unchanged_correct']}, "
            f"unchanged-incorrect={counts['unchanged_incorrect']} "
            f"({counts['paired_count']} pairs)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to the JSON submission list")
    parser.add_argument(
        "--plot",
        type=Path,
        metavar="PATH",
        help="Write the paired-outcomes bar chart (default: run directory)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON")
    args = parser.parse_args()

    report = analyze(_load_records(args.submission))
    plot_path = args.plot or _default_plot_path(args.submission)
    _save_plot(report, plot_path)
    print(f"Saved paired-outcomes plot to {plot_path}", file=sys.stderr)
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
