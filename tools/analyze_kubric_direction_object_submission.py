#!/usr/bin/env python3
"""Report per-source-family metrics from a Kubric direction/object submission.

The extended task produces four records for each source question:
native, inverse, direction_natural_language, and object_direction_exhaustive.
This tool reads the submission records written by the task's submission metric;
it does not rerun model inference or reconstruct the dataset.

Example:
    python tools/analyze_kubric_direction_object_submission.py \
        outputs/kubric_movi_a_direction_object_extended_0/submissions/\
kubric_movi_a_direction_object_extended_qwen3_vl_experiments.json \
        --plot outputs/kubric_movi_a_direction_object_extended_0/kubric_paired_outcomes.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SOURCE_FAMILIES = (
    "object_centric_relative_position",
    "object_centric_relative_position_multi",
)
VARIANTS = (
    "native",
    "inverse",
    "direction_natural_language",
    "object_direction_exhaustive",
)
RELATIONS = ("left", "right", "front", "behind")
OUTCOME_KEYS = ("improved", "worsened", "unchanged_correct", "unchanged_incorrect")


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    # The current submission writer emits a bare list. Accept a few common
    # wrappers as well, so the tool remains useful with copied submissions.
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
    value = row.get("score")
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Record {row.get('qid', '<unknown>')} has no numeric score") from exc


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_score(row) for row in rows]
    return {
        "count": len(scores),
        "correct": sum(score == 1.0 for score in scores),
        "accuracy": sum(scores) / len(scores) if scores else None,
    }


def _paired_outcomes(
    sources: dict[str, dict[str, dict[str, Any]]], left: str, right: str
) -> dict[str, int]:
    """Count correctness transitions when swapping ``right`` for ``left``."""
    outcomes = {
        "improved": 0,
        "worsened": 0,
        "unchanged_correct": 0,
        "unchanged_incorrect": 0,
    }
    for variants in sources.values():
        if left not in variants or right not in variants:
            continue
        left_correct = _score(variants[left]) == 1.0
        right_correct = _score(variants[right]) == 1.0
        if left_correct and not right_correct:
            outcomes["improved"] += 1
        elif right_correct and not left_correct:
            outcomes["worsened"] += 1
        elif left_correct:
            outcomes["unchanged_correct"] += 1
        else:
            outcomes["unchanged_incorrect"] += 1
    outcomes["paired_count"] = sum(outcomes.values())
    return outcomes


def _format_paired_outcomes(
    sources: dict[tuple[str, str], dict[str, dict[str, Any]]]
) -> dict[str, int]:
    """Compare the base object-answer and direction-answer prompts by format.

    ``native`` and ``inverse`` swap answer formats between the single- and
    multi-object source families, so choosing rows by ``answer_format`` is
    less error-prone than assuming one fixed variant represents each format.
    """
    outcomes = {key: 0 for key in OUTCOME_KEYS}
    for rows in sources.values():
        direction = next(
            (row for row in rows.values() if row.get("answer_format") == "direction"), None
        )
        object_answer = next(
            (row for row in rows.values() if row.get("answer_format") == "object"), None
        )
        if direction is None or object_answer is None:
            continue
        direction_correct = _score(direction) == 1.0
        object_correct = _score(object_answer) == 1.0
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
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    unknown_families: set[str] = set()
    unknown_variants: set[str] = set()

    for row in records:
        family = str(row.get("source_task_family", ""))
        variant = str(row.get("variant", ""))
        if family not in SOURCE_FAMILIES:
            unknown_families.add(family)
        if variant not in VARIANTS:
            unknown_variants.add(variant)
        grouped[(family, variant)].append(row)
        # Include family in the key so copied/generated datasets with reused
        # source ids cannot accidentally be paired across task families.
        source_key = (family, str(row.get("source_qid", "")))
        by_source[source_key][variant] = row

    by_family = {
        family: {variant: _summary(grouped[(family, variant)]) for variant in VARIANTS}
        for family in SOURCE_FAMILIES
    }
    by_variant = {variant: _summary(grouped_for_variant)
                  for variant in VARIANTS
                  for grouped_for_variant in [[row for row in records if row.get("variant") == variant]]}

    paired = {}
    paired_outcomes = {}
    for family in SOURCE_FAMILIES:
        family_sources = {
            source: variants
            for source, variants in by_source.items()
            if source[0] == family
        }
        paired[family] = {}
        paired_outcomes[family] = {}
        # The simple direction baseline is native for the single-object
        # source family, but inverse for the multi-object family: the latter's
        # original answer is an object, so its inverse is the direction row.
        direction_baseline = (
            "native"
            if family == "object_centric_relative_position"
            else "inverse"
        )
        for left, right in (
            ("inverse", "native"),
            ("direction_natural_language", direction_baseline),
            ("object_direction_exhaustive", direction_baseline),
        ):
            differences = [
                _score(variants[left]) - _score(variants[right])
                for variants in family_sources.values()
                if left in variants and right in variants
            ]
            paired[family][f"{left}_minus_{right}"] = {
                "paired_count": len(differences),
                "mean_gain": sum(differences) / len(differences) if differences else None,
            }
            paired_outcomes[family][f"{left}_vs_{right}"] = _paired_outcomes(
                family_sources, left, right
            )

    by_relation: dict[str, dict[tuple[str, str], dict[str, dict[str, Any]]]] = defaultdict(dict)
    for source, variants in by_source.items():
        relation = next((str(row.get("relation", "")) for row in variants.values()), "")
        if relation:
            by_relation[relation][source] = variants
    format_paired_outcomes = {
        "overall": _format_paired_outcomes(by_source),
        **{
            relation: _format_paired_outcomes(by_relation[relation])
            for relation in RELATIONS
            if relation in by_relation
        },
    }

    output: dict[str, Any] = {
        "records": len(records),
        "by_source_task_family": by_family,
        "by_variant": by_variant,
        "paired_gains": paired,
        "paired_outcomes": paired_outcomes,
        "format_paired_outcomes_by_relation": format_paired_outcomes,
    }
    if unknown_families:
        output["unknown_source_task_families"] = sorted(unknown_families)
    if unknown_variants:
        output["unknown_variants"] = sorted(unknown_variants)
    return output


def _print_report(report: dict[str, Any]) -> None:
    print(f"Records: {report['records']}")
    print("\nAccuracy by base question family and row variant")
    print(f"{'source_task_family':48} {'variant':32} {'correct/total':>13} {'accuracy':>10}")
    print("-" * 108)
    for family in SOURCE_FAMILIES:
        for variant in VARIANTS:
            row = report["by_source_task_family"][family][variant]
            accuracy = "N/A" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
            print(f"{family:48} {variant:32} {row['correct']:>6}/{row['count']:<6} {accuracy:>10}")

    print("\nOverall by variant")
    for variant in VARIANTS:
        row = report["by_variant"][variant]
        accuracy = "N/A" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
        print(f"  {variant:32} {row['correct']}/{row['count']} ({accuracy})")

    print("\nPaired mean gains")
    for family, gains in report["paired_gains"].items():
        print(f"  {family}")
        for name, value in gains.items():
            gain = "N/A" if value["mean_gain"] is None else f"{value['mean_gain']:+.4f}"
            print(f"    {name}: {gain} ({value['paired_count']} pairs)")

    print("\nPaired correctness transitions (left variant versus right variant)")
    for family, comparisons in report["paired_outcomes"].items():
        print(f"  {family}")
        for name, counts in comparisons.items():
            print(
                f"    {name}: improved={counts['improved']}, worsened={counts['worsened']}, "
                f"unchanged-correct={counts['unchanged_correct']}, "
                f"unchanged-incorrect={counts['unchanged_incorrect']} "
                f"({counts['paired_count']} pairs)"
            )

    print("\nDirection-to-object paired outcomes by relation")
    for relation, counts in report["format_paired_outcomes_by_relation"].items():
        print(
            f"  {relation}: improved={counts['improved']}, worsened={counts['worsened']}, "
            f"unchanged-correct={counts['unchanged_correct']}, "
            f"unchanged-incorrect={counts['unchanged_incorrect']} "
            f"({counts['paired_count']} pairs)"
        )


def _save_paired_outcomes_plot(report: dict[str, Any], path: Path) -> None:
    """Save a COMFORT-style stacked plot for object versus direction answers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; install it to use --plot.") from exc

    paired_by_relation = report["format_paired_outcomes_by_relation"]
    labels = ["overall", *(relation for relation in RELATIONS if relation in paired_by_relation)]
    labels = [label for label in labels if paired_by_relation[label]["paired_count"]]

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
        values = [paired_by_relation[label][key] for label in labels]
        axis.bar(labels, values, bottom=bottom, label=display[key], color=colors[key])
        bottom = [current + value for current, value in zip(bottom, values)]
    axis.set_ylabel("Number of matched source relations")
    axis.set_title("Kubric MOVi-A direction/object paired outcomes")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to the JSON submission list")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON")
    parser.add_argument("--plot", type=Path, metavar="PATH", help="Write a paired-outcomes stacked bar chart")
    args = parser.parse_args()

    report = analyze(_load_records(args.submission))
    if args.plot:
        _save_paired_outcomes_plot(report, args.plot)
        print(f"Saved paired-outcomes plot to {args.plot}", file=sys.stderr)
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
