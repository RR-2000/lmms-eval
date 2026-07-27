#!/usr/bin/env python3
"""Report per-source-family metrics from a Kubric direction/object submission.

The extended task produces four records for each source question:
native, inverse, direction_natural_language, and object_direction_exhaustive.
This tool reads the submission records written by the task's submission metric;
it does not rerun model inference or reconstruct the dataset.

Example:
    python tools/analyze_kubric_direction_object_submission.py \
        outputs/kubric_movi_a_direction_object_extended_0/submissions/\
kubric_movi_a_direction_object_extended_qwen3_vl_experiments.json
"""

from __future__ import annotations

import argparse
import json
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


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
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
        source_qid = str(row.get("source_qid", ""))
        by_source[source_qid][variant] = row

    by_family = {
        family: {variant: _summary(grouped[(family, variant)]) for variant in VARIANTS}
        for family in SOURCE_FAMILIES
    }
    by_variant = {variant: _summary(grouped_for_variant)
                  for variant in VARIANTS
                  for grouped_for_variant in [[row for row in records if row.get("variant") == variant]]}

    paired = {}
    for family in SOURCE_FAMILIES:
        family_sources = {
            source: variants
            for source, variants in by_source.items()
            if any(row.get("source_task_family") == family for row in variants.values())
        }
        paired[family] = {}
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

    output: dict[str, Any] = {
        "records": len(records),
        "by_source_task_family": by_family,
        "by_variant": by_variant,
        "paired_gains": paired,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to the JSON submission list")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON")
    args = parser.parse_args()

    report = analyze(_load_records(args.submission))
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
