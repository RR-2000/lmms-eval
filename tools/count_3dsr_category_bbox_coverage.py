#!/usr/bin/env python3
"""Count 3DSR samples by <qtype>_<relation> and bbox coverage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("/home/ramanathan/data/3DSR/dataset.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the number of 3DSR samples in each <qtype>_<relation> "
            "category, along with bbox coverage for each bbox_items slot."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the 3DSR dataset JSONL file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of a text table.",
    )
    return parser.parse_args()


def has_at_least_one_bbox(item_bboxes: Any) -> bool:
    """Return True if the bbox payload contains at least one bbox anywhere."""
    if isinstance(item_bboxes, list):
        return len(item_bboxes) > 0

    if isinstance(item_bboxes, dict):
        return any(has_at_least_one_bbox(value) for value in item_bboxes.values())

    return False


def summarize_dataset(dataset_path: Path) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    with dataset_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            qtype = record.get("qtype")
            relation = record.get("relation")
            if not isinstance(qtype, str) or not isinstance(relation, str):
                raise ValueError(
                    f"Line {line_number} is missing string qtype/relation fields."
                )

            category = f"{qtype}_{relation}"
            bbox_items = record.get("bbox_items") or []
            bboxes_by_item = record.get("bboxes_by_item") or {}

            category_summary = summary.setdefault(
                category,
                {
                    "total_samples": 0,
                    "samples_with_all_items_having_bbox": 0,
                    "samples_missing_bbox_for_any_item": 0,
                    "item_slot_coverage": defaultdict(int),
                },
            )
            category_summary["total_samples"] += 1

            all_items_have_bbox = True
            for slot_index, item_name in enumerate(bbox_items, start=1):
                has_bbox = has_at_least_one_bbox(bboxes_by_item.get(item_name))
                if has_bbox:
                    category_summary["item_slot_coverage"][f"item_{slot_index}"] += 1
                else:
                    all_items_have_bbox = False

            if all_items_have_bbox:
                category_summary["samples_with_all_items_having_bbox"] += 1
            else:
                category_summary["samples_missing_bbox_for_any_item"] += 1

    for category_summary in summary.values():
        category_summary["item_slot_coverage"] = dict(
            sorted(category_summary["item_slot_coverage"].items())
        )

    return dict(sorted(summary.items()))


def print_text_summary(summary: dict[str, dict[str, Any]]) -> None:
    header = [
        "category",
        "total",
        "all_items_have_bbox",
        "missing_any_item_bbox",
        "item_slot_coverage",
    ]
    rows = [header]

    for category, data in summary.items():
        slot_summary = ", ".join(
            f"{slot}={count}" for slot, count in data["item_slot_coverage"].items()
        )
        rows.append(
            [
                category,
                str(data["total_samples"]),
                str(data["samples_with_all_items_having_bbox"]),
                str(data["samples_missing_bbox_for_any_item"]),
                slot_summary or "-",
            ]
        )

    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.resolve()

    if not dataset_path.is_file():
        print(f"Dataset not found: {dataset_path}")
        return 1

    summary = summarize_dataset(dataset_path)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_text_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
