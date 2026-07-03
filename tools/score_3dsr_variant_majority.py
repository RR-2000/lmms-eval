#!/usr/bin/env python3
"""Score 3DSR variant predictions by majority vote across qid versions."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "/home/ramanathan/VLM/lmms-eval/outputs/3dsrbench_variants/submissions/3dsrbench_predictions_qwen3_vl_experiments.json"
)
QID_VERSION_RE = re.compile(r"^(?P<base>.+)__v\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group 3DSR variant predictions by qid prefix, score each group by "
            "majority correctness, and summarize category/main_category totals."
        )
    )
    parser.add_argument(
        "--input_json",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Prediction JSON file to score.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON.",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} is not a JSON object")
        records.append(item)
    return records


def base_qid(qid: str) -> str:
    match = QID_VERSION_RE.match(qid)
    if match is None:
        return qid
    return match.group("base")


def is_record_correct(record: dict[str, Any]) -> bool:
    score = record.get("score")
    if isinstance(score, (int, float)):
        return float(score) >= 1.0

    pred_answer = record.get("pred_answer")
    gt_answer = record.get("gt_answer")
    if not isinstance(pred_answer, str) or not isinstance(gt_answer, str):
        raise ValueError(
            f"Record {record.get('qid', '<unknown>')} is missing usable score "
            "and pred_answer/gt_answer fields"
        )
    return pred_answer.strip() == gt_answer.strip()


def compute_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        qid = record.get("qid")
        if not isinstance(qid, str):
            raise ValueError("Encountered record without string qid")
        grouped[base_qid(qid)].append(record)

    category_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    main_category_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    category_collections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    main_category_collections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_correct = 0
    total_groups = 0
    group_summaries: list[dict[str, Any]] = []

    for group_qid in sorted(grouped):
        group_records = grouped[group_qid]
        first = group_records[0]
        category = first.get("category")
        main_category = first.get("main_category")
        if not isinstance(category, str) or not isinstance(main_category, str):
            raise ValueError(f"Group {group_qid} is missing category/main_category")

        for record in group_records[1:]:
            if record.get("category") != category:
                raise ValueError(f"Group {group_qid} has inconsistent category values")
            if record.get("main_category") != main_category:
                raise ValueError(f"Group {group_qid} has inconsistent main_category values")

        num_correct = sum(is_record_correct(record) for record in group_records)
        num_variants = len(group_records)
        majority_correct = num_correct > (num_variants / 2.0)

        category_totals[category]["total"] += 1
        main_category_totals[main_category]["total"] += 1
        total_groups += 1

        if majority_correct:
            category_totals[category]["correct"] += 1
            main_category_totals[main_category]["correct"] += 1
            total_correct += 1

        group_summary = {
            "qid": group_qid,
            "category": category,
            "main_category": main_category,
            "num_correct_variants": num_correct,
            "num_variants": num_variants,
            "majority_correct": majority_correct,
        }
        group_summaries.append(group_summary)
        category_collections[category].append(group_summary)
        main_category_collections[main_category].append(group_summary)

    return {
        "total": build_score_entry(total_correct, total_groups),
        "by_category": build_score_map(category_totals),
        "by_main_category": build_score_map(main_category_totals),
        "collections_by_category": build_collection_map(category_collections),
        "collections_by_main_category": build_collection_map(main_category_collections),
        "group_summaries": group_summaries,
    }


def build_collection_map(
    raw_collections: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    return {
        key: sorted(values, key=lambda item: item["qid"])
        for key, values in sorted(raw_collections.items())
    }


def build_score_entry(correct: int, total: int) -> dict[str, Any]:
    accuracy = 0.0 if total == 0 else correct / total
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "percentage": accuracy * 100.0,
    }


def build_score_map(raw_totals: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    return {
        key: build_score_entry(counts["correct"], counts["total"])
        for key, counts in sorted(raw_totals.items())
    }


def print_summary(summary: dict[str, Any]) -> None:
    total = summary["total"]
    print(
        "Total: "
        f"{total['correct']}/{total['total']} "
        f"({total['percentage']:.2f}%)"
    )

    print("\nBy main_category:")
    for name, score in summary["by_main_category"].items():
        print(
            f"  {name}: {score['correct']}/{score['total']} "
            f"({score['percentage']:.2f}%)"
        )

    print("\nBy category:")
    for name, score in summary["by_category"].items():
        print(
            f"  {name}: {score['correct']}/{score['total']} "
            f"({score['percentage']:.2f}%)"
        )


def main() -> int:
    args = parse_args()
    input_path = args.input_json.resolve()

    if not input_path.is_file():
        print(f"Input JSON not found: {input_path}")
        return 1

    summary = compute_summary(load_predictions(input_path))
    print_summary(summary)

    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nWrote summary JSON: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
