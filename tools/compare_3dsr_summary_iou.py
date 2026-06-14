#!/usr/bin/env python3
"""Compare two 3DSR summary outputs and compute IoU by task/category."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two 3DSR summary roots or summary JSON files and compute "
            "IoU for each task/category using doc_id sets."
        )
    )
    parser.add_argument(
        "--first",
        type=Path,
        required=True,
        help="First summary root folder or single summary JSON file.",
    )
    parser.add_argument(
        "--second",
        type=Path,
        required=True,
        help="Second summary root folder or single summary JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <first>/summary_iou_vs_<second>.json for roots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def is_summary_file(path: Path) -> bool:
    return path.is_file() and (
        path.name.endswith("_doc_id_summary.json") or path.name == "doc_id_summary.json"
    )


def task_name_from_summary(path: Path) -> str:
    if path.name == "doc_id_summary.json":
        return path.parent.name
    return path.name.removesuffix("_doc_id_summary.json")


def collect_summary_files(path: Path) -> dict[str, Path]:
    resolved = path.resolve()
    if resolved.is_file():
        if not is_summary_file(resolved):
            raise ValueError(f"Not a summary JSON file: {resolved}")
        return {task_name_from_summary(resolved): resolved}

    if not resolved.is_dir():
        raise ValueError(f"Path not found: {resolved}")

    summary_files = sorted(
        file_path
        for file_path in resolved.glob("*_doc_id_summary.json")
        if file_path.is_file()
    )
    return {task_name_from_summary(file_path): file_path for file_path in summary_files}


def load_summary(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def doc_id_set(summary: dict[str, dict[str, object]], category: str) -> set[str]:
    records = summary.get(category, {})
    if not isinstance(records, dict):
        return set()
    return set(records.keys())


def compute_iou(first_ids: set[str], second_ids: set[str]) -> dict[str, object]:
    intersection = sorted(first_ids & second_ids, key=_sort_key)
    union = sorted(first_ids | second_ids, key=_sort_key)
    first_only = sorted(first_ids - second_ids, key=_sort_key)
    second_only = sorted(second_ids - first_ids, key=_sort_key)
    union_count = len(union)
    iou = (len(intersection) / union_count) if union_count else 1.0

    return {
        "iou": iou,
        "intersection_count": len(intersection),
        "union_count": union_count,
        "first_count": len(first_ids),
        "second_count": len(second_ids),
        "intersection_doc_ids": intersection,
        "first_only_doc_ids": first_only,
        "second_only_doc_ids": second_only,
    }


def _sort_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def default_output_path(first: Path, second: Path) -> Path:
    if first.resolve().is_dir():
        second_stem = second.resolve().name
        return first.resolve().parent / f"summary_iou_{first.resolve().name}vs_{second_stem}.json"
    return first.resolve().with_name(f"{first.resolve().stem}_iou.json")


def main() -> int:
    args = parse_args()

    try:
        first_files = collect_summary_files(args.first)
        second_files = collect_summary_files(args.second)
    except ValueError as exc:
        print(str(exc))
        return 1

    shared_tasks = sorted(set(first_files) & set(second_files))
    if not shared_tasks:
        print("No shared task summary files found between the two inputs.")
        return 1

    output_path = args.output.resolve() if args.output else default_output_path(args.first, args.second)
    if output_path.exists() and not args.overwrite:
        print(f"Output file already exists: {output_path}")
        print("Use --overwrite to replace it.")
        return 1

    results: dict[str, object] = {
        "first": str(args.first.resolve()),
        "second": str(args.second.resolve()),
        "shared_tasks": shared_tasks,
        "tasks": {},
    }

    overall_first_correct: set[str] = set()
    overall_second_correct: set[str] = set()
    overall_first_incorrect: set[str] = set()
    overall_second_incorrect: set[str] = set()

    for task in shared_tasks:
        first_summary = load_summary(first_files[task])
        second_summary = load_summary(second_files[task])

        first_correct = doc_id_set(first_summary, "correct")
        second_correct = doc_id_set(second_summary, "correct")
        first_incorrect = doc_id_set(first_summary, "incorrect")
        second_incorrect = doc_id_set(second_summary, "incorrect")

        overall_first_correct |= {f"{task}:{doc_id}" for doc_id in first_correct}
        overall_second_correct |= {f"{task}:{doc_id}" for doc_id in second_correct}
        overall_first_incorrect |= {f"{task}:{doc_id}" for doc_id in first_incorrect}
        overall_second_incorrect |= {f"{task}:{doc_id}" for doc_id in second_incorrect}

        results["tasks"][task] = {
            "correct": compute_iou(first_correct, second_correct),
            "incorrect": compute_iou(first_incorrect, second_incorrect),
        }

    results["overall"] = {
        "correct": compute_iou(overall_first_correct, overall_second_correct),
        "incorrect": compute_iou(overall_first_incorrect, overall_second_incorrect),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"Wrote IoU report: {output_path}")
    print(f"Shared tasks: {len(shared_tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
