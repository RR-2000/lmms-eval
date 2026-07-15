#!/usr/bin/env python3
"""Count Kubric MOVi-A validation samples by task family."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path("/home/ramanathan/data/movi_a_3dsr_new/movi_a_validation.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count MOVi-A JSONL samples in each task family."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the MOVi-A JSONL file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print counts as JSON instead of a text summary.",
    )
    return parser.parse_args()


def count_tasks(input_path: Path) -> tuple[int, Counter[str]]:
    total = 0
    task_counts: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")

            task_family = record.get("task_family")
            if not isinstance(task_family, str) or not task_family:
                raise ValueError(
                    f"Line {line_number} is missing a non-empty task_family field"
                )

            total += 1
            task_counts[task_family] += 1

    return total, task_counts


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()

    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        return 1

    try:
        total, task_counts = count_tasks(input_path)
    except ValueError as exc:
        print(exc)
        return 1

    counts = dict(sorted(task_counts.items()))
    if args.json:
        print(json.dumps({"total_samples": total, "task_family_counts": counts}, indent=2))
    else:
        print(f"Input: {input_path}")
        print(f"Total samples: {total}")
        print("\nSamples by task:")
        for task_family, count in counts.items():
            print(f"  {task_family}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
