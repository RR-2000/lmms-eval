#!/usr/bin/env python3
"""Summarize Kubric MOVi-A JSONL annotations by task family, difficulty, and option count."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/home/ramanathan/data/movi_a_3dsr/movi_a_validation.jsonl")
OPTION_KEYS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a Kubric MOVi-A JSONL file and summarize counts by task_family, "
            "difficulty, and number of answer options."
        )
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
        dest="json_output",
        type=Path,
        default=None,
        help="Optional path to write the summary as JSON.",
    )
    return parser.parse_args()


def count_options(record: dict[str, Any]) -> int:
    count = 0
    for key in OPTION_KEYS:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() == "nan":
            continue
        count += 1
    return count


def load_summary(path: Path) -> dict[str, Any]:
    task_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    option_counts: Counter[int] = Counter()
    task_difficulty_flat_counts: Counter[str] = Counter()
    task_difficulty_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_option_counts: dict[str, Counter[int]] = defaultdict(Counter)

    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")

            task_family = str(record.get("task_family", "missing"))
            difficulty = str(record.get("difficulty", "missing"))
            num_options = count_options(record)

            total += 1
            task_counts[task_family] += 1
            difficulty_counts[difficulty] += 1
            option_counts[num_options] += 1
            task_difficulty_flat_counts[f"{task_family}_{difficulty}"] += 1
            task_difficulty_counts[task_family][difficulty] += 1
            task_option_counts[task_family][num_options] += 1

    return {
        "input_path": str(path),
        "total_records": total,
        "task_family_counts": dict(sorted(task_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "option_count_distribution": {
            str(key): value for key, value in sorted(option_counts.items())
        },
        "task_difficulty_counts": dict(sorted(task_difficulty_flat_counts.items())),
        "task_family_by_difficulty": {
            task: dict(sorted(counter.items()))
            for task, counter in sorted(task_difficulty_counts.items())
        },
        "task_family_by_option_count": {
            task: {str(key): value for key, value in sorted(counter.items())}
            for task, counter in sorted(task_option_counts.items())
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Input: {summary['input_path']}")
    print(f"Total records: {summary['total_records']}")
    print()

    print("Task families:")
    for key, value in summary["task_family_counts"].items():
        print(f"  {key}: {value}")
    print()

    print("Difficulties:")
    for key, value in summary["difficulty_counts"].items():
        print(f"  {key}: {value}")
    print()

    print("Option counts:")
    for key, value in summary["option_count_distribution"].items():
        print(f"  {key} options: {value}")
    print()

    print("Task difficulty counts:")
    for key, value in summary["task_difficulty_counts"].items():
        print(f"  {key}: {value}")
    print()

    print("Task family x difficulty:")
    for task, counts in summary["task_family_by_difficulty"].items():
        pieces = ", ".join(f"{difficulty}={count}" for difficulty, count in counts.items())
        print(f"  {task}: {pieces}")
    print()

    print("Task family x option count:")
    for task, counts in summary["task_family_by_option_count"].items():
        pieces = ", ".join(f"{num_options} options={count}" for num_options, count in counts.items())
        print(f"  {task}: {pieces}")


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()

    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        return 1

    summary = load_summary(input_path)
    print_summary(summary)

    if args.json_output is not None:
        output_path = args.json_output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"Wrote JSON summary to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
