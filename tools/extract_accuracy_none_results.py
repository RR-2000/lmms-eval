#!/usr/bin/env python3
"""Print JSON result entries whose keys end with 'accuracy,none' in source order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_JSON = Path(
    "/home/ramanathan/VLM/lmms-eval/outputs/3dsrbench_4B_GT_0/"
    "rayruiyang__VST-7B-RL/20260614_230202_results.json"
)
TARGET_SUFFIX = "accuracy,none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and print only result entries whose key ends with "
            f"'{TARGET_SUFFIX}', preserving JSON key order."
        )
    )
    parser.add_argument(
        "--json_path",
        nargs="?",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to the results JSON file.",
    )
    parser.add_argument(
        "--print_key",
        action="store_true",
        help="Print the key along with the value.",
    )
    return parser.parse_args()


def iter_accuracy_entries(data: object) -> list[tuple[str, object]]:
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON value must be an object.")

    results = data.get("results")
    if not isinstance(results, dict):
        raise ValueError("Missing or invalid 'results' object.")

    entries: list[tuple[str, object]] = []
    for task_results in results.values():
        if not isinstance(task_results, dict):
            continue
        for key, value in task_results.items():
            if key.endswith(TARGET_SUFFIX):
                entries.append((key, value))
    return entries


def main() -> int:
    args = parse_args()

    try:
        data = json.loads(args.json_path.read_text())
        entries = iter_accuracy_entries(data)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    for key, value in entries:
        if value == []:
            continue
        if args.print_key:
            print(f"{key}: {value}")
        else:
            print(value)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
