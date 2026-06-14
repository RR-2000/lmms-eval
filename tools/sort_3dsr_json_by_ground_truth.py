#!/usr/bin/env python3
"""Copy 3DSR artifact JSON files into correct/incorrect folders by answer match."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_ROOT = Path(
    "/home/ramanathan/VLM/lmms-eval/experiment_artifacts_3dsr_split/qwen3_4B_GT_1/by_task_type"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy JSON files into correct/incorrect folders by comparing "
            "ground_truth with the last '.'-delimited segment of answer."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root by_task_type directory to scan.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "Write into root/correct and root/incorrect. By default, files are "
            "copied into task-local correct/incorrect folders."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    return parser.parse_args()


def iter_json_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file() and path.parent.name not in {"correct", "incorrect"}
    )


def extract_answer_tail(answer: object) -> str | None:
    if not isinstance(answer, str):
        return None
    return answer.rsplit(".", 1)[-1].strip()


def normalized_ground_truth(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip()


def destination_dir(root: Path, json_path: Path, is_correct: bool, flat_output: bool) -> Path:
    label = "correct" if is_correct else "incorrect"
    if flat_output:
        return root / label
    return json_path.parent / label


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        print(f"Root directory not found: {root}")
        return 1

    json_files = iter_json_files(root)
    if not json_files:
        print(f"No JSON files found under: {root}")
        return 1

    correct_count = 0
    incorrect_count = 0
    skipped_count = 0

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text())
        except Exception as exc:
            print(f"Skipping unreadable JSON {json_path}: {exc}")
            skipped_count += 1
            continue

        answer_tail = extract_answer_tail(data.get("answer"))
        ground_truth = normalized_ground_truth(data.get("ground_truth"))
        if answer_tail is None or ground_truth is None:
            print(f"Skipping {json_path}: missing string answer or ground_truth")
            skipped_count += 1
            continue

        is_correct = ground_truth.lower() in answer_tail.lower()
        dest_dir = destination_dir(root, json_path, is_correct, args.flat_output)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / json_path.name

        if dest_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {dest_path}")
            skipped_count += 1
            continue

        shutil.copy2(json_path, dest_path)
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

    print(f"Scanned: {len(json_files)}")
    print(f"Copied to correct: {correct_count}")
    print(f"Copied to incorrect: {incorrect_count}")
    print(f"Skipped: {skipped_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
